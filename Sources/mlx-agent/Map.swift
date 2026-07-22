// Map.swift - long-lived, file-spool-driven "map" server.
//
//   mlx-agent map --model <dir> --spool <dir> [--extra-eos-token <t>] [gen flags]
//
// Applies ONE prompt independently to each chunk of a long input - a map over document
// chunks. Each chunk is processed with a fresh cache (stateless, no cross-chunk context), so
// the operation is element-wise and order-independent, exactly like `map(f, chunks)`.
// Translation is one instantiation; proofreading, rewriting, redaction, normalization,
// per-chunk classification/extraction, and the map phase of map-reduce summarization are
// others - the task lives entirely in the job's message template, not in this code.
//
// The mode BYPASSES ChatSession (whose message content must be a plain String), so a job can
// carry STRUCTURED content (e.g. a translation template's {type, source_lang_code,
// target_lang_code, text}) and render it against the model's OWN unmodified chat template.
//
// The job supplies a template with a `{{chunk}}` placeholder - either chat `messages`
// rendered through the model's own chat template, or a raw completion `prompt` for models
// that ship no chat template and are prompted as plain completions (e.g. MT models like
// MiLMMT-46 or Seed-X). Per chunk the server substitutes the chunk text for the
// placeholder, tokenizes, and generates. Two output modes:
//   - stitch (default): reassemble the per-chunk outputs into one document, preserving the
//     verbatim inter-chunk separators -> result.txt (progressive).
//   - collect: emit a per-chunk record -> results.jsonl (one JSON object per line). Use for
//     text->data maps (classify/extract) or when the caller does its own reduce/reassembly.
//
// Spool protocol (writes are atomic - temp file + rename - except the append-only jsonl):
//   in:  job.json  {"epoch": N, "text": "...", "budget_tokens": 1200,
//                   "output": "stitch"|"collect",
//                   "messages": [ <chat messages, with "{{chunk}}" somewhere> ]
//                   | "prompt": "<completion prompt with {{chunk}}>"
//                     [, "add_special_tokens": true|false]}   // prompt only; default true
//        stop       empty flag file; requests cancellation of the current job
//   out: status.json {"state": loading|ready|mapping|done|cancelled|error,
//                     "epoch": N, "chunk": k, "total": N, "message": "...",
//                     // progress metrics (present on ready/mapping/done as available):
//                     "tok_per_sec": F, "tokens_done": D, "tokens_total": T, "eta_sec": S}
//        result.txt   (stitch) accumulated stitched output so far
//        results.jsonl (collect) one {"index","source","output","sep"} object per line
//
// Exits when the spool directory disappears (owner quit) or stdin hits EOF (parent death).

import Foundation
import MLX
import MLXLMCommon
import Tokenizers
import Chunking

/// Live progress for one map job, shared between the generation loop (which writes token counts)
/// and a background reporter task (which reads them and writes tok/s + ETA into status.json).
/// Lock-guarded so the two can touch it without a data race.
private final class JobMeter: @unchecked Sendable {
    let jobStart: Date
    let epoch: Int
    let totalChunks: Int
    let totalInputTokens: Int
    let primedRate: Double

    private let lock = NSLock()
    private var chunkIndex = 0     // 0-based chunk currently generating (== chunks completed)
    private var completedIn = 0    // input tokens of finished chunks
    private var completedOut = 0   // exact output tokens of finished chunks
    private var liveChunkOut = 0   // approximate output tokens in the in-flight chunk
    private var outEstimate: Int   // estimated total output tokens (refined once chunks finish)

    init(jobStart: Date, epoch: Int, totalChunks: Int, totalInputTokens: Int, primedRate: Double) {
        self.jobStart = jobStart
        self.epoch = epoch
        self.totalChunks = totalChunks
        self.totalInputTokens = totalInputTokens
        self.primedRate = primedRate
        self.outEstimate = totalInputTokens   // 1:1 seed; refined by the observed out/in ratio
    }

    func beginChunk(_ i: Int) { lock.lock(); defer { lock.unlock() }; chunkIndex = i; liveChunkOut = 0 }
    func liveToken() { lock.lock(); defer { lock.unlock() }; liveChunkOut += 1 }

    func endChunk(inputTokens: Int, outputTokens: Int) {
        lock.lock(); defer { lock.unlock() }
        completedIn += inputTokens
        completedOut += outputTokens
        liveChunkOut = 0
        if completedIn > 0 {
            let ratio = Double(completedOut) / Double(completedIn)
            let remainingIn = max(0, totalInputTokens - completedIn)
            outEstimate = completedOut + Int((ratio * Double(remainingIn)).rounded())
        }
    }

    /// (chunks completed so far, output tokens done, estimated total output tokens).
    func snapshot() -> (chunkIndex: Int, tokensDone: Int, totalOut: Int) {
        lock.lock(); defer { lock.unlock() }
        let done = completedOut + liveChunkOut
        return (chunkIndex, done, max(outEstimate, done))
    }
}

final class MapServer: @unchecked Sendable {
    private enum Output { case stitch, collect }

    /// The job's per-chunk template: chat `messages` rendered through the model's own chat
    /// template, or a raw completion `prompt` for models that ship none. `specialTokens`
    /// mirrors HF's `add_special_tokens` (whether the tokenizer wraps the prompt in BOS/EOS);
    /// chat templates emit their own special tokens, so it applies to `prompt` only.
    private enum Template: Sendable {
        case messages(Data)  // JSON of the `messages` template (carries {{chunk}})
        case prompt(String, specialTokens: Bool)
    }

    /// A map request read from the spool.
    private struct Job {
        let epoch: Int
        let text: String
        let budget: Int
        let output: Output
        let template: Template
    }

    /// One chunk's generation result plus the exact decode stats for tok/s.
    private struct ChunkResult {
        let text: String
        let genTokens: Int
        let genTime: Double
    }

    // Placeholder the job's message template must contain; replaced with each chunk's text.
    private static let placeholder = "{{chunk}}"

    private let modelDir: String
    private let spoolDir: URL
    private let gen: GenConfig
    private let extraEOSTokens: Set<String>

    // Decode speed (tokens/sec) measured by a warmup generation right after load; the fallback
    // rate for an upfront ETA before a job has produced any tokens. 0 if priming failed.
    private var primedRate: Double = 0

    // Idle-unload (shared policy - see IdleUnload.swift): after `idleClock.seconds` without a
    // job, the model weights are released and MLX's buffer cache returned to the OS; the next
    // job transparently reloads. Unlike ACPServer no timer, lock, or mid-turn guard is needed
    // here: the run loop is single-threaded, so the check runs only between polls, where no
    // generation can be in flight. 0 disables.
    private var idleClock: IdleClock
    private var container: ModelContainer?

    private let jobURL: URL
    private let statusURL: URL
    private let resultTxtURL: URL
    private let resultsJsonlURL: URL
    private let stopURL: URL

    // How often the generation loop checks the stop flag (in generation events).
    private let stopCheckStride = 16

    init(
        modelDir: String, spoolDir: String, gen: GenConfig, extraEOSTokens: Set<String>,
        idleUnloadSeconds: TimeInterval = IdleUnload.defaultSeconds
    ) {
        self.modelDir = modelDir
        self.spoolDir = URL(fileURLWithPath: spoolDir, isDirectory: true)
        self.gen = gen
        self.extraEOSTokens = extraEOSTokens
        self.idleClock = IdleClock(seconds: idleUnloadSeconds)
        self.jobURL = self.spoolDir.appending(component: "job.json")
        self.statusURL = self.spoolDir.appending(component: "status.json")
        self.resultTxtURL = self.spoolDir.appending(component: "result.txt")
        self.resultsJsonlURL = self.spoolDir.appending(component: "results.jsonl")
        self.stopURL = self.spoolDir.appending(component: "stop")
    }

    // MARK: Run loop

    func run() async {
        // Parent-death detection via stdin EOF, but ONLY when stdin is a pipe - i.e. the parent
        // holds the write end and its close is a real death signal. A daemon spawned by a shell
        // handler gets /dev/null (or a closed) stdin, which reads as instant EOF; installing the
        // handler there would kill us on launch. That case relies on the spool-directory-gone
        // check in the poll loop instead.
        var stdinStat = stat()
        if fstat(FileHandle.standardInput.fileDescriptor, &stdinStat) == 0,
            (stdinStat.st_mode & S_IFMT) == S_IFIFO
        {
            FileHandle.standardInput.readabilityHandler = { handle in
                if handle.availableData.isEmpty { exit(0) }
            }
        }

        let label = (modelDir as NSString).lastPathComponent
        writeStatus(state: "loading", message: "Loading model...")
        do {
            container = try await loadModel(modelDir, extraEOSTokens: extraEOSTokens)
        } catch {
            writeStatus(state: "error", message: "Model load failed: \(error.localizedDescription)")
            log("model load failed: \(error.localizedDescription)")
            return
        }
        // Warm the Metal kernels and measure decode speed, so the first real job has an upfront
        // ETA and the UI can show a comparable tok/s figure for the loaded model.
        primedRate = await primeAndMeasure(container: container!)
        writeStatus(
            state: "ready", tokPerSec: primedRate > 0 ? primedRate : nil, message: "Ready - \(label)")
        log(String(format: "ready on %@ (%.1f tok/s); watching %@", label, primedRate, spoolDir.path))
        idleClock.noteActivity()

        var lastEpoch: Int? = nil
        while spoolExists() {
            switch readJob() {
            case .none:
                break  // no job file, or mid-write / unparseable JSON: poll again silently
            case .invalid(let epoch, let reason):
                // A structurally bad job (present, has an epoch, but bad text/messages) gets a
                // clear error - not a silent drop that reads as a hung server. Dedup on epoch so
                // we report it once, not every poll.
                if epoch != lastEpoch {
                    lastEpoch = epoch
                    writeStatus(state: "error", epoch: epoch, message: reason)
                    log("job \(epoch): rejected - \(reason)")
                    // Activity, even though rejected: a producer iterating on a bad template is
                    // alive and about to submit a valid job - don't unload out from under it.
                    idleClock.noteActivity()
                }
            case .job(let job):
                if job.epoch != lastEpoch {
                    lastEpoch = job.epoch
                    await runJob(job)
                }
            }
            idleUnloadIfDue()
            try? await Task.sleep(nanoseconds: 150_000_000)  // 150 ms poll
        }
        log("spool directory gone; exiting")
    }

    /// Run one job, transparently reloading the model first if it was idle-unloaded. A reload
    /// failure is reported against the job's epoch and leaves the model unloaded, so the NEXT
    /// job (epoch contract: resubmit under a new epoch) retries the load.
    private func runJob(_ job: Job) async {
        let active: ModelContainer
        if let loaded = container {
            active = loaded
        } else {
            writeStatus(state: "loading", epoch: job.epoch, message: "Loading model...")
            log("idle-reload: loading \((modelDir as NSString).lastPathComponent)")
            do {
                active = try await loadModel(modelDir, extraEOSTokens: extraEOSTokens)
            } catch {
                writeStatus(
                    state: "error", epoch: job.epoch,
                    message: "Model load failed: \(error.localizedDescription)")
                log("idle-reload failed: \(error.localizedDescription)")
                idleClock.noteActivity()
                return
            }
            container = active
            // The measured decode rate survives the unload; re-prime only if it never took.
            if primedRate == 0 { primedRate = await primeAndMeasure(container: active) }
        }
        await process(job, container: active)
        idleClock.noteActivity()
    }

    /// Release the model weights after the idle window passes with no job. Runs between polls
    /// only, so it can never fire mid-generation. The spool protocol is untouched: status keeps
    /// its last state (still truthful - the server remains ready to accept jobs), and the next
    /// job passes through the same "loading" state the poller already handles at startup.
    /// Mechanics (snapshot ordering, clearCache, log format) are shared - see IdleUnload.swift.
    private func idleUnloadIfDue() {
        guard container != nil, idleClock.dueNow() else { return }
        IdleUnload.releaseModel(afterIdle: idleClock.seconds, log: { self.log($0) }) {
            self.container = nil
            return (self.modelDir as NSString).lastPathComponent
        }
    }

    // MARK: Job processing

    private func process(_ job: Job, container: ModelContainer) async {
        // Clear any stale stop flag so it cannot cancel this fresh job.
        try? FileManager.default.removeItem(at: stopURL)

        // The template must carry the placeholder in a substitutable value, else every chunk
        // would get the same text-less prompt - a producer mistake worth a clear error rather
        // than silent garbage.
        let placeholderMissing: String?
        switch job.template {
        case .messages(let data):
            placeholderMissing = Self.templateHasPlaceholder(data)
                ? nil : "messages template has no \(Self.placeholder) placeholder value"
        case .prompt(let prompt, _):
            placeholderMissing = prompt.contains(Self.placeholder)
                ? nil : "prompt template has no \(Self.placeholder) placeholder"
        }
        if let placeholderMissing {
            writeStatus(state: "error", epoch: job.epoch, message: placeholderMissing)
            return
        }

        // Reset the output surface: truncate the file this job writes and remove the other
        // mode's file, so a stale result from a prior job can never be read as this job's.
        switch job.output {
        case .stitch:
            writeText("", to: resultTxtURL)
            try? FileManager.default.removeItem(at: resultsJsonlURL)
        case .collect:
            writeText("", to: resultsJsonlURL)
            try? FileManager.default.removeItem(at: resultTxtURL)
        }

        if job.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            if job.output == .stitch { writeText(job.text, to: resultTxtURL) }
            writeStatus(state: "done", epoch: job.epoch, chunk: 0, total: 0, message: "Done")
            return
        }

        // Chunk with the real tokenizer, capturing each chunk's INPUT token count (for the ETA's
        // total-work estimate). One actor hop for the whole split + counts.
        let (chunks, inTokens): ([Chunk], [Int]) = await container.perform { context in
            let cs = Chunker.make(job.text, budget: job.budget) {
                context.tokenizer.encode(text: $0).count
            }
            let counts = cs.map { context.tokenizer.encode(text: $0.text).count }
            return (cs, counts)
        }

        let total = chunks.count
        let totalIn = inTokens.reduce(0, +)
        let params = gen.apply(to: GenerateParameters(maxTokens: 2048, temperature: 0))
        var accumulated = ""
        var sumGenTokens = 0
        var sumGenTime = 0.0

        let meter = JobMeter(
            jobStart: Date(), epoch: job.epoch, totalChunks: total,
            totalInputTokens: totalIn, primedRate: primedRate)

        writeStatus(
            state: "mapping", epoch: job.epoch, chunk: 0, total: total,
            tokPerSec: primedRate > 0 ? primedRate : nil, tokensDone: 0, tokensTotal: totalIn,
            etaSec: primedRate > 0 ? Int((Double(totalIn) / primedRate).rounded()) : nil,
            message: "Mapping 0 of \(total)")
        log("job \(job.epoch): \(job.output == .stitch ? "stitch" : "collect"), \(total) chunk(s), \(totalIn) input tokens")

        // Background reporter: turns the live token count into tok/s + ETA in status.json ~2x/s,
        // so the UI counts down smoothly instead of only jumping at chunk boundaries. It is the
        // SOLE writer of the running "mapping" status; process() writes only the terminal state
        // (done/cancelled/error) AFTER stopping it, so a stale tick can never land after "done".
        //
        // Rate = an EMA of the RECENT-window token rate, seeded with the primed decode speed. The
        // window (tokens since the last tick) reflects the true decode speed rather than the
        // since-start rate, which the one-time prompt prefill drags far down on the first tick;
        // seeding with the primed speed keeps that first tick from spiking the ETA.
        let reporter = Task { [self] in
            var emaRate = meter.primedRate
            var prevTokens = 0
            var prevTime = meter.jobStart
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 500_000_000)
                if Task.isCancelled { break }
                let now = Date()
                let s = meter.snapshot()
                let dt = now.timeIntervalSince(prevTime)
                let dTok = s.tokensDone - prevTokens
                if dt > 0.05 && dTok > 0 {
                    let inst = Double(dTok) / dt
                    emaRate = emaRate > 0 ? 0.6 * emaRate + 0.4 * inst : inst
                }
                prevTokens = s.tokensDone
                prevTime = now
                let rate = emaRate > 0 ? emaRate : meter.primedRate
                let eta = rate > 0 ? Int((Double(max(0, s.totalOut - s.tokensDone)) / rate).rounded()) : nil
                self.writeStatus(
                    state: "mapping", epoch: meter.epoch, chunk: s.chunkIndex, total: meter.totalChunks,
                    tokPerSec: rate > 0 ? rate : nil, tokensDone: s.tokensDone, tokensTotal: s.totalOut,
                    etaSec: eta, message: "Mapping \(s.chunkIndex) of \(meter.totalChunks)")
            }
        }
        func stopReporter() async { reporter.cancel(); _ = await reporter.value }

        for (i, chunk) in chunks.enumerated() {
            if stopRequested() {
                await stopReporter()
                writeStatus(
                    state: "cancelled", epoch: job.epoch, chunk: i, total: total, message: "Cancelled")
                log("job \(job.epoch): cancelled before chunk \(i + 1)")
                return
            }
            meter.beginChunk(i)
            let result: ChunkResult
            do {
                result = try await generate(
                    chunk: chunk.text, template: job.template,
                    params: params, container: container, meter: meter)
            } catch {
                await stopReporter()
                writeStatus(
                    state: "error", epoch: job.epoch, chunk: i, total: total,
                    message: "Generation failed: \(error.localizedDescription)")
                log("job \(job.epoch): chunk \(i + 1) error: \(error.localizedDescription)")
                return
            }
            meter.endChunk(inputTokens: inTokens[i], outputTokens: result.genTokens)
            sumGenTokens += result.genTokens
            sumGenTime += result.genTime
            switch job.output {
            case .stitch:
                accumulated += result.text + chunk.sep
                writeText(accumulated, to: resultTxtURL)
            case .collect:
                appendRecord(
                    ["index": i, "source": chunk.text, "output": result.text, "sep": chunk.sep])
            }
            // A stop that arrived mid-chunk (generate() cancelled/drained) lands here.
            if stopRequested() {
                await stopReporter()
                writeStatus(
                    state: "cancelled", epoch: job.epoch, chunk: i + 1, total: total,
                    message: "Cancelled")
                log("job \(job.epoch): cancelled after chunk \(i + 1)")
                return
            }
        }
        await stopReporter()
        let decodeRate = sumGenTime > 0 ? Double(sumGenTokens) / sumGenTime : primedRate
        writeStatus(
            state: "done", epoch: job.epoch, chunk: total, total: total,
            tokPerSec: decodeRate > 0 ? decodeRate : nil, tokensDone: sumGenTokens,
            tokensTotal: sumGenTokens, etaSec: 0, message: "Done")
        log(String(
            format: "job %d: done - %d tokens in %.1fs (%.1f tok/s)",
            job.epoch, sumGenTokens, sumGenTime, decodeRate))
    }

    /// Generate one chunk's output: render the job's template with the chunk text substituted
    /// for `{{chunk}}` - through the model's own chat template for `messages`, or by direct
    /// tokenization for a raw completion `prompt` - and generate.
    ///
    /// The whole generation runs INSIDE `container.perform`, so the model's serial-access mutex
    /// is held for the entire decode (not just prefill). A stop request `cancel()`s the decode
    /// task but keeps draining the stream to its natural end, so the background token loop is
    /// fully finished before this closure returns and releases the mutex - the next job's
    /// generation can never overlap a still-running decode.
    private func generate(
        chunk: String, template: Template, params: GenerateParameters,
        container: ModelContainer, meter: JobMeter
    ) async throws -> ChunkResult {
        let stopURL = self.stopURL
        let stride = self.stopCheckStride
        return try await container.perform { context -> ChunkResult in
            let tokens: [Int]
            switch template {
            case .messages(let messagesData):
                let messages = try Self.renderMessages(messagesData, chunk: chunk)
                tokens = try context.tokenizer.applyChatTemplate(messages: messages)
            case .prompt(let prompt, let specialTokens):
                tokens = context.tokenizer.encode(
                    text: prompt.replacingOccurrences(of: Self.placeholder, with: chunk),
                    addSpecialTokens: specialTokens)
            }
            let iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(tokens)), model: context.model,
                cache: nil, parameters: params)
            let (stream, task) = generateTask(
                promptTokenCount: tokens.count,
                modelConfiguration: context.configuration,
                tokenizer: context.tokenizer,
                iterator: iterator)
            var out = ""
            var pieces = 0     // fallback token count if no .info arrives (e.g. cancellation)
            var genTokens = 0
            var genTime = 0.0
            var seen = 0
            for await event in stream {
                switch event {
                case .chunk(let piece):
                    out += piece
                    pieces += 1
                    meter.liveToken()   // live count for the ETA reporter
                case .info(let info):
                    genTokens = info.generationTokenCount
                    genTime = info.generateTime
                default:
                    break
                }
                seen += 1
                // Cancel (do NOT break): keep consuming so the decode task observes cancellation
                // and finishes before we leave this closure and drop the model mutex.
                if seen % stride == 0
                    && FileManager.default.fileExists(atPath: stopURL.path)
                {
                    task.cancel()
                }
            }
            return ChunkResult(
                text: out.trimmingCharacters(in: .whitespacesAndNewlines),
                genTokens: genTokens > 0 ? genTokens : pieces, genTime: genTime)
        }
    }

    /// Warm the Metal kernels and measure the model's decode speed (tokens/sec) with a short
    /// throwaway generation. Runs right after load so the first real job has an upfront ETA and
    /// the UI can show a comparable tok/s figure. Uses a raw-token prompt (not the chat template)
    /// because a generic map model may reject a plain-string chat message, and a small fixed
    /// maxTokens so priming stays quick regardless of the job's --max-new-tokens. Returns 0 on
    /// any failure - the ETA then just falls back to the live rate once a job produces tokens.
    private func primeAndMeasure(container: ModelContainer) async -> Double {
        let params = GenerateParameters(maxTokens: 64, temperature: 0)
        return await container.perform { context -> Double in
            let prompt = context.tokenizer.encode(text: "The")
            guard !prompt.isEmpty else { return 0 }
            do {
                let iterator = try TokenIterator(
                    input: LMInput(tokens: MLXArray(prompt)), model: context.model,
                    cache: nil, parameters: params)
                let (stream, _) = generateTask(
                    promptTokenCount: prompt.count,
                    modelConfiguration: context.configuration,
                    tokenizer: context.tokenizer,
                    iterator: iterator)
                var rate = 0.0
                for await event in stream {
                    if case .info(let info) = event { rate = info.tokensPerSecond }
                }
                return rate
            } catch {
                return 0
            }
        }
    }

    // MARK: Template rendering

    /// Parse the job's `messages` template and substitute the chunk text for `{{chunk}}` in
    /// every string value, producing chat messages the tokenizer can render. Substitution is
    /// at the parsed-object level (not raw JSON text), so a chunk containing quotes/newlines
    /// needs no escaping.
    private static func renderMessages(_ data: Data, chunk: String) throws -> [[String: any Sendable]]
    {
        guard let raw = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw NSError(
                domain: "mlx-agent", code: 4,
                userInfo: [NSLocalizedDescriptionKey: "messages is not an array of objects"])
        }
        return raw.map { renderObject($0, chunk: chunk) }
    }

    private static func renderObject(_ dict: [String: Any], chunk: String) -> [String: any Sendable]
    {
        var out: [String: any Sendable] = [:]
        for (key, value) in dict {
            // A JSON null renders to nil and is OMITTED, so the template sees the field as
            // undefined (the correct shape for an absent optional) rather than a literal.
            if let rendered = renderValue(value, chunk: chunk) { out[key] = rendered }
        }
        return out
    }

    /// Substitute `{{chunk}}` into a JSON value while PRESERVING its type. The tokenizer
    /// ultimately converts each value via Jinja's `Value(any:)`, which recognizes Swift-native
    /// `String`/`Int`/`Double`/`Bool`/arrays/dicts (and throws on `NSNumber`/`NSNull`), so a
    /// bool must stay a bool, a number a number, and null must drop out - stringifying them
    /// (e.g. bool -> "0"/"1") would invert truthiness and break template branches.
    private static func renderValue(_ value: Any, chunk: String) -> (any Sendable)? {
        switch value {
        case let s as String:
            return s.replacingOccurrences(of: placeholder, with: chunk)
        case let a as [Any]:
            // Keep nulls positionally (as Optional.none): Jinja's Value(any:) maps a `[Any?]`
            // element-wise including nil -> .null, so array length/order is preserved. (A null
            // DICT value is instead dropped in renderObject - a key omission is an equivalent
            // "undefined" for template purposes, and dropping a key can't shift anything.)
            return a.map { renderValue($0, chunk: chunk) }
        case let d as [String: Any]:
            return renderObject(d, chunk: chunk)
        case is NSNull:
            return nil
        case let num as NSNumber:
            // JSONSerialization boxes both numbers and bools as NSNumber; the CFBoolean type id
            // is the reliable way to tell a JSON bool from a 0/1 number.
            if CFGetTypeID(num) == CFBooleanGetTypeID() { return num.boolValue }
            let d = num.doubleValue
            return (d.rounded() == d && abs(d) < 9.2e18) ? num.intValue : d
        default:
            return "\(value)"  // unreachable for well-formed JSON; a harmless last resort
        }
    }

    /// Whether the template has the `{{chunk}}` placeholder in a substitutable VALUE (not a
    /// key - `renderObject` only rewrites values). A raw-text search would wrongly accept a
    /// placeholder that appears only in a key.
    private static func templateHasPlaceholder(_ data: Data) -> Bool {
        guard let raw = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            return false
        }
        return raw.contains { valueHasPlaceholder($0) }
    }

    private static func valueHasPlaceholder(_ value: Any) -> Bool {
        switch value {
        case let s as String: return s.contains(placeholder)
        case let a as [Any]: return a.contains { valueHasPlaceholder($0) }
        case let d as [String: Any]: return d.values.contains { valueHasPlaceholder($0) }
        default: return false
        }
    }

    // MARK: Spool I/O

    private func spoolExists() -> Bool {
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(atPath: spoolDir.path, isDirectory: &isDir)
            && isDir.boolValue
    }

    private func stopRequested() -> Bool {
        FileManager.default.fileExists(atPath: stopURL.path)
    }

    /// Result of reading the spool's job.json: absent/unreadable (poll silently), present with
    /// an epoch but structurally invalid (report an error once), or a valid job. Both `invalid`
    /// and `job` advance the run loop's single `lastEpoch` cursor, so - per the "a new job is a
    /// higher epoch" contract - a rejected job must be resubmitted under a NEW epoch to be
    /// reconsidered; rewriting it in place under the same epoch stays rejected.
    private enum JobResult {
        case none
        case invalid(epoch: Int, reason: String)
        case job(Job)
    }

    private func readJob() -> JobResult {
        // No file, or a partial/unparseable write (atomic rename makes torn reads rare): treat
        // as "nothing to do yet" and poll again - not an error.
        guard let data = try? Data(contentsOf: jobURL),
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let epoch = (obj["epoch"] as? NSNumber)?.intValue
        else { return .none }
        // From here the job has an identity (epoch); a structural problem is a producer mistake
        // we surface as an error rather than dropping silently. The source text is either inline
        // ("text") or in a spool file ("text_file") - the latter lets a shell producer avoid
        // JSON-encoding arbitrary content and handles large inputs cleanly. text_file is confined
        // to the spool dir (basename only) so it cannot escape via a path.
        let text: String
        if let inline = obj["text"] as? String {
            text = inline
        } else if let name = obj["text_file"] as? String, !name.isEmpty,
            let data = try? Data(
                contentsOf: spoolDir.appending(component: (name as NSString).lastPathComponent)),
            let fromFile = String(data: data, encoding: .utf8)
        {
            text = fromFile
        } else {
            return .invalid(epoch: epoch, reason: "job has no \"text\" or readable \"text_file\"")
        }
        // The per-chunk template is EITHER chat `messages` OR a raw completion `prompt` -
        // exactly one. Both present is ambiguous (which would win?), so it is rejected rather
        // than silently preferring one. An explicit JSON null (NSNull) counts as absent, so
        // {"prompt": "...", "messages": null} means what the producer meant.
        func present(_ key: String) -> Any? {
            let v = obj[key]
            return v is NSNull ? nil : v
        }
        let template: Template
        switch (present("messages"), present("prompt")) {
        case (.some, .some):
            return .invalid(epoch: epoch, reason: "job has both \"messages\" and \"prompt\"")
        case (.some(let raw), nil):
            guard let messages = raw as? [[String: Any]], !messages.isEmpty,
                let messagesData = try? JSONSerialization.data(withJSONObject: messages)
            else {
                return .invalid(epoch: epoch, reason: "job has no valid \"messages\" array")
            }
            template = .messages(messagesData)
        case (nil, .some(let raw)):
            guard let prompt = raw as? String, !prompt.isEmpty else {
                return .invalid(epoch: epoch, reason: "job \"prompt\" is not a non-empty string")
            }
            // Mirrors HF's add_special_tokens: whether the tokenizer wraps the prompt in
            // BOS/EOS. Default true (the tokenizer's standard behavior); a model card that
            // says add_special_tokens=False (e.g. MiLMMT-46) sets false.
            let special = (obj["add_special_tokens"] as? NSNumber)?.boolValue ?? true
            template = .prompt(prompt, specialTokens: special)
        case (nil, nil):
            return .invalid(epoch: epoch, reason: "job has no \"messages\" array or \"prompt\" string")
        }
        let budget = (obj["budget_tokens"] as? NSNumber)?.intValue ?? 1200
        let output: Output = (obj["output"] as? String == "collect") ? .collect : .stitch
        return .job(
            Job(
                epoch: epoch, text: text, budget: max(64, budget), output: output,
                template: template))
    }

    private func writeText(_ text: String, to url: URL) {
        try? Data(text.utf8).write(to: url, options: .atomic)
    }

    /// Append one JSON object as a line to results.jsonl (collect mode). Single writer, so a
    /// per-line append is safe; a reader consumes complete lines as they land.
    private func appendRecord(_ obj: [String: Any]) {
        guard var data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
        else { return }
        data.append(0x0A)
        if let handle = try? FileHandle(forWritingTo: resultsJsonlURL) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
        } else {
            try? data.write(to: resultsJsonlURL, options: .atomic)
        }
    }

    private func writeStatus(
        state: String, epoch: Int? = nil, chunk: Int? = nil, total: Int? = nil,
        tokPerSec: Double? = nil, tokensDone: Int? = nil, tokensTotal: Int? = nil,
        etaSec: Int? = nil, message: String
    ) {
        var obj: [String: Any] = ["state": state, "message": message]
        if let epoch { obj["epoch"] = epoch }
        if let chunk { obj["chunk"] = chunk }
        if let total { obj["total"] = total }
        if let tokPerSec { obj["tok_per_sec"] = (tokPerSec * 10).rounded() / 10 }
        if let tokensDone { obj["tokens_done"] = tokensDone }
        if let tokensTotal { obj["tokens_total"] = tokensTotal }
        if let etaSec { obj["eta_sec"] = etaSec }
        if let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]) {
            try? data.write(to: statusURL, options: .atomic)
        }
    }

    private func log(_ s: String) {
        FileHandle.standardError.write(Data("[mlx-agent map] \(s)\n".utf8))
    }
}
