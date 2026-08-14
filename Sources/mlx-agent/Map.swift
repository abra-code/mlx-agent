// Map.swift - long-lived, file-spool-driven "map" server.
//
//   mlx-agent map --backend mlx    --model <dir>    --spool <dir> [--extra-eos-token <t>] [gen flags]
//   mlx-agent map --backend openai --base-url <url> --spool <dir> [gen flags]
//
// Two engines serve the SAME spool protocol, indistinguishably to producers except in speed
// (see MapEngine): in-process mlx-swift-lm on a local safetensors --model (default, RAM gate +
// idle-unload), or a separately-launched llama-server addressed over its OpenAI-compatible
// --base-url (--backend openai; the weights live in that process, so no RAM gate / idle-unload).
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
    fileprivate enum Template: Sendable {
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
    fileprivate struct ChunkResult {
        let text: String
        let genTokens: Int
        let genTime: Double
    }

    // Placeholder the job's message template must contain; replaced with each chunk's text.
    fileprivate static let placeholder = "{{chunk}}"

    // The engine that tokenizes + generates (MLX in-process, or a remote llama-server). It is the
    // ONLY thing that differs between the two backends; everything below - the spool protocol, the
    // epoch contract, chunking, the reporter, stitch/collect writers, stop-file cancellation - is
    // engine-agnostic and runs identically for both.
    private let engine: MapEngine
    private let spoolDir: URL
    private let gen: GenConfig

    // Decode speed (tokens/sec) measured by a warmup generation right after load; the fallback
    // rate for an upfront ETA before a job has produced any tokens. 0 if priming failed or the
    // engine does not warm up (openai: llama-server measures its own throughput).
    private var primedRate: Double = 0

    // Idle-unload (shared policy - see IdleUnload.swift): after `idleClock.seconds` without a
    // job, the MLX engine releases its weights and returns MLX's buffer cache to the OS; the
    // next job transparently reloads. Unlike ACPServer no timer, lock, or mid-turn guard is
    // needed here: the run loop is single-threaded, so the check runs only between polls, where
    // no generation can be in flight. 0 disables. The openai engine holds no weights, so it opts
    // out (supportsIdleUnload == false) and llama-server does its own idle sleep.
    private var idleClock: IdleClock

    private let jobURL: URL
    private let statusURL: URL
    private let resultTxtURL: URL
    private let resultsJsonlURL: URL
    private let stopURL: URL

    // How often the generation loop checks the stop flag (in generation events).
    private let stopCheckStride = 16

    /// Designated init on an already-resolved engine. Kept private so the engine types stay
    /// file-local; callers construct through the EngineSpec convenience init below.
    private init(
        engine: MapEngine, spoolDir: String, gen: GenConfig,
        idleUnloadSeconds: TimeInterval
    ) {
        self.engine = engine
        self.spoolDir = URL(fileURLWithPath: spoolDir, isDirectory: true)
        self.gen = gen
        self.idleClock = IdleClock(seconds: idleUnloadSeconds)
        self.jobURL = self.spoolDir.appending(component: "job.json")
        self.statusURL = self.spoolDir.appending(component: "status.json")
        self.resultTxtURL = self.spoolDir.appending(component: "result.txt")
        self.resultsJsonlURL = self.spoolDir.appending(component: "results.jsonl")
        self.stopURL = self.spoolDir.appending(component: "stop")
    }

    /// Build the map server for a resolved `--backend` engine. The MLX path loads a local model
    /// container into THIS process (RAM gate, idle-unload, warmup); the openai path serves the
    /// spool from a separately-launched llama-server over its OpenAI-compatible endpoint (no
    /// weights here, no RAM gate, no idle-unload). `extraEOSTokens` applies to the MLX path only -
    /// llama-server owns its own stop set.
    convenience init(
        engine spec: EngineSpec, spoolDir: String, gen: GenConfig,
        extraEOSTokens: Set<String>, idleUnloadSeconds: TimeInterval = IdleUnload.defaultSeconds
    ) {
        let engine: MapEngine
        switch spec {
        case .mlx(let modelDir):
            engine = MLXMapEngine(modelDir: modelDir, extraEOSTokens: extraEOSTokens)
        case .openai(let baseURL):
            engine = OpenAIMapEngine(baseURL: baseURL)
        case .foundation:
            // Refused rather than silently substituted. map exists to translate or classify whole
            // DOCUMENTS chunk by chunk, and the on-device system model's context is 4096 tokens
            // total - it would force chunks small enough that stitching quality collapses, on a
            // model with no control over the target language. `map` deliberately takes mlx or
            // openai only; main.swift refuses this earlier, and this is the backstop for a caller
            // that builds a MapServer directly.
            FileHandle.standardError.write(
                Data(
                    "map mode does not support --backend foundation: the on-device model's 4096-token context is too small for document chunking\n"
                        .utf8))
            exit(2)
        }
        self.init(engine: engine, spoolDir: spoolDir, gen: gen, idleUnloadSeconds: idleUnloadSeconds)
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

        writeStatus(state: "loading", message: "Loading model...")
        do {
            try await engine.load()
        } catch {
            writeStatus(state: "error", message: "Model load failed: \(error.localizedDescription)")
            log("model load failed: \(error.localizedDescription)")
            return
        }
        let label = engine.label
        // Warm the Metal kernels and measure decode speed, so the first real job has an upfront
        // ETA and the UI can show a comparable tok/s figure for the loaded model. The openai
        // engine returns 0 here (llama-server reports its own throughput per generation).
        primedRate = await engine.primeAndMeasure()
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
    /// job (epoch contract: resubmit under a new epoch) retries the load. The openai engine is
    /// always loaded (isLoaded == true), so this reload block is skipped there.
    private func runJob(_ job: Job) async {
        if !engine.isLoaded {
            writeStatus(state: "loading", epoch: job.epoch, message: "Loading model...")
            log("idle-reload: loading \(engine.label)")
            do {
                try await engine.reloadIfNeeded()
            } catch {
                writeStatus(
                    state: "error", epoch: job.epoch,
                    message: "Model load failed: \(error.localizedDescription)")
                log("idle-reload failed: \(error.localizedDescription)")
                idleClock.noteActivity()
                return
            }
            // The measured decode rate survives the unload; re-prime only if it never took.
            if primedRate == 0 { primedRate = await engine.primeAndMeasure() }
        }
        await process(job)
        idleClock.noteActivity()
    }

    /// Release the model weights after the idle window passes with no job. Runs between polls
    /// only, so it can never fire mid-generation. The spool protocol is untouched: status keeps
    /// its last state (still truthful - the server remains ready to accept jobs), and the next
    /// job passes through the same "loading" state the poller already handles at startup.
    /// Mechanics (snapshot ordering, clearCache, log format) are shared - see IdleUnload.swift.
    /// A no-op for an engine that holds no weights (openai).
    private func idleUnloadIfDue() {
        guard engine.supportsIdleUnload, engine.isLoaded, idleClock.dueNow() else { return }
        engine.idleUnload(afterIdle: idleClock.seconds, log: { self.log($0) })
    }

    // MARK: Job processing

    private func process(_ job: Job) async {
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

        // Chunk with the engine's real tokenizer, capturing each chunk's INPUT token count (for
        // the ETA's total-work estimate). Delegated to the engine because tokenization differs by
        // backend (MLX in-process vs llama-server's /tokenize). An unreachable tokenizer fails the
        // job as state "error" - the same failure surface as a model load failure.
        let chunks: [Chunk]
        let inTokens: [Int]
        do {
            (chunks, inTokens) = try await engine.chunkAndCount(text: job.text, budget: job.budget)
        } catch {
            writeStatus(
                state: "error", epoch: job.epoch,
                message: "Chunking failed: \(error.localizedDescription)")
            log("job \(job.epoch): chunking failed: \(error.localizedDescription)")
            return
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
                // `state` is the wire contract a reader switches on and keeps its spelling;
                // `message` is display prose and follows house style. Not a typo - do not
                // "fix" either one to match the other.
                writeStatus(
                    state: "cancelled", epoch: job.epoch, chunk: i, total: total, message: "Canceled")
                log("job \(job.epoch): canceled before chunk \(i + 1)")
                return
            }
            meter.beginChunk(i)
            let result: ChunkResult
            do {
                result = try await engine.generate(
                    chunk: chunk.text, template: job.template, params: params,
                    meter: meter, stopURL: stopURL, stopCheckStride: stopCheckStride)
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
            // A stop that arrived mid-chunk (generate() canceled/drained) lands here.
            if stopRequested() {
                await stopReporter()
                writeStatus(
                    state: "cancelled", epoch: job.epoch, chunk: i + 1, total: total,
                    message: "Canceled")
                log("job \(job.epoch): canceled after chunk \(i + 1)")
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

    // MARK: Template rendering

    /// Parse the job's `messages` template and substitute the chunk text for `{{chunk}}` in
    /// every string value, producing chat messages the tokenizer can render. Substitution is
    /// at the parsed-object level (not raw JSON text), so a chunk containing quotes/newlines
    /// needs no escaping.
    fileprivate static func renderMessages(_ data: Data, chunk: String) throws -> [[String: any Sendable]]
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

// MARK: - Map engines

/// The tokenization + generation seam MapServer runs against, so ONE spool protocol (job.json
/// in; status.json/result.txt/results.jsonl out; epoch + stop-file semantics) is served by either
/// backend, indistinguishably to producers except in speed:
///   MLXMapEngine    - in-process mlx-swift-lm on a local safetensors model (RAM gate, idle-unload,
///                     warmup decode-rate measurement).
///   OpenAIMapEngine - a separately-launched llama-server addressed over its OpenAI-compatible
///                     endpoint (the weights live in that process: no RAM gate, no idle-unload, no
///                     warmup - llama-server reports its own throughput per generation).
private protocol MapEngine: AnyObject, Sendable {
    /// Human label for status/log lines (the model's name).
    var label: String { get }
    /// Whether this engine holds weights that idle-unload can release (MLX yes, openai no).
    var supportsIdleUnload: Bool { get }
    /// Whether the engine is ready to generate. Always true for openai; false for MLX after an
    /// idle-unload, until `reloadIfNeeded()` reloads.
    var isLoaded: Bool { get }
    /// Prepare the engine. Throws on failure (model load failure / server unreachable) so the run
    /// loop reports state "error" - the SAME failure surface for both engines.
    func load() async throws
    /// Warm up and measure decode speed (tokens/sec); 0 if unavailable (openai always 0).
    func primeAndMeasure() async -> Double
    /// Reload if `isLoaded` is false (MLX idle-reload); a no-op otherwise.
    func reloadIfNeeded() async throws
    /// Release weights if the idle window elapsed (MLX); a no-op for openai.
    func idleUnload(afterIdle seconds: TimeInterval, log: (String) -> Void)
    /// Chunk the source with the engine's tokenizer and return each chunk's input token count (for
    /// the ETA). Throws if the tokenizer/endpoint is unreachable - fails the job as state "error",
    /// the same as a load failure.
    func chunkAndCount(text: String, budget: Int) async throws -> ([Chunk], [Int])
    /// Generate one chunk's output. Must poll `stopURL` (every `stopCheckStride` events) and cancel
    /// PROMPTLY when it appears, returning the partial output rather than throwing (the caller then
    /// writes state "cancelled"). A genuine generation failure throws (caller writes "error").
    func generate(
        chunk: String, template: MapServer.Template, params: GenerateParameters,
        meter: JobMeter, stopURL: URL, stopCheckStride: Int
    ) async throws -> MapServer.ChunkResult
}

// MARK: MLX engine

/// In-process mlx-swift-lm engine: the original map behavior, now behind `MapEngine`. Holds the
/// `ModelContainer` and all the MLX-specific mechanics (RAM-gated load, warmup, idle-unload, the
/// per-chunk `container.perform` decode) that were MapServer's before the openai backend arrived.
private final class MLXMapEngine: MapEngine, @unchecked Sendable {
    private let modelDir: String
    private let extraEOSTokens: Set<String>
    private var container: ModelContainer?

    init(modelDir: String, extraEOSTokens: Set<String>) {
        self.modelDir = modelDir
        self.extraEOSTokens = extraEOSTokens
    }

    var label: String { (modelDir as NSString).lastPathComponent }
    var supportsIdleUnload: Bool { true }
    var isLoaded: Bool { container != nil }

    func load() async throws {
        container = try await loadModel(modelDir, extraEOSTokens: extraEOSTokens)
    }

    func reloadIfNeeded() async throws {
        guard container == nil else { return }
        container = try await loadModel(modelDir, extraEOSTokens: extraEOSTokens)
    }

    func idleUnload(afterIdle seconds: TimeInterval, log: (String) -> Void) {
        IdleUnload.releaseModel(afterIdle: seconds, log: log) {
            self.container = nil
            return (self.modelDir as NSString).lastPathComponent
        }
    }

    func chunkAndCount(text: String, budget: Int) async throws -> ([Chunk], [Int]) {
        guard let container else { throw Self.notLoaded }
        // One actor hop for the whole split + counts, using the model's real tokenizer.
        return await container.perform { context in
            let cs = Chunker.make(text, budget: budget) {
                context.tokenizer.encode(text: $0).count
            }
            let counts = cs.map { context.tokenizer.encode(text: $0.text).count }
            return (cs, counts)
        }
    }

    /// Warm the Metal kernels and measure the model's decode speed (tokens/sec) with a short
    /// throwaway generation. Runs right after load so the first real job has an upfront ETA and the
    /// UI can show a comparable tok/s figure. Uses a raw-token prompt (not the chat template)
    /// because a generic map model may reject a plain-string chat message, and a small fixed
    /// maxTokens so priming stays quick regardless of the job's --max-new-tokens. Returns 0 on any
    /// failure - the ETA then just falls back to the live rate once a job produces tokens.
    func primeAndMeasure() async -> Double {
        guard let container else { return 0 }
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

    /// Generate one chunk's output: render the job's template with the chunk text substituted for
    /// `{{chunk}}` - through the model's own chat template for `messages`, or by direct tokenization
    /// for a raw completion `prompt` - and generate.
    ///
    /// The whole generation runs INSIDE `container.perform`, so the model's serial-access mutex is
    /// held for the entire decode (not just prefill). A stop request `cancel()`s the decode task but
    /// keeps draining the stream to its natural end, so the background token loop is fully finished
    /// before this closure returns and releases the mutex - the next job's generation can never
    /// overlap a still-running decode.
    func generate(
        chunk: String, template: MapServer.Template, params: GenerateParameters,
        meter: JobMeter, stopURL: URL, stopCheckStride stride: Int
    ) async throws -> MapServer.ChunkResult {
        guard let container else { throw Self.notLoaded }
        return try await container.perform { context -> MapServer.ChunkResult in
            let tokens: [Int]
            switch template {
            case .messages(let messagesData):
                let messages = try MapServer.renderMessages(messagesData, chunk: chunk)
                tokens = try context.tokenizer.applyChatTemplate(messages: messages)
            case .prompt(let prompt, let specialTokens):
                tokens = context.tokenizer.encode(
                    text: prompt.replacingOccurrences(of: MapServer.placeholder, with: chunk),
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
            return MapServer.ChunkResult(
                text: out.trimmingCharacters(in: .whitespacesAndNewlines),
                genTokens: genTokens > 0 ? genTokens : pieces, genTime: genTime)
        }
    }

    private static let notLoaded = NSError(
        domain: "mlx-agent", code: 6, userInfo: [NSLocalizedDescriptionKey: "model not loaded"])
}

// MARK: OpenAI (llama-server) engine

/// Serves the spool from a separately-launched llama-server over its OpenAI-compatible endpoint.
/// Templating is the SERVER's job (llama-server runs with --jinja and the GGUF's embedded chat
/// template), so this engine never renders a chat template client-side:
///   - `messages` mode: substitute {{chunk}} at the parsed-object level (MapServer.renderMessages),
///     then POST the array as-is to {base}/chat/completions (streaming SSE).
///   - `prompt`  mode: tokenize the rendered prompt via {root}/tokenize honoring add_special_tokens
///     EXACTLY, then POST the token-id array to {root}/completion (streaming SSE) - the way to feed
///     llama-server a completion whose special-token wrapping the caller controls.
/// ({root} is the base with its /v1 suffix trimmed - /tokenize and /completion are not under /v1.)
private final class OpenAIMapEngine: MapEngine, @unchecked Sendable {
    private let baseURL: URL
    private let chatURL: URL
    private let tokenizeURL: URL
    private let completionURL: URL
    private let urlSession: URLSession
    private var modelLabel = "llama-server"

    init(baseURL: URL) {
        self.baseURL = baseURL
        self.chatURL = baseURL.appendingPathComponent("chat/completions")
        let root = Self.serverRoot(baseURL)
        self.tokenizeURL = URL(string: root + "/tokenize") ?? baseURL
        self.completionURL = URL(string: root + "/completion") ?? baseURL
        let config = URLSessionConfiguration.ephemeral
        // Idle timeout between SSE frames, not a cap on the whole generation (mirrors OpenAIBackend).
        config.timeoutIntervalForRequest = 300
        config.waitsForConnectivity = false
        self.urlSession = URLSession(configuration: config)
    }

    var label: String { modelLabel }
    var supportsIdleUnload: Bool { false }
    var isLoaded: Bool { true }

    func load() async throws {
        // Same health gate the acp/oneshot openai path uses, but with a LONG patience window:
        // llama-server answers /health 503 for the whole time it is loading its gguf, which for a
        // 20 GB model is minutes, and the map broker is exactly the component whose job is to sit
        // in state "loading" until the server is ready (the applet spawns both and does not wait
        // itself). 120 attempts x ~2.5 s covers ~5 minutes; a genuinely absent server still fails
        // with the same clean startup error. acp keeps the short default - an interactive
        // session/new should not hang for minutes.
        try await OpenAIBackend.waitForHealth(baseURL: baseURL, attempts: 120)
        if let name = await fetchModelName() { modelLabel = name }
    }

    func primeAndMeasure() async -> Double { 0 }   // no local weights to warm; server reports tok/s
    func reloadIfNeeded() async throws {}          // nothing to reload here
    func idleUnload(afterIdle seconds: TimeInterval, log: (String) -> Void) {}  // server idles itself

    /// Best-effort model name for the ready status; the applet owns which gguf is loaded, so a
    /// failure just leaves the generic "llama-server" label. A short per-request timeout (not the
    /// session's 300s SSE-idle timeout) so a slow/hung /models can never delay "ready" - health is
    /// already confirmed by the time we get here.
    private func fetchModelName() async -> String? {
        var request = URLRequest(url: baseURL.appendingPathComponent("models"))
        request.timeoutInterval = 3
        guard let (data, response) = try? await urlSession.data(for: request),
            (response as? HTTPURLResponse)?.statusCode == 200,
            let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let arr = obj["data"] as? [[String: Any]], let id = arr.first?["id"] as? String,
            !id.isEmpty
        else { return nil }
        return (id as NSString).lastPathComponent
    }

    func chunkAndCount(text: String, budget: Int) async throws -> ([Chunk], [Int]) {
        // Chunker needs a SYNCHRONOUS token counter but /tokenize is an HTTP round trip, so a
        // memoizing counter coalesces the repeated counts Chunker makes of the same paragraph/atom
        // into ONE request each (Chunker counts each atom in `make` and again in `pack`, and this
        // method counts each finished chunk again - all cache hits after the first). It runs on a
        // background thread so the blocking requests never stall a cooperative-pool thread. A
        // tokenize failure is latched and rethrown after the split (the closure cannot throw),
        // failing the job as state "error".
        let counter = TokenCounter(url: tokenizeURL, session: urlSession)
        return try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                let chunks = Chunker.make(text, budget: budget) { counter.count($0) }
                if let error = counter.firstError { cont.resume(throwing: error); return }
                let counts = chunks.map { counter.count($0.text) }
                if let error = counter.firstError { cont.resume(throwing: error); return }
                cont.resume(returning: (chunks, counts))
            }
        }
    }

    func generate(
        chunk: String, template: MapServer.Template, params: GenerateParameters,
        meter: JobMeter, stopURL: URL, stopCheckStride stride: Int
    ) async throws -> MapServer.ChunkResult {
        // Build the request: messages -> OpenAI /chat/completions (server owns templating); prompt
        // -> /tokenize (honoring add_special_tokens) then a token-array /completion.
        let endpoint: URL
        let isChat: Bool
        var body: [String: Any]
        switch template {
        case .messages(let data):
            let messages = try MapServer.renderMessages(data, chunk: chunk)
            body = [
                "model": "auto",   // llama-server serves whatever it was launched with
                "messages": messages,
                "stream": true,
                "stream_options": ["include_usage": true],
            ]
            applySampler(&body, params: params, maxKey: "max_tokens")
            endpoint = chatURL
            isChat = true
        case .prompt(let prompt, let specialTokens):
            let rendered = prompt.replacingOccurrences(of: MapServer.placeholder, with: chunk)
            // Honor add_special_tokens EXACTLY: tokenize with add_special == specialTokens and feed
            // the resulting ids as the completion prompt, so BOS/EOS wrapping is the caller's call,
            // not the server's default. This pre-flight tokenize is not itself stop-cancellable, but
            // it is one small bounded request; a stop lands on the generation loop that follows.
            let ids = try await tokenizeIDs(rendered, addSpecial: specialTokens)
            body = ["prompt": ids, "stream": true]
            applySampler(&body, params: params, maxKey: "n_predict")
            endpoint = completionURL
            isChat = false
        }
        // renderMessages keeps a JSON null inside arrays as Optional.none for the MLX/Jinja
        // path; JSONSerialization RAISES (uncatchable) on such values rather than throwing, so
        // reject them here and fail the job as "error" like every other producer mistake.
        guard JSONSerialization.isValidJSONObject(body) else {
            throw Self.error("messages template contains values JSON cannot carry (e.g. null in an array)")
        }
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let handle = URLTaskHandle()
        var out = ""
        var pieces = 0        // fallback token count if the server reports no exact usage
        var reportedTokens = 0
        var timedGen = 0.0    // decode seconds the server reports (/completion timings)
        var firstToken: Date? = nil
        var stopped = false
        do {
            let (bytes, response) = try await urlSession.bytes(for: request)
            handle.adopt(bytes.task)
            guard let http = response as? HTTPURLResponse else {
                throw Self.error("no HTTP response from \(endpoint.absoluteString)")
            }
            guard http.statusCode == 200 else {
                var errBody = ""
                for try await line in bytes.lines { errBody = String(line.prefix(1000)); break }
                handle.cancel()
                throw Self.error("llama-server returned HTTP \(http.statusCode): \(errBody)")
            }
            var seen = 0
            for try await line in bytes.lines {
                seen += 1
                // Cancel PROMPTLY on stop, mid-generation: dropping the connection stops the server
                // generating (unlike MLX there is no shared mutex to drain), so break right after.
                if seen % stride == 0, FileManager.default.fileExists(atPath: stopURL.path) {
                    stopped = true
                    handle.cancel()
                    break
                }
                guard line.hasPrefix("data:") else { continue }
                let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                if payload == "[DONE]" { break }
                guard let data = payload.data(using: .utf8),
                    let frame = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
                else { continue }
                if isChat {
                    if let usage = frame["usage"] as? [String: Any],
                        let completion = (usage["completion_tokens"] as? NSNumber)?.intValue
                    {
                        reportedTokens = completion
                    }
                    guard let choice = (frame["choices"] as? [[String: Any]])?.first,
                        let delta = choice["delta"] as? [String: Any],
                        let piece = delta["content"] as? String, !piece.isEmpty
                    else { continue }
                    if firstToken == nil { firstToken = Date() }
                    out += piece
                    pieces += 1
                    meter.liveToken()
                } else {
                    if let piece = frame["content"] as? String, !piece.isEmpty {
                        if firstToken == nil { firstToken = Date() }
                        out += piece
                        pieces += 1
                        meter.liveToken()
                    }
                    // The final /completion frame carries the exact decode stats.
                    if (frame["stop"] as? NSNumber)?.boolValue == true {
                        if let predicted = (frame["tokens_predicted"] as? NSNumber)?.intValue {
                            reportedTokens = predicted
                        }
                        if let timings = frame["timings"] as? [String: Any],
                            let ms = (timings["predicted_ms"] as? NSNumber)?.doubleValue
                        {
                            timedGen = ms / 1000.0
                        }
                    }
                }
            }
        } catch {
            // The normal stop path returns via the `break` above (fallthrough past this do/catch),
            // NOT through here: the poll cancels the task and breaks synchronously, so that
            // iteration never throws. This branch is defensive for the race where handle.cancel()
            // makes an in-flight `bytes.lines` await throw URLError.cancelled before the poll runs -
            // treat that (and `stopped`) as the caller's "cancelled" (partial output), not an error.
            if stopped || (error as? URLError)?.code == .cancelled || error is CancellationError {
                return MapServer.ChunkResult(
                    text: out.trimmingCharacters(in: .whitespacesAndNewlines),
                    genTokens: reportedTokens > 0 ? reportedTokens : pieces, genTime: 0)
            }
            throw error
        }
        // Decode seconds: the server's own figure for /completion; else the span from the first
        // token to now (prefill excluded), the same approximation OpenAIBackend uses for chat.
        let genTime = timedGen > 0 ? timedGen : (firstToken.map { Date().timeIntervalSince($0) } ?? 0)
        return MapServer.ChunkResult(
            text: out.trimmingCharacters(in: .whitespacesAndNewlines),
            genTokens: reportedTokens > 0 ? reportedTokens : pieces, genTime: genTime)
    }

    /// Tokenize `content` via {root}/tokenize into bare token ids. `addSpecial` maps to the
    /// endpoint's `add_special` field (BOS/EOS wrapping) - this is how prompt mode honors the job's
    /// add_special_tokens exactly. Throws on any transport/HTTP failure (fails the job as "error").
    private func tokenizeIDs(_ content: String, addSpecial: Bool) async throws -> [Int] {
        var request = URLRequest(url: tokenizeURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(
            withJSONObject: ["content": content, "add_special": addSpecial])
        let (data, response) = try await urlSession.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw Self.error(
                "tokenize failed: HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)")
        }
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let tokens = obj["tokens"] as? [Any]
        else {
            throw Self.error("tokenize returned no \"tokens\" array")
        }
        // /tokenize returns bare ints when with_pieces is omitted.
        return tokens.compactMap { ($0 as? NSNumber)?.intValue }
    }

    /// Map the resolved GenerateParameters onto llama-server's sampler fields, shared by the chat
    /// and completion bodies (only the max-tokens key differs: /chat wants max_tokens, /completion
    /// wants n_predict). Mirrors OpenAIBackend.buildRequest.
    private func applySampler(_ body: inout [String: Any], params: GenerateParameters, maxKey: String)
    {
        body["temperature"] = Self.round6(params.temperature)
        if let maxTokens = params.maxTokens { body[maxKey] = maxTokens }
        if params.topP < 1 { body["top_p"] = Self.round6(params.topP) }
        if let penalty = params.repetitionPenalty { body["repeat_penalty"] = Self.round6(penalty) }
        if let seed = params.seed { body["seed"] = seed > UInt64(Int.max) ? Int.max : Int(seed) }
    }

    /// `<base minus /v1>` - /tokenize and /completion are NOT under the /v1 prefix, so the
    /// OpenAI base-url is trimmed back to the server root (mirrors OpenAIBackend.healthURL).
    private static func serverRoot(_ base: URL) -> String {
        var s = base.absoluteString
        while s.hasSuffix("/") { s.removeLast() }
        if s.hasSuffix("/v1") { s.removeLast(3) }
        while s.hasSuffix("/") { s.removeLast() }
        return s
    }

    private static func round6(_ v: Float) -> Double { (Double(v) * 1_000_000).rounded() / 1_000_000 }

    fileprivate static func error(_ message: String) -> NSError {
        NSError(domain: "mlx-agent", code: 7, userInfo: [NSLocalizedDescriptionKey: message])
    }
}

/// Synchronous, memoizing token counter over llama-server's /tokenize, for the Chunker's
/// `(String) -> Int` seam. `add_special` is FALSE: the budget measures source-content tokens, and
/// hardSplit counts many tiny fragments - a per-call BOS would inflate every one. Repeated counts of
/// the same string are cached, so it is one HTTP round trip per distinct atom. The first error is
/// latched (`firstError`); the caller rethrows it once the split completes.
private final class TokenCounter {
    private let url: URL
    private let session: URLSession
    private var cache: [String: Int] = [:]
    private(set) var firstError: Error?

    init(url: URL, session: URLSession) {
        self.url = url
        self.session = session
    }

    func count(_ s: String) -> Int {
        if let c = cache[s] { return c }
        if firstError != nil { return s.count }   // already failing: fall through cheaply
        do {
            let n = try tokenizeCount(s)
            cache[s] = n
            return n
        } catch {
            if firstError == nil { firstError = error }
            return s.count   // harmless fallback so the split still completes before we rethrow
        }
    }

    private func tokenizeCount(_ content: String) throws -> Int {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(
            withJSONObject: ["content": content, "add_special": false])
        let sem = DispatchSemaphore(value: 0)
        var outcome: Result<Int, Error> = .failure(
            OpenAIMapEngine.error("no response from \(url.absoluteString)"))
        let task = session.dataTask(with: request) { data, response, error in
            defer { sem.signal() }
            if let error {
                outcome = .failure(error)
                return
            }
            guard let http = response as? HTTPURLResponse, http.statusCode == 200, let data,
                let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                let tokens = obj["tokens"] as? [Any]
            else {
                outcome = .failure(
                    OpenAIMapEngine.error(
                        "tokenize failed: HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)"))
                return
            }
            outcome = .success(tokens.count)
        }
        task.resume()
        sem.wait()
        return try outcome.get()
    }
}
