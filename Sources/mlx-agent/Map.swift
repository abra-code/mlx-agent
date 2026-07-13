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
// The job supplies a chat-message template with a `{{chunk}}` placeholder; per chunk the
// server substitutes the chunk text for the placeholder, applies the model's template, and
// generates. Two output modes:
//   - stitch (default): reassemble the per-chunk outputs into one document, preserving the
//     verbatim inter-chunk separators -> result.txt (progressive).
//   - collect: emit a per-chunk record -> results.jsonl (one JSON object per line). Use for
//     text->data maps (classify/extract) or when the caller does its own reduce/reassembly.
//
// Spool protocol (writes are atomic - temp file + rename - except the append-only jsonl):
//   in:  job.json  {"epoch": N, "text": "...", "budget_tokens": 1200,
//                   "output": "stitch"|"collect",
//                   "messages": [ <chat messages, with "{{chunk}}" somewhere> ]}
//        stop       empty flag file; requests cancellation of the current job
//   out: status.json {"state": loading|ready|mapping|done|cancelled|error,
//                     "epoch": N, "chunk": k, "total": N, "message": "..."}
//        result.txt   (stitch) accumulated stitched output so far
//        results.jsonl (collect) one {"index","source","output","sep"} object per line
//
// Exits when the spool directory disappears (owner quit) or stdin hits EOF (parent death).

import Foundation
import MLX
import MLXLMCommon
import Tokenizers
import Chunking

final class MapServer {
    private enum Output { case stitch, collect }

    /// A map request read from the spool.
    private struct Job {
        let epoch: Int
        let text: String
        let budget: Int
        let output: Output
        let messagesData: Data  // JSON of the `messages` template (carries {{chunk}})
    }

    // Placeholder the job's message template must contain; replaced with each chunk's text.
    private static let placeholder = "{{chunk}}"

    private let modelDir: String
    private let spoolDir: URL
    private let gen: GenConfig
    private let extraEOSTokens: Set<String>

    private let jobURL: URL
    private let statusURL: URL
    private let resultTxtURL: URL
    private let resultsJsonlURL: URL
    private let stopURL: URL

    // How often the generation loop checks the stop flag (in generation events).
    private let stopCheckStride = 16

    init(modelDir: String, spoolDir: String, gen: GenConfig, extraEOSTokens: Set<String>) {
        self.modelDir = modelDir
        self.spoolDir = URL(fileURLWithPath: spoolDir, isDirectory: true)
        self.gen = gen
        self.extraEOSTokens = extraEOSTokens
        self.jobURL = self.spoolDir.appending(component: "job.json")
        self.statusURL = self.spoolDir.appending(component: "status.json")
        self.resultTxtURL = self.spoolDir.appending(component: "result.txt")
        self.resultsJsonlURL = self.spoolDir.appending(component: "results.jsonl")
        self.stopURL = self.spoolDir.appending(component: "stop")
    }

    // MARK: Run loop

    func run() async {
        // Parent death: stdin closes -> exit immediately (nothing to clean up but ourselves).
        FileHandle.standardInput.readabilityHandler = { handle in
            if handle.availableData.isEmpty { exit(0) }
        }

        let label = (modelDir as NSString).lastPathComponent
        writeStatus(state: "loading", message: "Loading model...")
        let container: ModelContainer
        do {
            container = try await loadModel(modelDir, extraEOSTokens: extraEOSTokens)
        } catch {
            writeStatus(state: "error", message: "Model load failed: \(error.localizedDescription)")
            log("model load failed: \(error.localizedDescription)")
            return
        }
        writeStatus(state: "ready", message: "Ready - \(label)")
        log("ready on \(label); watching \(spoolDir.path)")

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
                }
            case .job(let job):
                if job.epoch != lastEpoch {
                    lastEpoch = job.epoch
                    await process(job, container: container)
                }
            }
            try? await Task.sleep(nanoseconds: 150_000_000)  // 150 ms poll
        }
        log("spool directory gone; exiting")
    }

    // MARK: Job processing

    private func process(_ job: Job, container: ModelContainer) async {
        // Clear any stale stop flag so it cannot cancel this fresh job.
        try? FileManager.default.removeItem(at: stopURL)

        // The template must carry the placeholder in a substitutable value, else every chunk
        // would get the same text-less prompt - a producer mistake worth a clear error rather
        // than silent garbage.
        if !Self.templateHasPlaceholder(job.messagesData) {
            writeStatus(
                state: "error", epoch: job.epoch,
                message: "messages template has no \(Self.placeholder) placeholder value")
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

        // Chunk with the real tokenizer (one actor hop for the whole split).
        let chunks = await container.perform { context in
            Chunker.make(job.text, budget: job.budget) {
                context.tokenizer.encode(text: $0).count
            }
        }

        let total = chunks.count
        let params = gen.apply(to: GenerateParameters(maxTokens: 2048, temperature: 0))
        var accumulated = ""
        writeStatus(
            state: "mapping", epoch: job.epoch, chunk: 0, total: total,
            message: "Mapping 0 of \(total)")
        log("job \(job.epoch): \(job.output == .stitch ? "stitch" : "collect"), \(total) chunk(s)")

        for (i, chunk) in chunks.enumerated() {
            if stopRequested() {
                writeStatus(
                    state: "cancelled", epoch: job.epoch, chunk: i, total: total,
                    message: "Cancelled")
                log("job \(job.epoch): cancelled before chunk \(i + 1)")
                return
            }
            let output: String
            do {
                output = try await generate(
                    chunk: chunk.text, messagesData: job.messagesData,
                    params: params, container: container)
            } catch {
                writeStatus(
                    state: "error", epoch: job.epoch, chunk: i, total: total,
                    message: "Generation failed: \(error.localizedDescription)")
                log("job \(job.epoch): chunk \(i + 1) error: \(error.localizedDescription)")
                return
            }
            switch job.output {
            case .stitch:
                accumulated += output + chunk.sep
                writeText(accumulated, to: resultTxtURL)
            case .collect:
                appendRecord(
                    ["index": i, "source": chunk.text, "output": output, "sep": chunk.sep])
            }
            writeStatus(
                state: "mapping", epoch: job.epoch, chunk: i + 1, total: total,
                message: "Mapping \(i + 1) of \(total)")
            // A stop that arrived mid-chunk (generate() cancelled/drained) lands here.
            if stopRequested() {
                writeStatus(
                    state: "cancelled", epoch: job.epoch, chunk: i + 1, total: total,
                    message: "Cancelled")
                log("job \(job.epoch): cancelled after chunk \(i + 1)")
                return
            }
        }
        writeStatus(
            state: "done", epoch: job.epoch, chunk: total, total: total, message: "Done")
        log("job \(job.epoch): done")
    }

    /// Generate one chunk's output: render the job's message template with the chunk text
    /// substituted for `{{chunk}}`, apply the model's own chat template, and generate.
    ///
    /// The whole generation runs INSIDE `container.perform`, so the model's serial-access mutex
    /// is held for the entire decode (not just prefill). A stop request `cancel()`s the decode
    /// task but keeps draining the stream to its natural end, so the background token loop is
    /// fully finished before this closure returns and releases the mutex - the next job's
    /// generation can never overlap a still-running decode.
    private func generate(
        chunk: String, messagesData: Data, params: GenerateParameters, container: ModelContainer
    ) async throws -> String {
        let stopURL = self.stopURL
        let stride = self.stopCheckStride
        return try await container.perform { context -> String in
            let messages = try Self.renderMessages(messagesData, chunk: chunk)
            let tokens = try context.tokenizer.applyChatTemplate(messages: messages)
            let iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(tokens)), model: context.model,
                cache: nil, parameters: params)
            let (stream, task) = generateTask(
                promptTokenCount: tokens.count,
                modelConfiguration: context.configuration,
                tokenizer: context.tokenizer,
                iterator: iterator)
            var out = ""
            var seen = 0
            for await event in stream {
                if case .chunk(let piece) = event { out += piece }
                seen += 1
                // Cancel (do NOT break): keep consuming so the decode task observes cancellation
                // and finishes before we leave this closure and drop the model mutex.
                if seen % stride == 0
                    && FileManager.default.fileExists(atPath: stopURL.path)
                {
                    task.cancel()
                }
            }
            return out.trimmingCharacters(in: .whitespacesAndNewlines)
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
        // we surface as an error rather than dropping silently.
        guard let text = obj["text"] as? String else {
            return .invalid(epoch: epoch, reason: "job has no \"text\" string")
        }
        guard let messages = obj["messages"] as? [[String: Any]], !messages.isEmpty,
            let messagesData = try? JSONSerialization.data(withJSONObject: messages)
        else {
            return .invalid(epoch: epoch, reason: "job has no valid \"messages\" array")
        }
        let budget = (obj["budget_tokens"] as? NSNumber)?.intValue ?? 1200
        let output: Output = (obj["output"] as? String == "collect") ? .collect : .stitch
        return .job(
            Job(
                epoch: epoch, text: text, budget: max(64, budget), output: output,
                messagesData: messagesData))
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
        state: String, epoch: Int? = nil, chunk: Int? = nil, total: Int? = nil, message: String
    ) {
        var obj: [String: Any] = ["state": state, "message": message]
        if let epoch { obj["epoch"] = epoch }
        if let chunk { obj["chunk"] = chunk }
        if let total { obj["total"] = total }
        if let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]) {
            try? data.write(to: statusURL, options: .atomic)
        }
    }

    private func log(_ s: String) {
        FileHandle.standardError.write(Data("[mlx-agent map] \(s)\n".utf8))
    }
}
