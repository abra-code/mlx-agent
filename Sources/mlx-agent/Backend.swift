// Backend.swift - the generation seam the tool loop runs against.
//
// The tool loop (Agent.runTurn) is written against GenerationBackend, NOT against
// ChatSession directly, so an alternative backend (e.g. a remote OpenAI-compatible
// endpoint) can slot in without touching the loop. The currency types are MLXLMCommon's
// (Chat.Message, Generation, ToolSpec, ToolCall) - plain Codable data that any backend
// can translate to/from its own wire format. The MLX impl below wraps a ChatSession with
// toolDispatch LEFT NIL so tool calls surface as `.toolCall` Generation values for the
// external loop to dispatch; this is what lets us enforce the iteration cap / timeout /
// dedup guardrails that the library's internal restart loop (ChatSession.streamMap) does
// not provide.
//
// Two backends live here:
//   MLXBackend    - in-process mlx-swift-lm ChatSession (safetensors).
//   OpenAIBackend - a remote OpenAI-compatible /chat/completions endpoint (llama-server).

import Foundation
import MLXLLM
import MLXLMCommon

/// The generation seam the agent loop runs against.
protocol GenerationBackend: AnyObject, Sendable {
    /// Tool specifications the model should see. `nil` disables tool calling
    /// (chat mode). Setting it takes effect on the next `stream(_:)`.
    var tools: [ToolSpec]? { get set }

    /// Stream one model pass: append `messages` to the running context and generate
    /// until the model stops. Tool calls surface as `.toolCall` items - the CALLER
    /// (Agent.runTurn) dispatches them and feeds `.tool(...)` results back on the next
    /// call. The backend preserves conversational state (KV cache) across calls.
    func stream(_ messages: [Chat.Message]) -> AsyncThrowingStream<Generation, Error>

    /// Reset conversational state, keeping configuration (instructions/tools).
    func clear() async
}

/// mlx-swift-lm backend: a ChatSession driven with `toolDispatch == nil` so the tool
/// loop stays external. `streamDetails(to:)` yields `.chunk` / `.toolCall` / `.info`.
///
/// ChatSession is documented as not thread-safe and is not Sendable; this wrapper is
/// `@unchecked Sendable` on the same contract the rest of the agent honors - it is only
/// ever touched from within a single in-flight prompt Task at a time (ACPServer
/// serializes prompts; oneshot runs one turn). We never run two turns concurrently on
/// one session.
final class MLXBackend: GenerationBackend, @unchecked Sendable {
    private let session: ChatSession

    init(_ session: ChatSession) {
        self.session = session
    }

    var tools: [ToolSpec]? {
        get { session.tools }
        set { session.tools = newValue }
    }

    func stream(_ messages: [Chat.Message]) -> AsyncThrowingStream<Generation, Error> {
        session.streamDetails(to: messages)
    }

    func clear() async {
        await session.clear()
    }
}

// MARK: - OpenAI-compatible backend (llama-server)

/// Cancellation handle for the in-flight URLSession task. The stream's `onTermination`
/// can fire BEFORE `bytes(for:)` has returned its task (cancel arriving between request
/// send and first byte), so a cancel that lands early is latched and applied to the task
/// the moment it is handed over - otherwise llama-server would keep generating into a
/// dropped connection.
private final class URLTaskHandle: @unchecked Sendable {
    private let lock = NSLock()
    private var task: URLSessionTask?
    private var cancelled = false

    func adopt(_ task: URLSessionTask) {
        let cancelNow = lock.withLock { () -> Bool in
            if cancelled { return true }
            self.task = task
            return false
        }
        if cancelNow { task.cancel() }
    }

    func cancel() {
        let victim = lock.withLock { () -> URLSessionTask? in
            cancelled = true
            let t = task
            task = nil
            return t
        }
        victim?.cancel()
    }
}

/// Remote backend speaking the OpenAI `/chat/completions` streaming API, as served by
/// llama-server (`--jinja`, so tool calls are grammar-constrained and arrive as streaming
/// tool-call deltas). The applet owns the server process; this class only talks to it.
///
/// State the MLX path gets from ChatSession's KV cache, this class keeps explicitly:
/// `messages` is the running conversation, and every turn re-POSTs it in full (the server
/// prefix-caches, so a follow-up is still cheap). Serialization goes through the library's
/// `MessageGenerator` bridge rather than hand-rolled bookkeeping: `Chat.Message.Tool`'s
/// storage is fileprivate, so `tool_calls` / `tool_call_id` CANNOT be read back out of a
/// Chat.Message from here - `generate(message:)` is the only supported way to get them onto
/// the wire, and it is what makes a restored tool-bearing transcript (session/prime) emit a
/// well-formed request.
///
/// `@unchecked Sendable` on the same contract as MLXBackend: one in-flight turn at a time
/// (ACPServer serializes prompts; oneshot runs one turn), with the message array guarded by
/// a lock so the ACP thread can still read it.
final class OpenAIBackend: GenerationBackend, @unchecked Sendable {
    private let chatURL: URL
    private let systemPrompt: String?
    private let parameters: GenerateParameters
    private let urlSession: URLSession
    private let lock = NSLock()
    private var messages: [Chat.Message]
    private var toolSpecs: [ToolSpec]?
    private var passCounter = 0        // tags synthesized tool-call ids; see nextPassID()
    /// Text streamed by a pass that has not committed an assistant message yet. A turn
    /// abandoned mid-answer (Stop, or the server dying) leaves its text here and the NEXT
    /// stream(_:) flushes it - see flushAbandonedTextLocked().
    private var abandonedText = ""

    /// - Parameters:
    ///   - baseURL: the OpenAI-compatible root, e.g. `http://127.0.0.1:8099/v1`.
    ///   - parameters: the same GenerateParameters the MLX path builds (see
    ///     ACPServer.buildSessionStack); the sampling knobs are mapped onto the wire.
    ///   - systemPrompt: prepended as a `system` turn; nil prepends nothing (translator mode).
    ///   - seedHistory: prior turns to resume from (session/prime), system message excluded.
    init(
        baseURL: URL, parameters: GenerateParameters, systemPrompt: String?,
        seedHistory: [Chat.Message] = []
    ) {
        self.chatURL = baseURL.appendingPathComponent("chat/completions")
        self.systemPrompt = systemPrompt
        self.parameters = parameters
        self.messages = (systemPrompt.map { [Chat.Message.system($0)] } ?? []) + seedHistory
        let config = URLSessionConfiguration.ephemeral
        // Idle timeout between SSE frames, not a cap on the whole generation: a long
        // answer streams continuously, but a wedged server must not hang the turn forever.
        config.timeoutIntervalForRequest = 300
        config.waitsForConnectivity = false
        self.urlSession = URLSession(configuration: config)
    }

    /// session/prime builds a REPLACEMENT backend on every restore, so the outgoing one has
    /// to release its session rather than accumulate one per primed conversation. Cancelling
    /// is right at deinit: nothing references this backend, so any request still in flight is
    /// generating for a stack nobody will read (a live turn holds `self` inside runStream).
    deinit {
        urlSession.invalidateAndCancel()
    }

    var tools: [ToolSpec]? {
        get { lock.withLock { toolSpecs } }
        set { lock.withLock { toolSpecs = newValue } }
    }

    func clear() async {
        lock.withLock {
            messages = systemPrompt.map { [Chat.Message.system($0)] } ?? []
            abandonedText = ""
        }
    }

    /// Commit the text of a pass that ended without recording an assistant message (Stop, or
    /// the server dying mid-answer), so the conversation never jumps straight from one user
    /// turn to the next and the model's history matches the partial answer still on screen.
    ///
    /// This runs at the START of the next pass rather than in the abandoned pass's own catch
    /// block, and that is the whole point: cancellation resolves the turn as soon as the
    /// stream terminates, so the cancelled task's unwinding races the next prompt. Flushing
    /// under the same lock acquisition that appends the next input makes the ordering a fact
    /// rather than a hope. Caller must hold `lock`.
    private func flushAbandonedTextLocked() {
        guard !abandonedText.isEmpty else { return }
        messages.append(.assistant(abandonedText))
        abandonedText = ""
    }

    /// `<base minus /v1>/health` - llama-server's health endpoint is NOT under the /v1
    /// prefix, so the OpenAI base-url has to be trimmed back to the server root.
    static func healthURL(base: URL) -> URL {
        var s = base.absoluteString
        while s.hasSuffix("/") { s.removeLast() }
        if s.hasSuffix("/v1") { s.removeLast(3) }
        return URL(string: s + "/health") ?? base
    }

    /// Poll `/health` until the server answers 200, so session/new fails with a clean,
    /// actionable error instead of every prompt failing later. Throws on exhaustion.
    static func waitForHealth(baseURL: URL, attempts: Int = 3, timeout: TimeInterval = 2) async throws {
        let url = healthURL(base: baseURL)
        var lastError = "no response"
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = timeout
        config.waitsForConnectivity = false
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }
        for attempt in 1...max(1, attempts) {
            do {
                let (_, response) = try await session.data(from: url)
                if let http = response as? HTTPURLResponse, http.statusCode == 200 { return }
                lastError = "HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)"
            } catch {
                lastError = error.localizedDescription
            }
            if attempt < max(1, attempts) {
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
        }
        throw NSError(
            domain: "mlx-agent", code: 4,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "no OpenAI-compatible server at \(url.absoluteString): \(lastError)"
            ])
    }

    // MARK: - Streaming

    func stream(_ new: [Chat.Message]) -> AsyncThrowingStream<Generation, Error> {
        lock.withLock {
            flushAbandonedTextLocked()   // an abandoned answer lands BEFORE this pass's input
            messages.append(contentsOf: new)
        }
        let handle = URLTaskHandle()
        return AsyncThrowingStream { continuation in
            let task = Task { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                do {
                    try await self.runStream(continuation: continuation, handle: handle)
                    continuation.finish()
                } catch {
                    // A cancelled URLSession task surfaces as URLError.cancelled; report it
                    // as cancellation so the loop reads it as such rather than a failure.
                    if (error as? URLError)?.code == .cancelled || error is CancellationError {
                        continuation.finish(throwing: CancellationError())
                    } else {
                        continuation.finish(throwing: error)
                    }
                }
            }
            continuation.onTermination = { _ in
                handle.cancel()
                task.cancel()
            }
        }
    }

    /// One model pass: POST the running conversation, parse the SSE frames into Generation
    /// values, and append the assistant turn (text + any tool calls) to `messages` before
    /// returning, so the next pass carries it.
    private func runStream(
        continuation: AsyncThrowingStream<Generation, Error>.Continuation, handle: URLTaskHandle
    ) async throws {
        let request = try buildRequest()
        let started = Date()
        let (bytes, response) = try await urlSession.bytes(for: request)
        handle.adopt(bytes.task)

        guard let http = response as? HTTPURLResponse else {
            throw Self.error("no HTTP response from \(chatURL.absoluteString)")
        }
        guard http.statusCode == 200 else {
            // The error body explains WHY (bad request, no model loaded, ...); surface a
            // bounded prefix of it rather than a bare status code. One line is the whole
            // body for llama-server's JSON errors, and stopping after it keeps a pathological
            // body from being walked line by line.
            var body = ""
            for try await line in bytes.lines {
                body = String(line.prefix(1000))
                break
            }
            handle.cancel()   // we are abandoning the response; do not leave the task running
            throw Self.error("llama-server returned HTTP \(http.statusCode): \(body)")
        }

        var text = ""                      // accumulated assistant content, <think> tags VERBATIM
        var calls = ToolCallAccumulator()
        var usage: (prompt: Int, completion: Int)? = nil
        var firstToken: Date? = nil

        // A throw from here (Stop, or the server dying mid-answer) deliberately carries no
        // recovery: the text streamed so far is already in `abandonedText`, and the NEXT
        // pass flushes it. Committing it from a catch block here would race the next prompt,
        // which the turn's own cancellation has already unblocked.
        for try await line in bytes.lines {
            try Task.checkCancellation()
            // SSE: skip blank separators and ":" keep-alive comments; only data: matters.
            guard line.hasPrefix("data:") else { continue }
            let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            if payload == "[DONE]" { break }
            guard let data = payload.data(using: .utf8),
                let frame = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            else {
                log("dropping unparseable SSE frame: \(payload.prefix(160))")
                continue
            }
            if let u = frame["usage"] as? [String: Any] {
                usage = (
                    (u["prompt_tokens"] as? NSNumber)?.intValue ?? 0,
                    (u["completion_tokens"] as? NSNumber)?.intValue ?? 0
                )
            }
            guard let choice = (frame["choices"] as? [[String: Any]])?.first else { continue }
            if let delta = choice["delta"] as? [String: Any] {
                if let chunk = delta["content"] as? String, !chunk.isEmpty {
                    if firstToken == nil { firstToken = Date() }
                    text += chunk
                    // Publish as we go: if this pass never reaches its commit below, the
                    // next one flushes this text instead of losing it.
                    lock.withLock { abandonedText = text }
                    continuation.yield(.chunk(chunk))
                }
                if let raw = delta["tool_calls"] as? [[String: Any]] {
                    if firstToken == nil { firstToken = Date() }
                    calls.feed(raw)
                }
            }
        }

        let finished = Date()
        let resolved = calls.finish(passID: nextPassID()) { [weak self] in self?.log($0) }
        // The pass completed, so it commits its own assistant turn: ONE message carrying this
        // pass's text AND all its tool calls (with the server's ids), so Agent's
        // `.tool(result, id: call.id)` answers announce correctly. Clearing `abandonedText`
        // under the same lock is what stops the next pass from flushing this text twice.
        lock.withLock {
            abandonedText = ""
            if !resolved.isEmpty {
                messages.append(.assistant(text, toolCalls: resolved))
            } else if !text.isEmpty {
                messages.append(.assistant(text))
            }
        }
        for call in resolved { continuation.yield(.toolCall(call)) }

        if let usage {
            // Split the wall clock at the first token: everything before it is prefill, the
            // rest is decode. The server reports no timings, and Agent derives tok/s from
            // generateTime, so an honest approximation beats zeros.
            let decodeStart = firstToken ?? finished
            continuation.yield(
                .info(
                    GenerateCompletionInfo(
                        promptTokenCount: usage.prompt,
                        generationTokenCount: usage.completion,
                        promptTime: decodeStart.timeIntervalSince(started),
                        generationTime: finished.timeIntervalSince(decodeStart))))
        }
    }

    private func buildRequest() throws -> URLRequest {
        let (wire, specs) = lock.withLock {
            (Self.wireMessages(messages, log: { self.log($0) }), toolSpecs)
        }
        var body: [String: Any] = [
            // llama-server serves whatever model it was launched with; any id is accepted.
            "model": "auto",
            "messages": wire,
            "stream": true,
            "stream_options": ["include_usage": true],
            // Raw <think> tags in delta.content: ThinkSplitter already routes them to
            // agent_thought_chunk, and KV-cache parity wants them kept in the history.
            "reasoning_format": "none",
            "temperature": Self.round6(parameters.temperature),
        ]
        if let maxTokens = parameters.maxTokens { body["max_tokens"] = maxTokens }
        if parameters.topP < 1 { body["top_p"] = Self.round6(parameters.topP) }
        if let penalty = parameters.repetitionPenalty { body["repeat_penalty"] = Self.round6(penalty) }
        if let seed = parameters.seed { body["seed"] = seed > UInt64(Int.max) ? Int.max : Int(seed) }
        if let specs, !specs.isEmpty {
            body["tools"] = specs.map { $0 as [String: Any] }
            body["tool_choice"] = "auto"
        }
        var request = URLRequest(url: chatURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        return request
    }

    /// Serialize the conversation to OpenAI wire dictionaries via the library's
    /// MessageGenerator bridge (role/content + tool_calls/tool_call_id), fix up the one
    /// place it diverges from the OpenAI spec (`addToolMetadata` emits
    /// `function.arguments` as a JSON OBJECT; the spec and llama-server's parser want a
    /// JSON STRING), then strip any tool-call announcement the conversation never answered.
    ///
    /// Normalizing HERE rather than on `messages` is forced by the library: `Chat.Message`'s
    /// tool storage is fileprivate, so the calls are only readable once serialized.
    static func wireMessages(
        _ messages: [Chat.Message], log: (String) -> Void = { _ in }
    ) -> [[String: Any]] {
        let generator = DefaultMessageGenerator()
        let raw = messages.map { message -> [String: Any] in
            var out = generator.generate(message: message) as [String: Any]
            if let entries = out["tool_calls"] as? [[String: Any]] {
                out["tool_calls"] = entries.map { entry -> [String: Any] in
                    var call = entry
                    if var function = call["function"] as? [String: Any] {
                        if let arguments = function["arguments"], !(arguments is String) {
                            function["arguments"] = jsonString(arguments)
                        }
                        call["function"] = function
                    }
                    return call
                }
            }
            return out
        }
        // Order matters: the strip can DELETE a text-less announcement, which is one of the
        // ways two user turns end up adjacent - so merge after it, not before.
        return mergeAdjacentUserTurns(stripUnansweredToolCalls(raw, log: log))
    }

    /// Drop every `tool_calls` entry that no following tool message answers, keeping the
    /// announcement's answered entries and its text (and dropping the message when that
    /// leaves nothing) - the rule `ACPServer.primeHistory` enforces on a restored
    /// transcript, applied to the LIVE conversation for the same reason: an announcement
    /// with no results leaves the chat template mid-tool-turn, and llama-server renders that
    /// template server-side on every single turn, so a strict template (Mistral-family
    /// `raise_exception`) would hard-fail every subsequent prompt in the session, not just
    /// the turn that caused it.
    ///
    /// This state is reachable and NOT a race: `Agent.runTurn` dispatches tool calls but
    /// only feeds the results back on its NEXT `stream(_:)` call, and it has exits that
    /// never make that call - the tool-iteration cap, a permission cancel, and a
    /// session/cancel landing during dispatch. All three are non-fatal, so the session
    /// lives on carrying the dangling announcement.
    ///
    /// Filtering PER ENTRY rather than dropping the whole array matters for the partially
    /// answered shape (2 calls announced, 1 answered), which `primeHistory` accepts and
    /// passes through mid-transcript: dropping the array wholesale would leave the answered
    /// tool message behind as an ORPHAN - the mirror image of the bug this exists to
    /// prevent, and just as malformed.
    static func stripUnansweredToolCalls(
        _ raw: [[String: Any]], log: (String) -> Void = { _ in }
    ) -> [[String: Any]] {
        var out: [[String: Any]] = []
        for (index, message) in raw.enumerated() {
            guard let entries = message["tool_calls"] as? [[String: Any]], !entries.isEmpty else {
                out.append(message)
                continue
            }
            // The answers are the contiguous run of tool messages right after this one.
            var answered: Set<String> = []
            var scan = index + 1
            while scan < raw.count, (raw[scan]["role"] as? String) == "tool" {
                if let id = raw[scan]["tool_call_id"] as? String { answered.insert(id) }
                scan += 1
            }
            // An entry with no id can never be answered (a tool message correlates by id),
            // so it is unanswerable by construction and goes with the rest.
            let kept = entries.filter { ($0["id"] as? String).map(answered.contains) ?? false }
            if kept.count == entries.count {
                out.append(message)
                continue
            }
            log("stripping \(entries.count - kept.count) unanswered tool call(s) from the wire history")
            var fixed = message
            let content = (fixed["content"] as? String) ?? ""
            if kept.isEmpty {
                fixed["tool_calls"] = nil
                if !content.isEmpty { out.append(fixed) }
            } else {
                fixed["tool_calls"] = kept
                out.append(fixed)
            }
        }
        return out
    }

    /// Collapse consecutive user turns into one, joined by a blank line.
    ///
    /// A pass that produced NO content leaves nothing for `flushAbandonedTextLocked` to
    /// flush, so the conversation can genuinely hold two user turns in a row: the server
    /// refused the connection (the applet restarting llama-server to switch models is
    /// exactly this), Stop landed during prefill before the first token, or a text-less
    /// tool-call announcement hit the iteration cap and lost its message to the strip above.
    /// All three are ordinary, and all three produce a role sequence a strict template
    /// rejects. Merging keeps every word the user typed; dropping either turn would not.
    static func mergeAdjacentUserTurns(_ raw: [[String: Any]]) -> [[String: Any]] {
        var out: [[String: Any]] = []
        for message in raw {
            if (message["role"] as? String) == "user",
                let previous = out.last, (previous["role"] as? String) == "user"
            {
                var merged = previous
                let a = (previous["content"] as? String) ?? ""
                let b = (message["content"] as? String) ?? ""
                merged["content"] = a.isEmpty ? b : (b.isEmpty ? a : a + "\n\n" + b)
                out[out.count - 1] = merged
                continue
            }
            out.append(message)
        }
        return out
    }

    fileprivate static func jsonString(_ value: Any) -> String {
        guard JSONSerialization.isValidJSONObject(value),
            let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
            let s = String(data: data, encoding: .utf8)
        else { return "{}" }
        return s
    }

    private static func round6(_ v: Float) -> Double {
        (Double(v) * 1_000_000).rounded() / 1_000_000
    }

    /// Monotonic per-pass tag for synthesized tool-call ids. Slot indices restart at 0 on
    /// every pass, so an id built from the slot alone would repeat across passes and put two
    /// tool messages with the SAME tool_call_id into one conversation.
    private func nextPassID() -> Int {
        lock.withLock {
            passCounter += 1
            return passCounter
        }
    }

    private static func error(_ message: String) -> NSError {
        NSError(domain: "mlx-agent", code: 5, userInfo: [NSLocalizedDescriptionKey: message])
    }

    private func log(_ s: String) {
        FileHandle.standardError.write(Data("[mlx-agent openai] \(s)\n".utf8))
    }
}

/// Reassembles streamed tool-call deltas. llama-server emits a tool call across many
/// frames: the first carries `id` + `function.name`, later ones append fragments of
/// `function.arguments`, and several calls can interleave by `index`. Slots are keyed by
/// that index; a frame without one is matched to an existing slot by id, else appended.
private struct ToolCallAccumulator {
    private struct Partial {
        var id: String?
        var name = ""
        var arguments = ""
    }
    private var partials: [Int: Partial] = [:]
    private var order: [Int] = []

    mutating func feed(_ entries: [[String: Any]]) {
        for entry in entries {
            let key = slot(for: entry)
            var partial = partials[key] ?? Partial()
            if partials[key] == nil { order.append(key) }
            if let id = entry["id"] as? String, !id.isEmpty { partial.id = id }
            if let function = entry["function"] as? [String: Any] {
                if let name = function["name"] as? String, !name.isEmpty { partial.name = name }
                // Normally a String fragment to concatenate; tolerate a server that sends
                // the whole object at once rather than dropping the arguments on the floor.
                if let fragment = function["arguments"] as? String {
                    partial.arguments += fragment
                } else if let object = function["arguments"] {
                    partial.arguments = OpenAIBackend.jsonString(object)
                }
            }
            partials[key] = partial
        }
    }

    /// The slot an entry belongs to. `index` is what the OpenAI spec and llama-server always
    /// send; the fallbacks only matter for a non-conforming endpoint. An id matches an
    /// existing slot; otherwise the frame extends the MOST RECENT slot rather than opening a
    /// new one - an arguments-only continuation frame (the case with neither index nor id)
    /// is by definition more of the call already in progress, and minting a fresh slot per
    /// fragment would strand the name in slot 0 and drop every fragment as unnamed.
    private func slot(for entry: [String: Any]) -> Int {
        if let index = (entry["index"] as? NSNumber)?.intValue { return index }
        if let id = entry["id"] as? String, let match = order.first(where: { partials[$0]?.id == id }) {
            return match
        }
        if let id = entry["id"] as? String, !id.isEmpty {
            return (order.max() ?? -1) + 1   // a new id genuinely opens a new call
        }
        return order.last ?? 0
    }

    /// Materialize the accumulated calls in arrival order.
    ///
    /// Dropped rather than dispatched: a call the server left unnamed (nothing could route
    /// it), and one whose accumulated `arguments` are non-empty but unparseable - that is
    /// what a call truncated at max_tokens looks like (`{"path": "/etc/ho`), and dispatching
    /// it as `read_file()` with silently-empty arguments would run the WRONG call, record a
    /// false `{}` announcement into the history, and collide in Agent's name+args dupCache
    /// with any other truncated call to the same tool. Empty arguments stay `{}` - that is a
    /// legitimate no-argument call.
    ///
    /// A call without an id gets a synthetic one (tagged with `passID`, unique across the
    /// conversation) so the announcement and Agent's `.tool(_, id:)` answer agree: a tool
    /// message with no tool_call_id would break the server's chat template on the next pass.
    func finish(passID: Int, log: (String) -> Void = { _ in }) -> [ToolCall] {
        var out: [ToolCall] = []
        for key in order {
            guard let partial = partials[key], !partial.name.isEmpty else { continue }
            let trimmed = partial.arguments.trimmingCharacters(in: .whitespacesAndNewlines)
            var arguments: [String: JSONValue] = [:]
            if !trimmed.isEmpty {
                guard let decoded = try? JSONDecoder().decode(
                    [String: JSONValue].self, from: Data(trimmed.utf8))
                else {
                    log("dropping tool call \(partial.name): unparseable arguments (\(trimmed.prefix(160)))")
                    continue
                }
                arguments = decoded
            }
            out.append(
                ToolCall(
                    function: .init(name: partial.name, arguments: arguments),
                    id: partial.id ?? "call_\(partial.name)_\(passID)_\(key)"))
        }
        return out
    }
}
