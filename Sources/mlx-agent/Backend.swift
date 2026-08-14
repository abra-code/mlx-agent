// Backend.swift - the generation seam the tool loop runs against.
//
// The tool loop (Agent.runTurn) is written against GenerationBackend, NOT against a
// concrete engine, so an alternative backend (e.g. a remote OpenAI-compatible endpoint)
// can slot in without touching the loop. The currency types are MLXLMCommon's
// (Chat.Message, Generation, ToolSpec, ToolCall) - plain Codable data that any backend
// can translate to/from its own wire format. Tool calls surface as `.toolCall` events for
// the external loop to dispatch, which is what lets us enforce the iteration cap /
// timeout / dedup guardrails that mlx-swift-lm's own internal restart loop does not
// provide.
//
// Two backends live here, and they are deliberately symmetric: each owns its conversation
// as [Chat.Message], sends the whole of it every pass, and guards that state with a lock.
//   MLXBackend    - in-process mlx-swift-lm (safetensors), driving TokenIterator directly.
//   OpenAIBackend - a remote OpenAI-compatible /chat/completions endpoint (llama-server).

import Foundation
import AgentText
import MLX
import MLXLMCommon
import MLXLLM

/// What one model pass emits. This is MLXLMCommon's `Generation` plus the one case it has
/// no room for: reasoning the SERVER already separated out.
///
/// `Generation` is a closed public enum (`chunk` / `toolCall` / `info`), so `.reasoning`
/// cannot be added to it. The distinction has to survive the trip to Agent, because the
/// two kinds of reasoning arrive with different certainty:
///   .text      - raw model output. May carry inline <think> markers; ThinkSplitter has to
///                parse it to find out, and a marker may straddle chunk boundaries.
///   .reasoning - llama-server ALREADY classified this (`reasoning_content`). It must
///                bypass ThinkSplitter: re-parsing text that is known to be reasoning would
///                let a literal "</think>" inside it close the block early and spill the
///                remainder into the visible answer - the very bug class this all exists to
///                prevent.
enum BackendEvent: Sendable {
    /// Raw model text, to be classified by ThinkSplitter.
    case text(String)
    /// Text the backend knows is reasoning; goes straight to thought.
    case reasoning(String)
    case toolCall(ToolCall)
    case info(GenerateCompletionInfo)
}

/// The generation seam the agent loop runs against.
protocol GenerationBackend: AnyObject, Sendable {
    /// Tool specifications the model should see. `nil` disables tool calling
    /// (chat mode). Setting it takes effect on the next `stream(_:)`.
    var tools: [ToolSpec]? { get set }

    /// Stream one model pass: append `messages` to the running context and generate
    /// until the model stops. Tool calls surface as `.toolCall` items - the CALLER
    /// (Agent.runTurn) dispatches them and feeds `.tool(...)` results back on the next
    /// call. The backend preserves conversational state (KV cache) across calls.
    func stream(_ messages: [Chat.Message]) -> AsyncThrowingStream<BackendEvent, Error>

    /// Reset conversational state, keeping configuration (instructions/tools).
    func clear() async
}

/// mlx-swift-lm backend. Owns the conversation and the KV cache, and renders the FULL
/// conversation through the model's chat template on every pass.
///
/// WHY NOT ChatSession - this is the fix for a tool-call loop, so do not "simplify" it back.
///
/// ChatSession is incremental: it keeps conversation state in the KV cache and applies the chat
/// template to only the messages appended since the previous pass. But it passes `tools` (and
/// re-prepends `instructions`) on EVERY application. Chat templates are written to render a
/// COMPLETE conversation, so applying one to a suffix re-emits whatever preamble it puts at the
/// top. For Qwen3 that top is `{%- if tools %}`, so every tool-result pass appended:
///
///     <|im_start|>system
///     {the whole system prompt}
///
///     # Tools
///     You may call one or more functions to assist with the user query.
///     ...<tools>{...}</tools>...
///     <|im_end|>
///     <|im_start|>user
///     <tool_response>
///     ...the result...
///     </tool_response><|im_end|>
///     <|im_start|>assistant
///
/// A fresh "you may call one or more functions" instruction landed immediately before every
/// assistant turn, so the model called the tool again, and again, until Agent's iteration cap
/// ended the turn with no answer. Measured in Cadabra 2026-07-27: eight identical `pdf_text`
/// calls on "summarize this PDF", thoughts empty from the third on. ChatSession's own internal
/// tool loop (`toolDispatch`) has the same defect, so using that instead would not help.
///
/// Rendering the whole conversation each pass is simply what the template contract asks for, and
/// it is template-agnostic - no per-model special-casing, which is what every narrower workaround
/// would have needed (this codebase has already been bitten once by a model-specific template
/// quirk). Conversation ownership mirrors OpenAIBackend, which already works exactly this way.
///
/// The KV cache is still reused: the freshly rendered token sequence is matched against the
/// tokens the cache already holds, the cache is trimmed back to the common prefix, and only the
/// genuinely new tokens are prefilled. In steady state that is the new tool result plus a
/// re-render of the last assistant turn - everything before it is reused. If the cache cannot be
/// trimmed, it is rebuilt instead: slower, never wrong.
///
/// Two costs this design pays that the old incremental one did not, both logged rather than hidden:
///
/// - **Sliding-window models re-prefill everything.** `RotatingKVCache.isTrimmable` goes false once
///   the conversation passes the window, and a trim is always needed (the re-rendered assistant turn
///   never matches the raw generated tokens), so Gemma 3/4/3n, GPT-OSS, Exaone4, Mistral3 and
///   friends rebuild every pass. Append-only never had to trim, so it never hit this.
/// - **The whole conversation is re-rendered and re-tokenized every pass**, even to feed a hundred
///   new tokens. Cheap next to prefill, but it scales with conversation length times tool
///   iterations.
///
/// `@unchecked Sendable`: `toolSpecs` is lock-guarded and everything else is reachable only from
/// inside `PassGate`, which serializes entire passes. Do NOT go back to relying on "one prompt at a
/// time" - the ACP cancel window breaks that assumption, and the crash it caused is described on
/// PassGate.
final class MLXBackend: GenerationBackend, @unchecked Sendable {

    /// Carries the non-Sendable model pieces out of `ModelContainer.perform`; the library's own
    /// `SendableBox` is `package`-scoped. Using the model outside the lock is the same contract
    /// ChatSession relies on: the weights are not mutated, and each backend owns a private
    /// KVCache, which is the part that genuinely cannot be shared.
    private final class Box<T>: @unchecked Sendable {
        let value: T
        init(_ value: T) { self.value = value }
    }

    /// Serializes a whole pass - append, render, prefill, generate, record - against this backend.
    ///
    /// The type and the rationale now live in PassGate.swift, because a second backend needed
    /// exactly the same guarantee. What it protects HERE: without it, two overlapping passes
    /// interleave and the abandoned one's `cachedTokens`/`messages` writes land while the new
    /// pass trims the same cache. Measured as a hard `[broadcast_shapes]` SIGTRAP that kills the
    /// process, at round 6-11 of a cancel/prompt loop. It replaces protection ChatSession used
    /// to provide and that dropping it silently removed (`streamMap` wrapped its whole
    /// generation in `cache.update { }` on a SerialAccessContainer).
    private let gate = PassGate()

    /// Runs `body` as the only pass in flight against this backend.
    private func withPass<T>(_ body: () async throws -> T) async rethrows -> T {
        try await mlx_agent.withPass(gate, body)
    }

    private let container: ModelContainer
    private let instructions: String?
    private let parameters: GenerateParameters

    /// Guards `toolSpecs` only. Everything else is reachable exclusively from inside `gate`.
    private let lock = NSLock()
    private var toolSpecs: [ToolSpec]?

    /// The authoritative conversation, system message included. Gate-owned.
    private var messages: [Chat.Message]
    /// Whether the last recorded assistant message carried tool calls that have not been answered.
    /// Tracked here because `Chat.Message.Tool.storage` is fileprivate to the library - a recorded
    /// message cannot be inspected for tool calls after the fact. Gate-owned.
    private var lastAssistantHasUnansweredCalls = false

    /// Projects the recorded assistant text down to the ANSWER, so reasoning never re-enters the
    /// prompt as ordinary assistant content. Gate-owned.
    ///
    /// This is the parity OpenAIBackend already had: its `text` accumulates `delta.content` and
    /// never `reasoning_content`, so its history has always been reasoning-free. The MLX path
    /// recorded the raw stream instead, markers and all.
    ///
    /// Gemma 4's template hid that for the common case - its `strip_thinking()` macro removes
    /// channel blocks when re-rendering assistant content - but it removes EVERY channel, and
    /// Gemma 4 names the channel carrying the ANSWER `content`. Rendered against the real
    /// template, an assistant turn recorded as `<|channel>content ... <channel|>` comes back
    /// EMPTY: the reply is erased from the history it belongs in. `<think>` families have no
    /// such macro at all, so for them the raw markers went straight back into the next prompt.
    ///
    /// Deliberately a SECOND splitter rather than a reclassification of the event stream: Agent
    /// stays the sole classifier of what the reader sees. A backend yielding `.reasoning` in
    /// place of raw `.text` would let a thought segment overtake message text still sitting in
    /// Agent's hold-back and reorder the two on screen. Two splitters over the same bytes is the
    /// price of exact display ordering; it is pure string scanning, invisible next to decode.
    private var historySplitter = ThinkSplitter()

    /// The KV cache, and the prompt tokens it is known to hold. The cache also holds whatever the
    /// model generated after that prompt, whose token ids we never see; the surplus is trimmed on
    /// the next pass, which is also how a canceled or failed pass self-heals. Gate-owned.
    private var cache: [KVCache] = []
    private var cachedTokens: [Int] = []

    init(
        container: ModelContainer, instructions: String?, history: [Chat.Message] = [],
        parameters: GenerateParameters, tools: [ToolSpec]? = nil
    ) {
        self.container = container
        self.instructions = instructions
        self.parameters = parameters
        self.toolSpecs = tools
        self.messages = (instructions.map { [Chat.Message.system($0)] } ?? []) + history
    }

    var tools: [ToolSpec]? {
        get { lock.withLock { toolSpecs } }
        set { lock.withLock { toolSpecs = newValue } }
    }

    /// The library's `Generation` values, relabeled as `BackendEvent`. This path never yields
    /// `.reasoning`: mlx-swift-lm streams raw text, so any <think> markers are inline and it is
    /// ThinkSplitter's job to find them.
    ///
    /// `new` is appended INSIDE the gate rather than here, so that a pass abandoned by a cancel
    /// finishes recording its own turn before the next prompt's turn lands. Appending here would
    /// make the ordering a race - the hazard OpenAIBackend documents on `flushAbandonedTextLocked`.
    func stream(_ new: [Chat.Message]) -> AsyncThrowingStream<BackendEvent, Error> {
        // Boxed because Chat.Message is not Sendable and a Task closure is `sending`. Ownership
        // really is transferred - the caller builds this array for us and does not keep it.
        let input = Box(new)
        return AsyncThrowingStream { continuation in
            // weak: an idle-unload drops the backend to release the model, and a pass Task holding
            // self strongly would pin the weights it is trying to free.
            let task = Task { [weak self] in
                guard let self else { return continuation.finish() }
                do {
                    try await self.withPass {
                        try await self.runPass(appending: input.value, yielding: continuation)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    func clear() async {
        await withPass {
            messages = instructions.map { [Chat.Message.system($0)] } ?? []
            lastAssistantHasUnansweredCalls = false
            historySplitter = ThinkSplitter()
            cache = []
            cachedTokens = []
        }
    }

    private func log(_ s: String) {
        FileHandle.standardError.write(Data("[mlx-agent mlx] \(s)\n".utf8))
    }

    /// Keeps the conversation renderable before the new turn lands.
    ///
    /// Both hazards are the price of rendering the whole history every pass, and OpenAIBackend
    /// already pays it (`stripUnansweredToolCalls`, `mergeAdjacentUserTurns`) for the same reason:
    /// a strict template `raise_exception`s on a malformed history, and because every later pass
    /// re-renders the same messages, that failure would repeat for the rest of the session rather
    /// than spoiling one turn.
    ///
    /// - An assistant turn announcing tool calls that were never answered - the iteration cap fires
    ///   between announcement and dispatch, a permission is canceled, or a cancel lands mid-
    ///   dispatch. Reachable today with `--max-tool-iters 1`.
    /// - Two user turns in a row, when a pass throws and records no assistant reply.
    private func reconcileHistory(before new: [Chat.Message]) {
        // A user turn starts a new reasoning scope, so the projection splitter restarts with it -
        // the same per-turn lifetime Agent gives its display splitter. WITHIN a turn the state
        // must survive, because a reasoning block may span a tool call and the pass after it has
        // to know it is still inside one.
        if new.first?.role == .user { historySplitter = ThinkSplitter() }

        if lastAssistantHasUnansweredCalls, new.first?.role != .tool, let last = messages.indices.last {
            log("dropping unanswered tool-call announcement from the rendered history")
            messages[last] = .assistant(messages[last].content)
        }
        lastAssistantHasUnansweredCalls = false

        guard let first = new.first, first.role == .user,
            let last = messages.indices.last, messages[last].role == .user
        else {
            messages.append(contentsOf: new)
            return
        }
        messages[last] = .user(messages[last].content + "\n\n" + first.content)
        messages.append(contentsOf: new.dropFirst())
    }

    private func runPass(
        appending new: [Chat.Message],
        yielding continuation: AsyncThrowingStream<BackendEvent, Error>.Continuation
    ) async throws {
        reconcileHistory(before: new)
        let toolSpecs = self.tools
        let pieces = await container.perform { context in
            Box(
                (
                    model: context.model, processor: context.processor,
                    tokenizer: context.tokenizer, configuration: context.configuration
                ))
        }
        let (model, processor, tokenizer, configuration) = pieces.value

        let input = try await processor.prepare(
            input: UserInput(chat: messages, tools: toolSpecs))

        // Images/videos/audio opt out of token-level reuse: the processor builds media payloads
        // alongside the token array, and slicing the tokens would desync them. Text-only is the
        // only case this agent actually produces, but degrading safely beats assuming.
        let hasMedia = messages.contains {
            !$0.images.isEmpty || !$0.videos.isEmpty || !$0.audios.isEmpty
        }

        let fullTokens = input.text.tokens.asArray(Int32.self).map(Int.init)
        var reused = 0

        // Reuse requires a cache that can be trimmed AND that reports where it is. Both checks are
        // out here rather than inside the `surplus > 0` branch below, because the failure they
        // catch is precisely one where `surplus` computes to 0 and the branch never runs:
        //
        //   - `BaseKVCache.offset` is a plain stored property that ArraysCache / MambaCache /
        //     CacheList never write, so `offset` reads 0 forever on recurrent and hybrid models
        //     (Mamba2, FalconH1, BaichuanM1, and - depending on which layer lands first -
        //     Qwen3-Next, Qwen3.5, LFM2, Jamba, NemotronH). Trusting it would feed the full prompt
        //     into a cache that already holds the conversation: no crash, just silently doubled
        //     context. `isTrimmable` is false for exactly those, so this catches them.
        //   - RotatingKVCache reports `isTrimmable` false once the conversation passes its sliding
        //     window, so those families (Gemma 3/4/3n, GPT-OSS, Exaone4, Mistral3, ...) rebuild and
        //     re-prefill every pass. That is a real cost this design pays and ChatSession did not,
        //     because append-only never needed to trim. Logged rather than hidden.
        let offsets = cache.map(\.offset)
        let uniformOffsets = offsets.allSatisfy { $0 == offsets.first }
        if hasMedia || cache.isEmpty || !canTrimPromptCache(cache) || !uniformOffsets {
            if !cache.isEmpty && !hasMedia {
                log("cache not reusable (trimmable=\(canTrimPromptCache(cache)) uniform=\(uniformOffsets)); rebuilding and re-prefilling \(fullTokens.count) tokens")
            }
            cache = makePromptCache(model: model, parameters: parameters)
        } else {
            // Never reuse the entire prompt: there has to be at least one token left to decode
            // the next logits from.
            let matched = min(
                Self.commonPrefixLength(fullTokens, cachedTokens),
                max(fullTokens.count - 1, 0))
            reused = min(matched, offsets[0])
            let surplus = offsets[0] - reused
            if surplus > 0, trimPromptCache(cache, numTokens: surplus) != surplus {
                log("cache trim came up short; rebuilding")
                cache = makePromptCache(model: model, parameters: parameters)
                reused = 0
            }
        }

        // Recorded BEFORE the prefill, not after: the cache now holds exactly fullTokens[0..<reused],
        // so the invariant "the cache is a prefix of cachedTokens, plus an untracked generated tail"
        // is true at every instant from here on. Recording it after the iterator instead would leave
        // a throw out of prefill with the cache half-filled and cachedTokens naming the OLD prompt -
        // a later pass could then match a prefix against slots holding different tokens.
        cachedTokens = fullTokens

        // With media the processor's own LMInput has to go through whole; otherwise feed only the
        // tokens the cache does not already cover.
        let passInput: LMInput =
            hasMedia ? input : LMInput(tokens: MLXArray(fullTokens[reused...].map(Int32.init)))

        let iterator = try TokenIterator(
            input: passInput, model: model, cache: cache, parameters: parameters)
        let (generations, genTask) = MLXLMCommon.generateTask(
            promptTokenCount: passInput.text.tokens.size,
            modelConfiguration: configuration,
            tokenizer: tokenizer,
            iterator: iterator,
            tools: toolSpecs)

        // The ANSWER, for the history: reasoning is streamed to the client but never recorded.
        // The chunk yielded below is still the RAW one - classifying the event stream here would
        // reorder thought against message downstream (see `historySplitter`).
        var answer = ""
        var calls: [ToolCall] = []
        for await generation in generations {
            if Task.isCancelled { break }
            switch generation {
            case .chunk(let chunk):
                for (kind, seg) in historySplitter.feed(chunk) where kind == .message {
                    answer += seg
                }
                continuation.yield(.text(chunk))
            case .toolCall(let call):
                calls.append(call)
                continuation.yield(.toolCall(call))
            case .info(let info):
                continuation.yield(.info(info))
            @unknown default:
                break
            }
        }
        // Explicit rather than relying on the stream iterator's deinit to fire onTermination: the
        // generate task keeps writing to the KVCache until it stops, and the next pass must not
        // begin against a moving cache. (The gate would hold it off anyway; this makes the stop
        // independent of iterator-lifetime timing.)
        genTask.cancel()
        await genTask.value

        // This pass's stream is over, so no further chunk can complete a marker that straddled a
        // chunk boundary: whatever the splitter still holds is final text. Same contract, and the
        // same reason, as Agent's end-of-pass flush.
        for (kind, seg) in historySplitter.flush() where kind == .message { answer += seg }

        // Recorded even on cancellation, deliberately: the partial answer is what the reader
        // already saw, so leaving it out would make the history disagree with the screen. A pass
        // that produced ONLY reasoning and no tool call now records nothing, which is correct and
        // is what OpenAIBackend already did - `reconcileHistory` handles the resulting two user
        // turns in a row.
        if !answer.isEmpty || !calls.isEmpty {
            messages.append(.assistant(answer, toolCalls: calls.isEmpty ? nil : calls))
            lastAssistantHasUnansweredCalls = !calls.isEmpty
        } else {
            // Logged rather than silent because of one shape this can leave behind: a pass that
            // ANSWERED tool results but produced only reasoning (a long think cut off by
            // maxTokens) records nothing, so the history goes ... assistant(calls), tool(result),
            // user(next) - a tool -> user adjacency with no assistant turn between. That is
            // exactly what OpenAIBackend has always handed llama-server, so it is the parity this
            // change buys rather than a defect; but Gemma-family templates have historically
            // raise_exception'd on role-alternation violations, and if a strict one ever trips,
            // this line is what points at it. The fix would then be to record `assistant("")`
            // here when the pass consumed tool results.
            log("pass recorded no assistant turn (no answer, no tool calls)"
                + (new.first?.role == .tool ? "; this pass answered tool results" : ""))
        }
    }

    private static func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
        var i = 0
        let limit = min(a.count, b.count)
        while i < limit, a[i] == b[i] { i += 1 }
        return i
    }
}

// MARK: - OpenAI-compatible backend (llama-server)

/// Cancellation handle for the in-flight URLSession task. The stream's `onTermination`
/// can fire BEFORE `bytes(for:)` has returned its task (cancel arriving between request
/// send and first byte), so a cancel that lands early is latched and applied to the task
/// the moment it is handed over - otherwise llama-server would keep generating into a
/// dropped connection.
final class URLTaskHandle: @unchecked Sendable {
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
/// State the MLX path keeps in its KV cache, this class keeps explicitly:
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
    /// to release its session rather than accumulate one per primed conversation. Canceling
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
    /// stream terminates, so the canceled task's unwinding races the next prompt. Flushing
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

    func stream(_ new: [Chat.Message]) -> AsyncThrowingStream<BackendEvent, Error> {
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
                    // A canceled URLSession task surfaces as URLError.cancelled; report it
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

    /// One model pass: POST the running conversation, parse the SSE frames into
    /// BackendEvent values, and append the assistant turn (text + any tool calls) to
    /// `messages` before returning, so the next pass carries it.
    private func runStream(
        continuation: AsyncThrowingStream<BackendEvent, Error>.Continuation, handle: URLTaskHandle
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

        // Accumulated assistant content. Reasoning is deliberately NOT part of it: with
        // reasoning_format "auto" the server hands reasoning over separately, and the
        // history we send back should carry the answer the model committed to, not its
        // scratchpad. (Keeping it in would re-feed a previous turn's <|channel|>analysis
        // back through the chat template as ordinary assistant content.)
        var text = ""
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
                // Reasoning the server split out for us (`reasoning_content`; some servers
                // spell it `reasoning`). Streamed to the client as thought, but kept OUT of
                // `text` - see the declaration of `text` above.
                let reasoning = (delta["reasoning_content"] as? String)
                    ?? (delta["reasoning"] as? String)
                if let reasoning, !reasoning.isEmpty {
                    if firstToken == nil { firstToken = Date() }
                    continuation.yield(.reasoning(reasoning))
                }
                if let chunk = delta["content"] as? String, !chunk.isEmpty {
                    if firstToken == nil { firstToken = Date() }
                    text += chunk
                    // Publish as we go: if this pass never reaches its commit below, the
                    // next one flushes this text instead of losing it.
                    lock.withLock { abandonedText = text }
                    continuation.yield(.text(chunk))
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
            // Let the server separate reasoning into `reasoning_content` (we forward it as
            // .reasoning). NOT "none": that leaves reasoning inline in content as RAW
            // markers, and the markers are per-family - gpt-oss emits harmony
            // (<|channel|>analysis<|message|>...<|end|>), which ThinkSplitter does not know
            // and would render verbatim in the answer. llama-server already has a parser per
            // family, so this is its job, not ours. Measured on llama-server 9553: tool_calls
            // are parsed identically under "none" and "auto" - only reasoning placement
            // differs. The MLX backend has no server to do this and still relies on
            // ThinkSplitter.
            "reasoning_format": "auto",
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
