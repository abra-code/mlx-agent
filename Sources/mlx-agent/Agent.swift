// Agent.swift - the tool loop with guardrails + permission gating.
//
// The heart of the agentic program. Drives a multi-pass tool loop against a
// GenerationBackend (backend-agnostic, so an alternative backend reuses it unchanged):
//
//   pass: stream a model turn -> collect tool calls (surfaced as .toolCall) -> for each,
//   gate (permission round-trip if the tool is mutating) -> call the MCP server -> feed
//   the result back -> repeat until the model stops calling tools.
//
// Guardrails the library's internal loop does NOT provide, enforced here:
//   - max tool iterations per prompt (stops a model that loops forever)
//   - per-tool-call timeout
//   - result truncation to a byte budget (bounds memory on huge tool output)
//   - duplicate-call short-circuit (same name+args in one turn returns the cached result)
//
// Streaming and the permission round-trip go through AgentDelegate so the same loop
// serves both the ACP server (streams session/update, asks the client) and --oneshot
// (prints to stdout, auto-answers).

import Foundation
import AgentText
import MLXLMCommon

// MARK: - Guardrails

struct AgentGuardrails: Sendable {
    var maxToolIterations: Int = 10
    var toolTimeout: TimeInterval = 60
    var resultByteBudget: Int = 32_768
}

// MARK: - Delegate

enum PermissionOutcome: Sendable { case allow, deny, cancel }

/// How the loop talks to the outside world. Implemented by ACPServer (wire updates +
/// client permission request) and by the oneshot console runner.
protocol AgentDelegate: AnyObject, Sendable {
    func agentEmitText(kind: ThinkSplitter.Kind, _ text: String)
    func agentToolCallStarted(id: String, title: String, kind: String, rawInput: String)
    func agentToolCallProgress(id: String)
    func agentToolCallFinished(id: String, status: String, output: String)
    /// Gated-tool approval. Returns .allow to run, .deny to reject (result fed back to
    /// the model), .cancel to abort the whole turn.
    ///
    /// `toolName` is the exposed tool name - the registry's unique key. It is passed separately
    /// from `title` because an implementation may key a standing decision on it (ACPServer does);
    /// `title` is display text and must not be parsed for identity.
    func agentRequestPermission(toolCallId: String, toolName: String, title: String) async
        -> PermissionOutcome
    func agentTurnUsage(totalTokens: Int, tokensPerSecond: Double)
}

// MARK: - Backends that dispatch tools themselves

/// What a bridged tool call produced. The backend-facing counterpart of the loop's own
/// `DispatchResult`, with one extra case the loop does not need.
enum ToolRunOutcome: Sendable {
    case result(String)
    /// The user answered a permission prompt with Cancel: abandon the whole turn.
    case cancelled
    /// The per-pass tool budget is gone. Carries text to hand the model instead of a result,
    /// because a backend that dispatches internally cannot be stopped mid-answer - see
    /// `InternalToolDispatchingBackend`.
    case budgetExhausted(String)
}

/// Implemented by a backend whose model library calls tools ITSELF, so the agent's loop never
/// sees a tool call and cannot mediate one.
///
/// Foundation Models is the only such backend today: `Tool.call` is invoked by the framework
/// from inside `respond`/`streamResponse`, and the result is folded into the answer before the
/// stream continues. There is no mode that surfaces the call for external dispatch. So instead
/// of the loop pulling calls out, the backend is handed a runner that reaches back INTO the
/// loop's guardrails - permission gate, timeout, truncation, dedup - and the interception point
/// moves from `runTurn` to inside the pass.
///
/// MLXBackend and OpenAIBackend do not conform, and `GenerationBackend` is untouched: this is
/// purely additive, so the ordinary path is exactly as it was.
protocol InternalToolDispatchingBackend: AnyObject {
    /// - Parameter maxCallsPerPass: the iteration cap, enforced by the backend because only it
    ///   knows where a pass begins. It is `AgentGuardrails.maxToolIterations`, passed rather than
    ///   duplicated so the number cannot drift - but note that the UNIT differs by necessity. On
    ///   the loop path it bounds ROUNDS, and a round can carry several calls; here there are no
    ///   rounds to count, so it bounds calls. The same setting is therefore stricter on this
    ///   backend, which suits a 4096-token window.
    func installToolRunner(
        maxCallsPerPass: Int,
        _ runner: @escaping @Sendable (ToolCall) async -> ToolRunOutcome)
}

// MARK: - Turn outcome

enum TurnOutcome: Sendable {
    case stop(reason: String)  // ACP StopReason: end_turn / max_turn_requests
    case cancelled
    case failed(String)
}

// MARK: - Agent

final class Agent: @unchecked Sendable {
    private let backend: GenerationBackend
    private let registry: MCPToolRegistry?
    private let guardrails: AgentGuardrails
    weak var delegate: AgentDelegate?

    /// Duplicate-call short-circuit, scoped to one turn.
    ///
    /// It used to be a `runTurn` local, which was both simpler and enough - the loop dispatched
    /// every call itself, in order. It is Agent state now because on an
    /// `InternalToolDispatchingBackend` the calls arrive from inside a pass, where `runTurn` has
    /// no frame to hold them in. The lifetime is unchanged (cleared at the top of every turn).
    ///
    /// Moving it out of the frame is what makes the rest of this necessary. Foundation Models
    /// enters tool calls CONCURRENTLY - measured, three at once - so the cache is now written by
    /// several calls at a time, and `withTimeout` cannot be cancelled, so a call can outlive the
    /// turn that started it. `inFlight` keeps identical concurrent calls down to one trip to the
    /// server; `turnEpoch` keeps a straggler from writing into the turn that followed it.
    private let dupLock = NSLock()
    private var dupCache: [String: String] = [:]
    private var inFlight: Set<String> = []
    private var dupWaiters: [String: [CheckedContinuation<DispatchResult, Never>]] = [:]
    /// The last outcome per in-flight key, success or failure, for a waiter that registers just
    /// after the owner released. Keyed like `inFlight` (turn-qualified) and cleared per turn.
    private var lastOutcomes: [String: DispatchResult] = [:]
    private var turnEpoch = 0

    init(backend: GenerationBackend, registry: MCPToolRegistry?, guardrails: AgentGuardrails) {
        self.backend = backend
        self.registry = registry
        self.guardrails = guardrails

        // Weak on purpose, and load-bearing: the agent owns the backend, so a strong capture
        // here would be a retain cycle that keeps both alive for the life of the process.
        (backend as? InternalToolDispatchingBackend)?
            .installToolRunner(maxCallsPerPass: guardrails.maxToolIterations) { [weak self] call in
                guard let self else { return .result("{\"error\": \"agent went away\"}") }
                return await self.runToolForBackend(call)
            }
    }

    /// Enable/disable tool calling for the current mode.
    func setToolsEnabled(_ enabled: Bool) {
        backend.tools = (enabled ? registry?.toolSpecs : nil)
    }

    var hasTools: Bool { backend.tools?.isEmpty == false }

    /// Run one agent turn. Must be called from within a cancellable Task (the caller's
    /// prompt Task); the loop honors `Task.isCancelled` between and within passes.
    func runTurn(_ input: [Chat.Message]) async -> TurnOutcome {
        var messages = input
        var iteration = 0
        // A new turn, and a new epoch: anything still in flight from the last one keeps running
        // (nothing can stop it) but can no longer write into this turn's cache.
        dupLock.withLock {
            dupCache.removeAll()
            // `inFlight` and `dupWaiters` are deliberately NOT cleared: a straggler still owns
            // its key and must be able to release it. They are turn-qualified instead, so this
            // turn cannot collide with what is left of the last one. `lastOutcomes` is bounded
            // the same way, and dropped here so it cannot grow for the life of the session.
            lastOutcomes.removeAll()
            turnEpoch += 1
        }
        var totalTokens = 0
        var genTokens = 0       // output tokens across this turn's model responses
        var genTime = 0.0       // decode time across them, for tokens/sec
        let splitter = ThinkSplitter()

        while true {
            if Task.isCancelled { return .cancelled }

            if iteration >= guardrails.maxToolIterations {
                delegate?.agentEmitText(
                    kind: .message,
                    "\n[reached the tool-call limit of \(guardrails.maxToolIterations); stopping]\n")
                delegate?.agentTurnUsage(
                    totalTokens: totalTokens,
                    tokensPerSecond: genTime > 0 ? Double(genTokens) / genTime : 0)
                return .stop(reason: "max_turn_requests")
            }
            iteration += 1

            var pendingCalls: [ToolCall] = []
            do {
                for try await event in backend.stream(messages) {
                    if Task.isCancelled { return .cancelled }
                    switch event {
                    case .text(let chunk):
                        for (kind, seg) in splitter.feed(chunk) {
                            delegate?.agentEmitText(kind: kind, seg)
                        }
                    case .reasoning(let text):
                        // Already classified by the server: straight to thought, never
                        // through the splitter (see BackendEvent).
                        delegate?.agentEmitText(kind: .thought, text)
                    case .toolCall(let call):
                        pendingCalls.append(call)
                    case .info(let info):
                        totalTokens += info.promptTokenCount + info.generationTokenCount
                        genTokens += info.generationTokenCount
                        genTime += info.generateTime
                    }
                }
            } catch is CancellationError {
                return .cancelled
            } catch {
                return .failed(error.localizedDescription)
            }

            // This pass's stream is over, so no further chunk can complete a marker that
            // straddled a chunk boundary: whatever the splitter is still holding is final
            // text and must go out NOW. Doing this only at end-of-turn (the old behavior)
            // stranded the tail whenever the pass ended in a tool call, and it resurfaced
            // glued to the front of the next pass's answer. `inThink` survives on purpose -
            // a <think> block may span a tool call.
            for (kind, seg) in splitter.flush() { delegate?.agentEmitText(kind: kind, seg) }

            messages = []  // consumed by the backend; next input is tool results (if any)

            if Task.isCancelled { return .cancelled }

            if pendingCalls.isEmpty {
                delegate?.agentTurnUsage(
                    totalTokens: totalTokens,
                    tokensPerSecond: genTime > 0 ? Double(genTokens) / genTime : 0)
                return .stop(reason: "end_turn")
            }

            for call in pendingCalls {
                if Task.isCancelled { return .cancelled }
                switch await dispatch(call) {
                case .cancelled:
                    return .cancelled
                case .result(let text):
                    messages.append(.tool(text, id: call.id))
                }
            }
        }
    }

    // MARK: - Tool dispatch

    private enum DispatchResult {
        case result(String)
        case cancelled
    }

    /// The entry point for an `InternalToolDispatchingBackend`: the same guardrails the loop
    /// applies, reachable from inside a pass.
    ///
    /// Deliberately thin. Everything that could drift between the two paths - the gate, the
    /// timeout, the byte budget, the dedup, every delegate notification - stays in `dispatch`,
    /// so a client cannot tell which backend answered by watching the wire.
    ///
    /// Two things it does that `runTurn` does for its own path. It checks cancellation, which the
    /// loop does before every dispatch and nothing else would do here. And it stamps the turn, so
    /// a call still in flight when its turn ends cannot write into the next one - measured: the
    /// framework enters tool calls CONCURRENTLY (three at once for a three-city prompt), and
    /// `withTimeout` is deliberately uncancellable, so a straggler outliving its turn is ordinary
    /// rather than exotic.
    func runToolForBackend(_ call: ToolCall) async -> ToolRunOutcome {
        if Task.isCancelled { return .cancelled }
        switch await dispatch(call, epoch: dupLock.withLock { turnEpoch }) {
        case .cancelled: return .cancelled
        case .result(let text): return .result(text)
        }
    }

    private func dispatch(_ call: ToolCall, epoch: Int? = nil) async -> DispatchResult {
        let name = call.function.name
        let rawInput = prettyArguments(call)
        let dupKey = name + "\u{1}" + rawInput

        // Read the cache and claim the key in ONE acquisition. Splitting them lets two identical
        // concurrent calls both miss and both run the tool, which is the guardrail failing in the
        // exact case it exists for. Sequential backends never take the `.joining` branch - there
        // is only ever one call in flight - so this costs them nothing.
        //
        // The IN-FLIGHT key carries the turn; the cache key does not. Without that, coalescing
        // would reopen the cross-turn channel the epoch closes: a straggler from a cancelled turn
        // holds its key for up to the tool timeout, so the next turn's identical call would wait
        // on a dead turn's work and adopt its result - or its CANCELLATION, silently cancelling a
        // turn because of a permission answer given in the previous one. A new turn now never
        // joins an old owner; it starts its own.
        let claimKey = "\(epoch ?? dupLock.withLock { turnEpoch })\u{1}\(dupKey)"
        let claim: DupClaim = dupLock.withLock {
            let seen = dupCache.count
            if let cached = dupCache[dupKey] { return .cached(cached, seen) }
            if inFlight.contains(claimKey) { return .joining(seen) }
            inFlight.insert(claimKey)
            // A new owner supersedes the last one's recorded outcome, so this holds at most one
            // entry per distinct call in a turn rather than accumulating results - and a tool
            // result can be up to `resultByteBudget`, which is 32 KB by default.
            lastOutcomes.removeValue(forKey: claimKey)
            return .owner(seen)
        }
        let toolCallID = call.id ?? "tc-\(name)-\(claim.seen)"

        func shortCircuit(_ text: String) -> DispatchResult {
            delegate?.agentToolCallStarted(
                id: toolCallID, title: name, kind: toolKind(name), rawInput: rawInput)
            delegate?.agentToolCallFinished(
                id: toolCallID, status: "completed",
                output: text + "\n[duplicate call short-circuited]")
            return .result(text)
        }

        switch claim {
        // Identical name+args already answered this turn.
        case .cached(let text, _):
            return shortCircuit(text)
        // Identical name+args running RIGHT NOW. Wait for it instead of calling the server a
        // second time, and report whatever it got - including a cancellation, which must not
        // become a result just because this call arrived second.
        case .joining:
            switch await joinInFlight(claimKey) {
            case .cancelled:
                // Say so on the wire. Every other dispatch path leaves a tool_call and a
                // tool_call_update behind, and a coalesced call that inherits a cancellation is
                // the one place a client would otherwise see nothing at all.
                delegate?.agentToolCallStarted(
                    id: toolCallID, title: name, kind: toolKind(name), rawInput: rawInput)
                delegate?.agentToolCallFinished(
                    id: toolCallID, status: "failed", output: "[canceled by user]")
                return .cancelled
            case .result(let text): return shortCircuit(text)
            }
        case .owner:
            break
        }

        // Past here this call owns the key and must release it on every exit.
        var outcome = DispatchResult.result("{\"error\": \"tool dispatch produced no result\"}")
        defer { releaseInFlight(claimKey, outcome: outcome) }

        guard let registry, let route = registry.route(name) else {
            let msg = "{\"error\": \"unknown tool: \(name)\"}"
            delegate?.agentToolCallStarted(
                id: toolCallID, title: name, kind: "other", rawInput: rawInput)
            delegate?.agentToolCallFinished(id: toolCallID, status: "failed", output: msg)
            outcome = .result(msg)
            return outcome
        }

        delegate?.agentToolCallStarted(
            id: toolCallID, title: name, kind: toolKind(route.toolName), rawInput: rawInput)

        // Permission gate: only mutating (gated) tools require approval, and only before
        // the call is dispatched - there is no path to the server that skips this.
        if route.gated {
            let decision =
                await delegate?.agentRequestPermission(
                    toolCallId: toolCallID, toolName: name, title: "Allow \(name)?") ?? .deny
            switch decision {
            case .cancel:
                delegate?.agentToolCallFinished(
                    id: toolCallID, status: "failed", output: "[canceled by user]")
                outcome = .cancelled
                return outcome
            case .deny:
                let msg = "{\"error\": \"permission denied by user for \(name)\"}"
                delegate?.agentToolCallFinished(id: toolCallID, status: "failed", output: msg)
                outcome = .result(msg)  // fed back so the model can adapt rather than hang
                return outcome
            case .allow:
                break
            }
        }

        delegate?.agentToolCallProgress(id: toolCallID)

        let output: String
        let status: String
        do {
            let (content, isError) = try await withTimeout(guardrails.toolTimeout) {
                try await route.server.client.callTool(
                    name: route.toolName, arguments: mcpArguments(call))
            }
            output = truncateToBudget(joinToolContent(content), guardrails.resultByteBudget)
            status = (isError ?? false) ? "failed" : "completed"
        } catch is TimeoutError {
            output = "{\"error\": \"tool timed out after \(Int(guardrails.toolTimeout))s\"}"
            status = "failed"
        } catch {
            output = "{\"error\": \"tool call failed: \(error.localizedDescription)\"}"
            status = "failed"
        }

        delegate?.agentToolCallFinished(id: toolCallID, status: status, output: output)
        if status == "completed" {
            dupLock.withLock {
                // Only into the turn this call belongs to. A tool that outlives its turn -
                // routine here, since the framework dispatches concurrently and `withTimeout` is
                // uncancellable - would otherwise seed the NEXT conversation's cache with a
                // result from a turn the user cancelled, and the client would be told the answer
                // was a duplicate of something it never saw.
                if epoch == nil || epoch == turnEpoch { dupCache[dupKey] = output }
            }
        }
        outcome = .result(output)
        return outcome
    }

    // MARK: - In-flight coalescing

    /// What claiming a dup key found. `seen` is only for naming a tool call that arrived
    /// without an id, and preserves the previous numbering.
    private enum DupClaim {
        case cached(String, Int)
        case joining(Int)
        case owner(Int)

        var seen: Int {
            switch self {
            case .cached(_, let n), .joining(let n), .owner(let n): return n
            }
        }
    }

    /// Park until the call that owns this key finishes, then take its result.
    private func joinInFlight(_ key: String) async -> DispatchResult {
        await withCheckedContinuation { (continuation: CheckedContinuation<DispatchResult, Never>) in
            // The lost-race branch below resumes INSIDE the lock, unlike `releaseInFlight`. It is
            // safe for a reason that does not apply there: this continuation belongs to the
            // calling task and has not suspended yet, so resuming it only records a value - it
            // cannot run anything that comes back for this lock.
            dupLock.withLock {
                // Re-check inside the lock: the owner can finish between the claim above and
                // this registration, and a waiter added after the release would never be woken.
                guard inFlight.contains(key) else {
                    // Lost the race: the owner released between the claim and here. Its outcome
                    // is recorded whether it succeeded or FAILED, so a waiter that arrives
                    // microseconds late gets the same answer as one that arrived in time -
                    // reading `dupCache` instead would hand it a synthetic error whenever the
                    // tool had failed, since only successes are cached.
                    continuation.resume(
                        returning: lastOutcomes[key]
                            ?? .result("{\"error\": \"duplicate tool call produced no result\"}"))
                    return
                }
                dupWaiters[key, default: []].append(continuation)
            }
        }
    }

    /// Release the key and hand every waiter the outcome.
    private func releaseInFlight(_ key: String, outcome: DispatchResult) {
        let waiting: [CheckedContinuation<DispatchResult, Never>] = dupLock.withLock {
            inFlight.remove(key)
            lastOutcomes[key] = outcome
            return dupWaiters.removeValue(forKey: key) ?? []
        }
        // Outside the lock. Resuming is cheap, but a continuation resumes its task, and holding
        // a lock across that invites the next acquisition from the woken side.
        for continuation in waiting { continuation.resume(returning: outcome) }
    }
}

// MARK: - Helpers

/// Map a tool name to an ACP ToolCallModel.Kind rawValue for nicer client rendering.
func toolKind(_ name: String) -> String {
    let n = name.lowercased()
    if n.contains("delete") || n.contains("remove") { return "delete" }
    if n.contains("move") || n.contains("rename") { return "move" }
    if n.contains("write") || n.contains("edit") || n.contains("create") || n.contains("append") {
        return "edit"
    }
    if n.contains("exec") || n.contains("command") || n.contains("shell") || n.contains("run") {
        return "execute"
    }
    if n.contains("search") || n.contains("grep") || n.contains("glob") || n.contains("find") {
        return "search"
    }
    if n.contains("fetch") || n.contains("http") || n.contains("web") { return "fetch" }
    if n.contains("read") || n.contains("list") || n.contains("cat") { return "read" }
    return "other"
}

/// Pretty-print a tool call's arguments as stable JSON for the `rawInput` panel.
func prettyArguments(_ call: ToolCall) -> String {
    guard let data = try? JSONEncoder().encode(call.function.arguments) else { return "{}" }
    if let obj = try? JSONSerialization.jsonObject(with: data),
        let pretty = try? JSONSerialization.data(
            withJSONObject: obj, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]),
        let s = String(data: pretty, encoding: .utf8)
    {
        return s
    }
    return String(decoding: data, as: UTF8.self)
}

/// Truncate a string to a UTF-8 byte budget on a codepoint boundary, appending a marker
/// noting how much was dropped. Bounds memory when a tool returns a huge blob.
func truncateToBudget(_ s: String, _ maxBytes: Int) -> String {
    let bytes = Array(s.utf8)
    guard bytes.count > maxBytes, maxBytes > 0 else { return s }
    var end = maxBytes
    // Back up off any UTF-8 continuation byte so we cut on a character boundary.
    while end > 0 && (bytes[end] & 0xC0) == 0x80 { end -= 1 }
    let head = String(decoding: bytes[0..<end], as: UTF8.self)
    return head + "\n[truncated \(bytes.count - end) of \(bytes.count) bytes]"
}

// MARK: - Timeout

struct TimeoutError: Error {}

/// One-shot latch so a continuation resumes exactly once.
private final class ResumeGuard: @unchecked Sendable {
    private let lock = NSLock()
    private var claimed = false
    func claim() -> Bool { lock.withLock { claimed ? false : { claimed = true; return true }() } }
}

/// Run `operation`, returning `TimeoutError` at `seconds` REGARDLESS of whether the
/// operation honors cancellation. This matters because the MCP SDK's `callTool` awaits a
/// detached request task that does not observe the caller's cancellation - a structured
/// task-group timeout would block at scope exit until the (possibly hung) server replied,
/// defeating the guardrail. Here the winner resumes the continuation and the loser is
/// canceled and abandoned (it holds one pending MCP request until the server replies or
/// the server process is torn down - bounded, one per timed-out call).
func withTimeout<T: Sendable>(
    _ seconds: TimeInterval, _ operation: @escaping @Sendable () async throws -> T
) async throws -> T {
    let guardOnce = ResumeGuard()
    return try await withCheckedThrowingContinuation { (cont: CheckedContinuation<T, Error>) in
        let opTask = Task {
            do {
                let value = try await operation()
                if guardOnce.claim() { cont.resume(returning: value) }
            } catch {
                if guardOnce.claim() { cont.resume(throwing: error) }
            }
        }
        Task {
            try? await Task.sleep(nanoseconds: UInt64(max(0, seconds) * 1_000_000_000))
            if guardOnce.claim() { cont.resume(throwing: TimeoutError()) }
            opTask.cancel()
        }
    }
}
