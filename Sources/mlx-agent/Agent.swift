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
    func agentRequestPermission(toolCallId: String, title: String) async -> PermissionOutcome
    func agentTurnUsage(totalTokens: Int, tokensPerSecond: Double)
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

    init(backend: GenerationBackend, registry: MCPToolRegistry?, guardrails: AgentGuardrails) {
        self.backend = backend
        self.registry = registry
        self.guardrails = guardrails
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
        var dupCache: [String: String] = [:]
        var totalTokens = 0
        var genTokens = 0       // output tokens across this turn's model responses
        var genTime = 0.0       // decode time across them, for tokens/sec
        let splitter = ThinkSplitter()

        while true {
            if Task.isCancelled { return .cancelled }

            if iteration >= guardrails.maxToolIterations {
                for (kind, seg) in splitter.flush() { delegate?.agentEmitText(kind: kind, seg) }
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
                for try await gen in backend.stream(messages) {
                    if Task.isCancelled { return .cancelled }
                    if let chunk = gen.chunk {
                        for (kind, seg) in splitter.feed(chunk) {
                            delegate?.agentEmitText(kind: kind, seg)
                        }
                    } else if let call = gen.toolCall {
                        pendingCalls.append(call)
                    } else if let info = gen.info {
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
            messages = []  // consumed by the backend; next input is tool results (if any)

            if Task.isCancelled { return .cancelled }

            if pendingCalls.isEmpty {
                for (kind, seg) in splitter.flush() { delegate?.agentEmitText(kind: kind, seg) }
                delegate?.agentTurnUsage(
                    totalTokens: totalTokens,
                    tokensPerSecond: genTime > 0 ? Double(genTokens) / genTime : 0)
                return .stop(reason: "end_turn")
            }

            for call in pendingCalls {
                if Task.isCancelled { return .cancelled }
                switch await dispatch(call, dupCache: &dupCache) {
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

    private func dispatch(_ call: ToolCall, dupCache: inout [String: String]) async -> DispatchResult {
        let name = call.function.name
        let toolCallID = call.id ?? "tc-\(name)-\(dupCache.count)"
        let rawInput = prettyArguments(call)
        let dupKey = name + "\u{1}" + rawInput

        // Duplicate-call short-circuit: identical name+args already answered this turn.
        if let cached = dupCache[dupKey] {
            delegate?.agentToolCallStarted(
                id: toolCallID, title: name, kind: toolKind(name), rawInput: rawInput)
            delegate?.agentToolCallFinished(
                id: toolCallID, status: "completed",
                output: cached + "\n[duplicate call short-circuited]")
            return .result(cached)
        }

        guard let registry, let route = registry.route(name) else {
            let msg = "{\"error\": \"unknown tool: \(name)\"}"
            delegate?.agentToolCallStarted(
                id: toolCallID, title: name, kind: "other", rawInput: rawInput)
            delegate?.agentToolCallFinished(id: toolCallID, status: "failed", output: msg)
            return .result(msg)
        }

        delegate?.agentToolCallStarted(
            id: toolCallID, title: name, kind: toolKind(route.toolName), rawInput: rawInput)

        // Permission gate: only mutating (gated) tools require approval, and only before
        // the call is dispatched - there is no path to the server that skips this.
        if route.gated {
            let outcome =
                await delegate?.agentRequestPermission(
                    toolCallId: toolCallID, title: "Allow \(name)?") ?? .deny
            switch outcome {
            case .cancel:
                delegate?.agentToolCallFinished(
                    id: toolCallID, status: "failed", output: "[cancelled by user]")
                return .cancelled
            case .deny:
                let msg = "{\"error\": \"permission denied by user for \(name)\"}"
                delegate?.agentToolCallFinished(id: toolCallID, status: "failed", output: msg)
                return .result(msg)  // fed back so the model can adapt rather than hang
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
        if status == "completed" { dupCache[dupKey] = output }
        return .result(output)
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
/// cancelled and abandoned (it holds one pending MCP request until the server replies or
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
