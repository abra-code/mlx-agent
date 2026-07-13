// ACPServer.swift - Agent Client Protocol server over stdio.
//
// Newline-delimited JSON-RPC 2.0, matching the framing OpenCode uses (and validated
// against the ActionUIChat ACP client). Serves both plain chat and agentic tool use;
// the methods it implements:
//
//   initialize                -> { protocolVersion: 1, ... }
//   session/new               -> { sessionId, configOptions: [ model, (mode when tools) ] }
//   session/prompt            -> streams session/update notifications, resolves { stopReason }
//   session/cancel  (notif)   -> cancels the in-flight turn
//   session/set_config_option -> switches the model, or toggles chat/agent mode
//   session/prime             -> REPLACES the session context with a supplied transcript
//                                (client-driven resume/fresh; empty messages = reset)
//
// In agent mode a prompt runs the tool loop (see Agent): it streams tool_call /
// tool_call_update / usage_update, and sends session/request_permission before any
// gated tool. Tools come from the MCP servers named in --mcp-config (see MCPClients).
//
// mlx-swift-lm streams raw text with <think></think> tags and no reasoning separation,
// so this server splits those markers into agent_thought_chunk vs agent_message_chunk
// for the client's folded-reasoning UX. One live ChatSession per process (KV cache
// persists across prompts, so a follow-up turn's prefill is cheaper).
//
// Everything on stdout is JSON-RPC; all logging goes to stderr.

import Foundation
import MLXLLM
import MLXLMCommon
import MLXHuggingFace

/// Splits a raw model stream into thought (inside <think>...</think>) and message
/// (everything else) segments, tolerating markers that straddle chunk boundaries.
final class ThinkSplitter {
    enum Kind { case thought, message }
    private var buffer = ""
    private var inThink = false
    private let open = "<think>"
    private let close = "</think>"
    private var keep: Int { max(open.count, close.count) - 1 }

    func feed(_ s: String) -> [(Kind, String)] {
        buffer += s
        var out: [(Kind, String)] = []
        while true {
            let marker = inThink ? close : open
            if let r = buffer.range(of: marker) {
                let before = String(buffer[buffer.startIndex..<r.lowerBound])
                if !before.isEmpty { out.append((inThink ? .thought : .message, before)) }
                buffer.removeSubrange(buffer.startIndex..<r.upperBound)
                inThink.toggle()
            } else {
                // Hold back a short tail that could be the prefix of a marker.
                if buffer.count > keep {
                    let end = buffer.index(buffer.endIndex, offsetBy: -keep)
                    let emit = String(buffer[buffer.startIndex..<end])
                    if !emit.isEmpty { out.append((inThink ? .thought : .message, emit)) }
                    buffer = String(buffer[end...])
                }
                break
            }
        }
        return out
    }

    func flush() -> [(Kind, String)] {
        guard !buffer.isEmpty else { return [] }
        let seg: (Kind, String) = (inThink ? .thought : .message, buffer)
        buffer = ""
        return [seg]
    }
}

final class ACPServer: @unchecked Sendable, AgentDelegate {

    // Resolved once from the CLI (see resolveSystemPrompt): nil means NO system message is
    // prepended - the mode a translator uses, where the instruction lives in the user turn.
    private let systemPrompt: String?
    private let gen: GenConfig
    private let lock = NSLock()

    private var currentModelDir: String
    private let mcpConfigPath: String?
    private let guardrails: AgentGuardrails
    private var mode: String

    private var container: ModelContainer?
    private var session: ChatSession?
    private var backend: MLXBackend?
    private var agent: Agent?
    private var registry: MCPToolRegistry?
    private var registryBuildTask: Task<MCPToolRegistry?, Never>?
    private var signalSources: [DispatchSourceSignal] = []
    private var sessionID: String?
    private var promptTask: Task<Void, Never>?
    private var stdinBuffer = Data()
    private var doneContinuation: CheckedContinuation<Void, Never>?
    private let writeLock = NSLock()

    // Outbound request correlation (agent -> client, e.g. session/request_permission).
    private var outboundCounter = 1000
    private var pendingPermissions: [Int: CheckedContinuation<PermissionOutcome, Never>] = [:]

    init(
        modelDir: String, mcpConfigPath: String? = nil,
        guardrails: AgentGuardrails = .init(), initialMode: String? = nil,
        systemPrompt: String? = defaultSystemPrompt, gen: GenConfig = .init()
    ) {
        self.currentModelDir = modelDir
        self.mcpConfigPath = mcpConfigPath
        self.guardrails = guardrails
        self.systemPrompt = systemPrompt
        self.gen = gen
        // Default to agent mode when tools are configured, else chat.
        self.mode = initialMode ?? (mcpConfigPath != nil ? "agent" : "chat")
    }

    // MARK: - Run loop

    func serve() async {
        log("ACP server ready (stdio JSON-RPC). initial model: \(currentModelDir)")
        installSignalHandlers()
        let stdin = FileHandle.standardInput
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            lock.withLock { doneContinuation = cont }
            stdin.readabilityHandler = { [weak self] handle in
                guard let self else {
                    handle.readabilityHandler = nil
                    return
                }
                let data = handle.availableData
                if data.isEmpty {  // stdin EOF: the client went away
                    handle.readabilityHandler = nil
                    self.finish()
                    return
                }
                let lines = self.lock.withLock {
                    Self.splitLines(buffer: &self.stdinBuffer, appending: data)
                }
                for line in lines { self.dispatch(line) }
            }
        }
    }

    private func finish() {
        // Terminate any MCP server children synchronously so they do not orphan to
        // launchd when this process goes away (SIGTERM is immediate; disconnect is not).
        failPendingPermissions(with: .cancel)
        let reg = lock.withLock { () -> MCPToolRegistry? in
            let r = registry
            registry = nil
            return r
        }
        reg?.terminateAll()
        let cont = lock.withLock { () -> CheckedContinuation<Void, Never>? in
            let c = doneContinuation
            doneContinuation = nil
            return c
        }
        cont?.resume()
    }

    private static func splitLines(buffer: inout Data, appending data: Data) -> [String] {
        buffer.append(data)
        var lines: [String] = []
        while let nl = buffer.firstIndex(of: 0x0A) {
            let lineData = buffer[buffer.startIndex..<nl]
            lines.append(String(decoding: lineData, as: UTF8.self))
            buffer.removeSubrange(buffer.startIndex...nl)
        }
        return lines
    }

    // MARK: - Dispatch

    private func dispatch(_ line: String) {
        guard !line.isEmpty else { return }
        guard let data = line.data(using: .utf8),
            let msg = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else {
            log("dropping non-JSON line: \(line.prefix(160))")
            return
        }
        guard let method = msg["method"] as? String else {
            // No method: this is a RESPONSE to one of our outbound requests
            // (session/request_permission). Route it by id to the waiting continuation.
            if let rid = (msg["id"] as? NSNumber)?.intValue {
                handlePermissionResponse(id: rid, msg: msg)
            } else {
                log("message without method or id; dropping")
            }
            return
        }
        let params = msg["params"] as? [String: Any] ?? [:]
        // JSON-RPC ids: the ActionUIChat client (ACPConnection) always uses integer
        // ids. Keep the id as a Sendable Int? so it can cross into prompt Tasks; a
        // missing/non-integer id reads as nil and is treated as a notification.
        let id = (msg["id"] as? NSNumber)?.intValue

        switch method {
        case "initialize":
            // We support only stable protocol v1. Per ACP, if we can't match the
            // client's requested version we answer with the latest we support (1).
            if let requested = (params["protocolVersion"] as? NSNumber)?.intValue, requested != 1 {
                log("client requested protocolVersion \(requested); negotiating to our latest supported (1)")
            }
            respond(
                id,
                [
                    "protocolVersion": 1,
                    "authMethods": [],
                    "agentInfo": ["name": "mlx-agent", "version": "0.1"],
                    "agentCapabilities": [
                        "promptCapabilities": ["audio": false, "image": false, "embeddedContext": false],
                        // Custom extension: this agent accepts session/prime (context replacement
                        // from a restored transcript). Clients that don't know the key ignore it;
                        // clients that do gate all priming wire traffic on it.
                        "sessionPrime": true,
                    ],
                ])
        case "session/new":
            Task { await self.handleNewSession(id: id) }
        case "session/prompt":
            handlePrompt(id: id, params: params)
        case "session/cancel":
            handleCancel(params: params)
        case "session/set_config_option":
            Task { await self.handleSetConfigOption(id: id, params: params) }
        case "session/prime":
            handleSessionPrime(id: id, params: params)
        default:
            if id != nil {
                respondError(id, -32601, "method not supported by mlx-agent: \(method)")
            } else {
                log("unhandled notification: \(method)")
            }
        }
    }

    // MARK: - Handlers

    private func handleNewSession(id: Int?) async {
        let dir = lock.withLock { currentModelDir }
        do {
            let container = try await loadModel(dir)
            // Build the MCP tool registry once (spawns the server processes) if configured,
            // BEFORE the session stack (buildSessionStack reads self.registry).
            let registry = await ensureRegistry()
            let (session, backend, agent) = buildSessionStack(container: container, history: [])
            let sid = "mlx-session-1"
            let modeNow = lock.withLock { () -> String in
                self.container = container
                self.session = session
                self.backend = backend
                self.agent = agent
                self.sessionID = sid
                return mode
            }
            let toolCount = agent.hasTools ? (registry?.toolSpecs.count ?? 0) : 0
            log(
                "session ready: \(sid) on \((dir as NSString).lastPathComponent) "
                    + "[mode=\(modeNow), tools=\(toolCount)]")
            respond(id, ["sessionId": sid, "configOptions": configOptionsJSON()])
        } catch {
            respondError(id, -32000, "session/new failed: \(error.localizedDescription)")
        }
    }

    /// Build the session stack on `container`, optionally seeded with prior history
    /// (session/prime). Keeps the MCP registry (servers are model- and context-independent)
    /// and re-applies the current mode. History must NOT contain a system message: the
    /// session prepends `.system(systemPrompt)` itself when `systemPrompt` is non-nil (a nil
    /// prompt prepends nothing - the translator path). The default acp parameters (0.7 /
    /// 4096) are overlaid with any --temperature/--top-p/--max-new-tokens/--seed/
    /// --repetition-penalty the CLI supplied.
    private func buildSessionStack(
        container: ModelContainer, history: [Chat.Message]
    ) -> (ChatSession, MLXBackend, Agent) {
        let parameters = gen.apply(to: GenerateParameters(maxTokens: 4096, temperature: 0.7))
        let session: ChatSession
        if history.isEmpty {
            session = ChatSession(
                container, instructions: systemPrompt, generateParameters: parameters)
        } else {
            session = ChatSession(
                container, instructions: systemPrompt, history: history,
                generateParameters: parameters)
        }
        let registry = lock.withLock { self.registry }
        let backend = MLXBackend(session)
        let agent = Agent(backend: backend, registry: registry, guardrails: guardrails)
        agent.delegate = self
        let modeNow = lock.withLock { mode }
        agent.setToolsEnabled(modeNow == "agent")
        return (session, backend, agent)
    }

    /// Build the MCP registry on first use (idempotent). Concurrent callers share ONE
    /// build task so two overlapping session/new calls cannot spawn the server processes
    /// twice (the loser's processes would be untracked and orphan). Returns nil when no
    /// config was provided or it failed to load; a partial launch (some servers down)
    /// still returns a usable registry with whatever tools came up.
    private func ensureRegistry() async -> MCPToolRegistry? {
        if let existing = lock.withLock({ registry }) { return existing }
        guard mcpConfigPath != nil else { return nil }
        let task: Task<MCPToolRegistry?, Never> = lock.withLock {
            if let inFlight = registryBuildTask { return inFlight }
            let created = Task { await self.buildRegistry() }
            registryBuildTask = created
            return created
        }
        let built = await task.value
        lock.withLock { if registry == nil { registry = built } }
        return built
    }

    private func buildRegistry() async -> MCPToolRegistry? {
        guard let path = mcpConfigPath else { return nil }
        let configs: [MCPServerConfig]
        do {
            configs = try MCPConfigLoader.load(path)
        } catch {
            log("MCP config load failed: \(error.localizedDescription)")
            return nil
        }
        return await MCPToolRegistry.build(configs)
    }

    /// Terminate MCP children on SIGTERM/SIGINT so they do not orphan when the client
    /// kills us rather than closing stdin (the pipe-EOF path goes through finish()).
    private func installSignalHandlers() {
        for sig in [SIGTERM, SIGINT] {
            signal(sig, SIG_IGN)  // disable default terminate so the source can fire
            let source = DispatchSource.makeSignalSource(signal: sig, queue: .global())
            source.setEventHandler { [weak self] in
                self?.log("received signal \(sig); terminating MCP servers and exiting")
                let reg = self?.lock.withLock { () -> MCPToolRegistry? in
                    let r = self?.registry
                    self?.registry = nil
                    return r
                }
                (reg ?? nil)?.terminateAll()
                exit(0)
            }
            source.resume()
            lock.withLock { signalSources.append(source) }
        }
    }

    private func handlePrompt(id: Int?, params: [String: Any]) {
        let (agent, ourSid, busy) = lock.withLock {
            (self.agent, self.sessionID, self.promptTask != nil)
        }
        guard let agent, let ourSid else {
            respondError(id, -32002, "no active session; call session/new first")
            return
        }
        // One turn at a time: the ChatSession/backend is not safe to drive concurrently
        // (see MLXBackend). Reject rather than corrupt the KV cache with an overlapping run.
        if busy {
            respondError(id, -32003, "a prompt is already in progress for this session")
            return
        }
        // ACP carries the target sessionId on every prompt; reject one that isn't ours
        // rather than silently answering for the wrong session.
        if let reqSid = params["sessionId"] as? String, reqSid != ourSid {
            respondError(id, -32602, "unknown session: \(reqSid)")
            return
        }
        let text = Self.promptText(params)
        // The agent drives the full turn (streaming + tool loop + permission gate) and
        // reports back through the AgentDelegate methods below. Task.isCancelled inside
        // runTurn is wired to session/cancel via this promptTask.
        let task = Task { [weak self] in
            guard let self else { return }
            defer { self.lock.withLock { if self.promptTask != nil { self.promptTask = nil } } }
            let outcome = await agent.runTurn([.user(text)])
            // A turn that ended between a gate and its answer leaves no permission parked,
            // but be defensive: resolve any stragglers so no continuation leaks.
            self.failPendingPermissions(with: .cancel)
            switch outcome {
            case .cancelled:
                self.log("turn resolved: stopReason=cancelled")
                self.respond(id, ["stopReason": "cancelled"])
            case .failed(let message):
                // Reply with a JSON-RPC error, not an out-of-spec stopReason. ACP's
                // StopReason enum has no "error" member.
                self.log("turn failed: \(message)")
                self.respondError(id, -32000, "generation failed: \(message)")
            case .stop(let reason):
                self.log("turn resolved: stopReason=\(reason)")
                self.respond(id, ["stopReason": reason])
            }
        }
        lock.withLock { promptTask = task }
    }

    private func handleCancel(params: [String: Any]) {
        let (task, ourSid) = lock.withLock { (promptTask, sessionID) }
        if let reqSid = params["sessionId"] as? String, let ourSid, reqSid != ourSid {
            log("cancel: ignoring - unknown session \(reqSid)")
            return
        }
        log(task == nil ? "cancel: no in-flight turn" : "cancel: cancelling in-flight turn")
        // Unblock any in-flight permission wait so the loop can observe cancellation.
        failPendingPermissions(with: .cancel)
        task?.cancel()
    }

    private func handleSetConfigOption(id: Int?, params: [String: Any]) async {
        guard let configId = params["configId"] as? String,
            let value = params["value"] as? String
        else {
            respondError(id, -32602, "set_config_option requires configId and value")
            return
        }
        switch configId {
        case "mode":
            guard value == "chat" || value == "agent" else {
                respondError(id, -32602, "unknown mode: \(value)")
                return
            }
            let agent = lock.withLock { () -> Agent? in
                self.mode = value
                return self.agent
            }
            agent?.setToolsEnabled(value == "agent")
            log("mode -> \(value)")
            respond(id, ["configOptions": configOptionsJSON()])

        case "model":
            guard let target = availableModels().first(where: { $0.value == value }) else {
                respondError(id, -32602, "unknown model: \(value)")
                return
            }
            do {
                // Rebuild the session/backend/agent on the new model (context resets, as a
                // KV cache cannot carry across models); registry and mode are re-applied
                // by buildSessionStack.
                let container = try await loadModel(target.dir)
                let (session, backend, agent) = buildSessionStack(container: container, history: [])
                lock.withLock {
                    self.container = container
                    self.session = session
                    self.backend = backend
                    self.agent = agent
                    self.currentModelDir = target.dir
                }
                log("switched model -> \(value)")
                respond(id, ["configOptions": configOptionsJSON()])
            } catch {
                respondError(id, -32000, "model switch failed: \(error.localizedDescription)")
            }

        default:
            respondError(id, -32601, "unknown config option: \(configId)")
        }
    }

    // MARK: - session/prime (context replacement from a restored transcript)

    /// Replaces the session's conversational context with the supplied messages (empty =
    /// fresh context). Cheap: reuses the loaded model container and the MCP registry; the
    /// KV prefill of the primed history happens lazily on the NEXT session/prompt (the
    /// engine builds [system, history..., user] in one pass), so the first resumed turn
    /// pays the prefill and its usage_update includes the history tokens.
    private func handleSessionPrime(id: Int?, params: [String: Any]) {
        let (container, ourSid) = lock.withLock { (self.container, self.sessionID) }
        guard let container, let ourSid else {
            respondError(id, -32002, "no active session; call session/new first")
            return
        }
        if let reqSid = params["sessionId"] as? String, reqSid != ourSid {
            respondError(id, -32602, "unknown session: \(reqSid)")
            return
        }
        guard let rawMessages = params["messages"] as? [[String: Any]] else {
            respondError(id, -32602, "session/prime requires messages (an array)")
            return
        }
        let history = Self.primeHistory(from: rawMessages) { [weak self] in self?.log($0) }
        let (session, backend, agent) = buildSessionStack(container: container, history: history)
        // Refuse to swap the stack under a running turn. Checked inside the same lock
        // that publishes the swap, so a prompt accepted before us keeps its stack and
        // we bail; the client's ordering contract (cancel + await resolution before
        // priming) makes a first -32003 a rare transient it retries once.
        let busy = lock.withLock { () -> Bool in
            if promptTask != nil { return true }
            self.session = session
            self.backend = backend
            self.agent = agent
            return false
        }
        if busy {
            respondError(id, -32003, "cannot prime while a prompt is in progress")
            return
        }
        var wireBytes = 0
        for m in rawMessages { wireBytes += (m["content"] as? String)?.utf8.count ?? 0 }
        log("session primed: \(history.count) messages (\(rawMessages.count) supplied, ~\(wireBytes) content bytes)")
        respond(id, ["primed": history.count])
    }

    /// Map wire messages ({role, content, toolCalls?, toolCallId?}) to Chat.Message
    /// history, enforcing template well-formedness so a malformed transcript can never
    /// surface as a chat-template failure at the NEXT prompt (prefill is lazy): a tool
    /// message must answer a tool-call id announced by the nearest preceding assistant
    /// message; orphans are dropped with a log, never fatal. No system role on the wire -
    /// the agent owns its system prompt.
    static func primeHistory(
        from raw: [[String: Any]], log: (String) -> Void
    ) -> [Chat.Message] {
        var out: [Chat.Message] = []
        var openToolCallIDs: Set<String> = []
        var announcementIndex: Int? = nil   // out-index of the most recent assistant WITH calls
        var announcementText = ""
        var announcementAnswered = false
        for (index, entry) in raw.enumerated() {
            let role = entry["role"] as? String ?? ""
            let content = entry["content"] as? String ?? ""
            switch role {
            case "user":
                guard !content.isEmpty else { continue }
                out.append(.user(content))
                openToolCallIDs = []
                announcementIndex = nil
            case "assistant":
                let calls = Self.parseToolCalls(entry["toolCalls"], log: log)
                if content.isEmpty && (calls?.isEmpty ?? true) { continue }
                out.append(.assistant(content, toolCalls: calls))
                openToolCallIDs = Set((calls ?? []).compactMap { $0.id })
                announcementIndex = openToolCallIDs.isEmpty ? nil : out.count - 1
                announcementText = content
                announcementAnswered = false
            case "tool":
                guard let callID = entry["toolCallId"] as? String, openToolCallIDs.contains(callID)
                else {
                    log("prime: dropping orphan tool message at index \(index) (no matching assistant tool call)")
                    continue
                }
                out.append(.tool(content, id: callID))
                announcementAnswered = true
            default:
                log("prime: skipping message with unknown role \"\(role)\" at index \(index)")
            }
        }
        // A TRAILING tool-call announcement with no results after it would leave the
        // template mid-tool-turn, and prefill is lazy - the failure would surface as a
        // confusing error on the NEXT prompt, not at prime time. Strip the dangling
        // calls (keeping the assistant text); drop the message when that leaves nothing.
        if let index = announcementIndex, index == out.count - 1, !announcementAnswered {
            log("prime: stripping a trailing unanswered tool-call announcement")
            out.removeLast()
            if !announcementText.isEmpty {
                out.append(.assistant(announcementText))
            }
        }
        return out
    }

    /// Parse the wire toolCalls array ([{id?, name, arguments?}]) into ToolCalls; entries
    /// without a name are dropped with a log. Returns nil when absent or empty so plain
    /// text turns carry no tool metadata.
    private static func parseToolCalls(_ raw: Any?, log: (String) -> Void) -> [ToolCall]? {
        guard let entries = raw as? [[String: Any]], !entries.isEmpty else { return nil }
        var calls: [ToolCall] = []
        for entry in entries {
            guard let name = entry["name"] as? String, !name.isEmpty else {
                log("prime: dropping tool call without a name")
                continue
            }
            // JSON round-trip into the Codable JSONValue argument type (the wire value is
            // plain JSON, so this cannot lose information; {} on any failure).
            let rawArguments = (entry["arguments"] as? [String: Any]) ?? [:]
            let arguments =
                (try? JSONSerialization.data(withJSONObject: rawArguments))
                .flatMap { try? JSONDecoder().decode([String: JSONValue].self, from: $0) } ?? [:]
            calls.append(
                ToolCall(function: .init(name: name, arguments: arguments), id: entry["id"] as? String))
        }
        return calls.isEmpty ? nil : calls
    }

    // MARK: - Permission round-trip (agent -> client request)

    /// Send session/request_permission and await the client's choice. Parked on a
    /// continuation keyed by request id; resumed by handlePermissionResponse (or by
    /// failPendingPermissions on cancel/teardown).
    func agentRequestPermission(toolCallId: String, title: String) async -> PermissionOutcome {
        let sid = lock.withLock { sessionID } ?? ""
        let requestID = lock.withLock { () -> Int in
            outboundCounter += 1
            return outboundCounter
        }
        return await withCheckedContinuation {
            (cont: CheckedContinuation<PermissionOutcome, Never>) in
            lock.withLock { pendingPermissions[requestID] = cont }
            send([
                "jsonrpc": "2.0",
                "id": requestID,
                "method": "session/request_permission",
                "params": [
                    "sessionId": sid,
                    "toolCall": ["toolCallId": toolCallId, "title": title],
                    "options": [
                        ["optionId": "allow", "name": "Allow", "kind": "allow_once"],
                        ["optionId": "reject", "name": "Reject", "kind": "reject_once"],
                    ],
                ],
            ])
        }
    }

    private func handlePermissionResponse(id: Int, msg: [String: Any]) {
        let cont = lock.withLock { () -> CheckedContinuation<PermissionOutcome, Never>? in
            let c = pendingPermissions[id]
            pendingPermissions[id] = nil
            return c
        }
        guard let cont else {
            log("response for unknown/expired request id \(id); dropping")
            return
        }
        // result: { outcome: { outcome: "selected", optionId } } | { outcome: "cancelled" }
        var outcome: PermissionOutcome = .cancel
        if let result = msg["result"] as? [String: Any],
            let inner = result["outcome"] as? [String: Any]
        {
            switch inner["outcome"] as? String {
            case "selected":
                outcome = (inner["optionId"] as? String) == "allow" ? .allow : .deny
            case "cancelled":
                outcome = .cancel
            default:
                outcome = .cancel
            }
        } else if msg["error"] != nil {
            // The client errored on the request: treat as a denial, not a crash.
            outcome = .deny
        }
        cont.resume(returning: outcome)
    }

    private func failPendingPermissions(with outcome: PermissionOutcome) {
        let continuations = lock.withLock {
            () -> [CheckedContinuation<PermissionOutcome, Never>] in
            let waiting = Array(pendingPermissions.values)
            pendingPermissions.removeAll()
            return waiting
        }
        for cont in continuations { cont.resume(returning: outcome) }
    }

    // MARK: - AgentDelegate (streaming)

    func agentEmitText(kind: ThinkSplitter.Kind, _ text: String) {
        guard let sid = lock.withLock({ sessionID }) else { return }
        sendUpdate(sid, kind, text)
    }

    func agentToolCallStarted(id: String, title: String, kind: String, rawInput: String) {
        guard let sid = lock.withLock({ sessionID }) else { return }
        send(
            sessionUpdateEnvelope(
                sid,
                [
                    "sessionUpdate": "tool_call",
                    "toolCallId": id,
                    "title": title,
                    "kind": kind,
                    "status": "pending",
                    "rawInput": rawInput,
                ]))
    }

    func agentToolCallProgress(id: String) {
        guard let sid = lock.withLock({ sessionID }) else { return }
        send(
            sessionUpdateEnvelope(
                sid,
                [
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": id,
                    "status": "in_progress",
                ]))
    }

    func agentToolCallFinished(id: String, status: String, output: String) {
        guard let sid = lock.withLock({ sessionID }) else { return }
        send(
            sessionUpdateEnvelope(
                sid,
                [
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": id,
                    "status": status,
                    "content": [
                        ["type": "content", "content": ["type": "text", "text": output]]
                    ],
                    "rawOutput": output,
                ]))
    }

    func agentTurnUsage(totalTokens: Int) {
        guard totalTokens > 0, let sid = lock.withLock({ sessionID }) else { return }
        send(sessionUpdateEnvelope(sid, ["sessionUpdate": "usage_update", "used": totalTokens]))
    }

    private func sessionUpdateEnvelope(_ sid: String, _ update: [String: Any]) -> [String: Any] {
        [
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": ["sessionId": sid, "update": update],
        ]
    }

    // MARK: - Model registry

    private func availableModels() -> [(value: String, dir: String)] {
        let current = lock.withLock { currentModelDir }
        let parent = (current as NSString).deletingLastPathComponent
        var out: [(String, String)] = []
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: parent)) ?? []
        for name in entries.sorted() {
            let dir = parent + "/" + name
            var isDir: ObjCBool = false
            let hasConfig = FileManager.default.fileExists(
                atPath: dir + "/config.json", isDirectory: &isDir)
            if hasConfig {
                out.append((name, dir))
            }
        }
        if !out.contains(where: { $0.1 == current }) {
            out.insert(((current as NSString).lastPathComponent, current), at: 0)
        }
        return out
    }

    private func configOptionsJSON() -> [[String: Any]] {
        let (current, modeNow, haveTools) = lock.withLock {
            (
                (currentModelDir as NSString).lastPathComponent, mode,
                (registry?.toolSpecs.isEmpty == false)
            )
        }
        let choices = availableModels().map {
            ["value": $0.value, "name": $0.value] as [String: Any]
        }
        var options: [[String: Any]] = [
            [
                "id": "model",
                "name": "Model",
                "category": "model",
                "currentValue": current,
                "type": "select",
                "options": choices,
            ]
        ]
        // Only surface the chat/agent toggle when there are tools to enable.
        if haveTools {
            options.append([
                "id": "mode",
                "name": "Mode",
                "category": "mode",
                "currentValue": modeNow,
                "type": "select",
                "options": [
                    ["value": "chat", "name": "Chat"],
                    ["value": "agent", "name": "Agent"],
                ],
            ])
        }
        return options
    }

    private static func promptText(_ params: [String: Any]) -> String {
        let blocks = params["prompt"] as? [[String: Any]] ?? []
        return
            blocks
            .compactMap { ($0["type"] as? String) == "text" ? $0["text"] as? String : nil }
            .joined(separator: "\n")
    }

    // MARK: - Outbound

    private func sendUpdate(_ sid: String, _ kind: ThinkSplitter.Kind, _ text: String) {
        let su = kind == .thought ? "agent_thought_chunk" : "agent_message_chunk"
        send([
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": [
                "sessionId": sid,
                "update": [
                    "sessionUpdate": su,
                    "content": ["type": "text", "text": text],
                ],
            ],
        ])
    }

    private func respond(_ id: Int?, _ result: [String: Any]) {
        guard let id else { return }  // a notification arrived where a request was expected
        send(["jsonrpc": "2.0", "id": id, "result": result])
    }

    private func respondError(_ id: Int?, _ code: Int, _ message: String) {
        guard let id else {
            log("error for a notification (\(message)); nothing to reply to")
            return
        }
        send(["jsonrpc": "2.0", "id": id, "error": ["code": code, "message": message]])
    }

    private func send(_ message: [String: Any]) {
        guard
            var data = try? JSONSerialization.data(
                withJSONObject: message, options: [.withoutEscapingSlashes])
        else {
            log("failed to encode outbound message")
            return
        }
        data.append(0x0A)
        writeLock.withLock {
            FileHandle.standardOutput.write(data)
        }
    }

    private func log(_ s: String) {
        FileHandle.standardError.write(Data("[mlx-agent acp] \(s)\n".utf8))
    }
}
