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

    private let agentInstructions =
        "You are a helpful assistant. Use the provided tools when they are relevant."
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
        guardrails: AgentGuardrails = .init(), initialMode: String? = nil
    ) {
        self.currentModelDir = modelDir
        self.mcpConfigPath = mcpConfigPath
        self.guardrails = guardrails
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
                        "promptCapabilities": ["audio": false, "image": false, "embeddedContext": false]
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
            let session = ChatSession(
                container,
                instructions: agentInstructions,
                generateParameters: GenerateParameters(maxTokens: 4096, temperature: 0.7))
            // Build the MCP tool registry once (spawns the server processes) if configured.
            let registry = await ensureRegistry()
            let backend = MLXBackend(session)
            let agent = Agent(backend: backend, registry: registry, guardrails: guardrails)
            agent.delegate = self
            let modeNow = lock.withLock { mode }
            agent.setToolsEnabled(modeNow == "agent")
            let sid = "mlx-session-1"
            lock.withLock {
                self.container = container
                self.session = session
                self.backend = backend
                self.agent = agent
                self.sessionID = sid
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
                // Rebuild the session/backend/agent on the new model, but KEEP the MCP
                // registry (servers are model-independent). Re-apply the current mode.
                let container = try await loadModel(target.dir)
                let session = ChatSession(
                    container,
                    instructions: agentInstructions,
                    generateParameters: GenerateParameters(maxTokens: 4096, temperature: 0.7))
                let registry = lock.withLock { self.registry }
                let backend = MLXBackend(session)
                let agent = Agent(backend: backend, registry: registry, guardrails: guardrails)
                agent.delegate = self
                let modeNow = lock.withLock { mode }
                agent.setToolsEnabled(modeNow == "agent")
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
