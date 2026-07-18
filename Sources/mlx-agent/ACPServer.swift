// ACPServer.swift - Agent Client Protocol server over stdio.
//
// Newline-delimited JSON-RPC 2.0, matching the framing OpenCode uses (and validated
// against the ActionUIChat ACP client). Serves both plain chat and agentic tool use;
// the methods it implements:
//
//   initialize                -> { protocolVersion: 1, ... }
//   session/new               -> { sessionId, configOptions: [] (always empty - see
//                                configOptionsJSON: neither model nor mode is advertised, so no
//                                client renders a picker; both remain settable over the wire)
//   session/prompt            -> streams session/update notifications, resolves { stopReason }
//   session/cancel  (notif)   -> cancels the in-flight turn
//   session/set_config_option -> switches the model; model and mode are both accepted but
//                                neither is advertised (the host owns both decisions)
//   session/prime             -> REPLACES the session context with a supplied transcript
//                                (client-driven resume/fresh; empty messages = reset)
//
// In agent mode a prompt runs the tool loop (see Agent): it streams tool_call /
// tool_call_update / usage_update, and sends session/request_permission before any
// gated tool. Tools come from the MCP servers named in --mcp-config (see MCPClients).
//
// Reasoning reaches the client as agent_thought_chunk rather than agent_message_chunk, so
// it folds away in the UI. The two backends separate it differently: the MLX path streams
// raw text with inline <think></think> tags that ThinkSplitter (see the AgentText target)
// classifies, while llama-server classifies it for us into `reasoning_content`. One live
// ChatSession per process (KV cache persists across prompts, so a follow-up turn's
// prefill is cheaper).
//
// Everything on stdout is JSON-RPC; all logging goes to stderr.

import Foundation
import AgentText
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace

/// Which engine generates. Chosen once at launch (`--backend`), never at runtime: the two
/// have different lifecycles - the MLX path loads a model container into this process and
/// can switch models on the fly, while the openai path talks to a server the APPLET owns
/// (it launches, restarts and reaps llama-server), so model choice is not ours to make.
enum EngineSpec: Sendable {
    case mlx(modelDir: String)
    case openai(baseURL: URL)
}

final class ACPServer: @unchecked Sendable, AgentDelegate {

    // Resolved once from the CLI (see resolveSystemPrompt): nil means NO system message is
    // prepended - the mode a translator uses, where the instruction lives in the user turn.
    private let systemPrompt: String?
    private let gen: GenConfig
    // Extra generation stop tokens unioned in at model-load time (see loadModel).
    private let extraEOSTokens: Set<String>
    private let lock = NSLock()

    /// openai: the endpoint root. nil selects the MLX path - the ONE branch condition, set
    /// in init and never mutated, so no lock is needed to read it.
    private let openaiBaseURL: URL?
    /// MLX path only; "" on the openai backend (the applet decides what llama-server serves).
    private var currentModelDir: String
    private let mcpConfigPath: String?
    private let guardrails: AgentGuardrails
    private var mode: String

    // MLX-only: the loaded weights and the session driving them.
    private var container: ModelContainer?
    private var session: ChatSession?
    private var backend: GenerationBackend?
    private var agent: Agent?
    private var registry: MCPToolRegistry?

    // MLX idle-unload: mirror llama-server's --sleep-idle-seconds on the in-process MLX path.
    // After `idleUnloadSeconds` with no prompt, the model weights are released and MLX's buffer
    // cache is returned to the OS; the next prompt reloads the model and re-prefills context from
    // `mlxTranscript`. Only meaningful on the MLX path - the openai path holds no weights here
    // (llama-server does its own idle sleep, and the OpenAIBackend keeps only a tiny messages
    // array). 0 disables. The timer lives on `idleQueue` and is only ever touched from it.
    private let idleUnloadSeconds: TimeInterval
    private let idleQueue = DispatchQueue(label: "com.abracode.mlx-agent.idle")
    private var idleTimer: DispatchSourceTimer?
    // True once the model has been idle-unloaded: the session is still "live" (sessionID set,
    // registry up) but `container`/`agent` are nil, so the next prompt reloads instead of being
    // rejected as "no active session".
    private var idleUnloaded = false
    // A shadow of the conversation as clean (user, assistant) text turns, kept ONLY so an
    // idle-unload can rebuild the ChatSession's context on reload. ChatSession's message history
    // is private and collapses into an opaque KV cache after the first turn, so it cannot be read
    // back - hence this parallel record, the same thing the OpenAIBackend keeps explicitly and
    // llama-server's own idle-sleep relies on the client resending. Text turns only: intra-turn
    // tool calls/results are not replayed, so after an idle reload the model keeps the
    // conversation but not the exact tool outputs (documented trade, appended only on a clean
    // .stop so a cancelled/failed turn never enters the replayed context). Reset to the primed
    // history on session/prime and to [] on session/new.
    private var mlxTranscript: [Chat.Message] = []
    // Accumulates THIS turn's assistant answer (kind == .message only, never .thought) so it can
    // be appended to `mlxTranscript` at turn end. Written from agentEmitText, read/reset under lock.
    private var assistantTurnBuffer = ""
    private var registryBuildTask: Task<MCPToolRegistry?, Never>?
    private var signalSources: [DispatchSourceSignal] = []
    private var sessionID: String?
    // The absolute working directory the client declared in session/new (ACP's `cwd`). The
    // client also launched us with the process cwd set to this, so relative paths already
    // resolve here; this copy exists ONLY so the model is TOLD where it is - a model cannot
    // call getcwd(), it knows the directory only if it is in the system prompt (see
    // effectiveSystemPrompt). Empty/"/" are treated as "unknown" and inject nothing. Set on
    // session/new and left untouched by session/prime (same process, same cwd).
    private var sessionCwd: String?
    private var promptTask: Task<Void, Never>?
    // Claimed under `lock` the instant a prompt is accepted, and held until `promptTask` is
    // committed (the Task exists only after its closure is built, so there is a gap between the
    // busy-check and `promptTask = task`). Without it, `idleFired` or `session/prime` could see
    // `promptTask == nil` in that gap and unload/swap the stack mid-turn. Both treat it as busy.
    private var promptStarting = false
    private var stdinBuffer = Data()
    private var doneContinuation: CheckedContinuation<Void, Never>?
    private let writeLock = NSLock()

    // Outbound request correlation (agent -> client, e.g. session/request_permission).
    private var outboundCounter = 1000
    /// The parked continuation, the tool it is about, and the session epoch it was asked under:
    /// handlePermissionResponse needs the tool name to record an "always" choice, and the
    /// response carries only the request id.
    /// Keeping the name here rather than parsing it back out of the title is deliberate - the
    /// title is display text and would silently stop matching the moment it is reworded.
    private struct PendingPermission {
        let cont: CheckedContinuation<PermissionOutcome, Never>
        let toolName: String
        let epoch: Int
    }
    private var pendingPermissions: [Int: PendingPermission] = [:]

    /// Bumped every time the session is replaced (session/new AND session/prime). A permission
    /// request parked under the old session can still be answered after the swap; without this
    /// its "always" choice would be recorded into the NEW session's grants, resurrecting consent
    /// the clear just dropped. `sessionID` cannot serve as the stamp - it is the same literal
    /// ("mlx-session-1") for every session, so it compares equal across a swap and would defend
    /// nothing.
    private var sessionEpoch = 0

    // Session-scoped "always" decisions, keyed by EXPOSED tool name (the registry's unique key -
    // see MCPClients.routes, which disambiguates collisions across servers). Cleared by
    // resetSessionPermissions() on EVERY session replacement - see there for why session/prime
    // is the one that actually matters. Deliberately NOT persisted and NOT path-scoped:
    //
    //  - Not path-scoped, because the sandbox already answers "where". replay's kernel sandbox
    //    plus its allowed-dir list is the spatial boundary; a second path system here would be a
    //    competing source of truth that the kernel would overrule, i.e. a UI that lies.
    //    This layer only answers "whether", which is why the tool name is the whole key.
    //  - Not persisted, because the mcp-config (and thus the sandbox) is generated per window and
    //    frozen at spawn, so a grant made here can never widen under the session that made it.
    //    A grant that outlived the window would inherit whatever sandbox a LATER window is given -
    //    the user would have consented to a narrow boundary and silently received a wider one.
    //    Session scope also needs no revoke UI: closing the window is the revoke.
    //
    private var alwaysAllowed: Set<String> = []
    private var alwaysRejected: Set<String> = []

    /// Drop every standing permission and invalidate any parked request. MUST be called with
    /// `lock` held, inside the same critical section that publishes the session swap, so no
    /// in-flight dispatch can read a grant belonging to the session that just ended.
    ///
    /// Called from session/new AND session/prime. **prime is the one that matters**: the
    /// shipping client (ChatView) sends session/new exactly once per transport spawn, in
    /// start(); New Chat and every sidebar switch go through session/prime, and `prime []` is
    /// the documented reset. Clearing only on session/new would make the clear dead code in
    /// production and quietly turn "session-scoped" into "process-scoped" - a grant made in one
    /// conversation would still be standing in the next.
    private func resetSessionPermissionsLocked() {
        alwaysAllowed.removeAll()
        alwaysRejected.removeAll()
        sessionEpoch &+= 1
    }

    init(
        engine: EngineSpec, mcpConfigPath: String? = nil,
        guardrails: AgentGuardrails = .init(), initialMode: String? = nil,
        systemPrompt: String? = defaultSystemPrompt, gen: GenConfig = .init(),
        extraEOSTokens: Set<String> = [], idleUnloadSeconds: TimeInterval = IdleUnload.defaultSeconds
    ) {
        self.idleUnloadSeconds = idleUnloadSeconds
        switch engine {
        case .mlx(let modelDir):
            self.currentModelDir = modelDir
            self.openaiBaseURL = nil
        case .openai(let baseURL):
            self.currentModelDir = ""
            self.openaiBaseURL = baseURL
        }
        self.mcpConfigPath = mcpConfigPath
        self.guardrails = guardrails
        self.systemPrompt = systemPrompt
        self.gen = gen
        self.extraEOSTokens = extraEOSTokens
        // Default to agent mode when tools are configured, else chat.
        self.mode = initialMode ?? (mcpConfigPath != nil ? "agent" : "chat")
    }

    // MARK: - Run loop

    func serve() async {
        if let openaiBaseURL {
            log("ACP server ready (stdio JSON-RPC). backend: openai at \(openaiBaseURL.absoluteString)")
        } else {
            log("ACP server ready (stdio JSON-RPC). initial model: \(currentModelDir)")
        }
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
        cancelIdleTimer()
        cont?.resume()
    }

    // MARK: - MLX idle-unload

    /// (Re)arm the idle-unload timer. No-op unless this is the MLX path, unloading is enabled,
    /// a model is actually loaded, and no prompt is running. Safe to call at every turn/session
    /// boundary. Must be called WITHOUT `lock` (it takes `lock` to sample state, then hops to
    /// `idleQueue` to touch the timer). A stale arm is harmless: `idleFired` re-checks under lock.
    private func rearmIdleTimerIfIdle() {
        guard openaiBaseURL == nil, idleUnloadSeconds > 0 else { return }
        let shouldArm = lock.withLock { self.container != nil && self.promptTask == nil }
        let interval = idleUnloadSeconds
        idleQueue.async { [weak self] in
            guard let self else { return }
            self.idleTimer?.cancel()
            self.idleTimer = nil
            guard shouldArm else { return }
            let timer = DispatchSource.makeTimerSource(queue: self.idleQueue)
            timer.schedule(deadline: .now() + interval)
            timer.setEventHandler { [weak self] in self?.idleFired() }
            self.idleTimer = timer
            timer.resume()
        }
    }

    /// Cancel any pending idle-unload (activity arrived, or we are shutting down).
    private func cancelIdleTimer() {
        idleQueue.async { [weak self] in
            self?.idleTimer?.cancel()
            self?.idleTimer = nil
        }
    }

    /// Fired on `idleQueue` after `idleUnloadSeconds` of no prompts. Releases the model weights
    /// (keeping the MCP registry and the shadow transcript) and returns MLX's buffer cache to the
    /// OS. A no-op if a prompt slipped in or the model is already unloaded - the promptTask guard
    /// is what guarantees we never unload mid-turn.
    private func idleFired() {
        // Release mechanics (snapshot ordering, clearCache, log format) are shared with the
        // other modes in IdleUnload.releaseModel; only the drop closure - what "clear every
        // strong reference" means for the ACP session stack, and when it must abort - is
        // this server's business.
        let released = IdleUnload.releaseModel(afterIdle: idleUnloadSeconds, log: { self.log($0) }) {
            lock.withLock {
                // promptStarting covers the gap before promptTask is committed: never unload
                // with a turn running OR about to run.
                guard self.promptTask == nil, !self.promptStarting, self.container != nil else {
                    return nil
                }
                self.container = nil
                self.session = nil
                self.backend = nil
                self.agent = nil
                self.idleUnloaded = true
                return (self.currentModelDir as NSString).lastPathComponent
            }
        }
        if released { idleTimer = nil }
    }

    /// Reload the model after an idle-unload and rebuild the session stack around the shadow
    /// transcript, so the conversation continues with its context intact. Mirrors the MLX branch
    /// of session/prime (same buildSessionStack, same registry, same mode); the re-prefill cost is
    /// paid here, on the first prompt after idle. Throws if the model fails to (re)load.
    private func reloadAfterIdle() async throws -> Agent {
        let (dir, history) = lock.withLock { (self.currentModelDir, self.mlxTranscript) }
        log(
            "idle-reload: loading \((dir as NSString).lastPathComponent), "
                + "replaying \(history.count) messages")
        let container = try await loadModel(dir, extraEOSTokens: extraEOSTokens)
        let (session, backend, agent) = buildSessionStack(container: container, history: history)
        lock.withLock {
            self.container = container
            self.session = session
            self.backend = backend
            self.agent = agent
            self.idleUnloaded = false
        }
        return agent
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
            Task { await self.handleNewSession(id: id, params: params) }
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

    private func handleNewSession(id: Int?, params: [String: Any]) async {
        // ACP requires `cwd` to be an absolute path; ignore anything else (empty, relative, or
        // a bare "/") so we never inject a misleading directory into the model's context.
        let cwd = (params["cwd"] as? String).flatMap { $0.hasPrefix("/") && $0 != "/" ? $0 : nil }
        lock.withLock { self.sessionCwd = cwd }
        if let openaiBaseURL {
            await handleNewSessionOpenAI(id: id, baseURL: openaiBaseURL)
        } else {
            await handleNewSessionMLX(id: id)
        }
    }

    /// openai: nothing to load - the applet already launched llama-server on a pinned port.
    /// Health-check it so a dead server is ONE clean error here rather than a failure on
    /// every prompt, then build the registry + stack exactly as the MLX path does.
    private func handleNewSessionOpenAI(id: Int?, baseURL: URL) async {
        do {
            try await OpenAIBackend.waitForHealth(baseURL: baseURL)
        } catch {
            respondError(id, -32000, "session/new failed: \(error.localizedDescription)")
            return
        }
        let registry = await ensureRegistry()
        let (backend, agent) = buildOpenAIStack(baseURL: baseURL, history: [])
        let sid = "mlx-session-1"
        let modeNow = lock.withLock { () -> String in
            self.backend = backend
            self.agent = agent
            self.sessionID = sid
            // A new conversation is a new consent context: standing permissions do not survive it.
            resetSessionPermissionsLocked()
            return mode
        }
        let toolCount = agent.hasTools ? (registry?.toolSpecs.count ?? 0) : 0
        log(
            "session ready: \(sid) on openai backend \(baseURL.absoluteString) "
                + "[mode=\(modeNow), tools=\(toolCount)]")
        respond(id, ["sessionId": sid, "configOptions": configOptionsJSON()])
    }

    private func handleNewSessionMLX(id: Int?) async {
        let dir = lock.withLock { currentModelDir }
        do {
            let container = try await loadModel(dir, extraEOSTokens: extraEOSTokens)
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
                self.idleUnloaded = false
                self.mlxTranscript = []   // fresh conversation: nothing to replay on a later reload
                // See handleNewSessionOpenAI: standing permissions are per session, not per process.
                resetSessionPermissionsLocked()
                return mode
            }
            rearmIdleTimerIfIdle()
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
    /// The configured system prompt with the session's working directory appended, so the
    /// model KNOWS its cwd (it cannot call getcwd - it only knows what is in its context).
    /// nil stays nil: the translator path (`--system-prompt ""`) prepends no system message,
    /// and injecting a directory line there would pollute every request. An unknown cwd
    /// (nil/empty/"/") appends nothing.
    private func effectiveSystemPrompt() -> String? {
        guard let base = systemPrompt else { return nil }
        guard let cwd = lock.withLock({ sessionCwd }) else { return base }
        return base
            + "\n\nYour current working directory is \(cwd). "
            + "Interpret relative file paths as relative to this directory."
    }

    private func buildSessionStack(
        container: ModelContainer, history: [Chat.Message]
    ) -> (ChatSession, MLXBackend, Agent) {
        let parameters = gen.apply(to: GenerateParameters(maxTokens: 4096, temperature: 0.7))
        let instructions = effectiveSystemPrompt()
        let session: ChatSession
        if history.isEmpty {
            session = ChatSession(
                container, instructions: instructions, generateParameters: parameters)
        } else {
            session = ChatSession(
                container, instructions: instructions, history: history,
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

    /// The openai counterpart of buildSessionStack: no container, no ChatSession - the
    /// backend holds the conversation itself. Same generation defaults (0.7 / 4096 overlaid
    /// with the CLI's --temperature etc.), same registry + mode re-application.
    private func buildOpenAIStack(
        baseURL: URL, history: [Chat.Message]
    ) -> (OpenAIBackend, Agent) {
        let parameters = gen.apply(to: GenerateParameters(maxTokens: 4096, temperature: 0.7))
        let backend = OpenAIBackend(
            baseURL: baseURL, parameters: parameters, systemPrompt: effectiveSystemPrompt(),
            seedHistory: history)
        let registry = lock.withLock { self.registry }
        let agent = Agent(backend: backend, registry: registry, guardrails: guardrails)
        agent.delegate = self
        let modeNow = lock.withLock { mode }
        agent.setToolsEnabled(modeNow == "agent")
        return (backend, agent)
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
        // One turn at a time: the ChatSession/backend is not safe to drive concurrently (see
        // MLXBackend). Claim the slot ATOMICALLY with the busy-check - see promptStarting.
        let (agent, ourSid, unloaded, busy) = lock.withLock {
            () -> (Agent?, String?, Bool, Bool) in
            let busy = self.promptTask != nil || self.promptStarting
            if !busy { self.promptStarting = true }
            return (self.agent, self.sessionID, self.idleUnloaded, busy)
        }
        if busy {
            respondError(id, -32003, "a prompt is already in progress for this session")
            return
        }
        // From here the claim (promptStarting) is held; every early return MUST release it.
        func releaseClaim() { lock.withLock { self.promptStarting = false } }
        // idleUnloaded means the session is live but its model was released to save RAM: the
        // agent is nil yet the prompt is valid, so reload rather than reject. A nil agent that is
        // NOT idle-unloaded is a genuine "no session yet".
        guard let ourSid, agent != nil || unloaded else {
            releaseClaim()
            respondError(id, -32002, "no active session; call session/new first")
            return
        }
        // ACP carries the target sessionId on every prompt; reject one that isn't ours
        // rather than silently answering for the wrong session.
        if let reqSid = params["sessionId"] as? String, reqSid != ourSid {
            releaseClaim()
            respondError(id, -32602, "unknown session: \(reqSid)")
            return
        }
        let text = Self.promptText(params)
        // Activity: stop any pending idle-unload before the turn starts. The claim already
        // blocks a queued idle event from unloading, and the timer is re-armed at turn end.
        cancelIdleTimer()
        // The agent drives the full turn (streaming + tool loop + permission gate) and
        // reports back through the AgentDelegate methods below. Task.isCancelled inside
        // runTurn is wired to session/cancel via this promptTask.
        let task = Task { [weak self] in
            guard let self else { return }
            defer {
                self.lock.withLock {
                    if self.promptTask != nil { self.promptTask = nil }
                    self.promptStarting = false
                }
                self.rearmIdleTimerIfIdle()
            }
            // Reload the model if it was idle-unloaded, then use the freshly rebuilt agent.
            let live: Agent
            if unloaded {
                do {
                    live = try await self.reloadAfterIdle()
                } catch {
                    self.log("idle-reload failed: \(error.localizedDescription)")
                    self.respondError(
                        id, -32000, "model reload failed: \(error.localizedDescription)")
                    return
                }
            } else if let agent {
                live = agent
            } else {
                self.respondError(id, -32002, "no active session; call session/new first")
                return
            }
            // Fresh buffer for this turn's assistant answer (see mlxTranscript).
            self.lock.withLock { self.assistantTurnBuffer = "" }
            let outcome = await live.runTurn([.user(text)])
            // A turn that ended between a gate and its answer leaves no permission parked,
            // but be defensive: resolve any stragglers so no continuation leaks.
            self.failPendingPermissions(with: .cancel)
            // Extend the shadow transcript ONLY on a clean stop and ONLY on the MLX path - it is
            // solely a reload aid, so a cancelled/failed turn (which the user interrupted or which
            // errored) must not enter the replayed context. A turn that produced no answer text
            // (e.g. tool-only, since tool exchanges are not replayed) is skipped whole so the
            // user/assistant pairing never carries an empty assistant message into prefill.
            if self.openaiBaseURL == nil, case .stop = outcome {
                self.lock.withLock {
                    guard !self.assistantTurnBuffer.isEmpty else { return }
                    self.mlxTranscript.append(.user(text))
                    self.mlxTranscript.append(.assistant(self.assistantTurnBuffer, toolCalls: nil))
                }
            }
            switch outcome {
            case .cancelled:
                self.log("turn resolved: stopReason=cancelled")
                self.respond(id, ["stopReason": "cancelled"])
            case .failed(let message):
                // Reply with a JSON-RPC error, not an out-of-spec stopReason. ACP's
                // StopReason enum has no "error" member. Log the RAW message (for debugging) but
                // send the client a message a user can act on (see userFacingTurnError).
                self.log("turn failed: \(message)")
                self.respondError(id, -32000, Self.userFacingTurnError(message))
            case .stop(let reason):
                self.log("turn resolved: stopReason=\(reason)")
                self.respond(id, ["stopReason": reason])
            }
        }
        // Publish the task ONLY if the turn is still starting. This class is not an actor, so the
        // Task above can begin (and, on a fast-fail path like a reload that throws, FINISH - defer
        // included) on another thread before this line runs. If the child's defer already cleared
        // the claim, `promptStarting` is false here and we must NOT store the now-dead task:
        // doing so would strand `promptTask` non-nil with nothing left to clear it, permanently
        // wedging prompts/idle-unload/prime. Serve dispatch is serialized (readabilityHandler), so
        // no other turn can claim between the child's defer and this line - `promptStarting` true
        // here means unambiguously our still-running turn.
        lock.withLock {
            if promptStarting {
                promptTask = task
                promptStarting = false
            }
        }
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
            // The applet owns llama-server, so it restarts the server to change models and
            // deliberately does NOT re-inject the transport - the conversation array here
            // survives, which IS the desired continue-with-a-new-model semantics. Nothing
            // for us to switch, and silently succeeding would be a lie.
            if openaiBaseURL != nil {
                respondError(id, -32601, "model is applet-managed on the openai backend")
                return
            }
            guard let target = availableModels().first(where: { $0.value == value }) else {
                respondError(id, -32602, "unknown model: \(value)")
                return
            }
            do {
                // Rebuild the session/backend/agent on the new model (context resets, as a
                // KV cache cannot carry across models); registry and mode are re-applied
                // by buildSessionStack.
                let container = try await loadModel(target.dir, extraEOSTokens: extraEOSTokens)
                let (session, backend, agent) = buildSessionStack(container: container, history: [])
                lock.withLock {
                    self.container = container
                    self.session = session
                    self.backend = backend
                    self.agent = agent
                    self.currentModelDir = target.dir
                    // Context resets with the model, so the reload shadow does too.
                    self.mlxTranscript = []
                    self.idleUnloaded = false
                }
                rearmIdleTimerIfIdle()
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
        guard let ourSid else {
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
        // Rebuild the stack around the restored history. The openai backend needs nothing but
        // the history itself; the MLX one additionally needs its loaded weights - and if those
        // were idle-unloaded, we do NOT reload here: prime just records the new context as the
        // reload shadow and leaves the model unloaded, so the next prompt reloads with it (the
        // KV prefill is lazy anyway). `container` was captured at the top, so a concurrent
        // idle-unload cannot free its buffers while this holds the strong ref.
        let session: ChatSession?
        let backend: GenerationBackend?
        let agent: Agent?
        if let openaiBaseURL {
            let (b, a) = buildOpenAIStack(baseURL: openaiBaseURL, history: history)
            (session, backend, agent) = (nil, b, a)
        } else if let container {
            let (s, b, a) = buildSessionStack(container: container, history: history)
            (session, backend, agent) = (s, b, a)
        } else {
            // MLX, idle-unloaded: no weights to build on. Recorded below; next prompt reloads.
            (session, backend, agent) = (nil, nil, nil)
        }
        // Refuse to swap the stack under a running turn. Checked inside the same lock
        // that publishes the swap, so a prompt accepted before us keeps its stack and
        // we bail; the client's ordering contract (cancel + await resolution before
        // priming) makes a first -32003 a rare transient it retries once.
        let busy = lock.withLock { () -> Bool in
            if promptTask != nil || promptStarting { return true }
            self.session = session
            self.backend = backend
            self.agent = agent
            if openaiBaseURL == nil {
                // Keep MLX reload state consistent with the rebuilt stack: a prime with the model
                // loaded is (re)loaded; one with it idle-unloaded stays unloaded (agent nil), and
                // either way `mlxTranscript` becomes the new context to replay on the next reload.
                self.container = (agent != nil) ? container : nil
                self.idleUnloaded = (agent == nil)
                self.mlxTranscript = history
            }
            // Priming REPLACES the conversation, so it replaces the consent context with it.
            // This is the clear that actually fires in production - see the helper. Inside the
            // busy check on purpose: a prime that bails leaves the running turn's stack alone,
            // and must leave its grants alone too.
            resetSessionPermissionsLocked()
            return false
        }
        if busy {
            respondError(id, -32003, "cannot prime while a prompt is in progress")
            return
        }
        rearmIdleTimerIfIdle()
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
    ///
    /// A standing session decision for this tool short-circuits the round-trip entirely: no
    /// request reaches the client, which is the whole point of "always" - the user asked not to
    /// be interrupted about this tool again.
    func agentRequestPermission(toolCallId: String, toolName: String, title: String) async
        -> PermissionOutcome
    {
        let standing = lock.withLock { () -> PermissionOutcome? in
            // Reject wins if a tool is ever in both: fail-closed. It cannot happen today (a
            // settled tool never prompts again, so only one set can gain it), but the ordering
            // should not be the thing standing between us and an unintended allow.
            if alwaysRejected.contains(toolName) { return .deny }
            if alwaysAllowed.contains(toolName) { return .allow }
            return nil
        }
        if let standing {
            log("permission: \(toolName) -> \(standing == .allow ? "allow" : "deny") (standing)")
            return standing
        }

        // Snapshot the epoch WITH the id, in one critical section: the answer to this request may
        // land after the session has been replaced, and the stamp is how that is detected.
        let (sid, requestID, epoch) = lock.withLock { () -> (String, Int, Int) in
            outboundCounter += 1
            return (sessionID ?? "", outboundCounter, sessionEpoch)
        }
        return await withCheckedContinuation {
            (cont: CheckedContinuation<PermissionOutcome, Never>) in
            lock.withLock {
                pendingPermissions[requestID] = PendingPermission(
                    cont: cont, toolName: toolName, epoch: epoch)
            }
            send([
                "jsonrpc": "2.0",
                "id": requestID,
                "method": "session/request_permission",
                "params": [
                    "sessionId": sid,
                    "toolCall": ["toolCallId": toolCallId, "title": title],
                    // ACP's standard four. The always variants are per TOOL, not per call, so
                    // approving write_file never approves the shell tool: a write is bounded by
                    // the sandbox's writable dirs, while a shell command inside the same sandbox
                    // can still delete the project or reach the network. Same option, different
                    // blast radius - so each tool is opted in on its own.
                    "options": [
                        ["optionId": "allow", "name": "Allow", "kind": "allow_once"],
                        ["optionId": "allow_always", "name": "Always Allow", "kind": "allow_always"],
                        ["optionId": "reject", "name": "Reject", "kind": "reject_once"],
                        ["optionId": "reject_always", "name": "Never Allow", "kind": "reject_always"],
                    ],
                ],
            ])
        }
    }

    private func handlePermissionResponse(id: Int, msg: [String: Any]) {
        let pending = lock.withLock { () -> PendingPermission? in
            let p = pendingPermissions[id]
            pendingPermissions[id] = nil
            return p
        }
        guard let pending else {
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
                // Anything that is not an explicit allow denies, including an optionId we never
                // advertised: an unrecognized choice must not run the tool.
                switch inner["optionId"] as? String {
                case "allow":
                    outcome = .allow
                case "allow_always":
                    record(pending, into: \.alwaysAllowed, verb: "always allow")
                    outcome = .allow
                case "reject_always":
                    record(pending, into: \.alwaysRejected, verb: "never allow")
                    outcome = .deny
                default:
                    outcome = .deny
                }
            case "cancelled":
                outcome = .cancel
            default:
                outcome = .cancel
            }
        } else if msg["error"] != nil {
            // The client errored on the request: treat as a denial, not a crash.
            outcome = .deny
        }
        pending.cont.resume(returning: outcome)
    }

    /// Record a standing decision, but ONLY if the session it was asked under is still current.
    /// The user answers a request that was parked before a session/new or session/prime landed;
    /// without the epoch check that answer would write consent into the session the clear just
    /// created. The turn's own outcome still honors the answer - the user did choose it - but it
    /// leaves no trace in the new conversation.
    private func record(
        _ pending: PendingPermission, into set: ReferenceWritableKeyPath<ACPServer, Set<String>>,
        verb: String
    ) {
        let stale = lock.withLock { () -> Bool in
            guard pending.epoch == sessionEpoch else { return true }
            self[keyPath: set].insert(pending.toolName)
            return false
        }
        if stale {
            log("permission: \(verb) \(pending.toolName) ignored - answered for a replaced session")
        } else {
            log("permission: \(verb) \(pending.toolName) for this session")
        }
    }

    private func failPendingPermissions(with outcome: PermissionOutcome) {
        let continuations = lock.withLock {
            () -> [CheckedContinuation<PermissionOutcome, Never>] in
            let waiting = pendingPermissions.values.map(\.cont)
            pendingPermissions.removeAll()
            return waiting
        }
        for cont in continuations { cont.resume(returning: outcome) }
    }

    // MARK: - AgentDelegate (streaming)

    func agentEmitText(kind: ThinkSplitter.Kind, _ text: String) {
        let sid: String? = lock.withLock {
            // Capture the answer (not the reasoning) for the reload shadow on the MLX path.
            if kind == .message, self.openaiBaseURL == nil { self.assistantTurnBuffer += text }
            return self.sessionID
        }
        guard let sid else { return }
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

    func agentTurnUsage(totalTokens: Int, tokensPerSecond: Double) {
        guard totalTokens > 0, let sid = lock.withLock({ sessionID }) else { return }
        var update: [String: Any] = ["sessionUpdate": "usage_update", "used": totalTokens]
        if tokensPerSecond > 0 {
            update["tokensPerSecond"] = (tokensPerSecond * 10).rounded() / 10
        }
        send(sessionUpdateEnvelope(sid, update))
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
        let options: [[String: Any]] = []
        // NO model picker, on EITHER backend - so no client renders one, and session/new's
        // configOptions is always empty.
        //
        // The openai backend never had one: the applet picks the gguf and restarts llama-server
        // itself, so a picker here would be a dead control. The MLX backend DID advertise one,
        // and that is what this removes. It produced a picker under the composer in MLX windows
        // and nothing in llama-server windows - an asymmetry that was the visible half of a real
        // problem: choosing a model there went straight to set_config_option and the APPLET
        // never saw it. The applet owns the window title, the model label, the RAM advisory and
        // the history stamping, so an agent-side switch left all of them describing a model that
        // was no longer loaded.
        //
        // Model choice belongs to the applet's picker, which is the only place that knows about
        // both engines, the tools choice that goes with a model, RAM headroom, and the window's
        // identity. Same reasoning that retired the mode picker: a live control for something the
        // applet owns can only disagree with it.
        //
        // The MECHANISM stays: session/set_config_option "model" still switches the model (see
        // handleSetConfigOption), and --model still works. Only the advertisement is gone, which
        // is what the applet's picker will drive for an in-place MLX switch.
        //
        // NO chat/agent mode picker. It is deliberately not advertised, so no client renders
        // it: whether a session is agentic is decided ONCE, before the agent is spawned, by
        // whether the host passes --mcp-config (the host's own "use tools" choice). "Chat
        // mode" IS an empty MCP config - same transport, no tools - so a second, live control
        // for the same question could only disagree with the first.
        //
        // It also could not honour the promise it made. The picker only ever appeared when
        // tools were ALREADY spawned (`haveTools`), so it could not turn a chat session
        // agentic - the servers were not there to enable. All it could do was mute tools the
        // user had explicitly asked for, which is not a decision worth a permanent control.
        //
        // `mode` survives internally as the agent's expression of that startup choice
        // (init: --mcp-config present -> "agent"), and `session/set_config_option` still
        // accepts it for any other ACP client. Only the affordance is gone.
        //
        // The three locals are read under one lock acquisition and kept: `currentModelDir` and
        // `registry` are the state a future advertised option would describe, and splitting the
        // locked read to drop them would trade a real invariant for cosmetics.
        _ = (current, modeNow, haveTools)
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

    /// Turn a raw generation-failure string into something a user can act on. The common opaque
    /// case is a chat-template (Jinja) render failure - e.g. "upper filter requires string" from
    /// a model whose tool-calling template our Jinja engine can't render. That text is meaningless
    /// to a user and reads like the app is broken, when it is a model-compatibility issue that
    /// disappears with tools off (the failing macros only run when tool schemas are rendered).
    /// The raw message is still logged (see the call site) for debugging.
    static func userFacingTurnError(_ raw: String) -> String {
        let lower = raw.lowercased()
        let templateSignals = [
            "filter requires", "no filter named", "unknown filter", "unknown tag",
            "jinja", "chat template", "template render", "template error",
        ]
        if templateSignals.contains(where: { lower.contains($0) }) {
            return
                "This model's chat template could not render the request, so it can't run here. "
                + "This usually affects tool calling - start a new chat with tools turned off, or "
                + "pick a different model. (details: \(raw))"
        }
        return "generation failed: \(raw)"
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
