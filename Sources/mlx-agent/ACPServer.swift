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
// backend per process (both keep the conversation and reuse the KV cache / prefix across
// prompts, so a follow-up turn's prefill is cheaper).
//
// Everything on stdout is JSON-RPC; all logging goes to stderr.

import AgentDigest
import AgentText
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace

/// Which engine generates. Chosen once at launch (`--backend`), never at runtime: they have
/// different lifecycles - the MLX path loads a model container into this process and can switch
/// models on the fly, the openai path talks to a server the HOST APP owns (it launches, restarts
/// and reaps llama-server), and the foundation path uses the OS's own resident model, which has
/// no model to choose at all.
enum EngineSpec: Sendable {
    case mlx(modelDir: String)
    case openai(baseURL: URL)
    /// Apple's on-device system model (macOS 26+, Apple Intelligence enabled). No weights of
    /// ours, nothing to load, nothing to switch. See FoundationBackend.
    case foundation
}

final class ACPServer: @unchecked Sendable, AgentDelegate {

    // Resolved once from the CLI (see resolveSystemPrompt): nil means NO system message is
    // prepended - the mode a translator uses, where the instruction lives in the user turn.
    private let systemPrompt: String?
    private let gen: GenConfig
    // Extra generation stop tokens unioned in at model-load time (see loadModel).
    private let extraEOSTokens: Set<String>
    private let lock = NSLock()

    /// Which engine this server was launched with. Set in init and never mutated, so no lock is
    /// needed to read it.
    ///
    /// This used to be `openaiBaseURL: URL?`, with nil meaning "the MLX path" - a binary switch
    /// read as `openaiBaseURL == nil` (five sites) or `if let` (three more). That encoding does
    /// not survive a third engine: every one of those sites means "MLX only" (weights to unload,
    /// a shadow transcript to replay, a model that can be switched), and under the old spelling a new
    /// backend would silently inherit all of them. `isMLXEngine` says what is actually meant.
    private let engine: EngineSpec

    /// True only for the in-process MLX path. What it gates: idle-unload of weights, the shadow
    /// transcript replayed after a reload, and runtime model switching - none of which exist for
    /// a backend that holds no weights of its own.
    private var isMLXEngine: Bool {
        if case .mlx = engine { return true }
        return false
    }

    /// openai: the endpoint root, nil on every other engine.
    private var openaiBaseURL: URL? {
        if case .openai(let baseURL) = engine { return baseURL }
        return nil
    }

    /// True only for Apple's on-device system model.
    private var isFoundationEngine: Bool {
        if case .foundation = engine { return true }
        return false
    }
    /// MLX path only; "" on the openai backend (the host app decides what llama-server serves).
    private var currentModelDir: String
    /// What `/props` said at session/new, on the openai engine only. `.unknown` everywhere else,
    /// and on a server that did not answer.
    ///
    /// This is the agent's ONLY way to learn what it is generating with on that engine: the host
    /// app owns llama-server and hands this process a port. Without it the summarizer is sized
    /// against a deliberately small constant and named after its engine, and both were wrong in
    /// the same measured failure - a server started with an 8192-token context, a restore sized
    /// for a guess, and an HTTP 400 on the first message afterwards.
    private var servedProps = OpenAIBackend.ServerProps.unknown
    private let mcpConfigPath: String?
    private let guardrails: AgentGuardrails
    private var mode: String

    // MLX-only: the loaded weights and the session driving them.
    private var container: ModelContainer?
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
    /// Where condensations are recorded, when --digest-dir asked for one. Handed to the backend
    /// too, so an overflow condensation inside a turn lands in the same place as a prime-time one.
    private let digestArchive: DigestArchive?
    /// Which model may summarize at prime time (`--digest-backend`). See DigestSelection.swift.
    private let digestBackend: DigestBackendChoice
    /// `--digest-window`: the summarizing model's context window, when the operator states it.
    /// nil means "work it out", which every engine can now do: foundation asks the framework, mlx
    /// reads config.json, and openai asks the server it was pointed at for its own `/props`. The
    /// flag is for the case none of those cover - an OpenAI-compatible server that reports no
    /// window - and it outranks all three.
    private let digestWindowOverride: Int?
    private let idleQueue = DispatchQueue(label: "com.abracode.mlx-agent.idle")
    private var idleTimer: DispatchSourceTimer?
    // True once the model has been idle-unloaded: the session is still "live" (sessionID set,
    // registry up) but `container`/`agent` are nil, so the next prompt reloads instead of being
    // rejected as "no active session".
    private var idleUnloaded = false
    // A shadow of the conversation as clean (user, assistant) text turns, kept ONLY so an
    // idle-unload can rebuild the backend's context on reload: the unload throws the backend away
    // along with its history, and this outlives it. (It predates MLXBackend owning a readable
    // conversation and could now be sourced from it instead; left as-is because it is also what
    // the openai path and llama-server's own idle-sleep rely on.) Text turns only: intra-turn
    // tool calls/results are not replayed, so after an idle reload the model keeps the
    // conversation but not the exact tool outputs. That trade got slightly worse when MLXBackend
    // took ownership of the conversation: its history now also holds tool exchanges and the partial
    // answers of canceled turns, none of which are mirrored here, so post-reload context is poorer
    // than the live context in a way it was not when the history was unreadable. (Appended only on
    // a clean .stop so a canceled/failed turn never enters the replayed context.) Reset to the primed
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
    // Claimed for the whole of a condensing session/prime, which generates and is therefore async.
    // A separate flag from `promptStarting` so the refusal messages stay accurate; it is folded
    // into EVERY busy check that tests promptStarting (idleFired, handlePrompt,
    // handleSetConfigOption, handleSessionPrime). The one place it is deliberately NOT tested is
    // `finishPrime`'s publish check - that runs while this flag is held BY US.
    private var condensing = false
    /// The in-flight condensation, so session/cancel can reach it. AgentDigest checks
    /// `Task.isCancelled` once per slice and FMDigestGenerator checks it on entry - all of which
    /// was unreachable from the wire until this existed.
    private var condenseTask: Task<Void, Never>?
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
        extraEOSTokens: Set<String> = [], idleUnloadSeconds: TimeInterval = IdleUnload.defaultSeconds,
        digestArchive: DigestArchive? = nil, digestBackend: DigestBackendChoice = .auto,
        digestWindowOverride: Int? = nil
    ) {
        self.digestArchive = digestArchive
        self.digestBackend = digestBackend
        self.digestWindowOverride = digestWindowOverride
        self.idleUnloadSeconds = idleUnloadSeconds
        self.engine = engine
        switch engine {
        case .mlx(let modelDir):
            self.currentModelDir = modelDir
        case .openai, .foundation:
            // The host app decides what llama-server serves, and the OS decides what the system
            // model is; there is no model dir on either side.
            self.currentModelDir = ""
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
        switch engine {
        case .openai(let baseURL):
            log("ACP server ready (stdio JSON-RPC). backend: openai at \(baseURL.absoluteString)")
        case .foundation:
            log("ACP server ready (stdio JSON-RPC). backend: foundation (on-device system model)")
        case .mlx:
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
        guard isMLXEngine, idleUnloadSeconds > 0 else { return }
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
                guard self.promptTask == nil, !self.promptStarting, !self.condensing,
                    self.container != nil
                else {
                    return nil
                }
                self.container = nil
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
        let (backend, agent) = buildSessionStack(container: container, history: history)
        lock.withLock {
            self.container = container
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
                    "agentInfo": ["name": "mlx-agent", "version": agentVersion],
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
        switch engine {
        case .openai(let baseURL):
            await handleNewSessionOpenAI(id: id, baseURL: baseURL)
        case .foundation:
            await handleNewSessionFoundation(id: id)
        case .mlx:
            await handleNewSessionMLX(id: id)
        }
    }

    /// foundation: nothing to load and no server to reach - the model is the OS's and already
    /// resident. The one thing that CAN be wrong is availability, so it is checked here, which
    /// makes an unusable machine one clean error at session/new instead of a failure on every
    /// prompt (the same reason the openai path health-checks).
    private func handleNewSessionFoundation(id: Int?) async {
        let status = FMAvailability.probe()
        guard status.isAvailable else {
            respondError(id, -32000, "session/new failed: \(status.summary)")
            return
        }
        let registry = await ensureRegistry()
        let (backend, agent) = buildFoundationStack(history: [])
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
            "session ready: \(sid) on foundation backend "
                + "[mode=\(modeNow), tools=\(toolCount)]")
        respond(id, ["sessionId": sid, "configOptions": configOptionsJSON()])
    }

    /// openai: nothing to load - the host app already launched llama-server on a pinned port.
    /// Health-check it so a dead server is ONE clean error here rather than a failure on
    /// every prompt, then build the registry + stack exactly as the MLX path does.
    private func handleNewSessionOpenAI(id: Int?, baseURL: URL) async {
        do {
            try await OpenAIBackend.waitForHealth(baseURL: baseURL)
        } catch {
            respondError(id, -32000, "session/new failed: \(error.localizedDescription)")
            return
        }
        // AFTER health, and never a reason to fail: what it reports only sharpens decisions that
        // have a fallback. Read once per session because the host app restarts llama-server rather
        // than reconfiguring it, and a restart replaces this whole agent process anyway.
        let props = await OpenAIBackend.probeProps(baseURL: baseURL)
        lock.withLock { self.servedProps = props }
        // The model path is a SERVER-supplied string, so it is logged through the same naming that
        // bounds and flattens it everywhere else rather than whole - see DigestGeneratorName.
        log(
            "openai engine: server reports context=\(props.contextSize.map(String.init) ?? "unstated") "
                + "model=\(props.modelPath.flatMap(DigestGeneratorName.fromModelPath) ?? "unstated")")
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
            let (backend, agent) = buildSessionStack(container: container, history: [])
            let sid = "mlx-session-1"
            let modeNow = lock.withLock { () -> String in
                self.container = container
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
    ) -> (MLXBackend, Agent) {
        let parameters = gen.apply(to: GenerateParameters(maxTokens: 4096, temperature: 0.7))
        let instructions = effectiveSystemPrompt()
        let registry = lock.withLock { self.registry }
        let backend = MLXBackend(
            container: container, instructions: instructions, history: history,
            parameters: parameters)
        let agent = Agent(backend: backend, registry: registry, guardrails: guardrails)
        agent.delegate = self
        let modeNow = lock.withLock { mode }
        agent.setToolsEnabled(modeNow == "agent")
        return (backend, agent)
    }

    /// The openai counterpart of buildSessionStack: no container, no model in-process - the
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

    /// The foundation counterpart of buildSessionStack: no container and no server - the OS owns
    /// the model. Same generation defaults and the same registry + mode re-application. Ordering
    /// matters here in a way it does not for the other two: `Agent.init` installs the tool runner
    /// this backend dispatches through, so it has to precede `setToolsEnabled` - which it does,
    /// because the tool set cannot be built until both have arrived.
    private func buildFoundationStack(history: [Chat.Message]) -> (FoundationBackend, Agent) {
        let parameters = gen.apply(to: GenerateParameters(maxTokens: 4096, temperature: 0.7))
        let backend = FoundationBackend(
            parameters: parameters, systemPrompt: effectiveSystemPrompt(), seedHistory: history,
            archive: digestArchive, log: { [weak self] message in self?.log(message) })
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
        // One turn at a time: the backend is not safe to drive concurrently (see
        // MLXBackend). Claim the slot ATOMICALLY with the busy-check - see promptStarting.
        let (agent, ourSid, unloaded, blocker) = lock.withLock {
            () -> (Agent?, String?, Bool, String?) in
            let blocker = self.busyReasonLocked()
            if blocker == nil { self.promptStarting = true }
            return (self.agent, self.sessionID, self.idleUnloaded, blocker)
        }
        if let blocker {
            respondError(id, -32003, blocker)
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
            // solely a reload aid, so a canceled/failed turn (which the user interrupted or which
            // errored) must not enter the replayed context. A turn that produced no answer text
            // (e.g. tool-only, since tool exchanges are not replayed) is skipped whole so the
            // user/assistant pairing never carries an empty assistant message into prefill.
            if self.isMLXEngine, case .stop = outcome {
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
        // The session check comes FIRST. It used to sit after the condense cancel below, so a
        // cancel naming somebody else's session killed this one's summarization while logging
        // "ignoring - unknown session".
        let (task, ourSid) = lock.withLock { (promptTask, sessionID) }
        if let reqSid = params["sessionId"] as? String, let ourSid, reqSid != ourSid {
            log("cancel: ignoring - unknown session \(reqSid)")
            return
        }
        // A condensation is cancellable and can otherwise hold the session for minutes; Stop
        // must reach it too, not just a running prompt.
        if let condense = lock.withLock({ self.condenseTask }) {
            condense.cancel()
            log("session/cancel: canceling an in-flight context summarization")
        }
        log(task == nil ? "cancel: no in-flight turn" : "cancel: canceling in-flight turn")
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
            // Busy-checked like every other stack mutation in this file. Flipping tools under a
            // live turn would change the backend's tool set between passes of one tool loop, so
            // the model would be told mid-turn that the tools it just called no longer exist.
            let (blocker, agent) = lock.withLock { () -> (String?, Agent?) in
                if let blocker = busyReasonLocked() { return (blocker, nil) }
                self.mode = value
                return (nil, self.agent)
            }
            if let blocker {
                respondError(id, -32003, blocker)
                return
            }
            agent?.setToolsEnabled(value == "agent")
            log("mode -> \(value)")
            respond(id, ["configOptions": configOptionsJSON()])

        case "model":
            // No longer latent. This branch publishes container, backend, agent AND mlxTranscript,
            // and it was the one stack mutation in this file with no busy check at all; now that a
            // condensing prime can borrow the LIVE MLX backend to summarize with, switching the
            // model underneath it would leave that summarization generating against a backend
            // nobody holds while a new one loads over the same container.
            if let blocker = lock.withLock({ busyReasonLocked() }) {
                respondError(id, -32003, blocker)
                return
            }
            // The host app owns llama-server, so it restarts the server to change models and
            // deliberately does NOT re-inject the transport - the conversation array here
            // survives, which IS the desired continue-with-a-new-model semantics. Nothing
            // for us to switch, and silently succeeding would be a lie.
            if !isMLXEngine {
                // Silently succeeding would be a lie on either engine, for different reasons:
                // llama-server is restarted to change models (by the host app, or by whoever
                // launched it), and the foundation backend has exactly one model, which belongs
                // to the OS.
                let why =
                    isFoundationEngine
                    ? "the foundation backend has a single OS-provided model"
                    : "the openai backend's model is whatever llama-server was launched with; "
                        + "restart the server to change it"
                respondError(id, -32601, why)
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
                let (backend, agent) = buildSessionStack(container: container, history: [])
                lock.withLock {
                    self.container = container
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
        var wireBytes = 0
        for m in rawMessages { wireBytes += (m["content"] as? String)?.utf8.count ?? 0 }
        let supplied = rawMessages.count

        // No `condense` key: byte-for-byte the behavior this method has always had, including
        // staying synchronous. That is the contract - a plain prime is near-instant and every
        // existing client depends on it.
        guard let request = params["condense"] as? [String: Any] else {
            let history = Self.primeHistory(from: rawMessages) { [weak self] in self?.log($0) }
            finishPrime(
                id: id, container: container, history: history, suppliedCount: supplied,
                wireBytes: wireBytes, extra: [:])
            return
        }

        // Condensing generates, so this path is async - and that is the whole difficulty. The
        // busy slot is claimed HERE, synchronously, before the Task exists: between "we decided
        // to condense" and "the Task is running" a prompt could otherwise be accepted, and it
        // would then be generating against a stack we are about to swap. Same discipline as
        // `promptStarting`, different flag so the refusal messages stay honest.
        //
        // The epoch captured alongside it is the OTHER half, and the more important one. Blocking
        // prompts is not enough: session/new and a plain session/prime both legitimately REPLACE
        // the session, and neither waits for us. Publishing our stack afterwards would resurrect
        // a conversation the user discarded - measured, with the model then quoting from it. So
        // this records which session we started summarizing for, and finishPrime refuses to
        // publish into a different one.
        let (claimed, epoch) = lock.withLock { () -> (Bool, Int) in
            if busyReasonLocked() != nil { return (false, 0) }
            condensing = true
            return (true, sessionEpoch)
        }
        guard claimed else {
            respondError(id, -32003, "cannot prime while a prompt is in progress")
            return
        }

        // Only what the client ASKED FOR is read here - including which summarizer, which is a
        // request rather than a decision. Whether that summarizer can run, and therefore which
        // window the slice budget is derived from, depends on how many slices the history needs,
        // which is not known until the history has been parsed. So the policy is finished inside
        // the Task, next to the choice it belongs to.
        let overrides = Self.condenseOverrides(from: request)
        // Boxed because `Chat.Message` is not Sendable and the raw wire array is a function
        // PARAMETER, which strict concurrency treats as belonging to the caller's region rather
        // than a disconnected one - so neither can simply be captured. The box is honest here:
        // it is written before the Task exists and read only inside it.
        let boxed = PrimeHistoryBox(
            Self.primeHistory(from: rawMessages) { [weak self] in self?.log($0) })
        let accepted = boxed.messages.count
        // Stored so session/cancel can reach it. Without that the slot is held for up to
        // maxSlices * the per-slice timeout - minutes - during which every prompt, mode switch
        // and prime is refused, and the cancellation support the planner and generator already
        // implement is unreachable from the wire.
        let task = Task { [weak self] in
            guard let self else { return }
            defer {
                self.lock.withLock {
                    self.condensing = false
                    self.condenseTask = nil
                }
            }
            // Re-read rather than capture, and safe for a specific reason: idleFired tests
            // `condensing`, which we are holding, so no unload can free the container underneath
            // us for as long as this runs.
            let container = self.lock.withLock { self.container }
            let outcome = await self.condenseForPrime(
                history: boxed.messages, overrides: overrides)
            self.finishPrime(
                id: id, container: container, history: outcome.history, suppliedCount: accepted,
                wireBytes: wireBytes, extra: outcome.extra, expectedEpoch: epoch)
        }
        // Publish ONLY if the condensation is still running, for the reason spelled out on
        // `promptTask` above: this class is not an actor, so the Task can finish - defer included -
        // before this line runs, and storing a dead handle would leave `condenseTask` non-nil
        // forever. Newly easy to reach: with `--digest-backend none` the whole body is synchronous.
        // The consequence is smaller here (a later cancel logs a cancellation that is not
        // happening, and can cancel the dead task instead of a live one) but it is the same bug.
        lock.withLock {
            if condensing { condenseTask = task }
        }
    }

    /// Why a stack-mutating request must be refused right now, or nil when it may proceed.
    /// MUST be called with `lock` held.
    ///
    /// One place rather than four copies of the same disjunction, and it names the ACTUAL blocker:
    /// telling a client "a prompt is already in progress" while what is really running is a
    /// ten-second summarization sends it looking for a prompt that does not exist.
    private func busyReasonLocked() -> String? {
        if promptTask != nil || promptStarting {
            return "a prompt is already in progress for this session"
        }
        if condensing {
            return "the session context is being summarized; retry shortly"
        }
        return nil
    }

    /// Carries a primed history across the Task boundary. One writer, one reader, and the write
    /// happens-before the Task exists - there is no concurrency here to check.
    private final class PrimeHistoryBox: @unchecked Sendable {
        let messages: [Chat.Message]
        init(_ messages: [Chat.Message]) { self.messages = messages }
    }

    /// Build the stack around `history`, publish it, and respond.
    ///
    /// Shared by the plain and condensing prime paths so the two cannot drift - the stack-building
    /// rules here (which engine needs what, the idle-unload interaction, the permission reset) are
    /// subtle enough that a second copy would be wrong within a release.
    /// - Parameter expectedEpoch: nil on the synchronous path, which cannot be stale. On the
    ///   condensing path it is the session epoch captured when the claim was taken; publishing
    ///   into a different one would restore a conversation the client already replaced.
    private func finishPrime(
        id: Int?, container: ModelContainer?, history: [Chat.Message], suppliedCount: Int,
        wireBytes: Int, extra: [String: Any], expectedEpoch: Int? = nil
    ) {
        // Rebuild the stack around the restored history. The openai backend needs nothing but
        // the history itself; the MLX one additionally needs its loaded weights - and if those
        // were idle-unloaded, we do NOT reload here: prime just records the new context as the
        // reload shadow and leaves the model unloaded, so the next prompt reloads with it (the
        // KV prefill is lazy anyway). `container` was captured by the caller, so a concurrent
        // idle-unload cannot free its buffers while this holds the strong ref.
        let backend: GenerationBackend?
        let agent: Agent?
        if let openaiBaseURL {
            let (b, a) = buildOpenAIStack(baseURL: openaiBaseURL, history: history)
            (backend, agent) = (b, a)
        } else if case .foundation = engine {
            let (b, a) = buildFoundationStack(history: history)
            (backend, agent) = (b, a)
        } else if let container {
            let (b, a) = buildSessionStack(container: container, history: history)
            (backend, agent) = (b, a)
        } else {
            // MLX, idle-unloaded: no weights to build on. Recorded below; next prompt reloads.
            (backend, agent) = (nil, nil)
        }
        // Refuse to swap the stack under a running turn. Checked inside the same lock
        // that publishes the swap, so a prompt accepted before us keeps its stack and
        // we bail; the client's ordering contract (cancel + await resolution before
        // priming) makes a first -32003 a rare transient it retries once.
        //
        // `condensing` is deliberately NOT tested here: on the condensing path this runs while
        // that flag is held by the very Task calling us, and testing it would deadlock the
        // feature against itself. `expectedEpoch` is what guards the foreign-caller case instead.
        var superseded = false
        let busy = lock.withLock { () -> Bool in
            if promptTask != nil || promptStarting { return true }
            // The session was replaced while we were summarizing (session/new, or another prime).
            // Publishing now would resurrect the conversation the client just discarded.
            if let expectedEpoch, expectedEpoch != sessionEpoch {
                superseded = true
                return true
            }
            // Released HERE rather than in the Task's defer, so the slot opens the instant the
            // stack is published rather than after the response is written. The defer stays as a
            // backstop for the paths that never reach this point.
            if expectedEpoch != nil { condensing = false }
            self.backend = backend
            self.agent = agent
            if isMLXEngine {
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
            if superseded {
                log("session/prime: discarding a condensed context - the session was replaced")
                respondError(
                    id, -32003,
                    "the session was replaced while its context was being summarized")
            } else {
                respondError(id, -32003, "cannot prime while a prompt is in progress")
            }
            return
        }
        rearmIdleTimerIfIdle()
        log(
            "session primed: \(history.count) messages (\(suppliedCount) supplied, "
                + "~\(wireBytes) content bytes)")
        var payload: [String: Any] = ["primed": history.count]
        for (key, value) in extra { payload[key] = value }
        respond(id, payload)
    }

    // MARK: - session/prime condensation

    /// The knobs a condensing prime exposes on the wire.
    ///
    /// `sliceBudgetTokens` and `maxSlices` are deliberately NOT among them: they are properties of
    /// the summarizing model's context window, which the agent knows and the client does not - and
    /// which is not even decided until a summarizer has been chosen. `PrimePolicy` clamps whatever
    /// arrives, so a hostile value cannot get past this into the slicing loop.
    ///
    /// `backend` IS among them, and it is the one that is a request rather than a bound. It names
    /// the summarizer for this restore, defaulting to `--digest-backend` when absent. A client
    /// that offers the choice to a person needs it: the launch flag belongs to the agent process,
    /// so without this a user who picks a summarizer is answered by whichever one the agent was
    /// started with, and the only trace of the difference is a name in the response.
    struct CondenseOverrides: Sendable {
        var keepRecentTurns: Int?
        var maxDigestTokens: Int?
        /// The summarizer this restore asked for, or nil to use the launch flag.
        var backend: DigestBackendChoice?
        /// Why a `backend` that was present could not be used, ready to be reported verbatim.
        /// Set INSTEAD of `backend`, never alongside it.
        var backendRefusal: String?
    }

    /// A client's `backend` value, rendered short enough to put in a reason and a log line.
    ///
    /// Echoing it raw is wrong in two ways that both showed up in review: a JSON object's
    /// description is multi-line, and `log()` writes one prefixed line, so one refusal became
    /// several unprefixed lines of stderr; and there is no length bound on a client's string, so
    /// a megabyte of it would land in `reason` and in the log. Whitespace is collapsed and the
    /// value is capped - the point is to let a person recognize what they sent, not to reproduce
    /// it.
    private static func describeBackendValue(_ value: String) -> String {
        // SCALARS, NOT CHARACTERS. A `Character` is a grapheme cluster, so one base letter
        // followed by twenty thousand combining marks counts as ONE - a cap measured in
        // Characters passes it through whole, which is 40 KB into `reason` and into a single log
        // line. Measured, not theorized.
        //
        // Controls are stripped alongside whitespace: `isWhitespace` is Unicode White_Space,
        // which does not include ESC, so a value carrying terminal escapes would arrive intact at
        // a terminal reading the log. The scan is bounded too, so a megabyte of whitespace cannot
        // be walked in full to produce nothing.
        let strip = CharacterSet.whitespacesAndNewlines.union(.controlCharacters)
        let limit = 40
        let scanned = 4096
        var out = String.UnicodeScalarView()
        var gap = false
        var truncated = false
        for scalar in value.unicodeScalars.prefix(scanned) {
            if strip.contains(scalar) {
                gap = !out.isEmpty
                continue
            }
            if out.count + (gap ? 2 : 1) > limit {
                truncated = true
                break
            }
            if gap {
                out.append(" ")
                gap = false
            }
            out.append(scalar)
        }
        if !truncated, value.unicodeScalars.count > scanned { truncated = true }
        return truncated ? String(out) + "..." : String(out)
    }

    private static func condenseOverrides(from request: [String: Any]) -> CondenseOverrides {
        func int(_ key: String) -> Int? { (request[key] as? NSNumber)?.intValue }
        var overrides = CondenseOverrides(
            keepRecentTurns: int("keepRecentTurns"), maxDigestTokens: int("maxDigestTokens"))
        // NULL AND EMPTY ARE ABSENT; ANYTHING ELSE UNRECOGNIZED IS AN ERROR. A host storing "the
        // user has not chosen" as an empty string (or as JSON null) is ordinary, and refusing to
        // summarize over it would turn a non-answer into a visible failure. A value that is
        // present and is not one of the four words - including one of the wrong TYPE, which is a
        // client bug rather than a preference - is reported rather than quietly replaced by the
        // launch flag, because replacing it is how a summary gets written by a model nobody chose.
        if let rawValue = request["backend"], !(rawValue is NSNull) {
            if let raw = rawValue as? String {
                // Trimmed but not lowercased for the message: the refusal should quote what they
                // sent. Lowercasing is for the match only. Trimming first also keeps the
                // description's scan budget for the value rather than for leading spaces.
                let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                let shown = Self.describeBackendValue(trimmed)
                if trimmed.isEmpty || shown.isEmpty {
                    // Nothing requested. `shown` is empty when the value was made entirely of
                    // characters that cannot be displayed - controls, bidi overrides - which is
                    // the same class of non-answer as whitespace and is treated the same way
                    // rather than being refused as a summarizer named "".
                } else if let choice = DigestBackendChoice(rawValue: trimmed.lowercased()) {
                    overrides.backend = choice
                } else {
                    overrides.backendRefusal =
                        "unknown summarizer \"\(shown)\" requested: "
                        + "expected \(DigestBackendChoice.usage)"
                }
            } else {
                // The TYPE, not the value. JSON booleans arrive as NSNumber, so `true` echoed as a
                // value reads as a summarizer named "1" - a diagnostic naming something the client
                // never sent.
                let kind: String
                if let number = rawValue as? NSNumber,
                    CFGetTypeID(number as CFTypeRef) == CFBooleanGetTypeID()
                {
                    kind = "a boolean"
                } else if rawValue is NSNumber {
                    kind = "a number"
                } else if rawValue is [Any] {
                    kind = "an array"
                } else if rawValue is [String: Any] {
                    kind = "an object"
                } else {
                    kind = "\(type(of: rawValue))"
                }
                overrides.backendRefusal =
                    "condense.backend must be a string naming one of "
                    + "\(DigestBackendChoice.usage), got \(kind)"
            }
        }
        return overrides
    }

    /// The summarizer this restore runs under: what the request asked for, else the launch flag.
    ///
    /// `--digest-backend none` is NOT overridable. Every other value is a default the client may
    /// refine per restore, but `none` is the operator saying this agent does not summarize, and a
    /// request that could switch it back on would make the flag advisory. The refusal below says
    /// so rather than silently ignoring the request.
    private func effectiveBackend(_ overrides: CondenseOverrides) -> DigestBackendChoice {
        if digestBackend == DigestBackendChoice.none { return DigestBackendChoice.none }
        if let requested = overrides.backend { return requested }
        return digestBackend
    }

    /// A summarizer that could run, or why one could not.
    private enum SummarizerOption {
        /// `backend == nil` means Apple's on-device model: `FMDigestGenerator` opens its own
        /// session and never touches this session's backend. Otherwise the session's LIVE backend
        /// drives the generation - never a second one built for the purpose, which would have its
        /// own PassGate and therefore no mutual exclusion with this one (see BackendDigest.swift).
        case ready(backend: GenerationBackend?, policy: PrimePolicy, name: String)
        /// Reported to the client verbatim as the `reason` the history was primed whole.
        case unavailable(String)
    }

    /// A slice on the session's own model is a full generation - prefill of the slice plus up to
    /// `maxDigestTokens` of decode - so it is tens of seconds, not the on-device model's 3-5.
    ///
    /// Four, not six, because four is what fits `condenseDeadline`: a slice measured up to a minute
    /// on a large model, and a limit the deadline cannot honor is a limit that spends five minutes
    /// and then throws away every slice it finished. At the budget a 32k window gives (about 103 KB
    /// of text per slice, after the generation reserve below) four still covers far more
    /// conversation than any client sends; past it the planner primes the full history, which is
    /// slow but immediate.
    private static let sessionMaxSlices = 4
    /// Hard bound on ONE session-backed pass. A 30k-token prefill plus 700 tokens of decode on a
    /// large local model is a minute in the worst case measured; this is twice that, and it is a
    /// wedged-model backstop rather than an expected wait.
    ///
    /// It is NOT the bound on a condensation: a slice can take two passes (the repair retry gets
    /// its own), so this alone would allow `sessionMaxSlices * 2 * 120` = 16 minutes with the busy
    /// slot held the whole time. `condenseDeadline` is what actually bounds it.
    private static let sessionSliceTimeout: TimeInterval = 120
    /// Bound on the WHOLE condensation - every pass, every retry, and BOTH summarizers when `auto`
    /// falls back. Captured once as an absolute time in `condenseForPrime` and handed to each
    /// generator, so "how long can a condensing prime take" has one answer rather than one per
    /// attempt.
    ///
    /// Five minutes is already past what a restore should cost; what it really guards is the case
    /// where every pass is slow rather than wedged, which no per-pass timeout can see. An
    /// interactive client can escape with `session/cancel`; an automated one cannot, and this is
    /// what it has instead.
    private static let condenseDeadline: TimeInterval = 300
    /// How little time may remain and a second summarizer still be worth starting. Below this it
    /// would burn the rest of the deadline to arrive at the full-history prime that is already
    /// available for free.
    private static let secondChanceMinimumRemaining: TimeInterval = 60

    /// The on-device summarizer, or why it cannot run.
    private func foundationOption(overrides: CondenseOverrides) -> SummarizerOption {
        let status = FMAvailability.probe()
        guard status.isAvailable else { return .unavailable(status.summary) }
        #if canImport(FoundationModels)
            guard #available(macOS 26.0, *) else { return .unavailable(status.summary) }
            return .ready(
                backend: nil,
                policy: FoundationBackend.defaultDigestPolicy().with(
                    keepRecentTurns: overrides.keepRecentTurns,
                    maxDigestTokens: overrides.maxDigestTokens),
                name: foundationSummarizerName)
        #else
            return .unavailable("this build has no Foundation Models support")
        #endif
    }

    /// The session's own model as the summarizer, or why it cannot be one.
    private func sessionOption(overrides: CondenseOverrides) -> SummarizerOption {
        if isFoundationEngine {
            // Driving FoundationBackend through the generic generator would summarize with the
            // same model, in the same window, minus the guided generation that makes its output
            // parseable. There is no version of that which is the better choice.
            return .unavailable(
                "on the foundation engine the session's model IS the on-device model - "
                    + "use --digest-backend foundation")
        }
        let (live, modelDir, unloaded, props) = lock.withLock {
            (self.backend, self.currentModelDir, self.idleUnloaded, self.servedProps)
        }
        guard let live else {
            // Deliberately not a reload. Priming does not reload weights (the KV prefill is lazy
            // anyway), and loading a multi-gigabyte model in order to summarize is not what a
            // client asking for a fast restore wants.
            return .unavailable(
                unloaded
                    ? "the model is idle-unloaded; the full history was primed rather than "
                        + "reloading it to summarize"
                    : "this session has no live backend to summarize with")
        }
        // Window, generation reserve and the starvation check all live in DigestSizing: the
        // offline `digest` mode drives a backend the same way, and a reservation that differed
        // between the two would be silent in both. The reason it can return is client-facing here
        // and names flags that exist in both modes.
        let policy = DigestSizing.backendPolicy(
            window: DigestSizing.window(
                override: digestWindowOverride, modelDir: isMLXEngine ? modelDir : "",
                discovered: isMLXEngine ? nil : props.contextSize),
            generationCeiling: DigestSizing.generationCeiling(gen),
            maxSlices: Self.sessionMaxSlices,
            keepRecentTurns: overrides.keepRecentTurns,
            maxDigestTokens: overrides.maxDigestTokens)
        switch policy {
        case .refused(let reason):
            return .unavailable(reason)
        case .sized(let policy):
            return .ready(
                backend: live, policy: policy,
                name: BackendDigestGenerator.name(
                    engine: isMLXEngine ? "mlx" : "openai",
                    model: isMLXEngine ? modelDir : props.modelPath))
        }
    }

    /// Pick the summarizer for THIS history.
    ///
    /// `auto` measures rather than prefers. The on-device model costs the session nothing - no
    /// weights, no disturbance to the loaded model - so it wins whenever it can do the job; but its
    /// 4096-token window turns a long conversation into many sequential passes, and past its
    /// `maxSlices` it refuses outright. `DigestPlanner.sliceCount` answers "how many passes would
    /// this actually take" from the real history with pure string arithmetic, which turns the
    /// choice into a measurement.
    private func chooseSummarizer(history: [DigestTurn], overrides: CondenseOverrides)
        -> SummarizerOption
    {
        // A value that is not one of the four words DECLINES rather than falling back to the
        // launch flag. Falling back would summarize with a model the client did not ask for and
        // report it only in `summarizer`, which is the failure this parameter exists to end; the
        // full history is primed instead, and the reason names what was wrong with the value.
        //
        // Checked BEFORE `none`, which otherwise wins over everything: a client that sent a
        // malformed value should hear about it even on an agent that would have declined anyway,
        // or the same request looks fine everywhere summarization happens to be off.
        if let refusal = overrides.backendRefusal { return .unavailable(refusal) }
        switch effectiveBackend(overrides) {
        case .none:
            if digestBackend == DigestBackendChoice.none {
                if let requested = overrides.backend, requested != DigestBackendChoice.none {
                    return .unavailable(
                        "summarization is disabled for this agent (--digest-backend none), so "
                            + "the requested \(requested.rawValue) summarizer was not used")
                }
                return .unavailable("summarization is disabled (--digest-backend none)")
            }
            return .unavailable("this restore asked for no summarizer")
        case .foundation:
            return foundationOption(overrides: overrides)
        case .session:
            return sessionOption(overrides: overrides)
        case .auto:
            let foundation = foundationOption(overrides: overrides)
            var why: String
            switch foundation {
            case .ready(_, let policy, _):
                let passes = DigestPlanner.sliceCount(of: history, policy: policy)
                // A count of 0 means there is nothing to summarize at all, and the on-device model
                // is the cheapest way to arrive at that answer - the planner reports it without
                // generating anything.
                if passes <= policy.maxSlices { return foundation }
                why =
                    "the on-device model would need \(passes) passes, above its limit of "
                    + "\(policy.maxSlices)"
            case .unavailable(let reason):
                why = reason
            }
            let session = sessionOption(overrides: overrides)
            switch session {
            case .ready(_, _, let name):
                log("digest backend: chose \(name) because \(why)")
                return session
            case .unavailable(let reason):
                // Both reasons, because either one alone reads as the whole story and is not.
                return .unavailable("\(why); and \(reason)")
            }
        }
    }

    /// The context window of the model that will ANSWER, when this side knows it.
    ///
    /// A DIFFERENT QUESTION FROM THE SUMMARIZER'S WINDOW, which is what `DigestSizing.window`
    /// answers. They are the same model on the session path and different models whenever the
    /// on-device model summarizes for a local one, and only this one decides whether the result
    /// can be primed at all.
    ///
    /// `--digest-window` is deliberately NOT consulted: it is documented as the SUMMARIZING
    /// model's window, and an operator lowering it to force smaller slices would otherwise also be
    /// silently trimming their conversations.
    ///
    /// NIL ON THE FOUNDATION ENGINE, WHICH IS NOT AN OVERSIGHT. Its window is 4096 and perfectly
    /// well known, so this could answer - but `FoundationBackend` already handles an overflow by
    /// SUMMARIZING (`needsCondensingBefore` before a pass, one reactive retry after), and that is
    /// strictly better than dropping turns. Trimming here would pre-empt it: the turns would be
    /// gone before the backend that knows how to keep their content ever saw them. Every other
    /// engine has no such recovery, which is what makes the trim the right answer there.
    private var servingWindowTokens: Int? {
        if isFoundationEngine { return nil }
        if isMLXEngine {
            return DigestWindow.fromModelDirectory(lock.withLock { currentModelDir })
        }
        return lock.withLock { servedProps.contextSize }
    }

    /// How much of that window a restored conversation may occupy.
    ///
    /// OPTIMISTIC BY DESIGN, in a way worth stating: what is subtracted is what the model may
    /// GENERATE plus a fixed allowance for the system prompt and chat template. Tool DEFINITIONS
    /// are not counted, and a full MCP registry is thousands of tokens - so this bounds the
    /// history and not the whole prompt. The generation reserve is what absorbs the difference
    /// (nothing generates its entire ceiling on the first turn of a restore), and the turn error
    /// on a real overflow is what catches the rest.
    ///
    /// NEVER MORE THAN HALF THE WINDOW FOR OUTPUT. `generationCeiling` is what the model MAY
    /// generate, and it is an absolute number - 4096 by default - so a model whose whole window is
    /// 4096 would reserve all of it and leave a restore the 256-token floor, gutting every
    /// conversation on a small-context engine to protect an output nobody asked for. Half is the
    /// most a reserve can honestly claim when the same window has to hold the conversation the
    /// output is about.
    private var primeBudgetTokens: Int? {
        guard let window = servingWindowTokens else { return nil }
        let reserve = min(DigestSizing.generationCeiling(gen), max(1, window / 2))
        return max(256, window - reserve - PrimePolicy.promptOverheadTokens)
    }

    /// Drop the oldest turns of a verbatim tail until the serving model can hold it.
    ///
    /// Applied to BOTH prime paths - the condensed tail and the full history a decline falls back
    /// to - because the decline is the one that fails: a conversation too large to summarize is
    /// also too large to prime, and priming it anyway is what turned a restore into an HTTP 400 on
    /// the first message afterwards.
    ///
    /// THE PREAMBLE IS NOT PASSED HERE, it is `reserving`. Dropping a digest while keeping the
    /// turns it stands in for is the one thing this must never do, and the way to guarantee that
    /// is for the preamble not to be in the array at all. It still costs its tokens, so the caller
    /// charges them against the budget.
    ///
    /// Nothing happens when the window is unknown. That is the honest answer for an
    /// OpenAI-compatible server that does not report one: trimming against a guess would discard
    /// conversation to satisfy a number nobody stated.
    private func fitTailToServingWindow(_ tail: [Chat.Message], reserving preamble: Int = 0)
        -> (history: [Chat.Message], droppedTurns: Int, droppedBytes: Int)
    {
        guard let budget = primeBudgetTokens else { return (tail, 0, 0) }
        let fitted = DigestPlanner.fit(
            tail,
            budgetTokens: max(0, budget - preamble),
            cost: promptTokenCost(of:),
            // What the CLIENT is told it lost, which is content and not framing.
            bytes: { $0.content.utf8.count },
            // A tool result cannot begin a history: it answers an announcement, and cutting
            // between the two is a transcript a strict chat template refuses outright. The
            // planner advances past them rather than reporting a smaller drop.
            canStartHere: { $0.role != .tool })
        if fitted.droppedTurns > 0 {
            log(
                "session/prime: trimmed \(fitted.droppedTurns) turns to fit a "
                    + "\(max(0, budget - preamble))-token budget "
                    + "(window \(servingWindowTokens.map(String.init) ?? "?"), "
                    + "preamble \(preamble))")
        }
        return fitted
    }

    /// Summarize the older part of `history`, or explain why not.
    ///
    /// NEVER throws: a summarizer that is unavailable, refuses, times out or returns nothing
    /// usable results in the full history being primed with `condensed: false` and a reason the
    /// client can show. That guarantee is `DigestPlanner.condense`'s; this method's job is only to
    /// not undermine it.
    ///
    /// The one thing it does that the planner will not is TRIM, and only when the serving model
    /// has stated a window the result does not fit in. See `fitToServingWindow`.
    private func condenseForPrime(history: [Chat.Message], overrides: CondenseOverrides) async -> (
        history: [Chat.Message], extra: [String: Any]
    ) {
        func declined(_ reason: String) -> ([Chat.Message], [String: Any]) {
            log("session/prime: condense requested but not performed - \(reason)")
            // The full history is what a decline falls back to, and it is the case that overflows:
            // "too large to summarize" and "too large to prime" are the same conversation. The
            // trim is folded into `reason` rather than reported beside it because `reason` is what
            // a client shows when `condensed` is false, and a trim the reader cannot see is a
            // context loss they can only infer from the answers.
            let fitted = fitTailToServingWindow(history)
            guard fitted.droppedTurns > 0 else {
                return (history, ["condensed": false, "reason": reason])
            }
            // WORDED TO FOLLOW THE CLIENT'S OWN LEAD-IN, which it has no way of knowing about.
            // ActionUIChat renders a decline as "Not summarized - the whole conversation was sent
            // to the model instead: <reason>", so a clause reading "the oldest messages were left
            // out" contradicts the sentence it lands in. "and it still did not fit, so ..."
            // continues that sentence instead of arguing with it, and is equally true read alone.
            let noun = fitted.droppedTurns == 1 ? "message" : "messages"
            return (
                fitted.history,
                [
                    "condensed": false,
                    "reason": reason
                        + "; and it still did not fit, so the \(fitted.droppedTurns) oldest "
                        + "\(noun) were left out",
                    "trimmed": ["turns": fitted.droppedTurns, "bytes": fitted.droppedBytes],
                ]
            )
        }

        let turns = digestTurns(from: history)
        let first = chooseSummarizer(history: turns, overrides: overrides)
        guard case .ready(let firstBackend, _, _) = first else {
            if case .unavailable(let reason) = first { return declined(reason) }
            return declined("no summarizer is available")
        }

        // `auto` gets a SECOND chance, and only `auto`. Preferring the on-device model is a
        // judgment about cost, not about reliability: its guided generation fails outright on some
        // ordinary content (measured, repeatably: "Failed to deserialize a Generable type from
        // model output" on a repetitive 7 KB transcript that it summarizes fine when the same text
        // arrives in one block). Falling back to the model already loaded costs one more pass on a
        // restore that was going to prime everything anyway. An EXPLICIT choice gets no fallback -
        // from the CLI or from this restore's `backend`, since both are somebody naming a
        // summarizer - because quietly using the other one would make `summarizer` in the response
        // the only trace of a decision they thought they had made.
        var attempts: [SummarizerOption] = [first]
        if effectiveBackend(overrides) == .auto, firstBackend == nil {
            let session = sessionOption(overrides: overrides)
            if case .ready = session { attempts.append(session) }
        }

        let started = Date()
        // ONE deadline for the whole request, not one per attempt: a client asking "how long can a
        // condensing prime take" deserves a single number, and a fallback that got its own fresh
        // budget would double it.
        let deadline = started.addingTimeInterval(Self.condenseDeadline)
        var lastReason = "the summarizer produced nothing"
        var winner: (result: CondenseResult, summarizer: String)?
        for (index, attempt) in attempts.enumerated() {
            guard case .ready(let backend, let policy, let name) = attempt else { continue }
            let model: DigestModel
            if let backend {
                model = BackendDigestGenerator(
                    backend: backend, name: name, policy: policy,
                    timeout: Self.sessionSliceTimeout, deadline: deadline,
                    log: { [weak self] message in self?.log(message) })
            } else {
                guard
                    let fm = Self.foundationGenerator(
                        policy: policy, deadline: deadline,
                        log: { [weak self] message in self?.log(message) })
                else { return declined("this build has no Foundation Models support") }
                model = fm
            }

            let result = await DigestPlanner.condense(
                history: turns, using: model, policy: policy, generator: name)

            if backend != nil {
                // THE DRAIN. Not tidying - this is the join. On `MLXBackend` `clear()` takes the
                // same PassGate a pass holds, so it cannot return until the last summarization
                // pass has fully torn down, including one the generator's timeout canceled but
                // did not wait for (see BackendDigest.answer). That matters because `finishPrime`
                // is about to build a SECOND backend over the same ModelContainer, and two live
                // passes over one container is the measured [broadcast_shapes] SIGTRAP.
                //
                // On `OpenAIBackend` there is no gate and `clear()` returns immediately - it is
                // only tidying there. Nothing is shared between two of those, so there is nothing
                // to join; if a third engine ever holds process-wide state, it needs its own
                // answer here rather than inheriting this one.
                //
                // The conversation this drops is the summarizer's own; the session's history is
                // whatever we are about to prime.
                await backend?.clear()
            }

            if result.digest != nil {
                winner = (result, name)
                break
            }
            lastReason = result.reason ?? lastReason
            guard index + 1 < attempts.count else { break }
            // A cancel is a decision, not a failure to work around; trying the other model would
            // ignore a Stop the user is waiting on.
            if Task.isCancelled { break }
            // An EMPTY source means the planner refused before it called a model at all - nothing
            // old enough to summarize, or nothing but instructions. Those depend on the history and
            // `keepRecentTurns`, which both attempts share, so the second would refuse identically:
            // it would cost a config.json read, a re-slice, and a `clear()` of the live backend's
            // conversation to arrive at the same answer. Only a summarizer that actually tried and
            // failed is worth replacing.
            guard !result.source.isEmpty else { break }
            // And not if there is no time left to do it in: the deadline is shared, so a second
            // summarizer starting now would spend what remains and arrive at the full-history
            // prime that is already available for free.
            guard deadline.timeIntervalSinceNow >= Self.secondChanceMinimumRemaining else {
                log("session/prime: \(name) failed with too little time left to try another")
                break
            }
            log("session/prime: \(name) did not produce a digest (\(lastReason)); trying next")
        }

        guard let winner, let digest = winner.result.digest else {
            return declined(lastReason)
        }
        let result = winner.result
        let summarizer = winner.summarizer
        digestArchive?.record(
            result, summarizer: summarizer, trigger: "session/prime",
            log: { [weak self] message in self?.log(message) })

        // The verbatim tail is spliced from the ORIGINAL messages so tool-call metadata is not
        // lost in a round trip through DigestTurn; only the turns the planner produced are
        // converted. See DigestSupport.
        let tail = Array(history.dropFirst(result.tailStartIndex))
        // A SUCCESSFUL condensation can still not fit. The verbatim tail is a message COUNT with
        // no size bound, so six turns carrying one large tool result outweigh the digest that
        // replaced sixty. Only the TAIL is offered for trimming; the preamble is charged against
        // the budget instead, so there is no arrangement of the arithmetic in which the digest
        // itself can be dropped while the turns it stands in for are kept.
        let preambleCost = result.injected.reduce(0) {
            $0 + DigestPlanner.estimateTokens($1.content) + DigestPlanner.turnOverheadTokens
        }
        let fitted = fitTailToServingWindow(tail, reserving: preambleCost)
        // The acknowledgment turn was chosen from the tail BEFORE the trim, and a trim can change
        // which turn the tail begins with - so the choice is re-made against what is actually
        // being primed. Left alone, it produces the back-to-back user or assistant turns it exists
        // to prevent.
        let injected = chatMessages(
            from: DigestPlanner.realignAcknowledgment(
                result.injected,
                tailStartsWithUser: fitted.history.first.map { $0.role == .user } ?? true))
        log(
            "session/prime: \(summarizer) summarized \(result.droppedTurns) turns "
                + "(\(result.droppedBytes) bytes) into digest "
                + "\(digest.sourceSHA256.prefix(8)) in "
                + "\(String(format: "%.1f", Date().timeIntervalSince(started))) s; "
                + "kept \(fitted.history.count) verbatim")
        var extra: [String: Any] = [
            "condensed": true,
            // The count `primed`, `dropped.turns` and `trimmed.turns` are all relative to:
            // primeHistory legitimately filters empty turns, orphan tool messages, unknown
            // roles and a trailing unanswered tool announcement, so the client's own message
            // count does NOT close the arithmetic and a client checking it would compute a
            // negative injected count on a transcript with any of those. The identity, which
            // docs/session-prime.md states for clients, is
            // `primed == injected + (accepted - dropped.turns - trimmed.turns)`.
            "accepted": history.count,
            // Which model made it. "The digest is thin" and "the digest was made by a 3B
            // model" are the same observation from the client's side; it should be able to
            // tell them apart.
            "summarizer": summarizer,
            "digest": digest.jsonObject,
            "dropped": ["turns": result.droppedTurns, "bytes": result.droppedBytes],
        ]
        // Reported SEPARATELY from `dropped`, which counts the turns the digest stands in for.
        // These are turns nothing stands in for, and folding them together would let a marker
        // claim they were summarized.
        if fitted.droppedTurns > 0 {
            extra["trimmed"] = ["turns": fitted.droppedTurns, "bytes": fitted.droppedBytes]
        }
        // The preamble is prepended HERE and nowhere else, which is what makes "the digest cannot
        // be trimmed away" structural rather than a rule to remember: it is not in the array the
        // fit was given.
        return (injected + fitted.history, extra)
    }

    /// The on-device generator, behind the two guards that cannot be expressed at a call site that
    /// also has to compile without the framework. nil only in a build that has no FoundationModels
    /// at all - `chooseSummarizer` has already established that the model itself is available.
    private static func foundationGenerator(
        policy: PrimePolicy, deadline: Date, log: @escaping @Sendable (String) -> Void
    ) -> DigestModel? {
        #if canImport(FoundationModels)
            guard #available(macOS 26.0, *) else { return nil }
            // 25 s per slice, not the session path's 120: a slice on the on-device model measures
            // 3-5 s, and this bound is also how long a Stop can go unheard while one is in flight.
            // The deadline is the same one the session path gets - 24 slices at 25 s apiece is ten
            // minutes of held busy slot otherwise, which is the bound this path used to have.
            return FMDigestGenerator(policy: policy, timeout: 25, deadline: deadline, log: log)
        #else
            return nil
        #endif
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
            if kind == .message, self.isMLXEngine { self.assistantTurnBuffer += text }
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
        // NO model picker, on ANY backend - so no client renders one, and session/new's
        // configOptions is always empty.
        //
        // The openai backend never had one: the host app picks the gguf and restarts llama-server
        // itself, so a picker here would be a dead control. The MLX backend DID advertise one,
        // and that is what this removes. It produced a picker under the composer in MLX windows
        // and nothing in llama-server windows - an asymmetry that was the visible half of a real
        // problem: choosing a model there went straight to set_config_option and the HOST APP
        // never saw it. The host app owns the window title, the model label, the RAM advisory and
        // the history stamping, so an agent-side switch left all of them describing a model that
        // was no longer loaded.
        //
        // Model choice belongs to the host app's picker, which is the only place that knows about
        // both engines, the tools choice that goes with a model, RAM headroom, and the window's
        // identity. Same reasoning that retired the mode picker: a live control for something the
        // host app owns can only disagree with it.
        //
        // The MECHANISM stays: session/set_config_option "model" still switches the model (see
        // handleSetConfigOption), and --model still works. Only the advertisement is gone, which
        // is what the host app's picker will drive for an in-place MLX switch.
        //
        // NO chat/agent mode picker. It is deliberately not advertised, so no client renders
        // it: whether a session is agentic is decided ONCE, before the agent is spawned, by
        // whether the host passes --mcp-config (the host's own "use tools" choice). "Chat
        // mode" IS an empty MCP config - same transport, no tools - so a second, live control
        // for the same question could only disagree with the first.
        //
        // It also could not honor the promise it made. The picker only ever appeared when
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

    /// Turn a raw generation-failure string into something a user can act on.
    ///
    /// The logic is `TurnErrorText`, in the MLX-free target where it has tests. It moved there
    /// when the context-overflow case was added: it is pure string work, this file links MLX, and
    /// the file header on `ThinkSplitter` records what that combination cost the last time.
    static func userFacingTurnError(_ raw: String) -> String { TurnErrorText.userFacing(raw) }

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
