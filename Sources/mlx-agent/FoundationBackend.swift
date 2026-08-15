// FoundationBackend.swift - Apple's on-device system model as a GenerationBackend.
//
// The third conformance to the seam Agent runs against, after MLXBackend and OpenAIBackend.
// What it is FOR is narrower than either: the system model is ~3B parameters with a 4096-token
// context, so it is not a replacement for a 7B-30B model under MLX. It earns its place because
// it needs no download, no weights resident, no eligible GPU budget and no server - which makes
// it the answer to "this Mac has no model installed yet" and to short bounded work.
//
// Structurally it differs from the other two in one way that shapes everything below: they own
// their conversation as [Chat.Message] and resend all of it every pass, whereas a
// LanguageModelSession owns an append-only Transcript internally and is fed only the new turn.
// So this backend holds a session rather than an array, and rebuilds it (from its own
// transcript) whenever configuration changes underneath it.
//
// Three gates guard every reference to the framework - compile, link and runtime. See
// FoundationSupport.swift, which is also where availability is decided.
//
// TOOLS ARE INVERTED HERE, and it is the other structural difference worth knowing before
// reading on. The framework calls tools ITSELF from inside respond/streamResponse and has no
// mode that surfaces a call for external dispatch, so Agent's loop cannot drive this backend.
// Instead the loop hands it a runner (see InternalToolDispatchingBackend) and the bridged tools
// reach back into it, which puts the permission gate, timeout, truncation and dedup on exactly
// the same code as every other backend. Consequences: a turn is ONE pass, no .toolCall events
// are ever emitted, and the iteration cap is enforced here rather than by the loop. See
// FoundationTools.swift.

import AgentDigest
import AgentText
import Foundation
import MLXLMCommon

#if canImport(FoundationModels)
    import FoundationModels
#endif

/// Apple's on-device system model, behind the same seam as MLX and llama-server.
///
/// `@unchecked Sendable` with an NSLock, matching the other two backends. `LanguageModelSession`
/// is itself `@unchecked Sendable` and its respond methods are `nonisolated(nonsending)`, so
/// nothing here is forced onto an actor; the lock guards this class's own fields.
final class FoundationBackend: GenerationBackend, InternalToolDispatchingBackend, @unchecked Sendable
{

    private let instructions: String?
    private let parameters: GenerateParameters
    private let lock = NSLock()
    private var toolSpecs: [ToolSpec]?
    private let log: @Sendable (String) -> Void

    /// Reaches back into `Agent.runToolForBackend`. Installed once, at Agent init, before any
    /// pass - so a nil runner means tools were configured on a backend no agent is driving,
    /// and the tool set is simply not built.
    private var toolRunner: (@Sendable (ToolCall) async -> ToolRunOutcome)?
    /// `AgentGuardrails.maxToolIterations`, arriving with the runner. The default is only what
    /// stands until then; nothing dispatches before the install.
    private var maxToolCallsPerPass = 10
    /// Reset at the top of every pass. Counts what the MODEL asked for, including calls the
    /// permission gate went on to deny - a model looping on a tool it is not allowed to use is
    /// exactly the runaway the cap exists to stop.
    ///
    /// Counted at the TOP of the runner, before the tool runs, so an increment always lands in
    /// the attempt that asked for it.
    private var toolCallsThisPass = 0
    /// Model-facing tool output so far this attempt, in bytes. Decides whether an overflow is
    /// worth condensing for - see `runPass` - and bounds what the tools can spend of the window.
    ///
    /// Budget is RESERVED when a call starts and reconciled when it returns, both stamped with
    /// `attemptGeneration`. The stamp is what makes it safe under a framework that dispatches
    /// concurrently and a timeout that cannot be cancelled: a call outliving its attempt has
    /// nothing to give back, because its reservation went with the attempt.
    private var toolResultBytesThisPass = 0
    /// Bumped whenever the per-attempt counters reset. See `toolResultBytesThisPass`.
    private var attemptGeneration = 0
    /// So the pass-budget refusal is logged once rather than per call.
    private var warnedResultBudget = false
    /// What the current tool set costs, from `FoundationToolBridge.tokenCost`. Kept rather than
    /// only logged, because the proactive condense has to reserve it.
    private var toolTokenCost = 0
    /// Bumped on every tool-set rebuild. The exact token count is measured asynchronously, so
    /// without a stamp its Task can land AFTER the tools it measured were replaced or turned off
    /// - reinstating a reserve for a tool set the model can no longer see, which is the very
    /// thing zeroing `toolTokenCost` is there to prevent. Two rebuilds racing would settle in
    /// completion order for the same reason.
    private var toolsGeneration = 0

    #if canImport(FoundationModels)
        /// `[MCPBridgedTool]`, typed as Any for the same reason `sessionBox` is: a stored
        /// property cannot be conditionally available.
        private var bridgedToolsBox: Any?
    #endif

    /// Serializes whole passes. The framework throws `GenerationError.concurrentRequests` when
    /// two overlap, and the ACP cancel window makes that reachable with a well-behaved client -
    /// see PassGate.swift for the sequence.
    private let gate = PassGate()

    /// The conversation this backend owns, in our currency rather than the framework's.
    ///
    /// It exists because a session cannot be reconfigured in place - changing tools, clearing, or
    /// recovering from a canceled pass all mean constructing a new one - and because the
    /// framework's own transcript is not a usable substitute: a CANCELED exchange is purged from
    /// it entirely (measured), so reading history back out of the session would silently lose
    /// exactly the turns that need preserving. Completed turns are appended here as they finish.
    private var seedHistory: [Chat.Message]

    /// Policy for summarize-and-retry on context overflow. nil turns it off, and overflow then
    /// surfaces as a clean turn failure.
    ///
    /// Not `@available`-gated and not an FM type, which is exactly why the decision of WHAT to
    /// summarize is testable: see AgentDigest.
    private let digestPolicy: PrimePolicy?
    /// Hard bound on one slice of summarization. See `FMDigestGenerator.timeout` for why a bound
    /// is not optional here.
    private let digestTimeout: TimeInterval
    /// Where to record a condensation, when `--digest-dir` asked for one. nil is the default and
    /// means the log line is the only trace.
    private let archive: DigestArchive?

    #if canImport(FoundationModels)
        /// Typed as Any so this stored property does not need `@available` (a stored property
        /// cannot be conditionally available). Every use casts back under `#available`, which is
        /// the same trick the rest of the file uses to keep the class itself unrestricted.
        private var sessionBox: Any?
    #endif

    /// - Parameters:
    ///   - parameters: the same GenerateParameters the other backends get; the sampling knobs
    ///     that have a Foundation Models equivalent are mapped in `generationOptions`.
    ///   - systemPrompt: becomes the session's instructions; nil supplies none (translator mode).
    ///   - seedHistory: prior turns to resume from (session/prime), system message excluded.
    ///   - digestPolicy: how to summarize when the window overflows; nil to fail instead.
    init(
        parameters: GenerateParameters, systemPrompt: String?, seedHistory: [Chat.Message] = [],
        digestPolicy: PrimePolicy? = FoundationBackend.defaultDigestPolicy(),
        // 25 s, not 45: a slice measures 3-5 s, and this is also how long a Stop can go unheard
        // while one is in flight. Five times the observed worst case is enough headroom.
        digestTimeout: TimeInterval = 25,
        archive: DigestArchive? = nil,
        log: @escaping @Sendable (String) -> Void = { _ in }
    ) {
        self.parameters = parameters
        self.instructions = systemPrompt
        self.seedHistory = seedHistory
        self.digestPolicy = digestPolicy
        self.digestTimeout = digestTimeout
        self.archive = archive
        self.log = log
        rebuildSession()
        // Say what silently will not happen. Every other GenerateParameters knob maps onto
        // something here; this one has no equivalent in the framework at all.
        if parameters.repetitionPenalty != nil {
            log(
                "foundation backend: --repetition-penalty has no equivalent in Foundation Models "
                    + "and is ignored")
        }
    }

    var tools: [ToolSpec]? {
        get { lock.withLock { toolSpecs } }
        set {
            lock.withLock { toolSpecs = newValue }
            // A session's tool set is fixed at construction, so changing it means building a new
            // one. Free of history loss: the rebuild replays `seedHistory`, which this backend
            // owns precisely because the framework's own transcript cannot be relied on.
            //
            // NOT gated on the PassGate, unlike `clear()` - and that is safe only because of who
            // calls this. `setToolsEnabled` runs when a session stack is built or the mode
            // changes, never during a pass. A caller that set tools AROUND a live pass (the shape
            // `BackendDigest` uses: save, set nil, restore in a defer) would swap the session out
            // from under it and the answer in flight would be written into an orphaned one. That
            // path cannot reach here today - `sessionOption` refuses backend-summarizing on the
            // foundation engine and `DigestMode` only wraps MLX and OpenAI - so this is the
            // invariant to preserve rather than a bug to fix. If it ever stops holding, gate it.
            rebuildToolsAndSession()
        }
    }

    // MARK: - Tools

    /// Take the loop's tool dispatcher. See `InternalToolDispatchingBackend`.
    func installToolRunner(
        maxCallsPerPass: Int,
        _ runner: @escaping @Sendable (ToolCall) async -> ToolRunOutcome
    ) {
        lock.withLock {
            toolRunner = runner
            // A cap below 1 would refuse the first call and leave the model insisting on a tool
            // it can never get - worse than no tools at all, which is what 0 presumably meant.
            maxToolCallsPerPass = max(1, maxCallsPerPass)
        }
        rebuildToolsAndSession()
    }

    /// Rebuild the bridged tool set from the current specs, then the session that holds it.
    ///
    /// Called from the `tools` setter and from `installToolRunner`, because either can arrive
    /// first: Agent installs the runner in its init, and ACPServer sets the specs when the mode
    /// is decided. Whichever is second is what produces a usable tool set.
    private func rebuildToolsAndSession() {
        #if canImport(FoundationModels)
            if #available(macOS 26.0, *) {
                let (specs, runner, generation) = lock.withLock {
                    () -> ([ToolSpec], (@Sendable (ToolCall) async -> ToolRunOutcome)?, Int) in
                    toolsGeneration += 1
                    return (toolSpecs ?? [], toolRunner, toolsGeneration)
                }
                guard !specs.isEmpty, let runner else {
                    let had = lock.withLock { () -> Bool in
                        let had = bridgedToolsBox != nil
                        bridgedToolsBox = nil
                        // Back to zero with the tools. Leaving it set makes `needsCondensingBefore`
                        // reserve room for a tool set the model can no longer see - so switching
                        // to chat mode after using tools would condense conversations that fit
                        // comfortably, which is exactly what that reserve's own comment warns
                        // against.
                        toolTokenCost = 0
                        return had
                    }
                    // Only rebuild if there were tools to remove; otherwise this is the ordinary
                    // chat path and rebuilding would throw away a live session for nothing.
                    if had { rebuildSession() }
                    return
                }

                let built = FoundationToolBridge.tools(
                    from: specs, runner: gatedRunner(runner), log: { [log] in log($0) })
                lock.withLock {
                    bridgedToolsBox = built
                    // A rough figure NOW rather than nothing until the exact count arrives.
                    // `tokenCount` is async, and a prompt that lands before it returns would
                    // otherwise reserve zero for the tool set - the very under-reserve this is
                    // here to prevent. It is replaced as soon as the Task below finishes.
                    //
                    // Estimated from what was actually BUILT, not from every spec: a tool whose
                    // schema the framework refused is not offered to the model, and reserving
                    // window for it would be reserving for nothing.
                    toolTokenCost = FoundationToolBridge.estimatedTokenCost(
                        of: specs, built: built)
                }
                rebuildSession()

                // The window is the binding constraint on this backend, so say what the tool set
                // costs out of it before the conversation starts rather than after it overflows.
                let model = SystemLanguageModel.default
                Task { [weak self, log] in
                    guard let cost = await FoundationToolBridge.tokenCost(of: built) else { return }
                    // Stored, not just logged: `needsCondensingBefore` has to reserve it, and a
                    // reserve that ignores a third of the window is not a reserve.
                    guard let self else { return }
                    self.lock.withLock {
                        // Only if these are still the tools in force. Without the stamp, a mode
                        // switch to chat between the rebuild and this line would be undone here.
                        guard self.toolsGeneration == generation else { return }
                        self.toolTokenCost = cost
                    }
                    let percent = Int((Double(cost) / Double(model.contextSize) * 100).rounded())
                    log(
                        "foundation backend: \(built.count) tool(s) cost \(cost) of the "
                            + "\(model.contextSize)-token window (\(percent)%) before any "
                            + "conversation")
                }
                return
            }
        #endif
        if (lock.withLock { toolSpecs })?.isEmpty == false {
            log(
                "foundation backend: tools require macOS 26 and the Foundation Models framework; "
                    + "this turn behaves as chat")
        }
    }

    /// How much of a tool result the MODEL is shown, in bytes.
    ///
    /// `AgentGuardrails.resultByteBudget` defaults to 32 KB, which is eight times this model's
    /// entire 4096-token window - measured: a 31 KB result, comfortably inside that guardrail,
    /// throws `exceededContextWindowSize` reporting 7149 tokens. And the overflow lands where
    /// nothing can recover it: the oversized text is in the FRAMEWORK's transcript, never in
    /// `seedHistory`, so condensing summarizes a history that was not the problem, the retry
    /// calls the same tool, and the turn fails after spending seconds on it.
    ///
    /// 1500 bytes is roughly 375 tokens, under a tenth of the window. The CLIENT still sees the
    /// full output - `Agent.dispatch` has already emitted it - so this shortens what the model
    /// reads, not what the user gets. mcp_server_fetch alone defaults to 5000 characters, so
    /// this is one ordinary tool away rather than a hypothetical.
    static let modelFacingResultBytes = 1500

    /// And the same bound across a whole pass.
    ///
    /// Clamping each call is not enough on its own: the iteration cap allows ten of them, so ten
    /// clamped results still put 15 KB - about 3750 tokens - into a 4096-token window. Past this
    /// total the model is told the budget is spent, which is the same shape of answer the
    /// iteration cap gives and something it can act on.
    static let passResultByteBudget = 6000

    /// Cut a tool result down to what this window can carry, and say so in-band so the model
    /// knows the answer is partial rather than the whole truth.
    ///
    /// Returns the model-facing byte count alongside, so the caller can keep the pass total.
    private static func clamped(
        _ outcome: ToolRunOutcome, log: @Sendable (String) -> Void
    ) -> (ToolRunOutcome, Int) {
        guard case .result(let text) = outcome else { return (outcome, 0) }
        guard text.utf8.count > modelFacingResultBytes else { return (outcome, text.utf8.count) }
        let kept = truncateToBudget(text, modelFacingResultBytes)
        log(
            "foundation backend: a tool returned \(text.utf8.count) bytes; the model is shown "
                + "\(kept.utf8.count) of them (the whole result reached the client)")
        return (.result(kept), kept.utf8.count)
    }

    /// Wrap Agent's runner with the per-pass iteration cap.
    ///
    /// The cap has to live here because only this backend knows where a pass begins - and it is
    /// necessarily SOFTER than the loop's version. `Agent.runTurn` can refuse to start another
    /// pass; nothing can halt the framework mid-`streamResponse` without discarding the answer
    /// in progress. So the budget is reported to the model as a tool result and the model is
    /// asked to conclude, which it can act on. That is a real difference in behavior between
    /// backends, and it is recorded in docs/foundation-models.md rather than hidden.
    private func gatedRunner(
        _ runner: @escaping @Sendable (ToolCall) async -> ToolRunOutcome
    ) -> @Sendable (ToolCall) async -> ToolRunOutcome {
        return { [weak self] call in
            guard let self else { return .result("{\"error\": \"backend went away\"}") }
            let (count, cap) = self.lock.withLock { () -> (Int, Int) in
                self.toolCallsThisPass += 1
                return (self.toolCallsThisPass, self.maxToolCallsPerPass)
            }
            guard count <= cap else {
                // Once only. A model that keeps asking past the budget gets the same refusal
                // every time, and a line per attempt would bury everything else in the log.
                if count == cap + 1 {
                    self.log(
                        "foundation backend: tool-call budget of \(cap) reached in one pass; "
                            + "telling the model to answer with what it has")
                }
                return .budgetExhausted(
                    "{\"error\": \"tool call limit reached; answer with what you have\"}")
            }
            // The pass budget is checked BEFORE dispatching, so a tool whose output cannot be
            // read is not run at all - the server is spared the work and the user is spared a
            // side effect whose result the model will never see.
            //
            // And the budget is RESERVED at the check rather than added at the end. Calls arrive
            // concurrently here, so a plain check-then-add lets every call in a batch read the
            // same "spent" before any of them adds: three concurrent calls would all pass a
            // budget with room for one. Each claims a whole call's worth up front and gives back
            // the difference once its real size is known.
            let (spent, alreadyWarned, attempt) = self.lock.withLock { () -> (Int, Bool, Int) in
                let spent = self.toolResultBytesThisPass
                let warned = self.warnedResultBudget
                if spent >= Self.passResultByteBudget {
                    self.warnedResultBudget = true
                } else {
                    self.toolResultBytesThisPass += Self.modelFacingResultBytes
                }
                return (spent, warned, self.attemptGeneration)
            }
            guard spent < Self.passResultByteBudget else {
                // Once per pass, like the iteration cap: a model that keeps asking gets the same
                // refusal every time, and a line each would bury everything else.
                if !alreadyWarned {
                    self.log(
                        "foundation backend: tool output budget of \(Self.passResultByteBudget) "
                            + "bytes spent in one pass; telling the model to answer with what it "
                            + "has")
                }
                return .budgetExhausted(
                    "{\"error\": \"tool output limit reached for this turn; answer with what you "
                        + "have\"}")
            }
            let (outcome, bytes) = Self.clamped(await runner(call), log: self.log)
            self.lock.withLock {
                // Reconcile against the reservation above - but ONLY against the attempt that
                // made it. The reconcile is always a refund (`bytes` never exceeds the
                // reservation), so a straggler landing after its attempt was torn down would
                // credit an attempt that never reserved anything, driving the counter negative
                // and disabling the budget outright: ten stragglers refunding ~1400 each leave
                // the next attempt near -14000 against a 6000 bound. The reservation died with
                // its attempt; there is nothing to give back.
                guard self.attemptGeneration == attempt else { return }
                self.toolResultBytesThisPass += bytes - Self.modelFacingResultBytes
            }
            return outcome
        }
    }

    /// Reset the conversation, keeping instructions.
    ///
    /// Serialized against passes like MLXBackend's, so it cannot swap the session out from under
    /// a stream in flight - that pass would go on writing into an orphaned session and its answer
    /// would vanish.
    func clear() async {
        await withPass(gate) {
            rebuildSession(replacingHistoryWith: [])
        }
    }

    // MARK: - Session lifecycle

    /// Sized from the model's actual window rather than a constant.
    ///
    /// The arithmetic - and the reasoning about what the window has to hold at once - lives in
    /// `PrimePolicy.sized(window:)`, because the session-backed summarizer needs exactly the same
    /// derivation from a different window. `PrimePolicy` clamps whatever comes out, so a future
    /// model with a tiny window degrades to the floor instead of producing a negative budget.
    ///
    /// `maxSlices` has to move with the budget, not stay at the library default. A slice here is
    /// about 1900 tokens against the library's 2800, so the same conversation needs ~45% more
    /// slices - leaving the count alone would make this backend refuse to condense histories the
    /// library's own sizing considers well within range, and the pass would then fail on the very
    /// overflow the condense exists to prevent. 24 slices is roughly 180 KB of conversation, and
    /// at the measured 3-5 s per slice it is also the point where the wait stops being defensible.
    ///
    /// The default is what a person is waiting on: a mid-turn overflow condense or a restore. The
    /// offline `digest` mode raises it, because nothing there is interactive and the only cost of
    /// another slice is time in a script.
    static func defaultDigestPolicy(maxSlices: Int = 24) -> PrimePolicy {
        .sized(window: FMAvailability.contextSize() ?? 4096, maxSlices: maxSlices)
    }

    /// Build a fresh session from `instructions` + `seedHistory`, optionally replacing the history
    /// in the same breath.
    ///
    /// Cheap: it allocates a session and a transcript, and loads nothing. The system model is
    /// already resident - that being true is most of the reason this backend exists.
    ///
    /// Replaces, reads and publishes under ONE lock acquisition. Doing it as separate steps leaves
    /// a window in which a concurrent rebuild publishes a session built from older history, which
    /// is the kind of race that shows up once a month as a lost turn. Building the transcript
    /// inside the lock is fine - it allocates and does no I/O.
    private func rebuildSession(replacingHistoryWith newHistory: [Chat.Message]? = nil) {
        #if canImport(FoundationModels)
            if #available(macOS 26.0, *) {
                lock.withLock {
                    if let newHistory { seedHistory = newHistory }
                    let bridged = bridgedToolsBox as? [MCPBridgedTool] ?? []
                    sessionBox = LanguageModelSession(
                        tools: bridged as [any FoundationModels.Tool],
                        transcript: Self.transcript(
                            instructions: instructions, history: seedHistory))
                }
                return
            }
        #endif
        // No framework, or too old for it: there is no session to build, but a caller asking to
        // replace the history still gets that. Falling out of the #if without doing it would make
        // `clear()` silently do nothing on macOS 14.
        if let newHistory { lock.withLock { seedHistory = newHistory } }
    }

    #if canImport(FoundationModels)
        /// Translate our conversation currency into the framework's.
        ///
        /// Tool turns are DROPPED rather than reconstructed. `Transcript` can express tool calls
        /// and outputs, but a RESTORED exchange refers to tools by name, and the set this session
        /// is given depends on the mode and the servers that launched - so a replayed call to a
        /// tool that is not in it shows the model itself using something it cannot use. An honest
        /// gap is better. ChatView's prime wire carries text turns only in practice, so in the
        /// common path nothing is lost.
        @available(macOS 26.0, *)
        static func transcript(instructions: String?, history: [Chat.Message]) -> Transcript {
            var entries: [Transcript.Entry] = []
            // No toolDefinitions here, and that is measured rather than assumed. Building a
            // session as `LanguageModelSession(tools:transcript:)` makes the framework put the
            // definitions into its own transcript from the `tools:` argument: a session built
            // from a transcript whose instructions entry carries NONE still ends up with all of
            // them, and `tokenCount(for: session.transcript)` is identical either way (522 for
            // the same 3 tools, both ways). So passing them here is redundant, not load-bearing
            // - and the model does call tools without it, which was the thing worth checking.
            if let instructions, !instructions.isEmpty {
                entries.append(
                    .instructions(
                        Transcript.Instructions(
                            segments: [.text(Transcript.TextSegment(content: instructions))],
                            toolDefinitions: [])))
            }
            for message in history {
                // Skip empty turns instead of emitting an empty text segment. This is not
                // hypothetical tidiness: a primed tool-calling turn arrives as an assistant
                // message whose CONTENT is "" and whose payload is the tool call, and since tool
                // turns are dropped here that left a bare empty response in the transcript.
                // Measured effect: the next prompt returned NOTHING AT ALL - not a refusal, an
                // empty answer - which reads as a hang rather than a gap.
                guard !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                else { continue }
                let segments: [Transcript.Segment] = [
                    .text(Transcript.TextSegment(content: message.content))
                ]
                switch message.role {
                case .user:
                    entries.append(.prompt(Transcript.Prompt(segments: segments)))
                case .assistant:
                    entries.append(
                        .response(Transcript.Response(assetIDs: [], segments: segments)))
                case .system:
                    // A system turn inside `history` is a caller error - the session prepends
                    // its own instructions - but folding it in beats dropping it silently.
                    entries.append(
                        .instructions(
                            Transcript.Instructions(segments: segments, toolDefinitions: [])))
                case .tool:
                    continue
                }
            }
            return Transcript(entries: entries)
        }

        /// Map the shared sampling knobs onto the framework's options.
        ///
        /// `seed` IS honored - both random sampling modes take one - which matters because it is
        /// the only way to make a run reproducible. `repetitionPenalty` has no equivalent at all
        /// and is dropped; that is logged once at init rather than silently, since a caller who
        /// set it deserves to know it does nothing here.
        @available(macOS 26.0, *)
        private var generationOptions: GenerationOptions {
            let seed = parameters.seed
            let sampling: GenerationOptions.SamplingMode
            if parameters.temperature == 0 {
                sampling = .greedy
            } else if parameters.topP < 1 {
                sampling = .random(probabilityThreshold: Double(parameters.topP), seed: seed)
            } else {
                sampling = .random(top: 50, seed: seed)
            }
            return GenerationOptions(
                sampling: sampling,
                temperature: parameters.temperature == 0 ? nil : Double(parameters.temperature),
                // The model's whole context is 4096, so an unbounded request cannot be honored
                // anyway; capping keeps a runaway generation from spending the window.
                maximumResponseTokens: parameters.maxTokens)
        }
    #endif

    // MARK: - Streaming

    func stream(_ new: [Chat.Message]) -> AsyncThrowingStream<BackendEvent, Error> {
        // Reduced to a String BEFORE the Task, for two reasons. Chat.Message is not Sendable, so
        // capturing the array in a Task closure does not compile under strict concurrency (the
        // other backends dodge this by appending to their own state under lock first). And tool
        // results have nowhere to go until tools are bridged - the framework consumed any tool
        // call itself - so anything that is not user text is dropped here rather than pasted
        // into the prompt as if the model had asked for it.
        let prompt =
            new
            .filter { $0.role == .user }
            .map(\.content)
            .joined(separator: "\n\n")

        return AsyncThrowingStream { continuation in
            let task = Task { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                do {
                    try await self.runPass(prompt: prompt, continuation: continuation)
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func runPass(
        prompt: String, continuation: AsyncThrowingStream<BackendEvent, Error>.Continuation
    ) async throws {
        #if canImport(FoundationModels)
            guard #available(macOS 26.0, *) else {
                throw Self.unavailable(FMAvailability.probe().summary)
            }
            guard !prompt.isEmpty else {
                // Nothing to say. Finish cleanly rather than sending an empty prompt, which the
                // framework rejects.
                return
            }

            try await withPass(gate) {
                var answer = ""
                var completed = false
                // Only turns that actually reached the model are recorded. A pass that failed
                // before generating - no session, unavailable - would otherwise append a bare
                // user turn with no reply, and repeated failures would stack consecutive user
                // turns into the transcript.
                var reachedModel = false
                // A pass that does NOT reach the end - canceled, or failed mid-answer - leaves
                // the framework's session in a state that must not be reused. See recordTurn.
                defer {
                    if reachedModel {
                        recordTurn(prompt: prompt, answer: answer, completed: completed)
                    }
                }

                // Proactive condense. Cheaper than the reactive path in the case that matters:
                // overflow raised MID-ANSWER cannot be retried (see below), and this is what keeps
                // a long conversation from reaching that state at all.
                var triedCondensing = false
                if await needsCondensingBefore(prompt: prompt) {
                    // A success here also disarms the reactive retry: if the pass still overflows
                    // after this, condensing again would summarize the preamble this one just
                    // wrote - a digest of a digest, losing a second time to buy nothing.
                    triedCondensing = await condenseHistory(
                        because: "the next turn would exceed the window")
                }

                while true {
                    // The tool budget is per ATTEMPT, and this is where one begins - inside the
                    // retry loop, not above it. A condense between attempts builds a NEW session,
                    // so the previous attempt's tool output is not in the transcript the retry
                    // runs against; carrying its counters forward would bill the retry for bytes
                    // that no longer occupy the window, and could make it report that tools
                    // overflowed a session holding almost none. Inside the gate either way, so
                    // two queued prompts still cannot spend each other's budget.
                    lock.withLock {
                        toolCallsThisPass = 0
                        toolResultBytesThisPass = 0
                        warnedResultBudget = false
                        attemptGeneration += 1
                    }
                    // Re-read inside the loop: a condense between attempts replaces the session.
                    guard let session = lock.withLock({ sessionBox }) as? LanguageModelSession
                    else { throw Self.unavailable(FMAvailability.probe().summary) }
                    reachedModel = true

                    var delta = CumulativeDelta()
                    // What the READER actually has, accumulated from the yields themselves rather
                    // than read back from CumulativeDelta. Those differ: `emittedText` is the last
                    // SNAPSHOT, and a snapshot that rewrites or truncates already-sent text
                    // replaces it - an empty snapshot after "Hello" leaves `emittedText` empty
                    // while the client is still holding "Hello". Both decisions below turn on this
                    // value, and both get it wrong in the dangerous direction if it lies.
                    var streamed = ""
                    // Publish the partial answer on EVERY exit from this scope, including the
                    // cancellation that unwinds past the catch below. Without it, the outer defer
                    // would record an empty answer and the reader's visible text would vanish from
                    // the model's context.
                    defer { answer = streamed }
                    let started = Date()
                    var firstToken: Date?
                    do {
                        // Snapshots are CUMULATIVE - each carries the whole answer so far - while
                        // BackendEvent.text is a fragment. CumulativeDelta is the adapter, and it
                        // lives in the MLX-free target so its Unicode edge cases are tested.
                        for try await snapshot in session.streamResponse(
                            to: prompt, options: generationOptions)
                        {
                            try Task.checkCancellation()
                            let text = delta.advance(to: snapshot.content) { [log] anomaly in
                                log("foundation backend: \(anomaly)")
                            }
                            guard !text.isEmpty else { continue }
                            if firstToken == nil { firstToken = Date() }
                            // Always .text, never .reasoning: the system model emits no thought
                            // markers, so ThinkSplitter passes this through untouched.
                            continuation.yield(.text(text))
                            streamed += text
                        }
                    } catch let error as LanguageModelSession.GenerationError {
                        // Overflow is the one error a caller can act on. Retry ONCE, and only
                        // while nothing has reached the client: text already streamed cannot be
                        // un-sent, so a second attempt would append a fresh answer to a partial
                        // one and the reader would see two half-replies stitched together.
                        // Overflow raised mid-answer (the response itself filled the window)
                        // therefore surfaces as a failure, which is honest - the request cannot
                        // be satisfied by making the history smaller.
                        // ... and only when the history is plausibly what overflowed.
                        //
                        // Tool output lives in the FRAMEWORK's transcript, never in `seedHistory`,
                        // so condensing cannot remove a byte of it: where tools filled the window,
                        // summarizing spends seconds on the wrong thing, calls the same tools
                        // again and fails anyway. But "a tool ran" is far too broad a test for
                        // that now the results are clamped - one small call contributes a few
                        // hundred tokens, and an overflow after it is almost certainly history
                        // plus the tool DEFINITIONS, which condensing does fix. So the test is
                        // whether tools actually spent their budget.
                        let toolOverflow =
                            lock.withLock { toolResultBytesThisPass } >= Self.passResultByteBudget
                        if case .exceededContextWindowSize = error, !triedCondensing,
                            streamed.isEmpty, !toolOverflow
                        {
                            triedCondensing = true
                            if await condenseHistory(because: "the window overflowed") {
                                continue
                            }
                        }
                        // Only when the model never spoke. An overflow raised MID-ANSWER is the
                        // response filling the window - measured, with a model that echoed a tool
                        // result back at 4000 words - and blaming the tool for that would send the
                        // reader after the wrong thing entirely.
                        if case .exceededContextWindowSize = error, toolOverflow, streamed.isEmpty {
                            throw Self.generationFailed(
                                "the tools this turn returned more than the on-device model's "
                                    + "\(FMAvailability.contextSize() ?? 4096)-token window can "
                                    + "hold, and summarizing cannot recover it - ask for less, or "
                                    + "use --backend mlx for this conversation")
                        }
                        throw Self.generationFailed(fmUserFacingError(error))
                    } catch let error as LanguageModelSession.ToolCallError {
                        // A bridged tool threw. The one case that is not a failure is the user
                        // answering a permission prompt with Cancel: `MCPBridgedTool.call` throws
                        // CancellationError for that, because throwing is the only way to stop a
                        // pass from inside a tool, and the framework wraps whatever it threw. Left
                        // wrapped it would surface as a failed turn - the client would see an
                        // error where it asked for a cancel.
                        if error.underlyingError is CancellationError { throw CancellationError() }
                        throw Self.generationFailed(
                            "the tool \(error.tool.name) failed: "
                                + fmUserFacingError(error.underlyingError))
                    }
                    // "Did not throw" is NOT "finished". Canceling the task iterating
                    // `streamResponse` usually makes the framework END the stream early rather
                    // than throw, so `checkCancellation` above only fires if another snapshot
                    // happens to arrive after the cancel. Asking directly is what guarantees a
                    // canceled pass still forces the session rebuild in recordTurn - without it
                    // the next prompt reuses a post-cancel session, which traps and kills the
                    // process.
                    completed = !Task.isCancelled

                    let finished = Date()
                    // Token counts are 26.4+; below that estimate rather than report zeros, since
                    // Agent derives tok/s from these and a zero reads as a stall.
                    let promptTokens = await Self.tokenCount(for: prompt) ?? (prompt.utf8.count / 4)
                    let answerTokens =
                        await Self.tokenCount(for: streamed) ?? (streamed.utf8.count / 4)
                    let decodeStart = firstToken ?? finished
                    continuation.yield(
                        .info(
                            GenerateCompletionInfo(
                                promptTokenCount: promptTokens,
                                generationTokenCount: answerTokens,
                                promptTime: decodeStart.timeIntervalSince(started),
                                generationTime: finished.timeIntervalSince(decodeStart))))
                    return
                }
            }
        #else
            throw Self.unavailable("this build has no Foundation Models support")
        #endif
    }

    #if canImport(FoundationModels)
        /// Real token count where the OS offers one, else nil so the caller can estimate.
        @available(macOS 26.0, *)
        private static func tokenCount(for text: String) async -> Int? {
            guard #available(macOS 26.4, *) else { return nil }
            return try? await SystemLanguageModel.default.tokenCount(for: text)
        }
    #endif

    // MARK: - Condensing

    #if canImport(FoundationModels)
        /// Would the next pass not fit?
        ///
        /// Two-stage on purpose. The byte estimate is free and answers "no" for every short
        /// conversation, which is nearly all of them; only when it says the window is at risk do we
        /// spend an actual `tokenCount` call to confirm. Asking the OS on every turn would put a
        /// real API round trip in front of every prompt to answer a question whose answer is
        /// almost always the same.
        ///
        /// Pre-26.4 there is no exact counter, so the estimate decides alone. `estimateTokens`
        /// takes the larger of bytes/4 and the non-ASCII scalar count precisely so that case errs
        /// toward condensing early rather than late - a bytes-only rule under-reports CJK by about
        /// a third, and under-reporting here means arriving at the un-retryable mid-answer
        /// overflow instead of condensing before it.
        @available(macOS 26.0, *)
        private func needsCondensingBefore(prompt: String) async -> Bool {
            guard digestPolicy != nil else { return false }
            let history = lock.withLock { seedHistory }
            guard !history.isEmpty else { return false }
            let text = history.map(\.content).joined(separator: "\n") + "\n" + prompt
            let window = FMAvailability.contextSize() ?? 4096
            // Room for the answer, plus the instructions and template the framework adds.
            //
            // The answer allowance is CAPPED at a quarter of the window rather than taken from
            // `maxTokens` directly. ACP's default is 4096 - the entire window - which is a "no
            // practical limit" sentinel rather than a prediction, and reserving all of it would
            // make this check fire on literally every turn that has any history at all, condensing
            // conversations that fit comfortably. A caller who asks for SHORT answers still gets
            // the smaller reserve, which is the case where the number means something.
            //
            // The tool set is part of the reserve too, and it is not small: measured at 615
            // tokens for 3 tools and 1832 for 14, out of 4096. Leaving it out made this check
            // stop firing on exactly the configuration the tool bridge introduced - it would
            // conclude there was room, and the pass would then overflow for real.
            let reserve =
                min(parameters.maxTokens ?? 512, window / 4) + 256
                + lock.withLock { toolTokenCost }
            guard DigestPlanner.estimateTokens(text) + reserve > window else { return false }
            if let exact = await Self.tokenCount(for: text) { return exact + reserve > window }
            return true
        }

        /// Summarize the older part of the conversation and rebuild the session around it.
        ///
        /// THE SUMMARIZER HERE IS ALWAYS FMDigestGenerator, AND MUST BE. `BackendDigestGenerator`
        /// (BackendDigest.swift) summarizes with the session's own model and looks like the obvious
        /// generalization of this call, but it cannot run here: this method is reached from inside
        /// a pass, `PassGate.acquire()` is non-reentrant and non-cancellable, so calling a
        /// backend's `stream()` from within its own pass deadlocks permanently - no timeout, no
        /// cancel. FMDigestGenerator is safe precisely because it opens its own
        /// `LanguageModelSession` and never touches this backend's gate. Backend-backed
        /// summarization is a prime-time capability; see the header of BackendDigest.swift.
        ///
        /// Returns false when nothing changed - the caller must then treat the situation as it
        /// would have without this path, because the history is exactly as it was. Every reason
        /// for a false is logged: this is the only trace, and a condensation that silently did not
        /// happen looks identical to one that did nothing useful.
        @available(macOS 26.0, *)
        private func condenseHistory(because reason: String) async -> Bool {
            guard let policy = digestPolicy else { return false }
            let history = lock.withLock { seedHistory }
            guard !history.isEmpty else { return false }

            let result = await DigestPlanner.condense(
                history: digestTurns(from: history),
                using: FMDigestGenerator(policy: policy, timeout: digestTimeout, log: log),
                policy: policy,
                generator: FMDigestGenerator.name)

            guard let digest = result.digest else {
                log(
                    "foundation backend: \(reason), but the history was left whole - "
                        + (result.reason ?? "no reason given"))
                return false
            }
            archive?.record(
                result, summarizer: FMDigestGenerator.name, trigger: "context overflow",
                log: log)

            // The verbatim tail is spliced from the ORIGINAL messages, not from the planner's
            // round-tripped copies. DigestTurn carries role and text only, so rebuilding the tail
            // from it would quietly discard tool-call metadata on any turn that had it. Only the
            // turns the planner PRODUCED are converted, and it reports both halves rather than
            // leaving this to arithmetic on a count.
            let tail = Array(history.dropFirst(result.tailStartIndex))
            rebuildSession(replacingHistoryWith: chatMessages(from: result.injected) + tail)

            log(
                "foundation backend: \(reason); summarized \(result.droppedTurns) turns "
                    + "(\(result.droppedBytes) bytes) into digest \(digest.sourceSHA256.prefix(8)), "
                    + "kept \(tail.count) verbatim")
            return true
        }
    #endif

    // The Chat.Message <-> DigestTurn conversions live in DigestSupport.swift: nothing about them
    // is Foundation Models specific, and the prime path needs the same pair.

    /// Record a finished pass into our own history, and replace the session if the pass did not
    /// finish cleanly.
    ///
    /// TWO separate problems, both measured on macOS 26.6.1, both solved by the same rebuild:
    ///
    /// 1. REUSING A SESSION AFTER A CANCELED PASS CRASHES THE PROCESS. Starting the next
    ///    generation on a session whose previous `streamResponse` was canceled traps inside
    ///    FoundationModels (EXC_BREAKPOINT/SIGTRAP), even though the canceled pass has fully
    ///    unwound, `isResponding` is false and the transcript reads back fine. Reproduced 14
    ///    times in 20 with a gap under a second; a fresh session survived every attempt. This is
    ///    exactly the ACP flow - Stop, then ask again - so without this the whole agent dies on
    ///    an ordinary user action.
    /// 2. A CANCELED EXCHANGE IS PURGED from the framework's transcript, so the model's context
    ///    would silently lose a turn the user can still see on screen, and "why did you stop?"
    ///    would confabulate. MLXBackend deliberately records the partial answer and OpenAIBackend
    ///    keeps `abandonedText` for the same reason; this is that mechanism here.
    ///
    /// The partial answer is kept, exactly as the other two backends keep theirs: it is what the
    /// reader already saw.
    private func recordTurn(prompt: String, answer: String, completed: Bool) {
        lock.withLock {
            seedHistory.append(.user(prompt))
            if !answer.isEmpty { seedHistory.append(.assistant(answer)) }
        }
        // A clean pass leaves the session healthy and already holding this exchange, so keep it -
        // that is what preserves whatever prefix reuse the framework does internally.
        guard !completed else { return }
        rebuildSession()
    }

    // Codes 2-7 in domain "mlx-agent" are taken (main.swift, Backend, Bench, Map). Nothing
    // dispatches on them today, but reusing one would make that impossible to start doing.
    private static func unavailable(_ detail: String) -> NSError {
        NSError(
            domain: "mlx-agent", code: 8,
            userInfo: [NSLocalizedDescriptionKey: "on-device model unavailable: \(detail)"])
    }

    private static func generationFailed(_ detail: String) -> NSError {
        NSError(domain: "mlx-agent", code: 9, userInfo: [NSLocalizedDescriptionKey: detail])
    }
}
