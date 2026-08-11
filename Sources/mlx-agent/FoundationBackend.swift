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
// NOT HERE YET, deliberately:
//   tools     - the framework calls tools ITSELF inside respond/streamResponse and has no mode
//               that surfaces a call for external dispatch, so Agent's loop cannot drive it.
//               Bridging MCP tools means moving the permission gate, iteration cap, timeout and
//               dedup inside the call, which is its own change. Setting `tools` logs and is
//               otherwise ignored, so agent mode degrades to chat rather than pretending.
//   condense  - on overflow this reports a clean error. Summarizing the transcript and retrying
//               is the next step, and wants the digest type that does not exist yet.

import Foundation
import AgentText
import MLXLMCommon

#if canImport(FoundationModels)
    import FoundationModels
#endif

/// Apple's on-device system model, behind the same seam as MLX and llama-server.
///
/// `@unchecked Sendable` with an NSLock, matching the other two backends. `LanguageModelSession`
/// is itself `@unchecked Sendable` and its respond methods are `nonisolated(nonsending)`, so
/// nothing here is forced onto an actor; the lock guards this class's own fields.
final class FoundationBackend: GenerationBackend, @unchecked Sendable {

    private let instructions: String?
    private let parameters: GenerateParameters
    private let lock = NSLock()
    private var toolSpecs: [ToolSpec]?
    private var warnedAboutTools = false
    private let log: @Sendable (String) -> Void

    /// Serializes whole passes. The framework throws `GenerationError.concurrentRequests` when
    /// two overlap, and the ACP cancel window makes that reachable with a well-behaved client -
    /// see PassGate.swift for the sequence.
    private let gate = PassGate()

    /// The conversation this backend owns, in our currency rather than the framework's.
    ///
    /// It exists because a session cannot be reconfigured in place - changing tools, clearing, or
    /// recovering from a cancelled pass all mean constructing a new one - and because the
    /// framework's own transcript is not a usable substitute: a CANCELLED exchange is purged from
    /// it entirely (measured), so reading history back out of the session would silently lose
    /// exactly the turns that need preserving. Completed turns are appended here as they finish.
    private var seedHistory: [Chat.Message]

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
    init(
        parameters: GenerateParameters, systemPrompt: String?, seedHistory: [Chat.Message] = [],
        log: @escaping @Sendable (String) -> Void = { _ in }
    ) {
        self.parameters = parameters
        self.instructions = systemPrompt
        self.seedHistory = seedHistory
        self.log = log
        rebuildSessionLocked()
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
            let count = newValue?.count ?? 0
            let shouldWarn: Bool = lock.withLock {
                toolSpecs = newValue
                guard count > 0, !warnedAboutTools else { return false }
                warnedAboutTools = true
                return true
            }
            // Say so once rather than failing quietly at the first tool the model wants. The
            // framework dispatches tools internally, so Agent's loop cannot mediate them yet.
            if shouldWarn {
                log(
                    "foundation backend: tool calling is not wired up yet - \(count) MCP tool(s) "
                        + "will not be offered to the model; this turn behaves as chat")
            }
        }
    }

    /// Reset the conversation, keeping instructions.
    ///
    /// Serialized against passes like MLXBackend's, so it cannot swap the session out from under
    /// a stream in flight - that pass would go on writing into an orphaned session and its answer
    /// would vanish.
    func clear() async {
        await withPass(gate) {
            lock.withLock { seedHistory = [] }
            rebuildSessionLocked()
        }
    }

    // MARK: - Session lifecycle

    /// Build a fresh session from `instructions` + `seedHistory`.
    ///
    /// Cheap: it allocates a session and a transcript, and loads nothing. The system model is
    /// already resident - that being true is most of the reason this backend exists.
    ///
    /// Reads the history and publishes the session under ONE lock acquisition. Doing it as two
    /// (read, build, write) leaves a window in which a concurrent rebuild publishes a session
    /// built from older history, which is the kind of race that shows up once a month as a lost
    /// turn. Building the transcript inside the lock is fine - it allocates and does no I/O.
    private func rebuildSessionLocked() {
        #if canImport(FoundationModels)
            guard #available(macOS 26.0, *) else { return }
            lock.withLock {
                let transcript = Self.transcript(instructions: instructions, history: seedHistory)
                sessionBox = LanguageModelSession(transcript: transcript)
            }
        #endif
    }

    #if canImport(FoundationModels)
        /// Translate our conversation currency into the framework's.
        ///
        /// Tool turns are DROPPED rather than reconstructed. `Transcript` can express tool calls
        /// and outputs, but this backend does not run tools yet, and a restored tool exchange
        /// referring to tools the session was not given is worse than an honest gap - the model
        /// would see itself calling something it cannot call. ChatView's prime wire carries text
        /// turns only in practice, so in the common path nothing is lost.
        @available(macOS 26.0, *)
        static func transcript(instructions: String?, history: [Chat.Message]) -> Transcript {
            var entries: [Transcript.Entry] = []
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
                guard let session = lock.withLock({ sessionBox }) as? LanguageModelSession else {
                    throw Self.unavailable(FMAvailability.probe().summary)
                }

                var delta = CumulativeDelta()
                let started = Date()
                var firstToken: Date?
                var completed = false
                // A pass that does NOT reach the end - cancelled, or failed mid-answer - leaves
                // the framework's session in a state that must not be reused. See recordTurn.
                defer { recordTurn(prompt: prompt, answer: delta.emittedText, completed: completed) }
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
                    }
                    completed = true
                } catch let error as LanguageModelSession.GenerationError {
                    // Rethrown with a message that says what to do. Overflow is the one a caller
                    // could act on by summarizing; that retry is the next change.
                    throw Self.generationFailed(fmUserFacingError(error))
                }

                let finished = Date()
                // Token counts are 26.4+; below that estimate rather than report zeros, since
                // Agent derives tok/s from these and a zero reads as a stall.
                let answer = delta.emittedText
                let promptTokens = await Self.tokenCount(for: prompt) ?? (prompt.utf8.count / 4)
                let answerTokens = await Self.tokenCount(for: answer) ?? (answer.utf8.count / 4)
                let decodeStart = firstToken ?? finished
                continuation.yield(
                    .info(
                        GenerateCompletionInfo(
                            promptTokenCount: promptTokens,
                            generationTokenCount: answerTokens,
                            promptTime: decodeStart.timeIntervalSince(started),
                            generationTime: finished.timeIntervalSince(decodeStart))))
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

    /// Record a finished pass into our own history, and replace the session if the pass did not
    /// finish cleanly.
    ///
    /// TWO separate problems, both measured on macOS 26.6.1, both solved by the same rebuild:
    ///
    /// 1. REUSING A SESSION AFTER A CANCELLED PASS CRASHES THE PROCESS. Starting the next
    ///    generation on a session whose previous `streamResponse` was cancelled traps inside
    ///    FoundationModels (EXC_BREAKPOINT/SIGTRAP), even though the cancelled pass has fully
    ///    unwound, `isResponding` is false and the transcript reads back fine. Reproduced 14
    ///    times in 20 with a gap under a second; a fresh session survived every attempt. This is
    ///    exactly the ACP flow - Stop, then ask again - so without this the whole agent dies on
    ///    an ordinary user action.
    /// 2. A CANCELLED EXCHANGE IS PURGED from the framework's transcript, so the model's context
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
        rebuildSessionLocked()
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
