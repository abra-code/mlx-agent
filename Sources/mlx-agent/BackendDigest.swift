// BackendDigest.swift - summarizing with the session's own model.
//
// The second `DigestModel` conformance, and the one that makes condensation a property of the
// AGENT rather than of Foundation Models. FM is preferred when it is there - it costs no weights
// and does not disturb the session - but it is macOS 26 only, and a 4096-token window means a long
// conversation needs many sequential slices. A 32k-window model that is ALREADY LOADED does the
// same job in one pass. So: same planner, same digest type, same renderer, different provider.
//
// TWO RULES THIS FILE EXISTS UNDER. Both are measured, not theoretical:
//
// 1. PRIME TIME ONLY. `PassGate.acquire()` is non-reentrant and non-cancellable (PassGate.swift),
//    so driving a backend's `stream()` from inside that backend's own pass deadlocks permanently -
//    no timeout, no cancel, nothing to break it. That is exactly where
//    `FoundationBackend.condenseHistory` sits, which is why this conformance must never be wired
//    there. On the session/prime path the busy claim taken synchronously in
//    `ACPServer.handleSessionPrime`, before its Task exists, guarantees no pass
//    is in flight and none can start, which is what makes this safe there and only there.
// 2. THE LIVE BACKEND, NEVER A SECOND ONE. Two backends over one `ModelContainer` have separate
//    gates and therefore no mutual exclusion at all - the measured `[broadcast_shapes]` SIGTRAP
//    that PassGate exists to prevent. Constructing a private backend "just for summarizing" would
//    reintroduce it exactly.
//
// WHAT THIS DOES TO THE BACKEND IT USES. A `GenerationBackend` owns its conversation; there is no
// stateless pass. So summarizing REPLACES the backend's conversation with the summarizer's own
// exchange (`clear()` at the top of each slice), and the caller must rebuild or discard the
// backend afterwards. The prime path does exactly that - `finishPrime` builds a fresh stack around
// whatever history condensation produced - so the mutation is invisible there. Anywhere else it
// would be data loss, which is the other half of why rule 1 is a rule.
//
// The tolerant JSON decode is NOT here. It is `DigestContent(lenientJSON:)` in AgentDigest, where
// a unit test can reach it without a model. What is left in this file is a prompt, a stream and
// one retry - the parts that can only be judged against a real model.

import AgentDigest
import AgentText
import Foundation
import MLXLMCommon

/// One pass took longer than the generator's bound.
///
/// `LocalizedError` rather than a marker type because this string is what `DigestPlanner.condense`
/// hands the client as the reason its history was primed whole, and "TimeoutError()" is not one.
struct DigestPassTimeout: LocalizedError {
    let seconds: TimeInterval
    var errorDescription: String? {
        "the session's model did not finish a summary pass within \(Int(seconds)) s"
    }
}

/// The whole condensation ran out of time. Distinct from `DigestPassTimeout` because they mean
/// different things to whoever reads the `reason`: one pass is wedged, versus the conversation is
/// simply too much to summarize in the time a restore is allowed.
///
/// Shared with `FMDigestGenerator`, which takes the same deadline on the prime path.
struct DigestCondenseDeadline: LocalizedError {
    var errorDescription: String? {
        "summarizing this conversation was taking too long"
    }
}

/// Summarizes one slice of conversation with whatever model the session is already running.
struct BackendDigestGenerator: DigestModel {

    /// What the prime response's `summarizer` field reports: THE MODEL, when this side can find
    /// out what it is, and `session:<engine>` when it cannot.
    ///
    /// A client renders this verbatim - "64 earlier messages summarized by X" - and X is the
    /// reader's only basis for deciding how much to trust the summary. `session:mlx` was the whole
    /// answer for a while, and it answers the wrong question: which engine wrote it is not
    /// something a reader can act on, while a 2B versus a 31B is the entire judgment. The paths
    /// have several conventions to get wrong (a cached model's directory is `snapshots/main`), so
    /// the decoding is DigestGeneratorName, in the target where it has tests.
    ///
    /// - Parameter model: the model path this engine is running, or nil where nothing knows it.
    static func name(engine: String, model: String? = nil) -> String {
        DigestGeneratorName.session(
            engine: engine, model: model.flatMap(DigestGeneratorName.fromModelPath))
    }

    /// The LIVE backend. See rule 2 in the file header.
    let backend: GenerationBackend
    /// Recorded in the digest as its generator; build it with `name(engine:)`.
    let name: String
    let policy: PrimePolicy
    /// Hard bound on ONE pass. Required rather than defaulted: a slice here is a full generation on
    /// the session's own model, so the right number is a property of that model (a 30B local model
    /// writing 700 tokens is a different order of magnitude from the on-device model's measured
    /// 3-5 s) and guessing it centrally would be guessing for every caller at once.
    ///
    /// A slice can take TWO of them (the repair retry gets its own), so this alone bounds a
    /// condensation only at `maxSlices * 2 * timeout` - minutes on any real setting, with the
    /// session's busy slot held throughout. `deadline` is what bounds the whole thing.
    let timeout: TimeInterval
    /// When the whole condensation must be over, whatever it has managed by then. Each pass gets
    /// the smaller of `timeout` and what is left; once nothing is left, the next pass throws
    /// instead of starting, and the planner primes the full history.
    ///
    /// A deadline rather than a slice count because the failure it guards is "every pass is slow",
    /// which no per-pass bound can see: each one finishes honestly, well inside its timeout, and
    /// the client waits half an hour.
    let deadline: Date
    let log: @Sendable (String) -> Void

    func digest(source: String, priorDigest: DigestContent?) async throws -> DigestContent {
        try Task.checkCancellation()

        // Tools off for the duration. A model that can see tools will happily answer a "summarize
        // this" prompt with a tool call, and the pass would then end with no text at all. Restored
        // afterwards so this is not a lasting edit to a backend the caller may still hold.
        let savedTools = backend.tools
        backend.tools = nil
        defer { backend.tools = savedTools }

        let started = Date()
        let prompt = Self.prompt(source: source, prior: priorDigest, policy: policy)
        let first = try await answer(to: prompt, fresh: true)
        do {
            let content = try DigestContent(lenientJSON: first)
            log(Self.timing(source: source, started: started, retried: false))
            return content
        } catch {
            log(
                "digest slice: the reply did not read as a digest "
                    + "(\(Self.reason(error))); asking once for the JSON alone")
        }

        try Task.checkCancellation()
        // NOT a fresh conversation: the slice is already in the model's context and in its KV
        // cache, so the repair pass prefills one short turn instead of the whole section.
        //
        // The failed reply is QUOTED into the prompt rather than left to the conversation to
        // carry, because on the MLX path it may not be there at all: a reply that was pure
        // reasoning records no assistant turn, and `MLXBackend.reconcileHistory` then merges this
        // prompt into the slice's own user turn - leaving "that was not JSON" pointing at nothing.
        let second = try await answer(
            to: Self.repairPrompt(policy: policy, failedReply: first), fresh: false)
        // A second failure throws, and `DigestPlanner.condense` turns that into a full-history
        // prime. Never a half-digest: see the header of LenientDecode.swift.
        let content = try DigestContent(lenientJSON: second)
        log(Self.timing(source: source, started: started, retried: true))
        return content
    }

    /// One pass: send `prompt`, return the model's text.
    ///
    /// - Parameter fresh: clear the backend's conversation first. True for a slice (prior slices
    ///   travel in the prompt, where their size is bounded; carrying them as conversation would
    ///   grow the context by the whole history and cause the overflow this feature exists to
    ///   avoid), false for the repair pass.
    private func answer(to prompt: String, fresh: Bool) async throws -> String {
        if fresh { await backend.clear() }
        let backend = self.backend
        let answerCap = Self.replyByteCap(policy)
        // Whichever runs out first: this pass's own bound, or the whole condensation's.
        let seconds = min(timeout, deadline.timeIntervalSinceNow)
        guard seconds > 0 else { throw DigestCondenseDeadline() }

        // A task GROUP rather than `withTimeout`, and the difference is the whole reason to spell
        // it out here. `withTimeout` runs its operation in an unstructured Task that inherits no
        // cancellation and is never awaited - correct for FMDigestGenerator, where the operation
        // genuinely cannot be interrupted, and wrong here, where it can: a Stop would otherwise go
        // unheard for a full pass (measured: 10 s on a small local model, and a slice on a large
        // one is minutes). Child tasks inherit cancellation, so a cancel reaches the stream.
        //
        // What this does NOT do is wait for the backend's own pass to finish unwinding. Exiting
        // the group awaits the two children here, but a canceled AsyncThrowingStream terminates
        // and resumes the consumer with nil while the producer is still tearing down - so after a
        // timeout the pass is canceled, not drained. `ACPServer.condenseForPrime` does the actual
        // join with `clear()`, which takes the backend's PassGate.
        let text = try await withThrowingTaskGroup(of: String.self) { group in
            group.addTask {
                var answer = ""
                var reasoningBytes = 0
                // The MLX path never yields `.reasoning`: it streams raw text, so a <think> block
                // arrives inline as `.text`. Counting those bytes against the answer cap would cut
                // a reasoning model off BEFORE it emits the object - the same model summarizes
                // fine through llama-server, which splits reasoning out for us. So the same
                // splitter MLXBackend uses to project its own history is used to tell the two
                // apart, and only the message channel is kept.
                let splitter = ThinkSplitter()
                // `.user` is built in here rather than passed in: Chat.Message is not Sendable, so
                // it cannot cross into this closure.
                for try await event in backend.stream([.user(prompt)]) {
                    guard case .text(let chunk) = event else { continue }
                    for (kind, piece) in splitter.feed(chunk) {
                        switch kind {
                        case .message: answer += piece
                        case .thought: reasoningBytes += piece.utf8.count
                        }
                    }
                    // A model that never stops is otherwise bounded only by the timeout, and on a
                    // slow local model that is minutes of a restore the user is waiting on.
                    // Breaking terminates the stream, which cancels the pass.
                    if answer.utf8.count > answerCap || reasoningBytes > Self.reasoningByteCap {
                        break
                    }
                }
                // Whatever the splitter was holding back. An UNTERMINATED <think> flushes as
                // thought, so a reply that was pure reasoning correctly yields no answer at all.
                for (kind, piece) in splitter.flush() where kind == .message { answer += piece }
                return answer
            }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(max(0, seconds) * 1_000_000_000))
                throw DigestPassTimeout(seconds: seconds)
            }
            defer { group.cancelAll() }
            guard let first = try await group.next() else { throw CancellationError() }
            return first
        }
        // A canceled stream FINISHES rather than throwing, so the loop above returns a partial
        // answer instead of failing. Asking directly is what makes a Stop report itself as a
        // cancellation rather than as a decode failure and a pointless repair pass.
        try Task.checkCancellation()
        return text
    }

    /// Room for the digest the policy asked for, and no more: four bytes per token is the same
    /// crude ratio `DigestPlanner.estimateTokens` uses, doubled so a reply with a fence and a
    /// sentence of preamble is not cut short, and floored so a small `maxDigestTokens` cannot make
    /// the cap bite on a well-behaved answer.
    ///
    /// `maxItemsPerList` is in it because the PROMPT advertises that number: at the clamp extremes
    /// (32 items per list, 128 digest tokens) the model is invited to write up to 192 entries into
    /// a cap sized for a paragraph, and would be cut off on both passes.
    static func replyByteCap(_ policy: PrimePolicy) -> Int {
        max(4096, policy.maxDigestTokens * 8, policy.maxItemsPerList * 6 * 96)
    }

    /// Separate, and generous, because reasoning is not the answer: it is bounded by the session's
    /// own `maxTokens` anyway, and this exists only so a model stuck in a thought loop cannot hold
    /// the pass open for the whole timeout. 64 KB is roughly 16k tokens of thinking.
    static let reasoningByteCap = 64 * 1024

    // MARK: - The prompt

    // The first paragraph tracks `FMDigestGenerator.instructions` because the job is identical.
    // Kept as a separate string rather than factored into one: this prompt must also carry the
    // wire format (the on-device model gets that from the @Generable schema, for free and more
    // reliably), and prompt text that serves two different models is prose to be tuned per model,
    // not code to be deduplicated.
    //
    // The shape is spelled out as a literal example rather than described. Instruct-tuned models
    // reproduce an example far more reliably than they follow a description of one, and every
    // token spent here buys back a repair pass.
    //
    // THE EXAMPLE COMES FROM THE LIBRARY (`DigestContent.jsonTemplate`), AND THAT IS LOAD-BEARING.
    // A model restating the format before answering - one of the most ordinary things a model does
    // - emits a valid, schema-shaped object whose values are the placeholders. If those read as
    // content, the decoder accepts a digest that says "what the user is still trying to
    // accomplish", `isEmpty` is false, and the planner reports success while dropping every turn
    // it replaced. The angle brackets are what let `DigestContent(lenientJSON:)` refuse it, so the
    // template and the decoder have to be one agreement rather than two texts that resemble each
    // other. The decoder ALSO prefers the last schema-shaped object in a reply, so an echo
    // followed by the real answer resolves to the answer. Two defenses, because this one is silent.
    static func instructions(policy: PrimePolicy) -> String {
        """
        You compress conversation transcripts into structured notes.

        The notes are read by an assistant that will continue the conversation and will NOT see \
        the original text. Record only what the transcript states. Never guess, never infer intent \
        that was not expressed, and never add advice of your own. Prefer exact names, paths, \
        numbers and versions over descriptions of them. If a field has nothing to record, leave it \
        empty rather than filling it.

        Reply with ONE JSON object and nothing else: no explanation before it, no code fence, no \
        commentary after it. Use exactly these keys, replacing every <...> with real content:

        \(DigestContent.jsonTemplate)

        Every key must be present. Each list holds at most \(policy.maxItemsPerList) entries, \
        most important first, and every entry outside toolEvents is a plain string.
        """
    }

    static func prompt(source: String, prior: DigestContent?, policy: PrimePolicy) -> String {
        var out = instructions(policy: policy)
        out += "\n\n"
        if let prior, !prior.isEmpty {
            out += "Earlier parts of this same conversation already established:\n"
            out += DigestRenderer.renderPriorContext(prior)
            out += "\n\nDo not repeat those. "
        }
        out += "Record what the following section of the conversation establishes.\n\n"
        out += "CONVERSATION SECTION:\n"
        out += source
        return out
    }

    /// The one retry. Self-contained: the section is still in the conversation, but the failed
    /// reply may not be (see the call site), so a bounded prefix of it is quoted back.
    static func repairPrompt(policy: PrimePolicy, failedReply: String) -> String {
        var out = "That reply was not a single JSON object I can read"
        let quoted = failedReply.trimmingCharacters(in: .whitespacesAndNewlines).prefix(400)
        if !quoted.isEmpty {
            // Enough to point at, not enough to re-anchor the model on its own mistake.
            out += ". It began:\n\n\(quoted)\n\n"
        } else {
            out += " - it contained no JSON at all. "
        }
        out += """
            Send the notes again for the same conversation section as ONE JSON object with the \
            keys unresolvedIntent, establishedFacts, decisions, toolEvents, openThreads and \
            userPreferences, at most \(policy.maxItemsPerList) entries per list. Output the object \
            alone - no reasoning, no code fence, no text before or after it.
            """
        return out
    }

    private static func timing(source: String, started: Date, retried: Bool) -> String {
        "digest slice: \(source.utf8.count) bytes summarized in "
            + "\(String(format: "%.1f", Date().timeIntervalSince(started))) s"
            + (retried ? " (after one repair pass)" : "")
    }

    /// One line, for the log. `localizedDescription` is useless on a bridged Swift error - see
    /// `DigestPlanner.describe` for the same problem and the same shape of fix.
    private static func reason(_ error: Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? "\(error)"
    }
}
