// FMDigest.swift - Apple's on-device model as a summarizer.
//
// The production conformance to `DigestModel`, and deliberately the thinnest file in the feature:
// splitting, slicing, merging, fallback and rendering all live in AgentDigest where they are unit
// tested. What is left here is a schema, a prompt, and the mapping back. That split is the whole
// reason the seam exists - none of this can be asserted in a test, because the output is
// nondeterministic and availability is a property of the machine.
//
// Summarizing is the one job the on-device model is unambiguously RIGHT for. It is bounded (one
// pass over text that is already sized to fit), it is structured (a schema, not prose), it needs
// no weights resident, and its output is consumed by a bigger model rather than shown to the user.
// The reverse - using a 30B MLX model to summarize before handing off to itself - costs a load and
// a long generation to save a prefill.
//
// THE @Generable TWIN. `GeneratedDigest` duplicates the fields of `DigestContent` because the
// storage type must not link FoundationModels: a digest generated here is primed into MLX and
// OpenAI sessions, and on macOS 14 where the framework does not exist. Six field names in two
// places is the price of that, and `generatedDigestMatchesStorage` in the tests is what keeps them
// honest.

import AgentDigest
import Foundation

#if canImport(FoundationModels)
    import FoundationModels
#endif

#if canImport(FoundationModels)

    /// The shape the model is asked to fill in.
    ///
    /// Every `@Guide` description is prompt text - the framework puts the schema in front of the
    /// model - so they are written as instructions to a writer, not as API documentation.
    ///
    /// The `maximumCount` caps are compile-time constants and therefore cannot follow
    /// `PrimePolicy.maxItemsPerList`. They match its default of 8, and `DigestPlanner.merge`
    /// applies the runtime cap after the fact, so a lowered policy still takes effect - it just
    /// does not save the model any work.
    @available(macOS 26.0, *)
    @Generable
    struct GeneratedDigest {

        @Guide(
            description:
                "What the user is still trying to accomplish, in one sentence. Empty if the "
                + "conversation reached a conclusion and nothing is outstanding.")
        var unresolvedIntent: String

        @Guide(
            description:
                "Specific things this conversation established as true: names, paths, versions, "
                + "numbers, measured results. Facts, not summaries of the discussion.",
            .maximumCount(8))
        var establishedFacts: [String]

        @Guide(
            description:
                "Choices that were made, each with its reason, so they are not reopened later.",
            .maximumCount(8))
        var decisions: [String]

        @Guide(
            description: "Tools that were used and what each one established.",
            .maximumCount(8))
        var toolEvents: [GeneratedToolEvent]

        @Guide(
            description: "Work explicitly left unfinished or unanswered.",
            .maximumCount(8))
        var openThreads: [String]

        @Guide(
            description:
                "Constraints the user stated about how the work should be done - style, tools, "
                + "things to avoid. Only ones the user actually stated.",
            .maximumCount(8))
        var userPreferences: [String]
    }

    @available(macOS 26.0, *)
    @Generable
    struct GeneratedToolEvent {
        @Guide(description: "The tool's name, exactly as it appears in the conversation.")
        var tool: String
        @Guide(description: "Why it was called, in a few words.")
        var whatFor: String
        @Guide(description: "What the call established. The conclusion, not the raw output.")
        var materialResult: String
    }

    @available(macOS 26.0, *)
    extension GeneratedDigest {
        /// Cross the boundary into the model-free storage type.
        ///
        /// An empty `unresolvedIntent` becomes nil rather than "": the schema cannot express an
        /// optional String, so "nothing outstanding" has to arrive as an empty string and be
        /// interpreted here. Blank list entries are dropped for the same reason - a model asked
        /// for up to eight items will pad.
        var asDigestContent: DigestContent {
            func kept(_ items: [String]) -> [String] {
                items.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
            }
            let intent = unresolvedIntent.trimmingCharacters(in: .whitespacesAndNewlines)
            return DigestContent(
                unresolvedIntent: intent.isEmpty ? nil : intent,
                establishedFacts: kept(establishedFacts),
                decisions: kept(decisions),
                toolEvents: toolEvents.compactMap { event in
                    let name = event.tool.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !name.isEmpty else { return nil }
                    return ToolEvent(
                        tool: name,
                        whatFor: event.whatFor.trimmingCharacters(in: .whitespacesAndNewlines),
                        materialResult: event.materialResult.trimmingCharacters(
                            in: .whitespacesAndNewlines))
                },
                openThreads: kept(openThreads),
                userPreferences: kept(userPreferences))
        }
    }

    /// Summarizes one slice of conversation with the on-device model.
    @available(macOS 26.0, *)
    struct FMDigestGenerator: DigestModel {

        /// Recorded in every digest this produces. The string itself lives in DigestSelection.swift,
        /// which compiles even where this type does not.
        static let name = foundationSummarizerName

        let policy: PrimePolicy
        /// Hard bound on one slice. Guided generation on this model is NOT bounded by the schema -
        /// a 57-character input once ran 113 seconds before throwing - so `maximumResponseTokens`
        /// is the first defense and this is the second.
        ///
        /// It doubles as the bound on how long a Stop can be ignored. `withTimeout` runs the
        /// operation in an UNSTRUCTURED task, which does not inherit the caller's cancellation and
        /// is awaited through a continuation that is not cancellation-aware - that is the whole
        /// point of it (a hard bound regardless of whether the operation cooperates), but it means
        /// a cancel lands only between slices. `DigestPlanner` checks cancellation at the top of
        /// each slice, so the residual is exactly one slice: measured at 3-5 s, capped here.
        let timeout: TimeInterval
        /// When the whole condensation must be over, whatever it has managed by then.
        ///
        /// `.distantFuture` for the backend's own mid-turn overflow condensation, which is already
        /// inside a pass the caller is waiting on and has nothing better to do than finish. The
        /// prime path passes a real one: 24 slices at `timeout` apiece is ten minutes of a held
        /// busy slot, and no per-slice bound can see that.
        let deadline: Date
        let log: @Sendable (String) -> Void

        init(
            policy: PrimePolicy, timeout: TimeInterval, deadline: Date = .distantFuture,
            log: @escaping @Sendable (String) -> Void
        ) {
            self.policy = policy
            self.timeout = timeout
            self.deadline = deadline
            self.log = log
        }

        func digest(source: String, priorDigest: DigestContent?) async throws -> DigestContent {
            // Cheap, and it means the conformance does not depend on the planner remembering to
            // check: a cancel that lands between slices stops here rather than paying for one more.
            try Task.checkCancellation()
            let prompt = Self.prompt(source: source, prior: priorDigest)
            let policy = self.policy
            let started = Date()
            // Whichever runs out first: this slice's own bound, or the whole condensation's.
            let timeout = min(timeout, deadline.timeIntervalSinceNow)
            guard timeout > 0 else { throw DigestCondenseDeadline() }
            let content = try await withTimeout(timeout) {
                // A FRESH session per slice, not one reused across the loop. A session's transcript
                // is append-only, so reusing it would carry every previous slice into the window
                // and guarantee the overflow this whole feature exists to avoid. Prior context
                // travels in the prompt instead, where its size is bounded.
                //
                // permissiveContentTransformations because the DEFAULT guardrails refuse ordinary
                // documents - measured: an innocuous paragraph came back guardrailViolation. A
                // summarizer that declines the user's own conversation is useless, and this is a
                // transformation of text the user already has, not generation of new content.
                //
                // .general, not .contentTagging: contentTagging is incompatible with a custom
                // @Generable type and fails every request with decodingFailure. Measured, and the
                // name makes it look like exactly the right choice, so it is worth stating.
                let model = SystemLanguageModel(
                    useCase: .general, guardrails: .permissiveContentTransformations)
                let session = LanguageModelSession(model: model, instructions: Self.instructions)
                let response = try await session.respond(
                    to: prompt,
                    generating: GeneratedDigest.self,
                    // Greedy so a repeated condensation of the same history reads the same, which
                    // is what makes the audit artifact worth keeping.
                    options: GenerationOptions(
                        sampling: .greedy, maximumResponseTokens: policy.maxDigestTokens))
                return response.content.asDigestContent
            }
            log(
                "digest slice: \(source.utf8.count) bytes summarized in "
                    + "\(String(format: "%.1f", Date().timeIntervalSince(started))) s")
            return content
        }

        // Framed as a mechanical transformation rather than a request for a good summary: the
        // model is a 3B one, and "record what is stated" is a job it can do, while "judge what
        // matters" invites invention. The consumer is another model, which is why the wording asks
        // for retrievable specifics rather than readable prose.
        static let instructions = """
            You compress conversation transcripts into structured notes.

            The notes are read by an assistant that will continue the conversation and will NOT \
            see the original text. Record only what the transcript states. Never guess, never \
            infer intent that was not expressed, and never add advice of your own. Prefer exact \
            names, paths, numbers and versions over descriptions of them. If a field has nothing \
            to record, leave it empty rather than filling it.
            """

        static func prompt(source: String, prior: DigestContent?) -> String {
            var out = ""
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
    }

#endif
