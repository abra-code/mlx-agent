// DigestPlanner.swift - deciding what to summarize, what to keep verbatim, and what to do when
// the summarizer fails.
//
// All of the judgment in this feature is here, and all of it is pure and synchronous except the
// one `await` that calls the model. That is on purpose: summarization is nondeterministic and
// untestable, but "which turns are safe to drop", "what happens when the model throws" and "does
// the assembled history still alternate roles" are exactly the questions a unit test can settle,
// and exactly the ones that go wrong quietly.
//
// THE RULE THAT OUTRANKS EVERYTHING ELSE HERE: history is never lost. A digest REPLACES turns in
// what gets primed, so any doubt - the model threw, timed out, returned nothing usable, the work
// was canceled, there were more slices than the budget allows, or there was nothing old enough to
// summarize - resolves to priming the FULL history unchanged. Condensing is an optimization;
// truncating is data loss wearing an optimization's clothes. `condense` cannot throw, and that is
// the enforcement.

import Foundation

/// Knobs for one condensation. Clamped on construction, because these arrive over the wire.
public struct PrimePolicy: Sendable, Equatable {

    /// How many trailing MESSAGES (not exchanges) survive verbatim. The default of 6 is three
    /// exchanges - enough that "it" and "that file" in the next prompt still resolve against real
    /// text rather than against a summary of it.
    public private(set) var keepRecentTurns: Int
    /// Ceiling on the generated digest, in tokens. Bounds generation time as much as size: on the
    /// on-device model an unbounded guided generation can run for minutes.
    public private(set) var maxDigestTokens: Int
    /// How much conversation goes into the model at once. Must leave room in the context window
    /// for the schema and the generated digest, so it is well under the full window.
    public private(set) var sliceBudgetTokens: Int
    /// Cap per list PER SLICE, matching the per-slice `@Guide(.maximumCount:)`. The merged digest
    /// allows more - see `mergedCap` - because one cap across a ten-slice conversation would make
    /// the digest worse the longer the conversation got, which is backwards.
    public private(set) var maxItemsPerList: Int
    /// Hard ceiling on model calls for one condensation. Slice SIZE is bounded by
    /// `sliceBudgetTokens`; without this, slice COUNT is bounded only by the size of the history a
    /// client sends, and each slice is a sequential generation on the critical path of a restore.
    /// Exceeding it falls back to the full history, which is the safe direction.
    public private(set) var maxSlices: Int
    /// The summarizing model's context window, when the caller knew it. nil means "unstated", and
    /// then `sliceBudgetTokens` is clamped at `defaultSliceBudgetCeiling` - a bound sized for a
    /// 4096-token model, which is the only safe assumption in the absence of information.
    ///
    /// Stored rather than consumed at construction so `with(...)` can RE-derive the budget when a
    /// caller changes `maxDigestTokens`. The two are not independent: the window has to hold the
    /// slice, the prior digest, the generated digest and the framework's own preamble at once, so
    /// raising the digest allowance has to lower the slice budget or it overflows.
    public private(set) var windowTokens: Int?

    public static let `default` = PrimePolicy()

    /// What the slice budget is clamped to when no window is stated. See `windowTokens`.
    public static let defaultSliceBudgetCeiling = 8192

    /// Room reserved, beyond the two digest allowances, for whatever the summarizer's own stack
    /// puts in the window: instructions, a chat template, and - on Foundation Models - the schema
    /// the framework echoes into the prompt.
    public static let promptOverheadTokens = 768

    /// Values outside the ranges below are CLAMPED, not rejected. They come from a client, and a
    /// nonsense number is not worth failing a restore over - but honoring `sliceBudgetTokens:
    /// 100000` would guarantee a context overflow, and `keepRecentTurns: 0` would summarize the
    /// question the user is waiting on an answer to.
    ///
    /// The properties are `private(set)` for the same reason the clamps exist. "Clamped on
    /// construction" is only a safety story if construction is the ONLY way in; a plain `var`
    /// would let a caller assign `keepRecentTurns = 0` past every check here, and `split` would
    /// then index off the end of the history.
    public init(
        keepRecentTurns: Int = 6,
        maxDigestTokens: Int = 700,
        sliceBudgetTokens: Int = 2800,
        maxItemsPerList: Int = 8,
        maxSlices: Int = 16,
        windowTokens: Int? = nil
    ) {
        self.keepRecentTurns = min(max(keepRecentTurns, 2), 64)
        self.maxDigestTokens = min(max(maxDigestTokens, 128), 2048)
        self.windowTokens = windowTokens
        // THE CEILING IS THE DERIVATION, and it is here rather than in `sized` so that no path can
        // route around it. The window has to hold four things at once - the slice, the prior digest
        // carried into the prompt, the digest being generated, and the instructions and template
        // around them - so the budget a stated window allows is what is left after the other three.
        //
        // Doing this arithmetic in `sized` instead was wrong in a way worth recording: it
        // subtracted the CALLER's `maxDigestTokens` rather than the clamped one, so a client
        // sending `maxDigestTokens: -5000` over the wire got a slice budget of the entire window -
        // an overflow arriving through the one field the clamps were supposed to make safe. Using
        // `self.maxDigestTokens`, after its own clamp, is what closes that.
        //
        // Fixed at 8192 this used to be a ceiling sized for a 4096-token model applied to every
        // model: a 32k-window one then sliced at 8192 and did four sequential passes where one
        // would have done.
        let digestAllowance = self.maxDigestTokens
        let ceiling =
            windowTokens.map { max(256, $0 - 2 * digestAllowance - Self.promptOverheadTokens) }
            ?? Self.defaultSliceBudgetCeiling
        self.sliceBudgetTokens = min(max(sliceBudgetTokens, 256), max(256, ceiling))
        // The floor is 3, not 1. A one-item-per-list digest standing in for thirty turns passes
        // every emptiness check and is still useless - the caller is told `condensed: true` and
        // has no way to know the summary is a stub.
        self.maxItemsPerList = min(max(maxItemsPerList, 3), 32)
        self.maxSlices = min(max(maxSlices, 1), 256)
    }

    /// Sizing derived from the summarizing model's context window: ask for the whole window and
    /// let the ceiling in `init` take out what the window has to hold alongside the slice.
    ///
    /// This is the constructor every caller that knows its window should use; the memberwise one
    /// is for a caller that only knows what it wants and has to be clamped conservatively.
    ///
    /// The window passed here must already have room taken out for whatever the SUMMARIZER will
    /// generate beyond one digest - a backend that will happily produce `maxTokens` of reasoning
    /// before its answer is spending window this arithmetic does not know about. See
    /// `ACPServer.sessionOption`.
    public static func sized(
        window: Int,
        keepRecentTurns: Int = 6,
        maxDigestTokens: Int = 700,
        maxItemsPerList: Int = 8,
        maxSlices: Int = 16
    ) -> PrimePolicy {
        PrimePolicy(
            keepRecentTurns: keepRecentTurns,
            maxDigestTokens: maxDigestTokens,
            sliceBudgetTokens: window,
            maxItemsPerList: maxItemsPerList,
            maxSlices: maxSlices,
            windowTokens: window)
    }

    /// The same policy with the two knobs a client may set over the wire changed.
    ///
    /// Not a pair of `var`s: those would let a caller set one knob past every clamp here. Where a
    /// window is known the budget is RE-DERIVED (by asking for the window again and letting the
    /// ceiling do its work) rather than carried, because `maxDigestTokens` is one of the things
    /// that window has to hold - raising it has to lower the slice.
    public func with(keepRecentTurns: Int? = nil, maxDigestTokens: Int? = nil) -> PrimePolicy {
        PrimePolicy(
            keepRecentTurns: keepRecentTurns ?? self.keepRecentTurns,
            maxDigestTokens: maxDigestTokens ?? self.maxDigestTokens,
            sliceBudgetTokens: windowTokens ?? sliceBudgetTokens,
            maxItemsPerList: maxItemsPerList,
            maxSlices: maxSlices,
            windowTokens: windowTokens)
    }

    /// The cap that applies to the MERGED digest: the per-slice cap, scaled by how many slices
    /// contributed, up to 3x. A single global cap means an early slice can fill it and silence
    /// every later one, so the more turns a digest replaces the less of each it keeps. The 3x
    /// ceiling stops a fifty-slice history from rendering a preamble longer than the conversation.
    func mergedCap(slices: Int) -> Int {
        maxItemsPerList * min(max(slices, 1), 3)
    }
}

/// What a condensation produced, including the case where it produced nothing.
///
/// INVARIANT, asserted in the tests: `history == injected + input[tailStartIndex...]`, where
/// `input` is the history passed to `condense`. When nothing was condensed, `injected` is empty
/// and `tailStartIndex` is 0, so the identity still holds and `history` is the input.
public struct CondenseResult: Sendable {
    /// The history to prime.
    public var history: [DigestTurn]
    /// The turns the planner PRODUCED - hoisted system turns, the digest preamble, and its
    /// acknowledgment when one is needed. Everything in `history` that is not from the input.
    ///
    /// Exposed so a caller can splice its own richer message type: `injected` needs converting,
    /// the tail does not, and re-deriving that boundary from `droppedTurns` would couple the
    /// caller to an identity this type does not otherwise promise.
    public var injected: [DigestTurn]
    /// Where the untouched suffix of the input begins.
    public var tailStartIndex: Int
    /// nil when nothing was condensed - and then `history` is the input, byte for byte.
    public var digest: SessionDigest?
    /// Why nothing was condensed. Set whenever `digest` is nil; reported to the client verbatim.
    public var reason: String?
    /// Turns replaced by the digest. Excludes hoisted system turns, which are kept, not dropped.
    public var droppedTurns: Int
    /// Content bytes those turns held. Together with `droppedTurns` this is what a client needs to
    /// tell the user what a restore actually cost them.
    public var droppedBytes: Int
    /// The exact text handed to the summarizer, for the audit artifact - the concatenated slices,
    /// not a separate rendering of the same turns, so `sourceSHA256` covers what the model saw.
    /// Empty when nothing ran.
    public var source: String

    /// Derived rather than stored so it cannot disagree with `digest`.
    public var condensed: Bool { digest != nil }
}

public enum DigestPlanner {

    // MARK: - Splitting

    /// Split history into the part to summarize and the part to keep verbatim.
    ///
    /// The boundary is snapped BACKWARD to a user turn where one is close enough, so the verbatim
    /// tail begins with a user message and the assembled history alternates cleanly. The search is
    /// bounded to `keepRecentTurns` extra messages: without a bound, a transcript with one early
    /// user turn and a long monologue after it drags the boundary to zero and condenses nothing.
    ///
    /// When no user turn is in reach the boundary STAYS at `keepRecentTurns` from the end. It
    /// deliberately does not search forward: that would keep fewer messages verbatim, which is the
    /// unsafe direction, and the alternation it would buy is handled instead by omitting the
    /// acknowledgment turn when the tail already begins with a non-user turn (see `injectedTurns`).
    public static func split(_ history: [DigestTurn], policy: PrimePolicy = .default)
        -> (summarize: [DigestTurn], verbatim: [DigestTurn])
    {
        // `policy` cannot carry an out-of-range value today, but split indexes an array with it,
        // and a crash inside a restore is the worst possible way to learn that changed.
        let keep = min(max(policy.keepRecentTurns, 1), history.count)
        guard history.count > keep else { return ([], history) }
        let target = history.count - keep
        let floor = max(0, target - keep)

        var boundary = target
        if let back = (floor...target).reversed().first(where: { history[$0].role == .user }) {
            boundary = back
        }
        return (Array(history[..<boundary]), Array(history[boundary...]))
    }

    // MARK: - Rendering the source

    /// One labeled block per turn: `(label, body)`. Empty turns produce nothing.
    private static func blocks(_ turns: [DigestTurn]) -> [(label: String, body: String)] {
        turns.compactMap { turn in
            let body = turn.content.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !body.isEmpty else { return nil }
            let label: String
            switch turn.role {
            case .user: label = "User"
            case .assistant: label = "Assistant"
            case .system: label = "System instructions"
            case .tool: label = "Tool result"
            }
            return (label, body)
        }
    }

    /// The turns as one labeled transcript. Used for display and by callers that want the whole
    /// thing; `condense` hashes the SLICES instead, since those are what the model actually reads.
    public static func renderSource(_ turns: [DigestTurn]) -> String {
        blocks(turns).map { "\($0.label): \($0.body)" }.joined(separator: "\n\n")
    }

    /// Cut the source into pieces that each fit `sliceBudgetTokens`.
    ///
    /// Turn boundaries are preferred. A SINGLE turn over budget - a pasted file, a long tool
    /// result - is hard-split rather than dropped or sent whole: sending it whole guarantees a
    /// context overflow, and dropping it loses exactly the content most likely to matter.
    ///
    /// Continuation pieces repeat the role label. Without that, every piece after the first is
    /// anonymous mid-sentence prose and the summarizer cannot tell the user's words from the
    /// assistant's from a pasted tool result - which is the only reason the labels exist.
    public static func slices(of turns: [DigestTurn], policy: PrimePolicy = .default) -> [String] {
        var out: [String] = []
        var current = ""

        func flush() {
            let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { out.append(trimmed) }
            current = ""
        }

        for block in blocks(turns) {
            // Reserve room for the label before splitting. `hardSplit` sizes the BODY, so a piece
            // that exactly fills the budget would overflow it the moment "User (continued): " is
            // prepended - the longer of the two forms, so the bound holds for every piece. Rounded
            // UP with one token to spare, because `estimateTokens` truncates: a body sized to N
            // tokens can carry up to 4N+3 bytes, and that slack is enough to push the labeled
            // piece back over on its own.
            let labelBytes = "\(block.label) (continued): ".utf8.count
            let labelOverhead = (labelBytes + 3) / 4 + 1
            let pieces = hardSplit(
                block.body, maxTokens: max(1, policy.sliceBudgetTokens - labelOverhead))
            for (index, piece) in pieces.enumerated() {
                let label = index == 0 ? block.label : "\(block.label) (continued)"
                let text = "\(label): \(piece)"
                if !current.isEmpty,
                    estimateTokens(current) + estimateTokens(text) > policy.sliceBudgetTokens
                {
                    flush()
                }
                if !current.isEmpty { current += "\n\n" }
                current += text
            }
        }
        flush()
        return out
    }

    /// How many model passes `condense` would need for this history under `policy`.
    ///
    /// Pure, cheap (string arithmetic, no model, no availability probe) and - the part that
    /// matters - derived by running the SAME steps `condense` runs, so it cannot answer a
    /// different question than the one the caller is about to ask. It exists so a caller choosing
    /// between summarizers can compare a 4096-token window against a 32k one on the conversation
    /// it actually has, rather than preferring one on principle.
    ///
    /// 0 means there is nothing to summarize, and `condense` would fall back to the full history.
    public static func sliceCount(of history: [DigestTurn], policy: PrimePolicy = .default) -> Int {
        let (older, _) = split(history, policy: policy)
        return slices(of: older.filter { $0.role != .system }, policy: policy).count
    }

    /// Split on character boundaries so no piece exceeds `maxTokens` by `estimateTokens`.
    ///
    /// Characters, not scalars or bytes: cutting inside a grapheme would hand the model a broken
    /// cluster, and cutting inside a UTF-8 sequence is not expressible as a String at all.
    static func hardSplit(_ text: String, maxTokens: Int) -> [String] {
        guard maxTokens > 0, estimateTokens(text) > maxTokens else { return [text] }
        var pieces: [String] = []
        var piece = ""
        var bytes = 0
        var wide = 0
        for ch in text {
            let chBytes = String(ch).utf8.count
            let chWide = ch.unicodeScalars.reduce(0) { $1.isASCII ? $0 : $0 + 1 }
            // A single character larger than the whole budget cannot be split further; emit it
            // alone and accept the overrun rather than looping forever.
            if !piece.isEmpty,
                max((bytes + chBytes) / 4, wide + chWide) > maxTokens
            {
                pieces.append(piece)
                piece = ""
                bytes = 0
                wide = 0
            }
            piece.append(ch)
            bytes += chBytes
            wide += chWide
        }
        if !piece.isEmpty { pieces.append(piece) }
        return pieces
    }

    /// Rough token count, from two crude signals: UTF-8 bytes / 4, and the number of non-ASCII
    /// scalars, whichever is larger.
    ///
    /// The bytes/4 rule alone is an English rule. CJK runs about 3 UTF-8 bytes per character at
    /// roughly a token per character, so bytes/4 under-reports it by about a third - and a slice
    /// budget that under-reports produces exactly the context overflow the slicing exists to
    /// prevent. Counting non-ASCII scalars as a token apiece is not accurate either, but it is
    /// wrong in the safe direction for the scripts where the byte rule is wrong in the dangerous
    /// one.
    ///
    /// Deliberately not the real tokenizer: `SystemLanguageModel.tokenCount(for:)` is macOS 26.4+
    /// and async, and threading an async estimator through the slicing loop would make this whole
    /// file async to buy accuracy that only decides how many slices to use.
    public static func estimateTokens(_ text: String) -> Int {
        var wide = 0
        for scalar in text.unicodeScalars where !scalar.isASCII { wide += 1 }
        return max(1, max(text.utf8.count / 4, wide))
    }

    // MARK: - Fitting the result to the model that has to hold it

    /// What one turn costs beyond its own text: the role marker and whatever the chat template
    /// wraps it in. Small, but a 60-turn tail pays it 60 times, and under-reporting here is the
    /// direction that produces an overflow.
    public static let turnOverheadTokens = 4

    /// Drop the oldest turns of a verbatim tail until what is left fits `budgetTokens`.
    ///
    /// A DIFFERENT BOUND FROM THE ONE `condense` APPLIES. Condensation is sized for the
    /// SUMMARIZER's window - how much conversation one pass can read - and a summary that came out
    /// fine can still be primed into a model that cannot hold it: the two models need not be the
    /// same one, the verbatim tail is a message COUNT with no size bound, and every failure path in
    /// `condense` falls back to the FULL history. That fallback is exactly the case that breaks - a
    /// conversation too large to summarize is also too large to prime - and it broke as an HTTP 400
    /// from llama-server on the first message after a restore, naming no cause a reader could act
    /// on.
    ///
    /// Dropping is not free and must not be hidden: the caller reports what went. On the condense
    /// path those turns are at least represented in the digest that replaced their predecessors.
    /// It is still better than the alternative, which is a conversation that cannot answer at all.
    ///
    /// GENERIC BECAUSE THE CALLER THAT MATTERS DOES NOT HOLD `DigestTurn`s. A primed history is
    /// `Chat.Message`, which carries tool-call metadata a round trip through `DigestTurn` would
    /// discard - and the first version of this had a `DigestTurn` implementation that tests
    /// exercised beside a hand-written copy in the caller that actually shipped. One
    /// implementation, driven through closures, is the fix for that.
    ///
    /// - Parameters:
    ///   - history: the verbatim tail, oldest first. Any preamble is the caller's to keep: it is
    ///     not passed here, because dropping a digest while keeping the turns it stands in for is
    ///     the one thing this must never do.
    ///   - budgetTokens: how much PROMPT is left for these turns, after the serving model's own
    ///     reserve for what it will generate and after anything the caller is prepending.
    ///   - cost: what one turn occupies in the window, INCLUDING whatever rides outside its text.
    ///   - bytes: what one turn is worth reporting as, for the client's "what did this cost me".
    ///   - canStartHere: whether the kept region may BEGIN at this turn. See `alignedDrop`.
    /// - Returns: what to prime, and what it cost.
    public static func fit<T>(
        _ history: [T],
        budgetTokens: Int,
        cost: (T) -> Int,
        bytes: (T) -> Int,
        canStartHere: (T) -> Bool
    ) -> (history: [T], droppedTurns: Int, droppedBytes: Int) {
        let costs = history.map(cost)
        var drop = dropCount(costs: costs, budgetTokens: budgetTokens)
        drop = alignedDrop(drop, count: history.count) { canStartHere(history[$0]) }
        guard drop > 0 else { return (history, 0, 0) }
        let gone = history[..<drop]
        return (Array(history[drop...]), drop, gone.reduce(0) { $0 + bytes($1) })
    }

    /// How many leading turns have to go, given what each one costs.
    ///
    /// Index arithmetic rather than a filter over turns, so a caller holding any element type can
    /// drop from its own array using the same rule, stated once.
    public static func dropCount(costs: [Int], budgetTokens: Int) -> Int {
        guard !costs.isEmpty else { return 0 }
        var running = 0
        var kept = 0
        // From the END: what survives is the most recent conversation, which is what the next
        // prompt refers to. ALWAYS at least one turn, even when that turn alone busts the budget -
        // priming a preamble with no conversation after it would answer the next question with no
        // question in front of it, and the caller's report is what makes a remaining overflow
        // visible rather than silent.
        for cost in costs.reversed() {
            if kept > 0 && running + cost > budgetTokens { break }
            kept += 1
            running += cost
        }
        return costs.count - kept
    }

    /// Advance a drop count to the first index the kept region may legally begin at.
    ///
    /// WHY A TRIM CAN CORRUPT A TRANSCRIPT, and the reason this exists. A `tool` turn is only
    /// meaningful after the assistant turn that announced its call: cutting between the two leaves
    /// an ORPHAN tool result, which is the shape a strict chat template (Mistral-family
    /// `raise_exception`) rejects - and llama-server renders that template on every turn, so it
    /// hard-fails the whole session rather than one message. That is the same dead session this
    /// fit was written to prevent, arrived at from the other side.
    ///
    /// Advancing is always safe against the budget: dropping MORE can only make the kept region
    /// smaller. The last turn is never dropped, for the reason `dropCount` gives - a history with
    /// nothing in it answers the next question with no question in front of it - so a tail whose
    /// every candidate start is illegal still keeps its final turn, and the caller's own
    /// well-formedness pass is what has the last word on that shape.
    public static func alignedDrop(_ drop: Int, count: Int, canStartHere: (Int) -> Bool) -> Int {
        guard count > 0 else { return 0 }
        var index = min(max(drop, 0), count - 1)
        while index < count - 1 && !canStartHere(index) { index += 1 }
        return index
    }

    // MARK: - Merging slices

    /// Combine per-slice digests into one, deduplicated and capped.
    ///
    /// Lists are interleaved round-robin across slices rather than concatenated. Each slice's list
    /// arrives in the model's own order - most important first - so taking the first item from
    /// every slice, then the second from every slice, keeps each slice's best material when the
    /// cap bites. Concatenation would let an early slice fill the cap and silence the most recent
    /// one, which is the opposite of what a resumed session needs.
    public static func merge(_ parts: [DigestContent], policy: PrimePolicy = .default)
        -> DigestContent
    {
        let cap = policy.mergedCap(slices: parts.count)
        return DigestContent(
            // Most recent statement of what the user is after wins: intent evolves, and the last
            // slice is the closest thing to now.
            unresolvedIntent: parts.compactMap {
                let t = $0.unresolvedIntent?.trimmingCharacters(in: .whitespacesAndNewlines)
                return (t?.isEmpty ?? true) ? nil : t
            }.last,
            establishedFacts: interleave(parts.map(\.establishedFacts), cap: cap, key: normalized),
            decisions: interleave(parts.map(\.decisions), cap: cap, key: normalized),
            toolEvents: interleave(parts.map(\.toolEvents), cap: cap) {
                normalized("\($0.tool) \($0.whatFor)")
            },
            openThreads: interleave(parts.map(\.openThreads), cap: cap, key: normalized),
            userPreferences: interleave(parts.map(\.userPreferences), cap: cap, key: normalized))
    }

    /// Case- and whitespace-insensitive identity. Two slices describing the same fact rarely match
    /// byte for byte, but they match often enough that this removes visible repetition.
    private static func normalized(_ text: String) -> String {
        text.lowercased().split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    private static func interleave<T>(_ lists: [[T]], cap: Int, key: (T) -> String) -> [T] {
        var out: [T] = []
        var seen = Set<String>()
        var column = 0
        while out.count < cap {
            var sawAny = false
            for list in lists where column < list.count {
                sawAny = true
                let item = list[column]
                let k = key(item)
                guard !k.isEmpty, seen.insert(k).inserted else { continue }
                out.append(item)
                if out.count == cap { break }
            }
            guard sawAny else { break }
            column += 1
        }
        return out
    }

    // MARK: - Assembly

    /// The turns the planner contributes ahead of the verbatim tail: any system turns rescued from
    /// the summarized half, the digest as a user turn, and an acknowledgment when one is needed.
    ///
    /// The acknowledgment exists ONLY to preserve role alternation, so it is emitted only when the
    /// tail would otherwise put a second user turn next to the digest - that is, when the tail is
    /// empty or starts with a user turn. A tail that starts with an assistant turn (the ordinary
    /// shape once tool calls are involved: one user turn followed by many assistant/tool pairs)
    /// already alternates against the digest, and adding an acknowledgment there would create the
    /// back-to-back assistant turns that strict chat templates reject.
    public static func injectedTurns(
        systemTurns: [DigestTurn] = [], digest: SessionDigest, verbatimTail: [DigestTurn]
    ) -> [DigestTurn] {
        var out = systemTurns
        out.append(
            .user(DigestRenderer.renderPreamble(digest, verbatimTurns: verbatimTail.count)))
        let head = verbatimTail.first?.role
        if head == nil || head == .user {
            out.append(.assistant(DigestRenderer.acknowledgment))
        }
        return out
    }

    /// Re-decide the acknowledgment after the verbatim tail has been trimmed under it.
    ///
    /// `injectedTurns` chooses it from the tail it was handed, and its reason is role alternation
    /// against the digest turn. A trim afterwards can change which turn the tail now BEGINS with,
    /// and the choice then produces exactly the shape it was made to avoid: two user turns back to
    /// back when the tail became user-headed, or two assistant turns when it became
    /// assistant-headed. Both are what strict chat templates reject.
    ///
    /// Re-decided here rather than by rebuilding the preamble, because the digest turn carries a
    /// verbatim-turn count rendered into its text; the acknowledgment is the only part of a
    /// preamble that a trim can invalidate.
    public static func realignAcknowledgment(_ injected: [DigestTurn], tailStartsWithUser: Bool)
        -> [DigestTurn]
    {
        let hasAcknowledgment =
            injected.last?.role == .assistant
            && injected.last?.content == DigestRenderer.acknowledgment
        if tailStartsWithUser && !hasAcknowledgment {
            return injected + [.assistant(DigestRenderer.acknowledgment)]
        }
        if !tailStartsWithUser && hasAcknowledgment {
            return Array(injected.dropLast())
        }
        return injected
    }

    /// The full primed history: `injectedTurns` followed by the verbatim tail.
    public static func assemblePrimedHistory(
        systemTurns: [DigestTurn] = [], digest: SessionDigest, verbatimTail: [DigestTurn]
    ) -> [DigestTurn] {
        injectedTurns(systemTurns: systemTurns, digest: digest, verbatimTail: verbatimTail)
            + verbatimTail
    }

    // MARK: - The whole operation

    /// Summarize the old part of `history` and return what to prime.
    ///
    /// Cannot throw and cannot lose a turn: every failure path returns the input history with
    /// `digest == nil` and a `reason` the client can show. See the file header.
    ///
    /// - Parameters:
    ///   - generator: recorded in the digest, e.g. "apple-foundation-models".
    ///   - now: injected so tests are deterministic.
    public static func condense(
        history: [DigestTurn],
        using model: DigestModel,
        policy: PrimePolicy = .default,
        generator: String,
        now: Date = Date()
    ) async -> CondenseResult {
        let (older, tail) = split(history, policy: policy)
        guard !older.isEmpty else {
            return fallback(
                history,
                reason:
                    "nothing to summarize: the conversation is not longer than the "
                    + "\(policy.keepRecentTurns)-message verbatim tail",
                source: "")
        }

        // System turns are instructions, not conversation: summarizing "never write to disk" into
        // one bullet competing for a slot in a capped list is how a stated constraint quietly
        // stops being one. They are rescued verbatim to the front of the primed history instead.
        let systemTurns = older.filter { $0.role == .system }
        let summarizable = older.filter { $0.role != .system }
        guard !summarizable.isEmpty else {
            return fallback(
                history, reason: "nothing to summarize: the older turns are all instructions",
                source: "")
        }

        let pieces = slices(of: summarizable, policy: policy)
        guard !pieces.isEmpty else {
            return fallback(
                history, reason: "nothing to summarize: the older turns carry no text", source: "")
        }
        // Hash and record what the model is actually given, not a parallel rendering of the same
        // turns - slicing normalizes whitespace, so the two are not always the same text, and an
        // audit artifact that records the one nobody read is worse than useless.
        let source = pieces.joined(separator: "\n\n")

        guard pieces.count <= policy.maxSlices else {
            // Each slice is a sequential generation on the critical path of a restore. Refusing is
            // the safe direction: the caller primes everything, which is slow, rather than waiting
            // minutes for a summary it may not even be able to use.
            return fallback(
                history,
                reason:
                    "too large to summarize: \(pieces.count) slices needed, limit is "
                    + "\(policy.maxSlices)",
                source: source)
        }

        var parts: [DigestContent] = []
        var running: DigestContent?
        for piece in pieces {
            // Checked here rather than left to the conformance: cancellation during a multi-slice
            // condensation is the difference between a Stop that takes a second and one that takes
            // minutes, and this loop is the only place that knows how many are left.
            guard !Task.isCancelled else {
                return fallback(history, reason: "summarization was canceled", source: source)
            }
            do {
                // The running merge goes in as prior context so a later slice can resolve
                // references backward instead of summarizing in a vacuum.
                parts.append(try await model.digest(source: piece, priorDigest: running))
                running = merge(parts, policy: policy)
            } catch {
                return fallback(history, reason: describe(error), source: source)
            }
        }

        guard let content = running, !content.isEmpty else {
            // An empty digest is a failed one. Priming a preamble that says nothing WHILE dropping
            // the turns it stands for is worse than not condensing at all.
            return fallback(
                history, reason: "the summarizer returned an empty digest", source: source)
        }

        let digest = SessionDigest(
            content: content,
            sourceTurnCount: summarizable.count,
            sourceSHA256: SessionDigest.sha256Hex(source),
            generator: generator,
            createdAt: now)
        let injected = injectedTurns(
            systemTurns: systemTurns, digest: digest, verbatimTail: tail)
        return CondenseResult(
            history: injected + tail,
            injected: injected,
            tailStartIndex: older.count,
            digest: digest,
            reason: nil,
            droppedTurns: summarizable.count,
            droppedBytes: summarizable.reduce(0) { $0 + $1.content.utf8.count },
            source: source)
    }

    private static func fallback(_ history: [DigestTurn], reason: String, source: String)
        -> CondenseResult
    {
        CondenseResult(
            history: history, injected: [], tailStartIndex: 0, digest: nil, reason: reason,
            droppedTurns: 0, droppedBytes: 0, source: source)
    }

    /// One line, suitable for both a log and a client-facing `reason`.
    ///
    /// The NSError branch reads `userInfo` directly rather than `localizedDescription`, because
    /// EVERY Swift error bridges to NSError and a bridged one answers `localizedDescription` with
    /// "The operation couldn't be completed" - which would replace a useful message with a useless
    /// one for exactly the errors this seam throws.
    private static func describe(_ error: Error) -> String {
        if error is CancellationError { return "summarization was canceled" }
        if let localized = (error as? LocalizedError)?.errorDescription,
            !localized.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            return "summarization failed: \(localized)"
        }
        if let written = (error as NSError).userInfo[NSLocalizedDescriptionKey] as? String,
            !written.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            return "summarization failed: \(written)"
        }
        return "summarization failed: \(error)"
    }
}
