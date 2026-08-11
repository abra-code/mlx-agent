// DigestPlannerTests.swift - the policy, and every way it is allowed to fail.
//
// Roughly half of these assert the same thing from different angles: that a condensation which
// does not fully succeed primes the FULL history. That is the one property whose violation is
// invisible - the session simply knows less than it should, and nothing logs an error - so it is
// worth pinning from the model throwing, from an empty digest, from cancellation, from too many
// slices, and from a history too short to split.
//
// The other property worth stating up front is the partition: `summarize + verbatim` must be the
// input, with no gap and no overlap. Everything downstream splices on that boundary.

import Foundation
import Testing

@testable import AgentDigest

// MARK: - Fakes

/// Returns canned content, one per call, and records what it was asked.
final class RecordingModel: DigestModel, @unchecked Sendable {
    private let lock = NSLock()
    private var canned: [DigestContent]
    private var recordedSources: [String] = []
    private var recordedPriors: [DigestContent?] = []

    init(_ canned: [DigestContent]) { self.canned = canned }

    // Read under the lock like the writes. The reads all happen after `condense` returns today,
    // which is its own happens-before edge - but the type advertises Sendable, and the obvious
    // next change (slicing concurrently) would turn that convention into a race.
    var sources: [String] { lock.withLock { recordedSources } }
    var priors: [DigestContent?] { lock.withLock { recordedPriors } }

    func digest(source: String, priorDigest: DigestContent?) async throws -> DigestContent {
        lock.withLock {
            recordedSources.append(source)
            recordedPriors.append(priorDigest)
            return canned.isEmpty ? DigestContent() : canned.removeFirst()
        }
    }
}

struct ThrowingModel: DigestModel {
    let error: Error
    func digest(source: String, priorDigest: DigestContent?) async throws -> DigestContent {
        throw error
    }
}

struct FailAfterModel: DigestModel {
    let succeedCount: Int
    let counter: Counter
    final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var n = 0
        func next() -> Int { lock.withLock { n += 1; return n } }
    }
    func digest(source: String, priorDigest: DigestContent?) async throws -> DigestContent {
        guard counter.next() <= succeedCount else {
            throw NSError(
                domain: "test", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "the model gave up"])
        }
        return DigestContent(establishedFacts: ["partial"])
    }
}

/// Parks inside the first slice so a test can cancel the surrounding task deterministically.
///
/// Yield-polling rather than a semaphore: `DispatchSemaphore.wait` is unavailable from an async
/// context (it blocks a cooperative thread), and the iteration bound turns a logic error into a
/// failed test rather than a hung suite.
final class GateModel: DigestModel, @unchecked Sendable {
    private let lock = NSLock()
    private var didEnter = false
    private var didRelease = false

    var hasEntered: Bool { lock.withLock { didEnter } }
    func release() { lock.withLock { didRelease = true } }

    func digest(source: String, priorDigest: DigestContent?) async throws -> DigestContent {
        lock.withLock { didEnter = true }
        for _ in 0..<1_000_000 {
            if lock.withLock({ didRelease }) { break }
            await Task.yield()
        }
        return DigestContent(establishedFacts: ["from the first slice"])
    }
}

/// Spin until `condition`, bounded so a broken test fails instead of hanging.
func yieldUntil(_ condition: () -> Bool) async -> Bool {
    for _ in 0..<1_000_000 {
        if condition() { return true }
        await Task.yield()
    }
    return false
}

// MARK: - Helpers

func alternating(_ count: Int, bodySize: Int = 20) -> [DigestTurn] {
    (0..<count).map { i in
        let body = String(repeating: "\(i % 10)", count: bodySize)
        return i % 2 == 0 ? .user("u\(i) " + body) : .assistant("a\(i) " + body)
    }
}

/// Long enough to need several slices at a small budget.
func bulky(_ count: Int, chunk: Int = 1500) -> [DigestTurn] {
    (0..<count).map { i in
        let body = String(repeating: "z", count: chunk)
        return i % 2 == 0 ? .user("\(i) " + body) : .assistant("\(i) " + body)
    }
}

let filledContent = DigestContent(
    unresolvedIntent: "finish the notarization step",
    establishedFacts: ["the binary is arm64-only"],
    decisions: ["ship a pkg"],
    toolEvents: [ToolEvent(tool: "read_file", whatFor: "entitlements", materialResult: "no sandbox")],
    openThreads: ["stapling untested"],
    userPreferences: ["ASCII only"])

// MARK: - Tests

@Suite("DigestPlanner: splitting")
struct DigestPlannerSplitTests {

    /// The property everything downstream splices on.
    @Test(
        "the split partitions the input exactly",
        arguments: [0, 1, 2, 5, 6, 7, 13, 20, 41])
    func partitions(count: Int) {
        let history = alternating(count)
        for keep in [2, 6, 9, 64] {
            let (older, tail) = DigestPlanner.split(
                history, policy: PrimePolicy(keepRecentTurns: keep))
            #expect(older + tail == history)
        }
    }

    @Test("the tail is kept verbatim and begins with a user turn")
    func tailIsVerbatim() {
        let history = alternating(20)
        let (older, tail) = DigestPlanner.split(history, policy: PrimePolicy(keepRecentTurns: 6))
        #expect(older.count == 14)
        #expect(tail.count == 6)
        #expect(tail.first?.role == .user)
        #expect(tail == Array(history.suffix(6)))
    }

    @Test("the boundary snaps backward so the tail never starts mid-exchange")
    func boundarySnapsBackward() {
        // 21 turns: index 15 (the naive boundary) is an assistant reply, so the split must move
        // back to 14 and keep one extra message rather than orphan a response.
        let history = alternating(21)
        let (older, tail) = DigestPlanner.split(history, policy: PrimePolicy(keepRecentTurns: 6))
        #expect(older.count == 14)
        #expect(tail.count == 7)
        #expect(tail.first?.role == .user)
    }

    @Test("a history no longer than the tail is summarized not at all")
    func shortHistoryUntouched() {
        let history = alternating(6)
        let (older, tail) = DigestPlanner.split(history, policy: PrimePolicy(keepRecentTurns: 6))
        #expect(older.isEmpty)
        #expect(tail == history)
    }

    @Test("the backward search is bounded, so a lone early user turn cannot cancel the split")
    func backwardSearchBounded() {
        // One user turn at the start, then a long monologue. An unbounded search for a user
        // boundary would walk all the way to index 0 and condense nothing at all.
        var history: [DigestTurn] = [.user("start")]
        history += (1..<20).map { .assistant("reply \($0)") }
        let (older, tail) = DigestPlanner.split(history, policy: PrimePolicy(keepRecentTurns: 6))
        #expect(older.count == 14)
        #expect(tail.count == 6)
    }

    @Test("with no user turn in reach, the full verbatim tail is still kept")
    func noUserInReach() {
        // The shape an agentic loop produces: one user turn, then assistant/tool pairs forever.
        // Searching FORWARD for a user boundary here would keep 1 message verbatim instead of 6,
        // which is the unsafe direction - so the boundary stays put and the acknowledgment turn
        // is what adapts (see the assembly suite).
        var history: [DigestTurn] = [.user("do the thing")]
        for i in 0..<19 {
            history.append(.assistant("calling tool \(i)"))
            history.append(.tool("result \(i)"))
        }
        let (older, tail) = DigestPlanner.split(history, policy: PrimePolicy(keepRecentTurns: 6))
        #expect(tail.count == 6)
        #expect(tail.first?.role != .user)
        #expect(older + tail == history)
    }
}

@Suite("DigestPlanner: slicing")
struct DigestPlannerSliceTests {

    @Test("slices stay inside the token budget")
    func slicesFitBudget() {
        let policy = PrimePolicy(sliceBudgetTokens: 256)
        let turns = (0..<10).map { DigestTurn.user(String(repeating: "x", count: 400) + "\($0)") }
        let pieces = DigestPlanner.slices(of: turns, policy: policy)
        #expect(pieces.count > 1)
        for piece in pieces { #expect(DigestPlanner.estimateTokens(piece) <= 256) }
    }

    @Test("CJK slices stay inside the budget too")
    func cjkFitsBudget() {
        // The case bytes/4 alone gets wrong: ~3 UTF-8 bytes per character at roughly a token per
        // character, so a byte-only budget would over-fill every slice by about a third.
        let policy = PrimePolicy(sliceBudgetTokens: 256)
        let turns = (0..<6).map { _ in
            DigestTurn.user(String(repeating: "\u{5305}\u{88C5}\u{624B}\u{9806}", count: 200))
        }
        let pieces = DigestPlanner.slices(of: turns, policy: policy)
        for piece in pieces { #expect(DigestPlanner.estimateTokens(piece) <= 256) }
        #expect(pieces.count >= 4)
    }

    @Test("one oversized turn is split rather than dropped or sent whole")
    func oversizedTurnSplit() {
        let policy = PrimePolicy(sliceBudgetTokens: 256)
        let huge = DigestTurn.user(String(repeating: "y", count: 5000))
        let pieces = DigestPlanner.slices(of: [huge], policy: policy)
        #expect(pieces.count >= 5)
        for piece in pieces { #expect(DigestPlanner.estimateTokens(piece) <= 256) }
        // Nothing vanished: every 'y' is still accounted for.
        #expect(pieces.joined().filter { $0 == "y" }.count == 5000)
    }

    @Test("continuation pieces keep their role label")
    func continuationLabeled() {
        // Without this the second piece onward is anonymous mid-sentence prose and the summarizer
        // cannot tell the user's words from a pasted tool result.
        let policy = PrimePolicy(sliceBudgetTokens: 256)
        let pieces = DigestPlanner.slices(
            of: [.user(String(repeating: "w", count: 4000))], policy: policy)
        #expect(pieces.count > 1)
        #expect(pieces[0].hasPrefix("User: "))
        for piece in pieces.dropFirst() { #expect(piece.hasPrefix("User (continued): ")) }
    }

    @Test("multi-turn content is covered in full")
    func multiTurnCoverage() {
        let turns = (0..<12).map { DigestTurn.user("marker\($0) " + String(repeating: "q", count: 900)) }
        let joined = DigestPlanner.slices(of: turns, policy: PrimePolicy(sliceBudgetTokens: 256))
            .joined(separator: "\n")
        for i in 0..<12 { #expect(joined.contains("marker\(i)")) }
    }

    @Test("empty turns contribute nothing")
    func emptyTurnsSkipped() {
        #expect(DigestPlanner.slices(of: [.user("   "), .assistant("")]).isEmpty)
    }

    @Test("hard splitting never cuts a character in half")
    func hardSplitRespectsCharacters() {
        // Each emoji is 4 UTF-8 bytes and one non-ASCII scalar; a byte-oriented split would
        // produce invalid sequences.
        let text = String(repeating: "\u{1F600}", count: 100)
        let pieces = DigestPlanner.hardSplit(text, maxTokens: 2)
        #expect(pieces.count == 50)
        for piece in pieces { #expect(DigestPlanner.estimateTokens(piece) <= 2) }
        #expect(pieces.joined() == text)
    }

    @Test("roles are labeled so the model can tell who said what")
    func sourceIsLabeled() {
        let source = DigestPlanner.renderSource([.user("hello"), .assistant("hi"), .tool("42")])
        #expect(source == "User: hello\n\nAssistant: hi\n\nTool result: 42")
    }

    @Test("the estimator does not under-report non-ASCII")
    func estimator() {
        #expect(DigestPlanner.estimateTokens(String(repeating: "a", count: 400)) == 100)
        // 400 CJK characters: 1200 bytes, so bytes/4 would say 300. One token per character is
        // the safer read.
        #expect(DigestPlanner.estimateTokens(String(repeating: "\u{6F22}", count: 400)) == 400)
    }
}

@Suite("DigestPlanner: merging")
struct DigestPlannerMergeTests {

    @Test("lists interleave across slices instead of letting the first fill the cap")
    func interleaved() {
        let a = DigestContent(establishedFacts: ["a1", "a2", "a3"])
        let b = DigestContent(establishedFacts: ["b1", "b2", "b3"])
        let merged = DigestPlanner.merge([a, b], policy: PrimePolicy(maxItemsPerList: 3))
        #expect(merged.establishedFacts == ["a1", "b1", "a2", "b2", "a3", "b3"])
    }

    @Test("the cap is per slice, scaled to how many contributed, and ceilinged at 3x")
    func capScales() {
        func facts(_ prefix: String) -> DigestContent {
            DigestContent(establishedFacts: (1...8).map { "\(prefix)\($0)" })
        }
        let policy = PrimePolicy(maxItemsPerList: 3)
        // One slice cannot exceed its own cap; more slices earn more room, up to 3x.
        #expect(DigestPlanner.merge([facts("a")], policy: policy).establishedFacts.count == 3)
        #expect(DigestPlanner.merge([facts("a"), facts("b")], policy: policy)
            .establishedFacts.count == 6)
        let five = (1...5).map { facts("s\($0)") }
        #expect(DigestPlanner.merge(five, policy: policy).establishedFacts.count == 9)
    }

    @Test("duplicates collapse case- and whitespace-insensitively")
    func dedup() {
        let a = DigestContent(decisions: ["Ship a  PKG"])
        let b = DigestContent(decisions: ["ship a pkg", "sign it"])
        let merged = DigestPlanner.merge([a, b])
        #expect(merged.decisions == ["Ship a  PKG", "sign it"])
    }

    @Test("blank items never become bullets")
    func blanksDropped() {
        let merged = DigestPlanner.merge([DigestContent(openThreads: ["", "   ", "real"])])
        #expect(merged.openThreads == ["real"])
    }

    @Test("the most recent intent wins")
    func latestIntent() {
        let merged = DigestPlanner.merge([
            DigestContent(unresolvedIntent: "old"),
            DigestContent(unresolvedIntent: "   "),
            DigestContent(unresolvedIntent: "current"),
        ])
        #expect(merged.unresolvedIntent == "current")
    }

    @Test("tool events dedup on tool plus purpose")
    func toolEventDedup() {
        let a = DigestContent(toolEvents: [
            ToolEvent(tool: "read_file", whatFor: "config", materialResult: "x")
        ])
        let b = DigestContent(toolEvents: [
            ToolEvent(tool: "read_file", whatFor: "config", materialResult: "y"),
            ToolEvent(tool: "write_file", whatFor: "config", materialResult: "z"),
        ])
        let merged = DigestPlanner.merge([a, b])
        #expect(merged.toolEvents.count == 2)
        #expect(merged.toolEvents.first?.materialResult == "x")
    }

    @Test("merging one slice returns it")
    func identity() {
        #expect(DigestPlanner.merge([filledContent]) == filledContent)
    }
}

@Suite("DigestPlanner: assembly")
struct DigestPlannerAssemblyTests {

    static func digest() -> SessionDigest {
        SessionDigest(
            content: filledContent, sourceTurnCount: 14, sourceSHA256: "abc",
            generator: "fake", createdAt: Date(timeIntervalSince1970: 0))
    }

    @Test("a user-led tail gets the acknowledgment, so roles alternate")
    func ackWhenTailStartsWithUser() {
        let tail: [DigestTurn] = [.user("next"), .assistant("sure")]
        let turns = DigestPlanner.assemblePrimedHistory(digest: Self.digest(), verbatimTail: tail)
        #expect(turns.map(\.role) == [.user, .assistant, .user, .assistant])
    }

    @Test("an assistant-led tail gets NO acknowledgment, because it would break alternation")
    func noAckWhenTailStartsWithAssistant() {
        // The agentic shape. With an acknowledgment this would be user, assistant, assistant...
        let tail: [DigestTurn] = [.assistant("calling a tool"), .tool("result")]
        let turns = DigestPlanner.assemblePrimedHistory(digest: Self.digest(), verbatimTail: tail)
        #expect(turns.map(\.role) == [.user, .assistant, .tool])
    }

    @Test("an empty tail still gets the pair")
    func ackWithEmptyTail() {
        let turns = DigestPlanner.assemblePrimedHistory(digest: Self.digest(), verbatimTail: [])
        #expect(turns.map(\.role) == [.user, .assistant])
        #expect(turns[1].content == DigestRenderer.acknowledgment)
    }

    @Test("hoisted system turns lead")
    func systemTurnsLead() {
        let turns = DigestPlanner.assemblePrimedHistory(
            systemTurns: [.system("never write to disk")], digest: Self.digest(),
            verbatimTail: [.user("go")])
        #expect(turns.map(\.role) == [.system, .user, .assistant, .user])
        #expect(turns[0].content == "never write to disk")
    }
}

@Suite("DigestPlanner: condensing")
struct DigestPlannerCondenseTests {

    static let stamp = Date(timeIntervalSince1970: 1_700_000_000.75)

    @Test("a successful condensation primes preamble, acknowledgment, then the verbatim tail")
    func assembly() async {
        let history = alternating(20)
        let result = await DigestPlanner.condense(
            history: history, using: RecordingModel([filledContent]),
            policy: PrimePolicy(keepRecentTurns: 6), generator: "fake", now: Self.stamp)

        #expect(result.condensed)
        #expect(result.history.count == 8)  // preamble + ack + 6 verbatim
        #expect(result.history[0].role == .user)
        #expect(result.history[1].content == DigestRenderer.acknowledgment)
        #expect(Array(result.history.dropFirst(2)) == Array(history.suffix(6)))
        #expect(result.history[0].content.contains("finish the notarization step"))
    }

    @Test("history equals injected plus the input from tailStartIndex")
    func spliceIdentity() async {
        // The identity the consumer splices on. It re-reads the tail from its OWN richer message
        // type, so if this ever stops holding it silently duplicates or loses turns.
        for count in [8, 20, 21] {
            let history = alternating(count)
            let result = await DigestPlanner.condense(
                history: history, using: RecordingModel([filledContent]), generator: "fake",
                now: Self.stamp)
            #expect(
                result.history == result.injected + Array(history.dropFirst(result.tailStartIndex)))
        }
    }

    @Test("a fallback reports an untouched input through the same identity")
    func spliceIdentityOnFallback() async {
        let history = alternating(4)
        let result = await DigestPlanner.condense(
            history: history, using: RecordingModel([filledContent]), generator: "fake",
            now: Self.stamp)
        #expect(result.injected.isEmpty)
        #expect(result.tailStartIndex == 0)
        #expect(result.history == history)
    }

    @Test("provenance describes exactly what was summarized")
    func provenance() async throws {
        let history = alternating(20)
        let result = await DigestPlanner.condense(
            history: history, using: RecordingModel([filledContent]),
            policy: PrimePolicy(keepRecentTurns: 6), generator: "fake", now: Self.stamp)

        let digest = try #require(result.digest)
        #expect(digest.sourceTurnCount == 14)
        #expect(digest.generator == "fake")
        #expect(digest.createdAt == Self.stamp)
        // The hash covers what the model was actually given, not a parallel rendering of it.
        #expect(digest.sourceSHA256 == SessionDigest.sha256Hex(result.source))
        #expect(result.source == RecordingModel([]).sources.joined() || !result.source.isEmpty)
        #expect(result.droppedTurns == 14)
        #expect(
            result.droppedBytes == history.prefix(14).reduce(0) { $0 + $1.content.utf8.count })
        // The verbatim tail is not in the summarized source.
        #expect(!result.source.contains("u18"))
    }

    @Test("the hashed source is exactly what the model read")
    func sourceIsWhatWasSent() async {
        let history = alternating(20)
        let model = RecordingModel([filledContent])
        let result = await DigestPlanner.condense(
            history: history, using: model, generator: "fake", now: Self.stamp)
        #expect(result.source == model.sources.joined(separator: "\n\n"))
    }

    @Test("a throwing model primes the full history unchanged")
    func fallbackOnThrow() async {
        let history = alternating(20)
        let result = await DigestPlanner.condense(
            history: history,
            using: ThrowingModel(
                error: NSError(
                    domain: "test", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Apple Intelligence is off"])),
            generator: "fake", now: Self.stamp)

        #expect(!result.condensed)
        #expect(result.digest == nil)
        #expect(result.history == history)
        #expect(result.droppedTurns == 0)
        #expect(result.reason?.contains("Apple Intelligence is off") == true)
    }

    @Test("a failure on a later slice still primes everything")
    func fallbackMidway() async {
        // Partial success is not success: half a digest plus no history is the worst outcome.
        let history = bulky(40)
        let result = await DigestPlanner.condense(
            history: history, using: FailAfterModel(succeedCount: 1, counter: .init()),
            policy: PrimePolicy(sliceBudgetTokens: 512, maxSlices: 64), generator: "fake",
            now: Self.stamp)

        #expect(!result.condensed)
        #expect(result.history == history)
        #expect(result.reason?.contains("the model gave up") == true)
    }

    @Test("an empty digest counts as a failure")
    func fallbackOnEmptyDigest() async {
        let history = alternating(20)
        let result = await DigestPlanner.condense(
            history: history, using: RecordingModel([DigestContent()]), generator: "fake",
            now: Self.stamp)

        #expect(!result.condensed)
        #expect(result.history == history)
        #expect(result.reason?.contains("empty digest") == true)
    }

    @Test("a digest of only blank strings counts as a failure too")
    func fallbackOnBlankDigest() async {
        let history = alternating(20)
        let blank = DigestContent(unresolvedIntent: "  ", establishedFacts: ["", "   "])
        let result = await DigestPlanner.condense(
            history: history, using: RecordingModel([blank]), generator: "fake", now: Self.stamp)
        #expect(!result.condensed)
        #expect(result.history == history)
    }

    @Test("a short history is left alone with a reason, not condensed to nothing")
    func nothingToDo() async {
        let history = alternating(4)
        let model = RecordingModel([filledContent])
        let result = await DigestPlanner.condense(
            history: history, using: model, generator: "fake", now: Self.stamp)

        #expect(!result.condensed)
        #expect(result.history == history)
        #expect(result.reason?.contains("nothing to summarize") == true)
        #expect(model.sources.isEmpty)  // the model was never called
    }

    @Test("too many slices falls back instead of running for minutes")
    func sliceBudgetRefused() async {
        // Slice SIZE is capped by policy; slice COUNT is capped by this. Each one is a sequential
        // generation on the critical path of a restore.
        let history = bulky(60)
        let model = RecordingModel(Array(repeating: filledContent, count: 100))
        let result = await DigestPlanner.condense(
            history: history, using: model,
            policy: PrimePolicy(sliceBudgetTokens: 256, maxSlices: 3), generator: "fake",
            now: Self.stamp)

        #expect(!result.condensed)
        #expect(result.history == history)
        #expect(result.reason?.contains("too large to summarize") == true)
        #expect(model.sources.isEmpty)  // refused before spending a single call
    }

    @Test("the model is called once per slice and no more")
    func callCountBounded() async {
        let history = bulky(40)
        let model = RecordingModel(Array(repeating: filledContent, count: 100))
        let policy = PrimePolicy(sliceBudgetTokens: 512, maxSlices: 64)
        let expected = DigestPlanner.slices(
            of: DigestPlanner.split(history, policy: policy).summarize, policy: policy
        ).count
        _ = await DigestPlanner.condense(
            history: history, using: model, policy: policy, generator: "fake", now: Self.stamp)
        #expect(model.sources.count == expected)
        #expect(model.sources.count <= policy.maxSlices)
    }

    @Test("cancelling mid-condensation falls back rather than grinding through every slice")
    func cancellation() async {
        let history = bulky(40)
        let model = GateModel()
        let policy = PrimePolicy(sliceBudgetTokens: 512, maxSlices: 64)
        let task = Task {
            await DigestPlanner.condense(
                history: history, using: model, policy: policy, generator: "fake", now: Self.stamp)
        }
        // Wait until the first slice is genuinely in flight, then cancel and let it finish. The
        // loop must not start slice two.
        #expect(await yieldUntil { model.hasEntered })
        task.cancel()
        model.release()
        let result = await task.value

        #expect(!result.condensed)
        #expect(result.history == history)
        #expect(result.reason?.contains("cancelled") == true)
    }

    @Test("system turns are kept verbatim rather than summarized into a bullet")
    func systemTurnsHoisted() async {
        var history: [DigestTurn] = [.system("Never write to disk.")]
        history += alternating(20)
        let model = RecordingModel([filledContent])
        let result = await DigestPlanner.condense(
            history: history, using: model, policy: PrimePolicy(keepRecentTurns: 6),
            generator: "fake", now: Self.stamp)

        #expect(result.condensed)
        #expect(result.history.first?.role == .system)
        #expect(result.history.first?.content == "Never write to disk.")
        // And it was never handed to the summarizer to be paraphrased away.
        #expect(!(model.sources.first ?? "").contains("Never write to disk"))
        #expect(result.droppedTurns == 14)  // the system turn is kept, so not counted as dropped
    }

    @Test("later slices are given what earlier slices established")
    func priorDigestThreaded() async {
        let history = bulky(40, chunk: 1500)
        let model = RecordingModel(
            Array(repeating: DigestContent(establishedFacts: ["fact"]), count: 20))
        _ = await DigestPlanner.condense(
            history: history, using: model,
            policy: PrimePolicy(sliceBudgetTokens: 512, maxSlices: 64), generator: "fake",
            now: Self.stamp)

        #expect(model.sources.count > 1)
        #expect(model.priors.first ?? nil == nil)  // the first slice has no prior
        #expect(model.priors.dropFirst().allSatisfy { $0 != nil })
    }
}

@Suite("PrimePolicy")
struct PrimePolicyTests {

    @Test("wire values are clamped rather than honored or rejected")
    func clamping() {
        // keepRecentTurns: 0 would summarize the question the user is waiting on.
        #expect(PrimePolicy(keepRecentTurns: 0).keepRecentTurns == 2)
        #expect(PrimePolicy(keepRecentTurns: 9999).keepRecentTurns == 64)
        // A slice budget over the context window guarantees an overflow.
        #expect(PrimePolicy(sliceBudgetTokens: 1_000_000).sliceBudgetTokens == 8192)
        #expect(PrimePolicy(sliceBudgetTokens: -5).sliceBudgetTokens == 256)
        #expect(PrimePolicy(maxDigestTokens: 1).maxDigestTokens == 128)
        // A one-item-per-list digest passes every emptiness check and is still a stub.
        #expect(PrimePolicy(maxItemsPerList: 0).maxItemsPerList == 3)
        #expect(PrimePolicy(maxSlices: 0).maxSlices == 1)
        #expect(PrimePolicy(maxSlices: 100_000).maxSlices == 256)
    }

    @Test("the defaults are the documented ones")
    func defaults() {
        let p = PrimePolicy.default
        #expect(p.keepRecentTurns == 6)
        #expect(p.maxDigestTokens == 700)
        #expect(p.sliceBudgetTokens == 2800)
        #expect(p.maxItemsPerList == 8)
        #expect(p.maxSlices == 16)
    }

    @Test("the merged cap scales with slices and stops at 3x")
    func mergedCap() {
        let p = PrimePolicy(maxItemsPerList: 4)
        #expect(p.mergedCap(slices: 0) == 4)
        #expect(p.mergedCap(slices: 1) == 4)
        #expect(p.mergedCap(slices: 3) == 12)
        #expect(p.mergedCap(slices: 50) == 12)
    }
}
