// PrimeFitTests.swift - making a restored conversation fit the model that has to hold it.
//
// The failure this closes, measured: a conversation was switched onto a llama-server whose context
// had been sized down to 8192 tokens, condensation was refused ("too large to summarize"), the
// planner fell back to the full history as it is designed to, and the first message afterwards
// came back as an HTTP 400 quoting 11036 tokens against 8192. Every layer behaved as documented
// and the session was dead.
//
// Three properties are pinned here. What comes back must FIT; it must be the END of the
// conversation rather than the beginning; and it must be a transcript a chat template will accept,
// which is the one a trim is most likely to break - cutting between an assistant's tool call and
// the result answering it leaves an orphan that strict templates reject outright, turning one
// dead session into another.

import Foundation
import Testing

@testable import AgentDigest

@Suite("DigestPlanner: fitting a prime to the serving window")
struct PrimeFitTests {

    /// ~50 tokens each (200 bytes), so a budget in the hundreds selects a predictable count.
    private func sized(_ count: Int, bytes: Int = 200) -> [DigestTurn] {
        (0..<count).map { i in
            let body = String(repeating: "z", count: max(1, bytes - 4))
            return i % 2 == 0 ? .user("\(i) " + body) : .assistant("\(i) " + body)
        }
    }

    /// The production caller costs tool-call arguments that live outside `content`; a DigestTurn
    /// has no such thing, so here cost and bytes are both about the text.
    private func fit(_ history: [DigestTurn], budget: Int) -> (
        history: [DigestTurn], droppedTurns: Int, droppedBytes: Int
    ) {
        DigestPlanner.fit(
            history, budgetTokens: budget,
            cost: { DigestPlanner.estimateTokens($0.content) + DigestPlanner.turnOverheadTokens },
            bytes: { $0.content.utf8.count },
            canStartHere: { $0.role != .tool })
    }

    private func tokens(_ history: [DigestTurn]) -> Int {
        history.reduce(0) {
            $0 + DigestPlanner.estimateTokens($1.content) + DigestPlanner.turnOverheadTokens
        }
    }

    @Test("a history that already fits is returned untouched")
    func alreadyFits() {
        let history = sized(6)
        let result = fit(history, budget: 100_000)
        #expect(result.history == history)
        #expect(result.droppedTurns == 0)
        #expect(result.droppedBytes == 0)
    }

    @Test("what comes back fits the budget it was given")
    func resultFits() {
        let history = sized(40)
        for budget in [200, 500, 1200, 3000] {
            #expect(tokens(fit(history, budget: budget).history) <= budget)
        }
    }

    @Test("it is the OLDEST turns that go")
    func dropsFromTheFront() {
        let history = sized(20)
        let result = fit(history, budget: 600)
        #expect(result.droppedTurns > 0)
        #expect(result.history == Array(history.suffix(result.history.count)))
        #expect(result.history.last == history.last)
    }

    @Test("the count and the bytes describe what actually went")
    func reportsWhatWent() {
        let history = sized(20)
        let result = fit(history, budget: 600)
        #expect(result.droppedTurns == history.count - result.history.count)
        let gone = history.prefix(result.droppedTurns)
        #expect(result.droppedBytes == gone.reduce(0) { $0 + $1.content.utf8.count })
    }

    /// A history with nothing in it answers the next question with no question in front of it.
    /// The remaining overflow is then reported rather than fixed, which is the caller's job.
    @Test("at least one turn survives an impossible budget")
    func alwaysOneTurn() {
        let history = sized(5, bytes: 4000)
        let result = fit(history, budget: 10)
        #expect(result.history == [history[4]])
        #expect(tokens(result.history) > 10)
    }

    @Test("an empty history is not something to trim")
    func empty() {
        let result = fit([], budget: 10)
        #expect(result.history.isEmpty)
        #expect(result.droppedTurns == 0)
    }

    @Test("a negative budget still leaves the conversation something to answer")
    func negativeBudget() {
        let history = sized(4)
        let result = fit(history, budget: -1)
        #expect(result.history == [history[3]])
        #expect(result.droppedTurns == 3)
    }

    // MARK: - The boundary a trim must not leave behind

    /// THE DEFECT THIS SECTION EXISTS FOR. A tool result only means anything after the assistant
    /// turn that announced its call. Cutting between the two leaves an orphan, which a strict chat
    /// template (Mistral-family `raise_exception`) refuses - and llama-server renders that template
    /// on every turn, so it kills the session rather than the message. That is the same dead
    /// session the fit was written to prevent, reached from the other side.
    @Test("a trim never leaves a tool result as the first turn")
    func neverStartsWithAnOrphanToolResult() {
        let history: [DigestTurn] = [
            .user("what is in the file"),
            .assistant(String(repeating: "x", count: 3000)),   // the announcement, and it is big
            DigestTurn(role: .tool, content: "ok"),
            .assistant("done"),
        ]
        let result = fit(history, budget: 120)
        #expect(result.history.first?.role != .tool)
        #expect(result.history == [.assistant("done")])
    }

    /// Several in a row, since one call can produce several results.
    @Test("it advances past every leading tool result, not just one")
    func advancesPastRunsOfToolResults() {
        let history: [DigestTurn] = [
            .assistant(String(repeating: "x", count: 3000)),
            DigestTurn(role: .tool, content: "a"),
            DigestTurn(role: .tool, content: "b"),
            DigestTurn(role: .tool, content: "c"),
            .user("and now"),
        ]
        let result = fit(history, budget: 60)
        #expect(result.history == [.user("and now")])
    }

    /// Dropping MORE is always safe against a budget, so the alignment can never overflow it.
    @Test("aligning the boundary cannot break the budget it just satisfied")
    func alignmentStaysInBudget() {
        let history: [DigestTurn] = [
            .user("q"), .assistant(String(repeating: "x", count: 2000)),
            DigestTurn(role: .tool, content: String(repeating: "y", count: 400)),
            .assistant("small"),
        ]
        let result = fit(history, budget: 200)
        #expect(tokens(result.history) <= 200)
        #expect(result.history.first?.role != .tool)
    }

    /// The last turn is never dropped, so a tail whose only candidate start is illegal keeps it
    /// anyway - the caller's own well-formedness pass has the last word on that shape.
    @Test("a tail that is nothing but tool results still keeps its last turn")
    func allToolResults() {
        let history = [
            DigestTurn(role: .tool, content: "a"),
            DigestTurn(role: .tool, content: "b"),
        ]
        let result = fit(history, budget: 1)
        #expect(result.history.count == 1)
        #expect(result.history.first?.content == "b")
    }

    // MARK: - The acknowledgment the trim invalidates

    /// `injectedTurns` decides the acknowledgment from the tail it is handed. A trim changes which
    /// turn the tail begins with, and the decision then produces exactly the back-to-back roles it
    /// was made to avoid.
    @Test("an acknowledgment is dropped when the tail stops starting with a user turn")
    func acknowledgmentDropped() {
        let injected: [DigestTurn] = [
            .user("summary of 60 turns"), .assistant(DigestRenderer.acknowledgment),
        ]
        let realigned = DigestPlanner.realignAcknowledgment(injected, tailStartsWithUser: false)
        #expect(realigned == [.user("summary of 60 turns")])
    }

    @Test("and added when it starts with one and there was none")
    func acknowledgmentAdded() {
        let injected: [DigestTurn] = [.user("summary of 60 turns")]
        let realigned = DigestPlanner.realignAcknowledgment(injected, tailStartsWithUser: true)
        #expect(realigned.count == 2)
        #expect(realigned.last == .assistant(DigestRenderer.acknowledgment))
    }

    @Test("a decision that is already right is left alone", arguments: [true, false])
    func acknowledgmentUnchanged(startsWithUser: Bool) {
        let base: [DigestTurn] = [.system("never write to disk"), .user("summary")]
        let injected =
            startsWithUser ? base + [.assistant(DigestRenderer.acknowledgment)] : base
        #expect(
            DigestPlanner.realignAcknowledgment(injected, tailStartsWithUser: startsWithUser)
                == injected)
    }

    /// Only the acknowledgment is ever removed. A digest whose preamble was rebuilt into nothing
    /// would drop the summary while keeping the turns it stands in for.
    @Test("an assistant turn that is not the acknowledgment is never mistaken for one")
    func onlyTheAcknowledgment() {
        let injected: [DigestTurn] = [.user("summary"), .assistant("a real answer")]
        #expect(
            DigestPlanner.realignAcknowledgment(injected, tailStartsWithUser: false) == injected)
    }

    // MARK: - The shared arithmetic

    @Test("every turn is charged for more than its own text")
    func perTurnOverhead() {
        #expect(
            DigestPlanner.dropCount(costs: [10, 10, 10], budgetTokens: 20) == 1,
            "two turns of ten fit a budget of twenty; three do not")
    }

    @Test("dropCount keeps as much of the end as fits")
    func dropCountKeepsTheEnd() {
        #expect(DigestPlanner.dropCount(costs: [], budgetTokens: 100) == 0)
        #expect(DigestPlanner.dropCount(costs: [5, 5, 5], budgetTokens: 100) == 0)
        #expect(DigestPlanner.dropCount(costs: [5, 5, 5], budgetTokens: 0) == 2)
    }
}
