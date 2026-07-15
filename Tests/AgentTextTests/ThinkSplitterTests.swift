import Foundation
import Testing

import AgentText

// Collapse a segment list into one string per kind, so assertions read against the text the
// user would actually see rather than against incidental chunk boundaries.
private func joined(_ segs: [(ThinkSplitter.Kind, String)], _ kind: ThinkSplitter.Kind) -> String {
    segs.filter { $0.0 == kind }.map(\.1).joined()
}

@Suite("ThinkSplitter")
struct ThinkSplitterTests {

    @Test("plain text with no markers is all message")
    func plainText() {
        let s = ThinkSplitter()
        let out = s.feed("hello world") + s.flush()
        #expect(joined(out, .message) == "hello world")
        #expect(joined(out, .thought) == "")
    }

    @Test("a complete think block splits into thought and message")
    func completeBlock() {
        let s = ThinkSplitter()
        let out = s.feed("<think>reasoning</think>answer") + s.flush()
        #expect(joined(out, .thought) == "reasoning")
        #expect(joined(out, .message) == "answer")
    }

    @Test("a marker split across chunk boundaries is still recognised")
    func markerStraddlesChunks() {
        let s = ThinkSplitter()
        var out: [(ThinkSplitter.Kind, String)] = []
        // "<think>" arriving one character at a time is the case `keep` exists for.
        for ch in "<think>reasoning</think>answer" {
            out += s.feed(String(ch))
        }
        out += s.flush()
        #expect(joined(out, .thought) == "reasoning")
        #expect(joined(out, .message) == "answer")
    }

    @Test("flush emits the withheld tail")
    func flushEmitsTail() {
        let s = ThinkSplitter()
        // Shorter than `keep` (6 < 7), so feed alone must withhold all of it.
        let fed = s.feed("abcdef")
        #expect(joined(fed, .message) == "")
        #expect(joined(s.flush(), .message) == "abcdef")
    }

    @Test("flush preserves inThink so a block can span a tool call")
    func flushPreservesInThink() {
        let s = ThinkSplitter()
        var out = s.feed("<think>before")
        out += s.flush()                       // end of the pass that made a tool call
        out += s.feed("after</think>answer")   // the pass after the tool result
        out += s.flush()
        #expect(joined(out, .thought) == "beforeafter")
        #expect(joined(out, .message) == "answer")
    }

    // The regression this suite exists for. A model that emits an unknown marker right
    // before a tool call used to have it torn in two: `feed` emits everything except the
    // last `keep` (7) characters, and the tool-call path never flushed, so the tail sat in
    // the buffer and reappeared prepended to the NEXT pass's answer. "<|channel|>" is 11
    // characters, which is why the transcript showed "<|ch" ending one message and
    // "annel|>" starting the next.
    @Test("no text is stranded across a flushed pass boundary")
    func noCarryOverAcrossPasses() {
        let s = ThinkSplitter()
        var out = s.feed("Looking that up.<|channel|>")
        out += s.flush()                        // the fix: flush at the END OF EVERY PASS
        #expect(joined(out, .message) == "Looking that up.<|channel|>")

        // The next pass must start clean - nothing left over glued to its front.
        let next = s.feed("The time is 8:14 AM.") + s.flush()
        #expect(joined(next, .message) == "The time is 8:14 AM.")
        #expect(!joined(next, .message).hasPrefix("annel|>"))
    }

    @Test("an unflushed pass strands exactly the withheld tail")
    func unflushedPassStrandsTail() {
        // Pins the mechanism itself, so this stays honest if `keep` ever changes: WITHOUT
        // the flush, the last 7 characters of "<|channel|>" are exactly what goes missing.
        let s = ThinkSplitter()
        let out = s.feed("Looking that up.<|channel|>")
        #expect(joined(out, .message) == "Looking that up.<|ch")
        #expect(joined(s.flush(), .message) == "annel|>")
    }

    @Test("text is never lost or reordered across many small chunks")
    func noLossAcrossManyChunks() {
        let source = "<think>step one. step two.</think>Here is the answer, at last."
        let s = ThinkSplitter()
        var out: [(ThinkSplitter.Kind, String)] = []
        for chunk in stride(from: 0, to: source.count, by: 3).map({ i -> String in
            let start = source.index(source.startIndex, offsetBy: i)
            let end = source.index(start, offsetBy: min(3, source.count - i))
            return String(source[start..<end])
        }) {
            out += s.feed(chunk)
        }
        out += s.flush()
        #expect(joined(out, .thought) == "step one. step two.")
        #expect(joined(out, .message) == "Here is the answer, at last.")
    }
}
