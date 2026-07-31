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
        // Shorter than `keep` (6 < 9), so feed alone must withhold all of it.
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
        // Pins the mechanism itself: WITHOUT the flush, exactly `keep` characters go
        // missing. `keep` is now 9 (the longest marker is Gemma 4's 10-character
        // "<|channel>" / "<channel|>", not "</think>"), so the split moved by two.
        let s = ThinkSplitter()
        let out = s.feed("Looking that up.<|channel|>")
        #expect(joined(out, .message) == "Looking that up.<|")
        #expect(joined(s.flush(), .message) == "channel|>")
    }

    // MARK: - Gemma 4 channel markers

    // The reported bug, verbatim from a Cadabra transcript
    // (History/20260731T111708Z-33141). Gemma 4's chat template prefills a closed, empty
    // thought block when thinking is off - but SKIPS that prefill when the previous message
    // was a tool response, so the model opens the channel itself and the markers land in the
    // visible stream. The answer bubble read "<|channel>thought <channel|>The current time
    // in Dubai is...".
    @Test("an empty Gemma 4 thought channel disappears instead of leaking")
    func gemma4EmptyThoughtChannel() {
        let s = ThinkSplitter()
        let out =
            s.feed("<|channel>thought\n<channel|>The current time in Dubai is 3:17 PM.")
            + s.flush()
        #expect(joined(out, .message) == "The current time in Dubai is 3:17 PM.")
        #expect(joined(out, .thought) == "")
    }

    @Test("a Gemma 4 thought channel routes its body to thought")
    func gemma4ThoughtChannel() {
        let s = ThinkSplitter()
        let out =
            s.feed("<|channel>thought\nI should read the clock.<channel|>It is 3:17 PM.")
            + s.flush()
        #expect(joined(out, .thought) == "I should read the clock.")
        #expect(joined(out, .message) == "It is 3:17 PM.")
    }

    @Test("Gemma 4 markers arriving one character at a time are still recognised")
    func gemma4MarkersStraddleChunks() {
        let s = ThinkSplitter()
        var out: [(ThinkSplitter.Kind, String)] = []
        for ch in "<|channel>thought\nreasoning<channel|>answer" {
            out += s.feed(String(ch))
        }
        out += s.flush()
        #expect(joined(out, .thought) == "reasoning")
        #expect(joined(out, .message) == "answer")
    }

    // Gemma 4 names the channel carrying the ANSWER `content`. Routing every channel to the
    // thought bubble would swallow the reply whole.
    @Test("the content channel routes to message, not thought")
    func gemma4ContentChannel() {
        let s = ThinkSplitter()
        let out = s.feed("<|channel>content\nThe answer.<channel|>") + s.flush()
        #expect(joined(out, .message) == "The answer.")
        #expect(joined(out, .thought) == "")
    }

    @Test("an unknown channel label is hidden in thought rather than shown")
    func gemma4UnknownChannel() {
        let s = ThinkSplitter()
        let out = s.feed("<|channel>commentary\nscratch<channel|>answer") + s.flush()
        #expect(joined(out, .thought) == "scratch")
        #expect(joined(out, .message) == "answer")
    }

    // Both delimiters are special tokens (soc_token / eoc_token), so a closer with no opener
    // cannot be prose - it means the PROMPT opened the channel. The text before it has
    // already streamed and cannot be reclassified, but the marker must not reach the answer.
    @Test("a stray channel closer is swallowed, not printed")
    func gemma4StrayCloser() {
        let s = ThinkSplitter()
        let out = s.feed("Reasoning done.<channel|>The answer.") + s.flush()
        #expect(!joined(out, .message).contains("<channel|>"))
        #expect(joined(out, .message) == "Reasoning done.The answer.")
    }

    @Test("a Gemma 4 channel can span a flushed pass boundary")
    func gemma4ChannelSpansPasses() {
        let s = ThinkSplitter()
        var out = s.feed("<|channel>thought\nlooking")
        out += s.flush()                        // end of the pass that made a tool call
        out += s.feed(" it up<channel|>Done.")  // the pass after the tool result
        out += s.flush()
        #expect(joined(out, .thought) == "looking it up")
        #expect(joined(out, .message) == "Done.")
    }

    // An opener whose label never terminates must not buffer the whole pass. After
    // `maxLabelLength` the splitter gives up on the label and treats what arrived as body -
    // visible in the thought bubble, and crucially never in the answer.
    @Test("an unterminated channel label degrades to thought instead of buffering forever")
    func gemma4UnterminatedLabel() {
        let s = ThinkSplitter()
        let long = String(repeating: "x", count: 64)
        let out = s.feed("<|channel>" + long) + s.flush()
        #expect(joined(out, .message) == "")
        #expect(joined(out, .thought) == long)
    }

    // A malformed opener whose "label" runs long and is then CLOSED must not be swallowed.
    // The first cut decided the empty-channel case on "is there a closer before a newline",
    // ignoring how far away that closer was, so a single feed() dropped everything up to it
    // while the same bytes arriving one at a time showed all of it. Silent unbounded discard,
    // and the result depended on where the transport split the stream.
    @Test("a long label followed by a closer is shown, not swallowed")
    func gemma4LongLabelIsNotSwallowed() {
        let junk = String(repeating: "x", count: 40)
        let s = ThinkSplitter()
        let out = s.feed("<|channel>" + junk + "<channel|>done") + s.flush()
        #expect(joined(out, .thought) == junk)
        #expect(joined(out, .message) == "done")
    }

    // Absolute expectations either side of the window, so a regression that moves the
    // boundary consistently cannot hide behind `chunkingIndependence` - that test proves the
    // splits AGREE, not that they agree on the right answer.
    @Test("the label window boundary sits between 32 and 33 characters")
    func gemma4LabelWindowBoundary() {
        let inWindow = String(repeating: "x", count: 32)
        let past = String(repeating: "x", count: 33)

        let a = ThinkSplitter()
        let outA = a.feed("<|channel>" + inWindow + "<channel|>done") + a.flush()
        #expect(joined(outA, .thought) == "")        // consumed as a label
        #expect(joined(outA, .message) == "done")

        let b = ThinkSplitter()
        let outB = b.feed("<|channel>" + past + "<channel|>done") + b.flush()
        #expect(joined(outB, .thought) == past)      // too far out to be a label: shown
        #expect(joined(outB, .message) == "done")
    }

    // The settle-then-newline path: once the window is given up on, a newline arriving later
    // is ordinary body text, not a label terminator.
    @Test("a newline past the label window is body, not a terminator")
    func gemma4NewlinePastWindow() {
        let past = String(repeating: "x", count: 33)
        let s = ThinkSplitter()
        let out = s.feed("<|channel>" + past + "\nreason<channel|>ans") + s.flush()
        #expect(joined(out, .thought) == past + "\nreason")
        #expect(joined(out, .message) == "ans")

        // ...whereas one INSIDE the window still ends the label and is consumed with it.
        let inWindow = String(repeating: "x", count: 32)
        let t = ThinkSplitter()
        let out2 = t.feed("<|channel>" + inWindow + "\nreason<channel|>ans") + t.flush()
        #expect(joined(out2, .thought) == "reason")
        #expect(joined(out2, .message) == "ans")
    }

    // The property the above is one instance of: classification MUST NOT depend on chunk
    // boundaries. Feeds the same stream whole, one character at a time, and split at every
    // single position, and requires all of them to agree.
    @Test("classification is independent of where the stream is chunked")
    func chunkingIndependence() {
        let sources = [
            "<|channel>thought\n<channel|>The answer.",
            "<|channel>content\nThe answer.<channel|>",
            "<|channel>" + String(repeating: "x", count: 40) + "<channel|>done",
            "<|channel>" + String(repeating: "x", count: 32) + "<channel|>done",
            "<think>reasoning</think>The answer.",
            "<|channel|>analysis<|message|>harmony stays put",
        ]
        for source in sources {
            let whole = ThinkSplitter()
            let expected = whole.feed(source) + whole.flush()
            let wantThought = joined(expected, .thought)
            let wantMessage = joined(expected, .message)

            let single = ThinkSplitter()
            var out: [(ThinkSplitter.Kind, String)] = []
            for ch in source { out += single.feed(String(ch)) }
            out += single.flush()
            #expect(joined(out, .thought) == wantThought, "char-by-char: \(source)")
            #expect(joined(out, .message) == wantMessage, "char-by-char: \(source)")

            for cut in 1 ..< source.count {
                let i = source.index(source.startIndex, offsetBy: cut)
                let s = ThinkSplitter()
                var split = s.feed(String(source[source.startIndex ..< i]))
                split += s.feed(String(source[i...]))
                split += s.flush()
                #expect(joined(split, .thought) == wantThought, "cut at \(cut): \(source)")
                #expect(joined(split, .message) == wantMessage, "cut at \(cut): \(source)")
            }
        }
    }

    // Harmony (gpt-oss) spells its marker "<|channel|>" - pipes on BOTH sides - which is not
    // a substring of either Gemma 4 token. Pinning that keeps the new markers from
    // half-matching harmony and eating text; harmony itself stays llama-server's job, where
    // reasoning_format "auto" splits it into reasoning_content (see BackendEvent.reasoning).
    @Test("harmony's <|channel|> is not mistaken for a Gemma 4 marker")
    func harmonyMarkerIsNotGemma4() {
        let s = ThinkSplitter()
        let out = s.feed("<|channel|>analysis<|message|>hmm") + s.flush()
        #expect(joined(out, .message) == "<|channel|>analysis<|message|>hmm")
        #expect(joined(out, .thought) == "")
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
