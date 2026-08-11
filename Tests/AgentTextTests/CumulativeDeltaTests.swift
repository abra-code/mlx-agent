// CumulativeDeltaTests.swift - the invariant, and the boundaries that threaten it.
//
// The property that matters is that concatenating every delta reproduces the last snapshot.
// Most of these assert it directly rather than checking individual return values, because that
// is the thing a consumer actually depends on.

import Testing

@testable import AgentText

@Suite("CumulativeDelta")
struct CumulativeDeltaTests {

    @Test("growing snapshots yield only the new suffix")
    func growth() {
        var delta = CumulativeDelta()
        #expect(delta.advance(to: "Hel") == "Hel")
        #expect(delta.advance(to: "Hello") == "lo")
        #expect(delta.advance(to: "Hello, world") == ", world")
    }

    @Test("an unchanged snapshot yields nothing")
    func idempotent() {
        var delta = CumulativeDelta()
        #expect(delta.advance(to: "Hello") == "Hello")
        #expect(delta.advance(to: "Hello") == "")
        #expect(delta.advance(to: "Hello") == "")
    }

    @Test("the first snapshot is returned whole")
    func firstSnapshot() {
        var delta = CumulativeDelta()
        #expect(delta.advance(to: "complete answer") == "complete answer")
    }

    @Test("an empty stream stays empty")
    func empty() {
        var delta = CumulativeDelta()
        #expect(delta.advance(to: "") == "")
        #expect(delta.emittedText == "")
    }

    @Test("concatenated deltas reconstruct the final snapshot")
    func invariantHolds() {
        let final = "The quick brown fox jumps over the lazy dog."
        var delta = CumulativeDelta()
        var rebuilt = ""
        for end in 1...final.count {
            rebuilt += delta.advance(to: String(final.prefix(end)))
        }
        #expect(rebuilt == final)
    }

    // The reason comparison is by scalar and not by Character: a combining mark arriving in a
    // later snapshot than its base letter changes the grapheme, so Character-wise the new
    // snapshot is not an extension of the old one and the whole string would be re-emitted.
    @Test("a combining mark arriving late is growth, not a rewrite")
    func combiningMarkAcrossSnapshots() {
        var delta = CumulativeDelta()
        var anomalies: [String] = []
        #expect(delta.advance(to: "cafe", onAnomaly: { anomalies.append($0) }) == "cafe")
        let composed = "cafe\u{0301}"  // cafe + combining acute = "café"
        let second = delta.advance(to: composed, onAnomaly: { anomalies.append($0) })
        #expect(second == "\u{0301}")
        #expect(anomalies.isEmpty)
        #expect("cafe" + second == composed)
    }

    @Test("emoji built up across snapshots reconstruct exactly")
    func emojiInvariant() {
        // A ZWJ sequence: the family emoji is several scalars joined, and a snapshot can land
        // in the middle of it.
        let final = "done \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467} ok"
        var delta = CumulativeDelta()
        var rebuilt = ""
        for end in 1...final.unicodeScalars.count {
            let prefix = String(String.UnicodeScalarView(final.unicodeScalars.prefix(end)))
            rebuilt += delta.advance(to: prefix)
        }
        #expect(rebuilt == final)
    }

    @Test("a rewritten snapshot resyncs from the common prefix and reports it")
    func rewriteAnomaly() {
        var delta = CumulativeDelta()
        var anomalies: [String] = []
        #expect(delta.advance(to: "The answer is 41", onAnomaly: { anomalies.append($0) })
                == "The answer is 41")
        let corrected = delta.advance(to: "The answer is 42", onAnomaly: { anomalies.append($0) })
        #expect(corrected == "2")
        #expect(anomalies.count == 1)
        #expect(delta.emittedText == "The answer is 42")
    }

    @Test("a truncated snapshot yields nothing and is reported")
    func truncationAnomaly() {
        var delta = CumulativeDelta()
        var anomalies: [String] = []
        _ = delta.advance(to: "Hello, world", onAnomaly: { anomalies.append($0) })
        // Nothing can un-send what the client already has, so the delta is empty.
        #expect(delta.advance(to: "Hello", onAnomaly: { anomalies.append($0) }) == "")
        #expect(anomalies.count == 1)
    }

    @Test("a completely different snapshot is emitted whole")
    func totalDivergence() {
        var delta = CumulativeDelta()
        var anomalies: [String] = []
        _ = delta.advance(to: "aaa", onAnomaly: { anomalies.append($0) })
        #expect(delta.advance(to: "bbb", onAnomaly: { anomalies.append($0) }) == "bbb")
        #expect(anomalies.count == 1)
    }

    @Test("chunking does not change the reconstructed text")
    func chunkingIndependence() {
        let final = "Streaming text with punctuation, numbers 123, and unicode: zazolc gesla jazn."
        for step in [1, 2, 3, 7, 13] {
            var delta = CumulativeDelta()
            var rebuilt = ""
            var end = step
            while end < final.count {
                rebuilt += delta.advance(to: String(final.prefix(end)))
                end += step
            }
            rebuilt += delta.advance(to: final)
            #expect(rebuilt == final, "step \(step)")
        }
    }
}
