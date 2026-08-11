// CumulativeDelta.swift - cumulative snapshots in, deltas out.
//
// Apple's Foundation Models streams CUMULATIVE snapshots: every element of its ResponseStream
// carries the whole answer so far, not the piece just produced. Everything downstream of a
// backend here expects deltas - `BackendEvent.text` is a fragment, ThinkSplitter consumes a
// running stream, and ACP's session/update sends increments the client appends. Something has
// to sit between those two shapes, and this is it.
//
// Pure and in the MLX-free target on purpose: it is the sort of thing that looks trivial and
// then loses a character at a chunk boundary, so it is worth testing without compiling MLX.
//
// THE INVARIANT: for snapshots that genuinely extend one another - the only kind a well-behaved
// stream produces - concatenating every returned delta reproduces the last snapshot exactly.
// Everything else here follows from defending that. The one case where it cannot hold is a
// snapshot that REWRITES already-emitted text, because nothing can un-send what the consumer
// already has; see `advance(to:onAnomaly:)` for what happens instead.
//
// Comparison is by UNICODE SCALAR, not by Character, and that is deliberate. Swift compares
// Characters by canonical equivalence, so "e" is NOT a prefix of "e" + U+0301 (they are one
// grapheme, "e-acute", which does not equal "e"). A model emitting a base letter in one
// snapshot and its combining mark in the next would therefore look like a rewrite rather than
// growth, and the whole answer would be re-emitted. Scalars have no such trap. The cost is that
// a delta can begin mid-grapheme; that is fine for every consumer here, because they
// concatenate, and the cluster is complete again as soon as the next delta lands.

import Foundation

/// Turns a stream of cumulative snapshots into the deltas between them.
///
/// One instance per stream. Not thread-safe: feed it from one place, in order.
public struct CumulativeDelta {

    /// Everything handed out so far. Kept whole rather than as a count so an anomalous snapshot
    /// can be diffed against it properly.
    private var emitted = ""

    public init() {}

    /// The part of `full` not yet returned.
    ///
    /// Normal case: `full` extends what was emitted, and the new suffix comes back.
    ///
    /// Anomalous case: `full` is NOT an extension - the model rewrote or truncated something
    /// already streamed. That should not happen for String content, but "should not happen" is
    /// not a plan when the alternative is silently emitting garbage. The response is to fall
    /// back to the longest common prefix and return everything after it, so the consumer at
    /// least converges on the correct final text, and to report it through `onAnomaly` so it
    /// shows up in a log rather than as a mystery in the transcript. A pure truncation yields
    /// an empty delta: nothing can un-send what the client already has.
    public mutating func advance(to full: String, onAnomaly: (String) -> Void = { _ in }) -> String
    {
        if full.unicodeScalars.starts(with: emitted.unicodeScalars) {
            let delta = String(full.unicodeScalars.dropFirst(emitted.unicodeScalars.count))
            emitted = full
            return delta
        }

        let shared = commonScalarPrefixCount(full, emitted)
        onAnomaly(
            "stream snapshot is not an extension of what was emitted "
                + "(emitted \(emitted.unicodeScalars.count) scalars, snapshot "
                + "\(full.unicodeScalars.count), common \(shared)); resyncing")
        let delta = String(full.unicodeScalars.dropFirst(shared))
        emitted = full
        return delta
    }

    /// Everything returned so far, i.e. the last snapshot seen.
    public var emittedText: String { emitted }

    private func commonScalarPrefixCount(_ a: String, _ b: String) -> Int {
        var count = 0
        var left = a.unicodeScalars.makeIterator()
        var right = b.unicodeScalars.makeIterator()
        while let l = left.next(), let r = right.next(), l == r { count += 1 }
        return count
    }
}
