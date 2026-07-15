// ThinkSplitter.swift - raw model text -> thought / message segments.
//
// Pure string logic with no MLX dependency, in its own target for the same reason
// Chunking is: it can then be unit-tested WITHOUT compiling MLX or the Metal toolchain
// (see the UnitTests scheme). It used to live in ACPServer.swift, which links MLX, and
// so went untested - and shipped the carry-over bug the tests here now pin down.
//
// This is the ONE place that decides thought vs message for text the model emitted
// inline. Text a SERVER already classified as reasoning (llama-server's
// `reasoning_content`) must NOT come through here - see Agent's `.reasoning` handling.

import Foundation

/// Splits a raw model stream into thought (inside <think>...</think>) and message
/// (everything else) segments, tolerating markers that straddle chunk boundaries.
///
/// Lifetime: one splitter per model pass is NOT required - `inThink` is deliberately
/// carried across passes so a <think> block may span a tool call - but `flush()` MUST be
/// called at the end of every pass. See the note on `flush()`.
public final class ThinkSplitter {
    public enum Kind: Sendable { case thought, message }

    private var buffer = ""
    private var inThink = false
    private let open = "<think>"
    private let close = "</think>"

    /// How much of the tail to hold back: the longest marker minus one, i.e. the longest
    /// possible incomplete marker prefix. Anything shorter than a full marker could still
    /// be completed by the next chunk, so it cannot be emitted yet.
    private var keep: Int { max(open.count, close.count) - 1 }

    public init() {}

    /// Feed one chunk of raw model text. Returns the segments that can be classified with
    /// certainty; a trailing `keep` characters may be withheld pending the next chunk.
    public func feed(_ s: String) -> [(Kind, String)] {
        buffer += s
        var out: [(Kind, String)] = []
        while true {
            let marker = inThink ? close : open
            if let r = buffer.range(of: marker) {
                let before = String(buffer[buffer.startIndex..<r.lowerBound])
                if !before.isEmpty { out.append((inThink ? .thought : .message, before)) }
                buffer.removeSubrange(buffer.startIndex..<r.upperBound)
                inThink.toggle()
            } else {
                // Hold back a short tail that could be the prefix of a marker.
                if buffer.count > keep {
                    let end = buffer.index(buffer.endIndex, offsetBy: -keep)
                    let emit = String(buffer[buffer.startIndex..<end])
                    if !emit.isEmpty { out.append((inThink ? .thought : .message, emit)) }
                    buffer = String(buffer[end...])
                }
                break
            }
        }
        return out
    }

    /// Emit whatever is still held back. MUST be called at the end of EVERY model pass,
    /// not just at the end of a turn: once a pass's stream ends no further chunk can
    /// complete a straddling marker, so the withheld tail is final text. Skipping this
    /// (as the tool-call path used to) strands up to `keep` characters in the buffer,
    /// where they silently become the PREFIX of the next pass's first segment - e.g. a
    /// model emitting "<|channel|>" right before a tool call had "<|ch" emitted and
    /// "annel|>" reappear glued to the front of its post-tool answer.
    ///
    /// `inThink` is intentionally NOT reset: a <think> block may legitimately span a tool
    /// call, and the text after it is still thought.
    public func flush() -> [(Kind, String)] {
        guard !buffer.isEmpty else { return [] }
        let seg: (Kind, String) = (inThink ? .thought : .message, buffer)
        buffer = ""
        return [seg]
    }
}
