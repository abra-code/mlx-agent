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
//
// Two marker vocabularies are understood, because the native MLX path gets raw model text
// and mlx-swift-lm has no reasoning parser of any kind (it parses tool calls per family,
// but nothing in MLXLMCommon so much as mentions reasoning - checked against 3.31.4):
//
//   <think>...</think>                      - the de-facto standard, most families
//   <|channel>LABEL\n...<channel|>          - Gemma 4, whose delimiters are real special
//                                             tokens (`soc_token` id 100 / `eoc_token`
//                                             id 101), not prose the model typed
//
// Gemma 4's shape is worth spelling out, because it is why the second vocabulary exists.
// Its chat template PREFILLS a closed, empty thought block after `<|turn>model` whenever
// thinking is off - so an ordinary answer never generates the markers at all. But the
// template deliberately skips that prefill when the previous message was a tool call or
// tool response, since the model turn is already open. The model then opens the channel
// ITSELF, and `<|channel>thought\n<channel|>` lands in the visible stream. That is why the
// leak only ever showed up on the answer AFTER a tool call.
//
// Every other stack that meets Gemma 4 has had to special-case this the same way: vLLM
// (`vllm/reasoning/gemma4_utils`), llama.cpp (the `peg-gemma4` parser), Ollama, LM Studio.
// See Private/commit-notes-gemma4-thought-channel.md.

import Foundation

/// Splits a raw model stream into thought and message segments, tolerating markers that
/// straddle chunk boundaries.
///
/// Lifetime: one splitter per model pass is NOT required - the state is deliberately
/// carried across passes so a reasoning block may span a tool call - but `flush()` MUST be
/// called at the end of every pass. See the note on `flush()`.
public final class ThinkSplitter {
    public enum Kind: Sendable { case thought, message }

    /// Where in the stream we are. Carried across passes; see `flush()`.
    private enum State {
        /// Ordinary answer text.
        case outside
        /// Inside `<think>...</think>`.
        case think
        /// `<|channel>` has been consumed and the role label is being read, up to the
        /// newline that ends it. The label is metadata and is never shown.
        case channelLabel
        /// Inside a channel body, routing to the kind the label resolved to.
        case channelBody(Kind)
    }

    private var buffer = ""
    private var state = State.outside

    private let thinkOpen = "<think>"
    private let thinkClose = "</think>"
    private let channelOpen = "<|channel>"
    private let channelClose = "<channel|>"

    /// How far past the opener a label may run. A label is one short word (`thought`,
    /// `content`); a terminator further out than this means the opener was malformed and the
    /// text is body, which is SHOWN (in the thought bubble) rather than consumed as a label.
    /// Bounding it matters in both directions: without the bound a malformed opener either
    /// buffers an entire pass or, worse, silently swallows everything up to a distant closer.
    private let maxLabelLength = 32

    /// How much of the tail to hold back: the longest marker minus one, i.e. the longest
    /// possible incomplete marker prefix. Anything shorter than a full marker could still
    /// be completed by the next chunk, so it cannot be emitted yet.
    private var keep: Int {
        [thinkOpen, thinkClose, channelOpen, channelClose].map(\.count).max()! - 1
    }

    public init() {}

    /// Feed one chunk of raw model text. Returns the segments that can be classified with
    /// certainty; a trailing `keep` characters may be withheld pending the next chunk.
    public func feed(_ s: String) -> [(Kind, String)] {
        buffer += s
        var out: [(Kind, String)] = []

        scan: while true {
            switch state {
            case .outside:
                // A stray `<channel|>` is scanned for alongside the openers. Both channel
                // delimiters are special tokens, so one arriving with no opener cannot be
                // legitimate prose - it means the PROMPT opened the channel (a template
                // that prefills only the opener). Swallow it: the text before it has
                // already streamed out as message and cannot be reclassified after the
                // fact, but the marker itself must not reach the answer.
                guard let hit = firstMarker(of: [thinkOpen, channelOpen, channelClose]) else {
                    holdBack(as: .message, into: &out)
                    break scan
                }
                take(upTo: hit.range, as: .message, into: &out)
                if hit.marker == thinkOpen {
                    state = .think
                } else if hit.marker == channelOpen {
                    state = .channelLabel
                }

            case .think:
                guard let r = buffer.range(of: thinkClose) else {
                    holdBack(as: .thought, into: &out)
                    break scan
                }
                take(upTo: r, as: .thought, into: &out)
                state = .outside

            case .channelLabel:
                // Nothing can be emitted until the label resolves, so there is no hold-back
                // here - the label is short and the next chunk settles it.
                guard let next = resolvedLabelState() else { break scan }
                state = next

            case .channelBody(let kind):
                guard let r = buffer.range(of: channelClose) else {
                    holdBack(as: kind, into: &out)
                    break scan
                }
                take(upTo: r, as: kind, into: &out)
                state = .outside
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
    /// The state is intentionally NOT reset: a reasoning block may legitimately span a
    /// tool call, and the text after it is still thought.
    public func flush() -> [(Kind, String)] {
        let kind: Kind
        switch state {
        case .outside:
            kind = .message
        case .think:
            kind = .thought
        case .channelLabel:
            // The label never terminated. We are still inside the channel, so keep routing
            // to thought rather than pretending the block ended - and stop waiting for a
            // label that a fresh pass would never supply.
            kind = .thought
            state = .channelBody(.thought)
        case .channelBody(let k):
            kind = k
        }

        guard !buffer.isEmpty else { return [] }
        let seg: (Kind, String) = (kind, buffer)
        buffer = ""
        return [seg]
    }

    // MARK: - Buffer mechanics

    /// Emit everything before `marker` as `kind`, then drop the marker itself.
    private func take(
        upTo marker: Range<String.Index>, as kind: Kind, into out: inout [(Kind, String)]
    ) {
        let before = String(buffer[buffer.startIndex ..< marker.lowerBound])
        if !before.isEmpty { out.append((kind, before)) }
        buffer.removeSubrange(buffer.startIndex ..< marker.upperBound)
    }

    /// No marker in sight: emit all but the tail that could still become one.
    private func holdBack(as kind: Kind, into out: inout [(Kind, String)]) {
        guard buffer.count > keep else { return }
        let end = buffer.index(buffer.endIndex, offsetBy: -keep)
        let emit = String(buffer[buffer.startIndex ..< end])
        if !emit.isEmpty { out.append((kind, emit)) }
        buffer = String(buffer[end...])
    }

    /// The earliest of `markers` present in the buffer, or nil if none is.
    private func firstMarker(of markers: [String]) -> (marker: String, range: Range<String.Index>)?
    {
        var best: (marker: String, range: Range<String.Index>)?
        for marker in markers {
            guard let r = buffer.range(of: marker) else { continue }
            if let found = best, found.range.lowerBound <= r.lowerBound { continue }
            best = (marker, r)
        }
        return best
    }

    /// Consume the channel's role label and decide where its body goes. Returns nil while
    /// the label is still incomplete.
    ///
    /// Every decision here is made on where a terminator STARTS, never on how much has been
    /// buffered, because the two disagree: buffered length counts the half-arrived closer,
    /// so a rule phrased in terms of it classifies the same bytes differently depending on
    /// where the transport happened to split them. A label ends at the first newline or
    /// closer BEGINNING within `maxLabelLength` of the opener, whenever that is observed.
    private func resolvedLabelState() -> State? {
        let newline = withinLabelWindow(buffer.range(of: "\n"))
        let close = withinLabelWindow(buffer.range(of: channelClose))

        // Closed before any newline: an empty channel with an unterminated label, which is
        // exactly what Gemma 4 emits after a tool response when thinking is off. There is
        // no body and the label is metadata, so the whole thing simply disappears.
        if let close, newline.map({ close.lowerBound < $0.lowerBound }) ?? true {
            buffer.removeSubrange(buffer.startIndex ..< close.upperBound)
            return .outside
        }

        if let newline {
            // `.whitespacesAndNewlines`, not `.whitespaces`: the latter excludes CR, so a
            // CRLF-terminated label would keep its "\r" and stop matching "content".
            let label = String(buffer[buffer.startIndex ..< newline.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            buffer.removeSubrange(buffer.startIndex ..< newline.upperBound)
            return .channelBody(kind(ofChannel: label))
        }

        // No terminator begins inside the window. Waiting is only pointless once enough text
        // has arrived that one starting at the very last in-window position would have
        // completed - before that, a closer may still be half here. Nothing is consumed:
        // `.channelBody` re-scans the same buffer and shows the text rather than dropping it.
        let settled = buffer.count >= maxLabelLength + channelClose.count
        return settled ? .channelBody(.thought) : nil
    }

    /// A terminator only ends the label if it begins within `maxLabelLength` of the opener.
    /// Past that this is not a label, and its text is content that must stay visible.
    private func withinLabelWindow(_ r: Range<String.Index>?) -> Range<String.Index>? {
        guard let r else { return nil }
        return buffer.distance(from: buffer.startIndex, to: r.lowerBound) <= maxLabelLength
            ? r : nil
    }

    /// Gemma 4 names the channel carrying the ANSWER `content`; `thought` carries reasoning.
    /// Anything else is unknown, and hiding an unknown channel in the thought bubble is the
    /// safe direction: nothing is lost (it stays readable), and no raw markers or scratchpad
    /// text can reach the answer.
    private func kind(ofChannel label: String) -> Kind {
        label == "content" ? .message : .thought
    }
}
