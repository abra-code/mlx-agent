// Chunker.swift - token-accurate text chunking for the map mode.
//
// Foundation-only (NSRegularExpression) so it can be unit-tested with swift-testing without
// pulling in MLX/Metal. The token count is injected as a closure, so tests use a stub counter
// while the translate server passes the real tokenizer's `encode(text:).count`.

import Foundation

/// One unit of translation work: `text` is the source to translate, `sep` is the verbatim
/// inter-chunk separator that follows it. Reassembly is `concat(text + sep)` over the chunks,
/// so paragraph structure at chunk boundaries survives exactly.
public struct Chunk: Sendable, Equatable {
    public let text: String
    public let sep: String

    public init(text: String, sep: String) {
        self.text = text
        self.sep = sep
    }
}

/// Splits source text into chunks each under a token budget, packing on paragraph then
/// sentence boundaries and hard-splitting only pathological runs. The token count comes from
/// the real tokenizer (passed as `count`), so the budget is accurate for the target model.
///
/// Invariant: `concat(chunk.text + chunk.sep)` over all chunks reproduces the input exactly,
/// so the source round-trips losslessly (a chunk's `sep` is the boundary separator preserved
/// verbatim; separators internal to a packed chunk are folded into its text).
public enum Chunker {

    /// A paragraph (or sentence) plus the verbatim separator that trails it.
    private struct Atom {
        var text: String
        var sep: String
    }

    // A blank-line run: two or more newlines (each optionally led by spaces/tabs) plus any
    // trailing spaces/tabs. Separates paragraphs without swallowing the next paragraph's text.
    private static let paragraphPattern = "(?:[ \\t]*\\r?\\n){2,}[ \\t]*"
    // Whitespace following a sentence terminator (ASCII or CJK, incl. ellipsis).
    private static let sentencePattern = "(?<=[.!?。！？…])\\s+"

    public static func make(_ text: String, budget: Int, count: (String) -> Int) -> [Chunk] {
        guard !text.isEmpty else { return [] }
        let budget = max(1, budget)
        // 1. Paragraphs, then expand any that overflow the budget into sentences / hard splits.
        let paragraphs = normalize(split(text, pattern: paragraphPattern))
        var atoms: [Atom] = []
        for p in paragraphs {
            if count(p.text) <= budget {
                atoms.append(p)
            } else {
                atoms.append(contentsOf: expandOversize(p, budget: budget, count: count))
            }
        }
        // 2. Greedy-pack atoms into chunks under the budget.
        return pack(atoms, budget: budget, count: count)
    }

    /// Split `text` on `pattern`, keeping each match as the separator that trails the unit
    /// before it. `concat(unit + sep) + tail == text`.
    private static func split(_ text: String, pattern: String) -> [Atom] {
        guard let re = try? NSRegularExpression(pattern: pattern) else {
            return [Atom(text: text, sep: "")]
        }
        let ns = text as NSString
        var out: [Atom] = []
        var last = 0
        for m in re.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let unit = ns.substring(with: NSRange(location: last, length: m.range.location - last))
            let sep = ns.substring(with: m.range)
            out.append(Atom(text: unit, sep: sep))
            last = m.range.location + m.range.length
        }
        let tail = ns.substring(from: last)
        if !tail.isEmpty || out.isEmpty {
            out.append(Atom(text: tail, sep: ""))
        }
        return out
    }

    /// Fold empty-text atoms (e.g. from a leading/among blank-line runs) into their neighbors
    /// so no chunk is ever empty, while keeping `concat(text + sep)` exact.
    private static func normalize(_ atoms: [Atom]) -> [Atom] {
        var out: [Atom] = []
        var prefix = ""  // separators seen before the first real text become its prefix
        for atom in atoms {
            if atom.text.isEmpty {
                if out.isEmpty {
                    prefix += atom.sep
                } else {
                    out[out.count - 1].sep += atom.sep
                }
            } else {
                out.append(Atom(text: prefix + atom.text, sep: atom.sep))
                prefix = ""
            }
        }
        if out.isEmpty && !prefix.isEmpty { out.append(Atom(text: prefix, sep: "")) }
        return out
    }

    /// Expand a paragraph that exceeds the budget: split into sentences, then hard-split any
    /// single sentence that is still too large. The paragraph's own trailing separator is
    /// carried on the last piece.
    private static func expandOversize(_ atom: Atom, budget: Int, count: (String) -> Int) -> [Atom]
    {
        var subs = normalize(split(atom.text, pattern: sentencePattern))
        guard !subs.isEmpty else { return [atom] }
        subs[subs.count - 1].sep += atom.sep
        var out: [Atom] = []
        for s in subs {
            if count(s.text) <= budget {
                out.append(s)
            } else {
                out.append(contentsOf: hardSplit(s, budget: budget, count: count))
            }
        }
        return out
    }

    /// Last-resort split of a single over-budget sentence. Break the text into indivisible
    /// units - whole words (with trailing whitespace) when there is any, else single characters
    /// - then greedy-pack the units up to the budget. A single word that alone exceeds the
    /// budget is itself broken into characters, so no unit is unbreakable. `concat` stays exact;
    /// the atom's separator rides on the final piece.
    ///
    /// Token counts are summed PER UNIT (additive) rather than re-encoding the whole growing
    /// buffer on every step: that keeps this O(units) instead of O(units x budget), which
    /// matters for long whitespace-free scripts (CJK, Thai) - exactly what this path handles.
    /// Additivity ignores cross-unit merges, so the estimate is a slight over-count, i.e.
    /// conservative (pieces stay at or under budget in real tokens).
    private static func hardSplit(_ atom: Atom, budget: Int, count: (String) -> Int) -> [Atom] {
        var units: [String] = []
        if atom.text.rangeOfCharacter(from: .whitespacesAndNewlines) != nil {
            for part in split(atom.text, pattern: "\\s+") {
                let word = part.text + part.sep
                if word.isEmpty { continue }
                if word.count > 1 && count(word) > budget {
                    units.append(contentsOf: word.map(String.init))  // oversize word -> chars
                } else {
                    units.append(word)
                }
            }
        } else {
            units = atom.text.map(String.init)
        }

        var pieces: [String] = []
        var current = ""
        var currentTokens = 0
        for unit in units {
            let t = count(unit)
            if !current.isEmpty && currentTokens + t > budget {
                pieces.append(current)
                current = unit
                currentTokens = t
            } else {
                current += unit
                currentTokens += t
            }
        }
        if !current.isEmpty { pieces.append(current) }
        guard !pieces.isEmpty else { return [atom] }
        var out = pieces.map { Atom(text: $0, sep: "") }
        out[out.count - 1].sep = atom.sep
        return out
    }

    /// Greedy-pack consecutive atoms into chunks. Within a chunk the atoms' internal
    /// separators are folded into the chunk text; the chunk's own `sep` is the trailing
    /// separator of its last atom (the boundary separator preserved verbatim for stitching).
    private static func pack(_ atoms: [Atom], budget: Int, count: (String) -> Int) -> [Chunk] {
        var chunks: [Chunk] = []
        var runText = ""
        var runSep = ""
        var runTokens = 0
        var haveRun = false
        for atom in atoms {
            let t = count(atom.text)
            if haveRun && runTokens + t > budget {
                chunks.append(Chunk(text: runText, sep: runSep))
                haveRun = false
            }
            if haveRun {
                runText += runSep + atom.text
                runSep = atom.sep
                runTokens += t
            } else {
                runText = atom.text
                runSep = atom.sep
                runTokens = t
                haveRun = true
            }
        }
        if haveRun { chunks.append(Chunk(text: runText, sep: runSep)) }
        return chunks
    }
}
