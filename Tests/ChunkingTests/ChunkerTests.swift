import Foundation
import Testing

import Chunking

// A deterministic stand-in for the real tokenizer: ~1 token per 4 characters, min 1. Good
// enough to exercise packing/splitting boundaries without loading a model.
private func stubCount(_ s: String) -> Int { max(1, s.count / 4) }
// A whitespace-token counter used where word boundaries matter for the assertion.
private func wordCount(_ s: String) -> Int {
    max(1, s.split(whereSeparator: { $0 == " " || $0 == "\n" }).count)
}

private func reassemble(_ chunks: [Chunk]) -> String {
    chunks.map { $0.text + $0.sep }.joined()
}

@Test("round-trips losslessly across many inputs and budgets")
func roundTrip() {
    let inputs = [
        "",
        "Hello.",
        "One two three four five six seven eight nine ten.",
        "Para one.\n\nPara two.\n\nPara three.",
        "Leading blanks.\n\n\n\nAfter many blanks.",
        "\n\nStarts with a blank line then text.",
        "Trailing blank lines follow.\n\n\n",
        "Windows\r\nline\r\nbreaks\r\n\r\nsecond para.",
        "Tabs\tand   spaces    between.\n\nNext.",
        "No terminators here just words that keep going and going without any punctuation at all",
        "CJK: 私は猫です。名前はまだありません。どこで生れたか。",
        "Mixed 中文 and English. 第二句。 Third sentence here.",
        String(repeating: "A long sentence with many words repeated over and over. ", count: 50),
        String(repeating: "一文。", count: 200),  // no whitespace, forces char hard-split
        "a.b.c.d.e.f.g.h.i.j.k.l.m.n.o.p",
    ]
    for input in inputs {
        for budget in [1, 2, 5, 12, 40, 200] {
            let chunks = Chunker.make(input, budget: budget, count: stubCount)
            #expect(
                reassemble(chunks) == input,
                "reassembly mismatch: budget=\(budget) input=\(input.debugDescription)")
        }
    }
}

@Test("never emits an empty chunk")
func noEmptyChunks() {
    let inputs = [
        "Para one.\n\nPara two.", "\n\n\n\n", "\n\nx", "x\n\n", "   \n\n   text   \n\n   ",
        String(repeating: "word ", count: 100),
    ]
    for input in inputs {
        for budget in [1, 3, 10, 50] {
            let chunks = Chunker.make(input, budget: budget, count: stubCount)
            for c in chunks {
                #expect(!c.text.isEmpty, "empty chunk text for input \(input.debugDescription)")
            }
        }
    }
}

@Test("empty input yields no chunks")
func emptyInput() {
    #expect(Chunker.make("", budget: 100, count: stubCount).isEmpty)
}

@Test("small paragraphs pack together; the boundary separator is preserved verbatim")
func packsSmallParagraphs() {
    // Three one-word paragraphs, budget large enough to hold all three in one chunk.
    let input = "Alpha\n\nBeta\n\nGamma"
    let chunks = Chunker.make(input, budget: 100, count: wordCount)
    #expect(chunks.count == 1)
    #expect(reassemble(chunks) == input)
}

@Test("packing respects the budget for normal prose (each chunk within budget)")
func respectsBudget() {
    // Ten single-word sentences; wordCount == number of words. Budget 3 -> chunks of <= 3 words
    // once packed (a single atom never exceeds the budget here).
    let input = (1...10).map { "w\($0)." }.joined(separator: " ")
    let budget = 3
    let chunks = Chunker.make(input, budget: budget, count: wordCount)
    #expect(chunks.count >= 3)
    for c in chunks {
        #expect(wordCount(c.text) <= budget, "chunk over budget: \(c.text.debugDescription)")
    }
    #expect(reassemble(chunks) == input)
}

@Test("an oversize single paragraph is split into multiple chunks")
func splitsOversizeParagraph() {
    let input = String(repeating: "Sentence number here. ", count: 30)
    let chunks = Chunker.make(input, budget: 5, count: wordCount)
    #expect(chunks.count > 1)
    #expect(reassemble(chunks) == input)
}

@Test("a whitespace-free run over budget is hard-split by characters and still round-trips")
func hardSplitsCJK() {
    let input = String(repeating: "字", count: 100)  // no whitespace, no terminators
    let chunks = Chunker.make(input, budget: 5, count: stubCount)
    #expect(chunks.count > 1)
    #expect(reassemble(chunks) == input)
    for c in chunks { #expect(!c.text.isEmpty) }
}

@Test("a single word longer than the budget is broken down, not left whole and unbounded")
func oversizeWordIsBrokenDown() {
    // One enormous "word" (e.g. a URL or hash) embedded in prose. The word must be broken up
    // rather than surfacing as a single chunk that blows past the budget by its full length.
    // (The budget is a soft knee: packing bounds the SUM of per-atom counts, so with a
    // non-additive counter a chunk may sit slightly over - we assert a loose bound, not equality.)
    let big = String(repeating: "x", count: 200)
    let input = "Start. \(big) end."
    let budget = 30
    let chunks = Chunker.make(input, budget: budget, count: stubCount)
    #expect(reassemble(chunks) == input)
    #expect(chunks.count > 1, "oversize word was not split")
    for c in chunks {
        #expect(c.text.count < big.count, "a chunk still holds the whole oversize word")
        #expect(stubCount(c.text) <= budget * 2, "chunk grossly over budget: \(stubCount(c.text))")
    }
}
