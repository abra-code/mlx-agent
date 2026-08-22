// GeneratorNameTests.swift - naming the model that wrote a summary.
//
// The bug these exist for: every cached Hugging Face model is a `snapshots/main` directory, so the
// obvious implementation - take the last path component - names all of them "main". That is worse
// than the "session:mlx" it replaced, because it looks like an answer.

import Foundation
import Testing

@testable import AgentDigest

@Suite("DigestGeneratorName")
struct DigestGeneratorNameTests {

    // MARK: - MLX directories

    @Test("a Hugging Face cache entry is named by its repo, not by its snapshot")
    func hfCacheEntry() {
        let dir =
            "/Users/x/.cache/huggingface/hub/"
            + "models--lmstudio-community--gemma-4-E2B-it-MLX-5bit/snapshots/main"
        #expect(
            DigestGeneratorName.fromModelDirectory(dir)
                == "lmstudio-community/gemma-4-E2B-it-MLX-5bit")
    }

    /// The revision is usually a hash rather than "main", and it is just as wrong an answer.
    @Test("a hashed revision is no more the model's name than main is")
    func hfCacheEntryHashedRevision() {
        let dir =
            "/Users/x/.cache/huggingface/hub/models--mlx-community--Qwen3-4B/snapshots/"
            + "9a1c2f3b4d5e6f708192a3b4c5d6e7f809a1b2c3"
        #expect(DigestGeneratorName.fromModelDirectory(dir) == "mlx-community/Qwen3-4B")
    }

    @Test("a plain directory is named by itself")
    func plainDirectory() {
        #expect(DigestGeneratorName.fromModelDirectory("/Users/x/models/Qwen3-4B") == "Qwen3-4B")
    }

    @Test("a trailing slash does not become the name")
    func trailingSlash() {
        #expect(DigestGeneratorName.fromModelDirectory("/Users/x/models/Qwen3-4B/") == "Qwen3-4B")
    }

    /// A snapshot layout that is NOT under the HF cache naming - the revision still must not win.
    @Test("a snapshots directory outside the cache still names the model above it")
    func bareSnapshotLayout() {
        #expect(DigestGeneratorName.fromModelDirectory("/m/Gemma-3n/snapshots/main") == "Gemma-3n")
    }

    @Test("nothing to read is nil, not an empty name")
    func nothingToRead() {
        #expect(DigestGeneratorName.fromModelDirectory("") == nil)
        #expect(DigestGeneratorName.fromModelDirectory("/") == nil)
        #expect(DigestGeneratorName.fromModelDirectory("/snapshots/main") == nil)
    }

    // MARK: - GGUF files

    @Test("a gguf file is named the way the app's own model bar names it")
    func ggufFile() {
        let path = "/Users/x/.lmstudio/models/lmstudio-community/g/gemma-4-31B-it-Q4_K_M.gguf"
        #expect(DigestGeneratorName.fromModelFile(path) == "gemma-4-31B-it-Q4_K_M")
    }

    @Test("a file that is not a gguf keeps its whole name")
    func notAGGUF() {
        #expect(DigestGeneratorName.fromModelFile("/m/weights.bin") == "weights.bin")
    }

    /// llama-server reports whatever it was given, and case is not guaranteed.
    @Test("the extension is matched without regard to case")
    func extensionCase() {
        #expect(DigestGeneratorName.fromModelFile("/m/Tiny-Q4.GGUF") == "Tiny-Q4")
    }

    // MARK: - Either kind

    @Test("a path of unknown kind is decided by its extension")
    func unknownKind() {
        #expect(DigestGeneratorName.fromModelPath("/m/Tiny-Q4_K_M.gguf") == "Tiny-Q4_K_M")
        #expect(
            DigestGeneratorName.fromModelPath("/hub/models--org--Name/snapshots/main")
                == "org/Name")
    }

    /// A NAME-SHAPE rule is what this must not use, and the first version did: "7-to-40 hex
    /// characters is a revision" also eats a model legitimately called `deadbeef`, and the walk
    /// upward then answered "models" - a parent directory presented as a model name, which is the
    /// exact failure this file exists to prevent.
    @Test("a model whose name happens to look like a git hash keeps it")
    func hexNamedModel() {
        #expect(DigestGeneratorName.fromModelDirectory("/Users/x/models/deadbeef") == "deadbeef")
        #expect(DigestGeneratorName.fromModelDirectory("/m/0123456789abcdef") == "0123456789abcdef")
    }

    /// The layout decides, so a revision is only a revision when "snapshots" says so.
    @Test("a directory literally called snapshots names nothing")
    func bareSnapshots() {
        #expect(DigestGeneratorName.fromModelDirectory("/m/snapshots") == nil)
        #expect(DigestGeneratorName.fromModelDirectory("/snapshots/main") == nil)
    }

    // MARK: - What a server is allowed to put in a marker

    /// `model_path` comes from `/props`, which is a server talking. It is logged, returned as
    /// `summarizer`, and stored in the digest, so an unbounded or control-laden value would land
    /// in all three. Driven through `fromModelPath` first, which is the order production uses.
    private func summarizerName(forServed path: String) -> String {
        DigestGeneratorName.session(
            engine: "openai", model: DigestGeneratorName.fromModelPath(path))
    }

    @Test("a control character never reaches the marker")
    func stripsControls() {
        let name = summarizerName(forServed: "/m/ev\u{1b}[2Jil.gguf")
        #expect(!name.contains("\u{1b}"))
        #expect(!name.contains("\n"))
        #expect(name.contains("ev"))
    }

    @Test("and a very long one is cut rather than printed whole")
    func capsLength() {
        let name = summarizerName(
            forServed: "/m/" + String(repeating: "n", count: 5000) + ".gguf")
        #expect(name.count <= 96)
        #expect(name.hasSuffix("..."))
    }

    @Test("a name that is nothing but whitespace falls back rather than emptying the marker")
    func whitespaceOnly() {
        #expect(summarizerName(forServed: "/m/   .gguf") == "session:openai")
    }

    // MARK: - What gets recorded

    @Test("a named model is what the digest reports")
    func namedModel() {
        #expect(DigestGeneratorName.session(engine: "mlx", model: "Qwen3-4B") == "Qwen3-4B")
    }

    /// The fallback has to survive, because an engine can be running something unnameable - a bare
    /// directory, or a server that does not report what it loaded. "session:openai" is at least
    /// true, where "" would be a digest attributed to nobody.
    @Test("an unnameable model falls back to the engine convention")
    func unnameableModel() {
        #expect(DigestGeneratorName.session(engine: "openai", model: nil) == "session:openai")
        #expect(DigestGeneratorName.session(engine: "openai", model: "") == "session:openai")
        #expect(DigestGeneratorName.session(engine: "mlx", model: "   ") == "session:mlx")
    }
}
