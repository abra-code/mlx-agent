// GeneratorName.swift - what to call the model that wrote a digest.
//
// `SessionDigest.generator` is reported to the client as `summarizer`, and a client renders it
// verbatim: "Resumed - 64 earlier messages summarized by X". So X is the reader's ONLY basis for
// deciding how much to trust the summary, and "session:mlx" - which is what this used to be -
// answers a question nobody asked. Whether the session's model or the on-device one wrote it is a
// distinction the reader cannot act on; whether it was a 2B or a 31B is the whole judgment.
//
// The naming lives here, in the MLX-free target, because it is string arithmetic over paths with
// several conventions to get wrong and no model needed to test any of them.

import Foundation

/// Turns a model path into something worth printing beside a summary.
public enum DigestGeneratorName {

    /// What a session-backed digest is called when the model behind it cannot be named.
    ///
    /// Kept as the fallback rather than removed: an engine can be running something this cannot
    /// resolve (a bare directory, a server that does not report its model), and "session:openai"
    /// is at least true. It is the answer of last resort, not the convention.
    public static func session(engine: String) -> String { "session:\(engine)" }

    /// The name for a digest written by the session's own model: the model itself when it can be
    /// named, and the engine convention when it cannot.
    public static func session(engine: String, model: String?) -> String {
        guard let model, let clean = sanitized(model) else { return session(engine: engine) }
        return clean
    }

    /// A readable name for an MLX model DIRECTORY.
    ///
    /// The last path component is wrong far more often than it is right, which is the trap this
    /// exists to avoid: a Hugging Face cache entry is
    /// `.../models--lmstudio-community--gemma-4-E2B-it-MLX-5bit/snapshots/main`, so the obvious
    /// implementation names every cached model "main". The `models--A--B` component is HF's own
    /// encoding of the repo id `A/B` (it replaces "/" with "--"), so decoding it gives back exactly
    /// what the user chose in the picker.
    ///
    /// nil rather than a guess when there is nothing to read: the caller falls back to the engine
    /// convention, and a digest attributed to "" would be worse than one attributed to "session:mlx".
    public static func fromModelDirectory(_ dir: String) -> String? {
        let parts = components(of: dir)
        guard !parts.isEmpty else { return nil }

        // Searched from the END: a path can contain the cache root more than once only in
        // pathological cases, and the deepest match is the model actually being pointed at.
        if let encoded = parts.last(where: { $0.hasPrefix("models--") }) {
            let repo = String(encoded.dropFirst("models--".count))
            let segments = repo.components(separatedBy: "--").filter { !$0.isEmpty }
            if !segments.isEmpty { return segments.joined(separator: "/") }
        }

        // Not an HF cache entry. The last component is the model, UNLESS the layout says it is a
        // revision - which is decided by the component above it rather than by what it looks like.
        //
        // Name-shape was the first rule here and it was wrong: "a 7-to-40 character hex string is a
        // revision" also eats a model legitimately named `deadbeef`, and the walk upward then
        // answered "models" - the parent directory. That is exactly the "it looks like an answer"
        // failure this file exists to avoid, where nil is the correct output.
        guard let last = parts.last else { return nil }
        if last == "snapshots" || last == "refs" { return nil }
        if parts.count >= 2, parts[parts.count - 2] == "snapshots" {
            return parts.count >= 3 ? parts[parts.count - 3] : nil
        }
        return last
    }

    /// A readable name for a GGUF model FILE, which is what llama-server reports as its
    /// `model_path`. The extension goes; the rest is the name the app itself shows in the model
    /// bar, so the marker and the window agree.
    public static func fromModelFile(_ path: String) -> String? {
        guard let last = components(of: path).last else { return nil }
        guard last.lowercased().hasSuffix(".gguf") else { return last }
        let name = String(last.dropLast(".gguf".count))
        return name.isEmpty ? nil : name
    }

    /// A model path of unknown kind - a directory for MLX, a file for llama-server. Decided by the
    /// extension rather than by touching the filesystem: this runs while a restore is on the
    /// critical path, and a stat that answers "no" for a model on a network volume that is merely
    /// slow would rename the summarizer for no reason.
    public static func fromModelPath(_ path: String) -> String? {
        components(of: path).last?.lowercased().hasSuffix(".gguf") == true
            ? fromModelFile(path) : fromModelDirectory(path)
    }

    // MARK: - Pieces

    private static func components(of path: String) -> [String] {
        path.components(separatedBy: "/").filter { !$0.isEmpty && $0 != "." }
    }

    /// The most this will ever print, and what it will not print.
    ///
    /// Everything here can originate with a SERVER: `/props` reports `model_path`, and this value
    /// is logged whole, returned to the client as `summarizer` and stored in the digest as its
    /// generator. A megabyte of it, or an embedded newline, would land in all three - the same
    /// hazard `ACPServer.describeBackendValue` already caps and strips for the same reasons.
    private static func sanitized(_ name: String) -> String? {
        let flattened = String(
            name.unicodeScalars.map { scalar -> Character in
                CharacterSet.controlCharacters.contains(scalar) ? " " : Character(scalar)
            })
        let trimmed = flattened.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return nil }
        return trimmed.count <= 96 ? trimmed : String(trimmed.prefix(93)) + "..."
    }
}
