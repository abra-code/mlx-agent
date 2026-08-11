// DigestModel.swift - the summarization seam, and the neutral conversation currency it works in.
//
// The seam exists so everything with logic worth asserting - splitting, slicing, merging,
// assembling, rendering - can be tested against a fake that returns canned answers or throws,
// with no model, no GPU and no macOS 26. The single production conformance (FMDigestGenerator)
// stays thin enough to read in one screen, which is the point: the parts that can be wrong are
// on this side of the seam.
//
// `DigestTurn` is a deliberate second message type rather than a reuse of MLXLMCommon's
// `Chat.Message`. This target links neither MLX nor FoundationModels - that is what makes its
// tests run in seconds - so it cannot see that type, and the conversion at the boundary is four
// lines. The same neutrality is why a digest made under Foundation Models can be primed into an
// MLX or OpenAI session.

import Foundation

public enum DigestRole: String, Codable, Sendable {
    case system
    case user
    case assistant
    case tool
}

/// One message, reduced to what summarization needs.
public struct DigestTurn: Sendable, Equatable {
    public var role: DigestRole
    public var content: String

    public init(role: DigestRole, content: String) {
        self.role = role
        self.content = content
    }

    public static func user(_ content: String) -> DigestTurn { .init(role: .user, content: content) }
    public static func assistant(_ content: String) -> DigestTurn {
        .init(role: .assistant, content: content)
    }
    public static func system(_ content: String) -> DigestTurn {
        .init(role: .system, content: content)
    }
    public static func tool(_ content: String) -> DigestTurn { .init(role: .tool, content: content) }
}

/// Something that can summarize a stretch of conversation.
///
/// - Parameters:
///   - source: rendered conversation text, already sized to fit the model's window by
///     `DigestPlanner.slices(of:policy:)`. A conformance does not have to chunk.
///   - priorDigest: what earlier slices of the SAME conversation established, so a later slice can
///     resolve references backward instead of summarizing in a vacuum. nil for the first slice.
///
/// Throwing is a supported outcome, not an exception: the caller's contract is that any failure
/// falls back to priming the full history, so a conformance should throw rather than return
/// something it does not believe.
public protocol DigestModel: Sendable {
    func digest(source: String, priorDigest: DigestContent?) async throws -> DigestContent
}
