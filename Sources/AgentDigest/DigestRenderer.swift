// DigestRenderer.swift - a digest as the text a model will actually read.
//
// This function is why a digest is portable. Every backend consumes a condensed session as an
// ordinary user turn in its own chat template - MLX, llama-server and Foundation Models alike -
// because what gets primed is prose, not a serialization anyone has to agree on. No backend-
// specific rendering, no structured-context API, nothing to keep in step across three engines.
//
// It is deterministic and pure: same digest in, same characters out. That is what makes a golden
// test meaningful, and a golden test is the only defense against the wording drifting into
// something that reads as an instruction to the model rather than as restored context.

import Foundation

public enum DigestRenderer {

    /// The assistant half of the primed pair. Short on purpose: it exists to preserve role
    /// alternation, and anything longer would be text the model treats as its own prior style.
    public static let acknowledgment = "Acknowledged."

    /// Render a digest as the user turn that carries restored context.
    ///
    /// - Parameter verbatimTurns: how many original messages follow this pair unmodified. Stated
    ///   in the text because the difference between "everything you see is a summary" and "the
    ///   recent part is exact" changes how much the model should trust the wording of what
    ///   follows.
    public static func renderPreamble(_ digest: SessionDigest, verbatimTurns: Int = 0) -> String {
        var out = [String]()

        // Framed as a record handed to the model, not as instructions. A preamble that says "you
        // previously decided X" invites the model to defend X; "the earlier conversation
        // established X" leaves it as evidence.
        var opening = "Context restored from an earlier conversation. The "
        opening += plural(digest.sourceTurnCount, "message")
        opening += digest.sourceTurnCount == 1 ? " before this point is" : " before this point are"
        opening += " summarized below rather than quoted, so treat the "
        opening += "wording as approximate and the content as established."
        if verbatimTurns > 0 {
            opening += " The " + plural(verbatimTurns, "message")
            opening += verbatimTurns == 1 ? " after this exchange is" : " after this exchange are"
            opening += " the original text, unmodified."
        }
        out.append(opening)

        if let intent = nonEmpty(digest.content.unresolvedIntent) {
            out.append("Unresolved intent: \(intent)")
        }
        appendList(&out, "Established facts", digest.content.establishedFacts)
        appendList(&out, "Decisions", digest.content.decisions)
        let toolLines = digest.content.toolEvents.compactMap { event -> String? in
            guard let name = nonEmpty(event.tool) else { return nil }
            let purpose = nonEmpty(event.whatFor).map { " (\($0))" } ?? ""
            let result = nonEmpty(event.materialResult).map { ": \($0)" } ?? ""
            return "- \(name)\(purpose)\(result)"
        }
        if !toolLines.isEmpty {
            out.append((["Tool activity:"] + toolLines).joined(separator: "\n"))
        }
        appendList(&out, "Open threads", digest.content.openThreads)
        appendList(&out, "Stated preferences", digest.content.userPreferences)

        return out.joined(separator: "\n\n")
    }

    private static func appendList(_ out: inout [String], _ heading: String, _ items: [String]) {
        let kept = items.compactMap(nonEmpty)
        guard !kept.isEmpty else { return }
        out.append(([heading + ":"] + kept.map { "- \($0)" }).joined(separator: "\n"))
    }

    private static func plural(_ count: Int, _ noun: String) -> String {
        "\(count) \(noun)\(count == 1 ? "" : "s")"
    }

    /// Trim, and flatten interior newlines to spaces.
    ///
    /// The flattening is not cosmetic. Items come from a model, and one containing "\n- something"
    /// would render as a second bullet under whatever heading it landed in - a summary inventing
    /// structure it was not given. Both readers of this text are models, so the cheap fix is to
    /// make a list item incapable of being more than one item.
    private static func nonEmpty(_ text: String?) -> String? {
        guard let text else { return nil }
        let flattened = text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        return flattened.isEmpty ? nil : flattened
    }
}
