// LenientDecode.swift - reading a digest out of what a model without guided generation actually
// returns.
//
// Foundation Models guarantees a parseable shape: `@Generable` constrains decoding, so
// FMDigestGenerator never sees malformed output. No other backend offers that. An MLX or
// llama-server model asked for JSON returns JSON *and* whatever else it felt like saying - a
// preamble, a code fence, a <think> block, snake_case keys, a bare string where a list was asked
// for. `JSONDecoder` refuses all of it, and refusing is the wrong answer when the content is
// sitting right there.
//
// So this is the other half of the backend-backed summarizer, and it lives in the LIBRARY rather
// than beside the generator on purpose: it is the only part of that feature a unit test can pin
// down. The generator is a prompt and a stream; this is the parser, and parsers are where the bugs
// are.
//
// WHAT IT WILL NOT DO. Tolerance stops at the point where guessing starts, and every rule below
// exists because the alternative is a NON-EMPTY digest that is wrong - the one failure nothing
// downstream can catch, since `condense`'s only quality gate is emptiness and the digest REPLACES
// the turns it summarizes.
//
//   - A truncated object is rejected rather than closed with a synthetic brace. Truncation means
//     the token cap: the last list is cut mid-thought and the ones after it are missing entirely.
//   - A value that cannot be read - an object where a list belongs, a list of objects where
//     strings belong - fails the whole decode rather than emptying that one field. Emptying it
//     would leave the siblings behind, which reads as a thin summary rather than as a failure.
//   - The prompt's own `<...>` placeholders are not content, and neither is an object that has
//     the right shape and nothing in it. A model restating the format is common; a digest that
//     says "what the user is still trying to accomplish" would be primed as fact.
//   - When a reply holds more than one digest-shaped object, the LAST one wins: a model echoes the
//     template before answering and corrects itself after, so the answer is at the end.
//
// Every throw here is caught by `DigestPlanner.condense`, which primes the full history instead.

import Foundation

/// Why a model's reply could not be read as a digest. `LocalizedError` because
/// `DigestPlanner.condense` surfaces `errorDescription` to the client as the `reason` a
/// condensation did not happen.
public enum DigestDecodeError: Error, Equatable, LocalizedError {
    /// No `{` at all - the model answered in prose.
    case noJSONObject
    /// An object was opened and never closed. Almost always the token cap.
    case truncatedJSON
    /// Braces balanced, but not valid JSON in between.
    case malformedJSON(String)
    /// Valid JSON, none of it ours.
    case noRecognizedFields
    /// Our shape, no content: every field empty or still holding the prompt's `<...>` placeholders.
    /// A model restating the format instead of answering.
    case emptyDigest

    public var errorDescription: String? {
        switch self {
        case .noJSONObject:
            return "the summarizer answered in prose, with no JSON object"
        case .truncatedJSON:
            return "the summarizer's JSON was cut off before it closed"
        case .malformedJSON(let detail):
            return "the summarizer's JSON did not parse: \(detail)"
        case .noRecognizedFields:
            return "the summarizer's JSON had none of the digest's fields"
        case .emptyDigest:
            return "the summarizer returned the empty template rather than a digest"
        }
    }
}

extension DigestContent {

    /// The digest's wire shape as a fillable example, for prompting a model that has no guided
    /// generation. Instruct-tuned models reproduce an example far more reliably than they follow a
    /// description of one.
    ///
    /// It lives HERE, next to the decoder, because the two are one contract: every slot is
    /// angle-bracketed precisely so `init(lenientJSON:)` can refuse a model that echoes this back
    /// instead of filling it in, and a copy of this text in the prompt file could drift out of
    /// that agreement without anything failing. `templateIsNotADigest` in the tests is the loop
    /// closed.
    public static let jsonTemplate = """
        {
          "unresolvedIntent": "<what the user is still trying to accomplish, one sentence, or \
        an empty string if nothing is outstanding>",
          "establishedFacts": ["<a specific thing this conversation established as true>"],
          "decisions": ["<a choice that was made, with its reason>"],
          "toolEvents": [{"tool": "<name>", "whatFor": "<why it ran>", "materialResult": \
        "<what it established>"}],
          "openThreads": ["<work explicitly left unfinished>"],
          "userPreferences": ["<a constraint the user stated about how the work should be done>"]
        }
        """

    /// Read a digest out of a model's raw reply.
    ///
    /// Accepts, in order of how often each is actually seen: a clean object; an object inside a
    /// ```` ``` ```` fence; an object with prose before or after it; keys in snake_case, kebab-case
    /// or with spaces; a single object wrapping the real one under one key; a bare string where a
    /// list was asked for; numbers in a list of strings; blank entries (dropped).
    ///
    /// Throws `DigestDecodeError` on the cases named in the file header.
    public init(lenientJSON text: String) throws {
        var sawUnterminated = false
        let candidates = Self.candidateObjects(in: text, sawUnterminated: &sawUnterminated)

        var parseFailure: String?
        var sawForeignObject = false
        var sawEmptyDigest = false
        var found: DigestContent?
        for candidate in candidates {
            do {
                let parsed = try JSONSerialization.jsonObject(with: Data(candidate.utf8))
                guard let object = parsed as? [String: Any] else { continue }
                // THE LAST schema-shaped object wins, not the first. A model that restates the
                // format before answering emits the prompt's own example first, and a model that
                // corrects itself ("actually, here it is:") emits the good one last. Either way
                // the answer is the last one; taking the first would mean summarizing a
                // conversation into the words of the instructions that asked for the summary.
                if let content = Self.content(from: object, depth: 0) {
                    // An all-placeholder or all-empty object is the TEMPLATE, not an answer, and
                    // taking it would let a later echo overwrite a real earlier digest. It is
                    // remembered only so the diagnosis can name what happened.
                    if content.isEmpty {
                        sawEmptyDigest = true
                    } else {
                        found = content
                    }
                    continue
                }
                sawForeignObject = true
            } catch {
                // Keep the FIRST failure: with prose around the JSON the later candidates tend to
                // be incidental braces, and their errors describe those rather than the real reply.
                if parseFailure == nil { parseFailure = Self.oneLine(error) }
            }
        }
        if let found {
            self = found
            return
        }

        // Ordered by how ACTIONABLE the diagnosis is, not by how specific. Truncation names a
        // cause the operator can do something about (the token cap); the others describe a model
        // that answered the wrong question.
        if sawUnterminated { throw DigestDecodeError.truncatedJSON }
        if sawEmptyDigest { throw DigestDecodeError.emptyDigest }
        if sawForeignObject { throw DigestDecodeError.noRecognizedFields }
        if let parseFailure { throw DigestDecodeError.malformedJSON(parseFailure) }
        throw DigestDecodeError.noJSONObject
    }

    // MARK: - Finding the object

    /// Every balanced top-level `{...}` in `text`, in order.
    ///
    /// More than one because a decoy is ordinary: a reasoning model narrates ("I will return
    /// {facts: ...} style output") before answering, and a fenced example in the prompt gets echoed
    /// back. The caller tries each in turn and takes the first that is actually a digest, which is
    /// cheaper and more predictable than trying to guess which brace starts the real answer.
    ///
    /// Nested braces and braces inside string literals are tracked so a fact containing "{" cannot
    /// end the object early. Scanning resumes AFTER a matched object rather than at the next
    /// character, so this stays linear.
    ///
    /// An UNTERMINATED brace does not end the search, because "everything after it is inside that
    /// object" is only true when the model was cut off - and a reasoning preamble that abandons a
    /// half-written draft ("{"facts": [ ... no, let me start over") opens a brace it never closes
    /// with the real answer still to come. The re-scan is BOUNDED, though: resuming at the next
    /// character on every failure is quadratic (measured: 2.7 s for 40k stray braces, and it grows
    /// from there), and past a handful of unclosed braces the reply is not a digest anyway.
    ///
    /// - Parameter sawUnterminated: set when an object opens and never closes. That is the
    ///   token-cap signature, and it is worth distinguishing from "no JSON here at all".
    private static func candidateObjects(in text: String, sawUnterminated: inout Bool) -> [String] {
        let chars = Array(text)
        var out: [String] = []
        var index = 0
        var abandoned = 0
        while index < chars.count {
            guard chars[index] == "{" else {
                index += 1
                continue
            }
            var depth = 0
            var inString = false
            var escaped = false
            var scan = index
            var closed = false
            while scan < chars.count {
                let c = chars[scan]
                if inString {
                    if escaped {
                        escaped = false
                    } else if c == "\\" {
                        escaped = true
                    } else if c == "\"" {
                        inString = false
                    }
                } else if c == "\"" {
                    inString = true
                } else if c == "{" {
                    depth += 1
                } else if c == "}" {
                    depth -= 1
                    if depth == 0 {
                        closed = true
                        break
                    }
                }
                scan += 1
            }
            guard closed else {
                sawUnterminated = true
                abandoned += 1
                // Eight failed re-scans is far past every real reply, and it keeps the worst case
                // at 8n rather than n^2.
                guard abandoned <= 8 else { break }
                index += 1
                continue
            }
            out.append(String(chars[index...scan]))
            index = scan + 1
        }
        return out
    }

    // MARK: - Reading the fields

    /// nil when `object` carries none of the digest's fields.
    ///
    /// - Parameter depth: guards the wrapper unwrap below. Three is past every real case
    ///   (`{"digest": {...}}`, `{"response": {"digest": {...}}}`) and stops a pathological
    ///   `{"a":{"a":{"a":...}}}` from recursing as deep as the input is long.
    private static func content(from object: [String: Any], depth: Int) -> DigestContent? {
        var byKey: [String: Any] = [:]
        // Sorted, first-wins, so two keys that normalize the same ("toolEvents" and "tool_events"
        // in one object) resolve the SAME WAY every run. Dictionary order is seeded per process,
        // so last-wins here made the digest for one input vary between runs - which quietly
        // undoes the determinism greedy sampling is chosen for.
        for key in object.keys.sorted() {
            guard let value = object[key], !(value is NSNull) else { continue }
            let normalized = normalizedKey(key)
            if byKey[normalized] == nil { byKey[normalized] = value }
        }

        let intent = byKey["unresolvedintent"]
        let facts = byKey["establishedfacts"]
        let decisions = byKey["decisions"]
        let events = byKey["toolevents"]
        let threads = byKey["openthreads"]
        let preferences = byKey["userpreferences"]

        let anyPresent = [intent, facts, decisions, events, threads, preferences].contains {
            $0 != nil
        }
        guard anyPresent else {
            // A model told to "reply with a JSON object" sometimes labels it first. One value, and
            // that value an object, is the only shape where descending is unambiguous.
            if depth < 3, object.count == 1, let inner = object.values.first as? [String: Any] {
                return content(from: inner, depth: depth + 1)
            }
            return nil
        }

        // A PRESENT key whose value cannot be read fails the whole object, and that asymmetry is
        // the point: an absent key is a model with nothing to record, while `"establishedFacts":
        // {"0": "..."}` is a model that answered in a shape this cannot read. Emptying the field
        // would leave a non-empty digest - `isEmpty` is false as long as one sibling parsed - so
        // the planner would report success while the facts that justified dropping those turns
        // silently vanished. Returning nil here surfaces as `.noRecognizedFields`, which primes
        // the full history.
        guard let facts = stringList(facts), let decisions = stringList(decisions),
            let threads = stringList(threads), let preferences = stringList(preferences),
            let events = toolEvents(events)
        else { return nil }
        // An intent that is present but unreadable fails for the same reason, and it matters more:
        // it is the single most useful line in a digest. Note that BLANK is not unreadable - the
        // prompt asks for "" when nothing is outstanding, so an empty string is a real answer.
        if let intent, !readableAsString(intent) { return nil }

        return DigestContent(
            unresolvedIntent: firstString(intent),
            establishedFacts: facts,
            decisions: decisions,
            toolEvents: events,
            openThreads: threads,
            userPreferences: preferences)
    }

    /// Case, underscores, hyphens and spaces all removed, so `established_facts`,
    /// `Established Facts` and `established-facts` are one key.
    private static func normalizedKey(_ key: String) -> String {
        String(key.lowercased().unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) })
    }

    /// A list of strings from a list, or from a single string asked to be one.
    ///
    /// nil means the value is not readable as a list at all - an object where a list belongs, or a
    /// list holding objects or nested lists. See `content(from:depth:)` for why that is a failure
    /// rather than an empty field.
    private static func stringList(_ value: Any?) -> [String]? {
        guard let value else { return [] }
        if let single = value as? String { return kept([single]) }
        guard let array = value as? [Any] else { return nil }
        var raw: [String] = []
        for element in array where !(element is NSNull) {
            guard let text = scalar(element) else { return nil }
            raw.append(text)
        }
        return kept(raw)
    }

    /// Whether a value can be read as a single string. An array counts: a model that puts every
    /// list in brackets tends to bracket this too. A blank string counts as well - the prompt asks
    /// for "" when nothing is outstanding, so empty is an answer, not a failure.
    private static func readableAsString(_ value: Any) -> Bool {
        if value is String { return true }
        if let array = value as? [Any] { return array.allSatisfy { $0 is String || $0 is NSNull } }
        return false
    }

    /// The first usable string, for a field that is one string.
    ///
    /// Strings only, deliberately: `"unresolvedIntent": 0` stringifies to "0", which would become
    /// the preamble's "Unresolved intent: 0" - never meaningful, and readable as a fact by the
    /// model that reads it back. Numbers are still kept inside LISTS, where a measurement is a
    /// plausible entry.
    private static func firstString(_ value: Any?) -> String? {
        guard let value else { return nil }
        if let single = value as? String {
            return trimmed(single).flatMap { isPlaceholder($0) ? nil : $0 }
        }
        if let array = value as? [Any] {
            return array.compactMap { $0 as? String }.compactMap(trimmed)
                .first { !isPlaceholder($0) }
        }
        return nil
    }

    /// nil when the value cannot be read as tool events at all.
    private static func toolEvents(_ value: Any?) -> [ToolEvent]? {
        guard let value else { return [] }
        let elements: [Any] = (value as? [Any]) ?? [value]
        var out: [ToolEvent] = []
        for element in elements where !(element is NSNull) {
            // A bare name is worth keeping: the renderer emits "- read_file" for it, which is still
            // more than the resumed session would otherwise know.
            if let name = element as? String {
                guard let tool = trimmed(name), !isPlaceholder(tool) else { continue }
                out.append(ToolEvent(tool: tool, whatFor: "", materialResult: ""))
                continue
            }
            guard let dictionary = element as? [String: Any] else { return nil }
            var byKey: [String: Any] = [:]
            for key in dictionary.keys.sorted() {
                guard let value = dictionary[key], !(value is NSNull) else { continue }
                let normalized = normalizedKey(key)
                if byKey[normalized] == nil { byKey[normalized] = value }
            }
            func text(_ names: String...) -> String {
                for name in names {
                    if let found = firstString(byKey[name]) { return found }
                }
                return ""
            }
            let tool = text("tool", "name", "toolname")
            // No name, no event: `whatFor` alone names nothing the resumed session can act on, and
            // the renderer drops it anyway. Dropped rather than failed because a model asked for up
            // to N events pads the list with empty ones - that is padding, not a wrong shape.
            guard !tool.isEmpty else { continue }
            out.append(
                ToolEvent(
                    tool: tool,
                    whatFor: text("whatfor", "purpose", "why"),
                    materialResult: text("materialresult", "result", "outcome")))
        }
        return out
    }

    /// An unfilled slot from the prompt's own example, e.g. `<a fact>`.
    ///
    /// Restating the format before answering is one of the most ordinary things a model does, and
    /// the echo is a perfectly valid digest-shaped object. Without this, the digest for that reply
    /// would be the instructions that asked for it - non-empty, so nothing downstream would catch
    /// it, and every summarized turn dropped in its favor. `BackendDigestGenerator.instructions`
    /// brackets every placeholder precisely so this rule can exist.
    private static func isPlaceholder(_ text: String) -> Bool {
        text.hasPrefix("<") && text.hasSuffix(">") && text.count > 1
    }

    /// A JSON scalar as text. Numbers are kept because a list of measurements comes back as
    /// numbers about as often as it comes back as strings; objects and nested arrays are not,
    /// because there is no reading of them that is not an invention.
    private static func scalar(_ value: Any) -> String? {
        if let text = value as? String { return text }
        // Booleans arrive as NSNumber too and stringify as "1"/"0". Nonsense in a list of facts,
        // but it is the model's nonsense, not a misreading, and `kept` drops nothing but blanks.
        if let number = value as? NSNumber { return number.stringValue }
        return nil
    }

    private static func kept(_ items: [String]) -> [String] {
        items.compactMap(trimmed).filter { !isPlaceholder($0) }
    }

    private static func trimmed(_ text: String) -> String? {
        let out = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return out.isEmpty ? nil : out
    }

    /// JSONSerialization's message, flattened - it is going into a one-line `reason`.
    private static func oneLine(_ error: Error) -> String {
        let raw =
            ((error as NSError).userInfo[NSDebugDescriptionErrorKey] as? String)
            ?? (error as NSError).localizedDescription
        return raw.split(whereSeparator: \.isNewline).joined(separator: " ")
    }
}
