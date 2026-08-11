// SessionDigest.swift - a conversation compressed to the facts, as stored, sent and audited.
//
// This is the STORAGE and WIRE type, and it deliberately knows nothing about Foundation Models:
// it is produced by anything conforming to `DigestModel`, consumed by every backend as plain
// history text (see DigestRenderer), and written to disk as an audit artifact. Keeping it free of
// framework types is what makes a digest generated on macOS 26 replayable into an MLX or OpenAI
// session on macOS 14.
//
// VERSIONING is real here rather than aspirational: a digest outlives the process that made it -
// it goes into a client's store and comes back at a restore that may be weeks and several
// releases later. So `schemaVersion` is checked on decode and a FUTURE version is REJECTED rather
// than silently half-read. Within a version, missing fields decode as empty: adding an optional
// field must not invalidate every digest already written.
//
// The wire form is FLAT - `content`'s fields sit alongside the metadata rather than nested - which
// is why Codable is hand-written. Nesting reads better in Swift (the generated fields are exactly
// the ones a model produces, and the protocol returns only those), flat reads better as JSON.

import CryptoKit
import Foundation

/// One tool the earlier conversation used, and what came of it.
///
/// Kept as a triple rather than a sentence because the parts get used separately: a client can
/// show "which tools ran" without the prose, and `materialResult` is the only part that carries
/// information the model would otherwise have to re-derive.
public struct ToolEvent: Codable, Sendable, Equatable {
    /// The tool's name as the model knew it, e.g. "read_file".
    public var tool: String
    /// Why it was called, in a few words.
    public var whatFor: String
    /// What the call actually established. Not the raw output - the part that mattered.
    public var materialResult: String

    public init(tool: String, whatFor: String, materialResult: String) {
        self.tool = tool
        self.whatFor = whatFor
        self.materialResult = materialResult
    }

    var jsonObject: [String: Any] {
        ["tool": tool, "whatFor": whatFor, "materialResult": materialResult]
    }
}

/// The part of a digest a model actually writes.
///
/// Split out from `SessionDigest` so the generation seam cannot produce metadata: `createdAt`,
/// `sourceSHA256` and friends are facts about the summarization, and a model asked to invent them
/// would invent them. `DigestModel` returns this; code supplies the rest.
public struct DigestContent: Codable, Sendable, Equatable {
    /// What the user is still trying to get done. The single most useful line for a resumed
    /// session, and the reason it is not just another list.
    public var unresolvedIntent: String?
    /// Things determined to be true - measurements, versions, names, paths.
    public var establishedFacts: [String]
    /// Choices made and the reason, so the resumed session does not relitigate them.
    public var decisions: [String]
    public var toolEvents: [ToolEvent]
    /// Work explicitly left open.
    public var openThreads: [String]
    /// Constraints the user stated - style, tooling, what not to do.
    public var userPreferences: [String]

    public init(
        unresolvedIntent: String? = nil,
        establishedFacts: [String] = [],
        decisions: [String] = [],
        toolEvents: [ToolEvent] = [],
        openThreads: [String] = [],
        userPreferences: [String] = []
    ) {
        self.unresolvedIntent = unresolvedIntent
        self.establishedFacts = establishedFacts
        self.decisions = decisions
        self.toolEvents = toolEvents
        self.openThreads = openThreads
        self.userPreferences = userPreferences
    }

    /// True when the model produced nothing usable.
    ///
    /// Worth naming: an empty digest is not a small digest, it is a FAILED one, and priming a
    /// preamble that says nothing while dropping the turns it replaced is strictly worse than
    /// priming the full history. Callers branch on this.
    ///
    /// A list of blank strings counts as empty. `DigestPlanner.merge` strips blanks before this is
    /// consulted, so that case does not arise on the planner's path - but the obvious other user
    /// of this property is a `DigestModel` checking its own output before returning it, and that
    /// caller has not been through merge.
    public var isEmpty: Bool {
        func blank(_ items: [String]) -> Bool {
            items.allSatisfy { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        }
        return (unresolvedIntent?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
            && blank(establishedFacts) && blank(decisions)
            && toolEvents.allSatisfy {
                $0.tool.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }
            && blank(openThreads) && blank(userPreferences)
    }
}

/// A digest plus the provenance needed to trust it later.
public struct SessionDigest: Codable, Sendable, Equatable {

    /// Bump when a change would make an older reader misinterpret a digest rather than merely
    /// miss part of it. Adding an optional field is NOT such a change.
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var createdAt: Date
    /// What produced it, e.g. "apple-foundation-models". Recorded because digest quality is a
    /// property of the summarizer, and a bad one has to be identifiable after the fact.
    public var generator: String
    /// How many turns were summarized. Excludes the verbatim tail.
    public var sourceTurnCount: Int
    /// SHA-256 of the exact text that was summarized, so an artifact can be correlated with the
    /// digest that came from it without storing the text twice.
    public var sourceSHA256: String
    public var content: DigestContent

    public init(
        content: DigestContent,
        sourceTurnCount: Int,
        sourceSHA256: String,
        generator: String,
        createdAt: Date,
        schemaVersion: Int = SessionDigest.currentSchemaVersion
    ) {
        self.schemaVersion = schemaVersion
        self.createdAt = createdAt
        self.generator = generator
        self.sourceTurnCount = sourceTurnCount
        self.sourceSHA256 = sourceSHA256
        self.content = content
    }

    // MARK: - Codable (flat wire, nested model)

    private enum CodingKeys: String, CodingKey {
        case schemaVersion, createdAt, generator, sourceTurnCount, sourceSHA256
        case unresolvedIntent, establishedFacts, decisions, toolEvents, openThreads,
            userPreferences
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let version = try c.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
        // Refuse the future rather than guess at it. A version we do not know may have moved a
        // meaning, not just added a field, and a half-understood digest primed as context is a
        // confident lie - the worst possible failure for this feature.
        guard version <= Self.currentSchemaVersion else {
            throw DecodingError.dataCorruptedError(
                forKey: .schemaVersion, in: c,
                debugDescription:
                    "digest schemaVersion \(version) is newer than this build understands "
                    + "(\(Self.currentSchemaVersion))")
        }
        schemaVersion = version
        // Dates are ISO 8601 strings on purpose: a digest is read by other tools and by humans,
        // and Codable's default (a bare seconds-since-2001 Double) is neither.
        //
        // A MISSING timestamp is tolerated (that is the within-version rule); a PRESENT but
        // unparsable one throws. Silently substituting 1970 for a corrupt field is the same
        // confident lie the version guard above exists to refuse.
        if let stamp = try c.decodeIfPresent(String.self, forKey: .createdAt) {
            guard let parsed = Self.parseTimestamp(stamp) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .createdAt, in: c,
                    debugDescription: "createdAt is not an ISO 8601 timestamp: \(stamp)")
            }
            createdAt = parsed
        } else {
            createdAt = Date(timeIntervalSince1970: 0)
        }
        generator = try c.decodeIfPresent(String.self, forKey: .generator) ?? "unknown"
        sourceTurnCount = try c.decodeIfPresent(Int.self, forKey: .sourceTurnCount) ?? 0
        sourceSHA256 = try c.decodeIfPresent(String.self, forKey: .sourceSHA256) ?? ""
        content = DigestContent(
            unresolvedIntent: try c.decodeIfPresent(String.self, forKey: .unresolvedIntent),
            establishedFacts: try c.decodeIfPresent([String].self, forKey: .establishedFacts) ?? [],
            decisions: try c.decodeIfPresent([String].self, forKey: .decisions) ?? [],
            toolEvents: try c.decodeIfPresent([ToolEvent].self, forKey: .toolEvents) ?? [],
            openThreads: try c.decodeIfPresent([String].self, forKey: .openThreads) ?? [],
            userPreferences: try c.decodeIfPresent([String].self, forKey: .userPreferences) ?? [])
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(schemaVersion, forKey: .schemaVersion)
        try c.encode(Self.formatTimestamp(createdAt), forKey: .createdAt)
        try c.encode(generator, forKey: .generator)
        try c.encode(sourceTurnCount, forKey: .sourceTurnCount)
        try c.encode(sourceSHA256, forKey: .sourceSHA256)
        try c.encodeIfPresent(content.unresolvedIntent, forKey: .unresolvedIntent)
        try c.encode(content.establishedFacts, forKey: .establishedFacts)
        try c.encode(content.decisions, forKey: .decisions)
        try c.encode(content.toolEvents, forKey: .toolEvents)
        try c.encode(content.openThreads, forKey: .openThreads)
        try c.encode(content.userPreferences, forKey: .userPreferences)
    }

    // MARK: - JSONSerialization form

    /// The same shape as `encode(to:)`, as a plain dictionary.
    ///
    /// It exists because the ACP wire is built with JSONSerialization, not JSONEncoder, so a
    /// digest going out in a response would otherwise have to be encoded to Data and parsed back
    /// just to be re-encoded. `jsonObjectMatchesCodable` in the tests pins the two together, which
    /// is what keeps this from drifting into a second, subtly different format.
    public var jsonObject: [String: Any] {
        var out: [String: Any] = [
            "schemaVersion": schemaVersion,
            "createdAt": Self.formatTimestamp(createdAt),
            "generator": generator,
            "sourceTurnCount": sourceTurnCount,
            "sourceSHA256": sourceSHA256,
            "establishedFacts": content.establishedFacts,
            "decisions": content.decisions,
            "toolEvents": content.toolEvents.map(\.jsonObject),
            "openThreads": content.openThreads,
            "userPreferences": content.userPreferences,
        ]
        if let intent = content.unresolvedIntent { out["unresolvedIntent"] = intent }
        return out
    }

    // MARK: - Helpers

    /// Lowercase hex SHA-256, used for `sourceSHA256`.
    public static func sha256Hex(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    // Formatters are built per call rather than cached in a static. They are used a handful of
    // times per session (once per condensation), so the allocation is free, and ISO8601DateFormatter
    // is a non-Sendable class that would need an unsafe annotation to live in a static under
    // strict concurrency. Not worth it for this call rate.
    /// Fractional seconds are WRITTEN, not just tolerated on read. Without them a digest does not
    /// survive its own round trip: `Date()` always carries a fraction, so every real digest would
    /// decode to a slightly different value than it encoded from, and any equality check against a
    /// reloaded copy would fail for a reason nobody would look for.
    static func formatTimestamp(_ date: Date) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f.string(from: date)
    }

    static func parseTimestamp(_ text: String) -> Date? {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = f.date(from: text) { return d }
        // Accept a whole-second stamp from another writer rather than losing the timestamp.
        f.formatOptions = [.withInternetDateTime]
        return f.date(from: text)
    }
}
