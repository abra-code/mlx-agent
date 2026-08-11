// SessionDigestTests.swift - the format contract.
//
// A digest is written by one build and read by another, possibly months later, so these are less
// about "does Codable work" and more about the two promises that make that safe: a future version
// is refused outright, and a missing field within a known version is tolerated.

import Foundation
import Testing

@testable import AgentDigest

@Suite("SessionDigest")
struct SessionDigestTests {

    static func sample() -> SessionDigest {
        SessionDigest(
            content: DigestContent(
                unresolvedIntent: "get the notarization step working",
                establishedFacts: ["the binary is arm64-only", "notarytool needs an app password"],
                decisions: ["ship as a .pkg, not a .dmg"],
                toolEvents: [
                    ToolEvent(
                        tool: "read_file", whatFor: "check the entitlements",
                        materialResult: "sandbox is off")
                ],
                openThreads: ["stapling is untested"],
                userPreferences: ["ASCII only in commit messages"]),
            sourceTurnCount: 22,
            sourceSHA256: SessionDigest.sha256Hex("source"),
            generator: "test-generator",
            // Deliberately fractional. A whole-second literal here would hide the writer dropping
            // sub-second precision, which is exactly what every real `Date()` carries.
            createdAt: Date(timeIntervalSince1970: 1_700_000_000.75))
    }

    @Test("round-trips through JSON unchanged")
    func roundTrip() throws {
        let original = Self.sample()
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(SessionDigest.self, from: data)
        #expect(decoded == original)
    }

    @Test("the wire form is flat, not nested under content")
    func flatWire() throws {
        let data = try JSONEncoder().encode(Self.sample())
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["content"] == nil)
        #expect(object["establishedFacts"] != nil)
        #expect(object["schemaVersion"] as? Int == 1)
        // ISO 8601, not a bare number - a digest gets read by humans and other tools.
        #expect(object["createdAt"] as? String == "2023-11-14T22:13:20.750Z")
    }

    @Test("jsonObject matches what Codable writes")
    func jsonObjectMatchesCodable() throws {
        // The two serializations exist for different call sites (JSONSerialization on the ACP
        // wire, JSONEncoder for artifacts). This is what stops them becoming two formats.
        let digest = Self.sample()
        let viaCodable = try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(digest))
                as? [String: Any])
        let direct = digest.jsonObject
        #expect(Set(viaCodable.keys) == Set(direct.keys))
        let a = try JSONSerialization.data(withJSONObject: viaCodable, options: [.sortedKeys])
        let b = try JSONSerialization.data(withJSONObject: direct, options: [.sortedKeys])
        #expect(a == b)
    }

    @Test("a newer schema version is refused rather than half-read")
    func futureVersionRejected() throws {
        var object = Self.sample().jsonObject
        object["schemaVersion"] = SessionDigest.currentSchemaVersion + 1
        let data = try JSONSerialization.data(withJSONObject: object)
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(SessionDigest.self, from: data)
        }
    }

    @Test("missing fields within a known version decode as empty")
    func missingFieldsTolerated() throws {
        // What a digest written by an older build looks like to this one.
        let data = try #require(#"{"schemaVersion":1,"decisions":["keep it"]}"#.data(using: .utf8))
        let decoded = try JSONDecoder().decode(SessionDigest.self, from: data)
        #expect(decoded.content.decisions == ["keep it"])
        #expect(decoded.content.establishedFacts.isEmpty)
        #expect(decoded.content.unresolvedIntent == nil)
        #expect(decoded.generator == "unknown")
        #expect(decoded.sourceTurnCount == 0)
    }

    @Test("an absent unresolvedIntent is omitted, not written as null")
    func intentOmitted() throws {
        var digest = Self.sample()
        digest.content.unresolvedIntent = nil
        #expect(digest.jsonObject["unresolvedIntent"] == nil)
        let object = try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(digest))
                as? [String: Any])
        #expect(object["unresolvedIntent"] == nil)
    }

    @Test("a whole-second timestamp from another writer still parses")
    func wholeSeconds() throws {
        #expect(SessionDigest.parseTimestamp("2023-11-14T22:13:20Z") != nil)
        #expect(SessionDigest.parseTimestamp("2023-11-14T22:13:20.500Z") != nil)
    }

    @Test("a present but unparsable timestamp throws instead of becoming 1970")
    func corruptTimestampThrows() throws {
        let data = try #require(
            #"{"schemaVersion":1,"createdAt":"14 Nov 2023"}"#.data(using: .utf8))
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(SessionDigest.self, from: data)
        }
    }

    @Test("an absent timestamp is still tolerated")
    func missingTimestampTolerated() throws {
        let data = try #require(#"{"schemaVersion":1,"decisions":["x"]}"#.data(using: .utf8))
        let decoded = try JSONDecoder().decode(SessionDigest.self, from: data)
        #expect(decoded.createdAt == Date(timeIntervalSince1970: 0))
    }

    @Test("sha256 is stable lowercase hex")
    func sha256() {
        #expect(
            SessionDigest.sha256Hex("abc")
                == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }

    @Test("emptiness is about content, not about whitespace")
    func emptiness() {
        #expect(DigestContent().isEmpty)
        #expect(DigestContent(unresolvedIntent: "   ").isEmpty)
        #expect(!DigestContent(unresolvedIntent: "finish the port").isEmpty)
        #expect(!DigestContent(decisions: ["use a pkg"]).isEmpty)
        // A non-empty array of blanks is still nothing usable, and this is the check a
        // DigestModel would use on its own output before returning it - a caller that has not
        // been through the planner's merge, which is what strips blanks on that path.
        #expect(DigestContent(establishedFacts: ["", "   "]).isEmpty)
        #expect(DigestContent(toolEvents: [ToolEvent(tool: " ", whatFor: "x", materialResult: "y")])
            .isEmpty)
    }
}
