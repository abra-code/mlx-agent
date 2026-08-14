// LenientDecodeTests.swift - the only testable half of backend-backed summarization.
//
// `BackendDigestGenerator` is a prompt, a stream and a retry; none of that can be asserted without
// a model. What CAN be asserted is what happens to the bytes that come back, and that is where the
// failure modes actually live - a model without guided generation returns JSON wrapped in a fence,
// or with a preamble, or with snake_case keys, or cut off at the token cap.
//
// The inputs below are shaped like real model output rather than like unit-test fixtures, and the
// negative cases carry as much weight as the positive ones: the contract is that a reply which
// cannot be read THROWS, so that DigestPlanner primes the full history. A decoder that returned a
// half-digest instead would drop turns and report success.

import Foundation
import Testing

@testable import AgentDigest

@Suite("DigestContent(lenientJSON:)")
struct LenientDecodeTests {

    /// A well-formed reply, as the prompt asks for it.
    static let clean = """
        {
          "unresolvedIntent": "get notarization working",
          "establishedFacts": ["the binary is arm64-only", "notarytool needs an app password"],
          "decisions": ["ship a .pkg, not a .dmg"],
          "toolEvents": [
            {"tool": "read_file", "whatFor": "check entitlements", "materialResult": "sandbox off"}
          ],
          "openThreads": ["stapling is untested"],
          "userPreferences": ["ASCII only"]
        }
        """

    // MARK: - What must parse

    @Test("reads a clean object")
    func cleanObject() throws {
        let content = try DigestContent(lenientJSON: Self.clean)
        #expect(content.unresolvedIntent == "get notarization working")
        #expect(content.establishedFacts.count == 2)
        #expect(content.decisions == ["ship a .pkg, not a .dmg"])
        #expect(content.toolEvents.count == 1)
        #expect(content.toolEvents.first?.tool == "read_file")
        #expect(content.toolEvents.first?.materialResult == "sandbox off")
        #expect(content.openThreads == ["stapling is untested"])
        #expect(content.userPreferences == ["ASCII only"])
    }

    @Test("reads an object inside a code fence")
    func fenced() throws {
        let reply = "```json\n\(Self.clean)\n```"
        let content = try DigestContent(lenientJSON: reply)
        #expect(content.unresolvedIntent == "get notarization working")
        #expect(content.establishedFacts.count == 2)
    }

    @Test("reads an object with prose before and after it")
    func prosePreamble() throws {
        let reply = """
            Sure! Here are the structured notes for that section of the conversation:

            \(Self.clean)

            Let me know if you would like more detail on any of these points.
            """
        let content = try DigestContent(lenientJSON: reply)
        #expect(content.decisions == ["ship a .pkg, not a .dmg"])
    }

    /// A reasoning model narrates before answering, and the narration contains braces. The decoder
    /// must walk past the decoy rather than stop at the first `{` it sees.
    @Test("skips a decoy object and takes the real one")
    func decoyObject() throws {
        let reply = """
            <think>The user wants {facts, decisions, tool events}. I will use the schema
            {"a": 1, "b": {"c": 2}} as a starting point.</think>
            \(Self.clean)
            """
        let content = try DigestContent(lenientJSON: reply)
        #expect(content.unresolvedIntent == "get notarization working")
    }

    @Test("tolerates an object with only two of the six fields")
    func partialObject() throws {
        let content = try DigestContent(
            lenientJSON: #"{"establishedFacts": ["the M5 Air is fanless"], "decisions": []}"#)
        #expect(content.establishedFacts == ["the M5 Air is fanless"])
        #expect(content.decisions.isEmpty)
        #expect(content.unresolvedIntent == nil)
        #expect(content.toolEvents.isEmpty)
        #expect(content.openThreads.isEmpty)
        #expect(content.userPreferences.isEmpty)
    }

    @Test("coerces a bare string where a list was asked for")
    func bareStringForList() throws {
        let content = try DigestContent(
            lenientJSON: #"{"establishedFacts": "the build is arm64-only"}"#)
        #expect(content.establishedFacts == ["the build is arm64-only"])
    }

    @Test("coerces a list where a single string was asked for")
    func listForBareString() throws {
        let content = try DigestContent(
            lenientJSON: #"{"unresolvedIntent": ["   ", "ship the release"]}"#)
        #expect(content.unresolvedIntent == "ship the release")
    }

    @Test("accepts snake_case, kebab-case and spaced keys")
    func keyStyles() throws {
        let content = try DigestContent(
            lenientJSON: """
                {"unresolved_intent": "finish the port",
                 "established-facts": ["swift 6 strict concurrency is on"],
                 "Open Threads": ["the smoke script is unwritten"],
                 "user_preferences": ["no em dashes"]}
                """)
        #expect(content.unresolvedIntent == "finish the port")
        #expect(content.establishedFacts == ["swift 6 strict concurrency is on"])
        #expect(content.openThreads == ["the smoke script is unwritten"])
        #expect(content.userPreferences == ["no em dashes"])
    }

    @Test("unwraps a single labeled wrapper object")
    func wrapperObject() throws {
        let content = try DigestContent(lenientJSON: #"{"digest": \#(Self.clean)}"#)
        #expect(content.unresolvedIntent == "get notarization working")
    }

    @Test("keeps tool events under alias keys, and a bare name")
    func toolEventShapes() throws {
        let content = try DigestContent(
            lenientJSON: """
                {"toolEvents": [
                   {"name": "grep", "why": "find the call site", "result": "one hit in Map.swift"},
                   "xcodebuild",
                   {"whatFor": "nameless, dropped"},
                   {"tool": "   ", "whatFor": "blank, dropped"}
                 ]}
                """)
        #expect(content.toolEvents.count == 2)
        #expect(content.toolEvents.first?.tool == "grep")
        #expect(content.toolEvents.first?.whatFor == "find the call site")
        #expect(content.toolEvents.first?.materialResult == "one hit in Map.swift")
        #expect(content.toolEvents.last?.tool == "xcodebuild")
        #expect(content.toolEvents.last?.whatFor.isEmpty == true)
    }

    @Test("drops blank entries and nulls, keeps numbers")
    func entryCleanup() throws {
        let content = try DigestContent(
            lenientJSON: """
                {"establishedFacts": ["  ", "peak was 43.2 tok/s", 128, null],
                 "decisions": null}
                """)
        #expect(content.establishedFacts == ["peak was 43.2 tok/s", "128"])
        #expect(content.decisions.isEmpty)
    }

    @Test("an empty intent is an answer, not a failure")
    func emptyIntentIsAnAnswer() throws {
        let content = try DigestContent(
            lenientJSON: #"{"unresolvedIntent": "", "decisions": ["ship it"]}"#)
        #expect(content.unresolvedIntent == nil)
        #expect(content.decisions == ["ship it"])
    }

    /// Braces inside a string must not end the object early - the scanner tracks string literals
    /// and their escapes.
    @Test("survives braces and escaped quotes inside values")
    func bracesInsideStrings() throws {
        let content = try DigestContent(
            lenientJSON: """
                {"establishedFacts": ["the template emits {%- if tools %}",
                                      "the flag is called \\"--digest-dir\\""]}
                """)
        #expect(content.establishedFacts.count == 2)
        #expect(content.establishedFacts[0] == "the template emits {%- if tools %}")
        #expect(content.establishedFacts[1] == "the flag is called \"--digest-dir\"")
    }

    /// The wire form this library WRITES has to be readable by the path that reads a model's
    /// output, or the two have drifted into different formats without anyone noticing.
    @Test("reads back our own encoded digest")
    func readsOwnWireForm() throws {
        let original = DigestContent(
            unresolvedIntent: "keep the fleet notes current",
            establishedFacts: ["the M5 Air throttles when fanless"],
            decisions: ["bench with the lid open"],
            toolEvents: [
                ToolEvent(tool: "bench", whatFor: "measure decode", materialResult: "43 tok/s")
            ],
            openThreads: ["the 128 GB machine is unmeasured"],
            userPreferences: ["ASCII only"])
        let digest = SessionDigest(
            content: original, sourceTurnCount: 4, sourceSHA256: "abc", generator: "test",
            createdAt: Date(timeIntervalSince1970: 1_700_000_000))
        let data = try JSONSerialization.data(withJSONObject: digest.jsonObject)
        let text = try #require(String(data: data, encoding: .utf8))
        #expect(try DigestContent(lenientJSON: text) == original)
    }

    // MARK: - The echo, which is the dangerous one

    /// A model restating the format before answering emits a schema-shaped object whose values are
    /// the prompt's placeholders. Accepting it would report `condensed: true` while priming the
    /// instructions as if they were the conversation - non-empty, so nothing downstream catches it.
    /// Against the REAL template the prompt ships, not a copy of it - that is why the template
    /// lives in this library rather than beside the prompt.
    @Test("the prompt's own template is not a digest")
    func templateIsNotADigest() {
        #expect(throws: DigestDecodeError.emptyDigest) {
            _ = try DigestContent(lenientJSON: DigestContent.jsonTemplate)
        }
        // Fenced and with a preamble, which is how a model actually restates it.
        #expect(throws: DigestDecodeError.emptyDigest) {
            _ = try DigestContent(
                lenientJSON: "Sure - the format is:\n```json\n\(DigestContent.jsonTemplate)\n```")
        }
    }

    /// The template must still be a legal object: a model asked to fill it in has to be able to
    /// parse it, and a malformed example is worse than none.
    @Test("the template is valid JSON with every field")
    func templateIsWellFormed() throws {
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(DigestContent.jsonTemplate.utf8))
                as? [String: Any])
        #expect(
            Set(object.keys) == [
                "unresolvedIntent", "establishedFacts", "decisions", "toolEvents", "openThreads",
                "userPreferences",
            ])
    }

    @Test("an echo followed by the real answer resolves to the answer")
    func echoThenAnswer() throws {
        let reply = """
            Understood. I will fill in this shape:
            {"unresolvedIntent": "<one sentence>", "establishedFacts": ["<a fact>"]}

            Here are the notes:
            \(Self.clean)
            """
        let content = try DigestContent(lenientJSON: reply)
        #expect(content.unresolvedIntent == "get notarization working")
        #expect(content.establishedFacts.count == 2)
    }

    /// A model that answers, notices a mistake, and answers again. The correction is last.
    @Test("the last digest-shaped object wins")
    func lastObjectWins() throws {
        let reply = """
            {"unresolvedIntent": "first attempt", "decisions": ["wrong"]}
            Wait - I misread the section. Corrected:
            {"unresolvedIntent": "second attempt", "decisions": ["right"]}
            """
        let content = try DigestContent(lenientJSON: reply)
        #expect(content.unresolvedIntent == "second attempt")
        #expect(content.decisions == ["right"])
    }

    /// The exact case BackendDigest's own comment claims is handled: a reasoning preamble that
    /// opens a brace and abandons it.
    @Test("an unclosed brace in a preamble does not hide the answer")
    func abandonedDraftBeforeTheAnswer() throws {
        let reply = """
            <think>Let me draft: {"unresolvedIntent": "x", "establishedFacts": [ ... no, I should
            start from the section itself.</think>
            \(Self.clean)
            """
        let content = try DigestContent(lenientJSON: reply)
        #expect(content.unresolvedIntent == "get notarization working")
    }

    @Test("a stray open brace in prose does not hide the answer")
    func strayBrace() throws {
        let content = try DigestContent(lenientJSON: "Notes for the section {\n\(Self.clean)")
        #expect(content.decisions == ["ship a .pkg, not a .dmg"])
    }

    /// Bounded so a pathological reply cannot go quadratic. Past the bound it gives up, which is
    /// the safe direction - the caller primes the full history.
    @Test("re-scanning after unclosed braces is bounded")
    func rescanIsBounded() {
        let noise = String(repeating: "{ draft ", count: 40_000)
        let started = Date()
        _ = try? DigestContent(lenientJSON: noise + Self.clean)
        #expect(Date().timeIntervalSince(started) < 2.0)
    }

    // MARK: - What must throw

    @Test("refuses a truncated object rather than closing it")
    func truncated() {
        let reply = """
            {
              "unresolvedIntent": "get notarization working",
              "establishedFacts": ["the binary is arm64-only", "notarytool needs an app
            """
        #expect(throws: DigestDecodeError.truncatedJSON) {
            _ = try DigestContent(lenientJSON: reply)
        }
    }

    @Test("refuses prose with no object at all")
    func noObject() {
        #expect(throws: DigestDecodeError.noJSONObject) {
            _ = try DigestContent(
                lenientJSON: "The conversation was about notarization and code signing.")
        }
    }

    /// The dangerous shape: a sibling key parses, so emptying the unreadable one would leave a
    /// non-empty digest that has quietly lost the facts it was supposed to carry.
    @Test("refuses a list that arrived as an object")
    func objectWhereAListBelongs() {
        #expect(throws: DigestDecodeError.noRecognizedFields) {
            _ = try DigestContent(
                lenientJSON: """
                    {"decisions": ["ship a .pkg"],
                     "establishedFacts": {"0": "arm64 only", "1": "needs an app password"}}
                    """)
        }
    }

    @Test("refuses a list holding objects or nested lists")
    func unreadableListElements() {
        #expect(throws: DigestDecodeError.noRecognizedFields) {
            _ = try DigestContent(
                lenientJSON: #"{"establishedFacts": [["a", "b"], "kept"], "decisions": ["x"]}"#)
        }
        #expect(throws: DigestDecodeError.noRecognizedFields) {
            _ = try DigestContent(
                lenientJSON: #"{"establishedFacts": [{"fact": "arm64 only"}], "decisions": ["x"]}"#)
        }
    }

    @Test("refuses an unresolvedIntent that is not a string")
    func unreadableIntent() {
        #expect(throws: DigestDecodeError.noRecognizedFields) {
            _ = try DigestContent(
                lenientJSON: #"{"unresolvedIntent": {"text": "ship it"}, "decisions": ["x"]}"#)
        }
        // A number is readable but never meaningful as the preamble's intent line.
        #expect(throws: DigestDecodeError.noRecognizedFields) {
            _ = try DigestContent(lenientJSON: #"{"unresolvedIntent": 0, "decisions": ["x"]}"#)
        }
    }

    @Test("refuses tool events that are neither objects nor names")
    func unreadableToolEvents() {
        #expect(throws: DigestDecodeError.noRecognizedFields) {
            _ = try DigestContent(
                lenientJSON: #"{"toolEvents": [["grep", "why"]], "decisions": ["x"]}"#)
        }
    }

    /// A raw newline inside a string is the most ordinary JSON malformation a model produces.
    @Test("refuses an unescaped newline inside a value")
    func rawNewlineInString() throws {
        let error = try #require(
            captureError {
                _ = try DigestContent(
                    lenientJSON: "{\"decisions\": [\"ship the\nrelease\"]}")
            } as? DigestDecodeError)
        guard case .malformedJSON = error else {
            Issue.record("expected .malformedJSON, got \(error)")
            return
        }
    }

    /// Two keys that normalize to the same field must resolve the SAME WAY every run. Swift's
    /// dictionary order is seeded per process, so "whichever comes out of the dictionary last"
    /// varied between runs - which quietly undoes the determinism greedy sampling is chosen for.
    /// The rule is first-wins over sorted key order, so this is checkable rather than merely
    /// stable-looking: "establishedFacts" sorts before "established_facts" ('F' < '_').
    @Test("colliding keys resolve deterministically")
    func collidingKeys() throws {
        let content = try DigestContent(
            lenientJSON: """
                {"toolEvents": [{"tool": "A"}], "tool_events": [{"tool": "B"}],
                 "established_facts": ["f1"], "establishedFacts": ["f2"]}
                """)
        #expect(content.establishedFacts == ["f2"])
        #expect(content.toolEvents.map(\.tool) == ["A"])
    }

    @Test("refuses an object that answers a different question")
    func foreignObject() {
        #expect(throws: DigestDecodeError.noRecognizedFields) {
            _ = try DigestContent(lenientJSON: #"{"summary": "they talked about notarization"}"#)
        }
    }

    @Test("refuses an empty object")
    func emptyObject() {
        #expect(throws: DigestDecodeError.noRecognizedFields) {
            _ = try DigestContent(lenientJSON: "{}")
        }
    }

    @Test("refuses a balanced object that is not valid JSON")
    func malformed() throws {
        let error = try #require(
            captureError { _ = try DigestContent(lenientJSON: #"{"decisions": ['a', 'b',]}"#) }
                as? DigestDecodeError)
        guard case .malformedJSON = error else {
            Issue.record("expected .malformedJSON, got \(error)")
            return
        }
    }

    /// The reason reaches the client verbatim through `DigestPlanner.condense`, so it has to read
    /// as an explanation rather than as a type name.
    @Test("every failure carries a readable reason")
    func readableReasons() {
        let errors: [DigestDecodeError] = [
            .noJSONObject, .truncatedJSON, .malformedJSON("unexpected token"), .noRecognizedFields,
            .emptyDigest,
        ]
        for error in errors {
            let text = error.errorDescription ?? ""
            #expect(text.count > 20)
            #expect(text.first?.isUppercase == false)
        }
    }

    private func captureError(_ body: () throws -> Void) -> Error? {
        do {
            try body()
            return nil
        } catch {
            return error
        }
    }
}
