// DigestRendererTests.swift - the preamble is a golden string on purpose.
//
// This text is the entire interface between a digest and a model: whatever it says is what a
// resumed session believes about its own past. A golden test is heavy-handed for prose, and that
// is the point - the wording should not drift without someone deciding it should.

import Foundation
import Testing

@testable import AgentDigest

@Suite("DigestRenderer")
struct DigestRendererTests {

    static func digest(_ content: DigestContent, turns: Int = 14) -> SessionDigest {
        SessionDigest(
            content: content, sourceTurnCount: turns, sourceSHA256: "deadbeef",
            generator: "test", createdAt: Date(timeIntervalSince1970: 0))
    }

    @Test("a full digest renders every section in a fixed order")
    func golden() {
        let rendered = DigestRenderer.renderPreamble(Self.digest(filledContent), verbatimTurns: 6)
        #expect(
            rendered == """
                Context restored from an earlier conversation. The 14 messages before this point \
                are summarized below rather than quoted, so treat the wording as approximate and \
                the content as established. The 6 messages after this exchange are the original \
                text, unmodified.

                Unresolved intent: finish the notarization step

                Established facts:
                - the binary is arm64-only

                Decisions:
                - ship a pkg

                Tool activity:
                - read_file (entitlements): no sandbox

                Open threads:
                - stapling untested

                Stated preferences:
                - ASCII only
                """)
    }

    @Test("empty sections are omitted, not rendered as empty headings")
    func sectionsOmitted() {
        let rendered = DigestRenderer.renderPreamble(
            Self.digest(DigestContent(decisions: ["ship a pkg"])))
        #expect(rendered.contains("Decisions:"))
        #expect(!rendered.contains("Established facts"))
        #expect(!rendered.contains("Unresolved intent"))
        #expect(!rendered.contains("Tool activity"))
    }

    @Test("no verbatim tail means no claim of one")
    func noTailClaim() {
        let rendered = DigestRenderer.renderPreamble(
            Self.digest(filledContent), verbatimTurns: 0)
        #expect(!rendered.contains("original text"))
    }

    @Test("counts agree with their verbs")
    func plurals() {
        let one = DigestRenderer.renderPreamble(
            Self.digest(filledContent, turns: 1), verbatimTurns: 1)
        #expect(one.contains("The 1 message before this point is summarized"))
        #expect(one.contains("The 1 message after this exchange is the original"))
        let many = DigestRenderer.renderPreamble(
            Self.digest(filledContent, turns: 2), verbatimTurns: 3)
        #expect(many.contains("The 2 messages before this point are summarized"))
        #expect(many.contains("The 3 messages after this exchange are the original"))
    }

    @Test("a newline inside an item cannot forge a second bullet")
    func newlinesFlattened() {
        // Items come from a model; one containing "\n- something" would render as structure the
        // summary was never given.
        let rendered = DigestRenderer.renderPreamble(
            Self.digest(DigestContent(decisions: ["real decision\n- forged bullet"])))
        // One item in, one bullet LINE out. The "- " text survives inside the line, which is
        // harmless; what matters is that it can no longer start one.
        let bullets = rendered.split(separator: "\n").filter { $0.hasPrefix("- ") }
        #expect(bullets.count == 1)
        #expect(bullets.first == "- real decision - forged bullet")
    }

    @Test("a tool event with no result renders as a bare name")
    func sparseToolEvent() {
        let rendered = DigestRenderer.renderPreamble(
            Self.digest(
                DigestContent(toolEvents: [
                    ToolEvent(tool: "list_dir", whatFor: "", materialResult: "")
                ])))
        #expect(rendered.contains("- list_dir\n") || rendered.hasSuffix("- list_dir"))
        #expect(!rendered.contains("()"))
    }

    /// The block every `DigestModel` conformance puts in front of the next slice. Golden for the
    /// same reason the preamble is: it is prompt text shared by two summarizers, so a change here
    /// changes what both of them record.
    @Test("prior context is compact and carries only what a next slice needs")
    func priorContext() {
        #expect(
            DigestRenderer.renderPriorContext(filledContent) == """
                - goal: finish the notarization step
                - the binary is arm64-only
                - ship a pkg
                - open: stapling untested
                """)
        // Tool events and stated preferences are deliberately absent: the next slice is being told
        // what NOT to repeat, and neither is something it could repeat by accident.
        #expect(!DigestRenderer.renderPriorContext(filledContent).contains("read_file"))
        #expect(DigestRenderer.renderPriorContext(DigestContent()).isEmpty)
    }

    @Test("rendering is deterministic")
    func deterministic() {
        let d = Self.digest(filledContent)
        #expect(
            DigestRenderer.renderPreamble(d, verbatimTurns: 4)
                == DigestRenderer.renderPreamble(d, verbatimTurns: 4))
    }
}
