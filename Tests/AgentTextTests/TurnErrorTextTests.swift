// TurnErrorTextTests.swift - the two failures that read like a crash, and the ones that must not
// be rewritten at all.
//
// The overflow case is what a real session produced: a conversation restored onto a llama-server
// whose context had been sized down to 8192 tokens came back as a raw JSON error object in the
// transcript. It names both numbers, and neither of them reached the reader.

import Foundation
import Testing

@testable import AgentText

@Suite("TurnErrorText")
struct TurnErrorTextTests {

    /// Verbatim from the failing session, including llama-server's own spelling and spacing.
    private let overflow = """
        ACP turn failed: generation failed: llama-server returned HTTP 400: \
        {"error":{"code":400,"message":"request (11036 tokens) exceeds the available context \
        size (8192 tokens), try increasing it","type":"exceed_context_size_error",\
        "n_prompt_tokens":11036,"n_ctx":8192}} (code -32000)
        """

    @Test("an overflow says what happened, with both numbers")
    func overflowNamesNumbers() {
        let text = TurnErrorText.userFacing(overflow)
        #expect(text.contains("11036 tokens"))
        #expect(text.contains("8192-token context"))
        #expect(text.contains("no longer fits"))
    }

    /// The raw body is what a bug report needs, and dropping it to make a friendly sentence would
    /// make the real cause unreportable.
    @Test("and keeps the raw text behind it")
    func overflowKeepsRaw() {
        #expect(TurnErrorText.userFacing(overflow).contains("exceed_context_size_error"))
    }

    @Test("it names things the user can actually do")
    func overflowIsActionable() {
        let text = TurnErrorText.userFacing(overflow)
        #expect(text.contains("Summarize"))
        #expect(text.contains("more room for context"))
    }

    /// llama.cpp has reworded this message across releases; the type name is the stable half.
    @Test("the type name alone is enough to recognize it", arguments: [
        "{\"type\":\"exceed_context_size_error\"}",
        "request (99 tokens) exceeds the available context size (10 tokens)",
    ])
    func overflowSignals(raw: String) {
        #expect(TurnErrorText.userFacing(raw).contains("no longer fits"))
    }

    /// The backend truncates the body to 1000 characters before it gets here, so the JSON is
    /// frequently cut off mid-object. A rewrite that needed valid JSON would fail exactly on the
    /// longest errors.
    @Test("a body cut off before its numbers still gets the sentence")
    func overflowTruncated() {
        let cut = "llama-server returned HTTP 400: {\"error\":{\"code\":400,\"type\":\"exceed_con"
        let text = TurnErrorText.userFacing(cut + "text_size_error\",\"n_prompt_tokens\":")
        #expect(text.contains("this conversation"))
        #expect(!text.contains("against a"))
    }

    @Test("a template failure still points at tools")
    func templateFailure() {
        let text = TurnErrorText.userFacing("generation failed: upper filter requires string")
        #expect(text.contains("chat template"))
        #expect(text.contains("tools turned off"))
    }

    /// Anything unclassified is passed through. Inventing a friendly sentence for an error nobody
    /// has looked at is how a real cause becomes unreportable.
    @Test("everything else is passed through untouched")
    func passthrough() {
        #expect(TurnErrorText.userFacing("connection reset") == "generation failed: connection reset")
        #expect(TurnErrorText.userFacing("") == "generation failed: ")
    }

    // MARK: - The number scan

    @Test("it reads the first integer after a key")
    func firstInteger() {
        #expect(TurnErrorText.firstInteger(in: "{\"n_ctx\":8192}", after: "\"n_ctx\":") == 8192)
        #expect(TurnErrorText.firstInteger(in: "{\"n_ctx\": 8192}", after: "\"n_ctx\":") == 8192)
    }

    @Test("a missing or empty value is nil, not zero")
    func firstIntegerMissing() {
        #expect(TurnErrorText.firstInteger(in: "{}", after: "\"n_ctx\":") == nil)
        #expect(TurnErrorText.firstInteger(in: "{\"n_ctx\":}", after: "\"n_ctx\":") == nil)
        #expect(TurnErrorText.firstInteger(in: "{\"n_ctx\":\"x\"}", after: "\"n_ctx\":") == nil)
    }

    /// `n_ctx` is a suffix of nothing here, but `n_prompt_tokens` and `n_ctx` sit next to each
    /// other in the real body and a sloppy scan would read one for the other.
    @Test("neighboring keys are not confused")
    func neighboringKeys() {
        let body = "\"n_prompt_tokens\":11036,\"n_ctx\":8192"
        #expect(TurnErrorText.firstInteger(in: body, after: "\"n_prompt_tokens\":") == 11036)
        #expect(TurnErrorText.firstInteger(in: body, after: "\"n_ctx\":") == 8192)
    }
}
