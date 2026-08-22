// TurnErrorText.swift - a failed turn, in words the person waiting for it can act on.
//
// Pure string logic with no MLX dependency, in this target for the same reason ThinkSplitter is:
// it can then be unit-tested without compiling MLX or the Metal toolchain. It lived in
// ACPServer.swift, which links MLX, and so went untested - which is exactly the history that file
// header records about ThinkSplitter, repeated.
//
// WHAT THIS IS FOR. A generation failure arrives as whatever the engine said, and two of those
// read like the app is broken when they are not:
//
//   "upper filter requires string"     - a chat template our Jinja engine cannot render
//   {"error":{"code":400,...}}         - a prompt larger than llama-server's context
//
// Both have a specific thing the user can do, and neither says what it is. Everything else is
// passed through with a prefix: inventing a friendly sentence for an error nobody has classified
// is how a real cause becomes unreportable. The raw text is kept in every case - it is what a bug
// report needs - and logged separately at the call site.

import Foundation

public enum TurnErrorText {

    /// Signals that a chat template, rather than the model, is what failed. The failing macros
    /// only run when tool schemas are rendered, which is why turning tools off is the way out.
    private static let templateSignals = [
        "filter requires", "no filter named", "unknown filter", "unknown tag",
        "jinja", "chat template", "template render", "template error",
    ]

    /// Signals that the prompt did not fit. llama-server answers with a JSON body carrying both
    /// numbers; the type name is matched as well as the prose because the prose has changed
    /// between llama.cpp releases and the type has not.
    private static let overflowSignals = [
        "exceed_context_size_error", "exceeds the available context size",
    ]

    /// Turn a raw generation-failure string into something a user can act on.
    public static func userFacing(_ raw: String) -> String {
        let lower = raw.lowercased()

        if templateSignals.contains(where: { lower.contains($0) }) {
            return
                "This model's chat template could not render the request, so it can't run here. "
                + "This usually affects tool calling - start a new chat with tools turned off, or "
                + "pick a different model. (details: \(raw))"
        }

        // Nothing can fix this one after the fact, which is why the message is entirely about what
        // to do next: the context was fixed when the server started, and the prompt is what the
        // user asked for. The two numbers ARE the explanation, so they are lifted out of the JSON
        // and put in the sentence rather than left in the body for the reader to find.
        if overflowSignals.contains(where: { lower.contains($0) }) {
            let needed = firstInteger(in: raw, after: "\"n_prompt_tokens\":")
            let window = firstInteger(in: raw, after: "\"n_ctx\":")
            // Three shapes rather than two, because a body carrying only the window would
            // otherwise read "this conversation against a 8192-token context". Truncation puts
            // the keys in the other order so that combination should not arise - which is a
            // reason to spell it correctly, not a reason to leave it wrong.
            let sizes: String
            switch (needed, window) {
            case let (needed?, window?): sizes = "\(needed) tokens against a \(window)-token context"
            case let (needed?, nil): sizes = "\(needed) tokens"
            case let (nil, window?): sizes = "the context is \(window) tokens"
            case (nil, nil): sizes = "this conversation is too long for it"
            }
            return
                "This conversation no longer fits the model's context (\(sizes)). Summarize it "
                + "with the On resume setting, start a new chat, or load a model that leaves more "
                + "room for context. (details: \(raw))"
        }

        return "generation failed: \(raw)"
    }

    /// The first run of digits following `key`, or nil.
    ///
    /// String scanning rather than a JSON decode on purpose: this runs on an error path where the
    /// body has already been truncated to 1000 characters by the backend, so it is frequently not
    /// parseable JSON at all - and a rewrite that only worked on short errors would be worse than
    /// one that reads the numbers out of whatever arrived.
    static func firstInteger(in text: String, after key: String) -> Int? {
        guard let start = text.range(of: key)?.upperBound else { return nil }
        let digits = text[start...].drop(while: { $0 == " " }).prefix(while: { $0.isNumber })
        return digits.isEmpty ? nil : Int(digits)
    }
}
