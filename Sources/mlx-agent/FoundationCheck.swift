// FoundationCheck.swift - `mlx-agent fm-check`: can this machine use the on-device system model?
//
// A liveness probe for Apple Foundation Models, in the same spirit as `--version`: it answers
// before anything expensive can fail, so an embedding app can ask the question at install or
// launch time and decide what to offer. Cadabra and Interpreter both pick engines from what
// the machine can actually do, and until now there was no way to ask about this one.
//
// It reports two different things, and the distinction matters to a caller:
//   availability - a settings/hardware/asset fact, obtained without inference (FMAvailability)
//   a live pass  - proof that a generation actually completes, which availability alone does
//                  not guarantee (assets can report ready and still fail on first use)
//
// The human-readable block is for a person at a terminal; the RESULT_JSON line is for scripts,
// following the convention `bench` already established. Exit code is 0 only when a real
// generation completed, so a shell can branch on `if mlx-agent fm-check >/dev/null; then`.
//
// It is a GATE, not a proof - see the doc comment on runFoundationCheck for what it deliberately
// does not check, and what to run after it when a caller wants that.

import Foundation

#if canImport(FoundationModels)
    import FoundationModels
#endif

/// Probe Foundation Models availability and, when available, run one small generation.
///
/// DELIBERATELY SHALLOW, and it is a gate rather than a proof. It answers "should this machine be
/// able to use the on-device model", which is a settings, hardware and asset question plus one
/// cheap generation. It does NOT prove that GUIDED generation works - the schema, the permissive
/// guardrails and the `.general` use case together - which is the combination that actually breaks
/// and the one this project depends on for summarizing.
///
/// That deeper check used to live here as `--digest <text>`, and it did not belong: it made this
/// command a second, Foundation-Models-only way to produce a digest, at a moment when the whole
/// point of `mlx-agent digest` was that summarizing is a capability of the AGENT. The two compose
/// instead, so a caller pays for the expensive half only when it wants it:
///
///     mlx-agent fm-check >/dev/null \
///       && mlx-agent digest --backend foundation --in probe.json --keep-recent 2
///
/// The `--keep-recent 2` is not decoration: the default is 6, and the planner summarizes only what
/// is OLDER than the verbatim tail, so a small probe under the default has nothing older and exits
/// 3 without ever calling the model - which reads exactly like a failure. See
/// docs/foundation-models.md for the probe transcript to use and the shapes that refuse.
///
/// Returns a process exit code: 0 usable, 1 not.
func runFoundationCheck(prompt: String?) async -> Int {
    let status = FMAvailability.probe()

    print("Apple Foundation Models")
    print("  availability:   \(status.summary)")

    guard status.isAvailable else {
        // Not an error in the "something broke" sense - it is the answer to the question.
        // Still exit non-zero so scripts can branch without parsing anything.
        emitResultJSON(available: false, detail: status.summary, generated: nil, seconds: nil)
        return 1
    }

    #if canImport(FoundationModels)
        guard #available(macOS 26.0, *) else {
            // Unreachable: probe() already returned .osTooOld in this case. Kept so the
            // availability narrowing is explicit rather than implied by control flow.
            emitResultJSON(available: false, detail: "macOS too old", generated: nil, seconds: nil)
            return 1
        }

        let model = SystemLanguageModel.default
        print("  context size:   \(model.contextSize) tokens")
        print("  current locale: \(model.supportsLocale() ? "supported" : "NOT supported")")

        // The list, not just the count. Which languages are in it decides whether this model
        // can be used for anything multilingual at all, and the set is smaller than people
        // assume - a prompt in an absent language fails outright with unsupportedLanguageOrLocale
        // rather than degrading, so the answer has to be checkable before a feature is designed
        // around it.
        //
        // The count and the code list measure different things, which is why both are printed.
        // `supportedLanguages` holds LOCALES (en-AU, en-GB, en-US; zh-Hans-CN, zh-Hant-HK,
        // zh-Hant-TW), so reducing it to primary language codes and deduping gives a
        // materially smaller number than the count above.
        let codes = Set(model.supportedLanguages.compactMap { $0.languageCode?.identifier })
            .sorted()
        print("  locales:        \(model.supportedLanguages.count) supported")
        print("  languages:      \(codes.count) distinct")
        print("                  \(codes.joined(separator: " "))")

        // One real pass. Greedy so a repeat run of this check reads the same, which is what
        // makes it usable as a smoke test rather than a curiosity.
        let text =
            prompt ?? "Reply with exactly one short sentence confirming you are running on device."
        let session = LanguageModelSession()
        let started = Date()
        do {
            let response = try await session.respond(
                to: text,
                options: GenerationOptions(sampling: .greedy, maximumResponseTokens: 128))
            let elapsed = Date().timeIntervalSince(started)
            print("  generation:     ok in \(String(format: "%.2f", elapsed)) s")
            print("")
            print(response.content)
            emitResultJSON(
                available: true, detail: "ok", generated: response.content, seconds: elapsed)
            return 0
        } catch {
            // Availability said yes and generation still failed. This is exactly the case the
            // check exists to surface, so report the mapped message rather than the raw error.
            let elapsed = Date().timeIntervalSince(started)
            let message = fmUserFacingError(error)
            print("  generation:     FAILED after \(String(format: "%.2f", elapsed)) s")
            FileHandle.standardError.write(Data("fm-check: \(message)\n".utf8))
            emitResultJSON(available: false, detail: message, generated: nil, seconds: elapsed)
            return 1
        }
    #else
        emitResultJSON(
            available: false, detail: "not built with FoundationModels", generated: nil,
            seconds: nil)
        return 1
    #endif
}

/// Machine-readable one-liner, mirroring `bench`'s RESULT_JSON convention so the scripts that
/// already scrape one output shape do not need a second one.
private func emitResultJSON(
    available: Bool, detail: String, generated: String?, seconds: TimeInterval?
) {
    var payload: [String: Any] = ["available": available, "detail": detail]
    if let generated { payload["generated"] = generated }
    // Integer milliseconds, not a rounded Double: JSONSerialization renders a Double by its
    // shortest round-trip representation, so 0.776 comes back out as 0.77600000000000002.
    if let seconds { payload["ms"] = Int((seconds * 1000).rounded()) }
    guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
    else { return }
    print("RESULT_JSON: " + String(decoding: data, as: UTF8.self))
}
