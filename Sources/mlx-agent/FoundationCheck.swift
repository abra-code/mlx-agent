// FoundationCheck.swift - `mlx-agent fm-check`: can this machine use the on-device system model?
//
// A liveness probe for Apple Foundation Models, in the same spirit as `--version`: it answers
// before anything expensive can fail, so an embedding app can ask the question at install or
// launch time and decide what to offer. Cadabra and Interpreter both pick engines from what
// the machine can actually do, and until now there was no way to ask about this one.
//
// It reports two different things, and the distinction matters enough that `--test-prompt`
// selects between them rather than merely decorating one:
//   availability - a settings/hardware/asset fact, obtained without inference (FMAvailability).
//                  This is `fm-check` with no arguments, and it is cheap enough (~0.05 s) to sit
//                  on a UI path that asks it every time a menu opens.
//   a live pass  - proof that a generation actually completes, which availability alone does
//                  not guarantee (assets can report ready and still fail on first use). This is
//                  `fm-check --test-prompt ...`, and it costs a real generation because it IS one.
//
// The default is the cheap one because that is the question with an impatient caller. Nothing is
// lost by making the expensive one explicit: a caller that wants proof is, by definition, willing
// to wait for it.
//
// The human-readable block is for a person at a terminal; the RESULT_JSON line is for scripts,
// following the convention `bench` already established. Exit 0 means "usable", and what that
// word is worth follows the mode: without `--test-prompt` it means availability said yes, with it
// that a generation completed. `if mlx-agent fm-check >/dev/null; then` is still a valid gate.
//
// Even with a live pass it is a GATE, not a proof - see the doc comment on runFoundationCheck
// for what it deliberately does not check, and what to run after it when a caller wants that.

import Foundation

#if canImport(FoundationModels)
    import FoundationModels
#endif

/// How deep the check actually went, reported as `probe` in the RESULT_JSON line.
///
/// Deliberately separate from `FMAvailability.Code`: the code says WHAT the answer is, this says
/// what the answer is WORTH. `available` is emitted in both modes, and only this distinguishes
/// "the framework says it is ready" from "it generated something in front of me".
///
/// Sorts before `reason` alphabetically, which keeps the greedy `"reason":"..."` scrape in the
/// shell consumers correct. See `emitResultJSON`.
private enum ProbeDepth: String {
    /// No inference. Settings, hardware and assets only.
    case availability
    /// A real generation was attempted; `ms` says how long it took.
    case generation
}

/// Probe Foundation Models availability and, with `--test-prompt`, run one small generation.
///
/// DELIBERATELY SHALLOW, and it is a gate rather than a proof even in its stronger mode. It
/// answers "should this machine be able to use the on-device model", which is a settings, hardware
/// and asset question, optionally plus one cheap generation. It does NOT prove that GUIDED
/// generation works - the schema, the permissive guardrails and the `.general` use case together -
/// which is the combination that actually breaks and the one this project depends on for
/// summarizing.
///
/// That deeper check does not belong here: it would make this command a second,
/// Foundation-Models-only way to produce a digest, when the whole point of `mlx-agent digest` is
/// that summarizing is a capability of the AGENT. The two compose instead, so a caller pays for
/// the expensive half only when it wants it:
///
///     mlx-agent fm-check >/dev/null \
///       && mlx-agent digest --backend foundation --in probe.json --keep-recent 2
///
/// The `--keep-recent 2` is not decoration: the default is 6, and the planner summarizes only what
/// is OLDER than the verbatim tail, so a small probe under the default has nothing older and exits
/// 3 without ever calling the model - which reads exactly like a failure. See
/// docs/foundation-models.md for the probe transcript to use and the shapes that refuse.
///
/// `livePass` decides whether anything is GENERATED, and the caller derives it from whether
/// `--test-prompt` was given at all. Without it this answers the settings/hardware/asset question
/// and stops - which is the cheap question, and the one a UI actually asks: whether to OFFER this
/// engine is decided every time a model picker opens, and a real generation there proves nothing
/// while costing ~6x the time (~0.3 s against 0.05 s, measured).
///
/// The live pass is the STRONGER answer and is why this command exists at all: availability can
/// report ready and the first generation still fail, which nothing but generating can discover.
/// So the two modes are not interchangeable - `fm-check` gates a menu, `fm-check --test-prompt ...`
/// proves a pass - and the exit code means correspondingly different things (see below).
///
/// Returns a process exit code: 0 usable, 1 not. Without a live pass "usable" means availability
/// said yes; with one it means a generation actually completed.
func runFoundationCheck(prompt: String?, livePass: Bool) async -> Int {
    let status = FMAvailability.probe()

    print("Apple Foundation Models")
    print("  availability:   \(status.summary)")

    guard status.isAvailable else {
        // Not an error in the "something broke" sense - it is the answer to the question.
        // Still exit non-zero so scripts can branch without parsing anything.
        emitResultJSON(
            available: false, reason: status.code, probe: .availability, detail: status.summary,
            generated: nil, seconds: nil)
        return 1
    }

    #if canImport(FoundationModels)
        guard #available(macOS 26.0, *) else {
            // Unreachable: probe() already returned .osTooOld in this case. Kept so the
            // availability narrowing is explicit rather than implied by control flow.
            emitResultJSON(
                available: false, reason: .osTooOld, probe: .availability, detail: "macOS too old",
                generated: nil, seconds: nil)
            return 1
        }

        let model = SystemLanguageModel.default
        print("  context size:   \(model.contextSize) tokens")
        print("  current locale: \(model.supportsLocale() ? "supported" : "NOT supported")")

        // The list, not just the count. Which languages are in it decides whether this model can
        // be used for anything multilingual at all, and the set is smaller than people assume.
        //
        // This has to be CHECKABLE before a feature is designed around it, because the failure
        // mode is silent. An absent language does not raise unsupportedLanguageOrLocale - measured
        // with Polish, which raised it never. It degrades instead: answering in English, or
        // answering in the requested language with confident nonsense, or dying mid-stream with
        // guardrailViolation, which reads as a safety refusal for an entirely benign prompt. And
        // the model will happily claim it speaks the language if asked. This list is the only
        // honest answer. See docs/foundation-models.md.
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

        // Everything above is a cheap READ - no inference - so the no-generation mode stops
        // here rather than earlier. Returning before the block would have made `fm-check`
        // report availability and nothing else, dropping the window size and the language
        // list, which are the two facts most likely to decide whether this model is usable
        // for a given job and cost nothing to obtain.
        guard livePass else {
            // No generation, so no `ms` and no `generated`. Their ABSENCE is how a caller tells
            // the two modes apart in the payload - there is no flag it would have to look for.
            emitResultJSON(
                available: true, reason: .available, probe: .availability,
                detail: "not generated (no --test-prompt)", generated: nil, seconds: nil)
            return 0
        }

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
                available: true, reason: .available, probe: .generation, detail: "ok",
                generated: response.content, seconds: elapsed)
            return 0
        } catch {
            // Availability said yes and generation still failed. This is exactly the case the
            // check exists to surface, so report the mapped message rather than the raw error.
            let elapsed = Date().timeIntervalSince(started)
            let message = fmUserFacingError(error)
            print("  generation:     FAILED after \(String(format: "%.2f", elapsed)) s")
            FileHandle.standardError.write(Data("fm-check: \(message)\n".utf8))
            emitResultJSON(
                available: false, reason: .generationFailed, probe: .generation, detail: message,
                generated: nil, seconds: elapsed)
            return 1
        }
    #else
        emitResultJSON(
            available: false, reason: .notBuilt, probe: .availability,
            detail: "not built with FoundationModels", generated: nil, seconds: nil)
        return 1
    #endif
}

/// Machine-readable one-liner, mirroring `bench`'s RESULT_JSON convention so the scripts that
/// already scrape one output shape do not need a second one.
///
/// `reason` is the field to BRANCH on and `detail` the one to show a person. A caller deciding
/// whether to offer this engine has to tell "never" (ineligible hardware, an older macOS) from
/// "one settings toggle away" from "still downloading", and `available: false` alone cannot -
/// while `detail` is prose that gets reworded. See `FMAvailability.Code`.
///
/// `probe` says WHICH QUESTION was answered, and it exists because `reason` alone cannot say.
/// `available` is emitted whether or not anything was generated, so a consumer holding only this
/// line - a log, a telemetry record, anything that did not choose the argv itself - would
/// otherwise read a bare availability check as proof of a completed pass. Absent `ms` and
/// `generated` keys hint at it, but inferring from key ABSENCE is exactly what the rest of this
/// contract tells callers not to do.
///
/// Key order is not decorative: `.sortedKeys` is what keeps `reason` the LAST `"reason":"..."`
/// on the line, which the shell consumers scrape with a greedy sed. Adding a key that sorts
/// after `reason` would let a crafted `detail` shadow it. Keep that in mind before extending
/// this payload.
private func emitResultJSON(
    available: Bool, reason: FMAvailability.Code, probe: ProbeDepth, detail: String,
    generated: String?, seconds: TimeInterval?
) {
    var payload: [String: Any] = [
        "available": available, "reason": reason.rawValue, "probe": probe.rawValue,
        "detail": detail,
    ]
    if let generated { payload["generated"] = generated }
    // Integer milliseconds, not a rounded Double: JSONSerialization renders a Double by its
    // shortest round-trip representation, so 0.776 comes back out as 0.77600000000000002.
    if let seconds { payload["ms"] = Int((seconds * 1000).rounded()) }
    guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
    else { return }
    print("RESULT_JSON: " + String(decoding: data, as: UTF8.self))
}
