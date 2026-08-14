// DigestMode.swift - `mlx-agent digest`: summarize a saved transcript, offline.
//
// Everything the digest can do inside a session, done to a file instead. A transcript in the
// SAME wire shape `session/prime` takes goes in; the digest, or the preamble text it renders to,
// comes out.
//
// Two reasons this mode exists, and they are different reasons:
//
//   1. THE DIGEST BECOMES AN ARTIFACT. Inside a session a digest is a side effect - it is produced,
//      primed and forgotten, and the only way to see one was `--digest-dir` on a live agent. Here
//      it is the output. A conversation can be condensed once, stored, and primed into a session
//      later (a digest is portable by construction: `DigestRenderer` emits prose, not a format any
//      engine has to agree on), and two summarizers can be compared on the same input.
//   2. IT IS THE OFFLINE TEST SURFACE. Every other way to exercise the summarizer needs an ACP
//      session, a client, and a live model holding a busy slot. This needs a JSON file.
//
// WHAT IS DIFFERENT FROM THE PRIME PATH, and deliberately so:
//
//   - `--backend` names the SUMMARIZER, not a session engine. There is no session here, so there is
//     nothing for `--digest-backend`'s auto/session/none distinction to be about.
//   - No PassGate hazard. `BackendDigest.swift`'s two rules exist because a live session's backend
//     is mid-pass or shared; here the backend is built for this one job, in a process that does
//     nothing else, and is thrown away with the process. It is the one place a private backend is
//     the correct answer rather than the measured SIGTRAP.
//   - The bounds are batch bounds. A restore is something a user waits on, so the prime path stops
//     at 4 slices and 5 minutes; a script is not waiting, so this stops at 32 and half an hour.
//
// EXIT CODES. 0 condensed, 3 fell back, 2 usage or unreadable input, 1 something failed. 3 is
// separate from 1 because a fallback is not a crash: "this conversation was not worth condensing"
// and "the model refused" are answers, and a script that treats them as failures would be wrong.

import AgentDigest
import Foundation
import MLXLMCommon

/// What `mlx-agent digest` was asked to do. The engine, the generation knobs and the model
/// directory come from the shared CLI parsing, exactly as they do for oneshot.
struct DigestModeOptions: Sendable {
    /// nil reads stdin.
    var inputPath: String?
    /// nil writes stdout.
    var outputPath: String?
    /// Print the preamble the digest renders to, instead of the digest as JSON.
    var render: Bool
    var keepRecentTurns: Int?
    var maxDigestTokens: Int?
    /// `--digest-window`, for the llama-server case where the window cannot be read from disk.
    var windowOverride: Int?
}

/// Slices allowed for one batch condensation.
///
/// Eight times the prime path's, and the reasoning is inverted rather than relaxed: there the bound
/// is what fits a restore a user is waiting on, and here nothing is waiting. 32 slices is millions
/// of characters on a 32k-window model and about 250 KB on the on-device one - past any transcript
/// a client actually stores, which is the point. Beyond it the planner refuses and says how many it
/// would have needed, which is a better answer than a run that takes an hour.
private let digestModeMaxSlices = 32

/// Bound on the WHOLE condensation. A wedged model is the failure this catches - one that answers
/// every pass honestly but slowly is indistinguishable from a big job here, so the bound is
/// generous. Without it a script hangs forever, which is the one outcome worse than exit 3.
private let digestModeDeadline: TimeInterval = 1800

/// Bound on ONE pass on a local model. Longer than the prime path's 120 s for the same reason the
/// slice count is higher: a slice can be four times the size here, and nothing is holding a session
/// busy while it runs.
private let digestModeSliceTimeout: TimeInterval = 300

/// Bound on one on-device pass. The measured worst case is 3-5 s and guided generation has been
/// seen to run away (see `FMDigestGenerator.timeout`); 60 s is a wedge detector, not a budget.
private let digestModeFoundationTimeout: TimeInterval = 60

func digestModeLog(_ message: String) {
    FileHandle.standardError.write(Data("[digest] \(message)\n".utf8))
}

/// Summarize a transcript and write the result out. Returns the process exit code.
func runDigestMode(
    engine: EngineSpec, options: DigestModeOptions, gen: GenConfig, extraEOSTokens: Set<String>
) async throws -> Int {
    guard let raw = loadTranscript(options.inputPath) else { return 2 }

    // The SAME filter a condensing `session/prime` applies, by calling the same function rather
    // than by resembling it: empty turns, orphan tool messages, unknown roles and a trailing
    // unanswered tool-call announcement are dropped here exactly as they would be there. A digest
    // made offline and one made in a session must be made from the same text, or this mode cannot
    // be used to predict what a session will do.
    let history = ACPServer.primeHistory(from: raw) { digestModeLog($0) }
    let turns = digestTurns(from: history)
    if turns.count != raw.count {
        digestModeLog("\(raw.count) messages supplied, \(turns.count) accepted")
    }

    let deadline = Date().addingTimeInterval(digestModeDeadline)
    let summarizer: Summarizer
    switch try await buildSummarizer(
        engine: engine, options: options, gen: gen, extraEOSTokens: extraEOSTokens,
        deadline: deadline)
    {
    case .refused(let reason):
        FileHandle.standardError.write(Data("mlx-agent digest: \(reason)\n".utf8))
        return 2
    case .ready(let built):
        summarizer = built
    }
    // The sizing, stated. It is derived from the model's window rather than asked for, so when a
    // run refuses on slice count this line is what says whether the window was read correctly.
    digestModeLog(
        "summarizing with \(summarizer.name): up to \(summarizer.policy.maxSlices) slices of "
            + "~\(summarizer.policy.sliceBudgetTokens) tokens, keeping "
            + "\(summarizer.policy.keepRecentTurns) messages verbatim")

    let started = Date()
    let result = await DigestPlanner.condense(
        history: turns, using: summarizer.model, policy: summarizer.policy,
        generator: summarizer.name)
    let elapsed = Date().timeIntervalSince(started)

    guard let digest = result.digest else {
        // NOT an error. The planner's whole contract is that anything doubtful resolves to "prime
        // the full history", and offline that is "produce no digest and say why" - the caller still
        // has the transcript it handed in.
        let reason = result.reason ?? "the summarizer produced nothing"
        FileHandle.standardError.write(Data("mlx-agent digest: not condensed - \(reason)\n".utf8))
        if !options.render {
            guard
                writeOutput(
                    jsonData([
                        "condensed": false,
                        "reason": reason,
                        "accepted": turns.count,
                        "primed": result.history.count,
                    ]), to: options.outputPath)
            else { return 1 }
        }
        emitDigestResultJSON(
            condensed: false, summarizer: summarizer.name, accepted: turns.count,
            primed: result.history.count, dropped: nil, reason: reason, seconds: elapsed)
        return 3
    }

    // How many original messages survive verbatim after the digest - what `renderPreamble` states
    // in its opening, and what a caller checks the arithmetic against.
    let verbatim = turns.count - result.tailStartIndex
    let payload: Data
    if options.render {
        payload = Data(DigestRenderer.renderPreamble(digest, verbatimTurns: verbatim).utf8)
    } else {
        payload = jsonData([
            "condensed": true,
            "summarizer": summarizer.name,
            // Both counts are relative to what survived `primeHistory`, not to what was supplied:
            // `primed == injected + (accepted - dropped.turns)` closes only against `accepted`.
            // Same contract, same trap, as the session/prime response.
            "accepted": turns.count,
            "primed": result.history.count,
            "dropped": ["turns": result.droppedTurns, "bytes": result.droppedBytes],
            "digest": digest.jsonObject,
        ])
    }
    guard writeOutput(payload, to: options.outputPath) else { return 1 }
    emitDigestResultJSON(
        condensed: true, summarizer: summarizer.name, accepted: turns.count,
        primed: result.history.count,
        dropped: (result.droppedTurns, result.droppedBytes), reason: nil, seconds: elapsed)
    return 0
}

// MARK: - The summarizer

/// The model that will do the summarizing, what to record as its name, and how much conversation
/// it can take at once.
private struct Summarizer {
    let model: DigestModel
    let name: String
    let policy: PrimePolicy
}

private enum SummarizerBuild {
    case ready(Summarizer)
    case refused(String)
}

/// Build the summarizer for `engine`, loading whatever it needs.
///
/// The policy is derived BEFORE the model is loaded. Sizing needs only the model directory and the
/// generation knobs, and a run that is going to refuse should refuse before it spends a minute
/// mapping several gigabytes of weights.
private func buildSummarizer(
    engine: EngineSpec, options: DigestModeOptions, gen: GenConfig, extraEOSTokens: Set<String>,
    deadline: Date
) async throws -> SummarizerBuild {

    func backendPolicy(modelDir: String) -> DigestSizing.Outcome {
        DigestSizing.backendPolicy(
            window: DigestSizing.window(override: options.windowOverride, modelDir: modelDir),
            generationCeiling: DigestSizing.generationCeiling(gen),
            maxSlices: digestModeMaxSlices,
            keepRecentTurns: options.keepRecentTurns, maxDigestTokens: options.maxDigestTokens)
    }

    // Greedy by default. A digest is an artifact here, and two runs over the same transcript
    // reading the same is most of what makes it one; `--temperature` still overrides.
    let parameters = gen.apply(to: GenerateParameters(maxTokens: 4096, temperature: 0))

    switch engine {
    case .foundation:
        let policy = FoundationBackend.defaultDigestPolicy(maxSlices: digestModeMaxSlices)
            .with(
                keepRecentTurns: options.keepRecentTurns,
                maxDigestTokens: options.maxDigestTokens)
        guard let model = foundationDigestGenerator(policy: policy, deadline: deadline) else {
            // `resolveEngine` already refused an unavailable on-device model with an actionable
            // message, so reaching here means the framework is absent from the build entirely.
            return .refused("this build has no Foundation Models support")
        }
        return .ready(Summarizer(model: model, name: foundationSummarizerName, policy: policy))

    case .mlx(let dir):
        let policy: PrimePolicy
        switch backendPolicy(modelDir: dir) {
        case .refused(let reason): return .refused(reason)
        case .sized(let sized): policy = sized
        }
        let container = try await loadModel(dir, extraEOSTokens: extraEOSTokens)
        // No system prompt: `BackendDigestGenerator` carries its own instructions in every prompt,
        // and a "you are a helpful assistant" turn would spend window on contradicting them.
        let backend = MLXBackend(container: container, instructions: nil, parameters: parameters)
        return .ready(backendSummarizer(backend, engine: "mlx", policy: policy, deadline: deadline))

    case .openai(let baseURL):
        // Empty model directory: there is no local config.json to read a window out of, so this
        // lands on `DigestWindow.conservativeDefault` unless --digest-window says otherwise.
        let policy: PrimePolicy
        switch backendPolicy(modelDir: "") {
        case .refused(let reason): return .refused(reason)
        case .sized(let sized): policy = sized
        }
        try await OpenAIBackend.waitForHealth(baseURL: baseURL)
        let backend = OpenAIBackend(baseURL: baseURL, parameters: parameters, systemPrompt: nil)
        return .ready(
            backendSummarizer(backend, engine: "openai", policy: policy, deadline: deadline))
    }
}

/// `BackendDigestGenerator` over a backend built for this run alone. See the file header for why
/// that is safe here and nowhere else.
private func backendSummarizer(
    _ backend: GenerationBackend, engine: String, policy: PrimePolicy, deadline: Date
) -> Summarizer {
    let name = BackendDigestGenerator.name(engine: engine)
    return Summarizer(
        model: BackendDigestGenerator(
            backend: backend, name: name, policy: policy, timeout: digestModeSliceTimeout,
            deadline: deadline, log: { digestModeLog($0) }),
        name: name, policy: policy)
}

/// The on-device generator, behind the availability guards that cannot be expressed at a call site
/// which also has to compile without the framework. Mirrors `ACPServer.foundationGenerator`.
private func foundationDigestGenerator(policy: PrimePolicy, deadline: Date) -> DigestModel? {
    #if canImport(FoundationModels)
        guard #available(macOS 26.0, *) else { return nil }
        return FMDigestGenerator(
            policy: policy, timeout: digestModeFoundationTimeout, deadline: deadline,
            log: { digestModeLog($0) })
    #else
        return nil
    #endif
}

// MARK: - Input

/// Read the transcript from `path`, or from stdin when it is nil.
///
/// Accepts either a bare array of messages or the whole `session/prime` params object - a captured
/// request pastes straight in. Only `messages` is read from the object form: `condense` is
/// expressed here as flags, and honoring both would give the same knob two sources.
///
/// nil on any failure, having already said why. Every case is a usage error rather than an
/// exception because the caller is a person or a script, and "expected an array of {role, content}"
/// is more useful than a decoding error.
private func loadTranscript(_ path: String?) -> [[String: Any]]? {
    let data: Data
    if let path {
        do {
            data = try Data(contentsOf: URL(fileURLWithPath: (path as NSString).expandingTildeInPath))
        } catch {
            FileHandle.standardError.write(
                Data("mlx-agent digest: cannot read \(path): \(error.localizedDescription)\n".utf8))
            return nil
        }
    } else {
        // A terminal on stdin means nobody piped anything in, and reading would hang with no
        // indication of why. Say what the mode wants instead.
        guard isatty(FileHandle.standardInput.fileDescriptor) == 0 else {
            FileHandle.standardError.write(
                Data(
                    "mlx-agent digest: reads a transcript as JSON on stdin, or from --in <file>\n"
                        .utf8))
            return nil
        }
        data = FileHandle.standardInput.readDataToEndOfFile()
    }

    guard !data.isEmpty else {
        FileHandle.standardError.write(Data("mlx-agent digest: the input is empty\n".utf8))
        return nil
    }
    guard let parsed = try? JSONSerialization.jsonObject(with: data) else {
        FileHandle.standardError.write(Data("mlx-agent digest: the input is not JSON\n".utf8))
        return nil
    }
    if let messages = parsed as? [[String: Any]] { return messages }
    if let object = parsed as? [String: Any], let messages = object["messages"] as? [[String: Any]] {
        return messages
    }
    FileHandle.standardError.write(
        Data(
            ("mlx-agent digest: expected an array of {role, content} messages, or an object "
                + "with a \"messages\" array (the session/prime wire shape)\n").utf8))
    return nil
}

// MARK: - Output

private func jsonData(_ payload: [String: Any]) -> Data {
    (try? JSONSerialization.data(
        withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])) ?? Data("{}".utf8)
}

/// Write to `--out`, or to stdout. Returns false having reported the failure.
private func writeOutput(_ data: Data, to path: String?) -> Bool {
    var data = data
    if data.last != 0x0A { data.append(0x0A) }
    guard let path else {
        FileHandle.standardOutput.write(data)
        return true
    }
    let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
    do {
        try data.write(to: url, options: .atomic)
        return true
    } catch {
        FileHandle.standardError.write(
            Data("mlx-agent digest: cannot write \(path): \(error.localizedDescription)\n".utf8))
        return false
    }
}

/// The machine-readable one-liner, following `bench`/`fm-check`'s RESULT_JSON convention.
///
/// On STDERR, unlike those two, and that is the one deliberate divergence: stdout here is the
/// artifact - a JSON document or rendered text that a caller pipes into a file or a parser - and a
/// summary line mixed into it would make the common case (`mlx-agent digest < t.json | jq .digest`)
/// need filtering. Scripts that want the line read stderr; scripts that want the result read the
/// exit code, which carries the same verdict.
private func emitDigestResultJSON(
    condensed: Bool, summarizer: String, accepted: Int, primed: Int,
    dropped: (turns: Int, bytes: Int)?, reason: String?, seconds: TimeInterval
) {
    var payload: [String: Any] = [
        "condensed": condensed,
        "summarizer": summarizer,
        "accepted": accepted,
        "primed": primed,
        // Integer milliseconds, not a rounded Double: JSONSerialization renders a Double by its
        // shortest round-trip representation, so 0.776 comes back out as 0.77600000000000002.
        "ms": Int((seconds * 1000).rounded()),
    ]
    if let dropped {
        payload["droppedTurns"] = dropped.turns
        payload["droppedBytes"] = dropped.bytes
    }
    if let reason { payload["reason"] = reason }
    guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
    else { return }
    FileHandle.standardError.write(
        Data(("RESULT_JSON: " + String(decoding: data, as: UTF8.self) + "\n").utf8))
}
