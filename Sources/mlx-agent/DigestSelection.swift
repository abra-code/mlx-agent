// DigestSelection.swift - who summarizes, and how much conversation they can take at once.
//
// Two questions that only have answers at the agent level, kept out of both generators so neither
// has to know the other exists.
//
// WHO. Foundation Models is cheaper - no weights, no disturbance to the session model - but its
// window is 4096 tokens, so a long conversation becomes many sequential passes and past
// `maxSlices` it refuses outright. A 32k-window model that is ALREADY LOADED does the same job in
// one or two passes. Neither is simply better, so `auto` measures instead of preferring: the slice
// count under the on-device model's budget is pure, cheap arithmetic (`DigestPlanner.sliceCount`),
// which turns "which summarizer" into a decision rather than a coin flip.
//
// HOW MUCH. A slice budget is a property of the summarizing model's context window. FM reports
// its own; nothing else here did, so the budget was clamped at 8192 - a ceiling sized for a
// 4096-token model, which made a 128k-context MLX model do fifteen passes where one would do.

import AgentDigest
import Foundation

/// What the prime response's `summarizer` field carries for Apple's on-device model.
///
/// Declared here rather than only on `FMDigestGenerator` because the selection logic has to be
/// able to NAME a summarizer in a build where the framework does not exist and that type is not
/// compiled at all. `FMDigestGenerator.name` is this constant.
let foundationSummarizerName = "apple-foundation-models"

/// Which model summarizes at prime time. `--digest-backend`, or one restore's `condense.backend`.
///
/// The flag is the agent's DEFAULT and the request wins, because the two answer to different
/// people: the flag is whoever launched the agent, the request is whoever is sitting in front of
/// the client choosing a summarizer for the conversation they are looking at. A client that offers
/// the choice cannot honor it otherwise - the agent outlives any one restore, so the answer would
/// always be the one that was true when it started.
///
/// `none` is the exception and does not yield to a request: see `ACPServer.effectiveBackend`.
enum DigestBackendChoice: String, Sendable, CaseIterable {
    /// Measure, then pick. See the file header.
    case auto
    /// Apple's on-device model, or decline. Never touches the session's model.
    case foundation
    /// The session's own backend, or decline. The only option that costs the session anything -
    /// and the only one that exists below macOS 26.
    case session
    /// Do not summarize. `session/prime` with `condense` still succeeds; it primes the full
    /// history and says why, which is the same contract as every other refusal on this path.
    case none

    static let usage = "auto|foundation|session|none"

    /// nil for an unrecognized value, so the caller can exit(2) with a usage message rather than
    /// silently falling back to a default the user did not ask for.
    static func parse(_ raw: String?) -> DigestBackendChoice? {
        guard let raw else { return .auto }
        return DigestBackendChoice(rawValue: raw.lowercased())
    }
}

/// The summarizing model's context window, in tokens.
enum DigestWindow {

    /// What a window is assumed to be when nothing states it - the llama-server case, where the
    /// window is a property of a process this agent did not start. Deliberately small: too low
    /// costs an extra pass, too high costs a failed generation deep into a restore.
    static let conservativeDefault = 8192

    /// The largest slice this will size, whatever the model claims.
    ///
    /// A slice is a prefill on the critical path of a restore, and prefill cost is linear in slice
    /// size while the SAVING from a bigger slice is only the passes it avoids. Past roughly 32k
    /// tokens a local model spends tens of seconds and a gigabyte of KV cache to save one pass,
    /// which is the wrong side of the trade - and a 128k claim is usually a RoPE-scaled ceiling the
    /// model does not actually reason well at anyway.
    static let practicalCeiling = 32768

    /// Read the window out of an MLX model directory's `config.json`.
    ///
    /// Safe to call at condense time rather than at load: `loadModel` already requires this file to
    /// exist in this directory, it is a few kilobytes, and the model can be SWITCHED at runtime
    /// (`session/set_config` on the MLX path), so a value captured at startup would be stale.
    ///
    /// nil when the file cannot be read or states nothing recognizable, which the caller turns into
    /// `conservativeDefault` rather than into a failure - a wrong guess here costs an extra pass,
    /// and refusing to summarize because a config.json is unusual would be a worse trade.
    static func fromModelDirectory(_ dir: String) -> Int? {
        guard !dir.isEmpty,
            let data = try? Data(
                contentsOf: URL(fileURLWithPath: dir).appendingPathComponent("config.json")),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }

        // Multimodal conversions nest the language model's config, and the outer object then
        // carries vision numbers that must not be mistaken for a window.
        let scopes: [[String: Any]] = [
            root,
            (root["text_config"] as? [String: Any]) ?? [:],
            (root["llm_config"] as? [String: Any]) ?? [:],
        ]
        // In descending order of how reliably each names the thing we mean. `sliding_window` is
        // deliberately absent: it is the ATTENTION window, not the prompt limit, and a Gemma-style
        // model happily takes a prompt many times longer than one.
        let keys = ["max_position_embeddings", "max_seq_len", "seq_length", "n_positions"]
        for scope in scopes {
            for key in keys {
                guard let value = (scope[key] as? NSNumber)?.intValue else { continue }
                // A config can carry a placeholder; treat anything outside the plausible range as
                // "not stated" rather than as a window.
                guard value >= 512, value <= 8_000_000 else { continue }
                return value
            }
        }
        return nil
    }

    /// The window to size a session-backed condensation for, given what the CLI said and what the
    /// model directory claims.
    static func forSession(override: Int?, modelDir: String) -> Int {
        if let override { return override }
        return fromModelDirectory(modelDir) ?? conservativeDefault
    }
}

/// Turning "a model with this window, allowed to generate this much" into a `PrimePolicy`.
///
/// Shared by the two callers that drive a `GenerationBackend` as a summarizer - `session/prime`
/// with `--digest-backend session`, and the offline `digest` mode - because the reservation below
/// is not obvious and getting it wrong is silent: the condensation runs, every pass overflows the
/// window, and the client is told its conversation was too large. One derivation, one place to fix.
///
/// Not applicable to Foundation Models: that path is told `maximumResponseTokens` and so generates
/// exactly one digest, which `PrimePolicy` already reserves for. See `FoundationBackend
/// .defaultDigestPolicy`.
enum DigestSizing {

    /// A policy to summarize with, or the reason this model cannot be a summarizer at all.
    ///
    /// Not `Result`: the failure is a sentence for a person, not an error to be thrown, and both
    /// callers turn it straight into client-facing text.
    enum Outcome {
        case sized(PrimePolicy)
        case refused(String)
    }

    /// Below this a slice is too small to be worth the passes it would take: at 1024 tokens a pass
    /// reads about 4 KB, so an ordinary conversation needs dozens of them and the planner refuses
    /// on slice count anyway - with a message about the conversation's size that points away from
    /// the actual cause.
    static let minimumSliceTokens = 1024

    /// The most one summarization pass can generate, in tokens.
    ///
    /// `GenerateParameters.maxTokens` is the only real bound on a pass - `BackendDigestGenerator`'s
    /// byte caps stop a runaway, but for a reasoning model they sit above this - so it is what the
    /// slice budget has to make room for. Reserving it exactly is what makes the reservation
    /// honest; an earlier version clamped it at 8192, which made the "exact" claim false for
    /// anyone passing a larger `--max-new-tokens`.
    ///
    /// Mirrors the 4096 default that `ACPServer.buildSessionStack` and the digest mode both start
    /// from. Lowering `--max-new-tokens` buys bigger slices, which is the correct trade; raising it
    /// past what the window can hold makes `backendPolicy` refuse and say so.
    static func generationCeiling(_ gen: GenConfig) -> Int { max(gen.maxNewTokens ?? 4096, 256) }

    /// The window to size for, from the CLI override and the model directory, capped at
    /// `DigestWindow.practicalCeiling`. Pass an empty `modelDir` where there is no local model to
    /// read a `config.json` from (the llama-server case), which lands on the conservative default.
    static func window(override: Int?, modelDir: String) -> Int {
        min(
            DigestWindow.forSession(override: override, modelDir: modelDir),
            DigestWindow.practicalCeiling)
    }

    /// The policy to summarize with, or the reason this model cannot summarize at all.
    ///
    /// Room is taken out for what the backend may GENERATE before the policy divides up what is
    /// left. `PrimePolicy` reserves one digest allowance for the answer, which is what the
    /// on-device model actually produces - it is told `maximumResponseTokens`. A
    /// `GenerationBackend` is not: there is no per-pass token limit on it, so a pass generates up
    /// to the model's own `maxTokens`, and a reasoning model spends most of that thinking before it
    /// writes a character of the answer. Reserving the whole ceiling is exact rather than
    /// optimistic; it double-counts one digest allowance, which is the safe direction.
    ///
    /// The reserve can eat the window: `--backend openai --max-new-tokens 8192` against the assumed
    /// 8192 window leaves nothing, and the arithmetic then produces a 256-token slice that "works" -
    /// a 40 KB conversation needs 51 passes, the planner refuses on slice count, and the caller is
    /// told its conversation is too large when the truth is that this model was configured to
    /// generate its own entire window. Say the true thing instead, and name both ways out.
    ///
    /// The two overrides are applied BEFORE the check, not after, because `maxDigestTokens` is one
    /// of the things the window has to hold: raising it lowers the slice budget (see
    /// `PrimePolicy.with`), so checking first would bless a budget that no longer exists.
    static func backendPolicy(
        window: Int, generationCeiling reserve: Int, maxSlices: Int,
        keepRecentTurns: Int? = nil, maxDigestTokens: Int? = nil
    ) -> Outcome {
        let policy = PrimePolicy.sized(window: max(256, window - reserve), maxSlices: maxSlices)
            .with(keepRecentTurns: keepRecentTurns, maxDigestTokens: maxDigestTokens)
        guard policy.sliceBudgetTokens >= minimumSliceTokens else {
            return .refused(
                "the model is configured to generate up to \(reserve) tokens of its "
                    + "\(window)-token window, which leaves no room to summarize in - lower "
                    + "--max-new-tokens, or state the model's real window with --digest-window")
        }
        return .sized(policy)
    }
}
