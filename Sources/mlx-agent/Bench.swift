// bench mode - reproduce Apple's M5 LLM benchmark methodology on the mlx-agent stack.
//
// Apple's "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU" post
// (machinelearning.apple.com/research/exploring-llms-mlx-m5) measures, per model:
//   - TTFT: seconds to process a 4096-token prompt (prefill; compute-bound, this is
//     where the M5's GPU Neural Accelerators show up - MLX >= 0.30.0, macOS >= 26.2)
//   - generation speed: tok/s over 128 generated tokens (memory-bandwidth-bound)
// This mode runs the same shape through mlx-swift-lm's generate() so the numbers
// reflect exactly what mlx-agent's acp/oneshot/chat/map modes get, not a separate
// Python stack. The prompt is raw token IDs (no chat template) so the prefill length
// is exact and identical across models and machines.
//
// One unmeasured warmup run precedes the measured runs: mlx-swift JIT-compiles the
// steel gemm/attention kernels (including the *_nax Neural Accelerator variants) on
// first use, and that Metal compile would otherwise pollute the first TTFT.

import Foundation
import MLX
import MLXLMCommon
import Tokenizers

private func sysctlString(_ name: String) -> String {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "unknown" }
    var buf = [UInt8](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return "unknown" }
    return String(decoding: buf.prefix(while: { $0 != 0 }), as: UTF8.self)
}

private func sysctlInt(_ name: String) -> Int {
    var value: Int64 = 0
    var size = MemoryLayout<Int64>.size
    guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return 0 }
    return Int(value)
}

private func median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let s = values.sorted()
    let mid = s.count / 2
    return s.count % 2 == 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid]
}

/// A filler passage tokenized and repeated until the requested prompt length is reached.
/// Ordinary English prose, so per-token compute is representative of real use.
private let benchPassage = """
    The quick brown fox jumps over the lazy dog while the river runs quietly past the \
    old mill. Seasons turn, markets open and close, and the town keeps its own steady \
    time. A librarian catalogs maps of coastlines that have long since shifted, and a \
    clockmaker adjusts a balance wheel by a breath of a turn.
    """

func runBench(
    model dir: String, promptTokens: Int, genTokens: Int, runs: Int, prefillStep: Int?,
    extraEOSTokens: Set<String>
) async throws -> Int {
    let chip = sysctlString("machdep.cpu.brand_string")
    let ram = sysctlInt("hw.memsize")
    let os = ProcessInfo.processInfo.operatingSystemVersion
    let osString = "\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)"
    // Heuristic mirror of mlx's is_nax_available(): the real check (GPU architecture
    // generation >= 17 on macOS/iOS/tvOS/visionOS >= 26.2) is not exposed through
    // mlx-swift, so approximate it from the chip name for the banner only. The kernels
    // themselves are picked by the real check inside mlx, not by this.
    let m5Family = chip.range(of: #"\bM(\d+)\b"#, options: .regularExpression)
        .flatMap { Int(chip[$0].dropFirst()) }.map { $0 >= 5 } ?? false
    let naxOS = ProcessInfo.processInfo.isOperatingSystemAtLeast(
        OperatingSystemVersion(majorVersion: 26, minorVersion: 2, patchVersion: 0))
    let naxExpected = m5Family && naxOS

    func log(_ s: String) { FileHandle.standardError.write(Data("[mlx-agent bench] \(s)\n".utf8)) }

    log("chip=\(chip) ram=\(gib(ram)) macos=\(osString)")
    log("neural accelerators expected: \(naxExpected ? "yes" : "no") (needs M5-family GPU + macOS >= 26.2)")
    log("loading \(dir) ...")

    let t0 = Date()
    let container = try await loadModel(dir, extraEOSTokens: extraEOSTokens)
    let loadDt = Date().timeIntervalSince(t0)
    let weights = MLX.Memory.snapshot().activeMemory
    log("loaded in \(String(format: "%.1f", loadDt))s; weights active=\(gib(weights))")

    struct RunStats {
        var ttft: Double  // seconds spent on prefill
        var prefillTPS: Double
        var genTPS: Double
        var promptCount: Int
        var genCount: Int
    }

    let stats: [RunStats] = try await container.perform { (context: ModelContext) in
        // Build exactly promptTokens ids by repeating the passage. No BOS bookkeeping:
        // a benchmark prompt needs realistic length, not semantic validity.
        var ids: [Int] = []
        while ids.count < promptTokens {
            ids.append(contentsOf: context.tokenizer.encode(text: benchPassage))
        }
        ids = Array(ids.prefix(promptTokens))

        // Greedy top-1 sampling. A model may emit its EOS before genTokens; such a run
        // reports its true genCount and the tok/s stays valid (rate, not total).
        var parameters = GenerateParameters(maxTokens: genTokens, temperature: 0)
        if let prefillStep { parameters.prefillStepSize = prefillStep }

        var collected: [RunStats] = []
        for run in 0...runs {  // run 0 is the unmeasured JIT warmup
            let input = LMInput(tokens: MLXArray(ids))
            var info: GenerateCompletionInfo? = nil
            let stream = try MLXLMCommon.generate(
                input: input, parameters: parameters, context: context)
            for await generation in stream {
                if case .info(let i) = generation { info = i }
            }
            guard let info else {
                throw NSError(
                    domain: "mlx-agent", code: 4,
                    userInfo: [NSLocalizedDescriptionKey: "generation produced no completion info"])
            }
            let s = RunStats(
                ttft: info.promptTime,
                prefillTPS: info.promptTime > 0
                    ? Double(info.promptTokenCount) / info.promptTime : 0,
                genTPS: info.generateTime > 0
                    ? Double(info.generationTokenCount) / info.generateTime : 0,
                promptCount: info.promptTokenCount,
                genCount: info.generationTokenCount)
            if run == 0 {
                FileHandle.standardError.write(
                    Data(
                        "[mlx-agent bench] warmup: ttft=\(String(format: "%.2f", s.ttft))s gen=\(String(format: "%.1f", s.genTPS)) tok/s\n"
                            .utf8))
            } else {
                collected.append(s)
                FileHandle.standardError.write(
                    Data(
                        "[mlx-agent bench] run \(run)/\(runs): ttft=\(String(format: "%.2f", s.ttft))s prefill=\(String(format: "%.0f", s.prefillTPS)) tok/s gen=\(String(format: "%.1f", s.genTPS)) tok/s (\(s.genCount) tokens)\n"
                            .utf8))
            }
        }
        return collected
    }

    let peak = MLX.Memory.snapshot().peakMemory
    let ttft = median(stats.map { $0.ttft })
    let prefill = median(stats.map { $0.prefillTPS })
    let gen = median(stats.map { $0.genTPS })
    // A HuggingFace cache path ends in snapshots/<revision-hash>; report the
    // models--org--name component instead of the meaningless hash.
    let components = URL(fileURLWithPath: dir).pathComponents
    let modelName =
        components.reversed().first { $0.hasPrefix("models--") }
            .map { $0.replacingOccurrences(of: "models--", with: "").replacingOccurrences(of: "--", with: "/") }
        ?? components.last ?? dir

    log("peak GPU memory=\(gib(peak)) of \(gib(ram)) physical")
    if peak > ram * 7 / 10 {
        log("WARNING: peak memory is over 70% of physical RAM; generation speed on this "
            + "machine is likely degraded by memory pressure, not by the GPU")
    }

    // Human summary on stdout, then one machine-readable line for cross-Mac collection.
    print("model:            \(modelName)")
    print("prompt tokens:    \(stats.first?.promptCount ?? promptTokens)")
    print("gen tokens:       \(genTokens) requested, \(stats.first?.genCount ?? 0) measured (first run)")
    print("TTFT (median):    \(String(format: "%.2f", ttft)) s")
    print("prefill (median): \(String(format: "%.0f", prefill)) tok/s")
    print("gen (median):     \(String(format: "%.1f", gen)) tok/s")

    let record: [String: Any] = [
        "chip": chip, "ram_bytes": ram, "macos": osString,
        "nax_expected": naxExpected,
        "model": modelName,
        "prompt_tokens": stats.first?.promptCount ?? promptTokens,
        "gen_tokens": stats.first?.genCount ?? 0,
        "runs": runs,
        "ttft_s": (ttft * 100).rounded() / 100,
        "prefill_tps": prefill.rounded(),
        "gen_tps": (gen * 10).rounded() / 10,
        "peak_bytes": peak,
    ]
    if let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]) {
        print("RESULT_JSON: " + String(decoding: data, as: UTF8.self))
    }
    return 0
}
