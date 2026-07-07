// mlx-agent - Phase 0 + engine-gate spike
//
// This is the seed of the native Swift ACP agent described in
// ~/Development/MLXApp/docs/10-development-plan.md. It does NOT yet speak ACP.
// It exists to answer the two questions that gate the whole MLX direction:
//
//   1. Phase 0 acceptance: does mlx-swift-lm load a local mlx-community model and
//      stream a coherent completion on this machine, in a signed-able Swift binary?
//      -> `mlx-agent chat --prompt "..."`
//
//   2. Engine gate (Phase 2 folded forward): does mlx-swift-lm's built-in
//      tool-call machinery (ChatSession tools + toolDispatch + ToolCallProcessor)
//      clear the Tier-1 correctness bar (5/5 on Qwen3-4B-4bit, matching omlx)?
//      -> `mlx-agent gate`
//
// The five gate cases mirror prototype/tool_smoke_test.py so the Swift result is
// directly comparable to the omlx / llama-server baselines in docs/09.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import Tokenizers

// MARK: - Defaults

let defaultModelDir =
    "/Users/tkukielk/Development/MLXApp/models/Qwen3-4B-4bit"
let systemPrompt =
    "You are a helpful assistant. Use the provided tools when they are relevant."

// MARK: - Tool schemas (mirror tool_smoke_test.py TOOLS)

func object(_ properties: [String: any Sendable], required: [String]) -> [String: any Sendable] {
    ["type": "object", "properties": properties, "required": required]
}

func function(_ name: String, _ description: String, _ parameters: [String: any Sendable]) -> ToolSpec {
    [
        "type": "function",
        "function": [
            "name": name,
            "description": description,
            "parameters": parameters,
        ] as [String: any Sendable],
    ]
}

func toolSpecs() -> [ToolSpec] {
    [
        function(
            "get_current_weather", "Get the current weather for a city.",
            object(
                [
                    "location": ["type": "string", "description": "City name"] as [String: any Sendable],
                    "unit": ["type": "string", "enum": ["celsius", "fahrenheit"]] as [String: any Sendable],
                ], required: ["location"])),
        function(
            "read_file", "Read a file from disk and return its contents.",
            object(["path": ["type": "string"] as [String: any Sendable]], required: ["path"])),
        function(
            "add_numbers", "Add two numbers and return the sum.",
            object(
                [
                    "a": ["type": "number"] as [String: any Sendable],
                    "b": ["type": "number"] as [String: any Sendable],
                ], required: ["a", "b"])),
    ]
}

/// Synthetic tool results so the model can complete the multi-turn follow-up.
func syntheticToolResult(for call: ToolCall) -> String {
    switch call.function.name {
    case "get_current_weather":
        return #"{"temperature": 18, "unit": "celsius", "conditions": "clear"}"#
    case "read_file":
        return #"{"content": "127.0.0.1 localhost\n::1 localhost"}"#
    case "add_numbers":
        return #"{"result": 42}"#
    default:
        return #"{"result": "OK (synthetic tool result for test)"}"#
    }
}

// MARK: - Utilities

func mib(_ bytes: Int) -> String { String(format: "%.0f MiB", Double(bytes) / 1_048_576.0) }

/// Decode a ToolCall's arguments to a plain dictionary via JSON round-trip
/// (JSONValue is Codable), so the checks below never touch its internals.
func argsDict(_ call: ToolCall) -> [String: Any] {
    guard let data = try? JSONEncoder().encode(call.function.arguments),
        let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return [:] }
    return obj
}

func asNumber(_ v: Any?) -> Double? {
    if let d = v as? Double { return d }
    if let i = v as? Int { return Double(i) }
    if let s = v as? String { return Double(s) }
    if let n = v as? NSNumber { return n.doubleValue }
    return nil
}

func containsCI(_ v: Any?, _ sub: String) -> Bool {
    guard let s = v as? String else { return false }
    return s.lowercased().contains(sub.lowercased())
}

let maxToolCallsPerCase = 8  // guardrail: a misbehaving model cannot loop forever

// MARK: - Model loading

func loadModel(_ dir: String) async throws -> ModelContainer {
    let url = URL(fileURLWithPath: dir)
    guard FileManager.default.fileExists(atPath: url.appending(component: "config.json").path) else {
        throw NSError(
            domain: "mlx-agent", code: 2,
            userInfo: [NSLocalizedDescriptionKey: "no config.json under \(dir)"])
    }
    // Local-directory load: no downloader needed. The tokenizer loader is the
    // opt-in swift-transformers integration (macro expands to AutoTokenizer).
    return try await LLMModelFactory.shared.loadContainer(
        from: url, using: #huggingFaceTokenizerLoader())
}

// MARK: - chat mode (Phase 0 acceptance)

func runChat(model dir: String, prompt: String) async throws {
    FileHandle.standardError.write(Data("[mlx-agent] loading \(dir) ...\n".utf8))
    let t0 = Date()
    let container = try await loadModel(dir)
    let loadDt = Date().timeIntervalSince(t0)
    let afterLoad = MLX.Memory.snapshot()
    FileHandle.standardError.write(
        Data(
            "[mlx-agent] loaded in \(String(format: "%.1f", loadDt))s; active=\(mib(afterLoad.activeMemory))\n"
                .utf8))

    let session = ChatSession(
        container,
        instructions: systemPrompt,
        generateParameters: GenerateParameters(maxTokens: 1024, temperature: 0))

    var chunkCount = 0
    let g0 = Date()
    for try await chunk in session.streamResponse(to: prompt) {
        FileHandle.standardOutput.write(Data(chunk.utf8))
        chunkCount += 1
    }
    FileHandle.standardOutput.write(Data("\n".utf8))
    let genDt = Date().timeIntervalSince(g0)
    let peak = MLX.Memory.snapshot().peakMemory
    let rate = genDt > 0 ? Double(chunkCount) / genDt : 0
    let msg =
        "[mlx-agent] \(chunkCount) chunks in \(String(format: "%.1f", genDt))s "
        + "(\(String(format: "%.1f", rate)) chunk/s); peak=\(mib(peak))\n"
    FileHandle.standardError.write(Data(msg.utf8))
}

// MARK: - gate mode (Tier-1 tool correctness)

struct GateCase {
    let name: String
    let prompt: String
    let expectTool: String?  // nil => must NOT call a tool
    let minCalls: Int
    let check: ([String: Any]) -> String?  // returns failure reason, or nil if ok
}

func gateCases() -> [GateCase] {
    [
        GateCase(
            name: "single_weather",
            prompt: "What is the weather in Paris right now, in celsius?",
            expectTool: "get_current_weather", minCalls: 1,
            check: { a in
                if !containsCI(a["location"], "paris") { return "location != Paris (got \(a["location"] ?? "nil"))" }
                if (a["unit"] as? String) != "celsius" { return "unit != celsius (got \(a["unit"] ?? "nil"))" }
                return nil
            }),
        GateCase(
            name: "file_read",
            prompt: "Read the file at /etc/hosts and show me what is in it.",
            expectTool: "read_file", minCalls: 1,
            check: { a in
                containsCI(a["path"], "/etc/hosts") ? nil : "path !~ /etc/hosts (got \(a["path"] ?? "nil"))"
            }),
        GateCase(
            name: "numeric_args",
            prompt: "Use the tool to add 17 and 25.",
            expectTool: "add_numbers", minCalls: 1,
            check: { a in
                if asNumber(a["a"]) != 17 { return "a != 17 (got \(a["a"] ?? "nil"))" }
                if asNumber(a["b"]) != 25 { return "b != 25 (got \(a["b"] ?? "nil"))" }
                return nil
            }),
        GateCase(
            name: "no_tool_needed",
            prompt: "In one sentence, who wrote the play Hamlet?",
            expectTool: nil, minCalls: 0, check: { _ in nil }),
        GateCase(
            name: "parallel_weather",
            prompt: "Get the current weather for BOTH Paris and Tokyo.",
            expectTool: "get_current_weather", minCalls: 2, check: { _ in nil }),
    ]
}

struct CaseResult {
    let name: String
    let ok: Bool
    let note: String
    let latency: TimeInterval
}

func runOneCase(_ c: GateCase, container: ModelContainer) async -> CaseResult {
    // We drive the tool loop MANUALLY (no toolDispatch), in two explicit turns,
    // so the multi-turn completion check tests ONLY the post-tool-result turn.
    // Rationale: Qwen3 emits a <think> block BEFORE its <tool_call>; if we checked
    // the cumulative respond() output, that pre-call thinking would falsely satisfy
    // "the model produced a final answer" even if the real follow-up turn was empty.
    // This mirrors tool_smoke_test.py's separate _multi_turn_followup call.
    let session = ChatSession(
        container,
        instructions: systemPrompt,
        generateParameters: GenerateParameters(maxTokens: 2048, temperature: 0),
        tools: toolSpecs())

    let t0 = Date()

    // Turn 1: collect the tool calls the model emits. With no toolDispatch set,
    // ChatSession yields toolCall items as Generation values instead of looping.
    var calls: [ToolCall] = []
    var turn1Text = ""
    do {
        for try await gen in session.streamDetails(to: c.prompt) {
            if let tc = gen.toolCall {
                calls.append(tc)
                if calls.count > maxToolCallsPerCase {
                    return CaseResult(
                        name: c.name, ok: false,
                        note: "exceeded max tool calls (\(maxToolCallsPerCase))",
                        latency: Date().timeIntervalSince(t0))
                }
            } else if let ch = gen.chunk {
                turn1Text += ch
            }
        }
    } catch {
        return CaseResult(
            name: c.name, ok: false, note: "turn1 error: \(error.localizedDescription)",
            latency: Date().timeIntervalSince(t0))
    }

    // Case: must NOT call a tool.
    if c.expectTool == nil {
        if !calls.isEmpty {
            return CaseResult(
                name: c.name, ok: false,
                note: "FALSE POSITIVE: called \(calls[0].function.name)",
                latency: Date().timeIntervalSince(t0))
        }
        let trimmed = turn1Text.trimmingCharacters(in: .whitespacesAndNewlines)
        return CaseResult(
            name: c.name, ok: !trimmed.isEmpty,
            note: trimmed.isEmpty ? "empty answer" : "answered directly (\(trimmed.count) chars)",
            latency: Date().timeIntervalSince(t0))
    }

    // Case: expect a tool call.
    let named = calls.filter { $0.function.name == c.expectTool }
    if calls.isEmpty {
        return CaseResult(
            name: c.name, ok: false, note: "NO tool_call emitted",
            latency: Date().timeIntervalSince(t0))
    }
    if named.isEmpty {
        let got = Set(calls.map { $0.function.name }).sorted().joined(separator: ",")
        return CaseResult(
            name: c.name, ok: false, note: "wrong tool: got [\(got)], want \(c.expectTool!)",
            latency: Date().timeIntervalSince(t0))
    }
    if let reason = c.check(argsDict(named[0])) {
        return CaseResult(
            name: c.name, ok: false, note: "arg check: \(reason)",
            latency: Date().timeIntervalSince(t0))
    }
    if named.count < c.minCalls {
        return CaseResult(
            name: c.name, ok: false,
            note: "expected >=\(c.minCalls) \(c.expectTool!) calls, got \(named.count)",
            latency: Date().timeIntervalSince(t0))
    }

    // Turn 2 (multi-turn completion): feed synthetic tool results back into the
    // SAME session (KV cache preserved) and require a non-empty follow-up. Only
    // this turn's text counts - it isolates the post-tool-result answer.
    let toolMessages = calls.map { Chat.Message.tool(syntheticToolResult(for: $0), id: $0.id) }
    let turn2Text: String
    do {
        turn2Text = try await session.respond(to: toolMessages)
    } catch {
        return CaseResult(
            name: c.name, ok: false, note: "turn2 (post-tool) error: \(error.localizedDescription)",
            latency: Date().timeIntervalSince(t0))
    }
    let dt = Date().timeIntervalSince(t0)
    if turn2Text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        return CaseResult(
            name: c.name, ok: false,
            note: "tool call emitted but post-tool turn produced no answer", latency: dt)
    }
    let parallelNote = named.count > 1 ? " [\(named.count) calls]" : ""
    return CaseResult(
        name: c.name, ok: true,
        note: "ok: \(named[0].function.name)(\(argsDict(named[0])))\(parallelNote)", latency: dt)
}

func runGate(model dir: String) async throws -> Int {
    print(String(repeating: "=", count: 78))
    print("mlx-agent Tier-1 tool-calling gate")
    print("model  : \(dir)")
    print("engine : mlx-swift-lm 3.31.4 (MLXLLM ChatSession + ToolCallProcessor)")
    print(String(repeating: "=", count: 78))

    let baseline = MLX.Memory.snapshot()
    let container = try await loadModel(dir)
    let afterLoad = MLX.Memory.snapshot()
    print("loaded; model active memory ~= \(mib(afterLoad.activeMemory - baseline.activeMemory))")
    print(String(repeating: "-", count: 78))

    var failures = 0
    for c in gateCases() {
        let r = await runOneCase(c, container: container)
        if !r.ok { failures += 1 }
        let status = r.ok ? "PASS" : "FAIL"
        print(String(format: "[%@] %-16@ %6.2fs  %@", status, r.name as NSString, r.latency, r.note))
    }

    let peak = MLX.Memory.snapshot().peakMemory
    print(String(repeating: "-", count: 78))
    print(
        "\(gateCases().count - failures)/\(gateCases().count) passed;  "
            + "peak GPU memory (full run, load + all cases) = \(mib(peak))")
    print(
        "gate bar: 5/5 (matches omlx + Qwen3-4B-4bit in docs/09). "
            + (failures == 0 ? "GATE PASSED - native path confirmed." : "GATE NOT MET - see failures above."))
    return failures
}

// MARK: - CLI

func usage() {
    let text = """
        mlx-agent - MLX engine spike (Phase 0 + Tier-1 tool gate)

        USAGE:
          mlx-agent acp  [--model <dir>]                  ACP server over stdio (Phase 1: chat)
          mlx-agent gate [--model <dir>]                  Tier-1 tool-calling gate
          mlx-agent chat [--model <dir>] --prompt <text>  load + stream one completion

        OPTIONS:
          --model <dir>    model directory (default: \(defaultModelDir))
          --prompt <text>  prompt for chat mode

        EXAMPLES:
          mlx-agent acp
          mlx-agent gate
          mlx-agent chat --prompt "Name three cities in Japan."
        """
    print(text)
}

func option(_ name: String, in args: [String]) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

let cliArgs = Array(CommandLine.arguments.dropFirst())
let mode = cliArgs.first
let modelDir = option("--model", in: cliArgs) ?? defaultModelDir

do {
    switch mode {
    case "acp":
        await ACPServer(modelDir: modelDir).serve()
    case "gate":
        let failures = try await runGate(model: modelDir)
        exit(Int32(failures))
    case "chat":
        guard let prompt = option("--prompt", in: cliArgs) else {
            FileHandle.standardError.write(Data("chat mode requires --prompt <text>\n".utf8))
            exit(2)
        }
        try await runChat(model: modelDir, prompt: prompt)
    default:
        usage()
    }
} catch {
    FileHandle.standardError.write(Data("mlx-agent: \(error.localizedDescription)\n".utf8))
    exit(1)
}
