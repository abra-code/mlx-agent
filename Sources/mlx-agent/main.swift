// mlx-agent - native Swift ACP agent on mlx-swift-lm.
//
// CLI entry point. Modes:
//   acp      - Agent Client Protocol server over stdio (chat + agentic tools)
//   oneshot  - run one agent turn and print it (non-interactive; drives test suites)
//   chat     - load a model and stream a single completion
//   gate     - built-in tool-calling correctness check (5 fixed cases)
//   tools    - launch the --mcp-config servers, dump their tool surface as JSON (no model)
//
// `gate` is a self-contained diagnostic: it drives ChatSession's tool machinery over a
// handful of prompts (single tool call, correct args, parallel calls, and a negative
// "must NOT call a tool" case) and reports pass/fail, so a regression in the underlying
// engine is caught without the full ACP stack.

import Foundation
import AgentText
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import Tokenizers

// MARK: - Defaults

// No hardcoded model path: the model directory comes from --model or the
// MLX_AGENT_MODEL environment variable (see requireModelDir).
let defaultModelDir = ProcessInfo.processInfo.environment["MLX_AGENT_MODEL"]

/// The system prompt applied when `--system-prompt` is not supplied. Passing
/// `--system-prompt ""` (an explicitly empty value) selects NO system message at all
/// (the `.system(...)` turn is skipped); passing any other text uses that text verbatim.
/// See `resolveSystemPrompt`.
let defaultSystemPrompt =
    "You are a helpful assistant. Use the provided tools when they are relevant."

// MARK: - Tool schemas (gate diagnostic)

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

// MARK: - RAM gate

func gib(_ bytes: Int) -> String { String(format: "%.1f GB", Double(bytes) / 1_073_741_824.0) }

/// Sum the on-disk size of every *.safetensors shard under `dir` (recursively). This is the
/// weights footprint MLX must hold resident in unified memory - the dominant term in whether a
/// model fits. Returns 0 if nothing is found (then the gate no-ops rather than guessing).
func modelWeightBytes(_ dir: URL) -> Int {
    let fm = FileManager.default
    guard let en = fm.enumerator(at: dir, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
    var total = 0
    for case let u as URL in en where u.pathExtension == "safetensors" {
        if let sz = (try? u.resourceValues(forKeys: [.fileSizeKey]))?.fileSize { total += sz }
    }
    return total
}

/// Hard backstop that refuses a load certain to exhaust unified memory, turning a would-be
/// OOM-kill of the whole process (no error reaches the client) into a clean, actionable ACP
/// error. This is intentionally a HIGHER bar than the GUI's advisory 75% warning (which the
/// user can override with "Load Anyway"): the agent only refuses in the genuinely-doomed
/// regime, so an overridden soft-warn still loads. Fraction is `MLX_AGENT_MAX_RAM_FRACTION`
/// (default 0.90); set it to 0 (or any value >= 1) to disable the gate entirely.
func ramGate(dir: URL) throws {
    // Honor only a finite, non-negative override; NaN / negative / unparseable falls back to
    // the safe default so a fat-fingered value never silently defeats the backstop (NaN fails
    // every comparison, so it would otherwise slip through the guard below and disable the gate).
    var fraction = 0.90
    if let p = ProcessInfo.processInfo.environment["MLX_AGENT_MAX_RAM_FRACTION"].flatMap(Double.init),
        p.isFinite, p >= 0
    {
        fraction = p
    }
    guard fraction > 0, fraction < 1 else { return }  // a deliberate 0 or >=1 disables the gate

    let weights = modelWeightBytes(dir)
    guard weights > 0 else { return }  // unknown footprint: don't block on a guess

    let physical = Int(ProcessInfo.processInfo.physicalMemory)
    guard physical > 0 else { return }
    let limit = Int(Double(physical) * fraction)
    guard weights > limit else { return }

    let name = dir.lastPathComponent
    let pct = Int((fraction * 100).rounded())
    throw NSError(
        domain: "mlx-agent", code: 3,
        userInfo: [
            NSLocalizedDescriptionKey:
                "model \"\(name)\" needs ~\(gib(weights)) of weights but this Mac has "
                + "\(gib(physical)) of unified memory (load limit \(gib(limit)) at \(pct)%). "
                + "Use a smaller model or a more aggressive quantization. "
                + "Set MLX_AGENT_MAX_RAM_FRACTION to override."
        ])
}

// MARK: - Model loading

func loadModel(_ dir: String, extraEOSTokens: Set<String> = []) async throws -> ModelContainer {
    let url = URL(fileURLWithPath: dir)
    guard FileManager.default.fileExists(atPath: url.appending(component: "config.json").path) else {
        throw NSError(
            domain: "mlx-agent", code: 2,
            userInfo: [NSLocalizedDescriptionKey: "no config.json under \(dir)"])
    }
    // Refuse a load that cannot fit in unified memory before touching the engine, so the
    // failure is a clean ACP error rather than an OOM-kill of the whole process.
    try ramGate(dir: url)
    // Local-directory load: no downloader needed. We build a ResolvedModelConfiguration by
    // hand (rather than the loadContainer(from:using:) directory convenience, which hardcodes
    // empty extraEOSTokens) so a caller-supplied stop token is unioned into the generation
    // stop set at runtime - fixing a model whose generation_config.json omits one (e.g. a
    // Gemma conversion missing <end_of_turn>) WITHOUT mutating the downloaded files. The
    // tokenizer loader is the opt-in swift-transformers integration (macro -> AutoTokenizer).
    var resolved = ResolvedModelConfiguration(directory: url)
    resolved.extraEOSTokens = extraEOSTokens
    let context = try await LLMModelFactory.shared._load(
        configuration: resolved, tokenizerLoader: #huggingFaceTokenizerLoader())
    return LLMModelFactory.shared._wrap(context)
}

// MARK: - chat mode

func runChat(
    model dir: String, prompt: String, systemPrompt: String?, gen: GenConfig,
    extraEOSTokens: Set<String>
) async throws {
    FileHandle.standardError.write(Data("[mlx-agent] loading \(dir) ...\n".utf8))
    let t0 = Date()
    let container = try await loadModel(dir, extraEOSTokens: extraEOSTokens)
    let loadDt = Date().timeIntervalSince(t0)
    let afterLoad = MLX.Memory.snapshot()
    FileHandle.standardError.write(
        Data(
            "[mlx-agent] loaded in \(String(format: "%.1f", loadDt))s; active=\(mib(afterLoad.activeMemory))\n"
                .utf8))

    let session = ChatSession(
        container,
        instructions: systemPrompt,
        generateParameters: gen.apply(to: GenerateParameters(maxTokens: 1024, temperature: 0)))

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

// MARK: - gate mode (tool-calling correctness)

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
    let session = ChatSession(
        container,
        instructions: defaultSystemPrompt,
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

func runGate(model dir: String, extraEOSTokens: Set<String>) async throws -> Int {
    print(String(repeating: "=", count: 78))
    print("mlx-agent tool-calling gate")
    print("model  : \(dir)")
    print("engine : mlx-swift-lm 3.31.4 (MLXLLM ChatSession + ToolCallProcessor)")
    print(String(repeating: "=", count: 78))

    let baseline = MLX.Memory.snapshot()
    let container = try await loadModel(dir, extraEOSTokens: extraEOSTokens)
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
        "gate bar: \(gateCases().count)/\(gateCases().count). "
            + (failures == 0 ? "GATE PASSED." : "GATE NOT MET - see failures above."))
    return failures
}

// MARK: - oneshot mode (non-interactive single turn)

/// Console delegate: streams the assistant answer to stdout, everything else (thoughts,
/// tool calls, usage) to stderr, and answers the permission gate from `--auto-permission`.
final class ConsoleAgentDelegate: AgentDelegate, @unchecked Sendable {
    let autoPermission: PermissionOutcome
    init(autoPermission: PermissionOutcome) { self.autoPermission = autoPermission }

    private func err(_ s: String) { FileHandle.standardError.write(Data(s.utf8)) }

    func agentEmitText(kind: ThinkSplitter.Kind, _ text: String) {
        switch kind {
        case .message: FileHandle.standardOutput.write(Data(text.utf8))
        case .thought: err(text)  // reasoning to stderr; stdout stays the final answer
        }
    }
    func agentToolCallStarted(id: String, title: String, kind: String, rawInput: String) {
        err("\n[tool_call] \(title) (\(kind))\n\(rawInput)\n")
    }
    func agentToolCallProgress(id: String) { err("[tool_call] running...\n") }
    func agentToolCallFinished(id: String, status: String, output: String) {
        err("[tool_call] \(status):\n\(output)\n")
    }
    func agentRequestPermission(toolCallId: String, toolName: String, title: String) async
        -> PermissionOutcome
    {
        // The console runner has no user to ask, so --permission decides every call and there is
        // no "always" to remember; toolName is unused here by design.
        err("[permission] \(title) -> auto-\(autoPermission)\n")
        return autoPermission
    }
    func agentTurnUsage(totalTokens: Int, tokensPerSecond: Double) {
        if tokensPerSecond > 0 {
            err(String(format: "[usage] %d tokens, %.1f tok/s\n", totalTokens, tokensPerSecond))
        } else {
            err("[usage] \(totalTokens) tokens\n")
        }
    }
}

func runOneshot(
    engine: EngineSpec, prompt: String, mcpConfigPath: String?,
    guardrails: AgentGuardrails, autoPermission: PermissionOutcome,
    systemPrompt: String?, gen: GenConfig, extraEOSTokens: Set<String>
) async throws -> Int {
    let parameters = gen.apply(to: GenerateParameters(maxTokens: 4096, temperature: 0))
    let backend: GenerationBackend
    switch engine {
    case .mlx(let dir):
        let container = try await loadModel(dir, extraEOSTokens: extraEOSTokens)
        backend = MLXBackend(
            ChatSession(container, instructions: systemPrompt, generateParameters: parameters))
    case .openai(let baseURL):
        // No model to load and no RAM gate to run: the weights live in llama-server.
        try await OpenAIBackend.waitForHealth(baseURL: baseURL)
        backend = OpenAIBackend(
            baseURL: baseURL, parameters: parameters, systemPrompt: systemPrompt)
    }

    var registry: MCPToolRegistry? = nil
    if let mcpConfigPath {
        registry = await MCPToolRegistry.build(try MCPConfigLoader.load(mcpConfigPath))
    }

    let agent = Agent(backend: backend, registry: registry, guardrails: guardrails)
    let delegate = ConsoleAgentDelegate(autoPermission: autoPermission)
    agent.delegate = delegate  // held by this local for the duration of the turn
    agent.setToolsEnabled(registry?.toolSpecs.isEmpty == false)

    let outcome = await agent.runTurn([.user(prompt)])
    FileHandle.standardOutput.write(Data("\n".utf8))
    registry?.terminateAll()

    switch outcome {
    case .stop(let reason):
        FileHandle.standardError.write(Data("[oneshot] stopReason=\(reason)\n".utf8))
        return 0
    case .cancelled:
        FileHandle.standardError.write(Data("[oneshot] cancelled\n".utf8))
        return 1
    case .failed(let message):
        FileHandle.standardError.write(Data("[oneshot] failed: \(message)\n".utf8))
        return 1
    }
}

// MARK: - tools mode (MCP tool-surface dump)

/// Launch every server in --mcp-config exactly as a session would - same MCPServer.launch
/// handshake (initialize + tools/list, 30s bound) and the same exposed-name collision rule
/// as MCPToolRegistry.build - then dump the resulting tool surface as JSON on stdout and
/// shut the children down. No model is loaded. This is the introspection backend for GUI
/// inspectors: `exposedName` is the function name the model would actually see, `gated`
/// marks tools that require a permission round-trip before dispatch, and a server that
/// fails its handshake is reported with status "error" instead of being silently absent.
func runToolsDump(mcpConfigPath: String) async -> Int {
    let configs: [MCPServerConfig]
    do {
        configs = try MCPConfigLoader.load(mcpConfigPath)
    } catch {
        FileHandle.standardError.write(
            Data("mlx-agent tools: \(error.localizedDescription)\n".utf8))
        return 1
    }

    var claimed: Set<String> = []  // exposed names, first server wins the bare name
    var serversOut: [[String: Any]] = []
    var launched: [MCPServer] = []

    for config in configs {
        var entry: [String: Any] = [
            "name": config.name,
            "command": config.command,
            "args": config.args,
        ]
        do {
            let server = try await MCPServer.launch(config)
            launched.append(server)
            var toolsOut: [[String: Any]] = []
            for tool in server.tools {
                // Same rule as MCPToolRegistry.build: first claim keeps the bare name,
                // later collisions are exposed as "<server>__<tool>".
                let exposed =
                    claimed.contains(tool.name) ? "\(config.name)__\(tool.name)" : tool.name
                if claimed.contains(exposed) { continue }  // double collision: absent, as in build
                claimed.insert(exposed)
                var t: [String: Any] = [
                    "name": tool.name,
                    "exposedName": exposed,
                    "description": tool.description ?? "",
                    "gated": config.gatedTools.contains(tool.name),
                ]
                // MCP.Value is Codable; round-trip the JSON-Schema into a plain JSON tree.
                if let data = try? JSONEncoder().encode(tool.inputSchema),
                    let obj = try? JSONSerialization.jsonObject(with: data)
                {
                    t["inputSchema"] = obj
                }
                toolsOut.append(t)
            }
            entry["status"] = "ok"
            entry["tools"] = toolsOut
        } catch {
            entry["status"] = "error"
            entry["error"] = error.localizedDescription
            entry["tools"] = [[String: Any]]()
        }
        serversOut.append(entry)
    }

    for server in launched { await server.shutdown() }

    guard
        let data = try? JSONSerialization.data(
            withJSONObject: ["servers": serversOut], options: [.prettyPrinted, .sortedKeys])
    else {
        FileHandle.standardError.write(Data("mlx-agent tools: could not serialize dump\n".utf8))
        return 1
    }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
    return 0
}

// MARK: - CLI

func usage() {
    let text = """
        mlx-agent - native MLX ACP agent (chat + tools)

        USAGE:
          mlx-agent acp     [--backend mlx|openai] [--model <dir> | --base-url <url>]
                            [--mcp-config <json>] [--mode chat|agent] [guardrails]
                                                          ACP server over stdio (chat + agentic tools);
                                                          session/prime replaces the session context with a
                                                          supplied transcript (resume/fresh, see docs/session-prime.md)
          mlx-agent oneshot [--backend mlx|openai] [--model <dir> | --base-url <url>]
                            --prompt <text> [--mcp-config <json>]
                            [--auto-permission allow|deny] [guardrails]
                                                          run one agent turn, print to stdout
          mlx-agent gate    [--model <dir>]               built-in tool-calling gate
          mlx-agent chat    [--model <dir>] --prompt <text>  load + stream one completion
          mlx-agent map     --model <dir> --spool <dir> [--extra-eos-token <t>] [gen flags]
                                                          long-lived, spool-driven document map:
                                                          loads once, watches <spool>/job.json,
                                                          applies the job's per-chunk message
                                                          template ({{chunk}} placeholder)
                                                          independently to each chunk of the input;
                                                          output stitch (-> result.txt) reassembles
                                                          one document, collect (-> results.jsonl)
                                                          emits a per-chunk record. Translation,
                                                          proofread, rewrite, redact, classify,
                                                          extract are all just message templates.
          mlx-agent tools   --mcp-config <json>           launch the configured MCP servers and
                                                          dump their tool surface (exposed names,
                                                          descriptions, schemas, gating) as JSON;
                                                          loads no model
          mlx-agent bench   --model <dir> [--prompt-tokens <n>] [--gen-tokens <n>] [--runs <n>]
                            [--prefill-step <n>]
                                                          inference benchmark (Apple's M5 post
                                                          methodology: 4096-token prompt, 128
                                                          generated tokens; reports TTFT, prefill
                                                          and generation tok/s, peak memory;
                                                          --prefill-step overrides the library's
                                                          512-token prompt chunking; 2048 matches
                                                          mlx_lm's default, measured equal on M5)

        OPTIONS:
          --backend mlx|openai     generation engine for acp/oneshot (default: mlx).
                                   mlx    = in-process mlx-swift-lm on a safetensors --model
                                   openai = a remote OpenAI-compatible --base-url (llama-server);
                                            no model is loaded here and the RAM gate is skipped
          --base-url <url>         OpenAI-compatible endpoint root, required by --backend openai
                                   (e.g. http://127.0.0.1:8099/v1); its /health is checked at
                                   session/new. The MODEL is whatever that server was launched
                                   with - it is not switchable from here.
          --model <dir>            model directory (or set MLX_AGENT_MODEL; required by the
                                   mlx backend and by gate/chat/map)
          --prompt <text>          prompt for chat/oneshot mode
          --spool <dir>            spool directory for map mode (job.json in, status/result out)
          --mcp-config <json>      stdio-direct MCP server config (enables tools)
          --mode chat|agent        initial ACP mode (default: agent if --mcp-config, else chat)
          --auto-permission <x>    oneshot gate answer: allow | deny (default: deny)
          --extra-eos-token <t>    extra generation stop token, unioned into the model's stop set
                                   at load (repeatable); e.g. "<end_of_turn>" for a Gemma
                                   conversion whose generation_config.json omits it
          --idle-unload-seconds <n> acp/map: release the MLX model weights after <n> seconds
                                   without a prompt/job and reload transparently on the next
                                   one (default 600; 0 disables). Mirrors llama-server's
                                   --sleep-idle-seconds on the gguf path
          --system-prompt <text>   system instructions (acp/oneshot/chat); "" = no system
                                   message at all (default: helpful-assistant prompt)

        GENERATION (acp/oneshot/chat; an unset flag keeps the per-mode default shown):
          --temperature <f>        sampling temperature; 0 = greedy/deterministic
                                   (default: acp 0.7, oneshot/chat 0)
          --top-p <f>              nucleus sampling cutoff, 0 < p <= 1 (default: 1.0 = off)
          --max-new-tokens <n>     cap on generated tokens per turn
                                   (default: acp/oneshot 4096, chat 1024)
          --seed <n>               RNG seed; inert at temperature 0 (default: unset/random)
          --repetition-penalty <f> penalty on repeated tokens, ~1.1 curbs greedy loops
                                   (default: unset/off)

        GUARDRAILS:
          --max-tool-iters <n>     max tool passes per turn (default: 10)
          --tool-timeout <sec>     per-tool-call timeout (default: 60)
          --tool-result-bytes <n>  max tool result bytes fed back (default: 32768)

        EXAMPLES:
          mlx-agent acp --mcp-config /path/to/mcp.json
          mlx-agent acp --backend openai --base-url http://127.0.0.1:8099/v1 --mcp-config mcp.json
          mlx-agent acp --system-prompt "" --temperature 0 --max-new-tokens 2048
          mlx-agent oneshot --prompt "Read /etc/hosts" --mcp-config mcp.json --auto-permission allow
          mlx-agent gate
        """
    print(text)
}

func option(_ name: String, in args: [String]) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

func intOption(_ name: String, in args: [String]) -> Int? { option(name, in: args).flatMap(Int.init) }
func doubleOption(_ name: String, in args: [String]) -> Double? {
    option(name, in: args).flatMap(Double.init)
}

/// --idle-unload-seconds, shared by acp and map. A supplied-but-unparseable (or negative /
/// non-finite) value is a hard usage error rather than a silent fall-back to the default:
/// an operator who typed "5m" must not run with 10 minutes of resident weights believing
/// their value took effect.
func idleUnloadSecondsOption(in args: [String]) -> Double {
    guard let raw = option("--idle-unload-seconds", in: args) else { return 600 }
    guard let v = Double(raw), v.isFinite, v >= 0 else {
        FileHandle.standardError.write(
            Data("invalid --idle-unload-seconds \"\(raw)\": expected a number of seconds (0 disables)\n".utf8))
        exit(2)
    }
    return v
}

/// Resolve the system prompt from the CLI:
///   - flag absent            -> `defaultSystemPrompt` (helpful-assistant)
///   - `--system-prompt ""`   -> nil (NO system message is prepended at all)
///   - `--system-prompt <s>`  -> `s` verbatim
/// The nil case is what a translator wants: TranslateGemma carries its instruction in the
/// user turn, and a stray "You are a helpful assistant" prefix would pollute every request.
func resolveSystemPrompt(_ args: [String]) -> String? {
    guard let provided = option("--system-prompt", in: args) else { return defaultSystemPrompt }
    return provided.isEmpty ? nil : provided
}

/// Collect every `--extra-eos-token <token>` occurrence (the flag is repeatable). Each token
/// is unioned into the model's generation stop set at load time (see loadModel). Empty values
/// are ignored. Generic: useful for any model whose conversion shipped a broken stop set.
func parseExtraEOSTokens(_ args: [String]) -> Set<String> {
    var out: Set<String> = []
    var i = 0
    while i < args.count {
        if args[i] == "--extra-eos-token", i + 1 < args.count {
            let token = args[i + 1]
            if !token.isEmpty { out.insert(token) }
            i += 2
        } else {
            i += 1
        }
    }
    return out
}

/// Generation knobs resolved from the CLI. Each field is optional: a nil field means "leave
/// this mode's built-in default untouched", so a mode a flag does not target keeps its
/// long-standing behavior. `apply(to:)` overlays only the fields the user actually set.
struct GenConfig {
    var temperature: Float?
    var topP: Float?
    var maxNewTokens: Int?
    var seed: UInt64?
    var repetitionPenalty: Float?

    func apply(to base: GenerateParameters) -> GenerateParameters {
        var p = base
        if let temperature { p.temperature = temperature }
        if let topP { p.topP = topP }
        if let maxNewTokens { p.maxTokens = maxNewTokens }
        if let seed { p.seed = seed }
        if let repetitionPenalty { p.repetitionPenalty = repetitionPenalty }
        return p
    }
}

/// Parse the generation flags. Values are lightly clamped to their sane domains so a
/// fat-fingered negative can never reach the sampler: temperature/penalty floor at 0,
/// top-p is confined to (0, 1], and max-new-tokens is at least 1.
func parseGenConfig(_ args: [String]) -> GenConfig {
    var c = GenConfig()
    if let v = doubleOption("--temperature", in: args) { c.temperature = Float(max(0, v)) }
    if let v = doubleOption("--top-p", in: args) { c.topP = Float(min(1, max(0.0001, v))) }
    if let v = intOption("--max-new-tokens", in: args) { c.maxNewTokens = max(1, v) }
    if let v = option("--seed", in: args).flatMap(UInt64.init) { c.seed = v }
    if let v = doubleOption("--repetition-penalty", in: args) {
        c.repetitionPenalty = Float(max(0, v))
    }
    return c
}

func parseGuardrails(_ args: [String]) -> AgentGuardrails {
    var g = AgentGuardrails()
    if let v = intOption("--max-tool-iters", in: args) { g.maxToolIterations = max(1, v) }
    if let v = doubleOption("--tool-timeout", in: args) { g.toolTimeout = max(1, v) }
    if let v = intOption("--tool-result-bytes", in: args) { g.resultByteBudget = max(256, v) }
    return g
}

func parseAutoPermission(_ args: [String]) -> PermissionOutcome {
    switch option("--auto-permission", in: args)?.lowercased() {
    case "allow": return .allow
    case "deny", "reject": return .deny
    default: return .deny  // safe default: never auto-approve a mutation
    }
}

let cliArgs = Array(CommandLine.arguments.dropFirst())
let mode = cliArgs.first
// Model dir comes from --model, else the MLX_AGENT_MODEL environment variable. There is
// no built-in default: the modes that need a model require one of the two.
let resolvedModelDir = option("--model", in: cliArgs) ?? defaultModelDir

func requireModelDir() -> String {
    guard let dir = resolvedModelDir else {
        FileHandle.standardError.write(
            Data("no model directory: pass --model <dir> or set MLX_AGENT_MODEL\n".utf8))
        exit(2)
    }
    return dir
}

/// Resolve `--backend mlx|openai` (default mlx) into the engine the acp/oneshot modes run.
/// The model directory is required ONLY by the mlx path - resolving the engine before
/// asking for it is what lets `--backend openai` run with no --model at all (its weights
/// live in llama-server, which the applet owns).
func resolveEngine(_ args: [String]) -> EngineSpec {
    switch option("--backend", in: args)?.lowercased() ?? "mlx" {
    case "mlx":
        return .mlx(modelDir: requireModelDir())
    case "openai":
        guard let raw = option("--base-url", in: args), let url = URL(string: raw),
            url.scheme != nil, url.host != nil
        else {
            FileHandle.standardError.write(
                Data(
                    "--backend openai requires --base-url <url>, e.g. http://127.0.0.1:8099/v1\n"
                        .utf8))
            exit(2)
        }
        return .openai(baseURL: url)
    case let other:
        FileHandle.standardError.write(
            Data("unknown --backend \"\(other)\": expected mlx or openai\n".utf8))
        exit(2)
    }
}

let extraEOSTokens = parseExtraEOSTokens(cliArgs)

do {
    switch mode {
    case "acp":
        await ACPServer(
            engine: resolveEngine(cliArgs),
            mcpConfigPath: option("--mcp-config", in: cliArgs),
            guardrails: parseGuardrails(cliArgs),
            initialMode: option("--mode", in: cliArgs),
            systemPrompt: resolveSystemPrompt(cliArgs),
            gen: parseGenConfig(cliArgs),
            extraEOSTokens: extraEOSTokens,
            // MLX-only idle-unload (mirrors llama-server --sleep-idle-seconds). Default 600s; 0
            // disables. Ignored on the openai backend (llama-server does its own idle sleep).
            idleUnloadSeconds: idleUnloadSecondsOption(in: cliArgs)
        ).serve()
    case "oneshot":
        guard let prompt = option("--prompt", in: cliArgs) else {
            FileHandle.standardError.write(Data("oneshot mode requires --prompt <text>\n".utf8))
            exit(2)
        }
        let code = try await runOneshot(
            engine: resolveEngine(cliArgs), prompt: prompt,
            mcpConfigPath: option("--mcp-config", in: cliArgs),
            guardrails: parseGuardrails(cliArgs),
            autoPermission: parseAutoPermission(cliArgs),
            systemPrompt: resolveSystemPrompt(cliArgs),
            gen: parseGenConfig(cliArgs), extraEOSTokens: extraEOSTokens)
        exit(Int32(code))
    case "gate":
        let failures = try await runGate(model: requireModelDir(), extraEOSTokens: extraEOSTokens)
        exit(Int32(failures))
    case "chat":
        guard let prompt = option("--prompt", in: cliArgs) else {
            FileHandle.standardError.write(Data("chat mode requires --prompt <text>\n".utf8))
            exit(2)
        }
        try await runChat(
            model: requireModelDir(), prompt: prompt,
            systemPrompt: resolveSystemPrompt(cliArgs),
            gen: parseGenConfig(cliArgs), extraEOSTokens: extraEOSTokens)
    case "map":
        guard let spool = option("--spool", in: cliArgs) else {
            FileHandle.standardError.write(
                Data("map mode requires --spool <dir>\n".utf8))
            exit(2)
        }
        await MapServer(
            modelDir: requireModelDir(), spoolDir: spool,
            gen: parseGenConfig(cliArgs), extraEOSTokens: extraEOSTokens,
            // Same idle-unload policy as the acp path (and llama-server's sleep): default 600s,
            // 0 disables. The map broker outlives its jobs by design, so without this a single
            // translation left the weights resident until the owner app quit.
            idleUnloadSeconds: idleUnloadSecondsOption(in: cliArgs)
        ).run()
    case "tools":
        guard let cfg = option("--mcp-config", in: cliArgs) else {
            FileHandle.standardError.write(Data("tools mode requires --mcp-config <json>\n".utf8))
            exit(2)
        }
        exit(Int32(await runToolsDump(mcpConfigPath: cfg)))
    case "bench":
        let code = try await runBench(
            model: requireModelDir(),
            promptTokens: max(1, intOption("--prompt-tokens", in: cliArgs) ?? 4096),
            genTokens: max(1, intOption("--gen-tokens", in: cliArgs) ?? 128),
            runs: max(1, intOption("--runs", in: cliArgs) ?? 3),
            prefillStep: intOption("--prefill-step", in: cliArgs).map { max(64, $0) },
            extraEOSTokens: extraEOSTokens)
        exit(Int32(code))
    default:
        usage()
    }
} catch {
    FileHandle.standardError.write(Data("mlx-agent: \(error.localizedDescription)\n".utf8))
    exit(1)
}
