# mlx-agent

Native Swift local-LLM agent built on Apple's first-party
[`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm). It speaks
[ACP](https://agentclientprotocol.com) (Agent Client Protocol) over stdio to a UI and
[MCP](https://modelcontextprotocol.io) over stdio to tools: a UI drives it with
`session/prompt`, and in agent mode it runs a tool loop against MCP servers with a
permission gate before any mutating tool.

## Modes

```
mlx-agent acp     [--model <dir>] [--mcp-config <json>] [--mode chat|agent] [guardrails]
mlx-agent oneshot [--model <dir>] --prompt <text> [--mcp-config <json>]
                  [--auto-permission allow|deny] [guardrails]
mlx-agent chat    [--model <dir>] --prompt <text>
mlx-agent gate    [--model <dir>]
```

- **acp** - ACP server over stdio (chat + agentic tools). See below.
- **oneshot** - run one agent turn and print it; non-interactive, for scripts and tests.
  `--auto-permission` answers the permission gate (default `deny`).
- **chat** - load a model and stream a single completion.
- **gate** - a self-contained tool-calling correctness check (a handful of fixed cases),
  useful to catch a regression in the underlying engine without the full ACP stack.

Guardrails (`acp` and `oneshot`): `--max-tool-iters <n>` (10), `--tool-timeout <sec>`
(60), `--tool-result-bytes <n>` (32768).

### System prompt and generation flags (`acp`, `oneshot`, `chat`)

`--system-prompt <text>` overrides the default helpful-assistant system prompt. An
explicitly empty value (`--system-prompt ""`) prepends **no** system message at all - the
mode a translator wants, where the whole instruction is composed into the user turn and a
stray system prefix would only pollute the request.

The generation flags overlay each mode's built-in defaults; a flag left unset keeps that
mode's long-standing value. The defaults differ by mode:

| flag | `acp` | `oneshot` | `chat` |
|---|---|---|---|
| `--temperature <f>` - sampling temperature; `0` = greedy/deterministic | `0.7` | `0` | `0` |
| `--max-new-tokens <n>` - cap on tokens generated per turn | `4096` | `4096` | `1024` |
| `--top-p <f>` - nucleus sampling cutoff, `0 < p <= 1` | `1.0` (off) | `1.0` (off) | `1.0` (off) |
| `--seed <n>` - RNG seed (inert at `temperature 0`, where argmax has no RNG) | unset | unset | unset |
| `--repetition-penalty <f>` - penalize repeats; `~1.1` curbs greedy loops | unset (off) | unset (off) | unset (off) |

For example, a deterministic translation server that carries its instruction in the user
turn runs as `mlx-agent acp --system-prompt "" --temperature 0 --max-new-tokens 2048`.

### ACP server (`acp`)

Newline-delimited JSON-RPC 2.0 over stdin/stdout. Implements `initialize`, `session/new`
(returns a `model` select, plus a `mode` select when tools are configured),
`session/prompt` (streams `agent_message_chunk` and, for thinking models,
`agent_thought_chunk` split out of `<think></think>`), `session/cancel`, and
`session/set_config_option` (switch model, or toggle chat/agent mode). One `ChatSession`
per process; its KV cache persists across prompts. All logging is on stderr; stdout is
JSON-RPC only.

In agent mode a prompt runs the tool loop: it streams `tool_call` / `tool_call_update` /
`usage_update`, and sends `session/request_permission` before any gated tool.

Smoke-test the chat path without a UI:

```
python3 tools/acp_smoke.py .build/xcode/Build/Products/Debug/mlx-agent
```

### Tools (agent mode)

Pass `--mcp-config <path>` to connect to one or more MCP stdio servers and expose their
tools to the model. The config format is documented in
[`docs/mcp-config.md`](docs/mcp-config.md) with an example in
[`Examples/mcp-config.example.json`](Examples/mcp-config.example.json). Tools listed in a
server's `gatedTools` require user permission before each call; all others dispatch
directly.

## Building (IMPORTANT: xcodebuild, not `swift build`)

MLX's Metal shaders CANNOT be compiled by SwiftPM's command line. `swift build`
links a binary that aborts at runtime with "Failed to load the default metallib".
The build must go through `xcodebuild`, which compiles the shaders into
`mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` next to the binary.

One-time setup on Xcode 26 (the Metal compiler is a separate component):

```
xcodebuild -downloadComponent MetalToolchain     # ~688 MB, once per machine
```

Build:

```
xcodebuild -scheme mlx-agent -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath .build/xcode -configuration Debug \
  -skipPackagePluginValidation -skipMacroValidation build
```

The `-skip...` flags trust the `CudaBuild` package plugin and the swift-syntax
macros (otherwise xcodebuild blocks on a validation prompt). The binary lands at
`.build/xcode/Build/Products/Debug/mlx-agent` with the metallib bundle beside it.

Run:

```
cd .build/xcode/Build/Products/Debug && ./mlx-agent gate
```

## Dependency pins

- `mlx-swift-lm` 3.31.4 (MLXLLM, MLXLMCommon, MLXHuggingFace)
- `mlx-swift` 0.31.6 (MLX; pinned `.upToNextMinor(from: 0.31.4)` to match mlx-swift-lm)
- `swift-transformers` 1.3.x (Tokenizers) - mlx-swift-lm 3.x decoupled the tokenizer
  integration into an OPT-IN dependency the consumer must supply; the
  `#huggingFaceTokenizerLoader()` macro expands to `Tokenizers.AutoTokenizer`.
  `swift-huggingface` 0.9.0 comes in transitively (the `HuggingFace`/Hub client).
- `swift-sdk` (MCP) 0.9.0+ - MCP stdio tool clients (one per configured server).

The graph resolves cleanly against mlx-swift-lm's `swift-syntax 602..<604`
constraint (resolves 603.0.2 with prebuilt macro binaries).
