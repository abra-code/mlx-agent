# mlx-agent

Native Swift local-LLM agent built on Apple's first-party
[`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm). It speaks
[ACP](https://agentclientprotocol.com) (Agent Client Protocol) over stdio to a UI and
[MCP](https://modelcontextprotocol.io) over stdio to tools: a UI drives it with
`session/prompt`, and in agent mode it runs a tool loop against MCP servers with a
permission gate before any mutating tool.

## Modes

```
mlx-agent acp       [--model <dir>] [--mcp-config <json>] [--mode chat|agent] [guardrails]
mlx-agent oneshot   [--model <dir>] --prompt <text> [--mcp-config <json>]
                    [--auto-permission allow|deny] [guardrails]
mlx-agent chat      [--model <dir>] --prompt <text>
mlx-agent translate --model <dir> --spool <dir> [--extra-eos-token <t>] [gen flags]
mlx-agent gate      [--model <dir>]
mlx-agent bench     --model <dir> [--prompt-tokens <n>] [--gen-tokens <n>] [--runs <n>]
                    [--prefill-step <n>]
```

- **acp** - ACP server over stdio (chat + agentic tools). See below.
- **oneshot** - run one agent turn and print it; non-interactive, for scripts and tests.
  `--auto-permission` answers the permission gate (default `deny`).
- **chat** - load a model and stream a single completion.
- **translate** - long-lived, spool-driven translator. See below.
- **gate** - a self-contained tool-calling correctness check (a handful of fixed cases),
  useful to catch a regression in the underlying engine without the full ACP stack.
- **bench** - inference speed benchmark using the methodology of Apple's M5 MLX post
  (machinelearning.apple.com/research/exploring-llms-mlx-m5): a 4096-token prompt, 128
  generated tokens, warmup plus 3 timed runs; reports TTFT, prefill tok/s, generation
  tok/s, and peak GPU memory, plus a `RESULT_JSON:` line for machine comparison.
  `tools/bench_reference.sh` wraps it with the reference model (Qwen3-8B-4bit) for
  running across machines. Numbers are only comparable on AC power on a COOL machine:
  a fanless Air throttles the GPU by ~40% for minutes after a sustained CPU load
  (e.g. a build).

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

### Extra stop tokens (`--extra-eos-token`, all model-loading modes)

`--extra-eos-token <token>` (repeatable) unions a token into the model's generation stop
set at load time, without editing the downloaded model files. It exists because some
converted checkpoints ship a `generation_config.json` with a broken/omitted `eos_token_id`
- e.g. an instruction-tuned Gemma conversion that emits `<end_of_turn>` (id 106) to end a
turn but never lists it as EOS, so generation runs to `--max-new-tokens` every time.
Passing `--extra-eos-token "<end_of_turn>"` restores a clean stop. Harmless if the model's
stop set is already correct (a set union). Implemented by carrying the token through
`ModelConfiguration.extraEOSTokens`, which the engine converts to a token id and adds to
the stop set at generation time.

### Map server (`map`)

A long-lived, file-spool-driven **map over document chunks**: it loads the model once, then
applies one prompt independently to each chunk of a long input dropped into a spool
directory - `map(f, chunks)`, where `f` is a per-chunk LLM call. Each chunk generates with a
fresh cache (stateless, no cross-chunk context), so the operation is element-wise and
order-independent. Translation, proofreading, rewriting, redaction, normalization,
per-chunk classification/extraction, and the map phase of map-reduce summarization are all
just different **message templates** - the task lives in the job, not in the agent.

Unlike the other modes it **bypasses `ChatSession`** (whose message content must be a plain
`String`) and drives generation directly, so a job can carry **structured content** - e.g. a
translation template's `{type, source_lang_code, target_lang_code, text}` - and render it
against the model's own unmodified chat template.

```
mlx-agent map --model <dir> --spool <dir> [--extra-eos-token <t>] [gen flags]
```

The job carries a chat-message template with a `{{chunk}}` placeholder; per chunk the server
substitutes the chunk text for `{{chunk}}` (at the parsed-object level, so no escaping is
needed), applies the model's template, and generates. Two output modes:

- **stitch** (default) - reassemble the per-chunk outputs into one document, preserving the
  verbatim inter-chunk separators. For text->text transforms (translate, proofread, rewrite,
  redact, normalize) where the result is a new version of the same document.
- **collect** - emit one record per chunk instead of stitching. For text->data maps
  (classify, tag, extract) or when the caller does its own reduce/reassembly (e.g.
  map-reduce summarization).

Spool protocol (writes are atomic - temp file + rename - except the append-only jsonl):

- **in** `job.json` - `{"epoch": N, "text": "...", "budget_tokens": 1200, "output":
  "stitch"|"collect", "messages": [ <chat messages, with "{{chunk}}" somewhere> ]}`.
  Producers write it via atomic rename; a new job is one with a higher `epoch`.
  `budget_tokens` (default 1200) caps the per-chunk source token count. In place of inline
  `text`, a job may set `"text_file": "<name>"` to read the source from that file in the spool
  dir (basename only) - handy for a shell producer avoiding JSON-encoding, and for large inputs.
- **in** `stop` - an empty flag file; requests cancellation of the current job. Checked
  between chunks and during generation (an in-flight chunk is cancelled).
- **out** `status.json` - `{"state": loading|ready|mapping|done|cancelled|error, "epoch": N,
  "chunk": k, "total": N, "message": "..."}`.
- **out** `result.txt` (stitch) - the accumulated stitched output so far (grows per chunk).
- **out** `results.jsonl` (collect) - one `{"index", "source", "output", "sep"}` object per
  line, appended as each chunk finishes. `concat(source + sep)` over the records reproduces
  the input, so a caller can reassemble or align outputs itself.

The source is chunked with the **real tokenizer** (token-accurate) on paragraph then
sentence boundaries, hard-splitting only pathological runs; inter-chunk separators are
preserved verbatim so paragraph structure at chunk boundaries survives exactly. The server
exits when the spool directory disappears (owner quit) or stdin hits EOF (parent death).

**Example - a deterministic translator** (the canonical stitch instantiation): write a
`job.json` with

```json
{"epoch": 1, "text": "...", "output": "stitch",
 "messages": [{"role": "user",
   "content": [{"type": "text", "source_lang_code": "de", "target_lang_code": "en",
                "text": "{{chunk}}"}]}]}
```

against `mlx-agent map --model <dir> --spool <dir> --extra-eos-token "<end_of_turn>"
--temperature 0 --max-new-tokens 2048`. A string-content instruct model instead takes e.g.
`"messages": [{"role": "user", "content": "Correct the grammar; output only the corrected
text:\n\n{{chunk}}"}]`.

### ACP server (`acp`)

Newline-delimited JSON-RPC 2.0 over stdin/stdout. Implements `initialize`, `session/new`
(`configOptions` is always empty - see below), `session/prompt` (streams
`agent_message_chunk` and, for thinking models, `agent_thought_chunk` split out of
`<think></think>`), `session/cancel`, and `session/set_config_option` (switches the model).
One `ChatSession` per process; its KV cache persists across prompts. All logging is on
stderr; stdout is JSON-RPC only.

**Nothing is advertised in `configOptions`, so no client renders a picker.** `model` and
`mode` both remain settable over the wire, and `--model` / `--mode` still work; only the
affordances are gone. Both decisions belong to the host: agentic-ness is fixed at spawn by
whether `--mcp-config` was passed, and the model is the host's to choose - it owns the
picker that knows about every engine, the tools choice that goes with a model, RAM
headroom, and the window's identity. An agent-side control for either could only disagree
with the host, and (for the model) switch it without the host ever knowing.

In agent mode a prompt runs the tool loop: it streams `tool_call` / `tool_call_update` /
`usage_update`, and sends `session/request_permission` before any gated tool. That request
offers ACP's standard four options (`allow_once`, `allow_always`, `reject_once`,
`reject_always`); an `always` choice is remembered for the rest of the SESSION and
short-circuits the round-trip, so no further request reaches the client for that tool. The
memory is per exposed tool name, is never persisted, and is cleared by `session/new`.

Smoke-test the chat path without a UI:

```
python3 tools/acp_smoke.py build/Build/Products/Debug/mlx-agent
```

`tools/acp_tool_leak.py` covers what `acp_smoke.py` structurally cannot: it drives a turn
that CALLS A TOOL and asserts no raw model markers (harmony `<|channel|>...`, `<think>`)
reach the visible message. Both regressions it guards only manifest across a tool call, so
they are invisible to a chat-only run. It needs a server and an MCP config:

```
python3 tools/acp_tool_leak.py build/Build/Products/Debug/mlx-agent \
  --base-url http://127.0.0.1:8080/v1 --mcp-config /path/to/mcp-config.json
```

### Tools (agent mode)

Pass `--mcp-config <path>` to connect to one or more MCP stdio servers and expose their
tools to the model. The config format is documented in
[`docs/mcp-config.md`](docs/mcp-config.md) with an example in
[`Examples/mcp-config.example.json`](Examples/mcp-config.example.json). Tools listed in a
server's `gatedTools` require user permission before each call; all others dispatch
directly. "Each call" is the default: the user can answer `allow_always` / `reject_always`
to settle a tool for the rest of the session. The decision is keyed on the tool name and
never on paths - where a tool may act is the sandbox's question, not this gate's.

## Building (Xcode project, generated by XcodeGen)

MLX's Metal shaders CANNOT be compiled by SwiftPM's command line. `swift build`
links a binary that aborts at runtime with "Failed to load the default metallib".
The build must go through `xcodebuild`, which compiles the shaders into
`mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` next to the binary.

Since the build has to be xcodebuild anyway, **`project.yml` is the single source of
truth** - targets, dependencies, pins and settings. There is no `Package.swift`: a
package plus a project meant two dependency graphs and two `Package.resolved` files
that could silently drift apart. `mlx-agent.xcodeproj` is generated from `project.yml`
and **is committed** (it carries the shared schemes and the resolved pins).

One-time setup on Xcode 26 (the Metal compiler is a separate component):

```
xcodebuild -downloadComponent MetalToolchain     # ~688 MB, once per machine
brew install xcodegen                            # only to regenerate the project
```

After editing `project.yml`, regenerate and commit the result:

```
xcodegen generate
```

Build:

```
xcodebuild -project mlx-agent.xcodeproj -scheme mlx-agent \
  -destination 'platform=macOS,arch=arm64' -configuration Debug \
  -derivedDataPath build \
  -skipPackagePluginValidation -skipMacroValidation build
```

The `-skip...` flags trust the `CudaBuild` package plugin and the swift-syntax
macros (otherwise xcodebuild blocks on a validation prompt). The binary lands at
`build/Build/Products/Debug/mlx-agent` with the metallib bundle beside it.

Run:

```
cd build/Build/Products/Debug && ./mlx-agent gate
```

Tests (`Chunking` + `AgentText` - no MLX, no Metal, ~0.01s):

```
xcodebuild -project mlx-agent.xcodeproj -scheme UnitTests \
  -destination 'platform=macOS,arch=arm64' \
  -skipPackagePluginValidation -skipMacroValidation test
```

Both are Foundation-only libraries, deliberately kept OUT of the MLX-linking `mlx-agent`
target so their logic is testable without a Metal build. That is the bar for putting pure
logic in its own target: `ThinkSplitter` lived in `ACPServer.swift`, was therefore
untestable in seconds, and shipped a bug for it. Anything that does not need MLX belongs
in a library target with tests.

### Code coverage: never ship an instrumented binary

Both schemes pin coverage OFF, and that matters more than it sounds. **Coverage is a
scheme-level setting, not a per-target one** - it applies to the WHOLE build graph, so
every dependency gets instrumented alongside the tool. Do not try to control it with
per-target `settings:` in `project.yml`: those never reach the SPM package targets, so
you get a MIXED binary (deps instrumented, product not) that is worse than either
extreme and nearly invisible. Command-line `CLANG_ENABLE_CODE_COVERAGE=NO` has exactly
that effect, and `-enableCodeCoverage NO` is rejected outside `test`.

An instrumented binary is ~35% larger, slower in a hot token loop, and dumps a multi-MB
`default.profraw` into its working directory on EVERY run - in the shipped app, wherever
the host happens to be running from.

This bit us before: this package used to be built via an auto-generated scheme (no
`Package.swift` scheme was checked in), and **adding a `.testTarget` silently switched
coverage on for the whole graph** - reproducible in a two-file package. That trap is gone
with a real project and committed schemes, but verify the shipped binary anyway:

```
otool -l <binary> | grep -c __llvm_prf_cnts     # 0 = clean
```

Use the section check above, NOT `nm | grep __llvm_prf`: nm reports 0 for a small
instrumented binary (false clean). The runtime check never lies - run the binary and
see whether a `default.profraw` appears next to it.

## Dependency pins

Version ranges live in `project.yml` (`packages:`); the exact resolved versions live in
`mlx-agent.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved`, which is
committed - that file is what makes a build reproducible, so review it in diffs.

- `mlx-swift-lm` 3.31.4 (MLXLLM, MLXLMCommon, MLXHuggingFace)
- `mlx-swift` 0.31.6 (MLX; `minorVersion: 0.31.4`, i.e. up-to-next-minor, to match mlx-swift-lm)
- `swift-transformers` 1.3.x (Tokenizers) - mlx-swift-lm 3.x decoupled the tokenizer
  integration into an OPT-IN dependency the consumer must supply; the
  `#huggingFaceTokenizerLoader()` macro expands to `Tokenizers.AutoTokenizer`.
  `swift-huggingface` 0.9.0 comes in transitively (the `HuggingFace`/Hub client).
- `swift-sdk` (MCP) 0.9.0+ - MCP stdio tool clients (one per configured server).

The graph resolves cleanly against mlx-swift-lm's `swift-syntax 602..<604`
constraint (resolves 603.0.2 with prebuilt macro binaries).
