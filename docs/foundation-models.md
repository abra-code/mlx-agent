# `--backend foundation` - Apple's on-device system model as an engine

mlx-agent's third generation backend, after in-process MLX and a llama-server over
`--base-url`. It runs Apple Foundation Models: the ~3B system model that ships with macOS 26 and
Apple Intelligence.

This document covers it as an ENGINE - what it can and cannot do when it is generating your
answers. Using it to SUMMARIZE a conversation is a different job with different tradeoffs, and it
is covered in [`session-digest.md`](session-digest.md); the on-device model is a good summarizer
even where it is a poor chat engine, because summarization is bounded, structured, and read by
another model rather than by a person.

## What it is for

It needs no download, no weights resident, no unified-memory budget and no server. That makes it
the answer to two situations:

- **This Mac has no model installed yet.** An embedding app can offer something on first launch
  instead of a 4 GB download.
- **Short, bounded work.** Summaries, classifications, one-line answers.

It is NOT a replacement for a 7B-30B model under MLX. The window is 4096 tokens, it does not do
tool calling here, and it is a 3B model. Choose it on those terms.

Supported modes: `acp`, `oneshot`, `digest`. Refused by `map` (chunked document work does not fit a
4096-token window, and the refusal names the mode rather than failing later inside the broker) and
irrelevant to `chat`, `gate` and `bench`, which are MLX diagnostics.

## Availability

Three gates gate every reference to the framework, and all three are needed because the tool
ships as ONE arm64 binary with a macOS 14.0 deployment target while the framework is macOS 26.0:

| Gate | What it catches |
| --- | --- |
| `#if canImport(FoundationModels)` | an older Xcode with no module to compile against |
| weak link (`project.yml`) | dyld refusing to launch the binary at all on macOS 14-25 |
| `#available` + `FMAvailability.probe()` | macOS 26 with Apple Intelligence off, or assets not downloaded |

The runtime gate is the one that is easy to get wrong: `canImport` and `#available` both pass on a
macOS 26 Mac with Apple Intelligence switched off. Without asking the framework itself, that
failure arrives as a thrown error out of the first generation rather than as a clean message before
any work starts.

`--backend foundation` is therefore refused at STARTUP, exit 2, with a reason and a remedy:

```
--backend foundation is not usable here: Apple Intelligence is switched off; turn it on in System Settings > Apple Intelligence & Siri, then retry
```

The three unavailable reasons are genuinely different situations, so they are never collapsed into
one string:

| Situation | What you get |
| --- | --- |
| ineligible hardware | "this Mac is not eligible for Apple Intelligence; the on-device model cannot be enabled on this hardware" |
| Apple Intelligence off | "Apple Intelligence is switched off; turn it on in System Settings > Apple Intelligence & Siri, then retry" |
| assets not downloaded | "the model assets are not downloaded yet; leave the Mac on power and network for a while, then retry" |

One is permanent, one is a settings toggle, one just needs a wait.

### `mlx-agent fm-check`

Ask the question without running anything expensive:

```
mlx-agent fm-check                    # availability, context size, languages, one real generation
mlx-agent fm-check --prompt "..."     # use your own prompt for the live pass
mlx-agent fm-check --digest "<text>"  # summarize instead: proves GUIDED generation works
```

It reports two different things, and the distinction matters to a caller:

- **availability** - a settings, hardware and asset fact, obtained without inference.
- **a live pass** - proof that a generation actually completes. Availability alone does not
  guarantee it: assets can report ready and still fail on first use.

Exit 0 only when a real generation completed, so `if mlx-agent fm-check >/dev/null; then` is a
valid probe. A `RESULT_JSON:` line follows the `bench` convention for scripts.

`--digest` is the stronger check. It exercises a custom `@Generable` schema, the permissive
guardrails and the `.general` use case TOGETHER, which is the combination that actually breaks -
each of those three has broken the pair before, and a plain generation succeeding proves none of
it.

Loads no MLX model and reads no config, so it is safe on a machine with no model installed.

## The window

**4096 tokens, and that is not a placeholder.** `SystemLanguageModel.contextSize` back-deploys to a
hard-coded 4096 before macOS 26.4, so on older systems the number is a constant rather than a
measurement - but it is still the right number to plan against.

Everything about this backend follows from that. It holds the whole conversation, the system
prompt, the framework's own template, and the answer.

**Overflow is handled by summarizing, not by failing.** Before a pass, the backend estimates
whether the conversation plus a reserve still fits; if not, it condenses the older part into a
digest and rebuilds the session around it, then retries. If the condensation cannot help, the
overflow surfaces as a clean turn failure:

```
the conversation is larger than the on-device model's 4096-token context window
```

**With one exception: overflow raised mid-answer is not retried.** Once any text has streamed to
the client it cannot be un-sent, so a second attempt would append a fresh answer to a partial one
and the reader would see two half-replies stitched together. That case fails with the message
above, which is honest - the request cannot be satisfied by making the history smaller, because it
is the RESPONSE that filled the window. The proactive estimate above exists to keep a turn out of
that state in the first place.

The reserve is `min(--max-new-tokens, window / 4) + 256`. The quarter-window cap is deliberate: the
ACP default of 4096 is a "no practical limit" sentinel rather than a prediction, and reserving all
of it would fire this check on every turn with any history at all - condensing conversations that
fit comfortably. A caller who asks for SHORT answers still gets the smaller reserve, which is the
case where the number means something.

The summarizer for that path is always the on-device model itself, opening its own session. It
cannot be the session's own backend: that call is reached from INSIDE a pass, and `PassGate` is
non-reentrant and non-cancellable, so it would deadlock permanently with no timeout and no cancel
to break it. See `Sources/mlx-agent/BackendDigest.swift`.

`--digest-dir` records these condensations alongside the prime-time ones.

## The tool budget

**There is none: tools are not bridged.** The framework calls tools ITSELF inside
`respond`/`streamResponse` and offers no mode that surfaces a call for external dispatch, so
mlx-agent's tool loop cannot drive it - and that loop is where the permission gate, the iteration
cap, the per-call timeout and result-size budget live. Bridging MCP tools means moving all of that
inside the framework's call, which is its own change.

So: **agent mode degrades to chat.** Setting tools on this backend logs once and is otherwise
ignored, rather than pretending. Concretely, with `--backend foundation`:

- `--mcp-config` still launches the servers and still selects agent mode, but the model never sees
  a tool and never calls one.
- `session/request_permission` is never sent.
- Tool turns in a primed history are DROPPED when translating into the framework's transcript.
  `Transcript` can express them, but replaying a tool exchange to a model that cannot run tools
  invites it to invent more of them.

Use `mlx-agent tools --mcp-config <json>` to inspect a tool surface, and an MLX or llama-server
backend to actually run one.

**And when the bridge does land, the budget is still the window.** Every tool definition - name,
description and JSON schema - is spent from the same 4096 tokens the conversation lives in. With a
dozen schema-heavy MCP tools, most of the window is gone before the user types anything. That is a
property of the model, not of the bridge, so it will not be fixed by building the bridge: expect
this backend to work with a handful of small tools, and use MLX or llama-server for a real
registry.

## Deliberately not built

Recorded so the questions are not reopened without new information:

- **FM as a general replacement for MLX in long chat or `map`.** The 4096-token window forces small
  chunks and forbids long context; MLX models run 32k and up. MLX remains the main engine.
- **Auto-summarization mid-turn on the MLX and llama-server backends.** Their windows are large,
  and silently rewriting history on a backend that does not need it costs traceability for no
  benefit. Overflow condensation exists ONLY inside this backend, where the window forces it.
- **An intent-extraction pre-pass ahead of tool routing.** The agent loop already routes on
  model-emitted tool calls. A classifier in front of it adds latency and a second opinion that can
  disagree with the model doing the dispatching, and fixes no measured failure.

## Generation flags

| Flag | On this backend |
| --- | --- |
| `--temperature` | mapped; `0` selects greedy |
| `--top-p` | mapped, as `random(probabilityThreshold:)` when below 1 |
| `--seed` | **honored** - both random sampling modes take one, so a SAMPLED run can be reproduced (at `--temperature 0` it is inert, and greedy is reproducible anyway) |
| `--max-new-tokens` | mapped to `maximumResponseTokens`, and also feeds the overflow reserve above |
| `--repetition-penalty` | no equivalent; DROPPED, and logged once at init rather than silently |
| `--system-prompt` | becomes the session's instructions; `""` supplies none |

Inert, because they describe things this backend does not have: `--model` (the model belongs to
the OS), `--extra-eos-token` (no tokenizer of ours), `--idle-unload-seconds` (no weights of ours to
release), and the RAM gate (nothing to fit).

`session/set_config_option` with `model` is refused with `-32601` and the reason "the foundation
backend has a single OS-provided model". Silently succeeding would be a lie.

## Guardrails and languages

Ordinary chat runs under the framework's DEFAULT guardrails. Summarization does not: it uses
`.permissiveContentTransformations`, because the default guardrails refuse ordinary documents -
measured, with an innocuous paragraph coming back as `guardrailViolation`. A summarizer that
declines the user's own conversation is useless, and summarizing is a transformation of text the
user already has rather than generation of new content.

Language support is smaller than people assume, and a prompt in an absent language fails outright
with `unsupportedLanguageOrLocale` rather than degrading. `fm-check` prints both the locale count
and the distinct language codes, because `supportedLanguages` holds LOCALES (`en-AU`, `en-GB`,
`en-US`; `zh-Hans-CN`, `zh-Hant-HK`, `zh-Hant-TW`), so reducing it to primary language codes gives
a materially smaller number.

## Verify

```
mlx-agent fm-check
python3 tools/acp_smoke.py                   <mlx-agent> --backend foundation
python3 tools/acp_cancel_reprompt_smoke.py   <mlx-agent> --backend foundation 6
python3 tools/acp_overflow_condense_smoke.py <mlx-agent> --backend foundation
python3 tools/acp_prime_condense_smoke.py    <mlx-agent> --backend foundation --expect condensed
python3 tools/digest_smoke.py                <mlx-agent> --backend foundation
mlx-agent oneshot --backend foundation --prompt "In one sentence, what is a semaphore?"
```

## Two traps worth knowing before you change this code

Both are measured, both cost real time to find, and neither is visible in the API:

1. **Reusing a `LanguageModelSession` after a canceled `streamResponse` traps** (SIGTRAP,
   reproduced 14/20) and kills the process. The backend rebuilds its session after any pass that
   did not end cleanly. Do not optimize that away.
2. **A canceled exchange is PURGED from the framework's own transcript.** That is why this backend
   owns its history as `[Chat.Message]` rather than reading it back out of the session - doing the
   latter would silently lose exactly the turns that need preserving.
