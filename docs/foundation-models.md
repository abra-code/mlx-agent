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

It is NOT a replacement for a 7B-30B model under MLX. The window is 4096 tokens, tool definitions
are spent from it, and it is a 3B model. Choose it on those terms.

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
mlx-agent fm-check                      # availability, context size, languages. No generation.
mlx-agent fm-check --test-prompt        # ...and one real generation, with the built-in probe text
mlx-agent fm-check --test-prompt "..."  # ...with your own probe text
```

It reports two different things, and `--test-prompt` chooses between them rather than merely
decorating one:

- **availability** - a settings, hardware and asset fact, obtained without inference. This is the
  bare command. ~0.05 s, so it is cheap enough to sit on a UI path that asks it every time a menu
  opens.
- **a live pass** - proof that a generation actually completes. Availability alone does not
  guarantee it: assets can report ready and still fail on first use. ~0.3 s, because it is a real
  generation.

**The cheap answer is the default because it is the question with an impatient caller.** A caller
that wants proof is by definition willing to wait for it, so the expensive mode is the explicit
one. Passing `--test-prompt` with no text after it still generates, using the built-in probe -
which is greedy-sampled, so repeat runs read identically and it works as a smoke test.

The name says what the text is FOR: it is a fixture chosen to prove a pass completes, not a
question whose answer anyone consumes. `--prompt` means the opposite elsewhere in this CLI - in
`oneshot` and `chat` it is the actual work - and reusing that spelling for a probe was a collision
worth one extra word.

Exit 0 means "usable", and what that is worth follows the mode: bare, that availability said yes;
with `--test-prompt`, that a generation completed. `if mlx-agent fm-check >/dev/null; then` is a
valid gate either way. A `RESULT_JSON:` line follows the `bench` convention for scripts.

Loads no MLX model and reads no config, so it is safe on a machine with no model installed.

#### Branch on `reason`, show `detail`

```
mlx-agent fm-check
RESULT_JSON: {"available":true,"detail":"not generated (no --test-prompt)","probe":"availability","reason":"available"}

mlx-agent fm-check --test-prompt
RESULT_JSON: {"available":true,"detail":"ok","generated":"...","ms":574,"probe":"generation","reason":"available"}
```

`probe` says which question was answered: `availability` or `generation`. It exists because
`reason` cannot say - `available` is emitted either way, so a consumer holding only this line (a
log, a telemetry record, anything that did not choose the argv) would otherwise read a bare
availability check as proof of a completed pass. `ms` and `generated` are absent without a
generation, but inferring from a missing key is exactly what this contract asks callers not to do.

`detail` is prose for a person and gets reworded; `reason` is a stable code and is the contract.
An app that decides whether to OFFER this engine needs to tell "never" from "not yet", which
`available: false` alone cannot say:

| `reason` | Means | What a caller should do |
| --- | --- | --- |
| `available` | usable; a generation completed too, if `--test-prompt` was given | offer it |
| `appleIntelligenceOff` | the one recoverable case | offer it, and point at System Settings |
| `modelNotReady` | assets still downloading | offer it, and say to retry shortly |
| `deviceNotEligible` | this hardware never will be | do not offer it |
| `osTooOld` | macOS < 26 | do not offer it |
| `notBuilt` | built against an SDK without the framework | do not offer it |
| `generationFailed` | availability said yes, the pass still failed | treat as unusable; `detail` says why |
| `unknown` | a reason added after this was written | unusable now, but do not report it as permanent |

`generationFailed` is the reason `--test-prompt` exists at all: every other code is answerable
without generating, and availability can report ready while the first generation still fails. It
is therefore the one code the bare command can never return - if you need to rule that case out,
you have to generate. Codes are added rather than renamed, so treat an unrecognized one as
"unusable for now" rather than hiding the feature.

### The deeper check, when a caller wants it

**`fm-check` is a gate, not a proof - even with `--test-prompt`.** Its live pass is a plain
`respond(to:)` with no schema, so it does not exercise the thing that actually breaks: a custom
`@Generable` type, the permissive guardrails and the `.general` use case TOGETHER. Each of those
three has broken the pair before, and a plain generation succeeding proves none of it.

That check is a real summarization, which is expensive, so it is not folded into the gate - it is
the second step, run only by a caller that wants it:

```
mlx-agent fm-check >/dev/null \
  && mlx-agent digest --backend foundation --in probe.json --keep-recent 2
```

**Note the gate is bare here.** `digest` performs a real generation, and a guided one, so adding
`--test-prompt` to the line above would pay for a plain generation to decide whether to run a
stronger one immediately after. The cheap gate is the right half of this pair.

**Copy the probe below rather than shrinking it, and keep the `--keep-recent 2`.** `digest` takes a
transcript, and the planner summarizes only what is OLDER than the verbatim tail - so a probe that
leaves nothing older exits 3 without ever calling the model, which reads exactly like a failure and
is not one. What decides it is not a message count but where the user turns fall: the tail boundary
snaps BACKWARD to the nearest user turn, and if it reaches index 0 the whole transcript becomes the
tail. All measured:

- `[user, assistant, user, assistant]` with `--keep-recent 2` summarizes. This is the probe below.
- `[user, assistant, assistant, assistant]` refuses - four messages, but the only user turn is
  first, so the boundary drags to 0.
- `[assistant, user, assistant]` summarizes, on three messages. There is no clean minimum to quote.
- The same four-message probe with the DEFAULT `--keep-recent 6` refuses, because six messages of
  tail is more than the file holds.

Four alternating messages from a user turn, with `--keep-recent 2`, is the smallest shape that is
reliably safe:

```json
[
  { "role": "user",      "content": "The installer ships Friday and the codename is QUETZAL-9." },
  { "role": "assistant", "content": "Noted: Friday, QUETZAL-9." },
  { "role": "user",      "content": "What is left before then?" },
  { "role": "assistant", "content": "Signing and notarization." }
]
```

The two cases are distinguishable without guessing, in the `RESULT_JSON:` line on stderr:

| | Exit | `ms` | `reason` |
| --- | --- | --- | --- |
| guided generation worked | 0 | the real elapsed time | absent |
| the model was never asked | 3 | `0` | "nothing to summarize ...", or "too large to summarize ..." |
| the model was asked and failed | 3 | the real elapsed time | "summarization failed ...", or "the summarizer returned an empty digest" |

**`ms` is the reliable signal, not the wording.** It is measured across the condensation only, so a
refusal the planner makes before calling anything reports 0 - branch on that, and treat `reason` as
text for a human. Matching on a reason prefix would misread an empty digest (a real model failure)
as a refusal.

Measured on the probe above: 4.7 s for the guided pass, against 0 ms for the shapes that never
reached the model.

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

## Tools

MCP tools work here, and the client-visible wire is the same as on the other backends: the same
`tool_call` and `tool_call_update` updates, the same `session/request_permission` round-trip, the
same denial text fed back to the model.

What differs is who is in charge. The framework calls tools ITSELF inside
`respond`/`streamResponse` and offers no mode that surfaces a call for external dispatch, so the
agent's loop cannot drive this backend. The bridge inverts the arrangement instead: the loop hands
the backend a runner, and each bridged tool calls back into it - so the permission gate, the
per-call timeout, the result-size budget and the duplicate-call short-circuit are literally the
same code that serves MLX and llama-server.

The differences are in timing and bounds rather than in the shape of what a client receives:

- **A turn is one pass.** The model calls tools, reads the results and answers inside a single
  `stream()`. `.toolCall` events are never emitted, so nothing double-dispatches.
- **Tools are called CONCURRENTLY.** Measured: a prompt naming three cities entered three tool
  calls in the same microsecond, and the pass took as long as the slowest rather than their sum.
  Every other backend dispatches one at a time. Two visible effects: several
  `session/request_permission` requests can be outstanding at once, where an MLX session would
  ask serially; and a call can still be running when the turn it belongs to is cancelled, since
  the per-call timeout is deliberately uncancellable. Neither loses work - a straggler's result
  is discarded rather than folded into the next turn.
- **The iteration cap is softer.** On the other backends the loop simply refuses to start another
  pass. Nothing can halt the framework mid-answer without discarding it, so past the cap a tool
  call returns `{"error": "tool call limit reached; answer with what you have"}` to the model
  rather than stopping the turn. The model is told, and the log records it - but nothing goes on
  the wire for a refused call, so a model looping past the cap is visible only in stderr.
- **Tool results are clamped: 1500 bytes per call, 6000 per attempt.** `--tool-result-bytes`
  defaults to 32 KB, which is eight times this model's entire window: a 31 KB result throws
  `exceededContextWindowSize` before the model reads a word of it, and no condensation can
  recover, because the oversized text is in the framework's transcript rather than in the history
  that gets summarized. The per-pass bound matters for the same reason - ten clamped results
  would still be most of the window. Past it, a call is refused with the same shape of message
  the iteration cap uses, without being dispatched. The client still receives every result in
  full; only what the model reads is shortened, and the log says when it happened.
- **A condense-and-retry gets fresh budgets.** Both the call cap and the byte budget are per
  ATTEMPT, not per turn, because a retry runs against a session rebuilt around the summary - the
  first attempt's tool output is not in its transcript, so billing the retry for it would refuse
  calls over window that is no longer occupied. A turn that condenses can therefore make up to
  twice the calls of one that does not.
- **Mid-turn tool-set changes are gone.** The set is fixed when the session is built.

Tool turns in a primed history are still DROPPED when translating into the framework's transcript:
a replayed call names a tool that may not be in this session's set, and showing the model itself
using a tool it does not have is worse than a gap.

### The budget is the window

Every tool definition - name, description and schema - is spent from the same 4096 tokens the
conversation lives in. Measured on this machine, with real servers:

| Tool set | Tokens | Share of the window |
| --- | --- | --- |
| mcp_server_time (2) + mcp_server_fetch (1) | 615 | 15% |
| the 11-tool schema-torture server in `tools/fm_tools_smoke.py` | 1235 | 30% |
| all three of those servers together (14 tools) | 1832 | 45% |

So: **a handful of small tools, not a registry.** That is a property of the model rather than of
the bridge, and building the bridge did not change it. Use MLX or llama-server for a real tool
surface. The backend logs the real number at startup, so there is no need to guess:

```
foundation backend: 3 tool(s) cost 615 of the 4096-token window (15%) before any conversation
```

### Schemas that do not fit

`DynamicGenerationSchema` cannot express all of JSON Schema. Rather than dropping a tool, the
bridge degrades the part it cannot build to a described string and says so once:

```
foundation tools: 4 of 8 tool(s) have simplified schemas - with_ref [item: uses $ref, which has
no equivalent], ...
```

Degrading is safe because the MCP server validates its own inputs authoritatively - the schema
only constrains decoding, so what is lost is guidance, not correctness. `$ref`, `allOf`/`oneOf`/
`anyOf`, mixed-type enums, genuine type unions and typeless properties degrade; `pattern`,
`format` and length bounds survive as sentences in the description; nested objects, arrays,
string enums, nullable types and numeric bounds convert exactly. The rules live in the `SchemaIR`
target and each one has a test.

A tool whose schema the framework rejects outright is dropped with a log line naming it, and the
rest of the set still works.

Use `mlx-agent tools --mcp-config <json>` to inspect a tool surface without a model.

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

Language support is smaller than people assume. `fm-check` prints both the locale count and the
distinct language codes, because `supportedLanguages` holds LOCALES (`en-AU`, `en-GB`, `en-US`;
`zh-Hans-CN`, `zh-Hant-HK`, `zh-Hant-TW`), so reducing it to primary language codes gives a
materially smaller number - 15 on this machine: `da de en es fr it ja ko nb nl pt sv tr vi zh`.

**An absent language DEGRADES; it does not fail cleanly.** Do not expect
`unsupportedLanguageOrLocale` to protect you - measured with Polish, which is not in the set
above, no prompt produced that error. What happens instead is one of three things, and which one
you get is not stable:

| Prompt (Polish) | Result |
| --- | --- |
| "Ile to jest dwa plus dwa?" | answered correctly, but **in English**: "Two plus two equals four." |
| "Mow po polsku. Co to jest pi?" | answered in Polish, fluent-looking and **factually nonsense** ("the proportion between the radius and the diameter of a square angle in a square circle") |
| "Powiedz mi dowcip o pi", "Napisz jedno zdanie o kotach" | began generating Polish, then died mid-stream with `guardrailViolation` |

The third is the trap, because it arrives as a SAFETY refusal for a request that is plainly
benign - "tell me a joke about pi". The output guardrail is itself a classifier, and low-quality
text in a language it does not cover appears to trip it. The same prompts in English and German
(both supported) answer normally, so it is the language rather than the content.

Ask the model whether it speaks a language and it will say yes. `supportedLanguages` is the
authority; the model's self-report is not.

## Verify

```
mlx-agent fm-check --test-prompt
python3 tools/fm_tools_smoke.py              <mlx-agent>
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
