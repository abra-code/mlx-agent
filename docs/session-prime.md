# session/prime - context replacement from a restored transcript

mlx-agent extension to the Agent Client Protocol. Lets a client replay a saved conversation into the live session so a resumed chat actually remembers its prior turns, or reset the context to fresh. mlx-agent owns this schema; ActionUIChat's ACP transport is one consumer.

## Capability

The `initialize` response advertises it inside `agentCapabilities`:

```json
"agentCapabilities": {
  "promptCapabilities": { "audio": false, "image": false, "embeddedContext": false },
  "sessionPrime": true
}
```

Clients MUST gate all `session/prime` traffic on this key; agents that do not advertise it never see the method.

## Request

```json
{
  "jsonrpc": "2.0", "id": 42, "method": "session/prime",
  "params": {
    "sessionId": "mlx-session-1",
    "messages": [
      { "role": "user",      "content": "..." },
      { "role": "assistant", "content": "...",
        "toolCalls": [ { "id": "tc-1", "name": "read_file", "arguments": { "path": "/etc/hosts" } } ] },
      { "role": "tool",      "content": "{...tool result...}", "toolCallId": "tc-1" }
    ]
  }
}
```

- `role`: `user` | `assistant` | `tool`. Unknown roles are skipped with a stderr log, never fatal.
- `toolCalls` (assistant only, optional): prior tool invocations, replayed into the chat template as `tool_calls`.
- `toolCallId` (tool only): must reference an id announced by the NEAREST PRECEDING assistant message's `toolCalls`; orphan tool messages are dropped (template well-formedness guard).
- No `system` role: the agent owns its system prompt and prepends it itself.

## Response

`{ "primed": <count of accepted messages> }`

## Errors

- `-32002` no active session (`session/new` has not run).
- `-32602` unknown `sessionId`, or `messages` missing / not an array.
- `-32003` a prompt is in flight. The agent refuses to swap the session under a running turn. A client that cancels a turn and primes immediately can see one transient `-32003` (the busy flag clears when the canceled prompt RESOLVES, a hair after `session/cancel` is processed); retry once after a short delay. A persistent `-32003` indicates a client ordering bug.

## Semantics

- REPLACES the session context; not additive. `messages: []` = fresh context (the reset a client should send on New Chat, or when the user opens a saved conversation read-only).
- Does not generate, does not touch model weights, mode, or the MCP tool registry. Near-instant.
- KV prefill of the primed history happens lazily on the NEXT `session/prompt` (the engine prefills `[system, history..., user]` in one pass). The first resumed turn pays the prefill latency and its `usage_update` includes the history token count.

## Optional: `condense`

A restored conversation can be summarized instead of replayed. Additive and opt-in - a prime with
no `condense` key behaves exactly as documented above, and an agent build that does not know the
key ignores it.

```json
{
  "jsonrpc": "2.0", "id": 42, "method": "session/prime",
  "params": {
    "sessionId": "mlx-session-1",
    "messages": [ ... ],
    "condense": { "keepRecentTurns": 6, "maxDigestTokens": 700, "backend": "session" }
  }
}
```

- `keepRecentTurns` (optional, default 6): trailing MESSAGES kept verbatim, not exchanges. Clamped
  to 2...64. The boundary is snapped backward to a user turn where one is nearby, so the tail may
  be one or two messages longer than asked.
- `maxDigestTokens` (optional, default 700): ceiling on the generated summary. Clamped to 128...2048.
- `backend` (optional): which model summarizes THIS restore - `auto`, `foundation`, `session` or
  `none`, the same four words as `--digest-backend`, which is the default when the key is absent.
  An empty (or all-whitespace) string and `null` are treated as absent, since a host storing "the
  user has not chosen" that way is ordinary. Any other value declines the condensation and says
  why in `reason` - an unrecognized word is quoted back, a value of the wrong type is named by
  type - rather than falling back to the flag and reporting a summarizer nobody chose. A
  malformed value is reported even on an agent running `--digest-backend none`, which otherwise
  refuses first.
- Slice sizing is NOT on the wire. It is a property of the summarizing model's context window,
  which the agent knows and the client does not.

**`backend` is why a client can offer this choice to a person at all.** The flag belongs to the
agent process, which outlives any one restore; the request belongs to the conversation in front of
the user. Without the request, someone who picks a summarizer is answered by whichever one the
agent happened to start with, and the only trace of the difference is the `summarizer` name in the
response - a mismatch that reads as the client ignoring them. The one value that does not yield is
`--digest-backend none`: that is the operator turning summarization off for this agent, and a
request that could switch it back on would make the flag advisory. A request under `none` is
refused with a reason saying so.

### Response, condensed

```json
{
  "primed": 8,
  "accepted": 70,
  "condensed": true,
  "summarizer": "apple-foundation-models",
  "digest": { "schemaVersion": 1, "createdAt": "...", "unresolvedIntent": "...", "establishedFacts": [...], ... },
  "dropped": { "turns": 64, "bytes": 18132 }
}
```

`primed` counts the REPLACEMENT history (preamble + acknowledgment + verbatim tail), not the input.

`accepted` is how many of the supplied messages survived the well-formedness filtering described
above - empty turns, orphan tool messages, unknown roles and a trailing unanswered tool
announcement are all dropped before anything else happens. **Do the arithmetic against `accepted`,
not against your own message count**: `primed == injected + (accepted - dropped.turns -
trimmed.turns)`, where `injected` is 1 or 2 and `trimmed.turns` is 0 unless the key below is
present. (The acknowledgment turn is omitted when the verbatim tail already begins
with a non-user turn, because adding it there would put two assistant turns back to back.) A client
that uses its own count computes a negative `injected` on any transcript with a filtered turn in
it.

`summarizer` names the MODEL that produced the digest - `mlx-community/Qwen3-4B`,
`gemma-4-31B-it-Q4_K_M`, `apple-foundation-models`. Worth surfacing: "the summary is thin" and "the
summary was made by a 3B on-device model" are the same observation from the client's side. It falls
back to the engine convention (`session:mlx`, `session:openai`) only where this agent cannot find
out what it is running: a model directory with no recognizable name, or an OpenAI-compatible server
that reports no `model_path` from `/props`.

### `trimmed`, when even the result did not fit

```json
{ "primed": 5, "condensed": true, "summarizer": "gemma-4-31B-it-Q4_K_M",
  "digest": { ... }, "dropped": { "turns": 64, "bytes": 18132 },
  "trimmed": { "turns": 3, "bytes": 9004 } }
```

Present only when the assembled history was still larger than the serving model's context and the
oldest turns after the preamble had to go. **`trimmed` is not `dropped`.** `dropped` counts turns
the digest stands in for; `trimmed` counts turns nothing stands in for, and a client that folds the
two together will show a marker claiming those turns were summarized. Surface it separately, or at
minimum say the conversation was shortened.

A prime with NO `condense` key is never trimmed. Nothing was asked to be summarized, the full
history is what the host requested, and a client that never sends `condense` has no place to read
`trimmed` from - so silently shortening the conversation there would be a loss with nowhere to
report it. Such a restore can still overflow at the next prompt; what it gets is the turn error,
which now names both token counts and what to do about them.

The window it is measured against is the serving model's own: `/props` on the openai engine and
`config.json` on mlx. The FOUNDATION engine is deliberately absent: its backend recovers from an
overflow by summarizing rather than by dropping turns, and trimming here would destroy the content
before the part that knows how to keep it ever saw it. When nothing states a window, nothing is
trimmed -
trimming against a guess would discard conversation to satisfy a number nobody gave.

### Response, not condensed

```json
{ "primed": 70, "condensed": false, "reason": "summarization is disabled (--digest-backend none)" }
```

**The fallback is full fidelity.** Every failure - no summarizer, Apple Intelligence off, the model
refused, timed out, produced an empty digest, or the conversation was too large to slice within
budget - primes the COMPLETE history and says why. A client that ignores `condensed` still gets a
correct session; it just gets a slower first turn.

The one exception is the case that made that rule expensive. A conversation too large to summarize
is usually also too large to PRIME, and priming it anyway produced an HTTP 400 from llama-server on
the first message afterwards - a dead session, reported as a JSON object nobody could act on. So a
decline whose history does not fit the serving model's window is trimmed to it: `reason` gains a
clause naming how many messages were left out, and `trimmed` carries the numbers. Full fidelity
remains the rule wherever the history fits, and wherever no window is known.

### What changes for the client

- **Latency.** A condensing prime GENERATES, and how long depends on which model does it. Measured
  with Apple's on-device model: about 8 seconds for a 70-message, 19 KB conversation. Measured with
  the session's own model (gemma-4-E4B on an M5 Air): 7 seconds for the same shape, 166 seconds for
  a 240-message, 210 KB one - it takes far fewer passes but each is a full generation on a bigger
  model. A plain prime remains near-instant. Do not put a condensing prime on a path where the user
  is waiting without an indicator. A condensing prime gives up after 5 minutes and primes the full
  history - that is one budget for the whole request, whichever model summarizes and including
  `auto`'s fallback, so the wait has a ceiling even on a slow model.
- **Ordering.** The agent refuses a prompt, a model switch, an idle-unload and a second condensing
  prime for the whole condensation, with `-32003` and a message naming the summarization rather
  than a nonexistent prompt. The same cancel-then-prime ordering rule as before applies, with a
  longer window.
- **Superseding is allowed and safe.** `session/new` and a plain `session/prime` are NOT refused
  during a condensation - a New Chat should never fail. They win: the condensation notices the
  session was replaced and discards its result with `-32003` ("the session was replaced while its
  context was being summarized") rather than publishing a context the user already dismissed.
- **`session/cancel` reaches it.** Canceling during a condensation primes the full history,
  `condensed: false`. On the session's own model it interrupts the pass in flight; on the
  on-device model it lands between passes, so allow one pass of latency there.
- **Fidelity.** The client owns the original transcript; the agent never persists it. Priming the
  full history again restores full fidelity, so a digest is never a one-way door.

### When to ask for it

Request `condense` on restore when the stored conversation is large, and on a model switch (which
is a `set_config_option` followed by a re-prime - that IS the cross-model handoff path). Do not
request it for short conversations: the agent will decline with a reason, having wasted a round
trip, and a summary of six messages is worse than the six messages.

### Which model summarizes

Condensation is a capability of the agent, not of one backend. Two summarizers exist: Apple's
on-device model (macOS 26+, `apple-foundation-models`), and the session's own model, whatever it
is (`session:mlx`, `session:openai`). The wire contract is identical either way; only `summarizer`
tells them apart. What a digest KEEPS and DROPS is the same either way - same schema, and the same
verbatim tail, which follows from `keepRecentTurns` and nothing else. See
[`session-digest.md`](session-digest.md). What differs is how many PASSES the summarized half takes,
since each sizes its slices from its own context window; that is a cost, and past a limit it is
also a refusal. For the on-device model as an engine rather than a summarizer, see
[`foundation-models.md`](foundation-models.md).

`--digest-backend` sets the default, and defaults to `auto`; `condense.backend` names the
summarizer for one restore. Same four words either way:

| Value | Behavior |
| --- | --- |
| `auto` | the on-device model when this conversation fits its budget, else the session's own model, else decline |
| `foundation` | the on-device model only |
| `session` | the session's own model only |
| `none` | never summarize; `condense` primes everything and says so |

`auto` measures rather than prefers. The on-device model costs the session nothing - no weights,
no disturbance to the loaded model - but its 4096-token window turns a long conversation into many
sequential passes, and past its limit it refuses outright; a 32k-window model that is already
loaded does the same job in one or two. So the agent counts the passes THIS history would need on
the on-device budget and picks accordingly. Under `auto` only, an on-device attempt that fails
outright falls back to the session's model; an explicit choice - the flag or the request - never
silently switches.

Precedence is one line: the request wins, except that `--digest-backend none` wins over
everything - other than the complaint about a malformed value, which is a client bug worth naming
wherever it is sent. `auto` is `auto` whichever of the two says it, so a request naming it gets
the same fallback the flag would.

Sizing follows the summarizing model's context window: reported by the framework on the on-device
path, read from the model's `config.json` on the MLX path, and assumed to be 8192 on
`--backend openai`, where the window belongs to a llama-server this process did not start -
`--digest-window <tokens>` states it when that is wrong.

Three things `--digest-backend session` cannot do. It is refused on `--backend foundation` (the
session's model IS the on-device model there - use `foundation`); it declines while the MLX model
is idle-unloaded rather than reloading multiple gigabytes of weights in order to summarize; and it
declines when the model is configured to generate so much that nothing is left to summarize in.
That last one is reachable with `--backend openai --max-new-tokens 8192` against the assumed 8192
window: a summarization pass may generate up to `--max-new-tokens`, so that has to come out of the
window before anything else does. The reason names both ways out.

### Producing a digest without a session

`mlx-agent digest` applies the same summarization to a transcript FILE - same wire shape in, the
digest (or the preamble it renders to) out, no ACP session and no client. It is how a conversation
can be condensed once and stored, how two summarizers can be compared on the same input, and how
this feature is tested offline. See [`session-digest.md`](session-digest.md).

### Audit

`--digest-dir <dir>` records every condensation (prime-time and the backend's own overflow
condensations) as `digest-<unixms>.json` plus a `digests.jsonl` index. Each file holds the digest
AND the exact text it was made from, so what a summary dropped can be diffed after the fact. Off by
default: it writes conversation text to disk.
