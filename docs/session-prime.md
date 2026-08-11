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
- `-32003` a prompt is in flight. The agent refuses to swap the session under a running turn. A client that cancels a turn and primes immediately can see one transient `-32003` (the busy flag clears when the cancelled prompt RESOLVES, a hair after `session/cancel` is processed); retry once after a short delay. A persistent `-32003` indicates a client ordering bug.

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
    "condense": { "keepRecentTurns": 6, "maxDigestTokens": 700 }
  }
}
```

- `keepRecentTurns` (optional, default 6): trailing MESSAGES kept verbatim, not exchanges. Clamped
  to 2...64. The boundary is snapped backward to a user turn where one is nearby, so the tail may
  be one or two messages longer than asked.
- `maxDigestTokens` (optional, default 700): ceiling on the generated summary. Clamped to 128...2048.
- Slice sizing is NOT on the wire. It is a property of the summarizing model's context window,
  which the agent knows and the client does not.

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
not against your own message count**: `primed == injected + (accepted - dropped.turns)`, where
`injected` is 1 or 2. (The acknowledgment turn is omitted when the verbatim tail already begins
with a non-user turn, because adding it there would put two assistant turns back to back.) A client
that uses its own count computes a negative `injected` on any transcript with a filtered turn in
it.

`summarizer` names the model that produced the digest. Worth surfacing: "the summary is thin" and
"the summary was made by a 3B on-device model" are the same observation from the client's side.

### Response, not condensed

```json
{ "primed": 70, "condensed": false, "reason": "this session's backend cannot summarize yet; only --backend foundation can" }
```

**The fallback is full fidelity, never truncation.** Every failure - no summarizer, Apple
Intelligence off, the model refused, timed out, produced an empty digest, or the conversation was
too large to slice within budget - primes the COMPLETE history and says why. A client that ignores
`condensed` still gets a correct session; it just gets a slower first turn.

### What changes for the client

- **Latency.** A condensing prime GENERATES. Measured on Apple's on-device model: about 8 seconds
  for a 70-message, 19 KB conversation. A plain prime remains near-instant. Do not put a condensing
  prime on a path where the user is waiting without an indicator.
- **Ordering.** The agent refuses a prompt, a model switch, an idle-unload and a second condensing
  prime for the whole condensation, with `-32003` and a message naming the summarization rather
  than a nonexistent prompt. The same cancel-then-prime ordering rule as before applies, with a
  longer window.
- **Superseding is allowed and safe.** `session/new` and a plain `session/prime` are NOT refused
  during a condensation - a New Chat should never fail. They win: the condensation notices the
  session was replaced and discards its result with `-32003` ("the session was replaced while its
  context was being summarized") rather than publishing a context the user already dismissed.
- **`session/cancel` reaches it.** Cancelling during a condensation stops it within one slice and
  primes the full history, `condensed: false`.
- **Fidelity.** The client owns the original transcript; the agent never persists it. Priming the
  full history again restores full fidelity, so a digest is never a one-way door.

### When to ask for it

Request `condense` on restore when the stored conversation is large, and on a model switch (which
is a `set_config_option` followed by a re-prime - that IS the cross-model handoff path). Do not
request it for short conversations: the agent will decline with a reason, having wasted a round
trip, and a summary of six messages is worse than the six messages.

### Availability

Only `--backend foundation` can summarize today. Every other backend declines with a reason and
primes everything. A backend-independent summarizer that drives whichever model the session already
has is planned; the wire contract above does not change when it lands - only which values of
`summarizer` appear.

### Audit

`--digest-dir <dir>` records every condensation (prime-time and the backend's own overflow
condensations) as `digest-<unixms>.json` plus a `digests.jsonl` index. Each file holds the digest
AND the exact text it was made from, so what a summary dropped can be diffed after the fact. Off by
default: it writes conversation text to disk.
