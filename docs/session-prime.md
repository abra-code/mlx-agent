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
