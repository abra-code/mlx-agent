# MCP config format (`--mcp-config`)

In agent mode, mlx-agent connects to one or more [Model Context
Protocol](https://modelcontextprotocol.io) servers over stdio and offers their tools to
the model. The set of servers is described by a JSON file passed with `--mcp-config
<path>`. mlx-agent owns this format; any producer must emit it.

## Schema

```json
{
  "servers": [
    {
      "name": "local",
      "command": "/absolute/path/to/server",
      "args": ["--flag", "value"],
      "env": { "KEY": "VALUE" },
      "gatedTools": ["write_file", "execute_command"]
    }
  ]
}
```

| Field        | Required | Type              | Meaning |
|--------------|----------|-------------------|---------|
| `name`       | yes      | string            | Unique label for the server (used in logs and to disambiguate tool-name collisions). |
| `command`    | yes      | string            | Executable to spawn: an absolute path, or a name resolvable on `PATH`. |
| `args`       | no       | array of string   | Arguments passed to `command` (argv). Defaults to none. |
| `env`        | no       | object            | Extra environment variables, merged over the inherited environment. |
| `gatedTools` | no       | array of string   | Tool names (the server's REAL tool names) that require user permission before each call. |

## Behavior

- Each server is spawned as a child process. mlx-agent performs the MCP handshake
  (`initialize` + `tools/list`) at startup and calls `tools/call` per dispatch.
- The tools of all servers are unioned and offered to the model. If two servers export
  the same tool name, the first server keeps the bare name and later ones are exposed as
  `<name>__<tool>`; routing maps the exposed name back to the real one.
- A tool whose real name is in that server's `gatedTools` triggers an ACP
  `session/request_permission` round-trip before it runs. In `oneshot` mode the answer
  comes from `--auto-permission allow|deny` (default `deny`). All non-gated tools dispatch
  directly.
- A server that fails to launch or hand-shake is logged and skipped; its tools are simply
  absent (one broken server does not disable the agent).
- `mlx-agent tools --mcp-config <json>` introspects a config without loading a model:
  it spawns the servers, performs the same handshake and exposed-name collision rules,
  prints the resulting tool surface (exposed names, descriptions, input schemas, gating,
  per-server handshake status) as JSON on stdout, and shuts the servers down. GUI
  inspectors (MLXChat's "Inspect MCP Servers" window) consume this dump.

## Guardrails

Per-turn tool behavior is bounded (all CLI-configurable):

- `--max-tool-iters <n>` (default 10) - max model/tool passes per turn.
- `--tool-timeout <sec>` (default 60) - per-tool-call timeout.
- `--tool-result-bytes <n>` (default 32768) - tool results larger than this are truncated
  before being fed back to the model.

A duplicate tool call (same name and arguments within one turn) short-circuits to the
cached result without re-dispatching.

## Example

See [`Examples/mcp-config.example.json`](../Examples/mcp-config.example.json).

The `command`/`args` for a given server come from that server's own documentation (for a
Python-packaged server, typically `python3 -m <module>` with `PYTHONPATH` set; for a
standalone binary, its path plus its stdio-server flag). The config is plain JSON, so any
tool or script that knows your servers can produce it.
