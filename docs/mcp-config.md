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
- A second server with a name already listed is skipped, with a line in the log.
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

## Reloading the servers (SIGHUP)

In ACP mode, `SIGHUP` makes mlx-agent read the `--mcp-config` file again and bring the
running servers in line with it, without ending the session: the loaded model, the
conversation and the session's standing permissions stay. A host uses it to change what a
server may do while a conversation runs, for example to give a file server another allowed
folder: it rewrites the file (write a new file and rename it over the old one, since the
agent may read at any moment) and sends the signal.

- A server whose `command`, `args` and `env` are unchanged keeps running. A change of
  `gatedTools` alone restarts nothing; the new gating applies to the next call.
- A server that changed is started anew, and the old process is stopped only once the new
  one answered. If the new one cannot start, the old one keeps serving and the failure is
  logged.
- A server no longer listed is stopped; a new one is started.
- A file that cannot be read or parsed changes nothing.
- The reload runs between turns. A signal that arrives during a turn or a summarization
  waits for its end; a prompt that arrives during a reload waits for the reload.
- When the tools offered to the model changed (a name, a description or a schema), the
  backend is given the new list, but the model's context still describes the old tools until
  the session is primed or started again. Changing only what a server is allowed to do
  changes no tool, so nothing needs priming. "Always allow" answers given in the session
  are forgotten when the tools changed, since a name may now belong to another server.
- Each reload writes one line to standard error, for example
  `reload: MCP servers follow mcp-config.json: restarted local; the tools are unchanged`.
- Before any server started (no session yet) the signal does nothing: the first session
  reads the file as it is then. `oneshot` and `tools` do not handle the signal.

`tools/acp_reload_smoke.py` exercises all of this with two fake servers.

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
