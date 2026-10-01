# The engine's sandbox (`--sandbox-profile`)

mlx-agent can confine itself with a macOS sandbox (Seatbelt) profile before it reads a model, an
MCP config or a prompt:

```
mlx-agent acp --backend mlx --model <dir> --sandbox-profile profile.json
```

The profile is applied once, at startup, and cannot be loosened afterwards. It covers this
process and every process it starts, the MCP servers included. A profile that cannot be read, or
that the system refuses, ends the process with status 2: it never starts unconfined when
confinement was asked for.

`--sandbox-print` (with `--sandbox-profile`) prints the profile in Seatbelt's policy language and
exits, so a host can show or test what it is about to ask for.

## Why

A model's output decides which tools are called and with what arguments. Tool servers have their
own limits; this one is for the process that holds the model and the conversation. With a
profile, a mistake or a hostile prompt cannot make that process read the user's files or reach the
network.

## The profile file

A JSON object. Every key is optional, and an unknown key is an error: a misspelled key in a
security profile must not be read as "nothing asked for". The shape is
[replay](https://github.com/abra-code/replay)'s, with more keys.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `read_only` | folders | none | Readable, with everything under them. |
| `read_write` | folders | none | Readable and writable, with everything under them. |
| `read_only_files` | paths | none | That path only, readable. |
| `read_write_files` | paths | none | That path only, readable and writable. |
| `exec_files` | paths | none | Programs this process may start, and read. |
| `allow_exec` | bool | `false` | May start any program. |
| `allow_fork` | bool | `true` | May fork. Starting a child needs it. |
| `allow_network` | bool | `false` | Every network operation. |
| `network_connect` | `"localhost:<port>"` | none | Outgoing connections to a port of this Mac. |
| `unix_socket_connect` | paths | none | Connections to these Unix sockets. |
| `gpu` | bool | `false` | What Metal needs (below). |
| `foundation_models` | bool | `false` | What Apple's on-device model needs (below). |
| `mach_services` | names | none | System services reachable by name. |
| `iokit_user_clients` | class names | none | Driver connections by class. |
| `import_baseline` | bool | `true` | Apple's `bsd.sb` baseline: the dynamic linker, `/dev`, basic Mach ports. |
| `extra_rules` | strings | none | Raw rules in Seatbelt's language, appended last. |

Always granted: reading this executable's own folder (the Metal shader bundle sits beside it) and
the LaunchServices preferences the system reads at startup.

A path is absolute, or starts at the home folder (`~/...`); a relative path is an error. Paths
are resolved to real paths (symbolic links followed), since the kernel compares real paths. A path
that resolves to `/` or to the home folder itself is dropped with a warning: grant the folders
inside that are needed. A folder that is inside another granted folder is not listed twice.

`--sandbox-profile` given twice, or with no value, ends the process.

The network is denied unless `allow_network` is true. `network_connect` and `unix_socket_connect`
open single destinations in that denial; a host name other than `localhost` is an error, because
Seatbelt can filter by port but not by remote host. One destination of Apple's baseline stays
open under the denial: the system log's socket (`/private/var/run/syslog`).

How rules combine, as measured on macOS 27: a rule with a filter (a path, a port) wins over a rule
without one, in either order, and between two rules with filters the later one wins. So a raw
`(deny ...)` in `extra_rules` does not take back an earlier allow that names a path or a port.

## Per backend

**mlx** (the model runs in this process, on the GPU):

```json
{ "gpu": true, "read_only": ["/Users/me/.cache/huggingface/hub/models--org--name"] }
```

Grant the model's whole repository folder, not the snapshot folder: a Hugging Face snapshot is
links into the repository's `blobs` folder. `gpu` grants the two driver connections Metal opens on
Apple silicon, its three cache folders under the per-user cache folder, and leave to hand its
separate compiler service access to those caches and to the shader bundle. Without `gpu`, Metal
sees no device and the process ends with an error that names neither Metal nor the sandbox.

**openai** (the model runs in a server on this Mac):

```json
{ "network_connect": ["localhost:8099"] }
```

**foundation** (Apple's on-device model):

```json
{ "foundation_models": true }
```

It grants the system service that runs the model and the two preference domains the framework
reads to learn whether the model can be used. The model itself runs in that service, outside this
process and outside this sandbox.

## MCP servers

A server mlx-agent starts inherits the profile, and a process under a profile cannot apply a
second one. So:

- the profile must let the server start (`exec_files`, or `allow_exec`) and read what it needs
  (its own files, an interpreter's library);
- a server that confines itself at startup will fail to, or will run with only the inherited
  profile. Either leave mlx-agent unconfined when its servers confine themselves, or give this
  profile everything those servers need and have them skip their own.

`--mcp-config`'s file must be readable: list it in `read_only_files`.

## Other files a session may need

- `--digest-dir`: a `read_write` folder.
- A model reload after an idle unload reads the model folder again, so it must stay granted.
- A SIGHUP reload of the MCP servers reads the config file again.

## Testing

`tools/sandbox_smoke.py` runs the tool under profiles and checks observed effects: a denied
network, a child that cannot read or write outside the granted folders, and each backend answering
under its own rules. The profile text is unit-tested in `Tests/SandboxProfileTests`.

To see what a profile refuses, watch the system log while the process runs:

```
log stream --style compact --predicate 'sender == "Sandbox" AND eventMessage CONTAINS "mlx-agent"'
```
