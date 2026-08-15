#!/usr/bin/env python3
# Bridge tests for `--backend foundation` with MCP tools - the schema walker, mostly.
#
# Stands up a REAL MCP server (below, over stdio) whose tools exist to be awkward: nested
# objects, arrays of objects, string enums, nullable unions, and four constructs the framework
# cannot express at all ($ref, allOf, a mixed enum, a typeless property). The point is not that
# the model calls them well - it is that every one of them BRIDGES, because the failure mode is
# silent: a tool whose schema cannot be built is dropped, and a dropped tool looks exactly like
# a model that chose not to call anything.
#
# What is asserted, and why each one:
#   1. Every tool bridges. No "dropped" line in the log for any of them.
#   2. The degradation summary names the four tools that lost something, and only those.
#   3. The tool-set token cost is reported, and is a real fraction of the 4096-token window.
#      This is the number that decides whether a registry is usable at all on this backend.
#   4. A nested-object tool actually DISPATCHES, with arguments that match its schema. Nesting
#      is the case most likely to be silently wrong - GenerationSchema takes a `dependencies`
#      array that inline nesting does not use, so "it compiles" proves nothing.
#   5. A degraded tool still dispatches. Degrading is only safe if the tool remains callable.
#
# Check 4 and 5 depend on the on-device model choosing to call a tool, which is not
# deterministic. Both prompts name the tool and its arguments explicitly, and both are retried
# once; a failure after that is reported as MODEL rather than BRIDGE so a bad day for the
# sampler does not read as a regression.
#
# Usage: python3 tools/fm_tools_smoke.py /path/to/mlx-agent
# Exit code = number of failed checks (0 = all good).

import json
import os
import subprocess
import sys
import tempfile

PY = sys.executable

# The MCP server, written to a temp file and launched by the agent. Kept inline so this smoke
# is one file with no fixtures to keep in step.
SERVER = r'''
import asyncio, json, sys

TOOLS = [
    {
        "name": "plain",
        "description": "A tool with no arguments at all.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        # Bounds at the integer limits. `exclusiveMinimum` becomes `>= n + 1`, and at Int.max
        # that arithmetic used to overflow and abort the process (exit 133) while the tool set
        # was being built - before the user's prompt was even read. A schema is server-controlled
        # input, so this is the hostile case, not a hypothetical one.
        "name": "limits",
        "description": "Bounds sitting exactly on the integer limits.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "hi": {"type": "integer", "exclusiveMinimum": 9223372036854775807},
                "lo": {"type": "integer", "exclusiveMaximum": -9223372036854775808},
                "huge": {"type": "integer", "minimum": 9223372036854775808},
            },
        },
    },
    {
        # Returns ~31 KB, which is INSIDE the agent's 32 KB result budget and eight times the
        # whole 4096-token window. Without a model-facing clamp this throws
        # exceededContextWindowSize, and the recovery path cannot help: the oversized text is in
        # the framework's transcript, never in seedHistory, so condensing summarizes the wrong
        # thing and the retry calls the same tool again.
        "name": "bulk",
        "description": "Return a large block of text.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "nested",
        "description": "Create a record for a person and their address.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "description": "Who this is",
                    "properties": {
                        "name": {"type": "string", "description": "Full name"},
                        "age": {"type": "integer", "minimum": 0, "maximum": 130},
                    },
                    "required": ["name"],
                },
                "tags": {
                    "type": "array",
                    "description": "Labels for this record",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["person"],
        },
    },
    {
        "name": "choices",
        "description": "Set a unit and a nullable note.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "note": {"type": ["string", "null"], "description": "Optional note"},
            },
            "required": ["unit"],
        },
    },
    {
        "name": "deep",
        "description": "An array of objects, each holding an array.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "groups": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "values": {"type": "array", "items": {"type": "number"}},
                        },
                        "required": ["label"],
                    },
                }
            },
            "required": ["groups"],
        },
    },
    {
        # Nested schema names are built from the property path, so "a" holding an object with
        # "b" wants the name t_a_b - and so does the sibling property literally called "a_b".
        # The framework does NOT reject the collision: it emits one $defs entry and points both
        # properties at it, so "q" disappears and a_b is silently constrained to {z: string}.
        # Underscore-separated MCP property names make this ordinary rather than exotic.
        "name": "collide",
        "description": "Store two records whose nested schema names would collide.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {
                        "b": {
                            "type": "object",
                            "properties": {"z": {"type": "string"}},
                            "required": ["z"],
                        }
                    },
                    "required": ["b"],
                },
                "a_b": {
                    "type": "object",
                    "properties": {"q": {"type": "integer"}},
                    "required": ["q"],
                },
            },
            "required": ["a", "a_b"],
        },
    },
    # The four that must DEGRADE rather than fail.
    {
        "name": "with_ref",
        "description": "Uses a $ref, which has no equivalent.",
        "inputSchema": {
            "type": "object",
            "properties": {"item": {"$ref": "#/definitions/thing"}},
            "required": ["item"],
        },
    },
    {
        "name": "with_allof",
        "description": "Uses allOf, which has no equivalent.",
        "inputSchema": {
            "type": "object",
            "properties": {"x": {"allOf": [{"type": "string"}, {"minLength": 2}]}},
        },
    },
    {
        "name": "mixed_enum",
        "description": "An enum that is not all strings.",
        "inputSchema": {
            "type": "object",
            "properties": {"level": {"enum": [1, 2, "high"]}},
        },
    },
    {
        "name": "typeless",
        "description": "A property with no type at all.",
        "inputSchema": {
            "type": "object",
            "properties": {"anything": {"description": "Any JSON value"}},
        },
    },
]


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        method, rid = req.get("method"), req.get("id")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "torture", "version": "1.0"}}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            name = req["params"]["name"]
            args = req["params"].get("arguments", {})
            if name == "bulk":
                send({"jsonrpc": "2.0", "id": rid, "result": {"content": [
                    {"type": "text", "text": "The deploy code is ZEBRA-7. " + ("filler. " * 4000)}]}})
                continue
            # Echo the arguments back: the test is what the model was able to construct.
            send({"jsonrpc": "2.0", "id": rid, "result": {"content": [
                {"type": "text", "text": json.dumps({"called": name, "args": args})}]}})
        elif rid is not None:
            send({"jsonrpc": "2.0", "id": rid, "result": {}})


main()
'''

DEGRADE = {"with_ref", "with_allof", "mixed_enum", "typeless"}
CLEAN = {"plain", "nested", "choices", "deep", "collide", "bulk", "limits"}

failures = 0


def check(ok, label, detail=""):
    global failures
    print(("  ok   " if ok else "  FAIL ") + label + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        failures += 1


def run(agent_bin, config, prompt, permission="allow", extra=()):
    proc = subprocess.run(
        [agent_bin, "oneshot", "--backend", "foundation", "--mcp-config", config,
         "--prompt", prompt, "--auto-permission", permission, *extra],
        capture_output=True, text=True, timeout=300)
    return proc.stdout, proc.stderr


def main():
    if len(sys.argv) < 2:
        print("usage: fm_tools_smoke.py /path/to/mlx-agent")
        return 2
    agent_bin = sys.argv[1]

    workdir = tempfile.mkdtemp(prefix="fm-tools-")
    server_path = os.path.join(workdir, "torture_server.py")
    with open(server_path, "w") as f:
        f.write(SERVER)
    config_path = os.path.join(workdir, "mcp-config.json")
    with open(config_path, "w") as f:
        json.dump({"servers": [{"name": "t", "command": PY, "args": [server_path]}]}, f)

    print("bridging:")
    proc = subprocess.run(
        [agent_bin, "oneshot", "--backend", "foundation", "--mcp-config", config_path,
         "--prompt", "Say hello. Do not call any tools.", "--auto-permission", "deny"],
        capture_output=True, text=True, timeout=300)
    err = proc.stderr

    # 0. The process survived the tool set at all. Exit 133 here means a schema killed it while
    # bridging - which is what a bound sitting on the integer limits used to do, before any
    # prompt was read. Checked first because every check below reads its output.
    check(proc.returncode == 0, "a hostile schema does not abort the process",
          f"exit={proc.returncode} (133 = trap)")

    # 1. Nothing dropped.
    dropped = [line for line in err.splitlines() if "dropped, its schema" in line]
    check(not dropped, "every tool bridges", "; ".join(dropped))

    # 2. The degradation summary names exactly the four that lose something.
    summary = [line for line in err.splitlines() if "simplified" in line]
    check(len(summary) == 1, "one degradation summary is logged", repr(summary))
    if summary:
        named = {name for name in DEGRADE | CLEAN if name + " [" in summary[0]}
        check(named == DEGRADE, "the summary names exactly the degrading tools",
              f"named={sorted(named)}")

    # 3. The window cost is reported and is a real fraction of it.
    cost = [line for line in err.splitlines() if "token window" in line]
    check(len(cost) == 1, "the tool-set token cost is logged", repr(cost))
    if cost:
        tokens = int(cost[0].split("cost ")[1].split(" ")[0])
        check(200 < tokens < 4096, f"the cost is plausible ({tokens} tokens)", str(tokens))

    # 4-6. Dispatch. Model-dependent, so named explicitly and retried once. `want` is a
    # substring that must appear in the echoed arguments, which is what makes check 6 a test of
    # the SCHEMA rather than of the model: a collided name drops the property entirely, so the
    # model cannot send it however well it is prompted.
    for label, prompt, tool, want in [
        ("a nested-object tool dispatches",
         "Call the nested tool with person name Ada and age 36, and tags 'engineer'.",
         "nested", '"name"'),
        ("a degraded tool dispatches",
         "Call the typeless tool with anything set to the text hello.", "typeless", None),
        # Both sides of the collision: the loser used to vanish, but a regression that broke the
        # FIRST claimant instead would pass a check that only looked for "q".
        ("colliding nested schema names stay distinct",
         "Call the collide tool with a.b.z set to hello and a_b.q set to 7.", "collide", '"z"'),
        ("the other side of the collision survives too",
         "Call the collide tool with a.b.z set to hello and a_b.q set to 7.", "collide", '"q"'),
    ]:
        called, matched, seen = False, False, ""
        for _ in range(2):
            out, err = run(agent_bin, config_path, prompt)
            for line in (out + err).splitlines():
                if '"called"' in line and tool in line:
                    called = True
                    seen = line
                    if want is None or want in line:
                        matched = True
            if matched:
                break
        if want is None:
            check(called, label, f"MODEL: the on-device model never called {tool}")
        else:
            check(called and matched, label,
                  f"called={called} args={seen[:160]}" if called
                  else f"MODEL: the on-device model never called {tool}")

    # 7. A tool result far larger than the window is still USABLE. Unclamped, a 31 KB result is
    # eight times the whole window and the turn dies with exceededContextWindowSize before the
    # model reads a word of it.
    #
    # --max-new-tokens 300 is not tuning the test to pass: with the ACP default of 4096 this
    # model answers by echoing the filler back until the RESPONSE fills the window, which is a
    # different failure (mid-answer overflow, already un-retryable by design) and would make this
    # check report the clamp broken when it is not.
    out, err = run(agent_bin, config_path, "Call the bulk tool and tell me the deploy code.",
                   extra=("--max-new-tokens", "300"))
    # Assert on STDOUT, which carries the model's answer. Asserting on out+err would be vacuous:
    # the console delegate dumps the full unclamped tool output to stderr before the clamp runs,
    # so "ZEBRA-7" appears there whether or not the clamp exists at all.
    check("ZEBRA-7" in out, "a 31 KB tool result is clamped and still answerable",
          [line for line in err.splitlines() if "window" in line or "failed" in line][:2])
    check("the model is shown 1" in err, "the clamp is logged and actually shortened the result",
          "no clamp line; did modelFacingResultBytes change?")
    # ... and the CLIENT still got the whole thing. This is the half the clamp must not break.
    check(len(err) > 30000, "the client's copy of the result is not truncated",
          f"stderr was only {len(err)} bytes")

    print(f"\n{failures} failed check(s)")
    return failures


if __name__ == "__main__":
    sys.exit(main())
