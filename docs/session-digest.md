# The session digest - what it keeps, what it drops, and who can make one

A digest is a conversation compressed into structured notes, rendered back as an ordinary user
turn. It exists so a long saved conversation can be resumed without replaying every token of it.

Two things consume digests, and they are the same digest:

- `session/prime` with `condense` - the live path, documented in
  [`session-prime.md`](session-prime.md). A restored transcript is summarized instead of replayed.
- `mlx-agent digest` - the batch path, below. The same summarization applied to a file, offline.

**A digest is portable by construction.** It renders to prose, not to a format any engine has to
agree on, so one made on macOS 26 with Apple's on-device model primes into an MLX or llama-server
session on macOS 14. Nothing in the digest records what produced it beyond a name.

## What it keeps

Six fields, and they are the whole schema (`schemaVersion` 1):

| Field | What goes in it |
| --- | --- |
| `unresolvedIntent` | what the user is still trying to accomplish, one sentence |
| `establishedFacts` | specifics the conversation settled: names, paths, versions, numbers |
| `decisions` | choices made, with their reasons, so they are not reopened |
| `toolEvents` | which tools ran, what for, and what each established |
| `openThreads` | work explicitly left unfinished |
| `userPreferences` | constraints the user stated about how the work should be done |

Plus provenance: `generator`, `createdAt`, `sourceTurnCount`, and `sourceSHA256` over the exact
text the model read.

Alongside the digest, two things survive VERBATIM rather than being summarized:

- **The recent tail.** `keepRecentTurns` trailing messages (default 6) are spliced in unchanged, so
  "it" and "that file" in the next prompt still resolve against real text. The boundary snaps
  backward to a user turn where one is nearby, so the tail may be a message or two longer than
  asked.
- **System turns.** Instructions are hoisted to the front of the primed history untouched.
  Summarizing "never write to disk" into one bullet competing for a slot in a capped list is how a
  stated constraint quietly stops being one. This applies to an agent condensing its OWN history
  mid-turn; a supplied transcript has no system role at all (see below).

## What it drops

Everything else in the older part of the conversation, replaced by the preamble. The response says
how much: `dropped.turns` and `dropped.bytes`.

Lists are capped - `maxItemsPerList` per slice, scaled up to 3x for the merged digest - and when
the cap bites, items are taken round-robin across slices rather than in order, so a long
conversation's early section cannot fill the cap and silence its most recent one.

**The fidelity guarantee: a digest is never a one-way door.** The caller owns the original
transcript and the agent never persists it, so priming the full history again restores everything.

**And condensation never truncates.** Every failure - no summarizer, the model refused, timed out,
returned nothing usable, the work was canceled, or the conversation needed more slices than the
budget allows - resolves to the FULL history, with a reason. Condensing is an optimization;
truncating is data loss wearing an optimization's clothes.

## Who can produce one

Two summarizers, same output either way:

- **Apple's on-device model** (`apple-foundation-models`), macOS 26+ with Apple Intelligence. Costs
  no weights and does not disturb a loaded model, but its window is 4096 tokens, so a long
  conversation becomes many sequential passes. See
  [`foundation-models.md`](foundation-models.md); note that summarizing is the job it is best at
  even where it is a poor chat engine.
- **A regular generation backend** (`session:mlx`, `session:openai`) - a model that is already
  loaded, or one loaded for the purpose in batch mode. Far fewer passes, but each is a full
  generation.

On the live path the restore's `condense.backend` chooses, falling back to `--digest-backend`,
which defaults to measuring; see [`session-prime.md`](session-prime.md#which-model-summarizes). In
batch mode `--backend` names the summarizer directly, since no session is involved.

## `mlx-agent digest` - the batch mode

```
mlx-agent digest [--backend mlx|openai|foundation] [--model <dir> | --base-url <url>]
                 [--in <file>] [--out <file>] [--render]
                 [--keep-recent <n>] [--max-digest-tokens <n>] [--digest-window <n>]
```

Input is a transcript as JSON on stdin, or from `--in`. It is the SAME wire shape as
`session/prime`'s `messages`, accepted either as a bare array or as the whole params object, so a
captured request pastes straight in:

```json
[
  { "role": "user",      "content": "..." },
  { "role": "assistant", "content": "...",
    "toolCalls": [ { "id": "tc-1", "name": "read_file", "arguments": { "path": "/etc/hosts" } } ] },
  { "role": "tool",      "content": "{...}", "toolCallId": "tc-1" }
]
```

Messages are filtered exactly as a prime filters them - empty turns, orphan tool messages, unknown
roles and a trailing unanswered tool-call announcement are dropped - because a digest made offline
has to be made from the same text a session would use.

**There is no `system` role on this wire, here or at prime time**: the agent owns its system prompt
and prepends its own. A `{"role": "system"}` entry is an unknown role, so it is dropped with a log
line rather than hoisted, and instructions stored that way are lost from the digest. Put them in
`--system-prompt` on the session that consumes the digest instead. (This is also what keeps the
arithmetic below simple: `injected` is exactly 1 or 2, never more.)

**Only `messages` is read from the object form.** A captured request's `condense` block is ignored -
`backend` included; those knobs are `--backend`, `--keep-recent` and `--max-digest-tokens` here,
and honoring both would give one knob two sources. State them as flags if the captured values
matter.

Output on stdout, or to `--out`:

```json
{
  "condensed": true,
  "summarizer": "session:mlx",
  "accepted": 38,
  "primed": 8,
  "dropped": { "turns": 32, "bytes": 12904 },
  "digest": { "schemaVersion": 1, "unresolvedIntent": "...", "establishedFacts": [...], ... }
}
```

`accepted` is how many messages survived filtering; `primed` is how many the replacement history
would have. **Do the arithmetic against `accepted`**: `primed == injected + (accepted -
dropped.turns)`, where `injected` is 1 or 2. Against your own message count it does not close.

`--render` prints the preamble the digest renders to - the text that actually gets primed - instead
of the JSON, so a person can read what a model would see.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | condensed; the artifact is on stdout or in `--out` |
| 3 | not condensed - the reason is always on stderr, and goes to the artifact stream as `{"condensed": false, "reason": ...}` unless `--render` was asked for, since there is no preamble to render |
| 2 | usage: unreadable input, a flag that cannot take effect, or a model that cannot summarize |
| 1 | something failed: the model would not load, the server was unreachable, `--out` was unwritable |

**3 is separate from 1 on purpose.** "This conversation is not longer than the verbatim tail" and
"the model returned nothing usable" are answers, not crashes - the same fallback that primes the
full history in a live session. A script that treated them as failures would be wrong.

A `RESULT_JSON:` line follows the `bench` / `fm-check` convention, but goes to **stderr**: stdout
here is the artifact, so `mlx-agent digest --in chat.json | jq .digest` works without filtering.

### Batch bounds

A restore is something a user waits on; a batch run is not. So the limits are different: 32 slices
rather than 4, half an hour for the whole run rather than five minutes, and 300 s for a single pass
on a local model. Nothing in it holds a session busy, and there is no PassGate hazard - the backend
is built for this one job in a process that does nothing else.

Sizing follows the summarizing model's window, read from the model's `config.json` on the MLX path,
reported by the framework on the on-device path, and assumed to be 8192 on `--backend openai` where
the window belongs to a llama-server this process did not start. `--digest-window` states it when
that assumption is wrong. The mode logs what it settled on:

```
[digest] summarizing with session:mlx: up to 32 slices of ~26504 tokens, keeping 6 messages verbatim
```

### Verify

```
python3 tools/digest_smoke.py <mlx-agent> --backend foundation
python3 tools/digest_smoke.py <mlx-agent> --model <dir>
```

Structure only - summary quality is nondeterministic, and asserting on content would be a flaky
lie. What it does assert is every clause of the contract above.
