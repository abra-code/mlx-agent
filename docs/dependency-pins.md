# Dependency pins

Version ranges live in `project.yml` (`packages:`); the exact resolved versions live in `mlx-agent.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved`, which is committed - that file is what makes a build reproducible, so review it in diffs. Keep this list, `project.yml`, and `Package.resolved` in step.

That reproducibility is the DEFAULT, not a guarantee about every build: the app-embedding script re-resolves the graph before it builds, deliberately, so a shipped agent is never frozen years behind the ranges above. See [Updating](#updating) - including how to turn it off.

- `mlx-swift-lm` 3.32.3 (MLXLLM, MLXLMCommon, MLXHuggingFace)
- `mlx-swift` 0.32.3 (MLX; `minorVersion: 0.32.3`, i.e. up-to-next-minor, to match mlx-swift-lm 3.32.3, which itself asks for `.upToNextMinor(from: "0.32.3")`). These two move TOGETHER: an mlx-swift-lm minor release that needs a newer mlx-swift minor cannot resolve until the `minorVersion` here is raised, so a re-resolve alone silently stays on the old mlx-swift-lm.
- `swift-transformers` 1.3.x (Tokenizers) - mlx-swift-lm 3.x decoupled the tokenizer integration into an OPT-IN dependency the consumer must supply; the `#huggingFaceTokenizerLoader()` macro expands to `Tokenizers.AutoTokenizer`. `swift-huggingface` (1.0.0 at present) comes in transitively (the `HuggingFace`/Hub client).
- `swift-sdk` (MCP) 0.9.0+ - MCP stdio tool clients (one per configured server).
- `swift-jinja` 2.4.2+ - transitive via swift-transformers (which only asks for 2.0.0+), pinned explicitly because agentic MLX turns need the 2.4.2 string-filter coercion fix (undefined/null coerce like Jinja2's `soft_str`; gemma-family tool templates do `value['type'] | upper` on optional MCP params).

The graph resolves cleanly against mlx-swift-lm's `swift-syntax 602..<604` constraint (resolves 603.0.2 with prebuilt macro binaries).

Three transitive packages stay behind their newest release because an upstream manifest caps them, not because of anything here: `swift-syntax` (604.x is out; mlx-swift-lm allows `..<604`), `swift-crypto` (5.x is out; swift-transformers allows `3.0.0..<5.0.0`), and `yyjson` (0.13 is out; swift-transformers pins `exact: "0.12.0"`).

## Updating

Run:

```
tools/resolve_latest_packages.sh
```

It re-resolves the graph to the newest versions the ranges allow, prints the resulting pins, and then compares each DIRECT dependency in `project.yml` with the newest release tag upstream, flagging any that a version range still holds back (`--no-upstream-check` skips that query, e.g. offline). Cadabra's `update-cadabra.sh` and Interpreter's `update_interpreter.sh` both run it before every agent build they perform (so: not under `--skip-build`), so an app never ships against a frozen set the ranges have long outgrown; `--skip-agent-package-update` opts out there (and refuses to run if `Package.resolved` is missing, since the build would then upgrade regardless).

The result is a diff in this repo, not a silent change: **review the `Package.resolved` diff and commit it** (and update the versions listed above when a pin moves).

What the script does, and why each step is there - every one dodges a way the build stays green on old versions without a word:

- **It deletes `Package.resolved` first.** There is no `xcodebuild -updatePackages`, and `-resolvePackageDependencies` on its own honors the existing file. `swift package update` is not an option here - no `Package.swift`. The previous file is copied to a temp dir and restored if resolution fails, and by a signal handler if the run is interrupted (Ctrl-C, or a closed terminal) while it is gone.
- **It deletes `build/SourcePackages/checkouts` and `workspace-state.json` too**, or the "upgrade" can go backwards. With no `Package.resolved`, SwiftPM still prefers whatever is already checked out whenever that satisfies the ranges, even though it fetches every repository first. Measured 2026-10-15: a `build/` last resolved in August re-resolved swift-jinja 2.5.1 -> 2.4.2, swift-huggingface 0.11.0 -> 0.9.0 and eight more packages DOWNWARD, while the log said "Updating from" for each. The fetched clones under `repositories/` stay, so this costs a local re-checkout (the whole run took 42 s on a warm cache), not a re-download. Any pin that still goes down is printed as a warning.
- **It checks upstream tags**, because a range can block a release and no resolver output says so. mlx-swift-lm 3.32 needed mlx-swift 0.32, which `minorVersion: 0.31.4` could not reach, so the resolve "succeeded" on 3.31.4. Raise a range in `project.yml` (then `xcodegen generate`) when it is ours; when another package's manifest caps it (swift-syntax, swift-crypto and yyjson above), it waits for that package.
