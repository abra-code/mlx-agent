# Dependency pins

Version ranges live in `project.yml` (`packages:`); the exact resolved versions live in `mlx-agent.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved`, which is committed - that file is what makes a build reproducible, so review it in diffs. Keep this list, `project.yml`, and `Package.resolved` in step.

That reproducibility is the DEFAULT, not a guarantee about every build: the app-embedding script re-resolves the graph before it builds, deliberately, so a shipped agent is never frozen years behind the ranges above. See [Updating](#updating) - including how to turn it off.

- `mlx-swift-lm` 3.31.4 (MLXLLM, MLXLMCommon, MLXHuggingFace)
- `mlx-swift` 0.31.6 (MLX; `minorVersion: 0.31.4`, i.e. up-to-next-minor, to match mlx-swift-lm)
- `swift-transformers` 1.3.x (Tokenizers) - mlx-swift-lm 3.x decoupled the tokenizer integration into an OPT-IN dependency the consumer must supply; the `#huggingFaceTokenizerLoader()` macro expands to `Tokenizers.AutoTokenizer`. `swift-huggingface` 0.9.0 comes in transitively (the `HuggingFace`/Hub client).
- `swift-sdk` (MCP) 0.9.0+ - MCP stdio tool clients (one per configured server).
- `swift-jinja` 2.4.2+ - transitive via swift-transformers (which only asks for 2.0.0+), pinned explicitly because agentic MLX turns need the 2.4.2 string-filter coercion fix (undefined/null coerce like Jinja2's `soft_str`; gemma-family tool templates do `value['type'] | upper` on optional MCP params).

The graph resolves cleanly against mlx-swift-lm's `swift-syntax 602..<604` constraint (resolves 603.0.2 with prebuilt macro binaries).

## Updating

`AIChatApp/update-cadabra.sh` re-resolves the graph on every agent build it performs (so: not under `--skip-build`): it moves `Package.resolved` aside into a temp dir, runs `xcodebuild -resolvePackageDependencies`, and prints the resulting versions, so the app never ships against a frozen set that the ranges above have long outgrown. (There is no `xcodebuild -updatePackages`, and `-resolvePackageDependencies` on its own honours the existing file; removing it first is what makes the resolution an upgrade. `swift package update` is not an option here - no `Package.swift`.) The previous file is restored if resolution fails, and by a signal handler if the run is interrupted while it is moved aside. `--skip-agent-package-update` opts out entirely (and refuses to run if `Package.resolved` is missing, since the build would then upgrade regardless).

The result is a diff in this repo, not a silent change: **review the `Package.resolved` diff and commit it** (and update the versions listed above when a pin moves). To do the same by hand, delete `mlx-agent.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved` and run:

```
xcodebuild -project mlx-agent.xcodeproj -scheme mlx-agent -derivedDataPath build \
  -skipPackagePluginValidation -skipMacroValidation -resolvePackageDependencies
```
