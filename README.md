# mlx-agent

Native Swift local-LLM agent built on Apple's first-party
[`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm). Planned to speak ACP
(Agent Client Protocol) over stdio to a UI and MCP over stdio to tools; today it
is the Phase-0 + engine-gate spike for `~/Development/MLXApp` (see that repo's
`docs/10-development-plan.md`). It does NOT yet speak ACP.

## Status

- Phase 0 acceptance: MET. Loads a local mlx-community model and streams.
- Engine gate (Tier-1 tool calling): PASSED 5/5 on Qwen3-4B-4bit (matches omlx).
  Details in `~/Development/MLXApp/docs/09-tier1-results.md`.

## Modes

```
mlx-agent gate [--model <dir>]              # run the 5-case Tier-1 tool-calling gate
mlx-agent chat [--model <dir>] --prompt <text>   # load + stream one completion
```

Default model dir: `/Users/tkukielk/Development/MLXApp/models/Qwen3-4B-4bit`.

## Building (IMPORTANT: xcodebuild, not `swift build`)

MLX's Metal shaders CANNOT be compiled by SwiftPM's command line. `swift build`
links a binary that aborts at runtime with "Failed to load the default metallib".
The build must go through `xcodebuild`, which compiles the shaders into
`mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` next to the binary.

One-time setup on Xcode 26 (the Metal compiler is a separate component):

```
xcodebuild -downloadComponent MetalToolchain     # ~688 MB, once per machine
```

Build:

```
xcodebuild -scheme mlx-agent -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath .build/xcode -configuration Debug \
  -skipPackagePluginValidation -skipMacroValidation build
```

The `-skip...` flags trust the `CudaBuild` package plugin and the swift-syntax
macros (otherwise xcodebuild blocks on a validation prompt). The binary lands at
`.build/xcode/Build/Products/Debug/mlx-agent` with the metallib bundle beside it.

Run:

```
cd .build/xcode/Build/Products/Debug && ./mlx-agent gate
```

## Dependency pins

- `mlx-swift-lm` 3.31.4 (MLXLLM, MLXLMCommon, MLXHuggingFace)
- `mlx-swift` 0.31.6 (MLX; pinned `.upToNextMinor(from: 0.31.4)` to match mlx-swift-lm)
- `swift-transformers` 1.3.x (Tokenizers) - mlx-swift-lm 3.x decoupled the tokenizer
  integration into an OPT-IN dependency the consumer must supply; the
  `#huggingFaceTokenizerLoader()` macro expands to `Tokenizers.AutoTokenizer`.
  `swift-huggingface` 0.9.0 comes in transitively (the `HuggingFace`/Hub client).

The graph resolves cleanly against mlx-swift-lm's `swift-syntax 602..<604`
constraint (resolves 603.0.2 with prebuilt macro binaries).

## Not done yet (the actual agent)

ACP server (JSON-RPC over stdio), MCP stdio clients (swift-sdk), the `--oneshot`
mode the gate should be ported into, prompt-cache persistence, model
registry/switching, the RAM gate. Tracked in MLXApp `docs/10-development-plan.md`.
