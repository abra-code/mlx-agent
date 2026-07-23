# Dependency pins

Version ranges live in `project.yml` (`packages:`); the exact resolved versions live in `mlx-agent.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved`, which is committed - that file is what makes a build reproducible, so review it in diffs. Keep this list, `project.yml`, and `Package.resolved` in step.

- `mlx-swift-lm` 3.31.4 (MLXLLM, MLXLMCommon, MLXHuggingFace)
- `mlx-swift` 0.31.6 (MLX; `minorVersion: 0.31.4`, i.e. up-to-next-minor, to match mlx-swift-lm)
- `swift-transformers` 1.3.x (Tokenizers) - mlx-swift-lm 3.x decoupled the tokenizer integration into an OPT-IN dependency the consumer must supply; the `#huggingFaceTokenizerLoader()` macro expands to `Tokenizers.AutoTokenizer`. `swift-huggingface` 0.9.0 comes in transitively (the `HuggingFace`/Hub client).
- `swift-sdk` (MCP) 0.9.0+ - MCP stdio tool clients (one per configured server).
- `swift-jinja` 2.4.2+ - transitive via swift-transformers (which only asks for 2.0.0+), pinned explicitly because agentic MLX turns need the 2.4.2 string-filter coercion fix (undefined/null coerce like Jinja2's `soft_str`; gemma-family tool templates do `value['type'] | upper` on optional MCP params).

The graph resolves cleanly against mlx-swift-lm's `swift-syntax 602..<604` constraint (resolves 603.0.2 with prebuilt macro binaries).
