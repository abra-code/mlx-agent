// swift-tools-version: 6.1
import PackageDescription

// mlx-agent - native Swift ACP agent on Apple's first-party mlx-swift-lm.
// Phase 0 + engine-gate spike (see ~/Development/MLXApp/docs/10-development-plan.md).
//
// Dependency notes:
// - mlx-swift-lm 3.x decoupled swift-transformers into an opt-in integration: the
//   MLXHuggingFace macros expand to code that references `Tokenizers` (swift-transformers)
//   and, for the hub downloader, `HuggingFace` (swift-huggingface). The CONSUMER must
//   supply those packages. That is why they appear here even though mlx-swift-lm itself
//   does not depend on them.
// - mlx-swift-lm pins swift-syntax to 602..<604; keep transitive deps compatible.
let package = Package(
    name: "mlx-agent",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "mlx-agent", targets: ["mlx-agent"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.4")),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMajor(from: "1.3.0")),
    ],
    targets: [
        .executableTarget(
            name: "mlx-agent",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/mlx-agent"
        )
    ]
)
