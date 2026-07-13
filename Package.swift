// swift-tools-version: 6.1
import PackageDescription

// mlx-agent - native Swift ACP agent on Apple's first-party mlx-swift-lm.
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
        // MCP stdio clients (one per configured server) speak MCP directly - no mcp-proxy.
        .package(url: "https://github.com/modelcontextprotocol/swift-sdk", from: "0.9.0"),
    ],
    targets: [
        // Foundation-only chunker for the map mode. Split into its own target so its logic
        // can be unit-tested with swift-testing WITHOUT linking MLX/Metal (which only builds
        // via xcodebuild). `swift test` builds just this library and its tests.
        .target(
            name: "Chunking",
            path: "Sources/Chunking"
        ),
        .executableTarget(
            name: "mlx-agent",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                // MCP stdio tool clients.
                .product(name: "MCP", package: "swift-sdk"),
                "Chunking",
            ],
            path: "Sources/mlx-agent"
        ),
        .testTarget(
            name: "ChunkingTests",
            dependencies: ["Chunking"],
            path: "Tests/ChunkingTests"
        ),
    ]
)
