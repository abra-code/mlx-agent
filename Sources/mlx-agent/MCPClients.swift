// MCPClients.swift - MCP stdio tool layer.
//
// Spawns one child process per configured MCP server, speaks MCP over its stdio
// (initialize + tools/list at start, tools/call per dispatch), and exposes:
//   - `toolSpecs`: the union of every server's tools, as model-facing ToolSpecs.
//   - `route(name)`: exposed-tool-name -> owning server + real tool name + gated flag.
//
// SECURITY: tool arguments are passed to the server as structured MCP JSON
// (`[String: MCP.Value]`), never interpolated into a shell string. The gate that stops
// a mutating tool is the permission round-trip in Agent.dispatch, keyed off `gated`.
//
// ── Config format (`--mcp-config <path>`) ─────────────────────────────────────
// A JSON object with a single "servers" array. mlx-agent OWNS this contract; any
// producer must emit it. See docs/mcp-config.md and Examples/mcp-config.example.json.
//
//     {
//       "servers": [
//         {
//           "name":       "local",              // required: unique label for the server
//           "command":    "/abs/path/to/exe",   // required: executable (abs path or on PATH)
//           "args":       ["--flag", "value"],  // optional: argv passed to command
//           "env":        { "KEY": "VALUE" },   // optional: added to the inherited environment
//           "gatedTools": ["write_file", ...]   // optional: tools requiring permission first
//         }
//       ]
//     }
//
// Each server is spawned and its tools are offered to the model. A tool whose REAL
// name is listed in that server's `gatedTools` triggers a permission round-trip
// before it is dispatched; all other tools dispatch directly. If two servers export
// the same tool name, the first keeps the bare name and later ones are exposed as
// "<server>__<tool>" (routing maps the exposed name back to the real one).

import Foundation
import MCP
import MLXLMCommon

#if canImport(System)
    import System
#else
    import SystemPackage
#endif

#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#endif

/// Handshake (connect + tools/list) must complete within this window or the server is
/// considered dead - otherwise a spawned process that never answers `initialize` would
/// hang session/new forever.
private let mcpHandshakeTimeout: TimeInterval = 30

// MARK: - Config

struct MCPServerConfig: Sendable {
    let name: String
    let command: String
    let args: [String]
    let env: [String: String]
    let gatedTools: Set<String>
}

enum MCPConfigError: LocalizedError {
    case unreadable(String)
    case malformed(String)

    var errorDescription: String? {
        switch self {
        case .unreadable(let p): return "cannot read MCP config at \(p)"
        case .malformed(let m): return "malformed MCP config: \(m)"
        }
    }
}

enum MCPConfigLoader {
    /// Parse the stdio-direct config file into server configs. A server missing
    /// `name` or `command` is skipped with a warning rather than failing the whole load.
    static func load(_ path: String) throws -> [MCPServerConfig] {
        guard let data = FileManager.default.contents(atPath: path) else {
            throw MCPConfigError.unreadable(path)
        }
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let servers = root["servers"] as? [[String: Any]]
        else {
            throw MCPConfigError.malformed("expected top-level { \"servers\": [ ... ] }")
        }
        var out: [MCPServerConfig] = []
        for entry in servers {
            guard let name = entry["name"] as? String,
                let command = entry["command"] as? String, !command.isEmpty
            else {
                FileHandle.standardError.write(
                    Data("[mlx-agent mcp] skipping server without name/command\n".utf8))
                continue
            }
            let args = (entry["args"] as? [String]) ?? []
            let env = (entry["env"] as? [String: String]) ?? [:]
            let gated = Set((entry["gatedTools"] as? [String]) ?? [])
            out.append(
                MCPServerConfig(
                    name: name, command: command, args: args, env: env, gatedTools: gated))
        }
        return out
    }
}

// MARK: - A spawned server + connected client

/// One MCP server: the child process, the connected MCP client, and the tools it
/// advertised. Retains the pipes so their file descriptors stay open for the client's
/// lifetime (a released Pipe FileHandle would close the fd out from under the transport).
final class MCPServer: @unchecked Sendable {
    let config: MCPServerConfig
    let client: Client
    let tools: [MCP.Tool]
    private let process: Process
    private let inPipe: Pipe
    private let outPipe: Pipe

    private init(
        config: MCPServerConfig, client: Client, tools: [MCP.Tool],
        process: Process, inPipe: Pipe, outPipe: Pipe
    ) {
        self.config = config
        self.client = client
        self.tools = tools
        self.process = process
        self.inPipe = inPipe
        self.outPipe = outPipe
    }

    /// Spawn the server command and complete the MCP handshake (connect auto-initializes),
    /// then fetch its tool list. Throws if the process can't launch or the handshake fails.
    static func launch(_ config: MCPServerConfig) async throws -> MCPServer {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: config.command)
        process.arguments = config.args
        var environment = ProcessInfo.processInfo.environment
        for (k, v) in config.env { environment[k] = v }
        process.environment = environment

        // inPipe: we write -> child stdin.  outPipe: child stdout -> we read.
        let inPipe = Pipe()
        let outPipe = Pipe()
        process.standardInput = inPipe
        process.standardOutput = outPipe
        process.standardError = FileHandle.standardError  // child logs interleave into ours

        try process.run()

        // Give the transport its OWN dup'd fds so neither it nor the retained Pipe
        // double-closes the same descriptor on teardown, then close the parent's copies
        // of the child-owned ends so (a) our reader sees EOF when the child dies and
        // (b) we hold no spurious writer on the child's stdin.
        let readFD = dup(outPipe.fileHandleForReading.fileDescriptor)
        let writeFD = dup(inPipe.fileHandleForWriting.fileDescriptor)
        try? inPipe.fileHandleForReading.close()
        try? inPipe.fileHandleForWriting.close()
        try? outPipe.fileHandleForReading.close()
        try? outPipe.fileHandleForWriting.close()
        guard readFD >= 0, writeFD >= 0 else {
            if readFD >= 0 { close(readFD) }
            if writeFD >= 0 { close(writeFD) }
            process.terminate()
            throw MCPConfigError.malformed("could not dup pipe descriptors for \(config.name)")
        }

        // Any failure after the process is running must terminate the child, or it
        // orphans to launchd. Bound the handshake so a silent server cannot hang us.
        do {
            let transport = StdioTransport(
                input: FileDescriptor(rawValue: readFD),
                output: FileDescriptor(rawValue: writeFD),
                logger: nil)
            let client = Client(name: "mlx-agent", version: "0.1")
            try await withTimeout(mcpHandshakeTimeout) {
                _ = try await client.connect(transport: transport)  // connect() calls initialize()
            }
            let tools = try await withTimeout(mcpHandshakeTimeout) {
                try await client.listTools()
            }.tools

            FileHandle.standardError.write(
                Data(
                    "[mlx-agent mcp] \(config.name): \(tools.count) tools (\(tools.map { $0.name }.joined(separator: ", ")))\n"
                        .utf8))

            return MCPServer(
                config: config, client: client, tools: tools,
                process: process, inPipe: inPipe, outPipe: outPipe)
        } catch {
            process.terminate()
            throw error
        }
    }

    func shutdown() async {
        await client.disconnect()
        if process.isRunning { process.terminate() }
    }

    /// Synchronous SIGTERM to the child, for teardown paths that cannot await
    /// (process exit / stdin EOF) - stops the server orphaning to launchd.
    func terminateProcess() {
        if process.isRunning { process.terminate() }
    }
}

// MARK: - Registry

/// Immutable-after-build routing table: the union of all servers' tools as ToolSpecs,
/// plus a lookup from the exposed (possibly namespaced) tool name back to its server,
/// real tool name, and gated flag. Built once at session start.
final class MCPToolRegistry: @unchecked Sendable {
    struct Route {
        let server: MCPServer
        let toolName: String  // the server's real tool name
        let gated: Bool
    }

    let toolSpecs: [ToolSpec]
    private let routes: [String: Route]  // exposed name -> route
    private let servers: [MCPServer]

    private init(toolSpecs: [ToolSpec], routes: [String: Route], servers: [MCPServer]) {
        self.toolSpecs = toolSpecs
        self.routes = routes
        self.servers = servers
    }

    func route(_ exposedName: String) -> Route? { routes[exposedName] }

    func shutdownAll() async {
        for s in servers { await s.shutdown() }
    }

    /// Synchronous teardown for exit paths (SIGTERM every child immediately).
    func terminateAll() {
        for s in servers { s.terminateProcess() }
    }

    /// Launch every configured server and assemble the registry. A server that fails to
    /// launch is logged and skipped (its tools are simply absent) rather than aborting -
    /// one broken server should not disable the whole agent.
    static func build(_ configs: [MCPServerConfig]) async -> MCPToolRegistry {
        var servers: [MCPServer] = []
        for config in configs {
            do {
                servers.append(try await MCPServer.launch(config))
            } catch {
                FileHandle.standardError.write(
                    Data(
                        "[mlx-agent mcp] server \(config.name) failed to launch: \(error.localizedDescription)\n"
                            .utf8))
            }
        }

        var routes: [String: Route] = [:]
        var specs: [ToolSpec] = []
        for server in servers {
            for tool in server.tools {
                // Namespace collision: first server wins the bare name; later servers
                // get "<server>__<tool>" so both remain reachable and distinct.
                let exposed = routes[tool.name] == nil ? tool.name : "\(server.config.name)__\(tool.name)"
                if routes[exposed] != nil { continue }  // extremely unlikely double collision
                let gated = server.config.gatedTools.contains(tool.name)
                routes[exposed] = Route(server: server, toolName: tool.name, gated: gated)
                specs.append(makeToolSpec(tool, exposedName: exposed))
            }
        }
        return MCPToolRegistry(toolSpecs: specs, routes: routes, servers: servers)
    }
}

// MARK: - Conversions (MCP <-> mlx-swift-lm)

/// MCP `Tool` -> model-facing ToolSpec. The tool's JSON-Schema `inputSchema` becomes the
/// function `parameters`. A non-object schema falls back to an empty object schema.
func makeToolSpec(_ tool: MCP.Tool, exposedName: String) -> ToolSpec {
    let parameters: [String: any Sendable]
    if case .object = tool.inputSchema,
        let obj = sendableFromMCPValue(tool.inputSchema) as? [String: any Sendable]
    {
        parameters = obj
    } else {
        parameters = ["type": "object", "properties": [String: any Sendable](), "required": [String]()]
    }
    return [
        "type": "function",
        "function": [
            "name": exposedName,
            "description": tool.description ?? "",
            "parameters": parameters,
        ] as [String: any Sendable],
    ]
}

/// Recursively lower an `MCP.Value` into a Sendable JSON tree usable inside a ToolSpec.
///
/// swift-jinja's `Value(any:)` (via swift-transformers' chat-template rendering) converts
/// arrays through a `[Any?]` cast that a heterogeneous `[any Sendable]` array fails - so
/// arrays are emitted with a CONCRETE homogeneous element type when possible ([String],
/// [Int], ...), matching the hand-built ToolSpecs that render correctly. JSON null is
/// dropped to an empty string rather than NSNull (which swift-jinja also cannot convert).
func sendableFromMCPValue(_ value: MCP.Value) -> any Sendable {
    switch value {
    case .null: return ""
    case .bool(let b): return b
    case .int(let i): return i
    case .double(let d): return d
    case .string(let s): return s
    case .data(_, let d): return d.base64EncodedString()
    case .array(let a):
        let elements = a.map { sendableFromMCPValue($0) }
        if let strings = elements as? [String] { return strings }
        if let ints = elements as? [Int] { return ints }
        if let doubles = elements as? [Double] { return doubles }
        if let bools = elements as? [Bool] { return bools }
        if let objects = elements as? [[String: any Sendable]] { return objects }
        return elements
    case .object(let o):
        return o.mapValues { sendableFromMCPValue($0) } as [String: any Sendable]
    }
}

/// ToolCall arguments -> `[String: MCP.Value]` for `callTool`, via a Codable round-trip
/// (both `JSONValue` and `MCP.Value` are Codable). Never builds a shell string.
func mcpArguments(_ call: ToolCall) -> [String: MCP.Value] {
    guard let data = try? JSONEncoder().encode(call.function.arguments),
        let dict = try? JSONDecoder().decode([String: MCP.Value].self, from: data)
    else { return [:] }
    return dict
}

/// Join a tool result's content blocks into a single string. Text passes through;
/// non-text blocks are noted so their presence is not silently dropped.
func joinToolContent(_ content: [MCP.Tool.Content]) -> String {
    content.compactMap { block -> String? in
        switch block {
        case .text(let text, _, _): return text
        case .image(_, let mimeType, _, _): return "[image \(mimeType)]"
        case .audio(_, let mimeType, _, _): return "[audio \(mimeType)]"
        case .resource: return "[resource]"
        case .resourceLink: return "[resource_link]"
        }
    }.joined(separator: "\n")
}
