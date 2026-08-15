// FoundationTools.swift - MCP tools as Foundation Models tools.
//
// The framework has no "surface the call to me" mode: `Tool.call` is invoked BY the session
// from inside `respond`/`streamResponse`, and its result is folded into the answer before the
// stream continues. So the agent's external tool loop cannot drive this backend, and bridging
// means going the other way - give the framework a Tool whose `call` reaches back into the
// agent's guardrails. See `InternalToolDispatchingBackend` in Agent.swift for that half.
//
// What makes the clean version possible is that `Tool` does NOT require the `@Generable` macro:
// `Arguments` need only be `ConvertibleFromGeneratedContent`, and `GeneratedContent` conforms to
// that trivially. So one dynamic conformance covers every MCP tool, with the schema built at
// runtime instead of compiled in.
//
// Splitting the schema work in two is deliberate: the decisions (what to do with a JSON Schema
// construct the framework cannot express) live in the SchemaIR target, where they are testable
// without an on-device model; what remains here is a mechanical transcription of the IR onto
// framework types.
//
// MEASURED COST, because it is the thing that decides whether this is usable at all: three real
// tools (mcp_server_time's two plus mcp_server_fetch's one) cost 615 tokens of the 4096-token
// window - 15%. Fourteen tools cost 1832, or 45%. This backend works with a handful of small
// tools and does not work with a large registry, and no amount of care here changes that; the
// budget is the constraint. `tokenCost` exists so the log can say so with a number rather than a
// warning, and FoundationBackend reserves it before deciding whether to condense.
//
// TOOL CALLS ARRIVE CONCURRENTLY. Measured: a prompt naming three cities entered three `call`
// bodies in the same microsecond, and the pass took as long as the slowest rather than their sum.
// Nothing here is serialized by the framework, so anything the runner reaches has to be safe for
// simultaneous calls - see the dedup coalescing in Agent.dispatch.

import Foundation
import MLXLMCommon
import SchemaIR

#if canImport(FoundationModels)
    import FoundationModels
#endif

// MARK: - ToolSpec -> the pieces a bridged tool needs

/// The model-facing description of one tool, pulled out of the OpenAI-shaped `ToolSpec` the
/// registry already builds. Plain data, so extracting it is testable and does not need the
/// framework.
struct BridgedToolSpec {
    let name: String
    let description: String
    let schema: ParsedSchema

    /// nil when the spec is not shaped like a function tool. Every spec `MCPToolRegistry`
    /// produces is, so this is defensive rather than expected.
    init?(_ spec: ToolSpec) {
        guard let function = spec["function"] as? [String: any Sendable],
            let name = function["name"] as? String, !name.isEmpty
        else { return nil }
        self.name = name
        self.description = function["description"] as? String ?? ""
        self.schema = SchemaIR.parse(function["parameters"] as? [String: Any] ?? [:])
    }
}

#if canImport(FoundationModels)

    // MARK: - The bridged tool

    /// One MCP tool, callable by the on-device model.
    ///
    /// `Output` is `String` because that is what an MCP server returns once its content blocks
    /// are joined, and `String` is `PromptRepresentable` - so the result goes back into the
    /// transcript with no further conversion.
    @available(macOS 26.0, *)
    struct MCPBridgedTool: FoundationModels.Tool {
        typealias Arguments = GeneratedContent
        typealias Output = String

        let name: String
        let description: String
        let parameters: GenerationSchema
        /// Reaches into `Agent.runToolForBackend`, which is where the permission gate, the
        /// timeout, the byte budget and the dedup live. Installed by Agent; never nil in
        /// practice, but the backend builds tools before it is handed one, so it is optional
        /// state on the backend rather than here.
        let runner: @Sendable (ToolCall) async -> ToolRunOutcome

        func call(arguments: GeneratedContent) async throws -> String {
            let call = ToolCall(
                function: .init(name: name, arguments: decodeArguments(arguments)),
                // The framework does not expose its own id for the call, and the agent needs a
                // stable one to correlate the tool_call and tool_call_update it sends the
                // client. UUID rather than a counter: two passes must not reuse an id.
                id: "fm-\(UUID().uuidString.prefix(8))")
            switch await runner(call) {
            case .result(let text): return text
            case .budgetExhausted(let text): return text
            case .cancelled:
                // The user answered the permission prompt with Cancel. Throwing is the only way
                // to stop a pass from inside a tool: the framework wraps this in
                // `ToolCallError`, which FoundationBackend unwraps back to a canceled turn.
                throw CancellationError()
            }
        }

        /// `GeneratedContent` -> the loop's argument currency, through JSON.
        ///
        /// A round-trip rather than a walk of `GeneratedContent.Kind`: the JSON is what the MCP
        /// server will ultimately be sent anyway, and going through it means the bridged path
        /// and the ordinary path hand `dispatch` byte-identical arguments.
        private func decodeArguments(_ content: GeneratedContent) -> [String: JSONValue] {
            let json = content.jsonString
            guard let data = json.data(using: .utf8),
                let decoded = try? JSONDecoder().decode([String: JSONValue].self, from: data)
            else {
                // Constrained decoding makes this very hard to reach - the schema says object -
                // but an empty argument set is a better answer than a crash, and the server
                // will reject it with a message the model can read.
                return [:]
            }
            return decoded
        }
    }

    // MARK: - IR -> DynamicGenerationSchema

    @available(macOS 26.0, *)
    enum FoundationSchemaBuilder {

        /// Build the framework schema for one tool.
        ///
        /// Total, like the IR parse below it: a construct that cannot be built degrades to a
        /// described string rather than failing, so one awkward tool never costs the session
        /// every other tool. Only a throw from `GenerationSchema` itself is propagated, and the
        /// caller drops that one tool.
        static func schema(for spec: BridgedToolSpec) throws -> GenerationSchema {
            // A FRESH set per tool, which is correct and was worth checking rather than assuming:
            // schema namespaces do not leak between tools. Measured with a tool `t` nesting
            // `a.b` (whose subschema is named t_a_b) beside a second tool named literally
            // `t_a_b` - both were called with their own correct arguments. Making `used` shared
            // across the tool set would only produce uglier names for no gain.
            var used: Set<String> = []
            let root = build(spec.schema.root, name: spec.name, used: &used)
            return try GenerationSchema(root: root, dependencies: [])
        }

        /// Names have to be unique within a schema, so every nested node is named for its path.
        /// The names are model-visible, which is the other reason to build them from the path -
        /// `fetch_url` reads as what it is.
        ///
        /// UNIQUENESS IS ENFORCED HERE, NOT BY THE FRAMEWORK. A path-derived name can collide -
        /// a property `a` holding an object with property `b` produces `t_a_b`, and so does a
        /// sibling property literally named `a_b`, which underscore-separated MCP property names
        /// make ordinary. Measured: `GenerationSchema(root:dependencies:)` does NOT reject that.
        /// It emits ONE `$defs` entry for the name and points both properties at it, so the
        /// second property is silently constrained to the first one's shape - the model is
        /// guided to emit the wrong thing, the server rejects the call, and nothing is logged.
        /// So collisions are broken here by suffixing, which is ugly and correct.
        private static func build(
            _ ir: SchemaNode, name: String, used: inout Set<String>
        ) -> DynamicGenerationSchema {
            switch ir {
            case .object(let fields):
                let unique = claim(name, &used)
                // `used` is threaded through the whole build rather than per level, because the
                // collision is between a nested name and a name anywhere else in the same schema.
                var properties: [DynamicGenerationSchema.Property] = []
                for field in fields {
                    properties.append(
                        DynamicGenerationSchema.Property(
                            name: field.name, description: field.description,
                            schema: build(field.node, name: "\(unique)_\(field.name)", used: &used),
                            isOptional: field.isOptional))
                }
                return DynamicGenerationSchema(name: unique, properties: properties)

            case .array(let items, let minItems, let maxItems):
                return DynamicGenerationSchema(
                    arrayOf: build(items, name: "\(name)_item", used: &used),
                    minimumElements: minItems, maximumElements: maxItems)

            case .choice(let values):
                return DynamicGenerationSchema(name: claim(name, &used), anyOf: values)

            case .string:
                return DynamicGenerationSchema(type: String.self)

            case .integer(let minimum, let maximum):
                var guides: [GenerationGuide<Int>] = []
                if let minimum { guides.append(.minimum(minimum)) }
                if let maximum { guides.append(.maximum(maximum)) }
                return DynamicGenerationSchema(type: Int.self, guides: guides)

            case .number(let minimum, let maximum):
                var guides: [GenerationGuide<Double>] = []
                if let minimum { guides.append(.minimum(minimum)) }
                if let maximum { guides.append(.maximum(maximum)) }
                return DynamicGenerationSchema(type: Double.self, guides: guides)

            case .boolean:
                return DynamicGenerationSchema(type: Bool.self)

            case .opaque:
                // A string, with the reason already folded into the enclosing property's
                // description by the IR. The model writes JSON into it and the server parses
                // it, which is what an unconstrained argument amounts to anyway.
                return DynamicGenerationSchema(type: String.self)
            }
        }

        /// Reserve a schema name, suffixing until it is free.
        ///
        /// Only NAMED nodes go through this. The unnamed ones - `type:`-built scalars and
        /// arrays - carry no name to collide with, which is why they are not claimed.
        private static func claim(_ name: String, _ used: inout Set<String>) -> String {
            var candidate = name
            var suffix = 2
            while used.contains(candidate) {
                candidate = "\(name)_\(suffix)"
                suffix += 1
            }
            used.insert(candidate)
            return candidate
        }
    }

    // MARK: - Building the set

    @available(macOS 26.0, *)
    enum FoundationToolBridge {

        /// Convert every spec the registry offers, dropping only the ones the framework refuses.
        ///
        /// - Parameter log: gets one line per dropped tool and one summary of degradations. A
        ///   tool that lost half its schema still works, so the trace is the only way to tell.
        static func tools(
            from specs: [ToolSpec],
            runner: @escaping @Sendable (ToolCall) async -> ToolRunOutcome,
            log: (String) -> Void
        ) -> [MCPBridgedTool] {
            var built: [MCPBridgedTool] = []
            var degraded: [String] = []

            for spec in specs {
                guard let parsed = BridgedToolSpec(spec) else {
                    log("foundation tools: skipping a tool spec that is not a function")
                    continue
                }
                do {
                    built.append(
                        MCPBridgedTool(
                            name: parsed.name, description: parsed.description,
                            parameters: try FoundationSchemaBuilder.schema(for: parsed),
                            runner: runner))
                } catch {
                    // The one unrecoverable case: the framework rejected the whole schema. Drop
                    // the tool rather than the session - the model simply will not see it.
                    log(
                        "foundation tools: \(parsed.name) dropped, its schema could not be built "
                            + "(\(error.localizedDescription))")
                    continue
                }
                if parsed.schema.isDegraded {
                    degraded.append("\(parsed.name) [\(parsed.schema.degradations.joined(separator: "; "))]")
                }
            }

            if !degraded.isEmpty {
                log(
                    "foundation tools: \(degraded.count) of \(built.count) tool(s) have simplified "
                        + "schemas - \(degraded.joined(separator: ", "))")
            }
            return built
        }

        /// What this tool set costs out of the 4096-token window, before a word is said.
        ///
        /// nil below macOS 26.4, where the framework offers no way to ask. Worth logging rather
        /// than assuming: the window is the binding constraint on this backend, and a registry
        /// that eats half of it before the conversation starts should say so.
        static func tokenCost(of tools: [MCPBridgedTool]) async -> Int? {
            guard #available(macOS 26.4, *) else { return nil }
            return try? await SystemLanguageModel.default.tokenCount(
                for: tools as [any FoundationModels.Tool])
        }

        /// A synchronous stand-in for `tokenCost`, from the schema text.
        ///
        /// The real count is async and below 26.4 there is none at all, so without this the
        /// window reserve is zero for as long as the measurement takes - and permanently on an
        /// older system.
        ///
        /// The half added on top, and the per-tool floor, are not padding for their own sake.
        /// Plain bytes/4 - the approximation the rest of this backend uses - reads LOW here,
        /// because the framework spends tokens per tool that the schema text does not show:
        /// measured, 554 against a true 615 for 3 tools and 1086 against 1235 for 11. Low is the
        /// wrong direction for a reserve; it is what lets a pass overflow. With both corrections
        /// the same sets read 831 and 1650, and a 14-tool set reads 2100 against 1832 - above the
        /// real figure in each, so the worst case is condensing a little earlier than needed.
        static func estimatedTokenCost(of specs: [ToolSpec], built tools: [MCPBridgedTool]) -> Int {
            guard !tools.isEmpty else { return 0 }
            let offered = Set(tools.map(\.name))
            let bytes = specs.reduce(0) { total, spec in
                // Only what the model will actually see. A tool whose schema the framework
                // refused was dropped, and reserving window for it reserves for nothing.
                guard let name = BridgedToolSpec(spec)?.name, offered.contains(name),
                    let data = try? JSONSerialization.data(
                        withJSONObject: spec, options: [.sortedKeys])
                else { return total }
                return total + data.count
            }
            // The per-tool floor is not belt-and-braces: the BYTE half alone reads low, and
            // increasingly so with the tool count, because the framework spends tokens per tool
            // that the schema text does not show. Measured, bytes-only gives 554 against a true
            // 615 for 3 tools and 1086 against 1235 for 11. With the half added on top and this
            // floor, the same sets read 831 and 1650, and a 14-tool set reads 2100 against 1832 -
            // above the real figure in each, which is the only safe direction for a reserve. It
            // matters most on macOS 26.0-26.3, where there is no exact count to replace it and
            // the estimate is what the reserve uses for the whole session.
            return max(bytes / 4 * 3 / 2, 150 * tools.count)
        }
    }

#endif
