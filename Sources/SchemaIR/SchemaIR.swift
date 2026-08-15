// SchemaIR.swift - JSON Schema (as an MCP tool advertises it) lowered to a small typed IR.
//
// This exists so that the hard half of the Foundation Models tool bridge is testable without
// Foundation Models. Bridging an MCP tool means turning its `inputSchema` into a
// `DynamicGenerationSchema`, and JSON Schema is far larger than what that type can express -
// so most of the work is deciding what to do with the constructs that do not fit. Those
// decisions are the part worth testing, and they are all here, in a target that links nothing
// but Foundation and builds in seconds. The walker that turns this IR into framework types is
// mechanical by comparison and lives next to the backend.
//
// WHY DEGRADING IS SAFE. The generation schema does not validate anything - the MCP server
// does, authoritatively, and rejects a bad call whatever we told the model. All the schema
// does is CONSTRAIN decoding, so a construct we cannot express costs the model some guidance
// and costs correctness nothing. That is the whole reason this file prefers "degrade with a
// note" to "fail the tool": a tool the model is slightly worse at calling still beats a tool
// it cannot see. Every degradation that changes a node's TYPE is recorded so the log can say
// what was lost; a bound dropped for being unrepresentable is not, on the same reasoning that
// keeps `pattern` out of the list - it narrows guidance without changing the shape the model
// must produce, and one line per `format: uri` would bury the losses that matter.
//
// The one thing NOT done here is regex. JSON Schema `pattern` is ECMA-262 and
// `GenerationGuide.pattern` takes a Swift `Regex`; the dialects differ, and building a Regex
// from a server-supplied string at runtime can throw on input we do not control. So a pattern
// becomes a sentence in the description - advice to the model rather than a constraint.

import Foundation

/// A node in the lowered schema. `indirect` because objects and arrays nest.
public indirect enum SchemaNode: Sendable, Equatable {
    case object([SchemaField])
    case array(items: SchemaNode, minItems: Int?, maxItems: Int?)
    case string
    /// A closed set of string values - JSON Schema `enum` with all-string members.
    case choice([String])
    case integer(minimum: Int?, maximum: Int?)
    case number(minimum: Double?, maximum: Double?)
    case boolean
    /// Something the framework cannot express. Rendered as a plain string whose description
    /// carries `reason`, so the model is told what shape to write rather than left guessing.
    case opaque(reason: String)
}

/// One property of an object, with the two things a `DynamicGenerationSchema.Property` needs
/// beyond the node itself.
public struct SchemaField: Sendable, Equatable {
    public let name: String
    public let description: String?
    public let node: SchemaNode
    public let isOptional: Bool

    public init(name: String, description: String?, node: SchemaNode, isOptional: Bool) {
        self.name = name
        self.description = description
        self.node = node
        self.isOptional = isOptional
    }
}

/// The result of lowering one tool's `inputSchema`.
public struct ParsedSchema: Sendable, Equatable {
    /// Always `.object`. A tool's parameters are an object even when the schema said otherwise;
    /// see `SchemaIR.parse`.
    public let root: SchemaNode
    /// One line per construct that could not be expressed, as `path: reason`. For the log -
    /// a bridged tool that quietly lost half its schema should not look identical to one that
    /// converted cleanly.
    public let degradations: [String]

    public var isDegraded: Bool { !degradations.isEmpty }
}

public enum SchemaIR {

    /// How deep to follow nesting before giving up on a subtree.
    ///
    /// Not a stack-overflow guard so much as a budget one: every level costs tokens out of a
    /// 4096-token window, and a schema nested deeper than this is not going to be usable on
    /// this model anyway.
    public static let maxDepth = 6

    /// Lower a tool's `parameters` / `inputSchema` dictionary.
    ///
    /// Total: never throws and never returns nil. A schema this cannot read at all becomes an
    /// object with no properties, which is a truthful description of what we know about it and
    /// still lets the model call the tool.
    public static func parse(_ schema: [String: Any]) -> ParsedSchema {
        var degradations: [String] = []
        let node = lower(schema, path: "", depth: 0, into: &degradations)
        guard case .object = node else {
            // A tool whose parameters are not an object - MCP allows the field to be anything,
            // and `makeToolSpec` already substitutes an empty object for the non-object case.
            // Reaching here means the schema said `type: object` nowhere; treat it as no
            // parameters rather than inventing one.
            //
            // Only report if `lower` did not already. A root of `{"$ref": ...}`, `{"allOf": []}`,
            // `{"type": ["object","null"]}` or `{}` degrades on the way down AND fails the guard
            // here, and listing it twice makes one loss read as two - which matters, because the
            // count is what the log reports.
            let alreadyReported = degradations.contains { $0.hasPrefix("(root): ") }
            if !alreadyReported {
                degradations.append("(root): not an object schema, treated as no parameters")
            }
            return ParsedSchema(root: .object([]), degradations: degradations)
        }
        return ParsedSchema(root: node, degradations: degradations)
    }

    // MARK: - The walk

    private static func lower(
        _ node: [String: Any], path: String, depth: Int, into degradations: inout [String]
    ) -> SchemaNode {
        if depth > maxDepth {
            let reason = "nested deeper than \(maxDepth) levels"
            degradations.append("\(display(path)): \(reason)")
            return .opaque(reason: reason)
        }

        // Composition keywords first: a node carrying one of these is not described by its
        // `type` even when it also has one, so reading `type` would silently ignore the
        // constraint that mattered.
        for keyword in ["$ref", "allOf", "oneOf", "anyOf", "not"] where node[keyword] != nil {
            let reason = "uses \(keyword), which has no equivalent"
            degradations.append("\(display(path)): \(reason)")
            return .opaque(reason: reason)
        }

        if let choices = node["enum"] {
            return lowerEnum(choices, path: path, into: &degradations)
        }

        let resolved = resolvedType(node)
        if case .union(let listed) = resolved {
            let reason = "type is a union (\(listed))"
            degradations.append("\(display(path)): \(reason)")
            return .opaque(reason: reason)
        }

        switch resolved.name {
        case .some("object"):
            let properties = node["properties"] as? [String: Any] ?? [:]
            let required = Set(node["required"] as? [String] ?? [])
            // Sorted so the schema - and therefore the token cost and every test - is stable.
            // Dictionary order is not.
            let fields = properties.keys.sorted().compactMap { key -> SchemaField? in
                let childPath = path.isEmpty ? key : "\(path).\(key)"
                guard let child = properties[key] as? [String: Any] else {
                    degradations.append("\(display(childPath)): not a schema object, dropped")
                    return nil
                }
                var notes: [String] = []
                let lowered = lower(child, path: childPath, depth: depth + 1, into: &degradations)
                collectNotes(child, into: &notes)
                return SchemaField(
                    name: key,
                    description: describe(child["description"] as? String, notes: notes, node: lowered),
                    node: lowered,
                    isOptional: !required.contains(key))
            }
            return .object(fields)

        case .some("array"):
            guard let items = node["items"] as? [String: Any] else {
                // `items: true`, `items: [..]` (tuple form) or nothing at all. An array of
                // unknown element type cannot be built.
                let reason = "array without a single item schema"
                degradations.append("\(display(path)): \(reason)")
                return .opaque(reason: reason)
            }
            let element = lower(items, path: path + "[]", depth: depth + 1, into: &degradations)
            return .array(
                items: element,
                minItems: integer(node["minItems"]), maxItems: integer(node["maxItems"]))

        case .some("string"):
            return .string

        case .some("integer"):
            return .integer(
                minimum: intBound(node, "minimum", "exclusiveMinimum", adjust: +1),
                maximum: intBound(node, "maximum", "exclusiveMaximum", adjust: -1))

        case .some("number"):
            return .number(
                minimum: doubleBound(node, "minimum"), maximum: doubleBound(node, "maximum"))

        case .some("boolean"):
            return .boolean

        case .some("null"):
            let reason = "type null carries no value"
            degradations.append("\(display(path)): \(reason)")
            return .opaque(reason: reason)

        case .some(let other):
            let reason = "unknown type '\(other)'"
            degradations.append("\(display(path)): \(reason)")
            return .opaque(reason: reason)

        case .none:
            // No `type` at all. Legal JSON Schema meaning "anything", and common in
            // hand-written MCP servers.
            let reason = "no type given, so any JSON value is allowed"
            degradations.append("\(display(path)): \(reason)")
            return .opaque(reason: reason)
        }
    }

    /// `enum` is the strongest signal a node can carry, so it wins over `type`.
    private static func lowerEnum(
        _ raw: Any, path: String, into degradations: inout [String]
    ) -> SchemaNode {
        guard let values = raw as? [Any], !values.isEmpty else {
            let reason = "empty or malformed enum"
            degradations.append("\(display(path)): \(reason)")
            return .opaque(reason: reason)
        }
        let strings = values.compactMap { $0 as? String }
        guard strings.count == values.count else {
            // A mixed enum (numbers, or strings and null) cannot become a choice of strings
            // without changing what is legal. Say what the choices were instead.
            let listed = values.map { "\($0)" }.joined(separator: ", ")
            let reason = "enum is not all strings; one of: \(listed)"
            degradations.append("\(display(path)): \(reason)")
            return .opaque(reason: reason)
        }
        return .choice(strings)
    }

    /// What `type` resolved to, kept separate from the `.opaque` decision so the union case is
    /// reported exactly once - returning a fake type name from here would have it counted again
    /// by the unknown-type branch.
    private enum TypeResolution {
        case named(String)
        case union(String)
        case absent

        var name: String? {
            if case .named(let n) = self { return n }
            return nil
        }
    }

    /// `type` as JSON Schema actually allows it: a string, or an array of them.
    ///
    /// The array form is nearly always `["string", "null"]` - a nullable field, which the IR
    /// already expresses through `isOptional` on the enclosing property. So null is dropped and
    /// a single remaining type is used; anything else is a real union and degrades.
    private static func resolvedType(_ node: [String: Any]) -> TypeResolution {
        if let single = node["type"] as? String { return .named(single) }
        guard let many = node["type"] as? [Any] else { return .absent }
        let names = many.compactMap { $0 as? String }.filter { $0 != "null" }
        if names.count == 1 { return .named(names[0]) }
        return .union(names.isEmpty ? "null" : names.joined(separator: " or "))
    }

    // MARK: - Descriptions

    /// Constraints that survive only as prose. Each is real guidance the model can act on, and
    /// none of them is expressible in a `DynamicGenerationSchema`.
    ///
    /// These are NOT recorded as degradations: nothing is lost that the model cannot still be
    /// told, and a log line per `format: uri` would drown the ones that matter.
    private static func collectNotes(
        _ node: [String: Any], into notes: inout [String], depth: Int = 0
    ) {
        // Its own cap: this walks the raw dictionary rather than the IR, so `lower`'s depth
        // budget does not apply to it, and the schema comes from a server we do not control.
        if depth > maxDepth { return }
        if let pattern = node["pattern"] as? String { notes.append("must match /\(pattern)/") }
        if let format = node["format"] as? String { notes.append("format: \(format)") }
        if let min = integer(node["minLength"]) { notes.append("at least \(min) characters") }
        if let max = integer(node["maxLength"]) { notes.append("at most \(max) characters") }
        if let value = node["default"] { notes.append("defaults to \(value)") }
        // `const` is an enum of one. It could be lowered to a single-member choice, but saying
        // it in the description costs less and reads better to a model that has to reproduce it.
        if let value = node["const"] { notes.append("must be exactly \(value)") }

        // An array's ELEMENT constraints have nowhere else to go: only a property carries a
        // description, and the elements are not properties. So they are folded into the
        // array's own description, prefixed to say what they apply to. This recurses, so an
        // array of arrays nests the prefix rather than losing the note.
        if (node["type"] as? String) == "array", let items = node["items"] as? [String: Any] {
            var itemNotes: [String] = []
            collectNotes(items, into: &itemNotes, depth: depth + 1)
            if let desc = items["description"] as? String, !desc.isEmpty {
                itemNotes.insert(desc, at: 0)
            }
            if !itemNotes.isEmpty {
                notes.append("each element: \(itemNotes.joined(separator: ", "))")
            }
        }
    }

    /// Fold the schema's own description, the prose-only constraints, and any degradation
    /// reason into the single description string a property can carry.
    private static func describe(_ base: String?, notes: [String], node: SchemaNode) -> String? {
        var parts: [String] = []
        if let base, !base.isEmpty { parts.append(base) }
        switch node {
        case .opaque(let reason):
            parts.append("any JSON value - \(reason)")
        case .array(.opaque(let reason), _, _):
            // An array whose ELEMENT degraded. Without this the model is handed an array of
            // unexplained strings while every other opaque node gets told what it is - the
            // reason reached the log and stopped there.
            parts.append("each element is any JSON value - \(reason)")
        default:
            break
        }
        parts.append(contentsOf: notes)
        guard !parts.isEmpty else { return nil }
        return parts.joined(separator: ". ")
    }

    // MARK: - Bounds

    /// An exclusive integer bound is an inclusive one on the adjacent integer, exactly. Doing
    /// the conversion here is safe in a way the float case is not.
    /// The `+1` / `-1` is REPORTING arithmetic, not plain arithmetic, and that is the whole
    /// point. `exclusiveMinimum: 9223372036854775807` is a legal JSON Schema, swift-sdk decodes
    /// it as `Int.max` (it tries `Int` before `Double`), and `Int.max + 1` traps - taking the
    /// process down while the tool set is being built, before the user has typed anything.
    /// Measured: exit 133, no log, no partial turn. A bound that cannot be moved to its
    /// inclusive neighbor is simply dropped, which is what every other unrepresentable bound
    /// here does.
    private static func intBound(
        _ node: [String: Any], _ inclusive: String, _ exclusive: String, adjust: Int
    ) -> Int? {
        if let value = integer(node[inclusive]) { return value }
        if let value = integer(node[exclusive]) {
            let (adjusted, overflowed) = value.addingReportingOverflow(adjust)
            return overflowed ? nil : adjusted
        }
        return nil
    }

    /// An integer bound, however the schema wrote it.
    ///
    /// `as? Int` alone is not enough and the miss is silent: JSON has one number type, so
    /// `"minimum": 0.0` is perfectly ordinary, and `sendableFromMCPValue` hands us a NATIVE
    /// `Double` rather than an `NSNumber` - so there is no bridging to rescue the cast and the
    /// bound is simply dropped. A non-integral value is refused rather than rounded, since
    /// rounding would silently change what is legal.
    ///
    /// `Bool` is deliberately not accepted: draft-04 writes `"exclusiveMinimum": true` as a
    /// MODIFIER of `minimum`, not a value, and reading it as 1 would invent a bound. Swift's
    /// `Bool as? Int` is already nil, so this is a note rather than a guard.
    /// The upper test is STRICTLY less than, and that is the whole point of writing it out.
    /// `Double(Int.max)` is not `Int.max`: 2^63 - 1 is not representable, so the conversion
    /// rounds UP to 2^63. A `<=` therefore admits exactly 2^63, and `Int(_:)` then traps and
    /// takes the process with it. Reachable from a tool schema: swift-sdk's `MCP.Value` decodes
    /// an out-of-range integer literal as a Double, so `{"minimum": 9223372036854775808}` from
    /// any MCP server would abort mlx-agent while building the tool set - before the user has
    /// typed anything. `Int.min` IS exactly representable, so that side stays inclusive.
    private static func integer(_ value: Any?) -> Int? {
        if let int = value as? Int { return int }
        if let double = value as? Double, double.rounded() == double,
            double >= Double(Int.min), double < Double(Int.max)
        {
            return Int(double)
        }
        return nil
    }

    /// Only the inclusive form: there is no nearest representable neighbor to move to for a
    /// float, so an exclusive bound is left off rather than quietly widened or narrowed.
    private static func doubleBound(_ node: [String: Any], _ key: String) -> Double? {
        if let value = node[key] as? Double { return value }
        if let value = node[key] as? Int { return Double(value) }
        return nil
    }

    /// A path for the log. The root has no name of its own.
    private static func display(_ path: String) -> String { path.isEmpty ? "(root)" : path }
}
