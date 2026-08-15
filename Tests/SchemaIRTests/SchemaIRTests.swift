// SchemaIRTests.swift - the degradation rules, which are the whole reason this target exists.
//
// The IR itself is a mechanical transcription; what needed pinning is every decision about a
// JSON Schema construct that `DynamicGenerationSchema` cannot express. Each of those has a
// test here, because the failure mode is silent: a schema that quietly loses a constraint
// still produces a tool the model can call, just one it calls slightly worse - and nothing
// downstream can tell the difference.
//
// Two schemas from real MCP servers (mcp_server_time, mcp_server_fetch) are transcribed
// verbatim at the bottom. They are the reason the exclusive-bound and format cases exist at
// all: both showed up on the first two servers tried.

import Foundation
import Testing

@testable import SchemaIR

// MARK: - Helpers

private func fields(_ parsed: ParsedSchema) -> [SchemaField] {
    guard case .object(let f) = parsed.root else { return [] }
    return f
}

private func field(_ parsed: ParsedSchema, _ name: String) -> SchemaField? {
    fields(parsed).first { $0.name == name }
}

private func object(_ properties: [String: Any], required: [String] = []) -> [String: Any] {
    ["type": "object", "properties": properties, "required": required]
}

// MARK: - The shape that works

@Test func plainObjectLowersToItsProperties() {
    let parsed = SchemaIR.parse(
        object(
            [
                "path": ["type": "string", "description": "Where to read"],
                "limit": ["type": "integer"],
            ], required: ["path"]))

    #expect(!parsed.isDegraded)
    #expect(fields(parsed).map(\.name) == ["limit", "path"])  // sorted, not dictionary order
    #expect(field(parsed, "path")?.isOptional == false)
    #expect(field(parsed, "path")?.description == "Where to read")
    #expect(field(parsed, "limit")?.isOptional == true)
    #expect(field(parsed, "limit")?.node == .integer(minimum: nil, maximum: nil))
}

@Test func propertyOrderIsStableAcrossParses() {
    // The token cost of a tool is paid per pass, so an unstable schema would make the window
    // budget unreproducible. Dictionary iteration order is not stable between runs.
    let schema = object([
        "zebra": ["type": "string"], "alpha": ["type": "string"], "middle": ["type": "string"],
    ])
    #expect(fields(SchemaIR.parse(schema)).map(\.name) == ["alpha", "middle", "zebra"])
}

@Test func noPropertiesIsAnEmptyObjectNotADegradation() {
    // Plenty of MCP tools take no arguments at all. That is a complete schema, not a loss.
    let parsed = SchemaIR.parse(["type": "object", "properties": [String: Any]()])
    #expect(parsed.root == .object([]))
    #expect(!parsed.isDegraded)
}

@Test func nestedObjectsAndArraysSurvive() {
    let parsed = SchemaIR.parse(
        object([
            "items": [
                "type": "array",
                "items": object(["id": ["type": "string"]], required: ["id"]),
                "minItems": 1, "maxItems": 10,
            ]
        ]))

    #expect(!parsed.isDegraded)
    guard case .array(let element, let min, let max) = field(parsed, "items")?.node else {
        Issue.record("expected an array node")
        return
    }
    #expect(min == 1)
    #expect(max == 10)
    #expect(element == .object([SchemaField(name: "id", description: nil, node: .string, isOptional: false)]))
}

// MARK: - Enums

@Test func allStringEnumBecomesAChoice() {
    let parsed = SchemaIR.parse(object(["unit": ["enum": ["celsius", "fahrenheit"]]]))
    #expect(field(parsed, "unit")?.node == .choice(["celsius", "fahrenheit"]))
    #expect(!parsed.isDegraded)
}

@Test func enumWinsOverType() {
    // A node with both is constrained by the enum; reading `type` would widen it back out.
    let parsed = SchemaIR.parse(object(["unit": ["type": "string", "enum": ["c", "f"]]]))
    #expect(field(parsed, "unit")?.node == .choice(["c", "f"]))
}

@Test func mixedEnumDegradesButKeepsTheChoicesAsProse() {
    let parsed = SchemaIR.parse(object(["level": ["enum": [1, 2, "high"]]]))
    guard case .opaque(let reason) = field(parsed, "level")?.node else {
        Issue.record("expected opaque")
        return
    }
    #expect(reason.contains("1, 2, high"))  // the model is still told what is legal
    #expect(parsed.degradations.count == 1)
    #expect(parsed.degradations[0].hasPrefix("level: "))
}

@Test func emptyEnumDegrades() {
    let parsed = SchemaIR.parse(object(["x": ["enum": [Any]()]]))
    #expect(field(parsed, "x")?.node == .opaque(reason: "empty or malformed enum"))
}

// MARK: - Composition keywords

@Test func compositionKeywordsDegradeAndAreNamed() {
    for keyword in ["$ref", "allOf", "oneOf", "anyOf", "not"] {
        let parsed = SchemaIR.parse(object(["x": [keyword: "whatever"]]))
        // `continue`, not `return`: returning on the first failure would silently skip the
        // remaining keywords and report one problem where there might be five.
        guard case .opaque(let reason) = field(parsed, "x")?.node else {
            Issue.record("expected \(keyword) to degrade")
            continue
        }
        #expect(reason.contains(keyword))
        #expect(parsed.degradations.count == 1)
    }
}

@Test func compositionWinsOverAConcurrentType() {
    // `{"type": "string", "allOf": [...]}` is constrained by the allOf. Honoring the type
    // would silently drop it and claim a clean conversion.
    let parsed = SchemaIR.parse(object(["x": ["type": "string", "allOf": [["minLength": 2]]]]))
    #expect(field(parsed, "x")?.node != .string)
    #expect(parsed.isDegraded)
}

// MARK: - Types

@Test func nullableTypeArrayUsesTheNonNullMember() {
    // ["string", "null"] is a nullable field, which `isOptional` already expresses.
    let parsed = SchemaIR.parse(object(["x": ["type": ["string", "null"]]]))
    #expect(field(parsed, "x")?.node == .string)
    #expect(!parsed.isDegraded)
}

@Test func genuineUnionDegradesExactlyOnce() {
    // Reported once, not twice: the union check and the unknown-type check must not both fire.
    let parsed = SchemaIR.parse(object(["x": ["type": ["string", "integer"]]]))
    #expect(parsed.degradations.count == 1)
    #expect(parsed.degradations[0].contains("string or integer"))
}

@Test func missingTypeDegrades() {
    let parsed = SchemaIR.parse(object(["x": ["description": "anything goes"]]))
    guard case .opaque = field(parsed, "x")?.node else {
        Issue.record("expected opaque")
        return
    }
    // The original description is kept and the reason appended, not replaced.
    #expect(field(parsed, "x")?.description?.hasPrefix("anything goes. any JSON value") == true)
}

@Test func unknownTypeDegrades() {
    let parsed = SchemaIR.parse(object(["x": ["type": "date-time"]]))
    #expect(field(parsed, "x")?.node == .opaque(reason: "unknown type 'date-time'"))
}

@Test func arrayWithoutItemsDegrades() {
    let parsed = SchemaIR.parse(object(["x": ["type": "array"]]))
    guard case .opaque(let reason) = field(parsed, "x")?.node else {
        Issue.record("expected opaque")
        return
    }
    #expect(reason.contains("without a single item schema"))
}

@Test func tupleFormItemsDegradesRatherThanGuessing() {
    // `items: [schemaA, schemaB]` means positional types, which an array-of-one-schema cannot say.
    let parsed = SchemaIR.parse(
        object(["x": ["type": "array", "items": [["type": "string"], ["type": "integer"]]]]))
    #expect(parsed.isDegraded)
}

@Test func nonObjectRootBecomesNoParameters() {
    let parsed = SchemaIR.parse(["type": "string"])
    #expect(parsed.root == .object([]))
    #expect(parsed.degradations.contains { $0.contains("not an object schema") })
}

@Test func aPropertyThatIsNotASchemaIsDroppedAndReported() {
    let parsed = SchemaIR.parse(["type": "object", "properties": ["x": "not a schema"]])
    #expect(fields(parsed).isEmpty)
    #expect(parsed.degradations.contains { $0.contains("not a schema object") })
}

// MARK: - Bounds

@Test func exclusiveIntegerBoundsBecomeInclusiveNeighbors() {
    // Exact for integers: > 0 is >= 1. This is the one bound conversion that loses nothing.
    let parsed = SchemaIR.parse(
        object(["n": ["type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 1_000_000]]))
    #expect(field(parsed, "n")?.node == .integer(minimum: 1, maximum: 999_999))
}

@Test func inclusiveIntegerBoundsWinOverExclusiveOnes() {
    let parsed = SchemaIR.parse(
        object(["n": ["type": "integer", "minimum": 5, "exclusiveMinimum": 100]]))
    #expect(field(parsed, "n")?.node == .integer(minimum: 5, maximum: nil))
}

@Test func exclusiveFloatBoundsAreDroppedRatherThanApproximated() {
    // There is no nearest representable neighbor to move to, so widening or narrowing would
    // both be wrong. The server validates the real bound anyway.
    let parsed = SchemaIR.parse(
        object(["n": ["type": "number", "exclusiveMinimum": 0.5, "maximum": 9.5]]))
    #expect(field(parsed, "n")?.node == .number(minimum: nil, maximum: 9.5))
}

@Test func integerWrittenBoundsOnANumberAreAccepted() {
    // JSON has one number type; a schema author writing `minimum: 0` on a `number` is normal.
    let parsed = SchemaIR.parse(object(["n": ["type": "number", "minimum": 0]]))
    #expect(field(parsed, "n")?.node == .number(minimum: 0, maximum: nil))
}

// MARK: - Prose-only constraints

@Test func patternAndFormatBecomeDescriptionNotesNotDegradations() {
    // Deliberate: these are advice the model can act on, and logging one line per `format: uri`
    // would bury the degradations that actually cost something.
    let parsed = SchemaIR.parse(
        object(["url": ["type": "string", "format": "uri", "pattern": "^https?://"]]))
    #expect(field(parsed, "url")?.node == .string)
    #expect(!parsed.isDegraded)
    let desc = field(parsed, "url")?.description ?? ""
    #expect(desc.contains("must match /^https?://"))  // the pattern verbatim, for the model to read
    #expect(desc.contains("format: uri"))
}

@Test func lengthBoundsAndDefaultsBecomeNotes() {
    let parsed = SchemaIR.parse(
        object(["s": ["type": "string", "minLength": 1, "maxLength": 40, "default": "hi"]]))
    let desc = field(parsed, "s")?.description ?? ""
    #expect(desc.contains("at least 1 characters"))
    #expect(desc.contains("at most 40 characters"))
    #expect(desc.contains("defaults to hi"))
}

@Test func arrayElementNotesAreFoldedIntoTheArrayDescription() {
    // An element is not a property, so it has nowhere of its own to carry a description.
    let parsed = SchemaIR.parse(
        object([
            "paths": [
                "type": "array", "description": "Files to read",
                "items": ["type": "string", "format": "path"],
            ]
        ]))
    let desc = field(parsed, "paths")?.description ?? ""
    #expect(desc.hasPrefix("Files to read"))
    #expect(desc.contains("each element: format: path"))
}

// MARK: - Depth

@Test func schemasDeeperThanTheCapDegradeAtTheBoundary() {
    // Pins WHERE the cut falls, not just that something was cut. Asserting only "some
    // degradation mentions nesting" would pass with the cap set to any smaller number.
    var node: [String: Any] = ["type": "string"]
    for _ in 0...(SchemaIR.maxDepth + 1) {
        node = object(["next": node], required: ["next"])
    }
    let parsed = SchemaIR.parse(node)
    let cut = parsed.degradations.filter { $0.contains("nested deeper than") }
    #expect(cut.count == 1)
    // The root is depth 0, so the node refused is the one at maxDepth + 1 - reached through
    // exactly that many `next` hops.
    let path = Array(repeating: "next", count: SchemaIR.maxDepth + 1).joined(separator: ".")
    #expect(cut.first == "\(path): nested deeper than \(SchemaIR.maxDepth) levels")
}

// MARK: - Losses that used to be silent

@Test func aRootThatDegradesIsReportedOnce() {
    // Both `lower` and the non-object guard in `parse` see this schema. Counting it twice makes
    // one loss read as two, and the count is what the log reports.
    for root in [["$ref": "#/definitions/x"], ["allOf": [["type": "object"]]], [String: Any]()] {
        let parsed = SchemaIR.parse(root)
        #expect(parsed.root == .object([]))
        #expect(parsed.degradations.count == 1)
    }
    // The genuinely-not-an-object root still gets the guard's own message.
    let scalar = SchemaIR.parse(["type": "string"])
    #expect(scalar.degradations == ["(root): not an object schema, treated as no parameters"])
}

@Test func boundsWrittenAsFloatsAreNotDropped() {
    // JSON has one number type, so `minimum: 0.0` on an integer is ordinary. `as? Int` alone
    // returns nil for a native Double - no NSNumber bridging to rescue it - and the bound
    // vanished with nothing logged.
    let parsed = SchemaIR.parse(
        object([
            "n": ["type": "integer", "minimum": 0.0, "maximum": 10.0],
            "xs": ["type": "array", "items": ["type": "string"], "minItems": 1.0],
        ]))
    #expect(field(parsed, "n")?.node == .integer(minimum: 0, maximum: 10))
    guard case .array(_, let minItems, _) = field(parsed, "xs")?.node else {
        Issue.record("expected an array")
        return
    }
    #expect(minItems == 1)
}

@Test func aNonIntegralBoundIsRefusedRatherThanRounded() {
    // Rounding would change what is legal. Dropping it leaves the server as the authority.
    let parsed = SchemaIR.parse(object(["n": ["type": "integer", "minimum": 2.5]]))
    #expect(field(parsed, "n")?.node == .integer(minimum: nil, maximum: nil))
}

@Test func draft4ExclusiveMinimumAsABooleanInventsNoBound() {
    // Draft-04 writes `exclusiveMinimum: true` as a MODIFIER of `minimum`, not a value.
    // Reading it as 1 would fabricate a constraint the schema never stated.
    let parsed = SchemaIR.parse(object(["n": ["type": "integer", "exclusiveMinimum": true]]))
    #expect(field(parsed, "n")?.node == .integer(minimum: nil, maximum: nil))
}

@Test func anOutOfRangeBoundIsRefusedRatherThanTrapping() {
    // The failure this pins is a CRASH, not a wrong answer. swift-sdk decodes an integer literal
    // too large for Int as a Double, so these values arrive from any MCP server that ships them.
    // 2^63 is the dangerous one: Double(Int.max) rounds UP to it, so an inclusive upper test
    // admits it and Int(_:) traps.
    for value in [9_223_372_036_854_775_808.0, -9_300_000_000_000_000_000.0, 1e300, -1e300] {
        let parsed = SchemaIR.parse(object(["n": ["type": "integer", "minimum": value]]))
        #expect(field(parsed, "n")?.node == .integer(minimum: nil, maximum: nil))
    }
    // The largest value that IS representable still converts.
    let ok = SchemaIR.parse(object(["n": ["type": "integer", "maximum": 9_007_199_254_740_992.0]]))
    #expect(field(ok, "n")?.node == .integer(minimum: nil, maximum: 9_007_199_254_740_992))
}

@Test func anExclusiveBoundAtTheIntegerLimitIsDroppedRatherThanTrapping() {
    // The EXCLUSIVE keys are the only ones that do arithmetic - `> n` becomes `>= n + 1` - and
    // these two values are exactly where that overflows. They arrive as native Int, not Double
    // (swift-sdk tries Int first), so the float guard above does not cover them. Measured before
    // the fix: a real MCP server advertising `exclusiveMinimum: 9223372036854775807` killed the
    // process with exit 133 while building the tool set.
    let atMax = SchemaIR.parse(
        object(["n": ["type": "integer", "exclusiveMinimum": Int.max]]))
    #expect(field(atMax, "n")?.node == .integer(minimum: nil, maximum: nil))

    let atMin = SchemaIR.parse(
        object(["n": ["type": "integer", "exclusiveMaximum": Int.min]]))
    #expect(field(atMin, "n")?.node == .integer(minimum: nil, maximum: nil))

    // One short of the limit still converts, so the guard is not simply refusing everything.
    let ok = SchemaIR.parse(object(["n": ["type": "integer", "exclusiveMinimum": Int.max - 1]]))
    #expect(field(ok, "n")?.node == .integer(minimum: Int.max, maximum: nil))
}

@Test func constBecomesANote() {
    let parsed = SchemaIR.parse(object(["mode": ["type": "string", "const": "fixed"]]))
    #expect(field(parsed, "mode")?.description?.contains("must be exactly fixed") == true)
}

@Test func aDegradedArrayElementTellsTheModelWhy() {
    // The reason reached the log but not the model: `describe` matched `.opaque` and an array
    // whose ITEM is opaque is `.array(items: .opaque)`, so the property carried no description
    // and the model saw an array of unexplained strings.
    let parsed = SchemaIR.parse(
        object(["xs": ["type": "array", "items": ["oneOf": [["type": "string"]]]]]))
    let desc = field(parsed, "xs")?.description ?? ""
    #expect(desc.contains("each element is any JSON value"))
    #expect(desc.contains("oneOf"))
}

@Test func aDroppedPropertyIsNamedWithItsFullPath() {
    // The label used to be built without a separator, so a bad property under `outer` logged
    // as `outerbad` - a path that does not exist.
    let parsed = SchemaIR.parse(
        object(["outer": object(["bad": "not a schema"])]))
    #expect(parsed.degradations == ["outer.bad: not a schema object, dropped"])
}

// MARK: - Real MCP schemas, transcribed verbatim

@Test func mcpServerTimeConvertsCleanly() {
    let parsed = SchemaIR.parse(
        object(
            [
                "source_timezone": [
                    "type": "string", "description": "Source IANA timezone name",
                ],
                "target_timezone": [
                    "type": "string", "description": "Target IANA timezone name",
                ],
                "time": ["type": "string", "description": "Time to convert in 24-hour format"],
            ], required: ["source_timezone", "time", "target_timezone"]))

    #expect(!parsed.isDegraded)
    #expect(fields(parsed).count == 3)
    #expect(fields(parsed).allSatisfy { !$0.isOptional })
}

@Test func mcpServerFetchConvertsCleanly() {
    // The schema that motivated the exclusive-bound rule. Note `title` on every property,
    // which is ignored - it duplicates the name and would only cost tokens.
    let parsed = SchemaIR.parse(
        object(
            [
                "url": [
                    "type": "string", "format": "uri", "minLength": 1, "title": "Url",
                    "description": "URL to fetch",
                ],
                "max_length": [
                    "type": "integer", "default": 5000, "exclusiveMaximum": 1_000_000,
                    "exclusiveMinimum": 0, "title": "Max Length",
                    "description": "Maximum number of characters to return.",
                ],
                "start_index": [
                    "type": "integer", "default": 0, "minimum": 0, "title": "Start Index",
                    "description": "On return output starting at this character index",
                ],
                "raw": [
                    "type": "boolean", "default": false, "title": "Raw",
                    "description": "Get the actual HTML content",
                ],
            ], required: ["url"]))

    #expect(!parsed.isDegraded)
    #expect(field(parsed, "url")?.isOptional == false)
    #expect(field(parsed, "raw")?.isOptional == true)
    #expect(field(parsed, "max_length")?.node == .integer(minimum: 1, maximum: 999_999))
    #expect(field(parsed, "start_index")?.node == .integer(minimum: 0, maximum: nil))
    #expect(field(parsed, "raw")?.node == .boolean)
    #expect(field(parsed, "url")?.description?.contains("format: uri") == true)
}
