// Backend.swift - the generation seam the tool loop runs against.
//
// The tool loop (Agent.runTurn) is written against GenerationBackend, NOT against
// ChatSession directly, so an alternative backend (e.g. a remote OpenAI-compatible
// endpoint) can slot in without touching the loop. The currency types are MLXLMCommon's
// (Chat.Message, Generation, ToolSpec, ToolCall) - plain Codable data that any backend
// can translate to/from its own wire format. The MLX impl below wraps a ChatSession with
// toolDispatch LEFT NIL so tool calls surface as `.toolCall` Generation values for the
// external loop to dispatch; this is what lets us enforce the iteration cap / timeout /
// dedup guardrails that the library's internal restart loop (ChatSession.streamMap) does
// not provide.

import Foundation
import MLXLLM
import MLXLMCommon

/// The generation seam the agent loop runs against.
protocol GenerationBackend: AnyObject, Sendable {
    /// Tool specifications the model should see. `nil` disables tool calling
    /// (chat mode). Setting it takes effect on the next `stream(_:)`.
    var tools: [ToolSpec]? { get set }

    /// Stream one model pass: append `messages` to the running context and generate
    /// until the model stops. Tool calls surface as `.toolCall` items - the CALLER
    /// (Agent.runTurn) dispatches them and feeds `.tool(...)` results back on the next
    /// call. The backend preserves conversational state (KV cache) across calls.
    func stream(_ messages: [Chat.Message]) -> AsyncThrowingStream<Generation, Error>

    /// Reset conversational state, keeping configuration (instructions/tools).
    func clear() async
}

/// mlx-swift-lm backend: a ChatSession driven with `toolDispatch == nil` so the tool
/// loop stays external. `streamDetails(to:)` yields `.chunk` / `.toolCall` / `.info`.
///
/// ChatSession is documented as not thread-safe and is not Sendable; this wrapper is
/// `@unchecked Sendable` on the same contract the rest of the agent honors - it is only
/// ever touched from within a single in-flight prompt Task at a time (ACPServer
/// serializes prompts; oneshot runs one turn). We never run two turns concurrently on
/// one session.
final class MLXBackend: GenerationBackend, @unchecked Sendable {
    private let session: ChatSession

    init(_ session: ChatSession) {
        self.session = session
    }

    var tools: [ToolSpec]? {
        get { session.tools }
        set { session.tools = newValue }
    }

    func stream(_ messages: [Chat.Message]) -> AsyncThrowingStream<Generation, Error> {
        session.streamDetails(to: messages)
    }

    func clear() async {
        await session.clear()
    }
}
