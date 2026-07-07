// ACPServer.swift - Phase 1 of the mlx-agent plan (MLXApp docs/10).
//
// A minimal Agent Client Protocol server over stdio: newline-delimited JSON-RPC 2.0,
// mirroring the framing in ActionUIChat's ACPConnection (the authoritative client,
// validated against OpenCode). Phase 1 is CHAT ONLY - no tools yet. It implements the
// subset the ActionUIChat ACP transport actually exercises:
//
//   initialize                -> { protocolVersion: 1, ... }
//   session/new               -> { sessionId, configOptions: [ model select ] }
//   session/prompt            -> streams session/update notifications, resolves { stopReason }
//   session/cancel  (notif)   -> cancels the in-flight turn
//   session/set_config_option -> switches the model, returns refreshed configOptions
//
// mlx-swift-lm streams raw text with <think></think> tags and no reasoning separation
// (verified: MLXLMCommon has no <think> handling), so this server splits those markers
// into agent_thought_chunk vs agent_message_chunk for the client's folded-reasoning UX.
// One live ChatSession per process for v1 (KV cache persists across prompts, so the
// second turn's prefill is cheaper - the plan's Phase-1 acceptance).
//
// Everything on stdout is JSON-RPC; all logging goes to stderr.

import Foundation
import MLXLLM
import MLXLMCommon
import MLXHuggingFace

/// Splits a raw model stream into thought (inside <think>...</think>) and message
/// (everything else) segments, tolerating markers that straddle chunk boundaries.
final class ThinkSplitter {
    enum Kind { case thought, message }
    private var buffer = ""
    private var inThink = false
    private let open = "<think>"
    private let close = "</think>"
    private var keep: Int { max(open.count, close.count) - 1 }

    func feed(_ s: String) -> [(Kind, String)] {
        buffer += s
        var out: [(Kind, String)] = []
        while true {
            let marker = inThink ? close : open
            if let r = buffer.range(of: marker) {
                let before = String(buffer[buffer.startIndex..<r.lowerBound])
                if !before.isEmpty { out.append((inThink ? .thought : .message, before)) }
                buffer.removeSubrange(buffer.startIndex..<r.upperBound)
                inThink.toggle()
            } else {
                // Hold back a short tail that could be the prefix of a marker.
                if buffer.count > keep {
                    let end = buffer.index(buffer.endIndex, offsetBy: -keep)
                    let emit = String(buffer[buffer.startIndex..<end])
                    if !emit.isEmpty { out.append((inThink ? .thought : .message, emit)) }
                    buffer = String(buffer[end...])
                }
                break
            }
        }
        return out
    }

    func flush() -> [(Kind, String)] {
        guard !buffer.isEmpty else { return [] }
        let seg: (Kind, String) = (inThink ? .thought : .message, buffer)
        buffer = ""
        return [seg]
    }
}

final class ACPServer: @unchecked Sendable {

    private let chatInstructions = "You are a helpful assistant."
    private let lock = NSLock()

    private var currentModelDir: String
    private var container: ModelContainer?
    private var session: ChatSession?
    private var sessionID: String?
    private var promptTask: Task<Void, Never>?
    private var stdinBuffer = Data()
    private var doneContinuation: CheckedContinuation<Void, Never>?
    private let writeLock = NSLock()

    init(modelDir: String) {
        self.currentModelDir = modelDir
    }

    // MARK: - Run loop

    func serve() async {
        log("ACP server ready (stdio JSON-RPC). initial model: \(currentModelDir)")
        let stdin = FileHandle.standardInput
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            lock.withLock { doneContinuation = cont }
            stdin.readabilityHandler = { [weak self] handle in
                guard let self else {
                    handle.readabilityHandler = nil
                    return
                }
                let data = handle.availableData
                if data.isEmpty {  // stdin EOF: the client went away
                    handle.readabilityHandler = nil
                    self.finish()
                    return
                }
                let lines = self.lock.withLock {
                    Self.splitLines(buffer: &self.stdinBuffer, appending: data)
                }
                for line in lines { self.dispatch(line) }
            }
        }
    }

    private func finish() {
        let cont = lock.withLock { () -> CheckedContinuation<Void, Never>? in
            let c = doneContinuation
            doneContinuation = nil
            return c
        }
        cont?.resume()
    }

    private static func splitLines(buffer: inout Data, appending data: Data) -> [String] {
        buffer.append(data)
        var lines: [String] = []
        while let nl = buffer.firstIndex(of: 0x0A) {
            let lineData = buffer[buffer.startIndex..<nl]
            lines.append(String(decoding: lineData, as: UTF8.self))
            buffer.removeSubrange(buffer.startIndex...nl)
        }
        return lines
    }

    // MARK: - Dispatch

    private func dispatch(_ line: String) {
        guard !line.isEmpty else { return }
        guard let data = line.data(using: .utf8),
            let msg = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else {
            log("dropping non-JSON line: \(line.prefix(160))")
            return
        }
        guard let method = msg["method"] as? String else {
            log("message without method; dropping")
            return
        }
        let params = msg["params"] as? [String: Any] ?? [:]
        // JSON-RPC ids: the ActionUIChat client (ACPConnection) always uses integer
        // ids. Keep the id as a Sendable Int? so it can cross into prompt Tasks; a
        // missing/non-integer id reads as nil and is treated as a notification.
        let id = (msg["id"] as? NSNumber)?.intValue

        switch method {
        case "initialize":
            // We support only stable protocol v1. Per ACP, if we can't match the
            // client's requested version we answer with the latest we support (1).
            if let requested = (params["protocolVersion"] as? NSNumber)?.intValue, requested != 1 {
                log("client requested protocolVersion \(requested); negotiating to our latest supported (1)")
            }
            respond(
                id,
                [
                    "protocolVersion": 1,
                    "authMethods": [],
                    "agentInfo": ["name": "mlx-agent", "version": "0.1"],
                    "agentCapabilities": [
                        "promptCapabilities": ["audio": false, "image": false, "embeddedContext": false]
                    ],
                ])
        case "session/new":
            Task { await self.handleNewSession(id: id) }
        case "session/prompt":
            handlePrompt(id: id, params: params)
        case "session/cancel":
            handleCancel(params: params)
        case "session/set_config_option":
            Task { await self.handleSetConfigOption(id: id, params: params) }
        default:
            if id != nil {
                respondError(id, -32601, "method not supported by mlx-agent: \(method)")
            } else {
                log("unhandled notification: \(method)")
            }
        }
    }

    // MARK: - Handlers

    private func handleNewSession(id: Int?) async {
        let dir = lock.withLock { currentModelDir }
        do {
            let container = try await loadModel(dir)
            let session = ChatSession(
                container,
                instructions: chatInstructions,
                generateParameters: GenerateParameters(maxTokens: 4096, temperature: 0.7))
            let sid = "mlx-session-1"
            lock.withLock {
                self.container = container
                self.session = session
                self.sessionID = sid
            }
            log("session ready: \(sid) on \((dir as NSString).lastPathComponent)")
            respond(id, ["sessionId": sid, "configOptions": configOptionsJSON()])
        } catch {
            respondError(id, -32000, "session/new failed: \(error.localizedDescription)")
        }
    }

    private func handlePrompt(id: Int?, params: [String: Any]) {
        let (ready, ourSid) = lock.withLock {
            (self.session != nil && self.sessionID != nil, self.sessionID)
        }
        guard ready, let ourSid else {
            respondError(id, -32002, "no active session; call session/new first")
            return
        }
        // ACP carries the target sessionId on every prompt; reject one that isn't ours
        // rather than silently answering for the wrong session.
        if let reqSid = params["sessionId"] as? String, reqSid != ourSid {
            respondError(id, -32602, "unknown session: \(reqSid)")
            return
        }
        let text = Self.promptText(params)
        let task = Task { [weak self] in
            guard let self else { return }
            // Read the (non-Sendable) ChatSession INSIDE the task from self
            // (@unchecked Sendable, lock-guarded) so it never crosses a task boundary.
            let (session, sid) = self.lock.withLock { (self.session, self.sessionID) }
            guard let session, let sid else { return }
            defer { self.lock.withLock { if self.promptTask != nil { self.promptTask = nil } } }
            let splitter = ThinkSplitter()
            var genError: String? = nil
            do {
                for try await chunk in session.streamResponse(to: text) {
                    if Task.isCancelled { break }  // fast path; correctness is the post-loop check
                    for (kind, s) in splitter.feed(chunk) { self.sendUpdate(sid, kind, s) }
                }
            } catch is CancellationError {
                // handled by the isCancelled check below
            } catch {
                genError = error.localizedDescription
                self.log("prompt error: \(error.localizedDescription)")
            }
            // Cancellation ends the stream by terminating iteration (often WITHOUT
            // delivering another chunk), so the in-loop check alone can miss it -
            // detect it here. Cancellation takes precedence over a generation error.
            if Task.isCancelled {
                self.log("turn resolved: stopReason=cancelled")
                self.respond(id, ["stopReason": "cancelled"])
            } else if let genError {
                // Generation failed: reply with a JSON-RPC error, not an out-of-spec
                // stopReason. ACP's StopReason enum is only end_turn / max_tokens /
                // max_turn_requests / refusal / cancelled - "error" is not valid.
                self.respondError(id, -32000, "generation failed: \(genError)")
            } else {
                // Clean turn: flush the splitter's held-back tail before resolving.
                for (kind, s) in splitter.flush() { self.sendUpdate(sid, kind, s) }
                self.log("turn resolved: stopReason=end_turn")
                self.respond(id, ["stopReason": "end_turn"])
            }
        }
        lock.withLock { promptTask = task }
    }

    private func handleCancel(params: [String: Any]) {
        let (task, ourSid) = lock.withLock { (promptTask, sessionID) }
        if let reqSid = params["sessionId"] as? String, let ourSid, reqSid != ourSid {
            log("cancel: ignoring - unknown session \(reqSid)")
            return
        }
        log(task == nil ? "cancel: no in-flight turn" : "cancel: cancelling in-flight turn")
        task?.cancel()
    }

    private func handleSetConfigOption(id: Int?, params: [String: Any]) async {
        guard let configId = params["configId"] as? String,
            let value = params["value"] as? String
        else {
            respondError(id, -32602, "set_config_option requires configId and value")
            return
        }
        guard configId == "model" else {
            respondError(id, -32601, "unknown config option: \(configId)")
            return
        }
        guard let target = availableModels().first(where: { $0.value == value }) else {
            respondError(id, -32602, "unknown model: \(value)")
            return
        }
        do {
            let container = try await loadModel(target.dir)
            let session = ChatSession(
                container,
                instructions: chatInstructions,
                generateParameters: GenerateParameters(maxTokens: 4096, temperature: 0.7))
            lock.withLock {
                self.container = container
                self.session = session
                self.currentModelDir = target.dir
            }
            log("switched model -> \(value)")
            respond(id, ["configOptions": configOptionsJSON()])
        } catch {
            respondError(id, -32000, "model switch failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Model registry

    private func availableModels() -> [(value: String, dir: String)] {
        let current = lock.withLock { currentModelDir }
        let parent = (current as NSString).deletingLastPathComponent
        var out: [(String, String)] = []
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: parent)) ?? []
        for name in entries.sorted() {
            let dir = parent + "/" + name
            var isDir: ObjCBool = false
            let hasConfig = FileManager.default.fileExists(
                atPath: dir + "/config.json", isDirectory: &isDir)
            if hasConfig {
                out.append((name, dir))
            }
        }
        if !out.contains(where: { $0.1 == current }) {
            out.insert(((current as NSString).lastPathComponent, current), at: 0)
        }
        return out
    }

    private func configOptionsJSON() -> [[String: Any]] {
        let current = lock.withLock { (currentModelDir as NSString).lastPathComponent }
        let choices = availableModels().map {
            ["value": $0.value, "name": $0.value] as [String: Any]
        }
        return [
            [
                "id": "model",
                "name": "Model",
                "category": "model",
                "currentValue": current,
                "type": "select",
                "options": choices,
            ]
        ]
    }

    private static func promptText(_ params: [String: Any]) -> String {
        let blocks = params["prompt"] as? [[String: Any]] ?? []
        return
            blocks
            .compactMap { ($0["type"] as? String) == "text" ? $0["text"] as? String : nil }
            .joined(separator: "\n")
    }

    // MARK: - Outbound

    private func sendUpdate(_ sid: String, _ kind: ThinkSplitter.Kind, _ text: String) {
        let su = kind == .thought ? "agent_thought_chunk" : "agent_message_chunk"
        send([
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": [
                "sessionId": sid,
                "update": [
                    "sessionUpdate": su,
                    "content": ["type": "text", "text": text],
                ],
            ],
        ])
    }

    private func respond(_ id: Int?, _ result: [String: Any]) {
        guard let id else { return }  // a notification arrived where a request was expected
        send(["jsonrpc": "2.0", "id": id, "result": result])
    }

    private func respondError(_ id: Int?, _ code: Int, _ message: String) {
        guard let id else {
            log("error for a notification (\(message)); nothing to reply to")
            return
        }
        send(["jsonrpc": "2.0", "id": id, "error": ["code": code, "message": message]])
    }

    private func send(_ message: [String: Any]) {
        guard
            var data = try? JSONSerialization.data(
                withJSONObject: message, options: [.withoutEscapingSlashes])
        else {
            log("failed to encode outbound message")
            return
        }
        data.append(0x0A)
        writeLock.withLock {
            FileHandle.standardOutput.write(data)
        }
    }

    private func log(_ s: String) {
        FileHandle.standardError.write(Data("[mlx-agent acp] \(s)\n".utf8))
    }
}
