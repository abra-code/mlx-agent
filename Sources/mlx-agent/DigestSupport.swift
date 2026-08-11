// DigestSupport.swift - the glue between AgentDigest and this target.
//
// Two things that belong together because both exist only at the boundary: converting between our
// conversation currency and the summarizer's, and writing down what a condensation did.
//
// The conversions live here rather than on FoundationBackend because nothing about them is
// Foundation Models specific - the prime path needs them too, and step 12's backend-backed
// summarizer will need them for MLX and llama-server.

import AgentDigest
import Foundation
import MLXLMCommon

// MARK: - Conversation currency

/// Our messages, as the summarizer sees them.
///
/// Tool turns are carried across (unlike `FoundationBackend.transcript`, which drops them) because
/// a tool result is often where the only hard fact in an exchange lives, and a SUMMARY of it is
/// safe in a way that replaying the call is not.
func digestTurns(from messages: [Chat.Message]) -> [DigestTurn] {
    messages.map { message in
        let role: DigestRole
        switch message.role {
        case .user: role = .user
        case .assistant: role = .assistant
        case .system: role = .system
        case .tool: role = .tool
        }
        return DigestTurn(role: role, content: message.content)
    }
}

/// The reverse, for the turns the planner PRODUCED.
///
/// Deliberately not used on the verbatim tail: `DigestTurn` carries role and text only, so a
/// round trip would discard tool-call metadata. Callers splice the tail from the originals using
/// `CondenseResult.tailStartIndex` and convert only `CondenseResult.injected`.
func chatMessages(from turns: [DigestTurn]) -> [Chat.Message] {
    turns.map { turn in
        switch turn.role {
        case .user: return .user(turn.content)
        case .assistant: return .assistant(turn.content)
        case .system: return .system(turn.content)
        case .tool: return Chat.Message(role: .tool, content: turn.content)
        }
    }
}

// MARK: - Audit trail

/// Writes one file per condensation, plus an index line.
///
/// Off unless `--digest-dir` is given, matching map's explicit-spool philosophy: an agent that
/// silently accumulated conversation transcripts on disk would be a surprise, and this writes the
/// SOURCE text, which is the whole conversation. Opt-in is the only defensible default.
///
/// What it is for: a digest replaces turns, and "what was dropped" is otherwise unanswerable after
/// the fact. The artifact holds the exact summarized text alongside the digest made from it, so
/// the two can be diffed - which is the only way to judge whether a summarizer is any good on real
/// conversations rather than on the canned ones in the tests.
struct DigestArchive: Sendable {

    let directory: URL

    /// nil when the directory cannot be created, so a bad path degrades to "no artifacts" rather
    /// than failing a restore. The reason is logged once, here, because every later failure would
    /// have the same cause and repeating it per condensation is noise.
    init?(path: String, log: (String) -> Void) {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        do {
            // 0700, not the umask default. What lands here is the user's conversation text, and
            // 0755 in a shared or temp directory makes it readable by every local account.
            try FileManager.default.createDirectory(
                at: url, withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700])
        } catch {
            log("digest archive disabled: cannot create \(url.path): \(error.localizedDescription)")
            return nil
        }
        self.directory = url
    }

    /// Record one condensation. Failures are silent by design - an audit trail that can fail a
    /// restore is worse than no audit trail.
    ///
    /// - Parameters:
    ///   - trigger: what asked for it, e.g. "session/prime" or "context overflow".
    ///   - summarizer: which conformance produced the digest.
    func record(
        _ result: CondenseResult, summarizer: String, trigger: String, now: Date = Date(),
        log: (String) -> Void = { _ in }
    ) {
        guard let digest = result.digest else { return }
        let stamp = Int((now.timeIntervalSince1970 * 1000).rounded())
        let url = uniqueURL(stamp: stamp)

        let payload: [String: Any] = [
            "trigger": trigger,
            "summarizer": summarizer,
            "digest": digest.jsonObject,
            // The exact text the model read, not a re-rendering of the same turns. Correlating it
            // with the digest is the point of keeping it, and `sourceSHA256` covers this string.
            "source": result.source,
            "dropped": ["turns": result.droppedTurns, "bytes": result.droppedBytes],
            "verbatimTailStart": result.tailStartIndex,
            "injectedTurns": result.injected.count,
        ]
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        else { return }
        do {
            try data.write(to: url, options: .atomic)
            // An atomic write creates a NEW inode, so it takes the umask default rather than the
            // directory's mode; tighten it explicitly.
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: url.path)
        } catch {
            log("digest archive: could not write \(url.lastPathComponent): \(error.localizedDescription)")
            return
        }
        appendIndex(
            [
                "ts": stamp,
                "file": url.lastPathComponent,
                "trigger": trigger,
                "summarizer": summarizer,
                "sha256": digest.sourceSHA256,
                "turns": result.droppedTurns,
                "bytes": result.droppedBytes,
            ])
    }

    /// Two condensations in the same millisecond would otherwise overwrite each other. They take
    /// seconds apiece so this should never fire, which is exactly why it is cheap to be sure.
    private func uniqueURL(stamp: Int) -> URL {
        let base = directory.appendingPathComponent("digest-\(stamp).json")
        guard FileManager.default.fileExists(atPath: base.path) else { return base }
        for n in 2...99 {
            let candidate = directory.appendingPathComponent("digest-\(stamp)-\(n).json")
            if !FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        }
        // Unreachable in practice (a condensation takes seconds), but returning `base` here would
        // silently overwrite an existing artifact - the one thing an audit trail must not do.
        return directory.appendingPathComponent("digest-\(stamp)-overflow-\(UUID().uuidString).json")
    }

    /// One line per condensation. Single writer (condensations are serialized by the busy claim on
    /// the prime path and by PassGate inside the backend), so a per-line append is safe and a
    /// reader consumes complete lines as they land. Same convention as map's results.jsonl.
    private func appendIndex(_ object: [String: Any]) {
        let url = directory.appendingPathComponent("digests.jsonl")
        guard var data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        else { return }
        data.append(0x0A)
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
        } else {
            try? data.write(to: url, options: .atomic)
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: url.path)
        }
    }
}
