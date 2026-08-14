// PassGate.swift - one generation pass at a time, per backend.
//
// Lifted out of MLXBackend unchanged when a second backend needed the same guarantee. The
// hazard it exists for is not MLX-specific: it is the ACP cancel window, and any backend whose
// passes touch shared state has it.
//
// What goes wrong without it. Agent returns `.cancelled` from inside its `for try await`, so
// the pass Task is only canceled ASYNCHRONOUSLY (via the stream iterator's deinit), while
// ACPServer's `defer` clears `promptTask` at once and accepts the next prompt. The abandoned
// pass is then still inside `await genTask.value` with its writes pending while the new pass
// runs against the same state. On MLX that was measured as a hard `[broadcast_shapes]` SIGTRAP
// that kills the process, at round 6-11 of a cancel/prompt loop. Apple's Foundation Models
// throws `GenerationError.concurrentRequests` for the same overlap, which is milder but still
// fails a turn the user did nothing wrong in.
//
// An `actor` alone cannot do this job: actors are reentrant across `await`, so a second call
// would interleave at the first suspension point - which is most of a pass. The gate has to
// serialize the WHOLE pass, which is what acquire/release around the body achieves.

/// Serializes whole passes against one backend.
actor PassGate {
    private var busy = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    /// Non-cancellable on purpose: a waiter that bailed on cancellation would let the next pass
    /// start while this one still owns the shared state, which is the bug this exists to prevent.
    func acquire() async {
        guard busy else {
            busy = true
            return
        }
        await withCheckedContinuation { waiting.append($0) }
    }

    func release() {
        if waiting.isEmpty { busy = false } else { waiting.removeFirst().resume() }
    }
}

/// Run `body` as the only pass in flight on `gate`.
///
/// A free function rather than a method on PassGate because a non-Sendable closure cannot cross
/// into an actor under strict concurrency. The mutual exclusion is the actor's; the scoping is
/// ours. It was a private method on MLXBackend before a second backend needed it verbatim.
func withPass<T>(_ gate: PassGate, _ body: () async throws -> T) async rethrows -> T {
    await gate.acquire()
    do {
        let result = try await body()
        await gate.release()
        return result
    } catch {
        await gate.release()
        throw error
    }
}
