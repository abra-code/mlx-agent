// IdleUnload.swift - the ONE idle-unload policy for every mode that holds MLX model
// weights in-process (mirrors llama-server's --sleep-idle-seconds on the gguf path).
//
// One policy, shared by every mode:
//   acp  - ACPServer arms a DispatchSourceTimer (prompts arrive concurrently, so due-ness
//          is timer + lock business there) and releases through `releaseModel`.
//   map  - MapServer's single-threaded poll loop tracks activity with an `IdleClock` and
//          releases through `releaseModel` between polls.
//   chat / oneshot / gate / bench - one-shot modes: the process exits right after its
//          single completion (verified: a chat invocation is fully gone ~1 s after the
//          answer), which releases everything an idle-unload would. They hold no idle
//          window by construction, so the policy applies trivially and nothing arms.
//          A future long-lived mode must adopt `IdleClock` + `releaseModel` (or the
//          timer pattern) rather than growing its own copy.
//
// The release mechanics live HERE, in one place, because their ordering is load-bearing
// and easy to get subtly wrong: the memory snapshot must be taken BEFORE the owner drops
// its references (dropping moves the weights from MLX's active memory into its buffer
// cache, so a later snapshot reads ~0 active and hides what was freed), and
// GPU.clearCache() must run AFTER (that is what returns the cache to the OS instead of
// holding it for the next allocation).

import Foundation
import MLX

enum IdleUnload {
    /// Seconds of inactivity before the model is released when --idle-unload-seconds is
    /// not supplied; 0 disables the policy.
    static let defaultSeconds: TimeInterval = 600

    /// Drop the model and return MLX's buffer cache to the OS, logging a uniform
    /// "idle-unload: released ..." line through `log`.
    ///
    /// `drop` must clear EVERY strong reference the caller holds to the model stack and
    /// return a label for the log line (typically the model directory's last path
    /// component) - or nil to abort, e.g. when a turn started concurrently; on nil
    /// nothing is released and nothing is logged. Returns whether the release happened.
    @discardableResult
    static func releaseModel(
        afterIdle seconds: TimeInterval, log: (String) -> Void, drop: () -> String?
    ) -> Bool {
        let before = MLX.Memory.snapshot()
        guard let label = drop() else { return false }
        MLX.Memory.clearCache()
        let after = MLX.Memory.snapshot()
        let beforeMiB = (before.activeMemory + before.cacheMemory) / 1_048_576
        let afterMiB = (after.activeMemory + after.cacheMemory) / 1_048_576
        log(
            "idle-unload: released \(label) after \(Int(seconds))s idle; "
                + "resident \(beforeMiB)MiB -> \(afterMiB)MiB")
        return true
    }
}

/// Activity clock for single-threaded loop consumers (map): call `noteActivity()` when
/// work arrives or finishes, ask `dueNow()` between iterations. Timer-style consumers
/// (acp) keep their own scheduling and share only `releaseModel`.
struct IdleClock {
    let seconds: TimeInterval
    private var lastActivity = Date()

    init(seconds: TimeInterval) { self.seconds = seconds }
    mutating func noteActivity() { lastActivity = Date() }
    func dueNow() -> Bool { seconds > 0 && Date().timeIntervalSince(lastActivity) >= seconds }
}
