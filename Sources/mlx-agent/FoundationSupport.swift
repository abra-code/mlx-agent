// FoundationSupport.swift - Apple Foundation Models: is it usable, and what to say when it is not.
//
// This is the ONE place that decides whether the on-device system model can be used, so no
// caller has to remember the three separate gates it sits behind. The tool ships as a single
// arm64 binary whose deployment target is macOS 14.0 while the framework is macOS 26.0, so
// all three are load-bearing:
//
//   compile   #if canImport(FoundationModels) - an older Xcode has no module to import at all.
//   link      FoundationModels.framework is WEAK-linked (project.yml). Without that, dyld
//             refuses to launch the binary on macOS 14-25, where the dylib does not exist -
//             the tool would stop starting for every mode, not just the FM ones.
//   runtime   #available around every use, PLUS the availability check below.
//
// The runtime one is the gate that is easy to get wrong. `canImport` and `#available` both
// pass on a macOS 26 Mac with Apple Intelligence switched off; without asking the framework
// itself, the failure then arrives as a thrown error out of the first generation instead of
// as a clean, actionable "not available" before any work starts. The three unavailable
// reasons are genuinely different situations for the person running this - one is permanent,
// one is a settings toggle, one just needs a wait - so they are not collapsed into one string.
//
// Nothing here loads a model or performs inference: `probe()` is cheap enough to call on a
// startup path, which is what lets a mode fail fast with a useful message.

import Foundation

#if canImport(FoundationModels)
    import FoundationModels
#endif

/// Whether the on-device system language model can be used right now, and why not if it cannot.
enum FMAvailability {

    /// The outcome of `probe()`. `unavailable` carries a description of the situation and a
    /// separate sentence saying what the user can do about it - callers print both, and the
    /// split keeps the actionable half from being lost when a caller wants a one-line summary.
    enum Status: Equatable {
        /// The framework is present and the model is ready to serve requests.
        case available
        /// This macOS predates the framework (< 26.0). Nothing to enable; only an OS upgrade helps.
        case osTooOld
        /// Built against an SDK with no FoundationModels module, so the support is not compiled in.
        case notBuilt
        /// The framework is here but the model is not usable. See `FMAvailability.reasonText`.
        case unavailable(reason: String, actionable: String)

        var isAvailable: Bool { self == .available }

        /// One line for logs and error payloads: reason and remedy joined, or a bare description.
        var summary: String {
            switch self {
            case .available:
                return "available"
            case .osTooOld:
                return
                    "Apple Foundation Models requires macOS 26 or later; this system is older"
            case .notBuilt:
                return
                    "this build has no Foundation Models support (compiled against an SDK without the framework)"
            case .unavailable(let reason, let actionable):
                return "\(reason); \(actionable)"
            }
        }
    }

    /// Ask the framework whether the system model is usable. Cheap, loads nothing.
    static func probe() -> Status {
        #if canImport(FoundationModels)
            guard #available(macOS 26.0, *) else { return .osTooOld }
            switch SystemLanguageModel.default.availability {
            case .available:
                return .available
            case .unavailable(let reason):
                let (what, todo) = reasonText(reason)
                return .unavailable(reason: what, actionable: todo)
            @unknown default:
                // A reason added after this was written. Report it as unavailable rather than
                // guessing it is fine: an optimistic default here fails later and less clearly.
                return .unavailable(
                    reason: "the on-device model is unavailable for an unrecognized reason",
                    actionable: "check Apple Intelligence in System Settings")
            }
        #else
            return .notBuilt
        #endif
    }

    /// The context size the model will accept, or nil when it is not usable.
    ///
    /// Worth reading before designing anything around this model: it is 4096 tokens, and that
    /// is not a placeholder. `contextSize` back-deploys to a hard-coded 4096 before macOS 26.4,
    /// so on older systems this is a constant rather than a measurement - it is still the right
    /// number to plan against, just not one the model was asked for.
    static func contextSize() -> Int? {
        #if canImport(FoundationModels)
            guard #available(macOS 26.0, *), SystemLanguageModel.default.isAvailable else {
                return nil
            }
            return SystemLanguageModel.default.contextSize
        #else
            return nil
        #endif
    }

    #if canImport(FoundationModels)
        @available(macOS 26.0, *)
        private static func reasonText(
            _ reason: SystemLanguageModel.Availability.UnavailableReason
        ) -> (String, String) {
            switch reason {
            case .deviceNotEligible:
                return (
                    "this Mac is not eligible for Apple Intelligence",
                    "the on-device model cannot be enabled on this hardware"
                )
            case .appleIntelligenceNotEnabled:
                return (
                    "Apple Intelligence is switched off",
                    "turn it on in System Settings > Apple Intelligence & Siri, then retry"
                )
            case .modelNotReady:
                return (
                    "the model assets are not downloaded yet",
                    "leave the Mac on power and network for a while, then retry"
                )
            @unknown default:
                return (
                    "the on-device model is unavailable for an unrecognized reason",
                    "check Apple Intelligence in System Settings"
                )
            }
        }
    #endif
}

#if canImport(FoundationModels)
    /// Turn a Foundation Models error into something worth showing a user.
    ///
    /// The framework's own `localizedDescription` is serviceable but says nothing about what to
    /// do, and two of these cases are routinely the caller's fault rather than a malfunction:
    /// `exceededContextWindowSize` is a 4096-token budget being spent, and `guardrailViolation`
    /// is a refusal, not a crash. Callers that can recover (by summarizing, by retrying) should
    /// match on the case itself - this function is for the point where the error is being
    /// reported rather than handled.
    @available(macOS 26.0, *)
    func fmUserFacingError(_ error: Error) -> String {
        if let generation = error as? LanguageModelSession.GenerationError {
            switch generation {
            case .exceededContextWindowSize:
                return
                    "the conversation is larger than the on-device model's \(FMAvailability.contextSize() ?? 4096)-token context window"
            case .assetsUnavailable:
                return
                    "the on-device model assets are unavailable; check Apple Intelligence in System Settings"
            case .guardrailViolation:
                return "the on-device model's safety system declined this request"
            case .unsupportedGuide:
                return "the requested output schema is not expressible for the on-device model"
            case .unsupportedLanguageOrLocale:
                return "the on-device model does not support this language"
            case .decodingFailure:
                return "the on-device model produced output that did not match the requested schema"
            case .rateLimited:
                return "the on-device model is rate limited; retry shortly"
            case .concurrentRequests:
                return "another request is already running on this session"
            case .refusal(let refusal, _):
                // The framework can explain a refusal, but only by generating - which is another
                // model call that can itself fail or hang. Report the fact synchronously here and
                // leave `refusal.explanation` to a caller that genuinely wants to spend a pass.
                _ = refusal
                return "the on-device model refused to answer this prompt"
            @unknown default:
                return "on-device model error: \(generation.localizedDescription)"
            }
        }
        if let toolCall = error as? LanguageModelSession.ToolCallError {
            return "tool \"\(toolCall.tool.name)\" failed: \(toolCall.underlyingError.localizedDescription)"
        }
        return error.localizedDescription
    }
#endif
