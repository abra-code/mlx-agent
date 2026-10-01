// SandboxProfile.swift - a Seatbelt (macOS sandbox) profile for the engine process itself.
//
// mlx-agent runs a model, and a model's output decides which tools are called with which
// arguments. Confining the process that holds the model means a mistake or a hostile prompt cannot
// make THIS process read the user's files or reach the network, whatever happens to the tools.
// `--sandbox-profile <json>` loads a config in the shape below, turns it into a profile in
// Seatbelt's policy language (SBPL), and applies it at startup, before a model or a prompt is read.
// Once applied it covers this process and every child it starts, and cannot be loosened.
//
// The JSON shape is replay's (github.com/abra-code/replay, sandbox/README.md) with more keys.
// Every key is optional; an unknown key is an error, since a misspelled key in a security profile
// must not be read as "nothing asked for".
//
//     {
//       "read_only":           ["/dir", ...],   folders readable, with everything under them
//       "read_write":          ["/dir", ...],   folders readable and writable
//       "read_only_files":     ["/file", ...],  single files or folders, that path only
//       "read_write_files":    ["/file", ...],
//       "exec_files":          ["/file", ...],  programs this process may start (and read)
//       "allow_exec":          false,           start ANY program; default false
//       "allow_fork":          true,            fork; default true (starting a child needs it)
//       "allow_network":       false,           every network operation; default false
//       "network_connect":     ["localhost:8080"],  outgoing connections to a port of this Mac
//       "unix_socket_connect": ["/path/control.sock"],
//       "gpu":                 false,           what Metal needs: the GPU, its caches, its compiler
//       "foundation_models":   false,           what Apple's on-device model needs: its service
//       "mach_services":       ["com.apple.x"], system services reachable by name
//       "iokit_user_clients":  ["ClassName"],   driver connections by class
//       "import_baseline":     true,            Apple's bsd.sb baseline (dyld, /dev, Mach basics)
//       "extra_rules":         ["(allow ...)"]  raw SBPL, appended last
//     }
//
// This file is Foundation-only so its rules are unit-tested without linking MLX or Metal. It
// builds the profile text; `Sandbox.apply` is the one call that changes the process.

import Foundation

public struct SandboxConfig: Equatable, Sendable {
    public var importBaseline = true
    public var readOnly: [String] = []
    public var readWrite: [String] = []
    public var readOnlyFiles: [String] = []
    public var readWriteFiles: [String] = []
    public var execFiles: [String] = []
    public var allowExec = false
    public var allowFork = true
    public var allowNetwork = false
    public var networkConnect: [String] = []
    public var unixSocketConnect: [String] = []
    public var gpu = false
    public var foundationModels = false
    public var machServices: [String] = []
    public var iokitUserClients: [String] = []
    public var extraRules: [String] = []

    public init() {}
}

public enum SandboxError: LocalizedError, Equatable {
    case unreadable(String)
    case malformed(String)
    case unavailable
    case refused(String)

    public var errorDescription: String? {
        switch self {
        case .unreadable(let path): return "cannot read the sandbox profile at \(path)"
        case .malformed(let why): return "sandbox profile: \(why)"
        case .unavailable: return "this system has no sandbox_init_with_parameters, so the sandbox cannot be applied"
        case .refused(let why): return "the sandbox profile was refused: \(why)"
        }
    }
}

extension SandboxConfig {
    /// The config in a JSON file. Throws for a file that cannot be read, is not a JSON object,
    /// holds a key this version does not know, or a value of the wrong type.
    public static func load(_ path: String) throws -> SandboxConfig {
        guard let data = FileManager.default.contents(atPath: path) else {
            throw SandboxError.unreadable(path)
        }
        return try parse(data)
    }

    public static func parse(_ data: Data) throws -> SandboxConfig {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw SandboxError.malformed("expected a JSON object")
        }
        var config = SandboxConfig()
        for (key, value) in root {
            switch key {
            case "import_baseline": config.importBaseline = try bool(value, key)
            case "read_only": config.readOnly = try strings(value, key)
            case "read_write": config.readWrite = try strings(value, key)
            case "read_only_files": config.readOnlyFiles = try strings(value, key)
            case "read_write_files": config.readWriteFiles = try strings(value, key)
            case "exec_files": config.execFiles = try strings(value, key)
            case "allow_exec": config.allowExec = try bool(value, key)
            case "allow_fork": config.allowFork = try bool(value, key)
            case "allow_network": config.allowNetwork = try bool(value, key)
            case "network_connect": config.networkConnect = try strings(value, key)
            case "unix_socket_connect": config.unixSocketConnect = try strings(value, key)
            case "gpu": config.gpu = try bool(value, key)
            case "foundation_models": config.foundationModels = try bool(value, key)
            case "mach_services": config.machServices = try strings(value, key)
            case "iokit_user_clients": config.iokitUserClients = try strings(value, key)
            case "extra_rules": config.extraRules = try strings(value, key)
            default: throw SandboxError.malformed("unknown key \"\(key)\"")
            }
        }
        // A path is absolute, or starts at the home folder ("~"). A relative one would depend on
        // the folder the process happened to start in.
        let pathKeys: [(String, [String])] = [
            ("read_only", config.readOnly), ("read_write", config.readWrite),
            ("read_only_files", config.readOnlyFiles), ("read_write_files", config.readWriteFiles),
            ("exec_files", config.execFiles), ("unix_socket_connect", config.unixSocketConnect),
        ]
        for (key, paths) in pathKeys {
            for path in paths where !(path.hasPrefix("/") || path == "~" || path.hasPrefix("~/")) {
                throw SandboxError.malformed("\"\(key)\" takes absolute paths (or \"~/...\"), not \"\(path)\"")
            }
        }
        for target in config.networkConnect where !isLocalPort(target) {
            throw SandboxError.malformed(
                "\"network_connect\" takes \"localhost:<port>\", not \"\(target)\"")
        }
        return config
    }

    /// JSON true or false only. NSNumber bridges 0 and 1 to Bool as well, so the check is on the
    /// boolean type itself: `"allow_network": 1` must not read as true.
    private static func bool(_ value: Any, _ key: String) throws -> Bool {
        guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
            throw SandboxError.malformed("\"\(key)\" must be true or false")
        }
        return number.boolValue
    }

    private static func strings(_ value: Any, _ key: String) throws -> [String] {
        guard let list = value as? [Any] else {
            throw SandboxError.malformed("\"\(key)\" must be an array of strings")
        }
        var out: [String] = []
        for item in list {
            guard let text = item as? String else {
                throw SandboxError.malformed("\"\(key)\" contains something that is not a string")
            }
            // A NUL ends a C string: realpath would resolve only what precedes it (a broader
            // folder than the one written), and the profile text itself would end there.
            guard !text.unicodeScalars.contains("\0") else {
                throw SandboxError.malformed("\"\(key)\" contains a string with a NUL character")
            }
            out.append(text)
        }
        return out
    }

    /// "localhost:<port>", a port from 1 to 65535. Seatbelt's `remote ip` filter takes only
    /// "localhost" or "*" as the host, and "*" would be any host on that port.
    static func isLocalPort(_ target: String) -> Bool {
        let prefix = "localhost:"
        guard target.hasPrefix(prefix) else { return false }
        let digits = target.dropFirst(prefix.count)
        guard !digits.isEmpty, digits.count <= 5, digits.allSatisfy({ $0 >= "0" && $0 <= "9" }),
            let port = Int(digits)
        else { return false }
        return port >= 1 && port <= 65535
    }
}

/// What the profile needs to know about the process and the Mac. Passed in, so the tests state
/// them and the tool reads them (`SandboxEnvironment.current`).
public struct SandboxEnvironment: Equatable, Sendable {
    /// The user's home folder; empty when unknown.
    public var home: String
    /// The folder of this executable, resolved: beside it sits the Metal shader bundle.
    public var executableDirectory: String
    /// The per-user cache folder (`getconf DARWIN_USER_CACHE_DIR`), resolved, no trailing slash.
    public var userCacheDirectory: String

    public init(home: String, executableDirectory: String, userCacheDirectory: String) {
        self.home = home
        self.executableDirectory = executableDirectory
        self.userCacheDirectory = userCacheDirectory
    }
}

public enum SandboxProfile {
    /// The driver connections Metal opens on Apple silicon. Without them Metal sees no device.
    static let gpuUserClients = ["AGXDeviceUserClient", "IOSurfaceRootUserClient"]
    /// The per-user folders Metal's shader compiler and caches use, under the user cache folder.
    static let gpuCacheFolders = ["com.apple.metal", "com.apple.metalfe", "com.apple.gpuarchiver"]
    /// The service that runs Apple's on-device model: the framework sends it the prompt and gets
    /// the answer back, so the model itself runs outside this process and outside its sandbox.
    static let foundationModelsService = "com.apple.modelmanager"
    /// The preference domains the framework reads to learn whether the model can be used.
    static let foundationModelsPreferences = ["kCFPreferencesAnyApplication", "com.apple.gms.availability"]

    /// The profile text for a config. A path is resolved (symbolic links followed, as the kernel
    /// compares real paths) and dropped when it is empty or resolves to "/", which would grant
    /// everything; `warn` hears why.
    public static func sbpl(
        _ config: SandboxConfig, environment: SandboxEnvironment,
        resolve: (String) -> String = SandboxPaths.resolve,
        warn: (String) -> Void = { _ in }
    ) -> String {
        // The whole disk and the whole home folder are never granted: the engine needs neither,
        // and either would make the profile pointless.
        let home = environment.home.isEmpty ? "" : resolve(environment.home)
        func refused(_ real: String, _ path: String) -> Bool {
            if real == "/" || real.isEmpty {
                warn("sandbox path \"\(path)\" resolves to \"/\" and was dropped: it would grant the whole disk")
                return true
            }
            if !home.isEmpty, real == home {
                warn("sandbox path \"\(path)\" is the home folder and was dropped: grant the folders inside it that are needed")
                return true
            }
            return false
        }
        func folders(_ paths: [String]) -> [String] {
            var out: [String] = []
            for path in paths where !path.isEmpty {
                let real = resolve(path)
                if !refused(real, path) { out.append(real) }
            }
            return SandboxPaths.withoutCovered(out)
        }
        func files(_ paths: [String]) -> [String] {
            var seen = Set<String>()
            var out: [String] = []
            for path in paths where !path.isEmpty {
                let real = resolve(path)
                if !refused(real, path), seen.insert(real).inserted { out.append(real) }
            }
            return out
        }

        var lines: [String] = ["(version 1)", "(deny default)", "(debug deny)"]
        if config.importBaseline { lines.append("(import \"bsd.sb\")") }
        lines.append("")
        if config.allowExec { lines.append("(allow process-exec*)") }
        if config.allowFork { lines.append("(allow process-fork)") }
        // A child this process started must be killable by it (a tool server at exit, a timeout).
        lines.append("(allow signal (target same-sandbox))")

        // What every process here needs: the LaunchServices preferences the system reads at
        // startup, and this executable's own folder.
        var readOnly = config.readOnly
        if !environment.home.isEmpty {
            readOnly.append(environment.home + "/Library/Preferences/com.apple.LaunchServices")
        }
        if !environment.executableDirectory.isEmpty { readOnly.append(environment.executableDirectory) }

        let readWrite = folders(config.readWrite)
        let readOnlyResolved = folders(readOnly).filter { !SandboxPaths.isCovered($0, by: readWrite) }
        if !readOnlyResolved.isEmpty {
            lines.append("")
            lines.append("; folders, read-only")
            for dir in readOnlyResolved { lines.append("(allow file-read* (subpath \(quoted(dir))))") }
        }
        if !readWrite.isEmpty {
            lines.append("")
            lines.append("; folders, read-write")
            for dir in readWrite { lines.append("(allow file-read* file-write* (subpath \(quoted(dir))))") }
        }
        let readOnlyFiles = files(config.readOnlyFiles)
        let readWriteFiles = files(config.readWriteFiles)
        let execFiles = files(config.execFiles)
        if !readOnlyFiles.isEmpty || !readWriteFiles.isEmpty || !execFiles.isEmpty {
            lines.append("")
            lines.append("; single paths")
            for file in readOnlyFiles { lines.append("(allow file-read* (literal \(quoted(file))))") }
            for file in readWriteFiles { lines.append("(allow file-read* file-write* (literal \(quoted(file))))") }
            for file in execFiles {
                lines.append("(allow file-read* (literal \(quoted(file))))")
                lines.append("(allow process-exec (literal \(quoted(file))))")
            }
        }

        // Always an explicit network rule, so the intent does not rest on the default alone.
        // Measured on macOS 27 with sandbox-exec: a rule with a filter wins over a rule without
        // one in either order, and between two rules with filters the later one wins. So the
        // narrow allows below hold against the deny (and are written after it anyway), and a
        // narrow allow of the baseline survives it too: system.sb's syslog socket.
        lines.append("")
        if config.allowNetwork {
            lines.append("(allow network*)")
        } else {
            lines.append("(deny network*)")
            for target in config.networkConnect where SandboxConfig.isLocalPort(target) {
                lines.append("(allow network-outbound (remote ip \(quoted(target))))")
            }
            for socket in files(config.unixSocketConnect) {
                lines.append("(allow network-outbound (remote unix-socket (path-literal \(quoted(socket)))))")
            }
        }

        var userClients = config.iokitUserClients
        if config.gpu {
            for name in gpuUserClients where !userClients.contains(name) { userClients.append(name) }
        }
        if config.gpu {
            // Measured on macOS 27 for MLX and for llama.cpp's Metal code: the two driver
            // connections, the Metal caches, and leave to hand Metal's separate compiler service
            // access to those caches and to the shader bundle beside this executable.
            lines.append("")
            lines.append("; the GPU (Metal)")
            lines.append("(allow syscall-mig)")
            lines.append("(allow system-info)")
            lines.append("(allow user-preference-read (preference-domain \"kCFPreferencesAnyApplication\"))")
            let caches = environment.userCacheDirectory.isEmpty
                ? [] : gpuCacheFolders.map { environment.userCacheDirectory + "/" + $0 }
            for cache in caches { lines.append("(allow file-read* file-write* (subpath \(quoted(cache))))") }
            if !environment.executableDirectory.isEmpty, environment.executableDirectory != "/" {
                lines.append(
                    "(allow file-issue-extension (require-all (extension-class \"com.apple.app-sandbox.read\") "
                        + "(subpath \(quoted(environment.executableDirectory)))))")
            }
            if !caches.isEmpty {
                let any = caches.map { "(subpath \(quoted($0)))" }.joined(separator: " ")
                lines.append(
                    "(allow file-issue-extension (require-all (extension-class \"com.apple.app-sandbox.read-write\") "
                        + "(require-any \(any))))")
            }
        }
        if !userClients.isEmpty {
            let names = userClients.map(quoted).joined(separator: " ")
            lines.append("(allow iokit-open-user-client (iokit-user-client-class \(names)))")
        }
        var services = config.machServices
        if config.foundationModels {
            // Measured on macOS 27: without the service the framework fails with ModelManagerError
            // 1008, and without the preferences it reports the model as not downloaded.
            lines.append("")
            lines.append("; Apple's on-device model (Foundation Models)")
            if !config.gpu { lines.append("(allow system-info)") }
            let domains = foundationModelsPreferences.map(quoted).joined(separator: " ")
            lines.append("(allow user-preference-read (preference-domain \(domains)))")
            if !services.contains(foundationModelsService) { services.append(foundationModelsService) }
        }
        if !services.isEmpty {
            lines.append("")
            lines.append("; system services")
            for name in services { lines.append("(allow mach-lookup (global-name \(quoted(name))))") }
        }
        if !config.extraRules.isEmpty {
            lines.append("")
            lines.append("; extra rules")
            lines.append(contentsOf: config.extraRules)
        }
        return lines.joined(separator: "\n") + "\n"
    }

    /// An SBPL string literal. Backslashes and double quotes are escaped, so a crafted folder
    /// name cannot close the string and add rules of its own.
    static func quoted(_ text: String) -> String {
        // Scalar by scalar: a quote followed by a combining mark is ONE Character, not equal to
        // "\"", and would pass unescaped while the profile parser, reading bytes, sees the quote.
        var out = String.UnicodeScalarView()
        out.append("\"")
        for scalar in text.unicodeScalars {
            if scalar == "\\" || scalar == "\"" { out.append("\\") }
            out.append(scalar)
        }
        out.append("\"")
        return String(out)
    }
}

public enum SandboxPaths {
    /// The real path: symbolic links followed with realpath(3), which keeps "/private" (the
    /// kernel compares real paths, and Foundation's own resolver drops that prefix). For a path
    /// that does not exist yet, the longest existing part is resolved and the rest kept.
    ///
    /// ".." is left to realpath wherever the path exists, so "link/.." is the parent of the
    /// link's target, as the kernel reads it. (NSString's standardizingPath removes "link/.."
    /// as text when the whole path does not exist, which names a different folder.) Only in the
    /// part that does not exist, where no link can be, is ".." applied as text.
    public static func resolve(_ path: String) -> String {
        var expanded = (path as NSString).expandingTildeInPath
        if !expanded.hasPrefix("/") {
            expanded = FileManager.default.currentDirectoryPath + "/" + expanded
        }
        var head = expanded.split(separator: "/").map(String.init)
        var tail: [String] = []
        while true {
            if let real = realpath("/" + head.joined(separator: "/"), nil) {
                defer { free(real) }
                var out = String(cString: real).split(separator: "/").map(String.init)
                for part in tail.reversed() where part != "." {
                    if part != ".." {
                        out.append(part)
                    } else if !out.isEmpty {
                        out.removeLast()
                    }
                }
                return "/" + out.joined(separator: "/")
            }
            if head.isEmpty { break }
            tail.append(head.removeLast())
        }
        return expanded
    }

    /// True when `path` is one of `folders` or under one.
    public static func isCovered(_ path: String, by folders: [String]) -> Bool {
        folders.contains { path == $0 || path.hasPrefix($0 + "/") }
    }

    /// The folders, without duplicates and without any that another one already covers.
    public static func withoutCovered(_ folders: [String]) -> [String] {
        var out: [String] = []
        for folder in Array(Set(folders)).sorted() where !isCovered(folder, by: out) {
            out.append(folder)
        }
        return out
    }
}

extension SandboxEnvironment {
    /// The facts of this process: its home folder, its executable's folder, the user cache folder.
    public static var current: SandboxEnvironment {
        var executable = ""
        if let path = Bundle.main.executablePath {
            executable = (SandboxPaths.resolve(path) as NSString).deletingLastPathComponent
        }
        var cache = ""
        var buffer = [CChar](repeating: 0, count: Int(PATH_MAX))
        if confstr(_CS_DARWIN_USER_CACHE_DIR, &buffer, buffer.count) > 0 {
            cache = SandboxPaths.resolve(String(cString: buffer))
            while cache.count > 1, cache.hasSuffix("/") { cache.removeLast() }
        }
        return SandboxEnvironment(
            home: NSHomeDirectory(), executableDirectory: executable, userCacheDirectory: cache)
    }
}

public enum Sandbox {
    private typealias InitFunction = @convention(c) (
        UnsafePointer<CChar>, UInt64, UnsafePointer<UnsafePointer<CChar>?>?,
        UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
    ) -> Int32

    /// Applies the profile to this process, for good. Throws when the system lacks the call or
    /// refuses the profile (a syntax error in a raw rule, or a process already sandboxed: a
    /// second profile cannot be applied over a first).
    ///
    /// `sandbox_init_with_parameters` is in libSystem without a public header, so it is looked up
    /// by name rather than declared: a system without it is an error to report, not a launch
    /// failure.
    public static func apply(_ profile: String) throws {
        guard let symbol = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "sandbox_init_with_parameters") else {
            throw SandboxError.unavailable
        }
        let initialize = unsafeBitCast(symbol, to: InitFunction.self)
        var message: UnsafeMutablePointer<CChar>?
        let status = profile.withCString { initialize($0, 0, nil, &message) }
        if status != 0 {
            let why = message.map { String(cString: $0) } ?? "unknown error"
            if let message { free(message) }
            throw SandboxError.refused(why)
        }
    }
}
