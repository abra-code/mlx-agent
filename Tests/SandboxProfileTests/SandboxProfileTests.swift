import Foundation
import Testing

import SandboxProfile

private let environment = SandboxEnvironment(
    home: "/Users/alice", executableDirectory: "/Apps/Engine/MLX",
    userCacheDirectory: "/private/var/folders/ab/xyz/C")

/// Paths are taken as they are: the tests are about the rules, not about this Mac's disk.
private func profile(_ config: SandboxConfig, warn: (String) -> Void = { _ in }) -> String {
    SandboxProfile.sbpl(config, environment: environment, resolve: { $0 }, warn: warn)
}

private func lines(_ text: String) -> [String] { text.split(separator: "\n").map(String.init) }

@Test("an empty config denies by default, the network included, and can fork but not exec")
func defaults() {
    let text = profile(SandboxConfig())
    let all = lines(text)
    #expect(all.prefix(4) == ["(version 1)", "(deny default)", "(debug deny)", "(import \"bsd.sb\")"])
    #expect(all.contains("(deny network*)"))
    #expect(all.contains("(allow process-fork)"))
    #expect(!text.contains("process-exec"))
    #expect(!text.contains("iokit-open-user-client"))
    #expect(all.contains("(allow signal (target same-sandbox))"))
}

@Test("the executable's folder and the LaunchServices preferences are always readable")
func automaticReads() {
    let all = lines(profile(SandboxConfig()))
    #expect(all.contains("(allow file-read* (subpath \"/Apps/Engine/MLX\"))"))
    #expect(all.contains("(allow file-read* (subpath \"/Users/alice/Library/Preferences/com.apple.LaunchServices\"))"))
}

@Test("folders: read-write covers read-only, a folder inside another is dropped")
func folders() {
    var config = SandboxConfig()
    config.readOnly = ["/models/a", "/models/a/blobs", "/work/out/logs"]
    config.readWrite = ["/work/out", "/work/out"]
    let all = lines(profile(config))
    #expect(all.contains("(allow file-read* (subpath \"/models/a\"))"))
    #expect(!all.contains("(allow file-read* (subpath \"/models/a/blobs\"))"))
    #expect(!all.contains("(allow file-read* (subpath \"/work/out/logs\"))"))
    #expect(all.filter { $0 == "(allow file-read* file-write* (subpath \"/work/out\"))" }.count == 1)
}

@Test("a folder that is only a prefix of another name is not taken as its parent")
func prefixIsNotParent() {
    var config = SandboxConfig()
    config.readOnly = ["/models/a", "/models/ab"]
    let all = lines(profile(config))
    #expect(all.contains("(allow file-read* (subpath \"/models/ab\"))"))
}

@Test("the root folder is dropped, with a warning")
func rootIsRefused() {
    var config = SandboxConfig()
    config.readWrite = ["/"]
    var warnings: [String] = []
    let text = profile(config) { warnings.append($0) }
    #expect(!text.contains("(subpath \"/\")"))
    #expect(warnings.count == 1)
}

@Test("the home folder itself is dropped, with a warning; a folder inside it is not")
func homeIsRefused() {
    var config = SandboxConfig()
    config.readOnly = ["/Users/alice", "/Users/alice/models"]
    config.readWriteFiles = ["/Users/alice"]
    var warnings: [String] = []
    let all = lines(profile(config) { warnings.append($0) })
    #expect(!all.contains("(allow file-read* (subpath \"/Users/alice\"))"))
    #expect(!all.contains("(allow file-read* file-write* (literal \"/Users/alice\"))"))
    #expect(all.contains("(allow file-read* (subpath \"/Users/alice/models\"))"))
    #expect(warnings.count == 2)
}

@Test("JSON: a relative path is an error; a path from the home folder is not")
func parseRelativePaths() {
    func fails(_ json: String) -> Bool {
        do {
            _ = try SandboxConfig.parse(Data(json.utf8))
            return false
        } catch {
            return true
        }
    }
    #expect(fails("{\"read_only\": [\"models\"]}"))
    #expect(fails("{\"read_write_files\": [\"./x\"]}"))
    #expect(fails("{\"exec_files\": [\"\"]}"))
    #expect(fails("{\"unix_socket_connect\": [\"~user/x\"]}"))
    #expect(!fails("{\"read_only\": [\"~/models\", \"~\", \"/abs\"]}"))
}

@Test("a quote or a backslash in a path cannot end the string and add a rule")
func escaping() {
    var config = SandboxConfig()
    config.readOnly = ["/tmp/x\")) (allow default) (allow file-read* (subpath \"/"]
    config.readOnlyFiles = ["/tmp/back\\slash"]
    let text = profile(config)
    #expect(!lines(text).contains("(allow default)"))
    #expect(text.contains("(subpath \"/tmp/x\\\")) (allow default) (allow file-read* (subpath \\\"/\")"))
    #expect(text.contains("(literal \"/tmp/back\\\\slash\")"))
}

@Test("single paths are literals; a program to run is readable and may be started")
func singlePaths() {
    var config = SandboxConfig()
    config.readOnlyFiles = ["/store/Boxes/b1/box.json"]
    config.readWriteFiles = ["/store/Boxes/b1/exec.jsonl"]
    config.execFiles = ["/Users/alice/.local/bin/agent-vm"]
    let all = lines(profile(config))
    #expect(all.contains("(allow file-read* (literal \"/store/Boxes/b1/box.json\"))"))
    #expect(all.contains("(allow file-read* file-write* (literal \"/store/Boxes/b1/exec.jsonl\"))"))
    #expect(all.contains("(allow process-exec (literal \"/Users/alice/.local/bin/agent-vm\"))"))
    #expect(all.contains("(allow file-read* (literal \"/Users/alice/.local/bin/agent-vm\"))"))
    #expect(!all.contains("(allow process-exec*)"))
}

@Test("the network: denied first, then a local port and a socket; or everything")
func network() {
    var config = SandboxConfig()
    config.networkConnect = ["localhost:8099"]
    config.unixSocketConnect = ["/store/Boxes/b1/control.sock"]
    let all = lines(profile(config))
    let deny = all.firstIndex(of: "(deny network*)")
    let port = all.firstIndex(of: "(allow network-outbound (remote ip \"localhost:8099\"))")
    let socket = all.firstIndex(of: "(allow network-outbound (remote unix-socket (path-literal \"/store/Boxes/b1/control.sock\")))")
    #expect(deny != nil && port != nil && socket != nil)
    // Between rules with filters the later one wins; keep the allows after the deny.
    #expect(deny! < port! && deny! < socket!)
    #expect(!all.contains("(allow network*)"))

    config.allowNetwork = true
    let open = lines(profile(config))
    #expect(open.contains("(allow network*)"))
    #expect(!open.contains("(deny network*)"))
}

@Test("the GPU: the two driver connections, the Metal caches, and leave to pass them on")
func gpu() {
    var config = SandboxConfig()
    config.gpu = true
    let text = profile(config)
    let all = lines(text)
    #expect(all.contains("(allow iokit-open-user-client (iokit-user-client-class \"AGXDeviceUserClient\" \"IOSurfaceRootUserClient\"))"))
    #expect(all.contains("(allow syscall-mig)"))
    #expect(all.contains("(allow system-info)"))
    for cache in ["com.apple.metal", "com.apple.metalfe", "com.apple.gpuarchiver"] {
        #expect(all.contains("(allow file-read* file-write* (subpath \"/private/var/folders/ab/xyz/C/\(cache)\"))"))
    }
    #expect(text.contains("(extension-class \"com.apple.app-sandbox.read\") (subpath \"/Apps/Engine/MLX\")"))
    #expect(text.contains("(extension-class \"com.apple.app-sandbox.read-write\") (require-any (subpath \"/private/var/folders/ab/xyz/C/com.apple.metal\")"))
}

@Test("Apple's on-device model: its service and the two preference domains, no GPU of ours")
func foundationModels() {
    var config = SandboxConfig()
    config.foundationModels = true
    let text = profile(config)
    let all = lines(text)
    #expect(all.contains("(allow mach-lookup (global-name \"com.apple.modelmanager\"))"))
    #expect(all.contains("(allow user-preference-read (preference-domain \"kCFPreferencesAnyApplication\" \"com.apple.gms.availability\"))"))
    #expect(all.contains("(allow system-info)"))
    #expect(!text.contains("iokit-open-user-client"))
    #expect(all.contains("(deny network*)"))
}

@Test("services, driver classes and raw rules, raw rules last")
func servicesAndExtras() {
    var config = SandboxConfig()
    config.machServices = ["com.apple.example"]
    config.iokitUserClients = ["SomeUserClient"]
    config.extraRules = ["(allow sysctl-read)"]
    let all = lines(profile(config))
    #expect(all.contains("(allow mach-lookup (global-name \"com.apple.example\"))"))
    #expect(all.contains("(allow iokit-open-user-client (iokit-user-client-class \"SomeUserClient\"))"))
    #expect(all.last == "(allow sysctl-read)")
}

@Test("JSON: every key, in replay's shape plus the engine's")
func parseAll() throws {
    let json = """
        {"import_baseline": false, "read_only": ["/a"], "read_write": ["/b"],
         "read_only_files": ["/c"], "read_write_files": ["/d"], "exec_files": ["/e"],
         "allow_exec": true, "allow_fork": false, "allow_network": true,
         "network_connect": ["localhost:1"], "unix_socket_connect": ["/s"], "gpu": true,
         "foundation_models": true,
         "mach_services": ["m"], "iokit_user_clients": ["i"], "extra_rules": ["(x)"]}
        """
    let config = try SandboxConfig.parse(Data(json.utf8))
    var expected = SandboxConfig()
    expected.importBaseline = false
    expected.readOnly = ["/a"]
    expected.readWrite = ["/b"]
    expected.readOnlyFiles = ["/c"]
    expected.readWriteFiles = ["/d"]
    expected.execFiles = ["/e"]
    expected.allowExec = true
    expected.allowFork = false
    expected.allowNetwork = true
    expected.networkConnect = ["localhost:1"]
    expected.unixSocketConnect = ["/s"]
    expected.gpu = true
    expected.foundationModels = true
    expected.machServices = ["m"]
    expected.iokitUserClients = ["i"]
    expected.extraRules = ["(x)"]
    #expect(config == expected)
    #expect(try SandboxConfig.parse(Data("{}".utf8)) == SandboxConfig())
}

@Test("JSON: an unknown key, a wrong type, a number for a boolean and a remote host are errors")
func parseErrors() {
    func fails(_ json: String) -> Bool {
        do {
            _ = try SandboxConfig.parse(Data(json.utf8))
            return false
        } catch {
            return true
        }
    }
    #expect(fails("[]"))
    #expect(fails("{\"allow_netwrok\": true}"))
    #expect(fails("{\"read_only\": \"/a\"}"))
    #expect(fails("{\"read_only\": [1]}"))
    #expect(fails("{\"allow_network\": 1}"))
    #expect(fails("{\"gpu\": \"yes\"}"))
    #expect(fails("{\"network_connect\": [\"example.com:443\"]}"))
    #expect(fails("{\"network_connect\": [\"*:8080\"]}"))
    #expect(fails("{\"network_connect\": [\"localhost:0\"]}"))
    #expect(fails("{\"network_connect\": [\"localhost:70000\"]}"))
    #expect(fails("{\"network_connect\": [\"localhost:80\\\")) (allow network*\"]}"))
}

@Test("paths resolve to real ones, keeping /private, also for a file that is not there yet")
func resolve() {
    #expect(SandboxPaths.resolve("/tmp") == "/private/tmp")
    #expect(SandboxPaths.resolve("/tmp/no-such-folder-here/x.json") == "/private/tmp/no-such-folder-here/x.json")
    #expect(SandboxPaths.resolve("/") == "/")
    #expect(SandboxPaths.resolve("~") == SandboxPaths.resolve(NSHomeDirectory()))
}

@Test("a quote or a backslash followed by a combining mark is still escaped")
func escapingCombiningMark() {
    var config = SandboxConfig()
    // U+0301 joins the character before it: "\"\u{301}" is one Character, and not "\"".
    config.machServices = ["a\"\u{301})) (allow default) (allow mach-lookup (global-name \"b", "c\\\u{301}"]
    let text = profile(config)
    #expect(text.contains("(global-name \"a\\\"\u{301})) (allow default) (allow mach-lookup (global-name \\\"b\")"))
    #expect(text.contains("(global-name \"c\\\\\u{301}\")"))
    // Every quote in the bytes of those two lines is either escaped or one of the two around
    // the literal.
    for line in lines(text) where line.contains("global-name") {
        var bare = 0
        var escaped = false
        for byte in line.utf8 {
            if escaped { escaped = false } else if byte == 0x5C { escaped = true } else if byte == 0x22 { bare += 1 }
        }
        #expect(bare == 2)
    }
}

@Test("JSON: a string with a NUL character is an error, in any list")
func parseNul() {
    for key in ["read_only", "read_write_files", "unix_socket_connect", "mach_services", "extra_rules"] {
        #expect(throws: SandboxError.self) {
            _ = try SandboxConfig.parse(Data("{\"\(key)\": [\"/Users/alice\\u0000/x\"]}".utf8))
        }
    }
}

@Test("\"..\" after a link is the parent of the link's target, also when the rest is not there yet")
func resolveParentOfLink() throws {
    let manager = FileManager.default
    let base = SandboxPaths.resolve(NSTemporaryDirectory()) + "/sandbox-profile-tests-\(UUID().uuidString)"
    defer { try? manager.removeItem(atPath: base) }
    try manager.createDirectory(atPath: base + "/real/deep/dir", withIntermediateDirectories: true)
    try manager.createSymbolicLink(atPath: base + "/link", withDestinationPath: base + "/real/deep/dir")
    try manager.createSymbolicLink(atPath: base + "/rootlink", withDestinationPath: "/")

    #expect(SandboxPaths.resolve(base + "/link") == base + "/real/deep/dir")
    #expect(SandboxPaths.resolve(base + "/link/..") == base + "/real/deep")
    #expect(SandboxPaths.resolve(base + "/link/../new") == base + "/real/deep/new")
    #expect(SandboxPaths.resolve(base + "/link/new/../../other") == base + "/real/deep/other")
    #expect(SandboxPaths.resolve(base + "/real//deep/./dir/") == base + "/real/deep/dir")
    #expect(SandboxPaths.resolve(base + "/new/./a/../b/") == base + "/new/b")
    // A link to the root, and a path that climbs out of everything, are both the root, which
    // the profile then refuses.
    #expect(SandboxPaths.resolve(base + "/rootlink") == "/")
    #expect(SandboxPaths.resolve(base + "/rootlink/new/..") == "/")
    #expect(SandboxPaths.resolve("/no-such-folder-here/../..") == "/")
}
