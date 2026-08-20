// mlx-agent - version string.
//
// SINGLE SOURCE OF TRUTH for the version the tool reports. Three places consume it and all
// three are user- or protocol-visible, which is why it is a constant rather than three
// literals that drift:
//   - `mlx-agent --version` (and the usage banner) - what an operator and update-cadabra.sh read
//   - the ACP `initialize` reply's `agentInfo.version` - what the host app displays
//   - the MCP client identity sent to every configured server
//
// Deliberately NOT derived from the Xcode project (MARKETING_VERSION / Info.plist): the tool
// target ships no Info.plist, so a plist lookup would return nil in exactly the shipping
// configuration and leave the CLI reporting nothing.
let agentVersion = "0.2"
