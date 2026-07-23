# Code coverage: never ship an instrumented binary

Both committed schemes pin coverage OFF, and that matters more than it sounds. **Coverage is a scheme-level setting, not a per-target one** - it applies to the WHOLE build graph, so every dependency gets instrumented alongside the tool. Do not try to control it with per-target `settings:` in `project.yml`: those never reach the SPM package targets, so you get a MIXED binary (deps instrumented, product not) that is worse than either extreme and nearly invisible. Command-line `CLANG_ENABLE_CODE_COVERAGE=NO` has exactly that effect, and `-enableCodeCoverage NO` is rejected outside `test`.

## Why it matters

An instrumented binary is ~35% larger, slower in a hot token loop, and dumps a multi-MB `default.profraw` into its working directory on EVERY run - in the shipped app, wherever the host happens to be running from.

## How it bit us

This package used to be built via an auto-generated scheme (no `Package.swift` scheme was checked in), and **adding a `.testTarget` silently switched coverage on for the whole graph** - reproducible in a two-file package. That trap is gone with a real project and committed schemes (both pin `gatherCoverageData: false` in `project.yml`), but verify the shipped binary anyway.

## Verifying the shipped binary

```
otool -l <binary> | grep -c __llvm_prf_cnts     # 0 = clean
```

Use the section check above, NOT `nm | grep __llvm_prf`: nm reports 0 for a small instrumented binary (false clean). The runtime check never lies - run the binary and see whether a `default.profraw` appears next to it.
