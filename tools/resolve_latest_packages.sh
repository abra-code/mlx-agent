#!/bin/bash
# resolve_latest_packages.sh - re-resolve mlx-agent's SPM dependencies to the newest versions the
# version ranges allow, then say which direct dependencies have a newer release that a range
# still blocks.
#
# The app update scripts (Cadabra's update-cadabra.sh, Interpreter's update_interpreter.sh) run
# this before every agent build, so a shipped agent never quietly carries a months-old MLX. It
# lives HERE, not in each app repo, because every step below exists to dodge a trap of this
# repo's package graph, and two copies of that knowledge had already drifted apart.
#
# Three traps, each of which leaves the build green on old versions without a word:
#
#   1. Package.resolved wins. There is no `xcodebuild -updatePackages`, -resolvePackageDependencies
#      HONORS an existing Package.resolved, and `swift package update` needs a Package.swift,
#      which this repo deliberately does not have (see project.yml's header). Resolving with the
#      file absent is the only CLI path that upgrades.
#   2. Stale checkouts win too. With no Package.resolved, SwiftPM still prefers whatever is
#      checked out under build/SourcePackages whenever that satisfies the ranges - even after
#      fetching every repository. Measured 2026-10-15: an August build/ re-resolved ten packages
#      DOWNWARD (swift-jinja 2.5.1 -> 2.4.2) while logging "Updating from" for each. So the
#      checkouts and workspace-state.json are removed first. The fetched clones under
#      repositories/ stay, so this costs a local re-checkout, not a re-download.
#   3. A range can block a release. mlx-swift-lm 3.32 needs mlx-swift 0.32, which
#      `minorVersion: 0.31.4` in project.yml cannot reach, so the resolve "succeeded" on 3.31.4.
#      No resolver output says so; only comparing against upstream's tags does, which is the
#      last step here (skip it with --no-upstream-check, e.g. offline).
#
# The old Package.resolved is copied aside first and put back if resolution fails, so a network
# outage or a yanked tag leaves the pins exactly as they were. The copy lives in a temp dir, not
# beside the original, and a signal handler restores it: the resolution takes minutes on a cold
# checkout, so Ctrl-C (or a closed terminal - HUP) lands inside the window where the file is
# gone, and a .bak beside it would be one `git add -A` from being committed. On success the
# update lands as a git diff here - deliberately: Package.resolved is committed so new versions
# get reviewed.
#
# Usage: tools/resolve_latest_packages.sh [--no-upstream-check]
# Exit status: 0 resolved (held-back releases are reported, not fatal); 1 failed, previous pins
# restored; 2 usage error; 128+N interrupted by signal N, previous pins restored.

SCRIPT_DIR="$(CDPATH= cd -P -- "$(/usr/bin/dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -P -- "$SCRIPT_DIR/.." && pwd)"
PROJECT="$REPO_ROOT/mlx-agent.xcodeproj"
RESOLVED="$PROJECT/project.xcworkspace/xcshareddata/swiftpm/Package.resolved"
SOURCE_PACKAGES="$REPO_ROOT/build/SourcePackages"
PIN_AWK="$SCRIPT_DIR/package_pin_versions.awk"

DO_UPSTREAM_CHECK="yes"
while [ $# -gt 0 ]; do
    case "$1" in
        --no-upstream-check) DO_UPSTREAM_CHECK="no" ;;
        --help|-h)
            echo "Usage: $0 [--no-upstream-check]"
            echo "  Re-resolve mlx-agent's packages to the newest versions project.yml allows,"
            echo "  then report direct dependencies whose newest release a version range blocks."
            echo "  --no-upstream-check  skip that report (it queries each package's git remote)"
            exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done

if [ -t 1 ]; then
    YELLOW=$(printf '\033[93m'); RED=$(printf '\033[91m'); RESET=$(printf '\033[0m')
else
    YELLOW=""; RED=""; RESET=""
fi

# Globals rather than locals because the signal handler has to see them.
WORK_DIR=""
BACKUP=""
PINS_MOVED="no"

cleanup() {
    # Only ever remove the mktemp dir this run created; never an inherited or empty value.
    if [ -n "$WORK_DIR" ] && [ -d "$WORK_DIR" ]; then
        /bin/rm -rf "$WORK_DIR"
    fi
}

fail() { echo "${RED}$*${RESET}" >&2; cleanup; exit 1; }

# Put the pins back as they were before this run: the saved copy, or no file at all if there
# was none. Always discards what the resolution wrote rather than keeping a file that happens to
# exist - SwiftPM writes Package.resolved in place, before the checkouts finish, so a signal can
# leave it truncated. The copy lands on a temp name in the same directory and is renamed into
# place, so a second, different signal arriving inside this function can never leave the real
# path half-written. Returns 0 if the previous state is back.
restore_pins() {
    if [ -z "$BACKUP" ]; then
        /bin/rm -f "$RESOLVED"
        return $?
    fi
    [ -f "$BACKUP" ] || return 1
    local _staged="$RESOLVED.restoring.$$"
    /bin/cp -f "$BACKUP" "$_staged" || { /bin/rm -f "$_staged"; return 1; }
    /bin/mv -f "$_staged" "$RESOLVED"
}

# How to get the pins back when the restore itself failed, into RECOVERY_HINT. The backup is the
# right source, not git: the pins being replaced may be uncommitted (a reviewed-but-not-yet-
# committed upgrade), and `git checkout` would silently trade them for older ones. So the temp
# dir holding the backup is kept - cleanup() skips it once WORK_DIR is cleared - and named.
# Sets globals rather than printing, because a $(...) caller would clear WORK_DIR in a subshell.
RECOVERY_HINT=""
set_recovery_hint() {
    if [ -n "$BACKUP" ] && [ -f "$BACKUP" ]; then
        WORK_DIR=""
        RECOVERY_HINT="copy it back from the backup this run kept: /bin/cp '$BACKUP' '$RESOLVED'"
    else
        RECOVERY_HINT="recover the committed one with: git -C '$REPO_ROOT' checkout -- mlx-agent.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved"
    fi
}

# Restore the pins, then fail with "$1" plus what actually happened to them. A full disk is a
# likely cause of the very failure being reported, and then the restore's copy can fail too:
# saying "restored" there would leave the repo without a Package.resolved and nobody told.
fail_restoring() {
    restore_pins
    local _restored=$?
    if [ "$_restored" -eq 0 ]; then
        fail "$1 (Package.resolved is back as it was before this run)"
    fi
    set_recovery_hint
    fail "$1 - and could NOT restore $RESOLVED; $RECOVERY_HINT"
}

on_signal() {
    echo >&2
    if [ "$PINS_MOVED" = "yes" ]; then
        restore_pins
        local _restored=$?
        if [ "$_restored" -eq 0 ]; then
            echo "${YELLOW}Interrupted - restored mlx-agent's Package.resolved (this run's resolution discarded)${RESET}" >&2
        else
            set_recovery_hint
            echo "${RED}Interrupted - could NOT restore $RESOLVED; $RECOVERY_HINT${RESET}" >&2
        fi
    fi
    cleanup
    exit "$1"
}

trap 'on_signal 130' INT
trap 'on_signal 143' TERM
trap 'on_signal 129' HUP
trap 'on_signal 131' QUIT

pin_versions() { /usr/bin/awk -f "$PIN_AWK" "$1"; }

# Prints a warning for every pin whose version went DOWN. With the checkouts cleared that is
# never the resolver's whim: a range was narrowed, or a tag was yanked upstream. Either way a
# human should know before it ships. Revision-pinned entries are not ordered, so skipped.
report_downgrades() {
    local _new_pins="$2"
    local _id
    local _old
    printf '%s\n' "$1" | while read -r _id _old; do
        local _new="$(printf '%s\n' "$_new_pins" | /usr/bin/awk -v id="$_id" '$1 == id { print $2 }')"
        [ -n "$_new" ] && [ "$_new" != "$_old" ] || continue
        case "$_old $_new" in *rev*) continue ;; esac
        local _lower="$(printf '%s\n%s\n' "$_old" "$_new" | /usr/bin/sort -V | /usr/bin/head -1)"
        [ "$_lower" = "$_new" ] && echo "  ${YELLOW}WARNING: $_id went DOWN, $_old -> $_new (a narrowed range or a yanked tag?)${RESET}"
    done
}

# For each package project.yml names directly, compare the resolved version with the newest
# release tag upstream. A gap means a version range is holding it back - project.yml's own, or
# one in another package's manifest (mlx-swift is capped by mlx-swift-lm's, for instance).
report_held_back() {
    local _urls="$(/usr/bin/sed -n 's/^[[:space:]]*url:[[:space:]]*//p' "$REPO_ROOT/project.yml" | /usr/bin/tr -d "\"'")"
    local _by_location="$(/usr/bin/awk -v by_location=1 -f "$PIN_AWK" "$RESOLVED")"
    local _held="no"
    local _url
    local _tags
    echo "  Newest upstream releases of the direct dependencies:"
    for _url in $_urls; do
        local _name="$(/usr/bin/basename "$_url" .git)"
        local _key="$(printf '%s\n' "$_url" | /usr/bin/awk '{ l = tolower($0); sub(/\/+$/, "", l); sub(/\.git$/, "", l); print l }')"
        local _resolved="$(printf '%s\n' "$_by_location" | /usr/bin/awk -v key="$_key" '$1 == key { print $2 }')"
        _tags="$(GIT_TERMINAL_PROMPT=0 /usr/bin/git ls-remote --tags --refs "$_url" 2>/dev/null)"
        local _git_status=$?
        if [ "$_git_status" -ne 0 ]; then
            printf '    %-24s %-10s %s\n' "$_name" "${_resolved:-?}" "(could not reach $_url)"
            continue
        fi
        local _latest="$(printf '%s\n' "$_tags" | /usr/bin/sed -n 's#.*refs/tags/v\{0,1\}\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)$#\1#p' | /usr/bin/sort -V | /usr/bin/tail -1)"
        if [ -z "$_resolved" ]; then
            printf '    %-24s %-10s %s\n' "$_name" "?" "(not found in Package.resolved${_latest:+; newest tag is $_latest})"
            continue
        fi
        if [ -z "$_latest" ] || [ "$_resolved" = "$_latest" ]; then
            printf '    %-24s %-10s %s\n' "$_name" "$_resolved" "${_latest:+(newest)}"
            continue
        fi
        local _lower="$(printf '%s\n%s\n' "$_resolved" "$_latest" | /usr/bin/sort -V | /usr/bin/head -1)"
        if [ "$_lower" = "$_resolved" ]; then
            printf '    %-24s %-10s %s\n' "$_name" "$_resolved" "${YELLOW}HELD BACK: $_latest is out${RESET}"
            _held="yes"
        else
            printf '    %-24s %-10s %s\n' "$_name" "$_resolved" "(newest tag is $_latest)"
        fi
    done
    if [ "$_held" = "yes" ]; then
        echo "  ${YELLOW}A version range blocks a newer release. Raise it in $REPO_ROOT/project.yml (then xcodegen generate) if it is that file's range; if it is another package's manifest, it has to wait for that package.${RESET}"
    fi
}

[ -d "$PROJECT" ] || fail "No $PROJECT - run this from an mlx-agent checkout"
[ -f "$PIN_AWK" ] || fail "No $PIN_AWK - it ships beside this script"

echo "  Re-resolving mlx-agent's SPM dependencies to the newest versions project.yml allows..."

WORK_DIR="$(/usr/bin/mktemp -d "${TMPDIR:-/tmp}/mlx-agent-resolve.XXXXXX")"
[ -n "$WORK_DIR" ] && [ -d "$WORK_DIR" ] || fail "mktemp failed"
RESOLVE_LOG="$WORK_DIR/resolve.log"

OLD_PINS=""
if [ -f "$RESOLVED" ]; then
    BACKUP="$WORK_DIR/Package.resolved"
    /bin/cp -f "$RESOLVED" "$BACKUP" || fail "Could not back up $RESOLVED"
    OLD_PINS="$(pin_versions "$BACKUP")"
fi

# From here until the new pins are checked, the signal handler restores them.
PINS_MOVED="yes"
# Checked, not assumed: a Package.resolved that survives is honored by the resolve, which then
# reports "unchanged" for pins it never re-resolved.
/bin/rm -f "$RESOLVED" && /bin/rm -rf "$SOURCE_PACKAGES/checkouts" "$SOURCE_PACKAGES/workspace-state.json"
rm_status=$?
if [ "$rm_status" -ne 0 ]; then
    fail_restoring "Could not remove $RESOLVED or the stale package checkouts in $SOURCE_PACKAGES"
fi

/usr/bin/xcodebuild \
    -project "$PROJECT" \
    -scheme mlx-agent \
    -derivedDataPath "$REPO_ROOT/build" \
    -skipPackagePluginValidation \
    -skipMacroValidation \
    -resolvePackageDependencies >"$RESOLVE_LOG" 2>&1
resolve_status=$?
if [ "$resolve_status" -ne 0 ]; then
    # A version conflict is the informative failure here, and the one where project.yml needs a
    # human, so show what xcodebuild said instead of only that it failed.
    /usr/bin/tail -20 "$RESOLVE_LOG" >&2
    fail_restoring "xcodebuild -resolvePackageDependencies failed (exit $resolve_status)"
fi

if [ ! -f "$RESOLVED" ]; then
    fail_restoring "Resolution wrote no Package.resolved"
fi

# Existence is not enough: a resolution that died mid-write leaves a file that parses to
# nothing, and an empty pin table would compare unequal to the old one and be announced as an
# ordinary "VERSIONS changed" with no versions under it. Checked while the backup is still valid.
NEW_PINS="$(pin_versions "$RESOLVED")"
if [ -z "$NEW_PINS" ]; then
    fail_restoring "Resolution produced an unreadable Package.resolved - no pins in it"
fi
PINS_MOVED="no"

printf '%s\n' "$NEW_PINS"

if [ -z "$BACKUP" ]; then
    echo "  ${YELLOW}Resolved from scratch (there was no Package.resolved) - review and commit it in $REPO_ROOT${RESET}"
elif [ "$NEW_PINS" != "$OLD_PINS" ]; then
    echo "  ${YELLOW}Package VERSIONS changed - review and commit Package.resolved in $REPO_ROOT${RESET}"
    report_downgrades "$OLD_PINS" "$NEW_PINS"
else
    /usr/bin/cmp -s "$BACKUP" "$RESOLVED"
    cmp_status=$?
    if [ "$cmp_status" -eq 0 ]; then
        echo "  Package.resolved unchanged (already at the newest allowed versions)"
    else
        # Every pin landed on the same version, but SwiftPM rewrote originHash (it hashes the
        # dependency declarations). Said out loud because the resulting git diff is otherwise
        # indistinguishable from a real upgrade.
        echo "  Package versions unchanged (already newest allowed); only Package.resolved metadata was rewritten"
    fi
fi

if [ "$DO_UPSTREAM_CHECK" = "yes" ]; then
    report_held_back
fi

cleanup
exit 0
