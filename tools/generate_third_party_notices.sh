#!/bin/sh
# generate_third_party_notices.sh - collect the license texts of everything that is
# statically linked into, or shipped beside, the mlx-agent binary.
#
# mlx-agent is Apache 2.0, but the executable is a static link of ~18 Swift packages under
# MIT / Apache 2.0 / BSD terms, several of which themselves vendor third-party C and C++
# (mlx-swift's Cmlx target carries Apple's MLX, mlx-c, fmt, nlohmann/json and metal-cpp;
# swift-nio carries llhttp; swift-crypto compiles in a vendored BoringSSL). Three packages
# also ship resource bundles that travel with the binary - mlx-swift_Cmlx.bundle holds the
# compiled Metal shaders. Every one of those licenses requires its notice to accompany the
# binary form, so an app embedding mlx-agent must deploy this file next to the binary.
#
# Reads the resolved package graph and the SPM checkouts that the documented xcodebuild
# -derivedDataPath build leaves behind, so the output tracks the actual pins rather than a
# hand-maintained list. Anything that would make the output incomplete - a package with no
# license text, a checkout with no pin, a missing supplemental directory - is a hard error.
# A notices file that is quietly short is worse than no notices file at all, because it
# looks complete.
#
# Usage: tools/generate_third_party_notices.sh [--checkouts DIR] [--resolved FILE]
#                                              [--supplemental DIR] [--output FILE]
#                                              [--allow-unpinned-checkout]
#
# pipefail is deliberately not set: it is not POSIX, and on a shell where `set` rejects it
# the whole option string fails and -u would be lost with it.
set -u

REPO_ROOT="$(cd "$(/usr/bin/dirname "$0")/.." && pwd)"
CHECKOUTS="$REPO_ROOT/build/SourcePackages/checkouts"
RESOLVED="$REPO_ROOT/mlx-agent.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved"
SUPPLEMENTAL="$REPO_ROOT/tools/notices-supplemental"
OUTPUT=""
ALLOW_UNPINNED="no"

fail() { echo "generate_third_party_notices: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --checkouts=*)    CHECKOUTS="${1#*=}" ;;
        --checkouts)      shift; CHECKOUTS="${1:-}" ;;
        --resolved=*)     RESOLVED="${1#*=}" ;;
        --resolved)       shift; RESOLVED="${1:-}" ;;
        --supplemental=*) SUPPLEMENTAL="${1#*=}" ;;
        --supplemental)   shift; SUPPLEMENTAL="${1:-}" ;;
        --output=*)       OUTPUT="${1#*=}" ;;
        --output|-o)      shift; OUTPUT="${1:-}" ;;
        --allow-unpinned-checkout) ALLOW_UNPINNED="yes" ;;
        -h|--help)
            echo "Usage: $0 [--checkouts DIR] [--resolved FILE] [--supplemental DIR]"
            echo "          [--output FILE] [--allow-unpinned-checkout]"
            exit 0 ;;
        *) fail "Unknown argument: $1" ;;
    esac
    shift
done

[ -f "$RESOLVED" ] || fail "No Package.resolved at $RESOLVED"
[ -d "$CHECKOUTS" ] || fail "No SPM checkouts at $CHECKOUTS - run the documented xcodebuild (it populates build/SourcePackages) first"
# The supplemental directory is the only carrier for notices that no checkout provides
# (BoringSSL today). Skipping it because a path was mistyped would silently drop a required
# notice, so its absence is fatal rather than a no-op.
[ -d "$SUPPLEMENTAL" ] || fail "No supplemental notices directory at $SUPPLEMENTAL - it carries the licenses no checkout ships (BoringSSL)"

_pins="$(/usr/bin/mktemp "${TMPDIR:-/tmp}/notices-pins.XXXXXX")" || fail "mktemp failed"
_liclist="$(/usr/bin/mktemp "${TMPDIR:-/tmp}/notices-lics.XXXXXX")" || fail "mktemp failed"
_stage=""
trap '/bin/rm -f "$_pins" "$_liclist" ${_stage:+"$_stage"}' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
trap 'exit 131' QUIT

# --- parse Package.resolved -------------------------------------------------------------
# SPM writes one key per line, so a line-oriented parser is enough. Every pattern requires a
# QUOTED value: Package.resolved v3 ends with a bare `"version" : 3` after the pins array,
# which an unguarded match would graft onto the last pin as its version.
/usr/bin/awk '
function qval(line,   n, s) {
    n = index(line, ":"); s = substr(line, n + 1)
    sub(/^[[:space:]]*"/, "", s); sub(/"[[:space:]]*,?[[:space:]]*$/, "", s)
    return s
}
function flush(   v) {
    if (id == "") return
    v = (ver != "") ? ver : ((rev != "") ? "rev " substr(rev, 1, 12) : "-")
    printf "%s\t%s\t%s\n", id, (loc != "" ? loc : "-"), v
}
/"identity"[[:space:]]*:[[:space:]]*"/ { flush(); id = qval($0); loc = ""; ver = ""; rev = ""; next }
/"location"[[:space:]]*:[[:space:]]*"/ { loc = qval($0); next }
/"revision"[[:space:]]*:[[:space:]]*"/ { rev = qval($0); next }
/"version"[[:space:]]*:[[:space:]]*"/  { ver = qval($0); next }
END { flush() }
' "$RESOLVED" > "$_pins" || fail "Could not parse $RESOLVED"

# This compares the file against itself - awk emits one row per identity line and grep counts
# identity lines - so it detects a renamed or unquoted key, NOT a short or truncated file.
# The real backstop against under-reporting is the unpinned-checkout check below, which
# compares the pin list against what SPM actually put on disk.
_want=$(/usr/bin/grep -c '"identity"[[:space:]]*:' "$RESOLVED" | /usr/bin/tr -d " ")
_got=$(/usr/bin/wc -l < "$_pins" | /usr/bin/tr -d " ")
[ "${_got:-0}" -gt 0 ] || fail "Parsed no packages out of $RESOLVED - the format changed"
[ "$_got" = "$_want" ] || fail "Parsed $_got packages but $RESOLVED declares $_want - the format changed"

# Every checkout on disk must correspond to a pin. An unmatched one is either a stale leftover
# from a dropped dependency or - the case that matters - a package this file would not cover.
# Fatal by default: for a compliance artifact the safe direction is to stop, not to warn in
# the middle of a wall of build output. --allow-unpinned-checkout is the escape hatch for a
# build directory that is merely stale.
_unpinned=""
for _c in "$CHECKOUTS"/*; do
    [ -d "$_c" ] || continue
    _cb=$(/usr/bin/basename "$_c")
    /usr/bin/awk -F'\t' -v n="$_cb" 'tolower($1)==tolower(n) { found=1 } END { exit !found }' "$_pins" \
        || _unpinned="$_unpinned $_cb"
done
if [ -n "$_unpinned" ]; then
    if [ "$ALLOW_UNPINNED" = "yes" ]; then
        echo "generate_third_party_notices: WARNING: checkouts with no pin (not covered):$_unpinned" >&2
    else
        fail "Checkouts with no pin in $RESOLVED:$_unpinned
  Either they are stale (clean $CHECKOUTS, or re-resolve) or they are linked and this file
  would not cover them. Re-run with --allow-unpinned-checkout once you know which."
    fi
fi

# --- per-package license discovery -------------------------------------------------------
# The checkout directory is named after the repository and the pin after the identity, and
# the two differ in case (identity "eventsource" -> checkout "EventSource").
checkout_dir() {   # $1 = identity
    [ -d "$CHECKOUTS/$1" ] && { echo "$CHECKOUTS/$1"; return 0; }
    /usr/bin/find "$CHECKOUTS" -maxdepth 1 -type d -iname "$1" -print | /usr/bin/head -1
}

# Nested, not just the checkout root: several packages vendor third-party C/C++ that is
# compiled straight into the binary and carries its own license beside the sources -
# mlx-swift/Source/Cmlx holds Apple's MLX, mlx-c, fmt, nlohmann/json and metal-cpp, and
# swift-nio/Sources/CNIOLLHTTP holds llhttp. Taking only the root LICENSE looked complete
# while omitting all six. Tests are excluded because their fixtures are not redistributed.
emit_package_licenses() {   # $1 = checkout dir; nonzero if it carries no license text at all
    /usr/bin/find "$1" -maxdepth 5 -type f \
        \( -iname "LICENSE*" -o -iname "NOTICE*" -o -iname "COPYING*" \) \
        ! -path "*/Tests/*" ! -path "*/.git/*" -print 2>/dev/null \
        | LC_ALL=C /usr/bin/sort > "$_liclist" || return 1
    _any=0
    while IFS= read -r _f; do
        [ -n "$_f" ] || continue
        [ -f "$_f" ] || continue
        echo "--- ${_f#"$1"/} ---"
        echo
        /bin/cat "$_f" || fail "Could not read $_f"
        echo
        _any=1
    done < "$_liclist"
    [ "$_any" = 1 ]
}

# --- emit --------------------------------------------------------------------------------
emit() {
    cat <<'HEADER'
mlx-agent - Third-Party Software Notices
========================================

mlx-agent itself is licensed under the Apache License, Version 2.0; see the LICENSE file
distributed beside this one.

The mlx-agent executable statically links the Swift packages listed below. Three of them
also ship resource bundles that are redistributed beside the binary:

    mlx-swift_Cmlx.bundle          - mlx-swift (compiled Metal shaders, default.metallib)
    swift-crypto_Crypto.bundle     - swift-crypto
    swift-transformers_Hub.bundle  - swift-transformers

Every license below is reproduced in full, as those licenses require when the covered work
is redistributed in binary form. Where a package vendors third-party code of its own, that
code's license is reproduced too and is labeled with its path inside the package.

The list is generated from the resolved package graph, so it covers the whole pinned
dependency set rather than only the direct dependencies. A few listed packages contribute
only build-time tooling (for example swift-syntax, which backs a compiler macro plugin) and
are not linked into the shipped binary; they are kept because over-reporting a notice is
harmless while omitting one is not.

HEADER

    _n=0
    _tab=$(/usr/bin/printf '\t')
    while IFS="$_tab" read -r _id _loc _ver; do
        [ -n "${_id:-}" ] || continue
        _dir=$(checkout_dir "$_id")
        [ -n "$_dir" ] && [ -d "$_dir" ] || fail "No checkout for '$_id' under $CHECKOUTS - build first so SPM resolves it"
        _n=$((_n + 1))
        echo "--------------------------------------------------------------------------------"
        echo "$_n. $_id ${_ver:--}"
        echo "   ${_loc:--}"
        echo "--------------------------------------------------------------------------------"
        echo
        emit_package_licenses "$_dir" \
            || fail "No license text anywhere under $_dir - '$_id' would ship with no notice. Vendor its license under $SUPPLEMENTAL if upstream ships none."
        echo
    done < "$_pins"

    # Vendored code that its own package ships without any license file. BoringSSL inside
    # swift-crypto is the live case: per-file headers only, and swift-crypto's NOTICE.txt
    # does not mention it. Each supplemental file explains what it covers and why.
    _sup=0
    for _s in "$SUPPLEMENTAL"/*.txt; do
        [ -f "$_s" ] || continue
        echo "--------------------------------------------------------------------------------"
        echo "Vendored component: $(/usr/bin/basename "$_s" .txt)"
        echo "--------------------------------------------------------------------------------"
        echo
        /bin/cat "$_s" || fail "Could not read $_s"
        echo
        _sup=$((_sup + 1))
    done
    [ "$_sup" -gt 0 ] || fail "No *.txt in $SUPPLEMENTAL - it must carry at least the BoringSSL notice"
}

if [ -n "$OUTPUT" ]; then
    # Staged inside the DESTINATION directory so the install is a same-volume rename. Writing
    # straight to $OUTPUT would truncate it up front, and a full disk or a signal mid-copy
    # would leave a short but non-empty notices file - which passes every "is it there and
    # non-empty" check downstream and ships looking complete.
    _outdir=$(/usr/bin/dirname "$OUTPUT")
    [ -d "$_outdir" ] || fail "No directory $_outdir for --output"
    _stage=$(/usr/bin/mktemp "$_outdir/.notices.XXXXXX") || fail "Could not stage in $_outdir"
    emit > "$_stage" || fail "Could not assemble the notices"
    /bin/chmod 644 "$_stage" || fail "Could not set permissions on the staged notices"
    /bin/mv -f "$_stage" "$OUTPUT" || fail "Could not install $OUTPUT"
    _stage=""
    echo "Wrote $OUTPUT ($(/usr/bin/wc -l < "$OUTPUT" | /usr/bin/tr -d " ") lines, $_got packages)"
else
    emit
fi
