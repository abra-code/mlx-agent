# package_pin_versions.awk - one line per pin of a SwiftPM Package.resolved (format version 3).
#
#   /usr/bin/awk -f package_pin_versions.awk Package.resolved
#       "    <identity>                 <version>"   - for display and for comparing two files
#   /usr/bin/awk -v by_location=1 -f package_pin_versions.awk Package.resolved
#       "<location> <version>"                       - for looking a pin up by its URL; the
#                                                      location is normalized (lowercase, no
#                                                      trailing "/" or ".git"), since project.yml
#                                                      and a manifest may spell one URL two ways
#
# Anchored on `"version" : "` so the trailing `"version" : 3` format marker at the end of the
# file (an int, not a string) is not mistaken for a package version. A pin with no version
# (branch- or revision-pinned) falls back to its revision, so a revision move is NOT reported as
# "versions unchanged": the output answers "did any pin actually move?", which is not the same
# question as "did the file change" - a re-resolution rewrites originHash even when every pin
# lands on the byte-identical version. The revision is printed in full rather than abbreviated
# because this output is COMPARED, and two revisions sharing a short prefix would compare equal.

BEGIN { FS = "\"" }

function emit() {
    if (id == "") return
    v = (ver != "" ? ver : "rev " rev)
    l = tolower(loc); sub(/\/+$/, "", l); sub(/\.git$/, "", l)
    if (by_location) printf "%s %s\n", l, v
    else printf "    %-24s %s\n", id, v
}

/"identity"/                            { emit(); id = $4; ver = ""; rev = ""; loc = "" }
/"location"[[:space:]]*:[[:space:]]*"/  { loc = $4 }
/"revision"[[:space:]]*:[[:space:]]*"/  { rev = $4 }
/"version"[[:space:]]*:[[:space:]]*"/   { ver = $4 }
END                                     { emit() }
