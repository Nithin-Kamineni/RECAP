#!/usr/bin/env bash
# Normalise every project text file to LF line endings.
#
# WHY THIS EXISTS. Editing a .sh on Windows can save it with CRLF, which bash
# inside the container cannot parse. `set -euo pipefail\r` fails with a mangled,
# self-overwriting message:
#
#     : invalid option namene 24: set: pipefail
#
# If any script dies like that, run this and try again:
#
#     bash tools/fix-eol.sh
#
# Only carriage returns are removed (tr -d '\r'); nothing else in the file is
# touched, so it is safe to run on YAML, JSON and cached results.
set -eu
# tools/ is one level down: normalise the PROJECT, not this directory.
cd "$(dirname "$0")/.."

n=0
while IFS= read -r f; do
    # -U: treat as binary so grep sees the CR instead of stripping it
    if LC_ALL=C grep -qU "$(printf '\r')" "$f"; then
        tr -d '\r' < "$f" > "$f.eoltmp" && mv "$f.eoltmp" "$f"
        echo "  fixed $f"
        n=$((n + 1))
    fi
done < <(find . -type f \( -name '*.sh' -o -name '*.py' -o -name '*.yaml' \
             -o -name '*.yml' -o -name '*.md' -o -name '*.json' \) \
             -not -path './ecc_energy_study/timeloop-accelergy-exercises/*' \
             -not -path './ecc_energy_study/outputs/*' \
             -not -path './legacy/*' \
             -not -name '*.eoltmp')

if [ "$n" -eq 0 ]; then
    echo "all files already LF"
else
    echo "$n file(s) normalised to LF"
fi

# Executability is the other thing Windows loses.
chmod +x run.sh tools/fix-eol.sh 2>/dev/null || true
