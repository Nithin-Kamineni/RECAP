#!/usr/bin/env bash
# =============================================================================
#  restructure/gate.sh -- ProjectRestructure.md section 9.1, as one command
# =============================================================================
#
#      module load apptainer
#      bash hpc/tl.sh bash restructure/gate.sh            # check
#      bash hpc/tl.sh bash restructure/gate.sh --update   # re-take the golden
#
#  Takes a fresh snapshot and diffs it against restructure/golden/. Exit 0 means
#  the phase moved code and nothing else:
#
#      1. arch_fingerprint() for every (arch, model) -- byte-identical
#      2. run.sh baseline --eval  -- Task 1 and 2 totals unchanged, to the pJ
#      3. run.sh embedded --eval  -- unchanged
#      4. run.sh recon    --eval  -- every bar unchanged
#      5. run.sh validate && diagnose -- clean, and diff vs before
#      6. the test suite -- no worse than it was
#
#  IF THE FINGERPRINT BLOCK FAILS, STOP. Every hash names a MAPPER CACHE
#  DIRECTORY and a moved hash is hours of SLURM gone. The gate prints that block
#  first and on its own for exactly that reason.
#
#  --update rewrites the golden snapshot. Use it ONLY to establish a new
#  baseline at the START of a phase, never to make a failing gate pass: the
#  whole point of the file is that it predates the change being judged.
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
GOLDEN="restructure/golden"
FRESH="${ECC_GATE_OUT:-restructure/.fresh}"

if [ "${1:-}" = "--update" ]; then
    python3 restructure/snapshot.py "$GOLDEN" || exit 1
    echo
    echo "golden snapshot rewritten: $GOLDEN   -- commit it before changing any code"
    exit 0
fi

if [ ! -d "$GOLDEN" ]; then
    echo "gate: no golden snapshot at $GOLDEN" >&2
    echo "  -> take one first:  bash hpc/tl.sh bash restructure/gate.sh --update" >&2
    exit 2
fi

python3 restructure/snapshot.py "$FRESH" || exit 1
echo
echo "======================================================================"
echo " GATE 1  arch_fingerprint() -- a moved hash is a COLD MAPPER CACHE"
echo "======================================================================"
if diff -u "$GOLDEN/fingerprints.tsv" "$FRESH/fingerprints.tsv"; then
    echo "  identical: $(grep -vc '^#' "$GOLDEN/fingerprints.tsv") rows"
    fp_rc=0
else
    echo
    echo "  STOP. Do not run the mapper. The architecture the mapper sees has moved."
    fp_rc=1
fi

echo
echo "======================================================================"
echo " GATES 2-6  evaluated totals, validate, diagnose, tests"
echo "======================================================================"
rest_rc=0
if diff -ur -x fingerprints.tsv "$GOLDEN" "$FRESH"; then
    echo "  identical: every evaluated total, every guard report, every test outcome"
else
    rest_rc=1
fi

echo
if [ "$fp_rc" = 0 ] && [ "$rest_rc" = 0 ]; then
    echo "GATE PASSED -- the phase moved code and nothing else."
    exit 0
fi
[ "$fp_rc" = 0 ] || echo "GATE FAILED on item 1 (fingerprints) -- STOP."
[ "$rest_rc" = 0 ] || echo "GATE FAILED on items 2-6 (totals / reports / tests)."
echo "  golden: $GOLDEN     fresh: $FRESH"
exit 1
