#!/usr/bin/env bash
# =============================================================================
#  restructure/gate.sh -- ProjectRestructure.md section 9.1, as one command
# =============================================================================
#
#      module load apptainer
#      bash hpc/tl.sh bash restructure/gate.sh            # check
#      bash hpc/tl.sh bash restructure/gate.sh --update   # re-take the golden
#      bash hpc/tl.sh bash restructure/gate.sh --accept-tests "<why>"
#
#  Takes a fresh snapshot and diffs it against restructure/golden/. Exit 0 means
#  the phase moved code and nothing else:
#
#      1. arch_fingerprint() for every (arch, model) -- byte-identical
#      2. run.sh baseline --eval  -- Task 1 and 2 totals unchanged, to the pJ
#      3. run.sh embedded --eval  -- unchanged
#      4. run.sh recon    --eval  -- every bar unchanged
#      5. run.sh validate && diagnose -- clean, and diff vs before
#      6. the test suite -- NO WORSE than it was
#
#  IF THE FINGERPRINT BLOCK FAILS, STOP. Every hash names a MAPPER CACHE
#  DIRECTORY and a moved hash is hours of SLURM gone. The gate prints that block
#  first and on its own for exactly that reason.
#
#  ITEM 6 IS THE ONE THAT IS NOT BYTE EQUALITY, because section 9.1 does not ask
#  for it: phase 1 fixes tests, phase 7 adds them, and a suite that grew is not
#  a suite that broke. So item 6 refuses a FAILURE, refuses a silent DROP in the
#  number passing, and accepts growth. A deliberate drop -- a test deleted on
#  purpose -- is a recorded override: `--accept-tests "<why>"` rewrites the test
#  artefact ALONE, leaving every energy in the golden snapshot untouched, and
#  appends the reason to restructure/golden/pytest.log.
#
#  --update rewrites the WHOLE golden snapshot. Use it ONLY to establish a new
#  baseline at the START of a phase, never to make a failing gate pass: the
#  whole point of the file is that it predates the change being judged.
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
GOLDEN="restructure/golden"
FRESH="${ECC_GATE_OUT:-restructure/.fresh}"

ACCEPT_TESTS=""
case "${1:-}" in
    --update)
        python3 restructure/snapshot.py "$GOLDEN" || exit 1
        echo
        echo "golden snapshot rewritten: $GOLDEN   -- commit it before changing any code"
        exit 0 ;;
    --accept-tests)
        ACCEPT_TESTS="${2:-}"
        if [ -z "$ACCEPT_TESTS" ]; then
            echo "gate: --accept-tests needs a reason, e.g." >&2
            echo "  bash restructure/gate.sh --accept-tests 'phase 1: +7 plumbing tests'" >&2
            exit 2
        fi ;;
esac

if [ ! -d "$GOLDEN" ]; then
    echo "gate: no golden snapshot at $GOLDEN" >&2
    echo "  -> take one first:  bash hpc/tl.sh bash restructure/gate.sh --update" >&2
    exit 2
fi

python3 restructure/snapshot.py "$FRESH" || exit 1

# The pytest summary line: "227 passed, 4 skipped in <T>s".
_summary() { tail -n 20 "$1" | grep -E '[0-9]+ (passed|failed|error)' | tail -n 1; }
_count()   { grep -oE "[0-9]+ $2" <<<"$1" | head -n 1 | cut -d' ' -f1; }

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
echo " GATES 2-5  evaluated totals, validate, diagnose"
echo "======================================================================"
eval_rc=0
if diff -ur -x fingerprints.tsv -x 'pytest.*' "$GOLDEN" "$FRESH"; then
    echo "  identical: every evaluated total and every guard report"
else
    eval_rc=1
fi

echo
echo "======================================================================"
echo " GATE 6  the test suite -- no worse than it was"
echo "======================================================================"
was="$(_summary "$GOLDEN/pytest.txt")"
now="$(_summary "$FRESH/pytest.txt")"
echo "  was : ${was:-(none)}"
echo "  now : ${now:-(none)}"
test_rc=0
if grep -qE '[0-9]+ (failed|error)' <<<"$now"; then
    echo "  RED. A failing suite is worse than it was, whatever else moved."
    test_rc=1
else
    was_p="$(_count "$was" passed)"; now_p="$(_count "$now" passed)"
    if [ "${now_p:-0}" -lt "${was_p:-0}" ]; then
        if [ -n "$ACCEPT_TESTS" ]; then
            echo "  ${was_p} -> ${now_p} passing: DECLARED -- $ACCEPT_TESTS"
        else
            echo "  ${was_p} -> ${now_p} passing: tests were LOST, and nothing says why."
            echo "  -> if that is deliberate, record it:"
            echo "     bash restructure/gate.sh --accept-tests '<why>'"
            test_rc=1
        fi
    elif [ "$was" != "$now" ]; then
        echo "  no failures, and nothing lost."
    else
        echo "  identical."
    fi
fi

echo
if [ "$fp_rc" = 0 ] && [ "$eval_rc" = 0 ] && [ "$test_rc" = 0 ]; then
    if [ "$was" != "$now" ]; then
        if [ -n "$ACCEPT_TESTS" ]; then
            cp "$FRESH/pytest.txt" "$GOLDEN/pytest.txt"
            printf '%s\t%s -> %s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                   "$was" "$now" "$ACCEPT_TESTS" >> "$GOLDEN/pytest.log"
            echo "golden/pytest.txt updated; the reason is in golden/pytest.log."
            echo "Every energy in the golden snapshot is untouched."
        else
            echo "NOTE: the test artefact moved and the golden still holds the old one."
            echo "      Record it when the phase is done:"
            echo "      bash restructure/gate.sh --accept-tests '<why>'"
        fi
    fi
    echo "GATE PASSED -- the phase moved code and nothing else."
    exit 0
fi
[ "$fp_rc"   = 0 ] || echo "GATE FAILED on item 1 (fingerprints) -- STOP."
[ "$eval_rc" = 0 ] || echo "GATE FAILED on items 2-5 (totals / reports)."
[ "$test_rc" = 0 ] || echo "GATE FAILED on item 6 (tests)."
echo "  golden: $GOLDEN     fresh: $FRESH"
exit 1
