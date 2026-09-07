#!/usr/bin/env bash
# =============================================================================
#  hpc/sweep.sh -- plot ONE sweep axis, everything else held fixed
#
#      bash hpc/sweep.sh arch          -> results/figures/ArchitectureSweep.png
#      bash hpc/sweep.sh model         -> results/figures/ModelSweep.png
#      bash hpc/sweep.sh bch           -> results/figures/BCHsweep.png
#      bash hpc/sweep.sh panels        -> ArchitectureSweep.png (one panel per model)
#
#  Exactly one axis varies per run; the other two are held at a constant, which
#  is what makes each figure a single claim. All three ECC arms (baseline,
#  embedded, recon) are always drawn -- they are arms, never an axis -- so
#  `embedded` and `recon` appear as placeholders until Tasks 2-5 implement them.
#
#  FILE NAMES ARE FIXED AND OVERWRITE. Each axis has exactly one output stem,
#  so re-running at different constants rewrites the same three files rather
#  than accumulating a directory of near-identical names. The manifest written
#  beside each figure records which constants produced the file currently on
#  disk. That only holds for WHOLE-MODEL runs: setting ECC_LAYERS appends the
#  layer scope to the stem on purpose, so a development figure can never be
#  mistaken for a full-model one. This script therefore forces ECC_LAYERS="".
#
#  Everything here is `--eval` (ECC_FROM_CACHE=1): mappings are read from the
#  mapper cache and Timeloop is never invoked, so it is safe on a login node.
#  Build the cache first with the job array:
#      sbatch hpc/map.sbatch
#
#  Override any constant from the shell, e.g.
#      ECC_CONST_MODEL=mobilenet_v2 bash hpc/sweep.sh arch
#      ECC_SWEEP_KS="57 51 45 39" bash hpc/sweep.sh bch
#      ECC_PANEL_MODELS="resnet18 mobilenet_v2" bash hpc/sweep.sh panels
# =============================================================================
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AXIS="${1:-}"
case "$AXIS" in
    arch|model|bch|panels) ;;
    *)  echo "usage: bash hpc/sweep.sh <arch|model|bch|panels>" >&2
        echo "  arch    sweep the accelerators, hold model + BCH K" >&2
        echo "  model   sweep the networks,     hold arch  + BCH K" >&2
        echo "  bch     sweep BCH(63,K),        hold arch  + model" >&2
        echo "  panels  arch sweep, one panel per ECC_PANEL_MODELS entry" >&2
        exit 2 ;;
esac

# ---- the converged search: identical to what the mapper cache was built at --
# These are hashed into the mapping fingerprint, so changing one is a different
# cache and a cache MISS. Keep them in step with hpc/map.sbatch.
export ECC_MAPPER_ALGORITHM="${ECC_MAPPER_ALGORITHM:-random_pruned}"
export ECC_MAPPER_SEARCH_SIZE="${ECC_MAPPER_SEARCH_SIZE-}"   # empty = uncapped, as in map.sbatch
export ECC_VICTORY="${ECC_VICTORY:-2000}"
export ECC_MAPPER_TIMEOUT="${ECC_MAPPER_TIMEOUT:-2000}"
export ECC_MAPPER_THREADS="${ECC_MAPPER_THREADS:-18}"

# All three arms, always. This is the project's rule, not a default to tune.
export ECC_APPROACHES="${ECC_APPROACHES:-baseline embedded recon}"

# Whole model -> the stem stays the fixed one. Do not set this to a layer list
# unless you WANT a separate development figure.
export ECC_LAYERS=""

ARCHS_ALL="eyeriss_v2_like eyeriss_like simple_weight_stationary simple_output_stationary simple_input_stationary simba_like"

case "$AXIS" in
    arch)
        export ECC_SWEEP=arch
        export ECC_SWEEP_ARCHS="${ECC_SWEEP_ARCHS:-$ARCHS_ALL}"
        export ECC_CONST_K="${ECC_CONST_K:-51}"
        MODELS="${ECC_CONST_MODEL:-resnet18}"
        # ONE model -> a single-panel architecture sweep. SEVERAL -> the panelled
        # layout, one panel per model, which is the only honest way to show more
        # than one network at once: `run.sh` holds the model fixed for a reason,
        # and stacking two models into one axis would vary two things per bar.
        # ECC_CONST_MODEL still takes one value, so the list is handed to the
        # panels experiment instead.
        if [ "$(echo "$MODELS" | wc -w)" -gt 1 ]; then
            export ECC_EXPERIMENT=panels
            export ECC_PANEL_MODELS="$MODELS"
            unset ECC_CONST_MODEL
        else
            export ECC_CONST_MODEL="$MODELS"
        fi
        ;;
    model)
        export ECC_SWEEP=model
        # Only the models whose whole-model cache exists will resolve; the run
        # stops with "nothing collected" naming the treatment if one is missing.
        export ECC_SWEEP_MODELS="${ECC_SWEEP_MODELS:-resnet18 mobilenet_v2}"
        export ECC_CONST_ARCH="${ECC_CONST_ARCH:-eyeriss_like}"
        export ECC_CONST_K="${ECC_CONST_K:-51}"
        ;;
    bch)
        # The cheapest sweep: BCH K changes only the ECC arithmetic on top of a
        # raw energy record, so this needs no new mapping at all.
        export ECC_SWEEP=bch
        export ECC_SWEEP_KS="${ECC_SWEEP_KS:-57 51 45 39 36 30}"
        export ECC_CONST_ARCH="${ECC_CONST_ARCH:-eyeriss_like}"
        export ECC_CONST_MODEL="${ECC_CONST_MODEL:-resnet18}"
        ;;
    panels)
        export ECC_EXPERIMENT=panels
        export ECC_SWEEP=arch
        export ECC_SWEEP_ARCHS="${ECC_SWEEP_ARCHS:-$ARCHS_ALL}"
        export ECC_PANEL_MODELS="${ECC_PANEL_MODELS:-resnet18 mobilenet_v2}"
        export ECC_CONST_K="${ECC_CONST_K:-51}"
        ;;
esac

# ONE output name per axis, whatever the layout underneath. Without this a
# panelled run would append `__panels__<models>` and a two-model figure would
# sit beside the one-model one instead of replacing it. Overriding is the point
# here: the axis is what names the file, and the manifest beside it records the
# constants. Safe only because ECC_LAYERS is forced empty above.
export ECC_STEM="${ECC_STEM:-$(case "$AXIS" in
    arch|panels) echo ArchitectureSweep ;;
    model)       echo ModelSweep ;;
    bch)         echo BCHsweep ;;
esac)}"

echo "=============================================================================="
echo " sweep axis : $AXIS   (all three ECC arms drawn: $ECC_APPROACHES)"
echo " scope      : whole model (ECC_LAYERS empty -> fixed output stem)"
if [ "${ECC_EXPERIMENT:-}" = "panels" ]; then
    echo " layout     : one panel per model -- ${ECC_PANEL_MODELS}"
fi
echo " output     : results/figures/${ECC_STEM}.png  (overwritten in place)"
echo "=============================================================================="

if [ "${ECC_EXPERIMENT:-}" = "panels" ]; then
    bash hpc/tl.sh bash run.sh panels --eval
else
    bash hpc/tl.sh bash run.sh --sweep "$AXIS" --eval
fi

echo
echo "figure   : results/figures/   (fixed stem, overwritten in place)"
echo "table    : results/tables/"
echo "manifest : results/manifests/ -- the constants behind the file on disk"
ls -la results/figures/*.png 2>/dev/null | tail -5
