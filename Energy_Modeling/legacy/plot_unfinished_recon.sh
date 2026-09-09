#!/usr/bin/env bash
# =============================================================================
#  TEMPORARY (2026-09-09): draw the ReconSweep figure from the layers that the
#  CURRENT v2 mapping run has already finished, while the rest are still
#  mapping. Uses the pipeline's own development mode (ECC_LAYERS) and a scratch
#  results dir, so nothing in the main pipeline or in results/_raw changes;
#  only the figure is copied into results/figures under a new name.
#
#      bash legacy/plot_unfinished_recon.sh                 # -> ReconSweep_unfinshed.png
#      bash legacy/plot_unfinished_recon.sh MyName          # -> MyName.png
#
#  The figure's own title says "DEVELOPMENT RUN — N-layer only" and lists the
#  layers, so it cannot be mistaken for the whole-model result.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
NAME="${1:-ReconSweep_unfinshed}"
SCRATCH="legacy/_unfinished_results"
LAYERS="$(bash hpc/tl.sh python3 legacy/unfinished_layers.py 2> >(grep -v '^mv:' >&2))"
[ -n "${LAYERS}" ] || { echo "no finished layers in the current cache yet" >&2; exit 1; }
echo "layers: ${LAYERS}"
rm -rf "${SCRATCH}"
ECC_LAYERS="${LAYERS}" ECC_RESULTS_DIR="${SCRATCH}" bash hpc/tl.sh bash run.sh recon --eval \
    2> >(grep -v '^mv:' >&2)
shopt -s nullglob
n=0
for f in "${SCRATCH}"/figures/*.png "${SCRATCH}"/figures/*.pdf; do
    ext="${f##*.}"
    cp -v "${f}" "results/figures/${NAME}.${ext}"; n=$((n + 1))
done
[ "${n}" -gt 0 ] || { echo "no figure was produced under ${SCRATCH}/figures" >&2; exit 1; }
for f in "${SCRATCH}"/tables/*.csv; do cp -v "${f}" "results/tables/${NAME}.csv"; done
echo "done: results/figures/${NAME}.png  (temporary, partial-layer figure)"
