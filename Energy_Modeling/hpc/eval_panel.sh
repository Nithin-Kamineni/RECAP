#!/usr/bin/env bash
# =============================================================================
#  hpc/eval_panel.sh -- evaluate the panel models across every architecture
#
#      cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
#      module load apptainer
#      bash hpc/eval_panel.sh
#
#  Runs AFTER hpc/map.sbatch has filled the mapper cache. Every command
#  here is `--eval` (ECC_FROM_CACHE=1): it reads solved mappings and never
#  invokes Timeloop, so it is safe on a login node and cannot change a mapping.
#
#  The environment below is the converged search setting -- byte-for-byte the
#  one that produced the laptop totals and the hpc/HIPERGATOR.md section 5
#  check 4 numbers. It must match what map.sbatch used, because the
#  mapper settings and the thread count are hashed into the cache key.
# =============================================================================
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ARCHS="${ECC_SWEEP_ARCHS:-eyeriss_v2_like eyeriss_like simple_weight_stationary simple_output_stationary simple_input_stationary simba_like}"
MODELS="${ECC_PANEL_MODELS:-resnet18 mobilenet_v2}"

# The mapper settings select the CACHE SLUG. They must equal what hpc/map.sbatch
# mapped with, so they default to its defaults and honour the same overrides.
# An exported EMPTY ECC_MAPPER_SEARCH_SIZE means uncapped, like map.sbatch.
export ECC_MAPPER_ALGORITHM="${ECC_MAPPER_ALGORITHM:-random_pruned}"
export ECC_MAPPER_SEARCH_SIZE="${ECC_MAPPER_SEARCH_SIZE-}"
export ECC_VICTORY="${ECC_VICTORY:-2000}"
export ECC_MAPPER_TIMEOUT="${ECC_MAPPER_TIMEOUT:-2000}"
export ECC_MAPPER_THREADS="${ECC_MAPPER_THREADS:-18}"
export ECC_SWEEP=arch
export ECC_SWEEP_ARCHS="$ARCHS"
export ECC_PANEL_MODELS="$MODELS"
export ECC_LAYERS=""          # empty = whole model, not the two dev layers

# ---- per-model result JSONs (results/evaluation/Pre/<arch>/<model>/...) ------
for m in $MODELS; do
    echo "############ baseline --eval : $m ############"
    ECC_CONST_MODEL="$m" bash hpc/tl.sh bash run.sh baseline --eval
done

# ---- the panel figure: one panel per model, architectures on the x axis -----
# ECC_STEM keeps this on the same `ArchitectureSweep` name hpc/sweep.sh uses,
# instead of appending `__panels__<models>`. One axis, one file name.
echo "############ panels --eval ############"
ECC_STEM="${ECC_STEM:-ArchitectureSweep}" bash hpc/tl.sh bash run.sh panels --eval

echo
echo "figures : results/figures/ArchitectureSweep__panels__*.png"
echo "tables  : results/tables/"
echo "results : results/evaluation/Pre/<arch>/<model>/..."
