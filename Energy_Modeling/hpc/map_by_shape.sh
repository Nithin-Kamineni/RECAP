#!/usr/bin/env bash
# =============================================================================
#  hpc/map_by_shape.sh -- map ONE (architecture, model) as many SLURM jobs,
#  one per distinct layer SHAPE, then evaluate+plot once they all finish.
#
#      bash hpc/map_by_shape.sh                 # the env.sh configuration
#      ECC_NOC_PE_LATCH_PJ=0.5 bash hpc/map_by_shape.sh
#
#  hpc/run_all.sh maps a whole model in ONE array task (one job walks every
#  layer with ECC_MAPPER_THREADS cores).  For a single architecture that is
#  hours of wall time on 18 cores while the rest of the allocation idles.
#  This launcher instead submits one job per shape, each with ECC_LAYERS set
#  to one layer of that shape.  It is safe because:
#    * the mapper cache is keyed by (architecture fingerprint, shape), NOT by
#      the layer list -- a layer-scoped map writes the same cache entry the
#      whole-model evaluation reads (paths.Results.mapper_cache);
#    * every shape is written under a ShapeLock, so two jobs cannot solve the
#      same shape twice;
#    * ECC_MAPPER_THREADS is in the fingerprint, so every job keeps the
#      configured thread count (18) and the entries are what a single-job run
#      would have produced.
#  The dependent eval job runs `hpc/run_all.sh --eval-only` WITHOUT a layer
#  list, so it evaluates and draws the whole model into ECC_RESULTS_DIR --
#  for ECC_RECON_MODELING=1 that is results/figures/ReconSweep.png.
#
#  Layer names are read from the workload file through the pipeline's own
#  loader inside the container; nothing here knows how a shape is named.
#
#  `--array=0-0` is REQUIRED: hpc/map.sbatch carries its own `#SBATCH --array`
#  directive sized for a whole ECC_ARCHS x ECC_MODELS task file. Without the
#  override each per-shape submission became a multi-task array whose extra
#  tasks exited 2 ("array index >= task-file lines"), and the dependent eval's
#  `afterok` was never satisfied (2026-09-09, jobs 41474947-58: every task 0
#  mapped its shape, every other task failed, eval 41474959 sat on
#  DependencyNeverSatisfied). The task file has exactly one line here, so
#  index 0 is the only valid task.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./env.sh

ARCH="$(set -- ${ECC_ARCHS}; echo "$1")"
MODEL="$(set -- ${ECC_MODELS}; echo "$1")"
[ "$(set -- ${ECC_ARCHS}; echo $#)" = 1 ] && [ "$(set -- ${ECC_MODELS}; echo $#)" = 1 ] || {
    echo "map_by_shape.sh: exactly one architecture and one model, got" \
         "ECC_ARCHS='${ECC_ARCHS}' ECC_MODELS='${ECC_MODELS}'" >&2; exit 2; }

# one representative layer per distinct shape, via the pipeline's loader
mapfile -t LAYERS < <(bash hpc/tl.sh python3 - "${MODEL}" 2>/dev/null <<'PY'
import sys
from eccenergy import config, workloads
cfg = config.load_config()
models, _ = workloads.load_workload(cfg)
seen = {}
for l in models[sys.argv[1]]:
    seen.setdefault(l.shape_name, l.name)
print("\n".join(seen.values()))
PY
)
[ "${#LAYERS[@]}" -gt 0 ] || { echo "map_by_shape.sh: no layers resolved for ${MODEL}" >&2; exit 2; }

mkdir -p hpc/logs
echo "map_by_shape: ${ARCH} x ${MODEL}: ${#LAYERS[@]} shapes -> ${#LAYERS[@]} jobs of ${ECC_MAP_CPUS} cores"
echo "  objective=${ECC_OPT_METRIC} victory=${ECC_VICTORY} latch=${ECC_NOC_PE_LATCH_PJ:-<noc.yaml>} results=${ECC_RESULTS_DIR}"
JOBS=()
for L in "${LAYERS[@]}"; do
    jid=$(ECC_LAYERS="${L}" sbatch --parsable \
        --job-name="map-${L}" \
        --account="${ECC_ACCOUNT}" --qos="${ECC_QOS}" --partition="${ECC_PARTITION}" \
        --cpus-per-task="${ECC_MAP_CPUS}" --mem="${ECC_MAP_MEM}" --time="${ECC_MAP_TIME}" \
        --array=0-0 \
        --output="hpc/logs/map-shape.%A.out" \
        --export=ALL,ECC_LAYERS="${L}" \
        hpc/map.sbatch)
    JOBS+=("${jid}")
    echo "  job ${jid}  ${L}"
done
DEP=$(IFS=:; echo "${JOBS[*]}")
EVAL=$(sbatch --parsable --dependency="afterok:${DEP}" \
    --job-name=ecc-eval \
    --account="${ECC_ACCOUNT}" --qos="${ECC_QOS}" --partition="${ECC_PARTITION}" \
    --cpus-per-task="${ECC_EVAL_CPUS}" --mem="${ECC_EVAL_MEM}" --time="${ECC_EVAL_TIME}" \
    --output="hpc/logs/ecc-eval.%j.out" \
    --export=ALL,ECC_LAYERS= \
    hpc/run_all.sh --eval-only)
echo "  eval job ${EVAL} (afterok all ${#JOBS[@]} map jobs) -> ${ECC_RESULTS_DIR}/figures/${ECC_STEM:-<self-describing>}"
echo "map_jobs=${DEP}" > hpc/.runtime/map_by_shape.last
echo "eval_job=${EVAL}" >> hpc/.runtime/map_by_shape.last
