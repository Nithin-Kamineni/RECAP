#!/usr/bin/env bash
# =============================================================================
#  hpc/map_by_shape.sh -- map ONE (architecture, model) as many SLURM jobs,
#  one per distinct layer SHAPE, then evaluate+plot once they all finish.
#
#      bash hpc/map_by_shape.sh                 # the env.sh configuration
#      ECC_NOC_PE_LATCH_PJ=0.5 bash hpc/map_by_shape.sh
#      ECC_RECON_ARCHS="simple_weight_stationary eyeriss_like" \
#          bash hpc/map_by_shape.sh             # both designs, one eval after
#      bash hpc/map_by_shape.sh --progress      # how much is really cached
#      bash hpc/map_by_shape.sh --no-eval       # maps only, no dependent eval
#
#  SEVERAL ARCHITECTURES, ONE EVAL. ECC_ARCHS may name more than one design
#  (with ECC_RECON_MODELING=1 that is ECC_RECON_ARCHS): the launcher then
#  submits shapes x architectures jobs, all in parallel, and ONE dependent eval
#  after all of them. That matters for the reconstruction placement study, whose
#  figure is one panel per architecture in a single file -- submitting one
#  launcher per design would give each its own eval, and the second would
#  overwrite the first one'''s ReconSweep.png with a one-panel figure.
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
#  THE TASK FILE IS SNAPSHOTTED, not shared. `map.sbatch` resolves its
#  (architecture, model) pair from line `--array` index + 1 of ECC_TASKFILE, and
#  it reads that file when the job RUNS, not when it is submitted. The shared
#  hpc/.runtime/tasks.txt is therefore regenerated here and then copied to a
#  per-submission file that every job of this submission is given through
#  ECC_TASKFILE, so a later `run_all.sh` (or a second launcher) cannot change
#  what a queued job maps. Before this, the launcher relied on the shared file
#  already happening to hold the right single line.
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

# --no-eval: submit the mapping jobs and NOT the dependent eval. Use it to map a
# design in stages, or to add architectures from several launchers, and then run
# the eval once by hand (or with `hpc/run_all.sh --eval-only`) when every shape
# is in the cache. Parsed before env.sh so it never reaches the pipeline.
WANT_EVAL=1
WANT_PROGRESS=0
for a in "$@"; do
    case "$a" in
        --no-eval)  WANT_EVAL=0 ;;
        --progress) WANT_PROGRESS=1 ;;
        *) echo "map_by_shape.sh: unknown option '$a'" >&2
           echo "  options: --no-eval  --progress" >&2
           echo "  (an unrecognised flag USED to fall through and submit a whole" >&2
           echo "   duplicate wave plus a second eval job, so it is fatal now)" >&2
           exit 2 ;;
    esac
done
set --

source ./env.sh

# --progress: how much of ECC_ARCHS is actually in the mapper cache, per
# architecture, at the CURRENT fingerprint. Submits nothing.
#
# COUNT THE stats.txt, NOT THE DIRECTORIES. A shape's cache directory is created
# when a job CLAIMS the shape, so `ls | wc -l` reports a whole wave as finished
# within seconds of it starting -- exactly the wrong answer to "is it done yet".
# A shape is solved when its `timeloop-mapper.stats.txt` exists; a
# `<shape>.lock` beside it means a job is working on it right now.
if [ "${WANT_PROGRESS}" = "1" ]; then
    echo "mapper cache: objective=${ECC_OPT_METRIC} victory=${ECC_VICTORY}" \
         "threads=${ECC_MAPPER_THREADS} fidelity=${ECC_ARCH_FIDELITY}"
    for A in ${ECC_ARCHS}; do
        D=$(ECC_SWEEP=arch ECC_SWEEP_ARCHS="${A}" bash hpc/tl.sh python3 -c "
from eccenergy import config, paths
from eccenergy.arch import fingerprint
cfg = config.load_config()
a = cfg.archs[0]
print(paths.Results(cfg).mapper_cache(a, fingerprint.effective_variant(a, cfg),
                                      fingerprint.arch_fingerprint(a, cfg)))" 2>/dev/null | tail -1)
        if [ -z "${D}" ] || [ ! -d "${D}" ]; then
            printf "  %-26s no cache directory at this fingerprint yet\n" "${A}"
            continue
        fi
        SOLVED=$(find "${D}" -mindepth 2 -maxdepth 2 -name timeloop-mapper.stats.txt 2>/dev/null | wc -l)
        LOCKED=$(find "${D}" -maxdepth 1 -name '*.lock' 2>/dev/null | wc -l)
        CLAIMED=$(find "${D}" -mindepth 1 -maxdepth 1 -type d -not -name '*.lock' 2>/dev/null | wc -l)
        printf "  %-26s solved=%-3s in-progress=%-3s claimed=%-3s  %s\n" \
               "${A}" "${SOLVED}" "${LOCKED}" "${CLAIMED}" "${D#${PWD}/}"
    done
    echo "  solved counts timeloop-mapper.stats.txt; claimed counts directories and"
    echo "  runs ahead of solved during a wave, so do not read it as progress."
    exit 0
fi

MODEL="$(set -- ${ECC_MODELS}; echo "$1")"
NARCH="$(set -- ${ECC_ARCHS}; echo $#)"
[ "${NARCH}" -ge 1 ] && [ "$(set -- ${ECC_MODELS}; echo $#)" = 1 ] || {
    echo "map_by_shape.sh: one or more architectures and exactly ONE model, got" \
         "ECC_ARCHS='${ECC_ARCHS}' ECC_MODELS='${ECC_MODELS}'" >&2; exit 2; }

# One line per architecture, in ECC_ARCHS order, so `map.sbatch --array=i-i`
# resolves architecture i. Snapshotted below so a queued job cannot be pointed
# at a different pair by a later run.
mkdir -p hpc/.runtime
ecc_write_taskfile
SNAP="hpc/.runtime/tasks.by-shape.$$.txt"
cp "${ECC_TASKFILE}" "${SNAP}"
NLINES=$(grep -cve '^[[:space:]]*$' "${SNAP}")
[ "${NLINES}" = "${NARCH}" ] || {
    echo "map_by_shape.sh: task file has ${NLINES} line(s) for ${NARCH}" \
         "architecture(s) and one model -- refusing rather than mapping the" \
         "wrong pair" >&2; exit 2; }

# One representative layer per distinct shape, via the pipeline's loader -- or
# exactly the layers ECC_LAYERS names, which is how a design is spot-checked on
# one or two shapes before its other ten are queued. The cache is keyed by
# (architecture fingerprint, shape), so a spot-check's mapping IS the entry the
# later whole-model run reads: nothing is mapped twice.
if [ -n "${ECC_LAYERS}" ]; then
    read -r -a LAYERS <<< "${ECC_LAYERS}"
    echo "map_by_shape: ECC_LAYERS is set -- mapping exactly ${#LAYERS[@]} named layer(s)"
else
mapfile -t LAYERS < <(bash hpc/tl.sh python3 - "${MODEL}" 2>/dev/null <<'PY'
import sys
from eccenergy import config
from eccenergy.arch import workloads
cfg = config.load_config()
models, _ = workloads.load_workload(cfg)
seen = {}
for l in models[sys.argv[1]]:
    seen.setdefault(l.shape_name, l.name)
print("\n".join(seen.values()))
PY
)
fi

[ "${#LAYERS[@]}" -gt 0 ] || { echo "map_by_shape.sh: no layers resolved for ${MODEL}" >&2; exit 2; }

mkdir -p hpc/logs
echo "map_by_shape: ${NARCH} architecture(s) x ${#LAYERS[@]} shapes x ${MODEL}" \
     "-> $((NARCH * ${#LAYERS[@]})) jobs of ${ECC_MAP_CPUS} cores"
echo "  architectures: ${ECC_ARCHS}"
echo "  objective=${ECC_OPT_METRIC} victory=${ECC_VICTORY} latch=${ECC_NOC_PE_LATCH_PJ:-<noc.yaml>} results=${ECC_RESULTS_DIR}"
echo "  task file: ${SNAP}"
JOBS=()
I=0
for A in ${ECC_ARCHS}; do
    for L in "${LAYERS[@]}"; do
        jid=$(sbatch --parsable \
            --job-name="map-${A}-${L}" \
            --account="${ECC_ACCOUNT}" --qos="${ECC_QOS}" --partition="${ECC_PARTITION}" \
            --cpus-per-task="${ECC_MAP_CPUS}" --mem="${ECC_MAP_MEM}" --time="${ECC_MAP_TIME}" \
            --array="${I}-${I}" \
            --output="hpc/logs/map-shape.%A.out" \
            --export=ALL,ECC_LAYERS="${L}",ECC_TASKFILE="${PWD}/${SNAP}" \
            hpc/map.sbatch)
        JOBS+=("${jid}")
        echo "  job ${jid}  ${A}  ${L}"
    done
    I=$((I + 1))
done
DEP=$(IFS=:; echo "${JOBS[*]}")
if [ "${WANT_EVAL}" != "1" ]; then
    echo "  --no-eval: no dependent evaluation submitted."
    echo "  when every shape is cached:  bash hpc/run_all.sh --eval-only"
    echo "  or depend on these:          --dependency=afterok:${DEP}"
    mkdir -p hpc/.runtime
    echo "map_jobs=${DEP}" > hpc/.runtime/map_by_shape.last
    exit 0
fi
EVAL=$(sbatch --parsable --dependency="afterok:${DEP}" \
    --job-name=ecc-eval \
    --account="${ECC_ACCOUNT}" --qos="${ECC_QOS}" --partition="${ECC_PARTITION}" \
    --cpus-per-task="${ECC_EVAL_CPUS}" --mem="${ECC_EVAL_MEM}" --time="${ECC_EVAL_TIME}" \
    --output="hpc/logs/ecc-eval.%j.out" \
    --export=ALL,ECC_LAYERS= \
    hpc/run_all.sh --eval-only)
echo "  eval job ${EVAL} (afterok all ${#JOBS[@]} map jobs) -> ${ECC_RESULTS_DIR}/figures/${ECC_STEM:-<self-describing>}"
echo "  the eval evaluates ALL of ECC_ARCHS in one run: ${ECC_ARCHS}"
echo "map_jobs=${DEP}" > hpc/.runtime/map_by_shape.last
echo "eval_job=${EVAL}" >> hpc/.runtime/map_by_shape.last
