#!/bin/bash
# =============================================================================
#  hpc/map.sbatch  --  the mapping half, as a SLURM array
# =============================================================================
#
#  Normally you do not run this by hand -- `bash hpc/run_all.sh` submits it and
#  chains the evaluation onto it. By hand it still works:
#
#      sbatch hpc/map.sbatch
#
#  THERE ARE NO KNOBS IN THIS FILE. What runs, how hard the mapper searches and
#  what the cluster is asked for all come from ../env.sh.
#
#  ONE ARRAY TASK = ONE (architecture, model) PAIR, from the task list env.sh
#  generates out of ECC_ARCHS x ECC_MODELS. Two models of one architecture may
#  share layer shapes (resnet18/resnet50 share 5, mobilenet_v2/efficientnet_b0
#  share 11); that is safe at any concurrency because `timeloop.ShapeLock` puts
#  an atomic-mkdir lock on each shape's cache entry -- the second task waits and
#  then takes the first task's mapping as a cache hit. A lock left by a killed
#  job is taken over once its owner pid is gone, or after 6 h.
#
#  The #SBATCH directives below are a FALLBACK for a bare `sbatch`: sbatch reads
#  them as text before the script runs, so they cannot contain variables.
#  hpc/run_all.sh passes --account/--qos/--partition/--cpus-per-task/--mem/
#  --time/--array on the command line from env.sh, which overrides all of them.
#
#  --cpus-per-task IS NOT A FREE CHOICE. The mapper thread count is hashed into
#  the mapping fingerprint, so a different value is a different cache key and
#  every mapping already computed becomes a MISS. 18 is what every cached entry
#  was built at. Change ECC_MAPPER_THREADS (and ECC_MAP_CPUS with it) in env.sh
#  only if you intend exactly that.
#
#SBATCH --job-name=ecc-map
#SBATCH --account=rewetz
#SBATCH --qos=rewetz
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=16gb
#SBATCH --time=24:00:00
#SBATCH --array=0-11%9
#SBATCH --output=hpc/logs/%x.%A_%a.out
# =============================================================================
set -euo pipefail

cd "/blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling"
source ./env.sh

# Threads follow the cores SLURM actually granted, so the cache key can never
# drift from the allocation.
if [ -n "${SLURM_CPUS_PER_TASK:-}" ] && [ "${SLURM_CPUS_PER_TASK}" != "${ECC_MAPPER_THREADS}" ]; then
    echo "map.sbatch: WARNING: SLURM gave ${SLURM_CPUS_PER_TASK} cores but"      \
         "ECC_MAPPER_THREADS=${ECC_MAPPER_THREADS}. Using ${SLURM_CPUS_PER_TASK}" \
         "-- set ECC_MAP_CPUS and ECC_MAPPER_THREADS together in env.sh." >&2
    export ECC_MAPPER_THREADS="${SLURM_CPUS_PER_TASK}"
fi

# The task list is generated from ECC_ARCHS x ECC_MODELS, never hand-written.
# A bare `sbatch hpc/map.sbatch` regenerates it if it is missing.
[ -f "${ECC_TASKFILE}" ] || ecc_write_taskfile

NLINES=$(grep -cve '^[[:space:]]*$' "${ECC_TASKFILE}")
IDX="${SLURM_ARRAY_TASK_ID:-0}"

# A stale --array is the likeliest mistake: too small and pairs are silently
# skipped, too large and tasks die one by one. Both are caught here.
if [ "${IDX}" -ge "${NLINES}" ]; then
    echo "map.sbatch: array index ${IDX} but ${ECC_TASKFILE} has only ${NLINES} line(s)." >&2
    echo "  -> submit with 'bash hpc/run_all.sh', which sizes --array from env.sh." >&2
    exit 2
fi
if [ "${IDX}" -eq 0 ] && [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] \
   && [ "${SLURM_ARRAY_TASK_COUNT}" -lt "${NLINES}" ]; then
    echo "map.sbatch: WARNING: --array covers ${SLURM_ARRAY_TASK_COUNT} task(s) but" \
         "${ECC_TASKFILE} has ${NLINES}: $((NLINES - SLURM_ARRAY_TASK_COUNT)) pair(s)" \
         "will NOT be mapped. Submit with 'bash hpc/run_all.sh'." >&2
fi

LINE="$(grep -ve '^[[:space:]]*$' "${ECC_TASKFILE}" | sed -n "$((IDX + 1))p")"
ARCH="$(echo "${LINE}" | awk '{print $1}')"
MODEL="$(echo "${LINE}" | awk '{print $2}')"
[ -n "${ARCH}" ] && [ -n "${MODEL}" ] || {
    echo "map.sbatch: line $((IDX + 1)) of ${ECC_TASKFILE} is not '<arch> <model>': '${LINE}'" >&2
    exit 2; }

module load apptainer

# One architecture, one model, per task.
export ECC_SWEEP=arch
export ECC_SWEEP_ARCHS="${ARCH}"
export ECC_CONST_MODEL="${MODEL}"

echo "=== task ${IDX}: arch=${ARCH} model=${MODEL} on $(hostname) ==="
echo "=== cores=${ECC_MAPPER_THREADS}  search=${ECC_MAPPER_ALGORITHM}/${ECC_MAPPER_SEARCH_SIZE:-uncapped}" \
     "victory=${ECC_VICTORY}  layers=${ECC_LAYERS:-<whole model>}" \
     "rerun=${ECC_RERUN_OPTIMISER} ==="
echo "=== started $(date -Is) ==="

time bash hpc/tl.sh bash run.sh map

echo "=== finished $(date -Is) ==="
