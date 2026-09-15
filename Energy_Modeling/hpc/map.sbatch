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
#  ONE ARRAY TASK = ONE BUNDLE of the task file, which is column 1 of every
#  row. The rows are seven columns:
#
#      <bundle> <arch> <model> <K> <depth> <arm> <layer>
#
#  and one row is one UNIT of work: one CHIP (architecture x code x
#  buffer-depth scale x arm -- every distinct thing the mapper is handed) x one
#  distinct layer SHAPE. With ECC_JOBS empty every row is its own bundle, so a
#  task is one unit, exactly as it was before EnvReorganisation phase 3; with
#  ECC_JOBS set, a task walks its bundle's rows SEQUENTIALLY and
#  ECC_MAP_TIME has to cover all of them.
#
#  THE ROW IS THE AXES, and this file exports them per unit. `run_all.sh`'s
#  `unit_env()` has the same list and `toolchain.units._chip_cfg` builds the
#  same configuration in python: the launcher decides a unit is cached by
#  resolving that configuration, so if the three drift, it skips units a job
#  would re-solve or submits ones it already has.
#
#  Two models of one architecture may share layer shapes (resnet18/resnet50
#  share 5, mobilenet_v2/efficientnet_b0 share 11); that is safe at any
#  concurrency because `toolchain.cache.ShapeLock` puts an atomic-mkdir lock on
#  each shape's cache entry -- the second task waits and then takes the first
#  task's mapping as a cache hit. A lock left by a killed job is taken over
#  once its owner pid is gone, or after 6 h.
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
#SBATCH --array=0-0
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

# The task list is GENERATED from the configuration, never hand-written. A bare
# `sbatch hpc/map.sbatch` regenerates it if it is missing; env.sh's
# `ecc_write_taskfile` falls back to the reference arm of the held point when
# the enumerator cannot run, which is the one unit a bare sbatch could mean.
[ -f "${ECC_TASKFILE}" ] || ecc_write_taskfile

IDX="${SLURM_ARRAY_TASK_ID:-0}"
NBUNDLE=$(awk 'NF {print $1}' "${ECC_TASKFILE}" | sort -un | wc -l)

# A stale --array is the likeliest mistake: too small and units are silently
# skipped, too large and tasks die one by one. Both are caught here.
if [ "${IDX}" -ge "${NBUNDLE}" ]; then
    echo "map.sbatch: array index ${IDX} but ${ECC_TASKFILE} has only" \
         "${NBUNDLE} bundle(s)." >&2
    echo "  -> submit with 'bash hpc/run_all.sh', which sizes --array from the" >&2
    echo "     task file it just wrote." >&2
    exit 2
fi
if [ "${IDX}" -eq 0 ] && [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] \
   && [ "${SLURM_ARRAY_TASK_COUNT}" -lt "${NBUNDLE}" ]; then
    echo "map.sbatch: WARNING: --array covers ${SLURM_ARRAY_TASK_COUNT} task(s) but" \
         "${ECC_TASKFILE} has ${NBUNDLE} bundle(s):" \
         "$((NBUNDLE - SLURM_ARRAY_TASK_COUNT)) will NOT be mapped." \
         "Submit with 'bash hpc/run_all.sh'." >&2
fi

mapfile -t ROWS < <(awk -v b="${IDX}" 'NF && $1 == b' "${ECC_TASKFILE}")
[ "${#ROWS[@]}" -gt 0 ] || {
    echo "map.sbatch: no row of ${ECC_TASKFILE} carries bundle ${IDX}." >&2
    exit 2; }

module load apptainer

echo "=== bundle ${IDX}: ${#ROWS[@]} unit(s) on $(hostname) ==="
echo "=== cores=${ECC_MAPPER_THREADS}  search=${ECC_MAPPER_ALGORITHM}/${ECC_MAPPER_SEARCH_SIZE:-uncapped}" \
     "victory=${ECC_VICTORY}  rerun=${ECC_RERUN_OPTIMISER} ==="
echo "=== started $(date -Is) ==="

# LOCKS LEFT BY A CANCELLED JOB, CLEARED HERE -- and it has to be HERE, before
# `hpc/tl.sh` below, because `squeue` is not in the Timeloop image and a task
# already inside the container cannot tell a dead owner from a working one.
# `release()` is in a `finally` and a `finally` does not run on SIGTERM or
# SIGKILL, so every `scancel` of a running map leaves its locks behind.
# Measured 2026-09-15: eight locks from a cancelled array put ten tasks to
# sleep for hours on work that takes minutes. Only a lock whose SLURM job is
# GONE is removed; "cannot ask SLURM" removes nothing.
${ECC_PYTHON:-python3} -m eccenergy.toolchain.cache --prune 2>/dev/null \
    || echo "=== (lock prune skipped: the package did not import here) ==="

I=0
for ROW in "${ROWS[@]}"; do
    read -r _B ARCH MODEL CODE_K DEPTH ARM LAYER <<< "${ROW}"
    [ -n "${ARCH:-}" ] && [ -n "${MODEL:-}" ] && [ -n "${ARM:-}" ] || {
        echo "map.sbatch: '${ROW}' is not"                                    \
             "'<bundle> <arch> <model> <K> <depth> <arm> <layer>'" >&2
        exit 2; }
    I=$((I + 1))
    echo
    echo "=== unit ${I}/${#ROWS[@]}: ${ARCH} / ${MODEL} / K${CODE_K} /" \
         "x${DEPTH} / ${ARM} / ${LAYER:-<whole model>} ==="
    # ONE UNIT, ONE CHIP. The same list as run_all.sh's `unit_env()` and
    # `toolchain.units._chip_cfg()`; a subshell so one unit's axes cannot leak
    # into the next.
    (
        export ECC_SWEEP=arch
        export ECC_SWEEP_ARCHS="${ARCH}"  ECC_CONST_ARCH="${ARCH}"
        export ECC_CONST_MODEL="${MODEL}" ECC_SWEEP_MODELS="${MODEL}"
        export ECC_PANEL_MODELS=""
        export ECC_CONST_K="${CODE_K}"    ECC_SWEEP_KS="${CODE_K}"
        export ECC_WEIGHT_DEPTH_SCALE="${DEPTH}"
        export ECC_RECON_ERT_ARM="${ARM}"
        export ECC_LAYERS="${LAYER}"
        time bash hpc/tl.sh bash run.sh map
    )
done

echo
echo "=== finished $(date -Is) ==="
