#!/bin/bash
# =============================================================================
#  hpc/map.sbatch  --  THE ONE FILE YOU EDIT TO RUN THE MAPPER
# =============================================================================
#
#      cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
#      sbatch hpc/map.sbatch
#
#  This runs the expensive half of the project: Timeloop + Accelergy solve a
#  mapping for every distinct layer shape, and each result is cached with a
#  `mapping.json` sidecar. The cheap half (energy evaluation and the figures)
#  reads that cache afterwards and needs no allocation -- see `hpc/sweep.sh`.
#
#  HOW THE WORK IS SPLIT
#  ---------------------
#  One array task = one (architecture, model) pair, taken from TASKFILE below,
#  one pair per line. Two models of one architecture MAY share layer shapes
#  (resnet18/resnet50: 5, resnet50/densenet121: 4, mobilenet_v2/
#  efficientnet_b0: 11, two more pairs: 1 each); since 2026-09-07 that is safe
#  at any concurrency: `timeloop.ShapeLock` puts an atomic-mkdir lock on each
#  shape's cache entry, the second task waits and then takes the first task's
#  mapping as a cache hit ("[lock] ... is being mapped by another task" in its
#  log). A lock left by a killed job is taken over when its owner pid is gone
#  or it is older than 6 h.
#
#  Task files:  hpc/tasks_panel.txt   6 archs x {resnet18, mobilenet_v2}  (12)
#               hpc/tasks_full.txt    8 archs x 8 CNNs                    (64)
#               hpc/tasks_conv.txt    v1 + v2 on resnet18, for the convergence
#                                     check (run at two ECC_VICTORY values)
#
#  TO CHANGE WHAT RUNS
#  -------------------
#  1. edit `hpc/tasks_panel.txt` (or point TASKFILE at your own list), then
#  2. set --array below to `0-<lines-1>%<concurrency>`.
#     The script refuses to start if the two disagree, so a stale --array is an
#     immediate error rather than a silently half-finished sweep.
#
#  SECTION 1 is the SLURM request (cores, memory, time, how many at once).
#  SECTION 2 is the model configuration, in the same `${VAR:=default}` style as
#  run.sh -- the environment wins, so you can override any of it per run
#  without editing this file:
#
#      ECC_VICTORY=500 sbatch hpc/map.sbatch          # a cheaper search
#
# =============================================================================

# --------------------------------------------------------------------------
#  SECTION 1 -- SLURM: what to ask the cluster for
# --------------------------------------------------------------------------
#  These are `#SBATCH` directives: sbatch reads them as text before the script
#  runs, so they CANNOT contain shell variables. Edit the numbers in place.
#
#  --array=0-11%9 means: 12 tasks (indices 0..11), at most 9 running at once.
#  The `%9` is the parallelism knob. The rewetz investment is 181 cores, so at
#  18 cores per task 10 fit; 9 leaves room for a VS Code ondemand session
#  (which holds 9 by itself). Anything above the limit just queues with reason
#  `JobArrayTaskLimit` -- it is not an error. Check headroom with
#  `slurmInfo -g rewetz`.
#
#  --cpus-per-task=18 IS NOT A FREE CHOICE. The mapper thread count is hashed
#  into the mapping fingerprint, so a different value is a different cache key:
#  every mapping already computed (the laptop's and this cluster's) becomes a
#  MISS and the whole sweep re-runs from cold. Change it only if you intend
#  exactly that. 18 is what every cached entry was built at.
#
#  --time: the longest BOUNDED task (search_size 20000) was 31 minutes
#  (simple_output_stationary x mobilenet_v2, 31 shapes cold). An UNCAPPED run
#  keeps each thread going until `victory` consecutive misses, typically 4-10x
#  the bounded cost, and twice that again on the 9/10-level designs whose
#  victory is depth-scaled to 4000/8000. 24 h covers the worst measured
#  extrapolation; a shorter request schedules sooner, so trim it for dev runs.
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

set -euo pipefail

PROJ="/blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling"
cd "$PROJ"

# --------------------------------------------------------------------------
#  SECTION 2 -- the model configuration (same style as run.sh: env wins)
# --------------------------------------------------------------------------

#  Which (architecture, model) pairs to map: one "<arch> <model>" per line.
: "${TASKFILE:=hpc/tasks_panel.txt}"

#  ---- the mapper search -------------------------------------------------
#  These four decide what the search returns AND are hashed into the mapping
#  fingerprint, so changing any of them starts a new cache. hpc/eval_panel.sh
#  and hpc/sweep.sh read the SAME variables with the SAME defaults, so a run
#  mapped here is found by them -- as long as you export the same values to
#  both sides when you override.
#
#  SEARCH_SIZE is a cap on VALID mappings per run, split across the threads
#  (20000 / 18 = 1112 per thread). VICTORY stops a thread after that many
#  consecutive non-improving valid mappings. Victory can only ever fire if
#  search_size/threads exceeds it -- at 20000/18 = 1112 < 2000 it never did,
#  and every 2026-09-06 mapping was cut by the cap (see FINDINGS.md). Leave
#  SEARCH_SIZE EMPTY for a converged run (the result then carries no BOUNDED
#  warning); set 20000 for a quick development pass.
#
#  2026-09-07: the interconnect is now costed (ECC_NOC=1 by default, see
#  archs/_shared/noc.yaml). That adds `noc` to the cache slug, so every
#  pre-NoC mapping is a MISS by design and the next run of this file is a
#  cold run for all (architecture, model) pairs. ECC_NOC=0 reaches the old
#  caches, but reports a free interconnect -- do not publish from it.
: "${ECC_MAPPER_ALGORITHM:=random_pruned}"
: "${ECC_MAPPER_SEARCH_SIZE=}"      # EMPTY = uncapped (converged, publishable). 20000 = bounded dev run
: "${ECC_VICTORY:=2000}"          # scaled by loop depth -> 4000 on 9-level designs
: "${ECC_MAPPER_TIMEOUT:=2000}"

#  ---- what to map -------------------------------------------------------
#  Empty ECC_LAYERS = the WHOLE model, every distinct shape. Set it to one or
#  two stable layer names for a quick development run instead.
: "${ECC_LAYERS:=}"

#  ---- do not change without meaning to ----------------------------------
#  Threads follow the cores SLURM actually granted, so the fingerprint can
#  never drift from the allocation.
ECC_MAPPER_THREADS="${SLURM_CPUS_PER_TASK:-18}"

export ECC_MAPPER_ALGORITHM ECC_MAPPER_SEARCH_SIZE ECC_VICTORY \
       ECC_MAPPER_TIMEOUT ECC_MAPPER_THREADS ECC_LAYERS

# --------------------------------------------------------------------------
#  SECTION 3 -- run it (nothing below here is a knob)
# --------------------------------------------------------------------------

[ -f "$TASKFILE" ] || { echo "map.sbatch: no such task file: $TASKFILE" >&2; exit 2; }

NLINES=$(grep -cve '^[[:space:]]*$' "$TASKFILE")
IDX="${SLURM_ARRAY_TASK_ID:-0}"

# A stale --array is the likeliest mistake after editing the task file: too
# small and pairs are silently skipped, too large and tasks die one by one.
if [ "$IDX" -ge "$NLINES" ]; then
    echo "map.sbatch: array index $IDX but $TASKFILE has only $NLINES line(s)." >&2
    echo "  -> set '#SBATCH --array=0-$((NLINES - 1))%<concurrency>' in this file." >&2
    exit 2
fi

LINE="$(grep -ve '^[[:space:]]*$' "$TASKFILE" | sed -n "$((IDX + 1))p")"
ARCH="$(echo "$LINE" | awk '{print $1}')"
MODEL="$(echo "$LINE" | awk '{print $2}')"
[ -n "$ARCH" ] && [ -n "$MODEL" ] || {
    echo "map.sbatch: line $((IDX + 1)) of $TASKFILE is not '<arch> <model>': '$LINE'" >&2
    exit 2; }

module load apptainer

# One architecture, one model, per task.
export ECC_SWEEP=arch
export ECC_SWEEP_ARCHS="$ARCH"
export ECC_CONST_MODEL="$MODEL"

echo "=== task $IDX: arch=$ARCH model=$MODEL on $(hostname) ==="
echo "=== cores=$ECC_MAPPER_THREADS  search=$ECC_MAPPER_ALGORITHM/$ECC_MAPPER_SEARCH_SIZE"\
     "victory=$ECC_VICTORY  layers=${ECC_LAYERS:-<whole model>} ==="
echo "=== started $(date -Is) ==="

time bash hpc/tl.sh bash run.sh map

echo "=== finished $(date -Is) ==="
