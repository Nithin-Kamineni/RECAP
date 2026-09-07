#!/bin/bash
# =============================================================================
#  hpc/run_all.sh -- THE ONE COMMAND.
#
#      cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
#      bash hpc/run_all.sh                          # map + evaluate + draw
#
#  Submits hpc/map.sbatch as a SLURM array over TASKFILE (one "<arch> <model>"
#  per line), then hpc/eval.sbatch with --dependency=afterok on it, so the
#  evaluation and the figures run by themselves when every mapping is done.
#  Returns at once; watch with `squeue -u $USER`. Resumable: rerun after a
#  failure and cached shapes are skipped.
#
#      TASKFILE=hpc/tasks_full.txt bash hpc/run_all.sh    # 8 archs x 8 CNNs
#      CONCURRENCY=6 bash hpc/run_all.sh                  # fewer tasks at once
#      MAP_TIME=08:00:00 bash hpc/run_all.sh              # shorter wall request
#                                                         # schedules sooner; the
#                                                         # 24 h default is for the
#                                                         # uncapped full matrix
#      ECC_VICTORY=4000 bash hpc/run_all.sh               # any ECC_* knob: both
#                                                         # jobs inherit it, so
#                                                         # map and eval agree
#  If a map task fails, the eval job is cancelled by SLURM (afterok). Fix, rerun
#  this script (cache hits skip the finished shapes), or evaluate what exists:
#      bash hpc/eval_panel.sh
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASKFILE:=hpc/tasks_panel.txt}"
: "${CONCURRENCY:=9}"          # 18 cores each; the rewetz investment is 181 cores
: "${MAP_TIME:=24:00:00}"      # per-task wall limit; a shorter one is scheduled sooner
[ -f "$TASKFILE" ] || { echo "run_all.sh: no such task file: $TASKFILE" >&2; exit 2; }
N=$(grep -cve '^[[:space:]]*$' "$TASKFILE")
ARCHS=$(awk 'NF{print $1}' "$TASKFILE" | awk '!seen[$0]++' | tr '\n' ' ' | sed 's/ $//')
MODELS=$(awk 'NF{print $2}' "$TASKFILE" | awk '!seen[$0]++' | tr '\n' ' ' | sed 's/ $//')
export TASKFILE ECC_SWEEP_ARCHS="$ARCHS" ECC_PANEL_MODELS="$MODELS"
MAP=$(sbatch --parsable --time="${MAP_TIME}" --array="0-$((N - 1))%${CONCURRENCY}" hpc/map.sbatch)
EVAL=$(sbatch --parsable --dependency="afterok:${MAP}" hpc/eval.sbatch)
echo "map  job ${MAP}: ${N} (arch, model) tasks from ${TASKFILE}, ${CONCURRENCY} at once, ${MAP_TIME} each"
echo "eval job ${EVAL}: runs when all map tasks succeed -> results/figures/, results/tables/, results/evaluation/"
echo "archs : ${ARCHS}"
echo "models: ${MODELS}"
echo "watch : squeue -u \$USER        logs: hpc/logs/ecc-map.${MAP}_*.out  hpc/logs/ecc-eval.${EVAL}.out"
