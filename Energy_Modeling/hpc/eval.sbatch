#!/bin/bash
# =============================================================================
#  hpc/eval.sbatch -- evaluate + draw from the mapper cache, as a SLURM job.
#  Submitted by hpc/run_all.sh with --dependency on the map array, so it starts
#  when every mapping exists. Never invokes Timeloop (ECC_FROM_CACHE=1), so it
#  asks for 2 cores. ECC_* variables pass through from the submitting shell.
# =============================================================================
#SBATCH --job-name=ecc-eval
#SBATCH --account=rewetz
#SBATCH --qos=rewetz
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=02:00:00
#SBATCH --output=/blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling/hpc/logs/%x.%j.out
set -euo pipefail
cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
module load apptainer
echo "=== evaluate + draw: archs=[${ECC_SWEEP_ARCHS:-default}] models=[${ECC_PANEL_MODELS:-default}] $(date -Is) ==="
bash hpc/eval_panel.sh
echo "=== done $(date -Is) ==="
