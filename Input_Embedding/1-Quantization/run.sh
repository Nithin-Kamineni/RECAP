#!/bin/bash
# =============================================================================
# SLURM submit script — 1-Quantization (QAT int8 for RECAP)
#
# Usage:
#   cd 1-Quantization && sbatch run.sh
#
# What it does:
#   For every arch in ARCHS, QAT-fine-tunes a torchvision-pretrained ImageNet
#   model with int8 fake-quantized weights (Jacob et al. 2018 scheme, via
#   qat_train.py copied from ECC-CODE-Engine). Only imagenet-val/ exists on
#   disk (no train/ split), so qat_train.py takes a deterministic stratified
#   80/20 per-class split of it: 40 img/class for fine-tuning, 10 img/class
#   held out for its own in-loop evaluation.
#
#   Output: 0-Data/artifacts/models/imagenet/<arch>/QAT/model_int8_qat.pth
#           (same {qstate_dict/state_dict, meta} payload format used
#           throughout ECC-CODE-Engine, loadable by 3-Testing.)
#
# Overrides (env before sbatch):
#   ARCHS="resnet18"      restrict to one architecture
#   QUANTIZE_BITS="8"     bit width (default 8 — this study is 8-bit only)
#   QAT_EPOCHS=15 QAT_LR=1e-4
# =============================================================================

#SBATCH --job-name=1-recap-quantize
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=32gb
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err

date; hostname; pwd
mkdir -p logs

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/../env.sh"

module load singularity

# ---- One SLURM array task per architecture ----
read -ra ARCH_ARR <<< "${ARCHS}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${IDX}" -ge "${#ARCH_ARR[@]}" ]; then
    echo "[1-Quantization] array index ${IDX} has no matching arch (only ${#ARCH_ARR[@]} archs); exiting."
    exit 0
fi
ARC="${ARCH_ARR[$IDX]}"

echo "[1-Quantization/run.sh] SIF=${SIF}"
echo "[1-Quantization/run.sh] ARCH (task ${IDX}) = ${ARC}"
echo "[1-Quantization/run.sh] DATASET=${DATASET}  QUANTIZE_BITS=${QUANTIZE_BITS}"
echo "[1-Quantization/run.sh] QAT_EPOCHS=${QAT_EPOCHS}  QAT_LR=${QAT_LR}"

for BITS in ${QUANTIZE_BITS}; do
    echo "========================================================"
    echo "[1-Quantization] QAT fine-tuning ${ARC} on ${DATASET} @ ${BITS}-bit"
    echo "========================================================"
    singularity exec \
        --nv \
        --bind /blue \
        "${SIF}" \
        python3 "${SCRIPT_DIR}/qat_train.py" \
            --data-root       "${DATASET_DIR:-${DATA_ROOT}/data}" \
            --artifacts-root  "${ARTIFACTS_DIR}" \
            --imagenet-root   "${IMAGENET_ROOT}" \
            --use-pretrained  1 \
            --dataset         "${DATASET}" \
            --arch            "${ARC}" \
            --bits            "${BITS}" \
            --epochs          "${QAT_EPOCHS}" \
            --lr              "${QAT_LR}" \
            --skip-first-last "${QAT_SKIP_FIRST_LAST}" \
            --batch-size      "${BATCH_SIZE}" \
            --workers         "${WORKERS}"
    echo "[1-Quantization] ${ARC} @ ${BITS}-bit done (exit $?)"
done

echo "[1-Quantization/run.sh] Array task ${IDX} (${ARC}) complete."
