#!/bin/bash
# =============================================================================
# SLURM submit script — 3-Testing (SECDED input-embedding accuracy)
#
# Usage:
#   cd 3-Testing && sbatch run.sh
#
# What it does:
#   One SLURM array task per architecture (mirrors 4-EmbeddingECC/run.sh's
#   pattern of parallelizing independent units across array tasks). Each
#   task loads that architecture's QAT int8 checkpoint, calibrates per-layer
#   activation scales, then evaluates Top-1/Top-5 on the FULL ImageNet-1k
#   val set three ways: (1) int8 baseline, no embedding; (2) every
#   Conv2d/Linear input SECDED-embedded; (3) — only if (2) drops accuracy
#   past ACC_DROP_TOLERANCE_PP — an auto-searched per-layer skip-list
#   ("regular ECC" on the worst layers) re-evaluated on the full set.
#
#   Within each task, the SECDED encoder itself is vectorized (GF(2) matrix
#   ops over the whole activation tensor) and thread-parallel across its 8
#   independent bit-chunk phases (implementations/secded_vectorized.py) —
#   i.e. parallelism happens at both the SLURM-array level (across
#   architectures) and the CPU-core level (within each encode call),
#   analogous to how 4-EmbeddingECC/run.sh parallelizes across t-values at
#   the array level and across chunks via its worker-process pool.
#
#   Output: 0-Data/artifacts/accuracy_results/imagenet/<arch>/QAT/8-bit/
#               secded_input_embed_accuracy.json
#
# Overrides (env before sbatch):
#   ARCHS="resnet18"          restrict to one architecture
#   ACC_DROP_TOLERANCE_PP=2.0 MAX_SKIP_FRACTION=0.3
# =============================================================================

#SBATCH --job-name=3-recap-input-embed
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l4:1
#SBATCH --mem=20gb
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err

date; hostname; pwd
mkdir -p logs

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/../env.sh"

module load singularity

read -ra ARCH_ARR <<< "${ARCHS}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${IDX}" -ge "${#ARCH_ARR[@]}" ]; then
    echo "[3-Testing] array index ${IDX} has no matching arch (only ${#ARCH_ARR[@]} archs); exiting."
    exit 0
fi
ARC="${ARCH_ARR[$IDX]}"

echo "[3-Testing/run.sh] SIF=${SIF}"
echo "[3-Testing/run.sh] ARCH (task ${IDX}) = ${ARC}"
echo "[3-Testing/run.sh] QUANTIZE_BITS=${QUANTIZE_BITS}  CALIB_BATCHES=${CALIB_BATCHES}"
echo "[3-Testing/run.sh] ACC_DROP_TOLERANCE_PP=${ACC_DROP_TOLERANCE_PP}  MAX_SKIP_FRACTION=${MAX_SKIP_FRACTION}"
echo "[3-Testing/run.sh] IMAGENET_ROOT=${IMAGENET_ROOT}"

for BITS in ${QUANTIZE_BITS}; do
    echo "========================================================"
    echo "[3-Testing] input-embedding eval: ${ARC} @ ${BITS}-bit"
    echo "========================================================"
    singularity exec \
        --nv \
        --bind /blue \
        "${SIF}" \
        python3 -u "${SCRIPT_DIR}/input_embed_test_accuracy.py" \
            --dataset          "${DATASET}" \
            --arch             "${ARC}" \
            --quant-bits       "${BITS}" \
            --qmode            QAT \
            --imagenet-root    "${IMAGENET_ROOT}" \
            --models-dir       "${MODELS_DIR}" \
            --results-dir      "${ACCURACY_RESULTS_DIR}" \
            --batch-size       "${BATCH_SIZE}" \
            --workers          "${WORKERS}" \
            --calib-batches    "${CALIB_BATCHES}" \
            --num-threads      "${SLURM_CPUS_PER_TASK:-6}" \
            --auto-fallback \
            --acc-drop-tolerance-pp "${ACC_DROP_TOLERANCE_PP}" \
            --max-skip-fraction     "${MAX_SKIP_FRACTION}"
    echo "[3-Testing] ${ARC} @ ${BITS}-bit done (exit $?)"
done

echo "[3-Testing/run.sh] Array task ${IDX} (${ARC}) complete."
