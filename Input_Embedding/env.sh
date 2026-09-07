#!/usr/bin/env bash
# =============================================================================
# Global environment for RECAP
#
# Source this from every run.sh:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../env.sh"
#
# RECAP applies the ECC-CODE-Engine "embed parity into the LSBs" idea to
# ACTIVATIONS (the inputs of each layer — the initial sensor input and every
# intermediate MAC output feeding the next layer) instead of weights. Weights
# are used at full (dequantized) precision in the MAC, unmodified. Because
# activations only ever live in on-chip SRAM (not DRAM like weights), a
# lighter SECDED-class code (single-error-correct, double-error-detect) is
# enough instead of the multi-bit-t BCH codes used for weights.
# =============================================================================

SIF="/blue/rewetz/vkamineni/RECC_MIP_v15.sif"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${PROJECT_ROOT}/0-Data"
ARTIFACTS_DIR="${DATA_ROOT}/artifacts"
MODELS_DIR="${ARTIFACTS_DIR}/models"
ACCURACY_RESULTS_DIR="${ARTIFACTS_DIR}/accuracy_results"
ACTIVATION_STATS_DIR="${ARTIFACTS_DIR}/activation_stats"

# ImageNet-1k validation set (symlinked once to the physical copy under
# ECC-CODE-Engine/0-Data/imagenet-val — same 50,000 images, 1000 classes).
IMAGENET_ROOT="${DATA_ROOT}/imagenet-val"

DATASET="${DATASET:-IMAGENET}"

# ---- Architectures for the study ----
ARCHS="${ARCHS:-resnet18 mobilenet_v2 efficientnet_b0}"

# ---- Quantization ----
QUANTIZE_BITS="${QUANTIZE_BITS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
WORKERS="${WORKERS:-5}"

# QAT fine-tuning (only imagenet-val exists on disk — no train/ split — so
# qat_train.py uses a stratified 80/20 per-class split of val: 40 img/class
# fine-tune, 10 img/class held out).
QAT_EPOCHS="${QAT_EPOCHS:-15}"
QAT_LR="${QAT_LR:-1e-4}"
QAT_SKIP_FIRST_LAST="${QAT_SKIP_FIRST_LAST:-0}"

# ---- SECDED input-embedding parameters (3-Testing) ----
# Codeword length matches ECC-CODE-Engine's weight-embedding mother code (63,
# reusing the same chunk/bit-weight-ordering utilities). message_size=56
# (rather than the weight pipeline's 57 for pure t=1 SEC) reserves one extra
# LSB slot per chunk for an overall XOR parity bit, upgrading plain Hamming
# SEC into true SECDED (dmin=4): 56 message + 6 Hamming parity + 1 overall
# parity = 63 bits, so no data bit is ever sacrificed.
CODEWORD="${CODEWORD:-63}"
SECDED_MESSAGE_SIZE="${SECDED_MESSAGE_SIZE:-56}"

# Number of calibration batches used to fix each layer's activation
# int8 quantization scale (simple max-abs / MinMax observer) before the
# embedding pass runs.
CALIB_BATCHES="${CALIB_BATCHES:-20}"

# Accuracy-recovery fallback: if embedding drops top-1 by more than this many
# percentage points vs the un-embedded int8 baseline, progressively exempt
# ("regular ECC" = no LSB distortion) the highest-distortion layers instead
# of embedding them, re-checking on a fast subset until accuracy recovers or
# the skip budget is exhausted.
ACC_DROP_TOLERANCE_PP="${ACC_DROP_TOLERANCE_PP:-2.0}"
MAX_SKIP_FRACTION="${MAX_SKIP_FRACTION:-0.3}"
