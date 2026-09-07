# RECAP

RECAP applies ECC-CODE-Engine's "embed ECC parity directly into a value's
own LSBs" idea to **activations** instead of weights.

ECC-CODE-Engine embeds multi-bit-error-correcting BCH parity into quantized
**weight** LSBs, because weights sit in DRAM under a comparatively high bit
error rate. RECAP embeds a lighter **SECDED**-class code (single-error
correct, double-error detect) into **activation** LSBs instead — the
network's raw sensor input and every intermediate MAC output that feeds the
next layer — because activations only ever live in on-chip SRAM, where the
bit error rate is much lower and full multi-bit BCH protection is overkill.
Weights are used at their original, unmodified (dequantized int8) precision
in every MAC; only inputs are embedded.

## Layout

```
RECAP/
  env.sh                    global config (paths, archs, SECDED params)
  0-Data/
    imagenet-val/            -> symlink to ECC-CODE-Engine's ImageNet-1k val set
    artifacts/
      models/                 QAT int8 checkpoints (per arch)
      accuracy_results/       final Top-1/Top-5 JSON results
      activation_stats/       (reserved for calibration-scale dumps)
  1-Quantization/            QAT int8 training for resnet18, mobilenet_v2,
                             efficientnet_b0 (copied from ECC-CODE-Engine's
                             generic test-Quantizer.py / qat_train.py — only
                             imagenet-val exists on disk, so QAT fine-tunes
                             and self-evaluates on a stratified 80/20
                             per-class split of it)
  3-Testing/                 the input-embedding study itself
    implementations/
      ParityOverwriteByTopWeightsEncode.py   copied verbatim from
                                              ECC-CODE-Engine for reference
      SECDEDInputEmbed.py    scalar (63,56) SECDED codec — the spec/oracle
      secded_vectorized.py   vectorized/threaded re-implementation of the
                              same codec, used at inference time
    utils/                   convert_to_binary / messageSliceBasedOnChunkSize
                              / reconstruct_numbers_from_chunks — copied
                              verbatim from ECC-CODE-Engine's 4-EmbeddingECC
    activation_embedding.py  forward-pre-hooks: quantize -> SECDED-embed ->
                              dequantize each Conv2d/Linear's input
    input_embed_test_accuracy.py   main eval script (baseline / embedded /
                              fallback), run.sh (SLURM array over archs)
    test_secded.py           codec correctness tests (run before trusting
                              any accuracy number — see below)
```

## The SECDED code

Same 63-bit chunk length and "rank bits by significance, keep the top
positions as message, overwrite the lowest-weight positions with parity"
convention as `ParityOverwriteByTopWeightsEncode.py`. Differs from the
weight pipeline's BCH(63,57) t=1 code (plain Hamming, SEC-only) by
reserving one more LSB slot for an overall XOR parity bit: 56 message + 6
Hamming/SEC parity + 1 overall/DED parity = 63 bits, a shortened extended-
Hamming SECDED code (dmin=4) — no data bit is sacrificed to get double-
error detection. See `implementations/SECDEDInputEmbed.py`'s docstring for
the full derivation.

`secded_vectorized.py` reformulates the same algorithm as batched GF(2)
matrix multiplications (message/parity index layout only depends on chunk
structure, never on data, and repeats with period 8 chunks) instead of a
Python loop per chunk — required to make per-inference activation
embedding tractable at ImageNet scale. Its 8 independent phase-groups run
on a small thread pool for an additional ~2x on multi-core nodes.
`test_secded.py` cross-checks it against the scalar reference bit-for-bit,
plus single/double-bit-flip decode round-trips.

## Running

```bash
# 1) QAT int8 for the 3 architectures (SLURM array, one task/arch)
cd 1-Quantization && sbatch run.sh

# 2) verify the codec before trusting any accuracy number
singularity exec --bind /blue "$SIF" python3 3-Testing/test_secded.py

# 3) input-embedding accuracy evaluation (SLURM array, one task/arch)
cd 3-Testing && sbatch run.sh
```

Results land in `0-Data/artifacts/accuracy_results/imagenet/<arch>/QAT/8-bit/
secded_input_embed_accuracy.json`: `baseline` (int8, unembedded),
`input_embed` (every layer SECDED-embedded, plus per-layer mean distortion),
and — only if embedding drops Top-1 by more than `ACC_DROP_TOLERANCE_PP`
vs baseline — `input_embed_fallback`, where the highest-distortion layers
are exempted from embedding ("regular ECC": protected losslessly instead
of via LSB overwrite) until accuracy recovers or `MAX_SKIP_FRACTION` of
layers is exhausted.

## Caveat

Because only `imagenet-val/` exists on disk (no `train/` split), QAT
fine-tunes on a stratified 40-images/class subset of val and the final
3-Testing evaluation runs on the full 50,000-image val set — so a few
thousand of the evaluation images were also seen during QAT fine-tuning.
This is the same protocol ECC-CODE-Engine's own `qat_train.py` uses (and
matches published PTQ/QAT papers that calibrate/fine-tune on a val
subset). It doesn't affect the *relative* comparison this study cares
about (baseline vs. embedded vs. fallback, all evaluated on the identical
checkpoint and identical image set), but the absolute Top-1 numbers should
be read as slightly optimistic versus a fully held-out test set.
