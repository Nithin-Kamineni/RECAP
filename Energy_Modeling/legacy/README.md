# legacy/ — the pre-restructure code, kept for reference

Nothing in the package imports from here. These are the fourteen scripts the
project used before `eccenergy/` existed, plus every figure and CSV they
produced. Keep them until you are satisfied the new numbers match; then this
directory can be deleted.

`configs/legacy_figures.sh` reproduces the old numbers, and has been verified
field-by-field against `figures/multiarch_bch63_k51_eyeriss_like_summary_uJ.csv`
to double precision.

## Old script → new command

| old script | new command |
|---|---|
| `scripts/run_ecc_multiarch_bch63.py` | `bash configs/proposal_bch63.sh` |
| `scripts/run_ecc_multimodel_n7.py` | `bash configs/n7_hamming.sh` |
| `scripts/run_ecc_multimodel.py` | `ECC_CODE_K=36 ECC_ARCHS=simple_weight_stationary bash run.sh` (experiment `multimodel`) |
| `scripts/run_ecc_multimodel_bch255.py` | `ECC_CODE_N=255 ECC_CODE_K=155 ECC_RECON_PJ=... bash run.sh` |
| `scripts/Run_ecc_multimodel_3way.py` | `bash configs/multimodel_8cnn.sh` |
| `scripts/run_ecc_multimodel_breakdown.py` | `ECC_SPLIT_READ_WRITE=1 bash configs/multimodel_8cnn.sh` |
| `scripts/Run_ecc_singlemodel_ksweep.py` | `bash configs/ksweep.sh` |
| `scripts/run_ecc_singlemodel_archs.py` | `ECC_EXPERIMENT=multiarch bash run.sh` |
| `scripts/run_ecc_singlemodel_archs_with_weakECC.py` | `ECC_WEAK=1 ECC_WEAK_N=63 ECC_WEAK_K=57 bash run.sh` |
| `scripts/run_ecc_study.py` | `ECC_ARCHS=eyeriss_like ECC_MODELS=resnet18 bash run.sh` |
| `scripts/run_llm_breakdown.py` | `bash configs/transformer.sh` |
| `scripts/run_ecc_transformer_block.py` | `ECC_MODELS=gpt2 bash configs/transformer.sh` |
| `scripts/generate_models.py` | `python3 -m eccenergy.generate models` |
| `scripts/generate_transformers.py` | `python3 -m eccenergy.generate transformers` |

`ECC_CODE_N=255` needs a reconstruction energy for that geometry —
`data/dc/BCH_N63_results.json` only covers N=63. Pass `ECC_RECON_PJ=<value>`
explicitly, or the loader falls back to the N=63 constants and says so.

## What changed in the numbers, and what did not

Identical: every energy total, every savings percentage, every DRAM/compute
figure, for every architecture and model.

Two deliberate differences, both opt-out:

1. **`ECC_CLASSIFY=instances` is now the default.** The old classifier assigned a
   storage level to "Global buffer" if its name contained `buffer`, which put
   33 mJ of Simba's *per-PE* `PEWeightBuffer` / `PEAccuBuffer` / `PEInputBuffer`
   into the global-buffer category. Totals are unchanged either way; only the
   split between the two on-chip categories moves. `ECC_CLASSIFY=name` restores
   the old behaviour.

2. **The embedded arm's decode count.** The two-arm scripts charged the embedded
   arm one decode per byte (`EMB_WEIGHTS_PER_CW = 8`) while the three-arm
   scripts charged it per codeword. The package defaults to per codeword —
   the shared-geometry model, which is what the three arms are meant to compare.
   Set `ECC_EMB_WEIGHTS_PER_CW=8` for the old two-arm behaviour. Visible only
   when `ECC_DECODE=1`, which is off by default.

## Bugs the restructure fixed

- **`globals.yaml` said 65nm while every arch.yaml said 45nm.** DRAM sits above
  the accelerator container, so it took the node from `globals.yaml` — i.e. the
  figures titled "45nm" costed DRAM at 65nm. Consistent across architectures, so
  relative comparisons held, but the label was wrong. `globals.yaml` now follows
  `ECC_FORCE_TECHNOLOGY`, and the figure title only claims a node when one is
  actually forced.
- **`ecc_energy_study/` was resolved from `cwd`.** Launching from the wrong
  directory created a second empty working directory, re-cloned the exercises
  repo and re-ran the whole mapper from scratch. The original `readme.txt` opened
  with a warning about this. The package resolves it from the project root, so
  the failure mode no longer exists.
- **The read/write split silently rescaled cached results** when a raw record
  written in one category layout was read back in the other. Now the raw cache
  is keyed by layout and a mismatched record is refused rather than
  partially read.

See `FINDINGS.md` in this directory for the architecture-level confounds as they
were understood before the 2026-09-06 rewrite. Archived; numbers are historical.
