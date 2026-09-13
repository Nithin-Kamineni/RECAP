# RECAP

Two complementary studies of **embedded error correction for quantized DNN
inference on hardware accelerators**, developed on UF HiPerGator.

Both start from the idea behind the sibling project
[ECC-CODE-Engine](https://github.com/Nithin-Kamineni/ECC-CODE-Engine): store
error-correcting-code (ECC) parity *inside* a quantized value's own
least-significant bits instead of beside it in memory. RECAP asks two follow-up
questions about that idea, one about accuracy and one about energy:

| sub-project | question | tooling |
|---|---|---|
| [`Input_Embedding/`](Input_Embedding/) | If **activations** (the input of every Conv2d/Linear layer) carry a SECDED code in their LSBs, how much ImageNet accuracy does an int8 network lose, and which layers cannot tolerate it? | PyTorch, torchvision, QAT int8, SLURM + Singularity |
| [`Energy_Modeling/`](Energy_Modeling/) | If **weight** parity no longer has to be stored in DRAM, how much inference energy is saved, and does the answer depend on the accelerator dataflow (Eyeriss v1/v2, Simba, WS/OS/IS)? | Timeloop + Accelergy, SLURM + Apptainer |

Each sub-project is self-contained, has its own `README.md`, and is run from
inside its own directory. This file is the map; the details live there.

---

## Repository layout

```
RECAP/
├── README.md                  this file
├── .gitignore  .gitattributes repo-wide rules (LF line endings, caches, images)
│
├── Input_Embedding/           SECDED parity embedded in activation LSBs -> accuracy
│   ├── README.md              what the code does, the code construction, how to run
│   ├── env.sh                 global config (paths, architectures, SECDED parameters)
│   ├── 0-Data/                artifacts: QAT checkpoints*, accuracy JSONs, ImageNet symlink*
│   ├── 1-Quantization/        QAT int8 fine-tuning (resnet18, mobilenet_v2, efficientnet_b0)
│   └── 3-Testing/             the study: SECDED codecs (scalar / vectorized / GPU),
│                              forward-hook embedding, accuracy evaluation with
│                              per-layer fallback, codec tests
│
└── Energy_Modeling/           ECC parity placement -> inference energy, per accelerator
    ├── README.md              the operating manual: one-command flow, knobs, logs, layout
    ├── CLAUDE.md              the detailed reference: architecture contract, caches, how to extend
    ├── run.sh                 the knob file; every ECC_* default lives here
    ├── eccenergy/             the Python package (config, workloads, Timeloop driver,
    │                          ECC arithmetic, result store, plots, experiments, tests)
    ├── archs/                 Timeloop architecture YAMLs + the shared comparison contract
    ├── hpc/                   SLURM job array, Apptainer wrapper, HIPERGATOR.md
    ├── ecc_energy_study/      auto-managed: exercises clone* + mapper cache* + workload shapes
    ├── results*/              raw energies + evaluation JSONs (figures/tables regenerate*)
    ├── docs/RESULTS_SCHEMA.md the evaluation-JSON namespace and document format
    ├── FINDINGS.md  PROJECT_STATUS.md  progress.txt   evidence, summary, status board
    ├── 01..04_*.txt           the original design documents the implementation follows
    └── legacy/                pre-restructure scripts and figures, kept for reference

*  not in git -- see "What is deliberately not in this repository" below
```

---

## Getting started

Both sub-projects assume HiPerGator: SLURM for compute, `module load apptainer`
(or `singularity`) for containers, and the `rewetz` account/QOS. Only
validation, tests and plotting run on a login node; everything else goes
through `sbatch`.

**Input_Embedding** ([its README](Input_Embedding/README.md))

```bash
cd Input_Embedding && source env.sh
(cd 1-Quantization && sbatch run.sh)        # QAT int8 checkpoints, one array task per arch
singularity exec --bind /blue "$SIF" python3 3-Testing/test_secded.py   # verify the codec first
(cd 3-Testing && sbatch run.sh)             # baseline / embedded / fallback accuracy
```

**Energy_Modeling** ([its README](Energy_Modeling/README.md))

```bash
cd Energy_Modeling
module load apptainer
bash hpc/tl.sh bash run.sh validate         # optional 5-second contract check
bash hpc/run_all.sh                         # map (SLURM array), then evaluate + plot
ECC_RECON_ERT_AWARE=1 bash hpc/map_ert_arms.sh
```

---

## What is deliberately not in this repository

The repository holds the code, configuration, documentation and the
*expensive-to-recompute* results. Everything below is a multi-GB build product,
a cache, or an external dataset. Each sub-project README says how to rebuild it.

| path | what it is | how to get it back |
|---|---|---|
| `Energy_Modeling/timeloop.sif` | Apptainer image of `timeloopaccelergy/timeloop-accelergy-pytorch` (1.9 GB) | `apptainer pull`, see `Energy_Modeling/hpc/HIPERGATOR.md` §3 |
| `Energy_Modeling/ecc_energy_study/outputs/` | the Timeloop mapper cache (hours of compute, ~830 MB) | re-map with `sbatch hpc/map.sbatch`; `results*/_raw/` is kept, so evaluation and plotting work without it |
| `Energy_Modeling/ecc_energy_study/timeloop-accelergy-exercises/` | upstream clone of the Timeloop/Accelergy exercises | the package re-clones it on first use |
| `Energy_Modeling/results*/{figures,tables,manifests}/` | plots, CSVs and their provenance | `bash hpc/sweep.sh <axis>` regenerates them from `_raw/` |
| `Input_Embedding/0-Data/imagenet-val` | symlink to the ImageNet-1k validation set | point it at your copy (ECC-CODE-Engine keeps one) |
| `Input_Embedding/0-Data/artifacts/models/**/*.pth` | QAT int8 checkpoints (14–45 MB each) | `cd 1-Quantization && sbatch run.sh` |
| `**/logs/*.out`, `**/logs/*.err` | SLURM job output | rerun the job |

---

## Conventions

* **LF line endings everywhere.** `.gitattributes` enforces it. A CRLF in a
  shell script breaks bash inside the containers;
  `Energy_Modeling/tools-fix-eol.sh` repairs anything that slips through.
* **Never run Timeloop's mapper on a login node.** Use the SLURM scripts in
  `Energy_Modeling/hpc/`.
* **Never delete `Energy_Modeling/ecc_energy_study/outputs/`** on a machine
  that has it. It is the shared mapper cache and is not in git.
* **Results are machine-readable first.** `Energy_Modeling/docs/RESULTS_SCHEMA.md`
  defines the evaluation JSONs; `Input_Embedding/0-Data/artifacts/accuracy_results/`
  holds the accuracy JSONs. Figures are derived from these, never the other way.

---

## Status

Work in progress. `Energy_Modeling/PROJECT_STATUS.md` is the current summary,
`Energy_Modeling/progress.txt` the task-by-task status board and
`Energy_Modeling/FINDINGS.md` the evidence. Read any number quoted in this
repository together with the caveats in those files.
