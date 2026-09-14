# `plans/` — every plan this study has been given, in order

**Nothing here is read programmatically.** No test opens these files and no code
imports them; they are cited BY NAME from docstrings, which is why they keep
their original filenames. The one document that *is* read is `CLAUDE.md`, and it
stays at the project root (`eccenergy/tests/test_code_widths.py` asserts its
PROTECTED width section is still there).

## The live one

**`prompt_7.md`** — modelling TIME and what it does to the energy answer: the
off-chip speed limit, clock gating of the reconstruction engines, component
standby power. Its §9 is the phase plan, §12 the reporting rules that must
travel with every figure, §13 what is fixed and what is deliberately left alone.

`prompt_6.md` is **still in force** for the four RULES that stop the model
double-counting itself. `prompt_3.md`'s constrained mapspace is carried by every
mapper job. `prompt_7.1.md` is a buffer-size sweep to run after prompt_7.

## The sequence

| file | what it asked for | state |
|---|---|---|
| `prompt_1.md` | the first energy comparison | superseded |
| `prompt_2.md` | the width table and the q ladder | **its width table is live** (CLAUDE.md, PROTECTED) |
| `prompt_3.md` | the constrained mapspace | **live** — every mapper job runs under it |
| `prompt_4.md` | — | superseded |
| `prompt_5.md` | the BCH sweep tables | `tools/bch_sweep_tables.py` produces them |
| `prompt_6.md` | placement study; the four anti-double-counting RULES | **RULES still in force** |
| `prompt_7.md` | time, clock gating, standby power | **LIVE** |
| `prompt_7.1.md` | buffer-size sweep | queued, after prompt_7 |
| `prompt_mac_constant.txt` | which per-MAC energy is the denominator | settled: `ECC_MAC_PJ_OVERRIDE=0.23` |
| `ProjectRestructure.md` | making the code hold still while the study grows | phases 0–6 **done**, 7 ongoing |
| `EnvReorganisation.md` | one launcher, one env.sh, any ablation: `run_all.sh` driven by `ECC_APPROACHES` / `ECC_SWEEP` / `ECC_METRICS` | **PLAN** — awaiting answers to its §9 before phase 1 |

## `specs/` — the three source specifications

These are not prompts. They are the documents the MODEL is built against, and
roughly twenty production docstrings cite them by name and section — they are
provenance for modelling decisions, not background reading:

| file | cited from |
|---|---|
| `01_project_context_and_architectures.txt` | `arch/placements.py`, `arch/weight_path.py`, `physics/baseline_dram.py`, `study/baseline.py`, `study/placement_study.py`, `env.sh` |
| `02_reconstruction_dse_and_implementation.txt` | `physics/packing.py` (§16), `physics/granularity.py` (§15), `settings/recon.py`, `toolchain/weight_stats.py` |
| `04_results_storage_spec.txt` | `toolchain/results_store.py`, `paths.py` — and `legacy/docs/RESULTS_SCHEMA.md` is its companion |

**Do not renumber or rename these.** A citation in a docstring is the only link
between a number in a figure and the sentence that asked for it.
