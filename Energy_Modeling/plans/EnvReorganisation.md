# EnvReorganisation — one command, one file, any ablation

**Status: PLAN, not started — ALL QUESTIONS ANSWERED, ready for phase 0.** Written
2026-09-13, revised 2026-09-14 after the user answered both rounds of questions.
Nothing in this document has been implemented. Section 6 lists the issues found
while auditing the request against the code and how each was decided; **section 9
records every decision.** Two env.sh knobs were already changed ahead of the plan
on 2026-09-14 at the user's request: `ECC_STATIC_ENERGY=1` and
`ECC_KS="39 57 45 30"` (first entry = the held code).

---

## 0. How a new session should read this

1. `CLAUDE.md` first — especially the two **PROTECTED** sections (THE WIDTH TABLE,
   and EVERY MAPPER CACHE IS COLD). Both are touched by this plan.
2. `GUARDS.md` — every guard has an id and a tier. Several guards named below are
   retired or re-tiered by this plan; `make guards` regenerates the table.
3. `plans/ProjectRestructure.md` §9.1 — the gate. `restructure/` was deleted on
   2026-09-13 after phase 7 (last present at commit `46a8df1`); **this plan
   wants it back** (section 7, phase 0 has the exact command).
4. Then this file.

The user's own words are in section 3, edited only for spelling and clarity.
Everything else is the audit that was done against the code on 2026-09-13.

---

## 1. The aim

> My aim is to run `bash hpc/run_all.sh`, changing only `env.sh` variables, and
> get any of my ablation studies. To achieve this I am going to ask for some big
> changes in how the `env.sh` variables are controlled and what state they are in.

Today that is not true. The placement study needs its own launcher
(`ECC_RECON_ERT_AWARE=1 bash hpc/map_ert_arms.sh`), the depth sweep needs another
(`hpc/map_depth_sweep.sh`), and `run_all.sh` can only fan out over
(architecture × model). After this plan there is **one launcher**, and what it
maps, evaluates and plots is decided by **four variables**.

---

## 2. The target, as a picture

### 2.1 Before — three launchers, three fan-outs

```
   env.sh ─┬─► hpc/run_all.sh ─────────► task file: (arch × model)
           │        └─ map.sbatch ×N ──► solves ONE arm (ECC_RECON_ERT_ARM from env)
           │        └─ run_all --eval-only ──► ECC_EVAL_EXPERIMENTS loop, then _plot
           │
           ├─► hpc/map_ert_arms.sh ────► task: (arm × shape), 6 × 12 = 72 jobs
           │        └─ eval job with ECC_RECON_ERT_AWARE=1 ─► placement figure
           │
           └─► hpc/map_depth_sweep.sh ─► task: (depth × datawidth × shape) + a gate
```

### 2.2 After — one launcher, one fan-out

```
   env.sh ──► hpc/run_all.sh
                 │
                 │  ECC_APPROACHES   which BARS       baseline embedded recon1 … recon5
                 │  ECC_SWEEP        which X AXIS     bch | model | arch | area | fix   (area = buffer DEPTH)
                 │  ECC_METRICS      which Y AXES     energy | edp | latency | area
                 │  ECC_ARCHS / ECC_MODELS / ECC_KS / ECC_DEPTH_SWEEP_SCALES   the lists
                 │  ECC_RECON_DEFAULT  which placement `recon` means          recon2
                 │  ECC_LAYERS         one layer, or empty for the model      (testing)
                 │  ECC_JOBS           bundle the units into N SLURM jobs     (optional)
                 │
                 ├─► 1. task file:  (arch, model, K, depth, ARM)  ── one row per CHIP
                 │        ARM = reference for baseline/embedded, reconN for each placement
                 │        ONLY the arms in ECC_APPROACHES that THIS design declares
                 │
                 ├─► 2. map.sbatch × rows ─► each job solves ONE chip into its own cache
                 │
                 └─► 3. eval job (afterok) ─► one figure:
                          rows    = ECC_METRICS   (energy / edp / latency / area)
                          columns = ECC_SWEEP     (the x axis)
                          bars    = ECC_APPROACHES (per design, missing ones warned)
```

The word **CHIP** is the organising idea. Every distinct thing the mapper is
handed — an architecture, at a code, at a depth, in an arm — is one chip with one
cache directory keyed by `arch_fingerprint()`. Section 2.1's three launchers
exist only because they enumerate chips along different axes. Section 2.2 has one
enumerator.

---

## 3. The requested changes (the user's words)

*Edited for spelling and clarity only. Audit notes are marked* ► *and are not
part of the request.*

### 3.1 The control surface

Honestly, `ECC_EVAL_EXPERIMENTS` seems outdated and mixes up things that are not
related. Currently it says which evaluations to run; I will delegate this to
`ECC_APPROACHES` and `ECC_SWEEP`. From now on there will not be different
variables for plotting, mapping and evaluation — those would be the same
variables, to remove redundancy.

**`ECC_APPROACHES`** will list `recon1 recon2 recon3 recon4 recon5` in addition to
`baseline` and `embedded`.

**`ECC_SWEEP`** can do `bch`, `model`, `arch`, `fix`, `area`.
► *`area` sweeps buffer DEPTH (`ECC_DEPTH_SWEEP_SCALES`); the user keeps the name
`area` for simplicity (decided 2026-09-14). Note the same word is also a value of
`ECC_METRICS` — see 6.3.*
The `area` axis uses:

| variable | role in the depth sweep |
|---|---|
| `ECC_WEIGHT_DEPTH_SCALE` | the held value when depth is FIXED (any other axis) |
| `ECC_WEIGHT_DEPTH_LEVELS` | which levels the scale applies to |
| `ECC_DEPTH_SWEEP_SCALES` | the values to sweep over |

`ECC_APPROACHES` must check whether the architecture being plotted has each
named placement. If it does not, plot the placements that exist and skip the
ones that do not — **warn in the log, do not error.** Example: if one arch has
`recon1 recon2 recon3` and another has `recon1 … recon5`, and `ECC_APPROACHES`
names all five, then the first arch shows three bars and the second shows five.

**`ECC_SWEEP=fix`** plots exactly what `ECC_APPROACHES` names, at the fixed model,
fixed arch and fixed code. Fixed values are the **first entries** of
`ECC_MODELS`, `ECC_KS`, `ECC_ARCHS`.

► *This is the existing `ECC_CONST_*` rule from env.sh §10 — first item of the
list is the held value. Nothing new is needed for the "fixed" half.*

**The generic `recon` bar means `recon2`.** A new variable — proposed name
**`ECC_RECON_DEFAULT`**, default `recon2` — says which placement the word `recon`
stands for wherever a figure wants a single reconstruction bar. ► *This is the
resolution of issue 6.2; the old abstract `recon` arm is retired.*

**One layer for testing.** `ECC_LAYERS` names one layer (or several) of the fixed
model; empty means the whole model. ► *Today there are two such knobs,
`ECC_LAYERS` (§1) and `ECC_RECON_LAYER` (§4, `all` = whole model), and §10 copies
one into the other. With `ECC_RECON_MODELING` gone only `ECC_LAYERS` is needed.*

**Parallel jobs.** A variable — proposed name **`ECC_JOBS`** — for how many SLURM
jobs to split the work into. If it is `12` and there are 72 units of work, each
job runs 6 of them. ► *See 6.10 for the trade-off and how it sits beside
`ECC_CONCURRENCY`.*

**A new variable for the plotted metric** — proposed name **`ECC_METRICS`** — takes
a list from `energy edp latency area`. With several values, e.g.
`ECC_METRICS="energy latency area"`, the figure has one **row** per metric,
stacked vertically: energy on top, latency below it, area below that.

### 3.2 Variables to remove, keep, or move

`ECC_RECON_MODELING` can be removed — `ECC_APPROACHES` naming `recon1 … recon5`
is the mode now.

`RECON_OPTIMIZER` and `ECC_PHASE` — remove. We map every placement from now on
and will never run `RECON_OPTIMIZER=False`, so they are always `True` and `Post`.
► *DECIDED 2026-09-14: `RECON_OPTIMIZER` goes. `ECC_PHASE` is not deleted but
**derived per arm** — `Pre` for baseline/embedded, `Post` for every placement —
because one run now spans both. See 6.1.*

Keep `ECC_RECON_ERT_AWARE`.

`ECC_RECON_ERT_ARM` is not necessary as a knob: the placement ablation is now
`recon1 … recon5` in `ECC_APPROACHES`, and both mapper and evaluator must solve
each of them individually anyway. *(Does this make sense? Of all the variables
this is the most confusing one and I am not sure about it. Please make sure it
works, confirm with me if it does, and ask if it does not.)*
► *Confirmed in section 5, with one condition.*

`ECC_RECON_PACKING` — keep; it prices DRAM and NoC. **Set to `stream`, leave it.**

`ECC_RECON_REQUIRE_GROUP_RESIDENCY` — always 0; remove.
`ECC_RECON_ONCHIP_FRACTION` — remove. ► *Confirmed dead: nothing reads it.*
`ECC_RECON_PLACEMENT_CHARGES_DECODE` — remove; `ECC_DECODE` already handles it.
`ECC_RECON_DECODE_SITE` — always `ondie`, never `controller`. Remove the
variable **and the controller code path**, everywhere, including the frozen
`physics/parity.py`. ► *DECIDED 2026-09-14; see 6.7.*

Keep `ECC_RECON_ENCODER_SITE`, `ECC_LATENCY_MODEL`.
Keep `ECC_RECON_BW_SCALE`, `ECC_ONCHIP_BW_BITAWARE` as they are — their current
values already favour recon.

`ECC_LEAKAGE_NW` is per design and belongs in the design's directory, exactly
like `design.yaml`, `weight_path.yaml` and `placements.yaml`.
`ECC_STATIC_ENERGY` — **keep the switch** (decided 2026-09-14): `0` = standby is
not charged and is not even a category, `1` = standby is charged to all three
arms. Only the DATA moves to the design YAML. See 6.4.

Keep `ECC_WEIGHT_BITS`, `ECC_ACTIVATION_BITS`.
Keep `ECC_FORCE_DATAWIDTH`, `ECC_FORCE_TECHNOLOGY`, empty.

`ECC_WEIGHT_CAPACITY_SCALE`, `ECC_WEIGHT_CAPACITY_SCOPE` — remove. I do not need
Task 4's N/K memory scaling: quantising the data already accounts for it.
► *DECIDED 2026-09-14: Task 4's capacity dilation is retired (6.5). Its FINDINGS
entries stay as history.*

**`ECC_WEIGHT_DATAWIDTH` and `ECC_WEIGHT_WIDTH_GLB_MULT`.** Currently there is one
datawidth in the arch YAML and, for recon, the code overrides both the
quantisation and the word width per BCH code. I want a **separate YAML file** that
records, for each BCH quantisation, the word width `filter_glb` and the spad
should have — `96 at q=8, 98 at q=7, 96 at q=6, 95 at q=5, 96 at q=4` — so that
both the quantisation and the widths come from data. Make sure no test or guard
objects that word widths differ across arms — embedded always uses 8-bit
quantisation, so its widths *will* differ. With that file,
`ECC_WEIGHT_DATAWIDTH`, `ECC_WEIGHT_DATAWIDTH_LEVELS` and
`ECC_WEIGHT_WIDTH_GLB_MULT` can go.
► *Good news in issue 6.6: the pair-geometry check already ignores width.*

Keep `ECC_DISABLE_ASSERT_PAIR_GEOMETRY`.

`ECC_WEIGHT_DEPTH_SCALE`, `ECC_WEIGHT_DEPTH_LEVELS`, `ECC_DEPTH_SWEEP_SCALES` are
the depth-sweep variables. With `ECC_SWEEP=depth`, bch, model and arch are held
and the run sweeps the scaled depths. **Move these up to the top of env.sh beside
`ECC_APPROACHES`** — they now say what to compute and plot.

Remove `ECC_DEPTH_SWEEP_RECON_DW` — it is empty and the empty path derives `q`.

Move `ECC_DRAM_DEPTH` to `archs/_shared/`, beside `standard.yaml` — it is shared
by every design.

`ECC_MAC_PJ_OVERRIDE` is fine.

### 3.3 The shape of the new env.sh

The current file is so full of comment lines that I cannot see where to change a
value. In the new version, at least the important variables get **one comment
line** listing their options, like:

```bash
# resnet18 | mobilenet_v2 | resnet50 | efficientnet_b0 | …
: "${ECC_MODELS:=resnet18 mobilenet_v2}"
```

Any architecture-related fact — placements, leakage, clock, and so on — must live
in that architecture's folder under `archs/`, not in code and not in env.sh, so
that adding a new architecture is easy. Placements must come from
`archs/<name>/placements.yaml` (I think this already works; make sure it does).
► *It does — `arch/placements.py:118` reads that file, and no Python names a
design. What still lives in env.sh is the per-design `ECC_RECON_PLACEMENTS`,
`ECC_ARCH_CLOCK_MHZ` and `ECC_LEAKAGE_NW` tables — section 4.3.*

Test the code thoroughly with these variables plugged in.

---

## 4. The variable ledger

### 4.1 What the user turns

Every knob that survives is one of four kinds. The new `env.sh` should be ordered
this way, with the section that colds the mapper cache walled off and marked.

| kind | variables | colds the mapper cache? |
|---|---|---|
| **what to compute & plot** | `ECC_APPROACHES` `ECC_RECON_DEFAULT` `ECC_SWEEP` `ECC_METRICS` `ECC_ARCHS` `ECC_MODELS` `ECC_KS` `ECC_CODE_N` `ECC_DEPTH_SWEEP_SCALES` `ECC_WEIGHT_DEPTH_SCALE` `ECC_WEIGHT_DEPTH_LEVELS` `ECC_LAYERS` `ECC_JOBS` | the lists: yes (they *are* the chips); `ECC_JOBS`: no |
| **the mapper** | `ECC_MAPPER_ALGORITHM` `ECC_VICTORY` `ECC_VICTORY_SCALING` `ECC_MAPPER_TIMEOUT` `ECC_MAPPER_MAX_PERMUTATIONS` `ECC_OPT_METRIC` `ECC_MAPPER_SEED` `ECC_MAPPER_THREADS` `ECC_RERUN_OPTIMISER` | **yes** |
| **the chip** | `ECC_WEIGHT_BITS` `ECC_ACTIVATION_BITS` `ECC_ACC_BITS` `ECC_ARCH_FIDELITY` `ECC_FORCE_DATAWIDTH` `ECC_FORCE_TECHNOLOGY` `ECC_WEIGHT_FACTOR_RELAX` `ECC_MAPSPACE_CONSTRAIN` `ECC_MAC_PJ_OVERRIDE` `ECC_GLOBAL_CYCLE_SECONDS` `ECC_NOC*` `ECC_RECON_BW_SCALE` `ECC_ONCHIP_BW_BITAWARE` `ECC_ENERGY_MODEL_REV` | **yes** |
| **the prices (evaluator only)** | `ECC_STATIC_ENERGY` `ECC_DRAM_PJ_PER_BIT` `ECC_BASELINE_DRAM_PJ_PER_BIT` `ECC_DRAM_BACKGROUND_PJ` `ECC_DRAM_REFRESH_PJ` `ECC_LATENCY_MODEL` `ECC_RECON_PACKING` `ECC_RECON_ENCODER_GRANULARITY` `ECC_RECON_ENCODER_SITE` `ECC_RECON_ERT_AWARE` `ECC_DECODE*` `ECC_RECON_JSON` `ECC_RECON_PJ` `ECC_RECON_*_FALLBACK_PJ` `ECC_RECON_CLOCK_GATING_PCT` `ECC_PARITY_*` `ECC_CLASSIFY` `ECC_SPLIT_READ_WRITE` `ECC_DISABLE_ASSERT_PAIR_GEOMETRY` | **no** — re-priced from cache in milliseconds |

Plus the unchanged cluster (§7), output (§8) and misc (§9) sections.

### 4.2 Removed from env.sh

| variable | fate | why | where its job goes |
|---|---|---|---|
| `ECC_EVAL_EXPERIMENTS` | **delete** | mixed sub-commands with arms; already overwritten by §10 in recon mode | `ECC_APPROACHES` + `ECC_SWEEP` |
| `ECC_RECON_MODELING` | **delete** | the mode is implied by `reconN` in `ECC_APPROACHES` | — |
| `RECON_OPTIMIZER` | **delete** | always True | constant in code |
| `ECC_PHASE` | **delete as a knob** (decided) | **derived per arm** — see 6.1 | `Pre` for baseline/embedded, `Post` for reconN |
| `ECC_RECON_ERT_ARM` | **delete as a knob, keep as the per-job parameter** | see section 5 | column 5 of the task file, set by `run_all.sh` |
| `ECC_RECON_PLACEMENTS` (bash array) | **delete** | redundant with `ECC_APPROACHES` ∩ `placements.yaml` | `archs/<name>/placements.yaml` |
| `ECC_RECON_REQUIRE_GROUP_RESIDENCY` | **delete** | settled decision (0) | constant in `placement_eval.py`, reasoning kept as a comment |
| `ECC_RECON_ONCHIP_FRACTION` | **delete** | dead — no read site | — |
| `ECC_RECON_PLACEMENT_CHARGES_DECODE` | **delete** | folded into `ECC_DECODE` | — |
| `ECC_RECON_DECODE_SITE` | **delete + remove `controller` code** (decided, incl. `parity.py`) | always `ondie` | see 6.7 |
| `ECC_RECON_STEM` | **delete** | a filename | constant `ReconSweep` |
| `ECC_RECON_LAYER` | **delete** | duplicate of `ECC_LAYERS` once `RECON_MODELING` goes | `ECC_LAYERS` (empty = whole model) |
| the abstract `recon` arm (`build_stacks` column 3) | **retire** | rule R-6: not a placement — see 6.2 | `ECC_RECON_DEFAULT=recon2` |
| `ECC_LEAKAGE_NW` (bash array) | **move** | per design | `archs/<name>/design.yaml` |
| `ECC_ARCH_CLOCK_MHZ` (bash array) | **move** | per design | `archs/<name>/design.yaml` |
| `ECC_WEIGHT_CAPACITY_SCALE` / `_SCOPE` | **delete** (decided) | Task 4's dilation retired — see 6.5 | — |
| `ECC_WEIGHT_DATAWIDTH` / `_LEVELS` | **delete** | derived from the arm + the width table | the width YAML — see 6.6 |
| `ECC_WEIGHT_WIDTH_GLB_MULT` | **move** | per design (Eyeriss v1's GLB word is 4× the spad's) | the width YAML |
| `ECC_DEPTH_SWEEP_RECON_DW` | **delete** | empty; the derivation is the single source | `widths.declared_datawidth()` |
| `ECC_DEPTH_SWEEP_GATE_VICTORIES` / `_GATE_SCALES` | **move** | arguments to one script | defaults inside `hpc/map_depth_sweep.sh` (or its successor) |
| `ECC_DRAM_DEPTH` | **move** | shared by every design | `archs/_shared/standard.yaml` |

### 4.3 What lands in `archs/`

```
archs/
  _shared/
    standard.yaml        + dram_depth: 1048576
  eyeriss_like_wglb/
    widths.yaml          NEW  q -> {spad_width, glb_width}  (THE WIDTH TABLE, as data,
                              one file PER DESIGN -- decided 2026-09-14)
    design.yaml          + clock_mhz: 200
                         + leakage_nw: {sram_bit: …, rf_bit: …, mac_instance: …}
    placements.yaml      (already the source of truth for recon1 … recon5)
```

**`widths.yaml` is PER DESIGN** (decided 2026-09-14, question 9.4): each
architecture carries its own `q -> {spad_width, glb_width}` table. The three
in-scope designs start from the same values (THE WIDTH TABLE, 96/98/96/95/96)
and may diverge later. `physics/widths.py` keeps the RULE (`nearest_multiple`)
as the fallback for a design whose file does not list a `q`.

---

## 5. `ECC_RECON_ERT_ARM` — confirmed, with one condition

**Yes, the request makes sense, and it works — provided the variable survives as
an internal per-job parameter.** It cannot simply be deleted, because it is the
only recon knob in the mapper fingerprint (`settings/recon.py:46`): it is what
gives each boundary its own cache directory.

```
reference  fp=2db6a4d92ff5     recon3  fp=04320e626edd
recon1     fp=410ee95e7f78     recon4  fp=653c73722712
recon2     fp=1519e6934e32     recon5  fp=32154e84bb0f
```

What changes is **who sets it**:

| | today | after |
|---|---|---|
| map job | `map_ert_arms.sh` sets it per job via `--export` | `run_all.sh` writes it as **column 5 of the task file**; `map.sbatch` exports it from that row |
| eval job | launched with `reference`; the evaluator walks all arms itself (`ert_view.py:214`: `cfg.with_(recon_ert_arm=arm.key)`) | **unchanged** |
| the user | leaves it empty in env.sh | it is **not in env.sh** |

So the evaluator half already works exactly as the user describes — it solves
each placement individually because it clones the config per arm. Only the map
launcher needs the third axis. `ECC_RECON_ERT_AWARE=1` stays as the switch that
makes the evaluator walk the arms; with it off, every bar would be read from the
reference plan, which is Task 3 and must stay labelled as such.

---

## 6. Issues found in the audit

Ordered by how much they change the request. Each ends with what is needed
from the user.

### 6.1 `ECC_PHASE` is per arm, not per run — so it cannot be "always Post"

Baseline and embedded are **`Pre` results by construction**: there is no reduced
weight representation, so no mapping could have been optimised for one. The
frozen `study/baseline.py:132` refuses `ECC_PHASE != Pre` outright, and the
results namespace is `results/evaluation/{Pre|Post}/{ARCH}/{MODEL}/…`
(`04_results_storage_spec.txt`, `paths.py:134`).

Recon placements with their own mapping are **`Post`**.

So a single run with `ECC_APPROACHES="baseline embedded recon1 … recon5"`
**spans both phases.** The knob cannot go to a constant; it has to become
**derived per arm**: `Pre` for baseline/embedded, `Post` for reconN. That is a
code change in `paths.evaluation_dir()` and in `results_store`, and it touches
a frozen file's guard.

→ **DECIDED 2026-09-14:** `ECC_PHASE` becomes derived per arm. The guard in the
frozen `study/baseline.py` is edited to match (the file is unfrozen for exactly
that hunk, and the change is gated).

### 6.2 The `recon` bar you see today is not any of recon1–5 — and the sweep figures can't draw those yet

**What the `recon` bar is today.** Open any bch / model / arch sweep figure. It has
three bars per group: `baseline`, `embedded`, `recon`. That third bar is drawn by
`build_stacks()` and it is an **abstraction**: it pretends one reconstruction engine
sits at the chip entrance and then applies the K/N saving to **every on-chip level**
of **every design** identically. No real boundary does that — once weights are
reconstructed at the entrance they are full width again on chip, so a real recon1
saves nothing on chip, and a real recon5 saves on chip but pays for engines in every
PE. The abstract bar is an optimistic upper bound, and reporting rule R-6 exists to
stop anyone quoting it as a placement.

**What recon1–5 are.** Real boundaries, defined per design in `placements.yaml`,
each **with its own mapping** (its own chip, its own cache directory) and its own
engine count. They are drawn today only by the placement figure
(`study/placement_study.py` → `report/recon_view.py`), never by the sweep figures.

**The issue, in one line:** the sweep figures and the placement figure are **two
separate programs** that read **two different kinds of data**. Putting `recon1 …
recon5` into `ECC_APPROACHES` asks the sweep figures to draw bars they have no code
for, from mappings they do not know how to find.

```
   TODAY                                          AFTER
   sweep.py / panels.py      placement_study.py   ONE figure builder
   ├ baseline                ├ reference           ├ baseline
   ├ embedded                ├ recon1  (own chip)  ├ embedded
   └ recon  (abstract,       ├ recon2  (own chip)  ├ recon1 … recon5  (own chips,
      no chip of its own)    ├ …                   │   looked up per (design, K, depth))
                             └ recon5  (own chip)  └ `recon` = ECC_RECON_DEFAULT (recon2)
```

**What the user's decision solves, and what it does not.**

* `ECC_RECON_DEFAULT=recon2` **retires the abstract bar** and answers "which
  placement is *the* reconstruction bar when a figure wants one". That is the right
  call: every recon bar on every figure is now a real, mapped chip. Question 9.2 is
  closed by it.
* It does **not** remove the code work. `recon2` still comes from the placement
  path, not from `build_stacks()`. So the sweep code must still learn to: enumerate
  the arms a design declares, look up each arm's own mapping **per swept point**
  (a bch sweep at six codes needs recon2 mapped at all six), and draw a bar per
  (design, placement). That is phase 6 and it is the single largest change here.
  **The user confirmed this change on 2026-09-14.**

**FINDINGS.** Every number built on the abstract `recon` arm (the three sweep
figures, and the "recon+ total" lines in the run reports) stops being reproducible
once it is retired. They stay in FINDINGS as history, labelled as the abstract
arm; the new figures do not overwrite them.

### 6.3 `area` is not computed today; `ECC_SWEEP=area` names the wrong thing

Two separate points.

**The axis.** `ECC_SWEEP=area` sweeps `ECC_DEPTH_SWEEP_SCALES` — buffer **depth**.
Nothing about silicon area is swept. **The user keeps the name `area`** (decided
2026-09-14, "to make it simple"). One consequence to live with: `area` then
means two things in adjacent variables — an x axis in `ECC_SWEEP`, a y axis in
`ECC_METRICS`. The one-line comment on each knob must say which.

**The metric.** `energy`, `edp` and `latency` exist: `OPT_METRICS` has
energy/edp/delay, the raw record carries `cycles`, and `ECC_LATENCY_MODEL=1`
re-times the plan. **`area` does not.** Accelergy writes an ART
(`timeloop-mapper.ART.yaml`) into every cache entry, but nothing sums it into a
per-bar area, and reconstruction adds an engine whose area would come from the
Design Compiler report (`data/dc/`), not from Accelergy. `area` in `ECC_METRICS`
is therefore a **new feature**, scoped in phase 5.

→ **Needed (still open, 9.7):** is the `area` METRIC wanted enough to build? It is
the only metric with no existing source.

### 6.4 `ECC_STATIC_ENERGY` and `ECC_LEAKAGE_NW` are two different knobs

`ECC_LEAKAGE_NW` is the **data** (per-bit and per-MAC leakage densities, per
design). `ECC_STATIC_ENERGY` is the **switch** that decides whether standby is a
charged category at all — with it at 0, `Standby` is not even in `phys_cats()`
(`study/energy.py:56`), which is what lets every pre-Phase-A total still
reproduce to the pJ.

Moving the data into `design.yaml` is right. But deleting the switch means
"always on", which **changes every total in the study** and every recorded
comparison of Phase A vs before. That may be exactly what the user wants — it is
a modelling decision, not a tidy-up — and it must be taken as one.

→ **DECIDED 2026-09-14: (a).** The switch stays (`0` off, `1` on); only the
`ECC_LEAKAGE_NW` data moves into `design.yaml`. No number moves.

### 6.5 Capacity scale and depth scale are different experiments

Both rewrite `depth:` on the weight levels, which is why they look redundant.
They are not (`arch/patch.py:219` and `:543`):

| | `ECC_WEIGHT_CAPACITY_SCALE` | `ECC_WEIGHT_DEPTH_SCALE` |
|---|---|---|
| the story | the *same* array holds N/K more weights because they are narrower | a *genuinely smaller* array |
| energy | Accelergy over-prices the "bigger" array; `capacity_dilation_correction()` **undoes** it | cheaper per-access is **real** and kept |
| the experiment | **Task 4** — does the extra room buy fewer DRAM refetches? | **prompt_2** — what depth makes recon win most? |
| cache slug | `wcap<s>` | `wdepth<s>` |

Removing the capacity knobs retires Task 4's dilation study. That may be
intended — the user wants the depth sweep instead — but it is a capability
removal, and FINDINGS carries Task 4 results.

→ **DECIDED 2026-09-14: retired.** The user quantises the data directly, which is
what the dilation was standing in for. `wcap<s>` cache directories on disk are
left alone; nothing reads them.

### 6.6 The width YAML — good news, and one PROTECTED section

**The worry about tests is already handled.** `assert_pair_geometry()` checks
**DEPTH ONLY**, deliberately (`arch/patch.py:1225`): *"only `width:` (from THE
WIDTH TABLE, per arm) and `datawidth:` differ"*. CLAUDE.md's PROTECTED section
says the same — *"95 % 8 is IRRELEVANT"*. Widths differing across arms is the
design, and no guard objects.

Two things to know before moving the table:

1. **THE WIDTH TABLE is PROTECTED** in CLAUDE.md — the rule has been re-derived
   wrongly in five sessions. Moving it to YAML is fine; **changing a value is
   not**, and the move must show byte-identical fingerprints before and after.
2. The table today is **keyed by `q`, not by design** (`physics/widths.py:99`).
   `declared_width(q)` reads the table first, else falls back to
   `nearest_multiple(q, base_width)`. The user has decided (9.4) the table is
   **per design** anyway — each `archs/<name>/widths.yaml` carries its own — with
   the rule kept in `physics/widths.py` as the fallback for an unlisted `q`.

### 6.7 Removing `controller` touches a frozen file

`ECC_RECON_DECODE_SITE=controller` is referenced in **23 places across 10 files**,
including `physics/parity.py` (**FROZEN**) and `physics/baseline_dram.py`. The
`ondie` model has been the study's since 2026-09-09 and `controller` is kept
only as the runnable diff behind that decision.

→ **DECIDED 2026-09-14: remove everywhere, `physics/parity.py` included.** That
file is unfrozen for this one removal, and the change is gated: with
`ondie` the default, every number must be byte-identical before and after.

### 6.8 `ECC_RECON_PLACEMENTS` becomes redundant — and three designs have no placements

Once `ECC_APPROACHES` names the placements and `placements.yaml` defines them,
the env.sh bash array `ECC_RECON_PLACEMENTS` is a third copy of the same list.
Delete it; the intersection of the two is what draws.

Note `simba_like`, `simple_input_stationary` and `simple_output_stationary`
have **no `placements.yaml`**. The "warn, don't error" rule must cover "this
design declares no boundaries at all" as well as "not this particular one".

**Scope (2026-09-14).** The user will run three designs: `eyeriss_like_wglb`
(Eyeriss v1, 5 placements — the only one known to work end to end),
`simple_weight_stationary` (5 placements) and **`eyeriss_v2_like_wglb`** (4
placements — this, not `eyeriss_v2_like`, is what the user means by "Eyeriss v2";
recorded in CLAUDE.md 2026-09-14). All three declare a `placements.yaml` and all
three pass `placements_for` / `mapper_arms` today. The latter two are **expected
to surface errors** further down the pipeline that have not been solved yet; that
debugging is part of phase 6, not a surprise. Two facts about the v2 choice to
carry into every figure label: its README calls it *"a diagnostic variant, not a
second model of the paper"* (the 64 kB `weight_noc` level is not silicon the chip
has — it stands in for the weight routers), and it is **cold at the live
configuration**, so its first run maps everything.

### 6.9 The mapper bill

One reason the launchers were separate is cost. One chip = one mapper job per
distinct layer shape (12 for resnet18). With `ECC_APPROACHES` naming six arms:

| `ECC_SWEEP` | chips | jobs (resnet18, 12 shapes) |
|---|---|---|
| `fix` | 6 arms | 72 |
| `bch` (6 codes) | 6 × 6 | 432 |
| `depth` (7 scales) | 6 × 7 | 504 |
| `arch` (8 designs, ~5 arms each) | ~40 | ~480 |
| `model` (2 models) | 6 × 2 | 144 (shapes differ per model) |

The task-file design must **skip every chip already in the cache** (today's
launchers already do this per job) and `--dry-run` must print the bill before
anything is submitted. `ECC_RERUN_OPTIMISER=1` on any of these re-solves every
row; `ECC_MAPPER_SEED` is empty, so a re-solve is not guaranteed to reproduce.

**Two things that shrink the bill for testing:**

* **`ECC_LAYERS=layer3.0.conv1`** — one layer of the fixed model instead of all 12
  shapes: `fix` drops from 72 units to 6. Empty means the whole model.
* Once `ECC_RECON_MODELING` is gone, **`ECC_KS` is live again** — today §4's
  `ECC_RECON_K` silently overrides it. So `ECC_KS="57 45 39 30"` with
  `ECC_SWEEP=arch` holds K at 57 (the first entry) — check the first entry is the
  code you mean.

### 6.10 `ECC_JOBS` — bundling units into fewer SLURM jobs

Today one unit of work (one chip × one layer shape) is one SLURM array task, and
`ECC_CONCURRENCY=9` caps how many run at once. The user wants a second knob:
`ECC_JOBS=12` over 72 units → 12 jobs of 6 units each, run back to back.

| | one unit per task (today) | `ECC_JOBS` bundling |
|---|---|---|
| SLURM jobs for 72 units | 72 (9 in flight) | 12 |
| wall time | the slowest single map | ≈ 6 × the slowest map in the worst bundle |
| a unit fails | that one task is red; rerun resumes | its 5 siblings still run (units are independent); rerun resumes |
| cache locking | per unit, already handled | unchanged — each unit still takes its own lock |
| queue / QOS pressure | 72 entries | 12 entries |

Both are legitimate. The plan implements **`ECC_JOBS` as optional**: empty keeps
one unit per task and `ECC_CONCURRENCY` caps the flight; set, it bundles. Units
are assigned round-robin so no bundle gets all the big layers. `--dry-run` prints
the bundling.

One caution: a bundle is serial, so `ECC_MAP_TIME` (24 h today) must cover the
whole bundle, not one map.

---

## 7. Phases

Each phase has a gate. Phases 1–4 must reproduce today's numbers exactly (the
restructure gate). Phases 5–6 change what is computed and are gated by tests and
by an explicit re-baseline.

```
  0 ─► 1 ─► 2 ─► 3 ─► 4 ─► 5 ─► 6 ─► 7
  gate  data  knobs launch  env  metrics sweep  docs
  back        gone  unify   file  +area  unify
```

| # | phase | what | numbers move? | gate |
|---|---|---|---|---|
| **0** | **Gate back** | From `Energy_Modeling/`: `git checkout 46a8df1 -- restructure/gate.sh restructure/snapshot.py restructure/_fingerprints.py restructure/golden restructure/README.md` (the gate ONLY — `gen_guards.py`/`scaffold_arch.py` now live in `tools/`, and the two migration scripts are spent). Then `bash hpc/tl.sh bash restructure/gate.sh --update` to re-take the golden on the current tree, and commit it. **Nothing else until a plain `gate.sh` run is green.** | no | gate green |
| **1** | **Arch facts → `archs/`** | `clock_mhz`, `leakage_nw` → `design.yaml`; `dram_depth` → `_shared/standard.yaml`; THE WIDTH TABLE + GLB multiplier → `widths.yaml`; delete the three bash arrays and `ECC_WEIGHT_WIDTH_GLB_MULT`. `cycle_seconds_for()` stays the ONLY MHz→s conversion (TRAP 2). | **no** — every fingerprint byte-identical | gate item 1 |
| **2** | **Dead & settled knobs out** | delete `ONCHIP_FRACTION`, `REQUIRE_GROUP_RESIDENCY`, `PLACEMENT_CHARGES_DECODE`, `RECON_STEM`, `DEPTH_SWEEP_RECON_DW`, `EVAL_EXPERIMENTS`; move the two `DEPTH_SWEEP_GATE_*` into the sweep script; retire `controller` per 6.7; make `ECC_PHASE` derived per arm per 6.1. Retire the matching guards in `settings/guards.py` and `make guards`. | no | gate |
| **3** | **One launcher** | task file becomes `(arch, model, K, depth, arm)`; `map.sbatch` exports `ECC_RECON_ERT_ARM` from the row; `run_all.sh` enumerates chips from `ECC_APPROACHES × ECC_SWEEP` and **skips cached ones**; `ECC_LAYERS` scopes to one layer; `ECC_JOBS` bundles (6.10); `--dry-run` prints the bill and the bundling (6.9). `map_ert_arms.sh` and `map_depth_sweep.sh` become thin wrappers or are deleted. `ECC_RECON_MODELING`, `RECON_OPTIMIZER`, `ECC_RECON_PLACEMENTS`, `ECC_RECON_LAYER` go. | no | gate + one real smoke unit (`--dry-run` cannot see the code a job runs) |
| **4** | **The new env.sh** | rewrite to the shape in 3.3: four "what to compute" knobs at the top, one option-line comment per important knob, the cache-colding section walled off. Every knob default identical. | no | gate + `test_settings.py` key order pinned |
| **5** | **`ECC_METRICS`** | figure rows for `energy` / `edp` / `latency` from the raw record; **`area` is new** — ART per level + DC engine area for the recon arm. | `area` is new; others no | new unit tests; gate on the three existing metrics |
| **6** | **Sweeps draw placements** | per 6.2: retire the abstract `recon` column; `report/sweep.py` and `panels.py` draw a bar per (design, placement) from each arm's own cache, `recon` = `ECC_RECON_DEFAULT`; missing placements warned per 3.1. Then run the three scoped designs (6.8) and fix what `simple_weight_stationary` / `eyeriss_v2_like_wglb` surface. | figures change shape; totals do not | gate on every total; visual review of one figure per axis; all three designs draw |
| **7** | **Docs** | CLAUDE.md, `plans/README.md`, `GUARDS.md`, `hpc/HIPERGATOR.md`; retire the two old launchers' docs. | no | doc tests |

Phase 4 before phase 5 on purpose: the env.sh rewrite must be gated on
**unchanged** numbers, and that is only possible while nothing new is being
computed.

---

## 8. What a finished run looks like

```bash
# env.sh — the four lines that decide the study
# baseline | embedded | recon1 | recon2 | recon3 | recon4 | recon5
: "${ECC_APPROACHES:=baseline embedded recon1 recon2 recon3 recon4 recon5}"
# bch | model | arch | area | fix          (area = buffer depth, ECC_DEPTH_SWEEP_SCALES)
: "${ECC_SWEEP:=fix}"
# recon1 | recon2 | recon3 | recon4 | recon5   what a bare `recon` means
: "${ECC_RECON_DEFAULT:=recon2}"
# one layer name, or empty for the whole model
: "${ECC_LAYERS:=}"
# empty = one SLURM task per unit;  N = bundle into N jobs
: "${ECC_JOBS:=}"
# energy | edp | latency | area
: "${ECC_METRICS:=energy latency}"
# eyeriss_like_wglb | simple_weight_stationary | eyeriss_v2_like_wglb | …   (first = held)
: "${ECC_ARCHS:=eyeriss_like_wglb}"
```

```
$ bash hpc/run_all.sh --dry-run
  chips     : 6   (reference recon1 recon2 recon3 recon4 recon5 on eyeriss_like_wglb)
  shapes    : 12  (resnet18)
  cached    : 72 / 72   -> nothing to map
  eval      : energy, latency  ×  fix  ×  7 bars
$ bash hpc/run_all.sh
  -> results/figures/ReconSweep__resnet18.png
```

---

## 9. Questions for the user — answer before phase 1

| # | question | status |
|---|---|---|
| 9.1 | `ECC_PHASE` derived per arm; guard in frozen `study/baseline.py` edited to match | **answered: yes** |
| 9.2 | the abstract `recon` arm (R-6) | **answered: retired; `recon` means `ECC_RECON_DEFAULT=recon2`** |
| 9.3 | `ECC_STATIC_ENERGY` | **answered: keep the switch (`0` off / `1` on); move only the data** |
| 9.4 | `widths.yaml`: one shared file keyed by `q`, or one per design? | **answered: one per design** (`archs/<name>/widths.yaml`) |
| 9.5 | retire Task 4's capacity dilation | **answered: yes** |
| 9.6 | remove `controller` from frozen `physics/parity.py` too | **answered: yes, everywhere** |
| 9.7 | is the `area` **metric** wanted enough to build? (only metric with no source) | **answered: yes** — phase 5, built last |
| 9.8 | axis name | **answered: stays `area`** |
| 9.9 | after phase 3, delete `map_ert_arms.sh` / `map_depth_sweep.sh` or keep as wrappers? | **answered: delete** (git has them) |
| 9.10 | `ECC_JOBS` bundling as specified in 6.10 (optional; empty = today's behaviour)? | **answered: yes** |
