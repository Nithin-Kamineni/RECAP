# CLAUDE.md

Energy modelling for relaxed error correction in DNN accelerators, on Timeloop +
Accelergy. The question: **if ECC parity does not have to be stored in DRAM, how
much inference energy does that save, and does the answer depend on the
accelerator?**

**The live plan is `prompt_7.md`** (since 2026-09-12) — modelling TIME, and what
that does to the energy answer: the off-chip speed limit, clock gating of the
reconstruction engines, and component standby power. Read it before starting
work; where it and this file differ on what to do next, it wins. **Its §9 is the
PHASE PLAN — one phase per working session, each with a gate you can run** — its
§0 is how a new session picks up, its §12 the reporting rules that must travel
with every figure, and its §13 what is already fixed and what is deliberately
left alone. The old numbered issue list is gone; Issue 9 is resolved into §12's
rule R-3.

**`prompt_6.md` is still in force** for the four RULES that stop the model
double-counting itself (one owner per effect, one charging site, two terms on two
denominators, one plan per bar) — prompt_7 extends RULE 3 rather than replacing
it. `prompt_7.1.md` is a later buffer-size sweep, to be run after prompt_7.
`prompt_3.md`'s constrained mapspace is carried by both and every mapper job runs
under it.

Phase-by-phase status lives in `progress.txt`, not here.

**No empirical claims in this file.** Numbers live in `FINDINGS.md`; the live
caveat list is `bash run.sh diagnose`, computed from the architectures as they
stand. A prose copy here went stale once and then contradicted the code.

**PROTECTED SECTIONS — do not delete or shorten when condensing this file.**
A section headed `<!-- PROTECTED -->` records a mistake that has already been
made more than once and costs a debugging session and a wave of compute each
time. Condensing this file is fine; those sections are not part of it. **If you
believe one must be cut or shortened, ASK THE USER FIRST and say which one and
why** — do not decide it yourself, and never drop one as a side effect of "make
this shorter". Today that is: THE WIDTH TABLE.

**prompt_6 phases 1–9 are DONE (2026-09-11); prompt_7's Phase 0 is DONE
(2026-09-12); its Phases A, B, C, D and E are NOT.** The placement study's numbers in
FINDINGS §2.9 were produced under RULE 3 (two encoder terms, two denominators)
from `bash hpc/map_ert_arms.sh`. The three sweep figures were regenerated only as
far as their caches reach (FINDINGS §6.1); `ModelSweep.png` is stale on disk.

`legacy/` is pre-rewrite and describes nothing current, except the dated
`FINDINGS_detail_*`, `progress_*` and `PROJECT_STATUS_*` snapshots, which are the
full working record.

## Running it

`env.sh` is the only file you edit: every `ECC_*` knob, ten commented sections,
sourced by `run.sh`, `hpc/run_all.sh`, `hpc/map.sbatch` and `hpc/tl.sh`, so a
value cannot mean one thing to the mapper and another to the evaluator. Sections
1–9 are knobs; section 10 is derived. **Read env.sh for what a knob does — it is
commented; do not restate it here.**

Every value is written `${VAR:=default}`, so **the environment wins over the
file** and a one-off never needs an edit.

**Since 2026-09-11 the defaults ARE prompt_3's constrained search**
(`ECC_MAPSPACE_CONSTRAIN=1`, `ECC_WEIGHT_FACTOR_RELAX=1`, `linear_pruned`,
timeout 100000000): a bare `run.sh` is the exhaustive constrained mapspace and
nothing needs exporting. Only a design with an `archs.MAPSPACE_FREE_LEVELS` entry
is constrained (today `eyeriss_like_wglb`). To map any other design, either write
its free-set first (prompt_3, "Porting it") or set `ECC_MAPPER_ALGORITHM=random_pruned
ECC_MAPPER_TIMEOUT=2000` back -- a systematic walk of an unconstrained space is the
wrong regime (FINDINGS §2.2).

    bash hpc/run_all.sh          # THE command: map array -> dependent eval+plot
    bash hpc/run_all.sh --map-only | --eval-only | --replot | --local

`run.sh` runs ONE stage and has no knobs of its own: `map`, `baseline`,
`embedded --eval`, `recon --eval`, `dilation`, `validate`, `diagnose`, `panels`,
`--replot`, `--dry-run`.

**Only `timeloop.py` needs the container.** `--eval` (`ECC_FROM_CACHE=1`) never
invokes Timeloop; validate, diagnose, replot and the whole `eccenergy/tests/`
suite run on any python with pandas, matplotlib and pyyaml.

**HiPerGator:** `module load apptainer`, then `bash hpc/tl.sh <any run.sh command>`.
Never run the mapper on a login node — use `srun`/`sbatch`. Keep
`ECC_MAPPER_THREADS=18` and `--cpus-per-task=18`: the thread count is in the
mapping fingerprint, so any other value is a cold cache. `hpc/HIPERGATOR.md` has
transfer, image build and the cost model.

**A cold map is hours.** Narrow before widening
(`ECC_ARCHS=<one> ECC_MODELS=<one> bash hpc/run_all.sh --map-only`) and watch the
filesystem rather than a pipe — a backgrounded `docker … | grep` buffers until
the pipeline ends and looks hung when it is fine.

**Fan out, don't queue.** `hpc/map_by_shape.sh` submits one job per layer SHAPE
instead of one per model, several architectures at once, and ONE dependent eval
after all of them. `--no-eval` submits maps only; `ECC_LAYERS` maps exactly those
layers. Each submission snapshots the SLURM task file, because `map.sbatch`
resolves its (arch, model) pair from it when the job RUNS. **`--array=I-I` is
required per submission** — without it the extra tasks exit 2 and the dependent
eval sits on `DependencyNeverSatisfied`.

## One sweep, two constants

All three ECC arms (`baseline`, `embedded`, `recon`) are always drawn, so the arms
are never an axis. A run sweeps exactly ONE of the remaining three axes:

| `ECC_SWEEP` | x axis | sweep list | held | stem |
|---|---|---|---|---|
| `bch` | BCH(63,K) | `ECC_SWEEP_KS` | arch, model | `BCHsweep` |
| `model` | networks | `ECC_SWEEP_MODELS` | arch, K | `ModelSweep` |
| `arch` | accelerators | `ECC_SWEEP_ARCHS` | model, K | `ArchitectureSweep` |

Nothing below `config.py` except `sweep.py` knows which axis is swept. A model
sweep is all-CNN or all-transformer — mixing families is a config error.

**`ECC_RECON_MODELING=1`** (env.sh §4) is not a fourth sweep: its axis is WHERE on
the weight path the reconstruction boundary sits. **§4 hard-assigns `ECC_ARCHS`,
`ECC_MODELS`, `ECC_CODE_N`, `ECC_KS` and (since 2026-09-11) `ECC_LAYERS` with a
bare `=`**, so setting those names on the command line does nothing and does it
silently — use the `ECC_RECON_*` spellings (`ECC_RECON_LAYER=<name>` or `all`).
Under `RECON_OPTIMIZER=True` the figure is always
`results/figures/ReconSweep_optimiser__<model>.png`, even on a one-layer run; the
scope is in the manifest and the title, the model in the name (since 2026-09-11,
so two networks never overwrite each other). `ECC_RECON_ARCHS` is ONE PANEL PER NAME; each panel keeps its own x
axis and its own two reference bars, so a percentage on one panel says nothing
about the other.

## The reduced representation is `datawidth`, not depth

**Capacity dilation (`depth × N/K`) is WITHDRAWN** — it answered negatively and it
was unfair besides (Accelergy priced the deeper array 1.18–1.46× dearer, so the
optimiser had a reason to leave the room unused). FINDINGS §2.4 has the cause: the
capacity mechanism moves in 2× steps and N/K = 1.6154 sits below the step.

The live mechanism is **`datawidth: q` on the weight levels at a FIXED `depth`**,
with each level's `width` taken from THE WIDTH TABLE below. CACTI receives `depth`
and `width` only — `datawidth` never reaches the energy model — and Timeloop bills
`vector_access_energy / block_size` per weight, `block_size = width/datawidth`. So
the reconstruction arm gets more effective capacity on an array within 2 % of the
reference's per-access read/write/leak.

**`q = round(8·K/N)`.** Two places used to say `ceil`; they agree everywhere except
BCH(63,57) (ceil 8, round 7) and BCH(63,51) (ceil 7, round 6) — and at q=8 the
"recon" arm *is* the embedded arm, so a gate run that way compares embedded with
itself and reports an effect of exactly zero. `hpc/map_depth_sweep.sh` and
`dilation.py` both call `code_widths.declared_datawidth()`. **`recon.py`'s
`Packing` still uses `ceil`** — it is Task 3's physical packing model — so Task 3
narrows those two codes less than the mapping study does.

<!-- PROTECTED -->
## THE WIDTH TABLE — read this before touching a width

*(PROTECTED: see the top of this file. Do not cut or condense without asking.)*

**THE ARMS DO NOT SHARE A DECLARED WIDTH. Each arm declares the width that suits
ITS OWN `datawidth`, and no arm has to be legal for any other arm's `datawidth`.**
This has been got wrong in five separate sessions. It is prompt_2.md's table and it
is applied automatically, on every run, by `archs._set_weight_geometry()` from
`eccenergy/code_widths.py`:

| arm | q | spad `width` | GLB `width` (×4) | weights/word | eff. capacity |
|---|---:|---:|---:|---:|---:|
| Baseline / Embedded | 8 | **96** | 384 | 12 | 1.0000× |
| BCH(63,57) | 7 | **98** | 392 | 14 | 1.1667× |
| BCH(63,45), BCH(63,51) | 6 | **96** | 384 | 16 | 1.3333× |
| BCH(63,39), BCH(63,36) | 5 | **95** | 380 | 19 | 1.5833× |
| BCH(63,30) | 4 | **96** | 384 | 24 | 2.0000× |

**BCH(63,39) runs at width 95 because `95 % 5 == 0`. `95 % 8 = 7` IS IRRELEVANT** —
the baseline/embedded arm is never mapped at width 95; it is mapped at 96, where
`96 % 8 == 0`. Same for 98 at q=7. `timeloop-mapper`'s constraint
(`width % (word_bits * block_size) == 0`, `buffer.cpp:302`, `block_size` defaults
to 1, **no floor path**, measured `exit=134`) is **per level, per mapper run**, and
one mapper run maps **one arm**. It never sees two arms at once and imposes no
constraint between them. Do not "fix" 95 or 98.

This is what the mapper actually sees on `eyeriss_like_wglb`, with no knob set
anywhere — every level legal for the datawidth **it** stores, and for nothing else:

    Baseline/Embedded   q=8  filter_glb 43x384b / 8b = 48 weights/word   (384 % 8 == 0)
    BCH(63,57)          q=7  filter_glb 43x392b / 7b = 56 weights/word   (392 % 7 == 0)
    BCH(63,45)          q=6  filter_glb 43x384b / 6b = 64 weights/word   (384 % 6 == 0)
    BCH(63,39)          q=5  filter_glb 43x380b / 5b = 76 weights/word   (380 % 5 == 0)
    BCH(63,30)          q=4  filter_glb 43x384b / 4b = 96 weights/word   (384 % 4 == 0)

Depth 43 on every row — that is the shared quantity. The width moves, the
datawidth moves, and each row's `%` is the only divisibility that exists.

**WITHDRAWN 2026-09-12: the `lcm(q, 8)` scheme** (56 / 24 / 40), which came from
reading the rule as "both arms share one width". It was wrong on the premise AND it
**made the 8-bit reference arm move between codes**, so the reference held 35/33/30
weights per PE at q = 7/6/5, fell off a tiling cliff at q=5, and reported
BCH(63,39) as a 37.69 % win that was really the reference breaking (FINDINGS
§2.4b). Under this table the 8-bit arm is width 96 at **every** code: one arm,
mapped once.

**There is no `ECC_WEIGHT_WIDTH`.** The table is not a knob — a knob is what let
the lookup ship on 2026-09-11 and sit unused until `map_ert_arms.sh` died on it at
K=39 (all 24 arm jobs on `filter_glb: width 64 % datawidth 5 != 0`, dependent eval
parked on `DependencyNeverSatisfied`). It is applied **per level**: at
`ECC_WEIGHT_DATAWIDTH_LEVELS=filter_glb` the GLB stores 5-bit weights at width 380
while `weights_spad` keeps 8-bit weights at width 96. Read it with
`python3 -m eccenergy.code_widths`.

**DEPTH is shared and is the only thing `assert_pair_geometry()` checks.**
`depth' = round(depth × width / 96)` is computed at the BASE width, not the arm's
own, so every arm declares the same depth — which is what makes the `eff. capacity`
column above come out as `(W_arm/q)/(96/8)`, and what leaves the depth check
something real to check. `ECC_DISABLE_ASSERT_PAIR_GEOMETRY=1` (env.sh §5, default
`0`) turns that depth check off for a study that varies depth between the arms on
purpose. It is the only thing that knob disables — **there is no width check for it
to disable**.

Three assertions, each with a mutation test (`test_dilation.py`):
`archs.assert_pair_geometry()` (both arms declare the same LEVELS at the same
DEPTH — never the same width, see above), `recon.assert_onchip_narrowing_once()` (the
mapper and the evaluator can both narrow, and doing both SQUARES the saving —
hence `ECC_RECON_PACKING=aligned`, the default), and the level table's own
row-level re-check.

**`--levels` always prints the per-DATASPACE table.** A weight-only table once
made a large win look like a null result; FINDINGS §2.8.

## Results layout

    results/_raw/<arch>/<treat>/fp-<hash>/<workload>/<scope>/cls-<mode>/rw-<split>/<model>.json
    results/evaluation/{Pre|Post}/<arch>/<model>/<bch>/<prec>/<scope>/<mapper>/<runid>.json
    results/{figures,tables,manifests}/<stem>.{png,pdf|csv|json}

**The stem comes from the configuration alone**, so re-running at different
constants REWRITES the file instead of adding one; the manifest beside it records
what produced what is on disk. Copy a figure out, or point `ECC_RESULTS_DIR`
elsewhere, to keep it.

`results/evaluation/` accumulates on purpose: one JSON per run, holding every ECC
variant including the ones not evaluated (`total_energy_pJ: null` plus a reason).
Nothing but `results_store.py` writes one. Schema: `docs/RESULTS_SCHEMA.md`.

## The two caches

1. **Mapper cache** — `ecc_energy_study/outputs/<arch>/<treat>/fp-<hash>/<shape>/`,
   one entry per (architecture, layer shape). **Never delete it**; it is hours of
   compute and every solved shape is saved immediately, so runs resume.
   `fp-<hash>` hashes the patched YAML the mapper actually sees plus the globals
   and every mapper setting, and each entry carries a `mapping.json` sidecar.
   **A different fingerprint is a MISS** — before this existed, editing an
   arch.yaml left the path unchanged and old mappings were reported as the new
   design.
2. **Raw energy cache** — `results/_raw/`, parsed from the mapper cache. Pure
   Timeloop output, independent of the ECC configuration, so changing the code
   geometry and redrawing is milliseconds.

Anything that changes what the mapper sees or optimises gets its own cache
subdirectory. **So anything touching env.sh §2/§5, or an arch YAML, takes the
whole matrix cold at once** — budget for it.

**The fingerprint hashes the ARCHITECTURE, not the price list Accelergy derives
from it.** So fixing an ENERGY ESTIMATOR changes every number in the cache while
leaving the directory it is stored under identical, and the stale entries are
reused with nothing to say so. That is not hypothetical: the Neurosim plug-in
answered 0 pJ for every smartbuffer address generator until 2026-09-12 (it
crashed writing scratch into the read-only SIF and Accelergy accepted the 0;
`hpc/tl.sh` now binds it a writable copy). **`ECC_ENERGY_MODEL_REV` is the
deliberate cold**: any non-empty value re-fingerprints the whole matrix, EMPTY
hashes byte-identically to every fingerprint that predates the knob. Bump it when
an estimator changes, not before — and expect to re-map.

**A comparison across two `fp-<hash>` directories is a comparison of two
ARCHITECTURES**, not two settings. `sibling_fingerprints()` enumerates every
solved fingerprint under one variant slug and the table prints a
`!! FINGERPRINT WARNINGS` block. One reported headline was a stale sibling.

## The three ECC arms

One BCH(N,K) codeword over `ECC_WEIGHT_BITS`-bit weights under all three, so they
differ only in where the parity lives. Codewords are counted from DRAM weight reads.

- **baseline** — parity beside the data in DRAM. It pays a **dearer per-bit price,
  not extra traffic**: the decoder is on the DRAM die, so its parity is read,
  corrected and discarded there and never crosses the datapath. What it pays for
  is an array that also holds the parity.
- **embedded** — parity inside the stored weights, laid out as the embedding
  pipeline does it: the MSB-first weight bit stream cut into n-bit codewords, so
  weights straddle codewords and the n−k lowest-significance positions carry parity.
- **recon** — DRAM as embedded, K/N of the weights held on chip, the rest
  regenerated by a synthesized datapath characterised in `data/dc/BCH_N63_results.json`.

`build_stacks()`'s `recon` arm is ONE point applied to every design at once, and no
physical boundary does what it does. It is **not** one of the placement study's
boundaries and must not be quoted as one.

## `recon.py` — the placement space

Two tables define it and **must be edited together**: `WEIGHT_PATHS[<arch>]` (the
stages of that design's weight path, each naming the Timeloop levels that are it)
and `PLACEMENTS[<arch>]` (the boundaries: which stages stay reduced, and what
drives the reconstruction count).

**Designs do not have the same boundaries.** `eyeriss_like_wglb` (Eyeriss v1) and
`simple_weight_stationary` have five; `eyeriss_v2_like` has four.
`eyeriss_v2_like_wglb` **is refused** — its `weight_glb` stage has no boundary
(prompt_6 Appendix B has the fix).

That is enforced, not advised. `validate_placement_space()` requires a placement's
`reduced` set to be a **prefix** of the path's reducible stages in path order, and
every reducible stage to be reached by some boundary. Without it, adding a stage to
one table and forgetting the other keeps every boundary below it reporting its own
saving while the new stage stays at full width — the whole list understated, with
nothing saying so. `weight_path()` also refuses if a level carrying Weights energy
is not claimed by exactly one stage, and `cross_check()` re-derives per-category
weight energy against the `Raw` record first.

**A NETWORK boundary's encoders run once per ARRIVAL, not once per injection** —
the count is `Ingresses × Multicast factor`. Charging the ingress count pairs a
destination-side saving with a source-side cost, which is not a placement.
`ECC_RECON_ENCODER_SITE=source` reproduces the earlier numbers for a diff.

**R4b (a retained-reconstruction register) IS REMOVED.** Consecutive weight reuse
is 1 on 20 of 21 resnet18 layers, so a latch catches nothing and a register that
pays would be a second scratchpad. **Do not reintroduce one without first
re-measuring `consecutive_run` on the target mapping** — that is the step the
original audit skipped (FINDINGS §2.7).

Three things the model refuses to fudge, each with a knob and a recorded check:
**physical packing** (`stream` scales every reduced stage by K/N; `aligned` gives
each weight whole bits), **reconstruction granularity** (`G_rec`, computed from the
layout, not assumed), and **feasibility** (a PE-local boundary whose resident tile
is smaller than `G_rec` is reported `unsupported` with the layers named).

The DRAM term is ONE stage and the whole of it is reducible: only the k message
bits are read out and driven off the die. `ECC_DRAM_PJ_PER_BIT` is the per-bit
cost; `ECC_MAC_PJ_OVERRIDE` rescales the Compute category evaluator-side, after
the raw cache, so no mapping moves. Both carry their citation from
`provenance.yaml` onto every figure and result.

**Every result prints the ceiling first** — the weight energy a boundary could
reduce, times `1 − K/N`. Without it a sub-percent saving reads as a missing term
rather than as arithmetic.

## Architecture rules

`archs/_shared/standard.yaml` states what must be IDENTICAL across designs;
`provenance.yaml` records where every declared number came from; `noc.yaml` holds
the interconnect coefficients and, per design, which spatial containers actually
are the NoC. Two NoC terms Timeloop cannot be given are charged after mapping by
`noc_post.py`. `archs/_shared/components/` is part of the mapper fingerprint:
editing a component colds every cache, on purpose. `bash run.sh validate` checks
all of it and exits non-zero on a violation.

Only one of the four `source:` values licenses using a design's name as the chip:
**`published`**. `derived` differs in one stated block, `reference_design` is
shipped by timeloop-accelergy-exercises and merely *named* after a paper,
`locally_authored` reproduces nothing. `simba_like` is `reference_design` — label
it "Simba-like (reference design)".

**Eyeriss v1 IS `eyeriss_like_wglb`** (decided 2026-09-10). JSSC 2017 publishes the
8 kB filter-weight allocation of the 108 kB GLB, so the file that models it is the
design; `eyeriss_like`, which declares `!Nothing` in its place, is retired — still
mappable by name for a diff, but no longer the design. The v1 bracketing-pair
doctrine is over (`config.BRACKET_PAIRS` is empty); the bracket rule still holds
for v2.

**A constrained/relaxed dataflow is a different accelerator.** `ECC_MAPSPACE_CONSTRAIN`
and `ECC_WEIGHT_FACTOR_RELAX` change the dataflow (v1's `M=1` at the filter spad IS
row-stationary), so such a run must never be quoted as the published chip. Label
bars "constrained dataflow".

Accumulator width is deliberately not standardized (v1 16b, v2 20b, Simba 24b, each
cited). `diagnose` flags a psum level narrower than its own accumulator; a level
that is legitimately narrower declares `# psum-width-ok: <reason>` in the YAML.

## Working on this code

- **Frozen**: `parity.py`, `baseline.py` and `external_parity()` are not
  refactored. `build_stacks()` **is no longer frozen** — prompt_6 RULE 3 changes
  its reconstruction term. After any change there, re-run `bash run.sh baseline
  --eval` and diff: Task 1 and 2 totals must not move.
- **Never** read `os.environ` outside `config.py`, and never resolve a path outside
  `paths.py`.
- **Handing Timeloop an energy table** (prompt_6): pass `ERT:` and `ART:` YAMLs
  beside `design_inputs()`, and pre-write them into the output directory as
  `timeloop-mapper.ERT.yaml` / `.ART.yaml` first -- with a supplied table Timeloop
  writes neither, timeloopfe's parser then raises after a successful search, and
  `Mapper` counts the shape failed. `timeloop.ErtTables` is the production hook
  (base table from Accelergy once per arm under `<cache>/_ert/`, two rows bumped,
  staged per shape, read back after the map); `python3 -m
  eccenergy.experiments.ert_probe` is the proof (FINDINGS §3.5) and writes under
  `paths.ert_probe_dir()`, never the mapper cache.
- **An ERT arm is a configuration, not a design.** `ECC_RECON_ERT_ARM=<placement
  key>` (set per job by `hpc/map_ert_arms.sh`) resolves in `config.py` into
  `datawidth: q` on the storage levels of that placement's `reduced` set and an
  ERT bump derived by `archs.ert_bump()`; which placements qualify is DERIVED
  (`recon.ert_arms()`: storage site, `reads`/`fills` counter, not the innermost
  level's reads) -- never `if key == "recon2"`. The arm is in the cache slug
  (`ert-recon2-filter_glb-read`) AND the fingerprint, and every entry's stored
  ERT is read back before a number is used: two arms with byte-identical YAML
  must never share a directory. `ECC_RECON_ERT_AWARE=1` (needs
  `RECON_OPTIMIZER=True`, `ECC_PHASE=Post`) bills each ERT bar from ITS OWN plan
  and the other boundaries from the reference plan; the toll Timeloop billed
  inside the level is MOVED into `Reconstruction`. Timeloop prints leakage
  outside the per-dataspace energies and the raw record never held it, so only
  the access toll is in the bill; the idle term is verified against the stats'
  leakage and charged by the evaluator.
- **Who narrows an on-chip stop is MEASURED, per bar per stage** (prompt_6 RULE
  1): `Word bits == q` in that bar's own stats means the mapper did (evaluator
  applies 1.0), `== weight_bits` means the evaluator does, anything else stops.
  Two live sites or none on a narrow stop are both hard refusals. The same
  measured guard sits in `build_stacks()`'s recon column.
- **Reconstruction is two terms on two denominators** (RULE 3): `incremental x
  events + idle_per_cycle x cycles x N_engines`; `load_recon_energy()` returns
  them separately and nothing adds them. **Since prompt_7 (2026-09-12) both terms
  are CLOCK-GATED** by `ECC_RECON_CLOCK_GATING_PCT` (default 99.5): the engine
  burns `incremental + idle` while it works and `idle x (1 - g)` while it is
  gated off, because the DC "idle" constant is 99.48% clock power and only 0.52%
  true leakage. `PCT=0` reproduces the ungated model to the pJ and is what every
  pre-gating assertion is pinned to. **The ERT bump carries the SAME gated
  numbers** (`incremental + idle x g` per access, `idle x (1 - g)` per cycle), so
  the mapper and the evaluator price one engine and RULE 5.3's "a split, not an
  addition" still holds. Cycles come from the billed plan's own
  record (`Raw.cycles`, re-gathered if absent, never charged zero); the idle
  denominator is `StageStats.engine_cycles` = sum over layers of (engines that
  leak x that layer's cycles): 1 at DRAM, THAT LAYER'S OWN fanout x instances at
  a network (prompt_7 Issue 3, fixed 2026-09-12 -- it used to charge the widest
  layer's fanout over the whole run, which over-billed mobilenet_v2 by x1.3336
  because its depthwise layers broadcast 2-12 wide, not 14), and
  at a storage level the UTILIZED instances of that layer's plan -- Timeloop
  power-gates each unused instance and bills `leak x utilized x cycles`
  (`buffer.cpp FinalizeBufferEnergy`, verified 2026-09-11 on 43 shapes of two
  models; the declared count is reported beside it). A PE count that differs
  between an ERT arm's own plan and the reference is REPORTED per shape (both
  counts, cycles, Timeloop EDP ratio; manifest `title_caveats`), never refused.
  The DC tables live in env.sh §6.
- **Adding an architecture**: `archs/<name>/arch.yaml` (plus `arch_paper.yaml` with
  each number cited), the name in `KNOWN_ARCHS` and `ARCH_LABELS` in `config.py`,
  entries in `standard.yaml` and `provenance.yaml`, then `bash run.sh validate` and
  `diagnose` before committing to a long sweep.
  **Declare each weight level's PUBLISHED `width`/`depth`/`datawidth` and stop
  there** — do not hand-pick a width to suit a code, and do not check one against
  another arm's datawidth. `archs._set_weight_geometry()` reshapes every weight
  level per arm from THE WIDTH TABLE (above) and renormalises depth to hold your
  declared total bits, so `width % datawidth == 0` is satisfied by construction at
  every code. If a `width 64 % datawidth 5 != 0` ever reaches you, the fix is in
  `code_widths.WIDTH_TABLE`, never in the arch YAML and never in the config.
- **Adding a model**: add to `CNN_MODELS` or `TRANSFORMER_MODELS` in `config.py`,
  then `python3 -m eccenergy.generate models <name>` in the container.
- **Adding an architecture to the placement study**: `WEIGHT_PATHS[<name>]` and
  `PLACEMENTS[<name>]` together, then the name in `ECC_RECON_PLACEMENTS` in env.sh
  §4. Get the stage-to-level match right by reading a real
  `timeloop-mapper.stats.txt` AND `timeloop-mapper.map.txt` from that design's
  cache. `test_every_placement_space_is_valid_for_every_supported_design` makes a
  half-finished pair fail the tests rather than understate a figure.
- **Adding a sweep axis**: a name in `SWEEPS`, a stem in `SWEEP_STEMS`, resolution
  in `Config.__post_init__`, and a `_<name>_groups()` in `experiments/sweep.py`. Do
  not write a second renderer.
- **Changing how a bar looks**: `draw_panel()` in `plots/stacked.py` is the only
  place a bar is drawn, so every figure moves together. Add a parameter to it
  rather than a second routine.
- **The placement figure's heading is ONE line** (`Config.recon_title` /
  `recon_panel_title`, since 2026-09-11): design, model and layer scope, weight
  width, code, regime tag. Everything it used to say below that -- regime with
  its arms, DRAM model and price, static DRAM terms, encoder site, MAC
  denominator, development-run warning, the multi-panel rule -- is
  `Config.recon_caveats()` and lands in the manifest beside the figure as
  `title_caveats` (with `title`). A new caveat goes THERE, never as a line on
  the figure.
- **Line endings**: the shell scripts run inside a Linux container and a CRLF makes
  bash die on `set -o pipefail` with a mangled message. `.gitattributes` forces LF;
  `bash tools-fix-eol.sh` repairs anything that slips through. **Always emit LF.**

## Layout

Everything not listed here is what its name says; `eccenergy/` module docstrings
carry the rest.

    env.sh              THE knob file        run.sh    one stage, no knobs
    prompt_7.md         THE LIVE PLAN        FINDINGS.md   what was learned
    prompt_6.md         the four RULES, still in force
    prompt_7.1.md       buffer-size sweep, AFTER prompt_7
    Claude-sandbox/     throwaway experiments; never writes to the caches
    hpc/                run_all.sh, map.sbatch, tl.sh (apptainer wrapper),
                        map_by_shape.sh, map_ert_arms.sh (prompt_6: one job per
                        arm x shape, one dependent eval), map_capacity_sweep.sh,
                        map_depth_sweep.sh, summary.py, HIPERGATOR.md
    eccenergy/          config.py (the ONLY reader of os.environ), paths.py (the
                        ONLY resolver of paths), workloads.py, timeloop.py (the
                        only module needing the container), energy.py, ecc.py,
                        recon.py, parity.py, embedded.py, code_widths.py,
                        noc_post.py, results_store.py (the ONLY writer of an
                        evaluation JSON), plots/, experiments/, tests/, generate.py
    archs/_shared/      standard.yaml, provenance.yaml, noc.yaml -- not an
                        architecture; skipped by the installer
    ecc_energy_study/   AUTO-MANAGED: cloned repo + mapper cache. Do not delete.
    legacy/             pre-rewrite material, plus the dated FINDINGS_detail_*,
                        progress_* and PROJECT_STATUS_* snapshots that hold the
                        full working record. Nothing imports it.
