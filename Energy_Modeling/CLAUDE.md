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

**`plans/ProjectRestructure.md` was implemented one phase per session** over
2026-09-13. It was organisation, not modelling: **every phase reproduced the
numbers exactly**, proved by a gate that refused unless `arch_fingerprint()` was
byte-identical for every (arch, model) and every evaluated total was unchanged
to the pJ. **Phases 0-6 are DONE and phase 7 (backfill tests) is ongoing, not
blocking**, so `restructure/` -- the gate, its golden snapshot and the two
migration scripts -- was REMOVED on 2026-09-13 rather than left as scaffolding
nobody runs. It is one command away if a future phase wants it back:

    git log -- Energy_Modeling/restructure     # then `git checkout <commit> -- ...`

ProjectRestructure §10 is still what not to do, and §9.1 says what the gate
covered and what it could not.
**`GUARDS.md` (generated, `make guards`) is the list of every guard** -- 136 of
them, each with an id and a TIER. Tiers 3 and 4 are liftable by naming them in
`ECC_ALLOW` (env.sh section 1), and an override is RECORDED on the manifest as
`guard_overrides` and on the figure's caveat list. Tiers 1 and 2 never lift.

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

**prompt_6 phases 1–9 are DONE (2026-09-11); prompt_7's Phase 0, A and B are DONE
(2026-09-12) and its Phase C1 is DONE (2026-09-13). C2 and E are NOT. PHASE D IS
DROPPED** — the user's call, 2026-09-13: this study is about CNNs, and §4.5/A.6
already reach the transformer ceiling from the CNN side, so quote 52.4% as a
PROJECTION and never as a measurement.

<!-- PROTECTED -->
## EVERY MAPPER CACHE IS COLD, ON PURPOSE, SINCE 2026-09-13

*(PROTECTED: see the top of this file. Do not cut or condense without asking.)*

**Phase C1 re-fingerprinted every arm and every design. Nothing was deleted.**
**Phase C2 has since run for `resnet18`** (2026-09-13): six arms, 12 shapes, 72
maps, all on their own plans. **Every other model is still cold**, and for those
there is no solved mapping at the live configuration -- which is the phase
working correctly rather than a cache that went missing.
`bash run.sh baseline --eval`, `--replot`, the three sweep figures and
`tests/test_latency.py` all have nothing to read; `test_latency` says so and
skips. **Do not "fix" this by pointing anything at an older `fp-` directory — a
comparison across two `fp-` directories is a comparison of two ARCHITECTURES.**

The cold is six declarations that reach the architecture the mapper reads: the
off-chip speed limit (`shared_bandwidth` on DRAM), the per-dataspace bandwidth
scale, a bit-aware port on a narrowed level, the 200 MHz clock, the MAC price
inside the ERT, and a bank count that now reaches CACTI — plus
`ECC_ENERGY_MODEL_REV`, which since 2026-09-13 reaches `arch_fingerprint()` and
not only `Config.fingerprint()`. prompt_7 §13 lists all of them with their
measured effects.

**`ECC_DRAM_BANDWIDTH_MBPS` IS NOW IN THE PATCHED YAML**, so changing it
re-fingerprints all six arms and colds them again. That is the knob working.
Never pin one of the new `fp-` hashes in a file — print the current set with
`bash hpc/tl.sh bash -c 'source env.sh; PYTHONPATH=. python3 Claude-sandbox/_fp.py'`,
and let `tests/test_mapper_arms.py` assert the rule (every arm moved off its
pre-Phase-C hash, all six distinct, nothing deleted) rather than any value.

**THE ONE COMMAND THAT ENDS THE COLD** (Phase C2 — hours of SLURM; 6 arms ×
the model's distinct shapes, so 72 jobs on resnet18 and 186 on mobilenet_v2):

    ECC_RECON_LAYER=all ECC_RERUN_OPTIMISER=1 ECC_RECON_ERT_AWARE=1 \
        bash hpc/map_ert_arms.sh

The pre-Phase-C directories are intact and are the only record of what FINDINGS
§2.9's numbers were computed from: `fp-718d53aac189` (reference),
`fp-a18a5b15fd8b` (recon2), `fp-55569ac34427` (recon4), 43 shapes each.
`tests/test_mapper_arms.py` asserts they are still there.

The placement study's numbers in FINDINGS §2.9 were produced under RULE 3 (two
encoder terms, two denominators) from `bash hpc/map_ert_arms.sh` **at the
pre-Phase-C fingerprints, so they are superseded rather than wrong.** The three
sweep figures were regenerated only as far as their caches reach (FINDINGS
§6.1); `ModelSweep.png` is stale on disk.

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

**Only `toolchain/invoke.py` needs the container.** `--eval` (`ECC_FROM_CACHE=1`) never
invokes Timeloop; validate, diagnose, replot and the whole `eccenergy/tests/`
suite run on any python with pandas, matplotlib and pyyaml.

**HiPerGator:** `module load apptainer`, then `bash hpc/tl.sh <any run.sh command>`.
Never run the mapper on a login node — use `srun`/`sbatch`. Keep
`ECC_MAPPER_THREADS=18` and `--cpus-per-task=18`: the thread count is in the
mapping fingerprint, so any other value is a cold cache. `hpc/HIPERGATOR.md` has
transfer, image build and the cost model.

## Changing a declared value is a normal thing to do

**ONE SUBMISSION MAPS ONE ARCHITECTURE.** `hpc/map_ert_arms.sh` snapshots
`archs/` at submit time into `hpc/.runtime/archpin.<pid>/` and exports
`ECC_ARCH_PIN_DIR` (env.sh section 7) to every map job and the dependent eval,
so **editing `archs/` while an array is in flight is free** — the queued jobs
keep mapping the chip you submitted. It prints the pin and the reference
fingerprint when it submits; quote those two lines when a run is questioned.
`hpc/run_all.sh` pins the same way (only the top-level submitter takes the
snapshot; the dependent eval inherits it). `map_by_shape.sh`,
`map_capacity_sweep.sh` and `map_depth_sweep.sh` do NOT pin yet.
Without it, an edit landing 78 seconds into a 282-job array cost the whole run
(2026-09-13, `efficientnet_b0`): the maps solved one geometry and the eval went
looking for another, and every layer reported `mapper failed`.

**The fingerprint hashes the GEOMETRY, not the prose.** `archs.hashable_arch_text`
strips comments, trailing whitespace and blank lines before hashing, so keeping
`arch_paper.yaml`'s citations honest costs nothing while `depth: 64` ->
`depth: 96` still colds the cache. `eccenergy/tests/test_arch_fingerprint.py`
asserts both halves with mutations.

**A cold cache is a SKIP, not a failure.** `eccenergy/tests/conftest.py` restores
`os.environ` around every test and reports each module's home-grown `_Skip` as a
skip. Before it the suite showed 64 failures at a cold cache with nothing wrong
in the code; it is now green, and the data-backed properties come back as they
are mapped. `bash hpc/tl.sh python3 -m pytest eccenergy/tests/ -q` runs it
(pytest is a `--user` install, not in the image). Since 2026-09-13 conftest also
offers **`cfg()`, one config fixture pinned to a STATED design** rather than to
whatever env.sh points at this week (nine modules carry a near-copy of it), and
`test_plumbing.py` mutation-tests all three pieces of plumbing -- break the
restore, the `_Skip` translation or the pin, and a test goes red. **A test
function returns None**: `check()` in the pre-pytest modules discards the value
and pytest warns on it.

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
`study/dilation.py` both call `physics.widths.declared_datawidth()`. **`physics/packing.py`'s
`Packing` still uses `ceil`** — it is Task 3's physical packing model — so Task 3
narrows those two codes less than the mapping study does.

<!-- PROTECTED -->
## THE WIDTH TABLE — read this before touching a width

*(PROTECTED: see the top of this file. Do not cut or condense without asking.)*

**THE ARMS DO NOT SHARE A DECLARED WIDTH. Each arm declares the width that suits
ITS OWN `datawidth`, and no arm has to be legal for any other arm's `datawidth`.**
This has been got wrong in five separate sessions. It is prompt_2.md's table and it
is applied automatically, on every run, by `arch.patch._set_weight_geometry()` from
`eccenergy/physics/widths.py`:

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
`python3 -m eccenergy.physics.widths`.

**DEPTH is shared and is the only thing `assert_pair_geometry()` checks.**
`depth' = round(depth × width / 96)` is computed at the BASE width, not the arm's
own, so every arm declares the same depth — which is what makes the `eff. capacity`
column above come out as `(W_arm/q)/(96/8)`, and what leaves the depth check
something real to check. `ECC_DISABLE_ASSERT_PAIR_GEOMETRY=1` (env.sh §5, default
`0`) turns that depth check off for a study that varies depth between the arms on
purpose. It is the only thing that knob disables — **there is no width check for it
to disable**.

Three assertions, each with a mutation test (`test_dilation.py`):
`arch.patch.assert_pair_geometry()` (both arms declare the same LEVELS at the same
DEPTH — never the same width, see above), `study.narrowing.assert_onchip_narrowing_once()` (the
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
Nothing but `toolchain/results_store.py` writes one. Schema: `docs/RESULTS_SCHEMA.md`.

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

## `arch/weight_path.py` + `arch/placements.py` — the placement space

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
layout, not assumed), and **group residency** (how many weights a PE-local
boundary's level holds against `G_rec`).

**GROUP RESIDENCY IS REPORTED, NOT REFUSED** (`ECC_RECON_REQUIRE_GROUP_RESIDENCY`,
default `0`, decided 2026-09-13). `G_rec` = 9 at BCH(63,·) over 8-bit weights:
63/8 = 7.875 is not whole, so a codeword drifts across weight boundaries and the
worst-aligned one reaches into `ceil(63/8)+1` weights. The old refusal assumed an
engine that can only rebuild from weights co-resident **at one instant**; RECAP's
accumulates the retained bits as they arrive, so a level holding 6 — or 1 — still
feeds it, over more accesses and with more buffering. A small tile is a COST, not
an impossibility. **The number is still measured and still on every bar's record**
(`layers_below_G_rec`, `infeasible_layers`, `group_residency_note`) because it
bounds the buffer the engine needs; set the knob to `1` for the conservative
reading. Measured: this recovers R2/R3/R4/R5a on mobilenet_v2 (1, 1, 4 and 14
layers of 31 below `G_rec`) and changes resnet18 by **nothing** — no layer of it
is below `G_rec`.

The DRAM term is ONE stage and the whole of it is reducible: only the k message
bits are read out and driven off the die. `ECC_DRAM_PJ_PER_BIT` is the per-bit
cost; `ECC_MAC_PJ_OVERRIDE` rescales the Compute category evaluator-side, after
the raw cache, so no mapping moves. Both carry their citation from
`provenance.yaml` onto every figure and result.

**Every result prints the ceiling first** — the weight energy a boundary could
reduce, times `1 − K/N`. Without it a sub-percent saving reads as a missing term
rather than as arithmetic.

## Architecture rules

**A LEVEL THAT DECLARES `n_banks:` IS INSTANTIATED AS `smartbuffer_SRAM_banked`**
(Phase C1.7), the only compound in the study that forwards the count to CACTI --
upstream's `smartbuffer_SRAM` drops it, so every `n_banks:` here was inert until
2026-09-13. **A level that declares none keeps the plain class and prices exactly
as it did**: timeloopfe hands EVERY storage level a default `n_banks: 2`,
published for none of them, and the CACTI wrapper's `depth >= 64 x n_banks` floor
would move a shallow array's price through the floor rather than through any
banking. CACTI rounds to the next power of two and drops its own `bankscale`
correction; `provenance.yaml` `sram_banking` records that rather than
compensating for it.

`archs/_shared/standard.yaml` states what must be IDENTICAL across designs;
`provenance.yaml` records where every declared number came from; `noc.yaml` holds
the interconnect coefficients and, per design, which spatial containers actually
are the NoC. Two NoC terms Timeloop cannot be given are charged after mapping by
`toolchain/noc_post.py`. `archs/_shared/components/` is in the mapper fingerprint:
editing a component colds every cache, on purpose. `bash run.sh validate` checks
all of it and exits non-zero on a violation.

Only one of the four `source:` values licenses using a design's name as the chip:
**`published`**. `derived` differs in one stated block, `reference_design` is
shipped by timeloop-accelergy-exercises and merely *named* after a paper,
`locally_authored` reproduces nothing. `simba_like` is `reference_design` — label
it "Simba-like (reference design)".

**Eyeriss v1 IS `eyeriss_like_wglb`** (decided 2026-09-10). JSSC 2017 publishes the
filter-weight allocation of the 108 kB GLB, so the file that models it is the
design; `eyeriss_like`, which declares `!Nothing` in its place, is retired — still
mappable by name for a diff, but no longer the design.

**THREE OF ITS DECLARED CAPACITIES DIVERGE FROM JSSC 2017 ON PURPOSE**
(confirmed and kept 2026-09-13, prompt_7 C1.7): `filter_glb` 2 kB against the
paper's 8 kB, `weights_spad` 32 weights/PE against 448, `psum_spad` 64 against
24. All three arrived in commit `3a770cc` with no note and in THIS FILE ONLY —
`archs/eyeriss_like/arch_paper.yaml` still carries 224 and 24, so the two v1
files do not declare the same PE. The divergence table heads the arch YAML and
`provenance.yaml` marks each `diverges_from_paper: true`. **Quote the design's
name for its STRUCTURE, never for its capacity.** The v1 bracketing-pair
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

- **Frozen**: `physics/parity.py`, `study/baseline.py` and `external_parity()` are not
  refactored. `build_stacks()` **is no longer frozen** — prompt_6 RULE 3 changes
  its reconstruction term. After any change there, re-run `bash run.sh baseline
  --eval` and diff: Task 1 and 2 totals must not move.
- **Never** read `os.environ` outside `settings/env.py`, and never resolve a path
  outside `paths.py`. **`dataclasses.replace(cfg, ...)` no longer works** --
  a Config is six frozen groups, so it is `cfg.with_(knob=value)`, which
  re-runs every check and every derivation exactly as `__post_init__` did.
- **A COMMENT IS NEVER A DECLARATION. Read geometry through
  `arch.patch.uncommented()`, write it through `arch.patch.write_attr()`** (2026-09-13).
  Every geometry regex here used to match the raw text, so a `depth:` written
  in a COMMENT was read as the attribute — and `re.sub(..., count=1)` then
  rewrote the COMMENT and left the attribute alone. One explanatory comment
  ("the paper's two banks would be `depth: 1024`") made `filter_glb` declare
  98,304 bits where 16,512 were intended: a six-fold capacity error on the
  level R2's whole saving rides on, with one line of stdout to say so.
  `uncommented()` masks comments while PRESERVING POSITIONS, so a caller
  searches the masked copy and splices into the real text; `write_attr()`
  RAISES when the rewrite lands on nothing, which is the silent half.
  **Adding a regex over an arch YAML? Mask first.**
- **A `def` inside `__init__` ENDS `__init__`.** A `@property` added in the
  middle of `toolchain.invoke.Mapper.__init__` orphaned every line after it, `self._memo`
  included, and killed 72 SLURM jobs ten seconds in — because no test had ever
  CONSTRUCTED a Mapper. `tests/test_phase_c.py` does now. Any class whose
  constructor sets state the hot path depends on needs one.
- **Handing Timeloop an energy table** (prompt_6): pass `ERT:` and `ART:` YAMLs
  beside `design_inputs()`, and pre-write them into the output directory as
  `timeloop-mapper.ERT.yaml` / `.ART.yaml` first -- with a supplied table Timeloop
  writes neither, timeloopfe's parser then raises after a successful search, and
  `Mapper` counts the shape failed. `toolchain.ert.ErtTables` is the production hook
  (base table from Accelergy once per arm under `<cache>/_ert/`, two rows bumped,
  staged per shape, read back after the map); `python3 -m
  eccenergy.toolchain.ert_probe` is the proof (FINDINGS §3.5) and writes under
  `paths.ert_probe_dir()`, never the mapper cache.
- **An ERT arm is a configuration, not a design.** `ECC_RECON_ERT_ARM=<placement
  key>` (set per job by `hpc/map_ert_arms.sh`) resolves in `config.py` into
  `datawidth: q` on the storage levels of that placement's `reduced` set and,
  where the boundary has one, an ERT bump derived by `arch.fingerprint.ert_bump()`. The arm
  is in the cache slug AND the fingerprint, and every entry's stored ERT is read
  back before a number is used: two arms with byte-identical YAML must never
  share a directory. **Since Phase C1.6 the REFERENCE arm has a supplied table
  too** -- no bump, only `ECC_MAC_PJ_OVERRIDE` on every `compute` row, `set`
  rather than `add`, so `apply_mac_override`'s ratio is exactly 1.0 and the
  mapper and the report price one MAC. `Mapper.supplies_ert` is the test, never
  `ert_bump is not None`. `ECC_RECON_ERT_AWARE=1` (needs `RECON_OPTIMIZER=True`,
  `ECC_PHASE=Post`) bills each bar from its own chip's plan; the toll Timeloop
  billed inside the level is MOVED into `Reconstruction`. Timeloop prints leakage
  outside the per-dataspace energies and the raw record never held it, so only
  the access toll is in the bill; the idle term is verified against the stats'
  leakage and charged by the evaluator.
- **THE ARMS TO MAP ARE THE DISTINCT CHIPS, not the ERT-injectable boundaries**
  (prompt_7 Phase B, 2026-09-12). `arch.arms.mapper_arms(arch, cfg)` is the reference
  plus every boundary that differs on any of three axes -- `datawidth: q`, the
  ERT bump, the declared per-dataspace bandwidth scale (a NETWORK stage is
  carried in that set and marked `no-op`, because Timeloop has no network timing
  model but a boundary that adds one is still a different declaration). Six on
  `eyeriss_like_wglb`, five on v2. `arch.arms.ert_arms()` keeps its older, narrower
  meaning -- which boundaries have an ERT-injectable encoder -- and is a strict
  subset. Never `if key == "recon2"`. An arm with a bump keeps prompt_6's
  `ert-<key>-<level>-<action>` slug byte for byte; one without gets `arm-<key>`,
  which is what stops R1 (whose patched YAML IS the reference's until Phase C1.2)
  sharing the reference's directory.
- **EVERY BAR NAMES THE PLAN IT WAS BILLED FROM** (`billed_from` on its record,
  and on the figure). `arch.arms.plan_assignment()` derives it: its own arm if that
  arm is mapped, else the OUTERMOST mapped arm in path order that narrows exactly
  the same storage levels, else the reference plan flagged
  `geometry_matches: false`. A plan is only valid for a bar when the narrowed
  levels match -- `datawidth: q` changes the words the loop nest moves and no
  post-processing can re-tile a loop nest. So `reference` is only ever right for
  R1, R3 borrows R2's plan (with R2's toll already moved out of the bill, so R3
  pays the un-bumped price of the shared geometry), and R5a is billed from the
  reference and FLAGGED as a chip that has never been mapped until Phase C maps
  it. A HALF-mapped arm is refused, not borrowed: that would be two chips in one
  bar. `ECC_RECON_ERT_AWARE=1` with no arm mapped at all is also refused -- that
  is Task 3 under a heading that says otherwise.
- **`ert_injectable()` condition 3 is derived from the CLOCK GATING**, not from a
  key name (prompt_7 6.3). The bump has two rows and the exclusion holds only
  when BOTH are constant across the mapspace: the access row on the innermost
  weight level's `reads` is the MAC count and is, but the per-cycle `leak` row is
  `idle x (1 - g)` and Timeloop bills it as `leak x utilized instances x cycles`
  -- both the mapper's choice. So R5a IS ERT-injectable at
  `ECC_RECON_CLOCK_GATING_PCT` 0 and 99.5, and is not at 100.
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
- **Adding an architecture**: **one directory, and nothing else** (phase 5).
  `make arch NEW=<name>` scaffolds `archs/<name>/` with five files —
  `arch_paper.yaml` (the chip, every number cited), `design.yaml` (label, axis
  order, the constrained mapspace, and any per-design modelling fact), plus
  `weight_path.yaml` and `placements.yaml` if it declares reconstruction
  boundaries, and `README.md`. **The stub is deliberately INVALID**: a `TODO`
  left in an `evidence:` or a `description:` is refused by name, because a
  design whose weight path nobody wrote would still report savings. Then
  `bash run.sh validate` and `diagnose` before committing to a long sweep, and
  add its entries to `standard.yaml` / `provenance.yaml`.
  **No Python file names a design** — `tests/contract/test_designs_are_data.py`
  holds that at zero, and `arch.design.flag(arch, "<field>")` is how a
  per-design fact reaches the code. `KNOWN_ARCHS`, `ARCH_LABELS` and the
  constrained mapspace are read from the directory.
  **Declare each weight level's PUBLISHED `width`/`depth`/`datawidth` and stop
  there** — do not hand-pick a width to suit a code, and do not check one against
  another arm's datawidth. `arch.patch._set_weight_geometry()` reshapes every weight
  level per arm from THE WIDTH TABLE (above) and renormalises depth to hold your
  declared total bits, so `width % datawidth == 0` is satisfied by construction at
  every code. If a `width 64 % datawidth 5 != 0` ever reaches you, the fix is in
  `physics.widths.WIDTH_TABLE`, never in the arch YAML and never in the config.
- **Adding a model**: add to `CNN_MODELS` or `TRANSFORMER_MODELS` in `config.py`,
  then `python3 -m eccenergy.arch.generate models <name>` in the container.
- **Adding an architecture to the placement study**: `weight_path.yaml` and
  `placements.yaml` **in that design's own directory** — they are loaded
  together or not at all, which is what makes "the two tables must be edited
  together" a property of the directory rather than a warning here. Get the
  stage-to-level match right by reading a real `timeloop-mapper.stats.txt` AND
  `timeloop-mapper.map.txt` from that design's cache. Then the name in
  `ECC_RECON_PLACEMENTS` in env.sh §4.
  `test_every_placement_space_is_valid_for_every_supported_design` covers the
  new design automatically, and the schema refuses a boundary that names a stage
  the weight path does not declare.
- **Adding a sweep axis**: a name in `SWEEPS`, a stem in `SWEEP_STEMS`, resolution
  in `Config.__post_init__`, and a `_<name>_groups()` in `report/sweep.py`. Do
  not write a second renderer.
- **Changing how a bar looks**: `draw_panel()` in `report/stacked.py` is the only
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
- **TIME IS IN THE ARCHITECTURE SINCE PHASE C1** (2026-09-13). The DRAM level
  declares `shared_bandwidth` (**not** `read_`+`write_bandwidth` -- the DQ bus
  is ONE wire set whose limit is on their SUM, which is what the roofline
  charges); each arm's boundary declares
  `per_dataspace_bandwidth_consumption_scale` (`K/N` at DRAM, `q/8` on chip --
  they differ by 5% and one factor everywhere is a silent inconsistency); a
  level the arm narrows declares its port `x 8/q`; and the design runs at its
  own `ECC_ARCH_CLOCK_MHZ` through `globals_<arch>.yaml`. All four are in the
  patched YAML and therefore in the fingerprint.
- **WHO APPLIES THE OFF-CHIP WEIGHT RELIEF IS MEASURED PER BAR**
  (`toolchain.latency_post.relief_owner`, prompt_6 RULE 1). The evaluator applied `K/N`
  from Phase A; C1.2 makes the MAPPER apply it, and both would give K/N
  SQUARED. The answer is on disk in each bar's own stats -- `Bandwidth
  Consumption Scale` is 1.00 where nothing was declared and K/N where it was --
  and a third value is REFUSED as a bar billed from another code's plan.
- **`ECC_RECON_IDLE_PJ` IS RESCALED TO THE DESIGN'S CLOCK, ONCE**
  (`Config.dc_idle_scale()` owns the factor, `study.stacks.load_recon_terms()` applies
  it after the three lookup branches converge). It is pJ per cycle at DC's 1 ns
  clock and it is CLOCK power, so 200 MHz is **x5** -- on the reconstruction
  engines only. `ECC_LEAKAGE_NW` is POWER in nW and must NOT be rescaled; that
  is why the two are declared in different units. env.sh section 6 TRAP 2 had
  documented this since before it existed in code.
- **TIME and STANDBY POWER are still evaluator-only in the REPORT, and the
  re-timing is not in the fingerprint** (prompt_7 Phase A). `toolchain.latency_post.
  roofline()` re-times a mapping Timeloop already chose -- `max(compute, each
  level's declared-bandwidth limit, off-chip items / ECC_DRAM_BANDWIDTH_MBPS)`
  -- and at an unlimited off-chip limit it reproduces Timeloop's per-level AND
  total cycles EXACTLY on all 43 cached shapes, which is the gate that stops it
  inventing time. `ECC_LATENCY_MODEL` defaults to **1** since Phase C1: the
  mapper now optimises against the same declared limit, so the roofline
  re-states a plan's time instead of being a second timing model. `ECC_STATIC_ENERGY=1` then charges
  `ECC_LEAKAGE_NW x stored bits x utilized instances x cycles` as a `Standby`
  category **in `Raw.base`** -- which every arm and every placement bar starts
  from, so no code path can charge it to one arm and not another. With both
  knobs off, `phys_cats()` is byte-identical to what predates them and every
  earlier total reproduces to the pJ. Densities are POWER in nW and the cycle
  period is applied once, in `standby_energy()`. `Raw.standby` reports what
  TIMELOOP billed for leakage beside it: the mapper's own objective always
  contained it and only the report dropped it (prompt_7 rule R-4).
- **Line endings**: the shell scripts run inside a Linux container and a CRLF makes
  bash die on `set -o pipefail` with a mangled message. `.gitattributes` forces LF;
  `bash tools/fix-eol.sh` repairs anything that slips through. **Always emit LF.**

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
    eccenergy/          LAYERED SINCE 2026-09-13 (ProjectRestructure phase 2),
                        and the six big files CUT ALONG THEIR BANNERS (phase 3).
                        A module may import only from LOWER layers;
                        tests/contract/test_layer_rule.py enforces it and lists
                        every edge that still points the wrong way. SINCE PHASE
                        4 THERE ARE NONE: the list is empty and a new upward
                        import fails the suite where it is written.
                        Each module's own docstring says what it is; this is
                        only where to look.
      L0  paths.py      the ONLY resolver of paths
          settings/     THE KNOBS (phase 4): six FROZEN dataclasses --
                        run code arch mapper recon energy -- plus env.py, the
                        ONLY reader of os.environ, and banner.py. Each group
                        declares IN_FINGERPRINT: which of its fields reach the
                        mapper. It imports nothing above itself
          contracts/    the shared types. errors.py (ConfigError) since phase 4
      L1  physics/      parity.py  embedded.py  widths.py (THE WIDTH TABLE)
                        baseline_dram.py (and the per-bit DRAM price read off a
                        record), packing.py (stream vs aligned),
                        granularity.py (G_rec, engines, engine_cycles),
                        recon_dc.py (the DC datapath's two measured terms --
                        the one module here that reads a file, on purpose)
      L2  arch/         load.py  patch.py  fingerprint.py  layout.py
                        validate.py  workloads.py  generate.py
                        design.py -- the ONLY reader of archs/<name>/*.yaml
                        weight_path.py  placements.py  arms.py -- the stages,
                        the boundaries, and which boundaries are DISTINCT CHIPS
      L3  config.py     THE RESOLVED CONFIGURATION: the six groups checked
                        against the designs they name, and the ONLY place MHz
                        becomes seconds. Frozen; `cfg.with_(...)` is the only
                        way to a different one, and it re-runs every check.
                        ABOVE arch/, which is what ended the import cycle
      L4  toolchain/    inputs.py (what Timeloop is given), ert.py (a supplied
                        ERT/ART, read back), cache.py (ShapeLock: one entry, one
                        writer), invoke.py (Mapper -- the ONLY module needing
                        the container), stats.py (parsing its output),
                        weight_stats.py (the weight path read back out of it),
                        noc_post.py, latency_post.py (the roofline AND the ONE
                        owner of the clock period every per-cycle term is
                        charged over), results_store.py (the ONLY writer of an
                        evaluation JSON), ert_probe.py
      L5  study/        stacks.py (build_stacks), energy.py, common.py, the
                        drivers baseline/embedded/validate/audit/diagnose, the
                        placement study (placement_study, placement_eval,
                        placement_tables, placement_notes, ert_view,
                        dilated_view, narrowing) and Task 4 (dilation,
                        dilation_cache, dilation_tables, capacity)
      L6  report/       stacked.py (draw_panel: the ONLY place a bar is drawn),
                        panels.py, style.py, and THE DRIVERS THAT DRAW:
                        sweep.py, recon_view.py, dilation_view.py
          __main__.py   the CLI. May import anything
    Makefile            make arch NEW=<name> | make layers | make test | make gate
    archs/<name>/       ONE DESIGN, ONE DIRECTORY (phase 5): arch_paper.yaml
                        (the chip), design.yaml (label, axis order, the
                        constrained mapspace, per-design modelling facts),
                        weight_path.yaml + placements.yaml (the placement
                        space, loaded together or not at all), README.md.
                        arch/design.py is the ONLY reader. No Python names a
                        design
    archs/_shared/      standard.yaml, provenance.yaml, noc.yaml -- not an
                        architecture; skipped by the installer.
                        components/ is IN THE FINGERPRINT: regfile_decoded.yaml
                        (the address-decoded RF) and smartbuffer_SRAM_banked.yaml
                        (the only compound that forwards n_banks to CACTI). A
                        local file may never redefine an upstream class name --
                        that is a duplicate-class error, not an override.
    ecc_energy_study/   AUTO-MANAGED: cloned repo + mapper cache. Do not delete.
    plans/              every plan and prompt this study has been given, in
                        sequence. prompt_7.md is the LIVE one; ProjectRestructure.md
                        is the (finished) reorganisation. Nothing reads these
                        programmatically -- they are cited by name from docstrings.
    plans/specs/        the three source SPECS the model is built against --
                        01_project_context_and_architectures.txt,
                        02_reconstruction_dse_and_implementation.txt,
                        04_results_storage_spec.txt. ~20 production docstrings
                        cite them by name; they are provenance, not prose.
    tools/              gen_guards.py (make guards), scaffold_arch.py (make arch),
                        bch_sweep_tables.py (prompt_5's tables, cache-only),
                        fix-eol.sh (CRLF repair). Each runs on its own; none is
                        imported by the package.
    legacy/             pre-rewrite material, plus the dated FINDINGS_detail_*,
                        progress_* and PROJECT_STATUS_* snapshots that hold the
                        full working record. Also docs/RESULTS_SCHEMA.md, the
                        two archived NoC runs, and preC-snapshots-*.tar.gz (the
                        19 hand-made pre-edit backups, archived 2026-09-13).
                        Nothing imports it.
