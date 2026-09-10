# CLAUDE.md

Energy modelling for relaxed error correction in DNN accelerators, on Timeloop +
Accelergy. The question: **if ECC parity does not have to be stored in DRAM, how
much inference energy does that save, and does the answer depend on the
accelerator?**

**The live plan is `prompt_2.md`** (rewritten 2026-09-10: the weight-memory depth
sweep, the per-code `datawidth` table, and Eyeriss v1 = `eyeriss_like_wglb`).
Read it before starting work; where it and this file differ on what to do next,
it wins.

**No empirical claims in this file.** Numbers live in `FINDINGS.md`; the live
caveat list is `bash run.sh diagnose`, computed from the architectures as they
stand. A prose copy here went stale once and then contradicted the code.
`legacy/` is pre-rewrite and describes nothing current.

## Running it

`env.sh` is the only file you edit: every `ECC_*` knob, ten commented sections,
sourced by `run.sh`, `hpc/run_all.sh`, `hpc/map.sbatch` and `hpc/tl.sh`, so a
value cannot mean one thing to the mapper and another to the evaluator. Sections
1-9 are knobs; section 10 is derived (it turns the `ECC_ARCHS`/`ECC_MODELS`/
`ECC_KS` lists into the swept-list-plus-two-constants form `config.py` reads,
picks the stem, and generates the SLURM task list). Read env.sh for what a knob
does — it is commented; do not restate it here.

Every value is written `${VAR:=default}`, so **the environment wins over the
file** and a one-off never needs an edit.

    bash hpc/run_all.sh          # THE command: map array -> dependent eval+plot
    bash hpc/run_all.sh --map-only | --eval-only | --replot | --local
    ECC_RECON_MODELING=1 bash hpc/run_all.sh --eval-only     # Task 3

`run.sh` runs ONE stage of that and has no knobs of its own: `map`, `baseline`,
`embedded --eval`, `recon --eval`, `dilation`, `validate`, `diagnose`, `panels`,
`--replot`, `--dry-run`.

**Only `timeloop.py` needs the container.** `--eval` (`ECC_FROM_CACHE=1`) never
invokes Timeloop; validate, diagnose, replot and the whole `eccenergy/tests/`
suite run on any python with pandas, matplotlib and pyyaml. `test_recon.py`
additionally re-runs its property tests against the real mapper cache when
pandas and that cache are present, and says so when it skips them.

**HiPerGator** (the machine this now lives on): `module load apptainer`, then
`bash hpc/tl.sh <any run.sh command>`. Never run the mapper on a login node —
`srun --account=rewetz --qos=rewetz --cpus-per-task=18 --mem=8gb --time=02:00:00 --pty bash -i`,
or `sbatch`. Keep `ECC_MAPPER_THREADS=18` and `--cpus-per-task=18`: the thread
count is in the mapping fingerprint, so any other value is a cold cache.
`hpc/HIPERGATOR.md` has transfer, image build, verification and the cost model.
On a laptop the same image runs under Docker; in Git Bash use
`MSYS_NO_PATHCONV=1` and `pwd -W`, or the bind mount and `-w` are rewritten into
nonsense and it looks like Docker is missing when it is not.

**A cold map is hours.** Narrow before widening
(`ECC_ARCHS=<one> ECC_MODELS=<one> bash hpc/run_all.sh --map-only`) and watch
the filesystem rather than a pipe — a backgrounded `docker ... | grep` buffers
until the pipeline ends and looks hung when it is fine.

**`hpc/map_by_shape.sh` is how a cold design is mapped without waiting a day.**
It submits one job per layer SHAPE instead of one per model, so the hours become
one wave, and it takes several architectures at once
(`ECC_RECON_ARCHS="a b" bash hpc/map_by_shape.sh`): shapes × designs jobs and
ONE dependent eval after all of them, which is what a two-panel `ReconSweep.png`
needs — one launcher per design would give each its own eval and the second
would overwrite the first's figure. `--no-eval` submits the maps only;
`ECC_LAYERS` maps exactly those layers, which is how a design is spot-checked on
one or two shapes before its other ten are queued (the cache is keyed by shape,
so a spot-check's mapping IS the entry the whole-model run reads). Each
submission snapshots the SLURM task file, because `map.sbatch` resolves its
(arch, model) pair from it when the job RUNS.

## One sweep, two constants

All three ECC arms (`baseline`, `embedded`, `recon`) are always drawn, so the
arms are never an axis. A run sweeps exactly ONE of the remaining three axes and
holds the other two at a constant:

| `ECC_SWEEP` | x axis | sweep list | held | stem |
|---|---|---|---|---|
| `bch` | BCH(63,K) | `ECC_SWEEP_KS` | arch, model | `BCHsweep` |
| `model` | networks | `ECC_SWEEP_MODELS` | arch, K | `ModelSweep` |
| `arch` | accelerators | `ECC_SWEEP_ARCHS` | model, K | `ArchitectureSweep` |

`cfg.archs`/`cfg.models`/`cfg.code_k` are derived in `config.py` from that
choice; nothing below `config.py` except `sweep.py` knows which axis is swept. A
model sweep is all-CNN or all-transformer — mixing families is a config error,
they come from different workload files.

**`ECC_EXPERIMENT=panels`** draws one image with one panel per
`ECC_PANEL_MODELS`, repeating the same sweep inside each. It adds no axis and no
renderer, exists only for the "does the ranking survive changing the network"
question, and refuses `ECC_SWEEP=bch` (that would vary two axes between panels).
Panels share a legend, categories and unit but deliberately **not** a y limit,
and the figure says so on itself.

**`ECC_RECON_MODELING=1`** (env.sh §4) is not a fourth sweep: its axis is WHERE
on the weight path the reconstruction boundary sits. §4 overrides §3, collapsing
the run to one model and one code, stem `ReconSweep`. The boundaries of two
designs never share an x axis — the three sweeps can put architectures on an
axis because every design has all three ECC *arms*, but a reconstruction
*boundary* is design-specific, and "reconstruct after the mesh" beside a design
with no mesh is meaningless.

**`ECC_RECON_ARCHS` is therefore ONE PANEL PER NAME**, top panel first, in one
`ReconSweep.png`. Each panel keeps its own x axis of its own boundaries and its
own two reference bars, so a percentage on one panel says nothing about the
other; what the panels share is the page, the legend, the category set and the
energy unit, and each heading names its design. One name draws exactly the
single-panel figure this study has always drawn (verified byte-identical when
the panelling went in). `plots/panels.stacked_panels` renders it — the same
routine the two-model sweep figure uses, widened with `draw_panel`'s bar
parameters rather than forked. Every design named needs its own mapper cache at
the current fingerprint; a missing one stops the run and says which.

## Task 4 — the mapping itself, not just the boundary

> **DIRECTION CHANGED 2026-09-10 — read `prompt_2.md` before using this section.**
> Everything below describes capacity dilation (`ECC_WEIGHT_CAPACITY_SCALE`),
> which **answered negatively** (FINDINGS §7.7, §7.8): `weights held` never moved
> over ranges up to 32×. The cause is arithmetic, not search — resident tiles are
> integers, growing a level's share of `C` is a **2× step**, and N/K = 1.6154
> delivers 117 weights/PE against the 128 needed.
>
> The replacement expresses the reduced representation as **`datawidth` on the
> weight levels, at FIXED `width` and `depth`**. VERIFIED 2026-09-10: CACTI
> receives `depth` and `width` only — `datawidth` never reaches the energy model
> (`accelergy.log`: `Calculated storage."width" as "width"`) — and Timeloop bills
> `vector_access_energy / block_size` per weight, `block_size = width/datawidth`.
> So the reconstruction arm gets more effective capacity at **byte-identical**
> per-access read/write/leak. That is the fairness condition dilation could never
> meet, and it makes `capacity_dilation_correction()` unnecessary rather than
> merely imperfect.
>
> **HARD CONSTRAINT: `width % datawidth == 0`, or `timeloop-mapper` aborts** —
> `buffer.cpp:302`, `Assertion 'width % (word_bits * block_size) == 0' failed`,
> measured. `block_size` defaults to 1 and is then checked, so there is **no
> floor path**: a partially-filled word cannot be modelled at all.
> `archs.py:1147` pre-checks it. Per-code widths are tabulated in prompt_2.md.
>
> The dilation machinery below is kept: it is still the right `depth:` sweep
> axis, and §7.7/§7.8's guards against reading a sweep wrong all still apply.

`RECON_OPTIMIZER=True` (with `ECC_PHASE=Post`) is Task 4 and is implemented
since 2026-09-09. It keeps every bar and every boundary of Task 3 and changes
one thing: the reconstruction bars come from a SECOND mapping, solved against
`ECC_WEIGHT_CAPACITY_SCALE × N/K` weight room instead of the reference's. The
reference bars stay on the reference mapping. So the two arms no longer refetch
identically — which is the one thing Task 3 structurally cannot show, because
`RECON_OPTIMIZER=False` pins one mapping on every arm. Stem
`ReconSweep_optimiser`, and a layer-scoped run keeps the suffix too, so it can
never land on the fixed-mapping figure.

**`ECC_WEIGHT_CAPACITY_SCALE` is the whole mechanism.** It multiplies `depth:`
on the WEIGHT-carrying storage levels of the architecture the mapper sees.
Weights-only levels by default (`ECC_WEIGHT_CAPACITY_SCOPE=exclusive`); a level
holding Weights beside another dataspace is a bracket, not a fact, and needs
`=shared`. A declared `depth: 1` register is never scaled under either — it is
a pipeline latch, and scaling it would invent a per-PE reuse level the design
does not have. Every scale is its own mapper cache (`wcap<scale>`), because
Task 4 is the DIFF of two mappings; a scale that rounds every weight depth back
to its declared value is not a dilation and keeps the undilated cache. The
scale is quantised to four decimals so one capacity has one spelling — 63/39
written `1.61539` by python and `1.6154` by the shell is the same architecture
filed under two directory names, and the evaluator then refuses a cache it has.

**Three artifacts of expressing capacity as depth, and what is done about each.**
Accelergy costs a level from its declared geometry, so `depth × N/K` is modelled
as physically bigger silicon — which the reconstruction arm does not have.
(1) Per-access energy rises (1.18–1.46× on these designs); `recon.capacity_dilation_correction()`
re-prices reads and writes SEPARATELY at the declared array's energies (they do
not scale together) as an access-weighted ratio, so the block size cancels, and
refuses if the split does not reconcile with Timeloop at a whole number of
values per physical word. (2) Timeloop derives the NoC hop length from the inner
level's Accelergy AREA wherever `noc.yaml` pins no `tile_width_um`, so the wires
into it lengthen too; that is NOT corrected and is recorded as a caveat.
(3) The MAPPER optimised against the dearer array, so it had a reason to leave
the room unused — every Task 4 saving is a lower bound on all three counts.

**When the two nests come back byte-identical the reference record is used
outright**, and the run says the dilation bought nothing. That is not a
shortcut: on an identical nest the reconstruction arm IS the reference mapping
on the same silicon, so everything that differs between the two cached records
is one of the artifacts above. Task 4 then reproduces Task 3 to the digit, which
is the property that makes a zero result trustworthy.

**Task 4's checks are its own** (`task4_checks`), not Task 3's. Task 3's
`dram_scaled_by_K_over_N` is too loose here by design — the reconstruction arm
also issues fewer reads and legitimately pays less for them — so it is replaced
by the tighter `dram_credit_equals_the_reads_the_mapper_removed`.
`both_arms_are_the_same_workload` compares the MAC counts, which are
mapping-invariant, and `reconstruction_arm_has_N_over_K_more_weight_capacity`
reads the delivered capacity off both mappings rather than off the YAML.

**Step 1 comes first, and it is a separate tool.** FINDINGS §9.0 requires the
capacity assumption be MEASURED before it is modelled. `hpc/map_capacity_sweep.sh`
fills the caches (one job per design × layer × capacity, no dependent eval) and
`bash run.sh dilation` diffs the per-layer DRAM weight reads. Its `binds` column
is the one that matters: a dilation enlarges the weight buffer only, so a
mapping whose weight buffer is at 21 % while its ifmap scratchpad is at 100 % has
nothing to gain, and that is a property of the DATAFLOW rather than of the code
rate. FINDINGS §7.8 records what it found.

`--table` is the mode a Task 4 result is READ off — one row per swept capacity,
per design, per layer. It maps nothing, and it needs env.sh's configuration, so
run it as `bash hpc/tl.sh python3 -m eccenergy.experiments.dilation --table`;
bare `python3 -m …` silently takes `config.py`'s defaults (`energy`, victory
500) instead, and a Task 4 comparison under a serialising objective is invalid.
The table says so in a banner when the objective is not `edp`.

**`weights held` is the column that decides whether a dilation did anything**,
and `room`/`held` sum EVERY weight-carrying level so a two-level design is not
described by its scratchpad alone. Three verdicts guard the reading, and each
exists because it caught a reported positive (§7.8):

* `capacity` — reads FELL and `held` ROSE. This is the hypothesis, and it is
  the only label that may be quoted as one.
* `PERM?` — reads moved while `held` is identical at every weight level.
  Nothing extra was stored, so the cause is the loop nest's ORDER, which the
  search can reach at the declared capacity too. Weight-stationary's
  "2.00× → 1.00×" was this: one swap of `for Q` and `for C` at the DRAM level.
* `ORDER?` — reads fell while `held` **fell**. The arm stored strictly less on
  chip and read less, which extra room cannot explain. `capacity` requires
  `held` to RISE strictly; omitting that check produced a fourth false positive
  on the shared-scope caches (FINDINGS §7.8).
* `PE!=` — the arms differ in PE count, so capacity and parallelism moved
  together and neither is separable. This is §7.7's withdrawal in a column.

**A comparison across two `fp-<hash>` directories is a comparison of two
ARCHITECTURES**, not two capacities. `sibling_fingerprints()` enumerates every
solved fingerprint under one variant slug, marks the one the current
configuration reads, and the table prints a `!! FINGERPRINT WARNINGS` block.
Eyeriss v2's reported "14.00× → 4.00×" was a stale sibling: the current
fingerprint already refetches 4.00× undilated.

**`ECC_WEIGHT_FACTOR_RELAX` is the second lever** (env.sh §4). Capacity makes
the buffer bigger; it does not make the mapper able to SPEND it. It drops the
`factors:` pins on the weight-indexing dimensions (M, C, R, S) of the temporal
constraints on weight-carrying levels — N, P and Q keep theirs, since weights do
not index them. Own cache (`wrelax`) and fingerprint, same no-op rule as `wcap`,
and **it is a different DATAFLOW**: Eyeriss v1's `M=1` at the filter spad IS
row-stationary, so a `wrelax` run must never be quoted as the published chip.
It did not make any dilation pay — but at the SAME capacity and the SAME
168/168 PEs it cut `eyeriss_like` `layer3.0.conv1`'s refetch 14.000 → 2.000 and
its energy 417 → 267 µJ, so it says where the DRAM weight traffic really comes
from.

**Refetch is set by loop ORDER at the DRAM level, not by capacity** — it is the
product of the DRAM-level loop factors that do not index Weights and sit
outside one that does (v1 `Q(2)×P(4)`, v2 `Q(4)`, WS `Q(2)`). A weight tile
cannot index P or Q, so a weight buffer INSIDE the PE array can never absorb
those loops however large it is made: v1 is flat at 8.00× to ×32 and 0.9 % fill.
Only a weight level ABOVE the array can, which is what makes the `_wglb`
variants and `ECC_WEIGHT_CAPACITY_SCOPE=shared` the levers rather than a bigger
scratchpad.

## Results layout

    results/_raw/<arch>/<treat>/fp-<hash>/<workload>/<scope>/cls-<mode>/rw-<split>/<model>.json
    results/evaluation/{Pre|Post}/<arch>/<model>/<bch>/<prec>/<scope>/<mapper>/<runid>.json
    results/{figures,tables,manifests}/<stem>.{png,pdf|csv|json}

**Three sweeps, three names, and that is the whole of `figures/`, `tables/` and
`manifests/`.** The stem comes from the configuration alone, so re-running at
different constants REWRITES the file instead of adding one; the manifest beside
it records the constants, code geometry and mapper fingerprint behind what is on
disk. Copy a figure out, or point `ECC_RESULTS_DIR` elsewhere, to keep it.
`panels` obeys the same rule (its stem carries the panel models).

`results/evaluation/` is the exception and accumulates on purpose: one JSON per
(phase, arch, model, code, precision, scope, mapper config, run), holding every
ECC variant including the ones not evaluated (`total_energy_pJ: null` plus a
reason). Nothing but `results_store.py` writes one. Schema:
`docs/RESULTS_SCHEMA.md`.

## The two caches

1. **Mapper cache** — `ecc_energy_study/outputs/<arch>/<treat>/fp-<hash>/<shape>/`,
   one entry per (architecture, layer shape). **Never delete it**; it is hours of
   compute and every solved shape is saved immediately, so runs resume.
   `fp-<hash>` is a hash of the patched YAML the mapper actually sees plus the
   globals and every mapper setting, and each entry carries a `mapping.json`
   sidecar that results reference. A different fingerprint is a MISS — before
   this existed, editing an arch.yaml left the path unchanged and the next run
   reported old mappings as the new design. Entries with no sidecar (pre-Task-1)
   are refused unless `ECC_CACHE_STRICT=0`, which labels the result `legacy`.
2. **Raw energy cache** — `results/_raw/`, parsed from the mapper cache. Pure
   Timeloop output, independent of the ECC configuration, so changing the code
   geometry and redrawing is milliseconds (`ECC_CONST_K=36 bash run.sh --replot`).
   The one thing under `results/` worth keeping.

`ECC_RERUN_OPTIMISER=1` re-solves a shape on a valid hit and overwrites both
records; it is refused together with `--eval`. Anything that changes what the
mapper sees or optimises gets its own cache subdirectory: mapper settings and
`ECC_NOC` (a costed interconnect is a different architecture to the mapper);
`globals.yaml` knobs, which invalidate every design at once; and
`ECC_ARCH_FIDELITY` / `ECC_FORCE_DATAWIDTH`, evaluated per architecture by
`archs.effective_variant()`. So **anything touching env.sh §2/§5, or an arch
YAML, takes the whole matrix cold at once** — budget for it.

## The three ECC arms

One BCH(N,K) codeword over `ECC_WEIGHT_BITS`-bit weights under all three, so
they differ only in where the parity lives. Codewords are counted from DRAM
weight reads. The first entry in `ECC_APPROACHES` is the reference savings are
measured against.

- **baseline** — parity beside the data in DRAM; weight traffic inflates by N/K.
- **embedded** — parity inside the stored weights, laid out as the embedding
  pipeline (ECC-CODE-Engine / Input_Embedding) does it: the MSB-first weight bit
  stream cut into n-bit codewords, so weights straddle codewords and the n-k
  lowest-significance positions carry parity. DRAM traffic is exactly
  Timeloop's — the complete codeword is read for correction.
- **recon** — DRAM as embedded, K/N of the weights held on chip, the rest
  regenerated by a synthesized datapath charged per codeword from
  `data/dc/BCH_N63_results.json`.

`build_stacks()`'s `recon` arm is ONE point applied to every design at once, and
no physical boundary does what it does (it both scales on-chip weight energy by
K/N *and* charges one reconstruction per DRAM codeword). It stays as the third
bar of the three sweeps; it is **not** one of Task 3's boundaries and must not be
quoted as one.

## `recon.py` — the placement space

Two tables define it and **must be edited together**:
`WEIGHT_PATHS[<arch>]` (the stages of that design's weight path, each naming the
Timeloop levels that are it) and `PLACEMENTS[<arch>]` (the boundaries: which
stages stay reduced, what drives the reconstruction count, whether a reuse
register is present).

**Four designs are registered, and they do not have the same boundaries.**
`eyeriss_v2_like` and `eyeriss_like` have five each — two network stages then
one PE scratchpad, and no weight GLB on either (v2's GLB banks are iact and
psum; v1's 8 kB filter allocation is a prefetch buffer the RS dataflow does not
need, and `eyeriss_like_wglb` is the bracketing variant that models it).
`simple_weight_stationary` has SIX, because it is the only design in the study
with both a weight global buffer above the network and a stationary weight
register below the scratchpad: §6.2's buffer-output row and its MAC-input row
both exist there. The MAC-input boundary is listed **so it can be reported
infeasible rather than omitted** — the register holds one weight and a rebuild
needs `G_rec` co-resident, so `feasibility()` rejects it and names the layers.
Its existing depth-1 `weight_reg` is a pipeline latch — the mapping fills it
once per read — not a reuse register.

**A NETWORK boundary's encoders run once per ARRIVAL, not once per injection**
(since 2026-09-09). R2 credits its network with carrying the reduced form, so
its encoders are at the network's *destinations*, and the count is Timeloop's
own destination-side arrivals — `Ingresses × Multicast factor`, summed exactly
off its `@multicast M @scatter S` breakdown. Charging the ingress count instead
paired a destination-side saving with a source-side cost, which is not a
placement: an encoder before the fanout makes that network full width, and that
boundary is R1. `ECC_RECON_ENCODER_SITE=source` reproduces the earlier numbers
for the diff, exactly as `ECC_RECON_DECODE_SITE=controller` does for the DRAM
change. It moved Eyeriss v2's R2 from +3.87% to +3.78% vs embedded and made
**R3 the best boundary instead of R2** — the mesh's arrivals equal the
cluster-local network's ingresses, so R2 and R3 pay the same encoder count and
R3 saves strictly more. Only network boundaries depend on the knob; R1 counts
DRAM codewords and every PE-local boundary already counts destination-side
scratchpad accesses. `multicast_chain()` records the identity that validates the
parse (a network's arrivals equal what the next stage takes in) on every result.

**R4b IS REMOVED (2026-09-10)** — `retention_stage()`, `retention_model()`,
`weight_loop_nest()`, `ReuseRegister` and the `ECC_RECON_REUSE_REG_*` knobs are
all gone; no boundary carries a per-PE register on any design. What settled it:
consecutive weight reuse is **1 on 20 of 21** resnet18 layers, so a latch
catches nothing, and a register that DOES pay would have to hold the whole inner
tile (16–384 weights) — a second scratchpad, not a latch. **R3 (reconstruct at
the PE weight-storage INPUT) is the result**: +11.07 % vs embedded on
`simple_weight_stationary`, +17.24 % on `eyeriss_like`. `Recon overhead` stays a
plot category and is structurally zero. **Do not reintroduce a
retained-reconstruction boundary without first re-measuring `consecutive_run` on
the target mapping** — that is the step the original audit skipped (FINDINGS
§7.1, archived). Rejected option on the record in `01_…` §2.3 and `02_…`
(REC-L5, §22).

Scoping trap, live for any future per-PE proposal: a retention claim must be
scoped to the storage stage it belongs to. Unscoped, `simple_weight_stationary`
would credit its `weight_reg` with reuse that lives in its `pe_spad` (tile 192).
Both eyeriss designs have one storage stage, so scoping is a no-op there.

That is enforced, not advised. `validate_placement_space()` requires a
placement's `reduced` set to be a **prefix** of the path's reducible stages in
path order, and every reducible stage to be reached by some boundary; it runs
before anything is evaluated. Without it, adding a stage to one table and
forgetting the other keeps every boundary below it reporting its own saving while
showing the new stage at full width — the whole list understated, nothing saying
so. `feasibility()` structurally cannot catch that: it inspects the stages a
placement claims, never the ones it should have claimed.
`weight_path()` also refuses to proceed if a level carrying Weights energy is not
claimed by exactly one stage, and `cross_check()` re-derives per-category weight
energy against the `Raw` record first.

Three things the model refuses to fudge, each with a knob and a recorded check:
**physical packing** (`stream` scales every reduced stage by K/N; `aligned`
gives each weight whole bits and moves no SRAM access count),
**reconstruction granularity** (`G_rec`, the co-resident group, is computed from
the layout, not assumed), and **feasibility** (a PE-local boundary whose resident
tile is smaller than `G_rec` is reported `unsupported` with the layers named,
never estimated).

The DRAM term is ONE stage and the whole of it is reducible (changed
2026-09-09): `dram` = the DRAM weight energy x K/N under **every** boundary, R1
included, because the BCH decoder is on the DRAM die and off the fetch path
(01_project_context §1/§4) so only the k message bits are read out and driven
off it. **`f_if` / `ECC_DRAM_IF_FRAC` is REMOVED.** It was 0.40 and it charged a
38.1% cut in bits fetched as a 15.2% energy cut. The DRAM access is
custom-designed to collect only the interleaved message bits of each codeword,
so the array reads fewer bits too and the whole term scales.
The per-bit cost is its own knob, `ECC_DRAM_PJ_PER_BIT` (energy.py
`apply_dram_override`, the DRAM counterpart of `ECC_MAC_PJ_OVERRIDE`): **40
pJ/bit** by default, against Accelergy CactiDRAM's own 8.0 pJ/bit (= 512 pJ per
64-bit access, verified at 64.0 pJ per 8-bit word). 40 is within the 28-45
pJ/bit band reported by FReaC Cache (MICRO 2020) and Gebhart et al. (MICRO
2012); every DRAM percentage scales linearly with it (`provenance.yaml`
`dram_access_energy`). `E_background` and `E_refresh` are env vars fixed at 0
and NOT modelled, which understates the embedded and reconstruction arms alike.
`ECC_RECON_DECODE_SITE=controller` is the pre-2026-09-09 model (DRAM identical
on every bar), kept for the diff only. The reference bars keep controller-side
correction and do not move. Two things it deliberately does not credit: the
DRAM array, and any operand-retention saving for anybody — scratchpad read
counts stay Timeloop's under every bar, and since R4b was removed
(2026-09-10) no boundary claims one. FINDINGS §7.1 has the original audit; read
it together with the R4b removal above rather than from the docstrings alone.

**The MAC cost is the denominator of every percentage** and is audited in
FINDINGS §7.3: the ERT's 1.16877 pJ/MAC is a single 40 nm table row per
primitive, scaled. `ECC_MAC_PJ_OVERRIDE` (env.sh §5) rescales the Compute
category evaluator-side, after the raw cache, so no saved pJ and no mapping
moves; a value listed in `provenance.yaml` `mac_energy_pj` carries its citation
on every figure, manifest and result file. The default is **0.23 pJ**
(Horowitz ISSCC 2014 int8 mult + add, 45 nm; adopted 2026-09-09) and
`ReconSweep.png` is drawn under it; set it EMPTY to reproduce the ERT
denominator, which is the sensitivity row. The mapper is unaffected either way:
it prices MACs from the ERT, so only an edit to the `intmac` component (a cold
cache) would change a mapping.

**Every result and console run prints the ceiling first** — the weight energy a
boundary could reduce, times `1 - K/N`. Without it a sub-percent saving reads as
a missing term rather than as arithmetic.

## Architecture rules

`archs/_shared/standard.yaml` states what must be IDENTICAL across designs
(operand precisions, DRAM geometry, process node, the dense-packing rule) and
what each keeps as its own with a citation; `provenance.yaml` records where every
declared number came from; `noc.yaml` holds the interconnect coefficients and,
per design and with the paper section, **which spatial containers actually are
the NoC**. Two NoC terms Timeloop cannot be given (spatial-reduction adders, the
psum word width on the wire) are charged after mapping by `noc_post.py` from
`noc.yaml`'s `evaluator_only` block; every `Raw` carries their stamp and is
re-gathered if it differs. `archs/_shared/components/` is part of the mapper
fingerprint: editing a component colds every cache, on purpose. `bash run.sh
validate` checks all of it and exits non-zero on a violation.

Only one of the four `source:` values licenses using a design's name as the chip:
`published`. `derived` is a variant differing in one stated block,
`reference_design` is shipped by timeloop-accelergy-exercises and merely *named*
after a paper, `locally_authored` reproduces nothing. `validate` prints it as
`origin:`. **`simba_like` is `reference_design`** — inspired by Simba, not
reproducing it (its geometry fails two checks against the paper, and its on-chip
weight capacity sets its DRAM refetch and therefore its ECC saving). Label it
"Simba-like (reference design)"; there is no "Numba" architecture here.

**Eyeriss v1 IS `eyeriss_like_wglb` (decided 2026-09-10).** JSSC 2017 publishes
the 8 kB filter-weight allocation of the 108 kB GLB, so the file that models it
is the design; `eyeriss_like`, which declares `!Nothing` in its place, is
retired. **The v1 bracketing-pair doctrine is OVER** — `BRACKET_PAIRS` in
`config.py` must be retired with it, or every manifest carries a caveat that is
no longer true. Blocking work first: `eyeriss_like_wglb` has no entry in
`recon.py`'s `WEIGHT_PATHS`/`PLACEMENTS`, and `weight_path()` refuses when a
weight-carrying level goes unclaimed, so `filter_glb` needs a stage and a
boundary. The swap colds every `eyeriss_like` cache. See prompt_2.md and
FINDINGS §8.5.

**The bracket rule still holds for v2**, with the asymmetry that
`eyeriss_v2_like_wglb`'s extra weight level is *not* in its paper.

Accumulator width is deliberately not standardized (v1 16b, v2 20b, Simba 24b,
each cited) — forcing one would equalise the architectures. `ECC_ACC_BITS` does
force one for a sensitivity study only, with its own cache and namespace.
`diagnose` flags a psum level narrower than its own accumulator; a level that is
legitimately narrower declares `# psum-width-ok: <reason>` in the YAML.

## Working on this code

- **Task 1's code is frozen**: `parity.py`, `baseline.py` and `build_stacks()`
  are not refactored. Add modules instead, then re-run `bash run.sh baseline
  --eval` and diff — the Task 1/2 totals must not move. `audit.py` mirrors
  `baseline.py`'s checks for later tasks; keep them in step.
- **Never** read `os.environ` outside `config.py`, and never resolve a path
  outside `paths.py`.
- **Adding an architecture**: `archs/<name>/arch.yaml` (plus `arch_paper.yaml`
  with each number cited if it comes from a paper), the name in `KNOWN_ARCHS` and
  `ARCH_LABELS` in `config.py`, entries in `standard.yaml` and `provenance.yaml`,
  then `bash run.sh validate` and `diagnose` against an existing design before
  committing to a long sweep. Declare partial sums at the accumulator width at
  every level that holds them.
- **Adding a model**: add to `CNN_MODELS` or `TRANSFORMER_MODELS` in `config.py`
  (that list decides which workload file is read), then
  `python3 -m eccenergy.generate models <name>` in the container. Grouped
  convolutions and per-pixel Linear layers are handled by the generator.
- **Adding an ECC approach**: `APPROACHES` in `config.py` plus a branch in
  `build_stacks()`.
- **Adding an architecture to the placement study**: `WEIGHT_PATHS[<name>]` and
  `PLACEMENTS[<name>]` together (see above), then the name in
  `ECC_RECON_PLACEMENTS` in env.sh §4 and in `ECC_RECON_ARCHS` when you want it
  drawn. Get the stage-to-level match right by reading a real
  `timeloop-mapper.stats.txt` AND `timeloop-mapper.map.txt` from that design's
  cache — the stats file gives the level names and the map file tells you which
  level each stage corresponds to. `eyeriss_v2_like_wglb` is currently
  refused for exactly this reason — its `weight_glb` stage has no boundary.
  `test_every_placement_space_is_valid_for_every_supported_design` runs the
  check on every registered design, so a half-finished pair fails the tests
  rather than understating a figure.
- **Adding a sweep axis**: a name in `SWEEPS`, a stem in `SWEEP_STEMS`, the
  resolution in `Config.__post_init__`, and a `_<name>_groups()` in
  `experiments/sweep.py` returning `(groups, stacks, labels, fontsize)`. Do not
  write a second renderer and do not add a second figure to a sweep.
- **Changing how a bar looks**: `draw_panel()` in `plots/stacked.py` is the only
  place a bar is drawn in this project, so every figure moves together. Want a
  second bar routine? Add a parameter to that one instead (`bar_notes` exists
  because a sub-percent result cannot be read off a full-height bar).
- **Line endings**: the shell scripts run inside a Linux container and a CRLF
  makes bash die on `set -o pipefail` with a mangled message. `.gitattributes`
  forces LF; `bash tools-fix-eol.sh` repairs anything that slips through.
  **Always emit LF.**

## Layout

Everything not listed here is what its name says; `eccenergy/` module docstrings
carry the rest.

    env.sh              THE knob file        run.sh    one stage, no knobs
    hpc/                run_all.sh (the one command), map.sbatch, tl.sh
                        (apptainer wrapper), summary.py, HIPERGATOR.md
    FINDINGS.md         the only place empirical claims about the current model
                        live      progress.txt / PROJECT_STATUS.md  status boards
    eccenergy/          config.py (the ONLY reader of os.environ), paths.py (the
                        ONLY resolver of paths), workloads.py, timeloop.py (the
                        only slow module and the only one needing the container),
                        energy.py, ecc.py, recon.py, parity.py, embedded.py,
                        noc_post.py (evaluator-only NoC terms Timeloop cannot cost),
                        results_store.py (the ONLY writer of an evaluation JSON),
                        plots/, experiments/, tests/, generate.py
    archs/_shared/      standard.yaml, provenance.yaml, noc.yaml -- not an
                        architecture; skipped by the installer
    ecc_energy_study/   AUTO-MANAGED: cloned repo + mapper cache. Do not delete.
    legacy/             pre-restructure scripts, figures and FINDINGS -- nothing
                        imports it, nothing in it describes the current model.
                        The exception is FINDINGS_detail_<date>.md: the full
                        working detail behind the current FINDINGS.md, archived
                        there to keep the live file readable.
