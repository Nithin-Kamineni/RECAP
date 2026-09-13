# Weight-memory sweep: where does Recon beat Embedded?

*Rewritten 2026-09-10. Supersedes the previous version, whose `floor(W/q)` word-widening
scheme and `x4` quarter-bit trick were both measured to be unimplementable — see WHAT
CHANGED at the bottom.*

## The question

For **Eyeriss V1**, one layer at a time: what on-chip weight-memory DEPTH makes
reconstruction save the most energy against embedded ECC? Recon stores the same weights
in fewer bits, so the same array holds more of them and refetches less from DRAM. Sweep
the weight GLB and scratchpad depths down (×0.5, ×0.25, ×0.125 of the published values)
and find where Recon's margin over Embedded is largest.

**The output is a `depth:` per weight level, to be written into a new arch YAML** — this
is a design-space search for Recon's optimal operating zone, not a characterisation of
the published chip. Absolute energies are therefore expected to differ from Eyeriss V1's;
what must stay exact is the Recon-vs-Embedded comparison at each depth.

Three arms: **Baseline**, **Embedded**, **Recon**. Baseline and Embedded both store 8-bit
weights, so **they share one hardware YAML and one mapping** — only the evaluator
separates them. Recon is the only arm with a different `datawidth`. That means **2 mapper
configs per (width, depth), not 3.**

## The method: quantisation via `datawidth` — VERIFIED, do not re-derive

Change `datawidth:` on weight-carrying levels. Leave `width:` and `depth:` alone.
Timeloop computes `block_size = width / datawidth`, so more weights ride each access
while the array geometry is unchanged.

Verified in this repo, 2026-09-10, from `timeloop-mapper.accelergy.log:97`
(`Calculated storage."width" as "width" = 16`): **CACTI receives `depth` and `width`
only. `datawidth` never reaches the energy model.** Timeloop then bills
`vector_access_energy / block_size` per weight. So halving `datawidth` at fixed geometry
exactly halves per-weight energy and exactly doubles effective capacity, with identical
per-access read/write/leak. This is the "same energies, more effective capacity"
condition, and it holds exactly.

### HARD CONSTRAINT: `width` must be an exact multiple of `datawidth`

`timeloop-mapper` aborts otherwise — measured, `width: 16, datawidth: 5`:

```
timeloop-mapper: src/model/buffer.cpp:302: Assertion
  `width % (word_bits * block_size) == 0' failed.
ERROR: data storage width: 16  block_size: 1  word_bits: 5      exit=134, core dumped
```

Note `block_size: 1` — **Timeloop does not attempt `floor(16/5)=3`. There is no floor
path in the code.** Partially-filled words cannot be modelled. `archs.py:1147` pre-checks
this so a bad YAML fails at validate time instead of aborting every layer.

## THE WIDTH TABLE — adopt these, do not sweep them

One declared width per code, chosen so `width % q == 0` and all five widths sit within
3% of each other. Both arms of a pair share the SAME `width` and `depth`; only
`datawidth` differs.

| arm | K/N | q true = 8K/N | **q declared** | **declare width** | weights/word | eff. capacity | pJ/access (rd/wr) | pJ/weight | **saving measured** | ideal (8K/N) | residual |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline / Embedded | 1.0000 | 8.000 | **8** | **96** | 12 | 1.0000× | 1.48668 / 2.37439 | 0.12389 | — | — | — |
| BCH(63,57) | 0.9048 | 7.238 | **7** | **98** | 14 | 1.1667× | 1.57809 / 2.49938 | 0.11272 | 9.02% | 9.52% | −0.51 pp |
| BCH(63,45) | 0.7143 | 5.714 | **6** | **96** | 16 | 1.3333× | 1.48668 / 2.37439 | 0.09292 | 25.00% | 28.57% | −3.57 pp |
| BCH(63,39) | 0.6190 | 4.952 | **5** | **95** | 19 | 1.5833× | 1.47584 / 2.35736 | 0.07768 | 37.30% | 38.10% | −0.79 pp |
| BCH(63,30) | 0.4762 | 3.810 | **4** | **96** | 24 | 2.0000× | 1.48668 / 2.37439 | 0.06195 | 50.00% | 52.38% | −2.38 pp |

Energies MEASURED 2026-09-10 (`weights_spad`, depth 37, 45 nm) — not modelled. Read them
as the calibration of this table, not as results.

Four things this table settles:

* **The three width-96 arms have byte-identical per-access energy** (1.48668 / 2.37439),
  so for BCH(63,45) and BCH(63,30) the array is literally the same silicon as Embedded's.
* **Benchmark against `8K/N`, not against the rounded integer `q`.** BCH(63,57) rounds
  DOWN (7 vs 7.238), which on its own flatters Recon; its wider word (98 vs 96) offsets
  that. Only the true code rate settles which way the net error runs.
* **All four residuals are NEGATIVE — every arm understates Recon.** That is the safe
  direction. State it; do not correct it.
* BCH(63,57) is the weakest test (9.02% effect) and BCH(63,30) the strongest (50.00%).

Quantisation is 8·K/N rounded to integer bits. Quarter-bit resolution (7.25 / 6.5 / 5.75)
is **abandoned**: it needs `datawidth` in quarter-bit units, which forces `width` into the
same units, which is the number CACTI prices. Measured inflation of the weight buffer:
**6.69× at a 64-bit word, 11.43× at a 256-bit word.** There is no way to compensate —
dividing the depth to fix the bits destroys the capacity being swept.

## THE ONLY SWEPT VARIABLE: `depth:` of the on-chip weight levels

Explicitly NOT swept: `width`, `datawidth`, DRAM (`ECC_DRAM_DEPTH`), any level not
holding Weights, bandwidths, `n_banks`, technology, PE counts, dataflow constraints.

Declared depths at these widths (total bits held ≈ the paper's):

| level | paper geometry | at width 96 | emb weights |
|---|---|---|---|
| `weights_spad` | 224 × 16b = 3,584 b | **depth 37** | 444 (paper 448) |
| `filter_glb` | 1024 × 64b = 65,536 b | **depth 682** | 8,184 (paper 8,192) |

### WHY: this sweep exists to pick the numbers for the new arch YAML

**The deliverable is not a trend, it is a set of `depth:` values.** The question is: at
what on-chip weight-memory size does Recon's margin over Embedded PEAK? That depth range
is the "optimal zone", and the depths at its centre are what get written into the new
architecture YAML the experiments then run on. A sweep that ends in a chart and no
recommended depth has not answered the question.

**×0.5, ×0.25 and ×0.125 are the points to REPORT** — they are the realistic design
choices (half, quarter, eighth of the published memory) and the ones a YAML will quote.

**But they are not enough points to SEARCH with.** The window where Embedded cannot hold
the tile and Recon can is exactly as wide, in depth, as the effective-capacity ratio. A
factor-2 grid steps clean over a 1.17× or 1.33× window and reports "no effect" that is a
grid artifact, not a result. So search on **√2 steps** (×1, ×0.71, ×0.5, ×0.35, ×0.25,
×0.18, ×0.125) — that resolves BCH(63,30) (2.00×) and BCH(63,39) (1.58×) — and quote the
factor-2 points out of it. BCH(63,45) (1.33×) and BCH(63,57) (1.17×) need ~×1.12 steps,
about 20 depths, so do them only after the strong codes show something.

### THE RESULTS TABLE — one row per (arm, scale, MEMORY LEVEL)

Save it to exactly this path:

    results/tables/EyerissV1_mem_arch_sweep.csv

Required columns. The first five are the ones the design decision is read off; the rest
are what make a row trustworthy:

| column | meaning |
|---|---|
| `memory` | the level's name AND kind — `filter_glb` (GLB) or `weights_spad` (scratchpad) |
| `scale` | `×depth` applied to that level (×1, ×0.71, ×0.5, ×0.35, ×0.25, ×0.18, ×0.125) |
| `refetch` | DRAM weight reads ÷ unique weights |
| `weights_held` | weights actually resident — the UTILISATION, in weights not percent |
| `level_pJ` | energy consumed by THIS memory level (approximate is fine) |
| `arm` | `baseline` / `embedded` / `recon` |
| `declared_depth`, `width`, `datawidth`, `weights_per_word` | the geometry that produced the row |
| `room`, `fill_pct` | capacity in weights, and `weights_held / room` |
| `dram_weight_reads`, `total_uJ` | the totals the margin is computed from |
| `pes_used` | disqualifies the row if it moves between arms |
| `recon_minus_embedded_pJ` | **the answer column** — the margin at this level and scale |
| `verdict`, `fingerprint` | `capacity` is the only quotable verdict; `fp-<hash>` guards cross-arch reads |

**Two of these do not exist yet.** `dilation.py --table --csv` writes one row per
(arch, layer, arm, scale) and packs every level into a single `levels` string
(`_weight_level_rows`, line 284, carries only `level`/`capacity`/`residency`/`instances`),
and it emits **no per-level energy at all** — only `total_pJ` for the whole run. So:

1. **Reshape to one row per level** instead of the packed `levels` string.
2. **Add `level_pJ`**: parse `Energy (total)` from that level's `Weights` block in
   `timeloop-mapper.stats.txt` (it is already there — e.g. `weights_spad` reports
   `Energy (total) : 46019432.15 pJ`). A parse, not a model.

`weights_held` is what proves the margin is capacity and not loop order; `pes_used` is
what disqualifies a row; `level_pJ` is what tells you WHICH memory to shrink.

### The two levels are swept TOGETHER by default — that is a limitation

`ECC_WEIGHT_CAPACITY_SCALE` rewrites `depth:` on **every** weight-carrying level at once,
so one scale moves `weights_spad` and `filter_glb` together. That is fine for locating the
zone, but it **cannot tell you which level bought it**, and the YAML needs a depth per
level. Plan a second pass that varies them independently (hold one at ×1, sweep the
other) on the two or three scales that looked best. FINDINGS 7.8 predicts `filter_glb`
is the one that matters — confirm it, do not assume it.

Also: `ECC_WEIGHT_CAPACITY_SCALE` **triggers `recon.capacity_dilation_correction()`**,
which re-prices the level at the undilated geometry. That correction is WRONG here — a
shallower array really IS a smaller array, and its cheaper access is a real saving, not an
artifact to undo. Add a separate knob, or a flag that suppresses the correction.

## START HERE: BCH(63,30) on the declared geometry

`width: 16, datawidth: 4` loads on Eyeriss V1's untouched `weights_spad` (exit 0,
verified): 2 weights/word for Embedded, 4 for Recon, exactly 2.000×, correct CACTI
pricing, zero residual, and the full `224 → 112 → 56 → 28` depth range. **No width change
needed at all for this one code.**

It is also the only code that clears the integer-tile step FINDINGS 7.8 measured: 1.6154×
capacity at 88.9% fill bought exactly nothing, because the tile could only grow in a 2×
jump. Ratios 1.17, 1.33, 1.58 all sit below that step. **If 2.00× shows nothing, the
weaker codes cannot.** Prove the mechanism here first — ~14 jobs.

Falsifiable prediction to run first. On `layer3.0.conv1` the resident tile is 192 weights
and Embedded holds `2 × depth`, so the tile stops fitting below depth 96 = ×0.4286. At
**×0.25** (depth 56 → 112 weights) Embedded cannot hold it and a 2.0× Recon arm (224) can.
So ×0.25 should show `weights held` RISING and refetch FALLING. If it does not, the
hypothesis is in trouble — for three hours of compute instead of two hundred.

## THE ARCH — Eyeriss V1 is now the `_wglb` file

DECIDED 2026-09-10: `archs/eyeriss_like_wglb/arch_paper.yaml` is the correct Eyeriss V1
and becomes the default; `archs/eyeriss_like/arch_paper.yaml` (which declares `!Nothing`
where the published 8 kB filter GLB sits) is retired. **The old file has exactly one
on-chip weight memory; the correct one has two** — `weights_spad` AND `filter_glb`.

**Expect the effect at `filter_glb`, not `weights_spad`.** FINDINGS 7.8: refetch on
Eyeriss V1 is set by the DRAM-level loop order over `P`/`Q`, and a weight tile cannot
index `P` or `Q`, so a weight buffer INSIDE the PE array cannot absorb those loops however
large — v1 measured flat to ×32 capacity at 1% fill. `filter_glb` sits above the array and
can. Sweep both depths; report both.

**The swap is not a file copy. Three things break:**

1. `eyeriss_like_wglb` has **no entry** in `recon.py`'s `WEIGHT_PATHS` (line 375) or
   `PLACEMENTS` (line 652). `weight_path()` refuses when a weight-carrying level goes
   unclaimed, so `filter_glb` needs a new GLB `Stage` and a placement before anything
   evaluates. Fail-loud, but real work.
2. `config.py:193` `BRACKET_PAIRS` makes the two files each other's bracket. Collapsing
   them to one design makes that self-referential — retire the pair.
3. Every fingerprint changes, so **the whole `eyeriss_like` cache (64 variant dirs) goes
   cold.** Budget a full re-map.

## THE LAYER

Use **`layer3.0.conv1`** = `C128_M256_R3_S3_P14_Q14_ws2_hs2` (294,912 weights). Verified
from cache at declared capacity: **168/168 PEs**, `weights_spad` 192/448 = **42.9% fill**,
**refetch 14.0**. Full PE occupancy means no `PE!=` confound at baseline.

Do NOT use `C64_M64_R3_S3_P56_Q56_ws1_hs1` as primary despite it being the most repeated
shape: verified at **96/168 PEs and 14.3% fill** (64 of 448 weights), so it carries the
confound at baseline and the mapper does not want the room it already has. Do not use
`layer4.*` — already refetches 1.0×.

## CONVERGENCE IS A GATE — run it before quoting any number

`ECC_VICTORY=4000` is the chosen budget. It is a choice, not a result: **the gate still
has to pass at 4000.** Why this matters — measured, `simple_weight_stationary`,
layer2.0.conv1:

| scale | victory 2000 | victory 10000 |
|---|---|---|
| ×0.5 | rf 1.000, 273.2 µJ | rf 2.000, 251.0 µJ |
| ×0.8077 | rf 1.000, 264.1 µJ | rf 8.000, 256.4 µJ |

At 2000 the Recon arm beat its Embedded reference by 9.1 µJ — the best result in the
study. At 10000 the SAME pair reverses: Recon 5.4 µJ WORSE, 4× the refetch. Energy falls
everywhere (a better search) but the ORDERING between arms flips. The sweep was measuring
the search, not the architecture.

The gate: map the Embedded arm at victory {2000, 4000, 10000}; converged means the last
two agree within a margin you STATE, and that margin must be smaller than the
Recon-vs-Embedded effect you intend to claim. Re-check at the SMALLEST depth as well as
the largest. Cost, measured: victory 10000 on one layer ran **1h13m–1h34m** wall at 18
threads; victory 50000 ran **6h02m–8h09m** (jobs 41582127-30, all COMPLETED
2026-09-10). So a 4-point gate at 50000 is a full day per arm — 4000 is the right
call, but verify it, do not assume it.

**The victory-50000 caches now EXIST** for weight-stationary layer2.0.conv1 at
scales {0.5, 0.75, 0.8077, 1.2115} (`results/_raw/simple_weight_stationary/vic50000__*`).
Read them before running anything new: they are the third point of the
2000/10000/50000 curve and may settle the gate for free.

**Do NOT buy speed by changing the search algorithm.** Measured 2026-09-10 (env.sh §2,
WS layer2.0.conv1, victory 10000, 4 scales — best pJ/MAC, lower is better):

| algorithm | best pJ/MAC across 4 scales | cost |
|---|---|---|
| `random_pruned` | **4.34 / 4.40 / 4.44 / 4.52** | 1.61 h/map |
| `linear_pruned` | 6.17 / 6.53 / 6.56 / 6.89 | 0.12 h/map |
| `hybrid` | 6.61 … 9.80 | — |

`linear_pruned` is 13× cheaper and **42% worse**; `hybrid` at victory 10000 was worse
than `random_pruned` at victory 2000. Keep `ECC_MAPPER_ALGORITHM=random_pruned` and buy
convergence with `ECC_VICTORY` instead. `ECC_MAPPER_SEARCH_SIZE` is the alternative
budget — 644,000 ≈ victory-10000 effort — and is the fairer knob for an A/B, because
victory is ADAPTIVE (an improvement resets the counter, so the arm that keeps improving
gets a bigger budget).

## BEFORE ANY NUMBER IS QUOTED

1. **DOUBLE-COUNTING — settle this before the first evaluation.** `recon.py`'s default
   `stream` packing already scales every reduced stage by K/N. The mapper-side
   `datawidth` now delivers that same on-chip saving inside the Timeloop number, with no
   mapping change at all. Apply both and the on-chip saving is squared.
   **The resolution is `ECC_RECON_PACKING=aligned`.** Its docstring (`recon.py:1378`)
   already describes exactly this model — "each reduced weight occupies a whole number of
   bits ... the access count, and the SRAM energy, do not move at all" — so `aligned`
   leaves the on-chip narrowing entirely to the mapper, which is now where it belongs.
   Two consequences to honour:
   * **DRAM `datawidth` stays 8 on every arm.** `recon.py` owns the DRAM K/N scaling;
     narrowing DRAM in the YAML too would double-count it there.
   * `aligned` computes its own width as `ceil(8·K/N)`, which is 8 for BCH(63,57) and
     disagrees with the declared `q=7`. Make the arch and the accounting agree, or the
     check will not reconcile.
   Add an assertion that the on-chip narrowing is applied exactly once.
2. **Assert the fairness rule mechanically.** Both arms of a pair must declare the SAME
   `width` and `depth` on every level. If they ever differ, the comparison is void.
3. **Report `PEs used` on every row.** FINDINGS 7.8 measured 168 → 84 → 96 → 112 PEs on
   Eyeriss V1 below ×0.5 under EDP: the mapper trades parallelism for capacity, and a pair
   with mismatched PE counts is not attributable to capacity. Discard those pairs; do not
   quote them beside clean ones.
4. **Separate the two Recon terms in the table.** The `datawidth` packing discount applies
   to every on-chip weight access regardless of mapping, so Recon wins monotonically at
   every depth even with a byte-identical loop nest. Report the flat packing discount and
   the differential-refetch term in SEPARATE columns, or the table looks like a win
   everywhere and tells you nothing about which memory size matters.
5. **`weights held` is the column that decides whether anything happened.** If DRAM reads
   move while `held` does not RISE, it was not capacity.
6. `ECC_OPT_METRIC=edp` always. Never `energy` — it serialises and trades PEs for
   capacity, which is what withdrew FINDINGS 7.7.

## HOW TO RUN

* `module load apptainer`, then `bash hpc/tl.sh <cmd>`. Read CLAUDE.md, then FINDINGS.md
  §7.8, then the last entries of progress.txt.
* `env.sh` is the only file to edit; every knob is `${VAR:=default}`. `ECC_RECON_MODELING=1`
  is the DEFAULT and HARD-ASSIGNS `ECC_ARCHS` from `ECC_RECON_ARCHS` — setting `ECC_ARCHS`
  on the command line is silently ignored. Use `ECC_RECON_ARCHS`.
* Keep `ECC_MAPPER_THREADS=18`; it is in the cache fingerprint.
* `ECC_QOS=rewetz-b` gives ~90 concurrent jobs.
* Report with:

      bash hpc/tl.sh python3 -m eccenergy.experiments.dilation --table \
           --csv results/tables/EyerissV1_mem_arch_sweep.csv

  Running it bare (without `hpc/tl.sh`) silently uses `config.py` defaults
  (objective=energy, victory=500) instead of env.sh's — the table prints a banner when
  the objective is not `edp`, but do not rely on noticing it.
* Run `eccenergy/tests/test_dilation.py` after any change to the reporting path.

## WHAT CHANGED, and why the previous sweep proved nothing

The previous sweep's eight defects, all still true and all still worth avoiding:
the two arms saw DIFFERENT read/write energy (capacity expressed as `depth × N/K`, so
Accelergy priced Recon's array 1.18–1.46× dearer and the optimiser had a reason to leave
the room unused); the ECC evaluator was NEVER RUN (only mapping jobs, every "saving" was
DRAM arithmetic from `dilation.py`); 5 archs × 5 layers × 28 capacities moved at once;
capacities went to ×0.03125 where PE count moves; one headline compared two different
fingerprints; the verdict logic called a win whenever DRAM reads fell without requiring
`held` to rise; every number came from an unconverged search; and 12 jobs went on
`ECC_MAPPER_SEED`, which is in the fingerprint but reaches nothing (`timeloop-mapper` v4
exposes no seed — env.sh says so).

Removed from this document in the 2026-09-10 rewrite: the `floor(W/q)` Table A / Table B
word-widening scheme (Timeloop has no floor path — it aborts); the `x4` quarter-bit trick
(inflates the buffer 6.69–11.43×); the `filter_glb` rows attributed to `eyeriss_like`
(that level was only in `_wglb`); a stale block treating `ECC_DRAM_IF_FRAC = 0.40` as an
unresolved blocker (`f_if` was REMOVED — see prompt_1.md and env.sh §4, DRAM is now
40 pJ/bit with the whole term scaled by K/N); and the ×0.25/×0.125 grid that contradicted
this document's own warning about PE trading below ×0.5.
