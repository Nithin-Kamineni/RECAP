# Why Eyeriss v2 costs more than v1 in this model

> **STATUS — ARCHIVED. Historical record; do not treat any number here as
> current.** This document analyses the model as it stood before the
> 2026-09-06 modelling rewrite, at `ECC_ARCH_FIDELITY=stock`,
> `ECC_OPT_METRIC=edp`, `ECC_VICTORY=500`, unscaled. It was already partly
> superseded when it was written (three defects had been found and fixed but
> the sweep had not been re-run), and the rewrite supersedes the rest.
>
> **Nothing here describes the current model.** Do not quote the v1-vs-v2
> ordering, the per-architecture energies, the capacity figures, or the
> "run that settles it" at the end. Read it for one thing only: the four
> defects it works through are a list of modelling mistakes worth not
> re-making.
>
> Kept because the derivation cost real analysis and the raw numbers cannot
> be cheaply reproduced — the mapper cache it was built from is gone from the
> current defaults.

All figures are dense resnet18, 8-bit weights, BCH(63,51), baseline arm removed
(these are the raw Timeloop energies, before any ECC arithmetic).

---

## The measurements

| architecture | total | vs best | DRAM | Global buffer | On-chip SRAM/RF | Compute | DRAM weight reads | fetches per weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Eyeriss v1 | **12.52 mJ** | 1.00x | 3.66 | 0.75 | 6.05 | 2.06 | 40,474,624 | **3.47** |
| Output stationary | 14.73 mJ | 1.18x | 4.34 | 3.17 | 5.16 | 2.06 | 21,199,360 | 1.82 |
| Weight stationary | 15.77 mJ | 1.26x | 3.10 | 9.24 | 1.37 | 2.06 | 13,628,608 | 1.17 |
| **Eyeriss v2** | **18.40 mJ** | **1.47x** | 7.05 | 2.55 | 6.68 | 2.12 | 85,482,496 | **7.32** |
| Simba | 39.45 mJ | 3.15x | 3.81 | 0.23 | 33.35 | 2.06 | 53,142,528 | 4.55 |

resnet18 has 11,678,912 conv+fc weights, so "fetches per weight" is
`DRAM weight reads / 11,678,912`.

**Both of your observations are confirmed by the data, and one of them is
inflated by a modelling bug.**

- Eyeriss v2 really does come out 47% above v1 in this model. It is not a
  plotting artifact.
- Weight- and output-stationary really are above Eyeriss v1 (1.26x and 1.18x) —
  so that intuition holds. But **~2x of their DRAM energy is a bug, not
  architecture** (see confound 1). Corrected, the ordering may change.

---

## Why v2 loses: it refetches every weight 7.32 times, v1 only 3.47

That single ratio explains v2's DRAM energy (7.05 vs 3.66 mJ, +3.39 mJ) and
therefore most of the 5.88 mJ gap. The energy per DRAM weight read is identical
(64.0 pJ on both), so this is pure traffic, not cost per access.

### Cause A — v2's weight scratchpad is half of v1's, and it is the only weight buffer

Both designs bypass Weights at the global buffer, so in **both** the PE weight
scratchpad is the sole on-chip weight level between DRAM and the MACs:

| | entries per PE | PEs | total weight capacity |
|---|---:|---:|---:|
| v1 `weights_spad` | 384 (depth 192 x 64/8... width 16 / dw 8) | 168 | **64,512** |
| v2 `weights_spad` | 192 (depth 96, width 16, dw 8) | 192 | **36,864** |

v2 has **43% less** on-chip weight capacity. That is faithful to the paper —
v2's spad is 96x24b against v1's 224x16b — because v2 expects CSC compression to
make 96 entries behave like far more. Modelled dense, it just gets a smaller
buffer.

The consequence is visible per layer. The four `C64_M64_R3_S3_P56_Q56` layers
have exactly **36,864 weights** — precisely v2's total capacity, with zero
slack:

| layer | weights | v1 reads | v2 reads | v2 excess |
|---|---:|---:|---:|---:|
| 1, 2, 3, 4 (each) | 36,864 | 516,096 (14x) | 8,257,536 (224x) | +7,741,440 each |
| 10 | 294,912 | 294,912 (1x) | 8,257,536 (28x) | +7,962,624 |

Those five shapes alone account for **31.2 M of v2's 45.0 M net excess reads**.
On layer 10, v1 achieves a perfect single pass over the weights; v2 refetches 28
times.

### Cause B — the weight NoC that makes this work in silicon is not modelled

In real v2, weights stream DRAM → weight router → PE spad, and the hierarchical
mesh NoC broadcasts one fetched weight to many PE clusters. Timeloop models
multicast reuse only across a *spatial* fanout under a shared parent. With the
`!Nothing` weight branch, the spads' parent is DRAM, so there is nowhere for
**temporal** reuse to live: every outer loop iteration goes back to DRAM.

`archs/eyeriss_v2_like/README.md` already flagged the router gap. What was not
obvious is that it costs v2 the *whole intermediate reuse level*, not just some
interconnect energy.

### Cause C — 20-bit partial sums in a 40-bit buffer word

v2's global-buffer energy is 2.55 mJ against v1's 0.75 mJ, a 3.4x gap on
comparable total capacity (192 kB vs 128 kB). v2's `psum_glb` is `width: 40,
datawidth: 20` — two psums per word — while v1's `shared_glb` is `width: 64,
datawidth: 8` — eight entries per word. Same output volume, ~4x the buffer
accesses. The 20-bit psum is faithful to the paper; the resulting access count
is an artifact of how the word is declared.

This also explains the compute gap (2.12 vs 2.06 mJ, +2.9%): v2's SIMD-2 lane
idles whenever `M` is odd, and the mapper pads.

### What is *not* the cause

- **Not the process node.** All five designs declare 45nm. (An earlier revision of
  `archs/eyeriss_v2_like/README.md` said 65nm; that has been corrected.)
- **Not the DRAM cost model.** Both v1 and v2 measure exactly 64.0 pJ per DRAM
  weight read.
- **Not a mapper failure.** All 21 layers mapped on every architecture, 0 skipped.

---

## The honest conclusion on v1 vs v2

The model is **structurally biased against v2**, and the bias is not a bug in
the YAML — it is the gap between what v2 is for and what a dense Timeloop model
can express. v2's contribution is CSC compression of weights *and* activations
plus a switchable NoC; it spent silicon area on those and took a *smaller*
weight scratchpad in exchange. A dense model charges v2 for the smaller spad and
credits it for none of what the area bought.

So: **do not report "Eyeriss v2 uses 47% more energy than v1"** as an
architecture result. Either report the pair of bounds below, or report the v1
number and state that v2 is not comparable without sparsity modelling.

### The bound to run next

`archs/eyeriss_v2_like_wglb/` is a diagnostic variant, added for this purpose:
identical to `eyeriss_v2_like` except the `!Nothing` weight branch becomes a
64 kB weight level standing in for the router network. It gives the two bounds:

- `eyeriss_v2_like` — **pessimistic**: none of v2's reuse machinery modelled.
- `eyeriss_v2_like_wglb` — **optimistic**: weight NoC credited as a full reuse
  level, and still charged no router energy.

```bash
ECC_SWEEP=arch \
  ECC_SWEEP_ARCHS="eyeriss_like eyeriss_v2_like eyeriss_v2_like_wglb" \
  ECC_CONST_MODEL=resnet18 bash run.sh
```

12 layer shapes, no cache — minutes to a couple of hours, resumable. Read
`archs/eyeriss_v2_like_wglb/README.md` before quoting a number from it.

---

## Confound 1: weight- and output-stationary are being charged for 16-bit data

This one is a straightforward bug and affects **every published figure that
compares eyeriss against the stationary baselines**.

```
pJ per DRAM weight read
  eyeriss_like               64.0
  eyeriss_v2_like            64.0
  simba_like                 64.0
  simple_weight_stationary  128.0   <-- 2x
  simple_output_stationary  128.0   <-- 2x
```

The stock `simple_weight_stationary` and `simple_output_stationary` designs
declare `datawidth: 16` on the DRAM, the shared GLB, the PE spad and all three
registers. The study declares 8-bit weights. So those two architectures pay
double DRAM energy per weight, and their 64-bit buffer words pack 4 values
instead of 8 — doubling buffer accesses too. Weight-stationary's 9.24 mJ global
buffer is the largest single number in the whole comparison and it is roughly
twice what an 8-bit design would spend.

Fix, which gets its own mapper cache and leaves existing results untouched:

```bash
ECC_FORCE_DATAWIDTH=8 bash run.sh        # equalise the storage datawidth
```

Two things about this run, both learned the hard way:

* **It only re-maps the two designs it actually changes.** Forcing the datawidth
  to 8 is a no-op on eyeriss v1, v2 and simba, which are already 8-bit, so those
  three reuse their existing mapper cache and report instantly. ~24 layer shapes
  are mapped, not 60.
* **Dedicated partial-sum levels are not forced.** Accumulator precision is a
  separate design choice from operand quantization, and Timeloop asserts
  `width % datawidth == 0` — 8 does not divide Eyeriss v2's `width: 20` psum
  scratchpad. Forcing it aborts the mapper on every single layer with
  `ERROR: data storage width: 20 block_size: 1 word_bits: 8`. The first version
  of this fix did exactly that.

`ECC_FORCE_TECHNOLOGY` is deliberately *not* part of this preset. All five arch
containers already declare 45nm; that knob only changes `globals.yaml`, i.e. the
node DRAM itself is costed at, and because `globals.yaml` sits above every
accelerator container it invalidates all five caches at once. Run it separately
if you want it, and change one variable at a time.

**Prediction to check:** weight-stationary's total should fall from 15.77 mJ
toward ~9–11 mJ and may drop *below* Eyeriss v1. If it does, "WS/OS are higher
than eyeriss" was an artifact of the datawidth mismatch, and the corrected
ordering is the one to publish. This needs a fresh mapper run — the mapping
itself changes when word packing changes, so it cannot be corrected
analytically.

---

## Confound 2: weight policy at the global buffer is not uniform

| keeps Weights in a shared buffer | bypasses Weights |
|---|---|
| simple_weight_stationary, simple_output_stationary | eyeriss_like, eyeriss_v2_like, simba_like |

The two stationary designs give weights a global-buffer level; the three
"realistic" designs do not. This is the single largest driver of DRAM weight
traffic — 1.17 fetches per weight for WS against 3.47 for v1 — and DRAM weight
traffic is exactly the quantity this ECC study measures. It is a real
architectural difference, not a bug, but it must be stated whenever the
`Baseline` DRAM inflation is compared across architectures, because the arm that
inflates DRAM traffic by N/K is inflating a quantity that differs 6x between
designs.

---

## Confound 3: Simba's 33 mJ was in the wrong category

Not an energy error — a labelling one, now fixed. The legacy classifier assigned
a level to "Global buffer" if its name contained `buffer`. Simba's
`PEInputBuffer`, `PEWeightBuffer` and `PEAccuBuffer` are replicated 16–64x
across the PE array, so 33.35 mJ of **distributed per-PE** storage was drawn and
tabulated as global buffer.

`ECC_CLASSIFY=instances` (the new default) classifies by instance count: one
instance is a shared buffer, replicated is local. Totals are unchanged — only
the split between the two on-chip categories moves. `ECC_CLASSIFY=name`
reproduces the old figures.

Simba's 3.15x total is real and expected: 2,113,536 weights of on-chip capacity
bought with a great deal of distributed SRAM, which then dominates the energy.

---

## Recommended order of work

1. `ECC_FORCE_DATAWIDTH=8 bash run.sh` — settles confound 1 and gives a defensible
   cross-architecture ordering. Highest value per hour of compute.
2. `ECC_SWEEP=arch ECC_SWEEP_ARCHS="eyeriss_like eyeriss_v2_like eyeriss_v2_like_wglb" bash run.sh` —
   brackets v2 between its two bounds.
3. Fix the stale "65nm" claims in `archs/eyeriss_v2_like/README.md`
   (`arch.yaml` says 45nm).
4. Decide how to present v2. Either publish the bounds, or restrict the
   architecture claim to v1 / WS / OS / Simba and treat v2 as future work
   pending sparse modelling.

---

## What changed, and why the ordering is not yet settled

Everything above was measured with Eyeriss v1 billing its partial sums as
bytes, with the mapper minimising energy-delay product, and with both designs
given the same search budget over search spaces of very different size. All
three favour v1. They are fixed; the sweep has not yet been re-run.

### Where the gap actually lived

Decomposing the cached `stats.txt` by level and dataspace — mobilenet_v2,
embedded arm, uJ — puts the entire v1 advantage in the **Outputs** dataspace:

| level / dataspace | v1 | v1 pJ/access | v2 | v2 pJ/access | delta |
|---|---:|---:|---:|---:|---:|
| DRAM / Outputs | 427.5 | 64.00 | 698.8 | 64.00 | **+271.4** |
| GLB / Outputs | 88.3 | **3.55** | 247.2 | **11.05** | **+158.9** |
| spad / Outputs | 270.4 | **0.32** | 553.2 | **0.79** | **+282.8** |
| DRAM / Weights | 594.0 | 64.00 | 748.4 | 64.00 | +154.4 |
| spad / Weights | 722.4 | 1.38 | 343.1 | 0.80 | **-379.3** |
| Inputs, all levels | 380.2 | | 348.9 | | -31.2 |
| MAC | 341.5 | | 351.5 | | +10.0 |
| **total** | **2824.3** | | **3291.2** | | **+466.8** |

Outputs alone are +713 uJ. v2 *wins* on weights (-225) and inputs (-31).
resnet18 has the identical signature: Outputs +4479 uJ, Weights +1512,
Inputs -176.

### Defect 1 — v1's partial sums were billed as bytes

Straight from the stats files:

```
eyeriss_like  shared_glb / Outputs : Word bits 8   Block size 8   3.64 pJ per psum
eyeriss_v2    psum_glb   / Outputs : Word bits 20  Block size 2  11.70 pJ per psum
```

`eyeriss_like`'s own YAML says its psums are 16-bit (`psum_spad: datawidth: 16`,
`mac: adder_width: 16`), but its `shared_glb` declares `datawidth: 8`, so
Timeloop packed 8 psums into each 64-bit word and charged each one a byte.
A 3.2x per-psum handicap on the dataspace that decides the comparison. The same
defect sat in `simba_like` (`adder_width: 16` against a 24-bit `PEAccuBuffer`).

It came from upstream, not from this project: `diff` of the patched
`eyeriss_like` against the cloned original is empty.

Fixed in `archs/eyeriss_like/arch_paper.yaml` by splitting the GLB into an 8b
ifmap buffer and a 16b psum buffer — which is also closer to the chip, whose 25
GLB banks are each assigned entirely to ifmaps or to psums (JSSC 2017 Sec. V-A).

### Defect 2 — the mapper was minimising EDP, not energy

`_include/mapper.yaml` ships with `optimization_metrics: [edp]`, and nothing
overrode it. On `C160_M960_R1_S1_P7_Q7` (three occurrences in mobilenet_v2):

| | chosen | cheapest its own search visited | overshoot |
|---|---|---|---|
| v1 | 7.74 pJ/MAC @ 63,700 cyc | 6.25 pJ/MAC | 1.24x |
| v2 | **15.32 pJ/MAC @ 23,030 cyc** | **6.84 pJ/MAC @ 94,080 cyc** | **2.24x** |

v2 threw away a mapping that used 2.2x less energy to buy 4x the latency,
because EDP rewarded it. v2 has 384 MACs to v1's 168, so it has far more
parallelism to spend energy on — EDP is systematically biased by array width.
That shape plus `C576_M160` accounts for ~208 uJ of the 467 uJ gap.

Fixed: `ECC_OPT_METRIC=energy` is the default, applied from `timeloop.py`.

### Defect 3 — unequal search effort

v2's nest is 9 loop levels to v1's 7 (as the stock model had them), and its
`PE_cluster` frees C, M, P and Q where v1's `PE_column` frees only Q and M.
Same `victory_condition: 500` over a much larger space is not the same search.
The consequence is visible in the nests: v1 puts the weight-invariant `P` loop
deep inside where the weight spad retains across it and achieves a single pass
over the weights (`DRAM Weights: 153600 (153600)`); v2 hoists `P` to the
outermost DRAM loop and re-streams the whole tensor 7x (`153600 (1075200)`).

Fixed: `ECC_VICTORY` defaults to 1000 and `ECC_VICTORY_SCALING=levels` doubles
it per level past 8, so v2 and Simba get 2000 and the v2+wglb variant 4000.

### Defect 4 (not a bias, but wrong) — scratchpads did not match the papers

| | published | stock model | now |
|---|---|---|---|
| v1 filter spad | 224 x 16b (JSSC 2017) | depth 192 | depth 224 |
| v1 psum spad | 24 x 16b (JSSC 2017) | depth 16 | depth 24 |
| v2 weight spad | 96 x 24b = 288 B data (JETCAS Table IV) | width 16 -> 192 weights | width 24 -> 288 weights |

The v2 reading is settled by the paper itself: an iact/weight router port "has
a bitwidth of 24 bits such that it can send and receive three 8b uncompressed
iact values or two 12b compressed iact run-data pairs per cycle", and Table IV
accounts for the CSC address vector separately (14 B weight addr RF). So the
dense-equivalent 24b word carries three 8-bit weights.

On-chip weight capacity moves from 64,512 (v1) vs 36,864 (v2) — a 1.75x gap —
to 75,264 vs 55,296, a 1.36x gap. Since DRAM refetch is what dominated v2's
weight energy, this matters directly.

### What has NOT changed

- **v2 is still modelled dense**, and that is still structurally against it.
  Even at 288 weights/PE its on-chip weight capacity is below v1's, precisely
  because it spent that area on CSC machinery a dense model gives it no credit
  for. mobilenet is exactly where that costs most — it is the network the v2
  paper leads with.
- **Router/NoC circuit energy is still unmodelled** for every design, and
  `eyeriss_v2_like_wglb` is still the right way to bracket the missing weight
  reuse level.
- Confounds 1, 2 and 3 below stand, except that confound 1 is now fixed by
  `ECC_ARCH_FIDELITY=paper` rather than by `ECC_FORCE_DATAWIDTH=8`.

### The run that settles it

```bash
ECC_SWEEP=arch ECC_CONST_MODEL=mobilenet_v2 \
  ECC_SWEEP_ARCHS="eyeriss_like eyeriss_v2_like eyeriss_v2_like_wglb" bash run.sh
```

Cold — the defaults changed, so none of this is in the mapper cache. Resumable.
Then repeat at `ECC_CONST_MODEL=resnet18`, and check convergence by raising
`ECC_VICTORY` and confirming the totals do not move.

**Prediction, stated before the run so it can be wrong:** v2 closes most of the
gap on mobilenet_v2 and may pass v1; on resnet18, where v2's excess was
dominated by DRAM weight refetch rather than psum billing, it closes less. If
v2 still loses on mobilenet after all four corrections, the remaining cause is
the dense-modelling bias, and the honest report is the v1/v2+wglb bracket rather
than a single v2 number.
