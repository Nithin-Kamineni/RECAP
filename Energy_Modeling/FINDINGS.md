# FINDINGS

The only place empirical claims about the current model live.

**The live plan is `prompt_7.md`** (prompt_6's four RULES are still in force).
Where this file and the plan differ on what to do NEXT, the plan wins; this file
records what was LEARNED.

> **PHASE C2 HAS RUN FOR resnet18** (2026-09-13, §3.10): 72 maps, six arms on their own
> plans, +15.552 % vs conventional ECC and −12.56 % latency, at the ceiling. Those are the
> live numbers for that model.
>
> **EVERY OTHER NUMBER IN §2 WAS PRODUCED AT A PRE-PHASE-C FINGERPRINT.**
> prompt_7 Phase C1 (2026-09-13, §3.9) put the off-chip speed limit, the
> per-dataspace bandwidth scale, a bit-aware port, the 200 MHz clock, the MAC
> price and a bank count into the architecture THE MAPPER READS, so every cache
> is cold on purpose and nothing in §2 can be reproduced until Phase C2 runs.
> **Those numbers are SUPERSEDED, not wrong** — they are what the model said
> about a machine whose memory had no speed limit. Two of them are known to
> move: the reconstruction column by +0.1699 % (the DC idle term is now rescaled
> to the design's clock) and any latency figure, which did not exist before.
> The one command that ends the cold, per model (`ECC_CONST_MODEL` selects it,
> and `hpc/map_ert_arms.sh` was folded into it on 2026-09-14):
>
>     ECC_CONST_MODEL=<model> bash hpc/run_all.sh --map-only
>
> mobilenet_v2 has NOT been re-mapped since Phase C; its §2.9/§2.10 numbers are superseded.

**Slimmed 2026-09-11.** Full working detail — every measurement pass, every
withdrawn result, every table — is archived verbatim in
`legacy/FINDINGS_detail_2026-09-11.md` (and the two earlier snapshots beside it).
Nothing was deleted, only moved. `legacy/FINDINGS.md` is pre-rewrite; do not quote it.

---

## Reconstruction energy is TWO terms (prompt_6 RULE 3, in force since 2026-09-11)

Every reconstruction number produced before 2026-09-11 used
`recon_pj = incremental_per_codeword + idle_per_cycle` — a per-codeword quantity
added to a per-cycle one — and was removed from this file. The model now charges

    E_recon = incremental x events  +  idle_per_cycle x cycles x N_engines

with the cycle count of the plan being billed and the engine count derived from
the boundary (1 at the chip ingress or the GLB, 14 at the column edge, 168 at the
PE scratchpads). **The placement study has been regenerated under it: §2.9.**
`BCHsweep` and `ArchitectureSweep` were regenerated only as far as their caches
reach and `ModelSweep` not at all (§6.1) — none of the three is a model total.

---

## Status

| Task | State |
|---|---|
| 1 conventional ECC, external parity | implemented; one-layer total in §2.9 (483.854 µJ at the 40 pJ/bit baseline price; 761.102 at 70) |
| 2 embedded ECC (DRAM effect only) | implemented; one-layer total in §2.9 (299.022 µJ at 20 pJ/bit; 483.854 at 40) |
| 3 reconstruction placement, fixed mapping | implemented, 5 boundaries on `eyeriss_like_wglb` |
| 4 reconstruction-aware mapping by **capacity dilation** | ❌ **withdrawn.** No clean capacity win across 196 EDP mappings on 5 designs, and the sweep ran unconverged. Cause is arithmetic, not search — see §2.4 |
| 4′ same question by **`datawidth`** | mechanism verified (§3.1), measured under the constrained mapspace |
| 5 **reconstruction-aware ERT** | ✅ **phases 1–9 DONE 2026-09-11** (`prompt_6.md`). Hook proven (§3.5), two ERT arms mapped and billed from their own plans; result §2.9: the toll moves neither arm's nest under the constrained family, and idle dominates |
| 5′ **full-model placement study, two networks** | ✅ **2026-09-11**, §2.10: resnet18 (12 shapes) and mobilenet_v2 (31 shapes), all three arms cached; idle engines are the UTILIZED PEs (Timeloop's own rule), PE differences reported per shape; figures `ReconSweep_optimiser__<model>.png` |
| convergence of the mapper search | ✅ solved by constraining the mapspace, not by raising the budget (§2.2) |
| Eyeriss v1 = `eyeriss_like_wglb` | DECIDED 2026-09-10, registered, validates |
| placement study on other designs | not started; `eyeriss_v2_like_wglb` refused until its GLB boundary exists (prompt_6 Appendix B) |
| 6 **TIME in the mapper** (prompt_7 Phase C) | ✅ **C1 authored and validated 2026-09-13**, §3.9 — off-chip limit, per-dataspace scale, bit-aware port, per-design clock, MAC price in the ERT, banked SRAM. **EVERY CACHE IS COLD ON PURPOSE**; C2 is the 186-job pass that refills it. Pre-tested on one `fc` shape: −37.6 % cycles and a different plan |
| transformer workload (prompt_7 Phase D) | ❌ **dropped 2026-09-13**, the user's call — this study is about CNNs. The 52.4 % ceiling stays a PROJECTION with two independent derivations (§4.5, A.6) and the measured `fc`-layer result beside it |

---

## §1 — Settings every future number must be produced at

`eyeriss_like_wglb`, resnet18, BCH(63,30), 8-bit weights, 45 nm, 1 ns cycle,
**NoC costed**, `paper` fidelity, `ECC_RECON_PACKING=aligned`, **18 threads**
(in the mapping fingerprint — change it and every cache is cold).

**Objective `edp`, never `energy`.** Under `energy` the mapper serialises and
trades PE count for capacity; that is what withdrew an earlier result.

**Denominators (env.sh defaults since 2026-09-11):** `ECC_DRAM_PJ_PER_BIT=20`
(Horowitz, ISSCC 2014), `ECC_BASELINE_DRAM_PJ_PER_BIT=40` (×2.0, the array that
also stores the parity), `ECC_MAC_PJ_OVERRIDE=0.23` (the same Horowitz table).
The 20/40 pair is the value env.sh has carried on disk since 08:54 that day,
adopted as the study's; 40/70 was the default from 2026-09-09/10 and every
number produced at it is labelled so (§2.9 keeps them as the sensitivity row).
Every DRAM percentage scales linearly with the price and nothing else moves.

**Search:** `ECC_MAPSPACE_CONSTRAIN=1` + `ECC_WEIGHT_FACTOR_RELAX=1`, algorithm
`linear_pruned`, `ECC_MAPPER_TIMEOUT=100000000`, `ECC_VICTORY=4000`. See §2.2 for
why this combination and not a bigger budget. **These are env.sh's defaults since
2026-09-11**; before that they had to be exported on every mapper command line.

Reference layer: `layer3.0.conv1` = `C128_M256_R3_S3_P14_Q14_ws2_hs2`, 294,912
weights. At the current YAML: **168/168 PEs**, weight refetch **1.000**,
**344,064 cycles**, `weights_spad` at 100 % fill.

---

## §2 — Mechanisms that survive re-measurement

### 2.1 The mapper search decides the ranking — the original central caveat

Same layers, same YAMLs, only the search differs: the architecture ordering
**inverts**, and nothing about the hardware changed. A 36,864-weight tensor
fetched 196× is a search freezing a bad index factorisation, not a property of
Eyeriss. A *more* thorough `hybrid` search gave *worse* numbers than a bounded
one — the signature of a search that is not converging.

The simple WS/OS/IS designs have a far smaller legal mapspace and barely move,
which is exactly why they used to "win".

**No figure produced under `hybrid`, or under a bounded development search, is
quotable.**

### 2.2 Shrink the mapspace; do not grow the budget

This is the finding that unblocked the study. The unconstrained space is
7.41e10 index factorizations; constraining the loop nest collapses it:

| constraint | factorizations | exhaustive h/map |
|---|---:|---:|
| unconstrained | 7.41e10 | 164,657 |
| pin R,S to one level | 2.06e9 | 4,574 |
| + P,Q across ≤ 2 levels | 1.32e7 | 29 |
| **+ C,M across ≤ 3 levels** | **3.63e4** | **0.08** |

The last tier is **exact** and roughly 8× cheaper than an approximate search that
was still ~10 % wrong. **A smaller exactly-solved problem beats a larger
approximately-solved one.**

Measured under it: **0.00 % residual** across victory 2000 / 4000 / 10000, at
35 s – 1 min 19 s per map against 24 min – 2 h 56 min unconstrained. Bit-identical
at every budget means the space is exhausted and each arm sits at its true optimum
within the family.

**Two caveats travel with every number produced this way.** It is a CONSTRAINED +
RELAXED dataflow with a reconfigured psum/weight geometry — **not** the chip
JSSC 2017 describes, and `source: published` does not licence its name. And the
result is conditional on that mapspace family: both arms are searched exhaustively
*within* it, so a positive is trustworthy while a negative would be weak evidence.

**Algorithm depends on which regime you are in.** Unconstrained, `random_pruned`
wins on every arm and `linear_pruned` sticks in a biased prefix. Constrained and
exhaustive, `linear_pruned` with a huge timeout is correct because the walk
completes. Do not carry one regime's choice into the other.

`search_size` is the fairer budget knob for an A/B: `victory` is *adaptive*, so
the arm that keeps improving gets searched longer, confounding search effort with
the hardware difference under test. Wall-time spread across four arms of one
layer: victory 2000 **3.28×**, victory 10000 1.57×, `search_size` 20000 **1.13×**.

### 2.3 Refetch is set by loop ORDER at the DRAM level, not by capacity

Refetch is the product of the DRAM-level loop factors that do **not** index
Weights and sit outside one that does. A weight tile cannot index `P` or `Q`, so
**a weight buffer inside the PE array can never absorb those loops however large
it is made** — measured flat to ×32 capacity at ~1 % fill.

Only a weight level *above* the array can. That is what makes the `_wglb`
variants and `ECC_WEIGHT_CAPACITY_SCOPE=shared` the levers, rather than a bigger
scratchpad.

### 2.4 The capacity mechanism is a 2× STEP — only BCH(63,30) reaches it

Resident tiles are integers, and a level's share of `C` can only **double**. So
extra weight capacity buys nothing until it reaches the next power of two:

| code | q | capacity `8/q` | did the extra room get spent? |
|---|---:|---:|---|
| BCH(63,57) | 7 | 1.1429× | **no** — `weights_held` identical on both arms, fill FALLS 77.5 → 67.8 % |
| BCH(63,45) | 6 | 1.3333× | **no** — identical, fill FALLS 80.9 → 60.7 % |
| BCH(63,39) | 5 | 1.6000× | **no** — see 2.4b; the ×1 row is the REFERENCE collapsing, not recon rising |
| **BCH(63,30)** | **4** | **2.0000×** | **yes** — halves the DRAM `C` chunk count |

This is why **BCH(63,30) is the code the study runs on**. It is also why
capacity dilation to N/K = 1.6154 could never work: it sits below the step.

`weights_held` identical across arms while fill falls is the "extra room not
spent" signature — not a win, however the energy column reads.

### 2.4b The measured BCH sweep — WITHDRAWN 2026-09-12, wrong widths

**EVERY NUMBER IN THIS SECTION WAS PRODUCED UNDER THE `lcm(q, 8)` WIDTH SCHEME
AND MUST NOT BE QUOTED.** That scheme read prompt_2's fairness rule as "both
arms share one declared width", concluded that prompt_2's 98 (q=7) and 95 (q=5)
were illegal because they do not divide 8, and substituted `lcm(q, 8)` —
56 / 24 / 40. **The premise is false.** The arms are never mapped on one
another's silicon: `timeloop-mapper`'s `width % datawidth == 0` is per level,
per mapper run, and one run maps one arm. BCH(63,39)'s arm runs at width 95
because `95 % 5 == 0`; `95 % 8 = 7` says nothing about it.

The cost is visible in the table below and is the reason it is withdrawn rather
than merely relabelled: **the lcm scheme made the 8-bit REFERENCE arm move
between codes** (spad 56 / 24 / 40), so the reference, not the treatment, is
what the `spad/GLB W` column is really sweeping. Under prompt_2's actual table
the 8-bit arm is width 96 at every code — one arm, mapped once — and the
`BCH(63,39) = 37.69 %` artifact dissected below cannot arise at all.

**Re-run required** before any BCH-sweep number is quoted. The three arms are
cold at every code but BCH(63,30)… which is also cold, because the 8-bit arm
moved from the published 16 b/64 b to 96 b/384 b.

Original text, kept for the diff:

> **2026-09-11**, `eyeriss_like_wglb`, `layer3.0.conv1`, no depth scaling, one
> mapper PAIR per code on identical declared silicon. Gate PASSES on every row at
> residual **0.00 %**. `results/tables/EyerissV1_bch_sweep_spad16.csv`.

| code | q | spad/GLB W | baseline µJ | embedded µJ | recon, map only | recon, +DRAM K/N | rec vs emb, map | rec vs emb, +DRAM |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| BCH(63,57) | 7 | 56 / 224 | 757.985 | 480.737 | 480.529 | 471.541 | **0.04 %** | 1.91 % |
| BCH(63,45) | 6 | 24 / 96 | 759.409 | 482.161 | 481.387 | 454.424 | **0.16 %** | 5.75 % |
| BCH(63,39) | 5 | 40 / 160 | 1240.578 | 770.654 | 480.197 | 444.246 | *37.69 % — NO* | *42.35 % — NO* |
| BCH(63,30) | 4 | published | 761.102 | 483.854 | 336.571 | 287.138 | **30.44 %** | **40.66 %** |

At q=7 and q=6 the mapping margin is the flat packing discount alone (0.21 and
0.77 µJ on the weight rows; Inputs and Outputs bit-identical between the arms),
so **the whole usable saving at the weak codes is the DRAM K/N fetch term** —
8.988 µJ (2.43 % of DRAM) and 26.963 µJ (7.29 %), against BCH(63,30)'s 49.433 µJ
plus 128.451 µJ of mapping saving on top.

**BCH(63,39)'s 37.69 % is the REFERENCE arm falling off a tiling cliff, and that
is measured** — a cliff the reference only reached because the withdrawn lcm
scheme kept changing its width. Under prompt_2's table the 8-bit arm is width 96
at every code and cannot move at all; this paragraph is the diagnosis of a
defect, not a finding about BCH(63,39). Reconstruction's own total is flat across all three weak codes
(480.529 / 481.387 / 480.197 µJ); the 8-bit arm is what moves
(480.737 / 482.161 / **770.654**). On the depth-16 scratchpad the width table's
depth renormalisation leaves the 8-bit reference holding **35 / 33 / 30
weights per PE** at q = 7 / 6 / 5, and at 30 it can no longer hold the
6,144-weight tile: it halves to 3,072 and pays `C = 16` where every other arm in
the study pays `C = 8`.

The control re-maps the q=5 pair at `ECC_WEIGHT_DEPTH_SCALE=1.1667`, which puts
the reference back at 35 weights/PE on the same width — its own gate passes at
0.00 % residual:

| BCH(63,39) at | baseline | embedded | recon, map only | emb held / room | `C` e→r | rec vs emb, map |
|---|---:|---:|---:|---|---|---:|
| depth ×1 | 1240.578 | 770.654 | 480.197 | 3,072 / 7,080 (43.4 %) | 16 → 8 | *37.69 %* |
| depth ×1.1667 | **758.375** | **481.127** | 480.357 | **6,144** / 8,260 (74.4 %) | **8 → 8** | **0.16 %** |

Two weights per PE of *reference* capacity move the reported margin from 37.69 %
to 0.16 %.

**Baseline and embedded are K-independent in the MODEL and K-dependent in this
RUN, and the difference is silicon, not code.** They share one 8-bit mapping;
the baseline's ECC cost is the flat `ECC_BASELINE_DRAM_PJ_PER_BIT` price (70 in
that run; 40 since 2026-09-11) with the parity traffic explicitly not charged; and `build_stacks()`'s only
K-dependent term, `ECC decode`, is zero under `ECC_DECODE=0`. They differ across
the rows above only because each code declares a different word width. **The fix
is a bigger scratchpad, not a different rounding rule**: at Eyeriss v1's
published 224 × 16 b spad the same widths renormalise to depths 64 / 149 / 90
(errors 0.00 / −0.22 / +0.45 %) and the reference holds **447–450 weights/PE at
every code** — a 0.7 % spread instead of 17 %. A BCH sweep with one baseline bar
needs that geometry.

### 2.5 The saving is in the PSUM path, not in weight refetch

Weight refetch is **1.000 on both arms at every depth** — its floor. `C` is the
only DRAM-level loop and `C` indexes Weights, so every weight is fetched exactly
once and a smaller weight buffer cannot make it worse.

What a smaller weight buffer does is force `C` into more chunks. **`C` does not
index Outputs**, so every chunk is a reduction step: partial sums for the same
outputs are written to DRAM and read back to accumulate, once per chunk. The
entire effect lives there.

It is a **staircase with a genuine optimum**: the reconstruction arm stays one
chunking step behind the embedded arm over a middle band of depths; above it both
fit the same chunk count and the gap closes, below it neither can hold the step
and the margin collapses.

### 2.6 The baseline's DRAM cost is a PRICE, not extra traffic

All three arms drive the same weight bits off the DRAM die — the decoder is on
it, so the conventional arm's parity is read, corrected and discarded there and
never crosses the datapath. The baseline therefore issues **no extra DRAM reads**.
What it pays for is an array that also holds the parity, and the indexing work the
other two arms do not do: a dearer per-bit price applied to its whole DRAM
category. Embedded and reconstruction share one array size and one price;
**reconstruction is the only arm whose traffic scales** (× K/N).

### 2.7 There is no consecutive weight reuse to amortise an encoder over

Consecutive weight reuse is **1 on 20 of 21** resnet18 layers. A latch between the
scratchpad and the MAC catches nothing; a register that would pay has to hold the
whole inner tile — a second scratchpad, not a latch. This killed the
retained-reconstruction boundary (R4b) on every dataflow in the study.

**Do not reintroduce a retained-reconstruction boundary without first re-measuring
consecutive reuse on the target mapping.** That is the step the original audit
skipped.

### 2.8 Report per DATASPACE, or a win looks like a null result

A weight-only table showed refetch `1.000 → 1.000` and level energy halving
exactly at every depth — i.e. nothing but the flat packing discount. It looked
like a null result on top of a large win, because the table had been built to the
hypothesis that the effect would appear in *weight* refetch. `dataspace_table()`
exists so that cannot happen again: reduction factor, refetch and energy **per
dataspace**, always printed by `--levels`.

---

### 2.9 The encoder toll cannot move the constrained nest; idle decides the ranking

> **STALE ON TWO COUNTS since 2026-09-12, and both move R3.** Clock gating is now
> the default (`ECC_RECON_CLOCK_GATING_PCT=99.5`), which re-ranks the boundaries;
> and prompt_7 Phase B stopped R3 borrowing the REFERENCE's plan — its geometry is
> R2's, and on R2's plan the mapper owns the `filter_glb` narrowing instead of the
> evaluator (§3.8: R3 +11.11 % → **+13.11 %** against conventional ECC on
> mobilenet_v2). The rows below are the pre-gating, pre-Phase-B numbers and are
> kept for the diff. **Regenerate before quoting.**


**2026-09-11**, prompt_6 phases 1–9, `eyeriss_like_wglb`, resnet18 `layer3.0.conv1`,
BCH(63,30), `aligned` packing, EDP, prompt_3's constrained mapspace, one job per
arm from `bash hpc/map_ert_arms.sh` (SLURM 41736521-24; the arm maps themselves
41735535/41735536). Gate on the reference arm: 179.22 µJ at victory 2000 / 4000 /
10000, **residual 0.00 %, PASS**. Prices: DRAM **20 pJ/bit**, baseline **40**,
MAC 0.23 pJ (§1; the table was re-billed from the same caches at 13:29Z the same
day when the price moved from 40/70 — no mapping touched). Result file
`results/evaluation/Post/eyeriss_like_wglb/resnet18/bch63_30/w8a8/layers1__layer3_0_conv1/…/20260911T132930Z__70efc0c7.json`,
figure `results/figures/ReconSweep_optimiser.png` (one-line heading; its caveats
are `title_caveats` in the manifest beside it); all 40 recorded checks pass
(prompt_6 §10 items 2–11, one per arm).

| bar | plan (mapping id) | total µJ | vs embedded | encoder µJ = incremental + idle | engines | who narrows on chip |
|---|---|---:|---:|---|---:|---|
| baseline | reference `a8de944a` | 483.854 | −61.81 % | — | — | — |
| embedded | reference `a8de944a` | 299.022 | 0 | — | — | — |
| R1 @ source | reference | **275.331** | **+7.92 %** | 1.026 = 0.052 + 0.974 | 1 | nothing on chip |
| R2 @ filter GLB | **own** (ERT, fp `5f46b2b186e5`, `6922b6f2`) | **275.201** | **+7.97 %** | 1.026 = 0.052 + 0.974 | 1 | mapper (Word bits 4) |
| R3 @ column edge | reference | 287.493 | +3.86 % | 13.998 = 0.361 + 13.637 | 14 | evaluator (Word bits 8) |
| R4 @ spad input | **own** (ERT, fp `91ce4687a22f`, `3faa4501`) | 436.208 | −45.88 % | 164.006 = 0.361 + 163.644 | 168 | mapper (GLB, Word bits 4) |
| R5a @ spad output | reference | 443.700 | −48.38 % | 173.763 = 10.119 + 163.644 | 168 | evaluator (GLB + spad) |

Positive `vs embedded` is a saving. DRAM saved is **24.716 µJ on every bar** (K/N of
the 47.186 µJ weight DRAM term at 20 pJ/bit); on-chip narrowing is worth 0.130 µJ at the GLB,
0.680 / 1.973 µJ on the two networks and 2.265 µJ at the scratchpads. Cycles are
**344,064 on every arm**, PEs **168/168 on every arm**.

**1. The ERT toll moved neither arm's loop nest** (on resnet18; on mobilenet_v2's 1×1 layers it does, §2.10). Both `recon2` (filter_glb.read
+2.80097 pJ, leak +2.8310811) and `recon4` (weights_spad.write +0.35012 pJ, leak
+2.8310811) came back **byte-identical to the reference nest** on the one shape,
so for both arms the ERT changed nothing (prompt_6 4.4.4, stated per arm). The
mechanism: an ERT arm narrows only the storage levels in its placement's `reduced`
set, which on both arms is `filter_glb` alone (`weights_spad` stays 8-bit), and
under prompt_3's free-sets `C` is not free at `filter_glb` — so the doubled GLB
room (Effective size 2048 → 4096 = 8/q exactly) cannot change the DRAM-level `C`
chunking that FINDINGS 2.5 identifies as the only capacity lever. A per-read toll
on the GLB cannot move the argmax either: GLB reads equal the weight count on every
nest of the family (§3.5). The leak term is a constant per cycle, and the nest
already sits at the family's minimum cycles with full PE occupancy. **So the
ERT-aware figure is Task 3's figure with RULE 3's idle term and measured ownership;
within this family the encoder's cost is purely additive.** A negative under the
constrained family is weaker evidence than one under the open space (2.2, caveat 3).

**2. Idle decides the ranking.** 168 engines × 344,064 cycles × 2.8310811 pJ =
**163.64 µJ, 54.7 % of the 299.0 µJ embedded run** — exactly the µJ prompt_6 4.3.2
predicted — so every PE-local boundary (R4, R5a) costs more than storing parity in
DRAM. The column-edge boundary (14 engines, 13.64 µJ idle) keeps 3.9 %; the
one-engine boundaries (R1, R2) keep 7.9–8.0 %, and their whole margin is the
24.7 µJ DRAM saving minus ≈1 µJ of encoder. Idle is 200–450× the incremental term at
every boundary except R5a, where reads = MACs make the incremental term 10.1 µJ.

**3. RULE 1 reconciles across owners.** `recon2`'s bill carries a mapper-narrowed
GLB (Word bits 4) at 24.503 µJ; `recon3` on the reference bill has the evaluator
narrow the same GLB by 0.1299 µJ from 24.632 µJ to 24.503 µJ — the two owners agree
to the printed digit. Every reduced storage stop has exactly one owner, recorded
per bar.

Caveats that travel with every number above: one layer; a constrained + relaxed
dataflow, not the published chip (label "constrained dataflow"); EDP objective,
under which the leak term also weights latency; `ECC_MAC_PJ_OVERRIDE=0.23`,
`ECC_DRAM_PJ_PER_BIT=20` and `ECC_BASELINE_DRAM_PJ_PER_BIT=40` as the
denominators, env.sh's defaults (§1). **DRAM-price sensitivity — the same caches
at the 2026-09-09/10 prices, 40 / 70 pJ/bit** (the figure first drawn at 08:58
that day, kept as `results/*/ReconSweep_optimiser__dram40_70__2026-09-11.*`,
result file `…/20260911T125827Z__23f43883.json`): the DRAM saving doubles to
49.433 µJ (K/N of 94.372) and the bars read baseline 761.102 µJ, embedded
483.854, R1 435.446 (+10.00 %), R2 435.317 (+10.03 %), R3 447.609 (+7.49 %),
R4 596.323 (−23.24 %), R5a 603.816 (−24.79 %); idle is then 33.8 % of the
embedded run. The encoder terms, cycles, ownership and nest verdicts do not move
with the price; only the DRAM band and every percentage do, linearly.

### 2.10 Full model on two networks: idle engines are the utilized PEs, and a PE count may legitimately differ

**2026-09-11**, `eyeriss_like_wglb`, BCH(63,30), `aligned`, EDP, prompt_3's constrained
mapspace, prices 20 / 40 pJ/bit and MAC 0.23 pJ (§1). All three arms (reference,
`recon2`, `recon4`) were already cached on every distinct shape — 12 for resnet18,
31 for mobilenet_v2 — so both studies are evaluation-only from
`ECC_RECON_MODEL=<model> ECC_RECON_LAYER=all ECC_RECON_ERT_AWARE=1 bash run.sh recon --eval`.
Figures `results/figures/ReconSweep_optimiser__resnet18.png` and
`…__mobilenet_v2.png` (the stem now carries the model, so the two never overwrite each
other), tables and manifests beside them; result files
`…/resnet18/…/layers-full/…/20260911T*__82b4c156.json` (40/40 checks) and
`…/mobilenet_v2/…/layers-full/…/20260911T*__575247dc.json` (37/37 after the DRAM-credit
check below was corrected). **Gate:** the convergence gate is one layer by doctrine
(`map_depth_sweep.sh`, prompt_6 8.4); `layer3.0.conv1`'s PASS at 0.00 % residual (§2.9)
is the gate these numbers stand on, and no other shape was gated.

The two evaluation guards that stopped the first attempts (jobs 41740440, 41742775) were
both wrong about Timeloop, not about the mappings:

**1. Timeloop bills leakage on the UTILIZED instances, not the declared ones.**
`buffer.cpp` sets `leaks_per_cycle` to the max utilized instance count when each instance
has its own power gate (`Instances sharing power gating: 1.00`, lines 2119–2127) and
charges `leak × total_cycles × leaks_per_cycle` (line 2376). Verified on all 43 cached
shapes of the two models: the measured multiplier on `recon4`'s `weights_spad` is the
utilized count every time — 168 where the array is full, 112 on the stride-2 1×1 layers,
98 on conv1, 16 on fc, 42/21/18 on the depthwise layers. The guard compared it to the
declared 168, which held on `layer3.0.conv1` only because that layer fills the array.
RULE 3's `N_engines` at a storage site is therefore the utilized instances **per layer**
(`StageStats.engine_cycles = Σ utilized × cycles`), the convention every arm's own
scratchpad leakage already follows; the declared count is reported beside it. Two
arithmetic details the fix needed: the reference's base leakage must be rescaled by the
utilization ratio as well as the cycle ratio before subtraction (on mobilenet_v2's G576
depthwise layer the reference uses 126 PEs and the arm 42, and rescaling by cycles alone
read as a multiplier of 41.99996), and the comparison is made in pJ against Timeloop's
0.01 pJ print precision.

**2. A PE count that differs between an ERT arm's own plan and the reference plan is
the arm's configuration acting through the mapper, and is reported, not refused.**
Seven of 43 shapes differ:

| shape | PEs ref / recon2 / recon4 | recon2 energy vs ref | recon2 EDP vs ref | recon4 cycles vs ref |
|---|---|---:|---:|---:|
| C576_M160 1×1 | 168 / 120 / 120 | 0.663 | 0.885 | 1.333 |
| C960_M320 1×1 | 168 / 120 / 168 | 0.676 | 0.947 | 1.000 |
| C320_M1280 1×1 | 140 / 100 / 140 | 0.771 | 0.987 | 1.000 |
| C64_M384 1×1 | 96 / 96 / 112 | — | — | 0.857 |
| depthwise G576 (14×14) | 126 / 126 / 42 | — | — | 1.000 |
| depthwise G960 (7×7) | 63 / 63 / 42 | — | — | 1.000 |
| depthwise G576 (7×7, s2) | 18 / 18 / 21 | — | — | 1.000 |

Every `recon2` case has lower Timeloop energy AND lower EDP than the reference plan: the
doubled `filter_glb` room changes the DRAM chunking (on C576_M160 it holds 3,840 weights
instead of 720 and reads outputs from DRAM 0 times instead of 54,880). The reference
itself fills the array on only 16 of 43 shapes, so prompt_6 10.9's 168/168 rule — written
from the one array-filling study layer — cannot hold on a full model. Timeloop does offer
a `type: utilization` mapspace constraint (`constraints.cpp` 887–898, `uber.cpp` 830),
applicable identically to every arm; it was **not** used: it is a mapspace change (every
cached mapping cold), a single floor is meaningless on layers that use 16–63 PEs, and it
removes EDP-optimal points by fiat. The per-shape rows (both PE counts, both cycle counts,
Timeloop's EDP ratio, nest identity) are printed, recorded on the result file
(`ert_<arm>_pes_used_vs_reference_reported_per_shape`) and carried in the manifest's
`title_caveats`; `PE!=` still suppresses Task 4's *capacity* verdict, a different claim.

**resnet18, whole model (21 layers, 12 shapes), µJ:**

| bar | plan | total | vs embedded | encoder (incremental + idle) | idle engines (cycle-weighted / declared) |
|---|---|---:|---:|---|---|
| baseline | reference | 19,494.5 | −66.03 % | — | — |
| embedded | reference | 11,741.7 | 0 | — | — |
| R1 @ source | reference | 10,701.9 | +8.86 % | 34.5 | 1 / 1 |
| R2 @ filter GLB | own (ERT) | **10,696.3** | **+8.90 %** | 34.5 | 1 / 1 |
| R3 @ column edge | reference | 11,099.7 | +5.47 % | 466.8 | 14 / — |
| R4 @ spad input | own (ERT) | 15,728.9 | −33.96 % | 5,151.3 (15.5 + 5,135.8) | 159.32 / 168 |
| R5a @ spad output | reference | 15,958.5 | −35.91 % | 5,453.4 | 159.32 / 168 |

DRAM saved 1,074.2 µJ on every bar. Both ERT arms returned the reference nest on 12/12
shapes — §2.9's one-layer verdict holds for the whole of resnet18 — so the figure is Task 3's
with the idle term; the idle engines are 159.3 rather than 168 because conv1 (98 PEs), the
stride-2 1×1 layers (112) and fc (16) do not fill the array.

**mobilenet_v2, whole model (53 layers, 31 shapes), µJ:**

| bar | plan | total | vs embedded | encoder | notes |
|---|---|---:|---:|---|---|
| baseline | reference | 12,154.4 | −84.66 % | — | — |
| embedded | reference | 6,582.0 | 0 | — | — |
| R1 @ source | reference | 6,276.1 | +4.65 % | 11.4 | DRAM saved 317.3 |
| R2 @ filter GLB | own (ERT) | **6,003.9** | **+8.78 %** | 11.7 | nest differs on 22/31 shapes, PEs on 3; DRAM 515.5 below embedded = 317.3 weight credit + 198.2 less activation traffic |
| R3 @ column edge | reference | 6,410.6 | +2.60 % | 153.0 | 14 engines |
| R4, R5a | — | unsupported | — | — | `G_rec` = 9: 3 and 15 (layer, stage) pairs keep fewer than 9 weights resident |

`recon4`'s own plan was mapped and passes every guard (nest differs on 21/31 shapes, PEs on
5, idle 911.5 µJ at 85.10 engines cycle-weighted of 168), but the bar is infeasible on this
network and is a table row only. **Here the ERT toll plus the doubled GLB room DID move the
nest**: §2.9's "C is not free at `filter_glb`" argument is about resnet18's 3×3 layers; on
mobilenet_v2's 1×1 layers the mapper found plans with less activation DRAM traffic, which is
where R2's margin over R1 (+8.78 % vs +4.65 %) comes from. The DRAM-credit check on an ERT
bar is now built from the arm's OWN DRAM bill (RULE 4) and records the activation-traffic
difference; the first version assumed the reference's bill and failed on exactly this bar.

Caveats: prices and dataflow as §2.9; `mobilenet_v2` was mapped under a mapspace free-set
written for resnet18's shapes (prompt_3), which is why the depthwise layers use 18–126 PEs
on every arm; percentages are against each network's own embedded bar and are not
comparable across the two panels.


## §3 — Verified mechanism facts

Each was measured, not modelled, and each is relied on by `prompt_6.md`.

### 3.1 `datawidth` narrowing is exact and free

CACTI receives `depth` and `width` **only** — `datawidth` never reaches the energy
model. Timeloop then bills `vector_access_energy / block_size` per weight, with
`block_size = width / datawidth`. So halving `datawidth` at fixed geometry exactly
halves per-weight energy and exactly doubles effective capacity, at **byte-identical**
per-access read/write/leak.

Measured on a matched `wdw8`/`wdw4` pair: `filter_glb` energy 159,221.51 →
79,610.76 pJ (exactly 0.5), utilised capacity 384 → 768, and the loop nest moved
(`for C in [0:16)` → `for C in [0:8)`).

**HARD CONSTRAINT: `width % datawidth == 0`, or `timeloop-mapper` aborts** —
`buffer.cpp:302`, `exit=134`. `block_size` defaults to 1 and is then checked, so
there is **no floor path**: a partially-filled word cannot be modelled at all.

### 3.2 Narrowing a storage level does NOT narrow the network

On the same matched pair, Network 0 reports `Word bits: 8`,
`Ingresses: 294912.00`, `Energy (total): 1783888.01 pJ` — **byte-identical in both
arms**. The evaluator therefore owns the network wire/switching saving. Do not
rewrite `network_word_bits`.

### 3.3 Timeloop bills `leak` per instance, per cycle

`weights_spad` 168 instances and `filter_glb` 1 instance both report
`Leakage energy (total)` = per-instance-cycle × instances × cycles, inside total
energy and therefore inside the mapper's objective. This is what lets a per-cycle
cost enter the ERT (prompt_6 RULE 3). Note the stats also print `Instances sharing
power gating`, and on one archived level the leakage total corresponded to 16
rather than the printed 64 instances — **read the multiplier off a real run.**

### 3.4 The embedded codeword layout

The codeword is n consecutive bits of the MSB-first weight bit stream, so weights
straddle codewords and the n−k lowest-significance positions carry parity. One
codeword spans `n / weight_bits` weights and touches up to `ceil(n/weight_bits)+1`
of them — **G_rec = 9** at BCH(63,K) over 8-bit weights.

### 3.5 A supplied ERT bypasses Accelergy and is billed exactly (prompt_6 phase 1)

Measured 2026-09-11 by `python3 -m eccenergy.experiments.ert_probe` on the
reference entry (fp `3cd00eb16801`, `layer3.0.conv1`, constrained mapspace),
SLURM jobs 41734429 and 41734507. Every number below was predicted from the cached
stats before the run and reconciled to the printed 0.01 pJ.

* **The hook works, on the production path.** Handing an `ERT:` and an `ART:` YAML
  to timeloopfe beside the design inputs puts both into the processed input
  unchanged (timeloopfe v4 declares them as top-level attributes and its v4→v3
  transpiler passes them through). `timeloop-mapper` then logs "Found Accelergy
  ERT … replacing internal energy model", never invokes `accelergy`, and bills
  the supplied prices.
* **Access action.** `filter_glb.read` 2.75566 → 1000 pJ moved `filter_glb`
  Weights energy 259,792.04 → 37,022,207.39 pJ, exactly
  `(reads × 1000 + fills × 4.29165) / block_size 8` from the cached counts
  (294,912 reads, 294,912 fills, 0 updates). No other level moved; the total went
  231.56 → 268.32 µJ, the level delta and nothing else.
* **Leak action.** +1000 pJ per instance per cycle moved `filter_glb` leakage
  25.52 → 344,064,025.52 pJ (× 1 instance × 344,064 cycles) and `weights_spad`
  84.24 → 57,802,752,084.24 pJ (× 168). **The leak multiplier is the full
  instance count on this design** (`Instances sharing power gating: 1.00`), so
  RULE 3's `idle × instances × cycles` enters the objective through `leak` with
  no correction.
* **Accelergy is deterministic.** Regenerated from the patched arch, all 30 ERT
  rows and the ART equal the cached ones bit for bit, so an ERT arm's base table
  can be produced at map time (phase 3) and does not depend on a cache entry
  existing. `datawidth` does not enter it (§3.1), so the `wdw4` arm's table is
  the reference's.
* **Observation, not a result.** With the read toll alone, under the constrained
  mapspace, the mapper returned the byte-identical reference nest: `filter_glb`
  reads equal the weight count on it, so a per-read toll on the GLB cannot move
  this layer's argmax within that family. The leak term was not in the mapper
  run.

Two facts phase 3 must build in:

1. With a supplied table Timeloop writes no `timeloop-mapper.ERT.yaml` /
   `.ART.yaml`, and timeloopfe's post-run parser then raises on the missing ART
   **after a successful search**; `timeloop.Mapper._map_now` would count that
   as a failed shape. Writing the supplied ERT and ART into the output directory
   under those two names before the call fixes it, and Timeloop leaves them
   untouched when it takes the "Found" branch (verified), so the cache entry then
   stores exactly the table the mapper saw, which is what RULE 4's read-back
   assertion reads.
2. `timeloop-model` refuses the mapper's own mapping while
   `architecture_constraints` are present ("mapping violates architecture
   constraints"); a fixed mapping is evaluated with the constraints removed. It
   also prints no "Found Accelergy ERT" line; its proof is
   `Vector access energy source : ERT` in the stats and no `.accelergy.log`.

---

### 3.6 What the ERT-arm runs verified (prompt_6 phases 3–9)

* **Leakage is outside the per-dataspace energies — in the REPORT, never in the
  MAPPER'S OBJECTIVE.** *(corrected 2026-09-12, prompt_7 Phase A step A4; the
  earlier wording below said only the first half and read as "the model has no
  leakage", which is wrong.)* Timeloop prints `Leakage energy (total)` as a
  separate per-level line, and `parse_stats` aggregated only the per-dataspace
  `Energy (total)` rows and Compute, so the RAW RECORD never held leakage
  (25.52 + 84.24 pJ on the reference — invisible until the ERT leak bump made it
  163.6 µJ). But Timeloop's own `Energy:` and `EDP(J*cycle)` — **the quantity the
  mapper minimises** — already contain it, verified arithmetically on
  `C128_M128_R3_S3_P28_Q28`, recon2 arm: `filter_glb` Weights 143,772.48 pJ +
  leakage 76,777.31 pJ = 220,549.79 pJ over 115,605,504 computes = 1.908
  fJ/compute, and Timeloop printed 1.91 (without leakage it would print 1.24).
  **So the asymmetry was sharper than "leakage is missing": the mapper saw it, at
  prices 10³–10⁴ too low, and the report threw the same number away.** That is
  reporting rule R-4 (prompt_7 §12), and since Phase A the parsed value
  reaches the record (`Raw.standby["timeloop_leakage_pJ"]`) whether or not
  `ECC_STATIC_ENERGY` charges anything. Consequence for prompt_6 5.3: only the ACCESS toll is inside the
  bill and is moved into `Reconstruction`; the idle term is verified against the
  stats' leakage delta and charged by the evaluator. The first attempt subtracted
  both and drove `filter_glb` negative.
* **The split reconciles to 1e-6.** Stats-side vs evaluator: `recon2` access
  51,627.39 / 51,627.39 pJ and leak 974,073.09 / 974,073.09 pJ; `recon4` access
  361,391.72 / 361,391.72 pJ and leak 163,644,278.72 / 163,644,278.72 pJ.
* **The leak multiplier is the declared instance count** on both arms (1.000000
  and 168.000000, `Instances sharing power gating: 1.00`).
* **Two arms with byte-identical YAML land in different directories**
  (`…__wdw4-filter_glb__mcons__wrelax__ert-recon2-filter_glb-read/fp-5f46b2b186e5`
  and `…ert-recon4-weights_spad-write/fp-91ce4687a22f`); the read-back of every
  stored ERT against the reference entry's table found only the two bumped rows
  moved. The reference fingerprint `3cd00eb16801` did not move through phases 2–8.
* **A quantisation arm delivers 8/q, not N/K:** `filter_glb` Effective size
  2048 → 4096, ×2.0000 at q = 4 (N/K = 2.1).
* **Timeloop's own totals** (leakage included) on the one shape: reference
  231.56 µJ, `recon2` 232.46 µJ, `recon4` 395.44 µJ — the last is the reference
  plus 168 encoders' idle for the whole run.

### 3.7 prompt_7 Phase A: the roofline reproduces Timeloop, and standby is symmetric

*(2026-09-12. Evaluator-only; no mapper run, no cache colded, every pre-Phase-A
number reproduced to the pJ.)*

* **The roofline is Timeloop's own timing model with one missing term, not a
  second one.** With no off-chip bandwidth declared, `latency_post.roofline()`
  reproduces Timeloop's per-level **and** total cycle counts EXACTLY on all 43
  cached shapes of `eyeriss_like_wglb` (301 storage levels) — including the 11
  shapes `ifmap_glb` throttles and the 5 `psum_glb` throttles. The arithmetic
  asserted: demand = items per COMPUTE cycle, slowdown = `min(1, declared /
  demand)` per port, level cycles = `ceil(compute / slowdown)`, run = the WORST
  level. `tests/test_latency.py` is the gate.
* **Only `ifmap_glb` and `psum_glb` ever throttle, and only under the
  CONSTRAINED mapspace.** Worst 0.184 (11 shapes) and 0.610 (5 shapes) — the
  values printed as 0.18 and 0.61. On the retired `random_pruned`/unconstrained
  cache `filter_glb` throttles to **0.320**, so reporting rule R-2 is a statement
  about the dataflow in force, not about the design. `filter_glb` never
  throttles under the live regime, which is not the same claim as never binding:
  at the FC layers it sits exactly ON its declared 16 items/cycle.
* **The one place time responds is off chip.** A narrow weight is still one
  item, so `weight_scale` applies to DRAM only; on chip the item count is
  identical by construction and a bit-aware port is an architecture change the
  mapper must see (Phase C). On `C512_M1000_R1_S1_P1_Q1` (resnet18 `fc`) at
  480 MB/s the reference takes 1,082,317 cycles and BCH(63,30) 523,587 — the
  51.6 % an FC layer's ~99 % weight traffic predicts.
* **Latency is FLAT across the five boundaries, as predicted.** mobilenet_v2 at
  BCH(63,39), 480 MB/s: baseline and embedded 72,931,937 cycles, every one of
  R1–R5a 69,784,911 — all five at the ceiling of **4.32 %** = weight share
  11.3 % of off-chip items × (1 − K/N) 0.3810. The whole gain is at the DRAM
  boundary, which every placement shares by construction (rule R-1).
* **Standby energy is charged to all three arms or to none.** It lands in
  `Raw.base["Standby"]`, which every arm and every placement bar starts from, so
  no code path can give it to one arm and not another. mobilenet_v2, all six
  bars moved by exactly **+42.050 µJ** at `ECC_STATIC_ENERGY=1`; the ECC result
  moved from +11.160 % to +11.095 % against conventional ECC.
* **The replacement densities are 3,000–4,600× Timeloop's own leakage bill** on
  the same plan (42.050 µJ against 0.013938 µJ on mobilenet_v2; ×4,573 on the
  reference-arm shape set). Squarely in §5.2c's 10³–10⁴, and measured rather
  than asserted. Billed on **utilized** instances, as Timeloop bills leakage
  (`leak × utilized × cycles`): `filter_glb` 7.39799e-05 × 32000 × 1 = 2.37 pJ,
  `weights_spad` 2.67882e-06 × 32000 × 16 = 1.37 pJ, both to the printed
  precision. DRAM is charged nothing, stated rather than left as a zero.
* **A cache slug decides what is being measured.** An `ert-` arm was mapped with
  the encoder's toll in its ERT, so its `Leakage energy (total)` is 147,351 pJ
  against the reference arm's 14 — reading one while testing the replacement
  densities measures an encoder. The suite resolves the LIVE configuration's
  reference cache and refuses rather than falling back.

### 3.8 prompt_7 Phase B: six chips, and R3 was on the wrong plan

*(2026-09-12. Evaluator-only; no mapper run, no cache colded. `eyeriss_like_wglb`,
mobilenet_v2, BCH(63,39), `ECC_RECON_ERT_AWARE=1`, gating 99.5 %.)*

* **The mapper sees SIX chips on this design, not three.** `recon.mapper_arms()`
  derives them from prompt_7 §6.4's three axes — `datawidth: q` × the ERT bump ×
  the declared per-dataspace bandwidth scale — and gets six here and five on
  `eyeriss_v2_like`, which has four boundaries. A NETWORK stage is carried in
  the bandwidth-scale set and marked `no-op`: Timeloop has no network timing
  model to apply it to, but a boundary that adds one is still a different
  declaration, and dropping those entries collapses v2's R1 and R2 into one chip
  (5 arms → 4). Measured, and the mutation is in `test_mapper_arms.py`.
* **R3 was billed from the wrong plan, and it cost 2.00 pp.** Its geometry is
  R2's — both narrow `filter_glb` and nothing else — and it was on the
  REFERENCE's. Billed from R2's plan instead:

  | bar | was (reference plan) | now (R2's plan) |
  |---|---:|---:|
  | R3 total | 6,376.502 µJ | **6,232.748 µJ** |
  | vs conventional ECC | +11.11 % | **+13.11 %** |
  | vs embedded only | +3.58 % | **+5.75 %** |
  | who narrows `filter_glb` (RULE 1) | evaluator, ×0.6316 | **mapper, ×1.0000** |
  | reconstruction | 9.094 µJ over 2,012,784 events | 7.718 µJ over 1,688,225 |

  The ownership row is the mechanism: on R2's plan `filter_glb`'s `Word bits` is
  5, so the mapper narrowed it and the evaluator applies 1.0. **No other bar
  moved.** R1 stays on the reference plan and that is now a positive statement
  rather than a default — R1 narrows nothing on chip, so the reference's
  geometry IS its geometry.
* **R5a is a chip that has never been mapped, and now says so.** It is the only
  bar that narrows `weights_spad`; no solved arm does, so no plan on disk knows
  that level is narrower. It is billed from the reference plan with
  `geometry_matches: false` and an empty candidate list, and the run warns. (On
  mobilenet_v2 it is separately `unsupported` — 14 layers keep fewer than
  `G_rec` = 9 weights resident — so the flag is a statement about the plan, not
  about this figure's bars.)
* **`ert_injectable()` condition 3 no longer excludes R5a, and that is derived.**
  The bump has two rows and the exclusion needs BOTH constant. The access row on
  the innermost level's `reads` is the MAC count and is; the per-cycle `leak` row
  is `idle × (1 − g)` and Timeloop bills it as `leak × utilized instances ×
  cycles`. Measured from the live DC constants: **2.7891299 pJ/cycle/instance at
  PCT=0, 0.0139456 at 99.5, exactly 0 at 100** — so R5a is ERT-injectable at both
  settings §11 item 7 requires side by side, and not at the fully-gated limit.
  Renaming the placement moves nothing.
* **A bar can now pass the latency ceiling, and the cause is NOT reconstruction.**
  The ceiling is `weight share × (1 − K/N)` on the REFERENCE plan's off-chip
  traffic, so it bounds the weight term only. R2/R3/R4, billed from their own
  plans, reach **5.84 % against a 4.32 % ceiling** — at *identical* weight reads
  (×1.000) and **×0.983 activation items** (30,508,939 against 31,042,059). The
  extra 1.52 pp is the arm's own loop nest moving less activation traffic off
  chip: a MAPPING effect. (Those plans are also *slower* by Timeloop's own count
  — 3,937,072 cycles against 3,855,536 — and faster under the 480 MB/s roofline,
  which is what a DRAM-bound machine looks like.) Every row now prints `wt` and
  `act`, and a check refuses a bar past the ceiling at ×1.000 on both.
* **Phase B colded nothing.** `reference` `718d53aac189`, `recon2` `a18a5b15fd8b`
  and `recon4` `55569ac34427` are the directories that were already on disk, at
  43 solved shapes each. `recon1` `718d53aac189`, `recon3` `64af0cb88d0c` and
  `recon5` `9211adf270da` are cold — Phase C's work. `bash run.sh baseline
  --eval` still gives Task 1 = 317,290,433.48952 pJ.
* **One collision was real, not theoretical.** R1's patched YAML **is** the
  reference's until Phase C1.2 declares the bandwidth scale, so an arm with no
  ERT bump and no narrowed level produced the reference's cache slug exactly.
  `arm-<key>` for the bump-less arms is what keeps them apart; the arms that
  have a bump keep prompt_6's `ert-<key>-<level>-<action>` spelling byte for
  byte, which is why nothing on disk moved.

### 3.9 prompt_7 Phase C1: TIME reaches the mapper, and two constants were wrong

**2026-09-13. Authored and validated; NO MAPPER JOBS SUBMITTED.** Every arm and every
design is now cold at a new fingerprint, on purpose. The numbers below are what C1
MEASURED while building it; the placement study's own numbers are Phase C2's.

**a. The off-chip limit binds, and it is one bus.** `shared_bandwidth: 2.40` on the DRAM
level (480 MB/s at the published 200 MHz) — not `read_` + `write_bandwidth`, because the DQ
bus is one wire set whose limit is on their SUM, which is the term the roofline already
charged. On a real search (SLURM 41915350, mobilenet `classifier.1`) DRAM prints
`Bandwidth throttling : 0.30` and is the binding level. Before this, no architecture in
`archs/` declared an off-chip bandwidth at all and off-chip traffic cost ZERO cycles.

**b. The bit-aware port moves the `fc` layer, and moves the PLAN.** Two real mapper
searches on `C1280_M1000_R1_S1_P1_Q1`:

| arm | `filter_glb` read bw | Bandwidth Consumption Scale | **Cycles** | Energy | PE utilisation |
|---|---:|---:|---:|---:|---:|
| reference | 16.00 | 1.00 | 540,117 | 89.88 µJ | 1.41 % |
| `recon2` | **25.60** | **0.62** | **336,943 (−37.6 %)** | 89.13 µJ | **2.26 %** |

Below that layer's own 51.7 % ceiling (§A.6), and the utilisation moved — so the mapper
chose a DIFFERENT plan, not the same plan re-timed. **This is prompt_7's C2 gate 4 passing
on one shape before the matrix is committed.** What C2 measures is how much survives
aggregation, where `classifier.1` is 2.10 % of mobilenet's cycles.

**c. A misspelled dataspace exits non-zero.** `{Weightz: 0.619048}` → Timeloop return code
1, no `stats.txt`. The declaration cannot degrade into a silent no-op.

**d. Banking was inert, and it is worth 30 % on a GLB read.** Upstream's `smartbuffer_SRAM`
declares no `n_banks`, so every `n_banks:` in this study was dropped and CACTI priced one
monolithic array. With `smartbuffer_SRAM_banked` forwarding it (Accelergy, 45 nm, this
design's own geometry):

| level | banks | read, monolithic | read, banked | |
|---|---:|---:|---:|---:|
| `ifmap_glb` | 13 | 23.5387 pJ | **16.5710** | ×0.704 |
| `psum_glb` | 12 | 22.7292 | **16.0527** | ×0.706 |
| `filter_glb` | 2 | 13.5910 | **11.7451** | ×0.864 |

**Only levels that declare `n_banks` themselves are switched.** timeloopfe v4 hands EVERY
storage level a default `n_banks: 2` — visible in the flattened architecture, published for
none of them — and the CACTI wrapper floors depth at `64 × n_banks`, so on the depth-3
`weights_spad` that default moves the price through the FLOOR rather than through any
banking. **Recorded, not compensated:** CACTI is called at `2**ceil(log2(n_banks))`, so the
13-bank ifmap GLB is modelled as 16; the wrapper computes `bankscale = 13/16` and never
applies it (`cacti_wrapper.py:177-178` — dead in the plug-in), while leakage uses the
declared count.

**e. `ECC_RECON_IDLE_PJ` had never been rescaled for the clock, and it is ×5.** `env.sh`
§6 TRAP 2 documents the rule in full and says where it belongs; it had never been written,
because until C1.5 gave a design its own clock the factor was 1.0 on every run. The term is
pJ **per cycle** measured by DC at 1 ns and it is CLOCK power (§5.3: the DC "idle" constant
is 99.48 % clock, 0.52 % true leakage), so a 5 ns cycle burns five times as much —
**on the reconstruction engines only, i.e. on the side of the comparison this study
measures.**

| | at DC's 1 ns | at 200 MHz |
|---|---:|---:|
| BCH(63,39) idle | 2.7891299 | **13.9456495** pJ/cycle/engine |
| BCH(63,30) ERT `leak` row | 2.8310811 | **14.1554055** |

The second row is the worked example `env.sh` already spelled out, now produced by the
code. The INCREMENTAL term is per codeword (CV²) and is NOT rescaled; `ECC_LEAKAGE_NW` is
POWER in nW and is NOT rescaled either — which is why the two are declared in different
units. Effect on the sweep arm, on one raw record: **recon +0.1699 %**, baseline and
embedded **bit-identical**.

**f. Declaring the scale opened a double-count, caught before it shipped.** The evaluator
has applied `weight_scale = K/N` to the off-chip weight items since Phase A, because no
architecture declared a bandwidth scale. The moment C1.2 declares one, the MAPPER applies
it too, and the reported saving would have been K/N **squared** — 0.6190 → 0.3832 on
mobilenet's BCH(63,39), a 38 % "latency saving" that is arithmetic.
`latency_post.relief_owner()` reads the answer off each bar's OWN stats
(`Bandwidth Consumption Scale` = 1.00 → the evaluator owns it; = K/N → the mapper does),
applies it exactly once, and REFUSES a third value as a bar billed from another code's plan.
prompt_6 RULE 1, measured per bar rather than assumed.

**g. The MAC price the mapper optimised against was never the one the report charged.**
`ECC_MAC_PJ_OVERRIDE = 0.23` is evaluator-side, so the search priced a MAC at Accelergy's
1.13555 pJ — 4.9× — and under `ECC_OPT_METRIC=edp` that can move the argmax. Every arm now
gets a supplied ERT with each `compute` row **`set`** to 0.23, so `apply_mac_override`'s
rescale ratio is exactly 1.0 and the two cannot double-count.

**h. The first C2 launch died in ten seconds, and the gate had not lied — it had been
asked the wrong question.** 72 jobs, all FAILED, the dependent eval on
`DependencyNeverSatisfied`. Three bugs, and all five C1 gate checks were green on every
one of them, because **every check ran without invoking the mapper**: `validate`,
`diagnose`, `--dry-run`, the fingerprints and a patched-YAML read construct no `Mapper`,
build no ERT and patch no YAML through the code a job runs.

| bug | what it would have cost |
|---|---|
| a `@property` inserted into the middle of `Mapper.__init__` — a `def` ENDS the function, so `self._memo` and five counters never existed | a crash, loud and immediate |
| **a `depth: 1024` written in a COMMENT read as the declaration**, and `re.sub(..., count=1)` then rewrote THE COMMENT; `filter_glb` declared `depth: 256` beside a fresh `width: 384` | **NOT a crash.** 98,304 bits where 16,512 were intended — six-fold, on the level R2's whole saving rides on — reported in one line of stdout and nowhere else |
| `ErtTables.ensure()` dereferencing `self.bump['placement']` on the reference arm, which since C1.6 has a table and no bump | a crash, on the first job of every matrix |

**The second one is the finding.** Had the first not killed the job, the matrix would have
mapped a six-fold-too-large weight buffer and returned plausible numbers. The fix is not
the comment: `archs.uncommented()` masks comments while preserving positions, 14 geometry
reads go through it, and `write_attr()` RAISES when a rewrite lands on nothing — the silent
half. **Any comment naming `depth:`/`width:`/`datawidth:`/`n_banks:` would have done it**,
in a file where comments are the project's main defence against exactly this class of error.

**C1 gained a sixth gate: one real shape, two arms, through `run.sh map` on a compute
node** (`Claude-sandbox/_c2_smoke.sbatch`). Two jobs and a few minutes.

### 3.10 prompt_7 Phase C2: the matrix ran — and RULE R-2 inverts

**2026-09-13. 72 map jobs (resnet18, 12 shapes × 6 arms), all COMPLETED; five defects
between the maps and the figure, every one of them in the EVALUATOR.** The expensive half
was right the first time.

**THE RESULT** (resnet18, BCH(63,39), Eyeriss v1 at its published 200 MHz, 120 MB/s off
chip, **all six arms on their own plans — zero borrowed**, which is what Phases B and C
were for):

| | |
|---|---:|
| lowest-energy feasible placement vs **conventional ECC** | **+15.552 %** |
| the same vs **embedded only** | **+9.961 %** |
| baseline / embedded | 61,905,881 cycles = 309.529 ms |
| every reconstruction bar | **54,132,176 cycles = 270.661 ms, −12.56 %** |
| ceiling: weight share 33.0 % × (1 − K/N) 0.3810 | **12.56 %** |

The bars sit **ON** the ceiling — to 9.3e-09 — and are **FLAT across all five boundaries**.
That is rule R-1 by construction: every placement's `reduced` set contains `dram`, so every
placement gets the same off-chip relief, and at 0.6 items/cycle against a 4.26 demand the
design is hard off-chip-bound, so the relief converts 1:1 into time. **Latency does not rank
the boundaries. Energy does.** §4.7 projected exactly this before the compute was spent.

**RULE R-2 IS RESTATED, and it is the real finding of Phase C.** R-2 said the levels that
bind carry INPUTS and PARTIAL SUMS, which reconstruction cannot touch — `ifmap_glb` 11 of 43
shapes, `psum_glb` 5 of 43, DRAM never. **That was never a fact about the chip.** It was the
absence of a declared number: no architecture declared an off-chip bandwidth, so Timeloop
skipped the DRAM throughput check entirely. C1.1 declares it, and **DRAM binds on 21 of 21
layers at worst throttling 0.074** — the binding resource is now exactly the one
reconstruction relieves. `config.reporting_rules()` and `tests/test_latency.py` carry the
restatement, and the test asserts it in BOTH directions: DRAM throttles if and only if the
cache's own DRAM level declares a bandwidth.

**BOTH MODELS, and the contrast IS the result** (2026-09-13; mobilenet_v2 ran after
resnet18, 31 shapes × 6 arms, same six arms on their own plans):

| | resnet18 | mobilenet_v2 |
|---|---:|---:|
| weight share of off-chip traffic | 33.0 % | **10.5 %** |
| latency ceiling = share × (1 − K/N) | 12.56 % | **4.00 %** |
| **measured latency saving** | **12.56 %** | **3.97 %** |
| vs conventional ECC | **+15.552 %** | **+10.974 %** |
| vs embedded only | +9.961 % | +3.285 % |
| lowest-energy placement | `recon_weight_glb_output` | `recon_source_noc_ingress` |

**The ceiling IS the weight share**, and this is §4.6's Reason 1 measured on two networks:
mobilenet_v2 is depthwise-separable, so its weights are a tenth of off-chip traffic against
resnet18's third, and the most any boundary can buy falls with it. Both land AT their own
ceiling — mobilenet at 3.97 against 4.00 because its arms' own plans move ×1.003 of the
reference's weight items, a MAPPING effect reported in the table's `wt` column. Latency is
FLAT across all five boundaries on both (R-1), and **the best placement differs by model**,
which is an ENERGY ranking and not a latency one — exactly the division §4.7 projected.

**A Lustre race cost one job of 186 on the mobilenet matrix.** Job 41920766 WROTE
`globals_eyeriss_like_wglb.yaml` at line 35 of its own log and hit `FileNotFoundError` on
that path at line 78, eight seconds later. All 186 jobs write the same two shared files and
`_write_atomic` REPLACED them in place — atomic in POSIX terms, and still not safe here: a
Lustre client holding a handle to the old inode sees it go stale. One failed job is enough
to leave the dependent eval on `DependencyNeverSatisfied`. **The fix is the NAME:** both
files are a pure function of (architecture, configuration), so both are content-addressed
and created with `O_EXCL`; nothing is ever replaced. No fingerprint moved and no compute was
lost — `arch_fingerprint()` hashes the CONTENT of those files, never their paths — so the
repair was one map job of 64 s.

**Two of the five evaluator defects are worth keeping in mind, because both produced NUMBERS
rather than crashes:**

* **The roofline re-timed every reconstruction bar as if its relief did not exist** —
  61,905,882 cycles against Timeloop's own 54,132,184 (×1.1436), while the reference
  reproduced Timeloop at ×1.0000. One arm exact beside five at ×1.14 *is* the Phase A gate
  failing. The cause was a wrong model of the double-count: **the roofline does not adjust
  Timeloop's cycles, it recomputes them**, so the scale belongs in that computation exactly
  once however many other places applied it.
* **One cycle, and "exact" turned out to be unachievable.** With an off-chip limit
  declared, both sides run a float ratio through a `ceil`. Sweeping all five candidate
  orderings over the whole cache — **301 levels, 43 shapes** — the best two miss 2 levels
  each, by exactly +1 cycle (6.4e-07 of the count); `ceil(items/rate)` misses 4.
  **No ordering reproduces all 301**, because Timeloop's own float ordering is not
  recoverable from the stats file. The gate now asserts the achievable precision — no level
  more than ONE cycle out **and** at least 95 % exact (measured 299/301 = 99.3 %), with the
  mutation at +2 — rather than an "exact" that was only ever true while DRAM had no declared
  limit. Both the bound and the exact fraction are asserted, so a systematic drift cannot
  hide inside the allowance.

---

## §4 — Timeloop defects worth knowing

1. **`timeout: 0` does not disable the criterion — it makes it fire immediately.**
   `search_size_` and `victory_condition_` guard with `X_ > 0 &&`; the timeout
   check guards on the *counter*, not the setting. All 18 threads quit with
   "0 invalid mappings", job dead in 16 s.
2. **`ECC_MAPPER_TIMEOUT` is algorithm-dependent.** Under `random_pruned` it never
   fires. Under `linear_pruned` at the default 2000 it is fatal: the linear walk
   begins where 100 % of the first 36,000 mappings are infeasible, so every thread
   quits before finding one valid mapping and the job dies in 10 s. Hence the
   `100000000` in §1.
3. **The 18 threads never share a best-so-far.** `sync_interval` is null and the
   sync is guarded by `sync_interval_ > 0`, so the reported mapping is the min over
   18 *independent* searches, each stopping against its own local best.
4. **Timeloop's wire model is a stub** (`pat.cpp:81`, `WireEnergy(...) {return 0;}`).
   Every NoC cost 0.00 pJ — in the evaluator *and in the mapper's objective* —
   until `archs/_shared/noc.yaml` supplied coefficients. Costing it therefore
   changes which mapping wins; it cannot be applied to cached mappings.

---

## §5 — Guards, and what each one caught

Each exists because it caught a reported positive that was an artifact.

| guard | what it caught |
|---|---|
| `capacity` verdict requires `held` to **RISE** | a "win" where the arm stored strictly less on chip and read less |
| `PERM?` verdict | reads moved while `held` was identical — the cause was loop ORDER, reachable at the declared capacity too |
| `PE!=` verdict | arms differing in PE count, so capacity and parallelism moved together and neither is separable |
| `sibling_fingerprints()` | a headline comparing two different *architectures* filed under one variant slug |
| `validate_placement_space()` | a stage added to `WEIGHT_PATHS` with no boundary in `PLACEMENTS`, understating every bar below it with nothing saying so |
| `assert_pair_geometry()` | two arms of a pair declaring different width or depth, which voids the comparison |
| `assert_onchip_narrowing_once()` | the mapper and the evaluator both narrowing, squaring the saving |
| `energy.load_raw()` shape check | a stale aggregate served forever after a workload fix |

---

## §6 — Open defects

1. **Two workload families need re-mapping before they are plotted.**
   mobilenet_v2, efficientnet_b0, convnext_tiny and xception hold mapper entries
   under pre-fix shape names. They are never hit again (new names), but they are
   cold. **The three sweep figures are affected (2026-09-11, prompt_6 phase 5):**
   regenerated under RULE 3 from what is cached at the current fingerprints,
   `BCHsweep.png` holds 4 of resnet18's 21 shapes on eyeriss_v2_like,
   `ArchitectureSweep.png` holds eyeriss_like and simple_weight_stationary in
   full, eyeriss_v2_like 4/21 and the other three designs not at all, and
   `ModelSweep.png` could not be regenerated (eyeriss_like is cold at its current
   fingerprint under the `ss20000` slug) and is stale on disk. Their manifests
   record the per-layer coverage; none is a model total.
2. **`eyeriss_v2_like_wglb` does not validate.** Its `weight_glb` stage has no
   boundary. prompt_6 Appendix B has the fix; out of scope until Eyeriss v1 is done.
3. **`Packing.storage_scale()` and `onchip_narrowing_audit()` disagree.** The audit
   keys off the packing mode's *name*; the narrowing is a measured multiplier. Under
   `aligned` with a power-of-two `q` in a byte-multiple word, `storage_scale` returns
   0.5 while the audit reports zero narrowing sites. Not yet live in any run — prompt_6
   RULE 1 fixes it.
4. **`round` vs `ceil` for `q`, and the two disagree on two codes.** The mapping
   study uses `q = round(8·K/N)` via `code_widths.declared_datawidth()`;
   `recon.py`'s `Packing` still uses `ceil`. They agree everywhere except
   BCH(63,57) (ceil 8, round 7) and BCH(63,51) (ceil 7, round 6), so Task 3 narrows
   those two codes **less** than the mapping study does. Worse, at `q = 8` the
   "recon" arm *is* the embedded arm, so a gate run that way compares embedded with
   itself and reports an effect of exactly zero. Only BCH(63,30) is affected by
   neither, which is a further reason it is the code in use.
5. **The NoC switching term is an assumption.** 0.25 pJ per flit at v2's
   router-cluster level and `simba_like`; no paper publishes a pJ/router.
6. **E_background and E_refresh are not modelled** (both 0). An arm storing fewer
   weight bits would save both, so this understates the embedded and reconstruction
   arms alike.

---

## Reproducing any of it

`env.sh` is the only file to edit. `bash hpc/tl.sh bash run.sh <stage>` inside an
allocation; never the mapper on a login node. Full detail behind every claim above:
`legacy/FINDINGS_detail_2026-09-11.md`.
