# FINDINGS

The only place empirical claims about the current model live.

**The live plan is `prompt_6.md`.** Where this file and that one differ on what to
do NEXT, prompt_6 wins; this file records what was LEARNED.

**Slimmed 2026-09-11.** Full working detail — every measurement pass, every
withdrawn result, every table — is archived verbatim in
`legacy/FINDINGS_detail_2026-09-11.md` (and the two earlier snapshots beside it).
Nothing was deleted, only moved. `legacy/FINDINGS.md` is pre-rewrite; do not quote it.

---

## ⚠ ENERGY NUMBERS ARE PENDING REGENERATION

Every reconstruction percentage this project has produced was computed with

    recon_pj = incremental_per_codeword + idle_per_cycle

— a per-codeword quantity added to a per-cycle quantity, then multiplied by the
codeword count. prompt_6 RULE 3 separates them:

    E_recon = incremental x events  +  idle_per_cycle x cycles x N_engines

**So every `% vs embedded`, every µJ total, and every ranking that includes an
encoder cost will move.** They have been removed from this file rather than left
to be quoted by mistake. `BCHsweep`, `ModelSweep`, `ArchitectureSweep` and every
`ReconSweep*` figure are invalid until regenerated under prompt_6.

What is NOT affected, and is kept below: cycle counts, PE counts, refetch ratios,
buffer fill percentages, capacity ratios, convergence residuals, and every
mechanism.

---

## Status

| Task | State |
|---|---|
| 1 conventional ECC, external parity | implemented; totals pending regeneration |
| 2 embedded ECC (DRAM effect only) | implemented; totals pending regeneration |
| 3 reconstruction placement, fixed mapping | implemented, 5 boundaries on `eyeriss_like_wglb` |
| 4 reconstruction-aware mapping by **capacity dilation** | ❌ **withdrawn.** No clean capacity win across 196 EDP mappings on 5 designs, and the sweep ran unconverged. Cause is arithmetic, not search — see §2.4 |
| 4′ same question by **`datawidth`** | mechanism verified (§3.1), measured under the constrained mapspace |
| 5 **reconstruction-aware ERT** | **NEXT — `prompt_6.md`** |
| convergence of the mapper search | ✅ solved by constraining the mapspace, not by raising the budget (§2.2) |
| Eyeriss v1 = `eyeriss_like_wglb` | DECIDED 2026-09-10, registered, validates |
| placement study on other designs | not started; `eyeriss_v2_like_wglb` refused until its GLB boundary exists (prompt_6 Appendix B) |

---

## §1 — Settings every future number must be produced at

`eyeriss_like_wglb`, resnet18, BCH(63,30), 8-bit weights, 45 nm, 1 ns cycle,
**NoC costed**, `paper` fidelity, `ECC_RECON_PACKING=aligned`, **18 threads**
(in the mapping fingerprint — change it and every cache is cold).

**Objective `edp`, never `energy`.** Under `energy` the mapper serialises and
trades PE count for capacity; that is what withdrew an earlier result.

**Search:** `ECC_MAPSPACE_CONSTRAIN=1` + `ECC_WEIGHT_FACTOR_RELAX=1`, algorithm
`linear_pruned`, `ECC_MAPPER_TIMEOUT=100000000`, `ECC_VICTORY=4000`. See §2.2 for
why this combination and not a bigger budget.

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

### 2.4b The measured BCH sweep, and the one number in it that must not be quoted

**2026-09-11**, `eyeriss_like_wglb`, `layer3.0.conv1`, no depth scaling, one
mapper PAIR per code on identical declared silicon. Gate PASSES on every row at
residual **0.00 %**. `results/tables/EyerissV1_bch_sweep_spad16.csv`.

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
is measured.** Reconstruction's own total is flat across all three weak codes
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
the baseline's ECC cost is the flat `ECC_BASELINE_DRAM_PJ_PER_BIT=70` price with
the parity traffic explicitly not charged; and `build_stacks()`'s only
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
   cold.
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
