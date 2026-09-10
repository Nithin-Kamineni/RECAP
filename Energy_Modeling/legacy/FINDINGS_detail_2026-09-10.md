# FINDINGS — archived working detail, 2026-09-10

Sections lifted verbatim out of `FINDINGS.md` on 2026-09-10 because they are
fully superseded and their length was crowding the live file. Nothing here
describes what the code does now. Kept for provenance and because both
lessons generalise.

* **§7.1 The R4b register audit** — R4b was removed 2026-09-10; the audit's
  question (what a register holds decides what it may be charged) still applies
  to any future per-PE proposal.
* **§7.7 Task 4 first pass** — withdrawn 2026-09-09 the same day it was
  measured, under `ECC_OPT_METRIC=energy`, which serialises. Superseded by §7.8
  under `edp`, which then withdrew two further positives of its own.

Live plan as of 2026-09-10: `prompt_2.md`. Earlier detail:
`legacy/FINDINGS_detail_2026-09-07.md`.

---

### 7.1 The R4b register audit — what it holds decides what it may be charged

> **HISTORICAL, 2026-09-10: R4b is removed, so this audit no longer describes
> anything the code does.** It is kept because its lesson generalises to any
> future per-PE proposal: a register that serves the reads must remove them from
> the level below, and one that only caches the regenerated bits must be read on
> every delivery. The audit asked the right question and still got R4b wrong by
> one step — it priced the register correctly but never checked whether the
> mappings deliver any reuse for it to capture. They do not.

**Found 2026-09-08.** The model did two things that could not both be true:

1. `N_rec = fills` for R4b (2.78 M, not R4a's 230 M) **because** the register
   retained what the encoder rebuilt and served 82.85 uses per reconstruction;
2. the whole `weights_spad` term scaled by K/N — i.e. Timeloop's
   **1,814,073,344** scalar reads, one per MAC, left intact, as if no register
   were there.

If the register serves 82.85 of every 82.85 uses, those uses do not read the
scratchpad, and the K/N discount was being taken on traffic the same register
had removed. That is not a rounding argument: **97.6% of the scratchpad's
369.276 µJ is read energy** (604,691,115 word reads × 0.596239 pJ = 360.540 µJ
against 8.735 µJ of fills; the design's own ERT reproduces Timeloop's total to
0.000%). The old rule — "no operand-retention saving for anybody, because the
same register would help a baseline PE too" — was a symmetric handicap papering
over the gap, and it was load-bearing.

**The fix: the register never needed to be full width.** Per `embedded.py`,
`ParityOverwriteByTopWeightsEncode` overwrites the `n−k` lowest-significance
bits of each 63-bit chunk of the weight bit stream with parity. On chip the
reduced form keeps the `k` message bits; reconstruction regenerates those `n−k`
bits and splices them back. So the register only has to cache the
**complement** — `8 × (1 − K/N)` = 1.52 b/weight at BCH(63,51), 3.05 b at
BCH(63,39) — not a whole weight. Then it *cannot* serve a read on its own, the
scratchpad is still read once per MAC in reduced form, and the K/N discount on
all 1.81 G reads is real rather than double-counted.

`ECC_RECON_REUSE_REG_MODEL` makes all three accountings runnable
(`recon.ReuseRegister`; R1–R4a have no register and do not move):

| register model | R4b vs embedded | PE storage / weight | status |
|---|---:|---:|---|
| **`complement`** — holds the `n−k` regenerated bits, read in lockstep | **+1.297%** | **8.00 b (1.00×)** | **the default** |
| `free` — writes charged, reads not, SPad reads intact | +1.362% | 14.48 b (1.81×) | pre-2026-09-08, inconsistent |
| `full_width` — whole weights, serves the reads, priced at the ERT's cheapest per-PE register write (0.0328 pJ) | +6.131% | 14.48 b (1.81×) | the auditor's reading |
| `full_width`, priced at `weights_spad`'s own read energy (0.1987 pJ/weight) | **−0.223%** | 14.48 b (1.81×) | the same reading, honestly priced |

**Why `full_width` is not the answer even though it is internally consistent.**
It reports what the same register gives a PE with **no ECC at all**
(`what_the_same_register_gives_a_pe_with_no_ecc` in every result), and at
0.0328 pJ that is +6.17% with reconstruction's own marginal value just +0.16% —
which is the audit's conclusion, and it is an artifact of the price. A
tile-sized register is 256 × 8 b = **2048 bits**; `weights_spad` is 96 × 24 b =
**2304 bits**. They are the same array, so an access costs the same, and
0.0328 pJ is the energy of `ifmap_spad`, a **192-bit** array — 10.7× smaller.
Priced by its own size the register is worth **−0.11%**: nothing. You cannot
beat a 288-entry register file by putting a 256-entry register file in front of
it, because those 82.85 reads/fill are already served by the innermost level of
the hierarchy — there is no untapped reuse for a second level to harvest. A free
6% from a register nobody has built should have been checked, not concluded.

Under `complement` the symmetry objection disappears rather than being managed:
a baseline PE stores whole weights, has no missing bits, and there is nothing to
hand it that would let it make the same saving. The register is genuinely an ECC
component. `recon.py`'s `reuse_register_pj` docstring also used to call
0.0328 pJ conservative "because a one-entry latch costs less than a 24-entry
register file" — backwards, since R4b needs 16–256 entries. Corrected, and the
sensitivity is now measured rather than asserted: pricing the complement
register at `weights_spad`'s own access energy takes `complement` from +1.297%
to **+0.900%** — a real dependence, but it never inverts the ordering, whereas
the same repricing takes `full_width` from +6.13% to −0.22%.

**Not modelled, and named rather than assumed away:** the splice. Weights
straddle chunk boundaries (7 of 8 boundaries fall mid-weight, `embedded.py`), so
serving a MAC read means a barrel shift and concat across a packed stream. Small
next to a 0.596 pJ SRAM word access, not zero.


---

### 7.7 Task 4 — capacity dilation: FIRST PASS WITHDRAWN, being re-run under EDP

> **WITHDRAWN 2026-09-09, same day, before anything was quoted.** Everything
> below was measured under `ECC_OPT_METRIC=energy`, and that objective
> SERIALISES: it uses 4, 8 or 56 of Eyeriss v1's 168 PEs and 16 of weight
> stationary's 256 depending on the capacity, because fewer active PEs is less
> energy whatever it costs in latency. So the mappings being compared differ in
> PE count by up to 14x while the capacity being tested differs by 17x, and the
> capacity effect cannot be separated from the parallelism change. The
> comparison is confounded and its conclusion does not stand.
>
> `edp` is env.sh's default, is what `ReconSweep.png` and every Task 3 number
> use, and penalises serialisation — so PE utilisation stays put and only the
> capacity moves. The sweep was re-run under it and **§7.8 is the result** —
> which withdrew two further positives of its own, for two different reasons. What survives from below
> is the METHOD and the two arithmetic facts (the resident-weight shortfall and
> the integer-tile step), not the measurements.
>
> Also note `simple_weight_stationary` is not a sound test bed for this at all
> until section 8's open defect is fixed: its mappings use 16 of 256 PEs
> (6.25 %, meshX never used) on 11 of 12 shapes.

#### (superseded) DRAM weight traffic was flat in weight capacity over 13x, under the ENERGY objective

**The claim under test** (section 9 item 0, the user's framing of 2026-09-09).
Under the reconstruction arm the on-chip weight representation is K/N = 0.619 of
full width, so the same physical SRAM holds N/K = 1.6154x more weights. A larger
weight tile should mean the reconstruction arm reloads weight tiles from DRAM
fewer times, and a read it never issues removes the DRAM **array** energy as
well as the interface energy — efficiency **1.0** per µJ against the
`f_if x (1 - K/N)` = **0.152** a fixed mapping buys. Section 9 item 0 required
this be measured before it was modelled.

It was measured, and the right measurement is not the 1.6154x pair on its own.
`ECC_WEIGHT_CAPACITY_SCALE` puts the room into the architecture the mapper
actually sees, so the whole curve can be swept — and **DRAM weight traffic does
not depend on weight capacity at all** over a 13x range. resnet18, energy
objective, two layers, 128 mapping jobs:

| weight capacity | `eyeriss_like` L2.0.c1 | `eyeriss_v2_like` L2.0.c1 | `simple_weight_stationary` L2.0.c1 | all three, L4.0.c2 |
|---|---:|---:|---:|---:|
| x0.125 | 589,824 | 294,912 | 73,728 | 2,359,296 |
| x0.2019 | 589,824 | 294,912 | 73,728 | 2,359,296 |
| x0.25 | **294,912** | 294,912 | 73,728 | 2,359,296 |
| x0.4038 | **294,912** | 294,912 | 73,728 | 2,359,296 |
| x0.5 | **294,912** | 294,912 | 73,728 | 2,359,296 |
| x0.8077 | 589,824 | 294,912 | 73,728 | 2,359,296 |
| x0.99 | 589,824 | 294,912 | 73,728 | 2,359,296 |
| **x1.0 (declared)** | 589,824 | 294,912 | 73,728 | 2,359,296 |
| **x1.6154 (= N/K)** | 589,824 | 294,912 | 73,728 | 2,359,296 |
| x2.1 (= N/K at BCH(63,30)) | — | — | — | 2,359,296 |

`eyeriss_v2_like` and `simple_weight_stationary` are **exactly flat at every
capacity from x0.125 to x1.6154** — a 13x range, two layers, not one read
different. `eyeriss_like` is two-valued, and the low value sits in the MIDDLE of
the range (x0.25–x0.5) with x0.8077 and x1.6154 back at the high one. A quantity
that is not monotone in capacity is not being set by capacity: that is the
search landing in a different local optimum, worth exactly one factor-of-2 loop
step. **A 1.6154x dilation cannot buy what 13x does not.**

**The mechanism, read off the mappings.** A dilation enlarges the WEIGHT buffer.
These mappings did not run out of weight room:

| design | layer | fullest **weight** level | what is actually full |
|---|---|---|---|
| `eyeriss_like` | layer2.0.conv1 | `weights_spad` 21 % | `ifmap_spad` **100 %** |
| `eyeriss_like` | layer4.0.conv2 | `weights_spad` 86 % | `ifmap_spad` **100 %** |
| `eyeriss_v2_like` | layer2.0.conv1 | `weights_spad` 67 % | `ifmap_spad` **100 %** |
| `simple_weight_stationary` | both | `operand_glb` 56 % | `output_activation_reg` **100 %** |
| `eyeriss_like_wglb` | layer4.0.conv2 | `filter_glb` 75 % | `ifmap_spad` **100 %** |

The single strongest row is `simple_weight_stationary`'s layer4.0.conv2 at
x0.125, where `pe_spad` is at **100 %** — the weight buffer completely
saturated, which is the one condition under which a dilation is supposed to
pay. Enlarging it 48 -> 78 weights per PE cut **nothing**, because
`output_activation_reg` is at 100 % too and it is the psum register that forces
the tiling. A full weight buffer is not sufficient; it has to be the *only*
full one.

On `eyeriss_like`'s layer2.0.conv1 the refetch of 8 is `for Q in [0:2)` and
`for P in [0:4)` sitting above the only weight-holding level, and at x1.0 vs
x1.6154 the two loop nests are **byte-identical**. Those loops are there because
the *input* path ran out of room. **This is the utilisation audit's own
prediction, confirmed**: PROJECT_STATUS already recorded, the same day, that v1
"refetches heavily **while holding capacity in reserve**", and that the full
buffers hold activations and psums.

**Three controls, so the flatness is not an artifact of how it was asked.**

1. **The ERT bias removed.** Expressing the dilation as `depth x N/K` makes
   Accelergy price a deeper array (1.18–1.46x per access here), giving an
   energy-objective search a reason to avoid it. Under
   `ECC_OPT_METRIC=last_level_accesses`, which minimises DRAM accesses and
   ignores on-chip energy entirely, `eyeriss_like`'s weight spad comes back at
   **1 % fill** (4 of 448) while `ifmap_glb` is at 78 %. Still flat.
2. **A weight-only bracket widened.** `simple_weight_stationary`'s
   `operand_glb` holds Inputs beside Weights, so `ECC_WEIGHT_CAPACITY_SCOPE=shared`
   dilates it too — handing the mapper free INPUT capacity as well. At
   x1.6154-shared its layer2.0.conv1 traffic **doubled**, 73,728 → 147,456.
   More room, worse mapping: the same search variance, in the other direction.
3. **Designs with a dedicated weight buffer.** `eyeriss_like_wglb`'s
   `filter_glb` (8,192 weights, 75 % full) dilates to 13,232. Still flat.

**The first-order model, and why it cannot hold.** `R_recon - 1 = (R_emb - 1) x K/N`
assumes capacity is spendable continuously. Refetch is a product of **integer
loop factors** — on layer2.0.conv1 it is exactly `Q(2) x P(4) = 8`, whose next
value down is 4, needing **2x** the resident set. N/K = 1.6154 < 2, so a step is
not even reachable at BCH(63,39), and BCH(63,30) is the only rate in the family
that reaches N/K >= 2 (2.100).

**That was tested, and it is NOT the limiter.** At x2.1 on `eyeriss_like`'s
layer2.0.conv1 — capacity 448 -> 940 weights per PE, comfortably more than the
factor of 2 a tile step needs — the refetch is still **8.000 -> 8.000**, zero
reads cut, and the resident tile is still **96** weights. So the integer-tile
argument is necessary but not sufficient: even a dilation big enough to cross a
step cannot cross it while `ifmap_spad` is the level at 100 %. The wrong buffer
is being enlarged, and no code rate fixes that.

| code | K/N | N/K | can cross a factor-2 tile step? |
|---|---:|---:|:--|
| BCH(63,57) | 0.905 | 1.105 | no |
| BCH(63,51) | 0.810 | 1.235 | no |
| BCH(63,45) | 0.714 | 1.400 | no |
| BCH(63,39) | 0.619 | 1.615 | no |
| BCH(63,36) | 0.571 | 1.750 | no |
| **BCH(63,30)** | **0.476** | **2.100** | **yes — and measured: still zero** |

**What this does and does not say.** Task 3 stands unchanged and its savings are
real (§7.5/§7.6). What is withdrawn is section 9 item 0's projected 13–26 %: the
*capacity-dilation* mechanism does not operate on these five designs, because
their mappings are not weight-capacity-limited. **The honest bound is that no
capacity effect is resolvable above a search-variance floor of one factor-of-2
loop step**, which is the same size as the effect being sought — so §2's open
item ("prove the search converges") is now a precondition for Task 4, not a
parallel concern.

**29 comparisons, four designs, and every one reads zero** except a single
negative (`eyeriss_like` x0.5, marked `<-NEG`, the dilated arm refetching 2x
MORE) which is the same search variance in the other direction. The rows are
`results/tables/dilation_energy.csv`; reproduce with `bash run.sh dilation`.

**Every number here is a LOWER bound on three counts**, all recorded on the
result files: the mapper optimised against an array priced 1.18–1.46x dearer per
access; Timeloop derives the NoC hop length from that level's Accelergy area
wherever `noc.yaml` pins no `tile_width_um`, so the wires into it lengthened too
(0.17 pp, not corrected); and the search is stochastic, as the non-monotone
column above shows directly.

**The implementation is complete and correct**, which is what makes a flat
result trustworthy rather than a bug: with `RECON_OPTIMIZER=True ECC_PHASE=Post`
the reconstruction bars are re-derived from a second mapping, and where that
mapping comes back byte-identical the run reproduces Task 3 **to the digit**
(R1 6.31 %, R2 6.30 %, R3 6.96 %, R4a −7.57 %, R4b 12.86 % on the two-layer
scope, in both). `test_an_identical_loop_nest_means_an_identical_dram_read_count`
holds that property against every cached capacity pair. See CLAUDE.md for the
three artifacts of expressing capacity as depth and what is done about each.

