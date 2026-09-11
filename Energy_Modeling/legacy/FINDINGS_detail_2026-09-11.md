# FINDINGS

The only place empirical claims about the current model live. Every number here
says how it was produced. Whole-model unless a row says otherwise; two-layer
development numbers validate the implementation and rank nothing.

**The live plan is `prompt_2.md`** (rewritten 2026-09-10). Where this file and
that one differ on what to do NEXT, prompt_2.md wins; this file remains the
record of what was MEASURED.

Full working detail is archived in `legacy/FINDINGS_detail_2026-09-07.md` and
`legacy/FINDINGS_detail_2026-09-10.md` (§7.1's R4b audit and §7.7's withdrawn
first pass, both verbatim). Companions: `progress.txt`, `PROJECT_STATUS.md`.
`legacy/FINDINGS.md` is pre-rewrite; do not quote it.

## Status

| Task | State |
|---|---|
| 1 conventional ECC, external parity | ✅ whole model, 6 archs × 2 models, NoC costed |
| 2 embedded ECC (DRAM effect only) | ✅ same 12 pairs, evaluation-only on Task 1 mappings |
| 3 reconstruction placement | ✅ `eyeriss_v2_like` × resnet18 + mobilenet_v2, 5 boundaries |
| 4 reconstruction-aware mapping, by CAPACITY DILATION | ❌ **withdrawn.** 196 EDP mappings, 5 designs, no clean capacity win, and the whole sweep ran at an UNCONVERGED budget (§7.8) — not quotable. `weights held` flat to 32×; four reported positives were artifacts. Cause is arithmetic, not search: tiles are integers and the step is 2×, but N/K = 1.6154 (§7.8) |
| 4′ same question, by `datawidth` | designed, unmeasured. VERIFIED that narrowing `datawidth` at fixed geometry gives more effective capacity at byte-identical per-access energy (§9.1). Plan and widths: `prompt_2.md` |
| 5 placement study on other designs | not started; `eyeriss_v2_like_wglb` refused until its GLB boundary exists |
| convergence proof for the mapper search | ❌ **open, and it gates every ranking** (§2). `ECC_VICTORY=4000` chosen 2026-09-10; gate not yet run at it |
| Eyeriss v1 = `eyeriss_like_wglb` | DECIDED 2026-09-10, unmapped. Ends the v1 bracketing pair (§8.5); needs `WEIGHT_PATHS`/`PLACEMENTS` first |

## 1. The settings every number below was produced at

`random_pruned`, uncapped search size, `victory 2000` with `levels` scaling
(so 4000 on the 9-level designs), `timeout 2000`, `perms/if 16`,
**`threads 18`** (in the fingerprint — change it and everything is cold), energy
objective, `paper` fidelity, **NoC costed**, BCH(63,51), 8-bit weights, 45 nm,
1 ns cycle. Six architectures: `eyeriss_v2_like`, `eyeriss_like`, `simba_like`,
`simple_{input,weight,output}_stationary`. Two models: resnet18 (21 layers,
12 shapes), mobilenet_v2 (53 layers, 31 shapes, 10 depthwise).

Mapping cost at these settings, measured (SLURM 41296638, 18 cores):
`eyeriss_v2_like` × resnet18 **2 h 26 m**, × mobilenet_v2 3 h 00 m,
`eyeriss_like` × resnet18 1 h 03 m. A 12-task array was 48 min wall at `%9`.

## 2. The mapper search decides the architecture ranking — the central caveat

Same layers, same YAMLs, only the search differs (µJ, dev pair
`layer3.0.downsample.0 + layer4.1.conv2`; "×" is DRAM weight refetch):

| setting | eyeriss_v2 | eyeriss_v1 | WS | OS | IS | simba_like |
|---|---:|---:|---:|---:|---:|---:|
| `hybrid`, victory 100 | 2036.8 (7.1×) | 1979.3 (6.9×) | 732.0 | 904.3 | 581.6 | 3166.2 |
| converged (§1) | **402.0** (1.0×) | **390.1** (1.0×) | 469.5 | 711.9 | 508.5 | 446.1 |

The ordering inverts and nothing about the architectures changed. A 36,864-weight
tensor fetched 196× is `hybrid` freezing a bad index factorisation, not a
property of Eyeriss; the simple WS/OS/IS designs have a far smaller legal
mapspace and barely move, which is exactly why they used to "win". A *more*
thorough hybrid search (victory 500, 5–6 h) gave *worse* Eyeriss numbers than a
10-minute bounded one — the signature of a search that is not converging.

**Consequences.** `random_pruned` at §1's settings is adopted as the default. No
figure produced under `hybrid` or the bounded development search is quotable.
**Convergence is still assumed, not proven**: the check — re-run at
`search_size 20000` and `40000` and require the totals to agree within a few
percent — has not been done. Until it is, every ranking below is provisional.

## 3. Two workload defects, both fixed

**Depthwise/grouped convolutions were modelled as 1-channel convs**: the loader
dropped `groups` and the problem template had no group dimension, so
mobilenet_v2's `features.15.conv.1.0` reached Timeloop as `C=1, M=960` — **81
input scalars read instead of 77,760**. MAC and weight counts happened to be
right, which is why the totals did not look absurd; input traffic, reuse, the
mappings and every depthwise layer's energy were wrong. Fixed with `Layer.G`
(C and M per group, `_G<g>` shape suffix) and `GROUPED_PROBLEM_TEMPLATE`; MACs
now match published values on all eight models. **ConvNeXt's per-pixel
`nn.Linear` layers** were separately recorded as `P=Q=1`, undercounting MACs by
H×W (0.32 G against 4.5 G published).

mobilenet_v2, efficientnet_b0, convnext_tiny and xception hold mapper entries for
the old shapes; they are never hit again (new names) but must be re-mapped before
any of the four is plotted. `energy.load_raw()` now rejects a raw record whose
per-layer shape list is not the current one — without that the stale aggregate
would have been served forever.

## 4. Interconnect (NoC) model — added 2026-09-07

Timeloop tracks fanout, multicast, hops and spatial reductions between adjacent
levels, but its wire model is a stub (`pat.cpp:81`, `WireEnergy(...) {return 0;}`)
and no architecture supplied coefficients, so every NoC cost 0.00 pJ — in the
evaluator *and in the mapper's objective*. Costing it therefore **changes which
mapping wins**; it cannot be applied to cached mappings (the slug gains `noc`).

The adopted constants, their citations, the calibration and the injection trap
are all in `archs/_shared/noc.yaml` — read it there, not here. In one line: a
single shared wire constant (**0.12 pJ/bit/mm**, Keckler/Dally IEEE Micro 2011),
a per-design switching term (v1's inter-PE transfer calibrated to its own paper;
0.25 pJ **per flit** at v2's router-cluster level and simba_like, **an assumption
— no paper publishes a pJ/router**), and per-design hop distances that Timeloop
derives from each floorplan's own area. What is recorded here is what it measured
and what was rejected.

Three things learned building it, each of which had inflated the NoC:

* **A flat per-ingress term is necessary**: wire is charged per hop, so a fully
  serial mapping moves its data free and an energy objective finds that corner.
* **Container attributes are inherited by every child**, so the hop-independent
  terms reached the spads inside each PE — `psum_spad <==> mac` (register → ALU)
  billed 1203 fJ/compute, 94% of a 45.5% "NoC" share.
* **Not every spatial level is a NoC**: v2's `SIMD` container is the two-MAC
  datapath inside a PE, and it was billed 750 fJ/compute. Naming which containers
  *are* the NoC is not enough on its own — the rest need **explicit zeros**,
  because a nested container inherits its parent's value.

Measured under the adopted defaults (dev pair, SLURM 41295428/41295683):

| design | total | NoC | share |
|---|---:|---:|---:|
| eyeriss_like | 397.83 µJ | 14.99 | 3.8% |
| eyeriss_v2_like | 415.51 | 22.10 | 5.3% |
| simple_weight_stationary | 506.6 | 39.2 | 7.7% |
| simple_output_stationary | 748.5 | 44.6 | 6.0% |
| simple_input_stationary | 536.8 | 25.7 | 4.8% |

The wire constant is the lever on v2 (74% of its NoC): the ablation ran
0.4 → 0.2 → 0.12 pJ/bit/mm at 15.4% → 7.6% → 5.3% NoC share. v1 barely moves,
because its transfer is anchored to its paper by construction. **Reading 5.3%
against the paper's 6–10% band**: two biases oppose (the paper's share is for
*sparse* runs at 65 nm), so a dense 45 nm share a few points either side is
consistent, where the 15% and 30% earlier versions produced were not.
`noc_share_within_published_band_eyeriss_v2` records FAIL and stays as written —
but see §8.2, which may be the whole story.

**A 2-D 8×2 cluster geometry was tested and NOT adopted**: it cut v2's NoC 18%
but raised the total 7% at identical DRAM reads, because the mesh changes the
mapspace and under the same budget the mapper found worse tilings than it saved
on wire — a NoC correction confounded with search quality. The 1-D row's unicast
over-count (7.5 hops instead of ~4) is small at 0.12 pJ.

Decision 2026-09-07: the recon arm scales the weight share of NoC energy by K/N
like the other on-chip categories, recorded as an approximation on every result.

### 4.1 NoC model revision — 2026-09-08, implemented, **not yet measured**

Five changes to the interconnect model, made together because the first three
are corrections and the last two are the missing terms. Nothing below has been
mapped yet: every v2 result above is still the 2026-09-07 model. The figures in
this section are **fixed-mapping estimates** on the converged 43-shape v2 cache
(`fp-71d497c0376a`), i.e. what each term adds before the mapper reacts to it.

| change | where | fixed-mapping effect on v2 NoC (125.9 µJ, 6.7% of on-chip) |
|---|---|---:|
| E. inner NoC level always gets its own `router_energy`/`energy-per-ingress`, zero included (closes open defect 8.1) | `archs._inject_noc` | −29.7 µJ |
| D. spatial reductions costed: one psum-width adder (Aladdin table, 0.0066 pJ/bit) + one hop of psum-wide wire | `noc_post.py` (**evaluator-only**) | +26.5 µJ |
| D'. Outputs wire re-billed from 8 to the design's accumulator width (20b on v2) | `noc_post.py` (**evaluator-only**) | not in the estimate; ~+10% of NoC |
| C. mesh hop length declared: 430 µm (PE cluster + GLB cluster + 2.6% routers) instead of Timeloop's area-derived 375 µm; PE row 108 µm declared so it does not inherit | `noc.yaml` `tile_width_um` | +6.1 µJ |
| A. per-PE latch on v2's PE row, 0.5 pJ/operand, bracketed 0 … 0.9152 (v1's calibrated switching) | `noc.yaml` `pe_latch_pj`, `ECC_NOC_PE_LATCH_PJ` | +38.5 µJ |
| **E + D + C + A** | | **+41.5 µJ → 4.9% of total, 8.8% of on-chip** (superseded by the EDP measurement in 4.1.2) |

Why each, in one line. E: a declared zero was never written, so v2's PE row
inherited the cluster router and paid it a second time. D: Timeloop counts the
partial sums added into a neighbouring PE but costs them with
`pat::AdderEnergy()`, a stub returning 0, and `ParseSpecs` reads no adder
coefficient — so it cannot be given to the mapper and is charged after the
fact from the printed counts (`noc_post.augment`, stamped into every `Raw`;
a record aggregated under different coefficients is rejected by
`load_raw`). C: `network-legacy.cpp` honours a declared `tile_width` over the
area-derived one; on the real die a mesh hop spans a (GLB cluster, router
cluster, PE cluster) triple, and Accelergy sizes only the PE cluster. A: v2
paid nothing to take an operand into a PE while v1 paid 0.9152 pJ for the same
act; the term also restores to v2 the anti-serial-mapping guard that
`charge_ingress_when_hops_zero` describes.

**Why the mapper found 17% utilization, and why A matters for it.** Every
converged v2 layer stopped on the victory condition (4000 non-improving
mappings per thread), so the search was not cut short. Timeloop's energy has
no cycles in it, and a loop run across 12 PEs or 12 times in one PE costs the
same accesses, so the objective is indifferent to fanout and the costed hops
tip it towards fewer PEs. A per-ingress term is the one NoC cost that rewards
multicast fanout (one ingress serves several PEs); v2 had none. The user's
decision on switching the objective to EDP is pending; A should be re-read
after that run, because higher utilization means more hops and a higher NoC
share than the fixed-mapping estimate.

**Also in this revision, outside the NoC:** Accelergy's Aladdin register table
(`aladdin_register.csv`) costs a register **write at 0 pJ** (read 0.009 pJ/bit),
so every register-file write and every partial-sum update in the study was
free — v2's `psum_spad` ERT read `write 0.0 / update 0.0` against `read 0.13`,
with more updates than reads on the layers examined. `regfile_decoded.yaml` now
charges write and update at the read row; `register_rw.yaml` does the same for
the bare registers of the three `simple_*` designs, whose `arch_paper.yaml`
now names it. `arch_fingerprint()` now hashes `archs/_shared/components/`, which
it did not before — a component edit changed every ERT without moving any
cache. **Consequence: every architecture's mapper cache is cold**, not only
v2's.

The v2 band check (`audit.py`) now reads the paper's 6–10% against **on-chip**
energy, reporting both denominators; Fig. 18 is a gate-level breakdown of the
chip. `baseline.py` is frozen and keeps the old denominator.

Verification so far: `eccenergy.tests.test_noc` (19 tests, including the
pitch re-derivation from the cached ART, the fixture-based evaluator-only
arithmetic and the fingerprint move), `test_recon`, `test_embedded`,
`test_results_store`; `run.sh validate` on all eight designs. A bounded
single-layer smoke mapping of the revised v2 and output-stationary designs is
in §4.1.1.

#### 4.1.1 Smoke mapping of the revised model (SLURM 41452254/41452255, 2026-09-08)

`layer3.0.downsample.0`, `search_size 400`, `victory 100`, 4 threads — a
**mechanics check, not a result** (the mapping it finds is poor: 11.5 mean hops
on the mesh). What it establishes, from the stats and ERT it produced:

* Timeloop honours the declared pitches: mesh per-hop energy **412.80 fJ**
  (= 8 b × 0.430 mm × 0.12), PE row **103.68 fJ** (= 8 b × 0.108 mm × 0.12);
  before, 359.79 and 103.86 fJ from area.
* The PE-row network prints `Router energy 0.00 pJ`, `Ingress energy 0.50 pJ`;
  the mesh prints `Router 0.08`, `Ingress 0.00`. Change E and A land where
  intended; nothing reaches the SIMD level or the scratchpads.
* `psum_spad` ERT: write 0.13263 / read 0.13263 / update 0.13263 pJ (was
  0 / 0.13263 / 0). `output_activation_reg` on output-stationary: write = read
  = 0.182 pJ (was write 0). `register_rw` compiles in Accelergy with no warning.
* The evaluator adds the two evaluator-only rows per network for Outputs and
  stamps the record (`noc_post: {adder_pj_per_bit: 0.0065625, network_word_bits:
  8, outputs_word_bits: 20, reduction_hops: 1.0, ...}`); `baseline --eval`
  hand-check PASS on the result.


#### 4.1.2 First EDP mappings (2026-09-09): the NoC was too high, and why

11 resnet18 layers (the first mapped by SLURM 41454163/65/67; conv1 through
layer2.1.conv2), same layers in every column, µJ. All three latch runs found
IDENTICAL mappings, so the latch is purely additive.

| term | old model, energy obj. | new, latch 0 | new, latch 0.5 | new, latch 0.9152 |
|---|---:|---:|---:|---:|
| mesh wire | 62.9 | 192.0 | 192.0 | 192.0 |
| mesh router | 15.5 | 40.2 | 40.2 | 40.2 |
| PE-row wire | 29.4 | 65.1 | 65.1 | 65.1 |
| PE-row router (defect 8.1) | 26.9 | 0 | 0 | 0 |
| PE latch | 0 | 0 | 56.1 | 103.2 |
| reductions (evaluator-only) | — | 57.2 | 57.2 | 57.2 |
| psum width (evaluator-only) | — | 102.2 | 102.2 | 102.2 |
| all interconnect, share of on-chip | 6.9% | 16.3% | 18.0% | 19.3% |
| mesh only (wire + router), share of on-chip | 3.9% | 8.6% | 8.6% | 8.6% |

Diagnosis. (i) **Mesh wire tripled** because EDP fills all 16 clusters and the
model laid them out as one 16-long row fed from one end: mean hop counts 7.5 to
13. That is the flattening the arch file had documented as a small over-count
at low utilization; under EDP it is the largest NoC term. (ii) The latch cannot
bring the share into the band and has no measurement behind it. (iii) The
evaluator-only terms are large under EDP (132.7 M spatial reductions on these
layers) and, like the PE-row wiring, are energy the paper's Fig. 18 is read as
booking under the PE array rather than under the hierarchical mesh.

Decisions, 2026-09-09:

* **`eyeriss_v2_like` PE clusters are declared 8x2** (Table II) in
  `arch_paper.yaml`, spatial split left to the search. The 1-D row is retired.
* **`pe_latch_pj` set to 0**; knob and bracket kept.
* **The band is compared with the mesh only**: noc.yaml `paper_band_levels:
  [PE_cluster]`, `audit.py` reports `mesh_share_of_onchip` beside the
  all-interconnect share and checks the band against the mesh. This is an
  ASSUMPTION about Fig. 18's category boundaries and must be verified against
  the figure's legend.
* Latch-bracket runs 41454165-68 cancelled (mappings identical to the default
  run; their answer is the table above). The default 1-D run 41454163/64 is
  left to finish for a full-model 1-D vs 8x2 comparison; its figure is kept as
  `ReconSweep_1Drow_latch0.5.*`.
* Mapping is now submitted **one SLURM job per layer shape**
  (`hpc/map_by_shape.sh`, 12 jobs of 18 cores for resnet18) with the eval and
  figure as a dependent job -- the mapper cache is keyed by shape, not by the
  layer list, so the per-shape entries are exactly what one job would write.

Expected from the 8x2 shape: mesh wire and router roughly halve. Unmodelled and
recorded for Task 4: each PE cluster's GLB cluster is beside it on the die, so a
unicast is one local hop, whereas the model still feeds all 16 clusters from one
GLB at the array's edge (a distributed-GLB model).


#### 4.1.3 Whole-model result on the 8x2 mesh (2026-09-09) — the current `ReconSweep.png`

*(Later on 2026-09-09 the figure was redrawn with the decoder on the DRAM die, §7.2; the numbers below are the `ECC_RECON_DECODE_SITE=controller` row of that section and are reproduced to the digit by it.)*

resnet18, 21 layers, EDP objective, `paper` fidelity, 45 nm, µJ. The three
columns differ ONLY in the interconnect model and geometry; compute is identical
(2120 µJ) because the MAC count is.

| | old model (09-07), energy obj., 1-D row | 1-D row, latch 0.5 | **8x2, latch 0 (current)** |
|---|---:|---:|---:|
| total | 4798 | 6467 | **5940** |
| DRAM / GLB / Local | 1433 / 393 / 632 | 1748 / 768 / 932 | 1729 / 545 / 921 |
| NoC, all terms | 220 | 898 | **626** |
| of which mesh (wire + router) | 112 | 394 | **249** |
| PE-row wire (+ latch) | 107 | 212 | 129 |
| spatial reductions (evaluator-only) | — | 133 | 145 |
| psum word width (evaluator-only) | — | 159 | 102 |
| **all interconnect, share of on-chip** | 6.5% | 19.0% | **14.9%** |
| **mesh only, share of on-chip** | 3.3% | 8.3% | **5.9%** |
| mesh mean hops, unicast | 7.5 | 7.5 | 4.0 |

Read against the paper's 6–10% band on the mesh-only reading (§4.1.2), the 8x2
model sits at the band's lower edge (`noc_share_within_published_band_eyeriss_v2`
fails by 0.1 point). The all-interconnect bar the figure draws is 14.9% of
on-chip energy; the gap between the two numbers is the intra-cluster wiring and
the partial-sum traffic, which the paper is read as booking under the PE array.
The register-write correction and EDP together lift Local and GLB against the
old model; that is expected and is not an interconnect effect.

**Timeloop's hop model, for the record** (nest-analysis.cpp:1714): data is
injected at the h=0 edge, at the centre of the V axis; a transfer's hop count is
the length of its multicast TREE — the farthest column reached plus the vertical
drops at every column touched. An 8-way psum multicast on 8x2 therefore reports
11 hops (7 across + 8 × 0.5 down), which is the wire actually driven, not a
per-destination distance. The 1-D row's unicast mean of 7.5 became 4.0 on 8x2.

**Two incidents in this run, both now guarded:**

* `hpc/map.sbatch` carries `#SBATCH --array=0-11%9`; per-shape submissions
  without an `--array` override became 12-task arrays whose tasks 1–11 exited 2,
  so the `afterok` eval never fired (`DependencyNeverSatisfied`). Fixed with
  `--array=0-0` in `hpc/map_by_shape.sh`.
* The 1-D run's queued eval (41454164) started AFTER `arch_paper.yaml` had been
  switched to 8x2, computed the 8x2 fingerprint, found the one 8x2 shape mapped
  at that minute and wrote a one-layer record under `layers-full`; the 8x2 eval
  then took it as a raw-cache hit and drew a one-layer figure labelled as the
  whole model. Both records and the copied figure were deleted, and
  `energy.load_raw` now rejects any record with an `unmapped` layer (test in
  `test_noc`). Rule: do not edit an arch YAML while an eval job is queued
  against it — the fingerprint is computed when the job runs, not when it is
  submitted. The 1-D run's own raw record (`fp-3eb860ea2b2a`, written by the
  map job) is intact and is the 1-D column above; its figure was not redrawn.

Also changed by the new mappings: **R4a and R4b are `unsupported`** on resnet18
under EDP (layer4.* mappings keep fewer than G_rec = 9 weights resident per PE),
so the figure draws five bars, not seven. Embedded saving vs conventional:
−14.6% (was −15.2% under the old model).


## 5. Whole model, Tasks 1 and 2 — conventional vs embedded ECC

BCH(63,51), µJ. "conventional" = Timeloop + external parity; "embedded" removes
the external-parity traffic and nothing else, so the saving is exactly
`parity / (Timeloop + parity)`.

| architecture | resnet18 conv. | parity | embedded | save | mnv2 conv. | parity | embedded | save |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eyeriss_v2_like | 5,075.371 | 276.997 | 4,798.375 | 5.46% | 1,862.971 | 70.591 | 1,792.380 | 3.79% |
| eyeriss_like | 5,357.602 | 298.623 | 5,058.978 | 5.57% | 1,984.785 | 81.692 | 1,903.093 | 4.12% |
| simba_like | 5,699.198 | 297.968 | 5,401.230 | 5.23% | 1,996.279 | 71.388 | 1,924.891 | 3.58% |
| simple_input_stationary | 6,647.611 | 271.572 | 6,376.039 | 4.09% | 2,102.472 | 71.123 | 2,031.349 | 3.38% |
| simple_weight_stationary | 7,782.205 | 249.544 | 7,532.661 | 3.21% | 2,348.220 | 79.359 | 2,268.861 | 3.38% |
| simple_output_stationary | 9,363.620 | 260.121 | 9,103.499 | 2.78% | 2,639.039 | 69.395 | 2,569.644 | 2.63% |

**The ranking survives changing the network** — `v2 ≈ v1 < simba_like ≈ IS < WS <
OS` on both, which is the claim the two-panel layout exists to test. The two
Eyeriss designs trade places between models but sit within 1.2%, so that swap is
not meaningful. DRAM weight refetch is 1.03–1.42× everywhere here, and **the
saving tracks DRAM weight-traffic share**, as it must.

**The development layer pair is NOT a proxy for the model.** On the pair the
order is v1 < v2 < simba < WS < IS < OS; whole-model, **WS and IS swap and the
gap is large** (WS 20% *worse* than IS, where the pair had it 7.7% better). Its
Task 2 savings (11.0/10.9/9.7/8.8/8.5/6.3%) are about twice the whole-model
figure because it is weight-dominated. The pair validates an implementation and
nothing else.

## 6. The embedded codeword layout, read from the embedding code

`ECC-CODE-Engine/4-EmbeddingECC/ecc_embed.py` (approach `replace`): one MSB-first
8-bit-per-weight stream, cut into **fixed 63-bit chunks for every K**, systematic
BCH with the k highest-significance positions as message and the n−k lowest
overwritten by parity. Consequences (`embedded.py`, `test_embedded.py`):
**7.875 weights per codeword**; gcd(63,8)=1, so the alignment repeats every 8
codewords = 63 weights and **7 of 8 codeword boundaries split a weight**; storage
is exactly 8 b/weight plus ≤62 b of tail padding per tensor, with no external
parity and no shortened code. Parity bits per weight by significance:
BCH(63,51) → bit0 1.000, bit1 0.524; BCH(63,30) → bits 0–3 every weight, bit4
0.19 — **that is what "do not treat all weight bits as independent payload" means
in numbers.**

The baseline's layout (`parity.py`) is the opposite rule — whole weights in a
separately stored codeword, 6 per BCH(63,51) codeword + 3 padding bits ⇒
**10.5 stored bits/weight, 31.25% traffic overhead**, *more* than the flat n/k
(9.88 b). Both are right for what they describe. resnet18: 11,678,912 weights,
15,328,593 B conventional vs 11,679,027 B embedded.

**What the embedded arm's DRAM energy is not.** The complete codeword is read for
correction, so every bit of every DRAM weight read is billed and **no K/N
reduction is applied at DRAM**. The physical word count is an estimate (bits/64,
rounded once) because Timeloop bills per 8-bit scalar; tail padding is reported,
not charged. Codec energy is charged to neither arm and would not cancel if it
were (1.95 M vs 1.48 M decodes). The mapper is *not* re-run — it maps 8-bit
weights under both arms, every mapping is a cache hit, and the fixed mapping is
what the comparison requires.

## 7. Task 3 — reconstruction placement on `eyeriss_v2_like`

> **SUPERSEDED 2026-09-10 — R4b IS REMOVED FROM THE STUDY.** Everything below
> this line is the record as it stood while R4b existed; the numbers are also
> pre-dating the DRAM cost change (f_if removed, 40 pJ/bit). R4b was scrapped
> because **consecutive weight reuse is 1 on 20 of the 21 resnet18 layers**, so
> a latch catches nothing, and a register that does pay must hold the whole
> cycled set — up to 384 weights, the entire `pe_spad` on
> `simple_weight_stationary`. **R3 (reconstruct at the PE weight-storage INPUT)
> is now the result: +11.07 % vs embedded on `simple_weight_stationary`,
> +17.24 % on `eyeriss_like`.** See PROJECT_STATUS.md (2026-09-10, later) and
> `01_project_context_and_architectures.txt` §2.3. Do not quote §7's headline or
> §7.1's audit as current.

**Headline: R4b — reconstruct at the weight-scratchpad output and retain what was
rebuilt — is the best boundary, beating embedded-only ECC by 1.30% (6.68% against
conventional ECC). It captures 61.6 of the 77.5 µJ available, i.e. 79% of the
ceiling, and costs the PE nothing in weight storage.** The ordering the source
discussion predicts holds: R4b (5/5) > R2/R3 (4/5, break-even) > R4a (3/5, 18%
*worse* than doing nothing).

**Revised 2026-09-08 from +1.36% to +1.30%**, and the reason matters more than
the 0.06 points: R4b's register was being charged a write and nothing else,
while R4b's whole reconstruction saving came from that register serving the
scratchpad reads. §7.1 has the audit and the fix. The number barely moved; what
changed is that it is now defensible, and R4b's storage overhead went from
1.81× the baseline PE to **1.00×**.

**The ceiling, printed first on every result and console run**, because a 0.06%
number is otherwise indistinguishable from a missing term (full resnet18):

    on-chip weight energy that CAN be reduced :  407.128 uJ =  8.48% of the embedded total
    x (1 - K/N) = x 0.1905                    :   77.548 uJ =  1.62%  <-- the most ANY placement can save
    DRAM weight energy, NOT reducible         :  886.387 uJ = 18.47%  (complete codeword read for correction)

The path is four stages — `dram` 886.387 µJ (not reducible), `inter_cluster_mesh`
24.158, `cluster_local` 13.694, `weight_spad` 369.276 — with 91% of the ceiling in
the scratchpad. **`eyeriss_v2_like` genuinely has no weight GLB** (its GLB banks
are iacts and psums, JETCAS Table II), confirmed level by level from the raw
record: 0.00 µJ there is a fact about the design, not a gap.

The five boundaries (`stream` packing, `weight` encoder charging, `tile`-sized
reuse register, 4.1296 pJ/codeword from `data/dc/BCH_N63_results.json`):

| bar | total | vs conventional | vs embedded | recon | overhead | N_rec |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (external parity) | 5,075.371 µJ | — | −5.773% | — | — | — |
| Embedded (no external parity) | 4,798.375 | +5.458% | — | — | — | — |
| R1 @ NoC source | 4,805.637 | +5.315% | −0.151% | 7.263 µJ | — | 1,758,704 |
| R2 @ cluster edge | 4,801.036 | +5.405% | −0.055% | 7.263 | — | 1,758,704 |
| R3 @ SPad input | 4,802.647 | +5.373% | −0.089% | 11.482 | — | 2,780,388 |
| R4a @ SPad output | 5,672.121 | −11.758% | −18.209% | 951.295 | — | 230,358,520 |
| **R4b @ SPad + reuse reg** | **4,736.086** | **+6.680%** | **+1.297%** | 11.482 | 3.778 | 2,780,388 |

R4b reconciles by hand: `407.128 × 0.1905 = 77.55` saved, `11.482` of encoder +
`0.718` of register writes + `3.060` of register reads `= 15.26` paid. R1 being
worse than embedded is correct and is why it is the control — it strips the bits
and immediately puts them back.

On **mobilenet_v2** (ceiling 66.8 µJ = 3.72%, 0.71% after K/N; depthwise layers
have little weight reuse to spend scratchpad energy on) R1/R2/R3 land at
−0.103/−0.063/−0.062% and **R4a/R4b are `unsupported`**: 7 depthwise layers keep
only 3 weights resident per PE against `G_rec = 9`, so the retained bits were
never delivered to that PE. Named, never estimated.

### The R4b bug, which is the methodological lesson

The first version reported R4b as the **worst** boundary. It asked how many uses
one weight gets *in a row* — what a one-entry latch serves — which the loop nests
say is **1** (the innermost temporal loop below `weights_spad` is `M` on all 12
resnet18 shapes). R4b then came out as R4a plus an unused register, inverting the
source's 5/5 rating and contradicting the plain physical argument that a retained
reconstruction needs fewer reconstructions than an unretained one.

That was an artifact of the question. §5.4's EV2-C is not a latch: it is *reload
when the required weight changes*, so the right question is **capacity**,
answered from the mapping — the loops below the scratchpad walk `inner_tile`
distinct weights (16–256, equal to the buffer's utilized capacity on every shape)
and repeat `reads/fills` times (82.85× overall). A register covering the working
set reconstructs once per **fill**: R3's encoder count with R4a's storage saving,
which is exactly why it is rated above both. One *smaller* catches nothing — a
cyclic walk is the LRU worst case, so there is no partial hit rate — and
`ECC_RECON_REUSE_REG_ENTRIES=1` or `=9` collapses R4b onto R4a at −19.45%.

**R4b's overhead is neither capacity nor much energy — but only because the
register holds the right thing.** It retains the `n−k` bits per chunk that the
encoder regenerates, which is `8 × (1 − K/N)` = **1.52 b per weight** at
BCH(63,51): 6.48 b of reduced scratchpad + 1.52 b of register = **8.00 b per
resident weight, 1.00× the baseline PE's weight storage, at every K**. It buys
access bit-volume and costs no capacity. Both its writes (0.718 µJ, one per
weight reconstructed) and its lockstep reads (3.060 µJ, one per reduced
scratchpad word read) are charged; area and leakage are still not modelled.

This paragraph used to describe a register holding *256 full-width weights,
1.81× the baseline PE*, and that description is what made the whole result
attackable — see §7.1.

### 7.1 The R4b register audit — HISTORICAL, R4b removed 2026-09-10

> **This audit no longer describes anything the code does. Full text archived in
> `legacy/FINDINGS_detail_2026-09-10.md`.**

The lesson generalises to any future per-PE proposal, so it is worth the ten
lines: a register that SERVES the reads must remove them from the level below,
and one that only caches regenerated bits must be read on every delivery. The
audit priced the register correctly and still got R4b wrong by one step — it
never checked whether the mappings deliver any reuse for the register to
capture. They do not: consecutive weight reuse is 1 on 20 of 21 resnet18 layers.

**Do not reintroduce a retained-reconstruction boundary without first
re-measuring `consecutive_run` on the target mapping.**

One number from it did flip. Since the DRAM term stopped being split by `f_if`
(2026-09-10, prompt_1.md), the claim that "most of what the full-width register
buys is what the same register would buy a PE with no ECC" is **no longer true**:
the register-alone saving is on-chip and unchanged at 466 µJ, while ECC's
marginal DRAM component went 122 µJ → 1,526 µJ — a ratio 3.4× the other way. The
test asserts the *separation* and reports the ratio rather than bounding it.

### Sensitivity — `vs embedded`, %, resnet18

Every cell measured at BCH(63,51), one run per row, R4b's own column (not the
best-placement summary line, which changes bar when R4b stops winning):

| | R1 | R2 | R3 | R4a | R4b |
|---|---|---|---|---|---|
| **default** (`stream`, `weight`, reg=`tile`+`complement`) | −0.15 | −0.06 | −0.09 | −18.21 | **+1.30** |
| `ECC_RECON_REUSE_REG_MODEL=free` | −0.15 | −0.06 | −0.09 | −18.21 | +1.36 |
| `ECC_RECON_REUSE_REG_MODEL=full_width` | −0.15 | −0.06 | −0.09 | −18.21 | +6.13 |
| `ECC_RECON_REUSE_REG_PJ=0.1987` (`weights_spad`'s own) | −0.15 | −0.06 | −0.09 | −18.21 | +0.90 |
| ↳ …with `full_width` as well | −0.15 | −0.06 | −0.09 | −18.21 | **−0.22** |
| `ECC_RECON_PACKING=aligned` | −0.15 | −0.10 | −0.17 | −19.76 | **−0.27** |
| `ECC_RECON_ENCODER_GRANULARITY=codeword` | −1.19 | −1.10 | −1.73 | −154.51 | −0.35 |
| `ECC_RECON_REUSE_REG_ENTRIES=1` or `9` | −0.15 | −0.06 | −0.09 | −18.21 | −19.51 |
| `ECC_RECON_INCLUDE_IDLE=0` | −0.07 | +0.03 | +0.04 | −7.50 | **+1.43** |

* **R4b's whole benefit is contingent on physical packing.** Under `aligned`,
  `ceil(8×51/63) = 7` bits/weight and `floor(24/7) = 3` values per 24-bit word —
  the same 3 as at 8 bits — so the scratchpad access count does not move and
  +1.30% becomes −0.27%. `stream` is defensible only because the embedded
  layout genuinely is a packed bit stream. **A real implementation must pack
  across weight boundaries or R4b buys nothing.**
* **The 5/5 rating is a claim about a tile-sized register, not a latch**, and the
  encoder's idle term (2.2301 of 4.1296 pJ/codeword) moves magnitudes but no
  ordering.
* The three register models span **−0.22% to +6.13%** on the same mappings and
  the same code. Which one is quoted is a physical claim about what the register
  holds, not a tuning choice — quote `complement` and say so.

### Evidence that the reduction is applied, and applied only where claimed

Measured on the real cached Timeloop output: every reduced stage is
`before × K/N` to within 4e-16 relative, every untouched stage bit-identical.
Seven property tests pin this, run on both a synthetic `stats.txt` and the real
mapper cache (`test_recon.py`, **31 tests**, none skipped in the container):
categories monotonic down the weight path, each saving matching its closed form
rather than an inequality, a stronger code saving strictly more at every reduced
stage. Five of the 31 pin the 2026-09-08 register model (§7.1): that the
complement is exactly `weight_bits × (1 − K/N)` and PE storage exactly 1.00× at
every K, that `free` still reproduces the pre-audit arithmetic to the digit,
that `full_width` refuses without an ERT and — on the real cache — actually
moves the scratchpad's read count from `reads` to `fills`, and that the
complement read term grows as K falls instead of being hard-wired.

**Mutation-tested, because a property test that cannot fail proves nothing.**
**Eleven** deliberate breakages of `recon.py`, each caught and each reverted. Six
on the reduction itself: dropping the SPad reduction (caught by 7 tests), forcing
network scales to 1.0 (5), reducing the wrong stage at R2 (2), a silent 1% DRAM
credit (4), the mesh saving applied twice (4) — and **hard-wiring `Packing.frac`
to 51/63 so it ignores K, which only the code sweep caught (1)**. Five on the
register model added in §7.1: forcing the complement to full `weight_bits` (1),
deleting the register READ term so the pre-audit accounting comes back (2),
hard-wiring the complement to 51/63 so it ignores K (2, one of them only the K
sweep), stopping `full_width` from removing the SPad reads it serves (1, and
this one went **uncaught** until a test was added that runs `full_width` against
the real cache — the synthetic fixture has no ERT, so the path never executed),
and charging the register read per scalar instead of per word (1). A reduction that is applied but not keyed to the code passes
everything else.

A latent bug this found: `eyeriss_v2_like_wglb` inserts a reducible `weight_glb`
stage but reuses `eyeriss_v2_like`'s boundaries, none of which reduces it — every
boundary would have reported the GLB at full width while claiming its own saving,
with nothing saying so. `validate_placement_space()` now enforces the prefix
invariant and refuses that design; no `eyeriss_v2_like` number moved.

**The figure could not show any of this** — 4.60 µJ on a 4,798 µJ bar is about one
pixel — so `draw_panel` gained `bar_notes` (µJ moved and µJ paid, because a
percentage cannot separate R2's 0.096% from R3's 0.150%). A zoom panel was tried
and removed 2026-09-07: it resolved to exactly the weight share of `Local` and
`NoC`, restating the bars above at 12× scale.

### 7.2 The decoder on the DRAM die — 2026-09-09: every boundary saves DRAM interface energy

**Model change (01_project_context §1/§4).** The BCH decoder sits on the DRAM
die, off the fetch path, so only the k message bits of each codeword cross the
DRAM interface. The DRAM weight energy is now two weight-path stages,
`dram_array = (1 − f_if) × DRAM_w` (never reduced: a burst reads the complete
codeword) and `dram_interface = f_if × DRAM_w` (× K/N under **every** boundary,
R1 included). The reference bars keep controller-side correction and do not
move; `ECC_RECON_DECODE_SITE=controller` reproduces the §4.1.3 numbers to the
digit (verified column by column on the CSV).

**f_if is uncited; 0.40 is assumed.** `archs/_shared/provenance.yaml`
(`dram_interface_share`) records the two candidates found: O'Connor et al.
(MICRO 2017), HBM2 3.92 pJ/bit of which I/O is 0.30 (≈ 7.7 %, an unterminated
interposer link — a lower bound for LPDDR4); and Ha (Stanford PhD 2018,
Fig. 4.8), an LPDDR4 energy/bit breakdown whose off-die I/O category is only
plotted. Neither gives the number. Later on 2026-09-09 the study **assumed
f_if = 0.40** (env.sh default; 40 % of the per-bit LPDDR4 read energy spent
driving bits off the die over the terminated link) and `results/figures/
ReconSweep.png` is drawn at that value, labelled ASSUMED in its subtitle. A first
draft of the figure that day was at 0.25, set on the command line; its numbers
are the 0.25 rows below. With `ECC_DRAM_IF_FRAC` empty the run refuses and
prints the ceiling at 0.10 / 0.25 / 0.50. The interface saving is linear in
f_if and identical on every R bar, so f_if moves every placement against
embedded by the same offset and decides no ordering.

resnet18, BCH(63,39), 8x2 EDP mappings (the §4.1.3 cache), µJ; DRAM_w =
1,046.819 µJ = 17.62 % of the embedded total:

| f_if | array (1−f_if) | interface f_if | interface × (1−K/N) = the DRAM saving on every R bar | on-chip + interface ceiling |
|---:|---:|---:|---:|---:|
| 0.10 | 942.137 | 104.682 | 39.879 (0.67 %) | 197.011 (3.32 %) |
| 0.25 | 785.114 | 261.705 | 99.697 (1.68 %) | 256.829 (4.32 %) |
| **0.40 (assumed, the figure)** | **628.091** | **418.728** | **159.515 (2.69 %)** | **316.648 (5.33 %)** |
| 0.50 | 523.409 | 523.409 | 199.394 (3.36 %) | 356.526 (6.00 %) |

The bars at the assumed f_if = 0.40 (`results/figures/ReconSweep.png`):

| bar | total | vs conventional | vs embedded | DRAM I/O removed | recon | N_rec |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (external parity) | 6,954.500 | — | −17.071 % | — | — | — |
| Embedded (no external parity) | 5,940.394 | +14.582 % | — | — | — | — |
| R1 @ NoC source | 5,789.696 | +16.75 % | **+2.54 %** | 159.515 | 8.817 | 2,077,021 |
| ~~R2 @ cluster edge~~ | ~~5,776.223~~ | ~~+16.94 %~~ | ~~+2.76 %~~ | 159.515 | ~~8.817~~ | ~~2,077,021~~ |
| R3 @ SPad input | 5,777.551 | +16.92 % | +2.74 % | 159.515 | 12.837 | 3,023,807 |

**The R2 row above is SUPERSEDED by §7.4** — its encoder count is the mesh's
injections, not its destination-side arrivals. R1, R3, R4a and R4b are
unaffected. §7.4 has the corrected numbers and the reason.
| R4a / R4b | unsupported | | | | | G_rec = 9 > resident tile in layer4.* |

Of the DRAM weight **interface** energy itself the saving is 1 − K/N = 38.1 %
at every f_if; f_if only decides how much of the DRAM weight energy that
interface is (15.2 % of it at 0.40, 5.8 % of the whole DRAM category, 2.69 % of
inference energy).

Sensitivity of `vs embedded`, %:

| | R1 | R2 | R3 |
|---|---:|---:|---:|
| controller (pre-2026-09-09, §4.1.3) | −0.15 | +0.08 | +0.06 |
| ondie, f_if = 0.10 | +0.52 | +0.75 | +0.73 |
| ondie, f_if = 0.25 | +1.53 | +1.76 | +1.73 |
| **ondie, f_if = 0.40 (assumed)** | **+2.54** | **+2.76** | **+2.74** |
| ondie, f_if = 0.50 | +3.21 | +3.43 | +3.41 |

What this changes in the reading: **R1 is no longer the zero-saving control.**
It now isolates the interface saving every boundary shares from any on-chip
saving — at f_if = 0.40 it beats embedded by 159.515 − 8.817 = 150.698 µJ, and
every other boundary's saving is R1's plus its own on-chip term minus the extra
reconstruction it pays. The on-chip ceiling (157 µJ, 2.65 %) is unchanged; the
DRAM interface adds 159.5 µJ to it at f_if = 0.40 and is the larger of the two
above f_if ≈ 0.39 — so at the assumed value the DRAM interface is, just, the
largest single term any boundary can save. The prediction made before running — same DRAM drop on every
R bar, R1 flipping to positive by that drop minus its recon cost, R2 > R3 > R1
preserved — held exactly.

Recorded checks (every result file): `dram_array_identical_to_embedded_reference`
(no bar sits below emb_DRAM − f_if·DRAM_w·(1−K/N); array row unreduced) and
`dram_interface_scaled_by_K_over_N` (no bar sits above it; interface row at
exactly K/N). The test suite proves each fails on a cheat in its own direction
only, and that `controller` differs from `ondie` by exactly the interface saving
on every bar and by nothing else. §8 defect 3 is superseded by this section.

### 7.3 The MAC cost is the denominator of every percentage — audited 2026-09-09

**Why it matters.** An ECC saving is saved µJ / total µJ. The saved µJ are weight
traffic (DRAM parity, DRAM interface, on-chip weight reads) and do not depend on
what a MAC costs; the total does, and on `eyeriss_v2_like` / resnet18 the MACs
are 2,120 of 5,940 µJ (35.7 % of the embedded total, 44 % of the Timeloop total
before the interface split; 50 % of on-chip energy).

**Where the number comes from, traced in the container** (recorded in
`archs/_shared/provenance.yaml` `mac_energy_pj`):

* `timeloop-mapper.ERT_summary.yaml`, `system_top_level.mac[1..384]`, action
  `compute`: **1.16877 pJ**. Primitive estimations: `estimator: Library` for
  both `intadder` and `intmultiplier` — the accelergy-library-plugin, not the
  Aladdin_table plug-in (which supports `intmac` too but was not chosen).
* The component: `arch_paper.yaml` `mac`, `class: intmac`, `multiplier_width 8`,
  `adder_width 20`, 45 nm. The compound is
  `timeloop-accelergy-exercises/.../_components/intmac.yaml`: intmac =
  `aladdin_adder(width = adder_width)` + `aladdin_multiplier(width_a = width_b =
  multiplier_width)`, `compute` = one read of each.
* The table rows, one each: `library/aladdin/aladdin_multiplier.csv` — `40nm,
  1e-9, 32, 32, 32, 12.68 pJ, multiply|read`; `library/aladdin/aladdin_adder.csv`
  — `40nm, 1e-9, 32, 0.21 pJ, add|read`. Both from Aladdin's HLS library,
  origin otherwise unstated.
* The scaling rule, from `accelergywrapper.match_entry` → `scaling.scale_energy`:
  `width`, `width_a`, `width_b` scale energy **linearly** (`v1/v0`);
  `technology` scales by `get_tech_node_energy_scale` (Stillmaker & Baas 2017
  polynomial at Vdd = 0.8, × (to/from)^0.5), which from 40 nm to **45 nm is
  1.265241×** — the table sits at a smaller node than the model, so the number
  is scaled *up*. Verbatim from `timeloop-mapper.accelergy.log`: `Scaled
  aladdin_multiplier.width_a from 32.0 to 8.0: 0.25x energy`, `...width_b ...
  0.25x`, `...technology from 40.0 to 45.0: 1.265241016612673x`,
  `aladdin_multiplier energy has been scaled 0.07907756353829207x`.
* The arithmetic: 12.68 × 0.25 × 0.25 × 1.265241 = 1.00270 pJ (multiplier) +
  0.21 × 20/32 × 1.265241 = 0.16606 pJ (adder) = **1.16876 pJ**, the ERT's value
  to the digit. `MACs × 1.16877 = Compute` on the raw record to 1e-11.

**Cited alternatives** (provenance.yaml has the full entries):

| source | node / precision | per MAC | note |
|---|---|---:|---|
| Accelergy ERT, as modelled | 45 nm (scaled from 40), int8×int8 + 20b add | 1.16877 pJ | one HLS table row each, linear width scaling, scaled up 40→45 nm |
| Horowitz, ISSCC 2014, Fig. 1.1.9 | 45 nm 0.9 V, int8 mult 0.2 + int8 add 0.03 | **0.23 pJ** | the row below; caption confirmed in the PDF, cell values as commonly reproduced |
| Horowitz, 20b add interpolated | int8 mult 0.2 + 20b add ≈ 0.065 | 0.265 pJ | the model's accumulator is 20 b |
| Eyeriss JSSC 2017, Fig. 16 (measured, 65 nm, 16 b) | share, not pJ | — | "the ALUs only account for less than 10% of the total power … data movement … up to 45%" — quote confirmed. Here the MACs are 50 % of on-chip energy |
| Eyeriss ISCA 2016 normalised table | ALU 1×, RF 1×, PE-PE 2×, buffer 6×, DRAM 200× | — | not re-read (PDF unreachable). Here MAC / RF read = 5.9× (24 b `weights_spad` word per weight) to 27.6× (8 b `ifmap_spad`) |
| `data/dc/` synthesis | — | none | only the BCH encoder/decoder datapaths were synthesized |

**The knob.** `ECC_MAC_PJ_OVERRIDE` (env.sh §5, read only in `config.py`)
rescales the Compute category to MACs × value in the *evaluator*
(`energy.apply_mac_override`, applied after the raw cache is read or written,
so `results/_raw/` stays pure Timeloop output). Nothing else moves: the DRAM,
buffer, scratchpad and NoC pJ, the external-parity pJ, the embedded DRAM pJ
and every boundary's saved and reconstruction pJ are bit-identical
(`tests/test_mac_override.py` asserts each). A value listed in provenance.yaml
carries its citation on the figure subtitle, the manifest (`mac_energy`) and
the result file; any other value is labelled uncited. The MAC count is
mapping-invariant, so under the energy objective the mapping optimum does not
move and the cache stays warm; the current mappings are EDP-optimal, so the run
prints a warning that a cheaper MAC could move an EDP optimum and the mapper
was not re-run. **Decided later on 2026-09-09: the default is 0.23 pJ** (Horowitz int8
multiply + add) and `results/figures/ReconSweep.png` is drawn under it, with the
value, its citation and the ERT number it replaced in the subtitle and the
manifest. The ERT row below is now the sensitivity; `ECC_DRAM_IF_FRAC` style,
set `ECC_MAC_PJ_OVERRIDE=` (empty) to reproduce it. The mapper was not re-run:
it prices MACs from Accelergy's ERT regardless of this knob, so re-mapping would
change nothing unless the `intmac` component itself were edited (a cold cache),
and Task 3 is a fixed-mapping study. The EDP caveat stands: a cheaper MAC could
move an EDP-optimal mapping; the numbers here are on the mappings as they are.

**The table.** resnet18, `eyeriss_v2_like`, fixed 8x2 EDP mapping, BCH(63,39),
f_if = 0.40 (assumed, §7.2), µJ:

| | MAC pJ | Compute | embedded total | conventional total | embedded vs conventional | Task 3 ceiling, on-chip | ceiling incl. DRAM interface | R1 vs emb | R2 vs emb | R3 vs emb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| as modelled (ERT) | 1.16877 | 2,120.235 | 5,940.394 | 6,954.500 | 14.58 % | 2.65 % | 5.33 % | +2.54 % | ~~+2.76 %~~ | +2.74 % |
| Horowitz int8 MAC | 0.23 | 417.237 | 4,237.397 | 5,251.503 | **19.31 %** | 3.71 % | 7.47 % | +3.56 % | ~~+3.87 %~~ | **+3.84 %** |

The **R2 column is superseded by §7.4** (its encoder count was the mesh's
injections, not its arrivals); at the corrected count R2 is +3.78 % on the
Horowitz row and **R3 is the best boundary on both rows**. Nothing else in this
table moves — the MAC denominator and the encoder count are independent, and
the 1.402× / 1.324× rescaling still holds column by column.

Saved µJ, identical on both rows to the digit (checked column by column on the
two CSVs; only `energy_Compute`, the totals and the percentages differ):
external parity 1,014.106; DRAM interface 159.515 on every R bar; R2 mesh
13.47; R3 mesh + cluster 16.16; reconstruction 8.817 / 8.817 / 12.837. Every
percentage on the second row is the first × 5,940.394 / 4,237.396 = **1.402×**
(the conventional-reference ones × 6,954.500 / 5,251.503 = 1.324×). No ordering
moves *with the denominator*: the same ordering appears on both rows, and
embedded beats conventional on both. (Which ordering that is changed for a
different reason — see §7.4.)

**Which row the paper quotes, and why.** The decision (2026-09-09, the user's)
is the **Horowitz row as the primary number: embedded ECC saves 19.3 % of
inference energy against conventional ECC, and the best boundary (R2) adds
3.87 % over embedded**, with the as-modelled row reported beside it as the
denominator sensitivity (14.6 % / 2.76 %) and one sentence saying the
difference is the MAC cost, not the ECC model. My recommendation before that
decision was the reverse ordering of the two rows, for these reasons, which
still hold and should appear in the paper's caveat: (i) the ERT number is what the whole study, every
architecture and every cached mapping was produced with, and the EDP mappings
were optimised against it, so swapping the denominator after the fact under
EDP is a fixed-mapping sensitivity, not a re-optimised design; (ii) the
Horowitz figure is a cited 45 nm number at the right precision, but its cells
were read from a reproduced table rather than the figure itself and its adder
is 8 b where the model's is 20 b (0.265 pJ with the interpolated add, 15 %
higher), so it is a bound on the denominator, not a measurement of this
design's MAC; (iii) the audit's actual finding — one 40 nm HLS row scaled up
in node and down linearly in width, 5× Horowitz and ~5× the measured Eyeriss
ALU share — is strong enough that the ERT row cannot be quoted *alone* either.
The right fix is a cited 45 nm int8 MAC in the `intmac` component (a component
edit, which colds every mapper cache); until that is agreed the knob's default
stays EMPTY and the table carries both rows.

### 7.4 A network boundary's encoders run once per ARRIVAL — corrected 2026-09-09

**The defect.** R2 credits its network with carrying the reduced form, which
means its encoders sit at that network's *destinations*. Its reconstruction
count was the network's **ingresses** — the words *injected*. Those are two
different placements, and only one of them is R2: an encoder placed before the
fanout expands the traffic before it enters the network, which is R1. So R2 was
taking a destination-side saving at a source-side price.

The context document names exactly this tradeoff (§7.1) and asks for it as an
experiment variable:

    BEFORE MULTICAST   4b -> Encoder -> 8b -+-> PE  (x fanout)
        one reconstruction at the source, but FULL-WIDTH network traffic
    AFTER MULTICAST    4b -+-> Encoder -> PE        (x fanout)
        replicated encoders, but REDUCED-WIDTH shared transport

**The count, from Timeloop's own numbers.** A multicast network prints its split
as `Ingresses : N` followed by `@multicast M @scatter S: N`, so the
destination-side arrivals are `sum(M × N)` over those lines. Summing the
breakdown matters rather than using the single `Multicast factor` field: on
`eyeriss_v2_like`'s resnet18 mapping the mesh reports factor 2 but the exact sum
is 1.456× its injections, because the words are not all multicast alike.

**It reconciles, and that is the check.** A network's arrivals should equal what
the next stage takes in. Recorded on every result file
(`multicast_arrival_chain`), and on `eyeriss_v2_like`:

| network | injected | arrivals | × | next stage takes in | match |
|---|---:|---:|---:|---:|:--|
| inter_cluster_mesh | 16,356,544 | 23,812,480 | 1.456 | cluster_local ingresses 23,812,480 | yes |
| cluster_local | 23,812,480 | 23,812,480 | 1.000 | weight_spad fills 23,812,480 | yes |

**What it changes.** `eyeriss_v2_like`, resnet18, 8×2 EDP mapping, BCH(63,39),
f_if = 0.40, MAC 0.23 pJ — i.e. the primary row of §7.3:

| bar | vs embedded, before | vs embedded, corrected | N_rec before | N_rec corrected |
|---|---:|---:|---:|---:|
| R1 @ NoC source | +3.56 % | +3.56 % | 2,077,021 | 2,077,021 |
| R2 @ cluster edge | **+3.87 %** | +3.78 % | 2,077,021 | 3,023,807 |
| R3 @ SPad input | +3.84 % | **+3.84 %** | 3,023,807 | 3,023,807 |

**So R3, not R2, is the best boundary on Eyeriss v2** — and the reason is
legible rather than numerical: the mesh's arrivals *equal* the cluster-local
network's ingresses, so R2 and R3 pay the **same** encoder count, and R3 keeps
one more stage reduced. R2 could only have won by paying less than R3, which is
what the ingress count was doing for it. R1 and both PE-local boundaries are
untouched: R1 counts DRAM codewords and a scratchpad boundary already counts
destination-side accesses.

`ECC_RECON_ENCODER_SITE=source` reproduces the earlier numbers exactly, the same
way `ECC_RECON_DECODE_SITE=controller` reproduces the pre-DRAM-split ones, so
the change is a diff and not a rewrite. The figure subtitle states which reading
it was drawn under, and every network bar's result records **both** counts, the
multicast factor and the multiplicity, so neither reading is hidden by the
choice of the other.

**How it was found:** adding the weight-stationary and Eyeriss v1 weight paths.
Eyeriss v1's column network multicasts up to 7-fold on resnet18 shapes, which
made the discrepancy a factor of 7 rather than 1.456 and impossible to read as
rounding.

### 7.5 Weight-stationary and Eyeriss v1 — whole resnet18, 2026-09-09

`results/figures/ReconSweep.png`, one panel per design. Fixed EDP mappings,
21/21 layers, BCH(63,39), f_if = 0.40 (assumed), MAC 0.23 pJ, encoder site
`destination` (§7.4). Each design is measured against **its own** two reference
bars, so a percentage on one panel says nothing about the other.

**Weight stationary** — conventional 7.471 mJ, embedded 6.695 mJ (embedded saves
**10.39 %**). Ceiling: on-chip 402.98 µJ (6.02 %) + DRAM interface 122.08 µJ
(1.82 %) = **525.07 µJ, 7.84 %**.

| boundary | rating | total mJ | vs conventional | vs embedded | recon µJ | N_rec |
|---|:--|---:|---:|---:|---:|---:|
| R1 @ chip ingress | control | 6.580 | +11.93 % | +1.72 % | 6.75 | 1,589,605 |
| R2 @ buffer output | 3/5 | 6.540 | +12.47 % | +2.32 % | 16.59 | 3,908,494 |
| R3 @ PE input | 4/5 | 6.521 | +12.72 % | +2.61 % | 16.59 | 3,908,494 |
| R4a @ RF output | 2/5 | 7.207 | +3.54 % | **−7.64 %** | 914.63 | 215,449,079 |
| **R4b @ RF + reg** | **5/5** | **6.329** | **+15.28 %** | **+5.46 %** | 16.59 | 3,908,494 |
| R5 @ MAC input | 2/5 | unsupported | | | | resident 1 < G_rec 9, all 21 layers |

**This is the source discussion's ordering, reproduced exactly**:
5/5 > 4/5 > 3/5 > control > 2/5. It is the only design in the study where every
predicted rank holds, and the two extremes are both large: R4b captures
**70 % of the whole ceiling** (5.46 of 7.84 %), and R4a — reconstruct on every
RF read — is the one boundary anywhere in this study that **costs more energy
than it saves**, by 7.6 %. The mechanism is the reconstruction count: 215.4 M
codewords against R4b's 3.9 M, a 55.1× amortization from a register covering the
384-weight inner tile. §6.1's illustrative "100 reduced reads and 100
reconstructions" is real, and it is expensive.

Weight path (weight energy only, µJ): dram_array 480.70, dram_interface 320.46,
operand_glb 130.41, weight_noc 50.72, pe_spad 556.89, weight_reg 319.82.

Two findings specific to this design, both from having levels no other
registered design has:

* **Its existing stationary register is not R4b's reuse register.** `weight_reg`
  is depth 1 and the mapping fills it once per read (1,696,661,504 fills against
  1,814,073,344 reads), so it is a pipeline latch, not a reuse mechanism. R4b's
  register has to cover the inner tile — 384 weights — and is charged as an
  addition. CLAUDE.md's rule for Simba ("determine whether the proposal can
  reuse an existing register") asked the question; the answer here is no.
* **The MAC-input boundary is rejected, not estimated.** A rebuild needs `G_rec`
  = 9 co-resident weights and that register holds one, on all 21 layers. §6.2
  rates the row 2/5; the model's answer is that it does not exist on this design
  as mapped, with the layers named.

**Eyeriss v1** — conventional 7.643 mJ, embedded 6.290 mJ (embedded saves
**17.70 %**). Ceiling: on-chip 397.99 µJ (6.33 %) + DRAM interface 212.81 µJ
(3.38 %) = **610.80 µJ, 9.71 %**.

| boundary | rating | total mJ | vs conventional | vs embedded | recon µJ | N_rec |
|---|:--|---:|---:|---:|---:|---:|
| R1 @ source | 3/5 | 6.089 | +20.33 % | +3.20 % | 11.76 | 2,770,976 |
| R2 @ column edge | 4/5 | 6.123 | +19.88 % | +2.65 % | 76.55 | 18,033,062 |
| **R3 @ spad input** | 4/5 | **6.060** | **+20.71 %** | **+3.65 %** | 76.55 | 18,033,062 |
| R4a / R4b | 3/5, 5/5 | unsupported | | | | resident 8 < G_rec 9, 4 layers |

**R2 falls BELOW R1 here, and the multicast factor is why.** The column network
injects 21,821,440 weight words and its 14 destinations receive 142,010,368 —
6.5× as many — so a destination-side encoder is replicated enough to cost
76.55 µJ where R1 pays 11.76 µJ, against a network saving of only 30.15 µJ.
This is §7.1's multicast tradeoff resolved *against* replicated encoders on this
design, and it is the first place in the study where a later boundary is worse
than an earlier one for a reason that is not reconstruction-per-read.

**R3 dominates R2 structurally, on this design and on Eyeriss v2.** A network's
destination-side arrivals *are* what the next stage takes in (§7.4), so a
boundary at the last network's output and a boundary at the scratchpad input pay
the **same** encoder count — R3 simply keeps one more stage reduced. R2 can only
win by paying less, which is what the pre-§7.4 ingress count was doing for it.

Weight path (µJ): dram_array 837.94, dram_interface 558.63, array_multicast
79.15, pe_local_multicast 166.02, weights_spad 799.54.

**Why v1's PE-local boundaries are unsupported and v2's are too.** The EDP
mappings keep 8 weights resident per PE on four layer4 shapes, one short of
`G_rec` = 9. `G_rec` is fixed by the code and the weight width
(`ceil(63/8) + 1`), so the only lever is the mapping, not a knob: an
energy-optimal mapping keeps 128–192 weights resident and every boundary
evaluates. That run is `ReconSweep_optEnergy.png`, and the pipeline's own
`diagnose` already warns that EDP "trades energy for latency" in an energy
study. **Reported as infeasible with the layers named, never estimated.**

**Cross-design reading, with the caveat that the three designs are measured
against three different reference bars.** Embedded ECC alone saves 10.4 %
(weight-stationary), 17.7 % (Eyeriss v1) and 19.3 % (Eyeriss v2) — the spread
is DRAM refetch, which is set by on-chip weight capacity. Reconstruction adds
+5.46 %, +3.65 % and +3.84 % on top. The best boundary is the reuse register
where the mapping leaves a PE-local boundary feasible, and the scratchpad input
where it does not.

### 7.6 The mapper objective, not the buffer sizes — `ReconSweep_optEnergy.png`

Same two designs, same architectures, **only `ECC_OPT_METRIC` changed from `edp`
to `energy`**. No YAML was edited. 21/21 layers, BCH(63,39), f_if = 0.40, MAC
0.23 pJ, encoder site `destination`.

| | Weight stationary | | Eyeriss v1 | |
|---|---:|---:|---:|---:|
| objective | EDP | energy | EDP | energy |
| conventional, mJ | 7.471 | 7.385 | 7.643 | **4.539** |
| embedded, mJ | 6.695 | 6.611 | 6.290 | **3.613** |
| embedded vs conventional | 10.39 % | 10.48 % | 17.70 % | **20.40 %** |
| ceiling (on-chip + interface) | 7.84 % | 7.79 % | 9.71 % | **12.33 %** |
| R1 | +1.72 % | +1.74 % | +3.20 % | +3.81 % |
| R2 | +2.32 % | +2.30 % | +2.65 % | +3.85 % |
| R3 | +2.61 % | +2.54 % | +3.65 % | +4.20 % |
| R4a | −7.64 % | −7.89 % | unsupported | **−14.74 %** |
| **R4b** | **+5.46 %** | **+5.42 %** | unsupported | **+11.38 %** |
| R5 | unsupported | unsupported | n/a | n/a |
| PE weight residency | 16 | 16 | **8** | **20** |
| DRAM refetch | 1.072× | 1.068× | 1.868× | 1.278× |

**Weight stationary does not move** — every bar within 0.05 pp. Its mapping was
never DRAM- or capacity-limited (refetch 1.07×), so there was nothing for a
different objective to recover.

**Eyeriss v1 changes completely.** Its scratchpad residency goes from 8 weights
to 20, clearing `G_rec` = 9, so **R4a and R4b become feasible and R4b is the
best result anywhere in this study: +11.38 % over embedded, +29.45 % over
conventional ECC, capturing 92 % of its own 12.33 % ceiling** (61.5×
reconstruction amortization from a 384-entry register). R4a is the mirror image
at −14.74 %: 230.4 M reconstructions against R4b's 3.7 M. The two extremes of
the source discussion's 5/5 and 3/5 rows are 26 percentage points apart on one
design.

**Why the SAVING PERCENTAGE rose even though refetch FELL.** This corrects an
over-strong inference. The interface saving is exactly proportional to DRAM
weight traffic — verified to four digits, EDP v1/WS = 1.7432 for reads, for DRAM
weight energy and for the interface saving alike — so cutting refetch really
does cut that term in µJ, and it did: 212.81 → 145.61 µJ. But the saving is a
*ratio*, and the objective switch cut the denominator harder:

| Eyeriss v1, µJ | EDP | energy | change |
|---|---:|---:|---:|
| DRAM weight | 1,396.6 | 955.6 | −31.6 % |
| NoC weight | 245.2 | 58.1 | **−76.3 %** |
| scratchpad weight | 799.5 | 729.0 | −8.8 % |
| everything NOT weight | 3,848.7 | 1,870.3 | **−51.4 %** |
| total | 6,290.0 | 3,613.0 | −42.6 % |

The energy objective cut activations, partial sums and interconnect far harder
than it cut the weight path, so weight movement became a **larger share** of
inference energy — 16.6 % → 21.8 % on chip — and the interface saving rose from
3.38 % to 4.03 % of the total despite falling 32 % in µJ.

**The rule this establishes.** What raises the ECC saving is not more weight
traffic and not bigger buffers; it is **weight movement being a larger fraction
of inference energy**. Cutting non-weight energy raises the saving. Cutting
weight energy lowers it. A change that cuts both — a larger buffer, a different
objective — can go either way, and the ratio has to be measured rather than
predicted from the refetch factor alone.

**Which run to quote.** `ReconSweep.png` (EDP) is the configuration the rest of
the study and every other cached mapping was produced under. `ReconSweep_optEnergy.png`
is the energy-objective run, and `bash run.sh diagnose` already warns that
`edp` "trades energy for latency" in an energy study. On Eyeriss v1 the two
differ by 42.6 % of total energy and by whether its 5/5 boundary exists at all,
so **the pair is the result and neither should be quoted alone**. The same rule
CLAUDE.md sets for `eyeriss_like` / `eyeriss_like_wglb` applies here.

**Still unsupported, and structurally so:** weight-stationary's R5 (MAC input),
on all 21 layers under both objectives. Its stationary register holds one
weight and a rebuild needs nine. No mapping fixes that; only a wider register
would, and the design declares `depth: 1`.

### 7.7 Task 4 first pass — WITHDRAWN 2026-09-09

> **Withdrawn the same day it was measured, before anything was quoted. Full
> text archived in `legacy/FINDINGS_detail_2026-09-10.md`.**

Measured under `ECC_OPT_METRIC=energy`, which SERIALISES: it used 4, 8 or 56 of
Eyeriss v1's 168 PEs depending on the capacity, because fewer active PEs is less
energy whatever it costs in latency. The compared mappings differed in PE count
by up to 14× while the capacity under test differed by 17×, so capacity and
parallelism could not be separated.

**The lesson that survives, and it is now a standing rule:** never run a capacity
comparison under `energy`. `edp` penalises serialisation and is env.sh's default.
The sweep was re-run under it — §7.8 is that result, and it withdrew two further
positives of its own for two more reasons.

### 7.8 Task 4 under EDP — the two positives were artifacts, and refetch is a loop-ORDER property

**Measured 2026-09-09, `ECC_OPT_METRIC=edp` throughout** (§7.7's `energy` pass
is withdrawn). resnet18, `layer2.0.conv1` = 73,728 unique weights, BCH(63,39),
N/K = 1.6154. Read with `python3 -m eccenergy.experiments.dilation --table`,
which now carries `PEs used`, `weights held` and a per-pair verdict.

| design | arm | ×cap | w/PE | PEs | room | **held** | fill | DRAM w rd | refetch | µJ | fullest other | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `eyeriss_like` | emb | ×1 | 448 | 168 | 75,264 | 21,504 | 28.6 % | 589,824 | 8.000 | 220.32 | `ifmap_spad` 67 % | — |
| `eyeriss_like` | rec | ×1.6154 | 724 | 168 | 121,632 | **21,504** | 17.7 % | 589,824 | 8.000 | 228.72 | `ifmap_spad` 67 % | flat |
| `eyeriss_like` | emb | ×4 | 1,792 | 168 | 301,056 | **21,504** | 7.1 % | 589,824 | 8.000 | 259.36 | `ifmap_spad` 67 % | flat |
| `eyeriss_like` | emb | ×8 | 3,584 | 168 | 602,112 | **21,504** | 3.6 % | 589,824 | 8.000 | 309.25 | `ifmap_spad` 67 % | flat |
| `eyeriss_like` | emb | ×16 | 7,168 | 168 | 1,204,224 | **21,504** | 1.8 % | 589,824 | 8.000 | 406.00 | `ifmap_spad` 67 % | flat |
| `eyeriss_like` | emb | ×32 | 14,336 | 168 | 2,408,448 | **21,504** | 0.9 % | 589,824 | 8.000 | 454.38 | `ifmap_spad` 67 % | flat |
| `eyeriss_v2_like` | emb | ×1 | 288 | 144 | 41,472 | 2,304 | 5.6 % | 294,912 | 4.000 | 132.45 | `psum_glb` 51 % | — |
| `eyeriss_v2_like` | rec | ×1.6154 | 465 | 144 | 66,960 | **2,304** | 3.4 % | 294,912 | 4.000 | 135.37 | `psum_glb` 51 % | flat |
| `eyeriss_v2_like` | emb | ×4 | 1,152 | 144 | 165,888 | **2,304** | 1.4 % | 294,912 | 4.000 | 145.41 | `psum_glb` 51 % | flat |
| `simple_weight_stationary` | emb | ×1 | 384 | 16 | 71,680 | 18,816 | 26.2 % | 147,456 | 2.000 | 225.35 | `output_activation_reg` **100 %** | — |
| `simple_weight_stationary` | rec | ×1.6154 | 620 | 16 | 75,456 | **18,816** | 24.9 % | 73,728 | **1.000** | 230.68 | `output_activation_reg` **100 %** | **PERM?** |
| `simple_weight_stationary` | emb | ×4 | 1,536 | 16 | 90,112 | **18,816** | 20.9 % | 147,456 | 2.000 | 259.27 | `output_activation_reg` **100 %** | flat |

**`weights held` never moves — not once, over a 32× capacity range.** That one
column is the whole result. Room grows 32× and the mapper stores exactly the
same number of weights on chip, so nothing was bought and nothing could be.
Total energy meanwhile rises monotonically (220 → 454 µJ on v1) because
Accelergy prices the deeper array as bigger silicon, which is the bias
CLAUDE.md records against the hypothesis.

#### Both positives reported before the guards existed were artifacts

**1. `eyeriss_v2_like` "14.00× → 4.00×" was a CROSS-FINGERPRINT comparison.**
The 14.00× came from `fp-3eb860ea2b2a`, solved 2026-09-08 23:49. The
fingerprint the current configuration resolves to is `fp-88656178371f`, solved
2026-09-09 01:33, and it **already refetches 4.00× undilated** — the ×1,
×1.6154 and ×4 loop nests are byte-identical. `fp-<hash>` hashes the patched
YAML plus the globals plus every mapper setting, so two fingerprints under one
variant slug are two *architectures*. The tell was in plain sight: PEs moved
192 → 144 across the "pair", which no capacity change can do.
`sibling_fingerprints()` now enumerates them and the table refuses the
comparison in a `!! FINGERPRINT WARNINGS` block.

**2. `simple_weight_stationary` "2.00× → 1.00×" was a LOOP PERMUTATION**, free
of capacity. The two nests differ in the order of two DRAM-level loops and in
nothing else whatsoever:

```
  x1.0    | for Q in [0:2)      x1.6154 | for C in [0:4)      x4.0    | for Q in [0:2)
          |   for C in [0:4)            |   for Q in [0:2)            |   for C in [0:4)
```

Below DRAM the three mappings are identical — `operand_glb` holds 18,432
weights, `pe_spad` 24/PE, the spatial fanout is M=8 × C=2 — at *every* scale.
With `C` outermost the weight tile is invariant across `Q`, so it is retained
and 4 × 18,432 = 73,728 reads suffice; with `Q` outermost every `Q` step
re-reads all four C-tiles, 8 × 18,432 = 147,456. The permutation was available
at the declared capacity and ×4 lost it again. That is `random_pruned` at
victory 2000, not silicon.

#### The mechanism: refetch is set by loop ORDER at the DRAM level

On all three designs the refetch factor is exactly the product of the
DRAM-level loop factors that do **not** index Weights and sit outside one that
does:

| design | DRAM-level loops | non-weight factor | refetch |
|---|---|---:|---:|
| `eyeriss_like` | `Q(2) M(4) P(4)` | Q×P = 8 | 8.000 |
| `eyeriss_v2_like` | `Q(4) C(4)` | Q = 4 | 4.000 |
| `simple_weight_stationary` | `Q(2) C(4)` | Q = 2 | 2.000 |
| `simple_weight_stationary` (dilated, reordered) | `C(4) Q(2)` | — (Q innermost) | 1.000 |

**A weight tile cannot index P or Q.** So a weight buffer *inside* the PE
array structurally cannot absorb these loops however large it is made, which is
why v1 is flat to ×32 at 1 % fill. Putting the non-weight loop innermost at
DRAM does absorb it — but it forces the input and psum tiles to be re-read
across that loop instead, and EDP decides which trade wins. Capacity only
becomes the binding constraint once the weight level is too small to hold the
tile the chosen order requires, which at declared sizes it is not: the fullest
level is an *activation* or *psum* buffer on every row above.

#### What this predicts, and what is being run to test it

Two levels in the study are weight buffers **above** the PE array, and both
hold 65,536 weights = **88.9 %** of this layer — they just miss, and N/K
clears it:

| design | level | declared holds | break at | money window (ref) |
|---|---|---:|---:|---|
| `eyeriss_v2_like_wglb` | `weight_noc` | 88.9 % | ×1.1250 | **[0.6964, 1.1250)** |
| `simple_weight_stationary` (`scope=shared`) | `operand_glb` | 88.9 % | ×1.1250 | **[0.6964, 1.1250)** |
| `eyeriss_like_wglb` | `filter_glb` | 11.1 % | ×9.0000 | [5.5714, 9.0000) |

Inside the money window the embedded arm cannot hold the layer and the
reconstruction arm can, so `weights held` must *rise* — which is what separates
a capacity effect from the two artifacts above. `ECC_WEIGHT_CAPACITY_SCOPE=shared`
is required for weight-stationary: under `exclusive` its `operand_glb` is never
dilated (it holds Inputs too), so the ×1.6154 run above enlarged only a PE
scratchpad that could not matter. **136 EDP mapping jobs** cover the downward
pairs on the three exclusive designs (refs 0.5 → 0.03125), weight-stationary
under `shared`, and both `_wglb` designs, which had **no EDP cache at all**.

#### EDP does NOT hold the PE count fixed once the buffer is SHRUNK — the downward sweep is confounded below ×0.5

§7.7 withdrew the `energy` pass because that objective traded PEs for capacity.
`edp` was adopted because it penalises serialisation, and **at and above the
declared capacity it does hold**: `eyeriss_like` runs 168/168 PEs at ×1, ×0.5,
×1.6154 and ×0.8077. **Below ×0.5 it stops holding.** Measured, `eyeriss_like`
layer2.0.conv1, all EDP:

| ×cap | w/PE | PEs | room | held | fill | DRAM w rd | refetch | µJ | fullest other | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| ×1 | 448 | 168 | 75,264 | 21,504 | 28.6 % | 589,824 | 8.000 | 220.32 | `ifmap_spad` 67 % | — |
| ×1.6154 | 724 | 168 | 121,632 | 21,504 | 17.7 % | 589,824 | 8.000 | 228.72 | `ifmap_spad` 67 % | flat |
| ×0.5 | 224 | 168 | 37,632 | 21,504 | 57.1 % | 589,824 | 8.000 | 209.96 | `ifmap_spad` 67 % | — |
| ×0.8077 | 362 | 168 | 60,816 | 21,504 | 35.4 % | 589,824 | 8.000 | 216.29 | `ifmap_spad` 67 % | **flat** |
| ×0.25 | 112 | **84** | 9,408 | 4,032 | 42.9 % | **294,912** | **4.000** | 173.16 | `ifmap_spad` **100 %** | — |
| ×0.4038 | 180 | **168** | 30,240 | 21,504 | 71.1 % | 589,824 | 8.000 | 208.58 | `ifmap_spad` 67 % | **PE!=** (−7.58 %) |
| ×0.125 | 56 | **96** | 5,376 | 1,152 | 21.4 % | 589,824 | 8.000 | 153.89 | `ifmap_glb` 51 % | — |
| ×0.2019 | 90 | **112** | 10,080 | 3,584 | 35.6 % | **294,912** | **4.000** | 184.77 | `ifmap_glb` 94 % | **PE!=** (+14.13 %) |
| ×0.0625 | 28 | 96 | 2,688 | 1,152 | 42.9 % | 589,824 | 8.000 | 166.41 | `ifmap_glb` 51 % | — |
| ×0.101 | 46 | 96 | 4,416 | 1,152 | 26.1 % | 589,824 | 8.000 | 152.90 | `ifmap_glb` 51 % | flat |

**Read the `PEs` column down.** 168 → 84 → 96 → 112: the mapper trades
parallelism for capacity under EDP too, once the weight buffer is small enough
that keeping the array fed costs more latency than it saves. So the two rows
that look like wins are not usable:

* **×0.125 → ×0.2019 shows +14.13 %** with `held` genuinely rising 1,152 →
  3,584 and refetch 8.000 → 4.000 — but PEs move 96 → 112. Some of that is 17 %
  more compute, not more capacity, and the two cannot be separated. This is
  exactly the failure mode §7.7 was withdrawn for, so `capacity_verdict()`
  makes `PE!=` **suppress** the capacity claim rather than sit beside it.
* **×0.25 → ×0.4038 shows −7.58 %** — the *embedded* arm is better. At ×0.25
  the mapper serialised to 84 PEs, drove `ifmap_spad` to 100 % and found a
  refetch-4.000 nest; the dilated arm kept 168 PEs and refetch 8.000. A
  **smaller** buffer refetching **less** is not a capacity effect in either
  direction.

**Within the range where the comparison is clean — ×0.5 and above, PEs pinned
at 168/168 — every pair is `flat`.** That is the result: the capacity dilation
buys exactly zero where it can be measured without confound, and where a
number does appear the parallelism has moved underneath it.

This is a **methodological limit on the whole downward sweep**, not a property
of one layer: shrinking the weight buffer far enough to make capacity bind also
changes what the objective wants to do with the PE array. A clean measurement
needs the spatial fanout pinned as well, which is not expressible per-layer in
these arch YAMLs.

#### HOW FAR N/K FALLS SHORT — measured at the one operating point where capacity nearly binds

The `eyeriss_v2_like` pair at **×0.25 → ×0.4038** is the cleanest measurement in
the sweep: the PE count matches (192 PEs, 384/384 MACs on both arms), so there
is no parallelism confound, and the embedded arm's weight buffer is at
**88.9 % fill** — capacity is all but binding, the one condition under which a
dilation is supposed to pay. The two loop nests come back **byte-identical**:

| | ×0.25 (embedded) | ×0.4038 (recon) |
|---|---:|---:|
| capacity | 72 weights/PE | **117 weights/PE** |
| resident tile | 64 weights/PE | **64 weights/PE** |
| fill | **88.9 %** | 54.7 % |
| PEs | 192 | 192 |
| DRAM weight reads | 2,064,384 | 2,064,384 |
| refetch | 28.000 | 28.000 |

**63 % more weight room, at 88.9 % fill, bought nothing.** The reason is
integer tiles, and here it is arithmetic rather than opinion. The resident tile
is `C(4) × M(8) × M(2 spatial)` = 64 weights. `C` = 64 is factored 4 (psum_glb
temporal) × 4 (spatial-X) × 4 (spad temporal), so the only way to grow the
spad's share is `C: 4 → 8` — **a 2× step in the tile**, to 128 weights. At
3 weights per 64-bit word that needs `depth ≥ 43`, i.e. scale ×0.4479, i.e. a
**×1.7917 dilation** over the ×0.25 reference.

> **N/K = 1.6154 delivers 117 weights/PE against the 128 needed. It falls short
> by a factor of 1.109 — eleven weights per PE.**

And the shortfall is not marginal in code terms. The dilation the step requires
is ×1.7917, so:

| code | N/K | clears ×1.7917? |
|---|---:|---|
| BCH(63,51) | 1.2353 | no |
| BCH(63,45) | 1.4000 | no |
| BCH(63,39) | **1.6154** | **no** |
| BCH(63,36) | 1.7500 | **no** — misses by 2.4 % |
| BCH(63,30) | 2.1000 | yes |

**No code weaker than BCH(63,30) buys this tile step**, and BCH(63,36) misses it
by 2.4 %. That is the concrete answer to "how far does N/K fall short": at the
one operating point in this sweep where weight capacity nearly binds, it needs
1.79× and has 1.62×, and the next code that would work costs a 3.3× parity
overhead instead of 1.6×.

Two caveats on generalising it. The step size is a property of how the layer's
`C` happens to factor across three levels, so another layer or another design
will have a different one — some will be 2×, some 1.5×, some unreachable at any
capacity. And the ×0.25 operating point is not the declared chip: it is the
buffer shrunk 4× to manufacture a binding constraint, which is the only way the
question can be asked at all on these designs.

#### The ceiling first: what capacity dilation could deliver at BEST, model-wide

Every result in this study prints its ceiling before its measurement, and for
Task 4 the ceiling is the refetch that exists to be removed. Layer-weighted over
all 21 resnet18 layers at declared capacity, EDP:

| design | DRAM weight reads | unique weights | refetch | **max cut** | vs Task 3's 0.152 |
|---|---:|---:|---:|---:|---:|
| `eyeriss_like` | 21,821,440 | 11,678,912 | 1.868 | **46.5 %** | **3.05×** |
| `eyeriss_v2_like` | 16,356,544 | 11,678,912 | 1.401 | **28.6 %** | **1.88×** |
| `simple_weight_stationary` | 12,518,144 | 11,678,912 | 1.072 | **6.7 %** | **0.44×** |

`max cut` is the fraction of DRAM weight reads that would vanish if **every**
layer were driven to refetch 1.0 — infinite weight capacity *and* perfect loop
ordering. A read never issued saves the whole `pJ_per_read` (efficiency 1.0)
against the `f_if × (1 − K/N)` = 0.152 a fixed mapping buys, which is what the
last column compares.

**On `simple_weight_stationary` the entire Task 4 prize at infinite capacity is
0.44× what Task 3 already delivers.** That design is one of the two
`ReconSweep.png` draws, and no amount of reconstruction-aware mapping can make
capacity dilation the better mechanism on it — its mappings already refetch only
1.072× model-wide. `eyeriss_like` is the design worth chasing at 3.05×, and its
headroom is concentrated in a few layers: `conv1` refetches **224×**,
`layer3.0.conv1` **14×** at 42.9 % weight-buffer fill (the highest fill of any
high-refetch layer, so the layer likeliest to bind), and `layer1.*` **16×**.
`layer2.0.conv1`, the layer both earlier passes used, is *not* the best test bed;
`layer4.*` holds most of the weights and already refetches 1.000, which is why
the model-weighted figure is so much lower than any single layer's.

Note also `simple_weight_stationary`'s `layer4.0.conv1`: `pe_spad` at
**384/384 = 100 %**, the weight buffer completely saturated, and refetch already
1.000. A full weight buffer is not sufficient for a dilation to pay — there has
to be refetch left to remove.

#### The completed sweep: 196 EDP mappings, five designs, and not one clean capacity win

All 196 jobs completed, none failed. Every design was swept in pairs
`(s, s x N/K)` with `PEs used` on every row.

| design | scope | clean capacity range swept | result |
|---|---|---|---|
| `eyeriss_like` | exclusive | ×0.5 – ×32 | **flat**, `held` = 21,504 throughout |
| `eyeriss_v2_like` | exclusive | ×0.125 – ×1.6154 | **flat**, `held` = 2,304, PEs 144 throughout |
| `eyeriss_v2_like_wglb` | exclusive | ×0.25 – ×6.4615 (**26×**) | **flat**, refetch 7.000, `held` = 3,072, PEs 192 throughout |
| `eyeriss_like_wglb` | exclusive | ×0.5 – ×16.15 | non-monotone; every non-flat row is `PE!=` |
| `simple_weight_stationary` | shared | ×0.25 – ×6.4615 | non-monotone reference; see below |
| all five | — | with `factors:` relaxed | **flat** |

**`eyeriss_v2_like_wglb` is the decisive negative.** It was the design predicted
to win: its `weight_noc` holds 65,536 weights = 88.9 % of the layer, sits ABOVE
the PE array where it can absorb a DRAM-level loop, and carries **no `factors:`
pin at all**, so the mapper is free to grow the tile to whatever capacity
allows. Across a **26× capacity range** with the PE count pinned at 192/192,
refetch is **7.000 at every single scale** and `held` is **3,072 at every single
scale**. At ×6.4615 the weight GLB alone has 423,936 weights of room and the
mapper puts 1,536 weights in it — **0.4 % fill**. The refetch is
`for P in [0:7)` at the DRAM level, and the mapper prefers to re-read weights
seven times rather than pay the activation and psum cost of keeping them.

**The `factors:` relaxation does not rescue it either, but it is the most
important result in the sweep for a different reason.** `ECC_WEIGHT_FACTOR_RELAX=1`
drops `M=1, S=1` from `weights_spad`'s temporal constraints. Every dilation pair
under it is still `flat` — `held` is identical in both arms at every scale, at
fills from 0.8 % to 57 %. But the *unrelaxed vs relaxed* comparison, at the SAME
declared capacity and the SAME 168/168 PEs, on `eyeriss_like` `layer3.0.conv1`:

| | refetch | DRAM weight reads | total energy |
|---|---:|---:|---:|
| declared dataflow | 14.000 | 4,128,768 | 417.17 µJ |
| `M=1, S=1` dropped | **2.000** | **589,824** | **266.60 µJ** |

**A 7× cut in DRAM weight traffic and 36 % of the layer's total energy, from
relaxing one dataflow constraint — where 61.9 % more capacity buys 0 %.** On
`layer2.0.conv1` the relaxation cuts refetch 8.000 → 2.000 and energy
220.32 → 182.77 µJ, but the PE count falls 168 → 96 there, so only the
`layer3.0.conv1` row is clean. This is not a reconstruction result: it is a
statement about Eyeriss v1's row-stationary constraint, and a design run under
`wrelax` is not the chip JSSC 2017 describes. It does say where the DRAM weight
traffic on these designs actually comes from, and it is not the buffer size.

#### `simple_weight_stationary` under `scope=shared` — suggestive, NOT established

`shared` finally dilates `operand_glb`, the 65,536-weight level above the
network, and it is the only place in the sweep where `capacity` verdicts appear
at all (2 of 8 pairs after the verdict bug below was fixed). But **the reference
arm is not converged**: its refetch across monotonically increasing capacity runs

    x0.25  x0.5   x0.75  x0.875  x1.0   x1.125  x2.0   x4.0
    4.000  1.000  4.000  1.000   2.000  2.000   2.000  1.000

The embedded arm reaches refetch **1.000 at ×0.5** and then **4.000 at ×0.75**
with *more* room. A quantity that is not monotone in capacity is not being set
by capacity, and the surviving `capacity` verdict at ×1.2115 (+6.52 %) is
measured against that ×0.75 anomaly. Across all 8 pairs the reconstruction arm
reached refetch 1.000 at 7 of 8 capacities against the embedded arm's 3 of 8 —
an asymmetry worth chasing, but it cannot be separated from search variance
until the reference converges.

**A seed-variance run cannot settle it, and the attempt is instructive.** Twelve
jobs were submitted at two extra `ECC_MAPPER_SEED` values; all three seeds
produced **byte-identical** `timeloop-mapper.stats.txt` (same md5) at every
capacity. env.sh says why, and it was there to be read first: *"timeloop-mapper
v4 exposes NO random seed, so this documents intent only."* The knob is in the
cache fingerprint but reaches nothing.

**This changes what "search noise" means here, and it is worth stating
carefully.** The mapper is **deterministic**: same architecture, same mapper
settings, same thread count → the same mapping, byte for byte. So the
non-monotonicity above is NOT run-to-run randomness that averaging would remove.
It is a deterministic but *chaotic sensitivity to the architecture*: changing a
buffer depth by a few entries sends `random_pruned` down a different path and it
lands somewhere else — reproducibly. The consequences:

* The results are reproducible. Re-running changes nothing.
* The 6 % "larger buffer refetches more" rate is a **real, repeatable property**
  of this search budget, not sampling noise.
* It cannot be averaged away with seeds. The only lever is a **stronger search**
  — raise `ECC_VICTORY` until the answer stops moving with capacity.

**That test has now run, and it REVERSES the result** (2026-09-10,
`simple_weight_stationary`, `scope=shared`, layer2.0.conv1):

| scale | victory 2000 | victory 10000 |
|---|---|---|
| ×0.5 (embedded reference) | rf 1.000, 273.2 µJ | rf 2.000, **251.0 µJ** |
| ×0.75 | rf 4.000, 280.4 µJ | rf 8.000, 254.2 µJ |
| ×0.8077 (reconstruction) | rf 1.000, 264.1 µJ | rf 8.000, **256.4 µJ** |
| ×1.2115 (reconstruction) | rf 1.000, 276.7 µJ | rf 2.000, 261.5 µJ |

At victory 2000 the reconstruction arm at ×0.8077 beat its ×0.5 embedded
reference by **9.1 µJ** — the single best result in the whole sweep, and the one
recommended for a YAML change. At victory 10000 the same pair **reverses**:
reconstruction is **5.4 µJ worse** and refetches 4× more. Energy falls at every
point, as a better search should, but the *ordering between the two arms flips*.

> **The −74 % `operand_glb` saving is WITHDRAWN.** It was the search failing to
> converge, not a capacity effect.

No number in this section taken at victory 2000 can be quoted without re-running
it at a converged budget, and convergence has to be treated as a **gate**: the
residual movement between victory settings must be smaller than the ECC effect
being claimed, or the instrument cannot resolve it. `prompt_2.md` carries the
gate procedure.

#### THE SEARCH IS NOT CONVERGED, measured against its OWN objective — and the gap is bigger than every ECC effect in the study

The capacity sweep is an unusually good convergence probe: it maps the *same
layer* many times at architectures that differ only in a buffer depth, so every
solution found at any capacity is a solution the search could in principle have
found at all of them. `eyeriss_like` `layer3.0.conv1`, all EDP, all 10 cached
capacities, sorted by PE count:

| ×cap | PEs | room | refetch | µJ | cycles | **EDP (µJ·Mcyc)** |
|---:|---:|---:|---:|---:|---:|---:|
| ×0.0625 | 64 | 1,792 | 2.000 | 134.42 | 903,168 | 121.4 |
| ×0.2019 | 84 | 7,560 | 2.000 | 170.10 | 688,128 | **117.0** |
| ×0.25 | 84 | 9,408 | 2.000 | 176.58 | 688,128 | 121.5 |
| ×0.5 | 84 | 18,816 | 2.000 | 168.40 | 688,128 | **115.9** |
| ×0.8077 | 84 | 30,408 | 2.000 | 180.80 | 688,128 | 124.4 |
| ×0.0505 | 168 | 3,696 | 14.000 | 476.00 | 344,064 | 163.8 |
| ×0.101 | 168 | 7,728 | 14.000 | 464.79 | 344,064 | 159.9 |
| ×0.125 | 168 | 9,408 | 14.000 | 422.38 | 344,064 | 145.3 |
| ×0.4038 | 168 | 30,240 | 14.000 | 413.83 | 344,064 | 142.4 |
| **×1.0 (declared)** | 168 | 75,264 | 14.000 | 417.17 | 344,064 | **143.5** |

Two clean facts. First, **capacity explains nothing**: `×0.125` and `×0.25` have
*identical* on-chip weight room (9,408) and differ 7× in refetch — 14.000 versus
2.000 — because one uses 168 PEs and the other 84. Within each PE cluster
refetch is exactly constant across a 40× range of room.

Second, and far more consequential for the whole study: **at the declared
capacity the mapper returned a mapping 24 % worse in its own objective than one
it found on the same layer at ×0.5** (EDP 143.5 vs 115.9). The 84-PE point is
not a different problem — it is the same layer, same workload, same objective,
and 2.5× lower energy at twice the cycles, which EDP prefers. The search simply
did not find it at ×1.0.

> **The mapper's shortfall against its own objective is up to 24 % on this
> layer. Every ECC saving this study reports is 2–12 %.** The noise floor of the
> instrument is larger than the signal it is being used to measure.

This is §2's "the mapper search decides the architecture ranking" with a number
on it, and it is why the 6 % monotonicity violation rate below matters more than
any individual pair. It bears on Task 1, 2 and 3 as much as on Task 4: those
numbers are differences taken *within* one cached mapping, so they are internally
consistent, but the mapping they sit on is not the optimum the objective asked
for, and a different capacity — or seed — moves it by more than the effect.

#### Monotonicity audit: refetch is not a function of capacity

Over every PE-matched pair in the completed sweep — 838 of them across 19
(design, layer, scope) cells:

> **52 of 838 PE-matched capacity pairs (6 %) have a LARGER weight buffer
> refetching MORE.** Under a capacity mechanism that must be 0 %.

And the violations are not spread evenly — they concentrate in exactly the three
cells that produced all four `capacity` verdicts in the whole sweep:

| cell | points | refetch range | larger-buffer-refetches-more |
|---|---:|---|---:|
| `simple_weight_stationary` layer2.0.conv1 (exclusive) | 12 | 1.000 – 4.000 | **45 %** |
| `eyeriss_v2_like` layer2.0.conv2 | 10 | 2.000 – 28.000 | **22 %** |
| `simple_weight_stationary` layer2.0.conv1 (shared) | 16 | 1.000 – 4.000 | **12 %** |
| `eyeriss_v2_like_wglb` layer2.0.conv1 | 16 | 7.000 – 7.000 | 0 % |
| every other cell | — | — | 0 % |

**All four `capacity` verdicts come from the three noisiest cells in the sweep,
and none from the fourteen quiet ones.** On `eyeriss_v2_like` layer2.0.conv2 the
`capacity` row is ×0.25 → ×0.4038, refetch 28.000 → 7.000 — but a buffer of
15 weights/PE, **five times smaller than the reference**, achieves 4.000 on the
same layer. That is not a capacity effect being detected; it is a bad reference
being escaped.

#### A fourth false positive, caught by the guard's own test — `capacity` did not require `held` to RISE

`capacity_verdict()` was documented and legended as "reads FELL **and** `held`
ROSE", but the branch only excluded the `held`-IDENTICAL case: a pair whose
reads fell while `held` **fell** still came back `capacity`. It hid because the
real-cache property test swept only `scope=exclusive`, where no such pair
existed. The shared-scope caches had four, e.g. `simple_weight_stationary`
×1 → ×1.6154: reads 147,456 → 73,728 while `held` fell 18,816 → 9,984 — the arm
stored strictly *less* on chip and read less, which extra room cannot explain.

Fixed: `capacity` now requires `held` to rise strictly, and the new `ORDER?`
verdict covers "reads fell, `held` fell". The property test now sweeps **both**
scopes and 12 reference scales — **37 real cached pairs** — and reverting the
fix makes it fail by name on that exact row.

#### The guards, and that they fail when broken

`eccenergy/tests/test_dilation.py`, 8 tests. `capacity_verdict()` labels a pair
`capacity` only when reads fall **and** `weights held` rises; `PERM?` when reads
move while held is identical at every weight level; `PE!=` when the arms differ
in PE count. Three deliberate mutations were run: removing the held check,
removing the PE check, and reading the fingerprint from the wrong directory —
each is caught, and removing the held check also trips
`test_no_pair_on_disk_is_reported_as_capacity_without_held_rising`, the property
test over the real cache, because the weight-stationary pair on disk is exactly
the one that would be misreported.


### 7.9 prompt_2's depth sweep on Eyeriss v1 — THE GATE FAILS, and `weights held` never rises

**Measured 2026-09-10.** `eyeriss_like_wglb` (= Eyeriss v1 since today),
resnet18 `layer3.0.conv1` (`C128_M256_R3_S3_P14_Q14_ws2_hs2`, 294,912 weights),
BCH(63,30), `edp`, `random_pruned`, 18 threads, victory 4000 (effective 8000 at
9 loop levels). 42 SLURM jobs, **all COMPLETED, none failed**, 24 min – 2 h 56 min
each. Read with `dilation --levels` / `--gate`; tables in
`results/tables/EyerissV1_mem_arch_sweep*.csv`.

The arms are the SAME silicon at two datawidths — Eyeriss v1's untouched
published geometry (`weights_spad` 224×16 b, `filter_glb` 1024×64 b), 8 b for
baseline/embedded and 4 b for recon, **exactly 2.000× effective capacity at
every depth**, byte-identical per-access read/write/leak. No width change was
applied or needed.

#### The convergence gate FAILS at both ends of the ladder

| depth | victory | PEs | refetch | total µJ | residual | effect claimed |
|---:|---:|---:|---:|---:|---:|---:|
| ×1 | 2,000 | 168/168 | 1.000 | 347.59 | — | −5.77 % |
| ×1 | 4,000 | 168/168 | 4.000 | 300.34 | **13.59 %** | −5.77 % |
| ×1 | 10,000 | 168/168 | 2.000 | 169.13 | **43.69 %** | −5.77 % |
| ×0.125 | 2,000 | 168/168 | 1.000 | 300.78 | — | −1.64 % |
| ×0.125 | 4,000 | 168/168 | 4.000 | 276.57 | **8.05 %** | −1.64 % |
| ×0.125 | 10,000 | 168/168 | 2.000 | 179.69 | **35.03 %** | −1.64 % |

> **Search noise is 6–21× the signal. NO number in this section is quotable**,
> and that is the finding, not a caveat attached to one.

It is worse than §7.8's precedent, not better: the minimum residual anywhere in
that chain was 9.04 %, and here the 4,000→10,000 step alone moves 43.69 %.
Refetch is also **non-monotone in budget** — 1.000 → 4.000 → 2.000 at both
depths — so the mapper's own answer about how often it reloads weights changes
identity twice as the search gets better. The victory-4000 mapping the sweep
runs at is **78 % worse in total energy** than the victory-10000 mapping of the
same silicon (300.34 vs 169.13 µJ at ×1).

#### Even taken at face value, there is no capacity effect at any depth

`weights held` is the column that decides whether anything happened, and across
the whole √2 ladder it does not move: **384 weights in `filter_glb` and 2,688
in `weights_spad` at ×0.7071, ×0.5, ×0.3536, ×0.25, ×0.1768 and ×0.125, on BOTH
arms.** 18 of 21 level-pairs come back `flat`; the other 3 are `reads-rose`.

| ×depth | spad depth | emb room | rec room | **held** | GLB fill | refetch e/r | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| ×1 | 224 | 75,264 | 150,528 | 2,688 / 10,752 | 4.7 % | 4.000 / **7.000** | **reads-rose** |
| ×0.7071 | 158 | 53,088 | 106,176 | **2,688** | 6.6 % | 4.000 / 4.000 | flat |
| ×0.5 | 112 | 37,632 | 75,264 | **2,688** | 9.4 % | 4.000 / 4.000 | flat |
| ×0.3536 | 79 | 26,544 | 53,088 | **2,688** | 13.3 % | 4.000 / 4.000 | flat |
| ×0.25 | 56 | 18,816 | 37,632 | **2,688** | 18.8 % | 4.000 / 4.000 | flat |
| ×0.1768 | 40 | 13,440 | 26,880 | **2,688** | 26.5 % | 4.000 / 4.000 | flat |
| ×0.125 | 28 | 9,408 | 18,816 | **2,688** | 37.5 % | 4.000 / 4.000 | flat |

The one row that moves goes the **wrong way**: at ×1 the reconstruction arm
refetches **7.000** against embedded's 4.000 and reads 2,064,384 weights against
1,179,648 — 75 % MORE DRAM traffic, at 168/168 PEs on both arms. Its
`filter_glb` held 12,288 against 384, so `held` rose 32× *and reads rose with
it*, which no capacity mechanism explains. That is the search, and the gate
above says so independently.

#### The margin is ENTIRELY the flat packing discount — the column split earns its keep

At every `flat` scale `refetch_term_pJ` is **exactly 0** and
`recon_minus_embedded_pJ` equals `−packing_discount_pJ` to the digit. DRAM
`level_pJ` is **identical between the arms** (75,497,472 pJ) at every flat
scale, because DRAM's datawidth is 8 on both and the loop nests match.

> Recon "wins" at every depth in the ladder, monotonically, purely from riding
> more weights per word on a **byte-identical loop nest**. Reported as one
> number this would have looked like a capacity result at every depth. Split
> into its two terms it is visibly a constant, and it says nothing whatever
> about which memory to shrink.

#### prompt_2's falsifiable prediction fails ON ITS PREMISE

The prediction was: the resident tile is 192 weights, embedded holds `2 × depth`,
so below depth 96 (×0.4286) the tile stops fitting and at ×0.25 (112 weights)
embedded cannot hold it while a 2.0× recon arm (224) can — so `held` must rise
and refetch fall.

**Measured, the resident tile is 16 weights per PE, not 192.** `weights_spad`
holds 2,688 over 168 instances at every scale, against 112/PE of room at ×0.25
— **14.3 % fill**. The tile is 12× smaller than the prediction assumed, so it
never comes close to binding and ×0.25 shows nothing because there is nothing
to show. `filter_glb` is no fuller: its peak fill anywhere in the ladder is
**37.5 %**, at the shallowest depth swept.

This is §7.8's mechanism again, unchanged by fixing the fairness problem:
refetch is set by loop ORDER at the DRAM level, and the weight buffers are not
the binding constraint at any depth a YAML would plausibly declare.

#### The second pass: neither level bought anything either

Sweeping each weight level with the other held at ×1 (`--depth-levels`), at
×0.5 / ×0.25 / ×0.125:

* **`filter_glb` alone** — `flat` at every scale. `held` 384, refetch 4.000,
  identical DRAM energy on both arms. FINDINGS §7.8 predicted this was the
  level that would matter, because it sits above the array and *can* absorb a
  DRAM-level `P`/`Q` loop. Confirmed present, measured inert.
* **`weights_spad` alone** — `reads-rose` at every scale: recon refetches 7.000
  against embedded's 4.000 and pays 56.6 µJ MORE at DRAM, identically at all
  three depths.

#### THE DELIVERABLE: there is no depth to recommend

prompt_2 asks for a `depth:` per weight level to write into a new arch YAML,
chosen where Recon's margin over Embedded peaks. **That number cannot be
extracted from this data**, for two independent reasons, either of which is
sufficient:

1. The gate fails by 6–21×, so no margin at any depth is resolvable.
2. Even ignoring that, the margin does not peak — it is the packing discount,
   which is present at every depth and is monotone in nothing but the datawidth.

Recommending a depth here would be recommending search noise. What the sweep
DOES establish is that the `datawidth` mechanism itself works exactly as
designed — 2.000× capacity at byte-identical per-access energy, verified at
seven depths — so the instrument is sound and the negative is about the
architecture, not the method.


### 7.10 The constrained mapspace — THE GATE PASSES, and the saving is in the PSUM path

**Measured 2026-09-10**, after §7.9's gate failure. `eyeriss_like_wglb`,
resnet18 `layer3.0.conv1`, BCH(63,30), `edp`, `linear_pruned`, 18 threads.
42 jobs, all COMPLETED, **35 s – 1 min 19 s each** (against 24 min – 2 h 56 min
unconstrained). Tables: `results/tables/EyerissV1_mem_arch_sweep{,__dataspace}.csv`.

Two changes make it work, and BOTH are needed:

* **`ECC_MAPSPACE_CONSTRAIN=1`** — pin every loop dimension to 1 at the levels
  `archs.MAPSPACE_FREE_LEVELS` does not name. 7.41e10 → **9.5e4** index
  factorizations, i.e. exhaustively searchable. Free-sets read off the best
  nest the search had ever found; both arms get the IDENTICAL constraint.
* **`ECC_WEIGHT_FACTOR_RELAX=1`** — drop `M=1` on the weight levels so the
  tile can grow. The constraint runs AFTER the relax and keeps C and M free at
  `weights_spad`, so the two compose instead of cancelling.

Plus the arch was reconfigured (2026-09-10): `psum_spad` 24 → **64**,
`weights_spad` 224 → **64**, `filter_glb` 1024 → **256**.

#### THE GATE PASSES — 0.00 % residual

| depth | v2000 | v4000 | v10000 | residual | effect |
|---:|---:|---:|---:|---:|---:|
| ×1 | 144.56 µJ | 144.56 | 144.56 | **0.00 %** | −14.63 % |
| ×0.125 | 273.73 µJ | 273.73 | 273.73 | **0.00 %** | −10.65 % |

Bit-identical at every budget: the space is exhausted, so each arm sits at its
TRUE optimum within the family and the difference is architectural rather than
sampled. **This is the first quotable ECC number in the study.**

#### `weights held` finally moves — recon holds exactly 2× at EVERY depth

`weights_spad` runs at **100 % fill** at ×1, ×0.5, ×0.25 and ×0.125 on both
arms; `filter_glb` at 75 %/53 %. Recon holds exactly 2× the weights at every
scale on both levels. PEs 168/168 and cycles 344,064 on both arms everywhere —
no parallelism or latency confound anywhere in the ladder.

#### BUT THE SAVING IS NOT IN WEIGHT REFETCH, AND STRUCTURALLY CANNOT BE

**Weight refetch is 1.000 on both arms at every depth** — its FLOOR. `C` is
the only DRAM-level loop and `C` indexes Weights, so every weight is fetched
exactly once and a smaller weight buffer cannot make it worse.

What a smaller weight buffer does is force `C` into more chunks. `C` does NOT
index Outputs, so every chunk is a **reduction step**: the partial sums for
the same outputs are written to DRAM and read back to accumulate, once per
chunk. That is where the entire effect lives:

| ×depth | C chunks e/r | Outputs Δ µJ | Weights Δ | Inputs Δ | **total margin** |
|---:|:---:|---:|---:|---:|---:|
| ×1 | 4 / 4 | 0.00 | −7.22 | −5.37 | −21.15 (−14.6 %) |
| ×0.7071 | **8 / 4** | −34.02 | −5.51 | 0.00 | −42.92 (−24.0 %) |
| ×0.5 | **8 / 4** | −34.30 | −4.24 | 0.00 | −41.92 (−23.8 %) |
| **×0.3536** | **16 / 8** | **−68.61** | −3.29 | 0.00 | **−78.65 (−31.5 %)** |
| **×0.25** | **16 / 8** | **−68.04** | −2.46 | 0.00 | **−77.24 (−31.3 %)** |
| ×0.1768 | 16 / 16 | 0.00 | −1.84 | −10.74 | −29.53 (−10.8 %) |
| ×0.125 | 16 / 16 | 0.00 | −1.45 | −10.74 | −29.15 (−10.6 %) |

**It is a staircase, and it has a genuine optimum.** Recon stays ONE chunking
step behind Embedded over ×0.71 → ×0.25; the µJ gap grows with the chunk
count, so it peaks at ×0.3536–×0.25. At ×1 both fit 4 chunks and there is no
gap; at ×0.1768 and below even 2× capacity cannot hold the step and both are
pinned at 16, so the margin collapses back to ~10 %.

> **THE REPORTING LESSON.** The weight-only table showed `refetch 1.000 → 1.000`
> and `level_pJ` halving exactly, at every depth — i.e. nothing but the flat 2×
> packing discount. It looked like a null result on a **31 % win**, because it
> was built to prompt_2's hypothesis that the effect would appear in WEIGHT
> refetch. `dataspace_table()` exists so that can never happen again: it reports
> reduction factor, refetch and energy per DATASPACE, and `--levels` always
> prints it.

#### The recommended depths

**×0.25 → `filter_glb: depth 64`, `weights_spad: depth 16`.** Margin −31.3 %,
gate passes, 168/168 PEs, weight scratchpad at 100 % fill. ×0.3536 is
marginally better (−31.5 %) but ×0.25 is the factor-2 point a YAML quotes.

**Two caveats travel with every number here.** This is the CONSTRAINED +
RELAXED dataflow with a reconfigured psum/weight geometry — **not** the chip
JSSC 2017 describes, and `source: published` does not licence its name. And
the margin is conditional on that mapspace family: both arms were searched
exhaustively WITHIN it, not outside it, so a negative under it would be weak
evidence while this positive is trustworthy.


### 7.11 The baseline's DRAM cost is a PRICE, not extra traffic — corrected 2026-09-10

**Measured 2026-09-10** on the cached record for `eyeriss_like_wglb`, resnet18
`layer3.0.conv1`, BCH(63,30), `ECC_DRAM_PJ_PER_BIT=40`,
`ECC_MAC_PJ_OVERRIDE=0.23`
(`results/_raw/.../mcons__wrelax/fp-48347c8b8194/.../resnet18.json`).

The decoder is on the DRAM die, so the conventional arm's parity is read,
corrected and discarded there: it never crosses the datapath. All three arms
therefore issue the SAME 294,912 DRAM weight reads and drive the same 8 bits per
weight off the die. What the baseline pays for is a bigger, indexed array —
6,193,152 stored bits against 2,359,296 of payload, **2.625×** — priced as
`ECC_BASELINE_DRAM_PJ_PER_BIT` = **70 pJ/bit against 40**, i.e. ×1.75 on the
whole DRAM category. Reconstruction remains the only arm whose *traffic* scales.

| | before (traffic model) | after (price model) | delta |
|---|---:|---:|---:|
| baseline DRAM category | 241.213 µJ | **422.124 µJ** | ×1.75 |
| baseline external-parity term | 153.354 µJ | **0** | −153.354 |
| **baseline total** | **495.389 µJ** | **522.945 µJ** | **+27.556 (+5.56 %)** |
| embedded total | 342.035 µJ | 342.035 µJ | **0** |
| recon total (sweep bar) | 331.231 µJ | 331.231 µJ | **0** |
| embedded saving vs baseline | 30.96 % | **34.59 %** | +3.63 pp |
| recon saving vs baseline | 33.14 % | **36.66 %** | +3.52 pp |

The two terms are NOT the same size: the traffic term deleted was 153.354 µJ and
the price added is 180.910 µJ, so the corrected model charges the baseline
**more**, not less, and every ECC saving quoted against it grows. Only the
baseline arm moved — embedded and recon are bit-identical, which is what makes
the delta attributable.

Mechanically: the whole DRAM category is rescaled (weights, inputs and outputs
alike — a per-bit cost is a property of the device, not of a dataspace), the
rescale is evaluator-side after the raw cache, and the knob is deliberately
absent from `Config.fingerprint()`, so **no mapping moved and baseline still
shares embedded's mapper cache**. The DRAM `depth:` remains ONE declared value
for all three arms; the array size is priced, not mapped. `parity.py` and
`ecc.external_parity()` were not edited — the parity accounting still runs on
every result as the array-SIZE evidence the price is charged for, and its hand
check still passes. With `ECC_BASELINE_DRAM_PJ_PER_BIT` empty the whole
pre-2026-09-10 model comes back **bit-for-bit** (verified: baseline components,
total and all three sweep-figure columns identical to the digit), and the result
carries the old check name, so the two models cannot be confused in a diff.

`eccenergy/baseline_dram.py`; 18 assertions in
`eccenergy/tests/test_baseline_dram.py`, four of them deliberate breakages
(price the weight rows only; charge the parity as well; a stale ratio; a
hardcoded 70/40 against a record priced at 8 pJ/bit) — each must FAIL the check
that is supposed to catch it.

E_background and E_refresh grow with the bigger array too. Both are still 0 and
unmodelled, so this is the whole of the baseline's penalty for now.


### 7.12 `weights_spad` depth 16, one layer — THE GATE PASSES and the margin is 22.4 % on the mapping, 40.7 % with the DRAM term

**Measured 2026-09-10** (prompt_5.md). `eyeriss_like_wglb`, resnet18
`layer3.0.conv1`, BCH(63,30) -> recon datawidth **4b**, `edp`, `linear_pruned`,
victory 4000, 18 threads, `ECC_RECON_PACKING=aligned`. The YAML as declared:
`weights_spad` **depth 16**, `filter_glb` depth 256, **no depth scaling**
(`ECC_DEPTH_SWEEP_SCALES=1.0`). 4 jobs (2 sweep + 2 gate), all COMPLETED,
**54 s - 1 min 05 s**. Tables: `results/tables/EyerissV1_spad16{,__dataspace}.csv`.
Mapper caches `fp-3cd00eb16801` (8-bit arm) and `fp-7b80cd0c9e2a` (4-bit arm).

**THE GATE PASSES.** Embedded arm at victory 2000 / 4000 / 10000: total
**179.22 uJ at all three**, residual **0.00 %** against a 40.07 uJ effect. The
constrained mapspace is searched exhaustively, so each arm sits at its true
optimum and the margin is not search noise.

| level | depth | emb held / room (fill) | rec held / room (fill) | refetch e->r | level pJ e->r |
|---|---:|---|---|---|---|
| `filter_glb` | 256 | 768 / 2,048 (37.5 %) | 1,536 / 4,096 (37.5 %) | 1.000 -> 1.000 | 0.26 M -> 0.13 M |
| `weights_spad` | 16 | 5,376 / 5,376 (**100 %**) | 10,752 / 10,752 (**100 %**) | 1.000 -> 1.000 | 4.53 M -> 2.27 M |

Both arms fill `weights_spad` completely, so capacity binds on both — and the
4-bit arm holds **2x the weights in the same array** (4 per 16-bit word against
2). DRAM weight reads are **294,912 on both arms, refetch 1.000**: the weight
floor, exactly as predicted, and NOT a null result.

**The margin is in the PARTIAL-SUM path**, per dataspace (whole datapath, at
Accelergy's own 8 pJ/bit, MAC 0.23):

| dataspace | x-reduce e->r | refetch e->r | emb | recon | delta |
|---|---|---|---:|---:|---:|
| Weights | 1 -> 1 | 1.000 -> 1.000 | 23.66 | 21.27 | -2.39 (-10.1 %) |
| Inputs | 1 -> 1 | 1.000 -> 1.000 | 19.44 | 19.44 | 0.00 |
| **Outputs** | **8 -> 4** | **15.000 -> 7.000** | 85.71 | 51.40 | **-34.30 (-40.0 %)** |
| TOTAL | | | **179.22** | **139.15** | **-40.07 (-22.4 %)** |

The extra weight capacity **halves the DRAM-level `C` chunking, 8 chunks -> 4**,
and `C` does not index Outputs, so each chunk removed is a psum round-trip to
DRAM that never happens. 86 % of the margin is that; the flat packing discount
on the weight rows is 2.39 uJ, 6 %.

**At the study's pricing** — DRAM 40 pJ/bit, baseline 70 (7.11), MAC 0.23,
production path so the NoC post terms are in (they are arm-dependent: +8.91 uJ
embedded, +4.45 recon, which is why they cannot be added as a constant):

| arm | total uJ | vs baseline | vs embedded |
|---|---:|---:|---:|
| baseline (DRAM @ 70) | 761.102 | — | — |
| embedded (DRAM @ 40) | 483.854 | **36.43 %** | — |
| recon, mapping only | 336.571 | 55.78 % | 30.44 % |
| **recon + DRAM x K/N** | **287.138** | **62.27 %** | **40.66 %** |

DRAM category alone, and the two recon mechanisms kept separable:

| arm | DRAM uJ | vs baseline | vs embedded |
|---|---:|---:|---:|
| baseline | 646.912 | — | — |
| embedded | 369.664 | 277.248 (42.86 %) | — |
| recon, mapping only | 241.213 | | 128.451 (34.75 %) — psum round-trips |
| **recon + DRAM x K/N** | **191.781** | **455.131 (70.35 %)** | **177.883 (48.12 %)** |

Recon's 177.883 uJ of DRAM saving against embedded is **128.451 from the
mapping** (chunks removed) **plus 49.433 from the K/N fetch** (94.372 -> 44.939
on the weight term; the decoder is on the DRAM die, so only the k message bits
are driven off it). Neither term is the other, and neither is quoted alone.
DRAM is 85.0 / 76.4 / 66.8 % of the baseline / embedded / recon run at this
pricing.

**Caveats that travel with these numbers.** ONE LAYER (2.53 % of resnet18) on
the **constrained + relaxed** dataflow with a reconfigured psum/weight geometry
— NOT the chip JSSC 2017 describes, so never labelled "Eyeriss v1" unqualified.
Both arms get the identical constraint, so the comparison is fair, but the
margin is conditional on that mapspace family. `--levels` also warns that three
solved fingerprints sit under the 8-bit variant slug; the current configuration
reads `fp-3cd00eb16801` and the other two are a different architecture.

**The recon total came from the mapping-aggregation path, not `recon --eval`.**
Task 4's production path (`RECON_OPTIMIZER=True ECC_PHASE=Post`) **refuses** this
cache and is right to: it wants the reconstruction arm expressed as
`ECC_WEIGHT_CAPACITY_SCALE=2.1` (`wcap2.1`), while prompt_5's sweep expresses it
as `datawidth 4` on the declared depth — deliberately, because `depth x N/K` made
Accelergy price the recon array 1.18-1.46x dearer and is what made 7.8 prove
nothing. The 336.571 uJ above is that same 4-bit mapping aggregated by the
production evaluator (`ECC_WEIGHT_DATAWIDTH=4`, throwaway results dir), so it
carries the NoC post terms; the K/N DRAM term is then applied as `recon.py`
applies it. Teaching `recon.dilated_view()` to accept a datawidth-expressed
recon cache is the open item, and it is a code change nobody has authorised.


### 7.13 The BCH sweep on the mapping study — the capacity mechanism is a 2x STEP, and only BCH(63,30) reaches it

**Measured 2026-09-11.** `eyeriss_like_wglb`, resnet18 `layer3.0.conv1`, `edp`,
`linear_pruned`, victory 4000, 18 threads, `ECC_RECON_PACKING=aligned`,
constrained mapspace + weight relax, **no depth scaling**. Four codes, each a
PAIR of mapper jobs on identical declared silicon: 12 new jobs (3 x 2 sweep +
3 x 2 gate), 39 s - 1 min 18 s each. Table:
`results/tables/EyerissV1_bch_sweep_spad16.csv`.

**THE GATE PASSES ON ALL THREE NEW CODES** at residual **0.00 %** across
victory 2000 / 4000 / 10000, against effects of 0.21 / 0.77 / 76.03 uJ.

| code | q | spad W / GLB W | capacity | emb held / room | rec held / room | DRAM `C` chunks e->r |
|---|---:|---|---:|---|---|---|
| BCH(63,57) | 7 | 56 / 224 | 1.1429x | 6,144 / 7,924 (77.5 %) | 6,144 / 9,056 (67.8 %) | 8 -> 8 |
| BCH(63,45) | 6 | 24 / 96 | 1.3333x | 6,144 / 7,596 (80.9 %) | 6,144 / 10,128 (60.7 %) | 8 -> 8 |
| BCH(63,39) | 5 | 40 / 160 | 1.6000x | **3,072** / 7,080 (43.4 %) | 6,144 / 11,328 (54.2 %) | **16** -> 8 |
| BCH(63,30) | 4 | published | 2.0000x | 6,144 / 7,424 (82.8 %) | **12,288** / 14,848 (82.8 %) | 8 -> **4** |

**THE RESULT, and it is negative for every code but the strongest.** The
mechanism §7.12 identified — extra weight capacity removes DRAM-level `C`
chunks, and each chunk removed is a partial-sum round trip — moves in **powers
of two**, because the resident tile is an integer and a level's share of `C` can
only double. `weights_held` is **6,144 on both arms** at q=7 and q=6: the extra
room is simply not spent (fill FALLS, 77.5 -> 67.8 % and 80.9 -> 60.7 %), which
is §7.8's `PERM?`/no-capacity signature, not a win. Only BCH(63,30)'s 2.000x
reaches the next step and halves `C` 8 -> 4.

| code | baseline uJ | embedded uJ | recon, map only | recon, +DRAM K/N | rec vs emb, map | rec vs emb, +DRAM |
|---|---:|---:|---:|---:|---:|---:|
| BCH(63,57) | 757.985 | 480.737 | 480.529 | 471.541 | **0.04 %** | 1.91 % |
| BCH(63,45) | 759.409 | 482.161 | 481.387 | 454.424 | **0.16 %** | 5.75 % |
| BCH(63,39) | 1240.578 | 770.654 | 480.197 | 444.246 | *37.69 %, see below* | *42.35 %* |
| BCH(63,30) | 761.102 | 483.854 | 336.571 | 287.138 | **30.44 %** | **40.66 %** |

At q=7 and q=6 the mapping margin is the flat packing discount alone — 0.21 and
0.77 uJ on the weight rows, Inputs and Outputs bit-identical between the arms —
and **the whole usable saving is the DRAM K/N fetch term**: 8.988 uJ (2.43 % of
DRAM) and 26.963 uJ (7.29 %), against BCH(63,30)'s 49.433 uJ (and 128.451 uJ of
mapping saving on top).

**BCH(63,39)'s 37.69 % IS NOT A RECONSTRUCTION SAVING — it is the reference arm
falling off a tiling cliff.** Reconstruction's own total is FLAT across all
three of the new codes (480.529 / 481.387 / 480.197 uJ); what moves is the
8-bit arm (480.737 / 482.161 / **770.654**). The cause is the declared width,
not the code: on the depth-16 scratchpad the width table's depth
renormalisation leaves the 8-bit reference holding **35 / 33 / 30 weights per
PE** at q=7 / q=6 / q=5, and at 30 it can no longer hold the 6,144-weight tile,
halves it to 3,072 and pays `C=16` where every other arm in the study pays
`C=8`. Do not quote the -37.69 %.

**THE FIX IS A BIGGER SCRATCHPAD, not a different rounding rule.** The
reshape's error is set by integer depth and shrinks as the array grows. At the
depth-16 spad (256 bits) it is -6.25 % to +9.38 % and the reference moves 17 %
in capacity; at Eyeriss v1's **published 224 x 16 b** spad (3,584 bits) the same
widths give depths 64 / 149 / 90, errors 0.00 / -0.22 / +0.45 %, and the 8-bit
reference holds **447-450 weights/PE at every code** — a 0.7 % spread. A BCH
sweep whose baseline and embedded bars are one number needs that geometry.

**Two derivations of `q` disagreed, and one of them was reading the wrong arm.**
`recon.Packing.reduced_bits_per_weight` uses `ceil(8*K/N)`; prompt_2's "q
declared" column, THE WIDTH TABLE and `hpc/map_depth_sweep.sh` use `round`. They
agree everywhere except **BCH(63,57)** (ceil 8, round 7) and **BCH(63,51)**
(ceil 7, round 6). `dilation --gate/--levels` took its default from the `ceil`
one, so the first BCH(63,57) gate loaded the **8-bit arm as "recon"**, reported
an effect of exactly 0.00 uJ and FAILED itself. Fixed by pointing dilation's
default at `code_widths.declared_datawidth()`; no cached number moved, because
the two agree at every code previously run. **`recon.py`'s `ceil` is NOT
changed** — it is Task 3's physical packing model and changing it would move
every Task 3 bar — but it means Task 3 models BCH(63,57) and BCH(63,51) with
strictly less on-chip narrowing than the mapping study does. Open item, §8.

**Baseline and embedded are K-INDEPENDENT in the model and K-dependent in this
run, and the difference is silicon, not code.** Nothing in the evaluator makes
them vary with K: they share one 8-bit mapping, the baseline's ECC cost is the
flat `ECC_BASELINE_DRAM_PJ_PER_BIT=70` price (`baseline_dram.py`, the parity
traffic explicitly NOT charged), and the one K-dependent term in
`build_stacks()` — `ECC decode`, `n_cw x decode_pJ` — is zero under
`ECC_DECODE=0`. They differ across the rows above only because each code
declares a different word width, which is different silicon. At the published
geometry they would be one number.


## 8. Open defects and limitations

1. ~~**`noc.yaml` declares `eyeriss_v2_like`'s intra-cluster `PE: {router_pj: 0.0}`
   but `_inject_noc` emits `router_energy` only `if router:`**~~ — **fixed in code
   2026-09-08 (§4.1, change E), unmeasured until the next cold pass.** Size on the
   2026-09-07 cache: 44.94 µJ, 20.5% of the whole NoC, 0.94% of the run. It
   affected every bar equally, so it did not bias Task 3.
2. **The band check's denominator** — `audit.py` now (2026-09-08) reads the band
   against on-chip energy and reports both shares; Fig. 18 is a gate-level
   breakdown of the chip, so that is the paper's denominator. On the 2026-09-07
   cache the on-chip share was `219.718 / (4798.375 − 1433.441)` = **6.53%,
   inside the published band**. `baseline.py` (frozen) still reports the
   DRAM-inclusive share.
3. **The memory-controller-to-accelerator link is not modelled at all.** *(Note
   2026-09-10: this entry previously carried a "superseded by §7.2, the DRAM term
   is now split by `f_if`" caveat. That caveat is itself dead — `f_if` was REMOVED
   2026-09-10 and the WHOLE DRAM term now scales by K/N. See prompt_1.md.)*
   Accelergy's `CactiDRAM` is a flat per-bit LPDDR4 constant using only
   `(type, width)`, with no array/IO decomposition to peel off, and Timeloop
   charges the DRAM↔chip network zero, so the link cannot be exposed. **No
   boundary in this study would claim it anyway** — correcting at the memory
   controller is a different partitioning assumption and a constant offset on
   every bar. Task 5 material.
4. **The weight NoC is not under-counted in any way that matters.** Weight
   ingresses at the mesh equal the DRAM weight reads exactly (13,849,792 — the
   floor), with multicast factors of 1, 2 or 4 across the 12 shapes, so there is
   no scatter error; and the weight share of the NoC is 0.79% of the run, so
   scaling the whole NoC to the top of the published band moves the Task 3 ceiling
   only from 1.616% to 1.711%. The NoC is not why the placement figure looks flat.
5. **`eyeriss_like_wglb` is unmapped, and it is now the Eyeriss v1 baseline.**
   DECIDED 2026-09-10 (prompt_2.md): the `_wglb` file is the correct Eyeriss v1 —
   JSSC 2017 publishes the 8 kB filter GLB — and `eyeriss_like` (which declares
   `!Nothing` in its place) is retired. **This ends the bracketing-pair doctrine
   for v1**; `BRACKET_PAIRS` in `config.py` must be retired with it. Blocking:
   `eyeriss_like_wglb` has NO entry in `recon.py`'s `WEIGHT_PATHS` or
   `PLACEMENTS`, and `weight_path()` refuses when a weight-carrying level goes
   unclaimed — so `filter_glb` needs a GLB `Stage` and a boundary before anything
   evaluates. Every fingerprint changes, so the whole `eyeriss_like` cache
   (64 variant dirs) goes cold. Budget **2.5–4 h wall / 45–70 core-hours for
   resnet18 alone**, a lower bound since it adds a storage level and `levels`
   scaling raises the search budget with nest depth. The asymmetry survives for
   v2 only: `eyeriss_v2_like_wglb`'s extra weight level is NOT in its paper.
6. **Register writes were free in every design** until 2026-09-08 (Aladdin
   `aladdin_register.csv`: write 0 pJ). Fixed in `regfile_decoded.yaml` /
   `register_rw.yaml` (§4.1); the size of the correction is unmeasured until the
   cold pass. Expect the `Local (spads/RF)` bar to rise on every design, most on
   output-stationary (an accumulator write per MAC).
7. **Neurosim plug-in returns 0 pJ inside the Apptainer image, and Accelergy
   prefers it.** Every `timeloop-mapper.accelergy.log` in the converged cache
   (and in the 2026-09-08 smoke run) has
   `Neurosim Plug-In ... OSError: [Errno 30] Read-only file system:
   .../accelergy-neurosim-plugin/neurosim_input_<pid>.cfg` followed by
   `Neurosim Plug-In estimated 0p with accuracy 70%`, and the ERT summaries show
   `estimator: Neurosim Plug-In` on some of the scratchpads' `address_generators`
   (Aladdin, which does return a number, is chosen for the others). Those
   intadders therefore cost **0 pJ**: v2's `psum_spad` write = update = read
   exactly (0.13263), i.e. no address-generation term, where `ifmap_spad` pays
   0.0328 pJ for the same 5-bit add from the Aladdin table. Small per access
   (~25% of a spad access on the affected levels), on every design, since the
   image was built. Fix is in `hpc/tl.sh`: bind a writable copy of the
   plug-in directory (as is already done for CACTI scratch), then re-verify
   that Neurosim's non-zero estimate is the one wanted over Aladdin's — or
   disable the plug-in so every adder comes from the same table. **Not
   changed in this revision**; it belongs with the cold pass.

6. **Six models unmapped**: resnet50, densenet121, squeezenet1_1,
   efficientnet_b0, convnext_tiny, xception — 221 shapes × 6 architectures, ≈5×
   what has been run, and the three with depthwise layers have no reusable cache.
7. **CSC is not modelled** — the published v2 PE stores weights CSC-compressed
   with a separate address scratchpad, so R4a/R4b's dense scratchpad saving is an
   upper bound. Warned on every result file.
8. **Fixed mapping everywhere**: no capacity-driven reuse or re-tiling is credited
   to any arm (Task 4's question). Direction matters — R4b already needs 1.81× the
   PE weight storage, so a reconstruction-aware mapper has less headroom than the
   reduced scratchpad alone suggests.
9. `simba_like` is the exercises' reference design, not Simba;
   `simple_input_stationary` reproduces no paper.
10. **`build_stacks()` counts the embedded arm's decodes at the baseline's 6
    weights/codeword.** Decode is off by default and outside Task 2; when codec
    energy is adopted it must move to the 7.875-weight layout, recorded.
11. **`eyeriss_v2_like` fidelity gaps**, all intended: no CSC, no NoC mode
    switching, 45 nm / 1 ns rather than 65 nm / 200 MHz, iact spad 24×8 b rather
    than 16×12 b (Timeloop needs `width % datawidth == 0`; same bits, narrower
    words). The paper's "v2-dense = 1.9× v1" claim is **not** reproduced — the two
    are within 1.2% whole-model — plausibly because the model omits what separates
    them (v1's flat multicast vs v2's hierarchical mesh) and because the paper's
    benchmark is MobileNet v1, where utilisation, not energy per access, is what
    v2 fixed.

#### The convergence gate FAILS — no affordable budget converges, and the residual exceeds the effect

**Measured 2026-09-10.** 24 mapping jobs, `simple_weight_stationary`,
`layer2.0.conv1`, `scope=shared`, `ECC_OPT_METRIC=edp`, 18 threads, capacities
{0.5, 0.75, 0.8077, 1.2115}. All completed; none was cancelled.

> **Cost-model drift, caught before it corrupted the comparison.** `dilation`
> now reports 198.70/201.90/204.01/209.19 µJ at victory 10000 where §7.8's
> 2026-09-09 pass recorded 251.0/254.2/256.4/261.5 — a **constant −52.3 µJ**
> from today's `f_if` removal. Refetch factors and fingerprints are identical,
> so the *mappings* are unchanged and only the evaluator moved. Every µJ below
> is re-read under the **current** model; mixing the two would repeat the
> cross-fingerprint error this section already documents.

| victory | wall/map | ×0.5 | ×0.75 | ×0.8077 | ×1.2115 | max residual |
|---:|---:|---:|---:|---:|---:|---:|
| 2 000 | 0.15 h | 220.87 | 228.08 | 211.74 | 224.35 | — |
| 5 000 | 0.69 h | 218.44 | 201.90 | 204.01 | 209.19 | 11.48 % |
| 10 000 | 1.61 h | 198.70 | 201.90 | 204.01 | 209.19 | 9.04 % |
| 20 000 | 2.77 h | 198.70 | 162.88 | 164.17 | 174.97 | **19.53 %** |
| 50 000 | 7.20 h | 177.40 | 162.88 | 164.17 | 163.51 | 10.72 % |

**The minimum residual anywhere in that chain is 9.04 %, and it does not shrink
with budget.** The gate requires < 2 %. The ECC effect this study claims is
2–12 %. **The search noise is larger than the signal**, so no ECC conclusion
survives at any of these budgets — the rule that "the residual must be smaller
than the ECC effect being claimed" is not met, and cannot be met by spending
more.

#### The conclusion inverts with the budget

Which arm shows a capacity win changes *identity*:

| victory | ×0.8077 | ×1.2115 |
|---:|---|---|
| 2 000 | flat +0.81 % | **capacity +6.99 %** |
| 5 000 | reads-rose −2.06 % | **capacity +15.80 %** |
| 10 000 | reads-rose −7.01 % | **capacity +15.80 %** |
| 20 000 | **capacity +3.28 %** | flat +1.10 % |
| 50 000 | **capacity +16.97 %** | flat +1.10 % |

At victory 10000 the win is ×1.2115 at +15.80 %; at 20000 that arm is *flat* and
the win has moved to ×0.8077 — whose own claimed effect then runs +3.28 % →
+16.97 % between 20000 and 50000. **The search budget, not the silicon, decides
the answer.**

#### A two-point agreement test is unsound — and this file used to recommend one

env.sh advised "run at two values and check the totals do not move".
`random_pruned` is **deterministic** (fixed thread count ⇒ same sequence), so a
larger budget walks the *same* sequence further: it **plateaus, then jumps**. On
×0.5, victory 10000 and 20000 return a **bit-identical 198.70 µJ** — "converged"
by that test — and 50000 then moves it to 177.40 (10.7 %). Only a bound on the
**unsearched** mapspace proves anything. The advice has been removed from env.sh.

#### Why no budget can work: it is a MAPSPACE problem

The mapper reports its own mapspace for this one layer:

```
Mapspace Dimension [IndexFactorization] Size: 74,095,560,000
Mapspace Dimension [LoopPermutation]   Size:  8,957,952,000     -> ~6.6e20
```

At a **measured 400,000 valid mappings per thread per hour**, coverage of *one*
thread's 4.12e9-factorization subspace is 0.0067 % at victory 5000, 0.0156 % at
10000 and **0.0700 % at 50000** — ignoring the permutation dimension entirely.
Single coverage of one thread's factorizations needs ~1500× victory-50000.

Runtime is **not** linear (env.sh's old claim) and **not** a single power law:
super-linear early, roughly linear past 10000 — steps of 4.6×, 2.3×, 1.7×, 2.5×
for budget steps of 2.5×, 2×, 2×, 2.5×. A fit to the 2000→10000 points
(t ∝ victory^1.47) predicts 17 h/map at 50000; measured **7.20 h**. The planned
"50000 does not complete in 8 h/map" fallback would therefore have been **false**
— per-arm walls were 6.03/7.08/7.57/8.14 h, expensive but practical.

#### Mapper knobs, measured rather than assumed

| algorithm @ victory 10000 | mean wall/map | mean pJ/MAC |
|---|---:|---:|
| **`random_pruned`** | 1.61 h | **4.425** |
| `hybrid` | 2.22 h | 7.211 |
| `linear_pruned` (timeout 1e8) | 0.12 h | 6.539 |

`random_pruned` wins on every arm; env.sh's choice is vindicated. `hybrid` walks
every pruned permutation of *one* factorization before moving on, and with 9.0e9
permutations available it barely advances through factorizations at all — so
despite §7.8's finding that refetch is a loop-**order** property, the
permutation-heavy algorithm is the wrong one. `linear_pruned` sticks in a biased
prefix.

**`search_size` is the fairer budget knob for an A/B ablation.** Wall-time spread
across the four arms of the same layer: victory 2000 **3.28×**, victory 10000
**1.57×**, victory 50000 **1.35×**, `search_size` 20000 **1.13×**. `victory` is
an *adaptive* budget — every improvement resets the counter, so the arm that
keeps getting lucky is searched **longer**, and unequal search effort is
confounded with the hardware difference under test. At the measured throughput,
`search_size=276000` ≈ victory 5000 and `search_size=644000` ≈ victory 10000.

Two Timeloop defects worth recording:

1. **`timeout: 0` does not disable the criterion — it makes it fire immediately.**
   `doc/mapper.md` says invalid mappings are then "ignored"; `mapper-thread.cpp`
   guards on the **counter**, not the setting:
   ```cpp
   if ((invalid_mappings_mapcnstr + invalid_mappings_eval) > 0 &&
       (invalid_mappings_mapcnstr + invalid_mappings_eval) >= timeout_)
   ```
   `search_size_` and `victory_condition_` both guard with `X_ > 0 &&`; this one
   does not. Measured: all 18 threads quit with "0 invalid mappings …", job
   failed in 16 s.
2. **`ECC_MAPPER_TIMEOUT` is algorithm-dependent.** Under `random_pruned` it
   never fires (every thread terminates by victory). Under `linear_pruned` at the
   default 2000 it is fatal: the linear walk begins where **100 % of the first
   36,000 mappings are infeasible** (~44 % fanout, ~56 % capacity), so all 18
   threads quit before finding one valid mapping and the job dies in 10 s.

Also unset: `sync_interval` is null and the sync is guarded by
`sync_interval_ > 0` (`mapper-thread.cpp:489`), so the 18 threads **never share a
best-so-far**. The reported mapping is the min over 18 *independent* searches,
each stopping against its own local best — a candidate explanation for the
deterministic-but-chaotic sensitivity recorded 2026-09-09. And two knobs
pytimeloop exposes that env.sh does not:
`max_temporal_loops_in_a_mapping: -1` (enforced for every algorithm,
`mapper-thread.cpp:551-559` — a direct mapspace lever) and
`filter_revisits: false` (**dead config** here: only `hybrid.cpp` and
`random.cpp` read it).

#### What to do instead: shrink the mapspace, don't grow the budget

The 7.4e10 factorizations come from 6 eligible temporal levels per dimension
(C=462, M=792, R=6, S=6, P=75, Q=75 options). Constraining the loop nest
collapses it; exhaustive cost is (factorizations × 16 permutations) / 18 threads
/ 400 k per thread-hour:

| constraint | factorizations | exhaustive h/map |
|---|---:|---:|
| unconstrained (today) | 7.41e10 | 164,657 |
| pin R,S to one level | 2.06e9 | 4,574 |
| + P,Q across ≤ 2 levels | 1.32e7 | 29 |
| **+ C,M across ≤ 3 levels** | **3.63e4** | **0.08** |

The last tier is **exact** and ~8× cheaper than victory-5000's 9–11 %-wrong
answer — run it with `linear_pruned` and `ECC_MAPPER_TIMEOUT=100000000`. **A
smaller exactly-solved problem beats a larger approximately-solved one for an
ablation.** This is the only path that makes the gate passable.

**Caveat.** One design (weight-stationary), one layer, and the old depth-scaling
geometry. Convergence budget depends on mapspace size, so none of these numbers
transfer to Eyeriss v1 with the widened 256/1024-bit word geometry the next phase
uses. Treat them as an order-of-magnitude starting point and **re-run the gate on
the real configuration — at the smallest depth in the sweep as well as the
largest**, since a budget that converges on a big buffer may not on a small one.


### 8.x Resolving a mapper-cache path used to CREATE it — fixed in code, residue left on disk

`paths.Results.mapper_cache()` ended with an unconditional
`d.mkdir(parents=True, exist_ok=True)`, so merely *asking* where a mapping would
live created its directory. `experiments/dilation.py` maps nothing and only
reads, but it enumerates (design × capacity) pairs — and a test that swept 5
designs × 9 capacities therefore left **89 empty `<slug>/fp-<hash>/`
directories** behind, advertising capacities that were never mapped. An `ls` of
`eyeriss_like_wglb` showed 17 `wcap` variants where only a few had been run.

**Fixed**: `mapper_cache(..., create=False)` for readers, and `dilation.cache_for()`
passes it. Anything that WRITES a mapping keeps the default.

**The 89 directories are still on disk and are deliberately not deleted.** They
are verified to contain nothing at all — no `timeloop-mapper.stats.txt`, no
`.lock`, no sidecar — so nothing reads them as a mapping: `map_capacity_sweep.sh
--progress` counts solved stats files and locks, and
`dilation.sibling_fingerprints()` only returns a fingerprint that has solved a
shape. CLAUDE.md's rule is that the mapper cache is never deleted, and 175 jobs
were writing into that tree when this was found. Sweep them with
`find ecc_energy_study/outputs -mindepth 3 -maxdepth 3 -type d -name 'fp-*' -empty -delete`
when the queue is idle, if the tidiness is wanted.

## 9. Next, in order

**The live plan is `prompt_2.md`.** It supersedes this list wherever they differ;
this section is the short form and the ordering constraint.

0. ~~**Task 4 by capacity dilation** (`ECC_WEIGHT_CAPACITY_SCALE × N/K`)~~ —
   **ANSWERED NEGATIVELY, §7.7 and §7.8.** `weights held` never moved over ranges
   up to 32×, and the reason is arithmetic rather than search: the resident tile
   is an integer, growing a level's share of `C` is a **2× step**, and N/K =
   1.6154 delivers 117 weights/PE against the 128 needed. The withdrawn
   projection table that used to sit here is archived in
   `legacy/FINDINGS_detail_2026-09-10.md`; it was built on
   `f_if × (1 − K/N) = 0.152`, and `f_if` no longer exists.

1. **The `datawidth` route, which replaces dilation.** VERIFIED 2026-09-10: CACTI
   receives `depth` and `width` only — `datawidth` never reaches the energy model
   (`accelergy.log`, `Calculated storage."width" as "width"`) — and Timeloop bills
   `vector_access_energy / block_size` per weight. So narrowing `datawidth` at
   FIXED geometry gives the reconstruction arm more effective capacity at
   **byte-identical** per-access energy. That is the fairness condition dilation
   could never meet. **Hard constraint: `width % datawidth == 0` or
   `timeloop-mapper` aborts** (`buffer.cpp:302`, measured — there is no floor
   path). Widths and codes are tabulated in prompt_2.md.

2. **Prove the search converges** (§2). Still open and still gates every ranking.
   Measured 2026-09-10: victory 10000 on ONE layer is 1h13m–1h34m wall at 18
   threads; victory 50000 is **6h02m–8h09m** (jobs 41582127-30, COMPLETED).
   `ECC_VICTORY=4000` is the chosen budget — the gate must be run AT 4000, not
   assumed. **The victory-50000 caches exist and are unread**
   (`results/_raw/simple_weight_stationary/vic50000__*`, 4 scales); they are the
   third point of the 2000/10000/50000 curve.

3. **Register `eyeriss_like_wglb`** in `WEIGHT_PATHS` and `PLACEMENTS` (§8.5). It
   is now the Eyeriss v1 baseline and nothing evaluates until `filter_glb` has a
   stage and a boundary.

4. **Settle the double-counting** before the first evaluation. `recon.py`'s
   default `stream` packing scales reduced stages by K/N; the mapper-side
   `datawidth` now delivers that same on-chip saving inside the Timeloop number.
   `ECC_RECON_PACKING=aligned` is the resolution — its docstring already
   describes this model and moves no SRAM access count. DRAM `datawidth` stays 8
   on every arm; `recon.py` keeps the DRAM K/N.

5. Fix defect 1 and check defect 2's denominator **in the same cold mapping
   pass**, since both change the fingerprint. Re-run Tasks 1–3 from it.

6. Task 5: `WEIGHT_PATHS` + `PLACEMENTS` for the remaining designs.

7. Widen to the six unmapped models only once the above holds.

## Reproducing any of it

All settings come from `env.sh`; `module load apptainer` first.

```bash
bash hpc/run_all.sh --eval-only                       # Tasks 1-2 from the cache
ECC_RECON_MODELING=1 bash hpc/run_all.sh --eval-only  # Task 3
python3 hpc/summary.py --scope layers-full --csv results/tables/panel_matrix.csv
bash hpc/tl.sh python3 -m eccenergy.tests.test_recon  # 26, incl. the real cache

# each sensitivity row of §7 is one evaluation, seconds:
ECC_RECON_MODELING=1 ECC_RECON_PACKING=aligned bash hpc/tl.sh bash run.sh recon --eval
#   ... likewise ECC_RECON_MODEL, ECC_RECON_ENCODER_GRANULARITY,
#       ECC_RECON_REUSE_REG_ENTRIES, ECC_RECON_INCLUDE_IDLE
```

Result files: `results/evaluation/Pre/<arch>/<model>/bch63_51/w8a8/layers-full/map-energy-vic2000l-random_pruned-seednone-ssnone-perm16/*.json`
(experiments `task2_embedded_ecc_dram_only`,
`task3_reconstruction_placement_fixed_mapping`). Every file carries its own
checks, approximations and warnings.
