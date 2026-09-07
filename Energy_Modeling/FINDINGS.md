# FINDINGS — audit of the Task 1 baseline (2026-09-06)

**Status:** the Task 1 *plumbing* holds (contract validation, parity accounting,
result store, mapping/evaluation split). The Task 1 *numbers* do not: two
defects in the workload and one in how the mapper is run make every cached
figure and the PROJECT_STATUS quick-run table unusable for ranking
architectures. Two of the three are fixed in this session; the third (mapper
search settings) is a decision the user has to make, with the evidence below.

Companion files: `progress.txt` (the ✅❌⚠️ status board),
`PROJECT_STATUS.md` (the session-rule summary), `legacy/FINDINGS.md`
(pre-rewrite, historical, do not quote). Numbers here are development
(one-to-four-layer) numbers unless a table says "full model"; none of them is
an architecture ranking.

---

## 0. Verdicts

| # | Question asked | Verdict | Where |
|---|---|---|---|
| 1 | Is the depthwise convolution representation right? | ❌ **It was wrong** — a depthwise layer was mapped as a 1-input-channel conv; the input tensor was 960× too small. **Fixed** (group dimension `G`), verified in Timeloop. | §1 |
| 2 | Are the baselines' values off? | ✅ **Yes, and the cause is the mapper search.** Under the Task 1 search Eyeriss v1/v2 cost 5× what a wider search finds on the same layers; the ranking inverts. | §2 |
| 3 | Is `eyeriss_v2_like` a dense Eyeriss v2? | ✅ Yes: own hierarchy from JETCAS 2019 Table IV / Fig. 17, RS+ constraints, dense by design. ⚠️ No CSC, no NoC energy, 45 nm, 1 GHz cycle. | §3 |
| 4 | Is NoC energy modelled? | ❌ No, for any architecture. Timeloop's "Networks" section is empty. | §4 |
| 5 | Are the architectures standardised consistently? | ✅ 7/7 pass the shared contract; WS/OS/IS differ only in the stationary operand. | §5 |
| 6 | Inconsistencies in running the mapper and the evaluation? | ⚠️ Six found; two fixed this session. | §6 |
| 7 | Energy breakdown, not totals | ✅ Available per (level, dataspace) for every record; tables in §7–8. | §7, §8 |
| 8 | Representative-layer evaluation at converged settings | ⚠️ Running (see §8). | §8 |

---

## 1. Depthwise / grouped convolutions were modelled wrongly — FIXED

### What was wrong

`eccenergy/generate.py` records a PyTorch `Conv2d(groups=g)` as
`C = in_channels/g`, `M = out_channels`, `groups = g`. The loader
(`eccenergy/workloads.py::_layer`) **dropped `groups`**, and the Timeloop
problem template (`eccenergy/timeloop.py::PROBLEM_TEMPLATE`) has no group
dimension. So MobileNetV2's `features.15.conv.1.0` (depthwise 3×3 over 960
channels) reached Timeloop as `C=1, M=960`: **one input channel shared by all
960 filters**.

Evidence from the cached mapping (`eyeriss_v2_like`, bounded-search cache,
`C1_M960_R3_S3_P7_Q7_ws1_hs1/timeloop-mapper.map.txt`):

```
DRAM [ Weights:8640 (8640) Inputs:81 (81) Outputs:47040 (47040) ]
```

The real layer reads 960 × 9 × 9 = **77,760** input scalars; the model read
**81**. MAC count (423,360) and weight count (8,640) happened to be right, which
is why the totals did not look absurd. What was wrong: input traffic, input
reuse (every PE "reused" the single channel), the mappings, and therefore the
energy of every depthwise layer.

Affected: `mobilenet_v2` (17 of 53 layers), `efficientnet_b0` (16 of 82),
`convnext_tiny` (18 of 59, 7×7 depthwise), `xception` (34 of 75). Not affected:
`resnet18`, `resnet50`, `densenet121`, `squeezenet1_1`, the transformers.

### The fix

* `Layer` gained `G` (group count); `C` and `M` are now **per group**
  (depthwise: `C=1, M=1, G=960`). `shape_name` carries `_G<g>` only when
  `G>1`, so every ungrouped shape keeps its cached name.
* `timeloop.GROUPED_PROBLEM_TEMPLATE`: dimensions `[G, C, M, R, S, N, P, Q]`,
  `G` projected into Weights, Inputs and Outputs. Ungrouped layers use the
  unchanged template; the generated files were checked byte-identical.
* `energy.load_raw()` now rejects a raw record whose per-layer shape list is
  not the current one (the mapper cache is keyed by shape, the raw record by
  model, so without this the stale aggregate would have been served forever).
* `Layer.macs` added and recorded per layer.

### Verification

Smoke test, 7 architectures, tiny search (`victory 20, search_size 50`), all
mapped; on every design:

```
DRAM [ Weights:8640 (8640) Inputs:77760 (77760) Outputs:47040 (47040) ]
```

with `G` appearing in the loop nest (spatial across PEs/clusters and temporal
in the spads). MAC totals now match published values:

| model | MACs (this loader) | published |
|---|---:|---:|
| resnet18 | 1.81 G | 1.8 G |
| resnet50 | 4.09 G | 4.1 G |
| mobilenet_v2 | 300.8 M | 300 M |
| efficientnet_b0 | 385.8 M | 390 M |
| convnext_tiny | 4.46 G (after §1b) | 4.5 G |

The smoke-test caches (`*vic20*ss50*`) were deleted afterwards; they were
this session's throwaway.

### 1b. A second workload defect: ConvNeXt's per-pixel Linear layers — FIXED

ConvNeXt's `pwconv1`/`pwconv2` are `nn.Linear` applied to `[N, H, W, C]`. The
hook recorded them as `P=Q=1` (one pixel), so their MACs and activation traffic
were undercounted by H×W (convnext_tiny came out at 0.32 G MACs against a
published 4.5 G). `generate.py` now keeps the spatial extent of a 4-D Linear
output; `convnext_tiny` was regenerated in the container. The other seven
models' entries are byte-identical to before (backup:
`ecc_energy_study/logs/model_layers.before_convnext_fix.json`).

### What this does NOT fix

The mapper caches for the four affected models hold mappings for the old
(wrong) shapes. They are simply never hit any more (new shape names). The
depthwise layers must be **re-mapped** before any of those four models is
plotted — §8 does this for two MobileNetV2 depthwise layers.

---

## 2. The mapper search decides the architecture ranking — NOT FIXED (decision needed)

### The two Task 1 development layers, four search settings

Same layers (`layer3.0.downsample.0` + `layer4.1.conv2`), same YAMLs, same
problem files. Only the mapper settings differ. Energies in µJ; "W refetch" is
DRAM weight reads ÷ weights.

| setting | eyeriss_v2 | eyeriss_v1 | WS | OS | IS | simba_like |
|---|---:|---:|---:|---:|---:|---:|
| **Task 1**: `hybrid`, victory 100, 8 thr (uncapped) | 2036.8 (7.1×) | 1979.3 (6.9×) | 732.0 (1.0×) | 904.3 (1.0×) | 581.6 (1.0×) | 3166.2 (1.1×) |
| **converged**: `random_pruned`, search_size 20000, victory 2000, 18 thr | **402.0** (1.0×) | **390.1** (1.0×) | 469.5 (1.0×) | 711.9 (1.0×) | 508.5 (1.0×) | 446.1 (1.0×) |
| ratio | 5.1× | 5.1× | 1.6× | 1.3× | 1.1× | 7.1× |

Under the Task 1 search the ordering is IS < WS < OS < v1 ≈ v2 < Simba. Under
the converged search it is **v1 < v2 < Simba < WS < IS < OS**. Nothing about the
architectures changed. The PROJECT_STATUS table ("eyeriss_v2 2,376 µJ vs WS
780 µJ") is the first row and is a search artefact.

What the mapper found for `layer4.1.conv2` on `eyeriss_like`
(`C512_M512_R3_S3_P7_Q7`, 2.36 M weights, ifmap 41 kB, psums 25 k):

| search | DRAM weight reads | mapping fault | energy |
|---|---:|---|---:|
| hybrid vic100 | 16,515,072 (7×) | `for P in [0:7)` placed at `ifmap_glb`, weights refetched per output row | 1915 µJ |
| hybrid vic500 (full-model cache) | 2,359,296 (1×) | — | 678 µJ |
| bounded ss300/perm4 | 2,359,296 (1×) | outputs spilled to DRAM (426 k reads) | 654 µJ |
| random_pruned ss20000 vic2000 | 2,359,296 (1×) | — | **363 µJ** |

And for `layer1.0.conv1` (`C64_M64_R3_S3_P56_Q56`, 36,864 weights — the
whole tensor fits in v1's 75 k on-chip weight entries):

| search | eyeriss_like refetch / energy | eyeriss_v2_like refetch / energy |
|---|---:|---:|
| hybrid vic100 | 196× / 1389 µJ | 112× / 1175 µJ |
| hybrid vic500 (full model) | 196× / 932 µJ | 112× / 793 µJ |
| bounded ss300/perm4 | 224× / 896 µJ | 32× / 394 µJ |
| random_pruned ss20000 | **7× / 307 µJ** | **16× / 315 µJ** |

A 36,864-weight tensor fetched 196 times is not a property of Eyeriss; it is
`hybrid` freezing a bad index factorisation. The simple WS/OS/IS designs have a
far smaller legal mapspace and barely move, which is exactly why they "won".

### Full-model resnet18, two settings, seven designs

| architecture | bounded ss300/perm4 (the figure) | hybrid vic500 uncapped | W refetch (bounded / vic500) |
|---|---:|---:|---|
| eyeriss_like | 10,867 µJ | 11,515 µJ | 7.0× / 5.4× |
| eyeriss_like_wglb | 8,007 | 16,211 | 1.7× / 7.2× |
| eyeriss_v2_like | 6,724 | 12,699 | 2.3× / 7.6× |
| simple_weight_stationary | 10,496 | 11,000 | 1.3× / 1.1× |
| simple_output_stationary | 10,251 | 12,303 | 1.0× / 1.1× |
| simple_input_stationary | 8,310 | 8,128 | 1.1× / 1.0× |
| simba_like | 7,679 | 9,149 | 1.4× / 2.2× |

A *more* thorough hybrid search (vic500, uncapped, 5–6 h of mapping) gave
*worse* Eyeriss numbers than the 10-minute bounded search. That is the
signature of a search that is not converging, not of a design. (These
full-model records also contain the wrong depthwise shapes for
`mobilenet_v2`; the resnet18 rows do not.)

### What this means and what to decide

1. **Do not quote any existing figure or table for an architecture ordering.**
   `results/figures/*`, `results/tables/*`, `diagnose.csv` and the
   PROJECT_STATUS quick-run table all come from `hybrid` or from the bounded
   search.
2. **Change the default search** (`run.sh`: `ECC_MAPPER_ALGORITHM=hybrid`,
   `ECC_MAPPER_SEARCH_SIZE=` empty). The evidence favours
   `random_pruned` with `search_size 20000`, `victory 2000`, `timeout 2000`,
   pinned threads. Cost at 18 threads, from the sidecars: 43–250 s per shape
   for the Eyeriss designs (v2 at victory 4000 because of level scaling), so a
   full 21-shape resnet18 sweep is roughly 30–60 min per architecture. This
   is a modelling default, so it is left for the user to set; the
   verification run in §8 uses these settings explicitly.
3. **Prove convergence, do not assume it**: run the same layers at
   `search_size 20000` and `40000` and require the totals to agree within a
   few percent. That test has not been done yet.
4. Timeloop's `victory_condition` is per thread and `search_size` is per
   thread; both are in the fingerprint, and so is the thread count, so pin
   `ECC_MAPPER_THREADS` (18 was used here on a 20-core host).

---

## 3. Is `eyeriss_v2_like` a dense Eyeriss v2? — Yes, with listed gaps

It is **not** the stock `eyeriss_like` renamed. Checked line by line against
`archs/eyeriss_v2_like/arch_paper.yaml` and the README:

| Eyeriss v2 (JETCAS 2019, Table IV / Fig. 17) | model | ok? |
|---|---|---|
| 192 PEs = 16 clusters × 12 PEs | `PE_cluster` meshX 16, `PE` meshY 12 | ✅ |
| 2 MACs per PE (SIMD) → 384 MACs | `SIMD` meshX 2, `mac` ×384 | ✅ |
| GLB 192 kB: iact 16×3×1.5 kB = 72 kB, psum 16×4×1.875 kB = 120 kB | `iact_glb` 9216×64b/48 banks, `psum_glb` 24576×40b/64 banks | ✅ |
| weights not stored in the GLB (DRAM → weight routers → PE spads) | `!Nothing` for Weights at the GLB | ✅ (this is what v1's `eyeriss_like` also does; v1's paper 8 kB filter GLB is bracketed by `eyeriss_like_wglb`) |
| weight spad 96×24b = 288 B | `weights_spad` 96×24b at datawidth 8 → 288 weights/PE, 55,296 on chip | ✅ (dense rule; the CSC-native reading would give 192, recorded as `contested` in provenance.yaml) |
| iact spad 16×12b = 24 B | 24×8b (Timeloop needs width % datawidth == 0) | ⚠️ same bits, narrower words |
| psum spad 32×20b | 32×20b | ✅ |
| psums 20b, weights/iacts 8b | datawidth 20 / 8, `adder_width 20` | ✅ |
| RS+ dataflow: any dimension tiled spatially, incl. channel groups | cluster: C, M, P, Q free spatially (N,R,S pinned); PE: R,S,C,M free (N,P,Q pinned); spad temporal constraints mirror v1 | ✅ |
| CSC compressed sparse processing | not modelled — dense on purpose (the study compares dense designs) | ⚠️ intended |
| hierarchical-mesh NoC router energy | not modelled (no network components; same gap for every design) | ❌ see §4 |
| NoC mode switching (unicast / multicast / broadcast) | one spatial mapping per layer, chosen by the mapper | ⚠️ proxy |
| 65 nm, 200 MHz | 45 nm and a 1 ns cycle for every design | ⚠️ deliberate, uniform |

So: the model is a **dense Eyeriss v2** at the granularity Timeloop can
express, and it is structurally different from `eyeriss_like` (v1: 168 PEs
12×14, 108 kB GLB split 52/48 kB, 448 weights/PE, 16b psums, RS with S across
PE rows and Q,M across columns).

**Paper sanity band (not a validation).** JETCAS 2019 reports 0.508 mJ per
inference for dense Eyeriss v2 on MobileNet v1 (0.5 width, 128×128) ≈ 46 M
MACs → ≈ 11 pJ/MAC at 65 nm, 200 MHz, including NoC, control, clocking and
leakage. This model's converged number on the two resnet18 development layers
is 3.3 pJ/MAC at 45 nm without NoC/control/clock. Scaling 65→45 nm by ~2× puts
the paper near 5.5 pJ/MAC, so the model sits ~1.7× below a figure that
includes terms it omits — consistent in order of magnitude, nothing more.

**The paper's v2-dense = 1.9× v1 claim is not reproduced**: on the converged
development layers v1 (390 µJ) and v2 (402 µJ) are within 3%. Reasons the
model cannot see: the missing NoC term (v1's flat multicast NoC vs v2's
hierarchical mesh), utilisation/leakage at 200 MHz, and the paper's benchmark
being MobileNet v1 where PE utilisation — not energy per access — is what v2
fixed. Do not treat parity here as a defect until §8's depthwise layers are in.

---

## 4. NoC energy is zero for every architecture — OPEN

Every `timeloop-mapper.stats.txt` has an empty `Networks` section (only
"Total scalar accesses" per level, no energy). No arch YAML declares a
`!Network` node and the Accelergy plug-in set in the image has no router
primitive. Timeloop *does* account multicast reuse in the access counts, so
data volumes are right; the interconnect **circuit** energy is missing. The
plotted category "On-chip SRAM/RF" (internally `Local (spads/RF/NoC)`)
therefore contained no NoC term despite its name.

**Superseded 2026-09-07.** The interconnect is now costed: `archs/_shared/noc.yaml`
supplies one shared 45nm wire constant plus per-architecture switching terms,
injected onto every spatial container so Timeloop's Legacy networks carry energy
inside the mapper's objective. NoC is its own plotted category ("NoC /
interconnect"); the amber category was renamed `Local (spads/RF)` so its key no
longer claims to contain it. See "Interconnect (NoC) model" below.

Effect on the comparison: symmetric in the sense that nobody pays, but not
neutral — designs whose advantage is a cheaper interconnect (v2's HM-NoC,
Simba's NoP/NoC) get none of it, and designs with large fan-out broadcast
(v1, the 16×16 simple arrays) are not charged for it. For the reconstruction
study this matters directly: R2 ("after the NoC") placements trade NoC bit
volume against encoder count, and that trade is currently invisible.

Fix options (not done): declare per-level `network` attributes with a
wire/router energy model, or add an Accelergy table-based router component and
`!Network` nodes for the eyeriss_v2 mesh, v1 multicast bus, and simba NoC/NoP.

---

## 5. Architecture standardisation — PASS, with the known caveats

`ECC_PYTHON=<anaconda> bash run.sh validate` on the host:

```
PASS: 6 architecture(s) obey the shared contract.      (7 with eyeriss_like_wglb)
Operand precision, DRAM geometry and process node are identical;
topology, capacity, weight bypass rules and accumulator width are each design's own.
```

| architecture | origin | acc | fanout | on-chip weights | GLB keeps W |
|---|---|---:|---:|---:|---|
| eyeriss_v2_like | published | 20 | 384 | 55,296 | no |
| eyeriss_like | published | 16 | 168 | 75,264 | no |
| eyeriss_like_wglb | derived | 16 | 168 | 83,456 | yes (8 kB) |
| simple_weight_stationary | locally authored | 16 | 256 | 164,096 | yes |
| simple_output_stationary | locally authored | 16 | 256 | 65,792 | yes |
| simple_input_stationary | locally authored | 16 | 256 | 65,792 | yes |
| simba_like | reference design | 24 | 256 | 2,113,536 | no |

WS/OS/IS are byte-for-byte the same design (16×16 PEs, 128 kB GLB split
64/64 kB, 192×16b PE spad, 8b operands, 16b psums, 45 nm) except for which
operand `pe_spad` keeps and the spatial permutation — a controlled dataflow
comparison, as the user's table requires. DRAM costs 64.0 pJ per 8-bit scalar
on every design (CACTI LPDDR4, 512 pJ per 64b word), checked in every raw
record.

Caveats that stand: `simba_like` is the exercises' reference design, not
Simba (256 MACs, 3 MB of PE-private buffers, 16 nm design modelled at 45 nm);
`simple_input_stationary` was authored here and reproduces no paper.

---

## 6. Inconsistencies in running the mapper and the evaluation

1. **The panel figure and `diagnose.csv` come from the bounded development
   search** (`random_pruned`, search_size 300, 4 permutations, victory 100,
   4 threads) and, for mobilenet_v2, from the wrong depthwise shapes. Its
   title says "NOT converged"; §2 shows how far. ⚠️ do not quote.
2. **The Task 1 result JSONs are orphaned from their mapper cache.** They
   were written from the `hybrid vic100 / 8 threads` cache
   (`fp-25b35c9b0d5a`, `fp-2b49d85baff2`, …). Since then
   `Config.mapper_settings()` gained `search_size` and
   `max_permutations_per_if_visit`, which are hashed into the fingerprint, so
   the same YAML + settings now hash to `fp-75fa520187e1` etc. `bash run.sh
   baseline --eval` at the Task 1 settings finds no usable cache. The
   mapping directories still exist on disk. ⚠️ documented, not repaired (the
   Task 1 numbers are superseded anyway, §2).
3. **`Config.mapper_slug` changed** (`…-seednone` → `…-seednone-ssnone-perm16`),
   so a re-evaluation lands in a different `results/evaluation/…` directory
   than the Task 1 files and `load_latest()` would not pair them. The offline
   test that pinned the old slug failed; the assertion was updated to the new
   slug (15/15 pass). ✅
4. **Two fingerprints per treatment** exist for several designs
   (e.g. `eyeriss_like … ss20000 …/fp-65833ebb89d3` and `/fp-c601e8244139`):
   the arch YAMLs were edited (address-decoded register files) between runs.
   Only the fingerprint the current YAML produces is valid; a glob by mtime
   picked the wrong one once during this audit. Always compute the
   fingerprint with the package (the `breakdown.py` helper in §7 does).
5. **The raw-energy cache did not check the layer list.** A record keyed by
   (arch, treatment, fingerprint, model) kept serving after the workload
   definition changed. Fixed (§1). ✅
6. **`eyeriss_like_wglb` costs more than `eyeriss_like` in the vic500
   full-model run** (16.2 vs 11.5 mJ, 7.2× vs 5.4× refetch). A design with
   an extra weight-reuse level cannot need *more* DRAM weight traffic unless
   the search failed; with `keep: [Weights]` the level is mandatory, which
   also shrinks the legal mapspace. Another convergence symptom (§2), plus
   `keep` forcing every weight through the 8 kB buffer.

Also checked and found consistent: the `globals.yaml` node (45 nm) matches
the contract; the DRAM per-access cost is identical across designs; the
`victory_for` level scaling gives v2/wglb/Simba 2× the base victory (9 loop
levels vs 8); parity accounting hand check passes on every existing result.

---

## 7. Energy breakdown, not totals

Every raw record (`results/_raw/…/<model>.json`) carries `levels[]` — energy,
reads and writes per (level, dataspace) — and every evaluation JSON carries it
as `detail.per_level`. The helper used for the tables below is
`breakdown.py` (session scratch; usage: same `ECC_*` environment as the run,
`PYTHONPATH=. python breakdown.py [arch …]`, prints per-arch categories,
per-layer pJ/MAC and refetch, and the per-level table using the fingerprint the
current YAML produces).

Two Task 1 development layers, **converged search**, µJ, largest terms:

| architecture | DRAM W | DRAM I/O | GLB | spads/RF | MAC | total | pJ/MAC |
|---|---:|---:|---:|---:|---:|---:|---:|
| eyeriss_v2_like | 155.2 | 13.2 | 35.3 (psum_glb 32.2) | 55.7 (weights_spad 26.7, psum_spad 21.2) | 142.6 | 402.0 | 3.29 |
| eyeriss_like | 155.2 | 16.4 | 9.2 | 70.7 (weights_spad 51.3, psum_spad 15.3) | 138.6 | 390.1 | 3.20 |
| simple_weight_stationary | 153.1 | 20.0 | 70.9 (psum_glb 29.4, operand_glb 41.5) | 86.9 (pe_spad 42.5) | 138.6 | 469.5 | 3.85 |
| simple_output_stationary | 153.1 | 34.6 | 69.8 | 315.8 (**pe_spad 271.4**: 16b psum accumulator, 29.7 M accesses) | 138.6 | 711.9 | 5.83 |
| simple_input_stationary | 153.1 | 10.6 | 150.1 (psum_glb 86.5, operand_glb W 62.4) | 56.2 | 138.6 | 508.5 | 4.17 |
| simba_like | 155.2 | 19.6 | 1.1 | 123.4 (**PEInputBuffer 99.2**: 64 kB SRAM per PE) | 146.7 | 446.1 | 3.66 |

Reading it the way the user's item 12 asks: DRAM weight energy is *identical*
(1× refetch everywhere, 64 pJ/scalar), so on these two weight-heavy layers the
architectures differ only in on-chip terms — OS pays for accumulating 16b
psums in a 192-entry SRAM per PE, Simba for its oversized input buffer, IS for
streaming weights through the GLB, v1/v2 almost nothing beyond the MACs. That
is believable. The Task 1 table (§2, first row), where v1/v2 lost on **DRAM**,
was not.

---

## 8. Representative-layer verification at converged settings — IN PROGRESS

Run launched 2026-09-06 14:44 local, detached in the container:

```
ECC_SWEEP=arch
ECC_SWEEP_ARCHS="eyeriss_v2_like eyeriss_like eyeriss_like_wglb simple_weight_stationary
                 simple_output_stationary simple_input_stationary simba_like"
ECC_MAPPER_ALGORITHM=random_pruned ECC_MAPPER_SEARCH_SIZE=20000 ECC_VICTORY=2000
ECC_MAPPER_TIMEOUT=2000 ECC_MAPPER_THREADS=18
ECC_CONST_MODEL=resnet18     ECC_LAYERS="layer1.0.conv1 layer3.0.downsample.0 layer4.1.conv2"      bash run.sh map
ECC_CONST_MODEL=mobilenet_v2 ECC_LAYERS="features.3.conv.0.0 features.8.conv.1.0 features.15.conv.0.0 features.15.conv.1.0" bash run.sh map
```

Logs: `ecc_energy_study/logs/map_resnet18.log`, `map_mobilenet_v2.log`
(exit codes in the matching `.exit` files). Cache slug
`multimodel__opt-energy__vic2000__vicx__alg-random_pruned__ss20000__to2000__paper`.

Why these layers: `layer1.0.conv1` is the activation-heavy 3×3 where the
Eyeriss designs' 196× refetch appeared; `layer3.0.downsample.0` and
`layer4.1.conv2` are the recorded development pair; `features.3.conv.0.0` is
a 56×56 pointwise expansion (activation-dominated, MobileNet's early stage);
`features.8.conv.1.0` (G=384, 14×14) and `features.15.conv.1.0` (G=960, 7×7)
are the corrected depthwise shapes; `features.15.conv.0.0` is the matching
pointwise 7×7.

*Results are appended below when the run completes.*

---

## 9. Expected ordering (plan §7 table) versus what the model says

Cannot be answered yet: the only converged numbers are two weight-dominated
resnet18 layers (§7), on which v1 ≈ v2 < Simba-like < WS < IS < OS. That is
compatible with the literature's "RS ≲ optimized WS" for ResNet-class 3×3
layers and with "different layers select different dataflows", and it says
nothing about MobileNetV2 until §8's depthwise numbers exist. Two structural
reasons to expect the model to *under*-reward RS/RS+ relative to the papers
remain: no NoC energy (§4) and dense-only (§3).

Against the user's "suspicious results" list:

| observation | in this model | reaction |
|---|---|---|
| Eyeriss v2 5–10× worse than every simple dataflow | was true under `hybrid`; gone under the converged search | search artefact, resolved |
| depthwise layers cost like dense convs | worse: they were *cheaper* than they should be (81 inputs) | workload defect, fixed |
| Eyeriss v2 = official `eyeriss_like` unchanged | no — separate hierarchy, §3 | not the case |
| sparse v2 = dense v2 | only dense exists | expected |
| WS/OS/IS with different SRAM/PE/precision/process | no — identical except stationary operand, §5 | comparison valid |
| changing mapper settings changes the winner | **yes, dramatically** (§2) | search under-converged — the central problem |

---

## 10. What was changed and how to reproduce

Code (all LF, all offline tests pass, `validate` passes):

| file | change |
|---|---|
| `eccenergy/workloads.py` | `Layer.G`, per-group `M`, `_G<g>` shape suffix, `macs`, `is_grouped`; loader reads `groups` |
| `eccenergy/timeloop.py` | `GROUPED_PROBLEM_TEMPLATE`; `problem_path()` selects it when `G>1` |
| `eccenergy/energy.py` | `load_raw(..., layers=)` rejects stale shape lists; `collect()` passes the layers; per-layer `macs`/`grouped` |
| `eccenergy/generate.py` | 4-D `nn.Linear` outputs keep `P×Q` |
| `eccenergy/tests/test_results_store.py` | mapper-slug assertion updated to the current slug |
| `ecc_energy_study/model_layers.json` | `convnext_tiny` regenerated; other models unchanged (backup in `ecc_energy_study/logs/`) |
| `FINDINGS.md`, `progress.txt` | new |

Nothing in `archs/`, `run.sh` or the ECC arms was changed. No mapper cache was
deleted except this session's own smoke test.

Commands used (host = Git Bash with `ECC_PYTHON=/c/Users/nithi/anaconda3/python.exe`;
container = the running `timeloopaccelergy/timeloop-accelergy-pytorch` with
this folder mounted at `/home/workspace`):

```
# audit
ECC_PYTHON=... bash run.sh validate
python -m eccenergy.tests.test_results_store
# fix verification (container)
python3 -m eccenergy.generate models convnext_tiny
ECC_SWEEP=arch ECC_CONST_MODEL=mobilenet_v2 ECC_LAYERS=features.15.conv.1.0 \
  ECC_VICTORY=20 ECC_MAPPER_SEARCH_SIZE=50 ECC_MAPPER_ALGORITHM=random_pruned bash run.sh map
# verification run: see §8
```

---

## 11. What the next session should do, in order

1. Read §8 once the run has finished (or re-run its two commands; they resume
   from the cache). Then run, on the host, the evaluation for both selections
   with the same `ECC_*` environment plus `bash run.sh baseline --eval`.
2. Decide the default mapper setting (§2 item 2) and write it into `run.sh`;
   then run the convergence check (§2 item 3) on the same layers.
3. Rebuild the full-model caches for resnet18 and mobilenet_v2 at that
   setting; only then re-draw `ArchitectureSweep` and the panel figure.
4. Decide whether to model NoC energy before Task 3 (it is on the critical
   path of the R2 placements).
5. Only then Task 2.


## 12. HiPerGator bring-up — PASS (2026-09-06)

All five §5 checks passed through `bash hpc/tl.sh` in 18-core allocation
41260181 on c0710a-s28: six architectures validated, 15 tests passed, dry-run
configuration valid, six exact cached totals with six parity hand-check PASSes,
and one uncached mapping with a valid sidecar. Logs: `hpc/logs/bringup-check*.log`.
Check 4 reported six raw-cache hits and `ECC_FROM_CACHE=1`; no Timeloop was
invoked. The final wrapper was also checked with another cache-only evaluation.

| Architecture | Check 4 Timeloop energy (µJ) |
|---|---:|
| eyeriss_v2_like | 401.977 |
| eyeriss_like | 390.110 |
| simple_weight_stationary | 469.457 |
| simple_output_stationary | 711.872 |
| simple_input_stationary | 508.517 |
| simba_like | 446.067 |

The project was copied with `cp -a Energy_modeling/. .` from the outer project
root, then `bash tools-fix-eol.sh` was run. The nested source remains.
All required directories exist; system `python3 -c "import eccenergy"` passed.
Byte comparisons found no differences in 23,249 mapper-cache files or 101 raw
cache files after copying. No existing mapper/raw cache was deleted.

Commands that worked (login shell, then the allocated shells):

```bash
srun --account=rewetz --qos=rewetz --cpus-per-task=4 --mem=16gb --time=01:00:00 --pty bash -i
module load apptainer
export ECC_MAPPER_THREADS=18
export APPTAINER_CACHEDIR=/blue/rewetz/vkamineni/.apptainer
export APPTAINER_TMPDIR=/blue/rewetz/vkamineni/.apptainer_tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
apptainer pull /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling/timeloop.sif docker://timeloopaccelergy/timeloop-accelergy-pytorch:latest-amd64
apptainer exec /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling/timeloop.sif bash -c 'command -v timeloop-mapper && python3 -c "import pytimeloop, pandas, matplotlib, yaml; print(\"ok\")"'
exit

srun --account=rewetz --qos=rewetz --cpus-per-task=18 --mem=8gb --time=02:00:00 --pty bash -i
module load apptainer
cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
export ECC_MAPPER_THREADS=18
bash hpc/tl.sh bash run.sh validate
bash hpc/tl.sh python3 -m eccenergy.tests.test_results_store
bash hpc/tl.sh bash run.sh --dry-run
export ECC_MAPPER_ALGORITHM=random_pruned ECC_MAPPER_SEARCH_SIZE=20000 ECC_VICTORY=2000 \
       ECC_MAPPER_TIMEOUT=2000 ECC_MAPPER_THREADS=18 ECC_SWEEP=arch \
       ECC_SWEEP_ARCHS="eyeriss_v2_like eyeriss_like simple_weight_stationary simple_output_stationary simple_input_stationary simba_like"
ECC_LAYERS="layer3.0.downsample.0 layer4.1.conv2" bash hpc/tl.sh bash run.sh baseline --eval
ECC_SWEEP_ARCHS=eyeriss_like ECC_LAYERS=layer2.0.downsample.0 bash hpc/tl.sh bash run.sh map
exit
slurmInfo -g rewetz
sacctmgr show assoc user=vkamineni format=account,qos,grptres
```

Image build: allocation 41258861 on c0704a-s27, Apptainer 1.4.2-1.el8,
exit 0. Account/QOS names matched the guide. `slurmInfo` reported **181
concurrent investment CPU cores**, 1414 GB RAM, 23 GPUs; `rewetz-b` is also
permitted, with availability dependent on idle resources.

Environment differences, fixed in `hpc/tl.sh` or `hpc/HIPERGATOR.md`:

- The inherited HiPerGator `which` function passes options unsupported by the
  container's `/usr/bin/which`; use Bash `command -v` for the image smoke check.
- The first mapping attempt failed because CACTI writes
  `cacti_inputs_outputs` beside its plugin in the read-only SIF. The wrapper
  now binds `hpc/.runtime/cacti_inputs_outputs` to that exact plugin scratch
  path under `/usr/local/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/`.
  The same command then passed without model or mapper-setting changes.
  Initial failure logs are preserved under `hpc/logs/bringup-check5-first-*`.
- `/blue` binding, `--pwd`, and `apptainer exec` worked as supplied. Accelergy
  created its config in `/home/vkamineni/.config/accelergy`; HOME needed no override.
- Agent-tool sandbox networking blocked SLURM queries; approved execution
  outside that sandbox worked. This was not a cluster account/QOS issue.

Check 5 was verified absent from both copied fingerprints before mapping.
It reported **0 cached, 1 newly mapped, 0 failed**, **28.724 s**;
`mapping.json` records source `newly mapped`, `num_threads_effective=18`,
fingerprint `65833ebb89d3`, mapping ID `4a1dd496acf6f42e`. Sidecar:

```
ecc_energy_study/outputs/eyeriss_like/multimodel__opt-energy__vic2000__vicx__alg-random_pruned__ss20000__to2000__paper/fp-65833ebb89d3/C64_M128_R1_S1_P28_Q28_ws2_hs2/mapping.json
```

No run.sh defaults, architecture YAMLs, ECC arms or mapper settings were changed.
No parallel execution was designed or built *at this point*. These are migration
checks, not new evidence of model convergence or full-model architecture
ranking; §13 is the run that provides the latter.


## 13. First whole-model run on HiPerGator — resnet18 + mobilenet_v2 × 6 architectures (2026-09-06)

The first result in this project that is a **whole-model** number at the
converged search, on every architecture, with the corrected grouped/depthwise
shapes. It supersedes the two-layer development figures for ranking purposes.

### How it was produced

```bash
cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
sbatch hpc/map.sbatch   # job 41269118
module load apptainer
bash hpc/eval_panel.sh
python3 hpc/summary.py --scope layers-full --csv results/tables/panel_matrix.csv
```

One array task per (architecture, model); 18 cores per task, `%9` = 162 of the
181-core `rewetz` investment. **12/12 COMPLETED, 0 failed**, 258 shape-mappings,
48 minutes wall. Per-task elapsed ranged 8m54s (`eyeriss_like` × resnet18, 5 of
12 shapes already cached from the laptop) to 30m49s
(`simple_output_stationary` × mobilenet_v2, all 31 shapes cold). The transferred
laptop cache produced hits exactly where expected (4, 5, 4, 3, 3, 2 on the
resnet18 tasks; mobilenet_v2 was entirely cold), which is further evidence the
cache and fingerprints survived the transfer intact.

Mapper settings, identical to the §12 check-4 setting:
`random_pruned`, `search_size=20000`, `victory=2000` (`levels`-scaled, so 4000
on the 9-level designs), `timeout=2000`, `perms/if=16`, **`threads=18`**.

### Whole-model energies (µJ)

resnet18 — 21/21 layers, 12 distinct shapes:

| architecture | Timeloop | + parity | = conventional ECC | DRAM weight refetch |
|---|---:|---:|---:|---:|
| eyeriss_v2_like | 4,946.067 | 295.674 | 5,241.742 | ×1.27 |
| eyeriss_like | 5,007.368 | 291.487 | 5,298.855 | ×1.25 |
| simba_like | 5,546.641 | 332.237 | 5,878.879 | ×1.42 |
| simple_input_stationary | 6,194.744 | 245.030 | 6,439.774 | ×1.05 |
| simple_weight_stationary | 7,427.596 | 239.477 | 7,667.073 | ×1.03 |
| simple_output_stationary | 8,839.347 | 261.596 | 9,100.943 | ×1.12 |

mobilenet_v2 — 53/53 layers, 31 distinct shapes, 10 of them depthwise:

| architecture | Timeloop | + parity | = conventional ECC | DRAM weight refetch |
|---|---:|---:|---:|---:|
| eyeriss_like | 1,834.253 | 81.208 | 1,915.462 | ×1.17 |
| eyeriss_v2_like | 1,842.746 | 82.239 | 1,924.986 | ×1.19 |
| simba_like | 1,937.962 | 75.781 | 2,013.742 | ×1.09 |
| simple_input_stationary | 1,961.365 | 70.484 | 2,031.850 | ×1.02 |
| simple_weight_stationary | 2,207.769 | 75.432 | 2,283.201 | ×1.09 |
| simple_output_stationary | 2,483.524 | 69.795 | 2,553.319 | ×1.01 |

Every parity hand check PASSed on all twelve results. Machine-readable copies:
`results/tables/panel_matrix.csv` and
`results/evaluation/Pre/<arch>/<model>/bch63_51/w8a8/layers-full/...`.

### The ranking is preserved across the two models

    v2 ≈ v1  <  Simba-like ≈ input-stationary  <  weight-stationary  <  output-stationary

This is the claim the two-panel layout exists to test, and it holds: swapping a
classic residual CNN for a depthwise-separable one does not reorder the
architectures. The two Eyeriss designs trade places between the models, but they
are within 1.2% of each other in both, so that swap is not meaningful.

`results/figures/ArchitectureSweep__panels__resnet18__mobilenet_v2.{png,pdf}`
now carries this data and is quotable. The previous bounded-search version was
moved to `results/figures/_superseded_bounded_search/` before being overwritten;
it must still not be quoted.

### The development layer pair is NOT a proxy for the model

This is the most important negative result here. On
`layer3.0.downsample.0 + layer4.1.conv2` the order was

    v1 (390.110)  <  v2 (401.977)  <  Simba (446.067)
      <  WS (469.457)  <  IS (508.517)  <  OS (711.872)

On the whole model, **weight-stationary and input-stationary swap**, and the gap
is large — WS is 20% *worse* than IS (7,427.6 vs 6,194.7 µJ) where the layer
pair had it 7.7% better. Ranking architectures from the two-layer quick run
would therefore have produced a wrong answer about weight-stationary. The layer
pair remains fine for what it was chosen for — validating the implementation —
but no ordering should be read off it.

### ECC savings at BCH(63,51)

From the regenerated panel figure, Recon+ against the conventional-ECC baseline:
6.8% / 8.0% / 5.7% / 3.5% / 5.9% / 6.0% on resnet18 (v2, v1, WS, OS, IS, Simba)
and 4.7% / 5.3% / 4.5% / 3.3% / 4.4% / 4.1% on mobilenet_v2. The saving tracks
DRAM weight traffic, so it is largest on the Eyeriss designs, which refetch most,
and smallest on output-stationary, which refetches least.

### Caveats that still stand

* **`eyeriss_like` refetches ×1.25 here, not ~7×.** `CLAUDE.md` describes a ~7×
  refetch for `eyeriss_like` against 1.0–2.3× elsewhere, and derives the
  bracketing-pair argument from it. That was measured under the earlier bounded
  search; at the converged setting the spread is far narrower (1.03–1.42×). The
  bracketing-pair claim needs re-measuring before it is repeated.
* **No Eyeriss v1 bound was produced.** `eyeriss_like_wglb` and
  `eyeriss_v2_like_wglb` were not in this sweep, so per `CLAUDE.md` the Eyeriss v1
  number above is one choice of bound, not a measurement. Add tasks for the
  `_wglb` pair before quoting an Eyeriss v1 saving.
* **NoC energy is still zero** for every architecture, so all of these totals
  omit the interconnect term.
* **Six models are still unmapped** — resnet50, densenet121, squeezenet1_1,
  efficientnet_b0, convnext_tiny, xception. Whole-model coverage of the full
  `ECC_SWEEP_MODELS` list is 221 distinct shapes × 6 architectures = 1,326 tasks,
  roughly five times what this run did. `efficientnet_b0`, `convnext_tiny` and
  `xception` contain depthwise layers and so have no reusable cache at all.
* `simba_like` is a reference design, not Simba; see `archs/_shared/provenance.yaml`.

### Environment change made for this run

`patched_arch_path()` and `write_globals()` in `eccenergy/archs.py` now write
through `_write_atomic()` (temp file in the same directory, then `os.replace`).
Both write a path shared by every array task; the content each task writes is
byte-identical, so they never disagree, but a plain `write_text` leaves a
truncate-then-write window in which a concurrent reader can see a partial file.
The content is unchanged and §12 check 4 still reproduces the six laptop totals
exactly after the change, which is how that was verified. No `run.sh` default,
architecture YAML, ECC arm or mapper setting was touched.

## Interconnect (NoC) model — added 2026-09-07

**What was wrong.** Timeloop instantiates a `LegacyNetwork` between every pair
of adjacent levels and tracks fanout, multicast factor, hops and spatial
reductions per mapping — but its built-in wire model is a stub
(`src/pat/pat.cpp:81`, `WireEnergy(...) { return 0; }`), and no architecture
supplied coefficients. Every NoC therefore cost 0.00 pJ, in the evaluator **and
in the mapper's objective**. The `Networks` section of every `stats.txt` was
empty; Timeloop hides networks whose energy is zero.

**Where the coefficients had to go.** Timeloop reads a network's specs from the
storage level it hangs off, and the network between levels *i-1* and *i* takes
them from the **outer** level. For the inter-PE networks — the only ones with
non-zero hop counts — that outer level is the `dummy_storage` node the v4
front-end synthesises from a `!Container ... spatial:` block, which is not in
any arch YAML. Injecting into the storage components alone lands the constants
on exactly the networks whose hop count is zero (measured: still 0.00). The
constants go on the spatial `!Container`s, whose attributes propagate onto the
synthesised level. `network_word_bits` must be set explicitly because that
level is declared `datawidth: 1`.

**The model** (`archs/_shared/noc.yaml`, injected by `archs._inject_noc`):

| term | value | applies to | source |
|---|---|---|---|
| `wire_energy` | **0.12 pJ/bit/mm** (first pass: 0.4) | every design | Keckler, Dally et al., IEEE Micro 2011, Table 1 @40 nm: "Wire energy (per transition) 240 fJ per bit per mm"; "Wire energy (256 bits, 10 mm) 310 pJ" → 0.121 pJ per bit moved per mm ([doi:10.1109/MM.2011.89](https://doi.org/10.1109/MM.2011.89)). The 0.4 figure ([arXiv 1207.6819](https://arxiv.org/pdf/1207.6819)) is for repeated *global* wire at full activity — an upper bound |
| `energy-per-ingress` | `calibrated` → 0.9152 pJ at wire 0.12 (0.4011 at 0.4) | Eyeriss v1 (+wglb), WS/OS/IS | Eyeriss ISCA'16 Table IV, inter-PE = 2× MAC @16b → 1.13555 pJ @8b, minus the wire term at v1's own 91.81 µm PE pitch over 2.5 hops; derived in code so v1's transfer never moves off its paper |
| `router_energy` | 0.25 pJ **per flit** per router traversed → v2 0.083 pJ/operand (24b port = 3 × 8b), Simba-like 0.25 (no published packing) | Eyeriss v2 (+wglb) at the **router-cluster level only**; Simba-like | v2 Sec. III-C: "circuit-switched routing, which mainly consists of muxes and is statically configured"; "each port has a bitwidth of 24 bits … three 8b … values"; routers "connect to one row of PEs within the cluster" → the intra-cluster level is wiring. Value is an assumption (see below) |

Hop distance is **not** a constant: Timeloop derives `tile_width` from the
Accelergy area of the inner level (`topology.cpp:1716`), so the same wire
constant produces per-architecture wire cost from each design's own floorplan
(v1 PE hop 91.8 µm, v2 108.2 µm, Simba-like 932 µm, WS/OS/IS 50.9 µm).

Structure is per-design and cited (v2's 10 routers per cluster, Simba's 3
routers per global PE, v1's multicast controllers); **energy constants are
shared**. That is deliberate: none of the papers publishes a pJ figure for its
on-chip NoC, and a cross-design gap must be architecture, not silicon.

**Validation target.** Eyeriss v2's paper states the hierarchical mesh is
"6%–10% of the total energy consumption" (JETCAS 2019 Sec. V, Fig. 18). The
`noc_share_within_published_band_eyeriss_v2` check in every v2 result records
whether the modelled share lands in that band.

**Measured while building it** (eyeriss_like, FC layer C1280 M1000, 6000
mappings, energy objective):

| configuration | PE utilisation | Networks reported |
|---|---|---|
| no NoC constants (previous model) | 23.81% | none |
| wire only, 1-bit words | 4.76% | 1, 0.126 µJ |
| wire, 8-bit words | 0.60% | none (fully serial mapping) |

The last row is the reason `energy-per-ingress` exists as a term: Timeloop
charges wire energy per **hop**, so a mapping with no spatial fanout moves all
its data for free, and an energy-only objective finds that corner. A flat
per-ingress cost removes the artifact.

**The inheritance trap, and its fix.** A spatial container's attributes are
inherited by every child component, so the two hop-independent terms
(`energy-per-ingress`, and `router_energy`, which Timeloop bills as
`1 + floor(hops)` routers — one even at zero hops) also reached the scratchpads
inside each PE. Their inferred networks were then charged as if a
register-to-ALU path were a NoC. First smoke run, eyeriss_like layer4.1.conv2:

| network | fJ/compute | what it is |
|---|---|---|
| `psum_spad <==> mac` | 1203.3 | register → ALU, inside a PE |
| `weights_spad <==> psum_spad` | 818.9 | two spads inside a PE |
| `ifmap_spad <==> weights_spad` | 426.0 | two spads inside a PE |
| `inter_PE_spatial <==> ifmap_spad` | 73.1 | **the inter-PE NoC** |
| `inter_PE_column_spatial <==> inter_PE_spatial` | 69.2 | **the column fanout NoC** |

94% of a 45.5% "NoC" share was the first three rows. `topology.cpp:855`
settles which level feeds which network — "inferred network *i* connects
outer-storage-level *i* to inner-storage-level *i-1*", i.e. the **outer**
level's spec — so every `!Component` now carries an explicit
`router_energy: 0` / `energy-per-ingress: 0`, overriding what it inherits;
the synthesised fanout levels keep the container's terms; and wire energy,
which needs hops, is left to inherit harmlessly. Verified in
`parsed-processed-input.yaml` (what Timeloop reads): the dummy levels carry
`wire 0.4 / ingress 0.4011 / 8 bits`, every component carries zeros.

**Not every spatial level is a NoC.** With the inheritance fixed, Eyeriss v1
landed at 5.1% NoC but Eyeriss v2 at 29%, and the breakdown showed
`inter_SIMD_spatial <==> mac` = **750 fJ/compute** — exactly 3 operands × the
0.25 pJ router charge per MAC. v2's `SIMD` container (`meshX: 2`) is the
two-MAC datapath inside a PE (JETCAS Sec. III-B, "two MACs per cycle"), not the
hierarchical mesh; Simba's `distributed_buffers` / `reg_mac` containers are its
vector-MAC lanes, likewise inside the PE. `noc.yaml` therefore names, per
design and with the paper section, which spatial containers *are* the NoC
(`noc_levels`); the rest receive **explicit zero** terms, and `validate`
rejects a name that is not a spatial container of the design. Explicit,
because omission is not enough: `SIMD` is nested inside `PE`, and the front-end
propagates `PE`'s `router_energy` onto the level it synthesises for `SIMD` —
with no attributes of its own, `inter_SIMD_spatial <==> mac` was still billed
750 fJ/compute (smoke run 3, identical totals to run 2). Removing the
SIMD charge puts v2 at roughly 11–15% on the dev layers — above the paper's
6–10%, in the direction a dense model should sit relative to a share measured
on sparse runs of a sparsity-aware design.

**Smoke result with the FIRST-PASS model** — wire 0.4, routers 0.25 per
operand at both v2 levels; **superseded by the second pass below** (resnet18,
the recorded dev layer pair, `random_pruned` / search 20000 / victory 2000, 18
threads; SLURM job 41294319). Timeloop totals, no ECC parity:

| arch | pre-NoC | with NoC | NoC | share | networks charged |
|---|---|---|---|---|---|
| Eyeriss v1 | 390.11 µJ | 403.34 µJ | 20.50 µJ | **5.1%** | `inter_PE_column_spatial`, `inter_PE_spatial` |
| Eyeriss v2 | ~402 µJ | 464.95 µJ | 71.54 µJ | **15.4%** | `inter_PE_cluster_spatial`, `inter_PE_spatial` |

Per compute on layer4.1.conv2: v1 69 + 73 fJ, v2 413 + 166 fJ, against MACs
of 1136 / 1169 fJ. v2's `noc_share_within_published_band_eyeriss_v2` check
records FAIL at 15.4% vs 6–10%: above the band, as a dense model should be
relative to a share measured on sparse runs, but the size of the excess is the
one thing in this model that only the full-model run and an `ECC_NOC_SCALE`
sweep can settle. DRAM energy is unchanged by the NoC (171.6 / 172.6 µJ), i.e.
the mapper kept its DRAM-level tiling and re-optimised below it. On these two
layers v2 stays behind v1 and the gap widens (+3.4% vs +15.6%); this is a
two-layer development number and not a ranking.

**The least-defended number.** `router_pj = 0.25` for v2 and Simba-like is an
assumption, not a measurement: no paper publishes a pJ/router for either NoC.
It is set so a one-router delivery costs ~60% and a two-router path ~125% of
v1's calibrated multicast-controller switching, i.e. router- and
controller-based designs are charged switching of the same order. The 6–10%
band check is its only external check; `ECC_NOC_SCALE` sweeps it.

### Second pass (2026-09-07): why 15% was still too high, and what was done

Decomposing v2's 15.4% on layer4.1.conv2 (`Networks` stats, wire vs router
computed from Timeloop's own formula): **wire 49.3 µJ, router 17.5 µJ** of
66.8. Three things were wrong, in decreasing order of size:

1. **The wire constant.** 0.4 pJ/bit/mm is a repeated-*global*-wire figure.
   Keckler/Dally's Table 1 gives 240 fJ/bit/mm *per transition* and
   310 pJ for 256 bits × 10 mm, i.e. **0.12 pJ per bit moved per mm** at
   40 nm. Adopted study-wide; v1's calibrated ingress re-splits to 0.9152 pJ
   so its inter-PE transfer stays at its published 2× MAC.
2. **Routers charged at both spatial levels, per 8-bit operand.** The paper's
   routers are statically configured muxes in the *router cluster*, with
   24-bit ports carrying three operands, and they "connect to one row of PEs
   within the cluster". Now: one shared 0.25 pJ **per flit**, divided by the
   design's published operands-per-flit (v2: 3), charged only at
   `PE_cluster`; the intra-cluster `PE` level is wire only. `noc_levels` is
   now a per-level mapping in `noc.yaml`.
3. **Cluster geometry.** The YAML flattens the paper's 8×2 cluster array to a
   16-wide row; Timeloop injects at the array edge, so a unicast averages
   7.5 hops instead of ~4. Tested, **not adopted** — see below.

**Ablation** (dev layer pair, resnet18, `random_pruned`/20000/victory 2000;
Timeloop totals without parity; item 2 applied throughout):

| variant | v1 total / NoC / share | v2 total / NoC / share | v2 band |
|---|---|---|---|
| first pass: wire 0.4, routers both levels per operand | 403.3 / 20.5 / 5.1% | 465.0 / 71.5 / **15.4%** | FAIL |
| routers cluster-only per flit, wire 0.4 | 403.3 / 20.5 / 5.1% | 452.2 / 58.8 / 13.0% | FAIL |
| + wire 0.2 | 399.4 / 16.6 / 4.1% | 426.0 / 32.6 / 7.6% | PASS |
| **+ wire 0.12 (adopted)** | **397.8 / 15.0 / 3.8%** | **415.5 / 22.1 / 5.3%** | below |
| + wire 0.12, router 0 (floor) | 397.8 / 15.0 / 3.8% | 409.1 / 15.7 / 3.8% | below |
| 8×2 clusters, wire 0.4 | — | 466.3 / 23.8 / 5.1% | — |
| 8×2 clusters, wire 0.12 | — | **446.1** / 18.0 / 4.0% | — |

v1 barely moves because its inter-PE transfer is anchored to its paper by
construction; only the wire/switching split shifts. v2 is wire-driven, so the
constant is the lever.

Confirmation run under the adopted defaults (SLURM job 41295428, default
slug, results in `results_noc_final/`): v1 397.83 µJ / NoC 14.99 µJ / 3.8%,
v2 415.51 µJ / NoC 22.10 µJ / 5.3% — identical to the ablation row, as the
arch text and mapper settings are the same. With external BCH parity: v1
446.3 → recon 388.8 µJ (−12.9%), v2 465.3 → 410.7 µJ (−11.7%). v2 trails v1
by 4.4% on this layer pair (pre-NoC: ~3%); a two-layer development number,
not a ranking.

The three generic dataflows under the same defaults (job 41295683, results in
`results_noc_generic/`; single 16×16 `PE` level, wire + calibrated ingress, no
routers): WS 506.6 µJ / NoC 39.2 / **7.7%**, OS 748.5 / 44.6 / **6.0%**,
IS 536.8 / 25.7 / **4.8%**. Their NoC is the only interconnect charged on a
2-D spatial level in this study, and Timeloop's hop model is visibly at work
there: a mapping that uses 16 of the 256 PEs pays 15 hops per unicast and
34–58 per broadcast, because the PEs it picked are spread across the array.

**Why 8×2 was not adopted.** It cut v2's NoC by 18% but *raised the total* by
7% (415.5 → 446.1 µJ) with identical DRAM weight reads — at wire 0.4 the
Global buffer went 31.6 → 81.1 µJ with 3.85% utilisation on layer4.1.conv2,
at wire 0.12 DRAM went 172.6 → 208.3 µJ. The 2-D mesh changes the mapspace
(a free X/Y split), and under the same 20 000-mapping budget the mapper found
worse tilings than it saved on wire. Adopting it would have confounded a NoC
correction with search quality, and changed the mapper's problem relative to
every other design. The 1-D row stays, its unicast over-count is recorded here
(small at 0.12), and the geometry is worth revisiting with a converged search
or with the 16 GLB clusters modelled as distributed (the deeper over-count:
Timeloop has one GLB at the array edge, the chip has one per cluster).

**Reading v2 at 5.3% against the paper's 6–10%.** Two biases oppose: the
paper's share is for sparse runs (fewer transfers → ours should be higher) at
65 nm (wire energy per bit-mm falls with node: 240 → 150 → 115 fJ across
Keckler's generations → ours should be lower). A dense 45 nm share a few
points either side of the band is consistent with the paper; 15% and 30% were
not. The band check stays as written; its detail now says how to read it. This also means **adding NoC cost
changes which mapping wins**, as intended — it is not a post-hoc correction and
cannot be applied to cached mappings. The cache slug gains `noc`; nothing
pre-NoC is ever read back.

**Category change.** `NoC` is its own plotted category ("NoC / interconnect",
grey, between On-chip SRAM/RF and Compute). The amber category's internal key
was renamed `Local (spads/RF/NoC)` → `Local (spads/RF)`; CSV columns follow.
The recon arm scales the weight share of NoC energy by K/N like the other
on-chip categories (decision 2026-09-07), recorded as an approximation on every
result.

## 14. Task 2 — embedded ECC, DRAM energy effect only (2026-09-07)

**Verdict.** Implemented as an evaluation-only experiment on the Task 1
mappings. Only the DRAM component differs between the two arms, and every
result file checks that rather than asserting it. Whole-model savings at
BCH(63,51) are 2.8–5.6% (resnet18) and 2.6–4.1% (mobilenet_v2), equal to
`parity / (Timeloop + parity)` on each architecture, because the embedded arm
removes the external-parity traffic and nothing else.

### The two questions asked before the work

*Does the mapper have to be re-run because the DRAM footprint differs between
the arms?* No. Timeloop maps 8-bit weights against the architecture; the
external parity is added afterwards, in evaluation (`ecc.external_parity()`),
and the embedded arm changes nothing the mapper sees either — the weights are 8
bits wide in DRAM and on chip under both arms. The fingerprint is unchanged,
every mapping is a cache hit (`evaluation_only_rerun: PASS`, 0 newly mapped
shapes on every file), and the plan requires the fixed mapping for this
comparison anyway. A parity-aware mapper could in principle trade a little more
on-chip reuse for fewer DRAM weight reads under the baseline's 1.3125× weight
traffic cost; that is a Task 4-class question, not a Task 2 one.

*Is the external parity already in the baseline's DRAM?* Yes, since Task 1:
6 whole weights per BCH(63,51) codeword, 3 message-padding bits, tail padding,
× the measured refetch, rounded to 64-bit DRAM words — 31.25% over payload
traffic, 10.5 stored bits per weight. Note this is *more* than the flat n/k the
Task 2 request described (8 · 63/51 = 9.88 b/weight); the difference is the
whole-weight packing rule, which is correct for a separately stored codeword.

### The actual embedded-codeword layout

Read from the embedding code rather than chosen. `ECC-CODE-Engine/4-EmbeddingECC/ecc_embed.py`
(approach `replace`, the copy in `../Input_Embedding/3-Testing/{utils,implementations}`
is verbatim):

* `convert_to_binary(vals, bit_size=8)` — one MSB-first bit stream, 8 bits per weight.
* `messageSliceBasedOnChunkSize(bits, chunk_size=n)` — fixed **63-bit chunks
  for every K**; a chunk spans several weights and slices them mid-value; only
  the final chunk is zero-padded.
* `ParityOverwriteByTopWeightsEncode(chunk, n, k)` — systematic BCH(n,k); the k
  highest-significance chunk positions are the message, the n−k lowest are
  overwritten with parity.

Consequences (`eccenergy/embedded.py`, checked by `test_embedded.py`):
7.875 weights per codeword; gcd(63,8)=1 so the alignment repeats every 8
codewords = 63 weights and 7 of 8 codeword boundaries split a weight; storage
is exactly 8 bits per weight plus ≤62 bits of tail padding per tensor; there
is no external parity and no shortened code. Parity bits per weight by
significance: BCH(63,51) → bit0 1.000, bit1 0.524 (1.524 total); BCH(63,30) →
bits 0–3 every weight, bit4 0.19. That table is what "do not treat all full
weight bits as independent payload" means in numbers, and Task 3 will need it.

The baseline's layout (`parity.py`) is the opposite rule — whole weights in a
separately stored codeword — and both are right for what they describe. The
result JSON says so under `comparison_to_baseline_layout`.

### What the embedded arm's DRAM energy is, and is not

The complete codeword is read for correction, so **every one of the 8 bits of
every DRAM weight read is billed** — Timeloop already does exactly that — and no
K/N reduction is applied at DRAM (`embedded_reads_complete_codeword`). The
external-parity term is zero (`external_parity_energy_pJ: 0.0`, written
explicitly). Stored bits, physical reads and energy are reported separately:
the physical DRAM word count is an *estimate* (bits/64 rounded up once) because
Timeloop bills per 8-bit scalar, and the tail padding is reported but not
charged, the same rule the baseline's payload is billed by. Both are in
`approximations` on every file. ECC codec energy is charged to neither arm and
the file says so; the arms would decode different codeword counts (1.95 M vs
1.48 M on resnet18), so a codec cost would not cancel.

### Results (NoC costed, converged search, BCH(63,51), 8-bit weights)

Whole model, µJ. "conv." is Task 1's conventional-ECC total; these are the
2026-09-07 NoC-costed totals, not the §13 pre-NoC ones.

| architecture | resnet18 conv. | parity | embedded | saving | mobilenet_v2 conv. | parity | embedded | saving |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eyeriss_v2_like | 5,075.371 | 276.997 | 4,798.375 | 5.46% | 1,862.971 | 70.591 | 1,792.380 | 3.79% |
| eyeriss_like | 5,357.602 | 298.623 | 5,058.978 | 5.57% | 1,984.785 | 81.692 | 1,903.093 | 4.12% |
| simba_like | 5,699.198 | 297.968 | 5,401.230 | 5.23% | 1,996.279 | 71.388 | 1,924.891 | 3.58% |
| simple_input_stationary | 6,647.611 | 271.572 | 6,376.039 | 4.09% | 2,102.472 | 71.123 | 2,031.349 | 3.38% |
| simple_weight_stationary | 7,782.205 | 249.544 | 7,532.661 | 3.21% | 2,348.220 | 79.359 | 2,268.861 | 3.38% |
| simple_output_stationary | 9,363.620 | 260.121 | 9,103.499 | 2.78% | 2,639.039 | 69.395 | 2,569.644 | 2.63% |

resnet18 footprint (identical on every architecture, as it must be):
11,678,912 weights; conventional 15,328,593 B stored in 1,946,488 codewords
(10.5 b/weight); embedded 11,679,027 B in 1,483,051 codewords (8.0001
b/weight; 115 B of tail padding across 21 tensors). mobilenet_v2: 4,554,073 B
vs 3,469,946 B. DRAM weight reads are identical between the arms on every
file (resnet18 on v2: 13,849,792 scalars, refetch ×1.19).

Development pair (resnet18, `layer3.0.downsample.0` + `layer4.1.conv2`,
2,392,064 weights): conventional 440.5 / 445.0 / 493.5 / 563.9 / 551.6 /
762.1 µJ and savings 11.0 / 10.9 / 9.7 / 8.5 / 8.8 / 6.3 % for v2 / v1 /
Simba-like / IS / WS / OS. The pair is weight-dominated, which is why its
saving is twice the whole-model figure; it validates the implementation and
ranks nothing (§13).

The saving orders the architectures by DRAM weight-traffic share, as it
should: the Eyeriss designs and Simba-like refetch most on resnet18
(×1.19–1.28) and save most; output-stationary refetches least (×1.00–1.11)
and saves least.

### What was verified

* `bash run.sh baseline --eval`, re-run after the change on all 12 whole-model
  (architecture, model) pairs, reproduces the pre-change files exactly: total,
  per-component energies, parity accounting (stored + traffic), validation
  outcomes, mapping ids and summary rows. `parity.py`, `baseline.py`,
  `build_stacks()` and `external_parity()` have no diff.
* On every Task 2 file the six Task 2 checks and all Task 1 checks pass; the
  single failing entry is the pre-existing `noc_share_within_published_band_eyeriss_v2`
  (5.3% against 6–10%), which Task 1's files fail identically.
* `embedded_matches_sweep_figure_arm` passes everywhere: the figure's embedded
  bar (`build_stacks()`, unchanged, = `raw.total`) and the JSON agree.
* Tests: `test_embedded` 11/11 (two need pandas and run in the container),
  `test_results_store` 15/15, `test_noc` 12/12, `test_mapper_lock` 4/4.

### How it was produced

```bash
module load apptainer
export ECC_MAPPER_ALGORITHM=random_pruned ECC_MAPPER_SEARCH_SIZE= ECC_VICTORY=2000 \
       ECC_MAPPER_TIMEOUT=2000 ECC_MAPPER_THREADS=18 ECC_SWEEP=arch \
       ECC_SWEEP_ARCHS="eyeriss_v2_like eyeriss_like simple_weight_stationary simple_output_stationary simple_input_stationary simba_like"
ECC_LAYERS= ECC_CONST_MODEL=resnet18     bash hpc/tl.sh bash run.sh embedded --eval
ECC_LAYERS= ECC_CONST_MODEL=mobilenet_v2 bash hpc/tl.sh bash run.sh embedded --eval
ECC_LAYERS="layer3.0.downsample.0 layer4.1.conv2" ECC_CONST_MODEL=resnet18 bash hpc/tl.sh bash run.sh embedded --eval
python3 hpc/summary.py --scope layers-full --field saving --csv results/tables/panel_matrix.csv
```

Result files: `results/evaluation/Pre/<arch>/<model>/bch63_51/w8a8/{layers-full,layers2__…}/map-energy-vic2000l-random_pruned-seednone-ssnone-perm16/2026-09-07T1942*.json`
(experiment `task2_embedded_ecc_dram_only`; `latest.json` points at them).

### Caveats that still stand

* Fixed mapping, DRAM effect only: no on-chip saving is credited to the
  embedded arm (that is Tasks 3–5), and no accuracy claim is made about
  overwriting weight LSBs with parity — that is the Input_Embedding study's
  question.
* `build_stacks()` still counts the embedded arm's *decodes* at the baseline's
  6 weights/codeword (`ECC_EMB_WEIGHTS_PER_CW`). Decode is off by default and
  outside this comparison; when codec energy is adopted, that count should
  move to the 7.875-weights layout and the decision be recorded.
* The whole-model numbers here are on six architectures without the `_wglb`
  bracketing pair, at the NoC-costed setting; the §13 caveats on convergence,
  the missing six models and Simba-like's provenance all still apply.
