Eyeriss v2 Architecture
=======================

A Timeloop/Accelergy model of **Eyeriss v2**, from:

> Y.-H. Chen, T.-J. Yang, J. Emer, V. Sze,
> *"Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on
> Mobile Devices"*, IEEE JETCAS 9(2), 2019. arXiv:1807.07928

It is written to slot into the same flow as `eyeriss_like`,
`simple_weight_stationary`, `simba_like` and `simple_output_stationary`:
same `_components/`, same `_include/mapper.yaml`, same dense 8-bit problems.

Files
-----

| file | use |
|---|---|
| `arch_paper.yaml` | **the one to use.** What `ECC_ARCH_FIDELITY=paper` (the default) maps. Identical to `arch.yaml` except the weight spad is `width: 24` — see *Dense-equivalent scratchpads*. |
| `arch.yaml` | the original reading, kept so `ECC_ARCH_FIDELITY=stock` reproduces every figure made before that correction. |
| `arch_physical_8x2_3x4.yaml` | optional variant with the literal 8x2 cluster / 3x4 PE meshes. See *Which arch.yaml* below. Still at the `width: 16` weight spad. |

Both parse and pass Timeloop's full front-end processor chain
(`Specification.from_yaml_files(...)._process()`).

Paper -> model
--------------

| quantity | paper | this model | note |
|---|---|---|---|
| technology | 65 nm | `technology: "45nm"` | **model deviates from the paper**, matching `eyeriss_like` and the other example designs so cross-architecture gaps are architecture, not silicon. Override with `ECC_FORCE_TECHNOLOGY`. |
| PEs | 192 = 16 clusters (8x2) x 12 PEs (3x4) | `PE_cluster` x16, `PE` x12 | |
| MACs / PE | 2 (SIMD-2) | `SIMD` container, meshX 2 | total fanout 384 |
| peak throughput | 153.6 GOPS @ 200 MHz | 384 MAC x 2 op x 200 MHz = 153.6 G | **consistency check passes** |
| iact / weight precision | 8b | `datawidth: 8` | |
| psum precision | 20b | `datawidth: 20`, `adder_width: 20` | |
| GLB total | 192 kB | 72 + 120 kB | |
| iact GLB | 16 clusters x 3 banks x 1.5 kB = 72 kB | `iact_glb` depth 9216 x 64b, `n_banks: 48` | 73728 iacts |
| psum GLB | 16 clusters x 4 banks x 1.875 kB = 120 kB | `psum_glb` depth 24576 x 40b, `n_banks: 64` | 49152 psums; 40b word = 2 psums, matching the 40b psum-router port |
| weights in GLB | **none** - DRAM -> router -> PE spad | `!Nothing` branch for `Weights` in the GLB `!Parallel` | this is the structural difference from v1 |
| iact spad | 16 x 12b | depth 16, width 8 -> 16 iacts | 4b CSC run-length field dropped, see *Dense-equivalent* |
| weight spad | 96 x 24b ("weight data: 288B (SRAM)", Table IV) | depth 96, width 24 -> **288 weights** | 24b read as 3x8b, per Sec. III-C. `arch.yaml` read it as 2x8b + 8b CSC (192) |
| psum spad | 32 x 20b | depth 32, width 20 -> 32 psums, 80 B | **exact** |
| iact addr reg | 9 x 4b | not modelled | CSC metadata |
| weight addr reg | 16 x 7b | not modelled | CSC metadata |
| dataflow | RS+ (row-stationary, any dimension tiled spatially) | see *Constraints* | |

Dense-equivalent scratchpads
----------------------------

The paper's spad widths are **compressed** widths: the 12b iact word is an 8b
value plus a 4b CSC run-length, and the separate address registers hold CSC
row pointers. A dense 8-bit workload generates none of that metadata, so only
the data bits are modelled. Consequence: this model understates iact-spad
leakage/area by roughly 1.5x versus the silicon, and omits ~22 B/PE of
metadata registers. It does **not** understate the data traffic, which is what
the energy study measures.

The 24b weight word was the one genuine interpretation, and **the paper settles
it: three bytes of payload, not two.** Sec. III-C, on the HM-NoC router ports:

> For iacts and weights, each port has a bitwidth of 24 bits such that it can
> send and receive **three 8b uncompressed** iact values or two 12b compressed
> iact run-data pairs per cycle.

and Table IV lists "weight data: 288B (SRAM)" per PE with the CSC address
vector accounted for separately as "weight addr: 14B (Reg)" = 16 x 7b. So the
288 B is payload, and the dense equivalent is **288 weights/PE** - `width: 24`,
which is what `arch_paper.yaml` uses.

`arch.yaml` reads the same word as `2 x 8b weight + 8b CSC field` for 192
weights/PE. That understated v2's on-chip weight capacity by a third
(36,864 against 55,296 across the 192 PEs) and inflated its DRAM refetch, which
was the largest single term in v2's excess energy on resnet18. It is kept only
so `ECC_ARCH_FIDELITY=stock` reproduces the earlier figures.

The iact spad is unaffected: Table IV's "iact data: 24B (Reg)" is the 16 x 12b
array *including* the 4b run-length in each word, so the dense payload really is
16 iacts, which is what `depth: 16, width: 8` already encodes.

Constraints / dataflow
----------------------

v1's contribution is the RS dataflow; v2's is that RS is *generalised* - the
paper tiles "through any layer dimension, including the channel group
dimension". That difference is expressed here **spatially**:

* `PE_cluster` (16): only `N`, `R`, `S` pinned to 1. `C`, `M`, `P`, `Q` are all
  free to tile across clusters - including `C`, which is the channel-group
  freedom v1 does not have.
* `PE` (12): `N`, `P`, `Q` pinned. `R`, `S`, `C`, `M` free - RS across PEs.
* `SIMD` (2): everything pinned except `M`, so the mapper takes `M=2` when `M`
  is even and falls back to one idle lane when it is not.

The per-spad **temporal** constraints deliberately mirror `eyeriss_like`. Intra-PE
behaviour in v2 is still row-stationary; keeping them identical means any energy
difference you measure between v1 and v2 comes from the memory hierarchy and the
spatial freedom, not from an incidental change in the temporal search space.

Which arch.yaml
---------------

`arch.yaml` flattens the 8x2 cluster array to a single 16-wide mesh and the
3x4 PE array to a single 12-wide mesh. This is **not** a simplification of the
energy model - Timeloop's example-design flow has no wire-length or topology
model, so the 8x2 shape changes nothing it computes. What the shape *would* do
is force the mapper to commit specific loop dimensions to the X axis and others
to the Y axis. Since the v2 hierarchical mesh NoC is explicitly designed so any
GLB cluster can reach any PE cluster, the flattened mesh is both the safer and
the more faithful choice.

`arch_physical_8x2_3x4.yaml` is provided if you want the literal partition
(M across the 8 cluster columns, C across the 2 cluster rows; C/M across the 4
PE columns, R/S across the 3 PE rows). It is schema-valid but **more
constrained**, so test it on a few layers before committing a long sweep - some
layer shapes may map worse or not at all.

What this tool cannot capture
-----------------------------

Reached the edge of Timeloop+Accelergy here; these are real gaps, not oversights:

1. **HM-NoC router energy is not modelled.** Per cluster the paper has 3 iact
   routers (4 ports, 24b), 3 weight routers (2 ports, 24b) and 4 psum routers
   (3 ports, 40b). The stock Accelergy plugin set has no router primitive, and
   the example-design flow instantiates no network components at all. Timeloop
   *does* model multicast/broadcast **reuse** in its access counts, so the data
   movement volume is right - only the interconnect circuit energy is missing.
   `eyeriss_like` has the same gap, so v1-vs-v2 comparisons stay fair.
2. **NoC mode switching** (high-bandwidth unicast vs high-reuse broadcast vs
   grouped/interleaved multicast) cannot be expressed. The mapper instead picks
   one spatial mapping per layer, which is a reasonable proxy: it will choose
   broadcast-like mappings for high-reuse layers on its own.
3. **CSC sparsity is not modelled.** A large share of v2's real-world win comes
   from compressing both weights and iacts. Timeloop can do sparse modelling
   (see `sparse_tensor_core_like` and `_components/*_metadata.yaml`), but it
   needs per-dataspace density specs and would make v2 incomparable to the dense
   v1 / weight-stationary baselines. Deliberately left dense.
4. **Clock rate.** `globals.yaml` sets `global_cycle_seconds: 1e-9` (1 GHz) for
   every architecture (`ECC_GLOBAL_CYCLE_SECONDS`); the real v2 runs at 200 MHz. Leakage is charged per cycle-second, so static energy here is
   understated relative to the silicon. Consistent across all archs, so relative
   comparisons hold.
5. **Mapper vs hand-mapping.** Same caveat as the `eyeriss_like` README: Timeloop
   only explores factors that divide the layer dimensions, so it will sometimes
   miss the paper's hand-tuned mapping. Raise `victory_condition` for final numbers.

Note on cross-architecture comparisons
--------------------------------------

All five architectures now declare `technology: "45nm"`, so Accelergy costs
their components at the same node. (An earlier revision of this README said
65nm; `arch.yaml` has said 45nm since the architecture was added to the sweep.)

The remaining cross-architecture confounds are NOT the node. Run
`bash run.sh diagnose` for the live list.

Two that used to be on this list are now fixed at `ECC_ARCH_FIDELITY=paper`
(the default), and both had been penalising *this* design:

* `eyeriss_like` declared ONE `shared_glb` at `datawidth: 8` holding both
  Inputs and Outputs, so Timeloop billed its 16-bit partial sums at one byte
  each - 3.6 pJ per psum against this model's honestly-declared 11.7 pJ. v1's
  GLB is now split into an 8b ifmap buffer and a 16b psum buffer.
* `simple_weight_stationary` and `simple_output_stationary` declared
  `datawidth: 16` throughout, paying 128 pJ per DRAM weight read where this
  model pays 64. Their paper-fidelity files are natively 8b operands / 16b
  psums, so `ECC_FORCE_DATAWIDTH=8` is no longer needed for them.

The mapper objective was a third bias in the same direction: it defaulted to
`edp`, which lets a design with more MACs buy latency by spending energy, and
this model has 384 MACs to v1's 168. It is now `energy`.

The pre-rewrite decomposition is archived in `../../legacy/FINDINGS.md` — its
numbers are historical and should not be quoted. See `../eyeriss_v2_like_wglb/`
for the diagnostic variant that brackets the missing weight-NoC reuse level.

Running it
----------

From `Energy_modeling/`, inside the container:

    cd /home/workspace
    ECC_SWEEP=model ECC_CONST_ARCH=eyeriss_v2_like bash run.sh

Mappings are cached in `ecc_energy_study/outputs/eyeriss_v2_like/multimodel/`,
so a cold run is hours and a warm one is seconds. Runs are resumable.
