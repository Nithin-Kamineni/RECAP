Eyeriss v1 — paper-fidelity overlay
===================================

This directory holds **only** `arch_paper.yaml`. There is deliberately no
`arch.yaml`: at `ECC_ARCH_FIDELITY=stock` the design falls back to
`example_designs/eyeriss_like/arch.yaml` from the cloned exercises repo,
unmodified, so `stock` really is the shipped design.

Source for every number:

> Y.-H. Chen, T. Krishna, J. Emer, V. Sze, *"Eyeriss: An Energy-Efficient
> Reconfigurable Accelerator for Deep Convolutional Neural Networks"*,
> IEEE JSSC 52(1), 2017.
> https://eems.mit.edu/wp-content/uploads/2016/11/eyeriss_jssc_2017.pdf

What changed, and why
---------------------

### 1. The global buffer was billing 16-bit partial sums as bytes

This is the one that mattered. The stock design declares a single level:

```yaml
- !Component
  name: shared_glb
  attributes: {depth: 16384, width: 64, n_banks: 32, datawidth: 8}
  constraints:
    dataspace: {keep: [Inputs, Outputs], bypass: [Weights]}
```

`datawidth: 8` — while the same file's `psum_spad` is `datawidth: 16` and its
`mac` is `adder_width: 16`. Timeloop packs `width / datawidth` values into each
physical word, so it put **8 partial sums in every 64-bit word** and charged
each one 3.64 pJ. Eyeriss v2, which declares its 20-bit psums honestly, pays
11.70 pJ for the same scalar. A 3.2x handicap, on the dataspace that turned out
to decide the whole v1-vs-v2 comparison. The pre-rewrite derivation is
archived in `../../legacy/FINDINGS.md`; its numbers are historical.

The fix is also the more faithful structure. From Sec. V-A:

> The Eyeriss accelerator has a GLB of 108 kB ... 100 kB of the GLB is
> allocated for ifmaps and psums as required by the RS dataflow for reuse. Even
> though it is not required by the dataflow, the remaining 8 kB (two banks of
> 512-b x 64-b SRAMs) of the GLB is allocated for filter weights to compensate
> for insufficient off-chip traffic bandwidth.

> ... the space is divided into 25 banks, each of which is a 512-b x 64-b (4 kB)
> SRAM. **Each bank is assigned entirely to ifmaps or psums**, and the
> assignment is reconfigurable based on the scan chain bits.

So the chip already partitions this buffer per dataspace; it just does it at
runtime. Timeloop cannot repartition, so the 25 banks are split statically
13 ifmap / 12 psum. The bank *geometry* is exact (512 x 64b), which is what sets
per-access energy; only the ifmap:psum capacity ratio is frozen.

Total capacity drops from the stock model's 128 kB to the paper's 100 kB.

### 2. Two scratchpads were the wrong size

> ... the filter spad is implemented in a 224-b x 16-b SRAM due to its large
> size; the ifmap and psum spads of size 12 b x 16 b and 24 b x 16 b,
> respectively, are implemented using registers.

| spad | paper | stock model | here |
|---|---|---|---|
| filter | 224 x 16b | depth 192 | depth 224 |
| ifmap | 12 x 16b | depth 12 | depth 12 (unchanged) |
| psum | 24 x 16b | depth 16 | depth 24 |

Both corrections *help* v1, and they are applied for the same reason the psum
correction is: the point is the published number, not the direction it moves.

Precision convention
--------------------

Eyeriss v1 is a 16-bit fixed-point design ("The computation consists of a 16-b
two-stage [pipelined multiplier and adder] ... truncated from 32 to 16 b"), but
this study quantizes to 8-bit weights. The convention, shared with the v2 model
so the two are comparable:

* **operands** (Weights, Inputs) → `datawidth: 8`
* **partial sums** (Outputs) → `datawidth: 16`, the accumulator width
* physical `width:` unchanged, so a 16b word simply holds two 8-bit operands

The one place this is approximate is DRAM, which is `datawidth: 8` for both
eyeriss designs: Timeloop cannot tell a spilled partial sum from a finished
activation at that boundary, so both are billed as activations. Symmetric
between the designs.

What is deliberately NOT modelled
---------------------------------

* **The 8 kB filter allocation in the GLB.** It exists on the chip, but the
  paper is explicit that the RS dataflow does not need it and that its job is
  double-buffering the *next* pass's filters against off-chip bandwidth. Giving
  Timeloop a weight level there would hand v1 a full extra stage of weight
  reuse it does not actually get. Weights go DRAM → filter spad, as in the
  stock model.
* **The NoC.** No router primitive exists in the example-design flow. Timeloop
  still counts multicast reuse, so the data volume is right and the omission is
  uniform across every architecture here — including v2, which has far more
  interconnect to not be charged for.
* **The RLC codec** and the zero-gating that skips filter-spad reads on zero
  ifmaps. Both are sparsity features, and this study is dense.
* **65nm.** `technology: "45nm"`, matching the other example designs so a
  cross-architecture gap is architecture rather than silicon.
  `ECC_FORCE_TECHNOLOGY` overrides it.
