Simple weight-stationary — paper-fidelity overlay
=================================================

Only `arch_paper.yaml` lives here. At `ECC_ARCH_FIDELITY=stock` the design falls
back to `example_designs/simple_weight_stationary/arch.yaml`, unmodified.

**There is no paper.** This is the textbook weight-stationary reference that
ships with timeloop-accelergy-exercises, so "paper fidelity" here does not mean
matching a published table — it means making the design internally consistent
with the study it is used in. `arch_paper.yaml` is the right file to read for
the reasoning; this README is the summary.

What changed
------------

The stock file declares `datawidth: 16` on every level — DRAM, the shared GLB,
the PE scratchpad and all three registers — while declaring an 8-bit multiplier
(`multiplier_width: 8`). In an 8-bit-weight study that means:

* **128 pJ per DRAM weight read**, where eyeriss and simba pay 64;
* 4 values per 64-bit buffer word instead of 8, doubling buffer accesses too.

`ECC_FORCE_DATAWIDTH=8` was the workaround. It could not do the whole job,
because Timeloop allows one datawidth per storage level and the stock
`shared_glb` holds all three dataspaces — so forcing it to 8 would have billed
the partial sums as bytes, which is exactly the defect that made
`eyeriss_like` look artificially cheap (see `../eyeriss_like/README.md`).

So the shared buffer is split by dataspace, and the convention becomes the same
one the eyeriss designs use:

* **operands** (Weights, Inputs) -> `datawidth: 8`
* **partial sums** (Outputs) -> `datawidth: 16`, the width `adder_width: 16`
  already declares

| level | stock | here |
|---|---|---|
| DRAM | dw 16 | dw 8 |
| shared_glb, 128 kB, dw 16 | one level, all dataspaces | `operand_glb` 64 kB dw 8 + `psum_glb` 64 kB dw 16 |
| pe_spad (keeps Weights) | dw 16 | dw 8 -> 384 weights |
| weight_reg / input_activation_reg | 16b | 8b |
| output_activation_reg | 16b | 16b (unchanged: it holds a psum) |

Total GLB capacity, word width and bank geometry are unchanged, so CACTI sees
the same arrays and per-access energy is the stock number — only the packing
changes.

Consequence: `ECC_FORCE_DATAWIDTH=8` is a **no-op** on this design at paper
fidelity. It is already 8-bit on the operand path.

Caveat that has NOT changed
---------------------------

This design keeps Weights in the global buffer, where eyeriss v1, v2 and simba
bypass them. That is one more level of weight reuse and it is the single
largest driver of the DRAM weight traffic this ECC study measures — 1.17
fetches per weight against Eyeriss v1's 3.47 on resnet18. It is a real
architectural difference, not a defect, but it must be stated whenever the
baseline arm's N/K DRAM inflation is compared across architectures.
