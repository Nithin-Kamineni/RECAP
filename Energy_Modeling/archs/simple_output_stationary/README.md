Simple output-stationary — paper-fidelity overlay
=================================================

Only `arch_paper.yaml` lives here. At `ECC_ARCH_FIDELITY=stock` the design falls
back to `example_designs/simple_output_stationary/arch.yaml`, unmodified.

**There is no paper.** This is the textbook output-stationary reference that
ships with timeloop-accelergy-exercises, so "paper fidelity" here means making
the design internally consistent with the study it is used in, not matching a
published table.

Identical treatment to `../simple_weight_stationary/` — read that README for the
reasoning. The short version: the stock file declares `datawidth: 16` on every
level while declaring an 8-bit multiplier, so in an 8-bit-weight study it paid
128 pJ per DRAM weight read against eyeriss's 64. The shared buffer is split by
dataspace so operands can be 8-bit without billing partial sums as bytes:

| level | stock | here |
|---|---|---|
| DRAM | dw 16 | dw 8 |
| shared_glb, 128 kB, dw 16 | one level, all dataspaces | `operand_glb` 64 kB dw 8 + `psum_glb` 64 kB dw 16 |
| pe_spad (keeps Outputs) | dw 16 | dw 16 — **unchanged**, this is the accumulator |
| weight_reg / input_activation_reg | 16b | 8b |
| output_activation_reg | 16b | 16b (unchanged: it holds a psum) |

The one difference from the weight-stationary case: here the PE scratchpad is
the output-stationary accumulator, so it was already correct at 16 bits and is
left alone. `_force_datawidth()` in `archs.py` skips it for the same reason —
any level whose `keep:` list is `Outputs` alone.

Consequence: `ECC_FORCE_DATAWIDTH=8` is a **no-op** on this design at paper
fidelity.

Caveat that has NOT changed
---------------------------

Like the weight-stationary design, this one keeps Weights in the global buffer
where eyeriss v1, v2 and simba bypass them — one more level of weight reuse,
and the largest single driver of the DRAM weight traffic this ECC study
measures. A real architectural difference, not a defect, but it has to be
stated whenever the baseline arm's N/K DRAM inflation is compared across
architectures.
