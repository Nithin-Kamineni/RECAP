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
| shared_glb, 128 kB, dw 16 | one level, all dataspaces | `input_glb` 28 kB dw 8 + `weight_glb` **24 kB** dw 8 + `psum_glb` 64 kB dw 16 — **116 kB, not 128**: see the note below |
| pe_spad (keeps Weights) | dw 16 | dw 8 -> 384 weights |
| weight_reg / input_activation_reg | 16b | 8b |
| output_activation_reg | 16b | 16b (unchanged: it holds a psum) |

Total GLB capacity and word width are unchanged (36 + 28 + 64 = 128 kB) — only
the packing and the partitioning change.

### Weights and inputs are separate levels (2026-09-13)

The 64 kB `operand_glb` this replaces held Inputs **and** Weights, and Timeloop
allows one `datawidth:` per storage level. So it could not store the
reconstruction arm's q-bit weights without also declaring the 8-bit input
activations to be q bits — and under the default
`ECC_WEIGHT_CAPACITY_SCOPE=exclusive` it was not a weight level at all
(`archs._weight_level_parts()` skips a level whose `keep:` is not Weights
alone), so THE WIDTH TABLE never reshaped it and there was no level for
`ECC_WEIGHT_DATAWIDTH_LEVELS` to name. `weight_glb` keeps Weights and nothing
else and is the design's global weight-buffer stage — `recon.py`'s R2 boundary
— in the same sense `filter_glb` is Eyeriss v1's.

**The 36/28 kB ratio is not cited and there is no paper to cite.** It comes from
this design's own cached mappings — 53 distinct layer shapes at the 65,536-word
shared `operand_glb`:

| | Weights | Inputs |
|---|---:|---:|
| summed partition size (values the level must hold) | 8,068,576 (53.3 %) | 7,064,919 (46.7 %) |
| refetch (fills / partition) | 1.088x | 1.113x |
| peak utilized capacity | 64,000 words | 57,344 words |
| median utilized capacity | 4,320 words | 25,056 words |

Both dataspaces ask for the level in almost equal measure and at 64 kB shared
**neither binds**. The demand split (53/47) is the base; the tilt to 56/44 is
the dataflow — weights are the resident operand here, and this design's low DRAM
weight refetch is the quantity the ECC study measures. The tilt is one bank-pair
wide and not a half because the counter-argument is real: the input path has no
PE scratchpad (inputs go GLB -> `input_activation_reg`, depth 1), so this level
is the only input reuse level on chip and takes 90 % of its reads.

**What it costs.** Both halves are smaller than the 64 kB either dataspace could
reach before, so fills — DRAM traffic — can only rise on both, and the
peak-demand layers re-tile.

Splitting one 64 kB array into a 24 kB and a 28 kB one also gives CACTI two
smaller arrays, so **per-access energy on the operand path is no longer the
stock number**. Measured (Accelergy, 45nm, reference arm, per 8-bit value so
the word widths compare):

| | read | write | leak, pJ/cycle |
|---|---:|---:|---:|
| `operand_glb`, pre-split (8192x64b, 16 banks) | 2.2821 | 2.6656 | 0.0025893 |
| `input_glb` (3584x64b, 8 banks) | 1.6305 (−28.6 %) | 1.7167 (−35.6 %) | 0.0010206 |
| `weight_glb` (768x384b, 8 banks) | 2.1394 (−6.3 %) | 2.3074 (−13.4 %) | 0.0020248 |

Dynamic energy per value **falls** on both dataspaces — smaller arrays are
cheaper per access and per bit — while leakage **rises** ×1.177, because two
arrays carry two sets of peripherals. `../_shared/provenance.yaml` records
this rather than compensating for it.

`n_banks` is 8 on both halves, conserving the 16 operand banks and pricing both
on the same banking model; entries per bank move (384 and 448) instead of the
bank count. `weight_glb`'s 384-bit word is THE WIDTH TABLE's, not a choice made
here: `archs._set_weight_geometry()` reshapes every weight level per arm and
renormalises depth to hold the declared 196,608 bits.

**`weight_glb` IS 24 kB, NOT THE 36 kB THE SPLIT WAS DERIVED FOR.** It was
reduced from `depth: 4608` to `depth: 3072` by hand on 2026-09-15 with no
reason recorded, and every number in this repository is computed at the new
value. The operand split therefore no longer preserves the stock 128 kB
(24 + 28 + 64 = 116 kB) — which is the whole rationale `provenance.yaml` gives
for splitting the shared buffer. **A one-line reason is still owed here**;
until it is written the reduction is a divergence, not a modelling choice.

**This design no longer brackets.** `ECC_WEIGHT_CAPACITY_SCOPE=exclusive` and
`shared` now give it the same answer — it has no level holding Weights beside
another dataspace. Its two siblings, `simple_output_stationary` and
`simple_input_stationary`, still declare one and are what the `shared` scope is
for.

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
