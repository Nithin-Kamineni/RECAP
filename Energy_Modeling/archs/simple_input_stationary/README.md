# simple_input_stationary

**Authored in this repository on 2026-09-06. It reproduces no paper.**

`timeloop-accelergy-exercises` ships `simple_weight_stationary` and
`simple_output_stationary` but no input-stationary reference design, and
section 9 of `01_project_context_and_architectures.txt` describes the
input-stationary *dataflow* rather than any particular chip. So there was
nothing to copy and nothing to cite. This file is a controlled sibling of
the two designs that do ship, not a claim about hardware anyone built.

## What is identical to its two siblings, and why

Everything except which operand the PE scratchpad keeps:

| | weight-stationary | output-stationary | input-stationary |
|---|---|---|---|
| PE mesh | 16 x 16 | 16 x 16 | 16 x 16 |
| operand GLB | 64 kB, 8192 x 64b, 16 banks, 8b | same | same |
| psum GLB | 64 kB, 8192 x 64b, 16 banks, 16b | same | same |
| `pe_spad` | 192 x 16b, **keeps Weights** | 192 x 16b, **keeps Outputs** | 192 x 16b, **keeps Inputs** |
| registers | W 8b / I 8b / O 16b | same | same |
| MAC | 8b mult, 16b add | same | same |

Holding all of that fixed is the point. The three designs are a controlled
comparison of *which operand is stationary*; if their capacities or word
widths also differed, no difference between them could be attributed to the
dataflow.

## How the dataflow is expressed to Timeloop

Two constraints do the work, and they have to agree with each other:

* `PE.spatial.factors: M=1` — output channels may **not** be spread across
  the array. Whatever else the mapper does, it cannot turn this into a
  weight- or output-stationary design by fanning filters out spatially.
  `C` goes to `meshX` and `P`,`Q` to `meshY` (`split: 1`) instead, because
  those are the dimensions that say *which input activation* a PE holds.
* `pe_spad.temporal.permutation: [M, R, S, P, Q]` — `M` innermost, so a
  resident activation is consumed by many filters before the scratchpad is
  refilled. That is the temporal half of "input stationary".

## What it is for in this study

Input-stationary gives weights **no local reuse**: every filter crosses the
whole hierarchy once per resident activation tile. It is therefore the
architecture where weight traffic is largest relative to everything else,
which makes it the useful extreme case for a study whose protected payload
*is* the weights. Expect it to look bad on total energy and interesting on
ECC-parity share; both are results, not defects.

## Caveats

* `M=1` spatially means the array is tiled over `C` (up to 16) and `P`x`Q`
  (up to 16). Layers with a small spatial output and few channels will
  under-fill the mesh. That is a real property of the dataflow, not a
  modelling artefact, but it means utilisation is not comparable to the
  weight-stationary sibling layer for layer.
* Like every design here it is modelled at 45 nm. See
  `archs/_shared/standard.yaml`.
