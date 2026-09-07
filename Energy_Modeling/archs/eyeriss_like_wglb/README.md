# eyeriss_like_wglb — Eyeriss v1 with its published 8 kB filter GLB

A variant of `eyeriss_like` that differs from it in **exactly one block**: the
Weights branch of the global-buffer `!Parallel`. Where `eyeriss_like` puts
`!Nothing`, this file puts the 8 kB the paper says the GLB allocates to filter
weights. Diff the two `arch_paper.yaml` files and that block plus the header
comment is the whole difference.

## Why it exists

`eyeriss_like` refetches resnet18's weights **7.0×** from DRAM. Every other
design in the study is between 1.0× and 2.3×. The ECC saving this project
measures is very nearly

    saving ≈ 0.3125 · E_dram_weights / (total + 0.3125 · E_dram_weights)

so that one number produces the largest saving in the whole study. It is worth
knowing how much of it is architecture and how much is a modelling choice.

The choice is this. JSSC 2017 Sec. V-A says, verbatim:

> Even though it is not required by the dataflow, the remaining 8 kB (two banks
> of 512-b × 64-b SRAMs) of the GLB is allocated for filter weights to
> compensate for insufficient off-chip traffic bandwidth. While the PE array is
> working on a processing pass, the GLB preloads the filters used by the next
> processing pass.

`eyeriss_like` does not model it, arguing that a double-buffered *prefetch*
store is not *reuse*: it hides DRAM latency but does not by itself reduce DRAM
weight traffic, and declaring it as a Timeloop storage level would hand v1 a
reuse stage the row-stationary dataflow does not have.

That argument is reasonable. It is also **unfalsifiable from the paper alone**,
and it is **not neutral** — it maximises exactly the quantity the study
measures, on the design that shows the biggest effect. One file should not
silently settle a question like that.

## How to use the pair

| | models the 8 kB as | gives |
|---|---|---|
| `eyeriss_like` | nothing | **upper bound** on DRAM weight traffic, and so on the ECC saving |
| `eyeriss_like_wglb` | a full reuse level | **lower bound** |

The truth is between them: a prefetch buffer captures *some* reuse when the
same filters serve several processing passes, and none when they do not.

**Quote the pair, never one alone.** A single Eyeriss v1 saving number from
this study is a choice of bound, not a measurement.

## Difference from `eyeriss_v2_like_wglb`

They look alike and are not the same kind of object.

* `eyeriss_v2_like_wglb` adds a weight level that is **not in the v2 paper** —
  a stand-in for the weight routers, which Timeloop cannot express. It is a
  diagnostic; its capacity was chosen, not cited.
* `eyeriss_like_wglb` adds a weight level that **is in the v1 paper**, at the
  published capacity (8 kB) and the published bank geometry (2 × 512 × 64b).
  What is uncertain here is not the structure but whether prefetch counts as
  reuse.

So this variant is arguably *more* faithful to its paper than the base design,
which is the opposite of the v2 case.

## No `stock` fidelity

This directory has `arch_paper.yaml` and no `arch.yaml`, and
timeloop-accelergy-exercises ships no counterpart, so
`ECC_ARCH_FIDELITY=stock` has nothing to select and the run will fail on a
missing file. That is correct: there is no "as shipped" version of a variant
that does not exist upstream, and inventing one would mean inventing a design
point. `simple_input_stationary` is in the same position for the same reason.
Use the default `paper` fidelity.

## Provenance

Every other number is inherited from `archs/eyeriss_like/` — see that
directory's README and `archs/_shared/provenance.yaml`, which records this
level's quote, its modelled geometry, and why it matters.
