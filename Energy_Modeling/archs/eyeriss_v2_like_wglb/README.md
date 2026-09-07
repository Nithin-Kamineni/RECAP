# eyeriss_v2_like_wglb — a diagnostic variant, not a second model of the paper

`eyeriss_v2_like` is the faithful model: in Eyeriss v2 weights have **no global
buffer level**. They stream DRAM → weight router → PE weight scratchpad, and the
stock model encodes that with a `!Nothing` branch for `Weights` in the GLB
`!Parallel` block.

That faithful model produces a result that reads as wrong: on dense resnet18 it
refetches every weight **7.32 times** from DRAM, against Eyeriss v1's 3.47, and
so lands 47% above v1 in total energy. This variant exists to test **why**.

## What it changes

One thing. The `!Nothing` weight branch is replaced by a 64 kB storage level
named `weight_noc` that keeps `Weights`.

## What that level represents

Not a buffer the silicon has. It stands in for the part of v2 that
Timeloop's dense example-design flow cannot express:

> Per GLB cluster the paper has **3 weight routers (2 ports, 24 b)**, and the
> hierarchical mesh NoC is explicitly designed so one fetched weight can be
> broadcast or multicast to many PE clusters.

Timeloop models multicast reuse only across a *spatial* fanout under a shared
parent. With `!Nothing`, the weight spads' parent is DRAM, so **temporal** reuse
across the outer loop nest has nowhere to live: each outer iteration refetches
the weight tile from DRAM. Inserting a storage level gives that reuse somewhere
to go, which is the closest available proxy for the router network.

It is sized at 65,536 8-bit weights — deliberately close to `eyeriss_like`'s
shared GLB weight capacity — so the comparison isolates *whether the reuse level
exists*, not *how much capacity it has*.

## How to read a number from it

As a bound, not a measurement.

- `eyeriss_v2_like` is the **pessimistic** bound: v2 with none of its reuse
  machinery modelled, on a dense workload it was never designed for.
- `eyeriss_v2_like_wglb` is the **optimistic** bound: v2 with the weight NoC
  credited as a full reuse level, and still with no router circuit energy
  charged against it.

The truth for a dense workload is between them. Quote the pair, or quote
`eyeriss_v2_like` and say what it omits — do not quote this variant alone as
"Eyeriss v2".

## Running the comparison

```bash
ECC_SWEEP=arch \
  ECC_SWEEP_ARCHS="eyeriss_like eyeriss_v2_like eyeriss_v2_like_wglb" \
  ECC_CONST_MODEL=resnet18 bash run.sh
```

The variant has no mapper cache, so the first run maps every layer shape from
scratch. resnet18 alone is 12 distinct shapes — minutes to a couple of hours,
not the multi-hour full sweep. It is resumable.

## What still is not modelled, in either

Unchanged from `../eyeriss_v2_like/README.md`, and these are the real reasons a
dense Timeloop model understates v2:

1. **CSC sparsity.** v2 compresses both weights and activations; a large share
   of its real-world win is traffic that never happens. Modelled dense here so
   that v1, v2 and the stationary baselines stay comparable.
2. **Router circuit energy.** Missing in every architecture in this project, so
   comparisons stay fair — but it means this variant gets v2's reuse benefit
   without paying v2's interconnect cost.
3. **NoC mode switching** between high-bandwidth unicast, high-reuse broadcast
   and grouped multicast. The mapper picks one spatial mapping per layer instead.
4. **Clock rate.** `globals.yaml` runs every architecture at 1 GHz; real v2 runs
   at 200 MHz, so leakage is understated. Uniform across architectures.

There is also a genuine architectural reason v2 looks worse dense: **v2's weight
scratchpad is smaller than v1's** — 96x24b against v1's 224x16b, which this
project models as 192 dense weights per PE against 384. v2 spent that area on
compression logic and the NoC. Dense modelling charges v2 for the smaller spad
and credits it for none of what the area bought.

Paper fidelity
--------------

`arch_paper.yaml` is what `ECC_ARCH_FIDELITY=paper` (the default) maps. It is
generated from `../eyeriss_v2_like/arch_paper.yaml` with the `!Nothing` weight
branch replaced by the `weight_noc` level, so the two differ in that branch and
nothing else. It therefore also carries the corrected weight spad
(`width: 24` -> 288 weights/PE, per JETCAS Table IV and Sec. III-C).

`arch.yaml` is kept for `ECC_ARCH_FIDELITY=stock`.

Note that this variant now has **10 loop levels**, the deepest design in the
study, so `ECC_VICTORY_SCALING=levels` (the default) gives it 4x the base mapper
effort. A cold run on it is correspondingly slower — that is intentional, since
under-searching the deepest hierarchy is exactly how a mapper artifact gets
reported as an architecture result.
