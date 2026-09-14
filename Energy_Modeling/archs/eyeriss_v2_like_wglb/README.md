# eyeriss_v2_like_wglb — a diagnostic variant, not a second model of the paper

`eyeriss_v2_like` is the faithful model: in Eyeriss v2 weights have **no global
buffer level**. They stream DRAM → weight router → PE weight scratchpad, and the
stock model encodes that with a `!Nothing` branch for `Weights` in the GLB
`!Parallel` block.

That faithful model produces a result that reads as wrong: on dense resnet18 it
refetches every weight **7.32 times** from DRAM, against Eyeriss v1's 3.47, and
so lands 47% above v1 in total energy. This variant exists to test **why**.

## What it changes

The `!Nothing` weight branch is replaced by **two** levels in series, both
keeping `Weights`:

| level | size | what it is |
|---|---|---|
| `weight_glb` | 24 kB SRAM, 16 banks | the weight global buffer |
| `weight_noc` | 288 B | the weight routers |

**This is the shape since 2026-09-14** (the user's decision). Before it there
was one level: a single 64 kB `weight_noc` that had to be the buffer *and* the
routers at once, because the variant declared no weight GLB for the reuse to
live in. The consequence was a weight path with a stage no boundary could sit
at, and a reconstruction study whose `recon2` meant something different on this
design than on every other one.

## What those levels represent

**`weight_glb` — a buffer the earlier v2 designs have, and JETCAS 2019 does
not.** The paper's Table IV "Global Buffer 192 KB" is fully accounted for by
Sec. III-D: per GLB cluster, three 1.5 kB iact banks (72 kB) and four 1.875 kB
psum banks (120 kB). In the *published* v2 weights are not stored in the GLB at
all. The earlier v2 designs this study verifies against do carry a weight global
buffer of `smartbuffer_SRAM` beside the routers, and that is what this level is.
Its **capacity is still derived from a published number rather than chosen**:
one weight bank per GLB cluster, at the paper's own iact bank size — 16 x 1.5 kB
= 24 kB. It is a divergence, and `provenance.yaml` marks it one.

**`weight_noc` — the routers, sized as routers.** Sec. III-C: 3 weight routers
per GLB cluster, each port with "a bitwidth of 24 bits such that it can send and
receive three 8b uncompressed iact values ... per cycle". 16 clusters x 3
routers x 2 ports = 96 ports x 24 b = **2,304 b = 288 B of state in flight**.
It is a storage level because that is how Timeloop maps it, not because it is a
buffer.

Why a level has to be there at all: Timeloop models multicast reuse only across
a *spatial* fanout under a shared parent. With `!Nothing` the weight spads'
parent is DRAM, so **temporal** reuse across the outer loop nest has nowhere to
live and each outer iteration refetches the weight tile from DRAM. That reuse
now lives in `weight_glb`, which is a buffer, rather than in a 64 kB level
called a NoC.

## How to read a number from it

As a bound, not a measurement.

- `eyeriss_v2_like` is the **pessimistic** bound: v2 with none of its reuse
  machinery modelled, on a dense workload it was never designed for.
- `eyeriss_v2_like_wglb` is the **optimistic** bound: v2 given a 24 kB weight
  GLB the published chip does not have, and still with no router circuit energy
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
branch replaced by the `weight_glb` level and the `weight_noc` routers in series
below it, so the two differ in that branch and nothing else. It therefore also carries the corrected weight spad
(`width: 24` -> 288 weights/PE, per JETCAS Table IV and Sec. III-C).

`arch.yaml` is kept for `ECC_ARCH_FIDELITY=stock`.

Note that this variant now has **11 loop levels**, the deepest design in the
study, so `ECC_VICTORY_SCALING=levels` (the default) gives it 4x the base mapper
effort. A cold run on it is correspondingly slower — that is intentional, since
under-searching the deepest hierarchy is exactly how a mapper artifact gets
reported as an architecture result.
