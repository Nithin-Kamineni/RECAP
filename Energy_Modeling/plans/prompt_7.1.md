# prompt_7.1 — Where is reconstruction at its best?

**A buffer-size sweep to find the design point at which reconstruction's advantage over
embedded is largest, and to say plainly what sets that ceiling.**

| | |
|---|---|
| **Run AFTER** | `prompt_7.md`. This needs its latency model (Issue 6) and its standby model (Issue 5) wired; without them every bar is identical and the sweep answers nothing. |
| **Design** | `eyeriss_like_wglb` (Eyeriss v1) |
| **Produces** | one table + one figure: buffer size on x, reconstruction's saving over embedded on y |
| **Nature** | **design-space exploration, not a claim about the published chip.** Every number must be reported with the buffer size that produced it. |
| **Written** | 2026-09-12 |

---

## §1 — The question

Reconstruction's benefit is proportional to **weight traffic**. Anything that removes weight
traffic removes its opportunity; anything that removes *other* traffic makes weights a larger
share and helps it. That gives a sharp, testable prediction — and one half of it is
counter-intuitive.

```
   saving ceiling  =  (weight share of off-chip traffic)  x  (1 - K/N)
                       \_____________________________/       \_______/
                          what this sweep MOVES                 0.5238, fixed by the code
```

Measured today, resnet18:

```
   OFF-CHIP TRAFFIC
   partial sums  ################################  68.6%   <- reconstruction never touches these
   weights       ##############                    26.5%   <- only this halves
   inputs        ##                                 4.9%

   ceiling today = 26.5% x 52.4% = 13.9%
   if partial-sum spill were ELIMINATED: 84.3% x 52.4% = 44.2%
```

---

## §2 — The prediction, stated before the run

**Write the expected direction down first; a result that contradicts it is a bug until
proven otherwise.**

| lever | what it changes | LATENCY saving | ENERGY saving | why |
|---|---|---|---|---|
| **`psum_glb` bigger** | fewer partial sums spill to DRAM | **RISES** | **RISES** | weights become a larger share of what is left |
| **`filter_glb` bigger** | fewer weight re-fetches | **FALLS** | **FALLS** | **there is less weight traffic left to halve** |
| `ifmap_glb` bigger | fewer input re-reads | rises slightly | slightly | inputs are only 4.9% |
| more PEs | compute finishes sooner | rises | rises | memory binds harder |

> **The counter-intuitive one, and the reason this sweep is worth running: making the WEIGHT
> buffer bigger makes reconstruction look WORSE.** A reader will assume the opposite. If the
> sweep confirms it, say it explicitly in the paper — it is the clearest statement of what
> the mechanism actually depends on.

---

## §3 — What to sweep

Two independent axes, each with the other held at its published value.

| axis | knob | points | published value |
|---|---|---|---|
| **A. partial-sum buffer** | `psum_glb` `depth` | 1x, 2x, 4x, 8x | 6144 (48 kB) |
| **B. weight buffer** | `filter_glb` `depth` | 0.5x, 1x, 2x, 4x | 256 |

Both models (`resnet18`, `mobilenet_v2`), BCH(63,30), the placement study's five boundaries.

**Axis A is the headline.** Axis B exists to demonstrate the counter-intuitive direction and
needs only three or four points.

---

## §4 — How to run it, cheapest first

### Stage 1 — evaluator-side estimate. NO mapper runs.

The ceiling formula needs only the off-chip traffic split, which is already in every cached
record. A larger `psum_glb` cannot be *simulated* this way — but the **ceiling** it would
reach can be bracketed by recomputing the formula with the partial-sum term scaled down.

```
   for each scale s in (1, 2, 4, 8):
       assume psum spill falls by some factor f(s)     <- STATE THE ASSUMPTION
       weight_share(s) = W / (W + I + O/f(s))
       ceiling(s)      = weight_share(s) x (1 - K/N)
```

This is an estimate, not a measurement, and **must be labelled as such**. Its purpose is to
decide whether Stage 2 is worth hours of compute. If the ceiling barely moves, stop here.

### Stage 2 — the real sweep. COLDS THE CACHE, one architecture per point.

Each buffer size is a **different accelerator**, so each needs its own mapper pass.

    bash hpc/map_by_shape.sh --no-eval       # fan out: one job per layer SHAPE, not per model
    # then ONE dependent eval over all of them

Budget: 4 points x 2 models x (21 + 53 shapes). Use `map_by_shape.sh`, never a serial loop.
**Narrow before widening** — run `ECC_MODELS=resnet18` and one scale first and confirm the
direction matches §2 before committing the rest.

---

## §5 — What to record per point

| column | why |
|---|---|
| buffer size, in entries AND kB | the size is part of the result, never dropped |
| off-chip traffic split (W / I / O) | this is the mechanism; the saving follows from it |
| weight share | the first factor of the ceiling |
| **ceiling** = weight share x (1 - K/N) | the prediction |
| measured latency saving vs embedded | the test of the prediction |
| measured energy saving vs embedded | |
| total energy, both arms | so a saving that comes from a worse baseline is visible |

**The last row matters.** A buffer change moves BOTH arms. A percentage that improves because
the *embedded* arm got worse is not a reconstruction result, and only the absolute totals
reveal it.

---

## §6 — Honesty rules for this sweep

This is exploration of a design space, which is legitimate. Three rules keep it so:

1. **Every number carries its buffer size.** A saving quoted without it is meaningless.
2. **The published configuration stays the headline.** The sweep is a sensitivity study
   around Eyeriss v1, not a replacement for it. If the best point is 8x the published
   partial-sum buffer, that is a *finding about when the mechanism pays*, not a claim about
   Eyeriss.
3. **Report both arms' absolute energy**, so nobody can mistake a degraded baseline for an
   improved reconstruction.

A design point where reconstruction wins because the accelerator was reshaped to suit it is a
useful and publishable result — *described as such*. It stops being useful the moment the
reshaping is dropped from the caption.

---

## §7 — Deliverable

One table (§5's columns) and one figure: buffer size on x, saving over embedded on y, one
line per boundary. Plus one sentence stating the design point at which reconstruction peaks
and what limits it there.

Record the outcome in `FINDINGS.md`; add a line to `progress.txt`.
