# Constrained mapping: how to make the search converge

*Written 2026-09-10, after prompt_2's sweep failed its convergence gate.
Read this together with prompt_2.md — it does not replace that plan, it fixes
the instrument the plan runs on.*

**STILL CURRENT: the two levers, the free-sets, and the four costs.** They are
what prompt_5 and prompt_6 run under, and `archs.MAPSPACE_FREE_LEVELS` matches
the table below entry for entry.

**HISTORICAL: every measured number here.** They were taken on the
pre-2026-09-10 `eyeriss_like_wglb` geometry — `filter_glb` depth **1024**,
`weights_spad` depth **224**, `psum_spad` depth **24**. The design has since
been reshaped to 256 / 16 / 64, so the tile sizes, refetch figures and ladder
windows below describe an architecture that is no longer on disk. What was
measured on the CURRENT geometry is in FINDINGS §7.12.

## The problem it solves

prompt_2's sweep produced no usable answer, for one reason: **the mapper never
converges**. Measured on Eyeriss v1 `layer3.0.conv1`:

* the mapper's own space is **7.41e10 index factorizations x 8.96e9
  permutations ~ 6.6e20**; victory 50000 covers **0.07 %** of one thread's
  factorization subspace
* the embedded arm's total energy moves **43.7 %** between victory 4000 and
  10000, where the ECC effect being claimed is **5.8 %**
* refetch is **non-monotone in the budget**: 1.000 -> 4.000 -> 2.000

A bigger budget cannot fix that — it samples more of the same enormous space,
both arms still land on arbitrary points, and *the difference between two
arbitrary points is noise*. The fix is to make the space small enough to
search **exhaustively**.

## What was changed — two independent levers

**1. Constrain WHICH LEVELS each loop dimension may be split across.**
Timeloop's factorization count explodes because every dimension can be split
across all 9 levels. Pinning a dimension to 1 at the levels it does not need
collapses the space. The free-sets live in `archs.MAPSPACE_FREE_LEVELS` and
were read off the **best mapping the search has ever found** for this design
(the victory-10000 embedded nest):

| dim | size | free at | levels | factorizations |
|---|---:|---|---:|---:|
| C | 128 | `DRAM`, `PE`(spatial), `weights_spad` | 3 | 36 |
| M | 256 | `ifmap_glb`, `PE_column`(spatial), `weights_spad`, `psum_spad` | 4 | 165 |
| R | 3 | `psum_glb` | 1 | 1 |
| S | 3 | `PE`(spatial) | 1 | 1 |
| P | 14 | `ifmap_glb`, `psum_glb` | 2 | 4 |
| Q | 14 | `filter_glb`, `PE_column`(spatial) | 2 | 4 |

**9.5e4 factorizations, down from 7.41e10 — a 780,000x reduction.**

**2. Unpin the weight TILE so capacity can bind.** This is the "fix the tile"
half, and without it lever 1 alone changes nothing. Eyeriss v1 declares

    weights_spad: temporal: factors: [N=1, M=1, P=1, Q=1, S=1]

`M=1` pins the M tile *at that level* to one, so the resident tile was
**16 weights/PE at every depth** — the buffer was never the limit, the
constraint was. `ECC_WEIGHT_FACTOR_RELAX=1` drops the `M=1, S=1` pins on
weight-carrying levels.

**The two must compose, not cancel.** The constraint runs **after** the relax,
and `weights_spad` is deliberately in the free-set for **both C and M** — so
the dimensions the relax frees stay free. Result: `weights_spad` ends up
`[N=1, R=1, S=1, P=1, Q=1]` and the tile stops being constraint-limited. On the
geometry of the day that took it from 16 to **128 weights/PE**; on the current
spad (depth 16) the array itself is the limit and it fills to **32 w/PE, 100 %**.
Either way the point holds: after the relax the BUFFER binds, not the pin.

## Every variable used

    # --- the two new levers -------------------------------------------
    ECC_MAPSPACE_CONSTRAIN=1        # lever 1; free-sets in archs.MAPSPACE_FREE_LEVELS
    ECC_WEIGHT_FACTOR_RELAX=1       # lever 2; drops M=1,S=1 on weight levels
    ECC_MAPPER_ALGORITHM=linear_pruned    # systematic walk, not random sampling
    ECC_MAPPER_TIMEOUT=100000000    # MANDATORY with linear_pruned -- at the
                                    # default 2000 it dies in 10 s, because its
                                    # walk starts where the first ~36,000
                                    # mappings are all infeasible
    # --- prompt_2's mechanism (unchanged) ------------------------------
    ECC_WEIGHT_DATAWIDTH=           # EMPTY = 8b baseline+embedded arm (one
                                    #   mapping); 4 = recon arm at BCH(63,30)
    ECC_WEIGHT_DEPTH_SCALE=1.0      # THE swept variable
    ECC_WEIGHT_DEPTH_LEVELS=        # empty = all weight levels together;
                                    #   name one for the per-level 2nd pass
    ECC_WEIGHT_WIDTH=               # EMPTY for BCH(63,30) -- q=4 divides the
                                    #   published 16b spad and 64b GLB words
    ECC_WEIGHT_WIDTH_GLB_MULT=4     # GLB word = 4x the scratchpad word
    # --- the study point ------------------------------------------------
    ECC_RECON_ARCHS=eyeriss_like_wglb    # NOT "Eyeriss v1" under wrelax -- cost 1
    ECC_MODELS=resnet18 ; ECC_LAYERS=layer3.0.conv1
    ECC_KS=30 ; ECC_CONST_K=30 ; ECC_CODE_N=63     # BCH(63,30), 4-bit
    ECC_OPT_METRIC=edp              # never `energy` -- it trades PEs for capacity
    ECC_RECON_PACKING=aligned       # NOT sufficient on its own -- see below
    ECC_MAPPER_THREADS=18           # in the fingerprint; any other value = cold cache
    ECC_QOS=rewetz-b                # ~90 concurrent jobs

**CORRECTION 2026-09-10 — `aligned` does NOT by itself stop the double count.**
This prompt originally said `ECC_RECON_PACKING=aligned` is what keeps the
on-chip narrowing from being applied twice. It is not. `aligned` is a no-op
only when `floor(W/q) == floor(W/8)`; at BCH(63,30), `q = 4` and
`Packing.storage_scale` returns **0.5** on both the 64-bit GLB word and the
16-bit spad word — the same halving `datawidth: 4` already gives the mapper.
Set both and the saving is squared, and `onchip_narrowing_audit()` will not
catch it, because it decides from the packing mode's NAME rather than from the
multiplier it produces. Whoever owns the narrowing must be DECLARED, and for
any mapping study it is the mapper. prompt_6.md RULE 1 carries the fix.

## The one command

    cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
    module load apptainer
    export ECC_MAPSPACE_CONSTRAIN=1 ECC_WEIGHT_FACTOR_RELAX=1 \
           ECC_MAPPER_ALGORITHM=linear_pruned ECC_MAPPER_TIMEOUT=100000000
    bash hpc/map_depth_sweep.sh            # 42 jobs: ladder + per-level + gate
    # a single depth point is 4 jobs instead, which is what prompt_5/prompt_6 run:
    #   ECC_DEPTH_SWEEP_SCALES="1.0" ECC_DEPTH_SWEEP_GATE_SCALES="1.0" \
    #     bash hpc/map_depth_sweep.sh --no-per-level
    # then, in order:
    bash hpc/tl.sh python3 -m eccenergy.experiments.dilation --gate
    bash hpc/tl.sh python3 -m eccenergy.experiments.dilation --levels \
         --csv results/tables/EyerissV1_mem_arch_sweep.csv

Own cache slug `mcons__wrelax`, so it never mixes with the unconstrained runs.
`--gate` must pass before `--levels` is quoted.

## What it costs — read this before quoting anything

On the OLD geometry (spad 224 / GLB 1024), which is what these were measured on:

| | unconstrained | constrained + relax |
|---|---:|---:|
| time per map | ~45 min | **52 s** (52x faster) |
| refetch @ x1 | 4.000 | **1.000** |
| `weights_spad` tile | 16 w/PE | **128 w/PE** |
| convergence gate | FAILS by 6-21x | passes (exhaustive) |

On the CURRENT geometry (spad 16 / GLB 256), measured 2026-09-10 under the same
two levers: **55 s – 1 min 05 s** per map, weight refetch **1.000 on both arms**,
`weights_spad` **32 w/PE at 100 % fill**, and the gate at **0.00 % residual**
across victory 2000 / 4000 / 10000. The speed and the convergence carried over;
the tile and the windows did not.

**Four costs, all of them real:**

1. **It is NOT the published chip.** `M=1` at the filter spad *is* Eyeriss v1's
   row-stationary dataflow. A design run under `wrelax` is a different
   accelerator and `source: published` does not licence its name. Label every
   number "constrained dataflow", never "Eyeriss v1".
2. **The free-sets came from a known-good nest**, so the exhaustive answer is
   the best mapping **in that family**, not in the whole space. Constraining
   around a region the search already liked is a choice, not a neutral act.
3. **The constraint may exclude the region where Recon wins.** So a NEGATIVE
   result under it is weaker evidence than a negative would be unconstrained;
   a POSITIVE is trustworthy. What keeps the comparison fair is that **both
   arms get the identical constraint** — neither is handed a region the other
   cannot reach.
4. **Weight refetch is already 1.000 at full depth** — `C` is the only
   DRAM-level loop and `C` indexes Weights, so every weight is fetched exactly
   once and no capacity can improve it. That is a floor, not a null result.

   **What this prompt originally concluded from it was WRONG and is corrected
   here (FINDINGS §7.12).** It read "no weight traffic left to remove" as "no
   capacity win until the Embedded arm is shrunk off that floor", and gave a
   ladder window of x0.29 -> x0.14. Measured on the current geometry at **x1,
   with no depth scaling at all**, the reconstruction arm wins by **22.4 %** —
   and not through weight refetch, which stayed 1.000 on both arms. The extra
   weight capacity halves the DRAM-level `C` CHUNKING, 8 chunks to 4, and `C`
   does not index Outputs, so each chunk removed is a partial-sum round trip to
   DRAM that never happens. 86 % of the margin is in the Outputs row.

   The lesson to carry forward: **look at the dataspace the capacity actually
   frees, not only at the one it stores.** A weight-buffer change pays in the
   PSUM path on this design.

## Porting it to another design or layer

`MAPSPACE_FREE_LEVELS` is keyed by architecture and is the only thing that
needs writing. To add one: map the layer once unconstrained at the largest
victory you can afford, read the level names and which dimensions appear where
out of `timeloop-mapper.map.txt`, and write that as the free-set. A design with
no entry is **not** constrained and says so on the console rather than
pretending. The free-sets are per-design, not per-layer — but a layer whose
shape factors very differently may need its own, and the tell is a mapspace
that is still too large or a mapper that returns nothing feasible.
