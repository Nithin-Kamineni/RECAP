# prompt_7 — Time, standby power, and one mapping per boundary

**The study never modelled time. Chasing that uncovered two larger things: standby power is
charged to one side of the comparison and not the other, and three of the five
reconstruction boundaries are evaluated on a mapping that belongs to a different chip.
All three defects are measured, none is speculative, and the fixes are sequenced below so
that each PHASE is one working session with its own gate.**

| | |
|---|---|
| **Design** | `eyeriss_like_wglb` (Eyeriss v1) for the numbers. **The defects are study-wide** — no architecture in `archs/` declares an off-chip speed limit, no arm is charged component standby power, and every design's placement list shares one reference mapping. |
| **Extends** | `prompt_6.md`, especially **RULE 3** (the idle term) and **RULE 4** (whose plan a bar is billed from). Does not replace either. |
| **Status** | **Phase 0 landed 2026-09-12** (§13). **Phases A–E are the remaining work** (§9). Nothing in this document is blocked on a decision; every open question is an evidence gap with a stated fallback. |
| **Numbers** | *measured* = real, reproduced from the cache or from runs recorded in Appendix A. *projected* = arithmetic on measured inputs, **not adopted**. Every table says which. |
| **Invalidates** | FINDINGS §2.9's placement ranking — clock gating is now the default, so R4 moves from −33.96% to about +9%. Regenerate before quoting. |
| **Written / revised** | written 2026-09-11; revised 2026-09-12 (Defect 3 added, Issue 9 resolved into a reporting rule, phases restructured into sessions) |

---

## §0 — How to use this document

### 0.1 It is a PHASE PLAN, not an issue list

`§9` is the spine. Each phase is sized to be **started and finished in one session**, ends
with a **gate you can run**, and states what it colds. Do not start a phase until the
previous one's gate has passed — the later phases read numbers the earlier ones produce.

```
   PHASE 0  DONE 2026-09-12 ............ five fixes already in the code (§13)
   PHASE A  report what is already there  no compute, no cold .... 1 session
   PHASE B  one mapping per boundary      no compute, no cold .... 1 session
   PHASE C  the single cold pass          HOURS of SLURM ......... 2 sessions (C1 author, C2 run)
   PHASE D  transformer workload          hours .................. 1 session
   PHASE E  prompt_7.1's buffer sweep     see that file
```

### 0.2 If you are a new session starting here

Read in this order. Each row says why you need it; skipping a row costs more than reading it.

| order | read | why |
|---|---|---|
| 1 | `CLAUDE.md` | the rules of the repo: caches, fingerprints, frozen files, `env.sh` |
| 2 | **this file §1–§3** | vocabulary, and the machine drawn as a picture |
| 3 | **this file §9**, your phase only | what to build, and the gate |
| 4 | `prompt_6.md` §1 and its four RULES | still in force; this plan extends them |
| 5 | `FINDINGS.md` | what was already learned; **§2.9 is stale, see the header** |

Then: `bash run.sh diagnose` before you change anything and after, and diff the two.

### 0.3 Three standing rules for every phase

1. **Batch cache-colding changes.** Anything touching `env.sh` §2/§5, an arch YAML, or
   `archs/_shared/components/` re-fingerprints the whole matrix. Phases A and B are
   deliberately free of such changes so that Phase C is the only rebuild.
2. **Property tests on real cached data, plus deliberate breakage.** A test that cannot
   fail proves nothing. Every phase's gate names the mutation that must make it fail.
3. **The frozen-baseline rule.** After any change: `bash run.sh baseline --eval` and diff.
   Task 1 and Task 2 totals must not move. `parity.py`, `baseline.py` and
   `external_parity()` stay frozen.

---

## §1 — Terminology

Terms already defined in `prompt_6.md` §1 (arm, bar, boundary, ERT, scalar access,
`block_size`, `q`, engine) are not repeated.

| Term | What it means, plainly |
|---|---|
| **cycle** | One tick of the clock. At 1 GHz (`ECC_GLOBAL_CYCLE_SECONDS=1e-9`), **one cycle = 1 ns**. Eyeriss v1 silicon is 200 MHz, so one cycle = 5 ns (`ECC_ARCH_CLOCK_MHZ`). |
| **latency** | Time for one inference = cycles × the cycle period. |
| **compute cycles** | How long the arithmetic alone would take with infinitely fast memory. |
| **item** (vs **bit**) | Timeloop's speed limit counts **items per cycle** — one weight, one input — **not bits per cycle**. A half-width weight is still one item. **This distinction is the whole of Defect 1.** |
| **declared bandwidth** | A number written in the arch YAML (`read_bandwidth: 16`): the most items that level can hand over per cycle. A ceiling, in items. |
| **demand** | What the chosen mapping actually asks of a level, in items/cycle. |
| **throttling** | `min(1, declared ÷ demand)`. `1.00` = this level is not slowing the chip. `0.18` = it keeps up with 18% of demand. |
| **bottleneck** | Total cycles = the **slowest** level, not the sum. Speeding up a non-bottleneck changes nothing. |
| **duty cycle** | Fraction of time a block is actually working. A reconstruction engine at the scratchpad: **0.62%**. |
| **arm** (mapper sense) | ONE mapper job's configuration = (which levels carry `datawidth: q`) + (which ERT rows are bumped) + (which levels carry a bandwidth scale). Two arms with different tuples are **different chips** to the mapper and must not share a cache directory. |

### §1.1 — Static, idle, and clock power are THREE DIFFERENT THINGS

| # | quantity | what burns it | removed by |
|---|---|---|---|
| 1 | **Static / leakage power** | transistors leaking because the supply is on, even with the clock stopped | **power gating** (switch the supply off; slow to wake) |
| 2 | **Clocked-idle dynamic power** | the clock tree toggling, and flip-flops toggling on clock edges, while inputs are held — "on and ticking", doing nothing useful | **clock gating** (stop the clock; wakes in ~1 cycle) |
| 3 | **Active dynamic power** | logic actually switching to do work | nothing — this is the useful cost |

```
   RECONSTRUCTION ENGINES                     THE ACCELERATOR (SRAM, spads, NoC, MACs)
   ----------------------                     ----------------------------------------
   idle_per_cycle = 2.8311 pJ/cyc             charged to the REPORT: NOTHING
            |                                 seen by the MAPPER:    a `leak` row whose
            +-- 2.8164 pJ  clocked-idle DYN      prices are 10^3-10^4 too low (§5.2)
            +-- 0.0147 pJ  true LEAKAGE
            (ONE number in the DC JSON)        source: Accelergy / CACTI, pinned to the
                                               least-leaky transistor recipe there is
   source: Synopsys Design Compiler, 45 nm
```

Quantities 1 and 2 are **lumped** for the reconstruction engines and **absent from the
report** for the accelerator, and the two sides come from tools that disagree by orders of
magnitude. Phase 0 settled the reconstruction side (both are now gated together, in the
evaluator *and* the ERT). Phase A settles the accelerator side.

---

## §2 — The one-page version: three defects

```
  +--------------------------------------------------------------------------------+
  |  DEFECT 1 -- TIME DOES NOT RESPOND                                    §4        |
  |  datawidth: q makes energy fall and cycles stand still, because Timeloop's      |
  |  speed model counts ITEMS and a narrow weight is still one item.  No arch in    |
  |  archs/ declares an off-chip limit at all, so DRAM costs zero cycles.           |
  |  -> reported latency gain 0.00% is an ABSENT TERM, never a result.              |
  +--------------------------------------------------------------------------------+
  +--------------------------------------------------------------------------------+
  |  DEFECT 2 -- STANDBY POWER IS CHARGED TO ONE SIDE ONLY                §5        |
  |  R4's engines are billed 5,135.8 uJ of standby -- 12.3x the whole 168-MAC       |
  |  array -- while the accelerator is billed 0.000 uJ, because parse_stats never   |
  |  reads `Leakage energy (total)`.  The mapper DOES see leakage; the report does  |
  |  not.  And the prices the mapper sees are 10^3-10^4 too low.                    |
  +--------------------------------------------------------------------------------+
  +--------------------------------------------------------------------------------+
  |  DEFECT 3 -- ONE MAPPING, SIX CHIPS                                   §6        |
  |  Five boundaries take THREE distinct shapes as far as the mapper is concerned,  |
  |  and only TWO of them are ever mapped.  R3 borrows the reference's plan when    |
  |  its geometry is R2's; R5a is the only bar that narrows weights_spad and no     |
  |  mapping in the cache knows that.  Measured: an arm's own plan differs from     |
  |  the reference's on 9/43 shapes (R2) and 13/43 (R4) -- TODAY, before any of     |
  |  this plan lands.                                                              |
  +--------------------------------------------------------------------------------+
```

---

## §3 — The machine, in pictures

Everything below refers to this one diagram. `eyeriss_like_wglb`, weights only.

### 3.1 The weight path and the five boundaries

```
   DRAM DIE                   |                     THE CHIP
   ========                   |                     ========
                              |
  +---------------+           |  +------------+   +===========+   +===========+   +----------+
  | weights       |           |  | filter_glb |   |  array    |   | PE-local  |   | weights_ |
  | + parity      |--k of n-->|  | 8 kB SRAM  |-->| multicast |-->| multicast |-->|  spad    |--> MAC
  | [decoder]     |  bits     |  | 2048 items |   | 14 wide   |   | 12 wide   |   | 32 items |
  +---------------+           |  +------------+   +===========+   +===========+   +----------+
                              |         ^                ^               ^              ^
        boundary:      R1 ----+---------+                |               |              |
                                    R2 -+----------------+               |              |
                                             R3 ---------+---------------+              |
                                                       R4 ----------------+--------------+
                                                                       R5a --------------+--> (spad read)

   ===  = a NETWORK level.  Timeloop has NO timing model for these (§4.4).
   +-+  = a STORAGE level.  These have capacity, declared bandwidth and leakage.
```

**A boundary's `reduced` set is a PREFIX of that path**: everything from the DRAM die down
to the boundary carries the reduced (`q`-bit) form; everything past it is full width.

### 3.2 What each boundary actually changes — the table that drives everything

Derived from `recon.WEIGHT_PATHS` / `recon.PLACEMENTS`, verified 2026-09-12:

| bar | `reduced` stages | **storage levels at `datawidth: q`** | encoder site / counter | ERT-injectable? |
|---|---|---|---|---|
| **R1** @ source | `dram` | **— (nothing on chip)** | `dram` / `reads` | ✗ DRAM is not a chip action |
| **R2** @ filter GLB | + `filter_glb` | **`filter_glb`** | `filter_glb` / `reads` | ✓ `read` |
| **R3** @ column edge | + `array_multicast` | **`filter_glb`** (network has no width) | `array_multicast` / `deliveries` | ✗ network — billed via `noc.yaml` |
| **R4** @ spad input | + `pe_local_multicast` | **`filter_glb`** | `weights_spad` / `fills` | ✓ `write` |
| **R5a** @ spad output | + `weights_spad` | **`filter_glb` + `weights_spad`** | `weights_spad` / `reads` | ✗ innermost reads (but see §6.3) |

```
   THE MAPPER SEES ONLY THREE DISTINCT CHIPS TODAY:

     chip #1  nothing narrowed .......... reference, R1
     chip #2  filter_glb narrowed ....... R2, R3, R4
     chip #3  filter_glb + spad narrowed  R5a          <-- NEVER MAPPED
```

### 3.3 Where the numbers live

| thing | file |
|---|---|
| every knob | `env.sh` — ten commented sections, the only file you edit |
| the boundaries | `eccenergy/recon.py` — `WEIGHT_PATHS` + `PLACEMENTS`, **edited together** |
| the DC engine constants | `data/dc/BCH_N63_results.json`, overridden by `env.sh` §6 |
| the price list Timeloop bills | `<mapper cache>/_ert/timeloop-mapper.ERT.yaml` |
| what was learned | `FINDINGS.md` (empirical), `CLAUDE.md` (rules) |

---

## §4 — Defect 1: time does not respond

### 4.1 The disconnect, in one diagram

```
                  reconstruction stores weights at q bits instead of 8
                                        |
                +-----------------------+-----------------------+
                |                                               |
                v                                               v
         ENERGY PATH  (works)                            TIME PATH  (broken)
                |                                               |
   datawidth: q  ->  block_size = width/q                 datawidth: q -> ???
                |                                               |
   Timeloop bills                                      Timeloop's speed model counts
   vector_access_energy / block_size                   ITEMS PER CYCLE, and a narrow
   per weight                (buffer.cpp:2225)         weight is still ONE item.
                |                                                    (buffer.cpp:2489)
                v                                               v
      energy falls, correctly                          NOTHING CHANGES. Not one cycle.

   demand   = items / compute_cycles                 buffer.cpp:2553
   slowdown = min(1, declared_bandwidth / demand)    buffer.cpp:2573-2600
   cycles   = ceil(compute_cycles / slowdown)        buffer.cpp:2622
   TOTAL    = max over all levels                    topology.cpp:1602

   `block_size` appears NOWHERE in lines 2476-2623.
```

### 4.2 Measured: the width change is invisible to the clock

Same layer, same mapping, from two caches already on disk
(`C128_M128_R3_S3_P28_Q28_ws1_hs1`):

| `filter_glb` | reference | recon2 (narrowed) | |
|---|---:|---:|---|
| Word bits | 8 | **4** | the change |
| Block size | 8 | **16** | twice the weights per physical word |
| **Size (items)** | 2048 | **4096** | **capacity doubles — see §6.2** |
| Weights energy | 143,772 pJ | ... | energy responds — correct |
| Declared read bandwidth | **16.00** | **16.00** | **unchanged — the defect** |
| Weight read demand | 0.21 items/cyc | 0.21 items/cyc | unchanged |
| Throttling | 1.00 | 1.00 | unchanged |
| **Cycles** | **688,128** | **688,128** | **unchanged** |

Confirmed in the strongest form (Appendix A.1): even with off-chip memory deliberately made
the bottleneck, narrowing 8 → 4 bits changes **0.0% of cycles** and 34% of energy.

> **The declared bandwidth is a hard-coded item count in the YAML.** `filter_glb` declares
> `read_bandwidth: 16` whether its words hold 8-bit or 4-bit weights. Physically a 64-bit ×
> 2-bank port delivering 4-bit items should hand over **32** items/cycle. **The model gives
> a narrowed level more CAPACITY and no extra BANDWIDTH**, which is half a physical change.

### 4.3 Off-chip memory has no speed limit at all

DRAM declares `type`, `depth`, `width`, `datawidth` and **no bandwidth**
(`arch_paper.yaml:64-72`); the stats print `Read bandwidth : -`. Timeloop assigns no default
(`buffer.cpp:437-447` has no `else`), so the check is skipped. **DRAM throttling is 1.00 on
all 43 cached shapes. No architecture in `archs/` declares it.**

### 4.4 Networks have no speed model at all — verified on disk

`LegacyNetwork::ComputePerformance()` is an empty stub (`network-legacy.cpp:1097-1107`).
This is visible in every cached stats file without reading the source:

```
   EVERY STORAGE LEVEL PRINTS            EVERY NETWORK LEVEL PRINTS
   ---------------------------           --------------------------
       STATS                                 STATS
       Cycles               : 688128         Weights:
       Bandwidth throttling : 1.00               Fanout           : 14
       ... Read Bandwidth   : 0.21 w/c          Multicast factor : 7
       ... Write Bandwidth  : 0.21 w/c          Ingresses        : 147456.00
                                                Energy (per-hop) : 389.32 fJ

   -> a network has NO Cycles, NO throttling, NO bandwidth field of any kind,
      so it can never enter `max over levels`.
```

`archs/_shared/noc.yaml` declares **energy coefficients only** — there is no interconnect
bandwidth number anywhere in the study, and none is cited. See **§12 rule R-3** for how this
must be reported; it is no longer an open issue.

### 4.5 **NEW (2026-09-12): there IS a weight level that binds, and it is `filter_glb`**

This is the finding the earlier draft of this plan did not have, and it changes what Part A
has to do.

Throttling across all 43 cached shapes of `eyeriss_like_wglb`, reference arm:

| level | dataspace | worst throttling | shapes it slowed |
|---|---|---:|---:|
| `ifmap_glb` | Inputs | **0.180** | **11 / 43** |
| `psum_glb` | Outputs | **0.610** | **5 / 43** |
| `filter_glb` | Weights | 1.000 | 0 / 43 |
| `weights_spad` | Weights | 1.000 | 0 / 43 |
| `DRAM` | all | 1.000 | 0 / 43 — *declares no limit* |

The two levels that ever bind carry **inputs and partial sums**, which reconstruction never
touches. **That is the honest headline and it must not be softened.**

But look at the fully-connected layers, where weights are ~99% of the traffic:

| shape | PE utilisation | `filter_glb` weight read demand | its declared limit | cycles |
|---|---:|---:|---:|---:|
| `C512_M1000_R1_S1_P1_Q1` (resnet18 `fc`) | **9.52%** = 16/168 PEs | **16.00 items/cyc** | **16.00** | 32,000 |
| `C1280_M1000_R1_S1_P1_Q1` (mobilenet `classifier`) | **9.52%** | **16.00** | **16.00** | 80,000 |

```
   512,000 computes / 16 PEs = 32,000 cycles = exactly the reported cycle count.
   Had the mapper chosen 32 PEs:  compute = 16,000 cycles
                                  weight demand = 512,000/16,000 = 32 items/cyc
                                  > filter_glb's 16  ->  throttle 0.5  ->  32,000 cycles
                                  same time, more energy  ->  the mapper picks 16 PEs.

   => filter_glb's 16 items/cycle is LITERALLY what caps this layer at 9.52% utilisation.
      A bit-aware port (32 items/cyc at q=4) would halve that layer's cycles.
```

**So the on-chip latency lever is real, it is at `filter_glb`, and it belongs to R2/R3/R4/R5a.**
Three caveats, all of which must travel with it:

1. Those layers are **0.28% of resnet18's cycles** and **2.10% of mobilenet's**. Aggregate
   effect on these CNNs: **~0.1–1%**.
2. **For a batch-1 transformer every layer is that layer.** Same conclusion the off-chip
   analysis reaches from the other direction (A.6) — two independent mechanisms agreeing
   on the same workload is why Phase D exists.
3. **`read_bandwidth: 16` is not cited in `archs/_shared/provenance.yaml`.** Nor are any of
   the on-chip bandwidth numbers. They are inherited from the reference design. A headline
   resting on one of them needs a citation first (Phase C, C1.4).

### 4.6 "We move half the data, so we should take half the time" — why not, here

Two things break the chain, both measured.

**Reason 1 — it is half the WEIGHTS, not half the data.** Measured off-chip traffic, resnet18:

```
   +--------------------------------------------------------------+
   |  Outputs (partial sums written and re-read)   33.2 M   68.6%  |
   |  Weights                                      12.8 M   26.5%  |  <-- only this shrinks
   |  Inputs                                        2.4 M    4.9%  |
   +--------------------------------------------------------------+
                              total 48.5 M items

   total traffic cut = 26.5% x 52.4% = 13.9%      (resnet18)
                     = 10.9% x 52.4% =  5.7%      (mobilenet_v2)
```

**Reason 2 — the memory is not the bottleneck, so making it lighter changes nothing.**
Total time is `max(compute, movement)`, not their sum:

```
   compute   |################################################| 100%  <-- the limit
   off-chip  |########                                        |  17%  (at LPDDR4-3200)
                      ^
                      reconstruction makes THIS bar 14% shorter. The finish line does not move.
```

> **Analogy.** A factory where the workers are the bottleneck and the delivery truck arrives
> one-fifth full. Lighter cargo does not make the factory finish sooner. You gain only when
> the truck is what everyone is waiting on.

**When the objection IS right — and it often is:**

| condition | why | ceiling |
|---|---|---:|
| **Weight-dominated layers** | fc / 1×1 are ~99% weight traffic, so Reason 1 disappears | **51.6%** measured on resnet18 `fc` |
| **Batch-1 transformers** | every layer behaves like `fc` | approaches `1 − K/N` = **52.4%** |
| **A narrower off-chip interface** | Reason 2 disappears — Eyeriss v1's real situation, in JSSC 2017's own words | 13.9% (resnet18) |
| **A larger PE array** | more throughput raises demand per cycle until memory binds | rises toward the traffic-cut ceiling |

```
   latency saving  =  (weight share of the binding resource) x (1 - K/N) x (how bound you are)
                      \______________________________/         \_______/    \_______________/
                        26.5% resnet18 / 10.9% mbnet             52.4%         0% .. 100%
                        99%   at fc / transformer
```

The third factor is what a bandwidth number sets, and it is the term the model is missing.
**A reconstruction scheme cannot speed up a compute-bound accelerator — that is a property
of the accelerator, not a defect in the scheme.**

### 4.7 What to expect per boundary — PROJECTED, NOT ADOPTED

At the decided operating point (`ECC_DRAM_BANDWIDTH_MBPS=480`, `ECC_ARCH_CLOCK_MHZ=200`
→ **2.4 items/cycle** against a measured **4.26 items/cycle** demand, i.e. hard off-chip-bound):

| bar | DRAM relief | `filter_glb` relief | network relief | spad relief | **projected latency gain** |
|---|---|---|---|---|---:|
| R1 | ✓ shared | — | — | — | 13.9% / 5.7% |
| R2 | ✓ shared | ✓ | — | — | 13.9% / 5.7% |
| R3 | ✓ shared | ✓ | **structurally 0** (§4.4) | — | 13.9% / 5.7% |
| R4 | ✓ shared | ✓ | **structurally 0** | — | 13.9% / 5.7% |
| R5a | ✓ shared | ✓ | **structurally 0** | **0 — measured 50% headroom** | 13.9% / 5.7% |

> **PLAN AROUND THIS: latency will be essentially FLAT across the five placements.** The
> whole gain lives at the DRAM boundary, which every placement shares by construction. Once
> DRAM binds at 2.4 items/cycle it is the worst level everywhere and `filter_glb`'s
> contribution collapses to zero. **Latency will not rank the boundaries. Energy will.**
> That is a result, not a disappointment — but it is the opposite of the intuition that
> motivated the work, so it must be stated before the compute is spent.

---

## §5 — Defect 2: standby power is charged inconsistently

**Independent of Defect 1, and the reason R4 looks like the worst boundary when physics says
it should be the best.**

### 5.1 Where reconstruction energy actually goes

`prompt_6` RULE 3: `E_recon = incremental × codewords + idle_per_cycle × cycles × engines`.
Measured, resnet18, BCH(63,30), **before** clock gating:

| bar | engines | codewords | **idle share of E_recon** | **duty cycle** |
|---|---:|---:|---:|---:|
| R1 @ source | 1 | 1,627,640 | 93.5% | 14.30% |
| R2 @ filter GLB | 1 | 1,627,640 | 93.5% | 14.30% |
| R3 @ column edge | 14 | 11,237,539 | 96.7% | 7.05% |
| **R4 @ spad input** | **159.3** | 11,237,539 | **99.7%** | **0.62%** |
| R5a @ spad output | 159.3 | 230,358,520 | 94.2% | 12.70% |

Because nothing can stall the pipeline today, Timeloop's cycle count is exactly
compute-bound, so at a scratchpad boundary:

```
   engine_cycles = SUM over layers of (utilized PEs x cycles) = total MACs
   measured:  weights_spad engine_cycles = 1,814,073,344
              total MACs                 = 1,814,073,344      <-- identical

   => charging 2.8311 pJ/cycle/engine there is ARITHMETICALLY IDENTICAL to charging
      2.83 pJ PER MAC for reconstruction, on a chip whose own MAC costs 0.23 pJ.
      The encoders are billed 12.3x the entire MAC array, for being switched on.
```

### 5.2 The accelerator's side is not credibly characterised

**(a) It never reaches the report.** `parse_stats` (`timeloop.py:1050-1096`) reads
`Energy (total)` only from inside each dataspace sub-block; `Leakage energy (total)` sits in
the SPECS block above it and is **discarded**. `PHYS_CATS` has no static category. So the
accelerator's standby energy is charged to **no arm**.

**(b) NEW 2026-09-12 — but the MAPPER has been paying attention to it all along.** Verified
arithmetically on `C128_M128_R3_S3_P28_Q28`, recon2 arm:

```
   filter_glb  Weights Energy (total)   =  143,772.48 pJ
   filter_glb  Leakage energy (total)   =   76,777.31 pJ
                                 sum    =  220,549.79 pJ
   / 115,605,504 computes               =    1.908 fJ/compute
   Timeloop printed                     =    1.91  fJ/compute   <-- LEAKAGE IS IN IT
   (without leakage it would print          1.24)
```

`Energy:` and therefore `EDP(J*cycle)` — the mapper's objective — **already include
leakage**. So the asymmetry is sharper than "leakage is missing":

```
   +----------------------------------------------------------------------+
   |  THE MAPPER      sees leakage, at prices 10^3-10^4 too low            |
   |  THE REPORT      throws the same number away entirely                 |
   +----------------------------------------------------------------------+
   This must be stated in FINDINGS; "leakage is absent" is wrong as written.
```

**(c) The prices are not believable.** From a real cached ERT, `leak` per instance per cycle
at 1 GHz:

| component | leak (pJ/cycle) | implied power | plausible? |
|---|---:|---:|---|
| DRAM | **0** | 0 µW | no |
| ifmap_glb (52 kB SRAM) | 0.00136375 | **1.36 µW** | **no** |
| psum_glb (48 kB SRAM) | 0.00126099 | 1.26 µW | no |
| filter_glb (8 kB SRAM) | 7.41733e-05 | 0.074 µW | no |
| ifmap_spad (×168 RF) | **0** | 0 µW | no |
| psum_spad (×168 RF) | **0** | 0 µW | no |
| weights_spad (×168 RF) | 1.45739e-06 | 0.0015 µW | no |
| mac (×168) | 0.00784449 | 7.84 µW | ? |
| **BCH(63,30) encoder**, for scale | 0.014681 | **14.68 µW** | measured by DC |

One line that captures it, from the same cached shape:

```
   filter_glb leakage, reference arm :        51.04 pJ    (the SRAM itself, whole layer)
   filter_glb leakage, recon2 arm    :    76,777.31 pJ    (SRAM + the ERT-bumped engine)
                                          ~1,500x the SRAM it sits next to
```

#### 5.2c — WHY. Four independent causes, all failing silently

| # | component | cause |
|---|---|---|
| 1 | **DRAM** | `CactiDRAM.leak()` is `return 0` — a literal. Accelergy synthesises the missing action and the estimator answers 0. |
| 2 | **`ifmap_spad`, `psum_spad`** | the leakage value **is 0 in the source table**: `aladdin_register.csv` row `40nm,...,0,5.98E+00,leak|update`. The wrapper asserts only that a leak action *exists*. |
| 3 | **address generators** | the **Neurosim plug-in** bid accuracy 70, tied `Aladdin_table`, won, and returned **0 pJ** — sometimes after crashing writing its `.cfg` into a read-only SIF. **Fixed in Phase 0** (`hpc/tl.sh` binds a writable copy) but **does not land until the caches are regenerated**. |
| 4 | **the SRAMs** | **no unit bug — the device model is wrong.** `default_cfg.cfg` pins `itrs-lstp` (Low STandby Power) at 300 K and nothing can select otherwise. |

| cell type | temp | 52 kB standby leakage | vs the study's value |
|---|---|---:|---:|
| `itrs-lstp` (**in use**) | 300 K | **1.364 µW** | 1× |
| `itrs-lop` (LP process — where an Eyeriss-class design belongs) | 300 K | **1.147 mW** | **841×** |
| `itrs-hp` | 300 K | 25.29 mW | **18,500×** |

**The incoherence:** MAC leakage comes from Aladdin's 40 nm *commercial-library* data; SRAM
leakage from *ITRS-LSTP* CACTI. The model therefore asserts that **108 kB of SRAM (2.6 µW)
leaks ~500× less than 168 multipliers (1.32 mW)**. Mixed device models, not a magnitude error
in either tool. **The error is 10³–10⁴, always under-counting.**

`env.sh` §6 already holds the replacement densities (`ECC_LEAKAGE_NW`: `sram_bit=2.693`,
`rf_bit=70.0`, `mac_instance=7844.9`) — Phase A wires them.

### 5.3 What the reconstruction "idle" number contains

`data/dc/BCH_N63_results.json`, entry `BCH_63_30_t6`:

| field | value | |
|---|---:|---|
| `idle.dynamic` | 2816.4 µW | clock tree + registers toggling, inputs held |
| `idle.leakage` | **14.6811 µW** | true static |
| `idle.total` | 2831.0811 µW | **what the ungated model charges** |
| `active.dynamic` | 4195.0 µW | |

So the charged number is **99.48% clocked-idle dynamic, 0.52% true leakage** — *if* the DC
report's "dynamic" means what it appears to mean.

> **OPEN EVIDENCE GAP (non-blocking).** Design Compiler's power report splits into
> *internal*, *switching* and *leakage*, and how much of the internal component is
> clock-network power (removable by gating) versus register internal power (which survives
> it) is **not determinable from this JSON**. The JSON points at
> `results/report_snapshots/BCH_63_30_t6/idle/power.rpt`; **those reports are not in this
> repository, nor in the sibling `ECC-CODE-Engine` or `RECC` projects.**
> **Fallback, in force:** `ECC_RECON_CLOCK_GATING_PCT=99.5` is the default and `PCT=0` is
> the pessimistic bound, and **§11 requires both to be reported side by side.** Locating or
> re-running the reports is a background task, not a gate.

### 5.4 What gating does to the ranking — measured at PCT=0, projected at 99.5

resnet18 — baseline 19,494.5 µJ, embedded 11,741.7 µJ:

| bar | PCT=0 (measured) | vs embedded | PCT=99.5 (projected) | vs embedded |
|---|---:|---:|---:|---:|
| R1 | 10,701.9 | +8.86% | 10,674.4 | +9.09% |
| R2 | 10,696.3 | +8.90% | 10,668.8 | +9.14% |
| R3 | 11,099.7 | +5.47% | 10,682.6 | +9.02% |
| **R4** | 15,728.9 | **−33.96%** | 10,651.6 | **+9.28%** |
| R5a | 15,958.5 | −35.91% | 11,501.5 | +2.05% |

> **The ranking inverts entirely: R4 from worst to best.** Which is exactly why §11 requires
> both columns on every figure. The gap between them is the value of one line of RTL and the
> reader is entitled to see it.

### 5.5 R5a gains far less than R4, and that is correct

| bar | codewords reconstructed | why |
|---|---:|---|
| R4 @ spad input | 11,237,539 | once per weight **written** into the scratchpad |
| **R5a @ spad output** | **230,358,520** | once per weight **read** — per multiply, **20× more** |

R5a's work term alone is `4.2096811 pJ × 230.4 M = 969.7 µJ`, **83% of its reconstruction
cost**. R4's penalty was almost all standby and evaporates under gating; **R5a's is real work
and survives it.** Correct for a boundary placed after the reuse buffer.

### 5.6 The three sweep figures are nearly immune

`build_stacks()`'s recon column has the same cycle-proportional term (`ecc.py:412-413`) but
pins `recon_engines = 1`. Measured: idle = 32.24 µJ of an 8,600.8 µJ column = **0.375%**. So
`BCHsweep`, `ModelSweep` and `ArchitectureSweep` barely move under any of this; only
`ReconSweep_optimiser__*` does.

---

## §6 — Defect 3: one mapping, six chips  **(NEW 2026-09-12)**

### 6.1 What is mapped today, and what is borrowed

`ECC_RERUN_OPTIMISER=1 ECC_RECON_ERT_AWARE=1 bash hpc/map_ert_arms.sh` submits **3 jobs**
per shape. `recon.ert_arms()` derives which boundaries qualify, from
`recon.ert_injectable()`'s three conditions — storage site, `reads`/`fills` counter, not the
innermost level's reads.

```
   +-------------+                                   +---------------------------+
   |  reference  |---- job 3 -------> own plan ----->|  baseline, embedded, R1,  |
   +-------------+                                   |  R3, R5a  ALL BILLED HERE |
   +-------------+                                   +---------------------------+
   |     R2      |---- job 1 -------> own plan ----->  R2
   +-------------+
   +-------------+
   |     R4      |---- job 2 -------> own plan ----->  R4
   +-------------+
```

The justification was that a boundary whose encoder cost the mapper cannot trade cannot
choose a different plan, so borrowing the reference's plan is exact. **That justification is
wrong for two of the three borrowers, and it is wrong TODAY, independently of Defect 1.**

### 6.2 The three things that are wrong

**(a) R3 is borrowing the wrong plan.** From §3.2, R3's narrowed-storage set is
`{filter_glb}` — **identical to R2's**. Its geometry is R2's chip, not the reference's. The
network stage it adds has no width and no timing, so it changes nothing the mapper can see.
The correct cheap answer is *borrow R2's plan*; the correct exact answer is *give R3 its own
job* (R2's geometry, without R2's ERT bump). Borrowing the reference's is the one option that
is definitely wrong.

**(b) R5a is a chip that has never been mapped.** It is the only bar that narrows
`weights_spad`, which **doubles that level's effective capacity** (`Size: 32 → 64` items) at
byte-identical per-access read/write/leak. No evaluator post-processing can re-tile a loop
nest, so the reference plan cannot use capacity it does not know exists.

> **Expected outcome, stated in advance so the result is a test and not a surprise:**
> FINDINGS 7.8 measured that a bigger `weights_spad` on Eyeriss v1 changes nothing — flat to
> ×32 capacity at 1% fill — because v1's refetch is set by the DRAM-level loop order over
> `P`/`Q` and a weight tile cannot index those. **So R5a's own plan will probably reproduce
> the reference's on this design.** The point is that it will then be *shown* rather than
> assumed, and the same is not expected to hold on `simple_weight_stationary`, which has both
> a weight GLB and a stationary weight register.

**(c) The borrowing is measurably not harmless.** Every cached `map.txt` diffed,
`eyeriss_like_wglb`, 43 shapes:

```
   R2's own plan differs from the reference's on    9 / 43 shapes   (21%)
   R4's own plan differs from the reference's on   13 / 43 shapes   (30%)

   every differing shape is a 1x1 conv (R1_S1) or a depthwise layer (C1_M1_R3_S3).
   resnet18's 3x3 shapes are unaffected -- which is how the premise
   "mapping does not affect these boundaries" survived: it was true for
   resnet18 and got generalised.
```

**The mapper is already sensitive to arm configuration on a fifth to a third of shapes.**

### 6.3 `ert_injectable()` condition 3 has weakened under clock gating

Condition 3 excludes R5a because *"the innermost level's reads equal the MAC count, which no
mapping can move, so a constant added to every mapping cannot move the argmax."* That was
sound when the ERT bump was one row. **Phase 0's clock gating made it two:**

```
   per access :  incremental + idle x g       on `reads`  -> MAC count, CONSTANT  (unchanged)
   per cycle  :  idle x (1 - g)               on `leak`   -> Timeloop bills it as
                                                             leak x UTILIZED INSTANCES x CYCLES
                                                          -> MAPPING-DEPENDENT
```

At `g = 99.5%`: `idle × (1−g) = 2.8311 × 0.005 = 0.01416 pJ/cycle/instance`. Over 168
instances that is **2.38 pJ/cycle**, against the MAC array's own `0.00784449 × 168 = 1.32
pJ/cycle`. **Not negligible, and not constant.** The exclusion argument therefore no longer
holds as written and must be re-derived in Phase B (B3).

### 6.4 What the arm list has to become

Under Phase C the distinguishing axes are three, not one: `datawidth: q`, the ERT bump, and
the per-dataspace bandwidth scale.

| arm | `datawidth: q` on | bandwidth scale `{Weights: …}` on | ERT bump | distinct? |
|---|---|---|---|---|
| `reference` | — | — | — | **#1** |
| `recon1` | — | `DRAM` | — | **#2** |
| `recon2` | `filter_glb` | `DRAM`, `filter_glb` | `filter_glb.read` + `leak` | **#3** |
| `recon3` | `filter_glb` | `DRAM`, `filter_glb` (+network: no-op) | — | **#4** |
| `recon4` | `filter_glb` | `DRAM`, `filter_glb` (+networks: no-op) | `weights_spad.write` + `leak` | **#5** |
| `recon5` | `filter_glb`, `weights_spad` | `DRAM`, `filter_glb`, `weights_spad` | *re-derive, §6.3* | **#6** |

```
                     TODAY                            AFTER PHASE B+C
        3 jobs/shape, 3 borrowers            6 jobs/shape, 0 borrowers
        = 3 x 43 = 129 jobs (all layers)     = 6 x 43 = 258 jobs (all layers)
        = 3 jobs  (one layer)                = 6 jobs  (one layer)
```

### 6.5 One thing you cannot shortcut

It is tempting to predict which arms need their own job from the reference run's throttling.
**You cannot.** A bandwidth scale changes only the throttling check, never the energy
(measured bit-identical, §7.1), so it can move the argmax only where *some candidate* plan was
bandwidth-limited. The `fc` layer in §4.5 is exactly that case: the *chosen* plan shows
throttling 1.00, yet a 32-PE candidate was rejected because it would have throttled. **A
reference plan that never throttles does not prove the relief is a no-op.** Run the arms.

---

## §7 — What the experiments settled

Three investigations ran 2026-09-11. Full detail in Appendix A.

### 7.1 The latency fix works mechanically — CONFIRMED

`per_dataspace_bandwidth_consumption_scale: {Weights: K/N}` multiplies **one dataspace's**
bandwidth demand (`buffer.cpp:2556`) and is declared in `timeloopfe` v4 (`arch.py:538`). The
field is printed in every cached stats file as `Bandwidth Consumption Scale : 1.00`, so the
version in the container supports it. On fixed mappings with off-chip bandwidth binding:
**−17.28%** and **−21.94%** cycles, with dynamic energy **bit-identical**. A misspelled
dataspace name exits with an error rather than being ignored, so it cannot pass as a no-op.

> **Which factor to use, per stage** — `stream` vs `aligned` packing is already a knob and the
> scale must match it, or the mapper and the evaluator disagree about the same wire:
> **DRAM is a bit stream → `K/N` = 0.47619.  An on-chip level holds whole weights at
> `q` bits → `q/8` = 0.5** at BCH(63,30). They differ by 5% and using one everywhere is a
> silent inconsistency.

### 7.2 A binding off-chip limit DOES change the chosen mapping — CONFIRMED

12 real mapper searches. With a binding `read_bandwidth` the mapper picks a different plan on
both shapes; one avoids throttling by shrinking the spatial fold (96 of 168 PEs, 57%
utilisation) — it spends parallelism to buy bandwidth compliance. **At LPDDR4-3200 (12.8
items/cycle) the limit never binds and nothing changes.**

### 7.3 The mapping-mediated ENERGY hypothesis is REFUTED

The hypothesis was: because latency was never modelled, the mapper picked worse plans, so
recon's energy is overstated. **Measured: −0.750 µJ (−0.164%) on one shape and exactly zero
on the other.** And against the same bandwidth *without* weight relief, the relieved plan
costs **+30.3% MORE energy** for 25% fewer cycles: under EDP the mapper spends bandwidth
relief on **latency, not energy**.

> **Record this plainly: "energy is overestimated because latency was not modelled" is TRUE
> for the idle term (§5) and FALSE for the mapping (§7.3).** Note the distinction from
> Defect 3: §7.3 says relief does not improve *energy*; §6 says the borrowed plan is the
> *wrong plan* — different claims, both true.

### 7.4 Leakage in those runs was ~100% irrelevant

Across every cell, leakage was 0.00033%–0.00098% of total energy, and it moved the *wrong*
way (it grows with cycles). Consistent with §5.2: the prices are too small to matter, which
is itself the problem.

---

## §8 — The knobs

All already declared in `env.sh` unless marked NEW. **`env.sh` is the only file you edit.**

| knob | section | default | meaning | phase |
|---|---|---|---|---|
| `ECC_RECON_CLOCK_GATING_PCT` | §6 | `99.5` | fraction of `idle` removed when the engine is gated off. `0` = the ungated pessimistic bound. | done |
| `ECC_ENERGY_MODEL_REV` | §9 | *(empty)* | any non-empty value re-fingerprints the whole matrix. **The deliberate cold.** EMPTY hashes byte-identically to every pre-knob fingerprint. | C |
| `ECC_STATIC_ENERGY` | §6 | `0` | `1` charges component standby energy to **all three arms** | A |
| `ECC_LEAKAGE_NW` | §6 | `sram_bit=2.693`, `rf_bit=70.0`, `mac_instance=7844.9` | replacement leakage densities, in nW (POWER — no cycle-period rescaling) | A |
| `ECC_DRAM_BANDWIDTH_MBPS` | §6 | `480` | off-chip limit in MB/s; §10 converts to words/cycle. EMPTY = unlimited = today. | A (roofline) / C (arch) |
| `ECC_ARCH_CLOCK_MHZ` | §6 | `eyeriss_like_wglb=200` | per-design clock → `ECC_GLOBAL_CYCLE_SECONDS`. **In the fingerprint.** | C |
| `ECC_LATENCY_MODEL` | — | `0` | **NEW.** `1` applies `latency_post.py`'s roofline. | A |
| `ECC_RECON_BW_SCALE` | — | `0` | **NEW.** `1` emits `per_dataspace_bandwidth_consumption_scale`. **Colds every cache.** | C |
| `ECC_ONCHIP_BW_BITAWARE` | — | `0` | **NEW.** `1` scales a narrowed level's declared `read_/write_bandwidth` by `8/q`. **Colds every cache.** §4.5. | C |

> **TRAP, already documented in `env.sh` §6 and repeated because it costs 5× if missed:**
> `ECC_RECON_IDLE_PJ` is pJ **per cycle** measured by DC at a 1 ns clock, so at 200 MHz it
> must be rescaled (`×5`). `ECC_LEAKAGE_NW` is **power in nW** and must NOT be. The rescaling
> belongs in `config.py` where the DC tables are read, never at the point of use.

---

## §9 — THE PHASE PLAN

```
  PHASE 0  DONE ........ five fixes already in the code                      (§13)
     |
  PHASE A  report what is already there .... no compute, no cold .... 1 session
     |        A1 standby energy reaches the report, all three arms
     |        A2 latency_post.py roofline + the first latency tables
     |        A3 the reporting rules (incl. the old Issue 9)
     |        A4 the FINDINGS correction about who sees leakage
     v
  PHASE B  one mapping per boundary ....... no compute, no cold .... 1 session
     |        B1 R3 stops borrowing the reference's plan
     |        B2 the arm list becomes 6
     |        B3 re-derive ert_injectable() condition 3 under gating
     |        B4 --dry-run / --progress prove the new arm list
     v
  PHASE C  THE SINGLE COLD PASS ........... HOURS ................. 2 sessions
     |        C1 author every architecture change + validate (session 1)
     |        C2 launch the 6-arm map, collect, regenerate (session 2)
     v
  PHASE D  transformer workload ........... hours ................. 1 session
     v
  PHASE E  prompt_7.1's buffer sweep
```

---

### PHASE A — report what is already there
**No mapper runs. No cache cold. One session.**

Phase A adds no physics; it stops throwing away numbers the toolchain already produces and
computes the roofline outside Timeloop. It must land first because its latency tables tell
you whether Phase C is worth its hours.

| step | what | where |
|---|---|---|
| **A1** | **Charge component standby energy to ALL THREE arms.** Add a `Standby` physical category. Source the densities from `ECC_LEAKAGE_NW` × the level's stored bits (from the cached ART area / declared geometry) × that bar's own cycles. Gate on `ECC_STATIC_ENERGY`; default stays `0`. | `energy.py`, `config.py` |
| **A2** | **`latency_post.py`** — a roofline computed after mapping, mirroring `noc_post.py`. `cycles = max(compute, each level's own limit, off-chip items / B)`, with a recon bar's off-chip items = `inputs + outputs + weights × K/N`. Gate on `ECC_LATENCY_MODEL`. **Include the on-chip levels, not only DRAM** — §4.5 is why. | `latency_post.py` **(NEW)** |
| **A3** | **The reporting rules of §12** into `Config.recon_caveats()`, so they land in the manifest beside every figure. Includes the old Issue 9's replacement wording. | `config.py` |
| **A4** | **Correct FINDINGS**: the mapper's objective includes leakage (§5.2b); it is `parse_stats` that drops it. The current text says leakage is absent, which is wrong as written. | `FINDINGS.md` |

**GATE — all four must pass before Phase B:**

```
  1. ECC_LATENCY_MODEL=1 at UNLIMITED bandwidth reproduces the cached cycle counts
     EXACTLY, on all 43 shapes.          <-- the roofline is not allowed to invent time
     mutation: perturb one level's cycles; the test must fail.
  2. ECC_STATIC_ENERGY=0 reproduces today's totals to the pJ.
     ECC_STATIC_ENERGY=1 moves BASELINE, EMBEDDED and RECON -- if only one moves, stop.
  3. bash run.sh baseline --eval  ->  Task 1 and 2 totals unchanged.
  4. The latency ceiling test: ceiling = weight share x (1 - K/N), to 2 dp.
     mutation: change K/N; the test must fail.
```

**Deliverable:** the first real latency table, and the first figure on which the accelerator
and the reconstruction engines are charged standby power on the same terms.

---

### PHASE B — one mapping per boundary
**No mapper runs. No cache cold. One session.** This is Defect 3.

| step | what | where |
|---|---|---|
| **B1** | **R3 stops borrowing the reference's plan.** Its geometry is R2's (§3.2). Bill it from R2's plan, or give it its own arm — decide from B3's outcome, but **do not leave it on the reference.** Whichever is chosen, the bar's record must name the plan it was billed from (prompt_6 RULE 4). | `recon.py`, `experiments/recon.py` |
| **B2** | **The arm list becomes 6.** `recon.ert_arms()` currently answers "which boundaries have an ERT-injectable encoder". That is no longer the right question — the right question is **"which boundaries are a distinct chip to the mapper"**, over all three axes of §6.4. Introduce `recon.mapper_arms(arch, cfg)` beside `ert_arms()`; `ert_arms()` keeps its present meaning for the ERT bump alone. `hpc/map_ert_arms.sh` fans out over `mapper_arms()`. | `recon.py`, `hpc/map_ert_arms.sh` |
| **B3** | **Re-derive `ert_injectable()` condition 3** under clock gating (§6.3). The per-cycle `leak` row is mapping-dependent, so "a constant cannot move the argmax" no longer follows. Either widen the condition or record, in the docstring, the measured reason it still holds. **Derive it — never `if key == "recon5"`.** | `recon.py` |
| **B4** | **Prove the new list without spending compute.** `bash hpc/map_ert_arms.sh --dry-run` must print **6** arms and the jobs it would submit; `--progress` must show which are already cached and which are cold. | `hpc/map_ert_arms.sh` |

**GATE:**

```
  1. --dry-run prints exactly 6 arms for eyeriss_like_wglb and the right count for
     every other supported design (v2 has four boundaries, not five).
  2. Every arm's cache slug is DISTINCT. Two arms with byte-identical YAML sharing one
     directory is the failure prompt_6 RULE 4.4.5 exists to prevent.
     mutation: force two arms to the same slug; the test must fail.
  3. test_every_placement_space_is_valid_for_every_supported_design still passes.
  4. With NO new maps run, every bar still reports a number, and every borrowed bar
     NAMES the plan it borrowed. R3 must name R2 (or itself), never `reference`.
  5. bash run.sh validate && bash run.sh diagnose  ->  clean, and diff against the
     pre-Phase-B output.
```

**Deliverable:** a launcher that will map six chips, and a figure in which no bar is billed
from a plan belonging to a different chip. **Do not submit the maps in this phase.**

---

### PHASE C — the single cold pass
**HOURS of SLURM. Two sessions: C1 authors and validates; C2 launches and collects.**

> **This is the only rebuild. Everything that changes what the mapper sees goes in here,
> together.** Separately these are five multi-hour rebuilds.

#### C1 — author and validate (session 1, no jobs submitted)

| step | what | colds? |
|---|---|---|
| **C1.1** | Declare DRAM `read_bandwidth` / `write_bandwidth` on the arch from `ECC_DRAM_BANDWIDTH_MBPS` and the design's own `ECC_ARCH_CLOCK_MHZ`. | yes |
| **C1.2** | Emit `per_dataspace_bandwidth_consumption_scale` on the stages in each arm's `reduced` set (`ECC_RECON_BW_SCALE=1`). **Use `K/N` at DRAM and `q/8` on chip** (§7.1). | yes |
| **C1.3** | **Make a narrowed level's declared bandwidth bit-aware** (`ECC_ONCHIP_BW_BITAWARE=1`): `read_bandwidth × 8/q`. This is §4.5's lever and it is the one that reaches the `fc` layers. | yes |
| **C1.4** | **Cite the on-chip bandwidth numbers in `provenance.yaml`,** or record that they are reference-design inheritance. `filter_glb`'s `read_bandwidth: 16` is currently the binding constraint on every FC layer and is cited nowhere. | no |
| **C1.5** | `ECC_ARCH_CLOCK_MHZ=200` for Eyeriss v1. | yes |
| **C1.6** | MAC priced at 0.23 pJ inside the mapper's ERT (was evaluator-side only). | yes |
| **C1.7** | Banked GLB geometry, so `n_banks` reaches CACTI (the log shows `n_banks=1`); absorbs the dead `bankscale` factor. **While you are there, check `filter_glb`'s `depth`:** it is `256 × 64 b = 2 kB`, while its own comment and JSSC 2017 both say **8 kB** (two banks of 512 × 64 b → `depth: 1024`), and `ifmap_glb` in the same file uses the 512-entries-per-bank convention (`6656 = 13 × 512`). **Confirm or correct before the cold pass — `filter_glb`'s capacity and bandwidth are exactly what R2's entire saving rides on.** | yes |
| **C1.8** | **SET `ECC_ENERGY_MODEL_REV`** (e.g. `2026-09-12-neurosim-adders`) so Phase 0's Neurosim price correction actually lands. An estimator fix does NOT move the fingerprint on its own. | yes — deliberately |

**C1 GATE:**

```
  1. bash run.sh validate && bash run.sh diagnose  -> clean.
  2. The fingerprint MOVED for every (arch, model) pair -- print old and new.
     A cold pass that does not change the fingerprint is a cold pass that will
     silently reuse stale entries.  This is the whole reason ECC_ENERGY_MODEL_REV exists.
  3. bash hpc/map_ert_arms.sh --dry-run  ->  6 arms, new fingerprint, nothing cached.
  4. A misspelled dataspace in the bandwidth scale exits NON-ZERO (`Weightz:`),
     so the attribute cannot pass as a silent no-op.
  5. ECC_ONCHIP_BW_BITAWARE=1 raises filter_glb's declared read bandwidth from
     16.00 to 32.00 in the narrowed arms and leaves it at 16.00 in the reference.
```

#### C2 — launch and collect (session 2)

```
   ECC_RECON_LAYER=all ECC_RERUN_OPTIMISER=1 ECC_RECON_ERT_AWARE=1 \
       bash hpc/map_ert_arms.sh

   6 arms x 43 shapes = 258 jobs of ECC_MAP_CPUS cores, one dependent eval.
   Narrow first:  ECC_RECON_LAYER=<one layer>  -> 6 jobs, and check the gate
                  before committing the full matrix.
```

**C2 GATE:**

```
  1. --progress shows solved = 43 for all 6 arms at the NEW fingerprint.
  2. sibling_fingerprints() reports NO `!! FINGERPRINT WARNINGS` block.
     A comparison across two fp- directories is a comparison of two ARCHITECTURES.
  3. Every arm's stored ERT is read back before a number is used (prompt_6 RULE 4.4.5).
  4. The `fc` shapes show a CHANGED cycle count in the narrowed arms.
     If they do not, C1.3 did not land -- stop and find out why before redrawing.
  5. PE-count differences between an arm's own plan and the reference are REPORTED
     per shape (both counts, cycles, Timeloop EDP ratio) -- never refused.
  6. FINDINGS 2.9 regenerated with BOTH gating columns (§11 item 7).
```

---

### PHASE D — the transformer workload
**One session, hours of compute.**

The ceiling on these CNNs is 6–14% (§4.6) because weights are a minority of traffic. On a
**batch-1 transformer every layer is `fc`-shaped**, so Reason 1 disappears and the ceiling
approaches `1 − K/N = 52.4%` — reached independently by the off-chip analysis (A.6) and the
on-chip `filter_glb` analysis (§4.5). **This is where the latency claim is worth making.**

**GATE:** the measured ceiling lands near 52%, and the weight share of traffic is reported
beside it so the number can be checked by arithmetic.

---

### PHASE E — `prompt_7.1.md`'s buffer-size sweep

Run after Phase D. See that file.

---

## §10 — Tests

Project practice: **property tests on real cached data, plus deliberate breakage.**

| test | asserts | breakage that must make it fail | phase |
|---|---|---|---|
| `test_latency_roofline_matches_timeloop` | at unlimited bandwidth the roofline reproduces cached cycles exactly | perturb one level's cycles | A |
| `test_latency_ceiling` | ceiling = weight share × (1 − K/N), 2 dp | change K/N | A |
| `test_static_energy_is_symmetric` | `ECC_STATIC_ENERGY=1` moves all three arms; `=0` reproduces today to the pJ | charge it to recon only | A |
| `test_leakage_is_parsed` | `Leakage energy (total)` reaches a reported category | drop the regex | A |
| `test_mapper_arms_are_six` | `mapper_arms()` returns 6 distinct arms for `eyeriss_like_wglb`, 5 for v2 | remove one axis from the distinctness key | B |
| `test_arm_slugs_are_distinct` | no two arms share a cache slug | force a collision | B |
| `test_no_bar_borrows_a_foreign_plan` | every bar's record names a plan whose geometry matches its own `reduced` set | put R3 back on `reference` | B |
| `test_ert_condition3_is_derived` | condition 3's outcome follows from the gating percentage, not a key name | set `PCT=0` and `PCT=99.5`; the answer must be derived either way | B |
| `test_bandwidth_scale_is_parsed` | a bad dataspace name is rejected | `Weightz:` must exit non-zero | C |
| `test_onchip_bandwidth_is_bit_aware` | a narrowed level's declared bandwidth scales by `8/q` | pin it to the reference value | C |
| `test_gating_reproduces_ungated` | `PCT=0` reproduces the pre-gating `reconstruction_uJ` to the pJ | any change to the idle formula | done |
| `test_idle_constants_are_read_not_hardcoded` | constants come from the DC JSON | edit the JSON; numbers must move | done |
| `test_network_engine_cycles_per_layer` | R3 uses per-layer fanout, not the max | restore the max; mobilenet must move 33% | done |

Existing suites must still pass: `test_dilation.py`, `test_code_widths.py`,
`test_every_placement_space_is_valid_for_every_supported_design`.

---

## §11 — Checklist before quoting a number

1. Which gating setting produced it? `PCT=0` and `PCT=99.5` differ by ~193× on the idle term.
2. Is the accelerator's standby energy in or out — and is the **same answer** true for the
   reconstruction engines?
3. What off-chip bandwidth was assumed, and does `provenance.yaml` cite it? **And the on-chip
   bandwidths?** (§4.5 caveat 3.)
4. Is the bar's latency from **its own plan** or a borrowed one — and if borrowed, from which
   arm? (prompt_6 RULE 4; after Phase B, "reference" is only ever correct for R1.)
5. Did the roofline gate pass — cached cycles reproduced at unlimited bandwidth?
6. If R3 is in the table, does it carry **rule R-3** below?
7. Are `PCT=0` and `PCT=99.5` **both shown**, or has one been silently chosen?
8. Is the ceiling printed first? A sub-percent saving without its ceiling reads as a missing
   term rather than as arithmetic.

---

## §12 — Reporting rules that travel with every table

These are not defects. They are facts about the model that a reader cannot infer from the
figure, and each must appear in `Config.recon_caveats()` → the manifest's `title_caveats`.

**R-1 — Latency is flat across boundaries by construction.** Every placement's `reduced` set
contains `dram`, so every placement gets the same off-chip relief. Differences between bars
on a latency figure are mapping noise unless they exceed the per-shape PE-count variation
reported beside them.

**R-2 — The binding levels carry inputs and partial sums.** On all 43 cached shapes only
`ifmap_glb` (11 shapes, worst 0.180) and `psum_glb` (5 shapes, worst 0.610) ever throttle.
Reconstruction cannot touch either. The exception is the FC layers, where `filter_glb` sits
exactly on its declared limit (§4.5).

**R-3 — R3's latency equals R2's by construction, not by measurement.**
*(This replaces the former "Issue 9". It is resolved, not open.)*

> Timeloop has no network timing model: `LegacyNetwork::ComputePerformance()` is an empty
> stub, and network stats blocks carry no `Cycles` and no bandwidth field of any kind.
> `archs/_shared/noc.yaml` declares energy coefficients only, and no interconnect bandwidth
> is cited anywhere in the study. **R3's `reduced` set contains `dram` and `filter_glb`, both
> real storage levels, so R3 receives exactly the same latency saving as R2 and R4. What is
> structurally unmodelled is the INCREMENTAL benefit of the array multicast carrying
> reduced-width words — that is, the difference between R3 and R2.**
>
> Report it as *"R3 = R2 by construction"*, never as *"R3 shows 0% latency benefit"*. The
> second sentence claims a mechanism was tested and found ineffective; it was not tested.
>
> **It is not unfixable, only unmodelled.** NoC *energy* is already charged outside Timeloop
> by `noc_post.py`; a `latency_post.py` roofline could carry a declared NoC items/cycle limit
> the same way. That needs a cited interconnect bandwidth number, which JSSC 2017 may or may
> not supply. Until one exists, R-3 stands.

**R-4 — The mapper sees leakage; the report did not.** Timeloop's `Energy:` and `EDP` include
`Leakage energy (total)` (arithmetic in §5.2b). Before Phase A, `parse_stats` discarded it.
Any statement of the form "leakage is absent from the model" is wrong; the correct statement
names which side dropped it.

**R-5 — Labels.** A constrained/relaxed dataflow is a different accelerator: label bars
"constrained dataflow" and never quote them as the published chip. `simba_like` is
`reference_design` — "Simba-like (reference design)".

**R-6 — `build_stacks()`'s `recon` arm is not a placement.** It is one point applied to every
design at once and no physical boundary does what it does. It must never be quoted as one of
the placement study's boundaries.

---

## §13 — What is already fixed, and what is deliberately left alone

### Phase 0 — landed 2026-09-12, code changed and tested

| what | where |
|---|---|
| **R3's standby energy over-billed ×1.3336 on mobilenet_v2.** The network branch of `engine_cycles_for` charged `max fanout × max instances × TOTAL RUN CYCLES`, discarding per-layer fanout that had already been computed. resnet18 was ×1.0024 (all layers multicast 14 wide); mobilenet's depthwise layers broadcast 2–12 wide, which is where the third went. | `recon.py` |
| `engines_declared` was never populated for network stages — R3's record read *"14 engines, of 0 declared"*. A network's declared count is now the widest broadcast the design performs. | `recon.py` |
| **Clock gating**, `ECC_RECON_CLOCK_GATING_PCT` (default 99.5), in `config.py`, `recon.py`, `ecc.py` **and the ERT bump**, so the mapper and the evaluator price the same engine. `PCT=0` reproduces the pre-gating model to the pJ. | four modules + tests |
| **Reconstruction counted codewords on the BASELINE layout** (3 weights/codeword) instead of the EMBEDDED one (7.875). recon's DRAM is laid out exactly as embedded's, so the sweep arm's incremental term was 2.625× too large. **The baseline keeps 3 and must** — its codeword is stored separately and packs only whole weights. | `config.py`, `ecc.py` |
| **Neurosim plug-in returned 0 pJ** for every address generator (open since 2026-09-08). It wrote scratch into its own read-only plug-in dir, crashed, and reported 0 pJ at accuracy 70%. `hpc/tl.sh` now binds a writable copy. | `hpc/tl.sh` |
| **The fingerprint could not see an estimator change at all** — it hashes the ARCHITECTURE, and Accelergy derives the price list FROM the architecture. `ECC_ENERGY_MODEL_REV` closes it: any non-empty value re-fingerprints everything; EMPTY hashes byte-identically to every pre-knob fingerprint (verified: `1900d8c3` before and after). | `config.py`, `env.sh`, `test_dilation.py` |

> **THE COLD IS DELIBERATE, NOT AUTOMATIC.** The Neurosim fix does not land until the caches
> are regenerated and nothing forces that for you. **Phase C1.8 must set
> `ECC_ENERGY_MODEL_REV`.** Setting it earlier would cold the cache Phases A and B read, for
> no benefit.

### Closed by this revision

| was | now |
|---|---|
| **Issue 9** — "R3 can never show a latency effect" | **CLOSED.** The underlying fact is verified (§4.4, on disk) but the conclusion was too strong. It is now **reporting rule R-3** (§12): R3 = R2 by construction; only the *incremental* network benefit is unmodelled, and a route to modelling it is named. |
| Issue 2 (banked GLB) | Phase C1.7, with the `filter_glb` depth question attached. |
| Issue 5 (chip standby power) | Phase A1. Knobs and densities already in `env.sh` §6. |
| Issue 6 (off-chip speed limit) | Phase A2 (roofline) + Phase C1.1 (arch). |
| Issue 8 (MAC price in the ERT) | Phase C1.6. |
| Issue 14 (per-design clock) | Phase C1.5. |

### Open evidence gap, non-blocking

| what | fallback in force |
|---|---|
| **The DC power reports are missing.** `BCH_N63_results.json` points at `results/report_snapshots/BCH_63_30_t6/idle/power.rpt`, absent from this repo and from `ECC-CODE-Engine` and `RECC`. So the split of `idle.dynamic` into clock-network power (gateable) and register internal power (not) is unverified. | `PCT=99.5` is the default, `PCT=0` is the pessimistic bound, and **§11 item 7 requires both on every figure.** Locating or re-synthesising the reports is a background task. |

### Deliberately left alone

| what | why |
|---|---|
| **DRAM background and refresh energy are 0 for every arm**, so the embedded arm gets no credit for holding fewer bits off-chip | deliberate and conservative; user's call to keep. `ECC_DRAM_BACKGROUND_PJ` / `ECC_DRAM_REFRESH_PJ` exist and default to 0. |
| `build_stacks()` pins `recon_engines = 1`, so the three sweep figures are ~immune (0.375% exposure) | informational; see §5.6 |
| **R4b (a retained-reconstruction register) stays REMOVED** | consecutive weight reuse is 1 on 20 of 21 resnet18 layers. Do not reintroduce one without first re-measuring `consecutive_run` on the target mapping (FINDINGS §2.7). |
| The hypothesis that poor latency modelling degraded the MAPPINGS' **energy** | tested and refuted (−0.16% and 0%), §7.3. Distinct from Defect 3. |

---

## §14 — Files this plan edits

| file | change | phase |
|---|---|---|
| `eccenergy/energy.py` | parse and bill component standby energy, all three arms | A |
| `eccenergy/latency_post.py` | **NEW.** The roofline, modelled on `noc_post.py` | A |
| `eccenergy/config.py` | `ECC_LATENCY_MODEL`; §12's rules into `recon_caveats()` | A |
| `eccenergy/timeloop.py` | keep `Leakage energy (total)` on the path to the report | A |
| `FINDINGS.md` | §5.2b correction; regenerated numbers | A, C2 |
| `eccenergy/recon.py` | `mapper_arms()`; R3's plan; `ert_injectable()` condition 3 | B |
| `hpc/map_ert_arms.sh` | fan out over `mapper_arms()` — 6 jobs, not 3 | B |
| `eccenergy/archs.py` | DRAM bandwidth; per-dataspace scale; bit-aware on-chip bandwidth | C1 |
| `archs/eyeriss_like_wglb/arch_paper.yaml` | DRAM bandwidth; banked GLB; the `filter_glb` depth question | C1 |
| `archs/_shared/provenance.yaml` | citations for the off-chip AND on-chip bandwidths | C1 |
| `env.sh` | the three NEW knobs, one commented block each | A, C1 |
| `eccenergy/tests/test_latency.py` | **NEW.** §10's Phase A rows | A |
| `eccenergy/tests/test_mapper_arms.py` | **NEW.** §10's Phase B rows | B |

**Not edited:** `parity.py`, `baseline.py`, `external_parity()` — still frozen.

---

## Appendix A — Measured evidence

Measured 2026-09-11 and 2026-09-12. No project file was modified; mapper work ran in
`Claude-sandbox/`, which may read the caches and must never write to them.

**A.1 — Narrowing is invisible to time, even when off-chip is the bottleneck**
(`timeloop-model`, fixed mapping):

| variant | Cycles | Energy |
|---|---:|---:|
| shape A, DRAM bw=4, datawidth 8 | 55,008 | 31.43 µJ |
| shape A, DRAM bw=4, **datawidth 4** | **55,008 (0.0%)** | **20.78 µJ** |
| shape B, DRAM bw=1, datawidth 8 | 1,408,001 | 457.16 µJ |
| shape B, DRAM bw=1, **datawidth 4** | **1,408,001 (0.0%)** | **386.41 µJ** |

**A.2 — The per-dataspace bandwidth attribute works** (same shapes, same mappings):

| shape | + `Weights: 0.476190` | change | printed scale (W/I/O) |
|---|---:|---:|---|
| A | 45,504 | −17.28% | **0.48** / 1.00 / 1.00 |
| B | 1,099,045 | −21.94% | **0.48** / 1.00 / 1.00 |

Mutation check: `Weightz:` → `ERROR: ... is not a valid dimension name`, exit 1.

**A.3 — Mapper experiments** (12 searches, SLURM 41803046, ~14 min). Controls: the
no-bandwidth config reproduced the cached study numbers exactly; a repeat run produced a
byte-identical `map.txt`; the ERT was numerically identical across all configs, so every
energy delta is a mapping effect.

| shape | config | mapping | Cycles | Energy µJ | ΔE vs base |
|---|---|---|---:|---:|---:|
| resnet | no bandwidth | base | 688,128 | 457.160 | — |
| resnet | bw 1.0231 | **DIFF** | 1,204,224 | 350.160 | −23.41% |
| resnet | bw + W scale | **DIFF** | 903,168 | 456.410 | **−0.164%** |
| resnet | LPDDR4 12.8 | same | 688,128 | 457.160 | 0.000% |
| mobilenet | no bandwidth | base | 18,900 | 15.760 | — |
| mobilenet | bw 3.6571 | **DIFF** | 23,940 | 13.590 | −13.77% |
| mobilenet | bw + W scale | same as above | 23,940 | 13.590 | −13.77% |
| mobilenet | LPDDR4 12.8 | same | 18,900 | 15.760 | 0.000% |

**A.4 — Which levels are ever the bottleneck**, `eyeriss_like_wglb`, reference arm, all 43
cached shapes (recomputed 2026-09-12, read/write limits checked separately):

| level | dataspace | worst throttling | shapes throttled |
|---|---|---:|---:|
| `ifmap_glb` | Inputs | 0.180 | 11 / 43 |
| `psum_glb` | Outputs | 0.610 | 5 / 43 |
| `psum_spad` | Outputs | 1.000 | 0 / 43 |
| `ifmap_spad` | Inputs | 1.000 | 0 / 43 |
| `filter_glb` | Weights | 1.000 | 0 / 43 |
| `weights_spad` | Weights | 1.000 | 0 / 43 |
| `DRAM` | all | 1.000 | 0 / 43 (no limit declared) |

resnet18: 0 of 12 shapes throttled. mobilenet_v2: 15 of 31, 55.5% of its cycles.

**A.5 — Off-chip demand at 1 GHz:** resnet18 4.26 items/cycle total, 1.13 weights.
mobilenet_v2 9.16 total, 1.00 weights. Against `ECC_DRAM_BANDWIDTH_MBPS=480` at 200 MHz =
**2.4 items/cycle**, so the design is hard off-chip-bound.

**A.6 — Per-layer ceilings.** The fully-connected layers are ~99% weight traffic:

| layer | demand | weights as % of its traffic | ceiling |
|---|---:|---:|---:|
| resnet18 `fc` | 16.2/cyc | 98.6% | **51.6%** |
| mobilenet `classifier.1` | 16.2/cyc | 98.7% | **51.7%** |

**A.7 — `filter_glb` is the binding level on the FC layers** (2026-09-12, reference arm):

| shape | PE util | `filter_glb` weight read demand | declared limit | cycles |
|---|---:|---:|---:|---:|
| `C512_M1000_R1_S1_P1_Q1` | 9.52% (16/168) | **16.00** | **16.00** | 32,000 |
| `C1280_M1000_R1_S1_P1_Q1` | 9.52% (16/168) | **16.00** | **16.00** | 80,000 |
| next-highest weight demand on any shape | — | 9.00 (56.2%) | 16.00 | — |

Share of run: resnet18 `fc` = 32,000 / 11,386,112 = **0.28%**; mobilenet `classifier` =
80,000 / 3,802,736 = **2.10%**.

**A.8 — Mapping divergence between arms** (2026-09-12, every cached `map.txt` diffed,
`eyeriss_like_wglb`, 43 shapes, BCH(63,30), constrained mapspace):

| comparison | identical | **different** |
|---|---:|---:|
| reference vs `recon2` (own plan) | 34 | **9 (21%)** |
| reference vs `recon4` (own plan) | 30 | **13 (30%)** |

Every differing shape is a 1×1 conv (`R1_S1`) or a depthwise layer (`C1_M1_R3_S3`).

**A.9 — Leakage is inside Timeloop's objective** (2026-09-12,
`C128_M128_R3_S3_P28_Q28_ws1_hs1`, recon2 arm): `filter_glb` Weights energy 143,772.48 pJ +
leakage 76,777.31 pJ = 220,549.79 pJ; ÷ 115,605,504 computes = 1.908 fJ/compute; Timeloop
printed **1.91**. Without leakage it would print 1.24. Same shape, three arms:

| arm | Cycles | Energy | map vs reference |
|---|---:|---:|---|
| reference | 688,128 | 421.49 µJ | — |
| recon2 | 688,128 | 421.58 µJ | identical |
| recon4 | 688,128 | 434.87 µJ | identical |

**A.10 — Accelerator leakage as reported today, all levels, resnet18:** 0.033 µJ against
8,733 µJ dynamic (0.0004%) — and see §5.2c for why those prices are not believable.

---

## Appendix B — The sandbox

```
   /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling/Claude-sandbox/
```

`tlstats.py` (stats parser), `make_archs.py` (derives bandwidth per shape, writes variant
YAMLs), `run_one.py` (one mapper search, mirroring `eccenergy/timeloop.py::_map_now`'s inputs
and knobs), `bw_experiment.sbatch`, `collect.py`, `README.md`, `results_table.txt`,
`results.csv`, `outputs/<shape>/<config>/`.

**Rules for anything in there:**

- It may **read** `ecc_energy_study/outputs/`, `results/` and `archs/` freely.
- It must **never write** to them. The mapper cache is hours of compute.
- Mapper jobs go through `sbatch` / `salloc`+`srun`, never a login node, always inside
  `timeloop.sif` (`hpc/HIPERGATOR.md`, `hpc/tl.sh`).
- Prefer `timeloop-model` with a mapping lifted from the cache over `timeloop-mapper`:
  seconds instead of minutes, and it holds the mapping fixed so a comparison is clean.
- Each script carries a one-line header saying what question it answers.
