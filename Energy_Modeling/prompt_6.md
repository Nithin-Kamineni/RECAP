# prompt_6 — Reconstruction-aware mapping

**Put the encoder's energy into the mapper's objective, so the mapping is solved knowing
what reconstruction costs.**

| | |
|---|---|
| **Design** | `eyeriss_like_wglb` (Eyeriss v1) — **the only design in scope** |
| **Extends** | Task 4 (`RECON_OPTIMIZER=True`). Does not replace it. |
| **Produces** | `results/figures/ReconSweep_optimiser.png` (overwrites whatever is there) |
| **Also obeys** | `prompt_3.md`'s constrained mapspace — both levers, its free-sets, its four costs. Do not re-derive them and do not run without them. |
| **Invalidates** | `BCHsweep`, `ModelSweep`, `ArchitectureSweep` — regenerated, see §4.3 |
| **Written** | 2026-09-10. Revised 2026-09-11 (third pass): reorganised; parallel execution made explicit; other designs deferred to Appendix B. |

---

## §0 — Reading order

Read §1–§3 to understand *what is being built*. Read §4 for the four rules that stop the
model double-counting itself — these are where the bugs live. §5–§9 are the mechanics.
§12 is the order to build it in.

| § | Section | Why you need it |
|---|---|---|
| 1 | Terminology | Words used precisely below |
| 2 | The question and the deliverable | What this run answers |
| 3 | The five boundaries of Eyeriss v1 | The thing being measured |
| 4 | **The four rules** | Every subtle bug this design can have |
| 5 | The numbers | ERT arithmetic, the arch, the attribution split |
| 6 | The capacity sanity check | One-line fix, currently wrong |
| 7 | **Running the arms in parallel** | All mapper jobs fan out at once |
| 8 | Configuration | `env.sh` knobs, including two new ones |
| 9 | The figure | Path, bars, labelling |
| 10 | Checklist before quoting a number | Gate |
| 11 | Tests | |
| 12 | **Implementation order** | Where to start |
| A | Files this plan edits | |
| B | Other designs — later, not now | Deferred |

---

## §1 — Terminology

Nothing here is a new concept; this just names the pieces so later sentences are
unambiguous.

| Term | What it means, plainly |
|---|---|
| **mapping / plan** | The loop nest Timeloop chose: which loops sit at which memory level, and how big each tile is. One plan per (architecture, layer). Cached on disk. |
| **arm** | One thing the mapper is asked to solve. Here: the reference chip, plus one chip per boundary that gets its own mapping. Each arm is an independent Timeloop search. |
| **bar** | One column on the output figure: `baseline`, `embedded`, and one per boundary. |
| **boundary / placement** | Where on the weight path the encoder sits. Keyed `recon1 … recon5`. |
| **ERT** | *Energy Reference Table.* Accelergy's price list: for each component, the energy of one `read`, one `write`, one `update`, one `leak`. Timeloop multiplies each price by how often that action happens. |
| **action** | One row of the price list. A weight arriving at a scratchpad is a `write`; a weight leaving it is a `read`. |
| **scalar access** | One weight. A **vector access** is one physical memory word, holding `block_size` weights. Timeloop prices per *vector* and counts per *scalar*. Confusing them is a `block_size`-sized error. |
| **`block_size`** | `width / datawidth` — weights per physical word. At `width: 64, datawidth: 4` it is 16. |
| **`q`** | The reduced weight width in whole bits, `q = round(8 x K/N)`. It is **4** at BCH(63,30). |
| **`E_w`** | Encoder energy per weight = `incremental_pJ_per_codeword / weights_per_codeword`. |
| **engine** | One physical copy of the encoder. A boundary at the PE scratchpad needs one per PE (168); at the GLB, one. |
| **bill** | A bar's complete starting energy, every category, before any reconstruction discount. In code: `base_by_cat` / `base_series`. |
| **sibling** | Another cached plan for the same architecture at a *different* fingerprint. Reading the wrong sibling silently compares two different chips. |

---

## §2 — The question and the deliverable

Task 3 fixes one mapping and prices every boundary on it. Task 4 gives the mapper more
weight capacity and lets it re-solve. **Neither tells the mapper that reconstruction costs
energy.**

This prompt does: put the encoder's energy into the ERT actions whose counts *are* the
encoder's workload, and let Timeloop's own objective trade it off.

**Deliverable:** `results/figures/ReconSweep_optimiser.png`.

---

## §3 — The five boundaries of Eyeriss v1

### 3.1 The weight path and the five positions

Weights travel from DRAM to the multiplier through five stops. The encoder can sit after
any of them. Everything **before** the encoder carries `q`-bit weights; everything **after**
carries 8-bit.

```
   DRAM array -- complete codeword, corrected on the die
        |
        |  DRAM interface: k of n bits      <-- saved under EVERY boundary
        v
   chip ingress ........ R1  recon1     1 engine    count = DRAM weight reads
        |                                           mapping-sensitive: weakly
        v
   filter_glb  (1 instance, 64-bit words)
        |
        +-------------- R2  recon2      1 engine    count = filter_glb READS
        |                               ^^^^^^^^    *** OWN MAPPING + ERT ***
        v
   array NoC -- 14-way multicast across 14 PE columns
        |
        +-------------- R3  recon3     14 engines   count = NoC deliveries
        |                                           no ERT hook: the count lives
        |                                           in noc.yaml, not an action
        v
   column-local distribution to 12 PEs per column
        |
        +-------------- R4  recon4    168 engines   count = weights_spad FILLS
        |                             ^^^^^^^^^^    *** OWN MAPPING + ERT ***
        v
   weights_spad  (168 instances, 16-bit words)
        |
        +-------------- R5a recon5    168 engines   count = spad reads = MACs
        v                                           mapping CANNOT move it
       MAC
```

Stated as which stops carry narrow weights:

| key | encoder sits | stops carrying narrow weights | engines |
|---|---|---|---|
| `recon1` | before the GLB | DRAM | 1 |
| `recon2` | **after the GLB** | DRAM, GLB | 1 |
| `recon3` | after the mesh | DRAM, GLB, mesh | 14 |
| `recon4` | **at the scratchpad input** | DRAM, GLB, mesh, local wires | 168 |
| `recon5` | at the scratchpad output | all five | 168 |

This matches `recon.py`'s `_EYERISS_V1_WGLB_PLACEMENTS` key for key, and
`ECC_RECON_PLACEMENTS[eyeriss_like_wglb]="recon1 recon2 recon3 recon4 recon5"` in `env.sh`.
Both already exist and both already validate — nothing to add for this design.

### 3.2 The contiguity rule — and what it does NOT mean

> **The narrow stops must be an unbroken run from the start.** Stops 1–2 narrow, or 1–2–3
> narrow. Never stop 1 narrow, stop 2 full, stop 3 narrow again — data cannot get thin, fat,
> then thin again on its own.

`validate_placement_space()` enforces exactly this, and Eyeriss v1 passes it today.

**This rule describes what is true INSIDE one bar. It is not an order between bars.**

The five boundaries are **five independent hypothetical chips**, each built and measured on
its own. They are not five stages of one pipeline, nothing flows from `recon1` into
`recon2`, and no bar needs another bar's result:

```
   recon1  = chip A  --------+
   recon2  = chip B  --------+
   recon3  = chip C  --------+---->  five separate measurements
   recon4  = chip D  --------+       of five separate designs
   recon5  = chip E  --------+
```

The numbering is only a left-to-right ordering on the chart, chosen so the encoder moves
steadily deeper into the accelerator. **Every mapper job for these arms is independent and
must be submitted in parallel** — see §7.

### 3.3 Which bars get their own mapping — DERIVED, never hardcoded

**Do not write `if key == "recon2"` anywhere.** Derive the set from the `Placement` record,
so adding a design later needs no new logic:

> A placement is **ERT-injectable** when all three hold:
> 1. its `site_stage` is a **storage** stage (`Stage.kind == "storage"`). DRAM and network
>    stages are excluded: a DRAM read is not a chip action the encoder attaches to, and a
>    network delivery is billed through `noc.yaml`, not through a component action.
> 2. its `site_counter` is `reads` or `fills` — it names one real ERT action.
> 3. it is **not** the innermost weight level's `reads`. That count equals the MAC count,
>    which no mapping can change, so adding a constant to every mapping cannot move the
>    argmax. Detect it as: site stage is the last storage stage on the weight path AND the
>    counter is `reads`.

On Eyeriss v1 that yields:

| bar | ERT-injectable? | why |
|---|---|---|
| `recon1` | no | DRAM stage, not a chip storage action |
| **`recon2`** | **yes** | `filter_glb` + `reads` |
| `recon3` | no | network stage — the count lives in `noc.yaml` |
| **`recon4`** | **yes** | `weights_spad` + `fills` |
| `recon5` | no | innermost level's reads = MAC count, mapping-invariant |

**Two ERT arms.** The code must work at 0, 1 or 2+; zero is legal and means the figure is
Task 3's with the idle term added.

---

## §4 — The four rules

Each rule answers the same question — *when two parts of the model can account for the same
physical thing, which one does it?* — in a different dimension.

### RULE 1 — one owner per physical effect, decided PER BAR

A 64-bit word holds 8 weights at 8 bits and 16 at 4 bits, so on-chip energy per weight
halves. Two places can apply that halving:

```
   arch YAML: datawidth: 4                 evaluator: Packing.storage_scale
          |                                          |
          v                                          v
   Timeloop reports HALF the      ---->      multiplies by 0.5 AGAIN
   energy already                                    |
                                                     v
                                              0.5 x 0.5 = 0.25   WRONG
```

**Measured 2026-09-10, BCH(63,30):** `storage_scale` returns **0.5** on the 64-bit GLB word
AND the 16-bit spad word, while `onchip_narrowing_audit()` reports `n_sites = 0, ok`. The
audit keys off the packing mode's *name* (`== "stream"`); the narrowing is a measured
multiplier. `aligned` is a no-op only when `floor(W/q) == floor(W/8)` — true for the
7-bit-in-24-bit example in its docstring, false for any power-of-two `q` in a byte-multiple
word. No published number is affected: the two sites have never been live in one run.

#### Ownership is a property of the BAR, not of the run

Only the ERT arms get a `datawidth: q` architecture. `recon1`, `recon3` and `recon5` are
evaluated on the ordinary 8-bit reference plan. A single run-wide "the mapper owns
narrowing" switch would therefore deny three bars a saving they physically have:

| bar | storage stops carrying narrow weights | plan it is read from | who narrows, under a run-wide switch |
|---|---|---|---|
| `recon1` | none (DRAM only) | reference | nobody — **correct**, it narrows nothing on chip |
| `recon2` | `filter_glb` | its own `q`-bit plan | mapper ✓ |
| `recon3` | `filter_glb` | reference (8-bit) | **NOBODY — silently zero** |
| `recon4` | `filter_glb` | its own `q`-bit plan | mapper ✓ |
| `recon5` | `filter_glb` + `weights_spad` | reference (8-bit) | **NOBODY — silently zero, twice** |

That would make the figure argue its own conclusion: the two ERT bars would win partly
because the other three were denied their narrowing. `recon5` loses both stops and is the
bar the study most wants to argue *against*.

**`recon3` and `recon5` must NOT be given their own `q`-bit architecture.** That would put
them on a different plan from the bars they are meant to be compared against, and the point
of the figure is that the boundaries are comparable to each other.

**The rule instead: for each bar, for each storage stop carrying narrow weights, the owner
is decided by MEASUREMENT from that bar's own stats file.**

```
   read `Word bits` for that level out of the bar's OWN timeloop-mapper.stats.txt

        Word bits == q            ->  the MAPPER already narrowed it.
                                      evaluator applies storage_scale = 1.0
        Word bits == weight_bits  ->  the mapper narrowed nothing.
                                      evaluator applies Packing.storage_scale as today
        anything else             ->  STOP. the arch and the code disagree.
```

Exactly one owner every time, no switch to set wrongly, and no extra mapper jobs.
**There is no `ECC_RECON_NARROW_AT` knob** — ownership is measured, never declared.

**What to change**

* `Packing.storage_scale()` gains a per-stage bypass driven by the measured `Word bits`.
* `onchip_narrowing_audit()` decides `evaluator_narrows` by **measurement** —
  `storage_scale(word_bits) != 1.0` — and runs **once per bar per stage**. It stays a hard
  refusal if two sites are ever live, and becomes one if **zero** sites are live on a stop
  the placement says carries narrow weights.
* **A fourth site exists and gets the same treatment:** `ecc.build_stacks()`'s recon column
  scales on-chip weight energy by K/N (`base_w[c] * sram_scale`) on **every** evaluation. It
  is safe only because it is normally fed an 8-bit mapping; feed it a `q`-bit one and it
  doubles. Guard it the same measured way.

#### Ownership table

| effect | owner | why |
|---|---|---|
| on-chip storage width | **mapper** for a bar with a `q`-bit plan; **evaluator** for a bar on the 8-bit plan | decided per bar by measured `Word bits` |
| DRAM x K/N | **evaluator**, always | The array still stores all n bits; only k are driven off the die. Declaring DRAM `q`-bit would also shrink the capacity Timeloop thinks it has — a lie about the hardware. `_set_weight_datawidth` already refuses DRAM: keep it. |
| network wire / switching | **evaluator**, always | **VERIFIED 2026-09-10**: on a matched `wdw8`/`wdw4` pair of `C128_M256_R3_S3_P14_Q14_ws2_hs2`, Network 0 reports `Word bits: 8`, `Ingresses: 294912.00`, `Energy (total): 1783888.01 pJ` — byte-identical in both arms. Narrowing a storage level does not narrow the network inside Timeloop. **Do not rewrite `network_word_bits`.** |
| encoder energy, **per access** | **mapper** (ERT `read` / `write`) | RULE 3 |
| encoder energy, **per cycle** | **mapper** (ERT `leak`) | RULE 3 |

---

### RULE 2 — one charging site per arm, and the count must match the action

Every weight crosses the encoder exactly once and pays `E_w` once. Charge it on the GLB read
AND the spad write and every weight pays twice:

```
   16 weights:  1 GLB read    x 2.80096 = 2.80096 pJ  |
                8 spad writes x 0.35012 = 2.80096 pJ  |  = 0.35012 pJ/weight
                                                         but E_w = 0.17506
```

The two adders are the same toll collected at two booths on one road. They describe **two
different chips**, not two views of one:

| arm | encoder sits | `filter_glb` | `weights_spad` | ERT bump | engines |
|---|---|---|---|---|---|
| reference / embedded | none | 8 | 8 | none | — |
| `recon2` | above the network | **4** | 8 | `filter_glb`: `read` + `leak` | 1 |
| `recon4` | at the spad write port | **4** | 8 | `weights_spad`: `write` + `leak` | 168 |

**One `Placement`, one `site_stage`, one `site_counter`, one ERT access action. Never two.**
When a boundary sits exactly between two levels, the site is the **lower-count side** —
also the side with fewer engines. That is a choice of *which chip you are building*, not an
accounting convenience.

#### The count must come from the action you bumped

When the energy is moved back out of the level (§5.3), the number you move must come from
the count of **the specific action you patched**:

* `recon2` bumped `filter_glb.read` -> use `filter_glb` **scalar reads**.
* `recon4` bumped `weights_spad.write` -> use `weights_spad` **scalar fills**.

**Never** use the level's total energy, and **never** use a blended per-access figure got by
dividing a level's total energy by its total accesses (a mixed ~0.44 pJ/access, say). Reads
and writes are priced differently — `filter_glb.read` is 2.75566 pJ while
`weights_spad.write` is 0.243626 pJ — so an average belongs to neither lane and the split
will not reconcile.

#### Assert `updates == 0` before trusting anything

Timeloop storage levels have a third access action, `update` (read-modify-write). It is
normally zero for a weights-only level, but nothing guarantees that on a layer you have not
run yet.

> **Rule: before any number is used, assert `Scalar updates` for Weights at the patched
> level is exactly 0.** If it is ever non-zero, either put the toll on that lane too, or stop
> and say so. It is a one-line assertion; without it a future run on a different layer
> silently under-reports the encoder by the update count.

---

### RULE 3 — incremental and idle are different quantities, and BOTH belong in the ERT

`ecc.py` currently does `value = inc + idle` and multiplies by codeword count. The source
JSON names the two fields `incremental_per_codeword` and `idle_per_cycle`. **Adding a
per-codeword number to a per-cycle number is dimensionally wrong.** They are separate terms:

    E_recon = incremental x events  +  idle_per_cycle x cycles x N_engines

* **incremental** — per actual reconstruction, i.e. per codeword event.
* **idle** — per cycle of the run, for every engine that exists.
* **N_engines** — 1 at the GLB, 14 at the column edge, 168 at the PE scratchpad.

#### 4.3.1 Idle goes in the ERT, via the `leak` action

Accelergy emits a `leak` action for every storage component, and Timeloop bills it **per
instance, per cycle**, inside total energy and therefore inside the mapper's objective.
Verified on the target design's own stats:

```
weights_spad    Instances : 168 (14*12)    Leakage energy (total) :   860.05 pJ
filter_glb      Instances : 1   (1*1)      Leakage energy (total) :    35.28 pJ
                                           Cycles                 : 344064
```

Timeloop supplies the `x instances x cycles` itself, so **the ERT delta is the same number
for both arms** — add `idle_per_cycle` once, per instance, and the engine count falls out of
the hardware:

    filter_glb.leak   += 2.8310811  ->  2.8310811 x   1 x 344064 =   0.974 uJ
    weights_spad.leak += 2.8310811  ->  2.8310811 x 168 x 344064 = 163.6   uJ

Those reproduce the hand-computed idle figures exactly.

#### 4.3.2 Why this is not optional

Measured on this layer (344,064 cycles; the whole embedded run is 483.854 uJ):

| placement | engines | incremental | **idle** | total | % of run |
|---|---:|---:|---:|---:|---:|
| `recon2` GLB output | 1 | 0.052 uJ | 0.974 uJ | **1.03 uJ** | 0.2 % |
| `recon3` column edge | 14 | — | 13.64 uJ | — | 2.8 % |
| `recon4` spad input | 168 | 0.36–0.72 uJ | **163.6 uJ** | **~164 uJ** | **~34 %** |

Idle is **200–450x** the incremental term. Put only the incremental term in the ERT and the
mapper is steered by ~0.5 % of the encoder cost while 99.5 % is invisible. With `leak`
patched too, the mapper sees all of it, and the pressure it feels is the physically right
one: *finish sooner, because 168 encoders burn standby power for the whole run.*

**Verify the multiplier before trusting it.** The stats print `Instances sharing power
gating`, and on one archived `simba_like` level the leakage total corresponded to 16 rather
than the printed 64 instances. Read the multiplier off a real run of *this* design, confirm
it equals the engine count you intend, and state it on the result.

**Verified 2026-09-11 (43 shapes, resnet18 + mobilenet_v2): the multiplier is the UTILIZED
instance count, not the declared one.** `buffer.cpp` sets `leaks_per_cycle` to the max
utilized instances when each instance has its own power gate (lines 2119–2127) and bills
`leak × total_cycles × leaks_per_cycle` (line 2376): 168 on a layer that fills the array,
98 on conv1, 16 on fc. So `N_engines` at a storage site is the utilized instances per layer,
summed as `engine_cycles = Σ_layers utilized × cycles`; the encoder in an unused PE is
power-gated with its scratchpad, exactly as every arm's own leakage already assumes. The
table above holds for `layer3.0.conv1` because it uses 168/168 PEs.

**Interaction to note, not to correct:** `ECC_OPT_METRIC=edp` already multiplies energy by
cycles, and the leak term is energy that grows with cycles. Both are legitimate; the
effective weight on latency simply rises. Say so beside the result.

**Re-derive the incremental column before quoting it.** The `recon4` figure spans
0.36–0.72 uJ because two available caches disagree by exactly `block_size`; on
`multimodel__vic4000__…__mcons__wrelax`, `weights_spad` reports 24,576 fills per instance
x 168 = 4,128,768 scalar fills, which at `E_w = 0.175060` is 0.723 uJ. `recon2`'s 0.052 uJ
reproduces exactly from that cache's 294,912 GLB reads. Recompute both from the cache the
run actually reads, and make sure the count is **scalar**, not vector.

#### 4.3.3 The two tables live in `env.sh`, not in Python

`declare -A` arrays **cannot be exported to a child process**, which is why env.sh flattens
`ECC_RECON_PLACEMENTS` into `ECC_RECON_PLACEMENT_LIST` before exporting. Either flatten
these the same way, or resolve the run's BCH key and export two plain scalars.

    declare -A ECC_RECON_INCREMENTAL_PJ=(   # pJ per codeword
        [BCH_63_57_t1]=1.6574   [BCH_63_51_t2]=1.8995   [BCH_63_45_t3]=1.6383
        [BCH_63_39_t4]=1.4561   [BCH_63_36_t5]=1.5082   [BCH_63_30_t6]=1.3786 )
    declare -A ECC_RECON_IDLE_PJ=(          # pJ per CYCLE per ENGINE
        [BCH_63_57_t1]=1.9359672  [BCH_63_51_t2]=2.2301273  [BCH_63_45_t3]=2.4120856
        [BCH_63_39_t4]=2.7891299  [BCH_63_36_t5]=2.8358254  [BCH_63_30_t6]=2.8310811 )

`ECC_RECON_INCLUDE_IDLE` is retired: idle is always charged, on its own denominator.

#### 4.3.4 What this invalidates, and that is accepted

`ecc.load_recon_energy()` is not local to the placement study. It feeds
`experiments/common.py` and `experiments/sweep.py -> build_stacks()`, whose recon column is
`n_cw_base * recon_pj`. Splitting idle out drops that term from 4.2097 to 1.3786 pJ per
codeword at BCH(63,30) — a 67 % cut — and adds a per-cycle term in its place.

**`BCHsweep`, `ModelSweep` and `ArchitectureSweep` are invalidated by this change and will be
regenerated from the new model.** That is the intended outcome, not a side effect to avoid.
`build_stacks()` is no longer frozen for this purpose; `parity.py`, `baseline.py` and
`external_parity()` remain frozen. `build_stacks()` needs a cycle count to charge idle with
— take it from the same `Raw` record its other inputs come from, and if a `Raw` predates
cycles being recorded, refuse rather than charge zero. `tests/test_baseline_dram.py` calls
`load_recon_energy` and its expectations move too.

---

#### 4.3.5 UNRESOLVED, added 2026-09-11 — is the idle number the right number?

**RULE 3's idle term is under review. Nothing here has changed; this records the open
question so it is not lost.** The full write-up, with the evidence, is **`prompt_7.md` §11
Q1**; this is a pointer, not a second copy.

The term is `E_idle = idle_per_cycle x engine_cycles`, with `idle_per_cycle = 2.8310811 pJ`
for BCH(63,30) taken from `data/dc/BCH_N63_results.json`. That constant is the Design
Compiler **idle-window total** power at 1 GHz, and the same JSON decomposes it as:

| field | value | share |
|---|---:|---:|
| `power_uW.idle.dynamic` | 2816.4 uW | 99.48% |
| `power_uW.idle.leakage` | 14.6811 uW | 0.52% |

So the model charges a free-running clock for every cycle of the whole inference, on engines
measured to be busy **0.62%** of the time at the `recon4` boundary. Because nothing can
stall the pipeline today, `engine_cycles` at a scratchpad boundary is **exactly the MAC
count** (measured: 1,814,073,344 both) — i.e. the term is arithmetically 2.83 pJ per MAC,
**12.3x the energy of the entire 168-MAC array**.

**Do not change this without verification.** Needed first: the DC report's internal /
switching / leakage split, how much of the internal term is clock-network power that gating
would actually remove, whether the synthesised RTL already contained clock gating, and
whether clock gating is even the right technique at a 0.62% duty cycle. **The reports the
JSON names (`results/report_snapshots/BCH_63_30_t6/`) are not in this repository, nor in
`ECC-CODE-Engine` or `RECC`.**

The stake: under a clock-gated model `recon4` would move from **-33.96%** to **+9.28%** vs
embedded — from the worst boundary to the best. A change that large needs more evidence
than one field of one JSON file.

Separately, and independently of the above, `prompt_7.md` §4.2 documents that the
accelerator's own standby power is charged to **no arm at all**, so this term is currently
compared against zero.

---

### RULE 4 — one plan per BAR, not one plan per run

#### 4.4.1 Why the code subtracts instead of calculating

Timeloop has never simulated a reconstruction chip — it simulated an ordinary 8-bit
accelerator. So `evaluate_placement()` does not build a bar from nothing. It starts from the
ordinary chip's complete energy bill and edits it:

```
   start with   the bill: the whole run's energy, every category
   subtract     for each stop now carrying narrow weights, what it no longer spends
   add          the encoder's energy
   = one bar
```

```python
components = dict(base_by_cat)                 # the bill
for cat, saved in saved_by_cat.items():
    components[cat] -= saved                   # the discounts
components["Reconstruction"] += recon_energy   # the new charge
```

**Because a bar is *bill minus discounts*, the bill and the discounts must describe the same
run.** Subtract one machine's savings from another machine's bill and the number describes
no chip at all.

#### 4.4.2 What the code does today, and where it breaks

`evaluate()` already understands this. It swaps in a second plan's bill, per-level energies
and raw record **as a set**:

```python
dil = dilated_view(...) if cfg.recon_optimizer else None
p_wpath       = dil.wpath       if dil else wpath
p_base_w      = dil.base_w      if dil else base_w
p_base_series = dil.base_series if dil else base_series
p_raw         = dil.raw         if dil else raw
```

Then it uses that **one** set for every bar:

```python
results = [evaluate_placement(cfg, arch, p, p_wpath, p_base_w, p_base_series, ...)
           for p in placements]
```

Correct for Task 4, which has two worlds: the reference plan, and one alternative every
reconstruction bar shares. **This prompt has four worlds:**

| bar | which chip / which plan |
|---|---|
| `baseline`, `embedded` | the ordinary 8-bit chip |
| `recon1`, `recon3`, `recon5` | the ordinary 8-bit chip, post-processed |
| `recon2` | `q`-bit GLB **and** a GLB-read/leak toll — its own plan |
| `recon4` | `q`-bit GLB **and** a spad-write/leak toll — its own plan |

`recon2` and `recon4` carry different ERT files, so the search behaves differently and lands
on different plans, with different DRAM traffic and different cycle counts. With one spare
slot, whichever plan is loaded gets used for all five bars: `recon4`'s discounts would be
subtracted from `recon2`'s bill; the three post-processed bars would inherit an encoder toll
their boundary does not have; and their cycle count, which the idle term needs, would come
from the wrong run. **Nothing crashes. Every number looks plausible.**

#### 4.4.3 The fix: a per-bar lookup

**Replace the single set of `p_*` variables with a lookup keyed by placement.** Each bar
carries its own:

| per bar | what it is | why it must be that bar's own |
|---|---|---|
| bill (`base_series` / `base_w`) | the whole run's energy by category | the discounts are subtracted from it |
| `wpath` | per-level energies and access counts | the discounts are computed from it |
| `raw` | DRAM weight reads, mapping ids | decode count and provenance |
| **cycles** | run length | the idle term is `idle x cycles x engines` |
| ERT delta | which action was bumped, by how much | the attribution split must undo exactly it |

Each bar picks its own set **before** the subtraction happens. A bar with no ERT arm points
at the reference set, so nothing about Task 3 or Task 4 changes.

#### 4.4.4 The guards run once per ERT arm

`dilated_view()` performs three checks — is the cache there, are both plans mapped on the
same layers, did the plan actually change (byte-identical loop nest). **All three must now
run per ERT arm.** With two alternatives you need two comparisons, and **they can disagree**:

```
   GLB bump    ->  nest CHANGED   (the mapper avoided GLB reads)
   spad bump   ->  nest IDENTICAL (the mapper had nothing to trade)
```

That divergence is itself a result and must be reported per arm, never collapsed into one
verdict. If an arm's nest is byte-identical to the reference, say **for that arm** that the
ERT changed nothing; do not report the residual ERT-pricing difference as a result.

#### 4.4.5 The wrong-sibling trap — the sharpest edge in this design

`recon2` and `recon4` declare **byte-identical architecture YAML**. Same `datawidth: 4` on
`filter_glb`, same 8 on `weights_spad`, same depths, same widths. They differ **only** in the
ERT file.

`arch_fingerprint()` hashes the patched YAML plus globals plus mapper settings. **If the ERT
is not in that hash, both arms resolve to the same `fp-<hash>` directory**, the second map
overwrites or skips the first, and both bars silently read one plan. `sibling_fingerprints()`
cannot warn you, because there is only one sibling. This is the one place in the design where
a wrong pick leaves no trace at all.

**Three defences, all required:**

1. **Slug.** `archs.effective_variant()` appends an ERT part —
   `ert-<placement key>-<level>-<action>` — so the two arms occupy differently *named* cache
   directories, legible in `ls`.
2. **Fingerprint.** `archs.arch_fingerprint()` hashes the ERT delta itself: level name,
   action name, and delta value at full precision, for every patched action including `leak`.
   Two arms differing only in the toll must hash differently.
3. **Read-back assertion.** Before any number is used, re-open the patched ERT stored beside
   the cache entry and assert the bumped actions are the ones this bar expects and each delta
   is within 1e-9 of its intended value (`E_w x block_size` for the access action,
   `idle_per_cycle` for `leak`). A cache entry whose ERT does not match the bar asking for it
   stops the run and names both.

Record the ERT bump in the mapping sidecar too, so a result file cannot claim the wrong arm
after the fact.

---

## §5 — The numbers

### 5.1 The ERT arithmetic

Timeloop's energy is `sum over actions of (count x ERT action energy)`. An access action's
energy is per **physical (vector) access**, and Timeloop bills `vector_access_energy /
block_size` per weight. So:

    ACCESS ACTION  (read for recon2, write for recon4)

        delta_pJ = E_w x block_size        block_size = width / datawidth, read off THAT
                                           level in THAT arm's patched arch
        E_w      = incremental_pj_per_codeword / weights_per_codeword

    LEAK ACTION  (both arms)

        delta_pJ = idle_per_cycle          Timeloop multiplies by instances x cycles

`weights_per_codeword` is `recon.Granularity.weights_per_codeword` = 63/8 = 7.875. **Do not
hardcode either.** At BCH(63,30): `E_w = 1.3786 / 7.875 = 0.175060 pJ/weight`.

| arm | level | width | datawidth | block_size | action | delta_pJ |
|---|---|---|---|---|---|---|
| `recon2` | `filter_glb` | 64 | **4** | 16 | `read` | **2.80096** |
| `recon2` | `filter_glb` | — | — | — | `leak` | **2.8310811** |
| `recon4` | `weights_spad` | 16 | **8** | 2 | `write` | **0.350120** |
| `recon4` | `weights_spad` | — | — | — | `leak` | **2.8310811** |

Recompute all four at run time; this table is the check, not the source.

For scale: `filter_glb.read` is **2.75566 pJ** today, so `recon2` roughly **doubles** it;
`weights_spad.write` is **0.243626 pJ**, so `recon4` multiplies it by **2.4**. The mapper
will respond, possibly by avoiding the level the encoder sits on — which is the experiment.

### 5.2 The arch each arm declares

Use **prompt_2's quantisation and nothing else.** Do not scale anything by K/N, do not
dilate a depth to N/K, do not touch `width:` (FINDINGS 7.8). The reduced representation is
`datawidth: q` at fixed `width` and `depth`. At BCH(63,30) `q = 4`, and it divides both
published words already — `filter_glb` 64 b, `weights_spad` 16 b — so **no width rewrite is
needed**.

The narrow weights stop AT the boundary, so only the storage levels in `placement.reduced`
get `q`. `archs._set_weight_datawidth()` rewrites every on-chip weight level at once, so it
needs a `levels=()` filter — **mirror `_scale_weight_depth(text, scale, levels=(), ...)`**,
which already takes exactly that parameter. Empty tuple must keep today's behaviour byte for
byte, so no cached fingerprint moves. Add the matching `weight_datawidth_levels` config
field beside `weight_depth_levels`.

DRAM `datawidth` stays 8 on every arm (RULE 1).

**Measured, so you know the mechanism works** (matched `wdw8`/`wdw4` pair, same layer):
`filter_glb` energy 159,221.51 -> 79,610.76 pJ (exactly 0.5), utilised capacity 384 -> 768,
and the nests are **not** identical — `for C in [0:16)` became `for C in [0:8)`. Cycles
unchanged at 344,064. The freed weight room also doubled the input tile
(`ifmap_glb Inputs 6728 -> 13456`): legitimate, same physical buffer, but it means an ERT
bump can surface as **activation** energy moving. Read the category decomposition, not just
the total.

### 5.3 Attribution: a split, not an addition

An ERT-injected arm carries the encoder energy *inside* the level's reported energy.
`evaluate_placement()` also adds a `Reconstruction` term. Adding both charges it twice.
Within one run they are the same number exactly:

    stats-side (access) = vector_accesses x (E_w x block_size) = scalar_accesses x E_w
    evaluator  (access) = events x incremental                 = scalar_accesses x E_w

    stats-side (leak)   = idle x instances  x cycles
    evaluator  (leak)   = idle x N_engines  x cycles

So **move** those two amounts out of the level's category into `Reconstruction`. The stack
then has the identical shape Task 3 produces. Assert each split reconciles to 1e-6 relative.

Three constraints on that arithmetic, from RULE 2 and RULE 4:

* Compute it from **that bar's own** access counts and cycle count, never a reference run's.
* Use the count of **the action you bumped** — `filter_glb` scalar *reads* for `recon2`,
  `weights_spad` scalar *fills* for `recon4` — never the level total, never a blended average.
* Assert `Scalar updates == 0` for Weights at that level first.

For the post-processed bars (`recon1`, `recon3`, `recon5`) there is nothing to move out —
their level energies contain no encoder — so both terms are simply added, with `N_engines`
stated explicitly per placement (1 / 14 / 168) and cycles taken from the reference plan.

---

## §6 — The capacity sanity check: compare against 8/q, not N/K

Task 4 checks that a re-planned arm really got the extra weight room it was supposed to:

```python
got = cap_dilated / cap_reference
if abs(got - nk) > 0.05 * nk:      # nk = N/K
    raise SystemExit(...)
```

That target is right for **depth dilation**, which asks for a physically deeper memory and
can request any fraction — N/K = 2.1 at BCH(63,30). It is wrong for **quantisation**, which
keeps the memory the same size and makes each weight thinner. You cannot store a weight in
3.81 bits, so `q` is rounded to a whole number and the room actually delivered is `8/q`,
**never** N/K:

| code | ideal bits `8K/N` | `q` | room delivered `8/q` | check wants `N/K` | off by | today |
|---|---|---|---|---|---|---|
| BCH(63,57) | 7.24 | 7 | 1.143 | 1.105 | 3.4 % | passes |
| BCH(63,51) | 6.48 | 6 | 1.333 | 1.235 | **7.9 %** | **refuses to run** |
| BCH(63,45) | 5.71 | 6 | 1.333 | 1.400 | 4.8 % | passes by 0.2 pp |
| BCH(63,39) | 4.95 | 5 | 1.600 | 1.615 | 1.0 % | passes |
| BCH(63,36) | 4.57 | 5 | 1.600 | 1.750 | **8.6 %** | **refuses to run** |
| BCH(63,30) | 3.81 | 4 | 2.000 | 2.100 | 4.8 % | passes by 0.2 pp |

Two codes cannot run at all, and the four that pass do so by luck. A check satisfied by
accident is worse than no check: if the `datawidth` edit silently missed a level, this one
might still pass.

**Fix: the target becomes `8/q`.**

```python
want = cfg.weight_bits / q          # exact for every code
if abs(got - want) > 0.05 * want:
    raise SystemExit(...)
```

Exact for every code, the 5 % slack goes back to catching real faults, and all six codes
run. On a quantisation arm the error message should say so and quote `q`.

---

## §7 — Running the arms in parallel

**Every mapper job in this study is independent. They all go out at once.** No arm reads
another arm's output; the five boundaries are five separate chips (§3.2), and the reference
is a sixth. Nothing is sequential except the single evaluation at the end, which needs all
of them finished.

### 7.1 What fans out

The unit of work is **(arm × layer shape)**:

| arm | architecture the mapper sees | jobs |
|---|---|---|
| `reference` | published YAML, all `datawidth: 8`, no ERT bump | 1 per shape |
| `recon2` | `filter_glb` at `datawidth: 4`; ERT: `filter_glb.read` + `.leak` | 1 per shape |
| `recon4` | `filter_glb` at `datawidth: 4`; ERT: `weights_spad.write` + `.leak` | 1 per shape |

```
                    submitted together, all at once
      +---------------------+---------------------+
      |                     |                     |
  [ reference ]        [ recon2 ]            [ recon4 ]      <-- N arms
      | L1 L2 ...          | L1 L2 ...           | L1 L2 ...  <-- x shapes
      +---------------------+---------------------+
                            |
                    afterok: ALL of them
                            |
                            v
                  ONE evaluation job  ->  ReconSweep_optimiser.png
```

Job count = `(1 + number of ERT arms) x number of shapes`. On Eyeriss v1 with
`ECC_RECON_LAYER` set to one layer: **3 jobs**, running simultaneously, wall time = the
slowest single map. With `ECC_RECON_LAYER` empty: `3 x (shapes in resnet18)`.

`recon1`, `recon3` and `recon5` need **no** mapper job of their own — they are evaluated on
the reference arm's plan, which is already in the fan-out.

### 7.2 How to submit them

**Copy `hpc/map_by_shape.sh` to `hpc/map_ert_arms.sh` and change the fan-out dimension from
architectures to arms.** That script already does exactly the right thing: one `sbatch` per
unit of work, collect the job ids, then one dependent evaluation.

```bash
JOBS=()
for ARM in ${ARMS}; do                    # reference + one per ERT-injectable placement
    for L in "${LAYERS[@]}"; do
        jid=$(sbatch --parsable \
            --job-name="map-${ARM}-${L}" \
            --array="${I}-${I}" \
            --export=ALL,ECC_LAYERS="${L}",ECC_RECON_ERT_ARM="${ARM}",ECC_TASKFILE="${PWD}/${SNAP}" \
            hpc/map.sbatch)
        JOBS+=("${jid}")
    done
done
DEP=$(IFS=:; echo "${JOBS[*]}")
sbatch --dependency="afterok:${DEP}" ... hpc/run_all.sh --eval-only
```

Four things carried over from `map_by_shape.sh`, each for a reason already paid for:

* **The arm travels in `--export`, not in the task file.** `map.sbatch` resolves its
  `(architecture, model)` pair from the task file by array index, and every arm here is the
  *same* architecture in a different configuration. The task file cannot express that.
  `ECC_RECON_ERT_ARM` is a new knob (§8) that tells one mapper job which arm it is solving.
* **`--array=I-I` is required**, one index per submission. `map.sbatch` carries its own
  `#SBATCH --array` sized for a whole task file; without the override the extra tasks exit 2
  and the dependent eval sits on `DependencyNeverSatisfied` (measured 2026-09-09, jobs
  41474947-58).
* **Snapshot the task file.** `map_by_shape.sh` copies it to `hpc/.runtime/tasks.…$$.txt` so
  a later launcher cannot change what a queued job reads.
* **One dependent eval**, `--dependency=afterok:` all map jobs, with `ECC_LAYERS=` cleared so
  the evaluation sees the whole configured scope.

### 7.3 Why parallel is safe here

Each arm writes to its **own** mapper cache directory — different variant slug and different
fingerprint (RULE 4's first two defences) — so two arms can never write the same entry.
The existing `ShapeLock` (atomic `mkdir`) remains as belt-and-braces for two jobs on the same
shape of the same arm, which this launcher does not produce anyway.

---

## §8 — Configuration

    ECC_RECON_ARCHS=eyeriss_like_wglb
    ECC_RECON_MODEL=resnet18
    ECC_RECON_LAYER=layer3.0.conv1          # NEW; empty = the whole model
    ECC_RECON_CODE_N=63 ; ECC_RECON_K=30    # q = 4
    ECC_RECON_PACKING=aligned
    ECC_OPT_METRIC=edp                      # never `energy`
    ECC_MAPSPACE_CONSTRAIN=1 ECC_WEIGHT_FACTOR_RELAX=1
    ECC_MAPPER_ALGORITHM=linear_pruned ECC_MAPPER_TIMEOUT=100000000
    ECC_VICTORY=4000 ; ECC_MAPPER_THREADS=18
    ECC_PHASE=Post ; RECON_OPTIMIZER=True ; ECC_RECON_ERT_AWARE=1

Use the `ECC_RECON_*` spellings. §4 of `env.sh` hard-assigns `ECC_ARCHS`, `ECC_MODELS`,
`ECC_CODE_N` and `ECC_KS` from them with a bare `=`, so setting the general names on the
command line does nothing, and does it silently.

### 8.1 New knob — `ECC_RECON_ERT_AWARE`

`env.sh` §4, beside `RECON_OPTIMIZER`. Default `0`. Requires `RECON_OPTIMIZER=True` and
`ECC_PHASE=Post`; errors otherwise.

### 8.2 New knob — `ECC_RECON_LAYER`

`env.sh` §4, **directly below `ECC_RECON_MODEL`**, named so it cannot be mistaken for it:

```bash
: "${ECC_RECON_MODEL:=resnet18}"   # ONE model
# ONE LAYER of that model, or EMPTY for every layer of it. A single-layer run is
# what makes an ERT-aware sweep affordable: (1 + ERT arms) jobs instead of that
# many times the layer count.
: "${ECC_RECON_LAYER:=layer3.0.conv1}"
```

| `ECC_RECON_LAYER` | behaviour |
|---|---|
| set to a layer name | that layer only — it seeds `ECC_LAYERS` |
| **empty** | **every layer of `ECC_RECON_MODEL`** |

It seeds `ECC_LAYERS`, which the rest of the pipeline already understands. **One consequence
to handle:** §10 of `env.sh` currently blanks `ECC_STEM` whenever `ECC_LAYERS` is non-empty,
so the layer scope lands in the filename. That must not happen here — see §9.

`layer3.0.conv1` is `C128_M256_R3_S3_P14_Q14_ws2_hs2`, 294,912 weights. On the current YAML:
**168/168 PEs**, weight refetch **1.000**, **344,064 cycles**. Full PE occupancy means no
`PE !=` confound at baseline.

### 8.3 New knob — `ECC_RECON_ERT_ARM`

Per-job selector, set by the launcher in `--export`, not normally set by hand. Names which
arm one mapper job is solving: `reference`, or a placement key. Empty means `reference`.
It selects the `datawidth` levels and the ERT bump, and it feeds the variant slug and the
fingerprint (RULE 4).

### 8.4 Convergence

Run the gate on the **reference** arm at victory {2000, 4000, 10000} and state the margin
before quoting any bar. It does not need repeating per ERT arm.

---

## §9 — The figure

**Output path, fixed per model** (the `__<model>` suffix since 2026-09-11, when the
study ran on resnet18 AND mobilenet_v2 and the second eval overwrote the first's figure):

    results/figures/ReconSweep_optimiser__<model>.png

Bars: `baseline`, `embedded`, then every key in `ECC_RECON_PLACEMENTS[eyeriss_like_wglb]` —
`recon1 recon2 recon3 recon4 recon5`. **Mark which bars came from their own mapping**
(`recon2`, `recon4`); the rest are on the reference plan. Task 4 already prints a notice for
mixing two mapping regimes on one figure — print an equivalent one and name the arms.

`env.sh` §10 already appends `_optimiser` to `ECC_RECON_STEM` when `RECON_OPTIMIZER=True`,
so the stem is correct by default. The one change needed: §10 must set
`ECC_STEM="${ECC_RECON_STEM}_optimiser"` **even when `ECC_LAYERS` is non-empty**, instead of
blanking it. The layer scope then lands in the manifest and the figure title rather than the
filename. **Any existing file at that path is overwritten** — intended; the manifest beside
it records which configuration produced what is on disk.

### G_rec

Ignore it. The co-residency constraint of §15 has a workaround the user has already devised.
Do not add a mapspace constraint, do not re-open `feasibility()`, do not report it as a
blocker.

---

## §10 — Checklist before quoting a number

1. The convergence gate passes on the reference arm and the residual is stated.
2. Every narrow storage stop of every bar has **exactly one** narrowing owner, decided by
   measured `Word bits` from that bar's own stats (RULE 1).
3. `Scalar updates` for Weights at each patched level is **0** (RULE 2).
4. The access split and the leak split each reconcile to 1e-6 against the un-bumped energy,
   each computed from that bar's own counts and cycles (RULE 4).
5. Each ERT cache entry's stored ERT is read back and matches the bar requesting it, in
   level, action and delta (RULE 4.4.5).
6. The capacity check compares against `8/q` and states `q` (§6).
7. `recon2` is charged on `reads`/`read`, `recon4` on `fills`/`write`. Print the counter, the
   action and the engine count on every row.
8. The leak multiplier equals the **utilized** instance count of the site level on every
   shape, and is printed with the declared count beside it. Verified 2026-09-11 on 43 shapes
   of two models: Timeloop bills `leak × cycles × leaks_per_cycle`, and with each instance
   power-gated on its own (`Instances sharing power gating: 1`) `leaks_per_cycle` is the
   utilized count (`buffer.cpp` 2119–2127, 2376). RULE 3's `N_engines` at a storage site is
   therefore the utilized instances **per layer** (`engine_cycles = Σ utilized × cycles`), the
   same convention every arm's own scratchpad leakage already follows. The declared 168 held
   on `layer3.0.conv1` only because that layer fills the array.
9. PEs used per shape is **reported** against the reference plan (both counts, both cycle
   counts, Timeloop's EDP ratio), printed and carried into the manifest's `title_caveats`; it
   is not a refusal. On a full model the arm's own EDP-optimal plan may legitimately use fewer
   PEs where the doubled GLB room changes the DRAM chunking (2026-09-11, mobilenet_v2: 3 of 31
   shapes on `recon2`, each at lower energy and lower EDP than the reference plan), and the
   reference itself fills the array on fewer than two thirds of the shapes. The 168/168 rule
   this item used to state came from the one array-filling study layer; `PE!=` still
   suppresses Task 4's *capacity* verdict (FINDINGS 7.8), which is a different claim.
10. DRAM `datawidth` is 8 on every arm.
11. Per ERT arm: if that arm's loop nest is byte-identical to the reference, say the ERT
    changed nothing **for that arm**. Do not average the verdicts.

---

## §11 — Tests

`test_dilation.py` must pass unchanged except the capacity target, whose expectation moves
from `N/K` to `8/q` — update it deliberately and say so. `test_recon.py` **will** move:
`Packing`, `onchip_narrowing_audit()` and `load_recon_energy()` all change.

Add tests for:

* the `levels=` filter on `_set_weight_datawidth` (empty tuple is a no-op);
* `delta_pJ = E_w x block_size`, and the `leak` delta;
* the access split and the leak split;
* the measured narrowing audit — both-sites-live must FAIL, and zero-sites-on-a-narrow-stop
  must FAIL;
* the `updates == 0` assertion tripping when updates are non-zero;
* the capacity target at all six codes — K=36 and K=51 must now PASS;
* **the wrong-sibling guard**: two placements with identical YAML and different ERT must
  produce different fingerprints, and a read-back mismatch must stop the run;
* the ERT-injectable predicate on every registered design, so a new architecture cannot
  silently get zero arms or the wrong ones;
* `ECC_RECON_LAYER` empty selects every layer, set selects one.

---

## §12 — Implementation order

Each phase is checkable on its own. Do not start the next until the current one is verified.

| # | Phase | Done when |
|---|---|---|
| 1 | **Prove the ERT hook.** Map one shape with `filter_glb.read = 1000`; confirm `stats.txt` energy moves. Repeat with an absurd `leak`; confirm `Leakage energy (total)` moves. | Both move. ~60 s. If either does not, Accelergy is recomputing the table — fall back to generating ERT+ART with `accelergy` and adding both to `design_inputs()`. |
| 2 | **Per-level `datawidth`.** Add `levels=()` to `_set_weight_datawidth` and the config field. | Empty tuple reproduces today's YAML byte for byte; naming `filter_glb` leaves `weights_spad` at 8. |
| 3 | **Cache identity.** ERT in the variant slug and in the fingerprint; read-back assertion. | Two arms with identical YAML and different ERT land in different `fp-` directories, and the test for it passes. |
| 4 | **RULE 1 ownership.** Per-stage measured bypass in `storage_scale`; audit per bar per stage. | `recon3`/`recon5` still get their narrowing; `recon2`/`recon4` do not get it twice. |
| 5 | **RULE 3 split.** `load_recon_energy()` returns incremental and idle separately; `build_stacks()` charges idle per cycle. | Three sweep figures regenerate; `test_baseline_dram.py` updated. |
| 6 | **RULE 4 per-bar lookup.** `ert_aware_view()` beside `dilated_view()`; guards per arm. | `recon4`'s discounts come from `recon4`'s bill; per-arm nest verdicts printed. |
| 7 | **Capacity check** `N/K` -> `8/q`. | All six codes pass their own target. |
| 8 | **Parallel launcher.** `hpc/map_ert_arms.sh`; `ECC_RECON_ERT_ARM`; `ECC_RECON_LAYER`. | 3 jobs submitted simultaneously on one layer, one dependent eval. |
| 9 | **Run and evaluate.** Convergence gate, then the figure. | `results/figures/ReconSweep_optimiser.png`, §10 checklist all green. |

### How to run

    module load apptainer
    bash hpc/map_ert_arms.sh              # fans out; submits the dependent eval
    # or, once every arm is cached:
    bash hpc/tl.sh bash run.sh recon --eval

Read `CLAUDE.md`, then `FINDINGS.md` §7.10-7.12, then the last entries of `progress.txt`.
Anything that invokes the mapper runs inside an allocation, never on a login node.

---

## Appendix A — Files this plan edits

| File | Change |
|---|---|
| [eccenergy/recon.py](eccenergy/recon.py) | ERT-injectable predicate; `Packing.storage_scale()` per-stage bypass on measured `Word bits`; `onchip_narrowing_audit()` measured and per-bar; capacity target `N/K` -> `8/q` |
| [eccenergy/experiments/recon.py](eccenergy/experiments/recon.py) | new `ert_aware_view()` beside `dilated_view()`; per-bar plan lookup replacing the single `p_*` set in `evaluate()`; guards per ERT arm; ERT read-back assertion; `updates == 0` assertion; access and leak attribution splits; per-arm reporting |
| [eccenergy/archs.py](eccenergy/archs.py) | `levels=()` filter on `_set_weight_datawidth`; ERT slug in `effective_variant()`; ERT delta in `arch_fingerprint()` |
| [eccenergy/ecc.py](eccenergy/ecc.py) | `load_recon_energy()` returns incremental and idle separately; `build_stacks()` recon column charges idle per cycle and guards the fourth narrowing site |
| [eccenergy/config.py](eccenergy/config.py) | `recon_ert_aware`, `recon_layer`, `recon_ert_arm`, `weight_datawidth_levels` fields; validation pairing `ECC_RECON_ERT_AWARE` with `RECON_OPTIMIZER` / `ECC_PHASE` |
| [eccenergy/timeloop.py](eccenergy/timeloop.py) | patched ERT written beside the patched arch and handed to `call_mapper` / `design_inputs()` |
| [env.sh](env.sh) | §4: `ECC_RECON_ERT_AWARE`, `ECC_RECON_LAYER` (directly below `ECC_RECON_MODEL`), `ECC_RECON_ERT_ARM`, the two DC tables, retire `ECC_RECON_INCLUDE_IDLE`; §10: keep `ECC_STEM` set when `ECC_LAYERS` is non-empty |
| **`hpc/map_ert_arms.sh`** *(new)* | copy of [hpc/map_by_shape.sh](hpc/map_by_shape.sh) fanning out over arms x shapes, one dependent eval |
| [eccenergy/tests/test_recon.py](eccenergy/tests/test_recon.py) | expectations move; new tests per §11 |
| [eccenergy/tests/test_dilation.py](eccenergy/tests/test_dilation.py) | capacity-target expectation `N/K` -> `8/q` |
| [eccenergy/tests/test_baseline_dram.py](eccenergy/tests/test_baseline_dram.py) | `load_recon_energy()` expectations move |

Read-only references: [archs/eyeriss_like_wglb/arch_paper.yaml](archs/eyeriss_like_wglb/arch_paper.yaml),
[archs/_shared/noc.yaml](archs/_shared/noc.yaml), [data/dc/BCH_N63_results.json](data/dc/BCH_N63_results.json),
[hpc/map.sbatch](hpc/map.sbatch).

Regenerated, not edited: `results/figures/ReconSweep_optimiser.png` with its manifest and
table; and — because RULE 3 changes the reconstruction term — `BCHsweep`, `ModelSweep` and
`ArchitectureSweep`.

---

## Appendix B — Other designs: later, not now

**Out of scope for this prompt.** Recorded so the work is not re-discovered. Finish Eyeriss
v1 first.

`eyeriss_v2_like_wglb` does not run today. Its weight path gained a `weight_glb` stop, but
its list of encoder positions was never updated, so three positions jump over it and no
position ever reaches it. `validate_placement_space()` refuses the design before any energy
is computed. Verified 2026-09-10:

```
eyeriss_like_wglb        ok=True      <- Eyeriss v1, in scope, passes
eyeriss_v2_like_wglb     ok=False
eyeriss_v2_like          ok=True      <- no GLB, nothing to jump over

reducible_stages_no_placement_reaches: ['weight_glb']
violations: "recon2 reduces ['dram', 'inter_cluster_mesh'] but the reducible stages
             between the ECC engine and that boundary are ['dram', 'weight_glb']:
             the reduced form cannot skip a stage on its way down the path"
```

**The fix, when its turn comes:** add `_EYERISS_V2_WGLB_PLACEMENTS` with five positions — one
per stop, no gaps — the same shape Eyeriss v1 already has:

```
recon1   before the GLB              narrow: DRAM
recon2   after the GLB               narrow: DRAM, GLB
recon3   after the mesh              narrow: DRAM, GLB, mesh
recon4   at the scratchpad input     narrow: DRAM, GLB, mesh, local wires
recon5   at the scratchpad output    narrow: all five
```

Then set `ECC_RECON_PLACEMENTS[eyeriss_v2_like_wglb]="recon1 recon2 recon3 recon4 recon5"`.
Note the renumbering: with a GLB in the path, `recon2` means "after the GLB", not "after the
mesh", so the existing four-key line would draw a different set of positions than it used to.

Because §3.3's ERT-injectable set is derived rather than hardcoded, **no other change is
needed** — that design picks up its two ERT arms automatically, as do `eyeriss_v2_like` (1
arm) and `simple_weight_stationary` (2 arms).
