# Run: two jobs, one layer, no depth scaling — embedded vs recon

*Self-contained. Assumes `prompt_4.md`'s DRAM fixes are already applied.
Background on the constrained-mapping method is in `prompt_3.md`; you do not
need to read it to run this.*

## What to run

`weights_spad` has been set to `depth: 16` in
`archs/eyeriss_like_wglb/arch_paper.yaml`. **Use the YAML exactly as it is —
do not scale any depth.** Two mapper jobs on one layer: the 8-bit arm
(baseline + embedded share one mapping) and the 4-bit recon arm.

    cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
    module load apptainer

    export ECC_MAPSPACE_CONSTRAIN=1 ECC_WEIGHT_FACTOR_RELAX=1 \
           ECC_MAPPER_ALGORITHM=linear_pruned ECC_MAPPER_TIMEOUT=100000000 \
           ECC_DEPTH_SWEEP_SCALES="1.0" ECC_DEPTH_SWEEP_GATE_SCALES="1.0"

    bash hpc/map_depth_sweep.sh --no-per-level     # 2 sweep + 2 gate = 4 jobs

Everything else comes from `env.sh` and must not be overridden:
`eyeriss_like_wglb` (Eyeriss v1), `resnet18`, `ECC_LAYERS=layer3.0.conv1`,
BCH(63,30) → recon datawidth **4**, `ECC_OPT_METRIC=edp`,
`ECC_RECON_PACKING=aligned`, `ECC_MAPPER_THREADS=18`, `ECC_QOS=rewetz-b`.

Jobs take **~1 minute each**; the wait is queue time. Watch with
`bash hpc/map_depth_sweep.sh --progress`, or `squeue -u $USER`.

## Then read it — IN THIS ORDER

**Everything must go through `hpc/tl.sh`.** Running `python3 -m ...` bare
silently uses `config.py` defaults (objective `energy`, victory 500) instead of
`env.sh`'s, and the answer is invalid.

    # 1. THE GATE. If it does not pass, nothing below may be quoted.
    bash hpc/tl.sh python3 -m eccenergy.experiments.dilation --gate --scales 1.0

    # 2. The tables + CSVs
    bash hpc/tl.sh python3 -m eccenergy.experiments.dilation --levels --scales 1.0 \
         --csv results/tables/EyerissV1_spad16.csv

The gate maps the embedded arm at victory 2000 / 4000 / 10000 and prints the
residual between budgets. With the constrained mapspace it should read
**0.00 %** — the search is exhaustive, so each arm sits at its true optimum.
A non-zero residual larger than the recon-vs-embedded margin means the answer
is search noise; stop and say so.

## The table to produce

Join the two CSVs `--levels` writes (`...spad16.csv` and
`...spad16__dataspace.csv`) into one row per weight level:

| ×depth | depth | emb held / room (fill) | rec held / room (fill) | refetch e→r | level pJ e→r | whole datapath µJ e→r | saving |

    python3 - <<'EOF'
    import csv
    L={(r["memory"],r["arm"]):r for r in csv.DictReader(open("results/tables/EyerissV1_spad16.csv"))}
    D={r["dataspace"]:r for r in csv.DictReader(open("results/tables/EyerissV1_spad16__dataspace.csv"))}
    te,tr=float(D["Outputs"]["total_uJ_emb"]),float(D["Outputs"]["total_uJ_rec"])
    for mem in ("filter_glb","weights_spad"):
        e,r=L[(mem,"embedded")],L[(mem,"recon")]
        print(f"{mem:<14} depth {e['declared_depth']:>4} | "
              f"emb {int(e['weights_held']):,}/{int(e['room']):,} ({float(e['fill_pct']):.1f}%) | "
              f"rec {int(r['weights_held']):,}/{int(r['room']):,} ({float(r['fill_pct']):.1f}%) | "
              f"refetch {float(e['refetch']):.3f}->{float(r['refetch']):.3f} | "
              f"level pJ {float(e['level_pJ'])/1e6:.2f}M->{float(r['level_pJ'])/1e6:.2f}M | "
              f"total {te:.2f}->{tr:.2f} uJ | saving {tr-te:.2f} ({100*(tr-te)/te:.1f}%)")
    EOF

Also print the **per-dataspace block** that `--levels` emits (Weights / Inputs /
Outputs, with the `x-reduce` C-chunk counts). It is not optional — see below.

## Three things that will look wrong and are not

1. **Weight `refetch` will be 1.000 on BOTH arms.** That is its FLOOR, not a
   bug: `C` is the only DRAM-level loop and `C` indexes Weights, so every
   weight is fetched exactly once and a smaller weight buffer cannot make it
   worse. Do not report this as "no effect".
2. **Weight `level pJ` will halve exactly** (8 b → 4 b). That is the flat
   packing discount and it is usually only ~3 % of the total saving.
3. **The real saving is in the `Outputs` row** — recon's extra weight capacity
   lets it use fewer DRAM-level `C` chunks, and each chunk removed is a
   partial-sum round-trip to DRAM that never happens. If the `x-reduce` counts
   are EQUAL on both arms, there is no capacity effect at this depth and the
   margin is packing only; say that plainly.

## Caveats that travel with the numbers

This is the **constrained + relaxed dataflow** on a reconfigured
psum/weight geometry — **not** the chip JSSC 2017 describes, so never label it
"Eyeriss v1" without that qualifier. Both arms get the identical constraint, so
the comparison is fair, but the margin is conditional on that mapspace family.

Record the result in `FINDINGS.md` (empirical claims live only there) and add a
line to `progress.txt`. Run `bash hpc/tl.sh python3 -m eccenergy.tests.test_dilation`
after any code change.

## Then add the column the table is missing: recon's DRAM K/N saving

*This is an addition to the table above, not a change to the run. It is
arithmetic on the records `--levels` has already read, so it cannot move a
mapping and the gate still governs it.*

`dilation.py --levels` reads **raw Timeloop output**, so its DRAM rows are
identical for embedded and recon — 18,874,368 pJ on both, verified, and that is
exactly `294,912 reads x 64 pJ`. Recon's K/N reduction is applied later, by
`recon.py`'s `dram` stage in `run.sh recon --eval`, and never reaches this
table.

Consequence: **every margin in FINDINGS 7.10 UNDERSTATES recon.** The -31.3 %
at x0.25 is the mapper-side capacity effect ALONE — it excludes the DRAM fetch
saving, which is the study's original Task 3 mechanism and the larger term at
`ECC_DRAM_PJ_PER_BIT=40`.

**Two things to get right before adding a number**, both of which the raw table
gets wrong for this purpose:

1. **The price.** Every energy in these tables is at Accelergy's ERT — 64 pJ per
   8-bit DRAM scalar, i.e. **8 pJ/bit** — while every number in FINDINGS is
   quoted at `ECC_DRAM_PJ_PER_BIT=40`. `total_pj_at()` substitutes the MAC
   denominator and nothing else, so the DRAM in these totals is **5x too
   cheap**. Rescale the DRAM contribution the same way, or the saving column is
   understated by 5x against everything it will be read beside.
2. **The accessor.** `energy_by_dataspace()` sums **all levels**, so it is
   whole-datapath-per-dataspace, not DRAM. Use the DRAM row —
   `dram_weight_reads x dram_pj_per_read` (the `dram_weight_pj` property) — or
   on-chip weight energy gets billed as if it were a DRAM fetch.

**The columns**, added to `dataspace_table()` and to `DATASPACE_CSV_COLUMNS`
(`k_over_n` is already computed at the top of that function and currently
unused):

    dram_weight_pJ_at_price = dram_weight_reads x dram_pj_per_read
                              x (ECC_DRAM_PJ_PER_BIT / ERT pJ per bit)
    dram_kn_saving_pJ       = that x (1 - K/N)              # RECON ROWS ONLY
    total_mapping_only_uJ   = total_uJ at the same DRAM price
    total_with_dram_uJ      = total_mapping_only_uJ - dram_kn_saving_pJ

Print **both** totals side by side — "mapping only" and "mapping + DRAM K/N" —
so the two mechanisms stay separable and are never silently summed, and put the
pJ/bit each column was priced at in the header. At BCH(63,30) on
`layer3.0.conv1` the arithmetic is fixed and checkable before you run anything:

| term | value |
|---|---|
| DRAM weight energy, ERT 8 pJ/bit | 18.874 uJ |
| DRAM weight energy at 40 pJ/bit | **94.372 uJ** |
| `1 - K/N` | 0.52381 |
| recon's DRAM K/N saving | **49.433 uJ** |

**The baseline belongs in this table too, and costs nothing to add.** Under
`prompt_4.md`'s model baseline shares embedded's mapping exactly and differs
only in price, so its column is derived from the embedded row with no second
mapper job:

    total_baseline_uJ = total_mapping_only_uJ
                        + dram_all_pJ_at_price x (70/40 - 1)

Three arms, one mapping, one table: baseline's DRAM at 70 pJ/bit, embedded's at
40, recon's at 40 x K/N.

---

# THE WIDTH RULE — settled 2026-09-11. Do not re-litigate it.

*Added because the same objection was raised and re-raised on every BCH run, and
each round of it scheduled mapper jobs that did not need to exist.*

## The rule

**THE 8-BIT ARM IS ONE ARM. It is declared at `width: 96` and it is mapped
ONCE, for the whole BCH sweep.** Baseline and embedded are that arm. They do not
get a width per code, they do not get re-mapped per code, and **no job may be
submitted that maps the 8-bit arm at any width other than 96.**

Each reconstruction code then takes its OWN width from prompt_2.md's WIDTH
TABLE, and that width only ever has to divide **its own `q`** — never 8:

| arm | q | declared width | weights/word | effective capacity |
|---|---:|---:|---:|---:|
| **baseline + embedded** | **8** | **96** | **12** | **1.0000x** |
| BCH(63,57) recon | 7 | **98** | 14 | 1.1667x |
| BCH(63,45) recon | 6 | 96 | 16 | 1.3333x |
| BCH(63,39) recon | 5 | **95** | 19 | 1.5833x |
| BCH(63,30) recon | 4 | 96 | 24 | 2.0000x |

98, 96 and 95 are all within **3 %** of each other. That closeness is the whole
point of the table and it is why the arms stay comparable; prompt_2 chose the
five widths for exactly this reason.

## What this costs in jobs — the reason the rule is written down

A four-code sweep is **1 + 4 = 5 mapper jobs**, not 8:

    1 job   the 8-bit arm at width 96          <- baseline AND embedded, ALL codes
    1 job   recon q=7 at width 98
    1 job   recon q=6 at width 96
    1 job   recon q=5 at width 95
    1 job   recon q=4 at width 96

Pairing a fresh 8-bit arm to each code costs one wasted map per code and, worse,
gives the sweep **four different baseline bars** when the model says there is
one. If a run has produced more than one embedded number across codes, the width
rule was broken.

## The one thing to check before believing "the mapper aborted"

`timeloop-mapper` asserts `width % (word_bits * block_size) == 0`
(`buffer.cpp:302`) and there is no floor path, so it ABORTS. But the check is
**per arm**:

    embedded arm:  96 % 8 == 0    OK      <- the only divisibility 96 must satisfy
    recon q=7:     98 % 7 == 0    OK
    recon q=5:     95 % 5 == 0    OK

`98 % 8 = 2` and `95 % 8 = 7` are **irrelevant**, because no arm ever declares
`width: 98, datawidth: 8`. An abort attributed to them is an abort that was
never going to happen.

## What DOES have to be said out loud when the rule is used

The two arms of a pair then declare **different silicon** (96 vs 98, or 96 vs
95), and CACTI prices depth x width, so Accelergy charges the two arrays
differently per access. That is a real difference and it is why
`archs.assert_pair_geometry()` refuses it: that guard implements the STRICTER
rule adopted 2026-09-10 (both arms identical, differing only in `datawidth`),
which is not the rule prompt_2's table was built on. **Using the width table
means running with that guard relaxed for the width field, deliberately and on
the record.** Its cost is bounded and its direction is known:

* q=7 at 98 vs 96: the recon array is **2.08 % dearer** per access than the
  reference's. Recon is penalised -> the saving is UNDERSTATED. Safe direction.
* q=5 at 95 vs 96: the recon array is **1.04 % cheaper**. Recon is flattered by
  about a percent. State it; do not correct it.
* q=6 and q=4 are at 96 on both arms -> literally the same silicon, no caveat.

prompt_2's own residual column already carries this: all five residuals are
negative, i.e. every arm understates reconstruction, and that is the direction
to keep.

## And the geometry the table actually needs

The table renormalises `depth` to hold each level's declared total bits, so it
needs an array deep enough that integer depth is not lossy. At Eyeriss v1's
**published `weights_spad` 224 x 16 b = 3,584 bits** it lands where prompt_2
says: **depth 37 at width 96**, 444 weights/PE against the paper's 448. On the
**depth-16** spad this prompt ran on (256 bits) width 96 renormalises to
**depth 3**, which is +12.5 % in bits and barely a reuse level. **Run the BCH
sweep on the published spad, not on the depth-16 one.**

## `ECC_WEIGHT_WIDTH=auto` — WITHDRAWN 2026-09-12, there is only one scheme

This section used to present `eccenergy/code_widths.py`'s `lcm(q, 8)` widths
(56 / 24 / 24 / 40 / 40 / published) as a conservative ALTERNATIVE to prompt_2's
WIDTH TABLE, to be picked per study. **It was not an alternative. It was a
misreading, and it is gone** — along with the `ECC_WEIGHT_WIDTH` knob itself.

It rested on "both arms of a pair share ONE declared width", which is false.
`timeloop-mapper` asserts `width % (word_bits * block_size) == 0` per level, per
mapper run, and **one mapper run maps one arm**. The arms are never mapped on
one another's silicon, so prompt_2's 98 (q=7) and 95 (q=5) are correct exactly
as written and the fact that they do not divide 8 is not a property of anything.

The table above already named the price, and it was paid: under lcm the **8-bit
reference arm moves between codes** (35 / 33 / 30 weights per PE on the depth-16
spad), which is what produced BCH(63,39)'s spurious −37.69 % (FINDINGS 2.4b, now
withdrawn). prompt_2's table has the 8-bit arm at width 96 at every code — ONE
arm, mapped ONCE — which is the whole reason its five widths sit within 3 % of
each other.

What is in force is prompt_2.md's WIDTH TABLE, applied automatically per level
and per arm by `archs._set_weight_geometry()`; `python3 -m eccenergy.code_widths`
prints it. `archs.assert_pair_geometry()` checks the LEVEL SET and the DEPTH and
does **not** check width or datawidth.
