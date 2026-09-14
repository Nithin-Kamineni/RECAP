"""ONE FIGURE ROW PER METRIC -- what `ECC_METRICS` names, as stacks a bar can be drawn from.

EnvReorganisation 3.1 and 2.2: rows = `ECC_METRICS`, columns = `ECC_SWEEP`,
bars = `ECC_APPROACHES`. This module owns the ROWS, and only the rows: it turns
one `Raw` record plus the energy stacks that were already built from it into
one DataFrame per metric, shaped exactly as `build_stacks()` shapes the energy
one -- rows = stack segments, columns = bars -- so `report/stacked.draw_panel`
draws all four without knowing there is more than one kind (CLAUDE.md: adding a
parameter to `draw_panel`, never a second routine).

THREE OF THE FOUR METRICS MOVE NO NUMBER. `energy` IS `build_stacks()`,
delegated to unchanged. `latency` is `Raw.latency` -- the roofline
`ECC_LATENCY_MODEL=1` already applies -- and, for the reconstruction arm, the
re-timing `Raw.latency`'s own docstring names: `latency_post.model_cycles(raw,
cfg, weight_scale=K/N)`. `edp` is those two multiplied. Nothing here re-solves
a mapping, re-times a plan a second time, or invents a term; every number is
read back off a plan already in the cache. That is what lets phase 5 be gated
on the three of them being byte-identical to what phase 4 left.

`area` IS NEW (EnvReorganisation 6.3, question 9.7 answered yes), and it is
built LAST because section 7 says so. It has TWO halves and they come from two
different measurements:

    the ACCELERATOR   Accelergy's ART, `timeloop-mapper.ART.yaml`, written into
                      every mapper cache entry. Per-level area x the instance
                      count in the table name, classified into the same
                      categories the energy row stacks by, so the two rows of
                      one figure are cut the same way.
    the ENGINE        Design Compiler, `archs/_shared/recon_area.yaml`, scraped
                      from `data/dc/report_snapshots/<cfg>/active/area.rpt` by
                      `tools/scrape_dc_area.py`. Recon bars only, one engine.

AND THE TWO ARE NOT THE SAME KIND OF NUMBER, which the figure has to say. The
ART is Accelergy's analytical model of a 45 nm array; the engine is a
synthesized pre-layout cell area from a different flow and a different library
(gscl45nm). They are added because the question "what does reconstruction cost
in silicon" has no other answer, and every area result carries
`area_provenance` naming both sources so nobody reads the sum as one
measurement. The scale is worth knowing before reading a bar: on
`eyeriss_like_wglb` the accelerator is 3.685 mm2 and one BCH(63,39) engine is
0.00228 mm2 -- 0.06 %. The engine segment is a sliver, and that is the result,
not a drawing bug.
"""
from __future__ import annotations

import pathlib

import pandas as pd

from ..physics.recon_dc import load_recon_area
from ..settings import guards
from ..toolchain import ert as ert_mod
from ..toolchain import latency_post
from ..toolchain.stats import classify
from .energy import phys_cats

#: The stack segment a single-valued metric draws. `latency` and `edp` are not
#: sums of parts -- a plan's run length does not decompose into DRAM time plus
#: buffer time, it is a MAX over the levels (`latency_post.roofline`) -- so
#: they are honestly one segment, and inventing a stacked breakdown of a max
#: would be a picture of an addition that never happens.
LATENCY_SEGMENT = "Latency"
EDP_SEGMENT = "Energy x delay"
#: The area segment the ART cannot supply.
ENGINE_SEGMENT = "Recon engine"


def metric_label(cfg, metric, unit):
    return {"energy": f"Inference energy ({unit})",
            "edp": f"Energy x delay ({unit})",
            "latency": f"Inference latency ({unit})",
            "area": f"Silicon area ({unit})"}[metric]

#: The AXIS SCALING for each metric lives in `report/style.unit_for_metric` --
#: `study/` is L5 and `report/` is L6, so a metric's unit is picked where the
#: axis is drawn rather than by an upward import from here.

# ------------------------------------------------------------------- latency
def arm_cycles(cfg, raw, arm, code_k=None):
    """This arm's own run length, in cycles. Returns (cycles, provenance).

    WHO GETS WHICH NUMBER, and why it is not a new timing model:

    `baseline` / `embedded`   the plan's own re-timing at weight_scale 1.0 --
        `Raw.latency`, which `study.energy.apply_latency_model()` has filled
        since prompt_7 Phase A. Both arms move the SAME weight bits across the
        DRAM interface (the baseline pays a dearer per-bit PRICE, not extra
        traffic -- CLAUDE.md), so they cannot differ in time.
    `recon`   `latency_post.model_cycles(raw, cfg, weight_scale=K/N)`, which is
        verbatim what `Raw.latency`'s docstring says a bar driving K/N of the
        weight bits off the die gets. Not a second model: the same roofline,
        handed this arm's own off-chip demand.

    WITH `ECC_LATENCY_MODEL=0` there is no re-timing to read and every arm gets
    Timeloop's own `Raw.cycles`, so all three bars are equal. That is an ABSENT
    TERM, not a measured null (`toolchain/latency_post.py`'s opening note), and
    the provenance string says so on every bar rather than letting a flat row
    read as "reconstruction buys no time".
    """
    if not getattr(cfg, "latency_model", False):
        if not raw.cycles:
            raise guards.refusal("latency-no-cycles",
                "ECC_METRICS names `latency` but this raw record carries no "
                "cycle count and ECC_LATENCY_MODEL=0, so there is nothing to "
                "read.\n"
                "  -> ECC_LATENCY_MODEL=1 re-times the plan, or re-gather the "
                "record from the mapper cache")
        return float(raw.cycles), ("Timeloop's own cycles (ECC_LATENCY_MODEL=0 "
                                   "-- no re-timing, so every arm is equal by "
                                   "construction: an ABSENT term, not a "
                                   "measured null)")
    if raw.latency is None:
        raise guards.refusal("latency-not-modelled",
            "ECC_METRICS names `latency` and ECC_LATENCY_MODEL=1, but this "
            "raw record has not been re-timed.\n"
            "  -> study.energy.apply_latency_model() fills Raw.latency; "
            "re-gather the record (load_raw does it unless ECC_REPLOT_ONLY=1)")
    if arm != "recon":
        return float(raw.latency["cycles"]), (
            f"roofline at weight_scale 1.0: {raw.latency['cycles']:,} cycles")
    k = code_k or cfg.code_k
    scale = k / cfg.code_n
    out = latency_post.model_cycles(raw, cfg, weight_scale=scale)
    if out is None:
        raise guards.refusal("latency-not-modelled",
            f"the reconstruction arm's re-timing at weight_scale {scale:.6g} "
            f"returned nothing -- this record has no per-layer physical detail "
            f"to re-time.\n  -> re-gather it from the mapper cache")
    return float(out["cycles"]), (
        f"roofline at weight_scale K/N = {scale:.6g}: {out['cycles']:,} cycles")


def latency_stacks(cfg, raw, bars, code_k=None, arm_of=None):
    """One row, one value per bar: seconds. Returns (DataFrame, provenance)."""
    period = float(latency_post.cycle_seconds(cfg))
    vals, prov = {}, {}
    for a in bars:
        cyc, why = arm_cycles(cfg, raw, (arm_of or {}).get(a, a), code_k)
        vals[a] = cyc * period
        prov[a] = why
    return pd.DataFrame({a: {LATENCY_SEGMENT: vals[a]} for a in bars}), prov


def edp_stacks(cfg, raw, energy_df, bars, code_k=None, arm_of=None):
    """One row, one value per bar: pJ x seconds.

    EACH ARM'S OWN ENERGY x ITS OWN DELAY. Multiplying every arm's energy by
    the reference's delay would report the energy saving twice over and call
    the product a second metric; pairing each bar with its own re-timing is the
    only reading of EDP that is not a restatement of the energy row.
    """
    period = float(latency_post.cycle_seconds(cfg))
    cols, prov = {}, {}
    for a in bars:
        cyc, why = arm_cycles(cfg, raw, (arm_of or {}).get(a, a), code_k)
        e = float(energy_df[a].sum())
        cols[a] = {EDP_SEGMENT: e * cyc * period}
        prov[a] = f"{e:.6g} pJ x {cyc * period:.6g} s ({why})"
    return pd.DataFrame(cols), prov


# ---------------------------------------------------------------------- area
def chip_area_by_category(cfg, cache_dir):
    """`{category: um2}` for one chip's ART, plus the provenance.

    The ART's levels are classified by `toolchain.stats.classify` -- the SAME
    routine that puts a level's ENERGY in a category -- so the area row and the
    energy row of one figure are cut along the same lines and a segment means
    the same component in both. Passing the instance count makes `instances`
    mode work exactly as it does for energy: one instance is a global buffer,
    many is local storage, whatever the level is called.
    """
    doc, path = ert_mod.chip_art(cache_dir)
    if doc is None:
        raise guards.refusal("area-chip-never-mapped",
            f"ECC_METRICS names `area` but no ART was found under\n"
            f"    {cache_dir}\n"
            f"  Accelergy writes `timeloop-mapper.ART.yaml` into every cache "
            f"entry, so a chip with none has never been mapped at this "
            f"fingerprint.\n"
            f"  -> map it (`bash hpc/run_all.sh --map-only`), or drop `area` "
            f"from ECC_METRICS. This is NOT charged as zero: an accelerator "
            f"with no silicon in it is not a result.")
    levels = ert_mod.art_level_areas(doc)
    out = {}
    for level, row in levels.items():
        cat = classify(level, row["instances"], cfg.classify_mode)
        out[cat] = out.get(cat, 0.0) + row["total_um2"]
    total = sum(out.values())
    return out, (f"Accelergy ART {path.name} under {pathlib.Path(cache_dir).name}: "
                 f"{len(levels)} levels, {total:,.1f} um2 total")


def area_stacks(cfg, bars, cache_dirs, *, code_k=None, engines=1,
                engine_bars=None):
    """Rows = area categories + the engine; columns = bars. um2.

    `cache_dirs` is {bar: the mapper cache directory of the chip that bar was
    BILLED FROM} -- the same `billed_from` rule every energy bar already obeys
    (CLAUDE.md), so an area bar and an energy bar for one arm are the same
    chip. Two bars billed from one plan legitimately share an area.

    THE ENGINE IS ADDED TO RECON BARS ONLY, once per engine, from the Design
    Compiler scrape. `engines` is the caller's count -- `build_stacks()`'s
    abstract recon arm reconstructs at the chip ingress where ONE engine sits,
    and a placement's count comes from `physics/granularity.py`. It is never a
    number in the YAML.
    """
    cats = list(phys_cats(cfg))
    cols, prov = {}, {}
    engine_um2 = None
    for a in bars:
        by_cat, why = chip_area_by_category(cfg, cache_dirs[a])
        col = {c: float(by_cat.get(c, 0.0)) for c in cats}
        col[ENGINE_SEGMENT] = 0.0
        wants = (a.startswith("recon") if engine_bars is None
                 else a in engine_bars)
        if wants:
            if engine_um2 is None:
                engine_um2, engine_why = load_recon_area(cfg, code_k)
            col[ENGINE_SEGMENT] = engine_um2 * engines
            why = (f"{why}; + {engines} x engine {engine_um2:.3f} um2 "
                   f"[{engine_why}]")
        cols[a] = col
        prov[a] = why
    df = pd.DataFrame(cols).reindex(cats + [ENGINE_SEGMENT], fill_value=0.0)
    return df, prov


# ------------------------------------------------------------------ the rows
def metric_stacks(cfg, metric, raw, energy_df, bars, *, code_k=None,
                  cache_dirs=None, engines=1, engine_bars=None, arm_of=None):
    """The DataFrame for ONE figure row. Returns (DataFrame, provenance).

    `energy_df` is what `build_stacks()` already returned for this group -- it
    is passed in rather than rebuilt so that the energy row and the EDP row can
    never disagree about an arm's energy.

    `arm_of` maps a DataFrame COLUMN onto the ECC ARM it represents, for the
    figures whose column name is not the arm name. The three sweeps put the arm
    in the column (`baseline`/`embedded`/`recon`) and pass nothing; the
    placement figure has one column called `energy` per group and passes the
    arm the group IS, so `arm_cycles` re-times a reconstruction boundary at
    K/N and a reference bar at 1.0 on both figures by the same rule.
    """
    if metric == "energy":
        return energy_df, {a: "build_stacks()" for a in bars}
    if metric == "latency":
        return latency_stacks(cfg, raw, bars, code_k, arm_of)
    if metric == "edp":
        return edp_stacks(cfg, raw, energy_df, bars, code_k, arm_of)
    if metric == "area":
        if not cache_dirs:
            raise guards.refusal("area-chip-never-mapped",
                "ECC_METRICS names `area` but the caller supplied no mapper "
                "cache directory for any bar, so there is no ART to read.")
        return area_stacks(cfg, bars, cache_dirs, code_k=code_k,
                           engines=engines, engine_bars=engine_bars)
    raise guards.refusal("unknown-metric",
        f"no builder for metric {metric!r}")
