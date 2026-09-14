"""ERT-aware evaluation: ONE PLAN PER BAR (prompt_6 RULE 4).

`ert_aware_view()` is what `ECC_RECON_ERT_AWARE=1` runs. Each bar is billed from
the plan of the chip it actually is -- its own mapper arm where that arm is
mapped, otherwise the plan `arch.arms.plan_assignment()` assigns and FLAGS -- and
the toll Timeloop billed inside the level is MOVED into `Reconstruction`, so the
mapper and the report price one engine and RULE 5.3's "a split, not an addition"
still holds.

`arm_plan_state()` is the record of that decision for every bar: which arm, which
fingerprint, whether the geometry matches, and the PE counts and cycles of both
plans. A PE count that differs between an ERT arm's own plan and the reference is
REPORTED per shape (both counts, cycles, Timeloop EDP ratio; manifest
`title_caveats`), never refused. `ECC_RECON_ERT_AWARE=1` with no arm mapped at
all IS refused -- that is Task 3 under a heading that says otherwise.

`ert_checks()` re-reads each entry's stored ERT before a number is used: two arms
with byte-identical YAML must never share a directory, and a cache entry must
prove which toll it was solved at.

ProjectRestructure phase 3 cut this out of `experiments/recon.py`.
"""
from __future__ import annotations

import dataclasses
import math
import pathlib
import yaml

from ..arch import fingerprint as fingerprint_mod
from ..arch import arms as arms_mod
from ..arch import placements
from . import placement_eval
from ..toolchain import weight_stats
from ..toolchain import ert
from ..toolchain import inputs
from ..toolchain import stats
from ..physics import baseline_dram
from .common import Session

from .dilated_view import _capacity_of
from .placement_notes import stats_paths_for
from ..settings import guards


# --------------------------------------------------------- prompt_6: the ERT
#  arms -- one plan per BAR (RULE 4)
# ----------------------------------------------------------------------------
EXPERIMENT_ERT = "task4_reconstruction_aware_mapping_ert"

ERT_NOTE = (
    "RECONSTRUCTION-AWARE MAPPING WITH THE ENCODER IN THE OBJECTIVE (prompt_6), "
    "ONE PLAN PER BOUNDARY (prompt_7 B1/B2). The arms are the DISTINCT CHIPS the "
    "mapper sees -- (datawidth q on the storage levels the boundary narrows) x "
    "(its ERT bump, where its site is a storage level whose counter names one "
    "real ERT action and whose bump is not constant across the mapspace) x (its "
    "declared per-dataspace bandwidth scale) -- derived from the placement "
    "records, never listed. An ERT-injectable arm was RE-MAPPED with its encoder "
    "toll in Timeloop's energy table (E_w x block_size on the site's access "
    "action, idle x (1 - g) on `leak`), so the mapper solved the loop nest "
    "knowing what reconstruction costs at that boundary; such a bar is billed "
    "from ITS OWN plan (bill, discounts, cycles, DRAM reads) and the toll "
    "Timeloop billed inside the level is MOVED into `Reconstruction`, never "
    "added a second time. A boundary whose own chip is not yet mapped is billed "
    "from a NAMED plan that narrows the SAME storage levels, with the lender's "
    "toll already moved out; a boundary for which no such plan exists on disk is "
    "billed from the reference plan and flagged `geometry_matches: false`. EVERY "
    "BAR'S RECORD NAMES THE PLAN IT WAS BILLED FROM in `billed_from`, and the "
    "figure marks it.")


@dataclasses.dataclass
class ErtArmView:
    """One ERT arm's OWN plan, and everything needed to bill that bar from it.

    prompt_6 RULE 4.4.3: a bar is bill minus discounts plus the encoder, so the
    bill, the per-level energies, the raw record, the cycle count and the ERT
    delta must all describe the same run -- this one. Built by
    `ert_aware_view()`; `evaluate()` keeps one per ERT arm and a bar with no
    arm points at the reference set.
    """
    placement: object
    cfg: object                 # the arm's Config (recon_ert_arm = its key)
    raw: object                 # the arm's Raw record (toll still inside)
    wpath: object               # its weight path, with the toll MOVED OUT
    base_series: object         # its bill by category, toll moved out
    base_w: object              # the weight share of that
    cycles: float               # its run length, the idle term's denominator
    fingerprint: str
    variant: str
    bump: dict                  # archs.ert_bump()
    split: dict                 # ert_split(), reconciled stats-side and evaluator-side
    nest_identical: bool        # byte-identical loop nest to the reference on EVERY shape
    per_shape_nest_identical: dict
    read_back: dict             # ert.read_back_ert() per shape
    checks: dict                # per-shape guard rows (updates, leak multiplier, PEs, ...)
    capacity: dict              # the narrowed level's Effective size, arm / reference = 8/q
    reconciles: bool
    recon_check: dict
    stats_paths: dict


def ert_split(bump, st, cycles, billed_vectors=None):
    """prompt_6 5.3 -- the two amounts an ERT arm's Timeloop run carries because
    of the toll, from THAT bar's own counts and cycles, on the action that was
    bumped (RULE 2), never a level total or a blended average:

        access   billed vector accesses x (E_w x block_size)
        leak     idle x engine_cycles  (= sum over shapes of idle x UTILIZED
                 instances x cycles -- what Timeloop bills, buffer.cpp)

    THE ACCESS TERM IS COUNTED IN WHOLE WORDS. Timeloop bills
    `ceil(scalar_accesses / block_size)` vector accesses -- a partially filled
    last word costs a whole word -- so `billed_vectors` is that ceiling summed
    over shapes (x utilized instances x repeats), measured from the arm's own
    stats by the caller. It used to be `scalar_accesses x E_w`, i.e. the same
    quantity with the ceiling dropped, which is right ONLY when the block size
    divides every count. It did while `filter_glb` was 64 b / 4 b = block 16
    and every weight count was a multiple of 16; prompt_2's WIDTH TABLE makes
    it 380 b / 5 b = block 76 (and 96, 56, 64 at the other codes), 8192 / 76 =
    107.79, and the partial word appears on EVERY shape. The 1e-6
    reconciliation below then failed by 825 pJ in 6.89 uJ (2026-09-12).
    `billed_vectors=None` falls back to the scalar form for a caller that has
    no per-shape counts.

    The ACCESS amount sits INSIDE the level's per-dataspace `Energy (total)`,
    i.e. inside the bill's category, and is MOVED into `Reconstruction`. The
    LEAK amount is not in the bill at all: Timeloop prints leakage as a
    separate per-level line outside the per-dataspace energies, and the raw
    record aggregates only those (a few tens of pJ of leakage on the
    reference, never a category). So the idle term is verified against the
    stats' leakage delta and CHARGED by the evaluator, not subtracted.
    `st` is the site stage's StageStats from the arm's own weight path.
    """
    engine_cycles = float(st.engine_cycles or 0.0)
    engines = (engine_cycles / float(cycles)) if cycles else 0.0
    # NO BUMP, NO TOLL. Since prompt_7 B2 the arms to map are the distinct
    # CHIPS, and two of them (R1, R3 on Eyeriss v1) declare no ERT row at all:
    # `ert_injectable()` refuses them, so Timeloop billed nothing extra inside
    # any level and there is nothing to move out of the bill. The split is
    # ZERO, stated -- not skipped, because the caller still reconciles it
    # against the stats and a silently-absent split would reconcile trivially.
    if bump is None:
        return {"level": None, "action": None, "counter": None,
                "access_toll_pJ": 0.0, "leak_toll_pJ": 0.0,
                "engines": engines,
                "engines_declared": float(st.declared_instances or 0.0),
                "engine_cycles": engine_cycles, "billed_vectors": 0.0,
                "no_bump": True,
                "why": ("this boundary declares no ERT-injectable encoder row "
                        "(recon.ert_injectable), so the mapper was given no toll "
                        "and there is none inside the level's billed energy. Its "
                        "reconstruction energy is charged wholly by the "
                        "evaluator, on the same two denominators as every other "
                        "bar (prompt_6 RULE 3).")}
    count = float(st.counter(bump["counter"]))
    if billed_vectors is None:
        access = count * float(bump["e_w_pj"])
    else:
        access = float(billed_vectors) * float(bump["access_delta_pj"])
    leak = float(bump["leak_delta_pj"]) * engine_cycles
    return {"level": bump["level"], "action": bump["action"], "counter": bump["counter"],
            "scalar_accesses": count, "e_w_pj": float(bump["e_w_pj"]),
            "billed_vector_accesses": (None if billed_vectors is None
                                       else float(billed_vectors)),
            "billed_weights": (None if billed_vectors is None
                               else float(billed_vectors) * float(bump["block_size"])),
            "access_delta_pj": float(bump["access_delta_pj"]),
            "access_toll_pJ": access, "engines": engines, "cycles": float(cycles),
            "engine_cycles": engine_cycles,
            "engines_declared": float(st.declared_instances or 0.0),
            "engines_utilized_max": float(st.instances or 0.0),
            "idle_pj_per_cycle_per_engine": float(bump["leak_delta_pj"]),
            "leak_toll_pJ": leak, "total_toll_pJ": access + leak,
            "rule": ("stats-side access toll = vector accesses x (E_w x block_size) = "
                     "scalar accesses x E_w; stats-side leak = idle x UTILIZED "
                     "instances x cycles per shape (Timeloop power-gates unused "
                     "instances: buffer.cpp FinalizeBufferEnergy); both moved out "
                     "of the level's category into Reconstruction so the stack has "
                     "Task 3's shape")}


def _close(a, b, rel=1e-6, abs_tol=0.0):
    """|a - b| within `rel` of the larger magnitude, OR within `abs_tol` --
    the latter for quantities read off Timeloop's stats, which print to 0.01
    pJ."""
    return abs(float(a) - float(b)) <= max(rel * max(abs(float(a)), abs(float(b)), 1e-9),
                                           float(abs_tol))


def arm_plan_state(cfg, ses, arch, model, arm, ref_shapes):
    """Is this MAPPER ARM's own mapping on disk, at ITS OWN fingerprint?
    `(state, detail)`, state one of `ready` | `cold` | `partial`.

    A CHEAP PRE-FILTER, not a second acceptance rule. The fingerprint is in
    the path (`<slug>/fp-<hash>/<shape>/`), and the ERT bump and the declared
    datawidth are in the fingerprint, so a `timeloop-mapper.stats.txt` under
    this arm's own directory can only be this arm's own mapping. Everything
    that decides whether an entry may be USED -- the sidecar, the stored ERT
    read back against the arm's own base table, the per-shape guards -- stays
    in `ert_aware_view()`, which is called only for a `ready` arm and still
    refuses rather than falling back.

    Why a probe exists at all (prompt_7 B1/B4): since the arm list became the
    six DISTINCT CHIPS rather than the two ERT-injectable boundaries, an arm
    with no mapping is the NORMAL state until Phase C runs, and the figure has
    to report a number for its bars. It gets one from a NAMED, geometry-
    compatible plan (`recon.plan_assignment`) -- never silently.

    `partial` is the dangerous middle and is NOT a borrow: a half-mapped arm
    would mix its own plan on some shapes with a borrowed one on the rest, and
    one bar would be two chips. The caller refuses on it.
    """
    acfg = cfg.with_(recon_ert_arm=arm.key, rerun_optimiser=False, from_cache=True)
    variant = fingerprint_mod.effective_variant(arch, acfg)
    fp = fingerprint_mod.arch_fingerprint(arch, acfg)
    cache = pathlib.Path(ses.results.mapper_cache(arch, variant, fp, create=False))
    have = sorted(s for s in ref_shapes
                  if (cache / s / "timeloop-mapper.stats.txt").exists())
    detail = {"arm": arm.key, "cache": str(cache), "fingerprint": fp,
              "variant": variant, "shapes_wanted": len(ref_shapes),
              "shapes_solved": len(have),
              "missing": sorted(set(ref_shapes) - set(have))}
    if len(have) == len(ref_shapes) and ref_shapes:
        return "ready", detail
    return ("cold" if not have else "partial"), detail


def ert_aware_view(cfg, ses, arch, model, base_cats, ref_raw, ref_paths, placement):
    """Load, GUARD and split ONE ERT arm's own mapping (prompt_6 RULE 4).

    Stops the run -- never falls back to the reference plan -- when the arm's
    cache is missing or short of shapes, when a stored ERT is not this bar's
    (RULE 4.4.5 read-back against the reference entry's un-bumped table), when
    `Scalar updates` for Weights at the patched level is not 0 (RULE 2), when
    the leak multiplier is not the UTILIZED instance count of the site level
    (RULE 3 -- Timeloop bills leak x utilized x cycles, power-gating each
    unused instance; verified 2026-09-11 on 43 shapes of two models), when
    DRAM is not 8-bit (RULE 1), when the narrowed level did not deliver 8/q
    (prompt_6 6), or when the attribution split does not reconcile to 1e-6
    (5.3).

    A PE count that differs from the reference is REPORTED, not refused
    (2026-09-11): on a full model the arm's own EDP-optimal plan may use fewer
    PEs where the doubled GLB room changes the DRAM chunking (mobilenet_v2:
    3 of 31 shapes on recon2, each at lower energy AND lower EDP than the
    reference plan), and the reference itself fills the array on fewer than
    two thirds of the shapes. Every such shape is recorded with both PE
    counts, both cycle counts and both EDPs, printed, and carried into the
    manifest's `title_caveats`. The equality rule of prompt_6 10.9 came from
    one layer that fills the array; `PE!=` still suppresses Task 4's
    CAPACITY verdict, which is a different claim.

    The loop-nest verdict is recorded PER ARM and never collapsed (4.4.4).
    """
    acfg = cfg.with_(recon_ert_arm=placement.key)
    bump = fingerprint_mod.ert_bump(arch, acfg)
    # THE ARMS ARE THE DISTINCT CHIPS, AND TWO OF THEM HAVE NO ERT ROW (prompt_7
    # B2): R1 and R3 on Eyeriss v1 are their own chips -- R1 by its declared
    # off-chip bandwidth scale, R3 by the network stage it adds -- but
    # `ert_injectable()` refuses both, so `ert_bump()` is None and there is no
    # toll anywhere in their bills. Everything below that is ABOUT THE TOLL is
    # guarded on `site`; everything about the PLAN (loop nest, PE count,
    # cycles, DRAM width, delivered capacity) runs for every arm.
    site = bump["level"] if bump else None
    variant = fingerprint_mod.effective_variant(arch, acfg)
    fp = fingerprint_mod.arch_fingerprint(arch, acfg)
    print(f"\n  ---- prompt_6: {placement.key}'s OWN mapping -- {ert.describe_bump(bump)} ----")
    print(f"       datawidth {acfg.weight_datawidth} on {'+'.join(acfg.weight_datawidth_levels)}; "
          f"cache {variant}  fp {fp}")
    ases = Session(acfg).setup(need_mapper=True)
    ases.collect_arch(arch)
    raw_a = (ases.raws.get(arch) or {}).get(model)
    if raw_a is None:
        raise guards.refusal("ert-arm-cache-cold",
            f"ECC_RECON_ERT_AWARE=1 needs {placement.key}'s OWN mapping for {arch}/{model} "
            f"({ert.describe_bump(bump)}), and it is not in the cache.\n"
            f"  expected: {ases.results.mapper_cache(arch, variant, fp, create=False)}\n"
            f"  -> map it:  bash hpc/run_all.sh --map-only   (one sbatch job per chip x shape)\n"
            f"  Refusing rather than falling back to the reference plan: that fallback IS "
            f"Task 3, and this heading says otherwise.")
    paths_a, _mapper_a = stats_paths_for(acfg, ases, arch, model)
    if set(paths_a) != set(ref_paths):
        raise guards.refusal("ert-arm-shapes-differ",
            f"{placement.key}'s own mapping and the reference are mapped on DIFFERENT "
            f"layer shapes for {arch}/{model}:\n"
            f"  only in the reference: {', '.join(sorted(set(ref_paths) - set(paths_a))) or 'none'}\n"
            f"  only in {placement.key}: {', '.join(sorted(set(paths_a) - set(ref_paths))) or 'none'}\n"
            f"  -> finish the missing maps before evaluating")

    # RULE 4.4.5, defence 3: every entry's stored ERT, read back against THIS
    # ARM'S OWN un-bumped table -- `<arm cache>/fp-*/_ert/base.ERT.yaml`, which
    # `ErtTables` wrote from this arm's own patched arch (Accelergy is
    # deterministic, FINDINGS 3.5, so one base serves every shape).
    #
    # NOT the reference entry's table. It used to be, on the stated grounds
    # that "the datawidth does not enter Accelergy" -- true, but WIDTH does,
    # and since prompt_2's WIDTH TABLE went in (2026-09-12) the arms declare
    # DIFFERENT widths: recon2's `filter_glb` is 380 b against the reference's
    # 384 b, which CACTI prices at 13.4921 pJ/read against 13.591. Comparing
    # the two stopped every ERT-aware eval with
    #   read_back_ert: ... recorded base filter_glb.read = 13.4921
    #                      but the un-bumped table says 13.591
    # -- the guard was right that two tables disagreed and wrong about which
    # table the arm should be held to. Its OWN base is the one that makes
    # "stored == base + delta, every other row untouched" mean anything.
    read_back = {}
    for shape, path in paths_a.items():
        shape_dir = pathlib.Path(path).parent
        own_base = shape_dir.parent / inputs.ERT_DIR / "base.ERT.yaml"
        base_prices = (ert.ert_prices(yaml.safe_load(own_base.read_text()))
                       if own_base.exists() else None)
        # prompt_7 C1.6: and the MAC price the entry was mapped at. It is in
        # the fingerprint, so a mismatch means a hand-copied entry -- which is
        # exactly the case RULE 4.4.5's read-back exists for.
        read_back[shape] = ert.read_back_ert(
            shape_dir, bump, base_prices=base_prices,
            mac_pj=getattr(cfg, "mac_pj_override", None))

    # ---- per-shape guards, off the arm's OWN stats -------------------------
    counts_by_shape = {}
    for layer in ases.models[model]:
        counts_by_shape[layer.shape_name] = counts_by_shape.get(layer.shape_name, 0) \
            + float(getattr(layer, "count", 1) or 1)
    nests, rows, problems = {}, [], []
    billed_vectors = 0.0
    stats_access = stats_leak = 0.0
    for shape, path in paths_a.items():
        a = pathlib.Path(ref_paths[shape]).parent / "timeloop-mapper.map.txt"
        b = pathlib.Path(path).parent / "timeloop-mapper.map.txt"
        nests[shape] = a.is_file() and b.is_file() and a.read_text() == b.read_text()
        lv, sm = stats.parse_levels(path)
        rlv, rsm = stats.parse_levels(ref_paths[shape])
        cyc_a, cyc_r = sm["cycles"] or 0, rsm["cycles"] or 0
        # THE SITE LEVEL exists only for an arm with an ERT row. Without one
        # there is no bumped action to audit, and the guards below that read
        # `w`/`n_inst` are guards ON THE TOLL.
        L = lv.get(site) if site else None
        R = rlv.get(site) if site else None
        if site and (L is None or R is None or "Weights" not in L["ds"]):
            raise guards.refusal("level-carries-no-weights",
                f"{placement.key}/{shape}: level {site!r} carries no "
                f"Weights in the stats")
        w = L["ds"]["Weights"] if L else {}
        updates = float(w.get("updates") or 0.0)
        other = sorted(d for d in (L["ds"] if L else {}) if d != "Weights")
        n_inst = float(w.get("utilized_instances")
                       or (L["instances"] if L else None) or 1)
        n_inst_r = float((R["ds"].get("Weights") or {}).get("utilized_instances")
                         or R["instances"] or 1) if R else 1.0
        # the arm's OWN stored table, and the un-bumped prices under it
        base_p, base_leak = {}, None
        if site:
            prices = ert.ert_prices(
                yaml.safe_load((pathlib.Path(path).parent / inputs.ERT_NAME).read_text()))
            base_p = {a_: prices[(site, a_)] for a_ in ("read", "write", "update")
                      if (site, a_) in prices}
            base_p[bump["action"]] = base_p[bump["action"]] - bump["access_delta_pj"]
            base_leak = prices.get((site, "leak"))
            if base_leak is not None:
                base_leak = base_leak - bump["leak_delta_pj"]
        # RULE 3, ARM-LOCALLY: Timeloop bills `leak x UTILIZED instances x
        # cycles` (buffer.cpp FinalizeBufferEnergy), and this arm's stored leak
        # price is its own base plus the bump. So the toll is the level's
        # printed leakage minus `base_arm x utilized_a x cycles_a`, with no
        # reference anywhere in it.
        #
        # It USED to be the reference's leakage rescaled by the cycle and
        # instance ratios, on the stated grounds that the base is the same per
        # instance-cycle in both arms (Accelergy is deterministic, FINDINGS
        # 3.5). Determinism was never the issue -- the arms stopped being the
        # same memory. Since prompt_2's WIDTH TABLE went in (2026-09-12)
        # recon2's `filter_glb` is 380 b against the reference's 384 b, and
        # CACTI leaks 7.32555e-05 against 7.39799e-05 pJ/instance/cycle. That
        # 7.244e-07 gap is 5.19e-05 of the intended bump, so every shape
        # reported a multiplier of 0.99995 instead of 1 and the run stopped.
        # Deriving the base from the arm's own table removes the reference
        # from the identity being checked, which is what RULE 3 was always
        # about.
        leak_mult = None
        leak_delta_pJ = None
        if site and L.get("leakage_pJ") is not None and base_leak is not None and cyc_a:
            leak_delta_pJ = L["leakage_pJ"] - base_leak * n_inst * cyc_a
            leak_mult = leak_delta_pJ / (bump["leak_delta_pj"] * cyc_a)
        mac_a = next((k for k in lv if "Compute" in lv[k]["ds"]), None)
        mac_r = next((k for k in rlv if "Compute" in rlv[k]["ds"]), None)
        pes_a = lv[mac_a]["utilized_instances"] if mac_a else None
        pes_r = rlv[mac_r]["utilized_instances"] if mac_r else None
        dram_wb = (lv.get("DRAM") or {}).get("word_bits")
        # The multiplier is judged in pJ, against the printed leakage total's
        # precision: Timeloop prints `Leakage energy (total)` to 0.01 pJ, so on
        # a small depthwise layer (42 PEs x 60k cycles) the quotient lands at
        # 41.99996 and a bare 1e-6 relative test refuses a multiplier that IS
        # the utilized count (mobilenet_v2 G576, 2026-09-11).
        # NO TOLL, NOTHING TO VERIFY: an arm with no ERT row is `ok` here by
        # construction, and saying so beats a None that reads as a failure.
        leak_ok = (not site) or (leak_delta_pJ is not None and _close(
            leak_delta_pJ, bump["leak_delta_pj"] * n_inst * cyc_a,
            abs_tol=0.01 * (1.0 + (cyc_a / cyc_r if cyc_r else 1.0))))
        bs = float((L["block_size"] if L else 1) or 1)
        # WHOLE WORDS, as Timeloop bills them: `ceil(scalar / block_size)`
        # vector accesses per instance (buffer.cpp), not `scalar / block_size`.
        # Dropping the ceiling under-counts the level's own energy by up to one
        # word per dataspace, which cancelled while the block size divided every
        # count (64 b / 4 b = 16) and stopped cancelling under prompt_2's WIDTH
        # TABLE (380 b / 5 b = 76): 0.195 % of the printed energy on EVERY
        # shape, on the reference arm as much as on this one.
        rep = counts_by_shape.get(shape, 1.0)
        if site:
            vec = {a_: math.ceil(float(w.get(a_ if a_ != "write" else "fills") or 0.0) / bs)
                   for a_ in ("read", "write", "update")}
            vec["read"] = math.ceil(float(w.get("reads") or 0.0) / bs)
            vec["update"] = math.ceil(updates / bs)
            at_base = ((vec["read"] * base_p.get("read", 0.0)
                        + vec["write"] * base_p.get("write", 0.0)
                        + vec["update"] * base_p.get("update", 0.0)) * n_inst)
            stats_access += (float(w["energy_pJ"]) - at_base) * rep
            billed_vectors += vec[bump["action"]] * n_inst * rep
            if leak_delta_pJ is not None:
                stats_leak += leak_delta_pJ * rep
        edp_a = float(sm["energy_uJ"] or 0.0) * float(cyc_a or 0)
        edp_r = float(rsm["energy_uJ"] or 0.0) * float(cyc_r or 0)
        row = dict(shape=shape, nest_identical=nests[shape], weights_updates=updates,
                   other_dataspaces_on_level=other, leak_multiplier=leak_mult,
                   leak_multiplier_expected=n_inst, leak_multiplier_ok=leak_ok,
                   leak_delta_pJ=leak_delta_pJ,
                   leak_multiplier_rule=("Timeloop bills leak x UTILIZED instances x "
                                         "cycles (buffer.cpp FinalizeBufferEnergy, "
                                         "power-gated per instance)"),
                   declared_instances=(L["instances"] if L else None),
                   utilized_instances=n_inst,
                   utilized_instances_reference=n_inst_r,
                   pes_used=pes_a, pes_used_reference=pes_r,
                   pes_differ=(pes_a != pes_r), dram_word_bits=dram_wb,
                   site_level=site,
                   level_word_bits=(L["word_bits"] if L else None),
                   block_size=(L["block_size"] if L else None),
                   cycles=cyc_a, cycles_reference=cyc_r,
                   timeloop_edp_uJ_cycles=edp_a, timeloop_edp_uJ_cycles_reference=edp_r,
                   timeloop_edp_ratio_arm_over_reference=(edp_a / edp_r) if edp_r else None,
                   vector_access_energy_source=(L["source"] if L else None),
                   energy_uJ=sm["energy_uJ"], energy_uJ_reference=rsm["energy_uJ"])
        rows.append(row)
        if site and updates:
            problems.append(f"{shape}: Scalar updates for Weights at {site} = "
                            f"{updates:g}, not 0 -- put the toll on `update` too or stop (RULE 2)")
        if site and other:
            problems.append(f"{shape}: {site} also holds {other}; the toll on its "
                            f"`{bump['action']}` would be billed to them as well")
        if site and not leak_ok:
            problems.append(f"{shape}: leak multiplier {leak_mult} != utilized instances "
                            f"{n_inst:g} of {site} (declared {L['instances']}; "
                            f"RULE 3: Timeloop bills leak x UTILIZED instances x cycles, "
                            f"buffer.cpp FinalizeBufferEnergy -- verify the multiplier)")
        if dram_wb != cfg.weight_bits:
            problems.append(f"{shape}: DRAM Word bits {dram_wb} != {cfg.weight_bits} "
                            f"(RULE 1: DRAM is never narrowed)")
        if site and L["source"] != "ERT":
            problems.append(f"{shape}: {site} vector access energy source is "
                            f"{L['source']!r}, not ERT -- the supplied table was not billed")
    if problems:
        raise guards.refusal("ert-arm-fails-guards",
            f"{placement.key}'s own mapping fails its guards:\n  "
            + "\n  ".join(problems))
    # ---- PE utilisation: REPORTED per shape, never refused (see docstring) ----
    differ = [r_ for r_ in rows if r_["pes_differ"]]
    pe_report = {
        "shapes_total": len(rows), "shapes_differ": len(differ),
        "per_shape": {r_["shape"]: {"pes_arm": r_["pes_used"], "pes_reference": r_["pes_used_reference"],
                                    "cycles_arm": r_["cycles"], "cycles_reference": r_["cycles_reference"],
                                    "timeloop_edp_ratio_arm_over_reference":
                                        r_["timeloop_edp_ratio_arm_over_reference"],
                                    "nest_identical": r_["nest_identical"]} for r_ in differ},
        "rule": ("PEs used may differ between the arm's OWN EDP-optimal plan and the "
                 "reference plan; the difference is the arm's configuration (datawidth q "
                 "on its narrowed levels) acting through the mapper, reported here and "
                 "in the manifest's title_caveats rather than refused (2026-09-11; "
                 "prompt_6 10.9's 168/168 rule came from one array-filling layer)")}
    if differ:
        print(f"  [note] {placement.key}: PEs used differ from the reference on "
              f"{len(differ)}/{len(rows)} shape(s) -- reported, not refused:")
        for r_ in differ:
            print(f"         {r_['shape']}: PEs {r_['pes_used']} vs {r_['pes_used_reference']}, "
                  f"cycles x{r_['cycles'] / r_['cycles_reference']:.3f}, Timeloop EDP "
                  f"x{r_['timeloop_edp_ratio_arm_over_reference']:.3f} of the reference plan")

    # ---- prompt_6 6: quantisation delivers 8/q, never N/K ---------------------
    # THE NARROWED LEVELS COME FROM THE ARM, not from the bump: R3 narrows
    # `filter_glb` and declares no ERT row at all, and R1 narrows NOTHING on
    # chip -- its whole difference from the reference is a declared off-chip
    # bandwidth scale (prompt_7 6.4). `config` resolves both from the arm spec.
    narrow = tuple(acfg.weight_datawidth_levels or ())
    q = acfg.weight_datawidth
    want = (cfg.weight_bits / q) if q else 1.0
    cap_ref, cap_arm = _capacity_of(ref_paths, narrow), _capacity_of(paths_a, narrow)
    got = (cap_arm / cap_ref) if cap_ref else 0.0
    if narrow and cap_ref and abs(got - want) > 0.05 * want:
        raise guards.refusal("ert-arm-capacity-wrong",
            f"{placement.key}: {narrow[0]} reports Effective size {cap_arm:,} against the "
            f"reference's {cap_ref:,} -- x{got:.4f}, but a quantisation arm at q = {q} "
            f"delivers exactly {cfg.weight_bits}/q = {want:.4f} (prompt_6 6). The "
            f"`datawidth` edit missed the level, or the plan is not this arm's.")
    capacity = {"level": narrow[0] if narrow else None,
                "reference_weights_per_instance": cap_ref,
                "arm_weights_per_instance": cap_arm, "delivered_factor": got,
                "wanted_factor": want, "q": q,
                "rule": ("a quantisation arm delivers weight_bits/q, never N/K "
                         "(prompt_6 6)" if narrow else
                         "this boundary narrows NOTHING on chip, so there is no "
                         "capacity factor to deliver: its whole difference from "
                         "the reference is the declared off-chip bandwidth scale "
                         "(prompt_7 6.4), which changes the throttling check and "
                         "not one picojoule of energy")}

    # ---- the weight path, reconciled with the arm's record BEFORE the split ---
    wpath_a = weight_stats.weight_path(acfg, arch, model, ases.models[model], paths_a)
    reconciles, recon_check = placement_eval.cross_check(
        acfg, arch, wpath_a, raw_a.base_w.reindex(base_cats, fill_value=0.0))

    # ---- 5.3: the split -- MOVE the toll, do not add it --------------------
    stage_def = next(s_ for s_ in placements.stages_for(arch, acfg)
                     if s_.key == placement.site_stage)
    st = wpath_a.stages[placement.site_stage]
    split = ert_split(bump, st, wpath_a.cycles, billed_vectors=billed_vectors)
    # An arm with no ERT row moved nothing, and the stats side must agree: the
    # zero split is reconciled like any other rather than skipped, because a
    # skipped check reconciles trivially and proves nothing.
    for name, stats_side, evaluator in (("access", stats_access, split["access_toll_pJ"]),
                                        ("leak", stats_leak, split["leak_toll_pJ"])):
        if not _close(stats_side, evaluator):
            raise guards.refusal("ert-split-does-not-reconcile",
                f"{placement.key}: the {name} split does not reconcile -- stats-side "
                f"{stats_side:.6f} pJ (printed level energy minus the same counts at the "
                f"un-bumped prices) vs evaluator {evaluator:.6f} pJ (prompt_6 5.3, 1e-6)")
    split.update(stats_side_access_pJ=stats_access, stats_side_leak_pJ=stats_leak,
                 reconciled_rel_tol=1e-6)
    cat = placement_eval._category_of(stage_def, acfg)
    base_series = raw_a.base.reindex(base_cats, fill_value=0.0).copy()
    base_w = raw_a.base_w.reindex(base_cats, fill_value=0.0).copy()
    # Only the ACCESS toll is inside the level's billed energy; the leak toll
    # is a separate Timeloop line the raw record does not aggregate, so it is
    # charged by the evaluator and must not be subtracted here.
    base_series[cat] -= split["access_toll_pJ"]
    base_w[cat] -= split["access_toll_pJ"]
    st.energy_pJ -= split["access_toll_pJ"]
    st.switch_pJ -= split["access_toll_pJ"]
    if st.energy_pJ <= 0:
        raise guards.refusal("ert-split-left-residue",
            f"{placement.key}: moving the access toll out of {cat} left "
            f"{placement.site_stage} at {st.energy_pJ:.3f} pJ; the split is wrong")
    split["moved_out_of_category"] = cat
    split["moved_pJ"] = split["access_toll_pJ"]
    split["leak_in_bill"] = False
    split["leak_note"] = ("Timeloop prints leakage per level OUTSIDE the per-dataspace "
                          "energies and the raw record aggregates only those, so the "
                          "idle term is not in the bill: verified against the stats' "
                          "leakage delta (stats_side_leak_pJ) and charged by the evaluator")

    identical = bool(nests) and all(nests.values())
    print(f"       loop nest {'IDENTICAL to' if identical else 'DIFFERENT from'} the reference "
          f"on {sum(nests.values())}/{len(nests)} shape(s); cycles {wpath_a.cycles:,.0f}; "
          + (f"NO ERT row on this boundary, so nothing was moved out of {cat}; "
             f"its reconstruction energy is charged wholly by the evaluator "
             f"({split['engines']:.2f} engines cycle-weighted, of "
             f"{split['engines_declared']:g} declared)" if split.get("no_bump") else
             f"moved {split['access_toll_pJ'] / 1e6:.4f} uJ (access) out of {cat} into "
             f"Reconstruction; leak {split['leak_toll_pJ'] / 1e6:.3f} uJ "
             f"({split['engines']:.2f} engines cycle-weighted, of "
             f"{split['engines_declared']:g} declared) verified "
             f"against the stats' leakage and charged by the evaluator"))
    return ErtArmView(
        placement=placement, cfg=acfg, raw=raw_a, wpath=wpath_a,
        base_series=base_series, base_w=base_w, cycles=float(wpath_a.cycles),
        fingerprint=fp, variant=variant, bump=bump, split=split,
        nest_identical=identical, per_shape_nest_identical=nests,
        read_back=read_back, checks={"per_shape": rows, "pe_utilization": pe_report},
        capacity=capacity,
        reconciles=reconciles, recon_check=recon_check, stats_paths=paths_a)


#: How far past the latency ceiling a bar must be before it is CALLED past it,
#: in percentage points.
#:
#: IT IS NOT A FUDGE, AND 1e-9 WAS NOT "EXACT". The ceiling is
#: `weight share x (1 - K/N)`, a ratio of real-valued item counts; a bar's gain
#: is a ratio of CYCLE COUNTS, and cycles are integers that come from a
#: per-layer `ceil(compute / throttling)`. The two therefore agree only to
#: within the rounding of 21 ceilings, and on a fully off-chip-bound run --
#: where the relief converts 1:1 into time and the bar lands exactly ON the
#: ceiling -- the difference is real but meaningless: measured 9.3e-09 on
#: resnet18 at 120 MB/s, which a 1e-9 tolerance flagged as a bar "past" it.
#:
#: 1e-4 pp is four orders of magnitude below the effect this check exists to
#: catch (a bar past the ceiling on its own plan's ACTIVATION traffic, measured
#: at 1.52 pp on mobilenet_v2, 2026-09-12) and four above the rounding.
LATENCY_CEILING_TOL_PP = 1e-4


def ert_checks(builder, cfg, arch, raw, views, results, base_components,
               emb_components, e_parity, pricing, newly_mapped=0, billing=None,
               arm_states=None, lat=None):
    """prompt_6 10 -- the checklist, recorded on the result file, one per arm."""
    frac = cfg.code_k / cfg.code_n
    space_ok, space = arms_mod.validate_placement_space(arch, cfg)
    builder.check("placement_space_covers_the_whole_weight_path", space_ok, space)
    stages = placements.stages_for(arch, cfg)
    leak = arms_mod.ert_leak_delta_pj(cfg)
    builder.check(
        "ert_arms_are_derived_from_the_placement_records", True,
        {"arms_with_their_own_plan": sorted(views),
         "per_cycle_leak_row_pJ": leak,
         "clock_gating_pct": float(getattr(cfg, "recon_clock_gating_pct", 0.0)),
         "per_placement": {p.key: dict(zip(("injectable", "why"),
                                            arms_mod.ert_injectable(p, stages, leak)))
                           for p in placements.placements_for(arch, cfg)},
         "rule": ("prompt_6 3.3 / prompt_7 6.3: never `if key == ...`. Condition 3 "
                  "excludes a bump only when BOTH its rows are constant across the "
                  "mapspace -- the innermost level's reads AND a zero per-cycle leak "
                  "row. The leak row is idle x (1 - g) and Timeloop bills it as "
                  "leak x utilized instances x cycles, both the mapper's choice.")})
    # prompt_7 B2 gate 1 and 2: the arms are the DISTINCT CHIPS, and no two of
    # them share a cache directory. Recorded on every ERT-aware result so the
    # arm list a figure was drawn from is auditable after the fact.
    arms = arms_mod.mapper_arms(arch, cfg)
    slugs = [a.slug_part for a in arms]
    builder.check(
        "mapper_arms_are_distinct_chips_with_distinct_cache_slugs",
        len(set(slugs)) == len(slugs),
        {"n_arms": len(arms),
         "arms": [{"key": a.key, "slug": a.slug_part,
                   "datawidth_q_on": list(a.narrow_levels),
                   "bandwidth_scale_on": [list(x) for x in a.bw_scale],
                   "ert": list(a.ert) if a.ert else None,
                   "members": list(a.members)} for a in arms],
         "cache_state_per_arm": arm_states or {},
         "rule": ("prompt_7 6.4: an arm is (datawidth: q) x (ERT bump) x (per-dataspace "
                  "bandwidth scale). Two arms with byte-identical YAML sharing one "
                  "directory is the failure prompt_6 RULE 4.4.5 exists to prevent.")})
    # prompt_7 B1 gate 4: which plan each bar was billed from, and whether that
    # plan's storage geometry is the bar's own.
    builder.check(
        "no_bar_is_billed_from_a_plan_of_a_different_geometry_when_one_exists",
        all(b["geometry_matches"] or not b["candidates"]
            for b in (billing or {}).values()),
        {"per_bar": billing or {},
         "rule": ("a plan is VALID for a bar only when it narrows exactly the levels "
                  "the bar narrows: `datawidth: q` changes the words the loop nest "
                  "moves, and no post-processing can re-tile a loop nest. A bar with "
                  "NO valid plan on disk is reported `geometry_matches: false` with "
                  "`candidates: []` -- prompt_7 6.2b, a chip that has never been "
                  "mapped -- never silently.")})
    # prompt_7 A2 x B1: once the bars stop sharing one plan, a bar can pass the
    # latency ceiling -- which is computed on the REFERENCE plan's traffic and
    # bounds the WEIGHT term only. That is legitimate ONLY if that bar's plan
    # moves different ACTIVATION traffic off chip. A bar at x1.000 on both
    # weights and activations that still beat the ceiling would be the roofline
    # inventing time.
    if lat and lat.get("rows"):
        ref_c = float(lat["reference_cycles"] or 0.0)
        cap = float(lat["ceiling"]["latency_ceiling_pct"])
        offenders = []
        for r in lat["rows"]:
            gain = ((ref_c - r["cycles"]) / ref_c * 100.0) if ref_c else 0.0
            if gain <= cap + LATENCY_CEILING_TOL_PP:
                continue
            wt = r.get("weight_items_vs_reference")
            act = r.get("activation_items_vs_reference")
            offenders.append({"bar": r["bar"], "plan": r.get("plan"),
                              "gain_pct": gain, "ceiling_pct": cap,
                              "weight_items_vs_reference": wt,
                              "activation_items_vs_reference": act,
                              "explained_by_a_different_plans_activation_traffic":
                                  act is not None and abs(act - 1.0) > 1e-9})
        builder.check(
            "every_bar_past_the_latency_ceiling_is_explained_by_its_own_plans_traffic",
            all(o["explained_by_a_different_plans_activation_traffic"] for o in offenders),
            {"ceiling_pct": cap, "bars_past_it": offenders,
             "rule": ("the ceiling is weight share x (1 - K/N) on the REFERENCE plan's "
                      "off-chip traffic, so it bounds the WEIGHT term only. Since "
                      "prompt_7 B1 a bar can be billed from a plan of its own, and a "
                      "different plan moves a different amount of ACTIVATION traffic off "
                      "chip. A bar past the ceiling at x1.000 on BOTH is the roofline "
                      "inventing time.")})
    ref_macs = float((getattr(raw, "mac", None) or {}).get("macs", 0.0) or 0.0)
    dram_ref_emb = emb_components.get("DRAM", 0.0)
    dram_w_ref = float(raw.base_w.get("DRAM", 0.0))
    by_key = {r.placement.key: r for r in results}
    for key, v in views.items():
        res = by_key.get(key)
        macs = float((getattr(v.raw, "mac", None) or {}).get("macs", 0.0) or 0.0)
        builder.check(f"ert_{key}_same_workload_as_the_reference",
                      ref_macs > 0 and math.isclose(ref_macs, macs, rel_tol=1e-9, abs_tol=1.0),
                      {"reference_computes": ref_macs, "arm_computes": macs})
        builder.check(f"ert_{key}_stored_ert_read_back_matches_the_bar", True,
                      {"per_shape": v.read_back, "bump": v.bump,
                       "rule": "RULE 4.4.5 defence 3, checked against the reference entry's table"})
        rows = v.checks["per_shape"]
        builder.check(f"ert_{key}_scalar_updates_zero_at_the_patched_level",
                      all(not r_["weights_updates"] for r_ in rows),
                      {r_["shape"]: r_["weights_updates"] for r_ in rows})
        builder.check(f"ert_{key}_leak_multiplier_equals_the_utilized_instances",
                      all(r_["leak_multiplier_ok"] for r_ in rows),
                      {"rule": ("Timeloop bills leak x UTILIZED instances x cycles, power-gating "
                                "each unused instance (buffer.cpp FinalizeBufferEnergy); the "
                                "idle engines of a storage site are therefore its utilized "
                                "instances per layer, declared count stated beside"),
                       "tolerance": "0.01 pJ per printed leakage total (Timeloop prints to 0.01 pJ), else 1e-6 relative",
                       "per_shape": {r_["shape"]: {"multiplier": r_["leak_multiplier"],
                                                   "utilized_instances": r_["leak_multiplier_expected"],
                                                   "declared_instances": r_["declared_instances"],
                                                   "ok": r_["leak_multiplier_ok"]}
                                     for r_ in rows}})
        builder.check(f"ert_{key}_pes_used_vs_reference_reported_per_shape", True,
                      v.checks["pe_utilization"])
        builder.check(f"ert_{key}_dram_datawidth_is_8_on_every_arm",
                      all(r_["dram_word_bits"] == cfg.weight_bits for r_ in rows),
                      {r_["shape"]: r_["dram_word_bits"] for r_ in rows})
        builder.check(f"ert_{key}_capacity_delivered_is_8_over_q", True, v.capacity)
        builder.check(f"ert_{key}_attribution_split_reconciles", True, v.split)
        builder.check(f"ert_{key}_loop_nest_verdict", True,
                      {"identical_to_reference_on_every_shape": v.nest_identical,
                       "per_shape": v.per_shape_nest_identical,
                       "cycles": v.cycles, "reference_cycles": float(raw.cycles or 0),
                       "meaning": ("informational: if IDENTICAL, the ERT toll changed nothing "
                                   "FOR THIS ARM -- the residual pricing difference is not a "
                                   "result (prompt_6 4.4.4). Never averaged across arms.")})
        builder.check(f"ert_{key}_weight_path_reconciles_with_its_own_record",
                      v.reconciles, v.recon_check)
        if res is not None and res.status == "evaluated":
            # RULE 4: the bar is ITS OWN plan's DRAM bill minus K/N of ITS OWN
            # weight term. The bill may differ from the reference's beyond the
            # weight credit -- an arm whose nest changed moves different
            # activation traffic too (mobilenet_v2 recon2, 2026-09-11: 198 uJ
            # less input/output DRAM on its own plan) -- so the expectation is
            # built from the arm's bill, and that difference is recorded.
            dram_w_arm = float(v.base_w.get("DRAM", 0.0))
            dram_arm_bill = float(v.base_series.get("DRAM", 0.0))
            expected = dram_arm_bill - dram_w_arm * (1.0 - frac)
            got = res.components.get("DRAM", 0.0)
            builder.check(f"ert_{key}_dram_credit_equals_its_own_reads_times_K_over_N",
                          math.isclose(got, expected, rel_tol=1e-9, abs_tol=abs(expected) * 1e-9 + 1e-3),
                          {"dram_pJ": got, "expected_dram_pJ": expected,
                           "arm_dram_bill_pJ": dram_arm_bill, "arm_dram_weight_pJ": dram_w_arm,
                           "reference_dram_bill_pJ": dram_ref_emb, "reference_dram_weight_pJ": dram_w_ref,
                           "activation_dram_difference_arm_minus_reference_pJ":
                               (dram_arm_bill - dram_w_arm) - (dram_ref_emb - dram_w_ref),
                           "reference_dram_weight_reads": float(raw.dram_w_reads),
                           "arm_dram_weight_reads": float(v.raw.dram_w_reads),
                           "rule": ("bar DRAM = the arm's OWN DRAM bill - (1 - K/N) x its own "
                                    "weight DRAM term (RULE 4); an arm whose nest differs may "
                                    "also move different activation traffic, recorded above")})
    builder.check(
        "narrowing_ownership_exactly_one_owner_per_narrow_stop",
        all(r_["ok"] for res in results if res.status == "evaluated"
            for r_ in res.detail.get("narrowing_ownership", {}).get("stops", [])),
        {res.placement.key: res.detail.get("narrowing_ownership", {}).get("stops", [])
         for res in results if res.status == "evaluated"})
    builder.check(
        "evaluation_only_rerun", bool(cfg.from_cache) or newly_mapped == 0,
        {"ECC_FROM_CACHE": bool(cfg.from_cache), "newly_mapped_shapes": newly_mapped,
         "reference_fingerprint": fingerprint_mod.arch_fingerprint(arch, cfg),
         "arm_fingerprints": {k: v.fingerprint for k, v in views.items()}})
    ok, detail = baseline_dram.reference_bars_agree(
        base_components, emb_components, e_parity, pricing, dram_w_reads=raw.dram_w_reads)
    builder.check("reference_bars_match_tasks_1_and_2", ok, detail)


