"""Evaluator-only TIME: the roofline Timeloop's speed model cannot express.

prompt_7 Defect 1. Timeloop's throughput check counts ITEMS per cycle, and a
narrow weight is still one item (`block_size` appears nowhere in
`buffer.cpp:2476-2623`), so `datawidth: q` makes energy fall and leaves the
cycle count untouched -- a reported "0.00 % latency gain" is an ABSENT TERM,
not a result. And no architecture in `archs/` declares an off-chip bandwidth
at all, so `Read bandwidth : -` on DRAM skips the check entirely
(`buffer.cpp:437-447` has no `else`) and off-chip traffic costs zero cycles.

This module re-times the mapping AFTER it has been chosen, the same way
`noc_post.py` charges the two interconnect terms Timeloop cannot be given:

    cycles = max( compute cycles,
                  every storage level's own declared-bandwidth limit,
                  off-chip items / ECC_DRAM_BANDWIDTH_MBPS )

WHAT MAKES IT TRUSTWORTHY. At an UNLIMITED off-chip bandwidth the roofline
reproduces Timeloop's own per-level AND total cycle counts exactly -- all 43
cached shapes of `eyeriss_like_wglb`, including the 11 shapes `ifmap_glb`
throttles (worst 0.180) and the 5 `psum_glb` throttles (worst 0.610). It is
therefore not a second timing model but the same one, with the one term the
architectures never declared. `tests/test_latency.py` is that gate, and the
mutation that must break it is perturbing a single level's cycles.

WHAT SCALES AND WHAT DOES NOT. Off chip the weights are a BIT stream, so a
reconstruction arm reads `inputs + outputs + weights x K/N` items and the
saving is real. On chip a level holds WHOLE WEIGHTS at `q` bits: the words are
narrower, the item count is identical, and a bit-aware port is a change to the
architecture the mapper must see (`ECC_ONCHIP_BW_BITAWARE`, prompt_7 Phase C).
So `weight_scale` applies OFF CHIP ONLY. The on-chip levels are in the `max`
at their own unscaled demand, because they are what actually binds today
(prompt_7 section 4.5) and leaving them out would hand every shape to DRAM.

NETWORKS ARE NOT IN THE MAX AND CANNOT BE. `LegacyNetwork::ComputePerformance()`
is an empty stub, a network stats block carries no `Cycles` and no bandwidth
field of any kind, and no interconnect bandwidth is cited anywhere in this
study. That is reporting rule R-3, and it is why R3's latency equals R2's BY
CONSTRUCTION rather than by measurement.

Evaluator-only, exactly like `noc_post`: none of this is in the mapper's
objective and none of it is in the mapping fingerprint, so turning
`ECC_LATENCY_MODEL` on costs nothing and colds nothing. Declaring the same
bandwidth in the arch YAML, where the mapper would see it and choose a
different plan, is prompt_7 Phase C.
"""
from __future__ import annotations

import math

#: Dataspace whose off-chip item count the reduced representation shrinks.
REDUCED_DATASPACE = "Weights"


def is_offchip(level):
    """The DRAM level, by the same name test `energy.classify` uses.

    Networks never reach this module (they carry no timing), so there is no
    "NoC: DRAM <==> ifmap_glb" to exclude here the way `classify` must.
    """
    return "dram" in str(level).lower()


def offchip_items_per_cycle(cfg):
    """`ECC_DRAM_BANDWIDTH_MBPS` as items per cycle, or None for unlimited.

    env.sh section 6 states the conversion and this is the only implementation
    of it:  MB/s x 1e6 x the cycle period / bytes per item. At 480 MB/s and a
    1 ns model cycle that is 0.48 8-bit words per cycle; at the 200 MHz
    Eyeriss-v1 clock (`ECC_ARCH_CLOCK_MHZ`, Phase C) the same 480 MB/s is 2.4.
    None means the check is skipped, which is what every architecture in
    `archs/` gets today.
    """
    mbps = getattr(cfg, "dram_bandwidth_mbps", None)
    if not mbps:
        return None
    bytes_per_item = cfg.weight_bits / 8.0
    if bytes_per_item <= 0:
        return None
    return float(mbps) * 1e6 * cycle_seconds(cfg) / bytes_per_item


def cycle_seconds(cfg):
    """THE CLOCK PERIOD THIS RUN IS TIMED AT, as a float.

    prompt_7 C1.5: the period is per DESIGN now, and the architecture declares
    its off-chip limit in items per cycle OF THAT CLOCK. If this module kept
    reading the study-wide `global_cycle_seconds` it would convert the same
    MB/s at 1 GHz while the mapper converted it at 200 MHz -- a factor of five
    between the limit the plan was solved against and the limit it is reported
    against. `Config.cycle_seconds_for()` is the one place MHz becomes seconds.

    A configuration on more than one design has no single answer, and the
    placement study is pinned to one by env.sh section 4; anything else falls
    back to the study default, which is what it meant before Phase C.
    """
    archs = getattr(cfg, "archs", None) or []
    if len(archs) == 1 and hasattr(cfg, "cycle_seconds_for"):
        return float(cfg.cycle_seconds_for(archs[0]))
    return float(cfg.global_cycle_seconds)


def layer_record(levels):
    """`timeloop.physical_record()`, named for the caller that re-times a plan.

    The record is shared with `energy.standby_energy()` -- one row per level,
    so the module that charges TIME and the module that charges POWER x TIME
    cannot disagree about the same level's geometry.
    """
    from .timeloop import physical_record
    return physical_record(levels)


#: How close a stats-file scale has to be to the intended one to count as it.
SCALE_TOL = 5e-3


def relief_owner(row, weight_scale, tol=SCALE_TOL, code_frac=None):
    """Check this bar's declared scale against the run's, and return the factor
    THIS RE-TIMING applies -- which is `weight_scale`, always.

    A CORRECTION, 2026-09-13. This function first returned 1.0 when the arch
    declared the scale, on the reasoning that the MAPPER had already applied
    the relief and applying it here too would take it twice. **That was wrong,
    and the run said so immediately**: the roofline does not ADJUST Timeloop's
    cycles, it RECOMPUTES them from the item counts, so the scale belongs in
    that computation exactly once no matter who else used it. Suppressing it
    re-timed every reconstruction bar as if its relief did not exist --
    61,905,882 cycles against Timeloop's own 54,132,184, x1.1436, while the
    reference reproduced Timeloop exactly at x1.0000. **The roofline must
    reproduce Timeloop on EVERY arm once both see the same declared limit**;
    that is the Phase A gate, and one arm at x1.0000 beside five at x1.1436 is
    it failing.

    What survives is the CHECK, and it is worth keeping: Timeloop prints
    `Bandwidth Consumption Scale` on every level, 1.00 where nothing is
    declared, so a bar's own stats say which scale its plan was solved at.

        1.00 (or absent)   the plan was solved at full weight demand -- a
                           pre-Phase-C cache, or the reference arm
        == weight_scale    the plan was solved at this run's K/N
        anything else      STOP: a bar billed from a plan belonging to a
                           DIFFERENT CODE, which no factor can repair

    `weight_scale=None` means "WHATEVER THIS PLAN WAS SOLVED AT" -- the honest
    default for re-timing a plan rather than asking a what-if of it, and what
    `Raw.latency` uses. `code_frac` is the run's own K/N, used to recover full
    precision: Timeloop prints the scale to 2 dp (`0.62`), and re-timing at
    0.62 instead of 0.619048 is a 0.15% error in demand that a `ceil` can turn
    into a different cycle count.

    Returns `(factor, what the plan was solved at)`.
    """
    if not row.get("offchip"):
        return 1.0, "on chip: a narrow weight is still one item (Defect 1)"
    got = row.get("bw_consumption_scale")
    solved_reduced = got is not None and abs(float(got) - 1.0) > tol
    if solved_reduced:
        # THE PLAN WAS SOLVED AT A REDUCED WEIGHT DEMAND. Re-time it there, or
        # this re-timing describes a plan the mapper never chose.
        precise = None
        if weight_scale is not None and abs(float(got) - float(weight_scale)) <= tol:
            precise = float(weight_scale)
        elif code_frac is not None and abs(float(got) - float(code_frac)) <= tol:
            precise = float(code_frac)
        elif weight_scale is not None:
            raise ValueError(
                f"{row.get('level')}: the plan was solved at Bandwidth Consumption "
                f"Scale {got!r}, which is neither 1.00 nor the {weight_scale:.6f} "
                f"this bar is being re-timed at. That is a plan belonging to a "
                f"DIFFERENT CODE (prompt_6 RULE 1) -- refusing rather than "
                f"re-timing it at a fraction its own mapper never saw.")
        return ((precise if precise is not None else float(got)),
                f"the plan was solved at {got:g} and is re-timed there")
    if weight_scale is None or weight_scale == 1.0:
        return 1.0, "the plan was solved at full weight demand"
    return weight_scale, ("the plan was solved at FULL weight demand; the relief "
                          "is this re-timing's alone (a pre-Phase-C plan)")


def _items(row, key, weight_scale, code_frac=None):
    """Items of one direction at this level, with the off-chip weight relief.

    The relief is applied HERE because this is a RECOMPUTATION of the plan's
    time, not an adjustment of Timeloop's -- see `relief_owner`, which checks
    that the plan was solved at a scale this run can re-time.
    """
    scale, _why = relief_owner(row, weight_scale, code_frac=code_frac)
    return sum(float(v) * (scale if ds == REDUCED_DATASPACE else 1.0)
               for ds, v in (row.get(key) or {}).items())


def roofline(rec, cfg, *, weight_scale=None, offchip_limit=None):
    """Re-time ONE layer's plan. Returns per-level cycles and the binding level.

    `weight_scale` is the fraction of the WEIGHT bits this arm actually drives
    off the die: 1.0 for baseline and embedded, K/N for a reconstruction arm.
    **`None` -- the default -- means "whatever THIS PLAN was solved at"**, read
    off each level's own `Bandwidth Consumption Scale`. That is the honest
    setting for re-timing a plan; pass an explicit fraction only to ask a
    what-if of a plan that was solved at full demand (a pre-Phase-C cache).
    Before prompt_7 C1.2 no plan carried a scale and the two were the same
    thing; now a reconstruction arm's own plan is solved at K/N, and re-timing
    it at 1.0 reports a plan its mapper never chose -- 61,905,882 cycles
    against Timeloop's own 54,132,184.
    `offchip_limit` is items/cycle, or None to leave DRAM unlimited -- pass
    `offchip_items_per_cycle(cfg)` unless a caller is deliberately sweeping it.

    The arithmetic is Timeloop's own, term for term (`buffer.cpp:2553-2622`,
    `topology.cpp:1602`): demand is items per COMPUTE cycle, the slowdown is
    `min(1, declared / demand)` over each declared port, a level's cycles are
    `ceil(compute / slowdown)`, and the run is the WORST level -- not the sum.
    """
    compute = rec.get("compute_cycles")
    if not compute:
        return {"compute_cycles": compute, "cycles": None, "binding": None,
                "levels": [], "reason": "the plan carries no compute cycle count"}
    # The run's own K/N, so a 2-dp printed scale can be restored to full
    # precision rather than re-timing at 0.62 where the mapper used 0.619048.
    code_frac = None
    if getattr(cfg, "code_n", None) and getattr(cfg, "code_k", None):
        code_frac = float(cfg.code_k) / float(cfg.code_n)
    rows = []
    for row in rec.get("levels", []):
        if row.get("arithmetic"):
            # no bandwidth, no ports: the arithmetic level IS `compute`
            continue
        r = _items(row, "reads", weight_scale, code_frac)
        w = _items(row, "writes", weight_scale, code_frac)
        d_r, d_w = r / compute, w / compute
        # TIMELOOP'S OWN ARITHMETIC, ORDERED THE WAY TIMELOOP ORDERS IT.
        # `ceil(compute / (declared / demand))`, with the SHARED demand summed
        # BEFORE the division -- `(r + w) / compute`, never `r/compute +
        # w/compute`. All three forms are one number in algebra and three in
        # floating point, and the `ceil` turns the difference into a cycle:
        #
        #   C256_M512_R1_S1   d_r + d_w -> 416,001   (r+w)/c -> 416,000  [TL]
        #   C1_M1_R3_S3_G384  items/rate -> 295,040  (r+w)/c -> 295,041  [TL]
        #
        # Two shapes disagreeing about which shortcut is right is how a wrong
        # ordering announces itself: only the ratio form with one division
        # matches BOTH, on all 43 cached shapes. The Phase A gate is EXACT
        # reproduction, and a gate loosened to pass has stopped saying
        # anything -- so this is fixed in the arithmetic, not the tolerance.
        cycles, slow, binding_port = compute, 1.0, None
        for port, declared, items in (("read", row.get("read_bw"), r),
                                      ("write", row.get("write_bw"), w),
                                      ("shared", row.get("shared_bw"), r + w)):
            if not declared or items <= 0:
                continue
            this = float(declared) / (items / compute)
            if this < slow:
                slow, binding_port = this, port
        # The declared OFF-CHIP limit is one bus: reads and writes share it,
        # which is what "off-chip items / B" means -- so its demand is summed
        # before the division too.
        if row.get("offchip") and offchip_limit and (r + w) > 0:
            this = float(offchip_limit) / ((r + w) / compute)
            if this < slow:
                slow, binding_port = this, "off-chip"
        cycles = math.ceil(compute / min(1.0, slow))
        _f, why = relief_owner(row, weight_scale, code_frac=code_frac)
        rows.append(dict(row, demand_read=d_r, demand_write=d_w,
                         roofline_throttling=min(1.0, slow), port=binding_port,
                         relief_applied_here=_f, plan_solved_at=why,
                         roofline_cycles=cycles))
    worst = max(rows, key=lambda x: x["roofline_cycles"]) if rows else None
    return {"compute_cycles": compute,
            "cycles": worst["roofline_cycles"] if worst else compute,
            "binding": (worst["level"] if worst and worst["roofline_cycles"] > compute
                        else "compute"),
            "levels": rows}


def model_cycles(raw, cfg, *, weight_scale=None, offchip_limit=None):
    """The whole model's re-timed run length, and which level bound each layer.

    Layers are summed at their repeat count, exactly as `Raw.cycles` is, so the
    two are comparable to the cycle. A record with no timing detail returns
    None rather than a plausible-looking number built from nothing.
    """
    if offchip_limit is None:
        offchip_limit = offchip_items_per_cycle(cfg)
    total, binding, per_layer, missing, owners = 0, {}, [], 0, set()
    for lp in raw.per_layer:
        if lp.get("status") != "ok":
            continue
        rec = lp.get("physical")
        if not rec:
            missing += 1
            continue
        n = float(lp.get("repeat_count", 1) or 1)
        out = roofline(rec, cfg, weight_scale=weight_scale,
                       offchip_limit=offchip_limit)
        if out["cycles"] is None:
            missing += 1
            continue
        total += out["cycles"] * n
        # prompt_6 RULE 1, reported: WHO applied this bar's off-chip relief.
        # One owner per bar; a run that shows both is a run in which some plans
        # were solved against the reduced demand and some were not, and its
        # bars are not comparable.
        owners.update(r.get("plan_solved_at") for r in out["levels"]
                      if r.get("offchip"))
        binding[out["binding"]] = binding.get(out["binding"], 0) + 1
        per_layer.append({"layer": lp.get("layer"), "shape": lp.get("shape"),
                          "repeat_count": n,
                          "compute_cycles": out["compute_cycles"],
                          "cycles": out["cycles"], "binding": out["binding"],
                          "timeloop_cycles": lp.get("cycles")})
    if missing or not per_layer:
        return None
    total = int(total)                       # cycles are whole, the repeat count is not
    return {"cycles": total,
            "seconds": total * cycle_seconds(cfg),
            "cycle_seconds": cycle_seconds(cfg),
            "weight_scale": weight_scale,
            "offchip_items_per_cycle": offchip_limit,
            "plans_solved_at": sorted(o for o in owners if o),
            "binding_levels": binding,
            "per_layer": per_layer}


def offchip_items(raw, *, weight_scale=None, code_frac=None):
    """Off-chip items moved by the whole model: total, and the weight share.

    The denominator of the latency CEILING. Counted at the repeat count and
    from the same rows the roofline throttles, so the ceiling and the cycles
    cannot be computed from two different traffic figures.
    """
    # RAW ITEM COUNTS, unscaled. `weight_scale` is applied once, at the end,
    # to `reduced_weight_items` -- this is the CEILING's denominator and it is
    # a statement about the traffic the plan moves, not a re-timing of it.
    weights = other = 0.0
    for lp in raw.per_layer:
        if lp.get("status") != "ok" or not lp.get("physical"):
            continue
        n = float(lp.get("repeat_count", 1) or 1)
        for row in lp["physical"]["levels"]:
            if not row.get("offchip") or row.get("arithmetic"):
                continue
            for key in ("reads", "writes"):
                for ds, v in (row.get(key) or {}).items():
                    if ds == REDUCED_DATASPACE:
                        weights += float(v) * n
                    else:
                        other += float(v) * n
    total = weights + other
    return {"weight_items": weights, "other_items": other, "total_items": total,
            "weight_share": (weights / total) if total else 0.0,
            "reduced_weight_items": weights * (1.0 if weight_scale is None
                                               else weight_scale)}


def ceiling(raw, cfg, *, weight_scale=1.0):
    """The most latency a reconstruction boundary could buy: weight share x (1 - K/N).

    CLAUDE.md's rule that every result prints the ceiling first, applied to
    time. The chain is `(weight share of the binding resource) x (1 - K/N) x
    (how bound you are)`; this is the first two factors, so it is an upper
    bound reached only when the machine is fully off-chip-bound. A measured
    saving well below it is arithmetic, not a missing term.
    """
    t = offchip_items(raw, weight_scale=weight_scale)
    cut = t["weight_share"] * (1.0 - weight_scale)
    return {"weight_share_of_offchip_items": t["weight_share"],
            "one_minus_k_over_n": 1.0 - weight_scale,
            "traffic_cut": cut,
            "latency_ceiling_pct": cut * 100.0,
            "offchip_items": t,
            "note": ("an UPPER bound: it is reached only when off-chip bandwidth "
                     "is the binding resource on every layer. Total time is "
                     "max(compute, movement), not their sum, so a compute-bound "
                     "accelerator cannot be sped up by moving fewer weight bits "
                     "-- that is a property of the accelerator, not of the "
                     "reconstruction scheme")}


def describe(cfg):
    """One line for consoles and result notes."""
    # SINCE prompt_7 C1.1 THE MAPPER SEES THE LIMIT TOO. It is declared on the
    # DRAM level as `shared_bandwidth`, so Timeloop's own cycles already carry
    # it and this roofline re-states the plan's time rather than supplying the
    # only estimate of it. What stays evaluator-only is the re-timing itself:
    # none of it is in the mapping fingerprint.
    if not getattr(cfg, "latency_model", False):
        return ("latency model: off (ECC_LATENCY_MODEL=0) -- cycles are "
                "Timeloop's own. Since prompt_7 C1.1 those already include the "
                "declared off-chip limit, so they are a real run length; this "
                "roofline is what re-states it per bar")
    lim = offchip_items_per_cycle(cfg)
    if lim is None:
        return ("latency model: roofline at UNLIMITED off-chip bandwidth "
                "(ECC_DRAM_BANDWIDTH_MBPS empty) -- reproduces Timeloop's "
                "cycles exactly; nothing moves until a limit is declared")
    return (f"latency model: roofline, off-chip limit "
            f"{cfg.dram_bandwidth_mbps:g} MB/s = {lim:.4g} items/cycle at "
            f"{cycle_seconds(cfg) * 1e9:g} ns; ALSO declared on the DRAM level "
            f"since prompt_7 C1.1, so the mapper optimised against it -- the "
            f"re-timing here is evaluator-only and not in the fingerprint")
