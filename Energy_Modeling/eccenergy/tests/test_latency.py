"""prompt_7 Phase A: the roofline, the ceiling, and standby energy.

PROPERTY TESTS ON REAL CACHED DATA, PLUS DELIBERATE BREAKAGE. Every assertion
here reads the mapper cache that is on disk -- no fixture, no synthetic stats
file -- and every one is followed by the mutation it exists to catch, applied
to a copy and checked to FAIL. An assertion nobody has seen fail is a comment.

THE FOUR PROPERTIES, AND WHY EACH ONE IS THE GATE IT IS
------------------------------------------------------
1. `test_latency_roofline_matches_timeloop`
   At UNLIMITED off-chip bandwidth the roofline must reproduce Timeloop's own
   per-level AND total cycle counts EXACTLY, on every cached shape. This is
   what makes `latency_post.py` the same timing model with one missing term
   rather than a second, plausible-looking one. THE ROOFLINE IS NOT ALLOWED TO
   INVENT TIME.  Breakage: perturb one level's cycles.
2. `test_latency_ceiling`
   ceiling = (weight share of off-chip items) x (1 - K/N), to 2 dp.
   Breakage: change K/N.
3. `test_static_energy_is_symmetric`
   ECC_STATIC_ENERGY=1 moves ALL THREE arms by the SAME amount; =0 reproduces
   the pre-Phase-A totals to the pJ. Defect 2 is that one side of the
   comparison was charged standby power and the other was not, so a fix that
   moves one arm is the same bug with a switch on it.
   Breakage: charge it to the reconstruction arm only.
4. `test_leakage_is_parsed`
   `Leakage energy (total)` -- which the MAPPER has been optimising against all
   along and the report threw away (rule R-4) -- reaches the reported record.
   Breakage: drop the regex.

Run it like every other suite:

    bash hpc/tl.sh python3 -m eccenergy.tests.test_latency
"""
from __future__ import annotations

import dataclasses
import math
import os
import pathlib
import sys

import pandas as pd

from .. import config
from ..study import energy, stacks
from ..toolchain import latency_post
from ..toolchain.stats import parse_cycles, parse_levels, physical_record

FAILED = []

#: The design prompt_7 quotes every measured number from, and the treatment the
#: live env.sh resolves to. Chosen by SHAPE COUNT, not by name: whichever
#: fingerprint of this design holds the most solved shapes is the one with
#: something to test. Nothing here writes to the cache.
ARCH = "eyeriss_like_wglb"

#: How far the roofline may sit from Timeloop on ONE level, in cycles.
#: Measured, not chosen: with an off-chip limit declared, both sides run a float
#: ratio through a `ceil`, and no candidate ordering reproduces all 301 cached
#: levels (the best two miss 2, each by exactly +1 -- 6.4e-07 of the count).
#: Timeloop's own float ordering is not recoverable from the stats file.
CYCLE_TOL = 1

#: ...and a one-cycle rounding must stay RARE. A systematic drift would show as
#: many levels off by one, which `MIN_EXACT_FRACTION` refuses. Measured: 299 of
#: 301 exact = 99.3%.
MIN_EXACT_FRACTION = 0.95
CACHE_ROOT = pathlib.Path("ecc_energy_study/outputs") / ARCH


def check(name, fn):
    try:
        fn()
    except Exception as exc:                       # noqa: BLE001 -- a report
        FAILED.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  FAIL  {name}\n          {type(exc).__name__}: {exc}")
    else:
        print(f"  ok    {name}")


def expect_raises(fn, what):
    """`fn` MUST raise. Used for every mutation test."""
    try:
        fn()
    except Exception:                              # noqa: BLE001 -- that is the point
        return
    raise AssertionError(f"the deliberate breakage did NOT fail: {what}")


def _assert_eq(a, b):
    assert a == b, f"{a} != {b}"


class _Cfg:
    """The fields `latency_post` reads, so the roofline can be tested without a
    Session, a container or an environment. Deliberately NOT a real Config: a
    test that had to resolve the whole env.sh would be testing env.sh."""
    weight_bits = 8
    global_cycle_seconds = "1e-9"
    dram_bandwidth_mbps = None                     # unlimited: the gate condition
    latency_model = True
    classify_mode = "instances"
    leakage_nw = {"sram_bit": 2.693, "rf_bit": 70.0, "mac_instance": 7844.9}
    static_energy = True
    split_read_write = False


def _reference_cache():
    """The cache THE LIVE CONFIGURATION resolves to, for the REFERENCE arm.

    Not "whichever directory holds the most shapes". Two reasons, both learned
    the hard way while writing this suite:

      * an `ert-` arm was MAPPED with the reconstruction engine's toll injected
        into its ERT, so its `Leakage energy (total)` is the SRAM's leak PLUS
        an encoder's -- 147,351 pJ against the reference arm's 14, which is
        prompt_7 section 5.2b's "~1,500x the SRAM it sits next to". Testing the
        replacement densities against that measures the encoder.
      * the retired `random_pruned` / unconstrained caches are A DIFFERENT
        DATAFLOW (CLAUDE.md: a constrained/relaxed mapspace is a different
        accelerator). On one of those `filter_glb` throttles to 0.320, so rule
        R-2 -- "only ifmap_glb and psum_glb ever bind" -- is FALSE there. R-2
        is a statement about the constrained dataflow in force, and the suite
        has to read that dataflow's cache or it is testing another chip.

    `hpc/tl.sh` sources env.sh, so the live knobs are in the environment and
    `config.load_config()` is the same resolution every other stage does.
    Nothing here writes to the cache (`create=False`).
    """
    from ..arch import fingerprint
    from ..paths import Results
    cfg = config.load_config()
    variant = fingerprint.effective_variant(ARCH, cfg)
    fp = Results(cfg).mapper_cache(ARCH, variant, fingerprint.arch_fingerprint(ARCH, cfg),
                                   create=False)
    n = (sum(1 for d in fp.iterdir()
             if d.is_dir() and (d / "timeloop-mapper.stats.txt").exists())
         if fp.exists() else 0)
    return fp, n, variant


#: How many solved shapes these property tests need to say anything. Below it
#: the suite SKIPS (see `main`): the live treatment is simply not mapped yet.
MIN_SHAPES = 10


def _biggest_cache():
    """The live reference cache, or a SKIP naming what is missing."""
    fp, n, variant = _reference_cache()
    if n < MIN_SHAPES:
        # A COLD CACHE IS NOT A FAILING PROPERTY, it is an unchecked one -- and
        # changing an architecture, which is what these knobs are for, colds it
        # by design. Raising here turned every ordinary experiment into ten red
        # tests that said nothing about the code (2026-09-13). `pytest.skip`
        # says the same thing without the false alarm; the message is unchanged
        # and still names the directory and what fills it.
        import pytest as _pytest
        _pytest.skip(
            f"the live configuration resolves to {fp} for {ARCH}, which holds "
            f"{n} solved shape(s). This suite is a PROPERTY TEST ON REAL CACHED "
            f"DATA and will not fall back to another treatment: a different "
            f"mapspace is a different accelerator, and rule R-2 is false on the "
            f"retired unconstrained caches. Map it first "
            f"(`bash hpc/run_all.sh --map-only`), or point ECC_* at a configuration "
            f"whose reference arm is already solved. Treatment: {variant}")
    return fp, n


def _shapes(fp):
    return sorted(d for d in fp.iterdir()
                  if d.is_dir() and (d / "timeloop-mapper.stats.txt").exists())


# ------------------------------------------------------------------- 1. gate 1
def test_latency_roofline_matches_timeloop():
    """THE GATE: the roofline reproduces Timeloop's cycles EXACTLY, every shape.

    Both halves are asserted: the TOTAL (`max` over levels, topology.cpp:1602)
    and every LEVEL's own throttled count (buffer.cpp:2553-2622). A roofline
    that got the total right by luck while mis-timing the levels would pass a
    total-only check and then mis-attribute every bottleneck.

    IT USED TO SAY "at UNLIMITED bandwidth", and that wording died with
    prompt_7 C1.1. Until Phase C no architecture declared an off-chip limit, so
    the cached cycles WERE the unlimited ones and the evaluator had to be told
    to leave DRAM alone to reproduce them. Now the DRAM level declares
    `shared_bandwidth`, Timeloop's own cycles include it, and the roofline
    reads it back out of the same stats -- so the gate is the same property
    against a cache that has a limit in it. `offchip_limit` stays None here
    because the limit is IN THE RECORD; passing it again would be the same
    number twice.

    HOW EXACT IT CAN BE, MEASURED RATHER THAN ASSUMED. Before C1.1 the answer
    was "to the cycle", and that was easy because DRAM had no declared limit to
    round. With one, the reproduction runs through a float ratio and a `ceil`,
    and Timeloop's own ordering of that arithmetic is NOT recoverable from the
    stats file. All five candidate orderings were swept over the whole cache
    (301 levels, 43 shapes):

        2 wrong   ceil(compute / (declared / ((r+w)/compute)))   <-- adopted
        2 wrong   the same with r/compute + w/compute
        4 wrong   ceil(items / declared)
        4 wrong   round-then-ceil
        6 wrong   ceil(compute * ((items/compute) / declared))

    NONE reproduces all 301, and every miss is +1 cycle -- 6.4e-07 of the
    level's count. So the gate asserts what is actually true and still catches
    what it exists to catch: NO LEVEL MAY BE MORE THAN ONE CYCLE OUT, and the
    overwhelming majority must be exact. A roofline that invented a term would
    be wrong by a lot, on many levels; float-plus-ceil is wrong by one, on two.
    This is the achievable precision stated, NOT a tolerance widened until a
    failure went away -- the difference is that the bound and the exact
    fraction are both asserted, so a systematic drift cannot hide inside it.
    """
    fp, n = _biggest_cache()
    assert fp is not None and n >= 10, (
        f"no usable mapper cache for {ARCH} (found {n} solved shapes) -- this "
        f"suite reads the cache that is on disk and cannot run without one")
    cfg = _Cfg()
    n_levels, n_exact, worst = 0, 0, []
    for d in _shapes(fp):
        stats = d / "timeloop-mapper.stats.txt"
        rec = physical_record(parse_levels(stats)[0])
        out = latency_post.roofline(rec, cfg)
        tl_total = parse_cycles(stats)
        assert abs(out["cycles"] - tl_total) <= CYCLE_TOL, (
            f"{d.name}: roofline {out['cycles']:,} vs Timeloop {tl_total:,} "
            f"-- more than {CYCLE_TOL} cycle out, so this is not a rounding "
            f"difference")
        for row in out["levels"]:
            if row["cycles"] is None:
                continue
            delta = row["roofline_cycles"] - row["cycles"]
            assert abs(delta) <= CYCLE_TOL, (
                f"{d.name}/{row['level']}: roofline {row['roofline_cycles']:,} vs "
                f"Timeloop {row['cycles']:,} ({delta:+,}) -- more than "
                f"{CYCLE_TOL} cycle out")
            n_levels += 1
            if delta == 0:
                n_exact += 1
            else:
                worst.append(f"{d.name}/{row['level']} {delta:+d}")
    # AND THE OVERWHELMING MAJORITY MUST BE EXACT. The 1-cycle allowance above
    # is float-plus-ceil; a systematic drift would show up as many levels off
    # by one, which this refuses.
    frac = n_exact / n_levels if n_levels else 0.0
    assert frac >= MIN_EXACT_FRACTION, (
        f"only {n_exact}/{n_levels} ({frac:.1%}) of levels reproduce Timeloop "
        f"exactly, below {MIN_EXACT_FRACTION:.0%}. A one-cycle rounding is rare; "
        f"this many is a different arithmetic: {'; '.join(worst[:6])}")
    print(f"          {n} shapes, {n_levels} storage levels, {n_exact} exact "
          f"({frac:.1%}), none more than {CYCLE_TOL} cycle out"
          + (f"  [{'; '.join(worst)}]" if worst else ""))

    # BREAKAGE: perturb ONE level's cycles. The per-level comparison must fail,
    # which is what stops a roofline that quietly re-times a level nobody looks
    # at from passing on the total alone.
    d = _shapes(fp)[0]
    rec = physical_record(parse_levels(d / "timeloop-mapper.stats.txt")[0])
    storage = [r for r in rec["levels"] if not r["arithmetic"]]
    # +2, not +1: one cycle is the float-plus-ceil rounding the gate allows,
    # so a mutation of one cycle would be indistinguishable from it. Two is
    # outside the bound and must fail.
    hurt = {"compute_cycles": rec["compute_cycles"],
            "levels": [dict(r, cycles=(r["cycles"] or 0) + 2) if i == 0 else r
                       for i, r in enumerate(storage)]}
    def _compare():
        out = latency_post.roofline(hurt, cfg)
        for row in out["levels"]:
            if row["cycles"] is None:
                continue
            assert abs(row["roofline_cycles"] - row["cycles"]) <= CYCLE_TOL
    expect_raises(_compare, "a level whose cycles were perturbed by 1 still matched")

    # BREAKAGE: halve EVERY declared bandwidth, `shared_bandwidth` included.
    # It used to halve `read_bw` alone, which stopped biting the moment C1.1
    # gave DRAM a `shared_bandwidth`: DRAM then binds every shape on this
    # design, and its limit is not a read limit, so the mutation moved a level
    # nobody was waiting on and the total did not budge. A mutation that
    # cannot fail is not a test.
    def _halve(r):
        return dict(r, **{k: (r[k] / 2 if r.get(k) else None)
                          for k in ("read_bw", "write_bw", "shared_bw")})
    slower = {"compute_cycles": rec["compute_cycles"],
              "levels": [_halve(r) for r in rec["levels"]]}
    tl = parse_cycles(d / "timeloop-mapper.stats.txt")
    expect_raises(
        lambda: _assert_eq(latency_post.roofline(slower, cfg)["cycles"], tl),
        "halving every declared bandwidth changed no cycle")


def test_roofline_binds_only_where_the_cache_says_it_does():
    """prompt_7 rule R-2, re-measured rather than quoted. RESTATED 2026-09-13.

    R-2 USED TO SAY the only levels that throttle carry INPUTS and PARTIAL
    SUMS, which reconstruction cannot touch -- the honest headline of Defect 1.
    **That was never a fact about the chip; it was the absence of a declared
    number.** No architecture declared an off-chip bandwidth, so Timeloop
    skipped the DRAM throughput check entirely. prompt_7 C1.1 declares it, and
    DRAM binds.

    So this test now asserts the INVERTED rule, still from the cache and never
    from the prose: with a limit declared, DRAM throttles and is the binding
    level; without one, it cannot be. `filter_glb` still never throttles, and
    still sits exactly on its declared 16 items/cycle at the fully-connected
    layers -- "never throttles" and "never binds" are different claims and only
    the first is made here.
    """
    fp, n = _biggest_cache()
    assert fp is not None
    cfg = _Cfg()
    worst, declared_offchip = {}, False
    for d in _shapes(fp):
        rec = physical_record(parse_levels(d / "timeloop-mapper.stats.txt")[0])
        for row in latency_post.roofline(rec, cfg)["levels"]:
            t = row["roofline_throttling"]
            worst[row["level"]] = min(worst.get(row["level"], 1.0), t)
            # WHETHER DRAM HAS A LIMIT IS A PROPERTY OF THE CACHE, not of this
            # evaluator's config: prompt_7 C1.1 puts `shared_bandwidth` in the
            # ARCH, so it arrives in the stats of every shape mapped under it,
            # and a cache mapped before C1.1 carries none however
            # ECC_DRAM_BANDWIDTH_MBPS is set today. Reading the knob instead
            # asked the wrong object and failed a correct run.
            if row.get("offchip") and any(row.get(k) for k in
                                          ("shared_bw", "read_bw", "write_bw")):
                declared_offchip = True
    throttled = {k: v for k, v in worst.items() if v < 1.0}
    assert throttled, "no level throttles anywhere -- the cache changed shape"
    for level in throttled:
        assert level in ("ifmap_glb", "psum_glb", "DRAM"), (
            f"{level} now throttles (worst {throttled[level]:.3f}). R-2 names "
            f"ifmap_glb and psum_glb (inputs and partial sums, which "
            f"reconstruction cannot touch) and, since prompt_7 C1.1 declared "
            f"an off-chip limit, DRAM. A FOURTH level is a new claim: restate "
            f"R-2 in config.reporting_rules() and prompt_7 section 4.5.")
    # THE INVERTED HALF, and the one that matters for every latency number:
    # DRAM throttles IF AND ONLY IF the architecture declares a limit for it.
    if declared_offchip:
        assert worst.get("DRAM", 1.0) < 1.0, (
            f"this cache's DRAM level declares a bandwidth (prompt_7 C1.1), "
            f"so DRAM must be able to throttle "
            f"-- it does not on any of these {n} shapes. Either the "
            f"declaration did not reach the mapper or the roofline is not "
            f"reading it back, and every latency saving in this run is then "
            f"measured against a memory with no speed limit.")
        assert "filter_glb" not in throttled, (
            f"filter_glb throttles (worst {throttled.get('filter_glb')}); R-2 "
            f"says it never does under the constrained mapspace")
    else:
        # NO DECLARED LIMIT, NO OFF-CHIP THROTTLE. This is the half that was
        # the whole of R-2 before prompt_7 C1.1, and it is still a real check:
        # with ECC_DRAM_BANDWIDTH_MBPS empty, nothing may slow DRAM down --
        # neither `offchip_limit` leaking into the roofline nor a
        # `shared_bandwidth` left behind in the arch.
        assert worst.get("DRAM", 1.0) == 1.0, (
            "DRAM throttles with no off-chip limit declared, which is "
            "impossible -- either offchip_limit leaked into the gate or the "
            "cache was mapped from an arch that declares one")
    print(f"          worst throttling: "
          + ", ".join(f"{k} {v:.3f}" for k, v in sorted(throttled.items()))
          + ("  (the cache's DRAM level declares a bandwidth)"
             if declared_offchip else "  (no off-chip limit in the cache)"))


# ------------------------------------------------------------------- 2. gate 4
class _FakeRaw:
    """The three attributes `latency_post` reads off a `Raw`."""
    def __init__(self, per_layer, cycles=None):
        self.per_layer = per_layer
        self.cycles = cycles


def _raw_from_cache(fp, limit=None):
    layers = []
    for d in _shapes(fp)[:limit]:
        stats = d / "timeloop-mapper.stats.txt"
        layers.append({"layer": d.name, "shape": d.name, "status": "ok",
                       "repeat_count": 1,
                       "cycles": parse_cycles(stats),
                       "physical": physical_record(parse_levels(stats)[0])})
    return _FakeRaw(layers, cycles=sum(l["cycles"] for l in layers))


def test_latency_ceiling():
    """ceiling = weight share of off-chip items x (1 - K/N), to 2 dp.

    CLAUDE.md's rule that every result prints its ceiling first, applied to
    time: without it a 4 % saving reads as a missing term rather than as
    arithmetic. It is an UPPER bound -- the third factor, how off-chip-bound the
    machine is, is not in it.
    """
    fp, _ = _biggest_cache()
    raw = _raw_from_cache(fp)
    cfg = _Cfg()
    for k, n in ((30, 63), (39, 63), (51, 63)):
        frac = k / n
        c = latency_post.ceiling(raw, cfg, weight_scale=frac)
        share = c["weight_share_of_offchip_items"]
        want = round(share * (1.0 - frac), 2)
        assert round(c["traffic_cut"], 2) == want, (
            f"BCH({n},{k}): ceiling {c['traffic_cut']:.4f} != weight share "
            f"{share:.4f} x (1 - K/N) {1 - frac:.4f} = {share * (1 - frac):.4f}")
        # and it must actually BE the roofline's best case: no arm can beat it
        ref = latency_post.model_cycles(raw, cfg, offchip_limit=1e-9)["cycles"]
        got = latency_post.model_cycles(raw, cfg, weight_scale=frac,
                                        offchip_limit=1e-9)["cycles"]
        gain = (ref - got) / ref
        assert gain <= c["traffic_cut"] + 1e-6, (
            f"BCH({n},{k}): a fully off-chip-bound run saved {gain * 100:.3f}%, "
            f"above its own ceiling {c['latency_ceiling_pct']:.3f}%")
        assert gain > c["traffic_cut"] - 1e-3, (
            f"BCH({n},{k}): fully off-chip-bound, the saving {gain * 100:.3f}% "
            f"should REACH the ceiling {c['latency_ceiling_pct']:.3f}%")
    print(f"          weight share of off-chip items: "
          f"{latency_post.ceiling(raw, cfg, weight_scale=30 / 63)['weight_share_of_offchip_items'] * 100:.2f}%")

    # BREAKAGE: change K/N. The ceiling MUST move -- a ceiling that does not
    # respond to the code is not a function of the code.
    a = latency_post.ceiling(raw, cfg, weight_scale=30 / 63)["traffic_cut"]
    b = latency_post.ceiling(raw, cfg, weight_scale=51 / 63)["traffic_cut"]
    expect_raises(lambda: _assert_eq(round(a, 2), round(b, 2)),
                  "BCH(63,30) and BCH(63,51) gave the same latency ceiling")


def test_only_offchip_weight_items_are_relieved():
    """`weight_scale` is OFF CHIP ONLY, and only on Weights.

    Off chip the weights are a BIT stream, so K/N of the bits is K/N of the
    traffic. On chip a level holds WHOLE WEIGHTS at q bits: the words are
    narrower and the ITEM count is identical, which is precisely why Timeloop's
    speed model cannot see the reduced representation at all (Defect 1). A
    bit-aware on-chip port is an architecture change the mapper must see
    (ECC_ONCHIP_BW_BITAWARE, Phase C), never an evaluator rescale here.
    """
    fp, _ = _biggest_cache()
    rec = physical_record(parse_levels(_shapes(fp)[0] / "timeloop-mapper.stats.txt")[0])
    cfg = _Cfg()
    full = {r["level"]: r for r in latency_post.roofline(rec, cfg)["levels"]}
    half = {r["level"]: r for r in
            latency_post.roofline(rec, cfg, weight_scale=0.5)["levels"]}
    for name, row in full.items():
        if row["offchip"]:
            continue
        assert half[name]["demand_read"] == row["demand_read"], (
            f"{name} is ON CHIP and its read demand moved with weight_scale "
            f"({row['demand_read']} -> {half[name]['demand_read']}); a narrow "
            f"weight is still one item")
    dram = [r for r in full.values() if r["offchip"]]
    assert dram, "no off-chip level in this plan"
    assert half["DRAM"]["demand_read"] < full["DRAM"]["demand_read"], (
        "DRAM's read demand did NOT fall at weight_scale 0.5 -- the one place "
        "the reduced representation really does move fewer bits")


# ------------------------------------------------------------------- 3. gate 2
def _cfgs():
    """The live configuration, with standby off and on. Nothing else moves."""
    cfg = config.load_config()
    off = cfg.with_(static_energy=False, latency_model=False)
    on = cfg.with_(static_energy=True, latency_model=False)
    return off, on


def _stack_totals(cfg, raw):
    recon_pj, idle_pj, _ = stacks.load_recon_energy(cfg)
    df = stacks.build_stacks(cfg, raw, recon_pj, recon_idle_pj=idle_pj)
    return {a: float(df[a].sum()) for a in cfg.approaches}


def test_static_energy_is_symmetric():
    """ECC_STATIC_ENERGY=1 moves all three arms, by the SAME pJ; =0 moves none.

    Both halves matter. The first is Defect 2's fix: standby power is a
    property of the chip, not of where the parity lives, so it lands in
    `Raw.base` -- which every arm and every placement bar starts from -- and
    there is no code path that can give it to one arm and not another. The
    second is what lets Phase A land without invalidating a single earlier
    number.
    """
    off, on = _cfgs()
    fp, _ = _biggest_cache()
    raw = _raw_from_cache(fp, limit=6)
    # a Raw with just enough for build_stacks, on real cached physical records
    cats = energy.plot_cats(on)
    base = pd.Series({c: 0.0 for c in cats})
    base["DRAM"] = 1.0e9
    base["Compute"] = 2.0e8
    full = energy.Raw(base.reindex(energy.plot_cats(off), fill_value=0.0),
                      base.reindex(energy.plot_cats(off), fill_value=0.0) * 0.0,
                      base.reindex(energy.plot_cats(off), fill_value=0.0) * 0.0,
                      1.0e9, 1.0e6, len(raw.per_layer), 0, 10_000,
                      per_layer=raw.per_layer, levels=[], cycles=raw.cycles)

    t_off = _stack_totals(off, energy.apply_standby_energy(full, off, verbose=False))
    assert all(v > 0 for v in t_off.values())
    e_off = energy.standby_energy(full, on)[0]
    assert e_off > 0, "the standby model charges nothing at all on real cached data"

    on_raw = energy.apply_standby_energy(full, on, verbose=False)
    t_on = _stack_totals(on, on_raw)
    deltas = {a: t_on[a] - t_off[a] for a in t_off}
    assert all(d > 0 for d in deltas.values()), (
        f"ECC_STATIC_ENERGY=1 did not move every arm: {deltas}. If only one "
        f"moves, STOP -- that is Defect 2 with a switch on it.")
    lo, hi = min(deltas.values()), max(deltas.values())
    assert math.isclose(lo, hi, rel_tol=1e-12), (
        f"the three arms were charged DIFFERENT standby energies {deltas}; "
        f"standby power is a property of the chip, not of where parity lives")
    assert math.isclose(lo, e_off, rel_tol=1e-12), (
        f"the arms moved by {lo:.3f} pJ but standby_energy says {e_off:.3f} pJ")
    print(f"          all {len(deltas)} arms +{lo / 1e6:,.3f} uJ; "
          f"ECC_STATIC_ENERGY=0 charges 0.000 uJ")

    # BREAKAGE: charge it to the reconstruction arm only -- the exact shape of
    # Defect 2, which is what this assertion exists to make impossible.
    bad = dict(t_off)
    bad["recon"] = bad["recon"] + e_off
    expect_raises(
        lambda: _assert_eq(len({round(bad[a] - t_off[a], 6) for a in bad}), 1),
        "standby charged to the recon arm alone was accepted as symmetric")

    # BREAKAGE: an ECC_LEAKAGE_NW with a missing density must REFUSE, not
    # silently charge zero. A knob that silently does nothing is the same bug.
    starved = on.with_(leakage_nw={"sram_bit": 2.693})
    expect_raises(lambda: energy.standby_energy(full, starved),
                  "a missing rf_bit density charged zero instead of refusing")


def test_standby_scales_with_time_not_with_traffic():
    """Standby energy is power x TIME: double the run, double the charge.

    And it is absent from `base_w`, so the reconstruction arm never scales it
    by K/N -- it carries no weight traffic, only run length. That is why
    ECC_LATENCY_MODEL feeds it and ECC_DRAM_PJ_PER_BIT does not.
    """
    off, on = _cfgs()
    fp, _ = _biggest_cache()
    raw = _raw_from_cache(fp, limit=6)
    base = pd.Series({c: 0.0 for c in energy.plot_cats(on)})
    full = energy.Raw(base, base * 0.0, base * 0.0, 0.0, 0.0,
                      len(raw.per_layer), 0, 0, per_layer=raw.per_layer,
                      levels=[], cycles=raw.cycles)
    e1 = energy.standby_energy(full, on, cycle_scale=1.0)[0]
    e2 = energy.standby_energy(full, on, cycle_scale=2.0)[0]
    assert math.isclose(e2, 2.0 * e1, rel_tol=1e-12), (
        f"twice the run length charged {e2 / e1:.6f}x the standby energy")
    # AND THE PERIOD IS APPLIED ONCE: five times the cycle PERIOD is five times
    # the standby energy for the same CYCLE count, because ECC_LEAKAGE_NW is
    # POWER in nW (env.sh section 6 TRAP 2).
    #
    # THE PERIOD'S OWNER CHANGED IN prompt_7 C1.5. This used to vary
    # `global_cycle_seconds` and assert the x5; that knob is now the STUDY
    # DEFAULT and `ECC_ARCH_CLOCK_MHZ` overrides it per design, so on a design
    # with a table entry -- eyeriss_like_wglb, at its published 200 MHz --
    # changing the global default correctly changes nothing. Vary the thing
    # that owns the period, and assert the fallback separately.
    period = latency_post.cycle_seconds(on)
    fast = on.with_(arch_clock_mhz={ARCH: 1000.0})   # 1 ns
    slow = on.with_(arch_clock_mhz={ARCH: 200.0})    # 5 ns
    e_fast = energy.standby_energy(full, fast)[0]
    e_slow = energy.standby_energy(full, slow)[0]
    assert math.isclose(e_slow, 5.0 * e_fast, rel_tol=1e-12), (
        f"ECC_LEAKAGE_NW is POWER in nW; five times the cycle PERIOD must be "
        f"five times the energy for the same cycle count -- got "
        f"{e_slow / e_fast:.6f}x (env.sh section 6 TRAP 2)")
    # the design in play is one of them, so `e1` is whichever its clock is
    assert math.isclose(e1, e_slow if period == 5e-09 else e_fast, rel_tol=1e-12), (
        f"the standby charge was not computed at this design's own period "
        f"({period:g} s)")
    # AND THE FALLBACK: a design with NO table entry keeps the study default,
    # which is what makes `ECC_ARCH_CLOCK_MHZ` add designs rather than replace
    # the global.
    none_ = on.with_(arch_clock_mhz={})
    assert math.isclose(
        energy.standby_energy(full, none_)[0],
        e_fast * float(none_.global_cycle_seconds) / 1e-9, rel_tol=1e-12), (
        "with no ECC_ARCH_CLOCK_MHZ entry the standby charge must fall back to "
        "ECC_GLOBAL_CYCLE_SECONDS")
    # BREAKAGE: the period applied TWICE is env.sh TRAP 2's silent 5x
    expect_raises(lambda: math.isclose(e_slow, 25.0 * e_fast, rel_tol=1e-12)
                  or _assert_eq(1, 0),
                  "the cycle period was applied twice")
    expect_raises(lambda: _assert_eq(e1, e2),
                  "the standby charge did not respond to the run length at all")


# ------------------------------------------------------------------- 4. R-4
def test_leakage_is_parsed():
    """`Leakage energy (total)` reaches the reported record (rule R-4).

    Until Phase A `parse_stats` read `Energy (total)` from inside each
    dataspace sub-block and left this one, which sits in the SPECS block above
    it, on the floor. It was never absent from the MODEL: Timeloop's `Energy:`
    and `EDP(J*cycle)` -- the mapper's objective -- already contain it. So the
    statement "leakage is absent" was wrong as written, and the correct one
    names which side dropped it.
    """
    fp, _ = _biggest_cache()
    raw = _raw_from_cache(fp, limit=6)
    off, on = _cfgs()
    comps = energy.standby_components(raw, on)
    assert comps, "no components parsed at all"
    billed = {c["level"]: c["timeloop_leakage_pJ"] for c in comps}
    live = {k: v for k, v in billed.items() if v > 0}
    assert live, (
        f"Timeloop's `Leakage energy (total)` reached the record as 0 on every "
        f"level ({sorted(billed)}) -- the regex missed")
    info = energy.apply_standby_energy(
        energy.Raw(pd.Series({c: 0.0 for c in energy.plot_cats(off)}),
                   pd.Series({c: 0.0 for c in energy.plot_cats(off)}),
                   pd.Series({c: 0.0 for c in energy.plot_cats(off)}),
                   0.0, 0.0, 1, 0, 0, per_layer=raw.per_layer, levels=[],
                   cycles=raw.cycles), off, verbose=False).standby
    assert info["timeloop_leakage_pJ"] > 0, (
        "with ECC_STATIC_ENERGY=0 the record must still report what Timeloop "
        "billed for leakage -- that is the whole of rule R-4")
    print(f"          Timeloop billed {info['timeloop_leakage_pJ']:.4f} pJ across "
          f"{len(live)} level(s); replacement densities charge "
          f"{energy.standby_energy(_raw_from_cache(fp, limit=6), on)[0]:.1f} pJ")

    # BREAKAGE: drop the regex. Every level then reports 0 leakage and R-4
    # cannot be stated at all.
    blind = _FakeRaw([dict(lp, physical={
        "compute_cycles": lp["physical"]["compute_cycles"],
        "levels": [dict(r, timeloop_leakage_pJ=None) for r in lp["physical"]["levels"]]})
        for lp in raw.per_layer], cycles=raw.cycles)
    def _still_reported():
        c = energy.standby_components(blind, on)
        assert any(x["timeloop_leakage_pJ"] > 0 for x in c), "all zero"
    expect_raises(_still_reported, "dropping the leakage regex still reported a bill")


def test_the_replacement_densities_are_orders_above_the_ert():
    """The prices are the defect, and the gap must stay visible.

    prompt_7 section 5.2c: the ERT's leak rows are 10^3-10^4 too low and three
    components price at exactly 0, because SRAM leakage comes from CACTI pinned
    to `itrs-lstp` while MAC leakage comes from Aladdin's commercial-library
    40 nm data. This asserts the gap rather than the values, so re-pricing
    ECC_LEAKAGE_NW does not break it and QUIETLY REVERTING to the ERT does.
    """
    fp, _ = _biggest_cache()
    raw = _raw_from_cache(fp, limit=6)
    _off, on = _cfgs()
    replacement, _rows = energy.standby_energy(raw, on)
    billed = sum(c["timeloop_leakage_pJ"] for c in energy.standby_components(raw, on))
    assert billed > 0, "Timeloop billed no leakage at all on this cache"
    ratio = replacement / billed
    assert ratio > 100, (
        f"the replacement densities are only {ratio:.1f}x Timeloop's own bill. "
        f"section 5.2c measures 10^3-10^4; if this is now right, the defect is "
        f"fixed and prompt_7 section 5.2 must be restated -- do not relax this "
        f"number to make the test pass")
    print(f"          replacement / Timeloop's own leakage bill = x{ratio:,.0f}")


def test_dram_is_charged_no_standby_and_says_so():
    """The ONE deliberate omission, made explicit rather than left as a zero.

    env.sh section 6 declares no off-chip leakage density, exactly as
    ECC_DRAM_BACKGROUND_PJ and ECC_DRAM_REFRESH_PJ are 0 on purpose. A silent
    zero and a stated omission look identical on a figure; only one of them can
    be audited.
    """
    fp, _ = _biggest_cache()
    raw = _raw_from_cache(fp, limit=4)
    _off, on = _cfgs()
    _total, rows = energy.standby_energy(raw, on)
    dram = [r for r in rows if r["kind"] == "dram"]
    assert dram, "no DRAM level in the record"
    for r in dram:
        assert r["energy_pJ"] == 0.0, "DRAM was charged standby energy"
        assert r.get("note"), "DRAM's zero carries no stated reason"
    # every other level must be charged something, or the knob is a no-op
    charged = [r for r in rows if r["kind"] != "dram" and r["energy_pJ"] > 0]
    assert len(charged) >= 4, (
        f"only {len(charged)} on-chip level(s) were charged standby energy; "
        f"the rest would be silently free")


def test_utilized_instances_not_declared():
    """Standby is billed on UTILIZED instances, as Timeloop bills leakage.

    Timeloop power-gates every unused instance and charges `leak x utilized x
    cycles` (`buffer.cpp FinalizeBufferEnergy`); prompt_6 RULE 3 settled the
    same convention for the reconstruction engines, and the two sides of the
    comparison must not use different ones. The DECLARED count is carried
    beside it and reported, never charged.
    """
    fp, _ = _biggest_cache()
    raw = _raw_from_cache(fp, limit=8)
    _off, on = _cfgs()
    comps = energy.standby_components(raw, on)
    part = [c for c in comps
            if c["declared_instance_cycles"] > c["instance_cycles"] > 0]
    assert part, (
        "no level in this cache has an unused instance, so the utilized/declared "
        "distinction cannot be tested here -- the mapper filled every PE on "
        "every shape, which would itself be news")
    for c in part:
        assert c["instances_declared"] is not None, "the declared count is not reported"
    print(f"          {len(part)} level(s) bill fewer than their declared instances, "
          f"e.g. {part[0]['level']}: {part[0]['instance_cycles']:,.0f} of "
          f"{part[0]['declared_instance_cycles']:,.0f} instance-cycles")

    # BREAKAGE: bill the declared count instead. It must change the answer, or
    # the distinction is not actually being made.
    e_util, _ = energy.standby_energy(raw, on)
    swapped = _FakeRaw([dict(lp, physical={
        "compute_cycles": lp["physical"]["compute_cycles"],
        "levels": [dict(r, utilized=r["instances"]) for r in lp["physical"]["levels"]]})
        for lp in raw.per_layer], cycles=raw.cycles)
    e_decl, _ = energy.standby_energy(swapped, on)
    expect_raises(lambda: _assert_eq(round(e_util, 3), round(e_decl, 3)),
                  "billing declared instances gave the same standby energy as "
                  "billing utilized ones")


def main():
    print("prompt_7 Phase A -- the roofline, the ceiling, and standby energy")
    # A DELIBERATELY COLD CACHE IS NOT A FAILURE, AND IT IS NOT A FALLBACK
    # EITHER. prompt_7 Phase C1 re-fingerprints every arm on purpose, so
    # between C1 and C2 the live reference arm has no solved shape and there is
    # no real data to hold a property against. That is a SKIP, stated loudly.
    # Resolving to some other treatment's cache instead would be the fallback
    # this suite exists to refuse: a different mapspace is a different
    # accelerator, and rule R-2 is false on the retired unconstrained caches.
    fp, n, variant = _reference_cache()
    # AN UNDER-FILLED LIVE CACHE IS A SKIP, NOT A FAILURE -- and it is not a
    # fallback either. `_reference_cache()` resolves THE LIVE CONFIGURATION's
    # treatment by construction, so a shortfall here can only mean "not mapped
    # yet"; it can never be the wrong accelerator, which is the case the
    # refusal below exists for. Between prompt_7 C1 and C2 that is the normal
    # state: C1 re-fingerprints every arm on purpose, and a C1 gate-6 smoke
    # leaves one or two shapes behind.
    if n < MIN_SHAPES:
        print(f"  SKIPPED -- the live reference arm holds {n} solved shape(s), "
              f"fewer than the {MIN_SHAPES} these property tests need:\n"
              f"    {fp}\n"
              f"    treatment {variant}\n"
              f"  Between prompt_7 Phase C1 and C2 that is expected -- C1 moves\n"
              f"  every fingerprint on purpose (its gate 2). Re-run this suite\n"
              f"  after C2 collects:\n"
              f"    ECC_RERUN_OPTIMISER=1 bash hpc/run_all.sh --map-only")
        return 0
    fp, n = _biggest_cache()
    print(f"  cache: {fp} ({n} solved shapes)\n")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED")
        for f in FAILED:
            print(f"  - {f}")
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
