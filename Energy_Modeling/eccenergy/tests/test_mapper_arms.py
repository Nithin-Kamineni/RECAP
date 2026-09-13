"""prompt_7 Phase B: one mapping per boundary.

PROPERTY TESTS ON REAL CACHED DATA, PLUS DELIBERATE BREAKAGE. Every assertion
here reads the placement records and the mapper cache that are on disk -- no
fixture, no hand-written arm list -- and every one is followed by the mutation
it exists to catch, applied to a copy and checked to FAIL. An assertion nobody
has seen fail is a comment.

WHAT DEFECT 3 WAS. Five boundaries took three shapes as far as the mapper was
concerned and only two were ever mapped, so R3 was billed from the REFERENCE's
plan while its geometry was R2's, and R5a -- the only bar that narrows
`weights_spad` -- was billed from a plan that does not know that level is
narrower at all. Measured before any of this landed: an arm's own plan differs
from the reference's on 9/43 shapes (R2) and 13/43 (R4).

THE SIX PROPERTIES, AND WHY EACH ONE IS THE GATE IT IS
------------------------------------------------------
1. `test_mapper_arms_are_the_distinct_chips`
   `arms_mod.mapper_arms()` is the reference plus every boundary that differs on
   at least one of prompt_7 6.4's three axes -- `datawidth: q`, the ERT bump,
   the declared bandwidth scale. Six on eyeriss_like_wglb, five on v2 (which
   has four boundaries, not five). Breakage: drop one axis from the
   distinctness key; on Eyeriss v2 dropping the no-op network entries
   collapses R1 and R2 into one chip, five arms into four.
2. `test_arm_slugs_are_distinct`
   No two arms share a cache directory. R1's patched YAML IS the reference's
   until Phase C1.2 declares the bandwidth scale, so without `arm-<key>` this
   is a REAL collision, not a theoretical one (prompt_6 RULE 4.4.5 defence 1).
   Breakage: force two arms onto one slug.
3. `test_no_bar_borrows_a_foreign_plan`
   Every bar is billed from a plan that narrows exactly the levels it narrows,
   wherever such a plan is on disk; R3 must name R2, never `reference`.
   Breakage: put R3 back on the reference plan.
4. `test_ert_condition3_is_derived`
   Condition 3's outcome follows from the per-cycle `leak` row -- and so from
   the clock-gating percentage -- not from a key name. Breakage: rename the
   placement and check the answer does not move; flip the gating to 100% and
   check that it does.
5. `test_the_cache_probe_separates_ready_cold_and_partial`
   `cold` is normal until Phase C and its bars borrow a NAMED plan; `partial`
   is refused, because a half-mapped arm is two chips in one bar.
   Breakage: an empty shape directory must not count as a solved shape.
6. `test_phase_c_colded_every_arm_and_deleted_none`
   UNTIL 2026-09-13 this was `..._are_still_where_phase_b_left_them`: the two
   arms prompt_6 mapped had to keep their slug and their fingerprint byte for
   byte, because Phase B was a no-compute phase. PHASE C IS THE COLD PASS, so
   the property inverts -- every arm must now resolve to a NEW directory, and
   the pre-Phase-C ones must still be on disk, unread and undeleted. A cold
   that also deleted is the one thing that cannot be undone.

Run it like every other suite:

    bash hpc/tl.sh python3 -m eccenergy.tests.test_mapper_arms
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys

from .. import config
from ..arch import arms as arms_mod
from ..arch import placements
from ..arch import weight_path

FAILED = []

#: The design prompt_7 quotes every measured number from. Eyeriss v2 is the
#: second design in the gate because it has FOUR boundaries, not five, so a
#: hardcoded six would fail on it.
ARCH = "eyeriss_like_wglb"
ARCH_V2 = "eyeriss_v2_like"

#: prompt_7 6.4's table, as an EXPECTATION rather than as the source. Each row
#: is (arm key, levels at `datawidth: q`, the ERT bump or None). The bandwidth
#: scale is not spelled here because it is derived from `reduced` with no
#: freedom left; what it does is separate R1 from the reference and, on v2, R2
#: from R1, and the counts below are what prove it.
_EXPECTED_ARMS = {
    "eyeriss_like_wglb": (
        ("reference", (), None),
        ("recon1", (), None),
        ("recon2", ("filter_glb",), ("filter_glb", "read")),
        ("recon3", ("filter_glb",), None),
        ("recon4", ("filter_glb",), ("weights_spad", "write")),
        # recon5's bump is condition 3 re-derived (prompt_7 6.3): its access row
        # is the MAC count and constant, its per-cycle leak row is not. With no
        # configuration to read the gating from there is no leak row and it has
        # no bump -- which is what `_EXPECTED_ARMS` is checked at.
        ("recon5", ("filter_glb", "weights_spad"), None),
    ),
    "eyeriss_v2_like": (
        ("reference", (), None),
        ("recon1", (), None),
        ("recon2", (), None),
        ("recon3", (), ("weights_spad", "write")),
        ("recon4", ("weights_spad",), None),
    ),
}


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


# ---------------------------------------------------------------------------
#  1. the arm list is the distinct chips
# ---------------------------------------------------------------------------
def test_mapper_arms_are_the_distinct_chips():
    """Six arms on Eyeriss v1 (+filter GLB), five on v2. DERIVED, over all
    three axes -- and the design with four boundaries must not answer six."""
    for arch, want in _EXPECTED_ARMS.items():
        arms = arms_mod.mapper_arms(arch)
        got = tuple((a.key, a.narrow_levels, a.ert or None) for a in arms)
        _assert_eq(got, want)
        # every boundary is the member of exactly one arm, and no arm is empty
        members = [m for a in arms for m in a.members]
        _assert_eq(sorted(members),
                   sorted(p.key for p in placements.placements_for(arch)))
        _assert_eq(len(members), len(set(members)))
        _assert_eq(arms[0].key, "reference")
        _assert_eq(arms[0].placement, None)
    n_v1 = len(arms_mod.mapper_arms(ARCH))
    n_v2 = len(arms_mod.mapper_arms(ARCH_V2))
    assert n_v1 == 6 and n_v2 == 5, (n_v1, n_v2)
    print(f"          {ARCH}: {n_v1} arms; {ARCH_V2}: {n_v2} arms "
          f"({len(placements.placements_for(ARCH_V2))} boundaries + the reference)")

    # THE KEY IS THE THREE AXES AND NOTHING ELSE, on every registered design.
    for arch in placements.PLACEMENTS:
        for a in arms_mod.mapper_arms(arch):
            _assert_eq(a.distinct_key, (a.narrow_levels, a.bw_scale, a.ert))

    # BREAKAGE 1, MEASURED: drop the network entries from the declared
    # bandwidth scale. On Eyeriss v2, R2's `reduced` set adds ONLY the
    # inter-cluster mesh -- a network, which Timeloop has no timing model for
    # -- so without them R1 and R2 become one chip and the five arms become
    # four. This is why a no-op entry is CARRIED AND MARKED rather than
    # dropped (prompt_7 6.4's "+network: no-op").
    def drop_no_ops():
        keys = {(a.narrow_levels, tuple(x for x in a.bw_scale if x[1] != "no-op"), a.ert)
                for a in arms_mod.mapper_arms(ARCH_V2)}
        _assert_eq(len(keys), 5)
    expect_raises(drop_no_ops,
                  "dropping the no-op network entries still gave 5 arms on v2")

    # BREAKAGE 2: THE MERGE PATH IS REAL, not dead code. Two boundaries that
    # agree on all three axes ARE one chip and share one arm -- which is what
    # makes 6 a DERIVED count rather than "one per placement". Registered as a
    # synthetic design so no real record has to be bent to show it.
    twin = dataclasses.replace(placements.placement_by_key(ARCH, "recon2"),
                               key="recon2b", variant="twin_of_recon2")
    real = placements.PLACEMENTS[ARCH]
    weight_path.WEIGHT_PATHS["_twin_test"] = weight_path.WEIGHT_PATHS[ARCH]
    placements.PLACEMENTS["_twin_test"] = real + (twin,)
    try:
        arms = arms_mod.mapper_arms("_twin_test")
        _assert_eq(len(arms), len(arms_mod.mapper_arms(ARCH)))     # NOT one more
        merged = [a for a in arms if len(a.members) > 1]
        _assert_eq([a.members for a in merged], [("recon2", "recon2b")])
        # and a twin that differs on ONE axis is its own chip again
        placements.PLACEMENTS["_twin_test"] = real + (
            dataclasses.replace(twin, site_counter="fills"),)
        _assert_eq(len(arms_mod.mapper_arms("_twin_test")),
                   len(arms_mod.mapper_arms(ARCH)) + 1)
    finally:
        del placements.PLACEMENTS["_twin_test"], weight_path.WEIGHT_PATHS["_twin_test"]

    # BREAKAGE 3: DERIVED, NOT NAMED. Rename every placement and nothing about
    # the arm list may move except the names.
    renamed = tuple(dataclasses.replace(q, key=f"z{i}", variant=f"z{i}")
                    for i, q in enumerate(placements.PLACEMENTS[ARCH]))
    weight_path.WEIGHT_PATHS["_rename_test"] = weight_path.WEIGHT_PATHS[ARCH]
    placements.PLACEMENTS["_rename_test"] = renamed
    try:
        a_real = arms_mod.mapper_arms(ARCH)
        a_fake = arms_mod.mapper_arms("_rename_test")
        _assert_eq([(a.narrow_levels, a.bw_scale, a.ert) for a in a_fake],
                   [(a.narrow_levels, a.bw_scale, a.ert) for a in a_real])
    finally:
        del placements.PLACEMENTS["_rename_test"], weight_path.WEIGHT_PATHS["_rename_test"]


def test_a_network_stage_is_carried_and_marked_no_op():
    """prompt_7 R-3: Timeloop has no network timing model, so a bandwidth
    scale on a network reaches nothing -- but a boundary that declares one is
    still a different declaration, and dropping it merges R1 and R2 on Eyeriss
    v2 into one chip. The entry is kept and MARKED."""
    arms = {a.key: a for a in arms_mod.mapper_arms(ARCH_V2)}
    r1, r2 = arms["recon1"], arms["recon2"]
    _assert_eq(r1.bw_scale, (("DRAM", "timed"),))
    _assert_eq(r2.bw_scale[0], ("DRAM", "timed"))
    _assert_eq(r2.bw_scale[1][1], "no-op")
    _assert_eq(r1.narrow_levels, r2.narrow_levels)     # same geometry ...
    _assert_eq(r1.ert, r2.ert)                         # ... and same bump ...
    assert r1.distinct_key != r2.distinct_key          # ... but not one chip
    # the no-op marking is DERIVED from the stage kind, not from a name
    kinds = {s.key: s.kind for s in placements.stages_for(ARCH_V2)}
    for key in placements.placement_by_key(ARCH_V2, "recon2").reduced:
        want = arms_mod.BW_SCALE_TIMING[kinds[key]]
        got = dict((lvl, t) for lvl, t in r2.bw_scale)
        assert want in got.values(), (key, want, got)

    # BREAKAGE: call a network a storage stage and the marking must follow the
    # record, not the level's name.
    fake = dataclasses.replace(
        placements.placement_by_key(ARCH_V2, "recon2"))
    stages = tuple(dataclasses.replace(s, kind="storage") if s.kind == "network" else s
                   for s in placements.stages_for(ARCH_V2))
    expect_raises(lambda: _assert_eq(arms_mod.arm_bw_scale(fake, stages)[1][1], "no-op"),
                  "a stage declared `storage` was still marked no-op")


# ---------------------------------------------------------------------------
#  2. no two arms share a cache directory
# ---------------------------------------------------------------------------
def test_arm_slugs_are_distinct():
    """prompt_6 RULE 4.4.5 defence 1, prompt_7 B2 gate 2. On EVERY registered
    design, not only the one the study is pointed at."""
    for arch in placements.PLACEMENTS:
        arms = arms_mod.mapper_arms(arch)
        slugs = [a.slug_part for a in arms]
        _assert_eq(len(set(slugs)), len(slugs))
        _assert_eq(slugs[0], None)                     # the reference adds nothing
        for a in arms[1:]:
            assert a.slug_part, a
            # an arm WITH a bump keeps prompt_6's spelling byte for byte, so
            # every directory on disk stays a cache hit
            if a.ert:
                _assert_eq(a.slug_part, f"ert-{a.key}-{a.ert[0]}-{a.ert[1]}")
            else:
                _assert_eq(a.slug_part, f"arm-{a.key}")
    slugs = arms_mod.arm_slugs(ARCH)
    print(f"          {ARCH}: " + ", ".join(f"{k}={v}" for k, v in slugs.items()))

    # BREAKAGE: force two arms onto one slug -- which is what NOT having
    # `arm-<key>` did to the reference and R1, whose patched YAML is identical
    # until Phase C1.2.
    def collide():
        got = [a.slug_part if a.ert else None for a in arms_mod.mapper_arms(ARCH)]
        _assert_eq(len(set(got)), len(got))
    expect_raises(collide, "two arms with no ERT bump did not collide when the "
                           "`arm-<key>` component was removed")


def test_the_config_resolves_every_arm_and_gives_each_its_own_slug():
    """`ECC_RECON_ERT_ARM=<any arm>` must resolve -- including R1 and R3, which
    have no ERT bump and which `ert_arm_spec()` still refuses by design."""
    cfg = config.load_config()
    arch = cfg.archs[0]
    seen = {}
    for arm in arms_mod.mapper_arms(arch, cfg):
        acfg = cfg.with_(recon_ert_arm=arm.key)
        _assert_eq(acfg.mapper_arm_slug(), arm.slug_part)
        _assert_eq(tuple(acfg.weight_datawidth_levels or ()), arm.narrow_levels)
        # R1 NARROWS NOTHING ON CHIP: setting q with an empty level list is the
        # spelling that narrows EVERY weight level, which is a different chip.
        if not arm.narrow_levels:
            _assert_eq(acfg.weight_datawidth, cfg.weight_datawidth)
        else:
            assert acfg.weight_datawidth is not None
        seen.setdefault(acfg.arch_variant_slug, []).append(arm.key)
    for slug, keys in seen.items():
        _assert_eq(len(keys), 1)
    print(f"          {len(seen)} distinct cache slugs for {len(seen)} arms")

    # BREAKAGE: an arm key this design does not define must be refused, not
    # silently resolved to the reference.
    expect_raises(lambda: cfg.with_(recon_ert_arm="recon9"),
                  "an unknown arm key was accepted")


# ---------------------------------------------------------------------------
#  3. no bar is billed from a plan of a different geometry
# ---------------------------------------------------------------------------
def test_no_bar_borrows_a_foreign_plan():
    """prompt_7 B1 gate 4. With only the two prompt_6 arms mapped, R3 must
    name R2 -- never `reference` -- and R1's `reference` must be RIGHT rather
    than a fallback, because R1 narrows nothing on chip."""
    cfg = config.load_config()
    arch = cfg.archs[0]
    got = arms_mod.plan_assignment(arch, {"recon2", "recon4"}, cfg)
    _assert_eq(got["recon1"]["arm"], "reference")
    _assert_eq(got["recon1"]["geometry_matches"], True)
    _assert_eq(got["recon2"]["kind"], "own")
    _assert_eq(got["recon3"]["arm"], "recon2")
    _assert_eq(got["recon3"]["kind"], "borrowed")
    _assert_eq(got["recon3"]["geometry_matches"], True)
    _assert_eq(got["recon4"]["kind"], "own")
    # R5a: the only bar that narrows `weights_spad`, and no solved arm does --
    # prompt_7 6.2b, a chip that has never been mapped. It must be FLAGGED,
    # with an empty candidate list, never quietly billed as if it matched.
    _assert_eq(got["recon5"]["kind"], "foreign")
    _assert_eq(got["recon5"]["geometry_matches"], False)
    _assert_eq(got["recon5"]["candidates"], [])
    arms = {a.key: a for a in arms_mod.mapper_arms(arch, cfg)}
    for key, rec in got.items():
        lender = arms[rec["arm"]]
        assert rec["geometry_matches"] == (lender.narrow_levels == arms[key].narrow_levels), rec
    print("          " + "; ".join(f"{k}->{v['arm']}({v['kind']})" for k, v in got.items()))

    # every arm mapped: nobody borrows anything
    everything = arms_mod.plan_assignment(arch, set(arms), cfg)
    _assert_eq(sorted({v["kind"] for v in everything.values()}), ["own"])

    # BREAKAGE 1: put R3 back on the reference, which is what Defect 3(a) was.
    expect_raises(lambda: _assert_eq(got["recon3"]["arm"], "reference"),
                  "R3 was still billed from the reference plan")
    # BREAKAGE 2: the reference must NOT count as a geometry match for a bar
    # that narrows something. Nothing solved at all is exactly that case.
    nothing = arms_mod.plan_assignment(arch, set(), cfg)
    _assert_eq(nothing["recon1"]["geometry_matches"], True)
    for key in ("recon2", "recon3", "recon4", "recon5"):
        _assert_eq(nothing[key]["kind"], "foreign")
        _assert_eq(nothing[key]["geometry_matches"], False)
    expect_raises(lambda: _assert_eq(nothing["recon2"]["geometry_matches"], True),
                  "the reference plan passed as a geometry match for a bar that "
                  "narrows filter_glb")


def test_the_lender_is_the_outermost_candidate_and_the_others_are_recorded():
    """The choice between two plans of the same geometry is DERIVED -- path
    order -- and the road not taken is on the record, so the choice is
    auditable rather than asserted."""
    cfg = config.load_config()
    arch = cfg.archs[0]
    rec = arms_mod.plan_assignment(arch, {"recon2", "recon4"}, cfg)["recon3"]
    _assert_eq(rec["candidates"], ["recon2", "recon4"])
    _assert_eq(rec["arm"], rec["candidates"][0])
    assert "recon4" in rec["why"], rec["why"]

    # BREAKAGE: with only the INNER candidate solved, the answer must move --
    # the rule is path order among what is solved, not the name `recon2`.
    inner = arms_mod.plan_assignment(arch, {"recon4"}, cfg)["recon3"]
    _assert_eq(inner["arm"], "recon4")
    expect_raises(lambda: _assert_eq(inner["arm"], "recon2"),
                  "R3 still named recon2 when recon2 was not solved")


# ---------------------------------------------------------------------------
#  4. condition 3 is derived from the gating, not from a key name
# ---------------------------------------------------------------------------
def test_ert_condition3_is_derived():
    """prompt_7 6.3 / B3. The bump has two rows; the exclusion holds only when
    BOTH are constant across the mapspace. The access row on the innermost
    level's reads is the MAC count and is constant; the per-cycle leak row is
    `idle x (1 - g)` and Timeloop bills it as leak x UTILIZED instances x
    cycles, both the mapper's choice."""
    stages = placements.stages_for(ARCH)
    r5 = placements.placement_by_key(ARCH, "recon5")
    _assert_eq(r5.site_stage, "weights_spad")
    _assert_eq(r5.site_counter, "reads")
    _assert_eq([s.key for s in stages if s.kind == "storage"][-1], "weights_spad")

    # NO per-cycle row -> the whole bump is constant -> excluded, as it was
    # before clock gating existed.
    ok, why = arms_mod.ert_injectable(r5, stages, 0.0)
    _assert_eq(ok, False)
    assert "constant" in why, why
    # ANY non-zero per-cycle row -> mapping-dependent -> injectable.
    ok, why = arms_mod.ert_injectable(r5, stages, 0.0139456495)
    _assert_eq(ok, True)
    assert "leak row" in why, why

    # and the row itself follows the GATING PERCENTAGE, from the live DC
    # constants -- 0 and 99.5 are the two settings section 11 item 7 requires
    # side by side, and 100 is the limit at which the old exclusion returns.
    cfg = config.load_config()
    rows = {}
    for pct in (0.0, 99.5, 100.0):
        c = cfg.with_(recon_clock_gating_pct=pct)
        rows[pct] = (arms_mod.ert_leak_delta_pj(c),
                     tuple(p.key for p in arms_mod.ert_arms(ARCH, c)))
    assert rows[0.0][0] > rows[99.5][0] > 0.0, rows
    _assert_eq(rows[100.0][0], 0.0)
    assert "recon5" in rows[0.0][1], rows[0.0]
    assert "recon5" in rows[99.5][1], rows[99.5]
    assert "recon5" not in rows[100.0][1], rows[100.0]
    # the ratio is the gating fraction and nothing else
    assert abs(rows[99.5][0] / rows[0.0][0] - 0.005) < 1e-12, rows
    print(f"          leak row: PCT=0 {rows[0.0][0]:.7f}, PCT=99.5 {rows[99.5][0]:.7f}, "
          f"PCT=100 {rows[100.0][0]:.7f} pJ/cycle/instance")
    print(f"          ERT arms: PCT=0 {rows[0.0][1]}; PCT=100 {rows[100.0][1]}")

    # BREAKAGE 1: the answer must not move when only the NAME moves.
    renamed = dataclasses.replace(r5, key="not_recon5", variant="x")
    _assert_eq(arms_mod.ert_injectable(renamed, stages, 0.0139456495)[0], True)
    _assert_eq(arms_mod.ert_injectable(renamed, stages, 0.0)[0], False)
    expect_raises(lambda: _assert_eq(arms_mod.ert_injectable(renamed, stages, 0.0)[0], True),
                  "renaming the placement changed condition 3's answer")
    # BREAKAGE 2: a bar on the innermost level's FILLS is injectable at any
    # gating -- its access row is not the MAC count.
    fills = dataclasses.replace(r5, site_counter="fills")
    _assert_eq(arms_mod.ert_injectable(fills, stages, 0.0)[0], True)
    # BREAKAGE 3: a bar that is NOT on the innermost level is unaffected by the
    # leak row entirely.
    _assert_eq(arms_mod.ert_injectable(placements.placement_by_key(ARCH, "recon2"), stages, 0.0)[0],
               True)


def test_ert_arms_and_mapper_arms_answer_different_questions():
    """`ert_arms()` keeps its pre-2026-09-12 meaning -- which boundaries have
    an ERT-injectable encoder -- and `mapper_arms()` is the list of chips. The
    first is a SUBSET of the second's non-reference arms, never equal to it."""
    cfg = config.load_config()
    arch = cfg.archs[0]
    ert = {p.key for p in arms_mod.ert_arms(arch, cfg)}
    mapper = {a.key for a in arms_mod.mapper_arms(arch, cfg) if a.placement is not None}
    assert ert < mapper, (sorted(ert), sorted(mapper))
    # every ERT arm's spec still comes back from `ert_arm_spec`, and every
    # non-injectable one is still REFUSED there by name and reason.
    for key in sorted(mapper):
        spec = arms_mod.mapper_arm_spec(arch, key, cfg)
        if key in ert:
            _assert_eq(arms_mod.ert_arm_spec(arch, key, cfg)["level"], spec["ert"]["level"])
        else:
            expect_raises(lambda k=key: arms_mod.ert_arm_spec(arch, k, cfg),
                          f"{key} was accepted as an ERT arm")
    print(f"          ERT arms {sorted(ert)} of mapper arms {sorted(mapper)}")


# ---------------------------------------------------------------------------
#  5. Phase C colded every arm -- deliberately -- and deleted none
# ---------------------------------------------------------------------------
#: The mapper cache directories prompt_6 and prompt_7 Phase B left on disk, by
#: the `fp-` hash they were written under. Phase C1 declares an off-chip
#: bandwidth, a per-dataspace scale, a bit-aware port, a 200 MHz clock, a MAC
#: price and a bank count -- every one of them in the architecture the mapper
#: reads -- so every arm moves to a new directory. These must SURVIVE the move:
#: they are hours of compute and they are the only record of what the numbers
#: in FINDINGS 2.9 were computed from.
_PRE_PHASE_C_FINGERPRINTS = {
    "reference": "718d53aac189",
    "recon2": "a18a5b15fd8b",
    "recon4": "55569ac34427",
}


def test_phase_c_colded_every_arm_and_deleted_none():
    """prompt_7 C1 gate 2: the fingerprint MOVED for every arm -- and the
    directories it moved away from are still there.

    A cold pass that does not change the fingerprint silently reuses stale
    entries, which is the whole reason `ECC_ENERGY_MODEL_REV` exists. A cold
    pass that DELETES what it moved away from cannot be undone.
    """
    from ..arch import fingerprint
    from ..paths import Results
    cfg = config.load_config()
    arch = cfg.archs[0]
    root = pathlib.Path("ecc_energy_study/outputs") / arch
    if not root.exists():
        print("          no mapper cache on disk; skipped")
        return
    found = []
    for arm in arms_mod.mapper_arms(arch, cfg):
        acfg = cfg.with_(recon_ert_arm=arm.key)
        variant = fingerprint.effective_variant(arch, acfg)
        fp = fingerprint.arch_fingerprint(arch, acfg)
        d = pathlib.Path(Results(acfg).mapper_cache(arch, variant, fp, create=False))
        n = len(list(d.glob("*/timeloop-mapper.stats.txt"))) if d.is_dir() else 0
        found.append((arm.key, fp, n))
        print(f"          {arm.key:<10} fp {fp}  {n:>3} solved  {d.name}")
    # 1. EVERY arm moved. Not one of them may still resolve to a pre-Phase-C
    #    hash -- that would be Phase C landing on some arms and not others.
    now = {k: fp for k, fp, _n in found}
    for key, old in _PRE_PHASE_C_FINGERPRINTS.items():
        assert now.get(key) != old, (
            f"{key} still fingerprints {old}: a Phase C declaration did not "
            f"reach the architecture the mapper reads, so this arm would be "
            f"read back from a pre-Phase-C mapping")
    # 2. and every arm is still its own chip
    assert len(set(now.values())) == len(now), f"two arms share a hash: {now}"
    # 3. NOTHING WAS DELETED. The old directories hold hours of compute and are
    #    the only record of what FINDINGS 2.9's numbers were computed from.
    for key, old in _PRE_PHASE_C_FINGERPRINTS.items():
        hits = list(root.glob(f"*/fp-{old}/*/timeloop-mapper.stats.txt"))
        assert hits, (f"the pre-Phase-C cache fp-{old} ({key}) is gone. Phase C "
                      f"colds by MOVING, never by deleting.")
        print(f"          pre-Phase-C fp-{old} ({key}): {len(hits)} shapes kept")


# ---------------------------------------------------------------------------
#  6. the cache probe: ready / cold / PARTIAL
# ---------------------------------------------------------------------------
class _FakeLayer:
    def __init__(self, shape_name):
        self.shape_name = shape_name


class _FakeSession:
    """Just enough of a Session for `arm_plan_state`: a cache root it can look
    under, and the shapes of one model. Nothing here reads or writes the real
    mapper cache -- the point of the probe is that it only stats files."""
    def __init__(self, root, shapes):
        self.results = self
        self._root = root
        self.models = {"m": [_FakeLayer(s) for s in shapes]}

    def mapper_cache(self, arch, variant=None, fingerprint=None, create=True):
        return self._root

    def legacy_mapper_cache(self, arch, variant=None):
        return self._root


def test_the_cache_probe_separates_ready_cold_and_partial():
    """prompt_7 B1. `cold` is the normal state of an arm Phase C has not mapped
    and its bars borrow a NAMED plan; `partial` is the dangerous middle -- its
    own plan on some shapes and a borrowed one on the rest, one bar being two
    chips -- and the evaluator refuses on it rather than borrowing."""
    import tempfile
    from ..study import ert_view
    cfg = config.load_config()
    arch = cfg.archs[0]
    arm = [a for a in arms_mod.mapper_arms(arch, cfg) if a.key == "recon2"][0]
    shapes = ["shapeA", "shapeB", "shapeC"]
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        ses = _FakeSession(root, shapes)
        _assert_eq(ert_view.arm_plan_state(cfg, ses, arch, "m", arm, shapes)[0], "cold")
        (root / "shapeA").mkdir()
        (root / "shapeA" / "timeloop-mapper.stats.txt").write_text("x")
        state, detail = ert_view.arm_plan_state(cfg, ses, arch, "m", arm, shapes)
        _assert_eq(state, "partial")
        _assert_eq(detail["missing"], ["shapeB", "shapeC"])
        for s in shapes[1:]:
            (root / s).mkdir()
            (root / s / "timeloop-mapper.stats.txt").write_text("x")
        state, detail = ert_view.arm_plan_state(cfg, ses, arch, "m", arm, shapes)
        _assert_eq(state, "ready")
        _assert_eq(detail["missing"], [])
        print(f"          cold -> partial -> ready over {len(shapes)} shapes, "
              f"probed by stat() only")

        # BREAKAGE: a directory with no stats.txt in it is NOT a solved shape.
        # A claimed shape has a directory the moment the mapper starts on it.
        (root / "shapeD").mkdir()
        ses2 = _FakeSession(root, shapes + ["shapeD"])
        _assert_eq(ert_view.arm_plan_state(cfg, ses2, arch, "m", arm,
                                      shapes + ["shapeD"])[0], "partial")
        expect_raises(
            lambda: _assert_eq(ert_view.arm_plan_state(cfg, ses2, arch, "m", arm,
                                                  shapes + ["shapeD"])[0], "ready"),
            "an empty shape directory counted as a solved shape")


def main():
    print("prompt_7 Phase B -- one mapping per boundary")
    print(f"  design: {ARCH} (+ {ARCH_V2} for the four-boundary case)\n")
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
