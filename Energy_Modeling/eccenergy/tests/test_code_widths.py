"""PROMPT_2's WIDTH TABLE: does every BCH configuration actually run, and does
it run on the silicon prompt_2 specifies?

Every assertion here has a DELIBERATE BREAKAGE beside it -- the mutation the
assertion exists to catch, applied to a copy of the real table or the real
config, checked to fail. An assertion nobody has seen fail is a comment.

THE PROPERTY UNDER TEST, STATED THE RIGHT WAY ROUND
---------------------------------------------------
`timeloop-mapper` asserts `width % (word_bits * block_size) == 0`
(`buffer.cpp:302`) with `block_size` defaulting to 1 and NO floor path. That
constraint is PER LEVEL, PER MAPPER RUN, and one mapper run maps ONE ARM:

    a level storing q-bit weights needs  width % q == 0        AND NOTHING ELSE

There is NO cross-arm constraint. BCH(63,39)'s arm runs at width 95 because
95 % 5 == 0; `95 % 8 = 7` is irrelevant, because the baseline/embedded arm is
never mapped at width 95 -- it is mapped at 96, where 96 % 8 == 0. The suite
below asserts that non-constraint explicitly
(`test_an_arm_width_need_not_divide_another_arms_datawidth`) so the
`lcm(q, 8)` misreading of 2026-09-11/12 cannot come back silently.

Run it like every other suite:

    bash hpc/tl.sh python3 -m eccenergy.tests.test_code_widths
"""
from __future__ import annotations

import dataclasses
import io
import contextlib
import os
import re
import sys

from .. import config
from ..arch import fingerprint
from ..arch import patch
from ..physics import widths

FAILED = []


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


#: prompt_2.md's WIDTH TABLE, transcribed. The suite asserts the code against
#: THIS, not against itself. (q, spad width, weights/word, eff. capacity)
PROMPT_2_TABLE = [
    ("Baseline / Embedded", 8, 96, 12, 1.0000),
    ("BCH(63,57)",          7, 98, 14, 1.1667),
    ("BCH(63,45)",          6, 96, 16, 1.3333),
    ("BCH(63,39)",          5, 95, 19, 1.5833),
    ("BCH(63,30)",          4, 96, 24, 2.0000),
]
_P2_ARCH = "eyeriss_like_wglb"


# ---------------------------------------------------------------- the q column
def test_q_is_round_not_ceil():
    """prompt_2's 'q declared' column, exactly."""
    import math
    want = {57: 7, 51: 6, 45: 6, 39: 5, 36: 5, 30: 4}
    got = {k: widths.declared_datawidth(63, k) for k in want}
    assert got == want, f"q column moved: {got} != {want}"
    # BREAKAGE: ceil, which is what hpc/map_depth_sweep.sh used to do. It agrees
    # everywhere except BCH(63,57), where it returns 8 -- the embedded arm's own
    # datawidth, so the reconstruction arm would be a silent no-op.
    ceil_q = {k: math.ceil(8 * k / 63) for k in want}
    assert ceil_q[57] == 8 and want[57] == 7, "the ceil/round divergence is gone"
    expect_raises(lambda: _assert_eq(ceil_q, want),
                  "ceil and round must NOT agree at BCH(63,57)")


def test_q_never_exceeds_the_payload_and_never_hits_zero():
    assert widths.declared_datawidth(63, 62) <= 8
    assert widths.declared_datawidth(63, 1) >= 1
    expect_raises(lambda: widths.declared_datawidth(63, 63),
                  "K == N accepted as a code")
    expect_raises(lambda: widths.declared_datawidth(63, 0),
                  "K == 0 accepted as a code")


# ------------------------------------------------------- prompt_2's own table
def test_the_widths_are_prompt_2s_widths():
    """Transcribed from prompt_2.md and compared value by value."""
    for label, q, width, per_word, _cap in PROMPT_2_TABLE:
        got = widths.declared_width(q)
        assert got == width, f"{label}: q={q} gave width {got}, prompt_2 says {width}"
        assert got // q == per_word, (
            f"{label}: {got}/{q} = {got // q} weights per word, "
            f"prompt_2 says {per_word}")
    # BREAKAGE: the withdrawn lcm(q, 8) scheme. It must not reproduce the table.
    import math
    lcm = {q: q * 8 // math.gcd(q, 8) for _l, q, _w, _p, _c in PROMPT_2_TABLE}
    assert lcm[7] == 56 and lcm[5] == 40, "the lcm scheme changed shape"
    expect_raises(
        lambda: _assert_eq([lcm[q] for _l, q, _w, _p, _c in PROMPT_2_TABLE],
                           [w for _l, _q, w, _p, _c in PROMPT_2_TABLE]),
        "lcm(q, 8) must NOT agree with prompt_2's WIDTH TABLE")


def test_an_arm_width_need_not_divide_another_arms_datawidth():
    """THE MISREADING THIS SUITE EXISTS TO PREVENT, asserted as a property.

    prompt_2's 98 and 95 do NOT divide 8, and that is FINE: the 8-bit arm is
    never mapped on them. What each width must divide is its OWN q, and every
    one does. Stated as an assertion so a future reader who "fixes" 95 has to
    delete a test that says, in words, why it is not broken.
    """
    for label, q, width, _p, _c in PROMPT_2_TABLE:
        assert width % q == 0, f"{label}: width {width} % q {q} != 0 -- REAL breakage"
    assert 98 % 8 == 2 and 95 % 8 == 7, "the arithmetic moved"
    # ... and the table keeps them anyway, because the constraint is per arm.
    assert widths.declared_width(7) == 98
    assert widths.declared_width(5) == 95
    # BREAKAGE: requiring a width to suit the OTHER arm's datawidth is what
    # rejects 98 and 95. If this ever stops failing, the rule came back.
    def _demand_cross_arm_legality():
        for _l, q, width, _p, _c in PROMPT_2_TABLE:
            assert width % 8 == 0, f"width {width} does not divide 8"
    expect_raises(_demand_cross_arm_legality,
                  "cross-arm width legality must NOT hold -- it is not a rule")


def test_the_rule_reproduces_the_table_without_reading_it():
    """`WIDTH_TABLE` is a record of a decision, not a second source of truth:
    the rule alone must give the same numbers."""
    for label, q, width, _p, _c in PROMPT_2_TABLE:
        got = widths.nearest_multiple(q, widths.BASE_WIDTH)
        assert got == width, (
            f"{label}: the rule gives {got}, the table says {width} -- one of "
            f"them is wrong and the dict must never win silently")
    # BREAKAGE: a table entry the rule contradicts is still returned (a
    # hand-picked width is allowed), but one its own q does not divide raises.
    saved = dict(widths.WIDTH_TABLE)
    try:
        widths.WIDTH_TABLE[5] = 97          # 97 % 5 = 2
        expect_raises(lambda: widths.declared_width(5),
                      "a table width its own q does not divide was accepted")
    finally:
        widths.WIDTH_TABLE.clear()
        widths.WIDTH_TABLE.update(saved)


def test_every_q_from_1_to_the_payload_has_a_legal_width():
    """No configuration can reach buffer.cpp:302, for any code at any N."""
    for q in range(1, 9):
        w = widths.declared_width(q)
        assert w % q == 0, f"q={q}: width {w} leaves {w % q}"
        assert widths.level_width(q, False, 4) % q == 0, f"q={q}: GLB width"


def test_the_glb_is_the_multiple_and_keeps_divisibility():
    for q in range(1, 9):
        spad = widths.level_width(q, True, 4)
        glb = widths.level_width(q, False, 4)
        assert glb == spad * 4, f"q={q}: GLB {glb} != 4 x {spad}"
        assert glb % q == 0
    # BREAKAGE: a GLB width that is not a multiple of the scratchpad's breaks
    # the "q | W implies q | mW" argument for a non-integer multiplier.
    expect_raises(
        lambda: _assert_eq(widths.level_width(7, False, 4), 98 * 3),
        "the GLB multiplier is 4, not 3")


# ------------------------------------------------- depth is SHARED, width is not
def test_depth_is_computed_at_the_base_width_so_every_arm_shares_it():
    """The arms differ in `width` and `datawidth`. They must NOT differ in
    `depth`, or Accelergy prices silicon one arm does not have."""
    for _l, q, _w, _p, _c in PROMPT_2_TABLE:
        spad = widths.renormalised_depth(224, 16, True)      # prompt_2's own
        glb = widths.renormalised_depth(1024, 64, False, 4)
        assert spad == 37, f"prompt_2 says weights_spad depth 37 at width 96, got {spad}"
        assert glb == 171, f"prompt_2 says filter_glb depth 171 at width 384, got {glb}"
        # and it does not depend on q at all -- that is the point
        assert widths.renormalised_depth(224, 16, True) == spad
    # BREAKAGE: computing depth at the ARM's width makes it q-dependent, and
    # the two arms then declare different depths (42 vs 43 on the real GLB).
    def _depth_at_arm_width(q):
        return max(1, int((256 * 64 / (widths.declared_width(q) * 4)) + 0.5))
    assert _depth_at_arm_width(8) != _depth_at_arm_width(7), \
        "the per-arm depth bug is no longer reachable to demonstrate"
    expect_raises(
        lambda: _assert_eq(_depth_at_arm_width(8), _depth_at_arm_width(7)),
        "depth computed at the arm's own width must NOT agree across arms")


def test_the_reshape_holds_the_published_total_bits():
    """A width change reshapes the word; it must not resize the array."""
    for depth, width, is_spad in ((224, 16, True), (1024, 64, False),
                                  (16, 16, True), (256, 64, False)):
        nd = widths.renormalised_depth(depth, width, is_spad, 4)
        base = widths.base_width() * (1 if is_spad else 4)
        before, after = depth * width, nd * base
        assert abs(after - before) <= base, (
            f"{depth}x{width}b -> {nd}x{base}b: {before:,} bits became "
            f"{after:,}, which is more than one word of rounding")


# ------------------------------------------------------ the real architecture
def _geom(k, arm, **env):
    saved = dict(os.environ)
    try:
        os.environ["ECC_CODE_N"] = "63"
        os.environ["ECC_ARCH_FIDELITY"] = "paper"
        os.environ["ECC_KS"] = os.environ["ECC_CONST_K"] = str(k)
        os.environ["ECC_RECON_K"] = str(k)
        os.environ["ECC_RECON_ERT_ARM"] = "" if arm == "reference" else arm
        os.environ.update({k2: str(v) for k2, v in env.items()})
        cfg = config.load_config()
        with contextlib.redirect_stdout(io.StringIO()):
            return cfg, patch.patched_weight_geometry(_P2_ARCH, cfg)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_every_code_and_every_ert_arm_patches_the_real_yaml():
    """THE REGRESSION. `ECC_RERUN_OPTIMISER=1 ECC_RECON_ERT_AWARE=1 bash
    hpc/map_ert_arms.sh` at ECC_RECON_K=39 failed all 24 arm jobs on
    `filter_glb: width 64 % datawidth 5 != 0`. Every code, every arm, no knob
    set: it must patch, and the width must divide the datawidth on every level.
    """
    for k in (57, 51, 45, 39, 36, 30):
        for arm in ("reference", "recon2", "recon4"):
            try:
                _cfg, g = _geom(k, arm)
            except Exception as exc:               # noqa: BLE001 -- the report
                raise AssertionError(
                    f"BCH(63,{k}) {arm} does not patch: "
                    f"{type(exc).__name__}: {str(exc).splitlines()[0]}")
            for level, v in g.items():
                assert v["width"] % v["datawidth"] == 0, (
                    f"BCH(63,{k}) {arm} {level}: width {v['width']} % "
                    f"datawidth {v['datawidth']} = "
                    f"{v['width'] % v['datawidth']} -- timeloop-mapper aborts")


def test_the_eight_bit_arm_is_width_96_at_every_code():
    """ONE ARM, MAPPED ONCE. The reference must not move between codes -- that
    is what produced BCH(63,39)'s spurious 37.69 % (FINDINGS 2.4b)."""
    seen = {}
    for k in (57, 51, 45, 39, 36, 30):
        _cfg, g = _geom(k, "reference")
        seen[k] = {lvl: (v["width"], v["depth"], v["datawidth"])
                   for lvl, v in g.items()}
        for lvl, v in g.items():
            assert v["datawidth"] == 8, f"BCH(63,{k}) reference {lvl} is not 8-bit"
    first = seen[57]
    for k, got in seen.items():
        assert got == first, (
            f"the 8-bit reference arm differs at BCH(63,{k}): {got} != {first}. "
            f"It must be ONE arm at every code, or its tiling moves with the "
            f"code and the margin measures the reference, not the treatment.")
    spad = [v for lvl, v in first.items() if v[0] == widths.BASE_WIDTH]
    assert spad, f"no level at the base width {widths.BASE_WIDTH}: {first}"


def test_capacity_ratio_reproduces_prompt_2s_eff_capacity_column():
    """The table's own cross-check: `eff. capacity` = (W_arm/q) / (96/8).
    It only reconciles if the arms declare DIFFERENT widths at a COMMON depth,
    so reproducing it is what proves the scheme is prompt_2's."""
    want = {57: 1.1667, 45: 1.3333, 39: 1.5833, 30: 2.0000}
    for k, cap in want.items():
        ref, gr = _geom(k, "reference")
        arm, ga = _geom(k, "recon2")
        info = patch.assert_pair_geometry(_P2_ARCH, ref, arm,
                                          ref_name="embedded", arm_name="recon")
        narrowed = [lvl for lvl, v in info.items()
                    if v["recon"] != v["embedded"]]
        assert narrowed, f"BCH(63,{k}): recon2 narrowed no level"
        for lvl in narrowed:
            got = info[lvl]["capacity_ratio"]
            assert abs(got - cap) < 5e-4, (
                f"BCH(63,{k}) {lvl}: capacity {got:.4f}x, prompt_2's "
                f"eff. capacity column says {cap:.4f}x")
            assert got >= 1.0, "a reduced word cannot hold FEWER weights"


# ---------------------------------------------- assert_pair_geometry's scope
def test_pair_geometry_ignores_width_and_datawidth_and_checks_depth():
    """It must PASS on arms whose widths differ (96 vs 95) -- that is the
    treatment -- and FAIL on arms whose depths differ."""
    ref, _ = _geom(39, "reference")
    arm, ga = _geom(39, "recon2")
    info = patch.assert_pair_geometry(_P2_ARCH, ref, arm)       # must not raise
    widths = {lvl: (v["embedded_width"], v["recon_width"]) for lvl, v in info.items()}
    assert any(a != b for a, b in widths.values()), (
        f"BCH(63,39)'s arms declare identical widths {widths} -- then this test "
        f"is not exercising the thing it claims to")
    # BREAKAGE: a DEPTH difference must still stop the run.
    deeper = arm.with_(weight_depth_scale=0.5)
    expect_raises(lambda: patch.assert_pair_geometry(_P2_ARCH, ref, deeper),
                  "a depth difference between the arms was accepted")
    # ... unless the study asked for it.
    allowed = deeper.with_(disable_pair_geometry_assert=True)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        patch.assert_pair_geometry(_P2_ARCH, ref, allowed)
    assert "ECC_DISABLE_ASSERT_PAIR_GEOMETRY" in buf.getvalue(), (
        "the disabled check must SAY it let a depth difference through, "
        f"got {buf.getvalue()!r}")


def test_the_disable_knob_ships_off():
    """It is for a deliberate depth-variation study, not a default."""
    from ..paths import ROOT
    text = (ROOT / "env.sh").read_text()
    m = re.search(r'^: "\$\{ECC_DISABLE_ASSERT_PAIR_GEOMETRY:=([^}]*)\}"',
                  text, re.M)
    assert m, "env.sh declares no ECC_DISABLE_ASSERT_PAIR_GEOMETRY"
    assert m.group(1).strip() in ("0", "false", "False"), (
        f"env.sh ships ECC_DISABLE_ASSERT_PAIR_GEOMETRY={m.group(1)!r}; a depth "
        f"difference nobody asked for must stop the run")
    assert config.load_config().disable_pair_geometry_assert is False


def test_there_is_no_width_knob_left_to_get_wrong():
    """THE WIDTH TABLE is automatic and unconditional. A knob is what let the
    lookup ship on 2026-09-11 and go unused until it broke a wave on
    2026-09-12."""
    from ..paths import ROOT
    text = (ROOT / "env.sh").read_text()
    assert not re.search(r'^: "\$\{ECC_WEIGHT_WIDTH:=', text, re.M), (
        "env.sh declares ECC_WEIGHT_WIDTH again. The width is derived from the "
        "datawidth each level stores; a knob can only disagree with it.")
    assert not hasattr(config.load_config(), "weight_width"), \
        "Config still carries a weight_width field"


# ------------------------------------------------------- the slug and caches
def test_the_two_variant_slugs_agree():
    """`config.arch_variant_slug` and `fingerprint.effective_variant()` name ONE
    chip; the `mcons` comment in config.py records what a split costs."""
    for k in (57, 45, 39, 30):
        for arm in ("reference", "recon2"):
            cfg, _ = _geom(k, arm)
            with contextlib.redirect_stdout(io.StringIO()):
                eff = fingerprint.effective_variant(_P2_ARCH, cfg)
            assert eff == cfg.arch_variant_slug, (
                f"BCH(63,{k}) {arm}: mapper cache {eff!r}, configuration "
                f"{cfg.arch_variant_slug!r} -- one chip, two directory names")
            assert f"wt{widths.BASE_WIDTH}" in eff, (
                f"the slug does not record THE WIDTH TABLE: {eff}")


def test_the_arms_of_one_code_never_share_a_cache():
    """recon2 and recon4 declare byte-identical YAML and differ only in the
    ERT; the reference differs in datawidth. All three must be separate."""
    seen = {}
    for arm in ("reference", "recon2", "recon4"):
        cfg, _ = _geom(39, arm)
        with contextlib.redirect_stdout(io.StringIO()):
            seen[arm] = (fingerprint.effective_variant(_P2_ARCH, cfg),
                         fingerprint.arch_fingerprint(_P2_ARCH, cfg))
    assert len(set(seen.values())) == 3, f"two arms share a cache: {seen}"


def test_claude_md_still_carries_the_protected_width_section():
    """CLAUDE.md's WIDTH TABLE section is marked PROTECTED; this is what makes
    that marker mean something.

    The rule has been re-derived wrongly in five separate sessions, each time
    costing a debugging session and a wave of compute, so the section is not
    ordinary prose that a "condense this file" pass may drop. If it has to go,
    the user says so -- not a test, and not a summariser.
    """
    from ..paths import ROOT
    text = (ROOT / "CLAUDE.md").read_text()
    for needle, why in (
            ("<!-- PROTECTED -->",
             "the PROTECTED marker itself"),
            ("THE ARMS DO NOT SHARE A DECLARED WIDTH",
             "the rule, stated the right way round"),
            ("IS IRRELEVANT",
             "the `95 % 8 = 7 IS IRRELEVANT` line that stops the next re-derivation"),
            ("384 % 8 == 0",
             "the per-level table of what the mapper actually sees"),
            ("380 % 5 == 0",
             "BCH(63,39)'s row -- the one that keeps being 'fixed'"),
            ("ASK THE USER FIRST",
             "the instruction not to cut a PROTECTED section unilaterally")):
        assert needle in text, (
            f"CLAUDE.md no longer contains {needle!r} ({why}). If this was a "
            f"deliberate edit, it needed the user's agreement first -- see "
            f"'PROTECTED SECTIONS' at the top of CLAUDE.md.")
    # The section must still carry the table, not just a pointer to it.
    sec = text.split("<!-- PROTECTED -->", 1)[1]
    rows = [q for q in ("384 % 8 == 0", "392 % 7 == 0", "384 % 6 == 0",
                        "380 % 5 == 0", "384 % 4 == 0") if q in sec]
    assert len(rows) == 5, (
        f"the protected section has {len(rows)} of the 5 arm rows; a summary "
        f"that keeps the prose and drops the numbers is the failure mode this "
        f"test exists for")


def main():
    print("THE WIDTH TABLE -- eccenergy/widths.py")
    print(widths.audit())
    print()
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
