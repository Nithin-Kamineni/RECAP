"""THE WIDTH TABLE: does every BCH configuration actually run?

Every assertion here has a DELIBERATE BREAKAGE beside it -- the mutation the
assertion exists to catch, applied to a copy of the real table or the real
config, checked to fail. An assertion nobody has seen fail is a comment.

The property under test is the one `timeloop-mapper` enforces with an abort:

    W % q == 0                  or the RECONSTRUCTION arm dies (buffer.cpp:302)
    W % ECC_WEIGHT_BITS == 0    or the BASELINE/EMBEDDED arm dies

Run it like every other suite:

    bash hpc/tl.sh python3 -m eccenergy.tests.test_code_widths
"""
from __future__ import annotations

import dataclasses
import math
import os
import re
import sys

from .. import archs, code_widths, config

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


# ---------------------------------------------------------------- the q column
def test_q_is_round_not_ceil():
    """prompt_2's 'q declared' column, exactly."""
    want = {57: 7, 51: 6, 45: 6, 39: 5, 36: 5, 30: 4}
    got = {k: code_widths.declared_datawidth(63, k) for k in want}
    assert got == want, f"q column moved: {got} != {want}"
    # BREAKAGE: ceil, which is what hpc/map_depth_sweep.sh used to do. It agrees
    # everywhere except BCH(63,57), where it returns 8 -- the embedded arm's own
    # datawidth, so the reconstruction arm would be a silent no-op.
    ceil_q = {k: math.ceil(8 * k / 63) for k in want}
    assert ceil_q[57] == 8 and want[57] == 7, "the ceil/round divergence is gone"
    expect_raises(
        lambda: _assert_eq(ceil_q, want),
        "ceil and round must NOT agree at BCH(63,57)")


def _assert_eq(a, b):
    assert a == b, f"{a} != {b}"


def test_q_never_exceeds_the_payload_and_never_hits_zero():
    assert code_widths.declared_datawidth(63, 62) <= 8
    assert code_widths.declared_datawidth(63, 1) >= 1
    # BREAKAGE: a code with K >= N is not a rate and must not be quantised.
    expect_raises(lambda: code_widths.declared_datawidth(63, 63),
                  "K == N accepted as a code")
    expect_raises(lambda: code_widths.declared_datawidth(63, 0),
                  "K == 0 accepted as a code")


# ------------------------------------------------------- the table's own rule
def test_every_table_width_suits_BOTH_arms():
    """The whole point: no entry may abort either arm."""
    for (n, k), w in code_widths.WIDTH_TABLE.items():
        q = code_widths.declared_datawidth(n, k)
        if w is None:
            assert 8 % q == 0, (
                f"WIDTH_TABLE[({n},{k})] is None but q={q} does not divide 8, "
                f"so the published widths cannot all admit it")
            continue
        assert w % q == 0, f"BCH({n},{k}): recon arm aborts, {w} % {q} != 0"
        assert w % 8 == 0, f"BCH({n},{k}): embedded arm aborts, {w} % 8 != 0"


def test_prompt_2s_two_illegal_rows_are_rejected_not_inherited():
    """98 (q=7) and 95 (q=5) divide their q and NOT 8. If the table ever adopts
    them verbatim, `declared_width` must refuse rather than pass them on."""
    for k, bad in ((57, 98), (39, 95)):
        q = code_widths.declared_datawidth(63, k)
        assert bad % q == 0, f"prompt_2's {bad} should divide q={q}"
        assert bad % 8 != 0, f"prompt_2's {bad} should FAIL the 8-bit arm"
        assert not code_widths.is_legal_width(bad, q), f"{bad} passed as legal"
        # BREAKAGE: adopt prompt_2's row and the lookup must raise.
        saved = dict(code_widths.WIDTH_TABLE)
        try:
            code_widths.WIDTH_TABLE[(63, k)] = bad
            expect_raises(lambda: code_widths.declared_width(63, k),
                          f"WIDTH_TABLE[(63,{k})]={bad} accepted")
        finally:
            code_widths.WIDTH_TABLE.clear()
            code_widths.WIDTH_TABLE.update(saved)


def test_a_None_entry_for_a_code_that_needs_a_width_is_rejected():
    saved = dict(code_widths.WIDTH_TABLE)
    try:
        # BREAKAGE: claim BCH(63,57) needs no width. q=7 divides no published
        # width in archs/, so the mapper would abort on every layer.
        code_widths.WIDTH_TABLE[(63, 57)] = None
        expect_raises(lambda: code_widths.declared_width(63, 57),
                      "None accepted for a code whose q does not divide 8")
    finally:
        code_widths.WIDTH_TABLE.clear()
        code_widths.WIDTH_TABLE.update(saved)


def test_unknown_codes_fall_back_to_the_same_rule():
    """'Make the code such that all the bch configurations work' -- including
    the ones not in the dict."""
    for k in range(1, 63):
        q = code_widths.declared_datawidth(63, k)
        w = code_widths.declared_width(63, k)
        if w is None:
            assert 8 % q == 0, f"BCH(63,{k}) declares nothing but q={q}"
        else:
            assert w % q == 0 and w % 8 == 0, f"BCH(63,{k}) -> illegal {w}"
    # and a different N
    for n in (127, 255, 31):
        for k in range(1, n):
            q = code_widths.declared_datawidth(n, k)
            w = code_widths.declared_width(n, k)
            assert w is None or (w % q == 0 and w % 8 == 0), \
                f"BCH({n},{k}) -> illegal {w}"


def test_glb_is_the_multiple_and_keeps_divisibility():
    for (n, k) in code_widths.WIDTH_TABLE:
        q = code_widths.declared_datawidth(n, k)
        w = code_widths.declared_width(n, k)
        g = code_widths.glb_width(n, k, 4)
        if w is None:
            assert g is None
            continue
        assert g == 4 * w
        assert g % q == 0 and g % 8 == 0, f"GLB {g} illegal at BCH({n},{k})"


# ------------------------------------------- against the REAL architectures
def _paper_text(arch):
    """The paper-fidelity YAML, through the only resolver there is."""
    return archs.paper_source(arch).read_text()


def _published_weight_widths(arch):
    text = _paper_text(arch)
    out = []
    for part, name, is_weight, _why in archs._weight_level_parts(text, "exclusive"):
        if not is_weight:
            continue
        w = re.search(r"\bwidth:\s*(\d+)", part)
        d = re.search(r"\bdepth:\s*(\d+)", part)
        out.append((name, int(d.group(1)), int(w.group(1))))
    return out


def test_the_real_yaml_accepts_every_code_under_the_table():
    """The claim the whole module rests on: with the table applied, the
    datawidth rewrite no longer raises on a real architecture."""
    cfg = config.load_config()
    for arch in ("eyeriss_like_wglb", "eyeriss_v2_like_wglb"):
        text = _paper_text(arch)
        for k in (57, 51, 45, 39, 36, 30):
            q = code_widths.declared_datawidth(63, k)
            w = code_widths.declared_width(63, k)
            t = text
            if w is not None:
                t = archs._set_weight_width(t, w, 4, "exclusive", arch, quiet=True)
            # must NOT raise -- this is the bug the module exists to remove
            archs._set_weight_datawidth(t, q, "exclusive", arch, quiet=True)
            # and the 8-bit arm on the SAME width must not raise either
            archs._set_weight_datawidth(t, cfg.weight_bits, "exclusive", arch,
                                        quiet=True)


def test_without_the_table_the_real_yaml_still_aborts():
    """The breakage that proves the test above is testing something: on the
    PUBLISHED widths, q=7/6/5 abort exactly as they did before."""
    for arch in ("eyeriss_like_wglb", "eyeriss_v2_like_wglb"):
        text = _paper_text(arch)
        for k in (57, 45, 39):
            q = code_widths.declared_datawidth(63, k)
            expect_raises(
                lambda t=text, q=q, a=arch: archs._set_weight_datawidth(
                    t, q, "exclusive", a, quiet=True),
                f"{arch}: q={q} should abort on the published widths")


def test_the_width_reshape_holds_the_declared_bits():
    """`_set_weight_width` renormalises depth to hold total bits. Check the
    error is bounded, and REPORT it -- it is the cost of the table."""
    worst = 0.0
    for arch in ("eyeriss_like_wglb", "eyeriss_v2_like_wglb"):
        pub = {n: (d, w) for n, d, w in _published_weight_widths(arch)}
        spad = list(pub)[-1]
        for k in (57, 51, 45, 39, 36, 30):
            w = code_widths.declared_width(63, k)
            if w is None:
                continue
            for name, (d0, w0) in pub.items():
                want = w if name == spad else w * 4
                nd = code_widths.renormalised_depth(d0, w0, want)
                assert nd >= 2, (
                    f"{arch}/{name} at BCH(63,{k}) renormalises to depth {nd}; "
                    f"a depth-1 level is dropped as a pipeline latch and would "
                    f"stop being narrowed at all")
                err = 100.0 * (nd * want - d0 * w0) / (d0 * w0)
                worst = max(worst, abs(err))
    print(f"        worst bit error across the table: {worst:.2f}%")
    assert worst < 15.0, f"bit error {worst:.2f}% is too large to call it the same array"


# ------------------------------------------------------------ the config knob
def test_auto_resolves_and_empty_does_not():
    saved = dict(os.environ)
    try:
        for k, want in ((57, 56), (51, 24), (45, 24), (39, 40), (36, 40),
                        (30, None)):
            os.environ["ECC_CODE_N"] = "63"
            os.environ["ECC_CONST_K"] = str(k)
            os.environ["ECC_WEIGHT_WIDTH"] = "auto"
            c = config.load_config()
            assert c.weight_width == want, \
                f"auto at BCH(63,{k}) gave {c.weight_width}, want {want}"
            # BREAKAGE: EMPTY must keep the published geometry at EVERY code,
            # or every pre-2026-09-11 cache silently changes meaning.
            os.environ["ECC_WEIGHT_WIDTH"] = ""
            assert config.load_config().weight_width is None, \
                f"EMPTY resolved a width at BCH(63,{k})"
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_BCH_63_30_fingerprints_identically_under_auto_and_empty():
    """The compatibility claim: turning the knob on must not cold the cache the
    prompt_5 run is held in."""
    saved = dict(os.environ)
    try:
        os.environ["ECC_CODE_N"] = "63"
        os.environ["ECC_CONST_K"] = "30"
        os.environ["ECC_WEIGHT_WIDTH"] = ""
        empty = config.load_config().fingerprint()
        os.environ["ECC_WEIGHT_WIDTH"] = "auto"
        auto = config.load_config().fingerprint()
        assert empty == auto, f"BCH(63,30) cache colded: {empty} != {auto}"
        # BREAKAGE: a code that DOES take a width must NOT share it.
        os.environ["ECC_CONST_K"] = "39"
        assert config.load_config().fingerprint() != auto, \
            "BCH(63,39) shares BCH(63,30)'s fingerprint -- a width change that " \
            "does not cold the cache is a width change nobody applied"
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_auto_and_an_explicit_width_cannot_both_be_set():
    saved = dict(os.environ)
    try:
        os.environ["ECC_CONST_K"] = "39"
        os.environ["ECC_WEIGHT_WIDTH"] = "auto"
        config.load_config()               # fine
        # BREAKAGE: config.py must still refuse an illegal hand-typed width.
        os.environ["ECC_WEIGHT_WIDTH"] = "95"
        expect_raises(config.load_config,
                      "prompt_2's 95 accepted as a hand-typed width")
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_both_arms_declare_the_same_silicon_under_auto():
    """`assert_pair_geometry` is the fairness rule; the table must not break
    it. This is the check hpc/map_depth_sweep.sh runs before it queues."""
    saved = dict(os.environ)
    try:
        os.environ["ECC_CODE_N"] = "63"
        os.environ["ECC_WEIGHT_WIDTH"] = "auto"
        os.environ["ECC_ARCH_FIDELITY"] = "paper"
        for k in (57, 51, 45, 39, 36, 30):
            os.environ["ECC_CONST_K"] = str(k)
            cfg = config.load_config()
            q = code_widths.declared_datawidth(63, k, cfg.weight_bits)
            arch = "eyeriss_like_wglb"
            emb = dataclasses.replace(cfg, weight_datawidth=None)
            rec = dataclasses.replace(cfg, weight_datawidth=q)
            info = archs.assert_pair_geometry(arch, emb, rec)
            for lvl, v in info.items():
                assert v["capacity_ratio"] >= 1.0, (
                    f"BCH(63,{k}) {lvl}: recon holds {v['capacity_ratio']:.4f}x "
                    f"the weights -- a reduced word cannot hold FEWER")
                want = 8 / q
                assert abs(v["capacity_ratio"] - want) < 1e-9, (
                    f"BCH(63,{k}) {lvl}: capacity {v['capacity_ratio']:.4f}x, "
                    f"expected 8/{q} = {want:.4f}x")
    finally:
        os.environ.clear()
        os.environ.update(saved)


def main():
    print("THE WIDTH TABLE -- eccenergy/code_widths.py")
    print(code_widths.audit())
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
