"""Tests for the Task 4 capacity-dilation diff and its two guards.

    python3 -m eccenergy.tests.test_dilation
    bash hpc/tl.sh python3 -m eccenergy.tests.test_dilation   # for the cache ones

Dependency-free and offline like the other suites: the stats parser is driven by
a SYNTHETIC `timeloop-mapper.stats.txt` built by `_stats()`, whose numbers are
chosen so `room`, `held`, `fill` and the PE counts all have hand-checkable
answers.

WHY THE GUARDS ARE TESTED AND NOT JUST THE ARITHMETIC. Task 4 asks whether
N/K more weight room cuts DRAM refetch, and on 2026-09-09 two separate
FALSE POSITIVES were reported before these guards existed:

  * `eyeriss_v2_like` layer2.0.conv1 "14.00x -> 4.00x at x1.6154". The 14.00x
    came from `fp-3eb860ea2b2a`, solved before a configuration change; the
    fingerprint the current config reads is `fp-88656178371f` and it ALREADY
    refetches 4.00x undilated. A cross-fingerprint comparison compares two
    ARCHITECTURES. `sibling_fingerprints()` is the guard.
  * `simple_weight_stationary` layer2.0.conv1 "2.00x -> 1.00x at x1.6154". The
    two nests differ only in the ORDER of `for Q in [0:2)` and `for C in [0:4)`
    at the DRAM level; `weights held` is identical at x1, x1.6154 and x4, and
    x4 refetches 2.00x again. The search found a permutation available at the
    declared capacity. `capacity_verdict()` is the guard.

Each guard therefore has a test that a deliberately-broken input trips it, and
a test that a genuine capacity win is NOT flagged -- a guard that fires on
everything would be as useless as no guard at all.
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import traceback

from .. import config

FAILURES = []
SKIPPED = []


def check(name, fn):
    try:
        fn()
    except _Skip as why:
        SKIPPED.append(name)
        print(f"  skip  {name}  ({why})")
    except Exception:
        FAILURES.append(name)
        print(f"  FAIL  {name}")
        traceback.print_exc()
    else:
        print(f"  ok    {name}")


class _Skip(Exception):
    pass


# ---------------------------------------------------------------------------
#  the synthetic mapping
# ---------------------------------------------------------------------------
def _stats(dram_reads=8000, weights=1000, mac_used=168, mac_declared=168,
           levels=(("weights_spad", 448, 128, 168),), other_fill=None,
           total_uj=200.0, cycles=451584, util=76.19):
    """One `timeloop-mapper.stats.txt`. `levels` is (name, cap, resid, inst).

    Defaults reproduce `eyeriss_like` layer2.0.conv1's shape of answer: 168
    PEs, a 448-weight spad holding 128, so room = 75,264 and held = 21,504.
    """
    out = ["Buffer and Arithmetic Levels", "----------------------------",
           "Level 0", "-------", "=== mac ===", "",
           "    SPECS", "    -----",
           "    Word bits             : 16",
           f"    Instances             : {mac_declared} (14*12)",
           "    Compute energy        : 1.14 pJ", "",
           "    STATS", "    -----",
           f"    Utilized instances      : {mac_used}", ""]
    n = 1
    for (name, cap, resid, inst) in levels:
        out += [f"Level {n}", "-------", f"=== {name} ===", "",
                "    SPECS", "    -----",
                "        Technology                      : SRAM",
                "        Word bits                       : 8", "",
                "    STATS", "    -----",
                f"    Effective size       : {cap}",
                "    Weights:",
                f"        Partition size                           : {weights}",
                f"        Utilized capacity                        : {resid}",
                f"        Utilized instances (max)                 : {inst}",
                "        Scalar reads (per-instance)              : 100",
                "        Energy (total)                           : 800.00 pJ", ""]
        n += 1
    if other_fill is not None:
        cap, resid = other_fill
        out += [f"Level {n}", "-------", "=== ifmap_spad ===", "",
                "    SPECS", "    -----",
                "        Word bits                       : 8", "",
                "    STATS", "    -----",
                f"    Effective size       : {cap}",
                "    Inputs:",
                f"        Utilized capacity                        : {resid}",
                "        Utilized instances (max)                 : 168",
                "        Scalar reads (per-instance)              : 77",
                "        Energy (total)                           : 111.00 pJ", ""]
        n += 1
    out += [f"Level {n}", "-------", "=== DRAM ===", "",
            "    SPECS", "    -----",
            "        Word bits                       : 8",
            "        Block size                      : 8", "",
            "    STATS", "    -----",
            "    Effective size       : 1048576",
            "    Weights:",
            f"        Partition size                           : {weights}",
            f"        Utilized capacity                        : {weights}",
            "        Utilized instances (max)                 : 1",
            f"        Scalar reads (per-instance)              : {dram_reads}",
            "        Energy (per-scalar-access)               : 128.00 pJ",
            "        Energy (total)                           : 5000.00 pJ", "",
            "Summary Stats", "-------------",
            f"Utilization: {util}%", f"Cycles: {cycles}",
            "Computes = 57802752", "", "Energy: %.2f uJ" % total_uj, "",
            "Computes = 57802752", "    mac                   = 1140.00", ""]
    return "\n".join(out)


def _read(tmp, name, **kw):
    """Write a synthetic mapping under a `fp-<hash>/<shape>/` path and parse it."""
    from eccenergy.experiments import dilation
    levels = kw.get("levels", (("weights_spad", 448, 128, 168),))
    d = pathlib.Path(tmp) / name / "C64_M128_R3_S3_P28_Q28_ws2_hs2"
    d.mkdir(parents=True, exist_ok=True)
    (d / "timeloop-mapper.stats.txt").write_text(_stats(**kw))
    return dilation.read_mapped_layer(d / "timeloop-mapper.stats.txt",
                                      levels[-1][0],
                                      tuple(l[0] for l in levels))


# ===========================================================================
#  the parser
# ===========================================================================
def test_the_parser_reads_pes_room_held_fill_and_the_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        m = _read(tmp, "fp-deadbeef0001")
    assert m is not None, "the synthetic mapping did not parse at all"
    assert m.pes_used == 168 and m.pes_declared == 168, (m.pes_used, m.pes_declared)
    assert m.weight_instances == 168, m.weight_instances
    assert m.weights_per_pe == 448, m.weights_per_pe
    # room and held are per-instance x instances, by hand
    assert m.weight_room == 448 * 168 == 75264, m.weight_room
    assert m.weights_held == 128 * 168 == 21504, m.weights_held
    assert abs(m.fill - 21504 / 75264) < 1e-12, m.fill
    assert m.dram_weight_reads == 8000 and m.weights == 1000
    assert m.refetch == 8.0, m.refetch
    assert m.cycles == 451584 and abs(m.arch_util - 76.19) < 1e-9
    # the fingerprint is the GRANDPARENT of the stats file, not the shape dir
    assert m.fingerprint == "fp-deadbeef0001", m.fingerprint


def test_room_and_held_sum_every_weight_carrying_level():
    """`eyeriss_like_wglb` holds weights in a GLB *and* a spad, and the GLB is
    the one whose dilation can absorb a DRAM-level loop. Reporting only the
    innermost level would say 21 % about a design whose GLB is three-quarters
    used."""
    two = (("filter_glb", 8192, 6144, 1), ("weights_spad", 448, 128, 168))
    with tempfile.TemporaryDirectory() as tmp:
        m = _read(tmp, "fp-deadbeef0002", levels=two)
    assert m.weight_room == 8192 * 1 + 448 * 168 == 83456, m.weight_room
    assert m.weights_held == 6144 * 1 + 128 * 168 == 27648, m.weights_held
    # `weights per PE` and `PEs` stay the INNERMOST level's, so w/PE x PEs is
    # that level's own room and the table row remains self-consistent
    assert m.weights_per_pe == 448 and m.weight_instances == 168


# ===========================================================================
#  GUARD 1 -- the permutation / parallelism verdict
# ===========================================================================
def test_a_permutation_only_difference_is_not_called_capacity():
    """THE `simple_weight_stationary` FALSE POSITIVE. Reads halve while `held`
    is identical at every weight level: nothing extra was stored, so the cause
    is the loop nest's order and not the silicon."""
    from eccenergy.experiments import dilation
    with tempfile.TemporaryDirectory() as tmp:
        ref = _read(tmp, "fp-a", dram_reads=2000,
                    levels=(("pe_spad", 384, 24, 16),), mac_used=16,
                    mac_declared=256)
        # capacity up, reads halved, HELD UNCHANGED -- exactly what was measured
        dil = _read(tmp, "fp-b", dram_reads=1000,
                    levels=(("pe_spad", 620, 24, 16),), mac_used=16,
                    mac_declared=256)
    assert ref.weights_held == dil.weights_held == 384
    v = dilation.capacity_verdict(ref, dil)
    assert v.startswith("PERM?"), v
    assert "capacity" not in v, (
        "reads fell while `held` did not move, and the verdict still claimed a "
        f"capacity effect: {v}")


def test_a_pe_count_change_is_flagged_rather_than_attributed():
    """THE FINDINGS 7.7 CONFOUND, in a column. Capacity and parallelism moved
    together, so neither is separable and the row must say so."""
    from eccenergy.experiments import dilation
    with tempfile.TemporaryDirectory() as tmp:
        ref = _read(tmp, "fp-a", dram_reads=14000, mac_used=384,
                    levels=(("weights_spad", 288, 16, 192),))
        dil = _read(tmp, "fp-b", dram_reads=4000, mac_used=288,
                    levels=(("weights_spad", 465, 16, 144),))
    v = dilation.capacity_verdict(ref, dil)
    assert v.startswith("PE!="), v
    assert "capacity" not in v, (
        f"parallelism moved and the verdict still claimed capacity: {v}")


def test_a_genuine_capacity_win_is_called_capacity_and_nothing_else():
    """A guard that fired on everything would be as useless as no guard. Reads
    FALL and `held` RISES: that is the hypothesis, and it must come back clean."""
    from eccenergy.experiments import dilation
    with tempfile.TemporaryDirectory() as tmp:
        ref = _read(tmp, "fp-a", dram_reads=2000, weights=1000,
                    levels=(("weight_noc", 65536, 58982, 1),))
        dil = _read(tmp, "fp-b", dram_reads=1000, weights=1000,
                    levels=(("weight_noc", 105882, 73728, 1),))
    assert dil.weights_held > ref.weights_held
    v = dilation.capacity_verdict(ref, dil)
    assert v == "capacity", v


def test_an_unchanged_mapping_is_flat_and_claims_nothing():
    from eccenergy.experiments import dilation
    with tempfile.TemporaryDirectory() as tmp:
        ref = _read(tmp, "fp-a", dram_reads=8000)
        dil = _read(tmp, "fp-b", dram_reads=8000,
                    levels=(("weights_spad", 724, 128, 168),))
    v = dilation.capacity_verdict(ref, dil)
    assert v == "flat", v
    assert "capacity" not in v


# ===========================================================================
#  GUARD 2 -- the fingerprint guard, on the REAL cache
# ===========================================================================
def test_the_fingerprint_guard_finds_the_real_sibling_fingerprints():
    """`sibling_fingerprints()` against the cache actually on disk.

    This is a property test on real data, not a fixture: it asserts the guard
    can enumerate what is there and that it marks exactly one entry current.
    Where a design has more than one solved fingerprint under one variant slug
    -- which is the trap that produced the withdrawn 14.00x -- the guard must
    return them all, so the table can refuse to compare across them.
    """
    try:
        from eccenergy import config
        from eccenergy.experiments import dilation
        cfg = config.load_config()
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"config unavailable: {exc}")
    shapes = ["C64_M128_R3_S3_P28_Q28_ws2_hs2"]
    seen_any = False
    for arch in ("eyeriss_like", "eyeriss_v2_like", "simple_weight_stationary"):
        try:
            sibs = dilation.sibling_fingerprints(cfg, arch, 1.0, shapes)
        except Exception as exc:
            raise AssertionError(f"{arch}: the guard itself raised: {exc}")
        if not sibs:
            continue
        seen_any = True
        current = [s for s in sibs if s["current"]]
        assert len(current) <= 1, (
            f"{arch}: {len(current)} fingerprints claim to be the current one")
        for s in sibs:
            assert (s["path"] / shapes[0] / "timeloop-mapper.stats.txt").is_file()
    if not seen_any:
        raise _Skip("no mapper cache on disk for the probe shape")


def test_no_pair_on_disk_is_reported_as_capacity_without_held_rising():
    """THE INVARIANT, over every capacity pair actually cached.

    Whatever the sweep has filled in, a pair the table calls a capacity effect
    must have stored strictly more weights on chip. This is the assertion that
    would have refused both of 2026-09-09's false positives, run against real
    mappings rather than a fixture.
    """
    try:
        from eccenergy import config
        from eccenergy.experiments import dilation
        cfg = config.load_config()
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"config unavailable: {exc}")
    import dataclasses
    k_over_n = cfg.code_k / cfg.code_n
    shapes = ["C64_M128_R3_S3_P28_Q28_ws2_hs2"]
    checked = 0
    # BOTH scopes. The `capacity`-without-`held`-rising bug hid because this
    # test only swept `exclusive`, where no such pair existed; the shared-scope
    # caches had four of them.
    for scope in ("exclusive", "shared"):
      cfg = dataclasses.replace(cfg, weight_capacity_scope=scope)
      for arch in ("eyeriss_like", "eyeriss_v2_like", "simple_weight_stationary",
                 "eyeriss_like_wglb", "eyeriss_v2_like_wglb"):
        for ref_scale in (0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 0.875,
                          1.0, 1.125, 2.0, 4.0, 8.0):
            dil_scale = round(ref_scale / k_over_n, 4)
            try:
                ref, _ = dilation.load(cfg, arch, ref_scale, shapes)
                dil, _ = dilation.load(cfg, arch, dil_scale, shapes)
            except Exception:
                continue
            for sh in shapes:
                if sh not in ref or sh not in dil:
                    continue
                checked += 1
                v = dilation.capacity_verdict(ref[sh], dil[sh])
                if v == "capacity":
                    assert dil[sh].weights_held > ref[sh].weights_held, (
                        f"{arch} {scope} x{ref_scale} -> x{dil_scale}: verdict "
                        f"'{v}' claims a capacity effect but weights held did "
                        f"not rise ({ref[sh].weights_held:,} -> "
                        f"{dil[sh].weights_held:,})")
                    assert dil[sh].pes_used == ref[sh].pes_used, (
                        f"{arch} {scope} x{ref_scale} -> x{dil_scale}: verdict "
                        f"'{v}' claims capacity while the PE count moved "
                        f"({ref[sh].pes_used} -> {dil[sh].pes_used})")
    if not checked:
        raise _Skip("no capacity pair cached yet for the probe shape")
    print(f"        ({checked} cached capacity pair(s) checked)", end="")


# ===========================================================================
#  LEVER 2 -- relaxing the dataflow `factors:` so the tile can grow
# ===========================================================================
_ARCH = """architecture:
  nodes:
  - !Component
    name: DRAM
    class: DRAM
    attributes: {depth: 1048576, width: 64, datawidth: 8}
    constraints:
      dataspace: {keep: [Weights]}
  - !Component
    name: weights_spad
    class: smartbuffer_SRAM
    attributes: {depth: 224, width: 16, datawidth: 8}
    constraints:
      dataspace: {keep: [Weights]}
      temporal:
        permutation: [N, M, P, Q, S, C, R]
        factors: [N=1, M=1, P=1, Q=1, S=1]
  - !Component
    name: psum_spad
    class: smartbuffer_RF
    attributes: {depth: 24, width: 16, datawidth: 16}
    constraints:
      dataspace: {keep: [Outputs]}
      temporal:
        factors: [N=1, C=1, R=1, S=1, P=1, Q=1]
  - !Component
    name: weight_reg
    class: smartbuffer_RF
    attributes: {depth: 1, width: 8, datawidth: 8}
    constraints:
      dataspace: {keep: [Weights]}
      temporal:
        factors: [N=1, M=1, C=1, R=1, S=1]
"""


def test_the_relax_drops_only_weight_indexed_pins_on_weight_levels():
    """M, C, R, S go; N, P, Q stay.

    Weights do not index N, P or Q, so relaxing those pins would retile the
    activations and partial sums instead of the weight tile -- a different
    experiment. The psum level must not be touched at all even though its pins
    name C, R and S, because it does not hold Weights.
    """
    from eccenergy import archs
    out = archs._relax_weight_factors(_ARCH, "fixture", quiet=True)
    # the weight level lost exactly M and S, and kept N, P, Q in order
    assert "factors: [N=1, P=1, Q=1]" in out, out
    assert "factors: [N=1, M=1, P=1, Q=1, S=1]" not in out
    # the PSUM level is untouched -- it holds Outputs, not Weights
    assert "factors: [N=1, C=1, R=1, S=1, P=1, Q=1]" in out, (
        "the partial-sum level's pins were relaxed; only weight-carrying "
        "levels may be")
    # the depth-1 LATCH is untouched -- FINDINGS 7.5, it is not a reuse level
    assert "factors: [N=1, M=1, C=1, R=1, S=1]" in out, (
        "the depth-1 weight latch was relaxed; scaling or relaxing it invents "
        "a reuse level the design does not have")


def test_the_relax_is_a_no_op_when_nothing_weight_indexed_is_pinned():
    """The no-op rule, which keeps a design on its existing cache.

    `simple_weight_stationary` pins nothing a weight tile is indexed by, so
    this lever cannot help it and it must NOT pay for a fresh map. A slug that
    said `wrelax` on an unchanged architecture would cold-start every design.
    """
    from eccenergy import archs
    free = _ARCH.replace("factors: [N=1, M=1, P=1, Q=1, S=1]",
                         "factors: [N=1, P=1, Q=1]")
    assert archs._relax_weight_factors(free, "fixture", quiet=True) == free


def test_the_relax_changes_the_cache_slug_and_the_fingerprint():
    """A relaxed dataflow is a different MAPSPACE, so it is a different
    architecture to the mapper and must never share a cache directory."""
    try:
        from eccenergy import archs, config
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"config unavailable: {exc}")
    import os
    saved = os.environ.get("ECC_WEIGHT_FACTOR_RELAX")
    try:
        seen = {}
        for flag in ("0", "1"):
            os.environ["ECC_WEIGHT_FACTOR_RELAX"] = flag
            cfg = config.load_config()
            arch = "eyeriss_like"
            seen[flag] = (archs.effective_variant(arch, cfg),
                          archs.arch_fingerprint(arch, cfg))
    finally:
        if saved is None:
            os.environ.pop("ECC_WEIGHT_FACTOR_RELAX", None)
        else:
            os.environ["ECC_WEIGHT_FACTOR_RELAX"] = saved
    assert not seen["0"][0].endswith("wrelax"), seen["0"]
    assert seen["1"][0].endswith("wrelax"), seen["1"]
    assert seen["0"][1] != seen["1"][1], (
        f"the relaxed and unrelaxed architectures share fingerprint "
        f"{seen['0'][1]} -- one would be read as the other")


# ===========================================================================
#  PROMPT_2 (2026-09-10): datawidth, depth, width and the fairness rule
# ===========================================================================
#  Each of these guards a claim prompt_2 makes about the mechanism, and each
#  fails by name if the claim stops holding. The mechanism is: the two arms
#  share ONE hardware YAML and differ ONLY in `datawidth:`, which CACTI never
#  sees -- so the reduced arm gets more effective capacity at byte-identical
#  per-access energy. Every part of that sentence is checked below.

_P2_ARCH = "eyeriss_like_wglb"


def _p2_cfgs(**over):
    """`(cfg, embedded, recon)` at BCH(63,30) on Eyeriss v1."""
    import dataclasses
    try:
        from eccenergy import config
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"config unavailable: {exc}")
    import os
    saved = {k: os.environ.get(k) for k in ("ECC_KS", "ECC_CONST_K", "ECC_SWEEP")}
    try:
        os.environ["ECC_KS"] = "30"
        os.environ["ECC_CONST_K"] = "30"
        os.environ["ECC_SWEEP"] = "arch"
        cfg = config.load_config()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    cfg = dataclasses.replace(cfg, **over) if over else cfg
    emb = dataclasses.replace(cfg, weight_datawidth=None)
    rec = dataclasses.replace(cfg, weight_datawidth=4)
    return cfg, emb, rec


def test_the_datawidth_knob_narrows_every_weight_level_and_never_dram():
    """DRAM `datawidth` STAYS 8 ON EVERY ARM.

    `recon.py` owns the DRAM K/N scaling. Narrowing DRAM in the YAML too would
    charge the same reduction twice -- once in the mapping and once in the
    evaluator -- which is prompt_2's first BEFORE-ANY-NUMBER item. It is
    structural, so it is asserted rather than remembered.
    """
    try:
        from eccenergy import archs
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"archs unavailable: {exc}")
    import re
    _, emb, rec = _p2_cfgs()
    text = archs._patched_text(_P2_ARCH, rec, quiet=True)
    # every WEIGHT level narrowed ...
    geo = archs.patched_weight_geometry(_P2_ARCH, rec)
    assert geo, "no weight levels found"
    for level, g in geo.items():
        assert g["datawidth"] == 4, (level, g)
    # ... and DRAM untouched.
    head = text.split("name: DRAM", 1)[1].split("- !", 1)[0]
    dram_dw = int(re.search(r"\bdatawidth:\s*(\d+)", head).group(1))
    assert dram_dw == 8, (
        f"DRAM datawidth is {dram_dw}, not 8. recon.py already scales the DRAM "
        f"term by K/N, so narrowing it here DOUBLE-COUNTS the saving in DRAM.")


def test_the_two_arms_declare_identical_silicon_at_every_swept_depth():
    """PROMPT_2'S FAIRNESS RULE. Same `width`, same `depth`, every level.

    This is the condition depth-dilation could never meet: expressing capacity
    as `depth x N/K` made Accelergy price the reconstruction arm's array
    1.18-1.46x dearer per access, so the optimiser had a positive reason to
    leave the room unused (FINDINGS 7.8). If a width or a depth ever differs
    between the arms the comparison is VOID, so this must fail loudly.
    """
    try:
        from eccenergy import archs
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"archs unavailable: {exc}")
    import dataclasses
    _, emb, rec = _p2_cfgs()
    for s in (1.0, 0.7071, 0.5, 0.3536, 0.25, 0.1768, 0.125):
        e = dataclasses.replace(emb, weight_depth_scale=round(s, 4))
        r = dataclasses.replace(rec, weight_depth_scale=round(s, 4))
        info = archs.assert_pair_geometry(_P2_ARCH, e, r)   # raises if not
        assert info, s
        for level, v in info.items():
            # the treatment, and the ONLY difference
            assert v["embedded"] == 8 and v["recon"] == 4, (s, level, v)


def test_a_depth_mismatch_between_the_arms_is_refused_but_a_width_one_is_not():
    """The fairness assertion is a STOP, not an annotation -- FOR DEPTH.

    Deliberately break it the way the previous sweep did (give the
    reconstruction arm a deeper array) and check the comparison is refused
    rather than corrected afterwards. Then check the thing it must NOT refuse:
    a WIDTH difference, which under prompt_2's WIDTH TABLE is the treatment.
    Each arm declares the width that suits its own datawidth -- 96 at q=8,
    95 at q=5 -- and neither has to be legal for the other's, because neither
    is ever mapped on the other's silicon. Asserting a shared width is what
    produced the withdrawn lcm(q, 8) scheme; see eccenergy/code_widths.py.
    """
    try:
        from eccenergy import archs
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"archs unavailable: {exc}")
    import dataclasses
    _, emb, rec = _p2_cfgs()
    dilated = dataclasses.replace(rec, weight_depth_scale=2.0)  # the old model
    try:
        archs.assert_pair_geometry(_P2_ARCH, emb, dilated)
    except ValueError as e:
        assert "depth" in str(e) and "silicon" in str(e), str(e)
    else:
        raise AssertionError(
            "a reconstruction arm with 2x the DEPTH was accepted as a fair "
            "pair. That is exactly the geometry FINDINGS 7.8 withdrew.")
    # ... and the knob that lets a deliberate depth study through.
    allowed = dataclasses.replace(dilated, disable_pair_geometry_assert=True)
    archs.assert_pair_geometry(_P2_ARCH, emb, allowed)          # must not raise

    # A WIDTH DIFFERENCE IS NOT A DEFECT. BCH(63,39) declares 95 against the
    # 8-bit arm's 96 and the pair is legal.
    saved = dict(os.environ)
    try:
        os.environ["ECC_KS"] = os.environ["ECC_CONST_K"] = "39"
        os.environ["ECC_RECON_K"] = "39"
        cfg39 = config.load_config()
        e39 = dataclasses.replace(cfg39, weight_datawidth=None)
        r39 = dataclasses.replace(cfg39, weight_datawidth=5)
        info = archs.assert_pair_geometry(_P2_ARCH, e39, r39)   # must not raise
    finally:
        os.environ.clear()
        os.environ.update(saved)
    differing = {lvl: (v["embedded_width"], v["recon_width"])
                 for lvl, v in info.items()
                 if v["embedded_width"] != v["recon_width"]}
    assert differing, (
        f"BCH(63,39)'s arms declare identical widths -- then prompt_2's table "
        f"is not being applied: {info}")
    for lvl, (we, wr) in differing.items():
        assert we % 8 == 0 and wr % 5 == 0, (lvl, we, wr)


def test_the_reduced_arm_gets_exactly_two_times_the_capacity_at_bch_63_30():
    """BCH(63,30) is the code with ZERO rounding residual, at every depth.

    q = round(8*30/63) = 4 divides Eyeriss v1's published 16-bit scratchpad
    word and its 64-bit GLB word, so no width change is needed and the
    effective-capacity ratio is EXACTLY 2.000 -- not 1.98, not 2.02. That
    exactness is why prompt_2 starts here: it is also the only code that
    clears the integer-tile step FINDINGS 7.8 measured.
    """
    try:
        from eccenergy import archs
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"archs unavailable: {exc}")
    import dataclasses
    _, emb, rec = _p2_cfgs()
    for s in (1.0, 0.7071, 0.5, 0.3536, 0.25, 0.1768, 0.125):
        e = dataclasses.replace(emb, weight_depth_scale=round(s, 4))
        r = dataclasses.replace(rec, weight_depth_scale=round(s, 4))
        for level, v in archs.assert_pair_geometry(_P2_ARCH, e, r).items():
            assert v["capacity_ratio"] == 2.0, (
                f"x{s:g} {level}: capacity ratio {v['capacity_ratio']} != "
                f"exactly 2.0 -- BCH(63,30) is chosen precisely because it "
                f"has no rounding residual")


def test_the_embedded_arm_spelled_8b_reads_the_same_cache_as_unset():
    """Both spellings of the 8-bit arm must be ONE cache directory.

    The reference arm may legitimately be written `ECC_WEIGHT_DATAWIDTH=8` on
    a design already declaring 8. If that produced its own slug, the two arms
    of a pair would be compared across two mapper caches of one IDENTICAL
    architecture -- hours of recompute, and a diff of the search rather than
    of the silicon.
    """
    try:
        from eccenergy import archs
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"archs unavailable: {exc}")
    import dataclasses
    cfg, emb, _ = _p2_cfgs()
    eight = dataclasses.replace(emb, weight_datawidth=8)
    assert (archs.effective_variant(_P2_ARCH, emb)
            == archs.effective_variant(_P2_ARCH, eight)), (
        archs.effective_variant(_P2_ARCH, emb),
        archs.effective_variant(_P2_ARCH, eight))
    assert (archs.arch_fingerprint(_P2_ARCH, emb)
            == archs.arch_fingerprint(_P2_ARCH, eight))


def test_the_depth_sweep_has_its_own_cache_and_never_shares_wcap_s():
    """`wdepth` and `wcap` rewrite the same field and mean different things.

    `ECC_WEIGHT_CAPACITY_SCALE` triggers `capacity_dilation_correction()`,
    which re-prices the level at the UNDILATED geometry. Under prompt_2 that
    correction is WRONG -- a shallower array really IS a smaller array and its
    cheaper access is a real saving. Sharing a directory would let a corrected
    run be read as an uncorrected one.
    """
    try:
        from eccenergy import archs
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"archs unavailable: {exc}")
    import dataclasses
    cfg, _, _ = _p2_cfgs()
    depth = dataclasses.replace(cfg, weight_depth_scale=0.5)
    cap = dataclasses.replace(cfg, weight_capacity_scale=0.5)
    vd = archs.effective_variant(_P2_ARCH, depth)
    vc = archs.effective_variant(_P2_ARCH, cap)
    assert "wdepth0.5" in vd and "wcap" not in vd, vd
    assert "wcap0.5" in vc and "wdepth" not in vc, vc
    assert vd != vc, f"the two knobs share the cache directory {vd}"
    # THE SEPARATION IS THE SLUG, NOT THE FINGERPRINT, and that is correct.
    # `arch_fingerprint` hashes the patched YAML the mapper actually sees, and
    # at the same scale both knobs rewrite the same `depth:` fields to the same
    # numbers -- the architecture IS identical, so an identical hash is the
    # honest answer. What must not be shared is the DIRECTORY, because what
    # differs is the ACCOUNTING applied afterwards: `wcap` is re-priced by
    # `capacity_dilation_correction()` at the undilated geometry and `wdepth`
    # deliberately is not. One directory would let a corrected run be read as
    # an uncorrected one.
    assert (archs.arch_fingerprint(_P2_ARCH, depth)
            == archs.arch_fingerprint(_P2_ARCH, cap)), (
        "the two knobs no longer produce identical YAML at the same scale; if "
        "that is intended, this test records the assumption that broke")
    from eccenergy import paths as pathsmod
    pd = pathsmod.Results(depth).mapper_cache(
        _P2_ARCH, vd, archs.arch_fingerprint(_P2_ARCH, depth), create=False)
    pc = pathsmod.Results(cap).mapper_cache(
        _P2_ARCH, vc, archs.arch_fingerprint(_P2_ARCH, cap), create=False)
    assert str(pd) != str(pc), f"both knobs resolve to {pd}"


def test_naming_a_level_the_design_does_not_have_is_refused():
    """A typo in ECC_WEIGHT_DEPTH_LEVELS must not silently sweep everything.

    The second pass exists to say WHICH level bought the margin. A name that
    matches nothing would fall back to sweeping every level and the result
    would be filed as a per-level answer.
    """
    try:
        from eccenergy import archs
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"archs unavailable: {exc}")
    import dataclasses
    cfg, _, _ = _p2_cfgs()
    bad = dataclasses.replace(cfg, weight_depth_scale=0.5,
                              weight_depth_levels=("filter_gbl",))  # typo
    try:
        archs.patched_weight_geometry(_P2_ARCH, bad)
    except ValueError as e:
        assert "filter_gbl" in str(e) and "no weight-carrying level" in str(e)
    else:
        raise AssertionError("a misspelt weight level was accepted and would "
                             "have swept every level instead")


def test_a_single_named_level_moves_only_that_level():
    """The second pass holds the other weight level(s) at x1."""
    try:
        from eccenergy import archs
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"archs unavailable: {exc}")
    import dataclasses
    cfg, _, _ = _p2_cfgs()
    base = archs.patched_weight_geometry(_P2_ARCH, cfg)
    only = dataclasses.replace(cfg, weight_depth_scale=0.25,
                               weight_depth_levels=("filter_glb",))
    got = archs.patched_weight_geometry(_P2_ARCH, only)
    assert got["filter_glb"]["depth"] == round(base["filter_glb"]["depth"] * 0.25)
    assert got["weights_spad"]["depth"] == base["weights_spad"]["depth"], (
        "the unnamed level moved too, so the row cannot say which level "
        "bought the margin")


def test_datawidth_levels_empty_reproduces_the_unfiltered_rewrite_byte_for_byte():
    """prompt_6 phase 2: `levels=()` must be exactly today's behaviour.

    Every `wdw4` cache on disk was solved without the parameter. If an empty
    tuple changed one byte of the patched YAML, every one of them would go
    cold and the reference/`wdw4` pair the study reads would silently move.
    The property: filtering to the FULL set of weight levels, or to none,
    gives the same text, and neither differs from the unfiltered call.
    """
    try:
        from eccenergy import archs
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"archs unavailable: {exc}")
    text = archs.arch_source(_P2_ARCH, _p2_cfgs()[0]).read_text()
    plain = archs._set_weight_geometry(text, 4, scope="exclusive",
                                       arch=_P2_ARCH, quiet=True)
    empty = archs._set_weight_geometry(text, 4, (), 4, "exclusive", _P2_ARCH,
                                       quiet=True)
    both = archs._set_weight_geometry(text, 4, ("filter_glb", "weights_spad"),
                                      4, "exclusive", _P2_ARCH, quiet=True)
    assert plain == empty, "levels=() changed the patched YAML"
    assert plain == both, "naming every weight level differs from naming none"
    assert plain != text, "the rewrite did nothing at all"
    assert plain.count("datawidth: 4") == 2, plain.count("datawidth: 4")


def test_naming_filter_glb_narrows_filter_glb_and_leaves_the_spad_at_eight():
    """The phase 2 acceptance criterion, on the patched geometry the mapper
    will see: `filter_glb` at 4, `weights_spad` still at 8, DRAM still at 8.
    That is what a `recon2`/`recon4` arm declares (prompt_6 5.2)."""
    try:
        from eccenergy import archs
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"archs unavailable: {exc}")
    import dataclasses
    _, emb, rec = _p2_cfgs()
    glb_only = dataclasses.replace(rec, weight_datawidth_levels=("filter_glb",))
    geo = archs.patched_weight_geometry(_P2_ARCH, glb_only)
    assert geo["filter_glb"]["datawidth"] == 4, geo["filter_glb"]
    assert geo["weights_spad"]["datawidth"] == 8, geo["weights_spad"]
    # THE WIDTH TABLE: the narrowed GLB is 96 x 4 = 384 b at datawidth 4, the
    # spad 96 b at datawidth 8. Both counts are width/datawidth, and both
    # widths come from the datawidth the level actually stores.
    assert geo["filter_glb"]["width"] == 384, geo["filter_glb"]
    assert geo["filter_glb"]["weights_per_word"] == 96, geo["filter_glb"]
    assert geo["weights_spad"]["width"] == 96, geo["weights_spad"]
    assert geo["weights_spad"]["weights_per_word"] == 12, geo["weights_spad"]
    text = archs._patched_text(_P2_ARCH, glb_only, quiet=True)
    dram = [p for p in text.split("\n- !") if "class: DRAM" in p or "name: DRAM" in p]
    assert dram and all("datawidth: 8" in p for p in dram), "DRAM moved"
    # and the slug says which level, so the two arms never share a directory
    v_all = archs.effective_variant(_P2_ARCH, rec)
    v_glb = archs.effective_variant(_P2_ARCH, glb_only)
    assert "wdw4-filter_glb" in v_glb and "wdw4-filter_glb" not in v_all, (v_all, v_glb)
    assert archs.arch_fingerprint(_P2_ARCH, rec) != archs.arch_fingerprint(_P2_ARCH, glb_only)
    # the embedded arm is untouched by the field: no datawidth, nothing to filter
    emb_glb = dataclasses.replace(emb, weight_datawidth_levels=("filter_glb",))
    assert archs.arch_fingerprint(_P2_ARCH, emb) == archs.arch_fingerprint(_P2_ARCH, emb_glb)


def test_a_misspelt_datawidth_level_is_refused():
    """Same rule as ECC_WEIGHT_DEPTH_LEVELS: a typo must not silently narrow
    every level and file the result as a per-boundary architecture."""
    try:
        from eccenergy import archs
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"archs unavailable: {exc}")
    import dataclasses
    _, _, rec = _p2_cfgs()
    bad = dataclasses.replace(rec, weight_datawidth_levels=("filter_gbl",))
    try:
        archs.patched_weight_geometry(_P2_ARCH, bad)
    except ValueError as e:
        assert "filter_gbl" in str(e) and "no weight-carrying level" in str(e), e
    else:
        raise AssertionError("a misspelt weight level was accepted and would "
                             "have narrowed every level instead")
    # DRAM is not a weight level the filter can name either
    bad2 = dataclasses.replace(rec, weight_datawidth_levels=("DRAM",))
    try:
        archs.patched_weight_geometry(_P2_ARCH, bad2)
    except ValueError:
        pass
    else:
        raise AssertionError("naming DRAM was accepted")


def _ert_cfgs():
    """`(reference, recon2, recon4)` configurations on Eyeriss v1 at BCH(63,30).

    The two DC tables are EMPTIED first, so the toll below comes from the DC
    JSON and nothing else. `ecc.load_recon_terms` reads env.sh section 6 BEFORE
    the JSON, and section 10 rebuilds the flattened lists from the `declare -A`
    tables on every source -- an unconditional assignment, so the environment
    cannot override them. Without this, a value typed into section 6 while
    working would silently become the number these tests assert against, and
    prompt_6 5.1's published table would stop being what they check.
    """
    import dataclasses
    cfg, _, _ = _p2_cfgs()
    # prompt_7 Issue 15: prompt_6 Table 5.1 is the UNGATED table, so these
    # assertions are pinned to PCT=0. The gated rows are checked in
    # test_recon.test_clock_gating_is_exact_at_zero_and_scales_the_idle_term.
    cfg = dataclasses.replace(cfg, recon_incremental_table={}, recon_idle_table={},
                              recon_clock_gating_pct=0.0)
    return (cfg, dataclasses.replace(cfg, recon_ert_arm="recon2"),
            dataclasses.replace(cfg, recon_ert_arm="recon4"))


def test_the_energy_model_revision_colds_only_when_set():
    """prompt_7, 2026-09-12. The fingerprint hashes the ARCHITECTURE, not the
    price list Accelergy derives from it, so a corrected estimator changes every
    cached energy without moving the directory it lives in.
    `ECC_ENERGY_MODEL_REV` closes that hole, and it must do so WITHOUT
    disturbing anything: EMPTY has to hash byte-identically to every fingerprint
    made before the knob existed, or Phase A loses the cache it reads."""
    import dataclasses
    try:
        from eccenergy import config
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"config unavailable: {exc}")
    cfg, _, _ = _p2_cfgs()
    base = dataclasses.replace(cfg, energy_model_rev="")
    same = dataclasses.replace(cfg, energy_model_rev="")
    assert base.fingerprint() == same.fingerprint()
    # the deliberate breakage: any non-empty value must move it, and two
    # different values must not collide
    a = dataclasses.replace(cfg, energy_model_rev="2026-09-12-neurosim-adders")
    b = dataclasses.replace(cfg, energy_model_rev="something-else")
    assert a.fingerprint() != base.fingerprint(), "a set revision must cold the cache"
    assert b.fingerprint() != base.fingerprint()
    assert a.fingerprint() != b.fingerprint(), "two revisions must not share a cache"
    # and it must be the ONLY thing that moved -- an empty string is not a value
    assert dataclasses.replace(cfg, energy_model_rev="").fingerprint() == base.fingerprint()
    return f"empty preserves {base.fingerprint()}; set -> {a.fingerprint()}"


def test_the_wrong_sibling_guard_two_arms_identical_yaml_different_fingerprints():
    """prompt_6 RULE 4.4.5, defences 1 and 2. recon2 and recon4 declare
    BYTE-IDENTICAL architecture YAML (datawidth 4 on filter_glb, 8 on
    weights_spad) and differ only in the ERT toll. They must land in
    differently NAMED directories and at different fingerprints; the reference
    arm, which has no toll, must hash exactly as it did before the ERT existed.
    """
    try:
        from eccenergy import archs
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"archs unavailable: {exc}")
    from eccenergy import paths as pathsmod
    ref, r2, r4 = _ert_cfgs()
    # the arm resolved its own datawidth half
    assert (r2.weight_datawidth, r2.weight_datawidth_levels) == (4, ("filter_glb",)), r2
    assert (r4.weight_datawidth, r4.weight_datawidth_levels) == (4, ("filter_glb",)), r4
    t2 = archs._patched_text(_P2_ARCH, r2, quiet=True)
    t4 = archs._patched_text(_P2_ARCH, r4, quiet=True)
    assert t2 == t4, "the two arms are supposed to differ ONLY in the ERT"
    v2, v4 = archs.effective_variant(_P2_ARCH, r2), archs.effective_variant(_P2_ARCH, r4)
    f2, f4 = archs.arch_fingerprint(_P2_ARCH, r2), archs.arch_fingerprint(_P2_ARCH, r4)
    assert v2 != v4 and "ert-recon2-filter_glb-read" in v2 and "ert-recon4-weights_spad-write" in v4, (v2, v4)
    assert f2 != f4, f"identical YAML, different ERT, SAME fingerprint {f2}"
    d2 = pathsmod.Results(r2).mapper_cache(_P2_ARCH, v2, f2, create=False)
    d4 = pathsmod.Results(r4).mapper_cache(_P2_ARCH, v4, f4, create=False)
    assert d2 != d4 and d2.parent != d4.parent, (d2, d4)
    # the slug the config computes and the slug archs computes agree
    assert r2.arch_variant_slug == v2 and r4.arch_variant_slug == v4
    # the reference has no toll and no `ert` part anywhere
    assert archs.ert_bump(_P2_ARCH, ref) is None
    assert "ert-" not in archs.effective_variant(_P2_ARCH, ref)
    assert archs.arch_fingerprint(_P2_ARCH, ref) not in (f2, f4)
    # BREAKAGE: the same toll spelled twice is ONE architecture
    import dataclasses
    again = dataclasses.replace(ref, recon_ert_arm="recon2")
    assert archs.arch_fingerprint(_P2_ARCH, again) == f2
    # and the fingerprint tracks the DELTA itself: codeword charging makes
    # E_w the whole incremental figure, so the toll moves on IDENTICAL YAML
    other = dataclasses.replace(ref, recon_granularity="codeword", recon_ert_arm="recon2")
    assert archs._patched_text(_P2_ARCH, other, quiet=True) == t2
    assert archs.effective_variant(_P2_ARCH, other) == v2
    assert archs.arch_fingerprint(_P2_ARCH, other) != f2, "the delta is not in the hash"


def test_the_ert_bump_is_recomputed_from_the_patched_arch_and_the_dc_table():
    """prompt_6 5.1's table, recomputed: recon2 bumps filter_glb.read by
    E_w x block_size, recon4 bumps weights_spad.write likewise, and both bump
    leak by the DC idle term AT THIS DESIGN'S CLOCK.

    THE LEAK ROW IS PER CYCLE, AND SINCE prompt_7 C1.5 THE CYCLE IS THE
    DESIGN'S. Design Compiler measured 2.8310811 pJ/cycle for BCH(63,30) at a
    1 ns clock; Eyeriss v1 runs at the published 200 MHz, so the engine burns
    5 ns of clock power per cycle and the row is 14.1554055 -- which is the
    worked example env.sh section 6's TRAP 2 spells out. Both clocks are
    asserted below, because pinning only one of them is how a factor of five
    hides.

    The BLOCK SIZES are read off the patched YAML, not pinned, because they
    move with THE WIDTH TABLE: at BCH(63,30) the narrowed `filter_glb` is
    384 b / 4 b = 96 weights per word and `weights_spad` 96 b / 8 b = 12.
    Pinning them made this test assert one arch revision rather than the rule.
    """
    try:
        import dataclasses as _dc
        from eccenergy import archs
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"archs unavailable: {exc}")
    _, r2, r4 = _ert_cfgs()
    b2, b4 = archs.ert_bump(_P2_ARCH, r2), archs.ert_bump(_P2_ARCH, r4)
    geo2 = archs.patched_weight_geometry(_P2_ARCH, r2)
    geo4 = archs.patched_weight_geometry(_P2_ARCH, r4)
    assert (b2["level"], b2["action"], b2["counter"], b2["block_size"]) == (
        "filter_glb", "read", "reads",
        geo2["filter_glb"]["weights_per_word"]), (b2, geo2["filter_glb"])
    assert (b4["level"], b4["action"], b4["counter"], b4["block_size"]) == (
        "weights_spad", "write", "fills",
        geo4["weights_spad"]["weights_per_word"]), (b4, geo4["weights_spad"])
    # THE RULE, not the literals: the toll is E_w x the level's own block size.
    assert abs(b2["access_delta_pj"]
               - b2["e_w_pj"] * geo2["filter_glb"]["weights_per_word"]) < 1e-9, b2
    assert abs(b4["access_delta_pj"]
               - b4["e_w_pj"] * geo4["weights_spad"]["weights_per_word"]) < 1e-9, b4
    # THE RULE: the leak row is the DC idle term x this design's clock / 1 ns.
    scale = r2.dc_idle_scale()
    assert b2["leak_delta_pj"] == b4["leak_delta_pj"], (b2, b4)
    assert abs(b2["leak_delta_pj"] - 2.8310811 * scale) < 1e-9, (b2, scale)
    # and both ends of it, spelled out: 1 ns is what DC measured, 5 ns is what
    # eyeriss_like_wglb runs at (prompt_7 C1.5, env.sh section 6 TRAP 2).
    at_1ns = archs.ert_bump(_P2_ARCH, _dc.replace(r2, arch_clock_mhz={}))
    assert abs(at_1ns["leak_delta_pj"] - 2.8310811) < 1e-9, at_1ns
    assert abs(b2["leak_delta_pj"] - 14.1554055) < 1e-6, (b2, "200 MHz")
    assert abs(b2["e_w_pj"] - 0.175060) < 1e-6 and b2["e_w_pj"] == b4["e_w_pj"]
    assert b2["narrow_levels"] == b4["narrow_levels"] == ["filter_glb"]
    # THE WIDTH TABLE, per level: the NARROWED GLB takes the q=4 width (96 x 4)
    # and the spad keeps the 8-bit width (96). Each is legal for what IT holds;
    # neither has to be legal for the other.
    assert (b2["level_width"], b2["level_datawidth"]) == (384, 4)
    assert (b4["level_width"], b4["level_datawidth"]) == (96, 8), \
        "weights_spad must stay 8-bit on recon4"


def _fake_ert_entry(tmp, bump, base_pj=None, tamper=0.0):
    """A cache entry as `timeloop.Mapper` writes one for an ERT arm: sidecar
    with the bump record, and the stored (patched) ERT beside it."""
    import json
    from eccenergy import timeloop as tl
    base_pj = base_pj or {"read": 2.75566, "write": 4.29165, "update": 4.2, "leak": 0.00010256}
    d = pathlib.Path(tmp) / "C128_M256_R3_S3_P14_Q14_ws2_hs2"
    d.mkdir(parents=True, exist_ok=True)
    level = bump["level"]
    doc = {"ERT": {"version": "0.4", "tables": [
        {"name": f"system_top_level.{level}[1..1]",
         "actions": [{"name": a, "arguments": {}, "energy": e} for a, e in base_pj.items()]},
        {"name": "system_top_level.ifmap_glb[1..1]",
         "actions": [{"name": "read", "arguments": {}, "energy": 23.4862},
                     {"name": "leak", "arguments": {}, "energy": 0.00136375}]}]}}
    patched = tl.patched_ert(doc, tl.ert_changes(bump))
    if tamper:
        for t in patched["ERT"]["tables"]:
            for a in t["actions"]:
                if tl.ert_level_of(t["name"]) == level and a["name"] == bump["action"]:
                    a["energy"] += tamper
    tl.write_yaml(d / tl.ERT_NAME, patched)
    prices = tl.ert_prices(patched)
    side = {"arch_fingerprint": "deadbeef", "ert_bump": {
        "bump": bump,
        "base_pj": {bump["action"]: base_pj[bump["action"]], "leak": base_pj["leak"]},
        "patched_pj": {bump["action"]: prices[(level, bump["action"])],
                       "leak": prices[(level, "leak")]}}}
    (d / tl.MAPPING_SIDECAR).write_text(json.dumps(side))
    return d, tl.ert_prices(doc)


def test_the_read_back_assertion_accepts_its_own_arm_and_stops_on_a_wrong_one():
    """prompt_6 RULE 4.4.5, defence 3: a cache entry whose ERT does not match
    the bar asking for it STOPS THE RUN and names both."""
    try:
        from eccenergy import archs
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"archs unavailable: {exc}")
    from eccenergy import timeloop as tl
    _, r2, r4 = _ert_cfgs()
    b2, b4 = archs.ert_bump(_P2_ARCH, r2), archs.ert_bump(_P2_ARCH, r4)
    with tempfile.TemporaryDirectory() as tmp:
        d, base = _fake_ert_entry(tmp, b2)
        got = tl.read_back_ert(d, b2)
        assert got["arm"] == "recon2" and any("filter_glb.read" in v for v in got["verified"]), got
        # with the un-bumped table the untouched rows are checked too
        got = tl.read_back_ert(d, b2, base_prices=base)
        assert any("other rows untouched" in v for v in got["verified"]), got
        # BREAKAGE 1: recon4 asks for recon2's entry
        try:
            tl.read_back_ert(d, b4)
        except tl.ErtMismatch as e:
            assert "recon2" in str(e) and "recon4" in str(e), e
        else:
            raise AssertionError("recon4 read recon2's entry without complaint")
        # BREAKAGE 2: the reference bar asks for an ERT entry
        try:
            tl.read_back_ert(d, None)
        except tl.ErtMismatch as e:
            assert "reference" in str(e), e
        else:
            raise AssertionError("the reference read an ERT entry without complaint")
        # BREAKAGE 3: the same arm, but the delta drifted by more than 1e-9
        import dataclasses
        drift = dict(b2, access_delta_pj=b2["access_delta_pj"] * (1 + 1e-6))
        try:
            tl.read_back_ert(d, drift)
        except tl.ErtMismatch as e:
            assert "recon2" in str(e), e
        else:
            raise AssertionError("a 1e-6 relative delta drift was accepted")
        # BREAKAGE 4: an entry with no ERT record at all
        d2 = pathlib.Path(tmp) / "plain"; d2.mkdir()
        import json
        (d2 / tl.MAPPING_SIDECAR).write_text(json.dumps({"arch_fingerprint": "x"}))
        assert tl.read_back_ert(d2, None)["arm"] == "reference"
        try:
            tl.read_back_ert(d2, b2)
        except tl.ErtMismatch:
            pass
        else:
            raise AssertionError("an un-bumped entry was accepted for recon2")
    with tempfile.TemporaryDirectory() as tmp:
        # BREAKAGE 5: the stored table was overwritten after the sidecar was written
        d, _ = _fake_ert_entry(tmp, b2, tamper=1e-6)
        try:
            tl.read_back_ert(d, b2)
        except tl.ErtMismatch as e:
            assert "filter_glb.read" in str(e), e
        else:
            raise AssertionError("a tampered stored ERT was accepted")
    # `same_bump` is what `Mapper._accept_cached` uses
    assert tl.same_bump(b2, dict(b2)) and not tl.same_bump(b2, b4)
    assert tl.same_bump(None, None) and not tl.same_bump(None, b2)


def test_patching_the_real_cached_ert_moves_only_the_two_rows():
    """On the reference entry's own Accelergy table (if it is on disk): the
    recon2 patch changes filter_glb.read and filter_glb.leak by exactly the
    bump and nothing else; a row that does not exist is refused."""
    try:
        from eccenergy import archs
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"archs unavailable: {exc}")
    import yaml
    from eccenergy import paths as pathsmod, timeloop as tl
    ref, r2, _ = _ert_cfgs()
    cache = pathsmod.Results(ref).mapper_cache(
        _P2_ARCH, archs.effective_variant(_P2_ARCH, ref),
        archs.arch_fingerprint(_P2_ARCH, ref), create=False)
    erts = sorted(cache.glob(f"*/{tl.ERT_NAME}"))
    if not erts:
        raise _Skip("no reference cache entry with an ERT on disk")
    doc = yaml.safe_load(erts[0].read_text())
    b2 = archs.ert_bump(_P2_ARCH, r2)
    base, got = tl.ert_prices(doc), tl.ert_prices(tl.patched_ert(doc, tl.ert_changes(b2)))
    assert set(base) == set(got)
    moved = {k for k in base if abs(got[k] - base[k]) > 1e-12}
    assert moved == {("filter_glb", "read"), ("filter_glb", "leak")}, moved
    assert abs((got[("filter_glb", "read")] - base[("filter_glb", "read")]) - b2["access_delta_pj"]) < 1e-9
    assert abs((got[("filter_glb", "leak")] - base[("filter_glb", "leak")]) - b2["leak_delta_pj"]) < 1e-9
    # THE RULE, not a literal band. The toll is `E_w x block_size` and the
    # block size comes from THE WIDTH TABLE, so a band pinned to one arch
    # revision ("prices filter_glb.read at 2.75566, so the arm roughly doubles
    # it") asserts the revision, not the rule -- it broke when the level went
    # from 64 b/4 b to 384 b/4 b (2026-09-12).
    geo = archs.patched_weight_geometry(_P2_ARCH, r2)
    assert abs(b2["access_delta_pj"]
               - b2["e_w_pj"] * geo["filter_glb"]["weights_per_word"]) < 1e-9, b2
    assert got[("filter_glb", "read")] > base[("filter_glb", "read")] > 0, (base, got)
    try:
        tl.patched_ert(doc, {("filter_glb", "no_such_action"): ("add", 1.0)})
    except ValueError:
        pass
    else:
        raise AssertionError("a change matching no row was silently dropped")


def test_build_stacks_does_not_narrow_a_level_the_mapper_already_narrowed():
    """prompt_6 RULE 1, the FOURTH site: `build_stacks()`'s recon column scales
    on-chip weight energy by K/N. Fed a q-bit plan's `Raw` (a level row at
    Word bits == q) it must leave that level's weight energy alone; an 8-bit
    row is scaled as before; a row at neither width stops; a row with no
    measurement (an older record, or a network row) is the evaluator's."""
    try:
        import pandas as pd
        from eccenergy import ecc
        from eccenergy.energy import Raw
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"pandas/ecc unavailable: {exc}")
    cfg, _, _ = _p2_cfgs()                          # BCH(63,30): q = 4, K/N = 30/63
    from eccenergy.energy import plot_cats
    cats = plot_cats(cfg)
    glb, spad = "Global buffer", "Local (spads/RF)"
    assert glb in cats and spad in cats, cats
    base = pd.Series({c: 0.0 for c in cats}); base_w = base.copy(); base_i = base.copy()
    base[glb], base_w[glb] = 1000.0, 600.0          # 600 of weight energy in the GLB
    base[spad], base_w[spad] = 500.0, 200.0
    base["DRAM"], base_w["DRAM"] = 4000.0, 4000.0
    base["Compute"] = 9000.0

    def raw_with(levels):
        r = Raw(base.copy(), base_w.copy(), base_i.copy(), 4000.0, 1000.0, 1, 0, 1000,
                per_layer=[], levels=levels, cycles=1000)
        return r

    def recon_col(levels):
        # idle 0 here: this test is about the on-chip categories (RULE 1);
        # the Reconstruction row (RULE 3) has its own test in test_baseline_dram
        return ecc.build_stacks(cfg, raw_with(levels), 1.0, recon_idle_pj=0.0)["recon"]

    kn = 30 / 63
    # 8-bit plan: both on-chip weight shares scale by K/N
    eight = recon_col([{"level": "filter_glb", "dataspace": "Weights", "category": glb,
                        "word_bits": 8, "energy_pJ": 600.0},
                       {"level": "weights_spad", "dataspace": "Weights", "category": spad,
                        "word_bits": 8, "energy_pJ": 200.0}])
    assert abs(eight[glb] - (400.0 + 600.0 * kn)) < 1e-9, eight[glb]
    assert abs(eight[spad] - (300.0 + 200.0 * kn)) < 1e-9, eight[spad]
    # no measurement at all (a record older than the field): identical to 8-bit
    legacy = recon_col([{"level": "filter_glb", "dataspace": "Weights", "category": glb,
                         "energy_pJ": 600.0}])
    assert abs(legacy[glb] - eight[glb]) < 1e-9 and abs(legacy[spad] - eight[spad]) < 1e-9
    # q-bit plan on the GLB only (an ERT arm): the GLB's weight energy is left
    # as Timeloop billed it, the spad (still 8-bit) is scaled
    q_glb = recon_col([{"level": "filter_glb", "dataspace": "Weights", "category": glb,
                        "word_bits": 4, "energy_pJ": 600.0},
                       {"level": "weights_spad", "dataspace": "Weights", "category": spad,
                        "word_bits": 8, "energy_pJ": 200.0}])
    assert abs(q_glb[glb] - 1000.0) < 1e-9, q_glb[glb]
    assert abs(q_glb[spad] - eight[spad]) < 1e-9
    # the other columns never move: the baseline and embedded arms are 8-bit by
    # construction and do not read the level rows
    df8 = ecc.build_stacks(cfg, raw_with([]), 1.0, recon_idle_pj=0.0)
    dfq = ecc.build_stacks(cfg, raw_with([{"level": "filter_glb", "dataspace": "Weights",
                                            "category": glb, "word_bits": 4,
                                            "energy_pJ": 600.0}]), 1.0, recon_idle_pj=0.0)
    for col in ("baseline", "embedded"):
        assert (df8[col] - dfq[col]).abs().max() < 1e-9, col
    # BREAKAGE: Word bits 5 is neither 8 nor q
    try:
        recon_col([{"level": "filter_glb", "dataspace": "Weights", "category": glb,
                    "word_bits": 5, "energy_pJ": 600.0}])
    except ValueError as e:
        assert "disagree" in str(e), e
    else:
        raise AssertionError("a Word bits 5 level was scaled without complaint")
    # a network row carries no Word bits and is never the mapper's
    noc = recon_col([{"level": "NoC: filter_glb <==> PE_column", "dataspace": "Weights",
                      "category": "NoC", "word_bits": None, "energy_pJ": 50.0}])
    assert abs(noc[glb] - eight[glb]) < 1e-9


def test_the_capacity_target_is_8_over_q_and_every_code_passes_its_own():
    """prompt_6 6, the table: a quantised weight is a whole number of bits, so
    the room a re-planned arm delivers is weight_bits/q, never N/K. Against N/K
    two codes (K=51, K=36) could not run at all and the other four passed by
    luck; against 8/q every code passes exactly. Each mutation checks that the
    old target would still refuse the two."""
    try:
        from eccenergy import recon
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"recon unavailable: {exc}")
    import dataclasses
    cfg, _, _ = _p2_cfgs()
    table = {57: (7, 1.143), 51: (6, 1.333), 45: (6, 1.333), 39: (5, 1.600),
             36: (5, 1.600), 30: (4, 2.000)}
    for k, (q_want, room) in table.items():
        c = dataclasses.replace(cfg, const_k=k)
        want, q = recon.capacity_target(c)
        assert q == q_want, (k, q, q_want)
        assert abs(want - room) < 5e-4, (k, want, room)
        # what the arm really delivers, 8/q exactly, passes ITS target...
        delivered = 8.0 / q
        assert abs(delivered - want) <= 0.05 * want, k
        # ...and would have been refused against N/K at K=51 and K=36
        nk = 63 / k
        off = abs(delivered - nk) > 0.05 * nk
        assert off == (k in (51, 36)), (k, delivered, nk)
    # BREAKAGE: the ideal rate is not the target -- at BCH(63,30) N/K = 2.1
    want, q = recon.capacity_target(cfg)
    assert q == 4 and want == 2.0 and abs(63 / 30 - want) > 0.05


def test_ecc_recon_layer_seeds_the_scope_and_the_optimiser_stem_stays_fixed():
    """prompt_6 8.2 / 9, on env.sh itself: under ECC_RECON_MODELING=1 the layer
    scope is ECC_RECON_LAYER -- a name selects that layer, `all` selects every
    layer -- and the optimiser stem is `ReconSweep_optimiser__<model>` either
    way (per model since 2026-09-11), while the fixed-mapping study keeps its
    layer suffix on a scoped run."""
    import subprocess
    from eccenergy.paths import ROOT
    env_sh = pathlib.Path(ROOT) / "env.sh"
    if not env_sh.is_file():
        raise _Skip("env.sh not found")

    def resolve(**env):
        import os
        e = {k: v for k, v in os.environ.items() if not k.startswith("ECC_") and k != "RECON_OPTIMIZER"}
        e.update(env)
        out = subprocess.run(
            ["bash", "-c", "source ./env.sh >/dev/null 2>&1; "
                           "printf '%s|%s|%s|%s' \"$ECC_LAYERS\" \"$ECC_STEM\" \"$ECC_RECON_MODEL\" \"$ECC_RECON_LAYER\""],
            cwd=str(ROOT), env=e, capture_output=True, text=True, check=True).stdout
        layers, stem, model, rlayer = out.split("|")
        return layers, stem, model, rlayer

    # env.sh's OWN default point, whatever it is today: the scope follows
    # ECC_RECON_LAYER and the stem names the model
    layers, stem, model, rlayer = resolve(ECC_RECON_MODELING="1")
    assert model and stem == f"ReconSweep_optimiser__{model}", (stem, model)
    if rlayer.lower() in ("", "all", "full"):
        assert layers == "", (layers, rlayer)
    else:
        assert layers == rlayer, (layers, rlayer)
    layers, stem, model, _ = resolve(ECC_RECON_MODELING="1", ECC_RECON_LAYER="layer4.1.conv2",
                                     ECC_RECON_MODEL="resnet18")
    assert layers == "layer4.1.conv2" and stem == "ReconSweep_optimiser__resnet18", (layers, stem)
    layers, stem, model, _ = resolve(ECC_RECON_MODELING="1", ECC_RECON_LAYER="all",
                                     ECC_RECON_MODEL="resnet18")
    assert layers == "" and stem == "ReconSweep_optimiser__resnet18", (layers, stem)  # every layer
    # the model is in the name: two networks never overwrite each other's figure
    layers, stem, model, _ = resolve(ECC_RECON_MODELING="1", ECC_RECON_LAYER="all",
                                     ECC_RECON_MODEL="mobilenet_v2")
    assert layers == "" and stem == "ReconSweep_optimiser__mobilenet_v2", (layers, stem)
    # ECC_LAYERS is NOT the knob under the placement study: it is overwritten
    layers, _, _, _ = resolve(ECC_RECON_MODELING="1", ECC_LAYERS="conv1", ECC_RECON_LAYER="layer3.0.conv1")
    assert layers != "conv1", layers
    # the fixed-mapping study keeps the layer suffix (an empty stem) when scoped
    layers, stem, _, _ = resolve(ECC_RECON_MODELING="1", RECON_OPTIMIZER="False", ECC_PHASE="Pre",
                                 ECC_RECON_LAYER="layer3.0.conv1")
    assert layers and stem == "", (layers, stem)
    layers, stem, model, _ = resolve(ECC_RECON_MODELING="1", RECON_OPTIMIZER="False", ECC_PHASE="Pre",
                                     ECC_RECON_LAYER="all")
    assert layers == "" and stem == f"ReconSweep__{model}", (layers, stem)


def test_the_width_table_holds_total_bits_and_puts_the_glb_at_four_times():
    """PROMPT_2'S WIDTH TABLE, and the three things that make it right.

    * Each arm's width suits ITS OWN datawidth: 96 b for an 8-bit level,
      95 b at q=5, 98 b at q=7. No arm has to be legal for another arm's
      datawidth, because no arm is ever mapped on another arm's silicon.
    * The GLB word is 4x the scratchpad's -- the ratio Eyeriss v1's published
      geometry already has (224 x 16 b spad, 512-b x 64-b GLB banks).
    * Each level's DEPTH is renormalised AT THE BASE WIDTH so its TOTAL BITS
      do not move AND every arm shares it. CACTI is handed depth and width, so
      a width change that grew the array would be a bigger array smuggled in
      as a word reshape. Checked to within one word of rounding.

    It is AUTOMATIC: no knob is set anywhere in this test, because there is no
    knob. That is the regression -- the lookup existed from 2026-09-11 and
    nothing reached for it until `map_ert_arms.sh` died on it at K=39.
    """
    try:
        from eccenergy import archs, code_widths
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"archs unavailable: {exc}")
    import dataclasses
    cfg, emb, rec = _p2_cfgs()
    src_geom = archs.weight_capacity_levels(_P2_ARCH, cfg)
    published = {r["level"]: r for r in src_geom}
    got = archs.patched_weight_geometry(_P2_ARCH, emb)
    base = code_widths.BASE_WIDTH
    spad = list(got)[-1]
    for level, v in got.items():
        want_w = base if level == spad else base * 4
        assert v["width"] == want_w, (level, v, want_w)
        assert v["datawidth"] == 8, (level, v)
        assert v["width"] % v["datawidth"] == 0, (level, v)
    # the q=4 arm keeps the SAME depth and takes its own width (96 at q=4)
    narrow = archs.patched_weight_geometry(_P2_ARCH, rec)
    for level in got:
        assert narrow[level]["depth"] == got[level]["depth"], (
            f"{level}: the arms must share a depth -- "
            f"{got[level]['depth']} vs {narrow[level]['depth']}")
        assert narrow[level]["width"] % narrow[level]["datawidth"] == 0, narrow[level]
    # TOTAL BITS held, to within one word of rounding, against the SOURCE yaml
    for level, v in got.items():
        p = published.get(level)
        if not p:
            continue
        before = p["depth"] * p["weights_per_word"] * 8
        after = v["depth"] * v["width"]
        assert abs(after - before) <= v["width"], (
            f"{level}: {before:,} bits became {after:,} -- a width change must "
            f"reshape the word, not resize the array")

def test_the_cross_arm_width_rule_is_withdrawn_and_must_not_come_back():
    """WITHDRAWN 2026-09-12, and this is its headstone.

    There used to be a test here asserting that a width must divide BOTH the
    code's `q` AND `ECC_WEIGHT_BITS`, on the grounds that both arms share one
    declared width. THE ARMS DO NOT SHARE A WIDTH. prompt_2's 98 (q=7) and 95
    (q=5) are correct exactly as written; `98 % 8 = 2` and `95 % 8 = 7` are
    irrelevant, because the 8-bit arm is mapped at 96 and never at 98 or 95.
    `timeloop-mapper`'s assertion is per level, per mapper run, and one mapper
    run maps one arm.

    The rule cost a real wave: it replaced the table with `lcm(q, 8)`
    (56 / 24 / 40), which made the 8-BIT REFERENCE move between codes, and
    that is where BCH(63,39)'s spurious 37.69 % came from (FINDINGS 2.4b).
    """
    try:
        from eccenergy import code_widths
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"code_widths unavailable: {exc}")
    assert code_widths.declared_width(7) == 98, "prompt_2's q=7 row moved"
    assert code_widths.declared_width(5) == 95, "prompt_2's q=5 row moved"
    assert code_widths.declared_width(8) == 96, "the 8-bit arm moved"
    # There is no knob to get this wrong with any more, either.
    assert not hasattr(config.load_config(), "weight_width"), \
        "Config carries a weight_width field again"

def test_the_onchip_narrowing_is_applied_exactly_once():
    """Both the mapper and the evaluator can narrow an on-chip weight now.

    `stream` scales every reduced stage by k/n in the EVALUATOR;
    `ECC_WEIGHT_DATAWIDTH` delivers the same saving inside the MAPPER's own
    Timeloop number. Apply both and the on-chip saving is SQUARED. prompt_2's
    resolution is `aligned`, and this is the assertion that enforces it.
    """
    try:
        from eccenergy import recon
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"recon unavailable: {exc}")
    import dataclasses
    cfg, _, _ = _p2_cfgs()
    both = dataclasses.replace(cfg, recon_packing="stream", weight_datawidth=4)
    try:
        recon.assert_onchip_narrowing_once(both)
    except ValueError as e:
        assert "TWICE" in str(e) and "SQUARED" in str(e), str(e)
    else:
        raise AssertionError(
            "stream packing beside a mapper-side datawidth was accepted; the "
            "on-chip saving would be counted twice")
    # the prompt_2 configuration passes
    good = dataclasses.replace(cfg, recon_packing="aligned", weight_datawidth=4)
    assert recon.assert_onchip_narrowing_once(good)["n_sites"] == 1
    # and a datawidth that disagrees with the code is refused
    wrong = dataclasses.replace(cfg, recon_packing="aligned", weight_datawidth=5)
    try:
        recon.assert_onchip_narrowing_once(wrong)
    except ValueError as e:
        assert "DISAGREE" in str(e), str(e)
    else:
        raise AssertionError("datawidth 5 was accepted at BCH(63,30), whose "
                             "ceil(8*K/N) is 4")


def test_aligned_without_a_mapper_datawidth_is_the_old_bound_not_an_error():
    """The pessimistic bound predates prompt_2 and must stay runnable."""
    try:
        from eccenergy import recon
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"recon unavailable: {exc}")
    import dataclasses
    cfg, _, _ = _p2_cfgs()
    old = dataclasses.replace(cfg, recon_packing="aligned", weight_datawidth=None)
    audit = recon.onchip_narrowing_audit(old)
    assert audit["ok"] and audit["n_sites"] == 0 and "note" in audit, audit


def test_eyeriss_v1_wglb_has_a_weight_path_and_every_stage_has_a_boundary():
    """The blocking work prompt_2 names: v1 IS `eyeriss_like_wglb` now.

    `weight_path()` refuses when a weight-carrying level goes unclaimed, so
    `filter_glb` needs a Stage AND a placement. Without the placement, every
    boundary below it would report its own saving while showing the GLB at
    full width -- the whole list understated, with nothing saying so.
    """
    try:
        from eccenergy import recon
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"recon unavailable: {exc}")
    assert _P2_ARCH in recon.WEIGHT_PATHS, "no weight path registered"
    assert _P2_ARCH in recon.PLACEMENTS, "no placements registered"
    keys = [s.key for s in recon.WEIGHT_PATHS[_P2_ARCH]]
    assert "filter_glb" in keys, keys
    ok, detail = recon.validate_placement_space(_P2_ARCH)
    assert ok, detail.get("violations")
    # five boundaries: one more than eyeriss_like, because of the GLB
    assert len(recon.PLACEMENTS[_P2_ARCH]) == len(recon.PLACEMENTS["eyeriss_like"]) + 1


def test_the_eyeriss_v1_bracket_pair_is_retired():
    """Collapsing the two v1 files to ONE design makes the pair
    self-referential: it would stamp every manifest with a caveat that is no
    longer true, and ask the run to plot a retired file beside the live one.
    """
    try:
        from eccenergy import config
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"config unavailable: {exc}")
    assert "eyeriss_like" not in config.BRACKET_PAIRS, config.BRACKET_PAIRS
    assert "eyeriss_like_wglb" not in config.BRACKET_PAIRS, config.BRACKET_PAIRS


def test_the_two_recon_terms_sum_to_the_margin_they_decompose():
    """PROMPT_2 ITEM 4: the flat packing discount and the differential-refetch
    term are SEPARATE columns, and their difference is the margin.

    The packing discount applies to every on-chip weight access regardless of
    mapping, so Recon wins monotonically at every depth even on a
    byte-identical loop nest. Summed with the refetch term the table looks
    like a win everywhere and says nothing about which memory size matters --
    which is the whole question. This checks the split is a decomposition and
    not two independently-computed numbers that can drift apart.
    """
    # emb_pJ at 8b, rec_pJ at 4b: ratio 0.5.
    for emb_pj, rec_pj, ratio in ((2_636_267.0, 383_917.0, 0.5),
                                  (18_463_326.0, 4_156_115.0, 0.5),
                                  (132_120_576.0, 75_497_472.0, 1.0)):
        pack = emb_pj * (1.0 - ratio)
        refet = rec_pj - emb_pj * ratio
        margin = rec_pj - emb_pj
        assert abs((refet - pack) - margin) <= 1e-6 * max(1.0, abs(margin)), (
            emb_pj, rec_pj, ratio, pack, refet, margin)
    # DRAM carries NO packing discount: its datawidth is 8 on both arms.
    assert 132_120_576.0 * (1.0 - 1.0) == 0.0


def test_the_level_parser_reads_geometry_and_energy_off_the_stats_file():
    """`declared_depth`, `width`, `datawidth`, `weights_per_word` and
    `level_pJ` all come from the run that produced the number, not from a YAML
    that may since have been edited. That is what makes the fairness rule
    checkable on the ROWS.
    """
    from eccenergy.experiments import dilation
    text = _stats()
    rows = dilation._weight_level_rows(text, ("weights_spad",))
    assert len(rows) == 1, rows
    r = rows[0]
    for key in ("datawidth", "weights_per_word", "width", "declared_depth",
                "energy_pJ", "vector_access_pJ", "reads", "fills"):
        assert key in r, f"{key} missing from the level row"
    assert r["width"] == r["datawidth"] * r["weights_per_word"], r
    if r["weights_per_word"]:
        assert r["declared_depth"] == r["capacity"] // r["weights_per_word"], r

def main():
    print("eccenergy capacity-dilation (Task 4 step 1) tests")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    print()
    if SKIPPED:
        print(f"{len(SKIPPED)} skipped: {', '.join(SKIPPED)}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
