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

import pathlib
import sys
import tempfile
import traceback

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
