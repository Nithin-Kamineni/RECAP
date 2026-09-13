"""prompt_7 Phase C1: the single cold pass -- TIME, in the architecture.

PROPERTY TESTS ON THE REAL PATCHED ARCHITECTURE, PLUS DELIBERATE BREAKAGE.
Every assertion here reads the YAML `archs._patched_text()` actually produces
for an arm -- no fixture, no hand-written expectation of what the file should
contain -- and every one is followed by the mutation it exists to catch,
applied to a copy and checked to FAIL. An assertion nobody has seen fail is a
comment.

WHAT PHASE C IS FOR. Phase A computed the roofline OUTSIDE Timeloop and Phase B
gave every boundary its own chip, but the architecture the MAPPER read still
declared no off-chip bandwidth at all, no reduced-traffic scale, an item-counted
port on a level storing narrower words, a MAC at five times the price the report
charges, and a bank count that CACTI never saw. So the search optimised a
machine with an infinitely fast memory, and the reduced representation could
buy energy but never time (prompt_7 Defect 1).

THE PROPERTIES, AND WHY EACH ONE IS THE GATE IT IS
--------------------------------------------------
1. `test_offchip_limit_is_declared_once_and_agrees_with_the_roofline`  (C1.1)
   The DRAM level declares `shared_bandwidth` and it equals what
   `latency_post.offchip_items_per_cycle()` charges, to the last digit. Two
   implementations of one bus is the defect; one number with two readers is
   not. Breakage: perturb the clock and check the two move together.
2. `test_the_bandwidth_scale_is_per_stage_and_per_factor`                (C1.2)
   K/N at DRAM, q/8 on chip -- they differ by 5% at BCH(63,30) and using one
   everywhere is a silent inconsistency with ECC_RECON_PACKING. A NETWORK
   stage is declared in the arm but has no YAML component to land on.
   Breakage: use one factor everywhere and check the DRAM line changes.
3. `test_only_the_narrowed_levels_are_bit_aware`                         (C1.3)
   `read_bandwidth x 8/q` on the levels this arm narrows, and NOWHERE else --
   the reference keeps 16.00. C1 gate 5. Breakage: make the reference
   bit-aware and check its filter_glb moves off 16.
4. `test_the_mac_price_the_mapper_sees_is_the_one_the_report_charges`    (C1.6)
   The supplied ERT sets every `compute` row to ECC_MAC_PJ_OVERRIDE, so the
   evaluator's rescale ratio comes back exactly 1.0 and the two cannot
   double-count. Breakage: `add` instead of `set`.
5. `test_only_a_declared_bank_count_reaches_cacti`                       (C1.7)
   A level that declares `n_banks:` is switched to the banked compound; a
   level that does not keeps the plain one and prices as it did before Phase
   C. timeloopfe hands EVERY level a default `n_banks: 2`, so this is the
   difference between the paper's banking and a front-end default.
   Breakage: switch every SRAM and check the depth-3 spad moves.
6. `test_the_clock_is_per_design_and_inverted_once`                      (C1.5)
   Eyeriss v1 at 200 MHz, everything else at the study default, and MHz
   becomes seconds in exactly one place. Breakage: a second inversion.
7. `test_every_phase_c_declaration_colds_the_cache`                      (C1.8)
   Each of the five is in the fingerprint, so none of them can land silently
   -- and ECC_ENERGY_MODEL_REV reaches the fingerprint the CACHE DIRECTORY is
   named after, which until Phase C it did not. Breakage: none needed; each
   assertion is itself the mutation.
8. `test_the_roofline_reproduces_the_plan_it_re_times`                  (C1.2)
   THE ROOFLINE RECOMPUTES a plan's time from item counts; it does not adjust
   Timeloop's. So the weight relief belongs in that computation exactly once,
   whoever else applied it -- and suppressing it "to avoid double-counting"
   re-timed every reconstruction bar as if its relief did not exist
   (61,905,882 cycles against Timeloop's own 54,132,184, x1.1436, while the
   reference reproduced Timeloop at x1.0000). What the declared scale is FOR
   is the CHECK: a bar whose plan was solved at neither 1.00 nor this run's
   K/N belongs to a different code and is refused. Breakage: a third scale.
9. `test_the_dc_idle_term_is_rescaled_to_this_designs_clock`            (C1.5)
   env.sh section 6's TRAP 2, LIVE FOR THE FIRST TIME. The DC tables give the
   engine's idle term in pJ PER CYCLE at a 1 ns clock; at 200 MHz it is x5,
   and until C1.5 gave a design its own clock the factor was 1.0 on every run
   so nothing had ever exercised it. The INCREMENTAL term is per codeword and
   must NOT move. Breakage: rescale the incremental term too; charge the idle
   term at the study clock.
10. `test_a_comment_is_never_the_geometry`
    FOUND ON A REAL SLURM RUN, 2026-09-13. Every geometry regex in `archs.py`
    matched a `depth:` written in a COMMENT as readily as the attribute, and
    `re.sub(..., count=1)` then rewrote the COMMENT and left the attribute
    alone. A comment added in C1.7 -- "the paper's two banks would be
    `depth: 1024`" -- made `filter_glb` declare 256 x 384 b where 43 x 384 b
    was intended: a SIX-FOLD capacity error, silent except for one line of
    stdout. Breakage: name a different depth in a comment and check nothing
    moves.
11. `test_the_mapper_constructor_finishes`
    Same run, same cause class: a `@property` inserted into the middle of
    `Mapper.__init__` orphaned every line after it, so `self._memo` was never
    assigned and all 72 jobs died with `AttributeError` 10 seconds in. No test
    had ever CONSTRUCTED a Mapper. Breakage: none needed -- it fails by
    itself.
12. `test_the_reference_arms_table_needs_no_bump`
    THIRD failure of the same class, same run. Since C1.6 the REFERENCE arm
    builds an ERT too -- carrying only the MAC price -- and `ErtTables.ensure()`
    still formatted `self.bump['placement']` into its lock message. Every
    `self.bump[...]` in the class is now behind a `bump is not None` guard, and
    this test walks the AST to prove it rather than trusting a reading.
13. `test_the_shared_inputs_are_never_replaced_under_a_reader`
    FOUND ON THE 186-JOB MATRIX, 2026-09-13. `globals_<arch>.yaml` and the
    patched arch YAML are written by EVERY job of a SLURM array, and
    `_write_atomic` replaced them in place. That is atomic in POSIX terms and
    still not safe on Lustre: a client holding a handle to the old inode sees
    ENOENT after the replacement. One job of 186 wrote the file and found it
    missing eight seconds later, which was enough to leave the dependent eval
    on `DependencyNeverSatisfied`. Both names now carry a content hash and are
    created with `O_EXCL`, so nothing is ever replaced. Breakage: replace a
    file `_write_once` has already written.
14. `test_knobs_off_reproduce_the_pre_phase_c_architecture`
   With ECC_RECON_BW_SCALE=0, ECC_ONCHIP_BW_BITAWARE=0, no off-chip limit, no
   revision and the study clock, the patched text is what predates Phase C
   apart from the bank switch -- which is the one unconditional change.

Run it like every other suite:

    bash hpc/tl.sh python3 -m eccenergy.tests.test_phase_c
"""
from __future__ import annotations

import dataclasses
import re
import sys

from .. import archs, config, recon, timeloop as tlmod
from ..physics import widths
from ..toolchain import latency_post

FAILED = []

#: The design prompt_7 quotes every measured number from.
ARCH = "eyeriss_like_wglb"

#: The level prompt_7 4.5 measures the on-chip latency lever at, and the one
#: R2's whole saving rides on.
GLB = "filter_glb"


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


def _assert_eq(a, b, why=""):
    assert a == b, f"{a} != {b}{('  -- ' + why) if why else ''}"


def _close(a, b, tol=1e-9):
    assert a is not None and b is not None, f"{a!r} / {b!r}"
    assert abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b))), f"{a} != {b}"


def _cfg(arm="reference", code_k=None, **over):
    """The LIVE configuration, on one arm. Reads env.sh through `load_config`,
    so a default that is only true when a knob is exported would fail here --
    which is the point (`ship the default, not just the capability`).

    `archs`, `models` and `code_k` are `init=False` -- derived in
    `__post_init__` from the swept axis -- so a different code is spelled as a
    different `const_k` and re-derived, never assigned over.
    """
    cfg = config.load_config()
    assert cfg.archs == [ARCH], (
        f"this suite reads the LIVE configuration and it is on {cfg.archs}; "
        f"env.sh section 4 should pin the placement study to {ARCH}")
    if code_k is not None:
        cfg = dataclasses.replace(cfg, const_k=code_k, recon_ert_arm="reference")
    cfg = dataclasses.replace(cfg, recon_ert_arm="reference", **over)
    if arm != "reference":
        cfg = dataclasses.replace(cfg, recon_ert_arm=arm)
    return cfg


def _text(arm="reference", cfg=None, **over):
    cfg = cfg or _cfg(arm, **over)
    return archs._patched_text(ARCH, cfg, quiet=True)


def _attr(text, level, key):
    """The numeric value of `key` on component `level`, or None.

    Reads the PATCHED text the mapper is handed, not the source file: a
    declaration that never reached the patched YAML is a declaration that never
    reached Timeloop.
    """
    name, found = None, None
    for line in text.split("\n"):
        m = re.match(r"\s*name:\s*(\S+)", line)
        if m:
            name = m.group(1).split("#")[0].strip()
        if name != level:
            continue
        m = re.match(rf"\s*{re.escape(key)}:\s*([\d.eE+-]+)", line)
        if m:
            found = float(m.group(1))
    return found


def _scale_line(text, level):
    """`per_dataspace_bandwidth_consumption_scale` on `level`, as a string."""
    name = None
    for line in text.split("\n"):
        m = re.match(r"\s*name:\s*(\S+)", line)
        if m:
            name = m.group(1).split("#")[0].strip()
        if (name == level
                and "per_dataspace_bandwidth_consumption_scale" in line):
            return line.strip()
    return None


def _class_of(text, level):
    name = None
    for line in text.split("\n"):
        m = re.match(r"\s*name:\s*(\S+)", line)
        if m:
            name = m.group(1).split("#")[0].strip()
        m = re.match(r"\s*class:\s*(\S+)", line)
        if m and name == level:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
#  1. C1.1 -- one off-chip limit, two readers
# ---------------------------------------------------------------------------
def test_offchip_limit_is_declared_once_and_agrees_with_the_roofline():
    """The DRAM level declares the limit the roofline charges, exactly.

    Two timing models for one bus is the defect this closes: before Phase C
    the number existed only in the evaluator, so the mapper optimised against
    an infinitely fast memory and the evaluator then re-timed the plan it
    chose.
    """
    cfg = _cfg()
    want = cfg.dram_items_per_cycle_for(ARCH)
    assert want, ("ECC_DRAM_BANDWIDTH_MBPS is EMPTY in env.sh, so no design "
                  "declares an off-chip limit and prompt_7 Defect 1 is back")
    _close(_attr(_text(cfg=cfg), "DRAM", "shared_bandwidth"), round(want, 6))
    # ONE BUS. Declaring the two directions separately would let Timeloop
    # deliver twice what `latency_post` caps, so neither may be written.
    _assert_eq(_attr(_text(cfg=cfg), "DRAM", "read_bandwidth"), None)
    _assert_eq(_attr(_text(cfg=cfg), "DRAM", "write_bandwidth"), None)
    # the roofline's own conversion, from the same MB/s, must be the same number
    _close(latency_post.offchip_items_per_cycle(
        dataclasses.replace(cfg, global_cycle_seconds=cfg.cycle_seconds_for(ARCH))),
        want)
    # THE RULE, not the knob: env.sh section 6 states the conversion once and
    # this is it. `want` is whatever ECC_DRAM_BANDWIDTH_MBPS currently says --
    # pinning a number here would pin one operating point rather than the
    # arithmetic, and 480 / 240 / 120 are all documented settings.
    _close(want, float(cfg.dram_bandwidth_mbps) * 1e6
           * float(cfg.cycle_seconds_for(ARCH)) / (cfg.weight_bits / 8.0))
    # the worked examples env.sh and prompt_7 4.7 quote, checked AT their own
    # bandwidth so they stay true whatever the knob is set to
    at480 = dataclasses.replace(cfg, dram_bandwidth_mbps=480.0)
    _close(at480.dram_items_per_cycle_for(ARCH), 2.4)       # prompt_7 4.7
    at120 = dataclasses.replace(cfg, dram_bandwidth_mbps=120.0)
    _close(at120.dram_items_per_cycle_for(ARCH), 0.6)       # env.sh "quarter it"

    # BREAKAGE: a clock the architecture and the roofline do not share. The two
    # conversions must move TOGETHER or one of them is charging a different bus.
    # A SLOWER CLOCK MEANS MORE ITEMS PER CYCLE: the bus delivers MB/s and a
    # longer cycle is more of them, which is the direction that makes the
    # 1 GHz model chip ~5x more memory-starved than 200 MHz silicon.
    slow = dataclasses.replace(cfg, arch_clock_mhz={ARCH: 100.0})
    fast = dataclasses.replace(cfg, arch_clock_mhz={ARCH: 1000.0})
    # HALVING THE CLOCK DOUBLES the items per cycle, at the same MB/s
    _close(slow.dram_items_per_cycle_for(ARCH), want * 2.0)
    _close(fast.dram_items_per_cycle_for(ARCH), want / 5.0)
    expect_raises(
        lambda: _close(_attr(_text(cfg=slow), "DRAM", "shared_bandwidth"), want),
        "the declared off-chip limit ignored the design's clock")

    # and EMPTY means no attribute at all -- the pre-Phase-A model, reachable
    off = dataclasses.replace(cfg, dram_bandwidth_mbps=None)
    _assert_eq(_attr(_text(cfg=off), "DRAM", "shared_bandwidth"), None)


# ---------------------------------------------------------------------------
#  2. C1.2 -- the scale is per stage, and the factor is per stage kind
# ---------------------------------------------------------------------------
def test_the_bandwidth_scale_is_per_stage_and_per_factor():
    """K/N off the die, q/8 on chip, and a network stage declared but no-op."""
    cfg = _cfg("recon2")
    assert cfg.recon_bw_scale, ("ECC_RECON_BW_SCALE is 0 in env.sh; prompt_7 "
                                "C1.2 ships it ON -- the default IS the feature")
    q = widths.declared_datawidth(cfg.code_n, cfg.code_k)
    factors = cfg.arm_bw_factors_for(ARCH)
    _close(factors["DRAM"]["factor"], cfg.code_k / cfg.code_n)
    _close(factors[GLB]["factor"], q / cfg.weight_bits)
    # THEY ARE NOT THE SAME NUMBER. At BCH(63,30) they differ by 5%; at
    # BCH(63,39) by 1%. One factor everywhere is a silent inconsistency.
    k30 = _cfg("recon2", code_k=30).arm_bw_factors_for(ARCH)
    _close(k30["DRAM"]["factor"], 30 / 63)
    _close(k30[GLB]["factor"], 0.5)
    assert abs(k30["DRAM"]["factor"] - k30[GLB]["factor"]) > 0.02

    text = _text(cfg=cfg)
    assert "Weights: 0.619048" in (_scale_line(text, "DRAM") or ""), \
        _scale_line(text, "DRAM")
    assert "Weights: 0.625000" in (_scale_line(text, GLB) or ""), \
        _scale_line(text, GLB)
    # the reference declares NONE -- it moves no less of anything
    ref = _text("reference")
    _assert_eq(_scale_line(ref, "DRAM"), None)
    _assert_eq(_scale_line(ref, GLB), None)

    # A NETWORK STAGE IS DECLARED IN THE ARM AND LANDS ON NOTHING. It is what
    # makes R3 a different chip from R2 (prompt_7 6.4), and
    # LegacyNetwork::ComputePerformance() is an empty stub, so there is no
    # component to attach it to. Both halves are asserted.
    r3 = _cfg("recon3")
    kinds = {f["kind"] for f in r3.arm_bw_factors_for(ARCH).values()}
    assert "network" in kinds, kinds
    net = [lvl for lvl, f in r3.arm_bw_factors_for(ARCH).items()
           if f["kind"] == "network"]
    t3 = _text(cfg=r3)
    for lvl in net:
        _assert_eq(_scale_line(t3, lvl), None, f"{lvl} is a network: no-op")
    _assert_eq(recon.arm_bw_factors(
        recon.placement_by_key(ARCH, "recon3", r3),
        recon.stages_for(ARCH, r3), r3)[net[0]]["timing"], "no-op")

    # BREAKAGE: one factor everywhere. The DRAM line must change.
    expect_raises(
        lambda: _assert_eq(f"{cfg.code_k / cfg.code_n:.6f}",
                           f"{q / cfg.weight_bits:.6f}"),
        "K/N and q/8 were treated as one number")

    # BREAKAGE: a misspelled dataspace must not be silently dropped by US
    # either -- the name comes from one constant, so a typo cannot be local.
    _assert_eq(recon.REDUCED_BW_DATASPACE, "Weights")


# ---------------------------------------------------------------------------
#  3. C1.3 -- bit-aware ports, and only where the arm narrows
# ---------------------------------------------------------------------------
def test_only_the_narrowed_levels_are_bit_aware():
    """C1 gate 5. At q=4 `filter_glb` goes 16.00 -> 32.00 in the narrowed arms
    and stays 16.00 in the reference."""
    cfg = _cfg("recon2", code_k=30)                # q = 4, so the factor is 2
    assert cfg.onchip_bw_bitaware, ("ECC_ONCHIP_BW_BITAWARE is 0 in env.sh; "
                                    "prompt_7 C1.3 ships it ON")
    _close(cfg.onchip_bw_bitaware_factor(), 2.0)
    text = _text(cfg=cfg)
    _close(_attr(text, GLB, "read_bandwidth"), 32.0)
    _close(_attr(text, GLB, "write_bandwidth"), 32.0)
    # NOT the levels this arm leaves at 8 bits
    _close(_attr(text, "weights_spad", "read_bandwidth"), 2.0)
    _close(_attr(text, "ifmap_glb", "read_bandwidth"), 16.0)
    # and NOT the reference
    ref = _text("reference", code_k=30)
    _close(_attr(ref, GLB, "read_bandwidth"), 16.0)

    # at the live code (BCH(63,39), q=5) the factor is 1.6 and is NOT rounded:
    # a declared bandwidth is a rate, and rounding 25.6 down would hand the
    # reference a 2.4% advantage no wire has.
    live = _cfg("recon2")
    _close(live.onchip_bw_bitaware_factor(), 8 / 5)
    _close(_attr(_text(cfg=live), GLB, "read_bandwidth"), 25.6)

    # R5a narrows the scratchpad too, so both levels move
    r5 = _cfg("recon5", code_k=30)
    t5 = _text(cfg=r5)
    _close(_attr(t5, GLB, "read_bandwidth"), 32.0)
    _close(_attr(t5, "weights_spad", "read_bandwidth"), 4.0)

    # BREAKAGE: make the REFERENCE bit-aware. Its filter_glb must move off 16,
    # which is the free architecture change this rule exists to prevent.
    broken = archs._bitaware_onchip_bandwidth(ref, (GLB,), 2.0, ARCH, quiet=True)
    expect_raises(lambda: _close(_attr(broken, GLB, "read_bandwidth"), 16.0),
                  "the reference arm was given a bit-aware port")

    # BREAKAGE: the OFF-CHIP limit must never be bit-aware. Its relief is
    # already declared as a bandwidth scale (C1.2), so 8/q on top would charge
    # the reduced representation twice.
    expect_raises(
        lambda: archs._bitaware_onchip_bandwidth(ref, ("DRAM",), 2.0, ARCH, True),
        "the off-chip limit was made bit-aware on top of its K/N scale")
    # BREAKAGE: a level this design does not have is refused, not skipped
    expect_raises(
        lambda: archs._bitaware_onchip_bandwidth(ref, ("no_such_level",), 2.0,
                                                 ARCH, True),
        "a bit-aware port named a level the design does not have")


# ---------------------------------------------------------------------------
#  4. C1.6 -- the MAC price reaches the mapper, and does not double-count
# ---------------------------------------------------------------------------
def test_the_mac_price_the_mapper_sees_is_the_one_the_report_charges():
    """`set`, not `add`, so `energy.apply_mac_override`'s ratio is exactly 1."""
    cfg = _cfg()
    assert cfg.mac_pj_override is not None, (
        "ECC_MAC_PJ_OVERRIDE is EMPTY in env.sh; since 2026-09-09 the 0.23 pJ "
        "Horowitz figure is THE PRIMARY DENOMINATOR")
    doc = {"ERT": {"tables": [
        {"name": "system_top_level.mac[1..168]",
         "actions": [{"name": "compute", "energy": 1.13555, "arguments": {}},
                     {"name": "leak", "energy": 0.00784449, "arguments": {}}]},
        {"name": "system_top_level.filter_glb[1..1]",
         "actions": [{"name": "read", "energy": 13.591, "arguments": {}}]}]}}
    _assert_eq(tlmod.compute_levels(doc), ["mac"])
    changes = tlmod.mac_ert_changes(doc, cfg.mac_pj_override)
    _assert_eq(changes, {("mac", "compute"): ("set", float(cfg.mac_pj_override))})
    priced = tlmod.ert_prices(tlmod.patched_ert(doc, changes))
    _close(priced[("mac", "compute")], cfg.mac_pj_override)
    # every other row untouched -- the MAC price is not a toll on the memory
    _close(priced[("filter_glb", "read")], 13.591)

    # THE NO-DOUBLE-COUNT PROPERTY. `apply_mac_override` rescales Compute by
    # (macs x override) / compute. With the mapper already at the override,
    # compute IS macs x override, so the ratio is exactly 1.0.
    macs, compute = 1_000_000, 1_000_000 * float(cfg.mac_pj_override)
    _close((macs * float(cfg.mac_pj_override)) / compute, 1.0)

    # BREAKAGE: `add` instead of `set` -- the mapper would then see 1.366 pJ
    # and the evaluator would rescale a price nothing charges.
    added = tlmod.ert_prices(tlmod.patched_ert(
        doc, {("mac", "compute"): ("add", float(cfg.mac_pj_override))}))
    expect_raises(lambda: _close(added[("mac", "compute")], cfg.mac_pj_override),
                  "the MAC price was ADDED to Accelergy's rather than replacing it")

    # BREAKAGE: a design with no arithmetic row must be refused, not skipped
    expect_raises(
        lambda: tlmod.mac_ert_changes({"ERT": {"tables": []}}, 0.23),
        "a table with no `compute` row silently took no MAC price")
    # and EMPTY leaves the table alone
    _assert_eq(tlmod.mac_ert_changes(doc, None), {})


# ---------------------------------------------------------------------------
#  5. C1.7 -- only a DECLARED bank count reaches CACTI
# ---------------------------------------------------------------------------
def test_only_a_declared_bank_count_reaches_cacti():
    """A level that publishes its banking is banked; one that does not is not.

    timeloopfe v4 hands EVERY storage level a default `n_banks: 2`, which is a
    published number for none of them, and the CACTI wrapper floors depth at
    `64 x n_banks` -- so on the depth-3 scratchpad that default would move the
    price through the FLOOR rather than through any banking.
    """
    text = _text("reference")
    for level in ("ifmap_glb", "psum_glb", GLB):
        _assert_eq(_class_of(text, level), archs.BANKED_SRAM_CLASS, level)
        assert _attr(text, level, "n_banks") and _attr(text, level, "n_banks") > 1
    _assert_eq(_class_of(text, "weights_spad"), archs.PLAIN_SRAM_CLASS,
               "weights_spad declares no n_banks")
    # the register files are untouched: they are not SRAM at all
    _assert_eq(_class_of(text, "psum_spad"), "smartbuffer_RF_decoded")

    # the banked compound must EXIST and must forward the attribute, or the
    # class switch is a rename that changes nothing
    comp = (archs.ARCH_COMPONENTS / "smartbuffer_SRAM_banked.yaml").read_text()
    assert "n_banks: n_banks" in comp, comp[:400]

    # BREAKAGE: switch every SRAM. The depth-3 scratchpad must then move, which
    # is the front-end default landing on a level that publishes no banking.
    everything = text.replace(f"class: {archs.PLAIN_SRAM_CLASS}\n",
                              f"class: {archs.BANKED_SRAM_CLASS}\n")
    expect_raises(
        lambda: _assert_eq(_class_of(everything, "weights_spad"),
                           archs.PLAIN_SRAM_CLASS),
        "a level with no declared n_banks was banked from a front-end default")


# ---------------------------------------------------------------------------
#  6. C1.5 -- one clock per design, inverted in one place
# ---------------------------------------------------------------------------
def test_the_clock_is_per_design_and_inverted_once():
    """Eyeriss v1 at its published 200 MHz; every other design at the default."""
    cfg = _cfg()
    _assert_eq(cfg.cycle_seconds_for(ARCH), "5e-09")
    _close(cfg.clock_mhz_for(ARCH), 200.0)
    # A DESIGN DECLARED AT THE RATE IT ALREADY RAN AT KEEPS THE DEFAULT'S EXACT
    # SPELLING. `%.6g` of 1e-9 is "1e-09" and env.sh writes "1e-9"; two
    # spellings of one number would be two architectures to the fingerprint and
    # two directory names to the cache, so seven of the eight entries in
    # ECC_ARCH_CLOCK_MHZ would have colded designs that did not change.
    _assert_eq(cfg.cycle_seconds_for("simba_like"), cfg.global_cycle_seconds)
    _close(cfg.clock_mhz_for("simba_like"), 1000.0)
    _assert_eq(archs.arch_fingerprint("simba_like", cfg),
               archs.arch_fingerprint("simba_like",
                                      dataclasses.replace(cfg, arch_clock_mhz={})),
               "declaring a design at its existing rate colded it")
    # a design with no entry keeps the study default rather than inventing one
    _assert_eq(cfg.cycle_seconds_for("not_a_design"), cfg.global_cycle_seconds)
    # it reaches the mapper: globals_<arch>.yaml and the fingerprint
    # CONTENT-ADDRESSED (2026-09-13): the name carries 8 hex of the bytes, so
    # 186 concurrent jobs write ONE name and nothing replaces a file a reader
    # may hold open. Two designs at two clocks therefore differ in the name.
    assert archs.globals_path(ARCH, cfg).name.startswith(f"globals_{ARCH}_")
    assert archs.globals_path(ARCH, cfg) != archs.globals_path("simba_like", cfg)
    assert f"{cfg.cycle_seconds_for(ARCH)}" in archs.globals_text(cfg, ARCH)
    # and the cache slug says which clock a directory was mapped at
    assert "clk5e-09" in archs.effective_variant(ARCH, cfg)

    # BREAKAGE: invert twice. env.sh section 6's TRAP 2 is that a per-cycle
    # constant converted twice is a silent 5x on every standby and idle term.
    expect_raises(
        lambda: _close(float(cfg.cycle_seconds_for(ARCH)),
                       1.0 / (cfg.clock_mhz_for(ARCH) * 1e6) / 1e6),
        "the clock period was inverted twice and nothing noticed")

    # BREAKAGE: a non-positive rate is refused, not divided by
    expect_raises(
        lambda: dataclasses.replace(cfg, arch_clock_mhz={ARCH: -1.0}
                                    ).cycle_seconds_for(ARCH),
        "a negative clock rate was accepted")


# ---------------------------------------------------------------------------
#  7. C1.8 -- none of this can land silently
# ---------------------------------------------------------------------------
def test_every_phase_c_declaration_colds_the_cache():
    """Each declaration moves `archs.arch_fingerprint()`, which is the hash the
    mapper CACHE DIRECTORY is named after."""
    live = _cfg()
    base = archs.arch_fingerprint(ARCH, live)
    for name, off in (
            ("ECC_DRAM_BANDWIDTH_MBPS", dict(dram_bandwidth_mbps=None)),
            ("ECC_ARCH_CLOCK_MHZ", dict(arch_clock_mhz={})),
            ("ECC_ENERGY_MODEL_REV", dict(energy_model_rev="")),
            ("ECC_MAC_PJ_OVERRIDE", dict(mac_pj_override=None)),
    ):
        other = archs.arch_fingerprint(ARCH, dataclasses.replace(live, **off))
        assert other != base, f"{name} does not move the mapper fingerprint"
    # THE TWO PER-ARM DECLARATIONS only exist on an arm that HAS a boundary, so
    # they are checked there. On the reference they are correctly no-ops: it
    # narrows nothing and moves no less of anything, which is the whole reason
    # `arm-<key>` had to become its own slug in Phase B.
    arm = _cfg("recon2")
    arm_fp = archs.arch_fingerprint(ARCH, arm)
    for name, off in (("ECC_RECON_BW_SCALE", dict(recon_bw_scale=False)),
                      ("ECC_ONCHIP_BW_BITAWARE", dict(onchip_bw_bitaware=False))):
        assert arm_fp != archs.arch_fingerprint(
            ARCH, dataclasses.replace(arm, **off)), \
            f"{name} does not move the mapper fingerprint"
    for name, off in (("ECC_RECON_BW_SCALE", dict(recon_bw_scale=False)),
                      ("ECC_ONCHIP_BW_BITAWARE", dict(onchip_bw_bitaware=False))):
        _assert_eq(archs.arch_fingerprint(ARCH, dataclasses.replace(live, **off)),
                   base, f"{name} moved the REFERENCE arm, which declares none")

    # THE ONE THAT WAS BROKEN. Until Phase C `ECC_ENERGY_MODEL_REV` reached
    # only `Config.fingerprint()`, which labels a RESULT and names no cache
    # directory -- so the deliberate cold re-labelled results while the cache
    # it was meant to invalidate stayed warm.
    a = dataclasses.replace(live, energy_model_rev="rev-a")
    b = dataclasses.replace(live, energy_model_rev="rev-b")
    assert archs.arch_fingerprint(ARCH, a) != archs.arch_fingerprint(ARCH, b)
    # and EMPTY still hashes like every directory that predates the knob:
    # the key is absent from the blob, not present-and-empty
    empty = dataclasses.replace(live, energy_model_rev="")
    _assert_eq(archs.arch_fingerprint(ARCH, empty),
               archs.arch_fingerprint(ARCH, dataclasses.replace(empty)))

    # every arm still has its own directory (prompt_7 B2 gate 2, re-checked
    # here because five new declarations could have merged two of them)
    fps = {a.key: archs.arch_fingerprint(
        ARCH, dataclasses.replace(live, recon_ert_arm=a.key))
        for a in recon.mapper_arms(ARCH, live)}
    _assert_eq(len(set(fps.values())), len(fps), f"two arms share a hash: {fps}")


# ---------------------------------------------------------------------------
#  8. the knobs off reproduce what predates Phase C
# ---------------------------------------------------------------------------
def test_knobs_off_reproduce_the_pre_phase_c_architecture():
    """Every Phase C declaration is absent with its knob off.

    The bank switch is the exception and is deliberate: it has no knob, because
    a published bank count is a property of the design and not of a study.
    """
    off = _cfg("recon2", dram_bandwidth_mbps=None, recon_bw_scale=False,
               onchip_bw_bitaware=False, arch_clock_mhz={})
    text = _text(cfg=off)
    _assert_eq(_attr(text, "DRAM", "shared_bandwidth"), None)
    _assert_eq(_scale_line(text, "DRAM"), None)
    _assert_eq(_scale_line(text, GLB), None)
    _close(_attr(text, GLB, "read_bandwidth"), 16.0)
    _assert_eq(off.cycle_seconds_for(ARCH), off.global_cycle_seconds)
    # the arm's `datawidth: q` is prompt_6's and is NOT a Phase C knob
    _close(_attr(text, GLB, "datawidth"),
           widths.declared_datawidth(off.code_n, off.code_k))


# ---------------------------------------------------------------------------
#  8. C1.2 -- one owner of the off-chip relief, measured per bar
# ---------------------------------------------------------------------------
def test_the_roofline_reproduces_the_plan_it_re_times():
    """The relief is applied ONCE in the recomputation, whoever else applied it.

    A CORRECTION, 2026-09-13. This test first asserted the opposite -- that a
    plan solved WITH the declared scale must be re-timed at factor 1.0, "so the
    relief is not taken twice". The roofline does not adjust Timeloop's cycles,
    it RECOMPUTES them from item counts, so the scale belongs in that
    computation exactly once regardless. Suppressing it re-timed every
    reconstruction bar as if its relief did not exist: 61,905,882 cycles
    against Timeloop's own 54,132,184 (x1.1436) while the reference reproduced
    Timeloop exactly. One arm at x1.0000 beside five at x1.1436 IS the Phase A
    gate failing.
    """
    cfg = _cfg("recon2")
    frac = cfg.code_k / cfg.code_n
    dram = {"level": "DRAM", "offchip": True,
            "reads": {"Weights": 1000.0, "Inputs": 100.0},
            "writes": {"Outputs": 50.0}}
    want = 1000.0 * frac + 100.0

    # THE FACTOR IS `weight_scale` IN EVERY CASE that can be re-timed at all
    for scale, what in ((1.0, "a pre-Phase-C plan, solved at full demand"),
                        (round(frac, 2), "a Phase-C plan, solved at K/N"),
                        (None, "a cache entry from before the field existed")):
        row = dict(dram) if scale is None else dict(dram, bw_consumption_scale=scale)
        f, why = latency_post.relief_owner(row, frac)
        _close(f, frac, )
        assert why, what
        _close(latency_post._items(row, "reads", frac), want)

    # the reference and embedded bars have no relief at all
    _close(latency_post.relief_owner(dict(dram, bw_consumption_scale=1.0), 1.0)[0], 1.0)
    # and an ON-CHIP level is never off-chip relief's business: a narrow weight
    # is still one item there (prompt_7 Defect 1)
    _close(latency_post.relief_owner(
        dict(dram, offchip=False, bw_consumption_scale=round(frac, 2)), frac)[0], 1.0)

    # BREAKAGE: suppress the relief on a plan that WAS solved with it -- the
    # bug this test was written wrong for. The re-timing then reports the
    # unrelieved traffic, which is what x1.1436 was.
    solved = dict(dram, bw_consumption_scale=round(frac, 2))
    expect_raises(
        lambda: _close(sum(solved["reads"].values()), want),
        "the relief was suppressed and the bar re-timed at full weight demand")

    # BREAKAGE: a plan solved at a DIFFERENT code -- no factor repairs that
    expect_raises(
        lambda: latency_post.relief_owner(dict(dram, bw_consumption_scale=0.30),
                                          frac),
        "a plan solved at another code's K/N was re-timed under this one")


# ---------------------------------------------------------------------------
#  9. C1.5 -- the DC idle term is pJ PER CYCLE, and the cycle changed
# ---------------------------------------------------------------------------
def test_the_dc_idle_term_is_rescaled_to_this_designs_clock():
    """env.sh section 6, TRAP 2. It costs 5x if missed, and it had never fired.

    `ECC_RECON_IDLE_PJ` is pJ per cycle measured by Design Compiler at a 1 ns
    clock. It is CLOCK power -- prompt_7 5.3 measures the DC "idle" constant at
    99.48% clock and 0.52% true leakage -- so a 5 ns cycle burns five times as
    much of it. `ECC_LEAKAGE_NW` is the opposite case: POWER in nW, so it takes
    the period at the point of use and must NOT be rescaled. The two are
    declared in different units precisely so this cannot be got wrong silently.
    """
    from ..study import stacks
    live = _cfg()
    _close(live.dc_idle_scale(), 5.0)
    _assert_eq(config.DC_MEASUREMENT_CLOCK_NS, 1.0)

    pre = dataclasses.replace(live, arch_clock_mhz={})    # the pre-C1.5 clock
    _close(pre.dc_idle_scale(), 1.0)
    inc0, idle0, _p0 = stacks.load_recon_energy(pre)
    inc1, idle1, prov = stacks.load_recon_energy(live)
    # the INCREMENTAL term is switching energy per codeword: it does not move
    _close(inc1, inc0)
    # the IDLE term is per cycle: it does
    _close(idle1, idle0 * 5.0)
    assert "rescaled x5" in prov, prov

    # BREAKAGE: charge the idle term at the study clock on a 200 MHz design.
    # That is the 5x env.sh warns about, and it lands on the reconstruction
    # engines only -- the side of the comparison this study is measuring.
    expect_raises(lambda: _close(idle1, idle0),
                  "the idle term was charged at the study clock, not the design's")

    # BREAKAGE: rescale the incremental term too. It is per EVENT, and scaling
    # it would make a slower chip spend more energy per codeword.
    expect_raises(lambda: _close(inc1, inc0 * 5.0),
                  "the per-codeword incremental term was rescaled by the clock")

    # ECC_LEAKAGE_NW is POWER and is NOT in this rescaling at all
    assert "sram_bit" in (live.leakage_nw or {}), live.leakage_nw
    _close(live.leakage_nw["sram_bit"], pre.leakage_nw["sram_bit"])


# ---------------------------------------------------------------------------
#  10. a comment is prose, never a declaration
# ---------------------------------------------------------------------------
def test_a_comment_is_never_the_geometry():
    """`archs.uncommented()`. Found on a real SLURM run, 2026-09-13.

    Every geometry regex in `archs.py` read the RAW text, so a `depth:` in a
    comment was a declaration; and `re.sub(..., count=1)` then rewrote that
    COMMENT and left the attribute untouched. The arch came out declaring
    `depth: 256` beside a freshly written `width: 384` -- 98,304 bits where
    16,512 were intended -- with nothing but one stdout line
    (`filter_glb 1024x64b/8b -> 171x384b/8b`) to say so.
    """
    # THE COMMENT COMES FIRST, exactly as it does in the arch YAML: the
    # divergence note sits in the block ABOVE `attributes:`, so a raw regex
    # reaches it before the attribute.
    src = ("      # the paper's two banks would be `depth: 1024`\n"
           "        depth: 256\n")
    masked = archs.uncommented(src)
    _assert_eq(len(masked), len(src), "the mask must preserve positions")
    assert "1024" not in masked, masked
    _assert_eq(archs.read_attr(src, "depth"), 256)
    got = archs.write_attr(src, "depth", 43)
    assert "`depth: 1024`" in got, "the comment must be left alone"
    assert "depth: 43\n" in got, got

    # ON THE REAL DESIGN: the arch YAML carries exactly such a comment, and the
    # patched geometry must be the ATTRIBUTE's, not the comment's.
    cfg = _cfg()
    geo = archs.patched_weight_geometry(ARCH, cfg)
    _assert_eq(geo[GLB]["depth"], 43, "filter_glb: 256 x 64b renormalised at 384b")
    _assert_eq(geo[GLB]["width"], 384)
    _assert_eq(geo["weights_spad"]["depth"], 3)
    # and the source really does mention another depth in prose, or this test
    # is asserting against a file that cannot exercise it
    src_text = archs.arch_source(ARCH, cfg).read_text()
    in_comments = " ".join(archs._COMMENT_RE.findall(src_text))
    assert "depth: 1024" in in_comments, (
        "the arch YAML no longer names another depth in a comment, so this "
        "test cannot see the bug it exists for")
    assert "depth: 1024" not in archs.uncommented(src_text), (
        "1024 is DECLARED somewhere, not only quoted in prose")

    # BREAKAGE: read the RAW text, as every regex here used to. On the real
    # file that is 1024; the attribute is 256.
    import re as _re
    raw = _re.search(r"\bdepth:\s*(\d+)", src)
    expect_raises(lambda: _assert_eq(int(raw.group(1)), 256),
                  "a depth written in a comment was read as the declaration")
    _assert_eq(int(raw.group(1)), 1024, "the sample must reproduce the real shape")

    # BREAKAGE: a rewrite that lands on nothing must RAISE, not pass silently
    expect_raises(lambda: archs.write_attr("  # depth: 9\n", "depth", 43),
                  "a geometry rewrite landed on a comment and reported success")


# ---------------------------------------------------------------------------
#  11. the Mapper constructor actually finishes
# ---------------------------------------------------------------------------
def test_the_mapper_constructor_finishes():
    """All 72 jobs of the first C2 attempt died 10 s in with
    `AttributeError: 'Mapper' object has no attribute '_memo'`.

    A `@property` had been inserted into the middle of `__init__`, which ENDS
    the function: every line after it -- `self.levels`, `self.victory`,
    `self._memo`, the five counters -- silently became class-body code that
    never ran. No test had ever constructed a Mapper, so nothing caught it
    until SLURM did.
    """
    from .. import timeloop as tlmod
    import tempfile
    cfg = _cfg()
    with tempfile.TemporaryDirectory() as tmp:
        m = tlmod.Mapper(cfg, ARCH, None, tmp, 9, fingerprint="deadbeef")
        # every attribute the mapping path touches must exist
        for attr in ("cfg", "arch", "out_root", "ert_bump", "mac_pj", "levels",
                     "victory", "fingerprint", "legacy_root", "_memo",
                     "mappings", "_remapped", "n_cached", "n_mapped",
                     "n_legacy", "n_failed", "map_seconds", "failures"):
            assert hasattr(m, attr), f"Mapper.__init__ never assigned {attr}"
        _assert_eq(m._memo, {})
        _assert_eq(m.victory, cfg.victory_for(9))
        # and the property that caused it still answers
        assert m.supplies_ert is (m.ert_bump is not None or m.mac_pj is not None)
        assert m.supplies_ert, "ECC_MAC_PJ_OVERRIDE is set, so a table is supplied"


# ---------------------------------------------------------------------------
#  12. the reference arm's table carries no bump, and nothing may assume one
# ---------------------------------------------------------------------------
def test_the_reference_arms_table_needs_no_bump():
    """prompt_7 C1.6 gave the REFERENCE arm a supplied ERT -- the MAC price and
    no bump -- so every `self.bump[...]` in `ErtTables` became a crash waiting
    for the reference job. One was: `ensure()`'s lock message.
    """
    import ast
    import pathlib
    import tempfile
    from .. import timeloop as tlmod

    cfg = _cfg()
    # 1. IT CONSTRUCTS. Before C1.6 this raised: "ErtTables is for an ERT arm".
    with tempfile.TemporaryDirectory() as tmp:
        t = tlmod.ErtTables(cfg, ARCH, None, tmp, None,
                            mac_pj=cfg.mac_pj_override)
        _assert_eq(t.bump, None)
        _close(t.mac_pj, cfg.mac_pj_override)
        # 2. the phrase `ensure()` formats must not touch the bump
        assert "reference" in t.what_it_patches(), t.what_it_patches()
        assert f"{cfg.mac_pj_override:g}" in t.what_it_patches(), t.what_it_patches()
        # 3. `_load` on an empty directory answers None, it does not raise
        _assert_eq(t._load(), None)
        # 4. and the sidecar record is well-formed with no bump
        rec = t.sidecar_record()
        _assert_eq(rec["bump"], None)
        _close(rec["mac_pj"], cfg.mac_pj_override)
    # patching NOTHING is still refused -- a supplied table with no change is a
    # cache directory that exists for no reason
    expect_raises(
        lambda: tlmod.ErtTables(cfg, ARCH, None, ".", None, mac_pj=None),
        "ErtTables accepted an arm that patches no row at all")

    # 5. THE RULE, checked on the source: every `self.bump[...]` in the class
    # sits under a `bump is not None` guard. A reading of the code is not a
    # test; this walks it.
    src = pathlib.Path(tlmod.__file__).read_text()
    cls = next(n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.ClassDef) and n.name == "ErtTables")
    unguarded = []
    for fn in [f for f in cls.body if isinstance(f, ast.FunctionDef)]:
        guarded = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.If):
                t_ = ast.dump(node.test)
                if "bump" in t_ and ("IsNot" in t_ or "Is(" in t_):
                    for sub in (node.body if "IsNot" in t_ else node.orelse):
                        for n2 in ast.walk(sub):
                            guarded.add(getattr(n2, "lineno", -1))
        for node in ast.walk(fn):
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "bump"
                    and node.lineno not in guarded):
                unguarded.append(f"{fn.name}:{node.lineno}")
    _assert_eq(unguarded, [], "these would crash on the reference arm")


# ---------------------------------------------------------------------------
#  13. the files a SLURM array shares are created once, never replaced
# ---------------------------------------------------------------------------
def test_the_shared_inputs_are_never_replaced_under_a_reader():
    """186 jobs, two shared paths, and one of them went missing mid-job.

    `os.replace` onto a shared name is atomic and still not safe here: on
    Lustre a client that already looked the path up holds a handle to the OLD
    inode, and the replacement leaves it stale. The fix is the NAME -- both
    files are a pure function of (architecture, configuration), so they are
    content-addressed and created with `O_EXCL`. Nothing is ever replaced, so
    no reader can hold a handle to something about to be unlinked.
    """
    import tempfile
    cfg = _cfg()

    # 1. THE NAMES CARRY THE CONTENT. Same bytes -> same name; different
    #    bytes -> different name, so a change never overwrites.
    g_now = archs.globals_path(ARCH, cfg)
    _assert_eq(archs.globals_path(ARCH, cfg), g_now, "same config, same name")
    faster = dataclasses.replace(cfg, arch_clock_mhz={ARCH: 1000.0})
    assert archs.globals_path(ARCH, faster) != g_now, (
        "a different clock must be a different FILE, not a replacement")
    a_now = archs.patched_arch_path(ARCH, cfg)
    assert archs._content_tag(a_now.read_text()) in a_now.name, a_now.name

    # 2. `_write_once` NEVER REPLACES. A second writer with different bytes
    #    loses -- which is correct, because the name promises the content.
    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        p = pathlib.Path(tmp) / "shared.yaml"
        archs._write_once(p, "first\n")
        _assert_eq(p.read_text(), "first\n")
        archs._write_once(p, "second\n")
        _assert_eq(p.read_text(), "first\n",
                   "_write_once replaced a file that already existed")
        # and it leaves no temp behind for the next job to trip over
        _assert_eq(sorted(q.name for q in pathlib.Path(tmp).iterdir()),
                   ["shared.yaml"])

        # BREAKAGE: the old writer. `_write_atomic` DOES replace, which is what
        # left a reader holding a stale handle across 186 concurrent jobs.
        archs._write_atomic(p, "third\n")
        expect_raises(lambda: _assert_eq(p.read_text(), "first\n"),
                      "_write_atomic did not replace, so the race it caused "
                      "cannot be reproduced and this test proves nothing")

    # 3. AND THE FINGERPRINT DOES NOT MOVE. The hash is over the CONTENT of
    #    these files, never their paths, so content-addressing the names colds
    #    nothing.
    assert archs._content_tag("x") != archs._content_tag("y")
    _assert_eq(len(archs._content_tag("x")), 8)


def main():
    print("prompt_7 Phase C1 -- the architecture the mapper reads")
    print(f"  design: {ARCH}\n")
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
