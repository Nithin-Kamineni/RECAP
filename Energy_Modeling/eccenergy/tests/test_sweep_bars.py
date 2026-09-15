"""EnvReorganisation phase 6: every bar is a real, mapped chip.

THE ABSTRACT `recon` ARM IS RETIRED (plan 6.2, answer 9.2). It was
`study.stacks.build_stacks()`'s third column: one engine at the chip entrance,
with K/N applied to every on-chip level of every design identically. No
physical boundary does that -- once weights are reconstructed at the entrance
they are full width on chip -- so it was an optimistic upper bound and
reporting rule R-6 existed only to stop it being quoted as a placement.

What replaces it is a bar per (design, placement), each looked up in ITS OWN
mapper cache at EVERY swept point. This module pins the three things that make
that safe:

1. THE RESOLUTION. `recon` in `ECC_APPROACHES` is exactly one placement, the
   one `ECC_RECON_DEFAULT` names, and nothing can make it mean "all" again or
   mean the retired arm.
2. THE WARNED SKIP (plan 3.1). A design that does not declare a named boundary
   drops that bar and says so -- including a design that declares none at all,
   which is three of the eight (`simba_like`, `simple_input_stationary`,
   `simple_output_stationary`, plan 6.8).
3. A MISSING BAR IS NOT A ZERO. `report.stacked.bar_value` returns None, the
   figure draws nothing and the CSV cell is empty -- because a zero-height bar
   in a group that also holds real ones reads as a measured zero.

And one property that must never change: `ECC_RECON_DEFAULT` is EVALUATOR ONLY.
It decides which cache is read, never what a mapper solves, so it may not reach
`arch_fingerprint()` -- a plotting choice that colds every mapper cache in the
project would be hours of SLURM per design for a decision about one bar.
"""
from __future__ import annotations

import pytest

from ..contracts.errors import ConfigError
from ..settings.run import RECON_PLACEMENT_APPROACHES, REFERENCE_ARMS


# ------------------------------------------------------- 1. the resolution
def test_a_bare_recon_is_one_placement_and_it_is_the_default(cfg):
    c = cfg(ECC_APPROACHES="baseline embedded recon", ECC_RECON_DEFAULT="recon2")
    assert c.bar_arms == ["baseline", "embedded", "recon2"], c.bar_arms
    assert "recon" not in c.bar_arms
    c4 = cfg(ECC_APPROACHES="baseline embedded recon", ECC_RECON_DEFAULT="recon4")
    assert c4.bar_arms == ["baseline", "embedded", "recon4"], c4.bar_arms
    # naming boundaries directly still selects exactly those, in bar order
    c2 = cfg(ECC_APPROACHES="recon4 baseline recon1 embedded")
    assert c2.bar_arms == ["baseline", "embedded", "recon1", "recon4"], c2.bar_arms
    # ...and naming `recon` beside them adds the default ONCE, never twice
    c3 = cfg(ECC_APPROACHES="baseline embedded recon recon2 recon5",
             ECC_RECON_DEFAULT="recon2")
    assert c3.bar_arms == ["baseline", "embedded", "recon2", "recon5"], c3.bar_arms


def test_an_unknown_recon_default_is_refused_by_name(cfg):
    for bad in ("recon", "baseline", "recon9", "", "all"):
        with pytest.raises((ConfigError, SystemExit)) as e:
            cfg(ECC_RECON_DEFAULT=bad)
        assert "ECC_RECON_DEFAULT" in str(e.value), (bad, str(e.value))
    # `recon` in particular: resolving `recon` to `recon` is the abstract arm
    # coming back under another name.
    with pytest.raises((ConfigError, SystemExit)) as e:
        cfg(ECC_RECON_DEFAULT="recon")
    assert "retired" in str(e.value).lower(), str(e.value)


def test_the_abstract_arm_cannot_come_back_through_build_stacks(cfg):
    """`build_stacks()` builds the two REFERENCE arms, whatever is asked for."""
    import pandas as pd
    from ..study.stacks import build_stacks
    from ..study.energy import Raw, plot_cats
    for approaches in ("baseline embedded",
                       "baseline embedded recon",
                       "baseline embedded recon1 recon2 recon3 recon4 recon5"):
        c = cfg(ECC_APPROACHES=approaches)
        cats = plot_cats(c)
        base = pd.Series({x: 0.0 for x in cats}).reindex(cats)
        base["DRAM"] = 1000.0
        raw = Raw(base, base.copy(), base * 0.0, 1000.0, 3000.0, 1, 0, 3000,
                  per_layer=[], levels=[], cycles=1000)
        df = build_stacks(c, raw, 1.0, recon_idle_pj=1.0)
        assert list(df.columns) == [a for a in REFERENCE_ARMS
                                    if a in c.approaches], (approaches, df.columns)
        assert float(df.loc["Reconstruction"].sum()) == 0.0, approaches


# --------------------------------------------------------- 2. the warned skip
def test_a_design_that_does_not_declare_a_boundary_drops_the_bar(cfg, capsys):
    """plan 3.1: warn, never refuse -- one ECC_APPROACHES for every design."""
    from ..arch import placements as placements_mod
    c = cfg(ECC_APPROACHES="baseline embedded recon1 recon2 recon3 recon4 recon5")
    for arch in ("eyeriss_like_wglb", "eyeriss_v2_like_wglb", "simba_like"):
        declared = [p.key for p in placements_mod.PLACEMENTS.get(arch, ())]
        got = c.bar_arms_for(arch)
        assert got[:2] == ["baseline", "embedded"], (arch, got)
        assert got[2:] == [k for k in RECON_PLACEMENT_APPROACHES
                           if k in declared], (arch, got, declared)
    out = capsys.readouterr().out
    # A DESIGN WITH NO `placements.yaml` AT ALL is the case plan 6.8 names, and
    # it must warn like any other rather than raising.
    assert "no placement at all" in out, out
    assert c.bar_arms_for("simba_like") == ["baseline", "embedded"]


def test_a_boundary_the_design_has_not_got_is_never_a_refusal(cfg):
    """`simple_weight_stationary` declares four; asking for five must not stop.

    The witness was `eyeriss_v2_like_wglb` until EnvReorganisation phase 6
    (2026-09-14). That design has five boundaries now -- it gained the weight
    GLB its weight path always named -- while simple_weight_stationary came
    down to four, its MAC-input boundary on the depth-1 latch withdrawn. The
    designs swapped places; the rule under test did not move.
    """
    arch = "simple_weight_stationary"
    c = cfg(ECC_APPROACHES="baseline embedded recon5",
            ECC_ARCHS=arch, ECC_CONST_ARCH=arch)
    assert c.bar_arms_for(arch, warn=False) == ["baseline", "embedded"]
    # ...and the design that DOES declare five still draws it, so this is not
    # passing because the name is unknown everywhere
    five = "eyeriss_v2_like_wglb"
    c5 = cfg(ECC_APPROACHES="baseline embedded recon5",
             ECC_ARCHS=five, ECC_CONST_ARCH=five)
    assert c5.bar_arms_for(five, warn=False) == ["baseline", "embedded", "recon5"]


# ----------------------------------------------------- 3. a missing bar is not 0
def test_a_missing_bar_is_none_not_zero():
    import pandas as pd
    from ..report.stacked import bar_value
    st = pd.DataFrame({"baseline": pd.Series({"DRAM": 10.0, "Compute": 2.0}),
                       "recon2": pd.Series({"DRAM": 8.0, "Compute": 2.0})})
    assert bar_value(st, "DRAM", "baseline") == 10.0
    assert bar_value(st, "DRAM", "recon5") is None      # this design has no R5
    assert bar_value(st, "Standby", "baseline") is None  # no such category here
    # A ZERO IS A MEASUREMENT: a bar that really is zero must NOT read as absent.
    st["recon1"] = pd.Series({"DRAM": 0.0, "Compute": 0.0})
    assert bar_value(st, "DRAM", "recon1") == 0.0


def test_the_table_leaves_a_missing_bar_empty(cfg, tmp_path):
    """The CSV cell of a bar a group has not got is "", never 0.0."""
    import pandas as pd
    from ..report.stacked import write_table
    from ..study.energy import plot_cats
    from ..paths import Results
    c = cfg(ECC_APPROACHES="baseline embedded recon1 recon2",
            ECC_RESULTS_DIR=str(tmp_path))
    cats = plot_cats(c)
    full = pd.DataFrame({a: pd.Series({x: 1.0 for x in cats}).reindex(cats)
                         for a in ("baseline", "embedded", "recon1", "recon2")})
    partial = full[["baseline", "embedded", "recon1"]]      # no recon2 here
    res = Results(c).prepare()
    path = write_table(c, res, [(None, ["a", "b"],
                                {"a": full, "b": partial},
                                {"a": "A", "b": "B"})], "phase6test",
                       bars=["baseline", "embedded", "recon1", "recon2"])
    text = path.read_text()
    rows = {r.split(",")[0]: r for r in text.splitlines()}
    assert "b" in rows, text
    # every recon2 column of row `b` is empty, and row `a`'s are not
    head = text.splitlines()[0].split(",")
    b = rows["b"].split(",")
    a = rows["a"].split(",")
    for i, col in enumerate(head):
        if col.startswith("recon2_"):
            assert b[i] == "", (col, b[i])
            assert a[i] != "", (col, a[i])


# ------------------------------------- 4. evaluator only: not in the fingerprint
def test_recon_default_never_reaches_the_mapper(cfg):
    """Two configurations differing ONLY in `ECC_RECON_DEFAULT` must produce the
    same `Config.fingerprint()` AND the same `arch_fingerprint()`.

    The same property phase 5 pinned for `ECC_METRICS`, and for the same
    reason: a plotting choice in `arch_fingerprint()` would cold every mapper
    cache in the project at once -- hours of SLURM per design -- for a decision
    about which single boundary a bare `recon` bar is.
    """
    from ..arch.fingerprint import arch_fingerprint
    from ..settings.run import IN_FINGERPRINT
    assert "recon_default" not in IN_FINGERPRINT
    a = cfg(ECC_RECON_DEFAULT="recon2")
    b = cfg(ECC_RECON_DEFAULT="recon5")
    assert a.fingerprint() == b.fingerprint()
    for arch in ("eyeriss_like_wglb", "eyeriss_v2_like_wglb"):
        assert arch_fingerprint(arch, a) == arch_fingerprint(arch, b), arch
    # ...and it IS on the record, so a figure can be traced back to it
    assert a.to_dict()["recon_default"] == "recon2"


# ----------------------------------------- 5. the guard was narrowed, not lifted
def test_every_point_sweep_draws_and_the_guard_is_the_mechanism(cfg):
    """BOTH point sweeps have a renderer now, and the guard is kept anyway.

    `fix` got one in phase 6 -- one group, bars = the placements. `area` got
    one on 2026-09-15: one group per rung of `ECC_DEPTH_SWEEP_SCALES`, each a
    full placement evaluation on that rung's own chip.

    `sweep-has-no-figure` was NOT retired with the last axis it refused. It is
    the mechanism by which an axis that can be MAPPED before it can be DRAWN
    says so -- `config.py` reads `NO_FIGURE_SWEEPS` rather than naming an axis
    -- so what is asserted here is that the tuple is empty AND that putting an
    axis back in it still refuses. A guard nobody can demonstrate is folklore.
    """
    from ..report import sweep as sweep_mod
    from ..settings import run as run_mod
    # `area` needs its ladder: the fixture builds a Config from a bare
    # environment, so nothing supplies env.sh's default and an axis with no
    # points is refused by `depth-ladder-empty` before this can be asked.
    for s, extra in (("fix", {}), ("area", {"ECC_DEPTH_SWEEP_SCALES": "1.0 0.5"})):
        c = cfg(ECC_EXPERIMENT="sweep", ECC_SWEEP=s, **extra)
        assert c.sweep == s and c.experiment == "sweep"
        assert s in sweep_mod.POINTS
    assert run_mod.NO_FIGURE_SWEEPS == ()

    # THE GUARD STILL BITES. Put an axis back in the tuple and the refusal is
    # the one `report/sweep.py` would otherwise fail on with a KeyError.
    import unittest.mock as mock
    from .. import config as config_mod
    with mock.patch.object(config_mod, "NO_FIGURE_SWEEPS", ("area",)):
        with pytest.raises((ConfigError, SystemExit)) as e:
            cfg(ECC_EXPERIMENT="sweep", ECC_SWEEP="area",
                ECC_DEPTH_SWEEP_SCALES="1.0 0.5")
    assert getattr(e.value, "guard_id", None) == "sweep-has-no-figure"
    assert "area" in str(e.value)


def test_the_depth_ladder_is_the_x_axis_and_each_rung_is_its_own_chip(cfg):
    """`ECC_SWEEP=area`'s groups ARE `ECC_DEPTH_SWEEP_SCALES`, in typed order.

    And each point pins `weight_depth_scale`, which is in
    `arch.IN_FINGERPRINT` -- so a rung reads the mapper cache of its own
    geometry and not the held one's. That is the whole axis: get it wrong and
    nine bars are drawn from one chip's plan under nine different labels.
    """
    from ..report import sweep as sweep_mod
    c = cfg(ECC_EXPERIMENT="sweep", ECC_SWEEP="area",
            ECC_DEPTH_SWEEP_SCALES="2 1.5 1.0 0.5")
    assert c.swept_values == [2.0, 1.5, 1.0, 0.5]
    assert c.swept_axis == "buffer depth"
    spec, _fontsize = sweep_mod.POINTS["area"](c)
    assert [row[4] for row in spec] == [2.0, 1.5, 1.0, 0.5]
    # every point holds the SAME design, network and code -- only depth moves
    assert {(row[1], row[2], row[3]) for row in spec} == {
        (c.const_arch, c.const_model, c.code_k)}
    fps = set()
    for row in spec:
        pcfg = sweep_mod._point_cfg(c, row[1], row[2], row[3], row[4])
        assert pcfg.weight_depth_scale == row[4]
        fps.add(pcfg.fingerprint())
    assert len(fps) == 4, "four rungs must not share one mapper fingerprint"


def test_an_empty_depth_ladder_is_refused_rather_than_drawn_empty(cfg):
    """An x axis with no points. `depth-ladder-empty`, and only on `area`."""
    with pytest.raises((ConfigError, SystemExit)) as e:
        cfg(ECC_EXPERIMENT="sweep", ECC_SWEEP="area", ECC_DEPTH_SWEEP_SCALES=" ")
    assert getattr(e.value, "guard_id", None) == "depth-ladder-empty"
    # the same empty ladder is harmless on every other axis, which holds ONE
    # depth (`ECC_WEIGHT_DEPTH_SCALE`) and never reads the list
    assert cfg(ECC_SWEEP="bch", ECC_DEPTH_SWEEP_SCALES=" ").sweep == "bch"


def test_the_depth_ladder_is_not_in_the_fingerprint(cfg):
    """Editing the LADDER must never cold a mapper cache.

    `weight_depth_scale` -- one rung -- is hashed and must be. The LIST is a
    plotting axis, exactly as `metrics` is: hashing it would give one chip a
    different fingerprint for every ladder that happens to contain its rung.
    """
    a = cfg(ECC_SWEEP="area", ECC_DEPTH_SWEEP_SCALES="1.0 0.5")
    b = cfg(ECC_SWEEP="area", ECC_DEPTH_SWEEP_SCALES="1.0 0.5 0.25 0.125")
    assert a.fingerprint() == b.fingerprint()


def test_every_sweep_axis_that_draws_has_a_point_builder(cfg):
    """CLAUDE.md's rule for adding an axis, held from the other end."""
    from ..report import sweep as sweep_mod
    from ..settings.run import SWEEPS, NO_FIGURE_SWEEPS
    assert set(sweep_mod.POINTS) == set(SWEEPS) - set(NO_FIGURE_SWEEPS)


# ------------------------------------------- 6. the cold-point sets are guards
def test_the_cold_point_ids_are_registered_guards():
    """A typo in `COLD_AT_THIS_POINT` would silently stop catching a refusal --
    the figure would die on a cold point instead of dropping one bar."""
    from ..report import sweep as sweep_mod
    from ..settings import guards as G
    for gid in (sweep_mod.COLD_AT_THIS_POINT | sweep_mod.NOTHING_AT_THIS_POINT):
        assert gid in G.GUARDS, gid
    assert not (sweep_mod.COLD_AT_THIS_POINT & sweep_mod.NOTHING_AT_THIS_POINT)


def test_a_refusal_carries_its_guard_id():
    """`report/sweep.py` tells a cold point from a real error BY ID, never by
    matching the message -- matching a message is how a swallowed `except`
    starts hiding the guards it was never meant to catch."""
    from ..settings import guards as G
    exc = G.refusal("unknown-recon-default", "x")
    assert getattr(exc, "guard_id", None) == "unknown-recon-default"
    exc2 = G.refusal("ert-arm-cache-cold", "y")          # a SystemExit guard
    assert getattr(exc2, "guard_id", None) == "ert-arm-cache-cold"
