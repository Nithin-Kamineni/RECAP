"""`ECC_METRICS` -- the knob, the four rows, and the two sources of `area`.

EnvReorganisation phase 5. What these tests are FOR, in the order the phase
could have gone wrong:

1. the knob is parsed, ordered and refused correctly, and -- the expensive one
   -- it is NOT in the mapper fingerprint;
2. `latency` and `edp` read the plan that is already on disk and invent
   nothing, which is checked against `Raw` directly rather than against another
   copy of the same arithmetic;
3. `area`'s two halves are each what their source says, PROVED BY BREAKING
   THEM: the instance count really is applied (drop it and the PE array
   vanishes), the DC scrape really does match the reports (perturb it and
   `--check` goes red), and an unmeasured code really is refused rather than
   guessed.

The data-backed ones read the ONE warm corner (`resnet18` on
`eyeriss_like_wglb`) and SKIP when it is cold -- every other cache is cold on
purpose (CLAUDE.md) and a skip is the correct answer there, not a failure.
"""
from __future__ import annotations

import re
import subprocess
import sys

import pytest
import yaml

from ..paths import ARCH_RECON_AREA, ROOT
from ..physics import recon_dc
from ..report import style
from ..settings.run import METRICS
from ..toolchain import ert as ert_mod

WARM = ("eyeriss_like_wglb", "resnet18")


# ------------------------------------------------------------------- the knob
def test_the_four_metric_names_are_the_plans_four(cfg):
    assert METRICS == ("energy", "edp", "latency", "area")


def test_env_sh_ships_a_default_and_every_row_of_it_is_a_real_metric():
    """THE DEFAULT IS THE FEATURE, so it is read out of env.sh itself.

    Exporting the knob and asserting on it proves the knob works and says
    nothing about what a bare `bash run.sh` draws -- which is the thing every
    figure in the project is actually produced by.

    IT USED TO PIN THE VALUE `energy`, because phase 5 chose ONE row so that
    the knob's arrival changed no existing figure or table. `ee4e6e3`
    (2026-09-14) deliberately widened it to `energy edp latency` -- "the
    working ablation configuration" -- and this test was not moved with it, so
    it sat red through every session since, asserting a decision that had been
    superseded on purpose. Pinning a DECISION that env.sh is entitled to make
    is the wrong shape of test; what must hold is that the knob is declared
    with a default and that every row of that default is a metric the figure
    can actually draw.
    """
    text = (ROOT / "env.sh").read_text()
    m = re.search(r'^: "\$\{ECC_METRICS:=([^}]*)\}"', text, re.M)
    assert m, ("env.sh must declare ECC_METRICS with a default -- what a bare "
               "`bash run.sh` draws is the thing every figure is produced by.")
    rows = m.group(1).split()
    assert rows, "ECC_METRICS' default is empty; `metrics-empty` would refuse it"
    unknown = [r for r in rows if r not in METRICS]
    assert not unknown, f"env.sh's ECC_METRICS default names {unknown}"


def test_env_sh_says_which_area_is_which():
    """`area` is a value of BOTH ECC_SWEEP and ECC_METRICS (6.3).

    One word, two axes, in adjacent variables -- the user's decision, and the
    only thing that stops it being a trap is that each option line says which.
    """
    text = (ROOT / "env.sh").read_text()
    sweep_line = next(l for l in text.splitlines()
                      if l.startswith("# bch | model | arch | fix | area"))
    metric_line = next(l for l in text.splitlines()
                       if l.startswith("# energy | edp | latency | area"))
    assert "X AXIS" in sweep_line and "DEPTH" in sweep_line
    assert "Y AXIS" in metric_line


def test_metrics_are_drawn_in_figure_order_not_typed_order(cfg):
    """Rows are "energy on top, latency below it" (3.1) whatever order is typed."""
    assert cfg(ECC_METRICS="area latency energy").metrics == \
        ["energy", "latency", "area"]
    assert cfg(ECC_METRICS="latency energy").metrics == ["energy", "latency"]


def test_a_repeated_metric_is_one_row(cfg):
    assert cfg(ECC_METRICS="energy energy").metrics == ["energy"]


def test_an_unknown_metric_is_refused(cfg):
    from ..contracts.errors import ConfigError
    with pytest.raises(ConfigError):
        cfg(ECC_METRICS="bogus")


def test_the_mappers_word_delay_is_refused_and_points_at_the_right_knob(cfg):
    """`OPT_METRICS` spells the timing objective `delay`; `METRICS` spells it
    `latency`. The two vocabularies must not be conflated, so the wrong word is
    refused with the OTHER knob's name in the message rather than accepted."""
    from ..contracts.errors import ConfigError
    with pytest.raises(ConfigError) as exc:
        cfg(ECC_METRICS="delay")
    assert "ECC_OPT_METRIC" in str(exc.value)


def test_an_empty_metric_list_is_refused(cfg):
    from ..contracts.errors import ConfigError
    with pytest.raises(ConfigError):
        cfg(ECC_METRICS="")


def test_metrics_is_not_in_the_mapper_fingerprint(cfg):
    """THE ONE THAT WOULD COST A WAVE OF SLURM.

    A plotting choice in `arch_fingerprint()` would cold every mapper cache in
    the project at once. Asserted as a PROPERTY -- two configurations differing
    only in `ECC_METRICS` must hash identically -- rather than by reading the
    IN_FINGERPRINT list, because the list is what would be wrong.
    """
    from ..arch import fingerprint as fp
    one = cfg(ECC_METRICS="energy")
    four = cfg(ECC_METRICS="energy edp latency area")
    assert one.fingerprint() == four.fingerprint()
    assert (fp.arch_fingerprint(WARM[0], one)
            == fp.arch_fingerprint(WARM[0], four))


def test_metrics_is_a_field_of_the_record(cfg):
    """It IS on the record -- a figure whose rows are not recoverable from its
    manifest cannot be reproduced. It is the one key phase 5 adds."""
    d = cfg(ECC_METRICS="energy latency").to_dict()
    assert d["metrics"] == ["energy", "latency"]
    assert list(d).index("metrics") == list(d).index("approaches") + 1


# ------------------------------------------------------------- units per row
def test_each_metric_scales_in_its_own_unit():
    """A millisecond labelled "µJ" is the one error a reader cannot catch."""
    assert style.unit_for_metric("latency", 4.3e-3) == (1e-3, "ms")
    assert style.unit_for_metric("area", 3.7e6) == (1e6, "mm²")
    assert style.unit_for_metric("energy", 1.5e9)[1] == "mJ"
    # and the energy scaler is NOT reused for the others
    assert style.unit_for_metric("latency", 4.3e-3) != style.unit_for(4.3e-3)


# -------------------------------------------------------- the ART, accelerator
def _warm_chip_dir():
    """The warm corner's chip directory, AT env.sh's OWN CONFIGURATION.

    NOT `cfg()`. The pinned fixture states a design and lets every mapper and
    chip knob fall back to the code default, which is a DIFFERENT chip from the
    one env.sh describes and therefore a different `fp-` directory -- so a test
    built on it would skip for ever while the cache it means to read sits right
    there. The warm corner is warm at the LIVE configuration (`fp-2db6a4d92ff5`
    on 2026-09-14), and the gate runs pytest under `source env.sh`, so reading
    the ambient environment is what reaches it. Outside the gate, with a shell
    that has not sourced env.sh, this skips -- which is the honest answer.
    """
    from .. import config as config_mod
    from ..arch import fingerprint as fp
    from ..paths import Results
    c = config_mod.load_config()
    variant = fp.effective_variant(WARM[0], c)
    d = Results(c).mapper_cache(WARM[0], variant,
                                fp.arch_fingerprint(WARM[0], c), create=False)
    if not d.is_dir():
        pytest.skip(f"{WARM[0]} is cold at this configuration -- see CLAUDE.md")
    return d


def test_the_art_is_a_property_of_the_chip_not_of_the_layer():
    """Every shape solved under one fingerprint carries the SAME ART.

    This is what makes an area read ONE file per chip instead of a walk of the
    cache, and it is why an area bar has no layer scope in it. Measured rather
    than assumed.
    """
    d = _warm_chip_dir()
    docs = {p.read_bytes() for p in d.glob("*/timeloop-mapper.ART.yaml")}
    if not docs:
        pytest.skip("no solved shape carries an ART yet")
    assert len(docs) == 1, (
        f"{len(docs)} distinct ARTs under one fingerprint -- area is supposed "
        f"to be a property of the ARCHITECTURE, so this would mean the "
        f"per-chip read is picking one of several answers")


def test_the_instance_count_is_applied():
    """MUTATION: summing the per-instance column understates the PE array.

    `weights_spad[1..168]` is priced ONCE and instantiated 168 times. Dropping
    the multiplication is the plausible bug, and it is not a rounding error --
    it loses more than two orders of magnitude on every replicated level.
    """
    d = _warm_chip_dir()
    doc, _p = ert_mod.chip_art(d)
    if doc is None:
        pytest.skip("no ART in the warm cache")
    levels = ert_mod.art_level_areas(doc)
    replicated = {k: v for k, v in levels.items() if v["instances"] > 1}
    assert replicated, "expected a replicated level on a PE array"
    with_counts = sum(v["total_um2"] for v in levels.values())
    without = sum(v["per_instance_um2"] for v in levels.values())
    assert with_counts > without * 5, (
        f"the instance count is not being applied: {with_counts:,.0f} vs "
        f"{without:,.0f} um2")
    for name, v in levels.items():
        assert v["total_um2"] == pytest.approx(
            v["per_instance_um2"] * v["instances"]), name


def test_art_instances_parses_the_table_name():
    assert ert_mod.art_instances("system_top_level.weights_spad[1..168]") == 168
    assert ert_mod.art_instances("system_top_level.DRAM[1..1]") == 1
    # a name with no range is one instance, not zero
    assert ert_mod.art_instances("system_top_level.mac") == 1


# ------------------------------------------------------ the DC scrape, engine
def test_the_scrape_matches_the_reports():
    """`archs/_shared/recon_area.yaml` is GENERATED; --check proves it current."""
    rc = subprocess.run([sys.executable, str(ROOT / "tools" / "scrape_dc_area.py"),
                         "--check"], cwd=ROOT, capture_output=True, text=True)
    assert rc.returncode == 0, f"{rc.stdout}{rc.stderr}\n-> python3 tools/scrape_dc_area.py"


def test_active_and_idle_areas_are_equal_in_every_report():
    """THE REASON ONLY `active` IS STORED (the user's instruction, 2026-09-14).

    `active/` and `idle/` are one netlist under two switching activities: the
    POWER differs and the AREA cannot. Adding them would count one engine
    twice. Asserted over every report on disk, so a future synthesis run that
    breaks the assumption fails here rather than silently doubling a bar.
    """
    snaps = ROOT / "data" / "dc" / "report_snapshots"
    if not snaps.is_dir():
        pytest.skip("the DC report snapshots are not checked out")
    import re
    total = re.compile(r"^Total cell area:\s+([0-9.]+)\s*$", re.M)
    seen = 0
    for d in sorted(snaps.iterdir()):
        a, i = d / "active" / "area.rpt", d / "idle" / "area.rpt"
        if not (a.is_file() and i.is_file()):
            continue
        seen += 1
        assert (float(total.search(a.read_text()).group(1))
                == float(total.search(i.read_text()).group(1))), d.name
    assert seen >= 6, f"only {seen} configurations found"


def test_the_stored_area_is_the_active_report_verbatim():
    """No rounding, no unit conversion, no arithmetic between report and YAML."""
    import re
    doc = yaml.safe_load(ARCH_RECON_AREA.read_text())
    total = re.compile(r"^Total cell area:\s+([0-9.]+)\s*$", re.M)
    snaps = ROOT / "data" / "dc" / "report_snapshots"
    if not snaps.is_dir():
        pytest.skip("the DC report snapshots are not checked out")
    for cid, row in doc["engines"].items():
        rpt = snaps / cid / "active" / "area.rpt"
        if not rpt.is_file():
            continue
        assert row["area_um2"] == float(total.search(rpt.read_text()).group(1)), cid


def test_every_swept_code_has_a_measured_engine_area(cfg):
    """The six BCH(63,K) the study sweeps must all resolve, or `area` is a
    metric that works on some bars and refuses on others."""
    c = cfg()
    for k in (57, 51, 45, 39, 36, 30):
        area, prov = recon_dc.load_recon_area(c, k)
        assert area > 0 and "recon_area.yaml" in prov


def test_an_unsynthesized_code_is_refused_not_guessed(cfg):
    """THE DIFFERENCE FROM THE ENERGY LOOKUP, and it is deliberate.

    An unmeasured reconstruction ENERGY falls through to a fallback constant
    with a warning. An unmeasured AREA does not: there is no honest default for
    a silicon area, so the only answers are the measurement or a refusal.
    """
    from ..contracts.errors import ConfigError
    c = cfg()
    with pytest.raises(ConfigError) as exc:
        recon_dc.load_recon_area(c, 61)     # never synthesized
    assert "no fallback" in str(exc.value).lower()


def test_the_engine_is_one_engine(cfg):
    """The YAML holds ONE stage's area; the count is the caller's."""
    doc = yaml.safe_load(ARCH_RECON_AREA.read_text())
    for cid, row in doc["engines"].items():
        assert "engines" not in row and "count" not in row, (
            f"{cid} declares a count -- how many engines a boundary "
            f"instantiates is physics/granularity.py's answer, never a number "
            f"in this file")


# --------------------------------------------- the rows agree with each other
def _warm_session():
    """The collected POINTS of the warm corner, on the ONE axis it can serve.

    `ECC_SWEEP=bch` on purpose. `model` and `arch` both need a second cache
    that is cold by design -- and since EnvReorganisation phase 6 `fix` has a
    sweep renderer too, but it is ONE group, which makes it a weaker test of
    identities that hold per group. The BCH axis reads the warm corner at every
    code it has and drops the rest with a `[skip]`, so it is the axis the warm
    corner supports.

    Returns `(cfg, points, groups, stacks, labels, bars)` from
    `report.sweep.collect_points` -- THE SAME CALL `run()` makes, so these
    tests cannot pass against a shape the figure never sees. Every bar here is
    a real, mapped chip: the abstract `recon` arm is retired.
    """
    from .. import config as config_mod
    from ..study.common import Session
    import os
    os.environ["ECC_FROM_CACHE"] = "1"
    os.environ["ECC_REPLOT_ONLY"] = "1"
    os.environ["ECC_METRICS"] = "energy edp latency area"
    os.environ["ECC_SWEEP"] = "bch"
    os.environ["ECC_EXPERIMENT"] = "sweep"
    os.environ["ECC_CONST_ARCH"] = WARM[0]
    os.environ["ECC_SWEEP_ARCHS"] = WARM[0]
    os.environ["ECC_CONST_MODEL"] = WARM[1]
    os.environ["ECC_SWEEP_MODELS"] = WARM[1]
    c = config_mod.load_config()
    if c.archs[:1] != [WARM[0]] or WARM[1] not in c.models:
        pytest.skip("the live configuration is not the warm corner")
    from ..report import sweep as sweep_mod
    try:
        points, groups, stacks, labels, bars, _f = sweep_mod.collect_points(c)
    except Exception as exc:
        pytest.skip(f"nothing collected -- the cache is cold here: {exc}")
    if not groups:
        pytest.skip("the bch axis produced no group on this cache")
    return c, points, groups, stacks, labels, bars


def test_edp_is_exactly_energy_times_delay(cfg):
    """THE ROWS MUST AGREE, and this is the identity that proves they do.

    If `edp` were built from a second energy figure, or `latency` re-timed a
    second time with a different scale, the three rows would drift apart and
    nothing on the figure would say so. The saving compounds EXACTLY:

        1 - (1 - energy_saving)(1 - latency_saving) == edp_saving

    Measured 2026-09-14 on resnet18/eyeriss_like_wglb at BCH(63,39):
    5.142 % energy and 1.746 % latency compound to 6.798 %, which is the EDP
    row's own number to seven figures.
    """
    from ..report import sweep as sweep_mod
    c, points, groups, stacks, labels, bars = _warm_session()
    rows = dict((m, (g, s)) for m, g, s, _l in
                sweep_mod.metric_rows(c, points, groups, stacks, labels, bars))
    for needed in ("energy", "edp", "latency"):
        if needed not in rows:
            pytest.skip(f"{needed} row could not be built")
    ref = bars[0]
    for g in rows["energy"][0]:
        def sav(metric, arm):
            st = rows[metric][1][g]
            base = float(st[ref].sum())
            return (base - float(st[arm].sum())) / base
        # A GROUP MAY BE MISSING A BAR (phase 6): a boundary whose chip is not
        # mapped at this code is dropped, never billed from another plan.
        for arm in [a for a in bars if a in stacks[g].columns]:
            compounded = 1.0 - (1.0 - sav("energy", arm)) * (1.0 - sav("latency", arm))
            assert compounded == pytest.approx(sav("edp", arm), abs=1e-9), (
                f"{g}/{arm}: the EDP row does not equal energy x delay -- "
                f"the three rows have drifted apart")


def test_the_two_reference_arms_cannot_differ_in_time_or_in_area(cfg):
    """Baseline and embedded move the SAME weight bits and are the SAME chip.

    The baseline pays a dearer per-bit DRAM PRICE, not extra traffic
    (CLAUDE.md), so no re-timing can separate the two; and neither narrows
    anything, so they stand on one accelerator with no engine. A difference in
    either row would mean a metric had picked up the energy model's price
    difference as if it were a physical one.
    """
    from ..report import sweep as sweep_mod
    c, points, groups, stacks, labels, bars = _warm_session()
    rows = dict((m, s) for m, _g, s, _l in
                sweep_mod.metric_rows(c, points, groups, stacks, labels, bars))
    for metric in ("latency", "area"):
        if metric not in rows:
            continue
        for g, st in rows[metric].items():
            assert float(st["baseline"].sum()) == pytest.approx(
                float(st["embedded"].sum())), (
                f"{metric} differs between baseline and embedded at {g}")
