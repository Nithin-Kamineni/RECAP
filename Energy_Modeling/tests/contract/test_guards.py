"""The guard registry and the code, held together. `ProjectRestructure.md` section 6.

`GUARDS.md` answers "what is protected, and by what" in one screen, and it is only
worth reading if it cannot drift. Three things have to stay true for that:

  * every registered id is CALLED somewhere (a guard that is deleted stops being
    listed, rather than becoming folklore);
  * every refusal site NAMES a registered id (a guard that is added is listed the
    day it is written, rather than being found by hand a phase later);
  * `GUARDS.md` is what the generator would write right now.

That is the same shape as `test_layer_rule.py`: assert the invariant, list the
exceptions, and make the list expensive to leave alone.

THE EXEMPTIONS ARE THE INTERESTING PART. Five refusal sites in the package do not
name a guard id, and each one is here with the reason. Two are `SystemExit(main())`
-- a CLI's exit status, not a refusal. One is `SystemExit(1)` after a report has
already printed everything it has to say. Two are in a FROZEN file, and those are
registered instead by the text they contain, so the registry still cannot drift
from them without this file going red.
"""
import ast
import collections
import os
import pathlib
import subprocess
import sys

import pytest

from eccenergy import config
from eccenergy.settings import guards as G

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / "eccenergy"

#: `file:line` -> why this refusal names no guard id. DELETE A LINE WHEN THE SITE
#: GOES, exactly as `KNOWN_UPWARD` works: a stale entry fails.
NOT_A_GUARD = {
    "__main__.py": "`raise SystemExit(main())` -- the CLI's exit status.",
    "arch/generate.py": "`raise SystemExit(main())` -- the CLI's exit status.",
    "toolchain/units.py": "`raise SystemExit(main())` -- the CLI's exit status.",
    "study/validate.py": "`raise SystemExit(1)` after the report has printed "
                         "every finding; the refusal IS the report.",
    "study/baseline.py": "FROZEN (CLAUDE.md). Registered by `frozen_match=` "
                         "instead, and checked below.",
}


def _guard_calls():
    """id -> ["config.py:244", ...] for every refusal()/refuse() call."""
    out = collections.defaultdict(list)
    for f in sorted(PKG.rglob("*.py")):
        if "__pycache__" in f.parts or "tests" in f.parts:
            continue
        tree = ast.parse(f.read_text())
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ("refusal", "refuse") and n.args
                    and isinstance(n.args[0], ast.Constant)):
                out[n.args[0].value].append(
                    (str(f.relative_to(PKG)), n.lineno, n.func.attr))
    return out


def _bare_raises():
    """Every `raise ConfigError(...)` / `raise SystemExit(...)` left in the package."""
    out = []
    for f in sorted(PKG.rglob("*.py")):
        if "__pycache__" in f.parts or "tests" in f.parts:
            continue
        tree = ast.parse(f.read_text())
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)):
                continue
            fn = n.exc.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name in ("ConfigError", "SystemExit"):
                out.append((str(f.relative_to(PKG)), n.lineno))
    return out


# --------------------------------------------------------------- the two ends
def test_every_registered_guard_is_called_somewhere():
    """A guard nobody raises is folklore. Delete it from the registry instead."""
    called = set(_guard_calls())
    frozen = {gid for gid, g in G.GUARDS.items() if g.frozen_match}
    orphans = sorted(set(G.GUARDS) - called - frozen)
    assert not orphans, (
        "these guards are registered but never raised -- delete them from "
        f"eccenergy/settings/guards.py, or raise them: {orphans}")


def test_every_called_guard_is_registered():
    """The other end: a `refusal("typo", ...)` must fail here, not at 3am."""
    unknown = sorted(set(_guard_calls()) - set(G.GUARDS))
    assert not unknown, (
        f"these ids are raised but not registered: {unknown}. Every refusal is a "
        f"row of GUARDS.md, so it needs a tier and a one-line `refuses`.")


def test_every_refusal_site_names_a_guard():
    """No `raise ConfigError(...)` may survive without an id, except the five listed.

    This is the one that stops the registry rotting: a guard added next month is
    in `GUARDS.md` the day it is written.
    """
    undeclared = [f"{f}:{ln}" for f, ln in _bare_raises() if f not in NOT_A_GUARD]
    assert not undeclared, (
        "these refusals name no guard id, so they are in no table and cannot be "
        "found:\n  " + "\n  ".join(undeclared) +
        "\n-> `raise guards.refusal(\"<id>\", ...)` for tiers 1-2, "
        "`guards.refuse(\"<id>\", ...)` for tiers 3-4.")


def test_every_exemption_is_still_real():
    """A file that no longer has a bare raise must leave `NOT_A_GUARD`."""
    live = {f for f, _ in _bare_raises()}
    stale = sorted(set(NOT_A_GUARD) - live)
    assert not stale, (
        f"these files no longer raise anything unattributed -- delete them from "
        f"NOT_A_GUARD: {stale}")


def test_every_exemption_says_why():
    for where, why in NOT_A_GUARD.items():
        assert len(why) > 30, f"{where}: reason too thin to act on -- {why!r}"


# ------------------------------------------------------------ tier discipline
def test_the_spelling_at_the_site_matches_the_tier():
    """`refusal()` is tiers 1-2, `refuse()` is tiers 3-4 -- so READING a site tells
    you whether it can be turned off, without looking anything up."""
    wrong = []
    for gid, sites in _guard_calls().items():
        g = G.GUARDS.get(gid)
        if g is None:
            continue
        for f, ln, fn in sites:
            want = "refuse" if g.overridable else "refusal"
            if fn != want:
                wrong.append(f"{f}:{ln} calls {fn}() for tier {g.tier} `{gid}` "
                             f"-- it must call {want}()")
    assert not wrong, "\n  ".join([""] + wrong)


def test_refusal_refuses_to_be_used_at_an_overridable_tier():
    """The runtime half of the rule above. A mutation must not pass silently."""
    with pytest.raises(AssertionError, match="OVERRIDABLE"):
        G.refusal("zero-price", "nope")


def test_refuse_refuses_to_be_used_at_a_fatal_tier():
    with pytest.raises(AssertionError, match="NEVER overridable"):
        G.refuse("need-k-lt-n", "nope")


def test_an_unregistered_id_is_refused_at_the_call():
    with pytest.raises(KeyError, match="not a registered guard"):
        G.refusal("no-such-guard", "nope")


def test_only_tiers_3_and_4_are_overridable():
    """Tiers 1 and 2 may never be lifted: a value that cannot be parsed has no
    meaning and a value that is impossible has no physics, so overriding either
    would not give an ablation, it would give a number with nothing behind it."""
    assert G.OVERRIDABLE == {3, 4}
    for gid, g in G.GUARDS.items():
        assert g.tier in G.TIER_NAMES, f"{gid}: tier {g.tier} is not a tier"
        assert g.overridable == (g.tier in (3, 4)), gid


def test_every_guard_says_what_it_refuses():
    """The `refuses` column is what makes GUARDS.md readable. It may not be a stub."""
    thin = sorted(gid for gid, g in G.GUARDS.items() if len(g.refuses) < 15)
    assert not thin, f"these guards do not say what they refuse: {thin}"


# ------------------------------------------------------------- the frozen two
def test_the_frozen_sites_are_still_where_the_registry_says():
    """`physics/parity.py` and `study/baseline.py` may not be edited (CLAUDE.md), so
    their guards are registered by the TEXT they contain. If somebody reworded one,
    the registry has drifted from the code and this says so."""
    for gid, g in G.GUARDS.items():
        if not g.frozen_match:
            continue
        path = ROOT / g.frozen_file
        assert path.exists(), f"{gid}: {g.frozen_file} is gone"
        assert g.frozen_match in path.read_text(), (
            f"{gid}: {g.frozen_file} no longer contains {g.frozen_match!r}. Either "
            f"the frozen file changed, or the guard moved and can now call "
            f"`guards.refusal()` like every other site.")


# ------------------------------------------------------------------ ECC_ALLOW
def test_ecc_allow_names_guards_and_never_switches_them_all_off():
    """Section 10: a blanket off would be set once and forgotten. Only ids count."""
    # Two LIVE ids, in both spellings. They were `zero-price derived-datawidth`
    # until EnvReorganisation phase 4 retired the second one; a parse test that
    # names a guard the registry has not got still passes, which is exactly why
    # it has to be kept honest by hand.
    os.environ["ECC_ALLOW"] = "zero-price recon-no-split-read-write"
    assert G.allowed() == {"zero-price", "recon-no-split-read-write"}
    os.environ["ECC_ALLOW"] = "zero-price,recon-no-split-read-write"
    assert G.allowed() == {"zero-price", "recon-no-split-read-write"}
    for blanket in ("1", "all", "off", "true"):
        os.environ["ECC_ALLOW"] = blanket
        assert G.allowed() == {blanket}          # a name, matching no guard
        G.reset_overrides()
        with pytest.raises(config.ConfigError):
            G.refuse("zero-price", "still refused")


def test_an_override_is_recorded_and_an_unlisted_one_is_not():
    G.reset_overrides()
    os.environ.pop("ECC_ALLOW", None)
    with pytest.raises(config.ConfigError, match="ECC_ALLOW=zero-price"):
        G.refuse("zero-price", "headline")
    assert G.overrides() == [], "a REFUSED guard must not record an override"

    os.environ["ECC_ALLOW"] = "zero-price"
    G.refuse("zero-price", "headline")
    rec = G.overrides()
    assert [r["guard"] for r in rec] == ["zero-price"]
    assert rec[0]["tier"] == 4 and rec[0]["tier_name"] == "DERIVED"
    G.reset_overrides()


# ------------------------------------------- Appendix B: the three prices
PRICES = ("ECC_MAC_PJ_OVERRIDE", "ECC_DRAM_PJ_PER_BIT", "ECC_BASELINE_DRAM_PJ_PER_BIT")


@pytest.mark.parametrize("knob", PRICES)
def test_a_zero_price_is_an_ablation_and_not_an_impossibility(knob):
    """ProjectRestructure Appendix B. These three used `> 0` where the invariant is
    `>= 0`: zero is the ablation "what if this term were free", which is the upper
    bound on how much the term was ever worth."""
    G.reset_overrides()
    os.environ.pop("ECC_ALLOW", None)
    os.environ[knob] = "0"
    with pytest.raises(config.ConfigError, match="zero-price"):
        config.load_config()

    os.environ["ECC_ALLOW"] = "zero-price"
    cfg = config.load_config()
    assert getattr(cfg, {"ECC_MAC_PJ_OVERRIDE": "mac_pj_override",
                         "ECC_DRAM_PJ_PER_BIT": "dram_pj_per_bit",
                         "ECC_BASELINE_DRAM_PJ_PER_BIT":
                             "baseline_dram_pj_per_bit"}[knob]) == 0
    assert [r["guard"] for r in G.overrides()] == ["zero-price"]
    G.reset_overrides()


@pytest.mark.parametrize("knob", PRICES)
def test_a_negative_price_is_still_refused_and_ecc_allow_cannot_lift_it(knob):
    """The other half of Appendix B: a negative price really IS impossible, so it
    is tier 2 and `ECC_ALLOW` has nothing to say about it."""
    os.environ[knob] = "-1"
    for allow in ("", "negative-price", "zero-price"):
        os.environ["ECC_ALLOW"] = allow
        with pytest.raises(config.ConfigError, match="may not be NEGATIVE"):
            config.load_config()


def test_the_ablation_travels_onto_the_figure_and_the_manifest():
    """Section 6.4 rule 2. You cannot publish an ablation without it saying so."""
    G.reset_overrides()
    os.environ["ECC_DRAM_PJ_PER_BIT"] = "0"
    os.environ["ECC_ALLOW"] = "zero-price"
    cfg = config.load_config()
    caveats = [l for l in cfg.recon_caveats() if l.startswith("ABLATION")]
    assert len(caveats) == 1 and "zero-price" in caveats[0]
    assert "ECC_DRAM_PJ_PER_BIT=0" in caveats[0]
    G.reset_overrides()


def test_a_clean_run_records_no_override_at_all():
    """The default must be byte-identical to what it was before phase 6: no key on
    the manifest, no line on the figure. Every published run of this study is one."""
    G.reset_overrides()
    os.environ.pop("ECC_ALLOW", None)
    cfg = config.load_config()
    assert G.overrides() == []
    assert not [l for l in cfg.recon_caveats() if l.startswith("ABLATION")]


# ------------------------------------------------------------------ GUARDS.md
def test_guards_md_is_not_stale():
    """Section 6.5: generated, never hand-written. `make guards` regenerates it."""
    rc = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_guards.py"),
                         "--check"], cwd=ROOT, capture_output=True, text=True)
    assert rc.returncode == 0, (
        f"{rc.stdout}{rc.stderr}\n-> run `make guards` and commit the result.")


def test_guards_md_lists_every_guard_and_no_others():
    text = (ROOT / "GUARDS.md").read_text()
    for gid in G.GUARDS:
        assert f"`{gid}`" in text, f"{gid} is registered but not in GUARDS.md"
    assert "NO SITE" not in text, (
        "GUARDS.md lists a guard with no call site -- the registry and the code "
        "have drifted.")
