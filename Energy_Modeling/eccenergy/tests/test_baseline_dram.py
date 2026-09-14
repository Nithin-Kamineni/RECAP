"""Tests for the baseline's DRAM PRICE model (ECC_BASELINE_DRAM_PJ_PER_BIT).

    bash hpc/tl.sh python3 -m eccenergy.tests.test_baseline_dram

The seven assertions prompt_4.md asks for, plus deliberate breakage for each:
a test that only ever sees the correct code cannot tell a check that works from
a check that always passes.

The property tests run on the REAL cached Timeloop record for the study point
(`eyeriss_like_wglb` / resnet18 / `layer3.0.conv1`, BCH(63,30)) where it is on
disk, and are SKIPPED -- reported, not hidden -- where it is not.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import traceback

FAILURES = []
SKIPPED = []

#: The cached raw record every property test below is measured on. Pure
#: Timeloop output at Accelergy's own 8 pJ/bit; the overrides are applied here.
#: The reference entry's raw record at the CURRENT fingerprint (3cd00eb16801,
#: the prompt_3 constrained mapspace on the 2026-09-10 geometry). It carries
#: `cycles` and per-level `word_bits` since prompt_6 phases 4-5; an older
#: fingerprint's record would be refused by build_stacks (no cycles) and is a
#: different architecture besides.
REAL_RECORD = pathlib.Path(
    "results/_raw/eyeriss_like_wglb/vic4000__vicx__alg-linear_pruned__to100000000"
    "__noc__paper__mcons__wrelax/fp-3cd00eb16801/cnn/layers1__layer3_0_conv1"
    "/cls-instances/rw-joint/resnet18.json")


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


def _cfg(**env):
    """A BCH(63,30) config on the study point. `ECC_BASELINE_DRAM_PJ_PER_BIT`
    is passed explicitly by every test, so neither default can hide a bug."""
    base = dict(
        ECC_EXPERIMENT="baseline", ECC_SWEEP="arch",
        ECC_CONST_ARCH="eyeriss_like_wglb", ECC_CONST_MODEL="resnet18",
        ECC_RECON_ARCHS="eyeriss_like_wglb",
        ECC_CONST_K="30", ECC_CODE_N="63", ECC_LAYERS="layer3.0.conv1",
        ECC_DRAM_PJ_PER_BIT="40", ECC_MAC_PJ_OVERRIDE="0.23",
        ECC_MAPPER_THREADS="18", ECC_VICTORY="4000", ECC_FROM_CACHE="1",
        # prompt_7 Issue 4: these assertions state the UNGATED idle formula,
        # which PCT=0 reproduces exactly. The gated path is tested in
        # test_recon.test_clock_gating_is_exact_at_zero_and_scales_the_idle_term.
        ECC_RECON_CLOCK_GATING_PCT="0",
    )
    base.update(env)
    for k in list(os.environ):
        if k.startswith("ECC_"):
            del os.environ[k]
    os.environ.update({k: v for k, v in base.items() if v is not None})
    from eccenergy.config import load_config
    return load_config()


def _real_raw(cfg):
    """The cached record, with the study's two overrides applied as `collect` does."""
    if not REAL_RECORD.is_file():
        raise _Skip(f"cached record absent: {REAL_RECORD}")
    try:
        from eccenergy.study.energy import Raw, apply_dram_override, apply_mac_override
    except ImportError as exc:                       # pandas, on the host python
        raise _Skip(f"pandas not available on this python: {exc}")
    raw = Raw.from_json(json.loads(REAL_RECORD.read_text()), cfg)
    return apply_dram_override(apply_mac_override(raw, cfg, verbose=False),
                               cfg, verbose=False)


def _arms(cfg, raw):
    """`(baseline components, embedded components, parity pJ, pricing)`.

    Built exactly as `experiments/baseline.py` and `experiments/embedded.py`
    build them, so a divergence between this and the experiments is a failure
    here rather than a silent difference in the results.
    """
    from eccenergy.physics import baseline_dram
    from eccenergy.study.stacks import embedded_dram, external_parity
    from eccenergy.study.energy import plot_cats
    from eccenergy.study import audit
    series = raw.base.reindex(plot_cats(cfg), fill_value=0.0)
    e_parity, pdetail = external_parity(cfg, raw)
    e_emb, _ = embedded_dram(cfg, raw)
    base = audit.components(series)
    pricing = baseline_dram.charge(cfg, raw, base, e_parity)
    emb = audit.components(series)
    emb[baseline_dram.PARITY_KEY] = e_emb
    return base, emb, e_parity, pricing, pdetail


# ------------------------------------------------------- 1. the traffic is gone
def test_1_no_arm_issues_extra_dram_reads():
    """The price model adds no traffic: same reads, same bits per weight."""
    cfg = _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT="70")
    raw = _real_raw(cfg)
    base, emb, e_parity, pricing, pdetail = _arms(cfg, raw)
    # the record's own read count is what both arms are billed on
    assert raw.dram_w_reads == 294912.0, raw.dram_w_reads
    assert pricing["model"] == "per_bit_price"
    # the parity traffic external_parity() accounts for is NOT charged
    assert e_parity > 0.0
    assert pricing["parity_traffic_energy_NOT_charged_pJ"] == e_parity
    # and the whole baseline-vs-embedded difference is the price, not traffic
    assert math.isclose(base["DRAM"] - emb["DRAM"], pricing["dram_delta_pJ"],
                        rel_tol=1e-12)


# --------------------------------------- 2. parity is zero, the evidence stays
def test_2_parity_component_is_zero_and_the_accounting_survives():
    cfg = _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT="70")
    raw = _real_raw(cfg)
    from eccenergy.physics.baseline_dram import PARITY_KEY
    base, _, e_parity, pricing, pdetail = _arms(cfg, raw)
    assert base[PARITY_KEY] == 0.0                    # explicitly zero, not missing
    assert PARITY_KEY in base
    # the accounting is still there, still hand-checks, and is now SIZE evidence
    assert pdetail["hand_check_passed"], pdetail["hand_check"]
    assert pdetail["stored"]["stored_bits"] == 6193152
    assert pdetail["stored"]["payload_bits"] == 2359296
    assert math.isclose(pdetail["stored"]["stored_bits"]
                        / pdetail["stored"]["payload_bits"], 2.625, rel_tol=1e-12)


# ------------------------------------------- 3. the price, on the whole category
def test_3_whole_dram_category_scales_by_the_price_ratio():
    cfg = _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT="70")
    raw = _real_raw(cfg)
    base, emb, _, pricing, _ = _arms(cfg, raw)
    assert pricing["ratio"] == 70.0 / 40.0 == 1.75
    assert math.isclose(base["DRAM"], emb["DRAM"] * 1.75, rel_tol=1e-12)
    # the WEIGHT share alone is not what moved: the whole category did, so the
    # delta must exceed the weight term's own share of it
    weight_only = raw.e_dram_w * 0.75
    assert pricing["dram_delta_pJ"] > weight_only, (pricing, weight_only)
    assert math.isclose(pricing["dram_delta_pJ"], emb["DRAM"] * 0.75, rel_tol=1e-12)


def test_3b_a_weights_only_price_would_fail_the_check():
    """MUTATION: price the weight rows only. The check must catch it."""
    cfg = _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT="70")
    raw = _real_raw(cfg)
    from eccenergy.physics import baseline_dram
    base, emb, e_parity, pricing, _ = _arms(cfg, raw)
    broken = dict(emb)
    broken["DRAM"] = emb["DRAM"] + raw.e_dram_w * 0.75      # weights only
    broken[baseline_dram.PARITY_KEY] = 0.0
    ok, _ = baseline_dram.reference_bars_agree(broken, emb, e_parity, pricing)
    assert ok is False
    good, _ = baseline_dram.reference_bars_agree(base, emb, e_parity, pricing)
    assert good is True


def test_3c_charging_the_parity_as_well_would_fail_the_check():
    """MUTATION: keep the old traffic term AND the new price. Double-charged."""
    cfg = _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT="70")
    raw = _real_raw(cfg)
    from eccenergy.physics import baseline_dram
    base, emb, e_parity, pricing, _ = _arms(cfg, raw)
    broken = dict(base)
    broken[baseline_dram.PARITY_KEY] = e_parity
    ok, _ = baseline_dram.reference_bars_agree(broken, emb, e_parity, pricing)
    assert ok is False


def test_3d_the_ratio_is_read_from_the_record_not_assumed():
    """MUTATION: the same 70 against Accelergy's own 8 pJ/bit is x8.75, not x1.75.

    A ratio hardcoded as 70/40 would silently misprice any run that does not
    set ECC_DRAM_PJ_PER_BIT.
    """
    cfg = _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT="70", ECC_DRAM_PJ_PER_BIT=None)
    raw = _real_raw(cfg)
    _, _, _, pricing, _ = _arms(cfg, raw)
    assert pricing["pj_per_bit_charged"] == 8.0, pricing
    assert math.isclose(pricing["ratio"], 70.0 / 8.0, rel_tol=1e-12)


# ----------------------------------------------- 4. recon is the arm that scales
def test_4_dram_is_reducible_by_k_over_n_on_every_recon_boundary():
    from ..arch.placements import PLACEMENTS
    from ..arch.weight_path import WEIGHT_PATHS
    stages = [s for stages in WEIGHT_PATHS.values()
              for s in stages if s.key == "dram"]
    assert stages, "no design declares a `dram` weight-path stage"
    assert all(s.reducible for s in stages), [s for s in stages if not s.reducible]
    # every design's boundaries reach it -- the DRAM term is x K/N under all
    for arch, places in PLACEMENTS.items():
        if arch not in WEIGHT_PATHS:
            continue
        assert all("dram" in set(pl.reduced) for pl in places), arch


def test_4b_recon_and_embedded_totals_do_not_move_at_all():
    """Only the baseline arm is repriced. The other two bars must be identical."""
    raw_a = _real_raw(_cfg(ECC_BASELINE_DRAM_PJ_PER_BIT="70"))
    raw_b = _real_raw(_cfg(ECC_BASELINE_DRAM_PJ_PER_BIT=None))
    from eccenergy.study.stacks import build_stacks, load_recon_energy
    cfg_a = _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT="70")
    cfg_b = _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT=None)
    recon_pj, recon_idle, _ = load_recon_energy(cfg_a)      # RULE 3: two terms
    a = build_stacks(cfg_a, raw_a, recon_pj, recon_idle_pj=recon_idle)
    b = build_stacks(cfg_b, raw_b, recon_pj, recon_idle_pj=recon_idle)
    # `recon` WAS IN THIS LIST until EnvReorganisation phase 6 retired the
    # abstract arm. `build_stacks()` builds the two reference arms and nothing
    # else now, so the claim is checked on the two columns that exist -- and
    # the claim about a reconstruction bar is the same one, made where those
    # bars are built: `test_dilation`'s placement tests.
    assert list(a.columns) == ["baseline", "embedded"], list(a.columns)
    assert a["embedded"].to_dict() == b["embedded"].to_dict()
    assert float(a["baseline"].sum()) > float(b["baseline"].sum())


# ------------------------------------------------ 4c. prompt_6 RULE 3
def test_4c_load_recon_energy_returns_incremental_and_idle_separately():
    """prompt_6 RULE 3: `incremental_per_codeword` and `idle_per_cycle` are two
    quantities with two denominators. `load_recon_energy` returns them apart;
    nothing adds them. At BCH(63,30) they are 1.3786 pJ/codeword and
    2.8310811 pJ/cycle/engine; the old combined 4.2096811 must appear nowhere."""
    from eccenergy.study.stacks import load_recon_energy, load_recon_terms, recon_pj_for_k
    cfg = _cfg(ECC_CONST_K="30")
    inc, idle, prov = load_recon_energy(cfg)
    assert abs(inc - 1.3786) < 1e-12 and abs(idle - 2.8310811) < 1e-12, (inc, idle)
    assert "incremental" in prov and "idle" in prov, prov
    assert recon_pj_for_k(cfg, 30) == inc
    assert load_recon_terms(cfg, 30)[:2] == (inc, idle)
    # every code in the table, from the JSON; and the env.sh tables agree with it
    for k, want_inc, want_idle in ((57, 1.6574, 1.9359672), (51, 1.8995, 2.2301273),
                                   (45, 1.6383, 2.4120856), (39, 1.4561, 2.7891299),
                                   (36, 1.5082, 2.8358254), (30, 1.3786, 2.8310811)):
        i, d, _ = load_recon_energy(cfg, k)
        assert abs(i - want_inc) < 1e-12 and abs(d - want_idle) < 1e-12, (k, i, d)
        if cfg.recon_incremental_table:        # env.sh sourced: the tables are there
            from eccenergy.study.stacks import _env_table_entry
            ti, _ = _env_table_entry(cfg.recon_incremental_table, 63, k)
            td, _ = _env_table_entry(cfg.recon_idle_table, 63, k)
            assert ti == want_inc and td == want_idle, (k, ti, td)
    # ECC_RECON_PJ overrides the INCREMENTAL term only
    over = _cfg(ECC_CONST_K="30", ECC_RECON_PJ="9.0")
    i, d, prov = load_recon_energy(over)
    assert i == 9.0 and abs(d - 2.8310811) < 1e-12 and "override" in prov, (i, d, prov)
    # the retired knob is gone
    assert not hasattr(cfg, "recon_include_idle")


def test_4d_build_stacks_builds_the_two_reference_arms_and_no_third():
    """THE ABSTRACT `recon` ARM IS RETIRED (EnvReorganisation phase 6, 2026-09-14).

    This test WAS `..._charges_idle_per_cycle_and_refuses_without_cycles`: it
    pinned prompt_6 RULE 3 in `build_stacks()`'s third column --
    `Reconstruction = codewords x incremental + idle x cycles x 1 engine`, and
    the refusal to charge the idle term on a record with no cycles. That column
    applied K/N to every on-chip level of every design at once, which no
    physical boundary does (plan 6.2, answer 9.2), so it is gone.

    RULE 3 IS NOT GONE WITH IT -- it moved to where the bars are real:
    `study/placement_eval.evaluate_placement()` charges the same two terms per
    boundary, with that boundary's own engine count, and refuses an idle term
    without a cycle count in exactly the same words. `test_dilation.py` and
    `test_latency.py` pin it there.

    What this pins now is the RETIREMENT: two columns, no third, and no
    `Reconstruction` energy on either -- because a bar that reconstructs
    nothing must not carry a reconstruction cost, and an arm that does not
    exist must not come back through a default.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise _Skip(f"pandas not available on this python: {exc}")
    from eccenergy.study.stacks import build_stacks
    from eccenergy.study.energy import Raw, plot_cats
    cfg = _cfg(ECC_CONST_K="30")
    cats = plot_cats(cfg)
    zero = pd.Series({c: 0.0 for c in cats}).reindex(cats)
    base = zero.copy(); base["DRAM"] = 1000.0
    from eccenergy.physics import parity
    per_cw = parity.CodeGeometry(63, 30, 8).validate().weights_per_codeword
    assert per_cw == 3, per_cw
    raw = Raw(base, base.copy(), zero, 1000.0, 3000.0, 1, 0, 3000, per_layer=[],
              levels=[], cycles=10_000)
    st = build_stacks(cfg, raw, 1.3786, recon_idle_pj=2.8310811)
    assert list(st.columns) == ["baseline", "embedded"], list(st.columns)
    assert "Reconstruction" in st.index                  # the category stays
    assert float(st.loc["Reconstruction"].sum()) == 0.0  # and is empty on both

    # A RECORD WITH NO CYCLES IS NO LONGER A REFUSAL HERE, because nothing here
    # charges a per-cycle term any more. It was one, and the refusal it became
    # lives in `placement_eval`.
    stale = Raw(base, base.copy(), zero, 1000.0, 3000.0, 1, 0, 3000,
                per_layer=[], levels=[])
    st2 = build_stacks(cfg, stale, 1.3786, recon_idle_pj=2.8310811)
    assert list(st2.columns) == ["baseline", "embedded"]
    assert float(st2.loc["Reconstruction"].sum()) == 0.0

    # AND THE FUNCTION THAT WENT WITH THE COLUMN IS GONE, not left dead:
    # `mapper_narrowed_weight_energy` was RULE 1's FOURTH narrowing site and
    # existed only because the abstract arm narrowed on chip evaluator-side.
    import eccenergy.study.stacks as stacks_mod
    assert not hasattr(stacks_mod, "mapper_narrowed_weight_energy")


# ------------------------------------------------ 5. nothing on chip moved
def test_5_no_component_outside_dram_differs():
    cfg = _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT="70")
    raw = _real_raw(cfg)
    from eccenergy.physics.baseline_dram import DRAM_KEY, PARITY_KEY
    base, emb, _, _, _ = _arms(cfg, raw)
    assert set(base) == set(emb)
    for c in base:
        if c in (DRAM_KEY, PARITY_KEY):
            continue
        assert base[c] == emb[c], (c, base[c], emb[c])


# -------------------------------------------- 6. no mapper input changed
def test_6_the_knob_is_not_in_any_fingerprint():
    from ..arch.fingerprint import arch_fingerprint
    with_knob = _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT="70")
    without = _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT=None)
    assert with_knob.baseline_dram_pj_per_bit == 70.0
    assert without.baseline_dram_pj_per_bit is None
    assert with_knob.fingerprint() == without.fingerprint()
    assert (arch_fingerprint("eyeriss_like_wglb", with_knob)
            == arch_fingerprint("eyeriss_like_wglb", without))
    # and the DRAM array is NOT scaled per arm: one declared depth, all arms
    assert with_knob.dram_depth == without.dram_depth


# ----------------------------------------- 7. the legacy path is untouched
def test_7_unset_reproduces_the_pre_2026_09_10_model():
    cfg = _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT=None)
    raw = _real_raw(cfg)
    from eccenergy.physics.baseline_dram import PARITY_KEY
    from eccenergy.study.energy import plot_cats
    from eccenergy.study import audit
    base, emb, e_parity, pricing, _ = _arms(cfg, raw)
    assert pricing["model"] == "parity_traffic" and pricing["ratio"] == 1.0
    # the old two lines, computed here, must give exactly today's numbers
    old = audit.components(raw.base.reindex(plot_cats(cfg), fill_value=0.0))
    old[PARITY_KEY] = e_parity
    assert base == old, {k: (base[k], old[k]) for k in old if base[k] != old[k]}
    assert base["DRAM"] == emb["DRAM"]          # priced identically, as before
    assert float(sum(base.values())) == float(sum(old.values()))


def test_7b_the_two_models_write_different_check_names():
    """One name meaning two things is how a model change hides in a diff."""
    from eccenergy.physics import baseline_dram
    assert (baseline_dram.check_name({"model": "per_bit_price"})
            == "baseline_dram_is_exactly_the_per_bit_price")
    assert (baseline_dram.check_name(None)
            == "dram_difference_is_exactly_the_external_parity")
    assert baseline_dram.caveat({"model": "parity_traffic"}) != baseline_dram.caveat(
        {"model": "per_bit_price", "pj_per_bit_baseline": 70,
         "pj_per_bit_charged": 40, "ratio": 1.75})


# ------------------------------------------- the result document and the report
def _task2_document(cfg, raw):
    """The Task 2 validation list, built the way `experiments/embedded` builds it."""
    import tempfile
    from eccenergy.physics.baseline_dram import PARITY_KEY
    from eccenergy.study.embedded import task2_checks
    from eccenergy.study.stacks import embedded_dram
    from eccenergy.paths import Results
    from eccenergy.toolchain.results_store import ResultBuilder, Variant
    base, emb, e_parity, pricing, _ = _arms(cfg, raw)
    _, edetail = embedded_dram(cfg, raw)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(ECC_RESULTS_DIR=tmp,
                   ECC_BASELINE_DRAM_PJ_PER_BIT=(
                       None if cfg.baseline_dram_pj_per_bit is None
                       else f"{cfg.baseline_dram_pj_per_bit:g}"))
        results = Results(cfg).prepare()
        b = ResultBuilder(cfg, results, "eyeriss_like_wglb", "resnet18",
                          experiment="unit_test", fixed_mapping=True)
        for name, kind, comps in (("baseline_external_parity", "baseline", base),
                                  ("embedded_ecc", "embedded", emb)):
            b.add(Variant(name, kind=kind, status="evaluated",
                          total_energy_pJ=float(sum(comps.values())),
                          energy_by_component_pJ=comps, mapping_ids=["aaaa1111"]))
        task2_checks(b, cfg, raw, base, emb, e_parity, edetail,
                     sweep_arm_total=None, newly_mapped=0, raw_cache_hit=True,
                     pricing=pricing)
        return {c["check"]: c["passed"] for c in b.document()["validation"]}, pricing


def test_the_price_model_writes_its_own_check_and_passes_it():
    cfg = _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT="70")
    raw = _real_raw(cfg)
    checks, pricing = _task2_document(cfg, raw)
    assert checks["baseline_dram_is_exactly_the_per_bit_price"] is True, checks
    assert "dram_difference_is_exactly_the_external_parity" not in checks
    assert checks["non_dram_components_match_task1_baseline"] is True
    assert checks["embedded_reads_complete_codeword"] is True


def test_the_legacy_model_writes_the_old_check_and_passes_it():
    cfg = _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT=None)
    raw = _real_raw(cfg)
    checks, pricing = _task2_document(cfg, raw)
    assert checks["dram_difference_is_exactly_the_external_parity"] is True, checks
    assert "baseline_dram_is_exactly_the_per_bit_price" not in checks
    assert checks["non_dram_components_match_task1_baseline"] is True


def test_a_repriced_baseline_with_a_stale_ratio_is_caught():
    """MUTATION: components priced at x1.75, pricing record claiming x1.0."""
    from eccenergy.physics import baseline_dram
    cfg = _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT="70")
    raw = _real_raw(cfg)
    base, emb, e_parity, pricing, _ = _arms(cfg, raw)
    stale = dict(pricing, ratio=1.0)
    ok, _ = baseline_dram.reference_bars_agree(base, emb, e_parity, stale)
    assert ok is False


def test_both_reports_render_under_both_models():
    """The console reports are format strings over the pricing record; a typo in
    one of them only shows up when it is actually printed.

    THE OUTPUT IS CAPTURED AND CHECKED, not just produced (ProjectRestructure
    section 7.1). Rendering into the terminal proves only that the call did not
    raise -- a report that printed NOTHING, because someone put it behind a
    verbosity flag, passed this test just as happily. Each report must now emit
    the design and the model it was handed. `redirect_stdout` rather than
    `capsys`, because `main()` below calls every test with no arguments and a
    pytest fixture would break it.
    """
    import contextlib
    import io

    from eccenergy.study.stacks import embedded_dram
    from eccenergy.study import baseline as baseline_exp
    from eccenergy.study import embedded as embedded_exp
    for knob in ("70", None):
        cfg = _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT=knob)
        raw = _real_raw(cfg)
        base, emb, e_parity, pricing, pdetail = _arms(cfg, raw)
        _, edetail = embedded_dram(cfg, raw)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            baseline_exp._report(cfg, "eyeriss_like_wglb", "resnet18", raw,
                                 float(sum(base.values())), pdetail, pricing)
        task1 = out.getvalue()

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            embedded_exp._report(cfg, "eyeriss_like_wglb", "resnet18", raw,
                                 float(sum(base.values())), float(sum(emb.values())),
                                 pdetail, edetail, pricing)
        task2 = out.getvalue()

        for label, text in (("Task 1", task1), ("Task 2", task2)):
            assert text.strip(), (
                f"{label} report printed NOTHING at "
                f"ECC_BASELINE_DRAM_PJ_PER_BIT={knob!r}")
            assert "eyeriss_like_wglb" in text, (label, knob, text[:200])
            assert "resnet18" in text, (label, knob, text[:200])


# ------------------------------------------------------------- refusals
def test_an_unpriceable_record_is_refused_not_guessed():
    """No reads and no ECC_DRAM_PJ_PER_BIT: there is no price to scale from."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise _Skip(f"pandas not available on this python: {exc}")
    from eccenergy.physics import baseline_dram
    from eccenergy.study.energy import Raw, plot_cats
    cfg = _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT="70", ECC_DRAM_PJ_PER_BIT=None)
    cats = plot_cats(cfg)
    zero = pd.Series({c: 0.0 for c in cats}).reindex(cats)
    raw = Raw(zero, zero, zero, 0.0, 0.0, 1, 0, 0, per_layer=[], levels=[])
    try:
        baseline_dram.price(cfg, raw)
    except ValueError as exc:
        assert "no per-bit DRAM price" in str(exc), str(exc)
    else:
        raise AssertionError("an unpriceable record was priced anyway")


def test_the_knob_must_be_positive():
    """A zero or negative baseline DRAM price is refused by `Config`.

    IT FAILS THROUGH THE `else:` BRANCH -- `raise AssertionError` when the bad
    value is ACCEPTED. ProjectRestructure section 7.1 counted this among three
    tests with "zero assertions"; the AST audit behind that number looked for
    `assert` statements and `pytest.*` calls and does not see a raised
    `AssertionError`, so the count was 1, not 3 (measured 2026-09-13). Left as
    it is: this module predates pytest and `main()` below runs it too.

    Appendix B will re-tier this guard in phase 6 -- zero is the "what if the
    baseline's DRAM were free" ablation and the invariant is `>= 0`. When that
    lands, this test keeps `-70` and drops `0`.
    """
    from eccenergy.config import ConfigError
    for bad in ("0", "-70"):
        try:
            _cfg(ECC_BASELINE_DRAM_PJ_PER_BIT=bad)
        except ConfigError:
            pass
        else:
            raise AssertionError(f"ECC_BASELINE_DRAM_PJ_PER_BIT={bad} was accepted")


def main():
    print("eccenergy baseline DRAM price (prompt_4) tests")
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
