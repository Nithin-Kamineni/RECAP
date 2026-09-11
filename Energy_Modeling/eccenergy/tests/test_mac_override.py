"""Tests for ECC_MAC_PJ_OVERRIDE -- the denominator knob (FINDINGS 7.3).

    python3 -m eccenergy.tests.test_mac_override
    bash hpc/tl.sh python3 -m eccenergy.tests.test_mac_override   # pandas ones

An ECC saving is saved_pJ / total_pJ. The saved pJ are weight traffic and must
not depend on what a MAC costs; the total must. These tests pin exactly that on
a synthetic record: under the override the Compute category is MACs x value and
NOTHING ELSE moves -- not a DRAM, buffer, scratchpad or NoC pJ, not the
external-parity pJ, not the embedded DRAM pJ, not one reconstruction
boundary's saved pJ -- while every percentage's denominator does.
"""
from __future__ import annotations

import math
import os
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


def _cfg(**env):
    base = dict(
        ECC_EXPERIMENT="recon", ECC_SWEEP="arch",
        ECC_SWEEP_ARCHS="eyeriss_v2_like", ECC_CONST_ARCH="eyeriss_v2_like",
        ECC_CONST_MODEL="resnet18", ECC_CONST_K="51",
        ECC_MAPPER_THREADS="8", ECC_VICTORY="100", ECC_FROM_CACHE="1",
        ECC_OPT_METRIC="energy",
        ECC_RECON_DECODE_SITE="ondie",
    )
    base.update(env)
    for k in list(os.environ):
        if k.startswith("ECC_") or k == "RECON_OPTIMIZER":
            del os.environ[k]
    os.environ.update(base)
    from eccenergy.config import load_config
    return load_config()


#: The synthetic record: two layers, 1000 and 3000 MACs (the second repeated
#: twice -> 7000 MACs), charged at exactly 1.2 pJ per MAC by the "ERT" = 8400 pJ
#: of Compute. Weight traffic in DRAM / Local / NoC as test_recon's fixture.
_ERT_PJ = 1.2
_MACS = 1000 + 2 * 3000


def _raw(cfg):
    try:
        import pandas as pd
    except ImportError as exc:
        raise _Skip(f"pandas not available on this python: {exc}")
    from eccenergy.energy import Raw, plot_cats
    cats = plot_cats(cfg)
    base = pd.Series({"DRAM": 5000.0, "Local (spads/RF)": 911.0, "NoC": 150.0,
                      "Compute": _MACS * _ERT_PJ}).reindex(cats, fill_value=0.0)
    base_w = pd.Series({"DRAM": 5000.0, "Local (spads/RF)": 800.0,
                        "NoC": 150.0}).reindex(cats, fill_value=0.0)
    base_i = pd.Series({"DRAM": 700.0, "Local (spads/RF)": 100.0}).reindex(cats, fill_value=0.0)
    per_layer = [
        {"layer": "l0", "shape": "C8_M8", "status": "ok", "weights": 1024,
         "macs": 1000, "repeat_count": 1, "mapping_id": "aaaa1111",
         "total_energy_pJ": 3000.0 + 1000 * _ERT_PJ},
        {"layer": "l1", "shape": "C8_M8", "status": "ok", "weights": 1024,
         "macs": 3000, "repeat_count": 2, "mapping_id": "aaaa1111",
         "total_energy_pJ": 3061.0 + 6000 * _ERT_PJ},
        {"layer": "skipped", "shape": "X", "status": "unmapped", "macs": 99999,
         "repeat_count": 1, "total_energy_pJ": 0.0},
    ]
    levels = [{"level": "DRAM", "dataspace": "Weights", "category": "DRAM",
               "energy_pJ": 5000.0, "reads": 2048.0, "writes": 0.0},
              {"level": "mac", "dataspace": "Compute", "category": "Compute",
               "instances": 4, "energy_pJ": _MACS * _ERT_PJ, "reads": 0.0, "writes": 0.0}]
    return Raw(base, base_w, base_i, 5000.0, 2048.0, 2, 0, 2048,
               per_layer=per_layer, levels=levels)


# ------------------------------------------------------------ the rescale itself
def test_only_the_compute_category_moves_and_by_exactly_macs_x_value():
    from eccenergy.energy import apply_mac_override, mac_count
    cfg0 = _cfg()
    raw0 = _raw(cfg0)
    assert mac_count(raw0) == _MACS            # unmapped layers do not count
    # unset: the record comes back unchanged, with the ERT's per-MAC energy noted
    same = apply_mac_override(raw0, cfg0, verbose=False)
    assert same is raw0
    assert math.isclose(same.mac["ert_pj_per_mac"], _ERT_PJ, rel_tol=1e-12)
    assert same.mac["source"] == "Accelergy ERT" and same.mac["override_pj_per_mac"] is None

    cfg = _cfg(ECC_MAC_PJ_OVERRIDE="0.23")
    raw = _raw(cfg)
    before = {c: float(v) for c, v in raw.base.items()}
    out = apply_mac_override(raw, cfg, verbose=False)
    assert out is not raw                                    # the input is not mutated...
    assert float(raw.base["Compute"]) == _MACS * _ERT_PJ     # ...so a cached record stays pure
    assert math.isclose(float(out.base["Compute"]), _MACS * 0.23, rel_tol=1e-12)
    for cat, v in before.items():
        if cat != "Compute":
            assert float(out.base[cat]) == v, cat            # bit-identical, no tolerance
    for cat in out.base_w.index:
        assert float(out.base_w[cat]) == float(raw.base_w[cat]), cat   # no weight share in Compute
        assert float(out.base_i[cat]) == float(raw.base_i[cat]), cat
    assert out.e_dram_w == raw.e_dram_w and out.dram_w_reads == raw.dram_w_reads
    # the levels and the per-layer totals moved by exactly the compute delta
    lv = {l["category"]: l for l in out.levels}
    assert lv["DRAM"]["energy_pJ"] == 5000.0
    assert math.isclose(lv["Compute"]["energy_pJ"], _MACS * 0.23, rel_tol=1e-12)
    delta = _MACS * (_ERT_PJ - 0.23)
    assert math.isclose(sum(l["total_energy_pJ"] for l in raw.per_layer)
                        - sum(l["total_energy_pJ"] for l in out.per_layer), delta, rel_tol=1e-12)
    assert math.isclose(out.per_layer[0]["total_energy_pJ"], 3000.0 + 1000 * 0.23, rel_tol=1e-12)
    assert out.per_layer[2]["total_energy_pJ"] == 0.0        # the unmapped layer is untouched
    # the record says what was done, and with which citation
    assert out.mac["source"] == "ECC_MAC_PJ_OVERRIDE"
    assert out.mac["override_pj_per_mac"] == 0.23
    assert math.isclose(out.mac["compute_scale"], 0.23 / _ERT_PJ, rel_tol=1e-12)
    assert "Horowitz" in out.mac["citation_short"], out.mac


def test_the_saved_pj_are_identical_across_the_rows_and_only_the_denominator_moves():
    """THE CLAIM OF FINDINGS 7.3, as an assertion.

    Task 1's external parity, Task 2's embedded DRAM term and every Task 3
    boundary's saved pJ and reconstruction pJ are computed on the ERT record
    and on the override record; they must be equal to the digit, and the two
    totals must differ by exactly MACs x (ERT - override). The percentages
    then differ by exactly the ratio of the totals -- which is the whole point.
    """
    from eccenergy import recon as reconmod
    from eccenergy.ecc import embedded_dram, external_parity
    from eccenergy.energy import apply_mac_override
    from eccenergy.tests.test_recon import _Layer, _write_cache

    rows = {}
    for label, env in (("as modelled", {}), ("Horowitz int8 MAC", {"ECC_MAC_PJ_OVERRIDE": "0.23"})):
        cfg = _cfg(**env)
        raw = apply_mac_override(_raw(cfg), cfg, verbose=False)
        e_par, _ = external_parity(cfg, raw)
        e_emb, _ = embedded_dram(cfg, raw)
        with tempfile.TemporaryDirectory() as tmp:
            stats = _write_cache(tmp)
            layer = _Layer()
            wp = reconmod.weight_path(cfg, "eyeriss_v2_like", "resnet18", [layer],
                                      {layer.shape_name: stats})
        gran = reconmod.Granularity(cfg.code_n, cfg.code_k, cfg.weight_bits,
                                    cfg.recon_granularity)
        packing = reconmod.Packing(cfg.recon_packing, cfg.weight_bits,
                                   cfg.code_k, cfg.code_n)
        placements = {}
        for p in reconmod.placements_for("eyeriss_v2_like", cfg):
            res = reconmod.evaluate_placement(cfg, "eyeriss_v2_like", p, wp,
                                              raw.base_w, raw.base, 4.0,
                                              gran, packing)
            assert res.status == "evaluated", (label, p.key, res.reason)
            placements[p.key] = res
        rows[label] = {"raw": raw, "parity": e_par, "embedded_dram": e_emb,
                       "placements": placements, "total": raw.total}

    a, b = rows["as modelled"], rows["Horowitz int8 MAC"]
    delta = _MACS * (_ERT_PJ - 0.23)
    # the totals differ by exactly the compute delta...
    assert math.isclose(a["total"] - b["total"], delta, rel_tol=1e-12)
    # ...and not one saved pJ does
    assert a["parity"] == b["parity"]
    assert a["embedded_dram"] == b["embedded_dram"]
    for key in a["placements"]:
        ra, rb = a["placements"][key], b["placements"][key]
        assert ra.detail["energy_saved_by_category_pJ"] == rb.detail["energy_saved_by_category_pJ"], key
        assert ra.components["Reconstruction"] == rb.components["Reconstruction"], key
        assert ra.components["Recon overhead"] == rb.components["Recon overhead"], key
        for cat in ra.components:
            if cat != "Compute":
                assert ra.components[cat] == rb.components[cat], (key, cat)
        assert math.isclose(ra.components["Compute"] - rb.components["Compute"], delta,
                            rel_tol=1e-12), key
        assert math.isclose(ra.total_pJ - rb.total_pJ, delta, rel_tol=1e-12), key
        # the percentage moves by exactly the ratio of the totals
        saved = a["total"] - ra.total_pJ
        assert math.isclose(b["total"] - rb.total_pJ, saved, rel_tol=1e-12), key
        pct_a, pct_b = saved / a["total"], saved / b["total"]
        assert math.isclose(pct_b / pct_a, a["total"] / b["total"], rel_tol=1e-12), key
        # a cheaper MAC makes every percentage LARGER IN MAGNITUDE and flips
        # none: a boundary that costs more than embedded still does (R1 does
        # on this fixture, where 1250 pJ of interface x 0.19 does not pay for
        # 1040 pJ of reconstruction)
        assert abs(pct_b) > abs(pct_a) and (pct_a > 0) == (pct_b > 0), (key, pct_a, pct_b)
    # Task 1's percentage moves the same way: same parity pJ over a smaller total
    conv_a = a["total"] + a["parity"]
    conv_b = b["total"] + b["parity"]
    assert math.isclose(a["parity"] / conv_a * (conv_a / conv_b), b["parity"] / conv_b,
                        rel_tol=1e-12)


# ------------------------------------------------------------------ the knob
def test_the_knob_is_validated_cited_labelled_and_not_in_the_fingerprint():
    from eccenergy.config import ConfigError
    for bad in ("0", "-1"):
        try:
            _cfg(ECC_MAC_PJ_OVERRIDE=bad)
        except ConfigError:
            pass
        else:
            raise AssertionError(f"ECC_MAC_PJ_OVERRIDE={bad} was accepted")
    base = _cfg()
    over = _cfg(ECC_MAC_PJ_OVERRIDE="0.23")
    # the mapper never sees it: same cache, same fingerprint
    assert base.fingerprint() == over.fingerprint()
    assert base.mac_pj_override is None and over.mac_pj_override == 0.23
    # a value listed in provenance.yaml carries its citation; any other is uncited
    short, long_ = over.mac_citation()
    assert "Horowitz" in short and "ISSCC 2014" in long_, (short, long_)
    assert "UNCITED" in _cfg(ECC_MAC_PJ_OVERRIDE="0.5").mac_citation()[0]
    assert "ERT" in base.mac_citation()[0]
    # ...and the sweep figure's title says so
    assert "MAC" not in base.title_suffix()
    assert "MAC 0.23 pJ/op" in over.title_suffix() and "Horowitz" in over.title_suffix()
    # The placement figure's heading is ONE line since 2026-09-11 and does not
    # carry it; the MAC line is in the manifest's `title_caveats`, said once.
    assert "\n" not in over.recon_title() and "MAC" not in over.recon_title()
    t = "\n".join(over.recon_caveats(mac_ert_pj=1.16877))
    assert "MAC 0.23 pJ/op" in t and "ERT was 1.1688" in t, t
    assert t.count("MAC 0.23") == 1, t                     # said once, not twice
    t0 = "\n".join(base.recon_caveats(mac_ert_pj=1.16877))
    assert "MAC 1.1688 pJ/op (Accelergy ERT" in t0, t0


def test_the_manifest_and_the_result_file_carry_the_override():
    """`Session.finish` writes `mac_energy`; `audit.common_caveats` warns."""
    try:
        import pandas  # noqa: F401
    except ImportError as exc:
        raise _Skip(f"pandas not available on this python: {exc}")
    from eccenergy.energy import apply_mac_override
    from eccenergy.experiments import audit
    from eccenergy.paths import Results
    from eccenergy.results_store import ResultBuilder, Variant
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(ECC_MAC_PJ_OVERRIDE="0.23", ECC_RESULTS_DIR=tmp, ECC_OPT_METRIC="edp")
        raw = apply_mac_override(_raw(cfg), cfg, verbose=False)
        assert "edp" in raw.mac["mapping_note"]              # the warning travels with the record
        b = ResultBuilder(cfg, Results(cfg).prepare(), "eyeriss_v2_like", "resnet18",
                          experiment="unit_test", fixed_mapping=True)
        b.add(Variant("baseline_external_parity", kind="baseline", status="evaluated",
                      total_energy_pJ=raw.total,
                      energy_by_component_pJ={c: float(v) for c, v in raw.base.items() if v},
                      mapping_ids=["aaaa1111"]))
        audit.common_caveats(b, cfg, "eyeriss_v2_like", raw, {"traffic": {"method": "x"}})
        doc = b.document()
        warns = " ".join(doc.get("warnings", []))
        assert "ECC_MAC_PJ_OVERRIDE=0.23" in warns and "Horowitz" in warns and "edp" in warns, warns
        # the knob itself is in the config dump every result file carries
        import json
        assert '"mac_pj_override": 0.23' in json.dumps(doc), "config dump lacks the override"


def main():
    print("eccenergy MAC-override (denominator) tests")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    print()
    if SKIPPED:
        print(f"{len(SKIPPED)} skipped (run inside the container): {', '.join(SKIPPED)}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
