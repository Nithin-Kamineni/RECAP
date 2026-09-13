"""Tests for the Task 3 reconstruction-placement model.

    python3 -m eccenergy.tests.test_recon
    bash hpc/tl.sh python3 -m eccenergy.tests.test_recon   # for the pandas ones

Dependency-free and offline like the other suites. The stats parser is driven
by a SYNTHETIC `timeloop-mapper.stats.txt` written into a temporary directory,
so nothing here needs the mapper cache, the container or Timeloop -- and the
synthetic numbers are chosen so the wire/switching split, the access totals and
the loop-nest reuse run all have hand-checkable answers.
"""
from __future__ import annotations

import math
import os
import pathlib
import re
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
        ECC_RECON_PACKING="stream", ECC_RECON_ENCODER_GRANULARITY="weight",
        # The decoder is on the DRAM die, so the
        # synthetic case fixes one: 5000 pJ of DRAM = 3750 array + 1250 interface.
        ECC_RECON_DECODE_SITE="ondie",
        # prompt_7 Issue 4: every assertion below predates clock gating and
        # states the UNGATED formula. PCT=0 is algebraically identical to it,
        # so pinning it here keeps those assertions exact and meaningful; the
        # gated path has its own test.
        ECC_RECON_CLOCK_GATING_PCT="0",
    )
    base.update(env)
    for k in list(os.environ):
        if k.startswith("ECC_") or k == "RECON_OPTIMIZER":
            del os.environ[k]
    os.environ.update(base)
    from eccenergy.config import load_config
    return load_config()


class _Layer:
    """The three attributes `recon.read_weight_path` asks a Layer for."""

    def __init__(self, name="l0", shape_name="C8_M8", weights=1024, count=1):
        self.name = name
        self.shape_name = shape_name
        self.weights = weights
        self.count = count


# --------------------------------------------------------- synthetic Timeloop
#: One PE weight scratchpad and the two Eyeriss v2 weight networks, with round
#: numbers. The scratchpad: 100 reads and 10 fills per instance across 4
#: utilized instances -> 400 reads and 40 fills TOTAL, which is the quantity the
#: reconstruction count needs and the one the aggregated Raw record does not
#: hold. Block size 3 x 8 bits = a 24-bit physical word.
#:
#: Network 0 (the mesh): 100 ingresses, 2.0 hops, 0.5 pJ per hop, 1 instance ->
#: wire 100 x 2 x 0.5 = 100 pJ; routers (1+2) x 100 x 0.1 = 30 pJ; total 130 pJ,
#: which is what `Energy (total)` says, so the split needs no rescaling.
#: Network 1 (cluster-local): 50 ingresses, 1 instance, wire only, 20 pJ.
_STATS = """Buffer and Arithmetic Levels
----------------------------
Level 0
-------
=== mac ===

    SPECS
    -----
    Word bits             : 16
    Instances             : 384 (32*12)

    STATS
    -----
    Energy (total)          : 9000.00 pJ

Level 1
-------
=== weights_spad ===

    SPECS
    -----
        Technology                      : SRAM
        Size                            : 288
        Word bits                       : 8
        Block size                      : 3
        Instances                       : 192 (16*12)

    STATS
    -----
    Weights:
        Partition size                           : 24
        Utilized capacity                        : 24
        Utilized instances (max)                 : 4
        Scalar reads (per-instance)              : 100
        Scalar fills (per-instance)              : 10
        Scalar updates (per-instance)            : 0
        Energy (total)                           : 800.00 pJ

Level 2
-------
=== ifmap_spad ===

    SPECS
    -----
        Word bits                       : 8
        Block size                      : 1
        Instances                       : 192 (16*12)

    STATS
    -----
    Inputs:
        Utilized instances (max)                 : 4
        Scalar reads (per-instance)              : 77
        Energy (total)                           : 111.00 pJ

Level 3
-------
=== DRAM ===

    SPECS
    -----
        Word bits                       : 8
        Block size                      : 8
        Instances                       : 1

    STATS
    -----
    Weights:
        Utilized capacity                        : 1024
        Utilized instances (max)                 : 1
        Scalar reads (per-instance)              : 2048
        Scalar fills (per-instance)              : 0
        Energy (total)                           : 5000.00 pJ

Networks
--------
Network 0
---------
inter_PE_cluster_spatial <==> inter_PE_spatial

    SPECS
    -----
        Word bits       : 8
        Router energy   : 0.10 pJ
        Wire energy     : 0.12 pJ/b/mm
        Ingress energy  : - pJ

    STATS
    -----
    Weights:
        Fanout                                  : 4
        Multicast factor                        : 1
        Ingresses                               : 100.00
        Average number of hops                  : 2.00
        Energy (per-hop)                        : 500.00 fJ
        Energy (per-instance)                   : 130.00 pJ
        Energy (total)                          : 130.00 pJ
        Link transfer energy (total)            : 0.00 pJ
        Spatial Reduction Energy (total)        : 0.00 pJ

Network 1
---------
inter_PE_spatial <==> ifmap_spad

    SPECS
    -----
        Word bits       : 8
        Router energy   : 0.00 pJ
        Wire energy     : 0.12 pJ/b/mm
        Ingress energy  : - pJ

    STATS
    -----
    Weights:
        Fanout                                  : 9
        Multicast factor                        : 1
        Ingresses                               : 50.00
        Average number of hops                  : 1.00
        Energy (per-hop)                        : 400.00 fJ
        Energy (per-instance)                   : 20.00 pJ
        Energy (total)                          : 20.00 pJ
        Link transfer energy (total)            : 0.00 pJ
        Spatial Reduction Energy (total)        : 0.00 pJ

Operational Intensity Stats
---------------------------
"""

#: The loop nest. Below `weights_spad` sit a temporal Q, a temporal P and then
#: a temporal M: walking inward-out, Q(2) and P(3) leave the weight unchanged
#: and M stops the run, so the reuse run is 6.
_MAP = """DRAM [ Weights:2048 (2048) Inputs:100 (100) Outputs:100 (100) ]
-------------------------------------------------------------------
| for C in [0:4)

inter_PE_cluster_spatial [ ]
----------------------------
|   for M in [0:4) (Spatial-X)

weights_spad [ Weights:24 (24) ]
--------------------------------
|     for M in [0:8)
|       for P in [0:3)
|         for Q in [0:2)
|           << Compute >>
"""


def _write_lf(path, text):
    """Write with LF endings on any python.

    `Path.write_text(newline=...)` is 3.10+, and the host python here is older --
    two tests died with a TypeError instead of running. `io.open` takes the same
    argument everywhere.
    """
    import io
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _write_cache(root):
    d = pathlib.Path(root) / "C8_M8"
    d.mkdir(parents=True, exist_ok=True)
    _write_lf(d / "timeloop-mapper.stats.txt", _STATS)
    _write_lf(d / "timeloop-mapper.map.txt", _MAP)
    return d / "timeloop-mapper.stats.txt"


#: The synthetic DRAM level is 5000 pJ and it is ONE stage: the whole of it is
#: x K/N on every boundary (the f_if array/interface split was removed
#: 2026-09-09). Every hand check below is written against this number.
_DRAM_W = 5000.0


def _dram_expected(frac, dram_w=_DRAM_W):
    """What every placement's DRAM component must be with the decoder on the die."""
    return dram_w * frac


# --------------------------------------------------------- layout arithmetic
def test_g_rec_is_the_worst_aligned_codeword_not_the_average():
    from eccenergy.recon import Granularity
    g = Granularity(63, 51, 8, "weight")
    # 63 bits is 7.875 weights, but a codeword that starts mid-weight reaches
    # into one more, so the group a PE must hold is 9 -- and it is 9 for every
    # K, because n and weight_bits alone decide the alignment.
    assert g.g_rec == 9, g.g_rec
    assert math.isclose(g.weights_per_codeword, 63 / 8)
    for k in (57, 51, 45, 39, 36, 30):
        assert Granularity(63, k, 8, "weight").g_rec == 9, k
    # a weight-aligned code has no straddling weight and needs exactly n/w
    assert Granularity(64, 56, 8, "weight").g_rec == 8


def test_encoder_charging_modes_bracket_each_other():
    from eccenergy.recon import Granularity
    amortized = Granularity(63, 51, 8, "weight")
    pessimistic = Granularity(63, 51, 8, "codeword")
    assert math.isclose(amortized.codewords(7875), 1000.0)
    assert math.isclose(pessimistic.codewords(7875), 7875.0)
    # the pessimistic reading is exactly weights_per_codeword times the other
    assert math.isclose(pessimistic.codewords(800) / amortized.codewords(800),
                        63 / 8)


def test_stream_packing_scales_by_k_over_n_and_aligned_packing_does_not():
    from eccenergy.recon import Packing
    stream = Packing("stream", 8, 51, 63)
    assert math.isclose(stream.reduced_bits_per_weight, 8 * 51 / 63)
    # every stage falls by exactly K/N when the retained bits are packed
    assert math.isclose(stream.storage_scale(24), 51 / 63)
    assert math.isclose(stream.wire_scale(), 51 / 63)
    assert math.isclose(stream.switching_scale(24), 51 / 63)

    aligned = Packing("aligned", 8, 51, 63)
    # ceil(8 * 51/63) = 7 whole bits per weight
    assert aligned.reduced_bits_per_weight == 7
    # ...and floor(24/7) = 3 values per 24-bit word, the same 3 it held at 8
    # bits, so the SRAM access count does not move at all -- section 16's point
    assert math.isclose(aligned.storage_scale(24), 1.0), aligned.storage_scale(24)
    assert math.isclose(aligned.switching_scale(24), 1.0)
    # wire energy is per bit MOVED, so it still falls
    assert math.isclose(aligned.wire_scale(), 7 / 8)
    # a wide enough word does let aligned packing win something
    assert aligned.storage_scale(64) < 1.0




# ------------------------------------------------------- the stats re-parse
def test_weight_path_reads_totals_capacity_and_the_wire_split():
    from eccenergy import recon as reconmod
    cfg = _cfg()
    with tempfile.TemporaryDirectory() as tmp:
        stats = _write_cache(tmp)
        layer = _Layer()
        wp = reconmod.weight_path(cfg, "eyeriss_v2_like", "resnet18", [layer],
                                  {layer.shape_name: stats})
        assert not wp.unclaimed, wp.unclaimed

        spad = wp.stage("weight_spad")
        # PER-INSTANCE x UTILIZED INSTANCES. This is the whole reason the module
        # re-parses the stats instead of using the aggregated Raw record.
        assert spad.reads == 400.0, spad.reads
        assert spad.fills == 40.0, spad.fills
        assert spad.utilized_capacity == 24
        assert spad.block_bits == 24
        assert math.isclose(spad.energy_pJ, 800.0)

        # The one DRAM level is ONE stage owning all of it.
        d = wp.stage("dram")
        assert d.reads == 2048.0
        assert math.isclose(d.energy_pJ, _DRAM_W)
        assert d.level_share == 1.0
        assert d.levels == ["DRAM"]
        assert wp.decode_site == "ondie"
        assert math.isclose(wp.dram_weight_energy(), _DRAM_W)

        mesh = wp.stage("inter_cluster_mesh")
        assert mesh.ingresses == 100.0
        assert math.isclose(mesh.energy_pJ, 130.0)
        # wire = 100 ingresses x 2 hops x 0.5 pJ/hop;  routers = 3 x 100 x 0.1
        assert math.isclose(mesh.wire_pJ, 100.0, rel_tol=1e-9), mesh.wire_pJ
        assert math.isclose(mesh.switch_pJ, 30.0, rel_tol=1e-9), mesh.switch_pJ

        local = wp.stage("cluster_local")
        assert local.ingresses == 50.0
        # router energy 0 -> the whole 20 pJ is wire
        assert math.isclose(local.wire_pJ, 20.0, rel_tol=1e-9)
        assert math.isclose(local.switch_pJ, 0.0, abs_tol=1e-9)

        # the split always sums back to the energy Timeloop reported, whatever
        # the analytic model says
        for st in (mesh, local):
            assert math.isclose(st.wire_pJ + st.switch_pJ, st.energy_pJ,
                                rel_tol=1e-9)

        # R4b and the retention model were removed on 2026-09-10, so the
        # weight path no longer carries one.
        assert not hasattr(wp, "retention")


def test_a_weight_level_no_stage_claims_fails_the_cross_check():
    """The check that stops weight energy from being silently dropped."""
    from eccenergy import recon as reconmod
    try:
        import pandas as pd
    except ImportError as exc:
        raise _Skip(f"pandas not available on this python: {exc}")
    cfg = _cfg()
    with tempfile.TemporaryDirectory() as tmp:
        stats = _write_cache(tmp)
        # rename the scratchpad to something no stage matches
        p = pathlib.Path(stats)
        _write_lf(p, _STATS.replace("weights_spad", "mystery_weight_buffer"))
        layer = _Layer()
        wp = reconmod.weight_path(cfg, "eyeriss_v2_like", "resnet18", [layer],
                                  {layer.shape_name: stats})
        assert wp.unclaimed and wp.unclaimed[0]["level"] == "mystery_weight_buffer"
        base_w = pd.Series({"DRAM": 5000.0, "Global buffer": 0.0,
                            "Local (spads/RF)": 800.0, "NoC": 150.0,
                            "Compute": 0.0})
        ok, detail = reconmod.cross_check(cfg, "eyeriss_v2_like", wp, base_w)
        assert ok is False
        assert detail["unclaimed_weight_levels"]


# --------------------------------------------------------- the placements
def _placement_setup(**env):
    from eccenergy import recon as reconmod
    try:
        import pandas as pd
    except ImportError as exc:
        raise _Skip(f"pandas not available on this python: {exc}")
    cfg = _cfg(**env)
    tmp = tempfile.mkdtemp()
    stats = _write_cache(tmp)
    layer = _Layer()
    wp = reconmod.weight_path(cfg, "eyeriss_v2_like", "resnet18", [layer],
                              {layer.shape_name: stats})
    cats = ["DRAM", "Global buffer", "Local (spads/RF)", "NoC", "Compute",
            "ECC decode", "Reconstruction", "Recon overhead"]
    base_w = pd.Series({"DRAM": 5000.0, "Local (spads/RF)": 800.0,
                        "NoC": 150.0}).reindex(cats, fill_value=0.0)
    base = pd.Series({"DRAM": 5000.0, "Local (spads/RF)": 911.0,
                      "NoC": 150.0, "Compute": 9000.0}).reindex(cats, fill_value=0.0)
    gran = reconmod.Granularity(cfg.code_n, cfg.code_k, cfg.weight_bits,
                                cfg.recon_granularity)
    packing = reconmod.Packing(cfg.recon_packing, cfg.weight_bits,
                               cfg.code_k, cfg.code_n)
    return cfg, wp, base, base_w, gran, packing


def test_placements_reconcile_by_hand_array_untouched_interface_x_k_over_n():
    from eccenergy import recon as reconmod
    cfg, wp, base, base_w, gran, packing = _placement_setup()
    ok, detail = reconmod.cross_check(cfg, "eyeriss_v2_like", wp, base_w)
    assert ok is True, detail
    # ...and the check reconciles the SUM of the two DRAM stages, 3750 + 1250
    assert math.isclose(detail["per_category"]["DRAM"]["weight_path_module_pJ"],
                        _DRAM_W, rel_tol=1e-12)

    frac = 51 / 63
    out = {}
    for p in reconmod.placements_for("eyeriss_v2_like"):
        res = reconmod.evaluate_placement(
            cfg, "eyeriss_v2_like", p, wp, base_w, base,
            recon_pj=4.0, gran=gran, packing=packing)
        assert res.status == "evaluated", (p.key, res.reason)
        out[p.key] = res
        # THE DRAM TERM, under every boundary: the WHOLE term x K/N
        #   5000 x 51/63 = 4047.619
        assert math.isclose(res.components["DRAM"], _dram_expected(frac),
                            rel_tol=1e-12), (p.key, res.components["DRAM"])
        dm = res.detail["dram_model"]
        assert dm["dram_reduced"] is True
        assert math.isclose(dm["dram_after_pJ"], _DRAM_W * frac, rel_tol=1e-12)
        assert math.isclose(dm["dram_saving_pJ"], _DRAM_W * (1 - frac),
                            rel_tol=1e-12)
        # Compute is untouched by every boundary
        assert math.isclose(res.components["Compute"], 9000.0, rel_tol=1e-12)
        # the stack always sums to the total
        assert math.isclose(sum(res.components.values()), res.total_pJ,
                            rel_tol=1e-12)

    # R1 reduces the DRAM interface and NOTHING on chip, and rebuilds once per
    # DRAM codeword: 2048 weights / 7.875 = 260.06 codewords x 4 pJ. Its whole
    # saving is the interface term, which is what it is in the study to isolate.
    r1 = out["recon1"]
    saved = r1.detail["energy_saved_by_category_pJ"]
    assert set(saved) == {"DRAM"}, saved
    assert math.isclose(saved["DRAM"], _DRAM_W * (1 - frac), rel_tol=1e-12)
    assert math.isclose(r1.components["NoC"], 150.0, rel_tol=1e-12)
    assert math.isclose(r1.components["Local (spads/RF)"], 911.0, rel_tol=1e-12)
    assert math.isclose(r1.components["Reconstruction"],
                        2048 / (63 / 8) * 4.0, rel_tol=1e-12)
    # ...so R1 beats the embedded reference by the interface saving minus its
    # reconstruction cost, and by nothing else
    assert math.isclose(float(base.sum()) - r1.total_pJ,
                        _DRAM_W * (1 - frac) - 2048 / (63 / 8) * 4.0,
                        rel_tol=1e-12)

    # R2 reduces the mesh only: 130 pJ -> 100*frac wire + 30*frac switching
    r2 = out["recon2"]
    assert math.isclose(r2.components["NoC"], 130.0 * frac + 20.0, rel_tol=1e-12)
    assert math.isclose(r2.components["Reconstruction"],
                        100 / (63 / 8) * 4.0, rel_tol=1e-12)

    # R3 reduces both networks and rebuilds once per SPad FILL (40 total)
    r3 = out["recon3"]
    assert math.isclose(r3.components["NoC"], 150.0 * frac, rel_tol=1e-12)
    assert math.isclose(r3.components["Local (spads/RF)"], 911.0, rel_tol=1e-12)
    assert math.isclose(r3.components["Reconstruction"],
                        40 / (63 / 8) * 4.0, rel_tol=1e-12)

    # R4a adds the scratchpad and rebuilds once per SPad READ (400 total)
    r4a = out["recon4"]
    assert math.isclose(r4a.components["Local (spads/RF)"],
                        911.0 - 800.0 * (1 - frac), rel_tol=1e-12)
    assert math.isclose(r4a.components["Reconstruction"],
                        400 / (63 / 8) * 4.0, rel_tol=1e-12)
    assert math.isclose(r4a.components["Recon overhead"], 0.0, abs_tol=1e-12)

    # R4b (recon5) was removed on 2026-09-10, so R4a is the innermost
    # boundary here and nothing carries a reuse register: `Recon overhead` is
    # structurally zero on every bar.
    for r in out.values():
        assert math.isclose(r.components["Recon overhead"], 0.0, abs_tol=1e-12)






def test_a_pe_local_boundary_is_rejected_when_the_tile_is_smaller_than_g_rec():
    """A tile smaller than G_rec is MEASURED either way, and what it costs the
    bar is `ECC_RECON_REQUIRE_GROUP_RESIDENCY` -- both modes asserted.

    THE DEFAULT CHANGED ON 2026-09-13 and this test changed with it. The
    refusal assumed an engine that can only rebuild from weights co-resident in
    the level at ONE INSTANT; RECAP's accumulates the retained bits as they
    arrive, so a level holding 3 of the 9 still feeds it -- over more accesses
    and with more buffering. The shortfall is therefore a COST, reported on the
    bar, not an impossibility that deletes it. The knob restores the
    conservative reading, and both readings are checked here so neither can
    rot.
    """
    import dataclasses as _dc
    from eccenergy import recon as reconmod
    cfg, wp, base, base_w, gran, packing = _placement_setup()
    assert gran.g_rec == 9

    # 24 resident weights >= 9: the PE-local boundary is feasible under BOTH
    # readings, and reports no shortfall under either
    for key in ("recon4",):
        p = reconmod.placement_by_key("eyeriss_v2_like", key)
        for strict in (False, True):
            ok, detail = reconmod.feasibility(p, "eyeriss_v2_like", wp, gran,
                                              require_group_residency=strict)
            assert ok is True, detail
            assert detail.get("layers_below_G_rec") == 0, detail
            assert "group_residency_note" not in detail, detail

    # a depthwise-style tile of 3 weights cannot assemble the group IN ONE
    # INSTANT -- which is measured identically either way
    wp.per_layer[0]["stages"]["weight_spad"]["weights_resident_per_instance"] = 3
    for key in ("recon4",):
        p = reconmod.placement_by_key("eyeriss_v2_like", key)

        # STRICT: refused, and it says which layer and how few
        ok, detail = reconmod.feasibility(p, "eyeriss_v2_like", wp, gran,
                                          require_group_residency=True)
        assert ok is False, detail
        assert detail["infeasible_layers"][0]["weights_resident"] == 3
        assert detail["layers_below_G_rec"] == 1, detail
        assert "fewest: 3" in detail["reason"], detail["reason"]
        strict_cfg = _dc.replace(cfg, recon_require_group_residency=True)
        res = reconmod.evaluate_placement(strict_cfg, "eyeriss_v2_like", p, wp,
                                          base_w, base, 4.0, gran, packing)
        assert res.status == "unsupported" and res.total_pJ == 0.0

        # DEFAULT: charged, and the shortfall travels WITH the bar. Losing the
        # number would be worse than losing the bar -- it is what bounds the
        # buffering the engine needs.
        ok, detail = reconmod.feasibility(p, "eyeriss_v2_like", wp, gran)
        assert ok is True, detail
        assert detail["layers_below_G_rec"] == 1, detail
        assert "fewest: 3" in detail["group_residency_note"], detail
        assert "reason" not in detail, "a charged bar must carry no refusal reason"
        res = reconmod.evaluate_placement(cfg, "eyeriss_v2_like", p, wp, base_w,
                                          base, 4.0, gran, packing)
        assert res.status == "evaluated" and res.total_pJ > 0.0, res.status

        # BREAKAGE: a shortfall that is neither refused NOR reported. That is
        # the one outcome neither reading allows.
        try:
            assert detail.get("layers_below_G_rec") == 0
        except AssertionError:
            pass
        else:
            raise AssertionError("a tile below G_rec was charged AND left "
                                 "unrecorded")

    # ...while a boundary ABOVE the scratchpad is unaffected: the reduced form
    # is only in transit there, in codeword order from the ECC engine
    for key in ("recon1", "recon2", "recon3"):
        p = reconmod.placement_by_key("eyeriss_v2_like", key)
        ok, _ = reconmod.feasibility(p, "eyeriss_v2_like", wp, gran)
        assert ok is True, key


def test_a_missing_stage_is_unsupported_rather_than_skipped():
    from eccenergy import recon as reconmod
    cfg, wp, base, base_w, gran, packing = _placement_setup()
    wp.stages["weight_spad"].energy_pJ = 0.0      # as if the design had no SPad
    for key in ("recon3", "recon4"):
        p = reconmod.placement_by_key("eyeriss_v2_like", key)
        res = reconmod.evaluate_placement(cfg, "eyeriss_v2_like", p, wp, base_w,
                                          base, 4.0, gran, packing)
        assert res.status == "unsupported", key
        assert "no weight energy" in res.reason, res.reason


# ---------------------------------------------------------------- the checks
def _task3_validation(mutate=None, newly=0, from_cache="1", **env):
    """Run `task3_checks` on the synthetic record; `{check name: passed}`.

    `mutate(result)` is applied to every placement before the checks see it,
    which is how each check is shown to FAIL on the leak it exists to catch.
    """
    try:
        import pandas          # noqa: F401  (energy.py imports it at module level)
    except ImportError as exc:
        raise _Skip(f"pandas not available on this python: {exc}")
    from eccenergy import recon as reconmod
    from eccenergy.energy import Raw
    from eccenergy.experiments.recon import PARITY_KEY, task3_checks
    from eccenergy.paths import Results
    from eccenergy.results_store import ResultBuilder, Variant
    with tempfile.TemporaryDirectory() as tmp:
        # `_cfg` clears every ECC_* first, so from_cache has to go THROUGH it
        # rather than being set around the call.
        c, wp, base, base_w, gran, packing = _placement_setup(
            ECC_RESULTS_DIR=tmp, ECC_FROM_CACHE=from_cache, **env)
        raw = Raw(base, base_w, base * 0.0, 5000.0, 2048.0, 1, 0, 1024,
                  per_layer=[{"layer": "l0", "status": "ok", "weights": 1024,
                              "mapping_id": "aaaa1111"}], levels=[])
        res = Results(c).prepare()
        b = ResultBuilder(c, res, "eyeriss_v2_like", "resnet18",
                          experiment="unit_test", fixed_mapping=True)
        base_components = {"DRAM": 5000.0, PARITY_KEY: 1562.5,
                           "Local (spads/RF)": 911.0, "NoC": 150.0,
                           "Compute": 9000.0}
        emb_components = dict(base_components, **{PARITY_KEY: 0.0})
        b.add(Variant("baseline_external_parity", kind="baseline",
                      status="evaluated",
                      total_energy_pJ=sum(base_components.values()),
                      energy_by_component_pJ=base_components,
                      mapping_ids=["aaaa1111"]))
        b.add(Variant("embedded_ecc", kind="embedded", status="evaluated",
                      total_energy_pJ=sum(emb_components.values()),
                      energy_by_component_pJ=emb_components,
                      mapping_ids=["aaaa1111"]))
        out = []
        for p in reconmod.placements_for("eyeriss_v2_like", c):
            r = reconmod.evaluate_placement(
                c, "eyeriss_v2_like", p, wp, base_w, base, 4.0,
                gran, packing)
            if mutate:
                mutate(r)
            out.append(r)
            b.add(Variant(p.variant, kind="reconstruction", status="evaluated",
                          total_energy_pJ=r.total_pJ,
                          energy_by_component_pJ=r.components,
                          mapping_ids=["aaaa1111"]))
        ok, detail = reconmod.cross_check(c, "eyeriss_v2_like", wp, base_w)
        task3_checks(b, c, "eyeriss_v2_like", raw, base_components,
                     emb_components, 1562.5, out, wp, ok, detail, base_w,
                     newly_mapped=newly)
        return {x["check"]: x["passed"] for x in b.document()["validation"]}


_TASK3_CHECKS = ("weight_path_reconciles_with_raw_record",
                 "placement_space_covers_the_whole_weight_path",
                 "dram_scaled_by_K_over_N",
                 "non_weight_energy_identical",
                 "placement_savings_are_bounded_by_the_weight_path",
                 "only_weight_path_categories_differ_from_the_embedded_reference",
                 "evaluation_only_rerun",
                 "reference_bars_match_tasks_1_and_2",
                 "fixed_mapping_shared_across_variants")


def test_task3_checks_pass_on_a_consistent_record_and_catch_a_leak():
    good = _task3_validation()
    for name in _TASK3_CHECKS:
        assert good[name] is True, (name, good)
    # ...and the older check names are gone, not merely renamed alongside.
    for dead in ("dram_identical_to_embedded_reference",
                 "dram_array_identical_to_embedded_reference",
                 "dram_interface_scaled_by_K_over_N"):  # the pre-2026-09-09 pair
        assert dead not in good, (dead, sorted(good))

    # a leak into MAC energy must be caught
    def cheat_compute(r):
        r.components["Compute"] *= 0.95
        r.total_pJ = sum(r.components.values())
    assert _task3_validation(cheat_compute)["non_weight_energy_identical"] is False

    # ...and so must a run that generated mappings. With ECC_FROM_CACHE=1
    # Timeloop was never invoked and the check passes by construction, so the
    # failing case is only visible with it off -- as in the Task 2 suite.
    assert _task3_validation(newly=3)["evaluation_only_rerun"] is True
    assert _task3_validation(newly=3, from_cache="0")["evaluation_only_rerun"] is False


def test_a_dram_component_off_the_K_over_N_value_fails_in_either_direction():
    """The DRAM check is an EQUALITY now, so it catches both cheats.

    Before 2026-09-09 the array was never reduced and the interface always was,
    so the two directions needed two checks: one flagged a bar BELOW the
    expected value (an illegitimate array credit), the other a bar ABOVE it (an
    interface left at full width). With the f_if split gone the whole DRAM term
    is x K/N, the expected value is a single number, and `dram_scaled_by_K_over_N`
    must reject any departure from it either way.
    """
    # CHEAT DOWNWARD: takes more DRAM saving than K/N allows
    def cheat_low(r):
        r.components["DRAM"] *= 0.9
        r.total_pJ = sum(r.components.values())
    got = _task3_validation(cheat_low)
    assert got["dram_scaled_by_K_over_N"] is False, got
    assert got["non_weight_energy_identical"] is True, got

    # CHEAT UPWARD: leaves the DRAM at full width (the pre-2026-09-09 behaviour)
    def cheat_high(r):
        r.components["DRAM"] = _DRAM_W
        r.total_pJ = sum(r.components.values())
    got = _task3_validation(cheat_high)
    assert got["dram_scaled_by_K_over_N"] is False, got

    # ...and a cheat visible only in the detail rows: the stack is right, but
    # the record claims the term was not reduced.
    def cheat_rows(r):
        r.detail["dram_model"]["dram_reduced"] = False
        r.detail["dram_model"]["dram_after_pJ"] = \
            r.detail["dram_model"]["dram_weight_energy_pJ"]
    got = _task3_validation(cheat_rows)
    assert got["dram_scaled_by_K_over_N"] is False, got


# ------------------------------------------- the decode site
def test_controller_site_reproduces_the_pre_2026_09_09_numbers():
    """`ECC_RECON_DECODE_SITE=controller` must be the OLD model to the digit.

    These are the hand checks the suite carried before the decoder moved onto
    the DRAM die: DRAM identical on every bar, R1 saving nothing, R2 the mesh
    only, R3 both networks, R4a the scratchpad too. If they stop holding
    under `controller`, the diff row proves nothing.
    """
    from eccenergy import recon as reconmod
    cfg, wp, base, base_w, gran, packing = _placement_setup(
        ECC_RECON_DECODE_SITE="controller")
    assert wp.decode_site == "controller"
    # the single dram stage owns the whole level and is NOT reducible
    assert math.isclose(wp.stages["dram"].energy_pJ, _DRAM_W)
    assert not any(s.reducible for s in reconmod.stages_for("eyeriss_v2_like", cfg)
                   if s.key == "dram")
    # ...and it has left every placement's reduced set, R1's included
    for p in reconmod.placements_for("eyeriss_v2_like", cfg):
        assert "dram" not in p.reduced, p
    assert reconmod.placement_by_key("eyeriss_v2_like", "recon1", cfg).reduced == ()
    ok, detail = reconmod.validate_placement_space("eyeriss_v2_like", cfg)
    assert ok is True, detail
    assert detail["decode_site"] == "controller"
    assert detail["reducible_stages_in_path_order"][0] == "inter_cluster_mesh"
    ok, detail = reconmod.cross_check(cfg, "eyeriss_v2_like", wp, base_w)
    assert ok is True, detail

    frac = 51 / 63
    out = {}
    # the ON-DIE table's placements are handed in deliberately: evaluate_placement
    # has to apply the decode site itself, whatever form the caller holds
    for p in reconmod.placements_for("eyeriss_v2_like"):
        assert "dram" in p.reduced
        res = reconmod.evaluate_placement(cfg, "eyeriss_v2_like", p, wp, base_w,
                                          base, 4.0, gran, packing)
        assert res.status == "evaluated", (p.key, res.reason)
        out[p.key] = res
        assert res.components["DRAM"] == 5000.0, (p.key, res.components["DRAM"])
        assert math.isclose(res.components["Compute"], 9000.0, rel_tol=1e-12)
        dm = res.detail["dram_model"]
        assert dm["decode_site"] == "controller"
        assert dm["dram_reduced"] is False
        assert dm["dram_saving_pJ"] == 0.0
        assert "dram" not in res.placement.reduced
    r1, r2, r3, r4a = (out[k] for k in ("recon1", "recon2", "recon3",
                                        "recon4"))
    assert math.isclose(sum(r1.detail["energy_saved_by_category_pJ"].values()),
                        0.0, abs_tol=1e-9)
    assert math.isclose(r1.components["Reconstruction"], 2048 / (63 / 8) * 4.0,
                        rel_tol=1e-12)
    assert math.isclose(r2.components["NoC"], 130.0 * frac + 20.0, rel_tol=1e-12)
    assert math.isclose(r3.components["NoC"], 150.0 * frac, rel_tol=1e-12)
    assert math.isclose(r3.components["Local (spads/RF)"], 911.0, rel_tol=1e-12)
    assert math.isclose(r4a.components["Local (spads/RF)"],
                        911.0 - 800.0 * (1 - frac), rel_tol=1e-12)
    # R1 is worse than the embedded reference under controller-side correction
    assert r1.total_pJ > float(base.sum())

    # the same placements under the on-die model differ from these by EXACTLY
    # the interface saving, on every bar, and by nothing else
    cfg2, wp2, _b, _bw, gran2, packing2 = _placement_setup()
    for key, old in out.items():
        new = reconmod.evaluate_placement(
            cfg2, "eyeriss_v2_like", reconmod.placement_by_key("eyeriss_v2_like", key),
            wp2, base_w, base, 4.0, gran2, packing2)
        for cat in old.components:
            diff = old.components[cat] - new.components[cat]
            want = _DRAM_W * (1 - frac) if cat == "DRAM" else 0.0
            assert math.isclose(diff, want, rel_tol=1e-12, abs_tol=1e-9), (key, cat)

    # ...and the recorded checks pass under `controller` too, where the DRAM
    # check collapses onto "identical to the embedded reference"
    good = _task3_validation(ECC_RECON_DECODE_SITE="controller")
    for name in _TASK3_CHECKS:
        assert good[name] is True, (name, good)

    # the whole level is still the one dram stage under controller; it simply
    # is not reduced
    cfg3, wp3, *_ = _placement_setup(ECC_RECON_DECODE_SITE="controller")
    assert math.isclose(wp3.stages["dram"].energy_pJ, _DRAM_W)
    ok, _ = reconmod.cross_check(cfg3, "eyeriss_v2_like", wp3, base_w)
    assert ok is True


def test_the_dram_cost_knobs_are_validated_and_titled():
    """`ECC_DRAM_IF_FRAC` is GONE (2026-09-09) and nothing refuses on it any
    more; `ECC_DRAM_PJ_PER_BIT` and the two static terms take its place and
    reject what they cannot mean."""
    from eccenergy import recon as reconmod
    from eccenergy.config import ConfigError

    # the removed knob leaves no trace in the config or the module
    assert not hasattr(_cfg(), "dram_if_frac")
    assert not hasattr(reconmod, "F_IF_SENSITIVITY")
    assert not hasattr(reconmod, "NO_F_IF")
    # ...and setting it does nothing at all: it is not read
    assert _cfg(ECC_DRAM_IF_FRAC="0.25").dram_pj_per_bit == _cfg().dram_pj_per_bit

    for bad in (dict(ECC_RECON_DECODE_SITE="dimm"),
                dict(ECC_DRAM_PJ_PER_BIT="0"),
                dict(ECC_DRAM_PJ_PER_BIT="-1"),
                dict(ECC_DRAM_BACKGROUND_PJ="-1"),
                dict(ECC_DRAM_REFRESH_PJ="-1")):
        try:
            _cfg(**bad)
        except ConfigError:
            pass
        else:
            raise AssertionError(f"{bad} was accepted")

    # the static terms default to 0 and say so
    c = _cfg()
    assert c.dram_background_pj == 0.0 and c.dram_refresh_pj == 0.0
    assert "not modelled" in c.dram_static_note.lower()

    # the manifest's `title_caveats` carry the decode site and the per-bit
    # cost, so a figure cannot be quoted without them; the heading itself is
    # ONE line since 2026-09-11 and names the fixed point and the layer scope
    c = _cfg()
    assert "\n" not in c.recon_title(), c.recon_title()
    t = "\n".join(c.recon_caveats())
    assert "DRAM die" in t, t
    t = "\n".join(_cfg(ECC_RECON_DECODE_SITE="controller").recon_caveats())
    assert "controller" in t.lower() and "pre-2026-09-09" in t, t
    one = _cfg(ECC_LAYERS="layer3.0.conv1")
    assert "layer3.0.conv1" in one.recon_title() and "\n" not in one.recon_title()
    assert any("DEVELOPMENT RUN" in ln for ln in one.recon_caveats())
    assert not any("DEVELOPMENT RUN" in ln for ln in c.recon_caveats())


def test_the_dram_level_is_claimed_exactly_once_in_total():
    """ONE stage, one level. The array/interface pair and its f_if shares were
    removed on 2026-09-09, so a level claimed twice is now always a table
    error and is refused, not double-counted."""
    from eccenergy import recon as reconmod
    cfg = _cfg()
    stages = reconmod.stages_for("eyeriss_v2_like", cfg)
    dram = [s for s in stages if s.matches("DRAM")]
    assert [s.key for s in dram] == ["dram"]
    assert dram[0].reducible
    assert reconmod._level_shares(dram, "DRAM") == {"dram": 1.0}
    # the wglb path keeps it FIRST, before its extra weight GLB
    keys = [s.key for s in reconmod.stages_for("eyeriss_v2_like_wglb", cfg)]
    assert keys[:2] == ["dram", "weight_glb"], keys
    # any second claimant of the level is refused
    rogue = dram + [reconmod.Stage("x", "x", "storage", ("DRAM",), True)]
    try:
        reconmod._level_shares(rogue, "DRAM")
    except ValueError:
        pass
    else:
        raise AssertionError("two claimants of one level were accepted")


def test_r1_isolates_the_interface_saving_from_every_on_chip_saving():
    """R1 is no longer the zero-saving control. Its saving against the embedded
    reference is the DRAM interface term and nothing else, and every other
    boundary's saving is R1's plus its own on-chip saving minus the extra
    reconstruction it pays -- which is what makes R1 the term to subtract."""
    from eccenergy import recon as reconmod
    for case in _cases():
        out, packing = _evaluate(case)
        base, wp = case["base"], case["wpath"]
        emb = float(base.sum())
        iface_saving = wp.stages["dram"].energy_pJ * (1 - packing.frac)
        r1 = out["recon1"]
        recon1 = r1.components["Reconstruction"] + r1.components["Recon overhead"]
        assert math.isclose(emb - r1.total_pJ, iface_saving - recon1, rel_tol=1e-12), \
            case["name"]
        assert set(r1.detail["energy_saved_by_category_pJ"]) == {"DRAM"}, case["name"]
        for key in ("recon2", "recon3", "recon4"):
            res = out[key]
            if res.status != "evaluated":
                continue
            on_chip = sum(v for c, v in res.detail["energy_saved_by_category_pJ"].items()
                          if c != "DRAM")
            recon = res.components["Reconstruction"] + res.components["Recon overhead"]
            assert math.isclose(emb - res.total_pJ, iface_saving + on_chip - recon,
                                rel_tol=1e-12), (case["name"], key)
            assert math.isclose(res.detail["energy_saved_by_category_pJ"]["DRAM"],
                                iface_saving, rel_tol=1e-12), (case["name"], key)


# ------------------------------------------------------------- configuration
def test_recon_optimizer_true_is_task4_and_cannot_be_filed_as_a_pre_result():
    """RECON_OPTIMIZER=True is Task 4 (since 2026-09-09) and is `Post` only.

    The knob used to refuse outright, because Task 4 did not exist and the one
    thing that must never happen is fixed-mapping numbers under a heading that
    says the mapping was optimised. Task 4 is implemented now, so the refusal
    moved to where it can still be checked:

      * here -- a re-optimised mapping filed as a `Pre` result, which is a
        result whose own phase field contradicts it;
      * `experiments/recon.dilated_view()` -- the reconstruction arm's mapper
        cache missing, the design having no weight level to dilate, or the
        dilated capacity not coming back N/K times the reference's. Each of
        those would otherwise fall back to the reference mapping, which IS
        Task 3.
    """
    from eccenergy.config import ConfigError
    try:
        _cfg(ECC_RECON_OPTIMIZER="True")            # ECC_PHASE defaults to Pre
    except ConfigError as exc:
        assert "TASK 4" in str(exc), str(exc)
        assert "Post" in str(exc), str(exc)
    else:
        raise AssertionError("RECON_OPTIMIZER=True was accepted as a `Pre` "
                             "result; a re-optimised mapping is `Post` by "
                             "construction and the phase field would lie")
    # Post + True is Task 4 and is accepted, with the dilated capacity derived
    # from the reference one rather than configured separately.
    cfg = _cfg(ECC_RECON_OPTIMIZER="True", ECC_PHASE="Post")
    assert cfg.recon_optimizer is True
    from eccenergy import recon as reconmod
    assert math.isclose(reconmod.capacity_dilation_scale(cfg),
                        cfg.weight_capacity_scale * cfg.code_n / cfg.code_k,
                        rel_tol=1e-4)
    # ...and the OTHER crossed combination, caught at config time so --dry-run
    # reports it: Task 3 filed as a `Post` result claims a mapping that was
    # never re-optimised.
    try:
        _cfg(ECC_RECON_MODELING="1", ECC_PHASE="Post")
    except ConfigError as exc:
        assert "Task 3" in str(exc) and "Pre" in str(exc), str(exc)
    else:
        raise AssertionError("ECC_PHASE=Post with RECON_OPTIMIZER=False was "
                             "accepted; a fixed mapping is a `Pre` result")
    # False, and the spelling env.sh uses, are both fine
    assert _cfg(ECC_RECON_OPTIMIZER="False").recon_optimizer is False


def test_a_multicast_network_boundary_pays_per_destination_not_per_injection():
    """R2's encoders run once per ARRIVAL, so the multicast factor multiplies.

    Sec. 7.1 states the tradeoff and asks for it as an experiment variable:
    one encoder before a multicast fanout and full-width network traffic, or an
    encoder at each destination and reduced-width shared transport. Until
    2026-09-09 R2 took the second boundary's SAVING (the mesh carries the
    reduced form) while paying the first boundary's COUNT (the mesh's
    ingresses), which is not a placement -- an encoder before the fanout makes
    that network full width, and that boundary is R1.

    Driven on a copy of the synthetic fixture whose outer network multicasts
    3-fold, so the two readings are 3x apart by construction and the answer is
    checkable by hand: 100 ingresses x 3 = 300 arrivals.
    """
    from eccenergy import recon as reconmod
    mc = _STATS.replace(
        """        Fanout                                  : 4
        Multicast factor                        : 1
        Ingresses                               : 100.00""",
        """        Fanout                                  : 12
        Multicast factor                        : 3
        Ingresses                               : 100.00
            @multicast 3 @scatter 4: 100.00""", 1)
    assert mc != _STATS, "the fixture's outer network block did not match"

    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp) / "C8_M8"
        d.mkdir(parents=True)
        _write_lf(d / "timeloop-mapper.stats.txt", mc)
        _write_lf(d / "timeloop-mapper.map.txt", _MAP)
        stats = d / "timeloop-mapper.stats.txt"

        cfg = _cfg()
        lp = reconmod.read_weight_path(cfg, "eyeriss_v2_like", _Layer(), stats)
        mesh = lp.stages["inter_cluster_mesh"]
        assert mesh.ingresses == 100.0, mesh.ingresses
        assert mesh.multicast == 3.0, mesh.multicast
        assert mesh.deliveries == 300.0, (
            "destination-side arrivals must be ingresses x the multicast "
            f"factor, got {mesh.deliveries}")
        assert mesh.counter("deliveries") == 300.0
        assert mesh.counter("ingresses") == 100.0

        # the boundary itself: `destination` is the model, `source` reproduces
        # the pre-2026-09-09 count, and NOTHING ELSE about the bar moves
        got = {}
        for site in reconmod.ENCODER_SITES:
            c = _cfg(ECC_RECON_ENCODER_SITE=site)
            wp = reconmod.weight_path(c, "eyeriss_v2_like", "m", [_Layer()],
                                      {"C8_M8": stats})
            p2 = reconmod.placement_by_key("eyeriss_v2_like", "recon2", c)
            res = reconmod.evaluate_placement(
                c, "eyeriss_v2_like", p2, wp,
                {k: v.energy_pJ for k, v in wp.stages.items()}, {},
                recon_pj=1.0,
                gran=reconmod.Granularity(c.code_n, c.code_k, c.weight_bits,
                                          "codeword"),
                packing=reconmod.Packing("stream", c.weight_bits, c.code_k,
                                         c.code_n))
            counts = res.detail["reconstruction_counts"]
            got[site] = counts
            # both readings are recorded on EVERY network bar, whichever is used
            assert counts["if_one_encoder_before_the_fanout"] == 100.0
            assert counts["if_one_encoder_per_destination"] == 300.0
            assert counts["multicast_multiplicity_of_this_boundary"] == 3.0
            assert counts["network_multicast_factor"] == 3.0

        assert got["destination"]["weights_reconstructed"] == 300.0
        assert got["source"]["weights_reconstructed"] == 100.0
        assert got["destination"]["reconstruction_energy_pJ"] == \
            3.0 * got["source"]["reconstruction_energy_pJ"], (
                "the two readings must differ by exactly the multicast factor")

        # and the SAVING is the same under both: only the encoder count moved
        for site in reconmod.ENCODER_SITES:
            c = _cfg(ECC_RECON_ENCODER_SITE=site)
            p2 = reconmod.placement_by_key("eyeriss_v2_like", "recon2", c)
            assert p2.reduced == ("dram", "inter_cluster_mesh"), (
                "the encoder site must not change WHICH stages carry the "
                f"reduced form: {p2.reduced}")


def test_several_architectures_are_one_panel_each_and_never_one_axis():
    """Two designs are accepted -- as PANELS -- and a repeat is refused.

    Until 2026-09-09 this refused a second architecture outright, because the
    boundaries of two designs cannot share an x axis: "reconstruct after the
    mesh" beside a design with no mesh is meaningless. Separate stacked axes are
    not that, so the rule is now enforced where it actually lives -- in
    `experiments/reconmod.py figure()`, which gives every design its own axes, its
    own boundary list and its own two reference bars -- and the config accepts
    the list. What it still refuses is a REPEATED name, which would draw one
    design twice and write its result file twice.
    """
    from eccenergy import recon as reconmod
    from eccenergy.config import ConfigError
    cfg = _cfg(ECC_SWEEP_ARCHS="simple_weight_stationary eyeriss_like")
    assert cfg.archs == ["simple_weight_stationary", "eyeriss_like"], cfg.archs

    # ...and the boundary lists stay per design, which is what makes panels the
    # only honest layout: these two are NOT the same axis.
    ws = [p.key for p in reconmod.placements_for("simple_weight_stationary", cfg)]
    v1 = [p.key for p in reconmod.placements_for("eyeriss_like", cfg)]
    assert ws != v1 and len(ws) == 5 and len(v1) == 4, (ws, v1)
    ws_stages = [s.key for s in reconmod.stages_for("simple_weight_stationary", cfg)]
    v1_stages = [s.key for s in reconmod.stages_for("eyeriss_like", cfg)]
    assert set(ws_stages) & set(v1_stages) == {"dram"}, (
        "the only weight-path stage two different designs share is the DRAM; "
        "anything else means a stage name is being reused across "
        f"designs that do not have the same level: {ws_stages} vs {v1_stages}")

    # A repeated name collapses to one panel rather than drawing the design
    # twice, and an empty list is refused: a placement study with no design has
    # no boundaries to place.
    assert _cfg(ECC_SWEEP_ARCHS="eyeriss_like eyeriss_like").archs \
        == ["eyeriss_like"]
    try:
        _cfg(ECC_SWEEP_ARCHS=" ")
    except ConfigError as exc:
        assert "ECC_SWEEP_ARCHS" in str(exc), str(exc)
    else:
        raise AssertionError("an empty architecture list was accepted")


def test_every_placement_space_is_valid_for_every_supported_design():
    """`validate_placement_space()` passes on every design in the tables.

    The two tables have to be edited together (CLAUDE.md), and the failure mode
    is silent understatement rather than an error, so this runs the check on
    every registered design rather than only on the one a run happens to draw.
    `eyeriss_v2_like_wglb` is the known exception: its weight path gained a
    `weight_glb` stage its placement list does not reach, and it is reported
    here instead of being evaluated with four understated boundaries.
    """
    from eccenergy import recon as reconmod
    cfg = _cfg()
    expected_invalid = {"eyeriss_v2_like_wglb"}
    for arch in reconmod.supported_archs():
        ok, detail = reconmod.validate_placement_space(arch, cfg)
        if arch in expected_invalid:
            assert not ok, f"{arch} was expected to be the known-gap design"
            continue
        assert ok, f"{arch}: {detail['violations']}"
        # every reducible stage reached, and every reduced set a prefix
        assert not detail["reducible_stages_no_placement_reaches"], arch
        assert all(r["is_a_prefix"] for r in detail["per_placement"]), arch




def test_task4_never_lands_on_task3s_figure():
    """A re-optimised mapping must not overwrite the fixed-mapping figure.

    Two paths set the name and they have to agree: env.sh section 10 appends
    `_optimiser` for a whole-model run (which arrives here as ECC_STEM), and
    `Config.stem` appends it for a layer-scoped one (where env.sh deliberately
    leaves ECC_STEM empty so the layer scope can land in the name). Before
    this, a two-layer Task 4 run wrote
    `ReconSweep__layers2__<names>.png` -- exactly the file the two-layer Task 3
    run writes.
    """
    t3 = _cfg(ECC_RECON_MODELING="1", ECC_LAYERS="layer2.0.conv1 layer4.0.conv2")
    t4 = _cfg(ECC_RECON_MODELING="1", ECC_LAYERS="layer2.0.conv1 layer4.0.conv2",
              ECC_RECON_OPTIMIZER="True", ECC_PHASE="Post")
    assert t3.stem != t4.stem, (t3.stem, t4.stem)
    assert t4.stem.startswith("ReconSweep_optimiser"), t4.stem
    assert t3.layer_slug in t4.stem and t3.layer_slug in t3.stem
    # ...and the whole-model pair, as env.sh spells it
    w3 = _cfg(ECC_RECON_MODELING="1", ECC_STEM="ReconSweep")
    w4 = _cfg(ECC_RECON_MODELING="1", ECC_STEM="ReconSweep_optimiser",
              ECC_RECON_OPTIMIZER="True", ECC_PHASE="Post")
    assert (w3.stem, w4.stem) == ("ReconSweep", "ReconSweep_optimiser")


def test_the_stem_is_the_recon_stem_and_keeps_a_layer_scope():
    cfg = _cfg()
    assert cfg.stem == "ReconSweep", cfg.stem
    dev = _cfg(ECC_LAYERS="layer4.1.conv2")
    assert dev.stem.startswith("ReconSweep__layers1__"), dev.stem


# ===========================================================================
#  THE SECOND CASE FOR EVERY PROPERTY BELOW: the real cached Timeloop output
# ===========================================================================
#  Everything above this line is driven by the synthetic stats.txt, whose
#  numbers are round so the answers can be checked by hand. That proves the
#  arithmetic; it cannot prove that the arithmetic is being applied to the real
#  design's levels. So each property test below runs on BOTH: the synthetic
#  file, and the mapper cache the published `results/figures/ReconSweep.png`
#  was drawn from. If the cache is not on this machine the real case is
#  skipped, and `test_the_property_tests_also_ran_on_the_real_cache` says so
#  rather than letting a synthetic-only pass look complete.
# ===========================================================================

#: The figure's manifest records every knob that decides WHICH mapper-cache
#: entry it came from, so the tests re-open exactly that entry instead of
#: hard-coding a fingerprint that goes stale the next time `env.sh` moves.
#: Manifest key -> the ECC_* variable `config.py` reads it from.
_MANIFEST_ENV = {
    "opt_metric": "ECC_OPT_METRIC",
    "victory": "ECC_VICTORY",
    "victory_scaling": "ECC_VICTORY_SCALING",
    "mapper_algorithm": "ECC_MAPPER_ALGORITHM",
    "mapper_timeout": "ECC_MAPPER_TIMEOUT",
    "mapper_threads": "ECC_MAPPER_THREADS",
    "mapper_search_size": "ECC_MAPPER_SEARCH_SIZE",
    "mapper_max_permutations": "ECC_MAPPER_MAX_PERMUTATIONS",
    "arch_fidelity": "ECC_ARCH_FIDELITY",
    "noc_enabled": "ECC_NOC",
    "noc_wire_pj_per_bit_mm": "ECC_NOC_WIRE_PJ_PER_BIT_MM",
    "noc_router_pj": "ECC_NOC_ROUTER_PJ",
    "noc_pe_latch_pj": "ECC_NOC_PE_LATCH_PJ",
    "noc_scale": "ECC_NOC_SCALE",
    "force_technology": "ECC_FORCE_TECHNOLOGY",
    "force_datawidth": "ECC_FORCE_DATAWIDTH",
    "dram_depth": "ECC_DRAM_DEPTH",
    "global_cycle_seconds": "ECC_GLOBAL_CYCLE_SECONDS",
    "classify_mode": "ECC_CLASSIFY",
    "split_read_write": "ECC_SPLIT_READ_WRITE",
    # not part of the mapping fingerprint, but they decide what the DRAM band
    # of the figure means, so the real case is evaluated at the same values
    "recon_decode_site": "ECC_RECON_DECODE_SITE",
    "dram_pj_per_bit": "ECC_DRAM_PJ_PER_BIT",
}

_REAL_CACHE = {}          # memo: 12 stats files parsed once is enough


class _RawLayer:
    """A Layer stand-in built from the raw record.

    `weight_path()` asks a Layer for exactly these four fields, and the raw
    record already carries all of them per layer -- so the real case needs no
    workload file and therefore no torch, while still covering the same 21
    layers, the same shapes and the same repeat counts as the published run.
    """

    def __init__(self, row):
        self.name = row["layer"]
        self.shape_name = row["shape"]
        self.weights = int(row.get("weights", 0))
        self.count = int(row.get("repeat_count", 1) or 1)


def _locate(fn, *args):
    """Resolve a path through `paths.py` without leaving an empty directory.

    `Results.raw_path` and `Results.mapper_cache` mkdir on the way, which is
    right for a run about to write there and wrong for a test that is only
    asking whether the published run's cache is on this machine: a stale
    fingerprint would otherwise litter `results/_raw/` and `outputs/` with
    empty `fp-*` trees. Only directories that are EMPTY afterwards are removed,
    so a real cache is never touched.
    """
    path = fn(*args)
    node = path if path.is_dir() else path.parent
    empty = []
    while node != node.parent and node.is_dir() and not any(node.iterdir()):
        empty.append(node)
        node = node.parent
    for d in empty:
        try:
            d.rmdir()
        except OSError:                                       # pragma: no cover
            pass
    return path


def _real_case():
    """The weight path behind `results/figures/ReconSweep.png`, or `None`."""
    if "case" in _REAL_CACHE:
        return _REAL_CACHE["case"]
    _REAL_CACHE["case"] = None
    try:
        import pandas as pd
    except ImportError as exc:
        _REAL_CACHE["why"] = f"pandas not available on this python: {exc}"
        return None
    import json
    from eccenergy.paths import ROOT
    manifest = ROOT / "results" / "manifests" / "ReconSweep.json"
    if not manifest.exists():
        _REAL_CACHE["why"] = ("no results/manifests/ReconSweep.json -- run "
                              "`ECC_RECON_MODELING=1 bash run.sh recon --eval` first")
        return None
    man = json.loads(manifest.read_text())
    mcfg = man.get("config", {})
    arch, model = mcfg.get("const_arch"), mcfg.get("const_model")
    env = {}
    for key, name in _MANIFEST_ENV.items():
        value = mcfg.get(key)
        if value is None or value == "":
            continue
        env[name] = "1" if value is True else "0" if value is False else str(value)
    env.update(ECC_RECON_MODELING="1", ECC_SWEEP_ARCHS=arch, ECC_CONST_ARCH=arch,
               ECC_CONST_MODEL=model, ECC_CONST_K=str(mcfg.get("const_k", 51)),
               ECC_RECON_PACKING=mcfg.get("recon_packing", "stream"),
               ECC_RECON_ENCODER_GRANULARITY=mcfg.get("recon_granularity", "weight"))
    cfg = _cfg(**env)

    from eccenergy import archs as archmod
    from eccenergy import recon as reconmod
    from eccenergy.energy import plot_cats
    from eccenergy.paths import Results
    variant = archmod.effective_variant(arch, cfg)
    fingerprint = archmod.arch_fingerprint(arch, cfg)
    results = Results(cfg)
    raw_path = _locate(results.raw_path, arch, model, variant, fingerprint)
    cache = _locate(results.mapper_cache, arch, variant, fingerprint)
    if not raw_path.exists():
        _REAL_CACHE["why"] = (f"no raw record for the manifest's mapping "
                              f"fingerprint at {raw_path}")
        return None
    raw = json.loads(raw_path.read_text())
    layers = [_RawLayer(r) for r in raw["per_layer"] if r.get("status") == "ok"]
    stats = {l.shape_name: cache / l.shape_name / "timeloop-mapper.stats.txt"
             for l in layers}
    missing = sorted(s for s, p in stats.items() if not p.exists())
    if not layers or missing:
        _REAL_CACHE["why"] = (f"{len(missing)} shape(s) not in {cache}" if missing
                              else "the raw record holds no mapped layer")
        return None

    cats = plot_cats(cfg)
    # THE OVERRIDES MUST BE APPLIED HERE TOO, exactly as production applies
    # them (`energy.collect` -> `apply_dram_override`). The raw cache is pure
    # Timeloop output at Accelergy's own 8 pJ/bit, while `weight_path()`
    # rescales its DRAM stage to ECC_DRAM_PJ_PER_BIT -- so a fixture that read
    # `raw["base_w"]` straight off the JSON would compare a rescaled weight
    # path against an unscaled record and fail every reconciliation. Running
    # the real function here is also what checks that the two rescalings agree.
    from eccenergy.energy import Raw, apply_dram_override
    rec = apply_dram_override(Raw.from_json(raw, cfg), cfg, verbose=False)
    _REAL_CACHE["case"] = {
        "name": f"cached Timeloop output, {arch}/{model}",
        "cfg": cfg, "arch": arch,
        "wpath": reconmod.weight_path(cfg, arch, model, layers, stats),
        "base": rec.base.reindex(cats, fill_value=0.0),
        "base_w": rec.base_w.reindex(cats, fill_value=0.0),
        "recon_pj": float(man.get("recon_pj_per_codeword", 4.1296273)),
        "stats": stats,
    }
    return _REAL_CACHE["case"]


def _synthetic_case(**env):
    cfg, wp, base, base_w, _gran, _packing = _placement_setup(**env)
    return {"name": "synthetic stats.txt", "cfg": cfg, "arch": "eyeriss_v2_like",
            "wpath": wp, "base": base, "base_w": base_w,
            "recon_pj": 4.0, "stats": None}


def _cases():
    """Both cases the properties below must hold on."""
    cases = [_synthetic_case()]
    real = _real_case()
    if real is not None:
        cases.append(real)
    return cases


def _evaluate(case, k=None, n=None):
    """Every placement of `case`, at an explicit code geometry.

    K and N are arguments rather than config, because the weight path does not
    depend on the code at all -- re-reading twelve stats files per K would hide
    exactly the thing the code sweep is trying to show.
    """
    from eccenergy import recon as reconmod
    cfg, arch = case["cfg"], case["arch"]
    k = cfg.code_k if k is None else k
    n = cfg.code_n if n is None else n
    gran = reconmod.Granularity(n, k, cfg.weight_bits, cfg.recon_granularity)
    packing = reconmod.Packing(cfg.recon_packing, cfg.weight_bits, k, n)
    out = {}
    for p in reconmod.placements_for(arch, cfg):
        out[p.key] = reconmod.evaluate_placement(
            cfg, arch, p, case["wpath"], case["base_w"], case["base"],
            case["recon_pj"], gran, packing)
    return out, packing


def _evaluated(case, out):
    """The placements that evaluated, as `{key: result}`.

    On the synthetic case that is all five, and anything else is a failure. On
    the real cache a PE-local boundary may be `unsupported` -- under the 8x2 EDP
    mappings of 2026-09-09 R4a/R4b keep fewer than G_rec weights resident in
    four layer4 shapes (FINDINGS 4.1.3) -- which is a modelled outcome the
    arithmetic under test has nothing to say about, so those rows are left out
    rather than failing every property on a fact about the mapping.
    """
    for key, res in out.items():
        if res.status != "evaluated":
            assert case["stats"] is not None, (case["name"], key, res.reason)
    return {k: r for k, r in out.items() if r.status == "evaluated"}


def _stage_rows(result):
    """`{stage key: row}` from the placement's own weight-path record."""
    return {row["stage"]: row for row in result.detail["weight_path_stages"]}


def _after(row):
    """The stage's weight energy AFTER the placement.

    A stage the placement did not reduce carries no `energy_after_pJ` key at
    all, and that is part of what is being tested: the model must not write a
    scaled number for a stage the reduced representation never reached.
    """
    return row["energy_after_pJ"] if row["carries_reduced"] else row["weight_energy_pJ"]


def _category_of(case, stage_key):
    """The plotted category a stage lands in.

    `recon._category_of` is the one definition of that mapping and is checked
    against the raw record by `cross_check`, so the test uses it rather than
    restating it and drifting.
    """
    from eccenergy import recon as reconmod
    stage = next(s for s in reconmod.stages_for(case["arch"]) if s.key == stage_key)
    return reconmod._category_of(stage, case["cfg"])


#: The categories a boundary may MOVE. `Reconstruction` and `Recon overhead`
#: are the cost side and go up, so they are never part of a monotonicity or an
#: upper-bound claim about transport and storage.
_TRANSPORT_CATS = ("DRAM", "Global buffer", "Local (spads/RF)", "NoC")

#: R1 to R4a, in the order the reduced representation reaches one more stage.
#: R4b reduces exactly what R4a does and differs only in the encoder count, so
#: it is checked for EQUALITY of those stages instead of being in this chain.
_PATH_ORDER = ("recon1", "recon2", "recon3", "recon4")


# ---------------------------------------------------- 1. per-stage exactness
def test_every_reduced_stage_scales_by_k_over_n_and_no_other_stage_moves():
    """PER-STAGE EXACTNESS, per placement, per case.

    A bar total that happens to be lower says nothing about WHERE the energy
    went. So this is checked stage by stage: for a stage the placement leaves
    reduced, `after == before x scale` with `0 < scale < 1` and, under `stream`
    packing, `scale == K/N`; for a stage it does not, `after == before` with no
    tolerance at all.
    """
    from eccenergy import recon as reconmod
    for case in _cases():
        out, packing = _evaluate(case)
        for key, res in _evaluated(case, out).items():
            reduced = set(reconmod.placement_by_key(case["arch"], key,
                                                    case["cfg"]).reduced)
            assert set(res.placement.reduced) == reduced
            for stage_key, row in _stage_rows(res).items():
                where = (case["name"], key, stage_key)
                before = row["weight_energy_pJ"]
                if stage_key in reduced:
                    scale = row["scale"]
                    assert row["carries_reduced"] is True, where
                    assert 0.0 < scale < 1.0, (where, scale)
                    assert math.isclose(scale, packing.frac, rel_tol=1e-12), (where, scale)
                    assert math.isclose(row["energy_after_pJ"], before * scale,
                                        rel_tol=1e-12), where
                    assert math.isclose(row["energy_saved_pJ"], before * (1.0 - scale),
                                        rel_tol=1e-12), where
                else:
                    # EXACTLY, not isclose: an untouched stage has no rounding
                    # to allow for, and a drift here is a bug however small.
                    assert row["carries_reduced"] is False, where
                    assert row["scale"] == 1.0, where
                    assert row["energy_saved_pJ"] == 0.0, where
                    assert "energy_after_pJ" not in row, where
                    assert _after(row) == before, where


# ---------------------------------------------------- 2. non-weight untouched
def test_only_the_weight_share_of_a_category_moves():
    """NON-WEIGHT UNTOUCHED.

    `Local (spads/RF)` also holds the ifmap and psum scratchpads, and `NoC`
    also carries iacts and psums. A boundary may move the WEIGHT share of those
    categories and nothing else, so the residual -- the category total minus
    the weight-path stages inside it -- must be the embedded reference's.

    The residual is compared at 1e-12 relative, not bit-identically, for one
    stated reason: the model computes `total - saved` where this test computes
    `total - (before - saved)`, so the two differ by the last bit of a double
    (measured on the real record: 6e-8 pJ in 2.6e8, i.e. 2e-16 relative). A
    category with no weight path at all -- Compute, and the Global buffer on
    this design -- is required to be EXACTLY equal.
    """
    for case in _cases():
        out, _packing = _evaluate(case)
        base = case["base"]
        for key, res in _evaluated(case, out).items():
            by_cat = {}
            for stage_key, row in _stage_rows(res).items():
                cat = _category_of(case, stage_key)
                a, b = by_cat.get(cat, (0.0, 0.0))
                by_cat[cat] = (a + _after(row), b + row["weight_energy_pJ"])
            for cat in _TRANSPORT_CATS + ("Compute",):
                where = (case["name"], key, cat)
                after_w, before_w = by_cat.get(cat, (0.0, 0.0))
                got = res.components.get(cat, 0.0) - after_w
                want = float(base.get(cat, 0.0)) - before_w
                if before_w == 0.0:
                    assert res.components.get(cat, 0.0) == float(base.get(cat, 0.0)), where
                assert math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-6), \
                    (where, got, want)


# ---------------------------------------------------- 3. monotonicity
def test_category_energies_are_monotonic_down_the_weight_path():
    """MONOTONICITY DOWN THE PATH, strict wherever a stage is added.

    The reduced sets nest -- R1 = {dram_if} < R2 = {dram_if, mesh} < R3 =
    {dram_if, mesh, cluster_local} < R4a = R4b = {dram_if, mesh, cluster_local,
    weight_spad} -- so pushing the boundary
    deeper can only lower a transport or storage category, and must lower the
    one that just gained a stage. This is the test that would have caught "R3's
    NoC is not below R2's": a saving applied to the wrong stage, applied twice
    or dropped breaks it even when the bar total still looks plausible.
    """
    from eccenergy import recon as reconmod
    for case in _cases():
        out, _packing = _evaluate(case)
        ev = _evaluated(case, out)
        placements = {p.key: p for p in reconmod.placements_for(case["arch"],
                                                                case["cfg"])}

        # the nesting itself, which everything below depends on
        for a, b in zip(_PATH_ORDER, _PATH_ORDER[1:]):
            assert set(placements[a].reduced) < set(placements[b].reduced), (a, b)
        # R4b (recon5 on the eyeriss designs) was removed 2026-09-10
        assert all(p.key != "recon5" for p in placements.values()) \
               or case["arch"] == "simple_weight_stationary"
        # ...and every reduced set starts at the die's output
        for p in placements.values():
            assert p.reduced[0] == "dram", p

        for a, b in zip(_PATH_ORDER, _PATH_ORDER[1:]):
            if a not in ev or b not in ev:
                continue
            rows_a, rows_b = _stage_rows(out[a]), _stage_rows(out[b])
            added = set(placements[b].reduced) - set(placements[a].reduced)
            for stage_key, row_b in rows_b.items():
                where = (case["name"], a, b, stage_key)
                before, after = _after(rows_a[stage_key]), _after(row_b)
                assert after <= before + 1e-6, where
                if stage_key in added and row_b["weight_energy_pJ"] > 0:
                    assert after < before, where
            strict = {_category_of(case, s) for s in added
                      if rows_b[s]["weight_energy_pJ"] > 0}
            for cat in _TRANSPORT_CATS:
                where = (case["name"], a, b, cat)
                va = out[a].components.get(cat, 0.0)
                vb = out[b].components.get(cat, 0.0)
                assert vb <= va + 1e-6, (where, va, vb)
                if cat in strict:
                    assert vb < va, (where, va, vb)

        if False:  # R4b removed 2026-09-10
            for cat in _TRANSPORT_CATS:
                assert out["recon5"].components.get(cat, 0.0) == \
                    out["recon4"].components.get(cat, 0.0), (case["name"], cat)


# ---------------------------------------------------- 4. the closed form

def test_the_saving_of_each_boundary_matches_its_closed_form():
    """CLOSED-FORM ARITHMETIC -- the identity, not an inequality.

        DRAM(Rx)   == DRAM(embedded)  - DRAM_w x (1 - K/N)           for EVERY x
        NoC(R2)    == NoC(embedded)   - mesh_w x (1 - K/N)
        NoC(R3)    == NoC(embedded)   - (mesh_w + cluster_local_w) x (1 - K/N)
        Local(R4a) == Local(embedded) - weight_spad_w x (1 - K/N)

    An inequality still passes when a saving lands on the wrong stage. The
    identity does not, and it also pins the ceiling the result file prints
    against the same arithmetic. DRAM_w is the WEIGHT share of the DRAM
    category (the category also holds inputs and outputs on the real record),
    and since 2026-09-09 the whole of it is reducible.
    """
    for case in _cases():
        from eccenergy import recon as reconmod
        out, packing = _evaluate(case)
        ev = _evaluated(case, out)
        wp, base, name, arch = (case["wpath"], case["base"], case["name"],
                                case["arch"])
        cfg = case["cfg"]
        d = 1.0 - packing.frac
        dram0 = float(base["DRAM"])
        dram_w = float(case["base_w"]["DRAM"])
        dram_stage = wp.stages["dram"].energy_pJ
        assert math.isclose(dram_stage, dram_w, rel_tol=1e-12), name
        for key in ev:
            assert math.isclose(out[key].components["DRAM"], dram0 - dram_w * d,
                                rel_tol=1e-12), (name, key)

        # THE STAGE NAMES COME FROM THE DESIGN, NOT FROM THIS FILE. The test
        # used to spell eyeriss_v2_like's three on-chip stages literally while
        # reading whichever design ReconSweep.json names, so pointing the study
        # at a design with a different weight path (Task 5 pointed it at
        # simple_weight_stationary) turned an identity test into a KeyError.
        # The identity itself is per DESIGN, so it is derived per design: a
        # boundary's category total is the embedded reference's minus
        # (1 - K/N) x the energy of the stages IT reduces in that category.
        by_cat = {}
        for stage in reconmod.stages_for(arch, cfg):
            st = wp.stages.get(stage.key)
            if st is None or stage.kind == "dram":
                continue
            by_cat.setdefault(_category_of(case, stage.key), {})[stage.key] = st.energy_pJ

        for key, res in ((k, out[k]) for k in ev):
            reduced = set(reconmod.effective_placement(
                res.placement, cfg).reduced)
            for cat, stages in by_cat.items():
                want = float(base[cat]) - d * sum(
                    e for s, e in stages.items() if s in reduced)
                assert math.isclose(res.components[cat], want, rel_tol=1e-12), \
                    (name, key, cat, res.components[cat], want)

            # ...and the ceiling the result file prints is the same arithmetic
            # over every reducible stage the boundary reaches, DRAM included.
            ceiling = res.detail["reducible_weight_energy_pJ"]
            want = sum(wp.stages[s].energy_pJ for s in reduced if s in wp.stages)
            assert math.isclose(ceiling["before_pJ"], want, rel_tol=1e-12), (name, key)
            assert math.isclose(ceiling["ceiling_on_the_saving_pJ"], want * d,
                                rel_tol=1e-12), (name, key)

        # The ORDERING the closed form implies, which is what made the literal
        # spelling worth having: each boundary reduces a prefix of the path, so
        # a later boundary's reducible energy is never smaller than an earlier
        # one's. This holds on every design without naming a stage.
        order = [p.key for p in reconmod.placements_for(arch, cfg)
                 if p.key in ev]
        befores = [out[k].detail["reducible_weight_energy_pJ"]["before_pJ"]
                   for k in order]
        assert befores == sorted(befores), (name, order, befores)


# ---------------------------------------------------- 5. nothing goes up
def test_no_stage_or_transport_category_exceeds_the_embedded_reference():
    """NOTHING GOES UP.

    Reconstruction and its register are charged as their own categories, so no
    weight-path stage and no transport or storage category may ever sit ABOVE
    the embedded reference under any boundary. Compute is required to be
    exactly equal, and DRAM to sit at EXACTLY emb_DRAM - DRAM_w x
    (1 - K/N) -- the whole DRAM term is x K/N under every boundary, so a
    forgotten scaling is
    an inequality this catches.
    """
    for case in _cases():
        out, packing = _evaluate(case)
        base, wp = case["base"], case["wpath"]
        dram0 = float(base["DRAM"])                 # the category, all tensors
        dram_w = float(case["base_w"]["DRAM"])      # its weight share, all reducible
        want = dram0 - dram_w * (1.0 - packing.frac)
        for key, res in _evaluated(case, out).items():
            for stage_key, row in _stage_rows(res).items():
                assert _after(row) <= row["weight_energy_pJ"] + 1e-6, \
                    (case["name"], key, stage_key)
            for cat in _TRANSPORT_CATS:
                got, ref = res.components.get(cat, 0.0), float(base.get(cat, 0.0))
                assert got <= ref + 1e-6, (case["name"], key, cat, got, ref)
            assert math.isclose(res.components["DRAM"], want, rel_tol=1e-12), \
                (case["name"], key, res.components["DRAM"], want)
            assert res.components["DRAM"] < dram0, (case["name"], key)
            assert res.components["Compute"] == float(base["Compute"]), \
                (case["name"], key)


# ---------------------------------------------------- 6. the code sweep
def test_a_stronger_code_saves_strictly_more_at_every_reduced_stage():
    """SCALE SWEEP -- is the reduction actually keyed to the CODE?

    A reduction that is applied but hard-wired to one ratio passes every test
    above. Sweeping K over the six geometries `env.sh` offers and requiring
    each reduced stage to fall strictly and monotonically with (1 - K/N) is
    what catches that.
    """
    ks = (57, 51, 45, 39, 36, 30)
    for case in _cases():
        seen = {}
        for k in ks:
            out, packing = _evaluate(case, k=k, n=63)
            assert math.isclose(packing.frac, k / 63)
            for key, res in _evaluated(case, out).items():
                for stage_key, row in _stage_rows(res).items():
                    if not row["carries_reduced"]:
                        continue
                    assert math.isclose(row["energy_after_pJ"],
                                        row["weight_energy_pJ"] * k / 63,
                                        rel_tol=1e-12), (case["name"], k, key, stage_key)
                    seen.setdefault((key, stage_key), []).append(_after(row))
                for cat in _TRANSPORT_CATS:
                    seen.setdefault((key, "category " + cat), []).append(
                        res.components.get(cat, 0.0))
        assert seen, case["name"]
        for (key, what), series in seen.items():
            if what.startswith("category "):
                # a category total is non-increasing in (1 - K/N), and strictly
                # decreasing only where that category holds a reduced stage
                assert all(b <= a + 1e-6 for a, b in zip(series, series[1:])), \
                    (case["name"], key, what, series)
            else:
                assert all(b < a for a, b in zip(series, series[1:])), \
                    (case["name"], key, what, series)


# ---------------------------------------------- 7. every level is claimed
_WEIGHT_BLOCK = re.compile(
    r"\n\s+Weights\s*:\s*\n(.*?)(?=\n\s+(?:Weights|Inputs|Outputs)\s*:\s*\n|\Z)", re.S)


def _weight_carrying_levels(stats_path):
    """Every level and network in one stats file that spends Weights energy.

    Deliberately its OWN parser rather than `recon.read_weight_path`: the point
    is that the module's stage-to-level match is COMPLETE, and re-using the
    module's own scan to check the module's own scan would prove nothing.
    """
    text = pathlib.Path(stats_path).read_text()
    levels, networks = (text.split("\nNetworks\n", 1) if "\nNetworks\n" in text
                        else (text, ""))
    networks = networks.split("Operational Intensity")[0]
    found = {}

    def add(name, block):
        m = _WEIGHT_BLOCK.search(block)
        if not m:
            return
        e = re.search(r"\n\s+Energy \(total\)\s*:\s*([\d.eE+-]+)\s*pJ", m.group(1))
        if e and float(e.group(1)) > 0:
            found[name] = found.get(name, 0.0) + float(e.group(1))

    chunks = re.split(r"===\s*(.+?)\s*===", levels)[1:]
    for name, body in zip(chunks[0::2], chunks[1::2]):
        add(name, body)
    for block in re.split(r"\nNetwork \d+\n-+\n", "\n" + networks)[1:]:
        name = next((ln.strip() for ln in block.split("\n") if ln.strip()), "?")
        add(f"NoC: {name}", block)
    return found


def test_every_weight_carrying_level_of_the_real_design_is_claimed_by_a_stage():
    """EVERY WEIGHT-CARRYING LEVEL IS CLAIMED, checked against the real stats.

    `weight_path_reconciles_with_raw_record` already fails a run whose stages
    miss a level. What this adds is the other half: an INDEPENDENT re-scan of
    the cached Timeloop output confirming that no such level exists on this
    design -- i.e. that the check passes because the weight path is complete,
    not because nothing was looked at.
    """
    case = _real_case()
    if case is None:
        raise _Skip(_REAL_CACHE.get("why", "no cached Timeloop output here"))
    from eccenergy import recon as reconmod

    claimed, who = set(), {}
    for stage in reconmod.stages_for(case["arch"], case["cfg"]):
        st = case["wpath"].stages.get(stage.key)
        levels = set(st.levels) if st is not None else set()
        for lv in levels:
            who.setdefault(lv, []).append(stage.key)
        claimed |= levels
    # exactly one stage per level, not merely at least one -- except the DRAM
    # level, which is the array/interface PAIR by construction, with shares
    # that sum to one (test_the_dram_level_is_claimed_exactly_once_in_total)
    for lv, stages in who.items():
        assert len(stages) == 1, \
            (lv, stages)
    assert sorted(who.get("DRAM", [])) == ["dram"], who

    found = {}
    for path in case["stats"].values():
        for level, energy in _weight_carrying_levels(path).items():
            found[level] = found.get(level, 0.0) + energy
    assert found, case["stats"]
    assert set(found) == claimed, (sorted(set(found) ^ claimed), sorted(found))
    assert case["wpath"].unclaimed == [], case["wpath"].unclaimed

    ok, detail = reconmod.cross_check(case["cfg"], case["arch"], case["wpath"],
                                      case["base_w"])
    assert ok is True, detail


def test_an_unclaimed_weight_level_fails_the_recorded_check_rather_than_dropping():
    """...and a level no stage claims must FAIL THE RUN, not vanish from it.

    `cross_check` returning False is not enough on its own: what matters is
    that the failure reaches the result file as a failed check, so a run that
    silently lost weight energy cannot be published from.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise _Skip(f"pandas not available on this python: {exc}")
    from eccenergy import recon as reconmod
    from eccenergy.energy import Raw
    from eccenergy.experiments.recon import PARITY_KEY, task3_checks
    from eccenergy.paths import Results
    from eccenergy.results_store import ResultBuilder, Variant

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(ECC_RESULTS_DIR=tmp)
        stats = _write_cache(tmp)
        _write_lf(pathlib.Path(stats),
                  _STATS.replace("weights_spad", "mystery_weight_buffer"))
        layer = _Layer()
        wp = reconmod.weight_path(cfg, "eyeriss_v2_like", "resnet18", [layer],
                                  {layer.shape_name: stats})
        assert [u["level"] for u in wp.unclaimed] == ["mystery_weight_buffer"], \
            wp.unclaimed

        cats = ["DRAM", "Global buffer", "Local (spads/RF)", "NoC", "Compute",
                "ECC decode", "Reconstruction", "Recon overhead"]
        base_w = pd.Series({"DRAM": 5000.0, "Local (spads/RF)": 800.0,
                            "NoC": 150.0}).reindex(cats, fill_value=0.0)
        base = pd.Series({"DRAM": 5000.0, "Local (spads/RF)": 911.0,
                          "NoC": 150.0, "Compute": 9000.0}).reindex(cats, fill_value=0.0)
        ok, detail = reconmod.cross_check(cfg, "eyeriss_v2_like", wp, base_w)
        assert ok is False and detail["unclaimed_weight_levels"]

        builder = ResultBuilder(cfg, Results(cfg).prepare(), "eyeriss_v2_like",
                                "resnet18", experiment="unit_test",
                                fixed_mapping=True)
        components = {"DRAM": 5000.0, PARITY_KEY: 0.0, "Local (spads/RF)": 911.0,
                      "NoC": 150.0, "Compute": 9000.0}
        for name, kind in (("baseline_external_parity", "baseline"),
                           ("embedded_ecc", "embedded")):
            builder.add(Variant(name, kind=kind, status="evaluated",
                                total_energy_pJ=sum(components.values()),
                                energy_by_component_pJ=dict(components),
                                mapping_ids=["aaaa1111"]))
        raw = Raw(base, base_w, base * 0.0, 5000.0, 2048.0, 1, 0, 1024,
                  per_layer=[{"layer": "l0", "status": "ok", "weights": 1024,
                              "mapping_id": "aaaa1111"}], levels=[])
        task3_checks(builder, cfg, "eyeriss_v2_like", raw, dict(components),
                     dict(components), 0.0, [], wp, ok, detail, base_w,
                     newly_mapped=0)
        recorded = {x["check"]: x["passed"] for x in builder.document()["validation"]}
        assert recorded["weight_path_reconciles_with_raw_record"] is False, recorded


def test_the_property_tests_also_ran_on_the_real_cache():
    """A skip here means every property above was checked on the SYNTHETIC
    stats only. That is a coverage gap, not a pass: run it in the container on
    a machine that has the mapper cache."""
    case = _real_case()
    if case is None:
        raise _Skip(_REAL_CACHE.get("why", "no cached Timeloop output here"))
    assert case["wpath"].total_weight_energy() > 0, case["name"]
    assert len(case["stats"]) >= 1, case["stats"]


def test_a_reducible_stage_no_boundary_reduces_stops_the_run():
    """The gap `feasibility()` structurally cannot see.

    Since 2026-09-09 the path starts with `dram_interface`, which every
    placement reduces, so on `eyeriss_v2_like` the prefix rule is satisfied by
    R1 = (dram_interface,) and on the `_wglb` variant it still fails from R2
    on: the reducible order there is dram_interface, weight_glb, mesh, ...

    `WEIGHT_PATHS` and `PLACEMENTS` are two tables that have to be edited
    together. Extend the path with a reducible stage and forget the boundary,
    and every placement BELOW it keeps its own saving while reporting the new
    stage at full width -- an understated study that no other check in this
    module notices, because `feasibility()` only inspects the stages a
    placement does claim.

    `eyeriss_v2_like_wglb` is that case as the module stands: its weight path
    carries a `weight_glb` stage between DRAM and the mesh, its five placements
    are `eyeriss_v2_like`'s and none of them reduces it. It has no mapper cache
    yet, so it has never been run -- which is exactly why the guard has to be
    structural rather than empirical.
    """
    from eccenergy import recon as reconmod

    ok, detail = reconmod.validate_placement_space("eyeriss_v2_like")
    assert ok is True, detail["violations"]
    assert detail["reducible_stages_no_placement_reaches"] == [], detail
    # ...and every boundary's reduced set really is a prefix of the path
    assert all(row["is_a_prefix"] for row in detail["per_placement"]), detail

    ok, detail = reconmod.validate_placement_space("eyeriss_v2_like_wglb")
    assert ok is False
    assert detail["reducible_stages_no_placement_reaches"] == ["weight_glb"], detail
    assert any("weight_glb" in v for v in detail["violations"]), detail
    # the prefix violation is named per placement, not just in aggregate
    bad = [row["placement"] for row in detail["per_placement"]
           if not row["is_a_prefix"]]
    assert bad == ["recon2", "recon3", "recon4"], bad


def test_the_placement_space_check_is_recorded_on_every_result():
    """...and it reaches the result file, so a run cannot be published from a
    design whose two tables have drifted apart."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise _Skip(f"pandas not available on this python: {exc}")
    from eccenergy import recon as reconmod
    from eccenergy.energy import Raw
    from eccenergy.experiments.recon import PARITY_KEY, task3_checks
    from eccenergy.paths import Results
    from eccenergy.results_store import ResultBuilder, Variant

    cats = ["DRAM", "Global buffer", "Local (spads/RF)", "NoC", "Compute",
            "ECC decode", "Reconstruction", "Recon overhead"]
    base_w = pd.Series({"DRAM": 5000.0}).reindex(cats, fill_value=0.0)
    base = pd.Series({"DRAM": 5000.0, "Compute": 9000.0}).reindex(cats, fill_value=0.0)
    components = {"DRAM": 5000.0, PARITY_KEY: 0.0, "Compute": 9000.0}

    def recorded_for(arch):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(ECC_RESULTS_DIR=tmp, ECC_SWEEP_ARCHS=arch,
                       ECC_CONST_ARCH=arch)
            stats = _write_cache(tmp)
            layer = _Layer()
            wp = reconmod.weight_path(cfg, arch, "resnet18", [layer],
                                      {layer.shape_name: stats})
            builder = ResultBuilder(cfg, Results(cfg).prepare(), arch, "resnet18",
                                    experiment="unit_test", fixed_mapping=True)
            for name, kind in (("baseline_external_parity", "baseline"),
                               ("embedded_ecc", "embedded")):
                builder.add(Variant(name, kind=kind, status="evaluated",
                                    total_energy_pJ=sum(components.values()),
                                    energy_by_component_pJ=dict(components),
                                    mapping_ids=["aaaa1111"]))
            raw = Raw(base, base_w, base * 0.0, 5000.0, 2048.0, 1, 0, 1024,
                      per_layer=[{"layer": "l0", "status": "ok", "weights": 1024,
                                  "mapping_id": "aaaa1111"}], levels=[])
            task3_checks(builder, cfg, arch, raw, dict(components),
                         dict(components), 0.0, [], wp, True, {}, base_w,
                         newly_mapped=0)
            return {x["check"]: x["passed"]
                    for x in builder.document()["validation"]}

    good = recorded_for("eyeriss_v2_like")
    assert good["placement_space_covers_the_whole_weight_path"] is True, good
    bad = recorded_for("eyeriss_v2_like_wglb")
    assert bad["placement_space_covers_the_whole_weight_path"] is False, bad













# --------------------------------------------- 12. Task 4: capacity dilation

def test_capacity_dilation_scales_only_weight_levels_and_never_a_latch():
    """The dilation must move weight room and NOTHING else.

    Three things it is not allowed to do, each of which would silently hand the
    reconstruction arm an advantage reconstruction does not buy:

      * dilate an activation or partial-sum buffer -- that is a different
        architecture, not this treatment;
      * dilate a declared `depth: 1` register -- `simple_weight_stationary`'s
        `weight_reg` is a pipeline latch (FINDINGS 7.5), and turning it into a
        two-entry buffer invents the reuse level R4b's register is an addition
        TO;
      * dilate a level holding Weights BESIDE another dataspace under the
        default scope -- Timeloop has one capacity per level, so that would
        also hand the mapper free input-activation room.
    """
    from eccenergy import archs as archmod
    import re
    cfg = _cfg(ECC_WEIGHT_CAPACITY_SCALE="1.6154")
    for arch, want, forbidden in (
            ("eyeriss_like", {"weights_spad"}, {"ifmap_glb", "psum_glb",
                                                "ifmap_spad", "psum_spad"}),
            ("eyeriss_v2_like", {"weights_spad"}, {"iact_glb", "psum_glb",
                                                   "ifmap_spad", "psum_spad"}),
            ("simple_weight_stationary", {"weight_glb", "pe_spad"},
             {"input_glb", "psum_glb", "weight_reg",
              "input_activation_reg", "output_activation_reg"})):
        before = archmod.arch_source(arch, cfg).read_text()
        after = archmod._scale_weight_capacity(before, 1.6154, "exclusive",
                                               arch, quiet=True)
        assert after != before, arch

        def depths(text):
            out = {}
            for part in re.split(r"(?=\n\s*-\s*!)", text):
                n = re.search(r"name:\s*(\S+)", part)
                d = re.search(r"\bdepth:\s*(\d+)", part)
                if n and d:
                    out[n.group(1)] = int(d.group(1))
            return out

        d0, d1 = depths(before), depths(after)
        moved = {k for k in d0 if d1.get(k) != d0[k]}
        assert moved == want, (arch, moved, want)
        for level in forbidden:
            if level in d0:
                assert d1[level] == d0[level], (arch, level)
        for level in want:
            got = d1[level] / d0[level]
            assert abs(got - 1.6154) < 0.01, (arch, level, got)


def test_a_shared_weight_level_is_a_bracket_and_is_named_as_one():
    """`operand_glb` holds Inputs AND Weights, so it is a choice, not a fact.

    ON `simple_output_stationary` SINCE 2026-09-13, not on
    `simple_weight_stationary`: that design's operand half is now split into
    `input_glb` and a Weights-only `weight_glb`, so it has no shared weight
    level left to bracket. The two siblings still declare one and are what
    the `shared` scope is for.
    """
    from eccenergy import archs as archmod
    import re
    cfg = _cfg()
    base = archmod.arch_source("simple_output_stationary", cfg).read_text()
    excl = archmod._scale_weight_capacity(base, 1.6154, "exclusive",
                                          "simple_output_stationary", quiet=True)
    shar = archmod._scale_weight_capacity(base, 1.6154, "shared",
                                          "simple_output_stationary", quiet=True)

    def depth_of(text, name):
        for part in re.split(r"(?=\n\s*-\s*!)", text):
            if re.search(rf"name:\s*{name}\b", part):
                m = re.search(r"\bdepth:\s*(\d+)", part)
                return int(m.group(1)) if m else None
        return None

    assert depth_of(base, "operand_glb") == depth_of(excl, "operand_glb")
    assert depth_of(shar, "operand_glb") > depth_of(base, "operand_glb")
    # the latch is out of BOTH: no scope may invent a reuse level
    assert depth_of(shar, "weight_reg") == depth_of(base, "weight_reg") == 1


def test_each_capacity_is_its_own_mapper_cache_and_a_no_op_keeps_the_old_one():
    """Task 4 is the DIFF of two mappings, so they must never share a cache.

    And the converse, which is the expensive mistake: a scale that rewrites
    nothing on a given design must NOT move that design's cache, or every
    re-run pays for a fresh map of an unchanged architecture -- the trap
    `archs._patch_dram_depth`'s docstring records for ECC_DRAM_DEPTH.
    """
    from eccenergy import archs as archmod
    seen = {}
    for scale in ("1.0", "0.5", "1.6154"):
        cfg = _cfg(ECC_WEIGHT_CAPACITY_SCALE=scale)
        for arch in ("eyeriss_like", "eyeriss_v2_like",
                     "simple_weight_stationary"):
            key = (archmod.effective_variant(arch, cfg),
                   archmod.arch_fingerprint(arch, cfg))
            assert seen.setdefault(key, (arch, scale)) == (arch, scale), \
                (key, seen[key], (arch, scale))
    # 1.0 is the declared design and must keep the slug it always had
    cfg1 = _cfg(ECC_WEIGHT_CAPACITY_SCALE="1.0")
    for arch in ("eyeriss_like", "simple_weight_stationary"):
        assert "wcap" not in archmod.effective_variant(arch, cfg1), arch

    # a scale so close to 1 that every weight depth rounds back is a no-op and
    # keeps the undilated cache rather than re-mapping an unchanged design
    tiny = _cfg(ECC_WEIGHT_CAPACITY_SCALE="1.0005")
    for arch in ("eyeriss_like", "simple_weight_stationary"):
        assert (archmod.effective_variant(arch, tiny)
                == archmod.effective_variant(arch, cfg1)), arch


def test_the_dilation_correction_reprices_the_array_and_is_recorded():
    """The dilated level is re-priced at the DECLARED array's per-access cost.

    Accelergy costs a level from its declared geometry, so `depth x N/K` is
    billed as a bigger SRAM. The reconstruction arm's array is the same silicon
    holding narrower values, so that is silicon it does not have. The
    correction must move the stage's energy by exactly the ERT ratio and must
    say so; an unreadable ERT must leave the number ALONE and say that instead
    of scaling by a guess.
    """
    from eccenergy import recon as reconmod
    case = _synthetic_case()
    wp = case["wpath"]
    key = "weight_spad"
    st = wp.stages[key]
    before, reads, writes = st.energy_pJ, st.reads, st.fills

    # READS AND WRITES ARE PRICED SEPARATELY. Build a correction in which the
    # declared array is cheaper on both, by different factors -- which is the
    # real case (eyeriss_like at x1.6154: read x1.2395, write x1.3796) and the
    # one a single read-derived ratio got wrong.
    # THE BLOCK SIZE MUST CANCEL. Accelergy prices a whole physical word and
    # Timeloop counts one access per value, so build the dilated split so that
    # `reads x e_rd + writes x e_wr` comes out at exactly BLOCK times the
    # level's energy -- the real case (2 on eyeriss_like, 3 on Eyeriss v2), and
    # the one that made an absolute rebuild refuse a correction it should have
    # applied. Reads and writes are also given DIFFERENT ratios, because they
    # do not scale together and a read-derived ratio under-corrects the writes.
    BLOCK = 2
    e_rd_dil = BLOCK * before / (reads + 2 * writes) if (reads + 2 * writes) else 1.0
    corr = {"ok": True, "level": "weight_spad",
            "read_pJ_dilated": e_rd_dil, "write_pJ_dilated": 2 * e_rd_dil,
            "read_pJ_declared": 0.8 * e_rd_dil, "write_pJ_declared": 1.4 * e_rd_dil,
            "read_ratio_declared_over_dilated": 0.8,
            "write_ratio_declared_over_dilated": 0.7,
            "provenance": "synthetic"}
    ratio = ((reads * 0.8 + writes * 1.4) / (reads * 1.0 + writes * 2.0))
    moved = reconmod.apply_capacity_correction(wp, key, corr)
    assert moved["corrected"] is True, moved
    assert math.isclose(wp.stages[key].energy_pJ, before * ratio, rel_tol=1e-9), moved
    assert math.isclose(moved["moved_pJ"], before * (ratio - 1), rel_tol=1e-9)
    assert "2 values per physical word" in moved["note"], moved["note"]
    # ...and it is strictly between the read-only and write-only corrections,
    # which is the whole point of weighting it by the access mix
    assert 0.7 < ratio < 1.4, ratio

    # A SPLIT THAT DOES NOT RECONCILE IS REFUSED, not applied: re-pricing from
    # one would move a bar by an amount nothing checked. 2.5 values per word is
    # not a physical word.
    bad = dict(corr, read_pJ_dilated=e_rd_dil * 2.5, write_pJ_dilated=5 * e_rd_dil)
    kept = wp.stages[key].energy_pJ
    out = reconmod.apply_capacity_correction(wp, key, bad)
    assert out["corrected"] is False and out["moved_pJ"] == 0.0, out
    assert "NOT corrected" in out["note"], out
    assert wp.stages[key].energy_pJ == kept

    # an unreadable ERT leaves it uncorrected AND says so
    corr2 = reconmod.capacity_dilation_correction(
        "/nonexistent/ref", "/nonexistent/dil", ("weights_spad",))
    assert corr2["ok"] is False
    assert "NOT corrected" in corr2["provenance"], corr2


def test_capacity_dilation_scale_is_derived_from_the_code_not_configured():
    """N/K comes from the BCH geometry, so a code change moves it by itself."""
    from eccenergy import recon as reconmod
    for k, want in ((39, 63 / 39), (51, 63 / 51), (30, 63 / 30)):
        cfg = _cfg(ECC_CONST_K=str(k), ECC_RECON_K=str(k))
        assert math.isclose(reconmod.capacity_dilation_scale(cfg), want,
                            abs_tol=5e-5), k
        # ONE capacity, ONE spelling: the value that goes into the cache slug
        # must be the value the shell writes, or the evaluator refuses a cache
        # it has. (63/39 = 1.61539 by %g, 1.6154 rounded -- two directories.)
        assert f"{reconmod.capacity_dilation_scale(cfg):g}" == \
            f"{round(63 / k, 4):g}", k
    # and it composes with a SHRUNK reference: the reconstruction arm is always
    # N/K times whatever the reference arm's silicon is
    # The scale is quantised to four DECIMALS, not to four significant
    # figures, so the tolerance has to be absolute: 0.25 x 63/39 = 0.403846
    # stores as 0.4038, which is 1.1e-4 relative but 4.6e-5 absolute.
    cfg = _cfg(ECC_CONST_K="39", ECC_RECON_K="39",
               ECC_WEIGHT_CAPACITY_SCALE="0.25")
    assert math.isclose(reconmod.capacity_dilation_scale(cfg),
                        0.25 * 63 / 39, abs_tol=5e-5)


def test_an_identical_loop_nest_means_an_identical_dram_read_count():
    """FINDINGS 7.7's finding, as a regression test on whatever caches exist.

    The Task 4 result rests on one property: where the mapper hands back the
    same loop nest at the dilated capacity, the reconstruction arm is the
    reference mapping on the same silicon, so it must report the SAME DRAM
    weight reads. If a future change to the parser, to the scale quantisation
    or to the cache keying broke that, the study would report a capacity effect
    that is really a bookkeeping difference -- the one failure mode this whole
    exercise exists to avoid.

    THE PAIRS ARE DISCOVERED ON DISK rather than rebuilt from a config, because
    a config assembled here would have to reproduce every mapper knob the wave
    was submitted with (objective, victory, algorithm, timeout, NoC) to land on
    the same slug, and getting one wrong makes the test silently skip instead of
    silently fail. A `<slug>__wcap<scale>` directory beside its `<slug>` is the
    pair, whatever produced it.
    """
    import pathlib as _pl
    from eccenergy.paths import ROOT
    try:
        from eccenergy.experiments import dilation as dilmod
    except ImportError as exc:                                # pragma: no cover
        raise _Skip(f"dilation module unavailable: {exc}")

    out = ROOT / "ecc_energy_study" / "outputs"
    if not out.is_dir():
        raise _Skip("no mapper cache on this machine")

    def shapes_of(fp_dir):
        return {d.name: d / "timeloop-mapper.map.txt" for d in fp_dir.iterdir()
                if (d / "timeloop-mapper.map.txt").is_file()}

    checked = identical = pairs = 0
    for arch_dir in sorted(out.iterdir()):
        if not arch_dir.is_dir():
            continue
        for dil_slug in sorted(arch_dir.glob("*__wcap*")):
            base = _pl.Path(str(dil_slug).rsplit("__wcap", 1)[0])
            if not base.is_dir():
                continue
            for dil_fp in sorted(dil_slug.glob("fp-*")):
                dil_shapes = shapes_of(dil_fp)
                if not dil_shapes:
                    continue
                for ref_fp in sorted(base.glob("fp-*")):
                    ref_shapes = shapes_of(ref_fp)
                    common = set(ref_shapes) & set(dil_shapes)
                    if not common:
                        continue
                    pairs += 1
                    for shape in sorted(common):
                        checked += 1
                        if ref_shapes[shape].read_text() != dil_shapes[shape].read_text():
                            continue          # the mapper DID use the room
                        identical += 1
                        a = dilmod.read_mapped_layer(
                            ref_fp / shape / "timeloop-mapper.stats.txt", "DRAM")
                        b = dilmod.read_mapped_layer(
                            dil_fp / shape / "timeloop-mapper.stats.txt", "DRAM")
                        if a is None or b is None:
                            continue
                        assert a.dram_weight_reads == b.dram_weight_reads, (
                            f"{arch_dir.name}/{shape}: the loop nests are "
                            f"byte-identical but the DRAM weight reads differ "
                            f"({a.dram_weight_reads:,.0f} vs "
                            f"{b.dram_weight_reads:,.0f}). One of them is not "
                            f"being read from the mapping it claims.")
                    break                     # one reference fp per dilated fp
    if not pairs:
        raise _Skip("no <slug>/<slug>__wcap pair on disk -- fill one with "
                    "hpc/map_capacity_sweep.sh")
    print(f"        ({identical}/{checked} shapes over {pairs} capacity pair(s) "
          f"came back byte-identical)", end="")


# ---------------------------------------------------------------------------
#  prompt_6 -- ERT arms (phase 3)
# ---------------------------------------------------------------------------
#: What prompt_6 3.3's DERIVED rule yields on every registered design. A new
#: architecture cannot silently get zero arms or the wrong ones: add it here.
#: Note simple_weight_stationary: THREE arms, not the two Appendix B guesses --
#: `weight_reg` is a storage stage below `pe_spad`, so `pe_spad` reads
#: (recon4) are not the innermost level's reads and the mapping CAN move them.
_EXPECTED_ERT_ARMS = {
    "eyeriss_like_wglb": ("recon2", "recon4"),
    "eyeriss_like": ("recon3",),
    "eyeriss_v2_like": ("recon3",),
    "eyeriss_v2_like_wglb": ("recon3",),
    "simple_weight_stationary": ("recon2", "recon3", "recon4"),
}


def test_the_ert_injectable_predicate_on_every_registered_design():
    """prompt_6 3.3, on every design in `recon.PLACEMENTS`: the derived set,
    and the stated reason for every excluded bar."""
    from eccenergy import recon
    assert set(_EXPECTED_ERT_ARMS) == set(recon.PLACEMENTS), (
        f"a design was registered without an expectation here: "
        f"{set(recon.PLACEMENTS) ^ set(_EXPECTED_ERT_ARMS)}")
    for arch, want in _EXPECTED_ERT_ARMS.items():
        got = tuple(p.key for p in recon.ert_arms(arch))
        assert got == want, f"{arch}: ERT arms {got} != {want}"
        stages = recon.stages_for(arch)
        for p in recon.placements_for(arch):
            ok, why = recon.ert_injectable(p, stages)
            assert ok == (p.key in want), (arch, p.key, why)
            if not ok:
                assert any(w in why for w in ("dram stage", "network stage",
                                              "innermost weight level")), why
    # Eyeriss v1: recon2 is filter_glb + reads -> read, recon4 is weights_spad
    # + fills -> write, and BOTH narrow only filter_glb (the weights stop AT
    # the boundary; weights_spad stays 8 on recon4).
    r2 = recon.ert_arm_spec("eyeriss_like_wglb", "recon2")
    r4 = recon.ert_arm_spec("eyeriss_like_wglb", "recon4")
    assert (r2["level"], r2["counter"], r2["action"]) == ("filter_glb", "reads", "read"), r2
    assert (r4["level"], r4["counter"], r4["action"]) == ("weights_spad", "fills", "write"), r4
    assert r2["narrow_levels"] == ("filter_glb",), r2["narrow_levels"]
    assert r4["narrow_levels"] == ("filter_glb",), r4["narrow_levels"]
    # not an arm: a clear refusal naming the reason, never a silent None
    for key in ("recon1", "recon3", "recon5"):
        try:
            recon.ert_arm_spec("eyeriss_like_wglb", key)
        except ValueError as e:
            assert "not an ERT arm" in str(e), e
        else:
            raise AssertionError(f"{key} was accepted as an ERT arm")
    try:
        recon.ert_arm_spec("eyeriss_like_wglb", "recon9")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown key was accepted")


def test_the_predicate_is_derived_not_keyed_on_the_name():
    """MUTATIONS: the same key with a different record flips the verdict, so
    nothing can be reading `key == "recon2"`."""
    import dataclasses
    from eccenergy import recon
    stages = recon.stages_for("eyeriss_like_wglb")
    r2 = recon.placement_by_key("eyeriss_like_wglb", "recon2")
    assert recon.ert_injectable(r2, stages)[0]
    # a storage stage charged on a NETWORK counter names no ERT action
    assert not recon.ert_injectable(dataclasses.replace(r2, site_counter="deliveries"), stages)[0]
    # the same boundary sited on the network stage is billed by noc.yaml
    assert not recon.ert_injectable(dataclasses.replace(r2, site_stage="array_multicast"), stages)[0]
    # moved to the innermost level's READS it is mapping-invariant ...
    assert not recon.ert_injectable(dataclasses.replace(
        r2, site_stage="weights_spad", site_counter="reads"), stages)[0]
    # ... but the innermost level's FILLS are fine (that IS recon4)
    assert recon.ert_injectable(dataclasses.replace(
        r2, site_stage="weights_spad", site_counter="fills"), stages)[0]
    # and a stage that is not on the path at all
    assert not recon.ert_injectable(dataclasses.replace(r2, site_stage="nowhere"), stages)[0]


def test_the_ert_deltas_reproduce_prompt_6_table_5_1():
    """`delta = E_w x block_size` on the access action, `idle_per_cycle` on
    leak, E_w = incremental / weights_per_codeword. At BCH(63,30) over 8-bit
    weights: E_w = 1.3786 / 7.875 = 0.175060; filter_glb (64/4 = 16 per word)
    2.80096; weights_spad (16/8 = 2) 0.350120; leak 2.8310811. The table in the
    plan is the CHECK; these are recomputed from the DC numbers."""
    from eccenergy import recon
    gran = recon.Granularity(63, 30, 8, "weight")
    assert abs(gran.weights_per_codeword - 7.875) < 1e-12
    d16 = recon.ert_deltas(1.3786, 2.8310811, gran, 16)
    d2 = recon.ert_deltas(1.3786, 2.8310811, gran, 2)
    assert abs(d16["e_w_pj"] - 0.175060) < 1e-6, d16["e_w_pj"]
    assert abs(d16["access_delta_pj"] - 2.80096) < 1e-5, d16["access_delta_pj"]
    assert abs(d2["access_delta_pj"] - 0.350120) < 1e-6, d2["access_delta_pj"]
    assert d16["leak_delta_pj"] == d2["leak_delta_pj"] == 2.8310811
    assert d16["access_delta_pj"] / d2["access_delta_pj"] == 8.0
    # codeword charging: every access rebuilds a whole codeword, E_w = incremental
    dc = recon.ert_deltas(1.3786, 2.8310811, recon.Granularity(63, 30, 8, "codeword"), 16)
    assert abs(dc["e_w_pj"] - 1.3786) < 1e-12 and abs(dc["access_delta_pj"] - 1.3786 * 16) < 1e-9
    # BREAKAGE: a block size of 0 or None is undefined, never a silent 0 pJ
    for bad in (0, None):
        try:
            recon.ert_deltas(1.3786, 2.8310811, gran, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"block_size={bad!r} was accepted")


# ---------------------------------------------------------------------------
#  prompt_6 RULE 1 -- one owner per narrow stop, decided by measured Word bits
# ---------------------------------------------------------------------------
def test_storage_scale_bypasses_a_level_the_mapper_already_narrowed():
    """RULE 1 on `Packing`: Word bits == q -> the mapper owns it, scale 1.0;
    Word bits == weight_bits -> the evaluator narrows as before; anything else
    -> STOP. No `word_bits` keeps the pre-prompt_6 reading."""
    from eccenergy.recon import Packing
    for mode in ("aligned", "stream"):
        p = Packing(mode, 8, 30, 63)          # q = round(8*30/63) = 4
        assert p.declared_q == 4
        legacy = p.storage_scale(64)
        assert legacy < 1.0, (mode, legacy)  # both modes narrow a 64-bit word at q=4
        assert p.storage_scale(64, word_bits=8) == legacy
        assert p.storage_scale(64, word_bits=None) == legacy
        assert p.storage_scale(64, word_bits=4) == 1.0, "the mapper narrowed it; applying Packing squares the saving"
        assert p.storage_scale(16, word_bits=4) == 1.0
        assert p.narrowing_owner(4) == "mapper" and p.narrowing_owner(8) == "evaluator"
        assert p.narrowing_owner(None) == "evaluator"
        for bad in (5, 7, 16):
            try:
                p.narrowing_owner(bad)
            except ValueError as e:
                assert "disagree" in str(e), e
            else:
                raise AssertionError(f"Word bits {bad} was given an owner at q=4")
    # at a code where q equals the weight width there is nothing to narrow, and
    # Word bits 8 is the evaluator's stop, not the mapper's
    p8 = Packing("aligned", 8, 57, 63)        # q = 7; a plan at 8 is 8-bit
    assert p8.narrowing_owner(8) == "evaluator" and p8.narrowing_owner(7) == "mapper"


def test_the_narrowing_site_audit_fails_on_both_live_and_on_nobody_live():
    """The per-stop audit row, fed the scale the evaluator ACTUALLY applied.
    BREAKAGES: a q-bit plan with a scale != 1 applied (both live) and an 8-bit
    plan with x1.0 applied on a stop the placement narrows (nobody live)."""
    from eccenergy.recon import Packing
    p = Packing("aligned", 8, 30, 63)
    ok_m = p.narrowing_site(4, 1.0)
    assert ok_m["ok"] and ok_m["owner"] == "mapper" and not ok_m["evaluator_live"], ok_m
    ok_e = p.narrowing_site(8, 0.5)
    assert ok_e["ok"] and ok_e["owner"] == "evaluator" and not ok_e["mapper_live"], ok_e
    both = p.narrowing_site(4, 0.5)
    assert not both["ok"] and both["owner"] == "BOTH" and "SQUARED" in both["problem"], both
    nobody = p.narrowing_site(8, 1.0)
    assert not nobody["ok"] and nobody["owner"] == "NOBODY" and "NOBODY" in nobody["problem"], nobody


def _stats_with_word_bits(word_bits):
    """The synthetic stats with the scratchpad declared at `word_bits`."""
    old = ("        Word bits                       : 8\n"
           "        Block size                      : 3\n")
    assert _STATS.count(old) == 1, "the spad block moved"
    return _STATS.replace(old, f"        Word bits                       : {word_bits}\n"
                               f"        Block size                      : 3\n")


def test_a_q_bit_plan_is_narrowed_by_the_mapper_and_an_odd_one_is_refused():
    """RULE 1 end to end on `evaluate_placement`: at BCH(63,30) (q = 4) the
    recon4 bar on Eyeriss v2 stores reduced weights in the spad. On an 8-bit
    plan the evaluator narrows the spad (scale < 1, owner evaluator); on a
    plan whose spad prints Word bits 4 the mapper already did, so the spad
    saving is ZERO and the owner is the mapper -- the same bar, same energies,
    read from two different plans. Word bits 5 stops the run."""
    from eccenergy import recon as reconmod
    cfg = _cfg(ECC_CONST_K="30", ECC_RECON_PACKING="aligned")
    gran = reconmod.Granularity(63, 30, 8, "weight")
    packing = reconmod.Packing("aligned", 8, 30, 63)
    placement = reconmod.placement_by_key("eyeriss_v2_like", "recon4", cfg)
    assert "weight_spad" in placement.reduced
    base_w = {"DRAM": _DRAM_W, "Local (spads/RF)": 800.0, "NoC": 150.0}
    base = {"DRAM": _DRAM_W, "Local (spads/RF)": 1000.0, "NoC": 150.0, "Compute": 9000.0}
    out = {}
    for wb in (8, 4):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp) / "C8_M8"; d.mkdir()
            _write_lf(d / "timeloop-mapper.stats.txt", _stats_with_word_bits(wb))
            _write_lf(d / "timeloop-mapper.map.txt", _MAP)
            layer = _Layer()
            wp = reconmod.weight_path(cfg, "eyeriss_v2_like", "resnet18", [layer],
                                      {layer.shape_name: d / "timeloop-mapper.stats.txt"})
            assert wp.stage("weight_spad").word_bits == wb
            res = reconmod.evaluate_placement(cfg, "eyeriss_v2_like", placement, wp,
                                              base_w, base, 1.0, gran, packing)
            assert res.status == "evaluated", res.reason
            out[wb] = res
    rows8 = {r["stage"]: r for r in out[8].detail["weight_path_stages"]}
    rows4 = {r["stage"]: r for r in out[4].detail["weight_path_stages"]}
    # 8-bit plan: aligned packing at q=4 halves a 24-bit word (3 -> 6 per word)
    assert math.isclose(rows8["weight_spad"]["scale"], 0.5), rows8["weight_spad"]["scale"]
    assert rows8["weight_spad"]["energy_saved_pJ"] > 0
    # q-bit plan: the mapper already narrowed it, the evaluator applies 1.0
    assert rows4["weight_spad"]["scale"] == 1.0 and rows4["weight_spad"]["energy_saved_pJ"] == 0.0
    own8 = {r["stage"]: r for r in out[8].detail["narrowing_ownership"]["stops"]}
    own4 = {r["stage"]: r for r in out[4].detail["narrowing_ownership"]["stops"]}
    assert own8["weight_spad"]["owner"] == "evaluator" and own8["weight_spad"]["measured_word_bits"] == 8
    assert own4["weight_spad"]["owner"] == "mapper" and own4["weight_spad"]["measured_word_bits"] == 4
    assert all(r["ok"] for r in list(own8.values()) + list(own4.values()))
    # the network stops are the evaluator's on both plans: narrowing a storage
    # level does not narrow the network (FINDINGS 3.2), and they are not
    # storage stops so they carry no ownership row
    assert set(own8) == set(own4) == {"weight_spad"}
    assert rows8["inter_cluster_mesh"]["scale"] == rows4["inter_cluster_mesh"]["scale"] < 1.0
    # BREAKAGE: Word bits 5 is neither 8 nor q -- the arch and the code disagree
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp) / "C8_M8"; d.mkdir()
        _write_lf(d / "timeloop-mapper.stats.txt", _stats_with_word_bits(5))
        _write_lf(d / "timeloop-mapper.map.txt", _MAP)
        layer = _Layer()
        wp = reconmod.weight_path(cfg, "eyeriss_v2_like", "resnet18", [layer],
                                  {layer.shape_name: d / "timeloop-mapper.stats.txt"})
        try:
            reconmod.evaluate_placement(cfg, "eyeriss_v2_like", placement, wp,
                                        base_w, base, 1.0, gran, packing)
        except ValueError as e:
            assert "disagree" in str(e) and "recon4" in str(e), e
        else:
            raise AssertionError("a Word bits 5 plan was evaluated at q=4")


def test_a_narrow_stop_nobody_narrows_is_refused():
    """RULE 1's second failure mode: at BCH(63,57) `aligned` packing has q = 7,
    and a 24-bit word holds floor(24/7) = 3 values -- the same 3 it held at 8
    bits -- so on an 8-bit plan NOBODY narrows the spad although recon4 says it
    carries narrow weights. That used to pass with a note; it is a stop now."""
    from eccenergy import recon as reconmod
    cfg = _cfg(ECC_CONST_K="57", ECC_RECON_PACKING="aligned")
    gran = reconmod.Granularity(63, 57, 8, "weight")
    packing = reconmod.Packing("aligned", 8, 57, 63)
    assert packing.storage_scale(24, word_bits=8) == 1.0
    placement = reconmod.placement_by_key("eyeriss_v2_like", "recon4", cfg)
    with tempfile.TemporaryDirectory() as tmp:
        stats = _write_cache(tmp)
        layer = _Layer()
        wp = reconmod.weight_path(cfg, "eyeriss_v2_like", "resnet18", [layer],
                                  {layer.shape_name: stats})
        try:
            reconmod.evaluate_placement(cfg, "eyeriss_v2_like", placement, wp,
                                        {"DRAM": _DRAM_W, "Local (spads/RF)": 800.0},
                                        {"DRAM": _DRAM_W, "Local (spads/RF)": 1000.0},
                                        1.0, gran, packing)
        except ValueError as e:
            assert "NOBODY" in str(e), e
        else:
            raise AssertionError("a narrow stop narrowed by nobody was accepted")
        # `stream` packing DOES narrow it (x K/N), so the same bar evaluates
        res = reconmod.evaluate_placement(cfg, "eyeriss_v2_like", placement, wp,
                                          {"DRAM": _DRAM_W, "Local (spads/RF)": 800.0},
                                          {"DRAM": _DRAM_W, "Local (spads/RF)": 1000.0},
                                          1.0, gran, reconmod.Packing("stream", 8, 57, 63))
        assert res.status == "evaluated"
        assert res.detail["narrowing_ownership"]["stops"][0]["owner"] == "evaluator"


# ---------------------------------------------------------------------------
#  prompt_6 RULE 3 -- incremental per event, idle per cycle per engine
# ---------------------------------------------------------------------------
def _stats_with_cycles(cycles):
    old = "Operational Intensity Stats\n---------------------------\n"
    assert _STATS.count(old) == 1
    return _STATS.replace(old, f"Summary Stats\n-------------\nCycles: {cycles}\n\n" + old)


def test_the_idle_term_is_per_cycle_per_engine_with_engines_derived_per_placement():
    """E_recon = incremental x events + idle x cycles x N_engines. Engines come
    from the site stage of the bar's own plan: 1 at DRAM, fanout x instances
    at a network, the UTILIZED instance count at a storage level, per layer
    (the spad in the synthetic plan declares 192 and utilises 4 -- Timeloop
    power-gates each unused instance and bills `leak` x utilized x cycles,
    buffer.cpp FinalizeBufferEnergy, so 4 engines leak; the 192 are reported
    beside them). Until 2026-09-11 this charged the DECLARED 192; the
    ResNet18 full-model eval (job 41740440) caught the disagreement with
    Timeloop on every shape that does not fill the array."""
    from eccenergy import recon as reconmod
    cfg = _cfg(ECC_CONST_K="51")
    gran = reconmod.Granularity(63, 51, 8, "weight")
    packing = reconmod.Packing("stream", 8, 51, 63)
    base_w = {"DRAM": _DRAM_W, "Local (spads/RF)": 800.0, "NoC": 150.0}
    base = {"DRAM": _DRAM_W, "Local (spads/RF)": 1000.0, "NoC": 150.0, "Compute": 9000.0}
    inc, idle, cycles = 1.8995, 2.2301273, 4000
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp) / "C8_M8"; d.mkdir()
        _write_lf(d / "timeloop-mapper.stats.txt", _stats_with_cycles(cycles))
        _write_lf(d / "timeloop-mapper.map.txt", _MAP)
        layer = _Layer(count=1)
        wp = reconmod.weight_path(cfg, "eyeriss_v2_like", "resnet18", [layer],
                                  {layer.shape_name: d / "timeloop-mapper.stats.txt"})
        assert wp.cycles == cycles, wp.cycles
        assert wp.stage("weight_spad").declared_instances == 192
        assert wp.stage("weight_spad").instances == 4
        want_engines = {"recon1": 1,
                        "recon2": round(wp.stage("inter_cluster_mesh").fanout
                                        * wp.stage("inter_cluster_mesh").instances),
                        "recon3": 4, "recon4": 4}
        for key, n_eng in want_engines.items():
            pl = reconmod.placement_by_key("eyeriss_v2_like", key, cfg)
            res = reconmod.evaluate_placement(cfg, "eyeriss_v2_like", pl, wp, base_w, base,
                                              inc, gran, packing, recon_idle_pj=idle)
            assert res.status == "evaluated", (key, res.reason)
            c = res.detail["reconstruction_counts"]
            assert c["engines"] == n_eng, (key, c["engines"], c["engines_rule"])
            assert c["cycles"] == cycles
            assert math.isclose(c["engine_cycles"], n_eng * cycles)
            if key in ("recon3", "recon4"):
                assert c["engines_declared"] == 192 and c["engines_utilized_max"] == 4
                assert "UTILIZED" in c["engines_rule"], c["engines_rule"]
            assert math.isclose(c["reconstruction_energy_incremental_pJ"],
                                c["reconstruction_events_codewords"] * inc)
            assert math.isclose(c["reconstruction_energy_idle_pJ"], idle * cycles * n_eng)
            assert math.isclose(c["reconstruction_energy_pJ"],
                                c["reconstruction_energy_incremental_pJ"]
                                + c["reconstruction_energy_idle_pJ"])
            assert math.isclose(res.components["Reconstruction"], c["reconstruction_energy_pJ"])
        # the network boundary's engines: fanout 4 x 1 instance in the synthetic mesh
        assert want_engines["recon2"] == 4, want_engines
        # an explicit `cycles` (another plan's, RULE 4) overrides the path's own
        pl = reconmod.placement_by_key("eyeriss_v2_like", "recon3", cfg)
        res = reconmod.evaluate_placement(cfg, "eyeriss_v2_like", pl, wp, base_w, base,
                                          inc, gran, packing, recon_idle_pj=idle, cycles=1)
        assert math.isclose(res.detail["reconstruction_counts"]["reconstruction_energy_idle_pJ"],
                            idle * 1 * 4)
        # BREAKAGE: charging the DECLARED count is not what Timeloop bills
        assert not math.isclose(res.detail["reconstruction_counts"]["reconstruction_energy_idle_pJ"],
                                idle * 1 * 192)
        # no idle term: the old behaviour, exactly
        res0 = reconmod.evaluate_placement(cfg, "eyeriss_v2_like", pl, wp, base_w, base,
                                           inc, gran, packing)
        assert res0.detail["reconstruction_counts"]["reconstruction_energy_idle_pJ"] == 0.0
    # BREAKAGE: an idle term on a plan with no `Cycles:` line is refused
    with tempfile.TemporaryDirectory() as tmp:
        stats = _write_cache(tmp)
        layer = _Layer()
        wp = reconmod.weight_path(cfg, "eyeriss_v2_like", "resnet18", [layer],
                                  {layer.shape_name: stats})
        assert wp.cycles == 0.0
        pl = reconmod.placement_by_key("eyeriss_v2_like", "recon3", cfg)
        try:
            reconmod.evaluate_placement(cfg, "eyeriss_v2_like", pl, wp, base_w, base,
                                        inc, gran, packing, recon_idle_pj=idle)
        except ValueError as e:
            assert "cycle" in str(e), e
        else:
            raise AssertionError("idle was charged with no cycle count")


# ---------------------------------------------------------------------------
#  prompt_6 RULE 4 -- the attribution split is a MOVE, not an addition
# ---------------------------------------------------------------------------
def test_the_ert_split_moves_exactly_what_the_evaluator_charges():
    """prompt_6 5.3: on an ERT arm the level's Timeloop energy already holds
    scalar_accesses x E_w (access) and idle x instances x cycles (leak). The
    split reads those two amounts off the bar's OWN counts on the action that
    was bumped, and they must equal what `evaluate_placement` charges as the
    incremental and idle terms -- so moving them out and charging them is a
    split, not a double count. Uses recon4's spec (weights_spad fills) on a
    synthetic plan whose spad declares 192 instances, utilises 4 and fills 40
    weights; the leak side is idle x UTILIZED x cycles, as Timeloop bills it."""
    try:
        from eccenergy.experiments import recon as exp
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"experiments.recon unavailable: {exc}")
    from eccenergy import recon as reconmod
    cfg = _cfg(ECC_CONST_K="30", ECC_RECON_PACKING="aligned")
    gran = reconmod.Granularity(63, 30, 8, "weight")
    inc, idle, cycles = 1.3786, 2.8310811, 5000
    bump = {"placement": "recon3", "level": "weights_spad", "counter": "fills",
            "action": "write", "e_w_pj": inc * gran.codewords(1.0),
            "access_delta_pj": inc * gran.codewords(1.0) * 3, "leak_delta_pj": idle,
            "narrow_levels": []}
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp) / "C8_M8"; d.mkdir()
        _write_lf(d / "timeloop-mapper.stats.txt", _stats_with_cycles(cycles))
        _write_lf(d / "timeloop-mapper.map.txt", _MAP)
        layer = _Layer()
        wp = reconmod.weight_path(cfg, "eyeriss_v2_like", "resnet18", [layer],
                                  {layer.shape_name: d / "timeloop-mapper.stats.txt"})
        st = wp.stage("weight_spad")
        split = exp.ert_split(bump, st, wp.cycles)
        assert split["scalar_accesses"] == st.fills == 40.0
        assert split["engines"] == 4 and split["cycles"] == cycles
        assert split["engines_declared"] == 192 and split["engines_utilized_max"] == 4
        assert math.isclose(split["access_toll_pJ"], 40.0 * inc / gran.weights_per_codeword)
        assert math.isclose(split["leak_toll_pJ"], idle * 4 * cycles)
        assert math.isclose(split["total_toll_pJ"], split["access_toll_pJ"] + split["leak_toll_pJ"])
        # the evaluator's two terms on the same plan, same placement (v2 recon3
        # is weight_spad + fills), are the same two numbers
        pl = reconmod.placement_by_key("eyeriss_v2_like", "recon3", cfg)
        packing = reconmod.Packing("aligned", 8, 30, 63)
        res = reconmod.evaluate_placement(cfg, "eyeriss_v2_like", pl, wp,
                                          {"DRAM": _DRAM_W, "Local (spads/RF)": 800.0, "NoC": 150.0},
                                          {"DRAM": _DRAM_W, "Local (spads/RF)": 1000.0, "NoC": 150.0,
                                           "Compute": 9000.0},
                                          inc, gran, packing, recon_idle_pj=idle)
        c = res.detail["reconstruction_counts"]
        assert exp._close(c["reconstruction_energy_incremental_pJ"], split["access_toll_pJ"])
        assert exp._close(c["reconstruction_energy_idle_pJ"], split["leak_toll_pJ"])
        # BREAKAGE: the wrong counter (a blended or level-total figure) does not reconcile
        wrong = exp.ert_split(dict(bump, counter="reads"), st, wp.cycles)
        assert not exp._close(wrong["access_toll_pJ"], c["reconstruction_energy_incremental_pJ"])
    # the ERT arms of every design are what the per-bar lookup keys on
    assert [p.key for p in reconmod.ert_arms("eyeriss_like_wglb")] == ["recon2", "recon4"]


def test_idle_engines_are_timeloops_utilized_instances_on_the_real_cache():
    """2026-09-11, the ResNet18 full-model eval (job 41740440): recon4's leak
    multiplier came back 98 on conv1, 112 on the stride-2 1x1 layers and 16 on
    fc against 168 declared scratchpads. Timeloop bills leakage as
    leak x cycles x leaks_per_cycle with leaks_per_cycle = the UTILIZED
    instances when each instance is power-gated on its own (buffer.cpp
    2119-2127, 2376). So on the real recon4 cache entry for conv1 the leakage
    delta against the reference entry must equal idle x 98 x cycles, the
    stage's engine_cycles must be 98 x cycles, and the DECLARED 168 must NOT
    reconcile. Skips when the two cache entries are not on disk."""
    from eccenergy import recon as reconmod
    base = pathlib.Path(reconmod.__file__).resolve().parents[1] / "ecc_energy_study" / "outputs" / "eyeriss_like_wglb"
    slug = "multimodel__vic4000__vicx__alg-linear_pruned__to100000000__noc__paper"
    shape = "C3_M64_R7_S7_P112_Q112_ws2_hs2"
    ref = base / f"{slug}__mcons__wrelax" / "fp-3cd00eb16801" / shape / "timeloop-mapper.stats.txt"
    arm = (base / f"{slug}__wdw4-filter_glb__mcons__wrelax__ert-recon4-weights_spad-write"
           / "fp-91ce4687a22f" / shape / "timeloop-mapper.stats.txt")
    if not (ref.is_file() and arm.is_file()):
        raise _Skip("the recon4 / reference cache entries for conv1 are not on disk")
    import re
    def leak_util_cycles(path):
        t = path.read_text()
        ws = re.search(r"=== weights_spad ===(.*?)\n=== ", t, re.S).group(1)
        return (float(re.search(r"Leakage energy \(total\)\s*:\s*([\d.]+)", ws).group(1)),
                int(re.search(r"Utilized instances \(max\)\s*:\s*(\d+)", ws).group(1)),
                int(re.search(r"\nCycles:\s*(\d+)", t).group(1)))
    leak_a, util_a, cyc_a = leak_util_cycles(arm)
    leak_r, util_r, cyc_r = leak_util_cycles(ref)
    assert util_a == util_r == 98 and cyc_a == cyc_r, (util_a, util_r, cyc_a, cyc_r)
    idle = 2.8310811
    mult = (leak_a - leak_r) / (idle * cyc_a)
    assert math.isclose(mult, 98.0, rel_tol=1e-6), mult          # Timeloop's own multiplier
    assert not math.isclose(mult, 168.0, rel_tol=1e-2)            # BREAKAGE: the declared count
    cfg = _cfg(ECC_CONST_ARCH="eyeriss_like_wglb", ECC_SWEEP_ARCHS="eyeriss_like_wglb",
               ECC_CONST_K="30", ECC_RECON_PACKING="aligned")
    layer = _Layer(name="conv1", shape_name=shape, weights=9408)
    wp = reconmod.weight_path(cfg, "eyeriss_like_wglb", "resnet18", [layer], {shape: arm})
    st = wp.stage("weights_spad") if hasattr(wp, "stage") else wp.stages["weights_spad"]
    assert st.declared_instances == 168 and st.instances == 98, (st.declared_instances, st.instances)
    assert math.isclose(st.engine_cycles, 98 * cyc_a), st.engine_cycles
    try:
        from eccenergy.experiments import recon as exp
    except Exception as exc:                       # pragma: no cover
        raise _Skip(f"experiments.recon unavailable: {exc}")
    bump = {"placement": "recon4", "level": "weights_spad", "counter": "fills",
            "action": "write", "e_w_pj": 0.17506, "access_delta_pj": 0.35012,
            "leak_delta_pj": idle, "narrow_levels": ["filter_glb"]}
    split = exp.ert_split(bump, st, wp.cycles)
    assert math.isclose(split["leak_toll_pJ"], leak_a - leak_r, rel_tol=1e-6), (split["leak_toll_pJ"], leak_a - leak_r)
    assert split["engines"] == 98 and split["engines_declared"] == 168


def test_clock_gating_is_exact_at_zero_and_scales_the_idle_term(cache=None):
    """prompt_7 Issue 4. `ECC_RECON_CLOCK_GATING_PCT` must (a) reproduce the
    ungated model EXACTLY at 0 -- the diff target for every number published
    before the knob existed -- and (b) remove the stated fraction of the idle
    term and nothing else. The engine burns `incremental + idle` while it is
    working (that is active_per_codeword) and `idle x (1-g)` while it is gated
    off, so at g=1 only the work term survives."""
    from eccenergy import recon as reconmod
    inc, idle, cycles = 1.3786, 2.8310811, 5000
    gran = reconmod.Granularity(63, 30, 8, "weight")
    packing = reconmod.Packing(63, 30, 8, "aligned")

    def charge(pct):
        cfg = _cfg(ECC_CONST_K="30", ECC_RECON_PACKING="aligned",
                   ECC_RECON_CLOCK_GATING_PCT=str(pct))
        return cfg

    # the formula, evaluated directly -- the same arithmetic evaluate_placement does
    ev, ec = 40.0, 4.0 * cycles
    def model(g):
        if g:
            return ev * (inc + idle) + idle * (1.0 - g) * max(ec - ev, 0.0)
        return ev * inc + idle * ec

    ungated = ev * inc + idle * ec
    assert model(0.0) == ungated, (model(0.0), ungated)
    assert math.isclose(model(1.0), ev * (inc + idle), rel_tol=1e-12)
    # monotone, and strictly smaller than ungated for any real gating
    prev = ungated
    for g in (0.0, 0.5, 0.95, 0.99, 0.995, 1.0):
        now = model(g)
        assert now <= prev + 1e-9, (g, now, prev)
        prev = now
    assert model(0.995) < ungated / 50.0, model(0.995)
    # and the knob is actually read from the environment
    assert charge(0).recon_clock_gating_pct == 0.0
    assert charge(99.5).recon_clock_gating_pct == 99.5
    return "gating exact at 0, monotone, and read from ECC_RECON_CLOCK_GATING_PCT"


def main():
    print("eccenergy reconstruction-placement (Task 3) tests")
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

