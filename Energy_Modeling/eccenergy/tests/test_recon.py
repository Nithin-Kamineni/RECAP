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
        # The decoder is on the DRAM die and f_if has no cited default, so the
        # synthetic case fixes one: 5000 pJ of DRAM = 3750 array + 1250 interface.
        ECC_RECON_DECODE_SITE="ondie", ECC_DRAM_IF_FRAC="0.25",
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


#: The synthetic DRAM level is 5000 pJ; at the fixture's f_if = 0.25 that is
#: 3750 pJ of array (never reduced) and 1250 pJ of interface (x K/N on every
#: boundary). Every hand check below is written against these two numbers.
_F_IF = 0.25
_DRAM_W = 5000.0
_DRAM_ARRAY = _DRAM_W * (1 - _F_IF)
_DRAM_IFACE = _DRAM_W * _F_IF


def _dram_expected(frac, f_if=_F_IF, dram_w=_DRAM_W):
    """What every placement's DRAM component must be with the decoder on the die."""
    return dram_w - f_if * dram_w * (1.0 - frac)


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


# ------------------------------------------------------------- the loop nest
def test_loop_nest_separates_the_tile_from_the_consecutive_run():
    """The distinction R4b turns on -- see recon.retention_model."""
    from eccenergy.recon import weight_loop_nest
    with tempfile.TemporaryDirectory() as tmp:
        _write_cache(tmp)
        nest = weight_loop_nest(
            pathlib.Path(tmp) / "C8_M8" / "timeloop-mapper.map.txt")
        # the loops below the buffer walk 8 DISTINCT weights (M is the only
        # weight dimension among them)...
        assert nest["inner_tile"] == 8, nest
        # ...while a ONE-ENTRY latch would serve Q(2) x P(3) = 6 uses in a row
        assert nest["consecutive_run"] == 6, nest
        assert nest["innermost_weight_dim"] == "M"
        # M is the slowest index of the [M][C][R][S] stream order, so
        # consecutively accessed weights are NOT consecutive in the bit stream
        assert nest["stream_consecutive"] is False
        assert nest["weight_level"] == "weights_spad"

        # the innermost loop walking a weight dimension kills the latch, which
        # is what every real eyeriss_v2_like mapping does
        q = pathlib.Path(tmp) / "C8_M8" / "timeloop-mapper.map.txt"
        _write_lf(q, _MAP.replace("|     for M in [0:8)\n"
                                  "|       for P in [0:3)\n"
                                  "|         for Q in [0:2)\n",
                                  "|     for P in [0:3)\n"
                                  "|       for C in [0:2)\n"))
        nest = weight_loop_nest(q)
        assert nest["consecutive_run"] == 1, nest
        assert nest["inner_tile"] == 2 and nest["innermost_weight_dim"] == "C"

        # a missing map.txt must be reported, not guessed at
        nest = weight_loop_nest(pathlib.Path(tmp) / "nope" / "map.txt")
        assert nest["inner_tile"] == 0 and "no map.txt" in nest["evidence"]


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

        # The one DRAM level is TWO stages whose shares sum to one. Both see
        # every access (R1 counts its reconstructions off the array's reads);
        # only the energy is split.
        array, iface = wp.stage("dram_array"), wp.stage("dram_interface")
        assert array.reads == 2048.0 and iface.reads == 2048.0
        assert math.isclose(array.energy_pJ, _DRAM_ARRAY) and \
            math.isclose(iface.energy_pJ, _DRAM_IFACE)
        assert math.isclose(array.energy_pJ + iface.energy_pJ, _DRAM_W)
        assert array.level_share == 0.75 and iface.level_share == 0.25
        assert array.levels == ["DRAM"] and iface.levels == ["DRAM"]
        assert wp.decode_site == "ondie" and wp.dram_if_frac == 0.25
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

        # 400 reads for 40 fills -> a register holding the working set
        # reconstructs 40 times: a 10x amortization
        ret = wp.retention
        assert ret["weights_reconstructed_with_retention"] == 40.0, ret
        assert ret["weights_reconstructed_without_retention"] == 400.0, ret
        assert math.isclose(ret["amortization_vs_no_retention"], 10.0)
        # the working set is the LARGER of the walked tile (8) and the resident
        # capacity (24): a register covering less than it catches nothing
        assert ret["register_entries_required_max"] == 24, ret


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
            recon_pj=4.0, reuse_reg_pj=0.03, gran=gran, packing=packing)
        assert res.status == "evaluated", (p.key, res.reason)
        out[p.key] = res
        # THE DRAM TERM, under every boundary: array untouched, interface x K/N
        #   5000 - 1250 x (1 - 51/63) = 4761.905
        assert math.isclose(res.components["DRAM"], _dram_expected(frac),
                            rel_tol=1e-12), (p.key, res.components["DRAM"])
        dm = res.detail["dram_model"]
        assert dm["dram_array_after_pJ"] == dm["dram_array_pJ"] == _DRAM_ARRAY
        assert dm["dram_array_reduced"] is False
        assert dm["dram_interface_reduced"] is True
        assert math.isclose(dm["dram_interface_after_pJ"], _DRAM_IFACE * frac,
                            rel_tol=1e-12)
        assert math.isclose(dm["dram_interface_saving_pJ"],
                            _DRAM_IFACE * (1 - frac), rel_tol=1e-12)
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
    assert math.isclose(saved["DRAM"], _DRAM_IFACE * (1 - frac), rel_tol=1e-12)
    assert math.isclose(r1.components["NoC"], 150.0, rel_tol=1e-12)
    assert math.isclose(r1.components["Local (spads/RF)"], 911.0, rel_tol=1e-12)
    assert math.isclose(r1.components["Reconstruction"],
                        2048 / (63 / 8) * 4.0, rel_tol=1e-12)
    # ...so R1 beats the embedded reference by the interface saving minus its
    # reconstruction cost, and by nothing else
    assert math.isclose(float(base.sum()) - r1.total_pJ,
                        _DRAM_IFACE * (1 - frac) - 2048 / (63 / 8) * 4.0,
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

    # R4b saves the SAME energy as R4a but rebuilds once per FILL (40) instead
    # of once per read (400), and pays one register write per weight retained.
    # Its reconstruction count therefore EQUALS R3's, which is the whole point:
    # R4b is R3's encoder count with R4a's storage saving.
    r4b = out["recon5"]
    assert math.isclose(r4b.components["Local (spads/RF)"],
                        r4a.components["Local (spads/RF)"], rel_tol=1e-12)
    assert math.isclose(r4b.components["Reconstruction"],
                        40 / (63 / 8) * 4.0, rel_tol=1e-12)
    assert math.isclose(r4b.components["Reconstruction"],
                        r3.components["Reconstruction"], rel_tol=1e-12)
    # Overhead = one register WRITE per weight retained, PLUS the register's
    # lockstep READS. The read term is the 2026-09-08 audit fix: before it, the
    # register was charged a write and nothing else while R4b still took the
    # K/N discount on all 400 SPad reads, which is not a consistent machine.
    # See recon.ReuseRegister.
    spad = wp.stages["weight_spad"]
    wpw = packing.weights_per_word(spad.block_bits, reduced=True)
    want = 40 * 0.03 + (400 / wpw) * 0.03 * (1 - frac)
    assert math.isclose(r4b.components["Recon overhead"], want, rel_tol=1e-12), (
        r4b.components["Recon overhead"], want)
    # ...so R4b must be the cheapest boundary, and cheaper than the embedded
    # reference it is measured against
    assert r4b.total_pJ < r4a.total_pJ
    assert r4b.total_pJ == min(r.total_pJ for r in out.values())
    assert r4b.total_pJ < float(base.sum())


def test_a_register_smaller_than_the_working_set_collapses_r4b_onto_r4a():
    """A cyclic walk is the LRU worst case, so a one-entry latch catches nothing.

    This is the case the first version of the model applied UNCONDITIONALLY,
    which is why R4b came out useless. It is a real case -- it is what a latch
    does -- but it is what `ECC_RECON_REUSE_REG_ENTRIES=1` asks for, not the
    default.
    """
    from eccenergy import recon as reconmod
    cfg, wp, base, base_w, gran, packing = _placement_setup(
        ECC_RECON_REUSE_REG_ENTRIES="1")
    assert wp.retention["weights_reconstructed_with_retention"] == 400.0
    assert math.isclose(wp.retention["amortization_vs_no_retention"], 1.0)
    r4a = reconmod.evaluate_placement(
        cfg, "eyeriss_v2_like", reconmod.placement_by_key("eyeriss_v2_like", "recon4"),
        wp, base_w, base, 4.0, 0.03, gran, packing)
    r4b = reconmod.evaluate_placement(
        cfg, "eyeriss_v2_like", reconmod.placement_by_key("eyeriss_v2_like", "recon5"),
        wp, base_w, base, 4.0, 0.03, gran, packing)
    assert math.isclose(r4b.components["Reconstruction"],
                        r4a.components["Reconstruction"], rel_tol=1e-12)
    # ...and the register is then pure overhead, so R4b must cost MORE
    assert r4b.total_pJ > r4a.total_pJ


def test_a_tile_sized_register_beats_every_other_boundary():
    """The ordering the source discussion's ratings predict: R4b > R3 > R4a."""
    from eccenergy import recon as reconmod
    cfg, wp, base, base_w, gran, packing = _placement_setup()
    got = {}
    for key in ("recon1", "recon2", "recon3", "recon4", "recon5"):
        p = reconmod.placement_by_key("eyeriss_v2_like", key)
        got[key] = reconmod.evaluate_placement(
            cfg, "eyeriss_v2_like", p, wp, base_w, base, 4.0, 0.03, gran,
            packing).total_pJ
    assert got["recon5"] < got["recon3"] < got["recon4"], got
    assert got["recon5"] == min(got.values()), got
    # ...and the register capacity it needs is reported, not hidden
    assert wp.retention["register_entries_required_max"] == 24


def test_a_pe_local_boundary_is_rejected_when_the_tile_is_smaller_than_g_rec():
    """"Reject infeasible local placements", checked, not assumed."""
    from eccenergy import recon as reconmod
    cfg, wp, base, base_w, gran, packing = _placement_setup()
    assert gran.g_rec == 9

    # 24 resident weights >= 9: both PE-local boundaries are feasible
    for key in ("recon4", "recon5"):
        p = reconmod.placement_by_key("eyeriss_v2_like", key)
        ok, detail = reconmod.feasibility(p, "eyeriss_v2_like", wp, gran)
        assert ok is True, detail

    # a depthwise-style tile of 3 weights cannot assemble the group
    wp.per_layer[0]["stages"]["weight_spad"]["weights_resident_per_instance"] = 3
    for key in ("recon4", "recon5"):
        p = reconmod.placement_by_key("eyeriss_v2_like", key)
        ok, detail = reconmod.feasibility(p, "eyeriss_v2_like", wp, gran)
        assert ok is False, detail
        assert detail["infeasible_layers"][0]["weights_resident"] == 3
        res = reconmod.evaluate_placement(cfg, "eyeriss_v2_like", p, wp, base_w,
                                          base, 4.0, 0.03, gran, packing)
        assert res.status == "unsupported" and res.total_pJ == 0.0

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
    for key in ("recon3", "recon4", "recon5"):
        p = reconmod.placement_by_key("eyeriss_v2_like", key)
        res = reconmod.evaluate_placement(cfg, "eyeriss_v2_like", p, wp, base_w,
                                          base, 4.0, 0.03, gran, packing)
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
                c, "eyeriss_v2_like", p, wp, base_w, base, 4.0, 0.03,
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
                 "dram_array_identical_to_embedded_reference",
                 "dram_interface_scaled_by_K_over_N",
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
    # ...and the pre-2026-09-09 single DRAM check is gone, not merely renamed
    # alongside: a result file carries the two directional checks instead.
    assert "dram_identical_to_embedded_reference" not in good, sorted(good)

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


def test_a_silent_dram_array_credit_fails_the_array_check_and_only_that_one():
    """CHEAT DOWNWARD: a placement that takes more DRAM than the interface saving.

    The array is never reduced, so a DRAM component BELOW emb_DRAM - f_if x
    DRAM_w x (1 - K/N) is an array credit. `dram_array_identical_to_embedded_
    reference` must fail; `dram_interface_scaled_by_K_over_N` bounds the other
    direction and must still pass, or the two checks are one check twice.
    """
    def cheat_array(r):
        r.components["DRAM"] *= 0.9
        r.total_pJ = sum(r.components.values())
    got = _task3_validation(cheat_array)
    assert got["dram_array_identical_to_embedded_reference"] is False, got
    assert got["dram_interface_scaled_by_K_over_N"] is True, got

    # ...and a cheat visible only in the detail rows (the stack is right, the
    # record claims the array was reduced) is caught by the same check
    def cheat_rows(r):
        r.detail["dram_model"]["dram_array_reduced"] = True
        r.detail["dram_model"]["dram_array_after_pJ"] *= 0.9
    got = _task3_validation(cheat_rows)
    assert got["dram_array_identical_to_embedded_reference"] is False, got
    assert got["dram_interface_scaled_by_K_over_N"] is True, got


def test_an_unscaled_dram_interface_fails_the_interface_check_and_only_that_one():
    """CHEAT UPWARD: a placement that leaves the interface at full width.

    Putting the interface saving back -- DRAM identical to the embedded
    reference, the pre-2026-09-09 behaviour -- is now the failure: with the
    decoder on the DRAM die only the k message bits leave it, and a bar whose
    DRAM component sits ABOVE emb_DRAM - f_if x DRAM_w x (1 - K/N) has not
    taken that saving. The array check bounds the other direction and passes.
    """
    def cheat_interface(r):
        r.components["DRAM"] = 5000.0
        r.total_pJ = sum(r.components.values())
    got = _task3_validation(cheat_interface)
    assert got["dram_interface_scaled_by_K_over_N"] is False, got
    assert got["dram_array_identical_to_embedded_reference"] is True, got

    # ...and the detail-row version of the same cheat
    def cheat_rows(r):
        r.detail["dram_model"]["dram_interface_after_pJ"] = \
            r.detail["dram_model"]["dram_interface_pJ"]
    got = _task3_validation(cheat_rows)
    assert got["dram_interface_scaled_by_K_over_N"] is False, got
    assert got["dram_array_identical_to_embedded_reference"] is True, got


# ------------------------------------------- the decode site and f_if
def test_controller_site_reproduces_the_pre_2026_09_09_numbers():
    """`ECC_RECON_DECODE_SITE=controller` must be the OLD model to the digit.

    These are the hand checks the suite carried before the decoder moved onto
    the DRAM die: DRAM identical on every bar, R1 saving nothing, R2 the mesh
    only, R3 both networks, R4a/R4b the scratchpad too. If they stop holding
    under `controller`, the diff row proves nothing.
    """
    from eccenergy import recon as reconmod
    cfg, wp, base, base_w, gran, packing = _placement_setup(
        ECC_RECON_DECODE_SITE="controller")
    assert wp.decode_site == "controller"
    # the interface stage exists, is split by the same f_if, and is NOT reducible
    assert math.isclose(wp.stages["dram_interface"].energy_pJ, _DRAM_IFACE)
    assert not any(s.reducible for s in reconmod.stages_for("eyeriss_v2_like", cfg)
                   if s.key == "dram_interface")
    # ...and it has left every placement's reduced set, R1's included
    for p in reconmod.placements_for("eyeriss_v2_like", cfg):
        assert "dram_interface" not in p.reduced, p
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
        assert "dram_interface" in p.reduced
        res = reconmod.evaluate_placement(cfg, "eyeriss_v2_like", p, wp, base_w,
                                          base, 4.0, 0.03, gran, packing)
        assert res.status == "evaluated", (p.key, res.reason)
        out[p.key] = res
        assert res.components["DRAM"] == 5000.0, (p.key, res.components["DRAM"])
        assert math.isclose(res.components["Compute"], 9000.0, rel_tol=1e-12)
        dm = res.detail["dram_model"]
        assert dm["decode_site"] == "controller"
        assert dm["dram_interface_reduced"] is False
        assert dm["dram_interface_saving_pJ"] == 0.0
        assert "dram_interface" not in res.placement.reduced
    r1, r2, r3, r4a, r4b = (out[k] for k in ("recon1", "recon2", "recon3",
                                             "recon4", "recon5"))
    assert math.isclose(sum(r1.detail["energy_saved_by_category_pJ"].values()),
                        0.0, abs_tol=1e-9)
    assert math.isclose(r1.components["Reconstruction"], 2048 / (63 / 8) * 4.0,
                        rel_tol=1e-12)
    assert math.isclose(r2.components["NoC"], 130.0 * frac + 20.0, rel_tol=1e-12)
    assert math.isclose(r3.components["NoC"], 150.0 * frac, rel_tol=1e-12)
    assert math.isclose(r3.components["Local (spads/RF)"], 911.0, rel_tol=1e-12)
    assert math.isclose(r4a.components["Local (spads/RF)"],
                        911.0 - 800.0 * (1 - frac), rel_tol=1e-12)
    assert math.isclose(r4b.components["Reconstruction"],
                        r3.components["Reconstruction"], rel_tol=1e-12)
    # R1 is worse than the embedded reference under controller-side correction
    assert r1.total_pJ > float(base.sum())

    # the same placements under the on-die model differ from these by EXACTLY
    # the interface saving, on every bar, and by nothing else
    cfg2, wp2, _b, _bw, gran2, packing2 = _placement_setup()
    for key, old in out.items():
        new = reconmod.evaluate_placement(
            cfg2, "eyeriss_v2_like", reconmod.placement_by_key("eyeriss_v2_like", key),
            wp2, base_w, base, 4.0, 0.03, gran2, packing2)
        for cat in old.components:
            diff = old.components[cat] - new.components[cat]
            want = _DRAM_IFACE * (1 - frac) if cat == "DRAM" else 0.0
            assert math.isclose(diff, want, rel_tol=1e-12, abs_tol=1e-9), (key, cat)

    # ...and the recorded checks pass under `controller` too, where both DRAM
    # checks collapse onto "identical to the embedded reference"
    good = _task3_validation(ECC_RECON_DECODE_SITE="controller")
    for name in _TASK3_CHECKS:
        assert good[name] is True, (name, good)

    # an unset f_if is fine under controller: nothing depends on the split
    cfg3, wp3, *_ = _placement_setup(ECC_RECON_DECODE_SITE="controller",
                                     ECC_DRAM_IF_FRAC="")
    assert cfg3.dram_if_frac is None
    assert wp3.dram_if_frac == 0.0
    assert math.isclose(wp3.stages["dram_array"].energy_pJ, _DRAM_W)
    assert wp3.stages["dram_interface"].energy_pJ == 0.0
    ok, _ = reconmod.cross_check(cfg3, "eyeriss_v2_like", wp3, base_w)
    assert ok is True


def test_ondie_without_f_if_refuses_and_the_knobs_are_validated():
    """No cited f_if, no number. `ECC_DRAM_IF_FRAC` unset under `ondie` is a
    refusal in `weight_path()` (and `run()` prints the ceiling at several
    values first); the two knobs reject what they cannot mean."""
    from eccenergy import recon as reconmod
    from eccenergy.config import ConfigError
    cfg = _cfg(ECC_DRAM_IF_FRAC="")
    assert cfg.dram_if_frac is None and cfg.recon_decode_site == "ondie"
    with tempfile.TemporaryDirectory() as tmp:
        stats = _write_cache(tmp)
        layer = _Layer()
        try:
            reconmod.weight_path(cfg, "eyeriss_v2_like", "resnet18", [layer],
                                 {layer.shape_name: stats})
        except ValueError as exc:
            assert "ECC_DRAM_IF_FRAC" in str(exc)
        else:
            raise AssertionError("weight_path evaluated with no f_if under ondie")
    # the sensitivity list the refusal prints is the one the spec names
    assert reconmod.F_IF_SENSITIVITY == (0.10, 0.25, 0.50)

    for bad in dict(ECC_RECON_DECODE_SITE="dimm"), dict(ECC_DRAM_IF_FRAC="1.5"), \
            dict(ECC_DRAM_IF_FRAC="-0.1"), dict(ECC_DRAM_IF_FRAC="0"):
        try:
            _cfg(**bad)
        except ConfigError:
            pass
        else:
            raise AssertionError(f"{bad} was accepted")
    # f_if = 0 is only refused where it would leave every placement an empty
    # interface stage; under controller it is the same as unset
    assert _cfg(ECC_DRAM_IF_FRAC="0", ECC_RECON_DECODE_SITE="controller").dram_if_frac == 0.0
    # the title carries the decode site and f_if, so a figure cannot be quoted
    # without them
    t = _cfg().recon_title()
    assert "f_if = 0.25" in t and "DRAM die" in t, t
    t = _cfg(ECC_RECON_DECODE_SITE="controller").recon_title()
    assert "controller" in t.lower() and "pre-2026-09-09" in t, t


def test_the_dram_level_is_claimed_exactly_once_in_total():
    """Two stages, one level: the shares must sum to one and nothing else may
    claim the DRAM level. A third claimant, or a pair that is not the
    array/interface pair, is a table error and is refused, not double-counted."""
    from eccenergy import recon as reconmod
    cfg = _cfg()
    stages = reconmod.stages_for("eyeriss_v2_like", cfg)
    dram = [s for s in stages if s.matches("DRAM")]
    assert [s.key for s in dram] == ["dram_array", "dram_interface"]
    assert {s.dram_share for s in dram} == {"array", "interface"}
    assert not dram[0].reducible and dram[1].reducible
    shares = reconmod._level_shares(dram, 0.25, "DRAM")
    assert shares == {"dram_array": 0.75, "dram_interface": 0.25}
    assert math.isclose(sum(shares.values()), 1.0)
    for f in (0.1, 0.5, 1.0):
        assert math.isclose(sum(reconmod._level_shares(dram, f, "DRAM").values()), 1.0)
    # the wglb path keeps the pair FIRST, before its extra weight GLB
    keys = [s.key for s in reconmod.stages_for("eyeriss_v2_like_wglb", cfg)]
    assert keys[:3] == ["dram_array", "dram_interface", "weight_glb"], keys
    # a stage that is not part of the pair may not share the level
    rogue = dram + [reconmod.Stage("x", "x", "storage", ("DRAM",), True)]
    try:
        reconmod._level_shares(rogue, 0.25, "DRAM")
    except ValueError:
        pass
    else:
        raise AssertionError("three claimants of one level were accepted")
    try:
        reconmod._level_shares([dram[1], dram[1]], 0.25, "DRAM")
    except ValueError:
        pass
    else:
        raise AssertionError("two interface stages were accepted")


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
        iface_saving = wp.stages["dram_interface"].energy_pJ * (1 - packing.frac)
        r1 = out["recon1"]
        recon1 = r1.components["Reconstruction"] + r1.components["Recon overhead"]
        assert math.isclose(emb - r1.total_pJ, iface_saving - recon1, rel_tol=1e-12), \
            case["name"]
        assert set(r1.detail["energy_saved_by_category_pJ"]) == {"DRAM"}, case["name"]
        for key in ("recon2", "recon3", "recon4", "recon5"):
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
def test_recon_optimizer_true_is_refused_rather_than_ignored():
    from eccenergy.config import ConfigError
    try:
        _cfg(ECC_RECON_OPTIMIZER="True")
    except ConfigError as exc:
        assert "TASK 4" in str(exc), str(exc)
    else:
        raise AssertionError("RECON_OPTIMIZER=True was accepted; it must stop "
                             "the run rather than silently produce "
                             "fixed-mapping numbers")
    # False, and the spelling env.sh uses, are both fine
    assert _cfg(ECC_RECON_OPTIMIZER="False").recon_optimizer is False


def test_the_placement_study_refuses_more_than_one_architecture():
    from eccenergy.config import ConfigError
    try:
        _cfg(ECC_SWEEP_ARCHS="eyeriss_v2_like eyeriss_like")
    except ConfigError as exc:
        assert "ONE architecture" in str(exc), str(exc)
    else:
        raise AssertionError("two architectures were accepted for the placement "
                             "study, whose boundaries are per architecture")


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
    "dram_if_frac": "ECC_DRAM_IF_FRAC",
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
    _REAL_CACHE["case"] = {
        "name": f"cached Timeloop output, {arch}/{model}",
        "cfg": cfg, "arch": arch,
        "wpath": reconmod.weight_path(cfg, arch, model, layers, stats),
        "base": pd.Series(raw["base"]).reindex(cats, fill_value=0.0),
        "base_w": pd.Series(raw["base_w"]).reindex(cats, fill_value=0.0),
        "recon_pj": float(man.get("recon_pj_per_codeword", 4.1296273)),
        "reg_pj": 0.0328125, "stats": stats,
    }
    return _REAL_CACHE["case"]


def _synthetic_case(**env):
    cfg, wp, base, base_w, _gran, _packing = _placement_setup(**env)
    return {"name": "synthetic stats.txt", "cfg": cfg, "arch": "eyeriss_v2_like",
            "wpath": wp, "base": base, "base_w": base_w,
            "recon_pj": 4.0, "reg_pj": 0.03, "stats": None}


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
            case["recon_pj"], case["reg_pj"], gran, packing)
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
        assert set(placements["recon5"].reduced) == set(placements["recon4"].reduced)
        # ...and every reduced set starts at the die's output
        for p in placements.values():
            assert p.reduced[0] == "dram_interface", p

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

        if "recon4" in ev and "recon5" in ev:
            for cat in _TRANSPORT_CATS:
                assert out["recon5"].components.get(cat, 0.0) == \
                    out["recon4"].components.get(cat, 0.0), (case["name"], cat)


# ---------------------------------------------------- 4. the closed form

def test_the_saving_of_each_boundary_matches_its_closed_form():
    """CLOSED-FORM ARITHMETIC -- the identity, not an inequality.

        DRAM(Rx)   == DRAM(embedded)  - f_if x DRAM_w x (1 - K/N)   for EVERY x
        NoC(R2)    == NoC(embedded)   - mesh_w x (1 - K/N)
        NoC(R3)    == NoC(embedded)   - (mesh_w + cluster_local_w) x (1 - K/N)
        Local(R4a) == Local(embedded) - weight_spad_w x (1 - K/N)

    An inequality still passes when a saving lands on the wrong stage. The
    identity does not, and it also pins the ceiling the result file prints
    against the same arithmetic. DRAM_w is the WEIGHT share of the DRAM
    category (the category also holds inputs and outputs on the real record),
    which is what f_if splits.
    """
    for case in _cases():
        out, packing = _evaluate(case)
        ev = _evaluated(case, out)
        wp, base, name = case["wpath"], case["base"], case["name"]
        d = 1.0 - packing.frac
        mesh = wp.stages["inter_cluster_mesh"].energy_pJ
        local = wp.stages["cluster_local"].energy_pJ
        spad = wp.stages["weight_spad"].energy_pJ
        noc0 = float(base["NoC"])
        loc0 = float(base["Local (spads/RF)"])
        dram0 = float(base["DRAM"])
        dram_w = float(case["base_w"]["DRAM"])
        iface = wp.stages["dram_interface"].energy_pJ
        assert math.isclose(iface, wp.dram_if_frac * dram_w, rel_tol=1e-12), name
        assert math.isclose(wp.stages["dram_array"].energy_pJ + iface, dram_w,
                            rel_tol=1e-12), name
        for key in ev:
            assert math.isclose(out[key].components["DRAM"], dram0 - iface * d,
                                rel_tol=1e-12), (name, key)

        assert math.isclose(out["recon1"].components["NoC"], noc0, rel_tol=1e-12), name
        assert math.isclose(out["recon1"].components["Local (spads/RF)"], loc0,
                            rel_tol=1e-12), name
        assert math.isclose(out["recon2"].components["NoC"], noc0 - mesh * d,
                            rel_tol=1e-12), name
        assert math.isclose(out["recon2"].components["Local (spads/RF)"], loc0,
                            rel_tol=1e-12), name
        for key in ("recon3", "recon4", "recon5"):
            if key in ev:
                assert math.isclose(out[key].components["NoC"],
                                    noc0 - (mesh + local) * d, rel_tol=1e-12), (name, key)
        assert math.isclose(out["recon3"].components["Local (spads/RF)"], loc0,
                            rel_tol=1e-12), name
        for key in ("recon4", "recon5"):
            if key in ev:
                assert math.isclose(out[key].components["Local (spads/RF)"],
                                    loc0 - spad * d, rel_tol=1e-12), (name, key)

        for key, stages in (("recon1", ("dram_interface",)),
                            ("recon2", ("dram_interface", "inter_cluster_mesh")),
                            ("recon3", ("dram_interface", "inter_cluster_mesh",
                                        "cluster_local")),
                            ("recon4", ("dram_interface", "inter_cluster_mesh",
                                        "cluster_local", "weight_spad"))):
            if key not in ev:
                continue
            ceiling = out[key].detail["reducible_weight_energy_pJ"]
            want = sum(wp.stages[s].energy_pJ for s in stages)
            assert math.isclose(ceiling["before_pJ"], want, rel_tol=1e-12), (name, key)
            assert math.isclose(ceiling["ceiling_on_the_saving_pJ"], want * d,
                                rel_tol=1e-12), (name, key)


# ---------------------------------------------------- 5. nothing goes up
def test_no_stage_or_transport_category_exceeds_the_embedded_reference():
    """NOTHING GOES UP.

    Reconstruction and its register are charged as their own categories, so no
    weight-path stage and no transport or storage category may ever sit ABOVE
    the embedded reference under any boundary. Compute is required to be
    exactly equal, and DRAM to sit at EXACTLY emb_DRAM - f_if x DRAM_w x
    (1 - K/N) -- the array is the embedded arm's under every boundary and the
    interface is x K/N under every boundary, so both a silent array credit and
    a forgotten interface scaling are inequalities this catches.
    """
    for case in _cases():
        out, packing = _evaluate(case)
        base, wp = case["base"], case["wpath"]
        dram0 = float(base["DRAM"])                 # the category, all tensors
        dram_w = float(case["base_w"]["DRAM"])      # its weight share, what f_if splits
        want = dram0 - wp.dram_if_frac * dram_w * (1.0 - packing.frac)
        for key, res in _evaluated(case, out).items():
            for stage_key, row in _stage_rows(res).items():
                assert _after(row) <= row["weight_energy_pJ"] + 1e-6, \
                    (case["name"], key, stage_key)
                if stage_key == "dram_array":
                    assert _after(row) == row["weight_energy_pJ"], (case["name"], key)
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
        assert len(stages) == 1 or sorted(stages) == ["dram_array", "dram_interface"], \
            (lv, stages)
    assert sorted(who.get("DRAM", [])) == ["dram_array", "dram_interface"], who

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
    assert bad == ["recon2", "recon3", "recon4", "recon5"], bad


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


# ------------------------------------------------- the reuse register's WIDTH
# Added 2026-09-08 with `recon.ReuseRegister`. These pin the audit finding and
# its fix: R4b used to cut its reconstruction count 82.85x BECAUSE the register
# served the scratchpad reads, while still billing every one of those reads at
# K/N. The three modes are the three ways to close that, and each is pinned
# here so the default cannot silently drift back to the inconsistent one.


def test_the_complement_register_is_n_minus_k_bits_and_costs_the_pe_nothing():
    """Reduced SPad + complement register = exactly weight_bits per weight.

    This is the claim that makes R4b defensible: the register holds only the
    bits the encoder regenerates, so the PE's weight storage per resident
    weight is UNCHANGED from the baseline at every K -- against 1.81x for a
    full-width register. If this ever stops being exactly 1.00x, either the
    width or the packing model has drifted.
    """
    from eccenergy import recon as reconmod
    for k in (57, 51, 45, 39, 36, 30):
        packing = reconmod.Packing("stream", 8, k, 63)
        reg = reconmod.ReuseRegister("complement", 8, k, 63, 0.03)
        assert math.isclose(reg.bits_per_weight, 8 * (1 - k / 63), rel_tol=1e-12)
        assert math.isclose(reg.storage_bits_per_resident_weight(packing), 8.0,
                            rel_tol=1e-12), k
        assert reg.serves_reads is False
        # a complement read is charged only its share of the word
        assert math.isclose(reg.read_pj, 0.03 * (1 - k / 63), rel_tol=1e-12)
    full = reconmod.ReuseRegister("full_width", 8, 51, 63, 0.03)
    assert full.bits_per_weight == 8.0
    assert full.serves_reads is True
    assert math.isclose(
        full.storage_bits_per_resident_weight(reconmod.Packing("stream", 8, 51, 63)),
        8 * 51 / 63 + 8, rel_tol=1e-12)


def test_free_mode_reproduces_the_pre_audit_accounting_exactly():
    """`free` must stay bit-identical to the old model, or a diff proves nothing.

    It is kept ONLY so the historical +1.36% / +2.97% can be reproduced. If this
    test fails, `free` has stopped being the historical accounting and the
    before/after comparison in FINDINGS section 7 is no longer checkable.
    """
    from eccenergy import recon as reconmod
    cfg, wp, base, base_w, gran, packing = _placement_setup(
        ECC_RECON_REUSE_REG_MODEL="free")
    r4b = [p for p in reconmod.placements_for("eyeriss_v2_like")
           if p.key == "recon5"][0]
    res = reconmod.evaluate_placement(cfg, "eyeriss_v2_like", r4b, wp, base_w,
                                      base, recon_pj=4.0, reuse_reg_pj=0.03,
                                      gran=gran, packing=packing)
    # writes only, no read term -- the old formula, to the digit
    assert math.isclose(res.components["Recon overhead"], 40 * 0.03,
                        rel_tol=1e-12)
    # and the SPad still carries the full K/N discount on all 400 reads
    frac = 51 / 63
    assert math.isclose(res.components["Local (spads/RF)"],
                        911.0 - 800.0 * (1 - frac), rel_tol=1e-12)


def test_a_full_width_register_must_remove_the_spad_reads_it_serves():
    """`full_width` may not both serve the reads and leave them billed.

    The audit's reading, as a runnable row. Two things must hold: the SPad's
    read count drops from `reads` to `fills`, and the register is charged a read
    per weight DELIVERED. It also has to REFUSE when the ERT is not there to
    split the SPad's read from its write energy, rather than estimating.
    """
    from eccenergy import recon as reconmod
    cfg, wp, base, base_w, gran, packing = _placement_setup(
        ECC_RECON_REUSE_REG_MODEL="full_width")
    r4b = [p for p in reconmod.placements_for("eyeriss_v2_like")
           if p.key == "recon5"][0]
    res = reconmod.evaluate_placement(cfg, "eyeriss_v2_like", r4b, wp, base_w,
                                      base, recon_pj=4.0, reuse_reg_pj=0.03,
                                      gran=gran, packing=packing)
    # the synthetic fixture has no ERT, so this must be a refusal, not a guess
    assert res.status == "unsupported", res.status
    assert "read/write energy split" in res.reason, res.reason


def test_the_complement_read_term_is_keyed_to_the_code_not_hard_wired():
    """A stronger code must charge MORE complement reads, not the same.

    The mutation this catches is the one the K sweep caught last time: a
    reduction that is applied but not keyed to K passes every fixed-K test.
    Here the complement is `weight_bits x (1 - k/n)`, so it must GROW as k
    falls, while the register write term stays flat.
    """
    from eccenergy import recon as reconmod
    seen = []
    for k in (57, 51, 45, 39):
        cfg, wp, base, base_w, gran, packing = _placement_setup(ECC_CONST_K=str(k))
        r4b = [p for p in reconmod.placements_for("eyeriss_v2_like")
               if p.key == "recon5"][0]
        res = reconmod.evaluate_placement(cfg, "eyeriss_v2_like", r4b, wp,
                                          base_w, base, recon_pj=4.0,
                                          reuse_reg_pj=0.03, gran=gran,
                                          packing=packing)
        rc = res.detail["reconstruction_counts"]
        seen.append((k, rc["reuse_register_read_energy_pJ"],
                     rc["reuse_register_write_energy_pJ"]))
    reads = [r for _, r, _ in seen]
    writes = [w for _, _, w in seen]
    assert reads == sorted(reads), seen        # k falls -> complement grows
    assert len(set(reads)) == len(reads), seen  # and never flat
    assert len(set(round(w, 12) for w in writes)) == 1, seen


def test_full_width_really_moves_the_spad_reads_on_the_real_cache():
    """On the REAL cache, `full_width` must bill the SPad once per fill.

    The synthetic fixture has no Accelergy ERT, so `full_width` refuses there
    and the read-removal path is never exercised -- a mutation that deleted the
    override went uncaught until this test existed. Here the mapper cache does
    carry the ERT, so the whole path runs and the numbers are checkable:

      * the site stage's energy must collapse towards the FILL count (the SPad
        is read once per weight filled, not once per MAC),
      * the register must be charged a read per weight DELIVERED, and
      * the ECC-marginal diagnostic must be much smaller than what the same
        register gives a PE with no ECC -- which is the audit's whole point,
        recorded as a number so it cannot be lost again.
    """
    from dataclasses import replace
    from eccenergy import recon as reconmod
    case = _real_case()
    if case is None:
        raise _Skip(_REAL_CACHE.get("why", "no real mapper cache"))
    cfg, wp, arch = case["cfg"], case["wpath"], case["arch"]
    r4b = [p for p in reconmod.placements_for(arch, cfg) if p.key == "recon5"][0]
    gran = reconmod.Granularity(cfg.code_n, cfg.code_k, cfg.weight_bits,
                                cfg.recon_granularity)
    packing = reconmod.Packing(cfg.recon_packing, cfg.weight_bits,
                               cfg.code_k, cfg.code_n)

    def run(mode):
        return reconmod.evaluate_placement(
            replace(cfg, recon_reuse_reg_model=mode), arch, r4b, wp,
            case["base_w"], case["base"], recon_pj=case["recon_pj"],
            reuse_reg_pj=case["reg_pj"], gran=gran, packing=packing)

    spad = wp.stages[r4b.site_stage]
    assert spad.reads > 10 * spad.fills, (spad.reads, spad.fills)

    comp = run("complement")
    if comp.status != "evaluated" and "infeasible local placement" in comp.reason:
        # a fact about the mapping, not about the register model: under the 8x2
        # EDP mappings R4b keeps fewer than G_rec weights resident in layer4
        raise _Skip(f"R4b is unsupported on this cache: {comp.reason[:80]}")
    assert comp.status == "evaluated", comp.reason
    row = [r for r in comp.detail["weight_path_stages"]
           if r["stage"] == r4b.site_stage][0]
    # complement: the SPad keeps every read, discounted by exactly K/N
    assert math.isclose(row["scale"], packing.frac, rel_tol=1e-9), row["scale"]

    full = run("full_width")
    assert full.status == "evaluated", full.reason
    notes = full.detail["reconstruction_counts"]["reuse_register_notes"]
    moved = notes["spad_reads_moved_to_the_register"]
    assert moved["spad_scalar_reads_before"] == spad.reads
    assert moved["spad_scalar_reads_after"] == spad.fills
    assert moved["ert_reconciles_timeloop_to"] < 1e-3, moved

    # the site stage must actually get much cheaper -- this is what the deleted
    # override used to break silently
    frow = [r for r in full.detail["weight_path_stages"]
            if r["stage"] == r4b.site_stage][0]
    assert frow["scale"] < 0.1 * packing.frac, (frow["scale"], packing.frac)
    assert frow["energy_after_pJ"] < 0.1 * row["energy_after_pJ"]

    # and the register is charged per weight DELIVERED, not per word
    rc = full.detail["reconstruction_counts"]
    assert math.isclose(rc["reuse_register_reads"], spad.reads, rel_tol=1e-12)

    # the diagnostic that names what is ECC's and what is just a register
    g = notes["what_the_same_register_gives_a_pe_with_no_ecc"]
    assert g["the_register_alone_saves_pJ"] > 0
    assert g["ecc_marginal_on_top_of_the_register_pJ"] < \
        0.1 * g["the_register_alone_saves_pJ"], g


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
