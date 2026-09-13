"""Tests for the interconnect (NoC) model: parsing and injection.

    python3 -m eccenergy.tests.test_noc

Offline and dependency-free like test_results_store: never invokes the mapper.
The fixture `fixtures/stats_with_networks.txt` is a real timeloop-mapper
stats.txt from eyeriss_like (C1280 M1000 FC layer) mapped WITH wire energy on
the spatial containers, so its `Networks` section is populated. Its numbers are
the ground truth: Timeloop's own summary prints the network at 98.38 fJ/compute
over 1,280,000 computes and a total of 85.20 uJ.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
import re
import sys
import traceback

import pytest
import yaml

FAILURES = []
HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "stats_with_networks.txt"
ARCHS = ["eyeriss_like", "eyeriss_like_wglb", "eyeriss_v2_like", "eyeriss_v2_like_wglb",
         "simba_like", "simple_weight_stationary", "simple_output_stationary",
         "simple_input_stationary"]


def check(name, fn):
    try:
        fn()
    except Exception:
        FAILURES.append(name)
        print(f"  FAIL  {name}")
        traceback.print_exc()
    else:
        print(f"  ok    {name}")


def _cfg(**env):
    from .. import config
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        cfg = config.load_config()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return cfg


def _plain_yaml(text):
    """Strip the timeloopfe tags so PyYAML can load the structure."""
    return re.sub(r"!(Container|Component|Nothing|Parallel|Hierarchical)\b", "", text)


def _duplicate_keys(text):
    """Mapping keys repeated within one block -- PyYAML silently keeps the last."""
    class Loader(yaml.SafeLoader):
        pass
    dups = []

    def construct_mapping(loader, node, deep=False):
        seen = set()
        for k, _ in node.value:
            key = loader.construct_object(k, deep=deep)
            if key in seen:
                dups.append((key, k.start_mark.line + 1))
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep)
    Loader.construct_mapping = construct_mapping
    yaml.load(_plain_yaml(text), Loader=Loader)
    return dups


# ------------------------------------------------------------------ parsing
def test_networks_section_is_parsed_and_reconciles():
    from ..toolchain.stats import classify, parse_stats
    rows = [r for r in parse_stats(FIXTURE, "L") if r["energy_pJ"] is not None]
    cats = {}
    for r in rows:
        cats.setdefault(classify(r["level"], r["instances"]), 0.0)
        cats[classify(r["level"], r["instances"])] += r["energy_pJ"]
    noc = [r for r in rows if r["level"].startswith("NoC")]
    assert len(noc) == 3, noc                         # Weights / Inputs / Outputs
    assert abs(sum(r["energy_pJ"] for r in noc) - 125925.04) < 0.05
    # Timeloop's summary: 98.38 fJ/compute x 1,280,000 computes
    assert abs(cats["NoC"] - 98.38e-3 * 1_280_000) < 20
    # and the grand total is Timeloop's `Energy: 85.20 uJ`
    assert abs(sum(cats.values()) / 1e6 - 85.20) < 0.01, sum(cats.values())
    assert {r["dataspace"] for r in noc} == {"Weights", "Inputs", "Outputs"}
    assert noc[0]["reads"] == 1_280_000.0             # ingresses ride in `reads`


def test_network_text_does_not_leak_into_dram():
    """The Networks section sits inside the DRAM `===` chunk; it must not be read as DRAM."""
    from ..toolchain.stats import classify, parse_stats
    rows = [r for r in parse_stats(FIXTURE, "L") if r["energy_pJ"] is not None]
    dram = [r for r in rows if classify(r["level"], r["instances"]) == "DRAM"]
    assert len(dram) == 3, [(r["level"], r["dataspace"]) for r in dram]
    assert abs(sum(r["energy_pJ"] for r in dram) - 64114.00e-3 * 1_280_000) < 20


def test_classify_puts_networks_first():
    from ..toolchain.stats import classify
    assert classify("NoC: DRAM <==> ifmap_glb") == "NoC"      # contains "dram"
    assert classify("NoC: psum_spad <==> mac") == "NoC"        # contains "mac"
    assert classify("DRAM") == "DRAM"
    assert classify("ifmap_spad", instances=168) == "Local (spads/RF)"


def test_split_networks_without_section_is_identity():
    from ..toolchain.stats import _split_networks
    text = "=== a ===\n x\n=== b ===\n y\n"
    assert _split_networks(text) == (text, "")


# ---------------------------------------------------------------- injection
def test_injection_is_valid_yaml_on_every_arch_and_moves_the_fingerprint():
    from ..arch import fingerprint
    from ..arch import layout
    from ..arch import load
    from ..arch import patch
    from ..arch import validate
    cfg = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1")
    off = dataclasses.replace(cfg, noc_enabled=False)
    for arch in ARCHS:
        on, plain = patch._patched_text(arch, cfg, quiet=True), patch._patched_text(arch, off, quiet=True)
        assert on != plain, arch
        assert not _duplicate_keys(on), (arch, _duplicate_keys(on))
        present = load.spatial_containers(on)
        levels = load.noc_levels(arch)
        charged = present if levels is None else [n for n in present if n in levels]
        n_spatial = len(charged)
        assert 1 <= n_spatial <= len(present), (arch, present, levels)
        assert on.count("# NoC: archs/_shared/noc.yaml") == n_spatial, arch
        assert on.count("network_word_bits: 8") == n_spatial, arch
        # the coefficients the contract says this design is charged
        terms = load.noc_terms(arch, cfg)
        assert terms["wire"] == 0.12, (arch, terms["wire"])   # Keckler/Dally Table 1
        per_level = {n: (terms["levels"][n] if terms["levels"] else (terms["router"], terms["ingress"]))
                     for n in charged}
        r_vals = [float(v) for v in re.findall(r"router_energy:\s*([\d.]+)", on)]
        g_vals = [float(v) for v in re.findall(r"energy-per-ingress:\s*([\d.]+)", on)]
        # charged ONLY on the NoC containers, each with ITS level's terms ...
        assert sum(1 for v in r_vals if v) == sum(1 for r, _ in per_level.values() if r), arch
        assert sum(1 for v in g_vals if v) == sum(1 for _, g in per_level.values() if g), arch
        # ... and explicitly zeroed on every storage component, so a child does
        # not inherit a switching charge for a register-to-ALU path.
        n_comp = sum(1 for b in load._node_blocks(on.split("\n"))
                     if b["kind"] == "Component" and b["attr"] is not None)
        n_storage = len(re.findall(r"^\s*depth:\s*\d", on, re.M))
        assert n_comp > n_storage, arch                 # the MAC is a component too
        n_datapath = len(present) - n_spatial           # nested lanes get explicit zeros
        # ... and, since 2026-09-08, a charged level whose own term is zero is
        # written as an explicit zero too (a nested NoC level would otherwise
        # inherit its parent's router: FINDINGS open defect 8.1).
        zero_r = sum(1 for r, _ in per_level.values() if not r)
        zero_g = sum(1 for _, g in per_level.values() if not g)
        assert sum(1 for v in r_vals if not v) == n_comp + n_datapath + zero_r, (arch, n_comp, n_datapath, r_vals)
        assert sum(1 for v in g_vals if not v) == n_comp + n_datapath + zero_g, (arch, n_comp, n_datapath, g_vals)
        # every charged level carries BOTH switching terms, whatever their value
        for b in load._node_blocks(on.split("\n")):
            if b["kind"] == "Container" and b["spatial"] is not None and b["name"] in charged:
                blk = "\n".join(on.split("\n")[b["spatial"] + 1: b["spatial"] + 8])
                assert "router_energy:" in blk and "energy-per-ingress:" in blk, (arch, b["name"], blk)
        # loop depth (and so victory scaling) must not move
        assert layout.loop_levels(on) == layout.loop_levels(plain), arch
        assert fingerprint.arch_fingerprint(arch, cfg) != fingerprint.arch_fingerprint(arch, off), arch


def test_contract_charges_routers_only_where_the_design_has_them():
    from ..arch import fingerprint
    from ..arch import layout
    from ..arch import load
    from ..arch import patch
    from ..arch import validate
    cfg = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1")
    routers = {a: load.noc_params(a, cfg)[1] for a in ARCHS}
    ingress = {a: load.noc_params(a, cfg)[2] for a in ARCHS}
    for a in ("eyeriss_v2_like", "eyeriss_v2_like_wglb", "simba_like"):
        assert routers[a] > 0 and ingress[a] == 0, a
    # v2: one shared per-flit router energy, three operands per 24b flit, and
    # only the router-cluster level is a router hop; the PE row is wiring.
    t = load.noc_terms("eyeriss_v2_like", cfg)
    assert abs(t["levels"]["PE_cluster"][0] - 0.25 / 3) < 1e-9, t
    assert t["levels"]["PE_cluster"][1] == 0.0, t              # switching is in the routers
    assert t["levels"]["PE"] == (0.0, 0.0), t                  # no second router; latch 0 by default (2026-09-09)
    assert load.noc_terms("simba_like", cfg)["levels"]["PE"][0] == 0.25   # no published packing
    for a in ("eyeriss_like", "eyeriss_like_wglb", "simple_weight_stationary",
              "simple_output_stationary", "simple_input_stationary"):
        assert routers[a] == 0 and ingress[a] > 0, a
    # the generic baselines share ONE explicitly defined model
    assert len({(routers[a], ingress[a]) for a in
                ("simple_weight_stationary", "simple_output_stationary",
                 "simple_input_stationary")}) == 1


def test_datapath_spatial_levels_are_not_charged():
    """v2's SIMD pair and Simba's vector-MAC lanes are spatial to Timeloop, datapath to the design."""
    from ..arch import fingerprint
    from ..arch import layout
    from ..arch import load
    from ..arch import patch
    from ..arch import validate
    cfg = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1")
    expect = {"eyeriss_v2_like": (["PE_cluster", "PE"], ["SIMD"]),
              "simba_like": (["PE"], ["distributed_buffers", "reg_mac"]),
              "eyeriss_like": (["PE_column", "PE"], []),
              "simple_weight_stationary": (["PE"], [])}
    for arch, (noc, datapath) in expect.items():
        text = patch._patched_text(arch, cfg, quiet=True)
        blocks = [b for b in load._node_blocks(text.split("\n"))
                  if b["kind"] == "Container" and b["spatial"] is not None]
        assert [b["name"] for b in blocks] == noc + datapath or \
            set(b["name"] for b in blocks) == set(noc + datapath), (arch, [b["name"] for b in blocks])
        lines = text.split("\n")
        for b in blocks:
            nxt = lines[b["spatial"] + 1]
            if b["name"] in noc:
                assert "# NoC: archs/_shared/noc.yaml" in nxt, (arch, b["name"], nxt)
            else:
                # nested inside a NoC container, so it must OVERRIDE, not omit
                assert "NoC-datapath" in nxt, (arch, b["name"], nxt)
                blk = "\n".join(lines[b["spatial"] + 1: b["spatial"] + 5])
                for k in ("wire_energy: 0.0", "router_energy: 0.0", "energy-per-ingress: 0.0"):
                    assert k in blk, (arch, b["name"], k, blk)


def test_validate_rejects_a_misspelt_noc_level():
    from ..arch import fingerprint
    from ..arch import layout
    from ..arch import load
    from ..arch import patch
    from ..arch import validate
    cfg = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1")
    saved = load._NOC["architectures"]["eyeriss_like"]["noc_levels"]
    load._NOC["architectures"]["eyeriss_like"]["noc_levels"] = ["PE_colum", "PE"]
    try:
        report = validate.validate_arch("eyeriss_like", cfg)
    finally:
        load._NOC["architectures"]["eyeriss_like"]["noc_levels"] = saved
    assert any("PE_colum" in v for v in report["violations"]), report["violations"]
    assert not any("PE_colum" in v for v in validate.validate_arch("eyeriss_like", cfg)["violations"])


def test_calibrated_ingress_follows_the_wire_constant():
    """v1's inter-PE transfer stays at its published 2x MAC whatever the wire constant."""
    from ..arch import fingerprint
    from ..arch import layout
    from ..arch import load
    from ..arch import patch
    from ..arch import validate
    cfg = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1")
    ref = 1.13555  # 2 x MAC for 16b, i.e. 1 x MAC for this study's 8b operand
    for wire in (0.4, 0.12, 0.0):
        c = _cfg(ECC_NOC="1", ECC_NOC_WIRE_PJ_PER_BIT_MM=str(wire))
        g = load.calibrated_ingress(c)
        wire_term = 2.5 * 8 * 0.09181 * wire
        assert abs(g + wire_term - ref) < 1e-6, (wire, g)
        assert abs(load.noc_params("eyeriss_like", c)[2] - g) < 1e-9
    assert abs(load.calibrated_ingress(cfg) - 0.9152) < 5e-4          # the documented value at 0.12
    assert abs(load.calibrated_ingress(_cfg(ECC_NOC="1", ECC_NOC_WIRE_PJ_PER_BIT_MM="0.4")) - 0.4011) < 5e-4
    r = _cfg(ECC_NOC="1", ECC_NOC_ROUTER_PJ="0.1")
    assert abs(load.noc_terms("eyeriss_v2_like", r)["levels"]["PE_cluster"][0] - 0.1 / 3) < 1e-9
    assert "nocr0.1" in r.arch_variant_slug


def test_unknown_arch_is_a_hard_error_not_a_free_noc():
    """A design with no `noc.yaml` entry must STOP, never be charged zero NoC.

    IT FAILS THROUGH THE LAST LINE -- `raise AssertionError` when no `SystemExit`
    arrives. ProjectRestructure section 7.1 counted this as having "zero
    assertions"; its AST audit does not see a raised `AssertionError`. Measured
    2026-09-13: one test in the suite has no failure mechanism, not three.
    """
    from ..arch import fingerprint
    from ..arch import layout
    from ..arch import load
    from ..arch import patch
    from ..arch import validate
    cfg = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1")
    try:
        load.noc_params("not_a_design", cfg)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit for an arch with no noc.yaml entry")


def test_noc_is_in_the_slug_and_scale_override_changes_it():
    on = _cfg(ECC_NOC="1")
    off = _cfg(ECC_NOC="0")
    scaled = _cfg(ECC_NOC="1", ECC_NOC_SCALE="2")
    assert "noc" in on.arch_variant_slug.split("__")
    assert "noc" not in off.arch_variant_slug
    assert "nocx2" in scaled.arch_variant_slug
    assert on.fingerprint() != off.fingerprint() != scaled.fingerprint()


def test_noc_category_and_recon_scaling():
    from ..study.energy import PHYS_CATS, onchip_cats, plot_cats
    cfg = _cfg(ECC_NOC="1")
    assert "NoC" in PHYS_CATS and "Local (spads/RF)" in PHYS_CATS
    assert "Local (spads/RF/NoC)" not in PHYS_CATS
    assert "NoC" in onchip_cats(cfg)                  # decision 2026-09-07
    cats = plot_cats(cfg)
    assert cats.index("NoC") == cats.index("Compute") - 1   # stacks just under Compute


# ------------------------------------------------ 2026-09-08 revision
def test_v2_pe_latch_is_bracketed_and_the_knob_moves_the_cache():
    """Default 0 (2026-09-09); ECC_NOC_PE_LATCH_PJ runs the bracket (0.5, v1's calibrated 0.9152)."""
    from ..arch import fingerprint
    from ..arch import layout
    from ..arch import load
    from ..arch import patch
    from ..arch import validate
    cfg = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1")
    assert load.pe_latch_pj(cfg) == 0.0
    assert load.noc_terms("eyeriss_v2_like", cfg)["levels"]["PE"][1] == 0.0
    assert load.noc_terms("eyeriss_v2_like_wglb", cfg)["levels"]["PE"][1] == 0.0
    # nobody else pays it: v1 and the generic designs are on the calibrated
    # ingress, simba on routers only
    for a in ("eyeriss_like", "simple_weight_stationary", "simba_like"):
        text = patch._patched_text(a, cfg, quiet=True)
        assert "energy-per-ingress: 0.5\n" not in text, a
    lo = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1", ECC_NOC_PE_LATCH_PJ="0.5")
    hi = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1", ECC_NOC_PE_LATCH_PJ="0.9152")
    assert load.noc_terms("eyeriss_v2_like", lo)["levels"]["PE"][1] == 0.5
    assert abs(load.noc_terms("eyeriss_v2_like", hi)["levels"]["PE"][1] - 0.9152) < 1e-12
    assert "nocl0.9152" in hi.arch_variant_slug and "nocl0.5" in lo.arch_variant_slug
    assert "nocl" not in cfg.arch_variant_slug
    fps = {fingerprint.arch_fingerprint("eyeriss_v2_like", c) for c in (cfg, lo, hi)}
    assert len(fps) == 3, fps
    # 8x2 cluster mesh (2026-09-09): both dims declared, no forced split
    text = patch._patched_text("eyeriss_v2_like", cfg, quiet=True)
    assert "spatial: {meshX: 8, meshY: 2}" in text and "spatial: {meshX: 16}" not in text
    # the upper bound really is v1's number
    assert abs(load.calibrated_ingress(cfg) - 0.9152) < 5e-4


def test_v2_injection_zeros_the_inner_router_and_declares_both_pitches():
    """The 8.1 defect: PE row inherited the cluster router. Now explicit, with tile widths."""
    from ..arch import fingerprint
    from ..arch import layout
    from ..arch import load
    from ..arch import patch
    from ..arch import validate
    cfg = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1")
    text = patch._patched_text("eyeriss_v2_like", cfg, quiet=True)
    lines = text.split("\n")
    blocks = {b["name"]: b for b in load._node_blocks(lines)
              if b["kind"] == "Container" and b["spatial"] is not None}
    clu = "\n".join(lines[blocks["PE_cluster"]["spatial"] + 1: blocks["PE_cluster"]["spatial"] + 8])
    pe = "\n".join(lines[blocks["PE"]["spatial"] + 1: blocks["PE"]["spatial"] + 8])
    assert "router_energy: 0.08333" in clu and "energy-per-ingress: 0.0" in clu, clu
    assert "tile_width: 430.0" in clu, clu
    assert "router_energy: 0.0\n" in pe + "\n" and "energy-per-ingress: 0.0" in pe, pe
    assert "tile_width: 108.0" in pe, pe
    assert load.noc_band_levels("eyeriss_v2_like") == ["PE_cluster"]
    assert load.noc_band_levels("eyeriss_like") == []
    # SIMD is datapath: wire, router, ingress zero and no pitch of its own
    simd = "\n".join(lines[blocks["SIMD"]["spatial"] + 1: blocks["SIMD"]["spatial"] + 5])
    assert "wire_energy: 0.0" in simd and "tile_width" not in simd, simd
    assert not _duplicate_keys(text)
    # nobody else declares a pitch: their hop length stays Timeloop's area-derived one
    for a in ARCHS:
        if not a.startswith("eyeriss_v2_like"):
            assert "tile_width:" not in patch._patched_text(a, cfg, quiet=True), a
    # and the contract refuses a partial declaration (the inner level would
    # inherit the outer pitch)
    saved = load._NOC["architectures"]["eyeriss_v2_like"]["noc_levels"]["PE"]
    load._NOC["architectures"]["eyeriss_v2_like"]["noc_levels"]["PE"] = {
        k: v for k, v in saved.items() if k != "tile_width_um"}
    try:
        rep = validate.validate_arch("eyeriss_v2_like", cfg)
    finally:
        load._NOC["architectures"]["eyeriss_v2_like"]["noc_levels"]["PE"] = saved
    assert any("tile_width_um" in v for v in rep["violations"]), rep["violations"]
    assert not any("tile_width_um" in v for v in validate.validate_arch("eyeriss_v2_like", cfg)["violations"])


def test_declared_pitches_match_the_cached_art():
    """108 um and 430 um are DERIVED numbers; re-derive them from any cached ART."""
    import glob
    import math
    from ..arch import fingerprint
    from ..arch import layout
    from ..arch import load
    from ..arch import patch
    from ..arch import validate
    from ..paths import WORK
    # THE ART HAS TO BE THE ONE THIS DESIGN'S PITCHES WERE DERIVED FROM, and
    # until 2026-09-13 this took the NEWEST cached ART instead. Any run that
    # reshapes the weight levels -- the capacity sweep (`wcap`), the depth ladder
    # (`wdepth`), THE WIDTH TABLE (`wt`/`ww`/`wdw`) -- builds a bigger or smaller
    # PE and leaves a newer ART behind, so an ordinary experiment on an unrelated
    # knob turned this into a red test comparing a declared pitch against a
    # different chip. Measured on 2026-09-13: `wcap0.4038` gives 105.432 um and
    # `wcap32` gives 199.520 um against the declared 108, while the undilated
    # ARTs (59 of them) agree at 109.094 um.
    WEIGHT_GEOMETRY = ("wcap", "wdepth", "ww", "wdw", "wt")
    arts = [f for f in glob.glob(str(WORK / "outputs" / "eyeriss_v2_like"
                                     / "*" / "*" / "*" / "timeloop-mapper.ART_summary.yaml"))
            if not any(("__" + k) in pathlib.Path(f).parts[-4] for k in WEIGHT_GEOMETRY)]
    arts.sort(key=os.path.getmtime)
    if not arts:
        pytest.skip("no cached eyeriss_v2_like ART at an undilated weight geometry "
                    "on this machine -- the declared pitches cannot be re-derived")
    art = yaml.safe_load(pathlib.Path(arts[-1]).read_text())   # newest UNDILATED ART
    area = {}
    # the summary is {"ART_summary": {"version":..., "table_summary": [{name, area, ...}]}}
    table = art["ART_summary"]["table_summary"]
    for e in table:
        name = re.sub(r"\[.*?\]", "", e["name"]).split(".")[-1]   # strip [1..192] BEFORE splitting on dots
        area[name] = float(e["area"])
    pe = area["ifmap_spad"] + area["weights_spad"] + area["psum_spad"] + 2 * area["mac"]
    glb_cluster = (area["iact_glb"] + area["psum_glb"]) / 16
    pe_pitch = math.sqrt(pe)
    cluster_pitch = math.sqrt((12 * pe + glb_cluster) / (1 - 0.026))
    lv = load.load_noc()["architectures"]["eyeriss_v2_like"]["noc_levels"]
    assert abs(pe_pitch - lv["PE"]["tile_width_um"]) / pe_pitch < 0.01, (pe_pitch, lv["PE"])
    assert abs(cluster_pitch - lv["PE_cluster"]["tile_width_um"]) / cluster_pitch < 0.01, \
        (cluster_pitch, lv["PE_cluster"])
    # and Timeloop's own derived pitch is the cluster alone, as documented
    assert abs(math.sqrt(12 * pe) - 374.8) < 2.0, math.sqrt(12 * pe)


def test_component_library_is_in_the_fingerprint():
    """Editing archs/_shared/components changes every ERT; the cache must move."""
    from ..arch import fingerprint
    from ..arch import layout
    from ..arch import load
    from ..arch import patch
    from ..arch import validate
    cfg = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1")
    before = fingerprint.arch_fingerprint("eyeriss_like", cfg)
    real = fingerprint.components_digest
    fingerprint.components_digest = lambda: "deadbeefcafe"
    try:
        after = fingerprint.arch_fingerprint("eyeriss_like", cfg)
    finally:
        fingerprint.components_digest = real
    assert before != after
    assert fingerprint.arch_fingerprint("eyeriss_like", cfg) == before
    assert len(real()) == 12 and real() != "deadbeefcafe"


def test_register_writes_are_costed_everywhere():
    """Aladdin's register table has write = 0; every register in the study now pays a read for it."""
    import glob
    from ..paths import ARCH_COMPONENTS, ARCH_SRC
    rfd = yaml.safe_load((ARCH_COMPONENTS / "regfile_decoded.yaml").read_text())
    rw = yaml.safe_load((ARCH_COMPONENTS / "register_rw.yaml").read_text())
    for doc, cls in ((rfd, "regfile_decoded"), (rw, "register_rw")):
        c = next(c for c in doc["compound_components"]["classes"] if c["name"] == cls)
        acts = {a["name"]: a for a in c["actions"]}
        for an in ("write", "update"):
            sub = acts[an]["subcomponents"][0]
            assert sub["name"].startswith("storage[1..width]"), (cls, an, sub)
            assert sub["actions"] == [{"name": "read"}], (cls, an, sub)
    for f in glob.glob(str(ARCH_SRC / "simple_*" / "arch_paper.yaml")):
        text = pathlib.Path(f).read_text()
        assert "subclass: aladdin_register" not in text, f
        assert text.count("subclass: register_rw") == 3, f


def test_evaluator_only_terms_are_parsed_and_charged():
    """Spatial reductions (adder + one psum-wide hop) and the psum word width, from the fixture."""
    from ..toolchain import noc_post
    from ..toolchain.stats import parse_stats
    cfg = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1")
    rows = parse_stats(FIXTURE, "L")
    net = [r for r in rows if r["level"].startswith("NoC")]
    out = next(r for r in net if r["dataspace"] == "Outputs")
    # the parser carries Timeloop's counts through unchanged
    assert out["reads"] == 2000.0 and out["hops"] == 2.5 and out["net_instances"] == 1
    assert abs(out["per_hop_pJ"] - 0.03672) < 1e-9 and out["spatial_reductions"] == 0
    # eyeriss_like accumulates at 16b, the network billed 8b of wire: +1x the wire term
    c = noc_post.coefficients("eyeriss_like", cfg)
    assert c["enabled"] and c["outputs_word_bits"] == 16 and c["network_word_bits"] == 8
    aug = noc_post.augment(rows, "eyeriss_like", cfg)
    added = [r for r in aug if r["level"].endswith("]")]
    assert [r["level"].split(" [")[-1] for r in added] == ["psum word width]"], added
    assert abs(added[0]["energy_pJ"] - 2000 * 2.5 * 0.03672 * 1) < 1e-6, added[0]
    assert added[0]["energy_pJ"] == out["energy_pJ"] * 1.0 or abs(added[0]["energy_pJ"] - 183.62) < 0.05
    # pure Timeloop rows are untouched and the total moved by exactly that row
    assert abs(sum(r["energy_pJ"] for r in aug) - sum(r["energy_pJ"] for r in rows)
               - added[0]["energy_pJ"]) < 1e-6
    # a reduction: one 16-bit add (0.0065625 x 16) + one hop of 16-bit wire, x instances
    row = dict(level="NoC: x <==> y", dataspace="Outputs", layer="L", reads=10.0, hops=2.0,
               per_hop_pJ=0.08, net_instances=4, spatial_reductions=1000.0, energy_pJ=1.0)
    red = [r for r in noc_post.augment([row], "eyeriss_like", cfg) if "reduction" in r["level"]][0]
    expect = 1000 * 4 * (0.0065625 * 16 + 1 * 16 * 0.08 / 8)
    assert abs(red["energy_pJ"] - expect) < 1e-9, (red, expect)
    # v2 accumulates at 20b
    c2 = noc_post.coefficients("eyeriss_v2_like", cfg)
    assert c2["outputs_word_bits"] == 20
    # off with the NoC, and the stamp follows the coefficients
    off = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="0")
    assert noc_post.augment(rows, "eyeriss_like", off) == rows
    assert noc_post.stamp("eyeriss_like", cfg) != noc_post.stamp("eyeriss_v2_like", cfg)
    assert noc_post.stamp("eyeriss_like", cfg) != noc_post.stamp("eyeriss_like", off)


def test_raw_record_is_stale_when_the_evaluator_terms_change():
    import tempfile
    from ..study import energy as E
    from ..toolchain import noc_post
    from ..paths import Results
    cfg = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1")
    import pandas as pd
    cats = E.plot_cats(cfg)
    ser = pd.Series({c: 1.0 for c in cats})
    with tempfile.TemporaryDirectory() as d:
        res = Results(dataclasses.replace(cfg, results_dir=d))
        res.prepare()
        raw = E.Raw(ser, ser, ser, 1.0, 1.0, 1, 0, 1, [], [],
                    noc_post=noc_post.stamp("eyeriss_like", cfg), cycles=1)
        E.save_raw(res, "eyeriss_like", "m", raw, "v", "fp")
        assert E.load_raw(res, cfg, "eyeriss_like", "m", "v", "fp") is not None
        # prompt_6 RULE 3: a record with no cycle count cannot be charged the
        # encoder's idle term and is re-gathered
        raw.cycles = None
        E.save_raw(res, "eyeriss_like", "m", raw, "v", "fp")
        assert E.load_raw(res, cfg, "eyeriss_like", "m", "v", "fp") is None
        raw.cycles = 1
        E.save_raw(res, "eyeriss_like", "m", raw, "v", "fp")
        assert E.load_raw(res, cfg, "eyeriss_like", "m", "v", "fp") is not None
        other = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1", ECC_NOC_SCALE="2")
        assert E.load_raw(res, other, "eyeriss_like", "m", "v", "fp") is None   # stale
        raw.noc_post = None                                                     # pre-2026-09-08 record
        E.save_raw(res, "eyeriss_like", "m", raw, "v", "fp")
        assert E.load_raw(res, cfg, "eyeriss_like", "m", "v", "fp") is None
        # a record aggregated while shapes were still mapping is stale too
        # (2026-09-09: a one-layer record was served as the whole model)
        raw.noc_post = noc_post.stamp("eyeriss_like", cfg)
        # `physical` is REQUIRED on an `ok` layer since prompt_7 Phase A: the
        # roofline and the standby charge both read it, so `load_raw` re-gathers
        # a record without it. Omitted here until 2026-09-13, and the test
        # passed only because an earlier module in the same pytest process had
        # wiped every ECC_* variable, leaving ECC_STATIC_ENERGY and
        # ECC_LATENCY_MODEL off and that rule dormant. With the environment
        # restored per test (conftest.py) the rule fires, as it should -- so the
        # record this test builds has to be a record the evaluator would write.
        phys = {"levels": [{"level": "filter_glb", "instances": 1,
                            "size": 256, "word_bits": 64, "arithmetic": False}]}
        raw.per_layer = [{"layer": "a", "shape": "s1", "status": "ok",
                          "repeat_count": 1, "physical": phys},
                         {"layer": "b", "shape": "s2", "status": "unmapped"}]
        E.save_raw(res, "eyeriss_like", "m", raw, "v", "fp")
        assert E.load_raw(res, cfg, "eyeriss_like", "m", "v", "fp") is None
        raw.per_layer[1]["status"] = "ok"
        raw.per_layer[1]["physical"] = phys
        raw.per_layer[1]["repeat_count"] = 1
        E.save_raw(res, "eyeriss_like", "m", raw, "v", "fp")
        assert E.load_raw(res, cfg, "eyeriss_like", "m", "v", "fp") is not None


def main():
    print("eccenergy NoC model tests")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
