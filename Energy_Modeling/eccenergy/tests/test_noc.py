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
    from ..timeloop import classify, parse_stats
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
    from ..timeloop import classify, parse_stats
    rows = [r for r in parse_stats(FIXTURE, "L") if r["energy_pJ"] is not None]
    dram = [r for r in rows if classify(r["level"], r["instances"]) == "DRAM"]
    assert len(dram) == 3, [(r["level"], r["dataspace"]) for r in dram]
    assert abs(sum(r["energy_pJ"] for r in dram) - 64114.00e-3 * 1_280_000) < 20


def test_classify_puts_networks_first():
    from ..timeloop import classify
    assert classify("NoC: DRAM <==> ifmap_glb") == "NoC"      # contains "dram"
    assert classify("NoC: psum_spad <==> mac") == "NoC"        # contains "mac"
    assert classify("DRAM") == "DRAM"
    assert classify("ifmap_spad", instances=168) == "Local (spads/RF)"


def test_split_networks_without_section_is_identity():
    from ..timeloop import _split_networks
    text = "=== a ===\n x\n=== b ===\n y\n"
    assert _split_networks(text) == (text, "")


# ---------------------------------------------------------------- injection
def test_injection_is_valid_yaml_on_every_arch_and_moves_the_fingerprint():
    from .. import archs as A
    cfg = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1")
    off = dataclasses.replace(cfg, noc_enabled=False)
    for arch in ARCHS:
        on, plain = A._patched_text(arch, cfg, quiet=True), A._patched_text(arch, off, quiet=True)
        assert on != plain, arch
        assert not _duplicate_keys(on), (arch, _duplicate_keys(on))
        present = A.spatial_containers(on)
        levels = A.noc_levels(arch)
        charged = present if levels is None else [n for n in present if n in levels]
        n_spatial = len(charged)
        assert 1 <= n_spatial <= len(present), (arch, present, levels)
        assert on.count("# NoC: archs/_shared/noc.yaml") == n_spatial, arch
        assert on.count("network_word_bits: 8") == n_spatial, arch
        # the coefficients the contract says this design is charged
        terms = A.noc_terms(arch, cfg)
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
        n_comp = sum(1 for b in A._node_blocks(on.split("\n"))
                     if b["kind"] == "Component" and b["attr"] is not None)
        n_storage = len(re.findall(r"^\s*depth:\s*\d", on, re.M))
        assert n_comp > n_storage, arch                 # the MAC is a component too
        n_datapath = len(present) - n_spatial           # nested lanes get explicit zeros
        assert sum(1 for v in r_vals if not v) == n_comp + n_datapath, (arch, n_comp, n_datapath, r_vals)
        assert sum(1 for v in g_vals if not v) == n_comp + n_datapath, (arch, n_comp, n_datapath, g_vals)
        # loop depth (and so victory scaling) must not move
        assert A.loop_levels(on) == A.loop_levels(plain), arch
        assert A.arch_fingerprint(arch, cfg) != A.arch_fingerprint(arch, off), arch


def test_contract_charges_routers_only_where_the_design_has_them():
    from .. import archs as A
    cfg = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1")
    routers = {a: A.noc_params(a, cfg)[1] for a in ARCHS}
    ingress = {a: A.noc_params(a, cfg)[2] for a in ARCHS}
    for a in ("eyeriss_v2_like", "eyeriss_v2_like_wglb", "simba_like"):
        assert routers[a] > 0 and ingress[a] == 0, a
    # v2: one shared per-flit router energy, three operands per 24b flit, and
    # only the router-cluster level is a router hop; the PE row is wiring.
    t = A.noc_terms("eyeriss_v2_like", cfg)
    assert abs(t["levels"]["PE_cluster"][0] - 0.25 / 3) < 1e-9, t
    assert t["levels"]["PE"] == (0.0, 0.0), t
    assert A.noc_terms("simba_like", cfg)["levels"]["PE"][0] == 0.25   # no published packing
    for a in ("eyeriss_like", "eyeriss_like_wglb", "simple_weight_stationary",
              "simple_output_stationary", "simple_input_stationary"):
        assert routers[a] == 0 and ingress[a] > 0, a
    # the generic baselines share ONE explicitly defined model
    assert len({(routers[a], ingress[a]) for a in
                ("simple_weight_stationary", "simple_output_stationary",
                 "simple_input_stationary")}) == 1


def test_datapath_spatial_levels_are_not_charged():
    """v2's SIMD pair and Simba's vector-MAC lanes are spatial to Timeloop, datapath to the design."""
    from .. import archs as A
    cfg = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1")
    expect = {"eyeriss_v2_like": (["PE_cluster", "PE"], ["SIMD"]),
              "simba_like": (["PE"], ["distributed_buffers", "reg_mac"]),
              "eyeriss_like": (["PE_column", "PE"], []),
              "simple_weight_stationary": (["PE"], [])}
    for arch, (noc, datapath) in expect.items():
        text = A._patched_text(arch, cfg, quiet=True)
        blocks = [b for b in A._node_blocks(text.split("\n"))
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
    from .. import archs as A
    cfg = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1")
    saved = A._NOC["architectures"]["eyeriss_like"]["noc_levels"]
    A._NOC["architectures"]["eyeriss_like"]["noc_levels"] = ["PE_colum", "PE"]
    try:
        report = A.validate_arch("eyeriss_like", cfg)
    finally:
        A._NOC["architectures"]["eyeriss_like"]["noc_levels"] = saved
    assert any("PE_colum" in v for v in report["violations"]), report["violations"]
    assert not any("PE_colum" in v for v in A.validate_arch("eyeriss_like", cfg)["violations"])


def test_calibrated_ingress_follows_the_wire_constant():
    """v1's inter-PE transfer stays at its published 2x MAC whatever the wire constant."""
    from .. import archs as A
    cfg = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1")
    ref = 1.13555  # 2 x MAC for 16b, i.e. 1 x MAC for this study's 8b operand
    for wire in (0.4, 0.12, 0.0):
        c = _cfg(ECC_NOC="1", ECC_NOC_WIRE_PJ_PER_BIT_MM=str(wire))
        g = A.calibrated_ingress(c)
        wire_term = 2.5 * 8 * 0.09181 * wire
        assert abs(g + wire_term - ref) < 1e-6, (wire, g)
        assert abs(A.noc_params("eyeriss_like", c)[2] - g) < 1e-9
    assert abs(A.calibrated_ingress(cfg) - 0.9152) < 5e-4          # the documented value at 0.12
    assert abs(A.calibrated_ingress(_cfg(ECC_NOC="1", ECC_NOC_WIRE_PJ_PER_BIT_MM="0.4")) - 0.4011) < 5e-4
    r = _cfg(ECC_NOC="1", ECC_NOC_ROUTER_PJ="0.1")
    assert abs(A.noc_terms("eyeriss_v2_like", r)["levels"]["PE_cluster"][0] - 0.1 / 3) < 1e-9
    assert "nocr0.1" in r.arch_variant_slug


def test_unknown_arch_is_a_hard_error_not_a_free_noc():
    from .. import archs as A
    cfg = _cfg(ECC_ARCH_FIDELITY="paper", ECC_NOC="1")
    try:
        A.noc_params("not_a_design", cfg)
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
    from ..energy import PHYS_CATS, onchip_cats, plot_cats
    cfg = _cfg(ECC_NOC="1")
    assert "NoC" in PHYS_CATS and "Local (spads/RF)" in PHYS_CATS
    assert "Local (spads/RF/NoC)" not in PHYS_CATS
    assert "NoC" in onchip_cats(cfg)                  # decision 2026-09-07
    cats = plot_cats(cfg)
    assert cats.index("NoC") == cats.index("Compute") - 1   # stacks just under Compute


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
