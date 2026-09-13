"""Architecture handling: install local designs, patch, and audit.

Three jobs:

* `install_local_archs()` copies the repo-authored designs in `archs/` into the
  cloned exercises tree, so a wipe of `ecc_energy_study/` cannot lose them.

* `patched_arch_path()` produces the arch.yaml the mapper actually runs. It
  always applies the DRAM-depth patch the historical scripts applied, and
  optionally forces the technology node and the storage datawidth -- the two
  knobs that decide whether a cross-architecture comparison is fair.

* `audit()` reads an arch.yaml and reports the handful of declarations that
  dominate a weight-energy study: technology node, DRAM word width, whether the
  global buffer keeps or bypasses Weights, and how many weights fit on chip.
  This is what `ECC_EXPERIMENT=diagnose` prints.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil

import yaml

from . import code_widths
from .paths import (ARCH_COMPONENTS, ARCH_NOC, ARCH_PROVENANCE, ARCH_SRC, ARCH_SRC_RESERVED,
                    ARCH_STANDARD, DESIGNS_DIR, WORK)


def install_local_archs(verbose=True):
    """Copy `archs/<name>/*` into the exercises example_designs tree."""
    if not ARCH_SRC.exists():
        return []
    installed = []
    for master in sorted(p for p in ARCH_SRC.iterdir() if p.is_dir()):
        if master.name in ARCH_SRC_RESERVED:
            continue          # archs/_shared holds the contract, not a design
        dst = DESIGNS_DIR / master.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in master.iterdir():
            if not f.is_file():
                continue
            target = dst / f.name
            if (not target.exists()) or f.stat().st_mtime > target.stat().st_mtime:
                shutil.copy2(f, target)
                installed.append(f"{master.name}/{f.name}")
                if verbose:
                    print(f"  [install] {master.name}/{f.name} -> example_designs/")
    return installed


# ------------------------------------------------- the standardization contract
_STANDARD = {}
_PROVENANCE = {}
_NOC = {}


def load_noc():
    """`archs/_shared/noc.yaml` -- the interconnect energy contract.

    Kept separate from `standard.yaml` because it is not an audit
    specification: every number in it is READ and injected into the arch the
    mapper sees. `standard.yaml` is checked against the YAMLs; this one
    changes them.
    """
    if not _NOC:
        if not ARCH_NOC.exists():
            raise SystemExit(
                f"missing {ARCH_NOC}.\n"
                "  -> that file supplies the only non-zero interconnect "
                "coefficients this study has; without it Timeloop's wire model "
                "is a stub that returns 0 and every NoC is free.")
        _NOC.update(yaml.safe_load(ARCH_NOC.read_text(encoding="utf-8")))
    return _NOC


def _noc_entry(arch):
    entry = (load_noc().get("architectures") or {}).get(arch)
    if entry is None:
        raise SystemExit(
            f"{arch} has no entry in {ARCH_NOC}.\n"
            "  -> add one (with its citation) rather than letting it map with "
            "a free interconnect while every other design pays for one.")
    return entry


def _shared_noc(cfg):
    """(shared block, wire pJ/b/mm, router pJ per FLIT) after env overrides."""
    sh = load_noc().get("shared", {})
    wire = (cfg.noc_wire_pj_per_bit_mm if cfg.noc_wire_pj_per_bit_mm is not None
            else float(sh["wire_pj_per_bit_mm"]))
    rflit = (cfg.noc_router_pj if cfg.noc_router_pj is not None
             else float(sh.get("router_pj_per_flit", 0.0)))
    return sh, wire, rflit


def pe_latch_pj(cfg):
    """Per-operand cost of taking an operand off a router port into a PE.

    noc.yaml `shared.pe_latch_pj`, or ECC_NOC_PE_LATCH_PJ. A bracketed
    assumption (0 .. v1's calibrated switching), not a measurement -- see the
    yaml for why it exists and how to run the two ends.
    """
    if getattr(cfg, "noc_pe_latch_pj", None) is not None:
        return float(cfg.noc_pe_latch_pj)
    return float(load_noc().get("shared", {}).get("pe_latch_pj", 0.0))


def calibrated_ingress(cfg, wire=None):
    """Eyeriss v1's switching term, derived so its inter-PE transfer stays 2x MAC.

    The anchor (noc.yaml `shared.calibration`) is the published cost of ONE
    inter-PE transfer, normalized to a MAC and to a 16b word. Scaled to this
    study's operand width and MAC, minus the wire this model already predicts
    for that transfer at v1's own PE pitch over its reference hop count, the
    remainder is the multicast-controller switching. Deriving it here rather
    than writing a number keeps the split consistent when the wire constant
    changes: a lower wire constant means MORE of the published transfer cost
    is switching, not a cheaper transfer.
    """
    sh, w, _ = _shared_noc(cfg)
    if wire is None:
        wire = w
    c = sh["calibration"]
    bits = cfg.weight_bits
    per_transfer = (float(c["reference_mac_pj"]) * float(c["published_inter_pe_x_mac"])
                    * bits / float(c["published_word_bits"]))
    wire_term = (float(c["reference_avg_hops"]) * bits
                 * float(c["reference_tile_width_um"]) / 1000.0 * wire)
    return max(per_transfer - wire_term, 0.0)


def noc_terms(arch, cfg):
    """Every interconnect coefficient this design is charged, resolved.

    Returns {"wire", "router", "ingress", "levels", "router_pj_per_flit",
    "operands_per_flit"}. `levels` is None (every spatial container, design
    defaults) or {container name: (router_pj, ingress_pj)} for the containers
    that ARE the NoC; the rest are datapath and get explicit zeros.

    Two symbolic values are resolved here so the yaml states the RULE and the
    code applies it:
      router_pj: per_flit    router_pj_per_flit / operands_per_flit -- one shared
                             router energy per flit, per-operand via the
                             design's published port packing (v2: 24b = 3 x 8b)
      ingress_pj: calibrated calibrated_ingress(), the Eyeriss v1 anchor
    """
    entry = _noc_entry(arch)
    sh, wire, rflit = _shared_noc(cfg)
    opf = float(entry.get("operands_per_flit", 1))

    def router(v):
        return rflit / opf if v == "per_flit" else float(v)

    def ingress(v):
        if v == "calibrated":
            return calibrated_ingress(cfg, wire)
        if v == "pe_latch":
            return pe_latch_pj(cfg)
        return float(v)

    d_router = router(entry.get("router_pj", 0.0))
    d_ingress = ingress(entry.get("ingress_pj", 0.0))
    lv = entry.get("noc_levels")
    tile_width = {}
    if lv is None:
        levels = None
    elif isinstance(lv, dict):
        levels = {str(n): (router((o or {}).get("router_pj", entry.get("router_pj", 0.0))),
                           ingress((o or {}).get("ingress_pj", entry.get("ingress_pj", 0.0))))
                  for n, o in lv.items()}
        # A pre-floorplanned hop length (um). Optional; Timeloop derives one
        # from the Accelergy area of the inner level when it is absent.
        tile_width = {str(n): float(o["tile_width_um"]) for n, o in lv.items()
                      if o and o.get("tile_width_um") is not None}
    else:
        levels = {str(n): (d_router, d_ingress) for n in lv}
    k = cfg.noc_scale
    return {"wire": wire * k, "router": d_router * k, "ingress": d_ingress * k,
            "levels": None if levels is None else {n: (r * k, g * k) for n, (r, g) in levels.items()},
            "tile_width_um": tile_width,
            "router_pj_per_flit": rflit, "operands_per_flit": opf}


def noc_params(arch, cfg):
    """(wire, router, ingress) at the design level -- see noc_terms() for per-level."""
    t = noc_terms(arch, cfg)
    return t["wire"], t["router"], t["ingress"]


def noc_levels(arch):
    """Names of the spatial containers that ARE this design's NoC, or None.

    None means "every spatial container" (the key is absent). Eyeriss v2's SIMD
    pair and Simba's vector-MAC lanes are spatial levels to Timeloop but
    datapath to the designs, and charging them router hops billed v2 750 fJ per
    MAC for wires that never leave the PE.
    """
    lv = _noc_entry(arch).get("noc_levels")
    return None if lv is None else [str(x) for x in lv]


def noc_band_levels(arch):
    """The spatial levels whose networks are what the design's paper calls its NoC.

    noc.yaml `paper_band_levels`. Empty when the entry declares none, in which
    case a published-band comparison uses the whole interconnect.
    """
    lv = _noc_entry(arch).get("paper_band_levels") or []
    return [str(x) for x in lv]


def spatial_containers(text):
    """Names of the `!Container`s that declare `spatial:` in this arch text."""
    return [b["name"] for b in _node_blocks(text.split("\n"))
            if b["kind"] == "Container" and b["spatial"] is not None]


_STORAGE_CLASSES = {"DRAM", "SRAM", "regfile", "storage", "smartbuffer_SRAM",
                    "smartbuffer_SRAM_banked",
                    "smartbuffer_RF", "smartbuffer_RF_decoded"}

#: prompt_7 C1.7. The upstream `smartbuffer_SRAM` compound declares no
#: `n_banks`, so the bank count every architecture in this study writes down --
#: 13 and 12 on the Eyeriss v1 GLB, 48/64/16 on v2, 16 on the simple designs --
#: never reached the `SRAM` primitive and CACTI modelled ONE monolithic array.
#: `archs/_shared/components/smartbuffer_SRAM_banked.yaml` forwards it.
BANKED_SRAM_CLASS = "smartbuffer_SRAM_banked"
PLAIN_SRAM_CLASS = "smartbuffer_SRAM"


def _node_blocks(lines):
    """(kind, indent, class, attributes-line index, has depth) per `- !Node`."""
    blocks, cur = [], None
    for idx, ln in enumerate(lines):
        m = re.match(r"^(\s*)- !(Component|Container|Nothing|Parallel|Hierarchical)\b", ln)
        if m:
            if cur:
                blocks.append(cur)
            cur = dict(kind=m.group(2), ind=len(m.group(1)), name=None, cls=None,
                       attr=None, spatial=None, depth=False)
            continue
        if cur is None:
            continue
        stripped = ln.strip()
        if not stripped or stripped.startswith("#"):
            continue
        ind = len(ln) - len(ln.lstrip())
        if ind <= cur["ind"]:
            blocks.append(cur)
            cur = None
            continue
        if ind == cur["ind"] + 2:
            if stripped.startswith("class:"):
                cur["cls"] = stripped.split(":", 1)[1].strip().strip('"')
            elif stripped.startswith("name:"):
                cur["name"] = stripped.split(":", 1)[1].split("#")[0].strip()
            elif stripped.startswith("attributes:"):
                cur["attr"] = idx
            elif stripped.startswith("spatial:"):
                cur["spatial"] = idx
        if re.match(r"\s*depth:\s*\d", ln):
            cur["depth"] = True
    if cur:
        blocks.append(cur)
    return blocks


def _inject_noc(text, arch, cfg):
    """Give every network in this design its energy coefficients.

    WHERE THE NUMBERS HAVE TO GO, AND WHY IT IS NOT OBVIOUS.
    Timeloop does not read a network's specs from the network. It reads them
    from the storage level the network hangs off (topology.cpp:1059 and 1121,
    `LegacyNetwork::ParseSpecs(storage[i], ...)`), and the network between
    levels i-1 and i takes them from the OUTER level. For the inter-PE
    networks -- the only ones with non-zero hop counts -- that outer level is
    not written in this file at all: it is the `dummy_storage` node the v4
    front-end synthesises from a `!Container` that declares `spatial:`.
    Injecting into the storage components alone therefore lands the
    coefficients on exactly the networks whose hop count is zero, and the
    measured NoC energy stays 0.00. So the coefficients go on the spatial
    `!Container`s, whose attributes the front-end propagates onto the
    synthesised level. `network_word_bits` is set explicitly because that
    level is declared `datawidth: 1`, which would otherwise bill an 8-bit
    operand as a single bit.

    THE TRAP ON THE WAY BACK DOWN. Container attributes are inherited by every
    child component too, so the hop-INdependent terms -- `energy-per-ingress`,
    and `router_energy` (Timeloop charges `1 + floor(hops)` routers, i.e. one
    even at zero hops) -- also reached the scratchpads inside each PE, and
    their inferred networks (`psum_spad <==> mac`, `weights_spad <==>
    psum_spad`, ...) were billed as if a register-to-ALU path were a NoC.
    Measured on eyeriss_like layer4.1.conv2: 94% of a 45% "NoC" share was
    that. Every storage `!Component` therefore gets EXPLICIT zeros for the two
    switching terms, so a child's own attributes override what it inherits and
    switching is charged only on the synthesised spatial fanout levels -- the
    networks that are actually a NoC. Wire energy needs hops and is harmless to
    inherit; it is left alone.

    AND NOT EVERY SPATIAL LEVEL IS A NoC. `noc.yaml`'s per-design `noc_levels`
    names the containers that are; the rest (v2's two-MAC SIMD pair, Simba's
    vector-MAC lanes) are PE datapath and receive no terms at all.
    """
    if not cfg.noc_enabled:
        return text
    terms = noc_terms(arch, cfg)
    wire, levels = terms["wire"], terms["levels"]
    tile_width = terms["tile_width_um"]
    bits = cfg.weight_bits
    lines = text.split("\n")
    blocks = _node_blocks(lines)
    # Every component, compute included: a MAC has no network of its own, but
    # an inherited non-zero switching term on it is an ambiguity the flattened
    # architecture would otherwise show, and an explicit zero costs nothing.
    zero_at = {b["attr"] for b in blocks
               if b["kind"] == "Component" and b["attr"] is not None}
    # Only the containers the contract names as the NoC. A spatial level that
    # is PE datapath (v2's SIMD pair, Simba's vector-MAC lanes) gets nothing.
    spatial_at = {b["spatial"] for b in blocks
                  if b["kind"] == "Container" and b["spatial"] is not None}
    charge_at = {b["spatial"]: (levels[b["name"]] if levels is not None
                                else (terms["router"], terms["ingress"]))
                 for b in blocks
                 if b["kind"] == "Container" and b["spatial"] is not None
                 and (levels is None or b["name"] in levels)}
    width_at = {b["spatial"]: tile_width[b["name"]] for b in blocks
                if b["kind"] == "Container" and b["spatial"] is not None
                and b["name"] in tile_width}
    out = []
    for idx, line in enumerate(lines):
        out.append(line)
        m = re.match(r"^(\s*)spatial:\s*\{", line)
        if m and idx in spatial_at and idx not in charge_at:
            # A datapath level NESTED inside a NoC container (v2's SIMD pair
            # sits inside PE) inherits the parent's router/ingress terms onto
            # its own synthesised level -- absence is not enough. Measured:
            # `inter_SIMD_spatial <==> mac` still billed 750 fJ/compute with
            # no attributes of its own. Explicit zeros override the parent.
            ind = m.group(1)
            out.append(f"{ind}attributes:   # NoC-datapath: not interconnect (archs/_shared/noc.yaml)")
            out.append(f"{ind}  wire_energy: 0.0")
            out.append(f"{ind}  router_energy: 0.0")
            out.append(f"{ind}  energy-per-ingress: 0.0")
            continue
        if m and idx in charge_at:
            router, ingress = charge_at[idx]
            ind = m.group(1)
            out.append(f"{ind}attributes:   # NoC: archs/_shared/noc.yaml")
            out.append(f"{ind}  wire_energy: {wire}")
            out.append(f"{ind}  network_word_bits: {bits}")
            # ALWAYS written, zero included. A NoC level nested inside another
            # (v2's PE row inside PE_cluster) inherits the parent's router and
            # ingress terms; until 2026-09-08 a declared zero was simply not
            # emitted, so v2's intra-cluster level paid the cluster router
            # again -- 20.5% of its whole NoC (FINDINGS.md, open defect 8.1).
            out.append(f"{ind}  router_energy: {router}")
            out.append(f"{ind}  energy-per-ingress: {ingress}")
            if idx in width_at:
                # Pre-floorplanned hop length (um); network-legacy.cpp's
                # SetTileWidth keeps a specified value over the area-derived one.
                out.append(f"{ind}  tile_width: {width_at[idx]}")
            continue
        if idx in zero_at:
            ind = " " * (len(line) - len(line.lstrip()) + 2)
            out.append(f"{ind}router_energy: 0.0         # NoC: not a fanout level")
            out.append(f"{ind}energy-per-ingress: 0.0    # (overrides the container)")
    return "\n".join(out)


def load_standard():
    """`archs/_shared/standard.yaml` -- what must be identical across designs."""
    if not _STANDARD:
        if not ARCH_STANDARD.exists():
            raise SystemExit(
                f"missing {ARCH_STANDARD}.\n"
                "  -> that file IS the apples-to-apples contract; without it there "
                "is nothing to validate the architectures against.")
        _STANDARD.update(yaml.safe_load(ARCH_STANDARD.read_text(encoding="utf-8")))
    return _STANDARD


def load_provenance():
    """`archs/_shared/provenance.yaml` -- where each declared number came from."""
    if not _PROVENANCE:
        if ARCH_PROVENANCE.exists():
            _PROVENANCE.update(
                yaml.safe_load(ARCH_PROVENANCE.read_text(encoding="utf-8")) or {})
    return _PROVENANCE


def mac_candidates():
    """`provenance.yaml` `mac_energy_pj.candidates`: the cited per-MAC energies
    ECC_MAC_PJ_OVERRIDE may be set to and labelled with (config.mac_citation)."""
    block = load_provenance().get("mac_energy_pj") or {}
    return list(block.get("candidates") or [])


def arch_standard(arch):
    """The per-architecture half of the contract, or None if undeclared."""
    return (load_standard().get("architectures") or {}).get(arch)


def accumulator_bits(arch, cfg):
    """The psum width this architecture is modelled at, and where it came from.

    PRIMARY: each design's PUBLISHED width (Eyeriss v1 16b, v2 20b, Simba 24b).
    Partial-sum precision is an architectural property; forcing one width on
    every design would be equalising the architectures, not the experiment.
    The user's decision on 2026-09-06 was explicit about this, and about the
    consequence: all ECC arms of one architecture must share its width, which
    they do because none of them touches the psum path.

    SENSITIVITY ONLY: `ECC_ACC_BITS` forces a common width, gets its own mapper
    cache and its own results namespace, and is labelled as a sensitivity study
    everywhere it appears. It is never the primary result.
    """
    if cfg.acc_bits_override is not None:
        return cfg.acc_bits_override, "ECC_ACC_BITS (common-width sensitivity study)"
    spec = arch_standard(arch)
    if spec and spec.get("accumulator_bits"):
        return int(spec["accumulator_bits"]), spec.get("accumulator_evidence", "")
    return None, "not declared in archs/_shared/standard.yaml"


def paper_source(arch):
    """`archs/<arch>/arch_paper.yaml`, or None if this design has no paper file.

    A paper file re-declares the storage precisions and scratchpad geometries
    from the published tables, with the citation for each number in a comment.
    It exists for every architecture in the five-way comparison; a design added
    later without one simply falls back to its stock YAML.
    """
    p = ARCH_SRC / arch / "arch_paper.yaml"
    return p if p.exists() else None


def arch_source(arch, cfg=None):
    """The arch.yaml the mapper should start from.

    `paper` fidelity (the default) prefers `archs/<arch>/arch_paper.yaml`;
    `stock` always takes the design as authored or as shipped. Passing cfg=None
    means stock, which is what the callers that only want the shipped file do.
    """
    if cfg is not None and getattr(cfg, "arch_fidelity", "stock") == "paper":
        paper = paper_source(arch)
        if paper is not None:
            return paper
    local = ARCH_SRC / arch / "arch.yaml"
    if local.exists():
        return local
    return DESIGNS_DIR / arch / "arch.yaml"


# --------------------------------------------------------------------- patching
def _patch_dram_depth(text, depth):
    """Make the DRAM level declare exactly `depth`.

    The stock example designs omit it and Timeloop needs it, so it is inserted.
    A design that DOES declare it -- every arch_paper.yaml does, and so did the
    locally authored v2 -- gets it rewritten instead of skipped: an earlier
    version returned the text unchanged in that case, which silently turned
    ECC_DRAM_DEPTH into a no-op on exactly those designs while still giving the
    run its own mapper cache, i.e. a fresh and expensive re-map of an
    unchanged architecture.
    """
    if "name: DRAM" not in text:
        return text
    head, tail = text.split("name: DRAM", 1)
    block, rest = tail.split("!", 1) if "!" in tail else (tail, "")
    if re.search(r"\bdepth:\s*\d+", block):
        block = re.sub(r"\bdepth:\s*\d+", f"depth: {depth}", block, count=1)
        return head + "name: DRAM" + block + ("!" + rest if rest else "")
    return text.replace(
        'class: DRAM\n    attributes:\n      type: "LPDDR4"',
        f'class: DRAM\n    attributes:\n      depth: {depth}\n      type: "LPDDR4"')


def _force_technology(text, node):
    """Rewrite every `technology:` declaration to the same node.

    Accelergy costs components at the node declared on their enclosing
    container. Designs that ship at different nodes cannot be compared as
    architectures until this is equalised.
    """
    return re.sub(r'technology:\s*"?[\w.]+"?', f'technology: "{node}"', text)


def _force_datawidth(text, bits, arch="?"):
    """Rewrite `datawidth:` to `bits` on the tensor-carrying levels only.

    Why: the stock simple_weight_stationary / simple_output_stationary designs
    declare datawidth 16 everywhere, so an "8-bit weights" study silently pays
    128 pJ per DRAM weight read on those two architectures while the eyeriss and
    simba designs pay 64. Forcing the datawidth removes that confound. `width:`
    is left alone -- it is the physical word width, and Timeloop derives
    entries-per-word as width/datawidth, so the same SRAM simply holds twice as
    many 8-bit values.

    Two levels are deliberately NOT rewritten:

    * A DEDICATED PARTIAL-SUM level -- one whose `keep:` list is Outputs and
      nothing else. Accumulator precision is a separate design choice from the
      operand quantization (Eyeriss v1 is an 8-bit design with 16-bit psums;
      v2 uses 20-bit psums), so forcing it would be wrong. It is also invalid:
      Timeloop asserts `width % datawidth == 0`, and v2's psum spad is
      `width: 20`, which 8 does not divide -- that assertion is what aborted
      the mapper before this exclusion existed.

    * Any level where `width` is not a multiple of `bits`. Reported, not
      silently skipped.

    A level shared between Outputs and the operands (a `shared_glb` with no
    dataspace constraint) IS rewritten, because Timeloop allows one datawidth
    per level and the operands are what this study measures. That matches how
    the eyeriss designs already declare their shared buffer.
    """
    out, skipped = [], []
    # split into "- !Node" blocks, keeping the delimiters so the text rebuilds
    parts = re.split(r"(\n\s*-\s*!)", text)
    for part in parts:
        m = re.search(r"\bdatawidth:\s*(\d+)", uncommented(part))
        if not m:
            out.append(part)
            continue
        name = re.search(r"name:\s*(\S+)", part)
        name = name.group(1) if name else "?"
        keep = re.search(r"keep:\s*\[([^\]]*)\]", part)
        keep = [s.strip() for s in keep.group(1).split(",")] if keep else []
        width = re.search(r"\bwidth:\s*(\d+)", uncommented(part))
        width = int(width.group(1)) if width else None

        if keep == ["Outputs"]:
            skipped.append(f"{name} (dedicated partial-sum level, "
                           f"datawidth {m.group(1)} kept)")
            out.append(part)
            continue
        if width is not None and width % bits != 0:
            skipped.append(f"{name} (width {width} is not a multiple of {bits}, "
                           f"datawidth {m.group(1)} kept)")
            out.append(part)
            continue
        out.append(re.sub(r"\bdatawidth:\s*\d+", f"datawidth: {bits}", part))

    if skipped and arch:
        print(f"  [datawidth] {arch}: forced to {bits}b except " + "; ".join(skipped))
    return "".join(out)


def _force_acc_bits(text, bits, arch="?", quiet=False):
    """SENSITIVITY STUDY ONLY: force one accumulator width on every design.

    The primary comparison keeps each design's published psum width -- that is
    an architectural property, and equalising it equalises the architectures.
    This exists so the sensitivity to that choice can be MEASURED rather than
    argued about, and everything it touches is namespaced separately.

    What it rewrites:

    * `mac.adder_width`, the accumulator arithmetic.
    * `datawidth` on every DEDICATED partial-sum level -- one whose `keep:` list
      is `Outputs` and nothing else. A level shared with the operands is left
      alone: Timeloop allows one datawidth per level and the operands are the
      standardized quantity.
    * `width` on those same levels, to the smallest multiple of `bits` that is
      at least the declared width. Timeloop asserts `width % datawidth == 0`,
      so a 16b word cannot hold 20b psums; widening the word is the only legal
      way to declare the same NUMBER of partial sums at a greater precision.
      Entry count is preserved, physical bits are not, and that is the point of
      the study -- it is asking what a wider accumulator costs.

    A level carrying Outputs that declares `# psum-width-ok:` is skipped: it
    holds requantized activations, not partial sums, and forcing an accumulator
    width onto it would be modelling a different design.
    """
    out, touched, skipped = [], [], []
    parts = re.split(r"(\n\s*-\s*!)", text)
    for part in parts:
        if re.search(r"\badder_width:\s*\d+", part):
            part = re.sub(r"\badder_width:\s*\d+", f"adder_width: {bits}", part)
            touched.append("mac.adder_width")
            out.append(part)
            continue
        if not re.search(r"\bdatawidth:\s*(\d+)", uncommented(part)):
            out.append(part)
            continue
        name = re.search(r"name:\s*(\S+)", part)
        name = name.group(1) if name else "?"
        keep = re.search(r"keep:\s*\[([^\]]*)\]", part)
        keep = [s.strip() for s in keep.group(1).split(",")] if keep else []
        if keep != ["Outputs"]:
            out.append(part)
            continue
        if re.search(r"#\s*psum-width-ok:", part):
            skipped.append(f"{name} (declared requantized, not a psum level)")
            out.append(part)
            continue
        width = re.search(r"\bwidth:\s*(\d+)", uncommented(part))
        if width:
            old_w = int(width.group(1))
            new_w = max(bits, -(-old_w // bits) * bits)
            part = re.sub(r"\bwidth:\s*\d+", f"width: {new_w}", part, count=1)
        part = re.sub(r"\bdatawidth:\s*\d+", f"datawidth: {bits}", part)
        touched.append(name)
        out.append(part)

    if not quiet and arch:
        note = f"  [acc-width] {arch}: forced to {bits}b on " + ", ".join(touched)
        if skipped:
            note += "; skipped " + "; ".join(skipped)
        print(note + "   (SENSITIVITY STUDY -- not the primary result)")
    return "".join(out)


#: Which storage levels `_scale_weight_capacity()` is allowed to touch.
#: `exclusive` only rewrites a level whose `keep:` list is Weights and nothing
#: else, so the extra capacity can only be spent on weights -- which is what
#: the reduced representation actually buys. `shared` also rewrites a level
#: that holds Weights ALONGSIDE another dataspace (`simple_output_stationary`'s
#: and `simple_input_stationary`'s `operand_glb` keep Inputs and Weights):
#: Timeloop has one capacity per level, so dilating it hands the mapper extra
#: INPUT capacity for free, which reconstruction does not pay for. The two are
#: an upper and a lower bound on one design and are quoted as a pair, the same
#: rule CLAUDE.md sets for the eyeriss `_wglb` variants.
#: `simple_weight_stationary` LEFT THAT SET on 2026-09-13: its operand half is
#: now `input_glb` + a Weights-only `weight_glb`, so both scopes give it the
#: same answer and its bracket has collapsed to a point.
WEIGHT_CAPACITY_SCOPES = ("exclusive", "shared")


def _scale_weight_capacity(text, scale, scope="exclusive", arch="?", quiet=False):
    """TASK 4: make a weight buffer hold `scale` x as many WEIGHTS.

    WHY THIS IS A MAPPER KNOB AND NOT AN EVALUATOR ONE. Under the
    reconstruction arm the on-chip weight representation is K/N of full width,
    so the same physical SRAM holds N/K = 1.615x more weights at BCH(63,39).
    Whether that buys anything is a question about the MAPPING -- a larger
    weight tile means fewer DRAM refetches of it -- and Task 3 structurally
    cannot answer it, because `RECON_OPTIMIZER=False` pins one mapping on every
    arm and both arms then refetch identically by construction. So the capacity
    has to be in the architecture the mapper sees, and this rewrites `depth:`
    on the weight-carrying levels to put it there.

    WHAT IT DOES NOT MODEL, AND WHY THE ENERGY MUST BE CORRECTED AFTERWARDS.
    Accelergy prices a level from its declared geometry, so a level at
    `depth x N/K` is costed as a physically LARGER array: more bits, more area,
    more energy per access. The reconstruction arm's array is not larger -- it
    is the same array holding narrower values -- so a Task 4 energy number read
    straight off a dilated mapping is charged for silicon the design does not
    have. `recon.capacity_dilation_correction()` re-prices those levels at the
    DECLARED geometry's per-access energy, and the dilated run records the
    ratio it corrected by. The mapping itself is unaffected either way except
    through the objective, which is why the ERT delta is reported: a level that
    got materially dearer per access biases the search AGAINST using the
    capacity, i.e. against the hypothesis, and that has to be visible rather
    than assumed away.

    THREE LEVELS ARE DELIBERATELY NOT REWRITTEN.

    * DRAM. It is not on-chip capacity and its depth is set by
      `ECC_DRAM_DEPTH`.
    * A level that holds no Weights. Dilating an activation or partial-sum
      buffer is a different architecture, not this treatment.
    * A DECLARED `depth: 1` register. `simple_weight_stationary`'s
      `weight_reg` is a pipeline latch, and FINDINGS 7.5 establishes that the
      mapping fills it once per read; scaling it to depth 2 would invent a
      reuse level the design does not have and would change what R4b's
      register is an addition TO. Reported, not silently skipped.

    A level holding Weights together with another dataspace is governed by
    `scope` -- see WEIGHT_CAPACITY_SCOPES.
    """
    if scale == 1.0:
        return text
    touched, skipped = [], []
    parts = re.split(r"(\n\s*-\s*!)", text)
    out = []
    for part in parts:
        bare = uncommented(part)           # see `uncommented()`: a `depth:` in
        depth = re.search(r"\bdepth:\s*(\d+)", bare)   # a COMMENT is prose
        if not depth:
            out.append(part)
            continue
        name = re.search(r"name:\s*(\S+)", bare)
        name = name.group(1) if name else "?"
        keep = re.search(r"keep:\s*\[([^\]]*)\]", bare)
        keep = [s.strip() for s in keep.group(1).split(",") if s.strip()] if keep else []
        d = int(depth.group(1))

        if "class: DRAM" in bare or name == "DRAM":
            out.append(part)               # never a candidate; not on-chip
            continue
        if "Weights" not in keep:
            out.append(part)
            continue
        if keep != ["Weights"] and scope != "shared":
            skipped.append(f"{name} (holds {'+'.join(keep)}; dilating it would "
                           f"also hand the mapper free {'/'.join(k for k in keep if k != 'Weights')} "
                           f"capacity -- ECC_WEIGHT_CAPACITY_SCOPE=shared includes it)")
            out.append(part)
            continue
        if d == 1:
            skipped.append(f"{name} (declared depth 1 -- a pipeline latch; "
                           f"scaling it would invent a reuse level the design "
                           f"does not have)")
            out.append(part)
            continue

        nd = max(1, int(round(d * scale)))
        if nd == d:
            skipped.append(f"{name} (depth {d} x {scale:g} rounds back to {d})")
            out.append(part)
            continue
        part = write_attr(part, "depth", nd)
        touched.append(f"{name} {d}->{nd}")
        out.append(part)

    if not quiet and arch:
        note = f"  [weight-capacity] {arch}: x{scale:g} on " + (
            ", ".join(touched) if touched else "NOTHING")
        if skipped:
            note += "; skipped " + "; ".join(skipped)
        print(note)
    return "".join(out)


#: A `#` comment, to the end of its line.
_COMMENT_RE = re.compile(r"#[^\n]*")


def uncommented(text):
    """`text` with every `#` comment blanked to spaces, POSITIONS PRESERVED.

    THE BUG THIS EXISTS FOR (found 2026-09-13, prompt_7 C1, on a real SLURM
    run). Every geometry regex in this module is `re.search(r"\bdepth:\s*(\d+)",
    part)` against the raw text, so it matches a `depth:` written in a COMMENT
    just as readily as the attribute -- and `re.sub(..., count=1)` then rewrites
    THE COMMENT and leaves the attribute alone.

    It fired the moment a comment was added that names a geometry it is NOT
    declaring: `# the paper's two banks of 512 x 64b would be `depth: 1024``
    made `_set_weight_geometry` read 1024 instead of 256, renormalise to 171,
    write "171" into the comment, and leave `depth: 256` beside the new
    `width: 384` -- 98,304 bits where 16,512 were intended, a SIX-FOLD capacity
    error with nothing on stdout to say so. The run's own log printed
    `filter_glb 1024x64b/8b -> 171x384b/8b`, which is the only reason it was
    caught.

    Blanking rather than deleting keeps every match span valid against the
    ORIGINAL string, so a caller can search the masked copy and splice into the
    real one.
    """
    return _COMMENT_RE.sub(lambda m: " " * len(m.group(0)), text)


def read_attr(text, key):
    """The integer value of `key:` in `text`, ignoring comments. None if absent."""
    m = re.search(rf"\b{re.escape(key)}:\s*(\d+)", uncommented(text))
    return int(m.group(1)) if m else None


def write_attr(text, key, value):
    """Replace the first UNCOMMENTED `key: <int>` in `text`. Returns the new text.

    Raises if there is none: a geometry rewrite that lands on nothing is the
    silent half of the bug `uncommented()` documents.
    """
    m = re.search(rf"\b{re.escape(key)}:\s*(\d+)", uncommented(text))
    if not m:
        raise ValueError(f"no uncommented `{key}:` to rewrite in:\n{text[:300]}")
    return text[:m.start(1)] + str(value) + text[m.end(1):]


def _weight_level_parts(text, scope="exclusive"):
    """Split `text` into `!Node` parts, tagging which ones hold WEIGHTS on chip.

    Shared by `_set_weight_geometry` and `_scale_weight_depth` so the two
    prompt_2 knobs can never disagree about which levels they are talking
    about -- if one narrowed a level the other did not shrink, the two arms of
    a pair would stop declaring the same geometry and the comparison would be
    void without anything saying so.

    Yields `(part, name, is_weight_level, why_not)`.
    """
    for part in re.split(r"(\n\s*-\s*!)", text):
        # COMMENT-BLIND REGEXES ARE HOW A COMMENT BECAME THE GEOMETRY
        # (`uncommented()`): every read below is against the masked copy, and
        # `_set_weight_geometry` writes through `write_attr()` for the same
        # reason.
        bare = uncommented(part)
        depth = re.search(r"\bdepth:\s*(\d+)", bare)
        name = re.search(r"name:\s*(\S+)", bare)
        name = name.group(1) if name else "?"
        if not depth:
            yield part, name, False, None
            continue
        keep = re.search(r"keep:\s*\[([^\]]*)\]", bare)
        keep = [s.strip() for s in keep.group(1).split(",") if s.strip()] if keep else []
        if "class: DRAM" in bare or name == "DRAM":
            # DRAM `datawidth` STAYS 8 ON EVERY ARM. `recon.py` owns the DRAM
            # K/N scaling in the evaluator; narrowing DRAM in the YAML too
            # would charge the same reduction twice (prompt_2, BEFORE ANY
            # NUMBER IS QUOTED, item 1).
            yield part, name, False, "DRAM (recon.py owns the DRAM K/N term)"
            continue
        if "Weights" not in keep:
            yield part, name, False, None
            continue
        if keep != ["Weights"] and scope != "shared":
            yield (part, name, False,
                   f"{name} holds {'+'.join(keep)}; "
                   f"ECC_WEIGHT_CAPACITY_SCOPE=shared includes it")
            continue
        if int(depth.group(1)) == 1:
            yield (part, name, False,
                   f"{name} declares depth 1 -- a pipeline latch, not a "
                   f"reuse level")
            continue
        yield part, name, True, None


def _set_weight_geometry(text, bits, levels=(), glb_mult=4, scope="exclusive",
                         arch="?", quiet=False,
                         weight_bits=code_widths.DEFAULT_WEIGHT_BITS):
    """PROMPT_2's WIDTH TABLE, applied PER LEVEL and PER ARM in one pass.

    THE ARMS DO NOT SHARE A DECLARED WIDTH. Each weight level declares the
    width that suits ITS OWN datawidth, and no level has to be legal for any
    other arm's datawidth:

        a level storing 8-bit weights    -> width  96 (spad) / 384 (GLB)
        a level storing 7-bit weights    -> width  98 / 392
        a level storing 6-bit weights    -> width  96 / 384
        a level storing 5-bit weights    -> width  95 / 380
        a level storing 4-bit weights    -> width  96 / 384

    **95 never has to divide 8.** The baseline/embedded arm is never mapped at
    width 95; it is mapped at 96. The only rule `timeloop-mapper` imposes is
    `width % (word_bits * block_size) == 0` (`buffer.cpp:302`, `block_size`
    defaults to 1, NO floor path, `exit=134` on a violation) -- and each width
    here is a multiple of the datawidth it is chosen for, by construction, so
    that abort is unreachable for every q. The lcm(q, 8) scheme of
    2026-09-11/12 is WITHDRAWN; `eccenergy/code_widths.py` records why.

    PER LEVEL, because an ERT arm narrows only the storage levels in its
    placement's `reduced` set: at `ECC_WEIGHT_DATAWIDTH_LEVELS=filter_glb` the
    GLB stores 5-bit weights at width 380 while `weights_spad` keeps 8-bit
    weights at width 96. One width for the whole design cannot express that.

    DEPTH IS COMMON TO EVERY ARM and holds the published TOTAL BITS:
    `depth' = round(depth * width / BASE_WIDTH)`, computed at the BASE width
    (96, x glb_mult above the PE array) rather than at this arm's own width.
    So the arms differ ONLY in `width` (by <= 2 %) and `datawidth`, which is
    what makes `capacity_ratio` reproduce prompt_2's `eff. capacity` column
    -- 1.1667 / 1.3333 / 1.5833 / 2.0000 -- and what leaves
    `assert_pair_geometry`'s depth check something real to check. It
    reproduces prompt_2's own depths: `weights_spad` 224 x 16 b = 3,584 b ->
    depth 37 at width 96; `filter_glb` 1024 x 64 b = 65,536 b -> depth 171 at
    width 384.

    `bits` is the reduced datawidth (`q`) or None for an all-8-bit arm;
    `levels` restricts WHICH levels take it (empty = every weight level).
    Every weight level is reshaped either way -- the reference arm reshapes
    too, or its depths would not match the arm it is compared with.

    DRAM IS NEVER TOUCHED: `recon.py` owns the DRAM K/N term, and narrowing
    DRAM here as well would charge the same reduction twice.

    Runs BEFORE `_scale_weight_depth`, so the swept ladder multiplies the
    renormalised depth rather than the published one.
    """
    want_levels = set(levels or ())
    names = [name for _p, name, is_w, _why
             in _weight_level_parts(text, scope) if is_w]
    if not names:
        return text
    spad = names[-1]
    unknown = want_levels - set(names)
    if unknown:
        raise ValueError(
            f"ECC_WEIGHT_DATAWIDTH_LEVELS names {', '.join(sorted(unknown))}, "
            f"which {arch} has no weight-carrying level called. It has: "
            f"{', '.join(sorted(names)) or 'none'}. Refusing rather than "
            f"narrowing every level and reporting it as a per-boundary "
            f"architecture.")

    out, touched, skipped, bad = [], [], [], []
    for part, name, is_weight, why_not in _weight_level_parts(text, scope):
        if not is_weight:
            if why_not:
                skipped.append(why_not)
            out.append(part)
            continue
        # READ OFF THE COMMENT-MASKED COPY. A `width:` or `depth:` written in
        # a comment is prose, not a declaration -- see `uncommented()` for the
        # six-fold capacity error that taught this.
        w0 = read_attr(part, "width")
        d0 = read_attr(part, "depth")
        dw0 = read_attr(part, "datawidth")
        if w0 is None or dw0 is None:
            skipped.append(f"{name} declares no "
                           + ("width:" if w0 is None else "datawidth:"))
            out.append(part)
            continue
        # WHICH datawidth this level ends up storing -- the arm's q only where
        # the arm says so, the payload width everywhere else.
        narrowed = bits is not None and (not want_levels or name in want_levels)
        q = int(bits) if narrowed else int(weight_bits)
        is_spad = name == spad
        want_w = code_widths.level_width(q, is_spad, glb_mult, weight_bits)
        want_d = code_widths.renormalised_depth(d0, w0, is_spad, glb_mult,
                                                weight_bits)
        if want_w % q != 0:                      # unreachable; a guard, not a path
            bad.append(f"{name}: width {want_w} % datawidth {q} != 0")
            out.append(part)
            continue
        if (w0, d0, dw0) == (want_w, want_d, q):
            skipped.append(f"{name} already declares {d0}x{w0}b/{dw0}b")
            out.append(part)
            continue
        # WRITE THROUGH `write_attr`, which finds the attribute on the masked
        # copy and splices into the real text -- `re.sub(..., count=1)` here
        # rewrote a COMMENT and left the attribute alone.
        part = write_attr(part, "width", want_w)
        part = write_attr(part, "depth", want_d)
        part = write_attr(part, "datawidth", q)
        touched.append(
            f"{name} {d0}x{w0}b/{dw0}b -> {want_d}x{want_w}b/{q}b "
            f"({want_w // q} weights/word, {d0 * w0:,} -> {want_d * want_w:,} bits)"
            + ("" if narrowed or bits is None else " [not in "
               "ECC_WEIGHT_DATAWIDTH_LEVELS: keeps the payload width]"))
        out.append(part)
    if bad:
        raise ValueError(
            f"{arch}: THE WIDTH TABLE produced a width its own datawidth does "
            f"not divide -- " + "; ".join(bad) + ".\n"
            f"  timeloop-mapper asserts `width % (word_bits * block_size) == 0` "
            f"(buffer.cpp:302) and ABORTS; there is no floor path.\n"
            f"  Every entry of eccenergy/code_widths.WIDTH_TABLE is a multiple "
            f"of its own q, so this is a table edit, not a configuration\n"
            f"  problem. Run `python3 -m eccenergy.code_widths` and fix the "
            f"entry; do NOT reach for a width that suits a DIFFERENT arm.")
    if not quiet and arch:
        note = (f"  [weight-geometry] {arch}: THE WIDTH TABLE (base "
                f"{code_widths.base_width(weight_bits)}b, GLB {glb_mult}x"
                + (f", q={bits} on "
                   + ("+".join(sorted(want_levels)) if want_levels else "every level")
                   if bits is not None else ", 8-bit arm")
                + ") on " + (", ".join(touched) if touched else "NOTHING"))
        if skipped:
            note += "; skipped " + "; ".join(skipped)
        print(note)
    return "".join(out)

def _scale_weight_depth(text, scale, levels=(), scope="exclusive", arch="?",
                        quiet=False):
    """PROMPT_2's ONLY SWEPT VARIABLE: `depth:` of the on-chip weight levels.

    Deliberately a SEPARATE knob from `_scale_weight_capacity`, even though
    the two rewrite the same field. That one exists to stand a narrower word
    up as a deeper array, so its energy has to be corrected back by
    `recon.capacity_dilation_correction()`. Here the depth change is the
    experiment: a shallower array really IS a smaller array, and its cheaper
    per-access energy is a real saving, not an artifact to undo. Sharing the
    slug would let a corrected run be read as an uncorrected one, so they get
    separate cache directories (`wdepth<scale>` vs `wcap<scale>`).

    `levels` restricts the rewrite to the named levels. Empty means every
    weight-carrying level, which is the default AND the limitation prompt_2
    records: one scale then moves `weights_spad` and `filter_glb` together, so
    it locates the zone but cannot say which level bought it. The second pass
    holds one at x1 and sweeps the other, which is what naming levels is for.
    A name that matches nothing is an error, not a silent no-op -- a typo
    there would quietly sweep every level and report it as a per-level result.
    """
    if scale == 1.0:
        return text
    want = set(levels or ())
    out, touched, skipped, seen = [], [], [], set()
    for part, name, is_weight, why_not in _weight_level_parts(text, scope):
        if not is_weight:
            if why_not:
                skipped.append(why_not)
            out.append(part)
            continue
        seen.add(name)
        if want and name not in want:
            skipped.append(f"{name} (not in ECC_WEIGHT_DEPTH_LEVELS)")
            out.append(part)
            continue
        d = read_attr(part, "depth")
        nd = max(1, int(round(d * scale)))
        if nd == d:
            skipped.append(f"{name} (depth {d} x {scale:g} rounds back to {d})")
            out.append(part)
            continue
        touched.append(f"{name} {d}->{nd}")
        out.append(write_attr(part, "depth", nd))
    unknown = want - seen
    if unknown:
        raise ValueError(
            f"ECC_WEIGHT_DEPTH_LEVELS names {', '.join(sorted(unknown))}, "
            f"which {arch} has no weight-carrying level called. It has: "
            f"{', '.join(sorted(seen)) or 'none'}. Refusing rather than "
            f"sweeping every level and reporting it as a per-level result.")
    if not quiet and arch:
        note = f"  [weight-depth] {arch}: x{scale:g} on " + (
            ", ".join(touched) if touched else "NOTHING")
        if skipped:
            note += "; skipped " + "; ".join(skipped)
        print(note)
    return "".join(out)


#: The loop dimensions a WEIGHT tile is indexed by. A `factors:` pin on any of
#: them caps how many weights a level can hold whatever its capacity is; a pin
#: on N, P or Q does not (weights do not index them) and is left alone, so the
#: activation and partial-sum structure of the dataflow is untouched.
WEIGHT_DIMENSIONS = ("M", "C", "R", "S")


# ---------------------------------------- prompt_7 Phase C: TIME in the MAPPER
def _component_attr_lines(text):
    """`{component name: (attributes-line index, attribute indent)}`.

    One reader for every Phase C patch below, so `read_bandwidth`, the
    bandwidth scale and the bank geometry can never disagree about which
    component they are editing. A component that declares no `attributes:`
    block is absent from the map, and the caller refuses rather than inventing
    one -- a level with no attributes is a level with no geometry, which is not
    something this study has.
    """
    lines = text.split("\n")
    out = {}
    for b in _node_blocks(lines):
        if b["kind"] != "Component" or b["name"] is None or b["attr"] is None:
            continue
        out[b["name"]] = (b["attr"], b["ind"] + 4)
    return out


def _insert_attributes(text, additions, arch="?"):
    """Add attribute lines to named components. `additions` = {name: [lines]}.

    Lines go DIRECTLY under the component's `attributes:` key, indented to
    match it, and each carries its own trailing comment. A name the file does
    not have is an ERROR: these attributes are what makes one mapper arm a
    different chip from another, and one that silently failed to land would be
    two arms sharing one architecture under two directory names -- exactly the
    failure prompt_6 RULE 4.4.5 exists to prevent.
    """
    if not additions:
        return text
    where = _component_attr_lines(text)
    missing = [n for n in additions if n not in where]
    if missing:
        raise ValueError(
            f"{arch}: no component with an `attributes:` block called "
            f"{', '.join(sorted(missing))}; the file has "
            f"{', '.join(sorted(where)) or 'none'}. Refusing rather than "
            f"dropping a declaration that separates two mapper arms.")
    at = {where[n][0]: (where[n][1], additions[n]) for n in additions}
    lines, out = text.split("\n"), []
    for idx, line in enumerate(lines):
        out.append(line)
        if idx in at:
            ind, add = at[idx]
            out.extend(" " * ind + a for a in add)
    return "\n".join(out)


def _declare_dram_bandwidth(text, items_per_cycle, arch="?", quiet=False):
    """prompt_7 C1.1: give the DRAM level the off-chip speed limit THE MAPPER
    READS.

    Until this landed no architecture in `archs/` declared an off-chip
    bandwidth at all, so Timeloop skipped the DRAM throughput check entirely
    (`buffer.cpp:2575` gates on `IsSpecified()`), off-chip traffic cost ZERO
    cycles, and the search optimised a machine with an infinitely fast memory.
    That is prompt_7's Defect 1, and it is why a reconstruction arm's K/N
    traffic saving could never appear as latency.

    `shared_bandwidth`, NOT `read_bandwidth` + `write_bandwidth`. The DQ bus is
    one wire set that reads and writes take turns on, so the limit is on their
    SUM -- which is exactly the term `latency_post.roofline()` already charges
    (`offchip_limit / (d_r + d_w)`). Declaring the two directions separately
    would let Timeloop deliver 2x this number while the evaluator capped it at
    1x: two timing models for one bus, and one of them wrong.

    UNITS ARE ITEMS PER CYCLE of THIS design's clock. `config` owns the
    conversion from MB/s (`dram_items_per_cycle_for`), so the number here and
    the number the roofline uses have one source.

    MEASURED CONSEQUENCE (prompt_7 7.2, 12 real mapper searches): with a
    binding limit the mapper picks a DIFFERENT plan on both test shapes -- one
    of them spends parallelism to comply, folding onto 96 of 168 PEs. At
    LPDDR4-3200 the limit never binds and nothing moves.
    """
    if items_per_cycle is None:
        return text
    add = [f"shared_bandwidth: {items_per_cycle:.6g}   # items/cycle: "
           f"ECC_DRAM_BANDWIDTH_MBPS at this design's own clock",
           "                     # ONE bus -- reads and writes share it, which is",
           "                     # the term latency_post.roofline() charges."]
    out = _insert_attributes(text, {"DRAM": add}, arch)
    if not quiet and arch:
        print(f"  [off-chip] {arch}: DRAM shared_bandwidth "
              f"{items_per_cycle:.6g} items/cycle")
    return out


def _declare_bw_scale(text, factors, arch="?", quiet=False):
    """prompt_7 C1.2: `per_dataspace_bandwidth_consumption_scale` on every
    stage of this arm's boundary.

    Timeloop multiplies ONE dataspace's bandwidth demand by the factor
    (`buffer.cpp:2556`); timeloopfe v4 declares the attribute
    (`arch.py:538`) and every stats file prints `Bandwidth Consumption Scale`,
    so a version that ignored it would be visible on disk. Measured on fixed
    mappings with off-chip bandwidth binding: -17.28% and -21.94% cycles, with
    dynamic energy BIT-IDENTICAL (prompt_7 A.2) -- it is a timing declaration
    and nothing else.

    THE FACTOR IS PER STAGE and `recon.arm_bw_factors()` derives it: `K/N` at
    DRAM, where the weights are a bit stream, and `q/8` on chip, where a level
    holds whole weights in a narrower word. Using one everywhere is a 5%
    silent inconsistency with `ECC_RECON_PACKING`, not a rounding choice.

    NETWORK STAGES ARE SKIPPED HERE AND ONLY HERE. They are part of what the
    arm DECLARES (`recon.arm_bw_scale`, which is what makes R3 a different
    chip from R2), but `LegacyNetwork::ComputePerformance()` is an empty stub
    and there is no YAML component to attach an attribute to. Reporting rule
    R-3 is that fact, written down.
    """
    timed = {lvl: f for lvl, f in (factors or {}).items()
             if f["kind"] in ("dram", "storage")}
    if not timed:
        return text
    add = {}
    for lvl, f in timed.items():
        add[lvl] = [f"per_dataspace_bandwidth_consumption_scale: "
                    f"{{{f['dataspace']}: {f['factor']:.6f}}}   # {f['why']}"]
    out = _insert_attributes(text, add, arch)
    if not quiet and arch:
        print("  [bw-scale] " + arch + ": "
              + ", ".join(f"{lvl} {f['dataspace']} x{f['factor']:.6f}"
                          for lvl, f in timed.items())
              + (("; declared but NOT timed (no network speed model, "
                  "reporting rule R-3): "
                  + ", ".join(lvl for lvl, f in factors.items()
                              if f["kind"] == "network"))
                 if any(f["kind"] == "network" for f in factors.values()) else ""))
    return out


def _bitaware_onchip_bandwidth(text, levels, factor, arch="?", quiet=False):
    """prompt_7 C1.3: a narrowed level's declared port, priced in BITS.

    Timeloop's throughput check counts ITEMS per cycle and a narrow weight is
    still one item, so `datawidth: q` alone is invisible to the clock. A real
    port moves BITS: a level storing `q`-bit weights delivers `8/q` times as
    many of them per cycle through the same wires, and THAT is what the
    architecture has to say for the mapper to see it.

    WHERE IT LANDS. `filter_glb`'s declared `read_bandwidth: 16` is literally
    what caps every fully-connected layer at 9.52% PE utilisation -- 16 of 168
    PEs, measured on resnet18 `fc` and mobilenet `classifier.1`, where a
    32-PE candidate was rejected because it would have throttled (prompt_7
    4.5, A.7). Those layers are 0.28% / 2.10% of their models' cycles, so the
    aggregate CNN effect is ~0.1-1%; on a batch-1 transformer every layer is
    that layer (Phase D).

    ONLY THE LEVELS THE ARM NARROWS. A level still storing 8-bit weights moves
    the same bits per cycle it always did, and raising its port would be a
    free architecture change credited to the code.
    """
    if not levels or factor == 1.0:
        return text
    # DRAM IS NEVER BIT-AWARE. `recon.py` owns the DRAM K/N term and C1.2
    # declares it as a bandwidth SCALE; raising the off-chip limit by 8/q as
    # well would charge one reduction twice -- the same trap
    # `_weight_level_parts` records for `datawidth`. Unreachable today
    # (`arm_narrow_levels` keeps only storage stages), which is exactly when a
    # guard is cheap.
    offchip = [n for n in levels if "dram" in n.lower()]
    if offchip:
        raise ValueError(
            f"{arch}: ECC_ONCHIP_BW_BITAWARE reached {', '.join(offchip)}. The "
            f"off-chip limit is ONE bus carrying a bit stream and its relief is "
            f"already declared as per_dataspace_bandwidth_consumption_scale "
            f"(prompt_7 C1.2); scaling it by 8/q as well would charge the "
            f"reduced representation twice.")
    where = _component_attr_lines(text)
    missing = [n for n in levels if n not in where]
    if missing:
        raise ValueError(f"{arch}: cannot make {', '.join(missing)} bit-aware; "
                         f"no such component with attributes")
    lines, out, touched = text.split("\n"), [], []
    current = None
    starts = {idx: name for name, (idx, _ind) in where.items()}
    for idx, line in enumerate(lines):
        if idx in starts:
            current = starts[idx]
        m = re.match(r"^(\s*)(read_bandwidth|write_bandwidth|shared_bandwidth):"
                     r"\s*([\d.eE+-]+)(.*)$", line)
        if m and current in levels:
            ind, key, val, rest = m.groups()
            new = float(val) * factor
            out.append(f"{ind}{key}: {new:.6g}"
                       f"   # x{factor:g} = 8/q: this level stores q-bit weights "
                       f"(was {val}){rest}")
            touched.append(f"{current}.{key} {val} -> {new:.6g}")
            continue
        out.append(line)
    if not touched:
        raise ValueError(
            f"{arch}: ECC_ONCHIP_BW_BITAWARE is on and the arm narrows "
            f"{', '.join(levels)}, but none of those levels declares a "
            f"bandwidth to scale. A bit-aware port that lands on nothing is a "
            f"no-op reported as an architecture change.")
    if not quiet and arch:
        print(f"  [bit-aware bw] {arch}: " + ", ".join(touched))
    return "\n".join(out)


def _bank_geometry(text, arch="?", quiet=False):
    """prompt_7 C1.7: let a level's DECLARED bank count reach CACTI.

    `smartbuffer_SRAM` declares no `n_banks`, so every `n_banks:` in this
    study's architectures was inert and CACTI priced one monolithic array. The
    plug-in itself takes the attribute (`cacti_wrapper.py:137`) and hands it to
    CACTI as `-UCA bank N`; only the compound in the way had to be replaced.
    Measured on `eyeriss_like_wglb` (Accelergy, 45nm, this design's own
    geometry): `ifmap_glb` read 23.539 -> 16.571 pJ, `psum_glb` 22.729 ->
    16.053, `filter_glb` 13.591 -> 11.745.

    ONLY LEVELS THAT DECLARE `n_banks:` THEMSELVES ARE SWITCHED, and that is
    the whole subtlety. `timeloopfe` v4 gives EVERY storage level a default
    `n_banks: 2`, which is visible in the flattened architecture and is a
    published number for none of them. Forwarding it wholesale would hand the
    depth-3 `weights_spad` a two-bank model out of a front-end default -- and
    the CACTI wrapper floors depth at `64 x n_banks`, so on a shallow array
    that default moves the price through the FLOOR rather than through any
    banking. A level whose paper says nothing about banking therefore keeps
    `smartbuffer_SRAM` and prices exactly as it did before Phase C.

    KNOWN, RECORDED, NOT FUDGED: CACTI is called at
    `2 ** ceil(log2(n_banks))`, so the 13-bank ifmap GLB is modelled as 16
    banks; the wrapper computes the linear correction `bankscale = 13/16` and
    then never applies it (`cacti_wrapper.py:177-178` -- it is dead in the
    plug-in, not here). Leakage does use the declared count. That is stated in
    `archs/_shared/provenance.yaml` and reported by `run.sh diagnose`; it is
    not silently compensated for here.
    """
    lines, out, touched = text.split("\n"), [], []
    blocks = {b["attr"]: b for b in _node_blocks(lines)
              if b["kind"] == "Component" and b["attr"] is not None}
    banked = set()
    for attr_idx, b in blocks.items():
        if b["cls"] != PLAIN_SRAM_CLASS:
            continue
        # the component's own body: from its `- !Component` line to the next
        end = min([i for i in blocks if i > attr_idx] or [len(lines)])
        body = "\n".join(lines[attr_idx:end])
        m = re.search(r"^\s*n_banks:\s*(\d+)", uncommented(body), re.M)
        if m and int(m.group(1)) > 1:
            banked.add(b["name"])
            touched.append(f"{b['name']} ({m.group(1)} banks)")
    if not banked:
        return text
    current = None
    for line in lines:
        m = re.match(r"^\s*name:\s*(\S+)", line)
        if m:
            current = m.group(1).split("#")[0].strip()
        c = re.match(r"^(\s*)class:\s*" + PLAIN_SRAM_CLASS + r"\s*(#.*)?$", line)
        if c and current in banked:
            out.append(f"{c.group(1)}class: {BANKED_SRAM_CLASS}"
                       f"   # prompt_7 C1.7: this level's declared n_banks "
                       f"reaches CACTI")
            continue
        out.append(line)
    if not quiet and arch:
        print(f"  [banks] {arch}: {BANKED_SRAM_CLASS} on " + ", ".join(touched)
              + "  (CACTI rounds to the next power of two and drops its own "
                "bankscale correction -- provenance.yaml `sram_banking`)")
    return "\n".join(out)


def _relax_weight_factors(text, arch="?", quiet=False):
    """TASK 4 LEVER 2: let the weight TILE grow into the room a dilation adds.

    WHY CAPACITY ALONE MEASURES ZERO. `_scale_weight_capacity` makes the buffer
    bigger; it does not make the mapper able to spend it. `eyeriss_like`'s
    `weights_spad` declares

        temporal:
          factors: [N=1, M=1, P=1, Q=1, S=1]

    and `M=1` pins the M tile AT THAT LEVEL to one, so the resident tile is
    M(8 from `psum_spad` below) x C(16) = 128 weights and stays 128 whatever the
    capacity is. Measured (FINDINGS 7.8): `weights held` is exactly 21,504 at
    x1, x1.6154, x4, x8, x16 AND x32, where the buffer is at 0.9 % fill. The
    binding constraint is the DATAFLOW CONSTRAINT, not the silicon, and no
    capacity sweep can find that out because the constraint does not move.

    WHAT IT REWRITES. On every weight-carrying level that
    `weight_capacity_levels()` reports (so: never DRAM, never a level holding no
    Weights, never a declared `depth: 1` latch), the `factors:` entries for
    M, C, R and S are dropped from the TEMPORAL constraints. N, P and Q keep
    their pins: weights do not index them, so relaxing those would change the
    activation and psum tiling instead of the weight tile, which is a different
    experiment.

    THIS IS A DIFFERENT DATAFLOW AND MUST BE LABELLED AS ONE. Eyeriss v1's
    `M=1` at the filter spad is the row-stationary dataflow; a design without it
    is not the chip JSSC 2017 describes, and `source: published` does not
    licence its name. It gets its own cache (`wrelax`) and both arms of a Task 4
    pair are mapped under it, so the COMPARISON stays fair even though neither
    arm is the published design.

    It also widens the mapspace, so the search has strictly more to explore at
    the same victory condition -- a relaxed run that comes back worse is
    evidence about the SEARCH, not about the dataflow.
    """
    touched, skipped = [], []
    parts = re.split(r"(\n\s*-\s*!)", text)
    out = []
    for part in parts:
        name = re.search(r"name:\s*(\S+)", part)
        name = name.group(1) if name else "?"
        keep = re.search(r"keep:\s*\[([^\]]*)\]", part)
        keep = [s.strip() for s in keep.group(1).split(",") if s.strip()] if keep else []
        depth = re.search(r"\bdepth:\s*(\d+)", uncommented(part))
        if ("Weights" not in keep or not depth or "class: DRAM" in part
                or name == "DRAM"):
            out.append(part)
            continue
        if int(depth.group(1)) == 1:
            skipped.append(f"{name} (declared depth 1 -- a latch)")
            out.append(part)
            continue
        # Only the TEMPORAL factors of this level, never a spatial container's.
        m = re.search(r"(temporal:\s*\n(?:\s+\w+:.*\n)*?\s+factors:\s*)"
                      r"\[([^\]]*)\]", part)
        if not m:
            skipped.append(f"{name} (no temporal factors: to relax)")
            out.append(part)
            continue
        entries = [e.strip() for e in m.group(2).split(",") if e.strip()]
        kept = [e for e in entries
                if e.split("=")[0].strip().upper() not in WEIGHT_DIMENSIONS]
        if len(kept) == len(entries):
            skipped.append(f"{name} (pins nothing a weight tile is indexed by)")
            out.append(part)
            continue
        dropped = [e for e in entries if e not in kept]
        part = part[:m.start(2) - 1] + "[" + ", ".join(kept) + "]" + part[m.end(2) + 1:]
        touched.append(f"{name} dropped {'/'.join(dropped)}")
        out.append(part)

    if not quiet and arch:
        note = f"  [weight-factor-relax] {arch}: " + (
            "; ".join(touched) if touched else "NOTHING")
        if skipped:
            note += "; skipped " + "; ".join(skipped)
        print(note)
    return "".join(out)


#: WHICH LEVELS EACH LOOP DIMENSION MAY BE SPLIT ACROSS, per design.
#:
#: THE PROBLEM THIS SOLVES, measured 2026-09-10 (FINDINGS 7.9). The mapper
#: reports its own mapspace for ONE layer of Eyeriss v1 as IndexFactorization
#: 7.41e10 x LoopPermutation 8.96e9 ~ 6.6e20. At the measured 400,000 valid
#: mappings per thread-hour, victory 50000 covers 0.07 % of ONE thread's
#: factorization subspace, ignoring permutations entirely. The consequence is
#: not slow convergence, it is NO convergence: the residual between victory
#: 4000 and 10000 is 43.7 % where the ECC effect being measured is 5.8 %, and
#: refetch is non-monotone in the budget. The search noise is larger than the
#: signal, so no Recon-vs-Embedded conclusion survives at ANY affordable
#: budget.
#:
#: RAISING THE BUDGET CANNOT FIX THAT AND SHRINKING THE SPACE CAN. A bigger
#: budget samples more of the same enormous space; both arms still land on
#: arbitrary points and the DIFFERENCE between two arbitrary points is noise
#: (measured: the ordering between the arms flips). Constraining the loop nest
#: collapses the space to something searchable EXHAUSTIVELY, and then each arm
#: gets its true optimum rather than a sample -- so the difference is
#: architectural by construction and the gate passes because there is nothing
#: left unsearched. progress.txt 2026-09-10 costs the tiers: pinning R and S to
#: one level each leaves 2.06e9 factorizations (still impossible), adding
#: P/Q across <=2 levels leaves 1.32e7 (29 h), and adding C/M across <=3
#: leaves 3.63e4 -- EXACT in about five minutes, and roughly 8x CHEAPER than
#: the victory-5000 run whose answer is 9-11 % wrong.
#:
#: WHY THE FREE SETS ARE THESE ONES. They are read off the BEST MAPPING THE
#: SEARCH HAS EVER FOUND for this design -- the victory-10000 embedded nest at
#: x1, 169.13 uJ against the victory-4000 mapping's 300.34 uJ on identical
#: silicon. Constraining around a known-good region is a choice and it is
#: stated: the exhaustive answer is the best mapping IN THIS FAMILY, not in
#: the whole space. What makes it a fair ECC comparison is that BOTH ARMS get
#: the identical constraint, so neither is handed a region the other cannot
#: reach.
#:
#: THIS IS A DIFFERENT DATAFLOW AND MUST BE LABELLED AS ONE, exactly as
#: `_relax_weight_factors` is: a design run under it is not the chip JSSC 2017
#: describes, `source: published` does not licence its name, and it gets its
#: own cache slug (`mcons`).
#:
#: `weights_spad` deliberately KEEPS M free, because this lever is meant to be
#: run together with `ECC_WEIGHT_FACTOR_RELAX=1` -- that relaxation exists to
#: let the weight TILE grow into the room a shallower/narrower array leaves,
#: and pinning M back at that level here would undo it.
MAPSPACE_FREE_LEVELS = {
    "eyeriss_like_wglb": {
        # dimension: the levels it may be split across. Pinned to 1 elsewhere.
        "C": ("DRAM", "PE", "weights_spad"),
        "M": ("ifmap_glb", "PE_column", "weights_spad", "psum_spad"),
        "R": ("psum_glb",),
        "S": ("PE",),
        "P": ("ifmap_glb", "psum_glb"),
        "Q": ("filter_glb", "PE_column"),
        "N": (),                      # N = 1 in every workload here
    },
}

#: Every loop dimension the constraint reasons about.
MAPSPACE_DIMENSIONS = ("N", "C", "M", "R", "S", "P", "Q")


def _merge_factor_list(existing, pins):
    """`existing` factor entries plus `pins`, with `pins` winning. Order-stable."""
    out, seen = [], set()
    for e in list(existing) + list(pins):
        k = e.split("=")[0].strip().upper()
        if k in seen:
            out = [x for x in out if x.split("=")[0].strip().upper() != k]
        seen.add(k)
        out.append(e)
    # `pins` appended last already won; de-duplicate keeping the LAST
    final, seen = [], set()
    for e in reversed(out):
        k = e.split("=")[0].strip().upper()
        if k in seen:
            continue
        seen.add(k)
        final.append(e)
    return list(reversed(final))


def _constrain_mapspace(text, arch="?", quiet=False):
    """Pin every loop dimension to 1 at the levels `MAPSPACE_FREE_LEVELS` does
    not list, so the index-factorization space collapses to something the
    mapper can search EXHAUSTIVELY.

    A storage level gets `constraints.temporal.factors`; a spatial container
    gets `constraints.spatial.factors`. Both are created if absent and merged
    if present, with these pins winning -- a design that already pins a
    dimension keeps that pin, and one that leaves it free has it pinned here
    unless the free set names the level.
    """
    spec = MAPSPACE_FREE_LEVELS.get(arch)
    if not spec:
        if not quiet and arch:
            print(f"  [mapspace] {arch}: no MAPSPACE_FREE_LEVELS entry -- "
                  f"NOT constrained (the search is unbounded here)")
        return text
    touched, out = [], []
    for part in re.split(r"(\n\s*-\s*!)", text):
        name = re.search(r"name:\s*(\S+)", part)
        if not name or "!" in part[:3] or not re.search(r"name:", part):
            out.append(part)
            continue
        name = name.group(1)
        if name == "DRAM" and "class: DRAM" not in part:
            out.append(part)
            continue
        if re.search(r"class:\s*intmac", part):
            out.append(part)               # the arithmetic level takes none
            continue
        # ONLY REAL LOOP LEVELS. A storage component declares `depth:`; a
        # spatial container declares `spatial: {meshX/meshY}`. A bare grouping
        # container (`system`, and the accelerator container itself) is
        # neither -- it carries no loops, so pinning every dimension to 1 on
        # it would be inventing a constraint on a level Timeloop does not map.
        is_storage = re.search(r"\bdepth:\s*\d+", uncommented(part)) is not None
        is_spatial = re.search(r"\n\s*spatial:\s*\{[^}]*mesh", part) is not None
        if not (is_storage or is_spatial):
            out.append(part)
            continue
        pins = [f"{d}=1" for d in MAPSPACE_DIMENSIONS
                if name not in spec.get(d, ())]
        if not pins:
            out.append(part)
            continue
        kind = "spatial" if is_spatial else "temporal"
        m = re.search(rf"({kind}:\s*\n(?:\s+\w+:.*\n)*?\s+factors:\s*)\[([^\]]*)\]",
                      part)
        if m:
            existing = [e.strip() for e in m.group(2).split(",") if e.strip()]
            merged = _merge_factor_list(existing, pins)
            part = part[:m.start(2) - 1] + "[" + ", ".join(merged) + "]" \
                + part[m.end(2) + 1:]
            touched.append(f"{name}({kind}) {','.join(pins)}")
            out.append(part)
            continue
        # no factors: list at that kind -- create the block
        km = re.search(rf"\n(\s+){kind}:\s*\n", part)
        if km:
            ind = km.group(1)
            part = (part[:km.end()] + f"{ind}  factors: [{', '.join(pins)}]\n"
                    + part[km.end():])
            touched.append(f"{name}({kind}, new factors) {','.join(pins)}")
            out.append(part)
            continue
        cm = re.search(r"\n(\s+)constraints:\s*\n", part)
        if cm:
            ind = cm.group(1)
            block = (f"{ind}  {kind}:\n"
                     f"{ind}    factors: [{', '.join(pins)}]\n")
            part = part[:cm.end()] + block + part[cm.end():]
            touched.append(f"{name}({kind}, new block) {','.join(pins)}")
            out.append(part)
            continue
        nm = re.search(r"\n(\s+)name:\s*\S+\s*\n", part)
        if nm:
            ind = nm.group(1)
            block = (f"{ind}constraints:\n{ind}  {kind}:\n"
                     f"{ind}    factors: [{', '.join(pins)}]\n")
            part = part.rstrip("\n") + "\n" + block
            touched.append(f"{name}({kind}, new constraints) {','.join(pins)}")
            out.append(part)
            continue
        out.append(part)
    if not quiet and arch:
        print(f"  [mapspace] {arch}: EXHAUSTIVE-SEARCH CONSTRAINT on "
              + ("; ".join(touched) if touched else "NOTHING")
              + "   (a DIFFERENT DATAFLOW -- not the published chip)")
    return "".join(out)


def weight_capacity_levels(arch, cfg):
    """Which levels a capacity dilation would rewrite, and by how much.

    Reported by `validate`/`diagnose` and by the Task 4 experiment, so a run
    that dilates NOTHING says so instead of quietly reproducing the declared
    mapping under a Task 4 heading.
    """
    text = arch_source(arch, cfg).read_text()
    rows = []
    for part in re.split(r"(?=\n\s*-\s*!)", text):
        depth = re.search(r"\bdepth:\s*(\d+)", uncommented(part))
        name = re.search(r"name:\s*(\S+)", part)
        if not depth or not name or name.group(1) == "DRAM":
            continue
        keep = re.search(r"keep:\s*\[([^\]]*)\]", part)
        keep = [s.strip() for s in keep.group(1).split(",") if s.strip()] if keep else []
        if "Weights" not in keep:
            continue
        width = re.search(r"\bwidth:\s*(\d+)", uncommented(part))
        dw = re.search(r"\bdatawidth:\s*(\d+)", uncommented(part))
        d = int(depth.group(1))
        per_word = (int(width.group(1)) // int(dw.group(1))) if width and dw else 1
        rows.append({"level": name.group(1), "depth": d, "weights_per_word": per_word,
                     "weights_per_instance": d * per_word,
                     "shared_with": [k for k in keep if k != "Weights"],
                     "latch": d == 1})
    return rows


def patched_weight_geometry(arch, cfg):
    """The geometry of every weight-carrying level AS THE MAPPER WILL SEE IT.

    `weight_capacity_levels()` reads the SOURCE YAML, which is what a
    dilation-scope question needs. This reads the PATCHED text -- after
    `ECC_WEIGHT_DEPTH_SCALE` and `ECC_WEIGHT_DATAWIDTH` -- which is what the
    fairness rule needs, because the whole prompt_2 claim is that the two arms
    differ in `datawidth` and in NOTHING ELSE.

    Returns `{level: {"depth", "width", "datawidth", "weights_per_word",
    "weights"}}`, outer to inner.
    """
    text = _patched_text(arch, cfg, quiet=True)
    out = {}
    for part in re.split(r"(?=\n\s*-\s*!)", text):
        depth = re.search(r"\bdepth:\s*(\d+)", uncommented(part))
        name = re.search(r"name:\s*(\S+)", part)
        if not depth or not name or name.group(1) == "DRAM":
            continue
        keep = re.search(r"keep:\s*\[([^\]]*)\]", part)
        keep = [s.strip() for s in keep.group(1).split(",") if s.strip()] if keep else []
        if "Weights" not in keep:
            continue
        width = re.search(r"\bwidth:\s*(\d+)", uncommented(part))
        dw = re.search(r"\bdatawidth:\s*(\d+)", uncommented(part))
        d = int(depth.group(1))
        w = int(width.group(1)) if width else None
        b = int(dw.group(1)) if dw else None
        per_word = (w // b) if (w and b) else 1
        out[name.group(1)] = {
            "depth": d, "width": w, "datawidth": b,
            "weights_per_word": per_word, "weights": d * per_word,
            "shared_with": [k for k in keep if k != "Weights"]}
    return out


def assert_pair_geometry(arch, cfg_ref, cfg_arm, ref_name="embedded",
                         arm_name="recon"):
    """PROMPT_2 FAIRNESS RULE, mechanically: the two arms of a pair must hold
    the same LEVELS at the same DEPTH.

    **IT DOES NOT CHECK `width` AND IT DOES NOT CHECK `datawidth`.** Under
    prompt_2's WIDTH TABLE the arms declare DIFFERENT widths on purpose -- each
    one the width that suits its own datawidth, 96 / 98 / 96 / 95 / 96 -- and
    neither has to be legal for the other's datawidth, because neither is ever
    mapped on the other's silicon. Asserting a shared width is what produced
    the withdrawn `lcm(q, 8)` scheme (56 / 24 / 40) and, through it,
    BCH(63,39)'s spurious 37.69 %: see `eccenergy/code_widths.py`. Do not
    reintroduce that check.

    WHAT IT STILL CHECKS, AND WHY IT IS AN ASSERTION AND NOT A CONVENTION.
    DEPTH. This is the exact defect that made the earlier sweep prove nothing:
    capacity was expressed as `depth x N/K`, so Accelergy priced the
    reconstruction arm's array 1.18-1.46x dearer per access and the optimiser
    had a REASON to leave the room unused (FINDINGS 7.8). A depth difference is
    real silicon one arm does not have. `_set_weight_geometry` renormalises
    every arm's depth at the BASE width precisely so this stays true while the
    widths differ, which is what makes the check meaningful rather than
    vacuous.

    `ECC_DISABLE_ASSERT_PAIR_GEOMETRY=1` turns the depth check off, for a study
    that deliberately varies depth between the reconstruction and embedded arms.
    It is the ONLY thing that knob disables -- there is no width check for it
    to disable. Default 0: a depth difference nobody asked for is still a void
    comparison.

    Returns the per-level record either way, so a caller that disabled the
    assertion can still print what differs.
    """
    a = patched_weight_geometry(arch, cfg_ref)
    b = patched_weight_geometry(arch, cfg_arm)
    disabled = bool(getattr(cfg_arm, "disable_pair_geometry_assert", False)
                    or getattr(cfg_ref, "disable_pair_geometry_assert", False))
    problems = []
    if set(a) != set(b):
        problems.append(
            f"different weight LEVELS: {ref_name} has "
            f"{', '.join(sorted(a)) or 'none'}; {arm_name} has "
            f"{', '.join(sorted(b)) or 'none'}")
    for level in sorted(set(a) & set(b)):
        # DEPTH ONLY. `width` and `datawidth` are the treatment.
        if a[level]["depth"] != b[level]["depth"]:
            problems.append(
                f"{level}.depth: {ref_name}={a[level]['depth']} "
                f"{arm_name}={b[level]['depth']}")
    if problems and not disabled:
        raise ValueError(
            f"{arch}: the two arms do NOT hold the same amount of silicon -- "
            + "; ".join(problems) + ".\n"
            f"  prompt_2's fairness rule is that both arms declare the same "
            f"LEVELS at the same DEPTH; only `width:` (from THE WIDTH TABLE,\n"
            f"  per arm) and `datawidth:` differ, and CACTI never sees "
            f"`datawidth`. A DEPTH difference is priced by Accelergy as real\n"
            f"  silicon one arm does not have, which is the defect that "
            f"invalidated the previous sweep (FINDINGS 7.8). The comparison\n"
            f"  is void; fix the configuration rather than correcting the "
            f"energy. A study that varies depth between the arms on purpose\n"
            f"  sets ECC_DISABLE_ASSERT_PAIR_GEOMETRY=1 (env.sh section 5), "
            f"which disables THIS check and nothing else.")
    if problems and disabled:
        print(f"  [pair-geometry] {arch}: ECC_DISABLE_ASSERT_PAIR_GEOMETRY=1 -- "
              f"depth differs and is ALLOWED: " + "; ".join(problems))
    return {level: {"depth": a[level]["depth"],
                    f"{ref_name}_width": a[level]["width"],
                    f"{arm_name}_width": b[level]["width"],
                    ref_name: a[level]["datawidth"],
                    arm_name: b[level]["datawidth"],
                    f"{ref_name}_weights": a[level]["weights"],
                    f"{arm_name}_weights": b[level]["weights"],
                    "capacity_ratio": (b[level]["weights"] / a[level]["weights"]
                                       if a[level]["weights"] else 0.0)}
            for level in sorted(set(a) & set(b), key=list(a).index)}

def _patched_text(arch, cfg, apply_per_arch=True, quiet=False):
    text = arch_source(arch, cfg).read_text()
    text = _patch_dram_depth(text, cfg.dram_depth)
    if cfg.force_technology:
        text = _force_technology(text, cfg.force_technology)
    if apply_per_arch and cfg.force_datawidth:
        text = _force_datawidth(text, cfg.force_datawidth, None if quiet else arch)
    if apply_per_arch and cfg.acc_bits_override is not None:
        text = _force_acc_bits(text, cfg.acc_bits_override, arch, quiet)
    # prompt_2: THE WIDTH TABLE (width + depth + datawidth together), then the
    # depth ladder. Together because a level's width is chosen FROM the
    # datawidth that level ends up storing -- they are one decision, and the
    # `width % datawidth == 0` constraint is then satisfied by construction
    # instead of being checked after the fact. UNCONDITIONAL: the table is a
    # property of the study, not a knob, so the 8-bit reference arm is
    # reshaped too (and must be, or its depths would not match the arm it is
    # compared with). The ladder runs after, so it multiplies the renormalised
    # depth rather than the published one.
    if apply_per_arch:
        text = _set_weight_geometry(text, getattr(cfg, "weight_datawidth", None),
                                    getattr(cfg, "weight_datawidth_levels", ()),
                                    cfg.weight_width_glb_mult,
                                    cfg.weight_capacity_scope, arch, quiet,
                                    weight_bits=cfg.weight_bits)
    # BOTH depth knobs run AFTER the reshape, on the renormalised depth. They
    # used to straddle it (capacity before, ladder after), which was harmless
    # only while the reshape was usually a no-op: once THE WIDTH TABLE applies
    # to every run, scaling before renormalising rounds twice and the two
    # knobs stop meaning the same thing at the same scale.
    if apply_per_arch and cfg.weight_capacity_scale != 1.0:
        text = _scale_weight_capacity(text, cfg.weight_capacity_scale,
                                      cfg.weight_capacity_scope, arch, quiet)
    if apply_per_arch and getattr(cfg, "weight_depth_scale", 1.0) != 1.0:
        text = _scale_weight_depth(text, cfg.weight_depth_scale,
                                   cfg.weight_depth_levels,
                                   cfg.weight_capacity_scope, arch, quiet)
    if apply_per_arch and cfg.weight_factor_relax:
        text = _relax_weight_factors(text, arch, quiet)
    # AFTER the relax, deliberately. The relax frees M/C/R/S on the weight
    # levels so the TILE can grow into the room; this then pins every
    # dimension at the levels the free-set does not name. `weights_spad` keeps
    # M and C free in the free-set precisely so the two levers compose instead
    # of cancelling.
    if apply_per_arch and getattr(cfg, "mapspace_constrain", False):
        text = _constrain_mapspace(text, arch, quiet)
    # ---------------------------------------------- prompt_7 Phase C: TIME
    # Three declarations that give the MAPPER what only the evaluator had.
    # They run AFTER the geometry so a bit-aware port is scaled from the width
    # table's final numbers, and BEFORE the NoC so every one of them is hashed.
    #
    # C1.1 -- the off-chip speed limit, study-wide. Unconditional like
    # `_patch_dram_depth`: it is a property of the modelled system, not of an
    # arm, and every arm shares it.
    text = _declare_dram_bandwidth(text, cfg.dram_items_per_cycle_for(arch),
                                   arch, quiet)
    if apply_per_arch:
        # C1.2 -- what THIS boundary declares it moves less of. Per-arm by
        # construction: it is one of the three axes that make two boundaries
        # two chips (prompt_7 6.4).
        text = _declare_bw_scale(text, cfg.arm_bw_factors_for(arch), arch, quiet)
        # C1.3 -- and how fast the narrowed levels' own ports then run. Only
        # the levels this arm narrows; a level still holding 8-bit weights
        # moves the same bits per cycle it always did.
        text = _bitaware_onchip_bandwidth(
            text, tuple(getattr(cfg, "weight_datawidth_levels", ()) or ()),
            cfg.onchip_bw_bitaware_factor(), arch, quiet)
    # C1.7 -- the declared bank count, study-wide. An energy declaration, not
    # a timing one, but it belongs to the same cold pass.
    text = _bank_geometry(text, arch, quiet)
    # Last, so the coefficients land on the final text and are hashed by
    # arch_fingerprint(). Applied regardless of apply_per_arch: the NoC model
    # is a study-wide treatment, not a per-architecture no-op candidate.
    text = _inject_noc(text, arch, cfg)
    return text


# ------------------------------------------------------ prompt_6: the ERT arm
def ert_bump(arch, cfg):
    """The ERT delta this configuration's arm applies, or None for the reference.

    prompt_6 5.1, recomputed at run time -- the table in the plan is the check,
    not the source:

        access action   delta = E_w x block_size    block_size = width / datawidth,
                                                    read off THAT level in THIS
                                                    arm's patched arch
        leak            delta = idle_per_cycle      Timeloop x instances x cycles

    The level, counter and action come from the Placement (`recon.ert_arm_spec`),
    the two DC terms from `ecc.load_recon_energy`, E_w from `recon.ert_deltas`.
    `load_recon_energy`, NOT `load_recon_terms`: the toll the mapper optimises
    against has to be the toll the evaluator bills, so `ECC_RECON_PJ` must reach
    both or the two disagree. It is therefore hashed into the fingerprint like
    every other ERT input, so changing a recon energy invalidates the mapper
    cache and the arm is re-solved instead of being read back at the old toll.
    The result is what `arch_fingerprint()` hashes (RULE 4.4.5, defence 2),
    what `timeloop.ErtTables` patches into the base table, and what the
    read-back assertion compares a cache entry against.
    """
    arm = cfg.ert_arm() if hasattr(cfg, "ert_arm") else None
    if arm is None:
        return None
    geo = patched_weight_geometry(arch, cfg)
    level = arm["level"]
    if level not in geo:
        raise ValueError(f"ERT arm {arm['key']}: {arch} has no weight-carrying "
                         f"level {level!r} in its patched YAML; it has "
                         f"{', '.join(geo) or 'none'}")
    g = geo[level]
    if not g["width"] or not g["datawidth"] or g["width"] % g["datawidth"]:
        raise ValueError(f"ERT arm {arm['key']}: {level} declares width {g['width']} "
                         f"and datawidth {g['datawidth']}; block_size is undefined")
    block_size = g["width"] // g["datawidth"]
    from . import ecc as _ecc            # lazily: ecc needs pandas, the reference arm does not
    from . import recon as _recon
    inc, idle, prov = _ecc.load_recon_energy(cfg)
    gran = _recon.Granularity(cfg.code_n, cfg.code_k, cfg.weight_bits,
                              cfg.recon_granularity)
    d = _recon.ert_deltas(inc, idle, gran, block_size,
                          getattr(cfg, "recon_clock_gating_pct", 0.0))
    d.update(placement=arm["key"], level=level, counter=arm["counter"],
             action=arm["action"], narrow_levels=list(arm["narrow_levels"]),
             level_width=g["width"], level_datawidth=g["datawidth"],
             code=f"BCH({cfg.code_n},{cfg.code_k})", dc_provenance=prov)
    return d


def ert_hash_view(bump):
    """The part of an ERT bump that identifies the ARCHITECTURE the mapper sees:
    level, every patched action and its delta at full precision. Two arms
    differing only in the toll must hash differently; two spellings of the
    same toll must not."""
    if bump is None:
        return None
    return {"placement": bump["placement"], "level": bump["level"],
            "actions": {bump["action"]: bump["access_delta_pj"],
                        "leak": bump["leak_delta_pj"]}}


# --------------------------------------------------------------- fingerprints
def hashable_arch_text(text):
    """`text` with everything Timeloop never reads taken out: comments, trailing
    whitespace, blank lines.

    WHY THE FINGERPRINT MUST NOT SEE A COMMENT (2026-09-13). `arch_fingerprint`
    hashes the patched YAML, and it hashed it byte for byte -- so correcting a
    comment that had gone stale re-keyed every mapper cache in the study and
    threw away hours of solved mappings for a documentation edit. That made the
    two things this project asks of an architecture file pull against each
    other: `arch_paper.yaml` is meant to carry the citation for every number it
    declares, and keeping those comments honest was priced at a full re-map.

    Timeloop is handed the file with its comments intact. It parses YAML, so a
    comment reaches no mapping decision and no energy. Hashing one therefore
    reported a change in the ARCHITECTURE that had not happened -- the exact
    false positive the fingerprint exists to avoid the mirror image of.

    A geometry edit still colds the cache, which is the whole point: `depth: 64`
    to `depth: 96` survives this normalisation and lands in the hash. What no
    longer does is the comment beside it.

    Quote-aware, so a `#` inside a YAML scalar (`name: "a#b"`) is data, not the
    start of a comment. No arch file has one today; one added later must not
    silently change meaning.
    """
    out = []
    for line in text.splitlines():
        quote = None
        cut = None
        for i, ch in enumerate(line):
            if quote is not None:
                if ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
            elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
                cut = i
                break
        if cut is not None:
            line = line[:cut]
        line = line.rstrip()
        if line:
            out.append(line)
    return "\n".join(out) + "\n"


def arch_fingerprint(arch, cfg, mapper_settings=None):
    """Content hash of the architecture the mapper will actually see.

    THE PROBLEM THIS SOLVES. The treatment slug (`paper`, `opt-energy`, ...)
    says what KIND of run this is, not what the architecture IS. Editing
    `archs/eyeriss_v2_like/arch_paper.yaml` -- correcting a scratchpad depth,
    say -- left the slug unchanged, so the next run happily reused mappings
    computed for the OLD geometry and reported them as the new design. That is
    silent, and it is exactly the class of error this task exists to remove.

    The hash covers the fully patched YAML text plus every mapper setting that
    could change which mapping comes back. Change either and the cache moves.

    THE TEXT IS NORMALISED FIRST (`hashable_arch_text`, 2026-09-13): comments,
    trailing whitespace and blank lines are stripped, because Timeloop parses
    YAML and none of them reaches a mapping decision. A geometry edit still
    colds the cache; correcting the comment beside it no longer does. That one
    change re-keyed every cache on disk once, on purpose -- the old directories
    are untouched and simply unused.

    `mapper_settings` defaults to `cfg.mapper_settings()`. It is a parameter so
    a caller that has already resolved the effective thread count can pass the
    real one rather than the configured `None`.
    """
    text = hashable_arch_text(_patched_text(arch, cfg, quiet=True))
    # globals.yaml is a mapper input too -- it sets the node DRAM is costed at,
    # which changes the energy the mapper is optimising. Hashing the values
    # rather than reading the file keeps this usable before it is written.
    globals_view = {
        "technology": cfg.force_technology or load_standard()["study"]["technology"],
        # prompt_7 C1.5: THIS design's rate. `globals_<arch>.yaml` carries it
        # into the mapper, so it has to be what is hashed -- hashing the study
        # default would give two designs at two clocks one fingerprint.
        "global_cycle_seconds": cfg.cycle_seconds_for(arch),
    }
    blob = {
        "arch": arch,
        "arch_yaml": text,
        "globals": globals_view,
        "mapper": mapper_settings or cfg.mapper_settings(),
        # The locally authored Accelergy components (archs/_shared/components)
        # set the per-access energies the mapper optimises against. Added
        # 2026-09-08: a register-write correction to regfile_decoded.yaml
        # changed every scratchpad's ERT without moving a single arch YAML,
        # and nothing would otherwise have told the cache.
        "components": components_digest(),
        "workload_shape_template_version": 1,
    }
    # prompt_7 C1.8. THE FINGERPRINT HASHES THE ARCHITECTURE, NOT THE PRICE
    # LIST Accelergy derives from it -- so a corrected ESTIMATOR (the Neurosim
    # plug-in that answered 0 pJ for every address generator) changes every
    # energy in the cache while leaving the directory it is stored under
    # identical. `ECC_ENERGY_MODEL_REV` is the deliberate cold, and until Phase
    # C it reached only `Config.fingerprint()` -- which labels a RESULT and
    # names no cache directory, so the knob re-labelled results while the cache
    # it was meant to invalidate stayed warm. Appended only when set, so EMPTY
    # still hashes byte-identically to every directory that predates it.
    if getattr(cfg, "energy_model_rev", ""):
        blob["energy_model_rev"] = cfg.energy_model_rev
    # prompt_7 C1.6. The MAC price the MAPPER optimises against. Until Phase C
    # `ECC_MAC_PJ_OVERRIDE` was evaluator-only: the mapper priced a MAC from
    # Accelergy's ERT at 1.16877 pJ and the report then rescaled Compute to
    # 0.23, so under ECC_OPT_METRIC=edp the plan was chosen for a machine whose
    # arithmetic cost 5x what the study charges. `timeloop.ErtTables` now
    # supplies the price, and a supplied price is part of the architecture.
    if getattr(cfg, "mac_pj_override", None) is not None:
        blob["mac_pj_override"] = float(cfg.mac_pj_override)
    # prompt_6 RULE 4.4.5, defence 2: an ERT arm's toll is part of what the
    # mapper optimises against. Only added when there IS one, so every
    # reference fingerprint on disk is unchanged.
    bump = ert_bump(arch, cfg)
    if bump is not None:
        blob["ert"] = ert_hash_view(bump)
    blob = json.dumps(blob, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def components_digest():
    """sha1 over the shared Accelergy component files, by name and content."""
    h = hashlib.sha1()
    for f in sorted(ARCH_COMPONENTS.glob("*.yaml")) if ARCH_COMPONENTS.exists() else []:
        h.update(f.name.encode("utf-8"))
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def effective_variant(arch, cfg):
    """The cache slug for THIS architecture, skipping no-op treatments.

    Two per-architecture treatments can turn out to be no-ops:

    * `paper` fidelity, on a design that has no `arch_paper.yaml`.
    * Forcing the datawidth to the weight width, on a design that already
      declares it. Those architectures keep their existing mapper cache, and
      only the designs the force actually rewrites pay for a fresh map.

    The globals.yaml and mapper knobs cannot be skipped this way: they change
    DRAM cost, or the search itself, for every design at once.
    """
    parts = list(cfg.global_variant_parts)
    # prompt_7 C1.5: THIS design's clock, not the study's. `ECC_ARCH_CLOCK_MHZ`
    # runs Eyeriss v1 at its published 200 MHz while the rest of the study stays
    # at the 1 GHz model default, so the rate is a per-architecture treatment
    # and the slug has to be written where the architecture is known. The
    # historical `1e-9` contributes nothing, so every directory on disk keeps
    # its name.
    clk = cfg.cycle_seconds_for(arch)
    if clk != "1e-9":
        parts.append(f"clk{clk}")
    if cfg.arch_fidelity != "stock" and paper_source(arch) is not None:
        parts.append(cfg.arch_fidelity)
    if cfg.force_datawidth:
        with_force = _patched_text(arch, cfg, apply_per_arch=True, quiet=True)
        without = _patched_text(arch, cfg, apply_per_arch=False, quiet=True)
        if with_force != without:
            parts.append(f"dw{cfg.force_datawidth}")
    if cfg.weight_capacity_scale != 1.0:
        # A design whose weight levels all round back to their declared depth
        # is NOT dilated, and must keep reading the undilated cache rather
        # than paying for a fresh map of an unchanged architecture -- the same
        # trap `_patch_dram_depth`'s docstring records.
        base = _set_weight_geometry(arch_source(arch, cfg).read_text(),
                                    getattr(cfg, "weight_datawidth", None),
                                    getattr(cfg, "weight_datawidth_levels", ()),
                                    getattr(cfg, "weight_width_glb_mult", 4),
                                    cfg.weight_capacity_scope, arch, quiet=True,
                                    weight_bits=cfg.weight_bits)
        if _scale_weight_capacity(base, cfg.weight_capacity_scale,
                                  cfg.weight_capacity_scope, arch,
                                  quiet=True) != base:
            parts.append(f"wcap{cfg.weight_capacity_scale:g}"
                         + ("-shared" if cfg.weight_capacity_scope == "shared" else ""))
    # THE WIDTH TABLE is applied to every run, so it is in every slug -- no
    # no-op rule, because there is no configuration in which it is off. It
    # marks the boundary in `ls`: a directory without it predates 2026-09-12
    # and was mapped on the published word shape.
    parts.append(f"wt{code_widths.base_width(cfg.weight_bits)}"
                 + (f"x{cfg.weight_width_glb_mult}"
                    if cfg.weight_width_glb_mult != 4 else ""))
    if getattr(cfg, "weight_depth_scale", 1.0) != 1.0:
        # Same no-op rule as the capacity scale, measured on the RESHAPED text
        # -- the ladder multiplies the renormalised depth.
        base = _set_weight_geometry(arch_source(arch, cfg).read_text(),
                                    getattr(cfg, "weight_datawidth", None),
                                    getattr(cfg, "weight_datawidth_levels", ()),
                                    getattr(cfg, "weight_width_glb_mult", 4),
                                    cfg.weight_capacity_scope, arch, quiet=True,
                                    weight_bits=cfg.weight_bits)
        if _scale_weight_depth(base, cfg.weight_depth_scale,
                               cfg.weight_depth_levels,
                               cfg.weight_capacity_scope, arch,
                               quiet=True) != base:
            parts.append(f"wdepth{cfg.weight_depth_scale:g}"
                         + ("-" + "+".join(cfg.weight_depth_levels)
                            if cfg.weight_depth_levels else ""))
    if getattr(cfg, "weight_datawidth", None) is not None:
        # No-op rule: the BASELINE/EMBEDDED arm may legitimately be spelled
        # `ECC_WEIGHT_DATAWIDTH=8` on a design whose levels already store 8-bit
        # weights, and that arm must then READ THE SAME CACHE as leaving the
        # knob empty -- otherwise the two arms of a pair would be compared
        # across two mapper caches of one identical architecture. Measured by
        # diffing the arm's geometry against the 8-bit arm's.
        src_text = arch_source(arch, cfg).read_text()
        eight = _set_weight_geometry(src_text, None, (),
                                     getattr(cfg, "weight_width_glb_mult", 4),
                                     cfg.weight_capacity_scope, arch, quiet=True,
                                     weight_bits=cfg.weight_bits)
        dw_levels = tuple(getattr(cfg, "weight_datawidth_levels", ()) or ())
        armed = _set_weight_geometry(src_text, cfg.weight_datawidth, dw_levels,
                                     getattr(cfg, "weight_width_glb_mult", 4),
                                     cfg.weight_capacity_scope, arch, quiet=True,
                                     weight_bits=cfg.weight_bits)
        if armed != eight:
            # Same spelling as `wdepth<scale>-<levels>`: an arm narrowing
            # only `filter_glb` is a different architecture from one
            # narrowing every weight level, and the directory name says so.
            parts.append(f"wdw{cfg.weight_datawidth}"
                         + ("-" + "+".join(dw_levels) if dw_levels else ""))
    if getattr(cfg, "mapspace_constrain", False) and arch in MAPSPACE_FREE_LEVELS:
        # A constrained loop nest is a different MAPSPACE and a different
        # DATAFLOW, so it is a different architecture to the mapper and gets
        # its own cache. A design with no free-set entry is NOT constrained and
        # keeps its existing cache rather than being silently mislabelled.
        parts.append("mcons")
    if cfg.weight_factor_relax:
        # Same no-op rule as the capacity scale: a design with no `factors:`
        # pin on a weight-indexing dimension is not relaxed by this and keeps
        # its existing cache rather than paying for a fresh map of an unchanged
        # architecture. `simple_weight_stationary` is exactly that case -- its
        # weight levels pin nothing a weight tile is indexed by, so its tile is
        # NOT capped by the dataflow constraints and this lever cannot help it.
        base = arch_source(arch, cfg).read_text()
        if _relax_weight_factors(base, arch, quiet=True) != base:
            parts.append("wrelax")
    part = cfg.mapper_arm_slug() if hasattr(cfg, "mapper_arm_slug") else None
    if part is not None:
        # prompt_6 RULE 4.4.5, defence 1. Never a no-op: the arm is the
        # directory. `ert-<key>-<level>-<action>` where the boundary declares
        # an ERT bump -- unchanged, so every entry on disk stays a hit -- and
        # prompt_7 B2's `arm-<key>` where it does not, because R1's patched
        # YAML IS the reference's until Phase C1.2 and without a slug of its
        # own the two would share one directory.
        parts.append(part)
    return "stock" if not parts else "__".join(parts)



# ------------------------------------------ standardized-comparison validation
#: Which study-wide precision a level's dataspaces are billed at.
def _expected_operand_bits(keep, bypass, std):
    """The operand width a level holding Weights and/or Inputs must declare.

    Both are standardized to the same value today, so this is normally one
    number. It is written as a lookup anyway because the two are *conceptually*
    distinct -- Task 1 says "keep these precisions explicit and distinct" -- and
    a future run that quantizes weights to 4 bits while leaving activations at
    8 must fail validation rather than silently pick one.
    """
    holds_w = ("Weights" in keep) or (not keep and "Weights" not in bypass)
    holds_i = ("Inputs" in keep) or (not keep and "Inputs" not in bypass)
    wanted = set()
    if holds_w:
        wanted.add(int(std["weight_bits"]))
    if holds_i:
        wanted.add(int(std["activation_bits"]))
    return wanted


def validate_arch(arch, cfg):
    """Check one architecture against archs/_shared/standard.yaml.

    Returns `{"arch", "violations", "notes", "facts"}`.

    A VIOLATION is a place where the design breaks the shared contract -- a
    different operand width, a different DRAM geometry, a psum level billed
    below its own accumulator. Those are bugs: they make a cross-architecture
    number mean something other than what it claims.

    A NOTE is a legitimate difference the contract expects and wants visible --
    a design-specific accumulator width, a declared `psum-width-ok` level, a
    node deviation. Notes never fail the run.

    FACTS are the quantities Task 1 says must be PRESERVED rather than
    equalised: topology, capacity, weight bypass rules. They are printed so a
    reader can confirm the standardization did not quietly flatten them.
    """
    std = load_standard()["study"]
    spec = arch_standard(arch)
    prov = load_provenance().get(arch)
    text = arch_source(arch, cfg).read_text()

    violations, notes, facts = [], [], {}

    if spec is None:
        violations.append(
            f"{arch} is not declared in archs/_shared/standard.yaml. Add it "
            f"there (with its accumulator width and a citation) before "
            f"comparing it against the others.")
        return {"arch": arch, "violations": violations, "notes": notes, "facts": facts}
    if prov is None:
        notes.append(f"no entry in archs/_shared/provenance.yaml -- the declared "
                     f"numbers are not traced to a source.")

    acc, acc_evidence = accumulator_bits(arch, cfg)
    facts["source"] = str(arch_source(arch, cfg))
    facts["origin"] = spec.get("source", "?")
    facts["label"] = spec.get("label", arch)
    facts["accumulator_bits"] = acc
    facts["accumulator_evidence"] = acc_evidence
    facts["citation"] = spec.get("citation", "")

    if spec.get("source") == "locally_authored":
        notes.append("locally authored reference dataflow -- reproduces no paper, "
                     "so its geometry is a design choice, not a citation.")
    paper_tech = spec.get("paper_technology")
    if paper_tech and paper_tech != std["technology"]:
        notes.append(f"published at {paper_tech}, modelled at {std['technology']}. "
                     f"Deliberate: a mixed-node comparison would report silicon as "
                     f"architecture.")

    # ---- technology --------------------------------------------------------
    wanted_tech = cfg.force_technology or std["technology"]
    declared = sorted(set(re.findall(r'technology:\s*"?([\w.]+)"?', text)))
    facts["technology_declared"] = declared
    for node in declared:
        if node != wanted_tech:
            violations.append(f"declares technology {node!r}; the contract says "
                              f"{wanted_tech!r}")

    # ---- interconnect ------------------------------------------------------
    # Same rule as the process node: the wire constant is study-wide metal and
    # may not differ per design; only the switching STRUCTURE is the design's
    # own, and it must be declared and cited in archs/_shared/noc.yaml.
    if cfg.noc_enabled:
        noc_entry = (load_noc().get("architectures") or {}).get(arch)
        if noc_entry is None:
            violations.append(
                f"{arch} has no entry in archs/_shared/noc.yaml, so it would map "
                f"with a free interconnect while every other design pays for one.")
        else:
            if "wire_pj_per_bit_mm" in noc_entry:
                violations.append(
                    f"noc.yaml gives {arch} its own wire_pj_per_bit_mm. The wire "
                    f"constant is shared study-wide (noc.yaml `shared:`); a "
                    f"per-design value would report metal as architecture.")
            terms = noc_terms(arch, cfg)
            wire, router, ingress = terms["wire"], terms["router"], terms["ingress"]
            levels = noc_levels(arch)
            present = spatial_containers(text)
            if levels is not None:
                for name in levels:
                    if name not in present:
                        violations.append(
                            f"noc.yaml names `{name}` as a NoC level but this design's "
                            f"spatial containers are {present}; a typo here silently "
                            f"leaves the interconnect free.")
            charged = present if levels is None else [n for n in present if n in levels]
            per_level = {n: (terms["levels"][n] if terms["levels"] else (router, ingress))
                         for n in charged}
            widths = terms["tile_width_um"]
            for name in widths:
                if name not in charged:
                    violations.append(
                        f"noc.yaml declares tile_width_um for `{name}`, which is not one "
                        f"of this design's NoC levels {charged}.")
            if widths and set(widths) != set(charged):
                violations.append(
                    f"noc.yaml declares tile_width_um for {sorted(widths)} but this design's "
                    f"NoC levels are {charged}: a container's attributes are inherited, so "
                    f"an undeclared inner level would take the outer pitch. Declare all or none.")
            facts["noc"] = {"structure": noc_entry.get("structure"),
                            "wire_pj_per_bit_mm": wire, "source": noc_entry.get("source"),
                            "router_pj_per_flit": terms["router_pj_per_flit"],
                            "operands_per_flit": terms["operands_per_flit"],
                            "pe_latch_pj": pe_latch_pj(cfg),
                            "levels": {n: {"router_pj": r, "ingress_pj": g,
                                           "tile_width_um": widths.get(n)}
                                       for n, (r, g) in per_level.items()},
                            "datapath_levels": [n for n in present if n not in charged],
                            "components_digest": components_digest()}
            notes.append(
                f"interconnect: {noc_entry.get('structure', '?')} -- wire {wire:g} pJ/b/mm "
                f"(shared); " + "; ".join(
                    f"{n}: router {r:g} pJ, ingress {g:g} pJ"
                    + (f", hop {widths[n]:g} um (declared)" if n in widths else ", hop from area")
                    for n, (r, g) in per_level.items())
                + (f"; datapath, not NoC: {[n for n in present if n not in charged]}"
                   if len(charged) != len(present) else "")
                + " (archs/_shared/noc.yaml)")
            if noc_entry.get("source") in ("published", "reference_design") \
                    and not noc_entry.get("citation_url"):
                notes.append("noc.yaml entry cites no URL for its interconnect structure.")

    # ---- per level ---------------------------------------------------------
    levels, weight_levels, fanout = [], [], 1
    for name, body in _blocks(text):
        mesh_x, mesh_y = _num(body, "meshX"), _num(body, "meshY")
        if mesh_x or mesh_y:
            fanout *= (mesh_x or 1) * (mesh_y or 1)
            continue

        cls = re.search(r"(?:class|subclass):\s*(\S+)", body)
        cls = cls.group(1) if cls else ""
        depth, width, dw = _num(body, "depth"), _num(body, "width"), _num(body, "datawidth")

        if cls == "DRAM":
            want = std["dram"]
            for key, got, exp in (("width", width, want["width_bits"]),
                                  ("datawidth", dw, want["datawidth_bits"]),
                                  ("depth", depth, want["depth_words"])):
                if got is not None and int(got) != int(exp):
                    violations.append(f"DRAM {key}={got}; the contract says {exp}")
            facts["dram"] = {"width": width, "datawidth": dw, "depth": depth}
            continue

        adder = _num(body, "adder_width")
        mult = _num(body, "multiplier_width")
        if adder or mult:
            facts["mac"] = {"multiplier_width": mult, "adder_width": adder}
            if mult is not None and int(mult) != int(std["weight_bits"]):
                violations.append(f"mac.multiplier_width={mult}; the contract "
                                  f"standardizes operands at {std['weight_bits']}b")
            if acc is not None and adder is not None and int(adder) != int(acc):
                violations.append(
                    f"mac.adder_width={adder} but this design's declared "
                    f"accumulator is {acc}b. Accumulator STORAGE and accumulator "
                    f"ARITHMETIC must agree -- the shipped simba_like got this "
                    f"wrong in exactly this way.")
            continue

        if depth is None or dw is None:
            continue

        keep = re.search(r"keep:\s*\[([^\]]*)\]", body)
        bypass = re.search(r"bypass:\s*\[([^\]]*)\]", body)
        keep = [s.strip() for s in keep.group(1).split(",")] if keep else []
        bypass = [s.strip() for s in bypass.group(1).split(",")] if bypass else []
        holds_outputs = ("Outputs" in keep) or (not keep and "Outputs" not in bypass)
        holds_weights = ("Weights" in keep) or (not keep and "Weights" not in bypass)
        ok_note = re.search(r"#\s*psum-width-ok:\s*(.+)", body)

        # Timeloop's own requirement, checked here so a bad YAML fails at
        # validate time rather than aborting the mapper on every layer.
        if width is not None and width % dw != 0:
            violations.append(f"{name}: width {width} is not a multiple of "
                              f"datawidth {dw}; Timeloop asserts "
                              f"width % datawidth == 0 and will abort.")

        if holds_outputs and not (keep and keep != ["Outputs"] and ok_note is None):
            pass  # handled below

        if keep == ["Outputs"] or (holds_outputs and not holds_weights and
                                   "Inputs" not in keep):
            # a dedicated partial-sum level: must be at the accumulator width
            if ok_note:
                notes.append(f"{name} carries Outputs at {dw}b, below the {acc}b "
                             f"accumulator, and declares why: {ok_note.group(1).strip()}")
            elif acc is not None and int(dw) != int(acc):
                violations.append(
                    f"{name} carries partial sums at datawidth {dw} but this "
                    f"design accumulates at {acc}b. Timeloop packs "
                    f"width/datawidth values per word, so this understates both "
                    f"the per-psum energy and the access count. Declare {acc}, "
                    f"or add a `# psum-width-ok: <reason>` comment if the level "
                    f"really holds requantized values.")
        else:
            wanted = _expected_operand_bits(keep, bypass, std)
            if holds_outputs and wanted:
                # A SHARED level. Timeloop allows one datawidth per level, so a
                # buffer holding both operands and partial sums must bill one of
                # them wrongly. Billing the psums at the operand width is the
                # defect that made the stock eyeriss_like look cheaper than
                # eyeriss_v2_like on every workload: 16-bit partial sums packed
                # eight to a 64-bit word and charged as bytes. The fix is to
                # split the level by dataspace, which every arch_paper.yaml
                # does. A level that legitimately holds requantized activations
                # says so with `# psum-width-ok:` and is only a note.
                if ok_note:
                    notes.append(
                        f"{name} holds Outputs alongside operands at {dw}b and "
                        f"declares why: {ok_note.group(1).strip()}")
                elif acc is not None and int(dw) < int(acc):
                    violations.append(
                        f"{name} holds partial sums AND operands at one datawidth "
                        f"({dw}b), below this design's {acc}b accumulator. "
                        f"Timeloop packs width/datawidth values per word, so the "
                        f"psums are billed as operands -- understating both their "
                        f"per-access energy and the access count. Split the level "
                        f"by dataspace (an operand half and a psum half at {acc}b), "
                        f"or add `# psum-width-ok: <reason>` if it really holds "
                        f"requantized values.")
                else:
                    notes.append(
                        f"{name} holds Outputs alongside operands; Timeloop allows "
                        f"one datawidth per level, so it is billed at the operand "
                        f"width {sorted(wanted)}.")
            if wanted and int(dw) not in wanted:
                violations.append(
                    f"{name} declares datawidth {dw}; it holds {keep or 'everything'} "
                    f"and the contract standardizes those operands at "
                    f"{sorted(wanted)}. THIS IS THE DEFECT THE CONTRACT EXISTS "
                    f"FOR: an operand width is a study-wide choice, not an "
                    f"architectural difference.")

        entries = depth * max(1, (width // dw) if (width and dw) else 1)
        levels.append({"name": name, "class": cls, "depth": depth, "width": width,
                       "datawidth": dw, "entries_each": entries,
                       "instances": fanout, "keep": keep, "bypass": bypass})
        if holds_weights:
            weight_levels.append({"name": name, "entries_each": entries,
                                  "instances": fanout,
                                  "entries_total": entries * fanout})

    facts["levels"] = levels
    facts["total_fanout"] = fanout
    facts["weight_levels"] = weight_levels
    facts["weight_capacity"] = sum(w["entries_total"] for w in weight_levels)
    facts["glb_keeps_weights"] = any(w["instances"] == 1 for w in weight_levels)
    return {"arch": arch, "violations": violations, "notes": notes, "facts": facts}

# ------------------------------------------------------- mapper search depth
def loop_levels(text):
    """How many loop levels Timeloop will build for this architecture.

    One per storage component plus one per spatial container -- exactly the
    `L0..Ln` the mapper prints. A `!Parallel` group contributes one level per
    branch (its three scratchpads are three levels, not one), and a `!Nothing`
    branch contributes none.

    This is what `cfg.victory_for()` scales mapper effort by, so that a deep
    hierarchy is not searched less thoroughly than a shallow one and the
    difference then reported as an architecture result.
    """
    storage = spatial = 0
    for _name, body in _blocks(text):
        if body.startswith("Nothing"):
            continue
        if re.search(r"\bmesh[XY]:\s*\d+", body):
            spatial += 1
            continue
        if re.search(r"\bdepth:\s*\d+", body):
            storage += 1
    return storage + spatial


def arch_levels(arch, cfg):
    """`loop_levels` for the arch.yaml this configuration actually maps."""
    return loop_levels(_patched_text(arch, cfg, quiet=True))


def _content_tag(text):
    """8 hex of the content -- the name a deterministic file is written under."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _write_once(dst, text):
    """Write `text` to `dst` if `dst` does not already exist. NEVER REPLACE IT.

    THE BUG THIS EXISTS FOR (2026-09-13, prompt_7 C2, one job of 186).
    `_write_atomic` wrote a private temp file and `os.replace`d it onto a
    SHARED path. That is atomic in POSIX terms -- the path always points at a
    complete file -- and it is still not safe here, because every one of 186
    concurrent SLURM jobs replaced the SAME two paths (`globals_<arch>.yaml`
    and the patched `arch_<arch>_patched__<variant>.yaml`) within seconds of
    each other. On Lustre a client that has already looked the path up holds a
    handle to the OLD inode, and a replacement leaves that handle stale:

        job 41920766, line 35   globals_eyeriss_like_wglb.yaml: technology=45nm
        job 41920766, line 78   FileNotFoundError: ... globals_eyeriss_like_wglb.yaml

    written and then missing, in one process, eight seconds apart. One failed
    job out of 186 was enough to leave the dependent eval on
    `DependencyNeverSatisfied` and the whole matrix unreadable.

    THE FIX IS THE NAME, NOT THE WRITE. Both files are a pure function of
    (architecture, configuration), so they are content-addressed: identical
    content means an identical name, and a name that exists already holds the
    bytes this caller wanted. `O_CREAT | O_EXCL` then creates it exactly once,
    atomically, and NOTHING EVER REPLACES AN EXISTING FILE -- so no reader can
    be holding a handle to something that is about to be unlinked. A second
    writer losing the race is not an error; it is the normal case.

    Fingerprints do not move: `arch_fingerprint()` hashes the CONTENT of these
    files (`globals_view`, `arch_yaml`), never their paths.
    """
    if dst.exists():
        return dst
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    with open(tmp, "w", newline="\n") as fh:
        fh.write(text)
    try:
        # O_EXCL via link(): create the name only if it is free, and never
        # clobber. `os.replace` would overwrite, which is the whole problem.
        os.link(tmp, dst)
    except FileExistsError:
        pass                       # another job wrote the same bytes first
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return dst


def _write_atomic(dst, text):
    """Replace `dst` with `text`, atomically. For a file only ONE process writes.

    Use `_write_once` for anything a SLURM array writes concurrently -- see the
    stale-handle failure recorded there. This remains for single-writer paths.
    """
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    with open(tmp, "w", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, dst)
    return dst


def patched_arch_path(arch, cfg):
    """Write and return the arch.yaml for this architecture + treatment.

    CONTENT-ADDRESSED and written ONCE (`_write_once`): every job of a SLURM
    array shares this path, and replacing it under a concurrent reader is what
    cost one job of 186 on 2026-09-13. The tag is 8 hex of the text, so two
    treatments that produce identical YAML share one file and a change makes a
    NEW one rather than overwriting the old.
    """
    text = _patched_text(arch, cfg)
    variant = effective_variant(arch, cfg)
    suffix = "" if variant == "stock" else f"__{variant}"
    dst = WORK / f"arch_{arch}_patched{suffix}__{_content_tag(text)}.yaml"
    return _write_once(dst, text)


def globals_text(cfg, arch):
    """The bytes THIS design's `globals.yaml` holds -- a pure function of
    (configuration, architecture), which is what lets it be content-addressed."""
    node = cfg.force_technology or load_standard()["study"]["technology"]
    return ("variables:\n"
            "  version: 0.4\n"
            f"  global_cycle_seconds: {cfg.cycle_seconds_for(arch)}\n"
            f'  technology: "{node}"\n')


def globals_path(arch, cfg):
    """Where THIS design's `globals.yaml` lives.

    ONE FILE PER DESIGN since prompt_7 C1.5, because `global_cycle_seconds` is
    no longer one number for the study: Eyeriss v1 runs at its published
    200 MHz while the rest stays at the 1 GHz model default. A shared file
    would clock every design on a multi-design figure at whichever rate was
    written last -- silently, and only for the designs that are not first.

    CONTENT-ADDRESSED since 2026-09-13: the name carries 8 hex of the bytes, so
    186 concurrent jobs writing "the same" file write the SAME NAME and nothing
    ever replaces a file a reader may be holding open. That replacement is what
    cost one job of 186 (`_write_once` records the failure).
    """
    return WORK / f"globals_{arch}_{_content_tag(globals_text(cfg, arch))}.yaml"


def write_globals(cfg, arch):
    """`globals.yaml` sets the node for anything OUTSIDE an arch container.

    DRAM sits above the accelerator container in every one of these designs, so
    it takes its technology from here, identically for every architecture. The
    CLOCK is per design (`Config.cycle_seconds_for`), which is why this takes
    one.

    THE DEFAULT USED TO BE 65nm while every accelerator container declared 45nm,
    so DRAM -- the level this study spends most of its energy in, and the level
    BCH parity lands on -- was costed at a different node from the logic it
    talks to. It was uniform across architectures, so it never showed up as an
    ordering error; it just made every absolute number and every savings
    percentage wrong by a fixed factor. The node now comes from
    archs/_shared/standard.yaml, which is the same file the architectures are
    validated against, so the two cannot drift apart again.
    """
    text = globals_text(cfg, arch)
    node = cfg.force_technology or load_standard()["study"]["technology"]
    return _write_once(globals_path(arch, cfg), text), node


# ---------------------------------------------------------------------- audit
_COMP_RE = re.compile(r"name:\s*(\S+)")


def _blocks(text):
    """Split an arch.yaml into (name, body) chunks, one per `- !` node."""
    out = []
    chunks = re.split(r"\n\s*-\s*!", text)
    for ch in chunks[1:]:
        m = _COMP_RE.search(ch)
        out.append((m.group(1) if m else "?", ch))
    return out


def _num(body, key):
    """First `key: <int>` in `body`, never from a comment.

    A comment that mentions a former value ("was meshX: 16") is prose, and
    the audit read it as the fanout once -- reporting 6,144 PEs for a design
    with 384. Comments are stripped line by line before matching; the
    `# psum-width-ok:` note is read by its own regex, not this one.
    """
    code = "\n".join(ln.split("#", 1)[0] for ln in body.split("\n"))
    m = re.search(rf"\b{key}:\s*(\d+)", code)
    return int(m.group(1)) if m else None


def audit(arch, cfg):
    """Return a dict of the declarations that drive a weight-energy study."""
    text = arch_source(arch, cfg).read_text()
    # The level count is taken from the PATCHED text, not this raw one: a stock
    # design that omits its DRAM depth grows a storage level when the patch adds
    # it, and the count has to match the L0..Ln the mapper will actually build.
    levels = arch_levels(arch, cfg)
    info = {
        "arch": arch,
        "source": str(arch_source(arch, cfg)),
        "fidelity": cfg.arch_fidelity,
        "loop_levels": levels,
        "victory": cfg.victory_for(levels),
        "adder_width": None,
        "technology": sorted(set(re.findall(r'technology:\s*"?([\w.]+)"?', text))),
        "levels": [],
        "spatial": [],
        "dram_datawidth": None,
        "dram_width": None,
        "glb_keeps_weights": None,
        "weight_capacity": 0,
        "weight_levels": [],
        "psum_datawidth": [],
        "psum_width_ok": {},
    }

    fanout = 1
    for name, body in _blocks(text):
        mesh_x = _num(body, "meshX")
        mesh_y = _num(body, "meshY")
        if mesh_x or mesh_y:
            n = (mesh_x or 1) * (mesh_y or 1)
            fanout *= n
            info["spatial"].append({"name": name, "meshX": mesh_x, "meshY": mesh_y,
                                    "instances": n, "cumulative_fanout": fanout})

        cls = re.search(r"(?:class|subclass):\s*(\S+)", body)
        cls = cls.group(1) if cls else ""
        adder = _num(body, "adder_width")
        if adder:
            info["adder_width"] = adder
        depth, width, dw = _num(body, "depth"), _num(body, "width"), _num(body, "datawidth")
        keep = re.search(r"keep:\s*\[([^\]]*)\]", body)
        bypass = re.search(r"bypass:\s*\[([^\]]*)\]", body)
        keep = [s.strip() for s in keep.group(1).split(",")] if keep else []
        bypass = [s.strip() for s in bypass.group(1).split(",")] if bypass else []

        if cls == "DRAM":
            info["dram_datawidth"] = dw
            info["dram_width"] = width
            continue
        if depth is None:
            continue

        entries_per_word = max(1, (width // dw) if (width and dw) else 1)
        capacity = depth * entries_per_word
        # No explicit constraint means the level keeps every dataspace.
        holds_weights = ("Weights" in keep) or (not keep and not bypass) or (
            "Weights" not in bypass and not keep)
        level = {"name": name, "class": cls, "depth": depth, "width": width,
                 "datawidth": dw, "entries": capacity, "instances": fanout,
                 "keep": keep, "bypass": bypass, "holds_weights": bool(holds_weights)}
        info["levels"].append(level)

        # Every on-chip level that carries Outputs, at the width it is BILLED
        # at. An absent constraint means the level keeps every dataspace, so
        # those count too -- that is exactly the case the stock
        # weight-/output-stationary designs are in.
        holds_outputs = ("Outputs" in keep) or (
            not keep and "Outputs" not in bypass)
        if holds_outputs and dw:
            info["psum_datawidth"].append((name, dw))
            # A level may legitimately carry Outputs below the accumulator
            # width -- Simba requantizes in the PE post-processing unit, so its
            # chip-level GlobalBuffer holds 8-bit activations, not 24-bit
            # psums. That is a claim about the design, so the design states it,
            # in the arch.yaml, next to the datawidth it justifies:
            #     datawidth: 8   # psum-width-ok: requantized before this level
            ok = re.search(r"#\s*psum-width-ok:\s*(.+)", body)
            if ok:
                info["psum_width_ok"][name] = ok.group(1).strip()

        # A shared (fanout 1) storage level is a global buffer whatever it is
        # called. The question is whether ANY such level holds Weights: that is
        # the level of weight reuse between DRAM and the PE array.
        if fanout == 1:
            info["glb_keeps_weights"] = bool(
                info["glb_keeps_weights"]) or bool(holds_weights)

        if holds_weights:
            info["weight_capacity"] += capacity * fanout
            info["weight_levels"].append(
                {"name": name, "entries_each": capacity, "instances": fanout,
                 "entries_total": capacity * fanout})

    info["total_fanout"] = fanout
    return info


def audit_findings(infos, cfg):
    """Turn a list of audits into cross-architecture warnings."""
    findings = []
    nodes = {i["arch"]: (i["technology"] or ["<inherited>"]) for i in infos}
    distinct = {tuple(v) for v in nodes.values()}
    if len(distinct) > 1 and not cfg.force_technology:
        findings.append(
            "MIXED TECHNOLOGY NODES. "
            + "; ".join(f"{a}={'/'.join(v)}" for a, v in nodes.items())
            + ". Accelergy costs components at the node declared on their container, so "
              "part of any energy gap is silicon, not architecture. "
              "Set ECC_FORCE_TECHNOLOGY=45nm to equalise.")

    dws = {i["arch"]: i["dram_datawidth"] for i in infos if i["dram_datawidth"]}
    if len(set(dws.values())) > 1 and not cfg.force_datawidth:
        findings.append(
            "MIXED DRAM DATAWIDTHS. "
            + "; ".join(f"{a}={v}b" for a, v in dws.items())
            + f". The study declares {cfg.weight_bits}-bit weights, but a design with "
              "datawidth 16 charges twice the DRAM energy per weight. "
              f"Set ECC_FORCE_DATAWIDTH={cfg.weight_bits} to equalise.")

    glb = {i["arch"]: i["glb_keeps_weights"] for i in infos}
    if len(set(glb.values())) > 1:
        keeps = [a for a, v in glb.items() if v]
        drops = [a for a, v in glb.items() if v is False]
        findings.append(
            "GLOBAL-BUFFER WEIGHT POLICY DIFFERS. "
            f"keep Weights in the GLB: {', '.join(keeps) or 'none'}; "
            f"bypass Weights: {', '.join(drops) or 'none'}. "
            "An architecture that bypasses Weights has one fewer level of weight reuse "
            "and must refetch from DRAM far more often. This is the single biggest "
            "driver of DRAM weight traffic, which is exactly what this study measures.")

    caps = {i["arch"]: i["weight_capacity"] for i in infos if i["weight_capacity"]}
    if caps:
        lo = min(caps.values())
        hi = max(caps.values())
        if hi >= 2 * lo:
            findings.append(
                "ON-CHIP WEIGHT CAPACITY SPANS >2x. "
                + "; ".join(f"{a}={v:,} weights" for a, v in
                            sorted(caps.items(), key=lambda kv: -kv[1]))
                + ". Less capacity means more DRAM refetch, independent of dataflow.")

    # ---- the one that hid the v1-vs-v2 result ------------------------------
    # A level that carries partial sums but declares a datawidth NARROWER than
    # the design's own accumulator is billing psums as operands. Timeloop packs
    # `width / datawidth` values into each physical word, so declaring a 16-bit
    # psum at datawidth 8 halves BOTH the per-scalar energy and the word count
    # -- and the error is invisible in the totals until it is compared against
    # a design that declared its psums honestly.
    under, declared_ok = [], []
    for i in infos:
        acc = i.get("adder_width")
        if not acc:
            continue
        for name, dw in i["psum_datawidth"]:
            if dw >= acc:
                continue
            reason = i.get("psum_width_ok", {}).get(name)
            if reason:
                declared_ok.append(f"{i['arch']}.{name} at {dw}b: {reason}")
            else:
                under.append(f"{i['arch']}.{name} bills {acc}b psums as {dw}b "
                             f"(x{acc // dw} too many per word)")
    if declared_ok:
        findings.append(
            "NARROWER-THAN-ACCUMULATOR OUTPUT LEVELS, DECLARED DELIBERATE. "
            + "; ".join(declared_ok)
            + ". Each carries a `# psum-width-ok:` note in its arch.yaml saying why. "
              "Listed so the claim is visible, not because anything is wrong.")
    if under:
        findings.append(
            "PARTIAL SUMS BILLED BELOW THE ACCUMULATOR WIDTH. "
            + "; ".join(under)
            + ". Timeloop packs width/datawidth values per word, so this understates "
              "both the per-psum energy and the access count at that level. It is the "
              "defect that made the stock eyeriss_like look cheaper than eyeriss_v2_like "
              "on every workload. Fix: ECC_ARCH_FIDELITY=paper (the default), which "
              "gives each design the psum precision its paper declares.")

    psums = {i["arch"]: sorted({dw for _n, dw in i["psum_datawidth"]}) for i in infos}
    psums = {a: v for a, v in psums.items() if v}
    if len({tuple(v) for v in psums.values()}) > 1:
        findings.append(
            "PARTIAL-SUM PRECISION DIFFERS. "
            + "; ".join(f"{a}={'/'.join(str(x) for x in v)}b" for a, v in psums.items())
            + ". A wider psum costs more per accumulate and, where the buffer word packs "
              "fewer psums, multiplies the number of buffer accesses. This is a real "
              "architectural difference (Eyeriss v1 accumulates at 16b, v2 at 20b, Simba "
              "at 24b), not a defect -- unlike the finding above.")

    # ---- unequal mapper effort ---------------------------------------------
    levels = {i["arch"]: (i.get("loop_levels"), i.get("victory")) for i in infos}
    levels = {a: v for a, v in levels.items() if v[0]}
    if len({v[0] for v in levels.values()}) > 1:
        detail = "; ".join(f"{a}={lv} levels -> victory {vic}"
                           for a, (lv, vic) in levels.items())
        if cfg.victory_scaling == "none":
            findings.append(
                "LOOP-NEST DEPTH DIFFERS AND MAPPER EFFORT IS FLAT. " + detail
                + ". Timeloop's search gives up after ECC_VICTORY consecutive "
                  "non-improving mappings, so the deeper design is searched less "
                  "thoroughly and the shortfall is reported as an architecture result. "
                  "Set ECC_VICTORY_SCALING=levels (the default) to scale effort with "
                  "depth.")
        else:
            findings.append(
                "LOOP-NEST DEPTH DIFFERS (effort is being scaled for it). " + detail
                + ". Scaling is a heuristic, not a proof of equal coverage: confirm by "
                  "raising ECC_VICTORY and checking the totals do not move.")

    if cfg.opt_metric != "energy":
        findings.append(
            f"MAPPER IS OPTIMISING {cfg.opt_metric.upper()}, NOT ENERGY. This study "
            "reports energy, but the mapper is choosing mappings by a different "
            "objective. Under 'edp' an architecture with more MACs can buy latency by "
            "spending energy, so the wider array is systematically pushed to a costlier "
            "mapping. Set ECC_OPT_METRIC=energy.")
    return findings
