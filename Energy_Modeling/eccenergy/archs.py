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
                    "smartbuffer_RF", "smartbuffer_RF_decoded"}


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
        m = re.search(r"\bdatawidth:\s*(\d+)", part)
        if not m:
            out.append(part)
            continue
        name = re.search(r"name:\s*(\S+)", part)
        name = name.group(1) if name else "?"
        keep = re.search(r"keep:\s*\[([^\]]*)\]", part)
        keep = [s.strip() for s in keep.group(1).split(",")] if keep else []
        width = re.search(r"\bwidth:\s*(\d+)", part)
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
        if not re.search(r"\bdatawidth:\s*(\d+)", part):
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
        width = re.search(r"\bwidth:\s*(\d+)", part)
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
#: that holds Weights ALONGSIDE another dataspace (`simple_weight_stationary`'s
#: `operand_glb` keeps Inputs and Weights): Timeloop has one capacity per
#: level, so dilating it hands the mapper extra INPUT capacity for free, which
#: reconstruction does not pay for. The two are an upper and a lower bound on
#: one design and are quoted as a pair, the same rule CLAUDE.md sets for the
#: eyeriss `_wglb` variants.
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
        depth = re.search(r"\bdepth:\s*(\d+)", part)
        if not depth:
            out.append(part)
            continue
        name = re.search(r"name:\s*(\S+)", part)
        name = name.group(1) if name else "?"
        keep = re.search(r"keep:\s*\[([^\]]*)\]", part)
        keep = [s.strip() for s in keep.group(1).split(",") if s.strip()] if keep else []
        d = int(depth.group(1))

        if "class: DRAM" in part or name == "DRAM":
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
        part = re.sub(r"\bdepth:\s*\d+", f"depth: {nd}", part, count=1)
        touched.append(f"{name} {d}->{nd}")
        out.append(part)

    if not quiet and arch:
        note = f"  [weight-capacity] {arch}: x{scale:g} on " + (
            ", ".join(touched) if touched else "NOTHING")
        if skipped:
            note += "; skipped " + "; ".join(skipped)
        print(note)
    return "".join(out)


def _weight_level_parts(text, scope="exclusive"):
    """Split `text` into `!Node` parts, tagging which ones hold WEIGHTS on chip.

    Shared by `_set_weight_datawidth` and `_scale_weight_depth` so the two
    prompt_2 knobs can never disagree about which levels they are talking
    about -- if one narrowed a level the other did not shrink, the two arms of
    a pair would stop declaring the same geometry and the comparison would be
    void without anything saying so.

    Yields `(part, name, is_weight_level, why_not)`.
    """
    for part in re.split(r"(\n\s*-\s*!)", text):
        depth = re.search(r"\bdepth:\s*(\d+)", part)
        name = re.search(r"name:\s*(\S+)", part)
        name = name.group(1) if name else "?"
        if not depth:
            yield part, name, False, None
            continue
        keep = re.search(r"keep:\s*\[([^\]]*)\]", part)
        keep = [s.strip() for s in keep.group(1).split(",") if s.strip()] if keep else []
        if "class: DRAM" in part or name == "DRAM":
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


def _set_weight_width(text, spad_width, glb_mult, scope="exclusive", arch="?",
                      quiet=False):
    """PROMPT_2's WIDTH TABLE: declare a physical word width the code's `q`
    divides, holding each level's TOTAL BITS at the published value.

    WHY A WIDTH KNOB EXISTS AT ALL. `timeloop-mapper` asserts
    `width % (word_bits * block_size) == 0` (`buffer.cpp:302`) with
    `block_size` defaulting to 1, and there is NO floor path -- a width the
    datawidth does not divide ABORTS the mapper (measured, `exit=134, core
    dumped`), it does not fall back to `floor(width/q)`. So a code whose
    `q = round(8*K/N)` does not divide the published word needs a declared
    width that it does. prompt_2 tabulates one per code, all five within 3 % of
    each other so the arms stay comparable across codes.

    BCH(63,30) NEEDS NONE, which is why it is the code this study starts from:
    q = 4 divides Eyeriss v1's published 16-bit scratchpad word and its 64-bit
    GLB word already, so the arms run on the UNTOUCHED published geometry.
    Leave this EMPTY for that case; it is not a no-op that costs nothing, it is
    a no-op that keeps the published silicon.

    THE WIDTH TABLE IS QUOTED FOR THE SCRATCHPAD. A weight level ABOVE the PE
    array declares `glb_mult` times that width -- 4x, which is the ratio
    Eyeriss v1's published geometry already has (the filter spad is a 224 x
    16-b SRAM, the filter GLB two 512-b x 64-b banks). The rule also preserves
    the divisibility the mapper demands: if `q` divides W it divides 4W, so one
    scratchpad width settles every weight level at once.

    DEPTH IS RENORMALISED TO HOLD THE DECLARED BITS. `depth' = round(depth *
    width / width')`, so the level is the same amount of silicon at a different
    word shape rather than a bigger array smuggled in as a width change -- and
    CACTI is handed `depth` and `width`, so that is exactly the quantity that
    must not move. It reproduces prompt_2's own table: `weights_spad`
    224 x 16 b = 3,584 b -> depth 37 at width 96; `filter_glb` 1024 x 64 b =
    65,536 b -> depth 171 at width 384.

    This runs BEFORE `_scale_weight_depth`, so the swept ladder multiplies the
    renormalised depth, not the published one.
    """
    if not spad_width:
        return text
    # The INNERMOST weight level is the scratchpad; everything above it is a
    # GLB and takes `glb_mult` times the width. Innermost = last in file order,
    # which is the order Timeloop reads levels in (outer to inner).
    names = [name for _p, name, is_w, _why
             in _weight_level_parts(text, scope) if is_w]
    if not names:
        return text
    spad = names[-1]
    out, touched, skipped = [], [], []
    for part, name, is_weight, why_not in _weight_level_parts(text, scope):
        if not is_weight:
            if why_not:
                skipped.append(why_not)
            out.append(part)
            continue
        want = int(spad_width) if name == spad else int(spad_width) * int(glb_mult)
        w = re.search(r"\bwidth:\s*(\d+)", part)
        d = re.search(r"\bdepth:\s*(\d+)", part)
        if not w:
            skipped.append(f"{name} declares no width:")
            out.append(part)
            continue
        w0, d0 = int(w.group(1)), int(d.group(1))
        if w0 == want:
            skipped.append(f"{name} already declares width {want}")
            out.append(part)
            continue
        nd = max(1, int(round(d0 * w0 / want)))
        part = re.sub(r"\bwidth:\s*\d+", f"width: {want}", part, count=1)
        part = re.sub(r"\bdepth:\s*\d+", f"depth: {nd}", part, count=1)
        touched.append(f"{name} {d0}x{w0}b -> {nd}x{want}b "
                       f"({d0 * w0:,} -> {nd * want:,} bits)")
        out.append(part)
    if not quiet and arch:
        note = (f"  [weight-width] {arch}: scratchpad {spad_width}b, GLB "
                f"{int(spad_width) * int(glb_mult)}b ({glb_mult}x) on "
                + (", ".join(touched) if touched else "NOTHING"))
        if skipped:
            note += "; skipped " + "; ".join(skipped)
        print(note)
    return "".join(out)


def _set_weight_datawidth(text, bits, scope="exclusive", arch="?", quiet=False):
    """PROMPT_2: express the reduced representation as `datawidth:` on the
    on-chip weight levels, at FIXED `width:` and `depth:`.

    WHY THIS AND NOT `_scale_weight_capacity`. Verified in this repo
    2026-09-10 from `timeloop-mapper.accelergy.log:97` (`Calculated
    storage."width" as "width" = 16`): CACTI receives `depth` and `width`
    ONLY -- `datawidth` never reaches the energy model. Timeloop then bills
    `vector_access_energy / block_size` per weight, `block_size =
    width / datawidth`. So halving `datawidth` at fixed geometry exactly
    halves per-weight energy and exactly doubles effective capacity, with
    IDENTICAL per-access read/write/leak. That is the "same energies, more
    effective capacity" condition, and it holds exactly -- which is the
    fairness condition depth-dilation could never meet (it priced the
    reconstruction arm's array 1.18-1.46x dearer and gave the optimiser a
    reason to leave the room unused; FINDINGS 7.8).

    HARD CONSTRAINT, and it ABORTS rather than degrades. `timeloop-mapper`
    asserts `width % (word_bits * block_size) == 0` (`buffer.cpp:302`) with
    `block_size` defaulting to 1: measured at `width: 16, datawidth: 5` it
    dies with `exit=134, core dumped`. There is NO floor path -- Timeloop does
    not attempt `floor(16/5) = 3`, so a partially-filled word cannot be
    modelled at all. This raises here, before the YAML is written, so a bad
    combination fails once instead of aborting every layer of a wave.

    WHAT IS NOT REWRITTEN: DRAM (`recon.py` owns its K/N scaling; narrowing it
    here as well would double-count), any level holding no Weights, a level
    holding Weights beside another dataspace unless `scope=shared`, and a
    declared `depth: 1` latch.
    """
    out, touched, skipped, bad = [], [], [], []
    for part, name, is_weight, why_not in _weight_level_parts(text, scope):
        if not is_weight:
            if why_not:
                skipped.append(why_not)
            out.append(part)
            continue
        width = re.search(r"\bwidth:\s*(\d+)", part)
        dw = re.search(r"\bdatawidth:\s*(\d+)", part)
        if not dw:
            skipped.append(f"{name} declares no datawidth:")
            out.append(part)
            continue
        if width is not None and int(width.group(1)) % bits != 0:
            bad.append(f"{name}: width {width.group(1)} % datawidth {bits} != 0")
            out.append(part)
            continue
        if int(dw.group(1)) == bits:
            skipped.append(f"{name} already declares datawidth {bits}")
            out.append(part)
            continue
        touched.append(f"{name} {dw.group(1)}->{bits}b "
                       f"({int(width.group(1)) // bits} weights/word)"
                       if width else f"{name} {dw.group(1)}->{bits}b")
        out.append(re.sub(r"\bdatawidth:\s*\d+", f"datawidth: {bits}", part,
                          count=1))
    if bad:
        raise ValueError(
            f"ECC_WEIGHT_DATAWIDTH={bits} on {arch}: " + "; ".join(bad) + ".\n"
            f"  timeloop-mapper asserts `width % (word_bits * block_size) == 0` "
            f"(buffer.cpp:302) and ABORTS -- there is no floor path, so a\n"
            f"  partially-filled word cannot be modelled. Declare a `width:` "
            f"the datawidth divides. prompt_2.md's WIDTH TABLE gives one per\n"
            f"  code: BCH(63,57) q=7 width 98; BCH(63,45) q=6 width 96; "
            f"BCH(63,39) q=5 width 95; BCH(63,30) q=4 needs NO width change.")
    if not quiet and arch:
        note = f"  [weight-datawidth] {arch}: {bits}b on " + (
            ", ".join(touched) if touched else "NOTHING")
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
        d = int(re.search(r"\bdepth:\s*(\d+)", part).group(1))
        nd = max(1, int(round(d * scale)))
        if nd == d:
            skipped.append(f"{name} (depth {d} x {scale:g} rounds back to {d})")
            out.append(part)
            continue
        touched.append(f"{name} {d}->{nd}")
        out.append(re.sub(r"\bdepth:\s*\d+", f"depth: {nd}", part, count=1))
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
        depth = re.search(r"\bdepth:\s*(\d+)", part)
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
        is_storage = re.search(r"\bdepth:\s*\d+", part) is not None
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
        depth = re.search(r"\bdepth:\s*(\d+)", part)
        name = re.search(r"name:\s*(\S+)", part)
        if not depth or not name or name.group(1) == "DRAM":
            continue
        keep = re.search(r"keep:\s*\[([^\]]*)\]", part)
        keep = [s.strip() for s in keep.group(1).split(",") if s.strip()] if keep else []
        if "Weights" not in keep:
            continue
        width = re.search(r"\bwidth:\s*(\d+)", part)
        dw = re.search(r"\bdatawidth:\s*(\d+)", part)
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
        depth = re.search(r"\bdepth:\s*(\d+)", part)
        name = re.search(r"name:\s*(\S+)", part)
        if not depth or not name or name.group(1) == "DRAM":
            continue
        keep = re.search(r"keep:\s*\[([^\]]*)\]", part)
        keep = [s.strip() for s in keep.group(1).split(",") if s.strip()] if keep else []
        if "Weights" not in keep:
            continue
        width = re.search(r"\bwidth:\s*(\d+)", part)
        dw = re.search(r"\bdatawidth:\s*(\d+)", part)
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
    """PROMPT_2 FAIRNESS RULE, mechanically: the two arms of a pair must
    declare the SAME `width` and the SAME `depth` on every weight level.

    WHY IT IS AN ASSERTION AND NOT A CONVENTION. This is the exact defect that
    made the previous sweep prove nothing: capacity was expressed as
    `depth x N/K`, so Accelergy priced the reconstruction arm's array
    1.18-1.46x dearer per access and the optimiser had a REASON to leave the
    room unused (FINDINGS 7.8). Under prompt_2 the arms share one hardware
    YAML and differ only in `datawidth`, which CACTI never sees -- so if a
    width or a depth ever differs between them, the comparison is void and
    must stop rather than be corrected afterwards.

    `datawidth` (and the `weights_per_word` it derives) is EXPECTED to differ;
    that is the treatment.
    """
    a = patched_weight_geometry(arch, cfg_ref)
    b = patched_weight_geometry(arch, cfg_arm)
    problems = []
    if set(a) != set(b):
        problems.append(
            f"different weight LEVELS: {ref_name} has "
            f"{', '.join(sorted(a)) or 'none'}; {arm_name} has "
            f"{', '.join(sorted(b)) or 'none'}")
    for level in sorted(set(a) & set(b)):
        for field in ("depth", "width"):
            if a[level][field] != b[level][field]:
                problems.append(
                    f"{level}.{field}: {ref_name}={a[level][field]} "
                    f"{arm_name}={b[level][field]}")
    if problems:
        raise ValueError(
            f"{arch}: the two arms do NOT declare the same silicon -- "
            + "; ".join(problems) + ".\n"
            f"  prompt_2's fairness rule is that both arms of a pair share one "
            f"hardware YAML and differ ONLY in `datawidth:`, which CACTI never\n"
            f"  sees. A width or depth difference is priced by Accelergy as "
            f"real silicon one arm does not have, which is the defect that\n"
            f"  invalidated the previous sweep (FINDINGS 7.8). The comparison "
            f"is void; fix the configuration rather than correcting the energy.")
    return {level: {"shared": {k: a[level][k] for k in ("depth", "width")},
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
    if apply_per_arch and cfg.weight_capacity_scale != 1.0:
        text = _scale_weight_capacity(text, cfg.weight_capacity_scale,
                                      cfg.weight_capacity_scope, arch, quiet)
    # prompt_2: WIDTH, then DEPTH, then DATAWIDTH.
    # WIDTH first because it renormalises `depth:` to hold the declared bits,
    # so the swept ladder must multiply the renormalised depth rather than the
    # published one. DATAWIDTH last because its hard constraint
    # (`width % datawidth == 0`) has to be checked against the FINAL width.
    if apply_per_arch and getattr(cfg, "weight_width", None):
        text = _set_weight_width(text, cfg.weight_width,
                                 cfg.weight_width_glb_mult,
                                 cfg.weight_capacity_scope, arch, quiet)
    if apply_per_arch and getattr(cfg, "weight_depth_scale", 1.0) != 1.0:
        text = _scale_weight_depth(text, cfg.weight_depth_scale,
                                   cfg.weight_depth_levels,
                                   cfg.weight_capacity_scope, arch, quiet)
    if apply_per_arch and getattr(cfg, "weight_datawidth", None) is not None:
        text = _set_weight_datawidth(text, cfg.weight_datawidth,
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
    # Last, so the coefficients land on the final text and are hashed by
    # arch_fingerprint(). Applied regardless of apply_per_arch: the NoC model
    # is a study-wide treatment, not a per-architecture no-op candidate.
    text = _inject_noc(text, arch, cfg)
    return text


# --------------------------------------------------------------- fingerprints
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

    `mapper_settings` defaults to `cfg.mapper_settings()`. It is a parameter so
    a caller that has already resolved the effective thread count can pass the
    real one rather than the configured `None`.
    """
    text = _patched_text(arch, cfg, quiet=True)
    # globals.yaml is a mapper input too -- it sets the node DRAM is costed at,
    # which changes the energy the mapper is optimising. Hashing the values
    # rather than reading the file keeps this usable before it is written.
    globals_view = {
        "technology": cfg.force_technology or load_standard()["study"]["technology"],
        "global_cycle_seconds": cfg.global_cycle_seconds,
    }
    blob = json.dumps({
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
    }, sort_keys=True)
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
        base = arch_source(arch, cfg).read_text()
        if _scale_weight_capacity(base, cfg.weight_capacity_scale,
                                  cfg.weight_capacity_scope, arch,
                                  quiet=True) != base:
            parts.append(f"wcap{cfg.weight_capacity_scale:g}"
                         + ("-shared" if cfg.weight_capacity_scope == "shared" else ""))
    if getattr(cfg, "weight_width", None):
        # Same no-op rule: a design already declaring that width keeps its
        # existing cache rather than paying for a fresh map of an unchanged
        # architecture.
        base = arch_source(arch, cfg).read_text()
        if _set_weight_width(base, cfg.weight_width, cfg.weight_width_glb_mult,
                             cfg.weight_capacity_scope, arch,
                             quiet=True) != base:
            parts.append(f"ww{cfg.weight_width}"
                         + (f"x{cfg.weight_width_glb_mult}"
                            if cfg.weight_width_glb_mult != 4 else ""))
    if getattr(cfg, "weight_depth_scale", 1.0) != 1.0:
        # Same no-op rule as the capacity scale.
        base = _set_weight_width(arch_source(arch, cfg).read_text(),
                                 getattr(cfg, "weight_width", None),
                                 getattr(cfg, "weight_width_glb_mult", 4),
                                 cfg.weight_capacity_scope, arch, quiet=True)
        if _scale_weight_depth(base, cfg.weight_depth_scale,
                               cfg.weight_depth_levels,
                               cfg.weight_capacity_scope, arch,
                               quiet=True) != base:
            parts.append(f"wdepth{cfg.weight_depth_scale:g}"
                         + ("-" + "+".join(cfg.weight_depth_levels)
                            if cfg.weight_depth_levels else ""))
    if getattr(cfg, "weight_datawidth", None) is not None:
        # Same no-op rule again: the BASELINE/EMBEDDED arm may legitimately be
        # spelled `ECC_WEIGHT_DATAWIDTH=8` on a design already declaring 8, and
        # that arm must then READ THE SAME CACHE as leaving the knob empty --
        # otherwise the two arms of a pair would be compared across two mapper
        # caches of one identical architecture.
        base = _set_weight_width(arch_source(arch, cfg).read_text(),
                                 getattr(cfg, "weight_width", None),
                                 getattr(cfg, "weight_width_glb_mult", 4),
                                 cfg.weight_capacity_scope, arch, quiet=True)
        # depth first, so the no-op test sees the geometry the arm really has
        base_d = _scale_weight_depth(base, cfg.weight_depth_scale,
                                     cfg.weight_depth_levels,
                                     cfg.weight_capacity_scope, arch, quiet=True) \
            if getattr(cfg, "weight_depth_scale", 1.0) != 1.0 else base
        if _set_weight_datawidth(base_d, cfg.weight_datawidth,
                                 cfg.weight_capacity_scope, arch,
                                 quiet=True) != base_d:
            parts.append(f"wdw{cfg.weight_datawidth}")
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


def _write_atomic(dst, text):
    """Write `text` to `dst` so a concurrent reader never sees a partial file.

    Both files this module writes -- the patched arch.yaml and globals.yaml --
    are byte-identical for every process that shares an (architecture,
    treatment), so parallel writers do not disagree about the CONTENT. What a
    plain `write_text` cannot promise is that a reader arriving mid-write sees
    all of it: the truncate-then-write window is real on a shared filesystem,
    and a SLURM job array over architectures has many mapper processes reading
    these two paths at once. Writing a private temp file in the same directory
    and renaming makes the replacement atomic, so a reader sees either the old
    complete file or the new one. Always LF, even when written from Windows.
    """
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    with open(tmp, "w", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, dst)
    return dst


def patched_arch_path(arch, cfg):
    """Write and return the arch.yaml for this architecture + treatment."""
    text = _patched_text(arch, cfg)
    variant = effective_variant(arch, cfg)
    suffix = "" if variant == "stock" else f"__{variant}"
    dst = WORK / f"arch_{arch}_patched{suffix}.yaml"
    return _write_atomic(dst, text)


def write_globals(cfg):
    """`globals.yaml` sets the node for anything OUTSIDE an arch container.

    DRAM sits above the accelerator container in every one of these designs, so
    it takes its technology from here, identically for every architecture.

    THE DEFAULT USED TO BE 65nm while every accelerator container declared 45nm,
    so DRAM -- the level this study spends most of its energy in, and the level
    BCH parity lands on -- was costed at a different node from the logic it
    talks to. It was uniform across architectures, so it never showed up as an
    ordering error; it just made every absolute number and every savings
    percentage wrong by a fixed factor. The node now comes from
    archs/_shared/standard.yaml, which is the same file the architectures are
    validated against, so the two cannot drift apart again.
    """
    node = cfg.force_technology or load_standard()["study"]["technology"]
    p = WORK / "globals.yaml"
    _write_atomic(p, "variables:\n"
                     "  version: 0.4\n"
                     f"  global_cycle_seconds: {cfg.global_cycle_seconds}\n"
                     f'  technology: "{node}"\n')
    return p, node


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
