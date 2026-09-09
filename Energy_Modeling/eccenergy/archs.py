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


def _patched_text(arch, cfg, apply_per_arch=True, quiet=False):
    text = arch_source(arch, cfg).read_text()
    text = _patch_dram_depth(text, cfg.dram_depth)
    if cfg.force_technology:
        text = _force_technology(text, cfg.force_technology)
    if apply_per_arch and cfg.force_datawidth:
        text = _force_datawidth(text, cfg.force_datawidth, None if quiet else arch)
    if apply_per_arch and cfg.acc_bits_override is not None:
        text = _force_acc_bits(text, cfg.acc_bits_override, arch, quiet)
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
