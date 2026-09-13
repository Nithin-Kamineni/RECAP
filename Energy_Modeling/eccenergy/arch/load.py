"""Loading a design, and the cross-design contract it is held to.

* `install_local_archs()` copies the repo-authored designs in `archs/` into the
  cloned exercises tree, so a wipe of `ecc_energy_study/` cannot lose them.
* `arch_source()` / `paper_source()` resolve which YAML a design IS, honouring
  `ECC_ARCH_PIN_DIR` through `paths.py` -- so a submission that pinned `archs/`
  keeps mapping the chip it submitted while the live tree is edited.
* `load_standard()`, `load_provenance()`, `load_noc()` read the three shared
  files in `archs/_shared/`: what must be IDENTICAL across designs, where every
  declared number came from, and the interconnect coefficients.
* `accumulator_bits()`, `mac_candidates()`, `spatial_containers()`,
  `noc_terms()` and friends answer one question about a design each.
* `_blocks()` and `_num()` are the two text helpers every other module in this
  package reads an arch YAML with. They live here because `arch/layout.py` and
  `arch/validate.py` both need them and neither may import the other.

ONLY ONE OF THE FOUR `source:` VALUES licenses using a design's name as the
chip: `published`. `derived` differs in one stated block, `reference_design` is
shipped by timeloop-accelergy-exercises and merely NAMED after a paper, and
`locally_authored` reproduces nothing.

ProjectRestructure phase 3 cut `archs.py` (2,762 lines) into this,
`arch/patch.py`, `arch/fingerprint.py`, `arch/layout.py` and
`arch/validate.py`, in that dependency order.
"""
from __future__ import annotations

import re
import shutil
import yaml

from ..paths import ARCH_NOC, ARCH_PROVENANCE, ARCH_SRC, ARCH_SRC_RESERVED, ARCH_STANDARD, DESIGNS_DIR


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


