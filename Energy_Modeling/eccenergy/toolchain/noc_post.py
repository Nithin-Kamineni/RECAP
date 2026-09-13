"""Evaluator-only interconnect terms: what Timeloop counts but cannot cost.

Two pieces of NoC physics have no hook in Timeloop's legacy network model, so
they cannot be handed to the mapper through `archs/_shared/noc.yaml` the way
wire, router and ingress energy are:

1. **Spatial reductions.** When a mapping accumulates partial sums across PEs,
   Timeloop counts every psum added into a neighbour ("Spatial reductions" in
   the stats) and costs it with `pat::AdderEnergy()`, which is a stub that
   returns 0 (pat-public/src/pat/pat.cpp:101). `LegacyNetwork::ParseSpecs`
   reads no adder coefficient, so there is nothing to inject.
2. **Partial-sum word width.** A legacy network has ONE `word_bits`, and it is
   set to the 8-bit operand (`network_word_bits`), so a 20-bit psum travels the
   wire as 8 bits.

Both are therefore charged HERE, after the mapping is chosen, from the counts
Timeloop printed. That makes them evaluator-only: they are not in the mapper's
objective, and every result that includes them says so through the stamp
`Raw.noc_post` carries. The coefficients live in noc.yaml
`shared.evaluator_only`, beside the ones the mapper does see, so the whole
interconnect model is read from one file.

Why not fold them into the injected terms instead: a reduction is not an
ingress (no ingress count moves when a mapping reduces spatially), and the
psum width is per dataspace where Timeloop's word width is per network. Both
approximations would charge the wrong traffic.
"""
from __future__ import annotations

from ..arch.load import load_noc, load_standard, noc_terms

#: Suffixes on the level name of an added row, so a reader of `Raw.levels` can
#: tell an evaluator-only term from what Timeloop itself reported.
REDUCTION_TAG = "[spatial reduction]"
WIDTH_TAG = "[psum word width]"


def accumulator_bits(arch, cfg):
    """The design's own partial-sum width (standard.yaml), or the ECC_ACC_BITS override."""
    forced = getattr(cfg, "acc_bits_override", None)
    if forced:
        return int(forced)
    spec = (load_standard().get("architectures") or {}).get(arch) or {}
    return int(spec.get("accumulator_bits", cfg.weight_bits))


def coefficients(arch, cfg):
    """Resolved evaluator-only coefficients for one design, or `enabled: False`."""
    ev = (load_noc().get("shared") or {}).get("evaluator_only") or {}
    if not cfg.noc_enabled or not ev:
        return {"enabled": False}
    sr = ev.get("spatial_reduction") or {}
    acc = accumulator_bits(arch, cfg)
    owb = ev.get("outputs_word_bits", None)
    outputs_word_bits = acc if owb == "accumulator" else (int(owb) if owb else cfg.weight_bits)
    return {
        "enabled": True,
        "network_word_bits": int(cfg.weight_bits),       # what Timeloop billed per word
        "outputs_word_bits": int(outputs_word_bits),     # what a psum actually is
        "adder_pj_per_bit": float(sr.get("adder_pj_per_bit", 0.0)),
        "reduction_hops": float(sr.get("hops", 0.0)),
        "noc_scale": float(cfg.noc_scale),
    }


def stamp(arch, cfg):
    """What a raw record must have been built with to be reused (see energy.load_raw)."""
    c = coefficients(arch, cfg)
    return {k: c[k] for k in sorted(c)}


def _reduction_energy(row, c):
    """One spatial reduction = one adder of psum width + `hops` hops of psum-wide wire."""
    n = row.get("spatial_reductions") or 0.0
    if n <= 0:
        return 0.0
    inst = row.get("net_instances") or 1
    per_bit_hop = (row.get("per_hop_pJ") or 0.0) / c["network_word_bits"]
    per_red = (c["adder_pj_per_bit"] * c["outputs_word_bits"]
               + c["reduction_hops"] * c["outputs_word_bits"] * per_bit_hop)
    return n * inst * per_red * c["noc_scale"]


def _width_energy(row, c):
    """The Outputs wire term, re-billed from `network_word_bits` to the psum width."""
    ing, hops, per_hop = row.get("reads"), row.get("hops"), row.get("per_hop_pJ")
    if not ing or not hops or not per_hop:
        return 0.0
    inst = row.get("net_instances") or 1
    wire = ing * hops * per_hop * inst          # exactly Timeloop's wire term
    return wire * (c["outputs_word_bits"] / c["network_word_bits"] - 1.0)


def augment(rows, arch, cfg):
    """Append the evaluator-only NoC rows for one layer's parsed stats.

    Only network rows (`level` starts with "NoC: ") for the Outputs dataspace
    contribute; every added row keeps the layer and dataspace and carries the
    source network's name plus a tag. Rows are returned, not mutated.
    """
    c = coefficients(arch, cfg)
    if not c["enabled"]:
        return list(rows)
    out = list(rows)
    for r in rows:
        lvl = str(r.get("level", ""))
        if not lvl.startswith("NoC: ") or r.get("dataspace") != "Outputs":
            continue
        e_red = _reduction_energy(r, c)
        if e_red > 0:
            out.append(dict(layer=r["layer"], level=f"{lvl} {REDUCTION_TAG}",
                            dataspace="Outputs", instances=None,
                            reads=r.get("spatial_reductions"), writes=0.0,
                            energy_pJ=e_red))
        e_w = _width_energy(r, c)
        if e_w > 0:
            out.append(dict(layer=r["layer"], level=f"{lvl} {WIDTH_TAG}",
                            dataspace="Outputs", instances=None,
                            reads=None, writes=0.0, energy_pJ=e_w))
    return out


def describe(arch, cfg):
    """One line for consoles and result notes."""
    c = coefficients(arch, cfg)
    if not c["enabled"]:
        return "evaluator-only NoC terms: off"
    return (f"evaluator-only NoC terms (not in the mapper objective): spatial reductions at "
            f"{c['adder_pj_per_bit'] * c['outputs_word_bits']:.3f} pJ + {c['reduction_hops']:g} hop of "
            f"{c['outputs_word_bits']}-bit wire; Outputs wire re-billed {c['network_word_bits']} -> "
            f"{c['outputs_word_bits']} bits")
