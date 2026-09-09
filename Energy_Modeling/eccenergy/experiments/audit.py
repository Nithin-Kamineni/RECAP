"""Checks, caveats and detail shared by the evaluation experiments (Task 2 on).

Everything here MIRRORS the inline block in `experiments/baseline.py::run()`,
from "the checks that make the number auditable" down to `set_detail`.
`baseline.py` is Task 1's delivered and verified experiment and is deliberately
left untouched rather than refactored to call this module -- the request was
that the baseline code not be risked. So the two copies must be kept in step:
if a check changes here, change it there too, and vice versa. Task 2
(`experiments/embedded.py`) is the first consumer; Task 3 is the next.
"""
from __future__ import annotations

from .. import noc_post
from ..archs import (accumulator_bits, arch_source, noc_band_levels, noc_terms,
                     pe_latch_pj, validate_arch)


def components(series):
    """Per-category energies as a plain dict, dropping the empty categories."""
    return {k: float(v) for k, v in series.items() if float(v) != 0.0}


def mapping_ids_of(raw):
    return sorted({lp["mapping_id"] for lp in raw.per_layer if lp.get("mapping_id")})


def common_checks(builder, cfg, arch, raw, parity_detail, mapping_ids):
    """Task 1's checks, verbatim. Returns the architecture contract report."""
    builder.check("parity_payload_hand_check", parity_detail["hand_check_passed"],
                  parity_detail["hand_check"])
    builder.check("all_layers_mapped", raw.layers_skipped == 0,
                  f"{raw.layers_ok} mapped, {raw.layers_skipped} skipped")
    builder.check("mapping_ids_recorded", bool(mapping_ids),
                  f"{len(mapping_ids)} distinct mapping(s) behind this result")
    arch_report = validate_arch(arch, cfg)
    builder.check("architecture_obeys_shared_contract",
                  not arch_report["violations"],
                  {"violations": arch_report["violations"],
                   "notes": arch_report["notes"]})

    # ---- interconnect: was it costed, and does it look like the paper? ------
    # A zero here with the NoC enabled means the coefficients never reached
    # the networks that carry hops -- the failure mode that kept every NoC at
    # 0.00 before archs/_shared/noc.yaml existed.
    noc_pj = float(raw.base.get("NoC", 0.0))
    noc_share = noc_pj / raw.total if raw.total > 0 else 0.0
    # The paper's share (Fig. 18) is a gate-level breakdown of the CHIP, so
    # the band is read against on-chip energy; the Timeloop total, which
    # includes DRAM, is reported beside it (FINDINGS.md, open defect 8.2).
    onchip = raw.total - float(raw.base.get("DRAM", 0.0))
    noc_share_onchip = noc_pj / onchip if onchip > 0 else 0.0
    # What the paper calls the NoC (noc.yaml `paper_band_levels`): the
    # networks hanging off those spatial levels, as Timeloop printed them --
    # not the intra-cluster wiring and not the evaluator-only rows (tagged
    # with a trailing "]"), which Fig. 18 is read as booking under the PEs.
    band_levels = noc_band_levels(arch) if cfg.noc_enabled else []
    mesh_pj = sum(float(l["energy_pJ"]) for l in raw.levels
                  if str(l["level"]).startswith("NoC: ")
                  and not str(l["level"]).endswith("]")
                  and any(str(l["level"]).startswith(f"NoC: inter_{n}_spatial") for n in band_levels))
    mesh_share_onchip = mesh_pj / onchip if onchip > 0 else 0.0
    eval_only_pj = sum(float(l["energy_pJ"]) for l in raw.levels
                       if str(l["level"]).startswith("NoC: ") and str(l["level"]).endswith("]"))
    terms = noc_terms(arch, cfg) if cfg.noc_enabled else {}
    post = noc_post.coefficients(arch, cfg) if cfg.noc_enabled else {"enabled": False}
    builder.check("noc_energy_costed",
                  (not cfg.noc_enabled) or noc_pj > 0,
                  {"enabled": cfg.noc_enabled, "noc_pJ": noc_pj,
                   "share_of_timeloop_total": noc_share,
                   "share_of_onchip": noc_share_onchip,
                   "paper_band_levels": band_levels,
                   "mesh_pJ": mesh_pj,
                   "mesh_share_of_onchip": mesh_share_onchip,
                   "evaluator_only_pJ": eval_only_pj,
                   "evaluator_only_terms": post,
                   "pe_latch_pj": pe_latch_pj(cfg) if cfg.noc_enabled else None,
                   "tile_width_um": terms.get("tile_width_um"),
                   "wire_pj_per_bit_mm": terms.get("wire"),
                   "router_pj_per_flit": terms.get("router_pj_per_flit"),
                   "operands_per_flit": terms.get("operands_per_flit"),
                   "levels": ({n: {"router_pj": r, "ingress_pj": g}
                               for n, (r, g) in terms["levels"].items()}
                              if terms.get("levels") else
                              {"*": {"router_pj": terms.get("router"),
                                     "ingress_pj": terms.get("ingress")}}),
                   "source": "archs/_shared/noc.yaml"})
    if cfg.noc_enabled and arch.startswith("eyeriss_v2_like"):
        # JETCAS 2019 Sec. V / Fig. 18: the hierarchical mesh is "6%-10% of the
        # total energy consumption". The published band is for the paper's own
        # workloads, so a miss on one model is a prompt to look, not proof the
        # constants are wrong -- but a miss on EVERY model is.
        share_for_band = mesh_share_onchip if band_levels else noc_share_onchip
        builder.check("noc_share_within_published_band_eyeriss_v2",
                      0.06 <= share_for_band <= 0.10,
                      {"compared": ("hierarchical mesh only (noc.yaml paper_band_levels)"
                                    if band_levels else "all interconnect"),
                       "mesh_share_of_onchip": mesh_share_onchip,
                       "all_interconnect_share_of_onchip": noc_share_onchip,
                       "all_interconnect_share_of_timeloop_total": noc_share,
                       "published_band": [0.06, 0.10],
                       "source": "arXiv:1807.07928 Sec. V, Fig. 18",
                       "read_as": "Fig. 18 is a gate-level breakdown of the chip, so the "
                                  "band is a share of ON-CHIP energy, and its 'hierarchical "
                                  "mesh' is the inter-cluster links and router clusters -- "
                                  "the PE-row wiring and PE-to-PE psum passing inside a "
                                  "cluster are read as booked under the PE array "
                                  "(2026-09-09; an ASSUMPTION about the figure's category "
                                  "boundaries). Published share is for SPARSE runs at 65nm; "
                                  "this model is dense (pushes the share up) at 45nm "
                                  "(pushes it down). A few points either side is "
                                  "consistent; 15%+ is not."})
    if cfg.noc_enabled:
        builder.approximate(
            "Interconnect energy uses Timeloop's Legacy network model with "
            "one shared 45nm wire constant and per-architecture switching "
            "terms (archs/_shared/noc.yaml); hop distance is derived by "
            "Timeloop from each level's Accelergy area. The recon arm scales "
            "the WEIGHT share of NoC energy by K/N like the other on-chip "
            "categories.")
    return arch_report


def common_caveats(builder, cfg, arch, raw, parity_detail):
    """Task 1's approximations and warnings, verbatim."""
    builder.approximate(parity_detail["traffic"]["method"])
    mac = getattr(raw, "mac", None) or {}
    if cfg.mac_pj_override is not None:
        builder.warn(
            f"ECC_MAC_PJ_OVERRIDE={cfg.mac_pj_override:g} pJ per MAC "
            f"({mac.get('citation', cfg.mac_citation()[1])}): the Compute category "
            f"was rescaled in the evaluator from the ERT's "
            f"{mac.get('ert_pj_per_mac', float('nan')):.5f} pJ/MAC to MACs x this "
            f"value. Every ECC PERCENTAGE in this file has a different denominator "
            f"from the primary result; no saved pJ changed. "
            f"{mac.get('mapping_note', '')}")
    else:
        builder.approximate(
            f"MAC energy is the Accelergy ERT's {mac.get('ert_pj_per_mac', float('nan')):.5f} "
            f"pJ per 8-bit MAC (intmac = aladdin_multiplier 8x8 + aladdin_adder 20b, "
            f"Library plug-in, one 32-bit/40 nm table row each scaled linearly in "
            f"width and 1.2652x from 40 to 45 nm). It is the denominator of every "
            f"percentage here; FINDINGS 7.3 and provenance.yaml mac_energy_pj record "
            f"the cited alternatives and ECC_MAC_PJ_OVERRIDE re-evaluates under one.")
    builder.approximate(
        "External parity is billed at the measured per-access energy of a "
        "DRAM weight read on this architecture, not at an independently "
        "modelled parity-region access cost. That is exact if parity is "
        "read from the same DRAM by the same controller, which is the "
        "conventional-ECC assumption.")
    if cfg.layers:
        builder.warn(
            f"DEVELOPMENT RUN: {cfg.layer_scope} "
            f"({', '.join(cfg.layers)}). These numbers validate the "
            f"implementation, not any architecture ranking. Layer "
            f"selection is in the path and in `identity.layers`.")
    if cfg.mapper_search_size is not None or cfg.mapper_max_permutations != 16:
        builder.warn(
            f"BOUNDED MAPPER SEARCH (search_size="
            f"{cfg.mapper_search_size or 'uncapped'}, "
            f"max_permutations_per_if_visit={cfg.mapper_max_permutations}). "
            f"The search was cut short for speed, so these mappings are "
            f"upper bounds on energy, not converged optima, and the "
            f"shortfall is LARGER for deeper hierarchies. Development "
            f"only -- do not publish or rank architectures from this.")
    if cfg.baseline_inflates_onchip:
        builder.warn(
            "ECC_BASELINE_INFLATES_ONCHIP=1: on-chip weight traffic was "
            "inflated by the parity overhead. This is NOT the conventional "
            "baseline -- external parity is consumed by the off-chip ECC "
            "correction and does not enter on-chip weight storage.")
    if raw.layers_skipped:
        builder.warn(f"{raw.layers_skipped} layer(s) could not be mapped on "
                     f"{arch} and are missing from every number here")
    legacy = [lp for lp in raw.per_layer
              if lp.get("mapping_cached") and lp.get("mapping_id") is None]
    if legacy:
        builder.warn("some mappings came from the pre-fingerprint cache; "
                     "nothing proves they match the current arch.yaml")


def common_detail(builder, cfg, ses, arch, model, raw, arch_report, prov):
    """Task 1's `detail` block, verbatim."""
    acc, acc_evidence = accumulator_bits(arch, cfg)
    builder.set_detail(
        architecture={
            "source_yaml": str(arch_source(arch, cfg)),
            "fingerprint": ses.fingerprints.get(arch),
            "accumulator_bits": acc,
            "accumulator_evidence": acc_evidence,
            "contract_report": arch_report,
            "provenance": prov.get(arch, "not recorded"),
        },
        workload={
            "file": cfg.workload,
            "layers": ses.layer_ids.get(model, []),
            "layer_scope": cfg.layer_scope,
            "total_weights": raw.weights,
        },
        per_layer=raw.per_layer,
        per_level=raw.levels,
        mac_energy=getattr(raw, "mac", None),
        mappings=[(ses.mappers.get(arch).mappings if arch in ses.mappers
                   else {})],
        raw_energy_cache=str(
            ses.results.raw_path(arch, model, None, ses.fingerprints.get(arch))),
        mapper_cache=str(ses.results.mapper_cache(
            arch, None, ses.fingerprints.get(arch))),
    )
