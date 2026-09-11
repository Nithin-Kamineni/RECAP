"""baseline: Task 1's conventional-ECC result, one JSON per architecture.

    bash run.sh baseline                       # the selected layers
    bash run.sh baseline --layers "conv1"      # a different selection
    bash run.sh baseline --eval                # evaluation only, no mapper

WHAT IT COMPUTES
----------------
One variant, and only one: the conventional-ECC baseline, where weights are the
protected payload and BCH parity is stored EXTERNALLY in DRAM. Parity is
counted by `parity.py` -- whole weights per codeword, message padding, tail
padding, whole-DRAM-word granularity -- not by a flat `n/k`.

WHAT IT DELIBERATELY DOES NOT COMPUTE
-------------------------------------
Embedded ECC (Task 2) and every reconstruction placement (Tasks 3-5) are
written into the same file, as the storage spec requires, with
`status: not_implemented` and a reason. They carry NO energy number. Filling
them with a plausible value now is exactly how a placeholder becomes a result.

WHY THE PHASE IS ALWAYS `Pre`
-----------------------------
`Post` means the mapper was told about the reduced weight representation.
Nothing in Task 1 has a reduced representation, so there is nothing it could
have been told. A Task 1 run that claimed `Post` would be claiming an
optimisation it did not perform.
"""
from __future__ import annotations

from .. import baseline_dram
from ..archs import accumulator_bits, arch_source, load_provenance, noc_terms, validate_arch
from ..ecc import external_parity
from ..energy import plot_cats
from ..results_store import ResultBuilder, Variant
from .common import Session

#: Reconstruction boundaries this study will evaluate, from section 5.3 of
#: 01_project_context_and_architectures.txt. Named here so that Task 1's output
#: already carries the full variant list -- as unavailable entries -- and later
#: tasks fill them in rather than changing the shape of the document.
RECONSTRUCTION_PLACEMENTS = [
    ("recon_source_noc_ingress", "reconstruct at the source / NoC ingress"),
    ("recon_destination_cluster", "reconstruct at the destination-cluster boundary"),
    ("recon_pe_spad_input", "reconstruct at the PE weight-SPad input"),
    ("recon_pe_spad_output", "reconstruct at the weight-SPad output, repeated"),
]


def _components(series):
    """Per-category energies as a plain dict, dropping the empty categories."""
    return {k: float(v) for k, v in series.items() if float(v) != 0.0}


def _placeholder_variants(builder):
    """Every variant Task 1 does not implement, recorded as unavailable."""
    builder.add(Variant(
        "embedded_ecc", kind="embedded", status="not_implemented",
        unavailable_reason=(
            "Task 2 has not been implemented. Embedded ECC needs the actual "
            "embedded-codeword layout and a DRAM-only comparison against this "
            "baseline; nothing in Task 1 models it, so no number is written."),
        label="Embedded ECC (no external parity)"))
    for name, description in RECONSTRUCTION_PLACEMENTS:
        builder.add(Variant(
            name, kind="reconstruction", status="not_implemented",
            unavailable_reason=(
                f"Tasks 3-5 have not been implemented. This entry names the "
                f"placement ({description}) so the document shape is final from "
                f"Task 1 onward, but no reconstruction has been evaluated."),
            label=description))


def _report(cfg, arch, model, raw, energy_total, detail, pricing):
    acc, _ = accumulator_bits(arch, cfg)
    p = detail["stored"]
    tr = detail["traffic"]
    g = detail["code"]
    print(f"\n  --- {arch} / {model} : conventional ECC, external BCH parity ---")
    print(f"    precisions            : weight {cfg.weight_bits}b   "
          f"activation {cfg.activation_bits}b   accumulator {acc}b (published)")
    print(f"    code                  : BCH({g['n']},{g['k']}) t={cfg.code_t}   "
          f"{g['weights_per_codeword']} whole weights/codeword   "
          f"{g['message_pad_bits_per_codeword']}b message padding   "
          f"{g['parity_bits_per_codeword']}b parity")
    print(f"    payload               : {p['weights']:,} weights = "
          f"{p['payload_bytes']:,.0f} B")
    print(f"    codewords             : {p['codewords']:,}")
    print(f"    parity stored         : {p['parity_bytes']:,.0f} B   "
          f"padding {(p['message_pad_bits'] + p['tail_pad_bits']) / 8:,.0f} B   "
          f"-> {p['overhead_frac'] * 100:.2f}% over payload")
    print(f"    (a flat n/k model would have charged "
          f"{detail['comparison_to_flat_model']['flat_n_over_k_minus_1'] * 100:.2f}%)")
    print(f"    DRAM weight reads     : {raw.dram_w_reads:,.0f}   "
          f"refetch x{tr['refetch_factor']:.2f}"
          + ("   (identical on all three arms)"
             if pricing["model"] == "per_bit_price" else ""))
    if pricing["model"] == "per_bit_price":
        # NOT traffic: the parity sits in the array and is corrected on the die.
        print(f"    parity words STORED   : {tr['external_dram_words']:,.0f}  "
              f"({tr['external_dram_scalars']:,.0f} weight-sized words held, "
              f"NOT fetched)")
        print(f"    pJ per DRAM weight    : "
              f"{detail['pJ_per_dram_weight_scalar'] * pricing['ratio']:.4f}   "
              f"(embedded/recon pay {detail['pJ_per_dram_weight_scalar']:.4f} "
              f"for the same 8 bits)")
    else:
        print(f"    parity DRAM words     : {tr['external_dram_words']:,.0f}  "
              f"({tr['external_dram_scalars']:,.0f} weight-sized accesses)")
        print(f"    pJ per DRAM weight    : {detail['pJ_per_dram_weight_scalar']:.4f}"
              f"   <- must match across architectures at the same node/datawidth")
    print(f"    Timeloop energy       : {raw.total / 1e6:12,.3f} uJ")
    if pricing["model"] == "per_bit_price":
        print(f"    DRAM price            : {pricing['pj_per_bit_charged']:g} -> "
              f"{pricing['pj_per_bit_baseline']:g} pJ/bit "
              f"(x{pricing['ratio']:.4f}; bigger array + indexing)")
        print(f"    + DRAM price delta    : {pricing['dram_delta_pJ'] / 1e6:12,.3f} uJ"
              f"   (whole DRAM category: {pricing['dram_pJ_before'] / 1e6:,.3f} -> "
              f"{pricing['dram_pJ_after'] / 1e6:,.3f} uJ)")
        print(f"    parity traffic        : "
              f"{pricing['parity_traffic_energy_NOT_charged_pJ'] / 1e6:12,.3f} uJ"
              f"   NOT CHARGED -- corrected on the DRAM die, never on the datapath")
    else:
        print(f"    + external parity     : {detail['energy_pJ'] / 1e6:12,.3f} uJ")
    print(f"    = conventional ECC    : {energy_total / 1e6:12,.3f} uJ")
    print(f"    hand check            : "
          f"{'PASS' if detail['hand_check_passed'] else 'FAIL'}")


def run(cfg):
    if cfg.phase != "Pre":
        raise SystemExit(
            f"ECC_PHASE={cfg.phase} but the Task 1 baseline is a `Pre` result by "
            f"construction: there is no reduced weight representation yet, so no "
            f"mapping could have been optimised for one.")

    ses = Session(cfg).setup()
    ses.collect_all()
    prov = load_provenance()
    written = []

    for arch in cfg.archs:
        raws = ses.raws.get(arch) or {}
        for model, raw in raws.items():
            e_parity, detail = external_parity(cfg, raw)
            components = _components(raw.base.reindex(plot_cats(cfg), fill_value=0.0))
            # The baseline's DRAM cost is a PRICE (70 pJ/bit against 40), not
            # extra traffic: decoding is on the DRAM die, so its parity never
            # crosses the datapath. `detail` stays on the result as the array-
            # SIZE evidence that price is charged for. See baseline_dram.py;
            # with ECC_BASELINE_DRAM_PJ_PER_BIT unset this is the old model.
            pricing = baseline_dram.charge(cfg, raw, components, e_parity)
            total = float(sum(components.values()))

            builder = ResultBuilder(
                cfg, ses.results, arch, model,
                experiment="task1_conventional_ecc_baseline",
                fixed_mapping=True)

            mapping_ids = sorted({lp["mapping_id"] for lp in raw.per_layer
                                  if lp.get("mapping_id")})
            builder.add(Variant(
                "baseline_external_parity", kind="baseline", status="evaluated",
                total_energy_pJ=total, energy_by_component_pJ=components,
                mapping_ids=mapping_ids,
                label="Conventional ECC, BCH parity external in DRAM",
                extra={"external_parity_accounting": detail,
                       "baseline_dram_pricing": pricing,
                       "timeloop_energy_pJ": float(raw.total),
                       "dram_weight_reads": raw.dram_w_reads,
                       "dram_weight_energy_pJ": raw.e_dram_w}))
            _placeholder_variants(builder)

            # ---- the checks that make the number auditable -----------------
            builder.check("parity_payload_hand_check", detail["hand_check_passed"],
                          detail["hand_check"])
            builder.check("all_layers_mapped", raw.layers_skipped == 0,
                          f"{raw.layers_ok} mapped, {raw.layers_skipped} skipped")
            builder.check("mapping_ids_recorded", bool(mapping_ids),
                          f"{len(mapping_ids)} distinct mapping(s) behind this result")
            arch_report = validate_arch(arch, cfg)
            builder.check("architecture_obeys_shared_contract",
                          not arch_report["violations"],
                          {"violations": arch_report["violations"],
                           "notes": arch_report["notes"]})

            # ---- interconnect: was it costed, and does it look like the paper?
            # A zero here with the NoC enabled means the coefficients never
            # reached the networks that carry hops -- the failure mode that
            # kept every NoC at 0.00 before archs/_shared/noc.yaml existed.
            noc_pj = float(raw.base.get("NoC", 0.0))
            noc_share = noc_pj / raw.total if raw.total > 0 else 0.0
            terms = noc_terms(arch, cfg) if cfg.noc_enabled else {}
            builder.check("noc_energy_costed",
                          (not cfg.noc_enabled) or noc_pj > 0,
                          {"enabled": cfg.noc_enabled, "noc_pJ": noc_pj,
                           "share_of_timeloop_total": noc_share,
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
                # JETCAS 2019 Sec. V / Fig. 18: the hierarchical mesh is
                # "6%-10% of the total energy consumption". The published band
                # is for the paper's own workloads, so a miss on one model is a
                # prompt to look, not proof the constants are wrong -- but a
                # miss on EVERY model is.
                builder.check("noc_share_within_published_band_eyeriss_v2",
                              0.06 <= noc_share <= 0.10,
                              {"modelled_share": noc_share, "published_band": [0.06, 0.10],
                               "source": "arXiv:1807.07928 Sec. V, Fig. 18",
                               "read_as": "published share is for SPARSE runs at 65nm; this "
                                          "model is dense (pushes the share up) at 45nm "
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

            # ---- what a reader must not over-read --------------------------
            builder.approximate(detail["traffic"]["method"])
            builder.approximate(baseline_dram.caveat(pricing))
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
                mappings=[(ses.mappers.get(arch).mappings if arch in ses.mappers
                           else {})],
                raw_energy_cache=str(
                    ses.results.raw_path(arch, model, None, ses.fingerprints.get(arch))),
                mapper_cache=str(ses.results.mapper_cache(
                    arch, None, ses.fingerprints.get(arch))),
            )

            _report(cfg, arch, model, raw, total, detail, pricing)
            path = builder.write()
            written.append(path)
            print(f"    -> {path}")

    if not written:
        raise SystemExit("nothing was evaluated; see the [skip] lines above")

    print("\n" + "=" * 78)
    print(f"Task 1 baseline: {len(written)} result file(s) written under "
          f"{ses.results.evaluation}")
    print("Every file also carries `embedded_ecc` and the reconstruction")
    print("placements as status=not_implemented, with no energy number.")
    print("=" * 78)
    return written
