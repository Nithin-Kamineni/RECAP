"""recon: Task 3 -- reconstruction placement on ONE architecture, fixed mapping.

    bash run.sh recon --eval                     # the whole model
    bash run.sh recon --layers "layer3.0.downsample.0 layer4.1.conv2" --eval
    bash hpc/tl.sh bash run.sh recon --eval      # HiPerGator (host python has no pandas)

or, from env.sh, by setting `ECC_RECON_MODELING=1` and running the one command:

    bash hpc/run_all.sh --eval-only

WHAT IT COMPUTES
----------------
One result file and one figure holding, side by side and from ONE set of cached
mappings and ONE raw energy record:

  baseline_external_parity   Task 1's conventional-ECC number, from
                             `ecc.external_parity()` -- the same function and
                             the same digits `bash run.sh baseline` writes.
  embedded_ecc               Task 2's number, from `ecc.embedded_dram()`. This
                             is the reference the plan names for Task 3.
  recon1 .. recon5           the five weight-path boundaries of
                             `01_project_context_and_architectures.txt` Sec. 5.2,
                             each either evaluated or explicitly `unsupported`.

WHY THIS IS ONE ARCHITECTURE AT A TIME
--------------------------------------
The three sweeps put architectures on an axis because all three ECC ARMS exist
on every design. A reconstruction BOUNDARY does not: Eyeriss v2's boundaries
are its mesh, its cluster-local fanout and its PE scratchpad, and a
weight-stationary design's are a different list. Drawing them on one axis would
put "reconstruct after the mesh" next to a design that has no mesh. So the
architecture is held, the boundary is the axis, and Task 5 repeats the study
per design.

WHAT IS HELD FIXED, AND CHECKED RATHER THAN ASSERTED
-----------------------------------------------------
Everything except the boundary. The result file proves it:

  fixed_mapping_shared_across_variants   the same mapping ids behind every bar
  evaluation_only_rerun                  the mapper was not invoked
  weight_path_reconciles_with_raw_record this module's own re-parse of the
                                         cached stats reproduces the `Raw`
                                         record's weight energy per category
  dram_array_identical_to_embedded_reference
                                         no placement credits the DRAM ARRAY:
                                         (1 - f_if) of the DRAM weight energy
                                         is the embedded arm's on every bar
  dram_interface_scaled_by_K_over_N      every placement's DRAM INTERFACE share
                                         (f_if of it) is x K/N -- the decoder
                                         is on the DRAM die and only the k
                                         message bits leave it -- and no bar
                                         leaves it at full width
  non_weight_energy_identical            inputs, partial sums and compute are
                                         the same numbers under every bar
  placement_savings_are_bounded_by_the_weight_path
                                         no bar saves more than the reducible
                                         weight energy that exists

THE DRAM TERM, SINCE 2026-09-09
-------------------------------
The BCH decoder is on the DRAM die and off the fetch path (01_project_context
Sec. 1 and 4), so the DRAM weight energy is two weight-path stages: the array
share `(1 - f_if)`, never reduced, and the interface share `f_if`, scaled by
K/N under every boundary because only the k message bits leave the die. `f_if`
is `ECC_DRAM_IF_FRAC` and has no cited default: unset, `run()` prints the DRAM
ceiling at `recon.F_IF_SENSITIVITY` and refuses. `ECC_RECON_DECODE_SITE=
controller` reproduces the pre-2026-09-09 numbers (complete codeword across the
interface) for the diff. The figure subtitle, the table, the manifest and every
result file carry the decode site and f_if.

WHAT IS OUTSIDE THE COMPARISON
------------------------------
ECC codec (decode/correction) energy, exactly as in Task 2: charged to nobody
unless `ECC_DECODE=1`, and the file says so. Moving the decoder onto the DRAM
die does not change that: it is the same decoder, run once per corrected
codeword, and it cancels between the embedded bar and every placement. Latency and area: Task 3 asks for
energy, Task 4 for "relevant latency/area overhead". No inference-accuracy
claim is made anywhere.
"""
from __future__ import annotations

import math

import pandas as pd

from .. import archs as archmod
from .. import recon as reconmod
from .. import timeloop as tlmod
from ..archs import accumulator_bits, load_provenance
from ..ecc import embedded_dram, external_parity, load_recon_energy
from ..energy import plot_cats
from ..plots.stacked import grouped_stacks, write_table
from ..results_store import ResultBuilder, Variant
from . import audit
from .common import Session

EXPERIMENT = "task3_reconstruction_placement_fixed_mapping"

PARITY_KEY = "DRAM external BCH parity"

#: Categories the per-bar "on chip" note leaves out. DRAM because it is not on
#: chip -- its interface saving, common to every boundary, gets its own line of
#: the note rather than being folded into the on-chip figure; reconstruction, its overhead and the codec because they are what a boundary
#: COSTS, and the panel is scoped to what it can save; external parity because
#: it exists only on the conventional bar. Uniform across every bar, so no
#: boundary is flattered by the scoping.
ONCHIP_EXCLUDE = ("DRAM", "Compute", "Reconstruction", "Recon overhead",
                  "ECC decode", PARITY_KEY)

CODEC_NOTE = (
    "ECC codec (decode/correction) energy is OUTSIDE this comparison unless "
    "ECC_DECODE=1, exactly as in Task 2. Every placement decodes the same "
    "codewords as the embedded reference -- the reduced representation is taken "
    "AFTER correction -- so a characterised codec cost would cancel between the "
    "embedded bar and every recon bar, and would not cancel against the "
    "conventional baseline, which counts codewords by a different layout. "
    "Since 2026-09-09 the decoder sits on the DRAM die (ECC_RECON_DECODE_SITE="
    "ondie), off the fetch path; it is the same BCH decoder relocated, runs once "
    "per corrected codeword, is DRAM-process logic, and stays outside the "
    "placement comparison.")

F_IF_NOTE = (
    "f_if, the interface share of Accelergy's flat per-bit DRAM energy, is an "
    "ASSUMED value, not a cited one: archs/_shared/provenance.yaml "
    "`dram_interface_share` records an HBM2 breakdown (I/O 0.3 of 3.92 pJ/bit, "
    "an interposer lower bound for LPDDR4) and an LPDDR4 figure whose I/O value "
    "is only plotted, and the study assumes 0.40 (env.sh, 2026-09-09) pending a "
    "read-off value. The DRAM interface saving is exactly f_if x (1 - K/N) x the "
    "DRAM weight energy on every placement, so it rescales linearly with f_if and "
    "changes no ordering among the placements.")

FIXED_MAPPING_NOTE = (
    "FIXED-MAPPING RESULT. Every access count, fanout, multicast factor and hop "
    "count here is the one Timeloop chose WITHOUT knowing about reconstruction, "
    "and the mapper was not re-run for any placement. So this file answers "
    "'what does moving the boundary cost and save at constant data movement', "
    "not 'what is the best this architecture can do' -- packed reduced weights "
    "raise the effective weight capacity and may enable better tiling, which is "
    "Task 4's question and is deliberately not credited here.")


# ---------------------------------------------------------------- stats paths
def stats_paths_for(cfg, ses, arch, model):
    """`{shape name: cached stats.txt}` for every mapped layer of this model.

    The placement model needs four things the aggregated `Raw` record does not
    carry -- total access counts, utilized capacity, physical word width and the
    loop nest -- so it re-reads the SAME cached Timeloop output the raw record
    was built from. The mapper is constructed only to resolve cache paths; with
    `ECC_FROM_CACHE=1` it cannot invoke Timeloop, and a shape that is not in the
    cache resolves to nothing and is skipped, exactly as `energy.gather()` skips
    it.
    """
    variant = archmod.effective_variant(arch, cfg)
    fingerprint = ses.fingerprints.get(arch) or archmod.arch_fingerprint(arch, cfg)
    mapper = tlmod.Mapper(
        cfg, arch, None, ses.results.mapper_cache(arch, variant, fingerprint),
        None, fingerprint=fingerprint,
        legacy_root=ses.results.legacy_mapper_cache(arch, variant))
    out = {}
    for layer in ses.models[model]:
        if layer.shape_name in out:
            continue
        path = mapper.stats_for(layer)
        if path is not None:
            out[layer.shape_name] = path
    return out, mapper


# --------------------------------------------------------------------- report
def _report(cfg, arch, model, wpath, gran, packing, rows, base_total, emb_total,
            raw=None):
    """The console table. Prints the CEILING as well as the result, because a
    0.06 % saving is only legible once a reader can see how much reducible
    energy there was to begin with."""
    acc, _ = accumulator_bits(arch, cfg)
    print(f"\n  --- {arch} / {model} : Task 3, reconstruction placement, "
          f"FIXED MAPPING ---")
    print(f"    precisions            : weight {cfg.weight_bits}b   "
          f"activation {cfg.activation_bits}b   accumulator {acc}b (published)")
    mac = getattr(raw, "mac", None) or {}
    print(f"    MAC energy            : {mac.get('pj_per_mac_charged', float('nan')):.5f} "
          f"pJ/op x {mac.get('macs', 0):,.0f} MACs = "
          f"{mac.get('compute_pJ_charged', 0.0) / 1e6:,.3f} uJ "
          f"({mac.get('source', '?')}"
          f"{'; ' + mac.get('citation_short', '') if cfg.mac_pj_override is not None else ''}) "
          f"<-- the DENOMINATOR of every percentage below")
    if cfg.mac_pj_override is not None:
        print(f"                            ERT was {mac.get('ert_pj_per_mac', float('nan')):.5f} "
              f"pJ/op = {mac.get('compute_pJ_ert', 0.0) / 1e6:,.3f} uJ; saved uJ are "
              f"unchanged, only the total moved")
    print(f"    reduced form          : {packing.reduced_bits_per_weight:.4f} of "
          f"{cfg.weight_bits} bits per weight retained "
          f"(K/N = {packing.frac:.4f}), packing={packing.mode}")
    print(f"    ECC group             : {gran.weights_per_codeword:.4f} weights per "
          f"codeword, G_rec = {gran.g_rec} weights must be co-resident")
    ret = wpath.retention
    # The register's WIDTH depends on what it holds (recon.ReuseRegister), and
    # so does the PE-storage figure -- print both, because `1.00x the baseline`
    # under `complement` versus `1.81x` under `full_width` is the whole of R4b's
    # capacity story and it used to be reported as the latter unconditionally.
    reg_w = ret["register_width_bits"]
    pe_bits = packing.reduced_bits_per_weight + reg_w
    print(f"    reuse register        : {ret['register_entries_requested']} entries "
          f"(max {ret['register_entries_required_max']:,} x {reg_w:.4g}b "
          f"required) -> reconstructions amortized "
          f"{ret['amortization_vs_no_retention']:,.1f}x vs no retention")
    print(f"    PE weight storage     : {packing.reduced_bits_per_weight:.4g}b "
          f"reduced SPad + {reg_w:.4g}b register = {pe_bits:.4g}b per resident "
          f"weight = {pe_bits / cfg.weight_bits:.2f}x the baseline PE "
          f"(register model: {ret['register_mode']})")
    latch = min([(r["consecutive_run_a_one_entry_latch_would_serve"] or 1)
                 for r in ret["per_layer"]] or [1])
    dims = sorted({r["innermost_weight_dimension"] for r in ret["per_layer"]
                   if r["innermost_weight_dimension"]})
    print(f"    a ONE-ENTRY latch     : would serve {latch:.0f} consecutive "
          f"use(s) -- the innermost temporal loop below the buffer walks "
          f"{'/'.join(dims) or '?'}, a WEIGHT dimension, so a latch is useless "
          f"and R4b needs the register above")

    print(f"\n    the weight path as modelled (weight energy only):")
    print(f"      {'stage':22s} {'kind':8s} {'energy uJ':>12s} {'reads':>15s} "
          f"{'fills':>13s} {'ingress':>15s} {'resident':>9s} {'reducible':>10s}")
    total = base_total  # only for the share column below
    for stage in reconmod.stages_for(arch, cfg):
        st = wpath.stages.get(stage.key)
        if st is None or st.energy_pJ <= 0:
            why = ("f_if unset: booked to dram_array (controller-side correction)"
                   if stage.key == "dram_interface"
                   else "absent from this architecture as modelled")
            print(f"      {stage.key:22s} {stage.kind:8s} {why:>12s}")
            continue
        red = ("yes" if stage.reducible else
               "NO (array)" if stage.key == "dram_array" else
               "NO (ctrl)" if stage.key == "dram_interface" else "NO (ECC)")
        print(f"      {stage.key:22s} {st.kind:8s} {st.energy_pJ / 1e6:12,.3f} "
              f"{st.reads:15,.0f} {st.fills:13,.0f} {st.ingresses:15,.0f} "
              f"{st.utilized_capacity:9,.0f} {red:>10s}")

    # THE CEILING. Without this, a 0.06% result reads as a missing term rather
    # than as arithmetic. It is the honest answer to "where did the storage
    # saving go": there is only this much of it on this architecture.
    # ON-CHIP only: the DRAM interface is reducible too, but it is not on chip
    # and gets its own lines below.
    reducible = [s.key for s in reconmod.stages_for(arch, cfg)
                 if s.reducible and s.kind != "dram"]
    on_chip = wpath.reducible_energy(reducible)
    print(f"\n    CEILING on any boundary's saving:")
    print(f"      on-chip weight energy that CAN be reduced : "
          f"{on_chip / 1e6:12,.3f} uJ = {on_chip / emb_total * 100:5.2f}% of the "
          f"embedded total")
    print(f"      x (1 - K/N) = x {1 - packing.frac:.4f}                     : "
          f"{on_chip * (1 - packing.frac) / 1e6:12,.3f} uJ = "
          f"{on_chip * (1 - packing.frac) / emb_total * 100:5.2f}%   <-- the most "
          f"ANY placement can save")
    split = wpath.dram_split()
    dram_w = split["dram_weight_energy_pJ"]
    arr, ifc = split["dram_array_pJ"], split["dram_interface_pJ"]
    ondie = split["decode_site"] == "ondie"
    print(f"      DRAM weight energy, both shares           : "
          f"{dram_w / 1e6:12,.3f} uJ = {dram_w / emb_total * 100:5.2f}%   "
          f"(f_if = {split['f_if']:.4f}; {split['f_if_provenance']})")
    print(f"        array share (1 - f_if), NOT reducible   : "
          f"{arr / 1e6:12,.3f} uJ = {arr / emb_total * 100:5.2f}%   (the complete "
          f"codeword is read for on-die correction)")
    if ondie:
        print(f"        interface share f_if, x K/N on EVERY bar: "
              f"{ifc / 1e6:12,.3f} uJ = {ifc / emb_total * 100:5.2f}%   (only the "
              f"k message bits leave the die)")
        dram_save = ifc * (1 - packing.frac)
        print(f"        interface x (1 - K/N)                   : "
              f"{dram_save / 1e6:12,.3f} uJ = {dram_save / emb_total * 100:5.2f}%   "
              f"<-- the DRAM saving every boundary shares")
        both = (on_chip + ifc) * (1 - packing.frac)
        print(f"      on-chip + interface, x (1 - K/N)          : "
              f"{both / 1e6:12,.3f} uJ = {both / emb_total * 100:5.2f}%   <-- the "
              f"most ANY placement can save in total")
    else:
        print(f"        interface share f_if, NOT reduced       : "
              f"{ifc / 1e6:12,.3f} uJ = {ifc / emb_total * 100:5.2f}%   "
              f"(ECC_RECON_DECODE_SITE=controller: the complete codeword "
              f"crosses the interface -- the pre-2026-09-09 model, for the diff)")

    print(f"\n    {'bar':26s} {'total uJ':>13s} {'vs base':>9s} {'vs emb':>9s} "
          f"{'DRAM i/f uJ':>12s} {'recon uJ':>12s} {'overhead uJ':>12s} {'N_rec':>16s}")
    for r in rows:
        if r["status"] != "evaluated":
            print(f"    {r['label'][:26]:26s} {'UNSUPPORTED':>13s}   {r['reason'][:60]}")
            continue
        vb = (base_total - r["total"]) / base_total * 100 if base_total else 0.0
        ve = (emb_total - r["total"]) / emb_total * 100 if emb_total else 0.0
        print(f"    {r['label'][:26]:26s} {r['total'] / 1e6:13,.3f} {vb:8.2f}% "
              f"{ve:8.2f}% {-r['dram_if_saved_pJ'] / 1e6:12,.3f} "
              f"{r['recon_pJ'] / 1e6:12,.3f} "
              f"{r['overhead_pJ'] / 1e6:12,.3f} {r['n_cw']:16,.0f}")
    print(f"    (vs base / vs emb are SAVINGS: a negative number costs more "
          f"than the reference; DRAM i/f is the interface energy the bar "
          f"REMOVED against the embedded reference)")


# ------------------------------------------------------------------- evaluate
def evaluate(cfg, ses, prov, arch, model, raw):
    cats = plot_cats(cfg)
    base_series = raw.base.reindex(cats, fill_value=0.0)
    base_w = raw.base_w.reindex(cats, fill_value=0.0)

    # ---- the two reference bars, from Task 1's and Task 2's own functions ---
    e_parity, pdetail = external_parity(cfg, raw)
    e_external, edetail = embedded_dram(cfg, raw)

    base_components = audit.components(base_series)
    base_components[PARITY_KEY] = e_parity
    base_total = float(sum(base_components.values()))

    emb_components = audit.components(base_series)
    emb_components[PARITY_KEY] = e_external          # 0.0 by construction
    emb_total = float(sum(emb_components.values()))

    # ---- the weight path, re-read from the cached Timeloop output -----------
    paths, mapper = stats_paths_for(cfg, ses, arch, model)
    wpath = reconmod.weight_path(cfg, arch, model, ses.models[model], paths)
    reconciles, recon_check = reconmod.cross_check(cfg, arch, wpath, base_w)

    gran = reconmod.Granularity(cfg.code_n, cfg.code_k, cfg.weight_bits,
                                cfg.recon_granularity)
    packing = reconmod.Packing(cfg.recon_packing, cfg.weight_bits,
                               cfg.code_k, cfg.code_n)
    recon_pj, recon_prov = load_recon_energy(cfg)
    if cfg.recon_reuse_reg_pj is not None:
        reg_pj = cfg.recon_reuse_reg_pj
        reg_prov = f"ECC_RECON_REUSE_REG_PJ override ({reg_pj} pJ per write)"
    else:
        any_dir = next(iter(paths.values()), None)
        reg_pj, reg_prov = reconmod.reuse_register_pj(
            any_dir.parent if any_dir is not None else ".")

    # Decode: charged to nobody by default (Task 2's rule). When ECC_DECODE=1
    # the embedded layout's codeword count is what every recon bar decodes,
    # because the reduced form is taken AFTER correction.
    n_cw_emb = raw.dram_w_reads / gran.weights_per_codeword
    decode_emb = n_cw_emb * cfg.decode_pj_emb if cfg.decode_enabled else 0.0
    decode_placement = (decode_emb if cfg.recon_placement_charges_decode else 0.0)

    # ---- the five placements ------------------------------------------------
    wanted = cfg.recon_placement_keys
    placements = [p for p in reconmod.placements_for(arch, cfg)
                  if not wanted or p.key in wanted or p.variant in wanted]
    results = [reconmod.evaluate_placement(
        cfg, arch, p, wpath, base_w, base_series, recon_pj, reg_pj, gran,
        packing, decode_pj=decode_placement) for p in placements]

    # ---- the result file ---------------------------------------------------
    builder = ResultBuilder(cfg, ses.results, arch, model,
                            experiment=EXPERIMENT, fixed_mapping=True)
    mapping_ids = audit.mapping_ids_of(raw)
    builder.add(Variant(
        "baseline_external_parity", kind="baseline", status="evaluated",
        total_energy_pJ=base_total, energy_by_component_pJ=base_components,
        mapping_ids=mapping_ids,
        label="Conventional ECC, BCH parity external in DRAM",
        extra={"external_parity_accounting": pdetail,
               "timeloop_energy_pJ": float(raw.total),
               "ecc_codec_energy": CODEC_NOTE}))
    builder.add(Variant(
        "embedded_ecc", kind="embedded", status="evaluated",
        total_energy_pJ=emb_total, energy_by_component_pJ=emb_components,
        mapping_ids=mapping_ids,
        label="Embedded ECC, parity inside the stored weights, no external parity",
        extra={"embedded_dram_accounting": edetail,
               "dram_energy_removed_vs_conventional_pJ": e_parity,
               "role": ("the reference Task 3 measures reconstruction against: "
                        "every placement reads the same complete codeword out of "
                        "the DRAM ARRAY as this bar; with the decoder on the DRAM "
                        "die (ECC_RECON_DECODE_SITE=ondie) every placement's DRAM "
                        "INTERFACE share is this bar's x K/N, because this bar's "
                        "on-chip datapath consumes every weight bit and so its "
                        "complete codeword still crosses the interface"),
               "dram_split": wpath.dram_split(),
               "ecc_codec_energy": CODEC_NOTE}))

    rows = []
    for res in results:
        p = res.placement
        label = f"{p.label}  [hypothesis {p.rating}]"
        if res.status != "evaluated":
            builder.add(Variant(
                p.variant, kind="reconstruction", status="unsupported",
                unavailable_reason=res.reason, label=label,
                extra=dict(res.detail, placement_key=p.key)))
            rows.append({"key": p.key, "label": p.short.replace("\n", " "),
                         "status": res.status, "reason": res.reason,
                         "total": None, "recon_pJ": 0.0, "overhead_pJ": 0.0,
                         "n_cw": 0.0, "dram_if_saved_pJ": 0.0})
            continue
        counts = res.detail["reconstruction_counts"]
        builder.add(Variant(
            p.variant, kind="reconstruction", status="evaluated",
            total_energy_pJ=res.total_pJ,
            energy_by_component_pJ={k: v for k, v in res.components.items()
                                    if v != 0.0},
            mapping_ids=mapping_ids, label=label,
            extra=dict(res.detail, placement_key=p.key,
                       fixed_mapping=FIXED_MAPPING_NOTE,
                       ecc_codec_energy=CODEC_NOTE)))
        rows.append({"key": p.key, "label": p.short.replace("\n", " "),
                     "status": "evaluated", "reason": "",
                     "total": res.total_pJ,
                     "recon_pJ": counts["reconstruction_energy_pJ"],
                     "overhead_pJ": counts["reuse_register_energy_pJ"],
                     "n_cw": counts["reconstruction_events_codewords"],
                     "dram_if_saved_pJ":
                         res.detail["dram_model"]["dram_interface_saving_pJ"]})

    # ---- the checks that make the comparison auditable ---------------------
    task3_checks(builder, cfg, arch, raw, base_components, emb_components,
                 e_parity, results, wpath, reconciles, recon_check, base_w,
                 newly_mapped=getattr(mapper, "n_mapped", 0))

    # ---- Task 1's checks and caveats, so the baseline bar is as audited ----
    arch_report = audit.common_checks(builder, cfg, arch, raw, pdetail, mapping_ids)
    audit.common_caveats(builder, cfg, arch, raw, pdetail)
    builder.approximate(edetail["traffic"]["method"])
    builder.approximate(CODEC_NOTE)
    builder.approximate(FIXED_MAPPING_NOTE)
    builder.approximate(packing.to_dict()["meaning"])
    builder.approximate(gran.to_dict()["meaning"])
    if cfg.recon_decode_site == "ondie":
        builder.approximate(F_IF_NOTE)
        builder.approximate(
            f"DRAM model (decoder on the DRAM die, off the fetch path): "
            f"dram_array = (1 - f_if) x DRAM weight energy is never reduced -- "
            f"whether a message-only fetch touches fewer array bits depends on "
            f"burst granularity and the codeword layout and is not assumed; "
            f"dram_interface = f_if x DRAM weight energy is x K/N under every "
            f"boundary, f_if = {cfg.dram_if_frac} ({cfg.dram_if_frac_note}).")
    else:
        builder.warn(
            "ECC_RECON_DECODE_SITE=controller: this is the pre-2026-09-09 model "
            "(controller-side correction on the fetch path, complete codeword "
            "across the DRAM interface, DRAM identical on every bar), kept as a "
            "runnable row for the diff. Do not quote it as the study's result.")
    builder.approximate(
        "R4b's reuse register is charged one full-width write per weight it "
        "retains and reduces the RECONSTRUCTION count only. It is not credited "
        "with removing scratchpad reads, because a baseline PE holding the same "
        "operand in the same register would remove exactly as many -- the plan "
        "asks for the reuse register to be compared against equivalent baseline "
        "operand-retention behaviour, and this is that comparison. Scratchpad "
        "read counts are therefore Timeloop's under every bar.")
    builder.approximate(
        f"Weight-path stages are matched to Timeloop levels by name "
        f"(eccenergy/recon.py WEIGHT_PATHS[{arch!r}]) and the match is verified "
        f"against the raw record's per-category weight energy before any "
        f"placement is evaluated. A level carrying weight energy that no stage "
        f"claims fails that check rather than being ignored.")
    if arch.startswith("eyeriss_v2"):
        builder.warn(
            "CSC METADATA IS NOT MODELLED. The published Eyeriss v2 PE stores "
            "weights CSC-compressed with a separate address scratchpad "
            "(Table IV: weight data 288B, weight address 16x7b); this study "
            "models the design DENSE, so there is no metadata array here to "
            "leave at full width. A CSC implementation would apply the reduced "
            "representation to the weight DATA only, so the scratchpad saving "
            "of R4a/R4b is an upper bound on what CSC v2 would see.")
    if cfg.decode_enabled:
        builder.warn(
            "ECC_DECODE=1: codec energy is charged to the reference bars and "
            f"{'to' if cfg.recon_placement_charges_decode else 'NOT to'} the "
            f"placements (ECC_RECON_PLACEMENT_CHARGES_DECODE). No characterised "
            f"codec energy exists in this project, so the value is a parameter, "
            f"not a measurement.")
    audit.common_detail(builder, cfg, ses, arch, model, raw, arch_report, prov)
    builder.set_detail(
        reconstruction_placement={
            "weight_path": wpath.to_dict(),
            "packing": packing.to_dict(),
            "granularity": gran.to_dict(),
            "reconstruction_datapath_pJ_per_codeword": recon_pj,
            "reconstruction_datapath_provenance": recon_prov,
            "reuse_register_pJ_per_write": reg_pj,
            "reuse_register_provenance": reg_prov,
            "decode_site": cfg.recon_decode_site,
            "dram_if_frac": cfg.dram_if_frac,
            "dram_if_frac_provenance": cfg.dram_if_frac_note,
            "dram_split": wpath.dram_split(),
            "placements_defined": [
                {"key": p.key, "variant": p.variant, "boundary": p.label,
                 "hypothesised_rating": p.rating,
                 "reduced_stages": list(p.reduced),
                 "reconstruction_site": p.site_stage,
                 "access_counter": p.site_counter,
                 "reuse_register": p.reuse_register,
                 "description": p.description}
                for p in reconmod.placements_for(arch, cfg)],
            "placements_requested": wanted or "all",
        })

    _report(cfg, arch, model, wpath, gran, packing, rows, base_total, emb_total,
            raw=raw)
    path = builder.write()
    print(f"    -> {path}")
    return {"path": path, "rows": rows, "results": results, "wpath": wpath,
            "base_components": base_components, "emb_components": emb_components,
            "base_total": base_total, "emb_total": emb_total,
            # the WEIGHT share of each plotted category, straight off the `Raw`
            # record. The figure's lower panel is scoped to it, and
            # `cross_check` has already proved it equals this module's own
            # re-parse of the cached stats before anything was evaluated.
            "weight_by_category": {c: float(v) for c, v in base_w.items()},
            "mac": getattr(raw, "mac", None),
            "builder": builder}


def task3_checks(builder, cfg, arch, raw, base_components, emb_components,
                 e_parity, results, wpath, reconciles, recon_check, base_w,
                 newly_mapped=0):
    """The checks that make Task 3's claim -- "only the boundary moved" -- auditable.

    Kept free of the Session so the offline test can drive it with a synthetic
    record, exactly as `embedded.task2_checks` is.
    """
    builder.check("weight_path_reconciles_with_raw_record", reconciles, recon_check)

    # 0. the two tables that define the study still agree with each other. A
    #    reducible stage no boundary reduces is not a small omission: every
    #    boundary below it keeps its own saving and reports that stage at FULL
    #    width, so the whole placement list is understated and nothing else in
    #    this file would say so.
    space_ok, space = reconmod.validate_placement_space(arch, cfg)
    builder.check("placement_space_covers_the_whole_weight_path", space_ok, space)

    evaluated = [r for r in results if r.status == "evaluated"]

    # 1. the DRAM term, in BOTH directions. With the decoder on the DRAM die the
    #    array share is the embedded arm's and the interface share is x K/N, so
    #    every placement's DRAM component must sit at EXACTLY
    #        emb_DRAM - f_if x DRAM_w x (1 - K/N).
    #    That equality is split into two one-sided checks on purpose: a bar
    #    that quietly credits the array sits BELOW it and fails the first; a
    #    bar that leaves the interface at full width sits ABOVE it and fails
    #    the second. Each also reads the placement's own dram_model rows, so a
    #    stack cannot say one thing and its detail another. Under `controller`
    #    the expected value is emb_DRAM and both checks collapse onto the
    #    pre-2026-09-09 `dram_identical_to_embedded_reference`.
    dram_ref = emb_components.get("DRAM", 0.0)
    dram_w = float(base_w.get("DRAM", 0.0))
    split = wpath.dram_split()
    ondie = split["decode_site"] == "ondie"
    frac = cfg.code_k / cfg.code_n
    f_if = float(split["f_if"] or 0.0)
    if_saving = f_if * dram_w * (1.0 - frac) if ondie else 0.0
    expected = dram_ref - if_saving
    tol = abs(expected) * 1e-12 + 1e-6
    rows_a, rows_i, ok_a, ok_i = {}, {}, True, True
    for r in evaluated:
        got = r.components.get("DRAM", 0.0)
        dm = r.detail.get("dram_model", {})
        a_row_ok = (not dm.get("dram_array_reduced", False)
                    and math.isclose(dm.get("dram_array_after_pJ", 0.0),
                                     dm.get("dram_array_pJ", 0.0),
                                     rel_tol=1e-12, abs_tol=1e-6))
        a_ok = a_row_ok and got >= expected - tol
        rows_a[r.placement.key] = {
            "dram_pJ": got, "expected_dram_pJ": expected,
            "below_expected_by_pJ": max(0.0, expected - got),
            "array_row_unreduced": a_row_ok, "passed": a_ok}
        ok_a = ok_a and a_ok
        want_scale = frac if ondie else 1.0
        i_row_ok = (bool(dm.get("dram_interface_reduced", False)) == ondie
                    and math.isclose(dm.get("dram_interface_after_pJ", 0.0),
                                     dm.get("dram_interface_pJ", 0.0) * want_scale,
                                     rel_tol=1e-12, abs_tol=1e-6))
        i_ok = i_row_ok and got <= expected + tol
        rows_i[r.placement.key] = {
            "dram_pJ": got, "expected_dram_pJ": expected,
            "above_expected_by_pJ": max(0.0, got - expected),
            "interface_row_scaled": i_row_ok, "interface_scale": want_scale,
            "passed": i_ok}
        ok_i = ok_i and i_ok
    common = {"decode_site": split["decode_site"], "f_if": f_if,
              "dram_weight_energy_pJ": dram_w,
              "embedded_reference_dram_pJ": dram_ref,
              "interface_saving_every_placement_pJ": if_saving,
              "expected_placement_dram_pJ": expected}
    builder.check(
        "dram_array_identical_to_embedded_reference", ok_a,
        dict(common, per_placement=rows_a,
             rule=("the array stores and reads the complete codeword for on-die "
                   "correction, so (1 - f_if) of the DRAM weight energy is the "
                   "embedded arm's under every boundary: no placement's DRAM "
                   "component may sit BELOW emb_DRAM - f_if x DRAM_w x (1 - K/N), "
                   "and its dram_array row must be unreduced")))
    builder.check(
        "dram_interface_scaled_by_K_over_N", ok_i,
        dict(common, per_placement=rows_i,
             rule=("with the decoder on the DRAM die only the k message bits "
                   "leave it, so f_if of the DRAM weight energy is x K/N under "
                   "every boundary: no placement's DRAM component may sit ABOVE "
                   "emb_DRAM - f_if x DRAM_w x (1 - K/N), and its dram_interface "
                   "row must carry exactly that scale (1.0 under "
                   "ECC_RECON_DECODE_SITE=controller, where the complete "
                   "codeword crosses the interface)")))

    # 2. nothing outside the weight path moved
    non_weight = {c: float(base_w.get(c, 0.0)) for c in ("Compute",)}
    pairs = {}
    ok = True
    for cat in ("Compute",):
        ref = emb_components.get(cat, 0.0)
        got = {r.placement.key: r.components.get(cat, 0.0) for r in evaluated}
        pairs[cat] = {"embedded": ref, "placements": got}
        ok = ok and all(math.isclose(v, ref, rel_tol=1e-12, abs_tol=1e-6)
                        for v in got.values())
    builder.check(
        "non_weight_energy_identical", ok,
        {"components": pairs, "weight_share_of_those_categories": non_weight,
         "rule": ("a reconstruction boundary touches the WEIGHT path only; MAC "
                  "energy is identical under every bar because both are "
                  "arithmetic on one raw record")})

    # 3. a placement cannot save more than the reducible weight energy there is
    reducible = {}
    for r in evaluated:
        avail = sum(wpath.stages[k].energy_pJ for k in r.placement.reduced
                    if k in wpath.stages)
        saved = sum(r.detail["energy_saved_by_category_pJ"].values())
        reducible[r.placement.key] = {
            "reducible_weight_energy_pJ": avail, "energy_saved_pJ": saved,
            "within_bound": saved <= avail + 1e-6 and saved >= -1e-6}
    builder.check(
        "placement_savings_are_bounded_by_the_weight_path",
        all(v["within_bound"] for v in reducible.values()),
        {"per_placement": reducible,
         "rule": ("the saving is a reduction of the weight energy of the stages "
                  "the reduced form actually reaches; it cannot exceed it and "
                  "cannot be negative")})

    # 4. the placements' stacks reconcile with the reference bars outside the
    #    weight path and the two ECC categories
    ecc_cats = ("Reconstruction", "Recon overhead", "ECC decode", PARITY_KEY)
    drift = {}
    for r in evaluated:
        d = {}
        for cat in set(emb_components) | set(r.components):
            if cat in ecc_cats:
                continue
            a = r.components.get(cat, 0.0)
            b = emb_components.get(cat, 0.0)
            if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-6):
                d[cat] = {"placement_pJ": a, "embedded_pJ": b,
                          "difference_pJ": a - b}
        drift[r.placement.key] = d
    weight_cats = {"DRAM", "Global buffer", "Local (spads/RF)", "NoC"}
    builder.check(
        "only_weight_path_categories_differ_from_the_embedded_reference",
        all(set(d) <= weight_cats for d in drift.values()),
        {"per_placement_differences": drift, "allowed_categories": sorted(weight_cats),
         "rule": ("outside the two ECC categories, a placement may differ from "
                  "the embedded reference only in the categories its weight "
                  "path lives in")})

    # 5. evaluation-only: nothing was mapped by this run
    builder.check(
        "evaluation_only_rerun",
        bool(cfg.from_cache) or newly_mapped == 0,
        {"ECC_FROM_CACHE": bool(cfg.from_cache),
         "newly_mapped_shapes": newly_mapped,
         "meaning": ("passed: every mapping behind this result already existed "
                     "and the mapper was not invoked -- which is what 'do not "
                     "modify or rerun the mapping optimizer for this comparison' "
                     "requires. Failed: re-run with --eval.")})

    # 6. the reference bars still say what Tasks 1 and 2 said
    dram_diff = (base_components.get("DRAM", 0.0) + base_components.get(PARITY_KEY, 0.0)
                 - emb_components.get("DRAM", 0.0) - emb_components.get(PARITY_KEY, 0.0))
    builder.check(
        "reference_bars_match_tasks_1_and_2",
        math.isclose(dram_diff, e_parity, rel_tol=1e-12, abs_tol=1e-6),
        {"baseline_minus_embedded_dram_pJ": dram_diff,
         "external_parity_pJ": e_parity,
         "source": "ecc.external_parity() and ecc.embedded_dram(), unchanged"})


# --------------------------------------------------------------------- figure
def figure(cfg, ses, arch, model, out):
    """ONE image: the boundary on the x axis, the energy breakdown in the bars.

    `plots/stacked.draw_panel` draws it -- the same routine the three sweeps and
    the panel figure use, widened rather than forked. One bar per group here,
    because a placement label is a sentence and does not fit as a rotated tag
    under one of six bars in a single group.

    ONE PANEL, AND WHY THE PER-BAR NOTE CARRIES THE RESULT. On
    `eyeriss_v2_like` the whole reducible weight path is 8.5 % of inference
    energy, so the largest saving any boundary can make is 1.62 % and R2's is
    0.096 % -- about one pixel of a 4,798 uJ bar at any sane figure size. The
    reduction IS applied (the per-stage tests prove it exactly), so the evidence
    is made visible in TEXT rather than in a second pair of axes: each bar
    carries one line giving the uJ moved on chip and the uJ paid for
    reconstruction. A percentage cannot separate 0.096 % from 0.150 %, and on
    this figure those two numbers are the result.

    There used to be a second panel below, the same bars rescaled to the on-chip
    weight path. It was removed because it drew no independent quantity: its two
    bands were the weight share of the NoC and of the scratchpads, both of which
    are already inside this figure's `NoC` and `Local (spads/RF)` bands. The
    numbers it carried survive -- `moved` is the per-bar note here, and the
    reducible energy and the ceiling are columns of the CSV and lines of the
    console run.
    """
    cats = plot_cats(cfg)
    groups, stacks, labels = [], {}, {}
    notes = {}
    emb_components = out["emb_components"]
    weight_by_cat = out["weight_by_category"]
    # The reducible on-chip weight path: every category that holds WEIGHT
    # energy except DRAM. Derived, not listed, so it follows
    # `ECC_SPLIT_READ_WRITE` and picks up a weight global buffer on a design
    # that has one. It scopes the per-bar note, not a second set of axes.
    onchip_cats = [c for c in cats
                   if c not in ONCHIP_EXCLUDE and float(weight_by_cat.get(c, 0.0)) > 0]

    def add(key, label, components):
        # The conventional baseline's external parity is a COMPONENT of the
        # result file, not a plotted category, so it has to be folded into the
        # DRAM band or the drawn bar would silently be 277 uJ shorter than the
        # total it is annotated with. `ecc.build_stacks()` folds it the same
        # way (`col["DRAM"] += e_parity`), so the two figures agree.
        stack = {c: float(components.get(c, 0.0)) for c in cats}
        stack["DRAM"] += float(components.get(PARITY_KEY, 0.0))
        drawn = sum(stack.values())
        total = sum(float(v) for v in components.values())
        if not math.isclose(drawn, total, rel_tol=1e-9, abs_tol=1e-3):
            raise SystemExit(
                f"figure would draw {drawn:.3f} pJ for {key!r} but its total is "
                f"{total:.3f} pJ -- a component of the result is not in a "
                f"plotted category, so the bar and its label would disagree. "
                f"Components: {sorted(components)}")
        groups.append(key)
        labels[key] = label
        stacks[key] = pd.DataFrame(
            {"energy": pd.Series(stack).reindex(cats, fill_value=0.0)})

        # The per-bar note: this bar scoped to the weight energy a boundary can
        # actually reduce. Taken from `components` minus the non-weight share of
        # the same categories, so the note and the bar cannot disagree -- the
        # non-weight share is identical under every bar (checked by
        # `non_weight_energy_identical` and `only_the_weight_share_of_a_category
        # _moves`), so subtracting it is a shift, never a rescale.
        weight_only = {}
        for cat in onchip_cats:
            non_weight = (float(emb_components.get(cat, 0.0))
                          - float(weight_by_cat.get(cat, 0.0)))
            weight_only[cat] = float(components.get(cat, 0.0)) - non_weight
        on_chip = sum(weight_only.values())
        recon = (float(components.get("Reconstruction", 0.0))
                 + float(components.get("Recon overhead", 0.0)))
        ref_on_chip = sum(float(weight_by_cat.get(c, 0.0)) for c in onchip_cats)
        moved = on_chip - ref_on_chip
        # Two SHORT lines. A bar slot is about two inches wide, so a note that
        # runs past ~16 characters collides with its neighbour and, because
        # every figure here is saved `bbox_inches="tight"`, drags the whole
        # image wider with it.
        lines = []
        # The DRAM-interface saving first: it is the same on every R bar (the
        # decoder is on the DRAM die), so seeing it repeated is the point.
        dram_moved = (float(components.get("DRAM", 0.0))
                      - float(emb_components.get("DRAM", 0.0)))
        if abs(dram_moved) > 1e-6 and key not in ("baseline", "embedded"):
            sign = "\u2212" if dram_moved < 0 else "+"
            lines.append(f"{sign}{abs(dram_moved) / 1e6:,.2f} \u00b5J DRAM I/O")
        if abs(moved) > 1e-6:
            # the same U+2212 minus `draw_panel` prints on the saving, so the
            # two annotations on one bar do not use two different signs
            sign = "\u2212" if moved < 0 else "+"
            lines.append(f"{sign}{abs(moved) / 1e6:,.2f} \u00b5J on chip")
        if recon:
            lines.append(f"+{recon / 1e6:,.2f} \u00b5J recon")
        if lines:
            notes[key] = "\n".join(lines)

    add("baseline", reconmod.REFERENCE_BARS[0][1], out["base_components"])
    add("embedded", reconmod.REFERENCE_BARS[1][1], out["emb_components"])
    for res in out["results"]:
        if res.status != "evaluated":
            print(f"  [note] {res.placement.key} is unsupported here and is not "
                  f"drawn: {res.reason}")
            continue
        add(res.placement.key, res.placement.short, res.components)

    # The table carries what Task 3 asks a table to carry: the reconstruction
    # counts, the overheads, the savings against BOTH references, and every
    # boundary -- including the ones the figure cannot draw because they are
    # unsupported. One table per stem, as everywhere else.
    base_t, emb_t = out["base_total"], out["emb_total"]

    def pct(ref, total):
        return ((ref - total) / ref * 100.0) if ref and total is not None else ""

    split = out["wpath"].dram_split()
    extra = {}
    for key, total in (("baseline", base_t), ("embedded", emb_t)):
        extra[key] = {
            "boundary": dict(reconmod.REFERENCE_BARS)[key].replace("\n", " "),
            "status": "reference", "hypothesised_rating": "",
            "saving_vs_conventional_ecc_pct": pct(base_t, total),
            "saving_vs_embedded_only_pct": pct(emb_t, total),
            "reconstruction_events_codewords": "", "reconstruction_uJ": "",
            "recon_overhead_uJ": "", "weights_reconstructed": "",
            "retention_amortization_x": "", "reducible_energy_uJ": "",
            "saving_ceiling_uJ": "",
            "decode_site": "controller (reference bar)",
            "f_if": split["f_if"],
            # the weight share only; the baseline's external parity is on top
            "dram_array_uJ": split["dram_array_pJ"] / 1e6 if key == "embedded" else "",
            "dram_interface_uJ": (split["dram_interface_pJ"] / 1e6
                                  if key == "embedded" else ""),
            "dram_interface_saving_uJ": 0.0 if key == "embedded" else "",
            "unavailable_reason": "",
        }
    for res in out["results"]:
        p = res.placement
        row = {"boundary": p.label, "status": res.status,
               "hypothesised_rating": p.rating}
        if res.status == "evaluated":
            c = res.detail["reconstruction_counts"]
            b = res.detail["reducible_weight_energy_pJ"]
            dm = res.detail["dram_model"]
            row.update(
                saving_vs_conventional_ecc_pct=pct(base_t, res.total_pJ),
                saving_vs_embedded_only_pct=pct(emb_t, res.total_pJ),
                reconstruction_events_codewords=c["reconstruction_events_codewords"],
                reconstruction_uJ=c["reconstruction_energy_pJ"] / 1e6,
                recon_overhead_uJ=c["reuse_register_energy_pJ"] / 1e6,
                weights_reconstructed=c["weights_reconstructed"],
                retention_amortization_x=c["amortization_vs_no_retention"],
                reducible_energy_uJ=b["before_pJ"] / 1e6,
                saving_ceiling_uJ=b["ceiling_on_the_saving_pJ"] / 1e6,
                decode_site=dm["decode_site"], f_if=dm["f_if"],
                dram_array_uJ=dm["dram_array_after_pJ"] / 1e6,
                dram_interface_uJ=dm["dram_interface_after_pJ"] / 1e6,
                dram_interface_saving_uJ=dm["dram_interface_saving_pJ"] / 1e6,
                unavailable_reason="")
        else:
            row.update(saving_vs_conventional_ecc_pct="",
                       saving_vs_embedded_only_pct="",
                       reconstruction_events_codewords="", reconstruction_uJ="",
                       recon_overhead_uJ="", weights_reconstructed="",
                       retention_amortization_x="", reducible_energy_uJ="",
                       saving_ceiling_uJ="", decode_site=split["decode_site"],
                       f_if=split["f_if"], dram_array_uJ="", dram_interface_uJ="",
                       dram_interface_saving_uJ="", unavailable_reason=res.reason)
            # An unsupported boundary has no bar, but it must still have a row:
            # a table that simply omits it reads as "not considered".
            groups.append(p.key)
            labels[p.key] = p.short
            stacks[p.key] = pd.DataFrame(
                {"energy": pd.Series({c: 0.0 for c in cats}).reindex(cats)})
        extra[p.key] = row

    drawn = [g for g in groups if float(stacks[g]["energy"].sum()) > 0]
    ref = {g: base_t for g in groups}
    figs, csv = grouped_stacks(
        cfg, ses.results,
        groups=drawn, stacks={g: stacks[g] for g in drawn}, group_labels=labels,
        title=cfg.recon_title(mac_ert_pj=(out.get("mac") or {}).get("ert_pj_per_mac")),
        stem=cfg.stem,
        group_fontsize=15, bars=["energy"], bar_tags={}, bar_width=0.92,
        ref_totals=ref, bar_notes={g: notes[g] for g in drawn if g in notes})
    # ...and the table gets every row, drawn or not.
    csv = write_table(cfg, ses.results, [(None, groups, stacks, labels)],
                      cfg.stem, bars=["energy"], ref_totals=ref,
                      extra_columns=extra)
    return figs, csv, drawn


# ------------------------------------------------------------ the refusal
def refuse_without_f_if(cfg, arch, model, raw):
    """No f_if, no number: print the DRAM ceiling at several values and stop.

    01_project_context Sec. 4: "f_if ... comes from a cited DRAM energy
    breakdown, not from code; when it is uncertain, print the DRAM saving at
    several values." The whole of what f_if decides is one product -- every
    placement's DRAM interface saving is f_if x DRAM_w x (1 - K/N), the same
    on every bar -- so the ceiling can be printed from the raw record alone,
    before anything is evaluated, and the run then refuses rather than
    drawing a figure whose largest new term is an assumption.
    """
    dram_w = float(raw.base_w.get("DRAM", 0.0))
    total = float(raw.total)
    frac = cfg.code_k / cfg.code_n
    print(f"\n  --- {arch} / {model} : the DRAM interface saving, at several "
          f"f_if, because ECC_DRAM_IF_FRAC is unset ---")
    print(f"    DRAM weight energy (both shares)     : {dram_w / 1e6:12,.3f} uJ = "
          f"{dram_w / total * 100:5.2f}% of the Timeloop total")
    print(f"    K/N = {frac:.4f}, so the interface share falls by x {1 - frac:.4f}")
    print(f"    {'f_if':>6s} {'array (1-f_if) uJ':>19s} {'interface f_if uJ':>19s} "
          f"{'interface x (1-K/N) uJ':>24s} {'of total':>9s}")
    for f in reconmod.F_IF_SENSITIVITY:
        save = f * dram_w * (1 - frac)
        print(f"    {f:6.2f} {(1 - f) * dram_w / 1e6:19,.3f} {f * dram_w / 1e6:19,.3f} "
              f"{save / 1e6:24,.3f} {save / total * 100:8.2f}%")
    print(f"    (every reconstruction boundary saves exactly the last column, in "
          f"addition to its own on-chip saving; the reference bars do not move)")
    raise SystemExit(
        "\nREFUSED: " + reconmod.NO_F_IF
        + f"\n  -> e.g. ECC_DRAM_IF_FRAC=0.25 ECC_RECON_MODELING=1 bash run.sh "
          f"recon --eval   (labelled on the figure as set for the run)")


# ------------------------------------------------------------------------ run
def run(cfg):
    if cfg.phase != "Pre":
        raise SystemExit(
            f"ECC_PHASE={cfg.phase} but Task 3 is a `Pre` result by "
            f"construction: the mapping is fixed and ECC-unaware, and the "
            f"placement effect is applied when evaluating. `Post` is Task 4, "
            f"where the mapping itself is optimised for the reduced width.")

    arch, model = cfg.archs[0], cfg.models[0]
    if arch not in reconmod.WEIGHT_PATHS:
        raise SystemExit(
            f"no weight path is defined for {arch!r}, so its reconstruction "
            f"boundaries are unknown.\n"
            f"  defined: {', '.join(reconmod.supported_archs())}\n"
            f"  -> Task 5 adds the rest; each design's weight path and its list "
            f"of feasible boundaries go in eccenergy/recon.py WEIGHT_PATHS and "
            f"PLACEMENTS, together.")

    space_ok, space = reconmod.validate_placement_space(arch, cfg)
    if not space_ok:
        raise SystemExit(
            f"{arch}'s weight path and its placement list have drifted apart, so "
            f"every boundary below the missing stage would be reported "
            f"UNDERSTATED rather than wrong-looking:\n  "
            + "\n  ".join(space["violations"])
            + f"\n  -> {space['fix']}")

    defined = reconmod.placements_for(arch, cfg)
    unknown = [k for k in cfg.recon_placement_keys
               if k not in {p.key for p in defined}
               and k not in {p.variant for p in defined}
               and k not in ("baseline", "embedded")]
    if unknown:
        raise SystemExit(
            f"ECC_RECON_PLACEMENTS[{arch}] names placements this architecture "
            f"does not define: {', '.join(unknown)}\n"
            f"  defined: "
            f"{', '.join(p.key for p in defined)}\n"
            f"  -> the boundaries are per architecture; see "
            f"eccenergy/recon.py PLACEMENTS")

    ses = Session(cfg).setup()
    ses.collect_all()
    raw = (ses.raws.get(arch) or {}).get(model)
    if raw is None:
        raise SystemExit(f"nothing collected for {arch}/{model}; see the [skip] "
                         f"lines above")

    if cfg.recon_decode_site == "ondie" and cfg.dram_if_frac is None:
        refuse_without_f_if(cfg, arch, model, raw)

    prov = load_provenance()
    out = evaluate(cfg, ses, prov, arch, model, raw)
    figs, csv, groups = figure(cfg, ses, arch, model, out)
    ses.finish(figs, csv, groups, extra={
        "recon_decode_site": cfg.recon_decode_site,
        "dram_if_frac": cfg.dram_if_frac,
        "dram_if_frac_provenance": cfg.dram_if_frac_note,
        "dram_split": out["wpath"].dram_split(),
    })

    best = out["builder"].best_reconstruction()
    print("=" * 78)
    print(f"Task 3 reconstruction placement: {arch} / {model}, fixed mapping.")
    if best.get("variant"):
        print(f"Lowest-energy feasible placement: {best['variant']}")
        print(f"  vs conventional ECC : "
              f"{best.get('savings_vs_conventional_ecc_percent'):+.3f}%")
        print(f"  vs embedded only    : "
              f"{best.get('savings_vs_embedded_only_percent'):+.3f}%")
    else:
        print(f"No placement was evaluated: {best.get('reason')}")
    print("This is an evaluator-only, FIXED-MAPPING result. The mapping optimiser")
    print("was not re-run for any placement -- that is Task 4 (RECON_OPTIMIZER).")
    print("=" * 78)
    return ses
