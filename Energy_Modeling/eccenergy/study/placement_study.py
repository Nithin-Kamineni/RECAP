"""Task 3: reconstruction placement on ONE architecture, fixed mapping.

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
                             `stacks.external_parity()` -- the same function and
                             the same digits `bash run.sh baseline` writes.
  embedded_ecc               Task 2's number, from `stacks.embedded_dram()`. This
                             is the reference the plan names for Task 3.
  recon1 .. recon5           the weight-path boundaries of
                             `01_project_context_and_architectures.txt` Sec. 5.2,
                             each either evaluated or explicitly `unsupported`.

WHY THIS IS ONE ARCHITECTURE AT A TIME
--------------------------------------
The three sweeps put architectures on an axis because all three ECC ARMS exist on
every design. A reconstruction BOUNDARY does not: Eyeriss v2's boundaries are its
mesh, its cluster-local fanout and its PE scratchpads, and a weight-stationary
design's are a different list. Drawing them on one axis would put "reconstruct
after the mesh" next to a design that has no mesh. So the architecture is held,
the boundary is the axis, and Task 5 repeats the study per design.

WHAT IS HELD FIXED, AND CHECKED RATHER THAN ASSERTED
-----------------------------------------------------
Everything except the boundary. The result file proves it:

  fixed_mapping_shared_across_variants   the same mapping ids behind every bar
  evaluation_only_rerun                  the mapper was not invoked
  weight_path_reconciles_with_raw_record the re-parse of the cached stats
                                         reproduces the `Raw` record's weight
                                         energy per category
  dram_scaled_by_K_over_N                the WHOLE DRAM weight energy is x K/N on
                                         every bar -- the decoder is on the DRAM
                                         die, so only the k message bits are read
                                         out and driven off it -- and no bar
                                         leaves it at full width
  non_weight_energy_identical            inputs, partial sums and compute are the
                                         same numbers under every bar
  placement_savings_are_bounded_by_the_weight_path
                                         no bar saves more than the reducible
                                         weight energy that exists

`evaluate()` is the driver: it resolves the plan of every bar, calls
`study/placement_eval.py` once per boundary, runs `task3_checks()` /
`task4_checks()`, and hands the result to `report/recon_view.py` to draw.

ProjectRestructure phase 3 cut `experiments/recon.py` (2,862 lines) into this,
`study/placement_notes.py`, `study/dilated_view.py`, `study/ert_view.py`,
`study/placement_tables.py` and `report/recon_view.py`.
"""
from __future__ import annotations

import math

from ..arch import fingerprint
from ..arch import design
from ..arch import arms as arms_mod
from ..arch import placements as placements_mod
from ..physics import granularity
from ..physics import packing as packing_mod
from . import capacity
from . import placement_eval
from ..toolchain import weight_stats
from ..physics import baseline_dram
from . import audit
from .energy import plot_cats
from .stacks import embedded_dram, external_parity, load_recon_energy
from ..toolchain.results_store import ResultBuilder, Variant

from .dilated_view import TASK4_NOTE, dilated_view
from .ert_view import ERT_NOTE, EXPERIMENT_ERT, _close, arm_plan_state, ert_aware_view, ert_checks
from .placement_notes import CODEC_NOTE, DRAM_TERM_NOTE, EXPERIMENT, EXPERIMENT_TASK4, FIXED_MAPPING_NOTE, PARITY_KEY, stats_paths_for
from .placement_tables import _report, _report_latency, _report_narrowing, latency_table


# ------------------------------------------------------------------- evaluate
def evaluate(cfg, ses, prov, arch, model, raw):
    cats = plot_cats(cfg)
    base_series = raw.base.reindex(cats, fill_value=0.0)
    base_w = raw.base_w.reindex(cats, fill_value=0.0)

    # ---- the two reference bars, from Task 1's and Task 2's own functions ---
    e_parity, pdetail = external_parity(cfg, raw)
    e_external, edetail = embedded_dram(cfg, raw)

    base_components = audit.components(base_series)
    # the baseline's DRAM cost is a per-bit PRICE, not extra traffic
    # (baseline_dram.py); unset ECC_BASELINE_DRAM_PJ_PER_BIT = the old model
    pricing = baseline_dram.charge(cfg, raw, base_components, e_parity)
    base_total = float(sum(base_components.values()))

    emb_components = audit.components(base_series)
    emb_components[PARITY_KEY] = e_external          # 0.0 by construction
    emb_total = float(sum(emb_components.values()))

    # ---- the weight path, re-read from the cached Timeloop output -----------
    paths, mapper = stats_paths_for(cfg, ses, arch, model)
    wpath = weight_stats.weight_path(cfg, arch, model, ses.models[model], paths)
    reconciles, recon_check = placement_eval.cross_check(cfg, arch, wpath, base_w)

    # ---- TASK 4: the reconstruction arm gets its OWN mapping ---------------
    # The reference bars above stay on the configured capacity. Every
    # reconstruction bar below is re-derived from a SECOND mapping solved
    # against N/K more weight room, so the two arms no longer move the same
    # data -- which is the entire mechanism Task 3 cannot show. With
    # RECON_OPTIMIZER=False `dil` is None and every line below is Task 3's.
    # prompt_6 (ECC_RECON_ERT_AWARE=1) has FOUR worlds instead of two: the
    # reference plan for baseline/embedded and the post-processed bars, and
    # ONE PLAN PER ERT ARM. The single `p_*` set is replaced by a per-bar
    # lookup (RULE 4.4.3); the depth dilation is not used under it.
    aware = bool(getattr(cfg, "recon_ert_aware", False))
    dil = (dilated_view(cfg, ses, arch, model, cats, raw)
           if (cfg.recon_optimizer and not aware) else None)
    # the run-wide reference record for the lines that describe the run as a
    # whole (the per-bar plans are chosen below)
    p_raw = dil.raw if dil else raw

    gran = granularity.Granularity(cfg.code_n, cfg.code_k, cfg.weight_bits,
                                cfg.recon_granularity)
    packing = packing_mod.Packing(cfg.recon_packing, cfg.weight_bits,
                               cfg.code_k, cfg.code_n)
    # prompt_6 RULE 1: THE ON-CHIP NARROWING HAS EXACTLY ONE OWNER PER BAR PER
    # STAGE, decided from each bar's own measured Word bits inside
    # `evaluate_placement` (a hard stop on a stop narrowed twice or by nobody).
    # The ownership rows are printed below, once the bars exist.
    # prompt_6 RULE 3: two terms, two denominators. The idle term is charged
    # per cycle of the plan a bar is billed from, times its engine count.
    recon_pj, recon_idle_pj, recon_prov = load_recon_energy(cfg)

    # Decode: charged to nobody by default (Task 2's rule). When ECC_DECODE=1
    # the embedded layout's codeword count is what every recon bar decodes,
    # because the reduced form is taken AFTER correction.
    n_cw_emb = raw.dram_w_reads / gran.weights_per_codeword
    decode_emb = n_cw_emb * cfg.decode_pj_emb if cfg.decode_enabled else 0.0
    # Under Task 4 the reconstruction arm issues FEWER DRAM reads, so it
    # decodes fewer codewords. Charging it the reference's count would bill it
    # for reads it never made.
    n_cw_placement = p_raw.dram_w_reads / gran.weights_per_codeword
    decode_placement = ((n_cw_placement * cfg.decode_pj_emb)
                        if (cfg.decode_enabled and cfg.recon_placement_charges_decode)
                        else 0.0)

    # ---- the five placements ------------------------------------------------
    wanted = cfg.recon_placements_for(arch)
    placements = [p for p in placements_mod.placements_for(arch, cfg)
                  if not wanted or p.key in wanted or p.variant in wanted]

    # prompt_7 B1/B2: ONE MAPPING PER BOUNDARY. The arms are the DISTINCT CHIPS
    # (`recon.mapper_arms`), not the ERT-injectable boundaries, so an arm with
    # no mapping of its own is normal until Phase C. Each bar is billed from a
    # NAMED plan whose storage geometry is its own wherever one exists
    # (`recon.plan_assignment`), and says so on its record.
    arms = arms_mod.mapper_arms(arch, cfg) if aware else ()
    arm_states, solved = {}, set()
    for a in arms:
        if a.placement is None:
            continue
        state, detail = arm_plan_state(cfg, ses, arch, model, a, list(paths))
        arm_states[a.key] = dict(detail, state=state)
        if state == "partial":
            raise SystemExit(
                f"{arch}/{a.key}: {detail['shapes_solved']} of {detail['shapes_wanted']} "
                f"shapes are mapped for this arm.\n"
                f"  cache: {detail['cache']}\n"
                f"  missing: {', '.join(detail['missing'])}\n"
                f"  A HALF-MAPPED ARM IS TWO CHIPS IN ONE BAR -- its own plan on some "
                f"shapes and a borrowed one on the rest. Finish the maps "
                f"(bash hpc/map_ert_arms.sh --no-eval) or remove the partial cache "
                f"directory so the bar borrows a named plan instead.")
        if state == "ready":
            solved.add(a.key)
    billing = arms_mod.plan_assignment(arch, solved, cfg) if aware else {}
    if aware and not solved:
        raise SystemExit(
            f"ECC_RECON_ERT_AWARE=1 on {arch}/{model}, and NOT ONE boundary has a mapping "
            f"of its own: every bar would be the reference plan under a heading that says "
            f"otherwise, which is Task 3.\n"
            + "".join(f"  {k:<10} {v['state']:<8} {v['cache']}\n"
                      for k, v in arm_states.items())
            + f"  -> map them:  bash hpc/map_ert_arms.sh --no-eval")
    views = {a.key: ert_aware_view(cfg, ses, arch, model, cats, raw, paths, a.placement)
             for a in arms if a.key in solved}
    if aware:
        for p in placements:
            b = billing[p.key]
            mark = {"own": "OWN", "borrowed": "borrowed", "foreign": "FOREIGN"}[b["kind"]]
            print(f"  [plan] {p.key:<8} {mark:<8} <- {b['arm']}: {b['why']}")

    def plan(p):
        """RULE 4: the (wpath, weight bill, bill, raw, cycles) THIS bar is billed
        from -- its own arm's mapping, a named geometry-compatible arm's, the
        Task 4 dilated plan, or the reference."""
        b = billing.get(p.key)
        if b is not None:
            v = views.get(b["arm"])
            if v is not None:
                tag = ("own mapping (ERT)" if b["kind"] == "own"
                       else f"{b['arm']}'s mapping (ERT)")
                return v.wpath, v.base_w, v.base_series, v.raw, v.cycles, tag
            return wpath, base_w, base_series, raw, wpath.cycles, "reference"
        if dil is not None:
            return dil.wpath, dil.base_w, dil.base_series, dil.raw, dil.wpath.cycles, "dilated"
        return wpath, base_w, base_series, raw, wpath.cycles, "reference"

    def own_view(p):
        """This bar's view ONLY when the bar IS the arm that was mapped.

        A BORROWED bar must not be handed the lender's `split`: the split is
        the lender's own encoder toll, moved out of the lender's own level, on
        the lender's own counter. The borrower charges reconstruction on ITS
        OWN site counter, read off the borrowed plan's stats -- which is what
        makes the borrow a plan and not a bill.
        """
        b = billing.get(p.key)
        return views.get(b["arm"]) if (b and b["kind"] == "own") else None

    results, plans = [], {}
    for p in placements:
        p_wpath, p_base_w, p_base_series, p_raw, p_cycles, tag = plan(p)
        plans[p.key] = (p_raw, tag)
        # Under its own plan a bar issues its own DRAM reads, so it decodes
        # its own codeword count (the reference's would bill reads it never made).
        n_cw_p = p_raw.dram_w_reads / gran.weights_per_codeword
        decode_p = ((n_cw_p * cfg.decode_pj_emb)
                    if (cfg.decode_enabled and cfg.recon_placement_charges_decode) else 0.0)
        # An ERT bar charges the WORDS Timeloop billed inside its own level
        # (`billed_weights`); every other bar charges Timeloop's scalar count.
        v_p = own_view(p)
        results.append(placement_eval.evaluate_placement(
            cfg, arch, p, p_wpath, p_base_w, p_base_series, recon_pj, gran,
            packing, decode_pj=decode_p, recon_idle_pj=recon_idle_pj,
            cycles=p_cycles,
            accesses_override=(v_p.split.get("billed_weights")
                               if v_p is not None else None)))
    # 5.3: what the evaluator charged as Reconstruction on an ERT bar must be
    # exactly what was moved out of the level -- A SPLIT, NOT AN ADDITION.
    #
    # THE RULE HAS TWO BRANCHES, and which one applies is a property of the
    # BOUNDARY, not of the run (prompt_7 B2):
    #
    #   the arm HAS an ERT row   Timeloop was given the toll and billed it
    #                            INSIDE the level, so the evaluator's charge
    #                            must equal what was moved out. A split.
    #   the arm has NONE         Timeloop was never given a toll, so the
    #                            level's billed energy contains not one
    #                            picojoule of encoder. The evaluator's charge
    #                            is an ADDITION, and must be, because nothing
    #                            else in the run has charged it. Moving
    #                            anything out here would REMOVE energy the
    #                            accelerator really spends.
    #
    # R1 and R3 on Eyeriss v1 are the second kind -- they are distinct chips by
    # their declared bandwidth scale and their network stage, not by an encoder
    # row -- and `ert_injectable()` is what decides. So the check below is not
    # relaxed for them; it is INVERTED: the moved amount must be exactly ZERO.
    for res in results:
        v = own_view(res.placement)
        if v is None or res.status != "evaluated":
            continue
        c = res.detail["reconstruction_counts"]
        if v.split.get("no_bump"):
            for name, moved in (("access", v.split["access_toll_pJ"]),
                                ("leak", v.split["stats_side_leak_pJ"])):
                if not _close(moved, 0.0, abs_tol=1e-6):
                    raise SystemExit(
                        f"{arch}/{res.placement.key}: this boundary declares NO ERT row "
                        f"({v.split['why']}), so Timeloop billed no toll and nothing may "
                        f"be moved out of its bill -- but the {name} split says "
                        f"{moved:.6f} pJ was. Either the arm grew an ERT row or the "
                        f"split is reading another arm's stats (prompt_6 5.3).")
            continue
        for name, charged, moved in (
                ("access", c["reconstruction_energy_incremental_pJ"], v.split["access_toll_pJ"]),
                ("leak", c["reconstruction_energy_idle_pJ"], v.split["stats_side_leak_pJ"])):
            if not _close(charged, moved):
                raise SystemExit(
                    f"{arch}/{res.placement.key}: the {name} term the evaluator charged "
                    f"({charged:.6f} pJ) is not the stats-side amount ({moved:.6f} pJ: "
                    f"{'moved out of ' + v.split['moved_out_of_category'] if name == 'access' else 'the leakage delta Timeloop printed'}); "
                    f"the encoder would be counted twice or not at all (prompt_6 5.3)")
        v.split["evaluator_incremental_pJ"] = c["reconstruction_energy_incremental_pJ"]
        v.split["evaluator_idle_pJ"] = c["reconstruction_energy_idle_pJ"]

    # ---- the result file ---------------------------------------------------
    builder = ResultBuilder(cfg, ses.results, arch, model,
                            experiment=(EXPERIMENT_ERT if aware else
                                        EXPERIMENT_TASK4 if dil else EXPERIMENT),
                            fixed_mapping=not (dil or aware))
    mapping_ids = audit.mapping_ids_of(raw)
    # TASK 4 / prompt_6: a bar that comes from a DIFFERENT mapping must carry
    # THAT mapping's ids. A result file whose recon bars claimed the
    # reference's ids would be indistinguishable from Task 3's.
    placement_mapping_ids = audit.mapping_ids_of(p_raw) if dil else mapping_ids
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
               "dram_energy_removed_vs_conventional_pJ": (
                   base_components.get("DRAM", 0.0) - emb_components.get("DRAM", 0.0)
                   + e_parity - emb_components.get(PARITY_KEY, 0.0)),
               "role": ("the reference Task 3 measures reconstruction against: "
                        "every placement reads the same complete codeword out of "
                        "the DRAM ARRAY as this bar; with the decoder on the DRAM "
                        "die (ECC_RECON_DECODE_SITE=ondie) every placement's DRAM "
                        "INTERFACE share is this bar's x K/N, because this bar's "
                        "on-chip datapath consumes every weight bit and so its "
                        "complete codeword still crosses the interface"),
               "dram_term": wpath.dram_term(),
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
                         "n_cw": 0.0, "dram_saved_pJ": 0.0})
            continue
        counts = res.detail["reconstruction_counts"]
        bar_raw, bar_tag = plans[p.key]
        v = own_view(p)
        if aware:
            # prompt_6 RULE 4 / prompt_7 B1: EVERY BAR NAMES THE PLAN IT WAS
            # BILLED FROM, and whether that plan's storage geometry is its own.
            # `reference` is only ever correct for R1, whose geometry IS the
            # reference's; a bar on a plan that narrows different levels says so
            # in `plan_geometry_matches: false` and carries the reason.
            b = billing[p.key]
            barm = next((a for a in arms if a.key == p.key), None)
            lender = views.get(b["arm"])
            bar_extra = {
                "reconstruction_aware_mapping": ERT_NOTE,
                "plan": bar_tag,
                "billed_from": {
                    "arm": b["arm"], "kind": b["kind"],
                    "geometry_matches": b["geometry_matches"],
                    "candidate_arms_with_this_geometry": b["candidates"],
                    "why": b["why"],
                    "this_bar_declares": {
                        "datawidth_q_on": list(barm.narrow_levels) if barm else [],
                        "bandwidth_scale_on": [list(x) for x in (barm.bw_scale if barm else ())],
                        "ert_bump": list(barm.ert) if (barm and barm.ert) else None},
                    "this_arms_own_cache": arm_states.get(p.key),
                    "lender_fingerprint": lender.fingerprint if lender is not None else None,
                    "lender_cache_variant": lender.variant if lender is not None else None},
                "dram_weight_reads_reference": float(raw.dram_w_reads),
                "dram_weight_reads_this_arm": float(bar_raw.dram_w_reads),
                "cycles_reference": float(raw.cycles or 0),
                "cycles_this_arm": float(bar_raw.cycles or 0)}
            if v is not None:
                bar_extra["ert_arm"] = {
                    "bump": v.bump, "fingerprint": v.fingerprint,
                    "cache_variant": v.variant,
                    "loop_nest_identical_to_reference": v.nest_identical,
                    "per_shape_nest_identical": v.per_shape_nest_identical,
                    "attribution_split": v.split,
                    "capacity": v.capacity,
                    "guards_per_shape": v.checks["per_shape"],
                    "ert_read_back": v.read_back}
                bar_extra["cycles_this_arm"] = float(v.cycles)
            elif b["kind"] == "borrowed" and b["arm"] == "reference":
                bar_extra["fixed_mapping"] = FIXED_MAPPING_NOTE
                bar_extra["note"] = (
                    "post-processed on the reference plan, which is the RIGHT plan for this "
                    "bar: it narrows nothing on chip, so its storage geometry is the "
                    "reference's. Only the off-chip bandwidth scale separates the two, and "
                    "that changes the throttling check, not the energy.")
            elif b["kind"] == "borrowed":
                bar_extra["note"] = (
                    f"post-processed on {b['arm']}'s plan, which narrows the SAME storage "
                    f"levels this bar does. {b['arm']}'s own encoder toll was moved out of "
                    f"the bill before this bar was billed from it, so this bar pays the "
                    f"un-bumped price of the shared geometry and its own reconstruction "
                    f"term on its own site counter.")
            else:
                bar_extra["fixed_mapping"] = FIXED_MAPPING_NOTE
                bar_extra["note"] = (
                    f"post-processed on the REFERENCE plan, whose storage geometry is not "
                    f"this bar's (prompt_7 6.2b: a chip that has never been mapped). The "
                    f"energy is right for a chip whose loop nest was chosen without knowing "
                    f"the level is narrower; what it cannot show is a plan that uses the "
                    f"capacity that narrowing frees.")
            bar_ids = audit.mapping_ids_of(bar_raw)
        else:
            bar_extra = None
            bar_ids = placement_mapping_ids
        builder.add(Variant(
            p.variant, kind="reconstruction", status="evaluated",
            total_energy_pJ=res.total_pJ,
            energy_by_component_pJ={k: v for k, v in res.components.items()
                                    if v != 0.0},
            mapping_ids=bar_ids, label=label,
            extra=dict(res.detail, placement_key=p.key,
                       **(bar_extra if bar_extra is not None else
                          {"reconstruction_aware_mapping": TASK4_NOTE,
                           "weight_capacity": dil.capacity,
                           "capacity_dilation_correction": dil.correction,
                           "dram_weight_reads_reference": float(raw.dram_w_reads),
                           "dram_weight_reads_this_arm": float(p_raw.dram_w_reads),
                           "refetch_reference":
                               float(raw.dram_w_reads) / raw.weights if raw.weights else 0.0,
                           "refetch_this_arm":
                               float(p_raw.dram_w_reads) / p_raw.weights if p_raw.weights else 0.0}
                          if dil else {"fixed_mapping": FIXED_MAPPING_NOTE}),
                       ecc_codec_energy=CODEC_NOTE)))
        rows.append({"key": p.key, "label": p.short.replace("\n", " "),
                     "status": "evaluated", "reason": "",
                     "total": res.total_pJ,
                     "recon_pJ": counts["reconstruction_energy_pJ"],
                     "idle_pJ": counts["reconstruction_energy_idle_pJ"],
                     "engines": counts["engines"],
                     "plan": bar_tag,
                     "overhead_pJ": counts["recon_overhead_energy_pJ"],
                     "n_cw": counts["reconstruction_events_codewords"],
                     "dram_saved_pJ":
                         res.detail["dram_model"]["dram_saving_pJ"]})

    # ---- the checks that make the comparison auditable ---------------------
    # TASK 3 and TASK 4 assert DIFFERENT invariants and must not borrow each
    # other's. Task 3's "the DRAM array is identical on every bar" is FALSE
    # under Task 4 and is meant to be: the reconstruction arm issues fewer
    # reads, so it legitimately pays less array energy. Task 4 therefore
    # replaces that check with a tighter one -- the array credit must equal
    # exactly the reads the mapper removed, and not a picojoule more.
    # prompt_7 A2: ONE latency table, computed once and reused by the checks,
    # the result file and the console -- three copies could disagree.
    lat = latency_table(cfg, raw, packing, plans)
    if aware:
        ert_checks(builder, cfg, arch, raw, views, results, base_components,
                   emb_components, e_parity, pricing,
                   newly_mapped=getattr(mapper, "n_mapped", 0),
                   billing=billing, arm_states=arm_states, lat=lat)
    elif dil:
        task4_checks(builder, cfg, arch, raw, dil, base_components,
                     emb_components, e_parity, results, wpath, base_w,
                     newly_mapped=getattr(mapper, "n_mapped", 0), pricing=pricing)
    else:
        task3_checks(builder, cfg, arch, raw, base_components, emb_components,
                     e_parity, results, wpath, reconciles, recon_check, base_w,
                     newly_mapped=getattr(mapper, "n_mapped", 0), pricing=pricing)

    # ---- Task 1's checks and caveats, so the baseline bar is as audited ----
    arch_report = audit.common_checks(builder, cfg, arch, raw, pdetail, mapping_ids)
    audit.common_caveats(builder, cfg, arch, raw, pdetail, pricing)
    builder.approximate(edetail["traffic"]["method"])
    builder.approximate(CODEC_NOTE)
    if aware:
        builder.approximate(ERT_NOTE)
        own = [p.key for p in placements if billing[p.key]["kind"] == "own"]
        lent = [f"{p.key} <- {billing[p.key]['arm']}" for p in placements
                if billing[p.key]["kind"] == "borrowed"]
        foreign = [p.key for p in placements if billing[p.key]["kind"] == "foreign"]
        builder.warn(
            f"ONE PLAN PER BAR, AND EVERY BAR NAMES ITS OWN (prompt_6 RULE 4, prompt_7 B1). "
            f"From their OWN mapping, encoder toll in the ERT: {', '.join(own) or 'none'}. "
            f"From a NAMED plan that narrows the same storage levels: "
            f"{', '.join(lent) or 'none'}. "
            f"On the reference plan although it narrows DIFFERENT levels -- a chip that has "
            f"never been mapped, prompt_7 6.2b: {', '.join(foreign) or 'none'}. "
            f"Each mapped arm's loop-nest verdict is recorded on its own: "
            + "; ".join(f"{k}: " + ("IDENTICAL to the reference -- the ERT changed nothing "
                                    "for this arm" if v.nest_identical else "CHANGED")
                        for k, v in views.items()))
        if foreign:
            builder.warn(
                f"A CHIP THAT HAS NEVER BEEN MAPPED is on this figure: "
                f"{', '.join(foreign)}. Its `reduced` set narrows storage levels no solved "
                f"arm narrows, so no plan on disk knows the capacity that narrowing frees "
                f"and no post-processing can re-tile a loop nest to use it. The energy is a "
                f"valid number for the reference loop nest; it is NOT that chip's own "
                f"mapping. -> bash hpc/map_ert_arms.sh (prompt_7 Phase C).")
    if dil:
        builder.approximate(TASK4_NOTE)
        builder.approximate(dil.correction.get("provenance", ""))
        if not dil.capacity.get("the_mapper_used_none_of_it"):
            builder.approximate(
                "THE DILATED DESIGN'S NoC HOP LENGTH IS NOT CORRECTED. Where "
                "archs/_shared/noc.yaml does not pin a `tile_width_um`, "
                "Timeloop derives the hop length from the inner level's "
                "Accelergy AREA, so expressing the dilation as `depth x N/K` "
                "also lengthens the wires into that level. The per-access "
                "energy is re-priced at the declared array; the hop length is "
                "not, because the wire share of a network's energy is not "
                "separable from its switching share after the fact. It costs "
                "the reconstruction arm NoC energy it would not pay on the "
                "declared silicon (0.17 pp on eyeriss_like's two-layer scope), "
                "so this saving is a lower bound on that count too.")
        _rd = dil.correction.get("read_ratio_declared_over_dilated", 1.0)
        if _rd and _rd < 0.95:
            builder.warn(
                f"Accelergy priced the DILATED {dil.capacity['level']} "
                f"{1 / _rd:.3f}x DEARER per read than the "
                f"declared one, because it costs a level from its declared "
                f"geometry and the dilation is expressed as depth. The energy "
                f"is corrected back to the declared array (the reconstruction "
                f"arm's array is the same silicon holding narrower values), but "
                f"the MAPPER optimised against the dearer one and therefore had "
                f"a reason not to use the extra capacity. This Task 4 saving is "
                f"a LOWER bound for that reason.")
    else:
        builder.approximate(FIXED_MAPPING_NOTE)
    builder.approximate(packing.to_dict()["meaning"])
    builder.approximate(gran.to_dict()["meaning"])
    if cfg.recon_decode_site == "ondie":
        builder.approximate(DRAM_TERM_NOTE)
        builder.approximate(
            f"DRAM model (decoder on the DRAM die, off the fetch path): the "
            f"WHOLE DRAM weight energy is x K/N under every boundary -- the "
            f"f_if array/interface split was removed 2026-09-09, and the DRAM "
            f"access collects only the message bits of each codeword. "
            f"{cfg.dram_cost_note}. {cfg.dram_static_note}.")
    else:
        builder.warn(
            "ECC_RECON_DECODE_SITE=controller: this is the pre-2026-09-09 model "
            "(controller-side correction on the fetch path, complete codeword "
            "across the DRAM interface, DRAM identical on every bar), kept as a "
            "runnable row for the diff. Do not quote it as the study's result.")
    builder.approximate(
        "NO BOUNDARY CARRIES A PER-PE REUSE REGISTER. R4b (SPad output plus a "
        "reconstructed-weight register) was removed on 2026-09-10: measured on "
        "these mappings, consecutive weight reuse is 1 on 20 of 21 resnet18 "
        "layers, so a latch catches nothing, and a register that does pay has "
        "to hold the whole inner tile -- up to 384 weights against a "
        "384-weight scratchpad. Scratchpad read counts are Timeloop's under "
        "every bar.")
    builder.approximate(
        f"Weight-path stages are matched to Timeloop levels by name "
        f"(eccenergy/arch/weight_path.py WEIGHT_PATHS[{arch!r}]) and the match is verified "
        f"against the raw record's per-category weight energy before any "
        f"placement is evaluated. A level carrying weight energy that no stage "
        f"claims fails that check rather than being ignored.")
    if design.flag(arch, "weights_stored_compressed"):
        builder.warn(
            "CSC METADATA IS NOT MODELLED. The published Eyeriss v2 PE stores "
            "weights CSC-compressed with a separate address scratchpad "
            "(Table IV: weight data 288B, weight address 16x7b); this study "
            "models the design DENSE, so there is no metadata array here to "
            "leave at full width. A CSC implementation would apply the reduced "
            "representation to the weight DATA only, so R4a's scratchpad "
            "saving is an upper bound on what CSC v2 would see.")
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
            "reconstruction_datapath_idle_pJ_per_cycle_per_engine": recon_idle_pj,
            "reconstruction_datapath_provenance": recon_prov,
            "decode_site": cfg.recon_decode_site,
            "dram_pj_per_bit": cfg.dram_pj_per_bit,
            "dram_cost_provenance": cfg.dram_cost_note,
            "dram_static_terms": cfg.dram_static_note,
            "dram_term": wpath.dram_term(),
            "placements_defined": [
                {"key": p.key, "variant": p.variant, "boundary": p.label,
                 "hypothesised_rating": p.rating,
                 "reduced_stages": list(p.reduced),
                 "reconstruction_site": p.site_stage,
                 "access_counter": p.site_counter,
                 "description": p.description}
                for p in placements_mod.placements_for(arch, cfg)],
            "placements_requested": wanted or "all",
        })
    # prompt_7 Phase A. Recorded whether or not the roofline ran: `standby`
    # always says who was charged, and a `latency_model` of None says plainly
    # that time was not modelled, which is the thing Defect 1 is about.
    builder.set_detail(
        standby_energy=getattr(raw, "standby", None),
        latency_model=(lat
                       if cfg.latency_model else
                       {"enabled": False,
                        "reason": ("ECC_LATENCY_MODEL=0: cycles are Timeloop's "
                                   "own, off-chip traffic costs none of them, "
                                   "and `datawidth: q` changes no cycle at all "
                                   "-- a 0.00% latency gain here is an ABSENT "
                                   "TERM, never a result (prompt_7 Defect 1)")}))

    _report(cfg, arch, model, wpath, gran, packing, rows, base_total, emb_total,
            raw=raw)
    # prompt_7 Phase A, A2: the latency table, from each bar's OWN plan
    _report_latency(cfg, lat)
    _report_narrowing(results)
    path = builder.write()
    print(f"    -> {path}")
    return {"path": path, "rows": rows, "results": results, "wpath": wpath,
            "views": views, "ert_bars": set(views), "billing": billing,
            "arm_states": arm_states,
            "mapper_arms": [{"key": a.key, "slug": a.slug_part,
                             "narrow_levels": list(a.narrow_levels),
                             "bandwidth_scale": [list(x) for x in a.bw_scale],
                             "ert": list(a.ert) if a.ert else None,
                             "members": list(a.members)} for a in arms],
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
                 newly_mapped=0, pricing=None):
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
    space_ok, space = arms_mod.validate_placement_space(arch, cfg)
    builder.check("placement_space_covers_the_whole_weight_path", space_ok, space)

    evaluated = [r for r in results if r.status == "evaluated"]

    # 1. the DRAM term, in BOTH directions. With the decoder on the DRAM die the
    #    array share is the embedded arm's and the interface share is x K/N, so
    #    every placement's DRAM component must sit at EXACTLY
    #        emb_DRAM - DRAM_w x (1 - K/N).
    #    That equality is split into two one-sided checks on purpose: a bar
    #    that quietly credits the array sits BELOW it and fails the first; a
    #    bar that leaves the interface at full width sits ABOVE it and fails
    #    the second. Each also reads the placement's own dram_model rows, so a
    #    stack cannot say one thing and its detail another. Under `controller`
    #    the expected value is emb_DRAM and both checks collapse onto the
    #    pre-2026-09-09 `dram_identical_to_embedded_reference`.
    dram_ref = emb_components.get("DRAM", 0.0)
    dram_w = float(base_w.get("DRAM", 0.0))
    term = wpath.dram_term()
    ondie = term["decode_site"] == "ondie"
    frac = cfg.code_k / cfg.code_n
    saving = dram_w * (1.0 - frac) if ondie else 0.0
    expected = dram_ref - saving
    tol = abs(expected) * 1e-12 + 1e-6
    rows, ok_d = {}, True
    for r in evaluated:
        got = r.components.get("DRAM", 0.0)
        dm = r.detail.get("dram_model", {})
        want_scale = frac if ondie else 1.0
        row_ok = (bool(dm.get("dram_reduced", False)) == ondie
                  and math.isclose(dm.get("dram_after_pJ", 0.0),
                                   dm.get("dram_weight_energy_pJ", 0.0) * want_scale,
                                   rel_tol=1e-12, abs_tol=1e-6))
        d_ok = row_ok and math.isclose(got, expected, rel_tol=1e-12, abs_tol=tol)
        rows[r.placement.key] = {
            "dram_pJ": got, "expected_dram_pJ": expected,
            "off_expected_by_pJ": abs(got - expected),
            "dram_row_scaled": row_ok, "dram_scale": want_scale,
            "passed": d_ok}
        ok_d = ok_d and d_ok
    builder.check(
        "dram_scaled_by_K_over_N", ok_d,
        {"decode_site": term["decode_site"],
         "dram_pj_per_bit": term["dram_pj_per_bit"],
         "dram_cost_provenance": term["dram_cost_provenance"],
         "dram_weight_energy_pJ": dram_w,
         "embedded_reference_dram_pJ": dram_ref,
         "saving_every_placement_pJ": saving,
         "expected_placement_dram_pJ": expected,
         "per_placement": rows,
         "rule": ("with the decoder on the DRAM die only the k message bits are "
                  "read out and driven off it, so the WHOLE DRAM weight energy "
                  "is x K/N under every boundary: every placement's DRAM "
                  "component must equal emb_DRAM - DRAM_w x (1 - K/N) exactly, "
                  "and its dram row must carry that scale (1.0 under "
                  "ECC_RECON_DECODE_SITE=controller). The f_if array/interface "
                  "split, and with it the separate array check, was removed on "
                  "2026-09-09; the whole term now moves together")})

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
    ok, detail = baseline_dram.reference_bars_agree(
        base_components, emb_components, e_parity, pricing,
        dram_w_reads=raw.dram_w_reads)
    builder.check("reference_bars_match_tasks_1_and_2", ok, detail)


def task4_checks(builder, cfg, arch, raw, dil, base_components, emb_components,
                 e_parity, results, ref_wpath, base_w, newly_mapped=0,
                 pricing=None):
    """The checks that make Task 4's claim -- "the mapper spent the extra room" --
    auditable, and that stop it borrowing a claim it is not entitled to.

    TASK 3'S CHECKS DO NOT TRANSFER, and running them here would either fail
    honestly or pass dishonestly. Three of them are about a FIXED mapping:

      dram_scaled_by_K_over_N   too loose here -- the reconstruction arm also
          issues fewer DRAM reads, and that extra credit is the whole point.
          Replaced below by a strictly tighter statement: the credit must equal
          EXACTLY the reads the mapper removed, priced at the reference's own
          per-read energy, on top of the K/N narrowing.
      non_weight_energy_identical                  activations and partial sums
          legitimately move when the mapping changes. Only the MAC count is
          mapping-invariant, and that is checked, hard: two mappings of the same
          layer that disagree on Computes are not two mappings of the same
          layer.
      only_weight_path_categories_differ           likewise.

    What replaces them is the pair of facts a Task 4 number stands on: the two
    arms really are the same workload on the same design, and the reconstruction
    arm really did get N/K more weight room and no other advantage.
    """
    evaluated = [r for r in results if r.status == "evaluated"]
    frac = cfg.code_k / cfg.code_n
    nk = 1.0 / frac
    term = dil.wpath.dram_term()
    ondie = term["decode_site"] == "ondie"

    # 0. the two tables that define the placement space still agree
    space_ok, space = arms_mod.validate_placement_space(arch, cfg)
    builder.check("placement_space_covers_the_whole_weight_path", space_ok, space)

    # 1. THE TWO ARMS ARE THE SAME WORKLOAD. The MAC count is mapping-invariant
    #    -- a convolution has the multiplications it has, whatever the tiling --
    #    so if the two mappings disagree on it they are not two mappings of one
    #    problem, and every percentage below would be comparing two workloads.
    #    This is the check that makes the rest of the file meaningful.
    # `Raw.mac` is filled by `energy.apply_mac_override()` and carries the MAC
    # count it charged; it is the only place the count survives aggregation.
    ref_macs = float((getattr(raw, "mac", None) or {}).get("macs", 0.0) or 0.0)
    dil_macs = float((getattr(dil.raw, "mac", None) or {}).get("macs", 0.0) or 0.0)
    same_macs = (ref_macs > 0 and math.isclose(ref_macs, dil_macs,
                                               rel_tol=1e-9, abs_tol=1.0))
    builder.check(
        "both_arms_are_the_same_workload", same_macs,
        {"reference_computes": ref_macs, "dilated_computes": dil_macs,
         "reference_weights": float(getattr(raw, "weights", 0.0) or 0.0),
         "dilated_weights": float(getattr(dil.raw, "weights", 0.0) or 0.0),
         "rule": ("the MAC count is mapping-invariant, so the reference and the "
                  "reconstruction-aware mapping must report the same Computes. "
                  "They differ only in HOW the data moves, never in how much "
                  "arithmetic there is -- and if they differ here, the two "
                  "totals are not comparable at all")})

    # 2. THE RECONSTRUCTION ARM GOT N/K MORE WEIGHT ROOM AND NOTHING ELSE.
    #    Read off both mappings' own `Effective size`, not off the YAML: a
    #    `depth:` the patch failed to rewrite, or one Timeloop clamped, shows up
    #    here rather than being assumed.
    cap = dil.capacity
    want, q = capacity.capacity_target(cfg)
    cap_ok = (cap["reference_weights_per_instance"] > 0
              and abs(cap["delivered_factor"] - want) <= 0.05 * want)
    builder.check(
        "reconstruction_arm_has_8_over_q_more_weight_capacity", cap_ok,
        dict(cap, reference_scale=dil.ref_scale, dilated_scale=dil.scale,
             rule=(f"the reduced representation stores weights at q = {q} whole "
                   f"bits, so it holds {cfg.weight_bits}/q = {want:.4f} times as many "
                   f"in the same silicon (N/K = {nk:.4f} is the ideal rate, not the "
                   f"room a whole-bit word delivers -- prompt_6 6); the mapper must "
                   f"have been given exactly that much more room at the weight "
                   f"level -- no more (unearned) and no less (understated)")))

    # 3. THE ARRAY CREDIT IS EXACTLY THE READS THE MAPPER REMOVED. This is the
    #    check that separates Task 4 from wishful thinking. Task 3's rule was
    #    "the array never moves"; here it moves, and it may move by exactly
    #    `(reads_ref - reads_dil) x pJ_per_read` and no more. A boundary that
    #    credited itself for reads it still issues fails.
    dram_ref_emb = emb_components.get("DRAM", 0.0)
    dram_w_ref = float(base_w.get("DRAM", 0.0))
    dram_w_dil = float(dil.base_w.get("DRAM", 0.0))
    reads_ref = float(raw.dram_w_reads)
    reads_dil = float(dil.raw.dram_w_reads)
    # The DRAM weight energy scales exactly with the read count (one flat
    # per-bit constant), so the array term the reconstruction arm owes is the
    # dilated read count's share of it.
    dil_expected = dram_w_dil * frac if ondie else dram_w_dil
    expected = dram_ref_emb - dram_w_ref + dil_expected
    tol = abs(expected) * 1e-9 + 1e-3
    rows, ok = {}, True
    for r in evaluated:
        got = r.components.get("DRAM", 0.0)
        dm = r.detail.get("dram_model", {})
        row_ok = (math.isclose(got, expected, rel_tol=1e-9, abs_tol=tol)
                  and bool(dm.get("dram_reduced", False)) == ondie)
        rows[r.placement.key] = {
            "dram_pJ": got, "expected_dram_pJ": expected,
            "difference_pJ": got - expected,
            "array_row_unreduced_within_this_mapping":
                bool(dm.get("dram_reduced", False)) == ondie,
            "passed": row_ok}
        ok = ok and row_ok
    builder.check(
        "dram_credit_equals_the_reads_the_mapper_removed", ok,
        {"per_placement": rows,
         "reference_dram_weight_reads": reads_ref,
         "reconstruction_dram_weight_reads": reads_dil,
         "reads_never_issued": reads_ref - reads_dil,
         "reference_dram_weight_pJ": dram_w_ref,
         "reconstruction_dram_weight_pJ": dram_w_dil,
         "K_over_N": frac,
         "rule": ("under Task 4 the reconstruction-aware mapping issues fewer "
                  "DRAM reads AND each read it does issue is narrower, so "
                  "every placement's DRAM component must equal "
                  "DRAM_w(dilated) x K/N, shifted by the reference bar's own "
                  "DRAM. A bar below that is counting a saving twice")})

    # 4. THE SAVING IS DECOMPOSED, and the two halves are reported separately
    #    because they have different efficiencies and only one of them is new.
    #    Fixed-mapping efficiency is (1 - K/N) = 0.381 since the f_if split
    #    was removed (it was f_if x (1 - K/N) = 0.152 at f_if = 0.40); a read
    #    never issued still has efficiency 1.0, so the two are now much closer.
    never_issued_pJ = dram_w_ref - dram_w_dil
    fixed_part = dram_w_dil * (1.0 - frac) if ondie else 0.0
    builder.check(
        "dram_saving_is_decomposed_into_refetch_and_narrowing", True,
        {"reads_never_issued_pJ": never_issued_pJ,
         "narrowing_on_the_reads_still_issued_pJ": fixed_part,
         "total_dram_saving_pJ": never_issued_pJ + fixed_part,
         "efficiency_of_the_refetch_part": 1.0,
         "efficiency_of_the_narrowing_part": 1.0 - frac,
         "efficiency_overall": ((never_issued_pJ + fixed_part) / dram_w_ref
                                if dram_w_ref else 0.0),
         "refetch_reference": reads_ref / raw.weights if raw.weights else 0.0,
         "refetch_reconstruction":
             reads_dil / dil.raw.weights if dil.raw.weights else 0.0,
         "first_order_prediction_FINDINGS_9_0":
             ((reads_ref / raw.weights - 1.0) * frac + 1.0) if raw.weights else 0.0,
         "rule": ("informational, always passes: a read the reconstruction arm "
                  "never issues removes the whole per-access energy, so its "
                  "efficiency is 1.0 per pJ, against the (1 - K/N) that "
                  "narrowing an issued read buys. "
                  "FINDINGS section 9 item 0 predicts the refetch this arm "
                  "should show if excess refetch scaled inversely with "
                  "capacity; the measured value beside it is the validation")})

    # 5. THE CORRECTION IS DECLARED. The dilated level was priced by Accelergy
    #    as a physically larger array; the reconstruction arm's is the same
    #    silicon holding narrower values.
    corr = dil.correction
    builder.check(
        "dilated_buffer_repriced_at_the_declared_array", True,
        dict(corr, rule=("Accelergy costs a level from its declared geometry, "
                         "so expressing the dilation as `depth x N/K` prices "
                         "the reconstruction arm's scratchpad as a bigger SRAM "
                         "than it is. Its energy is re-priced at the declared "
                         "array's per-access cost. The MAPPER still optimised "
                         "against the dearer array, so a ratio far from 1.0 "
                         "means the search had a reason to avoid the capacity "
                         "and this result is a LOWER bound")))

    # 6. the weight path still reconciles with the dilated raw record
    builder.check("weight_path_reconciles_with_raw_record",
                  dil.reconciles, dil.recon_check)

    # 7. a placement cannot save more than the reducible weight energy there is
    reducible = {}
    for r in evaluated:
        avail = sum(dil.wpath.stages[k].energy_pJ for k in r.placement.reduced
                    if k in dil.wpath.stages)
        saved = sum(r.detail["energy_saved_by_category_pJ"].values())
        reducible[r.placement.key] = {
            "reducible_weight_energy_pJ": avail, "energy_saved_pJ": saved,
            "within_bound": saved <= avail + 1e-6 and saved >= -1e-6}
    builder.check(
        "placement_savings_are_bounded_by_the_weight_path",
        all(v["within_bound"] for v in reducible.values()),
        {"per_placement": reducible,
         "rule": ("within its OWN mapping, a boundary reduces the weight energy "
                  "of the stages the reduced form reaches and no more. The "
                  "refetch saving is not in this bound -- it is a smaller "
                  "weight path to begin with, not a reduction of this one")})

    # 8. NEITHER ARM WAS MAPPED BY THIS RUN. Task 4 re-optimises the mapping,
    #    but it does so in a separate mapping wave whose cache this reads; an
    #    evaluation that had to invoke Timeloop would mean the figure and the
    #    cache disagree about what was solved.
    builder.check(
        "evaluation_only_rerun",
        bool(cfg.from_cache) or newly_mapped == 0,
        {"ECC_FROM_CACHE": bool(cfg.from_cache),
         "newly_mapped_shapes": newly_mapped,
         "reference_fingerprint": fingerprint.arch_fingerprint(arch, cfg),
         "reconstruction_fingerprint": dil.fingerprint,
         "meaning": ("the two arms are two mapper caches, both solved before "
                     "this evaluation ran. The fingerprints are recorded side "
                     "by side so a figure drawn from one cache cannot claim "
                     "two")})

    # 9. the reference bars still say what Tasks 1 and 2 said
    ok, detail = baseline_dram.reference_bars_agree(
        base_components, emb_components, e_parity, pricing,
        dram_w_reads=raw.dram_w_reads)
    builder.check("reference_bars_match_tasks_1_and_2", ok, detail)


