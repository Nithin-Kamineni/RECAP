"""Evaluating ONE placement: the bar, its checks, and what it may not claim.

`evaluate_placement()` takes a boundary, the design's weight path as read back
from the cache (`toolchain/weight_stats.py`), and the raw energy record, and
returns the bar: per-category energy, the reconstruction terms, the ceiling, and
the record of how every one of those was arrived at.

EVERY RESULT PRINTS THE CEILING FIRST -- the weight energy a boundary could
reduce, times `1 - K/N`. Without it a sub-percent saving reads as a missing term
rather than as arithmetic. `cross_check()` re-derives per-category weight energy
against the `Raw` record before any of it is believed.

WHAT IS DELIBERATELY NOT CREDITED
---------------------------------
* The WHOLE DRAM term IS credited, by exactly K/N: only the k message bits of
  each codeword are read out and driven off the die, so `dram` = the DRAM weight
  energy x K/N under every boundary (`dram_scaled_by_K_over_N` fails if a bar
  leaves it at full width). It is the same saving on every bar, so it moves every
  placement against the embedded reference by the same amount and changes no
  ordering among them.
* E_background and E_refresh are not modelled (both 0). An arm that stores fewer
  weight bits in DRAM would save both, so this understates the embedded and
  reconstruction arms alike.
* The DECODER is still outside the comparison. It is the same BCH decoder
  relocated to the DRAM die, runs once per corrected codeword off the fetch path,
  and is DRAM-process logic; it cancels between the embedded bar and every
  placement exactly as the controller-side codec did (`CODEC_NOTE`, in
  `study/placement_notes.py`).
* No capacity-driven reuse and no re-tiling: the mapping is fixed, so every
  access count is Timeloop's. Packed reduced weights would raise the effective
  SPad capacity -- that is Task 4's question, and crediting it here would mix the
  two answers.
* No operand-retention saving for anybody. Scratchpad read counts stay
  Timeloop's under every arm, so the K/N discount on them is real rather than
  double-counted. No boundary carries a per-PE reuse register (R4b, removed
  2026-09-10), so `Recon overhead` is structurally zero.

RECONSTRUCTION IS TWO TERMS ON TWO DENOMINATORS (prompt_6 RULE 3):
`incremental x events + idle_per_cycle x cycles x N_engines`, both CLOCK-GATED by
`ECC_RECON_CLOCK_GATING_PCT` since prompt_7. Nothing here adds them.

ProjectRestructure phase 3 cut this out of `recon.py`; `study/placement_study.py`
is the driver that calls it once per boundary and writes the result file.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..arch.placements import Placement, effective_placement, encoder_site, stages_for
from ..physics.granularity import engine_cycles_for, feasibility, narrowing_audit_for_bar


# ===========================================================================
#  evaluating one placement
# ===========================================================================
@dataclass
class PlacementResult:
    placement: Placement
    status: str                 # evaluated | unsupported
    components: dict            # plotted category -> pJ
    total_pJ: float
    detail: dict
    reason: str = ""


def evaluate_placement(cfg, arch, placement, wpath, base_w_by_cat, base_by_cat,
                       recon_pj, gran, packing, decode_pj=0.0, recon_idle_pj=0.0,
                       cycles=None, accesses_override=None):
    """Cost one boundary. Fixed mapping: every access count is Timeloop's.

    prompt_6 RULE 3: `recon_pj` is the INCREMENTAL term (pJ per codeword
    event) and `recon_idle_pj` the idle term (pJ per cycle per engine);
    `cycles` is the run length of the plan this bar is billed from (None =
    `wpath.cycles`). The encoder energy is

        E_recon = incremental x events + idle x cycles x N_engines

    with N_engines derived per placement by `engines_for` (1 / 14 / 168 on
    Eyeriss v1). An idle term without a cycle count is refused, never zero.

    `base_by_cat` / `base_w_by_cat` are the plotted-category totals and their
    weight share, from the SAME `Raw` record the baseline and embedded arms use.
    A stage's saving is subtracted from its own category, so the stack still
    sums to the total. The DRAM category moves by exactly its K/N reduction and
    no more: the single `dram` stage is in every `reduced` set (the decoder is
    on the die; `placement` is still passed through `effective_placement()`
    first, for the encoder-site knob).
    """
    placement = effective_placement(placement, cfg)
    stage_defs = stages_for(arch, cfg)
    # GROUP RESIDENCY IS REPORTED, NEVER REFUSED (decided 2026-09-13; the knob
    # ECC_RECON_REQUIRE_GROUP_RESIDENCY went on 2026-09-14). G_rec = 9 at
    # BCH(63,.) over 8-bit weights: 63/8 is not whole, so a codeword drifts
    # across weight boundaries and the worst-aligned one reaches into 9
    # weights. The old refusal assumed an engine that can only rebuild from
    # weights co-resident in the level AT ONE INSTANT; RECAP's accumulates the
    # retained bits as they arrive, so a level holding 6 -- or 1 -- still feeds
    # it, over more accesses and with more buffering: a COST, not an
    # impossibility. The shortfall is still measured and travels with the bar
    # (`layers_below_G_rec`, `infeasible_layers`, `group_residency_note`)
    # because it bounds the buffer the engine needs. Measured: reporting it
    # recovered R2/R3/R4/R5a on mobilenet_v2 (1, 1, 4 and 14 layers of 31
    # below G_rec) and changed resnet18 by nothing. `granularity.feasibility`
    # keeps the strict reading as a parameter, for the test that pins both.
    ok, feas = feasibility(placement, arch, wpath, gran,
                           require_group_residency=False)
    if not ok:
        return PlacementResult(placement, "unsupported", {}, 0.0,
                               {"feasibility": feas}, feas["reason"])
    # prompt_6 RULE 1: exactly one owner per narrow storage stop, decided per
    # bar from ITS plan's measured Word bits. A hard stop either way it fails.
    narrow_ok, narrowing = narrowing_audit_for_bar(placement, wpath, packing, stage_defs)
    if not narrow_ok:
        bad = [r for r in narrowing if not r["ok"]]
        raise ValueError(
            f"{arch}/{placement.key}: " + "; ".join(
                f"{r['stage']} ({'+'.join(r['levels'])}): {r['problem']}" for r in bad))

    # ---- 1. what the reduced representation saves, stage by stage -----------
    saved_by_cat = {}
    stage_rows = []
    for stage in stage_defs:
        st = wpath.stages.get(stage.key)
        if st is None or st.energy_pJ <= 0:
            continue
        cat = _category_of(stage, cfg)
        reduced = stage.key in placement.reduced
        if not reduced:
            stage_rows.append({**st.to_dict(), "category": cat,
                               "carries_reduced": False, "scale": 1.0,
                               "energy_saved_pJ": 0.0})
            continue
        if stage.kind == "dram":
            # The interface carries k of every n bits of the codeword STREAM
            # the die emits, so it scales by exactly K/N whatever the on-chip
            # packing does with the bits after ingress (`aligned` repacks on
            # chip; the die output is the embedded layout's packed stream).
            scale = packing.frac
            after = st.energy_pJ * scale
        elif stage.kind == "network":
            ws, ss = packing.wire_scale(), packing.switching_scale(st.block_bits)
            after = st.wire_pJ * ws + st.switch_pJ * ss
            scale = after / st.energy_pJ if st.energy_pJ else 1.0
        else:
            # RULE 1: the owner of this stop's narrowing is decided by the
            # bar's own measured Word bits -- 1.0 when the plan is q-bit.
            scale = packing.storage_scale(st.block_bits, st.word_bits)
            after = st.energy_pJ * scale
        saved = st.energy_pJ - after
        saved_by_cat[cat] = saved_by_cat.get(cat, 0.0) + saved
        stage_rows.append({**st.to_dict(), "category": cat,
                           "carries_reduced": True, "scale": scale,
                           "energy_after_pJ": after, "energy_saved_pJ": saved})

    # ---- 2. what reconstruction costs --------------------------------------
    st_site = wpath.stages[placement.site_stage]
    accesses = st_site.counter(placement.site_counter)
    if accesses_override is not None:
        # AN ERT-AWARE BAR IS BILLED ON THE WORDS TIMELOOP BILLED. Timeloop
        # charges `ceil(scalar / block_size)` vector accesses -- a partially
        # filled last word costs a whole word, and the encoder rebuilds that
        # whole word -- so the toll sitting inside the level covers
        # `ceil(scalar/bs) x bs` weights, slightly more than `scalar`. RULE
        # 5.3 is "a split, not an addition": what the evaluator charges must
        # be what it moved OUT of the level, to the pJ, so an ERT arm counts
        # the padded weights and every other bar counts Timeloop's scalars.
        # On resnet18/BCH(63,39) the padding is 876 weights in 12,817,664
        # (0.0068 %, 471 pJ in 6.89 uJ) -- immaterial to any bar, fatal to a
        # 1e-6 reconciliation. `ert_aware_view()` measures it per shape.
        accesses = float(accesses_override)
    n_cw = gran.codewords(accesses)
    e_incremental = n_cw * recon_pj
    if cycles is None:
        cycles = float(getattr(wpath, "cycles", 0.0) or 0.0)
    engine_cycles, engines, engines_rule = engine_cycles_for(
        placement, wpath, stage_defs, cycles)
    if recon_idle_pj and not cycles:
        raise ValueError(
            f"{arch}/{placement.key}: the idle term is {recon_idle_pj:g} pJ per cycle "
            f"per engine but the plan carries no cycle count (RULE 3: refusing to "
            f"charge it as zero; the stats file has no `Cycles:` line)")
    # prompt_7 Issue 4 -- CLOCK GATING. `recon_idle_pj` is the DC idle-window
    # TOTAL, which is ~99.5% clock/dynamic power and ~0.5% true leakage. An
    # engine that is clock-gated when no weight is arriving does not burn the
    # dynamic part, so:
    #
    #   E = (incremental + idle) x events                      <- working
    #     + idle x (1 - g) x (engine_cycles - events)          <- gated off
    #
    # `incremental + idle` is active_per_codeword, so no new constant is
    # needed. At g = 0 this is ALGEBRAICALLY IDENTICAL to the ungated model
    # (incremental x events + idle x engine_cycles), which is what
    # test_gating_reproduces_ungated asserts to the pJ.
    #
    # IT IS BOOKED IN THE EQUIVALENT REGROUPED FORM, because prompt_6 RULE 5.3
    # compares the ACCESS term alone against what the ERT bump moved out of the
    # level, and `archs.ert_bump()` splits the same energy the other way:
    #
    #   n x (incremental + idle) + idle x (1-g) x (ec - n)
    #     == n x (incremental + idle x g) + idle x (1-g) x ec       [exactly]
    #
    # The TOTAL is identical to the last bit -- this moves nothing between the
    # bars, it decides which of the two terms each piece is booked under. The
    # first grouping put `n x idle x (1-g)` in the ACCESS term that the ERT
    # bump had priced per CYCLE, so 5.3 failed by exactly that amount: 22,698
    # pJ in 6.91 uJ at BCH(63,39) (3.3e-3, tolerance 1e-6), and 907 pJ under
    # the DC table in force before 2026-09-12. It also made the access term
    # discontinuous at g -> 0; the regrouped form matches the ungated branch
    # below continuously.
    # EVALUATOR-SIDE ONLY: the ERT toll an ERT-aware arm was MAPPED under still
    # carries the ungated idle, so such a mapping was chosen against a
    # pessimistic standby assumption. Stated, not silently corrected -- fixing
    # it would cold every mapper cache.
    idle_pj = float(recon_idle_pj or 0.0)
    gate = max(0.0, min(1.0, float(getattr(cfg, "recon_clock_gating_pct", 0.0)) / 100.0))
    if gate:
        e_incremental = n_cw * (recon_pj + idle_pj * gate)
        e_idle = idle_pj * (1.0 - gate) * engine_cycles
    else:
        e_idle = idle_pj * engine_cycles
    recon_energy = e_incremental + e_idle

    # No reconstruction boundary carries a reuse register any more (R4b was
    # removed 2026-09-10: consecutive weight reuse is 1 on 20 of 21 resnet18
    # layers, so a latch catches nothing, and a register that DOES pay has to
    # hold the whole inner tile -- up to 384 weights, the entire scratchpad).
    # `Recon overhead` is structurally zero. It is kept ONLY so the result
    # schema and the CSV columns are unchanged -- `report.stacked.active_categories`
    # already drops any category that is zero across every bar, so it reaches
    # neither the stacks nor the legend (prompt_7 Issue 10, checked 2026-09-12:
    # a sliver on an R bar is `Reconstruction`, a different, non-zero category).
    overhead = 0.0

    # ---- 3. assemble the stack ---------------------------------------------
    components = {c: float(v) for c, v in base_by_cat.items()}
    for cat, saved in saved_by_cat.items():
        components[cat] = components.get(cat, 0.0) - saved
    components["Reconstruction"] = components.get("Reconstruction", 0.0) + recon_energy
    components["Recon overhead"] = components.get("Recon overhead", 0.0) + overhead
    components["ECC decode"] = components.get("ECC decode", 0.0) + decode_pj
    total = float(sum(components.values()))

    detail = {
        "placement": {
            "key": placement.key, "boundary": placement.label,
            "hypothesised_rating": placement.rating,
            "description": placement.description,
            "reduced_stages": list(placement.reduced),
            "reconstruction_site": {
                "stage": placement.site_stage, "counter": placement.site_counter,
                },
        },
        "feasibility": feas,
        "narrowing_ownership": {
            "rule": ("prompt_6 RULE 1: per storage stop carrying narrow weights, "
                     "the owner is decided by this bar's own measured Word bits -- "
                     "q means the mapper narrowed it (evaluator x1.0), weight_bits "
                     "means the evaluator does; exactly one site is live"),
            "stops": narrowing},
        "reduced_representation": packing.to_dict(),
        "reconstruction_granularity": gran.to_dict(),
        "reconstruction_counts": {
            "weights_reconstructed": accesses,
            # prompt_7 Issue 4: what gating removed. On an ERT-aware bar the
            # level was billed the UNGATED leak toll and the split moves all of
            # it out, so prompt_6 RULE 5.3 becomes "moved out == charged +
            # gating_credit_pJ". The credit is the real saving from gating and
            # must be visible, never silently dropped.
            "clock_gating_pct": gate * 100.0,
            "gating_credit_pJ": (idle_pj * engine_cycles + n_cw * recon_pj
                                 - recon_energy),
            "counter": placement.site_counter,
            "counter_meaning": {
                "reads": "one reconstruction per weight read out of the stage",
                "fills": "one reconstruction per weight written into the stage",
                "ingresses": ("one reconstruction per weight word INJECTED "
                              "into the network: ONE encoder before the fanout "
                              "(ECC_RECON_ENCODER_SITE=source)"),
                "deliveries": ("one reconstruction per weight word ARRIVING at "
                               "a destination of the network, i.e. ingresses x "
                               "multicast factor: one encoder PER DESTINATION "
                               "(ECC_RECON_ENCODER_SITE=destination)"),
            }[placement.site_counter],
            # Both readings of a network boundary, always, so the multiplicity
            # is visible rather than implied by which counter was chosen.
            "encoder_site": encoder_site(cfg),
            "network_multicast_factor": st_site.multicast,
            "network_fanout": st_site.fanout,
            "if_one_encoder_before_the_fanout": st_site.ingresses,
            "if_one_encoder_per_destination": st_site.deliveries,
            "multicast_multiplicity_of_this_boundary": (
                (st_site.deliveries / st_site.ingresses)
                if st_site.ingresses > 0 else 1.0),
            "reconstruction_events_codewords": n_cw,
            "pJ_per_codeword": recon_pj,
            "incremental_pJ_per_codeword": recon_pj,
            "idle_pJ_per_cycle_per_engine": float(recon_idle_pj or 0.0),
            "engines": engines,                 # cycle-weighted mean of the engines that leak
            "engines_declared": float(st_site.declared_instances or 0.0),
            "engines_utilized_max": float(st_site.instances or 0.0),
            "engine_cycles": engine_cycles,     # sum over layers: engines x cycles
            "engines_rule": engines_rule,
            "cycles": float(cycles or 0.0),
            "reconstruction_energy_incremental_pJ": e_incremental,
            "reconstruction_energy_idle_pJ": e_idle,
            "rule": ("prompt_6 RULE 3: incremental x events + idle_per_cycle x "
                     "engine_cycles, engine_cycles = sum over layers of (engines "
                     "that leak x cycles); a storage site's engines are the "
                     "UTILIZED instances, as Timeloop bills leak (power-gated "
                     "per instance); the two terms have different denominators "
                     "and are never added per codeword"),
            "reconstruction_energy_pJ": recon_energy,
            # R4b and its reuse register were removed on 2026-09-10; no
            # boundary carries per-PE buffering, so this is structurally zero.
            "recon_overhead_energy_pJ": overhead,
        },
        "reducible_weight_energy_pJ": {
            "stages_left_reduced": list(placement.reduced),
            "before_pJ": wpath.reducible_energy(placement.reduced),
            "ceiling_on_the_saving_pJ": (
                wpath.reducible_energy(placement.reduced) * (1.0 - packing.frac)),
            "note": ("the most this boundary could give back if reconstruction "
                     "were free; the DRAM interface share plus the whole "
                     "on-chip weight path is the ceiling for ANY boundary, and "
                     "on this architecture the on-chip part is a small share "
                     "of inference energy"),
        },
        "weight_path_stages": stage_rows,
        "energy_saved_by_category_pJ": saved_by_cat,
        "weight_energy_before_pJ": {c: float(v) for c, v in base_w_by_cat.items()},
        "dram_model": dram_model_detail(wpath, placement, stage_rows, packing),
    }
    return PlacementResult(placement, "evaluated", components, total, detail)


def dram_model_detail(wpath, placement, stage_rows, packing):
    """What this placement did to the two DRAM stages, as a record.

    Replaces the pre-2026-09-09 `dram_unchanged` string. Both checks in
    study/placement_study.py (`dram_scaled_by_K_over_N`) reads these rows AND the DRAM
    component, so a bar cannot report one thing here and draw another.
    """
    rows = {r["stage"]: r for r in stage_rows if r["kind"] == "dram"}
    d = rows.get("dram", {})
    before = d.get("weight_energy_pJ", 0.0)
    after = d.get("energy_after_pJ", before)
    reduced = "dram" in placement.reduced
    return {
        "decode_site": wpath.decode_site,
        "dram_pj_per_bit": wpath.dram_pj_per_bit,
        "dram_cost_provenance": wpath.dram_cost_note,
        "dram_weight_energy_pJ": before,
        "dram_after_pJ": after,
        "dram_reduced": reduced,
        "dram_scale": packing.frac if reduced else 1.0,
        "dram_saving_pJ": before - after,
        "rule": ("the decoder is on the DRAM die and off the fetch path: only "
                 "the k message bits of each n-bit codeword are read out and "
                 "driven off the die, so the WHOLE DRAM weight term is scaled "
                 "by K/N under every boundary, R1 included. The f_if "
                 "array/interface split was removed on 2026-09-09. "
                 "E_background and E_refresh are not modelled (both 0)"),
    }


def _category_of(stage, cfg):
    """The plotted category a weight-path stage lands in.

    Deliberately NOT `timeloop.classify()` on a level name: this has to agree
    with the categories the `Raw` record already holds, and the mapping from a
    stage to a category is a property of the stage, so it is stated here and
    checked against the record before anything is evaluated.
    """
    if stage.kind == "dram":
        return "DRAM"
    if stage.kind == "network":
        return "NoC"
    if stage.key.endswith("_glb"):
        return "Global buffer (read)" if cfg.split_read_write else "Global buffer"
    return "Local (read)" if cfg.split_read_write else "Local (spads/RF)"


def cross_check(cfg, arch, wpath, base_w_by_cat, tol=1e-6):
    """Every weight-path stage total must reconcile with the `Raw` record.

    The placement model subtracts savings from the categories the `Raw` record
    holds, so a stage whose energy this module measured differently from the
    record would move a bar by an amount that is nowhere in Timeloop's output.
    Returns `(ok, detail)` and is recorded as a check on every result.
    """
    mine = {}
    for stage in stages_for(arch, cfg):
        st = wpath.stages.get(stage.key)
        if st is None or st.energy_pJ <= 0:
            continue
        cat = _category_of(stage, cfg)
        mine[cat] = mine.get(cat, 0.0) + st.energy_pJ
    rows, ok = {}, True
    for cat in sorted(set(mine) | {c for c, v in base_w_by_cat.items()
                                   if float(v) > 0}):
        a, b = mine.get(cat, 0.0), float(base_w_by_cat.get(cat, 0.0))
        rel = abs(a - b) / b if b else (0.0 if a == 0 else 1.0)
        rows[cat] = {"weight_path_module_pJ": a, "raw_record_pJ": b,
                     "relative_difference": rel}
        if rel > 1e-9:
            ok = False
    return ok, {
        "per_category": rows,
        "unclaimed_weight_levels": wpath.unclaimed,
        "dram_term": wpath.dram_term(),
        "rule": ("this module re-parses the cached stats and must reproduce the "
                 "`Raw` record's WEIGHT energy per category exactly; a level "
                 "carrying weight energy that no stage claims is listed above "
                 "and makes this check fail. The DRAM category is the one "
                 "`dram` stage: the array/interface split by f_if was removed "
                 "on 2026-09-09 and the whole term is reducible"),
    }


