"""Reconstruction granularity and feasibility -- `G_rec`, and what it costs.

02_reconstruction_dse_and_implementation.txt section 15 defines `G_rec`: the
number of weights whose retained bits must be available at once to rebuild an
ECC group. Under the embedded layout the codeword IS n consecutive bits of the
weight bit stream (`physics/embedded.py`), so one codeword spans
`n / weight_bits` weights and, because it is not weight-aligned, touches up to
`ceil(n / weight_bits) + 1` of them -- 9 at BCH(63, K) over 8-bit weights.

GROUP RESIDENCY IS REPORTED, NOT REFUSED (`ECC_RECON_REQUIRE_GROUP_RESIDENCY`,
default 0, decided 2026-09-13). The old refusal assumed an engine that can only
rebuild from weights co-resident AT ONE INSTANT; RECAP's accumulates the
retained bits as they arrive, so a level holding 6 -- or 1 -- still feeds it,
over more accesses and with more buffering. A small tile is a COST, not an
impossibility. The number is still measured and still on every bar's record
(`layers_below_G_rec`, `infeasible_layers`, `group_residency_note`) because it
bounds the buffer the engine needs; set the knob to 1 for the conservative
reading. Measured: this recovers R2/R3/R4/R5a on mobilenet_v2 (1, 1, 4 and 14
layers of 31 below `G_rec`) and changes resnet18 by nothing.

`engines_for()` and `engine_cycles_for()` are the denominators RULE 3's second
term is charged over: 1 at DRAM, THAT LAYER'S OWN fanout x instances at a
network, and at a storage level the UTILIZED instances of that layer's plan --
Timeloop power-gates each unused instance and bills `leak x utilized x cycles`.

PURE ARITHMETIC over numbers handed in, which is why ProjectRestructure phase 3
filed it at L1.
"""
from __future__ import annotations

from . import embedded
from dataclasses import dataclass


# ===========================================================================
#  reconstruction granularity and feasibility  (02_..., section 15)
# ===========================================================================
@dataclass(frozen=True)
class Granularity:
    """`G_rec` and how encoder work is charged.

    Section 15: "The encoder may operate at codeword granularity rather than
    independent weight granularity ... reconstructing one weight can require
    retained bits from several weights." Under the embedded layout the codeword
    is n consecutive bits of the weight bit stream, so:

    `weights_per_codeword`  n / weight_bits, fractional (7.875 at BCH(63,K)
                            over 8-bit weights) -- the same number the embedded
                            arm counts DRAM codewords with, so the two arms
                            cannot disagree about what a codeword is.
    `g_rec`                 the most weights one codeword TOUCHES. A codeword
                            starting mid-weight reaches into one more weight
                            than its length implies, so this is
                            ceil(n/weight_bits)+1 for the worst-aligned phase
                            (9 at (63, 8)) and it is what a PE must hold at once
                            for a local boundary to be feasible at all.

    Charging, `ECC_RECON_ENCODER_GRANULARITY`:

    `weight`    (default) encoder work is proportional to the weights actually
                reconstructed, charged at the synthesized per-codeword energy
                per `weights_per_codeword` of them. This is the AMORTIZED
                reading: the group is rebuilt once and every weight in it is
                consumed.
    `codeword`  every access at the boundary rebuilds one whole codeword,
                whether or not the rest of the group is used. The pessimistic
                reading, and the right one if nothing buffers the group.
    """
    n: int
    k: int
    weight_bits: int
    mode: str

    @property
    def weights_per_codeword(self):
        return embedded.EmbeddedLayout(self.n, self.k,
                                       self.weight_bits).weights_per_codeword

    @property
    def g_rec(self):
        layout = embedded.EmbeddedLayout(self.n, self.k, self.weight_bits)
        w = self.weight_bits
        worst = 0
        for phase in range(layout.period_codewords):
            start = phase * self.n
            first = start // w
            last = (start + self.n - 1) // w
            worst = max(worst, last - first + 1)
        return worst

    def codewords(self, accesses):
        """Reconstruction events for `accesses` weights crossing the boundary."""
        if self.mode == "codeword":
            return float(accesses)
        return float(accesses) / self.weights_per_codeword

    def to_dict(self):
        return {
            "charging": self.mode,
            "meaning": ("weight: encoder work is proportional to the weights "
                        "reconstructed, charged at the synthesized per-codeword "
                        "energy per n/weight_bits of them (the group is rebuilt "
                        "once and all of it is consumed). codeword: every access "
                        "at the boundary rebuilds a whole codeword."),
            "weights_per_codeword": self.weights_per_codeword,
            "G_rec_weights_touched_per_codeword": self.g_rec,
            "G_rec_meaning": ("weights whose retained bits must be available at "
                              "once to rebuild one ECC group; a codeword that "
                              "starts mid-weight reaches into one weight more "
                              "than its length implies"),
            "source": ("02_reconstruction_dse_and_implementation.txt section 15; "
                       "the layout is eccenergy/embedded.py's"),
        }


def feasibility(placement, arch, wpath, gran, require_group_residency=False):
    """Can this boundary assemble a complete ECC group where it sits?

    Returns `(ok, detail)`. Two STRUCTURAL ways to fail, always refused:

    * the stage the encoder sits at is not in the model at all;
    * the stage it would reduce is not in the model.

    And one about GROUP RESIDENCY, which is a statement about the
    RECONSTRUCTION DATAPATH and not about Timeloop: a PE-local boundary whose
    level holds fewer than `G_rec` weights at once. Under the embedded layout a
    63-bit codeword drifts across weight boundaries and reaches into up to 9
    8-bit weights, so rebuilding from a level that holds 6 means 3 of those
    weights' retained bits are not there AT THAT INSTANT.

    **REPORTED, NOT REFUSED, SINCE 2026-09-13** (`ECC_RECON_REQUIRE_GROUP_RESIDENCY`,
    default 0 -- the user's decision about their own engine). The refusal
    assumed the engine can only rebuild from weights co-resident in the level
    in one cycle. RECAP's engine does not work that way: it accumulates the
    retained bits as they arrive and rebuilds when the group completes, so a
    level holding 6 -- or 1 -- still feeds it, just over more accesses. The
    residency figure is still MEASURED and still carried on the record and in
    the manifest (`group_residency`), because it bounds how much buffering the
    engine needs; it no longer deletes a bar.

    Set the knob to 1 to restore the refusal -- the conservative reading, and
    the right one for an engine with no group buffer.
    """
    missing_site = placement.site_stage not in wpath.stages or \
        wpath.stages[placement.site_stage].energy_pJ <= 0
    missing_reduced = [k for k in placement.reduced
                       if k not in wpath.stages or wpath.stages[k].energy_pJ <= 0]
    if missing_site:
        return False, {
            "reason": f"stage {placement.site_stage!r} carries no weight energy in "
                      f"{arch} as modelled, so there is no such boundary to "
                      f"evaluate",
            "stages_present": [k for k, s in wpath.stages.items() if s.energy_pJ > 0]}
    if missing_reduced:
        return False, {
            "reason": f"stage(s) {missing_reduced} would have to carry the reduced "
                      f"representation but carry no weight energy in {arch} as "
                      f"modelled",
            "stages_present": [k for k, s in wpath.stages.items() if s.energy_pJ > 0]}

    # Group assembly is only in question where the reduced form is STORED in a
    # PE: a boundary above the scratchpad reconstructs from a stream the memory
    # interface is feeding in codeword order, so the group is inherently whole.
    local = [k for k in placement.reduced
             if wpath.stages[k].kind == "storage"]
    if not local:
        return True, {"group_assembly": "not PE-local: the reduced form is only in "
                                        "transit, in codeword order from the ECC "
                                        "engine, so a whole group is always available",
                      "G_rec": gran.g_rec}
    bad = []
    for lp in wpath.per_layer:
        for key in local:
            st = lp["stages"].get(key)
            if st is None:
                continue
            cap = st.get("weights_resident_per_instance") or 0
            if cap < gran.g_rec:
                bad.append({"layer": lp["layer"], "shape": lp["shape"],
                            "stage": key, "weights_resident": cap,
                            "G_rec_required": gran.g_rec})
    # THE SHORTFALL IS A FACT ABOUT THE MAPPING; WHETHER IT DISQUALIFIES THE
    # BOUNDARY IS A FACT ABOUT THE ENGINE. Measure the first here, let the knob
    # decide the second.
    ok = (not bad) or (not require_group_residency)
    detail = {
        "G_rec": gran.g_rec,
        "group_residency_required": bool(require_group_residency),
        "layers_below_G_rec": len(bad),
        "layers_total": len(wpath.per_layer),
        "rule": ("a PE-local boundary needs G_rec weights of the codeword group "
                 "resident at once; a smaller resident tile means the retained "
                 "bits the rebuild depends on were never sent to that PE"),
        "checked_stages": local,
        "infeasible_layers": bad,
        "verdict": "feasible for every mapped layer" if ok else
                   f"{len(bad)} (layer, stage) pair(s) hold fewer than G_rec "
                   f"weights -- the placement is rejected rather than estimated",
    }
    if bad:
        names = ", ".join(sorted({b["layer"] for b in bad}))
        worst = min(b["weights_resident"] for b in bad)
        note = (f"{len(bad)} (layer, stage) pair(s) keep fewer than G_rec = "
                f"{gran.g_rec} weights resident (fewest: {worst}). Layers: {names}")
        if ok:
            # REPORTED. The engine accumulates retained bits across accesses,
            # so a level holding fewer than G_rec still feeds it -- over more
            # accesses, and with more buffering. The number is what bounds that
            # buffer, so it travels with the bar instead of deleting it.
            detail["group_residency_note"] = (
                note + ". CHARGED ANYWAY (ECC_RECON_REQUIRE_GROUP_RESIDENCY=0): "
                "the reconstruction engine assembles a group as the retained "
                "bits arrive rather than needing all G_rec weights resident at "
                "one instant, so a smaller tile costs buffering and accesses, "
                "not feasibility. Set the knob to 1 for the conservative "
                "reading, which refuses the bar instead.")
        else:
            # Every unsupported return has to carry `reason`: it is what the
            # result store writes as `unavailable_reason`, and it refuses a
            # variant that cannot say why it is unavailable.
            detail["reason"] = (
                f"infeasible local placement: {note} -- so the retained bits this "
                f"boundary would rebuild from were never delivered to that PE "
                f"(ECC_RECON_REQUIRE_GROUP_RESIDENCY=1)")
    return ok, detail


def narrowing_audit_for_bar(placement, wpath, packing, stage_defs):
    """prompt_6 RULE 1, once per BAR per STAGE: for every storage stop the
    placement says carries narrow weights, who narrows it -- from the bar's
    OWN measured `Word bits` and the scale the evaluator would apply.

    Returns `(ok, rows)`; a row with `problem` set names a stop narrowed twice
    or by nobody. `evaluate_placement` calls this and refuses on `not ok`.
    """
    rows = []
    for stage in stage_defs:
        if stage.kind != "storage" or stage.key not in placement.reduced:
            continue
        st = wpath.stages.get(stage.key)
        if st is None or st.energy_pJ <= 0:
            continue
        try:
            scale = packing.storage_scale(st.block_bits, st.word_bits)
            row = packing.narrowing_site(st.word_bits, scale)
        except ValueError as exc:            # Word bits neither q nor weight_bits
            row = {"measured_word_bits": st.word_bits, "q": packing.declared_q,
                   "weight_bits": packing.weight_bits, "owner": "UNDEFINED",
                   "mapper_live": None, "evaluator_live": None,
                   "applied_scale": None, "ok": False, "problem": str(exc)}
        rows.append(dict(row, stage=stage.key, levels=list(st.levels),
                         physical_word_bits=st.block_bits))
    return all(r["ok"] for r in rows), rows


def engine_cycles_for(placement, wpath, stage_defs, cycles=None):
    """prompt_6 RULE 3's idle denominator for one bar: engine-cycles, i.e.
    sum over the plan's layers of (engines that leak x that layer's cycles).
    Returns `(engine_cycles, effective_engines, rule)`.

        dram site      1 engine at the chip ingress                x cycles
        network site   fanout x network instances = the destinations
                       the network delivers to (14 at Eyeriss v1's
                       column edge)                                x cycles
        storage site   the level's UTILIZED instances, PER LAYER: Timeloop
                       power-gates each unused instance (`Instances sharing
                       power gating: 1`) and bills `leak` x utilized x cycles
                       (buffer.cpp FinalizeBufferEnergy; verified 2026-09-11
                       on 43 shapes, multiplier == utilized every time). An
                       encoder sits in its PE and is gated with it. The
                       declared count (168) is reported beside it.

    `cycles` is the run length the caller bills from (RULE 4: another plan's
    cycles override the path's own); the storage engine-cycles are rescaled
    to it so `idle x engine_cycles` and `idle x cycles x engines` agree.
    """
    by_key = {s.key: s for s in stage_defs}
    site = by_key[placement.site_stage]
    st = wpath.stages[placement.site_stage]
    own = float(getattr(wpath, "cycles", 0.0) or 0.0)
    cyc = float(cycles) if cycles is not None else own
    if site.kind == "dram":
        return 1.0 * cyc, 1.0, "one engine at the chip ingress"
    if site.kind == "network":
        # prompt_7 Issue 3, fixed 2026-09-12. This used to charge
        # `max fanout x max instances x TOTAL RUN CYCLES`, throwing away the
        # per-layer fanout that `weight_path` had already computed and
        # `_aggregate` had already summed. On a workload whose layers all
        # multicast the same width (resnet18: 14 everywhere) that is exact; on
        # one with a mix (mobilenet_v2: 2,3,4,7,10,12,14 -- depthwise layers
        # broadcast narrowly) it over-billed the standby term by x1.3336.
        # Now identical in form to the storage branch below.
        ec = float(st.engine_cycles or 0.0)
        if own and cyc != own:
            ec *= cyc / own
        declared = max(1.0, float(round(float(st.fanout) * float(st.instances or 1))))
        eff = (ec / cyc) if cyc else declared
        return ec, eff, (f"one engine per destination of the network, PER LAYER "
                         f"(fanout x instances, summed over layers); widest layer "
                         f"{declared:g}, cycle-weighted mean {eff:.2f}")
    ec = float(st.engine_cycles or 0.0)
    if own and cyc != own:
        ec *= cyc / own
    eff = (ec / cyc) if cyc else float(st.declared_instances or st.instances or 1)
    return ec, eff, (f"one engine per UTILIZED instance of {'+'.join(st.levels) or site.key}, "
                     f"per layer (Timeloop power-gates unused instances and bills leak x "
                     f"utilized x cycles); declared {st.declared_instances:g}, utilized "
                     f"max {st.instances:g}, cycle-weighted mean {eff:.2f}")


def engines_for(placement, wpath, stage_defs, cycles=None):
    """`(effective engines, rule)` -- the cycle-weighted mean of the engines
    that leak; see `engine_cycles_for`. 1 / 14 / <= 168 on Eyeriss v1."""
    _ec, eff, rule = engine_cycles_for(placement, wpath, stage_defs, cycles)
    return eff, rule


