"""What the placement study PRINTS: the per-bar table, latency, narrowing.

Three text reports, in the order `study/placement_study.py` emits them:
`_report()` (one row per bar: energy by category, saving, ceiling, the plan it
was billed from), `_report_latency()` / `latency_table()` (the roofline's
re-timing beside Timeloop's own cycles, with `LATENCY_CEILING_TOL_PP` as the
tolerance a bar may differ by before it is called a ceiling change), and
`_report_narrowing()` (who applied the on-chip narrowing, measured per bar per
stage -- `study/narrowing.py` is the rule, this is the table).

These print; they do not draw. A bar is drawn in exactly one place
(`report/stacked.py draw_panel`), which is why this stayed at L4 when the figure
half went to L5.

ProjectRestructure phase 3 cut this out of `experiments/recon.py`.
"""
from __future__ import annotations

from ..arch import placements
from ..arch.load import accumulator_bits
from ..toolchain import latency_post

from .ert_view import LATENCY_CEILING_TOL_PP


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
    # THE CEILING. Without this, a 0.06% result reads as a missing term rather
    # than as arithmetic. ON-CHIP only: the DRAM term is reducible too, but it
    # is not on chip and gets its own lines below.
    reducible = [s.key for s in placements.stages_for(arch, cfg)
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
    term = wpath.dram_term()
    dram_w = term["dram_weight_energy_pJ"]
    ondie = term["decode_site"] == "ondie"
    print(f"      DRAM weight energy (one term, all of it)  : "
          f"{dram_w / 1e6:12,.3f} uJ = {dram_w / emb_total * 100:5.2f}%   "
          f"({term['dram_cost_provenance']})")
    if ondie:
        print(f"        x K/N on EVERY bar                     : "
              f"{dram_w * packing.frac / 1e6:12,.3f} uJ  (only the k message "
              f"bits are read out and driven off the die)")

    print(f"\n    {'bar':26s} {'total uJ':>13s} {'vs base':>9s} {'vs emb':>9s} "
          f"{'DRAM saved uJ':>14s} {'recon uJ':>12s} {'of it idle':>12s} {'engines':>8s} "
          f"{'N_rec':>16s}")
    for r in rows:
        if r["status"] != "evaluated":
            print(f"    {r['label'][:26]:26s} {'UNSUPPORTED':>13s}   {r['reason'][:60]}")
            continue
        vb = (base_total - r["total"]) / base_total * 100 if base_total else 0.0
        ve = (emb_total - r["total"]) / emb_total * 100 if emb_total else 0.0
        print(f"    {r['label'][:26]:26s} {r['total'] / 1e6:13,.3f} {vb:8.2f}% "
              f"{ve:8.2f}% {r['dram_saved_pJ'] / 1e6:14,.3f} "
              f"{r['recon_pJ'] / 1e6:12,.3f} "
              f"{r.get('idle_pJ', 0.0) / 1e6:12,.3f} {float(r.get('engines', 0)):8.2f} "
              f"{r['n_cw']:16,.0f}")
    print(f"    (vs base / vs emb are SAVINGS: a negative number costs more "
          f"than the reference; DRAM saved is the DRAM energy the bar "
          f"REMOVED against the embedded reference; recon uJ = incremental x "
          f"events + idle x engine-cycles over {wpath.cycles:,.0f} cycles, engines = "
          f"cycle-weighted mean of the instances that LEAK, i.e. the utilized ones, "
          f"RULE 3)")


def latency_table(cfg, raw, packing, plans=None):
    """Re-time every bar of the placement study. prompt_7 Phase A, step A2.

    ONE table, built from the same `Raw` records the energy bars are billed
    from (prompt_6 RULE 4), so a bar's time and its energy come from the same
    plan. The reference bars drive every weight bit off the die
    (`weight_scale` 1.0); every reconstruction bar drives K/N of them, because
    off chip the weights are a BIT stream. Returns None when the roofline is
    off, so a caller prints nothing rather than a column of zeros.

    R-1 travels with it: every placement's `reduced` set contains `dram`, so
    once DRAM binds, the bars are EQUAL by construction and any difference
    between them is the per-shape variation of the plans they were billed from,
    not a property of the boundary.
    """
    if not getattr(cfg, "latency_model", False):
        return None
    # THE PLAN'S OWN SCALE, not an assumed 1.0 (prompt_7 C1.2): the reference
    # plan was solved at full weight demand and re-times there; every arm's own
    # plan was solved at K/N and re-times THERE, or this table reports plans
    # their mappers never chose.
    ref = latency_post.model_cycles(raw, cfg)
    if ref is None:
        return None
    frac = packing.frac                     # K/N, this run's reduced fraction
    rows = [{"bar": "baseline", "weight_scale": 1.0, **ref},
            {"bar": "embedded", "weight_scale": 1.0, **ref}]
    ref_t = latency_post.offchip_items(raw, weight_scale=1.0)
    for key, (bar_raw, tag) in sorted((plans or {}).items()):
        one = latency_post.model_cycles(bar_raw, cfg, weight_scale=frac)
        # `frac` is an assertion here, not an instruction: `relief_owner`
        # refuses if the bar's own plan was solved at a different code's scale.
        if one is None:
            continue
        # THE CEILING IS THE REFERENCE PLAN'S OFF-CHIP TRAFFIC, and it bounds
        # the WEIGHT term only. A bar billed from its own mapping is a
        # different plan, and a different plan moves a different amount of
        # ACTIVATION traffic off chip -- which the ceiling says nothing about.
        # Measured on mobilenet_v2 (2026-09-12): recon2's own plan drives
        # 30,508,939 activation items off chip against the reference plan's
        # 31,042,059, at IDENTICAL weight reads, which is the whole of the
        # 1.52 pp by which that bar passes the 4.32% ceiling. Before prompt_7
        # B1 every bar shared one plan and this was unreachable; it has to be
        # separable now or the table contradicts its own ceiling line.
        t = latency_post.offchip_items(bar_raw, weight_scale=1.0)
        rows.append({"bar": key, "plan": tag, "weight_scale": frac,
                     "offchip_items": t,
                     "offchip_items_reference": ref_t,
                     "weight_items_vs_reference": (
                         t["weight_items"] / ref_t["weight_items"]
                         if ref_t["weight_items"] else None),
                     "activation_items_vs_reference": (
                         t["other_items"] / ref_t["other_items"]
                         if ref_t["other_items"] else None),
                     **one})
    return {"rows": rows,
            "reference_cycles": ref["cycles"],
            "timeloop_cycles": raw.cycles,
            "ceiling": latency_post.ceiling(raw, cfg, weight_scale=frac),
            "ceiling_scope": (
                "THE CEILING IS COMPUTED ON THE REFERENCE PLAN'S OFF-CHIP TRAFFIC AND "
                "BOUNDS THE WEIGHT TERM ONLY. Since prompt_7 B1 a bar can be billed from "
                "a plan of its own, and a different plan moves a different amount of "
                "ACTIVATION traffic off chip -- which this ceiling says nothing about. A "
                "bar past the ceiling is therefore a MAPPING difference, not a "
                "reconstruction one, and the two are separable on every row: "
                "`weight_items_vs_reference` is the reconstruction side and "
                "`activation_items_vs_reference` the mapping side. A bar at 1.000 on BOTH "
                "that still beats the ceiling would be a bug"),
            "model": latency_post.describe(cfg),
            "rule_R1": ("latency is FLAT across the boundaries by construction: "
                        "every placement's `reduced` set contains `dram`, so "
                        "every placement gets the same off-chip relief. A "
                        "difference between two bars here is the per-shape "
                        "variation of the plans they are billed from"),
            "rule_R3": ("R3 = R2 BY CONSTRUCTION. Timeloop has no network "
                        "timing model, so the INCREMENTAL benefit of the array "
                        "multicast carrying reduced-width words is unmodelled "
                        "-- not measured and found ineffective"),
            # prompt_6 RULE 1, for the off-chip relief specifically. Before
            # prompt_7 C1.2 the evaluator was always the owner, because no
            # architecture declared a bandwidth scale and the mapper could not
            # know a reconstruction arm moves K/N of the weight bits. Now the
            # arm declares it, the mapper solves against the reduced demand,
            # and applying `weight_scale` here as well would take the relief
            # twice. `latency_post.relief_owner()` reads the answer off each
            # bar's OWN stats (`Bandwidth Consumption Scale`) and refuses a
            # third value -- it is measured, not assumed.
            "plans_solved_at": sorted({o for r in rows
                                       for o in (r.get("plans_solved_at") or [])})}


def _report_latency(cfg, table):
    """The latency table on the console, CEILING FIRST like every other result."""
    if not table:
        return
    c = table["ceiling"]
    t = c["offchip_items"]
    print(f"\n    --- LATENCY (prompt_7 Phase A roofline, evaluator-only) ---")
    print(f"    {table['model']}")
    print(f"      off-chip items          : {t['total_items']:,.0f}  "
          f"(weights {t['weight_items']:,.0f} = {t['weight_share'] * 100:.1f}%, "
          f"inputs+outputs {t['other_items']:,.0f})")
    print(f"      CEILING on any boundary : weight share {t['weight_share'] * 100:.1f}% "
          f"x (1 - K/N) {c['one_minus_k_over_n']:.4f} = "
          f"{c['latency_ceiling_pct']:.2f}%  <-- the most RECONSTRUCTION can save")
    print(f"      (an UPPER bound, reached only when off-chip bandwidth binds on "
          f"every layer; total time is max(compute, movement), not their sum. It is "
          f"computed on the")
    print(f"       REFERENCE plan's traffic and bounds the WEIGHT term only -- the "
          f"`wt` and `act` columns below are each bar's own off-chip traffic against it)")
    ref = table["reference_cycles"]
    print(f"\n    {'bar':26s} {'cycles':>14s} {'ms':>10s} {'vs reference':>13s} "
          f"{'wt':>7s} {'act':>7s} {'plan':>21s} {'binding':>12s}")
    for r in table["rows"]:
        gain = ((ref - r["cycles"]) / ref * 100.0) if ref else 0.0
        binding = ", ".join(f"{k} x{v}" for k, v in
                            sorted(r["binding_levels"].items(), key=lambda kv: -kv[1]))
        over = " *" if gain > c["latency_ceiling_pct"] + LATENCY_CEILING_TOL_PP else ""
        wt = r.get("weight_items_vs_reference")
        act = r.get("activation_items_vs_reference")
        print(f"    {r['bar'][:26]:26s} {r['cycles']:14,} {r['seconds'] * 1e3:10,.3f} "
              f"{gain:11.2f}%{over:>2s} "
              f"{('x%.3f' % wt) if wt else '      -':>7s} "
              f"{('x%.3f' % act) if act else '      -':>7s} "
              f"{str(r.get('plan', '-'))[:21]:>21s} {binding[:12]:>12s}")
    if any(((ref - r["cycles"]) / ref * 100.0 if ref else 0.0)
           > c["latency_ceiling_pct"] + LATENCY_CEILING_TOL_PP for r in table["rows"]):
        print(f"    (* past the ceiling. The ceiling bounds the WEIGHT term on the "
              f"REFERENCE plan's traffic; a bar billed from a plan of its own moves a "
              f"different amount of ACTIVATION traffic off chip, which the ceiling says "
              f"nothing about. `act` is that difference -- a MAPPING effect, not a "
              f"reconstruction one. A row at x1.000 on both wt and act that still passed "
              f"the ceiling would be a bug.)")
    print(f"    (Timeloop's own count for the reference plan was "
          f"{table['timeloop_cycles']:,} cycles; the roofline reproduces it "
          f"EXACTLY at unlimited off-chip bandwidth)")
    print(f"    R-1: {table['rule_R1']}")
    print(f"    R-3: {table['rule_R3']}")


def _report_narrowing(results):
    """RULE 1 on the console: per bar, per narrow storage stop, who narrowed
    it and from what measurement (prompt_6 10, item 2)."""
    print(f"\n    narrowing ownership (RULE 1), from each bar's own stats:")
    for res in results:
        if res.status != "evaluated":
            continue
        stops = res.detail.get("narrowing_ownership", {}).get("stops", [])
        if not stops:
            print(f"      {res.placement.key:8s} no on-chip storage stop carries narrow weights")
            continue
        parts = [f"{r['stage']}: Word bits {r['measured_word_bits']} -> "
                 f"{r['owner']} (x{r['applied_scale']:.4f})" for r in stops]
        print(f"      {res.placement.key:8s} " + "; ".join(parts))


