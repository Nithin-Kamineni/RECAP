"""The placement figure: one panel per design, and the `run.sh recon` entry point.

`figure()` draws what `study/placement_study.py` computed, through
`report/panels.py` and ultimately `stacked.draw_panel` -- still the only routine
in the project that draws a bar. `panel_for()` builds one design's panel: its own
x axis, its own boundary list and its own two reference bars, because a
percentage on one panel says nothing about another.

THE HEADING IS ONE LINE (`Config.recon_title` / `recon_panel_title`): design,
model and layer scope, weight width, code, regime tag. Everything it used to say
below that -- the regime with its arms, the DRAM model and price, the static DRAM
terms, the encoder site, the MAC denominator, the development-run warning, the
multi-panel rule -- is `Config.recon_caveats()` and lands in the MANIFEST beside
the figure as `title_caveats`. A new caveat goes THERE, never as a line on the
figure.

`run()` is what `run.sh recon` calls: evaluate, then draw, then finish.

ProjectRestructure phase 3 cut this out of `experiments/recon.py` -- the half
that draws is L5, and that is what ended the last upward import out of the
placement study.
"""
from __future__ import annotations

import math
import pandas as pd

from ..arch.weight_path import DECODE_SITE
from ..arch import arms
from ..arch import placements
from ..arch import weight_path
from ..study import capacity
from ..toolchain import ert
from ..arch.load import load_provenance
from . import panels as panels_mod
from .stacked import grouped_stacks, write_table
from ..study.common import Session
from ..study.energy import plot_cats

from ..study.placement_notes import ONCHIP_EXCLUDE, PARITY_KEY
from ..study.placement_study import evaluate
from ..settings import guards


# --------------------------------------------------------------------- figure
def panel_for(cfg, arch, model, out):
    """ONE panel: the boundary on the x axis, the energy breakdown in the bars.

    Returns everything `draw_panel` and `write_table` need and draws nothing, so
    the same construction serves a one-architecture figure and a panel of a
    multi-architecture one. `figure()` decides which.

    `report/stacked.draw_panel` draws it -- the same routine the three sweeps and
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
        # DRAM band or the drawn bar would silently be shorter than the total it
        # is annotated with. `ecc.build_stacks()` folds it the same way, so the
        # two figures agree. Under the price model (baseline_dram.py) that
        # component is 0 and the baseline's cost is already inside `DRAM`; the
        # fold is then a no-op and this stays correct either way.
        stack = {c: float(components.get(c, 0.0)) for c in cats}
        stack["DRAM"] += float(components.get(PARITY_KEY, 0.0))
        drawn = sum(stack.values())
        total = sum(float(v) for v in components.values())
        if not math.isclose(drawn, total, rel_tol=1e-9, abs_tol=1e-3):
            raise guards.refusal("figure-draws-the-total",
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
        bill = (out.get("billing") or {}).get(key)
        if key in out.get("ert_bars", ()):
            # prompt_6 9: mark which bars came from their own mapping
            lines.append("own mapping (ERT)")
        elif bill is not None:
            # prompt_7 B1: a bar that did NOT come from its own mapping names
            # the plan it did come from, ON THE PIXELS. A figure gets separated
            # from its manifest the moment it is dropped into a slide, and
            # "which chip is this bar" is not recoverable from the bar.
            lines.append(f"{bill['arm']}'s plan")
        if lines:
            notes[key] = "\n".join(lines)

    add("baseline", placements.REFERENCE_BARS[0][1], out["base_components"])
    add("embedded", placements.REFERENCE_BARS[1][1], out["emb_components"])
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

    term = out["wpath"].dram_term()
    extra = {}
    for key, total in (("baseline", base_t), ("embedded", emb_t)):
        extra[key] = {
            "boundary": dict(placements.REFERENCE_BARS)[key].replace("\n", " "),
            "status": "reference", "hypothesised_rating": "",
            "saving_vs_conventional_ecc_pct": pct(base_t, total),
            "saving_vs_embedded_only_pct": pct(emb_t, total),
            "reconstruction_events_codewords": "", "reconstruction_uJ": "",
            "recon_overhead_uJ": "", "weights_reconstructed": "",
            "idle_engines_cycle_weighted": "", "idle_engines_declared": "",
            "pe_shapes_differing_from_reference": "",
            "reducible_energy_uJ": "",
            "saving_ceiling_uJ": "",
            "decode_site": "controller (reference bar)",
            "dram_pj_per_bit": term["dram_pj_per_bit"],
            # the weight share only, and only for the embedded reference bar
            "dram_uJ": (term["dram_weight_energy_pJ"] / 1e6
                        if key == "embedded" else ""),
            "dram_saving_uJ": 0.0 if key == "embedded" else "",
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
                recon_overhead_uJ=c["recon_overhead_energy_pJ"] / 1e6,
                weights_reconstructed=c["weights_reconstructed"],
                idle_engines_cycle_weighted=c["engines"],
                idle_engines_declared=c["engines_declared"],
                pe_shapes_differing_from_reference=(
                    (out.get("views") or {}).get(res.placement.key).checks["pe_utilization"]["shapes_differ"]
                    if (out.get("views") or {}).get(res.placement.key) is not None else ""),
                reducible_energy_uJ=b["before_pJ"] / 1e6,
                saving_ceiling_uJ=b["ceiling_on_the_saving_pJ"] / 1e6,
                decode_site=dm["decode_site"],
                dram_pj_per_bit=dm["dram_pj_per_bit"],
                dram_uJ=dm["dram_after_pJ"] / 1e6,
                dram_saving_uJ=dm["dram_saving_pJ"] / 1e6,
                unavailable_reason="")
        else:
            row.update(saving_vs_conventional_ecc_pct="",
                       saving_vs_embedded_only_pct="",
                       reconstruction_events_codewords="", reconstruction_uJ="",
                       recon_overhead_uJ="", weights_reconstructed="",
                       idle_engines_cycle_weighted="", idle_engines_declared="",
                       pe_shapes_differing_from_reference="",
                       reducible_energy_uJ="",
                       saving_ceiling_uJ="", decode_site=term["decode_site"],
                       dram_pj_per_bit=term["dram_pj_per_bit"], dram_uJ="",
                       dram_saving_uJ="", unavailable_reason=res.reason)
            # An unsupported boundary has no bar, but it must still have a row:
            # a table that simply omits it reads as "not considered".
            groups.append(p.key)
            labels[p.key] = p.short
            stacks[p.key] = pd.DataFrame(
                {"energy": pd.Series({c: 0.0 for c in cats}).reindex(cats)})
        extra[p.key] = row

    drawn = [g for g in groups if float(stacks[g]["energy"].sum()) > 0]
    return {
        "arch": arch, "model": model,
        # `groups` is every bar including the unsupported ones, which have a
        # table row and no bar; `drawn` is what has height. The figure gets
        # `drawn`, the table gets `groups`.
        "groups": groups, "drawn": drawn, "stacks": stacks, "labels": labels,
        "notes": notes, "extra": extra,
        "ref_totals": {g: base_t for g in groups},
        "base_total": base_t, "emb_total": emb_t,
        "mac_ert_pj": (out.get("mac") or {}).get("ert_pj_per_mac"),
    }


#: What the multi-panel placement figure's heading said under its title until
#: 2026-09-11. The heading is one line since; this travels in the manifest's
#: `title_caveats` with the rest (`Config.recon_caveats`).
PANEL_NOTE = ("each panel is one design's OWN weight path, measured against its "
              "OWN two reference bars; the panels share a legend and a unit, not "
              "a y limit or an x axis")


def _title(cfg, panels):
    """The ONE-LINE heading. Everything it used to say below that line is
    `cfg.recon_caveats()`, written to the manifest by `run()`."""
    return cfg.recon_title() if len(panels) == 1 else cfg.recon_panel_title()


def figure(cfg, ses, panels):
    """Draw the placement figure: one panel per architecture, top to bottom.

    ONE ARCHITECTURE -> the single-panel figure this study has always drawn,
    through `grouped_stacks`, unchanged.

    SEVERAL -> `report/panels.stacked_panels`, one panel per design, each with
    its OWN x axis of its own boundaries and its own two reference bars. The
    rule this does not break is the one env.sh section 4 and CLAUDE.md state:
    the boundaries of two designs must never share an x axis, because
    "reconstruct after the mesh" beside a design with no mesh is meaningless.
    Separate stacked axes are not that -- what they share is the page, the
    legend, the category set and the energy unit, and each panel's heading names
    its design. The panels deliberately do NOT share a y limit: two accelerators
    of different size forced onto one scale makes the smaller unreadable.

    ONE PANEL PER DESIGN IS ALSO THE ONLY HONEST LAYOUT for the counts. Each
    design's bars are measured against ITS OWN conventional-ECC bar, so the
    percentages on one panel say nothing about the other; the table carries both
    with the panel in the row key.
    """
    one = len(panels) == 1
    title = _title(cfg, panels)
    if one:
        pan = panels[0]
        drawn, stacks = pan["drawn"], pan["stacks"]
        figs, _csv = grouped_stacks(
            cfg, ses.results,
            groups=drawn, stacks={g: stacks[g] for g in drawn},
            group_labels=pan["labels"], title=title, stem=cfg.stem,
            group_fontsize=15, bars=["energy"], bar_tags={}, bar_width=0.92,
            ref_totals=pan["ref_totals"],
            bar_notes={g: pan["notes"][g] for g in drawn if g in pan["notes"]})
        # ...and the table gets every row, drawn or not.
        csv = write_table(cfg, ses.results,
                          [(None, pan["groups"], stacks, pan["labels"])],
                          cfg.stem, bars=["energy"],
                          ref_totals=pan["ref_totals"],
                          extra_columns=pan["extra"])
        return figs, csv, drawn

    # ---- several designs: one panel each -----------------------------------
    # Every per-group dict is keyed "<panel>/<group>" for the table, because
    # both panels have a group called `recon1` and `write_table` would
    # otherwise give the second panel the first panel's reference total.
    spec, refs, note_by_panel, extra = [], {}, {}, {}
    for pan in panels:
        key = pan["arch"]
        # The heading names the design and says how much of ITS OWN boundary
        # list is on the axis, because the two panels do not have the same
        # number of boundaries and a reader comparing bar counts across panels
        # would otherwise be counting two different things. The reference bars
        # are not boundaries, so they are not in the count.
        n_ref = len(placements.REFERENCE_BARS)
        n_all = len(pan["groups"]) - n_ref
        n_ok = len(pan["drawn"]) - n_ref
        heading = f"{cfg.arch_label(key).replace(chr(10), ' ')}"
        heading += f"   ·   {n_ok} of {n_all} boundaries evaluated"
        if n_all - n_ok:
            heading += f", {n_all - n_ok} infeasible (see the table)"
        spec.append((key, heading, pan["drawn"],
                     {g: pan["stacks"][g] for g in pan["drawn"]}, pan["labels"]))
        refs[key] = pan["ref_totals"]
        note_by_panel[key] = {g: pan["notes"][g] for g in pan["drawn"]
                              if g in pan["notes"]}
        for g in pan["groups"]:
            extra[f"{key}/{g}"] = dict(pan["extra"].get(g, {}),
                                       architecture=cfg.arch_label(key)
                                       .replace(chr(10), " "))
    # The table needs the undrawn rows too, so it is written here rather than by
    # `stacked_panels`, whose figure only ever sees the drawn ones.
    figs, _csv = panels_mod.stacked_panels(
        cfg, ses.results, spec, title=title, stem=cfg.stem, group_fontsize=15,
        bars=["energy"], bar_tags={}, bar_width=0.92, ref_totals=refs,
        bar_notes=note_by_panel,
        panel_note=None)             # PANEL_NOTE goes to the manifest (run())
    csv = write_table(
        cfg, ses.results,
        [(pan["arch"], pan["groups"], pan["stacks"], pan["labels"])
         for pan in panels],
        cfg.stem, bars=["energy"],
        ref_totals={f"{pan['arch']}/{g}": t for pan in panels
                    for g, t in pan["ref_totals"].items()},
        extra_columns=extra)
    return figs, csv, [f"{pan['arch']}/{g}" for pan in panels
                       for g in pan["drawn"]]


# ------------------------------------------------------------------------ run
def run(cfg):
    # THE PHASE IS DERIVED PER ARM (`settings.run.result_phase`, 2026-09-14):
    # a placement mapped on its own chip files under `Post`, the two reference
    # bars under `Pre`, and no knob can contradict either any more.

    model = cfg.models[0]
    # ONE PANEL PER ARCHITECTURE, and every one of them checked BEFORE anything
    # is collected: a design whose two tables have drifted apart, or that has no
    # weight path at all, must stop the run rather than quietly draw one panel.
    for arch in cfg.archs:
        if arch not in weight_path.WEIGHT_PATHS:
            raise guards.refusal("no-weight-path",
                f"no weight path is defined for {arch!r}, so its reconstruction "
                f"boundaries are unknown.\n"
                f"  defined: {', '.join(placements.supported_archs())}\n"
                f"  -> each design's weight path and its list of feasible "
                f"boundaries go in eccenergy/recon.py WEIGHT_PATHS and "
                f"PLACEMENTS, together.")

        space_ok, space = arms.validate_placement_space(arch, cfg)
        if not space_ok:
            raise guards.refusal("weight-path-drifted",
                f"{arch}'s weight path and its placement list have drifted "
                f"apart, so every boundary below the missing stage would be "
                f"reported UNDERSTATED rather than wrong-looking:\n  "
                + "\n  ".join(space["violations"])
                + f"\n  -> {space['fix']}")

        defined = placements.placements_for(arch, cfg)
        unknown = [k for k in cfg.recon_placements_for(arch)
                   if k not in {p.key for p in defined}
                   and k not in {p.variant for p in defined}
                   and k not in ("baseline", "embedded")]
        if unknown:
            raise guards.refusal("unknown-placement",
                f"ECC_RECON_PLACEMENTS[{arch}] names placements this "
                f"architecture does not define: {', '.join(unknown)}\n"
                f"  defined: {', '.join(p.key for p in defined)}\n"
                f"  -> the boundaries are per architecture; see "
                f"eccenergy/recon.py PLACEMENTS")

    ses = Session(cfg).setup()
    ses.collect_all()

    prov = load_provenance()
    panels, outs = [], {}
    for arch in cfg.archs:
        raw = (ses.raws.get(arch) or {}).get(model)
        if raw is None:
            raise guards.refusal("recon-nothing-collected",
                f"nothing collected for {arch}/{model}; see the [skip] lines "
                f"above.\n  -> every architecture of a panelled placement study "
                f"needs its own mapper cache: map it first "
                f"(ECC_RECON_ARCHS={arch} bash hpc/map_by_shape.sh), or drop it "
                f"from ECC_RECON_ARCHS")
        out = evaluate(cfg, ses, prov, arch, model, raw)
        outs[arch] = out
        panels.append(panel_for(cfg, arch, model, out))

    figs, csv, groups = figure(cfg, ses, panels)
    # The heading is one line (2026-09-11). What it said below that line until
    # then -- the mapping regime with its arms, the DRAM model and price, the
    # static DRAM terms, the encoder site, the MAC denominator, the layer
    # scope, and the panel rule on a multi-design figure -- is RECORDED here,
    # beside the figure, instead of drawn on it.
    caveats = cfg.recon_caveats(mac_ert_pj=panels[0]["mac_ert_pj"])
    if len(panels) > 1:
        caveats.append(PANEL_NOTE)
    # PE utilisation per ERT arm (reported, never refused -- see ert_aware_view)
    for a, o in outs.items():
        for key, v in sorted((o.get("views") or {}).items()):
            pe = v.checks.get("pe_utilization") or {}
            if pe.get("shapes_differ"):
                names = ", ".join(f"{s} ({d['pes_arm']} vs {d['pes_reference']} PEs)"
                                  for s, d in pe["per_shape"].items())
                caveats.append(
                    f"{cfg.arch_label(a).replace(chr(10), ' ')} {key}: its own EDP-optimal "
                    f"plan uses a different PE count than the reference plan on "
                    f"{pe['shapes_differ']} of {pe['shapes_total']} layer shapes -- {names}; "
                    f"idle is charged on the instances that leak (utilized, per layer), "
                    f"as Timeloop bills it")
            elif pe:
                caveats.append(f"{cfg.arch_label(a).replace(chr(10), ' ')} {key}: PEs used "
                               f"identical to the reference plan on all {pe['shapes_total']} "
                               f"layer shapes")
    ses.finish(figs, csv, groups, extra={
        "title": _title(cfg, panels),
        "title_caveats": caveats,
        "recon_decode_site": DECODE_SITE,
        "dram_pj_per_bit": cfg.dram_pj_per_bit,
        "dram_cost_provenance": cfg.dram_cost_note,
        "dram_static_terms": cfg.dram_static_note,
        "panel_per_architecture": list(cfg.archs),
        "dram_term": {a: o["wpath"].dram_term() for a, o in outs.items()},
    })

    print("=" * 78)
    if cfg.recon_optimizer and getattr(cfg, "recon_ert_aware", False):
        print(f"prompt_6 reconstruction-AWARE MAPPING, encoder in the objective: {model}, "
              f"one panel per architecture.")
        print(f"  reference plan      : the published 8-bit chip, no toll")
        print(f"  ERT arms            : one mapping each, datawidth q on the storage levels "
              f"the boundary narrows, its toll in the ERT; the other boundaries are "
              f"post-processed on the reference plan")
    elif cfg.recon_optimizer:
        print(f"Task 4 reconstruction-AWARE MAPPING: {model}, one panel per "
              f"architecture.")
        print(f"  reference arm       : weight capacity x{cfg.weight_capacity_scale:g} "
              f"of the declared design")
        print(f"  reconstruction arm  : x"
              f"{capacity.capacity_dilation_scale(cfg):g} "
              f"= x{cfg.weight_capacity_scale:g} x N/K, solved as its own mapping")
    else:
        print(f"Task 3 reconstruction placement: {model}, fixed mapping, "
              f"one panel per architecture.")
    for arch in cfg.archs:
        best = outs[arch]["builder"].best_reconstruction()
        print(f"  {cfg.arch_label(arch).replace(chr(10), ' ')}:")
        if best.get("variant"):
            print(f"    lowest-energy feasible placement: {best['variant']}")
            print(f"      vs conventional ECC : "
                  f"{best.get('savings_vs_conventional_ecc_percent'):+.3f}%")
            print(f"      vs embedded only    : "
                  f"{best.get('savings_vs_embedded_only_percent'):+.3f}%")
        else:
            print(f"    no placement was evaluated: {best.get('reason')}")
    for arch in cfg.archs:
        views = outs[arch].get("views") or {}
        if not views:
            continue
        billing = outs[arch].get("billing") or {}
        print(f"  prompt_7 B1, {cfg.arch_label(arch).replace(chr(10), ' ')}: ONE PLAN PER "
              f"BAR on this panel, and every bar names it --")
        for r in outs[arch]["rows"]:
            b = billing.get(r["key"])
            if b is None:
                continue
            mark = {"own": "OWN mapping (ERT)", "borrowed": "plan borrowed",
                    "foreign": "FOREIGN plan"}[b["kind"]]
            print(f"    {r['key']:<8} {mark:<18} <- {b['arm']:<10} "
                  f"geometry {'matches' if b['geometry_matches'] else 'DOES NOT MATCH'}")
        for key, v in views.items():
            verdict = ("IDENTICAL to the reference -- the ERT changed nothing for this arm"
                       if v.nest_identical else "DIFFERENT from the reference")
            n_same = sum(v.per_shape_nest_identical.values())
            print(f"    {key}: {ert.describe_bump(v.bump)}")
            print(f"      loop nest {verdict} ({n_same}/{len(v.per_shape_nest_identical)} "
                  f"shapes identical); cycles {v.cycles:,.0f}; fp {v.fingerprint}")
            print(f"      moved out of {v.split['moved_out_of_category']}: access "
                  f"{v.split['access_toll_pJ'] / 1e6:.4f} uJ ({v.split['counter']} x E_w), "
                  f"leak {v.split['leak_toll_pJ'] / 1e6:.3f} uJ ({v.split['engines']:.2f} engines x "
                  f"{v.cycles:,.0f} cycles); both reconcile to 1e-6")
    if len(cfg.archs) > 1:
        print("Each design is measured against ITS OWN reference bars, so a")
        print("percentage on one panel says nothing about the other.")
    if cfg.recon_optimizer and getattr(cfg, "recon_ert_aware", False):
        pass                    # the per-arm block above said it all
    elif cfg.recon_optimizer:
        print("The reconstruction bars come from a DIFFERENT mapping than the")
        print("reference bars: same design, N/K more weight room, solved on its")
        print("own. So the two arms no longer refetch identically, and a DRAM")
        print("read the reconstruction arm never issues saves the ARRAY as well")
        print("as the interface -- efficiency 1.0, against the 0.152 a fixed")
        print("mapping buys. Both mapper caches pre-existed this evaluation.")
    else:
        print("This is an evaluator-only, FIXED-MAPPING result. The mapping optimiser")
        print("was not re-run for any placement -- that is Task 4 (RECON_OPTIMIZER).")
    print("=" * 78)
    return ses
