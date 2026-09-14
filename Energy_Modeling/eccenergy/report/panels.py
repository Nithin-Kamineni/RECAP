"""One figure, several stacked panels drawn by `stacked.draw_panel`.

TWO CALLERS, ONE RENDERER.

* the DRIVER AT THE FOOT OF THIS FILE: one panel per model, the same ARCHITECTURE SWEEP in
  each. It answers the one question a single sweep cannot -- does the
  architecture ranking survive changing the network? -- because the claim is
  about the SHAPE of the two panels, not about either one alone.
* `report/recon_view.py`: one panel per ARCHITECTURE, that design's own
  reconstruction boundaries in each. A reconstruction boundary is
  design-specific, so the boundaries of two designs must never share an x axis
  (CLAUDE.md, env.sh section 1) -- "reconstruct after the mesh" beside a design
  with no mesh is meaningless. Two panels is not that: each keeps its own axis,
  its own boundary list and its own reference bars, and what is shared is the
  page, the legend, the category set and the energy unit. The panel HEADING
  names the design, so no bar can be read as belonging to the other one.

The parameters after `panel_note` exist for the second caller and are passed
straight to `draw_panel`, which is still the only routine in the project that
draws a bar. `ref_totals`, `bar_notes` and `extra_columns` are per panel, keyed
by panel key, because two panels have groups of the same name.

NEITHER CALLER IS A NEW SWEEP and neither adds a renderer: no panel introduces
an x axis the single-panel figure did not already have, and every bar still
comes out of `draw_panel`. What is new is only the page they are drawn on.

Panels share their legend, their category set and their energy unit, so a
segment means the same thing in both. They do NOT share a y limit: two networks
-- or two accelerators -- an order of magnitude apart in size forced onto one
scale makes the smaller one unreadable, so each panel is scaled to itself and
says so on its own axis.
"""
from __future__ import annotations

import math

from . import style
from ..study.common import Session
from .stacked import BAR_WIDTH, active_categories, draw_panel, write_table
from .style import plt
from ..settings import guards


def stacked_panels(cfg, results, panels, title, stem, group_fontsize=17,
                   show_savings=True, panel_note=None, bars=None, bar_tags=None,
                   bar_width=None, ref_totals=None, bar_notes=None,
                   extra_columns=None, ylabel=None):
    """Draw and save one figure of stacked panels, plus its combined CSV.

    panels  ordered [(panel key, panel title, groups, stacks, labels)], top
            panel first. `stacks` is {group key: DataFrame} exactly as
            `grouped_stacks` takes it.

    `bars`, `bar_tags` and `bar_width` are `draw_panel`'s and apply to every
    panel. `ref_totals`, `bar_notes` and `extra_columns` are PER PANEL --
    {panel key: {group: ...}} -- so the placement figure's two panels cannot
    borrow each other's reference bar or per-bar note.
    """
    style.apply_rc()
    pal = style.palette(cfg)
    rows = len(panels)
    n = max(len(p[2]) for p in panels)
    cols = list(bars or cfg.bar_arms)

    # ONE unit across the whole figure: the panels are meant to be read against
    # each other, and two panels labelled in different units invite exactly the
    # comparison error the figure exists to prevent.
    max_pj = max(float(stacks[g][a].sum())
                 for _k, _t, groups, stacks, _l in panels
                 for g in groups for a in cols)
    div, unit = style.unit_for(max_pj)
    active = active_categories(cfg, [p[3] for p in panels])

    # A panelled figure is wide enough for one legend row, and one row keeps
    # the shared legend visibly attached to BOTH panels rather than looking
    # like a caption on the top one.
    ncol = min(len(active), 5)
    legend_rows = math.ceil(len(active) / ncol)

    # ---- the vertical budget, measured rather than tuned -------------------
    # The placement figure's heading is five lines (the fixed point, the mapping
    # note, the DRAM model, the MAC denominator, the panel note) where the sweep
    # figure's is one, and a fixed `top` put the legend inside it. So the space
    # above the panels is computed from the text that actually goes there: line
    # count x point size x leading, converted to a figure fraction.
    full_title = title if not panel_note else f"{title}\n{panel_note}"
    title_pt, legend_pt, panel_pt, panel_pad = 21, 17, 24, 16
    note_rows = 1.2 if bar_notes else 0.0
    fig_h = (8.2 + note_rows) * rows + 2.2
    title_in = (full_title.count("\n") + 1) * title_pt * 1.42 / 72.0
    legend_in = legend_rows * legend_pt * 2.0 / 72.0
    # `top` is where the AXES start, and each panel's own heading is drawn ABOVE
    # its axes (`set_title`, so title height + pad), which is why that height
    # has to be in the budget as well: without it the last legend row and the
    # first panel's heading land on each other.
    panel_in = (panel_pt * 1.45 + panel_pad) / 72.0
    top = 1.0 - (title_in + legend_in + panel_in + 0.30) / fig_h

    fig, axes = plt.subplots(rows, 1, figsize=(1.5 * len(cols) * n + 4, fig_h))
    axes = [axes] if rows == 1 else list(axes)

    for ax, (key, panel_title, groups, stacks, labels) in zip(axes, panels):
        draw_panel(ax, cfg, groups, stacks, labels, pal=pal, active=active,
                   div=div, group_fontsize=group_fontsize,
                   show_savings=show_savings, bars=bars, bar_tags=bar_tags,
                   bar_width=bar_width if bar_width is not None else BAR_WIDTH,
                   ref_totals=(ref_totals or {}).get(key),
                   bar_notes=(bar_notes or {}).get(key))
        ax.set_ylabel(ylabel or f"Inference energy ({unit})", fontsize=22,
                      labelpad=10)
        # The panel's own heading sits inside the axes rather than above it:
        # the space above each panel belongs to the figure legend and to the
        # bar totals, and a per-axes title there collides with both.
        ax.set_title(panel_title, fontsize=panel_pt, pad=panel_pad,
                     fontweight="medium")

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", frameon=False,
               fontsize=legend_pt, ncol=ncol,
               bbox_to_anchor=(0.5, 1.0 - (title_in + 0.12) / fig_h),
               handlelength=1.2, columnspacing=1.5, labelspacing=0.4)

    fig.suptitle(full_title, fontsize=title_pt, y=1.0 - 0.10 / fig_h,
                 va="top", fontweight="medium")

    # bottom and hspace: the group labels hang 27% of an axes height below zero
    # (see draw_panel), so every panel needs that much clear space under it and
    # the gap between panels has to clear it too -- but not by much more, or two
    # tall axes end up half empty with a band of white between them.
    fig.subplots_adjust(top=top, bottom=1.35 / fig_h,
                        left=0.075, right=0.99, hspace=0.42)

    figs = style.save(fig, results, stem)
    # The table gets EVERY group of every panel, drawn or not, with the panel in
    # the row key -- see `write_table`, which prefers a `"<panel>/<group>"` key
    # in `ref_totals` and `extra_columns` precisely so two panels with a group
    # called `recon1` do not share one row's reference.
    csv = write_table(cfg, results,
                      [(k, groups, stacks, labels)
                       for k, _t, groups, stacks, labels in panels], stem,
                      bars=bars, ref_totals=ref_totals,
                      extra_columns=extra_columns)
    return figs, csv

# ===========================================================================
#  the driver: one panel per model, the swept axis repeated in each
# ===========================================================================
# `ECC_EXPERIMENT=panels` with `ECC_PANEL_MODELS="resnet18 mobilenet_v2"`
# draws resnet18's architecture sweep on top and mobilenet_v2's underneath,
# in one figure, so the two can be read against each other.
#
# WHY THIS IS NOT A FOURTH SWEEP. The x axis is still `ECC_SWEEP`'s axis and
# the bars are still `stacked.draw_panel`; the models are a page layout, not
# an axis. The rule it does obey is the one that matters: it writes ONE
# figure, under one deterministic name, with its table and its manifest
# beside it -- and that name carries the panel models, so it can never
# overwrite `ArchitectureSweep.png`.
#
# The `bch` axis is refused here on purpose: a BCH panel per model would vary
# BOTH the model and the code between panels, which is two axes at once and
# not a comparison anyone can read.
#
# It was `study/panels.py` until ProjectRestructure phase 3: a driver that
# draws is L5, and the layer rule has no exception left for it.

def _panel_for(cfg, ses, model):
    """(groups, stacks, labels) for one model's panel, on the swept axis."""
    if cfg.sweep == "arch":
        groups = [a for a in cfg.archs
                  if a in ses.stacks and model in ses.stacks[a]]
        stacks = {a: ses.stacks[a][model] for a in groups}
        return groups, stacks, {a: cfg.arch_label(a) for a in groups}

    # sweep == "model" is a degenerate panel (one group), but it is what a user
    # asking for "these models, one panel each, on the held architecture" means,
    # so it is allowed rather than silently reinterpreted.
    arch = cfg.const_arch
    raw = ses.raws.get(arch, {}).get(model)
    if raw is None:
        return [], {}, {}
    return [model], {model: ses.stacks[arch][model]}, {model: model}


def run(cfg):
    ses = Session(cfg).setup()
    ses.collect_all()

    panels = []
    for model in cfg.panel_models:
        groups, stacks, labels = _panel_for(cfg, ses, model)
        if not groups:
            print(f"  [skip] {model}: nothing collected for it")
            continue
        panels.append((model, model, groups, stacks, labels))

    if not panels:
        raise guards.refusal("panels-nothing-to-plot",
            "nothing to plot: no panel model produced any group.\n"
            "  -> check the [skip] lines above; with ECC_REPLOT_ONLY=1 every "
            "point of every panel must already be in results/_raw/")

    for key, _title, groups, stacks, labels in panels:
        print(f"\n===== panel: {key} =====")
        ses.report(groups, stacks, labels)

    note = None
    if len(panels) > 1:
        note = ("each panel is scaled to itself — compare the SHAPE of the two, "
                "not their bar heights")

    figs, csv = stacked_panels(
        cfg, ses.results,
        panels=panels,
        title=cfg.panel_title(),
        stem=cfg.stem,
        panel_note=note)

    ses.finish(figs, csv, [g for _k, _t, groups, _s, _l in panels for g in groups])
    return ses
