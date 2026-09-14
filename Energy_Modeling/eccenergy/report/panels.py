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

THE THIRD CALLER, SINCE EnvReorganisation PHASE 5: `ECC_METRICS`. A panel may
now declare which METRIC it draws, and panels of different metrics share
nothing but the page -- not the unit (uJ against ms against um2), not the
legend, not the category set. `_metric_blocks()` below splits the panel list
into one CONTIGUOUS BLOCK per metric and computes the unit, the active
categories and the legend WITHIN each block, so a millisecond is never divided
by an energy figure's `div`.

WHY THE ROWS GREW HERE AND NOT IN `sweep.py` OR `recon_view.py` -- decided
2026-09-14, and recorded because the alternative was real. There are still TWO
figure programs (`sweep.py` + this file for bch/model/arch, `recon_view.py` for
the placement figure), and phase 6 unifies them. Teaching either one to stack
metric rows would have written the row logic twice and then had to merge it;
teaching BOTH through the routine they already share writes it once. This is
that routine: `recon_view.py` calls it directly for a multi-design figure, and
`stacked.grouped_stacks` -- which every other figure goes through -- now
DELEGATES to it as soon as more than one metric is asked for, keeping its own
single-panel path byte-identical for the one-metric case that is the default.
No renderer was added and `draw_panel` is still the only place a bar is drawn.

HOW METRIC ROWS COMPOSE WITH MODEL PANELS -- also decided 2026-09-14, because
`ECC_METRICS` x `ECC_PANEL_MODELS` is a grid and something had to say which way
it flattens. IT IS ONE COLUMN, METRIC-MAJOR: every panel of the first metric,
then every panel of the second. So `ECC_METRICS="energy latency"` with two
panel models is energy/resnet18, energy/mobilenet_v2, latency/resnet18,
latency/mobilenet_v2, top to bottom. Metric-major because EnvReorganisation 3.1
states the vertical order in metrics ("energy on top, latency below it, area
below that") and because the panels of one metric are the ones that share a
unit and a legend -- a true two-dimensional grid would put panels that share
nothing side by side and buy only page height. Each row's heading names both
the metric and the panel.
"""
from __future__ import annotations

import math

from . import style
from ..study import metrics as metrics_mod
from ..study.common import Session
from .stacked import (BAR_WIDTH, active_categories, draw_panel, panel_width,
                      write_table)
from .style import plt
from ..settings import guards


def _segments(panel_selection):
    """The stack rows a non-energy metric's DataFrames actually carry.

    Taken from the DataFrames rather than from a list kept here, so a metric
    that grows a segment does not have to be registered in two places. Order is
    the DataFrame's own -- `study.metrics` builds it deliberately (the area
    categories, then `Recon engine` on top).
    """
    seen = []
    for p in panel_selection:
        for st in p[3].values():
            for c in st.index:
                if c not in seen:
                    seen.append(c)
    return seen


def panel_metric(p):
    """The metric a panel tuple declares, or None for the 5-element form.

    A panel is `(key, title, groups, stacks, labels)` and, since
    EnvReorganisation phase 5, optionally `(..., metric)`. The short form is
    every caller that predates `ECC_METRICS` and means "one block, scaled and
    labelled exactly as this figure always was" -- which is what keeps the
    default one-metric figure byte-identical.
    """
    return p[5] if len(p) > 5 else None


def _metric_blocks(panels):
    """[(metric, [panel index, ...])] -- contiguous runs of one metric.

    METRIC-MAJOR, ONE COLUMN (see this module's docstring). Runs are contiguous
    rather than gathered, so the block structure is exactly the order the
    caller stacked the panels in and this routine never reorders the page.
    """
    blocks = []
    for i, p in enumerate(panels):
        m = panel_metric(p)
        if blocks and blocks[-1][0] == m:
            blocks[-1][1].append(i)
        else:
            blocks.append((m, [i]))
    return blocks


def stacked_panels(cfg, results, panels, title, stem, group_fontsize=17,
                   show_savings=True, panel_note=None, bars=None, bar_tags=None,
                   bar_width=None, ref_totals=None, bar_notes=None,
                   extra_columns=None, ylabel=None):
    """Draw and save one figure of stacked panels, plus its combined CSV.

    panels  ordered [(panel key, panel title, groups, stacks, labels)], top
            panel first, or the 6-element `(..., metric)` form since
            EnvReorganisation phase 5. `stacks` is {group key: DataFrame}
            exactly as `grouped_stacks` takes it.

    `bars`, `bar_tags` and `bar_width` are `draw_panel`'s and apply to every
    panel. `ref_totals`, `bar_notes` and `extra_columns` are PER PANEL --
    {panel key: {group: ...}} -- so the placement figure's two panels cannot
    borrow each other's reference bar or per-bar note.

    ONE BLOCK PER METRIC. Everything a figure normally shares -- the unit, the
    active category set, the legend -- is shared WITHIN a metric block and
    never across one. Two rows of one figure measuring milliseconds and square
    microns have no common divisor and no common legend, and computing either
    across the whole page is how a millisecond gets divided by an energy
    figure's `div`. With one block (every caller that names no metric, and the
    default `ECC_METRICS=energy`) the arithmetic below is exactly what it was.
    """
    style.apply_rc()
    pal = style.palette(cfg)
    rows = len(panels)
    n = max(len(p[2]) for p in panels)
    cols = list(bars or cfg.bar_arms)

    blocks = _metric_blocks(panels)
    one_block = len(blocks) == 1

    # PER BLOCK: its own unit and its own active category set. The panels of a
    # block are meant to be read against each other, and two panels of one
    # metric labelled in different units invite exactly the comparison error
    # the figure exists to prevent -- so the scale is still shared, just no
    # wider than the metric it belongs to.
    block_of, div_of, unit_of, active_of = {}, {}, {}, {}
    for metric, idxs in blocks:
        sel = [panels[i] for i in idxs]
        # A GROUP MAY BE MISSING A BAR (EnvReorganisation phase 6): designs do
        # not declare the same boundaries, so an arch sweep's groups answer
        # different bar sets. The block's unit is taken over what EXISTS.
        top = max(float(stacks[g][a].sum())
                  for _k, _t, groups, stacks, _l in
                  [(p[0], p[1], p[2], p[3], p[4]) for p in sel]
                  for g in groups for a in cols if a in stacks[g].columns)
        d, u = (style.unit_for(top) if metric is None
                else style.unit_for_metric(metric, top))
        act = active_categories(cfg, [p[3] for p in sel],
                                cats=None if metric in (None, "energy")
                                else _segments(sel))
        for i in idxs:
            block_of[i], div_of[i], unit_of[i], active_of[i] = metric, d, u, act

    # The legend is sized on the WIDEST block: with one block that is today's
    # arithmetic unchanged, and with several it is the one that needs the most
    # rows, so no block's legend is clipped by a budget computed for a smaller
    # one.
    active = max((active_of[i] for i in range(rows)), key=len)
    div, unit = div_of[0], unit_of[0]

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

    fig, axes = plt.subplots(rows, 1, figsize=(panel_width(len(cols), n), fig_h))
    axes = [axes] if rows == 1 else list(axes)

    #: The first panel of each block, so a multi-metric figure can hang that
    #: block's own legend on it -- one legend per metric, because two metrics
    #: share no category.
    block_head = {idxs[0] for _m, idxs in blocks}

    for i, (ax, panel) in enumerate(zip(axes, panels)):
        key, panel_title, groups, stacks, labels = panel[:5]
        draw_panel(ax, cfg, groups, stacks, labels, pal=pal,
                   active=active_of[i],
                   div=div_of[i], group_fontsize=group_fontsize,
                   show_savings=show_savings, bars=bars, bar_tags=bar_tags,
                   bar_width=bar_width if bar_width is not None else BAR_WIDTH,
                   ref_totals=(ref_totals or {}).get(key),
                   bar_notes=(bar_notes or {}).get(key))
        metric = block_of[i]
        if metric is None:
            ax.set_ylabel(ylabel or f"Inference energy ({unit_of[i]})",
                          fontsize=22, labelpad=10)
        else:
            # A metric row names its OWN quantity, whatever `ylabel` the caller
            # passed for the energy figure: a row of milliseconds labelled
            # "Inference energy" is the one mistake a multi-row figure can make
            # that a reader cannot catch.
            ax.set_ylabel(metrics_mod.metric_label(cfg, metric, unit_of[i]),
                          fontsize=22, labelpad=10)
        # The panel's own heading sits inside the axes rather than above it:
        # the space above each panel belongs to the figure legend and to the
        # bar totals, and a per-axes title there collides with both.
        ax.set_title(panel_title, fontsize=panel_pt, pad=panel_pad,
                     fontweight="medium")
        # ONE LEGEND PER METRIC BLOCK, on its first panel, when there is more
        # than one block. A single figure legend is right for one metric and
        # wrong for several: `Latency` and `DRAM` are not alternatives in one
        # key, and a reader given both would look for a DRAM segment in the
        # latency row.
        # A ONE-SEGMENT ROW NEEDS NO KEY. `Latency` and `Energy x delay` are a
        # single band each, and their legend says exactly what the y label
        # already says one inch to the left -- so it is not drawn, and the row
        # keeps the vertical space for the bars.
        if not one_block and i in block_head and len(active_of[i]) > 1:
            # ABOVE the panel heading, not on it. The heading is drawn at
            # `pad` points over the axes and a legend anchored just above 1.0
            # lands on exactly the same pixels -- measured on the first
            # four-row figure, 2026-09-14, where "Energy x delay" and its key
            # were printed on top of each other. The offset below clears the
            # heading's own height (`panel_pt` at 1.45 leading) and then the
            # legend's rows.
            h, l = ax.get_legend_handles_labels()
            rows_ = math.ceil(len(active_of[i]) / 5)
            head_frac = (panel_pt * 1.45 + panel_pad) / (72.0 * fig_h / rows)
            ax.legend(h, l, loc="lower center", frameon=False,
                      fontsize=legend_pt - 2,
                      ncol=min(len(active_of[i]), 5),
                      bbox_to_anchor=(0.5, 1.0 + head_frac),
                      handlelength=1.2, columnspacing=1.5, labelspacing=0.4)

    if one_block:
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
    # A METRIC BLOCK DRAWS ITS OWN LEGEND ABOVE ITS FIRST PANEL, into the space
    # the row above needs for its group labels -- which hang 27% of an axes
    # height below zero (see `draw_panel`). 0.42 clears the labels alone;
    # measured on the first four-row figure (2026-09-14) the area row's key
    # landed on the latency row's tick labels. So a multi-block page gets the
    # extra gap, and a single-block one is untouched.
    fig.subplots_adjust(top=top, bottom=1.35 / fig_h,
                        left=0.075, right=0.99,
                        hspace=0.42 if one_block else 0.66)

    figs = style.save(fig, results, stem)
    # The table gets EVERY group of every panel, drawn or not, with the panel in
    # the row key -- see `write_table`, which prefers a `"<panel>/<group>"` key
    # in `ref_totals` and `extra_columns` precisely so two panels with a group
    # called `recon1` do not share one row's reference.
    # ONE CSV PER STEM, AND IT IS THE ENERGY TABLE. `write_table` writes the
    # per-category energies every result of this project has carried, and a
    # metric row is not a category of it -- a `Latency` column of seconds
    # beside `recon_DRAM` picojoules in one row would be two units in one
    # record. So the table is written from the ENERGY block alone when there is
    # one, which keeps it byte-identical to what phase 4 left, and a
    # multi-metric figure's other rows live in the figure and in the manifest.
    # Carrying every metric in the table is phase 6's, with the sweep unify.
    energy_panels = [p for p in panels if panel_metric(p) in (None, "energy")]
    csv = write_table(cfg, results,
                      [(p[0], p[2], p[3], p[4])
                       for p in (energy_panels or panels)], stem,
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
