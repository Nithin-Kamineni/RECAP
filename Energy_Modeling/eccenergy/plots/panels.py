"""One figure, one panel per model, the same architecture sweep in each.

The three sweeps each answer "does the ECC saving depend on THIS axis?" for one
axis at a time. This answers a question none of them can: *does the
architecture ranking itself hold from one network to the next?* Two networks
side by side in one image is the only way to read that off, because the claim
is about the SHAPE of the two panels, not about either one alone.

It is deliberately NOT a fourth sweep. There is no new x axis and no second
renderer: the x axis is still the architecture sweep, and every bar is drawn by
`stacked.draw_panel`, the same routine `grouped_stacks` uses. What is new is
only the page it is drawn on.

Panels share their legend, their category set and their energy unit, so a
segment means the same thing in both. They do NOT share a y limit: two networks
an order of magnitude apart in size forced onto one scale makes the smaller one
unreadable, so each panel is scaled to itself and says so on its own axis.
"""
from __future__ import annotations

import math

from . import style
from .stacked import active_categories, draw_panel, write_table
from .style import plt


def stacked_panels(cfg, results, panels, title, stem, group_fontsize=17,
                   show_savings=True, panel_note=None):
    """Draw and save one figure of stacked panels, plus its combined CSV.

    panels  ordered [(panel key, panel title, groups, stacks, labels)], top
            panel first. `stacks` is {group key: DataFrame} exactly as
            `grouped_stacks` takes it.
    """
    style.apply_rc()
    pal = style.palette(cfg)
    rows = len(panels)
    n = max(len(p[2]) for p in panels)

    # ONE unit across the whole figure: the panels are meant to be read against
    # each other, and two panels labelled in different units invite exactly the
    # comparison error the figure exists to prevent.
    max_pj = max(float(stacks[g][a].sum())
                 for _k, _t, groups, stacks, _l in panels
                 for g in groups for a in cfg.approaches)
    div, unit = style.unit_for(max_pj)
    active = active_categories(cfg, [p[3] for p in panels])

    # A panelled figure is wide enough for one legend row, and one row keeps
    # the shared legend visibly attached to BOTH panels rather than looking
    # like a caption on the top one.
    ncol = min(len(active), 5)
    legend_rows = math.ceil(len(active) / ncol)

    fig, axes = plt.subplots(
        rows, 1,
        figsize=(1.5 * len(cfg.approaches) * n + 4, 8.2 * rows + 2.2))
    axes = [axes] if rows == 1 else list(axes)

    for ax, (_key, panel_title, groups, stacks, labels) in zip(axes, panels):
        draw_panel(ax, cfg, groups, stacks, labels, pal=pal, active=active,
                   div=div, group_fontsize=group_fontsize,
                   show_savings=show_savings)
        ax.set_ylabel(f"Inference energy ({unit})", fontsize=22, labelpad=10)
        # The panel's own heading sits inside the axes rather than above it:
        # the space above each panel belongs to the figure legend and to the
        # bar totals, and a per-axes title there collides with both.
        ax.set_title(panel_title, fontsize=24, pad=16, fontweight="medium")

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", frameon=False, fontsize=17,
               ncol=ncol, bbox_to_anchor=(0.5, 0.972), handlelength=1.2,
               columnspacing=1.5, labelspacing=0.4)

    full_title = title if not panel_note else f"{title}\n{panel_note}"
    fig.suptitle(full_title, fontsize=21, y=0.995, fontweight="medium")

    # bottom: the group labels hang 27% of an axes height below zero (see
    # draw_panel), so every panel needs that much clear space under it.
    fig.subplots_adjust(top=0.895 - 0.012 * legend_rows, bottom=0.085,
                        left=0.075, right=0.99, hspace=0.62)

    figs = style.save(fig, results, stem)
    csv = write_table(cfg, results,
                      [(k, groups, stacks, labels)
                       for k, _t, groups, stacks, labels in panels], stem)
    return figs, csv
