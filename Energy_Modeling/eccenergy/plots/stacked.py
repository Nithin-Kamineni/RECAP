"""The grouped stacked-bar figure -- the one chart every experiment draws.

One bar group per "group key" -- models, architectures, or K values, whichever
axis `ECC_SWEEP` walks -- and one stacked bar per ECC approach inside each
group. That single renderer is what lets the whole study be one thin driver
instead of three 500-line scripts.

`draw_panel` IS that renderer, and it draws into an axes handed to it.
`grouped_stacks` is the single-panel figure the three sweeps produce;
`panels.py` stacks the SAME routine once per model. Neither owns a second way
of drawing a bar, which is the point -- a change to the bars changes every
figure this project makes.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import APPROACH_TAGS
from ..energy import plot_cats
from . import style
from .style import plt

#: Bar geometry, shared by every panel so two figures are always comparable.
BAR_WIDTH, BAR_GAP = 0.34, 0.06


def active_categories(cfg, panel_stacks):
    """Categories non-zero SOMEWHERE across every panel handed in.

    A category that is zero everywhere is dropped from the stacks AND the
    legend -- that is how "decode charged as zero" disappears cleanly. It is
    computed across all panels at once so a two-model figure cannot end up with
    two different legends.

    `panel_stacks` is a list of {group key: DataFrame}.
    """
    cats = plot_cats(cfg)
    return [c for c in cats
            if sum(float(st.loc[c].sum()) for stacks in panel_stacks
                   for st in stacks.values()) > 1e-9]


def _bar_x(cfg, n, bars=None, width=BAR_WIDTH):
    """Group centres, and the per-bar offset within a group."""
    bars = list(bars or cfg.approaches)
    span = width + BAR_GAP
    centers = np.arange(n) * (len(bars) * span + 1.4)
    first = -(len(bars) - 1) / 2.0
    return centers, {a: (first + i) * span for i, a in enumerate(bars)}


def draw_panel(ax, cfg, groups, stacks, group_labels, *, pal, active, div,
               group_fontsize=20, show_savings=True, bars=None, bar_tags=None,
               bar_width=BAR_WIDTH, ref_totals=None, bar_notes=None):
    """Draw one grouped stacked-bar panel into `ax`; return its data ymax.

    Every panel of one figure is passed the SAME `active` and `div`, so they
    share a legend and a unit. Each panel still keeps its own y limit: two
    networks of very different size forced onto one scale is a worse figure
    than two axes that each say what they are.

    THE FOUR PARAMETERS THAT ARE NOT ABOUT THE THREE SWEEPS. This is the only
    routine in the project that draws a bar, so the reconstruction-placement
    figure widens it rather than forking it:

    `bars`        the DataFrame columns to draw per group, and their left-to-
                  right order. Defaults to `cfg.approaches`, which is what the
                  three sweeps and the panel figure pass.
    `bar_tags`    the small rotated label under each bar. Defaults to the ECC
                  approach tags; pass `{}` when the group label already names
                  the bar, as it does when there is one bar per group.
    `bar_width`   wider bars read better when a group holds only one of them.
    `ref_totals`  {group key: pJ} for the savings annotation. Defaults to the
                  first bar of the same group, which is right when the arms sit
                  side by side inside a group and wrong when each bar is its own
                  group -- the placement figure passes the reference bar's total.

    ONE MORE, ADDED SO A SUB-PERCENT RESULT CAN BE READ OFF THE FIGURE.

    `bar_notes`   {group key: one line of text} printed above the saving, in the
                  figure's own energy unit. A percentage cannot separate 0.096 %
                  from 0.150 %, and on the placement study those two numbers ARE
                  the result -- the reduction really is applied, it is just 4.6
                  and 7.2 uJ on a 4,798 uJ bar. The caller supplies the wording
                  so this routine stays generic; it is meant for figures with
                  ONE bar per group, and is drawn once per group.
    """
    approaches = list(bars or cfg.approaches)
    tags = APPROACH_TAGS if bar_tags is None else bar_tags
    n = len(groups)
    centers, xoff = _bar_x(cfg, n, approaches, bar_width)

    bottoms = {a: np.zeros(n) for a in approaches}
    for cat in active:
        for a in approaches:
            vals = np.array([float(stacks[g].loc[cat, a]) for g in groups]) / div
            ax.bar(centers + xoff[a], vals, bar_width, bottom=bottoms[a],
                   color=pal[cat], edgecolor="white", linewidth=1.0, zorder=3,
                   label=style.label(cat) if a == approaches[0] else None)
            bottoms[a] += vals

    ymax = max(b.max() for b in bottoms.values())
    dec = style.decimals(ymax)

    for i, g in enumerate(groups):
        ref = ((ref_totals[g] / div) if ref_totals is not None
               else bottoms[approaches[0]][i])
        for a in approaches:
            total = bottoms[a][i]
            ax.text(centers[i] + xoff[a], total + ymax * 0.006, f"{total:.{dec}f}",
                    ha="center", va="bottom", fontsize=12, fontweight="bold",
                    color=style.INK)
        if show_savings:
            # With an EXTERNAL reference every bar is measured against it,
            # including the first -- the leftmost bar is only the reference when
            # the reference is "the first bar of this group". A bar that IS the
            # reference is left unannotated rather than labelled "-0.0%".
            annotate = approaches if ref_totals is not None else approaches[1:]
            for a in annotate:
                pct = (ref - bottoms[a][i]) / ref * 100 if ref > 0 else 0.0
                if ref_totals is not None and abs(pct) < 1e-9:
                    continue
                # A bar that costs MORE than its reference is a result, and
                # `-{pct}` printed it as "--11.8%". The sign carries the
                # direction: a minus is energy saved, a plus is energy spent.
                sign = "\u2212" if pct >= 0 else "+"
                ax.text(centers[i] + xoff[a], bottoms[a][i] + ymax * 0.045,
                        f"{sign}{abs(pct):.1f}%", ha="center", va="bottom",
                        fontsize=13, fontweight="bold", color="#B03A2E")
        note = (bar_notes or {}).get(g)
        if note:
            ax.text(centers[i], max(bottoms[a][i] for a in approaches)
                    + ymax * 0.088, note, ha="center", va="bottom", fontsize=12,
                    linespacing=1.35, color="#444")

    ax.set_xticks([])
    for i, g in enumerate(groups):
        for a in approaches:
            if not tags.get(a):
                continue
            ax.text(centers[i] + xoff[a], -ymax * 0.015, tags[a],
                    ha="center", va="top", rotation=90, fontsize=15,
                    color="#555", clip_on=False)
        ax.text(centers[i], -ymax * 0.27, group_labels.get(g, g),
                ha="center", va="top", fontsize=group_fontsize,
                fontweight="medium", color="#111", clip_on=False)

    # A note of up to two lines fits under the 1.34 headroom the placement
    # figure has always had; the DRAM-interface line added 2026-09-09 makes it
    # three, so the headroom grows with the longest note rather than the note
    # colliding with the title.
    note_lines = max((str(n).count("\n") + 1 for n in (bar_notes or {}).values()),
                     default=0)
    ax.set_ylim(0, ymax * (1.22 if not bar_notes
                           else 1.34 + 0.06 * max(0, note_lines - 2)))
    ax.set_xlim(centers[0] - 1.4, centers[-1] + 1.4)
    ax.tick_params(axis="y", labelsize=20, width=2.0, length=9)
    ax.grid(axis="y", ls=":", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ymax


def grouped_stacks(cfg, results, groups, stacks, group_labels, title, stem,
                   group_fontsize=20, show_savings=True, bars=None,
                   bar_tags=None, bar_width=BAR_WIDTH, ref_totals=None,
                   ylabel=None, extra_columns=None, bar_notes=None):
    """Draw and save one grouped stacked-bar figure plus its CSV.

    groups        ordered list of group keys
    stacks        {group key: DataFrame(rows=categories, cols=bars)}
    group_labels  {group key: text under the group}

    `bars`, `bar_tags`, `bar_width`, `ref_totals` and `bar_notes` are passed
    straight to `draw_panel` -- see its docstring for what the placement figure
    does with them.

    ONE PANEL. The placement figure briefly carried a second one below, the same
    bars rescaled to the on-chip weight path. It was removed: its bands were the
    weight share of categories this panel already draws, so it restated the
    figure above it at a different scale rather than adding a quantity. What it
    was there to make visible -- a sub-percent saving -- is carried by
    `bar_notes`, in uJ, on the bar it belongs to.
    """
    style.apply_rc()
    pal = style.palette(cfg)
    n = len(groups)
    cols = list(bars or cfg.approaches)

    max_pj = max(float(stacks[g][a].sum()) for g in groups for a in cols)
    div, unit = style.unit_for(max_pj)
    active = active_categories(cfg, [stacks])

    fig, ax = plt.subplots(figsize=(1.5 * len(cols) * n + 4, 10))
    draw_panel(ax, cfg, groups, stacks, group_labels, pal=pal, active=active,
               div=div, group_fontsize=group_fontsize, show_savings=show_savings,
               bars=bars, bar_tags=bar_tags, bar_width=bar_width,
               ref_totals=ref_totals, bar_notes=bar_notes)

    ncol = min(len(active), 4)
    legend_rows = math.ceil(len(active) / ncol)
    ax.set_ylabel(ylabel or f"Inference energy ({unit})", fontsize=28, labelpad=12)
    ax.legend(loc="upper center", frameon=False, fontsize=17, ncol=ncol,
              bbox_to_anchor=(0.5, 1.03 + 0.055 * legend_rows),
              handlelength=1.2, columnspacing=1.5, labelspacing=0.4)
    ax.set_title(title, fontsize=22, pad=34 + 30 * legend_rows, fontweight="medium")
    fig.subplots_adjust(bottom=0.30, top=0.83, left=0.07, right=0.99)

    figs = style.save(fig, results, stem)
    csv = write_table(cfg, results, [(None, groups, stacks, group_labels)], stem,
                      bars=bars, ref_totals=ref_totals, extra_columns=extra_columns)
    return figs, csv


def write_table(cfg, results, panels, stem, bars=None, ref_totals=None,
                extra_columns=None):
    """Per-panel, per-group, per-approach, per-category energies in uJ.

    `panels` is [(panel key, groups, stacks, labels)]. A single-panel figure
    passes `None` as the key and the CSV is exactly the one this project always
    wrote; a two-model figure gets a `panel` column and a panel-qualified index,
    so both models live in ONE table instead of two files a reader has to join
    by eye.

    `ref_totals` is the same {group: pJ} `draw_panel` takes, and it matters for
    the same reason: with ONE bar per group the saving column would otherwise
    measure each bar against itself and print 0.0 for everything.
    `extra_columns` is {group: {column: value}}, appended per row -- the
    placement study puts its reconstruction counts, overheads and feasibility
    there, so its figure and its table carry the same facts and there is still
    exactly one table per stem.

    BOTH may be keyed `"<panel>/<group>"` as well as `"<group>"`, and the
    panel-qualified key wins. Two panels of a multi-architecture placement
    figure both have a group called `recon1`, and a bare-key lookup would give
    the second panel the first panel's reference total and reconstruction
    counts -- silently, and only in the table.
    """
    def _pick(d, panel_key, g):
        if not d:
            return None
        if panel_key is not None and f"{panel_key}/{g}" in d:
            return d[f"{panel_key}/{g}"]
        return d.get(g)

    cats = plot_cats(cfg)
    cols = list(bars or cfg.approaches)
    rows = {}
    for panel_key, groups, stacks, labels in panels:
        for g in groups:
            st = stacks[g]
            row = {}
            if panel_key is not None:
                row["panel"] = panel_key
            row["label"] = labels.get(g, g)
            for c in cats:
                for a in cols:
                    row[f"{a}_{c}"] = float(st.loc[c, a]) / 1e6
            ref = (_pick(ref_totals, panel_key, g) if ref_totals is not None
                   else float(st[cols[0]].sum()))
            ref = 0.0 if ref is None else float(ref)
            for a in cols:
                total = float(st[a].sum())
                if total <= 0.0:
                    # A GROUP WITH NO BAR IS NOT A BAR THAT SAVES EVERYTHING.
                    # The placement study gives an INFEASIBLE boundary a table
                    # row on purpose -- a table that omits it reads as "not
                    # considered" -- and its stack is all zeros because there is
                    # nothing to draw. Filling the derived columns from those
                    # zeros printed `total 0.0, saving 100.0%` next to
                    # `status unsupported` on the same line, which is a claim
                    # the row exists to deny. The per-category zeros stay: they
                    # are what "no bar" means to the figure. The study's own
                    # columns (`saving_vs_embedded_only_pct`, ...) were already
                    # left blank for these rows, so this makes the generic pair
                    # agree with them.
                    row[f"{a}_total_uJ"] = ""
                    row[f"{a}_saving_pct"] = ""
                    continue
                row[f"{a}_total_uJ"] = total / 1e6
                row[f"{a}_saving_pct"] = ((ref - total) / ref * 100) if ref > 0 else 0.0
            row.update(_pick(extra_columns, panel_key, g) or {})
            rows[g if panel_key is None else f"{panel_key}/{g}"] = row
    path = results.table_path(stem)
    # An explicit newline="" handle, exactly as Results.write_manifest uses.
    # This repo is LF-only (see .gitattributes), and a to_csv() straight to a
    # path opens in text mode, so a replot run from the Windows side rather
    # than in the container would write CRLF.
    with open(path, "w", newline="") as fh:
        pd.DataFrame(rows).T.to_csv(fh, index_label="group", lineterminator="\n")
    return path
