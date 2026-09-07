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


def _bar_x(cfg, n):
    """Group centres, and the per-approach offset within a group."""
    span = BAR_WIDTH + BAR_GAP
    centers = np.arange(n) * (len(cfg.approaches) * span + 1.4)
    first = -(len(cfg.approaches) - 1) / 2.0
    return centers, {a: (first + i) * span for i, a in enumerate(cfg.approaches)}


def draw_panel(ax, cfg, groups, stacks, group_labels, *, pal, active, div,
               group_fontsize=20, show_savings=True):
    """Draw one grouped stacked-bar panel into `ax`; return its data ymax.

    Every panel of one figure is passed the SAME `active` and `div`, so they
    share a legend and a unit. Each panel still keeps its own y limit: two
    networks of very different size forced onto one scale is a worse figure
    than two axes that each say what they are.
    """
    approaches = cfg.approaches
    n = len(groups)
    centers, xoff = _bar_x(cfg, n)

    bottoms = {a: np.zeros(n) for a in approaches}
    for cat in active:
        for a in approaches:
            vals = np.array([float(stacks[g].loc[cat, a]) for g in groups]) / div
            ax.bar(centers + xoff[a], vals, BAR_WIDTH, bottom=bottoms[a],
                   color=pal[cat], edgecolor="white", linewidth=1.0, zorder=3,
                   label=style.label(cat) if a == approaches[0] else None)
            bottoms[a] += vals

    ymax = max(b.max() for b in bottoms.values())
    dec = style.decimals(ymax)

    for i, g in enumerate(groups):
        ref = bottoms[approaches[0]][i]
        for a in approaches:
            total = bottoms[a][i]
            ax.text(centers[i] + xoff[a], total + ymax * 0.006, f"{total:.{dec}f}",
                    ha="center", va="bottom", fontsize=12, fontweight="bold",
                    color=style.INK)
        if show_savings:
            for a in approaches[1:]:
                pct = (ref - bottoms[a][i]) / ref * 100 if ref > 0 else 0.0
                ax.text(centers[i] + xoff[a], bottoms[a][i] + ymax * 0.045,
                        f"−{pct:.1f}%", ha="center", va="bottom",
                        fontsize=13, fontweight="bold", color="#B03A2E")

    ax.set_xticks([])
    for i, g in enumerate(groups):
        for a in approaches:
            ax.text(centers[i] + xoff[a], -ymax * 0.015, APPROACH_TAGS[a],
                    ha="center", va="top", rotation=90, fontsize=15,
                    color="#555", clip_on=False)
        ax.text(centers[i], -ymax * 0.27, group_labels.get(g, g), ha="center",
                va="top", fontsize=group_fontsize, fontweight="medium",
                color="#111", clip_on=False)

    ax.set_ylim(0, ymax * 1.22)
    ax.set_xlim(centers[0] - 1.4, centers[-1] + 1.4)
    ax.tick_params(axis="y", labelsize=20, width=2.0, length=9)
    ax.grid(axis="y", ls=":", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ymax


def grouped_stacks(cfg, results, groups, stacks, group_labels, title, stem,
                   group_fontsize=20, show_savings=True):
    """Draw and save one grouped stacked-bar figure plus its CSV.

    groups        ordered list of group keys
    stacks        {group key: DataFrame(rows=categories, cols=approaches)}
    group_labels  {group key: text under the group}
    """
    style.apply_rc()
    pal = style.palette(cfg)
    n = len(groups)

    max_pj = max(float(stacks[g][a].sum()) for g in groups for a in cfg.approaches)
    div, unit = style.unit_for(max_pj)
    active = active_categories(cfg, [stacks])

    fig, ax = plt.subplots(figsize=(1.5 * len(cfg.approaches) * n + 4, 10))
    draw_panel(ax, cfg, groups, stacks, group_labels, pal=pal, active=active,
               div=div, group_fontsize=group_fontsize, show_savings=show_savings)

    ncol = min(len(active), 4)
    legend_rows = math.ceil(len(active) / ncol)
    ax.set_ylabel(f"Inference energy ({unit})", fontsize=28, labelpad=12)
    ax.legend(loc="upper center", frameon=False, fontsize=17, ncol=ncol,
              bbox_to_anchor=(0.5, 1.03 + 0.055 * legend_rows),
              handlelength=1.2, columnspacing=1.5, labelspacing=0.4)
    ax.set_title(title, fontsize=22, pad=34 + 30 * legend_rows, fontweight="medium")
    fig.subplots_adjust(bottom=0.30, top=0.83, left=0.07, right=0.99)

    figs = style.save(fig, results, stem)
    csv = write_table(cfg, results, [(None, groups, stacks, group_labels)], stem)
    return figs, csv


def write_table(cfg, results, panels, stem):
    """Per-panel, per-group, per-approach, per-category energies in uJ.

    `panels` is [(panel key, groups, stacks, labels)]. A single-panel figure
    passes `None` as the key and the CSV is exactly the one this project always
    wrote; a two-model figure gets a `panel` column and a panel-qualified index,
    so both models live in ONE table instead of two files a reader has to join
    by eye.
    """
    cats = plot_cats(cfg)
    rows = {}
    for panel_key, groups, stacks, labels in panels:
        for g in groups:
            st = stacks[g]
            row = {}
            if panel_key is not None:
                row["panel"] = panel_key
            row["label"] = labels.get(g, g)
            for c in cats:
                for a in cfg.approaches:
                    row[f"{a}_{c}"] = float(st.loc[c, a]) / 1e6
            ref = float(st[cfg.approaches[0]].sum())
            for a in cfg.approaches:
                total = float(st[a].sum())
                row[f"{a}_total_uJ"] = total / 1e6
                row[f"{a}_saving_pct"] = ((ref - total) / ref * 100) if ref > 0 else 0.0
            rows[g if panel_key is None else f"{panel_key}/{g}"] = row
    path = results.table_path(stem)
    # An explicit newline="" handle, exactly as Results.write_manifest uses.
    # This repo is LF-only (see .gitattributes), and a to_csv() straight to a
    # path opens in text mode, so a replot run from the Windows side rather
    # than in the container would write CRLF.
    with open(path, "w", newline="") as fh:
        pd.DataFrame(rows).T.to_csv(fh, index_label="group", lineterminator="\n")
    return path
