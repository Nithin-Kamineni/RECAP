"""Figure style: one palette, one set of labels, one save path.

Every figure in the project goes through `save()`, so DPI, formats, and the
results layout are decided once.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

#: The house palette used by the published figures.
HOUSE = {
    "DRAM": "#2E5A87",
    "Global buffer": "#4E9F3D",
    "Global buffer (read)": "#6FBF5A",
    "Global buffer (write)": "#2E5F24",
    "Local (spads/RF)": "#E1A730",
    "Local (read)": "#F0C05A",
    "Local (write)": "#A9781A",
    "NoC": "#7F8C8D",
    "Compute": "#B03A2E",
    "ECC decode": "#7D3C98",
    "Reconstruction": "#1F9E8F",
}

#: Okabe-Ito. The house palette's global-buffer green and on-chip amber are
#: adjacent stack segments at dE 7.9 under protanopia -- the marginal band --
#: and the amber falls below 3:1 contrast on white. Use this for print and for
#: accessibility-sensitive venues (ECC_PALETTE=cvd).
CVD = dict(HOUSE)
CVD.update({
    "DRAM": "#0072B2",
    "Global buffer": "#009E73",
    "Global buffer (read)": "#56B4E9",
    "Global buffer (write)": "#006D5B",
    "Local (spads/RF)": "#E69F00",
    "Local (read)": "#F0C05A",
    "Local (write)": "#A9781A",
    "NoC": "#999999",
    "Compute": "#D55E00",
    "ECC decode": "#7D3C98",
    "Reconstruction": "#CC79A7",
})

#: Category names as they should read in a legend.
NICE_CATEGORY = {
    "Local (spads/RF)": "On-chip SRAM/RF",
    "NoC": "NoC / interconnect",
    "Local (read)": "On-chip SRAM/RF (read)",
    "Local (write)": "On-chip SRAM/RF (write)",
}

INK = "#2b2b2b"


def palette(cfg):
    return CVD if cfg.palette == "cvd" else HOUSE


def apply_rc():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.linewidth": 2.0,
        "axes.edgecolor": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def label(category):
    return NICE_CATEGORY.get(category, category)


def unit_for(max_pj):
    """Pick pJ/nJ/uJ/mJ so the axis numbers stay two-to-four digits."""
    if max_pj / 1e6 >= 1000:
        return 1e9, "mJ"
    if max_pj / 1e6 >= 1:
        return 1e6, "µJ"
    if max_pj / 1e3 >= 1:
        return 1e3, "nJ"
    return 1.0, "pJ"


def decimals(ymax):
    """Enough precision that 15.9 and 15.3 do not both print as 16."""
    return 0 if ymax >= 100 else (1 if ymax >= 10 else 2)


def save(fig, results, stem):
    """Write every configured format and return the paths."""
    paths = results.figure_paths(stem)
    for p in paths:
        fig.savefig(p, dpi=results.cfg.dpi if p.suffix == ".png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    return paths
