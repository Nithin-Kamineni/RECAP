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
    "Standby": "#5D6D7E",
    "ECC decode": "#7D3C98",
    "Reconstruction": "#1F9E8F",
    "Recon overhead": "#8E6C3A",
    # ---- the metric rows that are not energy (EnvReorganisation phase 5).
    # `Latency` and `Energy x delay` are ONE segment each -- a run length is a
    # max over the levels, not a sum of them -- so their colour only has to be
    # distinct from the energy stack, never to sit beside another segment.
    # `Recon engine` DOES sit beside the area stack's accelerator segments, and
    # it takes `Reconstruction`'s teal on purpose: the same component charged
    # in a different currency should read as the same component.
    "Latency": "#34495E",
    "Energy x delay": "#5B4B8A",
    "Recon engine": "#1F9E8F",
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
    "Standby": "#6E7B8B",
    "ECC decode": "#7D3C98",
    "Reconstruction": "#CC79A7",
    "Recon overhead": "#8E6C3A",
    "Latency": "#34495E",
    "Energy x delay": "#5B4B8A",
    "Recon engine": "#CC79A7",
})

#: Category names as they should read in a legend.
NICE_CATEGORY = {
    "Standby": "Standby (leakage)",
    "Local (spads/RF)": "On-chip SRAM/RF",
    "Recon overhead": "Recon buffer/control",
    "NoC": "NoC / interconnect",
    "Local (read)": "On-chip SRAM/RF (read)",
    "Local (write)": "On-chip SRAM/RF (write)",
    "Recon engine": "Reconstruction engine (DC)",
}

INK = "#2b2b2b"

#: THE SAVINGS ANNOTATION, IN TWO COLOURS SINCE TASK D SESSION 2 (2026-09-14).
#: Every percentage above a bar used to print in ONE red -- `Compute`'s own
#: #B03A2E -- so a saving and a PENALTY differed by their sign glyph alone, and
#: a reader scanning the figure read "+15.4%" as another win. The sign is still
#: there and is still the fallback; the colour now carries the direction too.
#: A penalty keeps the red it always had, so no bar that costs more than its
#: reference changed colour; a saving is what moved.
#: The CVD pair is Okabe-Ito -- bluish green against vermillion, which separate
#: under protanopia and deuteranopia; the house pair is a conventional
#: green/red and does not, which is what `ECC_PALETTE=cvd` is for.
ANNOTATION = {"saving": "#1B7837", "penalty": "#B03A2E"}
ANNOTATION_CVD = {"saving": "#009E73", "penalty": "#D55E00"}


def annotation(cfg):
    """`{"saving": colour, "penalty": colour}` for this run's palette."""
    return ANNOTATION_CVD if cfg.palette == "cvd" else ANNOTATION


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


def unit_for_metric(metric, max_value):
    """`(divisor, unit)` for one metric's own quantity. NOT `unit_for`.

    `unit_for` picks a scale for PICOJOULES and is right for `energy` and for
    nothing else: handing it seconds would label a millisecond "µJ". Each
    metric row carries its own base unit out of `study.metrics.metric_stacks`
    -- pJ, pJ·s, seconds, µm² -- and this is the one place each is scaled for
    the axis, so two rows of one figure can never be divided by each other's
    divisor.
    """
    if metric == "energy":
        return unit_for(max_value)
    if metric == "edp":
        # pJ·s. 1 µJ·ms = 1e6 pJ × 1e-3 s = 1e3 pJ·s.
        return (1e3, "µJ·ms") if max_value >= 1e3 else (1.0, "pJ·s")
    if metric == "latency":
        # seconds. Off a 200 MHz clock an inference is milliseconds.
        if max_value >= 1.0:
            return 1.0, "s"
        return (1e-3, "ms") if max_value >= 1e-3 else (1e-6, "µs")
    if metric == "area":
        # µm² straight out of Accelergy's ART and the DC report. An
        # accelerator is millions of them; one engine is thousands.
        return (1e6, "mm²") if max_value >= 1e5 else (1.0, "µm²")
    raise ValueError(f"no unit for metric {metric!r}")


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
