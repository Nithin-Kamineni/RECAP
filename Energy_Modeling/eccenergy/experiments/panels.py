"""One image, one panel per model, the architecture sweep repeated in each.

`ECC_EXPERIMENT=panels` with `ECC_PANEL_MODELS="resnet18 mobilenet_v2"` draws
resnet18's architecture sweep on top and mobilenet_v2's underneath, in one
figure, so the two can be read against each other.

WHY THIS IS NOT A FOURTH SWEEP. The x axis is still `ECC_SWEEP`'s axis and the
bars are still `stacked.draw_panel`; the models are a page layout, not an axis.
The rule it does obey is the one that matters: it writes ONE figure, under one
deterministic name, with its table and its manifest beside it -- and that name
carries the panel models, so it can never overwrite `ArchitectureSweep.png`.

The `bch` axis is refused here on purpose: a BCH panel per model would vary
BOTH the model and the code between panels, which is two axes at once and not a
comparison anyone can read.
"""
from __future__ import annotations

from ..plots.panels import stacked_panels
from .common import Session


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
        raise SystemExit(
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
