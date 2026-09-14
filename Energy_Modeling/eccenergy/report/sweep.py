"""The one experiment: walk a single axis, hold the other two, draw one figure.

All three ECC arms are always drawn, so the arms are never the axis. What the x
axis walks is `ECC_SWEEP`:

    bch    BCH(N, K) at fixed N, weak code -> strong code
    model  the networks in ECC_SWEEP_MODELS, on one architecture
    arch   the accelerators in ECC_SWEEP_ARCHS, for one network

Whichever it is, exactly one grouped stacked-bar figure comes out, named after
the sweep, plus its CSV and a manifest recording the constants that produced it.
Only the BCH sweep needs the ECC arithmetic redone per group -- the other two
vary the raw energies instead, which is why this is one function.

It was `study/sweep.py` until ProjectRestructure phase 3: a driver that draws is
L5, and the layer rule has no exception left for it.
"""
from __future__ import annotations

from .stacked import grouped_stacks
from ..study import metrics as metrics_mod
from ..study.common import Session
from ..study.stacks import build_stacks, k_label, recon_pj_for_k
from ..settings import guards


def _bch_groups(cfg, ses):
    """x = code strength. One raw record, re-costed at each K."""
    arch, model = cfg.const_arch, cfg.const_model
    raw = ses.raws.get(arch, {}).get(model)
    if raw is None:
        return [], {}, {}, 15
    stacks = {k: build_stacks(cfg, raw, recon_pj_for_k(cfg, k), code_k=k)
              for k in cfg.sweep_ks}
    labels = {k: k_label(cfg, k) for k in cfg.sweep_ks}
    return list(cfg.sweep_ks), stacks, labels, 15


def _model_groups(cfg, ses):
    """x = network, on the one held architecture."""
    stacks = ses.stacks.get(cfg.const_arch, {})
    groups = [m for m in cfg.models if m in stacks]
    return groups, stacks, {m: m for m in groups}, 17


def _arch_groups(cfg, ses):
    """x = accelerator, for the one held network."""
    model = cfg.const_model
    groups = [a for a in cfg.archs if a in ses.stacks and model in ses.stacks[a]]
    stacks = {a: ses.stacks[a][model] for a in groups}
    return groups, stacks, {a: cfg.arch_label(a) for a in groups}, 17


BUILDERS = {"bch": _bch_groups, "model": _model_groups, "arch": _arch_groups}


def _raw_for_group(cfg, ses, group):
    """The `Raw` record ONE group of the swept axis was built from.

    Which of the three axes is walking decides what a group key IS -- a code, a
    network, or a design -- so the record behind it is looked up differently in
    each case, and the metric rows have to read the same record the energy
    stack was built from or the two rows would describe different runs.
    """
    if cfg.sweep == "bch":
        return ses.raws.get(cfg.const_arch, {}).get(cfg.const_model), cfg.const_arch, group
    if cfg.sweep == "model":
        return ses.raws.get(cfg.const_arch, {}).get(group), cfg.const_arch, cfg.code_k
    return ses.raws.get(group, {}).get(cfg.const_model), group, cfg.code_k


def metric_rows(cfg, ses, groups, stacks, labels):
    """`[(metric, groups, stacks, labels)]` -- one entry per `ECC_METRICS` row.

    THE ENERGY ROW IS THE ONE THAT WAS ALREADY BUILT, passed straight through,
    so the figure's energy row and the CSV beside it cannot disagree. Every
    other row is derived from the SAME `Raw` record, per group, by
    `study.metrics` -- nothing is re-mapped and nothing is re-timed twice.

    Returns a single-entry list for the default `ECC_METRICS=energy`, which is
    what keeps `grouped_stacks` on its unchanged single-panel path.
    """
    rows = []
    for metric in cfg.metrics:
        if metric == "energy":
            rows.append((metric, groups, stacks, labels))
            continue
        per_group = {}
        for g in groups:
            raw, arch, k = _raw_for_group(cfg, ses, g)
            if raw is None:
                continue
            cache = ses.chip_dir(arch)
            df, _prov = metrics_mod.metric_stacks(
                cfg, metric, raw, stacks[g], list(cfg.bar_arms),
                code_k=k if cfg.sweep == "bch" else None,
                # EVERY ABSTRACT ARM IS BILLED FROM THE REFERENCE PLAN on the
                # three sweeps: `build_stacks()` applies the reconstruction
                # arm's K/N to the reference record rather than remapping, so
                # the accelerator all three bars stand on is one chip and one
                # ART. The recon bar differs by the ENGINE, which is the term
                # the DC scrape adds. A bar per MAPPED placement, each with its
                # own chip and its own ART, is the placement figure's job.
                cache_dirs={a: cache for a in cfg.bar_arms})
            per_group[g] = df
        drawn = [g for g in groups if g in per_group]
        if not drawn:
            print(f"  [skip] metric {metric}: no group could be built")
            continue
        rows.append((metric, drawn, per_group,
                     {g: labels.get(g, g) for g in drawn}))
    return rows


def run(cfg):
    ses = Session(cfg).setup()
    ses.collect_all()

    groups, stacks, labels, fontsize = BUILDERS[cfg.sweep](cfg, ses)
    if not groups:
        raise guards.refusal("sweep-nothing-to-plot",
            f"nothing to plot: the {cfg.sweep} sweep produced no groups.\n"
            f"  -> check the [skip] lines above; with ECC_REPLOT_ONLY=1 every "
            f"point must already be in results/_raw/")
    if len(groups) < 2:
        print(f"  [note] only one group ({groups[0]}) -- the figure is still "
              f"written, but a sweep of one is just a single measurement")

    ses.report(groups, stacks, labels)

    figs, csv = grouped_stacks(
        cfg, ses.results,
        groups=groups,
        stacks=stacks,
        group_labels=labels,
        title=cfg.figure_title(),
        stem=cfg.stem,
        group_fontsize=fontsize,
        metric_rows=metric_rows(cfg, ses, groups, stacks, labels))

    ses.finish(figs, csv, groups)
    return ses
