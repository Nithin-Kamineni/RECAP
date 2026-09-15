"""The one experiment: walk a single axis, hold the other two, draw one figure.

What the x axis walks is `ECC_SWEEP`:

    bch    BCH(N, K) at fixed N, weak code -> strong code
    model  the networks in ECC_SWEEP_MODELS, on one architecture
    arch   the accelerators in ECC_SWEEP_ARCHS, for one network
    fix    NO x axis -- ONE group, at the first entry of all three lists

Whichever it is, exactly one grouped stacked-bar figure comes out, named after
the sweep, plus its CSV and a manifest recording the constants that produced it.

EVERY BAR IS A REAL, MAPPED CHIP (EnvReorganisation phase 6, 2026-09-14).
Until then this module drew three bars per group -- `baseline`, `embedded` and
an ABSTRACT `recon` that `study.stacks.build_stacks()` synthesised by applying
K/N to every on-chip level of every design at once. No physical boundary does
that, so that arm is RETIRED (plan 6.2, answer 9.2) and the bars are now:

    baseline, embedded   the two reference arms, `build_stacks()`, one plan
    recon1 .. recon5     the boundaries the DESIGN declares, each evaluated by
                         `study/placement_study.evaluate()` from ITS OWN mapper
                         cache -- AT EVERY SWEPT POINT, because the chip a
                         boundary is depends on the code, the model and the
                         design. A bch sweep at four codes reads four caches
                         per boundary.

so one point of the x axis is one full placement evaluation, and a `recon` bar
is the single boundary `ECC_RECON_DEFAULT` names. TWO THINGS FOLLOW, and both
are deliberate:

* THE SWEEP IS AS COLD AS ITS COLDEST POINT. A boundary whose chip is not in
  the cache at one code is a `[skip]` LINE AND A DROPPED BAR at that point, not
  a refusal and not a bar billed from somebody else's plan -- the refusal
  (`ert-arm-cache-cold`) still fires inside the evaluation, and this module
  catches it BY GUARD ID and carries on, so one missing map costs one bar
  rather than the figure. `bash hpc/run_all.sh --dry-run` prints what is
  missing before anything is drawn.
* DESIGNS DO NOT DECLARE THE SAME BOUNDARIES, so an arch sweep has a different
  bar set per group (plan 3.1). `draw_panel` takes the union and draws no bar
  where a design has none.

It was `study/sweep.py` until ProjectRestructure phase 3: a driver that draws is
L5, and the layer rule has no exception left for it.
"""
from __future__ import annotations

import pandas as pd

from .stacked import grouped_stacks
from .recon_view import _bar_chip_dir, bar_series
from ..study import metrics as metrics_mod
from ..study.common import Session
from ..study.placement_study import evaluate
from ..study.stacks import build_stacks, k_label
from ..study.energy import plot_cats
from ..arch.load import load_provenance
from ..contracts.errors import ConfigError
from ..settings import guards


#: The guard ids a swept POINT may fail on without killing the figure. Every
#: one of them means "this chip is not in the cache at this point", which is a
#: mapping bill and not a modelling error -- and each is still a hard refusal
#: inside the evaluation, where billing a bar from another chip's plan would be
#: the silent failure. Anything else propagates.
COLD_AT_THIS_POINT = frozenset({
    "ert-arm-cache-cold",        # the boundary's own chip is not mapped here
    "ert-arm-shapes-differ",     # it is HALF mapped -- two chips in one bar
    "arm-half-mapped",           # the same, caught one level earlier
    "no-boundary-mapped",        # NONE of this point's boundaries is mapped
})

#: The two exception types a guard may raise. `refusal()` takes the type from
#: the registry, and the COLD ids above are spread across both -- five of them
#: raise `SystemExit` (`__main__` prints those without a traceback) and the
#: rest `ConfigError`. Catching only one of the two is how a point that should
#: have lost its reconstruction bars killed the whole figure instead.
REFUSALS = (ConfigError, SystemExit)

#: The guard ids that mean this point has NOTHING, not even its reference arms.
#: A point like that is dropped whole; the ones above keep their two reference
#: bars and lose their reconstruction bars.
NOTHING_AT_THIS_POINT = frozenset({
    "recon-nothing-collected",   # the reference itself is not mapped here
    "stacks-refused",            # nothing collected at all for this point
})


# ----------------------------------------------------------------- the points
def _bch_points(cfg):
    """x = code strength, on the held design and network.

    ORDERED BY THE AXIS, NOT BY `ECC_KS`. The knob's FIRST entry is the value
    every other axis HOLDS, so its order is "the one I want held, then the
    rest" -- at env.sh's default `39 57 45 30` that draws 39, 57, 45, 30, which
    is not a code-strength axis at all. Sorting by K DESCENDING puts increasing
    strength left to right, and with it increasing BER (1.5, 5, 6.5, 9.5 %),
    which is the direction the group labels already read in.

    IT ONLY BECAME A DEFECT IN PHASE 6. Until the abstract `recon` arm was
    retired, no bar on this figure moved with K -- the two reference bars still
    do not, because the baseline pays a dearer per-bit DRAM PRICE and the
    embedded arm stores no external parity, so neither depends on the code
    (session 1 measured this). An unordered axis under bars that are all flat
    hides nothing. Now that every reconstruction bar is a real chip mapped at
    its own code, the trend IS the figure, and reading it 39 -> 57 -> 45 -> 30
    makes a monotonic result look like noise. Found by LOOKING at the figure,
    which is what phase 6 asks for.

    The HELD value is untouched: `cfg.const_k` is resolved from `ECC_KS`' first
    entry in `config.py` and nothing here can move it.
    """
    order = sorted(cfg.sweep_ks, key=lambda k: -int(k))
    return ([(k, cfg.const_arch, cfg.const_model, k, k_label(cfg, k))
             for k in order], 15)


def _model_points(cfg):
    """x = network, on the one held architecture."""
    return ([(m, cfg.const_arch, m, cfg.code_k, m) for m in cfg.models], 17)


def _arch_points(cfg):
    """x = accelerator, for the one held network."""
    return ([(a, a, cfg.const_model, cfg.code_k, cfg.arch_label(a))
             for a in cfg.archs], 17)


def _fix_points(cfg):
    """NO x axis: ONE group, at the first entry of all three lists.

    `sweep-has-no-figure` refused this until phase 6 because there was no
    renderer for an axis that holds every list. There is one now, and it is
    this module's ordinary path with a one-entry list -- the bars are the
    placements, which is what `ECC_SWEEP=fix` has always meant. env.sh still
    routes `fix` to `ECC_EXPERIMENT=recon`, whose figure carries the per-bar
    notes, the placement table and the Task 3 checks; this is the diff.
    """
    return ([(cfg.const_model, cfg.const_arch, cfg.const_model, cfg.code_k,
              cfg.arch_label(cfg.const_arch))], 17)


POINTS = {"bch": _bch_points, "model": _model_points, "arch": _arch_points,
          "fix": _fix_points}


# ------------------------------------------------------------- one swept point
def _point_cfg(cfg, arch, model, k):
    """THE CONFIGURATION OF ONE POINT: the placement study, held there.

    `with_()` re-resolves and re-checks everything, so pinning the three axes
    also moves `code_k`, `code_t`, every cache slug, and -- for the per-model
    `ECC_LAYERS` spelling -- that model's own layer scope. Which is why a model
    sweep can carry a scope per model now: each point resolves its own.

    `experiment="recon"`, `sweep="fix"` because that IS what a point is: one
    design, one network, one code, every bar billed from its own chip. The
    result file each point writes therefore lands in the placement study's own
    namespace, where it belongs; the FIGURE, the CSV and the manifest are the
    sweep's and are written by `run()` from the outer configuration.
    """
    return cfg.with_(experiment="recon", sweep="fix",
                     const_arch=arch, const_model=model, const_k=k)


def _evaluate_point(cfg, prov, arch, model, k):
    """One point: `(pcfg, pses, out)`, or None with a `[skip]` line saying why.

    A COLD POINT KEEPS ITS TWO REFERENCE BARS AND LOSES ITS RECONSTRUCTION
    ONES. A bch sweep is a sweep of the CODE, and the reference arms exist at
    every code from one chip -- the reference plan is the same 8-bit chip
    whatever K is -- while each boundary is a different chip per code and may
    not be mapped yet. Dropping the whole point would throw away two measured
    bars because a third is not in the cache; billing the third from the
    reference plan would be Task 3 under a heading that says otherwise, which
    is what `no-boundary-mapped` refuses. So `out` comes back None and `run()`
    builds the two reference bars from `build_stacks()` -- the only two arms
    that function still builds.

    Every refusal outside the two sets propagates: a sweep that quietly
    dropped a point because the weight path had drifted would report a shorter
    axis and nothing else.
    """
    pcfg = _point_cfg(cfg, arch, model, k)
    try:
        pses = Session(pcfg).setup()
        pses.collect_arch(arch)
        raw = (pses.raws.get(arch) or {}).get(model)
        if raw is None:
            print(f"  [skip] {arch}/{model} at BCH({pcfg.code_n},{k}): nothing "
                  f"collected -- map it first")
            return None
    except REFUSALS as exc:
        if getattr(exc, "guard_id", None) not in NOTHING_AT_THIS_POINT:
            raise
        print(f"  [skip] {arch}/{model} at BCH({pcfg.code_n},{k}): nothing "
              f"collected at all -- the whole point is dropped\n      "
              + str(exc).replace("\n", "\n      "))
        return None
    try:
        out = evaluate(pcfg, pses, prov, arch, model, raw)
    except REFUSALS as exc:
        gid = getattr(exc, "guard_id", None)
        if gid not in COLD_AT_THIS_POINT:
            raise
        print(f"  [skip] no reconstruction bar at BCH({pcfg.code_n},{k}) on "
              f"{arch}/{model} ({gid}) -- its two reference bars are drawn and "
              f"no bar is billed from another chip's plan:\n      "
              + str(exc).replace("\n", "\n      "))
        out = None
    return pcfg, pses, out


def _reference_stack(pcfg, pses, raw):
    """The two reference bars alone, for a point whose boundaries are cold.

    `build_stacks()` is exactly the right function and nothing else: since the
    abstract `recon` arm was retired it builds `baseline` and `embedded` and no
    third column, so a reference-only point cannot accidentally acquire one.
    """
    return build_stacks(pcfg, raw, pses.recon_pj,
                        recon_idle_pj=pses.recon_idle_pj)


def _point_stack(pcfg, out, bars):
    """One group's DataFrame: rows = plotted categories, columns = the bars
    this design could actually answer.

    `bar_series` is the placement figure's own fold (external parity into the
    DRAM band, then the total check), so a bar means the same thing on both
    figures. An unsupported boundary -- one `placement_eval` returned
    `unsupported` for, e.g. a feasibility the design cannot meet -- gets NO
    column: a zero-height bar in a group that also holds real ones reads as a
    measured zero.
    """
    cats = plot_cats(pcfg)
    cols = {}
    have = {"baseline": out["base_components"], "embedded": out["emb_components"]}
    for res in out["results"]:
        if res.status == "evaluated":
            have[res.placement.key] = res.components
        else:
            print(f"  [note] {res.placement.key} is unsupported here and is "
                  f"not drawn: {res.reason}")
    for a in bars:
        if a in have:
            cols[a] = bar_series(pcfg, cats, a, have[a])
    return pd.DataFrame(cols)


# ------------------------------------------------------------- the metric rows
def metric_rows(cfg, points, groups, stacks, labels, bars):
    """`[(metric, groups, stacks, labels)]` -- one entry per `ECC_METRICS` row.

    THE ENERGY ROW IS THE ONE THAT WAS ALREADY BUILT, passed straight through,
    so the figure's energy row and the CSV beside it cannot disagree. Every
    other row is derived from the SAME `Raw` record each point's energy bars
    were evaluated from -- handed over by the point rather than looked up again
    -- and each bar reads the ART of the chip it was BILLED FROM
    (`_bar_chip_dir`), not the reference's. Getting that wrong is silent: seven
    identical accelerator areas is exactly what a correctly reference-billed
    figure looks like (EnvReorganisation phase 5 found it that way).

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
            pcfg, pses, out, raw = points[g]
            here = [a for a in bars if a in stacks[g].columns]
            # A REFERENCE-ONLY POINT still has its two rows: both bars stand on
            # the reference chip, which is exactly what `_pan({})` resolves to.
            pan = _pan(out or {}, pcfg)
            df, _prov = metrics_mod.metric_stacks(
                pcfg, metric, raw, stacks[g], here,
                code_k=pcfg.code_k,
                cache_dirs={a: _bar_chip_dir(pcfg, pses, pan, a)
                            for a in here},
                engine_bars={a for a in here if a not in ("baseline", "embedded")},
                arm_of={a: ("embedded" if a in ("baseline", "embedded")
                            else "recon") for a in here})
            per_group[g] = df
        drawn = [g for g in groups if g in per_group]
        if not drawn:
            print(f"  [skip] metric {metric}: no group could be built")
            continue
        rows.append((metric, drawn, per_group,
                     {g: labels.get(g, g) for g in drawn}))
    return rows


def _pan(out, pcfg):
    """The two keys `_bar_chip_dir` reads out of a placement panel, without
    building one: which arm each bar was billed from, and that arm's own view.
    `report/recon_view.panel_for` puts the same two on its panel dict."""
    return {"arch": pcfg.const_arch, "billing": out.get("billing") or {},
            "views": out.get("views") or {}}


# ------------------------------------------------------------- every point
def collect_points(cfg, prov=None, ses=None):
    """Walk the axis: `(points, groups, stacks, labels, bars, fontsize)`.

    THE WHOLE FIGURE'S DATA, and the only place it is built -- `run()` draws
    what this returns and the metric-row tests read the same thing, so a test
    cannot pass against a shape the figure never sees.

    `points[g]` is `(point cfg, point session, placement out or None, Raw)`.
    """
    prov = load_provenance() if prov is None else prov
    spec, fontsize = POINTS[cfg.sweep](cfg)
    points, groups, stacks, labels = {}, [], {}, {}
    bar_order = []
    for key, arch, model, k, label in spec:
        got = _evaluate_point(cfg, prov, arch, model, k)
        if got is None:
            continue
        pcfg, pses, out = got
        # THE BARS ARE PER DESIGN. `bar_arms_for` warns and drops a boundary
        # this design has not got, including the case of a design with no
        # `placements.yaml` at all, which then shows its two reference bars.
        bars = pcfg.bar_arms_for(arch)
        raw = (pses.raws.get(arch) or {}).get(model)
        st = (_point_stack(pcfg, out, bars) if out is not None
              else _reference_stack(pcfg, pses, raw))
        if st.empty:
            print(f"  [skip] {key}: no bar could be built")
            continue
        points[key] = (pcfg, pses, out, raw)
        groups.append(key)
        stacks[key] = st
        labels[key] = label
        for a in st.columns:
            if a not in bar_order:
                bar_order.append(a)
        # The outer session's manifest reports what a MAC was charged, and it
        # reads that off a `Raw`. The points own the records now, so hand one
        # over rather than collecting the whole matrix a second time.
        if ses is not None:
            ses.raws.setdefault(arch, {})[model] = raw

    # ONE BAR ORDER FOR THE WHOLE FIGURE, in `bar_arms` order rather than in
    # the order the groups happened to answer, so two panels of one figure put
    # the same boundary in the same slot.
    bars = [a for a in cfg.bar_arms if a in bar_order]
    return points, groups, stacks, labels, bars, fontsize


# -------------------------------------------------------------------- the run
def run(cfg):
    ses = Session(cfg).setup()
    prov = load_provenance()
    points, groups, stacks, labels, bars, fontsize = collect_points(cfg, prov, ses)

    if not groups:
        raise guards.refusal("sweep-nothing-to-plot",
            f"nothing to plot: the {cfg.sweep} sweep produced no groups.\n"
            f"  -> check the [skip] lines above; every bar is a MAPPED CHIP "
            f"since EnvReorganisation phase 6, so a point with no cache is a "
            f"point with no bars\n"
            f"  -> `bash hpc/run_all.sh --dry-run` prints what is missing")
    if len(groups) < 2 and cfg.sweep != "fix":
        print(f"  [note] only one group ({groups[0]}) -- the figure is still "
              f"written, but a sweep of one is just a single measurement")

    ses.report(groups, stacks, labels, bars=bars)

    figs, csv = grouped_stacks(
        cfg, ses.results,
        groups=groups,
        stacks=stacks,
        group_labels=labels,
        title=cfg.figure_title(),
        stem=cfg.stem,
        group_fontsize=fontsize,
        bars=bars,
        bar_tags={a: cfg.bar_tag(a) for a in bars},
        metric_rows=metric_rows(cfg, points, groups, stacks, labels, bars))

    ses.finish(figs, csv, groups, extra={
        "bars": list(bars),
        "bar_labels": {a: cfg.bar_label(cfg.const_arch, a) for a in bars},
        "recon_default": cfg.recon_default_for(cfg.const_arch),
        "bars_per_group": {str(g): list(stacks[g].columns) for g in groups},
        "bars_note": ("EVERY BAR IS A MAPPED CHIP (EnvReorganisation phase 6). "
                      "A reconstruction bar is a boundary the design declares, "
                      "evaluated from its own mapper cache at this point; the "
                      "abstract `recon` arm that applied K/N to every on-chip "
                      "level of every design is retired, and the FINDINGS "
                      "numbers built on it stay as history. A group missing a "
                      "bar is a design that does not declare that boundary, or "
                      "a chip that is not mapped at that point -- both are "
                      "`[skip]` lines in the run log, never a zero."),
        # The placement result each point wrote, or None where the point's
        # boundaries were cold and only its two reference bars were drawn.
        "placement_results": {str(g): (str(points[g][2]["path"])
                                       if points[g][2] else None)
                              for g in groups},
        "reference_only_points": [str(g) for g in groups if not points[g][2]],
    })
    return ses
