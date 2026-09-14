"""What to run: the axes, the layer scope, and where the output goes.

`RunSettings` is the only group that resolves anything: the swept axis takes its
list and the two held axes take their constant, so `archs` and `models` are
DERIVED here and every other module reads those two rather than the six knobs
behind them.

The panel layout (`ECC_EXPERIMENT=panels`) widens `models` to every panel model
so ONE collection pass fills all the panels. It is a page layout, not a fourth
axis -- the x axis is still `sweep`'s.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..contracts.errors import ConfigError
from .arch import CNN_MODELS, TRANSFORMER_MODELS
from .env import (_b, _f, _i, _list, _list_or_none, _of, _oi, _one, _s,
                  _scoped_list, _table)

# ------------------------------------------------------------------ registries
EXPERIMENTS = ("sweep", "diagnose", "baseline", "embedded", "recon", "validate", "dilation",
               "map", "panels")

#: The reconstruction PLACEMENTS `ECC_APPROACHES` may name (EnvReorganisation
#: 3.1). Since phase 6 a bare `recon` is ONE of them -- the one
#: `ECC_RECON_DEFAULT` names -- and a `reconN` name selects a subset directly;
#: `Config.recon_placements_for()` does that resolution. A design that
#: does not declare a named boundary is WARNED and drops the bar; it is not
#: refused, because two designs do not have the same boundaries (CLAUDE.md)
#: and one `ECC_APPROACHES` has to be legal for both.
RECON_PLACEMENT_APPROACHES = ("recon1", "recon2", "recon3", "recon4", "recon5")

APPROACHES = ("baseline", "embedded", "recon") + RECON_PLACEMENT_APPROACHES

#: The two REFERENCE arms, in bar order. They exist on every design, they are
#: billed from the reference plan, and they are the only two bars
#: `study.stacks.build_stacks()` still builds.
#:
#: THE ABSTRACT `recon` ARM IS RETIRED (EnvReorganisation phase 6, 2026-09-14;
#: plan 6.2 and answer 9.2). It was `build_stacks()`'s third column: one engine
#: at the chip entrance, with K/N applied to every on-chip level of every
#: design identically. No physical boundary does that -- once weights are
#: reconstructed at the entrance they are full width on chip -- so it was an
#: optimistic upper bound that reporting rule R-6 existed only to stop anyone
#: quoting as a placement. Every reconstruction bar is a real, mapped chip now,
#: and `Config.bar_arms` is `REFERENCE_ARMS` plus the placements the run names.
REFERENCE_ARMS = ("baseline", "embedded")

#: THE METRICS A FIGURE MAY PLOT -- one figure ROW per entry, top to bottom in
#: THIS order, whatever order `ECC_METRICS` names them in (EnvReorganisation
#: 3.1: "energy on top, latency below it, area below that"). The picture is
#: 2.2's: rows = `ECC_METRICS`, columns = `ECC_SWEEP`, bars = `ECC_APPROACHES`.
#:
#: EVALUATOR ONLY, AND NOT `settings/mapper.py`'s `OPT_METRICS`. Those two are
#: TWO VOCABULARIES and conflating them is the expensive mistake here:
#:
#:     OPT_METRICS  ("energy", "edp", "delay", "last_level_accesses")
#:                  what the MAPPER OPTIMISES. `ECC_OPT_METRIC` picks one and
#:                  it IS in `mapper.IN_FINGERPRINT`, because a plan solved
#:                  for delay is a different plan.
#:     METRICS      ("energy", "edp", "latency", "area")
#:                  what the FIGURE PLOTS. Nothing is re-solved and no mapping
#:                  moves; the numbers are read back off plans already in the
#:                  cache.
#:
#: Note `delay` there against `latency` here -- the words are deliberately not
#: the same, so a value of one can never be pasted into the other.
#:
#: `metrics` IS NOT IN `IN_FINGERPRINT`, AND MUST NEVER BE. It is a plotting
#: choice, and a plotting choice that reached `arch_fingerprint()` would cold
#: every mapper cache in the project at once -- hours of SLURM per design --
#: for a decision about which rows a PNG has. That is why the knob lives in
#: env.sh section 1 and not inside sections 2 and 3's walled-off cold zone.
#: `tests/contract/test_settings.py` pins the exclusion.
#:
#: `area` IS A VALUE OF BOTH THIS AND `SWEEPS`, and they mean different things
#: (EnvReorganisation 6.3, the user's decision 2026-09-14 to keep one word):
#: `ECC_SWEEP=area` is an X AXIS -- the buffer-DEPTH ladder
#: `ECC_DEPTH_SWEEP_SCALES` -- and `ECC_METRICS=area` is a Y AXIS, silicon
#: area in um2. env.sh's two option lines each say which.
METRICS = ("energy", "edp", "latency", "area")

#: The axes a run can walk. `bch`, `model` and `arch` put a list on the x axis
#: and hold the other two; `fix` and `area` hold ALL THREE -- `fix` is the
#: placement study at one point (the bars are what `ECC_APPROACHES` names) and
#: `area` walks the buffer-DEPTH ladder `ECC_DEPTH_SWEEP_SCALES`.
SWEEPS = ("bch", "model", "arch", "fix", "area")

#: Fixed output name per sweep. The whole point of the naming scheme is that a
#: re-run at different constants OVERWRITES rather than adding another file.
#: `fix` and `area` are only reached with ECC_STEM blanked by hand -- env.sh
#: names the placement figure `ReconSweep_optimiser__<model>` under both.
SWEEP_STEMS = {"bch": "BCHsweep", "model": "ModelSweep", "arch": "ArchitectureSweep",
               "fix": "FixedPoint", "area": "DepthSweep"}

#: Spellings accepted for ECC_SWEEP.
SWEEP_ALIASES = {
    "bch": "bch", "code": "bch", "k": "bch", "ksweep": "bch", "bchsweep": "bch",
    "model": "model", "models": "model", "modelsweep": "model",
    "arch": "arch", "archs": "arch", "architecture": "arch",
    "architectures": "arch", "architecturesweep": "arch",
    "fix": "fix", "fixed": "fix", "point": "fix",
    "area": "area", "depth": "area", "depths": "area", "depthsweep": "area",
    "areasweep": "area",
}

#: The two axes that hold every one of the three lists fixed, so the x axis is
#: something else entirely: `fix` has no x axis at all (the arms ARE the bars)
#: and `area` sweeps the depth ladder. Neither has a `report/sweep.py`
#: renderer until phase 6, which `sweep-has-no-figure` says rather than
#: letting the figure code fail on a missing group.
#:
#: SINCE PHASE 6 `fix` HAS ONE: the sweep renderer builds every bar from the
#: placement evaluation, so an axis with no x is simply one group. `area` has
#: none still -- its stem carries no depth, so two depths would overwrite one
#: figure -- and `sweep-has-no-figure` was NARROWED to it rather than retired.
POINT_SWEEPS = ("fix", "area")

#: The point sweeps a `report/sweep.py` FIGURE still has no renderer for.
NO_FIGURE_SWEEPS = ("area",)

#: Which half of the study a result belongs to. See legacy/docs/RESULTS_SCHEMA.md.
#:
#: Pre   the mapping was chosen WITHOUT knowing about reconstruction; the ECC
#:       effect is applied during energy evaluation only. Tasks 1-3.
#: Post  the mapping itself was optimised for the reduced weight width. Task 4+.
#: Task 1 produces `Pre` results by construction: there is no reconstruction
#: yet, so there is nothing a mapper could have been made aware of.
PHASES = ("Pre", "Post")


def result_phase(experiment):
    """THE PHASE IS DERIVED PER ARM, NOT CONFIGURED (EnvReorganisation 6.1,
    2026-09-14). Baseline and embedded are `Pre` BY CONSTRUCTION -- there is
    no reduced representation for a mapper to have been aware of -- and a
    placement mapped on its own chip is `Post`. One run now spans both, so
    `ECC_PHASE` could not be a single value and is gone; the namespace
    `results/evaluation/{Pre|Post}/...` is unchanged and every file lands
    where it always did.

    `RECON_OPTIMIZER` was the second argument until EnvReorganisation phase 3
    (2026-09-14) and is now a constant True: every placement is mapped on its
    own chip, so the fixed-mapping Task 3 reading of the placement study has
    no configuration left that selects it."""
    if experiment == "recon":
        return "Post"
    return "Pre"

#: The bar labels and the short rotated tag under each bar. A PLACEMENT's own
#: label is a property of the design (`placements.yaml`), so only the generic
#: fallback lives here: `Config.bar_label()` / `bar_tag()` ask the design first.
APPROACH_LABELS = {"baseline": "Baseline", "embedded": "Embedded", "recon": "Recon+"}

APPROACH_TAGS = {"baseline": "Base.", "embedded": "Embe.", "recon": "Recon+"}

#: The short tag for a placement bar. `R2` reads under a bar where "Recon at
#: the weight global buffer" does not; the full sentence is the design's own
#: label and goes in the legend-free per-bar note and the CSV.
PLACEMENT_TAGS = {k: f"R{k[-1]}" for k in RECON_PLACEMENT_APPROACHES}

#: The fields of this group the MAPPER sees, and therefore the ones
#: `Config.fingerprint()` hashes.
#:
#: The three axes a mapper job is FOR. `archs` and `models` are derived, so
#: the hash follows the resolved lists rather than the knobs behind them.
IN_FINGERPRINT = frozenset({
    "archs",
    "models",
    "layers",
})


@dataclass(frozen=True)
class RunSettings:
    experiment: str
    sweep: str
    sweep_archs: list
    sweep_models: list
    sweep_ks: list
    #: ECC_EXPERIMENT=panels only: one PANEL per model, the swept axis repeated
    #: inside each. Not a fourth axis -- the x axis is still `sweep`'s.
    panel_models: list
    const_arch: str
    const_model: str
    const_k: int
    approaches: list
    #: `ECC_RECON_DEFAULT` -- which placement a bare `recon` in
    #: `ECC_APPROACHES` means (EnvReorganisation 3.1, phase 6). A sweep wants
    #: ONE reconstruction bar beside its two reference bars; the placement
    #: study wants every boundary, and names them. EVALUATOR ONLY: it decides
    #: which mapper cache is READ, never what the mapper solves, so it is not
    #: in `IN_FINGERPRINT`.
    recon_default: str
    #: `ECC_METRICS` -- which figure ROWS to draw, in `METRICS` order. See
    #: `METRICS` above for why this is not `mapper.OPT_METRIC` and why it is
    #: not in the fingerprint.
    metrics: list
    layers: list
    #: `ECC_LAYERS`' per-model spelling, `{model: [layer, ...]}` -- empty for
    #: the bare-list form (EnvReorganisation 6.9). Layer names are per
    #: network, so a development scope that names two layers of each of three
    #: networks cannot be one list. `config._resolve()` picks the HELD model's
    #: entry into `layers`, which is what every consumer reads; a model with
    #: no entry runs whole. A run that evaluates more than one model at once
    #: is refused by name -- carrying a scope PER MODEL through
    #: `paths.layer_slug` and `Session.select_layers` is phase 6's work.
    layers_by_model: dict
    overwrite: bool
    cache_strict: bool
    #: ECC_RERUN_OPTIMISER=1 -- re-solve a mapping even when a valid cache entry
    #: exists, overwriting it. For refreshing a mapping after a change the
    #: fingerprint does not capture, or to check that a mapping reproduces.
    #: Meaningless (and refused) together with `from_cache`, which forbids
    #: mapping outright.
    rerun_optimiser: bool
    run_note: str
    results_dir: str
    replot_only: bool
    from_cache: bool
    palette: str
    dpi: int
    formats: list
    title_note: str
    nice_labels: bool
    stem_override: str

    # ---- derived: the swept axis takes its list, the held axes their constant
    #: Resolved by `config._resolve()`, never read from the environment. They are
    #: fields rather than properties because every consumer reads THESE two --
    #: `cfg.archs`, `cfg.models` -- and never the six knobs behind them, and
    #: because the mapper fingerprint hashes the resolved lists.
    archs: list = field(default_factory=list)
    models: list = field(default_factory=list)
    #: "cnn" or "transformer": which workload file the models come from. One
    #: sweep may not mix the two families.
    workload: str = "cnn"

    @staticmethod
    def from_env():
        """Every `ECC_*` knob of this group, read once.

        `archs`, `models` and `workload` are DERIVED and are not knobs: the
        swept axis takes its list, the two held axes take their constant, and
        `config._resolve()` fills them in -- the axis rules are validation, and
        the validation of a whole configuration cannot live in one of its six
        parts.
        """
        return RunSettings(
            experiment=_s("ECC_EXPERIMENT", "sweep").lower(),
            sweep=_s("ECC_SWEEP", "bch").lower(),
            # UNSET (None) means "every declared design" and is filled in by
            # `config._resolve()`: `settings/` may not read `archs/`. SET but
            # empty is still refused there -- asking for no design at all is an
            # error, and a plain default could not tell the two apart.
            sweep_archs=_list_or_none("ECC_SWEEP_ARCHS"),
            sweep_models=_list("ECC_SWEEP_MODELS", " ".join(CNN_MODELS)),
            sweep_ks=[int(k) for k in _list("ECC_SWEEP_KS", "57 51 45 39 36 30")],
            panel_models=_list("ECC_PANEL_MODELS"),
            const_arch=_one("ECC_CONST_ARCH", "eyeriss_like"),
            const_model=_one("ECC_CONST_MODEL", "resnet18"),
            const_k=_i("ECC_CONST_K", 51),
            approaches=[a.lower() for a in _list("ECC_APPROACHES", "baseline embedded recon")],
            # THE DEFAULT IS ONE ROW, `energy`. EnvReorganisation section 8's
            # picture of a finished run shows `energy latency`, and that is
            # what a finished STUDY asks for -- but phase 5's own gate is that
            # every existing number and every existing artefact is unchanged,
            # and a multi-row default changes the shape of every figure and
            # every CSV this project writes on the day the knob lands. One row
            # makes the phase's single expected divergence exactly the one the
            # plan predicts -- a new KEY in the config record, with no number
            # under it -- and turning the others on is one word in env.sh.
            recon_default=_one("ECC_RECON_DEFAULT", "recon2").lower(),
            metrics=[m.lower() for m in _list("ECC_METRICS", "energy")],
            # The two spellings of ECC_LAYERS, told apart by the `=`. The
            # resolution into ONE scope needs the held model, which is
            # `_resolve()`'s job; this layer only parses.
            layers=_scoped_list("ECC_LAYERS")[0],
            layers_by_model=_scoped_list("ECC_LAYERS")[1],
            overwrite=_b("ECC_OVERWRITE", False),
            cache_strict=_b("ECC_CACHE_STRICT", True),
            rerun_optimiser=_b("ECC_RERUN_OPTIMISER", False),
            run_note=_s("ECC_RUN_NOTE"),
            results_dir=_s("ECC_RESULTS_DIR", "results"),
            replot_only=_b("ECC_REPLOT_ONLY", False),
            from_cache=_b("ECC_FROM_CACHE", False),
            palette=_s("ECC_PALETTE", "house").lower(),
            dpi=_i("ECC_DPI", 400),
            formats=[f.lower() for f in _list("ECC_FORMATS", "png pdf")],
            title_note=_s("ECC_TITLE_NOTE"),
            nice_labels=_b("ECC_NICE_LABELS", True),
            stem_override=_s("ECC_STEM"),
        )
