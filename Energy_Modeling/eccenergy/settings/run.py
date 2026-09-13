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
                  _table)

# ------------------------------------------------------------------ registries
EXPERIMENTS = ("sweep", "diagnose", "baseline", "embedded", "recon", "validate", "dilation",
               "map", "panels")

APPROACHES = ("baseline", "embedded", "recon")

#: The three axes a run can walk. The other two are held at their constant.
SWEEPS = ("bch", "model", "arch")

#: Fixed output name per sweep. The whole point of the naming scheme is that a
#: re-run at different constants OVERWRITES rather than adding another file.
SWEEP_STEMS = {"bch": "BCHsweep", "model": "ModelSweep", "arch": "ArchitectureSweep"}

#: Spellings accepted for ECC_SWEEP.
SWEEP_ALIASES = {
    "bch": "bch", "code": "bch", "k": "bch", "ksweep": "bch", "bchsweep": "bch",
    "model": "model", "models": "model", "modelsweep": "model",
    "arch": "arch", "archs": "arch", "architecture": "arch",
    "architectures": "arch", "architecturesweep": "arch",
}

#: Which half of the study a result belongs to. See docs/RESULTS_SCHEMA.md.
#:
#: Pre   the mapping was chosen WITHOUT knowing about reconstruction; the ECC
#:       effect is applied during energy evaluation only. Tasks 1-3.
#: Post  the mapping itself was optimised for the reduced weight width. Task 4+.
#: Task 1 produces `Pre` results by construction: there is no reconstruction
#: yet, so there is nothing a mapper could have been made aware of.
PHASES = ("Pre", "Post")

APPROACH_LABELS = {"baseline": "Baseline", "embedded": "Embedded", "recon": "Recon+"}

APPROACH_TAGS = {"baseline": "Base.", "embedded": "Embe.", "recon": "Recon+"}

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
    layers: list
    phase: str
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
            layers=_list("ECC_LAYERS"),
            phase=(_one("ECC_PHASE", "Pre") or "Pre").capitalize(),
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
