"""THE RESOLVED CONFIGURATION of one run -- six frozen settings objects, checked
against the designs they name.

    cfg = config.load_config()          # the environment, read once
    cfg.weight_bits                     # any knob, by its own name
    cfg.code                            # or the group it belongs to
    cfg2 = cfg.with_(const_k=39)        # a changed copy, re-resolved and re-checked

WHAT IS HERE AND WHAT IS IN `settings/`
---------------------------------------
`settings/` (L0) holds the KNOBS: six frozen dataclasses, each owning its own
`ECC_*` variables, its own registry of legal values, and the declaration of which
of its fields reach the mapper. It imports nothing of ours but `contracts`.

This module holds the RESOLUTION, which is a different job and sits at a
different level: the swept axis takes its list and the held axes their constant;
`ECC_RECON_ERT_ARM` is looked up in THIS DESIGN's arm list and the datawidth it
implies is filled in; `code_t` comes out of the BCH table. Resolution needs
`arch/` and `physics/`, so this module sits ABOVE them -- which is what ended
`ProjectRestructure.md` section 2.3's import cycle. Nothing in `physics/`,
`arch/` or `settings/` imports this file, and the layer rule now says so.

THE CONFIGURATION IS FROZEN. `load_config()` resolves once; `with_()` is the only
way to a different one and it re-runs every check, so a Config that exists has
been validated. `dataclasses.replace(cfg, ...)` no longer works and must not:
replacing one field of a six-part object would leave the other five holding
values that were derived from the old one.

ONE VALIDATOR, NOT SIX. `_resolve()` is `Config.__post_init__` as it stood,
moved: the rules it enforces are about a WHOLE configuration -- the sweep against
the code, the experiment against the phase, the arm against the design -- and
splitting them into the six parts would have meant six partial validators that
each see a third of the picture, plus an order of error messages nobody chose.

    cfg.to_dict()     the flat 119-key record every manifest and result carries.
                      ITS KEY ORDER IS `FIELD_ORDER` AND IS PART OF THE FILE
                      BYTES -- `tests/contract/test_settings.py` pins it.
    cfg.fingerprint() the mapper cache key: the union of the six groups'
                      `IN_FINGERPRINT` sets, hashed sorted. A moved hash is a
                      cold matrix, so that union is pinned by the same test.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .arch import arms as arms_mod
from .arch import design as design_mod
from .arch import fingerprint as fingerprint_mod
from .arch import placements
from .arch.load import load_standard, mac_candidates
from .contracts.errors import ConfigError   # noqa: F401 -- RE-EXPORTED:
#: `config.ConfigError` is what `__main__` catches and what five test
#: modules import. Phase 6 moved every raise site in this file onto
#: `guards.refusal()`, so nothing here references the name any more --
#: it is kept because removing it would break those seven callers.
from .physics import widths
from .physics.embedded import EmbeddedLayout
from .settings import arch as arch_settings
from .settings import banner as banner_mod
from .settings import code as code_settings
from .settings import energy as energy_settings
from .settings import mapper as mapper_settings
from .settings import recon as recon_settings
from .settings import run as run_settings
from .settings.arch import (ARCH_FIDELITIES, CNN_MODELS, TRANSFORMER_MODELS,
                            WEIGHT_CAPACITY_SCOPES, ArchSettings)
from .settings.code import BCH63_KTOD, PARITY_GROUPINGS, CodeSettings
from .settings.energy import DC_MEASUREMENT_CLOCK_NS, EnergySettings
from .settings.mapper import (OPT_METRICS, VICTORY_MAX_SCALE,
                              VICTORY_REFERENCE_LEVELS, VICTORY_SCALINGS,
                              MapperSettings)
from .settings.recon import (RECON_ENCODER_SITES, RECON_GRANULARITIES,
                             RECON_PACKINGS, ReconSettings)
from .settings.run import (APPROACH_LABELS, APPROACH_TAGS, APPROACHES,
                           BAR_ARMS, EXPERIMENTS, POINT_SWEEPS,
                           RECON_PLACEMENT_APPROACHES, SWEEP_ALIASES,
                           SWEEP_STEMS, SWEEPS, RunSettings)
from .settings import guards

#: WHICH DESIGNS EXIST IS DATA (phase 5). These three names are what every
#: caller has always said; they are functions of `archs/<name>/design.yaml` now,
#: evaluated at import so a caller that iterates them still gets a plain tuple
#: or dict. A design added to `archs/` is in all three the moment it declares a
#: `design.yaml`, and in no Python file at all.
KNOWN_ARCHS = design_mod.known_archs()
ARCH_LABELS = design_mod.arch_labels()
BRACKET_PAIRS = design_mod.bracket_pairs()

#: The six groups, in the order a flat record lists them.
GROUPS = ("run", "code", "arch", "mapper", "recon", "energy")

_CLASSES = {"run": RunSettings, "code": CodeSettings, "arch": ArchSettings,
            "mapper": MapperSettings, "recon": ReconSettings,
            "energy": EnergySettings}

#: knob name -> the group that owns it. Built from the six classes, so a field
#: added to one of them is addressable by its own name the same day.
_OWNER = {f: g for g, c in _CLASSES.items() for f in c.__dataclass_fields__}

#: The six groups' own declarations of which of their fields reach the MAPPER.
#: `Config.fingerprint()` hashes the union, so a knob is marked in the module
#: that declares it and not in a list kept by hand at the other end of a file.
_FINGERPRINT_MODULES = {"run": run_settings, "code": code_settings,
                        "arch": arch_settings, "mapper": mapper_settings,
                        "recon": recon_settings, "energy": energy_settings}

#: FIELDS DERIVED FROM THE DESIGNS, not read from the environment
#: (EnvReorganisation phase 1, 2026-09-14). `from_env()` leaves each None and
#: `_resolve()` fills it from `archs/`. They stay record fields so every
#: manifest keeps the key and the value, and an explicit `with_(field=...)`
#: is still an override, sticky across later `with_()` calls exactly as the
#: env.sh tables were. Two of them depend on WHICH design is held, so
#: `with_()` re-derives those two whenever the call moves the design axis;
#: the other two are study-wide and are never re-derived once resolved.
DERIVED_FROM_HELD_DESIGN = ("leakage_nw", "weight_width_glb_mult")
DESIGN_AXIS_KNOBS = ("const_arch", "sweep_archs", "sweep", "experiment",
                     "weight_bits")

#: EVERY field, in the order `to_dict()` emits it -- which is the order it
#: appears in every manifest and every result JSON on disk. Changing this order
#: rewrites those files without changing a number in them, so it is pinned by a
#: test rather than left to the order the six groups happen to be listed in.
FIELD_ORDER = (
    "experiment", "sweep", "sweep_archs", "sweep_models", "sweep_ks",
    "panel_models", "const_arch", "const_model", "const_k", "approaches",
    "weight_bits", "activation_bits", "acc_bits_override", "code_n",
    "emb_weights_per_cw_override", "parity_grouping", "parity_charge_padding",
    "decode_enabled", "decode_pj_base", "decode_pj_emb",
    "recon_charges_decode", "recon_json", "recon_incremental_table",
    "recon_idle_table", "recon_pj_override",
    "recon_incremental_fallback_pj", "recon_idle_fallback_pj",
    "recon_clock_gating_pct", "energy_model_rev", "recon_packing",
    "recon_granularity",
    "recon_encoder_site", "recon_ert_aware", "recon_ert_arm",
    "dram_pj_per_bit", "baseline_dram_pj_per_bit", "dram_background_pj",
    "dram_refresh_pj", "static_energy", "leakage_nw", "latency_model",
    "dram_bandwidth_mbps", "arch_clock_mhz", "recon_bw_scale",
    "onchip_bw_bitaware", "weak_enabled", "weak_n", "weak_k",
    "baseline_inflates_onchip", "split_read_write", "classify_mode",
    "arch_fidelity", "force_technology", "force_datawidth", "dram_depth",
    "global_cycle_seconds", "weight_capacity_scale", "weight_capacity_scope",
    "weight_factor_relax", "mapspace_constrain", "weight_datawidth",
    "weight_depth_scale", "weight_width_glb_mult",
    "disable_pair_geometry_assert", "weight_depth_levels",
    "weight_datawidth_levels", "noc_enabled", "noc_wire_pj_per_bit_mm",
    "noc_router_pj", "noc_pe_latch_pj", "noc_scale", "mac_pj_override",
    "layers", "layers_by_model", "overwrite", "cache_strict", "rerun_optimiser",
    "run_note", "opt_metric", "victory", "victory_scaling", "mapper_threads",
    "mapper_timeout", "mapper_algorithm", "mapper_seed", "mapper_search_size",
    "mapper_max_permutations", "results_dir", "replot_only", "from_cache",
    "palette", "dpi", "formats", "title_note", "nice_labels", "stem_override",
    "archs", "models", "code_k", "code_t", "workload",
)


# --------------------------------------------------------------- resolving one
class _Draft:
    """A whole configuration, mutable, while it is being resolved.

    `_resolve()` is `Config.__post_init__` as it stood before phase 4, and it
    both checks and DERIVES -- `archs`, `models`, `code_t`, the arm's datawidth.
    A frozen object cannot be derived into, and six frozen objects cannot see
    each other's fields, so the resolution runs once on this and the six are
    built from the answer.

    It carries exactly the fields a `Config` does, so it is also what
    `arch.arms.mapper_arm_spec()` is handed while the arm is being looked up --
    the same duck it was handed before, at the same point in the same order.
    """

    def __init__(self, flat):
        self.__dict__.update(flat)


def _resolve(self):
    """Check a whole configuration and fill in everything derived from it.

    Moved from `Config.__post_init__`; the checks, their order and their
    messages are unchanged, because each of them is a finding somebody paid for.
    """
    if self.experiment not in EXPERIMENTS:
        raise guards.refusal("unknown-experiment",
            f"ECC_EXPERIMENT={self.experiment!r}; choose one of "
            f"{', '.join(EXPERIMENTS)}")

    self.sweep = SWEEP_ALIASES.get(self.sweep, self.sweep)
    if self.sweep not in SWEEPS:
        raise guards.refusal("unknown-sweep",
            f"ECC_SWEEP={self.sweep!r}; choose one of "
            f"{', '.join(SWEEPS)}")

    bad = [a for a in self.approaches if a not in APPROACHES]
    if bad:
        raise guards.refusal("unknown-approach",
            f"ECC_APPROACHES has unknown entries {bad}; choose from "
            f"{', '.join(APPROACHES)}")
    if not self.approaches:
        raise guards.refusal("approaches-empty",
            "ECC_APPROACHES is empty -- nothing to compare")
    # canonical left-to-right bar order, however it was typed
    self.approaches = [a for a in APPROACHES if a in self.approaches]

    # ---- resolve the three axes ----------------------------------------
    # ECC_SWEEP_ARCHS defaults to EVERY DECLARED DESIGN. The default is applied
    # here and not in `settings/` because which designs exist is read out of
    # `archs/<name>/design.yaml`, and the knobs sit below the layer that may
    # read a design (phase 5). The resolved list is what it always was, in
    # `design.yaml`'s declared `order:`.
    if self.sweep_archs is None:
        self.sweep_archs = list(design_mod.known_archs())

    # ECC_SWEEP=fix AND ECC_SWEEP=area HOLD ALL THREE LISTS (EnvReorganisation
    # 3.1, phase 3). `fix` is the placement study at one point -- the bars are
    # what ECC_APPROACHES names -- and `area` walks the buffer-DEPTH ladder
    # with architecture, model and code held. Neither puts one of the three
    # lists on the x axis, so neither may widen one: env.sh collapses
    # ECC_SWEEP_* onto the FIRST entry of each list under both, exactly as
    # section 4 collapsed them for ECC_RECON_MODELING=1 before it.
    #
    # `report/sweep.py` has no renderer for either axis until phase 6, and a
    # missing renderer must SAY so rather than raising a KeyError on a group
    # that was never collected.
    if self.sweep in POINT_SWEEPS and self.experiment in ("sweep", "panels"):
        raise guards.refusal("sweep-has-no-figure",
            f"ECC_SWEEP={self.sweep} holds all three lists fixed, so there is "
            f"no x axis for ECC_EXPERIMENT={self.experiment} to draw.\n"
            f"  -> ECC_SWEEP=fix is the PLACEMENT study (ECC_EXPERIMENT=recon), "
            f"which env.sh routes to automatically\n"
            f"  -> ECC_SWEEP=area maps the depth ladder; read it with "
            f"`python3 -m eccenergy.report.dilation_view --levels`\n"
            f"  -> a sweep FIGURE over either axis is EnvReorganisation phase 6")

    if self.sweep == "arch":
        if not self.sweep_archs:
            raise guards.refusal("sweep-archs-empty",
                "ECC_SWEEP=arch but ECC_SWEEP_ARCHS is empty")
        self.archs = list(dict.fromkeys(self.sweep_archs))
    else:
        if not self.const_arch:
            raise guards.refusal("const-arch-empty",
                f"ECC_SWEEP={self.sweep} holds the architecture "
                f"fixed, but ECC_CONST_ARCH is empty")
        self.archs = [self.const_arch]

    if self.sweep == "model":
        if not self.sweep_models:
            raise guards.refusal("sweep-models-empty",
                "ECC_SWEEP=model but ECC_SWEEP_MODELS is empty")
        self.models = list(dict.fromkeys(self.sweep_models))
    else:
        if not self.const_model:
            raise guards.refusal("const-model-empty",
                f"ECC_SWEEP={self.sweep} holds the model fixed, "
                f"but ECC_CONST_MODEL is empty")
        self.models = [self.const_model]

    # ---- the panel layout (ECC_EXPERIMENT=panels) ----------------------
    # One panel per model, the swept axis repeated inside each. `models` is
    # widened to every panel model so ONE collection pass fills all the
    # panels; the swept axis is untouched, which is what keeps this a page
    # layout rather than a fourth axis.
    if self.experiment == "panels":
        if self.sweep not in ("arch", "model"):
            guards.refuse("panels-needs-arch-or-model-sweep",
                f"ECC_EXPERIMENT=panels with ECC_SWEEP={self.sweep}: a panel "
                f"per model would then vary the model AND the code between "
                f"panels, which is two axes at once.\n"
                f"  -> use ECC_SWEEP=arch (an architecture sweep per model)")
        if not self.panel_models:
            raise guards.refusal("panels-need-panel-models",
                "ECC_EXPERIMENT=panels needs ECC_PANEL_MODELS, e.g.\n"
                '  ECC_PANEL_MODELS="resnet18 mobilenet_v2"')
        self.panel_models = list(dict.fromkeys(self.panel_models))
        self.models = list(self.panel_models)

    if self.sweep == "bch":
        if not self.sweep_ks:
            raise guards.refusal("sweep-ks-empty",
                "ECC_SWEEP=bch but ECC_SWEEP_KS is empty")
        self.sweep_ks = list(dict.fromkeys(self.sweep_ks))
        bad_k = [k for k in self.sweep_ks if k >= self.code_n]
        if bad_k:
            raise guards.refusal("sweep-k-lt-n",
                f"ECC_SWEEP_KS entries must be < N={self.code_n}: {bad_k}")
    self.code_k = self.const_k

    if self.code_k >= self.code_n:
        raise guards.refusal("need-k-lt-n",
            f"need K < N; got N={self.code_n} "
            f"K={self.code_k} (ECC_CONST_K)")
    if self.weight_bits <= 0:
        raise guards.refusal("weight-bits-positive",
            "ECC_WEIGHT_BITS must be positive")

    if self.activation_bits <= 0:
        raise guards.refusal("activation-bits-positive",
            "ECC_ACTIVATION_BITS must be positive")
    if self.acc_bits_override is not None:
        # The SENSITIVITY STUDY only. The primary comparison keeps each
        # design's published accumulator width, because psum precision is
        # an architectural property (v1 truncates to 16b, v2 accumulates at
        # 20b, Simba at 24b) and equalising it equalises the architectures.
        if self.acc_bits_override < self.weight_bits:
            raise guards.refusal("acc-bits-not-narrower",
                f"ECC_ACC_BITS={self.acc_bits_override} is narrower than "
                f"ECC_WEIGHT_BITS={self.weight_bits}; an accumulator cannot "
                f"be narrower than the operands it accumulates")
    if self.parity_grouping not in PARITY_GROUPINGS:
        raise guards.refusal("unknown-parity-grouping",
            f"ECC_PARITY_GROUPING must be one of "
            f"{', '.join(PARITY_GROUPINGS)}")

    # ---- the facts the DESIGNS declare, derived into the record ------------
    # EnvReorganisation phase 1. Each is None straight out of `from_env()` and
    # after a `with_()` that did not name it (`DERIVED_FROM_DESIGNS`).
    if self.dram_depth is None:
        self.dram_depth = int(load_standard()["study"]["dram"]["depth_words"])
    if self.arch_clock_mhz is None:
        self.arch_clock_mhz = design_mod.clock_table()
    held = self.archs[0] if self.archs else None
    if self.leakage_nw is None:
        self.leakage_nw = dict(design_mod.leakage_nw(held) or {}) if held else {}
    if self.weight_width_glb_mult is None:
        self.weight_width_glb_mult = (design_mod.width_table(held).glb_mult(self.weight_bits)
                                      if held else widths.DEFAULT_GLB_MULT)

    # ---- Task 3: the placement study --------------------------------
    if self.recon_packing not in RECON_PACKINGS:
        raise guards.refusal("unknown-recon-packing",
            f"ECC_RECON_PACKING must be one of "
            f"{', '.join(RECON_PACKINGS)}")
    if self.recon_granularity not in RECON_GRANULARITIES:
        raise guards.refusal("unknown-encoder-granularity",
            f"ECC_RECON_ENCODER_GRANULARITY must be one of "
            f"{', '.join(RECON_GRANULARITIES)}")
    # ---- the three PRICES. `>= 0`, not `> 0` -- ProjectRestructure Appendix B.
    # A width must be positive; a PRICE need not be. Zero is the ablation "what
    # if this term were free", which is the upper bound on how much the term was
    # ever worth, and refusing it confused "impossible" with "not the value we
    # used". Negative stays refused: that one really is impossible.
    _prices = (("ECC_MAC_PJ_OVERRIDE", self.mac_pj_override,
                "the per-MAC energy (pJ per 8-bit MAC), or empty for the ERT's "
                "value"),
               ("ECC_DRAM_PJ_PER_BIT", self.dram_pj_per_bit,
                "the per-bit DRAM dynamic access energy (8 = Accelergy LPDDR4 as "
                "modelled, 20 = Horowitz ISSCC 2014, 40 = this study's default)"),
               ("ECC_BASELINE_DRAM_PJ_PER_BIT", self.baseline_dram_pj_per_bit,
                "the baseline arm's per-bit DRAM dynamic access energy (70 = this "
                "study's value for the bigger, indexed conventional-ECC array; "
                "EMPTY = the pre-2026-09-10 parity-traffic model)"))
    for _knob, _price, _what in _prices:
        if _price is None:
            continue
        if _price < 0:
            raise guards.refusal("negative-price",
                f"{_knob}={_price}: {_what}. A price may be zero -- that is the "
                f"ablation -- but it may not be NEGATIVE.")
        if _price == 0:
            # A guard message LEADS WITH ONE SHORT LINE: `recon_caveats()` puts
            # that line, and only that line, on the figure.
            guards.refuse("zero-price",
                f"{_knob}=0 prices this term at NOTHING.\n"
                f"     A legitimate ABLATION -- it measures how much the term was "
                f"worth and is the upper bound on its importance -- but it is not a "
                f"number this study has published, and every percentage computed "
                f"against it means something different.")
    if self.recon_encoder_site not in RECON_ENCODER_SITES:
        raise guards.refusal("unknown-encoder-site",
            f"ECC_RECON_ENCODER_SITE must be one of "
            f"{', '.join(RECON_ENCODER_SITES)} -- `destination` is one "
            f"encoder per destination of a multicast network (the count is "
            f"Timeloop's destination-side arrivals, ingresses x multicast "
            f"factor); `source` is one encoder before the fanout and is the "
            f"pre-2026-09-09 row, kept for the diff. See recon.ENCODER_SITES")
    for _n, _v in (("ECC_DRAM_BACKGROUND_PJ", self.dram_background_pj),
                   ("ECC_DRAM_REFRESH_PJ", self.dram_refresh_pj)):
        if _v < 0:
            raise guards.refusal("dram-static-terms-nonnegative",
                f"{_n}={_v} must be >= 0 (0 = term not modelled)")
    # THE PHASE IS DERIVED PER ARM since 2026-09-14 (EnvReorganisation 6.1,
    # `settings.run.result_phase`): baseline and embedded are `Pre` by
    # construction, a placement mapped on its own chip is `Post`. The two
    # guards that used to hold ECC_PHASE and RECON_OPTIMIZER in agreement
    # (`task4-is-post`, `task3-is-pre`) checked a knob that no longer exists.
    if self.experiment == "recon":
        if not self.archs:
            raise guards.refusal("recon-needs-an-arch",
                "the reconstruction placement study needs at least one "
                "architecture.\n  -> set ECC_ARCHS (env.sh section 1); "
                "ECC_SWEEP=fix holds its FIRST entry")
        # SEVERAL ARCHITECTURES ARE ONE PANEL EACH, NOT ONE AXIS. Each
        # design has its own weight path and therefore its own list of
        # feasible boundaries, so they cannot share an x axis -- env.sh
        # section 4 and CLAUDE.md both say so, and `study/placement_study.py`
        # `figure()` honours it by giving every design its own axes, its own
        # boundary list and its own two reference bars. What is shared is
        # the page, the legend, the category set and the energy unit.
        # A repeated name is not an error: `archs` is de-duplicated above
        # (`dict.fromkeys`), so "a a" draws ONE panel for `a` rather than
        # the same design twice. The panel list is printed in the config
        # table and every panel heading names its design, so a typo that
        # collapses two panels into one is visible in the run.
        if len(self.models) != 1:
            raise guards.refusal("recon-one-model",
                f"the reconstruction placement study runs on ONE model, not "
                f"{len(self.models)} ({', '.join(self.models)}).\n"
                f"  -> ECC_SWEEP=fix holds the FIRST entry of ECC_MODELS "
                f"(env.sh section 1); set ECC_CONST_MODEL for another")
        if self.split_read_write:
            guards.refuse("recon-no-split-read-write",
                "ECC_SPLIT_READ_WRITE=1 splits the on-chip categories in "
                "proportion to their ACCESS COUNTS, but a reconstruction "
                "placement changes the read and write bit-volumes by "
                "different factors, so the split would be attributed "
                "wrongly.\n  -> run the placement study with "
                "ECC_SPLIT_READ_WRITE=0")
    d = BCH63_KTOD.get(self.code_k) if self.code_n == 63 else None
    self.code_t = (d - 1) // 2 if d else max(1, (self.code_n - self.code_k) // 6)

    # ---- which workload file the models come from ----------------------
    cnn = [m for m in self.models if m in CNN_MODELS]
    tfm = [m for m in self.models if m in TRANSFORMER_MODELS]
    if cnn and tfm:
        raise guards.refusal("no-cnn-transformer-mix",
            "one sweep cannot mix CNNs and transformers -- they come from "
            f"different workload files.\n  CNNs        : {' '.join(cnn)}\n"
            f"  transformers: {' '.join(tfm)}")
    self.workload = "transformer" if tfm else "cnn"

    # ---- the LAYER SCOPE, which is per MODEL (EnvReorganisation 6.9) ----
    # `ECC_LAYERS` has two spellings and `settings.env._scoped_list` tells
    # them apart by the `=`: a bare list is these layers of whichever model
    # runs, and `model=a b; model2=c` is per network. Layer names ARE per
    # network, so the development scope (two layers each of three models)
    # cannot be one list -- and `layers-not-in-model` must stay a refusal, so
    # naming resnet18's layers on a mobilenet run has to be impossible rather
    # than silent.
    #
    # ONE SCOPE PER RUN is what everything downstream assumes:
    # `paths.layer_slug`, the results namespace and `Session.select_layers`
    # all take ONE list and apply it to every model. So the table resolves
    # against the HELD model here, and `Session.setup()` -- the one place the
    # scope is CONSUMED -- refuses a run that would EVALUATE more than one
    # model under it (`layers-per-model-one-model`). The refusal cannot live
    # here: `toolchain.units` enumerates a multi-model map run and derives one
    # configuration per model from it, and that configuration has to resolve.
    #
    # A model with no entry runs WHOLE -- an empty scope, as if ECC_LAYERS
    # were unset. That is the spelling the plan asks for (6.9) and it is why
    # the table may be sparse.
    if self.layers_by_model:
        self.layers = list(self.layers_by_model.get(self.models[0], ()))

    # ECC_FROM_CACHE forbids invoking Timeloop at all, so it cannot also be
    # asked to re-run the mapper. Silently preferring one would mean a run
    # asked to refresh its mappings quietly refreshing nothing.
    if self.rerun_optimiser and (self.from_cache or self.replot_only):
        blocker = "ECC_FROM_CACHE=1 (--eval)" if self.from_cache \
            else "ECC_REPLOT_ONLY=1 (--replot)"
        guards.refuse("rerun-needs-the-mapper",
            f"ECC_RERUN_OPTIMISER=1 asks the mapper to re-solve every shape, "
            f"but {blocker} never invokes Timeloop at all.\n"
            f"  -> re-map with `bash run.sh map`, then evaluate with `--eval`")

    if self.weak_enabled and self.weak_k >= self.weak_n:
        raise guards.refusal("need-weak-k-lt-n",
            f"need WEAK_K < WEAK_N; got {self.weak_n}/{self.weak_k}")
    if self.classify_mode not in ("instances", "name"):
        raise guards.refusal("unknown-classify",
            "ECC_CLASSIFY must be 'instances' or 'name'")
    if self.weight_capacity_scale <= 0:
        raise guards.refusal("capacity-scale-positive",
            f"ECC_WEIGHT_CAPACITY_SCALE={self.weight_capacity_scale}: the "
            f"weight-capacity multiplier must be positive. 1.0 is the "
            f"declared design; N/K = {self.code_n / self.code_k:.4f} at "
            f"BCH({self.code_n},{self.code_k}) is the reconstruction arm's "
            f"effective capacity; below 1 shrinks the design")
    if self.weight_capacity_scope not in WEIGHT_CAPACITY_SCOPES:
        raise guards.refusal("unknown-capacity-scope",
            f"ECC_WEIGHT_CAPACITY_SCOPE must be one of "
            f"{', '.join(WEIGHT_CAPACITY_SCOPES)} -- `exclusive` dilates "
            f"only a level whose keep list is Weights alone, so the room "
            f"can only go to weights; `shared` also dilates a level that "
            f"holds Weights beside another dataspace, which hands the "
            f"mapper free capacity for that dataspace too. They bracket "
            f"one design and are quoted as a pair")
    if self.weight_depth_scale <= 0:
        raise guards.refusal("depth-scale-positive",
            f"ECC_WEIGHT_DEPTH_SCALE={self.weight_depth_scale}: the depth "
            f"multiplier must be positive. 1.0 is the declared design; "
            f"prompt_2's search grid is the sqrt(2) ladder "
            f"1 / 0.71 / 0.5 / 0.35 / 0.25 / 0.18 / 0.125")
    # `glb-width-mult-ge-1` is checked where the ratio is DECLARED now:
    # `arch.design.validate_widths()`, on the design's own widths.yaml.
    # NO "the width must also divide ECC_WEIGHT_BITS" CHECK LIVES HERE.
    # It used to, and it was wrong: it asserted that both arms share one
    # declared width, which made prompt_2's 98 (q=7) and 95 (q=5) look
    # illegal and got them replaced by lcm(q, 8). The arms do NOT share a
    # width -- each declares the one that suits its OWN datawidth, and
    # neither is ever mapped on the other's silicon. THE WIDTH TABLE is
    # applied per level by `archs._set_weight_geometry()`, which picks a
    # multiple of that level's own datawidth, so `width % datawidth == 0`
    # holds by construction and there is nothing here to validate.
    # See eccenergy/widths.py.
    if self.weight_datawidth is not None and self.weight_datawidth < 1:
        raise guards.refusal("datawidth-positive",
            f"ECC_WEIGHT_DATAWIDTH={self.weight_datawidth}: the on-chip "
            f"weight datawidth must be a positive integer number of bits. "
            f"Leave it EMPTY for the 8-bit baseline/embedded arm; set it "
            f"to round(8*K/N) for the reconstruction arm (4 at "
            f"BCH(63,30)). Per-code values are tabulated in prompt_2.md.")
    if (self.weight_datawidth is not None
            and self.weight_datawidth > self.weight_bits):
        raise guards.refusal("datawidth-le-weight-bits",
            f"ECC_WEIGHT_DATAWIDTH={self.weight_datawidth} exceeds "
            f"ECC_WEIGHT_BITS={self.weight_bits}. The reconstruction arm "
            f"stores a REDUCED weight; a wider one is not a code rate.")
    if self.arch_fidelity not in ARCH_FIDELITIES:
        raise guards.refusal("unknown-arch-fidelity",
            f"ECC_ARCH_FIDELITY must be one of "
            f"{', '.join(ARCH_FIDELITIES)}")
    if self.opt_metric not in OPT_METRICS:
        raise guards.refusal("unknown-opt-metric",
            f"ECC_OPT_METRIC={self.opt_metric!r}; choose one of "
            f"{', '.join(OPT_METRICS)}")
    if self.victory_scaling not in VICTORY_SCALINGS:
        raise guards.refusal("unknown-victory-scaling",
            f"ECC_VICTORY_SCALING must be one of "
            f"{', '.join(VICTORY_SCALINGS)}")
    if self.victory < 1:
        raise guards.refusal("victory-ge-1",
            "ECC_VICTORY must be >= 1")
    if self.palette not in ("house", "cvd"):
        raise guards.refusal("unknown-palette",
            "ECC_PALETTE must be 'house' or 'cvd'")
    for fmt in self.formats:
        if fmt not in ("png", "pdf", "svg"):
            raise guards.refusal("unknown-format",
                f"ECC_FORMATS: unsupported format {fmt!r}")

    # ---- prompt_6: reconstruction-aware mapping ----------------------
    # `ert-arm-needs-optimiser` RETIRED with RECON_OPTIMIZER (EnvReorganisation
    # phase 3): the knob it required is a constant True now, so the coupling
    # can no longer be violated and a guard that cannot fire is folklore.
    if self.recon_ert_arm in ("", "reference"):
        self.recon_ert_arm = "reference"
    else:
        # A placement key. The arm IS a datawidth configuration plus a
        # declared bandwidth scale plus (sometimes) an ERT bump -- prompt_7
        # 6.4's three axes -- so resolve the datawidth half here and let
        # every consumer of `weight_datawidth` / `weight_datawidth_levels`
        # see it. `mapper_arm_spec`, NOT `ert_arm_spec`: since prompt_7 B2
        # the arms to map are every DISTINCT CHIP, and R1 and R3 are chips
        # with no ERT bump at all. `ert_arm()` below still answers only for
        # the arms that have one, so the bump and the `ert-` slug are
        # unchanged for every directory already on disk.
        if len(self.archs) != 1:
            raise guards.refusal("ert-arm-one-arch",
                f"ECC_RECON_ERT_ARM={self.recon_ert_arm!r} names the arm of ONE "
                f"mapper job on ONE architecture; this configuration has "
                f"{len(self.archs)}: {', '.join(self.archs)}")
        try:
            spec = arms_mod.mapper_arm_spec(self.archs[0], self.recon_ert_arm, self)
        except (KeyError, ValueError) as exc:
            raise guards.refusal("unknown-ert-arm",
                f"ECC_RECON_ERT_ARM={self.recon_ert_arm!r}: {exc}") from None
        q = widths.declared_datawidth(self.code_n, self.code_k)
        # `derived-datawidth` and `derived-datawidth-levels` RETIRED with the
        # two env reads (EnvReorganisation phase 4). Both existed to refuse
        # ECC_WEIGHT_DATAWIDTH / _LEVELS set BESIDE an arm that derives them;
        # `settings/arch.py` no longer reads either name, so the arm is the
        # only thing that can set them and the coupling cannot be violated. A
        # guard that cannot fire is folklore -- the same rule that retired
        # `ert-arm-needs-optimiser` and `unknown-placement` in phase 3.
        if spec["narrow_levels"]:
            self.weight_datawidth = q
            self.weight_datawidth_levels = tuple(spec["narrow_levels"])
        # R1 NARROWS NOTHING ON CHIP, so it must leave `weight_datawidth`
        # alone: setting q with an EMPTY level list is the spelling that
        # narrows EVERY weight level, which is a different chip from the
        # one R1 declares. Its architecture is the reference's until Phase
        # C1.2 emits the DRAM bandwidth scale; what keeps the two caches
        # apart meanwhile is the `arm-recon1` slug component below.

    unknown = [a for a in self.archs if a not in KNOWN_ARCHS]
    if unknown:
        print(f"[config] note: looked up in example_designs/ as-is: {', '.join(unknown)}")
    unknown = [m for m in self.models if m not in CNN_MODELS + TRANSFORMER_MODELS]
    if unknown:
        print(f"[config] note: not a listed model; looked up in the workload "
              f"file as-is: {', '.join(unknown)}")


# ------------------------------------------------------------------ the config
@dataclass(frozen=True)
class Config:
    """One resolved run. Six frozen groups, addressable by any knob's own name.

    `cfg.victory` and `cfg.mapper.victory` are the same value; the first is what
    every call site says today and the second is what a signature should say
    when it is rewritten to state its dependencies (ProjectRestructure 4.3).
    """

    run: RunSettings
    code: CodeSettings
    arch: ArchSettings
    mapper: MapperSettings
    recon: ReconSettings
    energy: EnergySettings

    # ---- addressing ---------------------------------------------------------
    def __getattr__(self, name):
        """Any knob, by its own name, from whichever group declares it.

        Only reached when normal lookup fails, so it costs nothing for the six
        group names and the methods. It must raise `AttributeError` for anything
        unknown -- `copy`, `pickle` and `pytest` all probe for dunders.
        """
        group = _OWNER.get(name)
        if group is None:
            raise AttributeError(
                f"{name!r} is not a knob of any settings group; the groups are "
                f"{', '.join(GROUPS)}")
        return getattr(object.__getattribute__(self, group), name)

    def flat(self):
        """Every field of every group, by name -- the input `with_()` rebuilds from."""
        out = {}
        for g in GROUPS:
            out.update(asdict(getattr(self, g)))
        return out

    def with_(self, **kw):
        """A copy with these knobs changed, RE-RESOLVED and RE-CHECKED.

        This is what `dataclasses.replace(cfg, ...)` used to be, and it has to be
        a method now: replacing one field of a six-part object would leave the
        other five derived from the old value. It re-runs `_resolve()`, so
        `cfg.with_(const_k=39)` moves `code_k`, `code_t` and every slug with it,
        exactly as `replace()` did through `__post_init__`.
        """
        unknown = sorted(k for k in kw if k not in _OWNER)
        if unknown:
            raise guards.refusal("unknown-knob",
                f"with_(): {', '.join(unknown)} is not a knob of any settings "
                f"group. The groups are {', '.join(GROUPS)}.")
        flat = self.flat()
        # A fact the HELD DESIGN declares is re-derived when this call can move
        # the held design, unless the call overrides it by name.
        if any(k in kw for k in DESIGN_AXIS_KNOBS):
            for f in DERIVED_FROM_HELD_DESIGN:
                if f not in kw:
                    flat[f] = None
        return _build({**flat, **kw})

    # ---- what the arms are -------------------------------------------------
    @property
    def recon_optimizer(self):
        """TRUE, ALWAYS -- and no longer a knob (EnvReorganisation phase 3).

        `RECON_OPTIMIZER` chose between Task 3 (one fixed mapping billed to
        every arm) and Task 4 (the reconstruction arm re-mapped for the
        reduced width). Since prompt_7 Phase B every placement that is a
        DISTINCT CHIP gets its own mapping, so the fixed-mapping reading has
        nothing left to select and the knob was always `True` in practice. It
        is a property rather than a field so the configuration RECORD loses
        the key instead of carrying a constant, and so `with_()` refuses it as
        "not a knob".

        The Task 3 branches that read it are still here, and still read as
        they did; phase 6 deletes them with the abstract `recon` arm.
        """
        return True

    @property
    def bar_arms(self):
        """The ABSTRACT arms this run draws, in bar order.

        `ECC_APPROACHES` may name a placement (`recon2`), but the BAR it lands
        in is one of `baseline` / `embedded` / `recon` until phase 6 teaches
        `report/sweep.py` to draw a bar per placement. So any `reconN` implies
        the `recon` bar, and this -- not `approaches` -- is what
        `build_stacks()` and `draw_panel()` walk. With env.sh's default
        `baseline embedded recon` the two lists are identical, which is why no
        number moves.
        """
        named = set(self.approaches)
        if named & set(RECON_PLACEMENT_APPROACHES):
            named.add("recon")
        return [a for a in BAR_ARMS if a in named]

    @property
    def layer_scope(self):
        """`full`, or `one-layer`/`two-layer`/`N-layer` when layers are selected.

        The plan requires that a development result can never be mistaken for a
        full-model result, so this string goes into the results namespace, the
        raw-cache path and the result JSON. There is no way to write a
        two-layer number into a full-model file.
        """
        n = len(self.layers)
        if not n:
            return "full"
        return {1: "one-layer", 2: "two-layer"}.get(n, f"{n}-layer")

    @property
    def layer_slug(self):
        """Filesystem-safe layer identity for the namespace.

        Layer names are the STABLE names from the workload file
        (`layer4.1.conv2`, not `layer_17`), so a selection survives a
        regenerated workload and can be quoted in a paper. Dots and slashes
        become underscores; a selection too long for a path is truncated and
        disambiguated with a hash of the full list, which is still
        deterministic.
        """
        if not self.layers:
            return "layers-full"
        safe = ["".join(ch if (ch.isalnum() or ch in "-") else "_" for ch in name)
                for name in self.layers]
        body = "__".join(safe)
        if len(body) <= 80:
            return f"layers{len(self.layers)}__{body}"
        digest = hashlib.sha1("|".join(self.layers).encode()).hexdigest()[:8]
        return f"layers{len(self.layers)}__{body[:70]}__{digest}"

    @property
    def precision_slug(self):
        """`w8a8` plus the accumulator width only when one is FORCED.

        A default run leaves the accumulator paper-native, so the width differs
        per architecture and belongs in the JSON, not the path. A sensitivity
        run that forces a common width must not collide with it, hence the tag.
        """
        base = f"w{self.weight_bits}a{self.activation_bits}"
        return base if self.acc_bits_override is None else f"{base}acc{self.acc_bits_override}"

    @property
    def mapper_slug(self):
        """Everything about the SEARCH that could change which mapping is returned."""
        seed = "none" if self.mapper_seed is None else str(self.mapper_seed)
        cap = "none" if self.mapper_search_size is None else str(self.mapper_search_size)
        return (f"map-{self.opt_metric}-vic{self.victory}{self.victory_scaling[0]}"
                f"-{self.mapper_algorithm}-seed{seed}"
                f"-ss{cap}-perm{self.mapper_max_permutations}")

    def mapper_settings(self):
        """The mapper configuration, for the mapping fingerprint and the JSON."""
        return {
            "optimization_metric": self.opt_metric,
            "victory_condition_base": self.victory,
            "victory_scaling": self.victory_scaling,
            "algorithm": self.mapper_algorithm,
            "timeout": self.mapper_timeout,
            "search_size": self.mapper_search_size,
            "max_permutations_per_if_visit": self.mapper_max_permutations,
            "num_threads": self.mapper_threads,
            "seed": self.mapper_seed,
            "seed_supported": False,
            "seed_note": (
                "timeloop-mapper v4 exposes no random seed; pytimeloop's Mapper "
                "spec has no `random_seed` attribute. ECC_MAPPER_SEED is RECORDED "
                "so a result says which value was asked for, but it does not make "
                "the search deterministic. What does bound it is the thread count "
                "and victory condition, both of which are in the fingerprint -- "
                "so pin ECC_MAPPER_THREADS for a reproducible run rather than "
                "leaving it at 'all cores', which differs per machine."),
        }

    @property
    def stem(self):
        """The ONE output name for this run. Fixed per sweep, by design.

        ECC_MAC_PJ_OVERRIDE does NOT change the name (decided 2026-09-09): the
        cited MAC cost is the primary denominator and `ReconSweep.png` is drawn
        under it, with the value and its citation in the subtitle and the
        manifest. Briefly (the same day) an override run carried a `__mac<pJ>`
        suffix so the ERT-denominator figure could not be overwritten; that
        figure is now the sensitivity row of FINDINGS 7.3, not a file.

        ONE EXCEPTION, and it is the development mode. A selected-layer run
        appends its layer scope, so `ArchitectureSweep.png` ALWAYS means the
        full model and a three-layer sweep lands beside it as
        `ArchitectureSweep__layers3__<names>.png` rather than on top of it.

        Without this a three-layer figure and a full-model figure are the same
        file with no visible difference between them -- and a figure is a
        result, so the rule that development results must never be confusable
        with full-model results applies to it too. The three-names rule still
        holds for what it was written about: full-model runs at different
        constants overwrite in place, and the manifest beside them records
        which constants produced the file.
        """
        if self.stem_override:
            # ECC_STEM forces ONE output name, deliberately overriding the
            # two disambiguating suffixes below. env.sh section 8 sets it
            # so an architecture sweep always lands on `ArchitectureSweep.png`
            # whether it drew one model or a panel per model. The cost is
            # real: a one-model and a two-model sweep then overwrite each
            # other, and only the manifest beside the figure records which
            # is on disk. Do not set it by hand for a run with ECC_LAYERS --
            # that is exactly the case the layer suffix exists to protect.
            # ONE sanctioned exception (prompt_6 9): env.sh section 8 keeps
            # `ReconSweep_optimiser__<model>` fixed on a layer-scoped run,
            # because that study IS one layer; the scope is in the manifest and
            # the title. The model suffix (2026-09-11) is what keeps two
            # networks' figures apart.
            return self.stem_override
        if self.experiment == "diagnose":
            return "diagnose"
        if self.experiment == "recon":
            # The placement study has its own axis -- WHERE the encoder sits --
            # so it has its own name and cannot land on a sweep's figure.
            base = "ReconSweep"        # a fixed name; ECC_RECON_STEM went 2026-09-14
            if self.recon_optimizer:
                # TASK 4 OWNS ITS OWN NAME, on a layer-scoped run too. Its bars
                # come from a mapping solved against N/K more weight room, so
                # they are not comparable with the fixed-mapping figure and
                # must never overwrite it. env.sh section 8 appends the same
                # suffix for a whole-model run (where ECC_STEM is set and the
                # branch above returns early), so the two agree -- this is the
                # ECC_LAYERS case, which env.sh deliberately leaves nameless so
                # the layer scope lands in the name.
                base = f"{base}_optimiser"
            return base if not self.layers else f"{base}__{self.layer_slug}"
        base = SWEEP_STEMS[self.sweep]
        if self.experiment == "panels":
            # The panel models are IN the name, so a two-model figure can never
            # land on top of the single-model `ArchitectureSweep.png`, and two
            # different model pairs are two different files.
            base = f"{base}__panels__{'__'.join(self.panel_models)}"
        return base if not self.layers else f"{base}__{self.layer_slug}"

    @property
    def swept_axis(self):
        return {"bch": "code-strength", "model": "model", "arch": "architecture",
                # The two POINT_SWEEPS hold all three lists, so neither names
                # one of them. `fix` has no x axis at all -- the bars ARE the
                # comparison -- and `area` walks the buffer-depth ladder one
                # mapped depth at a time.
                "fix": "the held point",
                "area": "buffer depth"}[self.sweep]

    @property
    def swept_values(self):
        """The x axis, in order.

        Under `fix` that is the ARMS (plan 3.1: "plots exactly what
        ECC_APPROACHES names"), and under `area` the ONE depth this
        configuration is evaluated at -- the ladder is a set of runs, one
        `ECC_WEIGHT_DEPTH_SCALE` each, not a list inside one result.
        """
        return {"bch": self.sweep_ks, "model": self.models, "arch": self.archs,
                "fix": list(self.approaches),
                "area": [self.weight_depth_scale]}[self.sweep]

    @property
    def held(self):
        """[(axis, value)] for the two axes this run holds fixed -- for titles.

        A POINT_SWEEP (`fix`, `area`) holds all THREE, and drops the
        architecture from this list anyway: the study is OF that design, and
        the heading, the panel row and the figure title each name it already.
        What `held` is for is the axes that are not the subject, so it stays
        two entries on every axis and every consumer keeps working.
        `swept_axis` is what says there is no sweep.
        """
        pairs = [("architecture", self.arch_label(self.const_arch)),
                 ("model", self.const_model),
                 ("code", f"BCH({self.code_n},{self.code_k}) t={self.code_t}")]
        drop = {"arch": 0, "model": 1, "bch": 2, "fix": 0, "area": 0}[self.sweep]
        return [p for i, p in enumerate(pairs) if i != drop]

    @property
    def weights_per_codeword(self):
        """WHOLE weights carried by one codeword message field.

        Integer floor. `code_k / weight_bits` is 6.375 at BCH(63,51) over 8-bit
        weights, and 6.375 weights is not a thing a codeword can hold -- a
        weight cannot straddle a codeword boundary and still be correctable on
        its own. The 3 leftover message bits are padding, counted in parity.py.
        """
        return self.code_k // self.weight_bits

    @property
    def emb_weights_per_cw(self):
        """Weights per codeword used to COUNT embedded-arm decodes.

        The EMBEDDED layout, `n / weight_bits` = 7.875 at BCH(63,*) over 8-bit
        weights -- FRACTIONAL, because the embedding pipeline cuts the weight
        bit stream into n-bit chunks and weights straddle codeword boundaries
        by design (`embedded.EmbeddedLayout.weights_per_codeword`).

        It used to default to the BASELINE's `k // weight_bits` (3 at K=30),
        which is right for the baseline -- whose codeword is stored separately
        and packs only WHOLE weights -- and wrong here. prompt_7, 2026-09-12:
        the two layouts are different and their counts must be too.
        `ECC_EMB_WEIGHTS_PER_CW=8` still reproduces the older two-arm scripts.
        """
        if self.emb_weights_per_cw_override:
            return self.emb_weights_per_cw_override
        return EmbeddedLayout(self.code_n, self.code_k,
                              self.weight_bits).weights_per_codeword

    @property
    def parity_frac(self):
        """The IDEAL code-rate overhead, N/K - 1.

        Reported for comparison only. The baseline is charged by `parity.py`,
        which accounts for padding and DRAM word granularity and comes out
        higher (31.25% vs 23.53% at BCH(63,51) over 8-bit weights).
        """
        return self.code_n / self.code_k - 1.0

    @property
    def sram_scale(self):
        """Recon+ on-chip weight fraction, K/N."""
        return self.code_k / self.code_n

    @property
    def weak_overhead(self):
        return (self.weak_n / self.weak_k) if self.weak_enabled else 1.0

    @property
    def global_variant_parts(self):
        """Treatment knobs that change EVERY architecture's result.

        Two kinds live here. The `globals.yaml` knobs go through a file that
        sits above the accelerator container and therefore costs DRAM for every
        design at once. The mapper knobs change what the search returns for
        every design. Either way no architecture can reuse a stock mapping.

        The historical values -- `edp`, victory 500, unscaled -- are spelled as
        the empty treatment, so `ecc_energy_study/outputs/<arch>/multimodel`
        remains reachable and every mapping already on disk is still valid at
        those settings.
        """
        parts = []
        if self.opt_metric != "edp":
            parts.append(f"opt-{self.opt_metric}")
        if self.victory != 500:
            parts.append(f"vic{self.victory}")
        if self.victory_scaling != "none":
            parts.append("vicx")
        if self.mapper_algorithm != "hybrid":
            parts.append(f"alg-{self.mapper_algorithm}")
        if self.mapper_search_size is not None:
            # A capped search is a DIFFERENT search, and a capped run is a
            # development run by construction. Its mappings must never be
            # readable as an uncapped result, so they get their own cache.
            parts.append(f"ss{self.mapper_search_size}")
        if self.mapper_max_permutations != 16:
            parts.append(f"perm{self.mapper_max_permutations}")
        if self.mapper_timeout != 10000:
            parts.append(f"to{self.mapper_timeout}")
        if self.force_technology:
            parts.append(f"tech{self.force_technology}")
        if self.dram_depth != 1048576:
            parts.append(f"dramdepth{self.dram_depth}")
        # THE CLOCK IS NOT GLOBAL ANY MORE (prompt_7 C1.5). `ECC_ARCH_CLOCK_MHZ`
        # gives each design its own rate, so the `clk` slug part is appended by
        # `archs.effective_variant()`, which knows which design it is talking
        # about. Left here it would label every design with whichever rate the
        # STUDY default happened to be.
        if self.noc_enabled:
            # A costed interconnect is a different architecture to the mapper.
            # Every cache built before archs/_shared/noc.yaml existed was built
            # with a free NoC and must never be read back as this.
            noc = "noc"
            if self.noc_wire_pj_per_bit_mm is not None:
                noc += f"w{self.noc_wire_pj_per_bit_mm:g}"
            if self.noc_router_pj is not None:
                noc += f"r{self.noc_router_pj:g}"
            if self.noc_pe_latch_pj is not None:
                noc += f"l{self.noc_pe_latch_pj:g}"
            if self.noc_scale != 1.0:
                noc += f"x{self.noc_scale:g}"
            parts.append(noc)
        return parts

    def victory_for(self, levels):
        """Mapper effort for an architecture whose loop nest is `levels` deep.

        Timeloop's random search terminates a thread after `victory_condition`
        consecutive non-improving mappings. The number of candidate mappings
        grows combinatorially with the number of loop levels, so a flat setting
        searches a deep hierarchy less thoroughly than a shallow one -- and
        then reports the difference as an architecture result. Doubling per
        level beyond the reference depth is a heuristic, not a proof of equal
        coverage: the way to CONFIRM convergence is to raise ECC_VICTORY and
        check that the totals do not move.
        """
        if self.victory_scaling == "none" or not levels:
            return self.victory
        scale = min(VICTORY_MAX_SCALE,
                    2 ** max(0, int(levels) - VICTORY_REFERENCE_LEVELS))
        return int(self.victory * scale)

    @property
    def per_arch_variant_parts(self):
        """Treatment knobs that may or may not touch a given architecture.

        Two of them:

        * `paper` fidelity only applies to an architecture that actually has an
          `archs/<name>/arch_paper.yaml`. One without keeps its stock slug.
        * Forcing the datawidth is a no-op on a design already declared at the
          weight width, so that architecture keeps its existing mapper cache
          and only the mis-declared designs are re-mapped.

        `archs.effective_variant()` decides both per architecture, by looking
        for the paper file and by diffing the patched YAML.

        THE CLOCK JOINED THEM IN prompt_7 C1.5. `ECC_ARCH_CLOCK_MHZ` runs
        Eyeriss v1 at its published 200 MHz and leaves every other design at
        the study default, so `clk` is a treatment that touches SOME designs --
        the same shape as `paper` fidelity, and it is decided the same way.
        This property answers for the design the CONFIGURATION is about (the
        first, and the placement study pins it to one); `effective_variant()`
        answers per design and is what a cache path is built from.
        """
        parts = []
        clk = self.cycle_seconds_for(self.archs[0]) if self.archs else self.global_cycle_seconds
        if clk != "1e-9":
            parts.append(f"clk{clk}")
        if self.arch_fidelity != "stock":
            parts.append(self.arch_fidelity)
        if self.force_datawidth:
            parts.append(f"dw{self.force_datawidth}")
        if self.acc_bits_override is not None:
            # The common-accumulator SENSITIVITY study rewrites psum levels, so
            # it is a different architecture to the mapper and must never share
            # a cache with the paper-native primary result.
            parts.append(f"acc{self.acc_bits_override}")
        if self.weight_capacity_scale != 1.0:
            # A dilated (or shrunk) weight buffer is a different architecture
            # to the mapper, and the whole point of Task 4 is to DIFF the two
            # mappings -- so they must never land in one cache directory.
            # `archs.effective_variant()` drops this again on a design where
            # the scale rewrites nothing.
            parts.append(f"wcap{self.weight_capacity_scale:g}"
                         + ("-shared" if self.weight_capacity_scope == "shared" else ""))
        # THE WIDTH TABLE is in every slug because it is applied to every run
        # -- it is a property of the study, not a knob, so there is no
        # configuration in which it is off and nothing to make conditional.
        # MUST stay in step with `archs.effective_variant()`, same part, same
        # spelling, same POSITION. It also marks the boundary in `ls`: a
        # directory without it predates 2026-09-12 and was mapped on the
        # published word shape.
        parts.append(f"wt{self.width_table().base_width(self.weight_bits)}"
                     + (f"x{self.weight_width_glb_mult}"
                        if self.weight_width_glb_mult != 4 else ""))
        if self.weight_datawidth is not None:
            # A narrower on-chip weight IS a different architecture to the
            # mapper -- more values per word, so a different block size and a
            # different mapspace. The two arms of a prompt_2 pair are exactly
            # this and nothing else.
            parts.append(f"wdw{self.weight_datawidth}"
                         + ("-" + "+".join(self.weight_datawidth_levels)
                            if self.weight_datawidth_levels else ""))
        if self.mapspace_constrain:
            # A constrained loop nest is a different MAPSPACE and a different
            # DATAFLOW. MUST stay in step with `archs.effective_variant()`:
            # when only one of the two knew about a treatment, one mapping was
            # written under TWO slugs (measured 2026-09-10, `mcons` missing
            # here) -- harmless only because the fingerprint is
            # content-addressed on the patched YAML, but it doubles the cache
            # and makes the tree advertise architectures that do not exist.
            parts.append("mcons")
        if self.weight_factor_relax:
            # A relaxed dataflow constraint is a different MAPSPACE, so it is a
            # different architecture to the mapper and gets its own cache.
            parts.append("wrelax")
        part = self.mapper_arm_slug()
        if part is not None:
            # prompt_6 RULE 4.4.5, defence 1: recon2 and recon4 declare
            # byte-identical YAML and differ ONLY in the ERT, so without this
            # both would occupy one directory and the second map would
            # overwrite or skip the first. Legible in `ls`. MUST stay in step
            # with `archs.effective_variant()`.
            parts.append(part)
        return parts

    def mapper_arm(self):
        """The MAPPER arm this configuration solves -- `recon.mapper_arm_spec()`'s
        record -- or None for the reference arm (prompt_7 B2).

        Every distinct chip has one, ERT bump or not. `ert_arm()` is the
        subset that has one.
        """
        key = getattr(self, "recon_ert_arm", "reference")
        if key in ("", "reference", None):
            return None
        return arms_mod.mapper_arm_spec(self.archs[0], key, self)

    def ert_arm(self):
        """The ERT arm this configuration maps -- `recon.ert_arm_spec()`'s
        record -- or None when this arm has no ERT bump (prompt_6 8.3).

        None now means two different things -- the reference arm, and a mapper
        arm whose boundary is not ERT-injectable (R1, R3) -- and both are
        right for every caller: `archs.ert_bump()` must produce no bump for
        either, because neither declares one.
        """
        spec = self.mapper_arm()
        if spec is None or spec["ert"] is None:
            return None
        return {"placement": spec["placement"], "key": spec["key"],
                "level": spec["ert"]["level"], "counter": spec["ert"]["counter"],
                "action": spec["ert"]["action"],
                "narrow_levels": spec["narrow_levels"]}

    def mapper_arm_slug(self):
        """The cache-slug component that keeps this arm's directory its own,
        or None for the reference arm.

        `ert-<key>-<level>-<action>` where there IS a bump -- byte for byte
        prompt_6's spelling, so every directory on disk stays a cache hit --
        and `arm-<key>` where there is not. Without the second, R1's slug
        would be the reference's (its patched YAML is the reference's until
        Phase C1.2) and prompt_7 B2's gate 2 would fail on a real collision.
        """
        arm = self.ert_arm()
        if arm is not None:
            return self.ert_slug(arm)
        spec = self.mapper_arm()
        return None if spec is None else f"arm-{spec['key']}"

    @staticmethod
    def ert_slug(arm):
        """`ert-<placement key>-<level>-<action>`, the cache-directory part."""
        return f"ert-{arm['key']}-{arm['level']}-{arm['action']}"

    def width_table(self, arch=None):
        """THE WIDTH TABLE of `arch` (default: the held design) --
        `archs/<name>/widths.yaml`, or THE RULE where it declares none."""
        if arch is None:
            arch = self.archs[0] if self.archs else None
        return design_mod.width_table(arch) if arch else widths.WidthTable.rule()

    def leakage_nw_for(self, arch=None):
        """The standby densities to price `arch` at (EnvReorganisation 6.4).

        `leakage_nw` is the record field, derived from the HELD design and
        overridable by `with_()`; it answers for that design and for a caller
        with no design in hand. Any OTHER design of a multi-design run is
        priced from its own `design.yaml`, falling back to the record.
        """
        if arch is None or not self.archs or arch == self.archs[0]:
            return self.leakage_nw or {}
        return design_mod.leakage_nw(arch) or self.leakage_nw or {}

    def cycle_seconds_for(self, arch):
        """THIS DESIGN's clock period, as the string `globals.yaml` carries.

        prompt_7 C1.5, Issue 14. `ECC_GLOBAL_CYCLE_SECONDS` is ONE number for
        every design and it was 1 GHz, while Eyeriss v1 silicon runs at
        200 MHz; pairing a real chip's ABSOLUTE off-chip MB/s with a 5x faster
        model clock makes the modelled chip ~5x more memory-starved than the
        one the paper describes.

        THE ONLY PLACE MHz BECOMES SECONDS. env.sh section 4's TRAP is that
        a per-cycle constant converted twice, or not at all, is a silent 5x on
        every standby and idle term; keeping the inversion here means no
        caller can do either. A design that declares no `clock_mhz` in its
        design.yaml keeps `global_cycle_seconds` unchanged, so the table ADDS
        designs rather than replacing the study default. (The table was
        env.sh's `ECC_ARCH_CLOCK_MHZ` until 2026-09-14.)

        Returned as a STRING because that is what the fingerprint hashes and
        what globals.yaml prints; `repr` of a float would make `5e-09` and
        `5.0e-09` two different architectures.
        """
        mhz = (self.arch_clock_mhz or {}).get(arch)
        if not mhz:
            return self.global_cycle_seconds
        if float(mhz) <= 0:
            raise guards.refusal("clock-positive",
                f"clock_mhz={mhz} for {arch} (archs/{arch}/design.yaml, or a "
                f"with_() override): a clock rate must be positive")
        secs = 1.0 / (float(mhz) * 1e6)
        # A DESIGN THAT IS ALREADY AT THE STUDY DEFAULT KEEPS THE DEFAULT'S
        # EXACT SPELLING. Seven of the eight designs declare 1000 MHz, which
        # is `global_cycle_seconds` itself -- and `%.6g` of
        # 1e-9 is the string "1e-09" while env.sh writes "1e-9". Two spellings
        # of one number are two architectures to the fingerprint and two
        # directory names to the cache, so declaring a design at the rate it
        # already ran at would have colded it for nothing.
        try:
            if secs == float(self.global_cycle_seconds):
                return self.global_cycle_seconds
        except (TypeError, ValueError):
            pass
        return f"{secs:.6g}"

    def clock_mhz_for(self, arch):
        """`cycle_seconds_for()` back in MHz, for a report line."""
        return 1.0 / (float(self.cycle_seconds_for(arch)) * 1e6)

    def dc_idle_scale(self, arch=None):
        """What `ECC_RECON_IDLE_PJ` must be MULTIPLIED BY at this design's clock.

        env.sh section 4's TRAP, and it is live for the first time in
        prompt_7 C1.5. The DC tables give the reconstruction engine's idle term
        in pJ PER CYCLE, measured at a 1 ns clock
        (`data/dc/BCH_N63_results.json`, `measurement.clock_period_ns = 1.0`).
        It is CLOCK POWER, so a 5 ns cycle burns five times as much of it:

            idle_pJ_per_cycle(T) = idle_pJ_per_cycle(1 ns) x T / 1 ns

        At Eyeriss v1's published 200 MHz that is x5 -- so until C1.5 gave the
        design its own clock, this factor was 1.0 on every run and the code
        that should apply it had never had to. It is NOT applied to the
        INCREMENTAL term: that is switching energy per codeword, which is
        CV^2 and does not depend on how long the cycle is.

        `ECC_LEAKAGE_NW` is the opposite case and must NOT be rescaled -- it is
        declared in nW, i.e. POWER, and the period is applied at the point of
        use. That is why the two are declared in different units.

        THE FACTOR LIVES HERE and `ecc.load_recon_terms()` applies it at one
        site, so no caller can apply it twice or not at all.
        """
        if arch is None:
            archs = self.archs or []
            arch = archs[0] if len(archs) == 1 else None
        secs = (float(self.cycle_seconds_for(arch)) if arch
                else float(self.global_cycle_seconds))
        return secs / (DC_MEASUREMENT_CLOCK_NS * 1e-9)

    def dram_items_per_cycle_for(self, arch):
        """`ECC_DRAM_BANDWIDTH_MBPS` as ITEMS per cycle of THIS design's clock,
        or None for unlimited.

        env.sh section 4 states the conversion and there are exactly two
        implementations of it -- this one, which the ARCHITECTURE declares
        (prompt_7 C1.1), and `latency_post.offchip_items_per_cycle()`, which
        the evaluator charges. They must agree; `tests/test_phase_c.py` asserts
        that they do, because a mapper optimising against one limit and a
        roofline reporting another is two timing models for one bus.
        """
        if not self.dram_bandwidth_mbps:
            return None
        bytes_per_item = self.weight_bits / 8.0
        if bytes_per_item <= 0:
            return None
        return (float(self.dram_bandwidth_mbps) * 1e6
                * float(self.cycle_seconds_for(arch)) / bytes_per_item)

    def _arm_for(self, arch):
        """This configuration's mapper arm ON `arch`, or None.

        `ECC_RECON_ERT_ARM` names the arm of ONE mapper job on ONE design
        (`__post_init__` refuses it otherwise), so a request about any OTHER
        design is the reference arm rather than an error -- that is what
        `archs.arch_fingerprint()` asks when a sweep enumerates designs.
        """
        if not self.archs or arch != self.archs[0]:
            return None
        return self.mapper_arm()

    def arm_bw_factors_for(self, arch):
        """`{level: {factor, timing, kind, dataspace, why}` -- the per-dataspace
        bandwidth scale THIS arm declares on `arch` (prompt_7 C1.2), or `{}`.

        Empty for the reference arm and empty with `ECC_RECON_BW_SCALE=0`,
        which is what reproduces the pre-Phase-C architecture byte for byte.
        """
        if not self.recon_bw_scale:
            return {}
        spec = self._arm_for(arch)
        if spec is None or spec["placement"] is None:
            return {}
        return arms_mod.arm_bw_factors(spec["placement"],
                                     placements.stages_for(arch, self), self)

    def onchip_bw_bitaware_factor(self):
        """`weight_bits / q` -- how much MORE a narrowed level's port delivers
        per cycle when it is priced in BITS rather than items (prompt_7 C1.3).

        1.0 when the knob is off or the arm narrows nothing. At q=5 it is 1.6,
        so `filter_glb`'s declared 16 items/cycle becomes 25.6 -- and that
        level is what caps every `fc` layer at 9.52% PE utilisation
        (prompt_7 4.5, A.7).

        IT IS NOT ROUNDED. A declared bandwidth is a rate, not a word count,
        and rounding 25.6 down to 25 would hand the reference arm a 2.4%
        advantage that no wire has.
        """
        if not self.onchip_bw_bitaware or self.weight_datawidth is None:
            return 1.0
        return float(self.weight_bits) / float(self.weight_datawidth)

    @property
    def arch_variant_slug(self):
        """The full treatment slug, used by the raw cache and the mapper cache.

        The empty treatment is spelled 'stock' and maps to the historical
        `ecc_energy_study/outputs/<arch>/multimodel` path, which keeps every
        already-computed mapping valid.
        """
        parts = self.global_variant_parts + self.per_arch_variant_parts
        return "stock" if not parts else "__".join(parts)

    @property
    def code_slug(self):
        return f"bch{self.code_n}_{self.code_k}"

    def arch_label(self, arch):
        return ARCH_LABELS.get(arch, arch) if self.nice_labels else arch

    def title_suffix(self, include_mac=True):
        bits = [f"{self.weight_bits}-bit weights"]
        if self.sweep != "bch":
            bits.append(f"BCH({self.code_n},{self.code_k}) t={self.code_t}")
        if self.force_technology:
            bits.insert(0, self.force_technology)
        if self.weak_enabled:
            bits.append(f"weak BCH({self.weak_n},{self.weak_k})")
        if include_mac and self.mac_pj_override is not None:
            # A figure drawn under the override must say so on itself: the
            # denominator of every percentage on it is not the ERT's.
            bits.append(self.mac_line())
        if self.title_note:
            bits.append(self.title_note)
        return "  ·  ".join(bits)

    def mac_citation(self):
        """`(short, long)` citation for ECC_MAC_PJ_OVERRIDE, from provenance.yaml.

        A value listed under `mac_energy_pj.candidates` there carries its
        source; any other value is labelled an uncited override, because a
        number that changes every percentage in the study may not travel
        without saying where it came from.
        """
        v = self.mac_pj_override
        if v is None:
            return ("Accelergy ERT", "Accelergy ERT (intmac = aladdin_multiplier "
                                     "8x8 + aladdin_adder 20b, Library plug-in table "
                                     "32b/40nm scaled)")
        for c in mac_candidates():
            if c.get("pj") is not None and abs(float(c["pj"]) - v) < 1e-9:
                return (str(c.get("short", c.get("source", "cited"))),
                        str(c.get("source", "")))
        return ("UNCITED override", "ECC_MAC_PJ_OVERRIDE set to a value not listed in "
                                    "archs/_shared/provenance.yaml mac_energy_pj")

    def mac_line(self, ert_pj=None):
        """The one line every figure carries about the MAC cost.

        `ert_pj` is the per-MAC energy actually read off the ERT (the raw record
        knows it; the config does not), shown when given.
        """
        if self.mac_pj_override is None:
            shown = f"{ert_pj:.4f} pJ" if ert_pj else "ERT value"
            return f"MAC {shown}/op (Accelergy ERT, Aladdin table scaled)"
        short, _long = self.mac_citation()
        was = f", ERT was {ert_pj:.4f}" if ert_pj else ""
        return (f"MAC {self.mac_pj_override:g} pJ/op = ECC_MAC_PJ_OVERRIDE "
                f"({short}{was})")

    def figure_title(self):
        """<held axes>  .  <settings>  .  <swept axis> sweep.

        A selected-layer run says so ON THE FIGURE. A picture gets separated
        from its directory the moment someone drops it into a slide, so the
        caveat has to travel with the pixels and not only with the path.

        The code is dropped from the head whenever `title_suffix` already
        carries it -- that is, on every sweep except the BCH one, where `held`
        drops it instead. Printing BCH(N,K) twice in one heading reads like two
        different settings. `panel_title` drops it for the same reason.
        """
        head = "  ·  ".join(v for k, v in self.held
                            if not (k == "code" and self.sweep != "bch"))
        title = f"{head}  ·  {self.title_suffix()}  ·  {self.swept_axis} sweep"
        if self.layers:
            title += (f"\nDEVELOPMENT RUN — {self.layer_scope} only: "
                      f"{', '.join(self.layers)}  (not a full-model result)")
        return title

    def recon_placements_for(self, arch, warn=True):
        """Which boundaries to draw for `arch`: `[]` means every one it defines.

        THE SOURCE IS `ECC_APPROACHES` since EnvReorganisation phase 3. The
        per-architecture `ECC_RECON_PLACEMENTS` table went with
        `ECC_RECON_MODELING`: the design's own `placements.yaml` already says
        which boundaries it HAS, so the only thing left for the user to say is
        which of them to compare -- and that is a bar, which is what
        `ECC_APPROACHES` is for.

            baseline embedded recon              every placement this design
                                                 declares  (env.sh's default)
            baseline embedded recon2 recon4      exactly those two

        A NAMED BOUNDARY THIS DESIGN DOES NOT DECLARE IS WARNED AND DROPPED,
        never refused (plan 3.1): `eyeriss_v2_like_wglb` has four boundaries
        and `eyeriss_like_wglb` five, and one `ECC_APPROACHES` has to be legal
        for both -- so a five-name list draws five bars on one and four on the
        other. `warn=False` for a caller that only wants the set (a launcher
        enumerating chips prints its own bill).
        """
        named = [a for a in self.approaches if a in RECON_PLACEMENT_APPROACHES]
        if not named:
            return []
        declared = [p.key for p in placements.PLACEMENTS.get(arch, ())]
        keep = [k for k in named if k in declared]
        missing = [k for k in named if k not in declared]
        if missing and warn:
            print(f"  [skip] {arch} does not declare {', '.join(missing)} -- "
                  f"it has {', '.join(declared) or 'no placement at all'}. "
                  f"Drawing {len(keep)} bar(s), not {len(named)}.")
        return keep

    def recon_placement_bars(self, arch, warn=True):
        """The boundaries this run DRAWS on `arch` -- resolved, never empty.

        `recon_placements_for()` answers what was ASKED, and `[]` there means
        "every placement this design declares", which is what the abstract
        `recon` in `ECC_APPROACHES` means. This answers what will actually be
        on the figure, so a banner row and a result record can NAME the bars
        instead of saying "all" -- read off the design, which is the only
        place that knows.
        """
        named = self.recon_placements_for(arch, warn=warn)
        if named:
            return named
        return [p.key for p in placements.PLACEMENTS.get(arch, ())]

    def mapping_regime_line(self):
        """WHICH MAPPING REGIME the placement bars come from -- the claim a reader
        has to be able to check on the figure (prompt_6 9 names the arms).

        Since prompt_7 B2 the arms are the DISTINCT CHIPS -- `datawidth: q` x
        ERT bump x declared bandwidth scale -- not the ERT-injectable
        boundaries, so this line names the chips and says that WHICH PLAN each
        bar was billed from is on the bar's own record. It cannot say it here:
        that depends on which arms are mapped, which is a property of the
        cache and not of the configuration.
        """
        if getattr(self, "recon_ert_aware", False):
            try:
                arms = sorted({a.key for d in self.archs
                               for a in arms_mod.mapper_arms(d, self)
                               if a.placement is not None})
            except Exception:                    # the title must never kill a run
                arms = []
            return (f"ERT-AWARE MAPPING, ONE PLAN PER BOUNDARY: {len(arms) + 1} distinct "
                    f"chips -- reference"
                    + (f" + {', '.join(arms)}" if arms else "")
                    + f" -- each (datawidth q on the levels it narrows) x (its ERT bump) x "
                      f"(its declared bandwidth scale). EVERY BAR NAMES THE PLAN IT WAS "
                      f"BILLED FROM in `billed_from` on its own record, with whether that "
                      f"plan's storage geometry is its own")
        if self.recon_optimizer:
            return ("RECONSTRUCTION-AWARE MAPPING (Task 4: the reconstruction bars come "
                    "from a second mapping solved with more weight room)")
        return "FIXED MAPPING (evaluator only, the mapper was not re-run)"

    def mapping_regime_tag(self):
        """The regime in a few words, for the ONE-LINE heading. The full
        statement, arms and all, is `mapping_regime_line()`, which travels in
        the manifest beside the figure (`recon_caveats`)."""
        if getattr(self, "recon_ert_aware", False):
            return "ERT-aware mapping"
        if self.recon_optimizer:
            return "reconstruction-aware mapping (Task 4)"
        return "fixed mapping"

    def recon_scope(self):
        """`resnet18`, or `resnet18  ·  layer3.0.conv1` on a layer-scoped run.

        The optimiser figure has ONE path whatever the scope (env.sh section 8,
        prompt_6 9), so the scope has to be on the pixels: a picture gets
        separated from its manifest the moment it is dropped into a slide.
        """
        if self.layers:
            return f"{self.models[0]}  ·  {', '.join(self.layers)}"
        return self.models[0]

    def recon_caveats(self, mac_ert_pj=None):
        """What the placement heading said below its first line until
        2026-09-11, one string per line, for the manifest (`title_caveats`).

        Seven lines of heading made the figure two and a half times the width
        of its axes, and nobody reads a caveat set as a title. The heading is
        now `recon_title()`'s one line and these are RECORDED instead of drawn
        -- every one names a denominator or a regime that decides what the bars
        mean, so none may be dropped. `mac_ert_pj` is the per-MAC energy the
        raw record read off the ERT, so the MAC line can state the number the
        denominator rests on.
        """
        lines = [f"Reconstruction-boundary placement  ·  {self.mapping_regime_line()}"]
        lines += self.dram_model_line().split("\n")
        lines.append(f"{self.mac_line(mac_ert_pj)}  ·  the MAC cost is the "
                     f"denominator of every percentage on this figure")
        lines += self.reporting_rules()
        if self.layers:
            lines.append(f"DEVELOPMENT RUN — {self.layer_scope} only: "
                         f"{', '.join(self.layers)}  (not a full-model result)")
        # section 6.4 rule 2, the other half: an override travels with the FIGURE
        # as well as the manifest, because a picture gets separated from its
        # manifest the moment it is dropped into a slide. Nothing is appended on
        # a run that overrides nothing, which is every run published so far.
        for o in guards.overrides():
            lines.append(f"ABLATION — guard `{o['guard']}` (tier {o['tier']} "
                         f"{o['tier_name']}) was overridden by ECC_ALLOW: "
                         f"{o['refused'].splitlines()[0]}")
        return lines

    def reporting_rules(self):
        """prompt_7 section 12 -- the six rules that must travel with every table.

        They are not defects and not caveats about this run: they are facts
        about the MODEL that a reader cannot infer from the figure, and each one
        decides what a bar means. `recon_caveats()` carries them into the
        manifest's `title_caveats` beside every placement figure. R-3 replaces
        the former "Issue 9"; it is resolved, not open.
        """
        gate = float(getattr(self, "recon_clock_gating_pct", 0.0))
        rules = [
            "R-1 LATENCY IS FLAT ACROSS BOUNDARIES BY CONSTRUCTION. Every "
            "placement's `reduced` set contains `dram`, so every placement gets "
            "the same off-chip relief. A difference between two bars on a "
            "latency figure is mapping noise unless it exceeds the per-shape "
            "PE-count variation reported beside them.",

            "R-2 WHICH LEVEL BINDS IS A PROPERTY OF WHAT THE ARCHITECTURE "
            "DECLARES, AND IT CHANGED WITH prompt_7 C1.1. RESTATED 2026-09-13. "
            "BEFORE: no architecture in archs/ declared an off-chip bandwidth, "
            "so Timeloop skipped the DRAM throughput check entirely and only "
            "ifmap_glb (11 of 43 shapes, worst throttling 0.180) and psum_glb "
            "(5 of 43, worst 0.610) ever throttled -- levels carrying INPUTS "
            "and PARTIAL SUMS, which reconstruction cannot touch. That was the "
            "honest headline of Defect 1 and it was never a fact about the "
            "chip: it was the absence of a declared number. NOW: the DRAM "
            "level declares `shared_bandwidth` (ECC_DRAM_BANDWIDTH_MBPS at the "
            "design's own clock) and DRAM BINDS. Measured on resnet18, 12 "
            "shapes, 120 MB/s at 200 MHz = 0.6 items/cycle: DRAM is the "
            "binding level on 21 of 21 layers, worst throttling 0.074, and the "
            "reconstruction arms convert their K/N relief 1:1 into time -- "
            "12.56 % against a 12.56 % ceiling. SO THE CAVEAT INVERTS: the "
            "binding resource is now exactly the one reconstruction relieves, "
            "and the number to report beside any latency claim is HOW BOUND "
            "the design is, because at LPDDR4 speeds the limit never binds and "
            "the saving is 0.000 % (prompt_7 7.2, A.3). filter_glb still never "
            "throttles, and still sits exactly on its declared 16 items/cycle "
            "at the fully-connected layers -- 'never throttles' and 'never "
            "binds' remain different claims. Still a statement about the "
            "CONSTRAINED mapspace in force (see R-5).",

            "R-3 R3'S LATENCY EQUALS R2'S BY CONSTRUCTION, NOT BY MEASUREMENT. "
            "Timeloop has no network timing model (LegacyNetwork::"
            "ComputePerformance() is an empty stub; a network stats block "
            "carries no Cycles and no bandwidth field), archs/_shared/noc.yaml "
            "declares energy coefficients only, and no interconnect bandwidth "
            "is cited anywhere in this study. R3's `reduced` set contains dram "
            "and filter_glb, both real storage levels, so R3 receives exactly "
            "the same latency saving as R2 and R4; what is unmodelled is the "
            "INCREMENTAL benefit of the array multicast carrying reduced-width "
            "words, i.e. the difference between R3 and R2. Report it as "
            "\"R3 = R2 by construction\", never as \"R3 shows 0% latency "
            "benefit\" -- the second claims a mechanism was tested and found "
            "ineffective, and it was not tested. It is unmodelled, not "
            "unfixable: NoC energy is already charged outside Timeloop by "
            "noc_post.py, and latency_post.py could carry a declared NoC "
            "items/cycle limit the same way once one is cited.",

            "R-4 THE MAPPER SEES LEAKAGE; THE REPORT DID NOT. Timeloop's "
            "`Energy:` and `EDP(J*cycle)` -- the mapper's objective -- already "
            "include `Leakage energy (total)`, at ERT prices that are 10^3-10^4 "
            "too low and in three cases exactly 0. Before Phase A parse_stats "
            "discarded that number, so the accelerator's standby energy was "
            "charged to no arm while the reconstruction engines were charged "
            "theirs. Any statement of the form \"leakage is absent from the "
            "model\" is wrong; the correct statement names which side dropped "
            "it. ECC_STATIC_ENERGY=" + ("1: the replacement densities "
            "(ECC_LEAKAGE_NW) are charged to all three arms as the `Standby` "
            "category." if self.static_energy else "0: no arm is charged "
            "standby energy on this figure."),

            "R-5 LABELS. A constrained or relaxed dataflow is a DIFFERENT "
            "ACCELERATOR -- label such bars \"constrained dataflow\" and never "
            "quote them as the published chip. A design whose provenance is "
            "`reference_design` is named after a paper, not published as one: "
            "\"Simba-like (reference design)\".",

            "R-6 build_stacks()'s `recon` ARM IS NOT A PLACEMENT. It is one "
            "point applied to every design at once, at the chip ingress with a "
            "single engine, and no physical boundary does what it does. It must "
            "never be quoted as one of this figure's boundaries.",
        ]
        rules.append(
            f"CLOCK GATING: this figure is ECC_RECON_CLOCK_GATING_PCT={gate:g}. "
            f"The two settings that must be reported side by side are 0 (the "
            f"pessimistic bound: the engine's whole DC idle constant is charged "
            f"every cycle it is switched on) and 99.5 (the measured "
            f"clock/dynamic share of that constant). They differ by ~193x on "
            f"the idle term and they INVERT the ranking of the boundaries.")
        return rules

    def recon_panel_title(self):
        """ONE line for a placement figure with one panel PER ARCHITECTURE.

        The architecture is per-panel here, so unlike `recon_title()` it is not
        in the shared heading -- each panel's own heading names its design. What
        stays shared is what the study holds fixed across the panels: the model
        (and layer scope), the code geometry and the mapping regime. The DRAM
        model and the MAC denominator are `recon_caveats()`, in the manifest.
        """
        return (f"{self.recon_scope()}  ·  {self.title_suffix(include_mac=False)}"
                f"  ·  {len(self.archs)} accelerators, one panel each"
                f"  ·  {self.mapping_regime_tag()}")

    def recon_title(self):
        """ONE line for the placement figure: the fixed point, then the regime.

        Every axis of the study is HELD here except the one that is not an axis
        of the three sweeps at all -- where the reconstruction boundary sits --
        so the heading names the fixed point (design, model and layer scope,
        weight width, code) and the mapping regime, which is the claim a reader
        has to be able to check. Everything the heading carried below that
        until 2026-09-11 is `recon_caveats()`, in the manifest beside the figure.

        `title_suffix()` already carries BCH(N,K) whenever the sweep is not the
        BCH one, and a heading that says it twice reads like two settings.
        """
        return (f"{self.arch_label(self.archs[0]).replace(chr(10), ' ')}"
                f"  ·  {self.recon_scope()}"
                f"  ·  {self.title_suffix(include_mac=False)}"
                f"  ·  {self.mapping_regime_tag()}")

    @property
    def dram_cost_note(self):
        """Where the per-bit DRAM dynamic access energy came from, for the
        figure, the manifest and the result file.

        archs/_shared/provenance.yaml `dram_access_energy` carries the sources.
        """
        v = self.dram_pj_per_bit
        if v is None:
            return ("Accelergy CactiDRAM LPDDR4 as modelled (8 pJ/bit = 512 pJ "
                    "per 64-bit access); below every measured value in the "
                    "literature")
        if abs(v - 20.0) < 1e-9:
            return ("ECC_DRAM_PJ_PER_BIT=20 -- Horowitz, ISSCC 2014, "
                    "\"Computing's Energy Problem\": 32b DRAM read = 640 pJ "
                    "= 20 pJ/bit = 1.28 nJ/64b")
        if abs(v - 40.0) < 1e-9:
            return ("ECC_DRAM_PJ_PER_BIT=40 -- within the 28-45 pJ/bit band "
                    "(FReaC Cache, MICRO 2020; Gebhart et al., MICRO 2012); "
                    "2.56 nJ/64b")
        if abs(v - 8.0) < 1e-9:
            return "ECC_DRAM_PJ_PER_BIT=8 -- Accelergy CactiDRAM LPDDR4 as modelled"
        return (f"ECC_DRAM_PJ_PER_BIT={v:g} -- not one of the values listed in "
                f"provenance.yaml dram_access_energy; UNCITED for this run")

    @property
    def baseline_dram_note(self):
        """What a DRAM bit costs the conventional-ECC baseline, and why.

        archs/_shared/provenance.yaml `dram_access_energy` carries the sources
        for the per-bit constants; the baseline's own value is this study's,
        for an array that also stores the parity and indexes it.
        """
        v = self.baseline_dram_pj_per_bit
        if v is None:
            return ("pre-2026-09-10 model: the baseline is priced at the same "
                    "pJ/bit as the embedded arm and charged the external-parity "
                    "TRAFFIC on top (ECC_BASELINE_DRAM_PJ_PER_BIT unset)")
        return (f"ECC_BASELINE_DRAM_PJ_PER_BIT={v:g} -- the baseline's DRAM array "
                f"also stores the parity and indexes it, so a bit out of it costs "
                f"more than out of the embedded/recon array; the parity itself is "
                f"corrected on the DRAM die and never crosses the datapath")

    @property
    def baseline_dram_line(self):
        """One line for the console header: what each arm actually fetches."""
        if self.baseline_dram_pj_per_bit is None:
            return (f"baseline x{self.code_n / self.code_k:.4f} "
                    f"(+{self.parity_frac * 100:.1f}%), embedded/recon x1.0   "
                    f"(pre-2026-09-10 traffic model)")
        other = (f"{self.dram_pj_per_bit:g}" if self.dram_pj_per_bit is not None
                 else "the ERT's")
        return (f"identical on all three arms; baseline pays "
                f"{self.baseline_dram_pj_per_bit:g} pJ/bit against {other} "
                f"pJ/bit (bigger array + indexing), recon fetches "
                f"K/N = {self.code_k / self.code_n:.4f} of the bits")

    @property
    def dram_static_note(self):
        """E_background and E_refresh are 0 unless someone sets them."""
        if self.dram_background_pj == 0 and self.dram_refresh_pj == 0:
            return ("E_total(DRAM) = E_dynamic only; E_background and E_refresh "
                    "are NOT modelled (both 0) -- the embedded arm holds fewer "
                    "weight bits in DRAM and is given no credit for it")
        return (f"E_background={self.dram_background_pj:g} pJ/bit-s, "
                f"E_refresh={self.dram_refresh_pj:g} pJ/bit/window")

    def dram_model_line(self):
        """The one line every Task 3 figure has to carry: where the decoder is,
        and therefore what happened to the DRAM term and at what per-bit cost."""
        enc = ("one encoder per DESTINATION of a multicast network "
               "(count = ingresses \u00d7 multicast factor)"
               if self.recon_encoder_site == "destination" else
               "ECC_RECON_ENCODER_SITE=source: ONE encoder before the fanout "
               "(count = ingresses; the pre-2026-09-09 row)")
        return (f"Decoder on the DRAM die: the WHOLE DRAM weight term \u00d7 K/N on "
                f"every R bar  \u00b7  "
                f"{self.dram_cost_note}"
                f"\n{self.dram_static_note}"
                f"\nNetwork boundaries: {enc}")

    def panel_title(self):
        """Title for a panelled figure: the model is per-panel, so it is not here.

        `figure_title` names the held model, which a panel figure does not have
        -- each panel is its own model and says so above its axes. What the
        shared title carries is everything the panels genuinely have in common.

        The code is dropped from the head as well: `title_suffix` already
        carries BCH(N,K) whenever the sweep is not the BCH one, and a heading
        that says it twice reads like two different settings.
        """
        head = "  ·  ".join(v for k, v in self.held if k not in ("model", "code"))
        parts = [p for p in (head, self.title_suffix()) if p]
        title = ("  ·  ".join(parts) +
                 f"  ·  {self.swept_axis} sweep, one panel per model")
        if self.layers:
            title += (f"\nDEVELOPMENT RUN — {self.layer_scope} only: "
                      f"{', '.join(self.layers)}  (not a full-model result)")
        return title

    # ---- the record ---------------------------------------------------------
    def to_dict(self):
        """The flat record every manifest and every result file carries.

        KEY ORDER IS `FIELD_ORDER` + the derived tail, because these dicts are
        written with `indent=1` and no `sort_keys`: the order is in the bytes on
        disk, and a reordering would rewrite every result file in the study
        without changing one number in it.
        """
        flat = self.flat()
        d = {k: flat[k] for k in FIELD_ORDER}
        d.update(code_t=self.code_t, workload=self.workload, stem=self.stem,
                 archs=self.archs, models=self.models, code_k=self.code_k,
                 arch_variant_slug=self.arch_variant_slug,
                 weights_per_codeword=self.weights_per_codeword,
                 emb_weights_per_cw=self.emb_weights_per_cw,
                 parity_frac=self.parity_frac, sram_scale=self.sram_scale,
                 weak_overhead=self.weak_overhead)
        return d

    def fingerprint(self):
        """Hash of everything that changes MAPPER output (not ECC arithmetic).

        The hashed set is the union of the six groups' own `IN_FINGERPRINT`
        declarations, so a knob is marked where it is declared instead of in a
        list kept by hand at the other end of the file. `sort_keys=True` makes
        the union an unordered SET, which is what lets it be assembled from six
        pieces and still hash byte-identically to the hand-written tuple it
        replaced -- `tests/contract/test_settings.py` pins exactly that.
        """
        keep = set()
        for g in GROUPS:
            keep |= _FINGERPRINT_MODULES[g].IN_FINGERPRINT
        # prompt_7: the fingerprint hashes the ARCHITECTURE, not the price list
        # Accelergy derives from it. So a corrected estimator -- the Neurosim
        # plug-in that returned 0 pJ for every address generator, say -- changes
        # every energy in the cache while leaving the path it is stored under
        # identical, and the stale entries are reused with nothing to say so.
        # `ECC_ENERGY_MODEL_REV` closes that: set it to any non-empty string and
        # the whole matrix colds DELIBERATELY. It is added only when set, so
        # EMPTY hashes byte-identically to every fingerprint that predates it.
        d = self.to_dict()
        if self.energy_model_rev:
            keep.add("energy_model_rev")
        blob = json.dumps({k: d[k] for k in sorted(keep)}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:8]


# --------------------------------------------------------------------- loading
def read_env():
    """Every `ECC_*` knob, read once, flat. The only call into `settings.env`."""
    flat = {}
    for g in GROUPS:
        flat.update(asdict(_CLASSES[g].from_env()))
    return flat


def _build(flat):
    """Resolve a flat record into the six frozen groups. One validator, one pass."""
    draft = _Draft(flat)
    _resolve(draft)
    return Config(**{g: _CLASSES[g](**{f: getattr(draft, f)
                                      for f in _CLASSES[g].__dataclass_fields__})
                     for g in GROUPS})


def load_config():
    """The configuration this process runs under: the environment, resolved once."""
    return _build(read_env())


def _ert_arm_row(cfg):
    """One banner line naming the ERT arm this job maps (prompt_6 8.3)."""
    if cfg.recon_ert_arm == "reference":
        return "reference (no ERT toll; the published 8-bit chip)"
    try:
        b = fingerprint_mod.ert_bump(cfg.archs[0], cfg)
        return (f"{b['placement']}: {b['level']}.{b['action']} += "
                f"{b['access_delta_pj']:.6f} pJ (E_w {b['e_w_pj']:.6f} x block_size "
                f"{b['block_size']}), {b['level']}.leak += {b['leak_delta_pj']:.7f} "
                f"pJ/instance/cycle; datawidth {cfg.weight_datawidth} on "
                f"{'+'.join(cfg.weight_datawidth_levels)}")
    except Exception as exc:                 # the banner must never kill a run
        return f"{cfg.recon_ert_arm} (bump unavailable: {exc})"


def banner(cfg, recon_terms, recon_provenance):
    """The banner `run.sh` prints before anything runs.

    The printing itself is `settings/banner.py`, which may not import `arch/` --
    so the one line that needs it, the ERT arm's bump, is computed here and
    handed in. Injecting it is what keeps the banner at L0 with the knobs it
    prints.
    """
    return banner_mod.banner(cfg, recon_terms, recon_provenance,
                             ert_arm_row=_ert_arm_row)
