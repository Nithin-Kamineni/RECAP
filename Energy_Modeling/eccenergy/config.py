"""All configuration in one place, read from the environment.

`run.sh` exports ECC_* variables; this module turns them into a validated
`Config` object. No other module reads os.environ.

THE SWEEP MODEL
---------------
A run walks exactly ONE axis and holds the other two fixed. All three ECC arms
are always drawn, so the arms are never an axis:

    ECC_SWEEP=bch     x = BCH(63, K) for K in ECC_SWEEP_KS
                      held: ECC_CONST_ARCH, ECC_CONST_MODEL
    ECC_SWEEP=model   x = the models in ECC_SWEEP_MODELS
                      held: ECC_CONST_ARCH, ECC_CONST_K
    ECC_SWEEP=arch    x = the architectures in ECC_SWEEP_ARCHS
                      held: ECC_CONST_MODEL, ECC_CONST_K

`archs`, `models` and `code_k` are DERIVED from that choice -- the swept axis
takes its list, the two held axes take their constant. One figure comes out,
named after the sweep (BCHsweep, ModelSweep, ArchitectureSweep), so a run
overwrites its own output instead of accumulating directories.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------- env helpers
_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n", ""}


class ConfigError(RuntimeError):
    pass


def _s(name, default=""):
    return os.environ.get(name, default).strip()


def _b(name, default):
    raw = _s(name)
    if raw == "":
        return default
    low = raw.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise ConfigError(f"{name}={raw!r} is not a boolean (use 1/0, true/false, on/off)")


def _i(name, default):
    raw = _s(name)
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not an integer") from exc


def _f(name, default):
    raw = _s(name)
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a number") from exc


def _oi(name):
    raw = _s(name)
    return None if raw == "" else _i(name, 0)


def _of(name):
    raw = _s(name)
    return None if raw == "" else _f(name, 0.0)


def _list(name, default=""):
    """Space- or comma-separated list; anything after '#' is a comment."""
    raw = _s(name, default).split("#", 1)[0]
    return [tok for tok in raw.replace(",", " ").split() if tok]


def _one(name, default=""):
    """A single token. Extra tokens are an error, not a silent truncation."""
    got = _list(name, default)
    if len(got) > 1:
        raise ConfigError(f"{name} takes ONE value, not {len(got)}: {' '.join(got)}\n"
                          f"  -> only the swept axis takes a list")
    return got[0] if got else ""


# ------------------------------------------------------------------ registries
EXPERIMENTS = ("sweep", "diagnose", "baseline", "embedded", "validate", "map", "panels")
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

#: BCH(63, K) minimum distance, from the code tables behind the DC synthesis runs.
BCH63_KTOD = {57: 3, 51: 5, 45: 7, 39: 9, 36: 11, 30: 13}

#: What the mapper is allowed to minimise. Timeloop's own metric names.
#:
#: `energy` is the default because this is an ENERGY study. The stock
#: `_include/mapper.yaml` that ships with the exercises repo asks for `edp`,
#: and EDP systematically penalises the wider PE array: an architecture with
#: more parallelism can buy latency by spending energy, so EDP steers it to a
#: costlier mapping. Measured on mobilenet_v2's C160_M960_R1_S1_P7_Q7 layer,
#: Eyeriss v2 (384 MACs) discarded a 6.84 pJ/MAC mapping for a 15.32 pJ/MAC one
#: because the latter was 4x faster; Eyeriss v1 (168 MACs) gave up only 1.24x
#: on the same layer. Set ECC_OPT_METRIC=edp to reproduce the old numbers.
OPT_METRICS = ("energy", "edp", "delay", "last_level_accesses")

#: How faithfully an architecture is modelled.
#:
#: `paper`  -- archs/<name>/arch_paper.yaml, where the storage precisions and
#:             scratchpad geometries follow the published tables. Every number
#:             in those files is commented with the paper it came from.
#: `stock`  -- the design exactly as timeloop-accelergy-exercises ships it (or,
#:             for the locally authored v2 designs, arch.yaml). Reproduces every
#:             pre-correction figure.
ARCH_FIDELITIES = ("paper", "stock")

#: How mapper effort scales with the depth of the architecture's loop nest.
#:
#: Timeloop's random search gives up after `victory_condition` consecutive
#: non-improving mappings, so a deeper hierarchy is searched less thoroughly at
#: the same setting. Eyeriss v2 has 9 loop levels to v1's 8 and a strictly
#: larger spatial search space, and at a flat victory_condition it converged
#: visibly less well. `levels` doubles the effort per level beyond the
#: reference depth; `none` uses ECC_VICTORY flat, as before.
VICTORY_SCALINGS = ("levels", "none")

#: Loop-nest depth that `ECC_VICTORY` is quoted for: Eyeriss v1 at paper
#: fidelity (6 storage levels + 2 spatial). Deeper designs scale up from here.
VICTORY_REFERENCE_LEVELS = 8

#: Cap on the scaling, so a deep hierarchy cannot make a run open-ended.
VICTORY_MAX_SCALE = 8

#: Architectures this dense CNN/transformer problem can map onto. eyeriss_v2_like
#: is authored in this repo (archs/); the rest ship with
#: timeloop-accelergy-exercises' example_designs.
KNOWN_ARCHS = (
    "eyeriss_like",
    "eyeriss_like_wglb",
    "eyeriss_v2_like",
    "eyeriss_v2_like_wglb",
    "simple_weight_stationary",
    "simple_output_stationary",
    "simple_input_stationary",
    "simba_like",
)

#: Designs whose number is a BOUND, not a measurement, unless the partner named
#: here is plotted beside them.
#:
#: Each pair differs in ONE modelling judgement that the design's paper does not
#: settle, and the two choices land far apart: modelling Eyeriss v1's published
#: 8 kB filter GLB as a reuse level takes its resnet18 DRAM weight refetch from
#: 7.04x to 1.69x and its ECC saving from 13.1% to 4.7%. A single number from
#: either file is a choice of bound. `Session.setup()` writes the caveat into
#: the run manifest whenever a design appears without its partner, because a
#: caveat that lives only in a README does not travel with the numbers.
BRACKET_PAIRS = {
    "eyeriss_like": (
        "eyeriss_like_wglb",
        "the 8 kB filter GLB that JSSC 2017 publishes is NOT modelled as a "
        "reuse level, so DRAM weight traffic and the ECC saving are UPPER bounds"),
    "eyeriss_like_wglb": (
        "eyeriss_like",
        "the published 8 kB filter GLB IS modelled as a full reuse level, so "
        "DRAM weight traffic and the ECC saving are LOWER bounds"),
}

#: Which workload file a model comes from. Mixing the two in one sweep is an
#: error: they live in different JSONs and have different problem generators.
CNN_MODELS = ("resnet18", "resnet50", "densenet121", "squeezenet1_1",
              "mobilenet_v2", "efficientnet_b0", "convnext_tiny", "xception")
TRANSFORMER_MODELS = ("distilgpt2", "gpt2", "bert_base", "gpt2_medium",
                      "opt_125m", "distilbert", "tinyllama")

#: Human-facing names for figure axes.
ARCH_LABELS = {
    "eyeriss_like": "Eyeriss v1",
    "eyeriss_like_wglb": "Eyeriss v1\n(+8kB filter GLB)",
    "eyeriss_v2_like": "Eyeriss v2",
    "eyeriss_v2_like_wglb": "Eyeriss v2\n(+weight NoC)",
    "simple_weight_stationary": "Weight stationary",
    "simple_output_stationary": "Output stationary",
    "simple_input_stationary": "Input stationary",
    # NOT "Simba". The exercises' reference design has 256 MACs, which cannot
    # deliver the paper's published 4 TOPS at its published clock (that needs
    # ~1024), and its PE buffers are 4-21x the published PE geometry. See
    # archs/_shared/provenance.yaml.
    "simba_like": "Simba-like\n(reference design)",
}

#: Which half of the study a result belongs to. See docs/RESULTS_SCHEMA.md.
#:
#: Pre   the mapping was chosen WITHOUT knowing about reconstruction; the ECC
#:       effect is applied during energy evaluation only. Tasks 1-3.
#: Post  the mapping itself was optimised for the reduced weight width. Task 4+.
#: Task 1 produces `Pre` results by construction: there is no reconstruction
#: yet, so there is nothing a mapper could have been made aware of.
PHASES = ("Pre", "Post")

#: Where codeword boundaries fall when parity is counted. See parity.py.
PARITY_GROUPINGS = ("layer", "model")

APPROACH_LABELS = {"baseline": "Baseline", "embedded": "Embedded", "recon": "Recon+"}
APPROACH_TAGS = {"baseline": "Base.", "embedded": "Embe.", "recon": "Recon+"}


@dataclass
class Config:
    # ---- what to run -------------------------------------------------------
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

    # ---- quantization / code geometry --------------------------------------
    weight_bits: int
    activation_bits: int
    acc_bits_override: Optional[int]
    code_n: int
    emb_weights_per_cw_override: Optional[float]
    parity_grouping: str
    parity_charge_padding: bool

    # ---- decode -------------------------------------------------------------
    decode_enabled: bool
    decode_pj_base: float
    decode_pj_emb: float
    recon_charges_decode: bool

    # ---- reconstruction datapath (Design Compiler numbers) -----------------
    recon_json: str
    recon_include_idle: bool
    recon_pj_override: Optional[float]
    recon_incremental_fallback_pj: float
    recon_idle_fallback_pj: float

    # ---- weak (SRAM-side) ECC overlay --------------------------------------
    weak_enabled: bool
    weak_n: int
    weak_k: int

    # ---- modelling switches -------------------------------------------------
    baseline_inflates_onchip: bool
    split_read_write: bool
    classify_mode: str

    # ---- architecture fairness knobs ---------------------------------------
    arch_fidelity: str
    force_technology: str
    force_datawidth: Optional[int]
    dram_depth: int
    global_cycle_seconds: str

    # ---- interconnect (NoC) energy: archs/_shared/noc.yaml ------------------
    # Timeloop's built-in wire model is a stub returning 0, so without these
    # every network is free -- in the evaluator AND in the mapper's objective.
    noc_enabled: bool
    noc_wire_pj_per_bit_mm: Optional[float]   # override of the shared constant
    noc_router_pj: Optional[float]            # override of shared.router_pj_per_flit
    noc_scale: float                          # sensitivity multiplier on all terms

    # ---- development mode: which layers, which half of the study -----------
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

    # ---- mapper -------------------------------------------------------------
    opt_metric: str
    victory: int
    victory_scaling: str
    mapper_threads: Optional[int]
    mapper_timeout: int
    mapper_algorithm: str
    mapper_seed: Optional[int]
    mapper_search_size: Optional[int]
    mapper_max_permutations: int

    # ---- output -------------------------------------------------------------
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
    archs: list = field(init=False, default_factory=list)
    models: list = field(init=False, default_factory=list)
    code_k: int = field(init=False, default=0)
    code_t: int = field(init=False, default=0)
    workload: str = field(init=False, default="cnn")

    # ------------------------------------------------------------------ derive
    def __post_init__(self):
        if self.experiment not in EXPERIMENTS:
            raise ConfigError(f"ECC_EXPERIMENT={self.experiment!r}; choose one of "
                              f"{', '.join(EXPERIMENTS)}")

        self.sweep = SWEEP_ALIASES.get(self.sweep, self.sweep)
        if self.sweep not in SWEEPS:
            raise ConfigError(f"ECC_SWEEP={self.sweep!r}; choose one of "
                              f"{', '.join(SWEEPS)}")

        bad = [a for a in self.approaches if a not in APPROACHES]
        if bad:
            raise ConfigError(f"ECC_APPROACHES has unknown entries {bad}; choose from "
                              f"{', '.join(APPROACHES)}")
        if not self.approaches:
            raise ConfigError("ECC_APPROACHES is empty -- nothing to compare")
        # canonical left-to-right bar order, however it was typed
        self.approaches = [a for a in APPROACHES if a in self.approaches]

        # ---- resolve the three axes ----------------------------------------
        if self.sweep == "arch":
            if not self.sweep_archs:
                raise ConfigError("ECC_SWEEP=arch but ECC_SWEEP_ARCHS is empty")
            self.archs = list(dict.fromkeys(self.sweep_archs))
        else:
            if not self.const_arch:
                raise ConfigError(f"ECC_SWEEP={self.sweep} holds the architecture "
                                  f"fixed, but ECC_CONST_ARCH is empty")
            self.archs = [self.const_arch]

        if self.sweep == "model":
            if not self.sweep_models:
                raise ConfigError("ECC_SWEEP=model but ECC_SWEEP_MODELS is empty")
            self.models = list(dict.fromkeys(self.sweep_models))
        else:
            if not self.const_model:
                raise ConfigError(f"ECC_SWEEP={self.sweep} holds the model fixed, "
                                  f"but ECC_CONST_MODEL is empty")
            self.models = [self.const_model]

        # ---- the panel layout (ECC_EXPERIMENT=panels) ----------------------
        # One panel per model, the swept axis repeated inside each. `models` is
        # widened to every panel model so ONE collection pass fills all the
        # panels; the swept axis is untouched, which is what keeps this a page
        # layout rather than a fourth axis.
        if self.experiment == "panels":
            if self.sweep not in ("arch", "model"):
                raise ConfigError(
                    f"ECC_EXPERIMENT=panels with ECC_SWEEP={self.sweep}: a panel "
                    f"per model would then vary the model AND the code between "
                    f"panels, which is two axes at once.\n"
                    f"  -> use ECC_SWEEP=arch (an architecture sweep per model)")
            if not self.panel_models:
                raise ConfigError(
                    "ECC_EXPERIMENT=panels needs ECC_PANEL_MODELS, e.g.\n"
                    '  ECC_PANEL_MODELS="resnet18 mobilenet_v2"')
            self.panel_models = list(dict.fromkeys(self.panel_models))
            self.models = list(self.panel_models)

        if self.sweep == "bch":
            if not self.sweep_ks:
                raise ConfigError("ECC_SWEEP=bch but ECC_SWEEP_KS is empty")
            self.sweep_ks = list(dict.fromkeys(self.sweep_ks))
            bad_k = [k for k in self.sweep_ks if k >= self.code_n]
            if bad_k:
                raise ConfigError(f"ECC_SWEEP_KS entries must be < N={self.code_n}: {bad_k}")
        self.code_k = self.const_k

        if self.code_k >= self.code_n:
            raise ConfigError(f"need K < N; got N={self.code_n} "
                              f"K={self.code_k} (ECC_CONST_K)")
        if self.weight_bits <= 0:
            raise ConfigError("ECC_WEIGHT_BITS must be positive")
        if self.activation_bits <= 0:
            raise ConfigError("ECC_ACTIVATION_BITS must be positive")
        if self.acc_bits_override is not None:
            # The SENSITIVITY STUDY only. The primary comparison keeps each
            # design's published accumulator width, because psum precision is
            # an architectural property (v1 truncates to 16b, v2 accumulates at
            # 20b, Simba at 24b) and equalising it equalises the architectures.
            if self.acc_bits_override < self.weight_bits:
                raise ConfigError(
                    f"ECC_ACC_BITS={self.acc_bits_override} is narrower than "
                    f"ECC_WEIGHT_BITS={self.weight_bits}; an accumulator cannot "
                    f"be narrower than the operands it accumulates")
        if self.parity_grouping not in PARITY_GROUPINGS:
            raise ConfigError(f"ECC_PARITY_GROUPING must be one of "
                              f"{', '.join(PARITY_GROUPINGS)}")
        if self.phase not in PHASES:
            raise ConfigError(f"ECC_PHASE must be one of {', '.join(PHASES)}")

        d = BCH63_KTOD.get(self.code_k) if self.code_n == 63 else None
        self.code_t = (d - 1) // 2 if d else max(1, (self.code_n - self.code_k) // 6)

        # ---- which workload file the models come from ----------------------
        cnn = [m for m in self.models if m in CNN_MODELS]
        tfm = [m for m in self.models if m in TRANSFORMER_MODELS]
        if cnn and tfm:
            raise ConfigError(
                "one sweep cannot mix CNNs and transformers -- they come from "
                f"different workload files.\n  CNNs        : {' '.join(cnn)}\n"
                f"  transformers: {' '.join(tfm)}")
        self.workload = "transformer" if tfm else "cnn"

        # ECC_FROM_CACHE forbids invoking Timeloop at all, so it cannot also be
        # asked to re-run the mapper. Silently preferring one would mean a run
        # asked to refresh its mappings quietly refreshing nothing.
        if self.rerun_optimiser and (self.from_cache or self.replot_only):
            blocker = "ECC_FROM_CACHE=1 (--eval)" if self.from_cache \
                else "ECC_REPLOT_ONLY=1 (--replot)"
            raise ConfigError(
                f"ECC_RERUN_OPTIMISER=1 asks the mapper to re-solve every shape, "
                f"but {blocker} never invokes Timeloop at all.\n"
                f"  -> re-map with `bash run.sh map`, then evaluate with `--eval`")

        if self.weak_enabled and self.weak_k >= self.weak_n:
            raise ConfigError(f"need WEAK_K < WEAK_N; got {self.weak_n}/{self.weak_k}")
        if self.classify_mode not in ("instances", "name"):
            raise ConfigError("ECC_CLASSIFY must be 'instances' or 'name'")
        if self.arch_fidelity not in ARCH_FIDELITIES:
            raise ConfigError(f"ECC_ARCH_FIDELITY must be one of "
                              f"{', '.join(ARCH_FIDELITIES)}")
        if self.opt_metric not in OPT_METRICS:
            raise ConfigError(f"ECC_OPT_METRIC={self.opt_metric!r}; choose one of "
                              f"{', '.join(OPT_METRICS)}")
        if self.victory_scaling not in VICTORY_SCALINGS:
            raise ConfigError(f"ECC_VICTORY_SCALING must be one of "
                              f"{', '.join(VICTORY_SCALINGS)}")
        if self.victory < 1:
            raise ConfigError("ECC_VICTORY must be >= 1")
        if self.palette not in ("house", "cvd"):
            raise ConfigError("ECC_PALETTE must be 'house' or 'cvd'")
        for fmt in self.formats:
            if fmt not in ("png", "pdf", "svg"):
                raise ConfigError(f"ECC_FORMATS: unsupported format {fmt!r}")

        unknown = [a for a in self.archs if a not in KNOWN_ARCHS]
        if unknown:
            print(f"[config] note: looked up in example_designs/ as-is: {', '.join(unknown)}")
        unknown = [m for m in self.models if m not in CNN_MODELS + TRANSFORMER_MODELS]
        if unknown:
            print(f"[config] note: not a listed model; looked up in the workload "
                  f"file as-is: {', '.join(unknown)}")

    # -------------------------------------------------- development-mode view
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

    # ------------------------------------------------------------- properties
    @property
    def stem(self):
        """The ONE output name for this run. Fixed per sweep, by design.

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
            # two disambiguating suffixes below. env.sh section 10 sets it
            # so an architecture sweep always lands on `ArchitectureSweep.png`
            # whether it drew one model or a panel per model. The cost is
            # real: a one-model and a two-model sweep then overwrite each
            # other, and only the manifest beside the figure records which
            # is on disk. Do not set it by hand for a run with ECC_LAYERS --
            # that is exactly the case the layer suffix exists to protect.
            return self.stem_override
        if self.experiment == "diagnose":
            return "diagnose"
        base = SWEEP_STEMS[self.sweep]
        if self.experiment == "panels":
            # The panel models are IN the name, so a two-model figure can never
            # land on top of the single-model `ArchitectureSweep.png`, and two
            # different model pairs are two different files.
            base = f"{base}__panels__{'__'.join(self.panel_models)}"
        return base if not self.layers else f"{base}__{self.layer_slug}"

    @property
    def swept_axis(self):
        return {"bch": "code-strength", "model": "model", "arch": "architecture"}[self.sweep]

    @property
    def swept_values(self):
        """The x axis, in order."""
        return {"bch": self.sweep_ks, "model": self.models, "arch": self.archs}[self.sweep]

    @property
    def held(self):
        """[(axis, value)] for the two axes this run holds fixed -- for titles."""
        pairs = [("architecture", self.arch_label(self.const_arch)),
                 ("model", self.const_model),
                 ("code", f"BCH({self.code_n},{self.code_k}) t={self.code_t}")]
        drop = {"arch": 0, "model": 1, "bch": 2}[self.sweep]
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

        Defaults to the shared-codeword model (identical to the baseline). The
        older two-arm scripts charged the embedded arm on a smaller word; set
        ECC_EMB_WEIGHTS_PER_CW=8 to reproduce those.
        """
        return self.emb_weights_per_cw_override or self.weights_per_codeword

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
        if self.global_cycle_seconds != "1e-9":
            parts.append(f"clk{self.global_cycle_seconds}")
        if self.noc_enabled:
            # A costed interconnect is a different architecture to the mapper.
            # Every cache built before archs/_shared/noc.yaml existed was built
            # with a free NoC and must never be read back as this.
            noc = "noc"
            if self.noc_wire_pj_per_bit_mm is not None:
                noc += f"w{self.noc_wire_pj_per_bit_mm:g}"
            if self.noc_router_pj is not None:
                noc += f"r{self.noc_router_pj:g}"
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
        """
        parts = []
        if self.arch_fidelity != "stock":
            parts.append(self.arch_fidelity)
        if self.force_datawidth:
            parts.append(f"dw{self.force_datawidth}")
        if self.acc_bits_override is not None:
            # The common-accumulator SENSITIVITY study rewrites psum levels, so
            # it is a different architecture to the mapper and must never share
            # a cache with the paper-native primary result.
            parts.append(f"acc{self.acc_bits_override}")
        return parts

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

    def title_suffix(self):
        bits = [f"{self.weight_bits}-bit weights"]
        if self.sweep != "bch":
            bits.append(f"BCH({self.code_n},{self.code_k}) t={self.code_t}")
        if self.force_technology:
            bits.insert(0, self.force_technology)
        if self.weak_enabled:
            bits.append(f"weak BCH({self.weak_n},{self.weak_k})")
        if self.title_note:
            bits.append(self.title_note)
        return "  ·  ".join(bits)

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

    def to_dict(self):
        d = asdict(self)
        d.update(code_t=self.code_t, workload=self.workload, stem=self.stem,
                 archs=self.archs, models=self.models, code_k=self.code_k,
                 arch_variant_slug=self.arch_variant_slug,
                 weights_per_codeword=self.weights_per_codeword,
                 emb_weights_per_cw=self.emb_weights_per_cw,
                 parity_frac=self.parity_frac, sram_scale=self.sram_scale,
                 weak_overhead=self.weak_overhead)
        return d

    def fingerprint(self):
        """Hash of everything that changes MAPPER output (not ECC arithmetic)."""
        keep = ("archs", "models", "layers", "weight_bits", "activation_bits",
                "acc_bits_override", "arch_fidelity", "force_technology",
                "force_datawidth", "dram_depth", "global_cycle_seconds",
                "noc_enabled", "noc_wire_pj_per_bit_mm", "noc_router_pj", "noc_scale",
                "opt_metric", "victory", "victory_scaling", "mapper_algorithm",
                "mapper_seed", "mapper_timeout", "mapper_search_size",
                "mapper_max_permutations")
        blob = json.dumps({k: self.to_dict()[k] for k in keep}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:8]


# --------------------------------------------------------------------- loading
def load_config():
    return Config(
        experiment=_s("ECC_EXPERIMENT", "sweep").lower(),
        sweep=_s("ECC_SWEEP", "bch").lower(),
        sweep_archs=_list("ECC_SWEEP_ARCHS", " ".join(KNOWN_ARCHS)),
        sweep_models=_list("ECC_SWEEP_MODELS", " ".join(CNN_MODELS)),
        sweep_ks=[int(k) for k in _list("ECC_SWEEP_KS", "57 51 45 39 36 30")],
        panel_models=_list("ECC_PANEL_MODELS"),
        const_arch=_one("ECC_CONST_ARCH", "eyeriss_like"),
        const_model=_one("ECC_CONST_MODEL", "resnet18"),
        const_k=_i("ECC_CONST_K", 51),
        approaches=[a.lower() for a in _list("ECC_APPROACHES", "baseline embedded recon")],

        weight_bits=_i("ECC_WEIGHT_BITS", 8),
        activation_bits=_i("ECC_ACTIVATION_BITS", 8),
        acc_bits_override=_oi("ECC_ACC_BITS"),
        code_n=_i("ECC_CODE_N", 63),
        emb_weights_per_cw_override=_of("ECC_EMB_WEIGHTS_PER_CW"),
        parity_grouping=_s("ECC_PARITY_GROUPING", "layer").lower(),
        parity_charge_padding=_b("ECC_PARITY_CHARGE_PADDING", True),

        decode_enabled=_b("ECC_DECODE", False),
        decode_pj_base=_f("ECC_DECODE_PJ_BASE", 40.0),
        decode_pj_emb=_f("ECC_DECODE_PJ_EMB", 40.0),
        recon_charges_decode=_b("ECC_RECON_CHARGES_DECODE", True),

        recon_json=_s("ECC_RECON_JSON", "data/dc/BCH_N63_results.json"),
        recon_include_idle=_b("ECC_RECON_INCLUDE_IDLE", True),
        recon_pj_override=_of("ECC_RECON_PJ"),
        recon_incremental_fallback_pj=_f("ECC_RECON_INCREMENTAL_FALLBACK_PJ", 1.8995),
        recon_idle_fallback_pj=_f("ECC_RECON_IDLE_FALLBACK_PJ", 2.2301273),

        weak_enabled=_b("ECC_WEAK", False),
        weak_n=_i("ECC_WEAK_N", 63),
        weak_k=_i("ECC_WEAK_K", 57),

        baseline_inflates_onchip=_b("ECC_BASELINE_INFLATES_ONCHIP", False),
        split_read_write=_b("ECC_SPLIT_READ_WRITE", False),
        classify_mode=_s("ECC_CLASSIFY", "instances").lower(),

        arch_fidelity=_s("ECC_ARCH_FIDELITY", "paper").lower(),
        force_technology=_s("ECC_FORCE_TECHNOLOGY"),
        force_datawidth=_oi("ECC_FORCE_DATAWIDTH"),
        dram_depth=_i("ECC_DRAM_DEPTH", 1048576),
        global_cycle_seconds=_s("ECC_GLOBAL_CYCLE_SECONDS", "1e-9"),

        noc_enabled=_b("ECC_NOC", True),
        noc_wire_pj_per_bit_mm=_of("ECC_NOC_WIRE_PJ_PER_BIT_MM"),
        noc_router_pj=_of("ECC_NOC_ROUTER_PJ"),
        noc_scale=_f("ECC_NOC_SCALE", 1.0),

        layers=_list("ECC_LAYERS"),
        phase=(_one("ECC_PHASE", "Pre") or "Pre").capitalize(),
        overwrite=_b("ECC_OVERWRITE", False),
        cache_strict=_b("ECC_CACHE_STRICT", True),
        rerun_optimiser=_b("ECC_RERUN_OPTIMISER", False),
        run_note=_s("ECC_RUN_NOTE"),

        opt_metric=_s("ECC_OPT_METRIC", "energy").lower(),
        victory=_i("ECC_VICTORY", 500),
        victory_scaling=_s("ECC_VICTORY_SCALING", "levels").lower(),
        mapper_threads=_oi("ECC_MAPPER_THREADS"),
        mapper_timeout=_i("ECC_MAPPER_TIMEOUT", 10000),
        mapper_algorithm=_s("ECC_MAPPER_ALGORITHM", "hybrid"),
        mapper_seed=_oi("ECC_MAPPER_SEED"),
        mapper_search_size=_oi("ECC_MAPPER_SEARCH_SIZE"),
        mapper_max_permutations=_i("ECC_MAPPER_MAX_PERMUTATIONS", 16),

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


def banner(cfg, recon_pj, recon_provenance):
    w = 78
    lines = ["=" * w,
             f"ecc-energy  |  sweep={cfg.sweep} ({cfg.swept_axis})  "
             f"|  output={cfg.stem}",
             "=" * w]

    if cfg.sweep == "bch":
        swept = "K = " + ", ".join(str(k) for k in cfg.sweep_ks)
    else:
        swept = ", ".join(str(v) for v in cfg.swept_values)

    # PRECISIONS, EXPLICIT AND DISTINCT. Task 1 asks the configuration summary
    # to report them, because the defect that started this work was two designs
    # being compared at different operand widths without anyone noticing. The
    # accumulator line is deliberately not a single number: it is per design,
    # and saying so on every run is the point.
    if cfg.acc_bits_override is None:
        acc = ("per architecture, as published (v1 16b, v2 20b, Simba 24b, "
               "simple designs 16b)  [primary]")
    else:
        acc = (f"{cfg.acc_bits_override}b FORCED on every design "
               f"(ECC_ACC_BITS)  [SENSITIVITY STUDY, not the primary result]")

    rows = [
        ("sweeping", swept),
        ("holding", "   ".join(f"{k}={v}" for k, v in cfg.held
                               # a panel figure does not hold the model -- each
                               # panel IS a model, so saying "holding model=X"
                               # would name only the first one and mislead
                               if not (cfg.experiment == "panels" and k == "model"))),
    ]
    if cfg.experiment == "panels":
        rows.append(("panels (top to bottom)", ", ".join(cfg.panel_models)))
    rows += [
        ("approaches", ", ".join(cfg.approaches)),
        ("workload file", cfg.workload),
        ("layer scope", (cfg.layer_scope if not cfg.layers else
                         f"{cfg.layer_scope}: {', '.join(cfg.layers)}")),
        ("weight precision", f"{cfg.weight_bits}b (the protected payload)"),
        ("activation precision", f"{cfg.activation_bits}b"),
        ("accumulator precision", acc),
        ("result phase", cfg.phase + ("   (mapping is ECC-unaware; the ECC effect is "
                                      "applied during evaluation)" if cfg.phase == "Pre"
                                      else "   (mapping optimised for the reduced width)")),
    ]
    if cfg.sweep == "bch":
        rows.append(("recon pJ/codeword", recon_provenance))
    else:
        rows += [
            ("code", f"BCH({cfg.code_n},{cfg.code_k})  t={cfg.code_t}  "
                     f"r={cfg.code_n - cfg.code_k}"),
            ("weights / codeword", f"{cfg.weights_per_codeword:.4f}"),
            ("DRAM weight inflation", f"baseline x{cfg.code_n / cfg.code_k:.4f} "
                                      f"(+{cfg.parity_frac * 100:.1f}%), "
                                      f"embedded/recon x1.0"),
            ("recon on-chip scale", f"{cfg.sram_scale:.4f} (weights only)"),
            ("reconstruction", f"{recon_pj:.7f} pJ per codeword"),
            ("recon provenance", recon_provenance),
        ]
    rows += [
        ("parity accounting", f"grouping={cfg.parity_grouping}, "
                              f"message padding "
                              f"{'charged' if cfg.parity_charge_padding else 'NOT charged'}"),
        ("decode", (f"base {cfg.decode_pj_base} / emb {cfg.decode_pj_emb} pJ per codeword"
                    if cfg.decode_enabled else
                    "0 pJ for every arm (ECC_DECODE=0; category hidden)")),
        ("weak ECC overlay", (f"BCH({cfg.weak_n},{cfg.weak_k}) -> x{cfg.weak_overhead:.4f} on-chip"
                              if cfg.weak_enabled else "off")),
        ("arch fidelity", cfg.arch_fidelity
            + ("   (archs/<name>/arch_paper.yaml where present)"
               if cfg.arch_fidelity == "paper"
               else "   (example_designs as shipped -- psum precision NOT honest)")),
        ("arch treatment", cfg.arch_variant_slug
            + ("" if cfg.arch_variant_slug == "stock" else "   (separate mapper cache)")),
        ("level classifier", cfg.classify_mode
            + ("" if cfg.classify_mode == "instances" else "   (legacy name matching)")),
        ("mapper objective", cfg.opt_metric
            + ("" if cfg.opt_metric == "energy" else
               "   (WARNING: this is an energy study; 'edp' trades energy for latency "
               "and penalises the wider PE array)")),
        ("mapper", f"victory={cfg.victory} scaling={cfg.victory_scaling} "
                   f"algorithm={cfg.mapper_algorithm} timeout={cfg.mapper_timeout} "
                   f"perms/if={cfg.mapper_max_permutations} "
                   f"search_size={cfg.mapper_search_size or 'uncapped'}"),
        ("palette / formats", f"{cfg.palette} / {', '.join(cfg.formats)} @ {cfg.dpi} dpi"),
        ("replot only", str(cfg.replot_only)),
        ("cached mapper only", str(cfg.from_cache)
            + ("   (never invokes Timeloop)" if cfg.from_cache else "")),
        ("re-run the optimiser", str(cfg.rerun_optimiser)
            + ("   (ECC_RERUN_OPTIMISER=1: valid cache entries are IGNORED and "
               "OVERWRITTEN)" if cfg.rerun_optimiser else "")),
    ]
    for k, v in rows:
        lines.append(f"  {k:22s}: {v}")
    lines.append("=" * w)
    return "\n".join(lines)
