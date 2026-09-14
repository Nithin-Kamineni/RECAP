"""Every refusal in the project: its id, its TIER, and whether you may override it.

`ProjectRestructure.md` section 6. A **guard** is not a test. A test runs when
you type `pytest` and prints red; a guard runs on EVERY run, including a SLURM
job at 3am, and STOPS it. There are ~140 of them and until phase 6 there was no
list of either, which is the "impossible to know what is there" this file exists
to answer -- `GUARDS.md`, generated from this registry, is the one screen.

THE FOUR TIERS, and the only thing that matters about them: which ones you may
turn off.

    1 PARSE       "ECC_VICTORY=abc is not an integer".  NEVER overridable.
    2 IMPOSSIBLE  "need K < N";  "width must be positive".  Physically or
                  arithmetically cannot hold.  NEVER overridable.
    3 COUPLING    "ECC_SPLIT_READ_WRITE=1 inside the placement study".  Encodes
                  THE COMBINATIONS WE THOUGHT OF.  Overridable.
    4 DERIVED     "a price of ZERO is an ablation, not a price".  Refuses you
                  for setting a knob the system also derives, or for asking
                  for the bound rather than the number.  Overridable.

Tiers 3 and 4 are the ABLATION BLOCKERS (section 6.3: six of fifteen plausible
ablations were refused outright). They are refused by DEFAULT and run when you
name them:

    ECC_ALLOW="zero-price,recon-no-split-read-write"

-- never a blanket `ECC_GUARDS=off`, which section 10 rules out: it would be set
once, forgotten, and a wrong number would reach a figure with nothing saying so.
Naming the guard is the point. AND AN OVERRIDE IS RECORDED: `overrides()` lands
in the run manifest and on the figure's caveat list, through the same mechanism
`Config.recon_caveats()` already uses, so you cannot ablate by accident and you
cannot publish an ablation without the figure saying it is one.

## The two spellings, and why the SITE tells you which tier it is

    raise guards.refusal("need-k-lt-n", f"need K < N; got ...")   # tiers 1-2
    guards.refuse("zero-price", f"...")                           # tiers 3-4

`refusal()` BUILDS the exception and the caller raises it: it always stops the
run, so `raise` at the site is the truth. `refuse()` ACTS: it raises, or -- when
the id is in `ECC_ALLOW` -- prints a loud warning, records the override and
RETURNS, so the site continues. Reading a guard site therefore tells you whether
it can be turned off without looking anything up, and the two functions refuse to
be used at the wrong tier.

The exception TYPE is declared here, not at the site, so that converting a site
cannot silently change `SystemExit` into `ConfigError` -- `__main__` prints one
without a traceback and the other is an exit status, and the restructure gate's
golden snapshot recorded both. That is also why every message below is unchanged from what the
site raised before phase 6: a message is a finding somebody paid for.

## What is registered but NOT called from here

Three files are FROZEN (`CLAUDE.md`, "Working on this code"): `physics/parity.py`
and `study/baseline.py` may move but their contents may not change. Their guards
are real guards and belong in `GUARDS.md`, so they are registered with
`frozen_match=` -- a substring the site must still contain, checked by grep in
`tests/contract/test_guards.py`. The file is never edited and the registry still
cannot drift from it.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional, Tuple

from ..contracts.errors import ConfigError

#: Tier -> the one word that says what the tier MEANS.
TIER_NAMES = {1: "PARSE", 2: "IMPOSSIBLE", 3: "COUPLING", 4: "DERIVED"}

#: The tiers `ECC_ALLOW` can name. 1 and 2 are not negotiable: a value that
#: cannot be parsed has no meaning, and a value that is impossible has no
#: physics. Overriding either would not produce an ablation, it would produce a
#: number with nothing behind it.
OVERRIDABLE = frozenset({3, 4})

#: The knob. A comma- or space-separated list of guard ids, never a blanket off.
ALLOW_VAR = "ECC_ALLOW"


@dataclass(frozen=True)
class Guard:
    """One refusal: what it protects, what it costs you, and who may lift it."""
    tier: int
    refuses: str                       #: one line, for GUARDS.md
    raises: type = ConfigError         #: the exception the site raises
    note: str = ""                     #: anything GUARDS.md must carry beside it
    frozen_match: Optional[str] = None #: a FROZEN file's site, matched by text
    frozen_file: Optional[str] = None

    @property
    def overridable(self):
        return self.tier in OVERRIDABLE


def _g(tier, refuses, raises=ConfigError, note="", frozen_match=None, frozen_file=None):
    return Guard(tier, refuses, raises, note, frozen_match, frozen_file)


# =============================================================================
#  THE REGISTRY.  One row per invariant -- NOT one row per raise site: an
#  invariant checked in two places (`clock-positive` is checked at the design
#  file AND again where a with_() override reaches it) is ONE guard with two
#  sites, and `GUARDS.md` lists both. `tests/contract/test_guards.py` holds the two
#  ends together: every id here must be called somewhere, and every refusal site
#  in the package must name an id here.
# =============================================================================
GUARDS = {

    # ---- tier 1: PARSE -- the string does not mean anything --------------
    "not-a-boolean":            _g(1, "a boolean knob that is not 1/0/true/false"),
    "not-an-integer":           _g(1, "an integer knob that is not an integer"),
    "not-a-number":             _g(1, "a float knob that is not a number"),
    "not-a-key-value-table":    _g(1, "an `ECC_*_LIST` entry that is not `key=number`"),
    "not-a-key-value-list":     _g(1, "an ECC_LAYERS per-model entry that is not `model=layer layer`"),
    "one-value-not-a-list":     _g(1, "a list where ONE value is meant (only the swept axis takes a list)"),
    "unknown-experiment":       _g(1, "ECC_EXPERIMENT naming no known experiment"),
    "unknown-sweep":            _g(1, "ECC_SWEEP naming no known axis"),
    "unknown-approach":         _g(1, "ECC_APPROACHES naming an arm that does not exist"),
    "unknown-recon-default":    _g(1, "ECC_RECON_DEFAULT naming something that is not a reconstruction placement",
                                    note="what a bare `recon` bar MEANS. The abstract `recon` arm is retired (EnvReorganisation 6.2), so there is no value here that means 'not a placement'."),
    "unknown-metric":           _g(1, "ECC_METRICS naming a figure row that does not exist",
                                 note="the PLOTTED metric, not the mapper's "
                                      "`ECC_OPT_METRIC` -- note `latency` here "
                                      "against `delay` there."),
    "unknown-parity-grouping":  _g(1, "ECC_PARITY_GROUPING naming no known grouping"),
    "unknown-recon-packing":    _g(1, "ECC_RECON_PACKING naming no known packing"),
    "unknown-encoder-granularity": _g(1, "ECC_RECON_ENCODER_GRANULARITY naming no known granularity"),
    "unknown-encoder-site":     _g(1, "ECC_RECON_ENCODER_SITE naming no known site"),
    "unknown-classify":         _g(1, "ECC_CLASSIFY that is neither `instances` nor `name`"),
    "unknown-capacity-scope":   _g(1, "ECC_WEIGHT_CAPACITY_SCOPE naming no known scope"),
    "unknown-arch-fidelity":    _g(1, "ECC_ARCH_FIDELITY naming no known fidelity"),
    "unknown-opt-metric":       _g(1, "ECC_OPT_METRIC naming no known objective"),
    "unknown-victory-scaling":  _g(1, "ECC_VICTORY_SCALING naming no known scaling"),
    "unknown-palette":          _g(1, "ECC_PALETTE that is neither `house` nor `cvd`"),
    "unknown-format":           _g(1, "ECC_FORMATS naming a format matplotlib is not asked for"),
    "unknown-knob":             _g(1, "`cfg.with_()` naming a knob no settings group declares"),
    "leakage-not-a-table":      _g(1, "a design.yaml `leakage_nw:` that is not density -> nW"),
    "widths-not-a-table":       _g(1, "a widths.yaml that is not q -> {spad_width, glb_width}"),
    "values-unknown-sweep":     _g(1, "`--values` for a sweep that has no knob", SystemExit),
    "generate-usage":           _g(1, "`python3 -m eccenergy.arch.generate` with no known mode", SystemExit),
    "design-yaml-parse":        _g(1, "a design file that is not valid YAML"),
    "design-order-not-integer": _g(1, "`order:` in a design.yaml that is not an integer"),

    # ---- tier 2: IMPOSSIBLE -- arithmetic, physics, or absent data -------
    "need-k-lt-n":              _g(2, "a BCH code with K >= N"),
    "need-weak-k-lt-n":         _g(2, "a weak code with WEAK_K >= WEAK_N"),
    "sweep-k-lt-n":             _g(2, "an ECC_SWEEP_KS entry that is not below N"),
    "weight-bits-positive":     _g(2, "ECC_WEIGHT_BITS <= 0"),
    "activation-bits-positive": _g(2, "ECC_ACTIVATION_BITS <= 0"),
    "acc-bits-not-narrower":    _g(2, "an accumulator narrower than the operands it accumulates"),
    "victory-ge-1":             _g(2, "ECC_VICTORY below 1"),
    "clock-positive":           _g(2, "a declared clock rate that is not positive"),
    "capacity-scale-positive":  _g(2, "ECC_WEIGHT_CAPACITY_SCALE <= 0"),
    "depth-scale-positive":     _g(2, "ECC_WEIGHT_DEPTH_SCALE <= 0"),
    "glb-width-mult-ge-1":      _g(2, "a weight GLB word narrower than the scratchpad's"),
    "widths-not-a-multiple":    _g(2, "a declared word width its own datawidth q does not divide (timeloop-mapper aborts on it)"),
    "datawidth-positive":       _g(2, "ECC_WEIGHT_DATAWIDTH below 1 bit"),
    "datawidth-le-weight-bits": _g(2, "a REDUCED weight wider than the full one"),
    "negative-price":           _g(2, "a NEGATIVE per-bit or per-MAC price",
                                 note="the `>= 0` half of `zero-price`: zero is an "
                                      "ablation, negative is not a price."),
    "dram-static-terms-nonnegative": _g(2, "a negative DRAM background or refresh term"),
    "approaches-empty":         _g(2, "ECC_APPROACHES empty -- nothing to compare"),
    "metrics-empty":            _g(2, "ECC_METRICS empty -- a figure with no rows"),
    "recon-area-missing":       _g(2, "ECC_METRICS=area with no archs/_shared/recon_area.yaml scraped yet"),
    "recon-area-unmeasured":    _g(2, "ECC_METRICS=area at a code Design Compiler has not synthesized",
                                 note="the area metric has NO fallback "
                                      "constant, unlike the energy: an "
                                      "invented engine area is an invented "
                                      "silicon number."),
    "area-chip-never-mapped":   _g(2, "ECC_METRICS=area on a chip with no ART in the mapper cache"),
    "latency-no-cycles":        _g(2, "ECC_METRICS=latency on a raw record carrying no cycle count"),
    "latency-not-modelled":     _g(2, "ECC_METRICS=latency on a record ECC_LATENCY_MODEL=1 has not re-timed"),
    "sweep-has-no-figure":      _g(2, "a sweep or panel FIGURE on an axis that holds all three lists (fix, area)"),
    "layers-per-model-one-model": _g(2, "ECC_LAYERS' per-model form on a run that evaluates several models"),
    "sweep-archs-empty":        _g(2, "ECC_SWEEP=arch with no architectures"),
    "sweep-models-empty":       _g(2, "ECC_SWEEP=model with no models"),
    "sweep-ks-empty":           _g(2, "ECC_SWEEP=bch with no codes"),
    "const-arch-empty":         _g(2, "a held architecture axis with no ECC_CONST_ARCH"),
    "const-model-empty":        _g(2, "a held model axis with no ECC_CONST_MODEL"),
    "panels-need-panel-models": _g(2, "ECC_EXPERIMENT=panels with no ECC_PANEL_MODELS"),
    "no-cnn-transformer-mix":   _g(2, "one sweep mixing CNNs and transformers (two workload files)"),
    "recon-needs-an-arch":      _g(2, "the placement study with no architecture"),
    "recon-one-model":          _g(2, "the placement study on more than one model"),
    "ert-arm-one-arch":         _g(2, "ECC_RECON_ERT_ARM with more than one architecture"),
    "unknown-ert-arm":          _g(2, "ECC_RECON_ERT_ARM naming no arm of this design"),
    "no-weight-path":           _g(2, "a placement study on a design that declares no weight path", SystemExit),
    "weight-path-drifted":      _g(2, "a weight path and a placement list that have drifted apart", SystemExit),
    "missing-path":             _g(2, "a required input file that is not there", SystemExit),
    "no-arch-yamls":            _g(2, "an audit with no architecture YAML to read", SystemExit),
    "noc-file-missing":         _g(2, "the interconnect coefficients file being absent", SystemExit),
    "noc-no-entry":             _g(2, "a design with no interconnect entry (it would map with free wires)",
                                 SystemExit),
    "standard-file-missing":    _g(2, "the apples-to-apples contract file being absent", SystemExit),
    "workload-groups-indivisible": _g(2, "a workload layer whose channels do not divide by its groups",
                                    SystemExit),
    "workload-no-models":       _g(2, "none of the requested models being in the workload file", SystemExit),
    "layers-not-in-model":      _g(2, "ECC_LAYERS naming layers the model does not have", SystemExit),
    "dilation-unknown-layer":   _g(2, "a dilation spot check naming a layer the model does not have",
                                 SystemExit),
    "dilation-needs-layers":    _g(2, "a dilation spot check with no layer named", SystemExit),
    "dc-clock-mismatch":        _g(2, "a DC datapath entry measured at another clock period", SystemExit),
    "timeloop-mapper-not-on-path": _g(2, "timeloop-mapper not being on PATH", SystemExit),
    "timeloop-model-not-on-path": _g(2, "timeloop-model not being on PATH", SystemExit),
    "designs-dir-missing":      _g(2, "the cloned exercises repo lacking its designs directory", SystemExit),
    "design-inputs-not-written": _g(2, "a mapper input that `write_globals()` never wrote", SystemExit),
    "accelergy-wrote-no-ert":   _g(2, "Accelergy producing no ERT/ART at all", SystemExit),
    "ert-row-missing":          _g(2, "a generated ERT with no row for a level the study prices", SystemExit),
    "level-carries-no-weights": _g(2, "a level named as a weight site that carries no Weights", SystemExit),

    # -- the cold-cache family. A number with no cached mapping cannot be
    # -- computed, so it is REFUSED rather than fetched from another chip.
    "task4-cache-cold":         _g(2, "Task 4 without the reconstruction arm's OWN mapping", SystemExit),
    "ert-arm-cache-cold":       _g(2, "an ERT arm without its own mapping", SystemExit),
    "arm-half-mapped":          _g(2, "an arm with only some of its shapes mapped (two chips in one bar)",
                                 SystemExit),
    "no-boundary-mapped":       _g(2, "ECC_RECON_ERT_AWARE=1 with not one boundary mapped", SystemExit),
    "nothing-evaluated":        _g(2, "an arm that produced no result at all", SystemExit,
                                 frozen_match="nothing was evaluated; see the [skip] lines above",
                                 frozen_file="eccenergy/study/baseline.py"),
    "stacks-refused":           _g(2, "a figure with no stacks to draw", SystemExit),
    "recon-nothing-collected":  _g(2, "a panelled placement study with a panel that collected nothing",
                                 SystemExit),
    "panels-nothing-to-plot":   _g(2, "a panel figure with no group in any panel", SystemExit),
    "sweep-nothing-to-plot":    _g(2, "a sweep figure with no group", SystemExit),

    # -- the two-arms-are-one-chip family. prompt_2's fairness rule.
    "task4-shapes-differ":      _g(2, "Task 4's two arms mapped on DIFFERENT layer shapes", SystemExit),
    "ert-arm-shapes-differ":    _g(2, "an ERT arm and the reference mapped on DIFFERENT layer shapes",
                                 SystemExit),
    "task4-capacity-not-dilated": _g(2, "a re-planned mapping whose weight capacity is not N/K the reference's",
                                   SystemExit),
    "ert-arm-capacity-wrong":   _g(2, "a quantisation arm whose Effective size is not 8/q the reference's",
                                 SystemExit),
    "no-dilatable-level":       _g(2, "Task 4 on a design with no weight level a dilation can touch",
                                 SystemExit),
    "pair-geometry":            _g(2, "the two arms declaring a different DEPTH (real silicon one arm lacks)",
                                 ValueError,
                                 note="THE ONE GUARD WITH AN OVERRIDE OUTSIDE `ECC_ALLOW`: "
                                      "`ECC_DISABLE_ASSERT_PAIR_GEOMETRY=1` (env.sh section 4) "
                                      "predates the tiers and is left exactly as it was."),
    "datawidth-levels-unknown": _g(2, "ECC_WEIGHT_DATAWIDTH_LEVELS naming a level the design has not got",
                                 ValueError),
    "narrow-once":              _g(2, "the mapper AND the evaluator both narrowing (it squares the saving)",
                                 ValueError),

    # -- the accounting families: a term charged twice, or drawn but not billed
    "figure-draws-the-total":   _g(2, "a figure whose bars do not add up to the result's total", SystemExit),
    "ert-split-does-not-reconcile": _g(2, "an access toll the stats and the evaluator disagree about",
                                     SystemExit),
    "ert-split-left-residue":   _g(2, "an access toll moved out of a category that did not empty", SystemExit),
    "ert-toll-mismatch":        _g(2, "an evaluator-charged toll that is not the stats-side amount", SystemExit),
    "boundary-declares-no-ert-row": _g(2, "moving a toll out of a boundary Timeloop never billed", SystemExit),
    "ert-arm-fails-guards":     _g(2, "an arm's own mapping failing the arm's own guards", SystemExit),

    # -- the design schema. A design is a DIRECTORY (phase 5); these are what
    # -- make a half-written one LOUD instead of silently half-registered.
    "design-dir-missing":       _g(2, "a named design with no directory"),
    "design-not-a-mapping":     _g(2, "a design.yaml that is not a mapping of fields"),
    "design-name-mismatch":     _g(2, "a design.yaml whose `name:` is not its directory"),
    "design-label-empty":       _g(2, "a design with no `label:` for a figure axis"),
    "design-free-levels-shape": _g(2, "`mapspace_free_levels:` that is not dimension -> [levels]"),
    "design-todo":              _g(2, "a scaffolded design still saying TODO where evidence goes"),
    "noc-share-needs-band":     _g(2, "`noc_published_share:` without `low:` and `high:`"),
    "noc-share-needs-citation": _g(2, "a claim about a PUBLISHED design with no citation"),
    "noc-share-low-gt-high":    _g(2, "a published share band with low above high"),
    "weight-path-empty":        _g(2, "a weight_path.yaml with no stages"),
    "weight-path-missing-field": _g(2, "a weight-path stage missing a required field"),
    "weight-path-duplicate-stage": _g(2, "two weight-path stages sharing one key"),
    "weight-path-unknown-kind": _g(2, "a weight-path stage of no known kind"),
    "weight-path-stage-no-prefixes": _g(2, "a stage that matches no Timeloop level, so it can never be measured"),
    "weight-path-starts-at-dram": _g(2, "a weight path that does not begin at the DRAM"),
    "placements-empty":         _g(2, "a placements.yaml that declares no boundary"),
    "placement-missing-field":  _g(2, "a placement missing a required field"),
    "placement-duplicate-key":  _g(2, "two placements sharing one key"),
    "placement-prefix":         _g(2, "`reduced:` naming a stage the weight path does not declare"),
    "placement-site-stage-unknown": _g(2, "a placement whose `site_stage:` is not a stage of this design"),
    "placement-site-counter-unknown": _g(2, "a placement with no known site counter"),
    "placements-need-weight-path": _g(2, "boundaries declared without the weight path they cut through"),

    # -- the probe
    "ert-probe-failed":         _g(2, "the ERT probe failing to run", SystemExit),
    "ert-probe-reference-incomplete": _g(2, "a reference cache entry the probe cannot read", SystemExit),
    "ert-probe-fingerprint-moved": _g(2, "a sidecar fingerprint that is not the current architecture",
                                     SystemExit),
    "ert-probe-no-storage-stage": _g(2, "a design with no storage stage on its weight path", SystemExit),
    "ert-probe-level-no-weights": _g(2, "a probe level carrying no Weights in the reference stats", SystemExit),

    # ---- tier 3: COUPLING -- the combinations we thought of --------------
    "rerun-needs-the-mapper":   _g(3, "ECC_RERUN_OPTIMISER=1 together with a knob that never invokes Timeloop"),
    "recon-no-split-read-write": _g(3, "ECC_SPLIT_READ_WRITE=1 inside the placement study"),
    "panels-needs-arch-or-model-sweep": _g(3, "a panel layout that would vary two axes at once"),

    # ---- tier 4: DERIVED -- refusing you for setting what is also derived -
    "zero-price":               _g(4, "a ZERO per-bit or per-MAC price",
                                 note="THE ABLATION: `what if this term were free` is the upper "
                                      "bound on how much it was worth. `> 0` was the wrong rule "
                                      "for a PRICE (ProjectRestructure Appendix B); the invariant "
                                      "is `>= 0` and `negative-price` holds the other half."),
    # `derived-datawidth` and `derived-datawidth-levels` RETIRED
    # (EnvReorganisation phase 4, 2026-09-14) with the two env reads they
    # policed: `settings/arch.py` no longer reads ECC_WEIGHT_DATAWIDTH or
    # ECC_WEIGHT_DATAWIDTH_LEVELS, so the ERT arm is the only thing that can
    # set either and the coupling they refused cannot be constructed.
}


# =============================================================================
#  ECC_ALLOW, and the record it leaves behind
# =============================================================================
#: Every override that actually FIRED this run, in the order they fired.
#: `paths.Results.write_manifest()` copies it onto every manifest and
#: `Config.recon_caveats()` onto every figure's caveat list.
_RECORDED = []


def allowed():
    """The guard ids named in `ECC_ALLOW`, as a set. Read at every call.

    LAZY ON PURPOSE, and this is the one import in the package that has to be.
    `env.py` is the only module allowed to read `os.environ` (CLAUDE.md) and its
    own five PARSE guards are registered here, so a top-level import either way
    round is a cycle. Tiers 1 and 2 never reach this function, which is exactly
    why the cycle does not have to exist: the parse layer cannot be overridden,
    so the parse layer never needs to know what an override is.
    """
    from .env import _list
    return frozenset(_list(ALLOW_VAR))     # `_list` already splits on , and space


def overrides():
    """What was overridden this run: one dict per FIRED override, for the record.

    Empty on every run that overrides nothing -- which is every run the study has
    published -- so a manifest that carries the key is a manifest of an ablation.
    """
    return list(_RECORDED)


def reset_overrides():
    """Forget what fired. For tests only; a run resolves its config once."""
    _RECORDED.clear()


def _lookup(gid):
    g = GUARDS.get(gid)
    if g is None:
        raise KeyError(
            f"{gid!r} is not a registered guard. Every refusal names an id in "
            f"eccenergy/settings/guards.py, and `make guards` regenerates "
            f"GUARDS.md from it.")
    return g


def refusal(gid, message):
    """BUILD the exception for a tier 1 or 2 guard. The caller raises it.

        raise guards.refusal("need-k-lt-n", f"need K < N; got ...")

    The `raise` stays at the site because it is the truth: these tiers always
    stop the run. The TYPE comes from the registry so that giving a site an id
    cannot quietly turn a `SystemExit` into a `ConfigError`.
    """
    g = _lookup(gid)
    if g.overridable:
        raise AssertionError(
            f"guard {gid!r} is tier {g.tier} ({TIER_NAMES[g.tier]}), which is "
            f"OVERRIDABLE -- use `guards.refuse({gid!r}, ...)`, which consults "
            f"{ALLOW_VAR} and records the override, not `refusal()`, which cannot.")
    exc = g.raises(message)
    # THE ID TRAVELS ON THE EXCEPTION (EnvReorganisation phase 6). A caller that
    # legitimately continues past ONE named refusal -- the sweep figure, whose
    # point is cold at one code and mapped at the others -- has to be able to
    # tell WHICH guard fired without matching on the message. Matching a
    # message is how a swallowed `except ConfigError` starts hiding the guards
    # it was never meant to catch; `getattr(e, "guard_id", None) in {...}` does
    # not. The message is unchanged.
    exc.guard_id = gid
    return exc


def refuse(gid, message):
    """ACT on a tier 3 or 4 guard: raise it, or -- if allowed -- warn and RETURN.

        guards.refuse("zero-price", f"...")      # no `raise` at the site

    There is no `raise` because the site may continue: that is what makes it an
    ablation rather than an impossibility. When `ECC_ALLOW` names the id the
    refusal becomes a LOUD warning on stderr and a row in `overrides()`, which
    travels onto the manifest and the figure's caveat list.
    """
    g = _lookup(gid)
    if not g.overridable:
        raise AssertionError(
            f"guard {gid!r} is tier {g.tier} ({TIER_NAMES[g.tier]}), which is "
            f"NEVER overridable -- use `raise guards.refusal({gid!r}, ...)`.")
    if gid not in allowed():
        raise g.raises(
            f"{message}\n"
            f"  -> guard `{gid}` (tier {g.tier} {TIER_NAMES[g.tier]}). This is a "
            f"combination the study has not run, NOT an impossibility: set\n"
            f"     {ALLOW_VAR}={gid}   to run it anyway. The override is recorded "
            f"in the run manifest and on the figure's caveat list.")
    _RECORDED.append({"guard": gid, "tier": g.tier,
                      "tier_name": TIER_NAMES[g.tier], "refused": message})
    sys.stderr.write(
        f"\n!! GUARD OVERRIDDEN: `{gid}` (tier {g.tier} {TIER_NAMES[g.tier]}), "
        f"because {ALLOW_VAR} names it.\n"
        f"   {message}\n"
        f"   THIS RUN IS AN ABLATION. It is recorded on every manifest and every "
        f"caveat list it produces.\n\n")
    return None
