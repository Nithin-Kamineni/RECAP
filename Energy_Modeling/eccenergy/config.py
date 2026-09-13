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

from .physics import widths

# ---------------------------------------------------------------- env helpers
_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n", ""}


#: The clock the reconstruction datapath's DC power report was measured at, in
#: ns. Every entry of `data/dc/BCH_N63_results.json` carries it as
#: `measurement.clock_period_ns` and every one of them is 1.0; `ecc.py` checks
#: the entry it actually reads against this and refuses on a mismatch rather
#: than rescaling from the wrong base. env.sh section 6, TRAP 2.
DC_MEASUREMENT_CLOCK_NS = 1.0


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


def _table(name):
    """`key=value;key=value` -> `{key: float(value)}` (env.sh section 10's
    flattening of a `declare -A` table, which bash cannot export)."""
    out = {}
    for entry in os.environ.get(name, "").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        key, _, val = entry.partition("=")
        try:
            out[key.strip()] = float(val)
        except ValueError:
            raise ConfigError(f"{name}: entry {entry!r} is not `key=number`") from None
    return out


def _list(name, default="", sep=None):
    """Space- or comma-separated list; anything after '#' is a comment.

    `sep=";"` splits on semicolons and keeps each entry whole, spaces and all --
    for `ECC_RECON_PLACEMENT_LIST`, whose entries are `arch=key key key`.
    """
    raw = _s(name, default).split("#", 1)[0]
    if sep:
        return [e.strip() for e in raw.split(sep) if e.strip()]
    return [tok for tok in raw.replace(",", " ").split() if tok]


def _one(name, default=""):
    """A single token. Extra tokens are an error, not a silent truncation."""
    got = _list(name, default)
    if len(got) > 1:
        raise ConfigError(f"{name} takes ONE value, not {len(got)}: {' '.join(got)}\n"
                          f"  -> only the swept axis takes a list")
    return got[0] if got else ""


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
#: A pair differs in ONE modelling judgement that the design's paper does not
#: settle, and `Session.setup()` writes the caveat into the run manifest
#: whenever a design appears without its partner, because a caveat that lives
#: only in a README does not travel with the numbers.
#:
#: THE EYERISS v1 PAIR IS RETIRED (2026-09-10, prompt_2.md / CLAUDE.md).
#: `eyeriss_like_wglb` IS Eyeriss v1: JSSC 2017 Sec. V-A publishes a
#: filter-weight allocation inside the 108 kB GLB, so the file that models it
#: as a reuse level is the design and `eyeriss_like` -- which declares
#: `!Nothing` where that allocation sits -- is retired rather than bracketed.
#: (The paper's allocation is 8 kB; this study declares 2 kB deliberately --
#: see the divergence table at the top of the arch YAML. What makes the file
#: the design is that the LEVEL EXISTS, not what size it is.) Collapsing the two
#: files to ONE design makes an entry here self-referential: it would ask the
#: run to plot a retired file beside the live one and stamp every manifest
#: with a caveat that is no longer true.
#:
#: The mechanism is kept, not deleted: it is how any future undecided
#: modelling judgement is carried onto the numbers, and CLAUDE.md's bracket
#: rule still holds for `eyeriss_v2_like_wglb`, whose extra weight level is
#: NOT in its paper -- that pair has never been registered here and is not
#: registered now, because prompt_2 does not ask for it.
BRACKET_PAIRS = {}

#: Which workload file a model comes from. Mixing the two in one sweep is an
#: error: they live in different JSONs and have different problem generators.
CNN_MODELS = ("resnet18", "resnet50", "densenet121", "squeezenet1_1",
              "mobilenet_v2", "efficientnet_b0", "convnext_tiny", "xception")
TRANSFORMER_MODELS = ("distilgpt2", "gpt2", "bert_base", "gpt2_medium",
                      "opt_125m", "distilbert", "tinyllama")

#: Human-facing names for figure axes.
#:
#: A LABEL MAY NAME A STRUCTURE, NEVER A CAPACITY (2026-09-13, prompt_7 C1.7).
#: `eyeriss_like_wglb` was labelled "(+8kB filter GLB)" after JSSC 2017's
#: allocation, but the file declares `depth: 256` = 2 kB -- confirmed and kept
#: deliberately, with the divergence recorded in the arch YAML's own header and
#: in `archs/_shared/provenance.yaml`. A figure title is the last place a reader
#: meets the design, so it must not be the one place that still quotes a number
#: the file does not declare. The label now says WHICH LEVEL EXISTS, which is
#: the real difference from the retired `eyeriss_like`, and the capacity travels
#: in the manifest where it can carry its divergence with it.
ARCH_LABELS = {
    "eyeriss_like": "Eyeriss v1",
    "eyeriss_like_wglb": "Eyeriss v1\n(+filter GLB)",
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

#: How the reduced representation is physically stored and transported, and how
#: encoder work is charged. Both are `recon.py`'s, and both are answers to
#: sections 15/16 of `02_reconstruction_dse_and_implementation.txt` rather than
#: free parameters -- the docstrings there say what each one claims.
RECON_PACKINGS = ("stream", "aligned")
RECON_GRANULARITIES = ("weight", "codeword")

#: Where the BCH decoder sits (env.sh section 4). `ondie` is the model since
#: 2026-09-09: the decoder is on the DRAM die and off the fetch path, so only
#: the k message bits cross the DRAM interface and every placement's DRAM
#: interface term scales by K/N. `controller` is the pre-2026-09-09 model kept
#: as a runnable row for the diff. `recon.DECODE_SITES` must stay in step.
RECON_DECODE_SITES = ("ondie", "controller")
#: Where a NETWORK boundary's encoders sit, and therefore how many times they
#: run: `destination` (one per destination, count = the network's
#: destination-side arrivals) or `source` (one before the fanout, count = its
#: ingresses). `recon.ENCODER_SITES` must stay in step.
RECON_ENCODER_SITES = ("destination", "source")
#: Which levels a Task 4 capacity dilation may rewrite.
#: `archs.WEIGHT_CAPACITY_SCOPES` must stay in step with this tuple.
WEIGHT_CAPACITY_SCOPES = ("exclusive", "shared")

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
    #: prompt_6 RULE 3: the two Design Compiler tables from env.sh section 6
    #: (flattened by section 10), `{configuration id: pJ}` -- incremental per
    #: codeword, idle per cycle per engine. Read by `ecc.load_recon_terms`
    #: when the JSON has no entry for the (N,K) in play. ECC_RECON_INCLUDE_IDLE
    #: is retired: the two terms are never added, each has its own denominator.
    recon_incremental_table: dict
    #: ECC_RECON_REQUIRE_GROUP_RESIDENCY (env.sh section 4). 1 = a PE-local
    #: boundary whose level holds fewer than `G_rec` weights at once is refused
    #: as `unsupported`; 0 (the default, decided 2026-09-13) charges it and
    #: REPORTS the shortfall instead. RECAP's engine accumulates retained bits
    #: as they arrive rather than needing the whole group resident in one
    #: instant, so a small tile costs buffering and accesses, not feasibility.
    recon_require_group_residency: bool
    recon_idle_table: dict
    recon_pj_override: Optional[float]
    recon_incremental_fallback_pj: float
    recon_idle_fallback_pj: float
    #: prompt_7 Issue 4. Percentage of the reconstruction engine's
    #: CLOCKED-IDLE energy that clock gating removes. 0 reproduces the
    #: pre-gating model exactly; 99.5 is the measured clock/dynamic share.
    recon_clock_gating_pct: float
    #: prompt_7, 2026-09-12. Free-text revision of the ENERGY MODEL itself --
    #: the Accelergy plug-in stack, its configs, anything that changes what a
    #: component COSTS without changing the architecture YAML. EMPTY means
    #: 'whatever the image shipped', and is hashed as if the field did not
    #: exist, so every fingerprint predating this knob is preserved exactly.
    energy_model_rev: str

    # ---- Task 3: the reconstruction PLACEMENT study (env.sh section 4) -----
    #: The placement study runs on ONE architecture, ONE model and ONE code,
    #: because each architecture has its own weight path and therefore its own
    #: list of feasible boundaries. env.sh section 4 collapses section 3's lists
    #: onto this point when `recon_modeling` is on, so `archs`/`models`/`code_k`
    #: are already the single values by the time this object exists.
    recon_modeling: bool
    recon_optimizer: bool
    #: The raw `ECC_RECON_PLACEMENT_LIST`, as env.sh section 10 flattens the
    #: `ECC_RECON_PLACEMENTS` associative array (which bash cannot export):
    #: `;`-separated `arch=key key key` entries, or a bare space-separated key
    #: list that applies to every architecture. Read it through
    #: `recon_placements_for()`, never directly.
    recon_placement_keys: list
    recon_stem: str
    recon_packing: str
    recon_granularity: str
    recon_onchip_fraction: Optional[float]
    recon_placement_charges_decode: bool
    #: `ondie` | `controller` -- see RECON_DECODE_SITES.
    recon_decode_site: str
    #: `destination` | `source` -- see RECON_ENCODER_SITES and
    #: `recon.ENCODER_SITES`. Only network boundaries depend on it.
    recon_encoder_site: str
    # ---- prompt_6: reconstruction-aware mapping ----------------------------
    #: ECC_RECON_ERT_AWARE (prompt_6 8.1): the ERT arms get their own mapping,
    #: solved with the encoder's energy in the objective, and the figure marks
    #: which bars came from one. Requires RECON_OPTIMIZER=True and
    #: ECC_PHASE=Post; anything else is refused in `__post_init__`.
    recon_ert_aware: bool
    #: ECC_RECON_ERT_ARM (prompt_6 8.3): WHICH arm one mapper job is solving --
    #: `reference` (also the empty string), or the key of an ERT-injectable
    #: placement of the single configured architecture (`recon2`, `recon4`
    #: on Eyeriss v1). Set by the launcher in `--export`, not by hand.
    #: `__post_init__` resolves a placement key into `weight_datawidth = q`
    #: and `weight_datawidth_levels` = the storage levels in its `reduced`
    #: set, so everything downstream (patched YAML, slug, fingerprint) keys
    #: on fields that already exist; `archs.ert_bump()` derives the ERT
    #: delta from it. Read it through `ert_arm()`.
    recon_ert_arm: str
    #: ECC_RECON_LAYER (prompt_6 8.2): the placement study's layer scope --
    #: one layer name, or `all`/empty for the whole model. env.sh section 10
    #: seeds ECC_LAYERS from it whenever ECC_RECON_MODELING=1, so `layers` is
    #: already the resolved scope; this field records the spelling for the
    #: manifest. The launcher (hpc/map_ert_arms.sh) sets it per job.
    recon_layer: str
    #: ECC_DRAM_PJ_PER_BIT: pJ per bit of DYNAMIC DRAM access. Rescales the
    #: whole DRAM category evaluator-side (energy.apply_dram_override), exactly
    #: as mac_pj_override rescales Compute. None = leave Accelergy's own
    #: constant (8 pJ/bit for LPDDR4 as modelled) alone. The f_if array/interface
    #: split this replaced is GONE: the whole DRAM weight term scales by K/N.
    dram_pj_per_bit: Optional[float]
    #: ECC_BASELINE_DRAM_PJ_PER_BIT: pJ per bit of DYNAMIC DRAM access for the
    #: CONVENTIONAL-ECC BASELINE ARM ONLY (baseline_dram.charge). Its array is
    #: bigger -- parity is stored beside the weights -- and it does indexing
    #: work the other two arms do not, so a bit out of it costs more: 70
    #: against 40. It is a PRICE, not traffic: decoding is on the DRAM die, so
    #: the baseline drives the same weight bits off it as the embedded arm and
    #: the parity never crosses the datapath. None = the pre-2026-09-10 model
    #: (baseline at `dram_pj_per_bit`, charged the external-parity traffic).
    baseline_dram_pj_per_bit: Optional[float]
    #: The other two terms of E_total(DRAM) = E_dynamic + E_background + E_refresh.
    #: Both 0 for now, on purpose (the study's question is on-chip energy) --
    #: modelling them is a TODO and would give the embedded arm further credit,
    #: since it holds fewer weight bits in DRAM. Units: pJ per bit-second and
    #: pJ per bit per refresh window.
    dram_background_pj: float
    dram_refresh_pj: float

    # ---- prompt_7 Phase A: standby power, and time ------------------------
    #: ECC_STATIC_ENERGY (env.sh section 6). 1 = charge COMPONENT STANDBY
    #: ENERGY -- the accelerator's own leakage -- as a physical category,
    #: `Standby`, to ALL THREE ARMS. prompt_7 Defect 2: the reconstruction
    #: engines are billed standby power from a Design Compiler run while the
    #: accelerator beside them is billed none, because `parse_stats` never read
    #: `Leakage energy (total)`, so the comparison charged one side only.
    #: 0 (the default) reproduces every pre-Phase-A total to the pJ: the
    #: category is not even in `phys_cats()`, so a cached raw record still
    #: loads. Not in the mapper fingerprint -- this is evaluator arithmetic
    #: over a mapping the mapper already chose.
    static_energy: bool
    #: ECC_LEAKAGE_NW, flattened by env.sh section 10 into
    #: ECC_LEAKAGE_NW_LIST. Replacement leakage densities in nW: `sram_bit`
    #: and `rf_bit` per STORED BIT, `mac_instance` per MAC. They replace the
    #: ERT's own `leak` rows, which are 10^3-10^4 too low and in two cases
    #: exactly 0 (prompt_7 section 5.2c: CACTI pinned to `itrs-lstp`, an
    #: Aladdin table whose register leakage is a literal 0, and a Neurosim
    #: plug-in that answered 0 pJ). POWER, not energy: the cycle period is
    #: applied at the point of use, never here (env.sh section 6 TRAP 2).
    leakage_nw: dict
    #: ECC_LATENCY_MODEL. 1 = re-time the chosen mapping with
    #: `latency_post.roofline()`. Evaluator-only and NOT in the fingerprint,
    #: exactly like the two `noc_post` terms: the plan is Timeloop's, and this
    #: states how long that plan takes once off-chip bandwidth is declared.
    latency_model: bool
    #: ECC_DRAM_BANDWIDTH_MBPS: the off-chip speed limit in MB/s, or None for
    #: unlimited. Read by `latency_post.py` (the evaluator-side roofline) AND,
    #: since prompt_7 C1.1, written onto the DRAM level of the YAML THE MAPPER
    #: READS, as `shared_bandwidth` -- one bus that reads and writes share,
    #: which is the same term the roofline charges. In the fingerprint through
    #: the patched text, so changing it colds every cache.
    dram_bandwidth_mbps: Optional[float]
    #: ECC_ARCH_CLOCK_MHZ, flattened by env.sh section 10 into
    #: ECC_ARCH_CLOCK_MHZ_LIST: `{arch: MHz}`. `cycle_seconds_for()` is the
    #: ONLY place it is inverted to seconds (env.sh section 6 TRAP 2 -- two
    #: conversions of one period is a silent 5x). A design with no entry keeps
    #: `global_cycle_seconds`.
    arch_clock_mhz: dict
    #: ECC_RECON_BW_SCALE (prompt_7 C1.2). 1 = every stage in the arm's
    #: `reduced` set declares `per_dataspace_bandwidth_consumption_scale:
    #: {Weights: ...}` -- K/N at DRAM (a bit stream) and q/8 on chip (whole
    #: weights in a narrower word). COLDS EVERY CACHE: it is in the patched
    #: YAML.
    recon_bw_scale: bool
    #: ECC_ONCHIP_BW_BITAWARE (prompt_7 C1.3). 1 = a level the arm narrows to
    #: `datawidth: q` declares its `read_bandwidth`/`write_bandwidth` x 8/q,
    #: because the port moves BITS and a q-bit weight is fewer of them. This
    #: is the lever that reaches the `fc` layers, where `filter_glb`'s declared
    #: 16 items/cycle is what caps PE utilisation at 9.52% (prompt_7 4.5).
    #: COLDS EVERY CACHE.
    onchip_bw_bitaware: bool

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
    #: TASK 4 -- CAPACITY DILATION. Multiplies the declared `depth:` of the
    #: weight-carrying storage levels in the architecture THE MAPPER SEES, so
    #: the search can spend the reduced representation's extra room on a larger
    #: weight tile and refetch less from DRAM. 1.0 is the declared design.
    #: N/K (1.6154 at BCH(63,39)) is the reconstruction arm's effective
    #: capacity; a value below 1 SHRINKS the design, which is how the study
    #: finds a point where weight capacity is the binding constraint at all --
    #: at the declared sizes it usually is not (FINDINGS 7.7). See
    #: `archs._scale_weight_capacity` for what is and is not rewritten, and
    #: why the resulting energy needs an evaluator-side correction.
    weight_capacity_scale: float
    #: `exclusive` | `shared` -- see `archs.WEIGHT_CAPACITY_SCOPES`.
    weight_capacity_scope: str
    #: TASK 4 LEVER 2. Drop the `factors:` pins on the WEIGHT-INDEXING
    #: dimensions (M, C, R, S) of the temporal constraints on weight-carrying
    #: levels, so the weight TILE can grow into the room a dilation adds.
    #: Capacity is not the only thing that caps a tile: `eyeriss_like`'s
    #: `weights_spad` declares `factors: [N=1, M=1, P=1, Q=1, S=1]`, which pins
    #: the M tile at that level to 1 and holds `weights held` at exactly 21,504
    #: from x1 to x32 (FINDINGS 7.8). Relaxing it is what makes capacity the
    #: binding constraint at all -- and it is a DIFFERENT DATAFLOW, so a design
    #: run under it must never be quoted as the published chip.
    weight_factor_relax: bool
    #: TASK 4 LEVER 3 (2026-09-10, FINDINGS 7.9). Pin every loop dimension to 1
    #: at the levels `archs.MAPSPACE_FREE_LEVELS` does not name, collapsing the
    #: index-factorization space from ~7.4e10 to something the mapper searches
    #: EXHAUSTIVELY. This is the answer to a FAILED convergence gate: raising
    #: the budget samples more of the same enormous space and the difference
    #: between two sampled points is noise, whereas an exhaustive search gives
    #: each arm its true optimum and the difference becomes architectural.
    #: A DIFFERENT DATAFLOW -- own cache slug (`mcons`), never quotable as the
    #: published chip. Designed to be run WITH `weight_factor_relax`.
    mapspace_constrain: bool
    #: PROMPT_2 (2026-09-10) -- THE ON-CHIP QUANTISATION THE MAPPER SEES.
    #: `datawidth:` on the weight-carrying storage levels, at FIXED `width:`
    #: and `depth:`. This is how the reduced representation is now expressed:
    #: Timeloop computes `block_size = width / datawidth` and bills
    #: `vector_access_energy / block_size` per weight, while CACTI is handed
    #: `depth` and `width` ONLY -- verified 2026-09-10 from
    #: `timeloop-mapper.accelergy.log` (`Calculated storage."width" as
    #: "width"`). So halving it at fixed geometry exactly halves per-weight
    #: energy and exactly doubles effective capacity with BYTE-IDENTICAL
    #: per-access read/write/leak. That is the fairness condition
    #: `weight_capacity_scale` could never meet.
    #: None = leave the YAML alone (the 8-bit baseline/embedded arm).
    #: HARD CONSTRAINT: `width % datawidth == 0` on every level it rewrites, or
    #: `timeloop-mapper` aborts (`buffer.cpp:302`). Checked before the YAML is
    #: written, never discovered per-layer.
    weight_datawidth: Optional[int]
    #: PROMPT_2 -- THE ONLY SWEPT VARIABLE: `depth:` of the on-chip weight
    #: levels. Deliberately NOT `weight_capacity_scale`, even though the two
    #: rewrite the same field: that knob triggers
    #: `recon.capacity_dilation_correction()`, which re-prices the level at the
    #: UNDILATED geometry. That correction is right when depth is standing in
    #: for a narrower word and wrong here -- a shallower array really IS a
    #: smaller array, and its cheaper access is a real saving, not an artifact
    #: to undo. Separate knob, separate cache slug (`wdepth<scale>`), no
    #: correction.
    weight_depth_scale: float
    #: PROMPT_2's WIDTH TABLE is NOT A KNOB and has no field here. Every
    #: weight level's `width:` is chosen by `widths.level_width()` from
    #: the datawidth THAT LEVEL ends up storing, and `archs`
    #: `_set_weight_geometry()` applies it to every run: 96 b for an 8-bit
    #: level, 98 / 96 / 95 / 96 for q = 7 / 6 / 5 / 4, x this multiplier above
    #: the PE array. THE ARMS DO NOT SHARE A WIDTH -- each one's width suits
    #: its own datawidth and no other arm's, which is why 95 at q=5 is correct
    #: and does not have to divide 8. Depth is renormalised at the BASE width
    #: (96), so it IS shared, and that is what `assert_pair_geometry` checks.
    #: 4x is Eyeriss v1's published 16-b spad / 64-b GLB ratio, and
    #: `q | W` implies `q | 4W`, so one table settles every weight level.
    weight_width_glb_mult: int
    #: `ECC_DISABLE_ASSERT_PAIR_GEOMETRY=1`: let the reconstruction and
    #: embedded arms declare DIFFERENT `depth:` on a weight level, for a study
    #: that varies depth between them on purpose. It disables the DEPTH check
    #: in `archs.assert_pair_geometry()` and nothing else -- there is no width
    #: check to disable, because the arms are SUPPOSED to differ in width.
    #: Default False: a depth difference nobody asked for is a void comparison.
    disable_pair_geometry_assert: bool
    #: Which weight levels `weight_depth_scale` may rewrite. Empty = all of
    #: them, which is the default and the limitation prompt_2 records: one
    #: scale moves `weights_spad` and `filter_glb` TOGETHER, so it locates the
    #: zone but cannot say which level bought it. Naming levels here is the
    #: second pass -- hold one at x1 and sweep the other.
    weight_depth_levels: tuple
    #: Which weight levels `weight_datawidth` may rewrite (prompt_6 phase 2).
    #: Empty = every weight-carrying level, which is what every run before
    #: 2026-09-11 did and what keeps their fingerprints. An ERT arm names
    #: exactly the storage levels in its placement's `reduced` set, because
    #: the narrow weights stop AT the boundary: `recon2` and `recon4` both
    #: narrow `filter_glb` and leave `weights_spad` at 8. A name no weight
    #: level has is refused by `archs._set_weight_datawidth`.
    weight_datawidth_levels: tuple
    #: PROMPT_2's convergence gate, read by `dilation --gate` and submitted by
    #: `hpc/map_depth_sweep.sh`. The budgets the EMBEDDED arm is mapped at, and
    #: the depths the gate is re-checked at -- the largest AND the smallest,
    #: because a budget that converges on a big buffer may not on a small one.
    #: Not in the mapper fingerprint: they select which caches to READ.
    depth_sweep_gate_victories: tuple
    depth_sweep_gate_scales: tuple

    # ---- interconnect (NoC) energy: archs/_shared/noc.yaml ------------------
    # Timeloop's built-in wire model is a stub returning 0, so without these
    # every network is free -- in the evaluator AND in the mapper's objective.
    noc_enabled: bool
    noc_wire_pj_per_bit_mm: Optional[float]   # override of the shared constant
    noc_router_pj: Optional[float]            # override of shared.router_pj_per_flit
    noc_pe_latch_pj: Optional[float]          # override of shared.pe_latch_pj (v2 PE row)
    noc_scale: float                          # sensitivity multiplier on all terms
    #: ECC_MAC_PJ_OVERRIDE (env.sh section 5): rescale the Compute category to
    #: MACs x this value in the EVALUATOR. None = the ERT's value. It is the
    #: denominator of every ECC percentage and nothing else: the saved pJ do not
    #: depend on it (energy.apply_mac_override, tests/test_mac_override.py).
    #: Not in the mapper fingerprint -- the MAC count is mapping-invariant.
    mac_pj_override: Optional[float]

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

        # ---- Task 3: the placement study --------------------------------
        if self.recon_packing not in RECON_PACKINGS:
            raise ConfigError(f"ECC_RECON_PACKING must be one of "
                              f"{', '.join(RECON_PACKINGS)}")
        if self.recon_granularity not in RECON_GRANULARITIES:
            raise ConfigError(f"ECC_RECON_ENCODER_GRANULARITY must be one of "
                              f"{', '.join(RECON_GRANULARITIES)}")
        if self.mac_pj_override is not None and self.mac_pj_override <= 0:
            raise ConfigError(
                f"ECC_MAC_PJ_OVERRIDE={self.mac_pj_override}: the per-MAC energy "
                f"must be positive (pJ per 8-bit MAC), or empty for the ERT's value")
        if self.recon_encoder_site not in RECON_ENCODER_SITES:
            raise ConfigError(
                f"ECC_RECON_ENCODER_SITE must be one of "
                f"{', '.join(RECON_ENCODER_SITES)} -- `destination` is one "
                f"encoder per destination of a multicast network (the count is "
                f"Timeloop's destination-side arrivals, ingresses x multicast "
                f"factor); `source` is one encoder before the fanout and is the "
                f"pre-2026-09-09 row, kept for the diff. See recon.ENCODER_SITES")
        if self.recon_decode_site not in RECON_DECODE_SITES:
            raise ConfigError(
                f"ECC_RECON_DECODE_SITE must be one of "
                f"{', '.join(RECON_DECODE_SITES)} (got {self.recon_decode_site!r}): "
                f"`ondie` puts the BCH decoder on the DRAM die, off the fetch "
                f"path, so only the k message bits cross the DRAM interface; "
                f"`controller` is the pre-2026-09-09 model kept for the diff")
        if self.dram_pj_per_bit is not None and self.dram_pj_per_bit <= 0:
            raise ConfigError(
                f"ECC_DRAM_PJ_PER_BIT={self.dram_pj_per_bit}: the per-bit DRAM "
                f"dynamic access energy must be > 0 (8 = Accelergy LPDDR4 as "
                f"modelled, 20 = Horowitz ISSCC 2014, 40 = this study's default)")
        if (self.baseline_dram_pj_per_bit is not None
                and self.baseline_dram_pj_per_bit <= 0):
            raise ConfigError(
                f"ECC_BASELINE_DRAM_PJ_PER_BIT={self.baseline_dram_pj_per_bit}: the "
                f"baseline arm's per-bit DRAM dynamic access energy must be > 0 "
                f"(70 = this study's value for the bigger, indexed conventional-ECC "
                f"array; EMPTY = the pre-2026-09-10 parity-traffic model)")
        for _n, _v in (("ECC_DRAM_BACKGROUND_PJ", self.dram_background_pj),
                       ("ECC_DRAM_REFRESH_PJ", self.dram_refresh_pj)):
            if _v < 0:
                raise ConfigError(f"{_n}={_v} must be >= 0 (0 = term not modelled)")
        if self.recon_optimizer:
            # TASK 4 IS IMPLEMENTED (2026-09-09), and the guarantee the old
            # placeholder existed to give is KEPT INTACT: a `True` here must
            # never produce fixed-mapping numbers under a heading that says the
            # mapping was optimised for reconstruction. That is now enforced
            # where it can actually be checked instead of by refusing outright.
            # `experiments/recon.dilated_view()` stops the run when the
            # reconstruction arm's OWN mapper cache is absent, when the design
            # has no weight level to dilate, or when the dilated capacity does
            # not come back N/K times the reference's; and `task4_checks()`
            # records the two mapping fingerprints side by side on every
            # result, so a figure drawn from one cache cannot claim two.
            # What is refused here is the one combination that cannot mean
            # anything: a re-optimised mapping filed as a `Pre` result.
            if self.phase != "Post":
                raise ConfigError(
                    f"RECON_OPTIMIZER=True is TASK 4: the mapping itself is "
                    f"re-optimised for the reduced weight width, so the result "
                    f"is a `Post` result by construction -- not ECC_PHASE="
                    f"{self.phase}, which means 'the mapping is ECC-unaware and "
                    f"the ECC effect is applied when evaluating'.\n"
                    f"  -> ECC_PHASE=Post RECON_OPTIMIZER=True   is Task 4\n"
                    f"  -> RECON_OPTIMIZER=False                 is Task 3, the "
                    f"fixed-mapping placement study")
        if self.experiment == "recon" and not self.recon_optimizer \
                and self.phase != "Pre":
            # The other half of the pair above. Caught HERE rather than only in
            # `experiments/recon.run()` so `--dry-run` reports it too: a
            # configuration this contradictory should never survive to a run.
            raise ConfigError(
                f"ECC_PHASE={self.phase} but RECON_OPTIMIZER=False is Task 3, "
                f"which is a `Pre` result by construction: the mapping is "
                f"fixed and ECC-unaware, and the placement effect is applied "
                f"when evaluating.\n"
                f"  -> ECC_PHASE=Pre                         is Task 3\n"
                f"  -> ECC_PHASE=Post RECON_OPTIMIZER=True   is Task 4, where "
                f"the mapping itself is solved for the reduced width")
        if self.experiment == "recon":
            if not self.archs:
                raise ConfigError(
                    "the reconstruction placement study needs at least one "
                    "architecture.\n  -> set ECC_RECON_ARCHS (env.sh section 4)")
            # SEVERAL ARCHITECTURES ARE ONE PANEL EACH, NOT ONE AXIS. Each
            # design has its own weight path and therefore its own list of
            # feasible boundaries, so they cannot share an x axis -- env.sh
            # section 4 and CLAUDE.md both say so, and `experiments/recon.py`
            # `figure()` honours it by giving every design its own axes, its own
            # boundary list and its own two reference bars. What is shared is
            # the page, the legend, the category set and the energy unit.
            # A repeated name is not an error: `archs` is de-duplicated above
            # (`dict.fromkeys`), so "a a" draws ONE panel for `a` rather than
            # the same design twice. The panel list is printed in the config
            # table and every panel heading names its design, so a typo that
            # collapses two panels into one is visible in the run.
            if len(self.models) != 1:
                raise ConfigError(
                    f"the reconstruction placement study runs on ONE model, not "
                    f"{len(self.models)} ({', '.join(self.models)}).\n"
                    f"  -> set ECC_RECON_MODEL (env.sh section 4)")
            if self.split_read_write:
                raise ConfigError(
                    "ECC_SPLIT_READ_WRITE=1 splits the on-chip categories in "
                    "proportion to their ACCESS COUNTS, but a reconstruction "
                    "placement changes the read and write bit-volumes by "
                    "different factors, so the split would be attributed "
                    "wrongly.\n  -> run the placement study with "
                    "ECC_SPLIT_READ_WRITE=0")
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
        if self.weight_capacity_scale <= 0:
            raise ConfigError(
                f"ECC_WEIGHT_CAPACITY_SCALE={self.weight_capacity_scale}: the "
                f"weight-capacity multiplier must be positive. 1.0 is the "
                f"declared design; N/K = {self.code_n / self.code_k:.4f} at "
                f"BCH({self.code_n},{self.code_k}) is the reconstruction arm's "
                f"effective capacity; below 1 shrinks the design")
        if self.weight_capacity_scope not in WEIGHT_CAPACITY_SCOPES:
            raise ConfigError(
                f"ECC_WEIGHT_CAPACITY_SCOPE must be one of "
                f"{', '.join(WEIGHT_CAPACITY_SCOPES)} -- `exclusive` dilates "
                f"only a level whose keep list is Weights alone, so the room "
                f"can only go to weights; `shared` also dilates a level that "
                f"holds Weights beside another dataspace, which hands the "
                f"mapper free capacity for that dataspace too. They bracket "
                f"one design and are quoted as a pair")
        if self.weight_depth_scale <= 0:
            raise ConfigError(
                f"ECC_WEIGHT_DEPTH_SCALE={self.weight_depth_scale}: the depth "
                f"multiplier must be positive. 1.0 is the declared design; "
                f"prompt_2's search grid is the sqrt(2) ladder "
                f"1 / 0.71 / 0.5 / 0.35 / 0.25 / 0.18 / 0.125")
        if self.weight_width_glb_mult < 1:
            raise ConfigError(
                f"ECC_WEIGHT_WIDTH_GLB_MULT={self.weight_width_glb_mult}: a "
                f"weight GLB's word is a positive multiple of the "
                f"scratchpad's. 4 is Eyeriss v1's published ratio.")
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
            raise ConfigError(
                f"ECC_WEIGHT_DATAWIDTH={self.weight_datawidth}: the on-chip "
                f"weight datawidth must be a positive integer number of bits. "
                f"Leave it EMPTY for the 8-bit baseline/embedded arm; set it "
                f"to round(8*K/N) for the reconstruction arm (4 at "
                f"BCH(63,30)). Per-code values are tabulated in prompt_2.md.")
        if (self.weight_datawidth is not None
                and self.weight_datawidth > self.weight_bits):
            raise ConfigError(
                f"ECC_WEIGHT_DATAWIDTH={self.weight_datawidth} exceeds "
                f"ECC_WEIGHT_BITS={self.weight_bits}. The reconstruction arm "
                f"stores a REDUCED weight; a wider one is not a code rate.")
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

        # ---- prompt_6: reconstruction-aware mapping ----------------------
        if self.recon_ert_aware and not (self.recon_optimizer and self.phase == "Post"):
            raise ConfigError(
                f"ECC_RECON_ERT_AWARE=1 puts the encoder's energy into the mapper's "
                f"objective, so the ERT arms are RE-MAPPED: that is Task 4 extended, "
                f"and it needs RECON_OPTIMIZER=True and ECC_PHASE=Post (got "
                f"RECON_OPTIMIZER={self.recon_optimizer}, ECC_PHASE={self.phase}).")
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
                raise ConfigError(
                    f"ECC_RECON_ERT_ARM={self.recon_ert_arm!r} names the arm of ONE "
                    f"mapper job on ONE architecture; this configuration has "
                    f"{len(self.archs)}: {', '.join(self.archs)}")
            from . import recon as _recon      # recon imports nothing of ours but embedded
            try:
                spec = _recon.mapper_arm_spec(self.archs[0], self.recon_ert_arm, self)
            except (KeyError, ValueError) as exc:
                raise ConfigError(f"ECC_RECON_ERT_ARM={self.recon_ert_arm!r}: {exc}") from None
            q = widths.declared_datawidth(self.code_n, self.code_k)
            if (self.weight_datawidth is not None and spec["narrow_levels"]
                    and self.weight_datawidth != q):
                raise ConfigError(
                    f"ECC_RECON_ERT_ARM={self.recon_ert_arm} declares datawidth q = "
                    f"round({self.weight_bits}*{self.code_k}/{self.code_n}) = {q}, but "
                    f"ECC_WEIGHT_DATAWIDTH={self.weight_datawidth}. Leave it EMPTY: the arm "
                    f"sets it.")
            if (self.weight_datawidth_levels
                    and tuple(self.weight_datawidth_levels) != tuple(spec["narrow_levels"])):
                raise ConfigError(
                    f"ECC_RECON_ERT_ARM={self.recon_ert_arm} narrows "
                    f"{'+'.join(spec['narrow_levels']) or 'NOTHING on chip'} (the storage "
                    f"levels in its placement's reduced set), but "
                    f"ECC_WEIGHT_DATAWIDTH_LEVELS="
                    f"{'+'.join(self.weight_datawidth_levels)}. Leave it EMPTY: the arm "
                    f"sets it.")
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
            # two disambiguating suffixes below. env.sh section 10 sets it
            # so an architecture sweep always lands on `ArchitectureSweep.png`
            # whether it drew one model or a panel per model. The cost is
            # real: a one-model and a two-model sweep then overwrite each
            # other, and only the manifest beside the figure records which
            # is on disk. Do not set it by hand for a run with ECC_LAYERS --
            # that is exactly the case the layer suffix exists to protect.
            # ONE sanctioned exception (prompt_6 9): env.sh section 10 keeps
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
            base = self.recon_stem or "ReconSweep"
            if self.recon_optimizer:
                # TASK 4 OWNS ITS OWN NAME, on a layer-scoped run too. Its bars
                # come from a mapping solved against N/K more weight room, so
                # they are not comparable with the fixed-mapping figure and
                # must never overwrite it. env.sh section 10 appends the same
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
        from .physics.embedded import EmbeddedLayout
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
        parts.append(f"wt{widths.base_width(self.weight_bits)}"
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
        from . import recon as _recon
        return _recon.mapper_arm_spec(self.archs[0], key, self)

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

    # ------------------------------------------- prompt_7 Phase C: the clock
    def cycle_seconds_for(self, arch):
        """THIS DESIGN's clock period, as the string `globals.yaml` carries.

        prompt_7 C1.5, Issue 14. `ECC_GLOBAL_CYCLE_SECONDS` is ONE number for
        every design and it was 1 GHz, while Eyeriss v1 silicon runs at
        200 MHz; pairing a real chip's ABSOLUTE off-chip MB/s with a 5x faster
        model clock makes the modelled chip ~5x more memory-starved than the
        one the paper describes.

        THE ONLY PLACE MHz BECOMES SECONDS. env.sh section 6's TRAP 2 is that
        a per-cycle constant converted twice, or not at all, is a silent 5x on
        every standby and idle term; keeping the inversion here means no
        caller can do either. A design with no entry in `ECC_ARCH_CLOCK_MHZ`
        keeps `global_cycle_seconds` unchanged, so the table ADDS designs
        rather than replacing the study default.

        Returned as a STRING because that is what the fingerprint hashes and
        what globals.yaml prints; `repr` of a float would make `5e-09` and
        `5.0e-09` two different architectures.
        """
        mhz = (self.arch_clock_mhz or {}).get(arch)
        if not mhz:
            return self.global_cycle_seconds
        if float(mhz) <= 0:
            raise ConfigError(f"ECC_ARCH_CLOCK_MHZ[{arch}]={mhz}: a clock rate "
                              f"must be positive")
        secs = 1.0 / (float(mhz) * 1e6)
        # A DESIGN THAT IS ALREADY AT THE STUDY DEFAULT KEEPS THE DEFAULT'S
        # EXACT SPELLING. Seven of the eight entries in `ECC_ARCH_CLOCK_MHZ`
        # are 1000 MHz, which is `global_cycle_seconds` itself -- and `%.6g` of
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

        env.sh section 6, TRAP 2, and it is live for the first time in
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

        env.sh section 6 states the conversion and there are exactly two
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

    # ------------------------- prompt_7 Phase C: what THIS arm declares, C1.2/C1.3
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
        """`{level: {factor, timing, kind, dataspace, why}}` -- the per-dataspace
        bandwidth scale THIS arm declares on `arch` (prompt_7 C1.2), or `{}`.

        Empty for the reference arm and empty with `ECC_RECON_BW_SCALE=0`,
        which is what reproduces the pre-Phase-C architecture byte for byte.
        """
        if not self.recon_bw_scale:
            return {}
        spec = self._arm_for(arch)
        if spec is None or spec["placement"] is None:
            return {}
        from . import recon as _recon
        return _recon.arm_bw_factors(spec["placement"],
                                     _recon.stages_for(arch, self), self)

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
        from .archs import mac_candidates            # lazy: archs imports config
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

    def recon_placements_for(self, arch):
        """Which boundaries to draw for `arch`: `[]` means every one it defines.

        `ECC_RECON_PLACEMENT_LIST` is how env.sh section 10 flattens the
        `ECC_RECON_PLACEMENTS` associative array, which bash cannot export.
        Two accepted forms, and the per-architecture one wins:

            arch=recon1 recon2;other=recon1     per architecture
            recon1 recon2                       every architecture

        An architecture named with an empty list, or not named at all, draws
        every placement `recon.PLACEMENTS` defines for it -- which is the normal
        thing to want and what an empty knob gives.
        """
        entries = self.recon_placement_keys
        if any("=" in e for e in entries):
            for e in entries:
                if "=" not in e:
                    continue
                name, _, keys = e.partition("=")
                if name.strip() == arch:
                    return [k.lower() for k in keys.replace(",", " ").split()]
            return []
        return [k.lower() for e in entries
                for k in e.replace(",", " ").split()]

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
                from . import recon as _recon
                arms = sorted({a.key for d in self.archs
                               for a in _recon.mapper_arms(d, self)
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

        The optimiser figure has ONE path whatever the scope (env.sh section 10,
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
        if self.recon_decode_site == "controller":
            return ("Controller-side correction (ECC_RECON_DECODE_SITE=controller, "
                    "pre-2026-09-09 model): complete codeword read and driven "
                    "off die, DRAM identical on every bar")
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
                "force_datawidth", "weight_capacity_scale",
                "weight_capacity_scope", "weight_factor_relax",
                "mapspace_constrain",
                "weight_datawidth", "weight_datawidth_levels", "recon_ert_arm",
                "weight_depth_scale", "weight_depth_levels",
                "weight_width_glb_mult",
                "dram_depth", "global_cycle_seconds",
                "noc_enabled", "noc_wire_pj_per_bit_mm", "noc_router_pj",
                "noc_pe_latch_pj", "noc_scale",
                "opt_metric", "victory", "victory_scaling", "mapper_algorithm",
                "mapper_seed", "mapper_timeout", "mapper_search_size",
                "mapper_max_permutations")
        # prompt_7: the fingerprint hashes the ARCHITECTURE, not the price list
        # Accelergy derives from it. So a corrected estimator -- the Neurosim
        # plug-in that returned 0 pJ for every address generator, say -- changes
        # every energy in the cache while leaving the path it is stored under
        # identical, and the stale entries are reused with nothing to say so.
        # `ECC_ENERGY_MODEL_REV` closes that: set it to any non-empty string and
        # the whole matrix colds DELIBERATELY. It is appended only when set, so
        # EMPTY hashes byte-identically to every fingerprint that predates it.
        d = self.to_dict()
        keys = keep + (("energy_model_rev",) if self.energy_model_rev else ())
        blob = json.dumps({k: d[k] for k in keys}, sort_keys=True)
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
        recon_incremental_table=_table("ECC_RECON_INCREMENTAL_PJ_LIST"),
        recon_require_group_residency=_b("ECC_RECON_REQUIRE_GROUP_RESIDENCY", False),
        recon_idle_table=_table("ECC_RECON_IDLE_PJ_LIST"),
        recon_pj_override=_of("ECC_RECON_PJ"),
        recon_incremental_fallback_pj=_f("ECC_RECON_INCREMENTAL_FALLBACK_PJ", 1.8995),
        recon_idle_fallback_pj=_f("ECC_RECON_IDLE_FALLBACK_PJ", 2.2301273),
        recon_clock_gating_pct=_f("ECC_RECON_CLOCK_GATING_PCT", 99.5),
        energy_model_rev=_s("ECC_ENERGY_MODEL_REV", ""),

        recon_modeling=_b("ECC_RECON_MODELING", False),
        # env.sh spells this `RECON_OPTIMIZER` as well, and mirrors it into the
        # ECC_-prefixed name every other knob uses.
        recon_optimizer=_b("ECC_RECON_OPTIMIZER", False),
        recon_placement_keys=_list("ECC_RECON_PLACEMENT_LIST", sep=";"),
        recon_stem=_s("ECC_RECON_STEM", "ReconSweep"),
        recon_packing=_s("ECC_RECON_PACKING", "stream").lower(),
        recon_granularity=_s("ECC_RECON_ENCODER_GRANULARITY", "weight").lower(),
        recon_onchip_fraction=_of("ECC_RECON_ONCHIP_FRACTION"),
        recon_placement_charges_decode=_b("ECC_RECON_PLACEMENT_CHARGES_DECODE", True),
        recon_decode_site=_s("ECC_RECON_DECODE_SITE", "ondie").lower(),
        recon_encoder_site=_s("ECC_RECON_ENCODER_SITE", "destination").lower(),
        recon_ert_aware=_b("ECC_RECON_ERT_AWARE", False),
        recon_ert_arm=_s("ECC_RECON_ERT_ARM").strip().lower(),
        recon_layer=_s("ECC_RECON_LAYER").strip(),
        dram_pj_per_bit=_of("ECC_DRAM_PJ_PER_BIT"),
        baseline_dram_pj_per_bit=_of("ECC_BASELINE_DRAM_PJ_PER_BIT"),
        dram_background_pj=_f("ECC_DRAM_BACKGROUND_PJ", 0.0),
        dram_refresh_pj=_f("ECC_DRAM_REFRESH_PJ", 0.0),

        static_energy=_b("ECC_STATIC_ENERGY", False),
        leakage_nw=_table("ECC_LEAKAGE_NW_LIST"),
        latency_model=_b("ECC_LATENCY_MODEL", False),
        dram_bandwidth_mbps=_of("ECC_DRAM_BANDWIDTH_MBPS"),
        arch_clock_mhz=_table("ECC_ARCH_CLOCK_MHZ_LIST"),
        recon_bw_scale=_b("ECC_RECON_BW_SCALE", False),
        onchip_bw_bitaware=_b("ECC_ONCHIP_BW_BITAWARE", False),

        weak_enabled=_b("ECC_WEAK", False),
        weak_n=_i("ECC_WEAK_N", 63),
        weak_k=_i("ECC_WEAK_K", 57),

        baseline_inflates_onchip=_b("ECC_BASELINE_INFLATES_ONCHIP", False),
        split_read_write=_b("ECC_SPLIT_READ_WRITE", False),
        classify_mode=_s("ECC_CLASSIFY", "instances").lower(),

        arch_fidelity=_s("ECC_ARCH_FIDELITY", "paper").lower(),
        force_technology=_s("ECC_FORCE_TECHNOLOGY"),
        force_datawidth=_oi("ECC_FORCE_DATAWIDTH"),
        # ROUNDED AT LOAD, and that is not cosmetic. The cache slug is
        # `wcap{scale:g}`, so 63/39 spelled 1.61539 by python and 1.6154 by the
        # shell that submitted the mapping wave are the SAME architecture (both
        # round `depth: 224` to 362, so the fingerprints match) filed under two
        # different directory names -- and the evaluator then refuses a cache
        # it actually has. Four decimals is finer than any buffer depth can
        # resolve and is what env.sh documents.
        weight_capacity_scale=round(_f("ECC_WEIGHT_CAPACITY_SCALE", 1.0), 4),
        weight_capacity_scope=_s("ECC_WEIGHT_CAPACITY_SCOPE", "exclusive").lower(),
        # Quantised to four decimals for the same reason the capacity scale is:
        # one geometry must have exactly ONE spelling, or 0.7071 written
        # `0.71` by the shell and `0.7071` by python is the same architecture
        # filed under two cache directories.
        weight_depth_scale=round(_f("ECC_WEIGHT_DEPTH_SCALE", 1.0), 4),
        weight_depth_levels=tuple(_list("ECC_WEIGHT_DEPTH_LEVELS")),
        weight_datawidth=_oi("ECC_WEIGHT_DATAWIDTH"),
        weight_datawidth_levels=tuple(_list("ECC_WEIGHT_DATAWIDTH_LEVELS")),
        mapspace_constrain=_b("ECC_MAPSPACE_CONSTRAIN", False),
        # There is NO ECC_WEIGHT_WIDTH. THE WIDTH TABLE is automatic and
        # unconditional (eccenergy/widths.py): every weight level takes
        # the width that suits the datawidth it stores, on every run, so there
        # is nothing to set and nothing that can be set wrong.
        weight_width_glb_mult=int(_f("ECC_WEIGHT_WIDTH_GLB_MULT", 4)),
        disable_pair_geometry_assert=_b("ECC_DISABLE_ASSERT_PAIR_GEOMETRY", False),
        depth_sweep_gate_victories=tuple(
            _list("ECC_DEPTH_SWEEP_GATE_VICTORIES", "2000 4000 10000")),
        depth_sweep_gate_scales=tuple(
            _list("ECC_DEPTH_SWEEP_GATE_SCALES", "1.0 0.125")),
        weight_factor_relax=_b("ECC_WEIGHT_FACTOR_RELAX", False),
        dram_depth=_i("ECC_DRAM_DEPTH", 1048576),
        global_cycle_seconds=_s("ECC_GLOBAL_CYCLE_SECONDS", "1e-9"),

        noc_enabled=_b("ECC_NOC", True),
        noc_wire_pj_per_bit_mm=_of("ECC_NOC_WIRE_PJ_PER_BIT_MM"),
        noc_router_pj=_of("ECC_NOC_ROUTER_PJ"),
        noc_pe_latch_pj=_of("ECC_NOC_PE_LATCH_PJ"),
        noc_scale=_f("ECC_NOC_SCALE", 1.0),
        mac_pj_override=_of("ECC_MAC_PJ_OVERRIDE"),

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


def _ert_arm_row(cfg):
    """One banner line naming the ERT arm this job maps (prompt_6 8.3)."""
    if cfg.recon_ert_arm == "reference":
        return "reference (no ERT toll; the published 8-bit chip)"
    try:
        from . import archs as _archs
        b = _archs.ert_bump(cfg.archs[0], cfg)
        return (f"{b['placement']}: {b['level']}.{b['action']} += "
                f"{b['access_delta_pj']:.6f} pJ (E_w {b['e_w_pj']:.6f} x block_size "
                f"{b['block_size']}), {b['level']}.leak += {b['leak_delta_pj']:.7f} "
                f"pJ/instance/cycle; datawidth {cfg.weight_datawidth} on "
                f"{'+'.join(cfg.weight_datawidth_levels)}")
    except Exception as exc:                 # the banner must never kill a run
        return f"{cfg.recon_ert_arm} (bump unavailable: {exc})"


def banner(cfg, recon_terms, recon_provenance):
    """`recon_terms` is `(incremental pJ/codeword, idle pJ/cycle/engine)` --
    prompt_6 RULE 3's two denominators -- or the old single float."""
    if isinstance(recon_terms, (tuple, list)):
        recon_inc, recon_idle = recon_terms
    else:
        recon_inc, recon_idle = recon_terms, None
    w = 78
    # The placement study's axis is not one of the three sweeps -- saying
    # "sweep=arch" there would name the axis it HOLDS.
    axis = ("placement (where the reconstruction boundary sits)"
            if cfg.experiment == "recon"
            else f"sweep={cfg.sweep} ({cfg.swept_axis})")
    lines = ["=" * w,
             f"ecc-energy  |  {axis}  |  output={cfg.stem}",
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
            ("DRAM weight traffic", cfg.baseline_dram_line),
            ("recon on-chip scale", f"{cfg.sram_scale:.4f} (weights only)"),
            ("reconstruction", f"{recon_inc:.7f} pJ per codeword (incremental)"
                               + (f" + {recon_idle:.7f} pJ per cycle per engine (idle; "
                                  f"RULE 3: separate denominators, never added)"
                                  if recon_idle is not None else "")),
            ("recon provenance", recon_provenance),
        ]
    if cfg.experiment == "recon":
        rows += [
            ("PLACEMENT STUDY", f"Task 3: fixed mapping, the boundary is the "
                                f"axis, one panel per architecture -> {cfg.stem}"),
            ("panels", "  |  ".join(
                f"{cfg.arch_label(a).replace(chr(10), ' ')}: "
                + (", ".join(cfg.recon_placements_for(a))
                   or "every placement it defines")
                for a in cfg.archs)),
            ("reduced form packing", cfg.recon_packing
                + ("   (retained k bits packed with no per-weight alignment; "
                   "every reduced stage scales by K/N)"
                   if cfg.recon_packing == "stream" else
                   "   (whole bits per weight, no repacking: wire falls, "
                   "SRAM access count may not)")),
            ("encoder charging", cfg.recon_granularity
                + ("   (amortized: per weight, at the per-codeword energy per "
                   "n/weight_bits weights)" if cfg.recon_granularity == "weight"
                   else "   (pessimistic: one whole codeword per access)")),
            ("mapping optimiser",
             (f"TASK 4: the reconstruction arm is re-mapped at weight capacity "
              f"x{cfg.weight_capacity_scale * cfg.code_n / cfg.code_k:.4f} "
              f"(= x{cfg.weight_capacity_scale:g} x N/K), scope "
              f"{cfg.weight_capacity_scope}; the reference bars keep "
              f"x{cfg.weight_capacity_scale:g}"
              if cfg.recon_optimizer else
              "NOT re-run (RECON_OPTIMIZER=False) -- Task 4 is where the "
              "mapping becomes aware of the reduced width")),
            ("ERT-aware mapping",
             ("ON (prompt_6): the ERT-injectable boundaries are re-mapped with "
              "the encoder toll in the objective; the figure marks them"
              if cfg.recon_ert_aware else "off (ECC_RECON_ERT_AWARE=0)")),
            ("ERT arm", _ert_arm_row(cfg)),
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
