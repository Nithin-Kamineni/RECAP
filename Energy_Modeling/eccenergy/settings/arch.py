"""The chip the mapper is handed -- every knob that reaches the patched YAML.

ANYTHING TOUCHED HERE COLDS THE WHOLE MAPPER CACHE. `IN_FINGERPRINT` below is
that statement made checkable: 19 of these 25 fields are hashed into
`arch_fingerprint()`, and a mapper cache directory is hours of SLURM.

It carries the weight geometry (THE WIDTH TABLE's inputs), the fairness knobs
that make a cross-design comparison mean anything, the NoC coefficients, and --
since prompt_7 Phase C1 -- TIME: the design's clock and the off-chip speed limit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..contracts.errors import ConfigError
from .env import _b, _f, _i, _list, _of, _oi, _one, _s, _table

#: How faithfully an architecture is modelled.
#:
#: `paper`  -- archs/<name>/arch_paper.yaml, where the storage precisions and
#:             scratchpad geometries follow the published tables. Every number
#:             in those files is commented with the paper it came from.
#: `stock`  -- the design exactly as timeloop-accelergy-exercises ships it (or,
#:             for the locally authored v2 designs, arch.yaml). Reproduces every
#:             pre-correction figure.
ARCH_FIDELITIES = ("paper", "stock")

#: WHICH DESIGNS EXIST IS NOT A KNOB. `KNOWN_ARCHS`, `ARCH_LABELS` and
#: `BRACKET_PAIRS` lived here until ProjectRestructure phase 5 and are now
#: `arch/design.py`, reading `archs/<name>/design.yaml` -- one directory per
#: design, and nothing in Python names one.
#:
#: This package may not reach up into `arch/`, so the DEFAULT list for
#: `ECC_SWEEP_ARCHS` is left empty here and filled in by `config._resolve()`,
#: which is above both. The resolved list is identical; what moved is where the
#: default comes from.

#: Which levels a Task 4 capacity dilation may rewrite.
#: `archs.WEIGHT_CAPACITY_SCOPES` must stay in step with this tuple.
WEIGHT_CAPACITY_SCOPES = ("exclusive", "shared")

#: Which workload file a model comes from. Mixing the two in one sweep is an
#: error: they live in different JSONs and have different problem generators.
CNN_MODELS = ("resnet18", "resnet50", "densenet121", "squeezenet1_1",
              "mobilenet_v2", "efficientnet_b0", "convnext_tiny", "xception")

TRANSFORMER_MODELS = ("distilgpt2", "gpt2", "bert_base", "gpt2_medium",
                      "opt_125m", "distilbert", "tinyllama")

#: The fields of this group the MAPPER sees, and therefore the ones
#: `Config.fingerprint()` hashes.
#:
#: Everything that reaches the patched arch YAML. Every one of these colds the
#: whole matrix: they ARE the chip the mapper is handed.
IN_FINGERPRINT = frozenset({
    "arch_fidelity",
    "force_technology",
    "force_datawidth",
    "weight_capacity_scale",
    "weight_capacity_scope",
    "weight_factor_relax",
    "mapspace_constrain",
    "weight_datawidth",
    "weight_datawidth_levels",
    "weight_depth_scale",
    "weight_depth_levels",
    "weight_width_glb_mult",
    "dram_depth",
    "global_cycle_seconds",
    "noc_enabled",
    "noc_wire_pj_per_bit_mm",
    "noc_router_pj",
    "noc_pe_latch_pj",
    "noc_scale",
})


@dataclass(frozen=True)
class ArchSettings:
    #: ECC_DRAM_BANDWIDTH_MBPS: the off-chip speed limit in MB/s, or None for
    #: unlimited. Read by `latency_post.py` (the evaluator-side roofline) AND,
    #: since prompt_7 C1.1, written onto the DRAM level of the YAML THE MAPPER
    #: READS, as `shared_bandwidth` -- one bus that reads and writes share,
    #: which is the same term the roofline charges. In the fingerprint through
    #: the patched text, so changing it colds every cache.
    dram_bandwidth_mbps: Optional[float]
    #: `{arch: MHz}` -- DERIVED, since 2026-09-14, from every declared design's
    #: `clock_mhz:` in archs/<name>/design.yaml (`arch.design.clock_table()`);
    #: it was env.sh's ECC_ARCH_CLOCK_MHZ table, flattened. `from_env()` leaves
    #: it None and `config._resolve()` fills it, so the record every manifest
    #: carries keeps the same key and the same values. `cycle_seconds_for()`
    #: is the ONLY place it is inverted to seconds (env.sh section 6 TRAP 2 --
    #: two conversions of one period is a silent 5x). A design with no entry
    #: keeps `global_cycle_seconds`. `cfg.with_(arch_clock_mhz={...})` still
    #: overrides it, for one `with_()`: the tests that vary a clock use that.
    arch_clock_mhz: Optional[dict]
    #: ECC_ONCHIP_BW_BITAWARE (prompt_7 C1.3). 1 = a level the arm narrows to
    #: `datawidth: q` declares its `read_bandwidth`/`write_bandwidth` x 8/q,
    #: because the port moves BITS and a q-bit weight is fewer of them. This
    #: is the lever that reaches the `fc` layers, where `filter_glb`'s declared
    #: 16 items/cycle is what caps PE utilisation at 9.52% (prompt_7 4.5).
    #: COLDS EVERY CACHE.
    onchip_bw_bitaware: bool
    arch_fidelity: str
    force_technology: str
    force_datawidth: Optional[int]
    #: DERIVED since 2026-09-14 from archs/_shared/standard.yaml
    #: `study.dram.depth_words` -- the one DRAM geometry every design shares
    #: and `validate` already audits every arch YAML against. It was
    #: ECC_DRAM_DEPTH, which repeated that number. Still in the fingerprint:
    #: `_patch_dram_depth()` writes it onto every design's DRAM level.
    dram_depth: Optional[int]
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
    #: weight level's `width:` comes from the design's own
    #: archs/<name>/widths.yaml (`q -> {spad_width, glb_width}`, THE RULE in
    #: physics/widths.py for an unlisted q), and `arch.patch.
    #: _set_weight_geometry()` applies it to every run: 96 b for an 8-bit
    #: level, 98 / 96 / 95 / 96 for q = 7 / 6 / 5 / 4, x4 above the PE array.
    #: THE ARMS DO NOT SHARE A WIDTH -- each one's width suits its own
    #: datawidth and no other arm's, which is why 95 at q=5 is correct and
    #: does not have to divide 8. Depth is renormalised at the BASE width
    #: (96), so it IS shared, and that is what `assert_pair_geometry` checks.
    #: THIS FIELD IS DERIVED since 2026-09-14: the GLB/scratchpad ratio the
    #: held design's 8-bit row declares (4, Eyeriss v1's 16-b / 64-b ratio).
    #: It was ECC_WEIGHT_WIDTH_GLB_MULT; it stays a record field so every
    #: fingerprint and manifest keeps the key and the value.
    weight_width_glb_mult: Optional[int]
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
    # Timeloop's built-in wire model is a stub returning 0, so without these
    # every network is free -- in the evaluator AND in the mapper's objective.
    noc_enabled: bool
    noc_wire_pj_per_bit_mm: Optional[float]   # override of the shared constant
    noc_router_pj: Optional[float]            # override of shared.router_pj_per_flit
    noc_pe_latch_pj: Optional[float]          # override of shared.pe_latch_pj (v2 PE row)
    noc_scale: float                          # sensitivity multiplier on all terms

    @staticmethod
    def from_env():
        """Every `ECC_*` knob of this group, read once.

        `weight_datawidth` and `weight_datawidth_levels` are usually EMPTY here
        and are filled in by `config._resolve()` from the ERT arm: the arm IS a
        datawidth configuration, and letting the knob and the arm both set them
        is how two chips end up sharing one cache directory.
        """
        return ArchSettings(
            arch_fidelity=_s("ECC_ARCH_FIDELITY", "paper").lower(),
            force_technology=_s("ECC_FORCE_TECHNOLOGY"),
            force_datawidth=_oi("ECC_FORCE_DATAWIDTH"),
            dram_depth=None,               # standard.yaml study.dram.depth_words
            global_cycle_seconds=_s("ECC_GLOBAL_CYCLE_SECONDS", "1e-9"),
            # ROUNDED AT LOAD, and that is not cosmetic. The cache slug is
            # `wcap{scale:g}`, so 63/39 spelled 1.61539 by python and 1.6154 by the
            # shell that submitted the mapping wave are the SAME architecture (both
            # round `depth: 224` to 362, so the fingerprints match) filed under two
            # different directory names -- and the evaluator then refuses a cache
            # it actually has. Four decimals is finer than any buffer depth can
            # resolve and is what env.sh documents.
            weight_capacity_scale=round(_f("ECC_WEIGHT_CAPACITY_SCALE", 1.0), 4),
            weight_capacity_scope=_s("ECC_WEIGHT_CAPACITY_SCOPE", "exclusive").lower(),
            weight_factor_relax=_b("ECC_WEIGHT_FACTOR_RELAX", False),
            mapspace_constrain=_b("ECC_MAPSPACE_CONSTRAIN", False),
            weight_datawidth=_oi("ECC_WEIGHT_DATAWIDTH"),
            # Quantised to four decimals for the same reason the capacity scale is:
            # one geometry must have exactly ONE spelling, or 0.7071 written
            # `0.71` by the shell and `0.7071` by python is the same architecture
            # filed under two cache directories.
            weight_depth_scale=round(_f("ECC_WEIGHT_DEPTH_SCALE", 1.0), 4),
            # There is NO ECC_WEIGHT_WIDTH. THE WIDTH TABLE is automatic and
            # unconditional (eccenergy/widths.py): every weight level takes
            # the width that suits the datawidth it stores, on every run, so there
            # is nothing to set and nothing that can be set wrong.
            weight_width_glb_mult=None,    # archs/<name>/widths.yaml, 8-bit row
            disable_pair_geometry_assert=_b("ECC_DISABLE_ASSERT_PAIR_GEOMETRY", False),
            weight_depth_levels=tuple(_list("ECC_WEIGHT_DEPTH_LEVELS")),
            weight_datawidth_levels=tuple(_list("ECC_WEIGHT_DATAWIDTH_LEVELS")),
            depth_sweep_gate_victories=tuple(
            _list("ECC_DEPTH_SWEEP_GATE_VICTORIES", "2000 4000 10000")),
            depth_sweep_gate_scales=tuple(
            _list("ECC_DEPTH_SWEEP_GATE_SCALES", "1.0 0.125")),
            noc_enabled=_b("ECC_NOC", True),
            noc_wire_pj_per_bit_mm=_of("ECC_NOC_WIRE_PJ_PER_BIT_MM"),
            noc_router_pj=_of("ECC_NOC_ROUTER_PJ"),
            noc_pe_latch_pj=_of("ECC_NOC_PE_LATCH_PJ"),
            noc_scale=_f("ECC_NOC_SCALE", 1.0),
            arch_clock_mhz=None,           # archs/<name>/design.yaml clock_mhz
            dram_bandwidth_mbps=_of("ECC_DRAM_BANDWIDTH_MBPS"),
            onchip_bw_bitaware=_b("ECC_ONCHIP_BW_BITAWARE", False),
        )
