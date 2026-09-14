"""The weight path, read back out of the cached Timeloop output.

`weight_path()` turns one design's stage table (`arch/weight_path.py`) plus a
cached `timeloop-mapper.stats.txt` into per-stage weight energy, access counts
and packing -- the quantities every placement bar is computed from.

WHERE THE NUMBERS COME FROM
---------------------------
Everything is re-derived from the SAME cached Timeloop output the baseline and
embedded arms use -- `timeloop-mapper.stats.txt` and `timeloop-mapper.map.txt`
in the mapper cache. Nothing here can change a mapping: the mapper is never
invoked (`ECC_FROM_CACHE=1`), and the per-level energies this module scales are
cross-checked against the `Raw` record's own category totals before any
placement is evaluated. Four quantities the aggregated `Raw` record does not
carry are parsed here:

  total access counts   Timeloop prints scalar accesses PER INSTANCE. The
                        reconstruction count is a total, so per-instance counts
                        are multiplied by `Utilized instances (max)`.
  utilized capacity     how many weights are resident in a PE scratchpad. This
                        is the feasibility test for a PE-local boundary.
  physical packing      `Block size` x `Word bits` -- the physical word a scalar
                        access is a fraction of. Sec. 16 of
                        02_reconstruction_dse_and_implementation.txt: storing
                        4-bit values in 8-bit fields halves nothing.
  the loop nest         the temporal loops BELOW the innermost weight buffer,
                        which is what decides whether a reconstructed-weight
                        reuse register ever gets a hit.

`weight_path()` refuses if a level carrying Weights energy is not claimed by
exactly one stage, and `StageStats.engine_cycles` is the per-layer denominator
RULE 3's idle term is billed over.

ProjectRestructure phase 3 cut this out of `recon.py`. It is L3 -- parsing what
Timeloop wrote is toolchain work -- and it is named for the view it takes, since
`toolchain/stats.py` is the general parser it stands on.
"""
from __future__ import annotations

import math
import pathlib
import re

from dataclasses import dataclass, field

from ..arch.placements import stages_for
from ..arch.weight_path import DECODE_SITE


# ===========================================================================
#  reading the weight path out of the cached Timeloop output
# ===========================================================================
_NUM = r"([\d.eE+-]+)"


def _grab(pattern, text, cast=float):
    m = re.search(pattern, text)
    if m is None:
        return None
    try:
        return cast(m.group(1))
    except ValueError:
        return None


def _dataspace_block(body, dataspace):
    """The `Weights:` sub-block of one level's STATS section."""
    m = re.search(rf"\n\s+{dataspace}\s*:\s*\n(.*?)(?=\n\s+(?:Weights|Inputs|Outputs)\s*:\s*\n|\Z)",
                  body, re.S)
    return m.group(1) if m else ""


@dataclass
class StageStats:
    """Weight-only totals for one weight-path stage of one layer."""
    key: str
    kind: str
    levels: list = field(default_factory=list)
    energy_pJ: float = 0.0
    wire_pJ: float = 0.0          # networks: the bit-proportional half
    switch_pJ: float = 0.0        # networks: router + ingress switching
    reads: float = 0.0            # TOTAL scalar reads (all instances)
    fills: float = 0.0            # TOTAL scalar fills (all instances)
    ingresses: float = 0.0        # TOTAL network ingresses (all instances)
    deliveries: float = 0.0       # TOTAL destination-side arrivals = ingresses
                                  # x multicast factor (all instances)
    multicast: float = 1.0        # weight-only multicast factor of the network
    fanout: float = 1.0           # weight-only fanout of the network
    utilized_capacity: float = 0.0   # storage: weights resident per instance
    block_bits: int = 0              # storage: block size x word bits
    word_bits: int = 0               # storage: MEASURED `Word bits` of the level
                                     # (prompt_6 RULE 1: q -> the mapper narrowed
                                     # it; weight_bits -> the evaluator does)
    instances: float = 0.0           # utilized instances (max over layers)
    declared_instances: float = 0.0  # storage: the level's declared `Instances`
                                     # (the engines that EXIST)
    engine_cycles: float = 0.0       # RULE 3's idle denominator, summed over
                                     # layers: engines that LEAK x that layer's
                                     # cycles. Storage: UTILIZED instances --
                                     # Timeloop power-gates each unused instance
                                     # (`Instances sharing power gating: 1`) and
                                     # bills leak x utilized x cycles
                                     # (buffer.cpp FinalizeBufferEnergy, verified
                                     # 2026-09-11 on 43 shapes); dram: 1;
                                     # network: fanout x instances.
    level_share: float = 1.0      # dram: this stage's share of the Timeloop level

    def counter(self, name):
        if name == "reads":
            return self.reads
        if name == "fills":
            return self.fills
        if name == "ingresses":
            return self.ingresses
        if name == "deliveries":
            return self.deliveries
        raise ValueError(f"unknown access counter {name!r}")

    def to_dict(self):
        return {
            "stage": self.key, "kind": self.kind, "timeloop_levels": self.levels,
            "weight_energy_pJ": self.energy_pJ,
            "weight_wire_energy_pJ": self.wire_pJ,
            "weight_switching_energy_pJ": self.switch_pJ,
            "weight_scalar_reads_total": self.reads,
            "weight_scalar_fills_total": self.fills,
            "weight_network_ingresses_total": self.ingresses,
            "weight_network_deliveries_total": self.deliveries,
            "network_multicast_factor": self.multicast,
            "network_fanout": self.fanout,
            "weights_resident_per_instance": self.utilized_capacity,
            "physical_word_bits": self.block_bits,
            "measured_word_bits": self.word_bits,
            "instances": self.instances,
            "declared_instances": self.declared_instances,
            "engine_cycles": self.engine_cycles,
            "share_of_the_timeloop_level": self.level_share,
        }


@dataclass
class LayerWeightPath:
    """One layer's weight path: every stage, plus what the loop nest implies."""
    layer: str
    shape: str
    weights: int
    scale: float
    stages: dict
    unclaimed: list                 # weight-carrying levels no stage claimed
    stats_path: str
    cycles: float = 0.0             # this plan's `Cycles:` x repeat count (RULE 3)

    def total_weight_energy(self):
        return sum(s.energy_pJ for s in self.stages.values())


_LOOP = re.compile(r"for\s+([A-Za-z]+)\s+in\s+\[0:(\d+)\)(\s*\(Spatial-[XY]\))?")

#: Loop dimensions a weight depends on. C and M are per group and G selects the
#: group, so all five change WHICH weight is being used; P, Q and N do not.
WEIGHT_DIMS = frozenset("CMRSG")


#: Order the embedding pipeline flattens a conv weight tensor in: [M][C][R][S]
#: row-major, so S is the fastest-varying index and M the slowest. It decides
#: whether two consecutively ACCESSED weights are consecutive in the codeword
#: BIT STREAM, which is the only way a group-sized reconstruction buffer can
#: serve a second access without rebuilding.
STREAM_ORDER = ("M", "C", "R", "S")




def _network_split(specs, block, energy_total, instances):
    """Split one network's weight energy into wire and switching.

    Timeloop's Legacy network model (model/network-legacy.cpp:408) is

        energy = wire_hops x energy_per_hop
               + routers_touched x router_energy
               + ingresses x ingress_energy

    with `energy_per_hop = word_bits x tile_width_mm x wire_energy` and
    `routers_touched = (1 + floor(hops)) x ingresses`. Only the first term is
    proportional to the number of BITS moved, so only the first term falls
    automatically when the reduced form is transported; the switching term falls
    only if the reduced values are REPACKED into fewer flits. Every field below
    is printed by Timeloop, so the split is read off its own numbers rather than
    modelled -- and it is then rescaled to the energy Timeloop reported, so the
    two halves always sum to the figure the baseline arm is charged.
    """
    hops = _grab(rf"Average number of hops\s*:\s*{_NUM}", block) or 0.0
    per_hop_fj = _grab(rf"Energy \(per-hop\)\s*:\s*{_NUM}\s*fJ", block)
    per_hop = (per_hop_fj / 1000.0) if per_hop_fj is not None else 0.0
    ingress = _grab(rf"Ingresses\s*:\s*{_NUM}", block) or 0.0
    router_pj = _grab(rf"Router energy\s*:\s*{_NUM}\s*pJ", specs) or 0.0
    ingress_pj = _grab(rf"Ingress energy\s*:\s*{_NUM}\s*pJ", specs) or 0.0

    wire = ingress * hops * per_hop * instances
    switch = ((1.0 + math.floor(hops)) * ingress * router_pj
              + ingress * ingress_pj) * instances
    model = wire + switch
    if model <= 0 or energy_total <= 0:
        # Nothing to split (link-transfer or spatial-reduction energy only).
        # Charge it as switching: it is not proportional to the word width.
        return 0.0, energy_total, model
    k = energy_total / model
    return wire * k, switch * k, model




def _level_shares(claimants, level):
    """How much of one Timeloop level each claiming stage is.

    Exactly one claimant owns a level. The `dram_array`/`dram_interface` pair
    that used to split the DRAM level by `f_if` was removed on 2026-09-09 -- the
    DRAM level is one `dram` stage and the whole of it is reducible -- so a
    level claimed twice is now always a table error.
    """
    if len(claimants) == 1:
        return {claimants[0].key: 1.0}
    raise ValueError(
        f"Timeloop level {level!r} is claimed by {[s.key for s in claimants]}; "
        f"a level may be claimed by exactly one stage")


def apply_dram_pj_per_bit(stages, cfg):
    """Rescale the `dram` stage to ECC_DRAM_PJ_PER_BIT, in place.

    THE WEIGHT PATH IS RE-PARSED FROM THE CACHED TIMELOOP OUTPUT, so it carries
    Accelergy's own per-bit DRAM constant (8.0 pJ/bit for LPDDR4 as modelled),
    while the `Raw` record it must reconcile against has already been rescaled
    by `energy.apply_dram_override()`. Without this the two disagree by the
    override ratio and `cross_check()` fails -- which is exactly what happened
    when ECC_DRAM_PJ_PER_BIT was introduced, so the check earned its keep.

    The ERT per-bit is read back off the stage itself (energy / (reads x
    weight_bits)), so this needs no second source of truth for the 8.0.
    """
    tgt = getattr(cfg, "dram_pj_per_bit", None)
    if tgt is None:
        return stages
    for st in stages.values():
        if st.kind != "dram":
            continue
        bits = float(st.reads or 0.0) * float(cfg.weight_bits)
        if bits <= 0 or st.energy_pJ <= 0:
            continue
        ert = st.energy_pJ / bits
        st.energy_pJ *= float(tgt) / ert
    return stages


def read_weight_path(cfg, arch, layer, stats_path):
    """Parse ONE layer's weight path out of its cached Timeloop output.

    `stats_path` is the `timeloop-mapper.stats.txt` the mapper cache holds for
    this layer shape; `timeloop-mapper.map.txt` beside it supplies the loop
    nest. Both are read, never written, and neither can be produced by this
    module -- if a shape is not in the cache there is nothing to evaluate.
    """
    stats_path = pathlib.Path(stats_path)
    text = stats_path.read_text()
    scale = float(getattr(layer, "count", 1) or 1)
    stage_defs = stages_for(arch, cfg)
    stages = {s.key: StageStats(s.key, s.kind) for s in stage_defs}
    unclaimed = []
    util_by_stage = {}          # storage stage -> max utilized instances (this layer)

    # ---- storage and arithmetic levels -------------------------------------
    from ..toolchain.stats import _split_networks
    levels_text, networks_text = _split_networks(text)
    chunks = re.split(r"===\s*(.+?)\s*===", levels_text)[1:]
    for level, body in zip(chunks[0::2], chunks[1::2]):
        wblock = _dataspace_block(body, "Weights")
        if not wblock:
            continue
        energy = (_grab(rf"Energy \(total\)\s*:\s*{_NUM}\s*pJ", wblock) or 0.0) * scale
        if energy <= 0:
            continue
        claimants = [s for s in stage_defs if s.matches(level)]
        if not claimants:
            unclaimed.append({"level": level, "weight_energy_pJ": energy})
            continue
        shares = _level_shares(claimants, level)
        util = _grab(r"Utilized instances \(max\)\s*:\s*(\d+)", wblock, int) or 1
        reads = (_grab(rf"Scalar reads \(per-instance\)\s*:\s*{_NUM}", wblock)
                 or 0.0) * util * scale
        fills = ((_grab(rf"Scalar fills \(per-instance\)\s*:\s*{_NUM}", wblock) or 0.0)
                 + (_grab(rf"Scalar updates \(per-instance\)\s*:\s*{_NUM}", wblock) or 0.0)
                 ) * util * scale
        cap = _grab(r"Utilized capacity\s*:\s*(\d+)", wblock, int)
        word = _grab(r"Word bits\s*:\s*(\d+)", body, int) or cfg.weight_bits
        blk = _grab(r"Block size\s*:\s*(\d+)", body, int) or 1
        declared = _grab(r"Instances\s*:\s*(\d+)", body.split("STATS")[0], int) or util
        for stage in claimants:
            util_by_stage[stage.key] = max(util_by_stage.get(stage.key, 0), util)
            # ENERGY is the stage's share of the level (1.0 -- one claimant
            # per level since the f_if split was removed); the ACCESS COUNTS
            # are not scaled (R1 counts its reconstructions off dram's reads).
            share = shares[stage.key]
            st = stages[stage.key]
            st.levels.append(level)
            st.level_share = share
            st.energy_pJ += energy * share
            st.switch_pJ += energy * share    # storage: no wire/switching split
            st.reads += reads
            st.fills += fills
            st.instances = max(st.instances, util)
            st.declared_instances = max(st.declared_instances, declared)
            if cap is not None:
                st.utilized_capacity = (cap if st.utilized_capacity == 0
                                        else min(st.utilized_capacity, cap))
            st.block_bits = max(st.block_bits, word * blk)
            st.word_bits = max(st.word_bits, word)

    # ---- networks ----------------------------------------------------------
    for nblock in re.split(r"\nNetwork \d+\n-+\n", "\n" + networks_text)[1:]:
        name = next((ln.strip() for ln in nblock.split("\n") if ln.strip()), "?")
        level = f"NoC: {name}"
        wblock = _dataspace_block(nblock, "Weights")
        if not wblock:
            continue
        e_net = _grab(rf"\n\s+Energy \(total\)\s*:\s*{_NUM}\s*pJ", wblock) or 0.0
        e_link = _grab(rf"Link transfer energy \(total\)\s*:\s*{_NUM}\s*pJ", wblock) or 0.0
        e_red = _grab(rf"Spatial Reduction Energy \(total\)\s*:\s*{_NUM}\s*pJ", wblock) or 0.0
        energy = (e_net + e_link + e_red) * scale
        if energy <= 0:
            continue
        claimants = [s for s in stage_defs if s.matches(level)]
        if not claimants:
            unclaimed.append({"level": level, "weight_energy_pJ": energy})
            continue
        if len(claimants) != 1:
            raise ValueError(f"network level {level!r} is claimed by "
                             f"{[s.key for s in claimants]}; only the DRAM level "
                             f"may be split between two stages")
        stage = claimants[0]
        per_inst = _grab(rf"Energy \(per-instance\)\s*:\s*{_NUM}\s*pJ", wblock) or 0.0
        instances = round(e_net / per_inst) if per_inst > 0 else 1
        instances = max(1, instances)
        specs = nblock.split("STATS")[0]
        wire, switch, _model = _network_split(specs, wblock, energy, instances)
        ingress_pi = _grab(rf"Ingresses\s*:\s*{_NUM}", wblock) or 0.0
        ingress = ingress_pi * instances * scale
        # DESTINATION-SIDE ARRIVALS, from Timeloop's own multicast breakdown.
        # A multicast network injects one word and DELIVERS it to `multicast`
        # destinations, and Timeloop prints the split as
        #     Ingresses : 2359296.00
        #         @multicast 7 @scatter 2: 2359296.00
        # so the arrivals are sum(m x count) over those lines -- exact, and it
        # reconciles: on eyeriss_like's C512 shape the column network's
        # 2,359,296 ingresses x 7 equal the inner network's 16,515,072, which
        # equal the scratchpad fills. `Multicast factor` is the fallback when
        # the breakdown lines are absent.
        parts = re.findall(rf"@multicast\s*(\d+)\s*@scatter\s*(\d+)\s*:\s*{_NUM}",
                           wblock)
        mc = _grab(r"Multicast factor\s*:\s*(\d+)", wblock, int) or 1
        fan = _grab(r"Fanout\s*:\s*(\d+)", wblock, int) or 1
        if parts:
            deliv_pi = sum(int(m) * float(c) for m, _s, c in parts)
        else:
            deliv_pi = ingress_pi * mc
        deliveries = deliv_pi * instances * scale
        st = stages[stage.key]
        st.levels.append(level)
        st.energy_pJ += energy
        st.wire_pJ += wire
        st.switch_pJ += switch
        st.ingresses += ingress
        st.deliveries += deliveries
        st.multicast = max(st.multicast, float(mc))
        st.fanout = max(st.fanout, float(fan))
        st.instances = max(st.instances, instances)
        st.block_bits = max(st.block_bits,
                            _grab(r"Word bits\s*:\s*(\d+)", specs, int) or cfg.weight_bits)

    apply_dram_pj_per_bit(stages, cfg)
    cycles = _grab(r"\nCycles:\s*(\d+)", text, int)
    # RULE 3's idle denominator, PER LAYER: the engines that leak in this
    # layer's plan x this layer's cycles. A storage site's engines are the
    # UTILIZED instances of its level, because that is what Timeloop bills
    # `leak` on (each unused instance is power-gated: buffer.cpp
    # leaks_per_cycle = max utilized instances when `Instances sharing power
    # gating` is 1, FinalizeBufferEnergy = leak x cycles x leaks_per_cycle).
    # Summing per layer is what a full model needs -- conv1 uses 98 PEs, fc
    # 16, a 3x3 layer 168 -- and it is what reconciles to 1e-6 against the
    # stats' leakage delta on an ERT arm.
    for st in stages.values():
        if st.energy_pJ <= 0:
            continue
        if st.kind == "dram":
            engines = 1.0
        elif st.kind == "network":
            engines = max(1.0, float(round(float(st.fanout) * float(st.instances or 1))))
            # prompt_7: a network stage never recorded a "declared" count, so the
            # result read "14 engines, of 0 declared". Its declared count is the
            # WIDEST broadcast the design does -- the reference the utilised mean
            # is compared against, exactly as a storage level's `Instances` is.
            st.declared_instances = max(st.declared_instances, engines)
        else:
            engines = float(util_by_stage.get(st.key, st.instances) or 1)
        st.engine_cycles = engines * float(cycles or 0) * scale
    return LayerWeightPath(
        layer=getattr(layer, "name", "?"), shape=getattr(layer, "shape_name", "?"),
        weights=int(getattr(layer, "weights", 0)), scale=scale, stages=stages,
        unclaimed=unclaimed, stats_path=str(stats_path),
        cycles=(cycles * scale) if cycles is not None else 0.0)


@dataclass
class ModelWeightPath:
    """Every mapped layer's weight path, summed, with the per-layer detail kept."""
    arch: str
    model: str
    stages: dict
    per_layer: list
    unclaimed: list
    #: Directory holding the cached Timeloop output, hence the Accelergy ERT.
    #: Task 4's dilation correction needs it to reprice a level from the ERT.
    stats_dir: str = ""
    #: The stage definitions this path was read with -- `stages_for(arch, cfg)`,
    #: so `dram.reducible` reflects the decode site of the run.
    stage_defs: tuple = ()
    decode_site: str = "ondie"
    #: pJ per bit of dynamic DRAM access actually charged, and where it came
    #: from (energy.apply_dram_override / config.dram_cost_note). Replaced
    #: `dram_if_frac` when the array/interface split was removed 2026-09-09.
    dram_pj_per_bit: float = 0.0
    dram_cost_note: str = ""
    #: prompt_6 RULE 3: the summed `Cycles:` of every mapped layer (x repeat
    #: count), read off THESE stats -- the idle term's denominator for a bar
    #: billed from this plan.
    cycles: float = 0.0

    def stage(self, key):
        return self.stages.get(key)

    def dram_weight_energy(self):
        """The DRAM weight energy -- one stage, i.e. the Raw record's."""
        return sum(s.energy_pJ for s in self.stages.values() if s.kind == "dram")

    def dram_term(self):
        """The DRAM weight term, as a record. ONE stage since 2026-09-09.

        The `dram_array` / `dram_interface` pair and its `f_if` multiplier are
        gone: the whole DRAM weight energy is reducible by K/N.
        """
        d = self.stages.get("dram")
        return {
            "decode_site": self.decode_site,
            "dram_pj_per_bit": self.dram_pj_per_bit,
            "dram_cost_provenance": self.dram_cost_note,
            "dram_weight_energy_pJ": self.dram_weight_energy(),
            "dram_reducible": bool(d is not None and any(
                st.key == "dram" and st.reducible for st in self.stage_defs)),
            "rule": ("dram = DRAM weight energy x K/N under every boundary when "
                     "the decoder is on the DRAM die (01_project_context Sec. 4). "
                     "Accelergy's CactiDRAM is one flat per-bit DYNAMIC access "
                     "constant and the whole of it is credited: the DRAM "
                     "access fetches only the message bits of each codeword. "
                     "E_background and E_refresh are not modelled (both 0)"),
        }

    def total_weight_energy(self):
        return sum(s.energy_pJ for s in self.stages.values())

    def multicast_chain(self):
        """Evidence that the destination-side ARRIVAL count is parsed right.

        A network's arrivals are the words its destinations receive, so they
        should equal what the NEXT stage down the path takes in -- the next
        network's ingresses, or the scratchpad's fills. That identity is the
        cross-check on `deliveries`, which is what a network boundary's encoders
        are charged per (see `ENCODER_SITES`): on `eyeriss_v2_like` the mesh
        injects 16,356,544 weight words and its destinations receive 23,812,480,
        which is exactly the cluster-local network's ingresses AND the
        scratchpad's fills.

        Reported, not enforced. It is an identity of these designs' weight
        paths, not of Timeloop in general -- a level that drops or re-fetches
        words between two stages would break it legitimately -- so it is
        recorded on every result and read, rather than failing a run.
        """
        rows, order = [], [s.key for s in self.stage_defs]
        for i, key in enumerate(order):
            st = self.stages.get(key)
            if st is None or st.kind != "network" or st.energy_pJ <= 0:
                continue
            nxt = None
            for k2 in order[i + 1:]:
                s2 = self.stages.get(k2)
                if s2 is not None and s2.energy_pJ > 0:
                    nxt = s2
                    break
            takes_in = (0.0 if nxt is None else
                        nxt.ingresses if nxt.kind == "network" else nxt.fills)
            rows.append({
                "network": key,
                "words_injected_ingresses": st.ingresses,
                "words_received_by_destinations": st.deliveries,
                "multicast_factor_reported_by_timeloop": st.multicast,
                "effective_multiplicity": (st.deliveries / st.ingresses
                                           if st.ingresses > 0 else 1.0),
                "next_stage": None if nxt is None else nxt.key,
                "next_stage_takes_in": takes_in,
                "arrivals_match_the_next_stage": bool(
                    takes_in and abs(st.deliveries - takes_in)
                    <= 1e-6 * max(st.deliveries, takes_in)),
            })
        return {
            "rule": ("a network's destination-side arrivals should equal what "
                     "the next stage takes in (the next network's ingresses, or "
                     "the scratchpad's fills); this is the cross-check on the "
                     "arrival count a network boundary's encoders are charged "
                     "per, and it is recorded rather than enforced"),
            "per_network": rows,
        }

    def reducible_energy(self, keys):
        """Weight energy of the stages a placement would leave reduced.

        The CEILING on what any placement can save, before a single
        reconstruction is charged. Reported on every result because it is the
        number that makes a small saving legible as arithmetic rather than as a
        missing term: on `eyeriss_v2_like` the whole on-chip weight path is
        8.5 % of inference energy, so K/N = 0.81 can give back at most 1.6 %.
        """
        return sum(self.stages[k].energy_pJ for k in keys
                   if k in self.stages)

    def to_dict(self):
        return {
            "architecture": self.arch,
            "model": self.model,
            "stages": [self.stages[s.key].to_dict() for s in self.stage_defs
                       if s.key in self.stages],
            "stage_definitions": [
                {"stage": s.key, "label": s.label, "kind": s.kind,
                 "timeloop_level_prefixes": list(s.prefixes),
                 "carries_reduced_representation_possible": s.reducible,
                 "evidence": s.evidence}
                for s in self.stage_defs],
            "dram_term": self.dram_term(),
            "multicast_arrival_chain": self.multicast_chain(),
            "unclaimed_weight_levels": self.unclaimed,
            "per_layer": self.per_layer,
        }


def weight_path(cfg, arch, model, layers, stats_paths):
    """Sum every mapped layer's weight path into one record.

    `stats_paths` maps a layer's shape name to its cached stats file. A layer
    with no entry was not mapped and is skipped, exactly as `energy.gather()`
    skips it, so the two records cover the same layers.
    """
    stage_defs = stages_for(arch, cfg)
    stages = {s.key: StageStats(s.key, s.kind) for s in stage_defs}
    per_layer, unclaimed, layer_paths = [], [], []
    for layer in layers:
        path = stats_paths.get(layer.shape_name)
        if path is None:
            continue
        lp = read_weight_path(cfg, arch, layer, path)
        for key, st in lp.stages.items():
            agg = stages[key]
            agg.levels = sorted(set(agg.levels) | set(st.levels))
            agg.energy_pJ += st.energy_pJ
            agg.wire_pJ += st.wire_pJ
            agg.switch_pJ += st.switch_pJ
            agg.reads += st.reads
            agg.fills += st.fills
            agg.ingresses += st.ingresses
            agg.deliveries += st.deliveries
            agg.multicast = max(agg.multicast, st.multicast)
            agg.fanout = max(agg.fanout, st.fanout)
            agg.instances = max(agg.instances, st.instances)
            agg.declared_instances = max(agg.declared_instances, st.declared_instances)
            agg.engine_cycles += st.engine_cycles
            agg.block_bits = max(agg.block_bits, st.block_bits)
            agg.word_bits = max(agg.word_bits, st.word_bits)
            agg.level_share = st.level_share
            if st.utilized_capacity:
                agg.utilized_capacity = (st.utilized_capacity
                                         if agg.utilized_capacity == 0
                                         else min(agg.utilized_capacity,
                                                  st.utilized_capacity))
        unclaimed += [dict(u, layer=lp.layer) for u in lp.unclaimed]
        per_layer.append({
            "layer": lp.layer, "shape": lp.shape, "weights": lp.weights,
            "repeat_count": lp.scale,
            "weight_energy_pJ": lp.total_weight_energy(),
            "cycles": lp.cycles,
            "stages": {k: st.to_dict() for k, st in lp.stages.items()
                       if st.energy_pJ > 0},
        })
        layer_paths.append(lp)

    stats_dir = str(pathlib.Path(layer_paths[-1].stats_path).parent) \
        if layer_paths else ""
    return ModelWeightPath(arch=arch, model=model, stages=stages,
                           per_layer=per_layer,
                           unclaimed=unclaimed, stats_dir=stats_dir,
                           stage_defs=stage_defs, decode_site=DECODE_SITE,
                           dram_pj_per_bit=float(getattr(cfg, 'dram_pj_per_bit', None) or 0.0),
                           dram_cost_note=getattr(cfg, 'dram_cost_note', ''),
                           cycles=sum(lp.cycles for lp in layer_paths))




