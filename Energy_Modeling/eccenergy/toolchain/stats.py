"""Reading `timeloop-mapper.stats.txt`: levels, cycles, networks, classification.

`parse_stats()` and `parse_levels()` turn one solved mapping into per-level
energy and access counts; `parse_cycles()` reads the cycle count every per-cycle
term is charged over; `physical_record()` carries the declared geometry a level
was priced at; `classify()` decides which category a level's energy belongs to;
`_split_networks()` separates the network rows Timeloop prints outside the
storage hierarchy.

Timeloop prints leakage OUTSIDE the per-dataspace energies, so the raw record
never held it: only the access toll is in the bill here, and the idle term is
verified against the stats' leakage and charged by the evaluator (prompt_7).

ProjectRestructure phase 3 cut this out of `timeloop.py`. It needs no container:
it reads files Timeloop already wrote, which is why `--eval` runs on any python.
`toolchain/weight_stats.py` is the weight-path view built on top of it.
"""
from __future__ import annotations

import pathlib
import re


# ------------------------------------------------------------- stats  parsing
def parse_cycles(stats_path):
    """The run's `Cycles:` from the stats summary, or None if absent.

    prompt_6 RULE 3: the encoder's idle term is `idle_per_cycle x cycles x
    engines`, so every record that is billed needs the cycle count of ITS
    OWN plan. Recorded into `Raw.cycles` by `energy.gather`.
    """
    m = re.search(r"^Cycles:\s*(\d+)", pathlib.Path(stats_path).read_text(), re.M)
    return int(m.group(1)) if m else None


def _grab(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def _grab_as(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def parse_levels(stats_path):
    """Per-level view of one stats file, for the prompt_6 guards:

        {level: {instances, block_size, word_bits, size, leakage_pJ, gating,
                 source, utilized_instances, cycles, computes, throttling,
                 read_bandwidth, write_bandwidth, shared_bandwidth,
                 ds: {Weights|Inputs|Outputs|Compute: {reads, fills, updates,
                      energy_pJ}}}}, summary {energy_uJ, cycles, utilization}

    Per-instance counts, as Timeloop prints them. `source` is the `Vector
    access energy source` (ERT when a supplied table was billed). Written for
    `toolchain/ert_probe.py` (phase 1) and used by `study/ert_view.py`'s
    per-arm checks (RULE 2's `updates == 0`, RULE 3's leak multiplier, the
    attribution split, PEs used, DRAM word bits).
    """
    text = pathlib.Path(stats_path).read_text()
    main, _networks = _split_networks(text)
    main = main.split("Operational Intensity Stats")[0]
    chunks = re.split(r"===\s*(.+?)\s*===", main)[1:]
    levels = {}
    for level, body in zip(chunks[0::2], chunks[1::2]):
        rec = dict(
            instances=_grab_as(r"Instances\s*:\s*(\d+)", body, int),
            block_size=_grab_as(r"Block size\s*:\s*(\d+)", body, int),
            word_bits=_grab_as(r"Word bits\s*:\s*(\d+)", body, int),
            # SPECS `Size`, in ITEMS of `word_bits` bits. Anchored on the line
            # start so it cannot match `Block size` or `Effective size`. With
            # `word_bits` it is the level's STORED BITS per instance, which is
            # what a leakage DENSITY multiplies (prompt_7 Phase A / A1).
            size=_grab_as(r"\n\s+Size\s*:\s*(\d+)", body, int),
            leakage_pJ=_grab_as(r"Leakage energy \(total\)\s*:\s*([\d.eE+-]+)\s*pJ", body),
            gating=_grab_as(r"Instances sharing power gating\s*:\s*([\d.eE+-]+)", body),
            source=_grab_as(r"Vector access energy source\s*:\s*(\S+)", body, str),
            utilized_instances=_grab_as(r"Utilized instances(?: \(max\))?\s*:\s*(\d+)", body, int),
            cycles=_grab_as(r"Cycles\s*:\s*(\d+)", body, int),
            # DECLARED bandwidths, in ITEMS per cycle, from the SPECS block --
            # the ceiling `latency_post.py` throttles against. Capitalised as
            # Timeloop writes them there ("Read bandwidth"), which is what
            # keeps them apart from the STATS block's achieved "Read Bandwidth
            # (per-instance)". A level that declares none prints `-`, the regex
            # misses, and None means UNLIMITED -- which is exactly what DRAM
            # declares on every architecture in archs/ (prompt_7 section 4.3).
            read_bandwidth=_grab(r"Read bandwidth\s*:\s*(\d[\d.eE+-]*)", body),
            write_bandwidth=_grab(r"Write bandwidth\s*:\s*(\d[\d.eE+-]*)", body),
            shared_bandwidth=_grab(r"Shared bandwidth\s*:\s*(\d[\d.eE+-]*)", body),
            throttling=_grab(r"Bandwidth throttling\s*:\s*([\d.eE+-]+)", body),
            # prompt_7 C1.2: what the MAPPER was told this level moves less of.
            # Timeloop prints it on every level, 1.00 where nothing is
            # declared, so it is the MEASURED answer to "who owns the off-chip
            # weight relief for this bar" -- the mapper if it is K/N here, the
            # evaluator's roofline if it is 1.00. Two live owners on one bar
            # would apply the relief twice (prompt_6 RULE 1).
            bw_consumption_scale=_grab(
                r"Bandwidth Consumption Scale\s*:\s*([\d.eE+-]+)", body),
            computes=_grab_as(r"Computes \(total\)\s*:\s*(\d+)", body, int),
            ds={})
        parts = re.split(r"\n\s+(Weights|Inputs|Outputs)\s*:\s*\n", body)
        if len(parts) > 1:
            for ds, sec in zip(parts[1::2], parts[2::2]):
                rec["ds"][ds] = dict(
                    reads=_grab_as(r"Scalar reads \(per-instance\)\s*:\s*([\d.eE+-]+)", sec),
                    fills=_grab_as(r"Scalar fills \(per-instance\)\s*:\s*([\d.eE+-]+)", sec),
                    updates=_grab_as(r"Scalar updates \(per-instance\)\s*:\s*([\d.eE+-]+)", sec),
                    utilized_instances=_grab_as(
                        r"Utilized instances \(max\)\s*:\s*(\d+)", sec, int),
                    energy_pJ=_grab_as(r"Energy \(total\)\s*:\s*([\d.eE+-]+)\s*pJ", sec))
        else:
            e = _grab_as(r"Energy \(total\)\s*:\s*([\d.eE+-]+)\s*pJ", body)
            if e is not None:
                rec["ds"]["Compute"] = dict(reads=None, fills=None, updates=None,
                                            utilized_instances=None, energy_pJ=e)
        levels[level] = rec
    summary = dict(energy_uJ=_grab_as(r"\nEnergy:\s*([\d.eE+-]+)\s*uJ", text),
                   cycles=_grab_as(r"\nCycles:\s*(\d+)", text, int),
                   utilization=_grab_as(r"\nUtilization:\s*([\d.eE+-]+)%", text))
    return levels, summary


def physical_record(levels):
    """The compact, ECC-INDEPENDENT physical record for one mapped layer.

    One row per STORAGE level, from `parse_levels()[0]`: the declared geometry,
    the declared bandwidths, the per-instance access counts, and what Timeloop
    itself billed for leakage and cycles. Pure Timeloop output, so it belongs
    in the raw cache beside the energies (`energy.Raw.per_layer`) and every ECC
    configuration is then arithmetic on top of it.

    Two consumers, one record, so they cannot disagree about the same level:
    `latency_post.roofline()` re-times the plan from the counts and the
    bandwidths, and `energy.standby_energy()` charges standby power from the
    stored bits and the utilized instances. The ARITHMETIC level is not a row
    -- it has no bandwidth and cannot throttle -- but it supplies
    `compute_cycles`, which is how long the multiplies alone would take with
    infinitely fast memory and the floor every storage level is measured
    against.
    """
    compute, rows = None, []
    for name, rec in levels.items():
        if rec.get("computes") is not None:
            compute = rec.get("cycles")
            rows.append({
                "level": name, "arithmetic": True, "offchip": False,
                "instances": rec.get("instances"),
                "utilized": rec.get("utilized_instances"),
                "size": None, "word_bits": rec.get("word_bits"),
                "cycles": rec.get("cycles"),
                # Timeloop bills NO leakage on the arithmetic level: it prints
                # no `Leakage energy (total)` there, although the ERT carries a
                # `leak` row for the MAC. That absence is a term, not a zero.
                "timeloop_leakage_pJ": rec.get("leakage_pJ"),
                "read_bw": None, "write_bw": None, "shared_bw": None,
                "throttling": None, "reads": {}, "writes": {}})
            continue
        reads, writes = {}, {}
        for ds, d in (rec.get("ds") or {}).items():
            reads[ds] = float(d.get("reads") or 0.0)
            writes[ds] = float(d.get("fills") or 0.0) + float(d.get("updates") or 0.0)
        rows.append({
            "level": name, "arithmetic": False,
            "offchip": "dram" in str(name).lower(),
            "instances": rec.get("instances"),
            "utilized": rec.get("utilized_instances"),
            "size": rec.get("size"), "word_bits": rec.get("word_bits"),
            "cycles": rec.get("cycles"),
            "timeloop_leakage_pJ": rec.get("leakage_pJ"),
            "read_bw": rec.get("read_bandwidth"),
            "write_bw": rec.get("write_bandwidth"),
            "shared_bw": rec.get("shared_bandwidth"),
            "throttling": rec.get("throttling"),
            # prompt_7 C1.2: whether THIS level's plan was already solved
            # against the reduced weight demand. `latency_post` reads it to
            # decide who owns the relief; 1.00 (or None on a cache entry that
            # predates the field) means nobody did.
            "bw_consumption_scale": rec.get("bw_consumption_scale"),
            "reads": reads, "writes": writes})
    return {"compute_cycles": compute, "levels": rows}


def parse_stats(stats_path, layer_label, scale=1.0):
    """Flatten a timeloop-mapper.stats.txt into per-(level, dataspace) rows.

    `scale` multiplies energies and access counts, which is how a transformer
    block matmul is charged `count` times from a single mapping.

    `instances` is carried through because it, not the level's name, is what
    tells a shared global buffer apart from a per-PE scratchpad.
    """
    text = pathlib.Path(stats_path).read_text()
    rows = []
    # Timeloop prints its `Networks` section between the last `=== level ===`
    # block and the operational-intensity summary, and it is NOT a `===` block.
    # Left in place it lands inside the DRAM chunk, where its own
    # Weights:/Inputs:/Outputs: sub-blocks read as DRAM energy. That was
    # harmless while every network cost 0.00 pJ; it double-counts now.
    levels_text, networks_text = _split_networks(text)
    chunks = re.split(r"===\s*(.+?)\s*===", levels_text)[1:]
    for level, body in zip(chunks[0::2], chunks[1::2]):
        inst = _grab(r"Instances\s*:\s*(\d+)", body)
        # prompt_6 RULE 1: the level's declared word, as the mapper saw it.
        # `Word bits == q` on a Weights level says the MAPPER narrowed it.
        word_bits = _grab(r"Word bits\s*:\s*(\d+)", body)
        block_size = _grab(r"Block size\s*:\s*(\d+)", body)
        parts = re.split(r"\n\s+(Weights|Inputs|Outputs)\s*:\s*\n", body)
        if len(parts) > 1:
            for ds, sec in zip(parts[1::2], parts[2::2]):
                reads = _grab(r"Scalar reads \(per-instance\)\s*:\s*([\d.eE+]+)", sec)
                fills = _grab(r"Scalar fills \(per-instance\)\s*:\s*([\d.eE+]+)", sec)
                updates = _grab(r"Scalar updates \(per-instance\)\s*:\s*([\d.eE+]+)", sec)
                energy = _grab(r"Energy \(total\)\s*:\s*([\d.eE+]+)\s*pJ", sec)
                rows.append(dict(
                    layer=layer_label, level=level, dataspace=ds,
                    instances=inst,
                    word_bits=None if word_bits is None else int(word_bits),
                    block_size=None if block_size is None else int(block_size),
                    reads=None if reads is None else reads * scale,
                    writes=((fills or 0.0) + (updates or 0.0)) * scale,
                    energy_pJ=None if energy is None else energy * scale))
        else:
            energy = _grab(r"Energy \(total\)\s*:\s*([\d.eE+]+)\s*pJ", body)
            if energy is not None:
                rows.append(dict(layer=layer_label, level=level, dataspace="Compute",
                                 instances=inst, reads=None, writes=None,
                                 energy_pJ=energy * scale))
    rows += _parse_networks(networks_text, layer_label, scale)
    return rows


_NETWORKS_HDR = re.compile(r"\nNetworks\n-{3,}\n")
_NETWORKS_END = re.compile(r"\nOperational Intensity Stats")


def _split_networks(text):
    """(text with the Networks section removed, the Networks section)."""
    m = _NETWORKS_HDR.search(text)
    if not m:
        return text, ""
    rest = text[m.end():]
    e = _NETWORKS_END.search(rest)
    if not e:
        return text[:m.start()], rest
    return text[:m.start()] + rest[e.start():], rest[:e.start()]


def _parse_networks(ntext, layer_label, scale=1.0):
    """One row per (network, dataspace), energy = what Timeloop's total counts.

    `LegacyNetwork::Energy()` is NetworkEnergy + SpatialReductionEnergy, and
    NetworkEnergy is (link_transfer_energy + energy) x instances
    (network-legacy.cpp:690-697). The three "(total)" lines below are exactly
    those terms already multiplied out, so summing them reproduces the network's
    share of the `Energy:` summary. Timeloop only prints a network whose energy
    is non-zero, so an empty section means the NoC was free, not absent.

    Levels are prefixed `NoC:` so `classify()` can bucket them before any
    substring test -- "DRAM <==> ifmap_glb" is a network, not DRAM.
    `reads` carries the ingress count so the access columns stay meaningful.
    """
    rows = []
    if not ntext.strip():
        return rows
    blocks = re.split(r"\nNetwork \d+\n-+\n", "\n" + ntext)[1:]
    for block in blocks:
        name = next((ln.strip() for ln in block.split("\n") if ln.strip()), "?")
        parts = re.split(r"\n\s+(Weights|Inputs|Outputs)\s*:\s*\n", block)
        # Instances of this network = Energy (total) / Energy (per-instance),
        # printed per dataspace; take it from any dataspace that moved energy.
        # Not printed as such: Timeloop reports instances only for storage.
        net_instances = None
        for sec in parts[2::2]:
            e_t = _grab(r"\n\s+Energy \(total\)\s*:\s*([\d.eE+-]+)\s*pJ", sec)
            e_i = _grab(r"\n\s+Energy \(per-instance\)\s*:\s*([\d.eE+-]+)\s*pJ", sec)
            if e_t and e_i:
                net_instances = round(e_t / e_i)
                break
        for ds, sec in zip(parts[1::2], parts[2::2]):
            ingresses = _grab(r"\n\s+Ingresses\s*:\s*([\d.eE+-]+)", sec)
            e_net = _grab(r"\n\s+Energy \(total\)\s*:\s*([\d.eE+-]+)\s*pJ", sec) or 0.0
            e_link = _grab(r"\n\s+Link transfer energy \(total\)\s*:\s*([\d.eE+-]+)\s*pJ", sec) or 0.0
            e_red = _grab(r"\n\s+Spatial Reduction Energy \(total\)\s*:\s*([\d.eE+-]+)\s*pJ", sec) or 0.0
            # Counts the evaluator-only terms (noc_post.py) are charged from.
            # Pure Timeloop output, per instance except where scaled like
            # `reads`: `spatial_reductions` is the number of partial sums added
            # into a neighbour, `hops` the mean hop count of an ingress, and
            # `per_hop_pJ` what one hop of one `network_word_bits` word costs.
            reductions = _grab(r"\n\s+Spatial reductions\s*:\s*([\d.eE+-]+)", sec)
            hops = _grab(r"\n\s+Average number of hops\s*:\s*([\d.eE+-]+)", sec)
            per_hop = _grab(r"\n\s+Energy \(per-hop\)\s*:\s*([\d.eE+-]+)\s*fJ", sec)
            rows.append(dict(
                layer=layer_label, level=f"NoC: {name}", dataspace=ds,
                instances=None,
                reads=None if ingresses is None else ingresses * scale,
                writes=0.0,
                energy_pJ=(e_net + e_link + e_red) * scale,
                spatial_reductions=None if reductions is None else reductions * scale,
                hops=hops,
                per_hop_pJ=None if per_hop is None else per_hop / 1000.0,
                net_instances=net_instances))
    return rows


def classify(level, instances=None, mode="instances"):
    """Map a Timeloop storage level onto a plotted energy category.

    Two modes, because the historical scripts got this wrong on one design:

    `instances` (default, physically correct)
        A level with exactly one instance is a shared global buffer; a level
        replicated across the PE array is local storage, whatever it is called.
        Needed for simba_like, whose per-PE `PEWeightBuffer` / `PEAccuBuffer` /
        `PEInputBuffer` are replicated 16-64x yet match the name test below.

    `name` (legacy)
        Substring match on the level name. Reproduces the numbers in the old
        figures, including simba_like's ~33 mJ of "Global buffer" that is
        really distributed per-PE storage. Totals are identical either way; only
        the split between the two on-chip categories moves.
    """
    low = level.lower()
    if level.startswith("NoC"):
        # Before every substring test: a network is named after the two levels
        # it joins, so "NoC: DRAM <==> ifmap_glb" contains "dram".
        return "NoC"
    if "dram" in low:
        return "DRAM"
    if "mac" in low or "compute" in low:
        return "Compute"
    if mode == "instances" and instances is not None:
        return "Global buffer" if instances <= 1 else "Local (spads/RF)"
    if "glb" in low or "buffer" in low or "sram" in low:
        return "Global buffer"
    return "Local (spads/RF)"
