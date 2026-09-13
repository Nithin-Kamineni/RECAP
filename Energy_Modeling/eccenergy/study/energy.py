"""Aggregate Timeloop stats into the plotted energy categories.

`gather()` turns one (architecture, model) pair into a small `Raw` record. That
record is ECC-independent -- it is pure Timeloop output -- so it is cached to
`results/_raw/` and every ECC configuration is then a few milliseconds of
arithmetic on top. Changing BCH_N/K and re-running costs nothing.
"""
from __future__ import annotations

import json

import pandas as pd

from ..physics.baseline_dram import dram_ert_pj_per_bit
from ..toolchain import latency_post, noc_post
from ..toolchain.stats import classify, parse_cycles, parse_levels, parse_stats, physical_record

#: Categories that come from Timeloop.
#: "NoC" is the interconnect between levels -- wire, router and ingress energy
#: on every network Timeloop instantiates (archs/_shared/noc.yaml). It was
#: folded into the Local label historically but was always 0.00 there, because
#: Timeloop's wire model is a stub; it is a category of its own now that it
#: is costed, and the Local label no longer claims to contain it.
PHYS_CATS = ("DRAM", "Global buffer", "Local (spads/RF)", "NoC", "Compute")
#: prompt_7 Phase A / Defect 2: the accelerator's own STANDBY energy, charged
#: to ALL THREE ARMS or to none. It is appended by `phys_cats()` only when
#: ECC_STATIC_ENERGY=1, so with the knob off the category layout is
#: byte-identical to every layout that predates it and a cached raw record
#: still loads -- that is what makes "ECC_STATIC_ENERGY=0 reproduces today's
#: totals to the pJ" true by construction rather than by arithmetic.
#: It is NOT in `onchip_cats()`: standby energy is power x time and carries no
#: weight traffic, so the reconstruction arm does not scale it by K/N. What it
#: DOES respond to is the run length, which is why ECC_LATENCY_MODEL feeds it.
STANDBY_CAT = "Standby"
#: Which ECC_LEAKAGE_NW density prices a level, and what it is a density OF.
#: `sram`/`rf` are per STORED BIT, `mac` per instance; `dram` is off-chip and
#: no density is declared for it (env.sh section 6), so it is charged nothing
#: -- the same deliberate omission as ECC_DRAM_BACKGROUND_PJ.
STANDBY_DENSITY = {"sram": "sram_bit", "rf": "rf_bit", "mac": "mac_instance"}
#: Split variants, used when ECC_SPLIT_READ_WRITE=1. NoC is not split: a
#: network has ingresses, not reads and writes.
PHYS_CATS_SPLIT = ("DRAM", "Global buffer (read)", "Global buffer (write)",
                   "Local (read)", "Local (write)", "NoC", "Compute")
#: Categories the ECC model adds. `Recon overhead` is the buffer/control cost a
#: reconstruction PLACEMENT would carry beyond the encoder itself. It is a
#: category of its own rather than being folded into `Reconstruction` because
#: Task 3 asks for the overheads to be reported, and an overhead hidden inside
#: the thing it is an overhead ON cannot be read off a figure. Since R4b and
#: its reuse register were removed (2026-09-10) no placement carries one, so it
#: is structurally zero and `active_categories()` drops it from the figures.
ECC_CATS = ("ECC decode", "Reconstruction", "Recon overhead")

SPLITTABLE = ("Global buffer", "Local (spads/RF)")


def phys_cats(cfg):
    cats = list(PHYS_CATS_SPLIT if cfg.split_read_write else PHYS_CATS)
    if getattr(cfg, "static_energy", False):
        cats.append(STANDBY_CAT)
    return cats


def onchip_cats(cfg):
    """The categories whose WEIGHT portion the recon arm scales by K/N."""
    # NoC is included: reconstructing weights on-chip from K/N of their bits
    # moves K/N of the weight bits across the interconnect too. Recorded as an
    # approximation on every result so the assumption is visible.
    if cfg.split_read_write:
        return ["Global buffer (read)", "Global buffer (write)",
                "Local (read)", "Local (write)", "NoC"]
    return ["Global buffer", "Local (spads/RF)", "NoC"]


def plot_cats(cfg):
    return phys_cats(cfg) + list(ECC_CATS)


def category_energy(df, cfg):
    """Sum energy per plotted category for the rows given.

    With ECC_SPLIT_READ_WRITE=1 the two on-chip categories are split into read
    and write in proportion to their access counts. That is exact when a level's
    read and write per-access energies are equal, and a count-proportional
    approximation otherwise; totals are unaffected either way.
    """
    cats = plot_cats(cfg)
    out = {c: 0.0 for c in cats}
    if df.empty:
        return pd.Series(out).reindex(cats, fill_value=0.0)

    df = df.copy()
    df["reads"] = df["reads"].fillna(0.0)
    df["writes"] = df["writes"].fillna(0.0)
    for row in df.itertuples(index=False):
        cat, energy = row.category, row.energy_pJ
        if cfg.split_read_write and cat in SPLITTABLE:
            total = row.reads + row.writes
            frac_r = (row.reads / total) if total > 0 else 1.0
            e_r = energy * frac_r
            prefix = "Global buffer" if cat == "Global buffer" else "Local"
            out[f"{prefix} (read)"] += e_r
            out[f"{prefix} (write)"] += energy - e_r
        else:
            out[cat] = out.get(cat, 0.0) + energy
    return pd.Series(out).reindex(cats, fill_value=0.0)


class Raw:
    """Timeloop-only energies for one (architecture, model).

    base         per-category energy, all dataspaces
    base_w       per-category energy, Weights only
    base_i       per-category energy, Inputs only  (needed by the weak-ECC overlay)
    e_dram_w     DRAM energy attributable to Weights
    dram_w_reads scalar weight reads out of DRAM -- the codeword count driver
    """

    __slots__ = ("base", "base_w", "base_i", "e_dram_w", "dram_w_reads",
                 "layers_ok", "layers_skipped", "weights", "per_layer", "levels",
                 "noc_post", "mac", "dram", "cycles", "standby", "latency")

    def __init__(self, base, base_w, base_i, e_dram_w, dram_w_reads,
                 layers_ok, layers_skipped, weights, per_layer=None, levels=None,
                 noc_post=None, mac=None, cycles=None):
        #: prompt_6 RULE 3: total cycles of the mapped layers (x repeat
        #: count), the denominator of the encoder's idle term. None on a
        #: record older than the field; `build_stacks` refuses to charge idle
        #: on one and `load_raw` re-gathers it from the mapper cache.
        self.cycles = cycles
        self.base = base
        self.base_w = base_w
        self.base_i = base_i
        self.e_dram_w = e_dram_w
        self.dram_w_reads = dram_w_reads
        self.layers_ok = layers_ok
        self.layers_skipped = layers_skipped
        self.weights = weights
        #: per-layer detail, so a development result can be read layer by layer
        #: rather than only as a total. Empty on records written before Task 1.
        self.per_layer = per_layer or []
        #: per-(level, dataspace) access counts and energies -- the storage
        #: spec asks results to preserve enough to audit the accounting, and a
        #: category total cannot be checked against Timeloop without this.
        self.levels = levels or []
        #: the evaluator-only NoC coefficients (noc_post.stamp) this record was
        #: aggregated with. They are not in the mapper fingerprint -- the mapper
        #: never sees them -- so this is what tells a stale record from a
        #: current one. None on records written before 2026-09-08.
        self.noc_post = noc_post
        #: What a MAC was charged in THIS record: `apply_mac_override()` fills
        #: it (the ERT's per-MAC energy, the MAC count, and the override if
        #: one is in force). Evaluator-side only -- never written to the raw
        #: cache, which stays pure Timeloop output.
        self.mac = mac
        #: What a DRAM bit was charged in THIS record: `apply_dram_override()`
        #: fills it (Accelergy's own pJ/bit, the override if one is in force,
        #: and the E_background / E_refresh terms, which are 0 and unmodelled).
        #: Evaluator-side only, like `mac`.
        self.dram = None
        #: What COMPONENT STANDBY energy was charged in THIS record:
        #: `apply_standby_energy()` fills it (the densities, the per-level
        #: breakdown, what Timeloop billed for leakage beside it, and the run
        #: length it was charged over). Evaluator-side only -- the raw cache
        #: stays pure Timeloop output and ECC_LEAKAGE_NW is a price list, not
        #: an architecture.
        self.standby = None
        #: This plan's own re-timing under `latency_post.roofline()`:
        #: `apply_latency_model()` fills it, None when ECC_LATENCY_MODEL=0.
        #: The reference arm's (weight_scale 1.0); a reconstruction arm's is
        #: `latency_post.model_cycles(raw, cfg, weight_scale=K/N)`.
        self.latency = None

    @property
    def total(self):
        return float(self.base.sum())

    def to_json(self):
        return {
            "base": {k: float(v) for k, v in self.base.items()},
            "base_w": {k: float(v) for k, v in self.base_w.items()},
            "base_i": {k: float(v) for k, v in self.base_i.items()},
            "e_dram_w": float(self.e_dram_w),
            "dram_w_reads": float(self.dram_w_reads),
            "layers_ok": int(self.layers_ok),
            "layers_skipped": int(self.layers_skipped),
            "weights": int(self.weights),
            "per_layer": self.per_layer,
            "levels": self.levels,
            "noc_post": self.noc_post,
            "cycles": self.cycles,
        }

    @classmethod
    def from_json(cls, d, cfg):
        cats = plot_cats(cfg)
        stored = set(d.get("base", {}))
        # STANDBY_CAT is filled EVALUATOR-side (`apply_standby_energy`) and
        # never carries gather-time energy, so its absence says nothing about
        # the category layout the record was written under. Leaving it in this
        # check would report every pre-Phase-A record as a layout mismatch
        # instead of as what it is -- a record with no physical detail, which
        # `load_raw` says plainly a few lines further on.
        missing = [c for c in phys_cats(cfg)
                   if c not in stored and c != STANDBY_CAT]
        if missing:
            # The record was written under a different category layout (a
            # different ECC_CLASSIFY or ECC_SPLIT_READ_WRITE). Reindexing would
            # quietly drop those categories and understate every total, so
            # refuse it instead.
            raise ValueError(
                f"raw record holds categories {sorted(stored)}, but this "
                f"configuration expects {phys_cats(cfg)}; missing {missing}")

        def series(key):
            return pd.Series(d.get(key, {})).reindex(cats, fill_value=0.0)

        return cls(series("base"), series("base_w"), series("base_i"),
                   d["e_dram_w"], d["dram_w_reads"],
                   d.get("layers_ok", 0), d.get("layers_skipped", 0),
                   d.get("weights", 0), d.get("per_layer", []), d.get("levels", []),
                   d.get("noc_post"), cycles=d.get("cycles"))


def mac_count(raw):
    """Total MACs in the record: per-layer MACs x repeat count, mapped layers only.

    Equals Timeloop's `Computes (total)` summed over the layers, so
    `base["Compute"] / mac_count` is exactly the ERT's per-MAC energy (1.16877 pJ
    on eyeriss_v2_like, to 1e-11 relative).
    """
    return float(sum(float(l.get("macs", 0) or 0) * float(l.get("repeat_count", 1) or 1)
                     for l in raw.per_layer if l.get("status") == "ok"))


def apply_dram_override(raw, cfg, verbose=True):
    """Rescale the DRAM category to ECC_DRAM_PJ_PER_BIT, evaluator-side.

    THE NUMERATOR KNOB, and the counterpart of `apply_mac_override`. Accelergy's
    CactiDRAM charges 8 pJ/bit for LPDDR4 as modelled, which is below every
    measured figure in the literature (Horowitz ISSCC 2014: 20 pJ/bit; FReaC
    Cache MICRO 2020 and Gebhart MICRO 2012: 28-45 pJ/bit). The whole DRAM
    category is scaled by `override / ERT` -- weights, inputs and outputs alike,
    because the per-bit cost is a property of the device and not of a dataspace.
    Scaling weights alone would inflate the weight share and flatter every ECC
    percentage; `tests/test_dram_override.py` asserts it does not happen.

    Applied AFTER the raw cache is read or written, exactly like the MAC
    override, so `results/_raw/` stays pure Timeloop output.

    E_background and E_refresh (`ECC_DRAM_BACKGROUND_PJ`, `ECC_DRAM_REFRESH_PJ`)
    are 0 by default and this function does not add them; the `dram` record says
    so. They are a TODO, and they are not neutral -- an arm that stores fewer
    weight bits would save both.
    """
    ert = dram_ert_pj_per_bit(raw, cfg)
    tgt = cfg.dram_pj_per_bit
    info = {"ert_pj_per_bit": ert,
            "override_pj_per_bit": tgt,
            "pj_per_bit_charged": tgt if tgt is not None else ert,
            "source": "ECC_DRAM_PJ_PER_BIT" if tgt is not None else "Accelergy ERT",
            "citation": cfg.dram_cost_note,
            "background_pJ": cfg.dram_background_pj,
            "refresh_pJ": cfg.dram_refresh_pj,
            "static_note": cfg.dram_static_note,
            "e_dram_w_pJ_ert": float(raw.e_dram_w),
            "e_dram_w_pJ_charged": float(raw.e_dram_w)}
    if tgt is None or not ert:
        raw.dram = info
        return raw
    ratio = tgt / ert
    info["dram_scale"] = ratio
    info["e_dram_w_pJ_charged"] = float(raw.e_dram_w) * ratio

    def scaled(series):
        out = series.copy()
        if "DRAM" in out.index:
            out["DRAM"] = float(out["DRAM"]) * ratio
        return out

    levels = []
    for lv in raw.levels:
        lv = dict(lv)
        if lv.get("category") == "DRAM":
            lv["energy_pJ"] = float(lv.get("energy_pJ", 0.0)) * ratio
            lv["dram_pj_per_bit"] = tgt
        levels.append(lv)

    # Per-layer totals. The weight share is exact (the record carries it); the
    # rest of the DRAM delta is apportioned across layers by their share of
    # total energy, which is the only per-layer weighting the record supports
    # and keeps sum(per_layer) equal to the base totals.
    d_w = float(raw.e_dram_w) * (ratio - 1.0)
    d_all = float(raw.base.get("DRAM", 0.0)) * (ratio - 1.0)
    d_rest = d_all - d_w
    tot = sum(float(l.get("total_energy_pJ", 0.0)) for l in raw.per_layer
              if l.get("status") == "ok") or 1.0
    per_layer = []
    for l in raw.per_layer:
        l = dict(l)
        if l.get("status") == "ok":
            dw = float(l.get("dram_weight_energy_pJ", 0.0) or 0.0)
            share = float(l.get("total_energy_pJ", 0.0)) / tot
            l["total_energy_pJ"] = (float(l.get("total_energy_pJ", 0.0))
                                    + dw * (ratio - 1.0) + d_rest * share)
            l["dram_weight_energy_pJ_charged"] = dw * ratio
        per_layer.append(l)

    out = Raw(scaled(raw.base), scaled(raw.base_w), scaled(raw.base_i),
              float(raw.e_dram_w) * ratio, raw.dram_w_reads,
              raw.layers_ok, raw.layers_skipped, raw.weights,
              per_layer=per_layer, levels=levels,
              noc_post=raw.noc_post, mac=raw.mac, cycles=raw.cycles)
    out.standby, out.latency = raw.standby, raw.latency
    out.dram = info
    if verbose:
        print(f"  [DRAM OVERRIDE] DRAM rescaled {ert:.4g} -> {tgt:g} pJ/bit "
              f"(x{ratio:.4g}): weight DRAM {raw.e_dram_w / 1e6:,.3f} -> "
              f"{raw.e_dram_w * ratio / 1e6:,.3f} uJ. {cfg.dram_cost_note}")
        print(f"  [DRAM OVERRIDE] {cfg.dram_static_note}")
    return out


def apply_mac_override(raw, cfg, verbose=True):
    """Rescale the Compute category to MACs x ECC_MAC_PJ_OVERRIDE, evaluator-side.

    THE DENOMINATOR KNOB. An ECC saving is saved_pJ / total_pJ; the saved pJ are
    weight traffic and do not depend on what a MAC costs, the total does. This
    touches ONLY the Compute category -- `base`, `base_w`, `base_i`, the
    Compute rows of `levels` and each layer's total -- by the ratio
    override / ERT, and leaves the DRAM, buffer, scratchpad and NoC energies bit-
    identical, which is what makes the saved pJ identical across the rows of
    FINDINGS 7.3 (`tests/test_mac_override.py` asserts it).

    Applied AFTER the raw cache is read or written (`collect()`), so
    `results/_raw/` stays pure Timeloop output and the override cannot leak into
    a later run that did not ask for it. The record's `mac` field says what was
    done, and every result file, manifest and figure title repeats it.

    With the override unset the record is returned unchanged apart from `mac`
    being filled with the ERT's per-MAC energy, so a title can state the number
    the primary result rests on.
    """
    macs = mac_count(raw)
    compute = float(raw.base.get("Compute", 0.0))
    ert_pj = compute / macs if macs else None
    short, long_ = cfg.mac_citation()
    info = {"macs": macs, "compute_pJ_ert": compute, "ert_pj_per_mac": ert_pj,
            "override_pj_per_mac": cfg.mac_pj_override,
            "pj_per_mac_charged": cfg.mac_pj_override if cfg.mac_pj_override is not None else ert_pj,
            "source": "ECC_MAC_PJ_OVERRIDE" if cfg.mac_pj_override is not None else "Accelergy ERT",
            "citation_short": short, "citation": long_,
            "compute_pJ_charged": compute}
    if cfg.mac_pj_override is None or not macs:
        raw.mac = info
        return raw
    ratio = (macs * cfg.mac_pj_override) / compute if compute else 1.0
    info["compute_pJ_charged"] = compute * ratio
    info["compute_scale"] = ratio
    info["opt_metric"] = cfg.opt_metric
    info["mapping_note"] = (
        "the MAC count is mapping-invariant, so under the energy objective the "
        "mapping optimum does not move and the cache stays warm" if cfg.opt_metric == "energy"
        else f"ECC_OPT_METRIC={cfg.opt_metric}: a cheaper MAC can move the mapping "
             f"optimum, and the mapper was NOT re-run (it prices MACs from the ERT) "
             f"-- a fixed-mapping result")

    def scaled(series):
        out = series.copy()
        if "Compute" in out.index:
            out["Compute"] = float(out["Compute"]) * ratio
        return out

    levels = []
    for lv in raw.levels:
        lv = dict(lv)
        if lv.get("category") == "Compute":
            lv["energy_pJ"] = float(lv.get("energy_pJ", 0.0)) * ratio
            lv["mac_pj_override"] = cfg.mac_pj_override
        levels.append(lv)
    per_layer = []
    for l in raw.per_layer:
        l = dict(l)
        if l.get("status") == "ok" and ert_pj is not None:
            c = float(l.get("macs", 0) or 0) * float(l.get("repeat_count", 1) or 1) * ert_pj
            l["total_energy_pJ"] = float(l.get("total_energy_pJ", 0.0)) - c * (1.0 - ratio)
            l["compute_energy_pJ_charged"] = c * ratio
        per_layer.append(l)
    out = Raw(scaled(raw.base), scaled(raw.base_w), scaled(raw.base_i),
              raw.e_dram_w, raw.dram_w_reads, raw.layers_ok, raw.layers_skipped,
              raw.weights, per_layer=per_layer, levels=levels,
              noc_post=raw.noc_post, mac=info, cycles=raw.cycles)
    out.standby, out.latency = raw.standby, raw.latency
    out.dram = raw.dram
    if verbose:
        print(f"  [MAC OVERRIDE] Compute rescaled to {macs:,.0f} MACs x "
              f"{cfg.mac_pj_override:g} pJ = {compute * ratio / 1e6:,.3f} uJ "
              f"(ERT: {ert_pj:.5f} pJ/MAC, {compute / 1e6:,.3f} uJ); {short}. "
              f"Every percentage's DENOMINATOR moves; no saved pJ does.")
        if cfg.opt_metric != "energy":
            print(f"  [warn] ECC_OPT_METRIC={cfg.opt_metric}: a cheaper MAC can move an "
                  f"EDP-optimal mapping, and the mapper was not re-run (it prices "
                  f"MACs from the ERT) -- a fixed-mapping result, not a re-optimised "
                  f"design")
    return out


# ------------------------------------------------- prompt_7 Phase A: standby
def standby_kind(level, instances=None, arithmetic=False):
    """Which ECC_LEAKAGE_NW density prices this level.

    The same PHYSICAL test `classify()` uses, for the same reason: a level with
    one instance is a shared SRAM macro and a level replicated across the PE
    array is a register file, whatever the YAML calls it. Every scratchpad in
    this study declares `Technology: SRAM` and none of them is one.
    """
    low = str(level).lower()
    if "dram" in low:
        return "dram"
    if arithmetic or "mac" in low or "compute" in low:
        return "mac"
    return "sram" if (instances or 1) <= 1 else "rf"


def standby_components(raw, cfg):
    """Per-level standby POWER and the run length it is charged over.

    Aggregated across layers from `per_layer[*]["physical"]`, at the repeat
    count, so it describes exactly the run `Raw.cycles` describes.

    `instance_cycles` is the sum over layers of (UTILIZED instances x that
    layer's cycles), which is what Timeloop itself bills leakage on -- it power-
    gates every unused instance and charges `leak x utilized x cycles`
    (`buffer.cpp FinalizeBufferEnergy`, verified on this design's own cached
    ERT: filter_glb 7.39799e-05 x 32000 x 1 = 2.37 pJ, weights_spad 2.67882e-06
    x 32000 x 16 = 1.37 pJ, both to the printed precision). The DECLARED count
    is carried beside it, never used, and reported -- the same convention
    prompt_6 RULE 3 fixed for the reconstruction engines.
    """
    per_level, seen = {}, {}
    for lp in raw.per_layer:
        if lp.get("status") != "ok" or not lp.get("physical"):
            continue
        n = float(lp.get("repeat_count", 1) or 1)
        for row in lp["physical"]["levels"]:
            name = row["level"]
            kind = standby_kind(name, row.get("instances"), row.get("arithmetic"))
            size, wb = row.get("size"), row.get("word_bits")
            bits = (float(size) * float(wb)) if (size and wb) else None
            c = per_level.setdefault(name, {
                "level": name, "kind": kind,
                "category": classify(name, row.get("instances"), cfg.classify_mode),
                "bits_per_instance": bits,
                "instances_declared": row.get("instances"),
                "instance_cycles": 0.0, "declared_instance_cycles": 0.0,
                "cycles": 0.0, "timeloop_leakage_pJ": 0.0})
            # One architecture per fingerprint, so a level's geometry cannot
            # legitimately move between the layers of one record. If it does,
            # two chips are in one record and every total below is meaningless.
            key = (kind, bits, row.get("instances"))
            if seen.setdefault(name, key) != key:
                raise ValueError(
                    f"standby: level {name!r} changes geometry between layers of "
                    f"one record ({seen[name]} then {key}) -- two architectures "
                    f"are in one raw record and no per-level total is meaningful")
            cyc = float(row.get("cycles") or 0.0)
            c["instance_cycles"] += float(row.get("utilized") or 0.0) * cyc * n
            c["declared_instance_cycles"] += float(row.get("instances") or 0.0) * cyc * n
            c["cycles"] += cyc * n
            c["timeloop_leakage_pJ"] += float(row.get("timeloop_leakage_pJ") or 0.0) * n
    return [per_level[k] for k in sorted(per_level)]


def latency_post_cycle_seconds(cfg):
    """THIS DESIGN's clock period (prompt_7 C1.5), imported lazily.

    `latency_post` owns the resolution because it is the module that has to
    agree with the architecture's declared off-chip limit; standby energy has
    to be charged over the same period or POWER x TIME and ITEMS / TIME would
    be two different seconds.
    """
    from ..toolchain.latency_post import cycle_seconds
    return cycle_seconds(cfg)


def standby_energy(raw, cfg, *, cycle_scale=1.0):
    """Component standby energy for THIS plan, per level and in total (pJ).

    `E = density_nW x units x instance_cycles x cycle_period`, with `units` the
    stored bits of one instance for an SRAM or a register file and 1 for a MAC.
    `cycle_scale` re-times the run: 1.0 charges Timeloop's own cycle count,
    and ECC_LATENCY_MODEL=1 charges the roofline's instead, because standby
    energy is power x TIME and a plan that waits on DRAM leaks for longer.

    A missing density is a REFUSAL, not a zero. The whole point of Defect 2 is
    that a side of the comparison was silently charged nothing; a knob that
    silently charges nothing is the same failure with a switch on it. `dram` is
    the one deliberate omission -- env.sh declares no off-chip density, exactly
    as ECC_DRAM_BACKGROUND_PJ and ECC_DRAM_REFRESH_PJ are 0 on purpose.
    """
    dens = getattr(cfg, "leakage_nw", None) or {}
    # THIS DESIGN's clock, not the study's (prompt_7 C1.5). Standby energy is
    # power x TIME and `ECC_LEAKAGE_NW` is POWER, so the period is applied here
    # exactly once -- at 200 MHz that is 5 ns, and reading the 1 GHz study
    # default instead would under-charge every level by 5x.
    period = latency_post_cycle_seconds(cfg)
    rows, total = [], 0.0
    for c in standby_components(raw, cfg):
        key = STANDBY_DENSITY.get(c["kind"])
        row = dict(c)
        if key is None:                       # dram: off chip, not modelled here
            row.update(density_nW=None, units=None, power_nW=0.0, energy_pJ=0.0,
                       note="off chip: env.sh section 6 declares no density, "
                            "as E_background and E_refresh are also 0")
            rows.append(row)
            continue
        if key not in dens:
            raise ValueError(
                f"ECC_STATIC_ENERGY=1 but ECC_LEAKAGE_NW has no {key!r} entry "
                f"(it holds {sorted(dens) or 'nothing'}), so level {c['level']!r} "
                f"would be charged zero standby energy -- which is the defect "
                f"this knob exists to fix. Source env.sh, or set "
                f"ECC_LEAKAGE_NW_LIST={key}=<nW>;...")
        units = 1.0 if c["kind"] == "mac" else c["bits_per_instance"]
        if not units:
            raise ValueError(
                f"standby: level {c['level']!r} is a {c['kind']} priced per "
                f"stored bit, and its stats declare no Size x Word bits to "
                f"multiply -- refusing to charge it zero")
        nw = float(dens[key])
        # nW x s = 1e-9 J = 1e3 pJ
        e = nw * units * c["instance_cycles"] * cycle_scale * period * 1e3
        row.update(density_nW=nw, density_key=key, units=units,
                   power_nW=nw * units, energy_pJ=e)
        rows.append(row)
        total += e
    return total, rows


def apply_standby_energy(raw, cfg, verbose=True):
    """Charge the accelerator's standby energy to the `Standby` category.

    THE SYMMETRY FIX (prompt_7 Defect 2). Applied to the `Raw` record, beside
    the MAC and DRAM overrides and after the raw cache, so it lands in
    `raw.base` -- which is where `ecc.build_stacks()` and every placement bar in
    `study/placement_study.py` start from. Charging it here rather than per arm is
    what makes "all three arms or none" structural instead of a thing three
    call sites have to remember: there is no code path that can give it to one
    arm and not another.

    It is NOT weight energy, so it is absent from `base_w` and the
    reconstruction arm never scales it by K/N. What it responds to is TIME,
    which is why ECC_LATENCY_MODEL decides the cycle count it is charged over.

    With ECC_STATIC_ENERGY=0 the record is returned unchanged apart from
    `standby` being filled with the reason, so a result file can say that no
    arm was charged and what Timeloop's own leakage bill would have been.
    """
    tl_leak = sum(float(c["timeloop_leakage_pJ"])
                  for c in standby_components(raw, cfg)) if raw.per_layer else 0.0
    info = {"charged": bool(getattr(cfg, "static_energy", False)),
            "densities_nW": dict(getattr(cfg, "leakage_nw", None) or {}),
            "cycle_seconds": latency_post_cycle_seconds(cfg),
            "timeloop_leakage_pJ": tl_leak,
            "timeloop_leakage_note": (
                "what Timeloop itself billed for leakage on this plan. It is "
                "inside the `Energy:` and `EDP(J*cycle)` the MAPPER optimised "
                "and was discarded by the report until prompt_7 Phase A "
                "(reporting rule R-4). Its ERT prices are 10^3-10^4 too low "
                "and three components price at exactly 0, which is why the "
                "charge below uses ECC_LEAKAGE_NW instead"),
            "energy_pJ": 0.0, "per_level": []}
    if not info["charged"]:
        info["reason"] = ("ECC_STATIC_ENERGY=0: NO arm is charged component "
                          "standby energy, so the accelerator's side of the "
                          "comparison carries none while the reconstruction "
                          "engines carry theirs (prompt_7 Defect 2)")
        raw.standby = info
        return raw

    scale, source = 1.0, "Timeloop cycles"
    lat = getattr(raw, "latency", None)
    if lat and raw.cycles:
        scale = float(lat["cycles"]) / float(raw.cycles)
        source = (f"roofline cycles ({lat['cycles']:,.0f} against Timeloop's "
                  f"{raw.cycles:,.0f}, x{scale:.4f})")
    total, rows = standby_energy(raw, cfg, cycle_scale=scale)
    info.update(energy_pJ=total, per_level=rows, cycle_scale=scale,
                cycles_source=source,
                ratio_to_timeloop_leakage=(total / tl_leak) if tl_leak else None)

    # Per layer, exactly -- not apportioned. Every layer's own utilized
    # instances and own cycles, so sum(per_layer) still equals base.sum().
    per_layer = []
    for lp in raw.per_layer:
        lp = dict(lp)
        if lp.get("status") == "ok" and lp.get("physical"):
            one = Raw(raw.base, raw.base_w, raw.base_i, 0.0, 0.0, 1, 0, 0,
                      per_layer=[lp], levels=[])
            e, _ = standby_energy(one, cfg, cycle_scale=scale)
            lp["standby_energy_pJ"] = e
            lp["total_energy_pJ"] = float(lp.get("total_energy_pJ", 0.0)) + e
        per_layer.append(lp)

    def with_standby(series):
        out = series.copy()
        out[STANDBY_CAT] = float(out.get(STANDBY_CAT, 0.0)) + total
        return out

    out = Raw(with_standby(raw.base), raw.base_w, raw.base_i,
              raw.e_dram_w, raw.dram_w_reads, raw.layers_ok, raw.layers_skipped,
              raw.weights, per_layer=per_layer, levels=raw.levels,
              noc_post=raw.noc_post, mac=raw.mac, cycles=raw.cycles)
    out.dram, out.latency, out.standby = raw.dram, raw.latency, info
    if verbose:
        print(f"  [STANDBY] {total / 1e6:,.3f} uJ charged to ALL THREE arms over "
              f"{source}; Timeloop's own leakage bill on the same plan was "
              f"{tl_leak / 1e6:,.6f} uJ"
              f"{f' (x{total / tl_leak:,.0f})' if tl_leak else ''}")
        for r in rows:
            if r["energy_pJ"]:
                print(f"            {r['level']:14s} {r['kind']:4s} "
                      f"{r['power_nW'] / 1e6:9.4f} mW/instance x "
                      f"{r['instance_cycles'] / max(raw.cycles or 1, 1):6.2f} mean "
                      f"instances = {r['energy_pJ'] / 1e6:10,.3f} uJ")
    return out


def apply_latency_model(raw, cfg, verbose=True):
    """Re-time this plan with `latency_post.roofline()`; fill `Raw.latency`.

    Evaluator-only and arm-INDEPENDENT: this is the plan's own run length at
    weight_scale 1.0, which is what the standby charge is spread over and what
    a reconstruction arm's re-timing is measured against. A bar that drives
    K/N of the weight bits off the die gets its own number from
    `latency_post.model_cycles(raw, cfg, weight_scale=K/N)`.
    """
    if not getattr(cfg, "latency_model", False):
        raw.latency = None
        return raw
    out = latency_post.model_cycles(raw, cfg)
    if out is None:
        raise ValueError(
            "ECC_LATENCY_MODEL=1 but this raw record carries no per-layer "
            "physical record to re-time (it predates prompt_7 Phase A). "
            "Re-gather it from the mapper cache -- `load_raw` does this "
            "automatically unless ECC_REPLOT_ONLY=1.")
    out["timeloop_cycles"] = raw.cycles
    out["vs_timeloop"] = (out["cycles"] / raw.cycles) if raw.cycles else None
    raw.latency = out
    if verbose:
        print(f"  [LATENCY] {latency_post.describe(cfg)}")
        print(f"            {out['cycles']:,} cycles = {out['seconds'] * 1e3:,.3f} ms "
              f"against Timeloop's {raw.cycles:,} "
              f"(x{out['vs_timeloop']:.4f}); binding: "
              + ", ".join(f"{k} on {v} layer(s)"
                          for k, v in sorted(out["binding_levels"].items(),
                                             key=lambda kv: -kv[1])))
    return raw


def gather(cfg, mapper, model, layers, verbose=True):
    """Map every layer of `model` on one architecture and aggregate.

    Layers are labelled by their STABLE workload name, not by index, so the
    per-layer rows in a result JSON can be matched to the layers a development
    run selected even if the workload file is later regenerated.
    """
    rows, n_ok, n_skip = [], 0, 0
    per_layer = []
    cycles_total = 0
    cycles_known = True
    for i, layer in enumerate(layers):
        stats = mapper.stats_for(layer)
        if stats is None:
            n_skip += 1
            if verbose:
                print(f"    [skip] {model}/{layer.name} {layer.shape_name}: mapper failed")
            per_layer.append({"layer": layer.name, "shape": layer.shape_name,
                              "status": "unmapped", "weights": layer.weights})
            continue
        layer_rows = parse_stats(stats, layer.name, scale=layer.count)
        # prompt_6 RULE 3: this plan's cycles, x repeat count, so the idle
        # term can be charged for the run this record describes.
        cyc = parse_cycles(stats)
        if cyc is None:
            cycles_known = False
        else:
            cycles_total += cyc * layer.count
        # Spatial reductions and the psum word width: counted by Timeloop,
        # costed here (noc_post.py). Added per layer so the per-layer totals
        # below and the category totals agree.
        layer_rows = noc_post.augment(layer_rows, mapper.arch, cfg)
        rows += layer_rows
        n_ok += 1
        # prompt_7 Phase A: the PHYSICAL record of this plan -- declared
        # geometry, declared bandwidths, per-instance counts, and what Timeloop
        # billed for leakage. Pure Timeloop output, so it is cached beside the
        # energies and both the roofline (latency_post) and the standby charge
        # (standby_energy) read the SAME rows for the same level.
        physical = physical_record(parse_levels(stats)[0])

        ldf = pd.DataFrame(layer_rows)
        ldf = ldf[ldf.energy_pJ.notna()]
        w = ldf[ldf.dataspace == "Weights"] if not ldf.empty else ldf
        # Networks are named after the levels they join, so "NoC: DRAM <==>
        # ifmap_glb" contains "dram"; keep them out of the DRAM weight tally.
        dw = (w[[("dram" in str(lv).lower()) and not str(lv).startswith("NoC")
                 for lv in w.level]] if not w.empty else w)
        per_layer.append({
            "layer": layer.name,
            "shape": layer.shape_name,
            "status": "ok",
            "weights": layer.weights,
            "macs": layer.macs,
            "grouped": layer.is_grouped,
            "repeat_count": layer.count,
            "mapping_id": (mapper.mappings.get(layer.shape_name) or {}).get("mapping_id"),
            "mapping_cached": (mapper.mappings.get(layer.shape_name) or {}).get("cached"),
            "total_energy_pJ": float(ldf.energy_pJ.sum()) if not ldf.empty else 0.0,
            "dram_weight_reads": (float(dw.reads.fillna(0.0).sum())
                                  if not dw.empty else 0.0),
            "dram_weight_energy_pJ": (float(dw.energy_pJ.sum())
                                      if not dw.empty else 0.0),
            "cycles": None if cyc is None else cyc * layer.count,
            "physical": physical,
        })

    if verbose:
        print(f"  {model:18s} {n_ok}/{len(layers)} layers ok, {n_skip} skipped", flush=True)
    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df[df.energy_pJ.notna() & (df.energy_pJ > 0)].reset_index(drop=True)
    if df.empty:
        return None
    df["category"] = [classify(lv, inst, cfg.classify_mode)
                      for lv, inst in zip(df.level, df.instances)]

    weights_only = df[df.dataspace == "Weights"]
    dram_weights = weights_only[weights_only.category == "DRAM"]

    # Per-(level, dataspace) totals: what the storage spec calls the access
    # counts and per-component breakdown. Summed across layers, because that is
    # the granularity a mapping is shared at.
    if "word_bits" not in df.columns:
        df["word_bits"] = None
    lv = (df.groupby(["level", "dataspace", "category"], as_index=False)
            .agg(energy_pJ=("energy_pJ", "sum"), reads=("reads", "sum"),
                 writes=("writes", "sum"), instances=("instances", "max"),
                 word_bits=("word_bits", "max")))
    levels = [{"level": r.level, "dataspace": r.dataspace, "category": r.category,
               "instances": (None if pd.isna(r.instances) else int(r.instances)),
               # prompt_6 RULE 1: the measured word, so an evaluator can tell a
               # q-bit plan from an 8-bit one without re-opening the stats
               "word_bits": (None if pd.isna(r.word_bits) else int(r.word_bits)),
               "energy_pJ": float(r.energy_pJ),
               "reads": (None if pd.isna(r.reads) else float(r.reads)),
               "writes": (None if pd.isna(r.writes) else float(r.writes))}
              for r in lv.itertuples(index=False)]

    return Raw(
        base=category_energy(df, cfg),
        base_w=category_energy(weights_only, cfg),
        base_i=category_energy(df[df.dataspace == "Inputs"], cfg),
        e_dram_w=float(dram_weights.energy_pJ.sum()),
        dram_w_reads=float(dram_weights.reads.fillna(0.0).sum()),
        layers_ok=n_ok, layers_skipped=n_skip,
        weights=sum(l.weights for l in layers),
        per_layer=per_layer, levels=levels,
        noc_post=noc_post.stamp(mapper.arch, cfg),
        cycles=cycles_total if cycles_known else None,
    )


def finish(raw, cfg, verbose=True):
    """Every evaluator-side model, in the one order they must be applied.

    The raw cache holds pure Timeloop output; these four turn it into what is
    reported, and NONE of them is in the mapping fingerprint, so a change to
    any of them redraws in milliseconds:

      1. `apply_mac_override`  -- the DENOMINATOR of every ECC percentage
      2. `apply_dram_override` -- the per-bit DRAM price (the NUMERATOR knob)
      3. `apply_latency_model` -- how long this plan actually takes
      4. `apply_standby_energy` -- power x that time, to ALL THREE arms

    The order is not arbitrary: standby energy is charged over the run length,
    so the roofline has to exist before it.
    """
    raw = apply_dram_override(apply_mac_override(raw, cfg, verbose), cfg, verbose)
    raw = apply_latency_model(raw, cfg, verbose)
    return apply_standby_energy(raw, cfg, verbose)


# ------------------------------------------------------------------ raw cache
def load_raw(results, cfg, arch, model, variant=None, fingerprint=None,
             layers=None):
    """The cached Raw for (arch, model), or None if there is none worth using.

    `layers` -- the Layer list the caller is about to evaluate -- makes the
    cache self-checking: a record whose per-layer shapes are not exactly the
    shapes the workload now produces is REJECTED, not reused. The mapper cache
    is keyed by shape name, so a changed workload definition (the depthwise
    correction, for one) simply adds new shapes there; but the raw record is
    keyed by model and would otherwise keep serving energies aggregated from the
    old shapes for as long as the fingerprint stayed the same.
    """
    path = results.raw_path(arch, model, variant, fingerprint)
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text())
        raw = Raw.from_json(blob, cfg)
    except Exception as exc:
        print(f"  [warn] ignoring {path.name}: {exc}")
        return None
    # Same self-check for the evaluator-only NoC coefficients: they are applied
    # when the record is aggregated, not by the mapper, so the mapper
    # fingerprint in the path cannot protect against a change to them.
    want_post = noc_post.stamp(arch, cfg)
    if blob.get("noc_post") != want_post:
        print(f"  [stale] {arch}/{model}: raw record was aggregated with different "
              f"evaluator-only NoC terms ({blob.get('noc_post')} vs {want_post}) "
              f"-> re-gathering from the mapper cache")
        return None
    # A record aggregated while shapes were still being mapped (a parallel
    # per-shape run, or an evaluation started before the map finished) holds
    # `unmapped` layers and must not be served as a hit once those shapes exist:
    # it would draw a one-layer figure labelled as the whole model (2026-09-09,
    # eval 41479689). Re-gathering from the mapper cache costs seconds.
    # prompt_6 RULE 1 needs each Weights storage row's measured `Word bits`;
    # a record aggregated before the field existed cannot say whether the
    # mapper narrowed a level, so it is re-gathered from the mapper cache
    # (seconds) rather than trusted to be 8-bit.
    unmeasured = [r.get("level") for r in blob.get("levels", [])
                  if r.get("dataspace") == "Weights" and r.get("category") != "DRAM"
                  and not str(r.get("level", "")).startswith("NoC")
                  and "word_bits" not in r]
    if unmeasured and not cfg.replot_only:
        print(f"  [stale] {arch}/{model}: raw record predates Word bits recording "
              f"(prompt_6 RULE 1; e.g. {unmeasured[0]}) -> re-gathering from the "
              f"mapper cache")
        return None
    # prompt_7 Phase A: the roofline and the standby charge both read the
    # per-layer PHYSICAL record. A record written before it existed cannot
    # supply either, so it is re-gathered from the mapper cache (seconds) --
    # but ONLY when a feature that needs it is on, so a run with both knobs
    # off keeps reading every record written before this landed and reproduces
    # its totals to the pJ.
    if (cfg.static_energy or cfg.latency_model) and not cfg.replot_only:
        no_phys = [lp.get("layer") for lp in blob.get("per_layer", [])
                   if lp.get("status") == "ok" and not lp.get("physical")]
        if no_phys:
            print(f"  [stale] {arch}/{model}: raw record predates the per-layer "
                  f"physical record (prompt_7 Phase A; e.g. {no_phys[0]}), which "
                  f"ECC_STATIC_ENERGY / ECC_LATENCY_MODEL need -> re-gathering "
                  f"from the mapper cache")
            return None
    if blob.get("cycles") is None and not cfg.replot_only:
        print(f"  [stale] {arch}/{model}: raw record predates cycle recording "
              f"(prompt_6 RULE 3: the idle term is per cycle) -> re-gathering from "
              f"the mapper cache")
        return None
    unmapped = [lp.get("layer") for lp in blob.get("per_layer", [])
                if lp.get("status") == "unmapped"]
    if unmapped and not cfg.replot_only:
        print(f"  [stale] {arch}/{model}: raw record has {len(unmapped)} unmapped layer(s) "
              f"(e.g. {unmapped[0]}) -> re-gathering from the mapper cache")
        return None
    if layers is not None:
        want = [l.shape_name for l in layers]
        have = [lp.get("shape") for lp in blob.get("per_layer", [])]
        if have != want:
            changed = sorted(set(want) - set(have))
            print(f"  [stale] {arch}/{model}: raw record was built from a different "
                  f"layer list ({len(have)} shapes, now {len(want)}; "
                  f"{len(changed)} new, e.g. {changed[0] if changed else '-'}) "
                  f"-> re-gathering from the mapper cache")
            return None
    return raw


def save_raw(results, arch, model, raw, variant=None, fingerprint=None):
    path = results.raw_path(arch, model, variant, fingerprint)
    with open(path, "w", newline="\n") as fh:
        fh.write(json.dumps(raw.to_json(), indent=1))
    return path


def collect(cfg, results, arch, mapper_factory, models, variant=None,
            fingerprint=None):
    """Return `{model: Raw}` for one architecture, using the raw cache.

    With ECC_REPLOT_ONLY=1 the mapper is never constructed, so this runs on a
    laptop with no container: it reads `results/_raw/` and nothing else.
    """
    raws, need_mapping = {}, []
    # THE RAW CACHE SITS IN FRONT OF THE MAPPER: a hit returns before
    # `mapper_factory()` is even called. So ECC_RERUN_OPTIMISER=1 has to
    # invalidate the raw record as well, or the flag would silently do nothing
    # on every model that has been evaluated once -- which is all of them.
    if cfg.rerun_optimiser:
        print(f"  ECC_RERUN_OPTIMISER=1: ignoring the raw-energy cache for "
              f"{', '.join(models)} so the mapper is reached")
    for model in models:
        cached = (None if cfg.rerun_optimiser else
                  load_raw(results, cfg, arch, model, variant, fingerprint,
                           layers=models[model]))
        if cached is not None:
            raws[model] = cached
        else:
            need_mapping.append(model)

    if raws:
        print(f"  raw cache hit: {', '.join(raws)}")
    if not need_mapping:
        # The evaluator-side models are applied AFTER the cache, never to it.
        return {m: finish(r, cfg) for m, r in raws.items()}, None
    if cfg.replot_only:
        print(f"  [skip] ECC_REPLOT_ONLY=1 and no raw cache for: {', '.join(need_mapping)}")
        return {m: finish(r, cfg) for m, r in raws.items()}, None

    mapper = mapper_factory()
    for model in need_mapping:
        raw = gather(cfg, mapper, model, models[model])
        if raw is None:
            print(f"  !! {model}: no valid layers on {arch}")
            continue
        raws[model] = raw
        save_raw(results, arch, model, raw, variant, fingerprint)
    print(f"  {mapper.summary()}")
    return {m: finish(r, cfg) for m, r in raws.items()}, mapper
