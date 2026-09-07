"""Aggregate Timeloop stats into the plotted energy categories.

`gather()` turns one (architecture, model) pair into a small `Raw` record. That
record is ECC-independent -- it is pure Timeloop output -- so it is cached to
`results/_raw/` and every ECC configuration is then a few milliseconds of
arithmetic on top. Changing BCH_N/K and re-running costs nothing.
"""
from __future__ import annotations

import json

import pandas as pd

from .timeloop import classify, parse_stats

#: Categories that come from Timeloop.
#: "NoC" is the interconnect between levels -- wire, router and ingress energy
#: on every network Timeloop instantiates (archs/_shared/noc.yaml). It was
#: folded into the Local label historically but was always 0.00 there, because
#: Timeloop's wire model is a stub; it is a category of its own now that it
#: is costed, and the Local label no longer claims to contain it.
PHYS_CATS = ("DRAM", "Global buffer", "Local (spads/RF)", "NoC", "Compute")
#: Split variants, used when ECC_SPLIT_READ_WRITE=1. NoC is not split: a
#: network has ingresses, not reads and writes.
PHYS_CATS_SPLIT = ("DRAM", "Global buffer (read)", "Global buffer (write)",
                   "Local (read)", "Local (write)", "NoC", "Compute")
#: Categories the ECC model adds.
ECC_CATS = ("ECC decode", "Reconstruction")

SPLITTABLE = ("Global buffer", "Local (spads/RF)")


def phys_cats(cfg):
    return list(PHYS_CATS_SPLIT if cfg.split_read_write else PHYS_CATS)


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
                 "layers_ok", "layers_skipped", "weights", "per_layer", "levels")

    def __init__(self, base, base_w, base_i, e_dram_w, dram_w_reads,
                 layers_ok, layers_skipped, weights, per_layer=None, levels=None):
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
        }

    @classmethod
    def from_json(cls, d, cfg):
        cats = plot_cats(cfg)
        stored = set(d.get("base", {}))
        missing = [c for c in phys_cats(cfg) if c not in stored]
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
                   d.get("weights", 0), d.get("per_layer", []), d.get("levels", []))


def gather(cfg, mapper, model, layers, verbose=True):
    """Map every layer of `model` on one architecture and aggregate.

    Layers are labelled by their STABLE workload name, not by index, so the
    per-layer rows in a result JSON can be matched to the layers a development
    run selected even if the workload file is later regenerated.
    """
    rows, n_ok, n_skip = [], 0, 0
    per_layer = []
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
        rows += layer_rows
        n_ok += 1

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
    lv = (df.groupby(["level", "dataspace", "category"], as_index=False)
            .agg(energy_pJ=("energy_pJ", "sum"), reads=("reads", "sum"),
                 writes=("writes", "sum"), instances=("instances", "max")))
    levels = [{"level": r.level, "dataspace": r.dataspace, "category": r.category,
               "instances": (None if pd.isna(r.instances) else int(r.instances)),
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
    )


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
    for model in models:
        cached = load_raw(results, cfg, arch, model, variant, fingerprint,
                          layers=models[model])
        if cached is not None:
            raws[model] = cached
        else:
            need_mapping.append(model)

    if raws:
        print(f"  raw cache hit: {', '.join(raws)}")
    if not need_mapping:
        return raws, None
    if cfg.replot_only:
        print(f"  [skip] ECC_REPLOT_ONLY=1 and no raw cache for: {', '.join(need_mapping)}")
        return raws, None

    mapper = mapper_factory()
    for model in need_mapping:
        raw = gather(cfg, mapper, model, models[model])
        if raw is None:
            print(f"  !! {model}: no valid layers on {arch}")
            continue
        raws[model] = raw
        save_raw(results, arch, model, raw, variant, fingerprint)
    print(f"  {mapper.summary()}")
    return raws, mapper
