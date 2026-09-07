"""diagnose: audit the architectures instead of plotting the ECC arms.

Two halves.

STATIC AUDIT reads each arch.yaml and reports the declarations that decide a
weight-energy result: technology node, DRAM word width, whether the global
buffer keeps or bypasses Weights, total on-chip weight capacity, partial-sum
precision. Then it names the cross-architecture confounds.

MEASURED COMPARISON reads whatever is in `results/_raw/` (no container needed)
and reports, per architecture: total energy, the per-category split, DRAM weight
reads, and energy per weight read. An unexpected architecture ordering is almost
always visible as an unexpected DRAM weight-read count, and this is where you
see it.

The audit always covers every architecture in ECC_SWEEP_ARCHS, not just the one
a non-architecture sweep happens to hold fixed -- comparing one design against
itself says nothing.

Run:  bash run.sh diagnose
"""
from __future__ import annotations

import pandas as pd

from ..archs import (arch_fingerprint, audit, audit_findings, effective_variant,
                     install_local_archs)
from ..energy import load_raw, plot_cats
from ..paths import DESIGNS_DIR, Results


def diag_archs(cfg):
    """Every architecture worth auditing, whatever axis this run sweeps.

    A model or BCH sweep pins `cfg.archs` to the single held design; the point
    of the audit is the comparison BETWEEN designs, so it reads the architecture
    sweep list instead.
    """
    return list(dict.fromkeys(cfg.sweep_archs or cfg.archs))


def _hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def static_audit(cfg):
    if DESIGNS_DIR.exists():
        install_local_archs(verbose=False)
    infos = []
    for arch in diag_archs(cfg):
        try:
            infos.append(audit(arch, cfg))
        except FileNotFoundError:
            print(f"  [skip] {arch}: no arch.yaml found "
                  f"(looked in archs/ and example_designs/)")
    if not infos:
        raise SystemExit("no architecture YAMLs found to audit")

    _hr("STATIC AUDIT  --  what each arch.yaml declares")
    rows = {}
    for i in infos:
        rows[i["arch"]] = {
            "fidelity": i.get("fidelity", "?"),
            "technology": "/".join(i["technology"]) or "<inherited>",
            "DRAM datawidth (b)": i["dram_datawidth"],
            "DRAM word width (b)": i["dram_width"],
            "GLB keeps Weights": i["glb_keeps_weights"],
            "on-chip weight capacity": f"{i['weight_capacity']:,}",
            "accumulator width (b)": i.get("adder_width"),
            "psum datawidth (b)": "/".join(str(d) for _n, d in i["psum_datawidth"]) or "-",
            "total PE fanout": i["total_fanout"],
            "loop levels": i.get("loop_levels"),
            "mapper victory": i.get("victory"),
        }
    print(pd.DataFrame(rows).to_string())

    for i in infos:
        print(f"\n  {i['arch']}  ({i['source']})")
        for lv in i["levels"]:
            policy = ("keeps " + ",".join(lv["keep"])) if lv["keep"] else "keeps all"
            if lv["bypass"]:
                policy += "  bypass " + ",".join(lv["bypass"])
            print(f"    {lv['name']:20s} {str(lv['class']):22s} "
                  f"depth={lv['depth']:<7d} width={str(lv['width']):<5s} "
                  f"dw={str(lv['datawidth']):<3s} entries={lv['entries']:<7d} "
                  f"x{lv['instances']:<5d} {policy}")
        for wl in i["weight_levels"]:
            print(f"    -> weights fit: {wl['name']} {wl['entries_each']} x "
                  f"{wl['instances']} = {wl['entries_total']:,}")

    findings = audit_findings(infos, cfg)
    _hr(f"CONFOUNDS  --  {len(findings)} found")
    if not findings:
        print("  none: the architectures are declared comparably.")
    for n, f in enumerate(findings, 1):
        print(f"\n  {n}. {f}")
    return infos, findings


def measured_comparison(cfg):
    results = Results(cfg)
    cats = plot_cats(cfg)
    rows, missing = {}, []
    for arch in diag_archs(cfg):
        for model in cfg.models:
            # The raw cache is now keyed by the architecture fingerprint too, so
            # a diagnose run reads the records belonging to THIS arch.yaml rather
            # than whatever was collected under the same treatment slug before it
            # was edited.
            raw = load_raw(results, cfg, arch, model, effective_variant(arch, cfg),
                           arch_fingerprint(arch, cfg))
            if raw is None:
                missing.append(f"{arch}/{model}")
                continue
            row = {c: float(raw.base.get(c, 0.0)) / 1e6 for c in cats}
            row["total_uJ"] = raw.total / 1e6
            row["DRAM_weight_uJ"] = raw.e_dram_w / 1e6
            row["DRAM_weight_reads"] = raw.dram_w_reads
            row["pJ_per_DRAM_weight_read"] = (
                raw.e_dram_w / raw.dram_w_reads if raw.dram_w_reads else 0.0)
            row["model_weights"] = raw.weights
            row["DRAM_refetch_x"] = (raw.dram_w_reads / raw.weights) if raw.weights else 0.0
            row["layers_ok"] = raw.layers_ok
            row["layers_skipped"] = raw.layers_skipped
            rows[f"{arch} / {model}"] = row

    if not rows:
        _hr("MEASURED COMPARISON  --  no raw energies cached yet")
        print("  results/_raw/ is empty for this configuration.")
        print("  Run a sweep in the container first, e.g.:")
        print("     ECC_SWEEP=arch bash run.sh")
        if missing:
            print(f"  looked for: {', '.join(missing)}")
        return None

    df = pd.DataFrame(rows).T
    _hr("MEASURED COMPARISON  --  from results/_raw/ (energies in uJ)")
    show = [c for c in cats if df[c].abs().sum() > 0] + [
        "total_uJ", "DRAM_weight_uJ", "DRAM_weight_reads",
        "pJ_per_DRAM_weight_read", "DRAM_refetch_x", "layers_ok", "layers_skipped"]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df[show].to_string(float_format=lambda v: f"{v:,.2f}"))
    if missing:
        print(f"\n  not cached: {', '.join(missing)}")

    # ---- the ordering, and the one number that usually explains it ---------
    _hr("ORDERING  --  cheapest architecture first")
    order = df.sort_values("total_uJ")
    best = order["total_uJ"].iloc[0]
    for name, row in order.iterrows():
        print(f"  {name:44s} {row['total_uJ']:12,.2f} uJ   "
              f"{row['total_uJ'] / best:5.2f}x   "
              f"DRAM weight reads {row['DRAM_weight_reads']:>14,.0f}  "
              f"({row['DRAM_refetch_x']:.2f} fetches per weight)")

    print("\n  Reading this table:")
    print("    * DRAM_refetch_x is how many times the average weight crosses the DRAM")
    print("      boundary. It is set by on-chip weight capacity and by whether the")
    print("      global buffer keeps Weights -- not by the dataflow name.")
    print("    * pJ_per_DRAM_weight_read should be IDENTICAL across architectures at")
    print("      the same node and datawidth. If it differs by ~2x, one design")
    print("      declares datawidth 16 while another declares 8.")

    out = Results(cfg).prepare().table_path("diagnose")
    # An explicit newline="" handle: this repo is LF-only, and to_csv() to a
    # path opens in text mode, so an audit run from the Windows side rather
    # than in the container would write CRLF.
    with open(out, "w", newline="") as fh:
        df.to_csv(fh, index_label="arch / model", lineterminator="\n")
    print(f"\n  table written to {out}")
    return df


def run(cfg):
    infos, findings = static_audit(cfg)
    df = measured_comparison(cfg)

    results = Results(cfg).prepare()
    audit_path = results.write_manifest({"findings": findings, "audits": infos},
                                        stem="diagnose")

    _hr("NEXT STEPS")
    if cfg.arch_fidelity != "paper":
        print("  ECC_ARCH_FIDELITY=paper gives each design the psum precision and")
        print("  scratchpad geometry its paper declares. Without it, eyeriss_like bills")
        print("  its 16-bit partial sums as bytes at the global buffer and comes out")
        print("  ahead of eyeriss_v2_like for that reason alone. Start there.")
        print("")
    if cfg.opt_metric != "energy":
        print(f"  The mapper is minimising {cfg.opt_metric}, not energy. Set")
        print("  ECC_OPT_METRIC=energy before comparing architectures by energy.")
        print("")
    print("  Equalise the storage datawidth -- one variable, and the cheapest fix:")
    print(f"     ECC_FORCE_DATAWIDTH={cfg.weight_bits} bash run.sh")
    print("  That is a no-op on any design already declared at the weight width, so")
    print("  those architectures keep their existing mapper cache and only the")
    print("  mis-declared ones are re-mapped.")
    print("  The treated results get their own raw-cache subtree, so nothing")
    print("  already collected is overwritten or mixed in.")
    print("")
    print("  ECC_FORCE_TECHNOLOGY changes globals.yaml, which costs DRAM for EVERY")
    print("  design at once, so it invalidates every cache. Only worth it if you")
    print("  specifically want DRAM costed at the same node as the accelerators;")
    print("  all five arch containers already declare 45nm.")
    print(f"\n  audit written to {audit_path}")
    return df
