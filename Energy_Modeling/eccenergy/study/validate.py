"""validate: does every architecture obey the standardized-comparison contract?

    bash run.sh validate

`archs/_shared/standard.yaml` states what must be IDENTICAL across designs --
weight precision, activation precision, DRAM geometry, process node -- and what
each design is allowed to keep as its own, with a citation. This experiment
checks every architecture against it and exits non-zero on a violation.

WHY IT EXISTS. The work that started Task 1 was the discovery that two designs
were being compared at different operand widths. That was invisible because
nothing checked. It is a class of defect, not a single bug: a datawidth, a DRAM
word size, a process node, an accumulator declared one width in storage and
another in arithmetic. Each of them turns a modelling slip into an
"architectural" result. So the contract is written down, and a run checks it.

WHAT IT DELIBERATELY DOES NOT DO. It does not equalise topology, capacity or
weight bypass rules -- those are the architectures, and Task 1 is explicit that
they must be preserved. It prints them instead, so the reviewer can see that
standardizing the precisions did not flatten the designs.

`diagnose` is the other half of this: it reports cross-architecture confounds
in the MEASURED results. `validate` never needs a container or a mapper cache.
"""
from __future__ import annotations

import json

from ..arch.fingerprint import arch_fingerprint
from ..arch.load import arch_source, install_local_archs, load_provenance, load_standard
from ..arch.validate import validate_arch
from ..paths import DESIGNS_DIR, Results


def _hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def target_archs(cfg):
    """Every architecture worth checking, whatever axis this run sweeps.

    A model or BCH sweep pins `cfg.archs` to one design; the contract is about
    the comparison BETWEEN designs, so the architecture list is what matters.
    """
    return list(dict.fromkeys(list(cfg.sweep_archs) + list(cfg.archs)))


def run(cfg):
    if DESIGNS_DIR.exists():
        install_local_archs(verbose=False)

    std = load_standard()["study"]
    prov = load_provenance()

    _hr("THE SHARED CONTRACT  --  archs/_shared/standard.yaml")
    print(f"  weight precision      : {std['weight_bits']}b   (the protected payload)")
    print(f"  activation precision  : {std['activation_bits']}b")
    print(f"  accumulator policy    : {std['accumulator_policy']}")
    print("                          per architecture, as published. Partial-sum")
    print("                          width is a design property; equalising it")
    print("                          would equalise the architectures.")
    print(f"  packing rule          : {std['packing_rule']}")
    print("                          a level holds floor(physical_bits / operand_bits)")
    print("                          operands, applied to EVERY level of EVERY design")
    print(f"  technology            : {std['technology']}  (logic and DRAM)")
    d = std["dram"]
    print(f"  DRAM                  : {d['type']}, {d['width_bits']}b word, "
          f"{d['datawidth_bits']}b operand, depth {d['depth_words']:,}")

    reports, failed = [], []
    for arch in target_archs(cfg):
        try:
            rep = validate_arch(arch, cfg)
        except FileNotFoundError:
            print(f"\n  [skip] {arch}: no arch.yaml found")
            continue
        rep["fingerprint"] = arch_fingerprint(arch, cfg)
        rep["provenance_present"] = arch in prov
        reports.append(rep)
        if rep["violations"]:
            failed.append(arch)

    _hr("PER ARCHITECTURE")
    for rep in reports:
        f = rep["facts"]
        print(f"\n  {rep['arch']}  ({f.get('label', '')})")
        print(f"    source        : {f.get('source')}")
        print(f"    origin        : {f.get('origin')}   "
              f"provenance recorded: {'yes' if rep['provenance_present'] else 'NO'}")
        print(f"    fingerprint   : {rep['fingerprint']}")
        mac = f.get("mac") or {}
        print(f"    precisions    : weight/activation {std['weight_bits']}b   "
              f"multiplier {mac.get('multiplier_width')}b   "
              f"accumulator {f.get('accumulator_bits')}b")
        if f.get("accumulator_evidence"):
            print(f"    acc. evidence : {f['accumulator_evidence'][:150]}")
        dram = f.get("dram") or {}
        print(f"    DRAM          : width {dram.get('width')}b  "
              f"datawidth {dram.get('datawidth')}b  depth {dram.get('depth')}")

        # ---- what is PRESERVED, not equalised ---------------------------
        print(f"    PE fanout     : {f.get('total_fanout')} instances")
        print(f"    weight levels : ", end="")
        wl = f.get("weight_levels") or []
        if not wl:
            print("none on chip")
        else:
            print("; ".join(f"{w['name']} {w['entries_each']:,}x{w['instances']}"
                            for w in wl))
        print(f"    weight capacity: {f.get('weight_capacity', 0):,} weights on chip   "
              f"(GLB keeps weights: {f.get('glb_keeps_weights')})")

        for n in rep["notes"]:
            print(f"    . {n}")
        for v in rep["violations"]:
            print(f"    X {v}")

    # ---- the cross-architecture view --------------------------------------
    _hr("PRESERVED DIFFERENCES  --  these are the architectures, not defects")
    print(f"  {'architecture':26s}{'acc b':>7s}{'fanout':>9s}"
          f"{'on-chip weights':>18s}{'GLB keeps W':>13s}")
    for rep in reports:
        f = rep["facts"]
        print(f"  {rep['arch']:26s}{str(f.get('accumulator_bits')):>7s}"
              f"{f.get('total_fanout', 0):>9,}"
              f"{f.get('weight_capacity', 0):>18,}"
              f"{str(f.get('glb_keeps_weights')):>13s}")
    print("\n  On-chip weight capacity spanning a wide range is EXPECTED and is one")
    print("  of the things this study measures: less capacity means more DRAM")
    print("  refetch, which is where the BCH parity cost lands.")

    results = Results(cfg).prepare()
    out = results.write_manifest(
        {"standard": load_standard(), "reports": reports,
         "violations_total": sum(len(r["violations"]) for r in reports)},
        stem="validate")

    _hr("RESULT")
    if failed:
        print(f"  FAILED: {len(failed)} architecture(s) break the contract: "
              f"{', '.join(failed)}")
        print("  Each violation above says what to change and why it matters.")
        print(f"  Full report: {out}")
        raise SystemExit(1)

    print(f"  PASS: {len(reports)} architecture(s) obey the shared contract.")
    print("  Operand precision, DRAM geometry and process node are identical;")
    print("  topology, capacity, weight bypass rules and accumulator width are")
    print("  each design's own, and are recorded with their sources.")
    print(f"  Full report: {out}")
    return reports
