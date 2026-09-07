#!/usr/bin/env python3
"""Print the model x architecture energy matrix from results/evaluation/.

    python3 hpc/summary.py [--scope layers-full] [--csv results/tables/panel_matrix.csv]

Reads the newest result JSON in every
`results/evaluation/{phase}/{arch}/{model}/{code}/{prec}/{scope}/{mapper}/`
directory and lays the totals out as one row per model, one column per
architecture. Only the standard library is used, so it runs with the system
python outside the container.

Three numbers per cell are available; `--field` chooses which is shown:

    timeloop   the Timeloop/Accelergy energy alone (no ECC arithmetic on top)
    parity     the external BCH parity charged in DRAM
    total      timeloop + parity == the `baseline_external_parity` variant

`timeloop` is the default because it is the number the architectures are
actually being compared on; the parity term is nearly architecture-independent
and would compress the spread.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL = ROOT / "results" / "evaluation"

PARITY_KEY = "DRAM external BCH parity"


def newest(d: pathlib.Path):
    """The most recent timestamped result in one evaluation directory."""
    runs = sorted((p for p in d.glob("*.json") if p.name != "latest.json"),
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def cell(path: pathlib.Path):
    """(timeloop uJ, parity uJ, total uJ) from one result file, or None."""
    doc = json.loads(path.read_text())
    for var in doc.get("variants", []):
        if var.get("name") != "baseline_external_parity":
            continue
        if var.get("status") != "evaluated":
            return None
        comp = var.get("energy_by_component_pJ") or {}
        parity = comp.get(PARITY_KEY, 0.0)
        total = var["total_energy_pJ"]
        return ((total - parity) / 1e6, parity / 1e6, total / 1e6)
    return None


def collect(scope=None, mapper=None, phase="Pre"):
    """{(model, arch): (timeloop, parity, total, scope, mapper, path)}"""
    out = {}
    root = EVAL / phase
    if not root.is_dir():
        return out
    for arch_dir in sorted(root.iterdir()):
        if not arch_dir.is_dir():
            continue
        for model_dir in sorted(arch_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for run_dir in model_dir.glob("*/*/*/*"):
                if not run_dir.is_dir():
                    continue
                sc, mp = run_dir.parent.name, run_dir.name
                if scope and sc != scope:
                    continue
                if mapper and mp != mapper:
                    continue
                p = newest(run_dir)
                if p is None:
                    continue
                got = cell(p)
                if got is None:
                    continue
                key = (model_dir.name, arch_dir.name)
                prev = out.get(key)
                if prev is None or p.stat().st_mtime > prev[-1].stat().st_mtime:
                    out[key] = (*got, sc, mp, p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", help="layer scope, e.g. layers-full "
                                    "(default: every scope found)")
    ap.add_argument("--mapper", help="mapper slug filter")
    ap.add_argument("--phase", default="Pre", choices=("Pre", "Post"))
    ap.add_argument("--field", default="timeloop",
                    choices=("timeloop", "parity", "total"))
    ap.add_argument("--csv", help="also write the matrix here")
    args = ap.parse_args(argv)

    data = collect(args.scope, args.mapper, args.phase)
    if not data:
        print("no evaluated results found under", EVAL / args.phase)
        return 1

    idx = {"timeloop": 0, "parity": 1, "total": 2}[args.field]
    models = sorted({m for m, _ in data})
    archs = sorted({a for _, a in data})

    scopes = sorted({v[3] for v in data.values()})
    print(f"\n{args.field} energy, uJ   phase={args.phase}   "
          f"scope={'/'.join(scopes)}\n")
    w = max(14, max(len(a) for a in archs) + 2)
    print(f"{'model':<16}" + "".join(f"{a:>{w}}" for a in archs))
    print("-" * (16 + w * len(archs)))
    for m in models:
        row = f"{m:<16}"
        for a in archs:
            v = data.get((m, a))
            row += f"{v[idx]:>{w},.3f}" if v else f"{'-':>{w}}"
        print(row)

    missing = [(m, a) for m in models for a in archs if (m, a) not in data]
    if missing:
        print("\nmissing cells:")
        for m, a in missing:
            print(f"  {m} x {a}")

    if args.csv:
        out = pathlib.Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["model", "arch", "scope", "mapper",
                         "timeloop_uJ", "parity_uJ", "total_uJ", "source"])
            for (m, a), v in sorted(data.items()):
                wr.writerow([m, a, v[3], v[4],
                             f"{v[0]:.6f}", f"{v[1]:.6f}", f"{v[2]:.6f}",
                             str(v[5].relative_to(ROOT))])
        print(f"\ncsv -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
