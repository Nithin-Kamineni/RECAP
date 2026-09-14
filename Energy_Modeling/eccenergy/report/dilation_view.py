"""The Task 4 CLI and its CSV writers.

    python3 -m eccenergy.report.dilation_view --survey
    python3 -m eccenergy.report.dilation_view --gate --scales 1.0
    python3 -m eccenergy.report.dilation_view --levels --scales 1.0 --csv <path>

`main()` is the one entry point: it chooses among the reports in
`study/dilation.py` and the tables in `study/dilation_tables.py`, prints them,
and writes the CSV beside `results/tables/` when asked. `to_csv()`,
`level_table_to_csv()` and `dataspace_to_csv()` are the only writers here, each
with its own column list so a table's shape is declared rather than inferred.

This module draws no figure -- Task 4's output is text and CSV -- but it is the
DRIVER, and a driver belongs at the top of the stack. ProjectRestructure phase 3
cut it out of `experiments/dilation.py`.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from .. import config as configmod
from ..physics import widths

from ..study.dilation import render, survey, sweep
from ..study.dilation_tables import DATASPACE_CSV_COLUMNS, convergence_gate, dataspace_table, level_table, utilisation_table
from ..settings import guards


#: prompt_2's convergence gate, defaults shared with hpc/map_depth_sweep.sh.
GATE_VICTORIES = (2000, 4000, 10000)
GATE_SCALES = (1.0, 0.125)


def dataspace_to_csv(rows, path):
    import csv
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(DATASPACE_CSV_COLUMNS),
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


#: The column order of `results/tables/EyerissV1_mem_arch_sweep.csv`. The first
#: five are the ones the design decision is read off; the rest are what make a
#: row trustworthy (prompt_2, THE RESULTS TABLE).
LEVEL_CSV_COLUMNS = (
    "memory", "scale", "refetch", "weights_held", "level_pJ",
    "arm", "declared_depth", "width", "datawidth", "weights_per_word",
    "room", "fill_pct", "dram_weight_reads", "total_uJ", "pes_used",
    "recon_minus_embedded_pJ", "verdict", "fingerprint",
    # provenance beyond the required set
    "kind", "arch", "layer", "shape", "pes_declared", "cycles",
    "level_reads", "level_fills", "vector_access_pJ",
    "packing_discount_pJ", "refetch_term_pJ", "mapping", "depth_pass",
    "run_fingerprint",
)


def level_table_to_csv(csv_rows, path):
    """Write the level table in prompt_2's declared column order."""
    import csv
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(LEVEL_CSV_COLUMNS),
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(csv_rows)
    return path


def table_to_csv(csv_rows, path):
    import csv
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_rows:
        return path
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    return path


def to_csv(rows, path):
    import csv
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["arch", "layer", "shape", "ref_scale", "dil_scale",
            "ref_capacity", "dil_capacity", "ref_residency", "dil_residency",
            "weights", "ref_dram_reads", "dil_dram_reads", "reads_removed",
            "ref_refetch", "dil_refetch", "predicted_refetch", "refetch_ratio",
            "dram_pj_per_read", "ref_dram_weight_pJ",
            "fixed_mapping_saving_pJ", "task4_saving_pJ", "efficiency",
            "pct_of_layer_fixed", "pct_of_layer_task4", "ert_ratio",
            "ref_total_pJ", "dil_total_pJ"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r.arch, r.layer, r.shape, r.ref_scale, r.dil_scale,
                r.ref.capacity, r.dil.capacity, r.ref.residency, r.dil.residency,
                r.ref.weights, r.ref.dram_weight_reads, r.dil.dram_weight_reads,
                r.reads_removed, f"{r.ref.refetch:.6f}", f"{r.dil.refetch:.6f}",
                f"{r.predicted_refetch:.6f}", f"{r.refetch_ratio:.6f}",
                r.ref.dram_pj_per_read, f"{r.ref.dram_weight_pj:.3f}",
                f"{r.dram_pj_fixed_mapping:.3f}", f"{r.dram_pj_task4:.3f}",
                f"{r.efficiency:.6f}", f"{r.pct_fixed:.6f}", f"{r.pct_task4:.6f}",
                f"{r.ert_ratio:.6f}",
                f"{r.ref.total_pj_at(r.mac_pj):.3f}", f"{r.dil.total_pj_at(r.mac_pj):.3f}"])
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python3 -m eccenergy.report.dilation_view",
        description="Task 4 step 1: diff DRAM weight reads across weight-buffer "
                    "capacities. Reads caches only; maps nothing.")
    ap.add_argument("--survey", action="store_true",
                    help="per-layer refetch and buffer fill at ONE capacity, to "
                         "choose which layers are worth sweeping")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="the capacity --survey reports at (default 1.0)")
    ap.add_argument("--archs", default=None,
                    help="space-separated; default is the configured ECC_ARCHS")
    ap.add_argument("--layers", default=None,
                    help="space-separated layer names; default is ECC_LAYERS")
    ap.add_argument("--refs", default="1.0 0.5 0.25 0.125",
                    help="reference capacities; each is compared against ref*N/K")
    ap.add_argument("--table", action="store_true",
                    help="the UTILISATION table: one row per swept capacity per "
                         "design per layer, with PEs used, weights held, room, "
                         "fill, refetch, energy and the per-pair verdict")
    ap.add_argument("--scales", default=None,
                    help="--table only: the capacities to tabulate. Default is "
                         "every --refs value and its ref*N/K partner, so the "
                         "two arms of each pair sit on adjacent rows.")
    ap.add_argument("--levels", action="store_true",
                    help="PROMPT_2's results table: one row per (arm, scale, "
                         "MEMORY LEVEL), with per-level energy, geometry, "
                         "occupancy and the recon-minus-embedded margin. The "
                         "two arms are the SAME silicon at two datawidths, "
                         "not two depths -- --scales is the depth sweep")
    ap.add_argument("--recon-datawidth", type=int, default=None,
                    help="--levels only: the reconstruction arm's on-chip "
                         "weight datawidth in bits. Default is "
                         "widths.declared_datawidth() -- "
                         "round(ECC_WEIGHT_BITS * K/N), which is 4 at "
                         "BCH(63,30) and 7 at BCH(63,57). THE WIDTH TABLE "
                         "(eccenergy/widths.py) gives one per code")
    ap.add_argument("--depth-levels", default=None,
                    help="--levels only: read the SECOND PASS instead of the "
                         "joint one -- the caches where only these weight "
                         "level(s) had their depth scaled and the others were "
                         "held at x1. Space-separated. This is what says "
                         "WHICH memory bought the margin, and the deliverable "
                         "is a depth per level")
    ap.add_argument("--gate", action="store_true",
                    help="PROMPT_2's CONVERGENCE GATE: the embedded arm's "
                         "total energy at ECC_DEPTH_SWEEP_GATE_VICTORIES on "
                         "the largest and smallest depth, with the residual "
                         "between budgets set against the effect being "
                         "claimed. Run it BEFORE quoting any number")
    ap.add_argument("--victories", default=None,
                    help="--gate only: the budgets to compare. Default is "
                         "ECC_DEPTH_SWEEP_GATE_VICTORIES")
    ap.add_argument("--csv", default=None, help="also write the rows here")
    a = ap.parse_args(argv)

    cfg = configmod.load_config()
    arch_list = a.archs.split() if a.archs else list(cfg.archs)
    mac_pj = cfg.mac_pj_override

    if a.survey:
        print(survey(cfg, arch_list, a.scale, mac_pj=mac_pj))
        return 0

    layer_names = (a.layers or " ".join(cfg.layers)).split()
    if not layer_names:
        raise guards.refusal("dilation-needs-layers",
            "dilation: name the layers with --layers or ECC_LAYERS "
            "(this is a spot check by construction -- see --survey)")
    k_over_n = cfg.code_k / cfg.code_n

    if a.gate:
        # prompt_2's convergence gate: the budgets the EMBEDDED arm is mapped
        # at, re-checked at the largest AND the smallest depth. Script-local
        # defaults since 2026-09-14 (they were ECC_DEPTH_SWEEP_GATE_*), the
        # same ones hpc/map_depth_sweep.sh submits; --victories/--scales win.
        vs = ([int(float(x)) for x in a.victories.split()] if a.victories
              else list(GATE_VICTORIES))
        scales = ([float(s) for s in a.scales.split()] if a.scales
                  else list(GATE_SCALES))
        dw = a.recon_datawidth
        if dw is None:
            dw = widths.declared_datawidth(cfg.code_n, cfg.code_k,
                                                cfg.weight_bits)
        text, rows = convergence_gate(cfg, arch_list, layer_names, vs, scales,
                                      dw, mac_pj=mac_pj)
        print(text)
        if a.csv:
            print(f"\n  csv -> {table_to_csv(rows, a.csv)}")
        return 0

    if a.levels:
        # THE DEPTH LADDER. prompt_2 reports x0.5 / x0.25 / x0.125 -- the
        # realistic design choices a YAML quotes -- but SEARCHES on sqrt(2)
        # steps, because the window where Embedded cannot hold the tile and
        # Recon can is exactly as wide, in depth, as the effective-capacity
        # ratio: a factor-2 grid steps clean over a 1.17x or 1.33x window and
        # reports a grid artifact as "no effect".
        scales = ([float(s) for s in a.scales.split()] if a.scales
                  else [1.0, 0.7071, 0.5, 0.3536, 0.25, 0.1768, 0.125])
        dw = a.recon_datawidth
        if dw is None:
            dw = widths.declared_datawidth(cfg.code_n, cfg.code_k,
                                                cfg.weight_bits)
        dl = (a.depth_levels.split() if a.depth_levels
              else list(cfg.weight_depth_levels))
        text, csv_rows = level_table(cfg, arch_list, layer_names, scales, dw,
                                     mac_pj=mac_pj, depth_levels=dl)
        print(text)
        # THE PER-DATASPACE VIEW IS NOT OPTIONAL. On this design the weight
        # rows above sit at their refetch floor on both arms and show only the
        # flat packing discount; 97 % of the margin is in the Outputs row.
        dtext, drows = dataspace_table(cfg, arch_list, layer_names, scales, dw,
                                       mac_pj=mac_pj, depth_levels=dl)
        print()
        print(dtext)
        if a.csv:
            print(f"\n  csv -> {level_table_to_csv(csv_rows, a.csv)}")
            dpath = pathlib.Path(a.csv)
            dpath = dpath.with_name(dpath.stem + "__dataspace" + dpath.suffix)
            print(f"  csv -> {dataspace_to_csv(drows, dpath)}")
        return 0

    if a.table:
        refs = [float(s) for s in a.refs.split()]
        if a.scales:
            scales = [float(s) for s in a.scales.split()]
        else:
            scales = []
            for s in refs:
                scales += [round(s, 4), round(s / k_over_n, 4)]
            seen = set()
            scales = [s for s in scales if not (s in seen or seen.add(s))]
        text, csv_rows = utilisation_table(cfg, arch_list, layer_names, scales,
                                           k_over_n=k_over_n, mac_pj=mac_pj)
        print(text)
        if a.csv:
            print(f"\n  csv -> {table_to_csv(csv_rows, a.csv)}")
        return 0

    rows, gaps = sweep(cfg, arch_list, layer_names,
                       [float(s) for s in a.refs.split()],
                       k_over_n=k_over_n, mac_pj=mac_pj)
    if not rows:
        print("dilation: nothing cached yet for this sweep.")
        for g in gaps:
            print("   ", g)
        return 1
    print(render(rows, gaps, cfg, k_over_n, mac_pj))
    if a.csv:
        print(f"\n  csv -> {to_csv(rows, a.csv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
