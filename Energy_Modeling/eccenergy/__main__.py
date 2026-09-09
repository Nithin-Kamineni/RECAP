"""Entry point: `python3 -m eccenergy` (which is what run.sh calls).

Reads ECC_* from the environment, prints the resolved configuration, runs the
sweep. Command-line flags are conveniences only -- every one of them has an
environment variable, so `run.sh` stays the single source of truth.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

from .config import ConfigError, EXPERIMENTS, SWEEPS, banner, load_config


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="python3 -m eccenergy",
        description="ECC energy modelling (Timeloop + Accelergy). "
                    "Configure with ECC_* environment variables; see run.sh.")
    p.add_argument("experiment", nargs="?", choices=EXPERIMENTS,
                   help="sweep (default) | baseline | embedded | recon | validate "
                        "| map | diagnose | panels")
    p.add_argument("--sweep", choices=SWEEPS,
                   help="override ECC_SWEEP: which axis goes on the x axis")
    p.add_argument("--replot", action="store_true",
                   help="ECC_REPLOT_ONLY=1: rebuild the figure from results/_raw/ only")
    p.add_argument("--arch", help="override ECC_CONST_ARCH (the held architecture)")
    p.add_argument("--model", help="override ECC_CONST_MODEL (the held model)")
    p.add_argument("--k", type=int, help="override ECC_CONST_K (the held BCH K)")
    p.add_argument("--values", help="override the swept list for the chosen sweep "
                                    "(space or comma separated)")
    p.add_argument("--layers",
                   help="override ECC_LAYERS: one or two STABLE layer names "
                        "(e.g. 'layer3.0.downsample.0 layer4.1.conv2'), or 'all'")
    p.add_argument("--eval", action="store_true",
                   help="ECC_FROM_CACHE=1: evaluate from mappings already solved, "
                        "never invoke the mapper. No container needed.")
    p.add_argument("--overwrite", action="store_true",
                   help="ECC_OVERWRITE=1: allow replacing an existing result file")
    p.add_argument("--phase", choices=("Pre", "Post"),
                   help="override ECC_PHASE: which half of the study this is")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve and print the configuration, then exit")
    return p.parse_args(argv)


def _apply_overrides(args):
    if args.experiment:
        os.environ["ECC_EXPERIMENT"] = args.experiment
    if args.sweep:
        os.environ["ECC_SWEEP"] = args.sweep
    if args.replot:
        os.environ["ECC_REPLOT_ONLY"] = "1"
    if args.eval:
        os.environ["ECC_FROM_CACHE"] = "1"
    if args.overwrite:
        os.environ["ECC_OVERWRITE"] = "1"
    if args.phase:
        os.environ["ECC_PHASE"] = args.phase
    if args.layers is not None:
        # `--layers all` is the readable way to ask for the full model back
        # after a shell has ECC_LAYERS exported.
        os.environ["ECC_LAYERS"] = ("" if args.layers.strip().lower() in ("all", "full")
                                    else args.layers)
    if args.arch:
        os.environ["ECC_CONST_ARCH"] = args.arch
    if args.model:
        os.environ["ECC_CONST_MODEL"] = args.model
    if args.k is not None:
        os.environ["ECC_CONST_K"] = str(args.k)
    if args.values:
        sweep = os.environ.get("ECC_SWEEP", "bch").strip().lower()
        var = {"bch": "ECC_SWEEP_KS", "k": "ECC_SWEEP_KS", "code": "ECC_SWEEP_KS",
               "model": "ECC_SWEEP_MODELS", "models": "ECC_SWEEP_MODELS",
               "arch": "ECC_SWEEP_ARCHS", "archs": "ECC_SWEEP_ARCHS",
               "architecture": "ECC_SWEEP_ARCHS"}.get(sweep)
        if not var:
            raise SystemExit(f"--values: unknown sweep {sweep!r}")
        os.environ[var] = args.values


def _map_only(cfg):
    """Generate and cache mappings for the selected layers, then stop.

    The plan asks for mapping generation to be separable from energy evaluation.
    This is that separation made explicit: it needs the container, it writes
    `mapping.json` sidecars into the mapper cache, and it produces no result
    file. Afterwards `bash run.sh baseline --eval` runs anywhere, with no
    container, and cannot change a mapping.
    """
    from .experiments.common import Session
    ses = Session(cfg).setup()
    ses.collect_all()
    print("\n" + "=" * 78)
    print("Mappings generated and cached. Nothing was evaluated.")
    for arch, fp in ses.fingerprints.items():
        print(f"  {arch:26s} fp={fp}  ->  {ses.results.mapper_cache(arch, None, fp)}")
    print("Evaluate them with:   bash run.sh baseline --eval")
    print("=" * 78)
    return ses


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _apply_overrides(args)

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"\nconfiguration error: {exc}\n", file=sys.stderr)
        print("Fix the offending variable in run.sh and try again.", file=sys.stderr)
        return 2

    # imported here so a config error never pays for matplotlib/pandas import
    from .ecc import load_recon_energy, recon_pj_for_k
    from .experiments import (baseline, diagnose, embedded, panels, recon, sweep,
                              validate)

    recon_pj, provenance = load_recon_energy(cfg)
    if cfg.sweep == "bch" and cfg.recon_pj_override is None:
        # one number per swept K, so the banner cannot imply the held K's
        # datapath was used for the whole sweep
        provenance = "  ".join(f"K{k}={recon_pj_for_k(cfg, k):.4f}" for k in cfg.sweep_ks)
    print(banner(cfg, recon_pj, provenance))

    if args.dry_run:
        print("\n--dry-run: configuration is valid; nothing executed.")
        return 0

    runners = {"sweep": sweep.run, "diagnose": diagnose.run, "panels": panels.run,
               "baseline": baseline.run, "validate": validate.run,
               # Task 2: the embedded-ECC arm beside the conventional baseline,
               # from the same cached mappings; only DRAM differs, and the
               # result file checks that rather than asserting it.
               "embedded": embedded.run,
               # Task 3: the reconstruction PLACEMENT study on ONE architecture,
               # from the same cached mappings; the boundary is the axis and the
               # mapper is never re-run.
               "recon": recon.run,
               # `map` is the mapping-generation half of the split the plan
               # requires: solve and cache the mappings for the selected layers,
               # write no result, evaluate nothing.
               "map": _map_only}
    try:
        runners[cfg.experiment](cfg)
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(f"\n{exc}", file=sys.stderr)
            return exc.code if isinstance(exc.code, int) else 1
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted. Every solved mapping is cached, so re-running resumes.")
        return 130
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
