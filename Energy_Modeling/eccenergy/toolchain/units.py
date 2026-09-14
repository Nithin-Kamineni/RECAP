"""THE UNITS OF MAPPER WORK one run needs, and which of them are already cached.

`hpc/run_all.sh` is the ONE launcher since EnvReorganisation phase 3
(2026-09-14), and this is the half of it that cannot live in a shell script:
the ARMS are derived (`arch.arms.mapper_arms()` -- the reference plus every
boundary that is a DISTINCT CHIP to the mapper), the SHAPES come out of the
workload file, and whether a unit is CACHED is `toolchain.invoke.Mapper`'s own
answer and no other.

    A CHIP  = (architecture, code K, buffer-depth scale, arm)
              -- every distinct thing the mapper is handed, with its own
                 `arch_fingerprint()` and its own cache directory
    A UNIT  = one chip x one distinct layer SHAPE
              -- one Timeloop search, one cache entry, one SLURM task's work

Three launchers used to exist because they enumerated chips along different
axes: `run_all.sh` over (architecture x model), `map_ert_arms.sh` over
(arm x shape) and `map_depth_sweep.sh` over (depth x datawidth x shape). There
is one enumerator now, and `ECC_SWEEP` says which axis it walks:

    bch     the codes of ECC_KS            arch, model, depth held
    model   the networks of ECC_MODELS     arch, code,  depth held
    arch    the designs of ECC_ARCHS       model, code, depth held
    fix     nothing -- the point itself; the arms ARE the comparison
    area    the depth ladder ECC_DEPTH_SWEEP_SCALES (passed as `--depths`,
            because the ladder is a shell list in env.sh section 1 and not a
            field of the configuration)

ONE CONFIGURATION PER MODEL, derived here. The LAYER SCOPE is per model
(`ECC_LAYERS`' `model=a b;` spelling, EnvReorganisation 6.9) and one resolved
configuration carries one scope, so a two-model run becomes two
configurations and each one resolves its own scope and its own shapes. That
is why `layers-per-model-one-model` is refused by `Session.setup()`, where
the scope is CONSUMED, and not by `config._resolve()`, which this depends on
resolving.

WHY `Mapper._accept_cached` AND NOT A FILE COUNT. An entry is a hit only if it
PROVES it matches: `timeloop-mapper.stats.txt` present AND a `mapping.json`
sidecar whose architecture fingerprint, ERT bump and per-MAC price are this
run's. Counting `stats.txt` files (which the old `--progress` flags did) calls
a mapping of another geometry a hit, which is the failure the sidecar exists
to prevent. `ECC_RERUN_OPTIMISER=1` makes every unit uncached, as it does
inside the mapper.
"""
from __future__ import annotations

import argparse
import sys

from .. import paths
from ..arch import arms as arms_mod
from ..contracts.errors import ConfigError
from ..arch import fingerprint as fingerprint_mod
from ..arch import workloads
from ..settings.run import RECON_PLACEMENT_APPROACHES
from . import invoke


class Unit:
    """One chip x one layer shape: one Timeloop search, one cache entry."""

    __slots__ = ("arch", "model", "code_k", "depth", "arm", "layer", "shape",
                 "cached", "why", "cache_dir")

    def __init__(self, arch, model, code_k, depth, arm, layer, shape,
                 cached, why, cache_dir):
        self.arch, self.model, self.code_k, self.depth = arch, model, code_k, depth
        self.arm, self.layer, self.shape = arm, layer, shape
        self.cached, self.why, self.cache_dir = cached, why, cache_dir

    @property
    def row(self):
        """The task-file row, WITHOUT its bundle column (the caller assigns it).

        `%g` on the depth so `1.0` is written `1` and reads back as the same
        float through `ECC_WEIGHT_DEPTH_SCALE` -- `settings.arch` rounds it to
        four places, so the text and the value cannot drift.
        """
        return (self.arch, self.model, str(self.code_k), f"{self.depth:g}",
                self.arm, self.layer)

    def __repr__(self):
        return (f"<{self.arch} {self.model} K{self.code_k} d{self.depth:g} "
                f"{self.arm} {self.shape} "
                f"{'cached' if self.cached else 'COLD'}>")


def axis_values(cfg, depths=None):
    """`(archs, code Ks, depth scales)` for `cfg.sweep` -- one model each way.

    The two POINT_SWEEPS (`fix`, `area`) hold all three lists, and env.sh has
    already collapsed `ECC_SWEEP_*` onto the first entry of each, so `cfg`
    itself is the point. `area` is the one axis whose list is not a field of
    the configuration, so it arrives as `depths`.
    """
    archs = list(cfg.archs)
    ks = list(cfg.sweep_ks) if cfg.sweep == "bch" else [cfg.code_k]
    scales = [round(float(d), 4) for d in (depths or [])] or [cfg.weight_depth_scale]
    return archs, ks, scales


def arms_for(cfg, arch):
    """The chips to map on `arch`: `reference` plus the placements in scope.

    THE ARMS ARE DERIVED, never listed: `mapper_arms()` is the reference plus
    every boundary that differs on `datawidth: q` x the ERT bump x the
    declared per-dataspace bandwidth scale (prompt_7 6.4). Two boundaries that
    agree on all three ARE one chip and share one arm, and this returns that
    arm once.

    Which boundaries are IN SCOPE is `ECC_APPROACHES` through
    `Config.recon_placements_for()`: the abstract `recon` means every one the
    design declares, a `reconN` name selects a subset. `reference` is always
    mapped -- every figure draws the two reference bars, and a placement whose
    own chip is cold is billed from a named plan that has to exist.

    A design that declares NO placement at all (no `placements.yaml`) has one
    chip: the reference. That is not an error here -- five of the eight
    declared designs are in exactly that state. Neither is an ECC_APPROACHES
    that names no reconstruction bar at all (`baseline embedded`): that run
    compares two arms that share one chip, so the reference is all there is to
    map, and mapping five boundaries nobody asked to draw would be five waves
    of compute for a figure that has no bar for them.
    """
    if not any(a in RECON_PLACEMENT_APPROACHES or a == "recon"
               for a in cfg.approaches):
        return ["reference"]
    try:
        wanted = cfg.recon_placements_for(arch, warn=False)
        arms = arms_mod.mapper_arms(arch, cfg)
    except KeyError:
        return ["reference"]
    out = []
    for a in arms:
        if a.placement is None:
            out.append(a.key)
            continue
        if wanted and not (a.key in wanted or a.placement.variant in wanted
                           or any(m in wanted for m in a.members)):
            continue
        out.append(a.key)
    return out


def shapes_for(cfg, model):
    """`[(layer name, shape name)]` -- ONE representative layer per SHAPE.

    THE MAPPER CACHE IS KEYED BY SHAPE, so resnet18's 21 layers are 12 pieces
    of work: four share `C64_M64_R3_S3_P56_Q56_ws1_hs1` and three share each
    of three more. Submitting per layer NAME would map the same shape four
    times and take the other three as cache hits after waiting on a lock.

    `cfg.layers` is the resolved scope -- empty for the whole model, or the
    named layers (per model, if `ECC_LAYERS` used the `model=a b;` spelling).
    A named list is deduplicated by shape too: naming two layers of one shape
    is one unit of work, and the second job would only wait for the first.
    """
    loaded, _ = workloads.load_workload(cfg)
    layers = workloads.select_layers(
        workloads.select(loaded, [model]), cfg.layers, verbose=False)[model]
    seen, out = set(), []
    for l in layers:
        if l.shape_name in seen:
            continue
        seen.add(l.shape_name)
        out.append((l.name, l.shape_name))
    return out


def _chip_cfg(cfg, arch, model, code_k, depth, arm):
    """The configuration ONE chip is mapped under -- and the one a job runs.

    Every axis is pinned to a single value, which is what makes this a CHIP:
    `experiment="map"` because a map job evaluates nothing and draws nothing,
    `sweep="arch"` over a one-name list because `recon_ert_arm` refuses more
    than one architecture (`ert-arm-one-arch` -- an arm is a property of one
    design's weight path), and one model so the LAYER SCOPE resolves (a
    per-model `ECC_LAYERS` table gives each network its own).

    `hpc/map.sbatch` exports exactly these knobs per unit, so the
    configuration checked here and the one the job runs are the same object
    built the same way. If that ever drifts, the launcher skips units the job
    would re-solve, or submits ones it already has.
    """
    return cfg.with_(experiment="map", sweep="arch",
                     sweep_archs=[arch], const_arch=arch,
                     const_model=model, sweep_models=[model], panel_models=[],
                     const_k=code_k, sweep_ks=[code_k],
                     weight_depth_scale=round(float(depth), 4),
                     recon_ert_arm=arm)


def enumerate_units(cfg, depths=None):
    """Every unit this configuration needs, cached ones included and marked.

    One `Mapper` per CHIP, not per unit: the fingerprint, the cache directory
    and the ERT bump are the chip's, and only `_accept_cached` is per shape.

    The models come from `cfg.models`, which is the axis resolved: the swept
    list on a model sweep, every PANEL model on a two-model architecture
    sweep, and the held model otherwise. Each gets its own configuration, so
    each gets its own layer scope and its own shapes.
    """
    archs, ks, scales = axis_values(cfg, depths)
    units = []
    for model in cfg.models:
        for arch in archs:
            for code_k in ks:
                for depth in scales:
                    ref = _chip_cfg(cfg, arch, model, code_k, depth, "reference")
                    for arm in arms_for(ref, arch):
                        # A CHIP THE DESIGN CANNOT EXPRESS IS SKIPPED, NOT
                        # FATAL. `simple_weight_stationary`'s recon5 narrows a
                        # `weight_reg` level its YAML does not declare, and
                        # patching refuses -- a known, recorded refusal
                        # (restructure/README.md lists the six REFUSED
                        # fingerprint rows). It is a chip no submission could
                        # map, so a launcher that died on it would make the
                        # whole architecture axis unusable; the reason is
                        # printed instead, once per chip.
                        try:
                            ccfg = (ref if arm == "reference" else
                                    _chip_cfg(cfg, arch, model, code_k, depth, arm))
                            variant = fingerprint_mod.effective_variant(arch, ccfg)
                            fp = fingerprint_mod.arch_fingerprint(arch, ccfg)
                            root = paths.Results(ccfg).mapper_cache(
                                arch, variant, fp, create=False)
                            mapper = invoke.Mapper(
                                ccfg, arch, None, root, None, fingerprint=fp,
                                ert_bump=fingerprint_mod.ert_bump(arch, ccfg))
                            entries = shapes_for(ccfg, model)
                        except (ValueError, KeyError, ConfigError) as exc:
                            print(f"  [skip] {arch}/{arm} at K{code_k} "
                                  f"x{depth:g} cannot be built: {exc}",
                                  file=sys.stderr)
                            continue
                        for layer_name, shape in entries:
                            entry = root / shape
                            stats, why = mapper._accept_cached(entry, None)
                            cached = (stats is not None
                                      and not ccfg.rerun_optimiser)
                            if ccfg.rerun_optimiser:
                                why = "ECC_RERUN_OPTIMISER=1: re-solving"
                            units.append(Unit(arch, model, code_k, depth, arm,
                                              layer_name, shape, cached, why,
                                              str(entry)))
    return units


def bundle(units, jobs=None):
    """`[[unit, ...]]` -- the SLURM array tasks, one list per task.

    `ECC_JOBS` empty is one unit per task, which is what every run before
    EnvReorganisation phase 3 did and what `ECC_CONCURRENCY` caps. A number
    BUNDLES the units into that many tasks, ROUND-ROBIN so no bundle collects
    all the big layers -- the units are enumerated arch-major, arm-major,
    shape-minor, so consecutive units are consecutive shapes of one chip and
    a contiguous split would put a model's whole expensive tail in one job.

    A BUNDLE IS SERIAL: `ECC_MAP_TIME` has to cover the bundle, not one map.
    """
    if not jobs or jobs < 1 or jobs >= len(units):
        return [[u] for u in units]
    out = [[] for _ in range(jobs)]
    for i, u in enumerate(units):
        out[i % jobs].append(u)
    return [b for b in out if b]


# ------------------------------------------------------------------------ CLI
def _print_tasks(bundles):
    for i, b in enumerate(bundles):
        for u in b:
            print(" ".join((str(i),) + u.row))


def _print_bill(cfg, units, bundles, depths):
    archs, ks, scales = axis_values(cfg, depths)
    chips = {(u.arch, u.code_k, u.depth, u.arm) for u in units}
    shapes = {(u.model, u.shape) for u in units}
    cold = [u for u in units if not u.cached]
    print(f"  axis      : {cfg.sweep} ({cfg.swept_axis})")
    print(f"  models    : {' '.join(cfg.models)}   "
          f"scope {cfg.layer_scope}"
          + (f": {', '.join(cfg.layers)}" if cfg.layers else "")
          + ("   (PER MODEL: "
             + "; ".join(f"{m}={' '.join(v) or 'whole'}"
                         for m, v in cfg.layers_by_model.items()) + ")"
             if cfg.layers_by_model else ""))
    print(f"  archs     : {' '.join(archs)}")
    print(f"  codes     : {' '.join('BCH(%d,%d)' % (cfg.code_n, k) for k in ks)}")
    print(f"  depths    : {' '.join('x%g' % d for d in scales)}")
    print(f"  arms      : {' '.join(sorted({u.arm for u in units}))}")
    print(f"  chips     : {len(chips)}   (architecture x code x depth x arm)")
    print(f"  shapes    : {len(shapes)}   (distinct layer shapes in scope)")
    print(f"  units     : {len(units)}   cached {len(units) - len(cold)} / "
          f"{len(units)}  ->  {len(cold)} to map")
    # The jobs a SUBMISSION would be: the COLD units, bundled. A cached unit
    # is not submitted at all.
    cold_bundles = bundle(cold, None if len(bundles) == len(units)
                          else len(bundles))
    print(f"  jobs      : {len(cold_bundles)}"
          + ("" if not cold_bundles or len(cold_bundles) == len(cold)
             else f"   (ECC_JOBS bundling: up to "
                  f"{max(len(b) for b in cold_bundles)} units per job, SERIAL)"))
    if cold:
        print("  COLD units (these are the ones a submission solves):")
        for u in cold[:40]:
            print(f"    {u.arch:24s} {u.model:<16s} {u.arm:<10s} "
                  f"K{u.code_k:<3d} x{u.depth:<6g} {u.shape:<34s} {u.why}")
        if len(cold) > 40:
            print(f"    ... and {len(cold) - 40} more")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python3 -m eccenergy.toolchain.units",
        description="the units of mapper work this configuration needs")
    p.add_argument("--tasks", action="store_true",
                   help="print the task file (one row per COLD unit) and nothing else")
    p.add_argument("--all", action="store_true",
                   help="with --tasks: emit CACHED units too. For a smoke test "
                        "-- a cached unit costs a cache hit and proves the job "
                        "script runs; a submission normally skips them")
    p.add_argument("--depths", default="",
                   help="ECC_SWEEP=area's depth ladder (ECC_DEPTH_SWEEP_SCALES); "
                        "empty = the held ECC_WEIGHT_DEPTH_SCALE")
    p.add_argument("--jobs", default="",
                   help="ECC_JOBS: bundle the units into this many SLURM tasks")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    from ..config import load_config
    cfg = load_config()
    depths = [float(d) for d in args.depths.replace(",", " ").split()]
    jobs = int(args.jobs) if args.jobs.strip() else None
    units = enumerate_units(cfg, depths)
    # THE TASK FILE IS THE WORK, not the inventory: a unit the mapper cache
    # already holds is not submitted, so a rerun after a failure queues only
    # what is left and a warm matrix submits nothing at all. The BILL shows
    # both counts, because "cached 72 / 72" is the answer to "is this run
    # going to cost anything".
    todo = units if args.all else [u for u in units if not u.cached]
    bundles = bundle(todo, jobs)
    if args.tasks:
        _print_tasks(bundles)
    else:
        _print_bill(cfg, units, bundle(units, jobs), depths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
