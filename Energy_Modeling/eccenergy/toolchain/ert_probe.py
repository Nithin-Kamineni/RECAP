"""prompt_6 PHASE 1 -- prove the ERT hook.

    bash hpc/tl.sh python3 -m eccenergy.toolchain.ert_probe --plan        # preconditions +
                                                                           # predictions only
    bash hpc/tl.sh python3 -m eccenergy.toolchain.ert_probe --parse-only  # + build every input
                                                                           # file, invoke nothing
    sbatch ... bash hpc/tl.sh python3 -m eccenergy.toolchain.ert_probe    # the five runs

THE QUESTION. prompt_6 puts the encoder's energy into the mapper's objective by
bumping actions of the Energy Reference Table. That only works if a table handed
to Timeloop is USED -- not recomputed by Accelergy, not silently ignored.
Statically it should be: the timeloopfe v4 `Specification` declares top-level
`ERT` and `ART` attributes, its v4->v3 transpiler passes them through, and both
the `timeloop-mapper` and `timeloop-model` binaries carry the branch
"Found Accelergy ERT ... replacing internal energy model" beside the
"Generate Accelergy ERT" branch every cached run took. This module MEASURES it.

FIVE RUNS, EACH WITH A NUMBER PREDICTED BEFORE IT RUNS. The reference cache
entry supplies the base table, the area table, the solved mapping, the exact v3
input the mapper consumed, and the stats everything is predicted from.

  0 accelergy   the base ERT regenerated from the patched arch must equal the
                cached one. Without this an ERT arm (PHASE 3) has no base table
                to patch, because no cache entry exists for a new arm yet.
  1 control     timeloop-model on the cached mapper's OWN processed input plus
                its mapping, ERT and ART, all unchanged. The log must say "Found
                Accelergy ERT", Accelergy must not have run, and every level
                energy, every leakage total and the cycle count must reproduce
                the cached stats. A supplied table is honoured.
  2 read        same, `<read level>.read` = READ_BUMP_PJ. That level's Weights
                energy must equal (reads x bump + fills x write) / block_size
                from the cached counts; nothing else may move.
  3 leak        same, `.leak` += LEAK_BUMP_PJ on every storage stage of the
                weight path. Each level's `Leakage energy (total)` must equal
                leak x instances x cycles -- which also measures the multiplier
                RULE 3 asks to verify (the instance count, not a power-gated
                subset); nothing else may move.
  4 mapper      timeloop-mapper THROUGH timeloopfe -- the path `timeloop.Mapper`
                uses, so this is the production hook itself -- with the
                read-bumped ERT and ART as two extra input YAMLs, no mapping, at
                the configured search settings. The table must survive the
                spec round-trip into the processed input, the log must say
                "Found Accelergy ERT", and the level's energy must reconcile
                with its OWN access counts at the bumped price. What nest it
                picks is a RESULT to report, not a pass criterion.

WHY 4 PRE-WRITES THE TABLES UNDER TIMELOOP'S OWN NAMES. With a supplied
table Timeloop writes no `timeloop-mapper.ERT.yaml` / `.ART.yaml`, and
timeloopfe's post-run parser then fails on the missing ART and `call_mapper`
raises AFTER a successful search (measured 2026-09-11, job 41734337).
`timeloop.Mapper._map_now` treats that exception as a failed shape. So the
production hook must write the supplied ERT and ART into the output directory
under those two names before the call; Timeloop leaves them alone when it takes
the "Found" branch, the parser is satisfied, and the cache entry then stores
exactly the table the mapper saw -- which is what RULE 4's read-back assertion
needs. Step 4 does this and asserts the files were NOT overwritten.

TIMELOOP-MODEL IS SILENT ABOUT THE TABLE. It prints four lines and neither
"Found Accelergy ERT" nor "Generate Accelergy ERT" (measured 2026-09-11, job
41734429), so for steps 1-3 the bypass is proven by the stats themselves --
`Vector access energy source : ERT` on the bumped level -- plus the absence of
any `.accelergy.log` or regenerated table. The mapper prints both lines and is
checked on them.

WHY 1-3 CALL THE BINARY DIRECTLY. The cached `timeloop-mapper.map.yaml` is in
Timeloop's v3 dialect (`type: datatype`, factors `C8`, targets naming the
transpiler's dummy `inter_*_spatial` levels). timeloopfe's v4 `mapping` list
refuses it, and translating it to v4 and back would put a second parser between
the control and the thing it controls. The mapper's own `parsed-processed-
input.yaml` IS the v3 input Timeloop read, so the model steps merge it with the
mapping and the tables into one file and hand that to `timeloop-model`.

NOTHING HERE TOUCHES THE MAPPER CACHE. The mapping fingerprint does not include
the ERT until PHASE 3, so a probe run through `run.sh map` would sit at the
reference fingerprint and be read back as the reference design. Everything is
written under `paths.ert_probe_dir()`.

The bumped levels are DERIVED from the design's weight path (`recon.stages_for`,
kind == "storage"), not spelled per design (prompt_6 3.3).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

import yaml

from ..arch import fingerprint
from ..arch import layout
from .. import paths as pathsmod
from ..arch import placements
from . import ert
from . import inputs as inputs_mod
from . import stats as stats_mod
from ..config import load_config
from ..arch.workloads import load_workload, select, select_layers
from ..settings import guards

#: Relative tolerance for every reconciliation. Stats print to 0.01 pJ, so a
#: number in the pJ range reconciles far tighter than this.
REL_TOL = 1e-6

#: Absurd on purpose: ~360x today's filter_glb.read (2.75566 pJ). A bump this
#: size cannot be mistaken for search noise.
READ_BUMP_PJ = 1000.0
#: ADDED to each bumped level's `leak`, pJ per instance per cycle.
LEAK_BUMP_PJ = 1000.0

FOUND = "Found Accelergy ERT"
GENERATED = "Generate Accelergy ERT"

PROCESSED_INPUT = "parsed-processed-input.yaml"


def _backend_calls():
    try:
        from pytimeloop.timeloopfe.common import backend_calls
    except ImportError:  # older container images
        from timeloopfe.common import backend_calls
    return backend_calls


# --------------------------------------------------------------------- tables
# The table helpers moved into `timeloop.py` for PHASE 3 (the production hook
# uses them); this module keeps its names as aliases.
_level_of = ert.ert_level_of
_ert_prices = ert.ert_prices
_art_areas = ert.art_areas


def _patched_ert(doc, changes):
    try:
        return ert.patched_ert(doc, changes)
    except ValueError as exc:
        raise guards.refusal("ert-probe-failed",
            f"ert_probe: {exc}")


def _write_yaml(path, doc):
    return ert.write_yaml(path, doc)


# ---------------------------------------------------------------------- stats
def _grab(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


# `parse_levels` moved into timeloop.py for PHASE 6 (the per-arm checks use
# it); the name is kept here.
parse_levels = stats_mod.parse_levels


def _close(a, b, rel=REL_TOL, abs_=1e-9):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


# ------------------------------------------------------------------- the runs
class Probe:
    def __init__(self, cfg, no_mapper=False):
        self.cfg = cfg
        self.no_mapper = no_mapper
        self.checks = []            # (step, name, ok, detail)
        self.result = {"checks": []}

    def check(self, step, name, ok, detail=""):
        self.checks.append((step, name, bool(ok), detail))
        self.result["checks"].append(dict(step=step, name=name, ok=bool(ok), detail=detail))
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""),
              flush=True)
        return bool(ok)

    # ---------------------------------------------------------- preconditions
    def setup(self):
        cfg = self.cfg
        self.arch = cfg.archs[0]
        self.variant = fingerprint.effective_variant(self.arch, cfg)
        self.fp = fingerprint.arch_fingerprint(self.arch, cfg)
        models = select(load_workload(cfg)[0], cfg.models)
        models = select_layers(models, cfg.layers, verbose=False)
        self.model = cfg.models[0]
        self.layer = models[self.model][0]
        shape = self.layer.shape_name
        cache = pathsmod.Results(cfg).mapper_cache(self.arch, self.variant, self.fp, create=False)
        self.ref = cache / shape
        need = ["timeloop-mapper.stats.txt", "timeloop-mapper.ERT.yaml",
                "timeloop-mapper.ART.yaml", "timeloop-mapper.map.yaml",
                "timeloop-mapper.map.txt", PROCESSED_INPUT, inputs_mod.MAPPING_SIDECAR]
        missing = [n for n in need if not (self.ref / n).exists()]
        if missing:
            raise guards.refusal("ert-probe-reference-incomplete",
                f"ert_probe: reference cache entry {self.ref} lacks {missing}.\n"
                f"  Map the reference arm first (prompt_3's four exports set).")
        side = json.loads((self.ref / inputs_mod.MAPPING_SIDECAR).read_text())
        if side.get("arch_fingerprint") != self.fp:
            raise guards.refusal("ert-probe-fingerprint-moved",
                f"ert_probe: sidecar fingerprint {side.get('arch_fingerprint')} "
                f"!= current {self.fp}")
        self.out = pathsmod.ert_probe_dir(self.arch, self.fp, shape)

        # The bumped levels come from the weight path, not from a design name.
        storage = [s for s in placements.stages_for(self.arch, cfg) if s.kind == "storage"]
        if not storage:
            raise guards.refusal("ert-probe-no-storage-stage",
                f"ert_probe: {self.arch} has no storage stage on its weight path")
        self.read_level = storage[0].prefixes[0]
        self.leak_levels = [s.prefixes[0] for s in storage]

        self.ref_levels, self.ref_summary = parse_levels(self.ref / "timeloop-mapper.stats.txt")
        for lv in [self.read_level] + self.leak_levels:
            if lv not in self.ref_levels or "Weights" not in self.ref_levels[lv]["ds"]:
                raise guards.refusal("ert-probe-level-no-weights",
                    f"ert_probe: level {lv!r} carries no Weights in the reference stats")
        self.ert_doc = yaml.safe_load((self.ref / "timeloop-mapper.ERT.yaml").read_text())
        self.art_doc = yaml.safe_load((self.ref / "timeloop-mapper.ART.yaml").read_text())
        self.prices = _ert_prices(self.ert_doc)

        print(f"ert_probe: {self.arch}  variant {self.variant}  fp {self.fp}")
        print(f"  layer {self.layer.name} = {shape}")
        print(f"  reference entry  {self.ref}")
        print(f"  probe output     {self.out}")
        print(f"  read bump        {self.read_level}.read = {READ_BUMP_PJ:g} pJ "
              f"(today {self.prices[(self.read_level, 'read')]:g})")
        print(f"  leak bump        +{LEAK_BUMP_PJ:g} pJ/instance/cycle on "
              f"{', '.join(self.leak_levels)}")
        self.result.update(arch=self.arch, variant=self.variant, fingerprint=self.fp,
                           layer=self.layer.name, shape=shape, reference=str(self.ref),
                           read_level=self.read_level, leak_levels=self.leak_levels,
                           read_bump_pJ=READ_BUMP_PJ, leak_bump_pJ=LEAK_BUMP_PJ,
                           tool_versions=inputs_mod.tool_versions())
        return self

    def predictions(self):
        """Every number step 2 and 3 must hit, from the cached stats alone."""
        cyc = self.ref_summary["cycles"]
        rl = self.ref_levels[self.read_level]
        w = rl["ds"]["Weights"]
        for lv in dict.fromkeys([self.read_level] + self.leak_levels):
            u = self.ref_levels[lv]["ds"]["Weights"]["updates"]
            self.check("pre", f"{lv}: Scalar updates for Weights == 0 (RULE 2)", u == 0,
                       f"updates={u:g}")
        bs = rl["block_size"]
        read_pj = (w["reads"] * READ_BUMP_PJ
                   + w["fills"] * self.prices[(self.read_level, "write")]
                   + w["updates"] * self.prices[(self.read_level, "update")]) / bs
        ref_pj = (w["reads"] * self.prices[(self.read_level, "read")]
                  + w["fills"] * self.prices[(self.read_level, "write")]
                  + w["updates"] * self.prices[(self.read_level, "update")]) / bs
        self.check("pre", f"{self.read_level}: cached energy reconciles with cached ERT "
                   f"at block_size {bs}", _close(ref_pj, w["energy_pJ"]),
                   f"predicted {ref_pj:.2f} pJ, printed {w['energy_pJ']:.2f} pJ")
        leak_pred = {}
        for lv in self.leak_levels:
            r = self.ref_levels[lv]
            base = self.prices[(lv, "leak")]
            ref_leak = base * r["instances"] * cyc
            self.check("pre", f"{lv}: cached leakage == leak x {r['instances']} instances x "
                       f"{cyc} cycles", _close(ref_leak, r["leakage_pJ"], rel=1e-3),
                       f"predicted {ref_leak:.2f} pJ, printed {r['leakage_pJ']:.2f} pJ")
            leak_pred[lv] = (base + LEAK_BUMP_PJ) * r["instances"] * cyc
        self.pred = dict(cycles=cyc, read_level_pJ=read_pj, read_level_ref_pJ=ref_pj,
                         read_delta_uJ=(read_pj - ref_pj) / 1e6, leak_pJ=leak_pred,
                         ref_energy_uJ=self.ref_summary["energy_uJ"])
        self.result["predictions"] = self.pred
        print(f"  PREDICTIONS  cycles {cyc}")
        print(f"    step 2: {self.read_level} Weights energy {ref_pj:,.2f} -> {read_pj:,.2f} pJ "
              f"(total +{self.pred['read_delta_uJ']:.4f} uJ on {self.ref_summary['energy_uJ']} uJ)")
        for lv, v in leak_pred.items():
            print(f"    step 3: {lv} leakage {self.ref_levels[lv]['leakage_pJ']:,.2f} -> {v:,.2f} pJ")

    # --------------------------------------------------------------- inputs
    def _tables(self):
        """The three tables every step draws on, written once under the probe dir."""
        self.ert_base = _write_yaml(self.out / "ert_base.yaml", _patched_ert(self.ert_doc, {}))
        self.ert_read = _write_yaml(self.out / "ert_read_bump.yaml", _patched_ert(
            self.ert_doc, {(self.read_level, "read"): ("set", READ_BUMP_PJ)}))
        self.ert_leak = _write_yaml(self.out / "ert_leak_bump.yaml", _patched_ert(
            self.ert_doc, {(lv, "leak"): ("add", LEAK_BUMP_PJ) for lv in self.leak_levels}))
        self.art = _write_yaml(self.out / "art.yaml", self.art_doc)

    def _spec_inputs(self):
        """The YAMLs `timeloop.Mapper` hands timeloopfe, for this arch and layer."""
        layout.write_globals(self.cfg, self.arch)
        arch_yaml = layout.patched_arch_path(self.arch, self.cfg)
        problem = inputs_mod.problem_path(self.layer)
        return [pathlib.Path(p)
                for p in inputs_mod.design_inputs(arch_yaml, problem, self.arch, self.cfg)]

    def _model_input(self, sub, ert_path):
        """ONE v3 YAML for timeloop-model: the cached mapper's processed input
        (identical architecture bytes), its mapping, and the supplied tables."""
        d = self.out / sub
        d.mkdir(parents=True, exist_ok=True)
        doc = yaml.safe_load((self.ref / PROCESSED_INPUT).read_text())
        doc["mapping"] = yaml.safe_load(
            (self.ref / "timeloop-mapper.map.yaml").read_text())["mapping"]
        doc["ERT"] = yaml.safe_load(pathlib.Path(ert_path).read_text())["ERT"]
        doc["ART"] = self.art_doc["ART"]
        # timeloop-model re-checks a supplied mapping against
        # `architecture_constraints` and refuses the mapper's OWN output with
        # "mapping violates architecture constraints" (measured 2026-09-11, job
        # 41734337). Constraints only prune a search; they change nothing about
        # evaluating a fixed mapping, so the model input carries none. The
        # mapper-only keys go with them.
        for k in ("architecture_constraints", "mapper", "mapspace"):
            doc.pop(k, None)
        return d, _write_yaml(d / "model-input.yaml", doc)

    # --------------------------------------------------------------- running
    def _clean(self, d):
        for p in list(d.glob("timeloop-*")) + list(d.glob("*_console.log")):
            p.unlink()                      # a stale file must not fake a result

    def _run_model(self, sub, ert_path):
        d, inp = self._model_input(sub, ert_path)
        self._clean(d)
        log = d / "model_console.log"
        t0 = time.time()
        with open(log, "w") as fh:
            rc = subprocess.run(["timeloop-model", str(inp)], cwd=str(d),
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        err = None if rc == 0 else f"timeloop-model exit {rc}"
        print(f"  {sub}: timeloop-model finished in {dt:.1f}s" + (f"  ({err})" if err else ""),
              flush=True)
        return d, log, dt, err

    def _run_spec(self, kind, sub, inputs, mapper_setup=None, clean=True):
        tl = inputs_mod.load_timeloopfe()
        d = self.out / sub
        d.mkdir(parents=True, exist_ok=True)
        if clean:
            self._clean(d)
        spec = tl.Specification.from_yaml_files(*[str(p) for p in inputs])
        if mapper_setup:
            mapper_setup(spec)
        log = d / f"{kind}_console.log"
        t0, err = time.time(), None
        try:
            fn = {"mapper": tl.call_mapper, "accelergy": tl.call_accelergy_verbose}[kind]
            fn(spec, output_dir=str(d), log_to=str(log))
        except Exception as exc:            # timeloopfe's own output parser can
            err = repr(exc)                 # fail after Timeloop succeeded; the
        dt = time.time() - t0               # stats file is what is checked below
        print(f"  {sub}: {kind} finished in {dt:.1f}s" + (f"  ({err})" if err else ""), flush=True)
        return d, log, dt, err

    def _bypass_checks(self, step, d, log, prefix, pre_written=None, verbose_binary=True):
        text = log.read_text(errors="replace") if log.exists() else ""
        if verbose_binary:
            self.check(step, f"log says '{FOUND}'", FOUND in text)
            self.check(step, "log says 'Found Accelergy ART'", "Found Accelergy ART" in text)
        else:
            print(f"    [note] {prefix} prints no 'Found Accelergy ERT' line; the bypass is "
                  f"proven by the stats' energy source and the absence of an Accelergy log")
        self.check(step, f"log does NOT say '{GENERATED}'", GENERATED not in text)
        self.check(step, "Accelergy was not invoked (no .accelergy.log)",
                   not (d / f"{prefix}.accelergy.log").exists())
        ert_file = d / f"{prefix}.ERT.yaml"
        if pre_written is None:
            self.check(step, "no ERT was regenerated", not ert_file.exists())
        else:
            same = ert_file.exists() and _ert_prices(yaml.safe_load(ert_file.read_text())) == \
                _ert_prices(yaml.safe_load(pathlib.Path(pre_written).read_text()))
            self.check(step, f"pre-written {ert_file.name} still holds the supplied table "
                       f"(not overwritten by Accelergy)", same)

    def _same_except(self, step, got, ref, skip_energy=(), skip_leak=()):
        """Every level energy and leakage equal to `ref` except the named ones."""
        for lv, r in ref.items():
            g = got.get(lv)
            if g is None:
                self.check(step, f"{lv}: present in stats", False)
                continue
            for ds, rv in r["ds"].items():
                if (lv, ds) in skip_energy:
                    continue
                gv = g["ds"].get(ds, {})
                self.check(step, f"{lv}/{ds}: energy unchanged",
                           _close(gv.get("energy_pJ"), rv["energy_pJ"]),
                           f"{gv.get('energy_pJ')} vs {rv['energy_pJ']} pJ")
            if lv not in skip_leak and r["leakage_pJ"] is not None:
                self.check(step, f"{lv}: leakage unchanged",
                           _close(g["leakage_pJ"], r["leakage_pJ"]),
                           f"{g['leakage_pJ']} vs {r['leakage_pJ']} pJ")

    def _mapper_setup(self):
        cfg = self.cfg
        levels_n = layout.arch_levels(self.arch, cfg)
        threads = cfg.mapper_threads or os.cpu_count() or 4
        victory = cfg.victory_for(levels_n)

        def setup(spec):        # mirrors timeloop.Mapper._map_now
            spec.mapper.num_threads = threads
            spec.mapper.victory_condition = victory
            spec.mapper.timeout = cfg.mapper_timeout
            spec.mapper.algorithm = cfg.mapper_algorithm
            if cfg.mapper_search_size is not None:
                spec.mapper.search_size = cfg.mapper_search_size
            spec.mapper.max_permutations_per_if_visit = cfg.mapper_max_permutations
            spec.mapper.optimization_metrics = [cfg.opt_metric]
        return setup, victory, threads

    def _ert_in_processed_input(self, step, path):
        """The table must survive timeloopfe's spec round-trip byte-for-value."""
        if not self.check(step, "timeloopfe wrote the processed input", path.exists(), str(path)):
            return
        doc = yaml.safe_load(path.read_text())
        has = "ERT" in doc and "ART" in doc
        self.check(step, "processed input carries ERT and ART top-level keys", has,
                   f"keys: {sorted(doc)}")
        if has:
            got = _ert_prices(doc)
            want = _ert_prices(yaml.safe_load(self.ert_read.read_text()))
            same = set(got) == set(want) and all(_close(got[k], want[k], rel=1e-9) for k in want)
            self.check(step, "processed-input ERT == the bumped table, every row", same,
                       f"{self.read_level}.read = {got.get((self.read_level, 'read'))}")

    def parse_only(self):
        """Build every input file both paths will use; invoke nothing."""
        self._tables()
        print("\n== parse-only: model input (steps 1-3) ==")
        d, inp = self._model_input("1_control", self.ert_base)
        doc = yaml.safe_load(inp.read_text())
        self.check("parse", "model input has architecture, problem, mapping, ERT, ART",
                   all(k in doc for k in ("architecture", "problem", "mapping", "ERT", "ART")),
                   f"{len(doc['mapping'])} mapping entries, {len(doc['ERT']['tables'])} ERT tables")
        print("\n== parse-only: timeloopfe spec with ERT+ART extra inputs (step 4) ==")
        tl = inputs_mod.load_timeloopfe()
        inputs = self._spec_inputs() + [self.ert_read, self.art]
        spec = tl.Specification.from_yaml_files(*[str(p) for p in inputs])
        setup, victory, threads = self._mapper_setup()
        setup(spec)
        text = _backend_calls()._specification_to_yaml_string(spec, for_model=False)
        p = self.out / "4_mapper" / "parse-only-processed-input.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        self._ert_in_processed_input("parse", p)
        print(f"  mapper settings that would run: victory {victory}, {threads} threads, "
              f"{self.cfg.mapper_algorithm}, timeout {self.cfg.mapper_timeout}, "
              f"{self.cfg.opt_metric}")

    def run(self):
        self._tables()
        base = self._spec_inputs()

        # ---- 0 accelergy regenerates the same table -------------------------
        print("\n== step 0: accelergy on the patched arch ==")
        d, log, dt, err = self._run_spec("accelergy", "0_accelergy", base)
        erts = sorted(p for p in d.glob("*ERT.yaml") if "summary" not in p.name)
        arts = sorted(p for p in d.glob("*ART.yaml") if "summary" not in p.name)
        if self.check("0", "accelergy wrote an ERT and an ART", bool(erts) and bool(arts),
                      f"{[p.name for p in erts + arts]}"):
            got = _ert_prices(yaml.safe_load(erts[0].read_text()))
            same = set(got) == set(self.prices) and all(
                _close(got[k], self.prices[k], rel=1e-9) for k in self.prices)
            diffs = [f"{k}: {got.get(k)} vs {self.prices[k]}" for k in self.prices
                     if not _close(got.get(k), self.prices[k], rel=1e-9)]
            self.check("0", "regenerated ERT == cached ERT, every row", same,
                       f"{len(self.prices)} rows" + (f"; differs: {diffs[:4]}" if diffs else ""))
            ga = _art_areas(yaml.safe_load(arts[0].read_text()))
            ra = _art_areas(self.art_doc)
            self.check("0", "regenerated ART == cached ART", set(ga) == set(ra) and all(
                _close(ga[k], ra[k], rel=1e-9) for k in ra))
        self.result["step0"] = dict(seconds=dt, error=err)

        # ---- 1 control ------------------------------------------------------
        print("\n== step 1: control -- cached input + mapping + ERT + ART, unchanged, timeloop-model ==")
        d, log, dt, err = self._run_model("1_control", self.ert_base)
        self._bypass_checks("1", d, log, "timeloop-model", verbose_binary=False)
        stats = d / "timeloop-model.stats.txt"
        if self.check("1", "stats written", stats.exists(), str(stats)):
            lv, sm = parse_levels(stats)
            self.check("1", f"{self.read_level}: 'Vector access energy source' is ERT",
                       lv[self.read_level]["source"] == "ERT", str(lv[self.read_level]["source"]))
            self.check("1", "cycles reproduce", sm["cycles"] == self.ref_summary["cycles"],
                       f"{sm['cycles']} vs {self.ref_summary['cycles']}")
            self.check("1", "total energy reproduces (summary, uJ)",
                       _close(sm["energy_uJ"], self.ref_summary["energy_uJ"], rel=1e-4),
                       f"{sm['energy_uJ']} vs {self.ref_summary['energy_uJ']} uJ")
            self._same_except("1", lv, self.ref_levels)
        self.result["step1"] = dict(seconds=dt, error=err)

        # ---- 2 read bump ----------------------------------------------------
        print(f"\n== step 2: {self.read_level}.read = {READ_BUMP_PJ:g} pJ, same mapping ==")
        d, log, dt, err = self._run_model("2_read", self.ert_read)
        self._bypass_checks("2", d, log, "timeloop-model", verbose_binary=False)
        stats = d / "timeloop-model.stats.txt"
        if self.check("2", "stats written", stats.exists()):
            lv, sm = parse_levels(stats)
            self.check("2", f"{self.read_level}: 'Vector access energy source' is ERT",
                       lv[self.read_level]["source"] == "ERT", str(lv[self.read_level]["source"]))
            got = lv[self.read_level]["ds"]["Weights"]["energy_pJ"]
            self.check("2", f"{self.read_level}/Weights energy == predicted",
                       _close(got, self.pred["read_level_pJ"]),
                       f"got {got:,.2f}, predicted {self.pred['read_level_pJ']:,.2f} pJ")
            self.check("2", "cycles unchanged", sm["cycles"] == self.ref_summary["cycles"])
            want_uJ = self.ref_summary["energy_uJ"] + self.pred["read_delta_uJ"]
            self.check("2", "total energy moved by exactly the level delta",
                       _close(sm["energy_uJ"], want_uJ, rel=1e-4),
                       f"got {sm['energy_uJ']}, predicted {want_uJ:.2f} uJ")
            self._same_except("2", lv, self.ref_levels,
                              skip_energy={(self.read_level, "Weights")})
            self.result["step2"] = dict(seconds=dt, error=err, level_pJ=got,
                                        energy_uJ=sm["energy_uJ"])

        # ---- 3 leak bump ----------------------------------------------------
        print(f"\n== step 3: leak += {LEAK_BUMP_PJ:g} pJ/inst/cycle on "
              f"{', '.join(self.leak_levels)}, same mapping ==")
        d, log, dt, err = self._run_model("3_leak", self.ert_leak)
        self._bypass_checks("3", d, log, "timeloop-model", verbose_binary=False)
        stats = d / "timeloop-model.stats.txt"
        if self.check("3", "stats written", stats.exists()):
            lv, sm = parse_levels(stats)
            for l in self.leak_levels:
                self.check("3", f"{l}: 'Vector access energy source' is ERT",
                           lv[l]["source"] == "ERT", str(lv[l]["source"]))
            mult = {}
            for l in self.leak_levels:
                got = lv[l]["leakage_pJ"]
                self.check("3", f"{l}: leakage == (leak + bump) x instances x cycles",
                           _close(got, self.pred["leak_pJ"][l]),
                           f"got {got:,.2f}, predicted {self.pred['leak_pJ'][l]:,.2f} pJ")
                mult[l] = (got - self.ref_levels[l]["leakage_pJ"]) / (LEAK_BUMP_PJ * sm["cycles"])
                self.check("3", f"{l}: measured leak multiplier == instance count "
                           f"({self.ref_levels[l]['instances']})",
                           _close(mult[l], self.ref_levels[l]["instances"], rel=1e-6),
                           f"multiplier {mult[l]:.6f}, power-gating share printed "
                           f"{lv[l]['gating']}")
            self.check("3", "cycles unchanged", sm["cycles"] == self.ref_summary["cycles"])
            self._same_except("3", lv, self.ref_levels, skip_leak=set(self.leak_levels))
            self.result["step3"] = dict(seconds=dt, error=err, multipliers=mult,
                                        leakage_pJ={l: lv[l]["leakage_pJ"] for l in self.leak_levels},
                                        energy_uJ=sm["energy_uJ"])

        # ---- 4 the mapper, through timeloopfe -------------------------------
        if self.no_mapper:
            print("\n== step 4 skipped (--no-mapper) ==")
            return
        setup, victory, threads = self._mapper_setup()
        cfg = self.cfg
        print(f"\n== step 4: timeloop-mapper via timeloopfe with the read-bumped ERT, no mapping "
              f"(victory {victory}, {threads} threads, {cfg.mapper_algorithm}, "
              f"timeout {cfg.mapper_timeout}, {cfg.opt_metric}) ==")
        d4 = self.out / "4_mapper"
        d4.mkdir(parents=True, exist_ok=True)
        self._clean(d4)
        shutil.copyfile(self.ert_read, d4 / "timeloop-mapper.ERT.yaml")
        shutil.copyfile(self.art, d4 / "timeloop-mapper.ART.yaml")
        d, log, dt, err = self._run_spec("mapper", "4_mapper", base + [self.ert_read, self.art],
                                         setup, clean=False)
        self.check("4", "call_mapper returned without raising (timeloopfe's parser found "
                   "the pre-written ART)", err is None, err or "")
        self._ert_in_processed_input("4", d / PROCESSED_INPUT)
        self._bypass_checks("4", d, log, "timeloop-mapper", pre_written=self.ert_read)
        stats = d / "timeloop-mapper.stats.txt"
        if self.check("4", "stats written", stats.exists()):
            lv, sm = parse_levels(stats)
            r = lv[self.read_level]
            w = r["ds"]["Weights"]
            want = (w["reads"] * READ_BUMP_PJ
                    + w["fills"] * self.prices[(self.read_level, "write")]
                    + w["updates"] * self.prices[(self.read_level, "update")]) / r["block_size"]
            self.check("4", f"{self.read_level}/Weights energy reconciles with ITS OWN "
                       f"counts at the bumped price", _close(w["energy_pJ"], want),
                       f"got {w['energy_pJ']:,.2f}, from counts {want:,.2f} pJ "
                       f"(reads {w['reads']:g}, fills {w['fills']:g}, bs {r['block_size']})")
            ref_map = (self.ref / "timeloop-mapper.map.txt").read_text()
            got_map = (d / "timeloop-mapper.map.txt").read_text()
            rw = self.ref_levels[self.read_level]["ds"]["Weights"]
            print(f"  RESULT (not a pass criterion): nest "
                  f"{'IDENTICAL to' if got_map == ref_map else 'DIFFERENT from'} the reference")
            print(f"    cycles {sm['cycles']} (ref {self.ref_summary['cycles']}), "
                  f"utilization {sm['utilization']}% (ref {self.ref_summary['utilization']}%)")
            print(f"    {self.read_level} reads {w['reads']:g} (ref {rw['reads']:g}), "
                  f"fills {w['fills']:g} (ref {rw['fills']:g})")
            print(f"    total {sm['energy_uJ']} uJ (ref {self.ref_summary['energy_uJ']} uJ; "
                  f"the reference nest at the bumped price would be "
                  f"{self.ref_summary['energy_uJ'] + self.pred['read_delta_uJ']:.2f} uJ)")
            self.result["step4"] = dict(
                seconds=dt, error=err, nest_identical=(got_map == ref_map),
                cycles=sm["cycles"], utilization=sm["utilization"], energy_uJ=sm["energy_uJ"],
                read_level=dict(reads=w["reads"], fills=w["fills"], energy_pJ=w["energy_pJ"]),
                victory=victory, threads=threads)
            if got_map != ref_map:
                print("    --- reference nest ---"); print(ref_map)
                print("    --- bumped-ERT nest ---"); print(got_map)

    # --------------------------------------------------------------- report
    def report(self):
        fails = [c for c in self.checks if not c[2]]
        self.result["passed"] = len(self.checks) - len(fails)
        self.result["failed"] = len(fails)
        out = self.out / "probe_result.json"
        with open(out, "w", newline="\n") as fh:
            json.dump(self.result, fh, indent=1, default=str)
        print("\n" + "=" * 78)
        print(f"ert_probe: {self.result['passed']} checks passed, {len(fails)} failed  -> {out}")
        for step, name, _ok, detail in fails:
            print(f"  FAIL [{step}] {name}  {detail}")
        print("=" * 78)
        return not fails


def main(argv=None):
    ap = argparse.ArgumentParser(description="prompt_6 PHASE 1: prove the ERT hook")
    ap.add_argument("--plan", action="store_true",
                    help="preconditions and predictions only; invoke nothing")
    ap.add_argument("--parse-only", action="store_true",
                    help="also build every input file both paths use; invoke nothing")
    ap.add_argument("--no-mapper", action="store_true", help="skip step 4")
    args = ap.parse_args(argv)
    cfg = load_config()
    probe = Probe(cfg, no_mapper=args.no_mapper).setup()
    probe.predictions()
    if args.parse_only:
        probe.parse_only()
    if args.plan or args.parse_only:
        print("\nnothing invoked.")
        return 0 if all(c[2] for c in probe.checks) else 1
    inputs_mod.require_container()
    if not shutil.which("timeloop-model"):
        raise guards.refusal("timeloop-model-not-on-path",
            "ert_probe: timeloop-model is not on PATH")
    probe.run()
    return 0 if probe.report() else 1


if __name__ == "__main__":
    sys.exit(main())
