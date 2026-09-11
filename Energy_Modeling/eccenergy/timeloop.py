"""The Timeloop mapper interface: problem YAMLs, cached mapping, stats parsing.

This is the only module that talks to Timeloop, and the only one that can be
slow. Every mapping is cached on disk under
`ecc_energy_study/outputs/<arch>/<subdir>/<shape>/`, keyed by the layer shape
alone -- so a run is resumable and two models that share a layer shape share a
mapping.
"""
from __future__ import annotations

import contextlib
import copy
import glob
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import time

import yaml

from .paths import (ARCH_COMPONENTS, DESIGNS_DIR, EX_REPO, EXERCISES_URL,
                    PROB_DIR, WORK)

#: Name of the sidecar written beside every mapping this code produces.
MAPPING_SIDECAR = "mapping.json"

#: prompt_6: the supplied energy table of an ERT arm, under the names Timeloop
#: itself uses. With a supplied table Timeloop writes neither, and timeloopfe's
#: post-run parser then raises on the missing ART after a SUCCESSFUL search
#: (FINDINGS 3.5); pre-writing them under these names is what keeps a mapped
#: shape from being counted as failed, and the cache entry then stores exactly
#: the table the mapper saw.
ERT_NAME = "timeloop-mapper.ERT.yaml"
ART_NAME = "timeloop-mapper.ART.yaml"
#: Per-arm tables, once per (architecture, arm, fingerprint), beside the
#: shape entries of that arm's cache directory.
ERT_DIR = "_ert"
ERT_RECORD = "ert_bump.json"
ERT_FOUND = "Found Accelergy ERT"
ERT_GENERATED = "Generate Accelergy ERT"

PROBLEM_TEMPLATE = """problem:
  version: 0.4
  shape:
    name: cnn_layer
    coefficients:
      - name: Wstride
        default: 1
      - name: Hstride
        default: 1
      - name: Wdilation
        default: 1
      - name: Hdilation
        default: 1
    dimensions: [ C, M, R, S, N, P, Q ]
    data_spaces:
      - name: Weights
        projection:
          - [ [C] ]
          - [ [M] ]
          - [ [R] ]
          - [ [S] ]
      - name: Inputs
        projection:
          - [ [N] ]
          - [ [C] ]
          - [ [R, Wdilation], [P, Wstride] ]
          - [ [S, Hdilation], [Q, Hstride] ]
      - name: Outputs
        projection:
          - [ [N] ]
          - [ [M] ]
          - [ [Q] ]
          - [ [P] ]
        read_write: True
  instance:
    C: {C}
    M: {M}
    R: {R}
    S: {S}
    N: 1
    P: {P}
    Q: {Q}
    Wstride: {Wstride}
    Hstride: {Hstride}
"""


#: The same shape with a group dimension G, for depthwise / grouped
#: convolutions. G indexes every dataspace: group g's weights see only group g's
#: input channels and produce only group g's outputs, which is exactly the
#: cross-channel independence a depthwise layer has and the plain template
#: cannot express. Used only when Layer.G > 1, so ungrouped layers keep their
#: byte-identical problem files and their cached mappings.
GROUPED_PROBLEM_TEMPLATE = """problem:
  version: 0.4
  shape:
    name: grouped_cnn_layer
    coefficients:
      - name: Wstride
        default: 1
      - name: Hstride
        default: 1
      - name: Wdilation
        default: 1
      - name: Hdilation
        default: 1
    dimensions: [ G, C, M, R, S, N, P, Q ]
    data_spaces:
      - name: Weights
        projection:
          - [ [G] ]
          - [ [C] ]
          - [ [M] ]
          - [ [R] ]
          - [ [S] ]
      - name: Inputs
        projection:
          - [ [N] ]
          - [ [G] ]
          - [ [C] ]
          - [ [R, Wdilation], [P, Wstride] ]
          - [ [S, Hdilation], [Q, Hstride] ]
      - name: Outputs
        projection:
          - [ [N] ]
          - [ [G] ]
          - [ [M] ]
          - [ [Q] ]
          - [ [P] ]
        read_write: True
  instance:
    G: {G}
    C: {C}
    M: {M}
    R: {R}
    S: {S}
    N: 1
    P: {P}
    Q: {Q}
    Wstride: {Wstride}
    Hstride: {Hstride}
"""


_TOOL_VERSIONS = {}


def tool_versions():
    """Versions of the tools that produced a mapping, for the result JSON.

    The plan asks for mappings to be cached on the TOOL VERSION among other
    things, and for results to record it. Timeloop itself has no `--version`
    (it aborts on any argv that is not a config file), so what is recorded is
    the Python packages that wrap it plus the interpreter -- enough to tell two
    container images apart, which is what actually changes between runs here.

    Read once per process; a failure is recorded as such rather than guessed.
    """
    if _TOOL_VERSIONS:
        return _TOOL_VERSIONS
    import sys
    versions = {"python": sys.version.split()[0],
                "timeloop_mapper_on_path": bool(shutil.which("timeloop-mapper")),
                "timeloop_mapper_version": "not reported by the binary "
                                           "(timeloop-mapper exposes no --version)"}
    for pkg in ("accelergy", "pytimeloop", "accelergy-cacti-plug-in",
                "accelergy-aladdin-plug-in", "accelergy-table-based-plug-ins"):
        try:
            from importlib import metadata
            versions[pkg] = metadata.version(pkg)
        except Exception as exc:
            versions[pkg] = f"unavailable ({type(exc).__name__})"
    _TOOL_VERSIONS.update(versions)
    return _TOOL_VERSIONS


def load_timeloopfe():
    try:
        import pytimeloop.timeloopfe.v4 as tl
    except ImportError:  # older container images
        import timeloopfe.v4 as tl
    return tl


def require_container():
    if not shutil.which("timeloop-mapper"):
        raise SystemExit(
            "timeloop-mapper is not on PATH.\n"
            "  -> run inside the container:\n"
            '     docker run -it --rm -v "<project>":/home/workspace \\\n'
            "        timeloopaccelergy/timeloop-accelergy-pytorch:latest-amd64 bash")


def ensure_exercises_repo():
    """Clone timeloop-accelergy-exercises if `ecc_energy_study/` was wiped."""
    if DESIGNS_DIR.exists():
        return False
    WORK.mkdir(parents=True, exist_ok=True)
    if not EX_REPO.exists():
        print(f"  [clone] {EXERCISES_URL} -> {EX_REPO}")
        subprocess.run(["git", "clone", "--depth", "1", EXERCISES_URL, str(EX_REPO)],
                       check=True)
    if not DESIGNS_DIR.exists():
        raise SystemExit(f"cloned the exercises repo but {DESIGNS_DIR} is missing")
    return True


def problem_path(layer):
    """Write (once) and return the problem YAML for this layer shape.

    Written atomically -- temp file plus rename -- because `problems/` is shared
    by every architecture, so running several architectures in parallel (one
    shell each, `ECC_SWEEP_ARCHS="<one>"`) has two processes racing to create
    the same file. The content is identical either way, but a non-atomic write
    lets one process read a half-written YAML. Each architecture has its own
    mapper-cache directory, so parallel runs are otherwise independent.
    """
    PROB_DIR.mkdir(parents=True, exist_ok=True)
    path = PROB_DIR / f"{layer.shape_name}.yaml"
    if not path.exists():
        if layer.G > 1:
            text = GROUPED_PROBLEM_TEMPLATE.format(
                G=layer.G, C=layer.C, M=layer.M, R=layer.R, S=layer.S,
                P=layer.P, Q=layer.Q, Wstride=layer.Wstride, Hstride=layer.Hstride)
        else:
            text = PROBLEM_TEMPLATE.format(
                C=layer.C, M=layer.M, R=layer.R, S=layer.S,
                P=layer.P, Q=layer.Q, Wstride=layer.Wstride, Hstride=layer.Hstride)
        tmp = path.with_suffix(f".yaml.{os.getpid()}.tmp")
        # explicit LF: on Windows, write_text() would emit CRLF, and this file is
        # read by Timeloop inside a Linux container
        tmp.write_text(text, newline="\n")
        os.replace(tmp, path)
    return path


def design_inputs(arch_yaml, problem_yaml):
    """Every YAML the mapper is handed, in order.

    Locally authored components come AFTER the cloned repo's, so this project
    can add a component the exercises repo does not have without editing the
    clone. Names must not collide -- a local file redefining an upstream class
    would be a duplicate-class error, not an override -- so the corrected
    register file is called `smartbuffer_RF_decoded`, not `smartbuffer_RF`.
    """
    return ([str(arch_yaml)]
            + sorted(glob.glob(str(DESIGNS_DIR / "_components" / "*.yaml")))
            + sorted(glob.glob(str(ARCH_COMPONENTS / "*.yaml")))
            + [str(DESIGNS_DIR / "_include" / "mapper.yaml"),
               str(WORK / "globals.yaml"),
               str(problem_yaml)])


# ----------------------------------------------------- prompt_6: ERT tables
def ert_level_of(table_name):
    """`system_top_level.filter_glb[1..1]` -> `filter_glb` (timeloopfe's rule)."""
    return re.sub(r"\[\d+\.\.\d+\]", "", table_name).split(".")[-1]


def ert_prices(doc):
    """`{(level, action): pJ}` for every row of an ERT document."""
    return {(ert_level_of(t["name"]), a["name"]): float(a["energy"])
            for t in doc["ERT"]["tables"] for a in t["actions"]}


def art_areas(doc):
    return {ert_level_of(t["name"]): float(t["area"]) for t in doc["ART"]["tables"]}


def patched_ert(doc, changes):
    """Copy `doc` with `changes` = {(level, action): ("set"|"add", pJ)} applied.

    `arguments:` is emptied on every action: timeloopfe's `Ert` node declares no
    argument keys (the DRAM rows carry two), and Timeloop reads only `name` and
    `energy`. A change that matches nothing is an error, never a silent no-op.
    """
    out = copy.deepcopy(doc)
    pending = dict(changes)
    for t in out["ERT"]["tables"]:
        level = ert_level_of(t["name"])
        for a in t["actions"]:
            a["arguments"] = {}
            key = (level, a["name"])
            if key in pending:
                how, val = pending.pop(key)
                a["energy"] = float(val) if how == "set" else float(a["energy"]) + float(val)
    if pending:
        raise ValueError(f"no ERT row for {sorted(pending)}; the table has "
                         f"{sorted(set(ert_prices(doc)))}")
    return out


def write_yaml(path, doc):
    """LF, atomic: a concurrent reader sees the old file or the new one."""
    path = pathlib.Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", newline="\n") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)
    os.replace(tmp, path)
    return path


def ert_changes(bump):
    """The two rows an arm patches, as `patched_ert` takes them (prompt_6 5.1)."""
    return {(bump["level"], bump["action"]): ("add", bump["access_delta_pj"]),
            (bump["level"], "leak"): ("add", bump["leak_delta_pj"])}


#: Tolerance for "the same delta": RULE 4.4.5 asks for 1e-9 of the intended value.
ERT_TOL = 1e-9


def _close(a, b, tol=ERT_TOL):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def same_bump(a, b):
    """Do two bump records describe one toll? Identity fields exactly, deltas to ERT_TOL."""
    if a is None or b is None:
        return a is None and b is None
    return (a.get("placement") == b.get("placement") and a.get("level") == b.get("level")
            and a.get("action") == b.get("action")
            and _close(a.get("access_delta_pj"), b.get("access_delta_pj"))
            and _close(a.get("leak_delta_pj"), b.get("leak_delta_pj")))


class ErtMismatch(SystemExit):
    """A cache entry's ERT is not the one the bar asking for it expects."""


def describe_bump(bump):
    if bump is None:
        return "reference (no ERT bump)"
    return (f"{bump['placement']}: {bump['level']}.{bump['action']} "
            f"+= {bump['access_delta_pj']:.9g} pJ, {bump['level']}.leak "
            f"+= {bump['leak_delta_pj']:.9g} pJ/instance/cycle")


class ErtTables:
    """The supplied ERT/ART of ONE ERT arm, generated once and reused per shape.

    Lives at `<out_root>/_ert/`, i.e. beside the shape entries of the arm's own
    cache directory (own slug, own fingerprint -- RULE 4.4.5). Holds:

        base.ERT.yaml / base.ART.yaml   Accelergy's own table for THIS arm's
                                        patched arch (deterministic, FINDINGS
                                        3.5, so it needs no cache entry to exist)
        timeloop-mapper.ERT.yaml        the base with the arm's two rows bumped
        timeloop-mapper.ART.yaml        the base ART, unchanged
        ert_bump.json                   the bump, the base and patched prices of
                                        the bumped rows, tool versions

    `ensure()` generates under a `ShapeLock`, so concurrent jobs of one arm
    build it once; a set already on disk is reused only if its record matches
    the requested bump.
    """

    def __init__(self, cfg, arch, arch_yaml, out_root, bump):
        if bump is None:
            raise ValueError("ErtTables is for an ERT arm; the reference arm has none")
        self.cfg, self.arch, self.arch_yaml, self.bump = cfg, arch, arch_yaml, bump
        self.dir = pathlib.Path(out_root) / ERT_DIR
        self.ert = self.dir / ERT_NAME
        self.art = self.dir / ART_NAME
        self.base_ert = self.dir / "base.ERT.yaml"
        self.base_art = self.dir / "base.ART.yaml"
        self.record_path = self.dir / ERT_RECORD
        self.record = None

    # ------------------------------------------------------------ validity
    def _load(self):
        """The record on disk if it describes THIS bump and the files agree."""
        for f in (self.ert, self.art, self.base_ert, self.base_art, self.record_path):
            if not f.exists():
                return None
        try:
            rec = json.loads(self.record_path.read_text())
            if not same_bump(rec.get("bump"), self.bump):
                return None
            prices = ert_prices(yaml.safe_load(self.ert.read_text()))
            for action, want in rec["patched_pj"].items():
                if not _close(prices.get((self.bump["level"], action)), want):
                    return None
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return rec

    def ensure(self, layer):
        """Generate the tables if they are not already on disk for this bump."""
        self.record = self._load()
        if self.record is not None:
            return self
        lock = ShapeLock(self.dir)
        lock.acquire(shape=f"{self.arch} ERT tables ({self.bump['placement']})")
        try:
            self.record = self._load()
            if self.record is None:
                self._generate(layer)
        finally:
            lock.release()
        return self

    def _generate(self, layer):
        """Accelergy on THIS arm's patched arch, then the two bumped rows."""
        self.dir.mkdir(parents=True, exist_ok=True)
        scratch = self.dir / f"accelergy.{os.getpid()}"
        scratch.mkdir(parents=True, exist_ok=True)
        tl = load_timeloopfe()
        spec = tl.Specification.from_yaml_files(
            *design_inputs(self.arch_yaml, problem_path(layer)))
        with open(scratch / "accelergy_console.log", "w") as logf, \
                contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
            tl.call_accelergy_verbose(spec, output_dir=str(scratch),
                                      log_to=str(scratch / "accelergy.log"))
        erts = sorted(p for p in scratch.glob("*ERT.yaml") if "summary" not in p.name)
        arts = sorted(p for p in scratch.glob("*ART.yaml") if "summary" not in p.name)
        if not erts or not arts:
            raise SystemExit(f"ErtTables: accelergy wrote no ERT/ART under {scratch}; "
                             f"see {scratch / 'accelergy_console.log'}")
        base_ert = yaml.safe_load(erts[0].read_text())
        base_art = yaml.safe_load(arts[0].read_text())
        prices = ert_prices(base_ert)
        level = self.bump["level"]
        for action in (self.bump["action"], "leak"):
            if (level, action) not in prices:
                raise SystemExit(f"ErtTables: the generated ERT has no row "
                                 f"{level}.{action}; it has "
                                 f"{sorted(k for k in prices if k[0] == level)}")
        patched = patched_ert(base_ert, ert_changes(self.bump))
        pprices = ert_prices(patched)
        record = {
            "bump": self.bump,
            "base_pj": {a: prices[(level, a)] for a in (self.bump["action"], "leak")},
            "patched_pj": {a: pprices[(level, a)] for a in (self.bump["action"], "leak")},
            "arch_yaml": str(self.arch_yaml),
            "tool_versions": tool_versions(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        write_yaml(self.base_ert, patched_ert(base_ert, {}))
        write_yaml(self.base_art, base_art)
        write_yaml(self.ert, patched)
        write_yaml(self.art, base_art)
        tmp = self.record_path.with_name(f".{ERT_RECORD}.{os.getpid()}.tmp")
        with open(tmp, "w", newline="\n") as fh:
            fh.write(json.dumps(record, indent=1, default=str))
        os.replace(tmp, self.record_path)
        shutil.rmtree(scratch, ignore_errors=True)
        self.record = record
        print(f"      [ert] {self.arch}: {describe_bump(self.bump)}  "
              f"(base {level}.{self.bump['action']} {record['base_pj'][self.bump['action']]:g} "
              f"-> {record['patched_pj'][self.bump['action']]:g} pJ, leak "
              f"{record['base_pj']['leak']:g} -> {record['patched_pj']['leak']:g})",
              flush=True)

    # --------------------------------------------------------------- use
    def stage(self, out_dir):
        """Pre-write the two tables into a shape's output directory."""
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.ert, out_dir / ERT_NAME)
        shutil.copyfile(self.art, out_dir / ART_NAME)

    def inputs(self):
        """The two extra YAMLs `design_inputs()` is followed by."""
        return [str(self.ert), str(self.art)]

    def sidecar_record(self):
        """What a mapping sidecar records about its ERT (RULE 4.4.5)."""
        rec = self.record or {}
        return {"bump": self.bump, "base_pj": rec.get("base_pj"),
                "patched_pj": rec.get("patched_pj")}


def read_back_ert(out_dir, bump, base_prices=None, tol=ERT_TOL):
    """RULE 4.4.5, defence 3: re-open the ERT stored beside a cache entry and
    assert it is the one the bar asking for it expects.

    `bump` is the requesting bar's `archs.ert_bump()` (None for a reference bar).
    Checks, and raises `ErtMismatch` naming both sides on the first failure:

    * the sidecar's recorded bump has the same placement, level and action, and
      each delta is within `tol` of the intended value;
    * the stored table's bumped rows equal the recorded base price plus the
      intended delta, to `tol`;
    * with `base_prices` (the un-bumped table, e.g. the reference entry's ERT),
      the recorded base equals it on the bumped rows and every other row is
      untouched.

    Returns a dict describing what was verified.
    """
    out_dir = pathlib.Path(out_dir)
    side = out_dir / MAPPING_SIDECAR
    if not side.exists():
        raise ErtMismatch(f"read_back_ert: {out_dir} has no {MAPPING_SIDECAR}")
    rec = json.loads(side.read_text()).get("ert_bump")
    if bump is None:
        if rec is not None:
            raise ErtMismatch(
                f"read_back_ert: {out_dir} was mapped as an ERT arm "
                f"({describe_bump(rec.get('bump'))}) but the reference arm asked for it")
        return {"arm": "reference", "verified": ["no ERT bump recorded"]}
    if rec is None or rec.get("bump") is None:
        raise ErtMismatch(
            f"read_back_ert: {out_dir} records no ERT bump, but "
            f"{describe_bump(bump)} asked for it")
    got = rec["bump"]
    if not same_bump(got, bump):
        raise ErtMismatch(
            f"read_back_ert: {out_dir} was mapped as\n    {describe_bump(got)}\n"
            f"  but the bar asking for it is\n    {describe_bump(bump)}")
    ert_path = out_dir / ERT_NAME
    if not ert_path.exists():
        raise ErtMismatch(f"read_back_ert: {ert_path} is missing")
    prices = ert_prices(yaml.safe_load(ert_path.read_text()))
    level = bump["level"]
    verified = []
    for action, delta in ((bump["action"], bump["access_delta_pj"]),
                          ("leak", bump["leak_delta_pj"])):
        stored = prices.get((level, action))
        base = (rec.get("base_pj") or {}).get(action)
        if stored is None or base is None:
            raise ErtMismatch(f"read_back_ert: {out_dir}: no stored/base price for "
                              f"{level}.{action} (stored {stored}, base {base})")
        if not _close(stored - base, delta, tol):
            raise ErtMismatch(
                f"read_back_ert: {out_dir}: {level}.{action} stored {stored!r} - base "
                f"{base!r} = {stored - base!r}, but {describe_bump(bump)} intends "
                f"{delta!r} (tolerance {tol:g})")
        if base_prices is not None and not _close(base, base_prices.get((level, action)), tol):
            raise ErtMismatch(
                f"read_back_ert: {out_dir}: recorded base {level}.{action} = {base!r} "
                f"but the un-bumped table says {base_prices.get((level, action))!r}")
        verified.append(f"{level}.{action}: {base:g} + {delta:.9g} = {stored:g}")
    if base_prices is not None:
        bumped = {(level, bump["action"]), (level, "leak")}
        for key, want in base_prices.items():
            if key in bumped:
                continue
            if not _close(prices.get(key), want, tol):
                raise ErtMismatch(
                    f"read_back_ert: {out_dir}: {key[0]}.{key[1]} = {prices.get(key)!r} "
                    f"differs from the un-bumped table's {want!r}; only "
                    f"{level}.{bump['action']} and {level}.leak may move")
        verified.append(f"{len(base_prices) - len(bumped)} other rows untouched")
    return {"arm": bump["placement"], "level": level, "action": bump["action"],
            "verified": verified}


class ShapeLock:
    """One mapper-cache entry, one writer: an atomic-mkdir lock beside out_dir.

    `os.mkdir` is atomic on every filesystem this project runs on, including
    Lustre and NFS where `flock` is not reliable -- which is why it is a
    directory and not a lock file. The directory holds an `owner` file (host,
    pid, time) so a lock left behind by a killed job can be recognised:

      * same host and the pid is gone     -> stale, taken over at once
      * any host, older than `stale_s`    -> stale, taken over (a shape has
                                             never taken 6 h; a job that did
                                             would have been killed by SLURM)
      * otherwise                         -> wait, polling every `poll_s`

    `acquire()` returns True if it had to wait, so the caller knows to re-check
    the cache: the task that held the lock has very probably just written the
    mapping this task was about to compute.
    """

    def __init__(self, out_dir, stale_s=6 * 3600, poll_s=20.0):
        self.out_dir = pathlib.Path(out_dir)
        self.lockdir = self.out_dir.parent / (self.out_dir.name + ".lock")
        self.stale_s = stale_s
        self.poll_s = poll_s
        self.held = False

    def _owner(self):
        try:
            host, pid, t0 = (self.lockdir / "owner").read_text().split()
            return host, int(pid), float(t0)
        except (OSError, ValueError):
            return None

    def _stale(self):
        o = self._owner()
        if o is None:                       # mid-creation or unreadable: age it
            try:
                return time.time() - self.lockdir.stat().st_mtime > self.stale_s
            except OSError:
                return False
        host, pid, t0 = o
        if host == os.uname().nodename:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass
        return time.time() - t0 > self.stale_s

    def acquire(self, shape=""):
        self.out_dir.parent.mkdir(parents=True, exist_ok=True)
        waited = False
        announced = False
        while True:
            try:
                os.mkdir(self.lockdir)
            except FileExistsError:
                if self._stale():
                    print(f"      [lock] stale lock on {shape or self.out_dir.name} "
                          f"(owner {self._owner()}); taking it over", flush=True)
                    shutil.rmtree(self.lockdir, ignore_errors=True)
                    continue
                if not announced:
                    print(f"      [lock] {shape or self.out_dir.name} is being mapped by "
                          f"another task ({self._owner()}); waiting", flush=True)
                    announced = True
                waited = True
                time.sleep(self.poll_s)
                continue
            (self.lockdir / "owner").write_text(
                f"{os.uname().nodename} {os.getpid()} {time.time():.0f}\n")
            self.held = True
            return waited

    def release(self):
        if self.held:
            shutil.rmtree(self.lockdir, ignore_errors=True)
            self.held = False


class Mapper:
    """Runs (and caches) one mapping per layer shape for one architecture.

    MAPPING GENERATION IS SEPARATE FROM ENERGY EVALUATION, which is what the
    plan requires: this class produces `timeloop-mapper.stats.txt` and a
    `mapping.json` sidecar, and nothing here knows what BCH(N,K) is. Everything
    downstream reads those files, so an evaluation-only re-run (`--eval`,
    `ECC_FROM_CACHE=1`) never invokes the mapper and cannot change a mapping.

    A CACHE ENTRY IS ONLY REUSED IF IT PROVES IT MATCHES. Each entry carries a
    sidecar naming the architecture fingerprint and mapper settings that
    produced it. A hit whose fingerprint differs is a MISS, not a hit -- the
    previous scheme keyed only on a treatment slug, so editing an arch.yaml
    silently reused mappings computed for the old geometry.
    """

    def __init__(self, cfg, arch, arch_yaml, out_root, levels=None,
                 fingerprint=None, legacy_root=None, ert_bump=None):
        self.cfg = cfg
        self.arch = arch
        self.arch_yaml = arch_yaml
        self.out_root = pathlib.Path(out_root)
        #: prompt_6: the ERT toll this arm maps under (`archs.ert_bump`), or
        #: None for the reference arm. Recorded in every sidecar, required
        #: to match on every cache hit, read back after every fresh map.
        self.ert_bump = ert_bump
        self._ert_tables = None
        # Mapper effort is per architecture: a deeper loop nest needs more of
        # it to be searched as thoroughly. See Config.victory_for().
        self.levels = levels
        self.victory = cfg.victory_for(levels)
        self.fingerprint = fingerprint
        self.legacy_root = pathlib.Path(legacy_root) if legacy_root else None
        self._memo = {}
        self.mappings = {}      # shape_name -> mapping record for the result JSON
        #: Shapes this process has already re-solved under ECC_RERUN_OPTIMISER.
        #: The point of that flag is ONE fresh mapping per shape, not one per
        #: time a layer of that shape is reached -- resnet18 alone would
        #: otherwise re-map a shared shape several times over.
        self._remapped = set()
        self.n_cached = 0
        self.n_mapped = 0
        self.n_legacy = 0
        self.n_failed = 0
        self.map_seconds = 0.0
        self.failures = []

    # ------------------------------------------------------------- identity
    def mapping_id(self, layer):
        """A stable identifier for one (architecture, settings, shape) mapping.

        Results reference mappings by this, so a `Pre` result set can be
        checked -- mechanically, not by trust -- to have used the SAME mapping
        for every ECC variant, which the storage spec requires.
        """
        blob = f"{self.arch}|{self.fingerprint}|{layer.shape_name}|vic{self.victory}"
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def _sidecar(self, out_dir, layer, source):
        return {
            "mapping_id": self.mapping_id(layer),
            "architecture": self.arch,
            "arch_fingerprint": self.fingerprint,
            "arch_yaml": str(self.arch_yaml) if self.arch_yaml else None,
            "layer_shape": layer.shape_name,
            "loop_levels": self.levels,
            "victory_condition_effective": self.victory,
            "mapper": self.cfg.mapper_settings(),
            "tool_versions": tool_versions(),
            "stats": str(out_dir / "timeloop-mapper.stats.txt"),
            "source": source,
            "ert_bump": self._ert_record(),
        }

    def _ert_record(self):
        if self.ert_bump is None:
            return None
        if self._ert_tables is not None:
            return self._ert_tables.sidecar_record()
        return {"bump": self.ert_bump, "base_pj": None, "patched_pj": None}

    def _accept_cached(self, out_dir, layer):
        """Decide whether an existing cache entry may be reused, and say why not.

        Returns (stats_path or None, provenance string).
        """
        stats = out_dir / "timeloop-mapper.stats.txt"
        if not stats.exists():
            return None, "no stats.txt"
        side = out_dir / MAPPING_SIDECAR
        if not side.exists():
            if self.cfg.cache_strict:
                return None, ("no mapping.json sidecar: nothing proves this mapping "
                              "was computed for the current architecture YAML")
            self.n_legacy += 1
            return stats, "legacy-unfingerprinted (ECC_CACHE_STRICT=0)"
        try:
            rec = json.loads(side.read_text())
        except Exception as exc:
            return None, f"unreadable sidecar ({exc})"
        if self.fingerprint and rec.get("arch_fingerprint") != self.fingerprint:
            return None, (f"sidecar fingerprint {rec.get('arch_fingerprint')} != "
                          f"{self.fingerprint}: the architecture or the mapper "
                          f"settings changed since this mapping was computed")
        # prompt_6 RULE 4.4.5: the toll is in the fingerprint, but say it in
        # words too -- a wrong pick here leaves no other trace.
        theirs = (rec.get("ert_bump") or {}).get("bump")
        if not same_bump(theirs, self.ert_bump):
            return None, (f"sidecar ERT bump is {describe_bump(theirs)} but this "
                          f"arm is {describe_bump(self.ert_bump)}")
        return stats, "cached"

    def stats_for(self, layer):
        """Return the stats.txt path for `layer`, mapping it if necessary."""
        sig = layer.sig
        if sig in self._memo:
            return self._memo[sig]

        out_dir = self.out_root / layer.shape_name
        # ECC_RERUN_OPTIMISER=1: solve this shape again even though the cache
        # entry is valid, and overwrite it. Nothing about the entry is wrong --
        # this is for refreshing a mapping after a change the fingerprint does
        # not capture, or for checking that a mapping reproduces.
        if self.cfg.rerun_optimiser and layer.shape_name not in self._remapped:
            self._remapped.add(layer.shape_name)
            print(f"      ECC_RERUN_OPTIMISER=1: re-solving {layer.shape_name} "
                  f"and overwriting its cache entry", flush=True)
            lock = ShapeLock(out_dir)
            lock.acquire(shape=layer.shape_name)
            try:
                return self._map_now(layer, out_dir, sig)
            finally:
                lock.release()

        stats, why = self._accept_cached(out_dir, layer)
        if stats is None and self.legacy_root is not None and not self.cfg.cache_strict:
            legacy_dir = self.legacy_root / layer.shape_name
            stats, why = self._accept_cached(legacy_dir, layer)
            if stats is not None:
                out_dir = legacy_dir
        if stats is not None:
            self.n_cached += 1
            self._memo[sig] = stats
            self.mappings[layer.shape_name] = dict(
                self._sidecar(out_dir, layer, why), cached=True)
            return stats

        if self.cfg.from_cache:
            # ECC_FROM_CACHE=1: use what the mapper has already solved and never
            # invoke Timeloop. Lets raw energies be re-derived from the cache on
            # a machine with no container.
            self._fail(layer, out_dir,
                       f"not usable from the mapper cache ({why}), and "
                       f"ECC_FROM_CACHE=1 forbids mapping it")
            return None

        # Two SLURM tasks for the same architecture may reach the same shape at
        # the same time (resnet18 and resnet50 share 5 shapes, mobilenet_v2 and
        # efficientnet_b0 share 11). Timeloop writes a dozen files into out_dir
        # non-atomically, so two mappers in one directory corrupt the entry.
        # The lock makes the second task wait, then take the first task's
        # mapping as a cache hit -- identical result, and the work is done once.
        lock = ShapeLock(out_dir)
        waited = lock.acquire(shape=layer.shape_name)
        try:
            if waited:
                stats, why = self._accept_cached(out_dir, layer)
                if stats is not None:
                    self.n_cached += 1
                    self._memo[sig] = stats
                    self.mappings[layer.shape_name] = dict(
                        self._sidecar(out_dir, layer, why + " (mapped by a concurrent task)"),
                        cached=True)
                    return stats
            return self._map_now(layer, out_dir, sig)
        finally:
            lock.release()

    def _map_now(self, layer, out_dir, sig):
        """Run Timeloop for `layer` into `out_dir`; caller holds the ShapeLock."""
        threads = self.cfg.mapper_threads or os.cpu_count() or 4
        # Say what this shape is going to cost BEFORE it costs it. Without the
        # thread count and the effective victory on the line, "this is slow"
        # cannot be told apart from "this is oversubscribed" -- and under Docker
        # os.cpu_count() reports the HOST's cores, not the container's limit.
        print(f"      mapping {layer.shape_name} "
              f"(victory {self.victory}, {threads} threads) ...", flush=True)
        t0 = time.time()
        tl = load_timeloopfe()
        inputs = design_inputs(self.arch_yaml, problem_path(layer))
        if self.ert_bump is not None:
            # prompt_6: the arm's table, generated once per (arch, arm,
            # fingerprint), pre-written under Timeloop's own names (FINDINGS
            # 3.5 fact 1) and handed to timeloopfe as two extra inputs.
            if self._ert_tables is None:
                self._ert_tables = ErtTables(self.cfg, self.arch, self.arch_yaml,
                                             self.out_root, self.ert_bump)
            self._ert_tables.ensure(layer)
            self._ert_tables.stage(out_dir)
            inputs = inputs + self._ert_tables.inputs()
        spec = tl.Specification.from_yaml_files(*inputs)
        spec.mapper.num_threads = threads
        spec.mapper.victory_condition = self.victory
        spec.mapper.timeout = self.cfg.mapper_timeout
        spec.mapper.algorithm = self.cfg.mapper_algorithm
        # DEVELOPMENT SPEED KNOBS. Both bound the search directly rather than
        # by the victory heuristic, and both are in the mapping fingerprint AND
        # the cache slug, so a bounded run can never be read back as a full one.
        #
        #   search_size          hard cap on valid mappings per thread. None =
        #                        uncapped, which is the publishable setting.
        #   max_permutations...  how many loop permutations the hybrid search
        #                        tries per index factorization (Timeloop's
        #                        default is 16). Lowering it makes the search
        #                        move through FACTORIZATIONS -- the choice that
        #                        sets DRAM traffic -- proportionally faster.
        if self.cfg.mapper_search_size is not None:
            spec.mapper.search_size = self.cfg.mapper_search_size
        spec.mapper.max_permutations_per_if_visit = self.cfg.mapper_max_permutations
        # The shipped _include/mapper.yaml asks for `edp`. This is an ENERGY
        # study, and EDP does not merely add noise -- it is biased by array
        # width, because an architecture with more MACs can trade energy for
        # latency and EDP rewards that. Overriding here rather than editing the
        # cloned exercises repo keeps the clone pristine and re-clonable.
        spec.mapper.optimization_metrics = [self.cfg.opt_metric]
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(out_dir / "mapper_console.log", "w") as logf, \
                    contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
                tl.call_mapper(spec, output_dir=str(out_dir))
        except Exception as exc:  # a shape the mapper cannot fit
            self._fail(layer, out_dir, repr(exc))
            return None

        dt = time.time() - t0
        self.map_seconds += dt
        # `stats` is None here: _accept_cached() returned None to say the entry
        # could not be reused, which is why we mapped. The freshly written file
        # is the one to look for now.
        stats = out_dir / "timeloop-mapper.stats.txt"
        if stats.exists():
            self.n_mapped += 1
            self._memo[sig] = stats
            # The sidecar is what makes this entry reusable later. Written with
            # the ACTUAL thread count, not the configured one, because leaving
            # ECC_MAPPER_THREADS empty means "all cores" and that differs per
            # machine -- a result has to record which it got.
            record = self._sidecar(out_dir, layer, "newly mapped")
            record["mapper"] = dict(record["mapper"], num_threads_effective=threads)
            record["map_seconds"] = round(dt, 3)
            with open(out_dir / MAPPING_SIDECAR, "w", newline="\n") as fh:
                fh.write(json.dumps(record, indent=1, default=str))
            if self.ert_bump is not None:
                # The table must have been USED (not regenerated) and must
                # still be the one staged: RULE 4.4.5's read-back, on the
                # entry that was just written.
                log = (out_dir / "mapper_console.log").read_text(errors="replace")
                if ERT_FOUND not in log or ERT_GENERATED in log:
                    self.n_mapped -= 1
                    self._fail(layer, out_dir, f"Timeloop did not take the '{ERT_FOUND}' "
                               f"branch; the supplied ERT was not used")
                    return None
                try:
                    read_back_ert(out_dir, self.ert_bump)
                except ErtMismatch as exc:
                    self.n_mapped -= 1
                    self._fail(layer, out_dir, str(exc))
                    return None
            self.mappings[layer.shape_name] = dict(record, cached=False)
            print(f"        {dt:7.1f}s   (this architecture: {self.n_mapped} mapped, "
                  f"{self.map_seconds / 60:.1f} min so far)", flush=True)
        else:
            self._fail(layer, out_dir, "mapper produced no stats.txt")
        return self._memo[sig]

    def _fail(self, layer, out_dir, reason):
        self.n_failed += 1
        self.failures.append((layer.shape_name, reason, self._diagnose(out_dir)))
        self._memo[layer.sig] = None

    @staticmethod
    def _diagnose(out_dir):
        """Pull the actual complaint out of the mapper's console log.

        Timeloop aborts the process on a bad specification, so the Python
        exception says nothing useful; the reason is only in the log. Without
        this, a whole-architecture failure reports 21 identical "mapper failed"
        lines and no cause.
        """
        log = pathlib.Path(out_dir) / "mapper_console.log"
        if not log.exists():
            return ""
        lines = [ln.strip() for ln in log.read_text(errors="replace").splitlines()]
        for ln in reversed(lines):
            if any(k in ln for k in ("ERROR", "Assertion", "error:", "terminate called",
                                     "what():", "Aborted")):
                return ln
        return lines[-1] if lines else ""

    def summary(self):
        effort = f"{self.victory}"
        if self.levels:
            effort += f" ({self.levels} loop levels)"
        msg = (f"{self.arch}: {self.n_cached} cached, {self.n_mapped} newly mapped, "
               f"{self.n_failed} failed  [objective={self.cfg.opt_metric} "
               f"victory={effort} fp={self.fingerprint}]")
        if self.n_legacy:
            msg += (f"\n    {self.n_legacy} entr(ies) reused from the "
                    f"pre-fingerprint cache with ECC_CACHE_STRICT=0 -- nothing "
                    f"proves they match the current arch.yaml. Recorded in the "
                    f"result JSON as mapping_provenance=legacy.")
        if not self.failures:
            return msg
        # Group by cause: one bad specification fails every shape identically,
        # and that is the case worth calling out loudly.
        causes = {}
        for shape, reason, detail in self.failures:
            causes.setdefault(detail or reason, []).append(shape)
        lines = [msg]
        for cause, shapes in causes.items():
            lines.append(f"    {len(shapes)} shape(s): {cause}")
            lines.append(f"      e.g. {shapes[0]}")

        if self.cfg.from_cache:
            # Nothing was attempted, so nothing is broken: the cache for this
            # architecture treatment is simply empty.
            lines.append(f"    -> nothing is wrong with {self.arch}; this treatment "
                         f"has no mapper cache yet.")
            lines.append(f"       Run it in the container without ECC_FROM_CACHE to "
                         f"build one:  bash run.sh")
            lines.append(f"       cache dir: {self.out_root}")
        elif self.n_mapped == 0 and self.n_cached == 0:
            lines.append(f"    -> EVERY shape failed on {self.arch}. One bad "
                         f"architecture specification fails them all identically, "
                         f"so suspect the arch.yaml before the layers.")
            lines.append(f"       full log: {self.out_root / self.failures[0][0]}"
                         f"/mapper_console.log")
        return "\n".join(lines)


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

        {level: {instances, block_size, word_bits, leakage_pJ, gating, source,
                 ds: {Weights|Inputs|Outputs|Compute: {reads, fills, updates,
                      energy_pJ}}}}, summary {energy_uJ, cycles, utilization}

    Per-instance counts, as Timeloop prints them. `source` is the `Vector
    access energy source` (ERT when a supplied table was billed). Written for
    `experiments/ert_probe` (phase 1) and used by `experiments/recon`'s
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
            leakage_pJ=_grab_as(r"Leakage energy \(total\)\s*:\s*([\d.eE+-]+)\s*pJ", body),
            gating=_grab_as(r"Instances sharing power gating\s*:\s*([\d.eE+-]+)", body),
            source=_grab_as(r"Vector access energy source\s*:\s*(\S+)", body, str),
            utilized_instances=_grab_as(r"Utilized instances(?: \(max\))?\s*:\s*(\d+)", body, int),
            cycles=_grab_as(r"Cycles\s*:\s*(\d+)", body, int),
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
