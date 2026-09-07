"""The Timeloop mapper interface: problem YAMLs, cached mapping, stats parsing.

This is the only module that talks to Timeloop, and the only one that can be
slow. Every mapping is cached on disk under
`ecc_energy_study/outputs/<arch>/<subdir>/<shape>/`, keyed by the layer shape
alone -- so a run is resumable and two models that share a layer shape share a
mapping.
"""
from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import time

from .paths import (ARCH_COMPONENTS, DESIGNS_DIR, EX_REPO, EXERCISES_URL,
                    PROB_DIR, WORK)

#: Name of the sidecar written beside every mapping this code produces.
MAPPING_SIDECAR = "mapping.json"

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
                 fingerprint=None, legacy_root=None):
        self.cfg = cfg
        self.arch = arch
        self.arch_yaml = arch_yaml
        self.out_root = pathlib.Path(out_root)
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
        }

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
        spec = tl.Specification.from_yaml_files(
            *design_inputs(self.arch_yaml, problem_path(layer)))
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
def _grab(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


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
        for ds, sec in zip(parts[1::2], parts[2::2]):
            ingresses = _grab(r"\n\s+Ingresses\s*:\s*([\d.eE+-]+)", sec)
            e_net = _grab(r"\n\s+Energy \(total\)\s*:\s*([\d.eE+-]+)\s*pJ", sec) or 0.0
            e_link = _grab(r"\n\s+Link transfer energy \(total\)\s*:\s*([\d.eE+-]+)\s*pJ", sec) or 0.0
            e_red = _grab(r"\n\s+Spatial Reduction Energy \(total\)\s*:\s*([\d.eE+-]+)\s*pJ", sec) or 0.0
            rows.append(dict(
                layer=layer_label, level=f"NoC: {name}", dataspace=ds,
                instances=None,
                reads=None if ingresses is None else ingresses * scale,
                writes=0.0,
                energy_pJ=(e_net + e_link + e_red) * scale))
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
