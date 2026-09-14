"""THE only module that runs Timeloop, and the only one that can be slow.

`Mapper` resolves one (architecture, layer shape) to a solved mapping: it looks
in the cache first, takes the `ShapeLock` if it has to solve, stages the supplied
ERT/ART, runs `timeloop-mapper` inside the container, and saves the result
immediately -- so a run is resumable and two models that share a layer shape
share a mapping.

The cache lives at `ecc_energy_study/outputs/<arch>/<treat>/fp-<hash>/<shape>/`,
keyed by the layer shape and the architecture fingerprint. NEVER DELETE IT; it is
hours of compute. `ECC_FROM_CACHE=1` makes a miss an error instead of a mapper
run, which is what every `--eval` stage runs under.

A `def` INSIDE `__init__` ENDS `__init__`. A `@property` added in the middle of
this class's constructor orphaned every line after it, `self._memo` included, and
killed 72 SLURM jobs ten seconds in -- because no test had ever CONSTRUCTED a
Mapper. `tests/test_phase_c.py` does now.

ProjectRestructure phase 3 cut this out of `timeloop.py`.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import time

from .cache import ShapeLock
from .ert import ErtMismatch, ErtTables, _close, describe_bump, read_back_ert, same_bump
from .inputs import ERT_FOUND, ERT_GENERATED, MAPPING_SIDECAR, design_inputs, load_timeloopfe, problem_path, tool_versions


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
        #: prompt_7 C1.6: the MAC price the mapper optimises against, or None.
        #: Independent of the arm -- every arm, reference included, is mapped
        #: at the price the report charges.
        self.mac_pj = getattr(cfg, "mac_pj_override", None)
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

    @property
    def supplies_ert(self):
        """Is this run handing Timeloop a table at all?

        True as soon as ANY row is patched: an arm's bump (prompt_6) or the MAC
        price (prompt_7 C1.6). With neither, Accelergy writes the table itself
        and there is nothing to read back.
        """
        return self.ert_bump is not None or self.mac_pj is not None

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
        if not self.supplies_ert:
            return None
        if self._ert_tables is not None:
            return self._ert_tables.sidecar_record()
        return {"bump": self.ert_bump, "base_pj": None, "patched_pj": None,
                "mac_pj": self.mac_pj}

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
        # prompt_7 C1.6: and the MAC price, for the same reason. It is in the
        # fingerprint, so this can only fire on a hand-copied entry -- but a
        # cache entry that says in words what it was mapped at is the thing
        # that makes a wrong pick findable.
        their_mac = (rec.get("ert_bump") or {}).get("mac_pj")
        if (their_mac is None) != (self.mac_pj is None) or (
                self.mac_pj is not None and not _close(their_mac, self.mac_pj)):
            return None, (f"sidecar was mapped at {their_mac!r} pJ/MAC but this "
                          f"run charges {self.mac_pj!r} (ECC_MAC_PJ_OVERRIDE is "
                          f"in the mapper's objective since prompt_7 C1.6)")
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
        inputs = design_inputs(self.arch_yaml, problem_path(layer), self.arch, self.cfg)
        if self.supplies_ert:
            # prompt_6: the arm's table, generated once per (arch, arm,
            # fingerprint), pre-written under Timeloop's own names (FINDINGS
            # 3.5 fact 1) and handed to timeloopfe as two extra inputs.
            # prompt_7 C1.6: the REFERENCE arm has one too now, carrying the
            # MAC price and no bump.
            if self._ert_tables is None:
                self._ert_tables = ErtTables(self.cfg, self.arch, self.arch_yaml,
                                             self.out_root, self.ert_bump,
                                             mac_pj=self.mac_pj)
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
            if self.supplies_ert:
                # The table must have been USED (not regenerated) and must
                # still be the one staged: RULE 4.4.5's read-back, on the
                # entry that was just written. Since prompt_7 C1.6 this covers
                # the REFERENCE arm too -- its table carries the MAC price, and
                # a reference entry mapped at Accelergy's own 1.13 pJ/MAC while
                # the report charges 0.23 is exactly the mismatch this catches.
                log = (out_dir / "mapper_console.log").read_text(errors="replace")
                if ERT_FOUND not in log or ERT_GENERATED in log:
                    self.n_mapped -= 1
                    self._fail(layer, out_dir, f"Timeloop did not take the '{ERT_FOUND}' "
                               f"branch; the supplied ERT was not used")
                    return None
                try:
                    read_back_ert(out_dir, self.ert_bump, mac_pj=self.mac_pj)
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

    def _sibling_fingerprints(self):
        """Other `fp-*` directories under this treatment, and how many shapes each holds.

        A cold cache has exactly two causes and they need opposite responses:
        the architecture was never mapped, or it MOVED since it was. Listing the
        neighbours tells them apart at a glance -- a sibling holding 47 shapes
        an hour after a depth was edited is the second case.

        Reported only. Reading one of them back would be a comparison of two
        architectures, which is the thing `arch_fingerprint()` exists to stop.
        """
        try:
            parent = self.out_root.parent
            here = self.out_root.name
            rows = []
            for d in sorted(parent.glob("fp-*")):
                if d.name == here or not d.is_dir():
                    continue
                n = sum(1 for _ in d.glob("*/timeloop-mapper.stats.txt"))
                if n:
                    rows.append((d.name[3:], n))
            return sorted(rows, key=lambda r: -r[1])[:6]
        except Exception:
            return []

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
            # architecture treatment is simply empty. Say WHICH architecture is
            # being looked for and what builds it -- "nothing is wrong with
            # <arch>" on its own sent a reader hunting the model and the layers
            # (2026-09-13, efficientnet_b0) when the answer was that the
            # architecture had moved since the maps were solved.
            lines.append(f"    -> nothing is wrong with {self.arch} and nothing is "
                         f"wrong with the model: THE MAPPER CACHE FOR THIS EXACT "
                         f"ARCHITECTURE IS EMPTY.")
            lines.append(f"       looking for : fp-{self.fingerprint}")
            lines.append(f"       cache dir   : {self.out_root}")
            siblings = self._sibling_fingerprints()
            if siblings:
                lines.append(f"       BESIDE IT, solved for OTHER geometries -- each "
                             f"is a different chip, so none of them may be read "
                             f"back as this one:")
                for fp, n in siblings:
                    lines.append(f"         fp-{fp}  {n} shape(s)")
                lines.append(f"       If one of those IS the architecture you meant, "
                             f"put the YAML back the way it was and re-run; if this "
                             f"one is, the mappings have to be solved for it.")
            lines.append(f"       build it:  ECC_RERUN_OPTIMISER=1 "
                         f"bash hpc/run_all.sh --map-only")
            lines.append(f"       (or, one architecture at a time, in the container: "
                         f"bash run.sh)")
        elif self.n_mapped == 0 and self.n_cached == 0:
            lines.append(f"    -> EVERY shape failed on {self.arch}. One bad "
                         f"architecture specification fails them all identically, "
                         f"so suspect the arch.yaml before the layers.")
            lines.append(f"       full log: {self.out_root / self.failures[0][0]}"
                         f"/mapper_console.log")
        return "\n".join(lines)


