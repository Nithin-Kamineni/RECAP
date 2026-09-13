"""Every filesystem location the project uses, in one module.

Two rules that the old scripts got wrong often enough to cost hours:

1. The working directory (`ecc_energy_study/`, holding the cloned exercises repo
   and the expensive mapper cache) is resolved from the PROJECT ROOT, which is
   found by walking up from this file -- not from `pathlib.Path.cwd()`. So it no
   longer matters which directory you launch from.

2. Results never land next to a script, and there is exactly ONE output name per
   sweep -- `results/figures/BCHsweep.png`, `ModelSweep`, `ArchitectureSweep`.
   Re-running at different constants overwrites; it does not accumulate a new
   directory. The manifest beside it records which constants produced the file.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from .settings import guards

#: eccenergy/paths.py -> eccenergy/ -> project root
ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The auto-managed working directory: cloned repo + mapper cache + workloads.
WORK = ROOT / "ecc_energy_study"

#: timeloop-accelergy-exercises clone (kept: slow to re-clone).
EX_REPO = WORK / "timeloop-accelergy-exercises"
DESIGNS_DIR = EX_REPO / "workspace" / "example_designs" / "example_designs"

#: Locally authored architectures, master copies (survive a wipe of WORK).
#:
#: `ECC_ARCH_PIN_DIR` REDIRECTS THIS AT A SNAPSHOT, and the launchers set it so
#: that ONE SUBMISSION MAPS ONE ARCHITECTURE.
#:
#: THE FAILURE IT EXISTS FOR (2026-09-13, efficientnet_b0, 282 jobs wasted).
#: `archs/eyeriss_like_wglb/arch_paper.yaml` was edited 78 seconds after the
#: map array started -- a deliberate experiment with a new psum_spad depth.
#: Every map job then solved the pre-edit chip and filed its result under
#: `fp-eca1a94653f8`; the dependent eval, starting twelve minutes later, read
#: the edited file, computed `fp-d20f19432878`, found an empty directory and
#: reported all 82 layers as `mapper failed`. Nothing refused the run and
#: nothing was wrong with either the model or the architecture: the maps and
#: the eval simply described two different chips, because each resolved the
#: architecture WHEN IT RAN rather than when the batch was submitted.
#:
#: A pin is not a relaxation of `arch_fingerprint()` -- the fingerprint is
#: still computed, still names the cache directory, and is still what stops
#: two geometries being averaged into one bar. It just fixes WHICH bytes
#: everyone in one submission hashes, so editing `archs/` while jobs are in
#: flight is free. Editing between submissions works as it always did: the
#: next run snapshots the new file and maps it.
#:
#: Unset (an interactive run) this is `archs/` and nothing changes.
_ARCH_PIN = os.environ.get("ECC_ARCH_PIN_DIR", "").strip()
if _ARCH_PIN and pathlib.Path(_ARCH_PIN).is_dir():
    ARCH_SRC = pathlib.Path(_ARCH_PIN).resolve()
else:
    if _ARCH_PIN:
        # Warn, never stop: a missing pin is a stale launcher, and falling back
        # to the live `archs/` is the behaviour that still produces a figure.
        print(f"paths.py: ECC_ARCH_PIN_DIR={_ARCH_PIN!r} is not a directory; "
              f"reading the live {ROOT / 'archs'} instead", file=sys.stderr)
    ARCH_SRC = ROOT / "archs"

#: The standardized-comparison contract every design is audited against, and
#: the level-by-level record of where each design's numbers came from. Neither
#: is read by Timeloop; `bash run.sh validate` checks the YAMLs against them.
#: Compound components authored HERE, layered on top of the ones the cloned
#: exercises repo ships. Lets the study correct a defect in an upstream
#: component (see components/regfile_decoded.yaml) without editing the clone,
#: which must stay pristine and re-clonable.
ARCH_COMPONENTS = ARCH_SRC / "_shared" / "components"

ARCH_STANDARD = ARCH_SRC / "_shared" / "standard.yaml"
ARCH_PROVENANCE = ARCH_SRC / "_shared" / "provenance.yaml"
ARCH_NOC = ARCH_SRC / "_shared" / "noc.yaml"

#: Directories under archs/ that are shared data, not architectures.
ARCH_SRC_RESERVED = ("_shared",)

#: Design Compiler reconstruction-energy JSONs and other inputs.
DATA_DIR = ROOT / "data"

#: Generated problem YAMLs, shared by every experiment.
PROB_DIR = WORK / "problems"

#: Workload shape files produced by the generators.
CNN_LAYERS = WORK / "model_layers.json"
TRANSFORMER_LAYERS = WORK / "transformer_layers.json"

EXERCISES_URL = "https://github.com/Accelergy-Project/timeloop-accelergy-exercises.git"


# --------------------------------------------------------------- result layout
class Results:
    """One flat, fixed layout -- three sweeps, three names, nothing to prune.

        results/
          figures/{BCHsweep,ModelSweep,ArchitectureSweep}.{png,pdf}
          tables/  same stems, .csv   (plus diagnose.csv)
          manifests/ same stems, .json -- the config that produced the figure
          _raw/    the architecture-level energy cache

    The stem comes from `cfg.stem`, which is decided by the sweep alone. Change
    a constant, re-run, and the same three files are rewritten in place.

    `results/_raw/` is the architecture-level energy cache. Raw energies depend
    only on (architecture treatment, model) -- never on the ECC configuration --
    so changing K and re-running is instant. Its layout is unchanged, so an
    existing cache stays valid.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        base = pathlib.Path(cfg.results_dir)
        self.base = base if base.is_absolute() else ROOT / base
        self.run_dir = self.base
        self.figures = self.base / "figures"
        self.tables = self.base / "tables"
        self.manifests = self.base / "manifests"
        self.raw_root = self.base / "_raw"
        self.evaluation = self.base / "evaluation"

    def prepare(self):
        for d in (self.figures, self.tables, self.manifests, self.raw_root,
                  self.evaluation):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # ---- the evaluation namespace (04_results_storage_spec.txt) ------------
    def evaluation_dir(self, arch, model, phase=None):
        """`results/evaluation/{Pre|Post}/{ARCH}/{MODEL}/{BCH}/{PREC}/{SCOPE}/{MAPPER}`

        The spec asks for `Results/evaluation/{Pre|Post}/{ARCH}/{MODEL}/{BCH}/...`
        and then for the namespace to be extended only as far as needed to stop
        collisions. Four extensions were needed, and no more:

        `{PREC}`    `w8a8`, plus `accNN` when a common accumulator width is
                    FORCED for the sensitivity study. Without it a forced-width
                    run would overwrite the paper-native primary result, which
                    is the one comparison that must never be silently replaced.
        `{SCOPE}`   `layers-full`, or `layers2__<name>__<name>`. Development
                    results and full-model results are different claims and the
                    plan says they must not be confusable. The layer identity is
                    in the path AND in the JSON.
        `{MAPPER}`  objective, victory, scaling, algorithm, seed. Two mappings
                    found under different search settings are different results.
        `<runid>`   the file itself, timestamped, so a re-run never silently
                    overwrites a prior one.

        Everything below `{PREC}` is deterministic given the config: the same
        configuration always resolves to the same directory.
        """
        return (self.evaluation / (phase or self.cfg.phase) / arch / model /
                self.cfg.code_slug / self.cfg.precision_slug /
                self.cfg.layer_slug / self.cfg.mapper_slug)

    def evaluation_path(self, arch, model, run_id, phase=None):
        d = self.evaluation_dir(arch, model, phase)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{run_id}.json"

    # ---- raw energy cache ---------------------------------------------------
    def raw_path(self, arch, model, variant=None, fingerprint=None):
        """One JSON per (arch treatment, mapping fingerprint, workload, layer scope).

        The stored record is keyed by everything that changes its CONTENT but
        nothing that changes only the ECC arithmetic on top -- that is what makes
        `--replot` at a different BCH(N,K) instant. `classify_mode` and the
        read/write split both decide which category names the record holds, so
        both are in the path; mixing them would silently zero out categories.

        TWO SEGMENTS ADDED IN TASK 1.

        `fp-<hash>` is the mapping fingerprint: a hash of the actual patched
        architecture YAML plus the mapper settings. Before this existed, editing
        an arch.yaml left the treatment slug unchanged, so a re-run silently
        mixed energies from the old geometry with the new one. Editing a YAML
        now moves the cache, which is the only safe behaviour.

        `<layer scope>` keeps a two-layer development result from ever being
        read back as a full-model result. The plan requires that separation and
        a shared directory cannot provide it.
        """
        d = (self.raw_root / arch / (variant or self.cfg.arch_variant_slug) /
             f"fp-{fingerprint or 'none'}" / self.cfg.workload /
             self.cfg.layer_slug / f"cls-{self.cfg.classify_mode}" /
             ("rw-split" if self.cfg.split_read_write else "rw-joint"))
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{model}.json"

    # ---- mapper cache (the expensive one) ----------------------------------
    def mapper_cache(self, arch, variant=None, fingerprint=None, create=True):
        """`ecc_energy_study/outputs/<arch>/<subdir>`.

        `subdir` stays 'multimodel' / 'llm' for the stock architecture
        treatment, exactly as the historical scripts named it, so every mapping
        already on disk is still a cache hit. A non-stock treatment (forced
        technology node, forced datawidth, different clock) changes the
        architecture the mapper sees, so it gets its own subdirectory instead of
        silently reusing incomparable results.
        """
        variant = variant or self.cfg.arch_variant_slug
        subdir = "multimodel" if self.cfg.workload == "cnn" else "llm"
        if variant != "stock":
            subdir = f"{subdir}__{variant}"
        d = WORK / "outputs" / arch / subdir
        if fingerprint:
            # Content-addressed on the architecture YAML the mapper actually
            # sees plus the search settings. A cache entry can then never be a
            # mapping for a different design that happened to share a slug.
            d = d / f"fp-{fingerprint}"
        # `create=False` for a pure READER. Resolving a path used to create it,
        # so a read-only report that enumerates (design x capacity) left an
        # empty `<slug>/fp-<hash>/` behind for every combination it asked
        # about -- 45 of them from one test run on 2026-09-09, which makes an
        # `ls` of the cache tree claim capacities that were never mapped.
        # Anything that WRITES a mapping keeps the default.
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def legacy_mapper_cache(self, arch, variant=None):
        """The pre-fingerprint directory, for `ECC_CACHE_STRICT=0` only.

        Mappings computed before Task 1 have no sidecar recording which YAML
        produced them, so nothing can prove they match the current one. They
        are left on disk (they are hours of compute) and are readable only when
        the caller explicitly accepts unverifiable provenance -- which is then
        recorded in the result JSON as `mapping_provenance: legacy`.
        """
        variant = variant or self.cfg.arch_variant_slug
        subdir = "multimodel" if self.cfg.workload == "cnn" else "llm"
        if variant != "stock":
            subdir = f"{subdir}__{variant}"
        return WORK / "outputs" / arch / subdir

    # ---- outputs ------------------------------------------------------------
    def figure_paths(self, stem=None):
        stem = stem or self.cfg.stem
        return [self.figures / f"{stem}.{fmt}" for fmt in self.cfg.formats]

    def table_path(self, stem=None):
        return self.tables / f"{(stem or self.cfg.stem)}.csv"

    def manifest_path(self, stem=None):
        return self.manifests / f"{(stem or self.cfg.stem)}.json"

    def write_manifest(self, extra=None, stem=None):
        payload = {"config": self.cfg.to_dict(),
                   "mapper_fingerprint": self.cfg.fingerprint()}
        # ProjectRestructure section 6.4 rule 2: AN OVERRIDE IS RECORDED. A tier
        # 3 or 4 guard lifted by `ECC_ALLOW` lands here, so a manifest that
        # carries this key is the manifest of an ABLATION and says so itself.
        # WRITTEN ONLY WHEN SOMETHING FIRED: every published run of this study
        # overrides nothing, and an always-present `"overrides": []` would have
        # rewritten every manifest on disk without changing one number in it.
        fired = guards.overrides()
        if fired:
            payload["guard_overrides"] = fired
        if extra:
            payload.update(extra)
        p = self.manifest_path(stem)
        # An explicit LF: this file is read inside a Linux container, and a CRLF
        # written from a Windows-side run is exactly what `tools/fix-eol.sh` has
        # to repair otherwise.
        with open(p, "w", newline="\n") as fh:
            fh.write(json.dumps(payload, indent=1, default=str))
        return p


def require(path, hint):
    if not pathlib.Path(path).exists():
        raise guards.refusal("missing-path",
            f"missing: {path}\n  -> {hint}")
    return pathlib.Path(path)


# ------------------------------------------------------------ prompt_6 probes
#: Scratch output of `experiments/ert_probe.py` (prompt_6 PHASE 1). It runs
#: Timeloop with a deliberately absurd energy table and MUST NOT land in the
#: mapper cache: the mapping fingerprint does not include the ERT until PHASE 3,
#: so a probe written under `outputs/` would sit at the reference fingerprint
#: and be read back as the reference design.
ERT_PROBE = WORK / "ert_probe"


def ert_probe_dir(arch, fingerprint, shape):
    """`ecc_energy_study/ert_probe/<arch>/fp-<fingerprint>/<shape>` (created)."""
    d = ERT_PROBE / arch / f"fp-{fingerprint}" / shape
    d.mkdir(parents=True, exist_ok=True)
    return d
