"""The files Timeloop is given: the cloned repo, the problem YAMLs, the design.

`design_inputs()` builds the exact input list a mapper run is handed -- the
patched arch, its globals, the components, and (since prompt_6) a supplied
`ERT:`/`ART:` pair. `problem_path()` writes one layer shape as a problem YAML,
`ensure_exercises_repo()` clones the exercises tree once, `require_container()`
is the refusal that stops a mapper run outside the container, and
`tool_versions()` records what was actually run.

This is the bottom of the `toolchain/` package: `toolchain/ert.py` and
`toolchain/invoke.py` both stand on it, which is why it is its own module and
not part of either.

ProjectRestructure phase 3 cut `timeloop.py` (1,529 lines) into this,
`toolchain/ert.py`, `toolchain/cache.py`, `toolchain/invoke.py` and
`toolchain/stats.py`. Only `invoke.py` needs the container.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess

from ..paths import ARCH_COMPONENTS, DESIGNS_DIR, EX_REPO, EXERCISES_URL, PROB_DIR, WORK
from ..settings import guards


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
        raise guards.refusal("timeloop-mapper-not-on-path",
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
        raise guards.refusal("designs-dir-missing",
            f"cloned the exercises repo but {DESIGNS_DIR} is missing")
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


def design_inputs(arch_yaml, problem_yaml, arch, cfg):
    """Every YAML the mapper is handed, in order.

    Locally authored components come AFTER the cloned repo's, so this project
    can add a component the exercises repo does not have without editing the
    clone. Names must not collide -- a local file redefining an upstream class
    would be a duplicate-class error, not an override -- so the corrected
    register file is called `smartbuffer_RF_decoded`, not `smartbuffer_RF`,
    and the banked SRAM is `smartbuffer_SRAM_banked`.

    `arch` and `cfg` select `globals_<arch>_<content>.yaml`. ONE PER DESIGN
    since prompt_7 C1.5:
    `global_cycle_seconds` is no longer one number for the study, and a shared
    file would clock every design on a multi-design figure at whichever rate
    was written last. It is a required argument, not a default, so a caller
    cannot silently get somebody else's clock.
    """
    g = archs_globals_path(arch, cfg)
    if not g.exists():
        raise guards.refusal("design-inputs-not-written",
            f"design_inputs: {g} has not been written. `archs.write_globals(cfg, "
            f"{arch!r})` runs once per design in `experiments/common.Session.setup()`; "
            f"a caller that reaches Timeloop without it would map at the wrong clock.")
    return ([str(arch_yaml)]
            + sorted(glob.glob(str(DESIGNS_DIR / "_components" / "*.yaml")))
            + sorted(glob.glob(str(ARCH_COMPONENTS / "*.yaml")))
            + [str(DESIGNS_DIR / "_include" / "mapper.yaml"),
               str(g),
               str(problem_yaml)])


def archs_globals_path(arch, cfg):
    """`archs.globals_path`, imported lazily -- `archs` imports this module."""
    from .archs import globals_path
    return globals_path(arch, cfg)


