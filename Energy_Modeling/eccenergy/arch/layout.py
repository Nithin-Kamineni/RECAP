"""Where a design's files are, what its levels are called, and writing globals.

`arch_levels()` and `loop_levels()` name the storage levels of a design as
Timeloop prints them; `patched_arch_path()` is where the patched YAML is
written; `globals_path()` / `write_globals()` produce the per-design
`globals_<arch>.yaml` that carries its own clock (`ECC_ARCH_CLOCK_MHZ`, prompt_7
Phase C1) into the mapper.

THE FILES A SLURM ARRAY SHARES ARE CREATED ONCE, NEVER REPLACED. `_write_once()`
and `_write_atomic()` are why: 72 jobs starting together must not truncate a
file another one is reading, so a file whose `_content_tag()` already matches is
left alone and any rewrite goes through a temporary file and `os.replace`.

ProjectRestructure phase 3 cut this out of `archs.py`.
"""
from __future__ import annotations

import hashlib
import os
import re

from ..paths import WORK

from .fingerprint import effective_variant
from .load import _blocks, load_standard
from .patch import _patched_text


# ------------------------------------------------------- mapper search depth
def loop_levels(text):
    """How many loop levels Timeloop will build for this architecture.

    One per storage component plus one per spatial container -- exactly the
    `L0..Ln` the mapper prints. A `!Parallel` group contributes one level per
    branch (its three scratchpads are three levels, not one), and a `!Nothing`
    branch contributes none.

    This is what `cfg.victory_for()` scales mapper effort by, so that a deep
    hierarchy is not searched less thoroughly than a shallow one and the
    difference then reported as an architecture result.
    """
    storage = spatial = 0
    for _name, body in _blocks(text):
        if body.startswith("Nothing"):
            continue
        if re.search(r"\bmesh[XY]:\s*\d+", body):
            spatial += 1
            continue
        if re.search(r"\bdepth:\s*\d+", body):
            storage += 1
    return storage + spatial


def arch_levels(arch, cfg):
    """`loop_levels` for the arch.yaml this configuration actually maps."""
    return loop_levels(_patched_text(arch, cfg, quiet=True))


def _content_tag(text):
    """8 hex of the content -- the name a deterministic file is written under."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _write_once(dst, text):
    """Write `text` to `dst` if `dst` does not already exist. NEVER REPLACE IT.

    THE BUG THIS EXISTS FOR (2026-09-13, prompt_7 C2, one job of 186).
    `_write_atomic` wrote a private temp file and `os.replace`d it onto a
    SHARED path. That is atomic in POSIX terms -- the path always points at a
    complete file -- and it is still not safe here, because every one of 186
    concurrent SLURM jobs replaced the SAME two paths (`globals_<arch>.yaml`
    and the patched `arch_<arch>_patched__<variant>.yaml`) within seconds of
    each other. On Lustre a client that has already looked the path up holds a
    handle to the OLD inode, and a replacement leaves that handle stale:

        job 41920766, line 35   globals_eyeriss_like_wglb.yaml: technology=45nm
        job 41920766, line 78   FileNotFoundError: ... globals_eyeriss_like_wglb.yaml

    written and then missing, in one process, eight seconds apart. One failed
    job out of 186 was enough to leave the dependent eval on
    `DependencyNeverSatisfied` and the whole matrix unreadable.

    THE FIX IS THE NAME, NOT THE WRITE. Both files are a pure function of
    (architecture, configuration), so they are content-addressed: identical
    content means an identical name, and a name that exists already holds the
    bytes this caller wanted. `O_CREAT | O_EXCL` then creates it exactly once,
    atomically, and NOTHING EVER REPLACES AN EXISTING FILE -- so no reader can
    be holding a handle to something that is about to be unlinked. A second
    writer losing the race is not an error; it is the normal case.

    Fingerprints do not move: `arch_fingerprint()` hashes the CONTENT of these
    files (`globals_view`, `arch_yaml`), never their paths.
    """
    if dst.exists():
        return dst
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    with open(tmp, "w", newline="\n") as fh:
        fh.write(text)
    try:
        # O_EXCL via link(): create the name only if it is free, and never
        # clobber. `os.replace` would overwrite, which is the whole problem.
        os.link(tmp, dst)
    except FileExistsError:
        pass                       # another job wrote the same bytes first
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return dst


def _write_atomic(dst, text):
    """Replace `dst` with `text`, atomically. For a file only ONE process writes.

    Use `_write_once` for anything a SLURM array writes concurrently -- see the
    stale-handle failure recorded there. This remains for single-writer paths.
    """
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    with open(tmp, "w", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, dst)
    return dst


def patched_arch_path(arch, cfg):
    """Write and return the arch.yaml for this architecture + treatment.

    CONTENT-ADDRESSED and written ONCE (`_write_once`): every job of a SLURM
    array shares this path, and replacing it under a concurrent reader is what
    cost one job of 186 on 2026-09-13. The tag is 8 hex of the text, so two
    treatments that produce identical YAML share one file and a change makes a
    NEW one rather than overwriting the old.
    """
    text = _patched_text(arch, cfg)
    variant = effective_variant(arch, cfg)
    suffix = "" if variant == "stock" else f"__{variant}"
    dst = WORK / f"arch_{arch}_patched{suffix}__{_content_tag(text)}.yaml"
    return _write_once(dst, text)


def globals_text(cfg, arch):
    """The bytes THIS design's `globals.yaml` holds -- a pure function of
    (configuration, architecture), which is what lets it be content-addressed."""
    node = cfg.force_technology or load_standard()["study"]["technology"]
    return ("variables:\n"
            "  version: 0.4\n"
            f"  global_cycle_seconds: {cfg.cycle_seconds_for(arch)}\n"
            f'  technology: "{node}"\n')


def globals_path(arch, cfg):
    """Where THIS design's `globals.yaml` lives.

    ONE FILE PER DESIGN since prompt_7 C1.5, because `global_cycle_seconds` is
    no longer one number for the study: Eyeriss v1 runs at its published
    200 MHz while the rest stays at the 1 GHz model default. A shared file
    would clock every design on a multi-design figure at whichever rate was
    written last -- silently, and only for the designs that are not first.

    CONTENT-ADDRESSED since 2026-09-13: the name carries 8 hex of the bytes, so
    186 concurrent jobs writing "the same" file write the SAME NAME and nothing
    ever replaces a file a reader may be holding open. That replacement is what
    cost one job of 186 (`_write_once` records the failure).
    """
    return WORK / f"globals_{arch}_{_content_tag(globals_text(cfg, arch))}.yaml"


def write_globals(cfg, arch):
    """`globals.yaml` sets the node for anything OUTSIDE an arch container.

    DRAM sits above the accelerator container in every one of these designs, so
    it takes its technology from here, identically for every architecture. The
    CLOCK is per design (`Config.cycle_seconds_for`), which is why this takes
    one.

    THE DEFAULT USED TO BE 65nm while every accelerator container declared 45nm,
    so DRAM -- the level this study spends most of its energy in, and the level
    BCH parity lands on -- was costed at a different node from the logic it
    talks to. It was uniform across architectures, so it never showed up as an
    ordering error; it just made every absolute number and every savings
    percentage wrong by a fixed factor. The node now comes from
    archs/_shared/standard.yaml, which is the same file the architectures are
    validated against, so the two cannot drift apart again.
    """
    text = globals_text(cfg, arch)
    node = cfg.force_technology or load_standard()["study"]["technology"]
    return _write_once(globals_path(arch, cfg), text), node


# ---------------------------------------------------------------------- audit
