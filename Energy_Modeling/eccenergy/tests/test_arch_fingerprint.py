"""What moves a mapper cache, and what must not.

PROPERTY TESTS ON THE REAL ARCH FILES, PLUS DELIBERATE BREAKAGE. Every
assertion reads `archs/` as it stands and is followed by the mutation it exists
to catch, applied to a copy and checked to give the opposite answer. An
assertion nobody has seen fail is a comment.

WHAT THIS IS THE GATE FOR (2026-09-13)
--------------------------------------
`arch_fingerprint()` names the mapper cache directory, so whatever it hashes is
the definition of "a different chip". It hashed the patched YAML byte for byte,
which made two opposite mistakes possible and one of them happened twice in a
morning:

  * A GEOMETRY EDIT MUST COLD THE CACHE. Without that, mappings solved for a
    64-entry psum spad are read back as a 96-entry one and the figure averages
    two architectures. This is the failure the fingerprint was written for and
    it still holds -- `test_geometry_still_colds_the_cache`.

  * A COMMENT EDIT MUST NOT. `archs/<name>/arch_paper.yaml` is required to
    carry the citation for every number it declares, and while comments were
    hashed, correcting one that had gone stale threw away every solved mapping
    in the study. Timeloop parses YAML; a comment reaches no mapping decision
    and no energy. Hashing one reported an architecture change that had not
    happened -- `test_documentation_is_free`.

The second half is not a relaxation of the first. `hashable_arch_text()` strips
only what Timeloop never reads: comments, trailing whitespace, blank lines.
Every digit survives it.

THE THIRD PROPERTY is that the pin is honoured -- `ECC_ARCH_PIN_DIR` is what
makes one SLURM submission map one architecture, and a pin that silently did
nothing would put the study back where it was on 2026-09-13, with the maps
solving one chip and the dependent eval looking for another.
"""
import pathlib
import re
import shutil
import sys
import tempfile

import pytest

from .. import config, paths
from ..arch import fingerprint
from ..arch import load


# --------------------------------------------------------------- the harness
def _cfg():
    return config.load_config()


def _binders(name):
    """Every loaded `eccenergy` module that holds `name` in its own namespace.

    `arch_source` is defined once, in `arch/load.py`, and IMPORTED BY NAME into
    `arch/patch.py`, `arch/fingerprint.py` and `arch/layout.py` -- so each of
    them has its own binding and rebinding one of them redirects only that one.
    Before ProjectRestructure phase 3 all four were `archs.py` and one
    assignment covered the whole call chain. Asking `sys.modules` which modules
    hold the name keeps that true without naming them here: a fifth importer
    is covered the day it appears.
    """
    real = getattr(load, name)
    return [m for n, m in list(sys.modules.items())
            if n.startswith("eccenergy.") and getattr(m, name, None) is real]


def _fp_of_text(text, cfg, arch, tmp_path, tag):
    """The fingerprint `arch` would have if its source file held `text`."""
    p = tmp_path / f"arch_{tag}.yaml"
    p.write_text(text)
    real = load.arch_source
    mods = _binders("arch_source")
    for m in mods:
        m.arch_source = lambda *a, **k: p
    try:
        return fingerprint.arch_fingerprint(arch, cfg)
    finally:
        for m in mods:
            m.arch_source = real


@pytest.fixture
def live(tmp_path):
    cfg = _cfg()
    arch = cfg.archs[0]
    text = load.arch_source(arch, cfg).read_text()
    base = _fp_of_text(text, cfg, arch, tmp_path, "base")
    return cfg, arch, text, base, tmp_path


# ------------------------------------------------ 1. documentation is free
@pytest.mark.parametrize("name,mutate", [
    # a comment reworded on the line beside a declared number
    ("reword", lambda t: t.replace("# paper: 512-b x 64-b SRAM banks",
                                   "# COMPLETELY DIFFERENT WORDS HERE")),
    # a whole comment line added
    ("add_line", lambda t: t.replace("      name: psum_glb",
                                     "      # a note added later\n      name: psum_glb")),
    # blank lines
    ("blank_lines", lambda t: t.replace("      name: psum_glb",
                                        "\n\n      name: psum_glb")),
    # trailing whitespace
    ("trailing_ws", lambda t: t.replace("      name: psum_glb",
                                        "      name: psum_glb    ")),
])
def test_documentation_is_free(live, name, mutate):
    """Editing what the file SAYS must not re-map what the file IS.

    The breakage this catches is the byte-for-byte hash: under it every one of
    these moved the fingerprint and cost the study its entire mapper cache.
    """
    cfg, arch, text, base, tmp_path = live
    mutated = mutate(text)
    assert mutated != text, f"{name}: the mutation did not change the file"
    assert _fp_of_text(mutated, cfg, arch, tmp_path, name) == base, (
        f"{name} moved the fingerprint: a documentation edit re-keyed the "
        f"mapper cache, which is the 2026-09-13 defect")


def test_the_normaliser_keeps_every_digit(live):
    """`hashable_arch_text` may delete comments and whitespace -- nothing else."""
    _cfg_, _arch, text, _base, _tmp = live
    stripped = fingerprint.hashable_arch_text(text)
    # every `key: <number>` line survives, value intact
    for line in text.splitlines():
        body = line.split("#")[0].rstrip()
        if re.match(r"^\s*[A-Za-z_][\w-]*:\s*-?\d", body):
            assert body in stripped, f"normaliser dropped a declared value: {body!r}"
    assert "#" not in stripped, "a comment survived the normaliser"


def test_a_hash_inside_a_quoted_scalar_is_data(live):
    """`name: "a#b"` is a value, not the start of a comment.

    No arch file has one today. One added later must not be silently truncated
    into a different architecture that hashes the same as its neighbour.
    """
    cfg, arch, text, _base, tmp_path = live
    with_hash = text.replace('technology: "45nm"', 'technology: "45nm#A"')
    other = text.replace('technology: "45nm"', 'technology: "45nm#B"')
    assert with_hash != text and other != text
    a = _fp_of_text(with_hash, cfg, arch, tmp_path, "quoted_a")
    b = _fp_of_text(other, cfg, arch, tmp_path, "quoted_b")
    assert a != b, ("a '#' inside a quoted scalar was treated as a comment, so "
                    "two different architectures hash identically")


# --------------------------------------------- 2. geometry still colds it
@pytest.mark.parametrize("name,pattern,repl", [
    ("psum_glb_depth", r"(name: psum_glb.*?depth: )(\d+)", r"\g<1>999"),
    ("psum_spad_depth", r"(name: psum_spad.*?depth: )(\d+)", r"\g<1>999"),
    ("filter_glb_depth", r"(name: filter_glb.*?depth: )(\d+)", r"\g<1>999"),
    ("filter_glb_banks", r"(name: filter_glb.*?n_banks: )(\d+)", r"\g<1>9"),
    ("weights_spad_width", r"(name: weights_spad.*?width: )(\d+)", r"\g<1>999"),
])
def test_geometry_still_colds_the_cache(live, name, pattern, repl):
    """A declared number is the architecture. Change one and the cache MUST move.

    This is the property `arch_fingerprint` was written for: without it a
    mapping solved for one geometry is read back as another and two chips are
    averaged into one bar. Normalising comments out of the hash must not have
    weakened it.
    """
    cfg, arch, text, base, tmp_path = live
    mutated = re.sub(pattern, repl, text, count=1, flags=re.S)
    assert mutated != text, f"{name}: the mutation matched nothing -- test is blind"
    assert _fp_of_text(mutated, cfg, arch, tmp_path, name) != base, (
        f"{name} left the fingerprint alone: a geometry change would be read "
        f"back out of the previous architecture's cache")


# ------------------------------------------------------- 3. the pin is real
def test_the_arch_pin_is_honoured(tmp_path, monkeypatch):
    """`ECC_ARCH_PIN_DIR` must actually redirect where an architecture is read.

    It is what makes one SLURM submission map one architecture (hpc/
    map_ert_arms.sh). A pin that silently did nothing would leave the study
    where it was on 2026-09-13: an edit landing mid-array, half the run solving
    one chip and the dependent eval looking for another.
    """
    pin = tmp_path / "archpin"
    shutil.copytree(paths.ROOT / "archs", pin)
    monkeypatch.setenv("ECC_ARCH_PIN_DIR", str(pin))
    import importlib
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.ARCH_SRC == pin.resolve(), "the pin did not redirect ARCH_SRC"
        assert reloaded.ARCH_COMPONENTS.is_relative_to(pin.resolve()), (
            "the shared components still came from the live tree, so a pinned "
            "run would hash one thing and map another")
    finally:
        monkeypatch.delenv("ECC_ARCH_PIN_DIR", raising=False)
        importlib.reload(paths)


def test_a_missing_pin_falls_back_instead_of_stopping(tmp_path, monkeypatch):
    """A stale pin warns and reads `archs/`. It must never stop a run."""
    monkeypatch.setenv("ECC_ARCH_PIN_DIR", str(tmp_path / "does-not-exist"))
    import importlib
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.ARCH_SRC == reloaded.ROOT / "archs"
    finally:
        monkeypatch.delenv("ECC_ARCH_PIN_DIR", raising=False)
        importlib.reload(paths)
