"""Everything OUTSIDE `eccenergy/` that imports it -- audited statically.

THIS FILE EXISTS BECAUSE THE GATE HAS A BLIND SPOT, and the blind spot cost a
session. `restructure/gate.sh` runs `run.sh`, the test suite and the fingerprint
dump; it does NOT run `hpc/*.sh` (they submit SLURM jobs) or
`tools_bch_sweep_tables.py`. So when phase 2 moved 23 modules between packages,
FOUR of those scripts were left importing modules that no longer existed --
`from eccenergy import archs`, `from eccenergy import code_widths` -- and
nothing said so. Phase 3 found them BY HAND.

What makes it silent is not the move, it is this:

    D=$(... bash hpc/tl.sh python3 -c "..." 2>/dev/null | tail -1)

An `ImportError` goes to stderr, stderr goes to `/dev/null`, `D` comes back
empty, and the script prints "no cache directory at this fingerprint yet" --
which is a SENTENCE A HEALTHY RUN ALSO PRINTS. A broken import is indis-
tinguishable from an unmapped architecture at the only place anybody looks.

WHAT THIS AUDIT DOES. It extracts every block of Python embedded in a shell
script (heredocs and `python3 -c` alike), parses it, and resolves every
`eccenergy` name it uses against the package AS IT IS ON DISK:

  * `from eccenergy import X`            -- X must be a real submodule or attribute
  * `from eccenergy.a.b import X`        -- likewise
  * `python3 -m eccenergy.x.y`           -- must name a real, runnable module
  * `mod.attr` for any `mod` bound above -- the attribute must exist

The last one is the half that catches a SPLIT rather than a move: `arch/` still
exists after `archs.py` is cut into four, so the import survives and
`fingerprint.arch_fingerprint` is what actually goes missing.

IT NEVER EXECUTES THE SCRIPTS. Reading is the whole point -- running them
submits jobs. Import resolution is done on the real package, so a module that
imports cleanly but was renamed is still caught.
"""
import ast
import importlib
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = "eccenergy"

#: Shell scripts and loose Python that live OUTSIDE the package and import it.
#: A new one is picked up automatically -- this is a glob, not a list, so a
#: script added next month is audited the day it lands.
SHELL = sorted(ROOT.glob("hpc/*.sh")) + sorted(ROOT.glob("*.sh"))
LOOSE = sorted(p for p in ROOT.glob("*.py") if p.name != "setup.py")

#: A `${VAR}` or `$(cmd)` inside an embedded block is SHELL, not Python. Only
#: one block in the tree has one (`map_depth_sweep.sh`'s datawidth probe), and
#: a placeholder keeps it parseable without pretending to know the value.
_SUBST = re.compile(r"\$\{[^}]*\}|\$\([^)]*\)|\$[A-Za-z_][A-Za-z_0-9]*")


def _blocks(text, path):
    """Every embedded Python block, as (label, source). Heredocs and `-c`."""
    out = []
    # `python3 - arg <<'PY' ... PY`  /  `<<PY ... PY`
    for m in re.finditer(r"<<\s*'?([A-Za-z_][A-Za-z_0-9]*)'?\s*\n(.*?)^\1$",
                         text, re.S | re.M):
        out.append((f"{path.name} heredoc<<{m.group(1)} "
                    f"line {text[:m.start()].count(chr(10)) + 1}", m.group(2)))
    # `python3 -c "..."` -- double-quoted, and no escaped quote is used in the tree
    for m in re.finditer(r'python3 -c "(.*?)"(?=\s|\)|$)', text, re.S):
        out.append((f"{path.name} python3 -c "
                    f"line {text[:m.start()].count(chr(10)) + 1}", m.group(1)))
    return [(lab, src) for lab, src in out if PKG in src]


def _embedded():
    """(label, source) for every embedded block, plus every loose .py file."""
    found = []
    for f in SHELL:
        found += _blocks(f.read_text(), f)
    for f in LOOSE:
        src = f.read_text()
        if PKG in src:
            found.append((f.name, src))
    return found


def _imports(tree):
    """(module, name, alias) for every `eccenergy` import in one parsed block."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and (
                n.module == PKG or n.module.startswith(PKG + ".")):
            for a in n.names:
                out.append((n.module, a.name, a.asname or a.name))
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name == PKG or a.name.startswith(PKG + "."):
                    out.append((a.name.rsplit(".", 1)[0], a.name.rsplit(".", 1)[-1],
                                a.asname or a.name.split(".")[0]))
    return out


def _resolve(module, name):
    """'' if `from <module> import <name>` would work, else why it would not."""
    try:
        mod = importlib.import_module(module)
    except Exception as exc:                      # noqa: BLE001 -- report, never raise
        return f"`import {module}` fails: {type(exc).__name__}: {exc}"
    if hasattr(mod, name):
        return ""
    try:
        importlib.import_module(f"{module}.{name}")
        return ""
    except Exception:                             # noqa: BLE001
        pass
    return (f"{module} has no `{name}` -- it has "
            f"{', '.join(sorted(a for a in dir(mod) if not a.startswith('_'))[:12])}")


def _parsed():
    """(label, source, tree) for every block that parses. A block that does not
    parse is reported by its own test, never skipped silently."""
    for label, src in _embedded():
        try:
            yield label, src, ast.parse(_SUBST.sub('"<shell>"', src))
        except SyntaxError:
            continue


def test_every_embedded_block_parses():
    """A block the audit cannot read is a block the audit does not cover."""
    bad = []
    for label, src in _embedded():
        try:
            ast.parse(_SUBST.sub('"<shell>"', src))
        except SyntaxError as exc:
            bad.append(f"{label}: {exc}")
    assert not bad, (
        "these embedded Python blocks do not parse, so nothing below checks "
        "them:\n  " + "\n  ".join(bad))


def test_there_is_something_to_audit():
    """The globs must actually find the scripts -- an empty audit passes vacuously.

    Four of these were broken for a whole phase. A refactor that renames the
    directory must fail here rather than quietly auditing nothing.
    """
    labels = [lab for lab, _ in _embedded()]
    assert len(labels) >= 10, (
        f"only {len(labels)} embedded block(s) found: {labels}. `hpc/*.sh` and the "
        f"loose tools import `eccenergy` -- if they moved, move SHELL/LOOSE with them.")
    assert any("tools_bch_sweep_tables" in lab for lab in labels)
    assert any("map_ert_arms" in lab for lab in labels)


def test_every_out_of_package_import_resolves():
    """`from eccenergy... import X` in a script must name something that exists.

    This is the one that would have failed the day phase 2 landed.
    """
    broken = []
    for label, _src, tree in _parsed():
        for module, name, _alias in _imports(tree):
            why = _resolve(module, name)
            if why:
                broken.append(f"{label}: from {module} import {name} -- {why}")
    assert not broken, (
        "out-of-package imports that no longer resolve (the scripts send this to "
        "/dev/null, so ONLY this test says so):\n  " + "\n  ".join(broken))


def test_every_attribute_used_on_an_imported_module_exists():
    """The half that catches a SPLIT, not a move.

    `from eccenergy.arch import fingerprint` keeps working when `archs.py` is cut
    into four; what goes missing is `fingerprint.arch_fingerprint`. Only a name
    bound by an `eccenergy` import is checked, so a local variable of the same
    name is never mistaken for one.
    """
    broken = []
    for label, _src, tree in _parsed():
        bound = {}
        for module, name, alias in _imports(tree):
            target = f"{module}.{name}"
            try:
                bound[alias] = importlib.import_module(target)
            except Exception:                     # noqa: BLE001
                try:
                    bound[alias] = getattr(importlib.import_module(module), name)
                except Exception:                 # noqa: BLE001
                    continue                      # the import test owns this one
        for n in ast.walk(tree):
            if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                    and n.value.id in bound and not hasattr(bound[n.value.id], n.attr)):
                broken.append(f"{label}: {n.value.id}.{n.attr} does not exist "
                              f"(line {n.lineno} of the block)")
    assert not broken, (
        "attributes used on an imported eccenergy module that are not there:\n  "
        + "\n  ".join(sorted(set(broken))))


def test_every_dash_m_target_is_a_real_module():
    """`python3 -m eccenergy.report.dilation_view` must name a runnable module.

    These appear in RUNNING scripts and in the instructions they print, and a
    printed instruction that cannot work is the same defect one step later.
    """
    pattern = re.compile(r"python3 -m (eccenergy[\w.]*)")
    bad = []
    for f in SHELL + LOOSE + [ROOT / "run.sh"]:
        for m in pattern.finditer(f.read_text()):
            target = m.group(1)
            rel = pathlib.Path(target.replace(".", "/"))
            if (ROOT / rel).with_suffix(".py").exists() or (ROOT / rel / "__main__.py").exists():
                continue
            bad.append(f"{f.name}: python3 -m {target} -- no such module")
    assert not bad, "`-m` targets that do not exist:\n  " + "\n  ".join(sorted(set(bad)))
