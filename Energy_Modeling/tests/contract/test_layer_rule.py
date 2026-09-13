"""The layer rule, enforced. `ProjectRestructure.md` sections 1, 4.1 and 4.2.

> **A module may import only from strictly lower layers.** Enforcing it makes
> import cycles impossible.

Phase 2 created the layers; this is what stops them dissolving again. It reads
the source and never imports it, so it is fast, needs no environment, and cannot
be fooled by an import that only runs on some paths.

WHY THE EXCEPTION LIST IS THE POINT. The rule does not hold today -- there is one
import cycle and it is the reason `config.py`, `archs.py` and `recon.py` cannot
be changed independently (section 2.3). NO EDGE POINTS UPWARD ANY MORE. Phase 2 declared sixteen, phase 3 removed eight,
and phase 4 removed the rest: the knobs went down into `settings/` (L0) and the
RESOLUTION -- which needs the designs -- went up into `config.py` (L3), so the
cycle that made `config.py`, `archs.py` and `recon.py` one unit is gone rather
than hidden behind a lazy import. So:

  * a NEW upward edge fails immediately -- the rule cannot rot further;
  * a declared edge that is GONE also fails, which is what made the list shrink
    to nothing instead of accumulating.

That is the same shape as the `fp-` fingerprint rule: assert the invariant, list
the exceptions, and make the list expensive to leave alone.
"""
import ast
import collections
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2] / "eccenergy"

#: (dotted-name prefix, layer). First match wins, so exact names come first.
#: A prefix ending in "." matches a package; anything else must match exactly.
LAYERS = (
    ("contracts.", 0),          # the shared types, and ConfigError
    ("paths", 0),               # the ONLY resolver of paths
    ("settings.", 0),           # the KNOBS: six frozen groups, read from env once
    ("physics.", 1),            # parity, embedding, widths, packing, the DRAM price
    ("arch.", 2),               # the designs, their weight paths and their arms
    # `config.py` RESOLVES the knobs against the designs -- the ERT arm has to
    # exist on THIS chip, the axis lists have to name real models -- so it sits
    # ABOVE `arch/`, not beside the settings it is built from. That is what ended
    # the cycle in section 2.3: `arch/` and `physics/` do not import it, and the
    # lazy imports that used to hide the deadlock are plain top-level ones now.
    ("config", 3),
    ("toolchain.", 4),          # Timeloop in, its output parsed, what is charged after
    ("study.", 5),              # the arms, the placement study, Task 4
    ("report.", 6),             # the renderers, and the drivers that draw
    ("__main__", 7),            # the CLI: may import anything
    ("__init__", 0),            # the package docstring; imports nothing
)

#: (importer, imported) -> why it points the wrong way, and what removes it.
#: DELETE A LINE WHEN THE PHASE THAT OWNS IT LANDS. A stale entry fails.
KNOWN_UPWARD = {
    # EMPTY SINCE PHASE 4, 2026-09-13, and that is the point of the file.
    #
    # It held sixteen edges when phase 2 wrote it, eight after phase 3, and none
    # now. THE RULE IS THE DEFAULT AGAIN: an upward import fails the suite where
    # it is written, rather than being argued about later.
    #
    # If you must add one, it goes here with the phase that removes it -- and
    # `test_every_declared_exception_is_still_real` then fails the day it is
    # fixed, so the list shrinks instead of accumulating. What emptied the last
    # two was moving the DC reconstruction table out of `study/stacks.py` into
    # `physics/recon_dc.py`: `arch/` prices an ERT bump from it, and reaching UP
    # into a driver for that was the last cycle left.
}
def _modules():
    out = {}
    for f in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in f.parts or "tests" in f.parts:
            continue
        out[".".join(f.relative_to(ROOT).with_suffix("").parts)] = f
    return out


def layer_of(module):
    """The layer of a dotted module name, or None if nothing claims it."""
    stem = module[:-len(".__init__")] if module.endswith(".__init__") else module
    for prefix, level in LAYERS:
        if prefix.endswith(".") and (stem.startswith(prefix) or stem == prefix[:-1]):
            return level
        if stem == prefix:
            return level
    return None


def _edges():
    """Every intra-package import, as (importer, imported, lazy)."""
    mods = _modules()
    out = []
    for name, path in mods.items():
        if name.endswith("__init__"):
            continue
        tree = ast.parse(path.read_text())
        inside = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inside |= set(range(n.lineno, (n.end_lineno or n.lineno) + 1))
        pkg = tuple(name.split(".")[:-1])

        def add(target, lineno):
            if target in mods and target != name:
                out.append((name, target, lineno in inside))

        for n in ast.walk(tree):
            if not isinstance(n, ast.ImportFrom):
                continue
            if n.level:
                base = list(pkg[:len(pkg) - (n.level - 1)]) if n.level > 1 else list(pkg)
                mod = n.module or ""
                if mod:
                    cand = ".".join(base + mod.split("."))
                    if cand in mods:
                        add(cand, n.lineno)
                    else:
                        for a in n.names:
                            add(f"{cand}.{a.name}", n.lineno)
                else:
                    for a in n.names:
                        add(".".join(base + [a.name]), n.lineno)
            elif n.module == "eccenergy":
                for a in n.names:
                    add(a.name, n.lineno)
            elif n.module and n.module.startswith("eccenergy."):
                cand = n.module[len("eccenergy."):]
                if cand in mods:
                    add(cand, n.lineno)
                else:
                    for a in n.names:
                        add(f"{cand}.{a.name}", n.lineno)
    return out


def _upward():
    """Distinct (importer, imported) pairs that point at a higher layer."""
    pairs = collections.OrderedDict()
    for src, dst, lazy in _edges():
        ls, ld = layer_of(src), layer_of(dst)
        if ls is None or ld is None:
            continue
        if ld > ls:
            pairs.setdefault((src, dst), []).append(lazy)
    return pairs


def test_every_module_has_a_layer():
    """A new module must be given a layer, not left to fall through the rule."""
    homeless = sorted(m for m in _modules() if layer_of(m) is None)
    assert not homeless, (
        "these modules match no entry in LAYERS, so the layer rule says nothing "
        f"about them: {homeless}. Add them to LAYERS with the layer they belong to.")


def test_no_module_imports_from_a_higher_layer():
    """The rule itself. Every upward edge must be declared, with a reason."""
    undeclared = sorted(k for k in _upward() if k not in KNOWN_UPWARD)
    assert not undeclared, (
        "NEW upward imports -- the layer rule is being broken further:\n" +
        "\n".join(f"  {s} (L{layer_of(s)}) -> {d} (L{layer_of(d)})"
                  for s, d in undeclared) +
        "\nEither import from a lower layer, or add the edge to KNOWN_UPWARD "
        "with the phase that removes it.")


def test_every_declared_exception_is_still_real():
    """A fixed violation must be DELETED from the list, not left as folklore.

    This is the half that makes the list shrink. Phase 3 and phase 4 are done
    when `KNOWN_UPWARD` is empty.
    """
    live = _upward()
    stale = sorted(k for k in KNOWN_UPWARD if k not in live)
    assert not stale, (
        "these upward imports are GONE -- delete them from KNOWN_UPWARD:\n" +
        "\n".join(f"  {s} -> {d}" for s, d in stale))


def test_the_layer_rule_holds_below_the_drivers():
    """L0-L4 -- settings, physics, arch, config, toolchain -- may not reach up.

    Stated separately because it is the half that actually blocks work: an
    architecture cannot be added as DATA while `config` and `arch` call up into a
    driver. It was xfail until phase 4 and is a plain assertion now -- the rule
    holds, and a regression here is a failure rather than an expected one.
    """
    offenders = sorted((s, d) for s, d in _upward() if layer_of(s) <= 4)
    assert not offenders, (
        "the import cycle is back: " + ", ".join(f"{s}->{d}" for s, d in offenders))


def test_the_exception_list_is_documented():
    """Every declared exception says WHY and names the phase that removes it."""
    for key, why in KNOWN_UPWARD.items():
        assert "hase" in why, f"{key}: no phase named -- {why!r}"
        assert len(why) > 40, f"{key}: reason too thin to act on -- {why!r}"
