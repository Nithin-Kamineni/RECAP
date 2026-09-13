"""The layer rule, enforced. `ProjectRestructure.md` sections 1, 4.1 and 4.2.

> **A module may import only from strictly lower layers.** Enforcing it makes
> import cycles impossible.

Phase 2 created the layers; this is what stops them dissolving again. It reads
the source and never imports it, so it is fast, needs no environment, and cannot
be fooled by an import that only runs on some paths.

WHY THE EXCEPTION LIST IS THE POINT. The rule does not hold today -- there is one
import cycle and it is the reason `config.py`, `archs.py` and `recon.py` cannot
be changed independently (section 2.3). Eight edges point upward -- phase 3 removed eight more, and re-spelled
the rest at the module granularity the split created -- and every one of them is
DECLARED below with the phase that removes it. So:

  * a NEW upward edge fails immediately -- the rule cannot rot further;
  * a declared edge that is GONE also fails, which makes the list shrink rather
    than accumulate. Phase 3 emptied everything but the cycle; PHASE 4 IS
    FINISHED WHEN THE LIST IS EMPTY.

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
    ("contracts.", 0),          # the shared types
    ("paths", 0),               # the ONLY resolver of paths
    ("config", 0),              # -> settings/ in phase 4
    ("physics.", 1),            # parity, embedding, widths, packing, the DRAM price
    ("arch.", 2),               # the designs, their weight paths and their arms
    ("toolchain.", 3),          # Timeloop in, its output parsed, what is charged after
    ("study.", 4),              # the arms, the placement study, Task 4
    ("report.", 5),             # the renderers, and the drivers that draw
    ("__main__", 6),            # the CLI: may import anything
    ("__init__", 0),            # the package docstring; imports nothing
)

#: (importer, imported) -> why it points the wrong way, and what removes it.
#: DELETE A LINE WHEN THE PHASE THAT OWNS IT LANDS. A stale entry fails.
KNOWN_UPWARD = {
    # ---- the import cycle. `Config` resolves things it should be given. ----
    ("config", "physics.widths"):
        "Config resolves q from THE WIDTH TABLE at construction. Phase 4: "
        "CodeSettings holds q and physics is handed it.",
    ("config", "physics.embedded"):
        "Config asks EmbeddedLayout how many weights a codeword holds. Phase 4.",
    ("config", "arch.load"):
        "Config reads mac_candidates off the design to pick a MAC price. "
        "Phase 4: ArchSettings is data, not a reader.",
    ("config", "arch.fingerprint"):
        "Config asks for the effective variant slug while resolving itself. "
        "THE CYCLE (section 2.3). Phase 4.",
    ("config", "arch.placements"):
        "Config validates ECC_RECON_PLACEMENTS against the design's boundaries. "
        "Phase 4: the validation moves to the arch layer, which owns the table.",
    ("config", "arch.arms"):
        "Config resolves ECC_RECON_ERT_ARM through mapper_arm_spec. THE CYCLE, "
        "and the half phase 3 could not cut: the arm list is what a knob names. "
        "Phase 4.",
    # ---- two arch modules price a reconstruction engine ----
    ("arch.fingerprint", "study.stacks"):
        "ert_bump() reaches for the reconstruction energy to build the ERT bump "
        "the mapper is given. Phase 4: the bump is computed from settings, so "
        "the DC table is handed in rather than looked up. (Was archs -> ecc.)",
    ("arch.arms", "study.stacks"):
        "ert_leak_delta_pj() needs the same DC table, to decide whether a bump "
        "has a per-cycle row at all. Phase 4, with the edge above.",
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
    """L0-L3 -- config, physics, arch, toolchain -- must not reach into L4/L5.

    Stated separately because it is the half that actually blocks work: an
    architecture cannot be added as data while `config` and `archs` still call
    up into `recon`. It is expected to fail until phase 4, so it is marked
    xfail; when phase 4 lands it passes and the marker comes off.
    """
    offenders = sorted((s, d) for s, d in _upward() if layer_of(s) <= 3)
    if offenders:
        pytest.xfail("the import cycle is still here: " +
                     ", ".join(f"{s}->{d}" for s, d in offenders))


def test_the_exception_list_is_documented():
    """Every declared exception says WHY and names the phase that removes it."""
    for key, why in KNOWN_UPWARD.items():
        assert "hase" in why, f"{key}: no phase named -- {why!r}"
        assert len(why) > 40, f"{key}: reason too thin to act on -- {why!r}"
