"""Move modules between packages and repair every import that names them.

WRITTEN FOR PHASE 2 of ProjectRestructure and KEPT FOR PHASE 3, which moves the
halves of the six files it splits. `MOVES` below is phase 2's, left as the record
of exactly what that phase did; edit it and re-run for the next one.

WHAT IT DOES THAT sed CANNOT. It resolves every intra-package import to an
absolute module name FIRST, then re-spells it for the file's NEW location -- so
`from .paths import ROOT` in a file that moves down a level becomes
`from ..paths import ROOT` even though `paths` never moved. It splits a
`from . import a, b` whose targets land in different packages, preserves an
alias and a one-line trailing comment, and REFUSES a multi-line import that
carries a comment rather than guessing where the comment goes.

Nothing but import statements changes -- plus the two module RENAMES the plan
asks for (`code_widths` -> `physics.widths`, `ecc` -> `study.stacks`), whose new
binding name has to be carried into every qualified reference.

Run with --plan to see what it would do, or --apply to do it.
"""
import argparse, ast, collections, pathlib, subprocess, sys

ROOT = pathlib.Path("eccenergy")

#: old dotted name (under `eccenergy.`) -> new dotted name.
#: The six that PHASE 3 SPLITS stay where they are: config, archs, recon,
#: timeloop, experiments.recon, experiments.dilation. A module that is about to
#: be cut into four layers cannot be filed under one of them first.
MOVES = {
    # L1  physics -- what the code and the layout are, as arithmetic
    "parity":                "physics.parity",
    "embedded":              "physics.embedded",
    "code_widths":           "physics.widths",
    "baseline_dram":         "physics.baseline_dram",
    # L2  arch -- the designs and the workloads they run
    "workloads":             "arch.workloads",
    "generate":              "arch.generate",
    # L3  toolchain -- Timeloop, its outputs, and what is charged after it
    "noc_post":              "toolchain.noc_post",
    "latency_post":          "toolchain.latency_post",
    "results_store":         "toolchain.results_store",
    "experiments.ert_probe": "toolchain.ert_probe",
    # L4  study -- the arms, the drivers, the energy bookkeeping
    "ecc":                   "study.stacks",
    "energy":                "study.energy",
    "experiments.common":    "study.common",
    "experiments.baseline":  "study.baseline",
    "experiments.embedded":  "study.embedded",
    "experiments.audit":     "study.audit",
    "experiments.diagnose":  "study.diagnose",
    "experiments.validate":  "study.validate",
    "experiments.sweep":     "study.sweep",
    "experiments.panels":    "study.panels",
    # L5  report -- the renderers
    "plots.stacked":         "report.stacked",
    "plots.panels":          "report.panels",
    "plots.style":           "report.style",
}

#: A module whose LEAF name changes rebinds every `oldleaf.attr` in the files
#: that import it bare.
RENAMED_LEAF = {old.split(".")[-1]: new.split(".")[-1]
                for old, new in MOVES.items()
                if old.split(".")[-1] != new.split(".")[-1]}


def modules():
    """Every module in the package, old dotted name -> path."""
    out = {}
    for f in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        out[".".join(f.relative_to(ROOT).with_suffix("").parts)] = f
    return out


def new_name(old):
    return MOVES.get(old, old)


def pkg_of(dotted):
    parts = dotted.split(".")
    return tuple(parts[:-1])


def _common(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def relspec(from_pkg, target, as_module):
    """`from X import ...` spelling of `target` seen from package `from_pkg`.

    `as_module=True` puts the target module itself after the dots (so names can
    be imported OUT of it); False puts its parent there (so the module itself is
    the imported name).
    """
    t = tuple(target.split("."))
    path = t if as_module else t[:-1]
    n = _common(from_pkg, path)
    level = len(from_pkg) - n + 1
    return "." * level + ".".join(path[n:])


def resolve(node, file_pkg, known):
    """Absolute (package-relative) targets of one import node, or None."""
    if isinstance(node, ast.Import):
        out = []
        for a in node.names:
            if a.name.startswith("eccenergy."):
                out.append((a.name[len("eccenergy."):], a.asname, "abs-import"))
        return out or None
    if not isinstance(node, ast.ImportFrom):
        return None
    if node.level:
        base = list(file_pkg[:len(file_pkg) - (node.level - 1)]) if node.level > 1 else list(file_pkg)
        if node.level - 1 > len(file_pkg):
            return None
        mod = node.module or ""
        if mod:
            cand = ".".join(base + mod.split("."))
            if cand in known:                       # `from .mod import NAME`
                return [(cand, None, "names")]
            if cand + ".__init__" in known:         # `from .pkg import mod, mod`
                subs = [f"{cand}.{a.name}" for a in node.names]
                if all(s in known for s in subs):
                    return [(s, a.asname, "module")
                            for s, a in zip(subs, node.names)]
            return None
        return [(".".join(base + [a.name]), a.asname, "module")
                for a in node.names]                 # `from . import mod`
    if node.module == "eccenergy":
        return [(a.name, a.asname, "module") for a in node.names]
    if node.module and node.module.startswith("eccenergy."):
        cand = node.module[len("eccenergy."):]
        if cand in known:
            return [(cand, None, "names")]
        if cand + ".__init__" in known:
            subs = [f"{cand}.{a.name}" for a in node.names]
            if all(s in known for s in subs):
                return [(s, a.asname, "module") for s, a in zip(subs, node.names)]
    return None


def rewrite(path, old_dotted, known, apply):
    """Rewrite every intra-package import in one file. Returns (text, notes)."""
    src = path.read_text()
    lines = src.splitlines(keepends=True)
    file_pkg_new = pkg_of(new_name(old_dotted))
    file_pkg_old = pkg_of(old_dotted)
    tree = ast.parse(src)
    edits = []          # (start_idx, end_idx, [new lines])
    notes = []
    rebound = set()     # leaf names whose binding changed in THIS file

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        targets = resolve(node, file_pkg_old, known)
        if not targets:
            continue
        absolute = (isinstance(node, ast.ImportFrom) and not node.level) or \
                   isinstance(node, ast.Import)
        span = lines[node.lineno - 1:node.end_lineno]
        indent = span[0][:len(span[0]) - len(span[0].lstrip())]
        # A COMMENT INSIDE AN IMPORT IS LOAD-BEARING HERE -- six of them say why
        # the import is lazy, which is the whole subject of phase 4. A one-line
        # import keeps its trailing comment; a multi-line one is REFUSED and
        # listed, because splicing a comment back into a re-grouped statement
        # is guesswork.
        comment = ""
        if len(span) == 1:
            code, _, after = span[0].rstrip("\n").partition("#")
            if after and "\"" not in code and "'" not in code:
                comment = "  # " + after.strip()
        elif any("#" in s for s in span):
            notes.append("    ! REFUSED (comment inside a multi-line import): "
                         + " ".join(s.strip() for s in span))
            continue

        out = []
        kind = targets[0][2]
        if kind == "names":                          # from <mod> import A, B
            old_t = targets[0][0]
            new_t = new_name(old_t)
            if new_t == old_t and not absolute and file_pkg_new == file_pkg_old:
                continue                             # nothing moved; leave the text alone
            names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "")
                              for a in node.names)
            spec = f"eccenergy.{new_t}" if absolute else relspec(file_pkg_new, new_t, True)
            out.append(f"{indent}from {spec} import {names}\n")
        else:                                        # from . import a, b  /  from eccenergy import a
            groups = collections.OrderedDict()
            changed = False
            for old_t, asname, _ in targets:
                new_t = new_name(old_t)
                changed = changed or new_t != old_t
                spec = (f"eccenergy.{'.'.join(pkg_of(new_t))}".rstrip(".")
                        if absolute else relspec(file_pkg_new, new_t, False))
                leaf = new_t.split(".")[-1]
                bind = asname or old_t.split(".")[-1]
                if leaf != bind and not asname:
                    rebound.add((old_t.split(".")[-1], leaf))
                    bind = None
                groups.setdefault(spec, []).append(
                    leaf + (f" as {asname}" if asname else ""))
            if not changed and file_pkg_new == file_pkg_old:
                continue
            for spec, names in groups.items():
                out.append(f"{indent}from {spec} import {', '.join(sorted(names))}\n")
        if comment and out:
            out[-1] = out[-1].rstrip("\n") + comment + "\n"
        if out != span:
            edits.append((node.lineno - 1, node.end_lineno, out))

    if not edits and not rebound:
        return None, notes
    for start, end, new in sorted(edits, reverse=True):
        lines[start:end] = new
    text = "".join(lines)
    for old_leaf, new_leaf in sorted(rebound):
        import re
        text, n = re.subn(rf"\b{old_leaf}\.", f"{new_leaf}.", text)
        if n:
            notes.append(f"    rebound {old_leaf}.* -> {new_leaf}.*  ({n} sites)")
    if apply:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(path)
    return text, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-git", action="store_true")
    args = ap.parse_args()

    known = modules()
    # 1. move the files
    for old, new in MOVES.items():
        src = ROOT / (old.replace(".", "/") + ".py")
        dst = ROOT / (new.replace(".", "/") + ".py")
        assert src.is_file(), src
        print(f"  mv  {src}  ->  {dst}")
        if args.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if args.no_git:
                src.rename(dst)
            else:
                subprocess.run(["git", "mv", str(src), str(dst)], check=True)
    # 2. rewrite imports everywhere (tests included)
    print("\nimports:")
    for old_dotted, path in known.items():
        new_path = ROOT / (new_name(old_dotted).replace(".", "/") + ".py")
        target = new_path if args.apply else path
        if not target.exists():
            continue
        _, notes = rewrite(target, old_dotted, known, args.apply)
        if notes:
            print(f"  {old_dotted}")
            for n in notes:
                print(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
