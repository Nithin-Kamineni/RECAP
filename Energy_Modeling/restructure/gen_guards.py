"""Generate `GUARDS.md` -- every guard and every test-of-structure, in one table.

    make guards            regenerate it
    make guards CHECK=1    fail if it is stale (what tests/contract/test_guards.py runs)

`ProjectRestructure.md` section 6.5: **generated, never hand-written.** A list of
guards kept by hand is a list that is wrong within a month, and the complaint
this answers -- "it is impossible to know what is there" -- is not fixed by a
document that drifts.

WHERE EACH COLUMN COMES FROM. The id, tier, exception type and one-line "refuses"
are declared once, in `eccenergy/settings/guards.py`. The `where` column is NOT
declared: it is found by walking the package for `guards.refusal("<id>", ...)`
and `guards.refuse("<id>", ...)` calls, so a guard that moves file re-documents
itself and a guard that is deleted stops being listed. The two FROZEN sites
(`physics/parity.py`, `study/baseline.py` -- CLAUDE.md) cannot call either
function, so they are located by the `frozen_match` substring the registry
declares and `tests/contract/test_guards.py` checks that substring is still
there.
"""
import ast
import collections
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eccenergy.settings import guards as G          # noqa: E402

OUT = ROOT / "GUARDS.md"
PKG = ROOT / "eccenergy"


def call_sites():
    """id -> ["arch/patch.py:783", ...], by walking the package for the calls."""
    found = collections.defaultdict(list)
    for f in sorted(PKG.rglob("*.py")):
        if "__pycache__" in f.parts or "tests" in f.parts:
            continue
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ("refusal", "refuse") and n.args):
                continue
            first = n.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found[first.value].append(
                    f"{f.relative_to(ROOT / 'eccenergy')}:{n.lineno}")
    return found


def frozen_sites():
    """id -> [site] for the guards in FROZEN files, located by their text."""
    found = {}
    for gid, g in G.GUARDS.items():
        if not g.frozen_match:
            continue
        path = ROOT / g.frozen_file
        for i, line in enumerate(path.read_text().split("\n"), 1):
            if g.frozen_match in line:
                found.setdefault(gid, []).append(
                    f"{pathlib.Path(g.frozen_file).relative_to('eccenergy')}:{i}"
                    " *(frozen)*")
                break
    return found


def contract_tests():
    """The structure tests, for the second table. `tests/contract/` only."""
    rows = []
    for f in sorted((ROOT / "tests" / "contract").glob("test_*.py")):
        tree = ast.parse(f.read_text())
        for n in tree.body:
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"):
                doc = (ast.get_docstring(n) or "").split("\n")[0].strip()
                rows.append((f"tests/contract/{f.name}", n.name, doc))
    return rows


def render():
    calls, frozen = call_sites(), frozen_sites()
    lines = [
        "# GUARDS.md — what is protected, and by what",
        "",
        "**GENERATED. Do not edit.** `make guards` rewrites this file from",
        "`eccenergy/settings/guards.py` and from the call sites in the package;",
        "`tests/contract/test_guards.py` fails when it is stale.",
        "`ProjectRestructure.md` §6 is the design.",
        "",
        "A **guard** runs on *every* run, including a SLURM job at 3am, and stops it.",
        "A **test** runs when you type `pytest` and prints red. Both were invisible",
        "until phase 6; this file is the one screen that answers *what is there*.",
        "",
        "## The tiers, and the only thing that matters about them",
        "",
        "| tier | means | overridable? |",
        "|---|---|---|",
        "| **1 PARSE** | the string does not mean anything | **never** |",
        "| **2 IMPOSSIBLE** | physically or arithmetically cannot hold | **never** |",
        "| **3 COUPLING** | encodes the combinations we thought of | yes — `ECC_ALLOW` |",
        "| **4 DERIVED** | refuses you for setting what is also derived | yes — `ECC_ALLOW` |",
        "",
        "```",
        '   ECC_ALLOW="zero-price,derived-datawidth"     <- names guards, never a blanket off',
        "```",
        "",
        "An override is **recorded**: it lands on the run manifest as `guard_overrides`",
        "and on the figure's caveat list. You cannot ablate by accident, and you cannot",
        "publish an ablation without the figure saying it is one.",
        "",
    ]

    tiers = collections.Counter(g.tier for g in G.GUARDS.values())
    sites_n = sum(len(v) for v in calls.values()) + sum(len(v) for v in frozen.values())
    lines += [
        f"**{len(G.GUARDS)} invariants over {sites_n} refusal sites.** "
        + ", ".join(f"tier {t} ({G.TIER_NAMES[t]}): {tiers[t]}" for t in sorted(tiers)),
        "",
    ]

    for tier in sorted(tiers):
        lines += [
            f"## Tier {tier} — {G.TIER_NAMES[tier]}"
            + ("" if tier in G.OVERRIDABLE else "  *(never overridable)*"),
            "",
            "| id | raises | refuses | where |",
            "|---|---|---|---|",
        ]
        for gid, g in sorted(G.GUARDS.items()):
            if g.tier != tier:
                continue
            where = calls.get(gid, []) + frozen.get(gid, [])
            lines.append(
                f"| `{gid}` | `{g.raises.__name__}` | {g.refuses} | "
                + ("<br>".join(f"`{w}`" for w in where) or "**NO SITE**") + " |")
            if g.note:
                lines.append(f"| | | ↳ {g.note} | |")
        lines.append("")

    lines += ["## Tests of structure — `tests/contract/`", "",
              "Not guards: these run under `pytest`, not on every job. They hold the",
              "*shape* of the project still — the layer rule, the settings contract,",
              "this file's freshness, and every script outside the package that imports it.",
              "", "| file | test | holds |", "|---|---|---|"]
    for f, name, doc in contract_tests():
        lines.append(f"| `{f}` | `{name}` | {doc} |")
    lines.append("")
    return "\n".join(lines)


def main(argv):
    text = render()
    if "--check" in argv:
        cur = OUT.read_text() if OUT.exists() else ""
        if cur != text:
            sys.stderr.write("GUARDS.md is STALE -- regenerate it with `make guards`.\n")
            return 1
        print(f"GUARDS.md is current ({len(G.GUARDS)} guards).")
        return 0
    OUT.write_text(text)
    print(f"{OUT.relative_to(ROOT)}: {len(G.GUARDS)} guards, "
          f"{len(contract_tests())} contract tests")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
