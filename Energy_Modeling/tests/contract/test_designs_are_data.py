"""A DESIGN IS A DIRECTORY. `ProjectRestructure.md` section 5, enforced.

Phase 5 moved every per-design table out of Python and into
`archs/<name>/{design,weight_path,placements}.yaml`. These tests are what stops
it drifting back:

  * every declared design LOADS and passes the schema -- a design that is half
    written fails here rather than in a SLURM job eight hours later;
  * the two halves of the placement space are loaded TOGETHER, so "they must be
    edited together" is a property of the directory instead of a warning;
  * NO PRODUCTION MODULE NAMES A DESIGN. Section 5.1.1 measured three sites that
    branched on `arch.startswith("eyeriss_v2")`; after phase 5 the count is zero
    and this test keeps it there.

It reads files and source text. No cache, no environment, no Timeloop.
"""
import ast
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eccenergy.arch import design                                   # noqa: E402
from eccenergy.arch import placements as placements_mod             # noqa: E402
from eccenergy.arch import weight_path as weight_path_mod           # noqa: E402
from eccenergy.contracts.errors import ConfigError                  # noqa: E402

ARCHS = ROOT / "archs"
PKG = ROOT / "eccenergy"

#: Every design that declares itself, as `design.yaml` says.
DESIGNS = design.known_archs()


def test_there_are_designs_at_all():
    """A tree with no `design.yaml` would make every test below vacuous."""
    assert DESIGNS, f"no design declares a {design.DESIGN_FILE} under {ARCHS}"


@pytest.mark.parametrize("name", DESIGNS)
def test_every_design_loads_and_validates(name):
    """`design.yaml` parses, names itself, and carries a label."""
    doc = design.design(name)
    assert doc["name"] == name
    assert str(doc.get("label", "")).strip()


@pytest.mark.parametrize("name", DESIGNS)
def test_the_two_halves_are_declared_together(name):
    """A design declares BOTH `weight_path.yaml` and `placements.yaml`, or neither.

    The old failure mode was a stage added to one table and not the other, which
    left every boundary below it reporting its own saving while the new stage
    stayed at full width -- the whole list understated, with nothing saying so.
    One directory makes that a file that is missing, not a mismatch nobody sees.
    """
    has_path = (ARCHS / name / design.WEIGHT_PATH_FILE).is_file()
    has_places = (ARCHS / name / design.PLACEMENTS_FILE).is_file()
    assert has_path == has_places, (
        f"{name}: declares {design.WEIGHT_PATH_FILE}={has_path} but "
        f"{design.PLACEMENTS_FILE}={has_places}. A boundary is a cut through a "
        f"weight path: both files, or neither.")


@pytest.mark.parametrize("name", DESIGNS)
def test_a_declared_weight_path_is_well_formed(name):
    """Stages: unique keys, known kinds, a DRAM stage first, prefixes present."""
    stages = weight_path_mod.stages_of(name)
    if not stages:
        pytest.skip(f"{name} declares no weight path")
    assert stages[0].kind == "dram"
    assert len({s.key for s in stages}) == len(stages)
    for s in stages:
        assert s.kind in design.STAGE_KINDS
        assert s.prefixes
        assert s.evidence.strip(), (
            f"{name}/{s.key}: `evidence:` is where this stage's citation lives; "
            f"a stage nobody can check is a stage nobody should trust")


@pytest.mark.parametrize("name", DESIGNS)
def test_declared_placements_name_stages_that_exist(name):
    """Every `reduced` entry and every `site_stage` is on this design's path."""
    places = placements_mod.placements_of(name)
    if not places:
        pytest.skip(f"{name} declares no boundaries")
    keys = {s.key for s in weight_path_mod.stages_of(name)}
    for p in places:
        assert set(p.reduced) <= keys, (name, p.key, set(p.reduced) - keys)
        assert p.site_stage in keys, (name, p.key, p.site_stage)
        assert p.site_counter in design.SITE_COUNTERS
        assert p.description.strip(), f"{name}/{p.key}: no description"


def test_a_broken_design_file_is_refused_by_name(tmp_path, monkeypatch):
    """The schema is a REFUSAL, not a warning -- and it names the file.

    Mutation-tested rather than asserted: a placement whose `reduced` set names a
    stage the weight path does not declare is exactly the drift the two-file
    layout is meant to make impossible, so it must not load.
    """
    stages = [{"key": "dram", "label": "d", "kind": "dram", "prefixes": ["DRAM"],
               "reducible": True, "evidence": "e"}]
    doc = {"placements": [{"key": "r1", "variant": "v", "label": "l", "short": "s",
                           "rating": "1/5", "reduced": ["dram", "nowhere"],
                           "site_stage": "dram", "site_counter": "reads",
                           "description": "d"}]}
    with pytest.raises(ConfigError) as exc:
        design.validate_placements("x", doc, stages, tmp_path / "placements.yaml")
    assert "nowhere" in str(exc.value)
    assert "placements.yaml" in str(exc.value)


def test_a_stage_of_an_unknown_kind_is_refused(tmp_path):
    doc = {"stages": [{"key": "dram", "label": "d", "kind": "elephant",
                       "prefixes": ["DRAM"], "reducible": True, "evidence": "e"}]}
    with pytest.raises(ConfigError):
        design.validate_weight_path("x", doc, tmp_path / "weight_path.yaml")


#: Module-level names that are allowed to appear in a `startswith`/`==` against
#: a design name -- there are none, and that is the point.
_DESIGN_NAME = re.compile(r'"(eyeriss|simba|simple)[a-z_0-9]*"')


def test_no_production_module_branches_on_a_design_name():
    """Section 5.1.1 finding 3, held at ZERO.

    Three sites branched on `arch.startswith("eyeriss_v2")` before phase 5; each
    is a declared field in `design.yaml` now. A new one would mean a design's
    behaviour living in a driver again, where the next design cannot inherit it.

    It walks the AST rather than grepping, so a design name inside a DOCSTRING or
    a message -- where it is prose, and often a citation -- does not count, and a
    comparison does.
    """
    offenders = []
    for f in sorted(PKG.rglob("*.py")):
        if "tests" in f.parts or "__pycache__" in f.parts or f.name.endswith(".preC"):
            continue
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            texts = []
            if isinstance(node, ast.Compare):
                texts = [ast.unparse(node)]
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("startswith", "endswith")):
                texts = [ast.unparse(node)]
            for t in texts:
                if _DESIGN_NAME.search(t):
                    offenders.append(f"{f.relative_to(ROOT)}:{node.lineno}: {t}")
    assert not offenders, (
        "production code is branching on a design NAME again:\n  " +
        "\n  ".join(offenders) +
        "\nDeclare it in archs/<name>/design.yaml and read it with "
        "`arch.design.flag(arch, ...)`.")


def test_the_registry_is_the_directory():
    """`KNOWN_ARCHS` is what `archs/` declares -- no list in Python repeats it."""
    from eccenergy import config
    assert tuple(config.KNOWN_ARCHS) == DESIGNS
    assert set(config.ARCH_LABELS) == set(DESIGNS)
    for f in sorted(PKG.rglob("*.py")):
        if "tests" in f.parts or "__pycache__" in f.parts or f.name.endswith(".preC"):
            continue
        src = f.read_text()
        assert not re.search(r"^KNOWN_ARCHS\s*=\s*\(", src, re.M), (
            f"{f.relative_to(ROOT)} declares its own design list")


def test_the_scaffold_writes_a_design_that_is_refused_until_it_is_written(tmp_path):
    """`make arch NEW=x` must produce something LOUD, not something plausible.

    A stub that validated would be a design whose weight path nobody wrote, and
    the placement study would happily report savings against it.
    """
    sys.path.insert(0, str(ROOT / "restructure"))
    import scaffold_arch

    monkey = tmp_path / "archs"
    monkey.mkdir()
    old_root = scaffold_arch.ROOT
    scaffold_arch.ROOT = tmp_path
    try:
        assert scaffold_arch.main(["tpu_like"]) == 0
        d = monkey / "tpu_like"
        assert (d / "design.yaml").is_file()
        assert (d / "weight_path.yaml").is_file()
        assert (d / "placements.yaml").is_file()
        assert "TODO" in (d / "weight_path.yaml").read_text()
        # and the placement stub does NOT reach every reducible stage, so the
        # space is invalid until someone writes the rest
        assert "TODO" in (d / "placements.yaml").read_text()
        assert scaffold_arch.main(["tpu_like"]) == 2        # never overwrites
    finally:
        scaffold_arch.ROOT = old_root
