r"""`arch/patch.py`'s three text primitives -- the ones a PROTECTED rule rests on.

CLAUDE.md, 2026-09-13: **"A COMMENT IS NEVER A DECLARATION. Read geometry through
`arch.patch.uncommented()`, write it through `arch.patch.write_attr()`."**

That rule is there because every geometry regex in the module used to match the
RAW text, so a `depth:` written inside a comment was read as the attribute -- and
`re.sub(..., count=1)` then rewrote THE COMMENT and left the attribute alone. One
explanatory comment naming a geometry it was not declaring turned `depth: 256`
into 98,304 bits where 16,512 were intended: a six-fold capacity error, on a real
SLURM run, with nothing on stdout to say so.

`uncommented()`, `read_attr()` and `write_attr()` are what enforce it and none of
them had a test. These are the mutations that bug WAS.
"""
import pytest

from eccenergy.arch import patch

#: The shape that caused it: a comment that NAMES a geometry it is not declaring.
TRAP = """\
    filter_glb:
      # the paper's two banks of 512 x 64b would be `depth: 1024`
      depth: 256
      width: 96
"""


def test_a_comment_that_names_a_geometry_is_not_the_declaration():
    """THE BUG, as one assertion. 1024 is in the text; 256 is what is declared."""
    assert "depth: 1024" in TRAP
    assert patch.read_attr(TRAP, "depth") == 256


def test_blanking_preserves_every_position_so_spans_stay_valid():
    """`uncommented()` blanks rather than deletes, so a caller may search the
    masked copy and splice into the REAL one. If it deleted, every match offset
    past the first comment would point at the wrong character."""
    masked = patch.uncommented(TRAP)
    assert len(masked) == len(TRAP)
    for i, ch in enumerate(TRAP):
        assert ch == "\n" or masked[i] in (ch, " ")
    assert "1024" not in masked and "depth: 256" in masked


def test_the_two_comment_strippers_differ_and_this_records_which():
    """RECORDED DIVERGENCE, not a defect to fix silently.

    This project has TWO comment strippers and they are not the same function:

      `patch.uncommented()`             `#[^\n]*`, blanked. NAIVE: a `#` inside a
                                        quoted YAML scalar blanks the rest of the
                                        line.
      `fingerprint.hashable_arch_text()` QUOTE-AWARE, because a `#` inside a
                                        scalar is data and hashing it as a comment
                                        would re-key every mapper cache.

    IT DOES NOT BITE TODAY, and that is measured rather than assumed: no
    `archs/*/…yaml` that `uncommented()` reads has a `#` inside a quoted scalar
    (the one in the tree is `archs/_shared/provenance.yaml`, which carries
    citations and no geometry). It would take a quoted string AND a geometry
    attribute on the SAME line to lose an attribute, and YAML written one
    key per line cannot produce that.

    So this pins the behaviour as it IS. If somebody makes `uncommented()`
    quote-aware, this test fails and they read this paragraph -- which is the
    point. Changing it is a geometry change and belongs to whoever declares it,
    not to a test-backfilling phase.
    """
    from eccenergy.arch import fingerprint

    text = 'name: "weights # not a comment"\ndepth: 8\n'
    assert patch.read_attr(text, "depth") == 8, "the attribute is on its own line"
    assert "not a comment" not in patch.uncommented(text), (
        "patch.uncommented() is NAIVE -- see this test's docstring")
    assert "not a comment" in fingerprint.hashable_arch_text(text), (
        "the FINGERPRINT's stripper is quote-aware, and must stay so")


def test_write_attr_rewrites_the_declaration_and_leaves_the_comment_alone():
    """The other half of the bug: the WRITE landed on the comment too."""
    out = patch.write_attr(TRAP, "depth", 43)
    assert patch.read_attr(out, "depth") == 43
    assert "would be `depth: 1024`" in out, "the comment must be untouched"
    assert "depth: 43" in out
    assert len(out.splitlines()) == len(TRAP.splitlines())


def test_write_attr_refuses_when_there_is_nothing_uncommented_to_rewrite():
    """A geometry rewrite that lands on nothing is the SILENT half of the bug.
    Refusing is what turns it into a stack trace instead of a wrong energy."""
    only_a_comment = "    # depth: 1024\n    width: 96\n"
    with pytest.raises(ValueError, match="no uncommented `depth:`"):
        patch.write_attr(only_a_comment, "depth", 43)


def test_read_attr_is_none_when_the_key_is_absent_rather_than_zero():
    """None and 0 must not be confused: one means "this level declares no such
    attribute", the other is a declared zero."""
    assert patch.read_attr("width: 96\n", "depth") is None
    assert patch.read_attr("depth: 0\n", "depth") == 0


def test_only_the_first_uncommented_occurrence_is_rewritten():
    """`write_attr` is used per LEVEL, on that level's slice of the YAML. If it
    rewrote every match it would flatten two levels onto one geometry."""
    two = "  a:\n    depth: 256\n  b:\n    depth: 512\n"
    out = patch.write_attr(two, "depth", 43)
    assert "depth: 43" in out and "depth: 512" in out
    assert out.count("depth: 43") == 1


def test_a_key_that_is_a_prefix_of_another_is_not_matched():
    """`\\b` anchors the key: reading `depth` must not find `max_depth`."""
    assert patch.read_attr("max_depth: 999\ndepth: 8\n", "depth") == 8
