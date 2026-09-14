"""`settings/env.py` -- the only module that reads `os.environ`, parser by parser.

It had no test of its own: every knob in the study passes through these eight
functions, and what they are FOR is refusing rather than guessing. A silent
default where a refusal belongs is how a run ends up at a configuration nobody
chose, and the failure would show up as an energy, not as an error.

Each refusal is checked by its GUARD ID (`GUARDS.md`, tier 1 PARSE), not by its
message text, so rewording a message does not fail the suite and DELETING the
guard does.
"""
import os

import pytest

from eccenergy.contracts.errors import ConfigError
from eccenergy.settings import env, guards


def set_env(**kw):
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ------------------------------------------------------------------- booleans
@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "Yes", "on", " y "])
def test_every_spelling_of_true(raw):
    set_env(X=raw)
    assert env._b("X", False) is True


@pytest.mark.parametrize("raw", ["0", "false", "No", "off", "n"])
def test_every_spelling_of_false(raw):
    set_env(X=raw)
    assert env._b("X", True) is False


def test_an_unset_boolean_is_the_default_and_an_empty_one_is_too():
    """EMPTY means "unset" throughout this project -- `env.sh` writes every knob
    as `${VAR:=default}` and an exported empty string is what a cleared knob
    looks like. The two must not disagree."""
    set_env(X=None)
    assert env._b("X", True) is True and env._b("X", False) is False
    set_env(X="")
    assert env._b("X", True) is True and env._b("X", False) is False


def test_a_word_that_is_not_a_boolean_is_refused_and_never_guessed():
    """`ECC_STATIC_ENERGY=maybe` must stop the run, not quietly mean False."""
    set_env(X="maybe")
    with pytest.raises(ConfigError):
        env._b("X", False)
    assert "not-a-boolean" in guards.GUARDS


# ------------------------------------------------------------------- numbers
def test_integers_and_floats_parse_and_refuse():
    set_env(X="42")
    assert env._i("X", 0) == 42
    set_env(X="1.5")
    assert env._f("X", 0.0) == 1.5
    with pytest.raises(ConfigError):
        env._i("X", 0)                      # 1.5 is not an integer
    set_env(X="abc")
    with pytest.raises(ConfigError):
        env._f("X", 0.0)


def test_the_optional_numbers_tell_unset_apart_from_zero():
    """THE DISTINCTION THE THREE PRICES REST ON. `_of` returns None for unset and
    0.0 for "0", and those mean different things: None is "leave Accelergy's own
    constant alone", 0.0 is the `zero-price` ABLATION (ProjectRestructure
    Appendix B). Collapsing them would make the ablation unreachable."""
    set_env(X=None)
    assert env._of("X") is None and env._oi("X") is None
    set_env(X="0")
    assert env._of("X") == 0.0 and env._oi("X") == 0
    assert env._of("X") is not None, "0 must not be reported as unset"


def test_an_optional_number_that_is_not_a_number_still_refuses():
    set_env(X="free")
    with pytest.raises(ConfigError):
        env._of("X")


# ---------------------------------------------------------------------- lists
def test_a_list_splits_on_commas_and_spaces_alike():
    set_env(X="a, b  c,d")
    assert env._list("X") == ["a", "b", "c", "d"]


def test_a_comment_ends_a_list():
    """`env.sh` carries a comment on many of these lines; the value stops at `#`."""
    set_env(X="resnet18 mobilenet_v2  # and the rest are cold")
    assert env._list("X") == ["resnet18", "mobilenet_v2"]


def test_a_semicolon_list_keeps_its_entries_whole():
    """A `key=a b c` entry must survive whole, so splitting on spaces would
    shred it. `ECC_RECON_PLACEMENT_LIST` was the caller until
    EnvReorganisation phase 3 retired it; the parser is still how a `;`-form
    knob is read, and `_scoped_list` below is the live one."""
    set_env(X="eyeriss=r1 r2 r3; simple=r1")
    assert env._list("X", sep=";") == ["eyeriss=r1 r2 r3", "simple=r1"]


def test_ecc_layers_has_two_spellings_and_they_may_not_be_mixed():
    """EnvReorganisation 6.9: layer names are PER NETWORK, so one bare list
    cannot scope a run that spans two of them. The `=` tells the two forms
    apart, a model with no entry runs WHOLE (an empty list), and mixing the
    two is refused rather than half-read."""
    set_env(X="conv1 layer3.0.conv1")
    assert env._scoped_list("X") == (["conv1", "layer3.0.conv1"], {})
    set_env(X="")
    assert env._scoped_list("X") == ([], {})
    set_env(X="resnet18=conv1 layer3.0.conv1; mobilenet_v2=features.9.conv.2")
    assert env._scoped_list("X") == ([], {
        "resnet18": ["conv1", "layer3.0.conv1"],
        "mobilenet_v2": ["features.9.conv.2"]})
    # a model named with NO layers runs whole -- the table may be sparse
    set_env(X="resnet18=conv1; mobilenet_v2=")
    assert env._scoped_list("X") == ([], {"resnet18": ["conv1"],
                                          "mobilenet_v2": []})
    # BREAKAGE: a bare name beside a table entry is neither spelling
    set_env(X="resnet18=conv1; layer3.0.conv1")
    with pytest.raises(ConfigError):
        env._scoped_list("X")


def test_ecc_layers_third_spelling_is_model_dot_layer():
    """EnvReorganisation phase 6 (the user's ask, 2026-09-14): `resnet18.conv1`
    is one layer of one model and `resnet18.all` is that model whole, without
    the `=`/`;` punctuation. It resolves to the SAME per-model table, so
    nothing downstream learns a third shape.

    THE MODEL LIST HAS TO BE PASSED IN. Layer names contain dots
    (`layer3.0.conv1`) and model names do not, so "the text before the first
    dot is a known model" IS the rule that tells the two apart -- and
    `settings/env.py` may not reach for that list itself.
    """
    models = ("resnet18", "mobilenet_v2")
    set_env(X="resnet18.conv1")
    assert env._scoped_list("X", models=models) == ([], {"resnet18": ["conv1"]})
    # a layer name with dots of its own survives: the split is on the FIRST dot
    set_env(X="resnet18.layer3.0.conv1")
    assert env._scoped_list("X", models=models) == (
        [], {"resnet18": ["layer3.0.conv1"]})
    # `all` in the layer's place is the whole model -- an empty list, exactly
    # what a sparse `=` table means
    set_env(X="resnet18.all")
    assert env._scoped_list("X", models=models) == ([], {"resnet18": []})
    # several models at once
    set_env(X="resnet18.conv1 mobilenet_v2.features.9.conv.2")
    assert env._scoped_list("X", models=models) == (
        [], {"resnet18": ["conv1"], "mobilenet_v2": ["features.9.conv.2"]})

    # A BARE LAYER NAME IS STILL A BARE LAYER NAME. `layer3.0.conv1` has dots
    # but `layer3` is not a model, so it must not be read as the dotted form --
    # this is the whole reason `models` exists.
    set_env(X="conv1 layer3.0.conv1")
    assert env._scoped_list("X", models=models) == (
        ["conv1", "layer3.0.conv1"], {})
    # ...and WITHOUT the model list, nothing is ever the dotted form
    set_env(X="resnet18.conv1")
    assert env._scoped_list("X") == (["resnet18.conv1"], {})

    # BREAKAGE: the spellings may not be mixed, exactly as the `=` form refuses
    set_env(X="conv1 resnet18.conv1")
    with pytest.raises(ConfigError):
        env._scoped_list("X", models=models)
    # BREAKAGE: `.all` and a named layer of the SAME model contradict
    set_env(X="resnet18.all resnet18.conv1")
    with pytest.raises(ConfigError):
        env._scoped_list("X", models=models)


def test_one_value_is_one_value_and_extras_are_refused_not_truncated():
    """A silent truncation here would hold the wrong axis fixed and nothing would
    say so; only the SWEPT axis takes a list."""
    set_env(X="resnet18")
    assert env._one("X") == "resnet18"
    set_env(X="resnet18 mobilenet_v2")
    with pytest.raises(ConfigError):
        env._one("X")


def test_one_value_is_empty_when_unset():
    set_env(X=None)
    assert env._one("X") == ""


# ---------------------------------------------------------------------- table
def test_a_table_parses_env_shs_flattened_associative_array():
    """bash cannot export a `declare -A`, so env.sh section 8 flattens it."""
    set_env(X="sram_bit=1.5;rf_bit=0.25;mac_instance=30")
    assert env._table("X") == {"sram_bit": 1.5, "rf_bit": 0.25, "mac_instance": 30.0}


def test_a_table_tolerates_stray_separators_but_not_a_bad_entry():
    set_env(X=" a=1 ; ; b=2 ;")
    assert env._table("X") == {"a": 1.0, "b": 2.0}
    set_env(X="a=1;b=lots")
    with pytest.raises(ConfigError):
        env._table("X")


def test_an_unset_table_is_empty_not_an_error():
    set_env(X=None)
    assert env._table("X") == {}


# ------------------------------------------------------- unset vs set-to-empty
def test_list_or_none_is_the_one_parser_that_tells_unset_from_empty():
    """It exists for exactly one knob. `ECC_SWEEP_ARCHS` UNSET means "every
    declared design" and is filled in by `config._resolve()`; set to whitespace
    it means the user asked for NOTHING and is refused. A plain default cannot
    say both, which is why this function is not just `_list`."""
    set_env(X=None)
    assert env._list_or_none("X") is None
    set_env(X="   ")
    assert env._list_or_none("X") == []
    set_env(X="a b")
    assert env._list_or_none("X") == ["a", "b"]


def test_strings_are_stripped():
    set_env(X="  padded  ")
    assert env._s("X") == "padded"
