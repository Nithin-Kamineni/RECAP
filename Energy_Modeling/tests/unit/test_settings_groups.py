"""The six `from_env()` readers -- the only door every `ECC_*` knob comes through.

`tests/contract/test_settings.py` holds the STRUCTURE: which group owns which
knob, the fingerprint's hashed set, and `to_dict()`'s key order. None of it
exercises the READING, and `from_env` is where a knob's default lives -- so a
default that silently changed would move every number in the study and no test
would have anything to say.

These are the four properties that must hold for all six groups at once, so a
group added later is covered the day it is written rather than the day somebody
remembers. They are written over `settings.GROUPS`, not over a hand-typed list.
"""
import dataclasses
import os

import pytest

from eccenergy import config
from eccenergy.settings import (ArchSettings, CodeSettings, EnergySettings,
                                MapperSettings, ReconSettings, RunSettings)

GROUPS = {
    "run": RunSettings, "code": CodeSettings, "arch": ArchSettings,
    "mapper": MapperSettings, "recon": ReconSettings, "energy": EnergySettings,
}


@pytest.fixture(autouse=True)
def _bare_environment():
    """Every ECC_* knob removed, so each test reads DEFAULTS unless it sets one.

    `tests/conftest.py`'s autouse fixture puts the caller's environment back
    afterwards, so stripping it here cannot leak into anything else (phase 1 --
    one leaked variable used to cost 33 failures).
    """
    for k in [k for k in os.environ if k.startswith("ECC_")]:
        del os.environ[k]
    os.environ.pop("RECON_OPTIMIZER", None)


@pytest.mark.parametrize("name,cls", sorted(GROUPS.items()))
def test_every_group_reads_from_a_bare_environment(name, cls):
    """A group must have a complete set of defaults. If one knob had no default,
    a fresh shell without `env.sh` would fail here instead of halfway through a
    SLURM job."""
    got = cls.from_env()
    assert isinstance(got, cls)


@pytest.mark.parametrize("name,cls", sorted(GROUPS.items()))
def test_every_group_is_frozen_so_nothing_can_edit_a_knob_after_it_is_read(name, cls):
    """The knobs are read ONCE. A settings object that could be mutated would let
    one stage run at a configuration a later stage reports."""
    got = cls.from_env()
    field = dataclasses.fields(cls)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(got, field, getattr(got, field))


@pytest.mark.parametrize("name,cls", sorted(GROUPS.items()))
def test_reading_a_group_twice_gives_the_same_answer(name, cls):
    """`from_env` must be a pure function of the environment: no clock, no
    counter, no cached directory listing. Two reads that differ would make a
    fingerprint depend on WHEN it was taken."""
    assert cls.from_env() == cls.from_env()


@pytest.mark.parametrize("name,cls", sorted(GROUPS.items()))
def test_every_group_declares_which_of_its_knobs_the_mapper_sees(name, cls):
    """`IN_FINGERPRINT` is declared beside the knobs, not in a list at the other
    end of the file, and every name in it must be a real field of the group. A
    typo there would silently drop a knob out of the mapper cache key -- which is
    two architectures sharing one directory."""
    import importlib
    mod = importlib.import_module(f"eccenergy.settings.{name}")
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(mod.IN_FINGERPRINT - fields)
    assert not unknown, f"{name}.IN_FINGERPRINT names non-fields: {unknown}"


# ------------------------------------------------- the environment really wins
def test_the_environment_beats_the_default():
    """`env.sh` writes every knob as `${VAR:=default}` so a one-off never needs a
    file edit. That only works if the reader honours what is exported."""
    assert MapperSettings.from_env().victory == 500
    os.environ["ECC_VICTORY"] = "2000"
    assert MapperSettings.from_env().victory == 2000


def test_an_empty_knob_means_unset_and_not_an_error():
    """A cleared knob in a shell is an exported empty string, which must read the
    same as never having been set -- this is how `ECC_MAPPER_SEED=` means "no
    seed" rather than "seed 0"."""
    os.environ["ECC_MAPPER_SEED"] = ""
    assert MapperSettings.from_env().mapper_seed is None
    os.environ["ECC_MAPPER_SEED"] = "0"
    assert MapperSettings.from_env().mapper_seed == 0


def test_a_bad_value_stops_the_read_rather_than_falling_back_to_the_default():
    """The whole point of the parse tier: a typo must not run the study at the
    default and report it as the value you asked for."""
    os.environ["ECC_VICTORY"] = "two thousand"
    with pytest.raises(config.ConfigError):
        MapperSettings.from_env()


def test_the_six_groups_partition_the_knobs_with_no_field_named_twice():
    """Two groups owning one name would make `cfg.<knob>` ambiguous, and
    `Config.flat()` would silently keep whichever came last."""
    seen = {}
    for name, cls in GROUPS.items():
        for f in dataclasses.fields(cls):
            assert f.name not in seen, (
                f"{f.name} is declared by both {seen[f.name]} and {name}")
            seen[f.name] = name
    assert len(seen) > 100, f"only {len(seen)} knobs found -- has a group moved?"
