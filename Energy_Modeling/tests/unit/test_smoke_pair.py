"""`--smoke` picks ONE CACHED unit and ONE COLD one -- both halves, always.

THE RULE THIS ENFORCES, AND WHAT IT COST. EnvReorganisation phase 6 session 1
found that `toolchain/inputs.py` still imported `eccenergy.toolchain.archs`, a
module ProjectRestructure phase 3 had renamed. The import is LAZY and runs on
exactly one path -- `Mapper._map_now`, i.e. a shape that is NOT in the cache --
so NO COLD MAP COULD RUN AT ALL while every smoke test passed, because every
smoke job since the restructure had been submitted against an already-cached
unit, exactly as the rule then asked. A cached unit returns from `stats_for`
before it reaches the mapper path.

    A REAL JOB ON A CACHED UNIT PROVES THE CACHE PATH, NOT THE MAPPER PATH.

So `_smoke_pair` must never return one half and call it a smoke test, and must
never silently drop the cold half when a cached one exists. These are unit
tests over the selection itself; the SLURM half is `hpc/run_all.sh --smoke`.
"""
import pytest

from eccenergy.toolchain.units import _smoke_pair


class _U:
    """The only two fields the selection reads."""

    def __init__(self, name, cached):
        self.name, self.cached = name, cached

    def __repr__(self):
        return f"<{self.name} {'cached' if self.cached else 'COLD'}>"


def test_it_returns_one_of_each_and_the_cached_one_first():
    units = [_U("a", True), _U("b", True), _U("c", False), _U("d", False)]
    got = _smoke_pair(units)
    assert [u.name for u in got] == ["a", "c"], got
    assert [u.cached for u in got] == [True, False], got


def test_the_cold_half_is_picked_even_when_cached_units_come_last():
    """Order must not decide it: the cold half is the half that matters."""
    units = [_U("cold", False), _U("warm", True)]
    got = _smoke_pair(units)
    assert [u.cached for u in got] == [True, False], got
    assert {u.name for u in got} == {"warm", "cold"}


def test_a_wholly_cold_matrix_smokes_the_cold_half_alone():
    """Nothing is invented: there is no cached unit to pick."""
    units = [_U("c", False), _U("d", False)]
    got = _smoke_pair(units)
    assert [u.name for u in got] == ["c"], got
    assert got[0].cached is False


def test_a_wholly_warm_matrix_smokes_the_cache_half_alone():
    """This is the case that must be VISIBLE, not silently reported as a pass:
    a warm matrix cannot smoke the mapper path at all, which is exactly the
    situation that hid the cold-map bug. The selection returns one unit and the
    caller (`_print_smoke`) says the mapper path is not smoked."""
    units = [_U("a", True), _U("b", True)]
    got = _smoke_pair(units)
    assert [u.name for u in got] == ["a"], got
    assert all(u.cached for u in got)


def test_no_units_is_no_pair_rather_than_an_error():
    assert _smoke_pair([]) == ()


@pytest.mark.parametrize("cached_first", [True, False])
def test_the_pair_is_never_two_of_the_same_kind(cached_first):
    """The mutation this is really guarding: a selection that took the first
    two units, or two cached ones, would pass a smoke test that proves half of
    what it claims."""
    units = ([_U("a", True), _U("b", True), _U("c", False)] if cached_first
             else [_U("c", False), _U("d", False), _U("a", True)])
    got = _smoke_pair(units)
    assert len(got) == 2, got
    assert {u.cached for u in got} == {True, False}, got
