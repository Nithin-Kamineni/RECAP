"""The test plumbing, tested. `ProjectRestructure.md` sections 7.4 and 7.5.

`conftest.py` is what makes "red means red" true: it restores `os.environ`
around every test, translates a module's home-grown `_Skip` into a skip, and
offers one `cfg()` pinned to a stated design. Nothing checked any of that, which
is the wrong way round -- a silent regression in the plumbing does not fail
loudly, it makes some OTHER suite fail for a reason nobody can see. That cost 64
failures once.

Every test here is a MUTATION test: it breaks the thing on purpose and proves
the plumbing notices. None of them touches production code, a cache, or a file.
"""
import os

import pytest

from . import conftest as plumbing


def _raw_fixture(fixture):
    """The undecorated generator/function inside a pytest fixture."""
    return fixture.__wrapped__


# --------------------------------------------------------------- environment
def test_the_env_fixture_puts_back_a_wiped_environment():
    """Run `_ecc_env` by hand, wipe every `ECC_*` inside it, and prove teardown
    restores the environment EXACTLY -- keys, values and absences.

    This is cause 1 in `conftest.py`'s header: six modules delete every `ECC_*`
    variable before building a config, and one of them sets a deliberately
    invalid price to prove a validator fires. Before the fixture existed that
    price leaked into 33 later tests.
    """
    os.environ["ECC_TEST_CANARY"] = "before"
    before = dict(os.environ)

    gen = _raw_fixture(plumbing._ecc_env)()
    next(gen)                                        # setup
    for key in [k for k in os.environ if k.startswith("ECC_")]:
        del os.environ[key]
    os.environ["ECC_BASELINE_DRAM_PJ_PER_BIT"] = "-70"   # the real poison
    assert os.environ != before, "the wipe did not happen; the test proves nothing"
    with pytest.raises(StopIteration):
        next(gen)                                    # teardown

    assert dict(os.environ) == before
    del os.environ["ECC_TEST_CANARY"]


def test_the_env_fixture_removes_a_variable_the_test_invented():
    """Restoring means restoring: a knob a test ADDS must not survive it."""
    os.environ.pop("ECC_TEST_INVENTED", None)
    gen = _raw_fixture(plumbing._ecc_env)()
    next(gen)
    os.environ["ECC_TEST_INVENTED"] = "1"
    with pytest.raises(StopIteration):
        next(gen)
    assert "ECC_TEST_INVENTED" not in os.environ


# ---------------------------------------------------------------- cfg() point
def test_cfg_is_pinned_to_its_own_design_not_the_environments(cfg):
    """`cfg()` states its design; a hostile environment must not move it.

    Cause 3: a test that reads its design from `os.environ` tests whichever
    chip env.sh happens to be pointed at that week.
    """
    os.environ.update({"ECC_CONST_ARCH": "eyeriss_v2_like",
                       "ECC_CONST_MODEL": "mobilenet_v2",
                       "ECC_CONST_K": "30"})
    c = cfg()
    assert c.const_arch == plumbing.PINNED_ENV["ECC_CONST_ARCH"] == "eyeriss_like_wglb"
    assert c.const_model == "resnet18"
    assert c.code_k == 39 and c.code_n == 63


def test_cfg_overrides_win_over_the_pin(cfg):
    """The pin is a starting point, not a cage."""
    assert cfg(ECC_CONST_K="30").code_k == 30


def test_cfg_leaves_a_none_knob_unset(cfg):
    """`None` means "this knob is not set", not the string "None".

    The refusal tests need a config with no DRAM price at all, and
    `ECC_DRAM_PJ_PER_BIT="None"` would be a parse error rather than an absence.
    """
    cfg(ECC_DRAM_PJ_PER_BIT=None)
    assert "ECC_DRAM_PJ_PER_BIT" not in os.environ


def test_cfg_never_lets_a_test_invoke_timeloop(cfg):
    """`ECC_FROM_CACHE=1` is part of the pin: no test may run the mapper."""
    assert cfg().from_cache is True


# ------------------------------------------------------------------- _Skip
def test_a_modules_own_Skip_is_translated_into_a_skip():
    """Cause 2: five modules raise their own `_Skip` for "no cached data".

    `pytest_runtest_call` turns that into a skip. Proven by driving the hook
    directly -- a `_Skip` raised from a test body must come back as pytest's
    own `Skipped`, and any other exception must come back untouched.
    """
    class _Skip(Exception):
        pass

    def drive(exc):
        gen = plumbing.pytest_runtest_call(item=None)
        next(gen)
        return gen.throw(exc)

    with pytest.raises(pytest.skip.Exception, match="no cached data"):
        drive(_Skip("outputs/ is empty"))
    with pytest.raises(ValueError):
        drive(ValueError("a real failure"))
