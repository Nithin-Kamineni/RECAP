"""Test plumbing: RED MEANS RED.

`ProjectRestructure.md` section 7.5 diagnosed this and it is implemented here.
The suite had 64 failures with nothing wrong in the code, which is worse than
no suite at all -- a red run that nobody believes stops nothing and hides the
one failure that matters.

THREE CAUSES, ALL PLUMBING
--------------------------
1. A POISONED ENVIRONMENT. Several test modules predate pytest and carry their
   own `main()` runner. Their `_cfg()` helpers do

       for k in list(os.environ):
           if k.startswith("ECC_"):
               del os.environ[k]
       os.environ.update(...)

   which is correct for a module run on its own and catastrophic for a module
   run alongside others in one process: the first such test wipes env.sh's
   configuration for everything after it, and its deliberately-invalid values
   (`ECC_BASELINE_DRAM_PJ_PER_BIT=-70`, there to prove the validator fires)
   leak into every later `config.load_config()`. Measured: 33 failures blaming
   a negative DRAM price nobody set, and 14 more insisting the study was on
   `eyeriss_v2_like`. `_ecc_env` below snapshots and restores `os.environ`
   around EVERY test, so a module can still wipe what it likes.

2. A COLD CACHE READ AS A FAILURE. The data-backed suites assert on real
   solved mappings. When the architecture changes -- which is a normal thing to
   do, and the whole point of having knobs -- those mappings do not exist yet.
   The property is then UNTESTED, not violated, and `pytest.skip` is what says
   so. `_Skip` (each module's own) is translated below, and `pytest.skip` is
   used directly at the sites that used to `raise AssertionError`.

3. A DESIGN INHERITED FROM WHATEVER RAN LAST -- fixed by (1), since the live
   design now comes from env.sh for every test rather than from a neighbour.

WHAT THIS DOES NOT DO. It weakens no assertion. A property that can be checked
is still checked and still fails when it is false; `test_arch_fingerprint.py`
and the mutation tests in `test_mapper_arms.py` are untouched by it. The only
thing that changes is that "not mapped yet" now reads as `s` and not as `F`.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _ecc_env():
    """Snapshot `os.environ` before each test and put it back afterwards.

    Autouse, so no test module has to know it exists -- which matters, because
    the modules that poison the environment are the ones that predate pytest
    and are not going to be rewritten to opt in.
    """
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Report a module's home-grown `_Skip` as a skip, not as a failure.

    Five modules define their own `class _Skip(Exception)` for "there is no
    cached data to check this against". Under their own `main()` that printed
    `skip`; under pytest an uncaught exception is a failure, so the same
    condition read as a red suite. Nothing about the tests changes -- only how
    the outcome is spelled.

    Wrapping the CALL rather than the report, because a skip has to be raised
    while the test is still running: `pytest.skip()` from
    `pytest_runtest_makereport` is an INTERNALERROR.
    """
    try:
        return (yield)
    except BaseException as exc:                      # noqa: BLE001 - re-raised
        if type(exc).__name__ == "_Skip":
            pytest.skip(f"no cached data for this property: {exc}")
        raise


#: The point every `cfg()` starts from. STATED HERE, NOT INHERITED.
#:
#: env.sh is the study's configuration and it moves -- the live code was
#: BCH(63,30) in September and is BCH(63,39) now, and `ECC_SWEEP=fix` holds
#: the architecture at whatever heads `ECC_ARCHS`. A test that reads its
#: design from the environment therefore tests a different chip depending on
#: when it is run, which is the third of the three causes in this file's header.
#:
#: `eyeriss_like_wglb` at BCH(63,39) on resnet18 is chosen because it is the one
#: corner with a WARM mapper cache (CLAUDE.md, "EVERY MAPPER CACHE IS COLD, ON
#: PURPOSE"): a data-backed property written against this fixture can actually
#: find a solved mapping to check itself against. `ECC_FROM_CACHE=1` is in here
#: so that no test can invoke Timeloop by accident.
PINNED_ENV = {
    "ECC_CONST_ARCH": "eyeriss_like_wglb",
    "ECC_SWEEP_ARCHS": "eyeriss_like_wglb",
    "ECC_CONST_MODEL": "resnet18",
    "ECC_SWEEP_MODELS": "resnet18",
    "ECC_CODE_N": "63",
    "ECC_CONST_K": "39",
    "ECC_FROM_CACHE": "1",
}


@pytest.fixture
def cfg():
    """Build a `Config` from `PINNED_ENV`, overridden by whatever you pass.

        def test_something(cfg):
            c = cfg(ECC_DRAM_PJ_PER_BIT="40")

    WHY A FIXTURE AND NOT A HELPER. Nine test modules each carry their own
    `_cfg()`, six of which delete every `ECC_*` variable before building one.
    That is correct in isolation and was catastrophic in one pytest process
    until `_ecc_env` above started putting the environment back -- and it is
    still nine near-copies of the same twenty lines, each free to drift onto a
    different design. This is the one copy, and the design it pins is stated in
    `PINNED_ENV` rather than inherited from env.sh.

    The wipe is kept, because a knob left set by env.sh is exactly the kind of
    thing that makes a test pass on one machine and fail on another. It is safe
    here for the same reason: `_ecc_env` restores the environment afterwards.

    Pass `None` for a knob to leave it UNSET rather than setting it to the
    string "None" -- `ECC_DRAM_PJ_PER_BIT=None` means "there is no price", which
    several refusal tests need.
    """
    def _build(**knobs):
        env = dict(PINNED_ENV)
        env.update(knobs)
        for key in list(os.environ):
            if key.startswith("ECC_"):
                del os.environ[key]
        os.environ.update({k: str(v) for k, v in env.items() if v is not None})
        from .. import config
        return config.load_config()
    return _build
