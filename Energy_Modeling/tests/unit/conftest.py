"""`tests/unit/` -- pure functions. No cache, no filesystem, no Timeloop.

`ProjectRestructure.md` section 7.4's first bucket:

    tests/unit/       pure functions only. No env, no cache, no filesystem.
                      MUST be green at all times. If this is red, something is
                      broken.  -> physics/, contracts/, settings/ parsing

**"Red means red" is the whole point of the bucket.** A red suite used to mean a
real bug, a missing mapper cache or a forgotten environment variable, and you
could not tell which without an investigation -- which is what made the tests
untrustworthy. Nothing here can be red for any reason except a broken invariant:
these tests read no `results/`, invoke no mapper, and never skip.

ON THE ONE APPARENT EXCEPTION. `test_env_parsers.py` does touch `os.environ` --
it is the test of the module whose entire job is reading it. What the rule
forbids is depending on AMBIENT state, and these set every variable they read.
`tests/conftest.py`'s autouse fixture snapshots and restores `os.environ` around
every test, so nothing leaks either way (phase 1; it is why 33 failures went
away without a line of production code changing).
"""
