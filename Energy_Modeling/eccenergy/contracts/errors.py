"""The one exception every layer may raise, and every layer may catch.

`ConfigError` is what a RUN dies of: a knob that names something that does not
exist, two knobs that contradict each other, a value outside what it can mean.
It is not a bug report -- every message here is written for the person who set
the knob, and says what to set instead.

IT LIVES AT L0 BECAUSE BOTH ENDS NEED IT. `settings/` raises it while reading the
environment, and `config.py` raises it while resolving a configuration against
the designs -- and `settings/` sits below `config.py`, so neither may own it.
`__main__` catches it and prints it without a traceback.

ProjectRestructure phase 4 moved it here from `config.py`, which is also the
first thing `contracts/` has ever held: a type two layers share, and nothing
that computes with it.
"""
from __future__ import annotations


class ConfigError(RuntimeError):
    """A knob is wrong. Carries the message the user is shown, nothing else."""
