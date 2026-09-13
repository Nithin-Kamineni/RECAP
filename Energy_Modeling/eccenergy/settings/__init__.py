"""L0 -- the KNOBS: every `ECC_*` variable, read once, into six frozen objects.

    from .settings.code import CodeSettings
    code = CodeSettings.from_env()        # BCH(63, 39) over 8-bit weights
    code.code_n, code.weight_bits         # and nothing else can change them

    run      what to run: the axes, the layer scope, where the output goes
    code     BCH(N, K) over `weight_bits`-bit weights, and the decode knobs
    arch     THE CHIP the mapper is handed. Every knob here colds the cache
    mapper   the search: objective, victory condition, algorithm, budget
    recon    Task 3 and Task 4: the boundary, the DC datapath, the ERT arm
    energy   what a bit, a MAC and a cycle cost -- the evaluator's prices

`env.py` is the ONLY module in the project that reads `os.environ`, and each
group is the only place its own knobs are named. Every value is written
`${VAR:=default}` in `env.sh`, so the environment wins over the file.

WHAT THIS PACKAGE MAY NOT DO. It imports nothing of ours except
`contracts.errors`, and that is the whole point: these six objects are what
`config.py` RESOLVES against the designs, so they must sit below `arch/` --
before phase 4 the same knobs lived in a `Config` that reached up into `arch/`
and `physics/` to resolve itself, and Python could not import either file first
(`ProjectRestructure.md` section 2.3).

Each group declares `IN_FINGERPRINT`: which of its fields reach the MAPPER and
therefore the mapper cache key. `Config.fingerprint()` hashes the union, so a
knob is marked where it is declared, and `tests/contract/test_settings.py` pins
that union against the hand-written list it replaced -- a moved hash is hours of
SLURM gone.

`banner.py` prints a resolved configuration and is the one thing here that takes
a whole `Config`: it reports, it does not decide.
"""
from __future__ import annotations

from .arch import ArchSettings
from .code import CodeSettings
from .energy import EnergySettings
from .mapper import MapperSettings
from .recon import ReconSettings
from .run import RunSettings

__all__ = ["ArchSettings", "CodeSettings", "EnergySettings", "MapperSettings",
           "ReconSettings", "RunSettings"]
