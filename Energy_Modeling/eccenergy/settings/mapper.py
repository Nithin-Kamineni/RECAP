"""The search: what the mapper optimises, how hard, and with what budget.

Every field here except `mapper_threads` is in the fingerprint, because each of
them changes WHICH mapping comes back. `mapper_threads` changes only how long it
takes -- but it is in the mapper SLUG, so the cache still separates runs by it
(CLAUDE.md: keep `ECC_MAPPER_THREADS=18`).

prompt_3's constrained mapspace is the default since 2026-09-11:
`linear_pruned`, victory 2000, timeout 100000000, and `ECC_MAPSPACE_CONSTRAIN`
in `ArchSettings` -- a systematic walk of an UNconstrained space is the wrong
regime (FINDINGS 2.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..contracts.errors import ConfigError
from .env import _b, _f, _i, _list, _of, _oi, _one, _s, _table

#: What the mapper is allowed to minimise. Timeloop's own metric names.
#:
#: `energy` is the default because this is an ENERGY study. The stock
#: `_include/mapper.yaml` that ships with the exercises repo asks for `edp`,
#: and EDP systematically penalises the wider PE array: an architecture with
#: more parallelism can buy latency by spending energy, so EDP steers it to a
#: costlier mapping. Measured on mobilenet_v2's C160_M960_R1_S1_P7_Q7 layer,
#: Eyeriss v2 (384 MACs) discarded a 6.84 pJ/MAC mapping for a 15.32 pJ/MAC one
#: because the latter was 4x faster; Eyeriss v1 (168 MACs) gave up only 1.24x
#: on the same layer. Set ECC_OPT_METRIC=edp to reproduce the old numbers.
OPT_METRICS = ("energy", "edp", "delay", "last_level_accesses")

#: How mapper effort scales with the depth of the architecture's loop nest.
#:
#: Timeloop's random search gives up after `victory_condition` consecutive
#: non-improving mappings, so a deeper hierarchy is searched less thoroughly at
#: the same setting. Eyeriss v2 has 9 loop levels to v1's 8 and a strictly
#: larger spatial search space, and at a flat victory_condition it converged
#: visibly less well. `levels` doubles the effort per level beyond the
#: reference depth; `none` uses ECC_VICTORY flat, as before.
VICTORY_SCALINGS = ("levels", "none")

#: Loop-nest depth that `ECC_VICTORY` is quoted for: Eyeriss v1 at paper
#: fidelity (6 storage levels + 2 spatial). Deeper designs scale up from here.
VICTORY_REFERENCE_LEVELS = 8

#: Cap on the scaling, so a deep hierarchy cannot make a run open-ended.
VICTORY_MAX_SCALE = 8

#: The fields of this group the MAPPER sees, and therefore the ones
#: `Config.fingerprint()` hashes.
#:
#: The search itself. `mapper_threads` is NOT here and must not be -- the
#: thread count changes how long a search takes, never what it finds... except
#: that it IS in the mapper slug (`mapper_slug`), which is a different string
#: and a different decision. CLAUDE.md: keep ECC_MAPPER_THREADS=18.
IN_FINGERPRINT = frozenset({
    "opt_metric",
    "victory",
    "victory_scaling",
    "mapper_algorithm",
    "mapper_seed",
    "mapper_timeout",
    "mapper_search_size",
    "mapper_max_permutations",
})


@dataclass(frozen=True)
class MapperSettings:
    opt_metric: str
    victory: int
    victory_scaling: str
    mapper_threads: Optional[int]
    mapper_timeout: int
    mapper_algorithm: str
    mapper_seed: Optional[int]
    mapper_search_size: Optional[int]
    mapper_max_permutations: int

    @staticmethod
    def from_env():
        """Every `ECC_*` knob of this group, read once."""
        return MapperSettings(
            opt_metric=_s("ECC_OPT_METRIC", "energy").lower(),
            victory=_i("ECC_VICTORY", 500),
            victory_scaling=_s("ECC_VICTORY_SCALING", "levels").lower(),
            mapper_threads=_oi("ECC_MAPPER_THREADS"),
            mapper_timeout=_i("ECC_MAPPER_TIMEOUT", 10000),
            mapper_algorithm=_s("ECC_MAPPER_ALGORITHM", "hybrid"),
            mapper_seed=_oi("ECC_MAPPER_SEED"),
            mapper_search_size=_oi("ECC_MAPPER_SEARCH_SIZE"),
            mapper_max_permutations=_i("ECC_MAPPER_MAX_PERMUTATIONS", 16),
        )
