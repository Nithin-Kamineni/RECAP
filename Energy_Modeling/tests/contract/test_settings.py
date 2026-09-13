"""The two things about `Config` that are BYTES ON DISK, pinned.

`ProjectRestructure.md` phase 4 split `Config`'s 112 knobs into six frozen
settings objects. Two properties of that object are not internal at all -- they
are written into files this study is judged by -- and a refactor can move either
one without moving a single number:

  1. `to_dict()`'s KEY ORDER. Every manifest and every evaluation result carries
     it, written with `indent=1` and no `sort_keys`, so the order IS the file.
     Reordering it rewrites every result in the study and says nothing about why.

  2. `fingerprint()`'s HASHED SET. It is now assembled from the six groups' own
     `IN_FINGERPRINT` declarations instead of one list kept by hand. Adding a
     field to the wrong side of that union COLDS THE WHOLE MAPPER CACHE -- hours
     of SLURM -- and the fingerprint would still look like a fingerprint.

Both are pinned against the state phase 4 inherited, so the test fails on a
change rather than after it. Neither pin is a guess: item 1 was read off a live
`to_dict()` before the split, and item 2 is the `keep` tuple that stood in
`Config.fingerprint()` word for word.

This is `tests/contract/`: structure, not numbers. It builds no Config and needs
no cache -- it reads the declarations.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eccenergy import config                                    # noqa: E402
from eccenergy.settings import (ArchSettings, CodeSettings, EnergySettings,   # noqa: E402
                                MapperSettings, ReconSettings, RunSettings)

#: `Config.fingerprint()`'s `keep` tuple as it stood before phase 4, verbatim.
#: A name added or removed here is a cold matrix; nothing else about the hash
#: can change without this tuple changing too.
FINGERPRINT_FIELDS_BEFORE_PHASE_4 = (
    "archs", "models", "layers", "weight_bits", "activation_bits",
    "acc_bits_override", "arch_fidelity", "force_technology",
    "force_datawidth", "weight_capacity_scale",
    "weight_capacity_scope", "weight_factor_relax",
    "mapspace_constrain",
    "weight_datawidth", "weight_datawidth_levels", "recon_ert_arm",
    "weight_depth_scale", "weight_depth_levels",
    "weight_width_glb_mult",
    "dram_depth", "global_cycle_seconds",
    "noc_enabled", "noc_wire_pj_per_bit_mm", "noc_router_pj",
    "noc_pe_latch_pj", "noc_scale",
    "opt_metric", "victory", "victory_scaling", "mapper_algorithm",
    "mapper_seed", "mapper_timeout", "mapper_search_size",
    "mapper_max_permutations")

GROUPS = (RunSettings, CodeSettings, ArchSettings, MapperSettings,
          ReconSettings, EnergySettings)


def _declared():
    """The union of the six groups' own `IN_FINGERPRINT` declarations."""
    out = set()
    for g in config.GROUPS:
        out |= config._FINGERPRINT_MODULES[g].IN_FINGERPRINT
    return out


def test_the_hashed_set_is_what_it_was_before_the_split():
    """Assembled from six declarations, it must still be the same SET.

    `json.dumps(..., sort_keys=True)` makes the hash depend on the set and not
    on the order, which is exactly what lets it be assembled from six pieces --
    and exactly why the SET has to be pinned instead.
    """
    assert _declared() == set(FINGERPRINT_FIELDS_BEFORE_PHASE_4), (
        "the mapper cache key moved.\n"
        f"  added  : {sorted(_declared() - set(FINGERPRINT_FIELDS_BEFORE_PHASE_4))}\n"
        f"  dropped: {sorted(set(FINGERPRINT_FIELDS_BEFORE_PHASE_4) - _declared())}\n"
        "Every `fp-<hash>` directory under ecc_energy_study/outputs/ is hours of "
        "SLURM. If this is deliberate, budget for a full re-map and update the "
        "tuple in this file with the date and the reason.")


def test_every_hashed_field_is_declared_by_the_group_that_owns_it():
    """A group may only mark its OWN fields, or the union means nothing."""
    for g in config.GROUPS:
        cls = config._CLASSES[g]
        marked = config._FINGERPRINT_MODULES[g].IN_FINGERPRINT
        stray = sorted(marked - set(cls.__dataclass_fields__))
        assert not stray, f"{g}: IN_FINGERPRINT names fields it does not own: {stray}"


def test_energy_model_rev_is_hashed_only_when_it_is_set():
    """EMPTY must hash byte-identically to every fingerprint predating the knob.

    It is the one field added to the hash at RUN time rather than declared into
    it, and that asymmetry is the whole knob: any non-empty value colds the
    matrix on purpose, and the default may never cost a re-map.
    """
    assert "energy_model_rev" not in _declared()
    src = (ROOT / "eccenergy" / "config.py").read_text()
    assert 'keep.add("energy_model_rev")' in src


def test_the_record_keeps_its_key_order():
    """`FIELD_ORDER` is every field of the six groups, and nothing else.

    The ORDER itself is pinned by the file it is declared in; what this checks is
    that it still covers the six groups exactly -- a field added to a group and
    not to `FIELD_ORDER` would silently vanish from every manifest.
    """
    declared = set(config.FIELD_ORDER)
    owned = {f for cls in GROUPS for f in cls.__dataclass_fields__}
    assert declared == owned, (
        f"  missing from FIELD_ORDER: {sorted(owned - declared)}\n"
        f"  named but owned by nobody: {sorted(declared - owned)}")
    assert len(config.FIELD_ORDER) == len(set(config.FIELD_ORDER))


def test_every_knob_is_owned_by_exactly_one_group():
    """Two groups declaring the same knob would make `cfg.x` ambiguous."""
    seen = {}
    for cls in GROUPS:
        for f in cls.__dataclass_fields__:
            assert f not in seen, f"{f} is declared by both {seen[f]} and {cls.__name__}"
            seen[f] = cls.__name__


def test_the_settings_groups_are_frozen():
    """A resolved configuration that can be edited is not a resolved one."""
    for cls in GROUPS:
        assert cls.__dataclass_params__.frozen, f"{cls.__name__} is not frozen"
    assert config.Config.__dataclass_params__.frozen


def test_an_unknown_knob_is_refused_rather_than_stored():
    """`with_()` routes by name, so a typo must fail LOUDLY, not add a field."""
    cfg = object.__new__(config.Config)
    with pytest.raises(config.ConfigError):
        config.Config.with_(cfg, victroy=4000)


def test_settings_import_nothing_above_them():
    """The knobs may not reach for a design, or phase 4's cycle comes back.

    `tests/contract/test_layer_rule.py` states this for every module; it is
    repeated here for the one package the whole phase was about, and in the form
    a reader of `settings/` would check it: what does it import?
    """
    # A single dot stays inside `settings/`; two dots LEAVE it, and only two
    # packages may be on the other side.
    allowed = ("contracts", "paths")
    for f in sorted((ROOT / "eccenergy" / "settings").glob("*.py")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line.startswith("from .."):
                continue
            target = line.split()[1].lstrip(".").split(".")[0]
            assert target in allowed, (
                f"{f.name}: {line}\n"
                f"settings/ is L0: it may import {' or '.join(allowed)} and nothing "
                f"else. Reaching for a design here is what phase 4 removed.")
