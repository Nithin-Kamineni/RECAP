"""Task 3 and Task 4: where the reconstruction boundary sits, and what it costs.

The DC datapath numbers (`data/dc/BCH_N63_results.json` and the two override
tables), the placement study's scope, the packing and granularity models, the
encoder and decode sites, and the ERT arm.

ONLY `recon_ert_arm` REACHES THE MAPPER. Everything else here is evaluator-side
and must never enter the cache key: a bar re-costed at a different reconstruction
energy is the same mapping, and pretending otherwise would cold the matrix for
an arithmetic change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..contracts.errors import ConfigError
from .env import _b, _f, _i, _list, _of, _oi, _one, _s, _table

#: How the reduced representation is physically stored and transported, and how
#: encoder work is charged. Both are `recon.py`'s, and both are answers to
#: sections 15/16 of `02_reconstruction_dse_and_implementation.txt` rather than
#: free parameters -- the docstrings there say what each one claims.
RECON_PACKINGS = ("stream", "aligned")

RECON_GRANULARITIES = ("weight", "codeword")

#: Where the BCH decoder sits (env.sh section 4). `ondie` is the model since
#: 2026-09-09: the decoder is on the DRAM die and off the fetch path, so only
#: the k message bits cross the DRAM interface and every placement's DRAM
#: interface term scales by K/N. `controller` is the pre-2026-09-09 model kept
#: as a runnable row for the diff. `recon.DECODE_SITES` must stay in step.
RECON_DECODE_SITES = ("ondie", "controller")

#: Where a NETWORK boundary's encoders sit, and therefore how many times they
#: run: `destination` (one per destination, count = the network's
#: destination-side arrivals) or `source` (one before the fanout, count = its
#: ingresses). `recon.ENCODER_SITES` must stay in step.
RECON_ENCODER_SITES = ("destination", "source")

#: The fields of this group the MAPPER sees, and therefore the ones
#: `Config.fingerprint()` hashes.
#:
#: The ARM is a different chip (prompt_7 6.4); everything else in this group
#: is evaluator-side and must never reach the mapper cache key.
IN_FINGERPRINT = frozenset({
    "recon_ert_arm",
})


@dataclass(frozen=True)
class ReconSettings:
    recon_json: str
    #: prompt_6 RULE 3: the two Design Compiler tables from env.sh section 6
    #: (flattened by section 10), `{configuration id: pJ}` -- incremental per
    #: codeword, idle per cycle per engine. Read by `ecc.load_recon_terms`
    #: when the JSON has no entry for the (N,K) in play. ECC_RECON_INCLUDE_IDLE
    #: is retired: the two terms are never added, each has its own denominator.
    recon_incremental_table: dict
    #: ECC_RECON_REQUIRE_GROUP_RESIDENCY (env.sh section 4). 1 = a PE-local
    #: boundary whose level holds fewer than `G_rec` weights at once is refused
    #: as `unsupported`; 0 (the default, decided 2026-09-13) charges it and
    #: REPORTS the shortfall instead. RECAP's engine accumulates retained bits
    #: as they arrive rather than needing the whole group resident in one
    #: instant, so a small tile costs buffering and accesses, not feasibility.
    recon_require_group_residency: bool
    recon_idle_table: dict
    recon_pj_override: Optional[float]
    recon_incremental_fallback_pj: float
    recon_idle_fallback_pj: float
    #: prompt_7 Issue 4. Percentage of the reconstruction engine's
    #: CLOCKED-IDLE energy that clock gating removes. 0 reproduces the
    #: pre-gating model exactly; 99.5 is the measured clock/dynamic share.
    recon_clock_gating_pct: float
    #: The placement study runs on ONE architecture, ONE model and ONE code,
    #: because each architecture has its own weight path and therefore its own
    #: list of feasible boundaries. env.sh section 4 collapses section 3's lists
    #: onto this point when `recon_modeling` is on, so `archs`/`models`/`code_k`
    #: are already the single values by the time this object exists.
    recon_modeling: bool
    recon_optimizer: bool
    #: The raw `ECC_RECON_PLACEMENT_LIST`, as env.sh section 10 flattens the
    #: `ECC_RECON_PLACEMENTS` associative array (which bash cannot export):
    #: `;`-separated `arch=key key key` entries, or a bare space-separated key
    #: list that applies to every architecture. Read it through
    #: `recon_placements_for()`, never directly.
    recon_placement_keys: list
    recon_stem: str
    recon_packing: str
    recon_granularity: str
    recon_onchip_fraction: Optional[float]
    recon_placement_charges_decode: bool
    #: `ondie` | `controller` -- see RECON_DECODE_SITES.
    recon_decode_site: str
    #: `destination` | `source` -- see RECON_ENCODER_SITES and
    #: `recon.ENCODER_SITES`. Only network boundaries depend on it.
    recon_encoder_site: str
    #: ECC_RECON_ERT_AWARE (prompt_6 8.1): the ERT arms get their own mapping,
    #: solved with the encoder's energy in the objective, and the figure marks
    #: which bars came from one. Requires RECON_OPTIMIZER=True and
    #: ECC_PHASE=Post; anything else is refused in `__post_init__`.
    recon_ert_aware: bool
    #: ECC_RECON_ERT_ARM (prompt_6 8.3): WHICH arm one mapper job is solving --
    #: `reference` (also the empty string), or the key of an ERT-injectable
    #: placement of the single configured architecture (`recon2`, `recon4`
    #: on Eyeriss v1). Set by the launcher in `--export`, not by hand.
    #: `__post_init__` resolves a placement key into `weight_datawidth = q`
    #: and `weight_datawidth_levels` = the storage levels in its `reduced`
    #: set, so everything downstream (patched YAML, slug, fingerprint) keys
    #: on fields that already exist; `archs.ert_bump()` derives the ERT
    #: delta from it. Read it through `ert_arm()`.
    recon_ert_arm: str
    #: ECC_RECON_LAYER (prompt_6 8.2): the placement study's layer scope --
    #: one layer name, or `all`/empty for the whole model. env.sh section 10
    #: seeds ECC_LAYERS from it whenever ECC_RECON_MODELING=1, so `layers` is
    #: already the resolved scope; this field records the spelling for the
    #: manifest. The launcher (hpc/map_ert_arms.sh) sets it per job.
    recon_layer: str
    #: ECC_RECON_BW_SCALE (prompt_7 C1.2). 1 = every stage in the arm's
    #: `reduced` set declares `per_dataspace_bandwidth_consumption_scale:
    #: {Weights: ...}` -- K/N at DRAM (a bit stream) and q/8 on chip (whole
    #: weights in a narrower word). COLDS EVERY CACHE: it is in the patched
    #: YAML.
    recon_bw_scale: bool

    @staticmethod
    def from_env():
        """Every `ECC_*` knob of this group, read once.

        `recon_ert_arm` is normalised to `"reference"` by `config._resolve()`,
        which is also where it is checked against the design's arm list -- that
        check needs the architecture, and this layer is below it.
        """
        return ReconSettings(
            recon_json=_s("ECC_RECON_JSON", "data/dc/BCH_N63_results.json"),
            recon_incremental_table=_table("ECC_RECON_INCREMENTAL_PJ_LIST"),
            recon_require_group_residency=_b("ECC_RECON_REQUIRE_GROUP_RESIDENCY", False),
            recon_idle_table=_table("ECC_RECON_IDLE_PJ_LIST"),
            recon_pj_override=_of("ECC_RECON_PJ"),
            recon_incremental_fallback_pj=_f("ECC_RECON_INCREMENTAL_FALLBACK_PJ", 1.8995),
            recon_idle_fallback_pj=_f("ECC_RECON_IDLE_FALLBACK_PJ", 2.2301273),
            recon_clock_gating_pct=_f("ECC_RECON_CLOCK_GATING_PCT", 99.5),
            recon_modeling=_b("ECC_RECON_MODELING", False),
            # env.sh spells this `RECON_OPTIMIZER` as well, and mirrors it into the
            # ECC_-prefixed name every other knob uses.
            recon_optimizer=_b("ECC_RECON_OPTIMIZER", False),
            recon_placement_keys=_list("ECC_RECON_PLACEMENT_LIST", sep=";"),
            recon_stem=_s("ECC_RECON_STEM", "ReconSweep"),
            recon_packing=_s("ECC_RECON_PACKING", "stream").lower(),
            recon_granularity=_s("ECC_RECON_ENCODER_GRANULARITY", "weight").lower(),
            recon_onchip_fraction=_of("ECC_RECON_ONCHIP_FRACTION"),
            recon_placement_charges_decode=_b("ECC_RECON_PLACEMENT_CHARGES_DECODE", True),
            recon_decode_site=_s("ECC_RECON_DECODE_SITE", "ondie").lower(),
            recon_encoder_site=_s("ECC_RECON_ENCODER_SITE", "destination").lower(),
            recon_ert_aware=_b("ECC_RECON_ERT_AWARE", False),
            recon_ert_arm=_s("ECC_RECON_ERT_ARM").strip().lower(),
            recon_layer=_s("ECC_RECON_LAYER").strip(),
            recon_bw_scale=_b("ECC_RECON_BW_SCALE", False),
        )
