"""Task 3 and Task 4: where the reconstruction boundary sits, and what it costs.

The DC datapath numbers (`data/dc/BCH_N63_results.json` and the two override
tables), the placement study's scope, the packing and granularity models, the
encoder site, and the ERT arm.

FIVE KNOBS LEFT ON 2026-09-14 (EnvReorganisation phase 2), each a settled
decision that no longer needed a switch: the BCH decoder is ON THE DRAM DIE
(`ECC_RECON_DECODE_SITE`; the `controller` row and its code path are gone),
group residency is REPORTED and never refused (`ECC_RECON_REQUIRE_GROUP_
RESIDENCY`), a placement pays the decoder whenever ECC_DECODE=1 does
(`ECC_RECON_PLACEMENT_CHARGES_DECODE`), the figure is called `ReconSweep`
(`ECC_RECON_STEM`), and `ECC_RECON_ONCHIP_FRACTION` was read by nothing.

FOUR MORE WENT IN PHASE 3 (2026-09-14), when `hpc/run_all.sh` became the one
launcher: `ECC_RECON_MODELING` (the mode is `reconN` -- or the abstract
`recon` -- in `ECC_APPROACHES`, and `ECC_SWEEP=fix` is the axis),
`RECON_OPTIMIZER` (a constant True: every placement is mapped on its own
chip; `Config.recon_optimizer` is the property that says so),
`ECC_RECON_PLACEMENTS` / `ECC_RECON_PLACEMENT_LIST` (the boundaries come from
`ECC_APPROACHES` intersected with the design's own `placements.yaml`) and
`ECC_RECON_LAYER` (`ECC_LAYERS`, which now has a per-model spelling).

ONLY `recon_ert_arm` REACHES THE MAPPER. Everything else here is evaluator-side
and must never enter the cache key: a bar re-costed at a different reconstruction
energy is the same mapping, and pretending otherwise would cold the matrix for
an arithmetic change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..contracts.errors import ConfigError
from .env import _b, _f, _of, _s, _table

#: How the reduced representation is physically stored and transported, and how
#: encoder work is charged. Both are `recon.py`'s, and both are answers to
#: sections 15/16 of `02_reconstruction_dse_and_implementation.txt` rather than
#: free parameters -- the docstrings there say what each one claims.
RECON_PACKINGS = ("stream", "aligned")

RECON_GRANULARITIES = ("weight", "codeword")

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
    #: prompt_6 RULE 3: the two Design Compiler tables from env.sh section 4
    #: (flattened by section 10), `{configuration id: pJ}` -- incremental per
    #: codeword, idle per cycle per engine. Read by `ecc.load_recon_terms`
    #: when the JSON has no entry for the (N,K) in play. ECC_RECON_INCLUDE_IDLE
    #: is retired: the two terms are never added, each has its own denominator.
    recon_incremental_table: dict
    recon_idle_table: dict
    recon_pj_override: Optional[float]
    recon_incremental_fallback_pj: float
    recon_idle_fallback_pj: float
    #: prompt_7 Issue 4. Percentage of the reconstruction engine's
    #: CLOCKED-IDLE energy that clock gating removes. 0 reproduces the
    #: pre-gating model exactly; 99.5 is the measured clock/dynamic share.
    recon_clock_gating_pct: float
    recon_packing: str
    recon_granularity: str
    #: `destination` | `source` -- see RECON_ENCODER_SITES and
    #: `recon.ENCODER_SITES`. Only network boundaries depend on it.
    recon_encoder_site: str
    #: ECC_RECON_ERT_AWARE (prompt_6 8.1): the ERT arms get their own mapping,
    #: solved with the encoder's energy in the objective, and the figure marks
    #: which bars came from one. It was coupled to RECON_OPTIMIZER=True and
    #: ECC_PHASE=Post; both knobs are gone (EnvReorganisation phases 2 and 3)
    #: and the coupling with them, because every placement is now mapped on
    #: its own chip and the result phase is derived per arm.
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
            recon_idle_table=_table("ECC_RECON_IDLE_PJ_LIST"),
            recon_pj_override=_of("ECC_RECON_PJ"),
            recon_incremental_fallback_pj=_f("ECC_RECON_INCREMENTAL_FALLBACK_PJ", 1.8995),
            recon_idle_fallback_pj=_f("ECC_RECON_IDLE_FALLBACK_PJ", 2.2301273),
            recon_clock_gating_pct=_f("ECC_RECON_CLOCK_GATING_PCT", 99.5),
            recon_packing=_s("ECC_RECON_PACKING", "stream").lower(),
            recon_granularity=_s("ECC_RECON_ENCODER_GRANULARITY", "weight").lower(),
            recon_encoder_site=_s("ECC_RECON_ENCODER_SITE", "destination").lower(),
            recon_ert_aware=_b("ECC_RECON_ERT_AWARE", False),
            recon_ert_arm=_s("ECC_RECON_ERT_ARM").strip().lower(),
            recon_bw_scale=_b("ECC_RECON_BW_SCALE", False),
        )
