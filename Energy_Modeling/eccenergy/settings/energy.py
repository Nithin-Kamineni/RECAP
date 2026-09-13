"""What a bit, a MAC and a cycle cost -- the prices the evaluator charges.

The DRAM per-bit price and the baseline arm's dearer one, the two static DRAM
terms, standby power and the roofline switch, the MAC denominator, and the
classifier that decides which category a level's energy lands in.

NOTHING HERE REACHES THE MAPPER BY VALUE, and that is the point: a price change
re-costs a cached mapping in milliseconds. `energy_model_rev` is the deliberate
exception -- it colds the matrix BY DECREE when an estimator changes, which is
the one case where the price list really did change the numbers under a
fingerprint that could not see it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..contracts.errors import ConfigError
from .env import _b, _f, _i, _list, _of, _oi, _one, _s, _table

#: The clock the reconstruction datapath's DC power report was measured at, in
#: ns. Every entry of `data/dc/BCH_N63_results.json` carries it as
#: `measurement.clock_period_ns` and every one of them is 1.0; `ecc.py` checks
#: the entry it actually reads against this and refuses on a mismatch rather
#: than rescaling from the wrong base. env.sh section 6, TRAP 2.
DC_MEASUREMENT_CLOCK_NS = 1.0

#: The fields of this group the MAPPER sees, and therefore the ones
#: `Config.fingerprint()` hashes.
#:
#: Nothing here reaches the mapper by VALUE. `energy_model_rev` reaches it by
#: DECREE -- `Config.fingerprint()` appends it only when it is non-empty, so
#: EMPTY hashes byte-identically to every fingerprint that predates the knob
#: and any other value colds the whole matrix on purpose.
IN_FINGERPRINT = frozenset({
})


@dataclass(frozen=True)
class EnergySettings:
    #: prompt_7, 2026-09-12. Free-text revision of the ENERGY MODEL itself --
    #: the Accelergy plug-in stack, its configs, anything that changes what a
    #: component COSTS without changing the architecture YAML. EMPTY means
    #: 'whatever the image shipped', and is hashed as if the field did not
    #: exist, so every fingerprint predating this knob is preserved exactly.
    energy_model_rev: str
    #: ECC_DRAM_PJ_PER_BIT: pJ per bit of DYNAMIC DRAM access. Rescales the
    #: whole DRAM category evaluator-side (energy.apply_dram_override), exactly
    #: as mac_pj_override rescales Compute. None = leave Accelergy's own
    #: constant (8 pJ/bit for LPDDR4 as modelled) alone. The f_if array/interface
    #: split this replaced is GONE: the whole DRAM weight term scales by K/N.
    dram_pj_per_bit: Optional[float]
    #: ECC_BASELINE_DRAM_PJ_PER_BIT: pJ per bit of DYNAMIC DRAM access for the
    #: CONVENTIONAL-ECC BASELINE ARM ONLY (baseline_dram.charge). Its array is
    #: bigger -- parity is stored beside the weights -- and it does indexing
    #: work the other two arms do not, so a bit out of it costs more: 70
    #: against 40. It is a PRICE, not traffic: decoding is on the DRAM die, so
    #: the baseline drives the same weight bits off it as the embedded arm and
    #: the parity never crosses the datapath. None = the pre-2026-09-10 model
    #: (baseline at `dram_pj_per_bit`, charged the external-parity traffic).
    baseline_dram_pj_per_bit: Optional[float]
    #: The other two terms of E_total(DRAM) = E_dynamic + E_background + E_refresh.
    #: Both 0 for now, on purpose (the study's question is on-chip energy) --
    #: modelling them is a TODO and would give the embedded arm further credit,
    #: since it holds fewer weight bits in DRAM. Units: pJ per bit-second and
    #: pJ per bit per refresh window.
    dram_background_pj: float
    dram_refresh_pj: float
    #: ECC_STATIC_ENERGY (env.sh section 6). 1 = charge COMPONENT STANDBY
    #: ENERGY -- the accelerator's own leakage -- as a physical category,
    #: `Standby`, to ALL THREE ARMS. prompt_7 Defect 2: the reconstruction
    #: engines are billed standby power from a Design Compiler run while the
    #: accelerator beside them is billed none, because `parse_stats` never read
    #: `Leakage energy (total)`, so the comparison charged one side only.
    #: 0 (the default) reproduces every pre-Phase-A total to the pJ: the
    #: category is not even in `phys_cats()`, so a cached raw record still
    #: loads. Not in the mapper fingerprint -- this is evaluator arithmetic
    #: over a mapping the mapper already chose.
    static_energy: bool
    #: ECC_LEAKAGE_NW, flattened by env.sh section 10 into
    #: ECC_LEAKAGE_NW_LIST. Replacement leakage densities in nW: `sram_bit`
    #: and `rf_bit` per STORED BIT, `mac_instance` per MAC. They replace the
    #: ERT's own `leak` rows, which are 10^3-10^4 too low and in two cases
    #: exactly 0 (prompt_7 section 5.2c: CACTI pinned to `itrs-lstp`, an
    #: Aladdin table whose register leakage is a literal 0, and a Neurosim
    #: plug-in that answered 0 pJ). POWER, not energy: the cycle period is
    #: applied at the point of use, never here (env.sh section 6 TRAP 2).
    leakage_nw: dict
    #: ECC_LATENCY_MODEL. 1 = re-time the chosen mapping with
    #: `latency_post.roofline()`. Evaluator-only and NOT in the fingerprint,
    #: exactly like the two `noc_post` terms: the plan is Timeloop's, and this
    #: states how long that plan takes once off-chip bandwidth is declared.
    latency_model: bool
    baseline_inflates_onchip: bool
    split_read_write: bool
    classify_mode: str
    #: ECC_MAC_PJ_OVERRIDE (env.sh section 5): rescale the Compute category to
    #: MACs x this value in the EVALUATOR. None = the ERT's value. It is the
    #: denominator of every ECC percentage and nothing else: the saved pJ do not
    #: depend on it (energy.apply_mac_override, tests/test_mac_override.py).
    #: Not in the mapper fingerprint -- the MAC count is mapping-invariant.
    mac_pj_override: Optional[float]

    @staticmethod
    def from_env():
        """Every `ECC_*` knob of this group, read once."""
        return EnergySettings(
            dram_pj_per_bit=_of("ECC_DRAM_PJ_PER_BIT"),
            baseline_dram_pj_per_bit=_of("ECC_BASELINE_DRAM_PJ_PER_BIT"),
            dram_background_pj=_f("ECC_DRAM_BACKGROUND_PJ", 0.0),
            dram_refresh_pj=_f("ECC_DRAM_REFRESH_PJ", 0.0),
            static_energy=_b("ECC_STATIC_ENERGY", False),
            leakage_nw=_table("ECC_LEAKAGE_NW_LIST"),
            latency_model=_b("ECC_LATENCY_MODEL", False),
            mac_pj_override=_of("ECC_MAC_PJ_OVERRIDE"),
            energy_model_rev=_s("ECC_ENERGY_MODEL_REV", ""),
            baseline_inflates_onchip=_b("ECC_BASELINE_INFLATES_ONCHIP", False),
            split_read_write=_b("ECC_SPLIT_READ_WRITE", False),
            classify_mode=_s("ECC_CLASSIFY", "instances").lower(),
        )
