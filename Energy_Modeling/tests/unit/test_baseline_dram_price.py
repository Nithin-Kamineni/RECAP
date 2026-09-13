"""`physics/baseline_dram.py`'s three price helpers -- and the point of narrowing them.

WHY THIS FILE IS SHORT AND NEEDS NO CONFIG. Until phase 7 these three took a
whole `Config`, read ONE knob out of it, and could therefore only be tested by
building a configuration -- which needs the environment, the designs on disk and
`config._resolve()`. `Config` is L3 and this module is L1, so the signature was
also claiming a dependency the layer rule forbids; it only worked because Python
does not check.

They take `cfg.code` and `cfg.energy` now (`ProjectRestructure.md` section 4.3,
the incremental half phase 4 deliberately left). The payoff is exactly this file:
a stub with two attributes, no environment, no filesystem, no mapper.

WHAT THEY DECIDE. `ECC_BASELINE_DRAM_PJ_PER_BIT` is a PRICE, not traffic: the
baseline's array is bigger and does indexing work the other two arms do not, so a
bit out of it costs more, and decoding is on the DRAM die so the parity never
crosses the datapath (`FINDINGS`, DRAM cost model). `enabled()` is the switch
between that model and the pre-2026-09-10 one that charged parity TRAFFIC
instead, and getting it backwards would double-count the baseline arm.
"""
from dataclasses import dataclass
from typing import Optional

import pytest

from eccenergy.physics import baseline_dram


@dataclass(frozen=True)
class Code:
    """What `dram_ert_pj_per_bit` actually needs. Compare `CodeSettings`."""
    weight_bits: int = 8


@dataclass(frozen=True)
class Energy:
    """What `enabled` actually needs. Compare `EnergySettings`."""
    baseline_dram_pj_per_bit: Optional[float] = None


@dataclass
class Raw:
    """The fields of a raw record these three read, and nothing else."""
    dram_w_reads: float = 1000.0
    e_dram_w: float = 64000.0            # 64 pJ per 8-bit word -> 8 pJ/bit
    dram: dict = None


def test_the_ert_rate_is_energy_over_bits_not_over_accesses():
    """Timeloop counts a DRAM access in units of the DATASPACE DATAWIDTH, so the
    weight rows give pJ/bit only after dividing by `weight_bits` as well. Missing
    that factor is an 8x error in the single largest category of most bars."""
    assert baseline_dram.dram_ert_pj_per_bit(Raw(), Code(8)) == pytest.approx(8.0)


def test_the_documented_lpddr4_figure_comes_out():
    """64.0 pJ per 8-bit word = 8.0 pJ/bit = the documented 512 pJ per 64-bit
    access, which is the number `env.sh` section 6 calls "Accelergy CactiDRAM
    LPDDR4 as modelled"."""
    rate = baseline_dram.dram_ert_pj_per_bit(Raw(), Code(8))
    assert rate * 64 == pytest.approx(512.0)


def test_the_rate_tracks_the_payload_width():
    """The same energy over the same accesses is HALF the per-bit rate when each
    access carries twice the bits."""
    assert baseline_dram.dram_ert_pj_per_bit(Raw(), Code(16)) == pytest.approx(4.0)


def test_a_record_with_no_weight_traffic_has_no_rate_rather_than_zero():
    """None means "this record cannot tell you", which is what makes `price()`
    refuse instead of guessing. Returning 0.0 would price the baseline at nothing
    and look like a saving."""
    assert baseline_dram.dram_ert_pj_per_bit(Raw(dram_w_reads=0), Code()) is None


# ------------------------------------------------------------- which model?
def test_the_price_model_is_off_until_the_knob_is_set():
    """Unset is the pre-2026-09-10 model (parity charged as TRAFFIC). It is not
    the same as zero and must not be confused with it."""
    assert baseline_dram.enabled(Energy(None)) is False
    assert baseline_dram.enabled(Energy(40.0)) is True


def test_a_zero_baseline_price_still_COUNTS_AS_SET():
    """THE ABLATION MUST REACH THE MODEL IT ABLATES. `zero-price` (GUARDS.md,
    tier 4) lets `ECC_BASELINE_DRAM_PJ_PER_BIT=0` run; if `enabled()` read that
    as "unset" the run would silently fall back to the TRAFFIC model and measure
    something else entirely."""
    assert baseline_dram.enabled(Energy(0.0)) is True


# ------------------------------------------------- what this record was charged
def test_the_recorded_charge_wins_over_the_ert():
    """`apply_dram_override()` writes what it charged; that is the truth for this
    record, and re-deriving it from the ERT would ignore ECC_DRAM_PJ_PER_BIT."""
    raw = Raw(dram={"pj_per_bit_charged": 20.0})
    assert baseline_dram.charged_pj_per_bit(raw, Code()) == 20.0


def test_a_record_that_never_went_through_the_override_falls_back_to_the_ert():
    assert baseline_dram.charged_pj_per_bit(Raw(), Code()) == pytest.approx(8.0)
    assert baseline_dram.charged_pj_per_bit(Raw(dram={}), Code()) == pytest.approx(8.0)


def test_the_fallback_also_fires_when_the_recorded_charge_is_falsy():
    """A recorded 0.0 or None cannot be a price to scale FROM -- scaling by
    `tgt/0` is what `price()` refuses -- so it falls through to the ERT."""
    assert baseline_dram.charged_pj_per_bit(
        Raw(dram={"pj_per_bit_charged": 0.0}), Code()) == pytest.approx(8.0)
