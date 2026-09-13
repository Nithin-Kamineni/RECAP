"""`physics/granularity.py` -- what a codeword is, and how many weights it touches.

Section 15: *"The encoder may operate at codeword granularity rather than
independent weight granularity ... reconstructing one weight can require retained
bits from several weights."* Two numbers come out of that and they are different
numbers, which is the thing this module exists to keep straight:

  `weights_per_codeword`  n / weight_bits, FRACTIONAL (7.875 at BCH(63,K) over
                          8-bit weights). How many weights one codeword's worth
                          of bits amounts to. The embedded arm counts DRAM
                          codewords with the same number, so the two arms cannot
                          disagree about what a codeword is.
  `g_rec`                 how many weights one codeword TOUCHES -- 9, not 7.875
                          and not 8 -- because a codeword that starts mid-weight
                          reaches into one more weight than its length implies.
                          This is what a PE must hold at once for a local
                          boundary to be feasible at all.

Confusing them understates what a PE-local boundary costs, which is exactly the
kind of error that reaches a figure as a plausible energy.
"""
import math

import pytest

from eccenergy.physics.granularity import Granularity

WEIGHT = Granularity(n=63, k=39, weight_bits=8, mode="weight")
CODEWORD = Granularity(n=63, k=39, weight_bits=8, mode="codeword")


def test_weights_per_codeword_is_fractional_and_is_n_over_weight_bits():
    """7.875, not 8. Rounding it here would silently re-rate the embedded arm,
    which counts its DRAM codewords from the same property."""
    assert WEIGHT.weights_per_codeword == pytest.approx(63 / 8)
    assert WEIGHT.weights_per_codeword != 8


def test_g_rec_is_one_more_than_the_length_implies():
    """THE POINT OF SECTION 15. 63 bits spans ceil(63/8) = 8 whole weights only
    when it starts on a weight boundary; every other phase reaches into a 9th."""
    assert WEIGHT.g_rec == 9
    assert WEIGHT.g_rec == math.ceil(63 / 8) + 1


def test_g_rec_is_the_worst_phase_and_not_the_average():
    """It sizes a PE's holding requirement, so it is a MAXIMUM. An average would
    make an infeasible boundary look feasible."""
    aligned = Granularity(n=64, k=39, weight_bits=8, mode="weight")
    assert aligned.g_rec == 8, (
        "at n=64 over 8-bit weights every codeword starts on a boundary, so no "
        "phase reaches a 9th weight -- the +1 is a consequence, not a constant")


def test_g_rec_does_not_depend_on_k():
    """`k` is the RATE; which weights a codeword touches is set by `n` and the
    payload width alone."""
    for k in (30, 36, 39, 45, 51, 57):
        assert Granularity(n=63, k=k, weight_bits=8, mode="weight").g_rec == 9


def test_weight_mode_amortises_the_group_and_codeword_mode_does_not():
    """The two charging readings, and the factor between them is
    `weights_per_codeword`. `weight` is the amortized reading (the group is
    rebuilt once and all of it is consumed); `codeword` is the pessimistic one,
    and the right one if nothing buffers the group."""
    assert WEIGHT.codewords(7875) == pytest.approx(1000.0)
    assert CODEWORD.codewords(7875) == 7875.0
    assert CODEWORD.codewords(100) / WEIGHT.codewords(100) == pytest.approx(63 / 8)


def test_codeword_mode_is_never_cheaper_than_weight_mode():
    """A pessimistic reading that came out cheaper would be a sign error, and the
    bar would be drawn under the wrong heading either way."""
    for accesses in (1, 7, 8, 1000, 10 ** 6):
        assert CODEWORD.codewords(accesses) >= WEIGHT.codewords(accesses)


def test_zero_accesses_charge_nothing_in_either_mode():
    assert WEIGHT.codewords(0) == 0.0 and CODEWORD.codewords(0) == 0.0


def test_to_dict_carries_both_numbers_and_says_they_are_different():
    """A result must record which of the two it charged AND what the other was --
    a reader cannot infer either from the bar."""
    d = WEIGHT.to_dict()
    assert d["charging"] == "weight"
    assert d["weights_per_codeword"] == pytest.approx(63 / 8)
    assert d["G_rec_weights_touched_per_codeword"] == 9
    assert d["weights_per_codeword"] != d["G_rec_weights_touched_per_codeword"]
    assert "section 15" in d["source"]
