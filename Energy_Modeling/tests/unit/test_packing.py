"""`physics/packing.py` -- whether a reduced weight actually costs less, and where.

Section 16 of `02_reconstruction_dse_and_implementation.txt` is the reason this
module is not a one-liner: *"Reducing weights from 8 bits to 4 bits reduces SRAM
energy by 50%"* is not a claim the hardware supports unless the representation is
exploited physically. `stream` is the optimistic reading and `aligned` the
pessimistic bound it hides, and the difference is the whole result.

The pair that matters most here is `narrowing_owner` / `storage_scale`: prompt_6
RULE 1 says exactly one of the mapper and the evaluator narrows a level, decided
by MEASURING the level's own `Word bits`. Both narrowing SQUARES the saving --
which is the `narrow-once` guard in `GUARDS.md`, and is a defect that reached a
figure once.
"""
import math

import pytest

from eccenergy.physics.packing import Packing

WB = 8
STREAM = Packing(mode="stream", weight_bits=WB, k=39, n=63)
ALIGNED = Packing(mode="aligned", weight_bits=WB, k=39, n=63)


def test_the_rate_is_k_over_n_in_both_modes():
    assert STREAM.frac == pytest.approx(39 / 63)
    assert ALIGNED.frac == STREAM.frac


def test_stream_keeps_the_fractional_width_and_aligned_rounds_up():
    """`stream` stores a PACKED k-bit field, so the width need not be whole;
    `aligned` gives each weight a whole number of bits and pays the remainder."""
    assert STREAM.reduced_bits_per_weight == pytest.approx(8 * 39 / 63)
    assert ALIGNED.reduced_bits_per_weight == math.ceil(8 * 39 / 63) == 5


def test_declared_q_is_the_width_table_q_and_not_the_aligned_ceiling():
    """THE WIDTH TABLE rounds (`round`), `aligned` takes `ceil`, and at BCH(63,39)
    they agree at 5. They do NOT agree everywhere -- BCH(63,57) is round 7 against
    ceil 8 -- so the two must be read from their own functions, never swapped."""
    assert STREAM.declared_q == 5
    hi = Packing(mode="aligned", weight_bits=8, k=57, n=63)
    assert hi.declared_q == 7 and hi.reduced_bits_per_weight == 8, (
        "at BCH(63,57) `aligned` rounds up to the FULL width and declares no "
        "narrowing at all, while q is 7 -- the one code where they disagree")


# ------------------------------------------- prompt_6 RULE 1: ONE owner, measured
def test_a_plan_at_the_full_width_leaves_the_narrowing_to_the_evaluator():
    assert STREAM.narrowing_owner(WB) == "evaluator"


def test_a_plan_already_at_q_owns_its_own_narrowing():
    assert STREAM.narrowing_owner(STREAM.declared_q) == "mapper"


def test_no_measurement_keeps_the_pre_prompt_6_reading():
    """`None` is "there are no stats to measure from", not "zero bits"."""
    assert STREAM.narrowing_owner(None) == "evaluator"
    assert STREAM.narrowing_owner(0) == "evaluator"


def test_a_word_width_that_is_neither_stops_the_run():
    """The arch and the code disagreeing is not something to pick a winner for:
    no owner can be assigned, so the bar cannot be built."""
    with pytest.raises(ValueError, match="neither the weight width"):
        STREAM.narrowing_owner(6)


def test_the_evaluator_applies_nothing_when_the_mapper_already_narrowed():
    """THE SQUARING BUG, as one assertion. A mapper-narrowed level must scale by
    exactly 1.0 -- applying Packing on top would count the saving twice."""
    assert STREAM.storage_scale(96, word_bits=STREAM.declared_q) == 1.0
    assert ALIGNED.storage_scale(96, word_bits=ALIGNED.declared_q) == 1.0


def test_stream_scales_storage_by_exactly_n_over_k_on_an_eight_bit_plan():
    """Values per physical word rise by n/k when the field is packed, so the
    access count -- and the energy -- falls by k/n."""
    assert STREAM.storage_scale(96, word_bits=WB) == pytest.approx(39 / 63)


def test_aligned_can_deliver_no_storage_saving_at_all_and_that_is_the_point():
    """Section 16's warning, measured. A 24-bit word holds floor(24/5) = 4 five-bit
    values against floor(24/8) = 3 eight-bit ones, so SOME width wins -- but a
    width where the floor does not move gives exactly 1.0, and the optimistic
    reading hides that."""
    assert ALIGNED.storage_scale(24, word_bits=WB) == pytest.approx(3 / 4)
    #  16 bits: floor(16/8) = 2 full weights, floor(16/5) = 3 reduced
    assert ALIGNED.storage_scale(16, word_bits=WB) == pytest.approx(2 / 3)
    #  a word that fits the SAME count either way saves nothing
    nine = Packing(mode="aligned", weight_bits=8, k=57, n=63)   # ceil -> 8 bits
    assert nine.storage_scale(64, word_bits=WB) == 1.0, (
        "at BCH(63,57) `aligned` stores 8-bit values, so no word holds more")


def test_wire_scale_is_the_rate_because_a_wire_carries_bits_not_words():
    """Wire energy is per BIT moved, so it falls in both modes -- there is no word
    to round to. This is why `aligned` can save on the wire and nothing on SRAM,
    and why the two scales are separate functions rather than one number."""
    assert STREAM.wire_scale() == pytest.approx(39 / 63)
    assert ALIGNED.wire_scale() == pytest.approx(5 / 8)


def test_switching_is_per_flit_so_it_falls_only_with_repacking():
    """The third scale, and the one that tells the two modes apart on the NoC: a
    router bills per flit, so `aligned` moves it only when the floor moves."""
    assert STREAM.switching_scale(64) == pytest.approx(39 / 63)
    same = Packing(mode="aligned", weight_bits=8, k=57, n=63)   # ceil -> 8 bits
    assert same.switching_scale(64) == 1.0, (
        "a reduced weight that still occupies 8 bits packs no more per flit")


def test_to_dict_names_the_mode_so_a_result_says_which_reading_it_used():
    """One plan per bar (prompt_6 RULE 4): a result that does not record its
    packing mode cannot be compared with one that used the other reading."""
    d = STREAM.to_dict()
    assert d["mode"] == "stream"
    assert d["full_bits_per_weight"] == 8
    assert d["reduced_bits_per_weight"] == pytest.approx(8 * 39 / 63)
    assert ALIGNED.to_dict()["mode"] == "aligned"
