"""`physics/parity.py` -- Task 1's bit accounting, as PROPERTIES.

THIS FILE IS FROZEN (CLAUDE.md, "Working on this code"): `parity.py` may move but
its contents may not change. That makes tests here unusually valuable and
unusually cheap -- the thing under test is not going to move under them -- and it
is why they are written as properties rather than as pinned numbers:
`ProjectRestructure.md` section 7.2 asks for exactly that trade, because a pinned
number fails whenever K moves and tells you nothing about why.

`ParityAccount`, `overhead_frac` and `parity_energy_pj` had no test at all. The
invariants below are the ones the arm's whole claim rests on: every bit is
classified exactly once, padding is never negative, and the parity traffic is
billed at the SAME measured per-access cost as a weight rather than at a second
guessed one.
"""
import pytest

from eccenergy.physics import parity

GEOM = parity.CodeGeometry(n=63, k=39, weight_bits=8)
CODES = [(63, k) for k in (30, 36, 39, 45, 51, 57)]


def geom(n, k, weight_bits=8):
    return parity.CodeGeometry(n=n, k=k, weight_bits=weight_bits)


# ------------------------------------------------------- the conservation law
@pytest.mark.parametrize("n,k", CODES)
@pytest.mark.parametrize("weights", [1, 7, 8, 9, 1000, 589824])
def test_every_stored_bit_is_classified_exactly_once(n, k, weights):
    """THE ACCOUNTING INVARIANT. `stored_bits` is payload + parity + the two
    paddings, and nothing may be counted twice or fall between them. If this
    holds, the overhead fraction cannot be quietly wrong."""
    a = parity.account(weights, geom(n, k))
    assert a.stored_bits == (a.payload_bits + a.parity_bits
                             + a.message_pad_bits + a.tail_pad_bits)


@pytest.mark.parametrize("n,k", CODES)
@pytest.mark.parametrize("weights", [1, 7, 8, 9, 1000, 589824])
def test_no_component_of_the_account_is_ever_negative(n, k, weights):
    """A negative padding is how an off-by-one in the tail shows up: the totals
    still add up, and the overhead comes out too LOW."""
    a = parity.account(weights, geom(n, k))
    for field, value in a.to_dict().items():
        if isinstance(value, (int, float)):
            assert value >= 0, f"{field} = {value} at BCH({n},{k}), {weights} weights"


@pytest.mark.parametrize("n,k", CODES)
def test_the_payload_is_exactly_the_weights_and_nothing_else(n, k):
    """Padding belongs to overhead, never to payload -- otherwise the arm would
    be charged for its own padding twice and credited for it once."""
    a = parity.account(1000, geom(n, k))
    assert a.payload_bits == 1000 * 8


def test_a_fractional_weight_count_still_occupies_a_whole_codeword_slot():
    """`weights` can arrive as a Timeloop access count scaled by a layer repeat.
    Rounding DOWN would give the arm a free weight."""
    assert parity.account(8.2, GEOM).weights == 9
    assert parity.account(8.0, GEOM).weights == 8
    assert parity.account(0, GEOM).codewords == 0


def test_a_negative_weight_count_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        parity.account(-1, GEOM)


# ---------------------------------------------------------------- monotonicity
def test_more_parity_never_costs_less():
    """Lower K is a stronger code and must never come out CHEAPER. A sign error or
    a swapped n/k shows up here and nowhere else in a single run.

    Non-decreasing, not strictly increasing -- see the test below for why.
    """
    prev = -1.0
    for n, k in sorted(CODES, key=lambda c: -c[1]):        # 57 -> 30
        frac = parity.account(10_000, geom(n, k)).overhead_frac
        assert frac >= prev, f"BCH({n},{k}) overhead {frac} fell below {prev}"
        prev = frac


def test_the_external_overhead_is_BANDED_by_floor_k_over_8_not_set_by_k():
    """AND THE BAND IS WHY THE TEST ABOVE IS NOT STRICT. In the external layout

        overhead = (parity + message_pad) / used
                 = ((n - k) + (k - 8*floor(k/8))) / (8*floor(k/8))
                 = (n - 8*floor(k/8)) / (8*floor(k/8))

    -- `k` cancels. The overhead depends only on `n` and on how many WHOLE
    weights fit in the message field, so every code in one band costs exactly the
    same in DRAM. BCH(63,39) and BCH(63,36) both hold four weights per codeword
    and both inflate the tensor by 0.96875, though their ideal rates are 0.615
    and 0.750.

    THAT IS NOT A ROUNDING ARTEFACT, it is what the layout charges: the three
    extra parity bits BCH(63,36) carries come out of message bits BCH(63,39) was
    already wasting as padding. Quoting the ideal rate as the arm's DRAM cost
    understates it by 22% at BCH(63,39) and hides the banding completely.
    """
    by_band = {}
    for n, k in CODES:
        g = geom(n, k)
        by_band.setdefault(g.weights_per_codeword, []).append(
            (k, parity.account(589_824, g).overhead_frac))
    shared = {b: v for b, v in by_band.items() if len(v) > 1}
    assert shared, "no band holds two of this study's codes -- has CODES changed?"
    for band, rows in shared.items():
        fracs = {round(f, 12) for _k, f in rows}
        assert len(fracs) == 1, (
            f"{[k for k, _ in rows]} all hold {band} weights per codeword and must "
            f"cost the same, got {fracs}")

    #  The closed form is the LIMIT -- it drops the tail padding, which is the
    #  one part that does amortise -- so it is checked at a real layer's size.
    for n, k in CODES:
        g = geom(n, k)
        assert (parity.account(589_824, g).overhead_frac
                == pytest.approx((n - 8 * (k // 8)) / (8 * (k // 8)), rel=1e-3))


def test_the_real_overhead_is_never_below_the_ideal_rate():
    """The ideal `n/k - 1` assumes the message field is fully packed with weights.
    It is not: 39 bits hold four whole 8-bit weights and waste seven, because a
    weight split across two codewords would need both corrected before either was
    usable. So the real overhead is never BELOW the ideal, and a value under it
    means padding was subtracted somewhere."""
    for n, k in CODES:
        g = geom(n, k)
        a = parity.account(589_824, g)
        assert a.overhead_frac >= g.parity_overhead_ideal_frac - 1e-12


def test_the_overhead_converges_on_the_per_codeword_rate_and_not_on_the_ideal():
    """THE GAP IS MESSAGE PADDING, AND IT DOES NOT AMORTISE AWAY. Only the TAIL
    padding is amortised over a big tensor; the wasted `k % weight_bits` bits are
    paid by EVERY codeword. At BCH(63,39) that is 0.75 against an ideal 0.615 --
    a 22% understatement if the ideal were quoted as the arm's cost."""
    for n, k in CODES:
        g = geom(n, k)
        per_codeword = ((g.parity_bits + g.message_pad_bits)
                        / g.message_used_bits)
        assert parity.account(589_824, g).overhead_frac == pytest.approx(
            per_codeword, rel=1e-3)

    g = geom(63, 39)
    assert g.message_used_bits == 32 and g.message_pad_bits == 7
    assert parity.account(589_824, g).overhead_frac == pytest.approx(0.96875, rel=1e-3)
    assert g.parity_overhead_ideal_frac == pytest.approx(0.6154, rel=1e-3)


def test_the_two_layouts_count_weights_per_codeword_differently_on_purpose():
    """THE ONE CONFUSION THIS MODULE CANNOT AFFORD. `parity.py` is the EXTERNAL
    layout, where a codeword's k-bit message holds `k // weight_bits` WHOLE
    weights (4 at BCH(63,39)) because no weight may straddle two codewords.
    `embedded.py` is the EMBEDDED layout, where the codeword is n consecutive
    bits of the weight bit stream and the count is fractional (7.875). They are
    different numbers for different arms, and swapping them would re-rate one arm
    with the other's geometry."""
    from eccenergy.physics.embedded import EmbeddedLayout

    assert GEOM.weights_per_codeword == 39 // 8 == 4
    assert EmbeddedLayout(63, 39, 8).weights_per_codeword == pytest.approx(63 / 8)


def test_an_empty_account_has_no_overhead_rather_than_dividing_by_zero():
    assert parity.account(0, GEOM).overhead_frac == 0.0


# -------------------------------------------------- DRAM transfer granularity
def test_parity_is_read_in_whole_dram_words_because_a_partial_word_costs_a_full_one():
    """PHYSICAL, not arithmetic. External parity is a contiguous region and the
    controller moves whole words; charging the exact bit count would under-bill
    the baseline arm, which is the arm this study compares against."""
    a = parity.account(8, GEOM, dram_word_bits=64)
    assert a.parity_bits <= a.parity_dram_words * 64
    assert a.parity_dram_words == -(-a.parity_bits // 64)


def test_a_wider_dram_word_never_moves_fewer_bits_for_the_same_parity():
    narrow = parity.account(1000, GEOM, dram_word_bits=32)
    wide = parity.account(1000, GEOM, dram_word_bits=256)
    assert narrow.parity_bits == wide.parity_bits
    assert wide.parity_dram_words * 256 >= wide.parity_bits
    assert narrow.parity_dram_words * 32 >= narrow.parity_bits


# ---------------------------------------------------------- the energy is a RATE
def test_parity_is_billed_at_the_measured_cost_of_a_weight_not_a_second_guess():
    """Timeloop already reports what one weight read out of THIS DRAM costs,
    including the design's word width and packing. Parity comes out of the same
    DRAM through the same controller, so it is billed at that rate -- which is
    what makes the baseline arm comparable across architectures at all."""
    e, per_scalar = parity.parity_energy_pj(
        {"external_dram_scalars": 500.0}, e_dram_weights_pj=2000.0,
        dram_weight_reads=1000)
    assert per_scalar == pytest.approx(2.0)
    assert e == pytest.approx(1000.0)


def test_the_per_scalar_rate_is_independent_of_how_much_parity_there_is():
    """It is a PRICE. If it moved with the traffic, the two would be multiplied
    together somewhere and the parity term would grow quadratically."""
    _, a = parity.parity_energy_pj({"external_dram_scalars": 1.0}, 2000.0, 1000)
    _, b = parity.parity_energy_pj({"external_dram_scalars": 10 ** 9}, 2000.0, 1000)
    assert a == b == pytest.approx(2.0)


def test_no_weight_reads_means_no_parity_energy_and_no_division_by_zero():
    """A layer scope that reads no weights must charge nothing, not crash and not
    inherit the previous layer's rate."""
    assert parity.parity_energy_pj({"external_dram_scalars": 5.0}, 0.0, 0) == (0.0, 0.0)
