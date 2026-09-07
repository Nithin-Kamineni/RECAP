"""External BCH parity accounting: grouping, padding, DRAM granularity.

Task 1 of `03_staged_implementation_plan.txt` asks for the conventional-ECC
baseline -- weights are the protected payload, parity lives in DRAM beside them
-- and is specific about how to count it:

    "For BCH(n,k), each k payload bits requires n-k parity bits. Account for
     actual grouping, padding and physical DRAM transfer granularity."

WHY THAT IS NOT `n/k - 1`
-------------------------
The pre-Task-1 model inflated DRAM weight energy by a flat `n/k - 1`. That is
the *ideal* code rate and it is wrong in two directions at once:

* A codeword message field is `k` bits, but weights are `w` bits and a weight
  cannot straddle a codeword. At BCH(63,51) with 8-bit weights only
  `floor(51/8) = 6` whole weights fit, using 48 of the 51 message bits. The
  remaining 3 bits are PADDING. The real overhead is `12/48 = 25.00%`, not
  `63/51 - 1 = 23.53%`. The flat model understates the baseline by ~6% of the
  parity cost.
* The last codeword of a weight tensor is padded out too, and parity is read
  from DRAM in whole 64-bit words. Both round the cost UP.

Whether the padding is worth this much care is a fair question for a
whole-model run, where it is a sub-percent effect on the total. It is not a
fair question for a one-layer development run, where it is the difference
between a number that reconciles by hand and one that does not.

WHAT IS DELIBERATELY *NOT* MODELLED HERE
----------------------------------------
External parity is consumed by off-chip ECC correction. It never enters the
on-chip weight SRAM/RF, so nothing in this module touches on-chip energy.
`ECC_BASELINE_INFLATES_ONCHIP=1` still exists to reproduce the older behaviour
but is not the conventional baseline and says so.

TWO COUNTS, AND THEY ARE DIFFERENT
----------------------------------
* STORED parity is set by the number of UNIQUE weights. It is a capacity
  statement -- how much DRAM the protected model occupies.
* PARITY TRAFFIC is set by DRAM weight READS, which include refetch. A weight
  that crosses the DRAM boundary four times drags its parity across four times.

Energy uses traffic. Storage is reported because it is what a reader will
sanity-check against "the model is 11.7 MB".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class CodeGeometry:
    """BCH(n, k) over `weight_bits`-bit weights, with the packing resolved.

    Everything here is exact integer arithmetic on one codeword; the layer- and
    model-level counts are built from it.
    """
    n: int
    k: int
    weight_bits: int

    @property
    def parity_bits(self):
        """r = n - k, the parity a single codeword carries."""
        return self.n - self.k

    @property
    def weights_per_codeword(self):
        """WHOLE weights that fit in the k-bit message field.

        Integer floor, not `k / weight_bits`. A weight split across two
        codewords would need both to be corrected before either is usable,
        which is not what a systematic BCH layout does.
        """
        return self.k // self.weight_bits

    @property
    def message_used_bits(self):
        """Message bits actually carrying weights."""
        return self.weights_per_codeword * self.weight_bits

    @property
    def message_pad_bits(self):
        """Message bits wasted per codeword because a weight did not fit."""
        return self.k - self.message_used_bits

    @property
    def parity_overhead_frac(self):
        """Parity bits per USEFUL payload bit -- the real inflation factor.

        Compare with the ideal `n/k - 1`, which assumes the message field is
        fully packed. `parity_overhead_ideal_frac` is that number, kept so the
        two can be printed side by side.
        """
        return self.parity_bits / self.message_used_bits

    @property
    def parity_overhead_ideal_frac(self):
        return self.n / self.k - 1.0

    def validate(self):
        if self.k >= self.n:
            raise ValueError(f"BCH needs k < n; got n={self.n} k={self.k}")
        if self.weight_bits <= 0:
            raise ValueError("weight_bits must be positive")
        if self.weights_per_codeword < 1:
            raise ValueError(
                f"BCH({self.n},{self.k}) has a {self.k}-bit message field, which "
                f"cannot hold even one {self.weight_bits}-bit weight. Either "
                f"raise K or lower ECC_WEIGHT_BITS.")
        return self

    def to_dict(self):
        d = asdict(self)
        d.update(parity_bits_per_codeword=self.parity_bits,
                 weights_per_codeword=self.weights_per_codeword,
                 message_used_bits=self.message_used_bits,
                 message_pad_bits_per_codeword=self.message_pad_bits,
                 parity_overhead_frac=self.parity_overhead_frac,
                 parity_overhead_ideal_frac=self.parity_overhead_ideal_frac)
        return d


@dataclass(frozen=True)
class ParityAccount:
    """Payload/parity/padding for one group of weights, in bits and DRAM words."""
    weights: int
    payload_bits: int
    codewords: int
    message_pad_bits: int          # unused message bits inside codewords
    tail_pad_bits: int             # padding in the final, partly-filled codeword
    parity_bits: int
    parity_dram_words: int
    parity_dram_scalars: float     # parity expressed as weight-sized DRAM accesses
    dram_word_bits: int

    @property
    def stored_bits(self):
        """Everything the protected tensor occupies in DRAM."""
        return self.payload_bits + self.parity_bits + self.message_pad_bits + self.tail_pad_bits

    @property
    def overhead_frac(self):
        return (self.stored_bits - self.payload_bits) / self.payload_bits if self.payload_bits else 0.0

    def to_dict(self):
        d = asdict(self)
        d.update(stored_bits=self.stored_bits, overhead_frac=self.overhead_frac,
                 payload_bytes=self.payload_bits / 8.0,
                 parity_bytes=self.parity_bits / 8.0,
                 stored_bytes=self.stored_bits / 8.0)
        return d


def account(weights, geom, dram_word_bits=64):
    """Exact payload/parity/padding for `weights` weights under `geom`.

    `weights` may be fractional (it can come from a Timeloop access count
    scaled by a layer repeat), so it is rounded UP to a whole weight before the
    codeword count is taken -- a partial weight still occupies a codeword slot.
    """
    geom.validate()
    w = int(math.ceil(weights - 1e-9))
    if w < 0:
        raise ValueError(f"weights must be non-negative, got {weights}")
    per_cw = geom.weights_per_codeword
    codewords = int(math.ceil(w / per_cw)) if w else 0

    payload_bits = w * geom.weight_bits
    # The last codeword is padded out to a whole message field.
    tail_pad_bits = codewords * geom.message_used_bits - payload_bits
    message_pad_bits = codewords * geom.message_pad_bits
    parity_bits = codewords * geom.parity_bits

    # PHYSICAL DRAM TRANSFER GRANULARITY. External parity is a contiguous
    # region read in whole DRAM words; a partial word still costs a full one.
    parity_words = int(math.ceil(parity_bits / dram_word_bits)) if parity_bits else 0
    scalars_per_word = dram_word_bits / geom.weight_bits

    return ParityAccount(
        weights=w, payload_bits=payload_bits, codewords=codewords,
        message_pad_bits=message_pad_bits, tail_pad_bits=tail_pad_bits,
        parity_bits=parity_bits, parity_dram_words=parity_words,
        parity_dram_scalars=parity_words * scalars_per_word,
        dram_word_bits=dram_word_bits)


def account_layers(layer_weights, geom, dram_word_bits=64, grouping="layer"):
    """Aggregate `account()` over a list of per-layer weight counts.

    `grouping` decides where codeword boundaries fall:

    `layer` (default)
        Each layer's weight tensor is its own codeword stream, so each layer
        pays its own tail padding. This is what a real allocator does -- layers
        are separate tensors at separate addresses -- and it is the
        conservative choice.

    `model`
        One stream over the whole model: a single tail padding for everything.
        The difference is at most one codeword per layer, i.e. nothing on a
        full model and visible on a one-layer development run. Offered so the
        sensitivity can be measured rather than assumed.
    """
    if grouping not in ("layer", "model"):
        raise ValueError(f"grouping must be 'layer' or 'model', got {grouping!r}")
    if grouping == "model":
        return account(sum(layer_weights), geom, dram_word_bits)

    parts = [account(w, geom, dram_word_bits) for w in layer_weights]
    if not parts:
        return account(0, geom, dram_word_bits)
    return ParityAccount(
        weights=sum(p.weights for p in parts),
        payload_bits=sum(p.payload_bits for p in parts),
        codewords=sum(p.codewords for p in parts),
        message_pad_bits=sum(p.message_pad_bits for p in parts),
        tail_pad_bits=sum(p.tail_pad_bits for p in parts),
        parity_bits=sum(p.parity_bits for p in parts),
        parity_dram_words=sum(p.parity_dram_words for p in parts),
        parity_dram_scalars=sum(p.parity_dram_scalars for p in parts),
        dram_word_bits=dram_word_bits)


def traffic_account(dram_weight_reads, stored, geom, dram_word_bits=64,
                    charge_padding=True):
    """Non-payload bits that cross the DRAM boundary, given measured reads.

    A weight refetched from DRAM must be re-corrected, so its codeword's parity
    crosses with it. The refetch factor is measured -- Timeloop's DRAM weight
    reads divided by the unique weights -- and applied to the exact stored
    codeword count.

    WHAT COUNTS AS OVERHEAD TRAFFIC. Per codeword, `message_used_bits` carry
    weights and `n - message_used_bits` do not: `n - k` parity bits plus the
    `message_pad_bits` no whole weight could use. Both sit inside the stored
    codeword, so both cross the bus when the codeword is read. At BCH(63,51)
    with 8-bit weights that is 12 + 3 = 15 non-payload bits per 48 payload
    bits, i.e. 31.25% -- against the 23.53% a flat `n/k` model charges.

    `charge_padding=False` bills parity only (23.53%-equivalent), for the case
    where the layout packs weights across codeword boundaries in DRAM and
    re-splits them at the ECC engine. That is a different memory layout, not a
    cheaper version of this one, so it is not the default.

    APPROXIMATION, STATED. Timeloop reports scalar reads, not the tile
    boundaries they fall on, so this cannot know how many codewords a given
    tile fetch straddles. The codeword count is scaled by the measured refetch
    factor and rounded to whole DRAM words ONCE, at the end, rather than per
    tile. That understates granularity rounding when tiles are small relative
    to a codeword. Recorded in every result JSON under `approximations`.
    """
    empty = {"refetch_factor": 0.0, "codeword_reads": 0.0, "parity_bits": 0.0,
             "message_pad_bits": 0.0, "external_bits": 0.0,
             "external_dram_words": 0.0, "external_dram_scalars": 0.0,
             "charge_padding": charge_padding,
             "method": "no weight traffic measured"}
    if stored.weights <= 0 or dram_weight_reads <= 0:
        return empty

    refetch = dram_weight_reads / stored.weights
    codeword_reads = stored.codewords * refetch
    parity_bits = codeword_reads * geom.parity_bits
    pad_bits = codeword_reads * geom.message_pad_bits
    external_bits = parity_bits + (pad_bits if charge_padding else 0.0)
    words = math.ceil(external_bits / dram_word_bits)
    return {
        "refetch_factor": refetch,
        "codeword_reads": codeword_reads,
        "parity_bits": parity_bits,
        "message_pad_bits": pad_bits,
        "external_bits": external_bits,
        "external_dram_words": float(words),
        "external_dram_scalars": words * (dram_word_bits / geom.weight_bits),
        "charge_padding": charge_padding,
        "overhead_vs_payload_frac": (
            external_bits / (codeword_reads * geom.message_used_bits)
            if codeword_reads else 0.0),
        "method": ("exact stored codeword count x measured DRAM refetch factor; "
                   "parity" + (" + message padding" if charge_padding else "")
                   + "; rounded up to whole DRAM words once"),
    }


def parity_energy_pj(traffic, e_dram_weights_pj, dram_weight_reads):
    """Energy for the external-parity traffic, at the same per-access cost as a weight.

    Timeloop already reports what one 8-bit weight read out of this DRAM costs,
    including this design's word width and packing. Parity is read from the
    same DRAM by the same controller, so it is billed at that measured rate
    rather than at a second, independently guessed one.

    Returns (energy_pJ, pJ_per_scalar). `pJ_per_scalar` is worth printing: it
    must be identical across architectures at the same node and DRAM datawidth,
    and a 2x difference means one design declares its DRAM differently.
    """
    if dram_weight_reads <= 0:
        return 0.0, 0.0
    per_scalar = e_dram_weights_pj / dram_weight_reads
    return traffic["external_dram_scalars"] * per_scalar, per_scalar


# ------------------------------------------------------------------ hand check
def hand_check(geom, weights, dram_word_bits=64):
    """Recompute `account()` the long way and prove the two agree.

    Task 1: "a small hand-check confirms payload/parity accounting". This is
    that check, mechanised so it runs on every result rather than once in a
    notebook. It walks codeword by codeword -- filling whole weights until the
    message field cannot take another -- instead of using the closed form, so a
    mistake in the closed form cannot hide in both.

    Returns (ok, detail_dict). The caller stores `detail_dict` in the result
    JSON so the check is auditable after the fact, not just at run time.
    """
    fast = account(weights, geom, dram_word_bits)

    remaining = int(math.ceil(weights - 1e-9))
    cw = used = pad_msg = 0
    while remaining > 0:
        take = min(remaining, geom.weights_per_codeword)
        used += take * geom.weight_bits
        # bits of THIS codeword's message field left unused: the ones a whole
        # weight could not use, plus the ones no weight was left to fill
        pad_msg += geom.k - take * geom.weight_bits
        remaining -= take
        cw += 1

    slow_parity_bits = cw * geom.parity_bits
    slow_parity_words = int(math.ceil(slow_parity_bits / dram_word_bits)) if cw else 0
    # the long walk lumps tail padding in with message padding; the closed form
    # splits them, so compare the sum
    fast_pad_total = fast.message_pad_bits + fast.tail_pad_bits

    checks = {
        "codewords": (cw, fast.codewords),
        "payload_bits": (used, fast.payload_bits),
        "padding_bits_total": (pad_msg, fast_pad_total),
        "parity_bits": (slow_parity_bits, fast.parity_bits),
        "parity_dram_words": (slow_parity_words, fast.parity_dram_words),
    }
    mismatches = {k: {"walked": a, "closed_form": b}
                  for k, (a, b) in checks.items() if a != b}
    detail = {
        "method": ("codeword-by-codeword walk compared against the closed form; "
                   "both must agree exactly"),
        "code": f"BCH({geom.n},{geom.k}) over {geom.weight_bits}-bit weights",
        "weights_checked": fast.weights,
        "weights_per_codeword": geom.weights_per_codeword,
        "message_pad_bits_per_codeword": geom.message_pad_bits,
        "checks": {k: {"walked": a, "closed_form": b} for k, (a, b) in checks.items()},
        "mismatches": mismatches,
        "passed": not mismatches,
    }
    return (not mismatches), detail
