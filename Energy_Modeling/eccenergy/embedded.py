"""Embedded ECC: the codeword layout the embedding pipeline actually produces,
and the DRAM storage/traffic accounting Task 2 needs from it.

Task 2 of `03_staged_implementation_plan.txt`:

    "Add an embedded-ECC evaluation mode with parity contained within the
     stored weight representation and no external parity for this fully
     embedded case. Use the actual embedded-codeword layout rather than
     treating all full weight bits as independent payload. Read the complete
     embedded codeword for ECC correction; do not discard its embedded parity
     before that correction. ... Distinguish stored bits from physical reads:
     less storage does not guarantee fewer DRAM bursts or proportional energy."

THE ACTUAL LAYOUT (read from the embedding code, not chosen here)
------------------------------------------------------------------
The weights are embedded by ECC-CODE-Engine/4-EmbeddingECC (`ecc_embed.py`,
approach `replace`, `chunk_size = message_parity_size` = the codeword n); its
slicing and encoding routines are copied verbatim into
`../Input_Embedding/3-Testing/{utils,implementations}`. Three facts about that
code decide everything in this module:

1. `convert_to_binary(vals, bit_size=8)` turns the int8 tensor into ONE
   MSB-first bit stream, 8 bits per weight.
2. `messageSliceBasedOnChunkSize(bits, chunk_size=n)` cuts that stream into
   fixed n-bit chunks -- n = 63 for every K of the sweep -- and a chunk
   deliberately spans several weights and slices them mid-value. Only the
   FINAL chunk is zero-padded ("Pad this chunk to exactly chunk_size with
   zeros").
3. `ParityOverwriteByTopWeightsEncode(chunk, n, k)` builds the systematic
   BCH(n,k) generator, keeps the k HIGHEST-significance positions of the chunk
   as the message and OVERWRITES the n-k lowest-significance positions with
   the parity. The parity lives inside the weights' least significant bits.

So the embedded representation of W weights occupies exactly W x weight_bits
in DRAM, plus the zero padding of one final chunk per tensor: no external
parity, no shortened code, and NOT "floor(n / 8) whole weights per codeword".
Weights straddle codeword boundaries: n / weight_bits = 7.875 weights per
BCH(63,K) codeword, whatever K is. Because gcd(63, 8) = 1 the alignment repeats
every lcm(63, 8) = 504 bits = 8 codewords = 63 weights, and 7 of every 8
codeword boundaries fall inside a weight.

This is the opposite packing rule from the conventional baseline in
`parity.py`, and BOTH are right: the baseline stores a codeword beside the
data, so it may pack only whole weights into the k-bit message field; the
embedded pipeline cuts the weight stream itself, so the codeword boundary
falls wherever n bits end. Neither count is a modelling choice.

WHAT THIS MEANS FOR DRAM
------------------------
* STORED bits: payload + tail padding. External parity: ZERO. The n-k parity
  bits per codeword are weight bits that already exist; they are REPORTED as
  `embedded_parity_bits` so a reader sees where the parity went, and are never
  added to the footprint.
* READ bits: to correct a weight the ECC engine needs its whole codeword(s),
  message and parity alike. Timeloop already bills every one of the 8 bits of
  every DRAM weight read, so the embedded arm's DRAM weight traffic is EXACTLY
  the Timeloop weight traffic -- not K/N of it (that would be discarding the
  embedded parity before correction, which the plan forbids) and not more.
* ENERGY: the Timeloop DRAM weight energy, unchanged. What the embedded arm
  removes, relative to Task 1's conventional baseline, is the external-parity
  traffic `parity.py` charges; nothing else moves.

STORED vs PHYSICAL vs ENERGY -- kept apart on purpose
------------------------------------------------------
Timeloop counts scalar (8-bit) accesses and bills 1/8 of a 64-bit word per
scalar, so everything downstream of it is bit-proportional: a tile that ends
mid-word is charged for the bits it used, not the word it occupied. This
module reports the physical word count as an ESTIMATE (bits / word, rounded up
once at the end) beside the bit count and says so in `method`, rather than
pretending a burst-exact figure exists. The tail padding of the final chunk
is reported but not charged, which is how `parity.py` treats the baseline's
tail padding, so the two arms are billed by one rule. Both approximations are
written into every result under `approximations`.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

#: Where the layout comes from, recorded in every result so "this is the actual
#: layout" is a checkable claim rather than an assertion.
LAYOUT_SOURCE = {
    "pipeline": ("ECC-CODE-Engine/4-EmbeddingECC/ecc_embed.py -- approach `replace`, "
                 "chunk_size = message_parity_size (the codeword n); one int8 tensor "
                 "per call"),
    "slicing": ("Input_Embedding/3-Testing/utils/messageSliceBasedOnChunkSize.py -- "
                "fixed n-bit chunks over the MSB-first 8-bit weight stream; chunks "
                "span weights; only the last chunk is zero-padded"),
    "encoding": ("Input_Embedding/3-Testing/implementations/"
                 "ParityOverwriteByTopWeightsEncode.py -- systematic BCH(n,k) via "
                 "galois.BCH(n,k).G; the k highest-significance chunk positions are "
                 "the message, the n-k lowest-significance positions are overwritten "
                 "with the parity"),
    "phases": ("Input_Embedding/3-Testing/implementations/secded_vectorized.py -- "
               "the chunk/weight alignment repeats every lcm(n, 8)/n chunks"),
    "verified": "2026-09-07, read-only inspection of the files above",
}


def _lcm(a, b):
    return a * b // math.gcd(a, b)


@dataclass(frozen=True)
class EmbeddedLayout:
    """BCH(n, k) embedded in a stream of `weight_bits`-bit weights.

    The codeword IS n consecutive bits of the weight stream. Everything here is
    exact integer arithmetic on that statement; the tensor-level counts are
    built from it in `account()`.
    """
    n: int
    k: int
    weight_bits: int

    # ---------------------------------------------------------------- code
    @property
    def parity_bits(self):
        """n - k parity positions per codeword. They are WEIGHT bits."""
        return self.n - self.k

    @property
    def codeword_bits(self):
        """What one codeword occupies in DRAM: n, all of it weight bits."""
        return self.n

    @property
    def independent_frac(self):
        """k/n of the weight bits are free; the rest are set by the parity."""
        return self.k / self.n

    # -------------------------------------------------------------- packing
    @property
    def weights_per_codeword(self):
        """n / weight_bits -- FRACTIONAL, because weights straddle codewords.

        7.875 at n = 63 over 8-bit weights, for every K. Compare
        `parity.CodeGeometry.weights_per_codeword`, which is `k // weight_bits`
        because the baseline's codeword is stored separately and packs only
        whole weights. Different layouts, different counts, both exact.
        """
        return self.n / self.weight_bits

    @property
    def period_bits(self):
        """After lcm(n, weight_bits) bits the chunk/weight alignment repeats."""
        return _lcm(self.n, self.weight_bits)

    @property
    def period_codewords(self):
        return self.period_bits // self.n

    @property
    def period_weights(self):
        return self.period_bits // self.weight_bits

    @property
    def split_boundaries_per_period(self):
        """Codeword boundaries per period that fall INSIDE a weight.

        Boundary j sits at bit j*n and is weight-aligned iff weight_bits | j*n,
        i.e. iff period_codewords | j. So every boundary but the period's last
        splits a weight: 7 of every 8 at (63, 8).
        """
        return self.period_codewords - 1

    # --------------------------------------------- where the parity lands
    def local_significances(self, phase):
        """Bit significance (MSB = weight_bits-1 ... LSB = 0) of each position of
        the chunk that starts `phase` chunks into the stream.

        Mirrors `secded_vectorized._local_weights`: the significance of global
        bit g is `weight_bits - 1 - (g % weight_bits)` for an MSB-first stream.
        """
        start = phase * self.n
        w = self.weight_bits
        return [w - 1 - ((start + p) % w) for p in range(self.n)]

    def parity_positions(self, phase):
        """Chunk positions overwritten with parity in this phase.

        Mirrors `ParityOverwriteByTopWeightsEncode`: message = the k positions
        of highest significance (ties broken by lower index), parity = the
        rest. Sorted ascending for a stable record.
        """
        sig = self.local_significances(phase)
        ranked = sorted(range(self.n), key=lambda i: (-sig[i], i))
        message = set(ranked[:self.k])
        return [i for i in range(self.n) if i not in message]

    def parity_significance_profile(self):
        """How many parity bits land on each bit significance, per phase and
        averaged per weight over one period.

        This is the "actual embedded-codeword layout" in one table: at
        BCH(63,51) every weight gives up its LSB (bit0 = 1.000 per weight) and
        about half give up bit1 as well (0.524); at BCH(63,30) bits 0-3 are
        gone from every weight and part of bit4. It is what "do not treat all
        full weight bits as independent payload" refers to.
        """
        per_phase = []
        totals = {s: 0 for s in range(self.weight_bits)}
        for ph in range(self.period_codewords):
            sig = self.local_significances(ph)
            counts = {}
            for pos in self.parity_positions(ph):
                counts[sig[pos]] = counts.get(sig[pos], 0) + 1
            per_phase.append({f"bit{s}": c for s, c in sorted(counts.items())})
            for s, c in counts.items():
                totals[s] += c
        per_weight = {f"bit{s}": totals[s] / self.period_weights
                      for s in range(self.weight_bits) if totals[s]}
        return {
            "rule": ("per chunk, the n-k parity bits overwrite the positions of "
                     "lowest bit significance (ties: higher chunk index first)"),
            "phases": self.period_codewords,
            "parity_slots_by_significance_per_phase": per_phase,
            "parity_bits_per_weight_by_significance": per_weight,
            "parity_bits_per_weight_total": (self.period_codewords * self.parity_bits
                                             / self.period_weights),
        }

    # ------------------------------------------------------------ utility
    def validate(self):
        if self.k >= self.n:
            raise ValueError(f"BCH needs k < n; got n={self.n} k={self.k}")
        if self.k < 1:
            raise ValueError(f"BCH needs k >= 1; got k={self.k}")
        if self.weight_bits <= 0:
            raise ValueError("weight_bits must be positive")
        return self

    def to_dict(self):
        d = asdict(self)
        d.update(
            layout="bitstream",
            layout_meaning=("the MSB-first weight bit stream is cut into fixed n-bit "
                            "codewords; weights straddle codeword boundaries; the "
                            "parity overwrites the lowest-significance positions; "
                            "storage is weight_bits per weight with no external parity"),
            parity_bits_per_codeword=self.parity_bits,
            codeword_bits=self.codeword_bits,
            weights_per_codeword=self.weights_per_codeword,
            independent_frac=self.independent_frac,
            period_bits=self.period_bits,
            period_codewords=self.period_codewords,
            period_weights=self.period_weights,
            split_boundaries_per_period=self.split_boundaries_per_period,
            parity_significance=self.parity_significance_profile(),
        )
        return d


@dataclass(frozen=True)
class EmbeddedAccount:
    """Stored bits for one group of weights under the embedded layout.

    `payload_bits` is every weight bit, message and parity alike -- the parity
    is inside it. `embedded_parity_bits` says how many of those bits the code
    dictates; it is NOT additional storage. `external_parity_bits` is zero by
    construction and is carried so the two arms' records have the same shape.
    """
    weights: int
    payload_bits: int
    codewords: int
    tail_pad_bits: int             # zero padding of the final chunk
    embedded_parity_bits: int      # weight bits carrying parity, INSIDE payload_bits
    split_boundaries: int          # codeword boundaries that fall inside a weight
    straddling_weights: int        # weights that span two (or more) codewords
    external_parity_bits: int      # 0
    stored_dram_words: int
    dram_word_bits: int

    @property
    def stored_bits(self):
        """Everything the tensor occupies in DRAM: payload + tail padding."""
        return self.payload_bits + self.tail_pad_bits

    @property
    def overhead_frac(self):
        return self.tail_pad_bits / self.payload_bits if self.payload_bits else 0.0

    @property
    def bits_per_weight(self):
        return self.stored_bits / self.weights if self.weights else 0.0

    def to_dict(self):
        d = asdict(self)
        d.update(stored_bits=self.stored_bits, overhead_frac=self.overhead_frac,
                 bits_per_weight=self.bits_per_weight,
                 payload_bytes=self.payload_bits / 8.0,
                 external_parity_bytes=0.0,
                 stored_bytes=self.stored_bits / 8.0)
        return d


def _straddles(codewords, layout):
    """(split boundaries, straddling weights) for `codewords` consecutive chunks.

    Interior boundaries are j = 1 .. codewords-1 at bit j*n; the final boundary
    ends the padded stream and splits nothing. Counted per period plus a short
    remainder, so it is O(period) rather than O(codewords); `hand_check()` is
    the O(codewords) walk that proves it.
    """
    if codewords <= 1:
        return 0, 0
    n, wb, P = layout.n, layout.weight_bits, layout.period_codewords
    last = codewords - 1
    full = last // P
    # one full period: boundaries j = 1..P-1 all split (P does not divide j);
    # the weights they hit are distinct across periods because a period is a
    # whole number of weights
    per_period_split = P - 1
    per_period_weights = len({(j * n) // wb for j in range(1, P)})
    rem = range(full * P + 1, last + 1)
    rem_split = sum(1 for j in rem if (j * n) % wb)
    rem_weights = len({(j * n) // wb for j in rem if (j * n) % wb})
    return (full * per_period_split + rem_split,
            full * per_period_weights + rem_weights)


def account(weights, layout, dram_word_bits=64):
    """Exact stored footprint of `weights` weights under `layout`.

    `weights` may be fractional (a Timeloop count scaled by a layer repeat) and
    is rounded UP to a whole weight first, as `parity.account()` does.
    """
    layout.validate()
    w = int(math.ceil(weights - 1e-9))
    if w < 0:
        raise ValueError(f"weights must be non-negative, got {weights}")
    n = layout.n
    payload_bits = w * layout.weight_bits
    codewords = int(math.ceil(payload_bits / n)) if payload_bits else 0
    tail_pad_bits = codewords * n - payload_bits
    split, straddle = _straddles(codewords, layout)
    stored_bits = payload_bits + tail_pad_bits
    words = int(math.ceil(stored_bits / dram_word_bits)) if stored_bits else 0
    return EmbeddedAccount(
        weights=w, payload_bits=payload_bits, codewords=codewords,
        tail_pad_bits=tail_pad_bits,
        embedded_parity_bits=codewords * layout.parity_bits,
        split_boundaries=split, straddling_weights=straddle,
        external_parity_bits=0, stored_dram_words=words,
        dram_word_bits=dram_word_bits)


def account_layers(layer_weights, layout, dram_word_bits=64, grouping="layer"):
    """Aggregate `account()` over per-layer weight counts.

    `layer` (default): each layer's tensor is its own bit stream and pays its
    own tail padding -- which is what the embedding driver does, encoding one
    tensor per call. `model`: one stream for everything, one tail. The same
    knob (`ECC_PARITY_GROUPING`) the baseline uses, so the two arms group alike.
    """
    if grouping not in ("layer", "model"):
        raise ValueError(f"grouping must be 'layer' or 'model', got {grouping!r}")
    if grouping == "model":
        return account(sum(layer_weights), layout, dram_word_bits)
    parts = [account(w, layout, dram_word_bits) for w in layer_weights]
    if not parts:
        return account(0, layout, dram_word_bits)
    return EmbeddedAccount(
        weights=sum(p.weights for p in parts),
        payload_bits=sum(p.payload_bits for p in parts),
        codewords=sum(p.codewords for p in parts),
        tail_pad_bits=sum(p.tail_pad_bits for p in parts),
        embedded_parity_bits=sum(p.embedded_parity_bits for p in parts),
        split_boundaries=sum(p.split_boundaries for p in parts),
        straddling_weights=sum(p.straddling_weights for p in parts),
        external_parity_bits=0,
        stored_dram_words=sum(p.stored_dram_words for p in parts),
        dram_word_bits=dram_word_bits)


def traffic_account(dram_weight_reads, stored, layout, dram_word_bits=64):
    """What crosses the DRAM boundary for the embedded arm, given measured reads.

    Timeloop's scalar weight reads already ARE the complete-codeword traffic:
    each scalar is one whole 8-bit weight, every bit of which is a codeword
    bit, message or parity. Nothing is discarded before correction and nothing
    external is added. The refetch factor (reads / unique weights) is measured
    and applied to the stored codeword count so the codeword READS are
    reported on the same footing as `parity.traffic_account()`.

    APPROXIMATION, STATED. `physical_dram_words_estimate` is bits / word,
    rounded up once at the end. Timeloop bills per scalar and does not report
    which word a tile's last scalar fell in, so a burst-exact count is not
    available; this number is what a bit-proportional model implies and is
    labelled as such. `tail_pad_bits_read` is reported, not charged, exactly as
    the baseline's tail padding is not charged by `parity.py`.
    """
    empty = {"refetch_factor": 0.0, "codeword_reads": 0.0, "payload_bits_read": 0.0,
             "embedded_parity_bits_read": 0.0, "tail_pad_bits_read": 0.0,
             "external_bits": 0.0, "external_dram_words": 0.0,
             "external_dram_scalars": 0.0, "physical_dram_words_estimate": 0.0,
             "overhead_vs_payload_frac": 0.0, "method": "no weight traffic measured"}
    if stored.weights <= 0 or dram_weight_reads <= 0:
        return empty
    refetch = dram_weight_reads / stored.weights
    codeword_reads = stored.codewords * refetch
    payload_bits_read = dram_weight_reads * layout.weight_bits
    return {
        "refetch_factor": refetch,
        "codeword_reads": codeword_reads,
        "payload_bits_read": payload_bits_read,
        "embedded_parity_bits_read": codeword_reads * layout.parity_bits,
        "tail_pad_bits_read": stored.tail_pad_bits * refetch,
        "external_bits": 0.0,
        "external_dram_words": 0.0,
        "external_dram_scalars": 0.0,
        "physical_dram_words_estimate": float(math.ceil(payload_bits_read / dram_word_bits)),
        "overhead_vs_payload_frac": 0.0,
        "method": ("complete codeword read: every bit of every DRAM weight read that "
                   "Timeloop counts is billed, message and embedded parity alike, and "
                   "no K/N reduction is applied at DRAM; no external parity; physical "
                   "DRAM words are ESTIMATED as bits / word rounded up once at the end "
                   "(bit-proportional -- Timeloop bills per 8-bit scalar); the final "
                   "chunk's tail padding is reported, not charged, as for the baseline"),
    }


def dram_energy_pj(e_dram_weights_pj, dram_weight_reads):
    """(weight-traffic energy, external-parity energy, pJ per weight scalar).

    The first is Timeloop's DRAM weight energy, unchanged: the complete codeword
    is what Timeloop already billed. The second is 0.0 by construction. The
    third is the same per-scalar figure `parity.parity_energy_pj` reports and
    must match it, and match across architectures at one node and datawidth.
    """
    per_scalar = (e_dram_weights_pj / dram_weight_reads) if dram_weight_reads > 0 else 0.0
    return float(e_dram_weights_pj), 0.0, per_scalar


# ------------------------------------------------------------------ hand check
def hand_check(layout, weights, dram_word_bits=64):
    """Recompute `account()` the long way -- boundary by boundary -- and compare.

    Walks the padded stream one codeword at a time, asking of every interior
    boundary whether it lands inside a weight and which weight, instead of
    using the period arithmetic in `_straddles()`. A mistake in the closed form
    cannot hide in both. Returns (ok, detail); the caller stores `detail` so
    the check is auditable after the fact.
    """
    fast = account(weights, layout, dram_word_bits)
    n, wb = layout.n, layout.weight_bits
    w = int(math.ceil(weights - 1e-9))
    payload = w * wb

    cw = split = straddle = 0
    b = 0
    last_weight = -1
    while b < payload:
        cw += 1
        b += n
        if b < payload and b % wb:          # an interior boundary inside a weight
            split += 1
            wi = b // wb
            if wi != last_weight:
                straddle += 1
                last_weight = wi
    tail = cw * n - payload if cw else 0
    stored = payload + tail
    words = int(math.ceil(stored / dram_word_bits)) if stored else 0

    checks = {
        "codewords": (cw, fast.codewords),
        "payload_bits": (payload, fast.payload_bits),
        "tail_pad_bits": (tail, fast.tail_pad_bits),
        "stored_bits": (stored, fast.stored_bits),
        "split_boundaries": (split, fast.split_boundaries),
        "straddling_weights": (straddle, fast.straddling_weights),
        "stored_dram_words": (words, fast.stored_dram_words),
        "external_parity_bits": (0, fast.external_parity_bits),
    }
    mismatches = {k: {"walked": a, "closed_form": b}
                  for k, (a, b) in checks.items() if a != b}
    detail = {
        "method": ("codeword-by-codeword walk of the padded weight bit stream, "
                   "classifying every interior boundary, compared against the "
                   "closed form; both must agree exactly"),
        "code": f"BCH({layout.n},{layout.k}) embedded in {wb}-bit weights",
        "weights_checked": w,
        "weights_per_codeword": layout.weights_per_codeword,
        "checks": {k: {"walked": a, "closed_form": b} for k, (a, b) in checks.items()},
        "mismatches": mismatches,
        "passed": not mismatches,
    }
    return (not mismatches), detail


def hand_check_layers(layout, layer_weights, dram_word_bits=64, grouping="layer"):
    """`hand_check()` at the grouping the account used, aggregated to one record.

    Under `layer` grouping each tensor is walked separately and the walked
    counts are summed, so the check exercises the per-layer tail padding the
    account actually charges rather than a single model-wide stream.
    """
    if grouping == "model" or len(layer_weights) <= 1:
        ok, detail = hand_check(layout, sum(layer_weights) if layer_weights else 0,
                                dram_word_bits)
        detail["grouping"] = grouping
        detail["layers_checked"] = 1 if layer_weights else 0
        return ok, detail

    keys = None
    walked = {}
    closed = {}
    all_ok = True
    for w in layer_weights:
        ok, d = hand_check(layout, w, dram_word_bits)
        all_ok = all_ok and ok
        if keys is None:
            keys = list(d["checks"])
            walked = {k: 0 for k in keys}
            closed = {k: 0 for k in keys}
        for k in keys:
            walked[k] += d["checks"][k]["walked"]
            closed[k] += d["checks"][k]["closed_form"]
    mismatches = {k: {"walked": walked[k], "closed_form": closed[k]}
                  for k in keys if walked[k] != closed[k]}
    return (all_ok and not mismatches), {
        "method": ("per-layer codeword-by-codeword walk (each tensor its own "
                   "stream) summed over layers, compared against the summed "
                   "closed form; both must agree exactly"),
        "code": f"BCH({layout.n},{layout.k}) embedded in {layout.weight_bits}-bit weights",
        "grouping": grouping,
        "layers_checked": len(layer_weights),
        "weights_checked": sum(int(math.ceil(w - 1e-9)) for w in layer_weights),
        "weights_per_codeword": layout.weights_per_codeword,
        "checks": {k: {"walked": walked[k], "closed_form": closed[k]} for k in keys},
        "mismatches": mismatches,
        "passed": all_ok and not mismatches,
    }
