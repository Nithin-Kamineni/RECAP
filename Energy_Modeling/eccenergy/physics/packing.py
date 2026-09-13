"""The physical packing model: how a reduced weight is actually stored.

02_reconstruction_dse_and_implementation.txt section 16. A `Packing` answers the
one question that decides whether a narrowing is real: storing a q-bit value in
an 8-bit field halves nothing.

    stream    every reduced stage is scaled by K/N -- the bits are packed end to
              end and a weight may straddle a word boundary.
    aligned   every weight gets whole bits, `q = round(8K/N)` of them, so the
              saving is `q / weight_bits` and never more.

`aligned` is the default (`ECC_RECON_PACKING`), and it is what makes
`recon.assert_onchip_narrowing_once()`'s square-saving check meaningful: the
mapper and the evaluator can both narrow, and doing both SQUARES the saving.

PURE ARITHMETIC. No file I/O, no cache, no configuration beyond the numbers
handed in -- which is why ProjectRestructure phase 3 filed it at L1, out of the
`recon.py` that used to hold it beside the code that reads the mapper cache.
"""
from __future__ import annotations

import math

from . import widths
from dataclasses import dataclass


# ===========================================================================
#  the physical packing model  (02_..., section 16)
# ===========================================================================
@dataclass(frozen=True)
class Packing:
    """How the reduced representation is physically stored and transported.

    Section 16 of `02_reconstruction_dse_and_implementation.txt` is blunt about
    why this cannot be waved through: "Reducing weights from 8 bits to 4 bits
    reduces SRAM energy by 50%" is not a claim the hardware supports unless the
    representation is exploited physically. Two models, and the run says which
    one it used:

    `stream`  the retained portion of each n-bit codeword is stored and
              transported as a PACKED k-bit field. This is the layout the
              embedding pipeline already produces -- the codeword IS n
              consecutive bits of the weight bit stream, so its message is k
              consecutive bits with no per-weight alignment to preserve
              (`embedded.py`). Values per physical word and operands per flit
              therefore rise by exactly n/k, and every stage scales by k/n.

    `aligned` each reduced weight occupies a whole number of bits,
              `ceil(weight_bits x k/n)`, and nothing is repacked across word or
              flit boundaries. Wire energy still falls with the bit count, but
              a 24-bit scratchpad word holds floor(24/7) = 3 seven-bit values,
              which is what it already held at 8 bits -- so the access count,
              and the SRAM energy, do not move at all. This is the pessimistic
              bound section 16 warns the optimistic one hides.
    """
    mode: str
    weight_bits: int
    k: int
    n: int

    @property
    def frac(self):
        return self.k / self.n

    @property
    def reduced_bits_per_weight(self):
        if self.mode == "stream":
            return self.weight_bits * self.frac
        return math.ceil(self.weight_bits * self.frac)

    def _per_word(self, block_bits, bits):
        if block_bits <= 0 or bits <= 0:
            return 1.0
        return (block_bits / bits) if self.mode == "stream" \
            else float(max(1, math.floor(block_bits / bits)))

    @property
    def declared_q(self):
        """The mapper's reduced datawidth, `q = round(weight_bits x K/N)` --
        what `archs._set_weight_datawidth` writes and what a q-bit plan's
        stats print as `Word bits`."""
        return widths.declared_datawidth(self.n, self.k, self.weight_bits)

    def narrowing_owner(self, word_bits):
        """prompt_6 RULE 1: who narrows a storage stop, decided by MEASUREMENT.

            Word bits == q            ->  "mapper"    (the plan already narrowed it;
                                                      the evaluator applies 1.0)
            Word bits == weight_bits  ->  "evaluator" (an 8-bit plan; Packing applies)
            anything else             ->  STOP: the arch and the code disagree.

        `None` (no stats to measure from) keeps the pre-prompt_6 reading, the
        evaluator. Ownership is a property of the BAR's own plan, never a
        run-wide switch -- there is no ECC_RECON_NARROW_AT knob.
        """
        if word_bits in (None, 0):
            return "evaluator"
        wb = int(word_bits)
        if wb == self.weight_bits:
            return "evaluator"
        if wb == self.declared_q:
            return "mapper"
        raise ValueError(
            f"a weight level reports Word bits {wb}, which is neither the weight "
            f"width {self.weight_bits} nor q = round({self.weight_bits} x {self.k}/"
            f"{self.n}) = {self.declared_q}: the arch and the code disagree, and "
            f"no owner can be assigned to its narrowing (prompt_6 RULE 1)")

    def storage_scale(self, block_bits, word_bits=None):
        """Access-count-and-energy scale for a storage stage.

        `word_bits` is the level's MEASURED `Word bits` from the bar's own
        stats (RULE 1). When the mapper already narrowed the level the scale
        is exactly 1.0 -- applying Packing on top would square the saving.
        """
        if self.narrowing_owner(word_bits) == "mapper":
            return 1.0
        full = self._per_word(block_bits, self.weight_bits)
        red = self._per_word(block_bits, self.reduced_bits_per_weight)
        return full / red if red > 0 else 1.0

    def narrowing_site(self, word_bits, applied_scale):
        """One narrow stop's audit row: which sites are LIVE, given the scale
        the evaluator actually applied. Exactly one must be.

        both live  -> the saving is SQUARED (`problem`)
        none live  -> a stop the placement says carries narrow weights is
                      narrowed by nobody -- silently zero (`problem`)
        """
        owner = self.narrowing_owner(word_bits)
        mapper_live = owner == "mapper"
        evaluator_live = abs(float(applied_scale) - 1.0) > 1e-12
        row = {"measured_word_bits": None if word_bits is None else int(word_bits),
               "q": self.declared_q, "weight_bits": self.weight_bits,
               "mapper_live": mapper_live, "evaluator_live": evaluator_live,
               "applied_scale": float(applied_scale),
               "owner": ("mapper" if mapper_live and not evaluator_live else
                         "evaluator" if evaluator_live and not mapper_live else
                         "BOTH" if mapper_live and evaluator_live else "NOBODY"),
               "ok": mapper_live != evaluator_live}
        if mapper_live and evaluator_live:
            row["problem"] = (f"THE ON-CHIP NARROWING IS APPLIED TWICE and the saving is "
                              f"SQUARED: the plan already stores Word bits {word_bits} "
                              f"AND the evaluator applied x{applied_scale:.4f}")
        elif not mapper_live and not evaluator_live:
            row["problem"] = (f"NOBODY narrows this stop: the plan stores Word bits "
                              f"{word_bits} (not q = {self.declared_q}) and "
                              f"{self.mode} packing applied x1.0000, so a stop the "
                              f"placement says carries narrow weights is silently at "
                              f"full width")
        return row

    def weights_per_word(self, block_bits, reduced=True):
        """How many weights one physical word of `block_bits` carries.

        Public because a caller may need the scratchpad's WORD count rather
        than its scalar count.
        """
        return self._per_word(block_bits,
                              self.reduced_bits_per_weight if reduced
                              else self.weight_bits)

    def wire_scale(self):
        """Wire energy is per bit moved, so it always falls with the bit count."""
        return self.reduced_bits_per_weight / self.weight_bits

    def switching_scale(self, flit_bits):
        """Router/ingress energy is per FLIT, so it falls only with repacking."""
        full = self._per_word(flit_bits, self.weight_bits)
        red = self._per_word(flit_bits, self.reduced_bits_per_weight)
        return full / red if red > 0 else 1.0

    def to_dict(self):
        return {
            "mode": self.mode,
            "meaning": ("stream: the retained k bits of each codeword are packed "
                        "with no per-weight alignment, so values per physical word "
                        "and operands per flit rise by n/k and every stage scales "
                        "by k/n. aligned: each reduced weight occupies "
                        "ceil(weight_bits*k/n) whole bits and nothing is repacked, "
                        "so wire energy falls but a physical word may hold no more "
                        "values than before and its access count does not move."),
            "full_bits_per_weight": self.weight_bits,
            "reduced_bits_per_weight": self.reduced_bits_per_weight,
            "retained_fraction_k_over_n": self.frac,
            "source": ("02_reconstruction_dse_and_implementation.txt section 16; "
                       "the stream layout is eccenergy/embedded.py's"),
        }








