"""THE WIDTH TABLE as a LOOKUP: one declared word width per BCH code.

    ECC_WEIGHT_WIDTH=auto      # env.sh section 5 -- resolve it from (N, K)

WHY THIS MODULE EXISTS
----------------------
`timeloop-mapper` asserts `width % (word_bits * block_size) == 0`
(`buffer.cpp:302`) with `block_size` defaulting to 1, and there is NO floor
path: a width the datawidth does not divide ABORTS the mapper (`exit=134, core
dumped`), it does not fall back to `floor(width/q)`. So every BCH code whose
`q = round(8*K/N)` does not divide the published word needs a declared width
that it does, and until now that width had to be typed in by hand per run --
which is why only BCH(63,30) (q=4, divides every published width in the study)
was ever runnable without thinking about it.

This module makes the choice once, keyed by the code, so **every** BCH
configuration runs. `archs._set_weight_width()` does the rewriting; this only
decides the number.

THE TWO CONSTRAINTS, AND WHY prompt_2's TABLE FAILS ONE OF THEM
---------------------------------------------------------------
Both arms of a pair share ONE declared width (`archs.assert_pair_geometry`),
and the baseline/embedded arm stores `ECC_WEIGHT_BITS`-bit weights. So a legal
width must divide **both**:

    W % q == 0              or the RECONSTRUCTION arm aborts
    W % ECC_WEIGHT_BITS == 0    or the BASELINE/EMBEDDED arm aborts

prompt_2.md's WIDTH TABLE was written against the first constraint alone and
two of its five rows violate the second:

    BCH(63,57) q=7  W=98  ->  98 % 8 = 2   embedded arm ABORTS
    BCH(63,45) q=6  W=96  ->  96 % 8 = 0   legal
    BCH(63,39) q=5  W=95  ->  95 % 8 = 7   embedded arm ABORTS
    BCH(63,30) q=4  W=96  ->  96 % 8 = 0   legal

so 98 and 95 are NOT usable and are replaced here. `config.py` has refused them
at validate time since 2026-09-10; this module is what stops a run needing them
in the first place.

WHY THE MINIMUM LEGAL WIDTH, NOT prompt_2's ~96-BIT FAMILY
-----------------------------------------------------------
prompt_2 chose all five widths within 3 % of each other (96 / 98 / 96 / 95 /
96) so per-access energy stayed comparable ACROSS codes. Under the legality
rule above the nearest legal members of that family are 112 (q=7) and 80
(q=5) -- ±17 %, not ±3 %, so the family's whole point is already gone.

What replaces it is `lcm(q, ECC_WEIGHT_BITS)`, the SMALLEST legal width, and
the reason is `_set_weight_width`'s invariant: each level's depth is
renormalised as `round(depth * width / W)` to hold the published TOTAL BITS,
because CACTI is handed depth and width and that is the quantity that must not
move. Integer depth makes that renormalisation lossy, and the loss shrinks as
the declared width shrinks. On the CURRENT `eyeriss_like_wglb` scratchpad
(16 x 16 b = 256 bits) the difference is not academic:

    q=7   W=56  -> depth 5  (280 b, +9.4 %)  |  W=112 -> depth 2 (224 b, -12.5 %)
    q=5   W=40  -> depth 6  (240 b, -6.2 %)  |  W=80  -> depth 3 (240 b,  -6.2 %)
    q=6   W=24  -> depth 11 (264 b, +3.1 %)  |  W=96  -> depth 3 (288 b, +12.5 %)

A width that leaves a 2- or 3-entry scratchpad is also barely a reuse level at
all, and `_weight_level_parts` drops a `depth: 1` level entirely as a pipeline
latch -- so the large-width family can silently stop narrowing the thing the
study is about. prompt_2's table was calibrated on the 224 x 16 b scratchpad of
the day (3,584 bits), where depth granularity was fine; it is not fine at 256.

`WIDTH_TABLE` is still an explicit, reviewable dict rather than a formula, so a
code can be given a hand-picked width without touching the rule -- but every
entry is checked against the rule by `audit()` and by
`tests/test_code_widths.py`, so a typo cannot pass as a design decision.

None MEANS "KEEP THE PUBLISHED SILICON"
----------------------------------------
A `None` entry is not "no data", it is a decision: when `q` divides
`ECC_WEIGHT_BITS` (q in 1, 2, 4, 8) it divides every width in the study, since
every weight level in `archs/` declares a multiple of 8 bits (8, 16, 24, 64,
512 -- verified 2026-09-11). Those codes run on the UNTOUCHED published
geometry, which is what prompt_2, prompt_3 and env.sh all require of
BCH(63,30) and what keeps its cache bit-identical to the prompt_5 run.

THE GLB TAKES `ECC_WEIGHT_WIDTH_GLB_MULT` TIMES THIS
-----------------------------------------------------
The table is quoted for the SCRATCHPAD -- the innermost weight level. A weight
level above the PE array declares `glb_mult` (4, Eyeriss v1's published 16 b /
64 b ratio) times it, and divisibility survives: q | W implies q | 4W. So one
scratchpad width settles every weight level at once. `archs._set_weight_width`
owns that half; `glb_width()` below is the same arithmetic for a caller that
needs to print it.
"""
from __future__ import annotations

import math

#: The study's payload width. Only a default -- callers pass `cfg.weight_bits`.
DEFAULT_WEIGHT_BITS = 8

#: THE WIDTH TABLE. (N, K) -> declared SCRATCHPAD width in bits, or None to
#: keep the architecture's published widths.
#:
#: Every value is `lcm(q, 8)` and every None is a code with `q` in {1,2,4,8};
#: `audit()` re-derives both and `tests/test_code_widths.py` asserts them, so
#: this dict is a record of the decision, not a second source of truth that can
#: drift from it. The six keys are `config.BCH63_KTOD`'s codes -- the ones with
#: a published minimum distance in this study. Any other (N, K) falls back to
#: the same rule via `declared_width()`.
WIDTH_TABLE = {
    #  code          q     W     why
    (63, 57):  56,  # 7    56    published 16/24/64 all fail 7
    (63, 51):  24,  # 6    24    published 16/64 fail 6 (24 already legal)
    (63, 45):  24,  # 6    24    same q as K=51, same width
    (63, 39):  40,  # 5    40    published 16/24/64 all fail 5
    (63, 36):  40,  # 5    40    same q as K=39, same width
    (63, 30): None, # 4    --    q | 8, published geometry untouched
}


def declared_datawidth(n, k, weight_bits=DEFAULT_WEIGHT_BITS):
    """`q`: the reduced on-chip weight width the reconstruction arm stores.

    `round(weight_bits * K/N)`, which is prompt_2's "q declared" column --
    NOT `ceil`. `hpc/map_depth_sweep.sh` used to derive it with `ceil` and that
    disagreed with the table at exactly one code, BCH(63,57): ceil gives 8, so
    the reconstruction arm would have declared the SAME width as the embedded
    arm and the on-chip treatment would have been a silent no-op.

    Benchmark a saving against the true rate `8*K/N`, not against this integer:
    rounding DOWN (7 against 7.238 at BCH(63,57)) flatters the reconstruction
    arm, rounding UP understates it. prompt_2's residual column tracks which.
    """
    n, k = int(n), int(k)
    weight_bits = int(weight_bits)
    if weight_bits < 1:
        raise ValueError(f"weight_bits={weight_bits}: must be positive")
    if not 0 < k < n:
        raise ValueError(
            f"BCH({n},{k}): a code needs 0 < K < N. K == N is no code at all "
            f"and K > N is not a rate.")
    # floor(x + 0.5), NOT python's round(): round() is banker's rounding, so an
    # exact .5 goes to the EVEN neighbour and two adjacent codes could round in
    # opposite directions. No BCH(63,K) lands on .5, but the rule has to be
    # stated or the next N makes it a bug.
    q = int(math.floor(weight_bits * k / n + 0.5))
    return max(1, min(weight_bits, q))


def minimum_legal_width(q, weight_bits=DEFAULT_WEIGHT_BITS):
    """The smallest word width BOTH arms of a pair can declare: `lcm(q, bits)`.

    The reconstruction arm needs `W % q == 0`, the baseline/embedded arm needs
    `W % weight_bits == 0`, and they share one declared width -- so the legal
    widths are exactly the multiples of the least common multiple.
    """
    q, weight_bits = int(q), int(weight_bits)
    if q < 1 or weight_bits < 1:
        raise ValueError(f"q={q}, weight_bits={weight_bits}: both must be positive")
    return q * weight_bits // math.gcd(q, weight_bits)


def is_legal_width(width, q, weight_bits=DEFAULT_WEIGHT_BITS):
    """Does `width` suit BOTH arms? See `minimum_legal_width`."""
    return width % int(q) == 0 and width % int(weight_bits) == 0


def needs_width_change(q, weight_bits=DEFAULT_WEIGHT_BITS):
    """Does this code need a declared width at all?

    False when `q` divides `weight_bits` -- then q divides every width in
    `archs/`, all of which are multiples of 8, and the published silicon runs
    as published. That is the BCH(63,30) case the whole study is calibrated on.
    """
    return int(weight_bits) % int(q) != 0


def declared_width(n, k, weight_bits=DEFAULT_WEIGHT_BITS):
    """The declared SCRATCHPAD width for BCH(n, k), or None to keep published.

    `WIDTH_TABLE` first, so a hand-picked width wins; otherwise the rule. A
    table entry that is not legal for its own code raises rather than being
    quietly rounded up -- prompt_2's 98 and 95 are exactly that failure, and
    discovering it at validate time is the point.
    """
    q = declared_datawidth(n, k, weight_bits)
    key = (int(n), int(k))
    if key in WIDTH_TABLE:
        w = WIDTH_TABLE[key]
        if w is None:
            if needs_width_change(q, weight_bits):
                raise ValueError(
                    f"WIDTH_TABLE[{key}] is None, but BCH({n},{k}) declares "
                    f"q={q}, which does not divide weight_bits={weight_bits}. "
                    f"None means 'the published widths already admit q'; this "
                    f"code needs a declared width of "
                    f"{minimum_legal_width(q, weight_bits)} or a multiple.")
            return None
        if not is_legal_width(w, q, weight_bits):
            raise ValueError(
                f"WIDTH_TABLE[{key}] = {w} is not legal for BCH({n},{k}): "
                f"q={q} leaves {w % q}, weight_bits={weight_bits} leaves "
                f"{w % weight_bits}. Both arms share ONE width and "
                f"timeloop-mapper aborts on `width % datawidth != 0` "
                f"(buffer.cpp:302). Legal widths are multiples of "
                f"{minimum_legal_width(q, weight_bits)}.")
        return int(w)
    if not needs_width_change(q, weight_bits):
        return None
    return minimum_legal_width(q, weight_bits)


def glb_width(n, k, glb_mult=4, weight_bits=DEFAULT_WEIGHT_BITS):
    """What a weight level ABOVE the PE array declares, or None.

    `archs._set_weight_width` applies this itself; this is for printing.
    """
    w = declared_width(n, k, weight_bits)
    return None if w is None else w * int(glb_mult)


def renormalised_depth(depth, width, declared):
    """`_set_weight_width`'s own arithmetic, exposed so a caller can report the
    bit error BEFORE queueing a wave rather than reading it out of a log."""
    return max(1, int(round(int(depth) * int(width) / int(declared))))


def audit(codes=None, weight_bits=DEFAULT_WEIGHT_BITS, glb_mult=4):
    """A printable table of every code: q, declared width, GLB width, legality.

    Re-derives the rule instead of trusting `WIDTH_TABLE`, so running this is
    what makes the dict reviewable.
    """
    codes = codes or sorted(WIDTH_TABLE, key=lambda kv: (-kv[0], -kv[1]))
    lines = [f"  {'code':<12} {'K/N':>7} {'8K/N':>7} {'q':>3} "
             f"{'spad W':>7} {'GLB W':>7} {'min legal':>10}  note"]
    lines.append("  " + "-" * 74)
    for n, k in codes:
        q = declared_datawidth(n, k, weight_bits)
        w = declared_width(n, k, weight_bits)
        g = glb_width(n, k, glb_mult, weight_bits)
        lines.append(
            f"  BCH({n},{k})".ljust(14)
            + f"{k / n:>7.4f} {weight_bits * k / n:>7.3f} {q:>3} "
            + f"{(str(w) if w else 'published'):>7} "
            + f"{(str(g) if g else 'published'):>7} "
            + f"{minimum_legal_width(q, weight_bits):>10}  "
            + ("published geometry admits q" if w is None
               else f"q|{w} and {weight_bits}|{w}"))
    return "\n".join(lines)


if __name__ == "__main__":       # pragma: no cover -- `python3 -m` convenience
    print(audit())
