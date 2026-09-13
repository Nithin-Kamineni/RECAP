"""PROMPT_2's WIDTH TABLE: one declared word width PER ARM, keyed by that arm's
own `datawidth`. Automatic, unconditional, and not a knob.

    python3 -m eccenergy.code_widths        # print the table

THE ONE THING TO GET RIGHT, BECAUSE IT HAS BEEN GOT WRONG REPEATEDLY
--------------------------------------------------------------------
**THE ARMS DO NOT SHARE A DECLARED WIDTH.** Each arm declares the width that
suits ITS OWN datawidth, and no arm has to be legal for another arm's
datawidth. The baseline/embedded arm stores 8-bit weights and runs at width 96
(96 % 8 == 0). The BCH(63,39) reconstruction arm stores 5-bit weights and runs
at width 95 (95 % 5 == 0). **95 never has to divide 8**, because the
baseline/embedded arm is never mapped at width 95 -- it is mapped at 96.

    arm                  q    spad W   GLB W (4x)   weights/word   eff. capacity
    Baseline / Embedded  8      96        384            12           1.0000x
    BCH(63,57)           7      98        392            14           1.1667x
    BCH(63,45)           6      96        384            16           1.3333x
    BCH(63,39)           5      95        380            19           1.5833x
    BCH(63,30)           4      96        384            24           2.0000x

That is prompt_2.md's WIDTH TABLE verbatim, and its `eff. capacity` column is
`(W_arm/q) / (96/8)` -- which only reconciles if the arms declare DIFFERENT
widths at a COMMON depth. The table is self-checking in that sense.

A PREVIOUS IMPLEMENTATION (2026-09-11 to 2026-09-12) read prompt_2's rule as
"both arms share ONE width", concluded that 98 and 95 were illegal because they
do not divide 8, and substituted `lcm(q, 8)` -- 56 / 24 / 40. That is WITHDRAWN
and must not be reintroduced. It was wrong twice over:

  * the premise is false (see above), and
  * it made the 8-BIT REFERENCE ARM MOVE BETWEEN CODES (width 56, 24, 40), so
    the reference held 35 / 33 / 30 weights per PE at q = 7 / 6 / 5 and fell
    off a tiling cliff at q=5, which is where BCH(63,39)'s spurious 37.69 %
    came from (FINDINGS 2.4b). Under this table the 8-bit arm is width 96 at
    EVERY code: ONE arm, mapped ONCE, and that artifact cannot occur.

`archs.assert_pair_geometry()` does NOT check `width` or `datawidth` for this
reason. It checks the LEVEL SET and the DEPTH, which the arms really do share.

THE RULE, AND WHY IT REPRODUCES THE TABLE
-----------------------------------------
`declared_width(q)` is **the multiple of `q` nearest to `BASE_WIDTH` (96)**:

    q=8 -> 12x8 = 96     q=7 -> 14x7 = 98     q=6 -> 16x6 = 96
    q=5 -> 19x5 = 95     q=4 -> 24x4 = 96     q<=3 -> 96

which is prompt_2's table exactly, and is why all five widths sit within 3 % of
each other: per-access energy stays comparable ACROSS codes, and the three
width-96 arms are byte-identical silicon (1.48668 / 2.37439 pJ, measured).

The only constraint the mapper imposes is `width % datawidth == 0`
(`buffer.cpp:302`, `block_size` defaults to 1 and there is NO floor path -- a
partially-filled word ABORTS with `exit=134`). A multiple of `q` satisfies it
by construction, for every `q`, so no configuration can reach that abort.

DEPTH IS COMMON TO EVERY ARM, AND HOLDS THE PUBLISHED BITS
-----------------------------------------------------------
`depth' = round(depth * width / BASE_WIDTH)` -- computed at the BASE width, so
it is the SAME for every arm. Two consequences, both wanted:

  * the level is the same silicon at a different word shape rather than a
    bigger array smuggled in as a width change, and CACTI is handed `depth`
    and `width`, so that is exactly the quantity that must not move;
  * the arms differ ONLY in `width` (by <= 2 %) and `datawidth`, so
    `capacity_ratio` is `(W_arm/q)/(96/8)` and reproduces prompt_2's
    `eff. capacity` column to the digit.

It reproduces prompt_2's own depths: `weights_spad` 224 x 16 b = 3,584 b ->
depth 37 at width 96; `filter_glb` 1024 x 64 b = 65,536 b -> depth 171 at
width 384.

THE GLB TAKES `ECC_WEIGHT_WIDTH_GLB_MULT` TIMES THE SCRATCHPAD WIDTH
---------------------------------------------------------------------
4x -- Eyeriss v1's published 16 b / 64 b ratio. Divisibility survives, since
`q | W` implies `q | 4W`, so one scratchpad width settles every weight level.
`archs._set_weight_geometry()` applies all of this; this module only decides
the numbers.
"""
from __future__ import annotations

#: The study's payload width, and the baseline/embedded arm's datawidth.
DEFAULT_WEIGHT_BITS = 8

#: The declared scratchpad width of the BASELINE/EMBEDDED arm, and the width
#: every other arm's width is chosen nearest to. prompt_2's "declare width"
#: column for the 8-bit arm; also the depth denominator, so it is the one
#: number that fixes how much silicon each weight level is.
BASE_WIDTH = 96

#: THE WIDTH TABLE, keyed by the arm's own `q` (NOT by the code). prompt_2.md
#: tabulates it per code, but the width depends on the code only through
#: `q = round(8*K/N)`, so two codes with one `q` are one arm -- BCH(63,45) and
#: BCH(63,51) both declare 96, BCH(63,39) and BCH(63,36) both declare 95.
#: Every value is re-derived by `audit()` and asserted by
#: `tests/test_code_widths.py`, so this dict records the decision rather than
#: being a second source of truth that can drift from the rule.
WIDTH_TABLE = {
    # q   width   weights/word   prompt_2 row
    8:  96,     # 12            Baseline / Embedded
    7:  98,     # 14            BCH(63,57)
    6:  96,     # 16            BCH(63,45), BCH(63,51)
    5:  95,     # 19            BCH(63,39), BCH(63,36)
    4:  96,     # 24            BCH(63,30)
    3:  96,     # 32            no published code in this study
    2:  96,     # 48
    1:  96,     # 96
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
    q = int((weight_bits * k / n) + 0.5)
    return max(1, min(weight_bits, q))


def nearest_multiple(q, base=BASE_WIDTH):
    """The multiple of `q` nearest to `base`; ties go UP.

    This is the whole width rule. `round()` is avoided for the same
    banker's-rounding reason as above.
    """
    q = int(q)
    if q < 1:
        raise ValueError(f"q={q}: a datawidth must be at least 1 bit")
    n = int((int(base) / q) + 0.5)
    return max(q, n * q)


def declared_width(q, weight_bits=DEFAULT_WEIGHT_BITS):
    """The declared SCRATCHPAD width for an arm storing `q`-bit weights.

    `WIDTH_TABLE` first, so a hand-picked width wins; otherwise the rule. A
    table entry that `q` does not divide raises rather than being quietly
    rounded -- a width the datawidth does not divide ABORTS `timeloop-mapper`
    (`buffer.cpp:302`), and discovering that at import time is the point.

    `weight_bits` is accepted so a study at a payload other than 8 bits gets a
    consistent base, and is deliberately NOT used to constrain the answer: an
    arm's width must suit the arm's OWN datawidth and nothing else.
    """
    q = int(q)
    if q < 1:
        raise ValueError(f"q={q}: a datawidth must be at least 1 bit")
    w = WIDTH_TABLE.get(q)
    if w is None:
        return nearest_multiple(q, base_width(weight_bits))
    if int(w) % q != 0:
        raise ValueError(
            f"WIDTH_TABLE[{q}] = {w} is not a multiple of {q} (remainder "
            f"{int(w) % q}). timeloop-mapper asserts "
            f"`width % (word_bits * block_size) == 0` (buffer.cpp:302) and "
            f"ABORTS -- there is no floor path. The rule gives "
            f"{nearest_multiple(q, base_width(weight_bits))}.")
    return int(w)


def base_width(weight_bits=DEFAULT_WEIGHT_BITS):
    """The width the 8-bit arm declares, and the depth denominator.

    At the study's 8-bit payload this is `BASE_WIDTH` (96) exactly. At any
    other payload it is the multiple of that payload nearest 96, so the base
    arm is always legal for itself.
    """
    return nearest_multiple(int(weight_bits), BASE_WIDTH)


def level_width(q, is_scratchpad, glb_mult=4, weight_bits=DEFAULT_WEIGHT_BITS):
    """What ONE weight level declares: the table width, x `glb_mult` above the
    PE array. `q | W` implies `q | 4W`, so the divisibility survives."""
    w = declared_width(q, weight_bits)
    return w if is_scratchpad else w * int(glb_mult)


def renormalised_depth(depth, width, is_scratchpad, glb_mult=4,
                       weight_bits=DEFAULT_WEIGHT_BITS):
    """The depth EVERY arm declares for this level: the published total bits
    divided by the BASE width, not by the arm's own width.

    Computed at the base so the arms share a depth and differ only in `width`
    (by <= 2 %) and `datawidth` -- which is what makes `capacity_ratio`
    reproduce prompt_2's `eff. capacity` column, and what leaves
    `assert_pair_geometry`'s depth check meaningful.
    """
    base = base_width(weight_bits)
    if not is_scratchpad:
        base *= int(glb_mult)
    return max(1, int((int(depth) * int(width) / base) + 0.5))


def audit(weight_bits=DEFAULT_WEIGHT_BITS, glb_mult=4, codes=None):
    """A printable table of every arm: q, declared widths, weights per word,
    effective capacity. Re-derives the rule instead of trusting `WIDTH_TABLE`,
    so running this is what makes the dict reviewable."""
    codes = codes or [(63, 57), (63, 51), (63, 45), (63, 39), (63, 36), (63, 30)]
    base = base_width(weight_bits)
    per_word_base = base // int(weight_bits)
    lines = [f"  PROMPT_2 WIDTH TABLE -- base width {base} b, "
             f"GLB {glb_mult}x, payload {weight_bits} b",
             "",
             f"  {'arm':<20} {'q':>3} {'spad W':>7} {'GLB W':>7} "
             f"{'w/word':>7} {'eff. cap':>9}  legal",
             "  " + "-" * 68]
    rows = [("Baseline / Embedded", int(weight_bits))]
    seen = {int(weight_bits)}
    for n, k in codes:
        q = declared_datawidth(n, k, weight_bits)
        rows.append((f"BCH({n},{k})", q))
        seen.add(q)
    for label, q in rows:
        w = declared_width(q, weight_bits)
        g = level_width(q, False, glb_mult, weight_bits)
        per_word = w // q
        lines.append(
            f"  {label:<20} {q:>3} {w:>7} {g:>7} {per_word:>7} "
            f"{per_word / per_word_base:>8.4f}x  "
            + (f"{w} % {q} == 0" if w % q == 0 else f"!! {w} % {q} != 0"))
    lines.append("")
    lines.append("  Each arm's width suits its OWN datawidth. No arm has to be")
    lines.append("  legal for another arm's datawidth -- they are never mapped")
    lines.append("  on one another's silicon. Depth is common (base width).")
    return "\n".join(lines)


if __name__ == "__main__":       # pragma: no cover -- `python3 -m` convenience
    print(audit())
