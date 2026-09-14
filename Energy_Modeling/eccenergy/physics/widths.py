"""PROMPT_2's WIDTH TABLE: one declared word width PER ARM, keyed by that arm's
own `datawidth`. Automatic, unconditional, and not a knob.

    python3 -m eccenergy.physics.widths     # print the RULE's table
    archs/<name>/widths.yaml                # THE TABLE a design declares

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

THE GLB WORD IS A WHOLE MULTIPLE OF THE SCRATCHPAD'S
----------------------------------------------------
4x -- Eyeriss v1's published 16 b / 64 b ratio. Divisibility survives, since
`q | W` implies `q | 4W`, so one scratchpad width settles every weight level.

WHERE THE TABLE LIVES SINCE 2026-09-14 (EnvReorganisation phase 1)
-------------------------------------------------------------------
Each design declares its own `archs/<name>/widths.yaml` -- `q -> {spad_width,
glb_width}` -- read by `arch/design.py` into a `WidthTable` from this module.
THIS MODULE KEEPS THE RULE, not the numbers: it is L1 and reads no file. The
rule is the fallback for a q a design does not list, and for a design that
declares no file at all; it reproduces every row of the study's table, which
is what `tests/test_code_widths.py` asserts. `ECC_WEIGHT_WIDTH_GLB_MULT` is
gone: the GLB multiplier is the ratio the design's own 8-bit row declares.
`arch.patch._set_weight_geometry()` applies all of this; this module only
decides the numbers.
"""
from __future__ import annotations

#: The study's payload width, and the baseline/embedded arm's datawidth.
DEFAULT_WEIGHT_BITS = 8

#: The declared scratchpad width of the BASELINE/EMBEDDED arm, and the width
#: every other arm's width is chosen nearest to. prompt_2's "declare width"
#: column for the 8-bit arm; also the depth denominator, so it is the one
#: number that fixes how much silicon each weight level is.
BASE_WIDTH = 96

#: The GLB word is this multiple of the scratchpad's when a design declares no
#: table (Eyeriss v1's published 16 b / 64 b ratio).
DEFAULT_GLB_MULT = 4


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


class WidthTable:
    """ONE DESIGN's declared word widths, per `q` -- THE WIDTH TABLE as data.

    `spad` and `glb` map `q -> width` for the scratchpad row and the GLB row;
    either may be empty, and a `q` neither lists falls back to THE RULE:
    `nearest_multiple(q, base)` for the scratchpad, `x glb_mult` for a GLB.
    `source` names the file (or "the rule") in every refusal.

    Checked at construction, because a width its own `q` does not divide ABORTS
    `timeloop-mapper` (`buffer.cpp:302`) and discovering that at load is the
    point; a GLB word that is not a whole multiple of its scratchpad's breaks
    the `q | W implies q | mW` argument and is refused too.
    """

    def __init__(self, spad=None, glb=None, glb_mult=DEFAULT_GLB_MULT, source="the rule"):
        self.spad = {int(q): int(w) for q, w in (spad or {}).items()}
        self.glb = {int(q): int(w) for q, w in (glb or {}).items()}
        self.default_glb_mult = int(glb_mult)
        self.source = source
        if self.default_glb_mult < 1:
            raise ValueError(f"{source}: a GLB word is a POSITIVE multiple of the "
                             f"scratchpad's, not {glb_mult}")
        for which, table in (("spad_width", self.spad), ("glb_width", self.glb)):
            for q, w in table.items():
                if q < 1:
                    raise ValueError(f"{source}: q={q}: a datawidth is at least 1 bit")
                if w % q != 0:
                    raise ValueError(
                        f"{source}: {which} {w} at q={q} is not a multiple of {q} "
                        f"(remainder {w % q}). timeloop-mapper asserts "
                        f"`width % (word_bits * block_size) == 0` (buffer.cpp:302) "
                        f"and ABORTS -- there is no floor path. The rule gives "
                        f"{nearest_multiple(q) * (1 if which == 'spad_width' else self.default_glb_mult)}.")
        for q in set(self.spad) & set(self.glb):
            if self.glb[q] % self.spad[q] != 0 or self.glb[q] < self.spad[q]:
                raise ValueError(
                    f"{source}: at q={q} the GLB word ({self.glb[q]}) is not a whole "
                    f"multiple of the scratchpad's ({self.spad[q]}); `q | W` must "
                    f"imply `q | glb_width`, which only a whole multiple guarantees")

    @classmethod
    def rule(cls, glb_mult=DEFAULT_GLB_MULT, source="the rule (nearest multiple of q to 96)"):
        """No declared numbers at all: every width from the rule."""
        return cls({}, {}, glb_mult, source)

    # ---- the base row: what the 8-bit arm declares -------------------------
    def base_width(self, weight_bits=DEFAULT_WEIGHT_BITS):
        """The scratchpad width the `weight_bits`-bit arm declares -- the
        depth denominator. 96 at the study's 8-bit payload."""
        wb = int(weight_bits)
        return self.spad.get(wb) or nearest_multiple(wb, BASE_WIDTH)

    def glb_mult(self, weight_bits=DEFAULT_WEIGHT_BITS):
        """GLB word / scratchpad word on the base row (4 for Eyeriss v1)."""
        wb = int(weight_bits)
        if wb in self.glb:
            return self.glb[wb] // self.base_width(wb)
        return self.default_glb_mult

    # ---- per level -----------------------------------------------------------
    def spad_width(self, q, weight_bits=DEFAULT_WEIGHT_BITS):
        q = int(q)
        if q < 1:
            raise ValueError(f"q={q}: a datawidth must be at least 1 bit")
        return self.spad.get(q) or nearest_multiple(q, self.base_width(weight_bits))

    def glb_width(self, q, weight_bits=DEFAULT_WEIGHT_BITS):
        q = int(q)
        if q in self.glb:
            return self.glb[q]
        return self.spad_width(q, weight_bits) * self.glb_mult(weight_bits)

    def level_width(self, q, is_scratchpad, weight_bits=DEFAULT_WEIGHT_BITS):
        """What ONE weight level declares for the datawidth IT stores."""
        return (self.spad_width(q, weight_bits) if is_scratchpad
                else self.glb_width(q, weight_bits))

    def renormalised_depth(self, depth, width, is_scratchpad,
                           weight_bits=DEFAULT_WEIGHT_BITS):
        """The depth EVERY arm declares for this level: the published total
        bits divided by the BASE row's width, never by the arm's own."""
        base = self.base_width(weight_bits)
        if not is_scratchpad:
            base *= self.glb_mult(weight_bits)
        return max(1, int((int(depth) * int(width) / base) + 0.5))

    def listed(self, q):
        """Is `q` declared, or would it fall back to the rule?"""
        return int(q) in self.spad

    def to_dict(self):
        return {"source": self.source,
                "widths": {q: {"spad_width": self.spad_width(q),
                               "glb_width": self.glb_width(q)}
                           for q in sorted(set(self.spad) | set(self.glb), reverse=True)},
                "glb_mult": self.glb_mult()}


# ---- the module-level spellings, kept for every caller that has no design ---
def base_width(weight_bits=DEFAULT_WEIGHT_BITS, table=None):
    """The width the 8-bit arm declares, and the depth denominator."""
    return (table or WidthTable.rule()).base_width(weight_bits)


def declared_width(q, weight_bits=DEFAULT_WEIGHT_BITS, table=None):
    """The declared SCRATCHPAD width for an arm storing `q`-bit weights: the
    design's table where it lists `q`, THE RULE otherwise."""
    return (table or WidthTable.rule()).spad_width(q, weight_bits)


def level_width(q, is_scratchpad, glb_mult=DEFAULT_GLB_MULT,
                weight_bits=DEFAULT_WEIGHT_BITS, table=None):
    """What ONE weight level declares: the scratchpad width, x `glb_mult`
    above the PE array. `q | W` implies `q | mW`, so divisibility survives."""
    return (table or WidthTable.rule(glb_mult)).level_width(q, is_scratchpad, weight_bits)


def renormalised_depth(depth, width, is_scratchpad, glb_mult=DEFAULT_GLB_MULT,
                       weight_bits=DEFAULT_WEIGHT_BITS, table=None):
    """The depth EVERY arm declares for this level (see `WidthTable`)."""
    return (table or WidthTable.rule(glb_mult)).renormalised_depth(
        depth, width, is_scratchpad, weight_bits)


def audit(weight_bits=DEFAULT_WEIGHT_BITS, glb_mult=DEFAULT_GLB_MULT, codes=None,
          table=None):
    """A printable table of every arm: q, declared widths, weights per word,
    effective capacity. With no `table` it is THE RULE's table, which is what
    a design's widths.yaml must reproduce or deliberately depart from."""
    table = table or WidthTable.rule(glb_mult)
    codes = codes or [(63, 57), (63, 51), (63, 45), (63, 39), (63, 36), (63, 30)]
    base = table.base_width(weight_bits)
    per_word_base = base // int(weight_bits)
    lines = [f"  THE WIDTH TABLE -- {table.source}; base width {base} b, "
             f"GLB {table.glb_mult(weight_bits)}x, payload {weight_bits} b",
             "",
             f"  {'arm':<20} {'q':>3} {'spad W':>7} {'GLB W':>7} "
             f"{'w/word':>7} {'eff. cap':>9}  legal",
             "  " + "-" * 68]
    rows = [("Baseline / Embedded", int(weight_bits))]
    for n, k in codes:
        rows.append((f"BCH({n},{k})", declared_datawidth(n, k, weight_bits)))
    for label, q in rows:
        w = table.spad_width(q, weight_bits)
        g = table.glb_width(q, weight_bits)
        per_word = w // q
        lines.append(
            f"  {label:<20} {q:>3} {w:>7} {g:>7} {per_word:>7} "
            f"{per_word / per_word_base:>8.4f}x  "
            + (f"{w} % {q} == 0" if w % q == 0 else f"!! {w} % {q} != 0")
            + ("" if table.listed(q) else "  (rule)"))
    lines.append("")
    lines.append("  Each arm's width suits its OWN datawidth. No arm has to be")
    lines.append("  legal for another arm's datawidth -- they are never mapped")
    lines.append("  on one another's silicon. Depth is common (base width).")
    return "\n".join(lines)


if __name__ == "__main__":       # pragma: no cover -- `python3 -m` convenience
    print(audit())
