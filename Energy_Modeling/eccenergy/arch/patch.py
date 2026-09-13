"""The arch.yaml the mapper actually runs -- every patch, in one place.

`patched_weight_geometry()` and `_patched_text()` produce the YAML Timeloop is
handed: the DRAM-depth patch the historical scripts applied, the technology
node, the per-arm weight geometry out of THE WIDTH TABLE, the constrained
mapspace (prompt_3), the relaxed weight factors, and -- since prompt_7 Phase
C1 -- TIME: `shared_bandwidth` on DRAM, the per-dataspace bandwidth scale, a
bit-aware port on a narrowed level and the bank count that now reaches CACTI.
Everything here is in the fingerprint, so every one of them colds the cache.

A COMMENT IS NEVER A DECLARATION. Read geometry through `uncommented()`, write
it through `write_attr()`. Every geometry regex here used to match the raw text,
so a `depth:` written in a COMMENT was read as the attribute -- and
`re.sub(..., count=1)` then rewrote the COMMENT and left the attribute alone.
One explanatory comment made `filter_glb` declare 98,304 bits where 16,512 were
intended: a six-fold capacity error, with one line of stdout to say so.
`uncommented()` masks comments while PRESERVING POSITIONS, so a caller searches
the masked copy and splices into the real text; `write_attr()` RAISES when the
rewrite lands on nothing, which is the silent half. ADDING A REGEX OVER AN ARCH
YAML? MASK FIRST.

`_set_weight_geometry()` reshapes every weight level per arm from
`physics/widths.py`'s table and renormalises depth to hold the declared total
bits, so `width % datawidth == 0` holds by construction at every code. If a
`width 64 % datawidth 5 != 0` ever reaches you, the fix is in `WIDTH_TABLE`,
never in the arch YAML and never in the config. `assert_pair_geometry()` checks
that both arms declare the same LEVELS at the same DEPTH -- never the same
width, which is per arm.

ProjectRestructure phase 3 cut this out of `archs.py`.
"""
from __future__ import annotations

import re

from ..physics import widths

from .load import BANKED_SRAM_CLASS, PLAIN_SRAM_CLASS, _inject_noc, _node_blocks, arch_source


# --------------------------------------------------------------------- patching
def _patch_dram_depth(text, depth):
    """Make the DRAM level declare exactly `depth`.

    The stock example designs omit it and Timeloop needs it, so it is inserted.
    A design that DOES declare it -- every arch_paper.yaml does, and so did the
    locally authored v2 -- gets it rewritten instead of skipped: an earlier
    version returned the text unchanged in that case, which silently turned
    ECC_DRAM_DEPTH into a no-op on exactly those designs while still giving the
    run its own mapper cache, i.e. a fresh and expensive re-map of an
    unchanged architecture.
    """
    if "name: DRAM" not in text:
        return text
    head, tail = text.split("name: DRAM", 1)
    block, rest = tail.split("!", 1) if "!" in tail else (tail, "")
    if re.search(r"\bdepth:\s*\d+", block):
        block = re.sub(r"\bdepth:\s*\d+", f"depth: {depth}", block, count=1)
        return head + "name: DRAM" + block + ("!" + rest if rest else "")
    return text.replace(
        'class: DRAM\n    attributes:\n      type: "LPDDR4"',
        f'class: DRAM\n    attributes:\n      depth: {depth}\n      type: "LPDDR4"')


def _force_technology(text, node):
    """Rewrite every `technology:` declaration to the same node.

    Accelergy costs components at the node declared on their enclosing
    container. Designs that ship at different nodes cannot be compared as
    architectures until this is equalised.
    """
    return re.sub(r'technology:\s*"?[\w.]+"?', f'technology: "{node}"', text)


def _force_datawidth(text, bits, arch="?"):
    """Rewrite `datawidth:` to `bits` on the tensor-carrying levels only.

    Why: the stock simple_weight_stationary / simple_output_stationary designs
    declare datawidth 16 everywhere, so an "8-bit weights" study silently pays
    128 pJ per DRAM weight read on those two architectures while the eyeriss and
    simba designs pay 64. Forcing the datawidth removes that confound. `width:`
    is left alone -- it is the physical word width, and Timeloop derives
    entries-per-word as width/datawidth, so the same SRAM simply holds twice as
    many 8-bit values.

    Two levels are deliberately NOT rewritten:

    * A DEDICATED PARTIAL-SUM level -- one whose `keep:` list is Outputs and
      nothing else. Accumulator precision is a separate design choice from the
      operand quantization (Eyeriss v1 is an 8-bit design with 16-bit psums;
      v2 uses 20-bit psums), so forcing it would be wrong. It is also invalid:
      Timeloop asserts `width % datawidth == 0`, and v2's psum spad is
      `width: 20`, which 8 does not divide -- that assertion is what aborted
      the mapper before this exclusion existed.

    * Any level where `width` is not a multiple of `bits`. Reported, not
      silently skipped.

    A level shared between Outputs and the operands (a `shared_glb` with no
    dataspace constraint) IS rewritten, because Timeloop allows one datawidth
    per level and the operands are what this study measures. That matches how
    the eyeriss designs already declare their shared buffer.
    """
    out, skipped = [], []
    # split into "- !Node" blocks, keeping the delimiters so the text rebuilds
    parts = re.split(r"(\n\s*-\s*!)", text)
    for part in parts:
        m = re.search(r"\bdatawidth:\s*(\d+)", uncommented(part))
        if not m:
            out.append(part)
            continue
        name = re.search(r"name:\s*(\S+)", part)
        name = name.group(1) if name else "?"
        keep = re.search(r"keep:\s*\[([^\]]*)\]", part)
        keep = [s.strip() for s in keep.group(1).split(",")] if keep else []
        width = re.search(r"\bwidth:\s*(\d+)", uncommented(part))
        width = int(width.group(1)) if width else None

        if keep == ["Outputs"]:
            skipped.append(f"{name} (dedicated partial-sum level, "
                           f"datawidth {m.group(1)} kept)")
            out.append(part)
            continue
        if width is not None and width % bits != 0:
            skipped.append(f"{name} (width {width} is not a multiple of {bits}, "
                           f"datawidth {m.group(1)} kept)")
            out.append(part)
            continue
        out.append(re.sub(r"\bdatawidth:\s*\d+", f"datawidth: {bits}", part))

    if skipped and arch:
        print(f"  [datawidth] {arch}: forced to {bits}b except " + "; ".join(skipped))
    return "".join(out)


def _force_acc_bits(text, bits, arch="?", quiet=False):
    """SENSITIVITY STUDY ONLY: force one accumulator width on every design.

    The primary comparison keeps each design's published psum width -- that is
    an architectural property, and equalising it equalises the architectures.
    This exists so the sensitivity to that choice can be MEASURED rather than
    argued about, and everything it touches is namespaced separately.

    What it rewrites:

    * `mac.adder_width`, the accumulator arithmetic.
    * `datawidth` on every DEDICATED partial-sum level -- one whose `keep:` list
      is `Outputs` and nothing else. A level shared with the operands is left
      alone: Timeloop allows one datawidth per level and the operands are the
      standardized quantity.
    * `width` on those same levels, to the smallest multiple of `bits` that is
      at least the declared width. Timeloop asserts `width % datawidth == 0`,
      so a 16b word cannot hold 20b psums; widening the word is the only legal
      way to declare the same NUMBER of partial sums at a greater precision.
      Entry count is preserved, physical bits are not, and that is the point of
      the study -- it is asking what a wider accumulator costs.

    A level carrying Outputs that declares `# psum-width-ok:` is skipped: it
    holds requantized activations, not partial sums, and forcing an accumulator
    width onto it would be modelling a different design.
    """
    out, touched, skipped = [], [], []
    parts = re.split(r"(\n\s*-\s*!)", text)
    for part in parts:
        if re.search(r"\badder_width:\s*\d+", part):
            part = re.sub(r"\badder_width:\s*\d+", f"adder_width: {bits}", part)
            touched.append("mac.adder_width")
            out.append(part)
            continue
        if not re.search(r"\bdatawidth:\s*(\d+)", uncommented(part)):
            out.append(part)
            continue
        name = re.search(r"name:\s*(\S+)", part)
        name = name.group(1) if name else "?"
        keep = re.search(r"keep:\s*\[([^\]]*)\]", part)
        keep = [s.strip() for s in keep.group(1).split(",")] if keep else []
        if keep != ["Outputs"]:
            out.append(part)
            continue
        if re.search(r"#\s*psum-width-ok:", part):
            skipped.append(f"{name} (declared requantized, not a psum level)")
            out.append(part)
            continue
        width = re.search(r"\bwidth:\s*(\d+)", uncommented(part))
        if width:
            old_w = int(width.group(1))
            new_w = max(bits, -(-old_w // bits) * bits)
            part = re.sub(r"\bwidth:\s*\d+", f"width: {new_w}", part, count=1)
        part = re.sub(r"\bdatawidth:\s*\d+", f"datawidth: {bits}", part)
        touched.append(name)
        out.append(part)

    if not quiet and arch:
        note = f"  [acc-width] {arch}: forced to {bits}b on " + ", ".join(touched)
        if skipped:
            note += "; skipped " + "; ".join(skipped)
        print(note + "   (SENSITIVITY STUDY -- not the primary result)")
    return "".join(out)


#: Which storage levels `_scale_weight_capacity()` is allowed to touch.
#: `exclusive` only rewrites a level whose `keep:` list is Weights and nothing
#: else, so the extra capacity can only be spent on weights -- which is what
#: the reduced representation actually buys. `shared` also rewrites a level
#: that holds Weights ALONGSIDE another dataspace (`simple_output_stationary`'s
#: and `simple_input_stationary`'s `operand_glb` keep Inputs and Weights):
#: Timeloop has one capacity per level, so dilating it hands the mapper extra
#: INPUT capacity for free, which reconstruction does not pay for. The two are
#: an upper and a lower bound on one design and are quoted as a pair, the same
#: rule CLAUDE.md sets for the eyeriss `_wglb` variants.
#: `simple_weight_stationary` LEFT THAT SET on 2026-09-13: its operand half is
#: now `input_glb` + a Weights-only `weight_glb`, so both scopes give it the
#: same answer and its bracket has collapsed to a point.
WEIGHT_CAPACITY_SCOPES = ("exclusive", "shared")


def _scale_weight_capacity(text, scale, scope="exclusive", arch="?", quiet=False):
    """TASK 4: make a weight buffer hold `scale` x as many WEIGHTS.

    WHY THIS IS A MAPPER KNOB AND NOT AN EVALUATOR ONE. Under the
    reconstruction arm the on-chip weight representation is K/N of full width,
    so the same physical SRAM holds N/K = 1.615x more weights at BCH(63,39).
    Whether that buys anything is a question about the MAPPING -- a larger
    weight tile means fewer DRAM refetches of it -- and Task 3 structurally
    cannot answer it, because `RECON_OPTIMIZER=False` pins one mapping on every
    arm and both arms then refetch identically by construction. So the capacity
    has to be in the architecture the mapper sees, and this rewrites `depth:`
    on the weight-carrying levels to put it there.

    WHAT IT DOES NOT MODEL, AND WHY THE ENERGY MUST BE CORRECTED AFTERWARDS.
    Accelergy prices a level from its declared geometry, so a level at
    `depth x N/K` is costed as a physically LARGER array: more bits, more area,
    more energy per access. The reconstruction arm's array is not larger -- it
    is the same array holding narrower values -- so a Task 4 energy number read
    straight off a dilated mapping is charged for silicon the design does not
    have. `recon.capacity_dilation_correction()` re-prices those levels at the
    DECLARED geometry's per-access energy, and the dilated run records the
    ratio it corrected by. The mapping itself is unaffected either way except
    through the objective, which is why the ERT delta is reported: a level that
    got materially dearer per access biases the search AGAINST using the
    capacity, i.e. against the hypothesis, and that has to be visible rather
    than assumed away.

    THREE LEVELS ARE DELIBERATELY NOT REWRITTEN.

    * DRAM. It is not on-chip capacity and its depth is set by
      `ECC_DRAM_DEPTH`.
    * A level that holds no Weights. Dilating an activation or partial-sum
      buffer is a different architecture, not this treatment.
    * A DECLARED `depth: 1` register. `simple_weight_stationary`'s
      `weight_reg` is a pipeline latch, and FINDINGS 7.5 establishes that the
      mapping fills it once per read; scaling it to depth 2 would invent a
      reuse level the design does not have and would change what R4b's
      register is an addition TO. Reported, not silently skipped.

    A level holding Weights together with another dataspace is governed by
    `scope` -- see WEIGHT_CAPACITY_SCOPES.
    """
    if scale == 1.0:
        return text
    touched, skipped = [], []
    parts = re.split(r"(\n\s*-\s*!)", text)
    out = []
    for part in parts:
        bare = uncommented(part)           # see `uncommented()`: a `depth:` in
        depth = re.search(r"\bdepth:\s*(\d+)", bare)   # a COMMENT is prose
        if not depth:
            out.append(part)
            continue
        name = re.search(r"name:\s*(\S+)", bare)
        name = name.group(1) if name else "?"
        keep = re.search(r"keep:\s*\[([^\]]*)\]", bare)
        keep = [s.strip() for s in keep.group(1).split(",") if s.strip()] if keep else []
        d = int(depth.group(1))

        if "class: DRAM" in bare or name == "DRAM":
            out.append(part)               # never a candidate; not on-chip
            continue
        if "Weights" not in keep:
            out.append(part)
            continue
        if keep != ["Weights"] and scope != "shared":
            skipped.append(f"{name} (holds {'+'.join(keep)}; dilating it would "
                           f"also hand the mapper free {'/'.join(k for k in keep if k != 'Weights')} "
                           f"capacity -- ECC_WEIGHT_CAPACITY_SCOPE=shared includes it)")
            out.append(part)
            continue
        if d == 1:
            skipped.append(f"{name} (declared depth 1 -- a pipeline latch; "
                           f"scaling it would invent a reuse level the design "
                           f"does not have)")
            out.append(part)
            continue

        nd = max(1, int(round(d * scale)))
        if nd == d:
            skipped.append(f"{name} (depth {d} x {scale:g} rounds back to {d})")
            out.append(part)
            continue
        part = write_attr(part, "depth", nd)
        touched.append(f"{name} {d}->{nd}")
        out.append(part)

    if not quiet and arch:
        note = f"  [weight-capacity] {arch}: x{scale:g} on " + (
            ", ".join(touched) if touched else "NOTHING")
        if skipped:
            note += "; skipped " + "; ".join(skipped)
        print(note)
    return "".join(out)


#: A `#` comment, to the end of its line.
_COMMENT_RE = re.compile(r"#[^\n]*")


def uncommented(text):
    r"""`text` with every `#` comment blanked to spaces, POSITIONS PRESERVED.

    THE BUG THIS EXISTS FOR (found 2026-09-13, prompt_7 C1, on a real SLURM
    run). Every geometry regex in this module is `re.search(r"\bdepth:\s*(\d+)",
    part)` against the raw text, so it matches a `depth:` written in a COMMENT
    just as readily as the attribute -- and `re.sub(..., count=1)` then rewrites
    THE COMMENT and leaves the attribute alone.

    It fired the moment a comment was added that names a geometry it is NOT
    declaring: `# the paper's two banks of 512 x 64b would be `depth: 1024``
    made `_set_weight_geometry` read 1024 instead of 256, renormalise to 171,
    write "171" into the comment, and leave `depth: 256` beside the new
    `width: 384` -- 98,304 bits where 16,512 were intended, a SIX-FOLD capacity
    error with nothing on stdout to say so. The run's own log printed
    `filter_glb 1024x64b/8b -> 171x384b/8b`, which is the only reason it was
    caught.

    Blanking rather than deleting keeps every match span valid against the
    ORIGINAL string, so a caller can search the masked copy and splice into the
    real one.
    """
    return _COMMENT_RE.sub(lambda m: " " * len(m.group(0)), text)


def read_attr(text, key):
    """The integer value of `key:` in `text`, ignoring comments. None if absent."""
    m = re.search(rf"\b{re.escape(key)}:\s*(\d+)", uncommented(text))
    return int(m.group(1)) if m else None


def write_attr(text, key, value):
    """Replace the first UNCOMMENTED `key: <int>` in `text`. Returns the new text.

    Raises if there is none: a geometry rewrite that lands on nothing is the
    silent half of the bug `uncommented()` documents.
    """
    m = re.search(rf"\b{re.escape(key)}:\s*(\d+)", uncommented(text))
    if not m:
        raise ValueError(f"no uncommented `{key}:` to rewrite in:\n{text[:300]}")
    return text[:m.start(1)] + str(value) + text[m.end(1):]


def _weight_level_parts(text, scope="exclusive"):
    """Split `text` into `!Node` parts, tagging which ones hold WEIGHTS on chip.

    Shared by `_set_weight_geometry` and `_scale_weight_depth` so the two
    prompt_2 knobs can never disagree about which levels they are talking
    about -- if one narrowed a level the other did not shrink, the two arms of
    a pair would stop declaring the same geometry and the comparison would be
    void without anything saying so.

    Yields `(part, name, is_weight_level, why_not)`.
    """
    for part in re.split(r"(\n\s*-\s*!)", text):
        # COMMENT-BLIND REGEXES ARE HOW A COMMENT BECAME THE GEOMETRY
        # (`uncommented()`): every read below is against the masked copy, and
        # `_set_weight_geometry` writes through `write_attr()` for the same
        # reason.
        bare = uncommented(part)
        depth = re.search(r"\bdepth:\s*(\d+)", bare)
        name = re.search(r"name:\s*(\S+)", bare)
        name = name.group(1) if name else "?"
        if not depth:
            yield part, name, False, None
            continue
        keep = re.search(r"keep:\s*\[([^\]]*)\]", bare)
        keep = [s.strip() for s in keep.group(1).split(",") if s.strip()] if keep else []
        if "class: DRAM" in bare or name == "DRAM":
            # DRAM `datawidth` STAYS 8 ON EVERY ARM. `recon.py` owns the DRAM
            # K/N scaling in the evaluator; narrowing DRAM in the YAML too
            # would charge the same reduction twice (prompt_2, BEFORE ANY
            # NUMBER IS QUOTED, item 1).
            yield part, name, False, "DRAM (recon.py owns the DRAM K/N term)"
            continue
        if "Weights" not in keep:
            yield part, name, False, None
            continue
        if keep != ["Weights"] and scope != "shared":
            yield (part, name, False,
                   f"{name} holds {'+'.join(keep)}; "
                   f"ECC_WEIGHT_CAPACITY_SCOPE=shared includes it")
            continue
        if int(depth.group(1)) == 1:
            yield (part, name, False,
                   f"{name} declares depth 1 -- a pipeline latch, not a "
                   f"reuse level")
            continue
        yield part, name, True, None


def _set_weight_geometry(text, bits, levels=(), glb_mult=4, scope="exclusive",
                         arch="?", quiet=False,
                         weight_bits=widths.DEFAULT_WEIGHT_BITS):
    """PROMPT_2's WIDTH TABLE, applied PER LEVEL and PER ARM in one pass.

    THE ARMS DO NOT SHARE A DECLARED WIDTH. Each weight level declares the
    width that suits ITS OWN datawidth, and no level has to be legal for any
    other arm's datawidth:

        a level storing 8-bit weights    -> width  96 (spad) / 384 (GLB)
        a level storing 7-bit weights    -> width  98 / 392
        a level storing 6-bit weights    -> width  96 / 384
        a level storing 5-bit weights    -> width  95 / 380
        a level storing 4-bit weights    -> width  96 / 384

    **95 never has to divide 8.** The baseline/embedded arm is never mapped at
    width 95; it is mapped at 96. The only rule `timeloop-mapper` imposes is
    `width % (word_bits * block_size) == 0` (`buffer.cpp:302`, `block_size`
    defaults to 1, NO floor path, `exit=134` on a violation) -- and each width
    here is a multiple of the datawidth it is chosen for, by construction, so
    that abort is unreachable for every q. The lcm(q, 8) scheme of
    2026-09-11/12 is WITHDRAWN; `eccenergy/widths.py` records why.

    PER LEVEL, because an ERT arm narrows only the storage levels in its
    placement's `reduced` set: at `ECC_WEIGHT_DATAWIDTH_LEVELS=filter_glb` the
    GLB stores 5-bit weights at width 380 while `weights_spad` keeps 8-bit
    weights at width 96. One width for the whole design cannot express that.

    DEPTH IS COMMON TO EVERY ARM and holds the published TOTAL BITS:
    `depth' = round(depth * width / BASE_WIDTH)`, computed at the BASE width
    (96, x glb_mult above the PE array) rather than at this arm's own width.
    So the arms differ ONLY in `width` (by <= 2 %) and `datawidth`, which is
    what makes `capacity_ratio` reproduce prompt_2's `eff. capacity` column
    -- 1.1667 / 1.3333 / 1.5833 / 2.0000 -- and what leaves
    `assert_pair_geometry`'s depth check something real to check. It
    reproduces prompt_2's own depths: `weights_spad` 224 x 16 b = 3,584 b ->
    depth 37 at width 96; `filter_glb` 1024 x 64 b = 65,536 b -> depth 171 at
    width 384.

    `bits` is the reduced datawidth (`q`) or None for an all-8-bit arm;
    `levels` restricts WHICH levels take it (empty = every weight level).
    Every weight level is reshaped either way -- the reference arm reshapes
    too, or its depths would not match the arm it is compared with.

    DRAM IS NEVER TOUCHED: `recon.py` owns the DRAM K/N term, and narrowing
    DRAM here as well would charge the same reduction twice.

    Runs BEFORE `_scale_weight_depth`, so the swept ladder multiplies the
    renormalised depth rather than the published one.
    """
    want_levels = set(levels or ())
    names = [name for _p, name, is_w, _why
             in _weight_level_parts(text, scope) if is_w]
    if not names:
        return text
    spad = names[-1]
    unknown = want_levels - set(names)
    if unknown:
        raise ValueError(
            f"ECC_WEIGHT_DATAWIDTH_LEVELS names {', '.join(sorted(unknown))}, "
            f"which {arch} has no weight-carrying level called. It has: "
            f"{', '.join(sorted(names)) or 'none'}. Refusing rather than "
            f"narrowing every level and reporting it as a per-boundary "
            f"architecture.")

    out, touched, skipped, bad = [], [], [], []
    for part, name, is_weight, why_not in _weight_level_parts(text, scope):
        if not is_weight:
            if why_not:
                skipped.append(why_not)
            out.append(part)
            continue
        # READ OFF THE COMMENT-MASKED COPY. A `width:` or `depth:` written in
        # a comment is prose, not a declaration -- see `uncommented()` for the
        # six-fold capacity error that taught this.
        w0 = read_attr(part, "width")
        d0 = read_attr(part, "depth")
        dw0 = read_attr(part, "datawidth")
        if w0 is None or dw0 is None:
            skipped.append(f"{name} declares no "
                           + ("width:" if w0 is None else "datawidth:"))
            out.append(part)
            continue
        # WHICH datawidth this level ends up storing -- the arm's q only where
        # the arm says so, the payload width everywhere else.
        narrowed = bits is not None and (not want_levels or name in want_levels)
        q = int(bits) if narrowed else int(weight_bits)
        is_spad = name == spad
        want_w = widths.level_width(q, is_spad, glb_mult, weight_bits)
        want_d = widths.renormalised_depth(d0, w0, is_spad, glb_mult,
                                                weight_bits)
        if want_w % q != 0:                      # unreachable; a guard, not a path
            bad.append(f"{name}: width {want_w} % datawidth {q} != 0")
            out.append(part)
            continue
        if (w0, d0, dw0) == (want_w, want_d, q):
            skipped.append(f"{name} already declares {d0}x{w0}b/{dw0}b")
            out.append(part)
            continue
        # WRITE THROUGH `write_attr`, which finds the attribute on the masked
        # copy and splices into the real text -- `re.sub(..., count=1)` here
        # rewrote a COMMENT and left the attribute alone.
        part = write_attr(part, "width", want_w)
        part = write_attr(part, "depth", want_d)
        part = write_attr(part, "datawidth", q)
        touched.append(
            f"{name} {d0}x{w0}b/{dw0}b -> {want_d}x{want_w}b/{q}b "
            f"({want_w // q} weights/word, {d0 * w0:,} -> {want_d * want_w:,} bits)"
            + ("" if narrowed or bits is None else " [not in "
               "ECC_WEIGHT_DATAWIDTH_LEVELS: keeps the payload width]"))
        out.append(part)
    if bad:
        raise ValueError(
            f"{arch}: THE WIDTH TABLE produced a width its own datawidth does "
            f"not divide -- " + "; ".join(bad) + ".\n"
            f"  timeloop-mapper asserts `width % (word_bits * block_size) == 0` "
            f"(buffer.cpp:302) and ABORTS; there is no floor path.\n"
            f"  Every entry of eccenergy/widths.WIDTH_TABLE is a multiple "
            f"of its own q, so this is a table edit, not a configuration\n"
            f"  problem. Run `python3 -m eccenergy.code_widths` and fix the "
            f"entry; do NOT reach for a width that suits a DIFFERENT arm.")
    if not quiet and arch:
        note = (f"  [weight-geometry] {arch}: THE WIDTH TABLE (base "
                f"{widths.base_width(weight_bits)}b, GLB {glb_mult}x"
                + (f", q={bits} on "
                   + ("+".join(sorted(want_levels)) if want_levels else "every level")
                   if bits is not None else ", 8-bit arm")
                + ") on " + (", ".join(touched) if touched else "NOTHING"))
        if skipped:
            note += "; skipped " + "; ".join(skipped)
        print(note)
    return "".join(out)

def _scale_weight_depth(text, scale, levels=(), scope="exclusive", arch="?",
                        quiet=False):
    """PROMPT_2's ONLY SWEPT VARIABLE: `depth:` of the on-chip weight levels.

    Deliberately a SEPARATE knob from `_scale_weight_capacity`, even though
    the two rewrite the same field. That one exists to stand a narrower word
    up as a deeper array, so its energy has to be corrected back by
    `recon.capacity_dilation_correction()`. Here the depth change is the
    experiment: a shallower array really IS a smaller array, and its cheaper
    per-access energy is a real saving, not an artifact to undo. Sharing the
    slug would let a corrected run be read as an uncorrected one, so they get
    separate cache directories (`wdepth<scale>` vs `wcap<scale>`).

    `levels` restricts the rewrite to the named levels. Empty means every
    weight-carrying level, which is the default AND the limitation prompt_2
    records: one scale then moves `weights_spad` and `filter_glb` together, so
    it locates the zone but cannot say which level bought it. The second pass
    holds one at x1 and sweeps the other, which is what naming levels is for.
    A name that matches nothing is an error, not a silent no-op -- a typo
    there would quietly sweep every level and report it as a per-level result.
    """
    if scale == 1.0:
        return text
    want = set(levels or ())
    out, touched, skipped, seen = [], [], [], set()
    for part, name, is_weight, why_not in _weight_level_parts(text, scope):
        if not is_weight:
            if why_not:
                skipped.append(why_not)
            out.append(part)
            continue
        seen.add(name)
        if want and name not in want:
            skipped.append(f"{name} (not in ECC_WEIGHT_DEPTH_LEVELS)")
            out.append(part)
            continue
        d = read_attr(part, "depth")
        nd = max(1, int(round(d * scale)))
        if nd == d:
            skipped.append(f"{name} (depth {d} x {scale:g} rounds back to {d})")
            out.append(part)
            continue
        touched.append(f"{name} {d}->{nd}")
        out.append(write_attr(part, "depth", nd))
    unknown = want - seen
    if unknown:
        raise ValueError(
            f"ECC_WEIGHT_DEPTH_LEVELS names {', '.join(sorted(unknown))}, "
            f"which {arch} has no weight-carrying level called. It has: "
            f"{', '.join(sorted(seen)) or 'none'}. Refusing rather than "
            f"sweeping every level and reporting it as a per-level result.")
    if not quiet and arch:
        note = f"  [weight-depth] {arch}: x{scale:g} on " + (
            ", ".join(touched) if touched else "NOTHING")
        if skipped:
            note += "; skipped " + "; ".join(skipped)
        print(note)
    return "".join(out)


#: The loop dimensions a WEIGHT tile is indexed by. A `factors:` pin on any of
#: them caps how many weights a level can hold whatever its capacity is; a pin
#: on N, P or Q does not (weights do not index them) and is left alone, so the
#: activation and partial-sum structure of the dataflow is untouched.
WEIGHT_DIMENSIONS = ("M", "C", "R", "S")


# ---------------------------------------- prompt_7 Phase C: TIME in the MAPPER
def _component_attr_lines(text):
    """`{component name: (attributes-line index, attribute indent)}`.

    One reader for every Phase C patch below, so `read_bandwidth`, the
    bandwidth scale and the bank geometry can never disagree about which
    component they are editing. A component that declares no `attributes:`
    block is absent from the map, and the caller refuses rather than inventing
    one -- a level with no attributes is a level with no geometry, which is not
    something this study has.
    """
    lines = text.split("\n")
    out = {}
    for b in _node_blocks(lines):
        if b["kind"] != "Component" or b["name"] is None or b["attr"] is None:
            continue
        out[b["name"]] = (b["attr"], b["ind"] + 4)
    return out


def _insert_attributes(text, additions, arch="?"):
    """Add attribute lines to named components. `additions` = {name: [lines]}.

    Lines go DIRECTLY under the component's `attributes:` key, indented to
    match it, and each carries its own trailing comment. A name the file does
    not have is an ERROR: these attributes are what makes one mapper arm a
    different chip from another, and one that silently failed to land would be
    two arms sharing one architecture under two directory names -- exactly the
    failure prompt_6 RULE 4.4.5 exists to prevent.
    """
    if not additions:
        return text
    where = _component_attr_lines(text)
    missing = [n for n in additions if n not in where]
    if missing:
        raise ValueError(
            f"{arch}: no component with an `attributes:` block called "
            f"{', '.join(sorted(missing))}; the file has "
            f"{', '.join(sorted(where)) or 'none'}. Refusing rather than "
            f"dropping a declaration that separates two mapper arms.")
    at = {where[n][0]: (where[n][1], additions[n]) for n in additions}
    lines, out = text.split("\n"), []
    for idx, line in enumerate(lines):
        out.append(line)
        if idx in at:
            ind, add = at[idx]
            out.extend(" " * ind + a for a in add)
    return "\n".join(out)


def _declare_dram_bandwidth(text, items_per_cycle, arch="?", quiet=False):
    """prompt_7 C1.1: give the DRAM level the off-chip speed limit THE MAPPER
    READS.

    Until this landed no architecture in `archs/` declared an off-chip
    bandwidth at all, so Timeloop skipped the DRAM throughput check entirely
    (`buffer.cpp:2575` gates on `IsSpecified()`), off-chip traffic cost ZERO
    cycles, and the search optimised a machine with an infinitely fast memory.
    That is prompt_7's Defect 1, and it is why a reconstruction arm's K/N
    traffic saving could never appear as latency.

    `shared_bandwidth`, NOT `read_bandwidth` + `write_bandwidth`. The DQ bus is
    one wire set that reads and writes take turns on, so the limit is on their
    SUM -- which is exactly the term `latency_post.roofline()` already charges
    (`offchip_limit / (d_r + d_w)`). Declaring the two directions separately
    would let Timeloop deliver 2x this number while the evaluator capped it at
    1x: two timing models for one bus, and one of them wrong.

    UNITS ARE ITEMS PER CYCLE of THIS design's clock. `config` owns the
    conversion from MB/s (`dram_items_per_cycle_for`), so the number here and
    the number the roofline uses have one source.

    MEASURED CONSEQUENCE (prompt_7 7.2, 12 real mapper searches): with a
    binding limit the mapper picks a DIFFERENT plan on both test shapes -- one
    of them spends parallelism to comply, folding onto 96 of 168 PEs. At
    LPDDR4-3200 the limit never binds and nothing moves.
    """
    if items_per_cycle is None:
        return text
    add = [f"shared_bandwidth: {items_per_cycle:.6g}   # items/cycle: "
           f"ECC_DRAM_BANDWIDTH_MBPS at this design's own clock",
           "                     # ONE bus -- reads and writes share it, which is",
           "                     # the term latency_post.roofline() charges."]
    out = _insert_attributes(text, {"DRAM": add}, arch)
    if not quiet and arch:
        print(f"  [off-chip] {arch}: DRAM shared_bandwidth "
              f"{items_per_cycle:.6g} items/cycle")
    return out


def _declare_bw_scale(text, factors, arch="?", quiet=False):
    """prompt_7 C1.2: `per_dataspace_bandwidth_consumption_scale` on every
    stage of this arm's boundary.

    Timeloop multiplies ONE dataspace's bandwidth demand by the factor
    (`buffer.cpp:2556`); timeloopfe v4 declares the attribute
    (`arch.py:538`) and every stats file prints `Bandwidth Consumption Scale`,
    so a version that ignored it would be visible on disk. Measured on fixed
    mappings with off-chip bandwidth binding: -17.28% and -21.94% cycles, with
    dynamic energy BIT-IDENTICAL (prompt_7 A.2) -- it is a timing declaration
    and nothing else.

    THE FACTOR IS PER STAGE and `recon.arm_bw_factors()` derives it: `K/N` at
    DRAM, where the weights are a bit stream, and `q/8` on chip, where a level
    holds whole weights in a narrower word. Using one everywhere is a 5%
    silent inconsistency with `ECC_RECON_PACKING`, not a rounding choice.

    NETWORK STAGES ARE SKIPPED HERE AND ONLY HERE. They are part of what the
    arm DECLARES (`recon.arm_bw_scale`, which is what makes R3 a different
    chip from R2), but `LegacyNetwork::ComputePerformance()` is an empty stub
    and there is no YAML component to attach an attribute to. Reporting rule
    R-3 is that fact, written down.
    """
    timed = {lvl: f for lvl, f in (factors or {}).items()
             if f["kind"] in ("dram", "storage")}
    if not timed:
        return text
    add = {}
    for lvl, f in timed.items():
        add[lvl] = [f"per_dataspace_bandwidth_consumption_scale: "
                    f"{{{f['dataspace']}: {f['factor']:.6f}}}   # {f['why']}"]
    out = _insert_attributes(text, add, arch)
    if not quiet and arch:
        print("  [bw-scale] " + arch + ": "
              + ", ".join(f"{lvl} {f['dataspace']} x{f['factor']:.6f}"
                          for lvl, f in timed.items())
              + (("; declared but NOT timed (no network speed model, "
                  "reporting rule R-3): "
                  + ", ".join(lvl for lvl, f in factors.items()
                              if f["kind"] == "network"))
                 if any(f["kind"] == "network" for f in factors.values()) else ""))
    return out


def _bitaware_onchip_bandwidth(text, levels, factor, arch="?", quiet=False):
    """prompt_7 C1.3: a narrowed level's declared port, priced in BITS.

    Timeloop's throughput check counts ITEMS per cycle and a narrow weight is
    still one item, so `datawidth: q` alone is invisible to the clock. A real
    port moves BITS: a level storing `q`-bit weights delivers `8/q` times as
    many of them per cycle through the same wires, and THAT is what the
    architecture has to say for the mapper to see it.

    WHERE IT LANDS. `filter_glb`'s declared `read_bandwidth: 16` is literally
    what caps every fully-connected layer at 9.52% PE utilisation -- 16 of 168
    PEs, measured on resnet18 `fc` and mobilenet `classifier.1`, where a
    32-PE candidate was rejected because it would have throttled (prompt_7
    4.5, A.7). Those layers are 0.28% / 2.10% of their models' cycles, so the
    aggregate CNN effect is ~0.1-1%; on a batch-1 transformer every layer is
    that layer (Phase D).

    ONLY THE LEVELS THE ARM NARROWS. A level still storing 8-bit weights moves
    the same bits per cycle it always did, and raising its port would be a
    free architecture change credited to the code.
    """
    if not levels or factor == 1.0:
        return text
    # DRAM IS NEVER BIT-AWARE. `recon.py` owns the DRAM K/N term and C1.2
    # declares it as a bandwidth SCALE; raising the off-chip limit by 8/q as
    # well would charge one reduction twice -- the same trap
    # `_weight_level_parts` records for `datawidth`. Unreachable today
    # (`arm_narrow_levels` keeps only storage stages), which is exactly when a
    # guard is cheap.
    offchip = [n for n in levels if "dram" in n.lower()]
    if offchip:
        raise ValueError(
            f"{arch}: ECC_ONCHIP_BW_BITAWARE reached {', '.join(offchip)}. The "
            f"off-chip limit is ONE bus carrying a bit stream and its relief is "
            f"already declared as per_dataspace_bandwidth_consumption_scale "
            f"(prompt_7 C1.2); scaling it by 8/q as well would charge the "
            f"reduced representation twice.")
    where = _component_attr_lines(text)
    missing = [n for n in levels if n not in where]
    if missing:
        raise ValueError(f"{arch}: cannot make {', '.join(missing)} bit-aware; "
                         f"no such component with attributes")
    lines, out, touched = text.split("\n"), [], []
    current = None
    starts = {idx: name for name, (idx, _ind) in where.items()}
    for idx, line in enumerate(lines):
        if idx in starts:
            current = starts[idx]
        m = re.match(r"^(\s*)(read_bandwidth|write_bandwidth|shared_bandwidth):"
                     r"\s*([\d.eE+-]+)(.*)$", line)
        if m and current in levels:
            ind, key, val, rest = m.groups()
            new = float(val) * factor
            out.append(f"{ind}{key}: {new:.6g}"
                       f"   # x{factor:g} = 8/q: this level stores q-bit weights "
                       f"(was {val}){rest}")
            touched.append(f"{current}.{key} {val} -> {new:.6g}")
            continue
        out.append(line)
    if not touched:
        raise ValueError(
            f"{arch}: ECC_ONCHIP_BW_BITAWARE is on and the arm narrows "
            f"{', '.join(levels)}, but none of those levels declares a "
            f"bandwidth to scale. A bit-aware port that lands on nothing is a "
            f"no-op reported as an architecture change.")
    if not quiet and arch:
        print(f"  [bit-aware bw] {arch}: " + ", ".join(touched))
    return "\n".join(out)


def _bank_geometry(text, arch="?", quiet=False):
    """prompt_7 C1.7: let a level's DECLARED bank count reach CACTI.

    `smartbuffer_SRAM` declares no `n_banks`, so every `n_banks:` in this
    study's architectures was inert and CACTI priced one monolithic array. The
    plug-in itself takes the attribute (`cacti_wrapper.py:137`) and hands it to
    CACTI as `-UCA bank N`; only the compound in the way had to be replaced.
    Measured on `eyeriss_like_wglb` (Accelergy, 45nm, this design's own
    geometry): `ifmap_glb` read 23.539 -> 16.571 pJ, `psum_glb` 22.729 ->
    16.053, `filter_glb` 13.591 -> 11.745.

    ONLY LEVELS THAT DECLARE `n_banks:` THEMSELVES ARE SWITCHED, and that is
    the whole subtlety. `timeloopfe` v4 gives EVERY storage level a default
    `n_banks: 2`, which is visible in the flattened architecture and is a
    published number for none of them. Forwarding it wholesale would hand the
    depth-3 `weights_spad` a two-bank model out of a front-end default -- and
    the CACTI wrapper floors depth at `64 x n_banks`, so on a shallow array
    that default moves the price through the FLOOR rather than through any
    banking. A level whose paper says nothing about banking therefore keeps
    `smartbuffer_SRAM` and prices exactly as it did before Phase C.

    KNOWN, RECORDED, NOT FUDGED: CACTI is called at
    `2 ** ceil(log2(n_banks))`, so the 13-bank ifmap GLB is modelled as 16
    banks; the wrapper computes the linear correction `bankscale = 13/16` and
    then never applies it (`cacti_wrapper.py:177-178` -- it is dead in the
    plug-in, not here). Leakage does use the declared count. That is stated in
    `archs/_shared/provenance.yaml` and reported by `run.sh diagnose`; it is
    not silently compensated for here.
    """
    lines, out, touched = text.split("\n"), [], []
    blocks = {b["attr"]: b for b in _node_blocks(lines)
              if b["kind"] == "Component" and b["attr"] is not None}
    banked = set()
    for attr_idx, b in blocks.items():
        if b["cls"] != PLAIN_SRAM_CLASS:
            continue
        # the component's own body: from its `- !Component` line to the next
        end = min([i for i in blocks if i > attr_idx] or [len(lines)])
        body = "\n".join(lines[attr_idx:end])
        m = re.search(r"^\s*n_banks:\s*(\d+)", uncommented(body), re.M)
        if m and int(m.group(1)) > 1:
            banked.add(b["name"])
            touched.append(f"{b['name']} ({m.group(1)} banks)")
    if not banked:
        return text
    current = None
    for line in lines:
        m = re.match(r"^\s*name:\s*(\S+)", line)
        if m:
            current = m.group(1).split("#")[0].strip()
        c = re.match(r"^(\s*)class:\s*" + PLAIN_SRAM_CLASS + r"\s*(#.*)?$", line)
        if c and current in banked:
            out.append(f"{c.group(1)}class: {BANKED_SRAM_CLASS}"
                       f"   # prompt_7 C1.7: this level's declared n_banks "
                       f"reaches CACTI")
            continue
        out.append(line)
    if not quiet and arch:
        print(f"  [banks] {arch}: {BANKED_SRAM_CLASS} on " + ", ".join(touched)
              + "  (CACTI rounds to the next power of two and drops its own "
                "bankscale correction -- provenance.yaml `sram_banking`)")
    return "\n".join(out)


def _relax_weight_factors(text, arch="?", quiet=False):
    """TASK 4 LEVER 2: let the weight TILE grow into the room a dilation adds.

    WHY CAPACITY ALONE MEASURES ZERO. `_scale_weight_capacity` makes the buffer
    bigger; it does not make the mapper able to spend it. `eyeriss_like`'s
    `weights_spad` declares

        temporal:
          factors: [N=1, M=1, P=1, Q=1, S=1]

    and `M=1` pins the M tile AT THAT LEVEL to one, so the resident tile is
    M(8 from `psum_spad` below) x C(16) = 128 weights and stays 128 whatever the
    capacity is. Measured (FINDINGS 7.8): `weights held` is exactly 21,504 at
    x1, x1.6154, x4, x8, x16 AND x32, where the buffer is at 0.9 % fill. The
    binding constraint is the DATAFLOW CONSTRAINT, not the silicon, and no
    capacity sweep can find that out because the constraint does not move.

    WHAT IT REWRITES. On every weight-carrying level that
    `weight_capacity_levels()` reports (so: never DRAM, never a level holding no
    Weights, never a declared `depth: 1` latch), the `factors:` entries for
    M, C, R and S are dropped from the TEMPORAL constraints. N, P and Q keep
    their pins: weights do not index them, so relaxing those would change the
    activation and psum tiling instead of the weight tile, which is a different
    experiment.

    THIS IS A DIFFERENT DATAFLOW AND MUST BE LABELLED AS ONE. Eyeriss v1's
    `M=1` at the filter spad is the row-stationary dataflow; a design without it
    is not the chip JSSC 2017 describes, and `source: published` does not
    licence its name. It gets its own cache (`wrelax`) and both arms of a Task 4
    pair are mapped under it, so the COMPARISON stays fair even though neither
    arm is the published design.

    It also widens the mapspace, so the search has strictly more to explore at
    the same victory condition -- a relaxed run that comes back worse is
    evidence about the SEARCH, not about the dataflow.
    """
    touched, skipped = [], []
    parts = re.split(r"(\n\s*-\s*!)", text)
    out = []
    for part in parts:
        name = re.search(r"name:\s*(\S+)", part)
        name = name.group(1) if name else "?"
        keep = re.search(r"keep:\s*\[([^\]]*)\]", part)
        keep = [s.strip() for s in keep.group(1).split(",") if s.strip()] if keep else []
        depth = re.search(r"\bdepth:\s*(\d+)", uncommented(part))
        if ("Weights" not in keep or not depth or "class: DRAM" in part
                or name == "DRAM"):
            out.append(part)
            continue
        if int(depth.group(1)) == 1:
            skipped.append(f"{name} (declared depth 1 -- a latch)")
            out.append(part)
            continue
        # Only the TEMPORAL factors of this level, never a spatial container's.
        m = re.search(r"(temporal:\s*\n(?:\s+\w+:.*\n)*?\s+factors:\s*)"
                      r"\[([^\]]*)\]", part)
        if not m:
            skipped.append(f"{name} (no temporal factors: to relax)")
            out.append(part)
            continue
        entries = [e.strip() for e in m.group(2).split(",") if e.strip()]
        kept = [e for e in entries
                if e.split("=")[0].strip().upper() not in WEIGHT_DIMENSIONS]
        if len(kept) == len(entries):
            skipped.append(f"{name} (pins nothing a weight tile is indexed by)")
            out.append(part)
            continue
        dropped = [e for e in entries if e not in kept]
        part = part[:m.start(2) - 1] + "[" + ", ".join(kept) + "]" + part[m.end(2) + 1:]
        touched.append(f"{name} dropped {'/'.join(dropped)}")
        out.append(part)

    if not quiet and arch:
        note = f"  [weight-factor-relax] {arch}: " + (
            "; ".join(touched) if touched else "NOTHING")
        if skipped:
            note += "; skipped " + "; ".join(skipped)
        print(note)
    return "".join(out)


#: WHICH LEVELS EACH LOOP DIMENSION MAY BE SPLIT ACROSS, per design.
#:
#: THE PROBLEM THIS SOLVES, measured 2026-09-10 (FINDINGS 7.9). The mapper
#: reports its own mapspace for ONE layer of Eyeriss v1 as IndexFactorization
#: 7.41e10 x LoopPermutation 8.96e9 ~ 6.6e20. At the measured 400,000 valid
#: mappings per thread-hour, victory 50000 covers 0.07 % of ONE thread's
#: factorization subspace, ignoring permutations entirely. The consequence is
#: not slow convergence, it is NO convergence: the residual between victory
#: 4000 and 10000 is 43.7 % where the ECC effect being measured is 5.8 %, and
#: refetch is non-monotone in the budget. The search noise is larger than the
#: signal, so no Recon-vs-Embedded conclusion survives at ANY affordable
#: budget.
#:
#: RAISING THE BUDGET CANNOT FIX THAT AND SHRINKING THE SPACE CAN. A bigger
#: budget samples more of the same enormous space; both arms still land on
#: arbitrary points and the DIFFERENCE between two arbitrary points is noise
#: (measured: the ordering between the arms flips). Constraining the loop nest
#: collapses the space to something searchable EXHAUSTIVELY, and then each arm
#: gets its true optimum rather than a sample -- so the difference is
#: architectural by construction and the gate passes because there is nothing
#: left unsearched. progress.txt 2026-09-10 costs the tiers: pinning R and S to
#: one level each leaves 2.06e9 factorizations (still impossible), adding
#: P/Q across <=2 levels leaves 1.32e7 (29 h), and adding C/M across <=3
#: leaves 3.63e4 -- EXACT in about five minutes, and roughly 8x CHEAPER than
#: the victory-5000 run whose answer is 9-11 % wrong.
#:
#: WHY THE FREE SETS ARE THESE ONES. They are read off the BEST MAPPING THE
#: SEARCH HAS EVER FOUND for this design -- the victory-10000 embedded nest at
#: x1, 169.13 uJ against the victory-4000 mapping's 300.34 uJ on identical
#: silicon. Constraining around a known-good region is a choice and it is
#: stated: the exhaustive answer is the best mapping IN THIS FAMILY, not in
#: the whole space. What makes it a fair ECC comparison is that BOTH ARMS get
#: the identical constraint, so neither is handed a region the other cannot
#: reach.
#:
#: THIS IS A DIFFERENT DATAFLOW AND MUST BE LABELLED AS ONE, exactly as
#: `_relax_weight_factors` is: a design run under it is not the chip JSSC 2017
#: describes, `source: published` does not licence its name, and it gets its
#: own cache slug (`mcons`).
#:
#: `weights_spad` deliberately KEEPS M free, because this lever is meant to be
#: run together with `ECC_WEIGHT_FACTOR_RELAX=1` -- that relaxation exists to
#: let the weight TILE grow into the room a shallower/narrower array leaves,
#: and pinning M back at that level here would undo it.
MAPSPACE_FREE_LEVELS = {
    "eyeriss_like_wglb": {
        # dimension: the levels it may be split across. Pinned to 1 elsewhere.
        "C": ("DRAM", "PE", "weights_spad"),
        "M": ("ifmap_glb", "PE_column", "weights_spad", "psum_spad"),
        "R": ("psum_glb",),
        "S": ("PE",),
        "P": ("ifmap_glb", "psum_glb"),
        "Q": ("filter_glb", "PE_column"),
        "N": (),                      # N = 1 in every workload here
    },
}

#: Every loop dimension the constraint reasons about.
MAPSPACE_DIMENSIONS = ("N", "C", "M", "R", "S", "P", "Q")


def _merge_factor_list(existing, pins):
    """`existing` factor entries plus `pins`, with `pins` winning. Order-stable."""
    out, seen = [], set()
    for e in list(existing) + list(pins):
        k = e.split("=")[0].strip().upper()
        if k in seen:
            out = [x for x in out if x.split("=")[0].strip().upper() != k]
        seen.add(k)
        out.append(e)
    # `pins` appended last already won; de-duplicate keeping the LAST
    final, seen = [], set()
    for e in reversed(out):
        k = e.split("=")[0].strip().upper()
        if k in seen:
            continue
        seen.add(k)
        final.append(e)
    return list(reversed(final))


def _constrain_mapspace(text, arch="?", quiet=False):
    """Pin every loop dimension to 1 at the levels `MAPSPACE_FREE_LEVELS` does
    not list, so the index-factorization space collapses to something the
    mapper can search EXHAUSTIVELY.

    A storage level gets `constraints.temporal.factors`; a spatial container
    gets `constraints.spatial.factors`. Both are created if absent and merged
    if present, with these pins winning -- a design that already pins a
    dimension keeps that pin, and one that leaves it free has it pinned here
    unless the free set names the level.
    """
    spec = MAPSPACE_FREE_LEVELS.get(arch)
    if not spec:
        if not quiet and arch:
            print(f"  [mapspace] {arch}: no MAPSPACE_FREE_LEVELS entry -- "
                  f"NOT constrained (the search is unbounded here)")
        return text
    touched, out = [], []
    for part in re.split(r"(\n\s*-\s*!)", text):
        name = re.search(r"name:\s*(\S+)", part)
        if not name or "!" in part[:3] or not re.search(r"name:", part):
            out.append(part)
            continue
        name = name.group(1)
        if name == "DRAM" and "class: DRAM" not in part:
            out.append(part)
            continue
        if re.search(r"class:\s*intmac", part):
            out.append(part)               # the arithmetic level takes none
            continue
        # ONLY REAL LOOP LEVELS. A storage component declares `depth:`; a
        # spatial container declares `spatial: {meshX/meshY}`. A bare grouping
        # container (`system`, and the accelerator container itself) is
        # neither -- it carries no loops, so pinning every dimension to 1 on
        # it would be inventing a constraint on a level Timeloop does not map.
        is_storage = re.search(r"\bdepth:\s*\d+", uncommented(part)) is not None
        is_spatial = re.search(r"\n\s*spatial:\s*\{[^}]*mesh", part) is not None
        if not (is_storage or is_spatial):
            out.append(part)
            continue
        pins = [f"{d}=1" for d in MAPSPACE_DIMENSIONS
                if name not in spec.get(d, ())]
        if not pins:
            out.append(part)
            continue
        kind = "spatial" if is_spatial else "temporal"
        m = re.search(rf"({kind}:\s*\n(?:\s+\w+:.*\n)*?\s+factors:\s*)\[([^\]]*)\]",
                      part)
        if m:
            existing = [e.strip() for e in m.group(2).split(",") if e.strip()]
            merged = _merge_factor_list(existing, pins)
            part = part[:m.start(2) - 1] + "[" + ", ".join(merged) + "]" \
                + part[m.end(2) + 1:]
            touched.append(f"{name}({kind}) {','.join(pins)}")
            out.append(part)
            continue
        # no factors: list at that kind -- create the block
        km = re.search(rf"\n(\s+){kind}:\s*\n", part)
        if km:
            ind = km.group(1)
            part = (part[:km.end()] + f"{ind}  factors: [{', '.join(pins)}]\n"
                    + part[km.end():])
            touched.append(f"{name}({kind}, new factors) {','.join(pins)}")
            out.append(part)
            continue
        cm = re.search(r"\n(\s+)constraints:\s*\n", part)
        if cm:
            ind = cm.group(1)
            block = (f"{ind}  {kind}:\n"
                     f"{ind}    factors: [{', '.join(pins)}]\n")
            part = part[:cm.end()] + block + part[cm.end():]
            touched.append(f"{name}({kind}, new block) {','.join(pins)}")
            out.append(part)
            continue
        nm = re.search(r"\n(\s+)name:\s*\S+\s*\n", part)
        if nm:
            ind = nm.group(1)
            block = (f"{ind}constraints:\n{ind}  {kind}:\n"
                     f"{ind}    factors: [{', '.join(pins)}]\n")
            part = part.rstrip("\n") + "\n" + block
            touched.append(f"{name}({kind}, new constraints) {','.join(pins)}")
            out.append(part)
            continue
        out.append(part)
    if not quiet and arch:
        print(f"  [mapspace] {arch}: EXHAUSTIVE-SEARCH CONSTRAINT on "
              + ("; ".join(touched) if touched else "NOTHING")
              + "   (a DIFFERENT DATAFLOW -- not the published chip)")
    return "".join(out)


def weight_capacity_levels(arch, cfg):
    """Which levels a capacity dilation would rewrite, and by how much.

    Reported by `validate`/`diagnose` and by the Task 4 experiment, so a run
    that dilates NOTHING says so instead of quietly reproducing the declared
    mapping under a Task 4 heading.
    """
    text = arch_source(arch, cfg).read_text()
    rows = []
    for part in re.split(r"(?=\n\s*-\s*!)", text):
        depth = re.search(r"\bdepth:\s*(\d+)", uncommented(part))
        name = re.search(r"name:\s*(\S+)", part)
        if not depth or not name or name.group(1) == "DRAM":
            continue
        keep = re.search(r"keep:\s*\[([^\]]*)\]", part)
        keep = [s.strip() for s in keep.group(1).split(",") if s.strip()] if keep else []
        if "Weights" not in keep:
            continue
        width = re.search(r"\bwidth:\s*(\d+)", uncommented(part))
        dw = re.search(r"\bdatawidth:\s*(\d+)", uncommented(part))
        d = int(depth.group(1))
        per_word = (int(width.group(1)) // int(dw.group(1))) if width and dw else 1
        rows.append({"level": name.group(1), "depth": d, "weights_per_word": per_word,
                     "weights_per_instance": d * per_word,
                     "shared_with": [k for k in keep if k != "Weights"],
                     "latch": d == 1})
    return rows


def patched_weight_geometry(arch, cfg):
    """The geometry of every weight-carrying level AS THE MAPPER WILL SEE IT.

    `weight_capacity_levels()` reads the SOURCE YAML, which is what a
    dilation-scope question needs. This reads the PATCHED text -- after
    `ECC_WEIGHT_DEPTH_SCALE` and `ECC_WEIGHT_DATAWIDTH` -- which is what the
    fairness rule needs, because the whole prompt_2 claim is that the two arms
    differ in `datawidth` and in NOTHING ELSE.

    Returns `{level: {"depth", "width", "datawidth", "weights_per_word",
    "weights"}}`, outer to inner.
    """
    text = _patched_text(arch, cfg, quiet=True)
    out = {}
    for part in re.split(r"(?=\n\s*-\s*!)", text):
        depth = re.search(r"\bdepth:\s*(\d+)", uncommented(part))
        name = re.search(r"name:\s*(\S+)", part)
        if not depth or not name or name.group(1) == "DRAM":
            continue
        keep = re.search(r"keep:\s*\[([^\]]*)\]", part)
        keep = [s.strip() for s in keep.group(1).split(",") if s.strip()] if keep else []
        if "Weights" not in keep:
            continue
        width = re.search(r"\bwidth:\s*(\d+)", uncommented(part))
        dw = re.search(r"\bdatawidth:\s*(\d+)", uncommented(part))
        d = int(depth.group(1))
        w = int(width.group(1)) if width else None
        b = int(dw.group(1)) if dw else None
        per_word = (w // b) if (w and b) else 1
        out[name.group(1)] = {
            "depth": d, "width": w, "datawidth": b,
            "weights_per_word": per_word, "weights": d * per_word,
            "shared_with": [k for k in keep if k != "Weights"]}
    return out


def assert_pair_geometry(arch, cfg_ref, cfg_arm, ref_name="embedded",
                         arm_name="recon"):
    """PROMPT_2 FAIRNESS RULE, mechanically: the two arms of a pair must hold
    the same LEVELS at the same DEPTH.

    **IT DOES NOT CHECK `width` AND IT DOES NOT CHECK `datawidth`.** Under
    prompt_2's WIDTH TABLE the arms declare DIFFERENT widths on purpose -- each
    one the width that suits its own datawidth, 96 / 98 / 96 / 95 / 96 -- and
    neither has to be legal for the other's datawidth, because neither is ever
    mapped on the other's silicon. Asserting a shared width is what produced
    the withdrawn `lcm(q, 8)` scheme (56 / 24 / 40) and, through it,
    BCH(63,39)'s spurious 37.69 %: see `eccenergy/widths.py`. Do not
    reintroduce that check.

    WHAT IT STILL CHECKS, AND WHY IT IS AN ASSERTION AND NOT A CONVENTION.
    DEPTH. This is the exact defect that made the earlier sweep prove nothing:
    capacity was expressed as `depth x N/K`, so Accelergy priced the
    reconstruction arm's array 1.18-1.46x dearer per access and the optimiser
    had a REASON to leave the room unused (FINDINGS 7.8). A depth difference is
    real silicon one arm does not have. `_set_weight_geometry` renormalises
    every arm's depth at the BASE width precisely so this stays true while the
    widths differ, which is what makes the check meaningful rather than
    vacuous.

    `ECC_DISABLE_ASSERT_PAIR_GEOMETRY=1` turns the depth check off, for a study
    that deliberately varies depth between the reconstruction and embedded arms.
    It is the ONLY thing that knob disables -- there is no width check for it
    to disable. Default 0: a depth difference nobody asked for is still a void
    comparison.

    Returns the per-level record either way, so a caller that disabled the
    assertion can still print what differs.
    """
    a = patched_weight_geometry(arch, cfg_ref)
    b = patched_weight_geometry(arch, cfg_arm)
    disabled = bool(getattr(cfg_arm, "disable_pair_geometry_assert", False)
                    or getattr(cfg_ref, "disable_pair_geometry_assert", False))
    problems = []
    if set(a) != set(b):
        problems.append(
            f"different weight LEVELS: {ref_name} has "
            f"{', '.join(sorted(a)) or 'none'}; {arm_name} has "
            f"{', '.join(sorted(b)) or 'none'}")
    for level in sorted(set(a) & set(b)):
        # DEPTH ONLY. `width` and `datawidth` are the treatment.
        if a[level]["depth"] != b[level]["depth"]:
            problems.append(
                f"{level}.depth: {ref_name}={a[level]['depth']} "
                f"{arm_name}={b[level]['depth']}")
    if problems and not disabled:
        raise ValueError(
            f"{arch}: the two arms do NOT hold the same amount of silicon -- "
            + "; ".join(problems) + ".\n"
            f"  prompt_2's fairness rule is that both arms declare the same "
            f"LEVELS at the same DEPTH; only `width:` (from THE WIDTH TABLE,\n"
            f"  per arm) and `datawidth:` differ, and CACTI never sees "
            f"`datawidth`. A DEPTH difference is priced by Accelergy as real\n"
            f"  silicon one arm does not have, which is the defect that "
            f"invalidated the previous sweep (FINDINGS 7.8). The comparison\n"
            f"  is void; fix the configuration rather than correcting the "
            f"energy. A study that varies depth between the arms on purpose\n"
            f"  sets ECC_DISABLE_ASSERT_PAIR_GEOMETRY=1 (env.sh section 5), "
            f"which disables THIS check and nothing else.")
    if problems and disabled:
        print(f"  [pair-geometry] {arch}: ECC_DISABLE_ASSERT_PAIR_GEOMETRY=1 -- "
              f"depth differs and is ALLOWED: " + "; ".join(problems))
    return {level: {"depth": a[level]["depth"],
                    f"{ref_name}_width": a[level]["width"],
                    f"{arm_name}_width": b[level]["width"],
                    ref_name: a[level]["datawidth"],
                    arm_name: b[level]["datawidth"],
                    f"{ref_name}_weights": a[level]["weights"],
                    f"{arm_name}_weights": b[level]["weights"],
                    "capacity_ratio": (b[level]["weights"] / a[level]["weights"]
                                       if a[level]["weights"] else 0.0)}
            for level in sorted(set(a) & set(b), key=list(a).index)}

def _patched_text(arch, cfg, apply_per_arch=True, quiet=False):
    text = arch_source(arch, cfg).read_text()
    text = _patch_dram_depth(text, cfg.dram_depth)
    if cfg.force_technology:
        text = _force_technology(text, cfg.force_technology)
    if apply_per_arch and cfg.force_datawidth:
        text = _force_datawidth(text, cfg.force_datawidth, None if quiet else arch)
    if apply_per_arch and cfg.acc_bits_override is not None:
        text = _force_acc_bits(text, cfg.acc_bits_override, arch, quiet)
    # prompt_2: THE WIDTH TABLE (width + depth + datawidth together), then the
    # depth ladder. Together because a level's width is chosen FROM the
    # datawidth that level ends up storing -- they are one decision, and the
    # `width % datawidth == 0` constraint is then satisfied by construction
    # instead of being checked after the fact. UNCONDITIONAL: the table is a
    # property of the study, not a knob, so the 8-bit reference arm is
    # reshaped too (and must be, or its depths would not match the arm it is
    # compared with). The ladder runs after, so it multiplies the renormalised
    # depth rather than the published one.
    if apply_per_arch:
        text = _set_weight_geometry(text, getattr(cfg, "weight_datawidth", None),
                                    getattr(cfg, "weight_datawidth_levels", ()),
                                    cfg.weight_width_glb_mult,
                                    cfg.weight_capacity_scope, arch, quiet,
                                    weight_bits=cfg.weight_bits)
    # BOTH depth knobs run AFTER the reshape, on the renormalised depth. They
    # used to straddle it (capacity before, ladder after), which was harmless
    # only while the reshape was usually a no-op: once THE WIDTH TABLE applies
    # to every run, scaling before renormalising rounds twice and the two
    # knobs stop meaning the same thing at the same scale.
    if apply_per_arch and cfg.weight_capacity_scale != 1.0:
        text = _scale_weight_capacity(text, cfg.weight_capacity_scale,
                                      cfg.weight_capacity_scope, arch, quiet)
    if apply_per_arch and getattr(cfg, "weight_depth_scale", 1.0) != 1.0:
        text = _scale_weight_depth(text, cfg.weight_depth_scale,
                                   cfg.weight_depth_levels,
                                   cfg.weight_capacity_scope, arch, quiet)
    if apply_per_arch and cfg.weight_factor_relax:
        text = _relax_weight_factors(text, arch, quiet)
    # AFTER the relax, deliberately. The relax frees M/C/R/S on the weight
    # levels so the TILE can grow into the room; this then pins every
    # dimension at the levels the free-set does not name. `weights_spad` keeps
    # M and C free in the free-set precisely so the two levers compose instead
    # of cancelling.
    if apply_per_arch and getattr(cfg, "mapspace_constrain", False):
        text = _constrain_mapspace(text, arch, quiet)
    # ---------------------------------------------- prompt_7 Phase C: TIME
    # Three declarations that give the MAPPER what only the evaluator had.
    # They run AFTER the geometry so a bit-aware port is scaled from the width
    # table's final numbers, and BEFORE the NoC so every one of them is hashed.
    #
    # C1.1 -- the off-chip speed limit, study-wide. Unconditional like
    # `_patch_dram_depth`: it is a property of the modelled system, not of an
    # arm, and every arm shares it.
    text = _declare_dram_bandwidth(text, cfg.dram_items_per_cycle_for(arch),
                                   arch, quiet)
    if apply_per_arch:
        # C1.2 -- what THIS boundary declares it moves less of. Per-arm by
        # construction: it is one of the three axes that make two boundaries
        # two chips (prompt_7 6.4).
        text = _declare_bw_scale(text, cfg.arm_bw_factors_for(arch), arch, quiet)
        # C1.3 -- and how fast the narrowed levels' own ports then run. Only
        # the levels this arm narrows; a level still holding 8-bit weights
        # moves the same bits per cycle it always did.
        text = _bitaware_onchip_bandwidth(
            text, tuple(getattr(cfg, "weight_datawidth_levels", ()) or ()),
            cfg.onchip_bw_bitaware_factor(), arch, quiet)
    # C1.7 -- the declared bank count, study-wide. An energy declaration, not
    # a timing one, but it belongs to the same cold pass.
    text = _bank_geometry(text, arch, quiet)
    # Last, so the coefficients land on the final text and are hashed by
    # arch_fingerprint(). Applied regardless of apply_per_arch: the NoC model
    # is a study-wide treatment, not a per-architecture no-op candidate.
    text = _inject_noc(text, arch, cfg)
    return text


