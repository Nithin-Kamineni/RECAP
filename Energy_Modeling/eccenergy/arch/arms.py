"""Which placements get their own MAPPING -- the distinct chips, not the boundaries.

THE ARMS TO MAP ARE THE DISTINCT CHIPS (prompt_7 Phase B, 2026-09-12).
`mapper_arms(arch, cfg)` is the reference plus every boundary that differs on
any of three axes -- `datawidth: q`, the ERT bump, the declared per-dataspace
bandwidth scale. A NETWORK stage is carried in that set and marked `no-op`,
because Timeloop has no network timing model but a boundary that adds one is
still a different declaration. Six arms on `eyeriss_like_wglb`, five on v2.

`ert_arms()` keeps its older, narrower meaning -- which boundaries have an
ERT-injectable encoder -- and is a STRICT SUBSET of the above. Never
`if key == "recon2"`: every property here is DERIVED from the `Placement`
record, and `ert_injectable()` condition 3 is derived from the CLOCK GATING
(prompt_7 6.3), not from a key name. The bump has two rows and the exclusion
holds only when BOTH are constant across the mapspace: the access row on the
innermost weight level's `reads` is the MAC count and is, but the per-cycle
`leak` row is `idle x (1 - g)` and Timeloop bills it as
`leak x utilized instances x cycles` -- both the mapper's choice. So R5a IS
ERT-injectable at `ECC_RECON_CLOCK_GATING_PCT` 0 and 99.5, and is not at 100.

An arm with a bump keeps prompt_6's `ert-<key>-<level>-<action>` cache slug byte
for byte; one without gets `arm-<key>`, which is what stops R1 (whose patched
YAML IS the reference's until Phase C1.2) sharing the reference's directory.

EVERY BAR NAMES THE PLAN IT WAS BILLED FROM. `plan_assignment()` derives it: its
own arm if that arm is mapped, else the OUTERMOST mapped arm in path order that
narrows exactly the same storage levels, else the reference plan flagged
`geometry_matches: false`. A plan is only valid for a bar when the narrowed
levels match -- `datawidth: q` changes the words the loop nest moves and no
post-processing can re-tile a loop nest. A HALF-mapped arm is refused, not
borrowed: that would be two chips in one bar.

`validate_placement_space()` lives here too: it requires a placement's `reduced`
set to be a PREFIX of the path's reducible stages in path order, and every
reducible stage to be reached by some boundary. Without it, adding a stage to
`weight_path.py` and forgetting `placements.py` keeps every boundary below it
reporting its own saving while the new stage stays at full width -- the whole
list understated, with nothing saying so.

ProjectRestructure phase 3 cut this out of `recon.py` and filed it at L2 rather
than in `study/`: `config.py` resolves `ECC_RECON_ERT_ARM` through it and
`toolchain/ert_probe.py` reads it, and both of those sit below any study driver.
Its ONE remaining upward import -- the DC reconstruction table, for deciding
whether a bump has a per-cycle row at all -- is declared in
`tests/contract/test_layer_rule.py` and is phase 4's to remove.
"""
from __future__ import annotations

from ..physics import widths
from dataclasses import dataclass, replace

from .placements import decode_site, encoder_site, placement_by_key, placements_for, stages_for


# ===========================================================================
#  prompt_6 -- which placements get their own mapping (ERT arms)
# ===========================================================================
#: The ERT access action a placement's `site_counter` names. A weight arriving
#: at a level is a `write` (Timeloop counts it as a fill), a weight leaving it
#: is a `read`. Network counters name no component action: a delivery is
#: billed through `noc.yaml` (prompt_6 3.3).
ERT_ACTION_OF_COUNTER = {"reads": "read", "fills": "write"}


def ert_leak_delta_pj(cfg):
    """The per-cycle `leak` row of an ERT bump under this configuration:
    `idle_per_cycle x (1 - g)`, prompt_7 Phase 0's clock-gating split.

    It is a SEPARATE function because `ert_injectable()` condition 3 turns on
    whether that row exists (prompt_7 6.3) and because reading it costs a
    pandas import that a caller with no configuration should not pay. With no
    configuration it is 0.0 -- "this bump has no per-cycle row", which is the
    pre-clock-gating model and the honest answer when there is no gating
    percentage and no DC table to derive one from. A configuration that cannot
    be read RAISES rather than answering 0.0: a swallowed exception here would
    change the arm list, silently, on a machine without pandas.
    """
    if cfg is None:
        return 0.0
    g = max(0.0, min(1.0, float(getattr(cfg, "recon_clock_gating_pct", 0.0)) / 100.0))
    if g >= 1.0:
        return 0.0                 # fully gated: the row is identically zero
    from ..study import stacks as _ecc  # lazily: ecc needs pandas
    _inc, idle, _prov = _ecc.load_recon_energy(cfg)
    return float(idle) * (1.0 - g)


def ert_injectable(placement, stages, leak_delta_pj=0.0):
    """prompt_6 3.3: is this boundary's encoder cost an ERT action the mapper
    can trade against? `(ok, why)`, DERIVED from the Placement record -- never
    `if key == "recon2"`.

    Three conditions, all required:

    1. the site stage is a STORAGE stage. DRAM is not a chip action the
       encoder attaches to, and a network delivery is billed by `noc.yaml`.
    2. the site counter is `reads` or `fills`, naming exactly one ERT action.
    3. THE BUMP MUST NOT BE CONSTANT ACROSS THE MAPSPACE. A bump the mapper
       cannot trade cannot move the argmax, so injecting it buys nothing.

    CONDITION 3, RE-DERIVED 2026-09-12 (prompt_7 6.3). It used to read "it is
    not the innermost weight level's `reads`", on the grounds that that count
    equals the MAC count, which no mapping can move. That is still true of the
    ACCESS row -- and the bump has TWO rows:

        per access :  incremental + idle x g       on `reads`/`fills`
        per cycle  :  idle x (1 - g)               on `leak`, which Timeloop
                                                   bills as leak x UTILIZED
                                                   instances x CYCLES

    The per-cycle row is mapping-dependent whenever it is non-zero: both the
    utilized instance count and the cycle count are the mapper's choice. So
    the exclusion holds only when BOTH rows are constant, i.e. when the access
    counter is the innermost level's reads AND `leak_delta_pj` is 0. At
    `ECC_RECON_CLOCK_GATING_PCT=99.5` the row is 2.8311 x 0.005 = 0.01416
    pJ/cycle/instance, which over 168 scratchpads is 2.38 pJ/cycle against the
    MAC array's own 1.32 -- not negligible and not constant. At PCT=100 (or a
    zero DC idle constant) it vanishes and the old exclusion comes back.

    `leak_delta_pj` is that row, from `ert_leak_delta_pj(cfg)`. It defaults to
    0.0 -- "this bump has no per-cycle row" -- so a caller with no
    configuration gets the pre-clock-gating answer and says so in `why`.
    """
    by_key = {s.key: s for s in stages}
    site = by_key.get(placement.site_stage)
    if site is None:
        return False, f"site stage {placement.site_stage!r} is not on the weight path"
    if site.kind != "storage":
        return False, (f"{site.key} is a {site.kind} stage, not a chip storage "
                       f"action" + (" (the count lives in noc.yaml)"
                                     if site.kind == "network" else ""))
    action = ERT_ACTION_OF_COUNTER.get(placement.site_counter)
    if action is None:
        return False, f"counter {placement.site_counter!r} names no ERT action"
    storage = [s for s in stages if s.kind == "storage"]
    leak = abs(float(leak_delta_pj or 0.0))
    if site is storage[-1] and placement.site_counter == "reads":
        if leak == 0.0:
            return False, (f"{site.key} is the innermost weight level and its reads "
                           f"equal the MAC count, which no mapping can move, and the "
                           f"per-cycle leak row is 0 pJ, so the whole bump is constant")
        return True, (f"{site.key} + {placement.site_counter} -> ERT {action}: the "
                      f"access row is the MAC count and constant, but the per-cycle "
                      f"leak row is {leak:.7g} pJ/cycle/instance, which Timeloop bills "
                      f"as leak x utilized instances x cycles -- both the mapper's "
                      f"choice (prompt_7 6.3)")
    return True, f"{site.key} + {placement.site_counter} -> ERT {action}"


def ert_arms(arch, cfg=None):
    """The placements of `arch` whose encoder toll goes INTO THE ERT, in path
    order -- `ert_injectable()`'s set, and nothing else.

    This answers "which boundaries have an ERT-injectable encoder". It is NOT
    the list of chips to map: that is `mapper_arms()` (prompt_7 B2), which is
    this set widened by the two axes an ERT bump does not capture.

    Zero is legal (the figure is then Task 3's with the idle term added); the
    code must work at 0, 1 or 2+. On Eyeriss v1 with the filter GLB and the
    clock-gating row present this is `recon2` (filter_glb reads), `recon4`
    (weights_spad fills) and `recon5` (weights_spad reads, condition 3
    re-derived); with no configuration to read the gating from, the first two.
    """
    stages = stages_for(arch, cfg)
    leak = ert_leak_delta_pj(cfg)
    return tuple(p for p in placements_for(arch, cfg)
                 if ert_injectable(p, stages, leak)[0])


#: Which stage kinds Timeloop can be given a bandwidth limit for, and which
#: it cannot. A `per_dataspace_bandwidth_consumption_scale` on a storage or
#: DRAM level reaches `ComputePerformance()`; on a NETWORK level there is
#: nothing to reach -- `LegacyNetwork::ComputePerformance()` is an empty stub
#: and a network stats block carries no Cycles and no bandwidth field
#: (prompt_7 4.4, verified on disk; reporting rule R-3). A network stage is
#: therefore carried in an arm's declared scale set and MARKED `no-op`, which
#: is what makes R2 and R1 different arms on a design whose only difference is
#: a network (Eyeriss v2), rather than silently the same one.
BW_SCALE_TIMING = {"dram": "timed", "storage": "timed", "network": "no-op"}


def arm_narrow_levels(placement, stages):
    """The Timeloop storage levels this boundary declares `datawidth: q` on --
    the storage stages in its `reduced` set, in path order.

    The narrow weights stop AT the boundary, so for recon2 AND recon4 on
    Eyeriss v1 this is (`filter_glb`,) and `weights_spad` stays at 8; recon5
    is the only bar that adds `weights_spad` (prompt_7 3.2).
    """
    by_key = {s.key: s for s in stages}
    return tuple(by_key[k].prefixes[0] for k in placement.reduced
                 if k in by_key and by_key[k].kind == "storage")


def arm_bw_scale(placement, stages):
    """The per-dataspace bandwidth scale this boundary declares, as
    `((level, timing), ...)` in path order -- prompt_7 6.4's third axis.

    Every stage in the `reduced` set moves less weight data, so every stage in
    it carries the scale. `timing` is `timed` where Timeloop has a speed model
    to apply it to and `no-op` where it has none (`BW_SCALE_TIMING`). The
    no-op entries are kept, not dropped: they are part of what the arm
    DECLARES, and dropping them would merge two boundaries that differ only by
    a network into one chip.
    """
    by_key = {s.key: s for s in stages}
    return tuple((by_key[k].prefixes[0], BW_SCALE_TIMING[by_key[k].kind])
                 for k in placement.reduced if k in by_key)


def arm_bw_factors(placement, stages, cfg):
    """The bandwidth scale each stage of this boundary DECLARES, with its
    factor -- prompt_7 C1.2.

    Returns `{level: {"factor", "timing", "kind", "dataspace", "why"}}` in path
    order, for the stages `arm_bw_scale()` names.

    TWO FACTORS, AND THEY ARE NOT THE SAME NUMBER (prompt_7 7.1). The scale
    has to match `ECC_RECON_PACKING`, or the mapper and the evaluator disagree
    about the same wire:

        DRAM      a BIT stream off the die: only the k message bits are read
                  out, so the demand is  K/N          (0.47619 at BCH(63,30))
        on chip   whole weights in a narrower word: `datawidth: q` bits each,
                  so the demand is       q/8          (0.5     at q=4)

    They differ by 5% at BCH(63,30) and using one everywhere is a silent
    inconsistency, not a rounding choice.

    A NETWORK stage gets its factor computed and is marked `no-op`:
    `LegacyNetwork::ComputePerformance()` is an empty stub, so nothing reads
    it. It is still DECLARED -- dropping it would merge two boundaries that
    differ only by a network into one chip (`BW_SCALE_TIMING`).
    """
    from ..physics import widths
    by_key = {s.key: s for s in stages}
    q = widths.declared_datawidth(cfg.code_n, cfg.code_k)
    bits = cfg.weight_bits
    out = {}
    for level, timing in arm_bw_scale(placement, stages):
        kind = next(by_key[k].kind for k in placement.reduced
                    if k in by_key and by_key[k].prefixes[0] == level)
        if kind == "dram":
            factor, why = (cfg.code_k / cfg.code_n,
                           f"K/N = {cfg.code_k}/{cfg.code_n}: off the die the "
                           f"weights are a BIT stream and only the message bits "
                           f"are driven")
        else:
            factor, why = (q / bits,
                           f"q/{bits} = {q}/{bits}: on chip the level holds WHOLE "
                           f"weights, {q} bits each, in a word of the same width")
        out[level] = {"factor": factor, "timing": timing, "kind": kind,
                      "dataspace": REDUCED_BW_DATASPACE, "why": why}
    return out


#: The one dataspace a reconstruction boundary moves less of. Timeloop
#: validates the name against the problem's dimensions and exits non-zero on a
#: misspelling (`Weightz:` -> "is not a valid dimension name", prompt_7 A.2),
#: so this cannot degrade into a silent no-op.
REDUCED_BW_DATASPACE = "Weights"


def mapper_arm_spec(arch, key, cfg=None):
    """What ONE MAPPER ARM declares -- for ANY arm, ERT-injectable or not.

    `key` is `reference` (or the empty string) or a placement key. Returns a
    dict:

        placement       the Placement record, or None for the reference
        key             the arm's name
        narrow_levels   `arm_narrow_levels()` -- the levels at `datawidth: q`
        bw_scale        `arm_bw_scale()` -- the declared bandwidth scale
        ert             the ERT half (`level`, `counter`, `action`) when this
                        boundary is ERT-injectable, else None
        ert_why         the derived reason, injectable or not

    `ert_arm_spec()` is this plus a refusal when `ert` is None; it keeps its
    pre-2026-09-12 meaning, so `archs.ert_bump()` and the `ert-` cache slug are
    untouched for the arms that have a bump.

    Raises `KeyError` for an unknown key.
    """
    if key in (None, "", "reference"):
        return {"placement": None, "key": "reference", "narrow_levels": (),
                "bw_scale": (), "ert": None,
                "ert_why": "the reference arm declares no encoder"}
    p = placement_by_key(arch, key, cfg)
    if p is None:
        raise KeyError(f"{arch} has no placement {key!r}; it has "
                       f"{', '.join(q.key for q in placements_for(arch, cfg))}")
    stages = stages_for(arch, cfg)
    ok, why = ert_injectable(p, stages, ert_leak_delta_pj(cfg))
    by_key = {s.key: s for s in stages}
    ert = None
    if ok:
        site = by_key[p.site_stage]
        ert = {"level": site.prefixes[0], "counter": p.site_counter,
               "action": ERT_ACTION_OF_COUNTER[p.site_counter]}
    return {"placement": p, "key": p.key,
            "narrow_levels": arm_narrow_levels(p, stages),
            "bw_scale": arm_bw_scale(p, stages),
            "ert": ert, "ert_why": why}


def ert_arm_spec(arch, key, cfg=None):
    """What ONE ERT arm declares, from its Placement (prompt_6 5.2, RULE 2).

    Returns a dict:

        placement       the Placement record
        level           the Timeloop level the encoder's action is on (the
                        site stage's first prefix, e.g. `filter_glb`)
        counter         `reads` | `fills` -- the count the split must use
        action          `read` | `write`  -- the ERT row that is bumped
        narrow_levels   the storage levels in `placement.reduced`, i.e. the
                        levels that get `datawidth: q`. The narrow weights stop
                        AT the boundary, so for recon2 AND recon4 on Eyeriss v1
                        this is (`filter_glb`,) and `weights_spad` stays at 8.

    Raises `KeyError` for an unknown key and `ValueError` for a placement that
    is not ERT-injectable, naming the reason. A mapper arm that is NOT
    ERT-injectable is still a chip to map -- ask `mapper_arm_spec()` for it.
    """
    spec = mapper_arm_spec(arch, key, cfg)
    if spec["placement"] is None:
        raise ValueError(f"{arch}/reference is not an ERT arm: {spec['ert_why']}")
    if spec["ert"] is None:
        raise ValueError(f"{arch}/{key} is not an ERT arm: {spec['ert_why']}. The ERT arms of "
                         f"{arch} are {', '.join(a.key for a in ert_arms(arch, cfg)) or 'none'}")
    return {"placement": spec["placement"], "key": spec["key"],
            "level": spec["ert"]["level"], "counter": spec["ert"]["counter"],
            "action": spec["ert"]["action"],
            "narrow_levels": spec["narrow_levels"]}


# ---------------------------------------------- prompt_7 B2: the MAPPER arms
@dataclass(frozen=True)
class MapperArm:
    """ONE CHIP the mapper has to solve, and the boundaries it is the chip of.

    prompt_7 6.4. `ert_arms()` answers "which boundaries have an ERT-injectable
    encoder"; that is not the question a launcher has to answer. Two boundaries
    the ERT cannot tell apart can still be different architectures to the
    mapper, and mapping one and billing the other from it is Defect 3 -- R3
    borrowed the REFERENCE's plan while its geometry was R2's, and R5a is a
    chip that has never been mapped at all.

    THE ARM IS THE TUPLE OF THREE AXES, and `distinct_key` is exactly that
    tuple:

        narrow_levels   `datawidth: q` on these storage levels
        bw_scale        `per_dataspace_bandwidth_consumption_scale` here
        ert             the ERT bump, as (level, action), or ()

    `members` are the placement keys this arm is the chip of; it is longer than
    one only when two boundaries agree on all three axes, in which case they
    ARE one chip and one mapper job answers both.
    """
    key: str                  # `reference`, or the representative placement key
    placement: object         # the Placement record, None for the reference
    narrow_levels: tuple
    bw_scale: tuple
    ert: tuple                # (level, action) or ()
    members: tuple

    @property
    def distinct_key(self):
        return (self.narrow_levels, self.bw_scale, self.ert)

    @property
    def slug_part(self):
        """The cache-slug component that keeps this arm's directory its own.

        An arm WITH an ERT bump keeps prompt_6's `ert-<key>-<level>-<action>`
        spelling, byte for byte, so every directory already on disk stays a
        cache hit. An arm without one -- R1, R3 -- gets `arm-<key>`, because
        without it R1's slug would be the REFERENCE's (its YAML is the
        reference's until Phase C1.2 declares the bandwidth scale) and two
        arms would share one directory, which is the failure prompt_6 RULE
        4.4.5 exists to prevent. The reference arm contributes nothing.
        """
        if self.placement is None:
            return None
        if self.ert:
            return f"ert-{self.key}-{self.ert[0]}-{self.ert[1]}"
        return f"arm-{self.key}"

    def describe(self):
        return (f"{self.key:<10} datawidth q on "
                f"{'+'.join(self.narrow_levels) or '-':<26} "
                f"bw scale {','.join(l + ('' if t == 'timed' else '(no-op)') for l, t in self.bw_scale) or '-':<58} "
                f"ERT {(self.ert[0] + '.' + self.ert[1]) if self.ert else '-'}")


def mapper_arms(arch, cfg=None):
    """THE CHIPS TO MAP: the reference plus every distinct boundary, in path
    order (prompt_7 B2).

    Distinctness is DERIVED from the three axes of `MapperArm.distinct_key` --
    never from a key name and never from a hand-written list. Two boundaries
    that agree on all three are one chip and share one arm; the second is then
    a `member` of the first and is billed from its plan, which is a borrow
    between two chips that are the SAME chip.

    On `eyeriss_like_wglb` this is six: the reference, R1 (the DRAM scale
    alone), R2 (filter_glb narrowed, ERT on its reads), R3 (the same geometry,
    NO ERT bump -- which is why it is not R2's arm), R4 (the same geometry,
    ERT on the scratchpad's fills) and R5a (the scratchpad narrowed too).
    Eyeriss v2 has four boundaries and therefore five.
    """
    arms, by_key = [], {}
    for p in (None,) + placements_for(arch, cfg):
        spec = mapper_arm_spec(arch, p.key if p is not None else "reference", cfg)
        ert = ((spec["ert"]["level"], spec["ert"]["action"]) if spec["ert"] else ())
        arm = MapperArm(key=spec["key"], placement=spec["placement"],
                        narrow_levels=spec["narrow_levels"],
                        bw_scale=spec["bw_scale"], ert=ert,
                        members=((spec["key"],) if p is not None else ()))
        seen = by_key.get(arm.distinct_key)
        if seen is None:
            by_key[arm.distinct_key] = len(arms)
            arms.append(arm)
        else:
            prev = arms[by_key[arm.distinct_key]]
            arms[by_key[arm.distinct_key]] = replace(
                prev, members=prev.members + arm.members)
    return tuple(arms)


def mapper_arm_for(arch, key, cfg=None):
    """The `MapperArm` a placement key is billed from when its own plan exists
    -- i.e. the arm whose `members` contain it."""
    for a in mapper_arms(arch, cfg):
        if key in a.members or (key in ("", "reference", None) and a.placement is None):
            return a
    raise KeyError(f"{arch} has no mapper arm for {key!r}")


def arm_slugs(arch, cfg=None):
    """`{arm key: slug part}` -- the check that no two arms share a cache
    directory (prompt_6 RULE 4.4.5 defence 1, prompt_7 B2 gate 2)."""
    return {a.key: a.slug_part for a in mapper_arms(arch, cfg)}


def plan_assignment(arch, solved, cfg=None):
    """WHICH ARM'S PLAN EACH BAR IS BILLED FROM -- prompt_6 RULE 4, prompt_7 B1.

    `solved` is the set of arm keys whose OWN mapping is on disk at their own
    fingerprint; `reference` is always in it (it is the run's own plan).
    Returns `{placement key: record}` with

        arm                 the arm whose plan this bar is billed from
        kind                `own` | `borrowed` | `foreign`
        geometry_matches    does that plan narrow exactly the levels this bar
                            narrows? A plan is only VALID for a bar when it
                            does: `datawidth: q` is what changes the words the
                            loop nest moves, and no post-processing can re-tile
                            a loop nest to use capacity it does not know about.
        candidates          every solved arm with this bar's geometry
        why                 the sentence that goes on the bar's record

    THE RULE, in three lines, and derived -- never `if key == "recon3"`:

      1. the bar's own arm is solved            -> `own`
      2. else the FIRST solved arm in path order with the SAME
         `narrow_levels`                        -> `borrowed`
      3. else the reference plan                -> `foreign`, flagged

    On `eyeriss_like_wglb` with only `recon2` and `recon4` mapped, that is
    exactly prompt_7 6.2: R1 takes the reference (its geometry IS the
    reference's -- nothing on chip is narrowed -- which is why `reference` is
    only ever right for R1), R3 takes R2's (same `{filter_glb}` geometry, and
    R2 is the outer of the two candidates), and R5a has no valid plan on disk
    at all because no solved arm narrows `weights_spad`. Rule 3 is the one
    that must be loud: it is 6.2(b), "a chip that has never been mapped".

    A BORROWED PLAN IS NOT A FREE LUNCH. It was solved with the lending arm's
    ERT toll in the objective and this bar does not pay that toll; what makes
    it usable is that the toll changes the argmax, not the meaning of a plan,
    while `datawidth: q` changes both. The lender's toll is MOVED OUT of the
    bill before the borrower is billed from it, so the borrower pays the
    un-bumped price of the shared geometry.
    """
    arms = mapper_arms(arch, cfg)
    solved = set(solved) | {"reference"}
    by_key = {a.key: a for a in arms}
    order = {a.key: i for i, a in enumerate(arms)}
    out = {}
    for arm in arms:
        if arm.placement is None:
            continue
        for key in arm.members:
            cands = [a.key for a in arms
                     if a.key in solved and a.narrow_levels == arm.narrow_levels]
            cands.sort(key=lambda k: order[k])
            if arm.key in solved:
                out[key] = {"arm": arm.key, "kind": "own", "geometry_matches": True,
                            "candidates": cands,
                            "why": (f"billed from {arm.key}'s OWN mapping: "
                                    f"{_geometry_phrase(arm)}")}
            elif cands:
                lender = by_key[cands[0]]
                if lender.placement is None:
                    why = (f"billed from the REFERENCE plan, and for this bar that is the "
                           f"right plan: it narrows nothing on chip, so its storage "
                           f"geometry IS the reference's. What separates the two chips is "
                           f"the declared off-chip bandwidth scale, which changes the "
                           f"throttling check and not one picojoule of energy "
                           f"(prompt_7 7.1, measured bit-identical).")
                else:
                    why = (f"billed from {lender.key}'s mapping: this chip has no mapping "
                           f"of its own yet, and {lender.key} is the outermost solved arm "
                           f"with the SAME storage geometry ({_geometry_phrase(arm)}). What "
                           f"separates the two is the ERT bump and the declared bandwidth "
                           f"scale, neither of which changes what a loop nest means"
                           + (f"; the other candidate was {', '.join(cands[1:])}"
                              if len(cands) > 1 else "") + ".")
                out[key] = {"arm": lender.key, "kind": "borrowed",
                            "geometry_matches": True, "candidates": cands, "why": why}
            else:
                out[key] = {"arm": "reference", "kind": "foreign",
                            "geometry_matches": False, "candidates": [],
                            "why": (f"billed from the REFERENCE plan, whose geometry is NOT "
                                    f"this bar's: it narrows {_geometry_phrase(arm)} and no "
                                    f"solved arm does. This chip has never been mapped "
                                    f"(prompt_7 6.2b); its own mapping is Phase C. No "
                                    f"post-processing can re-tile a loop nest, so the plan "
                                    f"cannot use capacity it does not know exists.")}
    return out


def _geometry_phrase(arm):
    return (f"datawidth q on {'+'.join(arm.narrow_levels)}" if arm.narrow_levels
            else "nothing on chip narrowed")


def ert_deltas(incremental_pj, idle_pj, gran, block_size, clock_gating_pct=0.0):
    """prompt_6 5.1 -- the two ERT deltas of one arm, from the DC numbers.

        access action   delta = E_w x block_size     E_w = incremental / weights per codeword
        leak action     delta = idle_per_cycle       Timeloop multiplies by instances x cycles

    `block_size = width / datawidth` is read off THAT level in THAT arm's
    patched arch by the caller; `gran.codewords(1.0)` is the codeword events
    one weight causes under the configured charging mode, so E_w is
    `incremental / weights_per_codeword` under `weight` charging and the
    whole incremental under `codeword` charging -- the same rule
    `evaluate_placement` charges by, so the split in 5.3 reconciles.
    Nothing here is hardcoded: at BCH(63,30) E_w = 1.3786 / 7.875 = 0.175060.
    """
    if block_size is None or int(block_size) < 1:
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}")
    # prompt_7 Issue 15: THE MAPPER AND THE EVALUATOR MUST PRICE THE SAME ENGINE.
    # Under clock gating the engine burns `active` while it works and
    # `idle x (1-g)` while it is gated off. Timeloop can only express that as
    # (per-access toll) + (per-cycle leak), and the split is exact:
    #
    #   per access : incremental + idle x g     (active MINUS the leak already
    #                                            charged for that same cycle)
    #   per cycle  : idle x (1 - g)
    #
    # which expands to EXACTLY what evaluate_placement charges --
    #   (inc + idle*g)*events + idle*(1-g)*engine_cycles
    #   == (inc + idle)*events + idle*(1-g)*(engine_cycles - events)
    # -- so prompt_6 RULE 5.3's "a split, not an addition" is restored and the
    # gating credit is zero. At g = 0 both rows are prompt_6 Table 5.1 exactly.
    g = max(0.0, min(1.0, float(clock_gating_pct) / 100.0))
    inc_eff = float(incremental_pj) + float(idle_pj) * g
    idle_eff = float(idle_pj) * (1.0 - g)
    e_w = inc_eff * gran.codewords(1.0)
    return {"e_w_pj": e_w,
            "access_delta_pj": e_w * int(block_size),
            "leak_delta_pj": idle_eff,
            "clock_gating_pct": g * 100.0,
            "incremental_pj_per_codeword_gated": inc_eff,
            "idle_pj_per_cycle_gated": idle_eff,
            "block_size": int(block_size),
            "incremental_pj_per_codeword": float(incremental_pj),
            "idle_pj_per_cycle": float(idle_pj),
            "weights_per_codeword": gran.weights_per_codeword,
            "charging": gran.mode}




def validate_placement_space(arch, cfg=None):
    """Does this design's `PLACEMENTS` cover its `WEIGHT_PATHS`? `(ok, detail)`.

    The reduced representation flows from the ECC engine DOWNWARD: a boundary
    at stage S means every reducible stage between the correction and S carries
    the reduced form, and nothing below S does. Two consequences, and both are
    checked here rather than assumed:

    * a placement's `reduced` set must be a PREFIX of the path's reducible
      stages, in path order. `{mesh}` is a boundary; `{mesh}` on a design whose
      weight path puts a weight GLB ABOVE the mesh is not -- it would have the
      mesh carrying the reduced form while the buffer feeding it carries full
      width, which is a different (and unstated) pair of boundaries.
    * across the whole list, every reducible stage must be reached by at least
      one boundary, or the design has a stage the study never asks about.

    WHY THIS IS A CHECK AND NOT A COMMENT. `WEIGHT_PATHS` and `PLACEMENTS` are
    two tables that have to be edited together (CLAUDE.md, "Adding an
    architecture to the PLACEMENT study"), and extending one without the other
    fails silently in the worst way: every boundary below the new stage keeps
    its own saving and reports the new stage at FULL width, so the study
    understates every one of them and nothing in the result file says so.
    `feasibility()` cannot catch it -- it only looks at the stages a placement
    DOES claim, never at the ones it should have claimed and did not.

    `eyeriss_v2_like_wglb` is exactly this case today: its weight path gained a
    `weight_glb` stage, its placement list did not, and it has no mapper cache,
    so the gap has never been exercised. It is reported here instead of being
    evaluated with four understated boundaries.
    """
    stages = stages_for(arch, cfg)
    by_key = {s.key: s for s in stages}
    reducible = [s.key for s in stages if s.reducible]
    order = {key: i for i, key in enumerate(reducible)}
    rows, violations = [], []
    reached = set()

    # WHICH COUNTERS A STAGE CAN EVEN HAVE. `ingresses` and `deliveries` are
    # network quantities and `StageStats` leaves them 0.0 on a storage or DRAM
    # stage, so a counter on the wrong kind of stage does not raise -- it prices
    # the boundary's reconstruction at ZERO and makes it look like the cheapest
    # placement in the study. Checked here, before anything is evaluated,
    # because that is a table error and there is no number it could produce.
    _COUNTERS = {"reads": ("dram", "storage"), "fills": ("storage",),
                 "ingresses": ("network",), "deliveries": ("network",),
                 }
    for pl in placements_for(arch, cfg):
        want = _COUNTERS.get(pl.site_counter)
        st = by_key.get(pl.site_stage)
        if want is None:
            violations.append(
                f"{pl.key} counts reconstructions with {pl.site_counter!r}, "
                f"which is not one of {sorted(_COUNTERS)}")
        elif st is None:
            violations.append(
                f"{pl.key} reconstructs at stage {pl.site_stage!r}, which is not "
                f"a stage of {arch}'s weight path")
        elif st.kind not in want:
            violations.append(
                f"{pl.key} counts reconstructions with {pl.site_counter!r} at "
                f"{pl.site_stage!r}, a {st.kind} stage; that counter only exists "
                f"on a {' or '.join(want)} stage and would be zero here, "
                f"pricing the boundary's reconstruction at nothing")

    for p in placements_for(arch, cfg):
        got = list(p.reduced)
        reached |= set(got)
        unknown = [k for k in got if k not in order]
        want = reducible[:len(got)]
        prefix = not unknown and got == want
        rows.append({"placement": p.key, "boundary": p.label,
                     "reduced_stages": got,
                     "expected_prefix_of_the_weight_path": want,
                     "is_a_prefix": prefix})
        if unknown:
            violations.append(
                f"{p.key} reduces stage(s) {unknown} that are not reducible "
                f"stages of {arch}'s weight path")
        elif not prefix:
            violations.append(
                f"{p.key} reduces {got} but the reducible stages between the "
                f"ECC engine and that boundary are {want}: the reduced form "
                f"cannot skip a stage on its way down the path")
    never = [k for k in reducible if k not in reached]
    if never:
        violations.append(
            f"reducible stage(s) {never} of {arch}'s weight path are reduced by "
            f"no placement at all, so every boundary below them silently "
            f"reports them at full width")
    return not violations, {
        "decode_site": decode_site(cfg),
        "encoder_site": encoder_site(cfg),
        "reconstruction_counter_per_placement": {
            pl.key: {"stage": pl.site_stage,
                     "stage_kind": (by_key[pl.site_stage].kind
                                    if pl.site_stage in by_key else None),
                     "counter": pl.site_counter}
            for pl in placements_for(arch, cfg)},
        "reducible_stages_in_path_order": reducible,
        "per_placement": rows,
        "reducible_stages_no_placement_reaches": never,
        "violations": violations,
        "rule": ("the reduced representation is contiguous from the ECC engine "
                 "down to the boundary, so a placement's reduced set is a "
                 "PREFIX of the reducible stages in path order; and every "
                 "reducible stage must be reached by some boundary, or "
                 "WEIGHT_PATHS and PLACEMENTS have drifted apart"),
        "fix": ("edit WEIGHT_PATHS[<arch>] and PLACEMENTS[<arch>] together -- "
                "a stage without a boundary that reduces it is a stage the "
                "study never asks about (CLAUDE.md, 'Adding an architecture to "
                "the PLACEMENT study')"),
    }


