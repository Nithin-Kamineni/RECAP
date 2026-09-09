"""Task 3: reconstruction PLACEMENT on one architecture's weight path.

    03_staged_implementation_plan.txt, task 3:

        "Using Task 2 as the embedded-ECC reference, fix architecture=Eyeriss_v2,
         model/layers, BCH configuration, precision, physical hardware
         assumptions and mapping. Vary only reconstruction placement and its
         necessary overhead. Do not modify or rerun the mapping optimizer for
         this comparison."

WHAT THIS MODULE IS, AND WHAT IT IS NOT
---------------------------------------
`ecc.build_stacks()`'s `recon` arm is ONE placement, applied to every
architecture at once: it scales the weight share of every on-chip category by
K/N and charges one reconstruction per DRAM codeword. That is a single point in
the space this module explores, and it is not one of the five points Task 3
names -- it reduces the SPad *and* charges reconstruction only once per DRAM
fetch, which no physical boundary does.

This module is the placement study proper. It works on ONE architecture at a
time, because each architecture has its own weight path and therefore its own
list of feasible boundaries. Nothing here is used by the three sweeps, and
`build_stacks()` is untouched.

THE WEIGHT PATH IS READ FROM THE MODEL, NOT ASSUMED
---------------------------------------------------
`01_project_context_and_architectures.txt` Sec. 5.1 warns against substituting

    DRAM --> global weight SRAM --> RF --> MAC

for Eyeriss v2, whose GLB banks serve input activations and partial sums while
weights go from the memory interface through the hierarchical mesh to the PE
scratchpads. `WEIGHT_PATHS` below therefore names the Timeloop levels of the
architecture AS MODELLED, and `weight_path()` refuses to proceed if a level
that carries Weights energy is not claimed by exactly one stage. A placement
whose stage is absent from the model is reported `unsupported`, never silently
skipped -- for `eyeriss_v2_like` there is no weight GLB at all, so no
"reconstruct at the global buffer" boundary exists to evaluate.

The one level that is TWO stages is the DRAM. Accelergy's CactiDRAM bills a
DRAM read as one flat per-bit constant that contains both the array read and
the off-die transfer, and Timeloop charges the DRAM-to-chip network zero, so
the evaluator splits that constant itself with `f_if`, the interface share of
the per-bit DRAM energy (01_project_context Sec. 4; 02_..., Sec. 19 and risk
6):

    dram_array      (1 - f_if) x DRAM weight energy    never reduced
    dram_interface       f_if  x DRAM weight energy    x K/N under EVERY boundary

The two stages both match Timeloop's `DRAM` level and `read_weight_path()`
requires their shares to sum to one, so the level is still claimed exactly
once in total and `cross_check()` still reconciles their sum against the `Raw`
record's DRAM category. `f_if` is `ECC_DRAM_IF_FRAC` and has no default: it is
a cited DRAM energy breakdown, not a modelling choice, and with it unset the
study refuses to evaluate and prints the DRAM ceiling at several values
instead (experiments/recon.py). `archs/_shared/provenance.yaml`
(`dram_interface_share`) records what has been found so far.

THE DECODER IS ON THE DRAM DIE  (since 2026-09-09)
-------------------------------------------------
01_project_context Sec. 1: the BCH decoder sits on the DRAM die and OFF the
fetch path (it corrects at write, on a scrub pass or on a prior access), so at
fetch time the stored codeword is already corrected and only the k message
bits of each n-bit codeword are driven across the DRAM interface. The array
still stores and reads the complete codeword -- a row activation and a burst
move whole words -- so only `dram_interface` is reducible, and it is reduced
under every boundary, R1 included. `ECC_RECON_DECODE_SITE=controller` is the
pre-2026-09-09 model (decode on the fetch path, the complete codeword across
the interface, DRAM identical on every bar), kept as a runnable row so the
change can be diffed; `stages_for()` and `placements_for()` take the config
and hand back that space when it is asked for. The two reference bars keep
controller-side correction under both settings: Task 1's external parity is
read from the array AND crosses the interface, and Task 2's on-chip datapath
consumes every weight bit, so its complete codeword crosses too. Their
functions are frozen and do not change.

WHERE THE NUMBERS COME FROM
---------------------------
Everything is re-derived from the SAME cached Timeloop output the baseline and
embedded arms use -- `timeloop-mapper.stats.txt` and `timeloop-mapper.map.txt`
in the mapper cache. Nothing here can change a mapping: the mapper is never
invoked (`ECC_FROM_CACHE=1`), and the per-level energies this module scales are
cross-checked against the `Raw` record's own category totals before any
placement is evaluated. Four quantities the aggregated `Raw` record does not
carry are parsed here:

  total access counts   Timeloop prints scalar accesses PER INSTANCE. The
                        reconstruction count is a total, so per-instance counts
                        are multiplied by `Utilized instances (max)`.
  utilized capacity     how many weights are resident in a PE scratchpad. This
                        is the feasibility test for a PE-local boundary.
  physical packing      `Block size` x `Word bits` -- the physical word a
                        scalar access is a fraction of. Sec. 16 of
                        02_reconstruction_dse_and_implementation.txt: storing
                        4-bit values in 8-bit fields halves nothing.
  the loop nest         the temporal loops BELOW the innermost weight buffer,
                        which is what decides whether a reconstructed-weight
                        reuse register ever gets a hit.

THE FIVE BOUNDARIES
-------------------
Sec. 5.2/5.3 of 01_project_context_and_architectures.txt, in the order the plan
lists them. Ratings are the source discussion's HYPOTHESES, carried so the
result can be read against them; they are not results.

    R1  recon1  reconstruct at the source / weight-NoC ingress          [2/5]
                (reduces the DRAM interface and nothing on chip: it isolates
                the interface saving every boundary shares from any on-chip
                saving, and pays one reconstruction per DRAM codeword)
    R2  recon2  reconstruct at the destination-cluster boundary         [4/5]
    R3  recon3  reconstruct at the PE weight-SPad input                 [4/5]
    R4a recon4  reconstruct on every weight-SPad read                   [3/5]
    R4b recon5  SPad output plus a reconstructed-weight reuse register  [5/5]

RECONSTRUCTION GRANULARITY IS A CONSTRAINT, NOT A DETAIL
--------------------------------------------------------
Sec. 15 defines `G_rec`: the number of weights whose retained bits must be
available at once to rebuild an ECC group. Under the embedded layout the
codeword IS n consecutive bits of the weight bit stream (`embedded.py`), so one
codeword spans `n / weight_bits` weights and, because it is not weight-aligned,
touches up to `ceil(n / weight_bits) + 1` of them -- 9 at BCH(63, K) over 8-bit
weights. A PE-local boundary is FEASIBLE only if that many weights are resident
in the PE at once. `feasibility()` computes it per layer from the utilized
capacity, and a placement that fails it anywhere is reported `unsupported` with
the layers named. It is not evaluated with a plausible number.

WHAT IS DELIBERATELY NOT CREDITED
---------------------------------
* The DRAM ARRAY is not credited. The array stores and reads the complete
  codeword for on-die correction; whether a message-only fetch touches fewer
  array bits depends on burst granularity and the codeword layout, and it is
  not assumed. `dram_array` = (1 - f_if) x the DRAM weight energy is the
  embedded arm's under every boundary, to the digit, and a recorded check
  (`dram_array_identical_to_embedded_reference`) fails if a bar takes more
  than the interface saving.
* The DRAM INTERFACE IS credited, and by exactly K/N: only the k message bits
  of each codeword leave the die, so `dram_interface` = f_if x the DRAM weight
  energy x K/N under every boundary (`dram_interface_scaled_by_K_over_N`
  fails if a bar leaves it at full width). It is the same saving on every
  bar, so it moves every placement against the embedded reference by the same
  amount and changes no ordering among them.
* The DECODER is still outside the comparison. It is the same BCH decoder
  relocated to the DRAM die, runs once per corrected codeword off the fetch
  path, and is DRAM-process logic; it cancels between the embedded bar and
  every placement exactly as the controller-side codec did (see `CODEC_NOTE`
  in experiments/recon.py).
* No capacity-driven reuse and no re-tiling: the mapping is fixed, so every
  access count is Timeloop's. Packed reduced weights would raise the effective
  SPad capacity -- that is Task 4's question, and crediting it here would mix
  the two answers.
* No operand-retention saving for anybody, and since 2026-09-08 R4b's register
  is no longer able to claim one. It holds the n-k bits the encoder regenerates
  -- `weight_bits x (1 - k/n)` per weight -- so it cannot serve a scratchpad
  read on its own: SPad read counts stay Timeloop's under every arm, the K/N
  discount on them is real rather than double-counted, and the register is
  charged BOTH its writes and its lockstep reads. `ReuseRegister` has the full
  argument, including why the previous accounting (a write and nothing else,
  with the reconstruction count amortized 82.85x as if the register served the
  reads) was internally inconsistent, and what a full-width operand cache
  would really be worth. Set `ECC_RECON_REUSE_REG_MODEL=full_width` to price it
  that way, or `=free` to reproduce the pre-2026-09-08 numbers.
"""
from __future__ import annotations

import math
import pathlib
import re
from dataclasses import dataclass, field, replace

from . import embedded


# ===========================================================================
#  the weight path, per architecture
# ===========================================================================
@dataclass(frozen=True)
class Stage:
    """One stage of the weight path, and the Timeloop levels that ARE it.

    `prefixes` matches level names as Timeloop prints them. A network is named
    after the two levels it joins (`NoC: <outer> <==> <inner>`), so the outer
    level's name is the stable half and the prefix stops there.
    """
    key: str
    label: str
    kind: str                 # dram | network | storage
    prefixes: tuple
    reducible: bool           # can this stage ever carry the reduced form?
    evidence: str = ""
    #: Only for `kind == "dram"`: which share of the one Timeloop DRAM level
    #: this stage is, `array` ((1 - f_if) of it) or `interface` (f_if of it).
    #: The two together are the whole level; `read_weight_path()` insists.
    dram_share: str = ""

    def matches(self, level):
        return any(level == p or level.startswith(p) for p in self.prefixes)


#: Eyeriss v2 as modelled by `archs/eyeriss_v2_like/`, outer to inner. The
#: design has NO weight GLB (its GLB banks are iact and psum, JETCAS 2019
#: Table II), so the weight path goes straight from the memory interface into
#: the hierarchical mesh -- which is why there is no global-buffer boundary in
#: `PLACEMENTS` either.
_EYERISS_V2_PATH = (
    Stage("dram_array", "DRAM array (complete codeword read)", "dram", ("DRAM",),
          reducible=False, dram_share="array",
          evidence="the array stores and reads the complete embedded codeword "
                   "for on-die correction: a row activation and a burst move "
                   "whole words and cannot pick the k message bits out of a "
                   "chunk (01_project_context Sec. 1 and Sec. 4, E_array; "
                   "02_..., risk 6: do not credit the array)"),
    Stage("dram_interface", "DRAM interface (message bits leave the die)", "dram",
          ("DRAM",), reducible=True, dram_share="interface",
          evidence="the decoder is on the DRAM die and off the fetch path, so "
                   "only the k message bits of each n-bit codeword are driven "
                   "off the die: I/O drivers, link and controller port carry k "
                   "of every n bits under every boundary (01_project_context "
                   "Sec. 1 and Sec. 4, E_interface x k/n)"),
    Stage("inter_cluster_mesh", "Inter-cluster mesh (HM-NoC)", "network",
          ("NoC: inter_PE_cluster_spatial",), reducible=True,
          evidence="JETCAS 2019 Sec. III-C: the hierarchical mesh joins the GLB "
                   "clusters to the PE clusters through the router clusters. "
                   "Timeloop's PE_cluster fanout level (meshX 16)."),
    Stage("cluster_local", "Cluster-local distribution to PE rows", "network",
          ("NoC: inter_PE_spatial",), reducible=True,
          evidence="JETCAS 2019 Sec. III-C: each router 'connects to one row of "
                   "PEs within the cluster'. Timeloop's PE fanout level "
                   "(meshY 12)."),
    Stage("weight_spad", "PE weight scratchpad", "storage", ("weights_spad",),
          reducible=True,
          evidence="JETCAS 2019 Table IV: 'weight data: 288B (SRAM)' per PE."),
)

#: `eyeriss_v2_like_wglb` adds a weight GLB, so it has one more reducible stage
#: and one more boundary. Registered so the study can be repeated on the
#: bracketing variant without editing this file.
_EYERISS_V2_WGLB_PATH = _EYERISS_V2_PATH[:2] + (
    Stage("weight_glb", "Weight global buffer", "storage", ("weight_glb",),
          reducible=True,
          evidence="the variant's extra weight level; NOT in the v2 paper -- see "
                   "archs/eyeriss_v2_like_wglb/README.md"),
) + _EYERISS_V2_PATH[2:]

#: Where the BCH decoder sits. `ondie` is the model since 2026-09-09 (decoder
#: on the DRAM die, off the fetch path, `dram_interface` reducible under every
#: boundary); `controller` is the pre-2026-09-09 model kept as a runnable row
#: for the diff (`dram_interface` not reducible, DRAM identical on every bar).
#: `config.RECON_DECODE_SITES` must stay in step with this tuple.
DECODE_SITES = ("ondie", "controller")

#: What the study prints instead of a number when `ECC_DRAM_IF_FRAC` is unset
#: and the decoder is on the die (experiments/recon.py prints the ceiling at
#: each of these).
F_IF_SENSITIVITY = (0.10, 0.25, 0.50)

WEIGHT_PATHS = {
    "eyeriss_v2_like": _EYERISS_V2_PATH,
    "eyeriss_v2_like_wglb": _EYERISS_V2_WGLB_PATH,
}


# ===========================================================================
#  the placements
# ===========================================================================
@dataclass(frozen=True)
class Placement:
    """One reconstruction boundary: what stays reduced, and where the encoder is.

    `reduced` is the set of stage keys that carry the REDUCED representation --
    everything from the die's output down to the boundary, so it always starts
    with `dram_interface` (the decoder is on the DRAM die; see the module
    docstring) and `validate_placement_space()`'s prefix rule enforces that
    order. `site_stage` plus `site_counter` say which access count drives the
    reconstruction count:

        reads       scalar reads out of that stage (per weight delivered)
        fills       scalar fills into that stage (per weight stored)
        ingresses   network ingresses (per weight word transported)
    """
    key: str                  # the env.sh spelling: recon1 .. recon5
    variant: str              # the results-store variant name (baseline.py)
    label: str
    short: str                # figure label, may contain a newline
    rating: str               # the source discussion's hypothesis, not a result
    reduced: tuple
    site_stage: str
    site_counter: str
    reuse_register: bool
    description: str


_V2_PLACEMENTS = (
    Placement(
        "recon1", "recon_source_noc_ingress",
        "R1 - reconstruct at the source / weight-NoC ingress",
        "R1\n@ NoC source", "2/5",
        reduced=("dram_interface",), site_stage="dram_array",
        site_counter="reads", reuse_register=False,
        description=(
            "The die drives only the k message bits of each codeword across "
            "the DRAM interface and one (or a few) encoders at the weight-NoC "
            "ingress put the missing bits straight back before the weight "
            "enters the network. Nothing on chip carries the reduced form, so "
            "nothing on chip gets cheaper; the encoder runs once per codeword "
            "fetched from DRAM. It is no longer the zero-saving control: it "
            "isolates the DRAM-interface saving, which every boundary shares, "
            "from any on-chip saving, and pays the reconstruction cost with "
            "nothing but that interface saving to set against it."),
    ),
    Placement(
        "recon2", "recon_destination_cluster",
        "R2 - reconstruct at the destination-cluster boundary",
        "R2\n@ cluster edge", "4/5",
        reduced=("dram_interface", "inter_cluster_mesh"),
        site_stage="inter_cluster_mesh",
        site_counter="ingresses", reuse_register=False,
        description=(
            "The long-distance hierarchical mesh carries the reduced form and "
            "an encoder at each destination cluster restores it before the "
            "cluster-local fanout. One encoder per cluster rather than one per "
            "PE, and the mesh -- the expensive hop -- moves fewer bits."),
    ),
    Placement(
        "recon3", "recon_pe_spad_input",
        "R3 - reconstruct at the PE weight-SPad input",
        "R3\n@ SPad input", "4/5",
        reduced=("dram_interface", "inter_cluster_mesh", "cluster_local"),
        site_stage="weight_spad", site_counter="fills", reuse_register=False,
        description=(
            "Both networks carry the reduced form; the encoder sits at the "
            "scratchpad write port, so the SPad itself stays full width and "
            "keeps its capacity and read cost. The encoder runs once per weight "
            "FILLED into a PE, which is the smallest count of any boundary "
            "below the network."),
    ),
    Placement(
        "recon4", "recon_pe_spad_output",
        "R4a - reconstruct on every weight-SPad read",
        "R4a\n@ SPad output", "3/5",
        reduced=("dram_interface", "inter_cluster_mesh", "cluster_local",
                 "weight_spad"),
        site_stage="weight_spad", site_counter="reads", reuse_register=False,
        description=(
            "The scratchpad stores the reduced form, so its write and read "
            "bit-volume fall too, and the encoder sits at its read port. The "
            "risk the source discussion names is the whole story here: the "
            "encoder now runs once per weight DELIVERED to the datapath, and a "
            "weight-stationary inner loop delivers each resident weight "
            "thousands of times."),
    ),
    Placement(
        "recon5", "recon_pe_spad_output_reuse_reg",
        "R4b - SPad output with a reconstructed-weight reuse register",
        "R4b\n@ SPad + reg", "5/5",
        reduced=("dram_interface", "inter_cluster_mesh", "cluster_local",
                 "weight_spad"),
        site_stage="weight_spad", site_counter="retained",
        reuse_register=True,
        description=(
            "R4a's savings with the reconstruction amortized: a register holds "
            "what the encoder rebuilt and reloads only when the required weight "
            "changes, so one reconstruction serves every use until the "
            "scratchpad tile is replaced. The amortization is read off the "
            "mapping -- the loops below the scratchpad walk a tile of "
            "`inner_tile` weights and repeat reads/fills times -- so a register "
            "covering that tile reconstructs once per FILL, the same count as "
            "R3, while keeping the scratchpad reduced. A register smaller than "
            "the tile catches nothing, because a cyclic walk is the LRU worst "
            "case. See `retention_model()`. WHAT the register holds is a "
            "separate question from how often it is reloaded, and it decides "
            "what R4b may be charged: by default it retains only the n-k bits "
            "the encoder regenerates, which is why the scratchpad's read count "
            "is still Timeloop's. See `ReuseRegister`."),
    ),
)

PLACEMENTS = {
    "eyeriss_v2_like": _V2_PLACEMENTS,
    "eyeriss_v2_like_wglb": _V2_PLACEMENTS,
}

#: Reference bars every placement is read against. They are not placements;
#: they are Task 1's and Task 2's numbers, produced by Task 1's and Task 2's own
#: functions so the figure cannot disagree with those files.
REFERENCE_BARS = (("baseline", "Baseline\nexternal parity"),
                  ("embedded", "Embedded\nno ext. parity"))


def supported_archs():
    return tuple(sorted(WEIGHT_PATHS))


def decode_site(cfg):
    """`ondie` (the model) or `controller` (the pre-2026-09-09 diff row)."""
    site = getattr(cfg, "recon_decode_site", "ondie") if cfg is not None else "ondie"
    if site not in DECODE_SITES:
        raise ValueError(f"unknown decode site {site!r}; one of {DECODE_SITES}")
    return site


def stages_for(arch, cfg=None):
    """The weight path of `arch`, as the decode site makes it.

    The tables above are written for the on-die decoder. Under `controller`
    the complete codeword crosses the interface, so `dram_interface` is handed
    back with `reducible=False` and nothing else changes -- which is exactly
    what makes the old numbers come back rather than a different model.
    """
    if arch not in WEIGHT_PATHS:
        raise KeyError(arch)
    stages = WEIGHT_PATHS[arch]
    if decode_site(cfg) == "controller":
        stages = tuple(replace(s, reducible=False) if s.key == "dram_interface"
                       else s for s in stages)
    return stages


def effective_placement(placement, cfg):
    """`placement` as the decode site makes it: under `controller` the interface
    is not reducible, so it leaves every placement's `reduced` set."""
    if decode_site(cfg) == "controller" and "dram_interface" in placement.reduced:
        return replace(placement, reduced=tuple(
            k for k in placement.reduced if k != "dram_interface"))
    return placement


def placements_for(arch, cfg=None):
    if arch not in PLACEMENTS:
        raise KeyError(arch)
    return tuple(effective_placement(p, cfg) for p in PLACEMENTS[arch])


def placement_by_key(arch, key, cfg=None):
    for p in placements_for(arch, cfg):
        if p.key == key:
            return p
    return None


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
    reducible = [s.key for s in stages if s.reducible]
    order = {key: i for i, key in enumerate(reducible)}
    rows, violations = [], []
    reached = set()
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


# ===========================================================================
#  reading the weight path out of the cached Timeloop output
# ===========================================================================
_NUM = r"([\d.eE+-]+)"


def _grab(pattern, text, cast=float):
    m = re.search(pattern, text)
    if m is None:
        return None
    try:
        return cast(m.group(1))
    except ValueError:
        return None


def _dataspace_block(body, dataspace):
    """The `Weights:` sub-block of one level's STATS section."""
    m = re.search(rf"\n\s+{dataspace}\s*:\s*\n(.*?)(?=\n\s+(?:Weights|Inputs|Outputs)\s*:\s*\n|\Z)",
                  body, re.S)
    return m.group(1) if m else ""


@dataclass
class StageStats:
    """Weight-only totals for one weight-path stage of one layer."""
    key: str
    kind: str
    levels: list = field(default_factory=list)
    energy_pJ: float = 0.0
    wire_pJ: float = 0.0          # networks: the bit-proportional half
    switch_pJ: float = 0.0        # networks: router + ingress switching
    reads: float = 0.0            # TOTAL scalar reads (all instances)
    fills: float = 0.0            # TOTAL scalar fills (all instances)
    ingresses: float = 0.0        # TOTAL network ingresses (all instances)
    utilized_capacity: float = 0.0   # storage: weights resident per instance
    block_bits: int = 0              # storage: block size x word bits
    instances: float = 0.0
    level_share: float = 1.0      # dram: this stage's share of the Timeloop level

    def counter(self, name):
        if name == "reads":
            return self.reads
        if name == "fills":
            return self.fills
        if name == "ingresses":
            return self.ingresses
        raise ValueError(f"unknown access counter {name!r}")

    def to_dict(self):
        return {
            "stage": self.key, "kind": self.kind, "timeloop_levels": self.levels,
            "weight_energy_pJ": self.energy_pJ,
            "weight_wire_energy_pJ": self.wire_pJ,
            "weight_switching_energy_pJ": self.switch_pJ,
            "weight_scalar_reads_total": self.reads,
            "weight_scalar_fills_total": self.fills,
            "weight_network_ingresses_total": self.ingresses,
            "weights_resident_per_instance": self.utilized_capacity,
            "physical_word_bits": self.block_bits,
            "instances": self.instances,
            "share_of_the_timeloop_level": self.level_share,
        }


@dataclass
class LayerWeightPath:
    """One layer's weight path: every stage, plus what the loop nest implies."""
    layer: str
    shape: str
    weights: int
    scale: float
    stages: dict
    nest: dict                      # weight_loop_nest() for this layer
    unclaimed: list                 # weight-carrying levels no stage claimed
    stats_path: str

    def total_weight_energy(self):
        return sum(s.energy_pJ for s in self.stages.values())


_LOOP = re.compile(r"for\s+([A-Za-z]+)\s+in\s+\[0:(\d+)\)(\s*\(Spatial-[XY]\))?")

#: Loop dimensions a weight depends on. C and M are per group and G selects the
#: group, so all five change WHICH weight is being used; P, Q and N do not.
WEIGHT_DIMS = frozenset("CMRSG")


#: Order the embedding pipeline flattens a conv weight tensor in: [M][C][R][S]
#: row-major, so S is the fastest-varying index and M the slowest. It decides
#: whether two consecutively ACCESSED weights are consecutive in the codeword
#: BIT STREAM, which is the only way a group-sized reconstruction buffer can
#: serve a second access without rebuilding.
STREAM_ORDER = ("M", "C", "R", "S")


def weight_loop_nest(map_txt):
    """What the loop nest below the innermost weight buffer implies for reuse.

    Returns a dict. Three quantities, and they answer three different questions
    that the first version of this module ran together:

    `inner_tile`      the product of the loop bounds over WEIGHT dimensions
                      below the innermost weight buffer -- i.e. how many
                      DISTINCT weights those loops walk before repeating. It
                      equals the buffer's utilized capacity on every mapping
                      measured here, and it is the capacity a reconstructed-
                      weight register needs to cover to serve the repeats.
    `consecutive_run` the product of the innermost weight-INVARIANT temporal
                      loop bounds: how many uses in a row one weight gets, and
                      therefore what a ONE-ENTRY latch can serve. This is 1
                      whenever the innermost temporal loop walks a weight
                      dimension.
    `innermost_weight_dim`
                      the fastest-varying weight dimension below the buffer.
                      With the [M][C][R][S] stream order above, only `S` makes
                      consecutively accessed weights consecutive in the
                      codeword bit stream; anything else strides across
                      codewords, so a group-sized buffer catches nothing.

    THE DISTINCTION MATTERS AND GETTING IT WRONG IS WHAT MADE R4b LOOK USELESS.
    `consecutive_run` is 1 for every mapping in this study, because the
    innermost temporal loop is `M` or `C`. But the source discussion's R4b
    (`01_project_context_and_architectures.txt` Sec. 5.4, EV2-C) is not a
    one-entry latch that must see consecutive uses -- it is
    "Reload/reconstruct when the required weight CHANGES", a retained
    reconstruction that survives until the weight it holds is replaced. What
    that serves is `reads / fills` uses per reconstruction, which is 49x-6272x
    here, not 1x.
    """
    out = {"inner_tile": 0, "consecutive_run": 1.0, "innermost_weight_dim": None,
           "loops_below": [], "weight_level": "", "evidence": ""}
    path = pathlib.Path(map_txt)
    if not path.exists():
        out["evidence"] = "no map.txt in the mapper cache -- nothing read"
        return out
    lines = path.read_text().split("\n")
    idx = [i for i, ln in enumerate(lines) if re.search(r"\bWeights:\s*\d", ln)]
    if not idx:
        out["evidence"] = "no level in the mapping holds Weights"
        return out
    inner = max(idx)
    out["weight_level"] = lines[inner].split("[")[0].strip()
    loops = []
    for ln in lines[inner + 1:]:
        m = _LOOP.search(ln)
        if m:
            loops.append((m.group(1), int(m.group(2)), bool(m.group(3))))
    out["loops_below"] = [{"dimension": d, "bound": b,
                           "kind": "spatial" if s else "temporal"}
                          for d, b, s in loops]

    tile = 1
    for dim, bound, _spatial in loops:
        if dim in WEIGHT_DIMS:
            tile *= bound
    out["inner_tile"] = tile

    run, trace = 1.0, []
    for dim, bound, spatial in reversed(loops):
        if spatial:
            trace.append(f"skip spatial {dim}={bound}")
            continue
        if dim in WEIGHT_DIMS:
            trace.append(f"stop at temporal {dim}={bound} (changes the weight)")
            if out["innermost_weight_dim"] is None:
                out["innermost_weight_dim"] = dim
            break
        run *= bound
        trace.append(f"x{bound} temporal {dim} (weight-invariant)")
    out["consecutive_run"] = run
    if not loops:
        trace.append("no loops below the weight level")
    out["evidence"] = f"below {out['weight_level']}: " + "; ".join(trace)
    out["stream_consecutive"] = (out["innermost_weight_dim"] == STREAM_ORDER[-1])
    return out


def _network_split(specs, block, energy_total, instances):
    """Split one network's weight energy into wire and switching.

    Timeloop's Legacy network model (model/network-legacy.cpp:408) is

        energy = wire_hops x energy_per_hop
               + routers_touched x router_energy
               + ingresses x ingress_energy

    with `energy_per_hop = word_bits x tile_width_mm x wire_energy` and
    `routers_touched = (1 + floor(hops)) x ingresses`. Only the first term is
    proportional to the number of BITS moved, so only the first term falls
    automatically when the reduced form is transported; the switching term falls
    only if the reduced values are REPACKED into fewer flits. Every field below
    is printed by Timeloop, so the split is read off its own numbers rather than
    modelled -- and it is then rescaled to the energy Timeloop reported, so the
    two halves always sum to the figure the baseline arm is charged.
    """
    hops = _grab(rf"Average number of hops\s*:\s*{_NUM}", block) or 0.0
    per_hop_fj = _grab(rf"Energy \(per-hop\)\s*:\s*{_NUM}\s*fJ", block)
    per_hop = (per_hop_fj / 1000.0) if per_hop_fj is not None else 0.0
    ingress = _grab(rf"Ingresses\s*:\s*{_NUM}", block) or 0.0
    router_pj = _grab(rf"Router energy\s*:\s*{_NUM}\s*pJ", specs) or 0.0
    ingress_pj = _grab(rf"Ingress energy\s*:\s*{_NUM}\s*pJ", specs) or 0.0

    wire = ingress * hops * per_hop * instances
    switch = ((1.0 + math.floor(hops)) * ingress * router_pj
              + ingress * ingress_pj) * instances
    model = wire + switch
    if model <= 0 or energy_total <= 0:
        # Nothing to split (link-transfer or spatial-reduction energy only).
        # Charge it as switching: it is not proportional to the word width.
        return 0.0, energy_total, model
    k = energy_total / model
    return wire * k, switch * k, model


NO_F_IF = (
    "ECC_DRAM_IF_FRAC is not set. The decoder is on the DRAM die "
    "(ECC_RECON_DECODE_SITE=ondie), so the DRAM weight energy has to be split "
    "into its array and interface shares, and f_if -- the interface share of "
    "Accelergy's flat per-bit DRAM constant -- is a CITED DRAM energy breakdown, "
    "not a modelling choice. No LPDDR4 array-vs-I/O breakdown is cited in "
    "archs/_shared/provenance.yaml yet (see its `dram_interface_share` block "
    "for what has been found), so the study refuses to produce a single number. "
    "Set ECC_DRAM_IF_FRAC=<0..1] for a sensitivity run, or "
    "ECC_RECON_DECODE_SITE=controller for the pre-2026-09-09 model, in which "
    "nothing depends on the split.")


def dram_interface_fraction(cfg):
    """`(f_if, provenance)` -- the interface share the DRAM level is split by.

    Under `controller` an unset `f_if` is 0.0 and the whole level is booked to
    `dram_array`: neither share is reducible there, so no number depends on the
    split and refusing would only stop the diff row from running. Under `ondie`
    an unset `f_if` is a refusal, because every placement's DRAM term depends
    on it and there is no cited default (`NO_F_IF`).
    """
    f = getattr(cfg, "dram_if_frac", None) if cfg is not None else None
    site = decode_site(cfg)
    if f is None:
        if site == "controller":
            return 0.0, ("ECC_DRAM_IF_FRAC unset; the whole DRAM level is booked "
                         "to dram_array. Under controller-side correction "
                         "neither share is reduced, so no number depends on it")
        raise ValueError(NO_F_IF)
    f = float(f)
    if not 0.0 <= f <= 1.0:
        raise ValueError(f"ECC_DRAM_IF_FRAC={f} is not a fraction in [0, 1]")
    return f, f"ECC_DRAM_IF_FRAC={f} ({getattr(cfg, 'dram_if_frac_note', '') or 'set for this run'})"


def _level_shares(claimants, f_if, level):
    """How much of one Timeloop level each claiming stage is.

    One claimant owns the level. Two are allowed only when they are the
    `dram_array` / `dram_interface` pair, whose shares (1 - f_if) and f_if sum
    to one -- so the level is still claimed exactly once in total. Anything
    else is a table error and is refused rather than double-counted.
    """
    if len(claimants) == 1:
        return {claimants[0].key: 1.0}
    shares = {s.dram_share for s in claimants}
    if (len(claimants) != 2 or any(s.kind != "dram" for s in claimants)
            or shares != {"array", "interface"}):
        raise ValueError(
            f"Timeloop level {level!r} is claimed by {[s.key for s in claimants]}; "
            f"a level may be claimed by exactly one stage, or by the "
            f"dram_array/dram_interface pair that splits it by f_if")
    return {s.key: (f_if if s.dram_share == "interface" else 1.0 - f_if)
            for s in claimants}


def read_weight_path(cfg, arch, layer, stats_path):
    """Parse ONE layer's weight path out of its cached Timeloop output.

    `stats_path` is the `timeloop-mapper.stats.txt` the mapper cache holds for
    this layer shape; `timeloop-mapper.map.txt` beside it supplies the loop
    nest. Both are read, never written, and neither can be produced by this
    module -- if a shape is not in the cache there is nothing to evaluate.
    """
    stats_path = pathlib.Path(stats_path)
    text = stats_path.read_text()
    scale = float(getattr(layer, "count", 1) or 1)
    stage_defs = stages_for(arch, cfg)
    stages = {s.key: StageStats(s.key, s.kind) for s in stage_defs}
    f_if, _f_note = dram_interface_fraction(cfg)
    unclaimed = []

    # ---- storage and arithmetic levels -------------------------------------
    from .timeloop import _split_networks       # one definition of the split
    levels_text, networks_text = _split_networks(text)
    chunks = re.split(r"===\s*(.+?)\s*===", levels_text)[1:]
    for level, body in zip(chunks[0::2], chunks[1::2]):
        wblock = _dataspace_block(body, "Weights")
        if not wblock:
            continue
        energy = (_grab(rf"Energy \(total\)\s*:\s*{_NUM}\s*pJ", wblock) or 0.0) * scale
        if energy <= 0:
            continue
        claimants = [s for s in stage_defs if s.matches(level)]
        if not claimants:
            unclaimed.append({"level": level, "weight_energy_pJ": energy})
            continue
        shares = _level_shares(claimants, f_if, level)
        util = _grab(r"Utilized instances \(max\)\s*:\s*(\d+)", wblock, int) or 1
        reads = (_grab(rf"Scalar reads \(per-instance\)\s*:\s*{_NUM}", wblock)
                 or 0.0) * util * scale
        fills = ((_grab(rf"Scalar fills \(per-instance\)\s*:\s*{_NUM}", wblock) or 0.0)
                 + (_grab(rf"Scalar updates \(per-instance\)\s*:\s*{_NUM}", wblock) or 0.0)
                 ) * util * scale
        cap = _grab(r"Utilized capacity\s*:\s*(\d+)", wblock, int)
        word = _grab(r"Word bits\s*:\s*(\d+)", body, int) or cfg.weight_bits
        blk = _grab(r"Block size\s*:\s*(\d+)", body, int) or 1
        for stage in claimants:
            # ENERGY is split by the stage's share of the level; the ACCESS
            # COUNTS are not -- a scalar read of the DRAM is one read of the
            # array and one word across the interface, so both stages see all
            # of them (R1 counts its reconstructions off dram_array's reads).
            share = shares[stage.key]
            st = stages[stage.key]
            st.levels.append(level)
            st.level_share = share
            st.energy_pJ += energy * share
            st.switch_pJ += energy * share    # storage: no wire/switching split
            st.reads += reads
            st.fills += fills
            st.instances = max(st.instances, util)
            if cap is not None:
                st.utilized_capacity = (cap if st.utilized_capacity == 0
                                        else min(st.utilized_capacity, cap))
            st.block_bits = max(st.block_bits, word * blk)

    # ---- networks ----------------------------------------------------------
    for nblock in re.split(r"\nNetwork \d+\n-+\n", "\n" + networks_text)[1:]:
        name = next((ln.strip() for ln in nblock.split("\n") if ln.strip()), "?")
        level = f"NoC: {name}"
        wblock = _dataspace_block(nblock, "Weights")
        if not wblock:
            continue
        e_net = _grab(rf"\n\s+Energy \(total\)\s*:\s*{_NUM}\s*pJ", wblock) or 0.0
        e_link = _grab(rf"Link transfer energy \(total\)\s*:\s*{_NUM}\s*pJ", wblock) or 0.0
        e_red = _grab(rf"Spatial Reduction Energy \(total\)\s*:\s*{_NUM}\s*pJ", wblock) or 0.0
        energy = (e_net + e_link + e_red) * scale
        if energy <= 0:
            continue
        claimants = [s for s in stage_defs if s.matches(level)]
        if not claimants:
            unclaimed.append({"level": level, "weight_energy_pJ": energy})
            continue
        if len(claimants) != 1:
            raise ValueError(f"network level {level!r} is claimed by "
                             f"{[s.key for s in claimants]}; only the DRAM level "
                             f"may be split between two stages")
        stage = claimants[0]
        per_inst = _grab(rf"Energy \(per-instance\)\s*:\s*{_NUM}\s*pJ", wblock) or 0.0
        instances = round(e_net / per_inst) if per_inst > 0 else 1
        instances = max(1, instances)
        specs = nblock.split("STATS")[0]
        wire, switch, _model = _network_split(specs, wblock, energy, instances)
        ingress = (_grab(rf"Ingresses\s*:\s*{_NUM}", wblock) or 0.0) * instances * scale
        st = stages[stage.key]
        st.levels.append(level)
        st.energy_pJ += energy
        st.wire_pJ += wire
        st.switch_pJ += switch
        st.ingresses += ingress
        st.instances = max(st.instances, instances)
        st.block_bits = max(st.block_bits,
                            _grab(r"Word bits\s*:\s*(\d+)", specs, int) or cfg.weight_bits)

    nest = weight_loop_nest(stats_path.parent / "timeloop-mapper.map.txt")
    return LayerWeightPath(
        layer=getattr(layer, "name", "?"), shape=getattr(layer, "shape_name", "?"),
        weights=int(getattr(layer, "weights", 0)), scale=scale, stages=stages,
        nest=nest, unclaimed=unclaimed, stats_path=str(stats_path))


@dataclass
class ModelWeightPath:
    """Every mapped layer's weight path, summed, with the per-layer detail kept."""
    arch: str
    model: str
    stages: dict
    per_layer: list
    retention: dict
    unclaimed: list
    #: Directory holding the cached Timeloop output, hence the Accelergy ERT.
    #: `ReuseRegister(mode='full_width')` needs it to split the scratchpad's
    #: read energy from its write energy; every other mode ignores it.
    stats_dir: str = ""
    #: The stage definitions this path was read with -- `stages_for(arch, cfg)`,
    #: so `dram_interface.reducible` reflects the decode site of the run.
    stage_defs: tuple = ()
    decode_site: str = "ondie"
    dram_if_frac: float = 0.0
    dram_if_note: str = ""

    def stage(self, key):
        return self.stages.get(key)

    def dram_weight_energy(self):
        """The whole DRAM weight energy: both shares, i.e. the Raw record's."""
        return sum(s.energy_pJ for s in self.stages.values() if s.kind == "dram")

    def dram_split(self):
        """The array/interface split of the DRAM weight energy, as a record."""
        array = self.stages.get("dram_array")
        iface = self.stages.get("dram_interface")
        return {
            "decode_site": self.decode_site,
            "f_if": self.dram_if_frac,
            "f_if_provenance": self.dram_if_note,
            "dram_weight_energy_pJ": self.dram_weight_energy(),
            "dram_array_pJ": array.energy_pJ if array else 0.0,
            "dram_interface_pJ": iface.energy_pJ if iface else 0.0,
            "interface_reducible": bool(iface is not None and any(
                s.key == "dram_interface" and s.reducible for s in self.stage_defs)),
            "rule": ("dram_array = (1 - f_if) x DRAM weight energy, never reduced; "
                     "dram_interface = f_if x DRAM weight energy, x K/N under "
                     "every boundary when the decoder is on the DRAM die "
                     "(01_project_context Sec. 4). Accelergy's CactiDRAM is one "
                     "flat per-bit constant, so the split is the evaluator's, "
                     "and f_if is a cited parameter, not a default"),
        }

    def total_weight_energy(self):
        return sum(s.energy_pJ for s in self.stages.values())

    def reducible_energy(self, keys):
        """Weight energy of the stages a placement would leave reduced.

        The CEILING on what any placement can save, before a single
        reconstruction is charged. Reported on every result because it is the
        number that makes a small saving legible as arithmetic rather than as a
        missing term: on `eyeriss_v2_like` the whole on-chip weight path is
        8.5 % of inference energy, so K/N = 0.81 can give back at most 1.6 %.
        """
        return sum(self.stages[k].energy_pJ for k in keys
                   if k in self.stages)

    def to_dict(self):
        return {
            "architecture": self.arch,
            "model": self.model,
            "stages": [self.stages[s.key].to_dict() for s in self.stage_defs
                       if s.key in self.stages],
            "stage_definitions": [
                {"stage": s.key, "label": s.label, "kind": s.kind,
                 "timeloop_level_prefixes": list(s.prefixes),
                 "carries_reduced_representation_possible": s.reducible,
                 "share_of_the_dram_level": s.dram_share or None,
                 "evidence": s.evidence}
                for s in self.stage_defs],
            "dram_split": self.dram_split(),
            "reconstructed_weight_retention": self.retention,
            "unclaimed_weight_levels": self.unclaimed,
            "per_layer": self.per_layer,
        }


def weight_path(cfg, arch, model, layers, stats_paths):
    """Sum every mapped layer's weight path into one record.

    `stats_paths` maps a layer's shape name to its cached stats file. A layer
    with no entry was not mapped and is skipped, exactly as `energy.gather()`
    skips it, so the two records cover the same layers.
    """
    stage_defs = stages_for(arch, cfg)
    stages = {s.key: StageStats(s.key, s.kind) for s in stage_defs}
    f_if, f_note = dram_interface_fraction(cfg)
    per_layer, unclaimed, layer_paths = [], [], []
    for layer in layers:
        path = stats_paths.get(layer.shape_name)
        if path is None:
            continue
        lp = read_weight_path(cfg, arch, layer, path)
        for key, st in lp.stages.items():
            agg = stages[key]
            agg.levels = sorted(set(agg.levels) | set(st.levels))
            agg.energy_pJ += st.energy_pJ
            agg.wire_pJ += st.wire_pJ
            agg.switch_pJ += st.switch_pJ
            agg.reads += st.reads
            agg.fills += st.fills
            agg.ingresses += st.ingresses
            agg.instances = max(agg.instances, st.instances)
            agg.block_bits = max(agg.block_bits, st.block_bits)
            agg.level_share = st.level_share
            if st.utilized_capacity:
                agg.utilized_capacity = (st.utilized_capacity
                                         if agg.utilized_capacity == 0
                                         else min(agg.utilized_capacity,
                                                  st.utilized_capacity))
        unclaimed += [dict(u, layer=lp.layer) for u in lp.unclaimed]
        per_layer.append({
            "layer": lp.layer, "shape": lp.shape, "weights": lp.weights,
            "repeat_count": lp.scale,
            "loop_nest_below_weight_buffer": lp.nest,
            "weight_energy_pJ": lp.total_weight_energy(),
            "stages": {k: st.to_dict() for k, st in lp.stages.items()
                       if st.energy_pJ > 0},
        })
        layer_paths.append(lp)

    retention = retention_model(cfg, arch, layer_paths)
    stats_dir = str(pathlib.Path(layer_paths[-1].stats_path).parent) \
        if layer_paths else ""
    return ModelWeightPath(arch=arch, model=model, stages=stages,
                           per_layer=per_layer, retention=retention,
                           unclaimed=unclaimed, stats_dir=stats_dir,
                           stage_defs=stage_defs, decode_site=decode_site(cfg),
                           dram_if_frac=f_if, dram_if_note=f_note)


# ===========================================================================
#  R4b: how many uses does ONE reconstruction serve?
# ===========================================================================
def retention_model(cfg, arch, layer_paths):
    """The reconstruction count for a boundary that RETAINS what it rebuilt.

    THIS IS THE QUANTITY THE FIRST VERSION OF THIS MODULE GOT WRONG, so it is
    worth being explicit about what changed and why.

    R4b is `01_project_context_and_architectures.txt` Sec. 5.4's EV2-C:

        4-bit SPad --> XOR encoder --> 8-bit decoded weight register --> MAC
                          ^                         |
                          |                         +--> Reuse across MAC uses
                   Reload/reconstruct when the required weight changes

    The first model read that as a ONE-ENTRY latch and asked how many uses a
    weight gets IN A ROW. On every mapping in this study that is 1, because the
    innermost temporal loop below the scratchpad walks `M` or `C` -- both weight
    dimensions -- so R4b came out identical to R4a and the 5/5 boundary looked
    useless. That was an artifact of the question, not a property of the design:
    "reload when the required weight CHANGES" is a retained reconstruction, not
    one that must see back-to-back uses.

    What it actually serves is set by the register's CAPACITY against the
    working set the loops below the scratchpad walk:

    * The loops below the weight buffer walk `inner_tile` distinct weights and
      then repeat. Measured on all 12 resnet18 shapes here, `inner_tile` equals
      the buffer's utilized capacity exactly.
    * That walk repeats `reads / fills` times before the buffer is refilled --
      49x to 6272x on these mappings.
    * A register that holds the tile therefore reconstructs each weight ONCE
      PER FILL: `N_rec = fills`, the same count as R3, while keeping the
      scratchpad reduced -- which is exactly why the source discussion rates
      R4b above both R3 and R4a.
    * A register SMALLER than the tile catches nothing: the access pattern is a
      cyclic walk of period `inner_tile`, which is the standard worst case for
      LRU, so every access misses and `N_rec = reads`. That is not a
      simplification -- cyclic-walk-versus-LRU has no partial hit rate.

    `ECC_RECON_REUSE_REG_ENTRIES` picks the capacity: `tile` (the default)
    sizes it to whatever the mapping's tile turns out to be, and an integer
    fixes it so a small buffer can be asked about. Either way the register
    capacity REQUIRED is reported, because a tile-sized register is not
    obviously "small compared with the SPad" -- on these mappings it is 16-256
    full-width weights against a 288-weight scratchpad, and that is a real
    overhead the plan asks to be shown rather than assumed away.
    """
    want = cfg.recon_reuse_reg_entries          # "tile" or an int
    rows, rec_weights, reads_total, fills_total = [], 0.0, 0.0, 0.0
    need = 0
    for lp in layer_paths:
        spads = [st for st in lp.stages.values() if st.kind == "storage"]
        reads = sum(st.reads for st in spads)
        fills = sum(st.fills for st in spads)
        resident = max([st.utilized_capacity for st in spads] or [0])
        tile = lp.nest.get("inner_tile") or 0
        # The tile the loops walk and the tile the buffer holds should agree;
        # where they do not, take the LARGER as the working set a register has
        # to cover, because a register that covers less of it catches nothing.
        working_set = max(tile, resident)
        entries = working_set if want == "tile" else int(want)
        covers = entries >= working_set > 0
        need = max(need, working_set)
        rec = fills if covers else reads
        rec_weights += rec
        reads_total += reads
        fills_total += fills
        rows.append({
            "layer": lp.layer, "shape": lp.shape,
            "weights_resident_in_buffer": resident,
            "inner_tile_walked_by_the_loops": tile,
            "working_set_weights": working_set,
            "register_entries": entries,
            "register_covers_the_working_set": covers,
            "buffer_weight_reads_total": reads,
            "buffer_weight_fills_total": fills,
            "uses_per_reconstruction": (reads / fills) if fills else 0.0,
            "weights_reconstructed": rec,
            "consecutive_run_a_one_entry_latch_would_serve":
                lp.nest.get("consecutive_run"),
            "innermost_weight_dimension": lp.nest.get("innermost_weight_dim"),
            "accesses_are_consecutive_in_the_codeword_bit_stream":
                lp.nest.get("stream_consecutive"),
        })
    return {
        "register_entries_requested": want,
        "register_entries_required_max": need,
        # The width depends on WHAT the register holds -- see `ReuseRegister`.
        # `complement` (the default since 2026-09-08) retains only the n-k bits
        # the encoder regenerates, so reduced SPad + register is exactly
        # weight_bits per resident weight: 1.00x the baseline PE, not 1.81x.
        "register_width_bits": (
            cfg.weight_bits * (1.0 - cfg.code_k / cfg.code_n)
            if getattr(cfg, "recon_reuse_reg_model", "free") == "complement"
            else cfg.weight_bits),
        "register_mode": getattr(cfg, "recon_reuse_reg_model", "free"),
        "weights_reconstructed_with_retention": rec_weights,
        "weights_reconstructed_without_retention": reads_total,
        "weights_reconstructed_at_buffer_fill": fills_total,
        "amortization_vs_no_retention": (reads_total / rec_weights)
                                        if rec_weights else 1.0,
        "rule": ("the loops below the innermost weight buffer walk `inner_tile` "
                 "distinct weights and repeat reads/fills times; a register that "
                 "holds that working set reconstructs each weight once per fill, "
                 "and one smaller than it catches nothing because a cyclic walk "
                 "is the LRU worst case"),
        "source": ("01_project_context_and_architectures.txt Sec. 2.3 and 5.4 "
                   "(EV2-C): 'Reload/reconstruct when the required weight "
                   "changes' -- a retained reconstruction, not a latch that "
                   "must see consecutive uses"),
        "per_layer": rows,
    }


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

    def storage_scale(self, block_bits):
        """Access-count-and-energy scale for a storage stage."""
        full = self._per_word(block_bits, self.weight_bits)
        red = self._per_word(block_bits, self.reduced_bits_per_weight)
        return full / red if red > 0 else 1.0

    def weights_per_word(self, block_bits, reduced=True):
        """How many weights one physical word of `block_bits` carries.

        Public because the reuse register is accessed in LOCKSTEP with the
        scratchpad -- one register access per scratchpad access -- so its
        access count is the scratchpad's word count, not its scalar count.
        See `ReuseRegister`.
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


# ===========================================================================
#  R4b's reuse register: WHAT IT HOLDS, and therefore what it costs
#  Added 2026-09-08 after the audit described below. READ THIS BEFORE
#  TOUCHING R4b -- it is the whole argument for R4b's headline number.
# ===========================================================================
@dataclass(frozen=True)
class ReuseRegister:
    """What R4b's register physically stores, which decides three things at once:
    its width, whether it serves scratchpad reads, and what it may be charged.

    ------------------------------------------------------------------------
    THE HOLE THIS CLOSES  (the audit, 2026-09-08)
    ------------------------------------------------------------------------
    Until now `evaluate_placement` did two things that could not both be true:

      1. set `N_rec = fills` for R4b (2.78 M instead of R4a's 230 M) BECAUSE
         the register retains what the encoder rebuilt and serves 82.85 uses
         per reconstruction; and
      2. scaled the WHOLE `weights_spad` energy by K/N -- that is, kept
         Timeloop's 1,814,073,344 scalar reads (one per MAC) intact, as if the
         register were not there.

    If the register serves 82.85 of every 82.85 uses then those uses do not
    read the scratchpad, and the K/N discount is being taken on read traffic
    the same register has already removed. 97.6% of the scratchpad's 369.276
    uJ is read energy (604,691,115 word reads x 0.596239 pJ = 360.540 uJ,
    against 8.735 uJ of fills; the ERT reconciles Timeloop's total to 0.000%),
    so this is not a rounding argument -- it is the entire result. Under the
    old `free` accounting R4b scored +2.971% vs embedded at BCH(63,39); charge
    it consistently as a full-width operand cache and reconstruction's own
    marginal value drops to +0.16%, the rest being an ECC-independent register.

    ------------------------------------------------------------------------
    THE RESOLUTION: THE REGISTER NEVER NEEDED TO BE FULL WIDTH
    ------------------------------------------------------------------------
    Go back to the layout in `embedded.py`. The weight bit stream is cut into
    n-bit chunks and `ParityOverwriteByTopWeightsEncode` OVERWRITES the n-k
    lowest-significance bits of each chunk with parity. On chip the reduced
    form keeps the k message bits per chunk; reconstruction re-runs the encoder
    to regenerate the n-k bits that were overwritten and splices them back so
    the 8-bit weight boundaries line up again.

    So the register only has to hold the regenerated COMPLEMENT -- n-k bits per
    n-bit chunk, i.e. `weight_bits x (1 - k/n)` per weight (3.05 b at
    BCH(63,39), 1.52 b at BCH(63,51)) -- not a whole 8-bit weight. Then:

      * the scratchpad IS still read once per MAC, in reduced form, so the K/N
        discount on all 1.81 G reads is real and not double-counted;
      * the encoder still runs once per fill, so `N_rec = fills` still holds;
      * scratchpad + register = `weight_bits x k/n + weight_bits x (1 - k/n)`
        = exactly `weight_bits` per resident weight -- 1.00x the baseline PE's
        weight storage at EVERY K, against 1.81x for a full-width register;
      * and the symmetry objection dies: a baseline PE stores whole weights,
        has no missing bits, and there is nothing to hand it that would let it
        make the same saving. The register is genuinely an ECC component.

    ------------------------------------------------------------------------
    WHY `full_width` IS KEPT AS A MODE RATHER THAN ARGUED AGAINST IN PROSE
    ------------------------------------------------------------------------
    `full_width` reproduces the auditor's reading as a RUNNABLE row: the
    register holds whole weights, serves the reads, and the scratchpad is
    touched only on fills. It also reports what the same register would give a
    PE with no ECC at all, priced two ways, because that is the number the
    reading turns on. Charged at the ERT's cheapest per-PE register write
    (0.0328 pJ, `ifmap_spad`, a 24x8b = 192-bit array) the register alone looks
    worth ~6-7% of total inference energy; priced instead at `weights_spad`'s
    own read energy -- and a 256x8b register is 2048 bits against that
    scratchpad's 96x24b = 2304 bits, i.e. the SAME array -- it is worth about
    nothing, because you cannot beat a 288-entry register file by putting a
    256-entry register file in front of it. Both figures are emitted so the
    conclusion is checkable instead of asserted.

    ------------------------------------------------------------------------
    THE MODES
    ------------------------------------------------------------------------
    `complement`  the register holds the n-k regenerated bits per weight and is
                  read in LOCKSTEP with the scratchpad (one register access per
                  scratchpad word access), charged at the ERT register energy
                  scaled by the complement's share of the word. Scratchpad read
                  counts are Timeloop's. THE DEFAULT, and the only mode whose
                  storage, access counts and energy are mutually consistent.
    `full_width`  the register holds whole weights and SERVES the reads, so the
                  scratchpad is read once per fill instead of once per MAC.
                  Requires the design's ERT to split the scratchpad's read and
                  write energy; refuses rather than estimating if it cannot.
    `free`        the historical accounting, kept only so the pre-2026-09-08
                  numbers can be reproduced for a diff: register writes are
                  charged, register reads are not, and scratchpad read counts
                  are Timeloop's. Internally inconsistent -- see above.

    Only the WRITE term is common to all three, and it is deliberately charged
    at the unscaled per-access energy for one write per weight reconstructed,
    which over-charges `complement` (its writes install 3.05 b, not 8 b).
    """
    mode: str
    weight_bits: int
    k: int
    n: int
    pj: float                      # per-access energy, the design's own ERT

    @property
    def frac(self):
        return self.k / self.n

    @property
    def bits_per_weight(self):
        """Register width per resident weight."""
        if self.mode == "complement":
            return self.weight_bits * (1.0 - self.frac)
        return float(self.weight_bits)

    @property
    def serves_reads(self):
        """Does a MAC read get its operand from the register instead of the SPad?"""
        return self.mode == "full_width"

    @property
    def read_pj(self):
        """Per-access read energy, scaled by the fraction of the word it holds."""
        if self.mode == "complement":
            return self.pj * (1.0 - self.frac)
        if self.mode == "full_width":
            return self.pj
        return 0.0

    def storage_bits_per_resident_weight(self, packing):
        """SPad bits + register bits per weight, the honest capacity figure."""
        return packing.reduced_bits_per_weight + self.bits_per_weight

    def to_dict(self, packing):
        total = self.storage_bits_per_resident_weight(packing)
        return {
            "mode": self.mode,
            "meaning": {
                "complement": ("holds only the n-k bits per weight that the "
                               "encoder regenerates, so it cannot serve a read "
                               "on its own and the SPad read count is unchanged"),
                "full_width": ("holds whole reconstructed weights and serves the "
                               "reads, so the SPad is read once per fill"),
                "free": ("historical: writes charged, reads not charged, SPad "
                         "read count unchanged -- internally inconsistent, kept "
                         "only to reproduce the pre-2026-09-08 numbers"),
            }[self.mode],
            "register_bits_per_weight": self.bits_per_weight,
            "reduced_spad_bits_per_weight": packing.reduced_bits_per_weight,
            "total_pe_bits_per_resident_weight": total,
            "vs_baseline_pe_weight_storage": total / self.weight_bits,
            "serves_spad_reads": self.serves_reads,
            "pJ_per_write": self.pj,
            "pJ_per_read": self.read_pj,
            "why_the_complement_is_n_minus_k": (
                "embedded.py: ParityOverwriteByTopWeightsEncode overwrites the "
                "n-k lowest-significance bits of each n-bit chunk of the weight "
                "bit stream with parity. On chip the reduced form keeps the k "
                "message bits; reconstruction regenerates those n-k bits. Only "
                "they have to be retained, which is weight_bits*(1-k/n) per "
                "weight, and reduced SPad + register is then exactly "
                "weight_bits per resident weight at every K"),
        }


#: `ReuseRegister.mode` values, for config validation.
REUSE_REG_MODES = ("complement", "full_width", "free")


def storage_access_pj(stats_dir, prefixes):
    """(read_pJ, write_pJ, provenance) for a storage level, from the design's ERT.

    Only `ReuseRegister(mode='full_width')` needs this: to move the scratchpad's
    read count from `reads` to `fills` the two per-access energies have to be
    separable, and Timeloop's stats print one `Energy (total)` per dataspace.
    The split is taken from the same Accelergy ERT the mapper cache already
    holds beside every mapping, and `evaluate_placement` cross-checks it against
    Timeloop's own total before using it. Returns `(None, None, why)` when the
    ERT is missing, so the caller can REFUSE rather than estimate.
    """
    ert = pathlib.Path(stats_dir) / "timeloop-mapper.ERT_summary.yaml"
    if not ert.exists():
        return None, None, f"no {ert.name} in the mapper cache"
    try:
        import yaml
        blob = yaml.safe_load(ert.read_text())
        for entry in (blob.get("ERT_summary", {}).get("table_summary") or []):
            name = str(entry.get("name", ""))
            # Strip the instance range BEFORE splitting on `.`: Accelergy writes
            # `system_top_level.weights_spad[1..192]`, and `[1..192]` contains
            # dots of its own, so splitting first leaves `192]`.
            bare = name.split("[")[0].split(".")[-1]
            if not any(bare == p or bare.startswith(p) for p in prefixes):
                continue
            acts = {str(a.get("name")): float(a.get("energy", 0.0))
                    for a in (entry.get("actions") or [])}
            if acts.get("read", 0.0) > 0 and acts.get("write", 0.0) > 0:
                return acts["read"], acts["write"], (
                    f"{ert.name}: {name} read={acts['read']} pJ "
                    f"write={acts['write']} pJ (Accelergy, this design's own ERT)")
        return None, None, f"no read+write energy for {prefixes} in {ert.name}"
    except Exception as exc:                                  # pragma: no cover
        return None, None, f"could not read {ert.name} ({type(exc).__name__})"


# ===========================================================================
#  reconstruction granularity and feasibility  (02_..., section 15)
# ===========================================================================
@dataclass(frozen=True)
class Granularity:
    """`G_rec` and how encoder work is charged.

    Section 15: "The encoder may operate at codeword granularity rather than
    independent weight granularity ... reconstructing one weight can require
    retained bits from several weights." Under the embedded layout the codeword
    is n consecutive bits of the weight bit stream, so:

    `weights_per_codeword`  n / weight_bits, fractional (7.875 at BCH(63,K)
                            over 8-bit weights) -- the same number the embedded
                            arm counts DRAM codewords with, so the two arms
                            cannot disagree about what a codeword is.
    `g_rec`                 the most weights one codeword TOUCHES. A codeword
                            starting mid-weight reaches into one more weight
                            than its length implies, so this is
                            ceil(n/weight_bits)+1 for the worst-aligned phase
                            (9 at (63, 8)) and it is what a PE must hold at once
                            for a local boundary to be feasible at all.

    Charging, `ECC_RECON_ENCODER_GRANULARITY`:

    `weight`    (default) encoder work is proportional to the weights actually
                reconstructed, charged at the synthesized per-codeword energy
                per `weights_per_codeword` of them. This is the AMORTIZED
                reading: the group is rebuilt once and every weight in it is
                consumed.
    `codeword`  every access at the boundary rebuilds one whole codeword,
                whether or not the rest of the group is used. The pessimistic
                reading, and the right one if nothing buffers the group.
    """
    n: int
    k: int
    weight_bits: int
    mode: str

    @property
    def weights_per_codeword(self):
        return embedded.EmbeddedLayout(self.n, self.k,
                                       self.weight_bits).weights_per_codeword

    @property
    def g_rec(self):
        layout = embedded.EmbeddedLayout(self.n, self.k, self.weight_bits)
        w = self.weight_bits
        worst = 0
        for phase in range(layout.period_codewords):
            start = phase * self.n
            first = start // w
            last = (start + self.n - 1) // w
            worst = max(worst, last - first + 1)
        return worst

    def codewords(self, accesses):
        """Reconstruction events for `accesses` weights crossing the boundary."""
        if self.mode == "codeword":
            return float(accesses)
        return float(accesses) / self.weights_per_codeword

    def to_dict(self):
        return {
            "charging": self.mode,
            "meaning": ("weight: encoder work is proportional to the weights "
                        "reconstructed, charged at the synthesized per-codeword "
                        "energy per n/weight_bits of them (the group is rebuilt "
                        "once and all of it is consumed). codeword: every access "
                        "at the boundary rebuilds a whole codeword."),
            "weights_per_codeword": self.weights_per_codeword,
            "G_rec_weights_touched_per_codeword": self.g_rec,
            "G_rec_meaning": ("weights whose retained bits must be available at "
                              "once to rebuild one ECC group; a codeword that "
                              "starts mid-weight reaches into one weight more "
                              "than its length implies"),
            "source": ("02_reconstruction_dse_and_implementation.txt section 15; "
                       "the layout is eccenergy/embedded.py's"),
        }


def feasibility(placement, arch, wpath, gran):
    """Can this boundary assemble a complete ECC group where it sits?

    Returns `(ok, detail)`. Three ways to fail, and each is reported rather than
    worked around:

    * the stage the encoder sits at is not in the model at all;
    * the stage it would reduce is not in the model;
    * the boundary is PE-local and a PE does not hold `G_rec` weights at once,
      so the retained bits the rebuild needs were never sent to that PE. This
      is the "reject infeasible local placements" the plan asks for, and it is
      checked per layer against the mapping's own utilized capacity.
    """
    missing_site = placement.site_stage not in wpath.stages or \
        wpath.stages[placement.site_stage].energy_pJ <= 0
    missing_reduced = [k for k in placement.reduced
                       if k not in wpath.stages or wpath.stages[k].energy_pJ <= 0]
    if missing_site:
        return False, {
            "reason": f"stage {placement.site_stage!r} carries no weight energy in "
                      f"{arch} as modelled, so there is no such boundary to "
                      f"evaluate",
            "stages_present": [k for k, s in wpath.stages.items() if s.energy_pJ > 0]}
    if missing_reduced:
        return False, {
            "reason": f"stage(s) {missing_reduced} would have to carry the reduced "
                      f"representation but carry no weight energy in {arch} as "
                      f"modelled",
            "stages_present": [k for k, s in wpath.stages.items() if s.energy_pJ > 0]}

    # Group assembly is only in question where the reduced form is STORED in a
    # PE: a boundary above the scratchpad reconstructs from a stream the memory
    # interface is feeding in codeword order, so the group is inherently whole.
    local = [k for k in placement.reduced
             if wpath.stages[k].kind == "storage"]
    if not local:
        return True, {"group_assembly": "not PE-local: the reduced form is only in "
                                        "transit, in codeword order from the ECC "
                                        "engine, so a whole group is always available",
                      "G_rec": gran.g_rec}
    bad = []
    for lp in wpath.per_layer:
        for key in local:
            st = lp["stages"].get(key)
            if st is None:
                continue
            cap = st.get("weights_resident_per_instance") or 0
            if cap < gran.g_rec:
                bad.append({"layer": lp["layer"], "shape": lp["shape"],
                            "stage": key, "weights_resident": cap,
                            "G_rec_required": gran.g_rec})
    ok = not bad
    detail = {
        "G_rec": gran.g_rec,
        "rule": ("a PE-local boundary needs G_rec weights of the codeword group "
                 "resident at once; a smaller resident tile means the retained "
                 "bits the rebuild depends on were never sent to that PE"),
        "checked_stages": local,
        "infeasible_layers": bad,
        "verdict": "feasible for every mapped layer" if ok else
                   f"{len(bad)} (layer, stage) pair(s) hold fewer than G_rec "
                   f"weights -- the placement is rejected rather than estimated",
    }
    if not ok:
        # Every unsupported return has to carry `reason`: it is what the result
        # store writes as `unavailable_reason`, and it refuses a variant that
        # cannot say why it is unavailable.
        names = ", ".join(sorted({b["layer"] for b in bad}))
        detail["reason"] = (
            f"infeasible local placement: {len(bad)} (layer, stage) pair(s) keep "
            f"fewer than G_rec = {gran.g_rec} weights resident, so the retained "
            f"bits this boundary would rebuild from were never delivered to that "
            f"PE. Layers: {names}")
    return ok, detail


# ===========================================================================
#  evaluating one placement
# ===========================================================================
@dataclass
class PlacementResult:
    placement: Placement
    status: str                 # evaluated | unsupported
    components: dict            # plotted category -> pJ
    total_pJ: float
    detail: dict
    reason: str = ""


def evaluate_placement(cfg, arch, placement, wpath, base_w_by_cat, base_by_cat,
                       recon_pj, reuse_reg_pj, gran, packing, decode_pj=0.0):
    """Cost one boundary. Fixed mapping: every access count is Timeloop's.

    `base_by_cat` / `base_w_by_cat` are the plotted-category totals and their
    weight share, from the SAME `Raw` record the baseline and embedded arms use.
    A stage's saving is subtracted from its own category, so the stack still
    sums to the total. The DRAM category moves by exactly the interface share's
    K/N reduction and no more: `dram_array` is never in a `reduced` set, and
    `dram_interface` is in every one of them when the decoder is on the die
    (and in none when `ECC_RECON_DECODE_SITE=controller` -- `placement` is
    passed through `effective_placement()` first, so a caller handing in the
    table's on-die form still gets the right row).
    """
    placement = effective_placement(placement, cfg)
    stage_defs = stages_for(arch, cfg)
    ok, feas = feasibility(placement, arch, wpath, gran)
    if not ok:
        return PlacementResult(placement, "unsupported", {}, 0.0,
                               {"feasibility": feas}, feas["reason"])

    # ---- 0. the reuse register: WHAT IT HOLDS decides what it may be charged
    # See `ReuseRegister`. This block is the 2026-09-08 fix: a register that
    # serves the reads must remove them from the scratchpad, and one that only
    # caches the n-k regenerated bits must be READ on every delivery. Either
    # way the register can no longer be charged a write and nothing else.
    reuse_reg = ReuseRegister(getattr(cfg, "recon_reuse_reg_model", "free"),
                              cfg.weight_bits, cfg.code_k, cfg.code_n,
                              reuse_reg_pj)
    energy_override, reg_notes, fw = {}, {}, None
    if placement.reuse_register and reuse_reg.serves_reads:
        # `full_width`: the MAC's operand comes from the register, so the
        # scratchpad is touched once per FILL, not once per MAC. Splitting its
        # read from its write energy needs the design's own ERT; if that is not
        # in the mapper cache this mode REFUSES rather than estimating.
        site = next((s for s in stage_defs
                     if s.key == placement.site_stage), None)
        st_site0 = wpath.stages.get(placement.site_stage)
        e_rd, e_wr, prov = storage_access_pj(wpath.stats_dir,
                                            site.prefixes if site else ())
        if e_rd is None or st_site0 is None:
            return PlacementResult(
                placement, "unsupported", {}, 0.0,
                {"feasibility": feas,
                 "reuse_register": reuse_reg.to_dict(packing),
                 "reason": prov},
                f"reuse register mode `full_width` needs the scratchpad's "
                f"read/write energy split: {prov}")
        wpw_full = packing.weights_per_word(st_site0.block_bits, reduced=False)
        wpw_red = packing.weights_per_word(st_site0.block_bits, reduced=True)
        # Cross-check the ERT against Timeloop's own total before trusting it.
        rebuilt = (st_site0.reads / wpw_full * e_rd
                   + st_site0.fills / wpw_full * e_wr)
        rel = abs(rebuilt - st_site0.energy_pJ) / st_site0.energy_pJ
        if rel > 1e-3:
            return PlacementResult(
                placement, "unsupported", {}, 0.0,
                {"feasibility": feas,
                 "reuse_register": reuse_reg.to_dict(packing),
                 "ert_reconciliation": {"from_ert_pJ": rebuilt,
                                        "timeloop_pJ": st_site0.energy_pJ,
                                        "relative_error": rel,
                                        "provenance": prov}},
                f"the ERT read/write split does not reproduce Timeloop's "
                f"{placement.site_stage} energy ({rel:.2%} off); refusing to "
                f"re-bill its reads from an unreconciled split")
        # One scratchpad read per weight filled (to feed the encoder), plus the
        # fill write itself -- both in the reduced form.
        energy_override[placement.site_stage] = (
            st_site0.fills / wpw_red * (e_rd + e_wr))
        reg_notes["spad_reads_moved_to_the_register"] = {
            "spad_scalar_reads_before": st_site0.reads,
            "spad_scalar_reads_after": st_site0.fills,
            "provenance": prov,
            "ert_reconciles_timeloop_to": rel,
        }
        fw = {"e_rd": e_rd, "e_wr": e_wr, "wpw_full": wpw_full, "st": st_site0}

    # ---- 1. what the reduced representation saves, stage by stage -----------
    saved_by_cat = {}
    stage_rows = []
    for stage in stage_defs:
        st = wpath.stages.get(stage.key)
        if st is None or st.energy_pJ <= 0:
            continue
        cat = _category_of(stage, cfg)
        reduced = stage.key in placement.reduced
        if not reduced:
            stage_rows.append({**st.to_dict(), "category": cat,
                               "carries_reduced": False, "scale": 1.0,
                               "energy_saved_pJ": 0.0})
            continue
        if stage.key in energy_override:
            after = energy_override[stage.key]
            scale = after / st.energy_pJ if st.energy_pJ else 1.0
        elif stage.kind == "dram":
            # The interface carries k of every n bits of the codeword STREAM
            # the die emits, so it scales by exactly K/N whatever the on-chip
            # packing does with the bits after ingress (`aligned` repacks on
            # chip; the die output is the embedded layout's packed stream).
            scale = packing.frac
            after = st.energy_pJ * scale
        elif stage.kind == "network":
            ws, ss = packing.wire_scale(), packing.switching_scale(st.block_bits)
            after = st.wire_pJ * ws + st.switch_pJ * ss
            scale = after / st.energy_pJ if st.energy_pJ else 1.0
        else:
            scale = packing.storage_scale(st.block_bits)
            after = st.energy_pJ * scale
        saved = st.energy_pJ - after
        saved_by_cat[cat] = saved_by_cat.get(cat, 0.0) + saved
        stage_rows.append({**st.to_dict(), "category": cat,
                           "carries_reduced": True, "scale": scale,
                           "energy_after_pJ": after, "energy_saved_pJ": saved})

    # ---- 2. what reconstruction costs --------------------------------------
    st_site = wpath.stages[placement.site_stage]
    ret = wpath.retention
    if placement.site_counter == "retained":
        # R4b: the count comes from the retention model, which is per layer --
        # a layer whose tile the register cannot cover contributes its READS
        # while the others contribute their FILLS.
        accesses = ret["weights_reconstructed_with_retention"]
    else:
        accesses = st_site.counter(placement.site_counter)
    n_cw = gran.codewords(accesses)
    recon_energy = n_cw * recon_pj

    # ---- 2b. what the reuse register costs ----------------------------------
    # WRITES: one per weight the encoder installs, at the unscaled per-access
    # energy. Deliberately conservative under `complement`, whose writes install
    # only weight_bits*(1-k/n) bits.
    #
    # READS: charged from 2026-09-08, and this is the audit fix. Before that
    # date the register was charged a write and nothing else, on the argument
    # that its reads "replace scratchpad reads that are still billed". That
    # argument only holds if the register does NOT serve the reads -- and R4b's
    # whole reconstruction saving comes from claiming it does. `ReuseRegister`
    # documents the resolution; the short version is that a register holding
    # only the n-k regenerated bits cannot serve a read alone, so the SPad read
    # count stays Timeloop's AND the register is read alongside it, in lockstep
    # (one register access per SPad word access, not per scalar weight).
    reg_writes = accesses if placement.reuse_register else 0.0
    reg_reads = 0.0
    if placement.reuse_register and reuse_reg.read_pj > 0:
        if reuse_reg.serves_reads:
            # the register IS the operand source: one access per weight delivered
            reg_reads = st_site.reads
        else:
            # lockstep with the scratchpad: one access per reduced word read
            wpw = packing.weights_per_word(st_site.block_bits, reduced=True)
            reg_reads = st_site.reads / wpw if wpw > 0 else st_site.reads
    overhead = reg_writes * reuse_reg.pj + reg_reads * reuse_reg.read_pj

    # THE NUMBER THE `full_width` READING TURNS ON. If a whole-weight operand
    # cache is what R4b needs, then the SAME cache bolted onto a PE with NO ECC
    # at all would remove the same 1.81 G scratchpad reads -- and if that
    # ECC-free saving is most of R4b's headline, the headline is a memory-
    # hierarchy change wearing an ECC label. That is precisely the audit's
    # claim, so it is COMPUTED here rather than argued about. Read
    # `ecc_marginal_*` as the only part reconstruction can take credit for.
    #
    # Note this is a diagnostic computed inside recon.py, NOT a fourth bar: the
    # Task 1 baseline code is frozen and no register is added to it. What it
    # says under this design's own ERT is that the answer depends entirely on
    # what a register access is priced at -- at `ifmap_spad`'s 0.0328 pJ (a
    # 192-bit array) the register alone looks worth ~6%, and at `weights_spad`'s
    # own read energy (a 2304-bit array, which is what a 2048-bit register
    # actually is) it is worth about nothing.
    if fw is not None:
        others = sum(st.energy_pJ for k, st in wpath.stages.items()
                     if k in placement.reduced and k != placement.site_stage)
        onchip_before = others + fw["st"].energy_pJ
        base_reg = (fw["st"].fills / fw["wpw_full"] * (fw["e_rd"] + fw["e_wr"])
                    + others
                    + fw["st"].fills * reuse_reg.pj
                    + fw["st"].reads * reuse_reg.read_pj)
        r4b_onchip = (onchip_before - sum(saved_by_cat.values())
                      + recon_energy + overhead)
        reg_notes["what_the_same_register_gives_a_pe_with_no_ecc"] = {
            "embedded_no_register_onchip_weight_pJ": onchip_before,
            "baseline_plus_the_same_register_onchip_weight_pJ": base_reg,
            "the_register_alone_saves_pJ": onchip_before - base_reg,
            "r4b_register_plus_ecc_onchip_weight_pJ": r4b_onchip,
            "ecc_marginal_on_top_of_the_register_pJ": base_reg - r4b_onchip,
            "pJ_per_register_access_assumed": reuse_reg.read_pj,
            "note": ("a 256x8b register is 2048 bits against weights_spad's own "
                     "96x24b = 2304 bits, i.e. the same array -- so pricing its "
                     "accesses at ifmap_spad's 192-bit energy is the assumption "
                     "this row exists to expose. Set ECC_RECON_REUSE_REG_PJ to "
                     "weights_spad's per-weight read energy to price it by its "
                     "own size; `the_register_alone_saves_pJ` then goes to about "
                     "zero, because a 256-entry register file cannot beat the "
                     "288-entry register file it sits in front of"),
        }

    # ---- 3. assemble the stack ---------------------------------------------
    components = {c: float(v) for c, v in base_by_cat.items()}
    for cat, saved in saved_by_cat.items():
        components[cat] = components.get(cat, 0.0) - saved
    components["Reconstruction"] = components.get("Reconstruction", 0.0) + recon_energy
    components["Recon overhead"] = components.get("Recon overhead", 0.0) + overhead
    components["ECC decode"] = components.get("ECC decode", 0.0) + decode_pj
    total = float(sum(components.values()))

    detail = {
        "placement": {
            "key": placement.key, "boundary": placement.label,
            "hypothesised_rating": placement.rating,
            "description": placement.description,
            "reduced_stages": list(placement.reduced),
            "reconstruction_site": {
                "stage": placement.site_stage, "counter": placement.site_counter,
                "reuse_register": placement.reuse_register},
        },
        "feasibility": feas,
        "reduced_representation": packing.to_dict(),
        "reconstruction_granularity": gran.to_dict(),
        "reconstruction_counts": {
            "weights_reconstructed": accesses,
            "counter": placement.site_counter,
            "counter_meaning": {
                "reads": "one reconstruction per weight read out of the stage",
                "fills": "one reconstruction per weight written into the stage",
                "ingresses": "one reconstruction per weight word transported",
                "retained": ("one reconstruction per weight per buffer FILL "
                             "where the reuse register covers the tile the "
                             "loops walk, per read where it does not -- see "
                             "reconstructed_weight_retention"),
            }[placement.site_counter],
            "amortization_vs_no_retention": (
                ret["amortization_vs_no_retention"]
                if placement.site_counter == "retained" else 1.0),
            "reconstruction_events_codewords": n_cw,
            "pJ_per_codeword": recon_pj,
            "reconstruction_energy_pJ": recon_energy,
            "reuse_register_writes": reg_writes,
            "pJ_per_reuse_register_write": (reuse_reg.pj if placement.reuse_register
                                            else 0.0),
            "reuse_register_reads": reg_reads,
            "pJ_per_reuse_register_read": (reuse_reg.read_pj
                                           if placement.reuse_register else 0.0),
            "reuse_register_write_energy_pJ": reg_writes * reuse_reg.pj,
            "reuse_register_read_energy_pJ": reg_reads * reuse_reg.read_pj,
            "reuse_register_energy_pJ": overhead,
            "reuse_register_model": (reuse_reg.to_dict(packing)
                                     if placement.reuse_register else None),
            "reuse_register_notes": reg_notes or None,
            "reuse_register_entries_required": (
                ret["register_entries_required_max"]
                if placement.reuse_register else 0),
            "reuse_register_bits_required": (
                ret["register_entries_required_max"] * reuse_reg.bits_per_weight
                if placement.reuse_register else 0),
        },
        "reducible_weight_energy_pJ": {
            "stages_left_reduced": list(placement.reduced),
            "before_pJ": wpath.reducible_energy(placement.reduced),
            "ceiling_on_the_saving_pJ": (
                wpath.reducible_energy(placement.reduced) * (1.0 - packing.frac)),
            "note": ("the most this boundary could give back if reconstruction "
                     "were free; the DRAM interface share plus the whole "
                     "on-chip weight path is the ceiling for ANY boundary, and "
                     "on this architecture the on-chip part is a small share "
                     "of inference energy"),
        },
        "weight_path_stages": stage_rows,
        "energy_saved_by_category_pJ": saved_by_cat,
        "weight_energy_before_pJ": {c: float(v) for c, v in base_w_by_cat.items()},
        "dram_model": dram_model_detail(wpath, placement, stage_rows, packing),
    }
    return PlacementResult(placement, "evaluated", components, total, detail)


def dram_model_detail(wpath, placement, stage_rows, packing):
    """What this placement did to the two DRAM stages, as a record.

    Replaces the pre-2026-09-09 `dram_unchanged` string. Both checks in
    experiments/recon.py (`dram_array_identical_to_embedded_reference`,
    `dram_interface_scaled_by_K_over_N`) read these rows AND the DRAM
    component, so a bar cannot report one thing here and draw another.
    """
    rows = {r["stage"]: r for r in stage_rows if r["kind"] == "dram"}
    array = rows.get("dram_array", {})
    iface = rows.get("dram_interface", {})
    a_before = array.get("weight_energy_pJ", 0.0)
    i_before = iface.get("weight_energy_pJ", 0.0)
    i_after = iface.get("energy_after_pJ", i_before)
    reduced = "dram_interface" in placement.reduced
    return {
        "decode_site": wpath.decode_site,
        "f_if": wpath.dram_if_frac,
        "f_if_provenance": wpath.dram_if_note,
        "dram_weight_energy_pJ": a_before + i_before,
        "dram_array_pJ": a_before,
        "dram_array_after_pJ": array.get("energy_after_pJ", a_before),
        "dram_array_reduced": bool(array.get("carries_reduced", False)),
        "dram_interface_pJ": i_before,
        "dram_interface_after_pJ": i_after,
        "dram_interface_reduced": reduced,
        "dram_interface_scale": packing.frac if reduced else 1.0,
        "dram_interface_saving_pJ": i_before - i_after,
        "rule": (("the decoder is on the DRAM die and off the fetch path: the "
                  "array reads the complete codeword (never reduced) and only "
                  "the k message bits cross the interface (x K/N), under every "
                  "boundary") if wpath.decode_site == "ondie" else
                 ("controller-side correction (pre-2026-09-09 model): the "
                  "complete codeword crosses the interface, so neither DRAM "
                  "share is reduced and the DRAM term is the embedded arm's, "
                  "to the digit")),
    }


def _category_of(stage, cfg):
    """The plotted category a weight-path stage lands in.

    Deliberately NOT `timeloop.classify()` on a level name: this has to agree
    with the categories the `Raw` record already holds, and the mapping from a
    stage to a category is a property of the stage, so it is stated here and
    checked against the record before anything is evaluated.
    """
    if stage.kind == "dram":
        return "DRAM"
    if stage.kind == "network":
        return "NoC"
    if stage.key.endswith("_glb"):
        return "Global buffer (read)" if cfg.split_read_write else "Global buffer"
    return "Local (read)" if cfg.split_read_write else "Local (spads/RF)"


def cross_check(cfg, arch, wpath, base_w_by_cat, tol=1e-6):
    """Every weight-path stage total must reconcile with the `Raw` record.

    The placement model subtracts savings from the categories the `Raw` record
    holds, so a stage whose energy this module measured differently from the
    record would move a bar by an amount that is nowhere in Timeloop's output.
    Returns `(ok, detail)` and is recorded as a check on every result.
    """
    mine = {}
    for stage in stages_for(arch, cfg):
        st = wpath.stages.get(stage.key)
        if st is None or st.energy_pJ <= 0:
            continue
        cat = _category_of(stage, cfg)
        mine[cat] = mine.get(cat, 0.0) + st.energy_pJ
    rows, ok = {}, True
    for cat in sorted(set(mine) | {c for c, v in base_w_by_cat.items()
                                   if float(v) > 0}):
        a, b = mine.get(cat, 0.0), float(base_w_by_cat.get(cat, 0.0))
        rel = abs(a - b) / b if b else (0.0 if a == 0 else 1.0)
        rows[cat] = {"weight_path_module_pJ": a, "raw_record_pJ": b,
                     "relative_difference": rel}
        if rel > 1e-9:
            ok = False
    return ok, {
        "per_category": rows,
        "unclaimed_weight_levels": wpath.unclaimed,
        "dram_split": wpath.dram_split(),
        "rule": ("this module re-parses the cached stats and must reproduce the "
                 "`Raw` record's WEIGHT energy per category exactly; a level "
                 "carrying weight energy that no stage claims is listed above "
                 "and makes this check fail. The DRAM category is the SUM of "
                 "dram_array and dram_interface, whose shares of the one "
                 "Timeloop level are (1 - f_if) and f_if"),
    }


def reuse_register_pj(stats_dir, fallback=0.0328125):
    """Per-access energy of the reconstructed-weight register, from the design's ERT.

    R4b needs a small register after the encoder. Rather than invent a number,
    take the design's OWN Accelergy ERT (written into the mapper cache beside
    every mapping) and use the cheapest per-PE register write in it. For
    `eyeriss_v2_like` that is the 24 x 8b `ifmap_spad` at 0.0328 pJ.

    HOW CONSERVATIVE THAT IS DEPENDS ON THE MODE, and the docstring used to get
    this backwards -- it called 0.0328 pJ conservative because "a one-entry
    latch costs less than a 24-entry register file", but R4b needs a register
    sized to the working set (16-256 entries here), not a latch.

    * `complement` (the default): the register is `entries x weight_bits x
      (1-k/n)` = 390 bits at 256 entries and BCH(63,51), 780 at BCH(63,39) --
      2x-4x `ifmap_spad`'s 192 bits, and each access is charged only
      `(1-k/n)` of this energy while covering several weights at once. Roughly
      fair, and the write term is over-charged (billed unscaled, per weight).
    * `full_width`: the register is `entries x weight_bits` = 2048 bits at 256
      entries, which is 10.7x `ifmap_spad` and 0.89x `weights_spad`'s own
      96x24b = 2304 bits. Pricing THAT at `ifmap_spad`'s energy is not
      conservative, it is the assumption the mode exists to expose: an array
      the size of the scratchpad costs what the scratchpad costs, and at
      `weights_spad`'s 0.596 pJ per 24-bit word the register stops being worth
      anything at all. `ECC_RECON_REUSE_REG_PJ` overrides it for exactly this
      sensitivity.

    Returns `(pJ, provenance)`.
    """
    ert = pathlib.Path(stats_dir) / "timeloop-mapper.ERT_summary.yaml"
    if not ert.exists():
        return fallback, (f"no ERT in the mapper cache; fallback constant "
                          f"{fallback} pJ per write")
    try:
        import yaml
        blob = yaml.safe_load(ert.read_text())
        best, who = None, None
        for entry in (blob.get("ERT_summary", {}).get("table_summary") or []):
            name = str(entry.get("name", ""))
            m = re.search(r"\[1\.\.(\d+)\]$", name)
            if not m or int(m.group(1)) <= 1:
                continue          # a shared buffer, not a per-PE register
            for action in (entry.get("actions") or []):
                if action.get("name") != "write":
                    continue
                e = float(action.get("energy", 0.0))
                if e > 0 and (best is None or e < best):
                    best, who = e, name
        if best is None:
            return fallback, (f"no per-PE write energy in {ert.name}; fallback "
                              f"constant {fallback} pJ per write")
        return best, (f"{ert.name}: cheapest per-PE register write, "
                      f"{who} = {best} pJ (Accelergy, this design's own ERT). "
                      f"Conservative: a one-entry latch costs less than the "
                      f"register file this is taken from.")
    except Exception as exc:                                  # pragma: no cover
        return fallback, (f"could not read {ert.name} ({type(exc).__name__}); "
                          f"fallback constant {fallback} pJ per write")
