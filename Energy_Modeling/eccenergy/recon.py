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

The DRAM is ONE stage, and the whole of it is reducible (changed 2026-09-09).
Accelergy's CactiDRAM bills a DRAM read as one flat per-bit DYNAMIC ACCESS
constant -- 8 pJ/bit for LPDDR4 as modelled, verified at 64.0 pJ per 8-bit
word -- and Timeloop charges the DRAM-to-chip network zero, so that constant
is the entire DRAM term:

    dram = DRAM weight energy x K/N   under EVERY boundary, R1 included

It used to be TWO stages, `dram_array` ((1 - f_if) of it, never reduced) and
`dram_interface` (f_if of it, reduced), with `f_if = ECC_DRAM_IF_FRAC = 0.40`.
That split charged a 38.1% bit cut as a 15.2% energy cut and is REMOVED: the
DRAM access is designed to fetch only the message bits of each codeword, so the
whole per-bit constant carries k of every n bits.
`ECC_DRAM_PJ_PER_BIT` sets the per-bit cost itself (energy.apply_dram_override);
`archs/_shared/provenance.yaml` `dram_access_energy` carries the citations.
E_background and E_refresh are NOT modelled (both 0) -- see env.sh section 4.

THE DECODER IS ON THE DRAM DIE  (since 2026-09-09)
-------------------------------------------------
01_project_context Sec. 1: the BCH decoder sits on the DRAM die and OFF the
fetch path (it corrects at write, on a scrub pass or on a prior access), so at
fetch time the stored codeword is already corrected and only the k message
bits of each n-bit codeword are driven across the DRAM interface. The array
still stores and reads the complete codeword -- a row activation and a burst
move whole words -- so `dram` is reduced
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

R4b -- SPad output plus a reconstructed-weight reuse register -- was REMOVED on
2026-09-10. Measured on the cached mappings, consecutive weight reuse is 1 on
20 of 21 resnet18 layers, so a latch-sized register catches nothing (a cyclic
walk is the LRU worst case, with no partial hit rate); a register that DOES pay
has to hold the whole inner tile, up to 384 weights against a 384-weight
scratchpad. The same argument rules it out on every dataflow in the study:
output- and input-stationary hold the psum or the activation in the PE, so
weights stream past the MAC faster still. FINDINGS 7.1.

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
* The WHOLE DRAM term IS credited, by exactly K/N: only the k message bits of
  each codeword are read out and driven off the die, so `dram` = the DRAM
  weight energy x K/N under every boundary (`dram_scaled_by_K_over_N` fails if
  a bar leaves it at full width). It is the same saving on every bar, so it
  moves every placement against the embedded reference by the same amount and
  changes no ordering among them.
* E_background and E_refresh are not modelled (both 0). An arm that stores
  fewer weight bits in DRAM would save both, so this understates the embedded
  and reconstruction arms alike.
* The DECODER is still outside the comparison. It is the same BCH decoder
  relocated to the DRAM die, runs once per corrected codeword off the fetch
  path, and is DRAM-process logic; it cancels between the embedded bar and
  every placement exactly as the controller-side codec did (see `CODEC_NOTE`
  in experiments/recon.py).
* No capacity-driven reuse and no re-tiling: the mapping is fixed, so every
  access count is Timeloop's. Packed reduced weights would raise the effective
  SPad capacity -- that is Task 4's question, and crediting it here would mix
  the two answers.
* No operand-retention saving for anybody. Scratchpad read counts stay
  Timeloop's under every arm, so the K/N discount on them is real rather than
  double-counted. No boundary carries a per-PE reuse register (R4b, removed
  2026-09-10), so `Recon overhead` is structurally zero.
"""
from __future__ import annotations

import math
import pathlib
import re
from dataclasses import dataclass, field, replace

from . import code_widths, embedded


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

    def matches(self, level):
        return any(level == p or level.startswith(p) for p in self.prefixes)


#: Eyeriss v2 as modelled by `archs/eyeriss_v2_like/`, outer to inner. The
#: design has NO weight GLB (its GLB banks are iact and psum, JETCAS 2019
#: Table II), so the weight path goes straight from the memory interface into
#: the hierarchical mesh -- which is why there is no global-buffer boundary in
#: `PLACEMENTS` either.
_EYERISS_V2_PATH = (
    Stage("dram", "DRAM (dynamic access, whole term reducible)", "dram",
          ("DRAM",), reducible=True,
          evidence="the decoder is on the DRAM die and off the fetch path, so "
                   "only the k message bits of each n-bit codeword are read out "
                   "and driven off the die: the whole per-bit DYNAMIC access "
                   "constant carries k of every n bits under every boundary "
                   "(01_project_context Sec. 1 and Sec. 4). The DRAM access "
                   "is designed to fetch only the message bits of each "
                   "codeword. The f_if array/interface split was removed "
                   "2026-09-09."),
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
#: The DRAM is ONE stage since 2026-09-09, so the weight GLB is inserted
#: after index 1, not 2. Sliced by name rather than a literal so a future
#: change to the DRAM stages cannot silently reorder this path again.
_V2_DRAM_N = sum(1 for _s in _EYERISS_V2_PATH if _s.kind == "dram")
_EYERISS_V2_WGLB_PATH = _EYERISS_V2_PATH[:_V2_DRAM_N] + (
    Stage("weight_glb", "Weight global buffer", "storage", ("weight_glb",),
          reducible=True,
          evidence="the variant's extra weight level; NOT in the v2 paper -- see "
                   "archs/eyeriss_v2_like_wglb/README.md"),
) + _EYERISS_V2_PATH[_V2_DRAM_N:]

#: `eyeriss_like` -- Eyeriss v1 as modelled by `archs/eyeriss_like/`, outer to
#: inner. JSSC 2017 allocates 8 kB of the 108 kB GLB to filter weights, but the
#: paper is explicit that the RS dataflow does not need it ("even though it is
#: not required by the dataflow ... the GLB preloads the filters used by the
#: next processing pass"), so `eyeriss_like/arch_paper.yaml` models the weight
#: path as DRAM -> filter spad with a `!Nothing` node where a weight GLB would
#: sit, and `eyeriss_like_wglb` is the bracketing variant that does model it
#: (CLAUDE.md: "Eyeriss v1 is a bracketing PAIR, not a number"). There is
#: therefore NO global-buffer boundary here, exactly as there is none on v2.
#:
#: Sec. 7's abstraction also names a PE pFIFO between the network and the
#: scratchpad (Sec. 7.2 rates a FIFO-output boundary 3/5). The Timeloop model
#: has no such level -- the fanout delivers straight into `weights_spad` -- so
#: that boundary does not exist to evaluate and is absent by construction
#: rather than omitted; `weight_path()` would refuse if a weight-carrying level
#: went unclaimed.
_EYERISS_V1_PATH = (
    Stage("dram", "DRAM (dynamic access, whole term reducible)", "dram",
          ("DRAM",), reducible=True,
          evidence="the decoder is on the DRAM die and off the fetch path, so "
                   "only the k message bits of each n-bit codeword are read out "
                   "and driven off the die: the whole per-bit DYNAMIC access "
                   "constant carries k of every n bits under every boundary "
                   "(01_project_context Sec. 1 and Sec. 4). The DRAM access "
                   "is designed to fetch only the message bits of each "
                   "codeword. The f_if array/interface split was removed "
                   "2026-09-09."),
    Stage("array_multicast", "Flat array NoC (multicast across PE columns)",
          "network", ("NoC: inter_PE_column_spatial",), reducible=True,
          evidence="JSSC 2017 Sec. V: 168 PEs in a 12 x 14 array fed by a "
                   "multicast network. `archs/eyeriss_like/arch_paper.yaml` "
                   "declares PE_column with meshX 14, so this is Timeloop's "
                   "outer fanout level -- the long-distance half of Sec. 7.1's "
                   "'flat multicast NoC'."),
    Stage("pe_local_multicast", "Column-local distribution to the 12 PEs",
          "network", ("NoC: inter_PE_spatial",), reducible=True,
          evidence="the inner fanout level of the same array network: PE with "
                   "meshY 12, one column of the 12 x 14 array. Sec. 7.1's "
                   "multicast tradeoff -- one encoder before the fanout, or one "
                   "per destination after it -- is the R2/R3 pair below."),
    Stage("weights_spad", "PE filter scratchpad", "storage", ("weights_spad",),
          reducible=True,
          evidence="JSSC 2017 Sec. V-B: 'the filter spad is implemented in a "
                   "224-b x 16-b SRAM due to its large size'. This is the "
                   "innermost level holding weights: the MAC reads it directly, "
                   "with no decoded-weight register in the design -- so a "
                   "retained-reconstruction boundary here would have been a "
                   "pure ADDITION, which is part of why R4b was removed."),
)

#: `eyeriss_like_wglb` -- EYERISS v1 (decided 2026-09-10; see prompt_2.md and
#: CLAUDE.md). JSSC 2017 Sec. V-A publishes the 8 kB filter-weight allocation
#: of the 108 kB GLB, so the file that models it IS the design, and
#: `eyeriss_like` -- which declares `!Nothing` in its place -- is retired
#: rather than bracketed. The bracketing-pair doctrine for v1 is over.
#:
#: THE ONE STRUCTURAL DIFFERENCE from `_EYERISS_V1_PATH` is a weight level
#: ABOVE the PE array, and that is exactly the level this study needs.
#: FINDINGS 7.8: refetch on v1 is set by the DRAM-level loop order over
#: `P`/`Q`, and a weight tile cannot index `P` or `Q`, so a weight buffer
#: INSIDE the array cannot absorb those loops however large it is made --
#: measured flat to x32 capacity at 1 % fill. `filter_glb` sits above the
#: array and can.
#:
#: Inserted after the DRAM stages by name, not at a literal index, so a future
#: change to the DRAM stages cannot silently reorder this path -- the same
#: guard `_EYERISS_V2_WGLB_PATH` uses.
_V1_DRAM_N = sum(1 for _s in _EYERISS_V1_PATH if _s.kind == "dram")
_EYERISS_V1_WGLB_PATH = _EYERISS_V1_PATH[:_V1_DRAM_N] + (
    Stage("filter_glb", "Filter global buffer (8 kB of the 108 kB GLB)",
          "storage", ("filter_glb",), reducible=True,
          evidence="JSSC 2017 Sec. V-A: 'Even though it is not required by the "
                   "dataflow, the remaining 8 kB (two banks of 512-b x 64-b "
                   "SRAMs) of the GLB is allocated for filter weights to "
                   "compensate for insufficient off-chip traffic bandwidth. "
                   "While the PE array is working on a processing pass, the "
                   "GLB preloads the filters used by the next processing "
                   "pass.' The bank geometry is exact (2 x 512 x 64b), so it "
                   "is the same CACTI array as the other 23 banks. It keeps "
                   "Weights and nothing else, so every pJ read from it is "
                   "weight energy."),
) + _EYERISS_V1_PATH[_V1_DRAM_N:]

#: `simple_weight_stationary` as modelled by `archs/simple_weight_stationary/`,
#: outer to inner. Sec. 6's canonical WS hierarchy is
#:
#:     DRAM -> global/weight buffer -> weight NoC -> PE weight RF
#:          -> stationary weight -> MAC
#:
#: and this design has every one of those levels, which makes it the only
#: design in the study with BOTH a weight global buffer and a stationary weight
#: register. So it has two boundaries neither eyeriss design has: one at the
#: global buffer's output (Sec. 6.2, 3/5) and one below the stationary register
#: at the MAC input (Sec. 6.2, 2/5).
_WS_PATH = (
    Stage("dram", "DRAM (dynamic access, whole term reducible)", "dram",
          ("DRAM",), reducible=True,
          evidence="the decoder is on the DRAM die and off the fetch path, so "
                   "only the k message bits of each n-bit codeword are read out "
                   "and driven off the die: the whole per-bit DYNAMIC access "
                   "constant carries k of every n bits under every boundary "
                   "(01_project_context Sec. 1 and Sec. 4). The DRAM access "
                   "is designed to fetch only the message bits of each "
                   "codeword. The f_if array/interface split was removed "
                   "2026-09-09."),
    Stage("operand_glb", "Global operand buffer (weight share)", "storage",
          ("operand_glb",), reducible=True,
          evidence="Sec. 6's 'global/weight buffer'. "
                   "`archs/simple_weight_stationary/arch_paper.yaml` splits the "
                   "stock 128 kB shared_glb by dataspace, and the 64 kB operand "
                   "half keeps Inputs AND Weights; only its WEIGHT energy is "
                   "read here, from the level's own Weights block, so reducing "
                   "this stage cannot touch the activation share (checked by "
                   "`non_weight_energy_identical`)."),
    Stage("weight_noc", "Weight-distribution NoC to the PE array", "network",
          ("NoC: inter_PE_spatial",), reducible=True,
          evidence="Sec. 6's 'weight-distribution NoC'. The design declares ONE "
                   "spatial container (PE, meshX 16 x meshY 16), so Timeloop "
                   "prints one fanout level and one network for the whole "
                   "256-PE broadcast -- unlike the two-level eyeriss meshes, "
                   "this design has a single network stage."),
    Stage("pe_spad", "PE weight scratchpad (stationary weights)", "storage",
          ("pe_spad",), reducible=True,
          evidence="Sec. 6's 'PE weight RF': the 192 x 16b scratchpad that "
                   "holds the stationary weights while activations stream. It "
                   "keeps Weights only (`dataspace: {keep: [Weights]}`), so "
                   "every bit of its energy is weight energy."),
    Stage("weight_reg", "Stationary weight register (one weight)", "storage",
          ("weight_reg",), reducible=True,
          evidence="Sec. 6.1's 'full stationary-weight latch', and it is ALREADY "
                   "IN THE DESIGN -- a depth-1, 8-bit register between the "
                   "scratchpad and the MAC. CLAUDE.md's rule for Simba applies "
                   "here too: determine whether the proposal can reuse an "
                   "existing register rather than assuming a new one. R4b's "
                   "register would have had to cover the whole inner tile, "
                   "not one weight, which is why R4b was removed; and "
                   "a boundary BELOW this register (Sec. 6.2's MAC-input row, "
                   "2/5) needs G_rec weights co-resident in a level that holds "
                   "one, so it is reducible in the table and reported "
                   "infeasible by `feasibility()` rather than asserted "
                   "impossible in a comment."),
)


#: Where the BCH decoder sits. `ondie` is the model since 2026-09-09 (decoder
#: on the DRAM die, off the fetch path, `dram` reducible under every
#: boundary); `controller` is the pre-2026-09-09 model kept as a runnable row
#: for the diff (`dram` not reducible, DRAM identical on every bar).
#: `config.RECON_DECODE_SITES` must stay in step with this tuple.
DECODE_SITES = ("ondie", "controller")

#: WHERE A NETWORK BOUNDARY'S ENCODERS SIT, and therefore how many times they
#: run. 02_.../01_project_context Sec. 7.1 states the tradeoff and asks for it
#: to be an experiment variable:
#:
#:     BEFORE MULTICAST   4b -> Encoder -> 8b -+-> PE  (x fanout)
#:         one reconstruction at the source, but FULL-WIDTH network traffic
#:     AFTER MULTICAST    4b -+-> Encoder -> PE  (x fanout)
#:         replicated encoders, but REDUCED-WIDTH shared transport
#:
#: `destination` (the model since 2026-09-09) charges the second: an encoder at
#: each destination, so the count is Timeloop's destination-side ARRIVALS,
#: `ingresses x multicast factor`. That is the only count consistent with a
#: boundary that also credits the network with carrying the reduced form -- an
#: encoder placed BEFORE the fanout would make that network full width, which
#: is the boundary above it.
#: `source` charges the first: one encoder before the fanout, count =
#: `ingresses`. It is kept as a runnable row because it is what the study
#: charged before 2026-09-09, so the change can be diffed; on its own it pairs
#: a source-side encoder count with a destination-side network saving.
#: `config.RECON_ENCODER_SITES` must stay in step with this tuple.
ENCODER_SITES = ("destination", "source")

WEIGHT_PATHS = {
    "eyeriss_v2_like": _EYERISS_V2_PATH,
    "eyeriss_v2_like_wglb": _EYERISS_V2_WGLB_PATH,
    "eyeriss_like": _EYERISS_V1_PATH,
    "eyeriss_like_wglb": _EYERISS_V1_WGLB_PATH,
    "simple_weight_stationary": _WS_PATH,
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
        ingresses   network ingresses -- words INJECTED into that network, so
                    one encoder before the fanout
        deliveries  network destination-side ARRIVALS, `ingresses x multicast
                    factor`, so one encoder at each destination. This is what a
                    boundary that credits the network with carrying the reduced
                    form has to pay (Sec. 7.1's "after multicast" case); the
                    two differ by the multicast factor, which is 7 on some
                    eyeriss shapes. `ECC_RECON_ENCODER_SITE=source` swaps it
                    back to `ingresses` for the diff -- see `ENCODER_SITES`.
    """
    key: str                  # the env.sh spelling: recon1 .. recon5
    variant: str              # the results-store variant name (baseline.py)
    label: str
    short: str                # figure label, may contain a newline
    rating: str               # the source discussion's hypothesis, not a result
    reduced: tuple
    site_stage: str
    site_counter: str
    description: str


_V2_PLACEMENTS = (
    Placement(
        "recon1", "recon_source_noc_ingress",
        "R1 - reconstruct at the source / weight-NoC ingress",
        "R1\n@ NoC source", "2/5",
        reduced=("dram",), site_stage="dram",
        site_counter="reads",
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
        reduced=("dram", "inter_cluster_mesh"),
        site_stage="inter_cluster_mesh",
        site_counter="deliveries",
        description=(
            "The long-distance hierarchical mesh carries the reduced form and "
            "an encoder at each destination cluster restores it before the "
            "cluster-local fanout. One encoder per cluster rather than one per "
            "PE, and the mesh -- the expensive hop -- moves fewer bits. The "
            "encoders run once per word ARRIVING at a cluster, not once per "
            "word injected into the mesh: the mesh multicasts, so those two "
            "counts differ by its multicast factor, and only the arrival count "
            "is consistent with the mesh carrying reduced-width data at all "
            "(an encoder before the fanout is R1). Sec. 7.1's multicast "
            "tradeoff; `ECC_RECON_ENCODER_SITE` is the experiment variable it "
            "asks for."),
    ),
    Placement(
        "recon3", "recon_pe_spad_input",
        "R3 - reconstruct at the PE weight-SPad input",
        "R3\n@ SPad input", "4/5",
        reduced=("dram", "inter_cluster_mesh", "cluster_local"),
        site_stage="weight_spad", site_counter="fills",
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
        reduced=("dram", "inter_cluster_mesh", "cluster_local",
                 "weight_spad"),
        site_stage="weight_spad", site_counter="reads",
        description=(
            "The scratchpad stores the reduced form, so its write and read "
            "bit-volume fall too, and the encoder sits at its read port. The "
            "risk the source discussion names is the whole story here: the "
            "encoder now runs once per weight DELIVERED to the datapath, and a "
            "weight-stationary inner loop delivers each resident weight "
            "thousands of times."),
    ),
)

#: Eyeriss v1, Sec. 7.1/7.2. The same five-boundary shape as v2 -- two network
#: stages then the PE scratchpad -- so the two eyeriss panels of one figure read
#: left to right as the same story about a different design. Ratings are Sec.
#: 7.2's and Sec. 11's HYPOTHESES, carried so the result can be read against
#: them; they are not results.
_EYERISS_V1_PLACEMENTS = (
    Placement(
        "recon1", "recon_source_noc_ingress",
        "R1 - reconstruct at the source, before the array network",
        "R1\n@ source", "3/5",
        reduced=("dram",), site_stage="dram",
        site_counter="reads",
        description=(
            "Sec. 7.1's source-side reconstruction. The die drives only the k "
            "message bits across the DRAM interface and an encoder at the chip "
            "source restores them before the array network, so nothing on chip "
            "carries the reduced form. It is the control that isolates the "
            "DRAM-interface saving -- which every boundary shares -- from every "
            "on-chip saving, and it pays one reconstruction per codeword "
            "fetched from DRAM with nothing but that interface saving against "
            "it."),
    ),
    Placement(
        "recon2", "recon_after_array_multicast",
        "R2 - reconstruct after the array multicast, at the column edge",
        "R2\n@ column edge", "4/5",
        reduced=("dram", "array_multicast"),
        site_stage="array_multicast", site_counter="deliveries",
        description=(
            "Sec. 7.1's multicast tradeoff, resolved in favour of reduced-width "
            "shared transport: the 14-way column multicast carries the reduced "
            "form and one encoder per column restores it before the column's "
            "own fanout. Fewer encoders than one per PE, and the long half of "
            "the array network moves fewer bits. The encoders run once per word "
            "ARRIVING at a column, which on this design's own mappings is up to "
            "7x the number injected -- that multiplicity is the cost side of "
            "the tradeoff and `ECC_RECON_ENCODER_SITE=source` prices the other "
            "side (one encoder, full-width network) for comparison."),
    ),
    Placement(
        "recon3", "recon_pe_spad_input",
        "R3 - reconstruct at the PE filter-spad input",
        "R3\n@ spad input", "4/5",
        reduced=("dram", "array_multicast", "pe_local_multicast"),
        site_stage="weights_spad", site_counter="fills",
        description=(
            "Both halves of the array network carry the reduced form and the "
            "encoder sits at the scratchpad write port, so the spad keeps its "
            "published 224 x 16b capacity and its full-width read cost. The "
            "encoder runs once per weight FILLED into a PE, the smallest count "
            "of any boundary below the network. Sec. 7.2's PE-FIFO boundary "
            "would sit between R2 and R3; the model has no FIFO level, so this "
            "is the first boundary after the whole network."),
    ),
    Placement(
        "recon4", "recon_pe_spad_output",
        "R4a - reconstruct on every filter-spad read",
        "R4a\n@ spad output", "3/5",
        reduced=("dram", "array_multicast", "pe_local_multicast",
                 "weights_spad"),
        site_stage="weights_spad", site_counter="reads",
        description=(
            "The scratchpad stores the reduced form, so its write and read "
            "bit-volume fall with its capacity, and the encoder sits at its "
            "read port. Row-stationary reuse is what makes this expensive: the "
            "MAC reads the spad directly, once per MAC, so the encoder runs "
            "once per weight DELIVERED rather than once per weight stored."),
    ),
)

#: Weight-stationary, Sec. 6.1/6.2. SIX boundaries: this design is the only one
#: in the study with both a weight global buffer above the network and a
#: stationary weight register below the scratchpad, so it has a boundary at
#: each. Sec. 6.2's own list starts at the buffer OUTPUT (3/5); the chip-ingress
#: control below is this study's addition, and it exists for the reason the
#: 2026-09-09 revision gives R1 on every design -- with the decoder on the DRAM
#: die, a boundary that reduces nothing on chip still saves the interface, so it
#: is what isolates that saving from the on-chip ones. It is labelled `control`
#: rather than given a rating the source discussion does not state for WS.
_WS_PLACEMENTS = (
    Placement(
        "recon1", "recon_source_noc_ingress",
        "R1 - reconstruct at chip ingress, before the weight buffer",
        "R1\n@ chip ingress", "control",
        reduced=("dram",), site_stage="dram",
        site_counter="reads",
        description=(
            "The die drives only the k message bits across the DRAM interface "
            "and an encoder at chip ingress restores them before the global "
            "operand buffer, so nothing on chip carries the reduced form. Sec. "
            "6.2 does not list this row -- its own R1 is at the buffer OUTPUT "
            "-- and it is drawn here as the control that separates the "
            "DRAM-interface saving every boundary shares from any on-chip "
            "saving. It pays one reconstruction per codeword fetched from DRAM "
            "and has nothing but that interface saving against it."),
    ),
    Placement(
        "recon2", "recon_global_buffer_output",
        "R2 - reconstruct at the global weight-buffer output",
        "R2\n@ buffer output", "3/5",
        reduced=("dram", "operand_glb"),
        site_stage="operand_glb", site_counter="reads",
        description=(
            "Sec. 6.2's R1, rated 3/5: the 64 kB operand buffer holds weight "
            "tiles in the reduced form, so its weight capacity and its weight "
            "access bit-volume both fall, and one encoder at its read port "
            "restores full width before the distribution network. The buffer "
            "also holds input activations; only its weight share moves. "
            "Downstream -- network, scratchpad, register -- is full width."),
    ),
    Placement(
        "recon3", "recon_noc_output_pe_input",
        "R3 - reconstruct at the weight-NoC output / PE input",
        "R3\n@ PE input", "4/5",
        reduced=("dram", "operand_glb", "weight_noc"),
        site_stage="pe_spad", site_counter="fills",
        description=(
            "Sec. 6.2's R2, rated 4/5: the buffer AND the 256-PE distribution "
            "network carry the reduced form, and an encoder at each PE input "
            "restores it before the scratchpad. The encoder runs once per "
            "weight FILLED into a PE, so a broadcast that reaches many PEs "
            "replicates the encoder rather than the reconstruction count of any "
            "one of them. The scratchpad keeps its full width and its read "
            "cost."),
    ),
    Placement(
        "recon4", "recon_pe_rf_output",
        "R4a - reconstruct on every weight-RF read",
        "R4a\n@ RF output", "2/5",
        reduced=("dram", "operand_glb", "weight_noc", "pe_spad"),
        site_stage="pe_spad", site_counter="reads",
        description=(
            "Sec. 6.2's R3a, rated 2/5 and the one row of the WS table rated "
            "BELOW the early boundaries. The scratchpad stores the reduced "
            "form, so its capacity and its bit-volume fall, but the encoder "
            "sits at its read port and a weight-stationary inner loop reads "
            "each resident weight hundreds of times: Sec. 6.1's illustrative "
            "'100 reduced-width RF reads and 100 reconstructions'. The design's "
            "own depth-1 stationary register does not amortize this -- the "
            "mapping fills it once per read (Timeloop's own counts), so it is a "
            "pipeline latch here, not a reuse register."),
    ),
    Placement(
        "recon5", "recon_mac_input",
        "R5 - reconstruct at the MAC input (stationary register reduced too)",
        "R5\n@ MAC input", "2/5",
        reduced=("dram", "operand_glb", "weight_noc", "pe_spad",
                 "weight_reg"),
        site_stage="weight_reg", site_counter="reads",
        description=(
            "Sec. 6.2's MAC row, rated 2/5: the latest boundary the design "
            "admits, with even the stationary register holding the reduced form "
            "and the encoder on the MAC's operand path. It is evaluated rather "
            "than argued away, and `feasibility()` is expected to reject it: "
            "rebuilding one weight needs the retained bits of G_rec = 9 "
            "co-resident weights and this register holds ONE, so the bits the "
            "rebuild depends on are not there. The rejection names the layers "
            "and the resident count, which is the answer to Sec. 6.2's row -- "
            "not a number produced by pretending the register is wider than the "
            "design declares."),
    ),
)


#: `eyeriss_like_wglb` -- Eyeriss v1's boundaries WITH the published filter
#: GLB. Five, not four: the GLB is a reducible storage stage above the array
#: network, so it admits a boundary at its output that `eyeriss_like` has
#: nowhere to put. The numbering follows `simple_weight_stationary`'s, the
#: other design in the study with a weight buffer above its network -- recon2
#: is the global weight buffer's output on both -- rather than shifting
#: `eyeriss_like`'s keys, which name different boundaries anyway.
#:
#: `validate_placement_space()` enforces both invariants this list has to
#: satisfy: each `reduced` set is a PREFIX of the path's reducible stages in
#: path order, and every reducible stage is reached by some boundary. Without
#: the second, adding `filter_glb` to `WEIGHT_PATHS` and forgetting it here
#: would leave every boundary below it reporting its own saving while the GLB
#: stayed at full width -- the whole list understated, with nothing saying so.
_EYERISS_V1_WGLB_PLACEMENTS = (
    Placement(
        "recon1", "recon_source_noc_ingress",
        "R1 - reconstruct at chip ingress, before the filter GLB",
        "R1\n@ source", "3/5",
        reduced=("dram",), site_stage="dram", site_counter="reads",
        description=(
            "Sec. 7.1's source-side reconstruction. The die drives only the k "
            "message bits across the DRAM interface and an encoder at the chip "
            "source restores them before anything on chip stores them, so "
            "nothing on chip carries the reduced form -- not even the filter "
            "GLB. It is the control that isolates the DRAM-interface saving, "
            "which every boundary shares, from every on-chip saving, and it "
            "pays one reconstruction per codeword fetched from DRAM with "
            "nothing but that interface saving against it."),
    ),
    Placement(
        "recon2", "recon_weight_glb_output",
        "R2 - reconstruct at the filter-GLB output",
        "R2\n@ filter GLB", "3/5",
        reduced=("dram", "filter_glb"),
        site_stage="filter_glb", site_counter="reads",
        description=(
            "The 8 kB filter GLB stores the reduced form, so it holds N/K more "
            "weights per bank and its per-weight read cost falls with the bit "
            "count; the encoder sits at its read port, before the array "
            "network. This is the boundary `eyeriss_like` has nowhere to put, "
            "and it is the one FINDINGS 7.8 predicts matters: refetch on v1 is "
            "set by the DRAM-level loop order over P and Q, a weight tile "
            "cannot index either, and only a weight level ABOVE the PE array "
            "can absorb those loops. Its cost side is that the GLB is read "
            "once per weight DELIVERED into the array, not once per weight "
            "stored."),
    ),
    Placement(
        "recon3", "recon_after_array_multicast",
        "R3 - reconstruct after the array multicast, at the column edge",
        "R3\n@ column edge", "4/5",
        reduced=("dram", "filter_glb", "array_multicast"),
        site_stage="array_multicast", site_counter="deliveries",
        description=(
            "Sec. 7.1's multicast tradeoff, resolved in favour of reduced-width "
            "shared transport: the GLB and the 14-way column multicast both "
            "carry the reduced form and one encoder per column restores it "
            "before the column's own fanout. Fewer encoders than one per PE, "
            "and the long half of the array network moves fewer bits. The "
            "encoders run once per word ARRIVING at a column, which on this "
            "design's own mappings is up to 7x the number injected; "
            "`ECC_RECON_ENCODER_SITE=source` prices the other side of the "
            "tradeoff (one encoder, full-width network)."),
    ),
    Placement(
        "recon4", "recon_pe_spad_input",
        "R4 - reconstruct at the PE filter-spad input",
        "R4\n@ spad input", "4/5",
        reduced=("dram", "filter_glb", "array_multicast", "pe_local_multicast"),
        site_stage="weights_spad", site_counter="fills",
        description=(
            "The GLB and both halves of the array network carry the reduced "
            "form and the encoder sits at the scratchpad write port, so the "
            "spad keeps its published 224 x 16b capacity and its full-width "
            "read cost. The encoder runs once per weight FILLED into a PE, the "
            "smallest count of any boundary below the network. Sec. 7.2's "
            "PE-FIFO boundary would sit between R3 and R4; the model has no "
            "FIFO level, so this is the first boundary after the whole "
            "network."),
    ),
    Placement(
        "recon5", "recon_pe_spad_output",
        "R5a - reconstruct on every filter-spad read",
        "R5a\n@ spad output", "3/5",
        reduced=("dram", "filter_glb", "array_multicast", "pe_local_multicast",
                 "weights_spad"),
        site_stage="weights_spad", site_counter="reads",
        description=(
            "Every weight-carrying stage holds the reduced form, so the spad's "
            "write and read bit-volume fall with its capacity too, and the "
            "encoder sits at its read port. Row-stationary reuse is what makes "
            "this expensive: the MAC reads the spad directly, once per MAC, so "
            "the encoder runs once per weight DELIVERED rather than once per "
            "weight stored."),
    ),
)


PLACEMENTS = {
    "eyeriss_v2_like": _V2_PLACEMENTS,
    "eyeriss_v2_like_wglb": _V2_PLACEMENTS,
    "eyeriss_like": _EYERISS_V1_PLACEMENTS,
    "eyeriss_like_wglb": _EYERISS_V1_WGLB_PLACEMENTS,
    "simple_weight_stationary": _WS_PLACEMENTS,
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


def encoder_site(cfg):
    """`destination` (the model) or `source` (the pre-2026-09-09 diff row)."""
    site = (getattr(cfg, "recon_encoder_site", "destination")
            if cfg is not None else "destination")
    if site not in ENCODER_SITES:
        raise ValueError(f"unknown encoder site {site!r}; one of {ENCODER_SITES}")
    return site


def stages_for(arch, cfg=None):
    """The weight path of `arch`, as the decode site makes it.

    The tables above are written for the on-die decoder. Under `controller`
    the complete codeword crosses the interface, so `dram` is handed
    back with `reducible=False` and nothing else changes -- which is exactly
    what makes the old numbers come back rather than a different model.
    """
    if arch not in WEIGHT_PATHS:
        raise KeyError(arch)
    stages = WEIGHT_PATHS[arch]
    if decode_site(cfg) == "controller":
        stages = tuple(replace(s, reducible=False) if s.key == "dram"
                       else s for s in stages)
    return stages


def effective_placement(placement, cfg):
    """`placement` as the two site knobs make it.

    Under `ECC_RECON_DECODE_SITE=controller` the DRAM interface is not
    reducible, so it leaves every placement's `reduced` set. Under
    `ECC_RECON_ENCODER_SITE=source` a network boundary counts its
    reconstructions at the network's INGRESSES rather than its destination-side
    arrivals -- one encoder before the fanout instead of one per destination.
    Both are the pre-2026-09-09 model, kept runnable so the change is a diff.
    """
    if decode_site(cfg) == "controller" and "dram" in placement.reduced:
        placement = replace(placement, reduced=tuple(
            k for k in placement.reduced if k != "dram"))
    if encoder_site(cfg) == "source" and placement.site_counter == "deliveries":
        placement = replace(placement, site_counter="ingresses")
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


# ===========================================================================
#  prompt_6 -- which placements get their own mapping (ERT arms)
# ===========================================================================
#: The ERT access action a placement's `site_counter` names. A weight arriving
#: at a level is a `write` (Timeloop counts it as a fill), a weight leaving it
#: is a `read`. Network counters name no component action: a delivery is
#: billed through `noc.yaml` (prompt_6 3.3).
ERT_ACTION_OF_COUNTER = {"reads": "read", "fills": "write"}


def ert_injectable(placement, stages):
    """prompt_6 3.3: is this boundary's encoder cost an ERT action the mapper
    can trade against? `(ok, why)`, DERIVED from the Placement record -- never
    `if key == "recon2"`.

    Three conditions, all required:

    1. the site stage is a STORAGE stage. DRAM is not a chip action the
       encoder attaches to, and a network delivery is billed by `noc.yaml`.
    2. the site counter is `reads` or `fills`, naming exactly one ERT action.
    3. it is not the innermost weight level's `reads`: that count equals the
       MAC count, which no mapping can move, so a constant added to every
       mapping cannot move the argmax.
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
    if site is storage[-1] and placement.site_counter == "reads":
        return False, (f"{site.key} is the innermost weight level and its reads "
                       f"equal the MAC count, which no mapping can move")
    return True, f"{site.key} + {placement.site_counter} -> ERT {action}"


def ert_arms(arch, cfg=None):
    """The placements of `arch` that get their own mapping, in path order.

    Zero is legal (the figure is then Task 3's with the idle term added); the
    code must work at 0, 1 or 2+. On Eyeriss v1 with the filter GLB this is
    `recon2` (filter_glb reads) and `recon4` (weights_spad fills).
    """
    stages = stages_for(arch, cfg)
    return tuple(p for p in placements_for(arch, cfg) if ert_injectable(p, stages)[0])


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
    is not ERT-injectable, naming the reason.
    """
    p = placement_by_key(arch, key, cfg)
    if p is None:
        raise KeyError(f"{arch} has no placement {key!r}; it has "
                       f"{', '.join(q.key for q in placements_for(arch, cfg))}")
    stages = stages_for(arch, cfg)
    ok, why = ert_injectable(p, stages)
    if not ok:
        raise ValueError(f"{arch}/{key} is not an ERT arm: {why}. The ERT arms of "
                         f"{arch} are {', '.join(a.key for a in ert_arms(arch, cfg)) or 'none'}")
    by_key = {s.key: s for s in stages}
    site = by_key[p.site_stage]
    narrow = tuple(by_key[k].prefixes[0] for k in p.reduced
                   if by_key[k].kind == "storage")
    return {"placement": p, "key": p.key, "level": site.prefixes[0],
            "counter": p.site_counter, "action": ERT_ACTION_OF_COUNTER[p.site_counter],
            "narrow_levels": narrow}


def ert_deltas(incremental_pj, idle_pj, gran, block_size):
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
    e_w = float(incremental_pj) * gran.codewords(1.0)
    return {"e_w_pj": e_w,
            "access_delta_pj": e_w * int(block_size),
            "leak_delta_pj": float(idle_pj),
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
    deliveries: float = 0.0       # TOTAL destination-side arrivals = ingresses
                                  # x multicast factor (all instances)
    multicast: float = 1.0        # weight-only multicast factor of the network
    fanout: float = 1.0           # weight-only fanout of the network
    utilized_capacity: float = 0.0   # storage: weights resident per instance
    block_bits: int = 0              # storage: block size x word bits
    word_bits: int = 0               # storage: MEASURED `Word bits` of the level
                                     # (prompt_6 RULE 1: q -> the mapper narrowed
                                     # it; weight_bits -> the evaluator does)
    instances: float = 0.0           # utilized instances (max over layers)
    declared_instances: float = 0.0  # storage: the level's declared `Instances`
                                     # (the engines that EXIST)
    engine_cycles: float = 0.0       # RULE 3's idle denominator, summed over
                                     # layers: engines that LEAK x that layer's
                                     # cycles. Storage: UTILIZED instances --
                                     # Timeloop power-gates each unused instance
                                     # (`Instances sharing power gating: 1`) and
                                     # bills leak x utilized x cycles
                                     # (buffer.cpp FinalizeBufferEnergy, verified
                                     # 2026-09-11 on 43 shapes); dram: 1;
                                     # network: fanout x instances.
    level_share: float = 1.0      # dram: this stage's share of the Timeloop level

    def counter(self, name):
        if name == "reads":
            return self.reads
        if name == "fills":
            return self.fills
        if name == "ingresses":
            return self.ingresses
        if name == "deliveries":
            return self.deliveries
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
            "weight_network_deliveries_total": self.deliveries,
            "network_multicast_factor": self.multicast,
            "network_fanout": self.fanout,
            "weights_resident_per_instance": self.utilized_capacity,
            "physical_word_bits": self.block_bits,
            "measured_word_bits": self.word_bits,
            "instances": self.instances,
            "declared_instances": self.declared_instances,
            "engine_cycles": self.engine_cycles,
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
    unclaimed: list                 # weight-carrying levels no stage claimed
    stats_path: str
    cycles: float = 0.0             # this plan's `Cycles:` x repeat count (RULE 3)

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




def _level_shares(claimants, level):
    """How much of one Timeloop level each claiming stage is.

    Exactly one claimant owns a level. The `dram_array`/`dram_interface` pair
    that used to split the DRAM level by `f_if` was removed on 2026-09-09 -- the
    DRAM level is one `dram` stage and the whole of it is reducible -- so a
    level claimed twice is now always a table error.
    """
    if len(claimants) == 1:
        return {claimants[0].key: 1.0}
    raise ValueError(
        f"Timeloop level {level!r} is claimed by {[s.key for s in claimants]}; "
        f"a level may be claimed by exactly one stage")


def apply_dram_pj_per_bit(stages, cfg):
    """Rescale the `dram` stage to ECC_DRAM_PJ_PER_BIT, in place.

    THE WEIGHT PATH IS RE-PARSED FROM THE CACHED TIMELOOP OUTPUT, so it carries
    Accelergy's own per-bit DRAM constant (8.0 pJ/bit for LPDDR4 as modelled),
    while the `Raw` record it must reconcile against has already been rescaled
    by `energy.apply_dram_override()`. Without this the two disagree by the
    override ratio and `cross_check()` fails -- which is exactly what happened
    when ECC_DRAM_PJ_PER_BIT was introduced, so the check earned its keep.

    The ERT per-bit is read back off the stage itself (energy / (reads x
    weight_bits)), so this needs no second source of truth for the 8.0.
    """
    tgt = getattr(cfg, "dram_pj_per_bit", None)
    if tgt is None:
        return stages
    for st in stages.values():
        if st.kind != "dram":
            continue
        bits = float(st.reads or 0.0) * float(cfg.weight_bits)
        if bits <= 0 or st.energy_pJ <= 0:
            continue
        ert = st.energy_pJ / bits
        st.energy_pJ *= float(tgt) / ert
    return stages


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
    unclaimed = []
    util_by_stage = {}          # storage stage -> max utilized instances (this layer)

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
        shares = _level_shares(claimants, level)
        util = _grab(r"Utilized instances \(max\)\s*:\s*(\d+)", wblock, int) or 1
        reads = (_grab(rf"Scalar reads \(per-instance\)\s*:\s*{_NUM}", wblock)
                 or 0.0) * util * scale
        fills = ((_grab(rf"Scalar fills \(per-instance\)\s*:\s*{_NUM}", wblock) or 0.0)
                 + (_grab(rf"Scalar updates \(per-instance\)\s*:\s*{_NUM}", wblock) or 0.0)
                 ) * util * scale
        cap = _grab(r"Utilized capacity\s*:\s*(\d+)", wblock, int)
        word = _grab(r"Word bits\s*:\s*(\d+)", body, int) or cfg.weight_bits
        blk = _grab(r"Block size\s*:\s*(\d+)", body, int) or 1
        declared = _grab(r"Instances\s*:\s*(\d+)", body.split("STATS")[0], int) or util
        for stage in claimants:
            util_by_stage[stage.key] = max(util_by_stage.get(stage.key, 0), util)
            # ENERGY is the stage's share of the level (1.0 -- one claimant
            # per level since the f_if split was removed); the ACCESS COUNTS
            # are not scaled (R1 counts its reconstructions off dram's reads).
            share = shares[stage.key]
            st = stages[stage.key]
            st.levels.append(level)
            st.level_share = share
            st.energy_pJ += energy * share
            st.switch_pJ += energy * share    # storage: no wire/switching split
            st.reads += reads
            st.fills += fills
            st.instances = max(st.instances, util)
            st.declared_instances = max(st.declared_instances, declared)
            if cap is not None:
                st.utilized_capacity = (cap if st.utilized_capacity == 0
                                        else min(st.utilized_capacity, cap))
            st.block_bits = max(st.block_bits, word * blk)
            st.word_bits = max(st.word_bits, word)

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
        ingress_pi = _grab(rf"Ingresses\s*:\s*{_NUM}", wblock) or 0.0
        ingress = ingress_pi * instances * scale
        # DESTINATION-SIDE ARRIVALS, from Timeloop's own multicast breakdown.
        # A multicast network injects one word and DELIVERS it to `multicast`
        # destinations, and Timeloop prints the split as
        #     Ingresses : 2359296.00
        #         @multicast 7 @scatter 2: 2359296.00
        # so the arrivals are sum(m x count) over those lines -- exact, and it
        # reconciles: on eyeriss_like's C512 shape the column network's
        # 2,359,296 ingresses x 7 equal the inner network's 16,515,072, which
        # equal the scratchpad fills. `Multicast factor` is the fallback when
        # the breakdown lines are absent.
        parts = re.findall(rf"@multicast\s*(\d+)\s*@scatter\s*(\d+)\s*:\s*{_NUM}",
                           wblock)
        mc = _grab(r"Multicast factor\s*:\s*(\d+)", wblock, int) or 1
        fan = _grab(r"Fanout\s*:\s*(\d+)", wblock, int) or 1
        if parts:
            deliv_pi = sum(int(m) * float(c) for m, _s, c in parts)
        else:
            deliv_pi = ingress_pi * mc
        deliveries = deliv_pi * instances * scale
        st = stages[stage.key]
        st.levels.append(level)
        st.energy_pJ += energy
        st.wire_pJ += wire
        st.switch_pJ += switch
        st.ingresses += ingress
        st.deliveries += deliveries
        st.multicast = max(st.multicast, float(mc))
        st.fanout = max(st.fanout, float(fan))
        st.instances = max(st.instances, instances)
        st.block_bits = max(st.block_bits,
                            _grab(r"Word bits\s*:\s*(\d+)", specs, int) or cfg.weight_bits)

    apply_dram_pj_per_bit(stages, cfg)
    cycles = _grab(r"\nCycles:\s*(\d+)", text, int)
    # RULE 3's idle denominator, PER LAYER: the engines that leak in this
    # layer's plan x this layer's cycles. A storage site's engines are the
    # UTILIZED instances of its level, because that is what Timeloop bills
    # `leak` on (each unused instance is power-gated: buffer.cpp
    # leaks_per_cycle = max utilized instances when `Instances sharing power
    # gating` is 1, FinalizeBufferEnergy = leak x cycles x leaks_per_cycle).
    # Summing per layer is what a full model needs -- conv1 uses 98 PEs, fc
    # 16, a 3x3 layer 168 -- and it is what reconciles to 1e-6 against the
    # stats' leakage delta on an ERT arm.
    for st in stages.values():
        if st.energy_pJ <= 0:
            continue
        if st.kind == "dram":
            engines = 1.0
        elif st.kind == "network":
            engines = max(1.0, float(round(float(st.fanout) * float(st.instances or 1))))
        else:
            engines = float(util_by_stage.get(st.key, st.instances) or 1)
        st.engine_cycles = engines * float(cycles or 0) * scale
    return LayerWeightPath(
        layer=getattr(layer, "name", "?"), shape=getattr(layer, "shape_name", "?"),
        weights=int(getattr(layer, "weights", 0)), scale=scale, stages=stages,
        unclaimed=unclaimed, stats_path=str(stats_path),
        cycles=(cycles * scale) if cycles is not None else 0.0)


@dataclass
class ModelWeightPath:
    """Every mapped layer's weight path, summed, with the per-layer detail kept."""
    arch: str
    model: str
    stages: dict
    per_layer: list
    unclaimed: list
    #: Directory holding the cached Timeloop output, hence the Accelergy ERT.
    #: Task 4's dilation correction needs it to reprice a level from the ERT.
    stats_dir: str = ""
    #: The stage definitions this path was read with -- `stages_for(arch, cfg)`,
    #: so `dram.reducible` reflects the decode site of the run.
    stage_defs: tuple = ()
    decode_site: str = "ondie"
    #: pJ per bit of dynamic DRAM access actually charged, and where it came
    #: from (energy.apply_dram_override / config.dram_cost_note). Replaced
    #: `dram_if_frac` when the array/interface split was removed 2026-09-09.
    dram_pj_per_bit: float = 0.0
    dram_cost_note: str = ""
    #: prompt_6 RULE 3: the summed `Cycles:` of every mapped layer (x repeat
    #: count), read off THESE stats -- the idle term's denominator for a bar
    #: billed from this plan.
    cycles: float = 0.0

    def stage(self, key):
        return self.stages.get(key)

    def dram_weight_energy(self):
        """The DRAM weight energy -- one stage, i.e. the Raw record's."""
        return sum(s.energy_pJ for s in self.stages.values() if s.kind == "dram")

    def dram_term(self):
        """The DRAM weight term, as a record. ONE stage since 2026-09-09.

        The `dram_array` / `dram_interface` pair and its `f_if` multiplier are
        gone: the whole DRAM weight energy is reducible by K/N.
        """
        d = self.stages.get("dram")
        return {
            "decode_site": self.decode_site,
            "dram_pj_per_bit": self.dram_pj_per_bit,
            "dram_cost_provenance": self.dram_cost_note,
            "dram_weight_energy_pJ": self.dram_weight_energy(),
            "dram_reducible": bool(d is not None and any(
                st.key == "dram" and st.reducible for st in self.stage_defs)),
            "rule": ("dram = DRAM weight energy x K/N under every boundary when "
                     "the decoder is on the DRAM die (01_project_context Sec. 4). "
                     "Accelergy's CactiDRAM is one flat per-bit DYNAMIC access "
                     "constant and the whole of it is credited: the DRAM "
                     "access fetches only the message bits of each codeword. "
                     "E_background and E_refresh are not modelled (both 0)"),
        }

    def total_weight_energy(self):
        return sum(s.energy_pJ for s in self.stages.values())

    def multicast_chain(self):
        """Evidence that the destination-side ARRIVAL count is parsed right.

        A network's arrivals are the words its destinations receive, so they
        should equal what the NEXT stage down the path takes in -- the next
        network's ingresses, or the scratchpad's fills. That identity is the
        cross-check on `deliveries`, which is what a network boundary's encoders
        are charged per (see `ENCODER_SITES`): on `eyeriss_v2_like` the mesh
        injects 16,356,544 weight words and its destinations receive 23,812,480,
        which is exactly the cluster-local network's ingresses AND the
        scratchpad's fills.

        Reported, not enforced. It is an identity of these designs' weight
        paths, not of Timeloop in general -- a level that drops or re-fetches
        words between two stages would break it legitimately -- so it is
        recorded on every result and read, rather than failing a run.
        """
        rows, order = [], [s.key for s in self.stage_defs]
        for i, key in enumerate(order):
            st = self.stages.get(key)
            if st is None or st.kind != "network" or st.energy_pJ <= 0:
                continue
            nxt = None
            for k2 in order[i + 1:]:
                s2 = self.stages.get(k2)
                if s2 is not None and s2.energy_pJ > 0:
                    nxt = s2
                    break
            takes_in = (0.0 if nxt is None else
                        nxt.ingresses if nxt.kind == "network" else nxt.fills)
            rows.append({
                "network": key,
                "words_injected_ingresses": st.ingresses,
                "words_received_by_destinations": st.deliveries,
                "multicast_factor_reported_by_timeloop": st.multicast,
                "effective_multiplicity": (st.deliveries / st.ingresses
                                           if st.ingresses > 0 else 1.0),
                "next_stage": None if nxt is None else nxt.key,
                "next_stage_takes_in": takes_in,
                "arrivals_match_the_next_stage": bool(
                    takes_in and abs(st.deliveries - takes_in)
                    <= 1e-6 * max(st.deliveries, takes_in)),
            })
        return {
            "rule": ("a network's destination-side arrivals should equal what "
                     "the next stage takes in (the next network's ingresses, or "
                     "the scratchpad's fills); this is the cross-check on the "
                     "arrival count a network boundary's encoders are charged "
                     "per, and it is recorded rather than enforced"),
            "per_network": rows,
        }

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
                 "evidence": s.evidence}
                for s in self.stage_defs],
            "dram_term": self.dram_term(),
            "multicast_arrival_chain": self.multicast_chain(),
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
            agg.deliveries += st.deliveries
            agg.multicast = max(agg.multicast, st.multicast)
            agg.fanout = max(agg.fanout, st.fanout)
            agg.instances = max(agg.instances, st.instances)
            agg.declared_instances = max(agg.declared_instances, st.declared_instances)
            agg.engine_cycles += st.engine_cycles
            agg.block_bits = max(agg.block_bits, st.block_bits)
            agg.word_bits = max(agg.word_bits, st.word_bits)
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
            "weight_energy_pJ": lp.total_weight_energy(),
            "cycles": lp.cycles,
            "stages": {k: st.to_dict() for k, st in lp.stages.items()
                       if st.energy_pJ > 0},
        })
        layer_paths.append(lp)

    stats_dir = str(pathlib.Path(layer_paths[-1].stats_path).parent) \
        if layer_paths else ""
    return ModelWeightPath(arch=arch, model=model, stages=stages,
                           per_layer=per_layer,
                           unclaimed=unclaimed, stats_dir=stats_dir,
                           stage_defs=stage_defs, decode_site=decode_site(cfg),
                           dram_pj_per_bit=float(getattr(cfg, 'dram_pj_per_bit', None) or 0.0),
                           dram_cost_note=getattr(cfg, 'dram_cost_note', ''),
                           cycles=sum(lp.cycles for lp in layer_paths))




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
        return code_widths.declared_datawidth(self.n, self.k, self.weight_bits)

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








# ===========================================================================
#  PROMPT_2: the on-chip narrowing must be applied EXACTLY ONCE
# ===========================================================================
def onchip_narrowing_audit(cfg):
    """Where the K/N on-chip weight narrowing is applied, and how many times.

    CONFIGURATION-LEVEL ESTIMATE. Since prompt_6 phase 4 the placement study
    decides ownership per BAR per STAGE from measured `Word bits`
    (`Packing.narrowing_owner`, `narrowing_audit_for_bar`, called inside
    `evaluate_placement`), which is the authority; this function keeps the
    coarse pre-check `dilation --levels` prints. FINDINGS 6.3 records why the
    name-based `evaluator_narrows` below is not a measurement.

    THE DOUBLE-COUNTING THIS EXISTS TO STOP. There are now two places that can
    narrow an on-chip weight:

    * the EVALUATOR, via `Packing`. `stream` scales every reduced stage's
      access count and energy by k/n.
    * the MAPPER, via `ECC_WEIGHT_DATAWIDTH`. Timeloop bills
      `vector_access_energy / block_size` with `block_size = width/datawidth`,
      so a narrower declared datawidth delivers the same saving INSIDE the
      Timeloop number, with no mapping change at all.

    Apply both and the on-chip saving is SQUARED. prompt_2's resolution is
    `ECC_RECON_PACKING=aligned`, whose docstring already describes exactly the
    right model -- "each reduced weight occupies a whole number of bits ...
    the access count, and the SRAM energy, do not move at all" -- so `aligned`
    leaves the on-chip narrowing entirely to the mapper, which is now where it
    belongs.

    Two consequences this checks, both named in prompt_2:

    * DRAM `datawidth` stays 8 on every arm. `_set_weight_datawidth` refuses to
      touch DRAM for this reason; `recon.py` owns the DRAM K/N scaling and
      narrowing DRAM in the YAML too would double-count it there. That is
      structural, so it is asserted in `archs.py` rather than here.
    * `aligned` computes its own reduced width as `ceil(weight_bits*k/n)`, and
      that has to AGREE with the `datawidth` the arch declares, or the arch and
      the accounting are describing two different codes and the check cannot
      reconcile. At BCH(63,30) both are 4; at BCH(63,57) `aligned` says 8 and
      the declared q is 7, which is exactly the disagreement prompt_2 warns
      about.

    NEITHER site active is NOT a defect: `aligned` with no mapper-side
    datawidth is the pessimistic bound this study has always been able to run
    (`Packing`'s docstring, section 16). It comes back `ok` with a `note`, so
    it stays runnable but cannot be mistaken for a prompt_2 result.

    Returns a dict; `ok` False means the run must not be quoted.
    """
    mapper_bits = getattr(cfg, "weight_datawidth", None)
    packing = Packing(cfg.recon_packing, cfg.weight_bits, cfg.code_k, cfg.code_n)
    mapper_narrows = (mapper_bits is not None
                      and mapper_bits < cfg.weight_bits)
    # `stream` is the only packing that moves an on-chip ACCESS COUNT.
    evaluator_narrows = cfg.recon_packing == "stream"
    sites = ([f"mapper (ECC_WEIGHT_DATAWIDTH={mapper_bits})"] if mapper_narrows
             else []) + (["evaluator (ECC_RECON_PACKING=stream)"]
                         if evaluator_narrows else [])
    out = {
        "mapper_datawidth": mapper_bits,
        "packing": cfg.recon_packing,
        "aligned_bits_per_weight": packing.reduced_bits_per_weight,
        "sites": sites, "n_sites": len(sites),
        "ok": len(sites) == 1,
    }
    if len(sites) > 1:
        out["problem"] = (
            "THE ON-CHIP NARROWING IS APPLIED TWICE and the saving is "
            "SQUARED: " + " and ".join(sites) + ". prompt_2's resolution is "
            "ECC_RECON_PACKING=aligned, which leaves the on-chip narrowing "
            "entirely to the mapper.")
    elif not sites:
        # NOT a defect, and deliberately not a stop: `aligned` with no
        # mapper-side datawidth is the PESSIMISTIC BOUND this study has always
        # been able to run (`Packing`'s docstring, section 16). It predates
        # prompt_2 and stays runnable. It is simply not a prompt_2 result, so
        # it says so rather than being quoted as one.
        out["ok"] = True
        out["note"] = (
            f"NO on-chip narrowing is applied anywhere: "
            f"ECC_RECON_PACKING={cfg.recon_packing} moves no on-chip access "
            f"count and ECC_WEIGHT_DATAWIDTH is unset, so the reconstruction "
            f"arm stores its weights at the full {cfg.weight_bits} bits. That "
            f"is the pessimistic bound, not prompt_2's model -- for that, set "
            f"ECC_WEIGHT_DATAWIDTH={packing.reduced_bits_per_weight} "
            f"(= ceil({cfg.weight_bits}*K/N) at "
            f"BCH({cfg.code_n},{cfg.code_k})) on the reconstruction arm.")
    elif mapper_narrows and mapper_bits != packing.reduced_bits_per_weight:
        out["ok"] = False
        out["problem"] = (
            f"THE ARCH AND THE ACCOUNTING DISAGREE ABOUT THE CODE: the arch "
            f"declares datawidth {mapper_bits} on its weight levels, but "
            f"`aligned` computes ceil({cfg.weight_bits}*K/N) = "
            f"{packing.reduced_bits_per_weight} bits at "
            f"BCH({cfg.code_n},{cfg.code_k}). Make them agree -- prompt_2's "
            f"width table rounds 8*K/N to an integer q and the declared "
            f"datawidth IS that q -- or the check cannot reconcile.")
    return out


def assert_onchip_narrowing_once(cfg):
    """`onchip_narrowing_audit()` as a hard stop. Returns the audit."""
    audit = onchip_narrowing_audit(cfg)
    if not audit["ok"]:
        raise ValueError(audit["problem"])
    return audit


# ===========================================================================
#  TASK 4: capacity dilation -- the correction the dilated mapping needs
# ===========================================================================
def capacity_dilation_scale(cfg):
    """The reconstruction arm's effective weight capacity, as a scale factor.

    The reference arm's physical buffer is `cfg.weight_capacity_scale` times
    the declared design (1.0 normally; below 1 when the study is shrinking the
    design to find the regime where capacity binds at all). The reconstruction
    arm stores the same weights at K/N of full width in that SAME silicon, so
    it holds N/K times as many of them.
    """
    # Rounded to the same four decimals `config.load_config()` quantises the
    # reference scale to, so the derived capacity has ONE spelling and
    # therefore one cache directory. See the comment there.
    return round(cfg.weight_capacity_scale * cfg.code_n / cfg.code_k, 4)


def capacity_target(cfg):
    """prompt_6 6: the weight room a re-planned arm must show, `(factor, q)`.

    The reduced representation is `datawidth: q` with `q = round(weight_bits x
    K/N)` a WHOLE number of bits, so the room actually delivered is
    `weight_bits / q` -- 2.000 at BCH(63,30), 1.333 at BCH(63,51) -- and NEVER
    N/K (2.100, 1.235). Checking against N/K refused two codes outright and
    passed the other four by luck (within the 5 % slack); this target is exact
    for every code, so the slack goes back to catching real faults.
    """
    q = code_widths.declared_datawidth(cfg.code_n, cfg.code_k, cfg.weight_bits)
    return cfg.weight_bits / q, q


def capacity_dilation_correction(ref_stats_dir, dil_stats_dir, prefixes):
    """The DECLARED array's per-access energies, for re-pricing a dilated level.

    THE PROBLEM. To ask the mapper what it would do with N/K more weight room,
    the room has to be in the YAML it reads, so `archs._scale_weight_capacity`
    multiplies that level's `depth:`. Accelergy then costs the level from its
    declared geometry and prices it as a physically LARGER array. The
    reconstruction arm's array is not larger. It is the same array holding
    narrower values, which is the whole premise. Charging it CACTI's cost for
    the bigger array would bill the design for silicon it does not have, and
    would do so in the direction that makes reconstruction look worse.

    READS AND WRITES ARE CORRECTED SEPARATELY, because they do not scale
    together. On `eyeriss_like` at x1.6154 the weight scratchpad's read energy
    goes 0.783354 -> 0.970977 pJ (x1.2395) and its write energy 1.25362 ->
    1.72943 (x1.3796): a single read-derived ratio leaves the write half
    under-corrected, which showed up as a 0.22 pp residual between Task 4 and
    Task 3 on a mapping that was byte-identical. So the correction re-prices
    from the access counts -- `reads x e_read + writes x e_write` at the
    DECLARED energies -- and `correct_level_energy()` reconciles the same
    arithmetic at the DILATED energies against Timeloop's own total before
    trusting it.

    WHAT IT CANNOT FIX, AND WHY THAT IS THE CONSERVATIVE DIRECTION. The mapper
    optimised against the DEARER array, so a level that got materially more
    expensive per access was one the search had a reason to avoid -- the found
    mapping is therefore no better than the one a correctly-priced search would
    have found, and the Task 4 saving this yields is a LOWER bound.

    Returns a dict; `ok` False leaves the energy uncorrected and says why,
    rather than scaling by a guess.
    """
    e_rd_ref, e_wr_ref, prov_ref = storage_access_pj(ref_stats_dir, prefixes)
    e_rd_dil, e_wr_dil, prov_dil = storage_access_pj(dil_stats_dir, prefixes)
    level = prefixes[0] if prefixes else "?"
    if not (e_rd_ref and e_wr_ref and e_rd_dil and e_wr_dil):
        return {"ok": False, "level": level,
                "provenance": (
                    "NOT corrected: the declared and dilated ERTs are not both "
                    f"readable ({prov_ref} / {prov_dil}). The dilated level is "
                    "left at Accelergy's cost for the LARGER array, which "
                    "understates the reconstruction arm.")}
    return {
        "ok": True, "level": level,
        "read_pJ_declared": e_rd_ref, "read_pJ_dilated": e_rd_dil,
        "write_pJ_declared": e_wr_ref, "write_pJ_dilated": e_wr_dil,
        "read_ratio_declared_over_dilated": e_rd_ref / e_rd_dil,
        "write_ratio_declared_over_dilated": e_wr_ref / e_wr_dil,
        "provenance": (
            f"dilated {level} re-priced at the DECLARED array's per-access "
            f"energies: read {e_rd_dil:.6f} -> {e_rd_ref:.6f} pJ "
            f"(x{e_rd_ref / e_rd_dil:.4f}), write {e_wr_dil:.6f} -> "
            f"{e_wr_ref:.6f} pJ (x{e_wr_ref / e_wr_dil:.4f}). The "
            f"reconstruction arm's array is the same silicon holding narrower "
            f"values, so Accelergy's cost for the deeper array is not its cost."),
    }


def correct_level_energy(corr, energy_pJ, reads, writes, tol=0.02):
    """`energy_pJ` re-priced at the declared array, from its own access counts.

    THE BLOCK SIZE CANCELS, AND THAT IS WHY THIS IS A RATIO. Accelergy's ERT
    prices one VECTOR access -- a whole physical word -- while Timeloop counts
    SCALAR accesses, one per value, so `reads x e_read` overstates a level's
    energy by exactly its block size (2 on `eyeriss_like`'s 16-bit,
    8-bit-datawidth weight scratchpad; 3 on Eyeriss v2's 24-bit one). Rebuilding
    an absolute energy therefore needs the packing, and getting it wrong is
    silent. Re-pricing as an access-weighted RATIO does not:

        corrected = energy x  (reads x e_read_declared + writes x e_write_declared)
                             ---------------------------------------------------
                              (reads x e_read_dilated  + writes x e_write_dilated)

    Both sums carry the same block size, so it divides out, and what is left is
    exactly "how much cheaper the declared array is for THIS mix of reads and
    writes". That matters because reads and writes do not scale together: on
    `eyeriss_like` at x1.6154 the read energy rises 1.2395x and the write energy
    1.3796x, so a read-derived ratio leaves the write half under-corrected.

    THE RECONCILIATION IS STILL DONE, on the one thing the ratio cannot check:
    that the ERT split describes this level at all. `rebuilt / energy_pJ` must
    come out as a whole number of values per word -- the block size. Anything
    else means the split and the level do not belong together, and the energy
    is returned UNCHANGED with the discrepancy named, the same refusal
    `evaluate_placement` makes before re-billing scratchpad reads from an
    unreconciled ERT split.

    Returns `(corrected_pJ, note)`.
    """
    if not corr.get("ok") or energy_pJ <= 0:
        return energy_pJ, corr.get("provenance", "not corrected")
    dil = reads * corr["read_pJ_dilated"] + writes * corr["write_pJ_dilated"]
    ref = reads * corr["read_pJ_declared"] + writes * corr["write_pJ_declared"]
    if dil <= 0:
        return energy_pJ, (f"NOT corrected: {corr['level']} reports no accesses "
                           f"to re-price ({reads:,.0f} reads, {writes:,.0f} writes)")
    per_word = dil / energy_pJ
    if per_word < 1.0 - tol or abs(per_word - round(per_word)) > tol * max(1.0, per_word):
        return energy_pJ, (
            f"NOT corrected: the dilated ERT split does not reproduce "
            f"Timeloop's {corr['level']} energy as a whole number of values "
            f"per physical word -- {reads:,.0f} reads + {writes:,.0f} writes "
            f"price at {dil:,.0f} pJ against Timeloop's {energy_pJ:,.0f} pJ, a "
            f"factor of {per_word:.4f}. Refusing to re-price from an "
            f"unreconciled split.")
    return energy_pJ * (ref / dil), (
        f"{corr['provenance']} Applied as an access-weighted ratio "
        f"x{ref / dil:.6f} over {reads:,.0f} reads and {writes:,.0f} writes; "
        f"the split reconciles with Timeloop at {round(per_word)} values per "
        f"physical word.")


def apply_capacity_correction(wpath, stage_key, corr):
    """Re-price one weight-path stage at the declared array. Returns what moved.

    Kept deliberately small and explicit: the correction touches exactly the
    stage whose level the dilation rewrote, and `cross_check()` is re-run
    against the corrected totals afterwards, so a correction that does not
    reconcile fails the run instead of quietly shifting a bar.
    """
    st = wpath.stages.get(stage_key)
    if st is None or not corr.get("ok"):
        return {"stage": stage_key, "corrected": False,
                "moved_pJ": 0.0, "note": corr.get("provenance", "no such stage")}
    before = st.energy_pJ
    after, note = correct_level_energy(corr, before, st.reads, st.fills)
    if after == before:
        return {"stage": stage_key, "corrected": False, "before_pJ": before,
                "after_pJ": before, "moved_pJ": 0.0, "note": note}
    scale = after / before
    st.energy_pJ = after
    st.switch_pJ *= scale
    st.wire_pJ *= scale
    return {"stage": stage_key, "corrected": True, "before_pJ": before,
            "after_pJ": after, "moved_pJ": after - before, "scale": scale,
            "note": note}


def dilated_levels(arch, cfg):
    """`(stage_key, prefixes)` for the stage the dilation rewrote, or None.

    The dilation scales the WEIGHT-carrying storage levels; the one whose
    per-access cost the correction has to undo is the innermost of them, which
    is also the stage every PE-local boundary sits at. A design whose weight
    levels are all `depth: 1` latches is not dilated at all and returns None,
    which is what makes `RECON_OPTIMIZER=True` refuse on it rather than draw a
    figure that claims a capacity effect it cannot have.
    """
    from . import archs as archmod
    rows = [r for r in archmod.weight_capacity_levels(arch, cfg) if not r["latch"]]
    if not rows:
        return None
    level = rows[-1]["level"]
    for stage in stages_for(arch, cfg):
        if stage.matches(level):
            return stage.key, stage.prefixes
    return None


def storage_access_pj(stats_dir, prefixes):
    """(read_pJ, write_pJ, provenance) for a storage level, from the design's ERT.

    Task 4's dilation correction needs this: to reprice a level whose read and
    write counts move apart, the two per-access energies have to be separable,
    and Timeloop's stats print one `Energy (total)` per dataspace. The split is
    taken from the same Accelergy ERT the mapper cache already holds beside
    every mapping. Returns `(None, None, why)` when the ERT is missing, so the
    caller can REFUSE rather than estimate.
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


def narrowing_audit_for_bar(placement, wpath, packing, stage_defs):
    """prompt_6 RULE 1, once per BAR per STAGE: for every storage stop the
    placement says carries narrow weights, who narrows it -- from the bar's
    OWN measured `Word bits` and the scale the evaluator would apply.

    Returns `(ok, rows)`; a row with `problem` set names a stop narrowed twice
    or by nobody. `evaluate_placement` calls this and refuses on `not ok`.
    """
    rows = []
    for stage in stage_defs:
        if stage.kind != "storage" or stage.key not in placement.reduced:
            continue
        st = wpath.stages.get(stage.key)
        if st is None or st.energy_pJ <= 0:
            continue
        try:
            scale = packing.storage_scale(st.block_bits, st.word_bits)
            row = packing.narrowing_site(st.word_bits, scale)
        except ValueError as exc:            # Word bits neither q nor weight_bits
            row = {"measured_word_bits": st.word_bits, "q": packing.declared_q,
                   "weight_bits": packing.weight_bits, "owner": "UNDEFINED",
                   "mapper_live": None, "evaluator_live": None,
                   "applied_scale": None, "ok": False, "problem": str(exc)}
        rows.append(dict(row, stage=stage.key, levels=list(st.levels),
                         physical_word_bits=st.block_bits))
    return all(r["ok"] for r in rows), rows


def engine_cycles_for(placement, wpath, stage_defs, cycles=None):
    """prompt_6 RULE 3's idle denominator for one bar: engine-cycles, i.e.
    sum over the plan's layers of (engines that leak x that layer's cycles).
    Returns `(engine_cycles, effective_engines, rule)`.

        dram site      1 engine at the chip ingress                x cycles
        network site   fanout x network instances = the destinations
                       the network delivers to (14 at Eyeriss v1's
                       column edge)                                x cycles
        storage site   the level's UTILIZED instances, PER LAYER: Timeloop
                       power-gates each unused instance (`Instances sharing
                       power gating: 1`) and bills `leak` x utilized x cycles
                       (buffer.cpp FinalizeBufferEnergy; verified 2026-09-11
                       on 43 shapes, multiplier == utilized every time). An
                       encoder sits in its PE and is gated with it. The
                       declared count (168) is reported beside it.

    `cycles` is the run length the caller bills from (RULE 4: another plan's
    cycles override the path's own); the storage engine-cycles are rescaled
    to it so `idle x engine_cycles` and `idle x cycles x engines` agree.
    """
    by_key = {s.key: s for s in stage_defs}
    site = by_key[placement.site_stage]
    st = wpath.stages[placement.site_stage]
    own = float(getattr(wpath, "cycles", 0.0) or 0.0)
    cyc = float(cycles) if cycles is not None else own
    if site.kind == "dram":
        return 1.0 * cyc, 1.0, "one engine at the chip ingress"
    if site.kind == "network":
        n = max(1, int(round(float(st.fanout) * float(st.instances or 1))))
        return n * cyc, float(n), (f"one engine per destination of the network: fanout "
                                   f"{st.fanout:g} x {st.instances:g} network instance(s)")
    ec = float(st.engine_cycles or 0.0)
    if own and cyc != own:
        ec *= cyc / own
    eff = (ec / cyc) if cyc else float(st.declared_instances or st.instances or 1)
    return ec, eff, (f"one engine per UTILIZED instance of {'+'.join(st.levels) or site.key}, "
                     f"per layer (Timeloop power-gates unused instances and bills leak x "
                     f"utilized x cycles); declared {st.declared_instances:g}, utilized "
                     f"max {st.instances:g}, cycle-weighted mean {eff:.2f}")


def engines_for(placement, wpath, stage_defs, cycles=None):
    """`(effective engines, rule)` -- the cycle-weighted mean of the engines
    that leak; see `engine_cycles_for`. 1 / 14 / <= 168 on Eyeriss v1."""
    _ec, eff, rule = engine_cycles_for(placement, wpath, stage_defs, cycles)
    return eff, rule


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
                       recon_pj, gran, packing, decode_pj=0.0, recon_idle_pj=0.0,
                       cycles=None):
    """Cost one boundary. Fixed mapping: every access count is Timeloop's.

    prompt_6 RULE 3: `recon_pj` is the INCREMENTAL term (pJ per codeword
    event) and `recon_idle_pj` the idle term (pJ per cycle per engine);
    `cycles` is the run length of the plan this bar is billed from (None =
    `wpath.cycles`). The encoder energy is

        E_recon = incremental x events + idle x cycles x N_engines

    with N_engines derived per placement by `engines_for` (1 / 14 / 168 on
    Eyeriss v1). An idle term without a cycle count is refused, never zero.

    `base_by_cat` / `base_w_by_cat` are the plotted-category totals and their
    weight share, from the SAME `Raw` record the baseline and embedded arms use.
    A stage's saving is subtracted from its own category, so the stack still
    sums to the total. The DRAM category moves by exactly its K/N reduction and
    no more: the single `dram` stage is in every `reduced` set when the decoder
    is on the die
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
    # prompt_6 RULE 1: exactly one owner per narrow storage stop, decided per
    # bar from ITS plan's measured Word bits. A hard stop either way it fails.
    narrow_ok, narrowing = narrowing_audit_for_bar(placement, wpath, packing, stage_defs)
    if not narrow_ok:
        bad = [r for r in narrowing if not r["ok"]]
        raise ValueError(
            f"{arch}/{placement.key}: " + "; ".join(
                f"{r['stage']} ({'+'.join(r['levels'])}): {r['problem']}" for r in bad))

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
        if stage.kind == "dram":
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
            # RULE 1: the owner of this stop's narrowing is decided by the
            # bar's own measured Word bits -- 1.0 when the plan is q-bit.
            scale = packing.storage_scale(st.block_bits, st.word_bits)
            after = st.energy_pJ * scale
        saved = st.energy_pJ - after
        saved_by_cat[cat] = saved_by_cat.get(cat, 0.0) + saved
        stage_rows.append({**st.to_dict(), "category": cat,
                           "carries_reduced": True, "scale": scale,
                           "energy_after_pJ": after, "energy_saved_pJ": saved})

    # ---- 2. what reconstruction costs --------------------------------------
    st_site = wpath.stages[placement.site_stage]
    accesses = st_site.counter(placement.site_counter)
    n_cw = gran.codewords(accesses)
    e_incremental = n_cw * recon_pj
    if cycles is None:
        cycles = float(getattr(wpath, "cycles", 0.0) or 0.0)
    engine_cycles, engines, engines_rule = engine_cycles_for(
        placement, wpath, stage_defs, cycles)
    if recon_idle_pj and not cycles:
        raise ValueError(
            f"{arch}/{placement.key}: the idle term is {recon_idle_pj:g} pJ per cycle "
            f"per engine but the plan carries no cycle count (RULE 3: refusing to "
            f"charge it as zero; the stats file has no `Cycles:` line)")
    e_idle = float(recon_idle_pj or 0.0) * engine_cycles
    recon_energy = e_incremental + e_idle

    # No reconstruction boundary carries a reuse register any more (R4b was
    # removed 2026-09-10: consecutive weight reuse is 1 on 20 of 21 resnet18
    # layers, so a latch catches nothing, and a register that DOES pay has to
    # hold the whole inner tile -- up to 384 weights, the entire scratchpad).
    # `Recon overhead` is kept as a category so the stacks, the legend and the
    # result schema are unchanged; it is structurally zero.
    overhead = 0.0

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
                },
        },
        "feasibility": feas,
        "narrowing_ownership": {
            "rule": ("prompt_6 RULE 1: per storage stop carrying narrow weights, "
                     "the owner is decided by this bar's own measured Word bits -- "
                     "q means the mapper narrowed it (evaluator x1.0), weight_bits "
                     "means the evaluator does; exactly one site is live"),
            "stops": narrowing},
        "reduced_representation": packing.to_dict(),
        "reconstruction_granularity": gran.to_dict(),
        "reconstruction_counts": {
            "weights_reconstructed": accesses,
            "counter": placement.site_counter,
            "counter_meaning": {
                "reads": "one reconstruction per weight read out of the stage",
                "fills": "one reconstruction per weight written into the stage",
                "ingresses": ("one reconstruction per weight word INJECTED "
                              "into the network: ONE encoder before the fanout "
                              "(ECC_RECON_ENCODER_SITE=source)"),
                "deliveries": ("one reconstruction per weight word ARRIVING at "
                               "a destination of the network, i.e. ingresses x "
                               "multicast factor: one encoder PER DESTINATION "
                               "(ECC_RECON_ENCODER_SITE=destination)"),
            }[placement.site_counter],
            # Both readings of a network boundary, always, so the multiplicity
            # is visible rather than implied by which counter was chosen.
            "encoder_site": encoder_site(cfg),
            "network_multicast_factor": st_site.multicast,
            "network_fanout": st_site.fanout,
            "if_one_encoder_before_the_fanout": st_site.ingresses,
            "if_one_encoder_per_destination": st_site.deliveries,
            "multicast_multiplicity_of_this_boundary": (
                (st_site.deliveries / st_site.ingresses)
                if st_site.ingresses > 0 else 1.0),
            "reconstruction_events_codewords": n_cw,
            "pJ_per_codeword": recon_pj,
            "incremental_pJ_per_codeword": recon_pj,
            "idle_pJ_per_cycle_per_engine": float(recon_idle_pj or 0.0),
            "engines": engines,                 # cycle-weighted mean of the engines that leak
            "engines_declared": float(st_site.declared_instances or 0.0),
            "engines_utilized_max": float(st_site.instances or 0.0),
            "engine_cycles": engine_cycles,     # sum over layers: engines x cycles
            "engines_rule": engines_rule,
            "cycles": float(cycles or 0.0),
            "reconstruction_energy_incremental_pJ": e_incremental,
            "reconstruction_energy_idle_pJ": e_idle,
            "rule": ("prompt_6 RULE 3: incremental x events + idle_per_cycle x "
                     "engine_cycles, engine_cycles = sum over layers of (engines "
                     "that leak x cycles); a storage site's engines are the "
                     "UTILIZED instances, as Timeloop bills leak (power-gated "
                     "per instance); the two terms have different denominators "
                     "and are never added per codeword"),
            "reconstruction_energy_pJ": recon_energy,
            # R4b and its reuse register were removed on 2026-09-10; no
            # boundary carries per-PE buffering, so this is structurally zero.
            "recon_overhead_energy_pJ": overhead,
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
    experiments/recon.py (`dram_scaled_by_K_over_N`) reads these rows AND the DRAM
    component, so a bar cannot report one thing here and draw another.
    """
    rows = {r["stage"]: r for r in stage_rows if r["kind"] == "dram"}
    d = rows.get("dram", {})
    before = d.get("weight_energy_pJ", 0.0)
    after = d.get("energy_after_pJ", before)
    reduced = "dram" in placement.reduced
    return {
        "decode_site": wpath.decode_site,
        "dram_pj_per_bit": wpath.dram_pj_per_bit,
        "dram_cost_provenance": wpath.dram_cost_note,
        "dram_weight_energy_pJ": before,
        "dram_after_pJ": after,
        "dram_reduced": reduced,
        "dram_scale": packing.frac if reduced else 1.0,
        "dram_saving_pJ": before - after,
        "rule": ("the decoder is on the DRAM die and off the fetch path: only "
                 "the k message bits of each n-bit codeword are read out and "
                 "driven off the die, so the WHOLE DRAM weight term is scaled "
                 "by K/N under every boundary, R1 included. The f_if "
                 "array/interface split was removed on 2026-09-09. "
                 "E_background and E_refresh are not modelled (both 0)"),
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
        "dram_term": wpath.dram_term(),
        "rule": ("this module re-parses the cached stats and must reproduce the "
                 "`Raw` record's WEIGHT energy per category exactly; a level "
                 "carrying weight energy that no stage claims is listed above "
                 "and makes this check fail. The DRAM category is the one "
                 "`dram` stage: the array/interface split by f_if was removed "
                 "on 2026-09-09 and the whole term is reducible"),
    }


