"""The weight path of one design: the stages, and the Timeloop levels that ARE them.

`WEIGHT_PATHS[<arch>]` names the stages a weight crosses on its way from DRAM to
the MAC, outer to inner, and for each of them the Timeloop level names that make
it up. It is half of the placement space; `placements.py` is the other half, and
the two MUST BE EDITED TOGETHER -- `validate_placement_space()` (in
`arch/arms.py`) is what refuses a half-finished pair.

THE WEIGHT PATH IS READ FROM THE MODEL, NOT ASSUMED
---------------------------------------------------
`01_project_context_and_architectures.txt` Sec. 5.1 warns against substituting

    DRAM --> global weight SRAM --> RF --> MAC

for Eyeriss v2, whose GLB banks serve input activations and partial sums while
weights go from the memory interface through the hierarchical mesh to the PE
scratchpads. The tables here therefore name the levels of the architecture AS
MODELLED, and `weight_path()` (`toolchain/weight_stats.py`) refuses to proceed
if a level that carries Weights energy is not claimed by exactly one stage. A
placement whose stage is absent from the model is reported `unsupported`, never
silently skipped -- for `eyeriss_v2_like` there is no weight GLB at all, so no
"reconstruct at the global buffer" boundary exists to evaluate.

THE DRAM IS ONE STAGE, AND THE WHOLE OF IT IS REDUCIBLE  (2026-09-09)
---------------------------------------------------------------------
Accelergy's CactiDRAM bills a DRAM read as one flat per-bit DYNAMIC ACCESS
constant -- 8 pJ/bit for LPDDR4 as modelled, verified at 64.0 pJ per 8-bit
word -- and Timeloop charges the DRAM-to-chip network zero, so that constant is
the entire DRAM term:

    dram = DRAM weight energy x K/N   under EVERY boundary, R1 included

It used to be TWO stages, `dram_array` ((1 - f_if) of it, never reduced) and
`dram_interface` (f_if of it, reduced), with `f_if = ECC_DRAM_IF_FRAC = 0.40`.
That split charged a 38.1% bit cut as a 15.2% energy cut and is REMOVED: the
DRAM access is designed to fetch only the message bits of each codeword, so the
whole per-bit constant carries k of every n bits. `ECC_DRAM_PJ_PER_BIT` sets the
per-bit cost itself (`energy.apply_dram_override`); `archs/_shared/provenance.yaml`
`dram_access_energy` carries the citations. E_background and E_refresh are NOT
modelled (both 0) -- see env.sh section 4.

THE DECODER IS ON THE DRAM DIE  (since 2026-09-09)
--------------------------------------------------
01_project_context Sec. 1: the BCH decoder sits on the DRAM die and OFF the
fetch path (it corrects at write, on a scrub pass or on a prior access), so at
fetch time the stored codeword is already corrected and only the k message bits
of each n-bit codeword are driven across the DRAM interface. The array still
stores and reads the complete codeword -- a row activation and a burst move
whole words -- so `dram` is reduced under every boundary, R1 included.
`ECC_RECON_DECODE_SITE=controller` is the pre-2026-09-09 model (decode on the
fetch path, the complete codeword across the interface, DRAM identical on every
bar), kept as a runnable row so the change can be diffed; `stages_for()` and
`placements_for()` take the config and hand back that space when it is asked
for. The two reference bars keep controller-side correction under both settings:
Task 1's external parity is read from the array AND crosses the interface, and
Task 2's on-chip datapath consumes every weight bit, so its complete codeword
crosses too. Their functions are frozen and do not change.

ProjectRestructure phase 3 cut this out of `recon.py`, which was four layers in
one file. It is L2: a design's weight path is a fact about the design.
"""
from __future__ import annotations

from dataclasses import dataclass


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
    Stage("weight_glb", "Global weight buffer (36 kB)", "storage",
          ("weight_glb",), reducible=True,
          evidence="Sec. 6's 'global/weight buffer'. "
                   "`archs/simple_weight_stationary/arch_paper.yaml` splits the "
                   "stock 128 kB shared_glb by DATASPACE into three levels, and "
                   "this is the 36 kB weight one. It keeps Weights and nothing "
                   "else (`dataspace: {keep: [Weights]}`), so every bit of its "
                   "energy is weight energy -- which is also what lets it "
                   "declare the arm's `datawidth: q`. Until 2026-09-13 this "
                   "stage was a 64 kB `operand_glb` holding Inputs beside "
                   "Weights, and Timeloop's one-datawidth-per-level rule meant "
                   "the reduced representation could not be declared on it at "
                   "all."),
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


