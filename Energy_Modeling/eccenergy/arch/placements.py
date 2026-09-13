"""The reconstruction boundaries of a design: what stays reduced, and where.

`PLACEMENTS[<arch>]` is the list of boundaries evaluated for one design, each a
`Placement`: which stages of `weight_path.py`'s table keep the REDUCED
representation, where the encoder sits, and what drives the reconstruction
count. The two tables ARE the placement space and must be edited together.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
`stacks.build_stacks()`'s `recon` arm is ONE placement, applied to every
architecture at once: it scales the weight share of every on-chip category by
K/N and charges one reconstruction per DRAM codeword. That is a single point in
the space these tables describe, and it is NOT one of the boundaries below -- it
reduces the SPad *and* charges reconstruction only once per DRAM fetch, which no
physical boundary does, and it must never be quoted as one.

Designs do not have the same boundaries, because each has its own weight path.
`eyeriss_like_wglb` (Eyeriss v1) and `simple_weight_stationary` have five;
`eyeriss_v2_like` has four. `eyeriss_v2_like_wglb` is REFUSED -- its `weight_glb`
stage has no boundary (prompt_6 Appendix B has the fix).

THE FIVE BOUNDARIES
-------------------
Sec. 5.2/5.3 of `01_project_context_and_architectures.txt`, in the order the plan
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
2026-09-10. Measured on the cached mappings, consecutive weight reuse is 1 on 20
of 21 resnet18 layers, so a latch-sized register catches nothing (a cyclic walk
is the LRU worst case, with no partial hit rate); a register that DOES pay has
to hold the whole inner tile, up to 384 weights against a 384-weight scratchpad.
The same argument rules it out on every dataflow in the study: output- and
input-stationary hold the psum or the activation in the PE, so weights stream
past the MAC faster still. FINDINGS 7.1. Do not reintroduce one without first
re-measuring `consecutive_run` on the target mapping.

A NETWORK boundary's encoders run once per ARRIVAL, not once per injection --
the count is `Ingresses x Multicast factor`. Charging the ingress count pairs a
destination-side saving with a source-side cost, which is not a placement.
`ECC_RECON_ENCODER_SITE=source` reproduces the earlier numbers for a diff.

ProjectRestructure phase 3 cut this out of `recon.py`. It is L2, beside the
weight path it indexes, which is what lets `arch/fingerprint.py` reach a
placement without importing a study module.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .weight_path import DECODE_SITES, ENCODER_SITES, WEIGHT_PATHS


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
        reduced=("dram", "weight_glb"),
        site_stage="weight_glb", site_counter="reads",
        description=(
            "Sec. 6.2's R1, rated 3/5: the 36 kB weight buffer holds weight "
            "tiles in the reduced form, so its weight capacity and its weight "
            "access bit-volume both fall, and one encoder at its read port "
            "restores full width before the distribution network. Since "
            "2026-09-13 this level keeps Weights alone, so the whole level "
            "moves and the input activations -- now `input_glb` -- are a "
            "separate array that nothing here touches. Downstream -- network, "
            "scratchpad, register -- is full width."),
    ),
    Placement(
        "recon3", "recon_noc_output_pe_input",
        "R3 - reconstruct at the weight-NoC output / PE input",
        "R3\n@ PE input", "4/5",
        reduced=("dram", "weight_glb", "weight_noc"),
        site_stage="pe_spad", site_counter="fills",
        description=(
            "Sec. 6.2's R2, rated 4/5: the weight buffer AND the 256-PE "
            "distribution network carry the reduced form, and an encoder at "
            "each PE input "
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
        reduced=("dram", "weight_glb", "weight_noc", "pe_spad"),
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
        reduced=("dram", "weight_glb", "weight_noc", "pe_spad",
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


