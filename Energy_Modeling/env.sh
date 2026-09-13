#!/usr/bin/env bash

# =============================================================================
#  1. THE FEW YOU CHANGE MOST OFTEN
# =============================================================================

# WHICH GUARDS YOU HAVE DELIBERATELY LIFTED. Empty on every published run.
#
# A guard runs on EVERY run -- including a SLURM job at 3am -- and stops it.
# `GUARDS.md` lists all 135 of them with a stable id and a TIER, and the tier is
# the only thing that matters here:
#
#   1 PARSE       "ECC_VICTORY=abc is not an integer"        NEVER liftable
#   2 IMPOSSIBLE  "need K < N";  "a price may not be negative" NEVER liftable
#   3 COUPLING    "ERT_AWARE=1 needs RECON_OPTIMIZER=True"   liftable, named here
#   4 DERIVED     "the arm derives q; leave the knob EMPTY"  liftable, named here
#
# Tiers 3 and 4 are the ABLATION BLOCKERS. Name one and it becomes a loud warning
# instead of a refusal -- and THE OVERRIDE IS RECORDED, on the run manifest as
# `guard_overrides` and on the figure's caveat list, so an ablation cannot be
# published as if it were the study's own number.
#
# NAME THE GUARD. There is deliberately no blanket "off": a blanket would be set once, forgotten, and a wrong number would reach a figure with nothing saying so
#   ECC_ALLOW="zero-price"                what if this term were free?
#   ECC_ALLOW="derived-datawidth"         test a hypothetical q against an arm
#   ECC_ALLOW="zero-price,derived-datawidth"
: "${ECC_ALLOW:=}"

ECC_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${ECC_MAPPER_THREADS:=${SLURM_CPUS_PER_TASK:-18}}"

: "${ECC_LAYERS:=layer3.0.conv1}"

# Run every stage inside the Timeloop+Accelergy container (1) or with the host
# python3 (0). The mapper ALWAYS needs the container; evaluation and plotting do
# not, but they need pandas/matplotlib/pyyaml, which a login node may lack.
: "${ECC_USE_CONTAINER:=1}"


# =============================================================================
#  2. THE MAPPING OPTIMISER  --  what Timeloop's search does
# =============================================================================

# random | hybrid | exhaustive | linear_pruned | random_pruned
: "${ECC_MAPPER_ALGORITHM:=linear_pruned}"

: "${ECC_MAPPER_SEARCH_SIZE=}"

: "${ECC_VICTORY:=2000}"

# Does the mapper stop when it finds a mapping that is feasible and better than
: "${ECC_VICTORY_SCALING:=levels}"

: "${ECC_MAPPER_TIMEOUT:=100000000}"

: "${ECC_MAPPER_MAX_PERMUTATIONS:=16}"

: "${ECC_OPT_METRIC:=edp}"

: "${ECC_MAPPER_SEED:=}"


# =============================================================================
#  3. WHAT THE PIPELINE RUNS
# =============================================================================

# 1 -> re-run | 0 -> a valid cache hit is reused
: "${ECC_RERUN_OPTIMISER:=0}"


: "${ECC_ARCHS:=eyeriss_v2_like eyeriss_like_wglb simple_weight_stationary simple_output_stationary simple_input_stationary simba_like}"

: "${ECC_MODELS:=resnet18 mobilenet_v2}"

# the BCH code
: "${ECC_CODE_N:=63}"
: "${ECC_KS:=30}"

: "${ECC_APPROACHES:=baseline embedded recon}"

: "${ECC_SWEEP:=arch}"

# Which evaluations are written per model, in order. `baseline` is Task 1's
# conventional-ECC file; `embedded` is Task 2's, which holds the SAME baseline
# plus the embedded arm from the same cached mappings and checks that only the
# DRAM component moved.
: "${ECC_EVAL_EXPERIMENTS:=baseline embedded}"

# Which half of the study a result belongs to.
#   Pre   the mapping is ECC-unaware; the ECC effect is applied when evaluating
#         (Tasks 1-3). Task 1 is Pre by construction.
#   Post  the mapping itself was optimised for the reduced weight width (Task 4+)
: "${ECC_PHASE:=Post}"


# =============================================================================
#  4. RECONSTRUCTION PLACEMENT STUDY   --   TASK 3, ONE ARCHITECTURE AT A TIME
# =============================================================================
: "${ECC_RECON_MODELING:=1}"

#
# CORRECTED 2026-09-11: this comment used to say True "IS A PLACEHOLDER:
# setting it stops the run with an error". That has been FALSE since
# 2026-09-09 -- Task 4 is implemented. What the code refuses now is only the
# one combination that cannot mean anything, a re-optimised mapping filed as a
# `Pre` result. The guarantee the placeholder gave is kept by
# `experiments/recon.dilated_view()` (stops when the reconstruction arm's OWN
# mapper cache is absent, or when the dilated capacity does not come back N/K
# times the reference's) and by `task4_checks()`, which records both mapping
# fingerprints on every result.
#
# MUST AGREE WITH ECC_PHASE, and config.py refuses both contradictions:
#     ECC_PHASE=Post  RECON_OPTIMIZER=True    Task 4   <- a second mapping IS solved
#     ECC_PHASE=Pre   RECON_OPTIMIZER=False   Task 3   <- one fixed mapping
# `RECON_OPTIMIZER=True` with `ECC_PHASE=Pre` loads NOTHING -- every command
# that reads the config dies on it, including --dry-run.
: "${RECON_OPTIMIZER:=True}"

# prompt_6 -- RECONSTRUCTION-AWARE MAPPING, ONE PLAN PER BOUNDARY. 1 puts the
# encoder's energy into the mapper's objective and bills each boundary from
# the plan of ITS OWN CHIP. Requires RECON_OPTIMIZER=True and ECC_PHASE=Post;
# config.py refuses anything else. 0 = Task 4 as before.
#
# SINCE prompt_7 PHASE B (2026-09-12) the arms are the DISTINCT CHIPS, not
# the ERT-injectable boundaries: SIX on Eyeriss v1 (+filter GLB), five on
# v2. A bar whose own chip is not mapped yet is billed from a NAMED plan
# that narrows the SAME storage levels and says so on its record -- R3 from
# R2, never from the reference. `bash hpc/map_ert_arms.sh --progress` says
# which arms are cached and which are cold; Phase C maps the cold ones.
: "${ECC_RECON_ERT_AWARE:=0}"

# WHICH ARM ONE MAPPER JOB SOLVES -- set by hpc/map_ert_arms.sh in --export,
# not by hand. `reference` (or EMPTY) is the published 8-bit chip with no
# toll; ANY placement key (recon1 .. recon5) is that boundary's chip.
# config.py resolves it into ECC_WEIGHT_DATAWIDTH=q on the storage levels in
# the placement's reduced set -- filter_glb on R2/R3/R4, filter_glb AND
# weights_spad on R5a, and NOTHING on R1, which narrows nothing on chip and
# must therefore leave the datawidth alone (q with an empty level list is the
# spelling that narrows EVERY weight level, a different chip). Where the
# boundary is ERT-injectable, archs.ert_bump() also derives the ERT delta
# (incremental + idle x g on the access action, idle x (1 - g) on leak).
# The arm is in the cache slug AND the fingerprint, so two arms with
# byte-identical YAML never share a directory (RULE 4.4.5): an arm with a
# bump is spelled `ert-recon2-filter_glb-read` and one without is
# `arm-recon1`. R1's patched YAML IS the reference's until Phase C1.2
# declares the bandwidth scale, so that second spelling is what keeps them
# apart today.
: "${ECC_RECON_ERT_ARM:=}"

# The single point the placement study is run at. These REPLACE section 3's
# lists when ECC_RECON_MODELING=1, so change the point here, not there.
# ONE PANEL PER NAME, top panel first, each from section 3's list. Every design
# named here needs its own mapper cache at the CURRENT fingerprint -- map it
# with `ECC_RECON_ARCHS=<one> bash hpc/map_by_shape.sh` before adding it, or the
# run stops and says which one is missing rather than drawing a short figure.
# Only designs with a weight path in eccenergy/recon.py WEIGHT_PATHS are
# accepted; ECC_RECON_ARCH is the older one-architecture spelling and seeds the
# list when ECC_RECON_ARCHS is not set.
# : "${ECC_RECON_ARCH:=eyeriss_v2_like}"    # the FIRST architecture / the old knob
: "${ECC_RECON_ARCHS:=eyeriss_like_wglb}"
#  ^ PROMPT_2 IS A ONE-DESIGN STUDY: "For Eyeriss V1, one layer at a time".
#    eyeriss_like_wglb IS Eyeriss v1 since 2026-09-10, and it is newly
#    registered in recon.py's WEIGHT_PATHS/PLACEMENTS with FIVE boundaries --
#    one more than eyeriss_like, because the filter GLB is a reducible storage
#    stage above the array network and admits a boundary at its output.
#    The two-panel figure the previous plan drew is reproduced with
#        ECC_RECON_ARCHS="simple_weight_stationary eyeriss_like" \
#            bash hpc/run_all.sh --eval-only
#    though `eyeriss_like`'s caches are cold after the 2026-09-10 swap.
#    Unset, this falls back to ECC_RECON_ARCH, the old one-architecture knob.
# "resnet18", "resnet50", "densenet121", "squeezenet1_1",
# "mobilenet_v2", "efficientnet_b0", "convnext_tiny", "xception"
: "${ECC_RECON_MODEL:=resnet18}"          # ONE model
# ONE LAYER of that model, or every layer of it (prompt_6 8.2). A single-layer
# run is what makes an ERT-aware sweep affordable: (1 + ERT arms) jobs
# instead of that many times the layer count. It SEEDS ECC_LAYERS in section
# 10 whenever ECC_RECON_MODELING=1, so under the placement study set THIS
# name, not ECC_LAYERS. Empty here = the whole model; on the command line
# spell the whole model `ECC_RECON_LAYER=all` (an exported empty string is
# indistinguishable from unset to `:=` and would take this default).
# layer3.0.conv1 = C128_M256_R3_S3_P14_Q14_ws2_hs2, 294,912 weights, 168/168
# PEs, refetch 1.000, 344,064 cycles: full PE occupancy, no `PE !=` confound.
# : "${ECC_RECON_LAYER:=layer3.0.conv1}"
: "${ECC_RECON_LAYER:=all}"
: "${ECC_RECON_CODE_N:=63}"               # ONE code geometry
: "${ECC_RECON_K:=39}"                      # 54 51 45 39 36 30   # ONE code rate, from section 3's list
                                          # 30 is prompt_2's code -- see section 3.

# The placements explored, per architecture: one bar per entry, left to right.
# `baseline` and `embedded` are ALWAYS drawn as reference bars -- a placement is
# meaningless read on its own -- so they need not be listed. An EMPTY entry, or
# an architecture missing from this array, draws every placement the design
# defines, which is the normal thing to want.
#
# WHAT recon1..recon5 MEAN is a property of the architecture's weight path, not
# of this file, so the boundaries, their labels and which hierarchy levels each
# one leaves reduced are defined together in `eccenergy/recon.py`
# (WEIGHT_PATHS and PLACEMENTS). For eyeriss_v2_like, from Sec. 5.2/5.3 of
# 01_project_context_and_architectures.txt:
#   recon1  R1   reconstruct at the source / weight-NoC ingress        [2/5]
#                (since 2026-09-09 it reduces the DRAM interface and nothing on
#                chip: it isolates the interface saving every boundary shares
#                from any on-chip saving, and is no longer a zero-saving control)
#   recon2  R2   reconstruct at the destination-cluster boundary       [4/5]
#   recon3  R3   reconstruct at the PE weight-SPad input               [4/5]
#   recon4  R4a  reconstruct on every weight-SPad read                 [3/5]
# The bracketed ratings are the source discussion's HYPOTHESES, not results.
# R4b (SPad output plus a reconstructed-weight reuse register) was REMOVED on
# 2026-09-10: consecutive weight reuse is 1 on 20 of 21 resnet18 layers, so a
# latch catches nothing, and a register that does pay has to hold the whole
# inner tile -- up to 384 weights, the entire scratchpad. See FINDINGS 7.1.
#
# A bash associative array CANNOT be exported, so section 10 flattens the entry
# for ECC_RECON_ARCH into ECC_RECON_PLACEMENT_LIST, which is what the code reads.
# For eyeriss_like -- RETIRED 2026-09-10, kept only so the pre-swap figure can
# be reproduced. It declares `!Nothing` where the published 8 kB filter GLB
# sits, so it has FOUR boundaries and no global-buffer boundary at all. Eyeriss
# v1 is eyeriss_like_wglb; do not read these four keys as v1's:
#   recon1  R1   reconstruct at the source, before the array network      [3/5]
#   recon2  R2   reconstruct after the array multicast, at the column edge[4/5]
#   recon3  R3   reconstruct at the PE filter-spad input                  [4/5]
#   recon4  R4a  reconstruct on every filter-spad read                    [3/5]
#
# For simple_weight_stationary, from Sec. 6.1/6.2 -- SIX, because it is the only
# design in the study with both a weight global buffer above the network and a
# stationary weight register below the scratchpad:
#   recon1  R1   reconstruct at chip ingress, before the weight buffer [control]
#   recon2  R2   reconstruct at the global weight-buffer output           [3/5]
#   recon3  R3   reconstruct at the weight-NoC output / PE input          [4/5]
#   recon4  R4a  reconstruct on every weight-RF read                      [2/5]
#   recon5  R5   reconstruct at the MAC input (register reduced too)      [2/5]
# recon5 here is expected to be reported INFEASIBLE, not to produce a number:
# the register holds one weight and a rebuild needs G_rec = 9 co-resident. It
# is listed so the study answers Sec. 6.2's MAC row instead of omitting it.
# For eyeriss_like_wglb -- EYERISS v1 since 2026-09-10 -- FIVE, because the
# published 8 kB filter GLB is a reducible storage stage ABOVE the array
# network and admits a boundary at its output that eyeriss_like has nowhere to
# put. The numbering follows simple_weight_stationary's, the other design with
# a weight buffer above its network: recon2 is the global weight buffer on
# both.
#   recon1  R1   reconstruct at chip ingress, before the filter GLB    [control]
#   recon2  R2   reconstruct at the filter-GLB output                     [3/5]
#   recon3  R3   reconstruct after the array multicast, at the column edge[4/5]
#   recon4  R4   reconstruct at the PE filter-spad input                  [4/5]
#   recon5  R5a  reconstruct on every filter-spad read                    [3/5]
declare -A ECC_RECON_PLACEMENTS=(
    [eyeriss_v2_like]="recon1 recon2 recon3 recon4"
    [eyeriss_v2_like_wglb]="recon1 recon2 recon3 recon4"
    [eyeriss_like]="recon1 recon2 recon3 recon4"
    [eyeriss_like_wglb]="recon1 recon2 recon3 recon4 recon5"
    [simple_weight_stationary]="recon1 recon2 recon3 recon4 recon5"
    # Still to come, each with its own weight path in recon.py:
    #   [simba_like]="..."  [simple_output_stationary]="..."
    #   [simple_input_stationary]="..."
)

# Where the figure, table and manifest are called. Fixed name, like the three
# sweeps: a re-run at a different point REWRITES it and the manifest beside it
# records which point is on disk. A selected-layer run appends its layer scope.
: "${ECC_RECON_STEM:=ReconSweep}"

# ---- how the reduced representation is physically exploited ----------------
# Section 16 of 02_reconstruction_dse_and_implementation.txt: "Reducing weights
# from 8 bits to 4 bits reduces SRAM energy by 50%" is not a claim the hardware
# supports unless the representation is exploited PHYSICALLY.
#   stream   the retained k bits of each n-bit codeword are stored and moved as
#            a packed field with no per-weight alignment -- which is the layout
#            the embedding pipeline already produces, since the codeword IS n
#            consecutive bits of the weight bit stream. Values per physical word
#            and operands per flit rise by n/k, so every reduced stage scales by
#            K/N. This is the default because it is the actual layout.
#   aligned  each reduced weight occupies ceil(weight_bits*K/N) WHOLE bits and
#            nothing is repacked. Wire energy still falls, but a 24-bit
#            scratchpad word holds floor(24/7)=3 seven-bit values -- the same 3
#            it held at 8 bits -- so the access count, and the SRAM energy, do
#            not move. The pessimistic bound the optimistic one hides.
#
# `aligned` IS NOW THE DEFAULT, AND IT IS A DOUBLE-COUNTING FIX, not a change
# of physical assumption. Since 2026-09-10 the MAPPER delivers the on-chip
# narrowing directly, via ECC_WEIGHT_DATAWIDTH (section 5): Timeloop bills
# `vector_access_energy / block_size` with `block_size = width/datawidth`, so
# a narrower declared datawidth already halves the per-weight SRAM energy
# INSIDE the Timeloop number. `stream` would then scale that same saving by
# K/N a SECOND time in the evaluator and the on-chip saving would be SQUARED.
# `aligned`'s docstring describes exactly the right division of labour --
# "each reduced weight occupies a whole number of bits ... the access count,
# and the SRAM energy, do not move at all" -- so it leaves the on-chip
# narrowing entirely to the mapper, which is now where it belongs.
# `recon.assert_onchip_narrowing_once()` STOPS the run if both are active, and
# also if the declared datawidth disagrees with ceil(weight_bits*K/N).
# Set `stream` with ECC_WEIGHT_DATAWIDTH empty to reproduce the pre-2026-09-10
# evaluator-side model.
: "${ECC_RECON_PACKING:=aligned}"

# ---- does a PE-local boundary need a WHOLE group resident? ------------------
# 0 (default) = NO. 1 = refuse the placement where it does not.
#
# WHAT G_rec IS. The embedded layout treats the weights as ONE BIT STREAM and
# cuts it into n-bit codewords. 63 / 8 = 7.875 is not a whole number, so a
# codeword drifts across weight boundaries and the worst-aligned one reaches
# into ceil(63/8) + 1 = 9 weights. G_rec is that 9: the weights whose retained
# bits one codeword touches.
#
# WHAT THE REFUSAL USED TO ASSUME. That an engine can only rebuild from weights
# co-resident in the level AT ONE INSTANT, so a level holding 6 of the 9 could
# never complete a group and the boundary was reported `unsupported`.
#
# WHY IT IS OFF (decided 2026-09-13). RECAP's reconstruction engine ACCUMULATES
# the retained bits as they arrive and rebuilds when the group completes; it
# does not need all 9 weights resident simultaneously. A level holding 6 -- or
# 1 -- still feeds it, over more accesses and with more buffering. So a small
# resident tile is a COST, not an impossibility, and deleting the bar overstated
# what the hardware cannot do.
#
# WHAT IS STILL MEASURED AND REPORTED, on every bar, in the record and the
# manifest (`group_residency`): how many (layer, stage) pairs sit below G_rec
# and the fewest weights any of them holds. That number bounds the buffer the
# engine needs, so it must not be lost -- it is reported rather than refused,
# the same rule PE-count differences follow.
#
# MEASURED EFFECT of turning it off (mobilenet_v2, BCH(63,39), after prompt_7
# Phase C): R2/R3 recover from 1 layer of 31 below G_rec, R4 from 4, R5a from
# 14 -- four of five placements come back. resnet18 has no layer below G_rec at
# all, so nothing there moves either way.
: "${ECC_RECON_REQUIRE_GROUP_RESIDENCY:=0}"

# ---- how encoder work is charged -------------------------------------------
# Section 15: the encoder may work at CODEWORD granularity, because rebuilding
# one weight can need retained bits from several. G_rec (the weights that must
# be co-resident) is computed from the layout and is 9 at BCH(63,K) over 8-bit
# weights; a PE-local boundary whose resident tile is smaller is REJECTED, not
# estimated.
#   weight    encoder work is proportional to the weights actually rebuilt,
#             charged at the synthesized per-codeword energy per n/weight_bits
#             of them. The amortized reading: the group is rebuilt once and all
#             of it is consumed.
#   codeword  every access at the boundary rebuilds a whole codeword whether or
#             not the rest of the group is used. The pessimistic reading, and
#             the right one if nothing buffers the group.
: "${ECC_RECON_ENCODER_GRANULARITY:=weight}"




# Fraction of the weight bits held on chip. EMPTY = derived from the code (K/N),
# which is what the embedded layout dictates. Set it only for a sensitivity run.
: "${ECC_RECON_ONCHIP_FRACTION:=}"

# Does a placement pay the decoder as well as the rebuild? (section 6 has the
# decoder energies themselves, and ECC_DECODE=0 charges every bar zero.)
: "${ECC_RECON_PLACEMENT_CHARGES_DECODE:=1}"

# ---- where the BCH decoder sits, and what that does to the DRAM term --------
# 01_project_context_and_architectures.txt Sec. 1 and 4. Accelergy's CactiDRAM
# bills a DRAM read as ONE flat per-bit DYNAMIC access constant. Since
# 2026-09-09 that constant is a SINGLE weight-path stage `dram`
# (eccenergy/recon.py WEIGHT_PATHS) and it is reducible in full:
#     dram = DRAM weight energy x K/N   under EVERY boundary, R1 included
# The f_if array/interface split that used to sit here is REMOVED; see
# ECC_DRAM_PJ_PER_BIT below for what replaced it and what it assumes.
#   ondie       the decoder is on the DRAM die and OFF the fetch path (it
#               corrects at write, on a scrub pass or on a prior access), so at
#               fetch time only the k message bits of each n-bit codeword are
#               read out and driven off the die, so the WHOLE DRAM weight
#               term falls by K/N, on every R bar, R1 included. THE DEFAULT.
#   controller  the pre-2026-09-09 model: correction at the memory controller,
#               on the fetch path, so the complete codeword is read AND crosses
#               the interface and the DRAM term is identical on every bar. Kept
#               as a runnable row so the change can be diffed; do not quote it.
# The two reference bars (Task 1 conventional, Task 2 embedded) keep
# controller-side correction under BOTH settings and do not move.
: "${ECC_RECON_DECODE_SITE:=ondie}"

# WHERE A NETWORK BOUNDARY'S ENCODERS SIT, and therefore how many times they
# run. Sec. 7.1 of 01_project_context_and_architectures.txt states the tradeoff
# and asks for it to be an experiment variable:
#
#     BEFORE MULTICAST   4b -> Encoder -> 8b -+-> PE   (x fanout)
#         one reconstruction at the source, but FULL-WIDTH network traffic
#     AFTER MULTICAST    4b -+-> Encoder -> PE          (x fanout)
#         replicated encoder hardware, but REDUCED-WIDTH shared transport
#
#   destination  (the model since 2026-09-09) an encoder at each destination,
#                so the count is Timeloop's destination-side ARRIVALS,
#                `Ingresses x Multicast factor`, read off its own printed
#                breakdown. This is the only count consistent with a boundary
#                that ALSO credits that network with carrying the reduced form:
#                an encoder placed before the fanout would make the network
#                full width, which is the boundary above it.
#   source       ONE encoder before the fanout, count = `Ingresses`. This is
#                what the study charged BEFORE 2026-09-09 and is kept as a
#                runnable row so the change can be diffed. On eyeriss_like's
#                C512 shape the column network multicasts 7-fold, so the two
#                readings differ by 7x in reconstruction energy on R2.
# Only NETWORK boundaries depend on this. R1 counts DRAM codewords, and every
# PE-local boundary already counts destination-side scratchpad accesses.
: "${ECC_RECON_ENCODER_SITE:=destination}"

#20/40/70 pJ/bit = 0.5/1.0/1.75 nJ / 64b, which is the same 45 nm table Horowitz ISSCC 2014 reads from for ECC_MAC_PJ_OVERRIDE=0.23 (section 5), so the two denominators of every percentage share one source.
: "${ECC_DRAM_PJ_PER_BIT:=20}"

: "${ECC_BASELINE_DRAM_PJ_PER_BIT:=22}"

# The other two terms of E_total(DRAM) = E_dynamic + E_background + E_refresh.
: "${ECC_DRAM_BACKGROUND_PJ:=0}"
: "${ECC_DRAM_REFRESH_PJ:=0}"


# 480/240/120 MB/s per 8-bit weight, which is the same 45 nm table Horowitz ISSCC 2014 reads from for ECC_MAC_PJ_OVERRIDE=0.23 (section 5), so the two denominators of every percentage share one source.
: "${ECC_DRAM_BANDWIDTH_MBPS=120}"

# ---- the latency model (prompt_7 Phase A) ----------------------------------
# 1 = re-time the chosen mapping with a ROOFLINE computed AFTER mapping, in
# eccenergy/latency_post.py -- the same evaluator-only pattern noc_post.py uses
# for the two interconnect terms Timeloop cannot be given:
#
#     cycles = max( compute cycles,
#                   every storage level's own declared-bandwidth limit,
#                   off-chip items / ECC_DRAM_BANDWIDTH_MBPS )
#
# It never invokes the mapper and is NOT in the mapping fingerprint: the plan is
# the one Timeloop already chose and this states how long that plan takes. With
# ECC_DRAM_BANDWIDTH_MBPS EMPTY (unlimited) the roofline reproduces Timeloop's
# own per-level and total cycle counts EXACTLY on every cached shape, which is
# what stops it inventing time; only a declared off-chip limit moves a number.
#
# WHY IT EXISTS: `datawidth: q` makes energy fall and cycles stand still,
# because Timeloop's speed model counts ITEMS per cycle and a narrow weight is
# still one item (prompt_7 Defect 1). A reported "0.00% latency gain" is
# therefore an ABSENT TERM, never a result. Off chip the weights are a BIT
# stream, so there the reduced form really does move K/N of the traffic, and
# that is the one place this roofline lets it buy time.
#
# It also decides the run length ECC_STATIC_ENERGY below is charged over: a plan
# that waits on DRAM leaks for longer.
#
# DEFAULT 1 SINCE 2026-09-13 (prompt_7 Phase C). It shipped as 0 in Phase A
# because the roofline was then the ONLY place the off-chip limit existed: the
# mapper optimised a machine with an infinitely fast memory and the evaluator
# re-timed the plan it chose, which is a second timing model and was rightly
# opt-in. C1.1 declares the same limit on the DRAM level, so Timeloop's own
# cycles already carry it and this re-states the plan's time per bar instead of
# supplying the only estimate of it. With it OFF the placement study reports no
# cycles at all, which for a phase about TIME is a missing column, not a
# conservative default.
#
# WHO APPLIES THE OFF-CHIP WEIGHT RELIEF is now MEASURED per bar, not assumed
# (latency_post.relief_owner, prompt_6 RULE 1): the mapper if that bar's stats
# print `Bandwidth Consumption Scale` = K/N, this roofline if they print 1.00,
# and a third value is refused. Without that rule C1.2 and this knob would each
# apply K/N and the reported saving would be K/N squared.
#
# IT STILL MOVES NO ENERGY on its own -- it re-times, and only ECC_STATIC_ENERGY
# below charges anything over that time (default 0).
: "${ECC_LATENCY_MODEL:=1}"

# ---- the DECLARED bandwidth scale of a reduced stage (prompt_7 C1.2) -------
# 1 = every stage in the boundary's `reduced` set declares
#
#     per_dataspace_bandwidth_consumption_scale: {Weights: <factor>}
#
# in the arch YAML the MAPPER sees, so the reduced representation buys TIME and
# not only energy. Timeloop multiplies that one dataspace's bandwidth demand by
# the factor (buffer.cpp:2556); the attribute is declared in timeloopfe v4
# (arch.py:538) and is printed in every stats file as `Bandwidth Consumption
# Scale`, so a run that silently ignored it is visible on disk.
#
# WHICH FACTOR, PER STAGE -- the two differ by 5% and using one everywhere is a
# silent inconsistency with ECC_RECON_PACKING (prompt_7 7.1):
#     DRAM        a BIT stream off the die          -> K/N   (0.47619 at K=30)
#     on chip     whole weights at q bits per word  -> q/8   (0.5     at q=4)
# eccenergy/archs.py derives both from the code in play; neither is typed here.
#
# A NETWORK stage is DECLARED and marked no-op: LegacyNetwork::ComputePerformance()
# is an empty stub, so there is nothing for the factor to reach (reporting rule
# R-3). It still separates two boundaries that differ only by a network, which
# is why recon.mapper_arms() carries it.
#
# TRAP: THIS COLDS EVERY CACHE. The attribute is in the patched YAML and
# therefore in the mapping fingerprint. Batched into prompt_7 Phase C with the
# other architecture changes -- never paid separately.
#
# 0 reproduces the pre-Phase-C architecture byte for byte, which is the mutation
# eccenergy/tests/test_phase_c.py runs.
: "${ECC_RECON_BW_SCALE:=1}"

# ---- a narrowed level's port is BIT-aware (prompt_7 C1.3) ------------------
# 1 = a storage level the arm narrows to `datawidth: q` declares
# `read_bandwidth` and `write_bandwidth` scaled by 8/q, because the port moves
# BITS per cycle and a q-bit weight is fewer bits than an 8-bit one.
#
# WHY IT MATTERS AND WHERE IT LANDS. Timeloop's throughput check counts ITEMS
# per cycle and a narrow weight is still one item, so narrowing alone is
# invisible to the clock (prompt_7 Defect 1). `filter_glb`'s declared 16
# items/cycle is LITERALLY what caps every fully-connected layer at 9.52% PE
# utilisation -- 16 of 168 PEs, measured on resnet18 `fc` and mobilenet
# `classifier.1` (prompt_7 4.5, A.7). At q=5 this raises that level to 25.6
# items/cycle in the narrowed arms and leaves the reference at 16.
#
# HOW BIG IS IT HERE: those layers are 0.28% of resnet18's cycles and 2.10% of
# mobilenet's, so the aggregate CNN effect is ~0.1-1%. On a batch-1 transformer
# every layer is that layer -- which is what prompt_7 Phase D exists to measure.
#
# TRAP: THIS COLDS EVERY CACHE, same as the knob above, and for the same reason.
: "${ECC_ONCHIP_BW_BITAWARE:=1}"

# ---- component standby (leakage) power -------------------------------------
# prompt_7 Issue 5. The study charges NO standby power to the accelerator: the
# `leak` row exists in Accelergy's price list but is never parsed into a total,
# and the prices themselves are not credible (DRAM and two scratchpad types
# come out EXACTLY 0; a 52 kB SRAM is priced at 1.36 uW, ~1000x low, because
# the CACTI config is pinned to the least-leaky transistor recipe there is).
#
# These are the replacement densities, in nW. SRAM is CACTI 45nm `itrs-lop`
# measured on this design's own 52 kB array; RF is the 40nm register number
# from accelergy-aladdin-plug-in's reg.csv that a later reformat dropped; MAC
# is the ERT's own leak row. LOP is the right recipe for an EDGE accelerator --
# `itrs-hp` would be 59.37 nW/bit and is the wrong physics here.
#
# WHY THIS IS A KNOB AND NOT A CONSTANT: standby energy = power x TIME, so once
# reconstruction finishes sooner than embedded it pays proportionally less of
# it. With equal cycles more leakage SHRINKS the saving slightly; with
# reconstruction ahead on latency more leakage GROWS it. Sweep it.
declare -A ECC_LEAKAGE_NW=(
    [sram_bit]=2.693        # per stored bit   (CACTI 45nm itrs-lop; itrs-hp = 59.37)
    [rf_bit]=70.0           # per stored bit   (aladdin reg.csv, 40nm)
    [mac_instance]=7844.9 ) # per MAC          (ERT leak row, 0.00784449 pJ/cycle @1GHz)
: "${ECC_STATIC_ENERGY:=0}"          # 1 = charge the above to ALL THREE arms

# ---- clock rate, per architecture ------------------------------------------
# prompt_7 Issue 14. ECC_GLOBAL_CYCLE_SECONDS (section 5) is ONE number for every
# design and it was 1 GHz, while Eyeriss v1 silicon runs at 200 MHz. Pairing the
# real chip's ABSOLUTE off-chip MB/s with a 5x faster model clock makes the
# modelled chip ~5x more memory-starved than the real one (at 480 MB/s it would
# wait on DRAM ~89% of the run, against ~44% for silicon).
#
# UNITS ARE MHz. 1000 = 1 GHz, 200 = the JSSC 2017 rate. Section 10 flattens
# this table into ECC_ARCH_CLOCK_MHZ_LIST and config.Config.cycle_seconds_for()
# resolves it PER DESIGN into that design's own `global_cycle_seconds`, which is
# what globals.yaml carries into the mapper and what every per-cycle term is
# charged over. A design with no entry keeps ECC_GLOBAL_CYCLE_SECONDS (section
# 5), so this table adds designs rather than replacing the default.
#
# A MULTI-ARCHITECTURE RUN IS FINE: globals.yaml is written once PER DESIGN
# (ecc_energy_study/globals_<arch>.yaml), so two designs on one figure are each
# clocked at their own rate instead of sharing whichever was written last.
#
# WHAT THIS DOES NOT CHANGE: when a design is fully off-chip-bound its wall-clock
# time is set by bandwidth alone, so BOTH clocks give the same absolute run time
# and the SAME percentage saving. The saving stays at its ceiling for any clock
# above ~131 MHz at 480 MB/s. The clock buys defensibility, not a better number.
declare -A ECC_ARCH_CLOCK_MHZ=(
    [eyeriss_like_wglb]=200     # Eyeriss v1, JSSC 2017 core clock
    [eyeriss_like]=200          # the retired v1 variant, same silicon
    [eyeriss_v2_like]=1000      [eyeriss_v2_like_wglb]=1000   # no cited rate; model default
    [simba_like]=1000           [simple_weight_stationary]=1000
    [simple_input_stationary]=1000  [simple_output_stationary]=1000 )

# TRAP 1 -- THIS COLDS THE CACHE, ON PURPOSE. `global_cycle_seconds` is in the
# mapper fingerprint (config.py, `fingerprint()`), so changing a design's rate
# re-maps that design. Accepted: it is batched into prompt_7.md's Phase C with
# the other architecture changes, not paid separately.
#
# TRAP 2 -- EVERY PER-CYCLE CONSTANT MUST BE CONVERTED WITH THE SAME PERIOD, or
# standby energy silently moves by 5x. Two live cases, handled differently:
#
#   ECC_RECON_IDLE_PJ   is pJ PER CYCLE, measured by DC at a 1 ns clock
#                       (data/dc/BCH_N63_results.json, measurement.clock_period_ns
#                       = 1.0). At any other period it MUST be rescaled:
#                           idle_pJ_per_cycle(T) = idle_pJ_per_cycle(1ns) x T/1ns
#                       equivalently: use power_uW.idle.total x T. At 200 MHz the
#                       BCH(63,30) idle is 2.8310811 x 5 = 14.1554055 pJ/cycle.
#   ECC_LEAKAGE_NW      is POWER (nW), not energy, so it needs NO rescaling --
#                       energy per cycle = nW x T. That is why it is declared in
#                       nW and not in pJ/cycle.
#
# The wiring must apply the first rule in config.py where the DC tables are read,
# NOT at the point of use, so no caller can forget it.


# =============================================================================
#  5. HARDWARE / ARCHITECTURE MODEL
# =============================================================================

: "${ECC_WEIGHT_BITS:=8}"       # weight quantization: the protected payload
: "${ECC_ACTIVATION_BITS:=8}"   # input activations. Separate from the above on
                                # purpose -- `run.sh validate` checks that each
                                # architecture declares the right one at the
                                # right level.

# ACCUMULATOR PRECISION IS NOT STANDARDIZED, ON PURPOSE: Eyeriss v1 accumulates
# at 16b, v2 at 20b, Simba at 24b, each cited. Forcing one width would equalise
# the architectures rather than the experiment. Setting this is a SENSITIVITY
# STUDY only: it gets its own mapper cache and its own results namespace and is
# labelled as such everywhere. EMPTY = paper-native = the primary comparison.
: "${ECC_ACC_BITS:=}"

# Which YAML each design is mapped from.
#   paper : archs/<name>/arch_paper.yaml where it exists -- storage precisions
#           and scratchpad sizes as the design's paper publishes them, each
#           cited in a comment beside it.
#   stock : the designs exactly as timeloop-accelergy-exercises ships them.
: "${ECC_ARCH_FIDELITY:=paper}"

# Leave both EMPTY to model each design exactly as its YAML declares it. Set
# them to compare architectures rather than architectures+silicon.
#   ECC_FORCE_DATAWIDTH=8      equalise storage datawidth to the weight width.
#                              Evaluated PER ARCHITECTURE, so a design already
#                              at that width keeps its mapper cache. Dedicated
#                              partial-sum levels are never forced: accumulator
#                              precision is a separate design choice.
#   ECC_FORCE_TECHNOLOGY=45nm  equalise the Accelergy process node. This goes
#                              through globals.yaml, which costs DRAM for every
#                              design at once -- it invalidates EVERY cache.
: "${ECC_FORCE_DATAWIDTH:=}"
: "${ECC_FORCE_TECHNOLOGY:=}"

# ---- TASK 4: WEIGHT CAPACITY THE MAPPER SEES -------------------------------
# Multiplies the declared `depth:` of the WEIGHT-carrying storage levels in the
# architecture handed to Timeloop. This is the whole mechanism of Task 4.
#
# WHY IT IS A MAPPER KNOB. Under the reconstruction arm the on-chip weight
# representation is K/N of full width, so the same physical SRAM holds
# N/K = 1.6154x more weights at BCH(63,39) (eyeriss_like 75,264 -> 121,580
# weights; simple_weight_stationary 164,096 -> 265,078). A larger weight tile
# means the reconstruction arm RELOADS TILES FROM DRAM FEWER TIMES than the
# embedded reference, and a read never issued removes the DRAM ARRAY energy as
# well as the interface energy -- efficiency 1.0 per uJ, against the
# f_if x (1 - K/N) = 0.152 a fixed mapping buys. Task 3 cannot show it at all:
# RECON_OPTIMIZER=False pins ONE mapping on every arm, so both arms refetch
# identically by construction. The capacity therefore has to be in the
# architecture the SEARCH sees, which is what this rewrites.
#
#   1.0      the design as declared -- the embedded/conventional reference
#   1.6154   = N/K at BCH(63,39): the reconstruction arm's effective capacity
#   < 1.0    SHRINKS the design. NOT a curiosity: at the declared sizes weight
#            capacity is usually NOT the binding constraint (the mappings leave
#            most of the weight spad unused while still refetching), so the
#            dilation buys nothing until the buffer is small enough to bind.
#            Shrinking the reference and dilating from there is how the study
#            finds the regime where the two are MULTIPLICATIVE.
#
# Each value is a different architecture to the mapper and gets its OWN cache
# directory (slug `wcap<scale>`), which is the point: Task 4 is the DIFF of two
# mappings. A design where the scale rounds every weight level back to its
# declared depth is not dilated at all and keeps the undilated cache.
: "${ECC_WEIGHT_CAPACITY_SCALE:=1.0}"

# Which levels the scale above may rewrite.
#   exclusive  only a level whose `keep:` list is Weights and NOTHING else, so
#              the extra room can only be spent on weights. This is the
#              conservative bound and the default.
#   shared     also a level holding Weights beside another dataspace
#              (simple_output_stationary's and simple_input_stationary's
#              `operand_glb` keep Inputs and Weights). Timeloop has one
#              capacity per level, so dilating it hands the mapper free INPUT
#              capacity that reconstruction does not pay for -- the optimistic
#              bound. simple_weight_stationary LEFT THIS SET 2026-09-13: its
#              operand half is split into `input_glb` and a Weights-only
#              `weight_glb`, so both scopes give it the same answer.
# The two bracket one design and are quoted as a pair, the same rule CLAUDE.md
# sets for the eyeriss `_wglb` variants. A declared `depth: 1` register is
# never scaled under either: it is a pipeline latch, and turning it into a
# 2-entry buffer would invent a reuse level the design does not have.
: "${ECC_WEIGHT_CAPACITY_SCOPE:=exclusive}"

# TASK 4 LEVER 2 -- let the weight TILE grow into the room, not just the room.
# ECC_WEIGHT_CAPACITY_SCALE makes the buffer bigger; it does not make the mapper
# able to SPEND it, and measured (FINDINGS 7.8) that is the whole reason the
# capacity sweep reads zero. eyeriss_like's weights_spad declares
#     temporal: {factors: [N=1, M=1, P=1, Q=1, S=1]}
# and the M=1 pins the M tile AT that level to one, so the resident tile is
# M(8, from psum_spad below) x C(16) = 128 weights and stays 128 whatever the
# capacity is: `weights held` is EXACTLY 21,504 at x1, x1.6154, x4, x8, x16 and
# x32, the last of those at 0.9% fill. The binding constraint is the DATAFLOW
# CONSTRAINT, not the silicon, and no capacity sweep can discover that because
# the constraint never moves.
#   =1 drops the factors: pins on the WEIGHT-INDEXING dimensions (M, C, R, S)
#      of the temporal constraints on weight-carrying levels. N, P and Q keep
#      theirs -- weights do not index them, so relaxing those would retile the
#      activations and partial sums instead, which is a different experiment.
# THIS IS A DIFFERENT DATAFLOW. Eyeriss v1's M=1 at the filter spad IS the
# row-stationary dataflow, so a design run under this is not the chip JSSC 2017
# describes and must never be quoted as it -- `source: published` does not
# licence the name here. Its own cache (slug `wrelax`) and its own fingerprint;
# BOTH arms of a Task 4 pair are mapped under it, so the comparison stays fair
# even though neither arm is the published design. A design that pins nothing a
# weight tile is indexed by is not relaxed and keeps its existing cache --
# simple_weight_stationary is exactly that, so this lever cannot help it.
# It also WIDENS the mapspace, so at the same ECC_VICTORY the search has
# strictly more to explore: a relaxed run that comes back worse is evidence
# about the SEARCH, not about the dataflow.
# =1 IS THE DEFAULT since 2026-09-11: prompt_3's second lever, run together with
# ECC_MAPSPACE_CONSTRAIN=1 below (they compose; see there). Every prompt_5 and
# prompt_6 mapping already on disk was made with both exported, so the default
# changes no slug and no fingerprint -- the `mcons__wrelax` caches stay warm.
# 0 restores the published dataflow's pins.
: "${ECC_WEIGHT_FACTOR_RELAX:=1}"

# TASK 4 LEVER 3 -- MAKE THE SEARCH EXHAUSTIVE INSTEAD OF MAKING IT LONGER.
# Measured 2026-09-10 (FINDINGS 7.9): on Eyeriss v1 layer3.0.conv1 the
# convergence gate FAILS by 6-21x -- the embedded arm's total energy moves
# 43.7% between victory 4000 and 10000 where the ECC effect is 5.8%, and
# refetch is non-monotone in the budget (1.000 -> 4.000 -> 2.000). The mapper
# reports its own space as 7.41e10 index factorizations x 8.96e9 permutations;
# victory 50000 covers 0.07% of ONE thread's factorization subspace.
#
# A BIGGER BUDGET CANNOT FIX THAT. It samples more of the same enormous space,
# both arms still land on arbitrary points, and the DIFFERENCE between two
# arbitrary points is noise -- measured, the ORDERING between the arms flips.
# Constraining the loop nest collapses the space to ~9.5e4 factorizations,
# which is searched EXHAUSTIVELY: each arm then gets its TRUE optimum rather
# than a sample, the difference becomes architectural by construction, and the
# gate passes because nothing is left unsearched. It is also ~8x CHEAPER than
# the victory-5000 run whose answer is 9-11% wrong.
#
# WHICH LEVELS EACH DIMENSION MAY SPLIT ACROSS is archs.MAPSPACE_FREE_LEVELS,
# read off the BEST MAPPING THE SEARCH HAS EVER FOUND for this design (the
# victory-10000 embedded nest, 169.13 uJ against victory-4000's 300.34 on
# identical silicon). Constraining around a known-good region is a CHOICE and
# it is stated: the exhaustive answer is the best mapping IN THIS FAMILY, not
# in the whole space. What keeps it a fair ECC comparison is that BOTH ARMS
# get the identical constraint.
#
# RUN IT WITH ECC_WEIGHT_FACTOR_RELAX=1. The constraint makes the space
# searchable; the relax is what lets the weight TILE grow into the room a
# narrower word leaves. `weights_spad` keeps C and M free in the free-set
# precisely so the two compose instead of cancelling. Without the relax the
# tile stays pinned and capacity still cannot bind.
#
# THIS IS A DIFFERENT DATAFLOW. Own cache slug (`mcons`); a design run under
# it is NOT the chip JSSC 2017 describes and `source: published` does not
# licence its name -- the same rule ECC_WEIGHT_FACTOR_RELAX carries.
# A design with no MAPSPACE_FREE_LEVELS entry is NOT constrained and says so.
#
# =1 IS THE DEFAULT since 2026-09-11. This is prompt_3's constrained mapspace,
# the setting every prompt_5/prompt_6 mapper job runs under and the ONLY one
# whose convergence gate passes: 0.00% residual across victory 2000/4000/10000
# at ~1 min per map, against a 9-44% residual and hours per map unconstrained
# (FINDINGS 2.2). The four knobs prompt_3 exports together are now all
# defaults: this one, ECC_WEIGHT_FACTOR_RELAX=1, ECC_MAPPER_ALGORITHM=
# linear_pruned and ECC_MAPPER_TIMEOUT=100000000. Today only eyeriss_like_wglb
# has a free-set, so on any other design this is a no-op -- and that design is
# then searched UNCONSTRAINED with a systematic algorithm, which is the wrong
# regime (see ECC_MAPPER_ALGORITHM). 0 restores the unconstrained search.
: "${ECC_MAPSPACE_CONSTRAIN:=1}"

# ---- PROMPT_2: THE ON-CHIP QUANTISATION, AND THE DEPTH SWEEP ---------------
# These two replace ECC_WEIGHT_CAPACITY_SCALE as the Task 4 mechanism. The
# capacity knob above is kept -- it is still the right axis and its guards
# still apply -- but it expressed a narrower word as a DEEPER array, and
# Accelergy then priced the reconstruction arm's silicon 1.18-1.46x dearer per
# access, so the optimiser had a reason to leave the room unused. That is the
# defect that made the previous sweep prove nothing (FINDINGS 7.8).
#
# ECC_WEIGHT_DATAWIDTH rewrites `datawidth:` on the WEIGHT-carrying storage
# levels at FIXED `width:` and `depth:`. VERIFIED 2026-09-10 from
# timeloop-mapper.accelergy.log (`Calculated storage."width" as "width" = 16`):
# CACTI receives `depth` and `width` ONLY -- datawidth never reaches the energy
# model -- and Timeloop bills `vector_access_energy / block_size` per weight,
# `block_size = width/datawidth`. So halving it at fixed geometry EXACTLY
# halves per-weight energy and EXACTLY doubles effective capacity, with
# byte-identical per-access read/write/leak. That is the fairness condition
# depth-dilation could never meet.
#
#   EMPTY    the 8-bit arm: baseline AND embedded, which share one hardware
#            YAML and one mapping -- only the evaluator separates them. This
#            is the reference every margin is measured against.
#   4        the reconstruction arm at BCH(63,30): round(8*30/63) = 4 bits per
#            stored weight, 2.000x the weights per word on the SAME silicon.
#
# HARD CONSTRAINT: `width % datawidth == 0` on the level that declares it, or
# timeloop-mapper ABORTS -- `buffer.cpp:302`, measured at width 16 /
# datawidth 5 as `exit=134, core dumped`. block_size defaults to 1 and is then
# checked, so there is NO floor path and a partially-filled word cannot be
# modelled.
# YOU NEVER SATISFY THIS BY HAND AND CANNOT VIOLATE IT. THE WIDTH TABLE below
# gives every level a width that is a MULTIPLE OF THE DATAWIDTH THAT LEVEL
# STORES -- 96 at q=8, 98 at q=7, 96 at q=6, 95 at q=5, 96 at q=4 -- applied
# automatically on every run. The constraint is per level and per mapper run;
# it says nothing about what any OTHER arm declares, so 95 at q=5 is correct
# and the fact that 95 % 8 = 7 is irrelevant.
# DRAM IS NEVER REWRITTEN. recon.py owns the DRAM K/N scaling; narrowing DRAM
# here as well would double-count it there.
: "${ECC_WEIGHT_DATAWIDTH:=}"

# Restrict the datawidth rewrite to named weight levels (prompt_6 phase 2).
# EMPTY = every weight-carrying level, which is what every run before
# 2026-09-11 did, so their caches keep their fingerprints. An ERT arm sets
# it to the storage levels in its placement's `reduced` set, because the
# narrow weights stop AT the boundary: recon2 and recon4 both narrow
# `filter_glb` only and leave `weights_spad` at 8. Slug: `wdw4-filter_glb`.
# A name no weight level has is an ERROR, not a silent no-op. The launcher
# derives it per arm (ECC_RECON_ERT_ARM); set it by hand only for a diff.
#   : "${ECC_WEIGHT_DATAWIDTH_LEVELS:=filter_glb}"
: "${ECC_WEIGHT_DATAWIDTH_LEVELS:=}"

# PROMPT_2'S WIDTH TABLE -- AUTOMATIC, UNCONDITIONAL, AND NOT A KNOB.
#
# There is no ECC_WEIGHT_WIDTH. There was one until 2026-09-12 and it caused
# the exact failure it was supposed to prevent, so the width is now derived,
# on every run, from the datawidth each weight level stores:
#
#   arm                  q    spad W   GLB W (4x)   weights/word   eff. capacity
#   Baseline / Embedded  8      96        384            12           1.0000x
#   BCH(63,57)           7      98        392            14           1.1667x
#   BCH(63,45)           6      96        384            16           1.3333x
#   BCH(63,39)           5      95        380            19           1.5833x
#   BCH(63,30)           4      96        384            24           2.0000x
#
# BCH(63,51) shares q=6 with BCH(63,45) and BCH(63,36) shares q=5 with
# BCH(63,39), so they take the same widths. Read the live table with
#   python3 -m eccenergy.physics.widths
#
# THE ARMS DO NOT SHARE A DECLARED WIDTH. THIS IS THE POINT, AND IT HAS BEEN
# GOT WRONG REPEATEDLY. Each arm declares the width that suits ITS OWN
# datawidth, and no arm has to be legal for another arm's datawidth:
#
#   BCH(63,39) runs at width 95 because 95 % 5 == 0.
#   95 % 8 = 7 IS IRRELEVANT: the baseline/embedded arm is never mapped at
#   width 95. It is mapped at 96, where 96 % 8 == 0.
#
# Same for BCH(63,57) at width 98 (98 % 7 == 0; 98 % 8 = 2 is irrelevant).
# `timeloop-mapper` asserts `width % (word_bits * block_size) == 0`
# (buffer.cpp:302, block_size defaults to 1, NO floor path, exit=134 on a
# violation) PER DESIGN, per mapper run -- one arm at a time. It never sees
# two arms at once and there is no constraint between them.
#
# WITHDRAWN 2026-09-12: the lcm(q, 8) scheme (56 / 24 / 24 / 40 / 40), which
# came from reading the rule as "both arms share one width". It made the 8-BIT
# REFERENCE ARM MOVE BETWEEN CODES, so the reference held 35 / 33 / 30 weights
# per PE at q = 7 / 6 / 5, fell off a tiling cliff at q=5, and reported
# BCH(63,39) as a 37.69% win that was really the reference breaking
# (FINDINGS 2.4b). Under THIS table the 8-bit arm is width 96 at every code:
# ONE arm, mapped ONCE, and that artifact cannot occur.
#
# PER LEVEL, not per design. An ERT arm narrows only the storage levels in its
# placement's reduced set, so at ECC_WEIGHT_DATAWIDTH_LEVELS=filter_glb the
# GLB stores 5-bit weights at width 380 while weights_spad keeps 8-bit weights
# at width 96. Both are legal for what they hold; neither is legal for the
# other, and neither has to be.
#
# EACH LEVEL'S DEPTH IS RENORMALISED AT THE BASE WIDTH (96, x4 above the PE
# array), NOT at the arm's own width -- depth' = round(depth x width / 96) --
# so it holds the published TOTAL BITS and is THE SAME FOR EVERY ARM. CACTI is
# handed depth and width, so that is the quantity that must not move; and a
# shared depth is what makes `eff. capacity` come out as (W_arm/q)/(96/8),
# reproducing prompt_2's column to the digit. It reproduces prompt_2's own
# depths too: weights_spad 224 x 16 b = 3,584 b -> depth 37 at width 96;
# filter_glb 1024 x 64 b = 65,536 b -> depth 171 at width 384.
#
# DRAM IS NEVER REWRITTEN. recon.py owns the DRAM K/N scaling; narrowing DRAM
# here as well would double-count it there.
#
# The GLB multiplier below is the one number left to set: 4x is Eyeriss v1's
# published 16-b spad / 64-b GLB ratio, and q dividing W implies q dividing
# 4W, so one scratchpad width settles every weight level at once.
: "${ECC_WEIGHT_WIDTH_GLB_MULT:=4}"

# LET THE TWO ARMS DECLARE DIFFERENT `depth:` ON A WEIGHT LEVEL.
# archs.assert_pair_geometry() checks that the reconstruction arm and the
# embedded arm hold the SAME LEVELS AT THE SAME DEPTH -- and nothing else.
# It does NOT check `width:` and it does NOT check `datawidth:`, because under
# THE WIDTH TABLE above the arms are SUPPOSED to differ there (96 vs 95 vs 98,
# 8 vs 5 vs 7). Asserting a shared width is what produced the withdrawn
# lcm(q, 8) scheme; that check is gone and must not come back.
#
# DEPTH is the one thing left to assert, and it is worth asserting: a depth
# difference is real silicon one arm does not have, priced by Accelergy, which
# is the defect that invalidated the pre-prompt_2 sweep (capacity dilation,
# FINDINGS 7.8 -- the dilated array cost 1.18-1.46x more per access, so the
# optimiser had a reason to leave the room unused).
#
#   0   assert it. THE DEFAULT: a depth difference nobody asked for is a void
#       comparison and the run stops rather than reporting it.
#   1   allow it, and print what differs instead of raising. For a study that
#       varies depth between the reconstruction and the embedded arm ON
#       PURPOSE -- several different depth variations per arm, which is what
#       this knob exists for.
: "${ECC_DISABLE_ASSERT_PAIR_GEOMETRY:=0}"

# THE ONLY SWEPT VARIABLE: `depth:` of the on-chip weight levels. Explicitly
# NOT swept: width, datawidth, DRAM (ECC_DRAM_DEPTH), any level not holding
# Weights, bandwidths, n_banks, technology, PE counts, dataflow constraints.
#
# Its own cache slug (`wdepth<scale>`) and NOT `wcap`, because the two mean
# different things even though they rewrite the same field:
# ECC_WEIGHT_CAPACITY_SCALE triggers recon.capacity_dilation_correction(),
# which re-prices the level at the UNDILATED geometry. That correction is
# WRONG here -- a shallower array really IS a smaller array, and its cheaper
# access is a real saving, not an artifact to undo. Sharing a directory would
# let a corrected run be read as an uncorrected one.
: "${ECC_WEIGHT_DEPTH_SCALE:=1.0}"

# Restrict the depth scale to named levels. EMPTY = every weight-carrying
# level, which is the default AND a limitation: one scale then moves
# `weights_spad` and `filter_glb` TOGETHER, so it locates the zone but cannot
# say which level bought it -- and the YAML needs a depth per level. The
# SECOND PASS holds one at x1 and sweeps the other, which is what naming
# levels is for. FINDINGS 7.8 predicts `filter_glb` is the one that matters
# (refetch on v1 is set by the DRAM-level loop order over P/Q, a weight tile
# cannot index either, so a buffer INSIDE the array can never absorb them --
# measured flat to x32 at 1% fill); CONFIRM it, do not assume it.
# A name no weight level has is an ERROR, not a silent no-op.
#   : "${ECC_WEIGHT_DEPTH_LEVELS:=filter_glb}"
: "${ECC_WEIGHT_DEPTH_LEVELS:=}"

# The depth ladder hpc/map_depth_sweep.sh submits. sqrt(2) steps, NOT factor 2:
# the window where Embedded cannot hold the tile and Recon can is exactly as
# wide, in depth, as the effective-capacity ratio, so a factor-2 grid steps
# clean over a 1.17x or 1.33x window and reports a grid artifact as "no
# effect". sqrt(2) resolves BCH(63,30) (2.00x) and BCH(63,39) (1.58x); the
# x0.5 / x0.25 / x0.125 points a YAML will quote are QUOTED OUT OF IT.
# BCH(63,45) (1.33x) and BCH(63,57) (1.17x) would need ~x1.12 steps, about 20
# depths -- do them only after the strong codes show something.
: "${ECC_DEPTH_SWEEP_SCALES:=1.0 0.7071 0.5 0.3536 0.25 0.1768 0.125}"

# The reconstruction arm's on-chip datawidth for the sweep. EMPTY derives it as
# ceil(ECC_WEIGHT_BITS * K/N), which is 4 at BCH(63,30).
: "${ECC_DEPTH_SWEEP_RECON_DW:=}"

# CONVERGENCE IS A GATE, and it runs BEFORE any number is quoted. Map the
# EMBEDDED arm at each of these victory budgets; converged means the last two
# agree within a margin you STATE, and that margin must be SMALLER than the
# Recon-vs-Embedded effect being claimed. Re-checked at the SMALLEST depth as
# well as the largest, because a budget that converges on a big buffer may not
# on a small one. ECC_VICTORY (section 2) is the budget the sweep itself runs
# at and must be one of these.
# Measured cost, 2026-09-10 (jobs 41582127-30, one layer, 18 threads):
# victory 10000 ran 1h13m-1h34m wall; victory 50000 ran 6h02m-8h09m. A 4-point
# gate at 50000 is a full day per arm.
: "${ECC_DEPTH_SWEEP_GATE_VICTORIES:=2000 4000 10000}"

# Which depths the gate is re-checked at: the largest and the smallest of the
# ladder. A budget that converges on a big buffer may not on a small one.
: "${ECC_DEPTH_SWEEP_GATE_SCALES:=1.0 0.125}"

# ---- the MAC cost, i.e. the DENOMINATOR of every ECC percentage -------------
# An ECC saving is saved_uJ / total_uJ. The saved uJ are weight traffic and do
# not depend on what a MAC costs; the total does, and on eyeriss_v2_like the
# MAC is 44% of the run at 1.16877 pJ per 8-bit MAC. That number is Accelergy's
# `intmac` compound (an aladdin_multiplier 8x8 plus an aladdin_adder 20b) from
# the Library plug-in's ONE table row each -- 32-bit, 40 nm -- scaled linearly
# in each operand width and up from 40 to 45 nm (FINDINGS 7.3 quotes file, row
# and formula). Horowitz (ISSCC 2014, Fig. 1.1.9, 45 nm) puts an int8 multiply
# at ~0.2 pJ and an int8 add at ~0.03 pJ, 5x less; Eyeriss v1 measured its ALUs
# at <10% of chip power. If the MAC is 4-6x too expensive, EVERY percentage in
# Tasks 1-3 is diluted 1.5-2x, embedded as much as recon.
#   EMPTY   the ERT's value (1.16877 pJ); now the sensitivity row.
#   <pJ>    rescale the Compute category to MACs x this value, in the EVALUATOR
#           only. The MAC count is mapping-invariant, so under ECC_OPT_METRIC=
#           energy the mapping optimum does not move and the cache stays warm;
#           under edp it can, and the run prints a warning. Every figure, table,
#           manifest and result file carries the value and its citation
#           (archs/_shared/provenance.yaml `mac_energy_pj` -- a value listed
#           there is labelled with its source, any other value "uncited").
# THE DEFAULT IS 0.23 = Horowitz ISSCC 2014 int8 multiply + int8 add at 45 nm,
# adopted as THE PRIMARY DENOMINATOR on 2026-09-09 by decision (FINDINGS 7.3):
# the ERT's number rests on one 40 nm HLS table row per primitive and is 5x the
# cited figure. Set it EMPTY to reproduce the ERT-denominator numbers.
: "${ECC_MAC_PJ_OVERRIDE:=0.23}"

# DRAM geometry and the global clock, shared by every design (globals.yaml).
# Both invalidate every architecture's mapper cache.
: "${ECC_DRAM_DEPTH:=1048576}"
: "${ECC_GLOBAL_CYCLE_SECONDS:=1e-9}"

# ---- interconnect ----------------------------------------------------------
# Timeloop's own wire model is a stub that returns 0, so ECC_NOC=0 makes every
# network free -- in the evaluator AND in the mapper's objective. The
# coefficients and their citations live in archs/_shared/noc.yaml. All four are
# in the cache slug (`noc`), so a pre-NoC mapping is never read back as a costed
# one. NoC is its own plotted category.
: "${ECC_NOC:=1}"
: "${ECC_NOC_WIRE_PJ_PER_BIT_MM:=}"   # override the shared 45nm wire constant (0.12)
: "${ECC_NOC_ROUTER_PJ:=}"            # override the shared per-flit router energy (0.25)
: "${ECC_NOC_PE_LATCH_PJ:=}"          # override the per-PE latch on v2's PE row (0.5; bracket 0 / 0.9152)
: "${ECC_NOC_SCALE:=1}"               # multiply every NoC term, for sensitivity runs


# =============================================================================
#  6. ECC ACCOUNTING  --  how each arm is charged
# =============================================================================

# Where codeword boundaries fall when the baseline's external parity is counted.
# NOT a flat N/K: a weight cannot straddle a codeword, so at BCH(63,51) over
# 8-bit weights only 6 whole weights fit in the 51-bit message field and 3 bits
# are padding -- 31.25% overhead, against the 23.53% a flat N/K model charges.
#   layer : each layer's weight tensor is its own codeword stream, so each pays
#           its own tail padding. What a real allocator does, and conservative.
#   model : one stream over the whole model. Differs by at most one codeword per
#           layer -- nothing on a full model, visible on a single layer.
: "${ECC_PARITY_GROUPING:=layer}"

# Do the message-padding bits cross the DRAM bus with the parity? They sit
# inside the stored codeword, so yes by default. 0 charges parity only (25.00%),
# which models a layout that packs weights across codeword boundaries and
# re-splits them at the ECC engine -- a DIFFERENT memory layout, not a cheaper
# version of this one.
: "${ECC_PARITY_CHARGE_PADDING:=1}"

# Weights per codeword used to COUNT embedded-arm decodes. EMPTY means "the same
# codeword as the baseline", the shared-geometry model. 8 reproduces the older
# two-arm scripts, which charged the embedded arm per byte.
: "${ECC_EMB_WEIGHTS_PER_CW:=}"

# ---- decoder energy --------------------------------------------------------
# 0 -> decode is charged as ZERO for every arm; the category sums to zero and is
#      dropped from the bars and the legend. THIS IS THE PUBLISHED SETTING,
#      because the two numbers below are ESTIMATES, not measurements.
: "${ECC_DECODE:=0}"
: "${ECC_DECODE_PJ_BASE:=40.0}"      # pJ per codeword, BCH(63,51) syndrome+Chien
: "${ECC_DECODE_PJ_EMB:=40.0}"       # the same code -> the same decoder
: "${ECC_RECON_CHARGES_DECODE:=1}"   # does recon still detect before rebuilding?

# ---- reconstruction datapath (Design Compiler) -----------------------------
# FreePDK45/OSU gscl45nm, 1.1 V, 1 ns clock, pre-layout with a wire-load model.
# The synthesis run is matched on (N,K), so a K sweep and a fixed-K run read the
# same table and neither can be costed at the wrong code's datapath.
: "${ECC_RECON_JSON:=data/dc/BCH_N63_results.json}"
# prompt_6 RULE 3: the datapath has TWO terms on TWO denominators --
# incremental per codeword EVENT, idle per CYCLE per ENGINE:
#     E_recon = incremental x events + idle_per_cycle x cycles x N_engines
# ECC_RECON_INCLUDE_IDLE, which ADDED a per-codeword number to a per-cycle
# one, is RETIRED: idle is always charged, on its own denominator, with the
# cycle count of the plan that is being billed (build_stacks: the sweep's
# recon arm reconstructs at the chip ingress, one engine; the placement
# study: 1 / 14 / 168 engines derived from the site stage). The two tables
# below are the DC numbers, keyed by configuration, and they are read FIRST --
# before ECC_RECON_JSON, which is the synthesis archive behind them. Edit a
# value here and it is what the run prices the datapath at. BOTH tables must
# carry the (N,K) for them to win; if either lacks it the pair falls through to
# the JSON, so the two terms can never come from different sources. Changing
# either one moves the ERT toll, which is hashed into the mapper fingerprint,
# so the affected arms re-map by themselves on the next run. Bash cannot export
# an associative array, so section 10 flattens them into
# ECC_RECON_INCREMENTAL_PJ_LIST / ECC_RECON_IDLE_PJ_LIST.

# declare -A ECC_RECON_INCREMENTAL_PJ=(   # pJ per codeword
#     [BCH_63_57_t1]=1.6574   [BCH_63_51_t2]=1.8995   [BCH_63_45_t3]=1.6383
#     [BCH_63_39_t4]=1.4561   [BCH_63_36_t5]=1.5082   [BCH_63_30_t6]=1.3786 )
# declare -A ECC_RECON_IDLE_PJ=(          # pJ per CYCLE per ENGINE
#     [BCH_63_57_t1]=1.9359672  [BCH_63_51_t2]=2.2301273  [BCH_63_45_t3]=2.4120856
#     [BCH_63_39_t4]=2.7891299  [BCH_63_36_t5]=2.8358254  [BCH_63_30_t6]=2.8310811 )

declare -A ECC_RECON_INCREMENTAL_PJ=(   # pJ per codeword
    [BCH_63_57_t1]=1.6574   [BCH_63_51_t2]=1.8995   [BCH_63_45_t3]=1.6383
    [BCH_63_39_t4]=1.4561   [BCH_63_36_t5]=1.5082   [BCH_63_30_t6]=1.3786 )
declare -A ECC_RECON_IDLE_PJ=(          # pJ per CYCLE per ENGINE
    [BCH_63_57_t1]=1.9359672  [BCH_63_51_t2]=2.2301273  [BCH_63_45_t3]=2.4120856
    [BCH_63_39_t4]=2.7891299  [BCH_63_36_t5]=2.8358254  [BCH_63_30_t6]=2.8310811 )

: "${ECC_RECON_PJ:=}"                # overrides the INCREMENTAL term only (pJ per
                                     # codeword); idle has no override
# Used ONLY when the JSON has no entry for the (N,K) in play -- the BCH(63,51)
# numbers, so a missing entry degrades to a plausible cost instead of crashing.
# If you see these in a result's provenance, the table is missing a code.
: "${ECC_RECON_INCREMENTAL_FALLBACK_PJ:=1.8995}"
: "${ECC_RECON_IDLE_FALLBACK_PJ:=2.2301273}"

# ---- clock gating of the reconstruction engines ---------------------------
# THE IDLE NUMBER ABOVE IS A FREE-RUNNING CLOCK. `ECC_RECON_IDLE_PJ` is the DC
# idle-window TOTAL, and for BCH(63,30) that is 2.8310811 pJ/cycle of which
# 2.8164 pJ (99.48%) is CLOCK/dynamic power and only 0.0147 pJ (0.52%) is true
# leakage (data/dc/BCH_N63_results.json, power_uW.idle.{dynamic,leakage}).
# An engine that is clock-gated when no weight is arriving does not burn the
# dynamic part.  This knob is the percentage of the idle-cycle energy that
# gating removes:
#
#   E_recon = (incremental + idle) x events                      <- engine working
#           + idle x (1 - PCT/100) x (engine_cycles - events)     <- engine gated off
#
#   PCT=0     reproduces the pre-gating model EXACTLY, to the pJ.  Use it to
#             diff against any number published before this knob existed.
#   PCT=99.5  the default: the measured clock/dynamic share.
#   PCT=100   fully power-gated -- the optimistic bound, no standby at all.
#
# Measured duty cycles are 0.62% (recon4) to 14.3% (recon1), so this term is
# 93.5-99.7% of the reconstruction energy and the knob moves the headline
# result.  ALWAYS report PCT=0 beside whatever you choose.  See prompt_7.md
# section 11 Q1 for the verification this setting still needs.
: "${ECC_RECON_CLOCK_GATING_PCT:=99.5}"

# ---- energy-model revision (what invalidates the cache) ---------------------
# THE MAPPER FINGERPRINT HASHES THE ARCHITECTURE, NOT THE PRICE LIST. Accelergy
# derives the price list FROM the architecture, so fixing an estimator changes
# every energy in the cache while leaving the directory it is stored under
# identical -- the stale entries are then reused and nothing says so.
#
# That is not hypothetical: the Neurosim plug-in answered 0 pJ for every
# smartbuffer address generator until 2026-09-12 (it crashed writing scratch
# into a read-only SIF and Accelergy accepted the 0). hpc/tl.sh now gives it a
# writable directory, but THAT FIX CANNOT LAND UNTIL THE CACHE IS REGENERATED.
#
# Set this to any non-empty string and the whole matrix colds deliberately;
# EMPTY hashes byte-identically to every fingerprint made before the knob
# existed. Form: a date plus what changed.
#
# SET 2026-09-13 (prompt_7 C1.8). It reaches BOTH fingerprints -- Config's, and
# eccenergy/archs.arch_fingerprint(), which is the one the mapper CACHE
# DIRECTORY is named after. Until Phase C it reached only the first, so the knob
# re-labelled results while the cache it was supposed to invalidate stayed warm;
# that is fixed here, and EMPTY still hashes byte-identically to every
# pre-Phase-C directory.
#
# WHAT THIS REVISION CARRIES: the Neurosim address-generator price (0 pJ before
# 2026-09-12, see above).
: "${ECC_ENERGY_MODEL_REV:=2026-09-12-neurosim-adders}"

# ---- weak ECC overlay ------------------------------------------------------
# A light SRAM-side code on top of the strong one. Weights pay it under every
# arm; inputs pay it under the baseline only.
: "${ECC_WEAK:=0}"
: "${ECC_WEAK_N:=63}"
: "${ECC_WEAK_K:=57}"

# ---- what the baseline inflates -------------------------------------------
# 0 IS THE CONVENTIONAL BASELINE, and what Task 1 specifies: external parity is
# consumed by the off-chip ECC correction and is never written into on-chip
# weight SRAM/RF. 1 reproduces the older figures and is warned about in the
# result JSON.
: "${ECC_BASELINE_INFLATES_ONCHIP:=0}"

# Split the two on-chip categories into read and write.
: "${ECC_SPLIT_READ_WRITE:=0}"

# How a storage level is assigned to "Global buffer" vs "On-chip SRAM/RF":
#   instances : one instance = a shared global buffer, replicated = local.
#               Correct for simba_like, whose per-PE buffers are named
#               "...Buffer" and which name matching therefore mislabels.
#   name      : legacy substring matching; reproduces the old figures exactly.
# The totals are identical either way; only the split between the two moves.
: "${ECC_CLASSIFY:=instances}"


# =============================================================================
#  7. THE CLUSTER  --  SLURM and the container
# =============================================================================
#  hpc/run_all.sh passes these to sbatch ON THE COMMAND LINE, which overrides
#  the #SBATCH header inside hpc/map.sbatch. That header only matters for a bare
#  `sbatch hpc/map.sbatch`.

: "${ECC_ACCOUNT:=rewetz}"
: "${ECC_QOS:=rewetz-b}"          # rewetz-b is the burst QOS: idle cores, low
                                  # priority, a 4-day limit. prompt_2 asks for
                                  # it -- ~90 concurrent jobs against the
                                  # investment QOS's handful, and the depth
                                  # sweep is 18 independent maps that would
                                  # otherwise queue in waves of 9. Set
                                  # ECC_QOS=rewetz for a run that must not be
                                  # preempted.
: "${ECC_PARTITION:=hpg-default}"

# ---- the mapping array: one task per (architecture, model) pair ------------
# Keep the cores equal to ECC_MAPPER_THREADS. The thread count is in the cache
# key, so a mismatch maps at one key and evaluates at another.
: "${ECC_MAP_CPUS:=${ECC_MAPPER_THREADS}}"
: "${ECC_MAP_MEM:=16gb}"
# A shorter wall request is scheduled sooner. The longest measured BOUNDED task
# was 31 min; an uncapped search is typically 4-10x that, and twice again on the
# depth-scaled 9/10-level designs.
: "${ECC_MAP_TIME:=24:00:00}"
# How many array tasks run at once. The rewetz investment is 181 cores, so at 18
# cores per task 10 fit; 9 leaves room for a VS Code ondemand session. Anything
# above the limit just queues as `JobArrayTaskLimit` -- not an error.
: "${ECC_CONCURRENCY:=9}"

# ---- the evaluation job: never invokes Timeloop, so it is small ------------
: "${ECC_EVAL_CPUS:=2}"
: "${ECC_EVAL_MEM:=8gb}"
: "${ECC_EVAL_TIME:=02:00:00}"

# The Timeloop+Accelergy image, and where the generated task list is written.
# The task list is GENERATED from ECC_ARCHS x ECC_MODELS -- never hand-edited.
: "${ECC_SIF:=${ECC_PROJECT_ROOT}/timeloop.sif}"
: "${ECC_TASKFILE:=${ECC_PROJECT_ROOT}/hpc/.runtime/tasks.txt}"

# WHICH COPY OF `archs/` THIS RUN IS. Set by hpc/map_ert_arms.sh in --export,
# not by hand: the launcher copies `archs/` into hpc/.runtime/archpin.<pid>/ at
# SUBMISSION and points every map job and the dependent eval at that copy, so
# ONE SUBMISSION MAPS ONE ARCHITECTURE and editing `archs/` while the array is
# in flight cannot reach it. Try a new depth the second after you hit submit;
# the queued jobs keep mapping the chip you submitted, and the NEXT run
# snapshots the new file.
#
# Empty (an interactive run) = read `archs/` live, which is what it always did.
# Point it at any directory shaped like `archs/` to map that instead -- a
# snapshot kept from an earlier run, for instance. A path that does not exist
# warns on stderr and falls back to `archs/`; it never stops the run.
# The fingerprint is computed from whichever bytes this names, so the cache
# directory still moves when the architecture does -- see eccenergy/paths.py.
: "${ECC_ARCH_PIN_DIR:=}"


# =============================================================================
#  8. OUTPUT AND FIGURES
# =============================================================================

: "${ECC_RESULTS_DIR:=results}"
: "${ECC_PALETTE:=house}"        # house | cvd (colourblind-safe Okabe-Ito)
: "${ECC_FORMATS:=png pdf}"
: "${ECC_DPI:=400}"
: "${ECC_NICE_LABELS:=1}"        # 1 -> "Eyeriss v2"; 0 -> "eyeriss_v2_like"
: "${ECC_TITLE_NOTE:=}"          # free text appended to the figure title

# ECC_STEM -- the figure/table/manifest name -- is set in section 10 from
# ECC_SWEEP: one fixed name per axis (ArchitectureSweep / ModelSweep /
# BCHsweep), so a re-run at different constants REWRITES the file instead of
# growing the directory, and the manifest beside it records which constants
# produced what is on disk. Export ECC_STEM="" to get the self-describing name
# (ArchitectureSweep__panels__resnet18__mobilenet_v2) when you want to keep two
# panel figures side by side.


# =============================================================================
#  9. MISCELLANEOUS
# =============================================================================

# Results are never silently overwritten. 1 replaces an existing file.
: "${ECC_OVERWRITE:=0}"

# 1 -> a cached mapping is reused ONLY if its mapping.json sidecar proves it was
#      computed for the CURRENT architecture YAML and mapper settings.
# 0 -> also accept pre-Task-1 entries, which have no sidecar and can prove
#      nothing. Those results are labelled `legacy` and warned about: use it to
#      reuse old compute, never to publish.
: "${ECC_CACHE_STRICT:=1}"

# Free text recorded in every result JSON. Say what the run was for.
: "${ECC_RUN_NOTE:=}"

# 1 -> rebuild the figure from results/_raw/ alone: no mapper, no container.
#      Milliseconds, and what to use after changing anything in section 6.
: "${ECC_REPLOT_ONLY:=0}"

# 1 -> use only mappings ALREADY solved and never invoke Timeloop. This is what
#      `--eval` sets and what the evaluation stage runs under.
: "${ECC_FROM_CACHE:=0}"

: "${ECC_PYTHON:=python3}"

# Extra detail on stdout while workloads are generated.
: "${ECC_VERBOSE:=}"

# ---- workload generation ---------------------------------------------------
# Read ONLY by `python3 -m eccenergy.arch.generate models <name>`, which turns a
# network into the layer list the mapper walks. Regenerating a workload changes
# every shape name and therefore every cache entry, so these are not run-time
# knobs -- set them for the generation, then leave them alone.
: "${ECC_INPUT_HW:=224}"             # CNN input resolution
: "${ECC_SEQ:=1}"                    # transformer sequence length
: "${ECC_INCLUDE_LM_HEAD:=1}"        # count the LM head matmul
: "${ECC_INCLUDE_EMBEDDING:=0}"      # count the token-embedding table (a lookup,
                                     # not a matmul -- off on purpose)


# =============================================================================
#  10. DERIVED  --  NOT KNOBS. Nothing below here needs editing.
# =============================================================================
#  eccenergy/config.py speaks in terms of one SWEPT list plus two HELD
#  constants. The lists in section 3 are the editable form of exactly that, and
#  this block translates. Everything is still assigned with `:=`, so an explicit
#  ECC_SWEEP_ARCHS=... in the environment continues to win.

_ecc_first() { set -- ${1:-}; echo "${1:-}"; }
_ecc_count() { set -- ${1:-}; echo "$#"; }

# SECTION 4 TAKES PRECEDENCE OVER SECTION 3. The reconstruction placement study
# holds the architecture, the model and the code fixed and puts the BOUNDARY on
# the x axis, so section 3's lists are collapsed onto section 4's single point.
# This is a plain assignment, not `:=`: with ECC_RECON_MODELING=1 the point is
# what ECC_RECON_* says, and an ECC_ARCHS left over in the shell must not
# silently widen a study that only makes sense on one design. Change the point
# with ECC_RECON_ARCH / ECC_RECON_MODEL / ECC_RECON_K.
: "${ECC_RECON_OPTIMIZER:=${RECON_OPTIMIZER}}"
if [ "${ECC_RECON_MODELING}" = "1" ]; then
    ECC_ARCHS="${ECC_RECON_ARCHS}"
    ECC_MODELS="${ECC_RECON_MODEL}"
    ECC_CODE_N="${ECC_RECON_CODE_N}"
    ECC_KS="${ECC_RECON_K}"
    # prompt_6 8.2: the placement study's layer scope is ECC_RECON_LAYER --
    # `all`/`full` (or an empty default above) means every layer of the model.
    case "${ECC_RECON_LAYER}" in
        all|ALL|full|FULL|"") ECC_LAYERS="" ;;
        *)                    ECC_LAYERS="${ECC_RECON_LAYER}" ;;
    esac
    ECC_EXPERIMENT="recon"
    ECC_PANEL_MODELS=""
    # Which boundaries to draw, per architecture. A bash associative array
    # cannot be exported, so the entry for EVERY architecture on the figure is
    # flattened into one scalar of `;`-separated `arch=key key key` entries,
    # which `config.recon_placements_for()` reads. An architecture with no entry
    # contributes none, and no entry means "every placement eccenergy/recon.py
    # defines for that design" -- which is the normal thing to want.
    ECC_RECON_PLACEMENT_LIST=""
    for _a in ${ECC_RECON_ARCHS}; do
        _keys="${ECC_RECON_PLACEMENTS[${_a}]:-}"
        [ -n "${_keys}" ] || continue
        ECC_RECON_PLACEMENT_LIST="${ECC_RECON_PLACEMENT_LIST}${_a}=${_keys};"
    done
    unset _a _keys
    # The placement study owns its own output name (section 4), and a
    # selected-layer run must keep its layer scope in it, as everywhere else.
    # TASK 4 OWNS ITS OWN NAME. RECON_OPTIMIZER=True re-optimises the mapping
    # for the reduced weight width, so its bars are not comparable with a
    # fixed-mapping ReconSweep.png and must never overwrite it. Same rule as
    # the three sweeps: the stem comes from the configuration alone.
    _ecc_stem="${ECC_RECON_STEM}"
    _ecc_opt=0
    case "${ECC_RECON_OPTIMIZER}" in
        [Tt]rue|1|[Yy]es) _ecc_stem="${ECC_RECON_STEM}_optimiser"; _ecc_opt=1 ;;
    esac
    if [ -z "${ECC_LAYERS}" ] || [ "${_ecc_opt}" = "1" ]; then
        # prompt_6 9: the optimiser figure has ONE path PER MODEL,
        # results/figures/ReconSweep_optimiser__<model>.png, EVEN on a
        # layer-scoped run (the study is one layer by design); the layer scope
        # lands in the manifest and the figure title instead of the filename,
        # and an existing file at that path is overwritten on purpose. The
        # model suffix is there since 2026-09-11, when the study ran on two
        # networks: the stem comes from the configuration, and the model IS
        # configuration -- without it the second model's eval overwrote the
        # first's figure. The fixed-mapping ReconSweep keeps the layer suffix
        # on a scoped run and takes the same model suffix on a whole-model one.
        ECC_STEM="${_ecc_stem}__${ECC_RECON_MODEL}"
    else
        ECC_STEM=""
    fi
    unset _ecc_stem _ecc_opt
    # `recon` is the only evaluation this study writes: it holds Task 1's and
    # Task 2's bars itself, from Task 1's and Task 2's own functions.
    ECC_EVAL_EXPERIMENTS="recon"
fi
: "${ECC_RECON_PLACEMENT_LIST:=}"

# prompt_6 RULE 3: the two DC tables of section 6, flattened into `key=pJ;`
# scalars because bash cannot export a `declare -A` (config._table reads them).
ECC_RECON_INCREMENTAL_PJ_LIST=""
for _k in "${!ECC_RECON_INCREMENTAL_PJ[@]}"; do
    ECC_RECON_INCREMENTAL_PJ_LIST="${ECC_RECON_INCREMENTAL_PJ_LIST}${_k}=${ECC_RECON_INCREMENTAL_PJ[${_k}]};"
done
ECC_RECON_IDLE_PJ_LIST=""
for _k in "${!ECC_RECON_IDLE_PJ[@]}"; do
    ECC_RECON_IDLE_PJ_LIST="${ECC_RECON_IDLE_PJ_LIST}${_k}=${ECC_RECON_IDLE_PJ[${_k}]};"
done

# prompt_7 Phase A: section 6's leakage densities, flattened the same way. They
# are POWER in nW, so nothing here rescales them for the clock period -- the
# evaluator multiplies by the cycle period at the point of use (section 6's
# TRAP 2). `config._table` reads this into `Config.leakage_nw`.
ECC_LEAKAGE_NW_LIST=""
for _k in "${!ECC_LEAKAGE_NW[@]}"; do
    ECC_LEAKAGE_NW_LIST="${ECC_LEAKAGE_NW_LIST}${_k}=${ECC_LEAKAGE_NW[${_k}]};"
done

# prompt_7 Phase C1.5: section 6's per-design clock, flattened the same way.
# `config.Config.cycle_seconds_for(arch)` reads it; a design with no entry falls
# back to ECC_GLOBAL_CYCLE_SECONDS. UNITS ARE MHz -- the inversion to seconds
# happens once, in config.py, so no caller can do it twice or not at all.
ECC_ARCH_CLOCK_MHZ_LIST=""
for _k in "${!ECC_ARCH_CLOCK_MHZ[@]}"; do
    ECC_ARCH_CLOCK_MHZ_LIST="${ECC_ARCH_CLOCK_MHZ_LIST}${_k}=${ECC_ARCH_CLOCK_MHZ[${_k}]};"
done

: "${ECC_SWEEP_ARCHS:=${ECC_ARCHS}}"
: "${ECC_SWEEP_MODELS:=${ECC_MODELS}}"
: "${ECC_SWEEP_KS:=${ECC_KS}}"
: "${ECC_CONST_ARCH:=$(_ecc_first "${ECC_ARCHS}")}"
: "${ECC_CONST_MODEL:=$(_ecc_first "${ECC_MODELS}")}"
: "${ECC_CONST_K:=$(_ecc_first "${ECC_KS}")}"

# ...BUT THE SIX ABOVE ARE `:=`, AND SECTION 4's POINT IS A BARE `=`.
# Section 4 hard-assigns ECC_ARCHS/ECC_MODELS/ECC_CODE_N/ECC_KS precisely so a
# value left over in the shell cannot silently widen a one-design study. The
# names DERIVED from them here were still `:=`, so a leftover survived in the
# derived name instead and the two disagreed without saying so:
#
#     source ./env.sh                                  # ECC_CONST_MODEL=resnet18
#     ECC_RECON_MODEL=convnext_tiny bash hpc/tl.sh ...  # ECC_MODELS=convnext_tiny
#                                                      # ECC_CONST_MODEL=resnet18 (!)
#
# and `config.load_config()` reads the CONST name on a held axis, so that maps
# resnet18 while every banner says convnext_tiny. Found 2026-09-13 building
# hpc/smoke_models.sh, which sources this file and then re-invokes per model.
# Outside the placement study nothing changes: these stay `:=`, and inside it
# the values are identical on a shell that sourced this file once.
if [ "${ECC_RECON_MODELING}" = "1" ]; then
    ECC_SWEEP_ARCHS="${ECC_ARCHS}"
    ECC_SWEEP_MODELS="${ECC_MODELS}"
    ECC_SWEEP_KS="${ECC_KS}"
    ECC_CONST_ARCH="$(_ecc_first "${ECC_ARCHS}")"
    ECC_CONST_MODEL="$(_ecc_first "${ECC_MODELS}")"
    ECC_CONST_K="$(_ecc_first "${ECC_KS}")"
fi

# More than one model on an architecture sweep is the PANEL layout: one panel
# per model, top to bottom, the same x axis repeated inside each. It adds no
# axis and no renderer -- it answers the one question a single sweep cannot,
# whether the architecture ranking survives changing the network.
if [ "${ECC_RECON_MODELING}" = "1" ]; then
    :                                     # already decided above: recon
elif [ "$(_ecc_count "${ECC_MODELS}")" -gt 1 ] && [ "${ECC_SWEEP}" = "arch" ]; then
    : "${ECC_EXPERIMENT:=panels}"
    : "${ECC_PANEL_MODELS:=${ECC_MODELS}}"
else
    : "${ECC_EXPERIMENT:=sweep}"
    : "${ECC_PANEL_MODELS:=}"
fi

# One fixed figure name per axis -- but only for a WHOLE-MODEL run. With
# ECC_LAYERS set, the layer scope must stay in the name, which is what leaving
# ECC_STEM empty does. The `=` without a colon means an ECC_STEM explicitly
# exported as empty survives.
if [ -z "${ECC_LAYERS}" ] && [ "${ECC_RECON_MODELING}" != "1" ]; then
    case "${ECC_SWEEP}" in
        arch|archs|architecture*) : "${ECC_STEM=ArchitectureSweep}" ;;
        model*)                   : "${ECC_STEM=ModelSweep}" ;;
        bch|k|code*)              : "${ECC_STEM=BCHsweep}" ;;
    esac
fi
: "${ECC_STEM=}"

# Regenerate the (architecture, model) list the SLURM array walks, from the
# lists in section 3. GENERATED, never hand-edited: a stale list whose length
# disagrees with --array is the classic way to silently skip pairs.
ecc_write_taskfile() {
    local out="${1:-${ECC_TASKFILE}}" m a
    mkdir -p "$(dirname "${out}")"
    : > "${out}"
    for m in ${ECC_MODELS}; do
        for a in ${ECC_ARCHS}; do
            printf '%s %s\n' "${a}" "${m}" >> "${out}"
        done
    done
}

export ECC_PROJECT_ROOT ECC_SIF ECC_TASKFILE ECC_USE_CONTAINER ECC_PYTHON \
       ECC_MAPPER_THREADS ECC_MAPPER_ALGORITHM ECC_MAPPER_SEARCH_SIZE \
       ECC_VICTORY ECC_VICTORY_SCALING ECC_MAPPER_TIMEOUT \
       ECC_MAPPER_MAX_PERMUTATIONS ECC_MAPPER_SEED ECC_OPT_METRIC \
       ECC_RERUN_OPTIMISER ECC_ARCHS ECC_MODELS ECC_KS ECC_CODE_N \
       ECC_APPROACHES ECC_SWEEP ECC_EVAL_EXPERIMENTS ECC_PHASE ECC_LAYERS \
       ECC_RECON_MODELING ECC_RECON_ARCH ECC_RECON_ARCHS ECC_RECON_MODEL ECC_RECON_LAYER \
       ECC_RECON_CODE_N \
       ECC_RECON_K ECC_RECON_STEM ECC_RECON_PLACEMENT_LIST \
       ECC_RECON_OPTIMIZER RECON_OPTIMIZER ECC_RECON_ERT_AWARE ECC_RECON_ERT_ARM \
       ECC_RECON_PACKING \
       ECC_RECON_ENCODER_GRANULARITY \
       ECC_RECON_ONCHIP_FRACTION ECC_RECON_PLACEMENT_CHARGES_DECODE \
       ECC_RECON_REQUIRE_GROUP_RESIDENCY \
       ECC_RECON_DECODE_SITE ECC_RECON_ENCODER_SITE ECC_RECON_CLOCK_GATING_PCT ECC_ENERGY_MODEL_REV \
       ECC_DRAM_PJ_PER_BIT ECC_BASELINE_DRAM_PJ_PER_BIT \
       ECC_DRAM_BACKGROUND_PJ ECC_DRAM_REFRESH_PJ ECC_DRAM_BANDWIDTH_MBPS \
       ECC_STATIC_ENERGY ECC_LEAKAGE_NW_LIST ECC_LATENCY_MODEL \
       ECC_ARCH_CLOCK_MHZ_LIST ECC_RECON_BW_SCALE ECC_ONCHIP_BW_BITAWARE \
       ECC_WEIGHT_BITS ECC_ACTIVATION_BITS ECC_ACC_BITS ECC_ARCH_FIDELITY \
       ECC_FORCE_DATAWIDTH ECC_FORCE_TECHNOLOGY ECC_DRAM_DEPTH ECC_MAC_PJ_OVERRIDE \
       ECC_WEIGHT_CAPACITY_SCALE ECC_WEIGHT_CAPACITY_SCOPE \
       ECC_WEIGHT_FACTOR_RELAX ECC_MAPSPACE_CONSTRAIN \
       ECC_WEIGHT_DATAWIDTH ECC_WEIGHT_DATAWIDTH_LEVELS \
       ECC_WEIGHT_DEPTH_SCALE ECC_WEIGHT_DEPTH_LEVELS \
       ECC_WEIGHT_WIDTH_GLB_MULT ECC_DISABLE_ASSERT_PAIR_GEOMETRY \
       ECC_DEPTH_SWEEP_SCALES ECC_DEPTH_SWEEP_RECON_DW \
       ECC_DEPTH_SWEEP_GATE_VICTORIES ECC_DEPTH_SWEEP_GATE_SCALES \
       ECC_GLOBAL_CYCLE_SECONDS ECC_NOC ECC_NOC_WIRE_PJ_PER_BIT_MM \
       ECC_NOC_ROUTER_PJ ECC_NOC_PE_LATCH_PJ ECC_NOC_SCALE \
       ECC_PARITY_GROUPING ECC_PARITY_CHARGE_PADDING ECC_EMB_WEIGHTS_PER_CW \
       ECC_DECODE ECC_DECODE_PJ_BASE ECC_DECODE_PJ_EMB ECC_RECON_CHARGES_DECODE \
       ECC_RECON_JSON ECC_RECON_PJ ECC_RECON_INCREMENTAL_PJ_LIST ECC_RECON_IDLE_PJ_LIST \
       ECC_WEAK ECC_WEAK_N ECC_WEAK_K ECC_BASELINE_INFLATES_ONCHIP \
       ECC_SPLIT_READ_WRITE ECC_CLASSIFY ECC_ALLOW \
       ECC_ACCOUNT ECC_QOS ECC_PARTITION ECC_MAP_CPUS ECC_MAP_MEM ECC_MAP_TIME \
       ECC_CONCURRENCY ECC_EVAL_CPUS ECC_EVAL_MEM ECC_EVAL_TIME \
       ECC_RESULTS_DIR ECC_PALETTE ECC_FORMATS ECC_DPI ECC_NICE_LABELS \
       ECC_TITLE_NOTE ECC_STEM ECC_OVERWRITE ECC_CACHE_STRICT ECC_RUN_NOTE \
       ECC_REPLOT_ONLY ECC_FROM_CACHE ECC_VERBOSE \
       ECC_RECON_INCREMENTAL_FALLBACK_PJ ECC_RECON_IDLE_FALLBACK_PJ \
       ECC_INPUT_HW ECC_SEQ ECC_INCLUDE_LM_HEAD ECC_INCLUDE_EMBEDDING \
       ECC_SWEEP_ARCHS ECC_SWEEP_MODELS ECC_SWEEP_KS \
       ECC_CONST_ARCH ECC_CONST_MODEL ECC_CONST_K \
       ECC_EXPERIMENT ECC_PANEL_MODELS \
       ECC_ARCH_PIN_DIR
