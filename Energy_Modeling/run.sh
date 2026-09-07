#!/usr/bin/env bash
# =============================================================================
#  run.sh  --  THE ONE FILE YOU EDIT, AND THE ONE FILE YOU RUN
# =============================================================================
#
#      cd /home/workspace
#      bash run.sh
#
#  HOW A RUN IS SHAPED
#  -------------------
#  All three ECC arms (baseline / embedded / recon+) are ALWAYS drawn, so the
#  arms are never a sweep. Of the other three axes you sweep exactly ONE and
#  hold the other two at their constant:
#
#      ECC_SWEEP=bch     x = BCH(63, K)         holds  ARCH + MODEL
#      ECC_SWEEP=model   x = the networks       holds  ARCH + K
#      ECC_SWEEP=arch    x = the accelerators   holds  MODEL + K
#
#  Section 2 lists every legal value. Section 3 is what each sweep walks.
#  Section 4 is the constants -- those are what the two held axes actually use,
#  whichever sweep is chosen. Section 5 picks the sweep.
#
#  QUICK RUNS (development mode -- bounded search, two real layers)
#  ------------------------------------------------------------------
#      DEV_LAYERS="layer3.0.downsample.0 layer4.1.conv2"
#
#      bash run.sh validate                      # no container needed
#      ECC_LAYERS="$DEV_LAYERS" ECC_VICTORY=100 ECC_MAPPER_THREADS=8 \
#        bash run.sh map                         # container; ~minutes
#      ECC_LAYERS="$DEV_LAYERS" ECC_VICTORY=100 ECC_MAPPER_THREADS=8 \
#        bash run.sh baseline --eval             # no container; seconds
#
#  ECC_VICTORY=100 bounds the search and ECC_MAPPER_THREADS=8 pins the thread
#  count -- leaving it empty means "all cores", which differs per machine and
#  therefore changes the search. Timeloop's v4 mapper exposes no random seed
#  (ECC_MAPPER_SEED is recorded but cannot make the search deterministic), so
#  pinning the thread count is the strongest reproducibility available.
#
#  ONE FIGURE PER SWEEP, ALWAYS THE SAME NAME
#  ------------------------------------------
#      results/figures/BCHsweep.png          ECC_SWEEP=bch
#      results/figures/ModelSweep.png        ECC_SWEEP=model
#      results/figures/ArchitectureSweep.png ECC_SWEEP=arch
#
#  plus the matching tables/<name>.csv and manifests/<name>.json. Change a
#  constant, re-run, and those files are REWRITTEN -- the results directory
#  never grows past three sweeps. The manifest says which constants produced
#  the figure currently on disk.
#
#  Every value below uses ${VAR:=default}, so anything already in the
#  environment WINS over what is written here. One-off runs cost nothing:
#
#      ECC_SWEEP=arch bash run.sh                 # sweep architectures instead
#      ECC_CONST_MODEL=resnet50 bash run.sh       # same sweep, different constant
#      bash run.sh --replot                       # figure only, from results/_raw/
#      bash run.sh --dry-run                      # validate the config and stop
#      bash run.sh diagnose                       # audit the architectures
#      bash run.sh --sweep model --arch simba_like
# =============================================================================
set -euo pipefail

# ============================================================ 1. WHAT TO RUN
# sweep    : the figure (default)
# baseline : Task 1. The conventional-ECC result -- weights protected, BCH
#            parity external in DRAM -- written as one JSON per architecture
#            under results/evaluation/. See docs/RESULTS_SCHEMA.md.
# validate : check every architecture against the shared comparison contract in
#            archs/_shared/standard.yaml. Needs no container and no cache.
#            RUN THIS FIRST after touching any arch YAML.
# map      : generate and cache the mappings for the selected layers, then stop.
#            The mapping half of the mapping/evaluation split.
# diagnose : audit the architectures against MEASURED results.
# panels   : ONE image, one PANEL per model in ECC_PANEL_MODELS, with the
#            chosen sweep repeated inside each -- top panel first. Not a fourth
#            sweep: the x axis is still ECC_SWEEP's, and the bars are drawn by
#            the same renderer. It answers the one question a single sweep
#            cannot -- whether the architecture ranking survives changing the
#            network. Its output name carries the panel models, so it never
#            lands on top of ArchitectureSweep.png:
#
#              ECC_PANEL_MODELS="resnet18 mobilenet_v2" bash run.sh panels
: "${ECC_EXPERIMENT:=sweep}"

# Which models get a panel, top to bottom. Only read by ECC_EXPERIMENT=panels.
: "${ECC_PANEL_MODELS:=resnet18 mobilenet_v2}"

# ---------------------------------------------- 1b. DEVELOPMENT MODE (Task 1+)
#
#   THE POINT. A full-model sweep on a cold cache is hours. One or two real
#   layers is minutes, which is what makes it possible to change the model and
#   re-check it in the same sitting. Everything a development run writes is
#   namespaced by the layer selection, so a two-layer number can never be read
#   back as a full-model number.
#
#   ECC_LAYERS names layers by their STABLE workload name -- the name the
#   network gives the module, e.g. `layer4.1.conv2` -- never by index. An index
#   moves when the workload file is regenerated; a name does not.
#
#   EMPTY means the full model.
#
#   The recorded development pair for resnet18, and why these two:
#
#     layer3.0.downsample.0   C128 M256 R1 S1 P14 Q14 stride 2   32,768 weights
#         The quick check. A 1x1 projection: small tensors, few codewords,
#         maps in seconds on every architecture. Its job is to prove the
#         pipeline runs, not to say anything about energy.
#
#     layer4.1.conv2          C512 M512 R3 S3 P7 Q7             2,359,296 weights
#         The contrasting layer. 72x the weights of the first one and a tiny
#         7x7 feature map, so it is weight-dominated and stresses exactly what
#         this study measures: on-chip weight capacity and DRAM weight refetch.
#         Together the two span a factor of 72 in codeword count.
#
: "${ECC_LAYERS:=}"

# Which half of the study a result belongs to. Pre = the mapping is ECC-unaware
# and the ECC effect is applied during evaluation (Tasks 1-3). Post = the
# mapping itself was optimised for the reduced weight width (Task 4+).
# Task 1 is Pre by construction and refuses to run as Post.
: "${ECC_PHASE:=Pre}"

# Results are never silently overwritten. Set to 1 to replace an existing file.
: "${ECC_OVERWRITE:=0}"

# 1 -> a cached mapping is only reused if its mapping.json sidecar proves it was
#      computed for the CURRENT architecture YAML and mapper settings.
# 0 -> also accept pre-Task-1 cache entries, which have no sidecar and cannot
#      prove anything. Those results are labelled `legacy` and warned about.
#      Use it to reuse the existing hours of compute; do not publish from it.
: "${ECC_CACHE_STRICT:=1}"

# Free text recorded in every result JSON. Say what the run was for.
: "${ECC_RUN_NOTE:=}"

# ====================================================== 2. THE LEGAL VALUES
# Reference only -- nothing here is read by the code. Copy from these lists
# into sections 3 and 4.
#
#   ARCHITECTURES
#     eyeriss_like              Eyeriss v1
#     eyeriss_v2_like           Eyeriss v2 (authored here, modelled dense)
#     eyeriss_v2_like_wglb      Eyeriss v2 with weights kept in the GLB
#     simple_weight_stationary
#     simple_output_stationary
#     simple_input_stationary   authored here; reproduces no paper
#     simba_like                NVIDIA Simba
#
#   MODELS -- CNNs, easy to hard to map
#     resnet18  resnet50  densenet121  squeezenet1_1
#     mobilenet_v2  efficientnet_b0  convnext_tiny  xception
#   MODELS -- transformers (LLM weight matmuls)
#     distilgpt2  gpt2  bert_base  gpt2_medium  tinyllama
#   One sweep cannot mix the two families: they come from different workload
#   files. A model sweep is all-CNN or all-transformer.
#
#   BCH K, at N=63 -- weak to strong, all six have Design Compiler energies
#     57 (t=1)   51 (t=2)   45 (t=3)   39 (t=4)   36 (t=5)   30 (t=6)

# ================================================== 3. WHAT EACH SWEEP WALKS
# Only the list belonging to the CHOSEN sweep is used; the other two are
# ignored. Order here is the left-to-right order on the x axis.

# ECC_SWEEP=arch walks these:
: "${ECC_SWEEP_ARCHS:=eyeriss_v2_like eyeriss_like simple_weight_stationary simple_output_stationary simple_input_stationary simba_like}"

# ECC_SWEEP=model walks these:
: "${ECC_SWEEP_MODELS:=resnet18 resnet50 densenet121 squeezenet1_1 mobilenet_v2 efficientnet_b0 convnext_tiny xception}"

# ECC_SWEEP=bch walks these K values, at N below:
: "${ECC_SWEEP_KS:=57 51 45 39 36 30}"

# ==================================================== 4. THE HELD CONSTANTS
# The two axes this run does NOT sweep are pinned here. These win over
# anything in section 3 -- a BCH sweep runs on ECC_CONST_ARCH / ECC_CONST_MODEL
# no matter what the arch and model sweep lists say.
: "${ECC_CONST_ARCH:=eyeriss_like}"     # held when sweeping bch or model
: "${ECC_CONST_MODEL:=resnet18}"        # held when sweeping bch or arch
: "${ECC_CONST_K:=51}"                  # held when sweeping model or arch

# =========================================================== 5. WHICH SWEEP
# bch | model | arch   <- the one line that decides which figure comes out
: "${ECC_SWEEP:=arch}"

# =============================================================================
#  Below here: the modelling itself. Defaults are the published settings.
# =============================================================================

# ---------------------------------------------------------------- APPROACHES
# baseline : parity stored beside the data -> DRAM weight traffic x N/K
# embedded : parity inside the already-stored word -> no DRAM inflation
# recon    : as embedded, but only K/N of the weights are held on chip and the
#            parity is regenerated by the synthesized datapath
# The FIRST entry is the reference the savings percentages are measured against.
# All three are always plotted; only reorder this if you want a different
# reference arm.
: "${ECC_APPROACHES:=baseline embedded recon}"

# ------------------------------------------------------------ CODE GEOMETRY
: "${ECC_WEIGHT_BITS:=8}"      # weight quantization: the protected payload
: "${ECC_ACTIVATION_BITS:=8}"  # input activations. Distinct from the above on
                               # purpose -- `bash run.sh validate` checks each
                               # architecture declares the right one at the
                               # right level, and would fail if they diverged
                               # without the YAMLs following.
: "${ECC_CODE_N:=63}"          # codeword length. K comes from the sweep or
                               # from ECC_CONST_K above.

# ACCUMULATOR PRECISION IS NOT STANDARDIZED, ON PURPOSE.
# Each design keeps its PUBLISHED partial-sum width -- Eyeriss v1 16b, Eyeriss
# v2 20b, Simba 24b -- because psum precision is an architectural property and
# forcing one width on all of them would be equalising the architectures rather
# than the experiment. Every width is cited in archs/_shared/standard.yaml.
# All ECC arms of one architecture share its width (none of them touches the
# psum path), which is what makes an arm-to-arm comparison valid.
#
# Setting this forces a common width for a SENSITIVITY STUDY. It gets its own
# mapper cache and its own results namespace (w8a8accNN), is labelled as a
# sensitivity study in the banner and in every JSON, and is never the primary
# result. Empty = paper-native = the primary comparison.
: "${ECC_ACC_BITS:=}"

# --------------------------------------------------- EXTERNAL PARITY ACCOUNTING
# How the conventional baseline counts BCH parity in DRAM. NOT a flat n/k:
# a weight cannot straddle a codeword, so at BCH(63,51) over 8-bit weights only
# 6 whole weights fit in the 51-bit message field and 3 bits are padding. The
# accounted overhead is 31.25%, against the 23.53% a flat n/k model charges.
#
#   layer : each layer's weight tensor is its own codeword stream, so each pays
#           its own tail padding. What a real allocator does, and conservative.
#   model : one stream over the whole model. Differs by at most one codeword
#           per layer -- nothing on a full model, visible on one layer.
: "${ECC_PARITY_GROUPING:=layer}"

# Do the message-padding bits cross the DRAM bus with the parity? They sit
# inside the stored codeword, so yes by default. Set to 0 to charge parity only
# (25.00% instead of 31.25%), which models a layout that packs weights across
# codeword boundaries and re-splits them at the ECC engine -- a different
# memory layout, not a cheaper version of this one.
: "${ECC_PARITY_CHARGE_PADDING:=1}"

# Weights per codeword used to COUNT embedded-arm decodes. Empty means "the same
# codeword as the baseline", the shared-geometry model. Set to 8 to reproduce
# the older two-arm scripts, which charged the embedded arm per byte.
: "${ECC_EMB_WEIGHTS_PER_CW:=}"

# ------------------------------------------------------------- DECODE ENERGY
# 0 -> decode charged as ZERO for every arm; the category sums to zero and is
#      dropped from the bars and the legend. This is the published setting,
#      because the numbers below are ESTIMATES, not measurements.
: "${ECC_DECODE:=0}"
: "${ECC_DECODE_PJ_BASE:=40.0}"     # pJ/codeword, BCH(63,51) syndrome+Chien
: "${ECC_DECODE_PJ_EMB:=40.0}"      # same code -> same decoder
: "${ECC_RECON_CHARGES_DECODE:=1}"  # does recon+ still detect before rebuilding?

# ------------------------------------------- RECONSTRUCTION (Design Compiler)
# FreePDK45/OSU gscl45nm, 1.1 V, 1 ns clock, pre-layout with a wire-load model.
# The synthesis run is matched on (N, K), so the K sweep and a fixed-K run read
# the same table and neither can be costed at the wrong code's datapath.
: "${ECC_RECON_JSON:=data/dc/BCH_N63_results.json}"
: "${ECC_RECON_INCLUDE_IDLE:=1}"    # 1 -> incremental + idle; 0 -> incremental
: "${ECC_RECON_PJ:=}"               # set to bypass the JSON entirely

# ----------------------------------------------------------- WEAK ECC OVERLAY
# A light SRAM-side code on top of the strong code. Weights pay it under every
# arm; inputs pay it under the baseline only.
: "${ECC_WEAK:=0}"
: "${ECC_WEAK_N:=63}"
: "${ECC_WEAK_K:=57}"

# ------------------------------------------------------- MODELLING SWITCHES
# Does the baseline's parity inflate ON-CHIP traffic too, or only DRAM?
# 0 IS THE CONVENTIONAL BASELINE and is what Task 1 specifies: external parity
# is consumed by the off-chip ECC correction and is not written into on-chip
# weight SRAM/RF. 1 reproduces the older figures and is warned about in the
# result JSON.
: "${ECC_BASELINE_INFLATES_ONCHIP:=0}"

# Split the two on-chip categories into read and write.
: "${ECC_SPLIT_READ_WRITE:=0}"

# How a storage level is assigned to "Global buffer" vs "On-chip SRAM/RF":
#   instances : one instance = shared global buffer, replicated = local. Correct
#               for simba_like, whose per-PE buffers are named "...Buffer" and
#               which the legacy classifier therefore called a global buffer.
#   name      : legacy substring matching. Reproduces the old figures exactly.
# Totals are identical either way; only the split between the two moves.
: "${ECC_CLASSIFY:=instances}"

# ---------------------------------------------------- ARCHITECTURE FIDELITY
# paper : use archs/<name>/arch_paper.yaml where it exists. Those files declare
#         each design's storage precisions and scratchpad geometries as its
#         PAPER does, with the citation for every number in a comment. Three
#         things they fix, all of which changed the architecture ordering:
#
#           * Partial sums are billed at the accumulator width. The stock
#             eyeriss_like declares ONE shared_glb at `datawidth: 8` holding
#             both Inputs and Outputs, so Timeloop packed 8 psums per 64b word
#             and charged 3.6 pJ per 16-bit partial sum -- against 11.7 pJ for
#             Eyeriss v2's honestly-declared 20b psum GLB. That one line was
#             most of the measured v1-beats-v2 gap. Same defect in simba_like's
#             mac (adder_width 16 against a 24-bit accumulation buffer).
#           * Scratchpad sizes match the published tables: eyeriss v1 filter
#             spad 224x16b and psum spad 24x16b (stock said 192 and 16);
#             eyeriss v2 weight spad 96x24b = 288B (the model read the 24b word
#             as 2 bytes, understating v2's on-chip weights by a third).
#           * The two "simple" designs get 8-bit operands and 16-bit psums
#             natively, which is what ECC_FORCE_DATAWIDTH=8 was working around.
#
# stock : the designs exactly as timeloop-accelergy-exercises ships them.
#         Reproduces every figure produced before those corrections.
: "${ECC_ARCH_FIDELITY:=paper}"

# ---------------------------------------------------- ARCHITECTURE FAIRNESS
# Leave both EMPTY to model each architecture exactly as its YAML declares it.
# Set them to compare architectures rather than architectures+silicon:
#
#   ECC_FORCE_DATAWIDTH=8       equalise storage datawidth to the weight width.
#                               The stock weight- and output-stationary designs
#                               declare datawidth 16 everywhere, so an "8-bit
#                               weights" study otherwise charges them 128 pJ per
#                               DRAM weight read while eyeriss and simba pay 64.
#                               Dedicated partial-sum levels are NOT forced:
#                               accumulator precision is a separate design
#                               choice, and Timeloop requires
#                               width % datawidth == 0.
#                               This is evaluated PER ARCHITECTURE -- a design
#                               already declared at the weight width is
#                               unaffected and keeps its existing mapper cache,
#                               so only the mis-declared designs are re-mapped.
#
#   ECC_FORCE_TECHNOLOGY=45nm   equalise the Accelergy process node. Unlike the
#                               above this goes through globals.yaml, which costs
#                               DRAM for every design at once, so it invalidates
#                               EVERY architecture's cache. All five arch
#                               containers already declare 45nm, so this only
#                               changes the node DRAM itself is costed at.
#
# A treatment that actually changes an architecture gets its own mapper cache
# and its own raw-cache subtree, so existing results are never overwritten or
# silently mixed in. See `bash run.sh diagnose` for which confounds apply.
: "${ECC_FORCE_TECHNOLOGY:=}"
: "${ECC_FORCE_DATAWIDTH:=}"
: "${ECC_DRAM_DEPTH:=1048576}"
: "${ECC_GLOBAL_CYCLE_SECONDS:=1e-9}"
#
#   ECC_NOC=1                   cost the interconnect (default). Timeloop's own
#                               wire model is a stub that returns 0, so with
#                               ECC_NOC=0 every network is free -- in the
#                               evaluator AND in the mapper's objective. The
#                               coefficients and their citations live in
#                               archs/_shared/noc.yaml. Enabling it moves the
#                               mapper cache (slug part `noc`): no pre-NoC
#                               mapping is ever read back as a costed one.
#   ECC_NOC_WIRE_PJ_PER_BIT_MM  override the shared 45nm wire constant (0.4).
#   ECC_NOC_SCALE=1             multiply every NoC term, for sensitivity runs.
: "${ECC_NOC:=1}"
#   ECC_NOC_ROUTER_PJ           override the shared per-flit router energy (0.25).
: "${ECC_NOC_WIRE_PJ_PER_BIT_MM:=}"
: "${ECC_NOC_ROUTER_PJ:=}"
: "${ECC_NOC_SCALE:=1}"

# ---------------------------------------------------------------- THE MAPPER
# What the mapper minimises. THIS IS AN ENERGY STUDY, so the default is energy.
# The mapper.yaml shipped with the exercises repo asks for `edp`, and EDP is not
# neutral between architectures: a design with more MACs can buy latency by
# spending energy, and EDP rewards that. Measured on mobilenet_v2's
# C160_M960_R1_S1_P7_Q7 layer, Eyeriss v2 (384 MACs) threw away a 6.84 pJ/MAC
# mapping for a 15.32 pJ/MAC one because it was 4x faster; Eyeriss v1 (168 MACs)
# gave up only 1.24x on the same layer. Set to edp to reproduce the old numbers.
: "${ECC_OPT_METRIC:=energy}"  # energy | edp | delay | last_level_accesses

# Mapper effort: Timeloop's search gives up after this many consecutive
# non-improving mappings, PER THREAD. Runtime scales roughly linearly with it.
: "${ECC_VICTORY:=500}"

# ...scaled by the architecture's loop-nest depth, because the number of
# candidate mappings grows combinatorially with it and a flat setting therefore
# searches a deep hierarchy less thoroughly -- then reports the shortfall as an
# architecture result. `levels` doubles per level past 8 (Eyeriss v1's depth),
# capped at 8x: v1 and the two simple designs get 500, Eyeriss v2 and Simba
# 1000, the v2+wglb variant 2000. `none` uses ECC_VICTORY flat, as before.
#
# THE SCALING is what removes the v1-vs-v2 bias; the BASE only sets how hard
# everything is searched. So leave the scaling alone and treat the base as a
# cost dial. To check convergence, run twice and compare rather than guessing
# a large base up front -- if 500 and 1000 agree you are converged:
#
#   ECC_VICTORY=500  bash run.sh      # then
#   ECC_VICTORY=1000 bash run.sh      # the two tables should match
#
# Each base gets its own mapper cache (a different victory means different
# mappings), so a run at one base does NOT seed a run at another. Budget the
# second run in full; do not start with the expensive one.
: "${ECC_VICTORY_SCALING:=levels}"   # levels | none

: "${ECC_MAPPER_THREADS:=}"    # empty -> all cores. PIN IT for a reproducible
                               # run: the thread count changes the search, and
                               # "all cores" is a different number per machine.
# Consecutive INVALID mappings before a thread abandons a region. Timeloop's
# own default is 1000; 10000 makes threads grind through infeasible corners of
# the mapspace for no benefit. Lowering it is close to free speed.
: "${ECC_MAPPER_TIMEOUT:=10000}"

# random | hybrid | exhaustive | linear_pruned | random_pruned
# `hybrid` walks loop PERMUTATIONS locally around one index factorization. On a
# mapspace of ~4e9 factorizations that is the wrong locality: the factorization
# is what sets DRAM traffic, and hybrid can freeze it. `random_pruned` samples
# factorizations uniformly and is usually both faster and better for a quick run.
: "${ECC_MAPPER_ALGORITHM:=hybrid}"

# ---------------------------------------------- DEVELOPMENT SPEED (BOUNDED SEARCH)
#
#   THESE MAKE THE MAPPER FAST BY MAKING IT WORSE. Both bound the search
#   directly instead of through the victory heuristic. Both are in the mapping
#   fingerprint AND the mapper-cache slug, so a bounded run gets its own cache
#   and its mappings can never be read back as a converged result; every result
#   JSON from a bounded run also carries a warning saying so.
#
#   ECC_MAPPER_SEARCH_SIZE   hard cap on VALID mappings examined per thread.
#                            Empty = uncapped = the publishable setting.
#                            200-500 gives a usable answer in a fraction of the
#                            time. This is the strongest single speed knob.
#
#   ECC_MAPPER_MAX_PERMUTATIONS  loop permutations tried per index
#                            factorization (Timeloop default 16). Lowering it
#                            to 4 makes the search move through FACTORIZATIONS
#                            ~4x faster, which is the dimension that actually
#                            decides DRAM traffic.
#
#   A fast development run, all four knobs together:
#
#     ECC_MAPPER_SEARCH_SIZE=300 ECC_MAPPER_MAX_PERMUTATIONS=4 #     ECC_MAPPER_ALGORITHM=random_pruned ECC_MAPPER_TIMEOUT=500 #     ECC_MAPPER_THREADS=16 ECC_VICTORY=100 bash run.sh map
#
: "${ECC_MAPPER_SEARCH_SIZE:=}"
: "${ECC_MAPPER_MAX_PERMUTATIONS:=16}"

# Recorded in every mapping sidecar and result JSON, but timeloop-mapper v4
# exposes NO random seed -- pytimeloop's Mapper spec has no such attribute. So
# this documents what was asked for; it does not make the search deterministic.
# What bounds the search is ECC_VICTORY and ECC_MAPPER_THREADS, both of which
# are in the mapping fingerprint.
: "${ECC_MAPPER_SEED:=}"

# Everything above this line that changes what the mapper returns -- the
# objective, the effort, the fidelity -- gets its own mapper-cache subdirectory,
# so the ~hours of mappings already under ecc_energy_study/outputs/ are never
# overwritten or mixed in. To reproduce every pre-correction figure exactly:
#
#   ECC_ARCH_FIDELITY=stock ECC_OPT_METRIC=edp ECC_VICTORY=500 \
#     ECC_VICTORY_SCALING=none bash run.sh --replot
#
# which is spelled as the empty treatment and reads the original cache.

# --------------------------------------------------------------- THE OUTPUT
: "${ECC_RESULTS_DIR:=results}"
: "${ECC_REPLOT_ONLY:=0}"      # 1 -> skip the mapper entirely and rebuild the
                               #      figure from results/_raw/. No container
                               #      needed. Use after changing the ECC config.
: "${ECC_FROM_CACHE:=0}"       # 1 -> use only mappings ALREADY solved under
                               #      ecc_energy_study/outputs/ and never invoke
                               #      Timeloop. No container needed. Use after
                               #      changing ECC_CLASSIFY, which needs the
                               #      stats re-parsed but nothing re-mapped.
: "${ECC_PALETTE:=house}"      # house | cvd (colourblind-safe Okabe-Ito)
: "${ECC_FORMATS:=png pdf}"
: "${ECC_DPI:=400}"
: "${ECC_NICE_LABELS:=1}"      # 1 -> "Eyeriss v2"; 0 -> "eyeriss_v2_like"
: "${ECC_TITLE_NOTE:=}"        # free text appended to the figure title

# =============================================================================
export ECC_EXPERIMENT ECC_SWEEP ECC_SWEEP_ARCHS ECC_SWEEP_MODELS ECC_SWEEP_KS \
       ECC_PANEL_MODELS \
       ECC_CONST_ARCH ECC_CONST_MODEL ECC_CONST_K ECC_APPROACHES \
       ECC_LAYERS ECC_PHASE ECC_OVERWRITE ECC_CACHE_STRICT ECC_RUN_NOTE \
       ECC_ACTIVATION_BITS ECC_ACC_BITS \
       ECC_PARITY_GROUPING ECC_PARITY_CHARGE_PADDING ECC_MAPPER_SEED \
       ECC_WEIGHT_BITS ECC_CODE_N ECC_EMB_WEIGHTS_PER_CW \
       ECC_DECODE ECC_DECODE_PJ_BASE ECC_DECODE_PJ_EMB ECC_RECON_CHARGES_DECODE \
       ECC_RECON_JSON ECC_RECON_INCLUDE_IDLE ECC_RECON_PJ \
       ECC_WEAK ECC_WEAK_N ECC_WEAK_K \
       ECC_BASELINE_INFLATES_ONCHIP ECC_SPLIT_READ_WRITE ECC_CLASSIFY \
       ECC_ARCH_FIDELITY ECC_FORCE_TECHNOLOGY ECC_FORCE_DATAWIDTH ECC_DRAM_DEPTH \
       ECC_GLOBAL_CYCLE_SECONDS ECC_NOC ECC_NOC_WIRE_PJ_PER_BIT_MM ECC_NOC_ROUTER_PJ ECC_NOC_SCALE \
       ECC_OPT_METRIC ECC_VICTORY ECC_VICTORY_SCALING \
       ECC_MAPPER_THREADS ECC_MAPPER_TIMEOUT ECC_MAPPER_ALGORITHM        ECC_MAPPER_SEARCH_SIZE ECC_MAPPER_MAX_PERMUTATIONS \
       ECC_RESULTS_DIR ECC_REPLOT_ONLY ECC_FROM_CACHE ECC_PALETTE \
       ECC_FORMATS ECC_DPI ECC_NICE_LABELS ECC_TITLE_NOTE ECC_STEM

cd "$(dirname "${BASH_SOURCE[0]}")"
exec "${ECC_PYTHON:-python3}" -m eccenergy "$@"
