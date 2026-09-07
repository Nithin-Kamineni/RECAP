#!/usr/bin/env bash
# =============================================================================
#  env.sh  --  EVERY KNOB IN THIS PROJECT, IN ONE FILE
# =============================================================================
#
#  THE ONE COMMAND
#
#      cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
#      module load apptainer
#      bash hpc/run_all.sh        # map -> evaluate -> plot, all from this file
#
#  Edit a value below and every stage sees it: the mapping optimiser (Timeloop),
#  the energy evaluator, the figures, and the SLURM submission. No other file
#  declares a default any more -- run.sh, hpc/run_all.sh and hpc/map.sbatch all
#  source this one. That matters most for the mapper settings in section 2:
#  they are hashed into the mapping cache key, so when the mapping stage and the
#  evaluation stage kept separate copies of them, a drift between the copies
#  meant the evaluator silently read a DIFFERENT cache than the mapper wrote.
#
#  Every value is written  : "${VAR:=default}"  so THE ENVIRONMENT STILL WINS.
#  A one-off never needs an edit here:
#
#      ECC_VICTORY=500 bash hpc/run_all.sh          # a cheaper search, once
#      ECC_MODELS=resnet50 bash hpc/run_all.sh      # a different network, once
#
#  CONTENTS
#     1  the few you change most often
#     2  the mapping optimiser -- what the search does
#     3  what the pipeline runs -- archs x models x codes x arms
#     4  reconstruction placement study     <-- PLACEHOLDER, no code reads it
#     5  hardware / architecture model
#     6  ECC accounting -- how each arm is charged
#     7  the cluster -- SLURM and the container
#     8  output and figures
#     9  miscellaneous
#    10  derived -- NOT knobs, nothing to edit
# =============================================================================


# =============================================================================
#  1. THE FEW YOU CHANGE MOST OFTEN
# =============================================================================

# Where this project lives. Derived from this file, so it is correct whatever
# directory env.sh is sourced from.
ECC_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Mapper threads. THIS IS PART OF THE MAPPING CACHE KEY: change it and every
# mapping already computed becomes a MISS and the whole sweep re-runs from cold.
# 18 is what every cached entry was built at. Inside a SLURM job it follows the
# cores actually granted, so the key can never drift from the allocation.
: "${ECC_MAPPER_THREADS:=${SLURM_CPUS_PER_TASK:-18}}"

# Which layers to run. EMPTY = the whole model = the real result.
# A layer list is DEVELOPMENT MODE: minutes instead of hours. Everything such a
# run writes is namespaced by the selection, so a two-layer number can never be
# read back as a full-model one. Name layers by their STABLE workload name
# (`layer4.1.conv2`), never by index.
#   the recorded resnet18 pair -- a fast 1x1 projection and a weight-heavy 3x3:
#   : "${ECC_LAYERS:=layer3.0.downsample.0 layer4.1.conv2}"
: "${ECC_LAYERS:=}"

# Run every stage inside the Timeloop+Accelergy container (1) or with the host
# python3 (0). The mapper ALWAYS needs the container; evaluation and plotting do
# not, but they need pandas/matplotlib/pyyaml, which a login node may lack.
: "${ECC_USE_CONTAINER:=1}"


# =============================================================================
#  2. THE MAPPING OPTIMISER  --  what Timeloop's search does
# =============================================================================
#  ALL of these are hashed into the mapping fingerprint, so each combination
#  gets its OWN mapper cache: changing one means a cold run, not a cheap re-run.

# random | hybrid | exhaustive | linear_pruned | random_pruned
# `hybrid` walks loop PERMUTATIONS around one index factorization, which is the
# wrong locality on a mapspace of ~4e9 factorizations -- the factorization is
# what sets DRAM traffic. random_pruned samples factorizations uniformly.
: "${ECC_MAPPER_ALGORITHM:=random_pruned}"

# Hard cap on VALID mappings examined, split across the threads.
# EMPTY = uncapped = converged = publishable.
# 20000 = bounded development pass; its results carry a BOUNDED warning and must
# not be quoted as an architecture ranking.
# The missing colon is deliberate: an exported EMPTY value must stay empty.
: "${ECC_MAPPER_SEARCH_SIZE=}"

# The search abandons a thread after this many consecutive non-improving valid
# mappings. Runtime scales roughly linearly with it. To CONFIRM convergence, run
# at two values and check the totals do not move -- 500 then 1000, say.
: "${ECC_VICTORY:=2000}"

# ...scaled by loop-nest depth, because the candidate count grows
# combinatorially with depth: a flat number searches a deep hierarchy less
# thoroughly and then reports the shortfall as an architecture result.
#   levels : double per level past 8 (Eyeriss v1's depth), capped at 8x
#            -> 4000 on the 9-level designs
#   none   : use ECC_VICTORY flat
: "${ECC_VICTORY_SCALING:=levels}"

# Consecutive INVALID mappings before a thread abandons a region of the
# mapspace. Timeloop's own default is 1000; 10000 grinds through infeasible
# corners for no benefit, so lowering it is close to free speed.
: "${ECC_MAPPER_TIMEOUT:=2000}"

# Loop permutations tried per index factorization (Timeloop default 16).
# Lowering it to 4 moves the search through FACTORIZATIONS ~4x faster.
: "${ECC_MAPPER_MAX_PERMUTATIONS:=16}"

# What the mapper minimises: energy | edp | delay | last_level_accesses
# THIS IS AN ENERGY STUDY, so energy. EDP is not neutral between architectures:
# a design with more MACs can buy latency by spending energy, and EDP rewards
# that. Set edp only to reproduce the pre-correction numbers.
: "${ECC_OPT_METRIC:=energy}"

# Recorded in every mapping sidecar, but timeloop-mapper v4 exposes NO random
# seed, so this documents intent only. Pinning ECC_MAPPER_THREADS is the real
# reproducibility lever.
: "${ECC_MAPPER_SEED:=}"


# =============================================================================
#  3. WHAT THE PIPELINE RUNS
# =============================================================================
#  These lists ARE the run. The mapper solves every (architecture, model) pair
#  in them; the evaluator scores every arm; the figure puts ECC_SWEEP's list on
#  the x axis and holds the other two at the FIRST entry of their list.

# 1 -> re-run the mapping optimiser even when a valid cached mapping exists, and
#      overwrite that cache entry with the new one. Use it to refresh a mapping
#      after changing something the fingerprint does not capture, or to check a
#      mapping reproduces. Refused together with evaluation-only mode, which
#      forbids mapping outright.
# 0 -> a valid cache hit is reused. This is what makes a re-run seconds.
: "${ECC_RERUN_OPTIMISER:=0}"

# ---- architectures ---------------------------------------------------------
#   eyeriss_like              Eyeriss v1 (published)
#   eyeriss_like_wglb         ...with the published 8 kB filter GLB modelled
#   eyeriss_v2_like           Eyeriss v2 (authored here, modelled dense)
#   eyeriss_v2_like_wglb      ...with weights kept in the GLB
#   simple_weight_stationary  authored here; reproduces no paper
#   simple_output_stationary
#   simple_input_stationary
#   simba_like                a REFERENCE DESIGN named after Simba, not the chip
# eyeriss_like and eyeriss_like_wglb are a BRACKETING PAIR -- upper and lower
# bounds on Eyeriss v1's DRAM weight traffic. Quote them together or neither.
: "${ECC_ARCHS:=eyeriss_v2_like eyeriss_like simple_weight_stationary simple_output_stationary simple_input_stationary simba_like}"

# ---- models ----------------------------------------------------------------
#   CNNs          resnet18 resnet50 densenet121 squeezenet1_1
#                 mobilenet_v2 efficientnet_b0 convnext_tiny xception
#   transformers  distilgpt2 gpt2 bert_base gpt2_medium opt_125m distilbert tinyllama
# ONE RUN CANNOT MIX THE TWO FAMILIES: they come from different workload files.
# More than one model with ECC_SWEEP=arch draws one PANEL per model, which is
# the only honest way to show two networks at once.
: "${ECC_MODELS:=resnet18 mobilenet_v2}"

# ---- the BCH code ----------------------------------------------------------
: "${ECC_CODE_N:=63}"   # codeword length N; all three arms share ONE code
# K, weak to strong. All six have Design Compiler reconstruction energies:
#   57 (t=1)   51 (t=2)   45 (t=3)   39 (t=4)   36 (t=5)   30 (t=6)
# A list only matters when ECC_SWEEP=bch; otherwise the FIRST value is the code
# the whole run uses.
#   : "${ECC_KS:=57 51 45 39 36 30}"   # the full BCH sweep
: "${ECC_KS:=51}"

# ---- the ECC arms ----------------------------------------------------------
#   baseline  parity beside the data in DRAM -> weight traffic x N/K
#   embedded  parity inside the stored weights' own bits -> no DRAM inflation
#   recon     as embedded, but only K/N of the weights are held on chip and the
#             rest is regenerated by the synthesized datapath
# All three are ALWAYS drawn -- they are arms, never an axis. The FIRST entry is
# the reference the saving percentages are measured against.
: "${ECC_APPROACHES:=baseline embedded recon}"

# ---- which list goes on the x axis -----------------------------------------
#   arch   the accelerators  (holds model + code)
#   model  the networks      (holds arch  + code)
#   bch    BCH(63,K)         (holds arch  + model)
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
: "${ECC_PHASE:=Pre}"


# =============================================================================
#  4. RECONSTRUCTION PLACEMENT STUDY   --   PLACEHOLDER, NOT IMPLEMENTED
# =============================================================================
#  NOTHING IN THIS SECTION IS READ BY ANY CODE YET. It is written down now so
#  the file does not have to be restructured when the next step lands.
#
#  WHEN ECC_RECON_MODELING=1 this section will TAKE PRECEDENCE over section 3:
#  the study collapses to one architecture, one model and one code, and the x
#  axis becomes WHERE the reconstruction datapath sits rather than which
#  accelerator runs. Until then section 3 governs and these are inert.
: "${ECC_RECON_MODELING:=0}"

# The single point the placement study is run at.
: "${ECC_RECON_ARCH:=eyeriss_v2_like}"    # ONE architecture, from section 3's list
: "${ECC_RECON_MODEL:=resnet18}"          # ONE model
: "${ECC_RECON_CODE_N:=63}"               # ONE code geometry
: "${ECC_RECON_K:=51}"

# The placements explored, per architecture: one bar per entry, left to right.
# `baseline` and `embedded` ride along as reference bars so a placement is
# always read against them.
# A bash associative array CANNOT be exported, so anything that reads this must
# SOURCE env.sh -- run.sh, hpc/run_all.sh and hpc/map.sbatch all do.
declare -A ECC_RECON_PLACEMENTS=(
    [eyeriss_like]="baseline embedded recon1 recon2 recon3 recon4"
    [eyeriss_v2_like]="baseline embedded recon1 recon2 recon3 recon4"
    [simple_weight_stationary]="baseline embedded recon1 recon2 recon3 recon4"
)

# What each placement means, and what its bar is called in the figure. Fill the
# labels in as the placements are defined.
declare -A ECC_RECON_PLACEMENT_LABELS=(
    [baseline]="Baseline"
    [embedded]="Embedded"
    [recon1]="Recon @ DRAM"
    [recon2]="Recon @ global buffer"
    [recon3]="Recon @ PE array"
    [recon4]="Recon @ MAC"
)

# Where the placement results are cached, so a re-plot never re-runs the mapper,
# and what the figure/table/manifest are called.
: "${ECC_RECON_RESULTS_JSON:=results/recon/placements.json}"
: "${ECC_RECON_STEM:=ReconPlacement}"
: "${ECC_RECON_RERUN:=0}"                    # 1 = re-explore even when cached
# Fraction of the weights held on chip. EMPTY = derived from the code (K/N).
: "${ECC_RECON_ONCHIP_FRACTION:=}"
# Does a placement pay the decoder as well as the rebuild? (section 6 has the
# decoder energies themselves.)
: "${ECC_RECON_PLACEMENT_CHARGES_DECODE:=1}"


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
: "${ECC_NOC_WIRE_PJ_PER_BIT_MM:=}"   # override the shared 45nm wire constant (0.4)
: "${ECC_NOC_ROUTER_PJ:=}"            # override the shared per-flit router energy (0.25)
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
: "${ECC_RECON_INCLUDE_IDLE:=1}"     # 1 -> incremental + idle; 0 -> incremental
: "${ECC_RECON_PJ:=}"                # set to bypass the JSON entirely
# Used ONLY when the JSON has no entry for the (N,K) in play -- the BCH(63,51)
# numbers, so a missing entry degrades to a plausible cost instead of crashing.
# If you see these in a result's provenance, the table is missing a code.
: "${ECC_RECON_INCREMENTAL_FALLBACK_PJ:=1.8995}"
: "${ECC_RECON_IDLE_FALLBACK_PJ:=2.2301273}"

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
: "${ECC_QOS:=rewetz}"            # rewetz-b is the burst QOS: idle cores, low
                                  # priority, a 4-day limit
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
# Read ONLY by `python3 -m eccenergy.generate models <name>`, which turns a
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

: "${ECC_SWEEP_ARCHS:=${ECC_ARCHS}}"
: "${ECC_SWEEP_MODELS:=${ECC_MODELS}}"
: "${ECC_SWEEP_KS:=${ECC_KS}}"
: "${ECC_CONST_ARCH:=$(_ecc_first "${ECC_ARCHS}")}"
: "${ECC_CONST_MODEL:=$(_ecc_first "${ECC_MODELS}")}"
: "${ECC_CONST_K:=$(_ecc_first "${ECC_KS}")}"

# More than one model on an architecture sweep is the PANEL layout: one panel
# per model, top to bottom, the same x axis repeated inside each. It adds no
# axis and no renderer -- it answers the one question a single sweep cannot,
# whether the architecture ranking survives changing the network.
if [ "$(_ecc_count "${ECC_MODELS}")" -gt 1 ] && [ "${ECC_SWEEP}" = "arch" ]; then
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
if [ -z "${ECC_LAYERS}" ]; then
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
       ECC_RECON_MODELING ECC_RECON_ARCH ECC_RECON_MODEL ECC_RECON_CODE_N \
       ECC_RECON_K ECC_RECON_RESULTS_JSON ECC_RECON_STEM ECC_RECON_RERUN \
       ECC_RECON_ONCHIP_FRACTION ECC_RECON_PLACEMENT_CHARGES_DECODE \
       ECC_WEIGHT_BITS ECC_ACTIVATION_BITS ECC_ACC_BITS ECC_ARCH_FIDELITY \
       ECC_FORCE_DATAWIDTH ECC_FORCE_TECHNOLOGY ECC_DRAM_DEPTH \
       ECC_GLOBAL_CYCLE_SECONDS ECC_NOC ECC_NOC_WIRE_PJ_PER_BIT_MM \
       ECC_NOC_ROUTER_PJ ECC_NOC_SCALE \
       ECC_PARITY_GROUPING ECC_PARITY_CHARGE_PADDING ECC_EMB_WEIGHTS_PER_CW \
       ECC_DECODE ECC_DECODE_PJ_BASE ECC_DECODE_PJ_EMB ECC_RECON_CHARGES_DECODE \
       ECC_RECON_JSON ECC_RECON_INCLUDE_IDLE ECC_RECON_PJ \
       ECC_WEAK ECC_WEAK_N ECC_WEAK_K ECC_BASELINE_INFLATES_ONCHIP \
       ECC_SPLIT_READ_WRITE ECC_CLASSIFY \
       ECC_ACCOUNT ECC_QOS ECC_PARTITION ECC_MAP_CPUS ECC_MAP_MEM ECC_MAP_TIME \
       ECC_CONCURRENCY ECC_EVAL_CPUS ECC_EVAL_MEM ECC_EVAL_TIME \
       ECC_RESULTS_DIR ECC_PALETTE ECC_FORMATS ECC_DPI ECC_NICE_LABELS \
       ECC_TITLE_NOTE ECC_STEM ECC_OVERWRITE ECC_CACHE_STRICT ECC_RUN_NOTE \
       ECC_REPLOT_ONLY ECC_FROM_CACHE ECC_VERBOSE \
       ECC_RECON_INCREMENTAL_FALLBACK_PJ ECC_RECON_IDLE_FALLBACK_PJ \
       ECC_INPUT_HW ECC_SEQ ECC_INCLUDE_LM_HEAD ECC_INCLUDE_EMBEDDING \
       ECC_SWEEP_ARCHS ECC_SWEEP_MODELS ECC_SWEEP_KS \
       ECC_CONST_ARCH ECC_CONST_MODEL ECC_CONST_K \
       ECC_EXPERIMENT ECC_PANEL_MODELS
