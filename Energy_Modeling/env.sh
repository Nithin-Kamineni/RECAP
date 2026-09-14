#!/usr/bin/env bash
# =============================================================================
#  env.sh -- THE ONLY FILE YOU EDIT
# =============================================================================
#  Sourced by run.sh, hpc/run_all.sh, hpc/map.sbatch, hpc/tl.sh,
#  hpc/smoke_models.sh and restructure/snapshot.py, so a value cannot mean one
#  thing to the mapper and another to the evaluator.
#
#  EVERY VALUE IS `${VAR:=default}`, SO THE ENVIRONMENT WINS OVER THE FILE.
#  A one-off never needs an edit:  ECC_SWEEP=bch bash hpc/run_all.sh
#
#  HOW TO READ IT. Each knob has ONE comment line above it listing its options;
#  the long-form reasoning -- the decisions, the traps, what a value cost when
#  it was got wrong -- is in the PREAMBLE of the section it belongs to, out of
#  the way of the values. Scan the values; read the preamble when you change
#  one.
#
#  THE EIGHT SECTIONS, in the order EnvReorganisation 4.1 asks for -- what you
#  turn, the mapper, the chip, the prices, then the plumbing:
#
#    1  WHAT TO COMPUTE AND PLOT   the four lines that decide the study
#    2  THE MAPPER                 >>> COLDS THE MAPPER CACHE <<<
#    3  THE CHIP                   >>> COLDS THE MAPPER CACHE <<<
#    4  THE PRICES                 evaluator only -- re-priced from cache in ms
#    5  THE CLUSTER                SLURM and the container
#    6  OUTPUT AND FIGURES
#    7  MISCELLANEOUS
#    8  DERIVED                    NOT KNOBS. Nothing below it needs editing.
#
#  THE OLD TEN SECTIONS, AND WHERE THEY WENT (EnvReorganisation phase 4,
#  2026-09-14). A docstring elsewhere that still says "env.sh section N" and
#  has not been corrected means the number below on the left:
#
#    old 1 few you change most often -> 1 (ALLOW, LAYERS, JOBS), 2 (THREADS),
#                                       5 (USE_CONTAINER)
#    old 2 mapping optimiser         -> 2
#    old 3 what the pipeline runs    -> 1, except RERUN_OPTIMISER -> 2
#    old 4 reconstruction cost       -> 4
#    old 5 hardware / arch model     -> 3, except the depth trio -> 1 and
#                                       DISABLE_ASSERT_PAIR_GEOMETRY -> 4
#    old 6 ECC accounting            -> 4
#    old 7 the cluster               -> 5
#    old 8 output and figures        -> 6
#    old 9 miscellaneous             -> 7
#    old 10 derived                  -> 8
#
#  WHAT IS NOT IN THIS FILE, ON PURPOSE. Anything a DESIGN declares lives in
#  that design's own directory, so adding an architecture is one directory and
#  nothing else: `archs/<name>/design.yaml` (label, axis order, the constrained
#  mapspace, `clock_mhz:`, `leakage_nw:`), `archs/<name>/widths.yaml` (THE
#  WIDTH TABLE, q -> {spad_width, glb_width}), `archs/<name>/weight_path.yaml`
#  and `archs/<name>/placements.yaml` (the stages and the boundaries, loaded
#  together or not at all). `archs/_shared/standard.yaml`, `provenance.yaml`
#  and `noc.yaml` hold what must be identical across designs, where every
#  declared number came from, and the interconnect coefficients.
# =============================================================================

ECC_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# =============================================================================
#  1. WHAT TO COMPUTE AND PLOT
# =============================================================================
#  THE LINES THAT DECIDE THE STUDY are ECC_APPROACHES (which bars), ECC_SWEEP
#  (which x axis), ECC_METRICS (which y axes -- one figure ROW each), the three
#  lists (ECC_ARCHS / ECC_MODELS / ECC_KS, whose FIRST entry is the held value
#  on every axis that is not swept) and ECC_LAYERS (the scope).
#  `bash hpc/run_all.sh` then maps what is cold, evaluates and draws;
#  `--dry-run` prints the bill first. The whole figure is those three:
#  rows = ECC_METRICS, columns = ECC_SWEEP, bars = ECC_APPROACHES.
#
#  THE LISTS ARE THE CHIPS, so widening one ADDS mapper work -- it never moves
#  an existing cache entry, which is what separates them from sections 2 and 3.
#  The one pair here that does re-fingerprint is the buffer-depth scale: a
#  changed depth IS a different array, which is the whole point of ECC_SWEEP=
#  area, and each scale gets its own `wdepth<s>` cache.
#
#  DESIGNS DO NOT HAVE THE SAME BOUNDARIES. eyeriss_like_wglb and
#  simple_weight_stationary declare five placements, the two v2 variants four,
#  and three designs declare none at all -- so a five-name ECC_APPROACHES draws
#  five bars on one design and four on another, with a `[skip]` line saying so.
#  That is a warning, never a refusal: one ECC_APPROACHES has to be legal for
#  every design a sweep names.
#
#  EVERY RECONSTRUCTION BAR IS A REAL, MAPPED CHIP since EnvReorganisation
#  phase 6. The abstract `recon` arm that used to be drawn beside baseline and
#  embedded on the three sweeps -- one engine at the chip entrance, K/N applied
#  to every on-chip level of every design identically -- is RETIRED: no
#  boundary does that, it was an optimistic upper bound, and reporting rule R-6
#  existed only to stop it being quoted as a placement. `recon` in
#  ECC_APPROACHES now means ONE bar, the placement ECC_RECON_DEFAULT names, and
#  every bar on every figure is looked up in ITS OWN mapper cache at EVERY
#  swept point. The figures built on the abstract arm stay in FINDINGS as
#  history, labelled as the abstract arm.

# baseline | embedded | recon | recon1 | recon2 | recon3 | recon4 | recon5
#   `recon` on its own is ONE bar: the placement ECC_RECON_DEFAULT names
# : "${ECC_APPROACHES:=baseline embedded recon1 recon2 recon3 recon4 recon5}"
: "${ECC_APPROACHES:=baseline embedded recon}"

# recon1 | recon2 | recon3 | recon4 | recon5   <- what a bare `recon` bar MEANS
: "${ECC_RECON_DEFAULT:=recon2}"

# bch | model | arch | fix | area   <- an X AXIS. `area` here = buffer DEPTH;
#                                      silicon area is ECC_METRICS below
#   bch    BCH(63,K) over ECC_KS                 (arch, model held)
#   model  the networks of ECC_MODELS            (arch, code held)
#   arch   the designs of ECC_ARCHS              (model, code held)
#   fix    NO x axis: the placement study at ONE point -- the bars are what
#          ECC_APPROACHES names, at the FIRST entry of all three lists
#   area   the buffer-depth ladder ECC_DEPTH_SWEEP_SCALES, all three lists
#          held. Maps the ladder and submits NO eval: the sweep is a property
#          of the MAPPINGS, read with
#          `python3 -m eccenergy.report.dilation_view --levels`.
#          IT IS THE ONE AXIS WITH NO STACKED-BAR FIGURE, and
#          `sweep-has-no-figure` still refuses one for it -- reviewed and KEPT
#          in phase 6 session 2. What it needs, in order: a DEPTH in the
#          result namespace and the stem (two points of the ladder write one
#          `latest.json` today), a depth slot in `report/sweep.py`'s point
#          spec and `_point_cfg` (which pins arch/model/K and has nowhere to
#          put a fourth axis), and a dependent eval from `hpc/run_all.sh`,
#          which short-circuits this axis. None of that is the blocker: the
#          ladder is 7 scales x the arms of COLD mappings, so the figure could
#          not be LOOKED AT when it was built, and phase 6's own rule is that
#          every figure is reviewed by eye. Map the ladder first, then lift it.
: "${ECC_SWEEP:=bch}"

# energy | edp | latency | area   <- a Y AXIS: one figure ROW per entry, drawn
#                                    in THIS order. `area` here = SILICON area
#                                    (um2), not ECC_SWEEP's buffer depth
: "${ECC_METRICS:=energy edp latency}"

# eyeriss_like_wglb | simple_weight_stationary | eyeriss_v2_like_wglb | eyeriss_v2_like | simple_output_stationary | simple_input_stationary | simba_like   (FIRST = held)
#   THE THREE SCOPED DESIGNS since EnvReorganisation phase 6 (plan 6.8): the two
#   that declare a weight GLB above the array plus the weight-stationary one.
#   `eyeriss_v2_like_wglb` -- NOT `eyeriss_v2_like` -- is what "Eyeriss v2" means
#   here (CLAUDE.md, the user's decision 2026-09-14); the three designs with no
#   placements.yaml are still nameable and still draw their two reference bars.
: "${ECC_ARCHS:=eyeriss_like_wglb simple_weight_stationary eyeriss_v2_like_wglb}"

# resnet18 | mobilenet_v2 | resnet50 | efficientnet_b0 | densenet121 | squeezenet1_1 | convnext_tiny | xception   (FIRST = held; all-CNN or all-transformer, never mixed)
: "${ECC_MODELS:=resnet18 mobilenet_v2}"

# the BCH codeword length
: "${ECC_CODE_N:=63}"

# 57 (q=7) | 51 (q=6) | 45 (q=6) | 39 (q=5) | 36 (q=5) | 30 (q=4)   (FIRST = held)
: "${ECC_KS:=39 57 45 30}"

# the ladder ECC_SWEEP=area walks. sqrt(2) steps, NOT factor 2
: "${ECC_DEPTH_SWEEP_SCALES:=1.0 0.7071 0.5 0.3536 0.25 0.1768 0.125}"

# the HELD buffer depth on every axis that is not `area`. 1.0 = as declared
: "${ECC_WEIGHT_DEPTH_SCALE:=1.0}"

# which weight levels the depth scale rewrites. EMPTY = every weight-carrying one
#   : "${ECC_WEIGHT_DEPTH_LEVELS:=filter_glb}"
: "${ECC_WEIGHT_DEPTH_LEVELS:=}"

# dev | full   <- dev = THE SIX-LAYER DEVELOPMENT SCOPE (EnvReorganisation 6.9);
#                 full = whole models, which is what a published number is.
#                 An explicit ECC_LAYERS below WINS over either.
: "${ECC_SCOPE:=dev}"
case "$ECC_SCOPE" in
  dev)  : "${ECC_LAYERS:=resnet18=conv1 layer3.0.conv1; mobilenet_v2=features.1.conv.0.0 features.9.conv.2; efficientnet_b0=features.1.0.block.0.0 features.5.0.block.1.0}" ;;
  full) : "${ECC_LAYERS:=}" ;;
  *)    echo "env.sh: ECC_SCOPE=$ECC_SCOPE is not 'dev' or 'full'." >&2
        echo "  -> a typo here would silently sweep WHOLE MODELS, which is hours." >&2
        return 1 2>/dev/null || exit 1 ;;
esac

# EMPTY = the whole model | `conv1 layer3.0.conv1` | `resnet18=conv1 fc; mobilenet_v2=features.9.conv.2`
#                         | `resnet18.conv1` one layer of one model | `resnet18.all` that whole model
: "${ECC_LAYERS:=}"

# EMPTY = one SLURM array task per UNIT | N = bundle the units into N jobs
: "${ECC_JOBS:=8}"

# guards you have deliberately lifted, by id. EMPTY on every published run
#   ECC_ALLOW="zero-price"   ECC_ALLOW="zero-price,recon-no-split-read-write"
: "${ECC_ALLOW:=}"

# -----------------------------------------------------------------------------
#  SECTION 1, THE LONG VERSION
# -----------------------------------------------------------------------------
#  ECC_SWEEP=fix IS THE PLACEMENT STUDY. The axis is WHERE on the weight path
#  the reconstruction boundary sits, and what recon1..recon5 MEAN is a property
#  of the architecture, not of this file: each design declares its boundaries,
#  their labels and the levels each leaves reduced in its own weight_path.yaml
#  and placements.yaml. `recon` on its own means EVERY placement the design
#  declares; a reconN name selects a subset. Section 8 collapses ECC_SWEEP_* /
#  ECC_CONST_* onto the held point for `fix` and `area` with a bare `=`, so a
#  leftover in the shell cannot widen a study that only makes sense at one
#  point.
#
#  ECC_RECON_DEFAULT -- WHICH PLACEMENT A BARE `recon` BAR IS. A bch, model or
#  arch sweep wants ONE reconstruction bar beside its two reference bars; the
#  placement study wants every boundary. Both are ECC_APPROACHES: name `recon`
#  for the first, name recon1..recon5 for the second. `recon2` is the default
#  because it is the boundary the study reports (reconstruct at the weight
#  global buffer): it is mapped on its own chip at every code, its ERT bump is
#  real, and it is neither the degenerate R1 (which narrows nothing on chip)
#  nor the per-PE R5, whose engine count is the array. A design that does not
#  declare it draws no reconstruction bar and says so -- a `[skip]`, never a
#  refusal. It is EVALUATOR-ONLY in the same sense ECC_METRICS is: it decides
#  which cache is READ, never what a mapper solves, so it is not in the mapper
#  fingerprint. It does change the BILL, because the chips a run maps are the
#  bars it draws -- `--dry-run` prints them.
#
#  SEVERAL ARCHITECTURES ARE ONE PANEL PER NAME, each with its own x axis and
#  its own two reference bars, so a percentage on one panel says nothing about
#  the other. `fix` HOLDS the architecture, so the multi-panel placement figure
#  is `ECC_EXPERIMENT=recon ECC_SWEEP=arch ECC_SWEEP_ARCHS="a b"`.
#
#  ECC_LAYERS -- EMPTY IS THE WHOLE MODEL, which is what every published number
#  is. Two spellings, told apart by the `=`: a bare list is those layers of
#  whichever model is running; `model=a b; model2=c` is per network, and a
#  model with no entry runs whole. Layer names are per network, so a bare list
#  cannot scope a run that spans two of them -- a name the other model has not
#  got is REFUSED (`layers-not-in-model`), which is the point: a typo that
#  silently swept a whole model is the worse failure. The scope is in every
#  result path and figure title (`layers2__...`), so a two-layer number can
#  never be mistaken for a model. The commented line above is the development
#  scope EnvReorganisation 6.9 asks for -- six layers over three networks; use
#  it while changing code, never for a figure.
#
#  ECC_METRICS -- WHICH Y AXES, one figure ROW each, drawn in the order
#  `energy edp latency area` however they are typed. It is EVALUATOR ONLY: it
#  changes what is PLOTTED and never what is mapped, so it is not in the
#  mapper fingerprint and turning it up does not cold a single cache. Do not
#  confuse it with ECC_OPT_METRIC in section 2, which names what the MAPPER
#  OPTIMISES and IS in the fingerprint -- note that one spells the timing
#  objective `delay` and this one spells it `latency`, so a value of one can
#  never be pasted into the other.
#    energy   the stacked category breakdown this project has always drawn
#    edp      each bar's OWN energy x its OWN delay -- not the reference's, or
#             the row would restate the energy row
#    latency  the roofline's re-timing (ECC_LATENCY_MODEL, section 4). With
#             that knob at 0 every arm gets Timeloop's own cycles and the row
#             is flat BY CONSTRUCTION -- an absent term, not a measured null
#    area     SILICON area, and the one metric with no single source: the
#             accelerator from Accelergy's ART in the mapper cache, the
#             reconstruction engine from Design Compiler
#             (archs/_shared/recon_area.yaml, scraped from
#             data/dc/report_snapshots/ by tools/scrape_dc_area.py). Two
#             measurements from two flows, added because the question has no
#             other answer and labelled on every result so the sum is never
#             read as one measurement. There is NO fallback engine area: an
#             unsynthesized code is refused, because an invented area is an
#             invented silicon number.
#  THE DEFAULT IS ONE ROW. Section 8 of EnvReorganisation pictures a finished
#  run as `energy latency`; the default here is `energy` alone so that the
#  knob's arrival changed no figure and no table that already existed. Turn it
#  up when you want the rows -- nothing is re-mapped.
#
#  ECC_JOBS -- EMPTY is one array task per UNIT of work (one chip x one
#  distinct layer shape) with ECC_CONCURRENCY capping how many run at once,
#  which is what every run before EnvReorganisation phase 3 did. A number
#  BUNDLES: ECC_JOBS=12 over 72 units is 12 jobs of 6 units each, assigned
#  round-robin so no bundle collects all the big layers. A BUNDLE IS SERIAL, so
#  ECC_MAP_TIME (section 5) must cover the whole bundle and not one map.
#  `bash hpc/run_all.sh --dry-run` prints the bundling. It is BASH-ONLY on
#  purpose: it changes how the work is packaged, never what is computed, so it
#  is not a field of the configuration and not in the result record.
#
#  ECC_DEPTH_SWEEP_SCALES -- sqrt(2) STEPS, NOT FACTOR 2. The window where
#  Embedded cannot hold the tile and Recon can is exactly as wide, in depth, as
#  the effective-capacity ratio, so a factor-2 grid steps clean over a 1.17x or
#  1.33x window and reports a grid artifact as "no effect". sqrt(2) resolves
#  BCH(63,30) (2.00x) and BCH(63,39) (1.58x); BCH(63,45) (1.33x) and BCH(63,57)
#  (1.17x) would need ~x1.12 steps, about 20 depths -- do those only after the
#  strong codes show something. The ladder's convergence GATE (the embedded arm
#  at victories 2000/4000/10000, re-checked at depths x1.0 and x0.125) is READ,
#  not submitted, since EnvReorganisation phase 3 deleted hpc/map_depth_sweep.sh:
#  the budgets and the two depths are report/dilation_view.py's
#  GATE_VICTORIES / GATE_SCALES and --victories/--scales win over them.
#      bash hpc/tl.sh python3 -m eccenergy.report.dilation_view --gate
#  To MAP another budget, set ECC_VICTORY and re-run --map-only: the victory is
#  in the mapper fingerprint, so each budget is its own cache.
#
#  ECC_WEIGHT_DEPTH_LEVELS -- EMPTY is every weight-carrying level, which is
#  the default AND a limitation: one scale then moves `weights_spad` and
#  `filter_glb` TOGETHER, so it locates the zone but cannot say which level
#  bought it. A SECOND PASS holds one at x1 and sweeps the other, which is what
#  naming levels is for; that is two runs rather than one submission.
#  FINDINGS 7.8 predicts `filter_glb` is the one that matters (refetch on v1 is
#  set by the DRAM-level loop order over P/Q, a weight tile cannot index
#  either, so a buffer INSIDE the array can never absorb them -- measured flat
#  to x32 at 1% fill); CONFIRM it, do not assume it. A name no weight level has
#  is an ERROR, not a silent no-op.
#
#  ECC_ALLOW -- A GUARD RUNS ON EVERY RUN, including a SLURM job at 3am, and
#  stops it. GUARDS.md lists all 131 with a stable id and a TIER, and the tier
#  is the only thing that matters here:
#
#    1 PARSE       "ECC_VICTORY=abc is not an integer"          NEVER liftable
#    2 IMPOSSIBLE  "need K < N"; "a price may not be negative"  NEVER liftable
#    3 COUPLING    "SPLIT_READ_WRITE=1 in the placement study"  liftable, named
#    4 DERIVED     "a price of ZERO is an ablation"             liftable, named
#
#  Tiers 3 and 4 are the ABLATION BLOCKERS. Name one and it becomes a loud
#  warning instead of a refusal -- and THE OVERRIDE IS RECORDED, on the run
#  manifest as `guard_overrides` and on the figure's caveat list, so an
#  ablation cannot be published as if it were the study's own number.
#  NAME THE GUARD. There is deliberately no blanket "off": a blanket would be
#  set once, forgotten, and a wrong number would reach a figure with nothing
#  saying so.


# #############################################################################
# ##                                                                         ##
# ##   SECTIONS 2 AND 3 COLD THE MAPPER CACHE.                               ##
# ##                                                                         ##
# ##   A mapper cache directory is named after `arch_fingerprint()`, which    ##
# ##   hashes the patched YAML the mapper actually sees PLUS every mapper     ##
# ##   setting below. Change one value here and every entry already on disk   ##
# ##   becomes a MISS -- not wrong, not deleted, just no longer the chip or   ##
# ##   the search you are asking about. That is hours to days of SLURM, so    ##
# ##   BUDGET FOR IT AND BATCH IT: one submission, one architecture.          ##
# ##                                                                         ##
# ##   `bash hpc/run_all.sh --dry-run` prints what is cached and what is      ##
# ##   cold BEFORE anything is submitted. Read it after touching anything     ##
# ##   between here and the END OF THE COLD ZONE marker.                      ##
# ##                                                                         ##
# ##   NOTHING IN SECTIONS 4 TO 8 COLDS ANYTHING.                             ##
# ##                                                                         ##
# #############################################################################


# =============================================================================
#  2. THE MAPPER  --  what Timeloop's search does        >>> COLDS THE CACHE <<<
# =============================================================================
#  SINCE 2026-09-11 THE DEFAULTS ARE prompt_3's CONSTRAINED SEARCH, so a bare
#  `bash hpc/run_all.sh` is the exhaustive constrained mapspace and nothing
#  needs exporting: linear_pruned, victory 2000, timeout 100000000, and
#  ECC_MAPSPACE_CONSTRAIN=1 + ECC_WEIGHT_FACTOR_RELAX=1 in section 3.
#
#  ONLY A DESIGN WITH A `mapspace_free_levels:` ENTRY IS CONSTRAINED (today
#  eyeriss_like_wglb). On any other design these defaults are a SYSTEMATIC walk
#  of an UNCONSTRAINED space, which is the wrong regime (FINDINGS 2.2): either
#  write that design's free-set first (prompt_3, "Porting it") or set
#  ECC_MAPPER_ALGORITHM=random_pruned ECC_MAPPER_TIMEOUT=2000 back.
#
#  THE THREAD COUNT IS IN THE FINGERPRINT, so keep ECC_MAPPER_THREADS and
#  --cpus-per-task (section 5's ECC_MAP_CPUS, which defaults to it) equal: any
#  other value is a cold cache, not a faster run.

# random | hybrid | exhaustive | linear_pruned | random_pruned
: "${ECC_MAPPER_ALGORITHM:=linear_pruned}"

# consecutive valid-but-not-better mappings before the search gives up
: "${ECC_VICTORY:=2000}"

# levels | none        how the victory budget scales with the mapspace
: "${ECC_VICTORY_SCALING:=levels}"

# seconds; 100000000 = uncapped (prompt_3's constrained search is exhaustive)
: "${ECC_MAPPER_TIMEOUT:=100000000}"

# loop permutations the search may try per index factorization
: "${ECC_MAPPER_MAX_PERMUTATIONS:=16}"

# edp | energy | delay        what the mapper optimises
: "${ECC_OPT_METRIC:=edp}"

# EMPTY = unseeded, so a re-solve is NOT guaranteed to reproduce
: "${ECC_MAPPER_SEED:=}"

# EMPTY = uncapped         a hard cap on the number of mappings sampled
: "${ECC_MAPPER_SEARCH_SIZE=}"

# mapper threads. IN THE FINGERPRINT -- keep it equal to ECC_MAP_CPUS
: "${ECC_MAPPER_THREADS:=${SLURM_CPUS_PER_TASK:-16}}"

# 1 -> re-solve every unit | 0 -> a valid cache hit is reused
: "${ECC_RERUN_OPTIMISER:=0}"


# =============================================================================
#  3. THE CHIP  --  the architecture handed to the mapper >>> COLDS THE CACHE <<<
# =============================================================================
#  ACCUMULATOR PRECISION IS NOT STANDARDIZED, ON PURPOSE: Eyeriss v1
#  accumulates at 16b, v2 at 20b, Simba at 24b, each cited. Forcing one width
#  would equalise the architectures rather than the experiment, so
#  ECC_ACC_BITS EMPTY = paper-native = the primary comparison; setting it is a
#  SENSITIVITY STUDY with its own mapper cache and its own results namespace.
#  ECC_FORCE_DATAWIDTH and ECC_FORCE_TECHNOLOGY are the same kind of knob --
#  leave both EMPTY to model each design exactly as its YAML declares it. Note
#  ECC_FORCE_TECHNOLOGY goes through globals.yaml, which costs DRAM for every
#  design at once: it invalidates EVERY cache, not one design's.
#
#  THERE IS NO ECC_WEIGHT_WIDTH, AND NO ECC_WEIGHT_DATAWIDTH. The on-chip
#  quantisation is DERIVED: `q = round(8*K/N)` from the code in play, applied
#  to the storage levels of the ARM being mapped, and each level's `width:`
#  comes from THE WIDTH TABLE in that design's own `archs/<name>/widths.yaml`
#  (96 at q=8, 98 at q=7, 96 at q=6, 95 at q=5, 96 at q=4; the GLB word is the
#  ratio that file's 8-bit row declares, 4x on Eyeriss v1). THE ARMS DO NOT
#  SHARE A DECLARED WIDTH and no arm has to be legal for another arm's
#  datawidth: BCH(63,39) runs at width 95 because 95 % 5 == 0, and 95 % 8 = 7
#  IS IRRELEVANT because the 8-bit arm is never mapped at 95 -- it is mapped at
#  96, where 96 % 8 == 0. timeloop-mapper's `width % (word_bits * block_size)
#  == 0` (buffer.cpp:302, no floor path, exit=134 on a violation) is per level,
#  per mapper run, and one mapper run maps ONE arm.
#     There was an ECC_WEIGHT_WIDTH knob until 2026-09-12 and it caused the
#  exact failure it was meant to prevent; the withdrawn lcm(q,8) scheme
#  (56/24/40) made the 8-BIT REFERENCE ARM MOVE BETWEEN CODES and reported
#  BCH(63,39) as a 37.69% win that was really the reference breaking
#  (FINDINGS 2.4b). ECC_WEIGHT_DATAWIDTH and ECC_WEIGHT_DATAWIDTH_LEVELS went
#  the same way on 2026-09-14 (EnvReorganisation phase 4): the ERT arm IS a
#  datawidth configuration, so `config._resolve()` derives both from it and a
#  second spelling could only put two chips in one cache directory.
#     EACH LEVEL'S DEPTH IS RENORMALISED AT THE BASE WIDTH, not at the arm's
#  own -- depth' = round(depth x width / 96) -- so it holds the published TOTAL
#  BITS and is THE SAME FOR EVERY ARM. CACTI is handed depth and width, so that
#  is the quantity that must not move, and a shared depth is what leaves
#  `assert_pair_geometry()` (section 4) something real to check.
#  `python3 -m eccenergy.physics.widths` prints the RULE's table.
#
#  ECC_MAC_PJ_OVERRIDE IS THE DENOMINATOR OF EVERY ECC PERCENTAGE. An ECC
#  saving is saved_uJ / total_uJ: the saved uJ are weight traffic and do not
#  depend on what a MAC costs, the total does, and on eyeriss_v2_like the MAC
#  is 44% of the run at the ERT's 1.16877 pJ. That ERT number is Accelergy's
#  `intmac` compound from the Library plug-in's ONE 32-bit 40 nm table row per
#  primitive, scaled linearly in operand width and up to 45 nm (FINDINGS 7.3).
#  Horowitz ISSCC 2014 Fig. 1.1.9 puts an int8 multiply at ~0.2 pJ and an int8
#  add at ~0.03 pJ -- 5x less -- and Eyeriss v1 measured its ALUs at <10% of
#  chip power, so if the MAC is 4-6x too dear then EVERY percentage in Tasks
#  1-3 is diluted 1.5-2x, embedded as much as recon. 0.23 IS THE PRIMARY
#  DENOMINATOR by decision (2026-09-09); EMPTY reproduces the ERT-denominator
#  numbers and is now the sensitivity row. The value and its citation
#  (archs/_shared/provenance.yaml `mac_energy_pj`) travel onto every figure,
#  table, manifest and result; a value not listed there is labelled "uncited".
#  It rescales the Compute category in the EVALUATOR, after the raw cache, and
#  the MAC count is mapping-invariant -- so under ECC_OPT_METRIC=energy the
#  optimum does not move and the cache stays warm; under `edp` it can, and the
#  run prints a warning. It is in this section because it is in the supplied
#  ERT and therefore in the fingerprint.
#
#  THE TWO FAIRNESS LEVERS ARE A DIFFERENT DATAFLOW, AND MUST BE LABELLED ONE.
#  ECC_WEIGHT_FACTOR_RELAX=1 drops the `factors:` pins on the WEIGHT-INDEXING
#  dimensions (M, C, R, S) of weight-carrying levels -- and Eyeriss v1's M=1 at
#  the filter spad IS the row-stationary dataflow, so a design run under it is
#  NOT the chip JSSC 2017 describes and `source: published` does not licence
#  the name. N, P and Q keep their pins: weights do not index them, so relaxing
#  those would retile the activations instead. Without it the tile stays pinned
#  and capacity cannot bind at all (`weights held` was EXACTLY 21,504 at x1,
#  x1.6154, x4, x8, x16 and x32, the last at 0.9% fill -- FINDINGS 7.8).
#  ECC_MAPSPACE_CONSTRAIN=1 collapses the index-factorization space from the
#  mapper's own reported ~7.4e10 to ~9.5e4, which is then searched
#  EXHAUSTIVELY. That is the answer to a FAILED convergence gate: at victory
#  4000 vs 10000 the embedded arm's total energy moved 43.7% where the ECC
#  effect is 5.8% and the ORDERING between the arms flipped, because a bigger
#  budget samples more of the same enormous space and the difference between
#  two arbitrary points is noise. Constrained, the gate passes at 0.00%
#  residual across victory 2000/4000/10000 at ~1 min per map (FINDINGS 2.2).
#  WHICH LEVELS EACH DIMENSION MAY SPLIT ACROSS is the design's own
#  `mapspace_free_levels:`, read off the best mapping the search has ever found
#  for it -- constraining around a known-good region is a CHOICE and it is
#  stated: the exhaustive answer is the best mapping IN THIS FAMILY. What keeps
#  it a fair ECC comparison is that BOTH ARMS get the identical constraint.
#  The two COMPOSE and are meant to be run together; both were exported by
#  every prompt_5 and prompt_6 mapping already on disk, so the defaults change
#  no slug and no fingerprint.
#
#  TIME IS IN THE ARCHITECTURE SINCE PHASE C1, which is why ECC_RECON_BW_SCALE
#  and ECC_ONCHIP_BW_BITAWARE are here and not among the prices. The DRAM level
#  declares `shared_bandwidth` -- NOT read_+write_bandwidth, because the DQ bus
#  is ONE wire set whose limit is on their SUM, which is what the roofline
#  charges; each arm's boundary declares
#  `per_dataspace_bandwidth_consumption_scale`, K/N at DRAM (a BIT stream off
#  the die) and q/8 on chip (whole weights at q bits per word) -- the two
#  differ by 5% and one factor everywhere is a silent inconsistency with
#  ECC_RECON_PACKING; a level the arm narrows declares its port x 8/q, because
#  the port moves BITS per cycle; and the design runs at its own `clock_mhz:`
#  through globals_<arch>.yaml. All four are in the patched YAML and therefore
#  in the fingerprint. A NETWORK stage is declared and marked no-op --
#  LegacyNetwork::ComputePerformance() is an empty stub, so there is nothing
#  for the factor to reach (reporting rule R-3) -- but it still separates two
#  boundaries that differ only by a network, which is why mapper_arms() carries
#  it. Setting either to 0 reproduces the pre-Phase-C architecture byte for
#  byte, which is the mutation eccenergy/tests/test_phase_c.py runs.
#     WHY THE BIT-AWARE PORT MATTERS: Timeloop's throughput check counts ITEMS
#  per cycle and a narrow weight is still one item, so narrowing alone is
#  invisible to the clock (prompt_7 Defect 1). `filter_glb`'s declared 16
#  items/cycle is LITERALLY what caps every fully-connected layer at 9.52% PE
#  utilisation -- 16 of 168 PEs, measured on resnet18 `fc` and mobilenet
#  `classifier.1`. Those layers are 0.28% of resnet18's cycles and 2.10% of
#  mobilenet's, so the aggregate CNN effect is ~0.1-1%; on a batch-1
#  transformer every layer is that layer.
#
#  ECC_ENERGY_MODEL_REV IS THE DELIBERATE COLD. The fingerprint hashes the
#  ARCHITECTURE, not the price list Accelergy derives from it -- so fixing an
#  ESTIMATOR changes every energy in the cache while leaving the directory it
#  is stored under identical, and the stale entries are reused with nothing to
#  say so. That is not hypothetical: the Neurosim plug-in answered 0 pJ for
#  every smartbuffer address generator until 2026-09-12, because it crashed
#  writing scratch into a read-only SIF and Accelergy accepted the 0
#  (hpc/tl.sh now binds it a writable copy). Any non-empty value
#  re-fingerprints the whole matrix; EMPTY hashes byte-identically to every
#  fingerprint that predates the knob. Bump it when an estimator changes, not
#  before -- and expect to re-map. Form: a date plus what changed.
#
#  THE INTERCONNECT IS NOT FREE. Timeloop's own wire model is a stub returning
#  0, so ECC_NOC=0 makes every network free -- in the evaluator AND in the
#  mapper's objective. The coefficients and their citations live in
#  archs/_shared/noc.yaml; all four knobs are in the cache slug (`noc`), so a
#  pre-NoC mapping is never read back as a costed one. NoC is its own plotted
#  category, and the two terms Timeloop cannot be given are charged after
#  mapping by toolchain/noc_post.py.

# the protected payload: weight quantization in bits
: "${ECC_WEIGHT_BITS:=8}"

# input activations. Separate from the above on purpose -- `validate` checks it
: "${ECC_ACTIVATION_BITS:=8}"

# EMPTY = paper-native (v1 16b, v2 20b, Simba 24b) | <bits> = a sensitivity run
: "${ECC_ACC_BITS:=}"

# paper | stock         which YAML each design is mapped from
: "${ECC_ARCH_FIDELITY:=paper}"

# EMPTY = each design as declared | 8 = equalise storage datawidth per design
: "${ECC_FORCE_DATAWIDTH:=}"

# EMPTY = each design's own node | 45nm = equalise it, invalidating EVERY cache
: "${ECC_FORCE_TECHNOLOGY:=}"

# 0 | 1     prompt_3 lever 2: free the weight-indexing loop pins (M, C, R, S)
: "${ECC_WEIGHT_FACTOR_RELAX:=1}"

# 0 | 1     prompt_3 lever 3: pin every dimension the design's free-set omits
: "${ECC_MAPSPACE_CONSTRAIN:=1}"

# EMPTY = the ERT's 1.16877 pJ (sensitivity) | 0.23 = Horowitz int8 mul+add, 45nm
: "${ECC_MAC_PJ_OVERRIDE:=0.23}"

# the clock a design whose design.yaml declares no `clock_mhz:` runs at
: "${ECC_GLOBAL_CYCLE_SECONDS:=1e-9}"

# 0 | 1     declare `per_dataspace_bandwidth_consumption_scale` on reduced stages
: "${ECC_RECON_BW_SCALE:=1}"

# 0 | 1     scale a narrowed level's read/write_bandwidth by 8/q
: "${ECC_ONCHIP_BW_BITAWARE:=1}"

# EMPTY = hash as every pre-knob fingerprint did | <date>-<what changed> = COLD
: "${ECC_ENERGY_MODEL_REV:=2026-09-12-neurosim-adders}"

# 0 | 1     charge the interconnect at all (0 makes every network free)
: "${ECC_NOC:=1}"
: "${ECC_NOC_WIRE_PJ_PER_BIT_MM:=}"   # EMPTY = noc.yaml's 45nm wire constant (0.12)
: "${ECC_NOC_ROUTER_PJ:=}"            # EMPTY = noc.yaml's per-flit router energy (0.25)
: "${ECC_NOC_PE_LATCH_PJ:=}"          # EMPTY = noc.yaml's per-PE latch (0.5; bracket 0 / 0.9152)
: "${ECC_NOC_SCALE:=1}"               # multiply every NoC term, for a sensitivity run


# #############################################################################
# ##   END OF THE COLD ZONE. Nothing below re-fingerprints anything: every    ##
# ##   knob from here on is read by the EVALUATOR, after the raw cache, so    ##
# ##   changing one and redrawing is milliseconds --                          ##
# ##       bash run.sh --replot        (ECC_REPLOT_ONLY=1, section 7)         ##
# #############################################################################


# =============================================================================
#  4. THE PRICES  --  how each arm is charged, after the mapping
# =============================================================================
#  EVALUATOR ONLY: everything here is applied to the RAW cache, so changing a
#  value and redrawing is milliseconds (`bash run.sh --replot`). Nothing here
#  re-fingerprints the reference arm.
#
#  WITH FOUR MEASURED EXCEPTIONS, each marked `>>> BUMPED ARMS <<<` on its own
#  line below. The reconstruction datapath's price is carried into the ERT
#  BUMP, which IS hashed, so the two DC tables, ECC_RECON_PJ,
#  ECC_RECON_CLOCK_GATING_PCT and ECC_RECON_ENCODER_GRANULARITY re-fingerprint
#  every arm that HAS a bump -- recon2, recon4 and recon5 on Eyeriss v1 -- and
#  no other arm. Measured 2026-09-14 on eyeriss_like_wglb: the reference stays
#  at fp-2db6a4d92ff5 under all five, recon2 moves off fp-1519e6934e32 under
#  each of them. That is the knob working: the mapper and the evaluator must
#  price ONE engine (prompt_6 RULE 5.3), so a changed price has to reach both.
#  The affected arms re-map by themselves on the next run.
#
#  THE THREE ARMS. One BCH(N,K) codeword over ECC_WEIGHT_BITS-bit weights under
#  all three, so they differ only in WHERE THE PARITY LIVES, and codewords are
#  counted from DRAM weight reads.
#    baseline  parity beside the data in DRAM. It pays a DEARER PER-BIT PRICE,
#              NOT EXTRA TRAFFIC: the decoder is on the DRAM die, so its parity
#              is read, corrected and discarded there and never crosses the
#              datapath. What it pays for is an array that also holds parity.
#    embedded  parity inside the stored weights, laid out as the embedding
#              pipeline does it -- the MSB-first weight bit stream cut into
#              n-bit codewords, so weights straddle codewords and the n-k
#              lowest-significance positions carry parity.
#    recon     DRAM as embedded, K/N of the weights held on chip, the rest
#              regenerated by a synthesized datapath characterised in
#              data/dc/BCH_N63_results.json.
#
#  WHERE THE BCH DECODER SITS IS NOT A KNOB: IT IS ON THE DRAM DIE.
#  01_project_context_and_architectures.txt sections 1 and 4. The decoder is
#  OFF the fetch path -- it corrects at write, on a scrub pass or on a prior
#  access -- so at fetch time only the k message bits of each n-bit codeword
#  are read out and driven off the die, and the WHOLE DRAM weight term falls by
#  K/N on every R bar, R1 included. The two reference bars keep controller-side
#  correction and do not move. The `controller` row and its code path went on
#  2026-09-14 (EnvReorganisation 6.7); `weight_path.DECODE_SITE` is the
#  constant every record still names.
#
#  ECC_RECON_PACKING -- HOW THE REDUCED REPRESENTATION IS PHYSICALLY EXPLOITED.
#  Section 16 of 02_reconstruction_dse_and_implementation.txt: "reducing
#  weights from 8 bits to 4 bits reduces SRAM energy by 50%" is not a claim the
#  hardware supports unless the representation is exploited PHYSICALLY.
#    stream   the retained k bits of each n-bit codeword are stored and moved
#             as a packed field with no per-weight alignment -- the layout the
#             embedding pipeline already produces, since the codeword IS n
#             consecutive bits of the weight bit stream. Values per physical
#             word and operands per flit rise by n/k, so every reduced stage
#             scales by K/N.
#    aligned  each reduced weight occupies ceil(weight_bits*K/N) WHOLE bits and
#             nothing is repacked. Wire energy still falls, but a 24-bit word
#             holds floor(24/7)=3 seven-bit values -- the same 3 it held at 8
#             bits -- so the access count, and the SRAM energy, do not move.
#  ON-CHIP NARROWING HAS EXACTLY ONE OWNER (prompt_6 RULE 1): since 2026-09-10
#  the MAPPER delivers it, because Timeloop bills `vector_access_energy /
#  block_size` with `block_size = width/datawidth` and the arm's derived
#  datawidth already halves per-weight SRAM energy INSIDE the Timeloop number.
#  `stream` would then scale that same saving by K/N a SECOND time in the
#  evaluator and the on-chip saving would be SQUARED.
#  `study.narrowing.assert_onchip_narrowing_once()` STOPS the run if both are
#  live, and WHO NARROWS IS MEASURED per bar per stage: `Word bits == q` in
#  that bar's own stats means the mapper did, `== weight_bits` means the
#  evaluator does, anything else stops. `stream` is safe here because the DRAM
#  and NoC stages it prices are not levels the mapper narrows.
#
#  ECC_RECON_ENCODER_SITE -- WHERE A NETWORK BOUNDARY'S ENCODERS SIT, AND
#  THEREFORE HOW MANY TIMES THEY RUN. Section 7.1 of
#  01_project_context_and_architectures.txt states the tradeoff and asks for it
#  to be an experiment variable:
#      BEFORE MULTICAST   4b -> Encoder -> 8b -+-> PE   (x fanout)
#          one reconstruction at the source, but FULL-WIDTH network traffic
#      AFTER MULTICAST    4b -+-> Encoder -> PE          (x fanout)
#          replicated encoder hardware, but REDUCED-WIDTH shared transport
#  `destination` (the model since 2026-09-09) puts an encoder at each
#  destination, so the count is Timeloop's destination-side ARRIVALS --
#  `Ingresses x Multicast factor`, read off its own printed breakdown. It is
#  the only count consistent with a boundary that ALSO credits that network
#  with carrying the reduced form: an encoder placed before the fanout would
#  make the network full width, which is the boundary above it. `source` is one
#  encoder before the fanout, count = `Ingresses`, kept as a runnable row so
#  the 2026-09-09 change can be diffed -- on eyeriss_like's C512 shape the
#  column network multicasts 7-fold, so the two differ by 7x on R2. Only
#  NETWORK boundaries depend on it.
#
#  ECC_RECON_ENCODER_GRANULARITY -- section 15: the encoder may work at
#  CODEWORD granularity, because rebuilding one weight can need retained bits
#  from several. `weight` charges the synthesized per-codeword energy per
#  n/weight_bits of the weights actually rebuilt -- the amortized reading, the
#  group is rebuilt once and all of it is consumed. `codeword` charges a whole
#  codeword per access whether or not the rest of the group is used -- the
#  pessimistic reading, and the right one if nothing buffers the group.
#  GROUP RESIDENCY IS REPORTED, NOT REFUSED (decided 2026-09-13; the knob went
#  2026-09-14 and the reasoning now sits with the constant in
#  study/placement_eval.py). G_rec = 9 at BCH(63,.) over 8-bit weights, because
#  63/8 = 7.875 is not whole, so a codeword drifts across weight boundaries and
#  the worst-aligned one reaches into ceil(63/8)+1 weights. The old refusal
#  assumed an engine that can only rebuild from weights co-resident AT ONE
#  INSTANT; RECAP's accumulates the retained bits as they arrive, so a level
#  holding 6 -- or 1 -- still feeds it, over more accesses and with more
#  buffering. A small tile is a COST, not an impossibility. The number is still
#  measured and still on every bar's record (`layers_below_G_rec`,
#  `infeasible_layers`, `group_residency_note`) because it bounds the buffer
#  the engine needs.
#
#  THE DRAM PRICE. 20/40/70 pJ/bit = 0.5/1.0/1.75 nJ per 64 b, the same 45 nm
#  table Horowitz ISSCC 2014 reads from for ECC_MAC_PJ_OVERRIDE=0.23, so the
#  two denominators of every percentage share one source. The DRAM term is ONE
#  stage of the weight path and the whole of it is reducible. The baseline's
#  dearer per-bit price is ECC_BASELINE_DRAM_PJ_PER_BIT; both carry their
#  citation from archs/_shared/provenance.yaml onto every figure and result.
#  ECC_DRAM_BACKGROUND_PJ and ECC_DRAM_REFRESH_PJ are the other two terms of
#  E_total(DRAM) = E_dynamic + E_background + E_refresh.
#
#  ECC_LATENCY_MODEL -- the roofline, computed AFTER mapping
#  (toolchain/latency_post.py), the same evaluator-only pattern noc_post.py
#  uses for the two interconnect terms Timeloop cannot be given:
#      cycles = max( compute cycles,
#                    every storage level's own declared-bandwidth limit,
#                    off-chip items / ECC_DRAM_BANDWIDTH_MBPS )
#  It never invokes the mapper and is NOT in the mapping fingerprint: the plan
#  is the one Timeloop already chose and this states how long that plan takes.
#  With ECC_DRAM_BANDWIDTH_MBPS EMPTY (unlimited) it reproduces Timeloop's own
#  per-level AND total cycle counts EXACTLY on all 43 cached shapes, which is
#  what stops it inventing time. WHY IT EXISTS: `datawidth: q` makes energy
#  fall and cycles stand still, because Timeloop's speed model counts ITEMS per
#  cycle and a narrow weight is still one item (prompt_7 Defect 1), so a
#  reported "0.00% latency gain" is an ABSENT TERM and never a result. Off chip
#  the weights are a BIT stream, so there the reduced form really does move K/N
#  of the traffic. DEFAULT 1 SINCE 2026-09-13: C1.1 declares the same limit on
#  the DRAM level, so Timeloop's own cycles already carry it and this re-states
#  the plan's time per bar instead of supplying the only estimate of it. WHO
#  APPLIES THE OFF-CHIP WEIGHT RELIEF IS MEASURED PER BAR
#  (`latency_post.relief_owner`): the mapper if that bar's stats print
#  `Bandwidth Consumption Scale` = K/N, this roofline if they print 1.00, and a
#  third value is REFUSED as a bar billed from another code's plan. Without
#  that rule C1.2 and this knob would each apply K/N and the saving would be
#  K/N SQUARED. It moves no energy on its own; it decides the run length
#  ECC_STATIC_ENERGY is charged over.
#
#  ECC_STATIC_ENERGY -- 1 charges component standby energy (power x TIME, from
#  each design's own `leakage_nw:` densities in archs/<name>/design.yaml) to
#  ALL THREE arms as a `Standby` category IN `Raw.base`, which every arm and
#  every placement bar starts from, so no code path can charge it to one arm
#  and not another. 0 does not charge it and does not even make it a category,
#  which reproduces every pre-Phase-A total to the pJ.
#
#  TRAP -- EVERY PER-CYCLE CONSTANT MUST BE CONVERTED WITH THE SAME PERIOD, or
#  standby energy silently moves by 5x. Two live cases, handled differently:
#    ECC_RECON_IDLE_PJ  is pJ PER CYCLE, measured by DC at a 1 ns clock
#                       (data/dc/BCH_N63_results.json,
#                       measurement.clock_period_ns = 1.0). At any other period
#                       it MUST be rescaled:
#                           idle_pJ_per_cycle(T) = idle_pJ_per_cycle(1ns) x T/1ns
#                       At 200 MHz the BCH(63,30) idle is 2.8310811 x 5 =
#                       14.1554055 pJ/cycle. `Config.dc_idle_scale()` owns the
#                       factor and `study.stacks.load_recon_terms()` applies it
#                       ONCE, after the three lookup branches converge, so no
#                       caller can forget it.
#    leakage_nw         (design.yaml) is POWER in nW, not energy, so it needs
#                       NO rescaling -- energy per cycle = nW x T. That is why
#                       the two are declared in different units.
#  `Config.cycle_seconds_for()` is the ONLY place MHz becomes seconds.
#
#  RECONSTRUCTION IS TWO TERMS ON TWO DENOMINATORS (prompt_6 RULE 3):
#      E_recon = incremental x events + idle_per_cycle x cycles x N_engines
#  `load_recon_energy()` returns them separately and nothing adds them.
#  ECC_RECON_INCLUDE_IDLE, which ADDED a per-codeword number to a per-cycle
#  one, is retired: idle is always charged, on its own denominator, with the
#  cycle count of the plan being billed. The two tables below are the DC
#  numbers keyed by configuration and they are read FIRST -- before
#  ECC_RECON_JSON, the synthesis archive behind them -- so editing a value here
#  is what the run prices the datapath at. BOTH tables must carry the (N,K) for
#  them to win; if either lacks it the pair falls through to the JSON, so the
#  two terms can never come from different sources. Bash cannot export a
#  `declare -A`, so section 8 flattens them into ECC_RECON_INCREMENTAL_PJ_LIST
#  and ECC_RECON_IDLE_PJ_LIST.
#
#  ECC_RECON_CLOCK_GATING_PCT -- THE IDLE NUMBER ABOVE IS A FREE-RUNNING CLOCK.
#  For BCH(63,30) the 2.8310811 pJ/cycle is 2.8164 pJ (99.48%) CLOCK/dynamic
#  power and only 0.0147 pJ (0.52%) true leakage. An engine that is clock-gated
#  when no weight is arriving does not burn the dynamic part:
#      E_recon = (incremental + idle) x events                  <- engine working
#              + idle x (1 - PCT/100) x (engine_cycles - events) <- gated off
#  PCT=0 reproduces the pre-gating model EXACTLY, to the pJ, and is what every
#  pre-gating assertion is pinned to; PCT=99.5 is the measured clock share;
#  PCT=100 is fully power-gated, the optimistic bound. Measured duty cycles are
#  0.62% (recon4) to 14.3% (recon1), so this term is 93.5-99.7% of the
#  reconstruction energy and the knob MOVES THE HEADLINE RESULT. ALWAYS REPORT
#  PCT=0 BESIDE WHATEVER YOU CHOOSE. The ERT bump carries the SAME gated
#  numbers (`incremental + idle x g` per access, `idle x (1 - g)` per cycle), so
#  the mapper and the evaluator price one engine.
#  THE IDLE DENOMINATOR is `StageStats.engine_cycles` = sum over layers of
#  (engines that leak x that layer's cycles): 1 at DRAM; THAT LAYER'S OWN
#  fanout x instances at a network (it used to charge the widest layer's fanout
#  over the whole run, which over-billed mobilenet_v2 by x1.3336 because its
#  depthwise layers broadcast 2-12 wide, not 14); and at a storage level the
#  UTILIZED instances of that layer's plan, because Timeloop power-gates each
#  unused instance and bills `leak x utilized x cycles`.
#
#  ECC_PARITY_GROUPING IS NOT A FLAT N/K. A weight cannot straddle a codeword,
#  so at BCH(63,51) over 8-bit weights only 6 whole weights fit in the 51-bit
#  message field and 3 bits are padding -- 31.25% overhead against the 23.53% a
#  flat N/K model charges. `layer` makes each layer's weight tensor its own
#  codeword stream, so each pays its own tail padding: what a real allocator
#  does, and conservative. `model` is one stream over the whole model and
#  differs by at most one codeword per layer -- nothing on a full model,
#  visible on a single layer.
#
#  ECC_DISABLE_ASSERT_PAIR_GEOMETRY -- `arch.patch.assert_pair_geometry()`
#  checks that the reconstruction arm and the embedded arm declare the SAME
#  LEVELS AT THE SAME DEPTH, and NOTHING ELSE. It does NOT check `width:` and
#  it does NOT check `datawidth:`, because under THE WIDTH TABLE the arms are
#  SUPPOSED to differ there (96 vs 95 vs 98, 8 vs 5 vs 7) -- asserting a shared
#  width is what produced the withdrawn lcm(q,8) scheme, and that check is gone
#  and must not come back. DEPTH is the one thing left to assert and it is
#  worth asserting: a depth difference is real silicon one arm does not have,
#  priced by Accelergy, which is the defect that invalidated the pre-prompt_2
#  sweep (the dilated array cost 1.18-1.46x more per access, so the optimiser
#  had a reason to leave the room unused -- FINDINGS 7.8). 1 prints what
#  differs instead of raising, for a study that varies depth between the arms
#  ON PURPOSE. THE ONE GUARD WITH AN OVERRIDE OUTSIDE `ECC_ALLOW`: it predates
#  the tiers and is left exactly as it was.

# 1 = bill each boundary from ITS OWN chip's plan | 0 = every bar from the reference (Task 3)
: "${ECC_RECON_ERT_AWARE:=1}"

# which arm ONE mapper job solves. Column 6 of the task file, set by hpc/map.sbatch
#   EMPTY | reference = the published 8-bit chip | recon1 .. recon5 = that boundary's chip
: "${ECC_RECON_ERT_ARM:=}"

# stream | aligned        how the reduced representation is physically exploited
: "${ECC_RECON_PACKING:=stream}"

# weight | codeword       >>> BUMPED ARMS <<<  what one encoder event costs
: "${ECC_RECON_ENCODER_GRANULARITY:=weight}"

# destination | source    where a NETWORK boundary's encoders sit
: "${ECC_RECON_ENCODER_SITE:=destination}"

# pJ per bit off the DRAM die, embedded/recon arms       20 | 40 | 70
: "${ECC_DRAM_PJ_PER_BIT:=20}"

# pJ per bit for the BASELINE, whose array also holds the parity
: "${ECC_BASELINE_DRAM_PJ_PER_BIT:=22}"

# the other two terms of E_total(DRAM) = E_dynamic + E_background + E_refresh
: "${ECC_DRAM_BACKGROUND_PJ:=0}"
: "${ECC_DRAM_REFRESH_PJ:=0}"

# the off-chip speed limit, MB/s per 8-bit weight. EMPTY = unlimited
#   IN THE PATCHED YAML SINCE PHASE C1, so changing it COLDS EVERY ARM
: "${ECC_DRAM_BANDWIDTH_MBPS=120}"

# 0 | 1     re-time the chosen mapping with the post-mapping roofline
: "${ECC_LATENCY_MODEL:=1}"

# 0 | 1     charge component standby (leakage x time) as a `Standby` category
: "${ECC_STATIC_ENERGY:=1}"

# 0 = assert the two arms share a DEPTH | 1 = print what differs instead
: "${ECC_DISABLE_ASSERT_PAIR_GEOMETRY:=0}"

# layer | model        where codeword boundaries fall in the baseline's parity
: "${ECC_PARITY_GROUPING:=layer}"

# 1 = message padding crosses the DRAM bus with the parity | 0 = parity only (25.00%)
: "${ECC_PARITY_CHARGE_PADDING:=1}"

# EMPTY = the baseline's codeword (shared geometry) | 8 = the older per-byte model
: "${ECC_EMB_WEIGHTS_PER_CW:=}"

# 0 = decode charged as ZERO for every arm, THE PUBLISHED SETTING (the two
#     numbers below are ESTIMATES, not measurements) | 1 = charge it
: "${ECC_DECODE:=0}"
: "${ECC_DECODE_PJ_BASE:=40.0}"      # pJ per codeword, BCH(63,51) syndrome+Chien
: "${ECC_DECODE_PJ_EMB:=40.0}"       # the same code -> the same decoder
: "${ECC_RECON_CHARGES_DECODE:=1}"   # does recon still detect before rebuilding?

# the synthesis archive behind the two tables below. FreePDK45/OSU gscl45nm,
# 1.1 V, 1 ns clock, pre-layout with a wire-load model, matched on (N,K)
: "${ECC_RECON_JSON:=data/dc/BCH_N63_results.json}"

# >>> BUMPED ARMS <<<  the two DC terms, read BEFORE ECC_RECON_JSON. Edit a
# value here and it is what the run prices the datapath at; both tables must
# carry the (N,K) or the pair falls through to the JSON together.
declare -A ECC_RECON_INCREMENTAL_PJ=(   # pJ per codeword
    [BCH_63_57_t1]=1.6574   [BCH_63_51_t2]=1.8995   [BCH_63_45_t3]=1.6383
    [BCH_63_39_t4]=1.4561   [BCH_63_36_t5]=1.5082   [BCH_63_30_t6]=1.3786 )
declare -A ECC_RECON_IDLE_PJ=(          # pJ per CYCLE per ENGINE
    [BCH_63_57_t1]=1.9359672  [BCH_63_51_t2]=2.2301273  [BCH_63_45_t3]=2.4120856
    [BCH_63_39_t4]=2.7891299  [BCH_63_36_t5]=2.8358254  [BCH_63_30_t6]=2.8310811 )

# EMPTY = the table above | <pJ> = override the INCREMENTAL term only
#   >>> BUMPED ARMS <<<   (idle has no override)
: "${ECC_RECON_PJ:=}"

# used ONLY when the tables have no entry for the (N,K) in play -- the
# BCH(63,51) numbers, so a missing entry degrades to a plausible cost instead
# of crashing. Seeing these in a result's provenance means a code is missing.
: "${ECC_RECON_INCREMENTAL_FALLBACK_PJ:=1.8995}"
: "${ECC_RECON_IDLE_FALLBACK_PJ:=2.2301273}"

# 0 = the ungated model, to the pJ | 99.5 = the measured clock share | 100 = fully power-gated
#   >>> BUMPED ARMS <<<   and it MOVES THE HEADLINE RESULT
: "${ECC_RECON_CLOCK_GATING_PCT:=99.5}"

# 0 | 1     a light SRAM-side code on top of the strong one. Weights pay it
#           under every arm; inputs pay it under the baseline only
: "${ECC_WEAK:=0}"
: "${ECC_WEAK_N:=63}"
: "${ECC_WEAK_K:=57}"

# 0 = THE CONVENTIONAL BASELINE, what Task 1 specifies: external parity is
#     consumed off chip and never written into on-chip weight SRAM/RF
# 1 = reproduces the older figures, and is warned about in the result JSON
: "${ECC_BASELINE_INFLATES_ONCHIP:=0}"

# 0 | 1     split the two on-chip categories into read and write
: "${ECC_SPLIT_READ_WRITE:=0}"

# instances | name     how a storage level is assigned to "Global buffer" vs
#   "On-chip SRAM/RF". `instances` (one instance = shared, replicated = local)
#   is correct for simba_like, whose per-PE buffers are named "...Buffer" and
#   which name matching therefore mislabels. The TOTALS are identical either
#   way; only the split between the two categories moves.
: "${ECC_CLASSIFY:=instances}"


# =============================================================================
#  5. THE CLUSTER  --  SLURM and the container
# =============================================================================
#  hpc/run_all.sh passes these to sbatch ON THE COMMAND LINE, which overrides
#  the #SBATCH header inside hpc/map.sbatch. That header only matters for a
#  bare `sbatch hpc/map.sbatch`.
#
#  ONE SUBMISSION MAPS ONE ARCHITECTURE. hpc/run_all.sh snapshots `archs/` into
#  hpc/.runtime/archpin.<pid>/ at SUBMIT time and exports ECC_ARCH_PIN_DIR to
#  every map job and the dependent eval, so editing `archs/` while an array is
#  in flight is free -- the queued jobs keep mapping the chip you submitted. It
#  snapshots the TASK FILE for the same reason, because map.sbatch resolves its
#  rows when the job RUNS. It prints the pin when it submits; quote that line
#  when a run is questioned. Without it, an edit landing 78 seconds into a
#  282-job array cost the whole run (2026-09-13, efficientnet_b0): the maps
#  solved one geometry and the eval went looking for another.

# 1 = every stage inside the Timeloop+Accelergy container | 0 = the host python3
#   The MAPPER always needs the container; evaluation and plotting do not, but
#   they need pandas/matplotlib/pyyaml, which a login node may lack.
: "${ECC_USE_CONTAINER:=1}"

: "${ECC_ACCOUNT:=rewetz}"

# rewetz-b = the burst QOS: idle cores, low priority, a 4-day limit, ~90
# concurrent jobs. Set ECC_QOS=rewetz for a run that must not be preempted.
: "${ECC_QOS:=rewetz}"

: "${ECC_PARTITION:=hpg-default}"

# THE THREAD COUNT IS IN THE CACHE KEY -- keep this equal to ECC_MAPPER_THREADS
: "${ECC_MAP_CPUS:=${ECC_MAPPER_THREADS}}"
: "${ECC_MAP_MEM:=16gb}"
# a shorter wall request is scheduled sooner. The longest measured BOUNDED task
# was 31 min; an uncapped search is typically 4-10x that. A BUNDLE IS SERIAL,
# so this must cover the whole bundle when ECC_JOBS is set (section 1).
: "${ECC_MAP_TIME:=24:00:00}"
# how many array tasks run at once. 181 investment cores / 18 per task = 10, so
# 9 leaves room for an ondemand session. Above the limit just queues as
# `JobArrayTaskLimit` -- not an error.
: "${ECC_CONCURRENCY:=9}"

# the evaluation job never invokes Timeloop, so it is small
: "${ECC_EVAL_CPUS:=2}"
: "${ECC_EVAL_MEM:=8gb}"
: "${ECC_EVAL_TIME:=02:00:00}"

# the Timeloop+Accelergy image
: "${ECC_SIF:=${ECC_PROJECT_ROOT}/timeloop.sif}"

# the generated task list: ONE ROW PER UNIT, seven columns. GENERATED by
# eccenergy/toolchain/units.py (see `ecc_write_taskfile` in section 8), never
# hand-edited
: "${ECC_TASKFILE:=${ECC_PROJECT_ROOT}/hpc/.runtime/tasks.txt}"

# WHICH COPY OF `archs/` THIS RUN IS. Set by hpc/run_all.sh in --export, not by
# hand. EMPTY (an interactive run) = read `archs/` live. Point it at any
# directory shaped like `archs/` to map that instead -- a snapshot kept from an
# earlier run, for instance. A path that does not exist warns on stderr and
# falls back to `archs/`; it never stops the run. The fingerprint is computed
# from whichever bytes this names, so the cache directory still moves when the
# architecture does.
: "${ECC_ARCH_PIN_DIR:=}"


# =============================================================================
#  6. OUTPUT AND FIGURES
# =============================================================================
#  THE STEM COMES FROM THE CONFIGURATION ALONE, so re-running at different
#  constants REWRITES the file instead of adding one, and the manifest beside
#  it records what produced what is on disk. Copy a figure out, or point
#  ECC_RESULTS_DIR elsewhere, to keep it. ECC_STEM itself is set in section 8
#  from ECC_SWEEP -- one fixed name per axis -- and exporting ECC_STEM="" gives
#  the self-describing name instead, for keeping two panel figures side by side.

: "${ECC_RESULTS_DIR:=results}"
: "${ECC_PALETTE:=house}"        # house | cvd (colourblind-safe Okabe-Ito)
: "${ECC_FORMATS:=png}"          # png | pdf | svg, space-separated
: "${ECC_DPI:=400}"
: "${ECC_NICE_LABELS:=1}"        # 1 -> "Eyeriss v2"; 0 -> "eyeriss_v2_like"
: "${ECC_TITLE_NOTE:=}"          # free text appended to the figure title


# =============================================================================
#  7. MISCELLANEOUS
# =============================================================================

# 0 = results are never silently overwritten | 1 = replace an existing file
: "${ECC_OVERWRITE:=0}"

# 1 = reuse a cached mapping ONLY if its mapping.json sidecar proves it was
#     computed for the CURRENT architecture YAML and mapper settings
# 0 = also accept pre-Task-1 entries, which have no sidecar and can prove
#     nothing. Those results are labelled `legacy` and warned about: use it to
#     reuse old compute, never to publish.
: "${ECC_CACHE_STRICT:=1}"

# free text recorded in every result JSON. Say what the run was for
: "${ECC_RUN_NOTE:=}"

# 1 = rebuild the figure from results/_raw/ alone: no mapper, no container.
#     Milliseconds, and what to use after changing anything in section 4
: "${ECC_REPLOT_ONLY:=0}"

# 1 = use only mappings ALREADY solved and never invoke Timeloop. What `--eval`
#     sets and what the evaluation stage runs under
: "${ECC_FROM_CACHE:=0}"

: "${ECC_PYTHON:=python3}"

# extra detail on stdout while workloads are generated
: "${ECC_VERBOSE:=}"

# ---- workload generation ----------------------------------------------------
# Read ONLY by `python3 -m eccenergy.arch.generate models <name>`, which turns a
# network into the layer list the mapper walks. REGENERATING A WORKLOAD CHANGES
# EVERY SHAPE NAME AND THEREFORE EVERY CACHE ENTRY, so these are not run-time
# knobs: set them for the generation, then leave them alone.
: "${ECC_INPUT_HW:=224}"             # CNN input resolution
: "${ECC_SEQ:=1}"                    # transformer sequence length
: "${ECC_INCLUDE_LM_HEAD:=1}"        # count the LM head matmul
: "${ECC_INCLUDE_EMBEDDING:=0}"      # count the token-embedding table (a lookup,
                                     # not a matmul -- off on purpose)


# =============================================================================
#  8. DERIVED  --  NOT KNOBS. Nothing below here needs editing.
# =============================================================================
#  eccenergy/config.py speaks in terms of one SWEPT list plus two HELD
#  constants. The three lists in section 1 are the editable form of exactly
#  that, and this block translates. Everything is still assigned with `:=`, so
#  an explicit ECC_SWEEP_ARCHS=... in the environment continues to win.

_ecc_first() { set -- ${1:-}; echo "${1:-}"; }
_ecc_count() { set -- ${1:-}; echo "$#"; }

# prompt_6 RULE 3: the two DC tables of section 4, flattened into `key=pJ;`
# scalars because bash cannot export a `declare -A` (config._table reads them).
ECC_RECON_INCREMENTAL_PJ_LIST=""
for _k in "${!ECC_RECON_INCREMENTAL_PJ[@]}"; do
    ECC_RECON_INCREMENTAL_PJ_LIST="${ECC_RECON_INCREMENTAL_PJ_LIST}${_k}=${ECC_RECON_INCREMENTAL_PJ[${_k}]};"
done
ECC_RECON_IDLE_PJ_LIST=""
for _k in "${!ECC_RECON_IDLE_PJ[@]}"; do
    ECC_RECON_IDLE_PJ_LIST="${ECC_RECON_IDLE_PJ_LIST}${_k}=${ECC_RECON_IDLE_PJ[${_k}]};"
done

: "${ECC_SWEEP_ARCHS:=${ECC_ARCHS}}"
: "${ECC_SWEEP_MODELS:=${ECC_MODELS}}"
: "${ECC_SWEEP_KS:=${ECC_KS}}"
: "${ECC_CONST_ARCH:=$(_ecc_first "${ECC_ARCHS}")}"
: "${ECC_CONST_MODEL:=$(_ecc_first "${ECC_MODELS}")}"
: "${ECC_CONST_K:=$(_ecc_first "${ECC_KS}")}"

# ECC_SWEEP=fix AND ECC_SWEEP=area HOLD ALL THREE LISTS, and `fix` IS the
# reconstruction placement study (EnvReorganisation phase 3, 2026-09-14). The
# architecture, the model and the code are held at the FIRST entry of their
# list -- the rule every held axis has always followed -- and the x axis is
# WHERE the boundary sits. `area` holds the same three and walks the
# buffer-depth ladder of section 1 instead.
#
# ...AND THE SIX ABOVE ARE `:=`, SO THIS COLLAPSE IS A BARE `=`. A value left
# over in the shell must not silently widen a study that only makes sense at
# one point. That was got wrong once the other way round: a deleted section
# used to hard-assign ECC_ARCHS/ECC_MODELS/ECC_KS while the names DERIVED from
# them here stayed `:=`, so a leftover survived in the derived name and the two
# disagreed without saying so --
#
#     source ./env.sh                                  # ECC_CONST_MODEL=resnet18
#     ECC_RECON_MODEL=convnext_tiny bash hpc/tl.sh ...  # ECC_MODELS=convnext_tiny
#                                                      # ECC_CONST_MODEL=resnet18 (!)
#
# and `config.load_config()` reads the CONST name on a held axis, so that
# mapped resnet18 while every banner said convnext_tiny (found 2026-09-13,
# building hpc/smoke_models.sh). The LISTS are no longer rewritten at all --
# hpc/run_all.sh needs them to enumerate chips -- so only the derived names
# are pinned, and there is nothing left for the two to disagree about.
case "${ECC_SWEEP}" in
    fix|fixed|point|area|depth|depths|depthsweep|areasweep) ECC_POINT_SWEEP=1 ;;
    *)                                                      ECC_POINT_SWEEP=0 ;;
esac
if [ "${ECC_POINT_SWEEP}" = "1" ]; then
    ECC_SWEEP_ARCHS="$(_ecc_first "${ECC_ARCHS}")"
    ECC_SWEEP_MODELS="$(_ecc_first "${ECC_MODELS}")"
    ECC_SWEEP_KS="$(_ecc_first "${ECC_KS}")"
    ECC_CONST_ARCH="$(_ecc_first "${ECC_ARCHS}")"
    ECC_CONST_MODEL="$(_ecc_first "${ECC_MODELS}")"
    ECC_CONST_K="$(_ecc_first "${ECC_KS}")"
    # THE DEFAULT IS THE PLACEMENT STUDY, AND IT IS A DEFAULT AGAIN SINCE
    # EnvReorganisation PHASE 6. It was a bare `=` (the environment could not
    # win) because `report/sweep.py` had NO renderer for an axis that holds all
    # three lists, and routing to a renderer that raises KeyError is worse than
    # ignoring the knob. Phase 6 gave it one -- a point is one group, its bars
    # are the placements -- so `ECC_EXPERIMENT=sweep ECC_SWEEP=fix` is now a
    # legal diff of the same numbers through the sweep renderer, and the file's
    # own rule (`${VAR:=default}`, the environment wins) holds here too. The
    # three LISTS above stay a bare `=`: a leftover there would WIDEN a study
    # that only makes sense at one point, which is a different mistake.
    # `ECC_SWEEP=area` still has no renderer and `sweep-has-no-figure` still
    # refuses it.
    : "${ECC_EXPERIMENT:=recon}"
    ECC_PANEL_MODELS=""
fi

# More than one model on an architecture sweep is the PANEL layout: one panel
# per model, top to bottom, the same x axis repeated inside each. It adds no
# axis and no renderer -- it answers the one question a single sweep cannot,
# whether the architecture ranking survives changing the network.
if [ "${ECC_POINT_SWEEP}" = "1" ]; then
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
#
# THE PLACEMENT STUDY OWNS ITS OWN NAME, on a layer-scoped run too (prompt_6
# 9): its bars come from a mapping solved against the reduced weight width, so
# they are not comparable with a fixed-mapping figure and must never overwrite
# it. ONE path PER MODEL --
# results/figures/ReconSweep_optimiser__<model>.png -- EVEN on a layer-scoped
# run: the study is a few layers by design, the scope lands in the manifest
# and the figure title rather than the filename, and an existing file at that
# path is overwritten on purpose. The model suffix is there since 2026-09-11,
# when the study ran on two networks: without it the second model's eval
# overwrote the first's figure.
if [ "${ECC_POINT_SWEEP}" = "1" ]; then
    : "${ECC_STEM=ReconSweep_optimiser__${ECC_CONST_MODEL}}"
elif [ -z "${ECC_LAYERS}" ]; then
    case "${ECC_SWEEP}" in
        arch|archs|architecture*) : "${ECC_STEM=ArchitectureSweep}" ;;
        model*)                   : "${ECC_STEM=ModelSweep}" ;;
        bch|k|code*)              : "${ECC_STEM=BCHsweep}" ;;
    esac
fi
: "${ECC_STEM=}"

# THE TASK FILE -- ONE ROW PER UNIT OF MAPPER WORK, seven columns:
#
#     <bundle> <arch> <model> <K> <depth> <arm> <layer>
#
# A UNIT is one CHIP x one distinct layer SHAPE, and a CHIP is (architecture,
# code, buffer depth, arm) -- every distinct thing the mapper is handed, with
# its own `arch_fingerprint()` and its own cache directory. `<layer>` is a
# representative layer of the shape and `<bundle>` is the SLURM array index
# that walks this row (ECC_JOBS, section 1): with ECC_JOBS empty every row
# gets its own bundle, which is one array task per unit, exactly as it was
# before EnvReorganisation phase 3.
#
# GENERATED by `python3 -m eccenergy.toolchain.units`, never hand-edited: the
# arms are DERIVED (`arch.arms.mapper_arms()` -- the reference plus every
# boundary that is a distinct chip) and the shapes come out of the workload
# file, so neither can be listed in a shell script without drifting. This
# function is the FALLBACK for a bare `sbatch hpc/map.sbatch` with no task
# file: it writes the reference arm of the held point, which is the one unit
# a bare sbatch could have meant.
ecc_write_taskfile() {
    local out="${1:-${ECC_TASKFILE}}"
    mkdir -p "$(dirname "${out}")"
    if ${ECC_PYTHON:-python3} -m eccenergy.toolchain.units --tasks > "${out}.$$" 2>/dev/null \
       && [ -s "${out}.$$" ]; then
        mv "${out}.$$" "${out}"
        return 0
    fi
    rm -f "${out}.$$"
    echo "env.sh: ecc_write_taskfile: the enumerator did not run (no container?)," >&2
    echo "  falling back to the reference arm of the held point" >&2
    printf '0 %s %s %s %s reference %s\n' \
        "$(_ecc_first "${ECC_ARCHS}")" "$(_ecc_first "${ECC_MODELS}")" \
        "$(_ecc_first "${ECC_KS}")" "${ECC_WEIGHT_DEPTH_SCALE}" \
        "$(_ecc_first "${ECC_LAYERS}")" > "${out}"
}

# ECC_WEIGHT_CAPACITY_SCALE, ECC_WEIGHT_CAPACITY_SCOPE, ECC_WEIGHT_DATAWIDTH
# and ECC_WEIGHT_DATAWIDTH_LEVELS ARE NOT EXPORTED, because they are no longer
# read (EnvReorganisation phase 4, 2026-09-14). The first pair went with Task
# 4's capacity dilation (9.5) and `settings/arch.py` pins the constants at the
# declared design; the second pair is DERIVED FROM THE ARM by
# `config._resolve()`, which is why the two tier-4 guards that refused the knob
# beside an arm went with them. All four remain FIELDS of the configuration, so
# every key and value of the result record is exactly what it was.
export ECC_ACC_BITS ECC_ACCOUNT ECC_ACTIVATION_BITS ECC_ALLOW ECC_APPROACHES \
       ECC_ARCH_FIDELITY ECC_ARCH_PIN_DIR ECC_ARCHS \
       ECC_BASELINE_DRAM_PJ_PER_BIT ECC_BASELINE_INFLATES_ONCHIP \
       ECC_CACHE_STRICT ECC_CLASSIFY ECC_CODE_N ECC_CONCURRENCY \
       ECC_CONST_ARCH ECC_CONST_K ECC_CONST_MODEL ECC_DECODE \
       ECC_DECODE_PJ_BASE ECC_DECODE_PJ_EMB ECC_DEPTH_SWEEP_SCALES \
       ECC_DISABLE_ASSERT_PAIR_GEOMETRY ECC_DPI ECC_DRAM_BACKGROUND_PJ \
       ECC_DRAM_BANDWIDTH_MBPS ECC_DRAM_PJ_PER_BIT ECC_DRAM_REFRESH_PJ \
       ECC_EMB_WEIGHTS_PER_CW ECC_ENERGY_MODEL_REV ECC_EVAL_CPUS \
       ECC_EVAL_MEM ECC_EVAL_TIME ECC_EXPERIMENT ECC_FORCE_DATAWIDTH \
       ECC_FORCE_TECHNOLOGY ECC_FORMATS ECC_FROM_CACHE \
       ECC_GLOBAL_CYCLE_SECONDS ECC_INCLUDE_EMBEDDING ECC_INCLUDE_LM_HEAD \
       ECC_INPUT_HW ECC_JOBS ECC_KS ECC_LATENCY_MODEL ECC_LAYERS \
       ECC_MAC_PJ_OVERRIDE ECC_MAP_CPUS ECC_MAP_MEM ECC_MAPPER_ALGORITHM \
       ECC_MAPPER_MAX_PERMUTATIONS ECC_MAPPER_SEARCH_SIZE ECC_MAPPER_SEED \
       ECC_MAPPER_THREADS ECC_MAPPER_TIMEOUT ECC_MAPSPACE_CONSTRAIN \
       ECC_MAP_TIME ECC_METRICS ECC_MODELS ECC_NICE_LABELS ECC_NOC \
       ECC_NOC_PE_LATCH_PJ \
       ECC_NOC_ROUTER_PJ ECC_NOC_SCALE ECC_NOC_WIRE_PJ_PER_BIT_MM \
       ECC_ONCHIP_BW_BITAWARE ECC_OPT_METRIC ECC_OVERWRITE ECC_PALETTE \
       ECC_PANEL_MODELS ECC_PARITY_CHARGE_PADDING ECC_PARITY_GROUPING \
       ECC_PARTITION ECC_POINT_SWEEP ECC_PROJECT_ROOT ECC_PYTHON ECC_QOS \
       ECC_RECON_BW_SCALE ECC_RECON_CHARGES_DECODE \
       ECC_RECON_CLOCK_GATING_PCT ECC_RECON_DEFAULT \
       ECC_RECON_ENCODER_GRANULARITY \
       ECC_RECON_ENCODER_SITE ECC_RECON_ERT_ARM ECC_RECON_ERT_AWARE \
       ECC_RECON_IDLE_FALLBACK_PJ ECC_RECON_IDLE_PJ_LIST \
       ECC_RECON_INCREMENTAL_FALLBACK_PJ ECC_RECON_INCREMENTAL_PJ_LIST \
       ECC_RECON_JSON ECC_RECON_PACKING ECC_RECON_PJ ECC_REPLOT_ONLY \
       ECC_RERUN_OPTIMISER ECC_RESULTS_DIR ECC_RUN_NOTE ECC_SEQ ECC_SIF \
       ECC_SCOPE ECC_SPLIT_READ_WRITE ECC_STATIC_ENERGY ECC_STEM ECC_SWEEP \
       ECC_SWEEP_ARCHS ECC_SWEEP_KS ECC_SWEEP_MODELS ECC_TASKFILE \
       ECC_TITLE_NOTE ECC_USE_CONTAINER ECC_VERBOSE ECC_VICTORY \
       ECC_VICTORY_SCALING ECC_WEAK ECC_WEAK_K ECC_WEAK_N ECC_WEIGHT_BITS \
       ECC_WEIGHT_DEPTH_LEVELS ECC_WEIGHT_DEPTH_SCALE \
       ECC_WEIGHT_FACTOR_RELAX
