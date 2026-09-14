#!/usr/bin/env bash
# =============================================================================
#  hpc/run_all.sh  --  THE ONE COMMAND, AND SINCE 2026-09-14 THE ONLY LAUNCHER
# =============================================================================
#
#      cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
#      module load apptainer
#      bash hpc/run_all.sh                # map -> evaluate -> plot
#      bash hpc/run_all.sh --dry-run      # the BILL: chips, shapes, cached, jobs
#
#  Everything it runs comes from ../env.sh: which arms, which axis, which
#  architectures, models and codes, how hard the mapper searches, what the
#  cluster is asked for. There are no knobs in this file.
#
#  FOUR LINES OF env.sh DECIDE THE STUDY (EnvReorganisation phase 3):
#      ECC_APPROACHES   which BARS    baseline embedded recon | recon1 .. recon5
#      ECC_SWEEP        which X AXIS  bch | model | arch | fix | area
#      ECC_ARCHS / ECC_MODELS / ECC_KS / ECC_DEPTH_SWEEP_SCALES   the lists
#      ECC_LAYERS       which layers  empty = whole model; `model=a b;` per model
#  plus ECC_JOBS, which bundles the work into fewer SLURM jobs.
#
#  IT USED TO BE THREE LAUNCHERS, because they enumerated CHIPS along different
#  axes: this script over (architecture x model), `map_ert_arms.sh` over
#  (arm x shape) and `map_depth_sweep.sh` over (depth x shape). All three are
#  deleted (git has them; last present at 608982b). There is ONE enumerator
#  now -- `eccenergy/toolchain/units.py` -- and one unit of work:
#
#      A CHIP  = (architecture, code K, buffer-depth scale, arm)
#                every distinct thing the mapper is handed, with its own
#                arch_fingerprint() and its own cache directory
#      A UNIT  = one chip x one distinct layer SHAPE
#                one Timeloop search, one cache entry, one task's work
#
#  WHAT IT DOES
#    1. writes the task list -- one row per UNIT, seven columns
#       (bundle, arch, model, K, depth, arm, layer) -- and SKIPS every unit the
#       mapper cache already holds, which is `Mapper._accept_cached`'s own
#       answer and not a file count
#    2. submits hpc/map.sbatch as a SLURM array over the BUNDLES; each task
#       walks its bundle's rows and exports the row's axes per unit
#    3. submits ITSELF (`--eval-only`) with --dependency=afterok on that array,
#       so the evaluation and the figures happen by themselves the moment the
#       last mapping lands
#  It returns at once; watch with `squeue -u $USER`. Runs are resumable: rerun
#  after a failure and every solved shape is a cache hit -- and is not even
#  submitted the second time.
#
#  ONE-OFF OVERRIDES need no edit anywhere, because env.sh lets the environment
#  win:
#      ECC_MODELS="resnet18 resnet50 densenet121" bash hpc/run_all.sh
#      ECC_SWEEP=bch bash hpc/run_all.sh                 # the code axis
#      ECC_APPROACHES="baseline embedded recon2" bash hpc/run_all.sh
#      ECC_MAP_TIME=08:00:00 ECC_JOBS=12 bash hpc/run_all.sh
#
#  MODES
#      (none)        map, then evaluate and plot when the mapping succeeds
#      --map-only    submit the mapping array and stop
#      --eval-only   evaluate and plot from the mapper cache, right here. Never
#                    invokes Timeloop, so it is safe on a login node. This is
#                    also what the dependent SLURM job runs.
#      --replot      redraw the figure from results/_raw/ alone -- no mapper,
#                    no evaluation, milliseconds. Use after changing anything in
#                    section 6 of env.sh.
#      --local       run the mapping HERE instead of submitting it, then
#                    evaluate and plot. Only inside an allocation (srun), never
#                    on a login node.
#      --dry-run     print the bill and the bundling. Submits nothing.
#                    IT CANNOT SEE THE CODE A JOB RUNS: it resolves the same
#                    configuration and reads the same cache, but only a real
#                    job proves the job works.
#      --smoke       the answer to that: submit TWO real one-row jobs -- one
#                    unit the cache already holds and one it does not -- and
#                    print what to watch. No knobs. A cached unit returns
#                    before `Mapper._map_now` and proves only the CACHE path,
#                    which is how a broken cold-map import survived three
#                    phases of "one real job before the matrix" (phase 6
#                    session 1). Both halves, or neither.
#
#  ECC_SWEEP=area MAPS THE LADDER AND SUBMITS NO EVALUATION, deliberately: the
#  depth sweep is a property of the MAPPINGS, and an eval job over it would
#  draw one placement figure per depth on top of the last. Read it with
#      bash hpc/tl.sh python3 -m eccenergy.report.dilation_view --levels \
#           --csv results/tables/EyerissV1_mem_arch_sweep.csv
# =============================================================================

set -euo pipefail

# FIND THE PROJECT ROOT. Not derivable from ${BASH_SOURCE[0]} alone: when this
# script is submitted with sbatch (which is how the dependent eval job runs),
# SLURM executes a COPY at /var/spool/slurmd/job<id>/slurm_script, so the script
# location points outside the project entirely. Try, in order: this file's own
# location (interactive use), ECC_PROJECT_ROOT inherited from the submitting
# shell (env.sh exports it, sbatch passes it through), and the submit directory.
_ecc_root=""
for _cand in "$(dirname "${BASH_SOURCE[0]}")/.." "${ECC_PROJECT_ROOT:-}" "${SLURM_SUBMIT_DIR:-}"; do
    if [ -n "${_cand}" ] && [ -f "${_cand}/env.sh" ]; then
        _ecc_root="$(cd "${_cand}" && pwd)"
        break
    fi
done
[ -n "${_ecc_root}" ] || {
    echo "hpc/run_all.sh: cannot locate the project root -- no env.sh found via" >&2
    echo "  the script location, ECC_PROJECT_ROOT or SLURM_SUBMIT_DIR" >&2
    exit 2; }
cd "${_ecc_root}"
unset _ecc_root _cand
source ./env.sh

# The header comment above IS the help text: print every comment line after the
# shebang and stop at the first line of code, so the two can never drift.
usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' \
        "${BASH_SOURCE[0]}"
}

MODE=all
while [ $# -gt 0 ]; do
    case "$1" in
        --map-only)  MODE=map ;;
        --eval-only) MODE=eval ;;
        --replot)    MODE=replot ;;
        --local)     MODE=local ;;
        --dry-run)   MODE=dry ;;
        --smoke)     MODE=smoke ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "hpc/run_all.sh: unknown option '$1'" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

# ECC_SWEEP=area's ladder is a shell list (env.sh section 1) and not a field of
# the resolved configuration, so it travels to the enumerator as an argument.
# Every other axis is already in the configuration.
DEPTHS=""
[ "${ECC_SWEEP}" = "area" ] && DEPTHS="${ECC_DEPTH_SWEEP_SCALES}"

# --------------------------------------------------------------------------
#  running one stage
# --------------------------------------------------------------------------
# The mapper always needs the container; evaluation and plotting need only
# python3 with pandas/matplotlib/pyyaml, which a login node may not have --
# hence ECC_USE_CONTAINER, not a hard-coded wrapper.
_run() {
    if [ "${ECC_USE_CONTAINER}" = "1" ]; then
        bash hpc/tl.sh bash run.sh "$@"
    else
        bash run.sh "$@"
    fi
}

# The enumerator. It needs the package, so it goes through the same wrapper.
_units() {
    if [ "${ECC_USE_CONTAINER}" = "1" ]; then
        bash hpc/tl.sh python3 -m eccenergy.toolchain.units "$@"
    else
        "${ECC_PYTHON:-python3}" -m eccenergy.toolchain.units "$@"
    fi
}

# The figure. `panels` when env.sh resolved more than one model on an
# architecture sweep, one bar group per swept value otherwise. $1 is --eval
# (read the mapper cache) or --replot (read results/_raw/ only).
_plot() {
    if [ "${ECC_EXPERIMENT}" = "recon" ]; then
        # The reconstruction placement study evaluates and draws in ONE command
        # -- its bars ARE its result file -- so after stage_eval has run it a
        # separate plot stage would repeat the whole thing. In --replot mode
        # there is no stage_eval, so this IS the run.
        if [ "$1" = "--replot" ]; then
            echo "############ plot: reconstruction placement ############"
            _run recon --replot
        fi
        return 0
    fi
    echo "############ plot: ${ECC_EXPERIMENT} on the ${ECC_SWEEP} axis ############"
    if [ "${ECC_EXPERIMENT}" = "panels" ]; then
        _run panels "$1"
    else
        _run --sweep "${ECC_SWEEP}" "$1"
    fi
}

# WHICH MODELS THE EVALUATION WALKS. A point sweep (fix, area) holds the model
# at the first entry of ECC_MODELS, exactly as it holds the architecture and
# the code, so evaluating ECC_MODELS' whole list there would draw the held
# study twice and overwrite its own figure.
_eval_models() {
    if [ "${ECC_POINT_SWEEP}" = "1" ]; then
        echo "${ECC_CONST_MODEL}"
    else
        echo "${ECC_MODELS}"
    fi
}

# Evaluate every model, then draw. Everything here is --eval: mappings are read
# from the cache and Timeloop is never invoked, so this cannot change a mapping.
stage_eval() {
    # --eval forbids mapping outright, so a re-map request cannot survive into
    # this stage; it belongs to the mapping stage alone.
    export ECC_RERUN_OPTIMISER=0
    # WHICH evaluations are written is DERIVED (ECC_EVAL_EXPERIMENTS went on
    # 2026-09-14): the placement study when the axis routed the run to it --
    # its file holds Task 1's and Task 2's bars itself -- and Task 1 + Task 2
    # otherwise.
    local evals="baseline embedded" m x
    [ "${ECC_EXPERIMENT}" = "recon" ] && evals="recon"
    for m in $(_eval_models); do
        for x in ${evals}; do
            echo "############ evaluate: ${x} -- ${m} ############"
            ( export ECC_CONST_MODEL="${m}"; _run "${x}" --eval )
        done
    done
    _plot --eval
    echo
    echo "figures : ${ECC_RESULTS_DIR}/figures/"
    echo "tables  : ${ECC_RESULTS_DIR}/tables/"
    echo "results : ${ECC_RESULTS_DIR}/evaluation/{Pre|Post}/<arch>/<model>/...   (phase per arm)"
}

# Map every UNIT in this process, one at a time, in task-file order. For an
# interactive allocation; the SLURM array below is the real path.
stage_map_local() {
    # PIN archs/ HERE TOO. --local maps in THIS process over minutes or hours,
    # so it is exposed to the same mid-flight edit as an array is, and it was
    # the only mapping path that did not pin (FINDINGS 6, defect 5: a
    # design.yaml was edited while a --local run was in flight). Before
    # write_taskfile, exactly as submit_map and submit_smoke do it, so the
    # units are enumerated from the same copy the maps are solved against.
    # Not silenced: `local` runs banner() BEFORE this stage, so pin_archs' own
    # line is the only thing that tells the user where the pin is.
    pin_archs
    write_taskfile
    local line b a m k d arm l
    while read -r b a m k d arm l; do
        [ -n "${b:-}" ] || continue
        echo "############ map: ${a} / ${m} / K${k} / x${d} / ${arm} / ${l} ############"
        ( unit_env "${a}" "${m}" "${k}" "${d}" "${arm}" "${l}"; _run map )
    done < <(grep -ve '^[[:space:]]*$' "${ECC_TASKFILE}")
}

# --------------------------------------------------------------------------
#  the task file
# --------------------------------------------------------------------------
# THE ONE PLACE A UNIT'S AXES ARE TURNED INTO ENVIRONMENT. hpc/map.sbatch has
# the same list, and `toolchain.units._chip_cfg` builds the same configuration
# in python -- if the three drift, the launcher skips units a job would
# re-solve, or submits ones it already has.
unit_env() {
    export ECC_SWEEP=arch
    export ECC_SWEEP_ARCHS="$1"  ECC_CONST_ARCH="$1"
    export ECC_CONST_MODEL="$2"  ECC_SWEEP_MODELS="$2"  ECC_PANEL_MODELS=""
    export ECC_CONST_K="$3"      ECC_SWEEP_KS="$3"
    export ECC_WEIGHT_DEPTH_SCALE="$4"
    export ECC_RECON_ERT_ARM="$5"
    export ECC_LAYERS="$6"
}

write_taskfile() {
    mkdir -p "$(dirname "${ECC_TASKFILE}")"
    local tmp="${ECC_TASKFILE}.$$"
    _units --tasks ${DEPTHS:+--depths "${DEPTHS}"} ${ECC_JOBS:+--jobs "${ECC_JOBS}"} \
        > "${tmp}"
    mv "${tmp}" "${ECC_TASKFILE}"
}

# --------------------------------------------------------------------------
#  submitting
# --------------------------------------------------------------------------
# THE ARCH PIN -- see eccenergy/paths.py. `archs/` is snapshotted at SUBMISSION
# and both the map array and the dependent eval read the copy, so editing a
# declared value while the array is in flight cannot split one run across two
# architectures.
#
# ONLY THE TOP-LEVEL SUBMITTER PINS. `submit_eval` re-invokes this script as
# `hpc/run_all.sh --eval-only` inside the eval job, where ECC_ARCH_PIN_DIR is
# already set by --export=ALL: taking a second snapshot there would read the
# LIVE archs/ and undo the whole point.
#
# WHAT THIS COST BEFORE IT EXISTED (2026-09-13, efficientnet_b0). A psum_spad
# depth was changed 78 seconds after an array started. The 282 maps solved the
# pre-edit chip; the eval, twelve minutes later, read the edited file, computed
# a different fingerprint, found it empty and reported all 82 layers as
# `mapper failed`. Both halves were correct and they described two chips.
pin_archs() {
    [ -z "${ECC_ARCH_PIN_DIR}" ] || return 0      # already pinned by our submitter
    local pin="hpc/.runtime/archpin.$$"
    mkdir -p hpc/.runtime
    # Each pin is ~320 kB and is the record of what a run mapped, so it is kept
    # for as long as anything could still want it -- but not forever. A week is
    # far past ECC_MAP_TIME, so nothing queued can still be reading one.
    find hpc/.runtime -maxdepth 1 -name 'archpin.*' -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true
    rm -rf "${pin}"
    cp -a archs "${pin}"
    export ECC_ARCH_PIN_DIR="${PWD}/${pin}"
    echo " arch pin   : ${ECC_ARCH_PIN_DIR}"
    echo "              (this run maps THIS copy; edit archs/ freely from now on)"
}

# THE TASK FILE IS SNAPSHOTTED PER SUBMISSION, because hpc/map.sbatch resolves
# its rows when the job RUNS: a later launcher writing ECC_TASKFILE would
# otherwise repoint every queued job at different work.
submit_map() {
    pin_archs >/dev/null
    write_taskfile
    local snap n
    snap="hpc/.runtime/tasks.$$.txt"
    cp "${ECC_TASKFILE}" "${snap}"
    n=$(awk 'NF {print $1}' "${snap}" | sort -un | wc -l)
    [ "${n}" -gt 0 ] || {
        echo "hpc/run_all.sh: nothing to map -- every unit is already in the" >&2
        echo "  mapper cache. Evaluate it with: bash hpc/run_all.sh --eval-only" >&2
        echo "  (or re-solve with ECC_RERUN_OPTIMISER=1)" >&2
        exit 2; }
    mkdir -p hpc/logs
    sbatch --parsable \
        --job-name=ecc-map \
        --account="${ECC_ACCOUNT}" --qos="${ECC_QOS}" --partition="${ECC_PARTITION}" \
        --cpus-per-task="${ECC_MAP_CPUS}" --mem="${ECC_MAP_MEM}" \
        --time="${ECC_MAP_TIME}" \
        --array="0-$((n - 1))%${ECC_CONCURRENCY}" \
        --output="hpc/logs/ecc-map.%A_%a.out" \
        --export=ALL,ECC_TASKFILE="${PWD}/${snap}",ECC_ARCH_PIN_DIR="${ECC_ARCH_PIN_DIR}" \
        hpc/map.sbatch
}

# TWO REAL JOBS, ONE CACHED UNIT AND ONE COLD ONE. It is `submit_map` with a
# two-row task file: the same sbatch, the same map.sbatch, the same arch pin,
# so what it proves is what a matrix would do and not a special path.
submit_smoke() {
    pin_archs >/dev/null
    mkdir -p "$(dirname "${ECC_TASKFILE}")" hpc/logs
    local snap tmp n
    snap="hpc/.runtime/smoke.$$.txt"
    tmp="${snap}.part"
    _units --tasks --smoke > "${tmp}"
    mv "${tmp}" "${snap}"
    n=$(awk 'NF {print $1}' "${snap}" | sort -un | wc -l)
    [ "${n}" -gt 0 ] || {
        echo "hpc/run_all.sh --smoke: this configuration enumerates no units" >&2
        exit 2; }
    sbatch --parsable \
        --job-name=ecc-smoke \
        --account="${ECC_ACCOUNT}" --qos="${ECC_QOS}" --partition="${ECC_PARTITION}" \
        --cpus-per-task="${ECC_MAP_CPUS}" --mem="${ECC_MAP_MEM}" \
        --time="${ECC_MAP_TIME}" \
        --array="0-$((n - 1))" \
        --output="hpc/logs/ecc-smoke.%A_%a.out" \
        --export=ALL,ECC_TASKFILE="${PWD}/${snap}",ECC_ARCH_PIN_DIR="${ECC_ARCH_PIN_DIR}" \
        hpc/map.sbatch
}

submit_eval() {
    local dep=()
    [ -n "${1:-}" ] && dep=(--dependency="afterok:$1")
    mkdir -p hpc/logs
    sbatch --parsable ${dep[@]+"${dep[@]}"} \
        --job-name=ecc-eval \
        --account="${ECC_ACCOUNT}" --qos="${ECC_QOS}" --partition="${ECC_PARTITION}" \
        --cpus-per-task="${ECC_EVAL_CPUS}" --mem="${ECC_EVAL_MEM}" \
        --time="${ECC_EVAL_TIME}" \
        --output="hpc/logs/ecc-eval.%j.out" \
        hpc/run_all.sh --eval-only
}

banner() {
    echo "=============================================================================="
    if [ "${ECC_EXPERIMENT}" = "recon" ]; then
        echo " STUDY      : reconstruction PLACEMENT -- the x axis is WHERE the"
        echo "              boundary sits. ECC_SWEEP=${ECC_SWEEP} holds the"
        echo "              architecture, the model and the code at the FIRST entry"
        echo "              of their list; ECC_APPROACHES names the bars."
        echo " remapping  : every placement that is a DISTINCT CHIP is mapped on its"
        echo "              own plan (ert-aware=${ECC_RECON_ERT_AWARE}); a bar whose"
        echo "              chip is cold is billed from a NAMED plan and says so"
        echo "              on its record -> ${ECC_STEM:-ReconSweep_optimiser}"
    fi
    echo " axis       : ${ECC_SWEEP}"
    echo " archs      : ${ECC_ARCHS}"
    echo " models     : ${ECC_MODELS}"
    echo " code       : BCH(${ECC_CODE_N}, ${ECC_KS})   arms: ${ECC_APPROACHES}"
    echo " scope      : ${ECC_LAYERS:-whole model}"
    echo " mapper     : ${ECC_MAPPER_ALGORITHM} victory=${ECC_VICTORY}/${ECC_VICTORY_SCALING}"\
         "search=${ECC_MAPPER_SEARCH_SIZE:-uncapped} threads=${ECC_MAPPER_THREADS}"
    echo " figure     : ${ECC_RESULTS_DIR}/figures/${ECC_STEM:-<self-describing>}"
    [ -n "${ECC_ARCH_PIN_DIR}" ] && echo " arch pin   : ${ECC_ARCH_PIN_DIR}"
    echo "=============================================================================="
}

case "${MODE}" in
    dry)
        banner
        echo "--dry-run: the bill. Nothing is submitted."
        _units ${DEPTHS:+--depths "${DEPTHS}"} ${ECC_JOBS:+--jobs "${ECC_JOBS}"}
        echo
        echo "  eval      : $(_eval_models) x ${ECC_EXPERIMENT}"\
             "-> ${ECC_RESULTS_DIR}/figures/${ECC_STEM:-<self-describing>}"
        [ "${ECC_SWEEP}" = "area" ] && \
            echo "  (ECC_SWEEP=area submits NO dependent eval -- read the ladder with"\
                 "python3 -m eccenergy.report.dilation_view --levels)"
        echo
        echo "  --dry-run CANNOT SEE THE CODE A JOB RUNS. It resolves the same"
        echo "  configuration and reads the same cache; only a real job proves the"
        echo "  job works. Smoke ONE cached unit before a matrix."
        ;;
    smoke)
        banner
        echo "--smoke: TWO real jobs -- the cache path and the MAPPER path."
        _units --smoke
        echo
        SMOKE=$(submit_smoke)
        echo "smoke job ${SMOKE}   log: hpc/logs/ecc-smoke.${SMOKE}_*.out"
        echo "  One array task per row listed above, in that order."
        echo "  watch : sacct -j ${SMOKE} -n --format=JobID%20,State,Elapsed"
        echo "  read  : tail -40 hpc/logs/ecc-smoke.${SMOKE}_*.out"
        echo "  BOTH must COMPLETE before a matrix. A cached unit returns"
        echo "  before Mapper._map_now and proves nothing about mapping."
        ;;
    eval)
        banner
        stage_eval
        ;;
    replot)
        banner
        _plot --replot
        ;;
    local)
        banner
        stage_map_local
        stage_eval
        ;;
    map)
        MAP=$(submit_map); banner
        echo "map  job ${MAP}   log: hpc/logs/ecc-map.${MAP}_*.out"
        echo "evaluate when it finishes with:  bash hpc/run_all.sh --eval-only"
        ;;
    all)
        MAP=$(submit_map)
        if [ "${ECC_SWEEP}" = "area" ]; then
            banner
            echo "map  job ${MAP}   log: hpc/logs/ecc-map.${MAP}_*.out"
            echo "NO dependent eval: ECC_SWEEP=area is a property of the MAPPINGS."
            echo "Read the ladder when it lands:"
            echo "  bash hpc/tl.sh python3 -m eccenergy.report.dilation_view --levels"
            exit 0
        fi
        EVAL=$(submit_eval "${MAP}")
        banner
        echo "map  job ${MAP}   log: hpc/logs/ecc-map.${MAP}_*.out"
        echo "eval job ${EVAL}   log: hpc/logs/ecc-eval.${EVAL}.out"
        echo "                  (runs when every map task succeeds; if one fails"
        echo "                   SLURM cancels it -- fix, rerun, or evaluate what"
        echo "                   exists with: bash hpc/run_all.sh --eval-only)"
        echo "watch: squeue -u \$USER"
        ;;
esac
