#!/usr/bin/env bash
# =============================================================================
#  hpc/run_all.sh  --  THE ONE COMMAND
# =============================================================================
#
#      cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
#      module load apptainer
#      bash hpc/run_all.sh                # map -> evaluate -> plot
#
#  Everything it runs comes from ../env.sh: which architectures, which models,
#  which code, how hard the mapper searches, what the cluster is asked for.
#  There are no knobs in this file.
#
#  WHAT IT DOES
#    1. writes the (architecture, model) task list from ECC_ARCHS x ECC_MODELS
#    2. submits hpc/map.sbatch as a SLURM array over it -- the expensive half,
#       Timeloop + Accelergy solving one mapping per distinct layer shape
#    3. submits ITSELF (`--eval-only`) with --dependency=afterok on that array,
#       so the evaluation and the figures happen by themselves the moment the
#       last mapping lands
#  It returns at once; watch with `squeue -u $USER`. Runs are resumable: rerun
#  after a failure and every solved shape is a cache hit.
#
#  ONE-OFF OVERRIDES need no edit anywhere, because env.sh lets the environment
#  win:
#      ECC_MODELS="resnet18 resnet50 densenet121" bash hpc/run_all.sh
#      ECC_MAPPER_SEARCH_SIZE=20000 bash hpc/run_all.sh   # bounded dev pass
#      ECC_MAP_TIME=08:00:00 ECC_CONCURRENCY=6 bash hpc/run_all.sh
#
#  WITH ECC_RECON_MODELING=1 (env.sh section 4) every mode below runs the
#  reconstruction PLACEMENT study of Task 3 instead: one architecture, one model,
#  one code, and the x axis is WHERE the reconstruction boundary sits. It is an
#  evaluator-only comparison, so `--eval-only` is the mode to use and the
#  mapping array only has the one pair to solve.
#
#      ECC_RECON_MODELING=1 bash hpc/run_all.sh --eval-only
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
        -h|--help)   usage; exit 0 ;;
        *) echo "hpc/run_all.sh: unknown option '$1'" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

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

# Evaluate every model, then draw. Everything here is --eval: mappings are read
# from the cache and Timeloop is never invoked, so this cannot change a mapping.
stage_eval() {
    # --eval forbids mapping outright, so a re-map request cannot survive into
    # this stage; it belongs to the mapping stage alone.
    export ECC_RERUN_OPTIMISER=0
    # WHICH evaluations are written is DERIVED (ECC_EVAL_EXPERIMENTS went on
    # 2026-09-14): the placement study when env.sh section 4 routed the run to
    # it -- its file holds Task 1's and Task 2's bars itself -- and Task 1 +
    # Task 2 otherwise.
    local evals="baseline embedded" m x
    [ "${ECC_EXPERIMENT}" = "recon" ] && evals="recon"
    for m in ${ECC_MODELS}; do
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

# Map every (architecture, model) pair in this process, one at a time. For an
# interactive allocation; the SLURM array below is the real path.
stage_map_local() {
    local m a
    for m in ${ECC_MODELS}; do
        for a in ${ECC_ARCHS}; do
            echo "############ map: ${a} -- ${m} ############"
            ( export ECC_SWEEP=arch ECC_SWEEP_ARCHS="${a}" ECC_CONST_MODEL="${m}"
              _run map )
        done
    done
}

# --------------------------------------------------------------------------
#  submitting
# --------------------------------------------------------------------------
# THE ARCH PIN -- same contract as hpc/map_ert_arms.sh, see eccenergy/paths.py.
# `archs/` is snapshotted at SUBMISSION and both the map array and the dependent
# eval read the copy, so editing a declared value while the array is in flight
# cannot split one run across two architectures.
#
# ONLY THE TOP-LEVEL SUBMITTER PINS. `submit_eval` re-invokes this script as
# `hpc/run_all.sh --eval-only` inside the eval job, where ECC_ARCH_PIN_DIR is
# already set by --export=ALL: taking a second snapshot there would read the
# LIVE archs/ and undo the whole point.
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

submit_map() {
    pin_archs >/dev/null
    ecc_write_taskfile
    local n
    n=$(grep -cve '^[[:space:]]*$' "${ECC_TASKFILE}")
    [ "${n}" -gt 0 ] || {
        echo "hpc/run_all.sh: ECC_ARCHS x ECC_MODELS is empty -- nothing to map" >&2
        exit 2; }
    mkdir -p hpc/logs
    sbatch --parsable \
        --job-name=ecc-map \
        --account="${ECC_ACCOUNT}" --qos="${ECC_QOS}" --partition="${ECC_PARTITION}" \
        --cpus-per-task="${ECC_MAP_CPUS}" --mem="${ECC_MAP_MEM}" \
        --time="${ECC_MAP_TIME}" \
        --array="0-$((n - 1))%${ECC_CONCURRENCY}" \
        --output="hpc/logs/ecc-map.%A_%a.out" \
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
    local n
    n=$(grep -cve '^[[:space:]]*$' "${ECC_TASKFILE}" 2>/dev/null || echo 0)
    echo "=============================================================================="
    if [ "${ECC_RECON_MODELING}" = "1" ]; then
        echo " STUDY      : reconstruction PLACEMENT (Task 3), fixed mapping."
        echo "              section 4 of env.sh holds arch/model/code; the x axis"
        echo "              is WHERE the boundary sits."
        echo " placements : ${ECC_RECON_PLACEMENT_LIST:-every one this design defines}"\
             "  packing=${ECC_RECON_PACKING} encoder=${ECC_RECON_ENCODER_GRANULARITY}"
        if [ "${ECC_RECON_OPTIMIZER}" = "True" ] || [ "${ECC_RECON_OPTIMIZER}" = "1" ]; then
            echo " remapping  : RECON_OPTIMIZER=True -- TASK 4: the reconstruction bars"
            echo "              come from a SECOND mapping at weight capacity"
            echo "              x${ECC_WEIGHT_CAPACITY_SCALE} x N/K (scope=${ECC_WEIGHT_CAPACITY_SCOPE}); needs that"
            echo "              cache and refuses rather than falling back -> ${ECC_STEM:-ReconSweep_optimiser}"
        else
            echo " remapping  : RECON_OPTIMIZER=False -- Task 3, one fixed mapping on every arm"
            echo "              (True is Task 4; validate the capacity"
            echo "               assumption first with  bash run.sh dilation)"
        fi
    fi
    echo " archs      : ${ECC_ARCHS}"
    echo " models     : ${ECC_MODELS}"
    echo " code       : BCH(${ECC_CODE_N}, ${ECC_KS})   arms: ${ECC_APPROACHES}"
    echo " scope      : ${ECC_LAYERS:-whole model}"
    echo " mapper     : ${ECC_MAPPER_ALGORITHM} victory=${ECC_VICTORY}/${ECC_VICTORY_SCALING}"\
         "search=${ECC_MAPPER_SEARCH_SIZE:-uncapped} threads=${ECC_MAPPER_THREADS}"
    echo " figure     : ${ECC_RESULTS_DIR}/figures/${ECC_STEM:-<self-describing>}"
    [ "${n}" -gt 0 ] && echo " map tasks  : ${n} (arch, model) pairs, ${ECC_CONCURRENCY} at once"
    [ -n "${ECC_ARCH_PIN_DIR}" ] && echo " arch pin   : ${ECC_ARCH_PIN_DIR}"
    echo "=============================================================================="
}

case "${MODE}" in
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
