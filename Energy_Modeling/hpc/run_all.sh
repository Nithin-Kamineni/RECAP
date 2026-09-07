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
    local m x
    for m in ${ECC_MODELS}; do
        for x in ${ECC_EVAL_EXPERIMENTS}; do
            echo "############ evaluate: ${x} -- ${m} ############"
            ( export ECC_CONST_MODEL="${m}"; _run "${x}" --eval )
        done
    done
    _plot --eval
    echo
    echo "figures : ${ECC_RESULTS_DIR}/figures/"
    echo "tables  : ${ECC_RESULTS_DIR}/tables/"
    echo "results : ${ECC_RESULTS_DIR}/evaluation/${ECC_PHASE}/<arch>/<model>/..."
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
submit_map() {
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
    echo " archs      : ${ECC_ARCHS}"
    echo " models     : ${ECC_MODELS}"
    echo " code       : BCH(${ECC_CODE_N}, ${ECC_KS})   arms: ${ECC_APPROACHES}"
    echo " scope      : ${ECC_LAYERS:-whole model}"
    echo " mapper     : ${ECC_MAPPER_ALGORITHM} victory=${ECC_VICTORY}/${ECC_VICTORY_SCALING}"\
         "search=${ECC_MAPPER_SEARCH_SIZE:-uncapped} threads=${ECC_MAPPER_THREADS}"
    echo " figure     : ${ECC_RESULTS_DIR}/figures/${ECC_STEM:-<self-describing>}"
    [ "${n}" -gt 0 ] && echo " map tasks  : ${n} (arch, model) pairs, ${ECC_CONCURRENCY} at once"
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
