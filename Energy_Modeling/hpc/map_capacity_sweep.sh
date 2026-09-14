#!/usr/bin/env bash
# =============================================================================
#  hpc/map_capacity_sweep.sh -- TASK 4 STEP 1: map the SAME layers at several
#  weight-buffer capacities, so the capacity-dilation assumption can be
#  DIFFED rather than assumed.
#
#      bash hpc/map_capacity_sweep.sh                     # the env.sh point
#      ECC_CAPSWEEP_SCALES="1.0 1.6154" bash hpc/map_capacity_sweep.sh
#      bash hpc/map_capacity_sweep.sh --progress
#      bash hpc/map_capacity_sweep.sh --dry-run
#
#  WHAT IT IS FOR. Under the reconstruction arm the on-chip weight
#  representation is K/N of full width, so the same physical SRAM holds
#  N/K = 1.6154x more weights at BCH(63,39). The CLAIM is that a larger weight
#  tile makes the reconstruction arm reload tiles from DRAM fewer times than
#  the embedded reference, and that a read never issued removes the DRAM ARRAY
#  energy as well as the interface energy -- efficiency 1.0 per uJ against the
#  0.152 a fixed mapping buys. FINDINGS section 9 item 0 requires that claim be
#  validated BEFORE any of it is modelled, because real tiles are integers:
#  1.6154x capacity may buy a whole extra tile on one layer and nothing on the
#  next, and the whole projected table scales with that one number.
#
#  So this submits (architecture x layer x ECC_WEIGHT_CAPACITY_SCALE) mapping
#  jobs and NOTHING ELSE. There is deliberately no dependent evaluation:
#  the diff is a property of the MAPPINGS (DRAM weight reads per layer), it is
#  read with `python3 -m eccenergy.report.dilation_view`, and an eval job here
#  would draw a Task 3 figure over a Task 4 wave.
#
#  WHY SEVERAL SCALES AND NOT JUST 1.0 vs N/K. At the DECLARED buffer sizes
#  weight capacity is usually not the binding constraint at all -- the cached
#  mappings leave most of the weight scratchpad unused while still refetching,
#  so dilating it changes nothing and the honest answer is zero. Shrinking the
#  reference until capacity binds and dilating FROM THERE is what shows the
#  effect, and it is the same statement as FINDINGS section 9 item 0's
#  "shrinking SRAM and reconstruction-aware mapping are multiplicative": each
#  scale s is a reference design, and s x N/K is its reconstruction arm.
#
#  EVERY SCALE IS ITS OWN MAPPER CACHE (slug `wcap<scale>`, and its own
#  fingerprint), which is the point -- Task 4 is the diff of two mappings, so
#  they must never share a directory. A scale that rounds every weight level
#  back to its declared depth is not a dilation and keeps the undilated cache;
#  `fingerprint.effective_variant()` drops the slug in that case, so such a job is a
#  cache HIT and costs seconds.
#
#  Structure is the deleted `hpc/map_by_shape.sh`'s, deliberately: one job per unit of
#  work, `hpc/map.sbatch` unchanged, the SLURM task file snapshotted per
#  submission because map.sbatch resolves its (arch, model) pair from it when
#  the job RUNS, and unknown flags fatal rather than falling through to a
#  submission (2026-09-09: falling through submitted a duplicate wave).
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WANT_PROGRESS=0
DRY=0
for a in "$@"; do
    case "$a" in
        --progress) WANT_PROGRESS=1 ;;
        --dry-run)  DRY=1 ;;
        *) echo "map_capacity_sweep.sh: unknown option '$a'" >&2
           echo "  options: --progress  --dry-run" >&2
           exit 2 ;;
    esac
done
set --

source ./env.sh

# ---- the sweep ------------------------------------------------------------
# Each entry is one architecture handed to the mapper. The REFERENCE arm is a
# scale s and its RECONSTRUCTION arm is s x N/K, so the list is normally a set
# of such pairs. Defaults are computed from the code actually configured, not
# hard-coded to BCH(63,39).
NK=$(python3 -c "print(f'{${ECC_CODE_N}/${ECC_KS%% *}:.4f}')")
: "${ECC_CAPSWEEP_REFS:=1.0 0.5 0.25 0.125}"
if [ -z "${ECC_CAPSWEEP_SCALES:-}" ]; then
    ECC_CAPSWEEP_SCALES=""
    for s in ${ECC_CAPSWEEP_REFS}; do
        ECC_CAPSWEEP_SCALES="${ECC_CAPSWEEP_SCALES} ${s} $(python3 -c "print(f'{${s}*${NK}:.4f}')")"
    done
fi
# de-duplicate, keep order
ECC_CAPSWEEP_SCALES=$(printf '%s\n' ${ECC_CAPSWEEP_SCALES} | awk '!seen[$0]++' | tr '\n' ' ')

MODEL="$(set -- ${ECC_MODELS}; echo "$1")"
[ "$(set -- ${ECC_MODELS}; echo $#)" = 1 ] || {
    echo "map_capacity_sweep.sh: exactly ONE model, got ECC_MODELS='${ECC_MODELS}'" >&2
    exit 2; }
[ -n "${ECC_LAYERS}" ] || {
    echo "map_capacity_sweep.sh: ECC_LAYERS must name the layer(s) to sweep." >&2
    echo "  This is a SPOT CHECK by construction: it maps the same layers at" >&2
    echo "  $(set -- ${ECC_CAPSWEEP_SCALES}; echo $#) capacities x $(set -- ${ECC_ARCHS}; echo $#) designs, and a" >&2
    echo "  whole-model version of that is a week of compute. Pick the layers" >&2
    echo "  from a refetch survey:  python3 -m eccenergy.report.dilation_view --survey" >&2
    exit 2; }
read -r -a LAYERS <<< "${ECC_LAYERS}"

# ---- --progress: what is already solved, per (design, scale) --------------
if [ "${WANT_PROGRESS}" = "1" ]; then
    echo "capacity sweep: objective=${ECC_OPT_METRIC} victory=${ECC_VICTORY}" \
         "threads=${ECC_MAPPER_THREADS} scope=${ECC_WEIGHT_CAPACITY_SCOPE}"
    echo "  layers: ${ECC_LAYERS}"
    for A in ${ECC_ARCHS}; do
        for S in ${ECC_CAPSWEEP_SCALES}; do
            OUT=$(ECC_WEIGHT_CAPACITY_SCALE="${S}" ECC_SWEEP=arch ECC_SWEEP_ARCHS="${A}" \
                  python3 -c "
import pathlib
from eccenergy import config, paths
from eccenergy.arch import fingerprint
cfg = config.load_config()
a = cfg.archs[0]
d = pathlib.Path(paths.Results(cfg).mapper_cache(a, fingerprint.effective_variant(a, cfg),
                                                 fingerprint.arch_fingerprint(a, cfg)))
n = len(list(d.glob('*/timeloop-mapper.stats.txt'))) if d.is_dir() else 0
k = len(list(d.glob('*.lock'))) if d.is_dir() else 0
print(f'{n} {k} {d.name}')" 2>/dev/null | tail -1)
            printf "  %-26s x%-8s solved=%-3s locked=%-3s %s\n" "${A}" "${S}" \
                   "$(echo "${OUT}" | awk '{print $1}')" \
                   "$(echo "${OUT}" | awk '{print $2}')" \
                   "$(echo "${OUT}" | awk '{print $3}')"
        done
    done
    echo "  solved counts timeloop-mapper.stats.txt across ALL shapes in that cache,"
    echo "  so it can exceed the ${#LAYERS[@]} layer(s) swept if the design was mapped before."
    exit 0
fi

# ONE ROW PER UNIT, WRITTEN HERE, in hpc/map.sbatch's seven-column format
# (EnvReorganisation phase 3): bundle, arch, model, K, depth, arm, layer. One
# bundle per row, so `--array=i-i` maps exactly row i. The CAPACITY scale is
# not a column -- it is this sweep's own axis and travels in `--export`.
# Snapshotted so a later run cannot repoint a queued job at different work.
mkdir -p hpc/.runtime hpc/logs
SNAP="hpc/.runtime/tasks.capsweep.$$.txt"
: > "${SNAP}"
I=0
for A in ${ECC_ARCHS}; do
    for S in ${ECC_CAPSWEEP_SCALES}; do
        for L in "${LAYERS[@]}"; do
            printf '%s %s %s %s %s reference %s\n' "${I}" "${A}" "${MODEL}" \
                "${ECC_CONST_K}" "${ECC_WEIGHT_DEPTH_SCALE}" "${L}" >> "${SNAP}"
            I=$((I + 1))
        done
    done
done
NARCH="$(set -- ${ECC_ARCHS}; echo $#)"

NSCALE="$(set -- ${ECC_CAPSWEEP_SCALES}; echo $#)"
echo "map_capacity_sweep: ${NARCH} design(s) x ${#LAYERS[@]} layer(s) x ${NSCALE} capacit(ies)" \
     "-> $((NARCH * ${#LAYERS[@]} * NSCALE)) jobs of ${ECC_MAP_CPUS} cores"
echo "  designs   : ${ECC_ARCHS}"
echo "  layers    : ${ECC_LAYERS}"
echo "  capacities: ${ECC_CAPSWEEP_SCALES}   (N/K = ${NK}, scope=${ECC_WEIGHT_CAPACITY_SCOPE})"
echo "  objective : ${ECC_OPT_METRIC}   victory=${ECC_VICTORY}   qos=${ECC_QOS}"
echo "  task file : ${SNAP}"
echo "  NO dependent eval is submitted -- read the diff with"
echo "      python3 -m eccenergy.report.dilation_view"

if [ "${DRY}" = "1" ]; then
    echo "  --dry-run: nothing submitted."
    exit 0
fi

JOBS=()
I=0
for A in ${ECC_ARCHS}; do
    for S in ${ECC_CAPSWEEP_SCALES}; do
        for L in "${LAYERS[@]}"; do
            jid=$(sbatch --parsable \
                --job-name="cap-${A}-${S}-${L}" \
                --account="${ECC_ACCOUNT}" --qos="${ECC_QOS}" --partition="${ECC_PARTITION}" \
                --cpus-per-task="${ECC_MAP_CPUS}" --mem="${ECC_MAP_MEM}" --time="${ECC_MAP_TIME}" \
                --array="${I}-${I}" \
                --output="hpc/logs/cap-shape.%A.out" \
                --export=ALL,ECC_TASKFILE="${PWD}/${SNAP}",ECC_WEIGHT_CAPACITY_SCALE="${S}" \
                hpc/map.sbatch)
            JOBS+=("${jid}")
            echo "  job ${jid}  ${A}  x${S}  ${L}"
            I=$((I + 1))
        done
    done
done
DEP=$(IFS=:; echo "${JOBS[*]}")
echo "capsweep_jobs=${DEP}" > hpc/.runtime/map_capacity_sweep.last
echo "  ${#JOBS[@]} jobs submitted. Watch:  bash hpc/map_capacity_sweep.sh --progress"
