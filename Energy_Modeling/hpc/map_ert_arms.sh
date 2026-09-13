#!/usr/bin/env bash
# =============================================================================
#  hpc/map_ert_arms.sh -- prompt_6: map the reference arm AND every ERT arm of
#  the placement study as separate SLURM jobs, all at once, then evaluate once.
#
#      bash hpc/map_ert_arms.sh                 # env.sh's study point: one layer
#      ECC_RECON_LAYER=all bash hpc/map_ert_arms.sh   # every layer of the model
#      bash hpc/map_ert_arms.sh --no-eval       # maps only, no dependent eval
#      bash hpc/map_ert_arms.sh --progress      # what is cached, per arm
#      bash hpc/map_ert_arms.sh --dry-run       # print the jobs, submit nothing
#
#  A copy of hpc/map_by_shape.sh with the fan-out dimension changed from
#  ARCHITECTURES to ARMS (prompt_6 7.2). The unit of work is (arm x layer
#  shape), and every job is independent: the five boundaries are five separate
#  chips (prompt_6 3.2), the reference is a sixth, and no arm reads another's
#  output. Job count = arms x shapes; on Eyeriss v1 that is 6 x 43 = 258 jobs
#  for the whole model, or 6 with ECC_RECON_LAYER set to one layer. Wall time =
#  the slowest single map.
#
#  THE ARMS ARE DERIVED, never listed here: `recon.mapper_arms()` -- the
#  reference plus every boundary that is a DISTINCT CHIP to the mapper, over
#  the three axes of prompt_7 6.4:
#
#      datawidth: q on the storage levels the boundary narrows
#    x the ERT bump, where its encoder attaches to one real ERT action and the
#      bump is not constant across the mapspace
#    x the declared per-dataspace bandwidth scale (a network stage is carried
#      and MARKED no-op: Timeloop has no network timing model, but a boundary
#      that adds one is still a different declaration)
#
#  then filtered to ECC_RECON_PLACEMENTS. On eyeriss_like_wglb that is SIX arms
#  and on eyeriss_v2_like FIVE. It used to be `recon.ert_arms()` -- three arms
#  on Eyeriss v1 -- which answers "which boundaries have an ERT-injectable
#  encoder", a different question: R3 shares R2's geometry but declares no ERT
#  bump, and R5a narrows a level no other arm narrows, so three of the five
#  boundaries were evaluated on a plan belonging to a different chip
#  (prompt_7 Defect 3). Two boundaries that agree on all three axes ARE one
#  chip and share one job; --dry-run prints which.
#
#  Four things carried over from map_by_shape.sh, each paid for once already:
#    * THE ARM TRAVELS IN --export, not in the task file. map.sbatch resolves
#      its (architecture, model) pair from the task file by array index, and
#      every arm here is the SAME architecture in a different configuration.
#      ECC_RECON_ERT_ARM tells one mapper job which arm it solves; config.py
#      turns it into datawidth q on the boundary's storage levels plus the
#      ERT bump, and it is in the cache slug and the fingerprint (RULE 4.4.5),
#      so two arms never write one directory.
#    * --array=I-I IS REQUIRED, one index per submission: map.sbatch carries
#      its own #SBATCH --array sized for a whole task file; without the
#      override the extra tasks exit 2 and the dependent eval sits on
#      DependencyNeverSatisfied (2026-09-09, jobs 41474947-58).
#    * THE TASK FILE IS SNAPSHOTTED, so a later launcher cannot change what a
#      queued job maps.
#    * AND SO IS `archs/`, since 2026-09-13 -- into hpc/.runtime/archpin.<pid>/,
#      pointed at by ECC_ARCH_PIN_DIR in every --export below. ONE SUBMISSION
#      MAPS ONE ARCHITECTURE: edit a depth while the array is in flight and the
#      queued jobs still map the chip you submitted, instead of half the run
#      solving one geometry and the eval looking for another. Try new values
#      whenever you like; the next run snapshots them.
#    * ONE DEPENDENT EVAL, afterok on every map job, with ECC_RECON_ERT_AWARE=1
#      forced and ECC_RECON_ERT_ARM=reference, at the same ECC_RECON_LAYER
#      scope, so it sees exactly the arms it depends on. It writes
#      results/figures/ReconSweep_optimiser__<model>.png (env.sh section 10 keeps that
#      name on a layer-scoped run).
#
#  The layer travels as ECC_RECON_LAYER, not ECC_LAYERS: under
#  ECC_RECON_MODELING=1 section 10 seeds ECC_LAYERS from it, so an ECC_LAYERS
#  in --export would be overwritten.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WANT_EVAL=1
WANT_PROGRESS=0
DRY=0
for a in "$@"; do
    case "$a" in
        --no-eval)  WANT_EVAL=0 ;;
        --progress) WANT_PROGRESS=1 ;;
        --dry-run)  DRY=1 ;;
        *) echo "map_ert_arms.sh: unknown option '$a'" >&2
           echo "  options: --no-eval  --progress  --dry-run" >&2
           exit 2 ;;
    esac
done
set --

source ./env.sh

if [ "${ECC_RECON_MODELING}" != "1" ]; then
    echo "map_ert_arms.sh: ECC_RECON_MODELING=1 is required -- the arms are the placement" >&2
    echo "  study's boundaries (env.sh section 4)" >&2
    exit 2
fi
NARCH="$(set -- ${ECC_ARCHS}; echo $#)"
MODEL="$(set -- ${ECC_MODELS}; echo "$1")"
[ "${NARCH}" = 1 ] && [ "$(set -- ${ECC_MODELS}; echo $#)" = 1 ] || {
    echo "map_ert_arms.sh: exactly ONE architecture and ONE model, got" \
         "ECC_RECON_ARCHS='${ECC_ARCHS}' ECC_RECON_MODEL='${ECC_MODELS}'" >&2; exit 2; }
ARCH="${ECC_ARCHS}"

# ---------------------------------------------------------------- THE ARCH PIN
# ONE SUBMISSION MAPS ONE ARCHITECTURE. `archs/` is copied here and now, and
# every job below -- the arm derivation, all the maps, and the dependent eval --
# reads the COPY. Editing `archs/` afterwards cannot reach a job that is already
# queued or running, so trying a new value the moment a batch goes out is free.
#
# WHAT THIS COST BEFORE IT EXISTED (2026-09-13, efficientnet_b0). A psum_spad
# depth was changed 78 seconds after the array started. The 282 maps solved the
# pre-edit chip (fp-eca1a94653f8); the eval, twelve minutes later, read the
# edited file, computed fp-d20f19432878, found it empty and reported all 82
# layers as `mapper failed`. Both halves were correct and they described two
# different chips. eccenergy/paths.py has the long version.
#
# The fingerprint is still computed from the pinned bytes and still names the
# cache directory: this fixes WHICH architecture a run is, it does not let two
# of them share a bar.
PIN="hpc/.runtime/archpin.$$"
mkdir -p hpc/.runtime hpc/logs
# Each pin is ~320 kB and is the record of what a run mapped, so it is kept
# for as long as anything could still want it -- but not forever. A week is
# far past ECC_MAP_TIME, so nothing queued can still be reading one.
find hpc/.runtime -maxdepth 1 -name 'archpin.*' -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true
rm -rf "${PIN}"
cp -a archs "${PIN}"
export ECC_ARCH_PIN_DIR="${PWD}/${PIN}"
trap 'rm -rf "${PIN}"' EXIT      # cleared once the jobs are actually submitted

# The arms, derived inside the container from the placement records. One TAB
# separated line per arm: key, cache slug, what it declares, its members.
mapfile -t ARM_ROWS < <(bash hpc/tl.sh python3 - 2>/dev/null <<'PY'
from eccenergy import config, recon
cfg = config.load_config()
arch = cfg.archs[0]
wanted = cfg.recon_placements_for(arch)
for a in recon.mapper_arms(arch, cfg):
    if a.placement is not None and wanted and not (
            a.key in wanted or a.placement.variant in wanted
            or any(m in wanted for m in a.members)):
        continue
    print("\t".join((a.key, a.slug_part or "-", a.describe(),
                      "+".join(a.members) or "-")))
PY
)
ARMS=()
for row in "${ARM_ROWS[@]}"; do ARMS+=("${row%%$'\t'*}"); done
[ "${#ARMS[@]}" -ge 1 ] && [ "${ARMS[0]}" = "reference" ] || {
    echo "map_ert_arms.sh: could not derive the arms for ${ARCH} (is the container available?)" >&2
    exit 2; }

# GATE (prompt_7 B2, gate 2): every arm's cache slug must be its own. Two arms
# in one directory is the failure prompt_6 RULE 4.4.5 exists to prevent, and
# R1's patched YAML IS the reference's until Phase C1.2 declares the bandwidth
# scale -- so this is a live risk, not a theoretical one.
NSLUG=$(printf '%s\n' "${ARM_ROWS[@]}" | cut -f2 | sort -u | wc -l)
[ "${NSLUG}" = "${#ARMS[@]}" ] || {
    echo "map_ert_arms.sh: ${#ARMS[@]} arms share only ${NSLUG} cache slug(s) --" >&2
    echo "  two arms would write one directory and the second map would overwrite" >&2
    echo "  or skip the first (prompt_6 RULE 4.4.5 defence 1). Arms:" >&2
    printf '    %s\n' "${ARM_ROWS[@]}" | tr '\t' ' ' >&2
    exit 2; }

# One representative layer per distinct shape, or exactly ECC_RECON_LAYER.
if [ -n "${ECC_LAYERS}" ]; then
    read -r -a LAYERS <<< "${ECC_LAYERS}"
else
mapfile -t LAYERS < <(bash hpc/tl.sh python3 - "${MODEL}" 2>/dev/null <<'PY'
import sys
from eccenergy import config, workloads
cfg = config.load_config()
models, _ = workloads.load_workload(cfg)
seen = {}
for l in models[sys.argv[1]]:
    seen.setdefault(l.shape_name, l.name)
print("\n".join(seen.values()))
PY
)
fi
[ "${#LAYERS[@]}" -gt 0 ] || { echo "map_ert_arms.sh: no layers resolved for ${MODEL}" >&2; exit 2; }

# --progress: what is cached per arm at the CURRENT fingerprint. Counts
# stats.txt files, never directories (a claimed shape has a directory at once).
if [ "${WANT_PROGRESS}" = "1" ]; then
    echo "mapper cache per arm: ${ARCH} / ${MODEL}  objective=${ECC_OPT_METRIC} victory=${ECC_VICTORY}"
    echo "  ${#ARMS[@]} arm(s) -- the DISTINCT CHIPS of this design (prompt_7 6.4):"
    printf '%s\n' "${ARM_ROWS[@]}" | cut -f3 | sed 's/^/    /'
    echo "  cache state per arm (solved counts stats.txt files, never directories:"
    echo "  a claimed shape has a directory at once):"
    for ARM in "${ARMS[@]}"; do
        D=$(ECC_RECON_ERT_ARM="${ARM}" bash hpc/tl.sh python3 -c "
from eccenergy import archs, config, paths
cfg = config.load_config()
a = cfg.archs[0]
print(paths.Results(cfg).mapper_cache(a, archs.effective_variant(a, cfg),
                                      archs.arch_fingerprint(a, cfg), create=False))" 2>/dev/null | tail -1)
        if [ -z "${D}" ] || [ ! -d "${D}" ]; then
            printf "  %-10s no cache directory at this fingerprint yet\n" "${ARM}"
            continue
        fi
        SOLVED=$(find "${D}" -mindepth 2 -maxdepth 2 -name timeloop-mapper.stats.txt 2>/dev/null | wc -l)
        LOCKED=$(find "${D}" -maxdepth 1 -name '*.lock' 2>/dev/null | wc -l)
        printf "  %-10s solved=%-3s in-progress=%-3s  %s\n" "${ARM}" "${SOLVED}" "${LOCKED}" "${D#${PWD}/}"
    done
    echo "  layers in scope: ${#LAYERS[@]} (${LAYERS[*]})"
    exit 0
fi

mkdir -p hpc/.runtime hpc/logs
ecc_write_taskfile
SNAP="hpc/.runtime/tasks.ert-arms.$$.txt"
cp "${ECC_TASKFILE}" "${SNAP}"
NLINES=$(grep -cve '^[[:space:]]*$' "${SNAP}")
[ "${NLINES}" = 1 ] || {
    echo "map_ert_arms.sh: task file has ${NLINES} line(s) for one (architecture, model)" \
         "-- refusing rather than mapping the wrong pair" >&2; exit 2; }

echo "map_ert_arms: ${#ARMS[@]} arm(s) x ${#LAYERS[@]} shape(s) x ${MODEL} on ${ARCH}" \
     "-> $(( ${#ARMS[@]} * ${#LAYERS[@]} )) jobs of ${ECC_MAP_CPUS} cores, all at once"
echo "  arms      : ${ARMS[*]}   (the DISTINCT CHIPS, recon.mapper_arms(); prompt_7 6.4)"
printf '%s\n' "${ARM_ROWS[@]}" | cut -f3 | sed 's/^/    /'
echo "  layers    : ${LAYERS[*]}"
echo "  objective=${ECC_OPT_METRIC} victory=${ECC_VICTORY} code=BCH(${ECC_CODE_N},${ECC_KS})" \
     "constrain=${ECC_MAPSPACE_CONSTRAIN} relax=${ECC_WEIGHT_FACTOR_RELAX} results=${ECC_RESULTS_DIR}"
echo "  task file : ${SNAP}"
echo "  arch pin  : ${ECC_ARCH_PIN_DIR}   (this run maps THIS copy of archs/;"
echo "              edit archs/ freely from now on -- the queued jobs cannot see it)"
PIN_FP=$(bash hpc/tl.sh python3 -c "
from eccenergy import archs, config
cfg = config.load_config()
print(archs.arch_fingerprint(cfg.archs[0], cfg))" 2>/dev/null | tail -1)
echo "  reference fp: ${PIN_FP:-<unavailable>}   (the cache directory every arm is keyed under)"
if [ "${DRY}" = "1" ]; then
    echo "  cache slugs (one per arm, all distinct -- checked above):"
    printf '%s\n' "${ARM_ROWS[@]}" | awk -F'\t' '{printf "    %-10s %s\n", $1, $2}'
    for ARM in "${ARMS[@]}"; do for L in "${LAYERS[@]}"; do
        echo "  would submit: map-ert-${ARM}-${L}  (--export ECC_RECON_LAYER=${L} ECC_RECON_ERT_ARM=${ARM})"
    done; done
    [ "${WANT_EVAL}" = "1" ] && echo "  would submit: ecc-eval-ert (afterok all of the above, ECC_RECON_ERT_AWARE=1)"
    echo "  ${#ARMS[@]} arm(s) x ${#LAYERS[@]} shape(s) = $(( ${#ARMS[@]} * ${#LAYERS[@]} )) map job(s); --progress says which are already cached"
    rm -f "${SNAP}"
    exit 0
fi
JOBS=()
for ARM in "${ARMS[@]}"; do
    for L in "${LAYERS[@]}"; do
        jid=$(sbatch --parsable \
            --job-name="map-ert-${ARM}-${L}" \
            --account="${ECC_ACCOUNT}" --qos="${ECC_QOS}" --partition="${ECC_PARTITION}" \
            --cpus-per-task="${ECC_MAP_CPUS}" --mem="${ECC_MAP_MEM}" --time="${ECC_MAP_TIME}" \
            --array=0-0 \
            --output="hpc/logs/map-ert.%A.out" \
            --export=ALL,ECC_RECON_LAYER="${L}",ECC_RECON_ERT_ARM="${ARM}",ECC_TASKFILE="${PWD}/${SNAP}",ECC_ARCH_PIN_DIR="${ECC_ARCH_PIN_DIR}" \
            hpc/map.sbatch)
        JOBS+=("${jid}")
        echo "  job ${jid}  ${ARM}  ${L}"
    done
done
DEP=$(IFS=:; echo "${JOBS[*]}")
# The jobs are queued and they read ${PIN} when they RUN, so it must survive
# this script. Delete it by hand once the run is done, or leave it as the
# record of which architecture those numbers came from.
trap - EXIT
echo "map_jobs=${DEP}" > hpc/.runtime/map_ert_arms.last
echo "arch_pin=${ECC_ARCH_PIN_DIR}" >> hpc/.runtime/map_ert_arms.last
if [ "${WANT_EVAL}" != "1" ]; then
    echo "  --no-eval: no dependent evaluation submitted."
    echo "  when every arm is cached:  ECC_RECON_ERT_AWARE=1 bash hpc/tl.sh bash run.sh recon --eval"
    echo "  or depend on these:        --dependency=afterok:${DEP}"
    exit 0
fi
EVAL=$(sbatch --parsable --dependency="afterok:${DEP}" \
    --job-name=ecc-eval-ert \
    --account="${ECC_ACCOUNT}" --qos="${ECC_QOS}" --partition="${ECC_PARTITION}" \
    --cpus-per-task="${ECC_EVAL_CPUS}" --mem="${ECC_EVAL_MEM}" --time="${ECC_EVAL_TIME}" \
    --output="hpc/logs/ecc-eval-ert.%j.out" \
    --export=ALL,ECC_RECON_ERT_AWARE=1,ECC_RECON_ERT_ARM=reference,ECC_ARCH_PIN_DIR="${ECC_ARCH_PIN_DIR}" \
    hpc/run_all.sh --eval-only)
echo "  eval job ${EVAL} (afterok all ${#JOBS[@]} map jobs, ECC_RECON_ERT_AWARE=1)" \
     "-> ${ECC_RESULTS_DIR}/figures/${ECC_STEM:-<self-describing>}"
echo "eval_job=${EVAL}" >> hpc/.runtime/map_ert_arms.last
