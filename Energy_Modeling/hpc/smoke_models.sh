#!/usr/bin/env bash
# =============================================================================
#  hpc/smoke_models.sh -- DOES THE MAPPER SOLVE THIS MODEL AT ALL?
#
#      bash hpc/smoke_models.sh                      # every model in env.sh's CNN list
#      bash hpc/smoke_models.sh convnext_tiny xception
#      ECC_SMOKE_SHAPES=5 bash hpc/smoke_models.sh   # 5 shapes each instead of 3
#      bash hpc/smoke_models.sh --report             # read the last run's logs
#      bash hpc/smoke_models.sh --dry-run
#
#  ONE REAL MAP PER (MODEL, HARD SHAPE), REFERENCE ARM ONLY -- minutes, before
#  anyone commits to 6 arms x every shape x eight models (2026-09-13: that is
#  1,452 jobs). `bash hpc/run_all.sh --dry-run` cannot answer this question: it
#  checks the configuration and reads the cache, so it is blind to the one thing
#  that costs a wave of compute, which is whether Timeloop finds a legal mapping
#  for a shape this model has and the others do not.
#
#  THE SHAPES ARE DERIVED, never listed: the model's distinct layer shapes
#  ranked by R*S, then C*M, then P*Q -- big filters first, because those are
#  where the constrained mapspace (prompt_3) runs out of room. Plus the
#  smallest, because a 1x1 over a 1x1 feature map is the other end and has its
#  own way of having no legal mapping.
#
#  WHAT IT COST NOT TO HAVE THIS. efficientnet_b0 drew no figure on 2026-09-13
#  and the model was blamed for it; 282 jobs had in fact all succeeded and the
#  architecture had changed underneath them. Twelve jobs of this would have said
#  in four minutes that all eight models map, and pointed the search elsewhere.
#
#  This does NOT pin `archs/` -- it is a probe of the architecture as it stands
#  right now, which is the question being asked. The cache entries it writes are
#  ordinary, at the current fingerprint, and a later full run reads them as hits.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY=0
REPORT=0
MODELS=()
for a in "$@"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        --report)  REPORT=1 ;;
        -*) echo "smoke_models.sh: unknown option '$a'" >&2
            echo "  options: --dry-run  --report   (bare args are model names)" >&2
            exit 2 ;;
        *) MODELS+=("$a") ;;
    esac
done
set --

source ./env.sh

# The CNN list `config.CNN_MODELS` declares; this is that list, and it is the
# default scope.
[ "${#MODELS[@]}" -gt 0 ] || MODELS=(resnet18 resnet50 densenet121 squeezenet1_1
                                     mobilenet_v2 efficientnet_b0 convnext_tiny xception)
NSHAPES="${ECC_SMOKE_SHAPES:-3}"

if [ "${REPORT}" = "1" ]; then
    echo "smoke_models: last run (hpc/logs/smoke.*.out)"
    ok=0; bad=0
    for f in $(ls -t hpc/logs/smoke.*.out 2>/dev/null | head -60); do
        m=$(grep -m1 -o 'model=[^ ]*' "$f" 2>/dev/null | cut -d= -f2)
        l=$(grep -m1 "layers ok" "$f" 2>/dev/null | sed 's/^ *//')
        r=$(grep -m1 "newly mapped" "$f" 2>/dev/null | sed 's/.*, \([0-9]*\) failed.*/\1/')
        [ -n "${m}" ] || continue
        if [ "${r:-1}" = "0" ]; then ok=$((ok+1)); else bad=$((bad+1)); fi
        printf "  %-16s %-28s %s\n" "${m}" "${l}" "${f#hpc/logs/}"
    done
    echo "  ${ok} shape(s) mapped, ${bad} failed"
    exit 0
fi

mkdir -p hpc/.runtime hpc/logs
echo "smoke_models: ${#MODELS[@]} model(s) x ${NSHAPES} hard shape(s), reference arm," \
     "on ${ECC_ARCHS}"

for M in "${MODELS[@]}"; do
    # The hardest shapes of THIS model, one representative layer each.
    mapfile -t LAYERS < <(ECC_CONST_MODEL="${M}" ECC_SWEEP_MODELS="${M}" \
                          ECC_PANEL_MODELS="" ECC_LAYERS="" \
                          bash hpc/tl.sh python3 - "${NSHAPES}" 2>/dev/null <<'PY'
import re, sys
from eccenergy import config
from eccenergy.arch import workloads
cfg = config.load_config()
models, _ = workloads.load_workload(cfg)
name = cfg.models[0]
seen = {}
for l in models[name]:
    seen.setdefault(l.shape_name, l.name)
def hardness(shape):
    d = {k: int(v) for k, v in re.findall(r"([CMRSPQG])(\d+)", shape)}
    return (d.get("R", 1) * d.get("S", 1),
            d.get("C", 1) * d.get("M", 1) * d.get("G", 1),
            d.get("P", 1) * d.get("Q", 1))
ranked = sorted(seen.items(), key=lambda kv: hardness(kv[0]), reverse=True)
n = max(1, int(sys.argv[1]))
picked = [l for _s, l in ranked[:n]]
if ranked and ranked[-1][1] not in picked:      # the other end of the range
    picked.append(ranked[-1][1])
print("\n".join(picked))
PY
)
    [ "${#LAYERS[@]}" -gt 0 ] || {
        echo "  !! ${M}: no layers resolved -- is it in the workload file?" >&2
        continue; }
    # ONE TASK-FILE ROW PER SHAPE, in hpc/map.sbatch's seven-column format
    # (EnvReorganisation phase 3): bundle, arch, model, K, depth, arm, layer.
    # One bundle per row, so `--array=i-i` maps exactly row i.
    SNAP="hpc/.runtime/tasks.smoke.${M}.txt"
    : > "${SNAP}"
    I=0
    for L in "${LAYERS[@]}"; do
        printf '%s %s %s %s %s reference %s\n' "${I}" "$(set -- ${ECC_ARCHS}; echo "$1")" \
            "${M}" "${ECC_CONST_K}" "${ECC_WEIGHT_DEPTH_SCALE}" "${L}" >> "${SNAP}"
        I=$((I + 1))
    done
    I=0
    for L in "${LAYERS[@]}"; do
        if [ "${DRY}" = "1" ]; then
            echo "  would submit: smoke-${M}-${L}"
            I=$((I + 1))
            continue
        fi
        jid=$(sbatch --parsable \
            --job-name="smoke-${M}-${L}" \
            --account="${ECC_ACCOUNT}" --qos="${ECC_QOS}" --partition="${ECC_PARTITION}" \
            --cpus-per-task="${ECC_MAP_CPUS}" --mem="${ECC_MAP_MEM}" --time=02:00:00 \
            --array="${I}-${I}" \
            --output="hpc/logs/smoke.%A.out" \
            --export=ALL,ECC_TASKFILE="${PWD}/${SNAP}" \
            hpc/map.sbatch)
        echo "  job ${jid}  ${M}  ${L}"
        I=$((I + 1))
    done
done
[ "${DRY}" = "1" ] || echo "  read the result with:  bash hpc/smoke_models.sh --report"
