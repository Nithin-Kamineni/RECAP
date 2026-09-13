#!/usr/bin/env bash
# =============================================================================
#  hpc/map_depth_sweep.sh -- PROMPT_2: the WEIGHT-MEMORY DEPTH sweep.
#
#      bash hpc/map_depth_sweep.sh                # the env.sh point
#      bash hpc/map_depth_sweep.sh --all-shapes   # whole model, 1 job / SHAPE
#      bash hpc/map_depth_sweep.sh --no-per-level # skip the 2nd (per-level) pass
#      bash hpc/map_depth_sweep.sh --dry-run
#      bash hpc/map_depth_sweep.sh --progress
#      bash hpc/map_depth_sweep.sh --gate-only    # the convergence gate alone
#      bash hpc/map_depth_sweep.sh --no-gate      # the sweep alone
#      ECC_WEIGHT_DEPTH_LEVELS=filter_glb bash hpc/map_depth_sweep.sh  # 2nd pass
#
#  THE QUESTION. For Eyeriss v1, one layer at a time: what on-chip weight-memory
#  DEPTH makes reconstruction save the most energy against embedded ECC? Recon
#  stores the same weights in fewer bits, so the same array holds more of them
#  and refetches less from DRAM. The deliverable is NOT a trend, it is a set of
#  `depth:` values to write into a new arch YAML.
#
#  TWO MAPPER CONFIGS PER DEPTH, NOT THREE. Baseline and Embedded both store
#  8-bit weights, so they share one DEPTH and one mapping -- only the
#  evaluator separates them. Recon is the only arm with a different
#  `datawidth`. So each depth costs 2 maps: ECC_WEIGHT_DATAWIDTH empty (the
#  8-bit arm) and ECC_WEIGHT_DATAWIDTH=<q> (the reduced arm), on IDENTICAL
#  `width:` and `depth:`.
#
#  WHY DATAWIDTH AND NOT DEPTH x N/K. Verified 2026-09-10 from
#  timeloop-mapper.accelergy.log: CACTI receives `depth` and `width` ONLY --
#  `datawidth` never reaches the energy model -- and Timeloop bills
#  `vector_access_energy / block_size` per weight, `block_size =
#  width/datawidth`. So the two arms get identical per-access read/write/leak
#  and the reduced arm simply rides more weights per word. Expressing the same
#  thing as `depth x N/K` (the previous sweep) made Accelergy price the recon
#  arm's array 1.18-1.46x DEARER, so the optimiser had a reason to leave the
#  room unused -- FINDINGS 7.8, and the reason that sweep proved nothing.
#
#  THE LADDER IS sqrt(2), NOT 2. The window where Embedded cannot hold the tile
#  and Recon can is exactly as wide, in depth, as the effective-capacity ratio.
#  A factor-2 grid steps clean over a 1.17x or 1.33x window and reports a grid
#  artifact as "no effect". x0.5 / x0.25 / x0.125 -- the depths a YAML quotes --
#  are QUOTED OUT OF the sqrt(2) ladder, not searched on.
#
#  THE SECOND PASS RUNS IN THE SAME WAVE. ECC_WEIGHT_DEPTH_SCALE moves EVERY
#  weight level at once, which locates the zone but cannot say WHICH level
#  bought it -- and the deliverable is a `depth:` PER LEVEL. So the wave also
#  sweeps each weight level with the OTHER held at x1. FINDINGS 7.8 predicts
#  `filter_glb` is the one that matters: refetch on v1 is set by the DRAM-level
#  loop order over P/Q, a weight tile cannot index either, so a buffer INSIDE
#  the array can never absorb those loops however large (measured flat to x32
#  at 1 % fill) and only a level ABOVE the array can. CONFIRM it, do not assume
#  it -- which is why both levels are swept, not just the predicted one.
#  --no-per-level submits the joint pass alone.
#
#  THE CONVERGENCE GATE RUNS TOO, and it is a GATE: no number is quoted until
#  it passes. It maps the EMBEDDED arm at ECC_DEPTH_SWEEP_GATE_VICTORIES on
#  ECC_DEPTH_SWEEP_GATE_SCALES (the largest AND the smallest depth -- a budget
#  that converges on a big buffer may not on a small one). Converged means the
#  last two agree within a margin you STATE, and that margin must be smaller
#  than the Recon-vs-Embedded effect being claimed. Read it with
#      bash hpc/tl.sh python3 -m eccenergy.report.dilation_view --gate
#
#  NO KNOBS ARE DEFINED HERE. Every value comes from ../env.sh (section 5 for
#  the sweep, section 2 for the search budget, section 8 for SLURM), so a value
#  cannot mean one thing to this script and another to the mapper.
#
#  There is deliberately NO dependent evaluation: the sweep is a property of
#  the MAPPINGS, and an eval job here would draw a Task 3 figure over it. Read
#  the result with
#      bash hpc/tl.sh python3 -m eccenergy.report.dilation_view --levels \
#           --csv results/tables/EyerissV1_mem_arch_sweep.csv
#
#  Structure is hpc/map_capacity_sweep.sh's, deliberately: one job per unit of
#  work, hpc/map.sbatch unchanged, the SLURM task file snapshotted per
#  submission because map.sbatch resolves its (arch, model) pair from it when
#  the job RUNS, and unknown flags fatal rather than falling through to a
#  submission.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WANT_PROGRESS=0
DRY=0
WANT_SWEEP=1
WANT_GATE=1
WANT_PERLEVEL=1
ALL_SHAPES=0
for a in "$@"; do
    case "$a" in
        --progress)  WANT_PROGRESS=1 ;;
        --dry-run)   DRY=1 ;;
        --gate-only) WANT_SWEEP=0 ;;
        --no-gate)   WANT_GATE=0 ;;
        --all-shapes) ALL_SHAPES=1 ;;
        --no-per-level) WANT_PERLEVEL=0 ;;
        *) echo "map_depth_sweep.sh: unknown option '$a'" >&2
           echo "  options: --progress  --dry-run  --gate-only  --no-gate" >&2
           echo "           --all-shapes  --no-per-level" >&2
           exit 2 ;;
    esac
done
set --

source ./env.sh

# ---- what the two arms are -------------------------------------------------
# The reduced arm's datawidth. env.sh may pin it; otherwise it comes from
# eccenergy/physics/widths.py -- round(ECC_WEIGHT_BITS * K/N), which is 4 at
# BCH(63,30), the value that needs no width change at all on Eyeriss v1's
# 16-bit and 64-bit weight words.
#
# IT USED TO BE `ceil` HERE AND round() IN THE TABLE, and the two disagreed at
# exactly one code: BCH(63,57), where ceil gives 8. The reconstruction arm would
# then have declared the SAME datawidth as the embedded arm and the on-chip
# treatment would have been a silent no-op reported as a result. One derivation,
# in one place, so they cannot drift again.
RECON_DW="${ECC_DEPTH_SWEEP_RECON_DW}"
if [ -z "${RECON_DW}" ]; then
    RECON_DW=$(python3 -c "
from eccenergy.physics import widths as code_widths
print(code_widths.declared_datawidth(${ECC_CODE_N}, ${ECC_CONST_K}, ${ECC_WEIGHT_BITS}))")
fi

MODEL="$(set -- ${ECC_MODELS}; echo "$1")"
[ "$(set -- ${ECC_MODELS}; echo $#)" = 1 ] || {
    echo "map_depth_sweep.sh: exactly ONE model, got ECC_MODELS='${ECC_MODELS}'" >&2
    exit 2; }
# ONE JOB PER LAYER -- and with ECC_LAYERS EMPTY, one job per distinct layer
# SHAPE of the model instead of one per layer name. The mapper cache is keyed
# by SHAPE, so the 21 layers of resnet18 are only 12 pieces of work: four
# layers share C64_M64_R3_S3_P56_Q56_ws1_hs1 and three share each of three
# more. Submitting per layer name would map the same shape four times and take
# the other three as cache hits after waiting for a lock -- the same waste
# hpc/map_by_shape.sh exists to avoid.
# `ECC_LAYERS=` cannot express "whole model": env.sh writes every knob as
# ${VAR:=default}, and `:=` refills a NULL value as readily as an unset one,
# so an empty override comes back as the section-1 default. --all-shapes is
# the flag that says it.
if [ "${ALL_SHAPES}" = "0" ] && [ -n "${ECC_LAYERS}" ]; then
    read -r -a LAYERS <<< "${ECC_LAYERS}"
else
    read -r -a LAYERS <<< "$(python3 -c "
import dataclasses
from eccenergy import config
from eccenergy.arch import workloads
cfg = dataclasses.replace(config.load_config(), layers=[])
models, _ = workloads.load_workload(cfg)
seen, out = set(), []
for l in models[cfg.models[0]]:
    if l.shape_name not in seen:
        seen.add(l.shape_name)
        out.append(l.name)          # one representative layer name per shape
print(' '.join(out))")"
    echo "map_depth_sweep.sh: --all-shapes -> WHOLE MODEL, one job per distinct"
    echo "  SHAPE (${#LAYERS[@]} of them, from $(set -- ${ECC_MODELS}; echo $1)'s 21 layers):"
    echo "  ${LAYERS[*]}"
fi
[ "${#LAYERS[@]}" -gt 0 ] || {
    echo "map_depth_sweep.sh: no layers resolved." >&2; exit 2; }

# THE GATE IS ONE LAYER, ALWAYS. It asks whether the SEARCH BUDGET can resolve
# the effect, and that is answered on the layer whose number is being claimed
# -- prompt_2 re-checks it at the smallest and largest DEPTH, not on every
# shape in the model. Running it 12 times would be 44 extra maps saying the
# same thing about the budget.
read -r -a GATE_LAYERS <<< "${ECC_LAYERS}"
[ "${#GATE_LAYERS[@]}" -gt 0 ] || GATE_LAYERS=("${LAYERS[0]}")

# THE SECOND PASS: each weight level swept with the other held at x1. The level
# names come from the ARCHITECTURE, not from a list here, so a design with
# three weight levels gets three passes without editing this file. Skipped
# entirely on a design with only one weight level -- there the joint pass IS
# the per-level pass, and submitting it again would just re-map the same
# fingerprints.
read -r -a PERLEVEL <<< "$(python3 -c "
from eccenergy import config
from eccenergy.report import dilation_view as dilation
cfg = config.load_config()
levels = dilation.weight_levels_of(cfg.archs[0], cfg)
print(' '.join(levels) if len(levels) > 1 else '')")"
[ "${#PERLEVEL[@]}" -gt 1 ] || { WANT_PERLEVEL=0; PERLEVEL=(); }
# ECC_WEIGHT_DEPTH_LEVELS already pinned by the caller means the caller IS
# running one per-level pass by hand; do not fan it out again.
[ -z "${ECC_WEIGHT_DEPTH_LEVELS}" ] || { WANT_PERLEVEL=0; PERLEVEL=(); }

# ---- refuse a configuration that would double-count -----------------------
# The mapper now delivers the on-chip narrowing, so ECC_RECON_PACKING=stream
# would scale that same saving by K/N a SECOND time in the evaluator. Caught
# here as well as in recon.assert_onchip_narrowing_once(), because a wave of
# 18 jobs is expensive to discover it after.
[ "${ECC_RECON_PACKING}" = "aligned" ] || {
    echo "map_depth_sweep.sh: ECC_RECON_PACKING=${ECC_RECON_PACKING}, not aligned." >&2
    echo "  The MAPPER now applies the on-chip narrowing (ECC_WEIGHT_DATAWIDTH)." >&2
    echo "  With 'stream' the evaluator applies it AGAIN and the on-chip saving" >&2
    echo "  is SQUARED. prompt_2's resolution is aligned. Refusing." >&2
    exit 2; }
[ "${ECC_OPT_METRIC}" = "edp" ] || {
    echo "map_depth_sweep.sh: ECC_OPT_METRIC=${ECC_OPT_METRIC}, not edp." >&2
    echo "  'energy' serialises and trades PEs for capacity, which is what" >&2
    echo "  withdrew FINDINGS 7.7. Refusing." >&2
    exit 2; }

# ---- validate the geometry BEFORE queueing anything ------------------------
# timeloop-mapper ABORTS on `width % datawidth != 0` (buffer.cpp:302, measured
# exit=134). One python call now beats 14 core dumps in an hour.
python3 - "${RECON_DW}" <<'PY' || exit 2
import dataclasses, sys
from eccenergy import config
from eccenergy.arch import patch
dw = int(sys.argv[1])
cfg = config.load_config()
scales = [round(float(s), 4) for s in
          __import__("os").environ["ECC_DEPTH_SWEEP_SCALES"].split()]
print(f"  geometry check: recon datawidth = {dw}b")
for arch in cfg.archs:
    for s in scales:
        emb = dataclasses.replace(cfg, weight_depth_scale=s, weight_datawidth=None)
        rec = dataclasses.replace(cfg, weight_depth_scale=s, weight_datawidth=dw)
        try:
            info = patch.assert_pair_geometry(arch, emb, rec)
        except ValueError as e:
            print(f"  REFUSED  {arch} x{s:g}: {e}", file=sys.stderr)
            sys.exit(1)
        cells = "  ".join(
            f"{lvl} d{v['depth']} w{v['embedded_width']}/{v['recon_width']} "
            f"{v['embedded_weights']}->{v['recon_weights']} "
            f"({v['capacity_ratio']:.4f}x)" for lvl, v in info.items())
        print(f"  {arch:<20} x{s:<7g} {cells}")
PY

# ---- the jobs --------------------------------------------------------------
mkdir -p hpc/.runtime hpc/logs
ecc_write_taskfile
SNAP="hpc/.runtime/tasks.depthsweep.$$.txt"
cp "${ECC_TASKFILE}" "${SNAP}"
NARCH="$(set -- ${ECC_ARCHS}; echo $#)"
NLINES=$(grep -cve '^[[:space:]]*$' "${SNAP}")
[ "${NLINES}" = "${NARCH}" ] || {
    echo "map_depth_sweep.sh: task file has ${NLINES} line(s) for ${NARCH}" \
         "architecture(s) -- refusing rather than mapping the wrong pair" >&2; exit 2; }

NSCALE="$(set -- ${ECC_DEPTH_SWEEP_SCALES}; echo $#)"
NSWEEP=$([ "${WANT_SWEEP}" = 1 ] && echo $((NARCH * ${#LAYERS[@]} * NSCALE * 2)) || echo 0)
NGATES="$(set -- ${ECC_DEPTH_SWEEP_GATE_SCALES}; echo $#)"
# The sweep already maps the embedded arm at ECC_VICTORY, so that point of the
# gate is a cache hit and is not counted or resubmitted.
NGATEV=0
for V in ${ECC_DEPTH_SWEEP_GATE_VICTORIES}; do
    [ "${V}" = "${ECC_VICTORY}" ] && [ "${WANT_SWEEP}" = "1" ] && continue
    NGATEV=$((NGATEV + 1))
done
NGATE=$([ "${WANT_GATE}" = 1 ] && echo $((NARCH * ${#GATE_LAYERS[@]} * NGATEV * NGATES)) || echo 0)
# x1 is a no-op for a single named level -- it resolves to the SAME cache as
# the joint pass's x1 -- so the per-level passes skip it.
NPL=0
for S in ${ECC_DEPTH_SWEEP_SCALES}; do
    [ "$(python3 -c "print(1 if float('${S}')==1.0 else 0)")" = "1" ] && continue
    NPL=$((NPL + 1))
done
NPERLEVEL=$([ "${WANT_PERLEVEL}" = 1 ] \
    && echo $((NARCH * ${#LAYERS[@]} * NPL * ${#PERLEVEL[@]} * 2)) || echo 0)

echo "map_depth_sweep: ${ECC_ARCHS}   model: ${MODEL}"
echo "  layers    : ${#LAYERS[@]} -- ${LAYERS[*]}"
echo "  code      : BCH(${ECC_CODE_N},${ECC_CONST_K})  -> recon datawidth ${RECON_DW}b" \
     "vs the 8-bit baseline/embedded arm"
echo "  depths    : ${ECC_DEPTH_SWEEP_SCALES}" \
     "${ECC_WEIGHT_DEPTH_LEVELS:+  (levels: ${ECC_WEIGHT_DEPTH_LEVELS})}"
echo "  objective : ${ECC_OPT_METRIC}  victory=${ECC_VICTORY}" \
     "alg=${ECC_MAPPER_ALGORITHM}  threads=${ECC_MAPPER_THREADS}  qos=${ECC_QOS}"
echo "  gate      : victories {${ECC_DEPTH_SWEEP_GATE_VICTORIES}}" \
     "at depths {${ECC_DEPTH_SWEEP_GATE_SCALES}}, EMBEDDED arm only," \
     "on ${GATE_LAYERS[*]}"
if [ "${WANT_PERLEVEL}" = "1" ]; then
    echo "  2nd pass  : each of {${PERLEVEL[*]}} swept with the other(s) at x1," \
         "so the YAML gets a depth PER LEVEL"
else
    echo "  2nd pass  : none (single weight level, --no-per-level, or" \
         "ECC_WEIGHT_DEPTH_LEVELS already pinned)"
fi
echo "  task file : ${SNAP}"
echo "  NO dependent eval is submitted. Read the result with"
echo "      bash hpc/tl.sh python3 -m eccenergy.report.dilation_view --levels \\"
echo "           --csv results/tables/EyerissV1_mem_arch_sweep.csv"

if [ "${WANT_PROGRESS}" = "1" ]; then
    echo
    for A in ${ECC_ARCHS}; do
        for S in ${ECC_DEPTH_SWEEP_SCALES}; do
            for DW in "" "${RECON_DW}"; do
                OUT=$(ECC_WEIGHT_DEPTH_SCALE="${S}" ECC_WEIGHT_DATAWIDTH="${DW}" \
                      ECC_SWEEP=arch ECC_SWEEP_ARCHS="${A}" \
                      python3 -c "
import pathlib
from eccenergy import config, paths
from eccenergy.arch import fingerprint
cfg = config.load_config()
a = cfg.archs[0]
d = pathlib.Path(paths.Results(cfg).mapper_cache(a, fingerprint.effective_variant(a, cfg),
                                                 fingerprint.arch_fingerprint(a, cfg),
                                                 create=False))
n = len(list(d.glob('*/timeloop-mapper.stats.txt'))) if d.is_dir() else 0
k = len(list(d.glob('*.lock'))) if d.is_dir() else 0
print(f'{n} {k} {d.name}')" 2>/dev/null | tail -1)
                printf "  %-22s x%-8s %-9s solved=%-3s locked=%-3s %s\n" "${A}" "${S}" \
                       "$([ -z "${DW}" ] && echo 'emb(8b)' || echo "rec(${DW}b)")" \
                       "$(echo "${OUT}" | awk '{print $1}')" \
                       "$(echo "${OUT}" | awk '{print $2}')" \
                       "$(echo "${OUT}" | awk '{print $3}')"
            done
        done
    done
    exit 0
fi

if [ "${DRY}" = "1" ]; then
    echo "  --dry-run: would submit ${NSWEEP} joint + ${NPERLEVEL} per-level" \
         "+ ${NGATE} gate = $((NSWEEP + NPERLEVEL + NGATE)) job(s). Nothing submitted."
    exit 0
fi

JOBS=()
submit () {   # submit <arch-index> <name> <layer> <extra --export pairs...>
    local idx="$1"; shift
    local name="$1"; shift
    local layer="$1"; shift
    local extra="$1"; shift
    local jid
    jid=$(sbatch --parsable \
        --job-name="${name}" \
        --account="${ECC_ACCOUNT}" --qos="${ECC_QOS}" --partition="${ECC_PARTITION}" \
        --cpus-per-task="${ECC_MAP_CPUS}" --mem="${ECC_MAP_MEM}" --time="${ECC_MAP_TIME}" \
        --array="${idx}-${idx}" \
        --output="hpc/logs/depth.%A.out" \
        --export=ALL,ECC_LAYERS="${layer}",ECC_TASKFILE="${PWD}/${SNAP}",${extra} \
        hpc/map.sbatch)
    JOBS+=("${jid}")
    echo "  job ${jid}  ${name}"
}

I=0
for A in ${ECC_ARCHS}; do
    if [ "${WANT_SWEEP}" = "1" ]; then
        for S in ${ECC_DEPTH_SWEEP_SCALES}; do
            for L in "${LAYERS[@]}"; do
                # the 8-bit arm: baseline AND embedded, one mapping
                submit "${I}" "d-${A}-${S}-emb-${L}" "${L}" \
                       "ECC_WEIGHT_DEPTH_SCALE=${S},ECC_WEIGHT_DATAWIDTH="
                # the reduced arm: same width, same depth, narrower word
                submit "${I}" "d-${A}-${S}-rec${RECON_DW}-${L}" "${L}" \
                       "ECC_WEIGHT_DEPTH_SCALE=${S},ECC_WEIGHT_DATAWIDTH=${RECON_DW}"
            done
        done
    fi
    if [ "${WANT_PERLEVEL}" = "1" ]; then
        for LV in "${PERLEVEL[@]}"; do
            for S in ${ECC_DEPTH_SWEEP_SCALES}; do
                # x1 on one named level is the joint pass's x1 -- same cache.
                [ "$(python3 -c "print(1 if float('${S}')==1.0 else 0)")" = "1" ] \
                    && continue
                for L in "${LAYERS[@]}"; do
                    submit "${I}" "p-${A}-${LV}-${S}-emb-${L}" "${L}" \
                        "ECC_WEIGHT_DEPTH_SCALE=${S},ECC_WEIGHT_DATAWIDTH=,ECC_WEIGHT_DEPTH_LEVELS=${LV}"
                    submit "${I}" "p-${A}-${LV}-${S}-rec${RECON_DW}-${L}" "${L}" \
                        "ECC_WEIGHT_DEPTH_SCALE=${S},ECC_WEIGHT_DATAWIDTH=${RECON_DW},ECC_WEIGHT_DEPTH_LEVELS=${LV}"
                done
            done
        done
    fi
    if [ "${WANT_GATE}" = "1" ]; then
        for V in ${ECC_DEPTH_SWEEP_GATE_VICTORIES}; do
            # The sweep already maps the embedded arm at ECC_VICTORY, so that
            # point of the gate is a cache HIT and is not resubmitted.
            [ "${V}" = "${ECC_VICTORY}" ] && [ "${WANT_SWEEP}" = "1" ] && continue
            for S in ${ECC_DEPTH_SWEEP_GATE_SCALES}; do
                for L in "${GATE_LAYERS[@]}"; do
                    submit "${I}" "g-${A}-v${V}-${S}-${L}" "${L}" \
                           "ECC_WEIGHT_DEPTH_SCALE=${S},ECC_WEIGHT_DATAWIDTH=,ECC_VICTORY=${V}"
                done
            done
        done
    fi
    I=$((I + 1))
done

DEP=$(IFS=:; echo "${JOBS[*]}")
echo "depthsweep_jobs=${DEP}" > hpc/.runtime/map_depth_sweep.last
echo "  ${#JOBS[@]} jobs submitted. Watch:  bash hpc/map_depth_sweep.sh --progress"
