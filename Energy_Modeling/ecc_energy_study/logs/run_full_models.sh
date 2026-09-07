#!/usr/bin/env bash
# Full-model mapping for resnet18 and mobilenet_v2 on all seven architectures
# at the CONVERGED search setting, then the two figures. Run inside the
# container from /home/workspace:
#
#     bash ecc_energy_study/logs/run_full_models.sh
#
# Resumable: every solved shape is cached immediately, so re-running after an
# interruption continues where it stopped. Waits for the representative-layer
# run of 2026-09-06 to finish first so the two never compete for the cores.
set -uo pipefail
cd /home/workspace
LOG=ecc_energy_study/logs

export ECC_SWEEP=arch
export ECC_SWEEP_ARCHS="eyeriss_v2_like eyeriss_like eyeriss_like_wglb simple_weight_stationary simple_output_stationary simple_input_stationary simba_like"
export ECC_MAPPER_ALGORITHM=random_pruned
export ECC_MAPPER_SEARCH_SIZE=20000
export ECC_VICTORY=2000
export ECC_MAPPER_TIMEOUT=2000
export ECC_MAPPER_THREADS=18
export ECC_PANEL_MODELS="resnet18 mobilenet_v2"
export ECC_TITLE_NOTE="random_pruned, search_size 20000/thread, victory 2000 (x2 on 9-level designs), 18 threads; grouped depthwise shapes"
export ECC_RUN_NOTE="2026-09-06 full-model re-map at converged search settings, after the depthwise fix"

# 1. wait for the representative-layer job (same settings, subset of shapes)
until [ -f "$LOG/map_mobilenet_v2.exit" ]; do sleep 30; done

echo "[$(date)] full resnet18 map starting" >> "$LOG/run_full_models.progress"
ECC_CONST_MODEL=resnet18 bash run.sh map > "$LOG/full_map_resnet18.log" 2>&1
echo "$?" > "$LOG/full_map_resnet18.exit"
echo "[$(date)] full resnet18 map done, exit $(cat "$LOG/full_map_resnet18.exit")" >> "$LOG/run_full_models.progress"

# single-model figure for resnet18 as soon as it exists (everything cached now)
ECC_CONST_MODEL=resnet18 bash run.sh > "$LOG/full_sweep_resnet18.log" 2>&1
echo "[$(date)] ArchitectureSweep (resnet18) drawn, exit $?" >> "$LOG/run_full_models.progress"

echo "[$(date)] full mobilenet_v2 map starting" >> "$LOG/run_full_models.progress"
ECC_CONST_MODEL=mobilenet_v2 bash run.sh map > "$LOG/full_map_mobilenet_v2.log" 2>&1
echo "$?" > "$LOG/full_map_mobilenet_v2.exit"
echo "[$(date)] full mobilenet_v2 map done, exit $(cat "$LOG/full_map_mobilenet_v2.exit")" >> "$LOG/run_full_models.progress"

# 2. the panel figure, one panel per model, everything from the cache
bash run.sh panels > "$LOG/full_panels.log" 2>&1
echo "$?" > "$LOG/full_panels.exit"
echo "[$(date)] panels drawn, exit $(cat "$LOG/full_panels.exit")" >> "$LOG/run_full_models.progress"
