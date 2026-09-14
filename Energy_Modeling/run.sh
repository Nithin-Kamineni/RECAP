#!/usr/bin/env bash
# =============================================================================
#  run.sh  --  ONE STAGE OF THE PIPELINE
# =============================================================================
#
#  THERE ARE NO KNOBS IN THIS FILE. Every variable lives in ./env.sh, which this
#  script sources; the environment still wins over it, so a one-off costs
#  nothing:
#
#      ECC_CONST_MODEL=resnet50 bash run.sh baseline --eval
#
#  For the WHOLE pipeline -- map, then evaluate, then plot -- use the one
#  command instead, which drives every stage below from the same env.sh:
#
#      bash hpc/run_all.sh
#
#  THE STAGES
#      bash run.sh                 the figure for ECC_SWEEP's axis
#      bash run.sh map             solve and cache mappings, evaluate nothing.
#                                  The only stage that needs the container.
#      bash run.sh baseline --eval Task 1: conventional ECC, external parity
#      bash run.sh embedded --eval Task 2: embedded ECC beside that baseline,
#                                  from the same mappings; only DRAM differs
#      bash run.sh recon --eval    Task 3: the reconstruction PLACEMENT study on
#                                  ONE architecture, from the same mappings. The
#                                  x axis is WHERE the boundary sits; the mapper
#                                  is never re-run. Configure the point in
#                                  env.sh section 4, or just set
#                                  ECC_RECON_MODELING=1 and let the one command
#                                  route everything to it.
#                                  RECON_OPTIMIZER=True
#                                  makes it TASK 4 instead: the reconstruction
#                                  bars are re-derived from a SECOND mapping
#                                  solved against N/K more weight room, so the
#                                  two arms no longer refetch identically. It
#                                  needs that second mapper cache and refuses
#                                  rather than falling back. -> ReconSweep_optimiser
#      bash run.sh dilation        Task 4 STEP 1: diff the DRAM weight reads of
#                                  two mapper caches at different weight
#                                  capacities, so the capacity effect is
#                                  measured before it is modelled. Maps nothing,
#                                  draws nothing. Fill the caches with
#                                  hpc/map_capacity_sweep.sh first.
#      bash run.sh panels --eval   one image, one panel per ECC_PANEL_MODELS
#      bash run.sh validate        check the architectures against the shared
#                                  comparison contract. No container needed.
#      bash run.sh diagnose        audit the architectures, no figure
#      bash run.sh --replot        figure only, from results/_raw/
#      bash run.sh --dry-run       resolve and print the config, then stop
#
#  `--eval` (ECC_FROM_CACHE=1) never invokes Timeloop and needs no container.
#  Mapping and evaluation are separate so that an energy edit can never silently
#  change a mapping.
#
#  Flags that override env.sh for one invocation, for interactive use:
#      bash run.sh --sweep model --arch simba_like --k 45
#      bash run.sh --sweep bch --values "57 45 30" --replot
#      bash run.sh --layers "layer4.1.conv2" map
#
#  On HiPerGator every one of these goes through the container wrapper:
#      module load apptainer
#      bash hpc/tl.sh bash run.sh validate
#  and anything that invokes the mapper must be inside an allocation, never on
#  a login node.
# =============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

exec "${ECC_PYTHON:-python3}" -m eccenergy "$@"
