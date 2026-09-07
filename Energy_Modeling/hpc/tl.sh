#!/usr/bin/env bash
# =============================================================================
#  hpc/tl.sh -- run a command inside the Timeloop+Accelergy image on HiPerGator
#
#      cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
#      module load apptainer
#      bash hpc/tl.sh bash run.sh validate
#      ECC_LAYERS="layer3.0.downsample.0" bash hpc/tl.sh bash run.sh map
#
#  This is the HiPerGator equivalent of the Docker one-liner in CLAUDE.md.
#  Apptainer passes the caller's environment through, so every ECC_* knob set
#  in the shell reaches run.sh unchanged. /blue is bound so the project and
#  the caches are visible; the working directory inside the container is the
#  project root, whatever directory this is called from.
#
#  ECC_SIF overrides the image path. Anything that invokes the mapper must be
#  run inside an allocation (srun/sbatch), never on a login node -- see
#  hpc/HIPERGATOR.md.
# =============================================================================
set -euo pipefail

SIF="${ECC_SIF:-/blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling/timeloop.sif}"
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "$SIF" ]; then
    echo "hpc/tl.sh: image not found: $SIF" >&2
    echo "  -> build it once with 'apptainer pull' (hpc/HIPERGATOR.md, step 3)," >&2
    echo "     or point ECC_SIF at an existing .sif" >&2
    exit 1
fi

if ! command -v apptainer >/dev/null 2>&1; then
    # `module` is a shell function; it exists in interactive and login shells
    # and in sbatch scripts, but not necessarily in a bare `bash script.sh`.
    if type module >/dev/null 2>&1; then
        module load apptainer
    else
        echo "hpc/tl.sh: apptainer is not on PATH; run 'module load apptainer' first" >&2
        exit 1
    fi
fi

if [ $# -eq 0 ]; then
    echo "usage: bash hpc/tl.sh <command> [args...]      e.g.  bash hpc/tl.sh bash run.sh validate" >&2
    exit 2
fi

# CACTI writes scratch inputs/outputs next to its installed Python plugin.
# A SIF is read-only; bind just that scratch directory to writable /blue storage.
CACTI_SCRATCH="$PROJ/hpc/.runtime/cacti_inputs_outputs"
mkdir -p "$CACTI_SCRATCH"
exec apptainer exec --bind /blue \
    --bind "$CACTI_SCRATCH:/usr/local/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/cacti_inputs_outputs" \
    --pwd "$PROJ" "$SIF" "$@"
