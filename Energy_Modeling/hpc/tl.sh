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
#  ECC_SIF (env.sh, section 7) is the image path. Anything that invokes the mapper must be
#  run inside an allocation (srun/sbatch), never on a login node -- see
#  hpc/HIPERGATOR.md.
# =============================================================================
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# env.sh owns every path and knob, ECC_SIF included, and exports them into the
# container. Sourcing it here means `bash hpc/tl.sh ...` from a bare shell sees
# the same configuration as the pipeline does.
source "$PROJ/env.sh"
SIF="${ECC_SIF}"

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

# NEUROSIM needs the same treatment, and not having it has been costing real
# numbers (legacy/FINDINGS_detail_2026-09-11.md defect 7; prompt_7 section 11b).
# Accelergy asks every plug-in to bid on each component and takes the most
# confident answer. For the `intadder` inside every smartbuffer's address
# generator, Neurosim ties Aladdin at accuracy 70 and sometimes wins -- but it
# writes `neurosim_input_<pid>.cfg` into its OWN plug-in directory, which is
# read-only in a SIF. It dies with `OSError: [Errno 30] Read-only file system`,
# reports 0 pJ at accuracy 70%, and Accelergy accepts that as the price. The
# damage is visible in any cached ERT: the same adder costs 0.0853 pJ on a
# write and 0 pJ on a read.
#
# Unlike CACTI there is no separate scratch subdirectory to bind -- the file
# lands in the plug-in root -- so bind a writable COPY of the whole directory,
# extracted from the image once. If the copy is missing or empty we bind
# NOTHING rather than masking the plug-in with an empty directory, which would
# turn a wrong number into no plug-in at all.
NEUROSIM_IN_SIF="/usr/local/share/accelergy/estimation_plug_ins/accelergy-neurosim-plugin"
NEUROSIM_RW="$PROJ/hpc/.runtime/neurosim-plugin"
if [ ! -e "$NEUROSIM_RW/.populated" ]; then
    mkdir -p "$NEUROSIM_RW"
    if apptainer exec "$SIF" test -d "$NEUROSIM_IN_SIF" 2>/dev/null; then
        apptainer exec --bind "$NEUROSIM_RW:/mnt/out" "$SIF" \
            cp -a "$NEUROSIM_IN_SIF/." /mnt/out/ 2>/dev/null \
            && touch "$NEUROSIM_RW/.populated"
    fi
fi
NEUROSIM_BIND=()
if [ -e "$NEUROSIM_RW/.populated" ]; then
    NEUROSIM_BIND=(--bind "$NEUROSIM_RW:$NEUROSIM_IN_SIF")
else
    echo "tl.sh: WARNING -- Neurosim plug-in not made writable; its address-generator" >&2
    echo "       estimates will be 0 pJ (prompt_7 section 11b)." >&2
fi

exec apptainer exec --bind /blue \
    --bind "$CACTI_SCRATCH:/usr/local/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/cacti_inputs_outputs" \
    "${NEUROSIM_BIND[@]}" \
    --pwd "$PROJ" "$SIF" "$@"
