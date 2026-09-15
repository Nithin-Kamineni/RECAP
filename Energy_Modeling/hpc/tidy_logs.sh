#!/usr/bin/env bash
# Move finished SLURM job output out of hpc/logs/ into hpc/old-logs/, so that
# hpc/logs/ holds only the jobs that are still in the queue. Then
#     tail -f hpc/logs/*.out
# follows the current run instead of four thousand dead ones.
#
#     bash hpc/tidy_logs.sh              # move finished logs
#     bash hpc/tidy_logs.sh --dry-run    # list what would move, move nothing
#
# What counts as "still running": a log is KEPT when the job id in its filename
# is in `squeue` in any state, PENDING included. A pending array task has no log
# yet but will open one under the same base id once it starts -- keying on the
# base id rather than on the files present is what keeps its future log in
# hpc/logs/ instead of stranding it here on the next tidy.
#
# Only *.out and *.err move. The hand-kept bring-up evidence
# (hpc/logs/bringup-*.log) is tracked in git, cited by README.md and
# legacy/FINDINGS_detail_2026-09-07.md, and deliberately not matched by the
# repository-root .gitignore; it stays put.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="${HERE}/logs"
OLD="${HERE}/old-logs"

DRY_RUN=0
QUIET=0
case "${1:-}" in
    --dry-run|-n) DRY_RUN=1 ;;
    # --quiet is what hpc/run_all.sh uses when it sweeps on the way past: a
    # janitor that chatters on every invocation trains you to ignore the output
    # of the script it is attached to.
    --quiet|-q)   QUIET=1 ;;
    "")           ;;
    *) echo "tidy_logs: unknown argument '$1' (expected --dry-run or --quiet)" >&2; exit 2 ;;
esac

say() { [ "${QUIET}" = 1 ] || echo "$@"; }

[[ -d "${LOGS}" ]] || { echo "tidy_logs: no ${LOGS}"; exit 0; }
mkdir -p "${OLD}"

# Active base job ids, every state. If squeue is unavailable or errors we abort
# rather than fall back to an empty set: an empty set would read as "nothing is
# running" and sweep the live run's logs out from under it.
if ! command -v squeue >/dev/null 2>&1; then
    echo "tidy_logs: no squeue on PATH -- refusing to move anything" >&2
    exit 1
fi
if ! ACTIVE="$(squeue -u "${USER}" -h -o '%A' | sort -u)"; then
    echo "tidy_logs: squeue failed -- refusing to move anything" >&2
    exit 1
fi

active_count="$(printf '%s' "${ACTIVE}" | grep -c . || true)"
say "tidy_logs: ${active_count} job id(s) in the queue; their logs stay in hpc/logs/"

moved=0
kept=0
while IFS= read -r -d '' path; do
    name="$(basename "${path}")"
    # ecc-map.42244795_7.out, ecc-eval.42244796.out, ecc-map-41269118_10.out:
    # the job id is the last run of digits before an optional _<task> and the
    # extension, after either a '.' or a '-' separator.
    id="$(printf '%s' "${name}" | sed -nE 's/^.*[.-]([0-9]+)(_[0-9]+)?\.(out|err)$/\1/p')"
    if [[ -n "${id}" ]] && grep -qx "${id}" <<<"${ACTIVE}"; then
        kept=$(( kept + 1 ))
        continue
    fi
    if (( DRY_RUN )); then
        echo "would move  ${name}"
    elif [[ -e "${OLD}/${name}" ]]; then
        say "tidy_logs: ${name} already in old-logs/ -- left in place"
        continue
    elif ! mv -- "${path}" "${OLD}/${name}" 2>/dev/null; then
        # Lost a race with a concurrent sweep, or the file went away. Never a
        # reason to fail: this is a janitor attached to a submission, and a
        # tidy that errors must not take the run down with it.
        [[ -e "${path}" ]] && say "tidy_logs: could not move ${name} -- left in place"
        continue
    fi
    moved=$(( moved + 1 ))
done < <(find "${LOGS}" -maxdepth 1 -type f \( -name '*.out' -o -name '*.err' \) -print0)

if (( DRY_RUN )); then
    echo "tidy_logs: would move ${moved}, would keep ${kept} (dry run, nothing changed)"
elif [ "${QUIET}" = 1 ]; then
    # One line, and only when there was something to do. A plain `[ ] && echo`
    # here would return 1 when nothing moved and `set -e` would take the script
    # -- and with it the run that called it -- down on the quiet path.
    if [ "${moved}" -gt 0 ]; then
        echo "tidy_logs: swept ${moved} finished log(s) into hpc/old-logs/"
    fi
else
    echo "tidy_logs: moved ${moved} to hpc/old-logs/, kept ${kept} in hpc/logs/"
fi
exit 0
