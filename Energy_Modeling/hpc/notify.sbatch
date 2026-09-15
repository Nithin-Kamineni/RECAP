#!/bin/bash
# =============================================================================
#  ONE MAIL PER SUBMISSION, sent after every job hpc/run_all.sh launched has
#  terminated. Submitted by run_all.sh when ECC_MAIL_ON_DONE=1 (env.sh section
#  5, default 0); never run by hand.
#
#  WHY A JOB AND NOT `sbatch --mail-type=END`: --mail-type fires once per array
#  TASK, so a 12-task sweep mails you twelve times, and its body is SLURM's
#  fixed one-liner -- there is no way to put ECC_SWEEP or ECC_ARCHS in it. This
#  job is held on `--dependency=afterany:<map>:<eval>`, so it runs exactly once,
#  after everything, whatever everything did.
#
#  WHY THE CONFIGURATION IS READ FROM THE ENVIRONMENT AND NOT FROM env.sh: the
#  mail must describe the run as SUBMITTED, not as env.sh reads hours later when
#  this job finally starts. run_all.sh submits with --export=ALL, which snapshots
#  its shell -- env.sh already sourced -- into this job. Sourcing env.sh here
#  would report whatever the file says now, which is the one thing that would
#  make the mail lie.
# =============================================================================
#SBATCH --job-name=ecc-notify

set -uo pipefail

# The jobs to report on, colon-separated, exported by run_all.sh's submit_notify.
JOBS="${ECC_NOTIFY_JOBS:-}"
LABEL="${ECC_NOTIFY_LABEL:-run}"
TO="${ECC_MAIL_TO:-${USER}@ufl.edu}"

if [ -z "${JOBS}" ]; then
    echo "notify: ECC_NOTIFY_JOBS is empty -- nothing to report on" >&2
    exit 2
fi

ids="${JOBS//:/,}"

# sacct's accounting record can lag the job's exit by a few seconds. Wait for
# every row to reach a terminal state rather than mailing "RUNNING" for a job
# that has in fact finished. Capped: a mail that is slightly wrong beats no mail.
for _ in $(seq 1 30); do
    pending=$(sacct -j "${ids}" -X -n -P -o State 2>/dev/null \
              | grep -cE '^(PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED|COMPLETING)' || true)
    [ "${pending}" = "0" ] && break
    sleep 10
done

TABLE=$(sacct -j "${ids}" -X -n -P -o JobID,JobName,State,Elapsed,ExitCode 2>/dev/null)
if [ -z "${TABLE}" ]; then
    echo "notify: sacct returned nothing for ${ids}" >&2
    TABLE="(sacct returned no rows for ${ids})"
fi

# Anything that is not COMPLETED is a failure worth naming: FAILED, TIMEOUT,
# OUT_OF_MEMORY, NODE_FAIL, and CANCELLED -- which is what a dependent eval job
# becomes when a map task fails and SLURM tears the chain down.
BAD=$(printf '%s\n' "${TABLE}" | awk -F'|' '$3 != "COMPLETED" {print}')
NBAD=$(printf '%s' "${BAD}" | grep -c . || true)
NTOT=$(printf '%s' "${TABLE}" | grep -c . || true)

if [ "${NBAD}" -eq 0 ]; then
    VERDICT="COMPLETED"
    SUBJECT="[RECAP] ${LABEL} COMPLETED -- sweep=${ECC_SWEEP:-?} archs=${ECC_ARCHS:-?}"
else
    VERDICT="FAILED (${NBAD} of ${NTOT} job rows did not COMPLETE)"
    SUBJECT="[RECAP] ${LABEL} FAILED -- sweep=${ECC_SWEEP:-?} archs=${ECC_ARCHS:-?}"
fi

BODY=$(cat <<BODYEOF
${VERDICT}

WHAT WAS RUN  (the configuration as submitted, not as env.sh reads now)
  ECC_SWEEP      : ${ECC_SWEEP:-(unset)}
  ECC_ARCHS      : ${ECC_ARCHS:-(unset)}
  ECC_MODELS     : ${ECC_MODELS:-(unset)}
  ECC_APPROACHES : ${ECC_APPROACHES:-(unset)}

  code           : BCH(${ECC_CODE_N:-?}, ${ECC_KS:-?})
  layer scope    : ${ECC_LAYERS:-whole model}
  figure         : ${ECC_RESULTS_DIR:-results}/figures/${ECC_STEM:-<self-describing>}

JOBS  (JobID|JobName|State|Elapsed|ExitCode)
$(printf '%s\n' "${TABLE}" | sed 's/^/  /')
$(if [ "${NBAD}" -gt 0 ]; then
    printf '\nDID NOT COMPLETE\n'
    printf '%s\n' "${BAD}" | sed 's/^/  /'
    printf '\n  Evaluate whatever the cache already holds with:\n'
    printf '    bash hpc/run_all.sh --eval-only\n'
  fi)

LOGS
  hpc/logs/ecc-map.<jobid>_*.out      the mapping array
  hpc/logs/ecc-eval.<jobid>.out       the evaluation
  hpc/logs/ecc-notify.${SLURM_JOB_ID:-?}.out    this notifier

  Finished jobs' logs are swept into hpc/old-logs/ by hpc/tidy_logs.sh.

--
Sent because ECC_MAIL_ON_DONE=1 in env.sh section 5. Set it to 0 to stop.
$(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname)
BODYEOF
)

# The log gets the exact subject and body BEFORE the send is attempted, so a
# mail that never lands in the inbox is still diagnosable from hpc/logs/.
printf '%s\n' "${SUBJECT}" "${BODY}"

# The last sweep of the run. run_all.sh tidies on its way IN, which cannot
# reach the logs of the submission it is itself starting; this job runs after
# every one of them has terminated, so it is the only place that can retire the
# eval job's own log. Its own log survives -- this job is still in `squeue`.
if [ "${ECC_TIDY_LOGS:-0}" = "1" ] && [ -x "${SLURM_SUBMIT_DIR:-.}/hpc/tidy_logs.sh" ]; then
    (cd "${SLURM_SUBMIT_DIR:-.}" && bash hpc/tidy_logs.sh --quiet) || true
fi

# THE COMPUTE NODES CANNOT SEND MAIL BY THEMSELVES. mail(1) and mailx are not
# installed on them (they exist only on the login nodes), and /usr/sbin/sendmail
# is msmtp with no configuration file -- "account default not found". Both were
# tried, in that order, and both failed after composing a correct message. What
# does work is talking to the site relay directly: ECC_MAIL_SMTP is reachable
# from the nodes on port 25 and wants no credentials. The local sendmail stays
# as a fallback so this same script still works from a login node or anywhere
# an MTA is actually configured.
SMTP_HOST="${ECC_MAIL_SMTP:-smtp.ufl.edu}"
SMTP_PORT="${ECC_MAIL_SMTP_PORT:-25}"

if MAIL_ERR=$(SUBJECT="${SUBJECT}" BODY="${BODY}" TO="${TO}" \
              SMTP_HOST="${SMTP_HOST}" SMTP_PORT="${SMTP_PORT}" \
              "${ECC_PYTHON:-python3}" - <<'PYEOF' 2>&1
import os, smtplib, sys
from email.message import EmailMessage

msg = EmailMessage()
msg["From"] = "RECAP run_all.sh <%s>" % os.environ["TO"]
msg["To"] = os.environ["TO"]
msg["Subject"] = os.environ["SUBJECT"]
msg.set_content(os.environ["BODY"])
try:
    with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ["SMTP_PORT"]), timeout=60) as s:
        s.send_message(msg)
except Exception as exc:
    sys.stderr.write("%s: %r\n" % (type(exc).__name__, exc))
    raise SystemExit(1)
PYEOF
); then
    echo "notify: mailed ${TO} via ${SMTP_HOST}:${SMTP_PORT}"
    exit 0
fi
echo "notify: SMTP to ${SMTP_HOST}:${SMTP_PORT} failed -- ${MAIL_ERR}" >&2

# Fallback: a locally configured MTA, if this is running somewhere that has one.
for cand in /usr/sbin/sendmail /usr/lib/sendmail "$(command -v sendmail 2>/dev/null)"; do
    [ -n "${cand}" ] && [ -x "${cand}" ] || continue
    if {
        printf 'To: %s\n' "${TO}"
        printf 'From: RECAP run_all.sh <%s>\n' "${TO}"
        printf 'Subject: %s\n' "${SUBJECT}"
        printf 'Content-Type: text/plain; charset=UTF-8\n'
        printf '\n'
        printf '%s\n' "${BODY}"
    } | "${cand}" -t -i 2>/dev/null; then
        echo "notify: mailed ${TO} via ${cand}"
        exit 0
    fi
done

echo "notify: no working mail transport -- the message above was NOT sent" >&2
exit 1
