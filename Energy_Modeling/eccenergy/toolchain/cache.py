"""One mapper-cache entry, one writer.

`ShapeLock` is an atomic-mkdir lock beside a cache entry. `os.mkdir` is atomic on
every filesystem this project runs on, including Lustre and NFS where `flock` is
not reliable -- which is why the lock is a DIRECTORY and not a file. It holds an
`owner` file (host, pid, time, SLURM job) so a lock left behind by a killed job
can be recognised and taken over, and `acquire()` returns True if it had to wait,
so the caller knows to re-check the cache: the task that held the lock has very
probably just written the mapping this one was about to compute.

A 72-job array over one design is the reason it exists.

`prune_dead_locks()` is the other half, added 2026-09-15 after a cancelled array
cost ten running tasks four hours of sleeping: a lock whose SLURM job no longer
exists is removed by whoever can ask SLURM, which is the launcher and the batch
script, never the map job itself -- `squeue` is not in the Timeloop image.

ProjectRestructure phase 3 cut this out of `timeloop.py`, into its own module
because both `toolchain/ert.py` and `toolchain/invoke.py` take the lock and
neither may import the other.
"""
from __future__ import annotations

import getpass
import os
import pathlib
import shutil
import subprocess
import time


def slurm_job_id():
    """This task's job id in the spelling `squeue` prints, or "".

    THE ARRAY FORM, `<array job>_<task>`, because that is what `squeue -r`
    lists and what a human reads off `squeue`. `$SLURM_JOB_ID` inside an array
    task is a per-task id that matches neither.
    """
    array, task = (os.environ.get("SLURM_ARRAY_JOB_ID"),
                   os.environ.get("SLURM_ARRAY_TASK_ID"))
    if array and task:
        return f"{array}_{task}"
    return os.environ.get("SLURM_JOB_ID", "")


def live_slurm_jobs():
    """Every job id SLURM still knows about for this user, or None.

    None means "could not ask", and every caller treats that as "assume alive"
    -- never as "nothing is alive", which would delete live locks.

    IT IS None INSIDE THE CONTAINER. `squeue` is not in the Timeloop image
    (measured 2026-09-15), so a MAP JOB cannot use this: it runs under
    apptainer. The callers that can are the ones outside it -- `hpc/run_all.sh`
    before it submits, and `hpc/map.sbatch` before it enters the container --
    which is where the pruning happens.

    `-r` expands an array into one line per task, so a pending
    `42210586_[3-11%16]` becomes the nine ids a lock could name.
    """
    try:
        out = subprocess.run(
            ["squeue", "-h", "-r", "-u", getpass.getuser(), "-o", "%i"],
            capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return {tok.strip() for tok in out.stdout.split() if tok.strip()}


class ShapeLock:
    """One mapper-cache entry, one writer: an atomic-mkdir lock beside out_dir.

    `os.mkdir` is atomic on every filesystem this project runs on, including
    Lustre and NFS where `flock` is not reliable -- which is why it is a
    directory and not a lock file. The directory holds an `owner` file (host,
    pid, time, SLURM job) so a lock left behind by a killed job can be
    recognised:

      * same host and the pid is gone     -> stale, taken over at once
      * the owning SLURM job is gone      -> stale (only where `squeue` can be
                                             reached, which is NOT inside the
                                             container: see `live_slurm_jobs`)
      * any host, older than `stale_s`    -> stale, taken over (a shape has
                                             never taken 6 h; a job that did
                                             would have been killed by SLURM)
      * otherwise                         -> wait, polling every `poll_s`

    THE SECOND RULE IS THERE BECAUSE THE FIRST CANNOT REACH ANOTHER NODE. A
    cancelled array leaves its locks behind -- `release()` is in a `finally`
    and a `finally` does not run on SIGTERM or SIGKILL -- and every waiter on a
    different node then had only the six-hour timeout. `prune_dead_locks()` is
    the same test applied from outside, by the launcher, where SLURM can
    actually be asked.

    `acquire()` returns True if it had to wait, so the caller knows to re-check
    the cache: the task that held the lock has very probably just written the
    mapping this task was about to compute.
    """

    def __init__(self, out_dir, stale_s=6 * 3600, poll_s=20.0):
        self.out_dir = pathlib.Path(out_dir)
        self.lockdir = self.out_dir.parent / (self.out_dir.name + ".lock")
        self.stale_s = stale_s
        self.poll_s = poll_s
        self.held = False

    def _owner(self):
        """`(host, pid, t0, job)`, or None if there is nothing readable there.

        THREE FIELDS OR FOUR. The job id was added on 2026-09-15 and a lock
        written by an older process has only the first three; a lock written by
        a newer one may be read by an older `_owner` still running inside a
        live job. Both directions have to parse, or an array that spans the
        change starts reporting `owner None` and ageing every lock by its
        mtime instead of by its owner.
        """
        try:
            parts = (self.lockdir / "owner").read_text().split()
            host, pid, t0 = parts[0], int(parts[1]), float(parts[2])
            return host, pid, t0, (parts[3] if len(parts) > 3 else "")
        except (OSError, ValueError, IndexError):
            return None

    def _stale(self):
        o = self._owner()
        if o is None:                       # mid-creation or unreadable: age it
            try:
                return time.time() - self.lockdir.stat().st_mtime > self.stale_s
            except OSError:
                return False
        host, pid, t0, job = o
        if host == os.uname().nodename:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass
        # THE OWNING SLURM JOB IS GONE. The rule above only reaches a lock left
        # on THIS node -- a pid on another host cannot be signalled -- and the
        # cross-host case had nothing but `stale_s` under it. On 2026-09-15 a
        # cancelled array left eight locks on eight other nodes and ten running
        # tasks sat on them for hours with nothing to do; the work itself took
        # minutes. Asking SLURM is the liveness test the pid check cannot be.
        #
        # ONLY WHEN SLURM CAN BE ASKED. `live_slurm_jobs()` is None inside the
        # container, which is where every map job runs, so this is a no-op
        # there and `hpc/map.sbatch` prunes on the way in instead.
        if job:
            live = live_slurm_jobs()
            if live is not None and job not in live:
                return True
        return time.time() - t0 > self.stale_s

    def acquire(self, shape=""):
        self.out_dir.parent.mkdir(parents=True, exist_ok=True)
        waited = False
        announced = False
        while True:
            try:
                os.mkdir(self.lockdir)
            except FileExistsError:
                if self._stale():
                    print(f"      [lock] stale lock on {shape or self.out_dir.name} "
                          f"(owner {self._owner()}); taking it over", flush=True)
                    shutil.rmtree(self.lockdir, ignore_errors=True)
                    continue
                if not announced:
                    print(f"      [lock] {shape or self.out_dir.name} is being mapped by "
                          f"another task ({self._owner()}); waiting", flush=True)
                    announced = True
                waited = True
                time.sleep(self.poll_s)
                continue
            (self.lockdir / "owner").write_text(
                f"{os.uname().nodename} {os.getpid()} {time.time():.0f} "
                f"{slurm_job_id()}\n")
            self.held = True
            return waited

    def release(self):
        if self.held:
            shutil.rmtree(self.lockdir, ignore_errors=True)
            self.held = False


# --------------------------------------------------------------- the pruner
def prune_dead_locks(root, live=None, dry_run=False, quiet=False):
    """Remove every lock under `root` whose owning SLURM job is GONE.

    THE FIX FOR THE ONE FAILURE THIS LOCK CANNOT SURVIVE ON ITS OWN.
    `Mapper._map` releases in a `finally`, so an exception, a Timeloop crash
    and a bad shape all let go. A SIGNAL does not: `scancel` sends SIGTERM and
    then SIGKILL, and under either the interpreter exits without unwinding, so
    the lock stays. Measured 2026-09-15: array 42209843 was cancelled three
    minutes in, left eight locks on eight nodes, and the next two submissions
    put ten tasks to sleep on them -- a pid on another host cannot be
    signalled, so nothing but the six-hour `stale_s` was under that case.

    A lock is removed ONLY when it can be PROVEN dead:

      * `live` is None (SLURM unreachable -- inside the container) -> nothing
        is removed, because "cannot ask" must never read as "nothing is alive";
      * the owner file names no job (written before 2026-09-15, or outside
        SLURM) -> left alone, with a line saying so. Age is not proof;
      * the job is in `live` -> left alone;
      * otherwise -> removed, after RE-READING the owner file to confirm the
        same job still owns it. Between the decision and the removal the lock
        can be taken over by a live task, and deleting a live lock is the one
        thing this must never do.

    Returns `[(lockdir, job)]` for what was removed (or would be).
    """
    root = pathlib.Path(root)
    if live is None:
        live = live_slurm_jobs()
    if live is None:
        if not quiet:
            print("  [locks] SLURM cannot be asked here (no squeue); "
                  "nothing pruned", flush=True)
        return []
    removed, unknown = [], 0
    for lockdir in sorted(root.rglob("*.lock")):
        if not lockdir.is_dir():
            continue
        lock = ShapeLock(lockdir.parent / lockdir.name[:-len(".lock")])
        o = lock._owner()
        if o is None:
            continue
        _host, _pid, _t0, job = o
        if not job:
            unknown += 1
            continue
        if job in live:
            continue
        if not dry_run:
            # Compare-and-delete: re-read, and only remove if the same dead job
            # is still named. A takeover between the two reads is rare and is
            # exactly the case that must not lose a live lock.
            again = lock._owner()
            if again is None or again[3] != job:
                continue
            shutil.rmtree(lockdir, ignore_errors=True)
        removed.append((lockdir, job))
    if not quiet:
        for lockdir, job in removed:
            print(f"  [locks] {'would remove' if dry_run else 'removed'} "
                  f"{lockdir.name} (job {job} is gone)", flush=True)
        if unknown:
            print(f"  [locks] {unknown} lock(s) name no SLURM job and were "
                  f"left alone -- age is not proof of death", flush=True)
        if not removed and not unknown:
            print("  [locks] none stale", flush=True)
    return removed


def main(argv=None):
    """`python3 -m eccenergy.toolchain.cache --prune [--dry-run] [root]`

    RUN IT OUTSIDE THE CONTAINER: it needs `squeue`, which the Timeloop image
    does not have. `hpc/run_all.sh` calls it before submitting and
    `hpc/map.sbatch` before each array task enters the container.
    """
    import argparse
    from .. import paths
    p = argparse.ArgumentParser(prog="eccenergy.toolchain.cache")
    p.add_argument("--prune", action="store_true",
                   help="remove locks whose owning SLURM job is gone")
    p.add_argument("--dry-run", action="store_true",
                   help="say what would be removed, remove nothing")
    p.add_argument("root", nargs="?", default=None,
                   help="mapper-cache root (default: the study's outputs/)")
    a = p.parse_args(argv)
    if not a.prune:
        p.error("nothing to do; pass --prune")
    root = pathlib.Path(a.root) if a.root else paths.WORK / "outputs"
    if not root.is_dir():
        return 0
    prune_dead_locks(root, dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


