"""One mapper-cache entry, one writer.

`ShapeLock` is an atomic-mkdir lock beside a cache entry. `os.mkdir` is atomic on
every filesystem this project runs on, including Lustre and NFS where `flock` is
not reliable -- which is why the lock is a DIRECTORY and not a file. It holds an
`owner` file (host, pid, time) so a lock left behind by a killed job can be
recognised and taken over, and `acquire()` returns True if it had to wait, so the
caller knows to re-check the cache: the task that held the lock has very probably
just written the mapping this one was about to compute.

A 72-job array over one design is the reason it exists.

ProjectRestructure phase 3 cut this out of `timeloop.py`, into its own module
because both `toolchain/ert.py` and `toolchain/invoke.py` take the lock and
neither may import the other.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import time


class ShapeLock:
    """One mapper-cache entry, one writer: an atomic-mkdir lock beside out_dir.

    `os.mkdir` is atomic on every filesystem this project runs on, including
    Lustre and NFS where `flock` is not reliable -- which is why it is a
    directory and not a lock file. The directory holds an `owner` file (host,
    pid, time) so a lock left behind by a killed job can be recognised:

      * same host and the pid is gone     -> stale, taken over at once
      * any host, older than `stale_s`    -> stale, taken over (a shape has
                                             never taken 6 h; a job that did
                                             would have been killed by SLURM)
      * otherwise                         -> wait, polling every `poll_s`

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
        try:
            host, pid, t0 = (self.lockdir / "owner").read_text().split()
            return host, int(pid), float(t0)
        except (OSError, ValueError):
            return None

    def _stale(self):
        o = self._owner()
        if o is None:                       # mid-creation or unreadable: age it
            try:
                return time.time() - self.lockdir.stat().st_mtime > self.stale_s
            except OSError:
                return False
        host, pid, t0 = o
        if host == os.uname().nodename:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass
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
                f"{os.uname().nodename} {os.getpid()} {time.time():.0f}\n")
            self.held = True
            return waited

    def release(self):
        if self.held:
            shutil.rmtree(self.lockdir, ignore_errors=True)
            self.held = False


