"""Tests for the per-shape mapper-cache lock.

    python3 -m eccenergy.tests.test_mapper_lock

Offline: no Timeloop. Exercises the three states a second task can find the
lock in -- held by a live process, left by a dead process, older than the
stale limit -- and that acquire() reports whether it had to wait.
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import threading
import time
import traceback

FAILURES = []


def check(name, fn):
    try:
        fn()
    except Exception:
        FAILURES.append(name)
        print(f"  FAIL  {name}")
        traceback.print_exc()
    else:
        print(f"  ok    {name}")


def test_uncontended_acquire_does_not_wait_and_releases():
    from ..toolchain.cache import ShapeLock
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "fp-x" / "C1_M1"
        lk = ShapeLock(out)
        assert lk.acquire("C1_M1") is False
        assert lk.lockdir.is_dir() and (lk.lockdir / "owner").exists()
        lk.release()
        assert not lk.lockdir.exists()


def test_second_acquire_waits_until_release_then_reports_waited():
    from ..toolchain.cache import ShapeLock
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "fp-x" / "C2_M2"
        first = ShapeLock(out); first.acquire()
        second = ShapeLock(out, poll_s=0.05)
        result = {}
        t = threading.Thread(target=lambda: result.update(waited=second.acquire()))
        t.start(); time.sleep(0.3)
        assert t.is_alive(), "second acquire should still be blocked"
        first.release(); t.join(timeout=5)
        assert result.get("waited") is True
        second.release()


def test_lock_from_dead_pid_on_this_host_is_taken_over():
    from ..toolchain.cache import ShapeLock
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "fp-x" / "C3_M3"
        lk = ShapeLock(out, poll_s=0.05)
        lk.lockdir.parent.mkdir(parents=True); lk.lockdir.mkdir()
        dead = 2 ** 22 - 7                       # almost certainly not a live pid
        (lk.lockdir / "owner").write_text(f"{os.uname().nodename} {dead} {time.time():.0f}\n")
        t0 = time.time(); waited = lk.acquire()
        assert time.time() - t0 < 2 and waited is False
        lk.release()


def test_old_lock_from_another_host_is_taken_over():
    from ..toolchain.cache import ShapeLock
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "fp-x" / "C4_M4"
        lk = ShapeLock(out, stale_s=60, poll_s=0.05)
        lk.lockdir.parent.mkdir(parents=True); lk.lockdir.mkdir()
        (lk.lockdir / "owner").write_text(f"some-other-node 1 {time.time() - 3600:.0f}\n")
        assert lk.acquire() is False
        lk.release()


# ------------------------------------------- the owner file grew a SLURM job
# 2026-09-15. `release()` is in a `finally` and a `finally` does not run on
# SIGTERM or SIGKILL, so `scancel` on a running map array leaves its locks
# behind. The pid check only reaches a lock left on THIS node; across nodes
# there was nothing under it but the six-hour `stale_s`, and a cancelled array
# duly put ten running tasks to sleep for hours on work that takes minutes.
def _write_owner(lk, host, pid, age_s=0, job=None):
    lk.lockdir.parent.mkdir(parents=True, exist_ok=True)
    lk.lockdir.mkdir(exist_ok=True)
    t = time.time() - age_s
    tail = "" if job is None else f" {job}"
    (lk.lockdir / "owner").write_text(f"{host} {pid} {t:.0f}{tail}\n")


def test_an_owner_file_parses_with_or_without_the_job_id():
    """A lock written before the job id, read by a process that expects one --
    and the reverse, inside an array that spans the change. Both must parse or
    `_owner()` returns None and every lock is aged by its mtime instead."""
    from ..toolchain.cache import ShapeLock
    with tempfile.TemporaryDirectory() as tmp:
        lk = ShapeLock(pathlib.Path(tmp) / "fp-x" / "C5_M5")
        _write_owner(lk, "n1", 123, job=None)
        assert lk._owner() == ("n1", 123, lk._owner()[2], "")
        _write_owner(lk, "n1", 123, job="42210127_3")
        assert lk._owner()[3] == "42210127_3"


def test_a_lock_whose_slurm_job_is_gone_is_taken_over_at_once():
    """The rule the pid check cannot be: the owner is on another node, well
    inside `stale_s`, and its job no longer exists."""
    from ..toolchain import cache as cache_mod
    with tempfile.TemporaryDirectory() as tmp:
        lk = cache_mod.ShapeLock(pathlib.Path(tmp) / "fp-x" / "C6_M6",
                                 stale_s=6 * 3600, poll_s=0.05)
        _write_owner(lk, "some-other-node", 1, age_s=60, job="42209843_1")
        real = cache_mod.live_slurm_jobs
        cache_mod.live_slurm_jobs = lambda: {"42210127_0"}      # ours is gone
        try:
            t0 = time.time()
            assert lk.acquire() is False, "a dead job's lock must not be waited on"
            assert time.time() - t0 < 2
        finally:
            cache_mod.live_slurm_jobs = real
        lk.release()


def test_a_live_jobs_lock_is_never_taken_over():
    """The other half, and the one that matters: a job that IS in the queue
    keeps its lock however far from home it is."""
    from ..toolchain import cache as cache_mod
    with tempfile.TemporaryDirectory() as tmp:
        lk = cache_mod.ShapeLock(pathlib.Path(tmp) / "fp-x" / "C7_M7",
                                 stale_s=6 * 3600, poll_s=0.05)
        _write_owner(lk, "some-other-node", 1, age_s=60, job="42210127_0")
        real = cache_mod.live_slurm_jobs
        cache_mod.live_slurm_jobs = lambda: {"42210127_0"}
        try:
            assert lk._stale() is False
        finally:
            cache_mod.live_slurm_jobs = real


def test_prune_removes_a_dead_jobs_lock_and_keeps_a_live_ones():
    from ..toolchain import cache as cache_mod
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        dead = cache_mod.ShapeLock(root / "a" / "fp-x" / "C8_M8")
        live = cache_mod.ShapeLock(root / "a" / "fp-x" / "C9_M9")
        _write_owner(dead, "n2", 1, job="42209843_1")
        _write_owner(live, "n3", 2, job="42210127_0")
        got = cache_mod.prune_dead_locks(root, live={"42210127_0"}, quiet=True)
        assert [d.name for d, _j in got] == ["C8_M8.lock"]
        assert not dead.lockdir.exists()
        assert live.lockdir.is_dir(), "a live job's lock was deleted"


def test_prune_removes_nothing_when_slurm_cannot_be_asked():
    """THE SAFETY PROPERTY. Inside the container `squeue` does not exist, and
    "cannot ask" must never be read as "nothing is alive" -- that would delete
    the lock of every working task in the array."""
    from ..toolchain import cache as cache_mod
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        lk = cache_mod.ShapeLock(root / "a" / "fp-x" / "C10_M10")
        _write_owner(lk, "n4", 1, job="42209843_1")
        real = cache_mod.live_slurm_jobs
        cache_mod.live_slurm_jobs = lambda: None               # no squeue here
        try:
            assert cache_mod.prune_dead_locks(root, quiet=True) == []
        finally:
            cache_mod.live_slurm_jobs = real
        assert lk.lockdir.is_dir(), "a lock was pruned with no way to check it"


def test_prune_leaves_a_lock_that_names_no_job():
    """Written before the job id, or outside SLURM. Age is not proof of death,
    and `stale_s` is still the backstop for these."""
    from ..toolchain import cache as cache_mod
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        lk = cache_mod.ShapeLock(root / "a" / "fp-x" / "C11_M11")
        _write_owner(lk, "n5", 1, age_s=99999, job=None)
        assert cache_mod.prune_dead_locks(root, live={"x"}, quiet=True) == []
        assert lk.lockdir.is_dir()


def test_the_job_id_is_the_array_spelling_squeue_prints():
    """`$SLURM_JOB_ID` inside an array task matches neither `squeue -r`'s `%i`
    nor anything a human reads off the queue."""
    from ..toolchain.cache import slurm_job_id
    keep = {k: os.environ.get(k) for k in
            ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_ID")}
    try:
        os.environ.update(SLURM_ARRAY_JOB_ID="42210127",
                          SLURM_ARRAY_TASK_ID="3", SLURM_JOB_ID="42210130")
        assert slurm_job_id() == "42210127_3"
        del os.environ["SLURM_ARRAY_JOB_ID"], os.environ["SLURM_ARRAY_TASK_ID"]
        assert slurm_job_id() == "42210130"
    finally:
        for k, v in keep.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def main():
    print("eccenergy mapper-cache lock tests")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
