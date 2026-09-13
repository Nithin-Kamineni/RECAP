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
