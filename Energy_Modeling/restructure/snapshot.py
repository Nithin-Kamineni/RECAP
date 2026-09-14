"""Take the golden snapshot ProjectRestructure.md section 9.1 is a gate against.

    bash hpc/tl.sh python3 restructure/snapshot.py <outdir>

WHAT A SNAPSHOT IS. Everything the six gate items compare, captured as text that
`diff` can speak about:

    1. arch_fingerprint() for every (arch, model)   fingerprints.tsv
    2. run.sh baseline --eval  (Task 1)             baseline/
    3. run.sh embedded --eval  (Task 2)             embedded/
    4. run.sh recon    --eval  (Tasks 3/4)          recon_default/  recon_ert/
    5. run.sh validate, run.sh diagnose             validate.txt  diagnose.txt
    6. the test suite, BOTH roots                   pytest.txt

ENV.SH ALONE DECIDES WHAT IS SNAPSHOTTED. Every stage runs with every `ECC_*`
variable and `RECON_OPTIMIZER` STRIPPED from the environment, so `run.sh`'s own
`source ./env.sh` supplies them and a knob left exported in the caller's shell
cannot quietly become part of the golden record. The two stages that must differ from
env.sh's defaults say so in `STAGES` below, and the override lands in the
snapshot's own `context.md`.

WHAT IS SCRUBBED, AND WHY THAT IS SAFE. A run id, a wall-clock timestamp, a
hostname and the git commit move on every run and mean nothing to the gate; the
project root moves if the tree is copied. They are replaced by placeholders --
LISTED IN `context.md` -- and nothing else is touched. Every energy, every
fingerprint, every count and every refusal message is kept byte for byte.

A STAGE THAT REFUSES IS SNAPSHOTTED, NOT SKIPPED. A refusal is the current
behaviour of the code, so it is recorded with its exit status like any other
stage: a phase that changes it has changed behaviour, and the gate says so.
(`run.sh recon --eval` at env.sh's defaults refused until 2026-09-14, when the
defaults became the ERT-aware study; `stages.tsv` carries the exit code.)

THE OVERRIDES ABOVE MUST STAY INERT UNDER A LATER PHASE. EnvReorganisation
deletes `ECC_PHASE` and `RECON_OPTIMIZER` as knobs (phase 2) -- an environment
variable nothing reads changes nothing, so the stage keeps producing the same
totals, which is what the gate then checks.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: (name, argv, extra env). `--eval` never invokes Timeloop, so no stage here
#: can move a mapping -- the snapshot reads the caches, it does not fill them.
STAGES = (
    # Task 1 is a `Pre` result BY CONSTRUCTION (env.sh section 3): there is no
    # reduced weight representation yet, so no mapping could have been optimised
    # for one. env.sh ships Post + RECON_OPTIMIZER=True, which is Task 4.
    # ECC_RECON_ERT_AWARE=0 as well, since 2026-09-14: env.sh now ships it
    # at 1, and 1 is REFUSED beside Pre + optimizer-off by the coupling guard
    # `ert-arm-needs-optimiser` -- which is exactly the per-arm phase problem
    # EnvReorganisation 6.1 removes. Until then Task 1 and Task 2 are taken at
    # the configuration they are defined at, so the golden holds their totals
    # and not a refusal. (Under a derived phase these three overrides become
    # inert, and the totals must not move.)
    ("baseline",      ["bash", "run.sh", "baseline", "--eval"],
     {"ECC_PHASE": "Pre", "RECON_OPTIMIZER": "False", "ECC_RECON_ERT_AWARE": "0"}),
    ("embedded",      ["bash", "run.sh", "embedded", "--eval"],
     {"ECC_PHASE": "Pre", "RECON_OPTIMIZER": "False", "ECC_RECON_ERT_AWARE": "0"}),
    # env.sh's defaults, unchanged -- the configuration a bare `run.sh` runs.
    # Since 2026-09-14 that IS the ERT-aware placement study, so this stage and
    # the next produce the same files; both are kept so that a default that
    # drifts away from the live configuration shows up here.
    ("recon_default", ["bash", "run.sh", "recon", "--eval"], {}),
    # The live Phase C2 configuration: the six ERT arms mapped for resnet18.
    ("recon_ert",     ["bash", "run.sh", "recon", "--eval"],
     {"ECC_RECON_ERT_AWARE": "1"}),
    ("validate",      ["bash", "run.sh", "validate"], {}),
    ("diagnose",      ["bash", "run.sh", "diagnose"], {}),
)

#: JSON fields that move on every run and say nothing about the model.
VOLATILE_KEYS = frozenset((
    "run_id", "started_utc", "finished_utc", "runtime_seconds",
    "host", "user", "platform", "git_commit",
))

#: Result subtrees worth keeping.
KEEP = ("evaluation/", "manifests/", "tables/")


def _clean_env(extra):
    """os.environ with every study knob removed, plus this stage's overrides."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("ECC_") and k != "RECON_OPTIMIZER"}
    env.update(extra)
    return env


def _scrub_text(text):
    text = text.replace(str(ROOT), "<ROOT>")
    text = re.sub(r"\d{8}T\d{6}Z__[0-9a-f]+", "<RUNID>", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)?",
                  "<UTC>", text)
    text = re.sub(r"\bin \d+\.\d+s\b", "in <T>s", text)          # pytest summary
    return text


def scrub_json(obj):
    if isinstance(obj, dict):
        return {k: ("<scrubbed>" if k in VOLATILE_KEYS else scrub_json(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_json(v) for v in obj]
    if isinstance(obj, str):
        return _scrub_text(obj)
    return obj


def _snap_results(started, dest):
    """Copy every result file this stage wrote, scrubbed, under `dest`."""
    results = ROOT / "results"
    if not results.is_dir():
        return []
    kept = []
    for path in sorted(results.rglob("*")):
        if not path.is_file() or path.stat().st_mtime < started:
            continue
        rel = path.relative_to(results).as_posix()
        if not rel.startswith(KEEP):
            continue
        if rel.startswith("evaluation/"):
            # `latest.json` is a POINTER (a run id and a timestamp), so it is
            # wholly volatile and carries no energy. The timestamped file beside
            # it is the result -- kept under a stable name so the gate compares
            # content rather than filenames.
            if path.name == "latest.json":
                continue
            rel = rel.rsplit("/", 1)[0] + "/result.json"
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            out.write_text(json.dumps(
                scrub_json(json.loads(path.read_text())),
                indent=2, sort_keys=True) + "\n")
        else:
            out.write_text(_scrub_text(path.read_text()))
        kept.append(rel)
    return kept


def _run(argv, extra, log):
    proc = subprocess.run(argv, cwd=ROOT, env=_clean_env(extra),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(_scrub_text(proc.stdout.decode("utf-8", "replace")))
    return proc.returncode


def main(argv):
    if len(argv) != 2:
        sys.stderr.write(__doc__.split("\n\n")[1] + "\n")
        return 2
    out = Path(argv[1]).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    status = []

    # ---- gate item 1 -----------------------------------------------------
    fp = out / "fingerprints.tsv"
    rc = _run(["bash", "-c",
               "source env.sh; PYTHONPATH=. python3 restructure/_fingerprints.py"],
              {}, fp)
    status.append(("fingerprints", rc, "fingerprints.tsv"))

    # ---- gate items 2-5 --------------------------------------------------
    for name, argv_stage, extra in STAGES:
        started = time.time()
        time.sleep(1.1)              # mtime granularity: never miss a write
        rc = _run(argv_stage, extra, out / f"{name}.txt")
        kept = _snap_results(started, out / name)
        status.append((name, rc, f"{name}.txt" + (f" + {len(kept)} result file(s)"
                                                  if kept else "")))

    # ---- gate item 6 -----------------------------------------------------
    # `source env.sh` first, exactly as CLAUDE.md documents the suite being run:
    # the data-backed property tests read env.sh's knobs, and without them 25 of
    # them fail on a configuration nobody runs. pytest is a --user install, not
    # in the image.
    rc = _run(["bash", "-c",
               "source env.sh; exec python3 -m pytest eccenergy/tests/ tests/ -q "
               "--no-header -p no:cacheprovider"], {}, out / "pytest.txt")
    status.append(("pytest", rc, "pytest.txt"))

    (out / "stages.tsv").write_text(
        "# stage\texit\tartefacts\n"
        + "".join(f"{n}\t{r}\t{a}\n" for n, r, a in status))

    (out / "context.md").write_text(f"""# Golden snapshot

Taken by `restructure/snapshot.py`; compared by `restructure/gate.sh`.
See `ProjectRestructure.md` section 9.1 for what it is a gate against.

## How to retake it

    bash hpc/tl.sh python3 restructure/snapshot.py <outdir>

Every stage runs with every `ECC_*` variable and `RECON_OPTIMIZER` stripped from
the environment, so `env.sh` alone supplies them. Two stages override env.sh and
say why in `snapshot.py`'s `STAGES`.

## Placeholders

These four move on every run and are replaced before anything is written. Nothing
else is touched -- every energy, fingerprint, count and refusal message is byte
for byte what the run produced.

| placeholder | what it replaces |
|---|---|
| `<ROOT>` | the absolute project root |
| `<RUNID>` | a result store run id (`<date>T<time>Z__<confighash>`) |
| `<UTC>` | an ISO wall-clock timestamp |
| `<T>` | pytest's wall-clock duration |

JSON fields replaced by `<scrubbed>`: {", ".join("`%s`" % k for k in sorted(VOLATILE_KEYS))}.

## Stages

| stage | exit | artefacts |
|---|---|---|
""" + "".join(f"| `{n}` | {r} | {a} |\n" for n, r, a in status))

    print(f"snapshot -> {out}")
    for n, r, a in status:
        print(f"  {'ok ' if r == 0 else 'rc%-2d' % r} {n:14s} {a}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
