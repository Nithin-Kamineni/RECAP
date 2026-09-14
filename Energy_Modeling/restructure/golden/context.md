# Golden snapshot

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

JSON fields replaced by `<scrubbed>`: `finished_utc`, `git_commit`, `host`, `platform`, `run_id`, `runtime_seconds`, `started_utc`, `user`.

## Stages

| stage | exit | artefacts |
|---|---|---|
| `fingerprints` | 0 | fingerprints.tsv |
| `baseline` | 0 | baseline.txt + 1 result file(s) |
| `embedded` | 0 | embedded.txt + 1 result file(s) |
| `recon_default` | 0 | recon_default.txt + 3 result file(s) |
| `recon_ert` | 0 | recon_ert.txt + 3 result file(s) |
| `validate` | 0 | validate.txt + 1 result file(s) |
| `diagnose` | 0 | diagnose.txt + 2 result file(s) |
| `pytest` | 0 | pytest.txt |
