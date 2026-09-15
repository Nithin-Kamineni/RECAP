# Energy_Modeling

Inference-energy modelling for **relaxed error correction** in DNN accelerators,
using Timeloop + Accelergy on UF HiPerGator.

The question: *if ECC parity does not have to be stored in DRAM, how much
inference energy does that save, and does the answer depend on the accelerator?*

Three ECC arms are compared in every figure:

| arm | where the parity lives | status |
|---|---|---|
| `baseline` | beside the data in DRAM; weight traffic inflates by N/K | **implemented** |
| `embedded` | inside the stored weights' own bits, as the embedding pipeline lays them out (63-bit codewords over the weight bit stream, parity in the LSBs); no external parity, the complete codeword is read | **implemented** (Task 2, DRAM effect only, fixed mapping) |
| `recon` | as embedded, but only K/N of the weights are kept on chip and the rest regenerated | placeholder (Tasks 3–5) |

The `recon` placeholder is drawn in every figure so the layout is final from
Task 1 onward; its bar comes from the arithmetic in `eccenergy/ecc.py` and its
result-JSON entries carry `status: not_implemented` with a reason and no energy
number. **Do not quote `recon` as a measured result yet.** `embedded` is a
measured DRAM-only result since Task 2 (`bash run.sh embedded --eval`): it
credits no on-chip saving, keeps codec energy outside the comparison and makes
no accuracy claim -- read its `approximations` before quoting it.

`CLAUDE.md` is the detailed reference (architecture contract, caches, knobs,
how to add a model or an architecture). This file is the operating manual.

---

## Quick start — the whole flow in ONE command

```bash
cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
module load apptainer
bash hpc/run_all.sh --dry-run       # the bill: chips, shapes, cached, jobs
bash hpc/run_all.sh                # map -> evaluate -> plot
```

That reads `env.sh`, writes one task-file row per UNIT of mapper work —
`<bundle> <arch> <model> <K> <depth> <arm> <layer>`, a CHIP (architecture ×
code × buffer depth × arm) crossed with a distinct layer SHAPE — **skipping
every unit the mapper cache already holds**, submits the mapping as a SLURM
array (`hpc/map.sbatch`, one task per bundle) and submits the evaluation-and-plot stage
with a SLURM dependency on it, so the figures are drawn by themselves the moment
the last mapping lands. It returns immediately; `squeue -u $USER` shows both
jobs. Output: `results/figures/`, `results/tables/`, `results/evaluation/`.

**There is one command and one set of variables.** Everything the mapping
optimiser, the energy evaluator, the figures and the SLURM request need lives in
`env.sh`, and every script sources it.

```bash
bash hpc/run_all.sh --map-only     # submit the mapping array and stop
bash hpc/run_all.sh --eval-only    # evaluate + plot from the cache, right here
bash hpc/run_all.sh --replot       # redraw from results/_raw/ alone, seconds
bash hpc/run_all.sh --local        # map in THIS process, then evaluate + plot
                                   # (inside an srun allocation only)
```

Runs are resumable: rerun the same command after a failure and every solved
shape is a cache hit. If a map task fails, SLURM cancels the dependent eval job
— fix it, rerun, or evaluate what exists with `bash hpc/run_all.sh --eval-only`.

---

## `env.sh` — the only file you edit

Every knob in the project is in `env.sh`, one per line, each with a comment
saying what it does. `run.sh`, `hpc/run_all.sh`, `hpc/map.sbatch`, `hpc/tl.sh`,
`hpc/smoke_models.sh` and `restructure/snapshot.py` all source it, so a value
cannot mean one thing to the mapper and another to the evaluator.

**EIGHT SECTIONS SINCE 2026-09-14** (EnvReorganisation phase 4), ordered as
§4.1 asks: what you turn, the mapper, the chip, the prices, then the plumbing.
env.sh's own header maps the old ten section numbers onto these eight, for a
docstring that still says "section 6".

| section | what is in it |
|---|---|
| **1 what to compute and plot** | **the four lines that decide the study** — `ECC_APPROACHES`, `ECC_SWEEP`, `ECC_METRICS`, `ECC_RECON_DEFAULT` — plus the lists they range over (`ECC_ARCHS`, `ECC_MODELS`, `ECC_CODE_N`, `ECC_KS`, `ECC_DEPTH_SWEEP_SCALES`), `ECC_SCOPE`, `ECC_LAYERS`, `ECC_JOBS`, `ECC_ALLOW` |
| **2 the mapper** | **>>> COLDS THE MAPPER CACHE <<<** `ECC_MAPPER_ALGORITHM`, `ECC_VICTORY`, `ECC_VICTORY_SCALING`, `ECC_MAPPER_TIMEOUT`, `ECC_MAPPER_MAX_PERMUTATIONS`, `ECC_OPT_METRIC`, `ECC_MAPPER_SEED`, `ECC_MAPPER_THREADS`, `ECC_RERUN_OPTIMISER` |
| **3 the chip** | **>>> COLDS THE MAPPER CACHE <<<** precisions, `ECC_ARCH_FIDELITY`, `ECC_FORCE_DATAWIDTH`, `ECC_FORCE_TECHNOLOGY`, `ECC_MAPSPACE_CONSTRAIN`, `ECC_MAC_PJ_OVERRIDE`, `ECC_ENERGY_MODEL_REV`, the NoC |
| **4 the prices** | evaluator only — re-priced from cache in ms. The reconstruction terms, DRAM per-bit, latency and standby, parity grouping and padding, decoder energy, the weak overlay, the level classifier |
| **5 the cluster** | account, QOS, partition, cores, memory, wall time, concurrency, `ECC_SIF`, `ECC_USE_CONTAINER`, `ECC_ARCH_PIN_DIR` |
| **6 output and figures** | `ECC_RESULTS_DIR`, palette, formats, DPI, labels |
| **7 miscellaneous** | `ECC_OVERWRITE`, `ECC_CACHE_STRICT`, `ECC_RUN_NOTE`, `ECC_REPLOT_ONLY`, `ECC_FROM_CACHE` |
| **8 derived** | **not knobs.** Translates section 1's lists into the swept-list-plus-two-constants form `eccenergy/config.py` reads, and generates the task file |

Sections 2 and 3 are walled off and marked in the file itself: changing one
re-fingerprints the architecture and colds the mapper cache. Nothing in 4 to 8
colds anything.

### The four lines in section 1 are the run

```bash
# baseline | embedded | recon1 | recon2 | recon3 | recon4 | recon5
ECC_APPROACHES="baseline embedded recon1 recon2 recon3 recon4 recon5"
# bch | model | arch | area | fix     (area = buffer depth; fix and area are POINT sweeps)
ECC_SWEEP=fix
# energy | edp | latency | area
ECC_METRICS=energy
# what a bare `recon` bar MEANS
ECC_RECON_DEFAULT=recon2
```

`ECC_APPROACHES` is the BARS, `ECC_SWEEP` the x axis and `ECC_METRICS` the
figure's rows. The mapper solves every CHIP those imply — an architecture, at a
code, at a buffer depth, in an arm — crossed with each distinct layer shape, and
**units already in the cache are not submitted**. `ECC_SWEEP` picks which list
goes on the x axis; the other axes are held at the **first entry** of their
list.

**There is no abstract `recon` arm.** It was retired by EnvReorganisation phase
6: a bare `recon` in `ECC_APPROACHES` resolves to `ECC_RECON_DEFAULT`, and every
reconstruction bar is a boundary some design declares, billed from its own
mapper cache. A boundary a design has not got is a `[skip]` line, not a
refusal.

More than one model with `ECC_SWEEP=arch` gives the **panelled layout**: one
panel per model, top to bottom, the same x axis inside each. Two models cannot
share one bar axis (that would vary two things per bar), and each panel is
scaled to itself, so compare the *shape* of the panels, not bar heights between
them.

A BCH sweep (`ECC_SWEEP=bch`, `ECC_KS="57 51 45 39 36 30"`) needs **no new
mapping at all** — K changes only the ECC arithmetic on top of a cached
raw-energy record.

### You still never have to edit anything for a one-off

Every value in `env.sh` is written `: "${VAR:=default}"`, so **the environment
wins**:

```bash
ECC_MODELS="resnet18 resnet50 densenet121" bash hpc/run_all.sh
ECC_MAPPER_SEARCH_SIZE=20000 bash hpc/run_all.sh    # bounded development pass
ECC_MAP_TIME=08:00:00 ECC_CONCURRENCY=6 bash hpc/run_all.sh
ECC_VICTORY=4000 bash hpc/run_all.sh                # both jobs inherit it
```

Edit `env.sh` to change a default *permanently*. Be aware that the mapper knobs
(section 2), `ECC_ARCH_FIDELITY`, and anything reaching `globals.yaml` are
hashed into the mapping fingerprint, so changing one of those moves the cache
and the next run is cold. The ECC knobs (section 6, the code geometry, the arms)
are applied on top of cached raw energies and are free to change.

### Re-running the optimiser on purpose

`ECC_RERUN_OPTIMISER=1` makes the mapper re-solve a shape **even when a valid
cache entry exists**, overwriting it. Use it to refresh a mapping after a change
the fingerprint does not capture, or to check that a mapping reproduces. It is
refused together with `--eval` (`ECC_FROM_CACHE=1`), which forbids invoking
Timeloop at all, so a run asked to refresh its mappings can never quietly
refresh nothing. The default, `0`, reuses valid cache hits — which is what makes
a re-run seconds instead of hours.

---

## The stages, and running one by hand

Mapping and evaluation are deliberately separate: mapping is the expensive part
and needs compute nodes; evaluation is seconds and reads the cache. `run.sh` is
the entry point to a single stage and has no knobs of its own.

```bash
bash hpc/tl.sh bash run.sh validate        # architecture contract check, 5 s
bash hpc/tl.sh bash run.sh map             # solve + cache mappings (ALLOCATION)
bash hpc/tl.sh bash run.sh baseline --eval # Task 1 result JSON
bash hpc/tl.sh bash run.sh embedded --eval # Task 2 result JSON, same mappings
bash hpc/tl.sh bash run.sh panels --eval   # the panel figure
bash hpc/tl.sh bash run.sh --replot        # figure only, from results/_raw/
bash hpc/tl.sh bash run.sh --dry-run       # resolve and print the config, stop
```

`--eval` (`ECC_FROM_CACHE=1`) never invokes Timeloop, so those lines are safe on
a login node — they need only python with pandas/matplotlib/pyyaml, which is why
they still go through `hpc/tl.sh`. Set `ECC_USE_CONTAINER=0` to use the host
python instead.

`bash hpc/run_all.sh --eval-only` is exactly that loop followed by the figure,
which is what the dependent SLURM job runs. **Which evaluations it writes is
DERIVED, not selected** (`ECC_EVAL_EXPERIMENTS` was deleted by
EnvReorganisation phase 2): the placement study alone when the axis routed the
run to it — its file already holds Task 1's and Task 2's bars — and Task 1 +
Task 2 otherwise, over `ECC_MODELS`, or over the held model alone on a point
sweep (`fix`, `area`).

**Never run the mapper on a login node.** Get cores first:

```bash
srun --account=rewetz --qos=rewetz --cpus-per-task=18 --mem=8gb --time=02:00:00 --pty bash -i
```

### The SLURM request

`hpc/run_all.sh` passes account, QOS, partition, cores, memory, wall time and
the array size to `sbatch` **on the command line**, from section 7 of `env.sh`,
which overrides the `#SBATCH` header in `hpc/map.sbatch`. That header is only a
fallback for a bare `sbatch hpc/map.sbatch`.

| `env.sh` | default | what it does |
|---|---|---|
| `ECC_CONCURRENCY` | `9` | **the parallelism knob** — array tasks running at once |
| `ECC_MAP_CPUS` | `ECC_MAPPER_THREADS` | cores per task — **see the warning below** |
| `ECC_MAP_MEM` | `16gb` | memory per task |
| `ECC_MAP_TIME` | `24:00:00` | wall limit; a shorter request schedules sooner |
| `ECC_ACCOUNT` / `ECC_QOS` | `rewetz` | `rewetz-b` is the burst QOS (idle cores, low priority, 4-day limit) |
| `ECC_PARTITION` | `hpg-default` | partition |

> **Do not change `ECC_MAPPER_THREADS` casually.** The thread count is hashed
> into the mapping fingerprint, so a different value is a different cache key:
> every mapping already computed becomes a MISS and the whole sweep re-runs from
> cold. 18 is what every cached entry was built at. `hpc/map.sbatch` re-derives
> it from `$SLURM_CPUS_PER_TASK` and warns if the two disagree, so the key can
> never silently drift from the allocation.

The `rewetz` investment is **181 concurrent cores** (`slurmInfo -g rewetz`), so
at 18 cores per task 10 fit; the default `9` leaves room for a VS Code session.
Anything above the limit simply queues as `JobArrayTaskLimit`.

The task list is **generated** into `hpc/.runtime/tasks.txt` from
`ECC_ARCHS × ECC_MODELS` — there is no hand-written list to keep in step with
`--array` any more, which was the classic way to silently skip pairs.

Measured on the 6 × 2 default list: **~50 min wall** for the bounded pass
(`ECC_MAPPER_SEARCH_SIZE=20000`; longest single task 31 min), seconds when
everything is already cached. The uncapped default is typically 4–10× that.

### The model × architecture matrix

```bash
python3 hpc/summary.py --scope layers-full --csv results/tables/panel_matrix.csv
python3 hpc/summary.py --field total         # timeloop | ecc-cost | total | embedded | saving
```

Standard-library only, so it runs with the system python outside the container.

---

## Output names are fixed and overwrite

One axis, one name. Re-running at different constants **rewrites the same
files** rather than accumulating near-identical ones:

```
results/figures/ArchitectureSweep.{png,pdf}   ECC_SWEEP=arch   (1 or N models)
results/figures/ModelSweep.{png,pdf}          ECC_SWEEP=model
results/figures/BCHsweep.{png,pdf}            ECC_SWEEP=bch
results/tables/<same stem>.csv
results/manifests/<same stem>.json            the constants behind the file
```

Section 10 of `env.sh` sets `ECC_STEM` from `ECC_SWEEP` to do that, which
overrides the `__panels__<models>` suffix a panelled run would otherwise append
— so a panelled architecture sweep lands on `ArchitectureSweep.png` like a
single-model one. The cost is deliberate: a one-model and a two-model arch sweep
overwrite each other, and only the manifest says which is on disk. Export
`ECC_STEM=""` to get the self-describing name back.

The manifest is what makes overwriting safe: it records the constants, the code
geometry and the mapper fingerprint behind the file currently on disk. To keep
an old figure, copy it out, or point `ECC_RESULTS_DIR` elsewhere.

**The one exception is development mode.** Setting `ECC_LAYERS` appends the
layer scope to the stem (`ArchitectureSweep__layers2__<names>.png`) so a
two-layer figure can never be mistaken for a full-model one — `env.sh` leaves
`ECC_STEM` empty whenever `ECC_LAYERS` is set, for exactly that reason.

Evaluation JSONs are the opposite: they accumulate on purpose, timestamped, at

```
results/evaluation/{Pre|Post}/<arch>/<model>/<bch>/<prec>/<scope>/<mapper>/<runid>.json
```

---

## Where the logs are

| what | where |
|---|---|
| **SLURM job output** (stdout+stderr of each array task) | `hpc/logs/ecc-map.<arrayjob>_<task>.out`, and `hpc/logs/ecc-eval.<jobid>.out` for the evaluation job |
| **Job output of finished runs** | `hpc/old-logs/`, same filenames — swept there by `bash hpc/tidy_logs.sh` so that `hpc/logs/` shows only the jobs still in the queue |
| **Per-shape mapper logs** (the real Timeloop detail) | `ecc_energy_study/outputs/<arch>/<treatment>/fp-<hash>/<shape>/` — `mapper_console.log`, `timeloop-mapper.map.txt` (the chosen loop nest), `timeloop-mapper.stats.txt` (per-level energy), `timeloop-mapper.accelergy.log` |
| **Bring-up verification logs** | `hpc/logs/bringup-check*.log` |
| **Superseded figures / workload backups** | `ecc_energy_study/logs/` |
| **CACTI scratch** (bound writable into the read-only image) | `hpc/.runtime/cacti_inputs_outputs/` |
| **The generated task list** | `hpc/.runtime/tasks.txt` |

SLURM does not create the log directory; `hpc/run_all.sh` does it for you.

To be mailed once when a whole submission finishes, set `ECC_MAIL_ON_DONE=1`
(env.sh section 5; the default is `0`). `hpc/run_all.sh` then submits
`hpc/notify.sbatch` alongside the map and eval jobs, held on `afterany` of both,
and it mails the outcome once -- with `ECC_SWEEP`, `ECC_ARCHS`, `ECC_MODELS` and
`ECC_APPROACHES` as they were *at submission*, the `sacct` state of every job,
and anything that did not COMPLETE. `afterany` is what makes a failed run mail
too. Prove it works in about a minute, without running a mapping, with
`bash hpc/run_all.sh --mail-test`.

`hpc/logs/` accumulates one file per array task, so **every submission cleans up
after itself** (`ECC_TIDY_LOGS=1`, env.sh section 5, the default).
`hpc/run_all.sh` hangs `hpc/notify.sbatch` off `afterany` of the jobs it just
launched; once they have all terminated, that job moves *their* logs into
`hpc/old-logs/`. `hpc/logs/` therefore holds the run you are watching, and
`tail -f hpc/logs/*.out` follows it.

The sweep is **scoped by job id** (`tidy_logs.sh --jobs`), so a submission
retires its own logs and never touches a concurrent run's live output. No cron
is involved and none is possible -- HiPerGator answers `crontab` with *"not
allowed to use this program"* -- and hanging it off the jobs beats a timer
anyway: it fires exactly when a submission finishes and knows precisely which
logs are now dead. The same job mails you when `ECC_MAIL_ON_DONE=1`; the sweep
does not depend on that setting.

Run `bash hpc/tidy_logs.sh` by hand any time for an unscoped sweep (`--dry-run`
to see what it would take). A job that is still PENDING keeps its id reserved,
so the log a queued array task has not opened yet is never swept out from under
it. Nothing is ever deleted, and the hand-kept `bringup-*.log` evidence stays in
`hpc/logs/`.

Useful while a run is in flight:

```bash
tail -f hpc/logs/ecc-map.*.out
find ecc_energy_study/outputs -name mapper_console.log -newermt '-3 minutes'   # what is solving now
sacct -j <jobid> --format=JobID%18,State%12,Elapsed -X
```

---

## What must never happen

* **Never run the mapper on a login node.** `bash run.sh map` and any cold
  `bash run.sh` need an allocation. `validate`, `--replot`, `--eval`,
  `diagnose` and the tests do not call Timeloop and are fine on the login node.
* **Never delete `ecc_energy_study/outputs/`** — it is hours of compute and the
  shared cache between machines.
* **Never change `ECC_MAPPER_THREADS` away from 18** unless you mean to rebuild
  every mapping.
* **Always write LF line endings.** `bash tools/fix-eol.sh` repairs any file
  that slips through; a CRLF in a `.sh` breaks bash inside the container.

---

## Current state

**The live plan is `prompt_6.md`** — reconstruction-aware mapping: put the
encoder's energy into the ERT so the mapper solves the mapping knowing what
reconstruction costs. Scope is `eyeriss_like_wglb` (Eyeriss v1), one layer,
BCH(63,30).

⚠ **Every reconstruction energy number in the project is pending
regeneration.** prompt_6 RULE 3 separates the encoder's per-codeword and
per-cycle terms, which were previously added together, so every percentage and
every ranking that includes an encoder cost moves. Do not quote a figure or a
table produced before 2026-09-11.

* Tasks 1–3 (conventional / embedded / placement) are implemented; their totals
  await regeneration.
* Task 4 by capacity dilation is **withdrawn** — the capacity mechanism moves in
  2× steps and N/K sits below the step. The live mechanism is `datawidth`.
* The mapper search is converged only under the **constrained mapspace**; an
  unconstrained ranking is not quotable.

`FINDINGS.md` is what was learned, `progress.txt` what is happening now, and
`legacy/` holds the full dated working record.

---

## Layout

```
env.sh                 EVERY KNOB IN THE PROJECT, in ten commented sections
run.sh                 one stage of the pipeline; sources env.sh, no knobs
timeloop.sif           the Apptainer image (git-ignored)
hpc/
  run_all.sh           THE ONE COMMAND: map -> evaluate -> plot, from env.sh
  map.sbatch           the mapper job array; no knobs, sources env.sh
  tl.sh                run any command inside the image
  summary.py           the model x arch matrix (stdlib only)
  notify.sbatch        the one end-of-run mail (ECC_MAIL_ON_DONE=1)
  tidy_logs.sh         sweep finished job output into old-logs/
  HIPERGATOR.md        transfer, image build, verification, parallel design
  logs/                SLURM job output -- jobs still in the queue
  old-logs/            SLURM job output -- finished jobs (git-ignored)
  .runtime/            generated: the task list and CACTI scratch
eccenergy/             the package (config, paths, archs, workloads, timeloop,
                       energy, ecc, parity, results_store, plots, experiments)
archs/                 architecture YAMLs + the shared contract in _shared/
ecc_energy_study/      AUTO-MANAGED: exercises clone + the mapper cache
results/               figures, tables, manifests, evaluation, _raw
```
