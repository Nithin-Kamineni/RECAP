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
bash hpc/run_all.sh

ECC_RERUN_OPTIMISER=1 ECC_RECON_ERT_AWARE=1 bash hpc/map_ert_arms.sh
```

That reads `env.sh`, writes the `<arch> <model>` task list from
`ECC_ARCHS × ECC_MODELS`, submits the mapping as a SLURM array
(`hpc/map.sbatch`, one task per pair) and submits the evaluation-and-plot stage
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
saying what it does. `run.sh`, `hpc/run_all.sh`, `hpc/map.sbatch` and `hpc/tl.sh`
all source it, so a value cannot mean one thing to the mapper and another to the
evaluator.

| section | what is in it |
|---|---|
| 1 the few you change most often | `ECC_MAPPER_THREADS`, `ECC_LAYERS`, `ECC_USE_CONTAINER` |
| 2 the mapping optimiser | `ECC_MAPPER_ALGORITHM`, `ECC_MAPPER_SEARCH_SIZE`, `ECC_VICTORY`, `ECC_VICTORY_SCALING`, `ECC_MAPPER_TIMEOUT`, `ECC_MAPPER_MAX_PERMUTATIONS`, `ECC_OPT_METRIC` |
| 3 what the pipeline runs | `ECC_RERUN_OPTIMISER`, `ECC_ARCHS`, `ECC_MODELS`, `ECC_CODE_N`, `ECC_KS`, `ECC_APPROACHES`, `ECC_SWEEP`, `ECC_EVAL_EXPERIMENTS`, `ECC_PHASE` |
| 4 reconstruction placement study | **placeholder — no code reads it yet.** `ECC_RECON_MODELING`, `ECC_RECON_ARCH/MODEL/K`, `ECC_RECON_PLACEMENTS` |
| 5 hardware / architecture model | precisions, `ECC_ARCH_FIDELITY`, `ECC_FORCE_DATAWIDTH`, `ECC_FORCE_TECHNOLOGY`, DRAM, the NoC |
| 6 ECC accounting | parity grouping and padding, decoder energy, the DC reconstruction table, the weak overlay, the level classifier |
| 7 the cluster | account, QOS, partition, cores, memory, wall time, concurrency, the image path |
| 8 output and figures | `ECC_RESULTS_DIR`, palette, formats, DPI, labels |
| 9 miscellaneous | `ECC_OVERWRITE`, `ECC_CACHE_STRICT`, `ECC_RUN_NOTE`, `ECC_REPLOT_ONLY`, `ECC_FROM_CACHE` |
| 10 derived | **not knobs.** Translates the lists in section 3 into the swept-list-plus-two-constants form `eccenergy/config.py` reads, and generates the task list |

### The lists in section 3 are the run

```bash
ECC_ARCHS="eyeriss_v2_like eyeriss_like_wglb simple_weight_stationary ..."
ECC_MODELS="resnet18 mobilenet_v2"
ECC_KS="51"
ECC_APPROACHES="baseline embedded recon"
ECC_SWEEP=arch
```

The mapper solves every `(architecture, model)` pair in those lists. `ECC_SWEEP`
picks which list goes on the figure's x axis; the other two axes are held at the
**first entry** of their list. All three ECC arms are always drawn — they are
arms, never an axis.

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

`bash hpc/run_all.sh --eval-only` is exactly the loop over
`ECC_EVAL_EXPERIMENTS` × `ECC_MODELS` followed by the figure, which is what the
dependent SLURM job runs.

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
| **Per-shape mapper logs** (the real Timeloop detail) | `ecc_energy_study/outputs/<arch>/<treatment>/fp-<hash>/<shape>/` — `mapper_console.log`, `timeloop-mapper.map.txt` (the chosen loop nest), `timeloop-mapper.stats.txt` (per-level energy), `timeloop-mapper.accelergy.log` |
| **Bring-up verification logs** | `hpc/logs/bringup-check*.log` |
| **Superseded figures / workload backups** | `ecc_energy_study/logs/` |
| **CACTI scratch** (bound writable into the read-only image) | `hpc/.runtime/cacti_inputs_outputs/` |
| **The generated task list** | `hpc/.runtime/tasks.txt` |

SLURM does not create the log directory; `hpc/run_all.sh` does it for you.

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
* **Always write LF line endings.** `bash tools-fix-eol.sh` repairs any file
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
  HIPERGATOR.md        transfer, image build, verification, parallel design
  logs/                SLURM job output
  .runtime/            generated: the task list and CACTI scratch
eccenergy/             the package (config, paths, archs, workloads, timeloop,
                       energy, ecc, parity, results_store, plots, experiments)
archs/                 architecture YAMLs + the shared contract in _shared/
ecc_energy_study/      AUTO-MANAGED: exercises clone + the mapper cache
results/               figures, tables, manifests, evaluation, _raw
```
