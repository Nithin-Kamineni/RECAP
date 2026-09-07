# Energy_Modeling

Inference-energy modelling for **relaxed error correction** in DNN accelerators,
using Timeloop + Accelergy on UF HiPerGator.

The question: *if ECC parity does not have to be stored in DRAM, how much
inference energy does that save, and does the answer depend on the accelerator?*

Three ECC arms are compared in every figure:

| arm | where the parity lives | status |
|---|---|---|
| `baseline` | beside the data in DRAM; weight traffic inflates by N/K | **implemented** |
| `embedded` | inside the already-stored word; DRAM traffic unchanged | placeholder (Task 2) |
| `recon` | as embedded, but only K/N of the weights are kept on chip and the rest regenerated | placeholder (Tasks 3–5) |

The two placeholders are drawn in every figure so the layout is final from
Task 1 onward. Their bars come from the arithmetic in `eccenergy/ecc.py`, and
their result-JSON entries carry `status: not_implemented` with a reason and no
energy number. **Do not quote `embedded` or `recon` as measured results yet.**

`CLAUDE.md` is the detailed reference (architecture contract, caches, knobs,
how to add a model or an architecture). This file is the operating manual.

---

## Quick start — the whole flow in ONE command

```bash
cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
module load apptainer
bash hpc/run_all.sh
```

That submits the mapping as a SLURM array (`hpc/map.sbatch`, one task per
`<arch> <model>` line of `hpc/tasks_panel.txt`) and, with a SLURM dependency on
it, `hpc/eval.sbatch`, which evaluates every result and draws the figures the
moment the last mapping lands. It returns immediately; `squeue -u $USER` shows
both jobs. Output: `results/figures/`, `results/tables/`, `results/evaluation/`.

```bash
TASKFILE=hpc/tasks_full.txt bash hpc/run_all.sh   # all 8 architectures x 8 CNNs (64 tasks)
ECC_VICTORY=4000 bash hpc/run_all.sh              # any ECC_* knob: map and eval both inherit it
MAP_TIME=08:00:00 CONCURRENCY=6 bash hpc/run_all.sh
bash hpc/tl.sh bash run.sh validate               # optional 5-second contract check first
```

The defaults are the converged, publishable search: `random_pruned`, **no
search-size cap**, victory 2000 (doubled per loop level beyond 8), 18 threads.
`ECC_MAPPER_SEARCH_SIZE=20000 bash hpc/run_all.sh` is the quick bounded pass
(~50 min for the 12-task list; its results carry a BOUNDED warning). Runs are
resumable: rerun the same command after a failure and finished shapes are
cache hits. If a map task fails, SLURM cancels the dependent eval job — fix,
rerun, or evaluate what exists with `bash hpc/eval_panel.sh`.

### The same flow by hand, in 4 commands

```bash
# 1. go to the project and load Apptainer
cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
module load apptainer

# 2. (Optional) sanity-check the setup before spending cluster time
bash hpc/tl.sh bash run.sh validate

# 3. MAP: Timeloop + Accelergy, on compute nodes (wait for it to finish)
sbatch hpc/map.sbatch

# 4. EVALUATE + PLOT (reads the cache; login node is fine)
bash hpc/eval_panel.sh
#    -> results/figures/ArchitectureSweep__panels__resnet18__mobilenet_v2.png
```

**Step 2 is optional.** It checks the six architecture YAMLs against the shared
comparison contract (`archs/_shared/standard.yaml`) and exits non-zero on a
violation. Nothing else depends on it — it is worth the five seconds only after
you have edited an architecture, or the first time you use a new machine, to
find a broken setup before step 3 spends an hour discovering it.

**Step 3 is `sbatch`, not `bash`** — it is a real SLURM batch script, and its
`#SBATCH` header is what the scheduler reads. Running it with `bash` would
ignore the header and run all 12 pairs serially on the login node, which is
exactly what must never happen. It returns immediately after queueing; watch it
with `squeue -u $USER` and wait for it to finish before step 4. Skip step 3
entirely when the mappings you want are already cached.

### Configuring step 3

Everything lives at the top of `hpc/map.sbatch`, in two labelled sections.

**Section 1 — the SLURM request.** `#SBATCH` directives are read as plain text
before the script runs, so they cannot contain variables: edit the numbers in
place.

| directive | default | what it does |
|---|---|---|
| `--array=0-11%9` | 12 tasks, 9 at once | **the parallelism knob.** `0-<lines-1>` must match `hpc/tasks_panel.txt`; `%9` caps concurrency |
| `--cpus-per-task` | `18` | cores per task — **see the warning below** |
| `--mem` | `16gb` | memory per task |
| `--time` | `12:00:00` | wall limit; longest measured task was 31 min |
| `--partition` | `hpg-default` | partition |
| `--account` / `--qos` | `rewetz` | `rewetz-b` is the burst QOS (idle cores, low priority, 4-day limit) |
| `--output` | `hpc/logs/%x.%A_%a.out` | one log per array task |

> **Do not change `--cpus-per-task` casually.** The mapper thread count is
> hashed into the mapping fingerprint, so a different value is a different
> cache key: every mapping already computed becomes a MISS and the whole sweep
> re-runs from cold. 18 is what every cached entry was built at. The script
> derives `ECC_MAPPER_THREADS` from `$SLURM_CPUS_PER_TASK`, so the two can
> never silently disagree.

**Section 2 — the model configuration**, in the same `${VAR:=default}` style as
`run.sh`, so the environment wins and you can override per run without editing
the file:

```bash
ECC_VICTORY=500 sbatch hpc/map.sbatch                    # a cheaper search
ECC_LAYERS="layer4.1.conv2" sbatch hpc/map.sbatch        # one layer, not the model
TASKFILE=hpc/tasks_all.txt sbatch hpc/map.sbatch         # a different work list
```

`ECC_MAPPER_ALGORITHM`, `ECC_MAPPER_SEARCH_SIZE`, `ECC_VICTORY` and
`ECC_MAPPER_TIMEOUT` are the search; they are also in the fingerprint, so
changing one starts a new cache and `hpc/sweep.sh` must be given the same values
or every plot will miss.

**To change which pairs get mapped:** edit `hpc/tasks_panel.txt` (one
`<arch> <model>` per line), **then update `--array=0-<lines-1>%<N>` to match.**
The script refuses to start if the two disagree, so a stale `--array` is an
immediate error rather than a silently half-finished sweep.

### You never edit `run.sh`

`run.sh` defaults every knob with `${VAR:=default}`, so **the environment always
wins** and a one-off run needs no file edit. Put the configuration in front of
the command:

```bash
# pipeline: override a search knob for one run (SLURM limits are #SBATCH, above)
ECC_VICTORY=500 sbatch hpc/map.sbatch

# plot: swap the swept axis, or the held constants
bash hpc/sweep.sh arch                              # accelerators, hold model + K
bash hpc/sweep.sh model                             # networks,     hold arch  + K
bash hpc/sweep.sh bch                               # BCH(63,K),    hold arch  + model
ECC_CONST_MODEL=mobilenet_v2 bash hpc/sweep.sh arch # one model
ECC_SWEEP_KS="57 51 45 39"    bash hpc/sweep.sh bch # a shorter code sweep
```

To change which (architecture, model) pairs the pipeline maps, edit
`hpc/tasks_panel.txt` — one `<arch> <model>` per line — or write your own list
and pass it to `hpc/map.sbatch`. That file, not `run.sh`, is the work list.

The only reason to edit `run.sh` is to change a *default* permanently. Be aware
that the mapper knobs (`ECC_MAPPER_*`, `ECC_VICTORY*`, `ECC_OPT_METRIC`,
`ECC_ARCH_FIDELITY`, and anything reaching `globals.yaml`) are hashed into the
mapping fingerprint, so changing one of those moves the cache and the next run
is cold. The ECC knobs (`ECC_CODE_N`/`ECC_CONST_K`, the arms, parity accounting)
are applied on top of cached raw energies and are free to change.

Everything runs inside the Apptainer image through `hpc/tl.sh`. The image is
`timeloop.sif` in this directory (1.9 GB, git-ignored); `ECC_SIF` overrides the
path, and `hpc/HIPERGATOR.md` §3 says how to rebuild it.

---

## The two-step workflow

Mapping and evaluation are deliberately separate. Mapping is the expensive part
and needs compute nodes; evaluation is seconds and reads the cache.

### 1. Map — needs an allocation, never the login node

```bash
sbatch hpc/map.sbatch          # SLURM job array
squeue -u $USER
```

`hpc/tasks_panel.txt` is one `<arch> <model>` per line; the array index picks
the line. One task maps every distinct layer shape of that model on that
architecture and writes a `mapping.json` sidecar per shape.

Both the SLURM request and the search settings are edited at the top of
`hpc/map.sbatch` — see **Configuring step 3** above for the full table. The
short version: `#SBATCH --array=0-<lines-1>%<N>` is the parallelism knob,
`--cpus-per-task=18` must not change casually, and everything in Section 2 can
be overridden per run from the environment.

The `rewetz` investment is **181 concurrent cores** (`slurmInfo -g rewetz`), so
at 18 cores per task 10 tasks fit; the default `%9` leaves room for a VS Code
session. Anything above the limit simply queues as `JobArrayTaskLimit`.

Measured on the 12-task default list: **~50 min wall** cold (longest single task
31 min), seconds when everything is already cached.

### 2. Plot — cache-only, safe anywhere

```bash
bash hpc/sweep.sh arch      # sweep accelerators, hold model + BCH K
bash hpc/sweep.sh model     # sweep networks,     hold arch  + BCH K
bash hpc/sweep.sh bch       # sweep BCH(63,K),    hold arch  + model
bash hpc/sweep.sh panels    # arch sweep, one panel per model
```

Exactly one axis varies per run; the other two are held. That is what makes each
figure a single claim. All three ECC arms are always drawn.

Override any constant from the shell:

```bash
ECC_CONST_MODEL=mobilenet_v2 bash hpc/sweep.sh arch
ECC_CONST_ARCH=simba_like    bash hpc/sweep.sh bch
ECC_SWEEP_KS="57 51 45 39"   bash hpc/sweep.sh bch
ECC_SWEEP_ARCHS="eyeriss_like eyeriss_v2_like" bash hpc/sweep.sh arch
```

**Several models in one architecture sweep** — give `ECC_CONST_MODEL` a list and
the run switches to the panelled layout, one panel per model, still written to
`ArchitectureSweep.png`:

```bash
ECC_CONST_MODEL="resnet18 mobilenet_v2" bash hpc/sweep.sh arch
```

Two models cannot share one bar axis — `run.sh` holds the model fixed on
purpose, and stacking them would vary two things per bar — so the list becomes
panels instead. Each panel is scaled to itself, so compare the *shape* of the
panels, not the bar heights between them.

A BCH sweep needs **no new mapping at all** — changing K only changes the ECC
arithmetic on top of a cached raw-energy record, so it is instant.

There is also `bash hpc/eval_panel.sh`, which writes the per-model result JSONs
(`run.sh baseline --eval` per model) and then the panel figure. Use it when you
want the machine-readable results, not just the picture.

### 3. The model × architecture matrix

```bash
python3 hpc/summary.py --scope layers-full --csv results/tables/panel_matrix.csv
python3 hpc/summary.py --field total         # timeloop | parity | total
```

Standard-library only, so it runs with the system python outside the container.

---

## Output names are fixed and overwrite

One sweep, one name. Re-running at different constants **rewrites the same
files** rather than accumulating near-identical ones:

```
results/figures/ArchitectureSweep.{png,pdf}   sweep.sh arch   (1 or N models)
results/figures/ModelSweep.{png,pdf}          sweep.sh model
results/figures/BCHsweep.{png,pdf}            sweep.sh bch
results/tables/<same stem>.csv
results/manifests/<same stem>.json            the constants behind the file
```

Three axes, three names — a panelled architecture sweep lands on
`ArchitectureSweep.png` like a single-model one. `hpc/sweep.sh` does that by
setting `ECC_STEM`, which forces one output name and overrides the
`__panels__<models>` suffix `run.sh panels` would otherwise append. The cost is
deliberate: a one-model and a two-model arch sweep now overwrite each other, and
only the manifest says which is on disk.

The manifest is what makes overwriting safe: it records the constants, the code
geometry and the mapper fingerprint behind the file currently on disk. To keep
an old figure, copy it out, or point `ECC_RESULTS_DIR` elsewhere.

**The one exception is development mode.** Setting `ECC_LAYERS` appends the
layer scope to the stem (`ArchitectureSweep__layers2__<names>.png`) so a
two-layer figure can never be mistaken for a full-model one. `hpc/sweep.sh`
forces `ECC_LAYERS=""` for exactly this reason — if you see a long figure name,
something set `ECC_LAYERS`.

Evaluation JSONs are the opposite: they accumulate on purpose, timestamped, at

```
results/evaluation/{Pre|Post}/<arch>/<model>/<bch>/<prec>/<scope>/<mapper>/<runid>.json
```

---

## Where the logs are

| what | where |
|---|---|
| **SLURM job output** (stdout+stderr of each array task) | `hpc/logs/<jobname>-<arrayjob>_<task>.out`, e.g. `hpc/logs/ecc-map-41269118_7.out` — override with `ECC_LOGDIR` |
| **Per-shape mapper logs** (the real Timeloop detail) | `ecc_energy_study/outputs/<arch>/<treatment>/fp-<hash>/<shape>/` — `mapper_console.log`, `timeloop-mapper.map.txt` (the chosen loop nest), `timeloop-mapper.stats.txt` (per-level energy), `timeloop-mapper.accelergy.log` |
| **Bring-up verification logs** | `hpc/logs/bringup-check*.log` |
| **Superseded figures / workload backups** | `ecc_energy_study/logs/` |
| **CACTI scratch** (bound writable into the read-only image) | `hpc/.runtime/cacti_inputs_outputs/` |

SLURM does not create the log directory; `hpc/map.sbatch` does it for you.

Useful while a run is in flight:

```bash
tail -f hpc/logs/ecc-map-*.out
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

* resnet18 (21 layers) and mobilenet_v2 (53 layers, 10 depthwise) are evaluated
  **whole-model on all six architectures** at the converged search.
* The architecture ranking is preserved across both models:
  `v2 ≈ v1 < Simba-like ≈ input-stationary < weight-stationary < output-stationary`.
* **The two development layers are not a proxy for the model** — weight- and
  input-stationary swap places between them.
* Still open: six of eight models unmapped; NoC energy is zero for every
  architecture; the `_wglb` Eyeriss bracketing pair was not run.

`progress.txt` is the status board, `FINDINGS.md` §13 the evidence,
`PROJECT_STATUS.md` the summary.

---

## Layout

```
run.sh                 the model knob file (ECC_* defaults). Runs IN the container.
timeloop.sif           the Apptainer image (git-ignored)
hpc/
  tl.sh                run any command inside the image
  map.sbatch           STEP 3: the mapper job array. SLURM request + search
                       settings both live at the top of this file
  tasks_panel.txt      the work list: one "<arch> <model>" per line
  sweep.sh             plot one axis, fixed output names
  eval_panel.sh        per-model result JSONs + the panel figure
  summary.py           the model x arch matrix (stdlib only)
  HIPERGATOR.md        transfer, image build, verification, parallel design
  logs/                SLURM job output
eccenergy/             the package (config, paths, archs, workloads, timeloop,
                       energy, ecc, parity, results_store, plots, experiments)
archs/                 architecture YAMLs + the shared contract in _shared/
ecc_energy_study/      AUTO-MANAGED: exercises clone + the mapper cache
results/               figures, tables, manifests, evaluation, _raw
```
