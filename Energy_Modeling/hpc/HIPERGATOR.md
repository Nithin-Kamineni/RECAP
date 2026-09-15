# Running this project on HiPerGator

Target location on HiPerGator:

    /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling        the project (this folder)
    /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling/timeloop.sif           the container image (built once, §3)

Login host `hpg.rc.ufl.edu`, GatorLink `vkamineni`, group `rewetz`. Everything
below is written for those names; change them in one place (`hpc/tl.sh`) if
they differ.

**Scope of this document.** Steps 1–5 move the project and prove that the
single-process flow (`bash run.sh …`) behaves on HiPerGator exactly as it does
on the laptop; they passed on 2026-09-06. §7 is the parallel path, built on
2026-09-06 once §5 had passed as a SLURM job array over architecture × model
and re-cut by EnvReorganisation phase 3 (2026-09-14) into an array over UNITS —
a chip crossed with one layer shape.

Why this works without code changes: the package resolves every path from
its own location (`eccenergy/paths.py`), Docker is only ever used to run the
`timeloopaccelergy/timeloop-accelergy-pytorch` image, and Apptainer runs that
same image. The mapper cache is content-addressed (architecture YAML + mapper
settings + thread count), so mappings computed on the laptop are cache hits on
HiPerGator as long as `ECC_MAPPER_THREADS=18` is kept.

---

## 1. Stop the local run before copying (recommended)

The laptop may still be mapping into `ecc_energy_study/outputs/` (the chained
job started 2026-09-06). A copy taken mid-write can contain a shape directory
without its `mapping.json` sidecar; that is harmless (it is treated as a cache
miss and re-mapped) but a clean copy is simpler:

```
docker exec 653ecb6bc5de bash -lc "pkill -f run_full_models.sh; pkill -f run.sh; pkill -f timeloop-mapper"
```

Nothing is lost: every finished shape is already on disk and the run resumes
from the cache if restarted.

## 2. Transfer (from the laptop)

644 MB, ~26,000 files, 23,000 of them in the mapper cache. Tar first: `scp -r`
of that many small files takes an hour; the tarball is 44 MB and takes
seconds. You will be asked for the GatorLink password and Duo once per
`ssh`/`scp`.

**From Command Prompt (cmd.exe) or PowerShell** -- Windows ships `tar.exe`;
one line each, no backslash continuations (cmd treats a continued line as a
new command):

```
cd "C:\Users\nithi\jupyter-files\Relaxed error correction"
tar --exclude=Energy_modeling/ecc_energy_study/outputs/_archived_65nm_eyeriss_like --exclude=__pycache__ -czf Energy_Modeling.tgz Energy_modeling
ssh vkamineni@hpg.rc.ufl.edu "mkdir -p /blue/rewetz/vkamineni/Projects/RECAP"
scp Energy_Modeling.tgz vkamineni@hpg.rc.ufl.edu:/blue/rewetz/vkamineni/Projects/RECAP/
```

**From Git Bash** (backslash continuations are fine here):

```
cd "/c/Users/nithi/jupyter-files/Relaxed error correction"

tar --exclude='Energy_modeling/ecc_energy_study/outputs/_archived_65nm_eyeriss_like' \
    --exclude='*/__pycache__' \
    -czf Energy_Modeling.tgz Energy_modeling

ssh vkamineni@hpg.rc.ufl.edu 'mkdir -p /blue/rewetz/vkamineni/Projects/RECAP'
scp Energy_Modeling.tgz vkamineni@hpg.rc.ufl.edu:/blue/rewetz/vkamineni/Projects/RECAP/
```

The excluded directory is the unusable 65 nm cache from before the rewrite
(5,700 files). Everything else — the exercises clone, the mapper cache, the
raw energies, results, legacy — goes along; compute nodes cannot be relied on
to `git clone`, and the cache is hours of compute.

Then on HiPerGator (login node is fine for this):

```
cd /blue/rewetz/vkamineni/Projects/RECAP
tar -xzf Energy_Modeling.tgz && mv Energy_modeling Energy_Modeling && rm Energy_Modeling.tgz
cd Energy_Modeling && bash tools/fix-eol.sh          # LF guard; prints nothing if all is well
```

Alternative without tar (slow, but no extraction step), from PowerShell or Git Bash:

```
scp -r "C:\Users\nithi\jupyter-files\Relaxed error correction\Energy_modeling" vkamineni@hpg.rc.ufl.edu:/blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
```

If the transfer created `Energy_Modeling/Energy_modeling/`, copy its contents
(including top-level files and dotfiles) into the intended project root:

```
cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
cp -a Energy_modeling/. .
bash tools/fix-eol.sh
python3 -c "import eccenergy"
```

Run commands from the outer project root. Once `diff -rq Energy_modeling .`
reports nothing that exists only in the nested copy, it is a pure duplicate and
can be removed (`rm -rf Energy_modeling`) — done 2026-09-06, which freed 650 MB
and ~26,000 inodes. Check that diff before deleting: the outer root is the one
that accumulates new work, so a blind `cp -a Energy_modeling/. .` afterwards
would overwrite newer files with the stale originals.

Bringing results back later (figures, tables, evaluation JSONs) is the same
command in the other direction, e.g.

```
scp -r vkamineni@hpg.rc.ufl.edu:/blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling/results/figures "C:\Users\nithi\jupyter-files\Relaxed error correction\hpg_figures"
```

## 3. Build the container image once (10–30 min)

HiPerGator has no Docker; Apptainer runs the same Docker Hub image. Do this in
an interactive session, not on the login node, and keep Apptainer's cache off
`/home` (40 GB quota):

```
srun --account=rewetz --qos=rewetz --cpus-per-task=4 --mem=16gb --time=01:00:00 --pty bash -i
module load apptainer
export APPTAINER_CACHEDIR=/blue/rewetz/vkamineni/.apptainer APPTAINER_TMPDIR=/blue/rewetz/vkamineni/.apptainer_tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
apptainer pull /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling/timeloop.sif \
    docker://timeloopaccelergy/timeloop-accelergy-pytorch:latest-amd64
```

Quick check that the image is complete:

```
apptainer exec /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling/timeloop.sif bash -c \
  'command -v timeloop-mapper && python3 -c "import pytimeloop, pandas, matplotlib, yaml; print(\"ok\")"'
```

Use Bash's `command -v`, because the inherited HiPerGator `which` function
passes options unsupported by the container's `/usr/bin/which`.

Under `apptainer exec` the image's Jupyter/s6 entrypoint is not started, so
there is none of the startup noise the Docker invocation had to filter.

## 4. How to run anything: `hpc/tl.sh`

`hpc/tl.sh` is the HiPerGator equivalent of the Docker one-liner in
`CLAUDE.md`: it runs its arguments inside the image, with `/blue` visible and
the project as the working directory.

```
cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
module load apptainer
bash hpc/tl.sh bash run.sh validate
```

`ECC_*` variables set in your shell pass straight through to `run.sh` inside
the container, exactly as `-e VAR=…` did with Docker. `ECC_SIF` overrides the
image path.

The wrapper also binds `hpc/.runtime/cacti_inputs_outputs` to
`/usr/local/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/cacti_inputs_outputs`.
CACTI writes scratch files beside its installed plugin; without this bind,
check 5 fails with `OSError: [Errno 30] Read-only file system` and Timeloop
reports "Failed to run Accelergy". This gives only the scratch directory a
writable location; the plugin and model are unchanged. The directory is
created automatically on `/blue`. This wrapper is for single-process use.
Accelergy successfully created its default config under
`/home/vkamineni/.config/accelergy`; no HOME override was needed.

**Never run the mapper on a login node.** Anything that calls Timeloop
(`bash run.sh map`, a cold `bash run.sh`) needs an allocation:

```
srun --account=rewetz --qos=rewetz --cpus-per-task=18 --mem=8gb --time=02:00:00 --pty bash -i
module load apptainer
cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
```

`validate`, `--replot`, `--eval`, `diagnose` and the tests do not call
Timeloop and are light enough for the login node, but running them inside the
same interactive session is simpler.

Confirmed 2026-09-06: `module load apptainer` selects Apptainer 1.4.2-1.el8;
account `rewetz` permits QOS `rewetz` and `rewetz-b`. `slurmInfo -g rewetz`
reports an investment allocation of **181 concurrent CPU cores**, 1414 GB RAM
and 23 GPUs. Burst capacity depends on idle resources.

Check the group's limits (how many cores you may hold at once, and the burst
QOS `rewetz-b`) with `slurmInfo -g rewetz` or
`sacctmgr show assoc user=vkamineni format=account,qos,grptres`.

## 5. Verification checklist — the laptop results must reproduce

Run these in order inside an 18-core interactive session (§4). Expected
outcomes are what the laptop produced on 2026-09-06.

| # | command (prefix each with `bash hpc/tl.sh`) | expected |
|---|---|---|
| 1 | `bash run.sh validate` | `PASS: 6 architecture(s) obey the shared contract` |
| 2 | `python3 -m eccenergy.tests.test_results_store` | `all tests passed` (15) |
| 3 | `bash run.sh --dry-run` | banner, `configuration is valid` |
| 4 | evaluation from the copied cache, no Timeloop (below) | six results, hand check PASS, totals identical to the laptop |
| 5 | one real mapping (below) | maps in ~30–60 s and writes a `mapping.json` sidecar |

Step 4 proves the cache and the fingerprints survived the transfer. The
totals must be exactly: Eyeriss v2 401.977 µJ, Eyeriss v1 390.110, WS 469.457,
OS 711.872, IS 508.517, Simba-like 446.067 (Timeloop energy line), each plus
~48 µJ of external parity:

```
export ECC_MAPPER_ALGORITHM=random_pruned ECC_MAPPER_SEARCH_SIZE=20000 ECC_VICTORY=2000 \
       ECC_MAPPER_TIMEOUT=2000 ECC_MAPPER_THREADS=18 ECC_SWEEP=arch \
       ECC_SWEEP_ARCHS="eyeriss_v2_like eyeriss_like simple_weight_stationary simple_output_stationary simple_input_stationary simba_like"
ECC_LAYERS="layer3.0.downsample.0 layer4.1.conv2" bash hpc/tl.sh bash run.sh baseline --eval
```

Step 5, a shape that is not in the copied cache for this design (check the
`ecc_energy_study/outputs/eyeriss_like/multimodel__opt-energy__vic2000__vicx__alg-random_pruned__ss20000__to2000__noc__paper/fp-*/`
(since 2026-09-07 the slug carries `__noc`: the interconnect is costed, and every
pre-NoC cache under the old slug is deliberately unreachable)
listing if in doubt):

```
ECC_SWEEP_ARCHS=eyeriss_like ECC_LAYERS=layer2.0.downsample.0 bash hpc/tl.sh bash run.sh map
```

If all five pass, the project runs on HiPerGator exactly as on the laptop, and
the two caches can be merged in either direction by copying
`ecc_energy_study/outputs/<arch>/…` directories (they never overlap: the path
carries architecture, treatment, fingerprint and shape).

## 6. Day-to-day workflow with VS Code

* Remote-SSH to `hpg.rc.ufl.edu`; open the folder
  `/blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling`. Edit in the explorer
  as usual; the file watcher and git work normally on `/blue`.
* Use the integrated terminal for `srun … --pty bash -i` (interactive) or
  `sbatch` (batch). A minimal batch job for a long single-process map, kept in
  the project as `hpc/map_single.sbatch` once needed, looks like:

  ```
  #!/bin/bash
  #SBATCH --job-name=ecc-map --account=rewetz --qos=rewetz
  #SBATCH --nodes=1 --ntasks=1 --cpus-per-task=18 --mem=8gb --time=12:00:00
  #SBATCH --output=hpc/logs/%x-%j.out
  module load apptainer
  cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
  export ECC_MAPPER_THREADS=18
  bash hpc/tl.sh bash run.sh map
  ```

  (`mkdir -p hpc/logs` before the first `sbatch`; SLURM does not create output
  directories.)
* Keep `ECC_MAPPER_THREADS=18` equal to `--cpus-per-task`. The thread count is
  hashed into the mapping fingerprint, so a different value is a different
  cache and a different search.
* To be told when a submission finishes instead of watching `squeue`, set
  `ECC_MAIL_ON_DONE=1` (env.sh section 5, default `0`). `hpc/run_all.sh` then
  also submits `hpc/notify.sbatch`, held on `afterany` of the map array *and*
  the eval job, which mails once with the axis, archs, models and arms the run
  was submitted with plus the `sacct` state of every task. `afterany` is
  deliberate: a run that failed is exactly the one you want the mail about.
  Test it in a minute with `bash hpc/run_all.sh --mail-test`.
  * The compute nodes have no working MTA — `mail`/`mailx` are login-node only
    and `/usr/sbin/sendmail` is an unconfigured `msmtp`. The notifier therefore
    speaks SMTP to `ECC_MAIL_SMTP` (`smtp.ufl.edu:25`, reachable from the nodes,
    no credentials) and only falls back to a local `sendmail`. If mail ever
    stops arriving, that relay is the first thing to check.
* `hpc/tidy_logs.sh` moves the job output of finished jobs into `hpc/old-logs/`,
  so `hpc/logs/` keeps only what is still in the queue.
* Windows-side edits to `*.sh`/`*.py`/`*.yaml` may arrive with CRLF if they
  bypass the `.gitattributes`; `bash tools/fix-eol.sh` repairs them.
* Do not delete `ecc_energy_study/outputs/` — hours of compute, and now the
  shared cache between two machines.


## 7. Parallelisation — built 2026-09-06 (`hpc/map.sbatch`)

**THE UNIT OF WORK IS NO LONGER AN (architecture, model) PAIR.** Since
EnvReorganisation phase 3 (2026-09-14) one task-file row is one UNIT -- a CHIP
(architecture x code x buffer-depth scale x arm) crossed with one distinct
layer SHAPE, seven columns, generated by `eccenergy/toolchain/units.py`. The
paragraph that follows is how the fan-out worked until then; it is kept because
the cache layout and the shape-collision reasoning below are still exactly
right, and because the 2026-09-07 note is what lifted the constraint.

The unit of independent work WAS one **(architecture, model)** pair. Each pair
writes only into its own `ecc_energy_study/outputs/<arch>/<treatment>/fp-<hash>/`
subtree, so two tasks never touch the same cache entry — provided the task list
never pairs one architecture with two models that share a layer **shape**.
`resnet18` and `mobilenet_v2` share none (12 + 31 = 43 distinct shapes, zero
overlap), so the panel list is safe. Check before adding a model:

```python
python3 -c "
import json; d=json.load(open('ecc_energy_study/model_layers.json'))
sh=lambda m:{(l['C'],l['M'],l['R'],l['S'],l['P'],l['Q'],l['Wstride'],l['Hstride'],l.get('groups',1)) for l in d[m]}
print(len(sh('resnet18') & sh('mobilenet_v2')))"
```

If two models in the same list DO share shapes, either split them into separate
array submissions or accept that the duplicated shape is mapped twice.

> **2026-09-07:** the constraint above is lifted. `eccenergy/timeloop.py::ShapeLock`
> puts an atomic-mkdir lock on every shape's cache entry; a second task that
> reaches a shape being mapped waits (logged as `[lock] ... is being mapped by
> another task`) and then reuses the result. Any task layout is safe, including
> the full 8 architectures x 8 CNNs matrix (64 tasks -- widen `ECC_ARCHS` and
> `ECC_MODELS` in `env.sh`), where resnet18/resnet50, resnet50/densenet121 and
> mobilenet_v2/efficientnet_b0 share shapes.

**Submit:**

```
cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
bash hpc/run_all.sh --map-only        # or `bash hpc/run_all.sh` for map + eval
```

The task list is GENERATED into `hpc/.runtime/tasks.txt` by
`eccenergy/toolchain/units.py`, and `--array` is sized from it at submit time
-- so a stale hand-written list can no longer disagree with the array. Since
EnvReorganisation phase 3 (2026-09-14) one line is one UNIT of mapper work,
seven columns:

```
<bundle> <arch> <model> <K> <depth> <arm> <layer>
```

a CHIP (architecture × code × buffer-depth scale × arm -- every distinct thing
the mapper is handed) crossed with one distinct layer SHAPE. **Units already in
the mapper cache are not written to the file and are not submitted**, so a
rerun after a failure queues only what is left. `ECC_SWEEP` decides which axis
is enumerated and `ECC_JOBS` bundles the units into fewer jobs (one array task
per bundle, walked serially -- so `ECC_MAP_TIME` must cover the bundle).
`bash hpc/run_all.sh --dry-run` prints the bill and the bundling and submits
nothing.

**Concurrency.** `ECC_CONCURRENCY` (the `%N` in `--array`) is the only place
concurrency is capped. The
`rewetz` investment is **181 cores**; at `--cpus-per-task=18` that is 10 tasks,
and `%9` (162 cores) leaves room for a VS Code ondemand session, which itself
holds 9. Asking for more does not fail — the surplus simply waits with reason
`JobArrayTaskLimit`. `slurmInfo -g rewetz` shows the current headroom.

**`ECC_MAPPER_THREADS` must stay 18** and must equal `ECC_MAP_CPUS`
(`--cpus-per-task`). The
thread count is hashed into the mapping fingerprint, so a different value is a
different cache key and a different search — the results would not be
comparable with the laptop numbers or with anything already cached.

**The two shared-file races are fixed, not avoided.** `patched_arch_path()` and
`write_globals()` both write into `ecc_energy_study/` on a path shared by every
task. Every writer produces byte-identical content for a given
(architecture, treatment), so they never disagree; what a plain `write_text`
could not promise is that a reader arriving during the truncate-then-write
window sees a whole file. Both now go through `_write_atomic()` in
`eccenergy/archs.py`, which writes a per-PID temp file in the same directory
and `os.replace()`s it into place. The content and therefore every fingerprint
is unchanged — section 5 check 4 still reproduces the six laptop totals exactly
after the change. Problem files under `ecc_energy_study/problems/` were already
written atomically.

**Then evaluate** — no container work, no Timeloop, safe on a login node:

```
module load apptainer
bash hpc/run_all.sh --eval-only
```

which runs `bash run.sh baseline --eval` and `bash run.sh embedded --eval`
(Task 1 and Task 2) once per model and then
`bash run.sh panels --eval`, all at the same converged search settings the
array used -- guaranteed, because both stages read those settings from the same
`env.sh`. **WHICH evaluations are written is DERIVED, not selected**
(`ECC_EVAL_EXPERIMENTS` was deleted by EnvReorganisation phase 2, 2026-09-14):
`stage_eval` writes the placement study alone when the axis routed the run to
it -- its own file already holds Task 1's and Task 2's bars -- and Task 1 +
Task 2 otherwise. A point sweep (`fix`, `area`) also holds the model at the
first entry of `ECC_MODELS`, so the evaluation walks that one model and not the
whole list. A plain `bash hpc/run_all.sh` chains this onto the array with
`--dependency=afterok`, so it happens by itself. Finally, the model × architecture matrix:

```
python3 hpc/summary.py --scope layers-full --csv results/tables/panel_matrix.csv
```

`hpc/summary.py` is standard-library only, so it runs with the system python
outside the container. `--field timeloop|ecc-cost|total` chooses which of the
three numbers per cell is shown; `timeloop` is the default because it is what
the architectures are actually compared on.

**Cost model, measured.** One array task maps its BUNDLE of units serially at
18 threads — one unit being one chip at one layer shape, and a bundle being one
unit unless `ECC_JOBS` groups them. Timings from the 2026-09-06 run, when a
task was a whole (architecture, model) pair, are in FINDINGS.md § "HiPerGator
bring-up" and are still the right per-shape figures. Scaling to the full
eight-model `ECC_SWEEP_MODELS` list is 221 distinct shapes × 6 architectures =
1,326 mapping tasks, which is a far larger commitment than the 43 × 6 = 258 of
the panel pair — and widening `ECC_APPROACHES` multiplies it again, because
every placement is its own chip. **`bash hpc/run_all.sh --dry-run` prints the
bill before any of it is submitted**; size it against `slurmInfo -g rewetz`.
