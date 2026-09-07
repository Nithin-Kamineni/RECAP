# CLAUDE.md

Energy modelling for relaxed error correction in DNN accelerators, using
Timeloop + Accelergy. The question the project answers: **if ECC parity does not
have to be stored in DRAM, how much inference energy does that save, and does
the answer depend on the accelerator?**

> **The energy model is being rewritten (as of 2026-09-06).** This file has been
> deliberately stripped to *structure only* — how the code is organised, how a
> run is configured, where output goes, and the rules for changing it. Every
> empirical claim that was here (per-access energies, scratchpad depths, on-chip
> capacities, which architecture beats which, and why) has been REMOVED because
> it described the model as it stood before the rewrite.
>
> **Do not re-add empirical claims to this file until the new model produces
> them.** For the live, computed version of the caveat list, run
> `bash run.sh diagnose`. The pre-rewrite analysis is archived, clearly marked
> historical, in `legacy/FINDINGS.md`; the architecture `README.md` files under
> `archs/` still contain pre-rewrite justifications and should be read with the
> same suspicion.

## One sweep, two constants

All three ECC approaches (`baseline`, `embedded`, `recon`) are ALWAYS drawn, so
the arms are never an axis. Of the three remaining axes a run sweeps exactly
ONE and holds the other two at a constant:

| `ECC_SWEEP` | x axis | sweep list | held constants | output |
|---|---|---|---|---|
| `bch` | BCH(63,K) | `ECC_SWEEP_KS` | `ECC_CONST_ARCH`, `ECC_CONST_MODEL` | `BCHsweep` |
| `model` | the networks | `ECC_SWEEP_MODELS` | `ECC_CONST_ARCH`, `ECC_CONST_K` | `ModelSweep` |
| `arch` | the accelerators | `ECC_SWEEP_ARCHS` | `ECC_CONST_MODEL`, `ECC_CONST_K` | `ArchitectureSweep` |

The values each axis can take:

| axis | values |
|---|---|
| architecture | `eyeriss_like`, `eyeriss_like_wglb`, `eyeriss_v2_like`, `eyeriss_v2_like_wglb`, `simple_weight_stationary`, `simple_output_stationary`, `simple_input_stationary`, `simba_like` |
| model (CNN) | `resnet18`, `resnet50`, `densenet121`, `squeezenet1_1`, `mobilenet_v2`, `efficientnet_b0`, `convnext_tiny`, `xception` |
| model (transformer) | `distilgpt2`, `gpt2`, `bert_base`, `gpt2_medium`, `tinyllama` |
| code geometry | BCH(N,K), `ECC_CODE_N`=63 with K in {57,51,45,39,36,30} |
| ECC approach | `baseline`, `embedded`, `recon` (Recon+, formerly "patched") |

`cfg.archs`, `cfg.models` and `cfg.code_k` are DERIVED in `config.py` from that
choice — the swept axis takes its list, the held axes take their constant.
Nothing below `config.py` knows which axis is being swept except `sweep.py`.

A model sweep is all-CNN or all-transformer; mixing the families is a config
error, because they come from different workload files.

### The panel layout, which is NOT a fourth axis

`ECC_EXPERIMENT=panels` with `ECC_PANEL_MODELS="resnet18 mobilenet_v2"` draws
ONE image with one PANEL per model, top to bottom, repeating the chosen sweep
inside each. It exists for the question no single sweep can answer — whether
the architecture ranking survives changing the network — because that claim is
about the SHAPE of two panels, not about either one alone.

It adds no axis and no renderer: the x axis is still `ECC_SWEEP`'s, and every
bar is drawn by `plots/stacked.py::draw_panel`, the same routine the three
sweeps use. `cfg.models` is simply widened to the panel models so one
collection pass fills every panel. `ECC_SWEEP=bch` is refused here — a BCH
panel per model varies the model AND the code between panels, which is two axes
at once.

Panels share a legend, a category set and an energy unit. They do NOT share a y
limit, and the figure says so on itself: two networks of different size forced
onto one scale makes the smaller one unreadable.

The output stem carries the panel models —
`ArchitectureSweep__panels__resnet18__mobilenet_v2` — so it can never land on
top of `ArchitectureSweep.png`, and two different model pairs are two different
files.

`simba_like` is *inspired by* NVIDIA's Simba but **does not reproduce it**, and
must not be quoted as "Simba" — its figure label is "Simba-like (reference
design)" for that reason. Its 8b/8b/24b precisions are the paper's; its
geometry is timeloop-accelergy-exercises' example design and fails two checks
against the paper (256 MACs where the published 4 TOPS needs ~1024; 3 MB of
PE-private buffer behind a 64 kB shared buffer). Its 2.1M on-chip weights —
28× Eyeriss v1's — set its DRAM refetch and therefore its ECC saving, so this
is not a cosmetic caveat. `archs/_shared/provenance.yaml` has the full record
and what would be needed to fix it. It is sometimes miscalled "Numba" — there
is no Numba architecture in this project.

### Eyeriss v1 comes as a bracketing PAIR, not a number

JSSC 2017 allocates 8 kB of Eyeriss v1's 108 kB GLB to filter weights.
`eyeriss_like` does not model it (arguing prefetch is not reuse);
`eyeriss_like_wglb` models it at the published capacity and bank geometry. The
two files differ in exactly one block.

That choice is the single biggest lever on this study's headline result:
`eyeriss_like` refetches resnet18's weights ~7× from DRAM against 1.0–2.3× for
every other design, and the ECC saving is very nearly
`0.3125·E_dram_w / (total + 0.3125·E_dram_w)`.

    eyeriss_like        -> UPPER bound on DRAM weight traffic and ECC saving
    eyeriss_like_wglb   -> LOWER bound

**Quote the pair.** A single Eyeriss v1 saving number from this study is a
choice of bound, not a measurement. Same rule, opposite direction, for
`eyeriss_v2_like_wglb` — but note the asymmetry: v2's extra weight level is
*not* in its paper, while v1's *is*.

## How to run it

Everything goes through `run.sh`, which is the only file you edit. Inside the
container:

    cd /home/workspace
    bash run.sh                    # runs whatever run.sh currently says
    bash run.sh validate           # check the architectures against the shared
                                   # comparison contract. No container needed.
    bash run.sh map                # solve and cache mappings, evaluate nothing
    bash run.sh baseline           # Task 1: conventional ECC, external parity
    bash run.sh diagnose           # audit the architectures, no figure
    bash run.sh panels             # one image, one panel per ECC_PANEL_MODELS
    bash run.sh --replot           # figure only, from results/_raw/
    bash run.sh --dry-run          # validate the config and stop

Mapping generation and energy evaluation are separate, which is what makes
iteration cheap and what stops an energy edit from silently changing a mapping:

    ECC_LAYERS="layer3.0.downsample.0 layer4.1.conv2" bash run.sh map
    ECC_LAYERS="layer3.0.downsample.0 layer4.1.conv2" bash run.sh baseline --eval

`--eval` (`ECC_FROM_CACHE=1`) never invokes Timeloop and needs no container.

`run.sh` defaults with `${VAR:=default}`, so the environment wins over the file.
That makes one-off runs free and means **you never have to edit a file to change
a knob**:

    ECC_SWEEP=arch bash run.sh                  # a different axis
    ECC_CONST_MODEL=resnet50 bash run.sh        # same sweep, different constant
    ECC_SWEEP_KS="51 39 30" bash run.sh         # a shorter BCH sweep

The same overrides exist as flags, for interactive use:

    bash run.sh --sweep model --arch simba_like --k 45
    bash run.sh --sweep bch --values "57 45 30" --replot

## Getting into the container (agents: READ THIS BEFORE GIVING UP ON DOCKER)

**Docker works on this machine and the image is already pulled.** If a run
fails, it is one of the gotchas below — not a missing capability. Do not tell
the user to run the container by hand without trying these first.

Confirmed present: Docker Engine 29.6.1, and
`timeloopaccelergy/timeloop-accelergy-pytorch:latest-amd64` (7.7 GB) local, so
no pull is needed.

### The invocation that works

    MSYS_NO_PATHCONV=1 docker run --rm \
      -v "$(pwd -W)":/home/workspace -w /home/workspace \
      -e ECC_SWEEP=arch -e ECC_CONST_MODEL=resnet18 \
      -e ECC_LAYERS="layer1.0.conv1 layer1.0.conv2" \
      -e ECC_VICTORY=100 -e ECC_MAPPER_THREADS=8 \
      timeloopaccelergy/timeloop-accelergy-pytorch:latest-amd64 \
      bash -lc "cd /home/workspace && bash run.sh map"

### Why each piece is there

1. **`MSYS_NO_PATHCONV=1` and `pwd -W`, not `pwd`.** This is almost certainly
   what makes an agent conclude "Docker is unavailable". Git Bash rewrites any
   argument that looks like a POSIX path, so `-w /home/workspace` becomes
   `C:/Program Files/Git/home/workspace` and Docker rejects it:

       docker: Error response from daemon: the working directory
       'C:/Program Files/Git/home/workspace' is invalid, it needs to be an
       absolute path

   That error is about **argument mangling in the shell**, not about Docker,
   the daemon, or the image. `MSYS_NO_PATHCONV=1` stops the rewriting;
   `pwd -W` gives the Windows-style path the daemon needs for the bind mount.
   Plain `docker images` / `docker info` work without either, which is why
   Docker can look fine right up until the first `run`.

2. **`bash -lc "..."`.** The image's entrypoint starts a Jupyter server, so a
   bare command can be lost in its startup output. `bash -lc` runs the command
   and exits cleanly. Expect ~15 lines of s6/Jupyter noise around the real
   output; filter with
   `grep -vE "^\[|jupyter|Jupyter|token=|file:///"`.

3. **`-e VAR=...` for every knob.** `run.sh` uses `${VAR:=default}`, so the
   environment wins over the file. Passing knobs with `-e` is the intended way
   and means never editing `run.sh` for a one-off.

### What actually needs the container, and what does not

| needs the container | runs anywhere |
|---|---|
| `bash run.sh map` | `bash run.sh validate` |
| `bash run.sh` (cold sweep) | `bash run.sh baseline --eval` |
| any run that must invoke the mapper | `bash run.sh --replot` |
| | `python3 -m eccenergy.tests.test_results_store` |
| | `python3 -m eccenergy.tests.test_noc` |
| | `python3 -m eccenergy.tests.test_mapper_lock` |

Only `timeloop.py` needs Timeloop. Anything reading `results/_raw/` or the
mapper cache runs on the host — a local Python with `pandas`, `matplotlib` and
`pyyaml` is enough, which is the point of splitting mapping from evaluation.

### Two practical traps on long runs

* **The 600 s tool timeout.** A cold sweep is minutes to hours. Use
  `run_in_background`, then watch the filesystem for progress —
  `find ecc_energy_study/outputs -name mapper_console.log -newermt '-3 minutes'`
  shows which shape is being solved right now.
* **Piping through `grep` buffers everything.** A backgrounded
  `docker ... | grep ...` writes nothing to its output file until the pipeline
  ends, so it looks hung when it is fine. Watch the cache directories or the
  expected output file instead of tailing the pipe.

### Cost of a cold run, measured

At `ECC_VICTORY=100`, `ECC_MAPPER_THREADS=8`, six architectures:

* `layer3.0.downsample.0` + `layer4.1.conv2` (2 shapes): **~3.5 min total**
* `conv1` + `layer1.0.conv*` (2 shapes, but `conv1` is C3 M64 R7 S7 **P112
  Q112**): **~13 min total** — a large output feature map dominates mapper
  time, so pick development layers for their shape, not their position in the
  network.

## Running on HiPerGator (SLURM + Apptainer, no Docker)

The project also lives at `/blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling`
on UF HiPerGator (login `hpg.rc.ufl.edu`, group `rewetz`). Everything said
above about the container holds there with one substitution: the same image
runs under Apptainer, through the wrapper `hpc/tl.sh`:

    cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
    module load apptainer
    bash hpc/tl.sh bash run.sh validate          # any run.sh command goes here

`ECC_*` variables in the shell pass straight through, as `-e VAR=` did with
Docker. Rules that differ from the laptop:

- **Never run the mapper on a login node.** Get cores first:
  `srun --account=rewetz --qos=rewetz --cpus-per-task=18 --mem=8gb --time=02:00:00 --pty bash -i`,
  or submit with `sbatch`. `validate`, `--replot`, `--eval`, `diagnose` and
  the tests do not call Timeloop.
- Keep `ECC_MAPPER_THREADS=18` and `--cpus-per-task=18`: the thread count is in
  the mapping fingerprint, and 18 is what every laptop cache entry used, so the
  two machines' `ecc_energy_study/outputs/` trees merge as cache hits.
- The image is `/blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling/timeloop.sif` (`ECC_SIF`
  overrides); build it once with `apptainer pull` as `hpc/HIPERGATOR.md` §3 says.
- **One command runs everything**: `bash hpc/run_all.sh` submits the
  (architecture, model) job array `hpc/map.sbatch` and a dependent
  `hpc/eval.sbatch` that evaluates and draws when the array succeeds
  (`TASKFILE=hpc/tasks_full.txt` for the 8 x 8 matrix). By hand:
  `sbatch hpc/map.sbatch`, then `bash hpc/eval_panel.sh` and
  `python3 hpc/summary.py`. Built 2026-09-06 after
  the five single-process checks passed. `%N` is the only concurrency cap; the
  `rewetz` investment is 181 cores, so 18-core tasks fit 10 at a time.
  `hpc/HIPERGATOR.md` §7 has the race analysis and the cost model.

Transfer commands, image build, the verification checklist and the daily
VS Code workflow: `hpc/HIPERGATOR.md`.

## Development mode: one or two layers

`ECC_LAYERS` selects layers by their **stable workload name** (`layer4.1.conv2`,
never an index). Empty means the full model. Everything a selected-layer run
writes is namespaced by the selection, so a two-layer number can never be read
back as a full-model number.

The recorded resnet18 development pair:

| layer | shape | weights | why |
|---|---|---|---|
| `layer3.0.downsample.0` | C128 M256 R1 S1 P14 Q14 s2 | 32,768 | the quick check; maps in seconds everywhere |
| `layer4.1.conv2` | C512 M512 R3 S3 P7 Q7 | 2,359,296 | the contrast: 72x the weights, tiny fmap, so weight-dominated |

Quick-run numbers validate the implementation. They are not an architecture
ranking, and every result JSON says so in `warnings`.

## The architecture contract

`archs/_shared/standard.yaml` states what must be IDENTICAL across designs --
weight precision, activation precision, DRAM geometry, process node, the
dense-packing rule -- and what each design keeps as its own with a citation.
`archs/_shared/provenance.yaml` records, level by level, where each declared
number came from. `bash run.sh validate` checks every arch YAML against both
and exits non-zero on a violation.

Each design declares a `source:` in `standard.yaml`, and **only the first of
the four licenses using the design's name as if it were the chip**:

| `source:` | means |
|---|---|
| `published` | every geometry claim is in the cited paper |
| `derived` | a variant of a published design differing in one stated block; the variant's own level is not the paper unless its citation says so |
| `reference_design` | shipped by timeloop-accelergy-exercises, *named* after a paper but not reproducing its geometry — `simba_like` is the one |
| `locally_authored` | a reference dataflow written here; no paper, nothing to cite |

`validate` prints this as `origin:` per design. If you are about to write "our
Simba results", check the origin first.

Accumulator width is deliberately **not** standardized: Eyeriss v1 accumulates
at 16b, v2 at 20b, Simba at 24b, each cited. Forcing a common width would be
equalising the architectures. `ECC_ACC_BITS` does force one, for a **sensitivity
study only** -- it gets its own mapper cache and its own results namespace and
is labelled as such everywhere it appears.

## Layout

    run.sh                  the knob file. Editing anything else is unusual.
    hpc/                    HiPerGator: HIPERGATOR.md (transfer, image, verification,
                            workflow, parallel design), tl.sh (apptainer
                            wrapper), map.sbatch + tasks_panel.txt (the
                            job array), eval_panel.sh (cache-only evaluation)
                            and summary.py (the model x arch matrix).
    FINDINGS.md             what the most recent audit / evaluation found, with
                            the numbers and how they were obtained. The ONLY
                            place empirical claims about the current model live.
    progress.txt            the status board: every task and subtask marked
                            with one of ✅ ❌ ⚠️. Update both at session end.
    eccenergy/              the package
      config.py             every ECC_* variable -> validated Config. The ONLY
                            module that reads os.environ.
      paths.py              every filesystem location, incl. the results layout
      archs.py              install / patch / audit architecture YAMLs
      workloads.py          CNN layers and transformer matmuls -> [Layer].
                            Grouped/depthwise convolutions carry Layer.G (the
                            group count); C and M are PER GROUP, and the shape
                            name gets a `_G<g>` suffix so ungrouped shapes keep
                            their cached names.
      timeloop.py           the mapper interface + stats parsing. The only slow
                            module, and the only one that needs the container.
                            Two problem templates: the plain cnn_layer shape and
                            a grouped one with a G dimension, chosen per layer.
      energy.py             stats -> plotted categories; the raw-energy cache
      ecc.py                the three arms; DC reconstruction energy, by (N,K)
      parity.py             external BCH parity: grouping, padding, DRAM
                            granularity, and the payload/parity hand check
      results_store.py      THE result writer. Nothing else writes an
                            evaluation JSON. See docs/RESULTS_SCHEMA.md.
      plots/style.py        palette, units, save paths
      plots/stacked.py      draw_panel(), the ONLY thing that draws a bar in
                            this project, plus grouped_stacks(), the
                            single-panel figure the three sweeps produce
      plots/panels.py       the same draw_panel(), once per model, on one page
      experiments/sweep.py  the one driver: pick the groups, draw one figure
      experiments/panels.py one image, one panel per ECC_PANEL_MODELS entry
      experiments/common.py Session: setup, collection, reporting
      experiments/diagnose.py  the architecture audit
      experiments/baseline.py  Task 1: the conventional-ECC result
      experiments/validate.py  the standardized-comparison contract check
      tests/                the offline test suite (no pytest, no container)
      generate.py           workload generators (needs torch for CNNs)
    archs/_shared/          standard.yaml (the comparison contract),
                            provenance.yaml (where every number came from) and
                            noc.yaml (the interconnect energy coefficients,
                            each cited; injected into every design's spatial
                            containers by archs._inject_noc).
                            NOT an architecture; skipped by the installer.
    archs/<name>/           architectures authored here, with their README.
                            arch.yaml       the design as authored (or absent,
                                            meaning "use example_designs")
                            arch_paper.yaml the same design with storage
                                            precisions and scratchpad sizes set
                                            to published values, each cited in
                                            a comment. Used by default; see
                                            ECC_ARCH_FIDELITY below.
    data/dc/                Design Compiler reconstruction energies
    ecc_energy_study/       AUTO-MANAGED: cloned repo + mapper cache. Do not
                            delete outputs/ — it is hours of compute.
    results/                all output, see below
    legacy/                 pre-restructure scripts and figures, plus the
                            archived FINDINGS.md. Nothing imports from here,
                            and nothing in it describes the current model.

## Results layout

    results/
      _raw/<arch>/<treat>/fp-<hash>/<workload>/<scope>/cls-<mode>/rw-<split>/<model>.json
      evaluation/{Pre|Post}/<arch>/<model>/<bch>/<prec>/<scope>/<mapper>/<runid>.json
      figures/{BCHsweep,ModelSweep,ArchitectureSweep}.{png,pdf}
      figures/<sweep>__panels__<model>__<model>.{png,pdf}   ECC_EXPERIMENT=panels
      tables/     the same stems, .csv   (plus diagnose.csv)
      manifests/  the same stems, .json  -- the config behind each file
                  (plus validate.json)

`results/evaluation/` is the structured result store: one JSON per
(phase, architecture, model, code, precision, layer scope, mapper config, run),
holding every ECC variant together -- the ones that were evaluated and the ones
that were not, the latter with `total_energy_pJ: null` and a reason. It is
never silently overwritten. `docs/RESULTS_SCHEMA.md` is the full description.

The three-figures-three-names rule below still governs `figures/`, `tables/`
and `manifests/`; it does not govern `evaluation/`, which accumulates runs on
purpose.

**Three sweeps, three names, and that is the whole directory.** The stem comes
from `cfg.stem`, which is decided by the sweep alone, so re-running at different
constants REWRITES the file rather than adding another. To keep an old figure,
copy it out — or point `ECC_RESULTS_DIR` somewhere else.

The manifest is what makes that safe: it records the constants, the code
geometry and the mapper fingerprint behind the file currently on disk.

`ECC_EXPERIMENT=panels` obeys the same rule rather than escaping it: it still
writes exactly one figure, one table and one manifest, and its stem is still
decided by the configuration alone. Because the panel models are in that stem,
a two-model figure overwrites only the previous run of the SAME pair.

`results/_raw/` is the key to fast iteration. Raw energies are pure Timeloop
output and **do not depend on the ECC configuration**, so changing the code
geometry and re-running is milliseconds:

    ECC_CONST_K=36 bash run.sh --replot

## The two caches, and why they matter

1. **Mapper cache** --
   `ecc_energy_study/outputs/<arch>/<subdir>/fp-<hash>/<shape>/`.
   One entry per layer shape per architecture. Cold, a full 8-model sweep is
   hours; warm, it is seconds. Runs are resumable: every solved shape is saved
   immediately. **Never delete this.**

   `fp-<hash>` is the mapping fingerprint: a hash of the patched architecture
   YAML the mapper actually sees, the globals (node, clock) and every mapper
   setting. Each entry also carries a `mapping.json` sidecar naming the
   fingerprint, the tool versions and a `mapping_id` that results reference.
   An entry whose fingerprint differs is a MISS. Before this existed, editing
   an arch.yaml left the cache path unchanged, so the next run reused mappings
   computed for the previous geometry and reported them as the new design.

   Pre-Task-1 entries have no sidecar and are therefore unusable by default.
   They are left on disk. `ECC_CACHE_STRICT=0` accepts them, labels the result
   `legacy` and warns; do not publish from it.

2. **Raw energy cache** — `results/_raw/`. Derived from the mapper cache by
   parsing stats. Cheap to rebuild with `ECC_FROM_CACHE=1` (no container
   needed), which is what to do after changing `ECC_CLASSIFY`. It is the one
   thing under `results/` worth keeping.

Any treatment that changes what the mapper sees, or what it optimises, gets its
own mapper-cache subdirectory, so existing results are never mixed in. Three
classes:

- `ECC_OPT_METRIC`, `ECC_VICTORY`, `ECC_VICTORY_SCALING` change what the search
  returns for every design at once. So does `ECC_NOC` (and its two sub-knobs):
  a costed interconnect is a different architecture to the mapper.
- `ECC_FORCE_TECHNOLOGY`, `ECC_DRAM_DEPTH`, `ECC_GLOBAL_CYCLE_SECONDS` go through
  `globals.yaml`, which costs DRAM for every design at once — they invalidate
  every architecture's cache.
- `ECC_ARCH_FIDELITY` and `ECC_FORCE_DATAWIDTH` are evaluated **per
  architecture** by `archs.effective_variant()`: fidelity by whether the design
  has an `arch_paper.yaml`, datawidth by diffing the patched YAML. A design
  already declared at the weight width is unchanged and keeps its cache.

One historical setting — `stock` fidelity, `edp`, victory 500, unscaled — is
spelled as the *empty* treatment, so `outputs/<arch>/multimodel` stays reachable
and mappings computed under it are still addressable:

    ECC_ARCH_FIDELITY=stock ECC_OPT_METRIC=edp ECC_VICTORY=500 \
      ECC_VICTORY_SCALING=none bash run.sh --replot

Any change to the defaults means **a default run needs a fresh mapper cache**.
Budget a cold run accordingly; it is resumable, and nothing already computed is
lost.

`_force_datawidth` deliberately skips any level whose `keep:` list is `Outputs`
alone. Accumulator precision is a separate design choice from operand
quantization, and Timeloop asserts `width % datawidth == 0`, so forcing an
operand width onto a psum level whose width is not a multiple of it aborts the
mapper on every layer.

## The three ECC arms

All three share ONE BCH(N,K) codeword over `ECC_WEIGHT_BITS`-bit weights, so they
differ only in where the parity lives. Codewords are counted from DRAM weight
reads: `n_codewords = dram_weight_reads / (K / weight_bits)`.

- **baseline** — parity beside the data in DRAM; weight traffic inflates by N/K.
- **embedded** — parity inside the already-stored word; DRAM unchanged.
- **recon** — DRAM as embedded, but only K/N of the weights are held on chip;
  the parity portion is regenerated by a synthesized datapath, charged per
  codeword from `data/dc/BCH_N63_results.json`.

The first entry in `ECC_APPROACHES` is the reference the savings are measured
against.

## The knobs that decide what gets modelled

Mechanism only. What the right *setting* of each is, and why, is exactly what
the rewrite is re-deciding — do not restate old justifications here.

- **`ECC_ARCH_FIDELITY`** (`paper` | `stock`) chooses which YAML each design is
  mapped from. `paper` prefers `archs/<name>/arch_paper.yaml`, where storage
  precisions and scratchpad sizes are the numbers the design's paper publishes,
  each cited in a comment beside it. `stock` uses the design exactly as
  timeloop-accelergy-exercises ships it. A design without an `arch_paper.yaml`
  keeps its `arch.yaml` at both fidelities.
- **`ECC_OPT_METRIC`** selects the mapper's objective. Timeloop is told the
  metric from `timeloop.py`, so the cloned exercises repo stays pristine.
- **`ECC_MAPPER_ALGORITHM`, `ECC_MAPPER_SEARCH_SIZE`, `ECC_MAPPER_THREADS`**
  select Timeloop's search algorithm, cap the valid mappings examined per
  thread, and pin the thread count. All three are in the mapping fingerprint,
  so each combination has its own mapper cache. Which setting is converged
  enough to rank architectures is an empirical question -- see FINDINGS.md
  before quoting an ordering produced under any setting.
- **`ECC_NOC`** (default 1) costs the interconnect. Timeloop's own wire model
  is a stub that returns 0, so with it off every network is free -- in the
  evaluator and in the mapper's objective. Coefficients and citations:
  `archs/_shared/noc.yaml`. `ECC_NOC_WIRE_PJ_PER_BIT_MM` overrides the shared
  wire constant, `ECC_NOC_ROUTER_PJ` the shared per-flit router energy, and
  `ECC_NOC_SCALE` multiplies every term for sensitivity runs.
  All four are in the fingerprint and the slug (`noc`), so no pre-NoC mapping
  is ever read back as a costed one. NoC is its own plotted category; the
  on-chip storage category's internal key is `Local (spads/RF)`.
- **`ECC_VICTORY` and `ECC_VICTORY_SCALING`** set mapper effort. The search
  abandons a thread after `victory_condition` consecutive non-improving
  mappings, and the candidate count grows combinatorially with loop-nest depth,
  so a flat setting searches a deep hierarchy less thoroughly than a shallow
  one. `levels` scaling raises the budget with nest depth. It is a heuristic,
  not a proof: to *confirm* convergence, raise `ECC_VICTORY` and check the
  totals do not move.

`bash run.sh diagnose` flags a psum level narrower than its own accumulator as
a defect. A level that is legitimately narrower says so in the YAML with a
`# psum-width-ok: <reason>` comment, and the audit lists it as a declared claim
instead.

## Known modelling caveats

**Intentionally not written down here.** Run `bash run.sh diagnose`, which
computes the list from the architectures as they currently stand and prints it
with current numbers. A prose copy in this file went stale once already and
then contradicted the code.

## Working on this code

- **Adding an architecture**: drop `archs/<name>/arch.yaml` in, add the name to
  `KNOWN_ARCHS` and `ARCH_LABELS` in `eccenergy/config.py`, add it to
  `archs/_shared/standard.yaml` (with its accumulator width and a citation for
  it) and to `archs/_shared/provenance.yaml`, then run `bash run.sh validate`
  -- it will fail until the design obeys the shared contract. Then
  `ECC_SWEEP_ARCHS="<name> eyeriss_like" bash run.sh diagnose` to check it
  against the others before committing to a long sweep. Declare partial sums at
  the accumulator width at *every* level that holds them — the diagnose audit
  will tell you if you did not — or, where a level legitimately holds
  requantized values instead, say so with a `# psum-width-ok: <reason>` comment
  on the datawidth line. Add an `arch_paper.yaml` if the design comes from a
  paper, and cite each number in a comment.
- **Adding a model**: add it to `CNN_MODELS` or `TRANSFORMER_MODELS` in
  `config.py` — that list is what decides which workload file a sweep reads.
  Regenerate the workload file in the container (`python3 -m eccenergy.generate
  models <name>`); grouped convolutions and per-pixel Linear layers are handled
  by the generator, nothing else to declare.
- **Adding an ECC approach**: add it to `APPROACHES` in `config.py` and a branch
  in `build_stacks()` in `ecc.py`. Everything else follows.
- **Adding a sweep axis**: a name in `SWEEPS`, a stem in `SWEEP_STEMS`, the
  axis resolution in `Config.__post_init__`, and a `_<name>_groups()` builder in
  `experiments/sweep.py`. Return `(groups, stacks, labels, fontsize)` and the
  existing renderer draws it — do not write a new renderer, and do not add a
  second figure to a sweep.
- **Changing how a bar looks**: edit `draw_panel()` in `plots/stacked.py` and
  nothing else. It is the only place a bar is drawn, so every figure — the
  three sweeps and the panelled one — moves together. If you find yourself
  wanting a second bar-drawing routine, add a parameter to that one instead.
- **Never** read `os.environ` outside `config.py`, and never resolve a path
  outside `paths.py`.
- The old scripts resolved `ecc_energy_study/` from `cwd`, so launching from the
  wrong directory silently re-cloned the repo and re-ran the whole mapper. The
  package resolves it from the project root instead; launch directory no longer
  matters.

## Line endings

The shell scripts run inside a Linux container. A CRLF in one makes bash read
`pipefail` as an option name and die with a mangled, self-overwriting message:

    : invalid option namene 24: set: pipefail

`.gitattributes` forces LF on `*.sh *.py *.yaml *.yml *.md *.json`, and
`bash tools-fix-eol.sh` repairs any file that slips through (it removes only
carriage returns, so it is safe on cached results). **When writing or editing
any file in this project, always emit LF.**

## Verification

There is a legacy regression baseline —
`legacy/figures/multiarch_bch63_k51_*_summary_uJ.csv` — that the pipeline once
reproduced field-for-field at `ECC_SWEEP=arch ECC_CLASSIFY=name`, historical
treatment, BCH(63,51)/resnet18. Its legacy column prefixes are
`base_`/`embe_`/`recon_` where the new tables write the full approach names.

**Whether that comparison should still hold is a question for the rewrite, not
an assumption.** If the new model deliberately changes what is being computed,
a mismatch is the intended outcome, not a regression — decide which it is
before treating a difference as a bug.
