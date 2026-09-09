# CLAUDE.md

Energy modelling for relaxed error correction in DNN accelerators, on Timeloop +
Accelergy. The question: **if ECC parity does not have to be stored in DRAM, how
much inference energy does that save, and does the answer depend on the
accelerator?**

**No empirical claims in this file.** Numbers live in `FINDINGS.md`; the live
caveat list is `bash run.sh diagnose`, computed from the architectures as they
stand. A prose copy here went stale once and then contradicted the code.
`legacy/` is pre-rewrite and describes nothing current.

## Running it

`env.sh` is the only file you edit: every `ECC_*` knob, ten commented sections,
sourced by `run.sh`, `hpc/run_all.sh`, `hpc/map.sbatch` and `hpc/tl.sh`, so a
value cannot mean one thing to the mapper and another to the evaluator. Sections
1-9 are knobs; section 10 is derived (it turns the `ECC_ARCHS`/`ECC_MODELS`/
`ECC_KS` lists into the swept-list-plus-two-constants form `config.py` reads,
picks the stem, and generates the SLURM task list). Read env.sh for what a knob
does — it is commented; do not restate it here.

Every value is written `${VAR:=default}`, so **the environment wins over the
file** and a one-off never needs an edit.

    bash hpc/run_all.sh          # THE command: map array -> dependent eval+plot
    bash hpc/run_all.sh --map-only | --eval-only | --replot | --local
    ECC_RECON_MODELING=1 bash hpc/run_all.sh --eval-only     # Task 3

`run.sh` runs ONE stage of that and has no knobs of its own: `map`, `baseline`,
`embedded --eval`, `recon --eval`, `validate`, `diagnose`, `panels`, `--replot`,
`--dry-run`.

**Only `timeloop.py` needs the container.** `--eval` (`ECC_FROM_CACHE=1`) never
invokes Timeloop; validate, diagnose, replot and the whole `eccenergy/tests/`
suite run on any python with pandas, matplotlib and pyyaml. `test_recon.py`
additionally re-runs its property tests against the real mapper cache when
pandas and that cache are present, and says so when it skips them.

**HiPerGator** (the machine this now lives on): `module load apptainer`, then
`bash hpc/tl.sh <any run.sh command>`. Never run the mapper on a login node —
`srun --account=rewetz --qos=rewetz --cpus-per-task=18 --mem=8gb --time=02:00:00 --pty bash -i`,
or `sbatch`. Keep `ECC_MAPPER_THREADS=18` and `--cpus-per-task=18`: the thread
count is in the mapping fingerprint, so any other value is a cold cache.
`hpc/HIPERGATOR.md` has transfer, image build, verification and the cost model.
On a laptop the same image runs under Docker; in Git Bash use
`MSYS_NO_PATHCONV=1` and `pwd -W`, or the bind mount and `-w` are rewritten into
nonsense and it looks like Docker is missing when it is not.

**A cold map is hours.** Narrow before widening
(`ECC_ARCHS=<one> ECC_MODELS=<one> bash hpc/run_all.sh --map-only`) and watch
the filesystem rather than a pipe — a backgrounded `docker ... | grep` buffers
until the pipeline ends and looks hung when it is fine.

## One sweep, two constants

All three ECC arms (`baseline`, `embedded`, `recon`) are always drawn, so the
arms are never an axis. A run sweeps exactly ONE of the remaining three axes and
holds the other two at a constant:

| `ECC_SWEEP` | x axis | sweep list | held | stem |
|---|---|---|---|---|
| `bch` | BCH(63,K) | `ECC_SWEEP_KS` | arch, model | `BCHsweep` |
| `model` | networks | `ECC_SWEEP_MODELS` | arch, K | `ModelSweep` |
| `arch` | accelerators | `ECC_SWEEP_ARCHS` | model, K | `ArchitectureSweep` |

`cfg.archs`/`cfg.models`/`cfg.code_k` are derived in `config.py` from that
choice; nothing below `config.py` except `sweep.py` knows which axis is swept. A
model sweep is all-CNN or all-transformer — mixing families is a config error,
they come from different workload files.

**`ECC_EXPERIMENT=panels`** draws one image with one panel per
`ECC_PANEL_MODELS`, repeating the same sweep inside each. It adds no axis and no
renderer, exists only for the "does the ranking survive changing the network"
question, and refuses `ECC_SWEEP=bch` (that would vary two axes between panels).
Panels share a legend, categories and unit but deliberately **not** a y limit,
and the figure says so on itself.

**`ECC_RECON_MODELING=1`** (env.sh §4) is not a fourth sweep: its axis is WHERE
on the weight path the reconstruction boundary sits. §4 overrides §3, collapsing
the run to one architecture, one model and one code, stem `ReconSweep`. It is
one architecture at a time on purpose — the three sweeps can put architectures
on an axis because every design has all three ECC *arms*, but a reconstruction
*boundary* is design-specific, and drawing them together would put "reconstruct
after the mesh" beside a design with no mesh.

## Results layout

    results/_raw/<arch>/<treat>/fp-<hash>/<workload>/<scope>/cls-<mode>/rw-<split>/<model>.json
    results/evaluation/{Pre|Post}/<arch>/<model>/<bch>/<prec>/<scope>/<mapper>/<runid>.json
    results/{figures,tables,manifests}/<stem>.{png,pdf|csv|json}

**Three sweeps, three names, and that is the whole of `figures/`, `tables/` and
`manifests/`.** The stem comes from the configuration alone, so re-running at
different constants REWRITES the file instead of adding one; the manifest beside
it records the constants, code geometry and mapper fingerprint behind what is on
disk. Copy a figure out, or point `ECC_RESULTS_DIR` elsewhere, to keep it.
`panels` obeys the same rule (its stem carries the panel models).

`results/evaluation/` is the exception and accumulates on purpose: one JSON per
(phase, arch, model, code, precision, scope, mapper config, run), holding every
ECC variant including the ones not evaluated (`total_energy_pJ: null` plus a
reason). Nothing but `results_store.py` writes one. Schema:
`docs/RESULTS_SCHEMA.md`.

## The two caches

1. **Mapper cache** — `ecc_energy_study/outputs/<arch>/<treat>/fp-<hash>/<shape>/`,
   one entry per (architecture, layer shape). **Never delete it**; it is hours of
   compute and every solved shape is saved immediately, so runs resume.
   `fp-<hash>` is a hash of the patched YAML the mapper actually sees plus the
   globals and every mapper setting, and each entry carries a `mapping.json`
   sidecar that results reference. A different fingerprint is a MISS — before
   this existed, editing an arch.yaml left the path unchanged and the next run
   reported old mappings as the new design. Entries with no sidecar (pre-Task-1)
   are refused unless `ECC_CACHE_STRICT=0`, which labels the result `legacy`.
2. **Raw energy cache** — `results/_raw/`, parsed from the mapper cache. Pure
   Timeloop output, independent of the ECC configuration, so changing the code
   geometry and redrawing is milliseconds (`ECC_CONST_K=36 bash run.sh --replot`).
   The one thing under `results/` worth keeping.

`ECC_RERUN_OPTIMISER=1` re-solves a shape on a valid hit and overwrites both
records; it is refused together with `--eval`. Anything that changes what the
mapper sees or optimises gets its own cache subdirectory: mapper settings and
`ECC_NOC` (a costed interconnect is a different architecture to the mapper);
`globals.yaml` knobs, which invalidate every design at once; and
`ECC_ARCH_FIDELITY` / `ECC_FORCE_DATAWIDTH`, evaluated per architecture by
`archs.effective_variant()`. So **anything touching env.sh §2/§5, or an arch
YAML, takes the whole matrix cold at once** — budget for it.

## The three ECC arms

One BCH(N,K) codeword over `ECC_WEIGHT_BITS`-bit weights under all three, so
they differ only in where the parity lives. Codewords are counted from DRAM
weight reads. The first entry in `ECC_APPROACHES` is the reference savings are
measured against.

- **baseline** — parity beside the data in DRAM; weight traffic inflates by N/K.
- **embedded** — parity inside the stored weights, laid out as the embedding
  pipeline (ECC-CODE-Engine / Input_Embedding) does it: the MSB-first weight bit
  stream cut into n-bit codewords, so weights straddle codewords and the n-k
  lowest-significance positions carry parity. DRAM traffic is exactly
  Timeloop's — the complete codeword is read for correction.
- **recon** — DRAM as embedded, K/N of the weights held on chip, the rest
  regenerated by a synthesized datapath charged per codeword from
  `data/dc/BCH_N63_results.json`.

`build_stacks()`'s `recon` arm is ONE point applied to every design at once, and
no physical boundary does what it does (it both scales on-chip weight energy by
K/N *and* charges one reconstruction per DRAM codeword). It stays as the third
bar of the three sweeps; it is **not** one of Task 3's boundaries and must not be
quoted as one.

## `recon.py` — the placement space

Two tables define it and **must be edited together**:
`WEIGHT_PATHS[<arch>]` (the stages of that design's weight path, each naming the
Timeloop levels that are it) and `PLACEMENTS[<arch>]` (the boundaries: which
stages stay reduced, what drives the reconstruction count, whether a reuse
register is present).

That is enforced, not advised. `validate_placement_space()` requires a
placement's `reduced` set to be a **prefix** of the path's reducible stages in
path order, and every reducible stage to be reached by some boundary; it runs
before anything is evaluated. Without it, adding a stage to one table and
forgetting the other keeps every boundary below it reporting its own saving while
showing the new stage at full width — the whole list understated, nothing saying
so. `feasibility()` structurally cannot catch that: it inspects the stages a
placement claims, never the ones it should have claimed.
`weight_path()` also refuses to proceed if a level carrying Weights energy is not
claimed by exactly one stage, and `cross_check()` re-derives per-category weight
energy against the `Raw` record first.

Three things the model refuses to fudge, each with a knob and a recorded check:
**physical packing** (`stream` scales every reduced stage by K/N; `aligned`
gives each weight whole bits and moves no SRAM access count),
**reconstruction granularity** (`G_rec`, the co-resident group, is computed from
the layout, not assumed), and **feasibility** (a PE-local boundary whose resident
tile is smaller than `G_rec` is reported `unsupported` with the layers named,
never estimated). Retention (`ECC_RECON_REUSE_REG_ENTRIES`) is a CAPACITY
question, not a consecutive-use one — `retention_model()` carries the reasoning,
and getting that wrong is what made R4b look useless in the first version.

The DRAM term is two stages since 2026-09-09, because the BCH decoder is on the
DRAM die and off the fetch path (01_project_context §1/§4): `dram_array`,
`(1 − f_if)` of the DRAM weight energy, is never reduced (the array reads the
complete codeword); `dram_interface`, `f_if` of it, scales by K/N under **every**
boundary, R1 included, because only the k message bits leave the die. `f_if`
(`ECC_DRAM_IF_FRAC`) should be a cited DRAM energy breakdown; the current 0.40 is
an **assumption** (2026-09-09, `provenance.yaml` `dram_interface_share` says so and
why) and every figure carries the value it was drawn at, labelled assumed. Unset,
the study refuses and prints the ceiling at 0.10/0.25/0.50. `ECC_RECON_DECODE_SITE=controller` is
the pre-2026-09-09 model (complete codeword across the interface, DRAM identical
on every bar), kept for the diff only. The reference bars keep controller-side
correction and do not move. Two things it deliberately does not credit: the
DRAM array, and any operand-retention saving for anybody — scratchpad read counts
stay Timeloop's under every bar. That used to rest on "the same
register would help a baseline PE too", which was a handicap papering over a
hole: R4b cut its reconstruction count 82.9x BECAUSE its register served those
reads, while still billing them. `ReuseRegister` closes it —
`ECC_RECON_REUSE_REG_MODEL` says WHAT the register holds, and the default
(`complement`, only the n-k regenerated bits, `weight_bits x (1-K/N)` per
weight) cannot serve a read alone, so the read count legitimately does not move
and PE weight storage is 1.00x the baseline. `full_width` reproduces the
opposite reading as a runnable row, `free` reproduces the pre-2026-09-08
numbers. FINDINGS section 7.1 has the audit; do not re-open it from the
docstrings alone.

**The MAC cost is the denominator of every percentage** and is audited in
FINDINGS §7.3: the ERT's 1.16877 pJ/MAC is a single 40 nm table row per
primitive, scaled. `ECC_MAC_PJ_OVERRIDE` (env.sh §5) rescales the Compute
category evaluator-side, after the raw cache, so no saved pJ and no mapping
moves; a value listed in `provenance.yaml` `mac_energy_pj` carries its citation
on every figure, manifest and result file. The default is **0.23 pJ**
(Horowitz ISSCC 2014 int8 mult + add, 45 nm; adopted 2026-09-09) and
`ReconSweep.png` is drawn under it; set it EMPTY to reproduce the ERT
denominator, which is the sensitivity row. The mapper is unaffected either way:
it prices MACs from the ERT, so only an edit to the `intmac` component (a cold
cache) would change a mapping.

**Every result and console run prints the ceiling first** — the weight energy a
boundary could reduce, times `1 - K/N`. Without it a sub-percent saving reads as
a missing term rather than as arithmetic.

## Architecture rules

`archs/_shared/standard.yaml` states what must be IDENTICAL across designs
(operand precisions, DRAM geometry, process node, the dense-packing rule) and
what each keeps as its own with a citation; `provenance.yaml` records where every
declared number came from; `noc.yaml` holds the interconnect coefficients and,
per design and with the paper section, **which spatial containers actually are
the NoC**. Two NoC terms Timeloop cannot be given (spatial-reduction adders, the
psum word width on the wire) are charged after mapping by `noc_post.py` from
`noc.yaml`'s `evaluator_only` block; every `Raw` carries their stamp and is
re-gathered if it differs. `archs/_shared/components/` is part of the mapper
fingerprint: editing a component colds every cache, on purpose. `bash run.sh
validate` checks all of it and exits non-zero on a violation.

Only one of the four `source:` values licenses using a design's name as the chip:
`published`. `derived` is a variant differing in one stated block,
`reference_design` is shipped by timeloop-accelergy-exercises and merely *named*
after a paper, `locally_authored` reproduces nothing. `validate` prints it as
`origin:`. **`simba_like` is `reference_design`** — inspired by Simba, not
reproducing it (its geometry fails two checks against the paper, and its on-chip
weight capacity sets its DRAM refetch and therefore its ECC saving). Label it
"Simba-like (reference design)"; there is no "Numba" architecture here.

**Eyeriss v1 is a bracketing PAIR, not a number.** JSSC 2017 allocates 8 kB of
the 108 kB GLB to filter weights: `eyeriss_like` does not model it,
`eyeriss_like_wglb` does, and the files differ in exactly one block. Quote both
as an upper/lower bound on DRAM weight traffic and hence on the ECC saving; a
single number is a choice of bound. Same rule for `eyeriss_v2_like_wglb`, with
the asymmetry that v2's extra weight level is *not* in its paper while v1's is.

Accumulator width is deliberately not standardized (v1 16b, v2 20b, Simba 24b,
each cited) — forcing one would equalise the architectures. `ECC_ACC_BITS` does
force one for a sensitivity study only, with its own cache and namespace.
`diagnose` flags a psum level narrower than its own accumulator; a level that is
legitimately narrower declares `# psum-width-ok: <reason>` in the YAML.

## Working on this code

- **Task 1's code is frozen**: `parity.py`, `baseline.py` and `build_stacks()`
  are not refactored. Add modules instead, then re-run `bash run.sh baseline
  --eval` and diff — the Task 1/2 totals must not move. `audit.py` mirrors
  `baseline.py`'s checks for later tasks; keep them in step.
- **Never** read `os.environ` outside `config.py`, and never resolve a path
  outside `paths.py`.
- **Adding an architecture**: `archs/<name>/arch.yaml` (plus `arch_paper.yaml`
  with each number cited if it comes from a paper), the name in `KNOWN_ARCHS` and
  `ARCH_LABELS` in `config.py`, entries in `standard.yaml` and `provenance.yaml`,
  then `bash run.sh validate` and `diagnose` against an existing design before
  committing to a long sweep. Declare partial sums at the accumulator width at
  every level that holds them.
- **Adding a model**: add to `CNN_MODELS` or `TRANSFORMER_MODELS` in `config.py`
  (that list decides which workload file is read), then
  `python3 -m eccenergy.generate models <name>` in the container. Grouped
  convolutions and per-pixel Linear layers are handled by the generator.
- **Adding an ECC approach**: `APPROACHES` in `config.py` plus a branch in
  `build_stacks()`.
- **Adding an architecture to the placement study**: `WEIGHT_PATHS[<name>]` and
  `PLACEMENTS[<name>]` together (see above). Get the stage-to-level match right
  by reading a real `timeloop-mapper.stats.txt` from that design's cache.
  `eyeriss_v2_like_wglb` is currently refused for exactly this reason — its
  `weight_glb` stage has no boundary.
- **Adding a sweep axis**: a name in `SWEEPS`, a stem in `SWEEP_STEMS`, the
  resolution in `Config.__post_init__`, and a `_<name>_groups()` in
  `experiments/sweep.py` returning `(groups, stacks, labels, fontsize)`. Do not
  write a second renderer and do not add a second figure to a sweep.
- **Changing how a bar looks**: `draw_panel()` in `plots/stacked.py` is the only
  place a bar is drawn in this project, so every figure moves together. Want a
  second bar routine? Add a parameter to that one instead (`bar_notes` exists
  because a sub-percent result cannot be read off a full-height bar).
- **Line endings**: the shell scripts run inside a Linux container and a CRLF
  makes bash die on `set -o pipefail` with a mangled message. `.gitattributes`
  forces LF; `bash tools-fix-eol.sh` repairs anything that slips through.
  **Always emit LF.**

## Layout

Everything not listed here is what its name says; `eccenergy/` module docstrings
carry the rest.

    env.sh              THE knob file        run.sh    one stage, no knobs
    hpc/                run_all.sh (the one command), map.sbatch, tl.sh
                        (apptainer wrapper), summary.py, HIPERGATOR.md
    FINDINGS.md         the only place empirical claims about the current model
                        live      progress.txt / PROJECT_STATUS.md  status boards
    eccenergy/          config.py (the ONLY reader of os.environ), paths.py (the
                        ONLY resolver of paths), workloads.py, timeloop.py (the
                        only slow module and the only one needing the container),
                        energy.py, ecc.py, recon.py, parity.py, embedded.py,
                        noc_post.py (evaluator-only NoC terms Timeloop cannot cost),
                        results_store.py (the ONLY writer of an evaluation JSON),
                        plots/, experiments/, tests/, generate.py
    archs/_shared/      standard.yaml, provenance.yaml, noc.yaml -- not an
                        architecture; skipped by the installer
    ecc_energy_study/   AUTO-MANAGED: cloned repo + mapper cache. Do not delete.
    legacy/             pre-restructure scripts, figures and FINDINGS -- nothing
                        imports it, nothing in it describes the current model.
                        The exception is FINDINGS_detail_<date>.md: the full
                        working detail behind the current FINDINGS.md, archived
                        there to keep the live file readable.
