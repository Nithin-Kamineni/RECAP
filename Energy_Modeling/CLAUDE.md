# CLAUDE.md

Energy modelling for relaxed error correction in DNN accelerators, on Timeloop +
Accelergy. The question: **if ECC parity does not have to be stored in DRAM, how
much inference energy does that save, and does the answer depend on the
accelerator?**

**The live plan is `prompt_6.md`** — reconstruction-aware mapping: put the
encoder's energy into the mapper's objective via the ERT. Read it before starting
work; where it and this file differ on what to do next, it wins. It also carries
`prompt_3.md`'s constrained mapspace, which every mapper job now runs under.

**No empirical claims in this file.** Numbers live in `FINDINGS.md`; the live
caveat list is `bash run.sh diagnose`, computed from the architectures as they
stand. A prose copy here went stale once and then contradicted the code.

⚠ **Every reconstruction energy number in the project is pending regeneration.**
prompt_6 RULE 3 separates the encoder's per-codeword and per-cycle terms, which
were previously added together. See FINDINGS.md's banner.

`legacy/` is pre-rewrite and describes nothing current, except the dated
`FINDINGS_detail_*`, `progress_*` and `PROJECT_STATUS_*` snapshots, which are the
full working record.

## Running it

`env.sh` is the only file you edit: every `ECC_*` knob, ten commented sections,
sourced by `run.sh`, `hpc/run_all.sh`, `hpc/map.sbatch` and `hpc/tl.sh`, so a
value cannot mean one thing to the mapper and another to the evaluator. Sections
1–9 are knobs; section 10 is derived. **Read env.sh for what a knob does — it is
commented; do not restate it here.**

Every value is written `${VAR:=default}`, so **the environment wins over the
file** and a one-off never needs an edit.

    bash hpc/run_all.sh          # THE command: map array -> dependent eval+plot
    bash hpc/run_all.sh --map-only | --eval-only | --replot | --local

`run.sh` runs ONE stage and has no knobs of its own: `map`, `baseline`,
`embedded --eval`, `recon --eval`, `dilation`, `validate`, `diagnose`, `panels`,
`--replot`, `--dry-run`.

**Only `timeloop.py` needs the container.** `--eval` (`ECC_FROM_CACHE=1`) never
invokes Timeloop; validate, diagnose, replot and the whole `eccenergy/tests/`
suite run on any python with pandas, matplotlib and pyyaml.

**HiPerGator:** `module load apptainer`, then `bash hpc/tl.sh <any run.sh command>`.
Never run the mapper on a login node — use `srun`/`sbatch`. Keep
`ECC_MAPPER_THREADS=18` and `--cpus-per-task=18`: the thread count is in the
mapping fingerprint, so any other value is a cold cache. `hpc/HIPERGATOR.md` has
transfer, image build and the cost model.

**A cold map is hours.** Narrow before widening
(`ECC_ARCHS=<one> ECC_MODELS=<one> bash hpc/run_all.sh --map-only`) and watch the
filesystem rather than a pipe — a backgrounded `docker … | grep` buffers until
the pipeline ends and looks hung when it is fine.

**Fan out, don't queue.** `hpc/map_by_shape.sh` submits one job per layer SHAPE
instead of one per model, several architectures at once, and ONE dependent eval
after all of them. `--no-eval` submits maps only; `ECC_LAYERS` maps exactly those
layers. Each submission snapshots the SLURM task file, because `map.sbatch`
resolves its (arch, model) pair from it when the job RUNS. **`--array=I-I` is
required per submission** — without it the extra tasks exit 2 and the dependent
eval sits on `DependencyNeverSatisfied`.

## One sweep, two constants

All three ECC arms (`baseline`, `embedded`, `recon`) are always drawn, so the arms
are never an axis. A run sweeps exactly ONE of the remaining three axes:

| `ECC_SWEEP` | x axis | sweep list | held | stem |
|---|---|---|---|---|
| `bch` | BCH(63,K) | `ECC_SWEEP_KS` | arch, model | `BCHsweep` |
| `model` | networks | `ECC_SWEEP_MODELS` | arch, K | `ModelSweep` |
| `arch` | accelerators | `ECC_SWEEP_ARCHS` | model, K | `ArchitectureSweep` |

Nothing below `config.py` except `sweep.py` knows which axis is swept. A model
sweep is all-CNN or all-transformer — mixing families is a config error.

**`ECC_RECON_MODELING=1`** (env.sh §4) is not a fourth sweep: its axis is WHERE on
the weight path the reconstruction boundary sits. **§4 hard-assigns `ECC_ARCHS`,
`ECC_MODELS`, `ECC_CODE_N` and `ECC_KS` with a bare `=`**, so setting those names
on the command line does nothing and does it silently — use the `ECC_RECON_*`
spellings. `ECC_RECON_ARCHS` is ONE PANEL PER NAME; each panel keeps its own x
axis and its own two reference bars, so a percentage on one panel says nothing
about the other.

## The reduced representation is `datawidth`, not depth

**Capacity dilation (`depth × N/K`) is WITHDRAWN** — it answered negatively and it
was unfair besides (Accelergy priced the deeper array 1.18–1.46× dearer, so the
optimiser had a reason to leave the room unused). FINDINGS §2.4 has the cause: the
capacity mechanism moves in 2× steps and N/K = 1.6154 sits below the step.

The live mechanism is **`datawidth: q` on the weight levels at FIXED `width` and
`depth`**. CACTI receives `depth` and `width` only — `datawidth` never reaches the
energy model — and Timeloop bills `vector_access_energy / block_size` per weight,
`block_size = width/datawidth`. So the reconstruction arm gets more effective
capacity at **byte-identical** per-access read/write/leak.

**`q = round(8·K/N)`.** Two places used to say `ceil`; they agree everywhere except
BCH(63,57) (ceil 8, round 7) and BCH(63,51) (ceil 7, round 6) — and at q=8 the
"recon" arm *is* the embedded arm, so a gate run that way compares embedded with
itself and reports an effect of exactly zero. `hpc/map_depth_sweep.sh` and
`dilation.py` both call `code_widths.declared_datawidth()`. **`recon.py`'s
`Packing` still uses `ceil`** — it is Task 3's physical packing model — so Task 3
narrows those two codes less than the mapping study does.

**HARD CONSTRAINT: `width % datawidth == 0`, or `timeloop-mapper` aborts**
(`buffer.cpp:302`, measured `exit=134`). `block_size` defaults to 1 and is then
checked, so there is **no floor path** — a partially-filled word cannot be
modelled. A width must divide BOTH `q` AND `ECC_WEIGHT_BITS`, since both arms
share one declared width.

**`ECC_WEIGHT_WIDTH=auto` makes every BCH configuration runnable.**
`eccenergy/code_widths.py` holds the width table keyed by `(N, K)` and returns
`lcm(q, ECC_WEIGHT_BITS)`, or `None` where the published widths already admit `q`.
EMPTY still means the published widths, so no existing cache colds — BCH(63,30)
fingerprints identically under EMPTY and `auto`. Read it with
`python3 -m eccenergy.code_widths`; `tests/test_code_widths.py` asserts every entry.

Three assertions, each with a mutation test (`test_dilation.py`):
`archs.assert_pair_geometry()` (both arms declare the same width and depth at every
level, or the comparison is void), `recon.assert_onchip_narrowing_once()` (the
mapper and the evaluator can both narrow, and doing both SQUARES the saving —
hence `ECC_RECON_PACKING=aligned`, the default), and the level table's own
row-level re-check.

**`--levels` always prints the per-DATASPACE table.** A weight-only table once
made a large win look like a null result; FINDINGS §2.8.

## Results layout

    results/_raw/<arch>/<treat>/fp-<hash>/<workload>/<scope>/cls-<mode>/rw-<split>/<model>.json
    results/evaluation/{Pre|Post}/<arch>/<model>/<bch>/<prec>/<scope>/<mapper>/<runid>.json
    results/{figures,tables,manifests}/<stem>.{png,pdf|csv|json}

**The stem comes from the configuration alone**, so re-running at different
constants REWRITES the file instead of adding one; the manifest beside it records
what produced what is on disk. Copy a figure out, or point `ECC_RESULTS_DIR`
elsewhere, to keep it.

`results/evaluation/` accumulates on purpose: one JSON per run, holding every ECC
variant including the ones not evaluated (`total_energy_pJ: null` plus a reason).
Nothing but `results_store.py` writes one. Schema: `docs/RESULTS_SCHEMA.md`.

## The two caches

1. **Mapper cache** — `ecc_energy_study/outputs/<arch>/<treat>/fp-<hash>/<shape>/`,
   one entry per (architecture, layer shape). **Never delete it**; it is hours of
   compute and every solved shape is saved immediately, so runs resume.
   `fp-<hash>` hashes the patched YAML the mapper actually sees plus the globals
   and every mapper setting, and each entry carries a `mapping.json` sidecar.
   **A different fingerprint is a MISS** — before this existed, editing an
   arch.yaml left the path unchanged and old mappings were reported as the new
   design.
2. **Raw energy cache** — `results/_raw/`, parsed from the mapper cache. Pure
   Timeloop output, independent of the ECC configuration, so changing the code
   geometry and redrawing is milliseconds.

Anything that changes what the mapper sees or optimises gets its own cache
subdirectory. **So anything touching env.sh §2/§5, or an arch YAML, takes the
whole matrix cold at once** — budget for it.

**A comparison across two `fp-<hash>` directories is a comparison of two
ARCHITECTURES**, not two settings. `sibling_fingerprints()` enumerates every
solved fingerprint under one variant slug and the table prints a
`!! FINGERPRINT WARNINGS` block. One reported headline was a stale sibling.

## The three ECC arms

One BCH(N,K) codeword over `ECC_WEIGHT_BITS`-bit weights under all three, so they
differ only in where the parity lives. Codewords are counted from DRAM weight reads.

- **baseline** — parity beside the data in DRAM. It pays a **dearer per-bit price,
  not extra traffic**: the decoder is on the DRAM die, so its parity is read,
  corrected and discarded there and never crosses the datapath. What it pays for
  is an array that also holds the parity.
- **embedded** — parity inside the stored weights, laid out as the embedding
  pipeline does it: the MSB-first weight bit stream cut into n-bit codewords, so
  weights straddle codewords and the n−k lowest-significance positions carry parity.
- **recon** — DRAM as embedded, K/N of the weights held on chip, the rest
  regenerated by a synthesized datapath characterised in `data/dc/BCH_N63_results.json`.

`build_stacks()`'s `recon` arm is ONE point applied to every design at once, and no
physical boundary does what it does. It is **not** one of the placement study's
boundaries and must not be quoted as one.

## `recon.py` — the placement space

Two tables define it and **must be edited together**: `WEIGHT_PATHS[<arch>]` (the
stages of that design's weight path, each naming the Timeloop levels that are it)
and `PLACEMENTS[<arch>]` (the boundaries: which stages stay reduced, and what
drives the reconstruction count).

**Designs do not have the same boundaries.** `eyeriss_like_wglb` (Eyeriss v1) and
`simple_weight_stationary` have five; `eyeriss_v2_like` has four.
`eyeriss_v2_like_wglb` **is refused** — its `weight_glb` stage has no boundary
(prompt_6 Appendix B has the fix).

That is enforced, not advised. `validate_placement_space()` requires a placement's
`reduced` set to be a **prefix** of the path's reducible stages in path order, and
every reducible stage to be reached by some boundary. Without it, adding a stage to
one table and forgetting the other keeps every boundary below it reporting its own
saving while the new stage stays at full width — the whole list understated, with
nothing saying so. `weight_path()` also refuses if a level carrying Weights energy
is not claimed by exactly one stage, and `cross_check()` re-derives per-category
weight energy against the `Raw` record first.

**A NETWORK boundary's encoders run once per ARRIVAL, not once per injection** —
the count is `Ingresses × Multicast factor`. Charging the ingress count pairs a
destination-side saving with a source-side cost, which is not a placement.
`ECC_RECON_ENCODER_SITE=source` reproduces the earlier numbers for a diff.

**R4b (a retained-reconstruction register) IS REMOVED.** Consecutive weight reuse
is 1 on 20 of 21 resnet18 layers, so a latch catches nothing and a register that
pays would be a second scratchpad. **Do not reintroduce one without first
re-measuring `consecutive_run` on the target mapping** — that is the step the
original audit skipped (FINDINGS §2.7).

Three things the model refuses to fudge, each with a knob and a recorded check:
**physical packing** (`stream` scales every reduced stage by K/N; `aligned` gives
each weight whole bits), **reconstruction granularity** (`G_rec`, computed from the
layout, not assumed), and **feasibility** (a PE-local boundary whose resident tile
is smaller than `G_rec` is reported `unsupported` with the layers named).

The DRAM term is ONE stage and the whole of it is reducible: only the k message
bits are read out and driven off the die. `ECC_DRAM_PJ_PER_BIT` is the per-bit
cost; `ECC_MAC_PJ_OVERRIDE` rescales the Compute category evaluator-side, after
the raw cache, so no mapping moves. Both carry their citation from
`provenance.yaml` onto every figure and result.

**Every result prints the ceiling first** — the weight energy a boundary could
reduce, times `1 − K/N`. Without it a sub-percent saving reads as a missing term
rather than as arithmetic.

## Architecture rules

`archs/_shared/standard.yaml` states what must be IDENTICAL across designs;
`provenance.yaml` records where every declared number came from; `noc.yaml` holds
the interconnect coefficients and, per design, which spatial containers actually
are the NoC. Two NoC terms Timeloop cannot be given are charged after mapping by
`noc_post.py`. `archs/_shared/components/` is part of the mapper fingerprint:
editing a component colds every cache, on purpose. `bash run.sh validate` checks
all of it and exits non-zero on a violation.

Only one of the four `source:` values licenses using a design's name as the chip:
**`published`**. `derived` differs in one stated block, `reference_design` is
shipped by timeloop-accelergy-exercises and merely *named* after a paper,
`locally_authored` reproduces nothing. `simba_like` is `reference_design` — label
it "Simba-like (reference design)".

**Eyeriss v1 IS `eyeriss_like_wglb`** (decided 2026-09-10). JSSC 2017 publishes the
8 kB filter-weight allocation of the 108 kB GLB, so the file that models it is the
design; `eyeriss_like`, which declares `!Nothing` in its place, is retired — still
mappable by name for a diff, but no longer the design. The v1 bracketing-pair
doctrine is over (`config.BRACKET_PAIRS` is empty); the bracket rule still holds
for v2.

**A constrained/relaxed dataflow is a different accelerator.** `ECC_MAPSPACE_CONSTRAIN`
and `ECC_WEIGHT_FACTOR_RELAX` change the dataflow (v1's `M=1` at the filter spad IS
row-stationary), so such a run must never be quoted as the published chip. Label
bars "constrained dataflow".

Accumulator width is deliberately not standardized (v1 16b, v2 20b, Simba 24b, each
cited). `diagnose` flags a psum level narrower than its own accumulator; a level
that is legitimately narrower declares `# psum-width-ok: <reason>` in the YAML.

## Working on this code

- **Frozen**: `parity.py`, `baseline.py` and `external_parity()` are not
  refactored. `build_stacks()` **is no longer frozen** — prompt_6 RULE 3 changes
  its reconstruction term. After any change there, re-run `bash run.sh baseline
  --eval` and diff: Task 1 and 2 totals must not move.
- **Never** read `os.environ` outside `config.py`, and never resolve a path outside
  `paths.py`.
- **Adding an architecture**: `archs/<name>/arch.yaml` (plus `arch_paper.yaml` with
  each number cited), the name in `KNOWN_ARCHS` and `ARCH_LABELS` in `config.py`,
  entries in `standard.yaml` and `provenance.yaml`, then `bash run.sh validate` and
  `diagnose` before committing to a long sweep.
- **Adding a model**: add to `CNN_MODELS` or `TRANSFORMER_MODELS` in `config.py`,
  then `python3 -m eccenergy.generate models <name>` in the container.
- **Adding an architecture to the placement study**: `WEIGHT_PATHS[<name>]` and
  `PLACEMENTS[<name>]` together, then the name in `ECC_RECON_PLACEMENTS` in env.sh
  §4. Get the stage-to-level match right by reading a real
  `timeloop-mapper.stats.txt` AND `timeloop-mapper.map.txt` from that design's
  cache. `test_every_placement_space_is_valid_for_every_supported_design` makes a
  half-finished pair fail the tests rather than understate a figure.
- **Adding a sweep axis**: a name in `SWEEPS`, a stem in `SWEEP_STEMS`, resolution
  in `Config.__post_init__`, and a `_<name>_groups()` in `experiments/sweep.py`. Do
  not write a second renderer.
- **Changing how a bar looks**: `draw_panel()` in `plots/stacked.py` is the only
  place a bar is drawn, so every figure moves together. Add a parameter to it
  rather than a second routine.
- **Line endings**: the shell scripts run inside a Linux container and a CRLF makes
  bash die on `set -o pipefail` with a mangled message. `.gitattributes` forces LF;
  `bash tools-fix-eol.sh` repairs anything that slips through. **Always emit LF.**

## Layout

Everything not listed here is what its name says; `eccenergy/` module docstrings
carry the rest.

    env.sh              THE knob file        run.sh    one stage, no knobs
    prompt_6.md         THE LIVE PLAN        FINDINGS.md   what was learned
    hpc/                run_all.sh, map.sbatch, tl.sh (apptainer wrapper),
                        map_by_shape.sh, map_capacity_sweep.sh, map_depth_sweep.sh,
                        summary.py, HIPERGATOR.md
    eccenergy/          config.py (the ONLY reader of os.environ), paths.py (the
                        ONLY resolver of paths), workloads.py, timeloop.py (the
                        only module needing the container), energy.py, ecc.py,
                        recon.py, parity.py, embedded.py, code_widths.py,
                        noc_post.py, results_store.py (the ONLY writer of an
                        evaluation JSON), plots/, experiments/, tests/, generate.py
    archs/_shared/      standard.yaml, provenance.yaml, noc.yaml -- not an
                        architecture; skipped by the installer
    ecc_energy_study/   AUTO-MANAGED: cloned repo + mapper cache. Do not delete.
    legacy/             pre-rewrite material, plus the dated FINDINGS_detail_*,
                        progress_* and PROJECT_STATUS_* snapshots that hold the
                        full working record. Nothing imports it.
