# PROJECT_STATUS

**Last updated:** 2026-09-07 (Task 2 -- embedded ECC, DRAM effect only -- implemented and evaluated)
**Migration:** COMPLETE. Single-process Apptainer flow verified (all five checks
passed; the six copied-cache energies match the laptop exactly), and parallel
execution is built and proven: `sbatch hpc/map.sbatch` ran 12/12 tasks, 0 failed,
258 shape-mappings in 48 minutes on 162 of the 181-core rewetz investment.
**Environment fixes:** hpc/tl.sh binds writable CACTI scratch under /blue;
`eccenergy/archs.py` writes the shared patched-arch YAML and globals.yaml
atomically so array tasks cannot read a partial file (content unchanged —
check 4 still reproduces the laptop totals exactly).
**Results status:** resnet18 (21/21 layers) and mobilenet_v2 (53/53 layers, 10
depthwise) are now evaluated WHOLE-MODEL on all six architectures at the
converged search, and the panel figure is regenerated and quotable. The
architecture ranking is preserved across both models. **The two development
layers are not a proxy for the model** — weight-stationary and input-stationary
swap between them.
**Model status:** Tasks 1 and 2 are complete (Task 2 section next; FINDINGS.md
§14). The `embedded` arm is now a measured DRAM-only result on the Task 1
mappings; `recon` is still the placeholder. Prior caveats still apply — the
`_wglb` bracketing pair was not run and six of the eight models are still
unmapped. NoC energy has been costed since 2026-09-07 (FINDINGS.md
"Interconnect"). run.sh defaults, architectures and the baseline are unchanged.
See FINDINGS.md §12 (bring-up) and §13 (the whole-model run) for evidence.

---

## 2026-09-07 TASK 2 — EMBEDDED ECC, DRAM ENERGY EFFECT ONLY — DONE

**What was asked** (`03_staged_implementation_plan.txt` §2): an embedded-ECC
evaluation mode with the parity inside the stored weights and no external
parity, using the *actual* embedded-codeword layout, reading the complete
codeword for correction, with mappings / on-chip widths / access counts / MAC
behaviour fixed, changing only the modelled DRAM storage/traffic energy, and
distinguishing stored bits from physical reads.

**Two questions answered first, then the work:**

1. *Re-run the mapping optimiser?* **No.** The mapper never sees parity: Task 1
   maps 8-bit weights against the architecture and adds the external parity in
   evaluation, after Timeloop. The embedded arm changes nothing the mapper
   sees either (DRAM and on-chip weights are 8 bits wide under both arms), so
   the fingerprint is unchanged, the cached mappings are reused, and the plan
   itself requires the fixed mapping for this comparison. Every Task 2 file
   records `evaluation_only_rerun: PASS` (0 newly mapped shapes) and
   `fixed_mapping_shared_across_variants: PASS`.
2. *Is external parity already in the baseline DRAM?* **Yes**, since Task 1:
   `parity.py` + `ecc.external_parity()`, 6 whole weights per BCH(63,51)
   codeword, 3 message-padding bits, tail padding, ×measured refetch, rounded
   to 64-bit DRAM words — 31.25% over payload traffic, i.e. 10.5 stored bits
   per weight rather than the flat n/k (8·63/51 = 9.88). Nothing was
   re-implemented there.

**The actual embedded layout** was read from the embedding code
(`ECC-CODE-Engine/4-EmbeddingECC/ecc_embed.py`, copied verbatim into
`../Input_Embedding/3-Testing/{utils,implementations}`): the int8 tensor is
one MSB-first bit stream cut into fixed 63-bit chunks (`chunk_size =
message_parity_size`), weights straddle chunk boundaries (7.875 weights per
codeword for every K; gcd(63,8)=1 so the alignment repeats every 8 codewords =
63 weights and 7 of 8 boundaries split a weight), only the final chunk is
zero-padded, and `ParityOverwriteByTopWeightsEncode` overwrites the n−k
lowest-significance chunk positions with the parity. At BCH(63,51) every
weight loses its LSB and about half lose bit 1 as well (1.524 parity bits per
weight). Storage is therefore exactly 8 bits per weight plus one tail pad per
tensor, and there is no external parity. This is the opposite packing rule to
the baseline's whole-weight codewords, and both are correct for their layout.

**What the embedded arm's DRAM energy is:** Timeloop's DRAM weight energy,
unchanged. The complete codeword is read (every one of the 8 bits, message and
parity), so no K/N reduction is applied at DRAM; what disappears relative to
Task 1 is the external-parity traffic and nothing else. The saving is
therefore `parity / (Timeloop + parity)` on each architecture.

### Files changed

| File | Change |
|---|---|
| `eccenergy/embedded.py` | **new.** `EmbeddedLayout` (bit-stream layout, phases, parity-significance profile), `account()` / `account_layers()` (stored bits, tail padding, straddles, external parity = 0, DRAM words), `traffic_account()` (complete-codeword reads, physical-word estimate), `dram_energy_pj()`, `hand_check()` / `hand_check_layers()` (codeword-by-codeword walk vs closed form). `LAYOUT_SOURCE` cites the files the layout was read from. |
| `eccenergy/ecc.py` | `embedded_dram()` added beside `external_parity()`; module docstring's embedded paragraph rewritten. `build_stacks()`, `external_parity()`, `savings()` **unchanged** (verified by diff). |
| `eccenergy/experiments/embedded.py` | **new.** Task 2's experiment `task2_embedded_ecc_dram_only`: baseline + embedded evaluated in ONE file from the same raw record, recon placements unavailable, six Task 2 checks, report. |
| `eccenergy/experiments/audit.py` | **new.** Task 1's checks / caveats / detail as functions, mirrored from `baseline.py` so `baseline.py` did not have to be touched. Keep the two in step. |
| `eccenergy/config.py` | `"embedded"` in `EXPERIMENTS`. No new knob: the layout is fixed by the pipeline, and the per-tensor grouping reuses `ECC_PARITY_GROUPING`. |
| `eccenergy/__main__.py` | runner + help text. |
| `run.sh` | documentation of the experiment and the quick-run command (no default changed). |
| `hpc/run_all.sh --eval-only` | writes `embedded --eval` beside `baseline --eval` per model (`ECC_EVAL_EXPERIMENTS`). Was `hpc/eval_panel.sh` until the env.sh consolidation. |
| `hpc/summary.py` | `--field embedded` and `--field saving`; two CSV columns. |
| `eccenergy/tests/test_embedded.py` | **new.** 11 offline tests (2 need pandas → container). |
| `CLAUDE.md`, `README.md`, `docs/RESULTS_SCHEMA.md`, `hpc/HIPERGATOR.md`, `FINDINGS.md` §14, `progress.txt` | documentation. |

**Not touched:** `eccenergy/parity.py`, `eccenergy/experiments/baseline.py`,
`energy.py`, `results_store.py`, `paths.py`, every `archs/` YAML, every
`run.sh` default, the recon arm.

### Exact run commands (HiPerGator; `--eval` never invokes Timeloop)

```bash
module load apptainer
export ECC_MAPPER_ALGORITHM=random_pruned ECC_MAPPER_SEARCH_SIZE= ECC_VICTORY=2000 \
       ECC_MAPPER_TIMEOUT=2000 ECC_MAPPER_THREADS=18 ECC_SWEEP=arch \
       ECC_SWEEP_ARCHS="eyeriss_v2_like eyeriss_like simple_weight_stationary simple_output_stationary simple_input_stationary simba_like"
# whole model, both panel models
ECC_LAYERS= ECC_CONST_MODEL=resnet18     bash hpc/tl.sh bash run.sh embedded --eval
ECC_LAYERS= ECC_CONST_MODEL=mobilenet_v2 bash hpc/tl.sh bash run.sh embedded --eval
# the development pair (quick run)
ECC_LAYERS="layer3.0.downsample.0 layer4.1.conv2" ECC_CONST_MODEL=resnet18 \
  bash hpc/tl.sh bash run.sh embedded --eval
# the matrix, and the tests
python3 hpc/summary.py --scope layers-full --field saving
bash hpc/tl.sh python3 -m eccenergy.tests.test_embedded
```

### Results (BCH(63,51), 8-bit weights, NoC costed, converged search)

Whole model — conventional ECC (Task 1) vs embedded ECC (Task 2), µJ:

| architecture | resnet18 conv. | embedded | saving | mobilenet_v2 conv. | embedded | saving |
|---|---:|---:|---:|---:|---:|---:|
| eyeriss_v2_like | 5,075.371 | 4,798.375 | 5.46% | 1,862.971 | 1,792.380 | 3.79% |
| eyeriss_like | 5,357.602 | 5,058.978 | 5.57% | 1,984.785 | 1,903.093 | 4.12% |
| simba_like | 5,699.198 | 5,401.230 | 5.23% | 1,996.279 | 1,924.891 | 3.58% |
| simple_input_stationary | 6,647.611 | 6,376.039 | 4.09% | 2,102.472 | 2,031.349 | 3.38% |
| simple_weight_stationary | 7,782.205 | 7,532.661 | 3.21% | 2,348.220 | 2,268.861 | 3.38% |
| simple_output_stationary | 9,363.620 | 9,103.499 | 2.78% | 2,639.039 | 2,569.644 | 2.63% |

resnet18 DRAM footprint, both arms: 11,678,912 weights; conventional
15,328,593 B stored (1,946,488 codewords, 10.5 b/weight), embedded
11,679,027 B (1,483,051 codewords, 8.0001 b/weight, 115 B of tail padding
over 21 tensors); DRAM weight reads identical (13.85 M scalars on v2,
refetch ×1.19). Development pair (2 layers, resnet18): savings 11.0% (v2),
10.9% (v1), 9.7% (Simba-like), 8.8% (WS), 8.5% (IS), 6.3% (OS) — a
development check, not a ranking.

**Baseline regression:** `bash run.sh baseline --eval` re-run after the change
on all 12 whole-model (architecture, model) pairs reproduces the pre-change
files exactly — total, per-component energies, parity accounting, validation
outcomes, mapping ids and summary all identical.

**Checks on every Task 2 file:** the six Task 2 checks and Task 1's checks
all pass; the only failing entry is the pre-existing
`noc_share_within_published_band_eyeriss_v2` (5.3% vs the 6–10% band), which
Task 1's files fail identically. Tests: 11/11 `test_embedded` (in the
container), 15/15 `test_results_store`, 12/12 `test_noc`, 4/4
`test_mapper_lock`.

### Assumptions and approximations, recorded in every result

* **Bit-proportional DRAM.** Timeloop bills 1/8 of a 64-bit word per 8-bit
  scalar; physical DRAM words are *estimated* as bits/64 rounded up once, and
  the final chunk's tail padding is reported but not charged — the same rule
  the baseline's payload is billed by. A burst-exact count is not available.
* **Per-tensor grouping** (`ECC_PARITY_GROUPING=layer`, shared with the
  baseline): each layer is its own bit stream and pays its own tail pad, as the
  embedding driver encodes one tensor per call.
* **ECC codec energy is outside the comparison** — charged to neither arm. The
  arms would decode different codeword counts (1.95 M vs 1.48 M on resnet18),
  so a codec cost would not cancel; it is simply not costed until a
  characterised codec energy is adopted. `ECC_DECODE=1` affects the sweep
  figures only and is warned about.
* No on-chip saving is credited to the embedded arm and no inference-accuracy
  claim is made about overwriting weight LSBs.
* `build_stacks()`'s embedded bar was already `raw.total` and is unchanged;
  `embedded_matches_sweep_figure_arm` checks the JSON against it on every
  file. Its decode *count* for the embedded arm still uses the baseline's
  6 weights/codeword (`ECC_EMB_WEIGHTS_PER_CW`); that is a codec-energy
  question and was deliberately not changed here.

### Next: Task 3 — Eyeriss v2 reconstruction, evaluator only, fixed mapping

Not started. Uses the Task 2 file as the embedded-ECC reference. The recon
arm in `build_stacks()` is still the placeholder the user will correct first;
`experiments/audit.py` is the shared check set Task 3 should reuse, and
`EmbeddedLayout.parity_significance_profile()` already says which weight bits
are dependent per codeword phase, which Task 3's reconstruction counts need.

---

## 2026-09-06 AUDIT ADDENDUM — read this before anything below

The full write-up is `FINDINGS.md`; the status board is `progress.txt`.

1. **Depthwise/grouped convolutions were modelled wrongly and are now fixed.**
   The loader dropped `groups`, so a MobileNetV2 depthwise layer reached
   Timeloop as a 1-input-channel convolution (81 input scalars instead of
   77,760). `Layer.G` + a grouped problem shape fix it; MAC counts now match
   published values for all eight CNNs. Every model with depthwise layers
   (mobilenet_v2, efficientnet_b0, convnext_tiny, xception) needs its
   depthwise shapes re-mapped. ConvNeXt's per-pixel `nn.Linear` layers had a
   second, similar defect (P=Q=1); also fixed, `convnext_tiny` regenerated.
2. **The mapper search decides the ranking.** On the two Task 1 development
   layers, `hybrid`/victory 100 (the Task 1 setting, and `hybrid` is still the
   `run.sh` default) gives Eyeriss v1/v2 1979/2037 µJ; `random_pruned` with
   `search_size 20000`/victory 2000 gives 390/402 µJ on the same layers and
   the ordering inverts (v1 ≈ v2 < Simba-like < WS < IS < OS). The quick-run
   table further down this file is the first setting and must not be quoted.
   **Decision needed:** change the default search in `run.sh`.
3. **NoC energy is now costed** (was zero everywhere: Timeloop's wire model is
   a stub). `archs/_shared/noc.yaml` supplies one shared 45nm wire constant and
   per-architecture switching terms, injected onto the spatial containers so
   the Legacy networks carry energy inside the mapper's objective. It is a
   plotted category of its own; the mapper cache slug gains `noc`, so every
   architecture needs a fresh cold run. FINDINGS.md "Interconnect (NoC) model".
4. **Cache-key drift:** the fingerprint recipe changed after Task 1
   (`search_size`, `max_permutations` added), so the Task 1 evaluation JSONs
   under `results/evaluation/.../map-energy-vic100l-hybrid-seednone/` can no
   longer be re-evaluated from their mapper cache, and new evaluations land
   under `...-seednone-ssnone-perm16/`. Documented, not repaired.
5. **Migration decided:** local mapping is too slow (~12 h for the two
   full models), so the project moves to HiPerGator
   (`/blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling`). Instructions are
   in `hpc/HIPERGATOR.md`; the order is transfer → verify the single-process
   flow → then design parallel execution there. The local chained run was
   stopped at 15:15.
6. Files changed this session: `eccenergy/workloads.py`, `timeloop.py`,
   `energy.py`, `generate.py`, `tests/test_results_store.py` (stale slug),
   `ecc_energy_study/model_layers.json` (convnext_tiny only), plus
   `FINDINGS.md`, `progress.txt`, this header and structural notes in
   `CLAUDE.md`. No arch YAML, no `run.sh` default, no ECC arm was changed.

Everything below this line is the Task 1 session's own report, kept verbatim
for the record; where it quotes energies they are the superseded numbers.

---

---

## What Task 1 was asked for, and where it stands

| Requirement | State |
|---|---|
| Organise the architectures consistently, shared settings + per-design config | done — `archs/_shared/standard.yaml` + `provenance.yaml`, checked by `bash run.sh validate` |
| Same weight/activation precision, workload, BCH config across comparisons | done — enforced, not asserted; all 7 designs pass with 0 violations |
| Keep the precisions explicit and distinct | done — banner, validator and every result JSON report all three separately |
| Preserve hierarchy, capacity, weight bypass rules | done — printed by `validate` as "preserved differences"; nothing equalises them |
| Conventional-ECC baseline only, parity external in DRAM | done — `eccenergy/experiments/baseline.py` |
| Account for grouping, padding, physical DRAM transfer granularity | done — `eccenergy/parity.py` |
| External parity NOT stored in on-chip weight SRAM/RF | done — the baseline does not touch on-chip energy; the old switch is off and warns |
| Each architecture completes the selected-layer quick run | done — 6/6, both layers, 0 failures |
| Configuration summary reports precisions and BCH parameters | done — see the banner output below |
| Hand-check confirms payload/parity accounting | done — mechanised, runs on every result, recorded in the JSON |
| Existing baseline preserved apart from documented standardization | see **Deliberate changes to existing behaviour** below |

### One deliberate correction to the task text

Task 1 says to standardize accumulation precision. **You directed otherwise on
2026-09-06**, and that direction is what is implemented: weight precision,
activation precision, workload, BCH configuration and methodology are
standardized; **each design keeps its published accumulator width**, because
psum precision is an architectural property and equalising it would equalise
the architectures. All ECC arms of one design share its width. A forced common
width is available as an explicitly-labelled sensitivity study (`ECC_ACC_BITS`)
with its own mapper cache and results namespace, and is never the primary
result.

You also asked that the Simba 24-bit value not be trusted just because the YAML
said so. It is now traced, along with every other accumulator width — see
**Provenance** below.

---

## Files changed

### Architectures

| File | Change |
|---|---|
| `archs/_shared/standard.yaml` | **new.** The comparison contract: what must be identical across designs, and each design's own accumulator width with its citation. |
| `archs/_shared/provenance.yaml` | **new.** Level by level, where each declared number came from, with the quote and the date it was verified. |
| `archs/simple_input_stationary/arch_paper.yaml` | **new.** Authored as the exact mirror of the WS/OS siblings. |
| `archs/simple_input_stationary/README.md` | **new.** |
| `archs/eyeriss_v2_like/arch_paper.yaml` | corrected — see below. |
| `archs/eyeriss_v2_like_wglb/arch_paper.yaml` | same corrections mirrored. |

### Package

| File | Change |
|---|---|
| `eccenergy/parity.py` | **new.** External BCH parity accounting + the hand check. |
| `eccenergy/results_store.py` | **new.** The centralized result writer. |
| `eccenergy/experiments/baseline.py` | **new.** Task 1's experiment. |
| `eccenergy/experiments/validate.py` | **new.** The contract check. |
| `eccenergy/tests/test_results_store.py` | **new.** 15 offline tests. |
| `eccenergy/config.py` | layer selection, `Pre`/`Post`, precision knobs, parity knobs, cache strictness, `simple_input_stationary`, precisions in the banner. |
| `eccenergy/paths.py` | the evaluation namespace; fingerprinted raw and mapper caches. |
| `eccenergy/archs.py` | contract loaders, `arch_fingerprint()`, `validate_arch()`, the accumulator-override patcher, **globals node fix**. |
| `eccenergy/workloads.py` | `select_layers()` by stable name, `layer_identities()`. |
| `eccenergy/timeloop.py` | mapping sidecars, `mapping_id`, tool versions, content-verified cache reuse, post-mapping stats-path bug fix. |
| `eccenergy/energy.py` | per-layer and per-level detail in `Raw`; fingerprint threaded through the cache. |
| `eccenergy/ecc.py` | baseline uses `external_parity()`; one shared codeword geometry. |
| `eccenergy/experiments/common.py` | layer selection, fingerprints, mapper records. |
| `eccenergy/experiments/diagnose.py` | reads the fingerprinted raw cache. |
| `eccenergy/__main__.py` | the three new experiments, `--layers/--eval/--overwrite/--phase`. |
| `run.sh` | development-mode knobs, quick-run commands, `simple_input_stationary` in the default sweep. |
| `docs/RESULTS_SCHEMA.md` | **new.** |
| `CLAUDE.md` | structure updated: new commands, new layout, the contract, the fingerprint. |

---

## Exact run commands

```bash
DEV_LAYERS="layer3.0.downsample.0 layer4.1.conv2"
```

**1. Validate the architectures** (no container, no cache, seconds):

```bash
bash run.sh validate
```

**2. Generate the mappings** (container; ~3.5 min for 6 archs x 2 layers):

```bash
docker run --rm -v "<project>":/home/workspace -w /home/workspace \
  -e ECC_LAYERS="$DEV_LAYERS" -e ECC_VICTORY=100 -e ECC_MAPPER_THREADS=8 \
  -e ECC_SWEEP=arch \
  timeloopaccelergy/timeloop-accelergy-pytorch:latest-amd64 \
  bash -lc "cd /home/workspace && bash run.sh map"
```

**3. Evaluate** (no container, seconds):

```bash
ECC_LAYERS="$DEV_LAYERS" ECC_VICTORY=100 ECC_MAPPER_THREADS=8 \
ECC_SWEEP=arch bash run.sh baseline --eval
```

**4. Tests** (no container):

```bash
python3 -m eccenergy.tests.test_results_store
```

---

## Results

### Contract validation — 7/7 architectures, 0 violations

```
  architecture                acc b   fanout   on-chip weights  GLB keeps W
  eyeriss_like                   16      168            75,264        False
  eyeriss_v2_like                20      384            55,296        False
  eyeriss_v2_like_wglb           20      384           120,832         True
  simple_weight_stationary       16      256           164,096         True
  simple_output_stationary       16      256            65,792         True
  simple_input_stationary        16      256            65,792         True
  simba_like                     24      256         2,113,536        False
```

Run against `ECC_ARCH_FIDELITY=stock` the same validator reports the defects the
paper YAMLs exist to fix — `simple_weight_stationary` 5 violations (datawidth 16
on DRAM and every operand level), `simba_like` 2 (a shared psum/operand buffer,
and `adder_width: 16` against a 24-bit accumulation buffer), `eyeriss_like` 1
(the shared 8-bit GLB holding 16-bit partial sums). That is the mechanism
working, on exactly the class of defect that prompted this work.

### Selected-layer quick run — 6/6 architectures, both layers, 0 failures

resnet18, `layer3.0.downsample.0` + `layer4.1.conv2`, 2,392,064 weights,
BCH(63,51) t=2, 8-bit weights and activations.

| architecture | acc | Timeloop µJ | + parity µJ | conventional ECC µJ | DRAM refetch | pJ/DRAM weight |
|---|---|---|---|---|---|---|
| eyeriss_v2_like | 20b | 2,036.79 | 339.48 | 2,376.27 | ×7.10 | 64.0000 |
| eyeriss_like | 16b | 1,979.32 | 331.61 | 2,310.93 | ×6.93 | 64.0000 |
| simple_weight_stationary | 16b | 731.96 | 47.84 | 779.80 | ×1.00 | 64.0000 |
| simple_output_stationary | 16b | 904.31 | 47.84 | 952.15 | ×1.00 | 64.0000 |
| simple_input_stationary | 16b | 581.62 | 47.84 | 629.46 | ×1.00 | 64.0000 |
| simba_like | 24b | 3,166.24 | 51.77 | 3,218.01 | ×1.08 | 64.0000 |

**`pJ/DRAM weight` is identical across all six.** That column is the
apples-to-apples check: at the same node and DRAM datawidth it must be, and the
original complaint was that it was not. It is now checked on every run.

**These are two-layer development numbers. They are not an architecture
ranking**, and every result JSON says so in `warnings`. Two layers out of
twenty-one, chosen for speed and contrast, cannot rank six accelerators.

### Payload/parity hand check

Identical for every architecture, because it depends only on the workload and
the code:

```
payload    2,392,064 weights = 2,392,064 B
codewords  398,678            (6 whole weights per 51-bit message field)
parity       598,017 B
padding      149,508 B        (3 b/codeword message + 32 b tail)
overhead      31.25% over payload
```

A flat `n/k - 1` model charges **23.53%**. The difference is the 3 message bits
per codeword that no whole 8-bit weight can use, plus tail padding. The check
recomputes all of this codeword by codeword and compares against the closed
form; it passes for every architecture and is recorded inside each result JSON
under `validation`, so it is auditable after the fact.

The one-layer run shows **31.27%** — tail padding is a larger share of a
32,768-weight tensor. That is the granularity accounting doing its job.

### Result files

Twelve JSONs under `results/evaluation/Pre/…`, e.g.

```
results/evaluation/Pre/eyeriss_v2_like/resnet18/bch63_51/w8a8/
  layers2__layer3_0_downsample_0__layer4_1_conv2/
  map-energy-vic100l-hybrid-seednone/20260906T072817Z__303cee78.json
```

Each holds all seven variants: the conventional baseline `evaluated`, and
`embedded_ecc` plus the five reconstruction placements as `not_implemented`
with `total_energy_pJ: null` and a reason. Nothing is fabricated.

### Tests

15/15 pass, offline.

---

## Provenance

Verified against the primary sources on 2026-09-06 and recorded in
`archs/_shared/`:

| design | accumulator | source |
|---|---|---|
| Eyeriss v1 | 16b | JSSC 2017 §IV: *"a 16-b two-stage pipelined multiplier and adder … truncated from 32 to 16 b"* |
| Eyeriss v2 | 20b | JETCAS 2019 Table IV: *"psums: 20b fixed-point"*, repeated in Table V |
| Simba | 24b | *"4 tera-ops per second (TOPS) using 8-bit weights and activations and 24-bit accumulation"* |
| WS / OS / IS | 16b | **not published.** Reference dataflows, no paper; recorded as `locally_authored` |

Also verified and recorded: Eyeriss v1's 108 kB GLB in 25 banks of 512-b × 64-b,
the 8 kB filter allocation (and why it is *not* modelled as a reuse level), the
224×16b filter spad and 12×16b / 24×16b ifmap and psum spads; Eyeriss v2's
Table IV scratchpad sizes and Fig. 17 arrays.

**Not verified:** Simba's buffer and PE geometries below the MAC come from
`timeloop-accelergy-exercises`, not from a table checked against the paper.
Only its precisions are traced. Recorded as `not_verified` in
`archs/_shared/provenance.yaml` rather than presented as a citation.

---

## Deliberate changes to existing behaviour

Task 1 says the existing baseline is preserved "apart from documented
standardization and parity accounting". These are the changes, and each is a
correction rather than a preference. **All of them move numbers.**

1. **Parity accounting** — from a flat `n/k - 1` (23.53%) to accounted grouping,
   padding and DRAM granularity (31.25%). This is the parity accounting Task 1
   asks for.

2. **DRAM was costed at 65 nm while every accelerator was at 45 nm.**
   `globals.yaml` defaulted to `65nm`; the containers all declare `45nm`. DRAM
   is where most of the energy and all of the parity cost lands. It was uniform
   across architectures, so it never showed as an ordering error — it made every
   absolute number and every savings percentage wrong by a fixed factor. The
   node now comes from the same contract file the architectures are validated
   against.

3. **Eyeriss v2's iact spad: 16 → 24 entries.** Table IV says *"iact data: 24B
   (Reg)"*. The file was reading it CSC-natively (one value per 12b word, 16
   iacts) while reading its own weight spad dense-repacked (288 B → 288
   weights). One design cannot use two packing rules — that is the same class
   of defect as comparing two designs at different datawidths. The dense rule
   now applies to every level of every design.

4. **Eyeriss v2's weight-spad citation was wrong even though the number was
   right.** It cited §III-C, *"each port has a bitwidth of 24 bits such that it
   can send and receive three 8b uncompressed iact values"* — that sentence is
   about a **router port**, not the spad, and §III-D says the spad stores 12b
   count-data pairs. The capacity (288 weights/PE) now comes from Table IV's
   288 B plus the study-wide dense rule. The number is unchanged; the
   justification is now one that survives reading the paper.

5. **Codeword counting used `k / weight_bits` (6.375) instead of
   `k // weight_bits` (6).** A weight cannot straddle a codeword. This made the
   codeword count ~6% low wherever it was used.

6. **The mapper cache is now content-addressed.** It was keyed on a treatment
   slug, so editing an `arch.yaml` left the path unchanged and the next run
   reused mappings computed for the previous geometry. Entries now carry a
   `mapping.json` sidecar with an architecture fingerprint; a mismatch is a
   miss.

### The cost of change 6, and how to avoid paying it twice

Pre-Task-1 cache entries have no sidecar, so **nothing can prove they match the
current YAMLs** — and for `eyeriss_v2_like` they demonstrably do not, because
its iact spad changed. They are **left on disk** (43 shapes × 5 architectures,
hours of compute) and are simply not used by default.

A full-model run therefore needs a fresh mapper cache. It is resumable and
nothing already computed is lost. `ECC_CACHE_STRICT=0` will accept the old
entries; results built that way are labelled `legacy` and warned about, and
should not be published.

---

## Known limits of what Task 1 produced

Everything here is recorded in the result JSONs under `approximations` and
`warnings`; it is repeated here so a reviewer does not have to open one.

* **Two layers are not a ranking.** 2 of 21 resnet18 layers, ~20% of the
  weights. They validate the implementation.
* **Parity traffic scaling is approximate.** Timeloop reports scalar reads, not
  the tile boundaries they fall on, so the exact stored codeword count is scaled
  by the measured DRAM refetch factor and rounded to whole DRAM words **once**,
  at the end, rather than per tile. This understates granularity rounding when
  tiles are small relative to a codeword.
* **Parity is billed at the measured per-access cost of a DRAM weight read on
  that architecture**, not at an independently modelled parity-region access
  cost. Exact if parity is read from the same DRAM by the same controller, which
  is the conventional-ECC assumption.
* **Timeloop v4 exposes no random seed.** `ECC_MAPPER_SEED` is recorded but
  cannot make the search deterministic. What bounds it is `ECC_VICTORY` and
  `ECC_MAPPER_THREADS`, both in the fingerprint — so pin the thread count.
  Leaving it empty means "all cores", which differs per machine.
* **Eyeriss v2's 12-bit register words cannot be expressed** at datawidth 8
  (Timeloop asserts `width % datawidth == 0`). The iact spad is modelled as
  24×8b: same bits, same operands, narrower words, so its per-access register
  energy is for a narrower word than the design has. The only such case in the
  study.
* **The 288-vs-192 weights/PE reading of Eyeriss v2's weight spad** is the
  largest single modelling judgement here — a third of v2's on-chip weight
  capacity, which directly drives its DRAM refetch. The dense rule gives 288;
  a CSC-native reading gives 192. Recorded in
  `archs/_shared/provenance.yaml` under `contested` so it can be disagreed with.
* **Simba is a 16 nm design modelled at 45 nm**, the largest node deviation.
  Not corrected: correcting one design would reintroduce the mixed-node
  confound.
* **`legacy/figures/*_summary_uJ.csv` will not reproduce.** Changes 1, 2, 3 and
  5 above all move numbers deliberately. As `CLAUDE.md` says, decide whether a
  mismatch is intended before treating it as a regression — here it is intended.

---

## Next (as written at the end of Task 1): Task 2 — embedded ECC, DRAM energy effect only

**Done on 2026-09-07 — see the Task 2 section at the top of this file.** What
it needed, as listed then:

1. The **actual embedded-codeword layout**, not "all 8 weight bits are
   independent payload". `parity.py` has the geometry; the embedded arm needs
   its own layout description.
2. The complete embedded codeword must be **read** for correction — its parity
   is not discarded before the correction.
3. Mappings, on-chip widths, access counts and MAC behaviour held **fixed**
   against Task 1. Evaluation-only reruns already do this: `bash run.sh baseline
   --eval` never invokes the mapper, and the `fixed_mapping_shared_across_variants`
   check in every result will fail loudly if a variant is mapped differently.
4. Report DRAM payload, parity/padding, physical accesses and energy; every
   non-DRAM component must match Task 1 exactly.
5. Document any bit-proportional approximation. Fewer stored bits do not
   guarantee fewer DRAM bursts.
6. Do not change ECC codec energy, or keep it explicitly outside the comparison.

The `embedded_ecc` variant already exists in every Task 1 result file as
`not_implemented`; Task 2 fills it in. The document shape does not change.
