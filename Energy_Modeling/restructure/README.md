# `restructure/` — the gate `ProjectRestructure.md` is run against

**This directory is scaffolding for one job: proving that a restructuring phase
moved code and changed nothing else.** It holds no model and no result. When the
phase plan is finished it can be deleted in one line.

    module load apptainer
    bash hpc/tl.sh bash restructure/gate.sh            # check a phase
    bash hpc/tl.sh bash restructure/gate.sh --update   # re-take the golden snapshot
    bash hpc/tl.sh bash restructure/gate.sh --accept-tests "<why>"
    bash hpc/tl.sh bash restructure/gate.sh --accept-record "<why>"

**Brought back on 2026-09-14 for `plans/EnvReorganisation.md`** (phase 0), from
commit `46a8df1`, with one addition: `compare.py` and `--accept-record`, below.
The two migration scripts of the first restructure were spent and stayed deleted;
`gen_guards.py` and `scaffold_arch.py` live in `tools/` now.

| file | what it is |
|---|---|
| `gate.sh` | §9.1's six items as one command. Exit 0 = the phase is done. |
| `snapshot.py` | takes one snapshot: fingerprints, the three arms, validate, diagnose, the suite |
| `_fingerprints.py` | gate item 1 on its own — every `arch_fingerprint()`, as TSV |
| `golden/` | **the committed snapshot.** The safety net. Phase 0 is this directory existing. |
| `compare.py` | run by the gate when items 2-5 are not byte-identical: is it a NUMBER, or the record? |
| `golden/record.log` | every `--accept-record`, with its reason and the classifier's own list |

## The one rule

> **Gate item 1 is not advisory.** Every fingerprint names a mapper cache
> directory holding hours of SLURM. If the fingerprint block diffs, STOP and put
> the architecture back — do not run the mapper, and do not `--update` the golden
> snapshot to make the red go away.

`--update` is for the START of a phase, before any code moves. A golden snapshot
re-taken after a change is not a safety net; it is a record of the change.

## Items 2-5: byte equality first, a classified diff second

Every manifest and every result file carries `Config.to_dict()`, and
`plans/EnvReorganisation.md` phases 1-4 DELETE knobs. A deleted knob is a key that
vanishes from the record of every result while no number moves -- and `diff -ur`
reports that exactly as it would report a moved energy. `compare.py` tells them
apart, and its rules are in its docstring in full:

* every numeric leaf of every JSON, every numeric cell of every CSV and every
  numeric token of every changed stdout line must be identical -- a moved number
  is **always red** and no flag accepts it;
* a record key or a text column that appears or vanishes with **no number under
  it** is listed as RECORD; a changed string, string list or reworded line is
  listed as PROSE. Both still FAIL the gate, until the person running it has read
  the list and stated why: `--accept-record "<why>"` rewrites the affected golden
  artefacts alone (never `fingerprints.tsv`, never the test artefact) and appends
  the reason **and the classifier's list** to `golden/record.log`.

Mutation-tested on 2026-09-14 before the golden was taken: one perturbed
`total_energy_pJ` deep in a result, one changed numeric CSV cell and one changed
number inside a stdout line are each red on their own; three deleted config keys,
one dropped text column and one reworded banner line are each record/prose only.

## Item 6 is the one that is not byte equality

§9.1 asks for the test suite to be **no worse**, not identical — phase 1 fixes
tests and phase 7 adds them, and a suite that grew is not a suite that broke. So
gate 6 refuses a **failure**, refuses a **silent drop** in the number passing,
and accepts growth. A deliberate drop is a recorded override:
`--accept-tests "<why>"` rewrites `golden/pytest.txt` **alone** and appends the
reason to `golden/pytest.log`. Every energy in the golden snapshot stays put.

That branch is mutation-tested too: a golden claiming 999 passing against a fresh
227 exits 1 with `tests were LOST, and nothing says why`. The first version of it
did not — the count regex needed a non-digit before the number and the summary
line begins with one, so both counts parsed as 0 and compared equal. Which is the
argument for testing the gate and not only with it.

## The one recorded prose divergence, 2026-09-13 (phase 3)

Phase 3's gate ran green on the fingerprints and on every number, and diverged on
**one line of one file**: a caveat sentence in the placement result names the
module `WEIGHT_PATHS` lives in, and phase 3 moved it
(`eccenergy/recon.py` -> `eccenergy/arch/weight_path.py`). A provenance sentence
that points at a file which no longer exists is worse than a diff, so the
sentence was corrected and the golden snapshot re-taken **after** phase 3 was
verified, as phase 4's baseline.

The rule is unchanged, and this is the shape an exception has to have: the
divergence was reduced to a single hunk and read before anything was rewritten,
`fingerprints.tsv` was byte-identical, and no energy moved. **A number moving is
never this.**

## What the snapshot covers, and what it cannot

**Covers.** Every `arch_fingerprint()` over `arch x model` and over
`arch x K x mapper arm` (300 rows, plus the two digests that sit inside every one of them); Task 1 and Task 2 totals; the reconstruction
placement study at env.sh's defaults **and** at `ECC_RECON_ERT_AWARE=1`; the
`validate` and `diagnose` reports with their manifests; the whole test suite.

**Cannot.** A number that has no cached mapping cannot be compared, because it
cannot be computed: since Phase C2, `resnet18` on `eyeriss_like_wglb` is the
model whose six arms are mapped, and **every other model is cold on purpose**
(CLAUDE.md, "EVERY MAPPER CACHE IS COLD, ON PURPOSE"). The gate holds the warm
corner still. It is not evidence about `mobilenet_v2`, and the fix for that is
mapping time, never a wider snapshot.

Two stages record a **refusal** rather than a number, and that is deliberate:
`recon_default` (env.sh's Task 4 defaults, whose capacity cache is cold) and six
`REFUSED` rows in `fingerprints.tsv` (`simple_weight_stationary` at `recon5`,
whose placement narrows a `weight_reg` level that design's YAML does not
declare). A refusal is behaviour. If a phase changes one, the gate says so.

## Proven, not asserted

Both halves were measured on 2026-09-13, before the golden snapshot was taken:

* **Reproducible** — two consecutive snapshots of an unchanged tree are
  byte-identical, whole directory, no exclusions.
* **Sensitive** — `filter_glb  depth: 256 -> 512` in
  `archs/eyeriss_like_wglb/arch_paper.yaml` moves **51 fingerprint rows** (that
  design's 15 models and all 36 of its `K x arm` pairs) and takes every
  downstream stage with it.

One measurement worth keeping beside that: `depth: 256 -> 257` moves **nothing**,
and correctly. `_set_weight_geometry()` renormalises the declared depth onto the
arm's width (`depth' = round(depth x width / 96)` — THE WIDTH TABLE), and 256 and
257 both land on 43. The fingerprint hashes the architecture the MAPPER sees, not
the text you typed; a declared change smaller than that quantum is not a
different chip. `validate` and `diagnose` read the declared file and do notice.
