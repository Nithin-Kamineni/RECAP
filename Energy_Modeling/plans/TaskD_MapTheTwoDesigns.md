# Task D — the two new designs, mapped and drawn. ONE DESIGN PER SESSION.

**This is `plans/EnvReorganisation.md` phase 6's TASK D**, left over when that
phase was committed. The plan itself is delivered (phases 0–7, phase 7 at
`00bdd15`); this is the one piece of phase 6 that is queue time rather than
work, and it is unblocked.

`progress.txt`'s ENVREORGANISATION PHASE 7 entry is the state this starts from.
Nothing here is read programmatically, like everything else in `plans/`.

**THE SPLIT, and it is deliberate:**

| session | design | boundaries | chips | units at `ECC_SCOPE=dev` |
|---|---|---|---|---|
| **1** | `simple_weight_stationary` | recon1–4 | 5 | 10 |
| **2** | `eyeriss_v2_like_wglb` | recon1–5 | 6 | 12 |
| **3** | all three, one figure | — | 17 | 34 |

**Do only your own session's design.** Session 1 must not start session 2's
design even though both are described here: they are separate chips, separate
caches and separate progress.txt entries, and a session that does both makes
the record ambiguous about which change produced which number.

---

## 0. READ FIRST — the golden snapshot is STALE BY DECISION

**Do not be surprised by a red gate on your first run, and do not stop.**

`env.sh` was committed at `ee4e6e3` carrying a working ablation configuration.
Three of its values reach the gate:

* `ECC_MAPPER_THREADS=16` (was 18). **The thread count is hashed into
  `arch_fingerprint()`** via `cfg.mapper_settings()` — `arch/fingerprint.py`
  hashes the whole mapper dict — so **essentially every architecture
  fingerprint differs from the golden**, which was taken at 18. Measured on
  2026-09-14: 300 of the 306 lines in `fingerprints.tsv`, the rest being
  headers. **MEASURE IT YOURSELF rather than trusting that number** — env.sh is
  edited between sessions, and `_config_hash` moves on ANY env.sh change, so
  the count drifts:

      bash hpc/tl.sh bash -c 'source env.sh; PYTHONPATH=. python3 \
        restructure/_fingerprints.py' > /tmp/fp.tsv
      diff restructure/golden/fingerprints.tsv /tmp/fp.tsv | grep -c '^<'

  Record the number you actually measured in progress.txt. What matters is not
  the count but that the cause is the thread count and the env.sh edits, and
  that no ENERGY number moved.
* `ECC_APPROACHES="baseline embedded recon"` (was all seven) and
  `ECC_METRICS="energy edp latency"` (was `energy`) change what the
  `recon_default` and `recon_ert` stages produce, so items 2–5 move too.

**THIS IS THE COMMIT'S DOING AND NOT A MODEL CHANGE.** No energy number moved;
the configuration under which they are computed did.

**So the FIRST thing this session does, before any change of its own:**

    bash hpc/tl.sh bash restructure/gate.sh --update

`--update` is for the START of a phase, before any code moves
(`restructure/README.md`) — which is exactly where you are. Record in
progress.txt that the golden was re-taken, that it was re-taken because of
`ee4e6e3`, and the 300/306 figure. A re-baseline that is not written down is
indistinguishable from hiding a red gate.

**AFTER the re-take, the ordinary rule is back and it is absolute:** item 1
must be byte-identical for the rest of your session. Task D declares nothing
new — it runs the mapper and evaluates — so **no fingerprint should move
again**. If one does, STOP and find out why; do not `--update` a second time.

---

## 1. SESSION 1 — `simple_weight_stationary`

### 1.1 It is not broken. Do not "correct" it.

Phase 6 already did this design's correction work and the results are
committed. **Do not undo any of it:**

* `recon5` was **WITHDRAWN**, deliberately. Its `reduced` set ended at
  `weight_reg`, a depth-1 8-bit latch, and rebuilding one weight needs the
  retained bits of `G_rec = 9` co-resident weights. Two parts of the project
  had independently concluded it was impossible. The design declares **FOUR**
  boundaries, recon1–4, and that is correct.
* `weight_reg` is a stage with **`reducible: false`**, on purpose. Deleting the
  stage would leave that level's weight energy unclaimed and fail
  `cross_check()`. Non-reducible is the honest statement about the silicon.
* The free-set (`mapspace_free_levels`) is declared and gated.
* Widths are verified legal at every `q`; `weight_reg` stays 8/8 on every arm.

`validate_placement_space()` returns ok=True for this design. **The job is to
MAP it at scope and DRAW it — not to look for defects that were already
fixed.** If something genuinely surfaces, report it; do not repair a
declaration without saying why.

### 1.2 The command — and you MUST pin the sweep

`env.sh` now ships `ECC_SWEEP=bch` and a three-bar `ECC_APPROACHES`. **Task D
is the PLACEMENT study**, which is `ECC_SWEEP=fix` with every boundary named.
Running the bare launcher would silently produce a code sweep with two arms
instead — the wrong study, drawn to the wrong file.

    ECC_SCOPE=dev ECC_SWEEP=fix \
      ECC_APPROACHES="baseline embedded recon1 recon2 recon3 recon4 recon5" \
      ECC_ARCHS=simple_weight_stationary \
      ECC_CONST_ARCH=simple_weight_stationary \
      bash hpc/run_all.sh --dry-run

`--dry-run` FIRST, every time. Expect **5 chips, 2 shapes, 10 units**, arms
`recon1 recon2 recon3 recon4 reference` — five arms, because recon5 is
withdrawn. If the dry-run says anything else, stop and work out why before
submitting. Then drop `--dry-run` to submit.

`ECC_RERUN_OPTIMISER` now ships at `0` and `ECC_SCOPE` at `dev`, so cached
units are reused and the scope is the six-layer development one. Do not set
either back without saying why in progress.txt.

### 1.3 THE FIGURE PATH COLLIDES — protect what is already there

The placement study's stem is `ReconSweep_optimiser__<model>` and **carries no
architecture**. So this run writes

    results/figures/ReconSweep_optimiser__resnet18.png

which is **the file `eyeriss_like_wglb`'s placement study already occupies** —
the design every published number comes from. It will be overwritten.

Before you evaluate: copy the existing figure, its table and its manifest aside
(`results/figures/`, `results/tables/`, `results/manifests/`), and say in
progress.txt where you put them. The figure is reproducible from
`results/_raw/` with `--replot`, so this is cheap insurance, not a rescue.

The real multi-design deliverable is session 3's panel figure (§3). A
single-design run of this study is a checkpoint, not the end product.

### 1.4 Draw it, and predict before you look

Evaluate with `--eval-only`. **NOT `--replot`** — it reads `results/_raw/` only
and cannot draw newly-mapped units; phase 6 session 2 learned this the hard
way.

**Write down the expected DIRECTION of every trend in progress.txt BEFORE you
draw it.** A figure that contradicts the prediction is a bug until proven
otherwise — do not rationalise it. This design has four boundaries and a
withdrawn fifth, so its bar count differs from `eyeriss_like_wglb`'s by
construction; that is not a defect.

---

## 2. SESSION 2 — `eyeriss_v2_like_wglb`. DO NOT START IT.

Left for a later session, by the user's decision. Recorded here so session 1
knows where the boundary is, not so it can be done early.

    ECC_SCOPE=dev ECC_SWEEP=fix \
      ECC_APPROACHES="baseline embedded recon1 recon2 recon3 recon4 recon5" \
      ECC_ARCHS=eyeriss_v2_like_wglb \
      ECC_CONST_ARCH=eyeriss_v2_like_wglb \
      bash hpc/run_all.sh

Expect **6 chips, 2 shapes, 12 units**, arms `recon1..recon5 reference`. This
design declares FIVE boundaries: phase 6 added its 24 kB weight GLB and resized
`weight_noc` to its 288 B of routers, which is what made `recon2` mean the same
thing on all three designs. Its `weight_noc` is six words deep after
renormalisation and was flagged as a mapper risk — the smoke run mapped it in
14 minutes without trouble, so it is a watch item, not a blocker.

**Session 1: if you have finished §1 and the gate is green, STOP and say so.**
Write in progress.txt that session 2's design was deliberately not started.

---

## 3. SESSION 3 — the three-panel figure, after BOTH are mapped

Only once sessions 1 and 2 are both done. `fix` HOLDS the architecture, so
several designs in one placement figure is an ARCH sweep routed to the
placement renderer (CLAUDE.md):

    ECC_EXPERIMENT=recon ECC_SWEEP=arch ECC_SCOPE=dev ECC_MODELS=resnet18 \
      ECC_ARCHS="eyeriss_like_wglb simple_weight_stationary eyeriss_v2_like_wglb" \
      ECC_SWEEP_ARCHS="eyeriss_like_wglb simple_weight_stationary eyeriss_v2_like_wglb" \
      ECC_APPROACHES="baseline embedded recon1 recon2 recon3 recon4 recon5" \
      bash hpc/run_all.sh

17 chips, 34 units. **`ECC_MODELS=resnet18` is required**: an arch sweep with
two models is the PANEL layout and would evaluate `mobilenet_v2`, which is not
mapped for these designs.

**Each panel keeps its own x axis and its own two reference bars, so a
percentage on one panel says NOTHING about another.** Do not compare across
panels in prose.

---

## 4. Gate and rules

* **Re-take the golden FIRST** (§0), record why, then never `--update` again
  this session.
* After the re-take, **item 1 must stay byte-identical**. Task D declares
  nothing, so nothing should move. `eyeriss_like_wglb` especially — it is the
  design every published number comes from.
* **Run the gate ALONE.** `snapshot.py` collects each stage's output by mtime
  since that stage started, so a figure render in another shell lands inside a
  stage's window and is reported as a moved NUMBER. It is not one.
* Items 2–5 may gain result files for the newly-mapped design. That is RECORD,
  not a moved number: `compare.py` classifies it, and `--accept-record "<why>"`
  takes it with the reason written down. **A moved NUMBER is always red and no
  flag accepts it.**
* **The closing order is fixed and its ORDER is the point:**

      finish EVERY edit -> make guards -> python3 tools/gen_guards.py --check
      -> gate, with nothing else touching the repo -> only then --accept-record

  `make guards` must be the LAST thing before the gate, not merely somewhere
  before it: `tools/gen_guards.py` records each guard's call site as
  `file:line`, so editing ANY file holding a `guards.refusal()` call — even a
  comment above one — makes GUARDS.md stale, `test_guards_md_is_not_stale`
  goes red, that single failure flips pytest's exit status, and the gate reads
  it in `stages.tsv` as a MOVED NUMBER. One stale doc then looks like a broken
  model.
* Write files atomically (temp + `os.replace`), keep a pre-edit copy, run
  `bash -n env.sh` after every edit, and check every consumer still sources it.
* **`results/` is NOT committed by a phase commit** — check `git show --stat`
  on any recent one. It is generated and it churns on every gate run.

---

## 5. progress.txt and the commit — both, not one

**Record the session in `progress.txt`** in the house style: the gate state
before and after, what moved and why, what was predicted before it was drawn,
and what is still owed. Then **commit, with the attribution line**, and say in
your final message what you did NOT do.

A session that maps and draws but leaves no entry has produced numbers nobody
can trace back to a decision.

---

## 6. Still owed after session 1

* **Session 2's design**, `eyeriss_v2_like_wglb` (§2).
* **Session 3's panel figure** (§3), after both.
* **FINDINGS 2.11** is written but has NO NUMBERS from either new design. It is
  waiting on the mapping, not on a document. Whole models only for the final
  figures — the `dev` scope above is development, not a published number.
* Two figure defects phase 6 recorded and deliberately did not fix: a saving
  and a penalty print in the SAME RED, and each group reserves about a third
  more width than its bars use. Fix them or say why not.
