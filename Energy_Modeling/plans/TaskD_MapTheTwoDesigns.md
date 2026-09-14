# Task D — map the two new designs at scope, and draw them

**This is `plans/EnvReorganisation.md` phase 6's TASK D, left over when that
phase was committed.** The plan itself is delivered (phases 0–7, phase 7 at
00bdd15); this is the one piece of phase 6 that is queue time rather than work,
and it is unblocked. `progress.txt`'s PHASE 7 entry is the state it starts from.

Nothing here is read programmatically, like everything else in `plans/`.

---

/blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling

Read CLAUDE.md first (both PROTECTED sections), then GUARDS.md, then the
ENVREORGANISATION PHASE 7 entry at the end of progress.txt and the PHASE 6
(SESSION 2 OF 2) entry above it -- its "ALSO STILL OWED" list is your task.
Read plans/prompt_3.md §"Porting it to another design or layer" before you
touch a free-set, and restructure/README.md before you run the gate.

STATE. plans/EnvReorganisation.md IS DELIVERED: phases 0-7 are done and
committed, phase 7 at 00bdd15, gate GREEN before and after with no flag (302
fingerprint rows identical, every total identical, 535 passed / 10 skipped).
Do not re-open the plan. The launcher, env.sh, the metrics, the sweeps and the
docs are finished and gated.

WHAT IS LEFT IS PHASE 6's TASK D, AND IT IS UNBLOCKED. The two designs added in
phase 6 are still not mapped at scope and still have no figure. The cold-map
path is PROVEN on both -- smoke jobs 42155707_0 (simple_weight_stationary,
3m25s) and 42155709_0 (eyeriss_v2_like_wglb, 14m12s) both COMPLETED, 1 newly
mapped / 0 failed each, and neither predicted risk materialised: no shape came
back with nothing feasible, and v2's six-word-deep `weight_noc` did not
strangle the mapper. Do not re-smoke. The logs are in hpc/logs/.

  1. RUN IT. One line each, and they are queue time, not work:

       ECC_SCOPE=dev ECC_ARCHS=simple_weight_stationary \
         ECC_CONST_ARCH=simple_weight_stationary bash hpc/run_all.sh
       ECC_SCOPE=dev ECC_ARCHS=eyeriss_v2_like_wglb \
         ECC_CONST_ARCH=eyeriss_v2_like_wglb bash hpc/run_all.sh

     `--dry-run` first to see the bill. Units already cached are not
     submitted, so a rerun after a failure queues only what is left.

  2. DRAW IT with `--eval-only`. NOT `--replot`: it reads results/_raw/ only
     and cannot draw newly-mapped units. PREDICT THE DIRECTION of every trend
     BEFORE you draw it and write the prediction in progress.txt. A figure that
     contradicts the prediction is a bug until proven otherwise -- do not
     rationalise it.

  3. FINDINGS 2.11 is written but has NO NUMBERS from these two designs. It is
     waiting on step 2, not on a document. Whole models only for the final
     figures.

TWO FIGURE DEFECTS phase 6 recorded and deliberately did not fix; fix them or
say why not: a saving and a penalty print in the SAME RED, and each group
reserves about a third more width than its bars use.

GATE AND RULES:
  * Task D CHANGES WHAT IS COMPUTED for the two new designs, so gate item 1 is
    NOT expected to be byte-identical for them -- state which rows you expect
    to move and why BEFORE you run it, as the phase 6 entry does. Item 1 for
    `eyeriss_like_wglb` -- the design every published number comes from -- MUST
    be byte-identical; if it is not, STOP and put it back. Never `--update` to
    make red go away.
  * run the gate ALONE: nothing else may write results/ while it runs, or
    snapshot.py attributes another shell's writes to a stage and reports moved
    numbers that did not move.
  * the closing order is fixed: finish EVERY edit -> make guards ->
    python3 tools/gen_guards.py --check -> gate -> only then --accept-record.
    `make guards` must be the LAST thing before the gate: gen_guards.py records
    each guard's call site as file:line, so editing any file holding a
    guards.refusal() call makes GUARDS.md stale, test_guards_md_is_not_stale
    goes red, and the gate reads that as a MOVED NUMBER.
  * write files atomically (temp + os.replace), keep a pre-edit copy, `bash -n
    env.sh` after every edit, check every consumer still sources it;
  * results/ is NOT committed by a phase commit -- check `git show --stat` on
    any recent phase commit; it is generated and it churns on every gate run;
  * record the work in progress.txt and commit with the attribution line.
