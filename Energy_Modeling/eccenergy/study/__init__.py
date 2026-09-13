"""L4 -- the arms, the placement study, Task 4, and the energy bookkeeping.

`stacks` builds the three ECC arms (prompt_6's `build_stacks`); `energy` is the
category algebra over a Timeloop record; `baseline`, `embedded`, `validate`,
`audit` and `diagnose` are the drivers `run.sh` dispatches to.

The placement study (Task 3) is `placement_study` (the driver), `placement_eval`
(one boundary), `placement_tables` (what it prints), `placement_notes` (the prose
and the cache paths), `ert_view` (one plan per bar), `dilated_view` (the arm's own
mapping) and `narrowing` (the exactly-once rule). Task 4 is `dilation`,
`dilation_cache`, `dilation_tables` and `capacity`.

May import: `toolchain`, `arch`, `physics`, `contracts`, `settings`/`config`,
`paths` -- and each other.

DRIVERS THAT DRAW ARE L5. Phase 2 read section 4.2's tree as putting every
driver here; phase 3 applied Appendix C instead, for the two that call a
renderer: `sweep` and `panels` are now in `report/`, and NOTHING in this package
imports L5 any more. A driver that only prints -- `dilation`, `diagnose`,
`validate` -- stays here.
"""
