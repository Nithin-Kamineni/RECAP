"""L4 -- the arms, the drivers, and the energy bookkeeping.

`stacks` builds the three ECC arms (prompt_6's `build_stacks`); `energy` is the
category algebra over a Timeloop record; the rest are the drivers `run.sh`
dispatches to, one per stage.

May import: `toolchain`, `arch`, `physics`, `contracts`, `settings`/`config`,
`paths` -- and each other.

WHY THE DRIVERS ARE HERE AND THE RENDERERS ARE NOT. ProjectRestructure section
4.2's tree lists `sweep.py`, `panels.py`, `baseline.py`, `embedded.py`,
`validate.py`, `audit.py` and `diagnose.py` under `study/`, while Appendix C
sends `experiments/{sweep,panels}.py` to `report/`. Both cannot hold: there is
one `panels.py` in `plots/` (the renderer) and another in `experiments/` (the
driver that calls it). The tree wins, and the line is DRIVERS ARE L4, RENDERERS
ARE L5 -- so `study/panels.py` calls `report/panels.py`, and no L5 module is
imported by anything below it except through that call.
"""
