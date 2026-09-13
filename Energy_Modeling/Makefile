# =============================================================================
#  Makefile -- the three things that are easier to type than to remember.
#
#  IT RUNS NOTHING THAT COSTS COMPUTE. The mapper is `bash hpc/run_all.sh`, the
#  evaluation is `bash run.sh`, and neither belongs behind a target that could
#  be typed by accident. What is here is scaffolding and checking.
# =============================================================================

PY ?= python3

.PHONY: help arch layers guards test

help:
	@echo "make arch NEW=<name>   scaffold archs/<name>/ -- a design is a DIRECTORY"
	@echo "make layers            the layer rule and the settings contract"
	@echo "make guards            regenerate GUARDS.md -- every guard in one table"
	@echo "make test              the whole suite (needs the container for pandas)"
	@echo ""
	@echo "  module load apptainer && bash hpc/tl.sh make test"

# -----------------------------------------------------------------------------
#  A NEW ARCHITECTURE IS ONE DIRECTORY (ProjectRestructure section 5).
#
#  It writes the four files a design declares, each one a stub that FAILS the
#  schema until it is filled in -- a half-finished design must be loud, not
#  silently half-registered. Nothing else in the project needs editing.
# -----------------------------------------------------------------------------
arch:
	@test -n "$(NEW)" || { echo "usage: make arch NEW=<name>"; exit 2; }
	@test ! -d archs/$(NEW) || { echo "archs/$(NEW) already exists"; exit 2; }
	@$(PY) tools/scaffold_arch.py $(NEW)

layers:
	@$(PY) -m pytest tests/contract -q

# -----------------------------------------------------------------------------
#  GUARDS.md IS GENERATED, NEVER HAND-WRITTEN (ProjectRestructure section 6.5).
#
#  A guard runs on EVERY run -- including a SLURM job at 3am -- and stops it.
#  There are ~135 of them and no list of either guards or tests existed before
#  phase 6, which is the "impossible to know what is there" this answers.
#  `tests/contract/test_guards.py` fails when the file is stale, so it cannot
#  drift the way a hand-kept list would.
# -----------------------------------------------------------------------------
guards:
	@$(PY) tools/gen_guards.py

test:
	@$(PY) -m pytest eccenergy/tests tests -q
