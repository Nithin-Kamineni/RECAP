"""`make arch NEW=<name>` -- write the directory a new design IS.

ProjectRestructure section 5.2: after phase 5, adding an architecture is one
directory and no edit anywhere else. This writes that directory.

EVERY STUB IS DELIBERATELY INVALID. `design.yaml` has a label; the other three
files carry `TODO` where a citation belongs, and `arch_paper.yaml` is empty. A
design that is half-written must FAIL at load, naming the file -- a stub that
quietly validated would be a design whose weight path nobody ever wrote, and the
placement study would report savings against a path that was invented for it.

    make arch NEW=tpu_like
    $EDITOR archs/tpu_like/arch_paper.yaml      # the chip, every number cited
    $EDITOR archs/tpu_like/design.yaml          # label, mapspace, declared facts
    $EDITOR archs/tpu_like/weight_path.yaml     # the stages, outer to inner
    $EDITOR archs/tpu_like/placements.yaml      # the boundaries
    bash run.sh validate && bash run.sh diagnose

A design that declares NO reconstruction boundaries keeps `arch_paper.yaml`,
`design.yaml` and `README.md` and DELETES the other two -- three designs are in
that state today, and the schema accepts it rather than inventing a weight path
that was never cited.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

DESIGN = """# ONE DESIGN, DECLARED. `arch_paper.yaml` beside this file is the CHIP (every
# number cited); this is what the study needs to know ABOUT the chip.
#
name: {name}

# THE CORE CLOCK, in MHz. Reaches the mapper (globals_<arch>.yaml), so it is in
# the fingerprint. Omit it to run at ECC_GLOBAL_CYCLE_SECONDS (1 GHz).
clock_mhz: 1000

# COMPONENT STANDBY DENSITIES, in nW (POWER; the evaluator applies this
# design's cycle period). Charged to all three arms under ECC_STATIC_ENERGY=1.
# These are the study's values (see archs/eyeriss_like_wglb/design.yaml for
# where each came from); replace them if this design's silicon says otherwise.
leakage_nw:
  sram_bit: 2.693
  rf_bit: 70.0
  mac_instance: 7844.9

# What a figure axis says. A LABEL MAY NAME A STRUCTURE, NEVER A CAPACITY: a
# title is the last place a reader meets the design, so it must not be the one
# place quoting a number the file does not declare. `\\n` splits the line.
label: "{name}"

# Where this design sits on a default axis. Optional -- omit it and the design
# lands after the ordered ones, alphabetically.
# order: 8

# prompt_3's CONSTRAINED MAPSPACE: per loop dimension, the levels it may be
# split across. Every level not named is pinned to 1. WRITE THIS BEFORE MAPPING:
# a systematic walk of an unconstrained space is the wrong regime (FINDINGS 2.2)
# and a design with no entry here is searched unconstrained.
# mapspace_free_levels:
#   C: ["DRAM", "PE", "weights_spad"]
#   M: []
#   R: []
#   S: []
#   P: []
#   Q: []
#   N: []

# Declared per-design modelling facts. Each of these replaced a branch on the
# design's NAME in a driver, so add a field here rather than an `if` there.
#
# weights_stored_compressed: true
# compression_note: "TODO: what the published design stores, and what this
#                    study models instead"
# noc_published_share:
#   low: 0.06
#   high: 0.10
#   citation: "TODO: paper, section, figure"
"""

WIDTHS = """# THE WIDTH TABLE for this design (CLAUDE.md, PROTECTED SECTION): one word
# width PER ARM, keyed by the datawidth q that arm stores. The arms do NOT
# share a width; each row need only divide its OWN q. Depth is shared and is
# renormalised at the 8-bit row. These are the study's rows (the RULE in
# eccenergy/physics/widths.py reproduces them); a q not listed takes the rule.
widths:
  8: {spad_width: 96, glb_width: 384}
  7: {spad_width: 98, glb_width: 392}
  6: {spad_width: 96, glb_width: 384}
  5: {spad_width: 95, glb_width: 380}
  4: {spad_width: 96, glb_width: 384}
"""

WEIGHT_PATH = """# THE WEIGHT PATH of this design: the stages a weight crosses from DRAM to the
# MAC, OUTER TO INNER, and the Timeloop levels that ARE each of them.
#
# READ A REAL `timeloop-mapper.stats.txt` AND `timeloop-mapper.map.txt` from this
# design's cache before writing this. `weight_path()` refuses if a level carrying
# Weights energy is not claimed by exactly one stage, which is the check that
# catches a guessed path.
#
# kind:      dram | storage | network  (the FIRST stage must be the dram)
# prefixes:  level names as TIMELOOP PRINTS THEM
# reducible: can this stage ever carry the REDUCED representation?
# evidence:  why this stage exists, with its citation
stages:
  - key: "dram"
    label: "DRAM (dynamic access, whole term reducible)"
    kind: "dram"
    prefixes: ["DRAM"]
    reducible: true
    evidence: "TODO: the decoder is on the DRAM die and off the fetch path, so
               only the k message bits of each codeword are read out
               (01_project_context Sec. 1 and Sec. 4)"

  - key: "TODO_inner_stage"
    label: "TODO"
    kind: "storage"
    prefixes: ["TODO_level_name"]
    reducible: true
    evidence: "TODO: cite the paper section or the stats file this comes from"
"""

PLACEMENTS = """# THE RECONSTRUCTION BOUNDARIES of this design.
#
# `reduced` must be a PREFIX of the weight path's reducible stages in path
# order, and every reducible stage must be reached by some boundary --
# `arch.arms.validate_placement_space()` refuses anything else.
#
# site_counter: reads | fills | ingresses | deliveries. A NETWORK boundary's
# encoders run once per ARRIVAL: `deliveries` is `ingresses x multicast factor`.
placements:
  - key: "recon1"
    variant: "recon_source_noc_ingress"
    label: "R1 - reconstruct at the source"
    short: "R1\\n@ source"
    rating: "TODO/5"
    reduced: ["dram"]
    site_stage: "dram"
    site_counter: "reads"
    description: "TODO: what stays reduced, what is rebuilt, and what it pays"
"""

README = """# `{name}`

**TODO: what this design is, in one paragraph, and what it is NOT.**

| | |
|---|---|
| source | TODO: published / derived / reference_design / locally_authored |
| paper | TODO: citation |
| what diverges | TODO: every declared number that differs from the paper, and why |

Only `published` licenses using the design's name as the chip. `derived` differs
in one stated block; `reference_design` ships with timeloop-accelergy-exercises
and is merely NAMED after a paper; `locally_authored` reproduces nothing.

## Before mapping it

1. `arch_paper.yaml` -- every number cited, with a divergence table at the top
   if any of them is not the paper's.
2. `design.yaml` -- `mapspace_free_levels`, or the search is unconstrained.
3. `archs/_shared/standard.yaml` and `provenance.yaml` -- the cross-design
   contract this design has to satisfy.
4. `bash run.sh validate && bash run.sh diagnose`, then ONE shape before a
   matrix.
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[0])
        print("usage: make arch NEW=<name>")
        return 2
    name = argv[0]
    d = ROOT / "archs" / name
    if d.exists():
        print(f"{d} already exists")
        return 2
    d.mkdir(parents=True)
    files = {
        "design.yaml": DESIGN.format(name=name),
        "widths.yaml": WIDTHS,
        "weight_path.yaml": WEIGHT_PATH,
        "placements.yaml": PLACEMENTS,
        "README.md": README.format(name=name),
        "arch_paper.yaml": (
            "# TODO: the chip. Every number cited, with the divergence table at\n"
            "# the top if any of them is not the paper's. Copy the shape of a\n"
            "# design that is already here -- archs/eyeriss_like_wglb/ is the\n"
            "# most heavily annotated one.\n"),
    }
    for fname, text in files.items():
        (d / fname).write_text(text)
        print(f"  + archs/{name}/{fname}")
    print(f"\n{name} is scaffolded and is DELIBERATELY INVALID until the TODOs are")
    print("filled in. Nothing else in the project needs editing:")
    print(f"  bash run.sh validate      # the cross-design contract")
    print(f"  bash hpc/tl.sh python3 -m pytest tests/contract -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
