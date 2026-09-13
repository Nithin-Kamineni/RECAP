"""PHASE 5, one-off: write `archs/<name>/{design,weight_path,placements}.yaml`
from the tables that are about to stop being code.

Every string -- every label, every `evidence`, every `description` -- is moved
BYTE FOR BYTE out of the Python and into the YAML, because those strings are
citations and they are the reason the tables are worth keeping at all.

    bash hpc/tl.sh python3 restructure/emit_design_data.py --plan
    bash hpc/tl.sh python3 restructure/emit_design_data.py --apply
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, ".")

from eccenergy.arch import placements as P                       # noqa: E402
from eccenergy.arch import weight_path as W                      # noqa: E402
from eccenergy.arch.patch import MAPSPACE_FREE_LEVELS            # noqa: E402
from eccenergy.settings.arch import ARCH_LABELS, BRACKET_PAIRS, KNOWN_ARCHS  # noqa: E402
from eccenergy.paths import ARCH_SRC                             # noqa: E402

#: The three sites in production code that branched on a design NAME before
#: phase 5 (ProjectRestructure section 5.1.1, finding 3), as the fields that
#: replace them.
PER_DESIGN_FLAGS = {
    "eyeriss_v2_like": {
        "weights_stored_compressed": True,
        "noc_published_share": [0.06, 0.10],
    },
    "eyeriss_v2_like_wglb": {
        "weights_stored_compressed": True,
        "noc_published_share": [0.06, 0.10],
    },
}

NOC_CITATION = ("JETCAS 2019 Sec. V / Fig. 18: the hierarchical mesh is "
                "\"6%-10% of the total energy consumption\"")
CSC_NOTE = ("the published PE stores weights CSC-compressed with a separate "
            "address scratchpad (JETCAS 2019 Table IV: weight data 288B, weight "
            "address 16x7b); this study models the design DENSE")


def _q(s):
    """A YAML scalar that survives every character these strings contain.

    Always double-quoted, with `\n` escaped rather than written as a block
    scalar: a figure label like "R1\n@ NoC source" is two lines whose second one
    starts with `@`, and a block scalar's indentation rules make that a parse
    error waiting for the next person who reflows a file.
    """
    if s is None:
        return "null"
    s = (str(s).replace("\\", "\\\\").replace('"', '\\"')
         .replace("\n", "\\n").replace("\t", "\\t"))
    return '"' + s + '"'


def design_yaml(name):
    flags = PER_DESIGN_FLAGS.get(name, {})
    out = [
        "# ONE DESIGN, DECLARED. ProjectRestructure phase 5: a new architecture is",
        "# this directory and nothing else -- no Python anywhere names it.",
        "#",
        "# `arch_paper.yaml` is the CHIP (every number cited); this file is what the",
        "# study needs to know ABOUT the chip that the chip's own YAML does not say.",
        "#",
        "# The design's CLOCK is deliberately NOT here: env.sh section 7's",
        "# ECC_ARCH_CLOCK_MHZ table owns it, because it is a knob a run may override",
        "# and env.sh is the one file a user edits (CLAUDE.md).",
        f"name: {name}",
        f"label: {_q(ARCH_LABELS.get(name, name))}",
    ]
    partner, why = BRACKET_PAIRS.get(name, (None, None))
    if partner:
        out += ["# This design's number is a BOUND unless its partner is plotted beside it.",
                f"bracket_partner: {_q(partner)}", f"bracket_reason: {_q(why)}"]
    if name in MAPSPACE_FREE_LEVELS:
        out += ["",
                "# prompt_3's CONSTRAINED MAPSPACE: for each loop dimension, the levels it",
                "# may be split across. Every level not named here is pinned to 1, which is",
                "# what collapses the index-factorization space to something the mapper can",
                "# search exhaustively. A design with no entry is searched unconstrained --",
                "# and a systematic walk of an unconstrained space is the wrong regime",
                "# (FINDINGS 2.2), so write this before mapping a new design.",
                "mapspace_free_levels:"]
        for dim, levels in MAPSPACE_FREE_LEVELS[name].items():
            if levels:
                out.append(f"  {dim}: [{', '.join(_q(l) for l in levels)}]")
            else:
                out.append(f"  {dim}: []")
    if flags.get("weights_stored_compressed"):
        out += ["",
                "# The published design stores weights COMPRESSED and this study models it",
                "# DENSE. Every result carries the caveat; before phase 5 this was an",
                f"# `arch.startswith(\"eyeriss_v2\")` in study/placement_study.py.",
                "weights_stored_compressed: true",
                f"compression_note: {_q(CSC_NOTE)}"]
    if flags.get("noc_published_share"):
        lo, hi = flags["noc_published_share"]
        out += ["",
                "# The share of TOTAL energy the design's own paper reports for its",
                "# interconnect. `study/audit.py` and `study/baseline.py` check the modelled",
                "# share against it -- a miss on one model is a prompt to look, a miss on",
                "# EVERY model is a broken constant. Before phase 5 this band was two",
                "# `arch.startswith(\"eyeriss_v2_like\")` branches.",
                "noc_published_share:",
                f"  low: {lo}", f"  high: {hi}",
                f"  citation: {_q(NOC_CITATION)}"]
    return "\n".join(out) + "\n"


def weight_path_yaml(name):
    stages = W.WEIGHT_PATHS[name]
    out = [
        "# THE WEIGHT PATH of this design: the stages a weight crosses from DRAM to",
        "# the MAC, OUTER TO INNER, and the Timeloop levels that ARE each of them.",
        "#",
        "# It is half of the placement space; `placements.yaml` beside it is the other",
        "# half, and the two are loaded together or not at all -- which is what makes",
        "# \"the two tables must be edited together\" a property of the directory instead",
        "# of a warning in CLAUDE.md.",
        "#",
        "# kind:      dram | storage | network",
        "# prefixes:  level names as TIMELOOP PRINTS THEM. A network is named after the",
        "#            two levels it joins (`NoC: <outer> <==> <inner>`), so the outer",
        "#            level's name is the stable half and the prefix stops there.",
        "# reducible: can this stage ever carry the REDUCED representation?",
        "# evidence:  why this stage exists, with its citation. Not decoration: it is",
        "#            what a reader checks the model against.",
        "stages:",
    ]
    for s in stages:
        out += [f"  - key: {_q(s.key)}",
                f"    label: {_q(s.label)}",
                f"    kind: {_q(s.kind)}",
                f"    prefixes: [{', '.join(_q(p) for p in s.prefixes)}]",
                f"    reducible: {'true' if s.reducible else 'false'}",
                f"    evidence: {_q(s.evidence)}"]
    return "\n".join(out) + "\n"


def placements_yaml(name):
    out = [
        "# THE RECONSTRUCTION BOUNDARIES of this design: which stages of",
        "# `weight_path.yaml` keep the REDUCED representation, and where the encoder",
        "# that rebuilds the rest sits.",
        "#",
        "# `reduced` must be a PREFIX of the path's reducible stages in path order, and",
        "# every reducible stage must be reached by some boundary --",
        "# `arch.arms.validate_placement_space()` refuses anything else, because a stage",
        "# nobody reaches leaves every boundary below it reporting a saving that the",
        "# whole list understates.",
        "#",
        "# site_counter: reads | fills | ingresses | deliveries.  A NETWORK boundary's",
        "# encoders run once per ARRIVAL, not once per injection: `deliveries` is",
        "# `ingresses x multicast factor`. ECC_RECON_ENCODER_SITE=source swaps it back.",
        "#",
        "# `rating` is the source discussion's HYPOTHESIS, carried so the result can be",
        "# read against it. It is not a result.",
        "placements:",
    ]
    for p in P.PLACEMENTS[name]:
        out += [f"  - key: {_q(p.key)}",
                f"    variant: {_q(p.variant)}",
                f"    label: {_q(p.label)}",
                f"    short: {_q(p.short)}",
                f"    rating: {_q(p.rating)}",
                f"    reduced: [{', '.join(_q(r) for r in p.reduced)}]",
                f"    site_stage: {_q(p.site_stage)}",
                f"    site_counter: {_q(p.site_counter)}",
                f"    description: {_q(p.description)}"]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args(argv)
    if not (a.apply or a.plan):
        ap.error("--plan or --apply")
    for name in KNOWN_ARCHS:
        d = ARCH_SRC / name
        files = {d / "design.yaml": design_yaml(name)}
        if name in W.WEIGHT_PATHS:
            files[d / "weight_path.yaml"] = weight_path_yaml(name)
            files[d / "placements.yaml"] = placements_yaml(name)
        for path, text in files.items():
            print(f"  {'+' if a.apply else '?'} {path}  ({len(text.splitlines())} lines)")
            if a.apply:
                path.write_text(text, newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
