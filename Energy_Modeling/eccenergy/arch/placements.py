"""The reconstruction boundaries of a design: what stays reduced, and where.

`PLACEMENTS[<arch>]` is the list of boundaries evaluated for one design, each a
`Placement`: which stages of `weight_path.py`'s table keep the REDUCED
representation, where the encoder sits, and what drives the reconstruction
count. The two tables ARE the placement space and must be edited together.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
`stacks.build_stacks()`'s `recon` arm is ONE placement, applied to every
architecture at once: it scales the weight share of every on-chip category by
K/N and charges one reconstruction per DRAM codeword. That is a single point in
the space these tables describe, and it is NOT one of the boundaries below -- it
reduces the SPad *and* charges reconstruction only once per DRAM fetch, which no
physical boundary does, and it must never be quoted as one.

Designs do not have the same boundaries, because each has its own weight path.
`eyeriss_like_wglb` (Eyeriss v1) and `simple_weight_stationary` have five;
`eyeriss_v2_like` has four. `eyeriss_v2_like_wglb` is REFUSED -- its `weight_glb`
stage has no boundary (prompt_6 Appendix B has the fix).

THE FIVE BOUNDARIES
-------------------
Sec. 5.2/5.3 of `01_project_context_and_architectures.txt`, in the order the plan
lists them. Ratings are the source discussion's HYPOTHESES, carried so the
result can be read against them; they are not results.

    R1  recon1  reconstruct at the source / weight-NoC ingress          [2/5]
                (reduces the DRAM interface and nothing on chip: it isolates
                the interface saving every boundary shares from any on-chip
                saving, and pays one reconstruction per DRAM codeword)
    R2  recon2  reconstruct at the destination-cluster boundary         [4/5]
    R3  recon3  reconstruct at the PE weight-SPad input                 [4/5]
    R4a recon4  reconstruct on every weight-SPad read                   [3/5]

R4b -- SPad output plus a reconstructed-weight reuse register -- was REMOVED on
2026-09-10. Measured on the cached mappings, consecutive weight reuse is 1 on 20
of 21 resnet18 layers, so a latch-sized register catches nothing (a cyclic walk
is the LRU worst case, with no partial hit rate); a register that DOES pay has
to hold the whole inner tile, up to 384 weights against a 384-weight scratchpad.
The same argument rules it out on every dataflow in the study: output- and
input-stationary hold the psum or the activation in the PE, so weights stream
past the MAC faster still. FINDINGS 7.1. Do not reintroduce one without first
re-measuring `consecutive_run` on the target mapping.

A NETWORK boundary's encoders run once per ARRIVAL, not once per injection --
the count is `Ingresses x Multicast factor`. Charging the ingress count pairs a
destination-side saving with a source-side cost, which is not a placement.
`ECC_RECON_ENCODER_SITE=source` reproduces the earlier numbers for a diff.

ProjectRestructure phase 3 cut this out of `recon.py`. It is L2, beside the
weight path it indexes, which is what lets `arch/fingerprint.py` reach a
placement without importing a study module.
"""
from __future__ import annotations

import functools

from dataclasses import dataclass, replace

from ..contracts.errors import ConfigError
from ..paths import ARCH_SRC
from . import design, weight_path

from .weight_path import DECODE_SITES, ENCODER_SITES, WEIGHT_PATHS


# ===========================================================================
#  the placements
# ===========================================================================
@dataclass(frozen=True)
class Placement:
    """One reconstruction boundary: what stays reduced, and where the encoder is.

    `reduced` is the set of stage keys that carry the REDUCED representation --
    everything from the die's output down to the boundary, so it always starts
    with `dram_interface` (the decoder is on the DRAM die; see the module
    docstring) and `validate_placement_space()`'s prefix rule enforces that
    order. `site_stage` plus `site_counter` say which access count drives the
    reconstruction count:

        reads       scalar reads out of that stage (per weight delivered)
        fills       scalar fills into that stage (per weight stored)
        ingresses   network ingresses -- words INJECTED into that network, so
                    one encoder before the fanout
        deliveries  network destination-side ARRIVALS, `ingresses x multicast
                    factor`, so one encoder at each destination. This is what a
                    boundary that credits the network with carrying the reduced
                    form has to pay (Sec. 7.1's "after multicast" case); the
                    two differ by the multicast factor, which is 7 on some
                    eyeriss shapes. `ECC_RECON_ENCODER_SITE=source` swaps it
                    back to `ingresses` for the diff -- see `ENCODER_SITES`.
    """
    key: str                  # the env.sh spelling: recon1 .. recon5
    variant: str              # the results-store variant name (baseline.py)
    label: str
    short: str                # figure label, may contain a newline
    rating: str               # the source discussion's hypothesis, not a result
    reduced: tuple
    site_stage: str
    site_counter: str
    description: str


#: The boundaries themselves are `archs/<name>/placements.yaml` since
#: ProjectRestructure phase 5 -- four tuples of cited prose became four files
#: beside the designs they describe. `eyeriss_v2_like` and `eyeriss_v2_like_wglb`
#: shared one table here and now each declare their own: a copy that is DATA can
#: be diffed, and the schema checks both against their own weight path.


@functools.lru_cache(maxsize=None)
def _declared_placements(arch):
    """This design's `placements.yaml`, as `Placement` records, or `()`."""
    doc = design.placements_doc(arch)
    if doc is None:
        return ()
    path = ARCH_SRC / arch / design.PLACEMENTS_FILE
    stages = weight_path.stages_of(arch)
    if not stages:
        raise ConfigError(
            f"{path}: this design declares boundaries but no "
            f"{design.WEIGHT_PATH_FILE}. The two are loaded together or not at "
            f"all -- a boundary is a cut through a weight path.")
    rows = design.validate_placements(
        arch, doc, [{"key": s.key, "reducible": s.reducible} for s in stages], path)
    return tuple(Placement(key=r["key"], variant=r["variant"], label=r["label"],
                           short=r["short"], rating=str(r.get("rating", "")),
                           reduced=tuple(r["reduced"] or ()),
                           site_stage=r["site_stage"],
                           site_counter=r["site_counter"],
                           description=r["description"])
                 for r in rows)


class _Placements(dict):
    """`PLACEMENTS[arch]`, filled from `archs/<name>/placements.yaml` on first
    use. An entry assigned into it wins -- see `_WeightPaths` beside it."""

    def __missing__(self, arch):
        got = _declared_placements(arch)
        if not got:
            raise KeyError(arch)
        self[arch] = got
        return got

    def __contains__(self, arch):
        try:
            self[arch]
        except KeyError:
            return False
        return True

    def get(self, arch, default=None):
        try:
            return self[arch]
        except KeyError:
            return default

    def keys(self):
        declared = [a for a in design.known_archs() if a in self]
        return tuple(declared) + tuple(k for k in dict.keys(self)
                                       if k not in declared)

    def __iter__(self):
        return iter(self.keys())

    def values(self):
        return tuple(self[a] for a in self.keys())

    def items(self):
        return tuple((a, self[a]) for a in self.keys())


def placements_of(arch):
    """This design's boundaries, or `()` if it declares none."""
    return PLACEMENTS.get(arch) or ()


#: design -> its boundaries, read from `archs/<name>/placements.yaml`.
PLACEMENTS = _Placements()

#: Reference bars every placement is read against. They are not placements;
#: they are Task 1's and Task 2's numbers, produced by Task 1's and Task 2's own
#: functions so the figure cannot disagree with those files.
REFERENCE_BARS = (("baseline", "Baseline\nexternal parity"),
                  ("embedded", "Embedded\nno ext. parity"))


def supported_archs():
    return tuple(sorted(WEIGHT_PATHS))


def decode_site(cfg):
    """`ondie` (the model) or `controller` (the pre-2026-09-09 diff row)."""
    site = getattr(cfg, "recon_decode_site", "ondie") if cfg is not None else "ondie"
    if site not in DECODE_SITES:
        raise ValueError(f"unknown decode site {site!r}; one of {DECODE_SITES}")
    return site


def encoder_site(cfg):
    """`destination` (the model) or `source` (the pre-2026-09-09 diff row)."""
    site = (getattr(cfg, "recon_encoder_site", "destination")
            if cfg is not None else "destination")
    if site not in ENCODER_SITES:
        raise ValueError(f"unknown encoder site {site!r}; one of {ENCODER_SITES}")
    return site


def stages_for(arch, cfg=None):
    """The weight path of `arch`, as the decode site makes it.

    The tables above are written for the on-die decoder. Under `controller`
    the complete codeword crosses the interface, so `dram` is handed
    back with `reducible=False` and nothing else changes -- which is exactly
    what makes the old numbers come back rather than a different model.
    """
    if arch not in WEIGHT_PATHS:
        raise KeyError(arch)
    stages = WEIGHT_PATHS[arch]
    if decode_site(cfg) == "controller":
        stages = tuple(replace(s, reducible=False) if s.key == "dram"
                       else s for s in stages)
    return stages


def effective_placement(placement, cfg):
    """`placement` as the two site knobs make it.

    Under `ECC_RECON_DECODE_SITE=controller` the DRAM interface is not
    reducible, so it leaves every placement's `reduced` set. Under
    `ECC_RECON_ENCODER_SITE=source` a network boundary counts its
    reconstructions at the network's INGRESSES rather than its destination-side
    arrivals -- one encoder before the fanout instead of one per destination.
    Both are the pre-2026-09-09 model, kept runnable so the change is a diff.
    """
    if decode_site(cfg) == "controller" and "dram" in placement.reduced:
        placement = replace(placement, reduced=tuple(
            k for k in placement.reduced if k != "dram"))
    if encoder_site(cfg) == "source" and placement.site_counter == "deliveries":
        placement = replace(placement, site_counter="ingresses")
    return placement


def placements_for(arch, cfg=None):
    if arch not in PLACEMENTS:
        raise KeyError(arch)
    return tuple(effective_placement(p, cfg) for p in PLACEMENTS[arch])


def placement_by_key(arch, key, cfg=None):
    for p in placements_for(arch, cfg):
        if p.key == key:
            return p
    return None


