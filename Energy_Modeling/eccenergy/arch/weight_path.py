"""The weight path of one design: the stages, and the Timeloop levels that ARE them.

`WEIGHT_PATHS[<arch>]` names the stages a weight crosses on its way from DRAM to
the MAC, outer to inner, and for each of them the Timeloop level names that make
it up. It is half of the placement space; `placements.py` is the other half, and
the two MUST BE EDITED TOGETHER -- `validate_placement_space()` (in
`arch/arms.py`) is what refuses a half-finished pair.

THE WEIGHT PATH IS READ FROM THE MODEL, NOT ASSUMED
---------------------------------------------------
`01_project_context_and_architectures.txt` Sec. 5.1 warns against substituting

    DRAM --> global weight SRAM --> RF --> MAC

for Eyeriss v2, whose GLB banks serve input activations and partial sums while
weights go from the memory interface through the hierarchical mesh to the PE
scratchpads. The tables here therefore name the levels of the architecture AS
MODELLED, and `weight_path()` (`toolchain/weight_stats.py`) refuses to proceed
if a level that carries Weights energy is not claimed by exactly one stage. A
placement whose stage is absent from the model is reported `unsupported`, never
silently skipped -- for `eyeriss_v2_like` there is no weight GLB at all, so no
"reconstruct at the global buffer" boundary exists to evaluate.

THE DRAM IS ONE STAGE, AND THE WHOLE OF IT IS REDUCIBLE  (2026-09-09)
---------------------------------------------------------------------
Accelergy's CactiDRAM bills a DRAM read as one flat per-bit DYNAMIC ACCESS
constant -- 8 pJ/bit for LPDDR4 as modelled, verified at 64.0 pJ per 8-bit
word -- and Timeloop charges the DRAM-to-chip network zero, so that constant is
the entire DRAM term:

    dram = DRAM weight energy x K/N   under EVERY boundary, R1 included

It used to be TWO stages, `dram_array` ((1 - f_if) of it, never reduced) and
`dram_interface` (f_if of it, reduced), with `f_if = ECC_DRAM_IF_FRAC = 0.40`.
That split charged a 38.1% bit cut as a 15.2% energy cut and is REMOVED: the
DRAM access is designed to fetch only the message bits of each codeword, so the
whole per-bit constant carries k of every n bits. `ECC_DRAM_PJ_PER_BIT` sets the
per-bit cost itself (`energy.apply_dram_override`); `archs/_shared/provenance.yaml`
`dram_access_energy` carries the citations. E_background and E_refresh are NOT
modelled (both 0) -- see env.sh section 4.

THE DECODER IS ON THE DRAM DIE  (since 2026-09-09)
--------------------------------------------------
01_project_context Sec. 1: the BCH decoder sits on the DRAM die and OFF the
fetch path (it corrects at write, on a scrub pass or on a prior access), so at
fetch time the stored codeword is already corrected and only the k message bits
of each n-bit codeword are driven across the DRAM interface. The array still
stores and reads the complete codeword -- a row activation and a burst move
whole words -- so `dram` is reduced under every boundary, R1 included.
`ECC_RECON_DECODE_SITE=controller` is the pre-2026-09-09 model (decode on the
fetch path, the complete codeword across the interface, DRAM identical on every
bar), kept as a runnable row so the change can be diffed; `stages_for()` and
`placements_for()` take the config and hand back that space when it is asked
for. The two reference bars keep controller-side correction under both settings:
Task 1's external parity is read from the array AND crosses the interface, and
Task 2's on-chip datapath consumes every weight bit, so its complete codeword
crosses too. Their functions are frozen and do not change.

ProjectRestructure phase 3 cut this out of `recon.py`, which was four layers in
one file. It is L2: a design's weight path is a fact about the design.
"""
from __future__ import annotations

import functools

from dataclasses import dataclass

from ..paths import ARCH_SRC
from . import design


# ===========================================================================
#  the weight path, per architecture
# ===========================================================================
@dataclass(frozen=True)
class Stage:
    """One stage of the weight path, and the Timeloop levels that ARE it.

    `prefixes` matches level names as Timeloop prints them. A network is named
    after the two levels it joins (`NoC: <outer> <==> <inner>`), so the outer
    level's name is the stable half and the prefix stops there.
    """
    key: str
    label: str
    kind: str                 # dram | network | storage
    prefixes: tuple
    reducible: bool           # can this stage ever carry the reduced form?
    evidence: str = ""

    def matches(self, level):
        return any(level == p or level.startswith(p) for p in self.prefixes)




#: Where the BCH decoder sits. `ondie` is the model since 2026-09-09 (decoder
#: on the DRAM die, off the fetch path, `dram` reducible under every
#: boundary); `controller` is the pre-2026-09-09 model kept as a runnable row
#: for the diff (`dram` not reducible, DRAM identical on every bar).
#: `config.RECON_DECODE_SITES` must stay in step with this tuple.
DECODE_SITES = ("ondie", "controller")

#: WHERE A NETWORK BOUNDARY'S ENCODERS SIT, and therefore how many times they
#: run. 02_.../01_project_context Sec. 7.1 states the tradeoff and asks for it
#: to be an experiment variable:
#:
#:     BEFORE MULTICAST   4b -> Encoder -> 8b -+-> PE  (x fanout)
#:         one reconstruction at the source, but FULL-WIDTH network traffic
#:     AFTER MULTICAST    4b -+-> Encoder -> PE  (x fanout)
#:         replicated encoders, but REDUCED-WIDTH shared transport
#:
#: `destination` (the model since 2026-09-09) charges the second: an encoder at
#: each destination, so the count is Timeloop's destination-side ARRIVALS,
#: `ingresses x multicast factor`. That is the only count consistent with a
#: boundary that also credits the network with carrying the reduced form -- an
#: encoder placed BEFORE the fanout would make that network full width, which
#: is the boundary above it.
#: `source` charges the first: one encoder before the fanout, count =
#: `ingresses`. It is kept as a runnable row because it is what the study
#: charged before 2026-09-09, so the change can be diffed; on its own it pairs
#: a source-side encoder count with a destination-side network saving.
#: `config.RECON_ENCODER_SITES` must stay in step with this tuple.
ENCODER_SITES = ("destination", "source")


#: The paths themselves are `archs/<name>/weight_path.yaml` since
#: ProjectRestructure phase 5. What used to be five tuples here -- one per
#: design, two of them built by slicing another -- is one loader and one schema,
#: so a design's stages are declared beside the chip they describe and nothing in
#: Python names a design.


@functools.lru_cache(maxsize=None)
def _declared_stages(arch):
    """This design's `weight_path.yaml`, as `Stage` records, or `()`.

    `()` when the design declares no weight path -- three designs are in that
    state today, and that is an answer rather than a gap: no reconstruction
    boundary has ever been written down for them.
    """
    doc = design.weight_path_doc(arch)
    if doc is None:
        return ()
    path = ARCH_SRC / arch / design.WEIGHT_PATH_FILE
    rows = design.validate_weight_path(arch, doc, path)
    return tuple(Stage(key=r["key"], label=r["label"], kind=r["kind"],
                       prefixes=tuple(r["prefixes"]),
                       reducible=bool(r["reducible"]),
                       evidence=r.get("evidence", ""))
                 for r in rows)


class _WeightPaths(dict):
    """`WEIGHT_PATHS[arch]`, filled from `archs/<name>/weight_path.yaml` on first
    use.

    IT IS STILL A PLAIN DICT UNDERNEATH, and deliberately: a test registers a
    SYNTHETIC design by assigning into it -- that is how "two boundaries that
    agree on all three axes are one chip" is shown without bending a real
    record -- so an entry that is already here always wins over the file.
    """

    def __missing__(self, arch):
        got = _declared_stages(arch)
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


def stages_of(arch):
    """This design's stages, outer to inner, or `()` if it declares none."""
    return WEIGHT_PATHS.get(arch) or ()


#: design -> its stages. Reads `archs/<name>/weight_path.yaml` the first time a
#: design is asked for; `in`, `keys()` and `get()` answer without raising, which
#: is what the three designs that declare no boundaries need.
WEIGHT_PATHS = _WeightPaths()


