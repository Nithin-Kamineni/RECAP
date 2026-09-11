"""Task 4 step 1: does CAPACITY DILATION actually cut DRAM weight refetch?

FINDINGS section 9 item 0 states the mechanism and then refuses to let it be
modelled before it is measured:

    "Validate the assumption before building anything. Real tiles are
     integers, so 1.615x capacity may buy a whole extra tile on some layers and
     nothing on others, and the whole table scales with that one number. It
     needs no new modelling code: map each design twice, once at declared
     weight capacity and once at capacity x N/K, and diff the per-layer DRAM
     weight reads."

This module is that diff, and nothing else. It invokes no mapper, evaluates no
placement and writes no figure: it reads two (or more) mapper caches that
`hpc/map_capacity_sweep.sh` has already filled and reports, per layer, what the
extra capacity bought.

WHY THE ANSWER IS NOT AUTOMATICALLY POSITIVE
--------------------------------------------
Three independent reasons a dilation can buy nothing, all of them visible in
the table this prints:

1. WEIGHT CAPACITY IS NOT THE BINDING CONSTRAINT. A mapping that leaves most
   of the weight scratchpad unused is not refetching because it ran out of
   weight room -- it is refetching because the loop nest that would keep the
   tile resident costs more somewhere else (activations, partial sums, the
   spatial fanout). `residency` against `capacity` in the table says which.
   At the DECLARED sizes this is the common case, which is why the sweep
   shrinks the reference until capacity binds.

2. TILES ARE INTEGERS. Timeloop chooses integer loop factors, so a 1.6154x
   capacity is only spendable if some factor of the layer's C/M/R/S can grow
   into it. A 2x-or-nothing dimension gives 0 % on one layer and 50 % on the
   next, and the mean over layers is NOT the mean of the model: every layer
   carries its own DRAM traffic and its own instance count, so a spot check on
   two layers is a spot check and `--survey`'s layer-weighted row is the only
   line here that speaks for a whole model.

3. THE SEARCH IS STOCHASTIC. `random_pruned` at a finite victory condition can
   miss the better mapping the extra capacity permits. A NEGATIVE differential
   -- the dilated arm refetching MORE -- is proof of that, not of a mechanism,
   and is flagged rather than averaged in silently.

4. ACCELERGY PRICES THE DILATED ARRAY AS A BIGGER ONE, and the mapper believes
   it. Expressing the dilation as `depth x N/K` makes CACTI cost a deeper SRAM:
   measured at 1.18-1.45x per access on these designs. So an energy-objective
   search has a positive reason to leave the extra room unused, which is a bias
   AGAINST the hypothesis. The energy is corrected back to the declared array
   (`recon.capacity_dilation_correction`), the mapping cannot be, and the
   `ERT x` column reports it -- so every Task 4 number here is a lower bound.

WHAT THE FIRST MEASUREMENT SAID  (2026-09-09, eyeriss_like, resnet18, energy)
-----------------------------------------------------------------------------
Reason 1, and decisively. On `layer2.0.conv1` the declared and dilated loop
nests are BYTE-IDENTICAL, at every capacity from x0.125 to x1.6154, and the
reason is legible in one line of the table: `ifmap_spad` is at 24/24 = 100 %
while `weights_spad` is at 96/448 = 21 % and the dilation takes it to 96/724 =
13 %. The refetch is caused by the `for Q in [0:2) / for P in [0:4)` pair
sitting above the only weight-holding level, and those loops are there because
the PE ran out of INPUT-ACTIVATION room, not weight room. Reconstruction
enlarges the weight buffer only. It therefore relieves a constraint that was
not binding, and the mapper hands back the same mapping.

That is a property of the DATAFLOW, not of the code rate, and it is why this
module exists: the first-order model of FINDINGS section 9 item 0 predicts
refetch 8.000 -> 5.333 on that layer and the measurement is 8.000 -> 8.000.

WHAT THE DIFFERENTIAL IS WORTH, AND WHY IT IS NOT THE 0.152 OF TASK 3
---------------------------------------------------------------------
Under a fixed mapping (Task 3) both arms issue the same DRAM reads, and the
reconstruction arm saves the K/N narrowing of each: `(1 - K/N)` = 0.381 at
BCH(63,39), since the f_if split was removed. A read that is NEVER ISSUED saves the
array as well as the interface -- the whole `pJ_per_read` -- so its efficiency
is 1.0. That is the entire point of Task 4 and it is why the two savings are
reported separately below instead of being summed into one number.

The DRAM figure this prints is a LOWER BOUND on the Task 4 result and an
UPPER BOUND on the DRAM part of it, in this precise sense: it is exact for the
DRAM term, and it ignores that the dilated mapping also moves activation,
partial-sum and on-chip weight energy, which can go either way. The full
account is `experiments/recon.py` run against a dilated cache; this module
exists to decide whether that is worth building for a given design at all.

    python3 -m eccenergy.experiments.dilation --survey
    python3 -m eccenergy.experiments.dilation --layers "layer2.0.conv1 layer4.0.conv2"
    python3 -m eccenergy.experiments.dilation --csv results/tables/dilation.csv
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import sys

from .. import archs as archmod
from .. import code_widths
from .. import config as configmod
from .. import paths as pathsmod
from .. import recon as reconmod
from .. import workloads as workloadsmod

_NUM = r"([\d.eE+-]+)"


def _grab(pattern, text, cast=float, default=None):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else default


def _dataspace_block(body, dataspace):
    """The `Weights:` sub-block of one level's STATS section."""
    m = re.search(rf"\n    {dataspace}:\n(.*?)(?=\n    \w[\w ]*:\n|\Z)", body, re.S)
    return m.group(1) if m else None


def _level_block(text, level):
    for part in text.split("=== "):
        if part.startswith(level + " ===") and "STATS" in part:
            return part
    return None


@dataclasses.dataclass
class MappedLayer:
    """The handful of facts a capacity diff needs out of one cached mapping."""
    shape: str
    stats_path: str
    #: DRAM
    weights: int                  # Partition size: the layer's unique weights
    dram_weight_reads: float      # Scalar reads: what actually crosses
    dram_pj_per_read: float
    #: the weight-carrying storage level the dilation rewrote
    weight_level: str             # the INNERMOST dilated level: refetch, G_rec
    weight_levels: tuple          # every level the dilation rewrites
    residency: int                # Utilized capacity, per instance
    capacity: int                 # Effective size, per instance
    instances: int
    #: totals, for a denominator
    total_pj: float
    computes: float
    mac_pj: float
    #: PARALLELISM. Mandatory in every table: an objective that serialises
    #: changes this while the capacity changes too, and then no capacity effect
    #: can be separated from it. That is what withdrew the first Task 4 pass
    #: (FINDINGS 7.7) and the column is what caught it.
    pes_used: int = 0
    pes_declared: int = 0
    cycles: float = 0.0
    arch_util: float = 0.0
    #: EVERY weight-carrying level, so `room` and `held` are the design's, not
    #: the innermost level's -- `eyeriss_like_wglb` holds weights in two places
    #: and the GLB is the one whose dilation can absorb a DRAM-level loop.
    level_rows: tuple = ()
    #: the `fp-<hash>` this record was read from. Printed, because a comparison
    #: across two fingerprints is a comparison of two ARCHITECTURES.
    fingerprint: str = ""
    #: EVERY (level, dataspace) pair, not just the weight ones. The weight-only
    #: view hid 97 % of the measured effect -- see `_all_level_rows`.
    all_rows: tuple = ()
    #: `{dim: factor}` for the DRAM-level loops, and the per-dataspace
    #: re-traversal factors they imply.
    dram_loops: dict = dataclasses.field(default_factory=dict)

    @property
    def refetch(self):
        return self.dram_weight_reads / self.weights if self.weights else 0.0

    @property
    def weight_room(self):
        """Total on-chip WEIGHT room the design has, summed over every level
        the dilation rewrites and multiplied by the instances actually used."""
        return sum(r["capacity"] * r["instances"] for r in self.level_rows)

    @property
    def weights_held(self):
        """Weights the mapping actually keeps on chip, same summation.

        THIS IS THE COLUMN THAT SAYS WHETHER A DILATION DID ANYTHING. If DRAM
        reads move while this is unchanged at every level, the extra room was
        not spent and the reads moved for some other reason -- a different loop
        PERMUTATION, which the search can find at the declared capacity too.
        Measured on `simple_weight_stationary` layer2.0.conv1 (2026-09-09):
        reads halved at x1.6154 with `held` identical at x1, x1.6154 and x4,
        and x4 refetched 2.00x again. That is search noise, not capacity.
        """
        return sum(r["residency"] * r["instances"] for r in self.level_rows)

    @property
    def fill(self):
        return (self.weights_held / self.weight_room) if self.weight_room else 0.0

    @property
    def weight_instances(self):
        """Instances of the INNERMOST weight level -- the PE count that
        multiplies `weights per PE` into `room`.

        NOT the same number as `pes_used` on a design with more than one MAC
        per PE: `eyeriss_v2_like` runs 2 SIMD lanes per PE, so 144 PEs are 288
        arithmetic instances. Both are reported, because `room` is built from
        this one and the serialisation confound is visible in either.
        """
        return self.level_rows[-1]["instances"] if self.level_rows else 0

    @property
    def weights_per_pe(self):
        """Capacity per instance of the INNERMOST weight level -- "weights per
        PE" when that level is inside the PE, which it is on every design here.
        """
        return self.level_rows[-1]["capacity"] if self.level_rows else 0

    @property
    def dram_weight_pj(self):
        return self.dram_weight_reads * self.dram_pj_per_read

    def dram_row(self, dataspace):
        for r in self.all_rows:
            if r["level"] == "DRAM" and r["dataspace"] == dataspace:
                return r
        return None

    def dataspace_refetch(self, dataspace):
        """DRAM traffic for `dataspace` divided by its unique footprint.

        For Outputs this counts the accumulation round-trips: a partial sum
        written out and read back once per DRAM-level reduction loop. It is
        the quantity the weight-buffer size actually controls on this design.
        """
        r = self.dram_row(dataspace)
        if not r or not r["partition"]:
            return 0.0
        moved = r["reads"] + r["fills"] + r["updates"]
        return moved / r["partition"]

    def reduction_factor_for(self, dataspace):
        return reduction_factor(self.dram_loops, dataspace)

    def energy_by_dataspace(self, dataspace):
        return sum(r["energy_pJ"] for r in self.all_rows
                   if r["dataspace"] == dataspace)

    def total_pj_at(self, mac_pj_override):
        """Total energy with the study's MAC denominator substituted.

        Every percentage in FINDINGS is quoted against 0.23 pJ/MAC (section
        7.3), not against the ERT's 1.16877, and on these designs the MAC is
        up to 44 % of the run -- so a percentage against the raw summary would
        be diluted 1.5-2x relative to every other number in the study. The MAC
        count is mapping-invariant, so this is the same evaluator-side rescale
        `ECC_MAC_PJ_OVERRIDE` performs, applied to the summary line.
        """
        if mac_pj_override is None:
            return self.total_pj
        return self.total_pj - self.computes * self.mac_pj + self.computes * mac_pj_override


def read_mapped_layer(stats_path, weight_level, weight_levels=()):
    """Parse one `timeloop-mapper.stats.txt`. Returns None if it is not usable."""
    stats_path = pathlib.Path(stats_path)
    if not stats_path.is_file():
        return None
    text = stats_path.read_text()
    dram = _level_block(text, "DRAM")
    wl = _level_block(text, weight_level)
    if not dram or not wl:
        return None
    dw = _dataspace_block(dram, "Weights")
    ww = _dataspace_block(wl, "Weights")
    if not dw or not ww:
        return None

    computes = _grab(r"Computes\s*=\s*(\d+)", text, float, 0.0)
    mac_fj = _grab(rf"\n\s+mac\s+=\s+{_NUM}", text, float, 0.0)
    total_uj = _grab(rf"\nEnergy:\s*{_NUM}\s*uJ", text, float, 0.0)
    pes_used, pes_declared = _compute_instances(text)
    return MappedLayer(
        shape=stats_path.parent.name,
        stats_path=str(stats_path),
        weights=int(_grab(r"Partition size\s*:\s*(\d+)", dw, int, 0)),
        dram_weight_reads=_grab(rf"Scalar reads \(per-instance\)\s*:\s*{_NUM}", dw,
                                float, 0.0),
        dram_pj_per_read=_grab(rf"Energy \(per-scalar-access\)\s*:\s*{_NUM}", dw,
                               float, 0.0),
        weight_level=weight_level,
        weight_levels=tuple(weight_levels or (weight_level,)),
        residency=_grab(r"Utilized capacity\s*:\s*(\d+)", ww, int, 0),
        capacity=_grab(r"Effective size\s*:\s*(\d+)", wl, int, 0),
        instances=_grab(r"Utilized instances \(max\)\s*:\s*(\d+)", ww, int, 1),
        total_pj=total_uj * 1e6,
        computes=computes,
        mac_pj=mac_fj / 1000.0,
        pes_used=pes_used,
        pes_declared=pes_declared,
        cycles=_grab(r"\nCycles:\s*(\d+)", text, float, 0.0),
        arch_util=_grab(rf"\nUtilization:\s*{_NUM}", text, float, 0.0),
        level_rows=tuple(_weight_level_rows(text, weight_levels or (weight_level,))),
        fingerprint=stats_path.parent.parent.name,
        all_rows=tuple(_all_level_rows(text)),
        dram_loops=_dram_loops(stats_path.parent / "timeloop-mapper.map.txt"),
    )


def _compute_instances(text):
    """`(utilized, declared)` arithmetic instances -- the PE count.

    Timeloop's Level 0 is always the arithmetic level. Its SPECS carry
    `Instances : 256 (16*16)` and its STATS `Utilized instances : 16`, and BOTH
    belong in the table: the ratio is how much of the array the mapping used,
    which is `simple_weight_stationary`'s open defect (16 of 256, FINDINGS 8)
    and is the confound that withdrew the first Task 4 pass.
    """
    for part in text.split("=== "):
        if "STATS" not in part or "Compute energy" not in part:
            continue
        return (_grab(r"Utilized instances\s*:\s*(\d+)", part, int, 0),
                _grab(r"\n\s+Instances\s*:\s*(\d+)", part, int, 0))
    return (0, 0)


def _weight_level_rows(text, weight_levels):
    """Per-level geometry, occupancy, energy and instances for every weight level.

    THE GEOMETRY IS READ FROM THE STATS FILE, NOT FROM THE YAML. `Word bits`
    is the level's `datawidth:` as the mapper actually saw it and `Block size`
    is the `width/datawidth` Timeloop derived from it, so `width` and `depth`
    come back from the run that produced the number rather than from a file
    that may since have been edited. That is what makes prompt_2's fairness
    rule -- both arms declare the SAME width and depth -- checkable on the
    ROWS, not only on the configuration.

    `energy_pJ` is `Energy (total)` out of the level's own `Weights` block. It
    is a PARSE, not a model: prompt_2 asks which memory to shrink, and only a
    per-level number answers that. `total_pJ` alone cannot.
    """
    rows = []
    for level in weight_levels:
        b = _level_block(text, level)
        if not b:
            continue
        bw = _dataspace_block(b, "Weights")
        if not bw:
            continue
        # SPECS: `Word bits` = datawidth, `Block size` = width/datawidth.
        word_bits = _grab(r"Word bits\s*:\s*(\d+)", b, int, 0)
        block = _grab(r"Block size\s*:\s*(\d+)", b, int, 0)
        capacity = _grab(r"Effective size\s*:\s*(\d+)", b, int, 0)
        rows.append({
            "level": level,
            "capacity": capacity,
            "residency": _grab(r"Utilized capacity\s*:\s*(\d+)", bw, int, 0),
            "instances": _grab(r"Utilized instances \(max\)\s*:\s*(\d+)", bw, int, 1),
            #: the geometry that produced the row
            "datawidth": word_bits,
            "weights_per_word": block,
            "width": (word_bits * block) if (word_bits and block) else 0,
            "declared_depth": (capacity // block) if block else 0,
            "vector_access_pJ": _grab(rf"Vector access energy\s*:\s*{_NUM}", b,
                                      float, 0.0),
            #: THE ENERGY THIS LEVEL SPENT ON WEIGHTS. Parsed, not modelled.
            "energy_pJ": _grab(rf"Energy \(total\)\s*:\s*{_NUM}", bw, float, 0.0),
            "reads": _grab(rf"Scalar reads \(per-instance\)\s*:\s*{_NUM}", bw,
                           float, 0.0),
            "fills": _grab(rf"Scalar fills \(per-instance\)\s*:\s*{_NUM}", bw,
                           float, 0.0),
        })
    return rows


#: The loop dimensions each dataspace is indexed by. A DRAM-level loop over a
#: dimension a dataspace does NOT index is a loop that dataspace must be
#: re-read across -- which is what `refetch` measures, per dataspace.
DATASPACE_DIMS = {
    "Weights": {"C", "M", "R", "S"},
    "Inputs":  {"N", "C", "P", "Q", "R", "S"},
    "Outputs": {"N", "M", "P", "Q"},
}


def _dram_loops(map_path):
    """`{dim: factor}` for the loops Timeloop put at the DRAM level.

    Read from `timeloop-mapper.map.txt`, whose first block is DRAM's own
    temporal loops. These are what set every dataspace's refetch: a loop over
    a dimension a dataspace does not index forces that dataspace to be
    re-read (or, for Outputs, written out and read back to accumulate) once
    per iteration.
    """
    p = pathlib.Path(map_path)
    if not p.is_file():
        return {}
    text = p.read_text()
    head = text.split("---", 1)[-1]
    head = head.split("\n\n", 1)[0] if "\n\n" in head else head
    out = {}
    for dim, hi in re.findall(r"for (\w+) in \[0:(\d+)\)", head):
        out[dim] = out.get(dim, 1) * int(hi)
    return out


def reduction_factor(dram_loops, dataspace):
    """How many times `dataspace` is re-traversed at the DRAM level.

    The product of the DRAM-level loop factors over dimensions this dataspace
    is NOT indexed by. For Outputs those are REDUCTION loops -- each one
    forces the partial sums to be written out and read back to accumulate --
    and on this design that product is exactly what the weight-buffer size
    controls, because a smaller weight buffer forces C to be chopped finer.
    """
    f = 1
    for dim, n in dram_loops.items():
        if dim.upper() not in DATASPACE_DIMS.get(dataspace, set()):
            f *= n
    return f


def _all_level_rows(text):
    """Every (level, dataspace) pair in the stats file, with its own energy.

    WHY THIS EXISTS. The weight-only view (`_weight_level_rows`) was written
    for prompt_2's hypothesis -- that reconstruction's extra capacity would
    show up as less WEIGHT refetch. Measured 2026-09-10 on the reconfigured
    Eyeriss v1 it does not, and cannot: `C` is the only DRAM-level loop and
    `C` indexes weights, so every weight is fetched exactly once and weight
    refetch sits at its 1.000 floor on BOTH arms. The saving is real and
    large, and it lands in the OUTPUT path -- a smaller weight buffer forces
    `C` into more chunks, and every chunk boundary is a partial-sum
    round-trip. Reporting weights only showed the flat 2x packing discount
    and hid 97 % of the effect.
    """
    rows = []
    for part in text.split("=== "):
        if "STATS" not in part or "Compute energy" in part:
            continue
        level = part.split(" ===")[0]
        cap = _grab(r"Effective size\s*:\s*(\d+)", part, int, 0)
        word_bits = _grab(r"Word bits\s*:\s*(\d+)", part, int, 0)
        block = _grab(r"Block size\s*:\s*(\d+)", part, int, 0)
        for space in ("Weights", "Inputs", "Outputs"):
            b = _dataspace_block(part, space)
            if not b:
                continue
            rows.append({
                "level": level, "dataspace": space,
                "capacity": cap, "datawidth": word_bits,
                "values_per_word": block,
                "declared_depth": (cap // block) if block else 0,
                "residency": _grab(r"Utilized capacity\s*:\s*(\d+)", b, int, 0),
                "instances": _grab(r"Utilized instances \(max\)\s*:\s*(\d+)",
                                   b, int, 1),
                "partition": _grab(r"Partition size\s*:\s*(\d+)", b, int, 0),
                "reads": _grab(rf"Scalar reads \(per-instance\)\s*:\s*{_NUM}",
                               b, float, 0.0),
                "fills": _grab(rf"Scalar fills \(per-instance\)\s*:\s*{_NUM}",
                               b, float, 0.0),
                "updates": _grab(rf"Scalar updates \(per-instance\)\s*:\s*{_NUM}",
                                 b, float, 0.0),
                "energy_pJ": _grab(rf"Energy \(total\)\s*:\s*{_NUM}", b,
                                   float, 0.0),
            })
    return rows


def weight_levels_of(arch, cfg):
    """EVERY weight-carrying level the dilation rewrites, outer to inner.

    A design may have more than one, and which of them is full is the whole
    question: `eyeriss_like_wglb` dilates both its `filter_glb` (8,192 weights,
    75 % full on layer4.0.conv2) and its PE `weights_spad` (448, 21 % full), and
    reporting only the innermost would have said 21 % about a design whose
    weight GLB is three-quarters used. A `depth: 1` latch is not a candidate --
    the dilation does not scale it either.
    """
    rows = [r for r in archmod.weight_capacity_levels(arch, cfg) if not r["latch"]]
    if not rows:
        raise ValueError(f"{arch}: no weight-carrying storage level to dilate")
    return [r["level"] for r in rows]


def weight_level_of(arch, cfg):
    """The INNERMOST dilated weight level -- the one refetch and G_rec read."""
    return weight_levels_of(arch, cfg)[-1]


def cache_for(cfg, arch, scale):
    """The mapper cache directory for `arch` at weight-capacity `scale`."""
    # `config.py` is the only reader of os.environ, so a second capacity is
    # reached by replacing the field, never by re-reading the environment.
    sub = dataclasses.replace(cfg, weight_capacity_scale=float(scale))
    variant = archmod.effective_variant(arch, sub)
    fp = archmod.arch_fingerprint(arch, sub)
    # `create=False`: this module maps nothing and must leave no trace. Without
    # it, asking about a capacity that was never mapped CREATED its cache
    # directory, and the tree then advertised mappings that do not exist.
    return pathlib.Path(pathsmod.Results(sub).mapper_cache(
        arch, variant, fp, create=False)), sub


def load(cfg, arch, scale, shapes):
    """`{shape: MappedLayer}` for one design at one capacity scale."""
    root, sub = cache_for(cfg, arch, scale)
    levels = weight_levels_of(arch, sub)
    level = levels[-1]
    out = {}
    for shape in shapes:
        m = read_mapped_layer(root / shape / "timeloop-mapper.stats.txt", level,
                              levels)
        if m is not None:
            out[shape] = m
    return out, root


# ===========================================================================
#  the two reports
# ===========================================================================
def survey(cfg, arch_list, scale=1.0, model=None, mac_pj=None):
    """Per-layer refetch and residency at ONE capacity -- how to choose layers.

    A layer worth sweeping is one that BOTH refetches (`refetch > 1`, so there
    is something to remove) and fills its weight buffer (`residency` near
    `capacity`, so capacity is plausibly what is binding). A layer with a high
    refetch and a nearly empty buffer is refetching for some other reason and
    a dilation will not touch it -- which is a result, but a cheap one.
    """
    model = model or cfg.models[0]
    models, _ = workloadsmod.load_workload(cfg)
    layers = models[model]
    lines = []
    for arch in arch_list:
        shapes, seen = [], {}
        for l in layers:
            if l.shape_name not in seen:
                seen[l.shape_name] = l
                shapes.append(l.shape_name)
        got, root = load(cfg, arch, scale, shapes)
        lines.append(f"\n{arch}   capacity x{scale:g}   {len(got)}/{len(shapes)} shapes cached")
        lines.append(f"  {root}")
        lines.append(f"  {'layer':<24} {'shape':<34} {'weights':>10} {'DRAM rd':>12} "
                     f"{'refetch':>8} {'resid':>7} {'cap':>6} {'fill%':>6}")
        tot_w = tot_r = 0.0
        for l in layers:
            m = got.get(l.shape_name)
            if m is None:
                continue
            tot_w += m.weights * (l.count or 1)
            tot_r += m.dram_weight_reads * (l.count or 1)
            fill = 100.0 * m.residency / m.capacity if m.capacity else 0.0
            lines.append(f"  {l.name:<24} {l.shape_name:<34} {m.weights:>10,} "
                         f"{m.dram_weight_reads:>12,.0f} {m.refetch:>8.3f} "
                         f"{m.residency:>7,} {m.capacity:>6,} {fill:>5.1f}%")
        if tot_w:
            lines.append(f"  {'MODEL (layer-weighted)':<24} {'':<34} {tot_w:>10,.0f} "
                         f"{tot_r:>12,.0f} {tot_r / tot_w:>8.3f}")
    return "\n".join(lines)


@dataclasses.dataclass
class DilationRow:
    """One (design, layer, reference capacity) comparison."""
    arch: str
    layer: str
    shape: str
    ref_scale: float
    dil_scale: float
    ref: MappedLayer
    dil: MappedLayer
    k_over_n: float
    mac_pj: float | None

    # ---- the validation itself --------------------------------------------
    @property
    def reads_removed(self):
        return self.ref.dram_weight_reads - self.dil.dram_weight_reads

    @property
    def refetch_ratio(self):
        """`R_recon / R_emb`. The first-order model of FINDINGS section 9
        item 0 predicts `(R_emb - 1) x K/N + 1` over `R_emb`."""
        return self.dil.refetch / self.ref.refetch if self.ref.refetch else 0.0

    @property
    def predicted_refetch(self):
        """What section 9 item 0 assumed: excess refetch scales with 1/capacity."""
        return (self.ref.refetch - 1.0) * self.k_over_n + 1.0

    # ---- what it is worth --------------------------------------------------
    @property
    def dram_pj_fixed_mapping(self):
        """Task 3's saving on this layer: narrowing only, both arms issuing
        the SAME reads. Efficiency `(1 - K/N)` = 0.381 since the f_if split
        was removed on 2026-09-09 (it was f_if x (1 - K/N) = 0.152)."""
        return self.ref.dram_weight_pj * (1.0 - self.k_over_n)

    @property
    def dram_pj_task4(self):
        """Task 4's DRAM saving: the reads the dilated mapping never issues
        (whole per-access energy, efficiency 1.0) plus the K/N narrowing of the
        reads it does issue."""
        never_issued = self.reads_removed * self.ref.dram_pj_per_read
        issued = self.dil.dram_weight_pj * (1.0 - self.k_over_n)
        return never_issued + issued

    @property
    def efficiency(self):
        """Saved pJ per pJ of reference DRAM weight energy. (1 - K/N) = 0.381
        is the fixed-mapping ceiling; above it is capacity dilation."""
        return (self.dram_pj_task4 / self.ref.dram_weight_pj
                if self.ref.dram_weight_pj else 0.0)

    def denominator(self):
        return self.ref.total_pj_at(self.mac_pj)

    @property
    def pct_fixed(self):
        d = self.denominator()
        return 100.0 * self.dram_pj_fixed_mapping / d if d else 0.0

    @property
    def pct_task4(self):
        d = self.denominator()
        return 100.0 * self.dram_pj_task4 / d if d else 0.0

    @property
    def ert_ratio(self):
        """How much dearer Accelergy priced the DILATED weight level per access.

        The reconstruction arm's array is NOT larger -- it is the same array
        holding narrower values -- so any ratio above 1.0 is silicon the design
        does not have, and it biases the SEARCH against using the capacity,
        i.e. against the hypothesis. Reported so the bias is visible; corrected
        in the energy account by `recon.capacity_dilation_correction()`.
        """
        a = _ert_read_pj(self.ref.stats_path, self.ref.weight_level)
        b = _ert_read_pj(self.dil.stats_path, self.dil.weight_level)
        return (b / a) if (a and b) else float("nan")


def _ert_read_pj(stats_path, level):
    """Per-read energy Accelergy assigned that level, from the ERT summary.

    `recon.storage_access_pj()` is the one reader of this file for the energy
    model; this is the same lookup for the REPORT, kept here so the diff tool
    stays importable without the placement machinery.
    """
    f = pathlib.Path(stats_path).parent / "timeloop-mapper.ERT_summary.yaml"
    if not f.is_file():
        return None
    try:
        import yaml
        blob = yaml.safe_load(f.read_text())
    except Exception:
        return None
    for entry in (blob.get("ERT_summary", {}).get("table_summary") or []):
        # Accelergy writes `system_top_level.weights_spad[1..168]`, and the
        # instance range contains dots of its own -- strip it BEFORE splitting.
        bare = str(entry.get("name", "")).split("[")[0].split(".")[-1]
        if bare != level:
            continue
        for a in (entry.get("actions") or []):
            if str(a.get("name")) == "read":
                return float(a.get("energy", 0.0)) or None
    return None


def binding_level(stats_path, weight_levels):
    """Which PE-level buffer was FULL in this mapping, and how full the weight one was.

    THE QUESTION THIS ANSWERS IS THE WHOLE OF STEP 1. A dilation enlarges the
    WEIGHT buffer. If some OTHER buffer is what the mapping ran out of, the
    extra weight room relieves a constraint that was not binding and the mapper
    returns the same loop nest -- which is measured, not supposed: on
    `eyeriss_like`'s layer2.0.conv1 the declared and dilated nests are
    byte-identical, `ifmap_spad` is at 24/24 and `weights_spad` goes from
    96/448 to 96/724.

    Returns `(name, fill, weight_fill, weight_name)`: `fill` is the highest
    utilization of any level the dilation does NOT touch, and `weight_fill` is
    the highest of the ones it does. `weight_fill` well below `fill` is the
    signature of a dilation that cannot do anything -- it is enlarging a buffer
    the mapping had room to spare in, while the one it ran out of is untouched.
    """
    stats_path = pathlib.Path(stats_path)
    if not stats_path.is_file():
        return ("?", 0.0, 0.0, "-")
    weight_levels = ([weight_levels] if isinstance(weight_levels, str)
                     else list(weight_levels))
    text = stats_path.read_text()
    worst, worst_name = 0.0, "-"
    wfill, wname = 0.0, "-"
    for part in text.split("=== "):
        name = part.split(" ===")[0]
        if "STATS" not in part or name == "DRAM":
            continue
        size = _grab(r"Effective size\s*:\s*(\d+)", part, int, 0)
        if not size:
            continue
        for space in ("Weights", "Inputs", "Outputs"):
            b = _dataspace_block(part, space)
            if not b:
                continue
            used = _grab(r"Utilized capacity\s*:\s*(\d+)", b, int, 0)
            fill = used / size
            if name in weight_levels and space == "Weights":
                if fill > wfill:
                    wfill, wname = fill, name
            elif fill > worst:
                worst, worst_name = fill, f"{name}/{space[0]}"
    return (worst_name, worst, wfill, wname)


def sweep(cfg, arch_list, layer_names, refs, model=None, k_over_n=None,
          mac_pj=None):
    """Build every (design, layer, reference capacity) comparison that is cached."""
    model = model or cfg.models[0]
    models, _ = workloadsmod.load_workload(cfg)
    by_name = {l.name: l for l in models[model]}
    missing = [n for n in layer_names if n not in by_name]
    if missing:
        raise SystemExit(f"dilation: no such layer in {model}: {', '.join(missing)}")
    k_over_n = k_over_n if k_over_n is not None else cfg.code_k / cfg.code_n

    rows, gaps = [], []
    for arch in arch_list:
        shapes = [by_name[n].shape_name for n in layer_names]
        for s in refs:
            d = round(s / k_over_n, 4)
            ref, ref_root = load(cfg, arch, s, shapes)
            dil, dil_root = load(cfg, arch, d, shapes)
            for n in layer_names:
                sh = by_name[n].shape_name
                if sh not in ref or sh not in dil:
                    gaps.append(f"{arch:<26} x{s:<7g} {n:<22} "
                                f"ref={'ok' if sh in ref else 'MISSING'} "
                                f"dil(x{d:g})={'ok' if sh in dil else 'MISSING'}")
                    continue
                rows.append(DilationRow(arch=arch, layer=n, shape=sh, ref_scale=s,
                                        dil_scale=d, ref=ref[sh], dil=dil[sh],
                                        k_over_n=k_over_n, mac_pj=mac_pj))
    return rows, gaps


# ===========================================================================
#  THE TWO GUARDS -- each of them exists because it caught a false positive
# ===========================================================================
def sibling_fingerprints(cfg, arch, scale, shapes):
    """Every `fp-<hash>` under this design's variant slug that has SOLVED one
    of `shapes`, with the one the current configuration resolves to marked.

    WHY THIS IS A GUARD AND NOT A CURIOSITY. `fp-<hash>` is a hash of the
    patched YAML the mapper saw plus the globals and every mapper setting, so
    two fingerprints under one slug are two ARCHITECTURES. A capacity
    comparison drawn across them is not a capacity comparison at all.

    Measured, 2026-09-09: `eyeriss_v2_like` layer2.0.conv1 was reported as
    refetch 14.00x -> 4.00x at x1.6154. The 14.00x was `fp-3eb860ea2b2a`,
    solved 2026-09-08 23:49; the fingerprint the current configuration reads is
    `fp-88656178371f`, solved 2026-09-09 01:33, and it ALREADY refetches 4.00x
    at the declared capacity. The dilation bought nothing; the mapper cache had
    simply been re-fingerprinted between the two runs. The PE count moving
    192 -> 144 across the "pair" was the tell, which is why `PEs used` is
    mandatory in the table.
    """
    root, sub = cache_for(cfg, arch, scale)
    variant_dir = root.parent
    out = []
    if not variant_dir.is_dir():
        return out
    for fp in sorted(variant_dir.glob("fp-*")):
        solved = [s for s in shapes
                  if (fp / s / "timeloop-mapper.stats.txt").is_file()]
        if solved:
            out.append({"fp": fp.name, "current": fp.name == root.name,
                        "shapes": solved, "path": fp})
    return out


def capacity_verdict(ref, dil):
    """Why the DRAM weight reads differ between two capacities -- or do not.

    A dilation is only a capacity effect if the extra room was SPENT. Three
    things can move the reads instead, and all three are named rather than
    averaged in:

    * `PE!=`  the two mappings use different numbers of PEs, so parallelism
              moved with the capacity and nothing can be attributed to either.
              This is FINDINGS 7.7's withdrawal in a column.
    * `PERM?` the reads moved while `weights held` is IDENTICAL at every weight
              level. Nothing was stored that was not stored before, so the
              cause is the loop nest's ORDER, which the search can reach at the
              declared capacity too -- it is search noise, not silicon.
    * `capacity` the reads fell AND `weights held` rose. This is the hypothesis.
    """
    moved = abs(dil.dram_weight_reads - ref.dram_weight_reads) > 0.5
    direction = ("flat" if not moved else
                 "reads-fell" if dil.dram_weight_reads < ref.dram_weight_reads
                 else "reads-rose")
    if (ref.pes_used != dil.pes_used
            or ref.weight_instances != dil.weight_instances):
        # PE!= SUPPRESSES the capacity claim rather than sitting beside it.
        # If parallelism moved, the reads difference is not attributable to the
        # capacity at all -- that is exactly the confound that withdrew the
        # first Task 4 pass, and a row reading `PE!=,capacity` would invite the
        # same mistake a second time. The direction is still reported, so the
        # row keeps its information without making a claim it cannot support.
        return f"PE!=,{direction}"
    if not moved:
        return "flat"
    if ref.weights_held == dil.weights_held:
        # Nothing extra was stored ANYWHERE, so the reads moved for some other
        # reason -- the loop nest's order.
        return f"PERM?,{direction}"
    if dil.dram_weight_reads < ref.dram_weight_reads:
        if dil.weights_held > ref.weights_held:
            return "capacity"
        # Reads FELL while the mapping stored STRICTLY LESS on chip. Extra room
        # cannot be the cause of a saving the arm did not use the room for --
        # it found a different loop nest that both stores less and reads less,
        # which the reference could have found at its own capacity. Measured on
        # simple_weight_stationary (scope=shared) x1 -> x1.6154: reads
        # 147,456 -> 73,728 while `held` FELL 18,816 -> 9,984. Before this
        # branch existed that row was labelled `capacity`, which would have
        # been a fourth false positive.
        return "ORDER?,held-fell"
    return direction


def render(rows, gaps, cfg, k_over_n, mac_pj):
    """The console table the whole exercise exists to produce."""
    out = []
    nk = 1.0 / k_over_n
    out.append("=" * 118)
    out.append("TASK 4 STEP 1 -- CAPACITY DILATION: does N/K more weight room cut DRAM refetch?")
    out.append("=" * 118)
    out.append(f"  code BCH({cfg.code_n},{cfg.code_k})  K/N = {k_over_n:.4f}  "
               f"N/K = {nk:.4f}   objective = {cfg.opt_metric}   "
               f"whole DRAM term x K/N (f_if removed 2026-09-09)"
               if rows else "  (no rows)")
    out.append(f"  MAC denominator = "
               f"{'ERT (per-mapping)' if mac_pj is None else f'{mac_pj} pJ'}"
               f"   scope = {cfg.weight_capacity_scope}")
    out.append("")
    out.append("  ref  = the design at capacity x<ref>, i.e. the embedded/conventional arm")
    out.append("  dil  = the SAME design at x<ref>*N/K, i.e. the reconstruction arm's")
    out.append("         effective capacity for the same silicon")
    out.append("")

    hdr = (f"  {'design':<20} {'layer':<16} {'x ref':>7} {'x dil':>7} "
           f"{'cap r/d':>11} {'resid r/d':>11} "
           f"{'refetch r':>9} {'refetch d':>9} {'pred':>6} "
           f"{'reads cut':>11} {'%':>6}   {'fullest weight lvl':>16} {'binds':>16}")
    out.append(hdr)
    out.append("  " + "-" * (len(hdr) - 2))
    for r in rows:
        cut = r.reads_removed
        pct = 100.0 * cut / r.ref.dram_weight_reads if r.ref.dram_weight_reads else 0.0
        flag = " <-NEG" if cut < 0 else ""
        bname, bfill, wfill, wname = binding_level(r.ref.stats_path,
                                                   r.ref.weight_levels)
        out.append(f"  {r.arch:<20} {r.layer:<16} {r.ref_scale:>7g} {r.dil_scale:>7g} "
                   f"{r.ref.capacity:>5,}/{r.dil.capacity:<5,} "
                   f"{r.ref.residency:>5,}/{r.dil.residency:<5,} "
                   f"{r.ref.refetch:>9.3f} {r.dil.refetch:>9.3f} "
                   f"{r.predicted_refetch:>6.3f} "
                   f"{cut:>11,.0f} {pct:>5.1f}%{flag}   "
                   f"{wname[:10] + f' {100 * wfill:.0f}%':>16} "
                   f"{bname + f' {100 * bfill:.0f}%':>16}")

    out.append("")
    out.append("  WHAT THE DRAM DIFFERENTIAL IS WORTH (this layer's own total energy as denominator)")
    hdr2 = (f"  {'design':<20} {'layer':<16} {'x ref':>7} "
            f"{'DRAM w uJ':>10} {'fixed uJ':>9} {'task4 uJ':>9} "
            f"{'effic':>6} {'fixed %':>8} {'task4 %':>8} {'ERT x':>6}")
    out.append(hdr2)
    out.append("  " + "-" * (len(hdr2) - 2))
    for r in rows:
        out.append(f"  {r.arch:<20} {r.layer:<16} {r.ref_scale:>7g} "
                   f"{r.ref.dram_weight_pj / 1e6:>10.2f} "
                   f"{r.dram_pj_fixed_mapping / 1e6:>9.2f} "
                   f"{r.dram_pj_task4 / 1e6:>9.2f} "
                   f"{r.efficiency:>6.3f} {r.pct_fixed:>7.2f}% {r.pct_task4:>7.2f}% "
                   f"{r.ert_ratio:>6.3f}")
    out.append("")
    fixed_eff = (1.0 - k_over_n) if rows else 0.0
    out.append(f"  efficiency {fixed_eff:.3f} = (1 - K/N) = the Task 3 "
               f"fixed-mapping ceiling at BCH({cfg.code_n},{cfg.code_k}).")
    out.append(f"  Anything above it is capacity dilation: a read never issued removes the")
    out.append(f"  DRAM ARRAY as well as the interface, so its efficiency is 1.0.")
    out.append(f"  The two right-hand columns are the point. The first is the fullest level")
    out.append(f"  the DILATION TOUCHES, the second the fullest level it does not. A")
    out.append(f"  dilation enlarges weight buffers only, so a weight level well below")
    out.append(f"  `binds` means it relieved a")
    out.append(f"  constraint that was not binding -- and the mapper returns the same loop")
    out.append(f"  nest. That is the commonest reason a row reads 0, and it is a property")
    out.append(f"  of the DATAFLOW, not of the code rate.")
    out.append(f"  ERT x is how much dearer Accelergy priced the DILATED weight level per")
    out.append(f"  access. Above 1.0 is silicon the reconstruction arm does not have -- it")
    out.append(f"  biases the SEARCH against the hypothesis and is corrected in the energy")
    out.append(f"  account, never in the mapping.")
    if gaps:
        out.append("")
        out.append("  NOT YET CACHED (these comparisons are absent, not zero):")
        for g in gaps:
            out.append(f"    {g}")
    return "\n".join(out)


# ===========================================================================
#  the utilisation table -- one row per SWEPT CAPACITY, per design, per layer
# ===========================================================================
def utilisation_table(cfg, arch_list, layer_names, scales, model=None,
                      k_over_n=None, mac_pj=None):
    """The table Task 4 is read off. One row per (design, layer, capacity).

    Every column is here because something went wrong without it:

    ``PEs used``      the objective can serialise, and then the capacity and
                      the parallelism move together (FINDINGS 7.7, withdrawn).
                      Printed as used/declared so a design running its array
                      at 6 % says so on the row.
    ``held``          weights actually kept on chip, over EVERY weight level.
                      If this does not move, the capacity was not spent, and a
                      reads difference is a loop-ORDER difference.
    ``room``          what the design could hold, so `fill %` is the fraction
                      of the buffer the mapper chose to use. A dilation of a
                      buffer at 3 % fill cannot do anything.
    ``fullest other`` the fullest level the dilation does NOT touch. A weight
                      level far below it means the dilation relieved a
                      constraint that was not binding.
    ``fp``            the fingerprint the row came from -- two fingerprints
                      under one slug are two architectures.
    """
    model = model or cfg.models[0]
    models, _ = workloadsmod.load_workload(cfg)
    by_name = {l.name: l for l in models[model]}
    missing = [n for n in layer_names if n not in by_name]
    if missing:
        raise SystemExit(f"dilation: no such layer in {model}: {', '.join(missing)}")
    k_over_n = k_over_n if k_over_n is not None else cfg.code_k / cfg.code_n
    shapes = [by_name[n].shape_name for n in layer_names]

    out, gaps, warnings = [], [], []
    nk = 1.0 / k_over_n
    out.append("=" * 172)
    out.append("TASK 4 -- CAPACITY DILATION UTILISATION TABLE   "
               f"code BCH({cfg.code_n},{cfg.code_k})  N/K = {nk:.4f}  "
               f"objective = {cfg.opt_metric}  victory = {cfg.victory}  "
               f"scope = {cfg.weight_capacity_scope}")
    out.append("=" * 172)
    if cfg.opt_metric != "edp":
        out.append(f"  !! OBJECTIVE = {cfg.opt_metric.upper()}, NOT edp. A capacity "
                   f"comparison under a serialising objective is NOT VALID: it")
        out.append("     trades PEs for capacity, so the PE and MACs columns move "
                   "with the capacity and neither is separable. This is")
        out.append("     what withdrew the first Task 4 pass (FINDINGS 7.7). If "
                   "this line is showing and you did not intend it, env.sh")
        out.append("     was probably not sourced -- run through "
                   "`bash hpc/tl.sh python3 -m eccenergy.experiments.dilation`.")
        out.append("")
    out.append("  arm: emb = the embedded/conventional arm at capacity x s.   "
               "rec = the reconstruction arm on the SAME silicon, x s*N/K.")
    out.append("  'held' and 'room' sum EVERY weight-carrying level, times the "
               "instances used.  'PEs' = instances of the innermost weight")
    out.append("  level, so w/PE x PEs = room.  'MACs' = used/declared arithmetic "
               "instances; it exceeds PEs where a PE has several lanes")
    out.append("  (eyeriss_v2_like runs 2 SIMD lanes per PE). Either column "
               "exposes the serialisation confound of FINDINGS 7.7.")
    out.append("")

    hdr = (f"  {'design':<22} {'layer':<15} {'arm':<4} {'x cap':>8} {'w/PE':>7} "
           f"{'PEs':>6} {'MACs':>9} {'room':>11} {'held':>10} {'fill':>6} "
           f"{'DRAM w rd':>11} {'refetch':>8} {'uJ':>9} "
           f"{'fullest other':>22} {'saving vs emb':>14} {'verdict':<16} {'fp':<15}")
    out.append(hdr)
    out.append("  " + "-" * (len(hdr) - 2))

    csv_rows = []
    for arch in arch_list:
        # the fingerprint guard, once per design
        for s in scales:
            sibs = sibling_fingerprints(cfg, arch, s, shapes)
            if len(sibs) > 1:
                cur = [x["fp"] for x in sibs if x["current"]] or ["<none current>"]
                warnings.append(
                    f"{arch} x{s:g}: {len(sibs)} solved fingerprints under one "
                    f"variant slug -- {', '.join(x['fp'] for x in sibs)}; the "
                    f"current configuration reads {cur[0]}. The others are a "
                    f"DIFFERENT architecture and must not be compared against.")
        for layer in layer_names:
            sh = by_name[layer].shape_name
            ref_by_scale = {}
            for s in scales:
                got, root = load(cfg, arch, s, [sh])
                if sh not in got:
                    gaps.append(f"{arch:<22} x{s:<8g} {layer:<16} NOT CACHED")
                    continue
                ref_by_scale[round(s, 4)] = got[sh]
            for s in scales:
                m = ref_by_scale.get(round(s, 4))
                if m is None:
                    continue
                # is this scale the RECON arm of some swept reference?
                base = round(s * k_over_n, 4)
                partner = ref_by_scale.get(base)
                arm = "rec" if partner is not None else "emb"
                saving, verdict = "", ""
                if partner is not None:
                    row = DilationRow(arch=arch, layer=layer, shape=sh,
                                      ref_scale=base, dil_scale=round(s, 4),
                                      ref=partner, dil=m,
                                      k_over_n=k_over_n, mac_pj=mac_pj)
                    saving = (f"{row.dram_pj_task4 / 1e6:.2f}uJ "
                              f"{row.pct_task4:.2f}%")
                    verdict = capacity_verdict(partner, m)
                bname, bfill, _, _ = binding_level(m.stats_path, m.weight_levels)
                out.append(
                    f"  {arch:<22} {layer:<15} {arm:<4} {s:>8g} "
                    f"{m.weights_per_pe:>7,} "
                    f"{m.weight_instances:>6,} "
                    f"{str(m.pes_used) + '/' + str(m.pes_declared):>9} "
                    f"{m.weight_room:>11,} {m.weights_held:>10,} "
                    f"{100 * m.fill:>5.1f}% "
                    f"{m.dram_weight_reads:>11,.0f} {m.refetch:>8.3f} "
                    f"{m.total_pj_at(mac_pj) / 1e6:>9.2f} "
                    f"{bname + ' ' + format(100 * bfill, '.0f') + '%':>22} "
                    f"{saving:>14} {verdict:<16} {m.fingerprint:<15}")
                csv_rows.append({
                    "arch": arch, "layer": layer, "shape": sh, "arm": arm,
                    "scale": s, "weights_per_pe": m.weights_per_pe,
                    "pes_used": m.pes_used, "pes_declared": m.pes_declared,
                    "weight_level_instances": m.weight_instances,
                    "weight_room": m.weight_room, "weights_held": m.weights_held,
                    "fill": f"{m.fill:.6f}", "dram_weight_reads": m.dram_weight_reads,
                    "refetch": f"{m.refetch:.6f}",
                    "total_pJ": f"{m.total_pj_at(mac_pj):.3f}",
                    "cycles": m.cycles, "arch_util_pct": m.arch_util,
                    "fullest_other": bname, "fullest_other_fill": f"{bfill:.4f}",
                    "verdict": verdict, "fingerprint": m.fingerprint,
                    "levels": "|".join(
                        f"{r['level']}:{r['residency']}/{r['capacity']}x{r['instances']}"
                        for r in m.level_rows)})
            out.append("")

    if warnings:
        out.append("  !! FINGERPRINT WARNINGS -- a cross-fingerprint comparison is not a capacity comparison")
        for w in warnings:
            out.append(f"     {w}")
        out.append("")
    out.append("  VERDICT COLUMN. Exactly one verdict per pair, and only `capacity` may be quoted.")
    out.append("    capacity  reads FELL and `held` ROSE with the PE count UNCHANGED. The hypothesis.")
    out.append("    PE!=      the arms differ in PE count, so the reads difference is NOT")
    out.append("              attributable to capacity. This SUPPRESSES the capacity claim rather")
    out.append("              than sitting beside it -- it is the confound that withdrew the first")
    out.append("              Task 4 pass (FINDINGS 7.7), and EDP does NOT prevent it once the")
    out.append("              weight buffer is small: measured 168 -> 84 and 96 -> 112 PEs below")
    out.append("              x0.5 on eyeriss_like. The reads direction is still reported.")
    out.append("    PERM?     reads moved while `held` is IDENTICAL at every weight level. Nothing")
    out.append("              extra was stored, so the cause is loop ORDER, reachable at declared")
    out.append("              capacity -- search noise, not silicon.")
    out.append("    ORDER?    reads FELL while `held` FELL. The arm stored strictly LESS and read")
    out.append("              less, so the extra room is not what bought the saving -- a different")
    out.append("              nest the reference could have found at its own capacity.")
    out.append("    flat      the mapper returned the same DRAM weight traffic.")
    if gaps:
        out.append("")
        out.append("  NOT YET CACHED (absent, not zero):")
        for g in gaps:
            out.append(f"    {g}")
    return "\n".join(out), csv_rows


# ===========================================================================
#  PROMPT_2: THE LEVEL TABLE -- one row per (arm, scale, MEMORY LEVEL)
# ===========================================================================
#: The arms of a prompt_2 pair, and how many MAPPINGS they need.
#:
#: Baseline and Embedded both store 8-bit weights, so they share one hardware
#: YAML and one mapping -- only the EVALUATOR separates them, and it separates
#: them at DRAM alone: baseline keeps its parity beside the data, so its weight
#: traffic inflates by N/K (CLAUDE.md, "The three ECC arms"). Recon is the only
#: arm with a different `datawidth`. That is 2 mapper configs per (width,
#: depth), not 3, and the table says which rows came from which mapping.
ARMS = ("baseline", "embedded", "recon")


def arm_configs(cfg, scale, recon_datawidth, depth_levels=None):
    """`{arm: cfg}` for one depth scale. Baseline and embedded share one.

    The embedded arm is spelled with `weight_datawidth=None`, not `=8`:
    `archs.effective_variant()` drops a datawidth that rewrites nothing, so
    both spellings resolve to the SAME mapper cache -- but only the None
    spelling also reads the cache of a run made before this knob existed.

    `depth_levels` selects the SECOND PASS: the caches where only the named
    level's depth was scaled and the others were held at x1. The joint pass
    (every weight level moved together) locates the zone; only these say WHICH
    level bought it, and the deliverable is a `depth:` per level.
    """
    ref = dataclasses.replace(cfg, weight_depth_scale=round(float(scale), 4),
                              weight_datawidth=None,
                              weight_depth_levels=tuple(depth_levels or ()))
    rec = dataclasses.replace(ref, weight_datawidth=int(recon_datawidth))
    return {"baseline": ref, "embedded": ref, "recon": rec}


def _load_arm(cfg, arch, shape):
    """One `MappedLayer` for one already-resolved configuration."""
    variant = archmod.effective_variant(arch, cfg)
    fp = archmod.arch_fingerprint(arch, cfg)
    root = pathlib.Path(pathsmod.Results(cfg).mapper_cache(
        arch, variant, fp, create=False))
    levels = weight_levels_of(arch, cfg)
    m = read_mapped_layer(root / shape / "timeloop-mapper.stats.txt",
                          levels[-1], levels)
    return m, root


def _levels_of(m):
    """`{level: row}` for every weight level, plus a synthetic `DRAM` row.

    DRAM is in the table because it is where the refetch is PAID, and prompt_2
    asks for `dram_weight_reads` and `refetch` on every row. Its `datawidth`
    stays 8 on both arms by construction (`archs._set_weight_datawidth` refuses
    to touch it -- `recon.py` owns the DRAM K/N term), so its `level_pJ` moves
    ONLY when the mapping moves. That makes the DRAM row the pure
    differential-refetch term with no packing discount mixed into it.
    """
    out = {r["level"]: dict(r, kind="scratchpad" if "spad" in r["level"]
                            else "GLB") for r in m.level_rows}
    out["DRAM"] = {
        "level": "DRAM", "kind": "DRAM",
        "capacity": 0, "residency": 0, "instances": 1,
        "datawidth": 8, "weights_per_word": 0, "width": 0, "declared_depth": 0,
        "vector_access_pJ": m.dram_pj_per_read,
        "energy_pJ": m.dram_weight_pj,
        "reads": m.dram_weight_reads, "fills": 0.0,
    }
    return out


def level_table(cfg, arch_list, layer_names, scales, recon_datawidth,
                model=None, mac_pj=None, depth_levels=None):
    """PROMPT_2's results table: one row per (arm, scale, MEMORY LEVEL).

    THE DELIVERABLE IS NOT A TREND, IT IS A SET OF `depth:` VALUES. The
    question is at what on-chip weight-memory size Recon's margin over
    Embedded PEAKS, and `recon_minus_embedded_pJ` per LEVEL is what answers it:
    `total_pJ` for a whole run cannot say which memory to shrink.

    THE TWO RECON TERMS ARE REPORTED SEPARATELY, and that is the point of the
    per-level shape. The `datawidth` packing discount applies to every on-chip
    weight access regardless of mapping, so Recon wins monotonically at every
    depth even with a byte-identical loop nest; summed with the refetch term it
    would make the table look like a win everywhere and say nothing about which
    memory size matters. So:

        packing_discount_pJ = emb_level_pJ x (1 - dw_rec/dw_emb)
                              what Recon saves on a BYTE-IDENTICAL nest, purely
                              from riding more weights per word. Zero at DRAM,
                              whose datawidth is 8 on both arms.
        refetch_term_pJ     = rec_level_pJ - emb_level_pJ x (dw_rec/dw_emb)
                              what is left once that is taken out: the part
                              attributable to the two arms MAPPING differently.
                              Negative is Recon better.

    and the identity `recon_minus_embedded_pJ == refetch_term_pJ -
    packing_discount_pJ` is asserted on every row, so the split cannot drift
    from the total it decomposes.

    THE FAIRNESS RULE IS CHECKED ON THE ROWS, not only on the configuration:
    both arms must report the same `width` and `declared_depth` at every level
    (`archs.assert_pair_geometry` checks the YAML before the map; this checks
    what the mapper actually saw). A mismatch voids the comparison and is
    raised, not annotated.
    """
    model = model or cfg.models[0]
    models, _ = workloadsmod.load_workload(cfg)
    by_name = {l.name: l for l in models[model]}
    missing = [n for n in layer_names if n not in by_name]
    if missing:
        raise SystemExit(f"dilation: no such layer in {model}: {', '.join(missing)}")
    k_over_n = cfg.code_k / cfg.code_n
    nk = 1.0 / k_over_n

    out, gaps, warnings, csv_rows = [], [], [], []
    fingerprint_tag = f"fp-{cfg.fingerprint()}"
    out.append("=" * 200)
    out.append("PROMPT_2 -- WEIGHT-MEMORY DEPTH SWEEP: one row per (arm, scale, MEMORY LEVEL)   "
               f"code BCH({cfg.code_n},{cfg.code_k})  K/N = {k_over_n:.4f}  N/K = {nk:.4f}  "
               f"recon datawidth = {recon_datawidth}b")
    pass_name = ("JOINT (every weight level moved together)" if not depth_levels
                 else f"PER-LEVEL ({'+'.join(depth_levels)} swept, "
                      f"the other weight level(s) held at x1)")
    out.append(f"  depth pass = {pass_name}")
    out.append(f"  objective = {cfg.opt_metric}   victory = {cfg.victory}   "
               f"algorithm = {cfg.mapper_algorithm}   threads = {cfg.mapper_threads}   "
               f"packing = {cfg.recon_packing}   MAC = "
               f"{'ERT (per-mapping)' if mac_pj is None else f'{mac_pj} pJ'}")
    out.append("=" * 200)
    if cfg.opt_metric != "edp":
        out.append(f"  !! OBJECTIVE = {cfg.opt_metric.upper()}, NOT edp. It serialises and "
                   f"trades PEs for capacity, which is what withdrew FINDINGS 7.7.")
        out.append("     If you did not intend it, env.sh was probably not sourced -- run "
                   "through `bash hpc/tl.sh python3 -m eccenergy.experiments.dilation`.")
        out.append("")
    narrowing = reconmod.onchip_narrowing_audit(cfg)
    if not narrowing["ok"]:
        out.append(f"  !! {narrowing['problem']}")
        out.append("")
    out.append("  arm: baseline and embedded SHARE ONE MAPPING (both store 8-bit weights); only the")
    out.append("       evaluator separates them, at DRAM, where baseline's weight traffic inflates by N/K.")
    out.append("       recon is the same silicon at datawidth "
               f"{recon_datawidth}b -- same width, same depth, more weights per word.")
    out.append("  the answer column is `rec-emb pJ`: the margin AT THIS LEVEL AND SCALE. It is split into")
    out.append("       `pack` (the flat packing discount, present at every depth even on an identical nest)")
    out.append("       and `refetch` (what the two arms MAPPING differently contributed). rec-emb = refetch - pack.")
    out.append("")

    hdr = (f"  {'design':<20} {'layer':<15} {'memory':<14} {'kind':<10} {'arm':<9} "
           f"{'scale':>7} {'depth':>7} {'width':>6} {'dw':>4} {'w/word':>7} "
           f"{'room':>10} {'held':>9} {'fill':>6} "
           f"{'DRAM w rd':>12} {'refetch':>8} {'level pJ':>13} {'total uJ':>9} "
           f"{'PEs':>9} {'pack pJ':>12} {'refetch pJ':>13} {'rec-emb pJ':>13} "
           f"{'verdict':<16} {'fp':<15}")
    out.append(hdr)
    out.append("  " + "-" * (len(hdr) - 2))

    for arch in arch_list:
        for layer in layer_names:
            sh = by_name[layer].shape_name
            for s in scales:
                cfgs = arm_configs(cfg, s, recon_datawidth, depth_levels)
                emb, emb_root = _load_arm(cfgs["embedded"], arch, sh)
                rec, rec_root = _load_arm(cfgs["recon"], arch, sh)
                if emb is None:
                    gaps.append(f"{arch:<20} x{s:<8g} {layer:<16} embedded "
                                f"NOT CACHED  ({emb_root})")
                if rec is None:
                    gaps.append(f"{arch:<20} x{s:<8g} {layer:<16} recon    "
                                f"NOT CACHED  ({rec_root})")
                if emb is None or rec is None:
                    continue

                elev, rlev = _levels_of(emb), _levels_of(rec)
                # THE FAIRNESS RULE, on the rows the mapper produced.
                for name in sorted(set(elev) & set(rlev)):
                    if name == "DRAM":
                        continue
                    for field in ("width", "declared_depth"):
                        if elev[name][field] != rlev[name][field]:
                            raise ValueError(
                                f"{arch} {layer} x{s:g}: {name}.{field} differs "
                                f"between the arms (embedded="
                                f"{elev[name][field]}, recon={rlev[name][field]}). "
                                f"prompt_2's fairness rule is that both arms "
                                f"declare the SAME silicon and differ only in "
                                f"datawidth; this comparison is VOID.")
                verdict = capacity_verdict(emb, rec)

                order = [r["level"] for r in emb.level_rows] + ["DRAM"]
                for name in order:
                    e, r = elev.get(name), rlev.get(name)
                    if e is None or r is None:
                        continue
                    ratio = (r["datawidth"] / e["datawidth"]
                             if e["datawidth"] else 1.0)
                    pack = e["energy_pJ"] * (1.0 - ratio)
                    refet = r["energy_pJ"] - e["energy_pJ"] * ratio
                    margin = r["energy_pJ"] - e["energy_pJ"]
                    # The split must equal the total it decomposes. Floating
                    # point only -- these are the same three numbers.
                    assert abs((refet - pack) - margin) <= 1e-6 * max(
                        1.0, abs(margin)), (name, margin, refet, pack)

                    for arm in ARMS:
                        m = rec if arm == "recon" else emb
                        lv = r if arm == "recon" else e
                        room = lv["capacity"] * lv["instances"]
                        held = lv["residency"] * lv["instances"]
                        # BASELINE = EMBEDDED's mapping, with its DRAM weight
                        # traffic inflated by N/K: its parity sits beside the
                        # data in DRAM, so the codeword is n bits where
                        # embedded's payload is k. Nothing on chip moves
                        # (ECC_BASELINE_INFLATES_ONCHIP=0), so only the DRAM
                        # row differs -- which is exactly what the arm IS.
                        infl = nk if (arm == "baseline" and name == "DRAM") else 1.0
                        dram_rd = m.dram_weight_reads * (nk if arm == "baseline" else 1.0)
                        total = m.total_pj_at(mac_pj) + (
                            (nk - 1.0) * m.dram_weight_pj if arm == "baseline" else 0.0)
                        row = {
                            "memory": name, "kind": lv["kind"],
                            "scale": f"{s:g}", "arm": arm,
                            "arch": arch, "layer": layer, "shape": sh,
                            "declared_depth": lv["declared_depth"],
                            "width": lv["width"], "datawidth": lv["datawidth"],
                            "weights_per_word": lv["weights_per_word"],
                            "room": room, "weights_held": held,
                            "fill_pct": f"{100.0 * held / room:.2f}" if room else "",
                            "refetch": f"{m.refetch * (nk if arm == 'baseline' else 1.0):.6f}",
                            "dram_weight_reads": f"{dram_rd:.0f}",
                            "level_pJ": f"{lv['energy_pJ'] * infl:.3f}",
                            "total_uJ": f"{total / 1e6:.4f}",
                            "pes_used": m.pes_used,
                            "pes_declared": m.pes_declared,
                            "cycles": f"{m.cycles:.0f}",
                            "level_reads": f"{lv['reads']:.0f}",
                            "level_fills": f"{lv['fills']:.0f}",
                            "vector_access_pJ": f"{lv['vector_access_pJ']:.5f}",
                            # the three energy columns the decision is read off
                            "packing_discount_pJ":
                                f"{pack:.3f}" if arm == "recon" else "",
                            "refetch_term_pJ":
                                f"{refet:.3f}" if arm == "recon" else "",
                            "recon_minus_embedded_pJ":
                                f"{margin:.3f}" if arm == "recon" else "",
                            "verdict": verdict if arm == "recon" else "",
                            "mapping": ("recon" if arm == "recon"
                                        else "baseline+embedded (shared)"),
                            "depth_pass": ("joint" if not depth_levels
                                           else "+".join(depth_levels)),
                            "fingerprint": m.fingerprint,
                            "run_fingerprint": fingerprint_tag,
                        }
                        csv_rows.append(row)
                        if arm == "baseline":
                            continue          # console: two mappings, two rows
                        out.append(
                            f"  {arch:<20} {layer:<15} {name:<14} {lv['kind']:<10} "
                            f"{arm:<9} {s:>7g} {lv['declared_depth']:>7,} "
                            f"{lv['width']:>6} {lv['datawidth']:>4} "
                            f"{lv['weights_per_word']:>7} "
                            f"{room:>10,} {held:>9,} "
                            f"{(100.0 * held / room if room else 0):>5.1f}% "
                            f"{dram_rd:>12,.0f} {m.refetch:>8.3f} "
                            f"{lv['energy_pJ']:>13,.0f} {total / 1e6:>9.2f} "
                            f"{str(m.pes_used) + '/' + str(m.pes_declared):>9} "
                            + (f"{pack:>12,.0f} {refet:>13,.0f} {margin:>13,.0f} "
                               f"{verdict:<16} " if arm == "recon"
                               else f"{'':>12} {'':>13} {'':>13} {'':<16} ")
                            + f"{m.fingerprint:<15}")
                out.append("")

            for s in scales:
                sibs = sibling_fingerprints(
                    dataclasses.replace(cfg, weight_depth_scale=round(float(s), 4),
                                        weight_datawidth=None,
                                        weight_depth_levels=tuple(depth_levels or ())),
                    arch, 1.0, [sh])
                if len(sibs) > 1:
                    cur = [x["fp"] for x in sibs if x["current"]] or ["<none current>"]
                    warnings.append(
                        f"{arch} x{s:g} embedded: {len(sibs)} solved fingerprints "
                        f"under one variant slug -- "
                        f"{', '.join(x['fp'] for x in sibs)}; the current "
                        f"configuration reads {cur[0]}. The others are a "
                        f"DIFFERENT architecture and must not be compared against.")

    if warnings:
        out.append("  !! FINGERPRINT WARNINGS -- a cross-fingerprint comparison is not a capacity comparison")
        for w in warnings:
            out.append(f"     {w}")
        out.append("")
    out.append("  VERDICT COLUMN -- one verdict per (scale, layer) pair, shown on the recon rows.")
    out.append("    capacity  DRAM weight reads FELL and `held` ROSE with the PE count UNCHANGED. The")
    out.append("              hypothesis, and the ONLY label that may be quoted as one.")
    out.append("    PE!=      the arms differ in PE count, so the reads difference is NOT attributable to")
    out.append("              capacity. SUPPRESSES the capacity claim (FINDINGS 7.7).")
    out.append("    PERM?     reads moved while `held` is IDENTICAL at every weight level -- loop ORDER.")
    out.append("    ORDER?    reads FELL while `held` FELL -- the arm stored strictly less and read less.")
    out.append("    flat      the mapper returned the same DRAM weight traffic.")
    out.append("")
    out.append("  `weights held` IS THE COLUMN THAT DECIDES WHETHER ANYTHING HAPPENED. If DRAM reads move")
    out.append("  while `held` does not RISE, it was not capacity (prompt_2, item 5).")
    if gaps:
        out.append("")
        out.append("  NOT YET CACHED (absent, not zero):")
        for g in gaps:
            out.append(f"    {g}")
    return "\n".join(out), csv_rows


# ===========================================================================
#  PROMPT_2: THE CONVERGENCE GATE
# ===========================================================================
def convergence_gate(cfg, arch_list, layer_names, victories, scales,
                     recon_datawidth, model=None, mac_pj=None):
    """Is the mapper search converged enough to RESOLVE the effect being claimed?

    CONVERGENCE IS A GATE, NOT A CAVEAT. Measured 2026-09-10 on
    `simple_weight_stationary` layer2.0.conv1: at victory 2000 the
    reconstruction arm beat its embedded reference by 9.1 uJ -- the best result
    in that whole study -- and at victory 10000 the SAME pair REVERSED, recon
    5.4 uJ WORSE at 4x the refetch. Energy fell everywhere (a better search)
    but the ORDERING BETWEEN THE ARMS FLIPPED. The sweep was measuring the
    search, not the architecture.

    So: map the EMBEDDED arm at each budget, at the LARGEST and the SMALLEST
    depth in the ladder (a budget that converges on a big buffer may not on a
    small one), and report

      * `residual` -- the relative movement in total energy between one budget
        and the next. Converged means the LAST TWO agree within a margin you
        STATE.
      * `effect` -- the Recon-vs-Embedded margin actually being claimed at that
        depth, read off the same caches.

    THE GATE PASSES ONLY IF THE LAST RESIDUAL IS SMALLER THAN THE EFFECT. If
    the search noise is larger than the signal, no conclusion survives at any
    budget and the instrument cannot resolve the question -- which is what the
    2026-09-10 run found on the OLD geometry, with a minimum residual of 9.04 %
    against a 2-12 % effect.

    A TWO-POINT AGREEMENT TEST IS UNSOUND ON ITS OWN and this does not rely on
    one: `random_pruned` is DETERMINISTIC, so a bigger budget walks the SAME
    sequence further -- it PLATEAUS and then JUMPS. Victory 10000 and 20000
    returned a BIT-IDENTICAL 198.70 uJ on one arm and 50000 then moved it to
    177.40. Three points are the minimum, and the table prints every one so a
    plateau is visible as a plateau rather than read as convergence.
    """
    model = model or cfg.models[0]
    models, _ = workloadsmod.load_workload(cfg)
    by_name = {l.name: l for l in models[model]}
    missing = [n for n in layer_names if n not in by_name]
    if missing:
        raise SystemExit(f"dilation: no such layer in {model}: {', '.join(missing)}")

    out, rows = [], []
    out.append("=" * 150)
    out.append("PROMPT_2 -- THE CONVERGENCE GATE. Run it BEFORE quoting any number.")
    out.append(f"  EMBEDDED arm only, {cfg.mapper_algorithm}, "
               f"objective {cfg.opt_metric}, {cfg.mapper_threads} threads. "
               f"The sweep itself runs at victory {cfg.victory}.")
    out.append("=" * 150)
    out.append("  Converged means the LAST TWO budgets agree within a margin you STATE,")
    out.append("  and that margin must be SMALLER than the Recon-vs-Embedded effect claimed.")
    out.append("  A plateau is NOT convergence: random_pruned is deterministic, so a bigger")
    out.append("  budget walks the same sequence further -- it plateaus, then jumps.")
    out.append("")
    hdr = (f"  {'design':<20} {'layer':<15} {'scale':>7} {'victory':>8} "
           f"{'PEs':>9} {'refetch':>8} {'total uJ':>10} {'residual':>10} "
           f"{'|effect| uJ':>12} {'effect %':>9} {'gate':<26}")
    out.append(hdr)
    out.append("  " + "-" * (len(hdr) - 2))

    for arch in arch_list:
        for layer in layer_names:
            sh = by_name[layer].shape_name
            for s in scales:
                prev = None
                # the effect this depth is being asked to resolve
                cfgs = arm_configs(cfg, s, recon_datawidth)
                e_at, _ = _load_arm(cfgs["embedded"], arch, sh)
                r_at, _ = _load_arm(cfgs["recon"], arch, sh)
                effect = effect_pct = None
                if e_at is not None and r_at is not None:
                    effect = (r_at.total_pj_at(mac_pj)
                              - e_at.total_pj_at(mac_pj)) / 1e6
                    denom = e_at.total_pj_at(mac_pj) / 1e6
                    effect_pct = 100.0 * effect / denom if denom else 0.0
                for v in victories:
                    sub = dataclasses.replace(
                        cfg, victory=int(v),
                        weight_depth_scale=round(float(s), 4),
                        weight_datawidth=None)
                    m, root = _load_arm(sub, arch, sh)
                    if m is None:
                        out.append(f"  {arch:<20} {layer:<15} {s:>7g} {v:>8} "
                                   f"{'':>9} {'':>8} {'NOT CACHED':>10}")
                        rows.append({"arch": arch, "layer": layer, "scale": s,
                                     "victory": v, "cached": 0})
                        prev = None
                        continue
                    tot = m.total_pj_at(mac_pj) / 1e6
                    resid = (abs(tot - prev) / prev * 100.0) if prev else None
                    gate = ""
                    if resid is not None and effect is not None:
                        gate = ("PASS: residual < effect"
                                if abs(resid) < abs(effect_pct or 0.0)
                                else "FAIL: search noise > signal")
                    out.append(
                        f"  {arch:<20} {layer:<15} {s:>7g} {v:>8} "
                        f"{str(m.pes_used) + '/' + str(m.pes_declared):>9} "
                        f"{m.refetch:>8.3f} {tot:>10.2f} "
                        + (f"{resid:>9.2f}%" if resid is not None else f"{'--':>10}")
                        + (f" {effect:>12.2f} {effect_pct:>8.2f}%"
                           if effect is not None else f" {'--':>12} {'--':>9}")
                        + f" {gate:<26}")
                    rows.append({"arch": arch, "layer": layer, "scale": s,
                                 "victory": v, "cached": 1,
                                 "pes_used": m.pes_used,
                                 "refetch": f"{m.refetch:.6f}",
                                 "total_uJ": f"{tot:.4f}",
                                 "residual_pct": ("" if resid is None
                                                  else f"{resid:.4f}"),
                                 "effect_uJ": ("" if effect is None
                                               else f"{effect:.4f}"),
                                 "effect_pct": ("" if effect_pct is None
                                                else f"{effect_pct:.4f}"),
                                 "gate": gate})
                    prev = tot
                out.append("")
    out.append("  `residual` is the movement in TOTAL ENERGY from the previous budget in the list.")
    out.append("  `effect` is the Recon-minus-Embedded total at that depth, both at victory "
               f"{cfg.victory}.")
    out.append("  The gate is one-sided: a PASS says the instrument CAN resolve the effect at that")
    out.append("  depth, not that the effect is real. A FAIL means no conclusion at that depth survives.")
    return "\n".join(out), rows


def dataspace_table(cfg, arch_list, layer_names, scales, recon_datawidth,
                    model=None, mac_pj=None, depth_levels=None):
    """THE WHOLE PICTURE: per DATASPACE, not just weights.

    WHY THIS EXISTS, and it is the most important lesson of the study. The
    weight-only table reports `refetch` as DRAM weight reads over unique
    weights, because prompt_2 assumed the reconstruction arm's extra capacity
    would show up as fewer weight refetches. Measured 2026-09-10 on the
    reconfigured Eyeriss v1, it does not -- and it CANNOT:

        `C` is the only DRAM-level loop, and `C` indexes Weights. So every
        weight is fetched exactly once and weight refetch sits at its 1.000
        FLOOR on BOTH arms, at every depth. A smaller weight buffer cannot
        make it worse.

    What a smaller weight buffer DOES do is force `C` to be chopped into more
    chunks -- 16 for the 8-bit arm against 8 for the 4-bit arm at x0.25. And
    `C` does NOT index Outputs, so every one of those chunks is a REDUCTION
    step: the partial sums for the same outputs are written out to DRAM and
    read back to accumulate, once per chunk. Output refetch is 31.000 against
    15.000, and output energy 153.20 uJ against 85.16.

    So the saving is real, it is large, and it is in the PARTIAL-SUM path.
    Reporting weights alone showed nothing but the flat 2x packing discount
    and hid 97 % of the effect -- a null-looking result on a 31 % win.
    """
    model = model or cfg.models[0]
    models, _ = workloadsmod.load_workload(cfg)
    by_name = {l.name: l for l in models[model]}
    k_over_n = cfg.code_k / cfg.code_n
    out, rows = [], []

    out.append("=" * 150)
    out.append("PER-DATASPACE VIEW -- where the margin ACTUALLY is   "
               f"code BCH({cfg.code_n},{cfg.code_k})  recon datawidth "
               f"{recon_datawidth}b  objective {cfg.opt_metric}")
    out.append("=" * 150)
    out.append("  `x-reduce` = the product of the DRAM-level loop factors over dimensions this")
    out.append("               dataspace is NOT indexed by. For Outputs those are ACCUMULATION")
    out.append("               round-trips: psums written out and read back once per chunk.")
    out.append("  `refetch`  = all DRAM traffic for that dataspace / its unique footprint.")
    out.append("               Weights sit at the 1.000 FLOOR on both arms -- C is the only")
    out.append("               DRAM loop and C indexes weights, so nothing can be re-read.")
    out.append("")
    hdr = (f"  {'layer':<15} {'scale':>7} {'dataspace':<9} "
           f"{'x-reduce e/r':>13} {'refetch e/r':>17} "
           f"{'energy uJ  emb':>15} {'recon':>9} {'delta':>10} {'%':>8}")
    out.append(hdr)
    out.append("  " + "-" * (len(hdr) - 2))

    for arch in arch_list:
        for layer in layer_names:
            sh = by_name[layer].shape_name
            for s in scales:
                cfgs = arm_configs(cfg, s, recon_datawidth, depth_levels)
                emb, _ = _load_arm(cfgs["embedded"], arch, sh)
                rec, _ = _load_arm(cfgs["recon"], arch, sh)
                if emb is None or rec is None:
                    continue
                te = emb.total_pj_at(mac_pj) / 1e6
                tr = rec.total_pj_at(mac_pj) / 1e6
                for space in ("Weights", "Inputs", "Outputs"):
                    ee = emb.energy_by_dataspace(space) / 1e6
                    er = rec.energy_by_dataspace(space) / 1e6
                    fe, fr = emb.dataspace_refetch(space), rec.dataspace_refetch(space)
                    xe = emb.reduction_factor_for(space)
                    xr = rec.reduction_factor_for(space)
                    d = er - ee
                    pct = 100.0 * d / ee if ee else 0.0
                    out.append(
                        f"  {layer:<15} {s:>7g} {space:<9} "
                        f"{('x' + str(xe)):>6} / {('x' + str(xr)):<6} "
                        f"{fe:>8.3f} / {fr:<8.3f} "
                        f"{ee:>15.2f} {er:>9.2f} {d:>10.2f} {pct:>7.1f}%")
                    rows.append({
                        "arch": arch, "layer": layer, "scale": f"{s:g}",
                        "dataspace": space,
                        "reduction_emb": xe, "reduction_rec": xr,
                        "refetch_emb": f"{fe:.6f}", "refetch_rec": f"{fr:.6f}",
                        "energy_uJ_emb": f"{ee:.4f}", "energy_uJ_rec": f"{er:.4f}",
                        "delta_uJ": f"{d:.4f}", "delta_pct": f"{pct:.4f}",
                        "total_uJ_emb": f"{te:.4f}", "total_uJ_rec": f"{tr:.4f}",
                        "pes_emb": emb.pes_used, "pes_rec": rec.pes_used,
                        "cycles_emb": f"{emb.cycles:.0f}",
                        "cycles_rec": f"{rec.cycles:.0f}",
                        "dram_loops_emb": " ".join(
                            f"{k}={v}" for k, v in sorted(emb.dram_loops.items())),
                        "dram_loops_rec": " ".join(
                            f"{k}={v}" for k, v in sorted(rec.dram_loops.items())),
                    })
                out.append(f"  {'':<15} {s:>7g} {'TOTAL':<9} {'':>13} {'':>17} "
                           f"{te:>15.2f} {tr:>9.2f} {tr - te:>10.2f} "
                           f"{100.0 * (tr - te) / te if te else 0:>7.1f}%")
                out.append("")
    out.append("  READ THE `Outputs` ROW. That is where reconstruction's extra weight capacity")
    out.append("  pays on this design: it halves the DRAM-level C chunking, and every chunk")
    out.append("  removed is a partial-sum round-trip that never happens.")
    return "\n".join(out), rows


DATASPACE_CSV_COLUMNS = (
    "layer", "scale", "dataspace", "reduction_emb", "reduction_rec",
    "refetch_emb", "refetch_rec", "energy_uJ_emb", "energy_uJ_rec",
    "delta_uJ", "delta_pct", "total_uJ_emb", "total_uJ_rec",
    "pes_emb", "pes_rec", "cycles_emb", "cycles_rec",
    "dram_loops_emb", "dram_loops_rec", "arch",
)


def dataspace_to_csv(rows, path):
    import csv
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(DATASPACE_CSV_COLUMNS),
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


#: The column order of `results/tables/EyerissV1_mem_arch_sweep.csv`. The first
#: five are the ones the design decision is read off; the rest are what make a
#: row trustworthy (prompt_2, THE RESULTS TABLE).
LEVEL_CSV_COLUMNS = (
    "memory", "scale", "refetch", "weights_held", "level_pJ",
    "arm", "declared_depth", "width", "datawidth", "weights_per_word",
    "room", "fill_pct", "dram_weight_reads", "total_uJ", "pes_used",
    "recon_minus_embedded_pJ", "verdict", "fingerprint",
    # provenance beyond the required set
    "kind", "arch", "layer", "shape", "pes_declared", "cycles",
    "level_reads", "level_fills", "vector_access_pJ",
    "packing_discount_pJ", "refetch_term_pJ", "mapping", "depth_pass",
    "run_fingerprint",
)


def level_table_to_csv(csv_rows, path):
    """Write the level table in prompt_2's declared column order."""
    import csv
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(LEVEL_CSV_COLUMNS),
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(csv_rows)
    return path


def table_to_csv(csv_rows, path):
    import csv
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_rows:
        return path
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    return path


def to_csv(rows, path):
    import csv
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["arch", "layer", "shape", "ref_scale", "dil_scale",
            "ref_capacity", "dil_capacity", "ref_residency", "dil_residency",
            "weights", "ref_dram_reads", "dil_dram_reads", "reads_removed",
            "ref_refetch", "dil_refetch", "predicted_refetch", "refetch_ratio",
            "dram_pj_per_read", "ref_dram_weight_pJ",
            "fixed_mapping_saving_pJ", "task4_saving_pJ", "efficiency",
            "pct_of_layer_fixed", "pct_of_layer_task4", "ert_ratio",
            "ref_total_pJ", "dil_total_pJ"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r.arch, r.layer, r.shape, r.ref_scale, r.dil_scale,
                r.ref.capacity, r.dil.capacity, r.ref.residency, r.dil.residency,
                r.ref.weights, r.ref.dram_weight_reads, r.dil.dram_weight_reads,
                r.reads_removed, f"{r.ref.refetch:.6f}", f"{r.dil.refetch:.6f}",
                f"{r.predicted_refetch:.6f}", f"{r.refetch_ratio:.6f}",
                r.ref.dram_pj_per_read, f"{r.ref.dram_weight_pj:.3f}",
                f"{r.dram_pj_fixed_mapping:.3f}", f"{r.dram_pj_task4:.3f}",
                f"{r.efficiency:.6f}", f"{r.pct_fixed:.6f}", f"{r.pct_task4:.6f}",
                f"{r.ert_ratio:.6f}",
                f"{r.ref.total_pj_at(r.mac_pj):.3f}", f"{r.dil.total_pj_at(r.mac_pj):.3f}"])
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python3 -m eccenergy.experiments.dilation",
        description="Task 4 step 1: diff DRAM weight reads across weight-buffer "
                    "capacities. Reads caches only; maps nothing.")
    ap.add_argument("--survey", action="store_true",
                    help="per-layer refetch and buffer fill at ONE capacity, to "
                         "choose which layers are worth sweeping")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="the capacity --survey reports at (default 1.0)")
    ap.add_argument("--archs", default=None,
                    help="space-separated; default is the configured ECC_ARCHS")
    ap.add_argument("--layers", default=None,
                    help="space-separated layer names; default is ECC_LAYERS")
    ap.add_argument("--refs", default="1.0 0.5 0.25 0.125",
                    help="reference capacities; each is compared against ref*N/K")
    ap.add_argument("--table", action="store_true",
                    help="the UTILISATION table: one row per swept capacity per "
                         "design per layer, with PEs used, weights held, room, "
                         "fill, refetch, energy and the per-pair verdict")
    ap.add_argument("--scales", default=None,
                    help="--table only: the capacities to tabulate. Default is "
                         "every --refs value and its ref*N/K partner, so the "
                         "two arms of each pair sit on adjacent rows.")
    ap.add_argument("--levels", action="store_true",
                    help="PROMPT_2's results table: one row per (arm, scale, "
                         "MEMORY LEVEL), with per-level energy, geometry, "
                         "occupancy and the recon-minus-embedded margin. The "
                         "two arms are the SAME silicon at two datawidths, "
                         "not two depths -- --scales is the depth sweep")
    ap.add_argument("--recon-datawidth", type=int, default=None,
                    help="--levels only: the reconstruction arm's on-chip "
                         "weight datawidth in bits. Default is "
                         "code_widths.declared_datawidth() -- "
                         "round(ECC_WEIGHT_BITS * K/N), which is 4 at "
                         "BCH(63,30) and 7 at BCH(63,57). THE WIDTH TABLE "
                         "(eccenergy/code_widths.py) gives one per code")
    ap.add_argument("--depth-levels", default=None,
                    help="--levels only: read the SECOND PASS instead of the "
                         "joint one -- the caches where only these weight "
                         "level(s) had their depth scaled and the others were "
                         "held at x1. Space-separated. This is what says "
                         "WHICH memory bought the margin, and the deliverable "
                         "is a depth per level")
    ap.add_argument("--gate", action="store_true",
                    help="PROMPT_2's CONVERGENCE GATE: the embedded arm's "
                         "total energy at ECC_DEPTH_SWEEP_GATE_VICTORIES on "
                         "the largest and smallest depth, with the residual "
                         "between budgets set against the effect being "
                         "claimed. Run it BEFORE quoting any number")
    ap.add_argument("--victories", default=None,
                    help="--gate only: the budgets to compare. Default is "
                         "ECC_DEPTH_SWEEP_GATE_VICTORIES")
    ap.add_argument("--csv", default=None, help="also write the rows here")
    a = ap.parse_args(argv)

    cfg = configmod.load_config()
    arch_list = a.archs.split() if a.archs else list(cfg.archs)
    mac_pj = cfg.mac_pj_override

    if a.survey:
        print(survey(cfg, arch_list, a.scale, mac_pj=mac_pj))
        return 0

    layer_names = (a.layers or " ".join(cfg.layers)).split()
    if not layer_names:
        raise SystemExit("dilation: name the layers with --layers or ECC_LAYERS "
                         "(this is a spot check by construction -- see --survey)")
    k_over_n = cfg.code_k / cfg.code_n

    if a.gate:
        vs = ([int(float(x)) for x in a.victories.split()] if a.victories
              else [int(float(x)) for x in cfg.depth_sweep_gate_victories])
        scales = ([float(s) for s in a.scales.split()] if a.scales
                  else [float(s) for s in cfg.depth_sweep_gate_scales])
        dw = a.recon_datawidth
        if dw is None:
            dw = code_widths.declared_datawidth(cfg.code_n, cfg.code_k,
                                                cfg.weight_bits)
        text, rows = convergence_gate(cfg, arch_list, layer_names, vs, scales,
                                      dw, mac_pj=mac_pj)
        print(text)
        if a.csv:
            print(f"\n  csv -> {table_to_csv(rows, a.csv)}")
        return 0

    if a.levels:
        # THE DEPTH LADDER. prompt_2 reports x0.5 / x0.25 / x0.125 -- the
        # realistic design choices a YAML quotes -- but SEARCHES on sqrt(2)
        # steps, because the window where Embedded cannot hold the tile and
        # Recon can is exactly as wide, in depth, as the effective-capacity
        # ratio: a factor-2 grid steps clean over a 1.17x or 1.33x window and
        # reports a grid artifact as "no effect".
        scales = ([float(s) for s in a.scales.split()] if a.scales
                  else [1.0, 0.7071, 0.5, 0.3536, 0.25, 0.1768, 0.125])
        dw = a.recon_datawidth
        if dw is None:
            dw = code_widths.declared_datawidth(cfg.code_n, cfg.code_k,
                                                cfg.weight_bits)
        dl = (a.depth_levels.split() if a.depth_levels
              else list(cfg.weight_depth_levels))
        text, csv_rows = level_table(cfg, arch_list, layer_names, scales, dw,
                                     mac_pj=mac_pj, depth_levels=dl)
        print(text)
        # THE PER-DATASPACE VIEW IS NOT OPTIONAL. On this design the weight
        # rows above sit at their refetch floor on both arms and show only the
        # flat packing discount; 97 % of the margin is in the Outputs row.
        dtext, drows = dataspace_table(cfg, arch_list, layer_names, scales, dw,
                                       mac_pj=mac_pj, depth_levels=dl)
        print()
        print(dtext)
        if a.csv:
            print(f"\n  csv -> {level_table_to_csv(csv_rows, a.csv)}")
            dpath = pathlib.Path(a.csv)
            dpath = dpath.with_name(dpath.stem + "__dataspace" + dpath.suffix)
            print(f"  csv -> {dataspace_to_csv(drows, dpath)}")
        return 0

    if a.table:
        refs = [float(s) for s in a.refs.split()]
        if a.scales:
            scales = [float(s) for s in a.scales.split()]
        else:
            scales = []
            for s in refs:
                scales += [round(s, 4), round(s / k_over_n, 4)]
            seen = set()
            scales = [s for s in scales if not (s in seen or seen.add(s))]
        text, csv_rows = utilisation_table(cfg, arch_list, layer_names, scales,
                                           k_over_n=k_over_n, mac_pj=mac_pj)
        print(text)
        if a.csv:
            print(f"\n  csv -> {table_to_csv(csv_rows, a.csv)}")
        return 0

    rows, gaps = sweep(cfg, arch_list, layer_names,
                       [float(s) for s in a.refs.split()],
                       k_over_n=k_over_n, mac_pj=mac_pj)
    if not rows:
        print("dilation: nothing cached yet for this sweep.")
        for g in gaps:
            print("   ", g)
        return 1
    print(render(rows, gaps, cfg, k_over_n, mac_pj))
    if a.csv:
        print(f"\n  csv -> {to_csv(rows, a.csv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
