"""Reading a capacity sweep's mapper caches into one comparable record per layer.

`load()` and `read_mapped_layer()` turn `timeloop-mapper.stats.txt` and
`.map.txt`, at one declared capacity scale, into a `MappedLayer`: DRAM weight
reads, per-level utilised capacity and residency, the loop nest above the weight
level, and the reduction factor each dataspace got. `cache_for()` is where those
files are -- one `fp-<hash>` per SCALE, because a dilated array is a different
chip and its mapping lives under its own fingerprint.

Nothing here invokes a mapper; `hpc/map_capacity_sweep.sh` and
`ECC_SWEEP=area bash hpc/run_all.sh` fill the caches this reads (the second
replaced `hpc/map_depth_sweep.sh` in EnvReorganisation phase 3). A scale with
no cache is reported missing, never interpolated.

ProjectRestructure phase 3 cut `experiments/dilation.py` (1,851 lines) into this,
`study/dilation.py`, `study/dilation_tables.py` and `report/dilation_view.py`.
"""
from __future__ import annotations

import dataclasses
import pathlib
import re

from ..arch import fingerprint as fingerprint_mod
from ..arch import patch
from .. import paths as pathsmod


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
    rows = [r for r in patch.weight_capacity_levels(arch, cfg) if not r["latch"]]
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
    sub = cfg.with_(weight_capacity_scale=float(scale))
    variant = fingerprint_mod.effective_variant(arch, sub)
    fp = fingerprint_mod.arch_fingerprint(arch, sub)
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


