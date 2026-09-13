"""Task 4 step 1: does CAPACITY DILATION actually cut DRAM weight refetch?

FINDINGS section 9 item 0 states the mechanism and then refuses to let it be
modelled before it is measured:

    "Validate the assumption before building anything. Real tiles are integers,
     so 1.615x capacity may buy a whole extra tile on some layers and nothing on
     others, and the whole table scales with that one number. It needs no new
     modelling code: map each design twice, once at declared weight capacity and
     once at capacity x N/K, and diff the per-layer DRAM weight reads."

`survey()` and `sweep()` are that diff, and nothing else: they read caches
`hpc/map_capacity_sweep.sh` has already filled (`study/dilation_cache.py`) and
report, per layer, what the extra capacity bought. `binding_level()` says WHICH
level was actually full, which is the line that explains a null result.

WHY THE ANSWER IS NOT AUTOMATICALLY POSITIVE
--------------------------------------------
1. WEIGHT CAPACITY IS NOT THE BINDING CONSTRAINT. A mapping that leaves most of
   the weight scratchpad unused is not refetching because it ran out of weight
   room. `residency` against `capacity` in the table says which.
2. TILES ARE INTEGERS. A 1.6154x capacity is only spendable if some factor of the
   layer's C/M/R/S can grow into it, and the mean over layers is NOT the mean of
   the model: `--survey`'s layer-weighted row is the only line that speaks for a
   whole network.
3. THE SEARCH IS STOCHASTIC. A NEGATIVE differential -- the dilated arm
   refetching MORE -- is proof of that, not of a mechanism, and is flagged rather
   than averaged in silently.
4. ACCELERGY PRICES THE DILATED ARRAY AS A BIGGER ONE (1.18-1.45x per access
   here) and the mapper believes it, so the search has a reason to leave the room
   unused. `study/capacity.py` corrects the energy; the mapping cannot be
   corrected, and the `ERT x` column reports it -- so every Task 4 number is a
   LOWER BOUND.

WHAT THE FIRST MEASUREMENT SAID (2026-09-09, eyeriss_like, resnet18, energy):
reason 1, decisively. On `layer2.0.conv1` the declared and dilated loop nests are
BYTE-IDENTICAL at every capacity from x0.125 to x1.6154, because `ifmap_spad` is
at 100 % while `weights_spad` is at 21 %. That is a property of the DATAFLOW, not
of the code rate.

THE TWO GUARDS, each of which caught a false positive: `sibling_fingerprints()`
(a headline once came from a STALE sibling `fp-` directory -- a comparison across
two of them is a comparison of two ARCHITECTURES) and `capacity_verdict()` (a
differential that is really the search missing).

WHAT THE DIFFERENTIAL IS WORTH, AND WHY IT IS NOT TASK 3's NUMBER. Under a fixed
mapping both arms issue the same DRAM reads and the reconstruction arm saves the
K/N narrowing of each, `(1 - K/N)` = 0.381 at BCH(63,39). A read that is NEVER
ISSUED saves the array as well as the interface -- the whole `pJ_per_read` -- so
its efficiency is 1.0. That is the entire point of Task 4, and it is why the two
savings are reported separately instead of being summed.

    python3 -m eccenergy.report.dilation_view --survey
    python3 -m eccenergy.report.dilation_view --layers "layer2.0.conv1"
    python3 -m eccenergy.report.dilation_view --csv results/tables/dilation.csv
"""
from __future__ import annotations

import dataclasses
import pathlib

from ..arch import workloads as workloadsmod

from .dilation_cache import MappedLayer, _dataspace_block, _grab, cache_for, load


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


