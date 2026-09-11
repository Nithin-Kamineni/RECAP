"""recon: Task 3 -- reconstruction placement on ONE architecture, fixed mapping.

    bash run.sh recon --eval                     # the whole model
    bash run.sh recon --layers "layer3.0.downsample.0 layer4.1.conv2" --eval
    bash hpc/tl.sh bash run.sh recon --eval      # HiPerGator (host python has no pandas)

or, from env.sh, by setting `ECC_RECON_MODELING=1` and running the one command:

    bash hpc/run_all.sh --eval-only

WHAT IT COMPUTES
----------------
One result file and one figure holding, side by side and from ONE set of cached
mappings and ONE raw energy record:

  baseline_external_parity   Task 1's conventional-ECC number, from
                             `ecc.external_parity()` -- the same function and
                             the same digits `bash run.sh baseline` writes.
  embedded_ecc               Task 2's number, from `ecc.embedded_dram()`. This
                             is the reference the plan names for Task 3.
  recon1 .. recon5           the five weight-path boundaries of
                             `01_project_context_and_architectures.txt` Sec. 5.2,
                             each either evaluated or explicitly `unsupported`.

WHY THIS IS ONE ARCHITECTURE AT A TIME
--------------------------------------
The three sweeps put architectures on an axis because all three ECC ARMS exist
on every design. A reconstruction BOUNDARY does not: Eyeriss v2's boundaries
are its mesh, its cluster-local fanout and its PE scratchpad, and a
weight-stationary design's are a different list. Drawing them on one axis would
put "reconstruct after the mesh" next to a design that has no mesh. So the
architecture is held, the boundary is the axis, and Task 5 repeats the study
per design.

WHAT IS HELD FIXED, AND CHECKED RATHER THAN ASSERTED
-----------------------------------------------------
Everything except the boundary. The result file proves it:

  fixed_mapping_shared_across_variants   the same mapping ids behind every bar
  evaluation_only_rerun                  the mapper was not invoked
  weight_path_reconciles_with_raw_record this module's own re-parse of the
                                         cached stats reproduces the `Raw`
                                         record's weight energy per category
  dram_scaled_by_K_over_N                the WHOLE DRAM weight energy is x K/N
                                         on every bar -- the decoder is on the
                                         DRAM die, so only the k message bits
                                         are read out and driven off it -- and
                                         no bar leaves it at full width
  non_weight_energy_identical            inputs, partial sums and compute are
                                         the same numbers under every bar
  placement_savings_are_bounded_by_the_weight_path
                                         no bar saves more than the reducible
                                         weight energy that exists

THE DRAM TERM, SINCE 2026-09-09
-------------------------------
The BCH decoder is on the DRAM die and off the fetch path (01_project_context
Sec. 1 and 4), so the DRAM weight energy is ONE weight-path stage, `dram`,
scaled by K/N under every boundary because only the k message bits are read out
and driven off the die. The `f_if` array/interface split (0.40) was REMOVED on
2026-09-09: it charged a 38.1% bit cut as a 15.2% energy cut, and the DRAM
access is designed to collect only the message bits of each codeword. The
per-bit cost itself is `ECC_DRAM_PJ_PER_BIT` (40 pJ/bit; Accelergy's own is 8).
`ECC_RECON_DECODE_SITE=
controller` reproduces the pre-2026-09-09 numbers for the diff. The figure
subtitle, the table, the manifest and every result file carry the decode site
and the per-bit cost.

WHAT IS OUTSIDE THE COMPARISON
------------------------------
ECC codec (decode/correction) energy, exactly as in Task 2: charged to nobody
unless `ECC_DECODE=1`, and the file says so. Moving the decoder onto the DRAM
die does not change that: it is the same decoder, run once per corrected
codeword, and it cancels between the embedded bar and every placement. Latency and area: Task 3 asks for
energy, Task 4 for "relevant latency/area overhead". No inference-accuracy
claim is made anywhere.
"""
from __future__ import annotations

import dataclasses
import math
import pathlib
import re

import pandas as pd

from .. import archs as archmod
from .. import recon as reconmod
from .. import timeloop as tlmod
from ..archs import accumulator_bits, load_provenance
from .. import baseline_dram
from ..ecc import embedded_dram, external_parity, load_recon_energy
from ..energy import plot_cats
from ..plots import panels as panels_mod
from ..plots.stacked import grouped_stacks, write_table
from ..results_store import ResultBuilder, Variant
from . import audit
from .common import Session

EXPERIMENT = "task3_reconstruction_placement_fixed_mapping"
#: TASK 4. The same boundaries, but the reconstruction arm gets its OWN
#: mapping, solved against N/K more weight capacity. `RECON_OPTIMIZER=True`
#: selects it, `ECC_PHASE=Post` is required, and the two experiments never
#: share a result file name.
EXPERIMENT_TASK4 = "task4_reconstruction_aware_mapping_capacity_dilation"

PARITY_KEY = "DRAM external BCH parity"

#: Categories the per-bar "on chip" note leaves out. DRAM because it is not on
#: chip -- its interface saving, common to every boundary, gets its own line of
#: the note rather than being folded into the on-chip figure; reconstruction, its overhead and the codec because they are what a boundary
#: COSTS, and the panel is scoped to what it can save; external parity because
#: it exists only on the conventional bar. Uniform across every bar, so no
#: boundary is flattered by the scoping.
ONCHIP_EXCLUDE = ("DRAM", "Compute", "Reconstruction", "Recon overhead",
                  "ECC decode", PARITY_KEY)

CODEC_NOTE = (
    "ECC codec (decode/correction) energy is OUTSIDE this comparison unless "
    "ECC_DECODE=1, exactly as in Task 2. Every placement decodes the same "
    "codewords as the embedded reference -- the reduced representation is taken "
    "AFTER correction -- so a characterised codec cost would cancel between the "
    "embedded bar and every recon bar, and would not cancel against the "
    "conventional baseline, which counts codewords by a different layout. "
    "Since 2026-09-09 the decoder sits on the DRAM die (ECC_RECON_DECODE_SITE="
    "ondie), off the fetch path; it is the same BCH decoder relocated, runs once "
    "per corrected codeword, is DRAM-process logic, and stays outside the "
    "placement comparison.")

DRAM_TERM_NOTE = (
    "THE WHOLE DRAM WEIGHT TERM IS REDUCIBLE BY K/N (2026-09-09). The f_if "
    "array/interface split -- 0.40, which charged a 38.1% bit cut as a 15.2% "
    "energy cut -- was removed: the DRAM access is designed to collect only the "
    "message bits of each codeword, so the array reads fewer bits too. The "
    "per-bit DYNAMIC access cost is ECC_DRAM_PJ_PER_BIT (40 pJ/bit; FReaC Cache "
    "MICRO 2020 and Gebhart MICRO 2012 report 28-45) -- see "
    "archs/_shared/provenance.yaml `dram_access_energy`. Every DRAM number "
    "scales linearly with it and no ordering among the placements changes. "
    "E_background and E_refresh are NOT modelled (both 0).")

FIXED_MAPPING_NOTE = (
    "FIXED-MAPPING RESULT. Every access count, fanout, multicast factor and hop "
    "count here is the one Timeloop chose WITHOUT knowing about reconstruction, "
    "and the mapper was not re-run for any placement. So this file answers "
    "'what does moving the boundary cost and save at constant data movement', "
    "not 'what is the best this architecture can do' -- packed reduced weights "
    "raise the effective weight capacity and may enable better tiling, which is "
    "Task 4's question and is deliberately not credited here.")


# ---------------------------------------------------------------- stats paths
def stats_paths_for(cfg, ses, arch, model):
    """`{shape name: cached stats.txt}` for every mapped layer of this model.

    The placement model needs four things the aggregated `Raw` record does not
    carry -- total access counts, utilized capacity, physical word width and the
    loop nest -- so it re-reads the SAME cached Timeloop output the raw record
    was built from. The mapper is constructed only to resolve cache paths; with
    `ECC_FROM_CACHE=1` it cannot invoke Timeloop, and a shape that is not in the
    cache resolves to nothing and is skipped, exactly as `energy.gather()` skips
    it.
    """
    variant = archmod.effective_variant(arch, cfg)
    fingerprint = ses.fingerprints.get(arch) or archmod.arch_fingerprint(arch, cfg)
    mapper = tlmod.Mapper(
        cfg, arch, None, ses.results.mapper_cache(arch, variant, fingerprint),
        None, fingerprint=fingerprint,
        legacy_root=ses.results.legacy_mapper_cache(arch, variant))
    out = {}
    for layer in ses.models[model]:
        if layer.shape_name in out:
            continue
        path = mapper.stats_for(layer)
        if path is not None:
            out[layer.shape_name] = path
    return out, mapper


# --------------------------------------------------------- TASK 4: the second
#  mapping -- the reconstruction arm's own
# ----------------------------------------------------------------------------
TASK4_NOTE = (
    "RECONSTRUCTION-AWARE MAPPING (Task 4). The reference bars are the design "
    "at its configured weight capacity; every reconstruction bar is the SAME "
    "design re-mapped with N/K times as much weight room, because the reduced "
    "representation stores N/K more weights in the same silicon. So the two "
    "arms no longer move the same data: the reconstruction arm reloads weight "
    "tiles from DRAM fewer times, and a read it never issues removes the DRAM "
    "whole per-access energy, not just a share of it. That is why its saving "
    "efficiency can exceed the (1 - K/N) = 0.381 a fixed mapping buys, "
    "and it is the one thing Task 3 structurally cannot show -- with "
    "RECON_OPTIMIZER=False both arms refetch identically by construction.")


@dataclasses.dataclass
class DilatedView:
    """The reconstruction arm's own mapping, and everything needed to bill it.

    Built by `dilated_view()`. Holds a SECOND `Session`, because a different
    weight capacity is a different architecture to the mapper and therefore a
    different mapper cache, a different raw energy record and a different loop
    nest -- not a rescale of the first one.
    """
    cfg: object                 # the dilated Config (weight_capacity_scale x N/K)
    scale: float                # the dilated arm's capacity, x the declared design
    ref_scale: float            # the reference arm's capacity, x the declared design
    raw: object                 # the dilated mapping's Raw record
    wpath: object               # its weight path, corrected
    base_series: object         # its plotted-category totals
    base_w: object              # the weight share of those
    stage_key: str              # the stage whose level the dilation rewrote
    correction: dict            # what capacity_dilation_correction() did
    capacity: dict              # declared vs dilated Effective size, per design
    fingerprint: str
    reconciles: bool
    recon_check: dict


def dilated_view(cfg, ses, arch, model, base_cats, ref_raw):
    """Solve, read and CORRECT the reconstruction arm's own mapping.

    This function is where `RECON_OPTIMIZER=True`'s old refusal now lives. It
    stops the run -- rather than falling back to the fixed mapping -- when

      * the design has no weight-carrying storage level a dilation could touch
        (so there is no capacity effect to measure, only a relabelled Task 3);
      * the reconstruction arm's mapper cache is missing or short of shapes
        (the alternative is a figure whose bars come from two different layer
        sets);
      * the dilated capacity does not come back N/K times the reference's (a
        scale that rounded back to the declared depth is not a dilation).

    And one case it handles rather than refuses: when the mapper hands back a
    BYTE-IDENTICAL loop nest on every shape, the reconstruction arm is the
    reference mapping on the same silicon, so the reference record is used
    outright and the run reports that the dilation bought nothing. Anything
    else would report the artifacts of expressing the dilation as depth -- a
    dearer array per access, a longer Timeloop-derived hop into it -- as a
    Task 4 result.

    Each of those would otherwise produce a number that LOOKS like Task 4 and
    is Task 3, which is exactly what the placeholder existed to prevent.
    """
    nk = cfg.code_n / cfg.code_k
    dil_scale = reconmod.capacity_dilation_scale(cfg)
    dcfg = dataclasses.replace(cfg, weight_capacity_scale=dil_scale,
                               experiment=cfg.experiment)

    site = reconmod.dilated_levels(arch, cfg)
    if site is None:
        raise SystemExit(
            f"RECON_OPTIMIZER=True on {arch}, which has no weight-carrying "
            f"storage level that a capacity dilation can touch (every weight "
            f"level it declares is a depth-1 latch).\n"
            f"  Task 4's mechanism IS the extra weight capacity, so there is "
            f"nothing here to measure and the numbers would be Task 3's under "
            f"a Task 4 heading.\n"
            f"  -> RECON_OPTIMIZER=False for the fixed-mapping study")
    stage_key, prefixes = site

    # The second mapping. `Session` is built from the config alone, so a
    # different capacity is reached by replacing the field -- config.py stays
    # the only reader of os.environ.
    print(f"\n  ---- Task 4: the reconstruction arm's own mapping "
          f"(weight capacity x{dil_scale:g} = x{cfg.weight_capacity_scale:g} "
          f"x N/K, N/K = {nk:.4f}) ----")
    dses = Session(dcfg).setup(need_mapper=True)
    dses.collect_arch(arch)
    raw_d = (dses.raws.get(arch) or {}).get(model)
    if raw_d is None:
        variant = archmod.effective_variant(arch, dcfg)
        fp = archmod.arch_fingerprint(arch, dcfg)
        raise SystemExit(
            f"RECON_OPTIMIZER=True needs the reconstruction arm's OWN mapping "
            f"for {arch}/{model} at weight capacity x{dil_scale:g}, and it is "
            f"not in the cache.\n"
            f"  expected: {dses.results.mapper_cache(arch, variant, fp)}\n"
            f"  -> map it:  ECC_PHASE=Post ECC_WEIGHT_CAPACITY_SCALE={dil_scale:g} \\\n"
            f"                ECC_RECON_ARCHS={arch} bash hpc/map_by_shape.sh --no-eval\n"
            f"     or the whole sweep:  bash hpc/map_capacity_sweep.sh\n"
            f"  Refusing rather than falling back to the reference mapping: "
            f"that fallback IS Task 3, and this heading says otherwise.")

    paths_d, mapper_d = stats_paths_for(dcfg, dses, arch, model)
    ref_paths, _ = stats_paths_for(cfg, ses, arch, model)
    if set(paths_d) != set(ref_paths):
        only_ref = sorted(set(ref_paths) - set(paths_d))
        only_dil = sorted(set(paths_d) - set(ref_paths))
        raise SystemExit(
            f"the two arms of Task 4 are mapped on DIFFERENT layer shapes for "
            f"{arch}/{model}, so their totals are not comparable:\n"
            f"  only in the reference (x{cfg.weight_capacity_scale:g}): "
            f"{', '.join(only_ref) or 'none'}\n"
            f"  only in the dilated   (x{dil_scale:g}): "
            f"{', '.join(only_dil) or 'none'}\n"
            f"  -> finish the missing maps before evaluating")

    wpath_d = reconmod.weight_path(dcfg, arch, model, dses.models[model], paths_d)

    # ---- DID THE MAPPER ACTUALLY USE THE ROOM? -----------------------------
    # If every layer's loop nest came back byte-identical, the reconstruction
    # arm is running the REFERENCE MAPPING on the SAME SILICON, and its energy
    # is the reference's -- exactly, not approximately. Saying so is not a
    # shortcut, it is the only correct answer, because everything that differs
    # between the two cached records in that case is an artifact of expressing
    # the dilation as `depth x N/K`:
    #
    #   * Accelergy prices the deeper array dearer per access (1.24x on
    #     eyeriss_like), which `capacity_dilation_correction()` undoes;
    #   * Timeloop derives the NoC hop length from the inner level's Accelergy
    #     AREA when noc.yaml does not pin a `tile_width_um`, so the deeper
    #     array also LENGTHENS the wires into the PE -- worth +731 kpJ of NoC
    #     weight energy on eyeriss_like's two-layer scope, or 0.17 pp of the
    #     saving, all of it against the reconstruction arm.
    #
    # Neither is a property of the reconstruction arm's silicon, which is the
    # declared array holding narrower values. So on an identical nest the
    # reference record IS the answer, and the run says the dilation bought
    # nothing rather than reporting the artifacts as a result.
    nests = {}
    for shape, dil_p in paths_d.items():
        ref_p = ref_paths[shape]
        a = pathlib.Path(ref_p).parent / "timeloop-mapper.map.txt"
        b = pathlib.Path(dil_p).parent / "timeloop-mapper.map.txt"
        nests[shape] = (a.is_file() and b.is_file()
                        and a.read_text() == b.read_text())
    mapping_identical = bool(nests) and all(nests.values())

    # ---- the capacity actually delivered, read off both mappings -----------
    cap_ref = _capacity_of(ref_paths, prefixes)
    cap_dil = _capacity_of(paths_d, prefixes)
    got = (cap_dil / cap_ref) if cap_ref else 0.0
    if cap_ref and abs(got - nk) > 0.05 * nk:
        raise SystemExit(
            f"the dilated mapping of {arch} reports weight capacity "
            f"{cap_dil:,} against the reference's {cap_ref:,} -- a factor of "
            f"{got:.4f}, not the N/K = {nk:.4f} Task 4 is about.\n"
            f"  A `depth:` that rounded back to its declared value is not a "
            f"dilation, and the result would be Task 3's under a Task 4 "
            f"heading.\n"
            f"  -> check `archs._scale_weight_capacity`'s report for {arch}: a "
            f"buffer this shallow may not have a distinct integer depth at "
            f"x{dil_scale:g}")

    # ---- re-price the dilated array at the DECLARED array's cost -----------
    # Accelergy costs a level from its declared geometry, so the dilated level
    # is priced as a physically larger array. The reconstruction arm's array is
    # the same silicon holding narrower values, so that cost is not its cost.
    ref_dir = next(iter(ref_paths.values()), None)
    dil_dir = next(iter(paths_d.values()), None)
    corr = reconmod.capacity_dilation_correction(
        ref_dir.parent if ref_dir is not None else ".",
        dil_dir.parent if dil_dir is not None else ".", prefixes)

    if mapping_identical:
        # The reference mapping on the same silicon: use the reference record
        # outright. No correction is needed because nothing legitimate moved.
        print(f"  the mapper returned a BYTE-IDENTICAL loop nest on all "
              f"{len(nests)} shape(s) -- the extra weight room was not used, "
              f"so this arm IS the reference mapping and is priced as it")
        ref_wpath = reconmod.weight_path(cfg, arch, model, ses.models[model],
                                         ref_paths)
        return DilatedView(
            cfg=dcfg, scale=dil_scale, ref_scale=cfg.weight_capacity_scale,
            raw=ref_raw, wpath=ref_wpath,
            base_series=ref_raw.base.reindex(base_cats, fill_value=0.0),
            base_w=ref_raw.base_w.reindex(base_cats, fill_value=0.0),
            stage_key=stage_key,
            correction=dict(
                corr, stage=stage_key, corrected=False, moved_pJ=0.0,
                mapping_identical=True, per_shape_nest_identical=nests,
                note=("the dilated mapping is byte-identical to the reference "
                      "on every shape, so the reference record is used "
                      "outright: no re-pricing, and NO capacity effect. Every "
                      "difference between the two cached records here is an "
                      "artifact of expressing the dilation as depth (a dearer "
                      "array per access, and a longer Timeloop-derived NoC hop "
                      "into it), none of it a property of the reconstruction "
                      "arm's silicon.")),
            capacity={"reference_weights_per_instance": cap_ref,
                      "dilated_weights_per_instance": cap_dil,
                      "delivered_factor": got, "wanted_factor": nk,
                      "level": prefixes[0] if prefixes else "?",
                      "the_mapper_used_none_of_it": True},
            fingerprint=archmod.arch_fingerprint(arch, dcfg),
            reconciles=True,
            recon_check={"note": "the reference record's own check applies"})

    moved = reconmod.apply_capacity_correction(wpath_d, stage_key, corr)

    # ...AND ON THE PLOTTED CATEGORY, not only on the weight path. The bar is
    # drawn from the raw record's per-category totals, so correcting the
    # weight-path stage alone leaves the drawn stack carrying Accelergy's cost
    # for the bigger array -- which is how the first Task 4 run came out WORSE
    # than Task 3 on a byte-identical mapping (eyeriss_like, +8.61 % against
    # +12.86 %, the whole 18.3 uJ gap being the 1.24x ERT inflation of a
    # scratchpad whose loop nest had not moved). The dearer array is dearer for
    # EVERY dataspace it holds, not just for weights, so the correction is
    # applied to the level's whole energy and to its weight share separately --
    # they differ whenever the dilated level is shared (ECC_WEIGHT_CAPACITY_
    # SCOPE=shared puts Inputs in `operand_glb`).
    # THE CATEGORY MOVES BY EXACTLY WHAT THE STAGE MOVED. `wpath_d`'s stage is
    # the weight share of that level, re-derived from the cached stats with the
    # per-instance counts scaled up the way `read_weight_path()` does, and
    # `cross_check()` reconciles it against this very record -- so it is the
    # one number that is on the same footing as the category total. The raw
    # record's own per-level `reads`/`writes` are NOT: they are per instance
    # while its `energy_pJ` is a total, which priced the level 25x low and got
    # the correction refused (2026-09-09).
    base_all = raw_d.base.copy()
    base_wt = raw_d.base_w.copy()
    stage_cat = reconmod._category_of(
        next(s for s in reconmod.stages_for(arch, dcfg) if s.key == stage_key), dcfg)
    delta = float(moved.get("moved_pJ", 0.0))
    if delta and stage_cat in base_all.index:
        base_all[stage_cat] += delta
        if stage_cat in base_wt.index:
            base_wt[stage_cat] += delta
    moved["plotted_category_repriced"] = {"category": stage_cat, "delta_pJ": delta}

    # A SHARED LEVEL'S NON-WEIGHT SHARE IS LEFT UNCORRECTED, and says so. Under
    # ECC_WEIGHT_CAPACITY_SCOPE=shared the dilated level also holds Inputs, and
    # the dearer array is dearer for those accesses too -- but their access mix
    # is not in the weight path, so re-pricing them would be a guess. Leaving
    # them at Accelergy's cost for the bigger array charges the reconstruction
    # arm MORE, so the resulting saving is conservative.
    other = sorted({str(r.get("dataspace")) for r in (raw_d.levels or [])
                    if r.get("dataspace") != "Weights"
                    and any(str(r.get("level")) == pre
                            or str(r.get("level")).startswith(pre)
                            for pre in prefixes)})
    if other:
        moved["uncorrected_dataspaces_on_the_shared_level"] = {
            "dataspaces": other,
            "meaning": (f"{stage_key} also holds {', '.join(other)}; their share "
                        f"of it is left at Accelergy's cost for the DILATED "
                        f"array, which overcharges the reconstruction arm and "
                        f"makes this saving a lower bound")}

    # The correction changed a stage's energy, so the weight path no longer
    # reconciles with the DILATED raw record by construction -- it reconciles
    # with it MINUS what was corrected. Both numbers are recorded and the
    # figure's own total is rebuilt from the corrected path below, so the
    # reconciliation is reported rather than asserted.
    reconciles, recon_check = reconmod.cross_check(dcfg, arch, wpath_d, base_wt)
    return DilatedView(
        cfg=dcfg, scale=dil_scale, ref_scale=cfg.weight_capacity_scale,
        raw=raw_d, wpath=wpath_d,
        base_series=base_all.reindex(base_cats, fill_value=0.0),
        base_w=base_wt.reindex(base_cats, fill_value=0.0),
        stage_key=stage_key,
        correction=dict(moved, **{k: v for k, v in corr.items()
                                  if k != "provenance"},
                        provenance=corr.get("provenance", "")),
        capacity={"reference_weights_per_instance": cap_ref,
                  "dilated_weights_per_instance": cap_dil,
                  "delivered_factor": got, "wanted_factor": nk,
                  "level": prefixes[0] if prefixes else "?",
                  "the_mapper_used_none_of_it": False,
                  "per_shape_nest_identical": nests},
        fingerprint=archmod.arch_fingerprint(arch, dcfg),
        reconciles=reconciles, recon_check=recon_check)


def _capacity_of(paths, prefixes):
    """`Effective size` of the dilated weight level, from any mapped shape.

    Read off the mapping rather than computed from the YAML, so a `depth:` the
    patch failed to rewrite -- or one Timeloop clamped -- shows up here instead
    of being assumed.
    """
    for path in paths.values():
        text = pathlib.Path(path).read_text()
        for part in text.split("=== "):
            name = part.split(" ===")[0]
            if "STATS" not in part:
                continue
            if not any(name == p or name.startswith(p) for p in prefixes):
                continue
            m = re.search(r"Effective size\s*:\s*(\d+)", part)
            if m:
                return int(m.group(1))
    return 0


# --------------------------------------------------------------------- report
def _report(cfg, arch, model, wpath, gran, packing, rows, base_total, emb_total,
            raw=None):
    """The console table. Prints the CEILING as well as the result, because a
    0.06 % saving is only legible once a reader can see how much reducible
    energy there was to begin with."""
    acc, _ = accumulator_bits(arch, cfg)
    print(f"\n  --- {arch} / {model} : Task 3, reconstruction placement, "
          f"FIXED MAPPING ---")
    print(f"    precisions            : weight {cfg.weight_bits}b   "
          f"activation {cfg.activation_bits}b   accumulator {acc}b (published)")
    mac = getattr(raw, "mac", None) or {}
    print(f"    MAC energy            : {mac.get('pj_per_mac_charged', float('nan')):.5f} "
          f"pJ/op x {mac.get('macs', 0):,.0f} MACs = "
          f"{mac.get('compute_pJ_charged', 0.0) / 1e6:,.3f} uJ "
          f"({mac.get('source', '?')}"
          f"{'; ' + mac.get('citation_short', '') if cfg.mac_pj_override is not None else ''}) "
          f"<-- the DENOMINATOR of every percentage below")
    if cfg.mac_pj_override is not None:
        print(f"                            ERT was {mac.get('ert_pj_per_mac', float('nan')):.5f} "
              f"pJ/op = {mac.get('compute_pJ_ert', 0.0) / 1e6:,.3f} uJ; saved uJ are "
              f"unchanged, only the total moved")
    print(f"    reduced form          : {packing.reduced_bits_per_weight:.4f} of "
          f"{cfg.weight_bits} bits per weight retained "
          f"(K/N = {packing.frac:.4f}), packing={packing.mode}")
    print(f"    ECC group             : {gran.weights_per_codeword:.4f} weights per "
          f"codeword, G_rec = {gran.g_rec} weights must be co-resident")
    # THE CEILING. Without this, a 0.06% result reads as a missing term rather
    # than as arithmetic. ON-CHIP only: the DRAM term is reducible too, but it
    # is not on chip and gets its own lines below.
    reducible = [s.key for s in reconmod.stages_for(arch, cfg)
                 if s.reducible and s.kind != "dram"]
    on_chip = wpath.reducible_energy(reducible)
    print(f"\n    CEILING on any boundary's saving:")
    print(f"      on-chip weight energy that CAN be reduced : "
          f"{on_chip / 1e6:12,.3f} uJ = {on_chip / emb_total * 100:5.2f}% of the "
          f"embedded total")
    print(f"      x (1 - K/N) = x {1 - packing.frac:.4f}                     : "
          f"{on_chip * (1 - packing.frac) / 1e6:12,.3f} uJ = "
          f"{on_chip * (1 - packing.frac) / emb_total * 100:5.2f}%   <-- the most "
          f"ANY placement can save")
    term = wpath.dram_term()
    dram_w = term["dram_weight_energy_pJ"]
    ondie = term["decode_site"] == "ondie"
    print(f"      DRAM weight energy (one term, all of it)  : "
          f"{dram_w / 1e6:12,.3f} uJ = {dram_w / emb_total * 100:5.2f}%   "
          f"({term['dram_cost_provenance']})")
    if ondie:
        print(f"        x K/N on EVERY bar                     : "
              f"{dram_w * packing.frac / 1e6:12,.3f} uJ  (only the k message "
              f"bits are read out and driven off the die)")

    print(f"\n    {'bar':26s} {'total uJ':>13s} {'vs base':>9s} {'vs emb':>9s} "
          f"{'DRAM saved uJ':>14s} {'recon uJ':>12s} {'overhead uJ':>12s} {'N_rec':>16s}")
    for r in rows:
        if r["status"] != "evaluated":
            print(f"    {r['label'][:26]:26s} {'UNSUPPORTED':>13s}   {r['reason'][:60]}")
            continue
        vb = (base_total - r["total"]) / base_total * 100 if base_total else 0.0
        ve = (emb_total - r["total"]) / emb_total * 100 if emb_total else 0.0
        print(f"    {r['label'][:26]:26s} {r['total'] / 1e6:13,.3f} {vb:8.2f}% "
              f"{ve:8.2f}% {r['dram_saved_pJ'] / 1e6:14,.3f} "
              f"{r['recon_pJ'] / 1e6:12,.3f} "
              f"{r['overhead_pJ'] / 1e6:12,.3f} {r['n_cw']:16,.0f}")
    print(f"    (vs base / vs emb are SAVINGS: a negative number costs more "
          f"than the reference; DRAM saved is the DRAM energy the bar "
          f"REMOVED against the embedded reference)")


# ------------------------------------------------------------------- evaluate
def evaluate(cfg, ses, prov, arch, model, raw):
    cats = plot_cats(cfg)
    base_series = raw.base.reindex(cats, fill_value=0.0)
    base_w = raw.base_w.reindex(cats, fill_value=0.0)

    # ---- the two reference bars, from Task 1's and Task 2's own functions ---
    e_parity, pdetail = external_parity(cfg, raw)
    e_external, edetail = embedded_dram(cfg, raw)

    base_components = audit.components(base_series)
    # the baseline's DRAM cost is a per-bit PRICE, not extra traffic
    # (baseline_dram.py); unset ECC_BASELINE_DRAM_PJ_PER_BIT = the old model
    pricing = baseline_dram.charge(cfg, raw, base_components, e_parity)
    base_total = float(sum(base_components.values()))

    emb_components = audit.components(base_series)
    emb_components[PARITY_KEY] = e_external          # 0.0 by construction
    emb_total = float(sum(emb_components.values()))

    # ---- the weight path, re-read from the cached Timeloop output -----------
    paths, mapper = stats_paths_for(cfg, ses, arch, model)
    wpath = reconmod.weight_path(cfg, arch, model, ses.models[model], paths)
    reconciles, recon_check = reconmod.cross_check(cfg, arch, wpath, base_w)

    # ---- TASK 4: the reconstruction arm gets its OWN mapping ---------------
    # The reference bars above stay on the configured capacity. Every
    # reconstruction bar below is re-derived from a SECOND mapping solved
    # against N/K more weight room, so the two arms no longer move the same
    # data -- which is the entire mechanism Task 3 cannot show. With
    # RECON_OPTIMIZER=False `dil` is None and every line below is Task 3's.
    dil = (dilated_view(cfg, ses, arch, model, cats, raw)
           if cfg.recon_optimizer else None)
    p_wpath = dil.wpath if dil else wpath
    p_base_w = dil.base_w if dil else base_w
    p_base_series = dil.base_series if dil else base_series
    p_raw = dil.raw if dil else raw

    gran = reconmod.Granularity(cfg.code_n, cfg.code_k, cfg.weight_bits,
                                cfg.recon_granularity)
    packing = reconmod.Packing(cfg.recon_packing, cfg.weight_bits,
                               cfg.code_k, cfg.code_n)
    # prompt_2: THE ON-CHIP NARROWING MUST BE APPLIED EXACTLY ONCE. Since the
    # mapper can now deliver it via `ECC_WEIGHT_DATAWIDTH`, running that
    # beside the default `stream` packing would SQUARE the on-chip saving.
    # Stops here, before a number exists, rather than being caught in review.
    narrowing = reconmod.assert_onchip_narrowing_once(cfg)
    if narrowing.get("note"):
        print(f"  [narrowing] {narrowing['note']}")
    recon_pj, recon_prov = load_recon_energy(cfg)

    # Decode: charged to nobody by default (Task 2's rule). When ECC_DECODE=1
    # the embedded layout's codeword count is what every recon bar decodes,
    # because the reduced form is taken AFTER correction.
    n_cw_emb = raw.dram_w_reads / gran.weights_per_codeword
    decode_emb = n_cw_emb * cfg.decode_pj_emb if cfg.decode_enabled else 0.0
    # Under Task 4 the reconstruction arm issues FEWER DRAM reads, so it
    # decodes fewer codewords. Charging it the reference's count would bill it
    # for reads it never made.
    n_cw_placement = p_raw.dram_w_reads / gran.weights_per_codeword
    decode_placement = ((n_cw_placement * cfg.decode_pj_emb)
                        if (cfg.decode_enabled and cfg.recon_placement_charges_decode)
                        else 0.0)

    # ---- the five placements ------------------------------------------------
    wanted = cfg.recon_placements_for(arch)
    placements = [p for p in reconmod.placements_for(arch, cfg)
                  if not wanted or p.key in wanted or p.variant in wanted]
    results = [reconmod.evaluate_placement(
        cfg, arch, p, p_wpath, p_base_w, p_base_series, recon_pj, gran,
        packing, decode_pj=decode_placement) for p in placements]

    # ---- the result file ---------------------------------------------------
    builder = ResultBuilder(cfg, ses.results, arch, model,
                            experiment=EXPERIMENT_TASK4 if dil else EXPERIMENT,
                            fixed_mapping=not dil)
    mapping_ids = audit.mapping_ids_of(raw)
    # TASK 4: the reconstruction bars come from a DIFFERENT mapping, so they
    # must carry that mapping's ids. A result file whose recon bars claimed the
    # reference's ids would be indistinguishable from Task 3's.
    placement_mapping_ids = audit.mapping_ids_of(p_raw) if dil else mapping_ids
    builder.add(Variant(
        "baseline_external_parity", kind="baseline", status="evaluated",
        total_energy_pJ=base_total, energy_by_component_pJ=base_components,
        mapping_ids=mapping_ids,
        label="Conventional ECC, BCH parity external in DRAM",
        extra={"external_parity_accounting": pdetail,
               "timeloop_energy_pJ": float(raw.total),
               "ecc_codec_energy": CODEC_NOTE}))
    builder.add(Variant(
        "embedded_ecc", kind="embedded", status="evaluated",
        total_energy_pJ=emb_total, energy_by_component_pJ=emb_components,
        mapping_ids=mapping_ids,
        label="Embedded ECC, parity inside the stored weights, no external parity",
        extra={"embedded_dram_accounting": edetail,
               "dram_energy_removed_vs_conventional_pJ": (
                   base_components.get("DRAM", 0.0) - emb_components.get("DRAM", 0.0)
                   + e_parity - emb_components.get(PARITY_KEY, 0.0)),
               "role": ("the reference Task 3 measures reconstruction against: "
                        "every placement reads the same complete codeword out of "
                        "the DRAM ARRAY as this bar; with the decoder on the DRAM "
                        "die (ECC_RECON_DECODE_SITE=ondie) every placement's DRAM "
                        "INTERFACE share is this bar's x K/N, because this bar's "
                        "on-chip datapath consumes every weight bit and so its "
                        "complete codeword still crosses the interface"),
               "dram_term": wpath.dram_term(),
               "ecc_codec_energy": CODEC_NOTE}))

    rows = []
    for res in results:
        p = res.placement
        label = f"{p.label}  [hypothesis {p.rating}]"
        if res.status != "evaluated":
            builder.add(Variant(
                p.variant, kind="reconstruction", status="unsupported",
                unavailable_reason=res.reason, label=label,
                extra=dict(res.detail, placement_key=p.key)))
            rows.append({"key": p.key, "label": p.short.replace("\n", " "),
                         "status": res.status, "reason": res.reason,
                         "total": None, "recon_pJ": 0.0, "overhead_pJ": 0.0,
                         "n_cw": 0.0, "dram_saved_pJ": 0.0})
            continue
        counts = res.detail["reconstruction_counts"]
        builder.add(Variant(
            p.variant, kind="reconstruction", status="evaluated",
            total_energy_pJ=res.total_pJ,
            energy_by_component_pJ={k: v for k, v in res.components.items()
                                    if v != 0.0},
            mapping_ids=placement_mapping_ids, label=label,
            extra=dict(res.detail, placement_key=p.key,
                       **({"reconstruction_aware_mapping": TASK4_NOTE,
                           "weight_capacity": dil.capacity,
                           "capacity_dilation_correction": dil.correction,
                           "dram_weight_reads_reference": float(raw.dram_w_reads),
                           "dram_weight_reads_this_arm": float(p_raw.dram_w_reads),
                           "refetch_reference":
                               float(raw.dram_w_reads) / raw.weights if raw.weights else 0.0,
                           "refetch_this_arm":
                               float(p_raw.dram_w_reads) / p_raw.weights if p_raw.weights else 0.0}
                          if dil else {"fixed_mapping": FIXED_MAPPING_NOTE}),
                       ecc_codec_energy=CODEC_NOTE)))
        rows.append({"key": p.key, "label": p.short.replace("\n", " "),
                     "status": "evaluated", "reason": "",
                     "total": res.total_pJ,
                     "recon_pJ": counts["reconstruction_energy_pJ"],
                     "overhead_pJ": counts["recon_overhead_energy_pJ"],
                     "n_cw": counts["reconstruction_events_codewords"],
                     "dram_saved_pJ":
                         res.detail["dram_model"]["dram_saving_pJ"]})

    # ---- the checks that make the comparison auditable ---------------------
    # TASK 3 and TASK 4 assert DIFFERENT invariants and must not borrow each
    # other's. Task 3's "the DRAM array is identical on every bar" is FALSE
    # under Task 4 and is meant to be: the reconstruction arm issues fewer
    # reads, so it legitimately pays less array energy. Task 4 therefore
    # replaces that check with a tighter one -- the array credit must equal
    # exactly the reads the mapper removed, and not a picojoule more.
    if dil:
        task4_checks(builder, cfg, arch, raw, dil, base_components,
                     emb_components, e_parity, results, wpath, base_w,
                     newly_mapped=getattr(mapper, "n_mapped", 0), pricing=pricing)
    else:
        task3_checks(builder, cfg, arch, raw, base_components, emb_components,
                     e_parity, results, wpath, reconciles, recon_check, base_w,
                     newly_mapped=getattr(mapper, "n_mapped", 0), pricing=pricing)

    # ---- Task 1's checks and caveats, so the baseline bar is as audited ----
    arch_report = audit.common_checks(builder, cfg, arch, raw, pdetail, mapping_ids)
    audit.common_caveats(builder, cfg, arch, raw, pdetail, pricing)
    builder.approximate(edetail["traffic"]["method"])
    builder.approximate(CODEC_NOTE)
    if dil:
        builder.approximate(TASK4_NOTE)
        builder.approximate(dil.correction.get("provenance", ""))
        if not dil.capacity.get("the_mapper_used_none_of_it"):
            builder.approximate(
                "THE DILATED DESIGN'S NoC HOP LENGTH IS NOT CORRECTED. Where "
                "archs/_shared/noc.yaml does not pin a `tile_width_um`, "
                "Timeloop derives the hop length from the inner level's "
                "Accelergy AREA, so expressing the dilation as `depth x N/K` "
                "also lengthens the wires into that level. The per-access "
                "energy is re-priced at the declared array; the hop length is "
                "not, because the wire share of a network's energy is not "
                "separable from its switching share after the fact. It costs "
                "the reconstruction arm NoC energy it would not pay on the "
                "declared silicon (0.17 pp on eyeriss_like's two-layer scope), "
                "so this saving is a lower bound on that count too.")
        _rd = dil.correction.get("read_ratio_declared_over_dilated", 1.0)
        if _rd and _rd < 0.95:
            builder.warn(
                f"Accelergy priced the DILATED {dil.capacity['level']} "
                f"{1 / _rd:.3f}x DEARER per read than the "
                f"declared one, because it costs a level from its declared "
                f"geometry and the dilation is expressed as depth. The energy "
                f"is corrected back to the declared array (the reconstruction "
                f"arm's array is the same silicon holding narrower values), but "
                f"the MAPPER optimised against the dearer one and therefore had "
                f"a reason not to use the extra capacity. This Task 4 saving is "
                f"a LOWER bound for that reason.")
    else:
        builder.approximate(FIXED_MAPPING_NOTE)
    builder.approximate(packing.to_dict()["meaning"])
    builder.approximate(gran.to_dict()["meaning"])
    if cfg.recon_decode_site == "ondie":
        builder.approximate(DRAM_TERM_NOTE)
        builder.approximate(
            f"DRAM model (decoder on the DRAM die, off the fetch path): the "
            f"WHOLE DRAM weight energy is x K/N under every boundary -- the "
            f"f_if array/interface split was removed 2026-09-09, and the DRAM "
            f"access collects only the message bits of each codeword. "
            f"{cfg.dram_cost_note}. {cfg.dram_static_note}.")
    else:
        builder.warn(
            "ECC_RECON_DECODE_SITE=controller: this is the pre-2026-09-09 model "
            "(controller-side correction on the fetch path, complete codeword "
            "across the DRAM interface, DRAM identical on every bar), kept as a "
            "runnable row for the diff. Do not quote it as the study's result.")
    builder.approximate(
        "NO BOUNDARY CARRIES A PER-PE REUSE REGISTER. R4b (SPad output plus a "
        "reconstructed-weight register) was removed on 2026-09-10: measured on "
        "these mappings, consecutive weight reuse is 1 on 20 of 21 resnet18 "
        "layers, so a latch catches nothing, and a register that does pay has "
        "to hold the whole inner tile -- up to 384 weights against a "
        "384-weight scratchpad. Scratchpad read counts are Timeloop's under "
        "every bar.")
    builder.approximate(
        f"Weight-path stages are matched to Timeloop levels by name "
        f"(eccenergy/recon.py WEIGHT_PATHS[{arch!r}]) and the match is verified "
        f"against the raw record's per-category weight energy before any "
        f"placement is evaluated. A level carrying weight energy that no stage "
        f"claims fails that check rather than being ignored.")
    if arch.startswith("eyeriss_v2"):
        builder.warn(
            "CSC METADATA IS NOT MODELLED. The published Eyeriss v2 PE stores "
            "weights CSC-compressed with a separate address scratchpad "
            "(Table IV: weight data 288B, weight address 16x7b); this study "
            "models the design DENSE, so there is no metadata array here to "
            "leave at full width. A CSC implementation would apply the reduced "
            "representation to the weight DATA only, so R4a's scratchpad "
            "saving is an upper bound on what CSC v2 would see.")
    if cfg.decode_enabled:
        builder.warn(
            "ECC_DECODE=1: codec energy is charged to the reference bars and "
            f"{'to' if cfg.recon_placement_charges_decode else 'NOT to'} the "
            f"placements (ECC_RECON_PLACEMENT_CHARGES_DECODE). No characterised "
            f"codec energy exists in this project, so the value is a parameter, "
            f"not a measurement.")
    audit.common_detail(builder, cfg, ses, arch, model, raw, arch_report, prov)
    builder.set_detail(
        reconstruction_placement={
            "weight_path": wpath.to_dict(),
            "packing": packing.to_dict(),
            "granularity": gran.to_dict(),
            "reconstruction_datapath_pJ_per_codeword": recon_pj,
            "reconstruction_datapath_provenance": recon_prov,
            "decode_site": cfg.recon_decode_site,
            "dram_pj_per_bit": cfg.dram_pj_per_bit,
            "dram_cost_provenance": cfg.dram_cost_note,
            "dram_static_terms": cfg.dram_static_note,
            "dram_term": wpath.dram_term(),
            "placements_defined": [
                {"key": p.key, "variant": p.variant, "boundary": p.label,
                 "hypothesised_rating": p.rating,
                 "reduced_stages": list(p.reduced),
                 "reconstruction_site": p.site_stage,
                 "access_counter": p.site_counter,
                 "description": p.description}
                for p in reconmod.placements_for(arch, cfg)],
            "placements_requested": wanted or "all",
        })

    _report(cfg, arch, model, wpath, gran, packing, rows, base_total, emb_total,
            raw=raw)
    path = builder.write()
    print(f"    -> {path}")
    return {"path": path, "rows": rows, "results": results, "wpath": wpath,
            "base_components": base_components, "emb_components": emb_components,
            "base_total": base_total, "emb_total": emb_total,
            # the WEIGHT share of each plotted category, straight off the `Raw`
            # record. The figure's lower panel is scoped to it, and
            # `cross_check` has already proved it equals this module's own
            # re-parse of the cached stats before anything was evaluated.
            "weight_by_category": {c: float(v) for c, v in base_w.items()},
            "mac": getattr(raw, "mac", None),
            "builder": builder}


def task3_checks(builder, cfg, arch, raw, base_components, emb_components,
                 e_parity, results, wpath, reconciles, recon_check, base_w,
                 newly_mapped=0, pricing=None):
    """The checks that make Task 3's claim -- "only the boundary moved" -- auditable.

    Kept free of the Session so the offline test can drive it with a synthetic
    record, exactly as `embedded.task2_checks` is.
    """
    builder.check("weight_path_reconciles_with_raw_record", reconciles, recon_check)

    # 0. the two tables that define the study still agree with each other. A
    #    reducible stage no boundary reduces is not a small omission: every
    #    boundary below it keeps its own saving and reports that stage at FULL
    #    width, so the whole placement list is understated and nothing else in
    #    this file would say so.
    space_ok, space = reconmod.validate_placement_space(arch, cfg)
    builder.check("placement_space_covers_the_whole_weight_path", space_ok, space)

    evaluated = [r for r in results if r.status == "evaluated"]

    # 1. the DRAM term, in BOTH directions. With the decoder on the DRAM die the
    #    array share is the embedded arm's and the interface share is x K/N, so
    #    every placement's DRAM component must sit at EXACTLY
    #        emb_DRAM - DRAM_w x (1 - K/N).
    #    That equality is split into two one-sided checks on purpose: a bar
    #    that quietly credits the array sits BELOW it and fails the first; a
    #    bar that leaves the interface at full width sits ABOVE it and fails
    #    the second. Each also reads the placement's own dram_model rows, so a
    #    stack cannot say one thing and its detail another. Under `controller`
    #    the expected value is emb_DRAM and both checks collapse onto the
    #    pre-2026-09-09 `dram_identical_to_embedded_reference`.
    dram_ref = emb_components.get("DRAM", 0.0)
    dram_w = float(base_w.get("DRAM", 0.0))
    term = wpath.dram_term()
    ondie = term["decode_site"] == "ondie"
    frac = cfg.code_k / cfg.code_n
    saving = dram_w * (1.0 - frac) if ondie else 0.0
    expected = dram_ref - saving
    tol = abs(expected) * 1e-12 + 1e-6
    rows, ok_d = {}, True
    for r in evaluated:
        got = r.components.get("DRAM", 0.0)
        dm = r.detail.get("dram_model", {})
        want_scale = frac if ondie else 1.0
        row_ok = (bool(dm.get("dram_reduced", False)) == ondie
                  and math.isclose(dm.get("dram_after_pJ", 0.0),
                                   dm.get("dram_weight_energy_pJ", 0.0) * want_scale,
                                   rel_tol=1e-12, abs_tol=1e-6))
        d_ok = row_ok and math.isclose(got, expected, rel_tol=1e-12, abs_tol=tol)
        rows[r.placement.key] = {
            "dram_pJ": got, "expected_dram_pJ": expected,
            "off_expected_by_pJ": abs(got - expected),
            "dram_row_scaled": row_ok, "dram_scale": want_scale,
            "passed": d_ok}
        ok_d = ok_d and d_ok
    builder.check(
        "dram_scaled_by_K_over_N", ok_d,
        {"decode_site": term["decode_site"],
         "dram_pj_per_bit": term["dram_pj_per_bit"],
         "dram_cost_provenance": term["dram_cost_provenance"],
         "dram_weight_energy_pJ": dram_w,
         "embedded_reference_dram_pJ": dram_ref,
         "saving_every_placement_pJ": saving,
         "expected_placement_dram_pJ": expected,
         "per_placement": rows,
         "rule": ("with the decoder on the DRAM die only the k message bits are "
                  "read out and driven off it, so the WHOLE DRAM weight energy "
                  "is x K/N under every boundary: every placement's DRAM "
                  "component must equal emb_DRAM - DRAM_w x (1 - K/N) exactly, "
                  "and its dram row must carry that scale (1.0 under "
                  "ECC_RECON_DECODE_SITE=controller). The f_if array/interface "
                  "split, and with it the separate array check, was removed on "
                  "2026-09-09; the whole term now moves together")})

    # 2. nothing outside the weight path moved
    non_weight = {c: float(base_w.get(c, 0.0)) for c in ("Compute",)}
    pairs = {}
    ok = True
    for cat in ("Compute",):
        ref = emb_components.get(cat, 0.0)
        got = {r.placement.key: r.components.get(cat, 0.0) for r in evaluated}
        pairs[cat] = {"embedded": ref, "placements": got}
        ok = ok and all(math.isclose(v, ref, rel_tol=1e-12, abs_tol=1e-6)
                        for v in got.values())
    builder.check(
        "non_weight_energy_identical", ok,
        {"components": pairs, "weight_share_of_those_categories": non_weight,
         "rule": ("a reconstruction boundary touches the WEIGHT path only; MAC "
                  "energy is identical under every bar because both are "
                  "arithmetic on one raw record")})

    # 3. a placement cannot save more than the reducible weight energy there is
    reducible = {}
    for r in evaluated:
        avail = sum(wpath.stages[k].energy_pJ for k in r.placement.reduced
                    if k in wpath.stages)
        saved = sum(r.detail["energy_saved_by_category_pJ"].values())
        reducible[r.placement.key] = {
            "reducible_weight_energy_pJ": avail, "energy_saved_pJ": saved,
            "within_bound": saved <= avail + 1e-6 and saved >= -1e-6}
    builder.check(
        "placement_savings_are_bounded_by_the_weight_path",
        all(v["within_bound"] for v in reducible.values()),
        {"per_placement": reducible,
         "rule": ("the saving is a reduction of the weight energy of the stages "
                  "the reduced form actually reaches; it cannot exceed it and "
                  "cannot be negative")})

    # 4. the placements' stacks reconcile with the reference bars outside the
    #    weight path and the two ECC categories
    ecc_cats = ("Reconstruction", "Recon overhead", "ECC decode", PARITY_KEY)
    drift = {}
    for r in evaluated:
        d = {}
        for cat in set(emb_components) | set(r.components):
            if cat in ecc_cats:
                continue
            a = r.components.get(cat, 0.0)
            b = emb_components.get(cat, 0.0)
            if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-6):
                d[cat] = {"placement_pJ": a, "embedded_pJ": b,
                          "difference_pJ": a - b}
        drift[r.placement.key] = d
    weight_cats = {"DRAM", "Global buffer", "Local (spads/RF)", "NoC"}
    builder.check(
        "only_weight_path_categories_differ_from_the_embedded_reference",
        all(set(d) <= weight_cats for d in drift.values()),
        {"per_placement_differences": drift, "allowed_categories": sorted(weight_cats),
         "rule": ("outside the two ECC categories, a placement may differ from "
                  "the embedded reference only in the categories its weight "
                  "path lives in")})

    # 5. evaluation-only: nothing was mapped by this run
    builder.check(
        "evaluation_only_rerun",
        bool(cfg.from_cache) or newly_mapped == 0,
        {"ECC_FROM_CACHE": bool(cfg.from_cache),
         "newly_mapped_shapes": newly_mapped,
         "meaning": ("passed: every mapping behind this result already existed "
                     "and the mapper was not invoked -- which is what 'do not "
                     "modify or rerun the mapping optimizer for this comparison' "
                     "requires. Failed: re-run with --eval.")})

    # 6. the reference bars still say what Tasks 1 and 2 said
    ok, detail = baseline_dram.reference_bars_agree(
        base_components, emb_components, e_parity, pricing,
        dram_w_reads=raw.dram_w_reads)
    builder.check("reference_bars_match_tasks_1_and_2", ok, detail)


def task4_checks(builder, cfg, arch, raw, dil, base_components, emb_components,
                 e_parity, results, ref_wpath, base_w, newly_mapped=0,
                 pricing=None):
    """The checks that make Task 4's claim -- "the mapper spent the extra room" --
    auditable, and that stop it borrowing a claim it is not entitled to.

    TASK 3'S CHECKS DO NOT TRANSFER, and running them here would either fail
    honestly or pass dishonestly. Three of them are about a FIXED mapping:

      dram_scaled_by_K_over_N   too loose here -- the reconstruction arm also
          issues fewer DRAM reads, and that extra credit is the whole point.
          Replaced below by a strictly tighter statement: the credit must equal
          EXACTLY the reads the mapper removed, priced at the reference's own
          per-read energy, on top of the K/N narrowing.
      non_weight_energy_identical                  activations and partial sums
          legitimately move when the mapping changes. Only the MAC count is
          mapping-invariant, and that is checked, hard: two mappings of the same
          layer that disagree on Computes are not two mappings of the same
          layer.
      only_weight_path_categories_differ           likewise.

    What replaces them is the pair of facts a Task 4 number stands on: the two
    arms really are the same workload on the same design, and the reconstruction
    arm really did get N/K more weight room and no other advantage.
    """
    evaluated = [r for r in results if r.status == "evaluated"]
    frac = cfg.code_k / cfg.code_n
    nk = 1.0 / frac
    term = dil.wpath.dram_term()
    ondie = term["decode_site"] == "ondie"

    # 0. the two tables that define the placement space still agree
    space_ok, space = reconmod.validate_placement_space(arch, cfg)
    builder.check("placement_space_covers_the_whole_weight_path", space_ok, space)

    # 1. THE TWO ARMS ARE THE SAME WORKLOAD. The MAC count is mapping-invariant
    #    -- a convolution has the multiplications it has, whatever the tiling --
    #    so if the two mappings disagree on it they are not two mappings of one
    #    problem, and every percentage below would be comparing two workloads.
    #    This is the check that makes the rest of the file meaningful.
    # `Raw.mac` is filled by `energy.apply_mac_override()` and carries the MAC
    # count it charged; it is the only place the count survives aggregation.
    ref_macs = float((getattr(raw, "mac", None) or {}).get("macs", 0.0) or 0.0)
    dil_macs = float((getattr(dil.raw, "mac", None) or {}).get("macs", 0.0) or 0.0)
    same_macs = (ref_macs > 0 and math.isclose(ref_macs, dil_macs,
                                               rel_tol=1e-9, abs_tol=1.0))
    builder.check(
        "both_arms_are_the_same_workload", same_macs,
        {"reference_computes": ref_macs, "dilated_computes": dil_macs,
         "reference_weights": float(getattr(raw, "weights", 0.0) or 0.0),
         "dilated_weights": float(getattr(dil.raw, "weights", 0.0) or 0.0),
         "rule": ("the MAC count is mapping-invariant, so the reference and the "
                  "reconstruction-aware mapping must report the same Computes. "
                  "They differ only in HOW the data moves, never in how much "
                  "arithmetic there is -- and if they differ here, the two "
                  "totals are not comparable at all")})

    # 2. THE RECONSTRUCTION ARM GOT N/K MORE WEIGHT ROOM AND NOTHING ELSE.
    #    Read off both mappings' own `Effective size`, not off the YAML: a
    #    `depth:` the patch failed to rewrite, or one Timeloop clamped, shows up
    #    here rather than being assumed.
    cap = dil.capacity
    cap_ok = (cap["reference_weights_per_instance"] > 0
              and abs(cap["delivered_factor"] - nk) <= 0.05 * nk)
    builder.check(
        "reconstruction_arm_has_N_over_K_more_weight_capacity", cap_ok,
        dict(cap, reference_scale=dil.ref_scale, dilated_scale=dil.scale,
             rule=(f"the reduced representation stores N/K = {nk:.4f} times as "
                   f"many weights in the same silicon, so the mapper for the "
                   f"reconstruction arm must have been given exactly that much "
                   f"more room at the weight level -- no more (which would be "
                   f"unearned) and no less (which would understate it)")))

    # 3. THE ARRAY CREDIT IS EXACTLY THE READS THE MAPPER REMOVED. This is the
    #    check that separates Task 4 from wishful thinking. Task 3's rule was
    #    "the array never moves"; here it moves, and it may move by exactly
    #    `(reads_ref - reads_dil) x pJ_per_read` and no more. A boundary that
    #    credited itself for reads it still issues fails.
    dram_ref_emb = emb_components.get("DRAM", 0.0)
    dram_w_ref = float(base_w.get("DRAM", 0.0))
    dram_w_dil = float(dil.base_w.get("DRAM", 0.0))
    reads_ref = float(raw.dram_w_reads)
    reads_dil = float(dil.raw.dram_w_reads)
    # The DRAM weight energy scales exactly with the read count (one flat
    # per-bit constant), so the array term the reconstruction arm owes is the
    # dilated read count's share of it.
    dil_expected = dram_w_dil * frac if ondie else dram_w_dil
    expected = dram_ref_emb - dram_w_ref + dil_expected
    tol = abs(expected) * 1e-9 + 1e-3
    rows, ok = {}, True
    for r in evaluated:
        got = r.components.get("DRAM", 0.0)
        dm = r.detail.get("dram_model", {})
        row_ok = (math.isclose(got, expected, rel_tol=1e-9, abs_tol=tol)
                  and bool(dm.get("dram_reduced", False)) == ondie)
        rows[r.placement.key] = {
            "dram_pJ": got, "expected_dram_pJ": expected,
            "difference_pJ": got - expected,
            "array_row_unreduced_within_this_mapping":
                bool(dm.get("dram_reduced", False)) == ondie,
            "passed": row_ok}
        ok = ok and row_ok
    builder.check(
        "dram_credit_equals_the_reads_the_mapper_removed", ok,
        {"per_placement": rows,
         "reference_dram_weight_reads": reads_ref,
         "reconstruction_dram_weight_reads": reads_dil,
         "reads_never_issued": reads_ref - reads_dil,
         "reference_dram_weight_pJ": dram_w_ref,
         "reconstruction_dram_weight_pJ": dram_w_dil,
         "K_over_N": frac,
         "rule": ("under Task 4 the reconstruction-aware mapping issues fewer "
                  "DRAM reads AND each read it does issue is narrower, so "
                  "every placement's DRAM component must equal "
                  "DRAM_w(dilated) x K/N, shifted by the reference bar's own "
                  "DRAM. A bar below that is counting a saving twice")})

    # 4. THE SAVING IS DECOMPOSED, and the two halves are reported separately
    #    because they have different efficiencies and only one of them is new.
    #    Fixed-mapping efficiency is (1 - K/N) = 0.381 since the f_if split
    #    was removed (it was f_if x (1 - K/N) = 0.152 at f_if = 0.40); a read
    #    never issued still has efficiency 1.0, so the two are now much closer.
    never_issued_pJ = dram_w_ref - dram_w_dil
    fixed_part = dram_w_dil * (1.0 - frac) if ondie else 0.0
    builder.check(
        "dram_saving_is_decomposed_into_refetch_and_narrowing", True,
        {"reads_never_issued_pJ": never_issued_pJ,
         "narrowing_on_the_reads_still_issued_pJ": fixed_part,
         "total_dram_saving_pJ": never_issued_pJ + fixed_part,
         "efficiency_of_the_refetch_part": 1.0,
         "efficiency_of_the_narrowing_part": 1.0 - frac,
         "efficiency_overall": ((never_issued_pJ + fixed_part) / dram_w_ref
                                if dram_w_ref else 0.0),
         "refetch_reference": reads_ref / raw.weights if raw.weights else 0.0,
         "refetch_reconstruction":
             reads_dil / dil.raw.weights if dil.raw.weights else 0.0,
         "first_order_prediction_FINDINGS_9_0":
             ((reads_ref / raw.weights - 1.0) * frac + 1.0) if raw.weights else 0.0,
         "rule": ("informational, always passes: a read the reconstruction arm "
                  "never issues removes the whole per-access energy, so its "
                  "efficiency is 1.0 per pJ, against the (1 - K/N) that "
                  "narrowing an issued read buys. "
                  "FINDINGS section 9 item 0 predicts the refetch this arm "
                  "should show if excess refetch scaled inversely with "
                  "capacity; the measured value beside it is the validation")})

    # 5. THE CORRECTION IS DECLARED. The dilated level was priced by Accelergy
    #    as a physically larger array; the reconstruction arm's is the same
    #    silicon holding narrower values.
    corr = dil.correction
    builder.check(
        "dilated_buffer_repriced_at_the_declared_array", True,
        dict(corr, rule=("Accelergy costs a level from its declared geometry, "
                         "so expressing the dilation as `depth x N/K` prices "
                         "the reconstruction arm's scratchpad as a bigger SRAM "
                         "than it is. Its energy is re-priced at the declared "
                         "array's per-access cost. The MAPPER still optimised "
                         "against the dearer array, so a ratio far from 1.0 "
                         "means the search had a reason to avoid the capacity "
                         "and this result is a LOWER bound")))

    # 6. the weight path still reconciles with the dilated raw record
    builder.check("weight_path_reconciles_with_raw_record",
                  dil.reconciles, dil.recon_check)

    # 7. a placement cannot save more than the reducible weight energy there is
    reducible = {}
    for r in evaluated:
        avail = sum(dil.wpath.stages[k].energy_pJ for k in r.placement.reduced
                    if k in dil.wpath.stages)
        saved = sum(r.detail["energy_saved_by_category_pJ"].values())
        reducible[r.placement.key] = {
            "reducible_weight_energy_pJ": avail, "energy_saved_pJ": saved,
            "within_bound": saved <= avail + 1e-6 and saved >= -1e-6}
    builder.check(
        "placement_savings_are_bounded_by_the_weight_path",
        all(v["within_bound"] for v in reducible.values()),
        {"per_placement": reducible,
         "rule": ("within its OWN mapping, a boundary reduces the weight energy "
                  "of the stages the reduced form reaches and no more. The "
                  "refetch saving is not in this bound -- it is a smaller "
                  "weight path to begin with, not a reduction of this one")})

    # 8. NEITHER ARM WAS MAPPED BY THIS RUN. Task 4 re-optimises the mapping,
    #    but it does so in a separate mapping wave whose cache this reads; an
    #    evaluation that had to invoke Timeloop would mean the figure and the
    #    cache disagree about what was solved.
    builder.check(
        "evaluation_only_rerun",
        bool(cfg.from_cache) or newly_mapped == 0,
        {"ECC_FROM_CACHE": bool(cfg.from_cache),
         "newly_mapped_shapes": newly_mapped,
         "reference_fingerprint": archmod.arch_fingerprint(arch, cfg),
         "reconstruction_fingerprint": dil.fingerprint,
         "meaning": ("the two arms are two mapper caches, both solved before "
                     "this evaluation ran. The fingerprints are recorded side "
                     "by side so a figure drawn from one cache cannot claim "
                     "two")})

    # 9. the reference bars still say what Tasks 1 and 2 said
    ok, detail = baseline_dram.reference_bars_agree(
        base_components, emb_components, e_parity, pricing,
        dram_w_reads=raw.dram_w_reads)
    builder.check("reference_bars_match_tasks_1_and_2", ok, detail)


# --------------------------------------------------------------------- figure
def panel_for(cfg, arch, model, out):
    """ONE panel: the boundary on the x axis, the energy breakdown in the bars.

    Returns everything `draw_panel` and `write_table` need and draws nothing, so
    the same construction serves a one-architecture figure and a panel of a
    multi-architecture one. `figure()` decides which.

    `plots/stacked.draw_panel` draws it -- the same routine the three sweeps and
    the panel figure use, widened rather than forked. One bar per group here,
    because a placement label is a sentence and does not fit as a rotated tag
    under one of six bars in a single group.

    ONE PANEL, AND WHY THE PER-BAR NOTE CARRIES THE RESULT. On
    `eyeriss_v2_like` the whole reducible weight path is 8.5 % of inference
    energy, so the largest saving any boundary can make is 1.62 % and R2's is
    0.096 % -- about one pixel of a 4,798 uJ bar at any sane figure size. The
    reduction IS applied (the per-stage tests prove it exactly), so the evidence
    is made visible in TEXT rather than in a second pair of axes: each bar
    carries one line giving the uJ moved on chip and the uJ paid for
    reconstruction. A percentage cannot separate 0.096 % from 0.150 %, and on
    this figure those two numbers are the result.

    There used to be a second panel below, the same bars rescaled to the on-chip
    weight path. It was removed because it drew no independent quantity: its two
    bands were the weight share of the NoC and of the scratchpads, both of which
    are already inside this figure's `NoC` and `Local (spads/RF)` bands. The
    numbers it carried survive -- `moved` is the per-bar note here, and the
    reducible energy and the ceiling are columns of the CSV and lines of the
    console run.
    """
    cats = plot_cats(cfg)
    groups, stacks, labels = [], {}, {}
    notes = {}
    emb_components = out["emb_components"]
    weight_by_cat = out["weight_by_category"]
    # The reducible on-chip weight path: every category that holds WEIGHT
    # energy except DRAM. Derived, not listed, so it follows
    # `ECC_SPLIT_READ_WRITE` and picks up a weight global buffer on a design
    # that has one. It scopes the per-bar note, not a second set of axes.
    onchip_cats = [c for c in cats
                   if c not in ONCHIP_EXCLUDE and float(weight_by_cat.get(c, 0.0)) > 0]

    def add(key, label, components):
        # The conventional baseline's external parity is a COMPONENT of the
        # result file, not a plotted category, so it has to be folded into the
        # DRAM band or the drawn bar would silently be shorter than the total it
        # is annotated with. `ecc.build_stacks()` folds it the same way, so the
        # two figures agree. Under the price model (baseline_dram.py) that
        # component is 0 and the baseline's cost is already inside `DRAM`; the
        # fold is then a no-op and this stays correct either way.
        stack = {c: float(components.get(c, 0.0)) for c in cats}
        stack["DRAM"] += float(components.get(PARITY_KEY, 0.0))
        drawn = sum(stack.values())
        total = sum(float(v) for v in components.values())
        if not math.isclose(drawn, total, rel_tol=1e-9, abs_tol=1e-3):
            raise SystemExit(
                f"figure would draw {drawn:.3f} pJ for {key!r} but its total is "
                f"{total:.3f} pJ -- a component of the result is not in a "
                f"plotted category, so the bar and its label would disagree. "
                f"Components: {sorted(components)}")
        groups.append(key)
        labels[key] = label
        stacks[key] = pd.DataFrame(
            {"energy": pd.Series(stack).reindex(cats, fill_value=0.0)})

        # The per-bar note: this bar scoped to the weight energy a boundary can
        # actually reduce. Taken from `components` minus the non-weight share of
        # the same categories, so the note and the bar cannot disagree -- the
        # non-weight share is identical under every bar (checked by
        # `non_weight_energy_identical` and `only_the_weight_share_of_a_category
        # _moves`), so subtracting it is a shift, never a rescale.
        weight_only = {}
        for cat in onchip_cats:
            non_weight = (float(emb_components.get(cat, 0.0))
                          - float(weight_by_cat.get(cat, 0.0)))
            weight_only[cat] = float(components.get(cat, 0.0)) - non_weight
        on_chip = sum(weight_only.values())
        recon = (float(components.get("Reconstruction", 0.0))
                 + float(components.get("Recon overhead", 0.0)))
        ref_on_chip = sum(float(weight_by_cat.get(c, 0.0)) for c in onchip_cats)
        moved = on_chip - ref_on_chip
        # Two SHORT lines. A bar slot is about two inches wide, so a note that
        # runs past ~16 characters collides with its neighbour and, because
        # every figure here is saved `bbox_inches="tight"`, drags the whole
        # image wider with it.
        lines = []
        # The DRAM-interface saving first: it is the same on every R bar (the
        # decoder is on the DRAM die), so seeing it repeated is the point.
        dram_moved = (float(components.get("DRAM", 0.0))
                      - float(emb_components.get("DRAM", 0.0)))
        if abs(dram_moved) > 1e-6 and key not in ("baseline", "embedded"):
            sign = "\u2212" if dram_moved < 0 else "+"
            lines.append(f"{sign}{abs(dram_moved) / 1e6:,.2f} \u00b5J DRAM I/O")
        if abs(moved) > 1e-6:
            # the same U+2212 minus `draw_panel` prints on the saving, so the
            # two annotations on one bar do not use two different signs
            sign = "\u2212" if moved < 0 else "+"
            lines.append(f"{sign}{abs(moved) / 1e6:,.2f} \u00b5J on chip")
        if recon:
            lines.append(f"+{recon / 1e6:,.2f} \u00b5J recon")
        if lines:
            notes[key] = "\n".join(lines)

    add("baseline", reconmod.REFERENCE_BARS[0][1], out["base_components"])
    add("embedded", reconmod.REFERENCE_BARS[1][1], out["emb_components"])
    for res in out["results"]:
        if res.status != "evaluated":
            print(f"  [note] {res.placement.key} is unsupported here and is not "
                  f"drawn: {res.reason}")
            continue
        add(res.placement.key, res.placement.short, res.components)

    # The table carries what Task 3 asks a table to carry: the reconstruction
    # counts, the overheads, the savings against BOTH references, and every
    # boundary -- including the ones the figure cannot draw because they are
    # unsupported. One table per stem, as everywhere else.
    base_t, emb_t = out["base_total"], out["emb_total"]

    def pct(ref, total):
        return ((ref - total) / ref * 100.0) if ref and total is not None else ""

    term = out["wpath"].dram_term()
    extra = {}
    for key, total in (("baseline", base_t), ("embedded", emb_t)):
        extra[key] = {
            "boundary": dict(reconmod.REFERENCE_BARS)[key].replace("\n", " "),
            "status": "reference", "hypothesised_rating": "",
            "saving_vs_conventional_ecc_pct": pct(base_t, total),
            "saving_vs_embedded_only_pct": pct(emb_t, total),
            "reconstruction_events_codewords": "", "reconstruction_uJ": "",
            "recon_overhead_uJ": "", "weights_reconstructed": "",
            "reducible_energy_uJ": "",
            "saving_ceiling_uJ": "",
            "decode_site": "controller (reference bar)",
            "dram_pj_per_bit": term["dram_pj_per_bit"],
            # the weight share only, and only for the embedded reference bar
            "dram_uJ": (term["dram_weight_energy_pJ"] / 1e6
                        if key == "embedded" else ""),
            "dram_saving_uJ": 0.0 if key == "embedded" else "",
            "unavailable_reason": "",
        }
    for res in out["results"]:
        p = res.placement
        row = {"boundary": p.label, "status": res.status,
               "hypothesised_rating": p.rating}
        if res.status == "evaluated":
            c = res.detail["reconstruction_counts"]
            b = res.detail["reducible_weight_energy_pJ"]
            dm = res.detail["dram_model"]
            row.update(
                saving_vs_conventional_ecc_pct=pct(base_t, res.total_pJ),
                saving_vs_embedded_only_pct=pct(emb_t, res.total_pJ),
                reconstruction_events_codewords=c["reconstruction_events_codewords"],
                reconstruction_uJ=c["reconstruction_energy_pJ"] / 1e6,
                recon_overhead_uJ=c["recon_overhead_energy_pJ"] / 1e6,
                weights_reconstructed=c["weights_reconstructed"],
                reducible_energy_uJ=b["before_pJ"] / 1e6,
                saving_ceiling_uJ=b["ceiling_on_the_saving_pJ"] / 1e6,
                decode_site=dm["decode_site"],
                dram_pj_per_bit=dm["dram_pj_per_bit"],
                dram_uJ=dm["dram_after_pJ"] / 1e6,
                dram_saving_uJ=dm["dram_saving_pJ"] / 1e6,
                unavailable_reason="")
        else:
            row.update(saving_vs_conventional_ecc_pct="",
                       saving_vs_embedded_only_pct="",
                       reconstruction_events_codewords="", reconstruction_uJ="",
                       recon_overhead_uJ="", weights_reconstructed="",
                       reducible_energy_uJ="",
                       saving_ceiling_uJ="", decode_site=term["decode_site"],
                       dram_pj_per_bit=term["dram_pj_per_bit"], dram_uJ="",
                       dram_saving_uJ="", unavailable_reason=res.reason)
            # An unsupported boundary has no bar, but it must still have a row:
            # a table that simply omits it reads as "not considered".
            groups.append(p.key)
            labels[p.key] = p.short
            stacks[p.key] = pd.DataFrame(
                {"energy": pd.Series({c: 0.0 for c in cats}).reindex(cats)})
        extra[p.key] = row

    drawn = [g for g in groups if float(stacks[g]["energy"].sum()) > 0]
    return {
        "arch": arch, "model": model,
        # `groups` is every bar including the unsupported ones, which have a
        # table row and no bar; `drawn` is what has height. The figure gets
        # `drawn`, the table gets `groups`.
        "groups": groups, "drawn": drawn, "stacks": stacks, "labels": labels,
        "notes": notes, "extra": extra,
        "ref_totals": {g: base_t for g in groups},
        "base_total": base_t, "emb_total": emb_t,
        "mac_ert_pj": (out.get("mac") or {}).get("ert_pj_per_mac"),
    }


def figure(cfg, ses, panels):
    """Draw the placement figure: one panel per architecture, top to bottom.

    ONE ARCHITECTURE -> the single-panel figure this study has always drawn,
    through `grouped_stacks`, unchanged.

    SEVERAL -> `plots/panels.stacked_panels`, one panel per design, each with
    its OWN x axis of its own boundaries and its own two reference bars. The
    rule this does not break is the one env.sh section 4 and CLAUDE.md state:
    the boundaries of two designs must never share an x axis, because
    "reconstruct after the mesh" beside a design with no mesh is meaningless.
    Separate stacked axes are not that -- what they share is the page, the
    legend, the category set and the energy unit, and each panel's heading names
    its design. The panels deliberately do NOT share a y limit: two accelerators
    of different size forced onto one scale makes the smaller unreadable.

    ONE PANEL PER DESIGN IS ALSO THE ONLY HONEST LAYOUT for the counts. Each
    design's bars are measured against ITS OWN conventional-ECC bar, so the
    percentages on one panel say nothing about the other; the table carries both
    with the panel in the row key.
    """
    one = len(panels) == 1
    title = (cfg.recon_title(mac_ert_pj=panels[0]["mac_ert_pj"]) if one
             else cfg.recon_panel_title(mac_ert_pj=panels[0]["mac_ert_pj"]))
    if one:
        pan = panels[0]
        drawn, stacks = pan["drawn"], pan["stacks"]
        figs, _csv = grouped_stacks(
            cfg, ses.results,
            groups=drawn, stacks={g: stacks[g] for g in drawn},
            group_labels=pan["labels"], title=title, stem=cfg.stem,
            group_fontsize=15, bars=["energy"], bar_tags={}, bar_width=0.92,
            ref_totals=pan["ref_totals"],
            bar_notes={g: pan["notes"][g] for g in drawn if g in pan["notes"]})
        # ...and the table gets every row, drawn or not.
        csv = write_table(cfg, ses.results,
                          [(None, pan["groups"], stacks, pan["labels"])],
                          cfg.stem, bars=["energy"],
                          ref_totals=pan["ref_totals"],
                          extra_columns=pan["extra"])
        return figs, csv, drawn

    # ---- several designs: one panel each -----------------------------------
    # Every per-group dict is keyed "<panel>/<group>" for the table, because
    # both panels have a group called `recon1` and `write_table` would
    # otherwise give the second panel the first panel's reference total.
    spec, refs, note_by_panel, extra = [], {}, {}, {}
    for pan in panels:
        key = pan["arch"]
        # The heading names the design and says how much of ITS OWN boundary
        # list is on the axis, because the two panels do not have the same
        # number of boundaries and a reader comparing bar counts across panels
        # would otherwise be counting two different things. The reference bars
        # are not boundaries, so they are not in the count.
        n_ref = len(reconmod.REFERENCE_BARS)
        n_all = len(pan["groups"]) - n_ref
        n_ok = len(pan["drawn"]) - n_ref
        heading = f"{cfg.arch_label(key).replace(chr(10), ' ')}"
        heading += f"   ·   {n_ok} of {n_all} boundaries evaluated"
        if n_all - n_ok:
            heading += f", {n_all - n_ok} infeasible (see the table)"
        spec.append((key, heading, pan["drawn"],
                     {g: pan["stacks"][g] for g in pan["drawn"]}, pan["labels"]))
        refs[key] = pan["ref_totals"]
        note_by_panel[key] = {g: pan["notes"][g] for g in pan["drawn"]
                              if g in pan["notes"]}
        for g in pan["groups"]:
            extra[f"{key}/{g}"] = dict(pan["extra"].get(g, {}),
                                       architecture=cfg.arch_label(key)
                                       .replace(chr(10), " "))
    # The table needs the undrawn rows too, so it is written here rather than by
    # `stacked_panels`, whose figure only ever sees the drawn ones.
    figs, _csv = panels_mod.stacked_panels(
        cfg, ses.results, spec, title=title, stem=cfg.stem, group_fontsize=15,
        bars=["energy"], bar_tags={}, bar_width=0.92, ref_totals=refs,
        bar_notes=note_by_panel,
        panel_note=("each panel is one design's OWN weight path, measured "
                    "against its OWN two reference bars; the panels share a "
                    "legend and a unit, not a y limit or an x axis"))
    csv = write_table(
        cfg, ses.results,
        [(pan["arch"], pan["groups"], pan["stacks"], pan["labels"])
         for pan in panels],
        cfg.stem, bars=["energy"],
        ref_totals={f"{pan['arch']}/{g}": t for pan in panels
                    for g, t in pan["ref_totals"].items()},
        extra_columns=extra)
    return figs, csv, [f"{pan['arch']}/{g}" for pan in panels
                       for g in pan["drawn"]]


# ------------------------------------------------------------------------ run
def run(cfg):
    # PHASE AND TASK ARE ONE CHOICE, NOT TWO. `Pre` means the mapping is
    # ECC-unaware and the ECC effect is applied when evaluating, which is Task
    # 3; `Post` means the mapping itself was solved for the reduced weight
    # width, which is Task 4 and needs RECON_OPTIMIZER=True. The two crossed
    # combinations are both a result filed under a heading that misdescribes
    # it, so both stop the run. (`config.py` catches Post-without-optimizer
    # from the other side.)
    if cfg.recon_optimizer and cfg.phase != "Post":
        raise SystemExit(
            f"RECON_OPTIMIZER=True is Task 4 and is a `Post` result: the "
            f"mapping was re-optimised for the reduced weight width.\n"
            f"  -> ECC_PHASE=Post RECON_OPTIMIZER=True ...")
    if not cfg.recon_optimizer and cfg.phase != "Pre":
        raise SystemExit(
            f"ECC_PHASE={cfg.phase} but Task 3 is a `Pre` result by "
            f"construction: the mapping is fixed and ECC-unaware, and the "
            f"placement effect is applied when evaluating. `Post` is Task 4, "
            f"where the mapping itself is optimised for the reduced width.\n"
            f"  -> ECC_PHASE=Post RECON_OPTIMIZER=True   runs Task 4")

    model = cfg.models[0]
    # ONE PANEL PER ARCHITECTURE, and every one of them checked BEFORE anything
    # is collected: a design whose two tables have drifted apart, or that has no
    # weight path at all, must stop the run rather than quietly draw one panel.
    for arch in cfg.archs:
        if arch not in reconmod.WEIGHT_PATHS:
            raise SystemExit(
                f"no weight path is defined for {arch!r}, so its reconstruction "
                f"boundaries are unknown.\n"
                f"  defined: {', '.join(reconmod.supported_archs())}\n"
                f"  -> each design's weight path and its list of feasible "
                f"boundaries go in eccenergy/recon.py WEIGHT_PATHS and "
                f"PLACEMENTS, together.")

        space_ok, space = reconmod.validate_placement_space(arch, cfg)
        if not space_ok:
            raise SystemExit(
                f"{arch}'s weight path and its placement list have drifted "
                f"apart, so every boundary below the missing stage would be "
                f"reported UNDERSTATED rather than wrong-looking:\n  "
                + "\n  ".join(space["violations"])
                + f"\n  -> {space['fix']}")

        defined = reconmod.placements_for(arch, cfg)
        unknown = [k for k in cfg.recon_placements_for(arch)
                   if k not in {p.key for p in defined}
                   and k not in {p.variant for p in defined}
                   and k not in ("baseline", "embedded")]
        if unknown:
            raise SystemExit(
                f"ECC_RECON_PLACEMENTS[{arch}] names placements this "
                f"architecture does not define: {', '.join(unknown)}\n"
                f"  defined: {', '.join(p.key for p in defined)}\n"
                f"  -> the boundaries are per architecture; see "
                f"eccenergy/recon.py PLACEMENTS")

    ses = Session(cfg).setup()
    ses.collect_all()

    prov = load_provenance()
    panels, outs = [], {}
    for arch in cfg.archs:
        raw = (ses.raws.get(arch) or {}).get(model)
        if raw is None:
            raise SystemExit(
                f"nothing collected for {arch}/{model}; see the [skip] lines "
                f"above.\n  -> every architecture of a panelled placement study "
                f"needs its own mapper cache: map it first "
                f"(ECC_RECON_ARCHS={arch} bash hpc/map_by_shape.sh), or drop it "
                f"from ECC_RECON_ARCHS")
        out = evaluate(cfg, ses, prov, arch, model, raw)
        outs[arch] = out
        panels.append(panel_for(cfg, arch, model, out))

    figs, csv, groups = figure(cfg, ses, panels)
    ses.finish(figs, csv, groups, extra={
        "recon_decode_site": cfg.recon_decode_site,
        "dram_pj_per_bit": cfg.dram_pj_per_bit,
        "dram_cost_provenance": cfg.dram_cost_note,
        "dram_static_terms": cfg.dram_static_note,
        "panel_per_architecture": list(cfg.archs),
        "dram_term": {a: o["wpath"].dram_term() for a, o in outs.items()},
    })

    print("=" * 78)
    if cfg.recon_optimizer:
        print(f"Task 4 reconstruction-AWARE MAPPING: {model}, one panel per "
              f"architecture.")
        print(f"  reference arm       : weight capacity x{cfg.weight_capacity_scale:g} "
              f"of the declared design")
        print(f"  reconstruction arm  : x"
              f"{reconmod.capacity_dilation_scale(cfg):g} "
              f"= x{cfg.weight_capacity_scale:g} x N/K, solved as its own mapping")
    else:
        print(f"Task 3 reconstruction placement: {model}, fixed mapping, "
              f"one panel per architecture.")
    for arch in cfg.archs:
        best = outs[arch]["builder"].best_reconstruction()
        print(f"  {cfg.arch_label(arch).replace(chr(10), ' ')}:")
        if best.get("variant"):
            print(f"    lowest-energy feasible placement: {best['variant']}")
            print(f"      vs conventional ECC : "
                  f"{best.get('savings_vs_conventional_ecc_percent'):+.3f}%")
            print(f"      vs embedded only    : "
                  f"{best.get('savings_vs_embedded_only_percent'):+.3f}%")
        else:
            print(f"    no placement was evaluated: {best.get('reason')}")
    if len(cfg.archs) > 1:
        print("Each design is measured against ITS OWN reference bars, so a")
        print("percentage on one panel says nothing about the other.")
    if cfg.recon_optimizer:
        print("The reconstruction bars come from a DIFFERENT mapping than the")
        print("reference bars: same design, N/K more weight room, solved on its")
        print("own. So the two arms no longer refetch identically, and a DRAM")
        print("read the reconstruction arm never issues saves the ARRAY as well")
        print("as the interface -- efficiency 1.0, against the 0.152 a fixed")
        print("mapping buys. Both mapper caches pre-existed this evaluation.")
    else:
        print("This is an evaluator-only, FIXED-MAPPING result. The mapping optimiser")
        print("was not re-run for any placement -- that is Task 4 (RECON_OPTIMIZER).")
    print("=" * 78)
    return ses
