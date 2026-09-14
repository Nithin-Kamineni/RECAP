"""Task 4: the reconstruction arm read off its OWN, dilated, mapping.

`dilated_view()` is the second mapping the Task 4 question needs. Task 3 holds
one mapping and varies only the boundary; Task 4 asks what happens when the
reconstruction arm is mapped against the capacity its narrower weights actually
buy, so this reads that arm's own cache entry and reports what the extra room
bought -- refetch, residency, and the per-access price Accelergy charged for the
bigger array (`study/capacity.py` corrects the energy; the mapping cannot be
corrected, so every number here is a lower bound).

A CAPACITY DILATION IS A DIFFERENT CHIP, so its mapping lives under its own
fingerprint and `_capacity_of()` is how a bar states which one it was billed
from. A bar whose dilated entry is not on disk is REPORTED as unmapped, never
silently billed from the reference.

ProjectRestructure phase 3 cut this out of `experiments/recon.py`.
"""
from __future__ import annotations

import dataclasses
import pathlib
import re

from ..arch import fingerprint as fingerprint_mod
from ..arch import placements
from . import capacity as capacity_mod
from . import placement_eval
from ..toolchain import weight_stats
from .common import Session

from .placement_notes import stats_paths_for
from ..settings import guards


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
      * the dilated capacity does not come back `weight_bits / q` times the
        reference's (prompt_6 6: a quantised weight is a whole number of
        bits, so the room delivered is 8/q, never N/K; a scale that rounded
        back to the declared depth is not a dilation either).

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
    dil_scale = capacity_mod.capacity_dilation_scale(cfg)
    dcfg = cfg.with_(weight_capacity_scale=dil_scale, experiment=cfg.experiment)

    site = capacity_mod.dilated_levels(arch, cfg)
    if site is None:
        raise guards.refusal("no-dilatable-level",
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
        variant = fingerprint_mod.effective_variant(arch, dcfg)
        fp = fingerprint_mod.arch_fingerprint(arch, dcfg)
        raise guards.refusal("task4-cache-cold",
            f"RECON_OPTIMIZER=True needs the reconstruction arm's OWN mapping "
            f"for {arch}/{model} at weight capacity x{dil_scale:g}, and it is "
            f"not in the cache.\n"
            f"  expected: {dses.results.mapper_cache(arch, variant, fp)}\n"
            f"  -> map it:  ECC_WEIGHT_CAPACITY_SCALE={dil_scale:g} \\\n"
            f"                ECC_CONST_ARCH={arch} bash hpc/run_all.sh --map-only\n"
            f"     or the whole sweep:  bash hpc/map_capacity_sweep.sh\n"
            f"  Refusing rather than falling back to the reference mapping: "
            f"that fallback IS Task 3, and this heading says otherwise.")

    paths_d, mapper_d = stats_paths_for(dcfg, dses, arch, model)
    ref_paths, _ = stats_paths_for(cfg, ses, arch, model)
    if set(paths_d) != set(ref_paths):
        only_ref = sorted(set(ref_paths) - set(paths_d))
        only_dil = sorted(set(paths_d) - set(ref_paths))
        raise guards.refusal("task4-shapes-differ",
            f"the two arms of Task 4 are mapped on DIFFERENT layer shapes for "
            f"{arch}/{model}, so their totals are not comparable:\n"
            f"  only in the reference (x{cfg.weight_capacity_scale:g}): "
            f"{', '.join(only_ref) or 'none'}\n"
            f"  only in the dilated   (x{dil_scale:g}): "
            f"{', '.join(only_dil) or 'none'}\n"
            f"  -> finish the missing maps before evaluating")

    wpath_d = weight_stats.weight_path(dcfg, arch, model, dses.models[model], paths_d)

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
    # prompt_6 6: the target is weight_bits/q -- exact for every code -- not N/K.
    want, q = capacity_mod.capacity_target(cfg)
    if cap_ref and abs(got - want) > 0.05 * want:
        raise guards.refusal("task4-capacity-not-dilated",
            f"the re-planned mapping of {arch} reports weight capacity "
            f"{cap_dil:,} against the reference's {cap_ref:,} -- a factor of "
            f"{got:.4f}, not the {cfg.weight_bits}/q = {cfg.weight_bits}/{q} = "
            f"{want:.4f} a quantisation arm at q = {q} delivers (prompt_6 6; "
            f"N/K = {nk:.4f} is the ideal, not the whole-bit room).\n"
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
    corr = capacity_mod.capacity_dilation_correction(
        ref_dir.parent if ref_dir is not None else ".",
        dil_dir.parent if dil_dir is not None else ".", prefixes)

    if mapping_identical:
        # The reference mapping on the same silicon: use the reference record
        # outright. No correction is needed because nothing legitimate moved.
        print(f"  the mapper returned a BYTE-IDENTICAL loop nest on all "
              f"{len(nests)} shape(s) -- the extra weight room was not used, "
              f"so this arm IS the reference mapping and is priced as it")
        ref_wpath = weight_stats.weight_path(cfg, arch, model, ses.models[model],
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
                      "delivered_factor": got, "wanted_factor": want, "q": q,
                      "ideal_N_over_K": nk,
                      "level": prefixes[0] if prefixes else "?",
                      "the_mapper_used_none_of_it": True},
            fingerprint=fingerprint_mod.arch_fingerprint(arch, dcfg),
            reconciles=True,
            recon_check={"note": "the reference record's own check applies"})

    moved = capacity_mod.apply_capacity_correction(wpath_d, stage_key, corr)

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
    # SCOPE=shared puts Inputs in the simple designs' `operand_glb`; since
    # 2026-09-13 simple_weight_stationary is NOT one of them -- its operand
    # half is split into `input_glb` and a Weights-only `weight_glb`).
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
    stage_cat = placement_eval._category_of(
        next(s for s in placements.stages_for(arch, dcfg) if s.key == stage_key), dcfg)
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
    reconciles, recon_check = placement_eval.cross_check(dcfg, arch, wpath_d, base_wt)
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
                  "delivered_factor": got, "wanted_factor": want, "q": q,
                  "ideal_N_over_K": nk,
                  "level": prefixes[0] if prefixes else "?",
                  "the_mapper_used_none_of_it": False,
                  "per_shape_nest_identical": nests},
        fingerprint=fingerprint_mod.arch_fingerprint(arch, dcfg),
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


