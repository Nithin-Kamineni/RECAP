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
per-bit cost itself is `ECC_DRAM_PJ_PER_BIT` (20 pJ/bit since 2026-09-11, 40
before; Accelergy's own is 8).
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
import yaml

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
    "per-bit DYNAMIC access cost is ECC_DRAM_PJ_PER_BIT (its value and citation "
    "travel on every manifest and result as dram_cost_provenance) -- see "
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
        legacy_root=ses.results.legacy_mapper_cache(arch, variant),
        ert_bump=archmod.ert_bump(arch, cfg))      # prompt_6: an arm's entries carry its toll
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
    # prompt_6 6: the target is weight_bits/q -- exact for every code -- not N/K.
    want, q = reconmod.capacity_target(cfg)
    if cap_ref and abs(got - want) > 0.05 * want:
        raise SystemExit(
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
                      "delivered_factor": got, "wanted_factor": want, "q": q,
                      "ideal_N_over_K": nk,
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
                  "delivered_factor": got, "wanted_factor": want, "q": q,
                  "ideal_N_over_K": nk,
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


# --------------------------------------------------------- prompt_6: the ERT
#  arms -- one plan per BAR (RULE 4)
# ----------------------------------------------------------------------------
EXPERIMENT_ERT = "task4_reconstruction_aware_mapping_ert"

ERT_NOTE = (
    "RECONSTRUCTION-AWARE MAPPING WITH THE ENCODER IN THE OBJECTIVE (prompt_6). "
    "Each ERT-injectable boundary -- derived from its placement record: a storage "
    "site whose counter names one real ERT action and is not the innermost weight "
    "level's reads -- was RE-MAPPED with its encoder toll in Timeloop's energy "
    "table (E_w x block_size on the site's access action, idle_per_cycle on "
    "`leak`), so the mapper solved the loop nest knowing what reconstruction "
    "costs at that boundary. Such a bar is billed from ITS OWN plan (bill, "
    "discounts, cycles, DRAM reads), and the toll Timeloop billed inside the "
    "level is MOVED into `Reconstruction`, never added a second time. The other "
    "boundaries are post-processed on the reference plan. Two mapping regimes "
    "share this figure; the bars that came from their own mapping are marked.")


@dataclasses.dataclass
class ErtArmView:
    """One ERT arm's OWN plan, and everything needed to bill that bar from it.

    prompt_6 RULE 4.4.3: a bar is bill minus discounts plus the encoder, so the
    bill, the per-level energies, the raw record, the cycle count and the ERT
    delta must all describe the same run -- this one. Built by
    `ert_aware_view()`; `evaluate()` keeps one per ERT arm and a bar with no
    arm points at the reference set.
    """
    placement: object
    cfg: object                 # the arm's Config (recon_ert_arm = its key)
    raw: object                 # the arm's Raw record (toll still inside)
    wpath: object               # its weight path, with the toll MOVED OUT
    base_series: object         # its bill by category, toll moved out
    base_w: object              # the weight share of that
    cycles: float               # its run length, the idle term's denominator
    fingerprint: str
    variant: str
    bump: dict                  # archs.ert_bump()
    split: dict                 # ert_split(), reconciled stats-side and evaluator-side
    nest_identical: bool        # byte-identical loop nest to the reference on EVERY shape
    per_shape_nest_identical: dict
    read_back: dict             # tlmod.read_back_ert() per shape
    checks: dict                # per-shape guard rows (updates, leak multiplier, PEs, ...)
    capacity: dict              # the narrowed level's Effective size, arm / reference = 8/q
    reconciles: bool
    recon_check: dict
    stats_paths: dict


def ert_split(bump, st, cycles):
    """prompt_6 5.3 -- the two amounts an ERT arm's Timeloop run carries because
    of the toll, from THAT bar's own counts and cycles, on the action that was
    bumped (RULE 2), never a level total or a blended average:

        access   scalar <counter> x E_w      (= vector accesses x E_w x block_size)
        leak     idle x engine_cycles  (= sum over shapes of idle x UTILIZED
                 instances x cycles -- what Timeloop bills, buffer.cpp)

    The ACCESS amount sits INSIDE the level's per-dataspace `Energy (total)`,
    i.e. inside the bill's category, and is MOVED into `Reconstruction`. The
    LEAK amount is not in the bill at all: Timeloop prints leakage as a
    separate per-level line outside the per-dataspace energies, and the raw
    record aggregates only those (a few tens of pJ of leakage on the
    reference, never a category). So the idle term is verified against the
    stats' leakage delta and CHARGED by the evaluator, not subtracted.
    `st` is the site stage's StageStats from the arm's own weight path.
    """
    count = float(st.counter(bump["counter"]))
    engine_cycles = float(st.engine_cycles or 0.0)
    engines = (engine_cycles / float(cycles)) if cycles else 0.0
    access = count * float(bump["e_w_pj"])
    leak = float(bump["leak_delta_pj"]) * engine_cycles
    return {"level": bump["level"], "action": bump["action"], "counter": bump["counter"],
            "scalar_accesses": count, "e_w_pj": float(bump["e_w_pj"]),
            "access_toll_pJ": access, "engines": engines, "cycles": float(cycles),
            "engine_cycles": engine_cycles,
            "engines_declared": float(st.declared_instances or 0.0),
            "engines_utilized_max": float(st.instances or 0.0),
            "idle_pj_per_cycle_per_engine": float(bump["leak_delta_pj"]),
            "leak_toll_pJ": leak, "total_toll_pJ": access + leak,
            "rule": ("stats-side access toll = vector accesses x (E_w x block_size) = "
                     "scalar accesses x E_w; stats-side leak = idle x UTILIZED "
                     "instances x cycles per shape (Timeloop power-gates unused "
                     "instances: buffer.cpp FinalizeBufferEnergy); both moved out "
                     "of the level's category into Reconstruction so the stack has "
                     "Task 3's shape")}


def _close(a, b, rel=1e-6, abs_tol=0.0):
    """|a - b| within `rel` of the larger magnitude, OR within `abs_tol` --
    the latter for quantities read off Timeloop's stats, which print to 0.01
    pJ."""
    return abs(float(a) - float(b)) <= max(rel * max(abs(float(a)), abs(float(b)), 1e-9),
                                           float(abs_tol))


def ert_aware_view(cfg, ses, arch, model, base_cats, ref_raw, ref_paths, placement):
    """Load, GUARD and split ONE ERT arm's own mapping (prompt_6 RULE 4).

    Stops the run -- never falls back to the reference plan -- when the arm's
    cache is missing or short of shapes, when a stored ERT is not this bar's
    (RULE 4.4.5 read-back against the reference entry's un-bumped table), when
    `Scalar updates` for Weights at the patched level is not 0 (RULE 2), when
    the leak multiplier is not the UTILIZED instance count of the site level
    (RULE 3 -- Timeloop bills leak x utilized x cycles, power-gating each
    unused instance; verified 2026-09-11 on 43 shapes of two models), when
    DRAM is not 8-bit (RULE 1), when the narrowed level did not deliver 8/q
    (prompt_6 6), or when the attribution split does not reconcile to 1e-6
    (5.3).

    A PE count that differs from the reference is REPORTED, not refused
    (2026-09-11): on a full model the arm's own EDP-optimal plan may use fewer
    PEs where the doubled GLB room changes the DRAM chunking (mobilenet_v2:
    3 of 31 shapes on recon2, each at lower energy AND lower EDP than the
    reference plan), and the reference itself fills the array on fewer than
    two thirds of the shapes. Every such shape is recorded with both PE
    counts, both cycle counts and both EDPs, printed, and carried into the
    manifest's `title_caveats`. The equality rule of prompt_6 10.9 came from
    one layer that fills the array; `PE!=` still suppresses Task 4's
    CAPACITY verdict, which is a different claim.

    The loop-nest verdict is recorded PER ARM and never collapsed (4.4.4).
    """
    acfg = dataclasses.replace(cfg, recon_ert_arm=placement.key)
    bump = archmod.ert_bump(arch, acfg)
    variant = archmod.effective_variant(arch, acfg)
    fp = archmod.arch_fingerprint(arch, acfg)
    print(f"\n  ---- prompt_6: {placement.key}'s OWN mapping -- {tlmod.describe_bump(bump)} ----")
    print(f"       datawidth {acfg.weight_datawidth} on {'+'.join(acfg.weight_datawidth_levels)}; "
          f"cache {variant}  fp {fp}")
    ases = Session(acfg).setup(need_mapper=True)
    ases.collect_arch(arch)
    raw_a = (ases.raws.get(arch) or {}).get(model)
    if raw_a is None:
        raise SystemExit(
            f"ECC_RECON_ERT_AWARE=1 needs {placement.key}'s OWN mapping for {arch}/{model} "
            f"({tlmod.describe_bump(bump)}), and it is not in the cache.\n"
            f"  expected: {ases.results.mapper_cache(arch, variant, fp, create=False)}\n"
            f"  -> map it:  bash hpc/map_ert_arms.sh --no-eval   (one sbatch job per arm x shape)\n"
            f"  Refusing rather than falling back to the reference plan: that fallback IS "
            f"Task 3, and this heading says otherwise.")
    paths_a, _mapper_a = stats_paths_for(acfg, ases, arch, model)
    if set(paths_a) != set(ref_paths):
        raise SystemExit(
            f"{placement.key}'s own mapping and the reference are mapped on DIFFERENT "
            f"layer shapes for {arch}/{model}:\n"
            f"  only in the reference: {', '.join(sorted(set(ref_paths) - set(paths_a))) or 'none'}\n"
            f"  only in {placement.key}: {', '.join(sorted(set(paths_a) - set(ref_paths))) or 'none'}\n"
            f"  -> finish the missing maps before evaluating")

    # RULE 4.4.5, defence 3: every entry's stored ERT, read back against the
    # reference entry's un-bumped table (Accelergy is deterministic and the
    # datawidth does not enter it, FINDINGS 3.5).
    read_back = {}
    for shape, path in paths_a.items():
        ref_ert = pathlib.Path(ref_paths[shape]).parent / tlmod.ERT_NAME
        base_prices = (tlmod.ert_prices(yaml.safe_load(ref_ert.read_text()))
                       if ref_ert.exists() else None)
        read_back[shape] = tlmod.read_back_ert(pathlib.Path(path).parent, bump,
                                               base_prices=base_prices)

    # ---- per-shape guards, off the arm's OWN stats -------------------------
    counts_by_shape = {}
    for layer in ases.models[model]:
        counts_by_shape[layer.shape_name] = counts_by_shape.get(layer.shape_name, 0) \
            + float(getattr(layer, "count", 1) or 1)
    nests, rows, problems = {}, [], []
    stats_access = stats_leak = 0.0
    for shape, path in paths_a.items():
        a = pathlib.Path(ref_paths[shape]).parent / "timeloop-mapper.map.txt"
        b = pathlib.Path(path).parent / "timeloop-mapper.map.txt"
        nests[shape] = a.is_file() and b.is_file() and a.read_text() == b.read_text()
        lv, sm = tlmod.parse_levels(path)
        rlv, rsm = tlmod.parse_levels(ref_paths[shape])
        L = lv.get(bump["level"])
        R = rlv.get(bump["level"])
        if L is None or R is None or "Weights" not in L["ds"]:
            raise SystemExit(f"{placement.key}/{shape}: level {bump['level']!r} carries no "
                             f"Weights in the stats")
        w = L["ds"]["Weights"]
        updates = float(w.get("updates") or 0.0)
        other = sorted(d for d in L["ds"] if d != "Weights")
        cyc_a, cyc_r = sm["cycles"] or 0, rsm["cycles"] or 0
        # The arm's leakage at this level is (base + idle) x utilized_a x
        # cycles_a; the reference's is base x utilized_r x cycles_r. The base
        # is the same per instance-cycle in both (Accelergy is deterministic,
        # FINDINGS 3.5), so the toll is the arm's leakage minus the
        # reference's rescaled by BOTH ratios -- cycles AND utilized
        # instances. Rescaling by cycles alone assumed the two plans use the
        # same PEs; on mobilenet_v2's G576 depthwise layer (126 PEs on the
        # reference, 42 on recon4) that left base x 84 x cycles in the
        # residual and read as a multiplier of 41.99996 (2026-09-11).
        n_inst = float(w.get("utilized_instances") or L["instances"] or 1)
        n_inst_r = float((R["ds"].get("Weights") or {}).get("utilized_instances")
                         or R["instances"] or 1)
        leak_mult = None
        leak_delta_pJ = None
        if L.get("leakage_pJ") is not None and R.get("leakage_pJ") is not None and cyc_a and cyc_r:
            ref_base_here = R["leakage_pJ"] * (cyc_a / cyc_r) * (n_inst / n_inst_r)
            leak_delta_pJ = L["leakage_pJ"] - ref_base_here
            leak_mult = leak_delta_pJ / (bump["leak_delta_pj"] * cyc_a)
        mac_a = next((k for k in lv if "Compute" in lv[k]["ds"]), None)
        mac_r = next((k for k in rlv if "Compute" in rlv[k]["ds"]), None)
        pes_a = lv[mac_a]["utilized_instances"] if mac_a else None
        pes_r = rlv[mac_r]["utilized_instances"] if mac_r else None
        dram_wb = (lv.get("DRAM") or {}).get("word_bits")
        # the stats-side split: the level's printed Weights energy minus what
        # the same counts cost at the UN-bumped prices (5.3), x repeat count
        prices = tlmod.ert_prices(
            yaml.safe_load((pathlib.Path(path).parent / tlmod.ERT_NAME).read_text()))
        base_p = {a_: prices[(bump["level"], a_)] for a_ in ("read", "write", "update")
                  if (bump["level"], a_) in prices}
        base_p[bump["action"]] = base_p[bump["action"]] - bump["access_delta_pj"]
        # The multiplier is judged in pJ, against the two printed leakage
        # totals' precision: Timeloop prints `Leakage energy (total)` to 0.01
        # pJ, so on a small depthwise layer (42 PEs x 60k cycles) the quotient
        # lands at 41.99996 and a bare 1e-6 relative test refuses a multiplier
        # that IS the utilized count (mobilenet_v2 G576, 2026-09-11).
        leak_ok = (leak_delta_pJ is not None and _close(
            leak_delta_pJ, bump["leak_delta_pj"] * n_inst * cyc_a,
            abs_tol=0.01 * (1.0 + cyc_a / cyc_r)))
        bs = float(L["block_size"] or 1)
        at_base = ((float(w.get("reads") or 0.0) * base_p.get("read", 0.0)
                    + float(w.get("fills") or 0.0) * base_p.get("write", 0.0)
                    + updates * base_p.get("update", 0.0)) * n_inst / bs)
        rep = counts_by_shape.get(shape, 1.0)
        stats_access += (float(w["energy_pJ"]) - at_base) * rep
        if leak_delta_pJ is not None:
            stats_leak += leak_delta_pJ * rep
        edp_a = float(sm["energy_uJ"] or 0.0) * float(cyc_a or 0)
        edp_r = float(rsm["energy_uJ"] or 0.0) * float(cyc_r or 0)
        row = dict(shape=shape, nest_identical=nests[shape], weights_updates=updates,
                   other_dataspaces_on_level=other, leak_multiplier=leak_mult,
                   leak_multiplier_expected=n_inst, leak_multiplier_ok=leak_ok,
                   leak_delta_pJ=leak_delta_pJ,
                   leak_multiplier_rule=("Timeloop bills leak x UTILIZED instances x "
                                         "cycles (buffer.cpp FinalizeBufferEnergy, "
                                         "power-gated per instance)"),
                   declared_instances=L["instances"], utilized_instances=n_inst,
                   utilized_instances_reference=n_inst_r,
                   pes_used=pes_a, pes_used_reference=pes_r,
                   pes_differ=(pes_a != pes_r), dram_word_bits=dram_wb,
                   level_word_bits=L["word_bits"], block_size=L["block_size"],
                   cycles=cyc_a, cycles_reference=cyc_r,
                   timeloop_edp_uJ_cycles=edp_a, timeloop_edp_uJ_cycles_reference=edp_r,
                   timeloop_edp_ratio_arm_over_reference=(edp_a / edp_r) if edp_r else None,
                   vector_access_energy_source=L["source"],
                   energy_uJ=sm["energy_uJ"], energy_uJ_reference=rsm["energy_uJ"])
        rows.append(row)
        if updates:
            problems.append(f"{shape}: Scalar updates for Weights at {bump['level']} = "
                            f"{updates:g}, not 0 -- put the toll on `update` too or stop (RULE 2)")
        if other:
            problems.append(f"{shape}: {bump['level']} also holds {other}; the toll on its "
                            f"`{bump['action']}` would be billed to them as well")
        if not leak_ok:
            problems.append(f"{shape}: leak multiplier {leak_mult} != utilized instances "
                            f"{n_inst:g} of {bump['level']} (declared {L['instances']}; "
                            f"RULE 3: Timeloop bills leak x UTILIZED instances x cycles, "
                            f"buffer.cpp FinalizeBufferEnergy -- verify the multiplier)")
        if dram_wb != cfg.weight_bits:
            problems.append(f"{shape}: DRAM Word bits {dram_wb} != {cfg.weight_bits} "
                            f"(RULE 1: DRAM is never narrowed)")
        if L["source"] != "ERT":
            problems.append(f"{shape}: {bump['level']} vector access energy source is "
                            f"{L['source']!r}, not ERT -- the supplied table was not billed")
    if problems:
        raise SystemExit(f"{placement.key}'s own mapping fails its guards:\n  "
                         + "\n  ".join(problems))
    # ---- PE utilisation: REPORTED per shape, never refused (see docstring) ----
    differ = [r_ for r_ in rows if r_["pes_differ"]]
    pe_report = {
        "shapes_total": len(rows), "shapes_differ": len(differ),
        "per_shape": {r_["shape"]: {"pes_arm": r_["pes_used"], "pes_reference": r_["pes_used_reference"],
                                    "cycles_arm": r_["cycles"], "cycles_reference": r_["cycles_reference"],
                                    "timeloop_edp_ratio_arm_over_reference":
                                        r_["timeloop_edp_ratio_arm_over_reference"],
                                    "nest_identical": r_["nest_identical"]} for r_ in differ},
        "rule": ("PEs used may differ between the arm's OWN EDP-optimal plan and the "
                 "reference plan; the difference is the arm's configuration (datawidth q "
                 "on its narrowed levels) acting through the mapper, reported here and "
                 "in the manifest's title_caveats rather than refused (2026-09-11; "
                 "prompt_6 10.9's 168/168 rule came from one array-filling layer)")}
    if differ:
        print(f"  [note] {placement.key}: PEs used differ from the reference on "
              f"{len(differ)}/{len(rows)} shape(s) -- reported, not refused:")
        for r_ in differ:
            print(f"         {r_['shape']}: PEs {r_['pes_used']} vs {r_['pes_used_reference']}, "
                  f"cycles x{r_['cycles'] / r_['cycles_reference']:.3f}, Timeloop EDP "
                  f"x{r_['timeloop_edp_ratio_arm_over_reference']:.3f} of the reference plan")

    # ---- prompt_6 6: quantisation delivers 8/q, never N/K ---------------------
    q = acfg.weight_datawidth
    want = cfg.weight_bits / q
    narrow = tuple(bump["narrow_levels"])
    cap_ref, cap_arm = _capacity_of(ref_paths, narrow), _capacity_of(paths_a, narrow)
    got = (cap_arm / cap_ref) if cap_ref else 0.0
    if cap_ref and abs(got - want) > 0.05 * want:
        raise SystemExit(
            f"{placement.key}: {narrow[0]} reports Effective size {cap_arm:,} against the "
            f"reference's {cap_ref:,} -- x{got:.4f}, but a quantisation arm at q = {q} "
            f"delivers exactly {cfg.weight_bits}/q = {want:.4f} (prompt_6 6). The "
            f"`datawidth` edit missed the level, or the plan is not this arm's.")
    capacity = {"level": narrow[0] if narrow else "?", "reference_weights_per_instance": cap_ref,
                "arm_weights_per_instance": cap_arm, "delivered_factor": got,
                "wanted_factor": want, "q": q,
                "rule": "a quantisation arm delivers weight_bits/q, never N/K (prompt_6 6)"}

    # ---- the weight path, reconciled with the arm's record BEFORE the split ---
    wpath_a = reconmod.weight_path(acfg, arch, model, ases.models[model], paths_a)
    reconciles, recon_check = reconmod.cross_check(
        acfg, arch, wpath_a, raw_a.base_w.reindex(base_cats, fill_value=0.0))

    # ---- 5.3: the split -- MOVE the toll, do not add it --------------------
    stage_def = next(s_ for s_ in reconmod.stages_for(arch, acfg)
                     if s_.key == placement.site_stage)
    st = wpath_a.stages[placement.site_stage]
    split = ert_split(bump, st, wpath_a.cycles)
    for name, stats_side, evaluator in (("access", stats_access, split["access_toll_pJ"]),
                                        ("leak", stats_leak, split["leak_toll_pJ"])):
        if not _close(stats_side, evaluator):
            raise SystemExit(
                f"{placement.key}: the {name} split does not reconcile -- stats-side "
                f"{stats_side:.6f} pJ (printed level energy minus the same counts at the "
                f"un-bumped prices) vs evaluator {evaluator:.6f} pJ (prompt_6 5.3, 1e-6)")
    split.update(stats_side_access_pJ=stats_access, stats_side_leak_pJ=stats_leak,
                 reconciled_rel_tol=1e-6)
    cat = reconmod._category_of(stage_def, acfg)
    base_series = raw_a.base.reindex(base_cats, fill_value=0.0).copy()
    base_w = raw_a.base_w.reindex(base_cats, fill_value=0.0).copy()
    # Only the ACCESS toll is inside the level's billed energy; the leak toll
    # is a separate Timeloop line the raw record does not aggregate, so it is
    # charged by the evaluator and must not be subtracted here.
    base_series[cat] -= split["access_toll_pJ"]
    base_w[cat] -= split["access_toll_pJ"]
    st.energy_pJ -= split["access_toll_pJ"]
    st.switch_pJ -= split["access_toll_pJ"]
    if st.energy_pJ <= 0:
        raise SystemExit(f"{placement.key}: moving the access toll out of {cat} left "
                         f"{placement.site_stage} at {st.energy_pJ:.3f} pJ; the split is wrong")
    split["moved_out_of_category"] = cat
    split["moved_pJ"] = split["access_toll_pJ"]
    split["leak_in_bill"] = False
    split["leak_note"] = ("Timeloop prints leakage per level OUTSIDE the per-dataspace "
                          "energies and the raw record aggregates only those, so the "
                          "idle term is not in the bill: verified against the stats' "
                          "leakage delta (stats_side_leak_pJ) and charged by the evaluator")

    identical = bool(nests) and all(nests.values())
    print(f"       loop nest {'IDENTICAL to' if identical else 'DIFFERENT from'} the reference "
          f"on {sum(nests.values())}/{len(nests)} shape(s); cycles {wpath_a.cycles:,.0f}; "
          f"moved {split['access_toll_pJ'] / 1e6:.4f} uJ (access) out of {cat} into "
          f"Reconstruction; leak {split['leak_toll_pJ'] / 1e6:.3f} uJ ({split['engines']:.2f} "
          f"engines cycle-weighted, of {split['engines_declared']:g} declared) verified "
          f"against the stats' leakage and charged by the evaluator")
    return ErtArmView(
        placement=placement, cfg=acfg, raw=raw_a, wpath=wpath_a,
        base_series=base_series, base_w=base_w, cycles=float(wpath_a.cycles),
        fingerprint=fp, variant=variant, bump=bump, split=split,
        nest_identical=identical, per_shape_nest_identical=nests,
        read_back=read_back, checks={"per_shape": rows, "pe_utilization": pe_report},
        capacity=capacity,
        reconciles=reconciles, recon_check=recon_check, stats_paths=paths_a)


def ert_checks(builder, cfg, arch, raw, views, results, base_components,
               emb_components, e_parity, pricing, newly_mapped=0):
    """prompt_6 10 -- the checklist, recorded on the result file, one per arm."""
    frac = cfg.code_k / cfg.code_n
    space_ok, space = reconmod.validate_placement_space(arch, cfg)
    builder.check("placement_space_covers_the_whole_weight_path", space_ok, space)
    stages = reconmod.stages_for(arch, cfg)
    builder.check(
        "ert_arms_are_derived_from_the_placement_records", True,
        {"arms": sorted(views),
         "per_placement": {p.key: dict(zip(("injectable", "why"),
                                            reconmod.ert_injectable(p, stages)))
                           for p in reconmod.placements_for(arch, cfg)},
         "rule": "prompt_6 3.3: never `if key == ...`"})
    ref_macs = float((getattr(raw, "mac", None) or {}).get("macs", 0.0) or 0.0)
    dram_ref_emb = emb_components.get("DRAM", 0.0)
    dram_w_ref = float(raw.base_w.get("DRAM", 0.0))
    by_key = {r.placement.key: r for r in results}
    for key, v in views.items():
        res = by_key.get(key)
        macs = float((getattr(v.raw, "mac", None) or {}).get("macs", 0.0) or 0.0)
        builder.check(f"ert_{key}_same_workload_as_the_reference",
                      ref_macs > 0 and math.isclose(ref_macs, macs, rel_tol=1e-9, abs_tol=1.0),
                      {"reference_computes": ref_macs, "arm_computes": macs})
        builder.check(f"ert_{key}_stored_ert_read_back_matches_the_bar", True,
                      {"per_shape": v.read_back, "bump": v.bump,
                       "rule": "RULE 4.4.5 defence 3, checked against the reference entry's table"})
        rows = v.checks["per_shape"]
        builder.check(f"ert_{key}_scalar_updates_zero_at_the_patched_level",
                      all(not r_["weights_updates"] for r_ in rows),
                      {r_["shape"]: r_["weights_updates"] for r_ in rows})
        builder.check(f"ert_{key}_leak_multiplier_equals_the_utilized_instances",
                      all(r_["leak_multiplier_ok"] for r_ in rows),
                      {"rule": ("Timeloop bills leak x UTILIZED instances x cycles, power-gating "
                                "each unused instance (buffer.cpp FinalizeBufferEnergy); the "
                                "idle engines of a storage site are therefore its utilized "
                                "instances per layer, declared count stated beside"),
                       "tolerance": "0.01 pJ per printed leakage total (Timeloop prints to 0.01 pJ), else 1e-6 relative",
                       "per_shape": {r_["shape"]: {"multiplier": r_["leak_multiplier"],
                                                   "utilized_instances": r_["leak_multiplier_expected"],
                                                   "declared_instances": r_["declared_instances"],
                                                   "ok": r_["leak_multiplier_ok"]}
                                     for r_ in rows}})
        builder.check(f"ert_{key}_pes_used_vs_reference_reported_per_shape", True,
                      v.checks["pe_utilization"])
        builder.check(f"ert_{key}_dram_datawidth_is_8_on_every_arm",
                      all(r_["dram_word_bits"] == cfg.weight_bits for r_ in rows),
                      {r_["shape"]: r_["dram_word_bits"] for r_ in rows})
        builder.check(f"ert_{key}_capacity_delivered_is_8_over_q", True, v.capacity)
        builder.check(f"ert_{key}_attribution_split_reconciles", True, v.split)
        builder.check(f"ert_{key}_loop_nest_verdict", True,
                      {"identical_to_reference_on_every_shape": v.nest_identical,
                       "per_shape": v.per_shape_nest_identical,
                       "cycles": v.cycles, "reference_cycles": float(raw.cycles or 0),
                       "meaning": ("informational: if IDENTICAL, the ERT toll changed nothing "
                                   "FOR THIS ARM -- the residual pricing difference is not a "
                                   "result (prompt_6 4.4.4). Never averaged across arms.")})
        builder.check(f"ert_{key}_weight_path_reconciles_with_its_own_record",
                      v.reconciles, v.recon_check)
        if res is not None and res.status == "evaluated":
            # RULE 4: the bar is ITS OWN plan's DRAM bill minus K/N of ITS OWN
            # weight term. The bill may differ from the reference's beyond the
            # weight credit -- an arm whose nest changed moves different
            # activation traffic too (mobilenet_v2 recon2, 2026-09-11: 198 uJ
            # less input/output DRAM on its own plan) -- so the expectation is
            # built from the arm's bill, and that difference is recorded.
            dram_w_arm = float(v.base_w.get("DRAM", 0.0))
            dram_arm_bill = float(v.base_series.get("DRAM", 0.0))
            expected = dram_arm_bill - dram_w_arm * (1.0 - frac)
            got = res.components.get("DRAM", 0.0)
            builder.check(f"ert_{key}_dram_credit_equals_its_own_reads_times_K_over_N",
                          math.isclose(got, expected, rel_tol=1e-9, abs_tol=abs(expected) * 1e-9 + 1e-3),
                          {"dram_pJ": got, "expected_dram_pJ": expected,
                           "arm_dram_bill_pJ": dram_arm_bill, "arm_dram_weight_pJ": dram_w_arm,
                           "reference_dram_bill_pJ": dram_ref_emb, "reference_dram_weight_pJ": dram_w_ref,
                           "activation_dram_difference_arm_minus_reference_pJ":
                               (dram_arm_bill - dram_w_arm) - (dram_ref_emb - dram_w_ref),
                           "reference_dram_weight_reads": float(raw.dram_w_reads),
                           "arm_dram_weight_reads": float(v.raw.dram_w_reads),
                           "rule": ("bar DRAM = the arm's OWN DRAM bill - (1 - K/N) x its own "
                                    "weight DRAM term (RULE 4); an arm whose nest differs may "
                                    "also move different activation traffic, recorded above")})
    builder.check(
        "narrowing_ownership_exactly_one_owner_per_narrow_stop",
        all(r_["ok"] for res in results if res.status == "evaluated"
            for r_ in res.detail.get("narrowing_ownership", {}).get("stops", [])),
        {res.placement.key: res.detail.get("narrowing_ownership", {}).get("stops", [])
         for res in results if res.status == "evaluated"})
    builder.check(
        "evaluation_only_rerun", bool(cfg.from_cache) or newly_mapped == 0,
        {"ECC_FROM_CACHE": bool(cfg.from_cache), "newly_mapped_shapes": newly_mapped,
         "reference_fingerprint": archmod.arch_fingerprint(arch, cfg),
         "arm_fingerprints": {k: v.fingerprint for k, v in views.items()}})
    ok, detail = baseline_dram.reference_bars_agree(
        base_components, emb_components, e_parity, pricing, dram_w_reads=raw.dram_w_reads)
    builder.check("reference_bars_match_tasks_1_and_2", ok, detail)


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
          f"{'DRAM saved uJ':>14s} {'recon uJ':>12s} {'of it idle':>12s} {'engines':>8s} "
          f"{'N_rec':>16s}")
    for r in rows:
        if r["status"] != "evaluated":
            print(f"    {r['label'][:26]:26s} {'UNSUPPORTED':>13s}   {r['reason'][:60]}")
            continue
        vb = (base_total - r["total"]) / base_total * 100 if base_total else 0.0
        ve = (emb_total - r["total"]) / emb_total * 100 if emb_total else 0.0
        print(f"    {r['label'][:26]:26s} {r['total'] / 1e6:13,.3f} {vb:8.2f}% "
              f"{ve:8.2f}% {r['dram_saved_pJ'] / 1e6:14,.3f} "
              f"{r['recon_pJ'] / 1e6:12,.3f} "
              f"{r.get('idle_pJ', 0.0) / 1e6:12,.3f} {float(r.get('engines', 0)):8.2f} "
              f"{r['n_cw']:16,.0f}")
    print(f"    (vs base / vs emb are SAVINGS: a negative number costs more "
          f"than the reference; DRAM saved is the DRAM energy the bar "
          f"REMOVED against the embedded reference; recon uJ = incremental x "
          f"events + idle x engine-cycles over {wpath.cycles:,.0f} cycles, engines = "
          f"cycle-weighted mean of the instances that LEAK, i.e. the utilized ones, "
          f"RULE 3)")


def _report_narrowing(results):
    """RULE 1 on the console: per bar, per narrow storage stop, who narrowed
    it and from what measurement (prompt_6 10, item 2)."""
    print(f"\n    narrowing ownership (RULE 1), from each bar's own stats:")
    for res in results:
        if res.status != "evaluated":
            continue
        stops = res.detail.get("narrowing_ownership", {}).get("stops", [])
        if not stops:
            print(f"      {res.placement.key:8s} no on-chip storage stop carries narrow weights")
            continue
        parts = [f"{r['stage']}: Word bits {r['measured_word_bits']} -> "
                 f"{r['owner']} (x{r['applied_scale']:.4f})" for r in stops]
        print(f"      {res.placement.key:8s} " + "; ".join(parts))


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
    # prompt_6 (ECC_RECON_ERT_AWARE=1) has FOUR worlds instead of two: the
    # reference plan for baseline/embedded and the post-processed bars, and
    # ONE PLAN PER ERT ARM. The single `p_*` set is replaced by a per-bar
    # lookup (RULE 4.4.3); the depth dilation is not used under it.
    aware = bool(getattr(cfg, "recon_ert_aware", False))
    dil = (dilated_view(cfg, ses, arch, model, cats, raw)
           if (cfg.recon_optimizer and not aware) else None)
    ert_keys = {p.key for p in reconmod.ert_arms(arch, cfg)} if aware else set()
    # the run-wide reference record for the lines that describe the run as a
    # whole (the per-bar plans are chosen below)
    p_raw = dil.raw if dil else raw

    gran = reconmod.Granularity(cfg.code_n, cfg.code_k, cfg.weight_bits,
                                cfg.recon_granularity)
    packing = reconmod.Packing(cfg.recon_packing, cfg.weight_bits,
                               cfg.code_k, cfg.code_n)
    # prompt_6 RULE 1: THE ON-CHIP NARROWING HAS EXACTLY ONE OWNER PER BAR PER
    # STAGE, decided from each bar's own measured Word bits inside
    # `evaluate_placement` (a hard stop on a stop narrowed twice or by nobody).
    # The ownership rows are printed below, once the bars exist.
    # prompt_6 RULE 3: two terms, two denominators. The idle term is charged
    # per cycle of the plan a bar is billed from, times its engine count.
    recon_pj, recon_idle_pj, recon_prov = load_recon_energy(cfg)

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
    views = {p.key: ert_aware_view(cfg, ses, arch, model, cats, raw, paths, p)
             for p in placements if p.key in ert_keys}

    def plan(p):
        """RULE 4: the (wpath, weight bill, bill, raw, cycles) THIS bar is billed
        from -- its own ERT arm, the Task 4 dilated plan, or the reference."""
        v = views.get(p.key)
        if v is not None:
            return v.wpath, v.base_w, v.base_series, v.raw, v.cycles, "own mapping (ERT)"
        if dil is not None:
            return dil.wpath, dil.base_w, dil.base_series, dil.raw, dil.wpath.cycles, "dilated"
        return wpath, base_w, base_series, raw, wpath.cycles, "reference"

    results, plans = [], {}
    for p in placements:
        p_wpath, p_base_w, p_base_series, p_raw, p_cycles, tag = plan(p)
        plans[p.key] = (p_raw, tag)
        # Under its own plan a bar issues its own DRAM reads, so it decodes
        # its own codeword count (the reference's would bill reads it never made).
        n_cw_p = p_raw.dram_w_reads / gran.weights_per_codeword
        decode_p = ((n_cw_p * cfg.decode_pj_emb)
                    if (cfg.decode_enabled and cfg.recon_placement_charges_decode) else 0.0)
        results.append(reconmod.evaluate_placement(
            cfg, arch, p, p_wpath, p_base_w, p_base_series, recon_pj, gran,
            packing, decode_pj=decode_p, recon_idle_pj=recon_idle_pj, cycles=p_cycles))
    # 5.3: what the evaluator charged as Reconstruction on an ERT bar must be
    # exactly what was moved out of the level -- a split, not an addition.
    for res in results:
        v = views.get(res.placement.key)
        if v is None or res.status != "evaluated":
            continue
        c = res.detail["reconstruction_counts"]
        for name, charged, moved in (
                ("access", c["reconstruction_energy_incremental_pJ"], v.split["access_toll_pJ"]),
                ("leak", c["reconstruction_energy_idle_pJ"], v.split["stats_side_leak_pJ"])):
            if not _close(charged, moved):
                raise SystemExit(
                    f"{arch}/{res.placement.key}: the {name} term the evaluator charged "
                    f"({charged:.6f} pJ) is not the stats-side amount ({moved:.6f} pJ: "
                    f"{'moved out of ' + v.split['moved_out_of_category'] if name == 'access' else 'the leakage delta Timeloop printed'}); "
                    f"the encoder would be counted twice or not at all (prompt_6 5.3)")
        v.split["evaluator_incremental_pJ"] = c["reconstruction_energy_incremental_pJ"]
        v.split["evaluator_idle_pJ"] = c["reconstruction_energy_idle_pJ"]

    # ---- the result file ---------------------------------------------------
    builder = ResultBuilder(cfg, ses.results, arch, model,
                            experiment=(EXPERIMENT_ERT if aware else
                                        EXPERIMENT_TASK4 if dil else EXPERIMENT),
                            fixed_mapping=not (dil or aware))
    mapping_ids = audit.mapping_ids_of(raw)
    # TASK 4 / prompt_6: a bar that comes from a DIFFERENT mapping must carry
    # THAT mapping's ids. A result file whose recon bars claimed the
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
        bar_raw, bar_tag = plans[p.key]
        v = views.get(p.key)
        if aware:
            bar_extra = ({"reconstruction_aware_mapping": ERT_NOTE, "plan": bar_tag,
                          "ert_arm": {"bump": v.bump, "fingerprint": v.fingerprint,
                                      "cache_variant": v.variant,
                                      "loop_nest_identical_to_reference": v.nest_identical,
                                      "per_shape_nest_identical": v.per_shape_nest_identical,
                                      "attribution_split": v.split,
                                      "capacity": v.capacity,
                                      "guards_per_shape": v.checks["per_shape"],
                                      "ert_read_back": v.read_back},
                          "dram_weight_reads_reference": float(raw.dram_w_reads),
                          "dram_weight_reads_this_arm": float(bar_raw.dram_w_reads),
                          "cycles_reference": float(raw.cycles or 0),
                          "cycles_this_arm": float(v.cycles)}
                         if v is not None else
                         {"plan": bar_tag, "fixed_mapping": FIXED_MAPPING_NOTE,
                          "note": ("post-processed on the reference plan; this boundary "
                                   "is not ERT-injectable (prompt_6 3.3)")})
            bar_ids = audit.mapping_ids_of(bar_raw)
        else:
            bar_extra = None
            bar_ids = placement_mapping_ids
        builder.add(Variant(
            p.variant, kind="reconstruction", status="evaluated",
            total_energy_pJ=res.total_pJ,
            energy_by_component_pJ={k: v for k, v in res.components.items()
                                    if v != 0.0},
            mapping_ids=bar_ids, label=label,
            extra=dict(res.detail, placement_key=p.key,
                       **(bar_extra if bar_extra is not None else
                          {"reconstruction_aware_mapping": TASK4_NOTE,
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
                     "idle_pJ": counts["reconstruction_energy_idle_pJ"],
                     "engines": counts["engines"],
                     "plan": bar_tag,
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
    if aware:
        ert_checks(builder, cfg, arch, raw, views, results, base_components,
                   emb_components, e_parity, pricing,
                   newly_mapped=getattr(mapper, "n_mapped", 0))
    elif dil:
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
    if aware:
        builder.approximate(ERT_NOTE)
        own = sorted(views)
        rest = [p.key for p in placements if p.key not in views]
        builder.warn(
            f"TWO MAPPING REGIMES ON ONE FIGURE: {', '.join(own) or 'none'} come from "
            f"their own mapping (encoder toll in the ERT); {', '.join(rest) or 'none'} "
            f"are post-processed on the reference plan. Each ERT bar's loop-nest "
            f"verdict is recorded on its own: "
            + "; ".join(f"{k}: " + ("IDENTICAL to the reference -- the ERT changed nothing "
                                    "for this arm" if v.nest_identical else "CHANGED")
                        for k, v in views.items()))
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
            "reconstruction_datapath_idle_pJ_per_cycle_per_engine": recon_idle_pj,
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
    _report_narrowing(results)
    path = builder.write()
    print(f"    -> {path}")
    return {"path": path, "rows": rows, "results": results, "wpath": wpath,
            "views": views, "ert_bars": set(views),
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
    want, q = reconmod.capacity_target(cfg)
    cap_ok = (cap["reference_weights_per_instance"] > 0
              and abs(cap["delivered_factor"] - want) <= 0.05 * want)
    builder.check(
        "reconstruction_arm_has_8_over_q_more_weight_capacity", cap_ok,
        dict(cap, reference_scale=dil.ref_scale, dilated_scale=dil.scale,
             rule=(f"the reduced representation stores weights at q = {q} whole "
                   f"bits, so it holds {cfg.weight_bits}/q = {want:.4f} times as many "
                   f"in the same silicon (N/K = {nk:.4f} is the ideal rate, not the "
                   f"room a whole-bit word delivers -- prompt_6 6); the mapper must "
                   f"have been given exactly that much more room at the weight "
                   f"level -- no more (unearned) and no less (understated)")))

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
        if key in out.get("ert_bars", ()):
            # prompt_6 9: mark which bars came from their own mapping
            lines.append("own mapping (ERT)")
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
            "idle_engines_cycle_weighted": "", "idle_engines_declared": "",
            "pe_shapes_differing_from_reference": "",
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
                idle_engines_cycle_weighted=c["engines"],
                idle_engines_declared=c["engines_declared"],
                pe_shapes_differing_from_reference=(
                    (out.get("views") or {}).get(res.placement.key).checks["pe_utilization"]["shapes_differ"]
                    if (out.get("views") or {}).get(res.placement.key) is not None else ""),
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
                       idle_engines_cycle_weighted="", idle_engines_declared="",
                       pe_shapes_differing_from_reference="",
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


#: What the multi-panel placement figure's heading said under its title until
#: 2026-09-11. The heading is one line since; this travels in the manifest's
#: `title_caveats` with the rest (`Config.recon_caveats`).
PANEL_NOTE = ("each panel is one design's OWN weight path, measured against its "
              "OWN two reference bars; the panels share a legend and a unit, not "
              "a y limit or an x axis")


def _title(cfg, panels):
    """The ONE-LINE heading. Everything it used to say below that line is
    `cfg.recon_caveats()`, written to the manifest by `run()`."""
    return cfg.recon_title() if len(panels) == 1 else cfg.recon_panel_title()


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
    title = _title(cfg, panels)
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
        panel_note=None)             # PANEL_NOTE goes to the manifest (run())
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
    # The heading is one line (2026-09-11). What it said below that line until
    # then -- the mapping regime with its arms, the DRAM model and price, the
    # static DRAM terms, the encoder site, the MAC denominator, the layer
    # scope, and the panel rule on a multi-design figure -- is RECORDED here,
    # beside the figure, instead of drawn on it.
    caveats = cfg.recon_caveats(mac_ert_pj=panels[0]["mac_ert_pj"])
    if len(panels) > 1:
        caveats.append(PANEL_NOTE)
    # PE utilisation per ERT arm (reported, never refused -- see ert_aware_view)
    for a, o in outs.items():
        for key, v in sorted((o.get("views") or {}).items()):
            pe = v.checks.get("pe_utilization") or {}
            if pe.get("shapes_differ"):
                names = ", ".join(f"{s} ({d['pes_arm']} vs {d['pes_reference']} PEs)"
                                  for s, d in pe["per_shape"].items())
                caveats.append(
                    f"{cfg.arch_label(a).replace(chr(10), ' ')} {key}: its own EDP-optimal "
                    f"plan uses a different PE count than the reference plan on "
                    f"{pe['shapes_differ']} of {pe['shapes_total']} layer shapes -- {names}; "
                    f"idle is charged on the instances that leak (utilized, per layer), "
                    f"as Timeloop bills it")
            elif pe:
                caveats.append(f"{cfg.arch_label(a).replace(chr(10), ' ')} {key}: PEs used "
                               f"identical to the reference plan on all {pe['shapes_total']} "
                               f"layer shapes")
    ses.finish(figs, csv, groups, extra={
        "title": _title(cfg, panels),
        "title_caveats": caveats,
        "recon_decode_site": cfg.recon_decode_site,
        "dram_pj_per_bit": cfg.dram_pj_per_bit,
        "dram_cost_provenance": cfg.dram_cost_note,
        "dram_static_terms": cfg.dram_static_note,
        "panel_per_architecture": list(cfg.archs),
        "dram_term": {a: o["wpath"].dram_term() for a, o in outs.items()},
    })

    print("=" * 78)
    if cfg.recon_optimizer and getattr(cfg, "recon_ert_aware", False):
        print(f"prompt_6 reconstruction-AWARE MAPPING, encoder in the objective: {model}, "
              f"one panel per architecture.")
        print(f"  reference plan      : the published 8-bit chip, no toll")
        print(f"  ERT arms            : one mapping each, datawidth q on the storage levels "
              f"the boundary narrows, its toll in the ERT; the other boundaries are "
              f"post-processed on the reference plan")
    elif cfg.recon_optimizer:
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
    for arch in cfg.archs:
        views = outs[arch].get("views") or {}
        if not views:
            continue
        rest = [r["key"] for r in outs[arch]["rows"] if r["key"] not in views]
        print(f"  prompt_6, {cfg.arch_label(arch).replace(chr(10), ' ')}: TWO MAPPING REGIMES "
              f"on this panel -- {', '.join(sorted(views))} from their OWN mapping "
              f"(marked 'own mapping (ERT)'); {', '.join(rest)} on the reference plan.")
        for key, v in views.items():
            verdict = ("IDENTICAL to the reference -- the ERT changed nothing for this arm"
                       if v.nest_identical else "DIFFERENT from the reference")
            n_same = sum(v.per_shape_nest_identical.values())
            print(f"    {key}: {tlmod.describe_bump(v.bump)}")
            print(f"      loop nest {verdict} ({n_same}/{len(v.per_shape_nest_identical)} "
                  f"shapes identical); cycles {v.cycles:,.0f}; fp {v.fingerprint}")
            print(f"      moved out of {v.split['moved_out_of_category']}: access "
                  f"{v.split['access_toll_pJ'] / 1e6:.4f} uJ ({v.split['counter']} x E_w), "
                  f"leak {v.split['leak_toll_pJ'] / 1e6:.3f} uJ ({v.split['engines']:.2f} engines x "
                  f"{v.cycles:,.0f} cycles); both reconcile to 1e-6")
    if len(cfg.archs) > 1:
        print("Each design is measured against ITS OWN reference bars, so a")
        print("percentage on one panel says nothing about the other.")
    if cfg.recon_optimizer and getattr(cfg, "recon_ert_aware", False):
        pass                    # the per-arm block above said it all
    elif cfg.recon_optimizer:
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
