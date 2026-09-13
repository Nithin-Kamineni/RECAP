"""Task 4: the correction a DILATED mapping needs before its energy can be read.

Capacity dilation expresses the reconstruction arm's extra effective capacity as
`depth x N/K`, and Accelergy then prices the deeper array as a bigger one --
measured at 1.18-1.46x per access on these designs. So an energy-objective
search has a positive reason to leave the extra room unused, which is a bias
AGAINST the hypothesis. `capacity_dilation_correction()` takes the energy back to
the declared array; the MAPPING cannot be corrected, which is why every Task 4
number is a LOWER BOUND and the `ERT x` column reports the factor.

THE MECHANISM IS WITHDRAWN as the live one (FINDINGS 2.4): the capacity knob
moves in 2x steps and N/K = 1.6154 sits below the step, so the answer was
negative and unfair besides. The live mechanism is `datawidth: q` at a FIXED
depth, out of THE WIDTH TABLE (`physics/widths.py`). This module stays because
the depth sweep and `study/dilation*.py` still read it, and because withdrawing
a mechanism is not the same as deleting the evidence.

ProjectRestructure phase 3 cut this out of `recon.py`; `study/dilation.py` is
the Task 4 driver that uses it.
"""
from __future__ import annotations

import pathlib

from ..physics import widths

from ..arch.placements import stages_for


# ===========================================================================
#  TASK 4: capacity dilation -- the correction the dilated mapping needs
# ===========================================================================
def capacity_dilation_scale(cfg):
    """The reconstruction arm's effective weight capacity, as a scale factor.

    The reference arm's physical buffer is `cfg.weight_capacity_scale` times
    the declared design (1.0 normally; below 1 when the study is shrinking the
    design to find the regime where capacity binds at all). The reconstruction
    arm stores the same weights at K/N of full width in that SAME silicon, so
    it holds N/K times as many of them.
    """
    # Rounded to the same four decimals `config.load_config()` quantises the
    # reference scale to, so the derived capacity has ONE spelling and
    # therefore one cache directory. See the comment there.
    return round(cfg.weight_capacity_scale * cfg.code_n / cfg.code_k, 4)


def capacity_target(cfg):
    """prompt_6 6: the weight room a re-planned arm must show, `(factor, q)`.

    The reduced representation is `datawidth: q` with `q = round(weight_bits x
    K/N)` a WHOLE number of bits, so the room actually delivered is
    `weight_bits / q` -- 2.000 at BCH(63,30), 1.333 at BCH(63,51) -- and NEVER
    N/K (2.100, 1.235). Checking against N/K refused two codes outright and
    passed the other four by luck (within the 5 % slack); this target is exact
    for every code, so the slack goes back to catching real faults.
    """
    q = widths.declared_datawidth(cfg.code_n, cfg.code_k, cfg.weight_bits)
    return cfg.weight_bits / q, q


def capacity_dilation_correction(ref_stats_dir, dil_stats_dir, prefixes):
    """The DECLARED array's per-access energies, for re-pricing a dilated level.

    THE PROBLEM. To ask the mapper what it would do with N/K more weight room,
    the room has to be in the YAML it reads, so `archs._scale_weight_capacity`
    multiplies that level's `depth:`. Accelergy then costs the level from its
    declared geometry and prices it as a physically LARGER array. The
    reconstruction arm's array is not larger. It is the same array holding
    narrower values, which is the whole premise. Charging it CACTI's cost for
    the bigger array would bill the design for silicon it does not have, and
    would do so in the direction that makes reconstruction look worse.

    READS AND WRITES ARE CORRECTED SEPARATELY, because they do not scale
    together. On `eyeriss_like` at x1.6154 the weight scratchpad's read energy
    goes 0.783354 -> 0.970977 pJ (x1.2395) and its write energy 1.25362 ->
    1.72943 (x1.3796): a single read-derived ratio leaves the write half
    under-corrected, which showed up as a 0.22 pp residual between Task 4 and
    Task 3 on a mapping that was byte-identical. So the correction re-prices
    from the access counts -- `reads x e_read + writes x e_write` at the
    DECLARED energies -- and `correct_level_energy()` reconciles the same
    arithmetic at the DILATED energies against Timeloop's own total before
    trusting it.

    WHAT IT CANNOT FIX, AND WHY THAT IS THE CONSERVATIVE DIRECTION. The mapper
    optimised against the DEARER array, so a level that got materially more
    expensive per access was one the search had a reason to avoid -- the found
    mapping is therefore no better than the one a correctly-priced search would
    have found, and the Task 4 saving this yields is a LOWER bound.

    Returns a dict; `ok` False leaves the energy uncorrected and says why,
    rather than scaling by a guess.
    """
    e_rd_ref, e_wr_ref, prov_ref = storage_access_pj(ref_stats_dir, prefixes)
    e_rd_dil, e_wr_dil, prov_dil = storage_access_pj(dil_stats_dir, prefixes)
    level = prefixes[0] if prefixes else "?"
    if not (e_rd_ref and e_wr_ref and e_rd_dil and e_wr_dil):
        return {"ok": False, "level": level,
                "provenance": (
                    "NOT corrected: the declared and dilated ERTs are not both "
                    f"readable ({prov_ref} / {prov_dil}). The dilated level is "
                    "left at Accelergy's cost for the LARGER array, which "
                    "understates the reconstruction arm.")}
    return {
        "ok": True, "level": level,
        "read_pJ_declared": e_rd_ref, "read_pJ_dilated": e_rd_dil,
        "write_pJ_declared": e_wr_ref, "write_pJ_dilated": e_wr_dil,
        "read_ratio_declared_over_dilated": e_rd_ref / e_rd_dil,
        "write_ratio_declared_over_dilated": e_wr_ref / e_wr_dil,
        "provenance": (
            f"dilated {level} re-priced at the DECLARED array's per-access "
            f"energies: read {e_rd_dil:.6f} -> {e_rd_ref:.6f} pJ "
            f"(x{e_rd_ref / e_rd_dil:.4f}), write {e_wr_dil:.6f} -> "
            f"{e_wr_ref:.6f} pJ (x{e_wr_ref / e_wr_dil:.4f}). The "
            f"reconstruction arm's array is the same silicon holding narrower "
            f"values, so Accelergy's cost for the deeper array is not its cost."),
    }


def correct_level_energy(corr, energy_pJ, reads, writes, tol=0.02):
    """`energy_pJ` re-priced at the declared array, from its own access counts.

    THE BLOCK SIZE CANCELS, AND THAT IS WHY THIS IS A RATIO. Accelergy's ERT
    prices one VECTOR access -- a whole physical word -- while Timeloop counts
    SCALAR accesses, one per value, so `reads x e_read` overstates a level's
    energy by exactly its block size (2 on `eyeriss_like`'s 16-bit,
    8-bit-datawidth weight scratchpad; 3 on Eyeriss v2's 24-bit one). Rebuilding
    an absolute energy therefore needs the packing, and getting it wrong is
    silent. Re-pricing as an access-weighted RATIO does not:

        corrected = energy x  (reads x e_read_declared + writes x e_write_declared)
                             ---------------------------------------------------
                              (reads x e_read_dilated  + writes x e_write_dilated)

    Both sums carry the same block size, so it divides out, and what is left is
    exactly "how much cheaper the declared array is for THIS mix of reads and
    writes". That matters because reads and writes do not scale together: on
    `eyeriss_like` at x1.6154 the read energy rises 1.2395x and the write energy
    1.3796x, so a read-derived ratio leaves the write half under-corrected.

    THE RECONCILIATION IS STILL DONE, on the one thing the ratio cannot check:
    that the ERT split describes this level at all. `rebuilt / energy_pJ` must
    come out as a whole number of values per word -- the block size. Anything
    else means the split and the level do not belong together, and the energy
    is returned UNCHANGED with the discrepancy named, the same refusal
    `evaluate_placement` makes before re-billing scratchpad reads from an
    unreconciled ERT split.

    Returns `(corrected_pJ, note)`.
    """
    if not corr.get("ok") or energy_pJ <= 0:
        return energy_pJ, corr.get("provenance", "not corrected")
    dil = reads * corr["read_pJ_dilated"] + writes * corr["write_pJ_dilated"]
    ref = reads * corr["read_pJ_declared"] + writes * corr["write_pJ_declared"]
    if dil <= 0:
        return energy_pJ, (f"NOT corrected: {corr['level']} reports no accesses "
                           f"to re-price ({reads:,.0f} reads, {writes:,.0f} writes)")
    per_word = dil / energy_pJ
    if per_word < 1.0 - tol or abs(per_word - round(per_word)) > tol * max(1.0, per_word):
        return energy_pJ, (
            f"NOT corrected: the dilated ERT split does not reproduce "
            f"Timeloop's {corr['level']} energy as a whole number of values "
            f"per physical word -- {reads:,.0f} reads + {writes:,.0f} writes "
            f"price at {dil:,.0f} pJ against Timeloop's {energy_pJ:,.0f} pJ, a "
            f"factor of {per_word:.4f}. Refusing to re-price from an "
            f"unreconciled split.")
    return energy_pJ * (ref / dil), (
        f"{corr['provenance']} Applied as an access-weighted ratio "
        f"x{ref / dil:.6f} over {reads:,.0f} reads and {writes:,.0f} writes; "
        f"the split reconciles with Timeloop at {round(per_word)} values per "
        f"physical word.")


def apply_capacity_correction(wpath, stage_key, corr):
    """Re-price one weight-path stage at the declared array. Returns what moved.

    Kept deliberately small and explicit: the correction touches exactly the
    stage whose level the dilation rewrote, and `cross_check()` is re-run
    against the corrected totals afterwards, so a correction that does not
    reconcile fails the run instead of quietly shifting a bar.
    """
    st = wpath.stages.get(stage_key)
    if st is None or not corr.get("ok"):
        return {"stage": stage_key, "corrected": False,
                "moved_pJ": 0.0, "note": corr.get("provenance", "no such stage")}
    before = st.energy_pJ
    after, note = correct_level_energy(corr, before, st.reads, st.fills)
    if after == before:
        return {"stage": stage_key, "corrected": False, "before_pJ": before,
                "after_pJ": before, "moved_pJ": 0.0, "note": note}
    scale = after / before
    st.energy_pJ = after
    st.switch_pJ *= scale
    st.wire_pJ *= scale
    return {"stage": stage_key, "corrected": True, "before_pJ": before,
            "after_pJ": after, "moved_pJ": after - before, "scale": scale,
            "note": note}


def dilated_levels(arch, cfg):
    """`(stage_key, prefixes)` for the stage the dilation rewrote, or None.

    The dilation scales the WEIGHT-carrying storage levels; the one whose
    per-access cost the correction has to undo is the innermost of them, which
    is also the stage every PE-local boundary sits at. A design whose weight
    levels are all `depth: 1` latches is not dilated at all and returns None,
    which is what makes `RECON_OPTIMIZER=True` refuse on it rather than draw a
    figure that claims a capacity effect it cannot have.
    """
    from ..arch import patch
    rows = [r for r in patch.weight_capacity_levels(arch, cfg) if not r["latch"]]
    if not rows:
        return None
    level = rows[-1]["level"]
    for stage in stages_for(arch, cfg):
        if stage.matches(level):
            return stage.key, stage.prefixes
    return None


def storage_access_pj(stats_dir, prefixes):
    """(read_pJ, write_pJ, provenance) for a storage level, from the design's ERT.

    Task 4's dilation correction needs this: to reprice a level whose read and
    write counts move apart, the two per-access energies have to be separable,
    and Timeloop's stats print one `Energy (total)` per dataspace. The split is
    taken from the same Accelergy ERT the mapper cache already holds beside
    every mapping. Returns `(None, None, why)` when the ERT is missing, so the
    caller can REFUSE rather than estimate.
    """
    ert = pathlib.Path(stats_dir) / "timeloop-mapper.ERT_summary.yaml"
    if not ert.exists():
        return None, None, f"no {ert.name} in the mapper cache"
    try:
        import yaml
        blob = yaml.safe_load(ert.read_text())
        for entry in (blob.get("ERT_summary", {}).get("table_summary") or []):
            name = str(entry.get("name", ""))
            # Strip the instance range BEFORE splitting on `.`: Accelergy writes
            # `system_top_level.weights_spad[1..192]`, and `[1..192]` contains
            # dots of its own, so splitting first leaves `192]`.
            bare = name.split("[")[0].split(".")[-1]
            if not any(bare == p or bare.startswith(p) for p in prefixes):
                continue
            acts = {str(a.get("name")): float(a.get("energy", 0.0))
                    for a in (entry.get("actions") or [])}
            if acts.get("read", 0.0) > 0 and acts.get("write", 0.0) > 0:
                return acts["read"], acts["write"], (
                    f"{ert.name}: {name} read={acts['read']} pJ "
                    f"write={acts['write']} pJ (Accelergy, this design's own ERT)")
        return None, None, f"no read+write energy for {prefixes} in {ert.name}"
    except Exception as exc:                                  # pragma: no cover
        return None, None, f"could not read {ert.name} ({type(exc).__name__})"


