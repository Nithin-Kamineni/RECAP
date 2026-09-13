"""The three prompt_2 tables: utilisation, per-level, and the convergence gate.

`utilisation_table()` -- one row per SWEPT CAPACITY, per design, per layer: what
was resident, what was declared, and what the refetch did in response.

`level_table()` -- one row per (arm, scale, MEMORY LEVEL). THE ARMS OF A PROMPT_2
PAIR EACH NEED THEIR OWN MAPPING, which is what `arm_configs()` builds: the
reference at 8-bit weights and the reconstruction arm at `datawidth: q`, each at
the width THE WIDTH TABLE gives it. `--levels` always prints the per-DATASPACE
table: a weight-only table once made a large win look like a null result
(FINDINGS 2.8).

`convergence_gate()` -- the same measurement at several mapper victory
conditions, so a difference that is really the search not having converged is
visible as one.

All three read caches that already exist and refuse a scale that has none.

ProjectRestructure phase 3 cut this out of `experiments/dilation.py`.
"""
from __future__ import annotations

import dataclasses
import pathlib

from ..arch import fingerprint
from .. import paths as pathsmod
from . import narrowing as narrowing_mod
from ..arch import workloads as workloadsmod

from .dilation import DilationRow, binding_level, capacity_verdict, sibling_fingerprints
from .dilation_cache import load, read_mapped_layer, weight_levels_of


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
                   "`bash hpc/tl.sh python3 -m eccenergy.report.dilation_view`.")
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
    variant = fingerprint.effective_variant(arch, cfg)
    fp = fingerprint.arch_fingerprint(arch, cfg)
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
                   "through `bash hpc/tl.sh python3 -m eccenergy.report.dilation_view`.")
        out.append("")
    narrowing = narrowing_mod.onchip_narrowing_audit(cfg)
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


