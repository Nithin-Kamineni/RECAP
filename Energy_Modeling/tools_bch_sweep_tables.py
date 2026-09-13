"""prompt_5's tables, across several BCH codes. READS CACHE ONLY.

Changes nothing and invokes no mapper. Every energy comes from the project's
OWN aggregation path -- `energy_mod.gather()` (which applies `noc_post.augment`,
the evaluator-only NoC terms Timeloop cannot be given), then
`apply_mac_override` and `apply_dram_override`, then `baseline_dram_mod.price()`.
So the totals are the ones `run.sh` would print, not a second arithmetic.

    the stats file's own total        231.560 uJ   (what dilation.py reads)
    + noc_post evaluator-only terms     8.906 uJ
    = the production total            240.466 uJ   (what a Raw record holds)

The DRAM K/N columns prompt_5's last section specifies are NOT in
`dilation.dataspace_table()` (its `k_over_n` is computed and unused), so they
are done here, with prompt_5's own arithmetic:

    dram_weight_pJ_at_price = raw.e_dram_w at ECC_DRAM_PJ_PER_BIT
    dram_kn_saving_pJ       = that x (1 - K/N)          RECON ROWS ONLY
    total_mapping_only      = the whole run at that DRAM price
    total_with_dram         = total_mapping_only - dram_kn_saving_pJ
    total_baseline          = embedded's total, its DRAM category repriced
                              70/40 by baseline_dram_mod.price()

SELF-CHECK FIRST. BCH(63,30) is already in the cache and its numbers are on the
record (progress.txt, 2026-09-10). Nothing else prints unless they come back.

    python3 kn_tables.py [out.csv]
"""
from __future__ import annotations

import csv
import os
import pathlib
import sys

sys.path.insert(0, "/blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling")

from eccenergy import config, paths as P
from eccenergy.arch import fingerprint
from eccenergy.physics import baseline_dram as baseline_dram_mod
from eccenergy.physics import widths
from eccenergy.study import energy as energy_mod
from eccenergy.arch import workloads as workloads_mod
from eccenergy.report import dilation_view as dilation                              # noqa: E402

CODES = [int(x) for x in os.environ.get("KN_CODES", "57 45 39 30").split()]
#: The depth scale to read. 1.0 is prompt_5's point (the YAML as it stands).
SCALE = round(float(os.environ.get("KN_SCALE", "1.0")), 4)
LAYER = "layer3.0.conv1"
ARCH = "eyeriss_like_wglb"

#: progress.txt, 2026-09-10, in uJ. The gate this script must pass.
RECORDED_K30 = {
    "baseline": 761.102, "embedded": 483.854,
    "recon_mapping_only": 336.571, "recon_with_dram": 287.138,
    "dram_baseline": 646.912, "dram_embedded": 369.664, "dram_recon": 191.781,
}


class CachedMapper:
    """The two attributes `energy_mod.gather()` needs, served from the cache.

    Deliberately NOT eccenergy.toolchain.invoke.Mapper: that one can decide to SOLVE a
    shape. This one returns None instead, so a missing entry is reported rather
    than queued.
    """

    def __init__(self, cfg, arch):
        self.cfg, self.arch = cfg, arch
        self.mappings = {}
        self.root = pathlib.Path(P.Results(cfg).mapper_cache(
            arch, fingerprint.effective_variant(arch, cfg), fingerprint.arch_fingerprint(arch, cfg),
            create=False))

    def stats_for(self, layer):
        p = self.root / layer.shape_name / "timeloop-mapper.stats.txt"
        return p if p.is_file() else None

    def summary(self):
        return f"cache {self.root.name}"


def _with_env(**kw):
    saved = dict(os.environ)
    os.environ.update({k: str(v) for k, v in kw.items()})
    return saved


def load_code(k):
    # No ECC_WEIGHT_WIDTH: THE WIDTH TABLE is automatic (code_widths.py).
    saved = _with_env(ECC_CODE_N=63, ECC_CONST_K=k, ECC_KS=k,
                      RECON_OPTIMIZER="False")
    try:
        cfg = config.load_config()
        q = widths.declared_datawidth(cfg.code_n, cfg.code_k, cfg.weight_bits)
        models, _ = workloads_mod.load_workload(cfg)
        layers = [l for l in models[cfg.models[0]] if l.name == LAYER]
        if not layers:
            raise SystemExit(f"{LAYER} is not in {cfg.models[0]}")
        arms = dilation.arm_configs(cfg, SCALE, q)

        out = {}
        for name in ("embedded", "recon"):
            c = arms[name]
            raw = energy_mod.gather(c, CachedMapper(c, ARCH), c.models[0], layers,
                                verbose=False)
            if raw is None:
                return None
            # exactly the evaluator's order, exactly its functions
            raw = energy_mod.apply_dram_override(
                energy_mod.apply_mac_override(raw, c, verbose=False), c,
                verbose=False)
            out[name] = raw
        # the geometry both arms declare, off the SAME cached mappings the
        # energies came from -- so the table cannot describe one and price
        # the other
        geo = {}
        for name in ("embedded", "recon"):
            m, _r = dilation._load_arm(arms[name], ARCH, layers[0].shape_name)
            geo[name] = m
        return _derive(cfg, k, q, out, geo)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _derive(cfg, k, q, raws, geo):
    emb, rec = raws["embedded"], raws["recon"]
    d = {"K": k, "q": q, "N": cfg.code_n, "scale": SCALE, "k_over_n": k / cfg.code_n,
         # PER ARM: the reconstruction arm's width comes from ITS q, the
         # 8-bit arm's from 8. They are NOT the same number and do not have
         # to be -- see eccenergy/code_widths.py.
         "spad_width": widths.declared_width(q, cfg.weight_bits),
         "glb_width": widths.level_width(q, False,
                                              cfg.weight_width_glb_mult,
                                              cfg.weight_bits),
         "emb_spad_width": widths.declared_width(cfg.weight_bits,
                                                      cfg.weight_bits),
         "capacity_ratio": cfg.weight_bits / q,
         "dram_pj_per_bit": cfg.dram_pj_per_bit,
         "baseline_dram_pj_per_bit": cfg.baseline_dram_pj_per_bit,
         "mac_pj": cfg.mac_pj_override,
         "ert_pj_per_bit": emb.dram["ert_pj_per_bit"]}

    for name, raw, m in (("emb", emb, geo["embedded"]), ("rec", rec, geo["recon"])):
        d[f"{name}_total_pJ"] = float(raw.base.sum())
        d[f"{name}_dram_pJ"] = float(raw.base["DRAM"])
        d[f"{name}_dram_weight_pJ"] = float(raw.dram["e_dram_w_pJ_charged"])
        d[f"{name}_dram_weight_reads"] = float(raw.dram_w_reads)
        for cat in raw.base.index:
            d[f"{name}_cat_{cat}"] = float(raw.base[cat])
        d[f"{name}_weights_held"] = m.weights_held
        d[f"{name}_weight_room"] = m.weight_room
        d[f"{name}_fill"] = m.fill
        d[f"{name}_refetch"] = m.refetch
        d[f"{name}_pes"] = m.pes_used
        d[f"{name}_cycles"] = m.cycles
        d[f"{name}_dram_loops"] = " ".join(f"{a}={b}" for a, b in
                                           sorted(m.dram_loops.items()))
        for space in ("Weights", "Inputs", "Outputs"):
            d[f"{name}_{space}_uJ"] = m.energy_by_dataspace(space) / 1e6
            d[f"{name}_{space}_xreduce"] = m.reduction_factor_for(space)
            d[f"{name}_{space}_refetch"] = m.dataspace_refetch(space)

    # --- RECON's DRAM K/N term. recon.py owns it in the production path and it
    #     never reaches dilation's table; it is the study's Task 3 mechanism.
    d["dram_kn_saving_pJ"] = d["rec_dram_weight_pJ"] * (1.0 - k / cfg.code_n)
    d["rec_total_with_dram_pJ"] = d["rec_total_pJ"] - d["dram_kn_saving_pJ"]
    d["rec_dram_final_pJ"] = d["rec_dram_pJ"] - d["dram_kn_saving_pJ"]

    # --- BASELINE shares embedded's mapping exactly and differs only in price.
    #     baseline_dram_mod.price() is the evaluator's own function, so the 70/40
    #     ratio is read from the record rather than hardcoded here.
    rec_price = baseline_dram_mod.price(cfg, emb, 0.0)
    d["baseline_ratio"] = rec_price["ratio"]
    d["baseline_dram_pJ"] = d["emb_dram_pJ"] * rec_price["ratio"]
    d["baseline_total_pJ"] = (d["emb_total_pJ"]
                              + d["emb_dram_pJ"] * (rec_price["ratio"] - 1.0))
    return d


def uJ(pj):
    return pj / 1e6


def selfcheck(rows):
    if SCALE != 1.0:
        print(f"  SELF-CHECK skipped: it is defined at scale 1.0, "
              f"this is x{SCALE:g}")
        return []
    r = next((x for x in rows if x["K"] == 30), None)
    if r is None:
        return ["BCH(63,30) is not in the cache -- the self-check cannot run"]
    got = {"baseline": uJ(r["baseline_total_pJ"]),
           "embedded": uJ(r["emb_total_pJ"]),
           "recon_mapping_only": uJ(r["rec_total_pJ"]),
           "recon_with_dram": uJ(r["rec_total_with_dram_pJ"]),
           "dram_baseline": uJ(r["baseline_dram_pJ"]),
           "dram_embedded": uJ(r["emb_dram_pJ"]),
           "dram_recon": uJ(r["rec_dram_final_pJ"])}
    bad = []
    print("  SELF-CHECK -- recomputed against progress.txt's BCH(63,30) record")
    for key, want in RECORDED_K30.items():
        have = got[key]
        ok = abs(have - want) <= 0.002 * max(1.0, abs(want))
        print(f"    {key:<20} recorded {want:>10.3f}   recomputed {have:>10.3f}"
              f"   {'ok' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(f"{key}: recorded {want}, recomputed {have}")
    print(f"    {'1 - K/N':<20} {1 - 30 / 63:>10.5f}"
          f"   (prompt_5 says 0.52381)")
    print(f"    {'DRAM weight @ 40':<20} {uJ(r['rec_dram_weight_pJ']):>10.3f} uJ"
          f"   (prompt_5 says 94.372)")
    print(f"    {'K/N saving':<20} {uJ(r['dram_kn_saving_pJ']):>10.3f} uJ"
          f"   (prompt_5 says 49.433)")
    return bad


def main():
    rows, missing = [], []
    for k in CODES:
        try:
            r = load_code(k)
        except Exception as exc:                      # noqa: BLE001
            print(f"  BCH(63,{k}): {type(exc).__name__}: {exc}")
            r = None
        (rows.append(r) if r else missing.append(k))
    if missing:
        print(f"  NOT IN CACHE: " + ", ".join(f"BCH(63,{k})" for k in missing))
    if not rows:
        return 2

    bad = selfcheck(rows)
    print()
    if bad:
        print("  SELF-CHECK FAILED -- not printing the tables:")
        for b in bad:
            print(f"    - {b}")
        return 1

    rows.sort(key=lambda r: -r["K"])
    if len(sys.argv) > 1:
        out = pathlib.Path(sys.argv[1])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  CSV -> {out}")
    _print(rows)
    return 0


def _rule(h):
    print(h)
    print("  " + "-" * (len(h) - 2))


def _print(rows):
    r0 = rows[0]
    print("\n" + "=" * 134)
    print(f"TABLE 1 [depth x{SCALE:g}] -- THE THREE ARMS.  DRAM at "
          f"{r0['dram_pj_per_bit']:g} pJ/bit; BASELINE's DRAM at "
          f"{r0['baseline_dram_pj_per_bit']:g} pJ/bit; MAC {r0['mac_pj']} pJ.")
    print("           `recon map only` = the mapper's win (packing + psum "
          "round-trips).  `+DRAM K/N` adds the fetch saving.")
    print("           TWO MECHANISMS, reported separately and never summed "
          "silently.")
    print("=" * 134)
    _rule(f"  {'code':<11} {'q':>2} {'K/N':>7} {'baseline':>9} {'embedded':>9} "
          f"{'recon':>9} {'recon':>9} | {'emb vs base':>11} "
          f"{'rec vs base':>11} {'rec vs emb':>11} {'rec vs emb':>11}")
    print(f"  {'':<11} {'':>2} {'':>7} {'uJ':>9} {'uJ':>9} "
          f"{'map only':>9} {'+DRAM':>9} | {'':>11} {'(+DRAM)':>11} "
          f"{'map only':>11} {'(+DRAM)':>11}")
    print("  " + "-" * 132)
    for r in rows:
        b, e = uJ(r["baseline_total_pJ"]), uJ(r["emb_total_pJ"])
        m, t = uJ(r["rec_total_pJ"]), uJ(r["rec_total_with_dram_pJ"])
        print(f"  BCH(63,{r['K']}) {r['q']:>2} {r['k_over_n']:>7.4f} "
              f"{b:>9.3f} {e:>9.3f} {m:>9.3f} {t:>9.3f} | "
              f"{100 * (b - e) / b:>10.2f}% {100 * (b - t) / b:>10.2f}% "
              f"{100 * (e - m) / e:>10.2f}% {100 * (e - t) / e:>10.2f}%")

    print("\n" + "=" * 134)
    print("TABLE 2 -- THE SILICON EACH PAIR DECLARED, and the mapping it bought.")
    print("           The arms share ONE DEPTH (assert_pair_geometry checks it). Each")
    print("           declares the WIDTH ITS OWN datawidth needs -- 96 at q=8, 98 at q=7,")
    print("           95 at q=5 -- and neither has to be legal for the other's.")
    print("=" * 134)
    _rule(f"  {'code':<11} {'spad W':>8} {'GLB W':>8} {'cap':>7} "
          f"{'emb held/room (fill)':>25} {'rec held/room (fill)':>25} "
          f"{'wt refetch e/r':>15} {'PEs e/r':>9}")
    for r in rows:
        print(f"  BCH(63,{r['K']}) {str(r['spad_width']):>8} "
              f"{str(r['glb_width']):>8} {r['capacity_ratio']:>6.4f}x "
              f"{r['emb_weights_held']:>9,}/{r['emb_weight_room']:<7,}"
              f"({r['emb_fill'] * 100:>5.1f}%) "
              f"{r['rec_weights_held']:>9,}/{r['rec_weight_room']:<7,}"
              f"({r['rec_fill'] * 100:>5.1f}%) "
              f"{r['emb_refetch']:>6.3f}/{r['rec_refetch']:<7.3f} "
              f"{r['emb_pes']:>4}/{r['rec_pes']:<4}")

    print("\n" + "=" * 134)
    print("TABLE 3 -- DRAM ALONE, and recon's two mechanisms separated.")
    print("=" * 134)
    _rule(f"  {'code':<11} {'baseline':>10} {'embedded':>10} {'recon':>10} | "
          f"{'recon saves vs embedded':>24} = {'mapping':>10} + {'K/N fetch':>10}")
    for r in rows:
        bd, ed = uJ(r["baseline_dram_pJ"]), uJ(r["emb_dram_pJ"])
        rd = uJ(r["rec_dram_final_pJ"])
        mapping = uJ(r["emb_dram_pJ"] - r["rec_dram_pJ"])
        kn = uJ(r["dram_kn_saving_pJ"])
        print(f"  BCH(63,{r['K']}) {bd:>10.3f} {ed:>10.3f} {rd:>10.3f} | "
              f"{ed - rd:>13.3f} ({100 * (ed - rd) / ed:>5.2f}%) = "
              f"{mapping:>10.3f} + {kn:>10.3f}")

    print("\n" + "=" * 134)
    print("TABLE 4 -- PER DATASPACE (Timeloop's own energies, ERT DRAM): where "
          "the MAPPING margin is.")
    print("           `x-reduce` for Outputs = DRAM-level accumulation "
          "round-trips. Weight refetch sits at its 1.000 FLOOR on both arms.")
    print("=" * 134)
    _rule(f"  {'code':<11} {'dataspace':<9} {'x-reduce e/r':>13} "
          f"{'refetch e/r':>17} {'emb uJ':>9} {'rec uJ':>9} {'delta':>9} "
          f"{'%':>8} {'share':>8}")
    for r in rows:
        tot = sum(r[f"rec_{s}_uJ"] - r[f"emb_{s}_uJ"]
                  for s in ("Weights", "Inputs", "Outputs"))
        for space in ("Weights", "Inputs", "Outputs"):
            e, c = r[f"emb_{space}_uJ"], r[f"rec_{space}_uJ"]
            d = c - e
            print(f"  BCH(63,{r['K']}) {space:<9} "
                  f"{('x' + str(r[f'emb_{space}_xreduce'])):>6} /"
                  f"{('x' + str(r[f'rec_{space}_xreduce'])):<6} "
                  f"{r[f'emb_{space}_refetch']:>8.3f} /"
                  f"{r[f'rec_{space}_refetch']:<8.3f} "
                  f"{e:>9.2f} {c:>9.2f} {d:>9.2f} "
                  f"{(100 * d / e if e else 0):>7.1f}% "
                  f"{(100 * d / tot if tot else 0):>7.1f}%")
        print(f"  {'':<11} DRAM loops  emb {r['emb_dram_loops']}"
              f"   |   rec {r['rec_dram_loops']}\n")


if __name__ == "__main__":
    sys.exit(main())
