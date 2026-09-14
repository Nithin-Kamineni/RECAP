"""The banner every run prints before it does anything.

One screen: what is being run, on what, at what constants, with which prices and
which mapper settings -- and the mapper fingerprint the cache will be keyed by.
It is the last chance to notice that a knob is not what you thought, which is why
it prints BEFORE the first Timeloop call and why `run.sh --dry-run` is nothing
but this.

IT REPORTS, IT DOES NOT DECIDE. Every value here comes off the resolved `Config`;
nothing is recomputed and no default is applied. The one line that needs the
architecture -- the ERT arm's bump -- is HANDED IN by `config.banner()` as
`ert_arm_row`, because this package sits below `arch/` and may not import it.

ProjectRestructure phase 4 moved it out of `config.py`.
"""
from __future__ import annotations

from .code import BCH63_KTOD


def banner(cfg, recon_terms, recon_provenance, ert_arm_row=None):
    """`recon_terms` is `(incremental pJ/codeword, idle pJ/cycle/engine)` --
    prompt_6 RULE 3's two denominators -- or the old single float."""
    if isinstance(recon_terms, (tuple, list)):
        recon_inc, recon_idle = recon_terms
    else:
        recon_inc, recon_idle = recon_terms, None
    w = 78
    # The placement study's axis is not one of the three sweeps -- saying
    # "sweep=arch" there would name the axis it HOLDS.
    axis = ("placement (where the reconstruction boundary sits)"
            if cfg.experiment == "recon"
            else f"sweep={cfg.sweep} ({cfg.swept_axis})")
    lines = ["=" * w,
             f"ecc-energy  |  {axis}  |  output={cfg.stem}",
             "=" * w]

    if cfg.sweep == "bch":
        swept = "K = " + ", ".join(str(k) for k in cfg.sweep_ks)
    else:
        swept = ", ".join(str(v) for v in cfg.swept_values)

    # PRECISIONS, EXPLICIT AND DISTINCT. Task 1 asks the configuration summary
    # to report them, because the defect that started this work was two designs
    # being compared at different operand widths without anyone noticing. The
    # accumulator line is deliberately not a single number: it is per design,
    # and saying so on every run is the point.
    if cfg.acc_bits_override is None:
        acc = ("per architecture, as published (v1 16b, v2 20b, Simba 24b, "
               "simple designs 16b)  [primary]")
    else:
        acc = (f"{cfg.acc_bits_override}b FORCED on every design "
               f"(ECC_ACC_BITS)  [SENSITIVITY STUDY, not the primary result]")

    rows = [
        ("sweeping", swept),
        ("holding", "   ".join(f"{k}={v}" for k, v in cfg.held
                               # a panel figure does not hold the model -- each
                               # panel IS a model, so saying "holding model=X"
                               # would name only the first one and mislead
                               if not (cfg.experiment == "panels" and k == "model"))),
    ]
    if cfg.experiment == "panels":
        rows.append(("panels (top to bottom)", ", ".join(cfg.panel_models)))
    rows += [
        ("approaches", ", ".join(cfg.approaches)),
        ("workload file", cfg.workload),
        ("layer scope", (cfg.layer_scope if not cfg.layers else
                         f"{cfg.layer_scope}: {', '.join(cfg.layers)}")),
        ("weight precision", f"{cfg.weight_bits}b (the protected payload)"),
        ("activation precision", f"{cfg.activation_bits}b"),
        ("accumulator precision", acc),
        ("result phase", "derived per arm: Pre for baseline/embedded (the mapping is "
                         "ECC-unaware), Post for a placement mapped on its own chip"),
    ]
    if cfg.sweep == "bch":
        rows.append(("recon pJ/codeword", recon_provenance))
    else:
        rows += [
            ("code", f"BCH({cfg.code_n},{cfg.code_k})  t={cfg.code_t}  "
                     f"r={cfg.code_n - cfg.code_k}"),
            ("weights / codeword", f"{cfg.weights_per_codeword:.4f}"),
            ("DRAM weight traffic", cfg.baseline_dram_line),
            ("recon on-chip scale", f"{cfg.sram_scale:.4f} (weights only)"),
            ("reconstruction", f"{recon_inc:.7f} pJ per codeword (incremental)"
                               + (f" + {recon_idle:.7f} pJ per cycle per engine (idle; "
                                  f"RULE 3: separate denominators, never added)"
                                  if recon_idle is not None else "")),
            ("recon provenance", recon_provenance),
        ]
    if cfg.experiment == "recon":
        rows += [
            ("PLACEMENT STUDY", f"Task 3: fixed mapping, the boundary is the "
                                f"axis, one panel per architecture -> {cfg.stem}"),
            # `warn=False`: the banner is a summary, and the `[skip]` line
            # for a boundary this design does not declare belongs with the
            # collection that drops it, printed once, not inside a heading.
            # The RESOLVED bars, not the request: `ECC_APPROACHES`'s abstract
            # `recon` asks for "every placement this design declares", and a
            # heading that says so names nothing a reader can check.
            ("panels", "  |  ".join(
                f"{cfg.arch_label(a).replace(chr(10), ' ')}: "
                + (", ".join(cfg.recon_placement_bars(a, warn=False))
                   or "no placement it defines")
                for a in cfg.archs)),
            ("reduced form packing", cfg.recon_packing
                + ("   (retained k bits packed with no per-weight alignment; "
                   "every reduced stage scales by K/N)"
                   if cfg.recon_packing == "stream" else
                   "   (whole bits per weight, no repacking: wire falls, "
                   "SRAM access count may not)")),
            ("encoder charging", cfg.recon_granularity
                + ("   (amortized: per weight, at the per-codeword energy per "
                   "n/weight_bits weights)" if cfg.recon_granularity == "weight"
                   else "   (pessimistic: one whole codeword per access)")),
            ("mapping optimiser",
             (f"TASK 4: the reconstruction arm is re-mapped at weight capacity "
              f"x{cfg.weight_capacity_scale * cfg.code_n / cfg.code_k:.4f} "
              f"(= x{cfg.weight_capacity_scale:g} x N/K), scope "
              f"{cfg.weight_capacity_scope}; the reference bars keep "
              f"x{cfg.weight_capacity_scale:g}")),
            ("ERT-aware mapping",
             ("ON (prompt_6): the ERT-injectable boundaries are re-mapped with "
              "the encoder toll in the objective; the figure marks them"
              if cfg.recon_ert_aware else "off (ECC_RECON_ERT_AWARE=0)")),
            ("ERT arm", (ert_arm_row(cfg) if ert_arm_row else cfg.recon_ert_arm)),
        ]
    rows += [
        ("parity accounting", f"grouping={cfg.parity_grouping}, "
                              f"message padding "
                              f"{'charged' if cfg.parity_charge_padding else 'NOT charged'}"),
        ("decode", (f"base {cfg.decode_pj_base} / emb {cfg.decode_pj_emb} pJ per codeword"
                    if cfg.decode_enabled else
                    "0 pJ for every arm (ECC_DECODE=0; category hidden)")),
        ("weak ECC overlay", (f"BCH({cfg.weak_n},{cfg.weak_k}) -> x{cfg.weak_overhead:.4f} on-chip"
                              if cfg.weak_enabled else "off")),
        ("arch fidelity", cfg.arch_fidelity
            + ("   (archs/<name>/arch_paper.yaml where present)"
               if cfg.arch_fidelity == "paper"
               else "   (example_designs as shipped -- psum precision NOT honest)")),
        ("arch treatment", cfg.arch_variant_slug
            + ("" if cfg.arch_variant_slug == "stock" else "   (separate mapper cache)")),
        ("level classifier", cfg.classify_mode
            + ("" if cfg.classify_mode == "instances" else "   (legacy name matching)")),
        ("mapper objective", cfg.opt_metric
            + ("" if cfg.opt_metric == "energy" else
               "   (WARNING: this is an energy study; 'edp' trades energy for latency "
               "and penalises the wider PE array)")),
        ("mapper", f"victory={cfg.victory} scaling={cfg.victory_scaling} "
                   f"algorithm={cfg.mapper_algorithm} timeout={cfg.mapper_timeout} "
                   f"perms/if={cfg.mapper_max_permutations} "
                   f"search_size={cfg.mapper_search_size or 'uncapped'}"),
        ("palette / formats", f"{cfg.palette} / {', '.join(cfg.formats)} @ {cfg.dpi} dpi"),
        ("replot only", str(cfg.replot_only)),
        ("cached mapper only", str(cfg.from_cache)
            + ("   (never invokes Timeloop)" if cfg.from_cache else "")),
        ("re-run the optimiser", str(cfg.rerun_optimiser)
            + ("   (ECC_RERUN_OPTIMISER=1: valid cache entries are IGNORED and "
               "OVERWRITTEN)" if cfg.rerun_optimiser else "")),
    ]
    for k, v in rows:
        lines.append(f"  {k:22s}: {v}")
    lines.append("=" * w)
    return "\n".join(lines)
