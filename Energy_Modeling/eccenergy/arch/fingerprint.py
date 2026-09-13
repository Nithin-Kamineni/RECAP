"""What the mapper cache is keyed by -- and the ERT bump that is part of it.

`arch_fingerprint(arch, cfg)` hashes the patched YAML the mapper actually sees,
plus the globals, every mapper setting, the components digest and the ERT bump.
A different fingerprint is a MISS, and every `fp-<hash>` directory is hours of
SLURM: a comparison across two of them is a comparison of two ARCHITECTURES.

THE FINGERPRINT HASHES THE GEOMETRY, NOT THE PROSE. `hashable_arch_text()`
strips comments, trailing whitespace and blank lines before hashing -- it is
quote-aware, so a `#` inside a YAML scalar stays data -- which is what lets
`arch_paper.yaml` carry a citation for every number at no cost, while
`depth: 64 -> depth: 96` still colds the cache.
`eccenergy/tests/test_arch_fingerprint.py` asserts both halves with mutations.

It hashes the ARCHITECTURE, not the price list Accelergy derives from it, so
fixing an ENERGY ESTIMATOR changes every number in the cache while leaving the
directory identical. `ECC_ENERGY_MODEL_REV` is the deliberate cold for that: any
non-empty value re-fingerprints the whole matrix, EMPTY hashes byte-identically
to every fingerprint that predates the knob.

`ert_bump()` is the ERT delta an arm applies (prompt_6 5.1), recomputed at run
time from the placement, the level's block size and the DC reconstruction
table -- and hashed in like every other ERT input, so changing a recon energy
re-solves the arm instead of reading it back at the old toll. `ert_hash_view()`
is the part of it that identifies the ARCHITECTURE: two arms differing only in
the toll must hash differently, two spellings of the same toll must not.

ProjectRestructure phase 3 cut this out of `archs.py`. Its one upward import --
the DC table, via `study/stacks.py` -- is declared in
`tests/contract/test_layer_rule.py` and is phase 4's to remove.
"""
from __future__ import annotations

import hashlib
import json

from ..paths import ARCH_COMPONENTS
from ..physics import widths

from .load import arch_source, load_standard, paper_source
from .patch import MAPSPACE_FREE_LEVELS, _patched_text, _relax_weight_factors, _scale_weight_capacity, _scale_weight_depth, _set_weight_geometry, patched_weight_geometry


# ------------------------------------------------------ prompt_6: the ERT arm
def ert_bump(arch, cfg):
    """The ERT delta this configuration's arm applies, or None for the reference.

    prompt_6 5.1, recomputed at run time -- the table in the plan is the check,
    not the source:

        access action   delta = E_w x block_size    block_size = width / datawidth,
                                                    read off THAT level in THIS
                                                    arm's patched arch
        leak            delta = idle_per_cycle      Timeloop x instances x cycles

    The level, counter and action come from the Placement (`recon.ert_arm_spec`),
    the two DC terms from `ecc.load_recon_energy`, E_w from `recon.ert_deltas`.
    `load_recon_energy`, NOT `load_recon_terms`: the toll the mapper optimises
    against has to be the toll the evaluator bills, so `ECC_RECON_PJ` must reach
    both or the two disagree. It is therefore hashed into the fingerprint like
    every other ERT input, so changing a recon energy invalidates the mapper
    cache and the arm is re-solved instead of being read back at the old toll.
    The result is what `arch_fingerprint()` hashes (RULE 4.4.5, defence 2),
    what `timeloop.ErtTables` patches into the base table, and what the
    read-back assertion compares a cache entry against.
    """
    arm = cfg.ert_arm() if hasattr(cfg, "ert_arm") else None
    if arm is None:
        return None
    geo = patched_weight_geometry(arch, cfg)
    level = arm["level"]
    if level not in geo:
        raise ValueError(f"ERT arm {arm['key']}: {arch} has no weight-carrying "
                         f"level {level!r} in its patched YAML; it has "
                         f"{', '.join(geo) or 'none'}")
    g = geo[level]
    if not g["width"] or not g["datawidth"] or g["width"] % g["datawidth"]:
        raise ValueError(f"ERT arm {arm['key']}: {level} declares width {g['width']} "
                         f"and datawidth {g['datawidth']}; block_size is undefined")
    block_size = g["width"] // g["datawidth"]
    from ..study import stacks as _ecc  # lazily: ecc needs pandas, the reference arm does not
    from ..arch import arms
    from ..physics import granularity
    inc, idle, prov = _ecc.load_recon_energy(cfg)
    gran = granularity.Granularity(cfg.code_n, cfg.code_k, cfg.weight_bits,
                              cfg.recon_granularity)
    d = arms.ert_deltas(inc, idle, gran, block_size,
                          getattr(cfg, "recon_clock_gating_pct", 0.0))
    d.update(placement=arm["key"], level=level, counter=arm["counter"],
             action=arm["action"], narrow_levels=list(arm["narrow_levels"]),
             level_width=g["width"], level_datawidth=g["datawidth"],
             code=f"BCH({cfg.code_n},{cfg.code_k})", dc_provenance=prov)
    return d


def ert_hash_view(bump):
    """The part of an ERT bump that identifies the ARCHITECTURE the mapper sees:
    level, every patched action and its delta at full precision. Two arms
    differing only in the toll must hash differently; two spellings of the
    same toll must not."""
    if bump is None:
        return None
    return {"placement": bump["placement"], "level": bump["level"],
            "actions": {bump["action"]: bump["access_delta_pj"],
                        "leak": bump["leak_delta_pj"]}}


# --------------------------------------------------------------- fingerprints
def hashable_arch_text(text):
    """`text` with everything Timeloop never reads taken out: comments, trailing
    whitespace, blank lines.

    WHY THE FINGERPRINT MUST NOT SEE A COMMENT (2026-09-13). `arch_fingerprint`
    hashes the patched YAML, and it hashed it byte for byte -- so correcting a
    comment that had gone stale re-keyed every mapper cache in the study and
    threw away hours of solved mappings for a documentation edit. That made the
    two things this project asks of an architecture file pull against each
    other: `arch_paper.yaml` is meant to carry the citation for every number it
    declares, and keeping those comments honest was priced at a full re-map.

    Timeloop is handed the file with its comments intact. It parses YAML, so a
    comment reaches no mapping decision and no energy. Hashing one therefore
    reported a change in the ARCHITECTURE that had not happened -- the exact
    false positive the fingerprint exists to avoid the mirror image of.

    A geometry edit still colds the cache, which is the whole point: `depth: 64`
    to `depth: 96` survives this normalisation and lands in the hash. What no
    longer does is the comment beside it.

    Quote-aware, so a `#` inside a YAML scalar (`name: "a#b"`) is data, not the
    start of a comment. No arch file has one today; one added later must not
    silently change meaning.
    """
    out = []
    for line in text.splitlines():
        quote = None
        cut = None
        for i, ch in enumerate(line):
            if quote is not None:
                if ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
            elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
                cut = i
                break
        if cut is not None:
            line = line[:cut]
        line = line.rstrip()
        if line:
            out.append(line)
    return "\n".join(out) + "\n"


def arch_fingerprint(arch, cfg, mapper_settings=None):
    """Content hash of the architecture the mapper will actually see.

    THE PROBLEM THIS SOLVES. The treatment slug (`paper`, `opt-energy`, ...)
    says what KIND of run this is, not what the architecture IS. Editing
    `archs/eyeriss_v2_like/arch_paper.yaml` -- correcting a scratchpad depth,
    say -- left the slug unchanged, so the next run happily reused mappings
    computed for the OLD geometry and reported them as the new design. That is
    silent, and it is exactly the class of error this task exists to remove.

    The hash covers the fully patched YAML text plus every mapper setting that
    could change which mapping comes back. Change either and the cache moves.

    THE TEXT IS NORMALISED FIRST (`hashable_arch_text`, 2026-09-13): comments,
    trailing whitespace and blank lines are stripped, because Timeloop parses
    YAML and none of them reaches a mapping decision. A geometry edit still
    colds the cache; correcting the comment beside it no longer does. That one
    change re-keyed every cache on disk once, on purpose -- the old directories
    are untouched and simply unused.

    `mapper_settings` defaults to `cfg.mapper_settings()`. It is a parameter so
    a caller that has already resolved the effective thread count can pass the
    real one rather than the configured `None`.
    """
    text = hashable_arch_text(_patched_text(arch, cfg, quiet=True))
    # globals.yaml is a mapper input too -- it sets the node DRAM is costed at,
    # which changes the energy the mapper is optimising. Hashing the values
    # rather than reading the file keeps this usable before it is written.
    globals_view = {
        "technology": cfg.force_technology or load_standard()["study"]["technology"],
        # prompt_7 C1.5: THIS design's rate. `globals_<arch>.yaml` carries it
        # into the mapper, so it has to be what is hashed -- hashing the study
        # default would give two designs at two clocks one fingerprint.
        "global_cycle_seconds": cfg.cycle_seconds_for(arch),
    }
    blob = {
        "arch": arch,
        "arch_yaml": text,
        "globals": globals_view,
        "mapper": mapper_settings or cfg.mapper_settings(),
        # The locally authored Accelergy components (archs/_shared/components)
        # set the per-access energies the mapper optimises against. Added
        # 2026-09-08: a register-write correction to regfile_decoded.yaml
        # changed every scratchpad's ERT without moving a single arch YAML,
        # and nothing would otherwise have told the cache.
        "components": components_digest(),
        "workload_shape_template_version": 1,
    }
    # prompt_7 C1.8. THE FINGERPRINT HASHES THE ARCHITECTURE, NOT THE PRICE
    # LIST Accelergy derives from it -- so a corrected ESTIMATOR (the Neurosim
    # plug-in that answered 0 pJ for every address generator) changes every
    # energy in the cache while leaving the directory it is stored under
    # identical. `ECC_ENERGY_MODEL_REV` is the deliberate cold, and until Phase
    # C it reached only `Config.fingerprint()` -- which labels a RESULT and
    # names no cache directory, so the knob re-labelled results while the cache
    # it was meant to invalidate stayed warm. Appended only when set, so EMPTY
    # still hashes byte-identically to every directory that predates it.
    if getattr(cfg, "energy_model_rev", ""):
        blob["energy_model_rev"] = cfg.energy_model_rev
    # prompt_7 C1.6. The MAC price the MAPPER optimises against. Until Phase C
    # `ECC_MAC_PJ_OVERRIDE` was evaluator-only: the mapper priced a MAC from
    # Accelergy's ERT at 1.16877 pJ and the report then rescaled Compute to
    # 0.23, so under ECC_OPT_METRIC=edp the plan was chosen for a machine whose
    # arithmetic cost 5x what the study charges. `timeloop.ErtTables` now
    # supplies the price, and a supplied price is part of the architecture.
    if getattr(cfg, "mac_pj_override", None) is not None:
        blob["mac_pj_override"] = float(cfg.mac_pj_override)
    # prompt_6 RULE 4.4.5, defence 2: an ERT arm's toll is part of what the
    # mapper optimises against. Only added when there IS one, so every
    # reference fingerprint on disk is unchanged.
    bump = ert_bump(arch, cfg)
    if bump is not None:
        blob["ert"] = ert_hash_view(bump)
    blob = json.dumps(blob, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def components_digest():
    """sha1 over the shared Accelergy component files, by name and content."""
    h = hashlib.sha1()
    for f in sorted(ARCH_COMPONENTS.glob("*.yaml")) if ARCH_COMPONENTS.exists() else []:
        h.update(f.name.encode("utf-8"))
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def effective_variant(arch, cfg):
    """The cache slug for THIS architecture, skipping no-op treatments.

    Two per-architecture treatments can turn out to be no-ops:

    * `paper` fidelity, on a design that has no `arch_paper.yaml`.
    * Forcing the datawidth to the weight width, on a design that already
      declares it. Those architectures keep their existing mapper cache, and
      only the designs the force actually rewrites pay for a fresh map.

    The globals.yaml and mapper knobs cannot be skipped this way: they change
    DRAM cost, or the search itself, for every design at once.
    """
    parts = list(cfg.global_variant_parts)
    # prompt_7 C1.5: THIS design's clock, not the study's. `ECC_ARCH_CLOCK_MHZ`
    # runs Eyeriss v1 at its published 200 MHz while the rest of the study stays
    # at the 1 GHz model default, so the rate is a per-architecture treatment
    # and the slug has to be written where the architecture is known. The
    # historical `1e-9` contributes nothing, so every directory on disk keeps
    # its name.
    clk = cfg.cycle_seconds_for(arch)
    if clk != "1e-9":
        parts.append(f"clk{clk}")
    if cfg.arch_fidelity != "stock" and paper_source(arch) is not None:
        parts.append(cfg.arch_fidelity)
    if cfg.force_datawidth:
        with_force = _patched_text(arch, cfg, apply_per_arch=True, quiet=True)
        without = _patched_text(arch, cfg, apply_per_arch=False, quiet=True)
        if with_force != without:
            parts.append(f"dw{cfg.force_datawidth}")
    if cfg.weight_capacity_scale != 1.0:
        # A design whose weight levels all round back to their declared depth
        # is NOT dilated, and must keep reading the undilated cache rather
        # than paying for a fresh map of an unchanged architecture -- the same
        # trap `_patch_dram_depth`'s docstring records.
        base = _set_weight_geometry(arch_source(arch, cfg).read_text(),
                                    getattr(cfg, "weight_datawidth", None),
                                    getattr(cfg, "weight_datawidth_levels", ()),
                                    getattr(cfg, "weight_width_glb_mult", 4),
                                    cfg.weight_capacity_scope, arch, quiet=True,
                                    weight_bits=cfg.weight_bits)
        if _scale_weight_capacity(base, cfg.weight_capacity_scale,
                                  cfg.weight_capacity_scope, arch,
                                  quiet=True) != base:
            parts.append(f"wcap{cfg.weight_capacity_scale:g}"
                         + ("-shared" if cfg.weight_capacity_scope == "shared" else ""))
    # THE WIDTH TABLE is applied to every run, so it is in every slug -- no
    # no-op rule, because there is no configuration in which it is off. It
    # marks the boundary in `ls`: a directory without it predates 2026-09-12
    # and was mapped on the published word shape.
    parts.append(f"wt{widths.base_width(cfg.weight_bits)}"
                 + (f"x{cfg.weight_width_glb_mult}"
                    if cfg.weight_width_glb_mult != 4 else ""))
    if getattr(cfg, "weight_depth_scale", 1.0) != 1.0:
        # Same no-op rule as the capacity scale, measured on the RESHAPED text
        # -- the ladder multiplies the renormalised depth.
        base = _set_weight_geometry(arch_source(arch, cfg).read_text(),
                                    getattr(cfg, "weight_datawidth", None),
                                    getattr(cfg, "weight_datawidth_levels", ()),
                                    getattr(cfg, "weight_width_glb_mult", 4),
                                    cfg.weight_capacity_scope, arch, quiet=True,
                                    weight_bits=cfg.weight_bits)
        if _scale_weight_depth(base, cfg.weight_depth_scale,
                               cfg.weight_depth_levels,
                               cfg.weight_capacity_scope, arch,
                               quiet=True) != base:
            parts.append(f"wdepth{cfg.weight_depth_scale:g}"
                         + ("-" + "+".join(cfg.weight_depth_levels)
                            if cfg.weight_depth_levels else ""))
    if getattr(cfg, "weight_datawidth", None) is not None:
        # No-op rule: the BASELINE/EMBEDDED arm may legitimately be spelled
        # `ECC_WEIGHT_DATAWIDTH=8` on a design whose levels already store 8-bit
        # weights, and that arm must then READ THE SAME CACHE as leaving the
        # knob empty -- otherwise the two arms of a pair would be compared
        # across two mapper caches of one identical architecture. Measured by
        # diffing the arm's geometry against the 8-bit arm's.
        src_text = arch_source(arch, cfg).read_text()
        eight = _set_weight_geometry(src_text, None, (),
                                     getattr(cfg, "weight_width_glb_mult", 4),
                                     cfg.weight_capacity_scope, arch, quiet=True,
                                     weight_bits=cfg.weight_bits)
        dw_levels = tuple(getattr(cfg, "weight_datawidth_levels", ()) or ())
        armed = _set_weight_geometry(src_text, cfg.weight_datawidth, dw_levels,
                                     getattr(cfg, "weight_width_glb_mult", 4),
                                     cfg.weight_capacity_scope, arch, quiet=True,
                                     weight_bits=cfg.weight_bits)
        if armed != eight:
            # Same spelling as `wdepth<scale>-<levels>`: an arm narrowing
            # only `filter_glb` is a different architecture from one
            # narrowing every weight level, and the directory name says so.
            parts.append(f"wdw{cfg.weight_datawidth}"
                         + ("-" + "+".join(dw_levels) if dw_levels else ""))
    if getattr(cfg, "mapspace_constrain", False) and arch in MAPSPACE_FREE_LEVELS:
        # A constrained loop nest is a different MAPSPACE and a different
        # DATAFLOW, so it is a different architecture to the mapper and gets
        # its own cache. A design with no free-set entry is NOT constrained and
        # keeps its existing cache rather than being silently mislabelled.
        parts.append("mcons")
    if cfg.weight_factor_relax:
        # Same no-op rule as the capacity scale: a design with no `factors:`
        # pin on a weight-indexing dimension is not relaxed by this and keeps
        # its existing cache rather than paying for a fresh map of an unchanged
        # architecture. `simple_weight_stationary` is exactly that case -- its
        # weight levels pin nothing a weight tile is indexed by, so its tile is
        # NOT capped by the dataflow constraints and this lever cannot help it.
        base = arch_source(arch, cfg).read_text()
        if _relax_weight_factors(base, arch, quiet=True) != base:
            parts.append("wrelax")
    part = cfg.mapper_arm_slug() if hasattr(cfg, "mapper_arm_slug") else None
    if part is not None:
        # prompt_6 RULE 4.4.5, defence 1. Never a no-op: the arm is the
        # directory. `ert-<key>-<level>-<action>` where the boundary declares
        # an ERT bump -- unchanged, so every entry on disk stays a hit -- and
        # prompt_7 B2's `arm-<key>` where it does not, because R1's patched
        # YAML IS the reference's until Phase C1.2 and without a slug of its
        # own the two would share one directory.
        parts.append(part)
    return "stock" if not parts else "__".join(parts)



