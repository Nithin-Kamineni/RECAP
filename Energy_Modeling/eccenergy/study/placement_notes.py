"""The prose every placement bar carries, and where its cached stats live.

Two small things the placement study shares, and which both its compute half
(`study/placement_study.py`) and its figure (`report/recon_view.py`) need:

* the NOTES -- `EXPERIMENT`, `CODEC_NOTE`, `DRAM_TERM_NOTE`, `FIXED_MAPPING_NOTE`,
  `PARITY_KEY`, `ONCHIP_EXCLUDE` -- the sentences that must travel with a figure
  for it to be readable a month later (prompt_7 section 12).
* `stats_paths_for()`, which resolves one (arch, model, layer) to the cached
  `timeloop-mapper.stats.txt` and `.map.txt` the whole study re-reads.

WHAT IS OUTSIDE THE COMPARISON. ECC codec (decode/correction) energy, exactly as
in Task 2: charged to nobody unless `ECC_DECODE=1`, and the file says so. Moving
the decoder onto the DRAM die does not change that -- it is the same decoder, run
once per corrected codeword, and it cancels between the embedded bar and every
placement. Latency and area: Task 3 asks for energy, Task 4 for "relevant
latency/area overhead". No inference-accuracy claim is made anywhere.

ProjectRestructure phase 3 cut this out of `experiments/recon.py`, so that the
figure can have the notes without importing the evaluator.
"""
from __future__ import annotations

from ..arch import fingerprint as fingerprint_mod
from ..toolchain import invoke


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
    "Since 2026-09-09 the decoder sits on the DRAM die, off the fetch path "
    "(the only model since 2026-09-14); it is the same BCH decoder relocated, runs once "
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
    variant = fingerprint_mod.effective_variant(arch, cfg)
    fingerprint = ses.fingerprints.get(arch) or fingerprint_mod.arch_fingerprint(arch, cfg)
    mapper = invoke.Mapper(
        cfg, arch, None, ses.results.mapper_cache(arch, variant, fingerprint),
        None, fingerprint=fingerprint,
        legacy_root=ses.results.legacy_mapper_cache(arch, variant),
        ert_bump=fingerprint_mod.ert_bump(arch, cfg))      # prompt_6: an arm's entries carry its toll
    out = {}
    for layer in ses.models[model]:
        if layer.shape_name in out:
            continue
        path = mapper.stats_for(layer)
        if path is not None:
            out[layer.shape_name] = path
    return out, mapper


