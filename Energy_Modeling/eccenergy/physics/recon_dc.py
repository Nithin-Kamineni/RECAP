"""The reconstruction datapath's MEASURED energy, selected by the code.

prompt_6 RULE 3: the synthesized engine costs TWO things and they are never
added -- `incremental` pJ per reconstruction EVENT and `idle` pJ per CYCLE for
every engine that exists. `load_recon_energy()` returns them apart, and the
three sites that charge them (`study/stacks.py`'s recon column,
`study/placement_eval.py`, and the ERT bump the MAPPER is given) each pick their
own denominator.

LOOKUP ORDER: env.sh section 4's two tables, matched on (n, k); then
`data/dc/BCH_N63_results.json`, matched the same way; then the fallback
constants, with a warning. env.sh is FIRST so the run's own configuration prices
the datapath -- the JSON is the synthesis archive, and a value typed into
section 6 is a deliberate statement about this run that a file on disk must not
silently outrank.

THE IDLE TERM IS RESCALED TO THE DESIGN'S CLOCK, ONCE, here: it is pJ per cycle
measured at DC's 1 ns clock and it is CLOCK power, so a 5 ns cycle burns five
times as much of it. The incremental term is switching energy per codeword and
is NOT rescaled. An entry that states a different measurement clock is REFUSED
rather than rescaled from the wrong base (env.sh section 4's TRAP).

WHY THIS IS IN `physics/` AND READS A FILE. It is the one module here that
touches the disk, and the exception is deliberate: `data/dc/` is a cited, static
MEASUREMENT -- an input like the width table, not a cache and not a result.
Before ProjectRestructure phase 4 this lived in `study/stacks.py`, and the two
`arch/` modules that price an ERT bump had to reach UP into a study driver to
get it; that was the last upward import in the package.
"""
from __future__ import annotations

import json
import pathlib

from ..paths import ROOT
from ..settings.energy import DC_MEASUREMENT_CLOCK_NS
from ..settings import guards


def _dc_entries(cfg):
    """The Design Compiler JSON as a list, or [] with one warning."""
    path = pathlib.Path(cfg.recon_json)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        if not _dc_entries.warned:
            print(f"  [warn] {path} not found; using the built-in DC table")
            _dc_entries.warned = True
        return [], path
    try:
        entries = json.loads(path.read_text())
        return ([entries] if isinstance(entries, dict) else entries), path
    except Exception as exc:
        if not _dc_entries.warned:
            print(f"  [warn] could not parse {path}: {exc}; using the built-in DC table")
            _dc_entries.warned = True
        return [], path

def _env_table_entry(table, n, k):
    """The env.sh table's value for BCH(n,k): key `BCH_<n>_<k>` or `BCH_<n>_<k>_t<t>`."""
    want = f"BCH_{n}_{k}"
    for key, val in (table or {}).items():
        if key == want or key.startswith(want + "_"):
            return float(val), key
    return None, None

def load_recon_terms(cfg, k=None):
    """The synthesized datapath's TWO terms, separately: `(incremental_pJ_per_codeword,
    idle_pJ_per_cycle_per_engine, provenance)` for BCH(cfg.code_n, k).

    prompt_6 RULE 3: incremental is per reconstruction EVENT, idle is per CYCLE
    for every engine that exists, and adding them is dimensionally wrong. The
    ERT arm (archs.ert_bump) needs them apart -- incremental sets the access
    delta, idle sets the `leak` delta -- and `build_stacks()` /
    `recon.evaluate_placement()` charge them on their own denominators.

    Lookup order: env.sh section 4's two tables, matched on (n, k); the DC JSON,
    matched the same way; the fallback constants, with a warning. env.sh is
    FIRST so that the run's own configuration is what prices the datapath: the
    JSON is the synthesis archive, and a value typed into section 6 is a
    deliberate statement about this run that a file on disk must not silently
    outrank. Both terms must be present in the tables to win -- a half-populated
    table falls through to the JSON rather than mixing the two sources.
    `ECC_RECON_PJ` is not applied here -- see `load_recon_energy`.
    """
    inc, idle, prov = _recon_terms_at_dc_clock(cfg, k)
    # THE ONE SITE THE CLOCK RESCALING IS APPLIED (env.sh section 4's TRAP;
    # prompt_7 C1.5). `idle` is pJ PER CYCLE measured at a 1 ns DC clock and it
    # is CLOCK power, so a 5 ns cycle burns five times as much of it. `inc` is
    # switching energy per codeword (CV^2) and does NOT scale with the period.
    # Applied after the three lookup branches converge, so no source can be
    # rescaled twice or missed -- and it was x1.0 on every run before a design
    # had a clock of its own, which is why nothing had ever exercised it.
    scale = cfg.dc_idle_scale() if hasattr(cfg, "dc_idle_scale") else 1.0
    if abs(scale - 1.0) > 1e-12:
        idle = idle * scale
        prov += (f"; idle rescaled x{scale:g} to this design's clock "
                 f"({float(cfg.cycle_seconds_for(cfg.archs[0])) * 1e9:g} ns "
                 f"against the DC report's "
                 f"{DC_MEASUREMENT_CLOCK_NS:g} ns) -> "
                 f"{idle:.7f} pJ/cycle/engine. The INCREMENTAL term is per "
                 f"codeword and is NOT rescaled")
    return inc, idle, prov

def _recon_terms_at_dc_clock(cfg, k=None):
    """The two DC terms AS MEASURED, at the report's own 1 ns clock.

    Split out so `load_recon_terms` has exactly one place to apply the period
    rescaling -- three lookup branches with three rescalings is how a factor
    of five goes missing on one of them.
    """
    k = cfg.code_k if k is None else k
    inc, key_i = _env_table_entry(getattr(cfg, "recon_incremental_table", {}), cfg.code_n, k)
    idle, key_d = _env_table_entry(getattr(cfg, "recon_idle_table", {}), cfg.code_n, k)
    if inc is not None and idle is not None:
        return inc, idle, (f"env.sh DC tables [{key_i}]: incremental={inc:.7f} "
                           f"pJ/codeword, idle={idle:.7f} pJ/cycle/engine "
                           f"at the DC report's {DC_MEASUREMENT_CLOCK_NS:g} ns")
    entries, path = _dc_entries(cfg)
    for e in entries:
        if int(e.get("n", -1)) != cfg.code_n or int(e.get("k", -1)) != k:
            continue
        # The entry states the clock it was measured at. If it is not the one
        # `DC_MEASUREMENT_CLOCK_NS` assumes, rescaling from it would be
        # rescaling from the wrong base -- refuse rather than guess.
        got = (e.get("measurement") or {}).get("clock_period_ns")
        if got is not None and abs(float(got) - DC_MEASUREMENT_CLOCK_NS) > 1e-9:
            raise guards.refusal("dc-clock-mismatch",
                f"{path.name} [{e.get('configuration_id')}] was measured at "
                f"{got} ns, but DC_MEASUREMENT_CLOCK_NS is "
                f"{DC_MEASUREMENT_CLOCK_NS}. The idle term is pJ PER CYCLE "
                f"and is rescaled to the design's clock from that base (env.sh "
                f"section 6, TRAP 2), so the two must agree.")
        inc = float(e["energy_pJ"]["incremental_per_codeword"])
        idle = float(e["energy_pJ"]["idle_per_cycle"])
        return inc, idle, (f"{path.name} [{e.get('configuration_id')}] "
                           f"n={e['n']} k={e['k']} t={e.get('t')}: "
                           f"incremental={inc:.7f} pJ/codeword, "
                           f"idle={idle:.7f} pJ/cycle/engine at "
                           f"{DC_MEASUREMENT_CLOCK_NS:g} ns")
    print(f"  [warn] no reconstruction energy for BCH({cfg.code_n},{k}); "
          f"using the fallback constants")
    return (cfg.recon_incremental_fallback_pj, cfg.recon_idle_fallback_pj,
            f"hardcoded fallback constants, no entry for BCH({cfg.code_n},{k})")

def load_recon_energy(cfg, k=None):
    """`(incremental_pJ_per_codeword, idle_pJ_per_cycle_per_engine, provenance)`
    for BCH(cfg.code_n, k) -- prompt_6 RULE 3's two terms, never added.

    The synthesis run is selected by MATCHING (n, k), so the K sweep and a
    fixed-K run read the same table and neither can silently be costed at
    another code's datapath. `k` defaults to the held constant, ECC_CONST_K.
    `ECC_RECON_PJ` overrides the INCREMENTAL term only; a per-cycle term has
    no per-codeword override.
    """
    inc, idle, prov = load_recon_terms(cfg, k)
    if cfg.recon_pj_override is not None:
        return (cfg.recon_pj_override, idle,
                f"ECC_RECON_PJ override ({cfg.recon_pj_override} pJ/codeword incremental); "
                f"idle {idle:.7f} pJ/cycle/engine from {prov}")
    return inc, idle, prov
