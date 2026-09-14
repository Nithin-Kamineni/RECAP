"""The conventional-ECC baseline's DRAM cost: a PRICE, not extra traffic.

    ECC_BASELINE_DRAM_PJ_PER_BIT=40      # env.sh section 4, beside ECC_DRAM_PJ_PER_BIT=20
                                         # (since 2026-09-11; 70 beside 40 before)

WHAT CHANGED, AND WHY (2026-09-10)
----------------------------------
All three arms move the SAME number of weight bits across the DRAM boundary.
The BCH decoder is on the DRAM die (the study's only model since 2026-09-14,
01_project_context 1/4), so the baseline's parity is read, corrected and
discarded ON DIE exactly as the embedded arm's is: it never travels the
datapath. The baseline therefore issues no extra DRAM weight reads, and the
`external_parity()` traffic term -- 479,232 extra weight-equivalents on 294,912
payload reads at BCH(63,30), i.e. 2.625x the DRAM weight energy -- is not a
cost this architecture pays.

What the baseline DOES pay is a dearer bit. Its array holds the parity beside
the weights (6,193,152 stored bits against 2,359,296 of payload on
`layer3.0.conv1`), and it does the indexing work the embedded and reconstruction
arms do not. Both are priced by one constant: 40 pJ/bit against 20 since
2026-09-11 (70 against 40 from 2026-09-10 to 2026-09-11, x1.75). Embedded and
reconstruction share one array size and one price; reconstruction is the only
arm whose TRAFFIC scales, by K/N, and `recon.py` owns that.

    baseline   same reads, 8 bits/weight, 40 pJ/bit  -> DRAM x 2.0
    embedded   same reads, 8 bits/weight, 20 pJ/bit  -> reference
    recon      same reads, 8 x K/N bits/weight, 20   -> DRAM weight x K/N

E_background and E_refresh grow with the array too. They are 0 and unmodelled
(`ECC_DRAM_BACKGROUND_PJ`, `ECC_DRAM_REFRESH_PJ`), so this is the whole of the
baseline's penalty for now.

WHERE IT APPLIES
----------------
Evaluator-side, after `energy.apply_dram_override()` has set the per-bit price
every arm starts from -- the same shape as the MAC override, and for the same
reason: no saved pJ and no mapping moves, so `results/_raw/` stays pure Timeloop
output and the mapper fingerprint is untouched (`ECC_BASELINE_DRAM_PJ_PER_BIT`
is deliberately NOT in `Config.fingerprint()`). Baseline and embedded keep
sharing one mapper cache, which is what the array size NOT being scaled buys.

The WHOLE DRAM category scales -- weights, inputs and outputs alike -- because
a per-bit access cost is a property of the device, not of a dataspace. That is
the rule `apply_dram_override` follows and `tests/test_dram_override.py`
asserts; charging weights alone would inflate the weight share and flatter
every ECC percentage.

`parity.py` and `ecc.external_parity()` are NOT touched by any of this. The
baseline's parity accounting is frozen and still runs on every result -- it is
now the SIZE evidence the baseline's pJ/bit is priced against, and its hand check still
has to pass. Only the place where its number enters the total moved.

THE LEGACY PATH
---------------
With the knob unset every function here reproduces the pre-2026-09-10 model
bit-for-bit: the baseline is priced at `ECC_DRAM_PJ_PER_BIT` like the embedded
arm and charged the parity traffic on top. That is what a diff is read against,
and it is why the check names differ between the two models rather than one
name meaning two things.
"""
from __future__ import annotations

import math


#: The component the two arms used to differ in. Written explicitly as 0.0
#: under the price model, so a reader sees "zero", not "missing".
PARITY_KEY = "DRAM external BCH parity"

#: The category a per-bit DRAM price applies to. One category, never split.
DRAM_KEY = "DRAM"


# ---------------------------------------------------------------------------
#  Accelergy's own per-bit DRAM price, read back off a record
# ---------------------------------------------------------------------------
# It lived in `study/energy.py` until ProjectRestructure phase 3. It is
# arithmetic over two numbers of a Timeloop record and nothing else, so an L1
# module may own it -- and owning it is what ends this file's one import of an
# L4 module. `energy.apply_dram_override()` is still its other caller.

# -----------------------------------------------------------------------------
#  THESE THREE STATE WHAT THEY NEED (ProjectRestructure section 4.3, phase 7).
#
#  They took a whole `Config` and read ONE knob out of it. A `Config` is L3 and
#  this module is L1, so the signature was claiming a dependency the layer rule
#  says cannot exist -- it only worked because Python does not check, and it made
#  them untestable without building a whole configuration first.
#
#  `cfg.code` and `cfg.energy` are what a caller passes now. This is the
#  incremental half of phase 4 that phase 4 deliberately did not do: one module
#  at a time, never one big diff, and `charge()` and `charge_stack()` below still
#  take `cfg` because `study/baseline.py` calls them and that file is FROZEN.
# -----------------------------------------------------------------------------
def dram_ert_pj_per_bit(raw, code):
    """Accelergy's own per-BIT dynamic DRAM energy, read back off the record.

    Timeloop counts a DRAM access in units of the dataspace datawidth, so the
    weight rows give it directly: `e_dram_w / (dram_w_reads x weight_bits)`.
    For the LPDDR4 model these designs use that is 64.0 pJ per 8-bit word =
    8.0 pJ/bit = the documented 512 pJ per 64-bit access.

    `code` is a `settings.CodeSettings` (or anything carrying `weight_bits`).
    """
    bits = float(raw.dram_w_reads) * float(code.weight_bits)
    return (float(raw.e_dram_w) / bits) if bits else None


def enabled(energy):
    """Is the price model in force? False = the pre-2026-09-10 traffic model.

    `energy` is a `settings.EnergySettings`.
    """
    return getattr(energy, "baseline_dram_pj_per_bit", None) is not None


def charged_pj_per_bit(raw, code):
    """What a DRAM bit cost in THIS record, before the baseline's own price.

    `apply_dram_override()` records it; a record that never went through the
    override (a synthetic one in a test) falls back to Accelergy's own ERT,
    read off the weight rows.
    """
    v = (getattr(raw, "dram", None) or {}).get("pj_per_bit_charged")
    return float(v) if v else dram_ert_pj_per_bit(raw, code)


def price(cfg, raw, e_parity=0.0):
    """The baseline arm's DRAM pricing record, for the result JSON.

    Returns a dict for either model. `ratio` is what the DRAM category is
    multiplied by (1.0 under the legacy model, where the parity traffic is
    charged as a separate component instead).
    """
    charged = charged_pj_per_bit(raw, cfg.code)
    if not enabled(cfg.energy):
        return {
            "model": "parity_traffic",
            "ratio": 1.0,
            "pj_per_bit_charged": charged,
            "pj_per_bit_baseline": charged,
            "parity_traffic_energy_charged_pJ": float(e_parity),
            "note": cfg.baseline_dram_note,
        }
    tgt = float(cfg.baseline_dram_pj_per_bit)
    if not charged:
        raise ValueError(
            "ECC_BASELINE_DRAM_PJ_PER_BIT is set, but this record carries no "
            "per-bit DRAM price to scale from (no DRAM weight reads and no "
            "ECC_DRAM_PJ_PER_BIT). Refusing to guess one: set "
            "ECC_DRAM_PJ_PER_BIT, or evaluate a record with weight traffic.")
    return {
        "model": "per_bit_price",
        "ratio": tgt / charged,
        "pj_per_bit_charged": charged,
        "pj_per_bit_baseline": tgt,
        "parity_traffic_energy_NOT_charged_pJ": float(e_parity),
        "note": cfg.baseline_dram_note,
    }


def charge(cfg, raw, components, e_parity):
    """Put the baseline's DRAM cost into `components`, in place.

    `components` is the per-category dict of a baseline variant. Returns the
    pricing record; `extra["baseline_dram_pricing"]` is where it belongs.

        price model   DRAM x ratio, PARITY_KEY = 0.0
        legacy        DRAM unchanged, PARITY_KEY = the parity traffic energy
    """
    rec = price(cfg, raw, e_parity)
    before = float(components.get(DRAM_KEY, 0.0))
    if rec["model"] == "per_bit_price":
        components[DRAM_KEY] = before * rec["ratio"]
        components[PARITY_KEY] = 0.0
    else:
        components[PARITY_KEY] = float(e_parity)
    rec["dram_pJ_before"] = before
    rec["dram_pJ_after"] = float(components.get(DRAM_KEY, 0.0))
    rec["dram_delta_pJ"] = rec["dram_pJ_after"] - before
    return rec


def reference_bars_agree(base_components, emb_components, e_parity, pricing,
                         dram_w_reads=None):
    """Is the baseline-vs-embedded DRAM difference exactly what the model says?

    ONE implementation for every experiment that draws both reference bars
    (Tasks 1, 2, 3 and 4), so the three cannot drift into checking different
    things under the same claim. Returns `(ok, detail)`; the caller names the
    check, because each experiment states the claim in its own words.

        price model   baseline DRAM == embedded DRAM x ratio, parity term 0
        legacy        baseline DRAM  - embedded DRAM == the parity traffic
    """
    base_dram = (base_components.get(DRAM_KEY, 0.0)
                 + base_components.get(PARITY_KEY, 0.0))
    emb_dram = (emb_components.get(DRAM_KEY, 0.0)
                + emb_components.get(PARITY_KEY, 0.0))
    common = {"baseline_dram_pJ": base_dram, "embedded_dram_pJ": emb_dram,
              "difference_pJ": base_dram - emb_dram}
    if dram_w_reads is not None:
        common["dram_weight_reads_both_arms"] = dram_w_reads
    if pricing is not None and pricing["model"] == "per_bit_price":
        expect = emb_dram * pricing["ratio"]
        ok = (math.isclose(base_dram, expect, rel_tol=1e-12, abs_tol=1e-6)
              and base_components.get(PARITY_KEY, 0.0) == 0.0)
        common.update({
            "model": "per_bit_price",
            "expected_baseline_dram_pJ": expect,
            "ratio": pricing["ratio"],
            "pj_per_bit_baseline": pricing["pj_per_bit_baseline"],
            "pj_per_bit_embedded": pricing["pj_per_bit_charged"],
            "parity_component_pJ": base_components.get(PARITY_KEY, 0.0),
            "rule": ("the arms fetch the SAME weight bits -- the decoder is on "
                     "the DRAM die, so the baseline's parity is corrected there "
                     "and never crosses the datapath. The only difference is what "
                     "a bit costs out of the bigger, indexed array, so the "
                     "baseline's DRAM must be the embedded arm's x that ratio and "
                     "its parity component must be zero")})
        return ok, common
    ok = math.isclose(base_dram - emb_dram, e_parity, rel_tol=1e-12, abs_tol=1e-6)
    common.update({"model": "parity_traffic", "external_parity_pJ": e_parity,
                   "source": "ecc.external_parity() and ecc.embedded_dram(), unchanged"})
    return ok, common


#: The check name each model earns in the Task 2 result. Two names, because one
#: name meaning two different things is how a model change hides in a diff.
CHECK_NAMES = {
    "per_bit_price": "baseline_dram_is_exactly_the_per_bit_price",
    "parity_traffic": "dram_difference_is_exactly_the_external_parity",
}


def check_name(pricing):
    """Which DRAM check this result carries."""
    model = (pricing or {}).get("model", "parity_traffic")
    return CHECK_NAMES[model]


def caveat(pricing):
    """What a reader must not over-read about the baseline's DRAM cost.

    One text per model, so a result never carries the other model's wording.
    """
    if pricing is not None and pricing["model"] == "per_bit_price":
        return (
            f"The baseline's DRAM cost is a PER-BIT PRICE, not extra traffic: it "
            f"reads the same weight bits as the embedded arm (the decoder is on "
            f"the DRAM die, so parity is corrected there and never crosses the "
            f"datapath) and pays {pricing['pj_per_bit_baseline']:g} pJ/bit against "
            f"{pricing['pj_per_bit_charged']:g} for the bigger, indexed array "
            f"(x{pricing['ratio']:.4f} on the whole DRAM category, weights, inputs "
            f"and outputs alike -- a per-bit cost is a property of the device, not "
            f"of a dataspace). The parity accounting on this result is the array-"
            f"SIZE evidence for that price, not an energy term; it charges no pJ. "
            f"E_background and E_refresh grow with the array and are not modelled.")
    return (
        "External parity is billed at the measured per-access energy of a "
        "DRAM weight read on this architecture, not at an independently "
        "modelled parity-region access cost. That is exact if parity is "
        "read from the same DRAM by the same controller, which is the "
        "conventional-ECC assumption.")


def charge_stack(cfg, raw, col, e_parity):
    """The same charge on a sweep-figure column, where there is no parity row.

    `study.stacks.build_stacks()` plots the physical categories only, so the legacy
    model folds the parity traffic INTO `DRAM` rather than beside it. Returns
    the pricing record.
    """
    rec = price(cfg, raw, e_parity)
    before = float(col[DRAM_KEY])
    if rec["model"] == "per_bit_price":
        col[DRAM_KEY] = before * rec["ratio"]
    else:
        col[DRAM_KEY] = before + float(e_parity)
    rec["dram_pJ_before"] = before
    rec["dram_pJ_after"] = float(col[DRAM_KEY])
    rec["dram_delta_pJ"] = rec["dram_pJ_after"] - before
    return rec
