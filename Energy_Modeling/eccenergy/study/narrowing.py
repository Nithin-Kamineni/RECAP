"""The on-chip narrowing must be applied EXACTLY ONCE (prompt_2, prompt_6 RULE 1).

Two things can narrow an on-chip weight stop: the MAPPER, when the arm's patched
YAML declares `datawidth: q` on that level, and the EVALUATOR, when it scales
that level's energy by K/N afterwards. Doing both SQUARES the saving, and
nothing in the output says so.

WHO NARROWS IS MEASURED, per bar per stage, never assumed: `Word bits == q` in
that bar's own stats means the mapper did (the evaluator then applies 1.0),
`== weight_bits` means the evaluator does, and anything else stops the run. Two
live sites or none on a narrow stop are both HARD REFUSALS. `build_stacks()`'s
recon column carries the same measured guard.

`onchip_narrowing_audit()` is the report and `assert_onchip_narrowing_once()` the
refusal; `test_dilation.py` mutation-tests both.

ProjectRestructure phase 3 cut this out of `recon.py`.
"""
from __future__ import annotations

from ..physics.packing import Packing
from ..settings import guards


# ===========================================================================
#  PROMPT_2: the on-chip narrowing must be applied EXACTLY ONCE
# ===========================================================================
def onchip_narrowing_audit(cfg):
    """Where the K/N on-chip weight narrowing is applied, and how many times.

    CONFIGURATION-LEVEL ESTIMATE. Since prompt_6 phase 4 the placement study
    decides ownership per BAR per STAGE from measured `Word bits`
    (`Packing.narrowing_owner`, `narrowing_audit_for_bar`, called inside
    `evaluate_placement`), which is the authority; this function keeps the
    coarse pre-check `dilation --levels` prints. FINDINGS 6.3 records why the
    name-based `evaluator_narrows` below is not a measurement.

    THE DOUBLE-COUNTING THIS EXISTS TO STOP. There are now two places that can
    narrow an on-chip weight:

    * the EVALUATOR, via `Packing`. `stream` scales every reduced stage's
      access count and energy by k/n.
    * the MAPPER, via `ECC_WEIGHT_DATAWIDTH`. Timeloop bills
      `vector_access_energy / block_size` with `block_size = width/datawidth`,
      so a narrower declared datawidth delivers the same saving INSIDE the
      Timeloop number, with no mapping change at all.

    Apply both and the on-chip saving is SQUARED. prompt_2's resolution is
    `ECC_RECON_PACKING=aligned`, whose docstring already describes exactly the
    right model -- "each reduced weight occupies a whole number of bits ...
    the access count, and the SRAM energy, do not move at all" -- so `aligned`
    leaves the on-chip narrowing entirely to the mapper, which is now where it
    belongs.

    Two consequences this checks, both named in prompt_2:

    * DRAM `datawidth` stays 8 on every arm. `_set_weight_datawidth` refuses to
      touch DRAM for this reason; `recon.py` owns the DRAM K/N scaling and
      narrowing DRAM in the YAML too would double-count it there. That is
      structural, so it is asserted in `archs.py` rather than here.
    * `aligned` computes its own reduced width as `ceil(weight_bits*k/n)`, and
      that has to AGREE with the `datawidth` the arch declares, or the arch and
      the accounting are describing two different codes and the check cannot
      reconcile. At BCH(63,30) both are 4; at BCH(63,57) `aligned` says 8 and
      the declared q is 7, which is exactly the disagreement prompt_2 warns
      about.

    NEITHER site active is NOT a defect: `aligned` with no mapper-side
    datawidth is the pessimistic bound this study has always been able to run
    (`Packing`'s docstring, section 16). It comes back `ok` with a `note`, so
    it stays runnable but cannot be mistaken for a prompt_2 result.

    Returns a dict; `ok` False means the run must not be quoted.
    """
    mapper_bits = getattr(cfg, "weight_datawidth", None)
    packing = Packing(cfg.recon_packing, cfg.weight_bits, cfg.code_k, cfg.code_n)
    mapper_narrows = (mapper_bits is not None
                      and mapper_bits < cfg.weight_bits)
    # `stream` is the only packing that moves an on-chip ACCESS COUNT.
    evaluator_narrows = cfg.recon_packing == "stream"
    sites = ([f"mapper (ECC_WEIGHT_DATAWIDTH={mapper_bits})"] if mapper_narrows
             else []) + (["evaluator (ECC_RECON_PACKING=stream)"]
                         if evaluator_narrows else [])
    out = {
        "mapper_datawidth": mapper_bits,
        "packing": cfg.recon_packing,
        "aligned_bits_per_weight": packing.reduced_bits_per_weight,
        "sites": sites, "n_sites": len(sites),
        "ok": len(sites) == 1,
    }
    if len(sites) > 1:
        out["problem"] = (
            "THE ON-CHIP NARROWING IS APPLIED TWICE and the saving is "
            "SQUARED: " + " and ".join(sites) + ". prompt_2's resolution is "
            "ECC_RECON_PACKING=aligned, which leaves the on-chip narrowing "
            "entirely to the mapper.")
    elif not sites:
        # NOT a defect, and deliberately not a stop: `aligned` with no
        # mapper-side datawidth is the PESSIMISTIC BOUND this study has always
        # been able to run (`Packing`'s docstring, section 16). It predates
        # prompt_2 and stays runnable. It is simply not a prompt_2 result, so
        # it says so rather than being quoted as one.
        out["ok"] = True
        out["note"] = (
            f"NO on-chip narrowing is applied anywhere: "
            f"ECC_RECON_PACKING={cfg.recon_packing} moves no on-chip access "
            f"count and ECC_WEIGHT_DATAWIDTH is unset, so the reconstruction "
            f"arm stores its weights at the full {cfg.weight_bits} bits. That "
            f"is the pessimistic bound, not prompt_2's model -- for that, set "
            f"ECC_WEIGHT_DATAWIDTH={packing.reduced_bits_per_weight} "
            f"(= ceil({cfg.weight_bits}*K/N) at "
            f"BCH({cfg.code_n},{cfg.code_k})) on the reconstruction arm.")
    elif mapper_narrows and mapper_bits != packing.reduced_bits_per_weight:
        out["ok"] = False
        out["problem"] = (
            f"THE ARCH AND THE ACCOUNTING DISAGREE ABOUT THE CODE: the arch "
            f"declares datawidth {mapper_bits} on its weight levels, but "
            f"`aligned` computes ceil({cfg.weight_bits}*K/N) = "
            f"{packing.reduced_bits_per_weight} bits at "
            f"BCH({cfg.code_n},{cfg.code_k}). Make them agree -- prompt_2's "
            f"width table rounds 8*K/N to an integer q and the declared "
            f"datawidth IS that q -- or the check cannot reconcile.")
    return out


def assert_onchip_narrowing_once(cfg):
    """`onchip_narrowing_audit()` as a hard stop. Returns the audit."""
    audit = onchip_narrowing_audit(cfg)
    if not audit["ok"]:
        raise guards.refusal("narrow-once",
            audit["problem"])
    return audit


