# Fix the baseline's DRAM model: price its bigger array per bit, do not inflate its traffic

*Short, deliberate. One term to delete, one price to add. No mapper job moves,
no YAML changes, nothing on chip changes.*

## The model, stated once

All three arms move **the same number of weight bits across the DRAM
boundary**. Decoding happens on the DRAM die in this architecture, so a parity
bit never travels the datapath in ANY arm — the baseline's parity is read,
corrected and discarded on die, exactly as the embedded arm's is. What differs
between the arms is what a bit COSTS, and — for recon alone — how many bits are
driven off the die at all:

| arm | DRAM array | weight bits fetched | `pJ/bit` | DRAM weight energy |
|---|---|---|---|---|
| baseline | bigger: parity stored beside the weights | 8 per weight | **70** | emb x **1.75** |
| embedded | declared | 8 per weight | 40 | reference |
| recon | declared — **same size as embedded** | **8 x K/N** per weight | 40 | emb x **K/N** |

Baseline's **70 pJ/bit** is the whole of its penalty for now. It covers the
larger array *and* the indexing/addressing work the baseline must do that the
other two arms do not; it is one settled constant, not a transfer cost with an
overhead bolted on. `E_background` and `E_refresh` also grow with the array —
they are 0, they stay 0 in this change, and they are accounted for later.

## What is already correct — do not re-implement

**RECON's K/N DRAM fetch saving IS implemented.** `recon.py`'s `dram` stage is
`reducible=True` and scales the whole DRAM weight term by K/N under every
boundary — the decoder is on the DRAM die, so only the k message bits are read
out and driven off it. Verified in `recon.py` `WEIGHT_PATHS` and CLAUDE.md.
Leave it alone. Recon is the ONLY arm whose traffic scales.

**One DRAM depth serves all three arms, and it stays that way.**
`archs._patch_dram_depth(text, cfg.dram_depth)` writes the declared
`ECC_DRAM_DEPTH` for every arm. That is now correct by construction: baseline's
extra array size is priced by the 70 pJ/bit constant, so do **not** also scale
`depth:` for it. Doing so would double-charge the array AND split baseline off
embedded's mapping into its own fingerprint and a needless re-map. Baseline and
embedded must keep sharing one mapper cache.

## Defect 1 — the baseline inflates its TRAFFIC. Reverse it.

`ecc.external_parity()` bills the parity and message-padding bits as extra
weight-scalar reads on top of the measured ones. Verified at BCH(63,30), 8-bit
weights, `layer3.0.conv1` (294,912 weights):

    stored      2,359,296 payload + 3,244,032 parity + 589,824 message pad
                = 6,193,152 bits  = 2.625 x the payload
    traffic     external_dram_scalars = 479,232 extra weight-equivalents
                on top of 294,912 payload reads
    energy      those scalars x the measured per-scalar DRAM cost, added to
                the total as the component "DRAM external BCH parity"

So today the baseline is billed **2.625x** embedded's DRAM weight energy purely
in traffic (not the flat `N/K` = 2.1x — the accounting charges padding too).
**That traffic does not exist.** Nothing leaves the DRAM die but the k message
bits' worth of weights, and both arms fetch the same 8 bits per weight.

**Fix.** The baseline's parity energy term goes to zero.

* `experiments/baseline.py` writes `components["DRAM external BCH parity"]`
  = **0.0** — explicitly zero, so a reader sees "zero", not "missing", matching
  how `experiments/embedded.py` already spells `PARITY_KEY` for its own arm.
* **Keep calling `external_parity()` and keep its `detail` in the variant's
  `extra`.** It is no longer an energy term; it is the SIZE evidence — 6,193,152
  stored bits against 2,359,296 of payload — that the 70 pJ/bit constant is
  priced against, and its `hand_check` still has to pass.
* `parity.py` and `ecc.external_parity()` are **not edited.** The baseline
  accounting is frozen; only the place where its number enters the total moves.

## Defect 2 — the baseline is billed at embedded's per-bit price

`energy.apply_dram_override()` charges every arm the one `ECC_DRAM_PJ_PER_BIT`,
so after Defect 1 the baseline would be numerically identical to embedded. It
must not be: its array is bigger.

**Fix.** One new knob, evaluator-side:

    ECC_BASELINE_DRAM_PJ_PER_BIT=70     # env.sh, beside ECC_DRAM_PJ_PER_BIT

* **Set** (70) — the baseline arm's DRAM category is rescaled to that price and
  the parity energy term is 0. This is the model above.
* **EMPTY** — the pre-2026-09-10 model: baseline priced at
  `ECC_DRAM_PJ_PER_BIT` and charged the parity traffic term. Reproduces today's
  numbers exactly, which is what the diff is read against.

The rescale is the same shape as the MAC override and reads the price already
recorded by `apply_dram_override`, so it is correct whether
`ECC_DRAM_PJ_PER_BIT` is 40, 20 or empty:

    ratio = cfg.baseline_dram_pj_per_bit / raw.dram["pj_per_bit_charged"]
    components["DRAM"] *= ratio          # 70/40 = 1.75 at the study's default

* **The WHOLE DRAM category scales — weights, inputs and outputs alike.** The
  per-bit cost is a property of the device, not of a dataspace; this is the same
  rule `apply_dram_override` follows and `tests/test_dram_override.py` asserts.
* Record `pj_per_bit_baseline`, `pj_per_bit_charged` and the `ratio` in the
  variant's `extra`, so the 1.75x is visible in the result JSON.
* Put the rescale in a **new module** (e.g. `eccenergy/baseline_dram.py`) that
  `experiments/baseline.py` calls in one line. `ECC_BASELINE_DRAM_PJ_PER_BIT`
  goes in env.sh's DRAM section, one commented variable on one line, and into
  the export list beside `ECC_DRAM_PJ_PER_BIT`.
* It is **not** in `Config.fingerprint()` and must not be added to it: it
  changes no mapper input. Verify the baseline and embedded arms still hash to
  the same `arch_fingerprint`.

## Two consumers that move with this

1. **`experiments/embedded.py` `task2_checks()`**, check
   `dram_difference_is_exactly_the_external_parity` — under the new model that
   difference is no longer parity traffic. It becomes a PRICE check: baseline's
   DRAM equals embedded's DRAM x `70/40`, with the DRAM traffic identical on
   both arms. Rename it to say so. Check 1 (`non_dram_components_match`) is
   unchanged and must still pass exactly.
2. **`hpc/summary.py`** prints a parity column (`total - parity`, `parity`,
   `total`). It will read **0.00** and make the baseline look free. Replace it
   with the DRAM price delta (baseline DRAM - embedded DRAM), or the summary
   silently reports "conventional ECC costs nothing".

## The check that must pass afterwards

Three arms, one layer, one depth, printed together:

| arm | DRAM depth | DRAM weight reads | bits fetched/weight | pJ/bit | DRAM weight energy |
|---|---|---|---|---|---|
| baseline | `D` | `R` | 8 | **70** | emb x **1.75** |
| embedded | `D` | `R` | 8 | 40 | reference |
| recon | `D` | `R` | `8 x K/N` | 40 | emb x **K/N** = 0.47619x |

Assert, all of them:

1. baseline DRAM weight reads **==** embedded's, exactly — the inflation is gone;
2. baseline's `DRAM external BCH parity` component **== 0.0**, and its
   `external_parity_accounting` detail is still present and still hand-checks;
3. baseline's DRAM category **== 1.75x** embedded's, on every dataspace, and
   nothing else in the two component dicts differs by any amount;
4. recon DRAM weight reads **==** embedded's, but recon DRAM weight *energy*
   **==** embedded x K/N;
5. no on-chip level differs between baseline and embedded;
6. baseline and embedded resolve to the **same `arch_fingerprint` and the same
   mapper cache** — the proof that no depth was scaled and no re-map happened;
7. with `ECC_BASELINE_DRAM_PJ_PER_BIT` unset, `run.sh baseline --eval`
   reproduces the stored result JSONs bit-for-bit. Diff them and report it.

Report the measured before/after baseline total: the term deleted (2.625x on
the DRAM weight energy) and the term added (1.75x on the whole DRAM category)
are not the same size, and the direction is a result, not a rounding detail.
