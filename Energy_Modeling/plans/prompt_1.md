# DRAM cost model — status and to-do

*Last updated 2026-09-10. Supersedes the earlier version of this file (the
"f_if problem" writeup), which the change below resolves.*

---

## What was wrong

Accelergy's CactiDRAM charged **8 pJ/bit** (512 pJ per 64-bit access), and the
evaluator then split that into an array share and an interface share and gave
reconstruction credit only for the interface share (`f_if = 0.40`).

Both were wrong in the same direction, and together they were the reason a
**38% cut in bits fetched showed up as a 2% energy saving.**

---

## What is done

- **`ECC_DRAM_IF_FRAC` / `f_if` is REMOVED.** The DRAM access is
  custom-designed to collect only the interleaved message bits of each
  codeword, so the array reads fewer bits too. The DRAM is now one weight-path
  stage and the whole of it is credited:

  ```
  dram = DRAM weight energy x K/N       under EVERY boundary, R1 included
  ```

  Previously `dram_array = (1-f_if) x E` was never reduced and only
  `dram_interface = f_if x E` was. Both stages, the `f_if` knob, the refusal
  path and the 0.10/0.25/0.50 ceiling print are all gone.

- **Dynamic access cost raised 8 → 40 pJ/bit**, via a new
  `ECC_DRAM_PJ_PER_BIT` (env.sh §4). It rescales the *whole* DRAM category
  evaluator-side — weights, inputs and outputs alike, since per-bit cost is a
  property of the device, not of a dataspace. Built as the exact counterpart of
  `ECC_MAC_PJ_OVERRIDE` (`energy.apply_dram_override`), applied *after* the raw
  cache so `results/_raw/` stays pure Timeloop output.

  | pJ/bit | source | 64-bit access |
  |---|---|---|
  | 8 | Accelergy CactiDRAM LPDDR4 (verified: 64.0 pJ per 8-bit word) | 512 pJ |
  | 20 | **Horowitz, ISSCC 2014** — 32b DRAM read = 640 pJ | 1.28 nJ |
  | **40** | **default** — FReaC Cache (MICRO 2020), Gebhart et al. (MICRO 2012), 28–45 band | 2.56 nJ |

- **`ECC_DRAM_BACKGROUND_PJ` and `ECC_DRAM_REFRESH_PJ` added, both 0.** Only
  the dynamic term of `E_total = E_dynamic + E_background + E_refresh` is
  modelled, on purpose — the study's question is on-chip energy.

- Checks, tests, provenance and docs follow: the two directional DRAM checks
  collapse into one equality check `dram_scaled_by_K_over_N`; all
  `eccenergy.tests.test_recon` pass; `provenance.yaml` has a new
  `dram_access_energy` block and the retired `dram_interface_share` is kept as
  a comment for whoever revisits it.

**Effect, measured** (eyeriss_like / resnet18, BCH(63,39), edp mapping):

| pJ/bit | DRAM weight | DRAM saved / bar | best bar vs embedded |
|---|---|---|---|
| 8 (old cost, no f_if) | 1,396.6 µJ | 532.0 µJ | 8.73% |
| 20 (Horowitz) | 3,491.4 µJ | 1,330.1 µJ | 13.81% |
| **40 (default)** | **6,982.9 µJ** | **2,660.1 µJ** | **17.24%** |

Against the old model (8 pJ/bit **with** `f_if = 0.40`), which credited only
`0.40 × 532.0 = 212.8 µJ`, the DRAM saving is now **12.5× larger**
(2.5× from dropping `f_if`, 5× from the per-bit cost).

Task 4's fixed-mapping efficiency ceiling moves `f_if x (1-K/N) = 0.152` →
**`(1-K/N) = 0.381`**.

**One prior finding flipped.** FINDINGS 7.1's R4b audit ("most of what the
full-width register buys is what the same register would buy a PE with no ECC")
is no longer true: the register-alone saving is on-chip and unchanged at 466 µJ,
while ECC's marginal DRAM component went 122 µJ → 1,526 µJ, so the ratio is now
3.4 the other way. The test asserts the *separation* and reports the ratio
rather than bounding it.

---

## What is left

### Model E_background and E_refresh
Currently 0 (env vars exist and validate; only the physics is missing). **Not
neutral**: an arm holding fewer weight bits in DRAM saves both, so the embedded
and reconstruction arms are *understated* today.

```
E_total = E_dynamic + E_background + E_refresh
E_background ≈ P_standby x t_resident        (pJ per bit-second x bits x seconds)
E_refresh    ≈ bits x refreshes x pJ_per_bit_per_refresh
```

Needs: a residency time per weight tile (the mapper has it), an LPDDR4 standby
power and a refresh interval (tREFI, 3.9 µs typical).
