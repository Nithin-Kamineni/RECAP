# PROJECT_STATUS

**2026-09-09 (latest) — the MAPPER OBJECTIVE is what decides Eyeriss v1's result, not its buffer sizes (FINDINGS §7.6). `ReconSweep_optEnergy.png` is the same two designs with only `ECC_OPT_METRIC=energy` instead of `edp`; no YAML was edited. Weight stationary does not move (every bar within 0.05 pp; refetch 1.07x left nothing to recover). **Eyeriss v1 changes completely: scratchpad residency 8 -> 20 weights clears G_rec = 9, so R4a/R4b become feasible and R4b is the best result in the study — +11.38 % over embedded, +29.45 % over conventional, 92 % of its own 12.33 % ceiling** (61.5x amortization, 384-entry register), with R4a its mirror at −14.74 % (230.4 M reconstructions vs 3.7 M). Its total energy is 42.6 % lower than under EDP. THE PAIR IS THE RESULT: the two objectives differ by 42.6 % of total energy and by whether v1's 5/5 boundary exists at all, so neither should be quoted alone. Weight stationary's R5 stays infeasible under both — a depth-1 register cannot hold 9 co-resident weights, and no mapping fixes that.**

**2026-09-09 — utilisation audit answering "is anything at its limit?" (no).** Peak weight occupancy: WS operand_glb 56.2 % weights / 87.9 % with activations, pe_spad 6–26 % typical, weight_reg 100 % but one word deep by design; v1 weights_spad 42.9 % peak, 1.8 % low, and v1 has NO weight GLB. The buffers that ARE full hold activations and psums (v1 ifmap_glb 93.8 %, ifmap_spad 100 %). Refetch WS 1.072x, v1 1.868x — v1 refetches heavily **while holding capacity in reserve**, so its refetch is a missing level, not a full one. Two defects found in the WS design and NOT yet fixed: its PE array runs at **6.25 % utilisation** (16 of 256 PEs, never uses meshX, on 11 of 12 shapes) and it has **zero temporal loops below its register level** on all 12 shapes, so its "stationary" weight register serves 1.07 MAC uses per load and the scratchpad is read once per MAC (1.70 G reads). Fixing the second will SHRINK R4b's headline, which rests on amortizing 215 M reconstructions down to 3.9 M.


**2026-09-09 (latest) — weight-stationary and Eyeriss v1 join the placement study, and `ReconSweep.png` is now TWO PANELS (FINDINGS §7.5). Whole resnet18, fixed EDP mappings, 21/21 layers, BCH(63,39), f_if = 0.40, MAC 0.23 pJ. Weight stationary: embedded saves 10.39 % and **R4b (reduced RF + decoded stationary register) adds +5.46 % over embedded**, 70 % of its 7.84 % ceiling — the source discussion's full ordering 5/5 > 4/5 > 3/5 > control > 2/5 reproduced exactly, with R4a the one boundary in the study that LOSES energy (−7.64 %, 215.4 M reconstructions against R4b's 3.9 M). Eyeriss v1: embedded saves 17.70 % and **R3 (spad input) adds +3.65 %**; R2 falls below R1 because its column network's 14 destinations receive 6.5× the words injected. R5 on WS (MAC input, register holds 1 < G_rec 9, all 21 layers) and R4a/R4b on v1 (spad keeps 8 < 9 on four layer4 shapes) are reported INFEASIBLE with the layers named. `ECC_RECON_ARCHS` is one panel per design; the Eyeriss v2 study is preserved as `ReconSweep__eyeriss_v2_2026-09-09.*`. A second wave at `ECC_OPT_METRIC=energy`, where resident tiles are 128–192 and every boundary evaluates, lands as `ReconSweep_optEnergy.png`.**

**2026-09-09 (correction, FINDINGS §7.4) — a NETWORK boundary's encoders run once per destination-side ARRIVAL, not once per injection.** R2 credited its network with carrying the reduced form while charging the network's ingresses, which is a different placement (an encoder before the fanout makes that network full width — that is R1). The count is now `Ingresses × Multicast factor`, summed off Timeloop's own `@multicast M @scatter S` breakdown and cross-checked against the next stage's intake on every result (`multicast_arrival_chain`). **It changes a published number: `eyeriss_v2_like` R2 vs embedded 3.87 % → 3.78 %, and R3 (+3.84 %) is now the best boundary instead of R2** — the mesh's arrivals equal the cluster-local ingresses, so R2 and R3 pay the same encoder count and R3 saves strictly more. `ECC_RECON_ENCODER_SITE=source` reproduces the old numbers for the diff. Found while adding Eyeriss v1, whose column network multicasts 14-fold and made the discrepancy impossible to read as rounding.


**2026-09-09 (final) — Horowitz MAC cost adopted as the primary denominator, by the user's decision: `ECC_MAC_PJ_OVERRIDE:=0.23` is the env.sh default and `ReconSweep.png` is drawn under it: embedded vs conventional 19.31 %, R2 +3.87 % / R3 +3.84 % / R1 +3.56 % vs embedded (FINDINGS §7.3). The ERT-denominator numbers (14.58 % / +2.76 %) are the sensitivity row, reproduced with `ECC_MAC_PJ_OVERRIDE=` empty. `ReconSweep__mac0.23.*` and `ReconSweep_unfinshed.*` removed. No re-mapping: the mapper prices MACs from the ERT whatever this knob says, so the EDP caveat is recorded rather than resolved.**

**2026-09-09 (earlier) — MAC-cost audit (FINDINGS §7.3).** The 1.16877 pJ/MAC the ERT charges is one 40 nm Aladdin HLS table row each for a 32-bit multiplier (12.68 pJ) and adder (0.21 pJ), scaled linearly in width (×0.25 ×0.25 / ×20/32) and *up* 1.2652× from 40 to 45 nm by the Library plug-in — 5× Horowitz's int8 figure and ~5× the measured Eyeriss ALU share. New evaluator-side knob `ECC_MAC_PJ_OVERRIDE` (env.sh §5, default EMPTY = ERT) rescales Compute to MACs × value after the raw cache; saved µJ are bit-identical across rows (`tests/test_mac_override.py`, 4 tests). At the Horowitz 0.23 pJ row every percentage is ×1.40: embedded vs conventional 14.58 → 19.31 %, R2 vs embedded +2.76 → +3.87 %; no ordering moves. Primary figure and numbers stay at the ERT value; the figure subtitle now states the MAC cost, and the override run lands on its own stem, `results/figures/ReconSweep__mac0.23.png`, beside the primary. Verification: `test_recon` 36 ok / 1 skip, `test_embedded`, `test_noc` pass; `ECC_RECON_MODELING=1 ECC_MAC_PJ_OVERRIDE=0.23 bash hpc/run_all.sh --eval-only` then the default re-run restored `ReconSweep.png`.**

**2026-09-10 (later) — R4b IS SCRAPPED, on every design, in the code and in the specs.** The retained-reconstruction boundary (a reconstructed-weight register/latch between the weight SPad and the MAC, rated 5/5 by the source discussion) does not work, and the measurement is not close. **Consecutive weight reuse is 1 on 20 of the 21 resnet18 layers** (conv1 is the exception at 196): the mapper's innermost temporal loops below the scratchpad walk C/R/S, all weight dimensions, so the required weight changes every cycle and a 1–4 entry latch is reloaded on every access. Reuse need not be consecutive, but the register must then hold the whole cycled set — `inner_tile` is 16–384 weights, worst case **384, which is exactly `simple_weight_stationary`'s entire 192×16b@8b `pe_spad`** — and there is no useful middle, because a cyclic walk is the worst case for any replacement policy (0 % hit rate, no partial credit). It was worth only **+1.34 points over R3** even with that 384-entry register charged nothing for area or leakage, and it was already UNSUPPORTED on `eyeriss_like`. **No other dataflow rescues it:** output- and input-stationary hold the psum or the activation in the PE by definition, so weights stream past the MAC faster still; Simba's PE behaves like Eyeriss v2. The mapper could in principle be constrained to hold a weight while sweeping P/Q (Timeloop does produce such a nest for conv1), but weight reuse trades directly against psum residency — 196 consecutive uses needs 196 psums live in the PE — so buying it costs more than the boundary was worth. **R3 (reconstruct at the PE weight-storage INPUT) is the result: +11.07 % vs embedded on `simple_weight_stationary`, +17.24 % on `eyeriss_like`** — no extra state, no mapping constraint. REMOVED: `recon5` on all three placement lists (WS's MAC-input row renumbered `recon6`→`recon5`, still reported INFEASIBLE), `Placement.reuse_register`, the `retained` counter, `retention_stage()`, `retention_model()`, `weight_loop_nest()`, `ReuseRegister`, `reuse_register_pj()`, the `full_width` diagnostic, and `ECC_RECON_REUSE_REG_PJ` / `_ENTRIES` / `_MODEL`. KEPT: `storage_access_pj()` (Task 4's dilation correction needs it) and the `Recon overhead` plot category, now structurally zero, so the stacks, legend and result schema are unchanged. Specs updated in place rather than renumbered, because the code cites them by section: `01_…` §2.3 is now the rejection with its evidence, and every R4b-equivalent row (Eyeriss v2's post-SPad latch, WS's and v1's R3b, Simba's vector-MAC register, the cross-architecture "decoded reuse register" column, the two leading hypotheses in `02_…`, REC-L5, EV2-C, EV2-CACHE, §22.1–22.3) is marked REMOVED with the reason. Verification: `test_recon`, `test_dilation`, `test_mac_override`, `test_embedded`, `test_mapper_lock`, `test_results_store` all pass (9 R4b-only tests deleted); `recon --eval`, `baseline --eval` and `embedded --eval` all run clean; `Recon overhead` is 0.000 on every bar. `test_noc::test_declared_pitches_match_the_cached_art` still fails, pre-existing and unrelated. `parity.py`, `baseline.py`, `ecc.build_stacks` untouched.**

**2026-09-10 — the DRAM cost model was wrong in two ways at once, and both are fixed.** (1) **`ECC_DRAM_IF_FRAC` / `f_if` is REMOVED.** The array/interface split was a 0.40 that credited a reconstruction boundary only 15.2 % of the DRAM weight energy for a 38.1 % cut in bits fetched — the whole reason a 38 % traffic cut read as a 2 % energy saving. The DRAM is now ONE weight-path stage and the whole of it is × K/N: the DRAM access is custom-designed to collect only the interleaved message bits of each codeword, so the array reads fewer bits too. (2) **Dynamic access cost 8 → 40 pJ/bit** via the new `ECC_DRAM_PJ_PER_BIT` (env.sh §4, `energy.apply_dram_override`, the DRAM counterpart of `ECC_MAC_PJ_OVERRIDE`, applied after the raw cache). Accelergy's CactiDRAM charged 8.0 pJ/bit (verified at 64.0 pJ per 8-bit word = the documented 512 pJ/64 b); 40 pJ/bit is within the 28–45 band reported by FReaC Cache (MICRO 2020) and Gebhart et al. (MICRO 2012). Measured (eyeriss_like/resnet18, BCH(63,39)): DRAM weight 1,396.6 → 6,982.9 µJ, DRAM saved per bar 532.0 → 2,660.1 µJ, best bar 8.73 % → 17.24 % vs embedded; against the old `f_if`=0.40 model the DRAM saving is **12.5× larger**. `E_background` / `E_refresh` added as env vars, both **0 and NOT modelled** (a TODO, and not neutral: an arm storing fewer weight bits would save both). Also: the two directional DRAM checks collapse into one equality `dram_scaled_by_K_over_N`; a `_EYERISS_V2_WGLB_PATH` slice that hard-coded "two DRAM stages" was reordering that design's path and is now derived; the real-cache test fixture applies the override so the weight path and the `Raw` record are checked against each other; **FINDINGS §7.1's R4b reading flipped** (the register-alone saving is on-chip and unchanged at 466 µJ while ECC's marginal DRAM term went 122 → 1,526 µJ, so ECC is now the larger half). Verification: `test_recon`, `test_dilation`, `test_mac_override`, `test_embedded`, `test_mapper_lock`, `test_results_store` all pass; `test_noc::test_declared_pitches_match_the_cached_art` fails and is **pre-existing and unrelated** (NoC tile pitch 116.9 vs declared 108 µm). Task 1's hand check passes at 320.0 pJ per 8-bit DRAM weight. `parity.py`, `baseline.py`, `ecc.build_stacks` untouched.**

**2026-09-09 (later) — decoder moved onto the DRAM die (Task 3, `recon.py`): every reconstruction boundary now saves DRAM INTERFACE energy. `ECC_DRAM_IF_FRAC` defaults to **0.40, an ASSUMPTION** (decided by the user the same day; no cited LPDDR4 array/interface breakdown exists, `provenance.yaml` `dram_interface_share` says so). `ReconSweep.png` is drawn at 0.40: R2 +2.76 % / R3 +2.74 % / R1 +2.54 % vs embedded, DRAM interface saving 159.5 µJ on every R bar; ordering unchanged. A first draft at 0.25 (set on the command line) is the run pasted below. FINDINGS.md §7.2.**

---

## 2026-09-09 TASK 3 REVISION — THE BCH DECODER IS ON THE DRAM DIE

**What changed and why.** `01_project_context_and_architectures.txt` §1/§4 puts
the decoder on the DRAM die and OFF the fetch path, so at fetch time only the k
message bits of each n-bit codeword cross the DRAM interface. Accelergy's
CactiDRAM is one flat per-bit constant, so the evaluator splits the DRAM weight
energy with `f_if`, the interface share:

    dram_array     = (1 - f_if) x DRAM weight energy    never reduced
    dram_interface =      f_if  x DRAM weight energy    x K/N under EVERY boundary

The two reference bars keep controller-side correction and did not move
(`parity.py`, `baseline.py`, `embedded_dram()` untouched). This is a
`recon.py` / `experiments/recon.py` change only.

**f_if has no default, deliberately.** The spec requires an LPDDR4 array-vs-I/O
citation in `archs/_shared/provenance.yaml` before env.sh may carry one. A
search on 2026-09-09 found (recorded there under `dram_interface_share`):
O'Connor et al., MICRO 2017 — HBM2 3.92 pJ/bit = 1.21 activation + 2.24 on-die
data movement + 0.30 I/O, i.e. I/O ≈ 7.7 %, but for an unterminated interposer
link (a lower bound for LPDDR4); and Ha, Stanford PhD 2018, Fig. 4.8 — an
LPDDR4 energy/bit breakdown with the off-die I/O (read driver + termination) as
its own category, but only plotted, not stated. Neither is adopted. With
`ECC_DRAM_IF_FRAC` empty the study **refuses** and prints the ceiling at
0.10 / 0.25 / 0.50; the figure on disk was drawn at **0.25, set on the command
line**, and its subtitle, table, manifest and result file all say so. The DRAM
interface saving is linear in f_if and identical on every R bar, so it changes no
ordering among the placements, only their common offset against embedded.

### Files changed

| file | change |
|---|---|
| `env.sh` §4 | `ECC_RECON_DECODE_SITE` (ondie \| controller) and `ECC_DRAM_IF_FRAC` (EMPTY = refuse); recon1 comment; both exported in §10. |
| `eccenergy/config.py` | `recon_decode_site`, `dram_if_frac` (the only reader of the two variables); `RECON_DECODE_SITES`; validation (f_if in [0,1], f_if=0 refused under ondie); `dram_model_line()` in `recon_title()`. |
| `eccenergy/recon.py` | `WEIGHT_PATHS`: the single `dram` stage is now `dram_array` (reducible=False) + `dram_interface` (reducible=True), both matching Timeloop's `DRAM`; `_level_shares()` splits the level by f_if and refuses any other double claim, so the level is claimed once in total and `cross_check()` reconciles the pair's sum. `PLACEMENTS`: `dram_interface` prepended to every `reduced` tuple, R1 = `("dram_interface",)` with site `dram_array`/reads. `stages_for()`, `placements_for()`, `placement_by_key()`, `validate_placement_space()` take `cfg`; under `controller`, `dram_interface` is not reducible and leaves every reduced set. `dram_model_detail()` replaces the `dram_unchanged` string. Docstring "WHAT IS DELIBERATELY NOT CREDITED" rewritten. |
| `eccenergy/experiments/recon.py` | ceiling print: array share, interface share, interface × (1−K/N), on-chip + interface; `refuse_without_f_if()`; checks `dram_identical_to_embedded_reference` → `dram_array_identical_to_embedded_reference` + `dram_interface_scaled_by_K_over_N` (one-sided each, so each fails on a cheat in its own direction); `DRAM i/f uJ` column on the console; `decode_site`, `f_if`, `dram_array_uJ`, `dram_interface_uJ`, `dram_interface_saving_uJ` columns in the CSV; per-bar note gains a `DRAM I/O` line; manifest gains `recon_decode_site`, `dram_if_frac`, `dram_if_frac_provenance`, `dram_split`. |
| `eccenergy/experiments/common.py` | `Session.finish(..., extra=)` for the manifest. |
| `eccenergy/plots/stacked.py` | head-room grows with a three-line bar note (`draw_panel` widened, not forked). |
| `eccenergy/tests/test_recon.py` | every "DRAM untouched" assertion → array untouched + interface × K/N; two cheat tests (array credit fails only the array check, unscaled interface fails only the interface check, both on the stack and on the detail rows); `controller` reproduces the pre-2026-09-09 synthetic numbers to the digit and differs from `ondie` by exactly the interface saving on every bar; refusal / knob validation; the DRAM level claimed exactly once; R1 isolates the interface saving. Real-cache property tests now treat an `unsupported` PE-local boundary as the modelled outcome it is (R4a/R4b are infeasible on the 8x2 EDP mappings, FINDINGS §4.1.3) instead of failing on a fact about the mapping. 37 tests. |
| `archs/_shared/provenance.yaml` | `dram_interface_share` block: the two candidate citations, why neither is a default. |
| `CLAUDE.md`, `FINDINGS.md` §7.2, this file | the structure and the numbers. |

### Verification, in the prescribed order (outputs pasted)

```
$ bash hpc/tl.sh python3 -m eccenergy.tests.test_recon
eccenergy reconstruction-placement (Task 3) tests
  ok    test_a_full_width_register_must_remove_the_spad_reads_it_serves
  ok    test_a_missing_stage_is_unsupported_rather_than_skipped
  ok    test_a_pe_local_boundary_is_rejected_when_the_tile_is_smaller_than_g_rec
  ok    test_a_reducible_stage_no_boundary_reduces_stops_the_run
  ok    test_a_register_smaller_than_the_working_set_collapses_r4b_onto_r4a
  ok    test_a_silent_dram_array_credit_fails_the_array_check_and_only_that_one
  ok    test_a_stronger_code_saves_strictly_more_at_every_reduced_stage
  ok    test_a_tile_sized_register_beats_every_other_boundary
  ok    test_a_weight_level_no_stage_claims_fails_the_cross_check
  ok    test_an_unclaimed_weight_level_fails_the_recorded_check_rather_than_dropping
  ok    test_an_unscaled_dram_interface_fails_the_interface_check_and_only_that_one
  ok    test_category_energies_are_monotonic_down_the_weight_path
  ok    test_controller_site_reproduces_the_pre_2026_09_09_numbers
  ok    test_encoder_charging_modes_bracket_each_other
  ok    test_every_reduced_stage_scales_by_k_over_n_and_no_other_stage_moves
  ok    test_every_weight_carrying_level_of_the_real_design_is_claimed_by_a_stage
  ok    test_free_mode_reproduces_the_pre_audit_accounting_exactly
  skip  test_full_width_really_moves_the_spad_reads_on_the_real_cache  (R4b is unsupported on this cache: infeasible local placement: 4 (layer, stage) pair(s) keep fewer than G_rec = 9 w)
  ok    test_g_rec_is_the_worst_aligned_codeword_not_the_average
  ok    test_loop_nest_separates_the_tile_from_the_consecutive_run
  ok    test_no_stage_or_transport_category_exceeds_the_embedded_reference
  ok    test_ondie_without_f_if_refuses_and_the_knobs_are_validated
  ok    test_only_the_weight_share_of_a_category_moves
  ok    test_placements_reconcile_by_hand_array_untouched_interface_x_k_over_n
  ok    test_r1_isolates_the_interface_saving_from_every_on_chip_saving
  ok    test_recon_optimizer_true_is_refused_rather_than_ignored
  ok    test_stream_packing_scales_by_k_over_n_and_aligned_packing_does_not
  ok    test_task3_checks_pass_on_a_consistent_record_and_catch_a_leak
  ok    test_the_complement_read_term_is_keyed_to_the_code_not_hard_wired
  ok    test_the_complement_register_is_n_minus_k_bits_and_costs_the_pe_nothing
  ok    test_the_dram_level_is_claimed_exactly_once_in_total
  ok    test_the_placement_space_check_is_recorded_on_every_result
  ok    test_the_placement_study_refuses_more_than_one_architecture
  ok    test_the_property_tests_also_ran_on_the_real_cache
  ok    test_the_saving_of_each_boundary_matches_its_closed_form
  ok    test_the_stem_is_the_recon_stem_and_keeps_a_layer_scope
  ok    test_weight_path_reads_totals_capacity_and_the_wire_split

1 skipped (run inside the container): test_full_width_really_moves_the_spad_reads_on_the_real_cache
all tests passed
```

```
$ bash hpc/tl.sh bash run.sh baseline --eval        # Task 1/2 totals must not move
  resnet18           raw     5,940.39 uJ   DRAM weight reads     16,356,544
  --- eyeriss_v2_like / resnet18 : conventional ECC, external BCH parity ---
    Timeloop energy       :    5,940.394 uJ
    + external parity     :    1,014.106 uJ
    = conventional ECC    :    6,954.500 uJ
```
Identical to the 2026-09-09 morning record (`20260909T060947Z__6a30f5c4.json`):
embedded 5,940.394 µJ, conventional 6,954.500 µJ.

```
$ ECC_RECON_MODELING=1 ECC_RECON_DECODE_SITE=controller bash hpc/tl.sh bash run.sh recon --eval
    bar                             total uJ   vs base    vs emb  DRAM i/f uJ     recon uJ  overhead uJ            N_rec
    R1 @ NoC source                5,949.212    14.46%    -0.15%       -0.000        8.817        0.000        2,077,021
    R2 @ cluster edge              5,935.739    14.65%     0.08%       -0.000        8.817        0.000        2,077,021
    R3 @ SPad input                5,937.067    14.63%     0.06%       -0.000       12.837        0.000        3,023,807
    R4a @ SPad output            UNSUPPORTED   infeasible local placement: 4 (layer, stage) pair(s) keep fe
    R4b @ SPad + reg             UNSUPPORTED   infeasible local placement: 4 (layer, stage) pair(s) keep fe
```
`results/tables/ReconSweep.csv` from this run compared column by column with
the pre-change CSV: **IDENTICAL TO THE DIGIT** on every shared column of every
row; the only difference is five new columns (`decode_site`, `f_if`,
`dram_array_uJ`, `dram_interface_uJ`, `dram_interface_saving_uJ`).

```
$ ECC_RECON_MODELING=1 bash hpc/run_all.sh --eval-only      # f_if unset -> refuses, ceiling first
  --- eyeriss_v2_like / resnet18 : the DRAM interface saving, at several f_if, because ECC_DRAM_IF_FRAC is unset ---
    DRAM weight energy (both shares)     :    1,046.819 uJ = 17.62% of the Timeloop total
    K/N = 0.6190, so the interface share falls by x 0.3810
      f_if   array (1-f_if) uJ   interface f_if uJ   interface x (1-K/N) uJ  of total
      0.10             942.137             104.682                   39.879     0.67%
      0.25             785.114             261.705                   99.697     1.68%
      0.50             523.409             523.409                  199.394     3.36%
    (every reconstruction boundary saves exactly the last column, in addition to its own on-chip saving; the reference bars do not move)
```

```
$ ECC_RECON_MODELING=1 ECC_DRAM_IF_FRAC=0.25 bash hpc/run_all.sh --eval-only   # the new figure
  DRAM weight inflation : baseline x1.6154 (+61.5%), embedded/recon x1.0
      stage                  kind        energy uJ           reads         fills         ingress  resident  reducible
      dram_array             dram          785.114      16,356,544             0               0     8,192 NO (array)
      dram_interface         dram          261.705      16,356,544             0               0     8,192        yes
      inter_cluster_mesh     network        35.367               0             0      16,356,544         0        yes
      cluster_local          network         7.064               0             0      23,812,480         0        yes
      weight_spad            storage       370.041   1,814,073,344    23,812,480               0         4        yes
    CEILING on any boundary's saving:
      on-chip weight energy that CAN be reduced :      412.472 uJ =  6.94% of the embedded total
      x (1 - K/N) = x 0.3810                     :      157.132 uJ =  2.65%   <-- the most ANY placement can save
      DRAM weight energy, both shares           :    1,046.819 uJ = 17.62%   (f_if = 0.2500; ECC_DRAM_IF_FRAC=0.25 (ECC_DRAM_IF_FRAC, set for this run; no cited LPDDR4 array/interface breakdown -- a sensitivity value))
        array share (1 - f_if), NOT reducible   :      785.114 uJ = 13.22%   (the complete codeword is read for on-die correction)
        interface share f_if, x K/N on EVERY bar:      261.705 uJ =  4.41%   (only the k message bits leave the die)
        interface x (1 - K/N)                   :       99.697 uJ =  1.68%   <-- the DRAM saving every boundary shares
      on-chip + interface, x (1 - K/N)          :      256.829 uJ =  4.32%   <-- the most ANY placement can save in total
    bar                             total uJ   vs base    vs emb  DRAM i/f uJ     recon uJ  overhead uJ            N_rec
    R1 @ NoC source                5,849.515    15.89%     1.53%      -99.697        8.817        0.000        2,077,021
    R2 @ cluster edge              5,836.042    16.08%     1.76%      -99.697        8.817        0.000        2,077,021
    R3 @ SPad input                5,837.370    16.06%     1.73%      -99.697       12.837        0.000        3,023,807
    R4a @ SPad output            UNSUPPORTED   infeasible local placement: 4 (layer, stage) pair(s) keep fe
    R4b @ SPad + reg             UNSUPPORTED   infeasible local placement: 4 (layer, stage) pair(s) keep fe
    (vs base / vs emb are SAVINGS: a negative number costs more than the reference; DRAM i/f is the interface energy the bar REMOVED against the embedded reference)
```

**Prediction check.** Predicted: every R bar's DRAM band drops by the same
f_if × (1−K/N) × DRAM weight energy; R1 goes from worse than embedded to better
by that amount minus its reconstruction cost; R2..R4b ordering unchanged.
Measured: 0.25 × 0.3810 × 1,046.819 = **99.697 µJ on R1, R2 and R3 alike**
(the DRAM weight energy on the 8x2 EDP mappings is 1,046.819 µJ, not the
886 µJ of the 2026-09-07 mappings); R1 from −8.817 µJ to +90.880 µJ =
99.697 − 8.817 against embedded (−0.148 % → +1.530 %); R2 (+1.757 %) > R3
(+1.734 %) > R1, the same order as before. R4a/R4b remain `unsupported` (G_rec).

Sensitivity, `vs embedded`, %: f_if = 0.10 → R1 +0.52 / R2 +0.75 / R3 +0.73;
0.25 → +1.53 / +1.76 / +1.73; 0.50 → +3.21 / +3.43 / +3.41; controller
(pre-change) → −0.15 / +0.08 / +0.06.

### f_if set to 0.40 by assumption (same day, later)

After the runs above, the user set `ECC_DRAM_IF_FRAC=0.40` as the env.sh default
("assuming interface energy would be 40 %"). The value is labelled ASSUMED in
the figure subtitle, the table, the manifest and every result file, and
`provenance.yaml` records that it is a decision, not a citation. Re-run:

```
$ bash hpc/tl.sh python3 -m eccenergy.tests.test_recon      # 36 ok, 1 skipped (R4b unsupported on this cache)
$ ECC_RECON_MODELING=1 bash hpc/run_all.sh --eval-only      # the figure now on disk
      DRAM weight energy, both shares           :    1,046.819 uJ = 17.62%   (f_if = 0.4000; ECC_DRAM_IF_FRAC=0.4, an ASSUMED value)
        array share (1 - f_if), NOT reducible   :      628.091 uJ = 10.57%
        interface share f_if, x K/N on EVERY bar:      418.728 uJ =  7.05%
        interface x (1 - K/N)                   :      159.515 uJ =  2.69%   <-- the DRAM saving every boundary shares
      on-chip + interface, x (1 - K/N)          :      316.648 uJ =  5.33%   <-- the most ANY placement can save in total
    bar                             total uJ   vs base    vs emb  DRAM i/f uJ     recon uJ  overhead uJ            N_rec
    R1 @ NoC source                5,789.696    16.75%     2.54%     -159.515        8.817        0.000        2,077,021
    R2 @ cluster edge              5,776.223    16.94%     2.76%     -159.515        8.817        0.000        2,077,021
    R3 @ SPad input                5,777.551    16.92%     2.74%     -159.515       12.837        0.000        3,023,807
    R4a / R4b                    UNSUPPORTED   (G_rec = 9 > resident tile in layer4.*)
```
Predicted before running: 0.40 × 0.3810 × 1,046.819 = 159.5 µJ on every R bar,
ordering unchanged. Both held.

### Still open after this revision

* **A cited f_if.** 0.40 is an assumption. Read the LPDDR4 I/O share off Ha
  2018 Fig. 4.8 (or derive it from a datasheet's read-burst I/O energy), record
  it in `provenance.yaml`, and replace the default; the figure rescales linearly
  and no ordering moves.
* The result file's `noc_share_within_published_band_eyeriss_v2` check still
  fails by 0.1 point (FINDINGS §4.1.3) — unrelated to this change.


**2026-09-09 — objective is EDP; v2 clusters 8x2; latch 0; band vs mesh only (FINDINGS.md §4.1.2). 8x2 whole-model mapping done (12 per-shape jobs, `hpc/map_by_shape.sh`); `ReconSweep.png` is the 8x2 / EDP / latch-0 result: interconnect 14.9% of on-chip, mesh only 5.9% (FINDINGS §4.1.3). R4a/R4b unsupported under these mappings.**

**2026-09-08 — NoC model revision implemented (FINDINGS.md §4.1).**
Inner-level router zero written explicitly (defect 8.1), declared mesh hop
lengths for v2, a bracketed per-PE latch (`ECC_NOC_PE_LATCH_PJ`, default 0.5),
evaluator-only spatial-reduction and psum-width terms (`eccenergy/noc_post.py`),
register writes costed (Aladdin's table had them at 0), and the component library
in the mapper fingerprint. **Every mapper cache is cold.** Decision pending:
`ECC_OPT_METRIC=edp`. Every number below is the 2026-09-07 model until the cold
pass has run.

**Last updated:** 2026-09-07 (Task 3 -- Eyeriss v2 reconstruction placement, evaluator only, fixed mapping -- implemented, evaluated, and then REVIEWED: is the K/N reduction applied, and is its magnitude right? FINDINGS.md §15 "Review, second pass")
**Migration:** COMPLETE. Single-process Apptainer flow verified (all five checks
passed; the six copied-cache energies match the laptop exactly), and parallel
execution is built and proven: `sbatch hpc/map.sbatch` ran 12/12 tasks, 0 failed,
258 shape-mappings in 48 minutes on 162 of the 181-core rewetz investment.
**Environment fixes:** hpc/tl.sh binds writable CACTI scratch under /blue;
`eccenergy/archs.py` writes the shared patched-arch YAML and globals.yaml
atomically so array tasks cannot read a partial file (content unchanged —
check 4 still reproduces the laptop totals exactly).
**Results status:** resnet18 (21/21 layers) and mobilenet_v2 (53/53 layers, 10
depthwise) are now evaluated WHOLE-MODEL on all six architectures at the
converged search, and the panel figure is regenerated and quotable. The
architecture ranking is preserved across both models. **The two development
layers are not a proxy for the model** — weight-stationary and input-stationary
swap between them.
**Model status:** Tasks 1, 2 and 3 are complete (Task 3 section next;
FINDINGS.md §14 and §15). The `embedded` arm is a measured DRAM-only result on
the Task 1 mappings, and the five reconstruction BOUNDARIES of Task 3 are now
measured on `eyeriss_v2_like` from those same mappings — with the result that
none of them beats embedded-only ECC under a fixed, ECC-unaware mapping, which
is what makes Task 4 the interesting one. `ecc.build_stacks()`'s three-arm
`recon` is untouched and is a different (single-placement) model; see CLAUDE.md
"The `recon` ARM is one point". Prior caveats still apply — the
`_wglb` bracketing pair was not run and six of the eight models are still
unmapped. NoC energy has been costed since 2026-09-07 (FINDINGS.md
"Interconnect"). run.sh defaults, architectures and the baseline are unchanged.
See FINDINGS.md §12 (bring-up) and §13 (the whole-model run) for evidence.

---

## 2026-09-07 TASK 3 — EYERISS V2 RECONSTRUCTION, EVALUATOR ONLY, FIXED MAPPING — DONE

**What was asked** (`03_staged_implementation_plan.txt` §3): with Task 2 as the
embedded-ECC reference, fix architecture = Eyeriss v2, model/layers, BCH
configuration, precision, physical hardware assumptions and mapping; vary only
the reconstruction placement and its necessary overhead; do not modify or rerun
the mapping optimiser; evaluate every feasible weight-path boundary present in
the actual model and mark unsupported ones explicitly.

### The result, first

On `eyeriss_v2_like` / resnet18, whole model, BCH(63,51):

| bar | total | vs conventional ECC | vs embedded only |
|---|---|---|---|
| Baseline (external parity) | 5,075.371 µJ | — | −5.773 % |
| Embedded (no external parity) | 4,798.375 µJ | +5.458 % | — |
| R1 @ NoC source | 4,805.637 µJ | +5.315 % | −0.151 % |
| R2 @ cluster edge | 4,801.036 µJ | +5.405 % | −0.055 % |
| R3 @ SPad input | 4,802.647 µJ | +5.373 % | −0.089 % |
| R4a @ SPad output | 5,672.121 µJ | −11.758 % | −18.209 % |
| **R4b @ SPad + reuse reg** | **4,733.027 µJ** | **+6.745 %** | **+1.362 %** |

**R4b wins and beats embedded-only ECC**, capturing 65.3 of the 77.5 µJ the
reducible weight path makes available (84 % of the ceiling). The ordering the
source discussion's ratings predict holds: R4b (5/5) > R2/R3 (4/5, break-even) >
R4a (3/5, 18 % worse than doing nothing). R1 being worse than embedded is
correct and is why it is 2/5 — it strips the bits and puts them straight back.

**The ceiling is the number every result has to be read against**, and it is now
printed first on every run: the whole on-chip weight path is 407 µJ of 4,798
(8.5 %), so K/N = 0.81 is worth at most 1.62 % of inference energy; 91 % of it is
the PE scratchpad, and DRAM's 18.5 % may not be reduced at all because the
complete codeword is read for correction.

**A first version of this model reported R4b as the WORST boundary — that was a
bug, found by review, and it is fixed.** It asked how many uses one weight gets
*in a row* (what a one-entry latch serves), which the loop nest says is 1 on
every mapping here. But §5.4's EV2-C is "reload/reconstruct when the required
weight CHANGES" — a retained reconstruction, so the question is capacity, not
consecutiveness. The loops below the scratchpad walk a tile of 16–256 weights
and repeat 1–6272 times, so a register covering that tile reconstructs once per
FILL: 82.85× fewer than R4a, and the same count as R3. **R4b is R3's encoder
count with R4a's storage saving**, which is exactly why it is rated above both.
FINDINGS.md §15 has the measurement, the sensitivity corners and the
capacity overhead this buys with (1.81× the baseline's PE weight storage).

### Two questions answered first, then the work

1. *Re-run the mapping optimiser?* **No, and it is refused.** Task 3 says not
   to, so `RECON_OPTIMIZER=False` is the only accepted value; `True` stops the
   run with an error naming Task 4 rather than quietly producing fixed-mapping
   numbers under a heading that claims otherwise. Every result carries
   `evaluation_only_rerun` and `fixed_mapping_shared_across_variants`.
2. *New architecture YAMLs per placement?* **No.** A placement changes only how
   the evaluator charges the weight path; nothing the mapper sees moves, so the
   mapping fingerprint is unchanged and the whole Task 1/2 mapper cache is
   reused as-is. YAML variants become necessary at Task 4, where the reduced
   width has to reach the mapper.

### One architecture at a time, on purpose

Section 4 of `env.sh` now takes precedence over section 3 when
`ECC_RECON_MODELING=1`: the run collapses to one architecture, one model and one
code. The three sweeps can put architectures on an axis because all three ECC
*arms* exist on every design; a reconstruction *boundary* does not — v2's are its
mesh, its cluster-local fanout and its PE scratchpad, and a weight-stationary
design's are a different list. Task 5 repeats the study per design, and adding
one means adding `WEIGHT_PATHS[<name>]` and `PLACEMENTS[<name>]` to
`eccenergy/recon.py` together.

### The weight path is read from the model, not assumed

`01_project_context_and_architectures.txt` §5.1 warns against substituting
"DRAM → global weight SRAM → RF → MAC" for Eyeriss v2. `eyeriss_v2_like` has
**no weight GLB** (its GLB banks are iacts and psums), so the path is four
stages and there is deliberately no global-buffer boundary. The stage-to-level
match is verified, not trusted: `weight_path_reconciles_with_raw_record`
re-derives the per-category weight energy from the cached stats and compares it
with the `Raw` record, and a level carrying weight energy that no stage claims
fails the run.

### The three things the model refuses to fudge

* **Physical packing** (§16 of file 02). `ECC_RECON_PACKING=stream` (default,
  and the layout `embedded.py` actually produces) packs the retained k bits with
  no per-weight alignment, so every reduced stage scales by exactly K/N.
  `aligned` gives each weight `ceil(8×51/63) = 7` whole bits; `floor(24/7) = 3`
  values fit a 24-bit scratchpad word, the same 3 as at 8 bits, so **the SRAM
  saving disappears entirely**. Measured: R4a goes −18.21 % → −19.76 %.
* **Reconstruction granularity** (§15). `G_rec = 9` is computed from the layout
  (a 63-bit codeword spans 7.875 weights and, starting mid-weight, touches 9).
  `ECC_RECON_ENCODER_GRANULARITY=weight` charges per weight rebuilt; `codeword`
  charges a whole codeword per access and multiplies every reconstruction cost
  by 7.875. Neither knob changes the sign of any comparison.
* **Feasibility.** A PE-local boundary whose resident tile is under `G_rec` is
  `unsupported` with the layers named, never estimated. Exercised on real data:
  mobilenet_v2's 7 depthwise layers hold 3 weights per PE, so R4a/R4b are
  rejected there while R1–R3 evaluate.
* **Retention** (`ECC_RECON_REUSE_REG_ENTRIES`, default `tile`). What R4b's
  register serves, read off the mapping rather than assumed. `1` (a latch) or
  `9` (a group buffer) collapse R4b onto R4a at −19.45 %, because a cyclic walk
  is the LRU worst case and there is no partial hit rate. Two of the three
  things R4b's +1.36 % is contingent on are exactly these knobs: under
  `aligned` packing it becomes −0.19 %, and under a latch −19.45 %.

### What is deliberately not credited

No DRAM saving (the complete protected codeword is read for correction —
checked per placement against the embedded reference). No capacity-driven reuse
and no re-tiling. And no operand-retention saving for anybody: R4b's register
would cut scratchpad reads, but so would the same register in a baseline PE,
which is the comparison the plan asks for, so read counts stay Timeloop's under
every bar and the register only reduces the reconstruction count while paying
its own write energy (0.0328 pJ, the cheapest per-PE register write in this
design's own Accelergy ERT — conservative, since a one-entry latch costs less
than the 24-entry file it comes from).

### Files changed

| file | change |
|---|---|
| `eccenergy/recon.py` | **NEW.** The placement space: `WEIGHT_PATHS`, `PLACEMENTS`, the stats/loop-nest re-parse, `Packing`, `Granularity`, `feasibility()`, `evaluate_placement()`, `cross_check()`, `reuse_register_pj()`. |
| `eccenergy/experiments/recon.py` | **NEW.** The driver: two reference bars from Task 1's and Task 2's own functions, five boundaries, the result file, the checks, the figure. |
| `eccenergy/tests/test_recon.py` | **NEW.** 15 offline tests on a synthetic `stats.txt`/`map.txt`, so nothing needs the mapper cache or the container. Two of them pin the corrected R4b behaviour in both directions: a tile-sized register must beat every other boundary, and a register under the working set must collapse onto R4a. |
| `env.sh` | section 4 rewritten (was a placeholder): `RECON_OPTIMIZER`, `ECC_RECON_PACKING`, `ECC_RECON_ENCODER_GRANULARITY`, `ECC_RECON_REUSE_REG_ENTRIES`, `ECC_RECON_REUSE_REG_PJ`, stem `ReconSweep`; section 10 collapses sections 3's lists onto section 4's point and flattens the per-arch placement list into `ECC_RECON_PLACEMENT_LIST` (a bash associative array cannot be exported). `ECC_RECON_PLACEMENT_LABELS`, `ECC_RECON_RESULTS_JSON` and `ECC_RECON_RERUN` removed — the labels are a property of the weight path and live with it in `recon.py`; the results go through the normal result store. |
| `eccenergy/config.py` | `recon` experiment; nine `recon_*` fields; `RECON_OPTIMIZER=True` refused; one-arch/one-model and `ECC_SPLIT_READ_WRITE=0` enforced for the study; `recon_title()`; the stem; banner rows. |
| `eccenergy/energy.py` | one new plotted category, `Recon overhead`, so a placement's buffer/control cost is visible rather than hidden inside `Reconstruction`. Zero for all three sweep arms, so `active_categories()` drops it there. |
| `eccenergy/plots/style.py` | its colour (both palettes) and legend label. |
| `eccenergy/plots/stacked.py` | `draw_panel()` gained `bars`, `bar_tags`, `bar_width`, `ref_totals` — **widened, not forked**, per CLAUDE.md. `write_table()` gained `ref_totals` and `extra_columns`, so the one table per stem carries the placement facts and its saving column measures against the right reference. Also fixed: a bar costing more than its reference printed `−-11.8%`; the sign now carries the direction. |
| `eccenergy/__main__.py`, `run.sh`, `hpc/run_all.sh` | route the new stage; `_plot` knows the placement study draws its own figure. |
| `CLAUDE.md`, `FINDINGS.md` (§15) | the structure and the numbers. |

`ecc.py`, `parity.py`, `embedded.py`, `experiments/baseline.py` and
`experiments/embedded.py` are **untouched**. Verified: Task 1 and Task 2
re-evaluate to 5,075.3712 µJ and 4,798.3746 µJ, identical before and after.

### Exact run commands

```bash
cd /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling
module load apptainer

ECC_RECON_MODELING=1 bash hpc/run_all.sh --eval-only          # the one command
ECC_RECON_MODELING=1 bash hpc/tl.sh bash run.sh recon --eval  # just this stage
ECC_RECON_MODELING=1 bash hpc/run_all.sh --replot             # figure only

ECC_RECON_MODELING=1 ECC_RECON_MODEL=mobilenet_v2 bash hpc/tl.sh bash run.sh recon --eval
ECC_RECON_MODELING=1 ECC_RECON_PACKING=aligned    bash hpc/tl.sh bash run.sh recon --eval
ECC_RECON_MODELING=1 ECC_RECON_ENCODER_GRANULARITY=codeword bash hpc/tl.sh bash run.sh recon --eval

bash hpc/tl.sh python3 -m eccenergy.tests.test_recon
```

Outputs: `results/figures/ReconSweep.{png,pdf}`,
`results/tables/ReconSweep.csv`, `results/manifests/ReconSweep.json`,
`results/evaluation/Pre/eyeriss_v2_like/<model>/…/*.json` (experiment
`task3_reconstruction_placement_fixed_mapping`).

### Assumptions a reader should be able to challenge

1. `stream` packing — defensible only because the embedded layout genuinely is
   a packed bit stream; `aligned` is one env var away and is reported above.
2. Encoder work charged per weight rebuilt at the per-codeword synthesis energy
   (`weight`); the pessimistic `codeword` reading is also reported.
3. Reconstruction 4.1296 pJ/codeword = 1.8995 incremental + 2.2301
   idle-per-cycle. The idle term is 54 % of it;
   `ECC_RECON_INCLUDE_IDLE=0` halves every reconstruction cost and still does
   not rescue R4a (≈ 437 µJ cost against a 77 µJ saving).
4. CSC metadata is not modelled — the design is modelled dense, so R4a/R4b's
   scratchpad saving is an upper bound on what a CSC v2 would see. It is a
   warning on every result file.
5. `archs/_shared/noc.yaml` declares v2's intra-cluster level `router_pj: 0.0`
   but the flattened architecture carries `router_energy: 0.0833` on
   `inter_PE_spatial`. Found while parsing the NoC stats, **not fixed** — it
   affects every bar equally and scales identically under `stream`, so it does
   not bias this comparison, but the declaration and `archs._inject_noc`
   disagree and one of them is wrong.
   *Mechanism found on the second-pass review:* `_inject_noc` writes
   `router_energy` only `if router:`, so a declared **zero is never emitted**
   and the level inherits its parent's value. Size: **44.94 µJ**, 20.5 % of the
   whole NoC and 0.94 % of the run. Correcting it moves the mapping fingerprint
   (whole cache cold) and Tasks 1/2's frozen totals, so it belongs in the next
   cold mapping pass, not here.
6. **The DRAM-to-accelerator link is not modelled**, and the 886.387 µJ DRAM
   weight term cannot be split to expose it: Accelergy's `CactiDRAM` is a flat
   `8 pJ/bit × width` LPDDR4 constant using only `(type, width)`. Stated as a
   limitation rather than guessed at. It is the one omission that could move the
   answer materially (a 25 % link fraction would lift the ceiling from 1.62 % to
   2.50 %), and no boundary in this study reaches it — R1–R4b all sit downstream
   of the link. FINDINGS.md §15 "Review, second pass" has the arithmetic.
7. **The `noc_share` FAIL may be a denominator question.** `audit.py` divides by
   a total that includes DRAM (4.58 %); against on-chip energy alone the share
   is 6.53 %, inside the paper's 6–10 % band. Which one JETCAS Fig. 18 uses is a
   fact about the paper and is not settled in this repo. Verdict left unchanged.

### What the second-pass review settled (2026-09-07)

**The K/N reduction is applied correctly and completely** — proved per stage and
per placement against the real mapper cache, not inferred from bar totals. The
mesh falls 24.158 → 19.557 µJ from R2 on, the cluster-local fanout
13.694 → 11.085 µJ from R3 on, the scratchpad 369.276 → 298.938 µJ at R4a/R4b,
each `before × K/N` to 4e-16 relative, with every untouched stage bit-identical.
The savings really are 4.60 and 7.21 µJ on a 4,798 µJ bar. `test_recon.py` grew
from 15 to **26** tests: seven property tests (per-stage exactness, non-weight
untouched, monotonicity down the path, the closed-form identity, nothing-goes-up,
a K sweep, and an independent re-scan for unclaimed weight levels), each run on
the synthetic stats **and** the real cache, and all six deliberate mutations of
`recon.py` are caught — one of them, a reduction hard-wired to 51/63 rather than
keyed to K, only by the new code sweep.

**One latent bug found and fixed.** `eyeriss_v2_like_wglb`'s weight path carries
a reducible `weight_glb` stage that no placement reduces, so four of its five
boundaries would have silently reported it at full width.
`recon.validate_placement_space()` now requires a placement's reduced set to be
a **prefix** of the path's reducible stages and every reducible stage to be
reached by some boundary; it is recorded on every result and raises before
evaluation. No `eyeriss_v2_like` number moves.

**`eyeriss_v2_like_wglb` is still unmeasured, and a cold map is costed.** From
SLURM job `41296638` at the current settings: `eyeriss_v2_like` × resnet18
(12 shapes) took **2 h 26 m** on 18 cores, × mobilenet_v2 (31 shapes) 3 h 00 m.
The `_wglb` variant adds a storage level and `ECC_VICTORY_SCALING=levels` raises
the budget with depth, so 2.5–4 h (≈45–70 core-hours) for resnet18 is a lower
bound. Worth doing — it is the only way to measure a global-buffer boundary at
all — but **not launched**: a cold mapping run needs an explicit go-ahead, and
its number is a BOUND, not Eyeriss v2.

**The figure was widened so the evidence is visible**, since the model is not
the thing to change. `plots/stacked.py::draw_panel` gained `bar_notes` (one
caller-supplied line per bar — the placement figure prints the µJ moved on chip
and the µJ paid for reconstruction, because a percentage cannot separate R2's
0.096 % from R3's 0.150 %) and `grouped_stacks(zoom_stacks=…)` (a second panel
drawn by calling `draw_panel` again, as `panels.py` already does, scoped to the
on-chip weight path: 407.1 → 402.5 → 399.9 → 336.8 µJ, so the scratchpad saving
is 19 % of the panel instead of 1.5 % of a bar). Still one image, one table, one
manifest, one bar-drawing routine — and `--replot` reproduces the three sweeps'
figures **byte-for-byte**.

### The next unfinished task

**Task 4 — Eyeriss v2 reconstruction WITH the mapping optimiser.** Set
`RECON_OPTIMIZER=True` (it currently refuses, by design) and make the reduced
width reach the mapper: represent the packed weight width per level per
placement, keep the physical SRAM/RF capacities fixed, and let the extra
effective capacity change the tiling. `ECC_PHASE=Post` is the namespace for it.

Two things §15 says Task 4 should be pointed at. First, R4b already needs 1.81×
the baseline's PE weight storage (6.48 b of reduced scratchpad plus 8 b of
register per resident weight), so a reconstruction-aware mapper has *less*
headroom than the reduced scratchpad alone suggests — the right question is
whether it can shrink the working set the register has to cover. Second, R4b's
whole saving is contingent on `stream` packing; a mapping-aware model that also
represents the packed width per level is where that assumption stops being an
assumption.

---

## 2026-09-07 TASK 2 — EMBEDDED ECC, DRAM ENERGY EFFECT ONLY — DONE

**What was asked** (`03_staged_implementation_plan.txt` §2): an embedded-ECC
evaluation mode with the parity inside the stored weights and no external
parity, using the *actual* embedded-codeword layout, reading the complete
codeword for correction, with mappings / on-chip widths / access counts / MAC
behaviour fixed, changing only the modelled DRAM storage/traffic energy, and
distinguishing stored bits from physical reads.

**Two questions answered first, then the work:**

1. *Re-run the mapping optimiser?* **No.** The mapper never sees parity: Task 1
   maps 8-bit weights against the architecture and adds the external parity in
   evaluation, after Timeloop. The embedded arm changes nothing the mapper
   sees either (DRAM and on-chip weights are 8 bits wide under both arms), so
   the fingerprint is unchanged, the cached mappings are reused, and the plan
   itself requires the fixed mapping for this comparison. Every Task 2 file
   records `evaluation_only_rerun: PASS` (0 newly mapped shapes) and
   `fixed_mapping_shared_across_variants: PASS`.
2. *Is external parity already in the baseline DRAM?* **Yes**, since Task 1:
   `parity.py` + `ecc.external_parity()`, 6 whole weights per BCH(63,51)
   codeword, 3 message-padding bits, tail padding, ×measured refetch, rounded
   to 64-bit DRAM words — 31.25% over payload traffic, i.e. 10.5 stored bits
   per weight rather than the flat n/k (8·63/51 = 9.88). Nothing was
   re-implemented there.

**The actual embedded layout** was read from the embedding code
(`ECC-CODE-Engine/4-EmbeddingECC/ecc_embed.py`, copied verbatim into
`../Input_Embedding/3-Testing/{utils,implementations}`): the int8 tensor is
one MSB-first bit stream cut into fixed 63-bit chunks (`chunk_size =
message_parity_size`), weights straddle chunk boundaries (7.875 weights per
codeword for every K; gcd(63,8)=1 so the alignment repeats every 8 codewords =
63 weights and 7 of 8 boundaries split a weight), only the final chunk is
zero-padded, and `ParityOverwriteByTopWeightsEncode` overwrites the n−k
lowest-significance chunk positions with the parity. At BCH(63,51) every
weight loses its LSB and about half lose bit 1 as well (1.524 parity bits per
weight). Storage is therefore exactly 8 bits per weight plus one tail pad per
tensor, and there is no external parity. This is the opposite packing rule to
the baseline's whole-weight codewords, and both are correct for their layout.

**What the embedded arm's DRAM energy is:** Timeloop's DRAM weight energy,
unchanged. The complete codeword is read (every one of the 8 bits, message and
parity), so no K/N reduction is applied at DRAM; what disappears relative to
Task 1 is the external-parity traffic and nothing else. The saving is
therefore `parity / (Timeloop + parity)` on each architecture.

### Files changed

| File | Change |
|---|---|
| `eccenergy/embedded.py` | **new.** `EmbeddedLayout` (bit-stream layout, phases, parity-significance profile), `account()` / `account_layers()` (stored bits, tail padding, straddles, external parity = 0, DRAM words), `traffic_account()` (complete-codeword reads, physical-word estimate), `dram_energy_pj()`, `hand_check()` / `hand_check_layers()` (codeword-by-codeword walk vs closed form). `LAYOUT_SOURCE` cites the files the layout was read from. |
| `eccenergy/ecc.py` | `embedded_dram()` added beside `external_parity()`; module docstring's embedded paragraph rewritten. `build_stacks()`, `external_parity()`, `savings()` **unchanged** (verified by diff). |
| `eccenergy/experiments/embedded.py` | **new.** Task 2's experiment `task2_embedded_ecc_dram_only`: baseline + embedded evaluated in ONE file from the same raw record, recon placements unavailable, six Task 2 checks, report. |
| `eccenergy/experiments/audit.py` | **new.** Task 1's checks / caveats / detail as functions, mirrored from `baseline.py` so `baseline.py` did not have to be touched. Keep the two in step. |
| `eccenergy/config.py` | `"embedded"` in `EXPERIMENTS`. No new knob: the layout is fixed by the pipeline, and the per-tensor grouping reuses `ECC_PARITY_GROUPING`. |
| `eccenergy/__main__.py` | runner + help text. |
| `run.sh` | documentation of the experiment and the quick-run command (no default changed). |
| `hpc/run_all.sh --eval-only` | writes `embedded --eval` beside `baseline --eval` per model (`ECC_EVAL_EXPERIMENTS`). Was `hpc/eval_panel.sh` until the env.sh consolidation. |
| `hpc/summary.py` | `--field embedded` and `--field saving`; two CSV columns. |
| `eccenergy/tests/test_embedded.py` | **new.** 11 offline tests (2 need pandas → container). |
| `CLAUDE.md`, `README.md`, `docs/RESULTS_SCHEMA.md`, `hpc/HIPERGATOR.md`, `FINDINGS.md` §14, `progress.txt` | documentation. |

**Not touched:** `eccenergy/parity.py`, `eccenergy/experiments/baseline.py`,
`energy.py`, `results_store.py`, `paths.py`, every `archs/` YAML, every
`run.sh` default, the recon arm.

### Exact run commands (HiPerGator; `--eval` never invokes Timeloop)

```bash
module load apptainer
export ECC_MAPPER_ALGORITHM=random_pruned ECC_MAPPER_SEARCH_SIZE= ECC_VICTORY=2000 \
       ECC_MAPPER_TIMEOUT=2000 ECC_MAPPER_THREADS=18 ECC_SWEEP=arch \
       ECC_SWEEP_ARCHS="eyeriss_v2_like eyeriss_like simple_weight_stationary simple_output_stationary simple_input_stationary simba_like"
# whole model, both panel models
ECC_LAYERS= ECC_CONST_MODEL=resnet18     bash hpc/tl.sh bash run.sh embedded --eval
ECC_LAYERS= ECC_CONST_MODEL=mobilenet_v2 bash hpc/tl.sh bash run.sh embedded --eval
# the development pair (quick run)
ECC_LAYERS="layer3.0.downsample.0 layer4.1.conv2" ECC_CONST_MODEL=resnet18 \
  bash hpc/tl.sh bash run.sh embedded --eval
# the matrix, and the tests
python3 hpc/summary.py --scope layers-full --field saving
bash hpc/tl.sh python3 -m eccenergy.tests.test_embedded
```

### Results (BCH(63,51), 8-bit weights, NoC costed, converged search)

Whole model — conventional ECC (Task 1) vs embedded ECC (Task 2), µJ:

| architecture | resnet18 conv. | embedded | saving | mobilenet_v2 conv. | embedded | saving |
|---|---:|---:|---:|---:|---:|---:|
| eyeriss_v2_like | 5,075.371 | 4,798.375 | 5.46% | 1,862.971 | 1,792.380 | 3.79% |
| eyeriss_like | 5,357.602 | 5,058.978 | 5.57% | 1,984.785 | 1,903.093 | 4.12% |
| simba_like | 5,699.198 | 5,401.230 | 5.23% | 1,996.279 | 1,924.891 | 3.58% |
| simple_input_stationary | 6,647.611 | 6,376.039 | 4.09% | 2,102.472 | 2,031.349 | 3.38% |
| simple_weight_stationary | 7,782.205 | 7,532.661 | 3.21% | 2,348.220 | 2,268.861 | 3.38% |
| simple_output_stationary | 9,363.620 | 9,103.499 | 2.78% | 2,639.039 | 2,569.644 | 2.63% |

resnet18 DRAM footprint, both arms: 11,678,912 weights; conventional
15,328,593 B stored (1,946,488 codewords, 10.5 b/weight), embedded
11,679,027 B (1,483,051 codewords, 8.0001 b/weight, 115 B of tail padding
over 21 tensors); DRAM weight reads identical (13.85 M scalars on v2,
refetch ×1.19). Development pair (2 layers, resnet18): savings 11.0% (v2),
10.9% (v1), 9.7% (Simba-like), 8.8% (WS), 8.5% (IS), 6.3% (OS) — a
development check, not a ranking.

**Baseline regression:** `bash run.sh baseline --eval` re-run after the change
on all 12 whole-model (architecture, model) pairs reproduces the pre-change
files exactly — total, per-component energies, parity accounting, validation
outcomes, mapping ids and summary all identical.

**Checks on every Task 2 file:** the six Task 2 checks and Task 1's checks
all pass; the only failing entry is the pre-existing
`noc_share_within_published_band_eyeriss_v2` (5.3% vs the 6–10% band), which
Task 1's files fail identically. Tests: 11/11 `test_embedded` (in the
container), 15/15 `test_results_store`, 12/12 `test_noc`, 4/4
`test_mapper_lock`.

### Assumptions and approximations, recorded in every result

* **Bit-proportional DRAM.** Timeloop bills 1/8 of a 64-bit word per 8-bit
  scalar; physical DRAM words are *estimated* as bits/64 rounded up once, and
  the final chunk's tail padding is reported but not charged — the same rule
  the baseline's payload is billed by. A burst-exact count is not available.
* **Per-tensor grouping** (`ECC_PARITY_GROUPING=layer`, shared with the
  baseline): each layer is its own bit stream and pays its own tail pad, as the
  embedding driver encodes one tensor per call.
* **ECC codec energy is outside the comparison** — charged to neither arm. The
  arms would decode different codeword counts (1.95 M vs 1.48 M on resnet18),
  so a codec cost would not cancel; it is simply not costed until a
  characterised codec energy is adopted. `ECC_DECODE=1` affects the sweep
  figures only and is warned about.
* No on-chip saving is credited to the embedded arm and no inference-accuracy
  claim is made about overwriting weight LSBs.
* `build_stacks()`'s embedded bar was already `raw.total` and is unchanged;
  `embedded_matches_sweep_figure_arm` checks the JSON against it on every
  file. Its decode *count* for the embedded arm still uses the baseline's
  6 weights/codeword (`ECC_EMB_WEIGHTS_PER_CW`); that is a codec-energy
  question and was deliberately not changed here.

### Next: Task 3 — Eyeriss v2 reconstruction, evaluator only, fixed mapping

Not started. Uses the Task 2 file as the embedded-ECC reference. The recon
arm in `build_stacks()` is still the placeholder the user will correct first;
`experiments/audit.py` is the shared check set Task 3 should reuse, and
`EmbeddedLayout.parity_significance_profile()` already says which weight bits
are dependent per codeword phase, which Task 3's reconstruction counts need.

---

## 2026-09-06 AUDIT ADDENDUM — read this before anything below

The full write-up is `FINDINGS.md`; the status board is `progress.txt`.

1. **Depthwise/grouped convolutions were modelled wrongly and are now fixed.**
   The loader dropped `groups`, so a MobileNetV2 depthwise layer reached
   Timeloop as a 1-input-channel convolution (81 input scalars instead of
   77,760). `Layer.G` + a grouped problem shape fix it; MAC counts now match
   published values for all eight CNNs. Every model with depthwise layers
   (mobilenet_v2, efficientnet_b0, convnext_tiny, xception) needs its
   depthwise shapes re-mapped. ConvNeXt's per-pixel `nn.Linear` layers had a
   second, similar defect (P=Q=1); also fixed, `convnext_tiny` regenerated.
2. **The mapper search decides the ranking.** On the two Task 1 development
   layers, `hybrid`/victory 100 (the Task 1 setting, and `hybrid` is still the
   `run.sh` default) gives Eyeriss v1/v2 1979/2037 µJ; `random_pruned` with
   `search_size 20000`/victory 2000 gives 390/402 µJ on the same layers and
   the ordering inverts (v1 ≈ v2 < Simba-like < WS < IS < OS). The quick-run
   table further down this file is the first setting and must not be quoted.
   **Decision needed:** change the default search in `run.sh`.
3. **NoC energy is now costed** (was zero everywhere: Timeloop's wire model is
   a stub). `archs/_shared/noc.yaml` supplies one shared 45nm wire constant and
   per-architecture switching terms, injected onto the spatial containers so
   the Legacy networks carry energy inside the mapper's objective. It is a
   plotted category of its own; the mapper cache slug gains `noc`, so every
   architecture needs a fresh cold run. FINDINGS.md "Interconnect (NoC) model".
4. **Cache-key drift:** the fingerprint recipe changed after Task 1
   (`search_size`, `max_permutations` added), so the Task 1 evaluation JSONs
   under `results/evaluation/.../map-energy-vic100l-hybrid-seednone/` can no
   longer be re-evaluated from their mapper cache, and new evaluations land
   under `...-seednone-ssnone-perm16/`. Documented, not repaired.
5. **Migration decided:** local mapping is too slow (~12 h for the two
   full models), so the project moves to HiPerGator
   (`/blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling`). Instructions are
   in `hpc/HIPERGATOR.md`; the order is transfer → verify the single-process
   flow → then design parallel execution there. The local chained run was
   stopped at 15:15.
6. Files changed this session: `eccenergy/workloads.py`, `timeloop.py`,
   `energy.py`, `generate.py`, `tests/test_results_store.py` (stale slug),
   `ecc_energy_study/model_layers.json` (convnext_tiny only), plus
   `FINDINGS.md`, `progress.txt`, this header and structural notes in
   `CLAUDE.md`. No arch YAML, no `run.sh` default, no ECC arm was changed.

Everything below this line is the Task 1 session's own report, kept verbatim
for the record; where it quotes energies they are the superseded numbers.

---

---

## What Task 1 was asked for, and where it stands

| Requirement | State |
|---|---|
| Organise the architectures consistently, shared settings + per-design config | done — `archs/_shared/standard.yaml` + `provenance.yaml`, checked by `bash run.sh validate` |
| Same weight/activation precision, workload, BCH config across comparisons | done — enforced, not asserted; all 7 designs pass with 0 violations |
| Keep the precisions explicit and distinct | done — banner, validator and every result JSON report all three separately |
| Preserve hierarchy, capacity, weight bypass rules | done — printed by `validate` as "preserved differences"; nothing equalises them |
| Conventional-ECC baseline only, parity external in DRAM | done — `eccenergy/experiments/baseline.py` |
| Account for grouping, padding, physical DRAM transfer granularity | done — `eccenergy/parity.py` |
| External parity NOT stored in on-chip weight SRAM/RF | done — the baseline does not touch on-chip energy; the old switch is off and warns |
| Each architecture completes the selected-layer quick run | done — 6/6, both layers, 0 failures |
| Configuration summary reports precisions and BCH parameters | done — see the banner output below |
| Hand-check confirms payload/parity accounting | done — mechanised, runs on every result, recorded in the JSON |
| Existing baseline preserved apart from documented standardization | see **Deliberate changes to existing behaviour** below |

### One deliberate correction to the task text

Task 1 says to standardize accumulation precision. **You directed otherwise on
2026-09-06**, and that direction is what is implemented: weight precision,
activation precision, workload, BCH configuration and methodology are
standardized; **each design keeps its published accumulator width**, because
psum precision is an architectural property and equalising it would equalise
the architectures. All ECC arms of one design share its width. A forced common
width is available as an explicitly-labelled sensitivity study (`ECC_ACC_BITS`)
with its own mapper cache and results namespace, and is never the primary
result.

You also asked that the Simba 24-bit value not be trusted just because the YAML
said so. It is now traced, along with every other accumulator width — see
**Provenance** below.

---

## Files changed

### Architectures

| File | Change |
|---|---|
| `archs/_shared/standard.yaml` | **new.** The comparison contract: what must be identical across designs, and each design's own accumulator width with its citation. |
| `archs/_shared/provenance.yaml` | **new.** Level by level, where each declared number came from, with the quote and the date it was verified. |
| `archs/simple_input_stationary/arch_paper.yaml` | **new.** Authored as the exact mirror of the WS/OS siblings. |
| `archs/simple_input_stationary/README.md` | **new.** |
| `archs/eyeriss_v2_like/arch_paper.yaml` | corrected — see below. |
| `archs/eyeriss_v2_like_wglb/arch_paper.yaml` | same corrections mirrored. |

### Package

| File | Change |
|---|---|
| `eccenergy/parity.py` | **new.** External BCH parity accounting + the hand check. |
| `eccenergy/results_store.py` | **new.** The centralized result writer. |
| `eccenergy/experiments/baseline.py` | **new.** Task 1's experiment. |
| `eccenergy/experiments/validate.py` | **new.** The contract check. |
| `eccenergy/tests/test_results_store.py` | **new.** 15 offline tests. |
| `eccenergy/config.py` | layer selection, `Pre`/`Post`, precision knobs, parity knobs, cache strictness, `simple_input_stationary`, precisions in the banner. |
| `eccenergy/paths.py` | the evaluation namespace; fingerprinted raw and mapper caches. |
| `eccenergy/archs.py` | contract loaders, `arch_fingerprint()`, `validate_arch()`, the accumulator-override patcher, **globals node fix**. |
| `eccenergy/workloads.py` | `select_layers()` by stable name, `layer_identities()`. |
| `eccenergy/timeloop.py` | mapping sidecars, `mapping_id`, tool versions, content-verified cache reuse, post-mapping stats-path bug fix. |
| `eccenergy/energy.py` | per-layer and per-level detail in `Raw`; fingerprint threaded through the cache. |
| `eccenergy/ecc.py` | baseline uses `external_parity()`; one shared codeword geometry. |
| `eccenergy/experiments/common.py` | layer selection, fingerprints, mapper records. |
| `eccenergy/experiments/diagnose.py` | reads the fingerprinted raw cache. |
| `eccenergy/__main__.py` | the three new experiments, `--layers/--eval/--overwrite/--phase`. |
| `run.sh` | development-mode knobs, quick-run commands, `simple_input_stationary` in the default sweep. |
| `docs/RESULTS_SCHEMA.md` | **new.** |
| `CLAUDE.md` | structure updated: new commands, new layout, the contract, the fingerprint. |

---

## Exact run commands

```bash
DEV_LAYERS="layer3.0.downsample.0 layer4.1.conv2"
```

**1. Validate the architectures** (no container, no cache, seconds):

```bash
bash run.sh validate
```

**2. Generate the mappings** (container; ~3.5 min for 6 archs x 2 layers):

```bash
docker run --rm -v "<project>":/home/workspace -w /home/workspace \
  -e ECC_LAYERS="$DEV_LAYERS" -e ECC_VICTORY=100 -e ECC_MAPPER_THREADS=8 \
  -e ECC_SWEEP=arch \
  timeloopaccelergy/timeloop-accelergy-pytorch:latest-amd64 \
  bash -lc "cd /home/workspace && bash run.sh map"
```

**3. Evaluate** (no container, seconds):

```bash
ECC_LAYERS="$DEV_LAYERS" ECC_VICTORY=100 ECC_MAPPER_THREADS=8 \
ECC_SWEEP=arch bash run.sh baseline --eval
```

**4. Tests** (no container):

```bash
python3 -m eccenergy.tests.test_results_store
```

---

## Results

### Contract validation — 7/7 architectures, 0 violations

```
  architecture                acc b   fanout   on-chip weights  GLB keeps W
  eyeriss_like                   16      168            75,264        False
  eyeriss_v2_like                20      384            55,296        False
  eyeriss_v2_like_wglb           20      384           120,832         True
  simple_weight_stationary       16      256           164,096         True
  simple_output_stationary       16      256            65,792         True
  simple_input_stationary        16      256            65,792         True
  simba_like                     24      256         2,113,536        False
```

Run against `ECC_ARCH_FIDELITY=stock` the same validator reports the defects the
paper YAMLs exist to fix — `simple_weight_stationary` 5 violations (datawidth 16
on DRAM and every operand level), `simba_like` 2 (a shared psum/operand buffer,
and `adder_width: 16` against a 24-bit accumulation buffer), `eyeriss_like` 1
(the shared 8-bit GLB holding 16-bit partial sums). That is the mechanism
working, on exactly the class of defect that prompted this work.

### Selected-layer quick run — 6/6 architectures, both layers, 0 failures

resnet18, `layer3.0.downsample.0` + `layer4.1.conv2`, 2,392,064 weights,
BCH(63,51) t=2, 8-bit weights and activations.

| architecture | acc | Timeloop µJ | + parity µJ | conventional ECC µJ | DRAM refetch | pJ/DRAM weight |
|---|---|---|---|---|---|---|
| eyeriss_v2_like | 20b | 2,036.79 | 339.48 | 2,376.27 | ×7.10 | 64.0000 |
| eyeriss_like | 16b | 1,979.32 | 331.61 | 2,310.93 | ×6.93 | 64.0000 |
| simple_weight_stationary | 16b | 731.96 | 47.84 | 779.80 | ×1.00 | 64.0000 |
| simple_output_stationary | 16b | 904.31 | 47.84 | 952.15 | ×1.00 | 64.0000 |
| simple_input_stationary | 16b | 581.62 | 47.84 | 629.46 | ×1.00 | 64.0000 |
| simba_like | 24b | 3,166.24 | 51.77 | 3,218.01 | ×1.08 | 64.0000 |

**`pJ/DRAM weight` is identical across all six.** That column is the
apples-to-apples check: at the same node and DRAM datawidth it must be, and the
original complaint was that it was not. It is now checked on every run.

**These are two-layer development numbers. They are not an architecture
ranking**, and every result JSON says so in `warnings`. Two layers out of
twenty-one, chosen for speed and contrast, cannot rank six accelerators.

### Payload/parity hand check

Identical for every architecture, because it depends only on the workload and
the code:

```
payload    2,392,064 weights = 2,392,064 B
codewords  398,678            (6 whole weights per 51-bit message field)
parity       598,017 B
padding      149,508 B        (3 b/codeword message + 32 b tail)
overhead      31.25% over payload
```

A flat `n/k - 1` model charges **23.53%**. The difference is the 3 message bits
per codeword that no whole 8-bit weight can use, plus tail padding. The check
recomputes all of this codeword by codeword and compares against the closed
form; it passes for every architecture and is recorded inside each result JSON
under `validation`, so it is auditable after the fact.

The one-layer run shows **31.27%** — tail padding is a larger share of a
32,768-weight tensor. That is the granularity accounting doing its job.

### Result files

Twelve JSONs under `results/evaluation/Pre/…`, e.g.

```
results/evaluation/Pre/eyeriss_v2_like/resnet18/bch63_51/w8a8/
  layers2__layer3_0_downsample_0__layer4_1_conv2/
  map-energy-vic100l-hybrid-seednone/20260906T072817Z__303cee78.json
```

Each holds all seven variants: the conventional baseline `evaluated`, and
`embedded_ecc` plus the five reconstruction placements as `not_implemented`
with `total_energy_pJ: null` and a reason. Nothing is fabricated.

### Tests

15/15 pass, offline.

---

## Provenance

Verified against the primary sources on 2026-09-06 and recorded in
`archs/_shared/`:

| design | accumulator | source |
|---|---|---|
| Eyeriss v1 | 16b | JSSC 2017 §IV: *"a 16-b two-stage pipelined multiplier and adder … truncated from 32 to 16 b"* |
| Eyeriss v2 | 20b | JETCAS 2019 Table IV: *"psums: 20b fixed-point"*, repeated in Table V |
| Simba | 24b | *"4 tera-ops per second (TOPS) using 8-bit weights and activations and 24-bit accumulation"* |
| WS / OS / IS | 16b | **not published.** Reference dataflows, no paper; recorded as `locally_authored` |

Also verified and recorded: Eyeriss v1's 108 kB GLB in 25 banks of 512-b × 64-b,
the 8 kB filter allocation (and why it is *not* modelled as a reuse level), the
224×16b filter spad and 12×16b / 24×16b ifmap and psum spads; Eyeriss v2's
Table IV scratchpad sizes and Fig. 17 arrays.

**Not verified:** Simba's buffer and PE geometries below the MAC come from
`timeloop-accelergy-exercises`, not from a table checked against the paper.
Only its precisions are traced. Recorded as `not_verified` in
`archs/_shared/provenance.yaml` rather than presented as a citation.

---

## Deliberate changes to existing behaviour

Task 1 says the existing baseline is preserved "apart from documented
standardization and parity accounting". These are the changes, and each is a
correction rather than a preference. **All of them move numbers.**

1. **Parity accounting** — from a flat `n/k - 1` (23.53%) to accounted grouping,
   padding and DRAM granularity (31.25%). This is the parity accounting Task 1
   asks for.

2. **DRAM was costed at 65 nm while every accelerator was at 45 nm.**
   `globals.yaml` defaulted to `65nm`; the containers all declare `45nm`. DRAM
   is where most of the energy and all of the parity cost lands. It was uniform
   across architectures, so it never showed as an ordering error — it made every
   absolute number and every savings percentage wrong by a fixed factor. The
   node now comes from the same contract file the architectures are validated
   against.

3. **Eyeriss v2's iact spad: 16 → 24 entries.** Table IV says *"iact data: 24B
   (Reg)"*. The file was reading it CSC-natively (one value per 12b word, 16
   iacts) while reading its own weight spad dense-repacked (288 B → 288
   weights). One design cannot use two packing rules — that is the same class
   of defect as comparing two designs at different datawidths. The dense rule
   now applies to every level of every design.

4. **Eyeriss v2's weight-spad citation was wrong even though the number was
   right.** It cited §III-C, *"each port has a bitwidth of 24 bits such that it
   can send and receive three 8b uncompressed iact values"* — that sentence is
   about a **router port**, not the spad, and §III-D says the spad stores 12b
   count-data pairs. The capacity (288 weights/PE) now comes from Table IV's
   288 B plus the study-wide dense rule. The number is unchanged; the
   justification is now one that survives reading the paper.

5. **Codeword counting used `k / weight_bits` (6.375) instead of
   `k // weight_bits` (6).** A weight cannot straddle a codeword. This made the
   codeword count ~6% low wherever it was used.

6. **The mapper cache is now content-addressed.** It was keyed on a treatment
   slug, so editing an `arch.yaml` left the path unchanged and the next run
   reused mappings computed for the previous geometry. Entries now carry a
   `mapping.json` sidecar with an architecture fingerprint; a mismatch is a
   miss.

### The cost of change 6, and how to avoid paying it twice

Pre-Task-1 cache entries have no sidecar, so **nothing can prove they match the
current YAMLs** — and for `eyeriss_v2_like` they demonstrably do not, because
its iact spad changed. They are **left on disk** (43 shapes × 5 architectures,
hours of compute) and are simply not used by default.

A full-model run therefore needs a fresh mapper cache. It is resumable and
nothing already computed is lost. `ECC_CACHE_STRICT=0` will accept the old
entries; results built that way are labelled `legacy` and warned about, and
should not be published.

---

## Known limits of what Task 1 produced

Everything here is recorded in the result JSONs under `approximations` and
`warnings`; it is repeated here so a reviewer does not have to open one.

* **Two layers are not a ranking.** 2 of 21 resnet18 layers, ~20% of the
  weights. They validate the implementation.
* **Parity traffic scaling is approximate.** Timeloop reports scalar reads, not
  the tile boundaries they fall on, so the exact stored codeword count is scaled
  by the measured DRAM refetch factor and rounded to whole DRAM words **once**,
  at the end, rather than per tile. This understates granularity rounding when
  tiles are small relative to a codeword.
* **Parity is billed at the measured per-access cost of a DRAM weight read on
  that architecture**, not at an independently modelled parity-region access
  cost. Exact if parity is read from the same DRAM by the same controller, which
  is the conventional-ECC assumption.
* **Timeloop v4 exposes no random seed.** `ECC_MAPPER_SEED` is recorded but
  cannot make the search deterministic. What bounds it is `ECC_VICTORY` and
  `ECC_MAPPER_THREADS`, both in the fingerprint — so pin the thread count.
  Leaving it empty means "all cores", which differs per machine.
* **Eyeriss v2's 12-bit register words cannot be expressed** at datawidth 8
  (Timeloop asserts `width % datawidth == 0`). The iact spad is modelled as
  24×8b: same bits, same operands, narrower words, so its per-access register
  energy is for a narrower word than the design has. The only such case in the
  study.
* **The 288-vs-192 weights/PE reading of Eyeriss v2's weight spad** is the
  largest single modelling judgement here — a third of v2's on-chip weight
  capacity, which directly drives its DRAM refetch. The dense rule gives 288;
  a CSC-native reading gives 192. Recorded in
  `archs/_shared/provenance.yaml` under `contested` so it can be disagreed with.
* **Simba is a 16 nm design modelled at 45 nm**, the largest node deviation.
  Not corrected: correcting one design would reintroduce the mixed-node
  confound.
* **`legacy/figures/*_summary_uJ.csv` will not reproduce.** Changes 1, 2, 3 and
  5 above all move numbers deliberately. As `CLAUDE.md` says, decide whether a
  mismatch is intended before treating it as a regression — here it is intended.

---

## Next (as written at the end of Task 1): Task 2 — embedded ECC, DRAM energy effect only

**Done on 2026-09-07 — see the Task 2 section at the top of this file.** What
it needed, as listed then:

1. The **actual embedded-codeword layout**, not "all 8 weight bits are
   independent payload". `parity.py` has the geometry; the embedded arm needs
   its own layout description.
2. The complete embedded codeword must be **read** for correction — its parity
   is not discarded before the correction.
3. Mappings, on-chip widths, access counts and MAC behaviour held **fixed**
   against Task 1. Evaluation-only reruns already do this: `bash run.sh baseline
   --eval` never invokes the mapper, and the `fixed_mapping_shared_across_variants`
   check in every result will fail loudly if a variant is mapped differently.
4. Report DRAM payload, parity/padding, physical accesses and energy; every
   non-DRAM component must match Task 1 exactly.
5. Document any bit-proportional approximation. Fewer stored bits do not
   guarantee fewer DRAM bursts.
6. Do not change ECC codec energy, or keep it explicitly outside the comparison.

The `embedded_ecc` variant already exists in every Task 1 result file as
`not_implemented`; Task 2 fills it in. The document shape does not change.
