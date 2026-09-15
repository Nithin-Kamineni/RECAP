# GUARDS.md — what is protected, and by what

**GENERATED. Do not edit.** `make guards` rewrites this file from
`eccenergy/settings/guards.py` and from the call sites in the package;
`tests/contract/test_guards.py` fails when it is stale.
`ProjectRestructure.md` §6 is the design.

A **guard** runs on *every* run, including a SLURM job at 3am, and stops it.
A **test** runs when you type `pytest` and prints red. Both were invisible
until phase 6; this file is the one screen that answers *what is there*.

## The tiers, and the only thing that matters about them

| tier | means | overridable? |
|---|---|---|
| **1 PARSE** | the string does not mean anything | **never** |
| **2 IMPOSSIBLE** | physically or arithmetically cannot hold | **never** |
| **3 COUPLING** | encodes the combinations we thought of | yes — `ECC_ALLOW` |
| **4 DERIVED** | refuses you for setting what is also derived | yes — `ECC_ALLOW` |

```
   ECC_ALLOW="zero-price,recon-no-split-read-write"   <- names guards, never a blanket off
```

An override is **recorded**: it lands on the run manifest as `guard_overrides`
and on the figure's caveat list. You cannot ablate by accident, and you cannot
publish an ablation without the figure saying it is one.

**139 invariants over 153 refusal sites.** tier 1 (PARSE): 29, tier 2 (IMPOSSIBLE): 106, tier 3 (COUPLING): 3, tier 4 (DERIVED): 1

## Tier 1 — PARSE  *(never overridable)*

| id | raises | refuses | where |
|---|---|---|---|
| `design-order-not-integer` | `ConfigError` | `order:` in a design.yaml that is not an integer | `arch/design.py:301` |
| `design-yaml-parse` | `ConfigError` | a design file that is not valid YAML | `arch/design.py:84` |
| `generate-usage` | `SystemExit` | `python3 -m eccenergy.arch.generate` with no known mode | `arch/generate.py:188` |
| `leakage-not-a-table` | `ConfigError` | a design.yaml `leakage_nw:` that is not density -> nW | `arch/design.py:318` |
| `not-a-boolean` | `ConfigError` | a boolean knob that is not 1/0/true/false | `settings/env.py:43` |
| `not-a-key-value-list` | `ConfigError` | an ECC_LAYERS per-model entry that is not `model=layer layer` | `settings/env.py:145`<br>`settings/env.py:172`<br>`settings/env.py:224`<br>`settings/env.py:159` |
| `not-a-key-value-table` | `ConfigError` | an `ECC_*_LIST` entry that is not `key=number` | `settings/env.py:91` |
| `not-a-number` | `ConfigError` | a float knob that is not a number | `settings/env.py:65` |
| `not-an-integer` | `ConfigError` | an integer knob that is not an integer | `settings/env.py:54` |
| `one-value-not-a-list` | `ConfigError` | a list where ONE value is meant (only the swept axis takes a list) | `settings/env.py:184`<br>`settings/env.py:212` |
| `unknown-approach` | `ConfigError` | ECC_APPROACHES naming an arm that does not exist | `config.py:223` |
| `unknown-arch-fidelity` | `ConfigError` | ECC_ARCH_FIDELITY naming no known fidelity | `config.py:591` |
| `unknown-capacity-scope` | `ConfigError` | ECC_WEIGHT_CAPACITY_SCOPE naming no known scope | `config.py:551` |
| `unknown-classify` | `ConfigError` | ECC_CLASSIFY that is neither `instances` nor `name` | `config.py:541` |
| `unknown-encoder-granularity` | `ConfigError` | ECC_RECON_ENCODER_GRANULARITY naming no known granularity | `config.py:405` |
| `unknown-encoder-site` | `ConfigError` | ECC_RECON_ENCODER_SITE naming no known site | `config.py:440` |
| `unknown-experiment` | `ConfigError` | ECC_EXPERIMENT naming no known experiment | `config.py:186` |
| `unknown-format` | `ConfigError` | ECC_FORMATS naming a format matplotlib is not asked for | `config.py:610` |
| `unknown-knob` | `ConfigError` | `cfg.with_()` naming a knob no settings group declares | `config.py:716` |
| `unknown-metric` | `ConfigError` | ECC_METRICS naming a figure row that does not exist | `config.py:208`<br>`study/metrics.py:259` |
| | | ↳ the PLOTTED metric, not the mapper's `ECC_OPT_METRIC` -- note `latency` here against `delay` there. | |
| `unknown-opt-metric` | `ConfigError` | ECC_OPT_METRIC naming no known objective | `config.py:595` |
| `unknown-palette` | `ConfigError` | ECC_PALETTE that is neither `house` nor `cvd` | `config.py:606` |
| `unknown-parity-grouping` | `ConfigError` | ECC_PARITY_GROUPING naming no known grouping | `config.py:381` |
| `unknown-recon-default` | `ConfigError` | ECC_RECON_DEFAULT naming something that is not a reconstruction placement | `config.py:245`<br>`config.py:257` |
| | | ↳ what a bare `recon` bar MEANS. The abstract `recon` arm is retired (EnvReorganisation 6.2), so there is no value here that means 'not a placement'. | |
| `unknown-recon-packing` | `ConfigError` | ECC_RECON_PACKING naming no known packing | `config.py:401` |
| `unknown-sweep` | `ConfigError` | ECC_SWEEP naming no known axis | `config.py:192` |
| `unknown-victory-scaling` | `ConfigError` | ECC_VICTORY_SCALING naming no known scaling | `config.py:599` |
| `values-unknown-sweep` | `SystemExit` | `--values` for a sweep that has no knob | `__main__.py:77` |
| `widths-not-a-table` | `ConfigError` | a widths.yaml that is not q -> {spad_width, glb_width} | `arch/design.py:254`<br>`arch/design.py:269` |

## Tier 2 — IMPOSSIBLE  *(never overridable)*

| id | raises | refuses | where |
|---|---|---|---|
| `acc-bits-not-narrower` | `ConfigError` | an accumulator narrower than the operands it accumulates | `config.py:376` |
| `accelergy-wrote-no-ert` | `SystemExit` | Accelergy producing no ERT/ART at all | `toolchain/ert.py:352` |
| `activation-bits-positive` | `ConfigError` | ECC_ACTIVATION_BITS <= 0 | `config.py:368` |
| `approaches-empty` | `ConfigError` | ECC_APPROACHES empty -- nothing to compare | `config.py:227` |
| `area-chip-never-mapped` | `ConfigError` | ECC_METRICS=area on a chip with no ART in the mapper cache | `study/metrics.py:175`<br>`study/metrics.py:254` |
| `arm-half-mapped` | `SystemExit` | an arm with only some of its shapes mapped (two chips in one bar) | `study/placement_study.py:184` |
| `boundary-declares-no-ert-row` | `SystemExit` | moving a toll out of a boundary Timeloop never billed | `study/placement_study.py:288` |
| `capacity-scale-positive` | `ConfigError` | ECC_WEIGHT_CAPACITY_SCALE <= 0 | `config.py:544` |
| `clock-positive` | `ConfigError` | a declared clock rate that is not positive | `arch/design.py:309`<br>`config.py:1319` |
| `const-arch-empty` | `ConfigError` | a held architecture axis with no ECC_CONST_ARCH | `config.py:312` |
| `const-model-empty` | `ConfigError` | a held model axis with no ECC_CONST_MODEL | `config.py:324` |
| `datawidth-le-weight-bits` | `ConfigError` | a REDUCED weight wider than the full one | `config.py:586` |
| `datawidth-levels-unknown` | `ValueError` | ECC_WEIGHT_DATAWIDTH_LEVELS naming a level the design has not got | `arch/patch.py:792` |
| `datawidth-positive` | `ConfigError` | ECC_WEIGHT_DATAWIDTH below 1 bit | `config.py:578` |
| `dc-clock-mismatch` | `SystemExit` | a DC datapath entry measured at another clock period | `physics/recon_dc.py:128` |
| `depth-scale-positive` | `ConfigError` | ECC_WEIGHT_DEPTH_SCALE <= 0 | `config.py:560` |
| `design-dir-missing` | `ConfigError` | a named design with no directory | `arch/design.py:116` |
| `design-free-levels-shape` | `ConfigError` | `mapspace_free_levels:` that is not dimension -> [levels] | `arch/design.py:295` |
| `design-inputs-not-written` | `SystemExit` | a mapper input that `write_globals()` never wrote | `toolchain/inputs.py:263` |
| `design-label-empty` | `ConfigError` | a design with no `label:` for a figure axis | `arch/design.py:291` |
| `design-name-mismatch` | `ConfigError` | a design.yaml whose `name:` is not its directory | `arch/design.py:288` |
| `design-not-a-mapping` | `ConfigError` | a design.yaml that is not a mapping of fields | `arch/design.py:285` |
| `design-todo` | `ConfigError` | a scaffolded design still saying TODO where evidence goes | `arch/design.py:344` |
| `designs-dir-missing` | `SystemExit` | the cloned exercises repo lacking its designs directory | `toolchain/inputs.py:210` |
| `dilation-needs-layers` | `SystemExit` | a dilation spot check with no layer named | `report/dilation_view.py:186` |
| `dilation-unknown-layer` | `SystemExit` | a dilation spot check naming a layer the model does not have | `study/dilation.py:274`<br>`study/dilation_tables.py:66`<br>`study/dilation_tables.py:308`<br>`study/dilation_tables.py:551` |
| `dram-static-terms-nonnegative` | `ConfigError` | a negative DRAM background or refresh term | `config.py:450` |
| `ert-arm-cache-cold` | `SystemExit` | an ERT arm without its own mapping | `study/ert_view.py:275` |
| `ert-arm-capacity-wrong` | `SystemExit` | a quantisation arm whose Effective size is not 8/q the reference's | `study/ert_view.py:498` |
| `ert-arm-fails-guards` | `SystemExit` | an arm's own mapping failing the arm's own guards | `study/ert_view.py:462` |
| `ert-arm-one-arch` | `ConfigError` | ECC_RECON_ERT_ARM with more than one architecture | `config.py:630` |
| `ert-arm-shapes-differ` | `SystemExit` | an ERT arm and the reference mapped on DIFFERENT layer shapes | `study/ert_view.py:284` |
| `ert-probe-failed` | `SystemExit` | the ERT probe failing to run | `toolchain/ert_probe.py:142` |
| `ert-probe-fingerprint-moved` | `SystemExit` | a sidecar fingerprint that is not the current architecture | `toolchain/ert_probe.py:205` |
| `ert-probe-level-no-weights` | `SystemExit` | a probe level carrying no Weights in the reference stats | `toolchain/ert_probe.py:221` |
| `ert-probe-no-storage-stage` | `SystemExit` | a design with no storage stage on its weight path | `toolchain/ert_probe.py:213` |
| `ert-probe-reference-incomplete` | `SystemExit` | a reference cache entry the probe cannot read | `toolchain/ert_probe.py:200` |
| `ert-row-missing` | `SystemExit` | a generated ERT with no row for a level the study prices | `toolchain/ert.py:382` |
| `ert-split-does-not-reconcile` | `SystemExit` | an access toll the stats and the evaluator disagree about | `study/ert_view.py:531` |
| `ert-split-left-residue` | `SystemExit` | an access toll moved out of a category that did not empty | `study/ert_view.py:548` |
| `ert-toll-mismatch` | `SystemExit` | an evaluator-charged toll that is not the stats-side amount | `study/placement_study.py:299` |
| `figure-draws-the-total` | `SystemExit` | a figure whose bars do not add up to the result's total | `report/recon_view.py:72` |
| `glb-width-mult-ge-1` | `ConfigError` | a weight GLB word narrower than the scratchpad's | `arch/design.py:277` |
| `latency-no-cycles` | `ConfigError` | ECC_METRICS=latency on a raw record carrying no cycle count | `study/metrics.py:102` |
| `latency-not-modelled` | `ConfigError` | ECC_METRICS=latency on a record ECC_LATENCY_MODEL=1 has not re-timed | `study/metrics.py:113`<br>`study/metrics.py:125` |
| `layers-not-in-model` | `SystemExit` | ECC_LAYERS naming layers the model does not have | `arch/workloads.py:157` |
| `layers-per-model-one-model` | `ConfigError` | ECC_LAYERS' per-model form on a run that evaluates several models | `study/common.py:84` |
| `level-carries-no-weights` | `SystemExit` | a level named as a weight site that carries no Weights | `study/ert_view.py:341` |
| `metrics-empty` | `ConfigError` | ECC_METRICS empty -- a figure with no rows | `config.py:216` |
| `missing-path` | `SystemExit` | a required input file that is not there | `paths.py:287` |
| `narrow-once` | `ValueError` | the mapper AND the evaluator both narrowing (it squares the saving) | `study/narrowing.py:130` |
| `need-k-lt-n` | `ConfigError` | a BCH code with K >= N | `config.py:360` |
| `need-weak-k-lt-n` | `ConfigError` | a weak code with WEAK_K >= WEAK_N | `config.py:538` |
| `negative-price` | `ConfigError` | a NEGATIVE per-bit or per-MAC price | `config.py:427` |
| | | ↳ the `>= 0` half of `zero-price`: zero is an ablation, negative is not a price. | |
| `no-arch-yamls` | `SystemExit` | an audit with no architecture YAML to read | `study/diagnose.py:61` |
| `no-boundary-mapped` | `SystemExit` | ECC_RECON_ERT_AWARE=1 with not one boundary mapped | `study/placement_study.py:197` |
| `no-cnn-transformer-mix` | `ConfigError` | one sweep mixing CNNs and transformers (two workload files) | `config.py:496` |
| `no-dilatable-level` | `SystemExit` | Task 4 on a design with no weight level a dilation can touch | `study/dilated_view.py:108` |
| `no-weight-path` | `SystemExit` | a placement study on a design that declares no weight path | `report/recon_view.py:511` |
| `noc-file-missing` | `SystemExit` | the interconnect coefficients file being absent | `arch/load.py:84` |
| `noc-no-entry` | `SystemExit` | a design with no interconnect entry (it would map with free wires) | `arch/load.py:96` |
| `noc-share-low-gt-high` | `ConfigError` | a published share band with low above high | `arch/design.py:331` |
| `noc-share-needs-band` | `ConfigError` | `noc_published_share:` without `low:` and `high:` | `arch/design.py:324` |
| `noc-share-needs-citation` | `ConfigError` | a claim about a PUBLISHED design with no citation | `arch/design.py:327` |
| `nothing-evaluated` | `SystemExit` | an arm that produced no result at all | `study/embedded.py:344`<br>`study/baseline.py:303 *(frozen)*` |
| `pair-geometry` | `ValueError` | the two arms declaring a different DEPTH (real silicon one arm lacks) | `arch/patch.py:1307` |
| | | ↳ THE ONE GUARD WITH AN OVERRIDE OUTSIDE `ECC_ALLOW`: `ECC_DISABLE_ASSERT_PAIR_GEOMETRY=1` (env.sh section 4) predates the tiers and is left exactly as it was. | |
| `panels-need-panel-models` | `ConfigError` | ECC_EXPERIMENT=panels with no ECC_PANEL_MODELS | `config.py:342` |
| `panels-nothing-to-plot` | `SystemExit` | a panel figure with no group in any panel | `report/panels.py:368` |
| `placement-duplicate-key` | `ConfigError` | two placements sharing one key | `arch/design.py:403` |
| `placement-missing-field` | `ConfigError` | a placement missing a required field | `arch/design.py:400` |
| `placement-prefix` | `ConfigError` | `reduced:` naming a stage the weight path does not declare | `arch/design.py:408` |
| `placement-site-counter-unknown` | `ConfigError` | a placement with no known site counter | `arch/design.py:416` |
| `placement-site-stage-unknown` | `ConfigError` | a placement whose `site_stage:` is not a stage of this design | `arch/design.py:412` |
| `placements-empty` | `ConfigError` | a placements.yaml that declares no boundary | `arch/design.py:389` |
| `placements-need-weight-path` | `ConfigError` | boundaries declared without the weight path they cut through | `arch/placements.py:121` |
| `recon-area-missing` | `ConfigError` | ECC_METRICS=area with no archs/_shared/recon_area.yaml scraped yet | `physics/recon_dc.py:207` |
| `recon-area-unmeasured` | `ConfigError` | ECC_METRICS=area at a code Design Compiler has not synthesized | `physics/recon_dc.py:238` |
| | | ↳ the area metric has NO fallback constant, unlike the energy: an invented engine area is an invented silicon number. | |
| `recon-needs-an-arch` | `ConfigError` | the placement study with no architecture | `config.py:459` |
| `recon-nothing-collected` | `SystemExit` | a panelled placement study with a panel that collected nothing | `report/recon_view.py:546` |
| `recon-one-model` | `ConfigError` | the placement study on more than one model | `config.py:476` |
| `stacks-refused` | `SystemExit` | a figure with no stacks to draw | `study/common.py:226` |
| `standard-file-missing` | `SystemExit` | the apples-to-apples contract file being absent | `arch/load.py:391` |
| `sweep-archs-empty` | `ConfigError` | ECC_SWEEP=arch with no architectures | `config.py:307` |
| `sweep-has-no-figure` | `ConfigError` | a sweep or panel FIGURE on an axis that holds all three lists (fix, area) | `config.py:297` |
| `sweep-k-lt-n` | `ConfigError` | an ECC_SWEEP_KS entry that is not below N | `config.py:355` |
| `sweep-ks-empty` | `ConfigError` | ECC_SWEEP=bch with no codes | `config.py:350` |
| `sweep-models-empty` | `ConfigError` | ECC_SWEEP=model with no models | `config.py:319` |
| `sweep-nothing-to-plot` | `SystemExit` | a sweep figure with no group | `report/sweep.py:361` |
| `task4-cache-cold` | `SystemExit` | Task 4 without the reconstruction arm's OWN mapping | `study/dilated_view.py:130` |
| `task4-capacity-not-dilated` | `SystemExit` | a re-planned mapping whose weight capacity is not N/K the reference's | `study/dilated_view.py:196` |
| `task4-shapes-differ` | `SystemExit` | Task 4's two arms mapped on DIFFERENT layer shapes | `study/dilated_view.py:149` |
| `timeloop-mapper-not-on-path` | `SystemExit` | timeloop-mapper not being on PATH | `toolchain/inputs.py:193` |
| `timeloop-model-not-on-path` | `SystemExit` | timeloop-model not being on PATH | `toolchain/ert_probe.py:632` |
| `unknown-ert-arm` | `ConfigError` | ECC_RECON_ERT_ARM naming no arm of this design | `config.py:637` |
| `victory-ge-1` | `ConfigError` | ECC_VICTORY below 1 | `config.py:603` |
| `weight-bits-positive` | `ConfigError` | ECC_WEIGHT_BITS <= 0 | `config.py:364` |
| `weight-path-drifted` | `SystemExit` | a weight path and a placement list that have drifted apart | `report/recon_view.py:521` |
| `weight-path-duplicate-stage` | `ConfigError` | two weight-path stages sharing one key | `arch/design.py:365` |
| `weight-path-empty` | `ConfigError` | a weight_path.yaml with no stages | `arch/design.py:355` |
| `weight-path-missing-field` | `ConfigError` | a weight-path stage missing a required field | `arch/design.py:362` |
| `weight-path-stage-no-prefixes` | `ConfigError` | a stage that matches no Timeloop level, so it can never be measured | `arch/design.py:372` |
| `weight-path-starts-at-dram` | `ConfigError` | a weight path that does not begin at the DRAM | `arch/design.py:379` |
| `weight-path-unknown-kind` | `ConfigError` | a weight-path stage of no known kind | `arch/design.py:369` |
| `widths-not-a-multiple` | `ConfigError` | a declared word width its own datawidth q does not divide (timeloop-mapper aborts on it) | `arch/design.py:278` |
| `workload-groups-indivisible` | `SystemExit` | a workload layer whose channels do not divide by its groups | `arch/workloads.py:83` |
| `workload-no-models` | `SystemExit` | none of the requested models being in the workload file | `arch/workloads.py:126` |

## Tier 3 — COUPLING

| id | raises | refuses | where |
|---|---|---|---|
| `panels-needs-arch-or-model-sweep` | `ConfigError` | a panel layout that would vary two axes at once | `config.py:336` |
| `recon-no-split-read-write` | `ConfigError` | ECC_SPLIT_READ_WRITE=1 inside the placement study | `config.py:482` |
| `rerun-needs-the-mapper` | `ConfigError` | ECC_RERUN_OPTIMISER=1 together with a knob that never invokes Timeloop | `config.py:532` |

## Tier 4 — DERIVED

| id | raises | refuses | where |
|---|---|---|---|
| `zero-price` | `ConfigError` | a ZERO per-bit or per-MAC price | `config.py:433` |
| | | ↳ THE ABLATION: `what if this term were free` is the upper bound on how much it was worth. `> 0` was the wrong rule for a PRICE (ProjectRestructure Appendix B); the invariant is `>= 0` and `negative-price` holds the other half. | |

## Tests of structure — `tests/contract/`

Not guards: these run under `pytest`, not on every job. They hold the
*shape* of the project still — the layer rule, the settings contract,
this file's freshness, and every script outside the package that imports it.

| file | test | holds |
|---|---|---|
| `tests/contract/test_designs_are_data.py` | `test_there_are_designs_at_all` | A tree with no `design.yaml` would make every test below vacuous. |
| `tests/contract/test_designs_are_data.py` | `test_every_design_loads_and_validates` | `design.yaml` parses, names itself, and carries a label. |
| `tests/contract/test_designs_are_data.py` | `test_the_two_halves_are_declared_together` | A design declares BOTH `weight_path.yaml` and `placements.yaml`, or neither. |
| `tests/contract/test_designs_are_data.py` | `test_the_facts_env_sh_used_to_hold_are_declared_here` | EnvReorganisation phase 1: the clock, the leakage densities and THE |
| `tests/contract/test_designs_are_data.py` | `test_a_bad_declared_fact_is_refused_by_name` | The schema is a REFUSAL, not a warning -- for the new fields too. |
| `tests/contract/test_designs_are_data.py` | `test_a_declared_weight_path_is_well_formed` | Stages: unique keys, known kinds, a DRAM stage first, prefixes present. |
| `tests/contract/test_designs_are_data.py` | `test_declared_placements_name_stages_that_exist` | Every `reduced` entry and every `site_stage` is on this design's path. |
| `tests/contract/test_designs_are_data.py` | `test_a_broken_design_file_is_refused_by_name` | The schema is a REFUSAL, not a warning -- and it names the file. |
| `tests/contract/test_designs_are_data.py` | `test_a_stage_of_an_unknown_kind_is_refused` |  |
| `tests/contract/test_designs_are_data.py` | `test_no_production_module_branches_on_a_design_name` | Section 5.1.1 finding 3, held at ZERO. |
| `tests/contract/test_designs_are_data.py` | `test_the_registry_is_the_directory` | `KNOWN_ARCHS` is what `archs/` declares -- no list in Python repeats it. |
| `tests/contract/test_designs_are_data.py` | `test_the_scaffold_writes_a_design_that_is_refused_until_it_is_written` | `make arch NEW=x` must produce something LOUD, not something plausible. |
| `tests/contract/test_guards.py` | `test_every_registered_guard_is_called_somewhere` | A guard nobody raises is folklore. Delete it from the registry instead. |
| `tests/contract/test_guards.py` | `test_every_called_guard_is_registered` | The other end: a `refusal("typo", ...)` must fail here, not at 3am. |
| `tests/contract/test_guards.py` | `test_every_refusal_site_names_a_guard` | No `raise ConfigError(...)` may survive without an id, except the five listed. |
| `tests/contract/test_guards.py` | `test_every_exemption_is_still_real` | A file that no longer has a bare raise must leave `NOT_A_GUARD`. |
| `tests/contract/test_guards.py` | `test_every_exemption_says_why` |  |
| `tests/contract/test_guards.py` | `test_the_spelling_at_the_site_matches_the_tier` | `refusal()` is tiers 1-2, `refuse()` is tiers 3-4 -- so READING a site tells |
| `tests/contract/test_guards.py` | `test_refusal_refuses_to_be_used_at_an_overridable_tier` | The runtime half of the rule above. A mutation must not pass silently. |
| `tests/contract/test_guards.py` | `test_refuse_refuses_to_be_used_at_a_fatal_tier` |  |
| `tests/contract/test_guards.py` | `test_an_unregistered_id_is_refused_at_the_call` |  |
| `tests/contract/test_guards.py` | `test_only_tiers_3_and_4_are_overridable` | Tiers 1 and 2 may never be lifted: a value that cannot be parsed has no |
| `tests/contract/test_guards.py` | `test_every_guard_says_what_it_refuses` | The `refuses` column is what makes GUARDS.md readable. It may not be a stub. |
| `tests/contract/test_guards.py` | `test_the_frozen_sites_are_still_where_the_registry_says` | `physics/parity.py` and `study/baseline.py` may not be edited (CLAUDE.md), so |
| `tests/contract/test_guards.py` | `test_ecc_allow_names_guards_and_never_switches_them_all_off` | Section 10: a blanket off would be set once and forgotten. Only ids count. |
| `tests/contract/test_guards.py` | `test_an_override_is_recorded_and_an_unlisted_one_is_not` |  |
| `tests/contract/test_guards.py` | `test_a_zero_price_is_an_ablation_and_not_an_impossibility` | ProjectRestructure Appendix B. These three used `> 0` where the invariant is |
| `tests/contract/test_guards.py` | `test_a_negative_price_is_still_refused_and_ecc_allow_cannot_lift_it` | The other half of Appendix B: a negative price really IS impossible, so it |
| `tests/contract/test_guards.py` | `test_the_ablation_travels_onto_the_figure_and_the_manifest` | Section 6.4 rule 2. You cannot publish an ablation without it saying so. |
| `tests/contract/test_guards.py` | `test_a_clean_run_records_no_override_at_all` | The default must be byte-identical to what it was before phase 6: no key on |
| `tests/contract/test_guards.py` | `test_guards_md_is_not_stale` | Section 6.5: generated, never hand-written. `make guards` regenerates it. |
| `tests/contract/test_guards.py` | `test_guards_md_lists_every_guard_and_no_others` |  |
| `tests/contract/test_layer_rule.py` | `test_every_module_has_a_layer` | A new module must be given a layer, not left to fall through the rule. |
| `tests/contract/test_layer_rule.py` | `test_no_module_imports_from_a_higher_layer` | The rule itself. Every upward edge must be declared, with a reason. |
| `tests/contract/test_layer_rule.py` | `test_every_declared_exception_is_still_real` | A fixed violation must be DELETED from the list, not left as folklore. |
| `tests/contract/test_layer_rule.py` | `test_the_layer_rule_holds_below_the_drivers` | L0-L4 -- settings, physics, arch, config, toolchain -- may not reach up. |
| `tests/contract/test_layer_rule.py` | `test_the_exception_list_is_documented` | Every declared exception says WHY and names the phase that removes it. |
| `tests/contract/test_out_of_package.py` | `test_every_embedded_block_parses` | A block the audit cannot read is a block the audit does not cover. |
| `tests/contract/test_out_of_package.py` | `test_there_is_something_to_audit` | The globs must actually find the scripts -- an empty audit passes vacuously. |
| `tests/contract/test_out_of_package.py` | `test_every_out_of_package_import_resolves` | `from eccenergy... import X` in a script must name something that exists. |
| `tests/contract/test_out_of_package.py` | `test_every_attribute_used_on_an_imported_module_exists` | The half that catches a SPLIT, not a move. |
| `tests/contract/test_out_of_package.py` | `test_every_dash_m_target_is_a_real_module` | `python3 -m eccenergy.report.dilation_view` must name a runnable module. |
| `tests/contract/test_settings.py` | `test_the_hashed_set_is_what_it_was_before_the_split` | Assembled from six declarations, it must still be the same SET. |
| `tests/contract/test_settings.py` | `test_every_hashed_field_is_declared_by_the_group_that_owns_it` | A group may only mark its OWN fields, or the union means nothing. |
| `tests/contract/test_settings.py` | `test_energy_model_rev_is_hashed_only_when_it_is_set` | EMPTY must hash byte-identically to every fingerprint predating the knob. |
| `tests/contract/test_settings.py` | `test_the_record_keeps_its_key_order` | `FIELD_ORDER` is every field of the six groups, and nothing else. |
| `tests/contract/test_settings.py` | `test_every_knob_is_owned_by_exactly_one_group` | Two groups declaring the same knob would make `cfg.x` ambiguous. |
| `tests/contract/test_settings.py` | `test_the_settings_groups_are_frozen` | A resolved configuration that can be edited is not a resolved one. |
| `tests/contract/test_settings.py` | `test_an_unknown_knob_is_refused_rather_than_stored` | `with_()` routes by name, so a typo must fail LOUDLY, not add a field. |
| `tests/contract/test_settings.py` | `test_settings_import_nothing_above_them` | The knobs may not reach for a design, or phase 4's cycle comes back. |
