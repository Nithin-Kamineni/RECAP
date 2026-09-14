"""Tests for the parity accounting and the result writer.

    python3 -m eccenergy.tests.test_results_store

Deliberately dependency-free (no pytest) and offline: it never invokes the
mapper, never needs the container, and writes into a temporary results
directory, so it can run anywhere and in CI. `04_results_storage_spec.txt` asks
for "a small automated test proving that a JSON file containing baseline,
embedded, and multiple reconstruction variants is written to the correct
namespace and can be loaded again" -- that is `test_roundtrip`, and the rest
guard the invariants that make such a file worth trusting.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback


FAILURES = []


def check(name, fn):
    try:
        fn()
    except Exception:
        FAILURES.append(name)
        print(f"  FAIL  {name}")
        traceback.print_exc()
    else:
        print(f"  ok    {name}")


def _cfg(**env):
    """A Config with a temporary results directory and the dev layer pair."""
    base = dict(
        ECC_EXPERIMENT="baseline",
        ECC_SWEEP="arch",
        ECC_CONST_ARCH="eyeriss_v2_like",
        ECC_CONST_MODEL="resnet18",
        ECC_CONST_K="51",
        ECC_LAYERS="layer3.0.downsample.0 layer4.1.conv2",
        ECC_MAPPER_THREADS="8",
        ECC_VICTORY="100",
    )
    base.update(env)
    for k in list(os.environ):
        if k.startswith("ECC_"):
            del os.environ[k]
    os.environ.update(base)
    from eccenergy.config import load_config
    return load_config()


# --------------------------------------------------------------- parity maths
def test_codeword_packing_is_whole_weights():
    from eccenergy.physics.parity import CodeGeometry
    g = CodeGeometry(63, 51, 8).validate()
    # 51 // 8 == 6, NOT 6.375. Three message bits are padding.
    assert g.weights_per_codeword == 6, g.weights_per_codeword
    assert g.message_used_bits == 48
    assert g.message_pad_bits == 3
    assert g.parity_bits == 12
    # the whole point: the real overhead is above the ideal code rate
    assert abs(g.parity_overhead_frac - 12 / 48) < 1e-12
    assert g.parity_overhead_frac > g.parity_overhead_ideal_frac


def test_code_that_cannot_hold_a_weight_is_rejected():
    from eccenergy.physics.parity import CodeGeometry
    try:
        CodeGeometry(63, 7, 8).validate()
    except ValueError as exc:
        assert "cannot hold even one" in str(exc), exc
    else:
        raise AssertionError("a 7-bit message field must not accept 8-bit weights")


def test_hand_check_matches_closed_form():
    """The check Task 1 asks for, over sizes that exercise every rounding case."""
    from eccenergy.physics.parity import CodeGeometry, hand_check
    g = CodeGeometry(63, 51, 8).validate()
    for w in (0, 1, 5, 6, 7, 12, 13, 32768, 2359296, 11678912):
        ok, detail = hand_check(g, w)
        assert ok, (w, detail["mismatches"])


def test_parity_accounting_by_hand():
    """A worked example small enough to verify without the code.

    13 weights at BCH(63,51) over 8-bit weights:
      6 whole weights per codeword  -> ceil(13/6) = 3 codewords
      payload                        = 13 * 8              = 104 bits
      message padding                = 3 * 3               =   9 bits
      tail padding                   = 3*48 - 104          =  40 bits
      parity                         = 3 * 12              =  36 bits
      parity in whole 64b DRAM words = ceil(36/64)         =   1 word
    """
    from eccenergy.physics.parity import CodeGeometry, account
    a = account(13, CodeGeometry(63, 51, 8).validate(), dram_word_bits=64)
    assert a.codewords == 3, a.codewords
    assert a.payload_bits == 104, a.payload_bits
    assert a.message_pad_bits == 9, a.message_pad_bits
    assert a.tail_pad_bits == 40, a.tail_pad_bits
    assert a.parity_bits == 36, a.parity_bits
    assert a.parity_dram_words == 1, a.parity_dram_words
    assert a.parity_dram_scalars == 8, a.parity_dram_scalars


def test_layer_grouping_costs_at_least_as_much_as_model_grouping():
    """Per-layer codewords pay per-layer tail padding, so never fewer codewords."""
    from eccenergy.physics.parity import CodeGeometry, account_layers
    g = CodeGeometry(63, 51, 8).validate()
    layers = [32768, 2359296, 13, 7]
    per_layer = account_layers(layers, g, grouping="layer")
    whole = account_layers(layers, g, grouping="model")
    assert per_layer.codewords >= whole.codewords
    assert per_layer.parity_bits >= whole.parity_bits
    assert per_layer.payload_bits == whole.payload_bits


def test_padding_charge_is_the_larger_number():
    from eccenergy.physics.parity import CodeGeometry, account, traffic_account
    g = CodeGeometry(63, 51, 8).validate()
    stored = account(32768, g)
    with_pad = traffic_account(131072, stored, g, charge_padding=True)
    without = traffic_account(131072, stored, g, charge_padding=False)
    assert with_pad["external_bits"] > without["external_bits"]
    assert abs(with_pad["overhead_vs_payload_frac"] - 0.3125) < 1e-9
    assert abs(without["overhead_vs_payload_frac"] - 0.25) < 1e-9


# ------------------------------------------------------------- savings maths
def test_savings_definition():
    from eccenergy.toolchain.results_store import savings_percent
    assert savings_percent(100.0, 75.0) == 25.0
    assert savings_percent(100.0, 125.0) == -25.0   # negative savings allowed
    assert savings_percent(None, 75.0) is None      # unknown, not zero
    assert savings_percent(100.0, None) is None
    assert savings_percent(0.0, 10.0) is None


# ------------------------------------------------------- the writer contract
def _builder(cfg, results, arch="eyeriss_v2_like", model="resnet18"):
    from eccenergy.toolchain.results_store import ResultBuilder
    return ResultBuilder(cfg, results, arch, model,
                         experiment="unit_test", fixed_mapping=True)


def test_unavailable_variant_must_not_carry_a_number():
    from eccenergy.toolchain.results_store import ResultError, Variant
    try:
        Variant("x", kind="reconstruction", status="not_implemented",
                total_energy_pJ=1.0, unavailable_reason="because")
    except ResultError as exc:
        assert "must not carry a number" in str(exc), exc
    else:
        raise AssertionError("an unimplemented variant with an energy was accepted")


def test_unavailable_variant_must_explain_itself():
    from eccenergy.toolchain.results_store import ResultError, Variant
    try:
        Variant("x", kind="reconstruction", status="unsupported")
    except ResultError as exc:
        assert "unavailable_reason" in str(exc), exc
    else:
        raise AssertionError("an unavailable variant with no reason was accepted")


def test_evaluated_variant_must_carry_a_number():
    from eccenergy.toolchain.results_store import ResultError, Variant
    try:
        Variant("x", kind="baseline", status="evaluated")
    except ResultError as exc:
        assert "no total_energy_pJ" in str(exc), exc
    else:
        raise AssertionError("an evaluated variant with no energy was accepted")


def test_fixed_mapping_violation_is_detected():
    """Two variants mapped differently must not pass as a fixed-mapping result."""
    from eccenergy.paths import Results
    from eccenergy.toolchain.results_store import Variant
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(ECC_RESULTS_DIR=tmp)
        b = _builder(cfg, Results(cfg).prepare())
        b.add(Variant("baseline_external_parity", kind="baseline",
                      status="evaluated", total_energy_pJ=10.0,
                      energy_by_component_pJ={"DRAM": 10.0},
                      mapping_ids=["aaa"]))
        b.add(Variant("recon_pe_spad_input", kind="reconstruction",
                      status="evaluated", total_energy_pJ=9.0,
                      energy_by_component_pJ={"DRAM": 9.0},
                      mapping_ids=["bbb"]))
        doc = b.document()
        checks = {c["check"]: c["passed"] for c in doc["validation"]}
        assert checks["fixed_mapping_shared_across_variants"] is False, checks


def test_roundtrip():
    """THE TEST THE STORAGE SPEC ASKS FOR.

    One file holding the conventional baseline, embedded ECC and several
    reconstruction placements; written to the deterministic namespace; loaded
    back; and its savings recomputed to the definition in the spec.
    """
    from eccenergy.paths import Results
    from eccenergy.toolchain.results_store import Variant, load, load_latest, savings_percent
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(ECC_RESULTS_DIR=tmp)
        results = Results(cfg).prepare()
        b = _builder(cfg, results)
        b.add(Variant("baseline_external_parity", kind="baseline",
                      status="evaluated", total_energy_pJ=1000.0,
                      energy_by_component_pJ={"DRAM": 600.0, "Compute": 400.0},
                      mapping_ids=["m1", "m2"]))
        b.add(Variant("embedded_ecc", kind="embedded", status="evaluated",
                      total_energy_pJ=800.0,
                      energy_by_component_pJ={"DRAM": 400.0, "Compute": 400.0},
                      mapping_ids=["m1", "m2"]))
        b.add(Variant("recon_pe_spad_input", kind="reconstruction",
                      status="evaluated", total_energy_pJ=760.0,
                      energy_by_component_pJ={"DRAM": 400.0, "Compute": 360.0},
                      mapping_ids=["m1", "m2"]))
        b.add(Variant("recon_pe_spad_output", kind="reconstruction",
                      status="evaluated", total_energy_pJ=700.0,
                      energy_by_component_pJ={"DRAM": 400.0, "Compute": 300.0},
                      mapping_ids=["m1", "m2"]))
        b.add(Variant("recon_source_noc_ingress", kind="reconstruction",
                      status="unsupported",
                      unavailable_reason="this architecture has no weight NoC level",
                      mapping_ids=["m1", "m2"]))
        b.add(Variant("recon_mac_input", kind="reconstruction",
                      status="not_implemented",
                      unavailable_reason="Task 3 has not been implemented"))
        path = b.write()

        # ---- the namespace is the one the spec specifies ------------------
        rel = path.relative_to(results.base).as_posix()
        assert rel.startswith("evaluation/Pre/eyeriss_v2_like/resnet18/bch63_51/"
                              "w8a8/layers2__layer3_0_downsample_0__layer4_1_conv2/"
                              "map-energy-vic100l-hybrid-seednone-ssnone-perm16/"), rel
        assert path.name.endswith(".json") and "__" in path.name, path.name

        # ---- it loads back, and says what it is ---------------------------
        doc = load(path)
        assert doc["schema_version"]
        assert doc["experiment"]["phase"] == "Pre"
        assert doc["experiment"]["mapping_mode"] == "fixed-mapping"
        assert doc["identity"]["layer_scope"] == "two-layer"
        assert doc["identity"]["layers"] == ["layer3.0.downsample.0",
                                             "layer4.1.conv2"]
        assert doc["identity"]["bch"] == {"n": 63, "k": 51, "t": 2,
                                          "label": "BCH(63,51)"}
        assert doc["identity"]["precisions_bits"]["weight"] == 8
        assert doc["identity"]["precisions_bits"]["activation"] == 8
        # paper-native: Eyeriss v2 accumulates at 20b
        assert doc["identity"]["precisions_bits"]["accumulator"] == 20

        # ---- every variant is present, including the unavailable ones -----
        names = {v["name"] for v in doc["variants"]}
        assert {"baseline_external_parity", "embedded_ecc", "recon_pe_spad_input",
                "recon_pe_spad_output", "recon_source_noc_ingress",
                "recon_mac_input"} == names, names
        for v in doc["variants"]:
            if v["status"] != "evaluated":
                assert v["total_energy_pJ"] is None, v
                assert v["unavailable_reason"], v

        # ---- savings follow the spec's definition, both references --------
        row = {r["variant"]: r for r in doc["summary"]}
        assert row["embedded_ecc"]["savings_vs_conventional_ecc_percent"] == \
            savings_percent(1000.0, 800.0) == 20.0
        assert abs(row["recon_pe_spad_output"]["savings_vs_embedded_only_percent"]
                   - savings_percent(800.0, 700.0)) < 1e-12
        assert row["recon_source_noc_ingress"][
            "savings_vs_conventional_ecc_percent"] is None

        # ---- the best placement is the cheapest EVALUATED one -------------
        assert doc["best_reconstruction_placement"]["variant"] == "recon_pe_spad_output"
        assert doc["best_reconstruction_placement"]["evaluated_candidates"] == 2

        # ---- the fixed-mapping claim was verified, not asserted ------------
        checks = {c["check"]: c["passed"] for c in doc["validation"]}
        assert checks["fixed_mapping_shared_across_variants"] is True, checks

        # ---- latest.json points at it, and load_latest finds it -----------
        assert (path.parent / "latest.json").exists()
        again = load_latest(results, "eyeriss_v2_like", "resnet18")
        assert again["experiment"]["run_id"] == doc["experiment"]["run_id"]


def test_existing_result_is_not_silently_overwritten():
    from eccenergy.paths import Results
    from eccenergy.toolchain.results_store import ResultError, Variant
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(ECC_RESULTS_DIR=tmp)
        results = Results(cfg).prepare()

        def build():
            b = _builder(cfg, results)
            b.add(Variant("baseline_external_parity", kind="baseline",
                          status="evaluated", total_energy_pJ=1.0,
                          energy_by_component_pJ={"DRAM": 1.0}))
            return b

        first = build()
        path = first.write()
        second = build()
        second.started = first.started       # same run id, same file
        try:
            second.write()
        except ResultError as exc:
            assert "never silently overwritten" in str(exc), exc
        else:
            raise AssertionError("an existing result was overwritten without asking")
        # ...and explicitly allowing it works
        assert second.write(overwrite=True) == path


def test_missing_reference_is_refused():
    from eccenergy.paths import Results
    from eccenergy.toolchain.results_store import ResultError, Variant
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(ECC_RESULTS_DIR=tmp)
        b = _builder(cfg, Results(cfg).prepare())
        b.add(Variant("embedded_ecc", kind="embedded", status="evaluated",
                      total_energy_pJ=1.0, energy_by_component_pJ={"DRAM": 1.0}))
        try:
            b.document()
        except ResultError as exc:
            assert "reference variant" in str(exc), exc
        else:
            raise AssertionError("a result with no reference variant was accepted")


def test_namespace_separates_dev_from_full_and_forced_precision():
    """A development result can never land where a full-model result lives."""
    from eccenergy.paths import Results
    with tempfile.TemporaryDirectory() as tmp:
        dev = Results(_cfg(ECC_RESULTS_DIR=tmp)).prepare()
        dev_dir = dev.evaluation_dir("eyeriss_v2_like", "resnet18", "Pre")
        full = Results(_cfg(ECC_RESULTS_DIR=tmp, ECC_LAYERS="")).prepare()
        full_dir = full.evaluation_dir("eyeriss_v2_like", "resnet18", "Pre")
        assert dev_dir != full_dir
        assert "layers-full" in full_dir.as_posix()
        assert "layers2__" in dev_dir.as_posix()

        forced = Results(_cfg(ECC_RESULTS_DIR=tmp, ECC_ACC_BITS="24")).prepare()
        forced_dir = forced.evaluation_dir("eyeriss_v2_like", "resnet18", "Pre")
        assert forced_dir != dev_dir
        assert "w8a8acc24" in forced_dir.as_posix()


def main():
    print("eccenergy result-store and parity tests")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
