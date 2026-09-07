"""Tests for the embedded-ECC layout, its DRAM accounting and the Task 2 checks.

    python3 -m eccenergy.tests.test_embedded

Dependency-free and offline like the other suites. The two tests that need a
`Raw` record import pandas and are SKIPPED (reported, not hidden) where it is
missing -- the HiPerGator host python -- and run inside the container:

    bash hpc/tl.sh python3 -m eccenergy.tests.test_embedded
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import traceback

FAILURES = []
SKIPPED = []


def check(name, fn):
    try:
        fn()
    except _Skip as why:
        SKIPPED.append(name)
        print(f"  skip  {name}  ({why})")
    except Exception:
        FAILURES.append(name)
        print(f"  FAIL  {name}")
        traceback.print_exc()
    else:
        print(f"  ok    {name}")


class _Skip(Exception):
    pass


def _cfg(**env):
    base = dict(
        ECC_EXPERIMENT="embedded", ECC_SWEEP="arch",
        ECC_CONST_ARCH="eyeriss_v2_like", ECC_CONST_MODEL="resnet18",
        ECC_CONST_K="51", ECC_LAYERS="layer3.0.downsample.0 layer4.1.conv2",
        ECC_MAPPER_THREADS="8", ECC_VICTORY="100", ECC_FROM_CACHE="1",
    )
    base.update(env)
    for k in list(os.environ):
        if k.startswith("ECC_"):
            del os.environ[k]
    os.environ.update(base)
    from eccenergy.config import load_config
    return load_config()


# ---------------------------------------------------------------- the layout
def test_layout_is_the_bitstream_layout_the_pipeline_uses():
    from eccenergy.embedded import EmbeddedLayout
    lay = EmbeddedLayout(63, 51, 8).validate()
    # 63-bit chunks over an 8-bit stream: fractional weights per codeword, for every K
    assert lay.weights_per_codeword == 63 / 8 == 7.875
    assert EmbeddedLayout(63, 30, 8).weights_per_codeword == 7.875
    assert lay.parity_bits == 12 and lay.codeword_bits == 63
    # gcd(63, 8) = 1 -> the alignment repeats every 8 codewords = 63 weights
    assert lay.period_bits == 504
    assert lay.period_codewords == 8 and lay.period_weights == 63
    assert lay.split_boundaries_per_period == 7
    assert abs(lay.independent_frac - 51 / 63) < 1e-15


def test_parity_overwrites_the_lowest_significance_positions():
    """Mirror of ParityOverwriteByTopWeightsEncode: message = top-k significance."""
    from eccenergy.embedded import EmbeddedLayout
    lay = EmbeddedLayout(63, 51, 8).validate()
    for phase in range(lay.period_codewords):
        sig = lay.local_significances(phase)
        assert len(sig) == 63 and set(sig) <= set(range(8))
        parity = lay.parity_positions(phase)
        assert len(parity) == 12
        message = [i for i in range(63) if i not in set(parity)]
        assert max(sig[i] for i in parity) <= min(sig[i] for i in message)
    prof = lay.parity_significance_profile()
    per_w = prof["parity_bits_per_weight_by_significance"]
    # 8 phases x 12 parity slots = 96 over 63 weights: every LSB (63) + 33 of bit1
    assert per_w["bit0"] == 1.0, per_w
    assert abs(per_w["bit1"] - 33 / 63) < 1e-12, per_w
    assert set(per_w) == {"bit0", "bit1"}, per_w
    assert abs(prof["parity_bits_per_weight_total"] - 96 / 63) < 1e-12
    # the strongest code of the sweep eats bits 0-3 of every weight and part of bit4
    strong = EmbeddedLayout(63, 30, 8).parity_significance_profile()
    pw = strong["parity_bits_per_weight_by_significance"]
    assert all(pw[f"bit{b}"] == 1.0 for b in range(4)), pw
    assert 0 < pw["bit4"] < 1 and "bit5" not in pw, pw


def test_account_by_hand():
    """A worked example small enough to verify without the code.

    13 weights at BCH(63,51) over 8-bit weights, bit-stream layout:
      payload                      = 13 * 8                = 104 bits
      codewords                    = ceil(104 / 63)        =   2
      tail padding (last chunk)    = 2 * 63 - 104          =  22 bits
      embedded parity (inside)     = 2 * 12                =  24 bits
      external parity              =                          0
      interior boundaries          = j = 1 at bit 63: 63 % 8 = 7 -> splits weight 7
      stored                       = 104 + 22              = 126 bits -> 2 DRAM words
    """
    from eccenergy.embedded import EmbeddedLayout, account
    a = account(13, EmbeddedLayout(63, 51, 8).validate(), dram_word_bits=64)
    assert a.payload_bits == 104, a.payload_bits
    assert a.codewords == 2, a.codewords
    assert a.tail_pad_bits == 22, a.tail_pad_bits
    assert a.embedded_parity_bits == 24, a.embedded_parity_bits
    assert a.external_parity_bits == 0
    assert a.split_boundaries == 1 and a.straddling_weights == 1, (a.split_boundaries,
                                                                    a.straddling_weights)
    assert a.stored_bits == 126 and a.stored_dram_words == 2
    assert abs(a.bits_per_weight - 126 / 13) < 1e-12


def test_one_period_exactly():
    from eccenergy.embedded import EmbeddedLayout, account
    lay = EmbeddedLayout(63, 51, 8).validate()
    a = account(63, lay)                      # 504 bits = 8 codewords, no tail
    assert a.codewords == 8 and a.tail_pad_bits == 0
    assert a.split_boundaries == 7 and a.straddling_weights == 7
    b = account(64, lay)                      # 512 bits = 9 codewords; boundary 8 aligned
    assert b.codewords == 9 and b.tail_pad_bits == 55
    assert b.split_boundaries == 7 and b.straddling_weights == 7
    c = account(126, lay)                     # two periods
    assert c.codewords == 16 and c.split_boundaries == 14 and c.straddling_weights == 14


def test_storage_is_the_payload_plus_at_most_one_tail_pad():
    from eccenergy.embedded import EmbeddedLayout, account
    lay = EmbeddedLayout(63, 51, 8).validate()
    for w in (1, 5, 6, 7, 8, 9, 62, 63, 64, 127, 32768, 2359296, 11678912):
        a = account(w, lay)
        assert a.stored_bits == w * 8 + a.tail_pad_bits
        assert 0 <= a.tail_pad_bits < 63, (w, a.tail_pad_bits)
        assert a.external_parity_bits == 0
        assert 8.0 <= a.bits_per_weight < 8.0 + 63 / w


def test_hand_check_matches_closed_form():
    from eccenergy.embedded import EmbeddedLayout, hand_check
    for n, k in ((63, 51), (63, 57), (63, 30), (7, 4)):     # (7,4): n < weight_bits
        lay = EmbeddedLayout(n, k, 8).validate()
        for w in (0, 1, 7, 8, 9, 13, 63, 64, 126, 32768, 2359296, 11678912):
            ok, detail = hand_check(lay, w)
            assert ok, (n, k, w, detail["mismatches"])


def test_hand_check_layers_walks_each_tensor():
    from eccenergy.embedded import EmbeddedLayout, account_layers, hand_check_layers
    lay = EmbeddedLayout(63, 51, 8).validate()
    layers = [32768, 2359296, 13, 7]
    ok, d = hand_check_layers(lay, layers, grouping="layer")
    assert ok and d["layers_checked"] == 4, d
    per_layer = account_layers(layers, lay, grouping="layer")
    assert d["checks"]["codewords"]["walked"] == per_layer.codewords
    assert d["checks"]["tail_pad_bits"]["walked"] == per_layer.tail_pad_bits
    whole = account_layers(layers, lay, grouping="model")
    # each layer pads its own last chunk, so never fewer codewords than one stream
    assert per_layer.codewords >= whole.codewords
    assert per_layer.payload_bits == whole.payload_bits


def test_traffic_is_the_complete_codeword_and_nothing_external():
    from eccenergy.embedded import EmbeddedLayout, account, dram_energy_pj, traffic_account
    lay = EmbeddedLayout(63, 51, 8).validate()
    stored = account(32768, lay)
    tr = traffic_account(131072, stored, lay)          # refetch x4
    assert tr["refetch_factor"] == 4.0
    assert tr["payload_bits_read"] == 131072 * 8       # every bit of every read
    assert tr["codeword_reads"] == stored.codewords * 4
    assert tr["external_bits"] == 0.0 and tr["external_dram_words"] == 0.0
    assert tr["physical_dram_words_estimate"] == math.ceil(131072 * 8 / 64)
    assert tr["overhead_vs_payload_frac"] == 0.0
    e_w, e_ext, per_scalar = dram_energy_pj(64.0 * 131072, 131072)
    assert e_w == 64.0 * 131072 and e_ext == 0.0 and per_scalar == 64.0


def test_experiment_is_registered():
    from eccenergy.config import EXPERIMENTS
    assert "embedded" in EXPERIMENTS
    cfg = _cfg()
    assert cfg.experiment == "embedded" and cfg.from_cache


# -------------------------------------------------- the two arms on one record
def _fake_raw():
    try:
        import pandas as pd
    except ImportError as exc:
        raise _Skip(f"pandas not available on this python: {exc}")
    from eccenergy.energy import Raw
    cats = ["DRAM", "Global buffer", "Local (spads/RF)", "NoC", "Compute",
            "ECC decode", "Reconstruction"]
    # 2,392,064 weights (the dev pair), read x2 from DRAM at 64 pJ each
    reads = 2 * 2392064.0
    e_dram_w = 64.0 * reads
    base = pd.Series({"DRAM": e_dram_w + 1.0e8, "Global buffer": 3.0e8,
                      "Local (spads/RF)": 5.0e8, "NoC": 1.0e8, "Compute": 9.0e8,
                      "ECC decode": 0.0, "Reconstruction": 0.0}).reindex(cats)
    base_w = pd.Series({"DRAM": e_dram_w, "Global buffer": 1.0e8,
                        "Local (spads/RF)": 2.0e8, "NoC": 0.5e8}).reindex(cats, fill_value=0.0)
    base_i = pd.Series({"DRAM": 0.5e8}).reindex(cats, fill_value=0.0)
    per_layer = [
        {"layer": "layer3.0.downsample.0", "status": "ok", "weights": 32768,
         "mapping_id": "aaaa1111"},
        {"layer": "layer4.1.conv2", "status": "ok", "weights": 2359296,
         "mapping_id": "bbbb2222"},
    ]
    return Raw(base, base_w, base_i, e_dram_w, reads, 2, 0, 2392064, per_layer, [])


def test_embedded_arm_removes_exactly_the_external_parity():
    raw = _fake_raw()
    from eccenergy.ecc import embedded_dram, external_parity
    cfg = _cfg()
    e_parity, pdetail = external_parity(cfg, raw)
    e_emb, edetail = embedded_dram(cfg, raw)
    assert e_parity > 0 and e_emb == 0.0
    assert edetail["dram_weight_energy_pJ"] == raw.e_dram_w
    assert edetail["pJ_per_dram_weight_scalar"] == pdetail["pJ_per_dram_weight_scalar"] == 64.0
    assert edetail["hand_check_passed"] and edetail["stored"]["external_parity_bits"] == 0
    # same weights, same reads: the stored footprints differ only by parity+padding
    assert edetail["stored"]["payload_bits"] == pdetail["stored"]["payload_bits"]
    assert edetail["stored"]["stored_bits"] < pdetail["stored"]["stored_bits"]
    assert edetail["traffic"]["refetch_factor"] == pdetail["traffic"]["refetch_factor"] == 2.0
    assert edetail["layout"]["weights_per_codeword"] == 7.875
    assert edetail["comparison_to_baseline_layout"]["baseline_weights_per_codeword"] == 6


def test_task2_checks_pass_on_a_consistent_record_and_catch_a_leak():
    raw = _fake_raw()
    from eccenergy.ecc import embedded_dram, external_parity
    from eccenergy.energy import plot_cats
    from eccenergy.experiments import audit
    from eccenergy.experiments.embedded import PARITY_KEY, task2_checks
    from eccenergy.paths import Results
    from eccenergy.results_store import ResultBuilder, Variant
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _cfg(ECC_RESULTS_DIR=tmp)
        results = Results(cfg).prepare()
        series = raw.base.reindex(plot_cats(cfg), fill_value=0.0)
        e_parity, pdetail = external_parity(cfg, raw)
        e_emb, edetail = embedded_dram(cfg, raw)
        base = audit.components(series); base[PARITY_KEY] = e_parity
        emb = audit.components(series); emb[PARITY_KEY] = e_emb

        def build(emb_components, arm_total, newly, cfg=cfg):
            b = ResultBuilder(cfg, results, "eyeriss_v2_like", "resnet18",
                              experiment="unit_test", fixed_mapping=True)
            b.add(Variant("baseline_external_parity", kind="baseline", status="evaluated",
                          total_energy_pJ=sum(base.values()), energy_by_component_pJ=base,
                          mapping_ids=["aaaa1111", "bbbb2222"]))
            b.add(Variant("embedded_ecc", kind="embedded", status="evaluated",
                          total_energy_pJ=sum(emb_components.values()),
                          energy_by_component_pJ=emb_components,
                          mapping_ids=["aaaa1111", "bbbb2222"]))
            task2_checks(b, cfg, raw, base, emb_components, e_parity, edetail,
                         sweep_arm_total=arm_total, newly_mapped=newly, raw_cache_hit=True)
            return {c["check"]: c["passed"] for c in b.document()["validation"]}

        good = build(emb, sum(emb.values()), 0)
        for name in ("non_dram_components_match_task1_baseline",
                     "dram_difference_is_exactly_the_external_parity",
                     "embedded_reads_complete_codeword", "embedded_layout_hand_check",
                     "embedded_matches_sweep_figure_arm", "evaluation_only_rerun",
                     "fixed_mapping_shared_across_variants"):
            assert good[name] is True, (name, good)
        # a leak into an on-chip category must be caught, and so must a run that
        # generated mappings (only detectable when ECC_FROM_CACHE is off: with it
        # on, Timeloop was never invoked and the check passes by construction)
        leaky = dict(emb); leaky["Compute"] *= 0.9
        bad = build(leaky, None, 3, cfg=_cfg(ECC_RESULTS_DIR=tmp, ECC_FROM_CACHE="0"))
        assert bad["non_dram_components_match_task1_baseline"] is False
        assert bad["evaluation_only_rerun"] is False
        assert "embedded_matches_sweep_figure_arm" not in bad   # skipped when None


def main():
    print("eccenergy embedded-ECC (Task 2) tests")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    print()
    if SKIPPED:
        print(f"{len(SKIPPED)} skipped (run inside the container): {', '.join(SKIPPED)}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
