"""
Correctness tests for the SECDED input-embedding codec:
  1. Cross-check the vectorized encoder against the scalar reference
     (SECDEDInputEmbed) chunk-by-chunk on random data, across all 8 phases.
  2. Round-trip encode -> decode on error-free codewords (must recover the
     original message bits exactly).
  3. Single-bit-flip injection -> decode must correct it exactly.
  4. Double-bit-flip injection -> decode must report double_error_detected
     (never silently "correct" to the wrong value).

Run inside the project's Singularity image (needs numpy + galois):
  singularity exec --bind /blue <SIF> python3 3-Testing/test_secded.py
"""
import os, sys, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.convert_to_binary import convert_to_binary
from utils.messageSliceBasedOnChunkSize import messageSliceBasedOnChunkSize
from implementations.SECDEDInputEmbed import (
    SECDEDInputEmbed, SECDEDDecode, N_MOTHER, K_MSG,
)
from implementations.secded_vectorized import secded_embed_uint8
from implementations.secded_torch import secded_embed_uint8_torch
import torch


def test_vectorized_matches_scalar(n_values=2000, seed=0):
    rng = random.Random(seed)
    values = [rng.randint(0, 255) for _ in range(n_values)]
    arr_u8 = np.array(values, dtype=np.uint8)

    # --- scalar reference path ---
    rows = convert_to_binary(values, bit_size=8)
    chunks = messageSliceBasedOnChunkSize(rows, chunk_size=N_MOTHER)
    scalar_mutated_bits_stream = []
    for ch in chunks:
        out = SECDEDInputEmbed(ch, message_parity_size=N_MOTHER, message_size=K_MSG)
        scalar_mutated_bits_stream.extend(out["sliced_message_bits"])
    total_bits = n_values * 8
    scalar_mutated_bits_stream = scalar_mutated_bits_stream[:total_bits]
    scalar_packed = np.packbits(np.array(scalar_mutated_bits_stream, dtype=np.uint8))

    # --- vectorized path ---
    vec_mutated, distortion = secded_embed_uint8(arr_u8)

    assert np.array_equal(scalar_packed, vec_mutated), (
        "Vectorized SECDED encoder disagrees with the scalar reference!"
    )
    print(f"[OK] vectorized == scalar on {n_values} random uint8 values "
          f"(mean abs distortion = {distortion:.4f})")


def test_torch_matches_vectorized(seed=4):
    """GPU-native encoder (secded_torch) must agree bit-exactly with the
    numpy vectorized encoder across a range of shapes, including cases
    with and without the tail-chunk padding fixup."""
    rng = np.random.default_rng(seed)
    shapes = [(2000,), (4, 3, 5, 5), (3, 4, 4), (1,), (63,), (64,), (505,), (128, 3, 5, 5)]
    for shape in shapes:
        arr = rng.integers(0, 256, size=shape, dtype=np.uint8)
        ref_mut, ref_dist = secded_embed_uint8(arr)
        torch_mut, torch_dist = secded_embed_uint8_torch(torch.from_numpy(arr.copy()))
        assert np.array_equal(ref_mut, torch_mut.numpy()), (
            f"secded_torch disagrees with secded_vectorized on shape {shape}!"
        )
        assert abs(ref_dist - torch_dist.item()) < 1e-6, (
            f"distortion mismatch on shape {shape}: {ref_dist} vs {torch_dist.item()}"
        )
    print(f"[OK] secded_torch == secded_vectorized on {len(shapes)} shapes "
          f"(incl. padded tail-chunk cases)")


def test_decode_roundtrip_no_error(n_chunks=200, seed=1):
    rng = random.Random(seed)
    weights = [7 - (i % 8) for i in range(N_MOTHER)]  # phase-0 layout
    for _ in range(n_chunks):
        bits = [rng.randint(0, 1) for _ in range(N_MOTHER)]
        chunk = {
            "sliced_message_bits": bits,
            "sliced_bit_weights": weights,
            "sliced_message_nums": [],
        }
        out = SECDEDInputEmbed(chunk, message_parity_size=N_MOTHER, message_size=K_MSG)
        mutated = out["sliced_message_bits"]
        decoded, status = SECDEDDecode(mutated, weights)
        assert status == "ok", f"expected ok, got {status}"
        assert decoded == mutated
    print(f"[OK] {n_chunks} error-free encode/decode round-trips")


def test_single_bit_correction(n_trials=500, seed=2):
    rng = random.Random(seed)
    weights = [7 - (i % 8) for i in range(N_MOTHER)]
    corrected_ok = 0
    for _ in range(n_trials):
        bits = [rng.randint(0, 1) for _ in range(N_MOTHER)]
        chunk = {"sliced_message_bits": bits, "sliced_bit_weights": weights,
                  "sliced_message_nums": []}
        out = SECDEDInputEmbed(chunk, message_parity_size=N_MOTHER, message_size=K_MSG)
        codeword = out["sliced_message_bits"][:]
        flip_pos = rng.randrange(N_MOTHER)
        corrupted = codeword[:]
        corrupted[flip_pos] ^= 1
        decoded, status = SECDEDDecode(corrupted, weights)
        assert status == "corrected", f"expected corrected, got {status}"
        assert decoded == codeword, "single-bit correction produced wrong codeword"
        corrected_ok += 1
    print(f"[OK] {corrected_ok}/{n_trials} single-bit-flip corrections exact")


def test_double_bit_detection(n_trials=500, seed=3):
    rng = random.Random(seed)
    weights = [7 - (i % 8) for i in range(N_MOTHER)]
    detected = 0
    for _ in range(n_trials):
        bits = [rng.randint(0, 1) for _ in range(N_MOTHER)]
        chunk = {"sliced_message_bits": bits, "sliced_bit_weights": weights,
                  "sliced_message_nums": []}
        out = SECDEDInputEmbed(chunk, message_parity_size=N_MOTHER, message_size=K_MSG)
        codeword = out["sliced_message_bits"][:]
        p1, p2 = rng.sample(range(N_MOTHER), 2)
        corrupted = codeword[:]
        corrupted[p1] ^= 1
        corrupted[p2] ^= 1
        decoded, status = SECDEDDecode(corrupted, weights)
        assert status == "double_error_detected", (
            f"expected double_error_detected, got {status}"
        )
        detected += 1
    print(f"[OK] {detected}/{n_trials} double-bit-flip cases correctly flagged "
          f"(never mis-corrected)")


if __name__ == "__main__":
    test_vectorized_matches_scalar()
    test_torch_matches_vectorized()
    test_decode_roundtrip_no_error()
    test_single_bit_correction()
    test_double_bit_detection()
    print("\nAll SECDED codec tests passed.")
