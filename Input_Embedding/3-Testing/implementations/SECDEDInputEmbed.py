"""
SECDEDInputEmbed.py — scalar reference implementation of the SECDED-class
input/activation embedding used by RECAP.

This is the activation-embedding analogue of ECC-CODE-Engine's
implementations/ParityOverwriteByTopWeightsEncode.py: it keeps the same
"chunk + rank positions by bit-significance weight + overwrite the
lowest-weight (LSB) positions with parity" scheme, applied here to
quantized-int8 activations instead of quantized-int8 weights.

Why not reuse ParityOverwriteByTopWeightsEncode(...) verbatim?
----------------------------------------------------------------
That function builds a BCH(n, k) code via `galois.BCH(n, k)`, which for
n=63 only supports the mother-code k values in the weight pipeline's
NANDT_TO_K[63] table (57, 51, 45, ...) — every one of those is a pure
Hamming-style code: single-error-CORRECTING only (SEC, dmin=3), with no
reliable double-error detection.

Weights live in DRAM and need the stronger multi-bit-error protection those
larger-t BCH codes give (see ECC-CODE-Engine's t=1..8 sweep). Activations
here only ever sit in on-chip SRAM under a much lower bit-error rate, so a
plain SEC code is not what's being asked for — the brief calls for SECDED:
Single-Error-Correct, Double-Error-Detect (dmin=4).

The standard way to get dmin=4 out of a Hamming code is to EXTEND it with
one overall (whole-codeword) XOR parity bit. Extending the mother
BCH(63,57) SEC code the usual way needs 64 total bits (57 message + 6
Hamming parity + 1 overall parity), one more than our 63-bit chunk. Rather
than grow the chunk (which would desynchronize it from the rest of the
codebase's chunk/bit-weight-ordering utilities, copied verbatim from
utils/), this shortens the code by one message bit: 56 message + 6 Hamming
parity + 1 overall parity = 63, so it fits the existing 63-bit chunk with
zero data bits sacrificed. Shortening a linear code never decreases its
minimum distance, so the shortened code is still a valid dmin=4 SECDED
code. This is the same trick real SRAM SECDED designs use to hit specific
word widths (e.g. (39,32) SECDED is a shortened (72,64) EDAC code).

Bit-significance ordering ("importance of the bits") is inherited
unchanged from ParityOverwriteByTopWeightsEncode: within a chunk, the
highest-weight (most-significant) positions are kept as message bits
untouched; parity is written into the lowest-weight (least-significant)
positions, LSB-first. Among the 7 parity positions, the single lowest
weight slot carries the overall/DED parity bit (the cheapest, least
consequential bit to guess wrong) and the next 6 lowest-weight slots carry
the Hamming/SEC parity (in the same ascending-weight assignment order
ParityOverwriteByTopWeightsEncode uses).

This module is the scalar, easy-to-audit spec/oracle. The actual
inference-time embedding uses the vectorized re-implementation in
secded_vectorized.py (same algorithm, batched over whole activation
tensors via GF(2) matrix multiplication) — secded_vectorized.py's unit
tests assert exact agreement with this module chunk-by-chunk.
"""
import numpy as np
import galois

N_MOTHER = 63
K_MOTHER = 57          # BCH(63,57): the SEC-only Hamming mother code
HAMMING_PARITY = N_MOTHER - K_MOTHER   # 6
K_MSG = K_MOTHER - 1   # 56 real message bits (mother code shortened by 1)
PARITY_TOTAL = N_MOTHER - K_MSG        # 7 (6 Hamming + 1 overall/DED)

_bch = None
_P56 = None  # (56, 6) GF(2) systematic-parity submatrix, shortened mother code


def _get_P56():
    """Build (once, cached) the (56,6) GF(2) parity-generator submatrix."""
    global _bch, _P56
    if _P56 is None:
        _bch = galois.BCH(N_MOTHER, K_MOTHER)   # mother code: BCH(63,57), t=1
        G = _bch.G                              # (57, 63) systematic generator
        P_from_G = G[:, _bch.k:]                # (57, 6) parity submatrix
        # Shorten: drop the row for the 57th (virtual, fixed-to-0) message bit.
        _P56 = np.array(P_from_G[:K_MSG, :], dtype=np.uint8)   # (56, 6)
    return _P56


def SECDEDInputEmbed(
    chunk,
    message_parity_size=63,
    message_size=56,
):
    """
    SECDED-class analogue of ParityOverwriteByTopWeightsEncode: same chunk
    dict in/out contract, same top-weighted-positions-as-message convention,
    but produces a (63,56) shortened extended-Hamming SECDED codeword (6
    Hamming/SEC parity bits + 1 overall/DED parity bit) instead of a plain
    BCH(n,k) SEC-only codeword.

    Args:
        chunk: dict with "sliced_message_bits", "sliced_bit_weights",
               "sliced_message_nums" (see messageSliceBasedOnChunkSize).
        message_parity_size: n (must equal len(chunk bits)); only 63 supported.
        message_size: k (must equal K_MSG=56).

    Returns: dict with the same keys ParityOverwriteByTopWeightsEncode
        returns (mutated sliced_message_bits/nums, message_indices,
        parity_indices), plus "parity_hamming_indices" /
        "parity_overall_index" identifying which parity slot is which.
    """
    original_message_bits_slice = chunk['sliced_message_bits']
    original_bits_weights_slice = chunk['sliced_bit_weights']
    original_nums_spans        = chunk.get('sliced_message_nums', [])

    bits = list(map(int, original_message_bits_slice))
    weights = list(map(int, original_bits_weights_slice))

    if len(bits) != len(weights):
        raise ValueError("bits and weights must have the same length.")
    n = len(bits)
    if n == 0:
        return {"sliced_message_bits": [], "message_indices": [], "parity_indices": [],
                "parity_hamming_indices": [], "parity_overall_index": None,
                "sliced_bit_weights": [], "sliced_message_nums": []}
    if message_parity_size != n or n != N_MOTHER:
        raise ValueError(f"SECDEDInputEmbed only supports n={N_MOTHER} (got n={n}).")
    if message_size != K_MSG:
        raise ValueError(f"SECDEDInputEmbed only supports message_size={K_MSG} (got {message_size}).")
    if any(b not in (0, 1) for b in bits):
        raise ValueError("original_message_bits_slice must contain only 0/1.")

    k = K_MSG

    # --- 1) Rank positions by weight; top-k = message (kept unchanged) ---
    ranked_idx_desc = sorted(range(n), key=lambda i: (-weights[i], i))
    message_indices = ranked_idx_desc[:k]

    # --- 2) Remaining PARITY_TOTAL positions, ascending weight (LSB-first) ---
    parity_indices = sorted(
        set(range(n)) - set(message_indices),
        key=lambda i: (weights[i], i),
    )
    if len(parity_indices) != PARITY_TOTAL:
        raise RuntimeError("Internal error: parity index count mismatch.")

    # Lowest-weight slot -> overall/DED parity; next 6 -> Hamming/SEC parity.
    parity_overall_index = parity_indices[0]
    parity_hamming_indices = parity_indices[1:]

    # --- 3) Hamming parity via the shortened (56,6) GF(2) submatrix ---
    P56 = _get_P56()
    m_vec = np.array([bits[i] for i in message_indices], dtype=int)   # (56,)
    hamming_parity = (m_vec @ P56) % 2                                # (6,)

    # --- 4) Overall parity = XOR of the 56 message bits + 6 Hamming bits ---
    overall_parity = int(np.bitwise_xor.reduce(
        np.concatenate([m_vec, hamming_parity])
    ))

    # --- 5) Overwrite parity slots ---
    mutated = bits[:]
    for pos, pb in zip(parity_hamming_indices, hamming_parity.tolist()):
        mutated[pos] = int(pb)
    mutated[parity_overall_index] = overall_parity

    # --- 6) Recompute sliced_message_nums from mutated bits (same as
    #         ParityOverwriteByTopWeightsEncode) ---
    updated_nums = []
    for rec in original_nums_spans:
        idx = rec.get("index")
        s = rec.get("start")
        e = rec.get("end")
        if s is None or e is None or s >= e:
            updated_nums.append({"index": idx, "value": 0, "start": s, "end": e})
            continue
        val = 0
        for i in range(s, e):
            if mutated[i]:
                val += (1 << weights[i])
        updated_nums.append({"index": idx, "value": val, "start": s, "end": e})

    return {
        "sliced_message_bits": mutated,
        "sliced_bit_weights": original_bits_weights_slice,
        "message_indices": message_indices,
        "parity_indices": parity_indices,
        "parity_hamming_indices": parity_hamming_indices,
        "parity_overall_index": parity_overall_index,
        "sliced_message_nums": updated_nums,
    }


def SECDEDDecode(mutated_bits, weights):
    """
    Reference decoder for the (63,56) SECDED code: returns
    (corrected_bits, status) where status in
    {"ok", "corrected", "double_error_detected"}.

    Not used by the accuracy-evaluation path (which measures embedding-
    induced distortion with zero injected faults, mirroring
    6-BaseAccuracyTesting), but kept here so the encode/decode round-trip
    can be unit-tested and so a future fault-injection study can reuse it.
    """
    n = len(mutated_bits)
    if n != N_MOTHER:
        raise ValueError(f"expected n={N_MOTHER}, got {n}")

    ranked_idx_desc = sorted(range(n), key=lambda i: (-weights[i], i))
    message_indices = ranked_idx_desc[:K_MSG]
    parity_indices = sorted(
        set(range(n)) - set(message_indices), key=lambda i: (weights[i], i)
    )
    parity_overall_index = parity_indices[0]
    parity_hamming_indices = parity_indices[1:]

    bits = list(map(int, mutated_bits))
    m_vec = np.array([bits[i] for i in message_indices], dtype=int)
    received_hamming = np.array([bits[i] for i in parity_hamming_indices], dtype=int)
    received_overall = bits[parity_overall_index]

    P56 = _get_P56()
    expected_hamming = (m_vec @ P56) % 2
    syndrome_bits = (received_hamming ^ expected_hamming)
    syndrome_nonzero = bool(syndrome_bits.any())

    all_bits_no_overall = np.concatenate([m_vec, received_hamming])
    computed_overall = int(np.bitwise_xor.reduce(all_bits_no_overall))
    overall_mismatch = (computed_overall != received_overall)

    if not syndrome_nonzero and not overall_mismatch:
        return bits, "ok"
    if syndrome_nonzero and overall_mismatch:
        # Single-bit error: locate via syndrome -> column of H matching it.
        # H columns for message bits are P56 rows; for hamming-parity bits,
        # identity columns. (Overall-parity-bit-only errors are handled by
        # the syndrome-zero branch below.)
        H_msg = P56  # (56,6): column i of H (for message bit i) = P56[i,:]
        flipped = None
        for i in range(K_MSG):
            if np.array_equal(H_msg[i], syndrome_bits):
                flipped = ("message", i)
                break
        if flipped is None:
            for j in range(HAMMING_PARITY):
                col = np.zeros(HAMMING_PARITY, dtype=int)
                col[j] = 1
                if np.array_equal(col, syndrome_bits):
                    flipped = ("hamming", j)
                    break
        corrected = bits[:]
        if flipped is not None:
            kind, i = flipped
            pos = message_indices[i] if kind == "message" else parity_hamming_indices[i]
            corrected[pos] ^= 1
        return corrected, "corrected"
    if syndrome_nonzero and not overall_mismatch:
        return bits, "double_error_detected"
    # syndrome zero, overall mismatch -> error is in the overall-parity bit itself
    corrected = bits[:]
    corrected[parity_overall_index] ^= 1
    return corrected, "corrected"
