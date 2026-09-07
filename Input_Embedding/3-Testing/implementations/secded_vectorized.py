"""
secded_vectorized.py — vectorized (63,56) SECDED encoder for activations.

Implements exactly the algorithm in SECDEDInputEmbed.py (same chunking /
bit-significance ranking / parity-slot convention — see that module's
docstring for the code design), but reformulated as a handful of batched
GF(2) matrix multiplications over the WHOLE activation tensor at once
instead of a Python loop over 63-bit chunks.

Why this is safe to vectorize (and why it's fast)
---------------------------------------------------
messageSliceBasedOnChunkSize()/convert_to_binary() give every bit a
"weight" equal to its OWN number's bit-significance (7=MSB..0=LSB) — that
never depends on which chunk the bit lands in. So for the flattened,
MSB-first bitstream, the weight of global bit position g is simply
`7 - (g % 8)`, a closed-form function of position alone.

A chunk holds 63 consecutive bits; 63 and 8 are coprime, so a chunk's
*local* weight pattern (which local position gets which weight) shifts by
63 mod 8 = 7 (== -1 mod 8) bits from one chunk to the next, and repeats
with period 8 chunks (lcm(63,8)/63 = 8). That means there are only 8
distinct chunk layouts ("phases") ever used, and — crucially — each one is
computable from chunk-structure alone, never from the data. This lets us:

  1. Precompute, once, the message/parity index layout for each of the 8
     phases (pure index bookkeeping, no data).
  2. Reshape the whole flattened bitstream into (num_chunks, 63).
  3. Group chunk-rows by phase (chunk_idx % 8) and encode every phase's
     group in ONE matrix multiply: (rows, 56) @ (56, 6) mod 2.

This turns what was an O(chunks) Python/galois loop (the approach used for
static weight tensors in ECC-CODE-Engine, where a multiprocessing pool of
CPU workers parallelizes across independent chunks — see
4-EmbeddingECC/run.sh) into O(1) large numpy ops per phase — the same
embarrassingly-parallel workload, but expressed as SIMD/array parallelism
instead of process-level parallelism, which is what actually makes
per-inference activation embedding (millions of activation elements per
image, tens of thousands of images) tractable. 3-Testing/run.sh still
parallelizes at the SLURM-array/process level (one array task per
architecture, matching 4-EmbeddingECC/run.sh's pattern) on top of this.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from implementations.SECDEDInputEmbed import N_MOTHER, K_MSG, _get_P56

_PERIOD = 8   # lcm(63, 8) / 63


def _local_weights(phase: int):
    """Weight array (length 63) of the bit at local position p in a chunk
    whose global start offset is `phase` chunks into the stream (i.e. the
    chunk's first bit is global position phase*63)."""
    start = phase * N_MOTHER
    return [7 - ((start + p) % 8) for p in range(N_MOTHER)]


def _layout_from_weights(weights):
    ranked_desc = sorted(range(N_MOTHER), key=lambda i: (-weights[i], i))
    message_idx = ranked_desc[:K_MSG]
    parity_idx = sorted(
        set(range(N_MOTHER)) - set(message_idx), key=lambda i: (weights[i], i)
    )
    overall_idx = parity_idx[0]
    hamming_idx = parity_idx[1:]
    return {
        "message_idx": np.array(message_idx, dtype=np.int64),
        "hamming_idx": np.array(hamming_idx, dtype=np.int64),
        "overall_idx": int(overall_idx),
    }


_PHASE_LAYOUTS = [_layout_from_weights(_local_weights(ph)) for ph in range(_PERIOD)]


_MAX_PHASE_THREADS = min(_PERIOD, max(1, os.cpu_count() or 1))


def secded_embed_uint8(arr_u8: np.ndarray, num_threads: int = None):
    """
    Apply the (63,56) SECDED LSB-parity-overwrite embedding to a uint8
    array of arbitrary shape (values 0..255 — e.g. an int8 activation
    already shifted by +128).

    The 8 phase-groups (see module docstring) are independent, so they run
    on a small thread pool by default — numpy's C-level array ops release
    the GIL, so this measurably speeds up the large fancy-index / einsum
    calls per phase (~2x on a multi-core node) without any multiprocessing
    / IPC cost. Pass num_threads=1 to force single-threaded execution.

    Returns:
        mutated_u8: same shape/dtype as arr_u8, LSB positions overwritten
                    with SECDED parity per the 63-bit chunking scheme.
        mean_abs_distortion: float — mean absolute value change introduced
                    purely by the parity overwrite (zero injected faults;
                    this is the "cost of embedding", matching the
                    zero-fault accuracy measured throughout
                    6-BaseAccuracyTesting for the weight pipeline).
    """
    if num_threads is None:
        num_threads = _MAX_PHASE_THREADS
    orig_shape = arr_u8.shape
    flat = np.ascontiguousarray(arr_u8.reshape(-1)).astype(np.uint8)
    numel = flat.shape[0]
    if numel == 0:
        return arr_u8.copy(), 0.0

    bits = np.unpackbits(flat)               # (numel*8,), MSB-first per byte
    total_bits = bits.shape[0]

    pad_len = (-total_bits) % N_MOTHER
    if pad_len:
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
    num_chunks = bits.shape[0] // N_MOTHER

    chunks = bits.reshape(num_chunks, N_MOTHER).astype(np.uint8)
    chunk_phase = np.arange(num_chunks) % _PERIOD
    # The tail chunk (if padded) needs a pad-aware layout (see below) and
    # must NOT be touched by the generic phase-formula bulk pass below —
    # that formula doesn't know about padding, so it would pick the wrong
    # message/parity split for that one row and corrupt real message bits.
    if pad_len:
        chunk_phase[num_chunks - 1] = -1   # excluded from every phase group

    P56 = _get_P56().astype(np.uint8)         # (56, 6), entries in {0,1}

    # NOTE: `@`/np.matmul on integer dtypes falls back to a slow scalar loop
    # (no BLAS backend for ints) — for ~150k-row phase groups that alone
    # dominated the whole encode (~2s per phase, ~16s per 10M-element
    # tensor). np.einsum's own reduction loop is ~60x faster here. Every
    # partial sum stays <= K_MSG=56, so uint8 accumulation never overflows.
    def _encode_phase(ph):
        row_idx = np.nonzero(chunk_phase == ph)[0]
        if row_idx.size == 0:
            return
        layout = _PHASE_LAYOUTS[ph]
        rows = chunks[row_idx]                                 # (r, 63)
        message_block = rows[:, layout["message_idx"]]          # (r, 56)

        hamming = np.einsum('rk,kp->rp', message_block, P56, optimize=True) % 2   # (r, 6)
        overall = np.bitwise_xor.reduce(
            np.concatenate([message_block, hamming], axis=1),
            axis=1,
        )                                                       # (r,)

        rows[:, layout["hamming_idx"]] = hamming
        rows[:, layout["overall_idx"]] = overall
        chunks[row_idx] = rows

    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            list(ex.map(_encode_phase, range(_PERIOD)))
    else:
        for ph in range(_PERIOD):
            _encode_phase(ph)

    # ---- Tail chunk fixup ----
    # messageSliceBasedOnChunkSize pads a short final chunk with (bit=0,
    # weight=0) — i.e. padding positions tie with real LSB (weight-0) bits
    # but sort *after* them (higher local index), so they're preferentially
    # pushed into parity slots rather than stealing a message slot from real
    # data. The phase-formula weight (7 - global_pos % 8) doesn't know about
    # padding, so redo just this one chunk (if any) with the correct,
    # pad-aware weight array to match the scalar reference exactly.
    if pad_len:
        real_len = N_MOTHER - pad_len
        start = (num_chunks - 1) * N_MOTHER
        tail_weights = [7 - ((start + p) % 8) for p in range(real_len)] + [0] * pad_len
        tail_layout = _layout_from_weights(tail_weights)
        row = chunks[num_chunks - 1:num_chunks]                 # (1, 63)
        message_block = row[:, tail_layout["message_idx"]]
        hamming = np.einsum('rk,kp->rp', message_block, P56, optimize=True) % 2
        overall = np.bitwise_xor.reduce(
            np.concatenate([message_block, hamming], axis=1),
            axis=1,
        )
        row[:, tail_layout["hamming_idx"]] = hamming
        row[:, tail_layout["overall_idx"]] = overall
        chunks[num_chunks - 1:num_chunks] = row

    mutated_bits = chunks.reshape(-1)[:total_bits]
    mutated_flat = np.packbits(mutated_bits)                    # (numel,)
    mutated_u8 = mutated_flat.reshape(orig_shape)

    distortion = float(
        np.abs(mutated_flat.astype(np.int32) - flat.astype(np.int32)).mean()
    )
    return mutated_u8, distortion
