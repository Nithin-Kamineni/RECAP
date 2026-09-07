"""
secded_torch.py — GPU-native (63,56) SECDED LSB-parity-overwrite encoder.

Same algorithm as secded_vectorized.secded_embed_uint8 (same chunk /
phase / bit-significance-weight layout — see that module's docstring for
the derivation), reformulated as torch tensor ops that run entirely on
the activation tensor's own device (GPU), inline inside the forward-hook,
with no host<->device transfer.

Why this exists
----------------
Profiling input_embed_test_accuracy.py showed 0% GPU utilization during
the "input_embed" pass: activation_embedding.py's hook did
`.cpu().numpy()` -> secded_vectorized.secded_embed_uint8 (CPU, numpy) ->
`torch.from_numpy(...).to(device)`, once per hooked Conv2d/Linear layer,
per batch. That serializes every batch's forward pass behind ~60s of
single-node CPU work while the GPU sits idle.

Two rounds of optimization vs. a naive torch port of the numpy algorithm:

1. One matmul instead of eight. secded_vectorized.py's numpy version
   loops over the 8 independent bit-chunk "phases" (see its docstring),
   doing a separate index_select + einsum + scatter per phase. But every
   phase shares the same (56,6) parity-generator matrix P56 — only the
   *which-local-position-is-message-vs-parity* mapping differs per phase.
   So instead of 8 small phase-restricted matmuls, this builds one
   per-chunk-row index (via a gather from an (8,56)/(8,6)/(8,) phase
   table, keyed by `row_index % 8`) and does ONE gather + ONE
   (num_chunks,56)@(56,6) matmul + ONE scatter for the whole layer. This
   also means the phase tables (tiny, (8,56) etc.) are cached, but the
   full per-chunk index tensors are recomputed cheaply each call rather
   than cached — caching those would mean holding ~16GB of int64 index
   tensors resident (summed over all 21 hooked layers) for a ResNet18-
   sized model, which doesn't fit alongside the model/activations.

2. No host<->device sync inside the hot loop. The original vectorized
   code selected each phase's rows via `np.nonzero(chunk_phase == ph)`;
   a naive torch port of that (`torch.nonzero` on a CUDA tensor) forces a
   device-to-host sync to learn the result's size, once per phase per
   layer per batch (~8*21*391 syncs for a full ResNet18 val-set pass),
   each one stalling the CUDA queue. Row-phase assignment only depends on
   a chunk's *position* in the flattened bitstream (a closed-form
   function of shape, not data — see secded_vectorized.py's docstring),
   so it's computed here with plain index arithmetic (arange % 8) instead
   of a data-dependent boolean mask, and the per-layer distortion scalar
   this module used to return is left as an un-synced GPU tensor (the
   caller decides if/when to pull it to host) for the same reason.
"""
import torch

from implementations.SECDEDInputEmbed import N_MOTHER, K_MSG, _get_P56
from implementations.secded_vectorized import _PERIOD, _PHASE_LAYOUTS, _layout_from_weights

_device_tables = {}       # device str -> {"P56", "phase_message_idx", "phase_hamming_idx", "phase_overall_idx", "bit_shifts", "pow2"}
_tail_layout_cache = {}   # (num_chunks, pad_len, device str) -> layout dict of torch tensors (tiny; safe to cache)


def _get_device_tables(device: torch.device):
    key = str(device)
    tables = _device_tables.get(key)
    if tables is not None:
        return tables

    P56 = torch.as_tensor(_get_P56(), dtype=torch.float32, device=device)   # (56, 6)
    phase_message_idx = torch.stack([
        torch.as_tensor(layout["message_idx"], dtype=torch.int64, device=device)
        for layout in _PHASE_LAYOUTS
    ])   # (8, 56)
    phase_hamming_idx = torch.stack([
        torch.as_tensor(layout["hamming_idx"], dtype=torch.int64, device=device)
        for layout in _PHASE_LAYOUTS
    ])   # (8, 6)
    phase_overall_idx = torch.as_tensor(
        [int(layout["overall_idx"]) for layout in _PHASE_LAYOUTS], dtype=torch.int64, device=device
    )    # (8,)
    bit_shifts = torch.arange(7, -1, -1, dtype=torch.int64, device=device)   # MSB..LSB
    pow2 = (2 ** bit_shifts)                                                  # [128,...,1]

    tables = {
        "P56": P56,
        "phase_message_idx": phase_message_idx,
        "phase_hamming_idx": phase_hamming_idx,
        "phase_overall_idx": phase_overall_idx,
        "bit_shifts": bit_shifts,
        "pow2": pow2,
    }
    _device_tables[key] = tables
    return tables


def _get_tail_layout(num_chunks: int, pad_len: int, device: torch.device):
    key = (num_chunks, pad_len, str(device))
    layout = _tail_layout_cache.get(key)
    if layout is not None:
        return layout

    real_len = N_MOTHER - pad_len
    start = (num_chunks - 1) * N_MOTHER
    tail_weights = [7 - ((start + p) % 8) for p in range(real_len)] + [0] * pad_len
    raw = _layout_from_weights(tail_weights)
    layout = {
        "message_idx": torch.as_tensor(raw["message_idx"], dtype=torch.int64, device=device),
        "hamming_idx": torch.as_tensor(raw["hamming_idx"], dtype=torch.int64, device=device),
        "overall_idx": int(raw["overall_idx"]),
    }
    _tail_layout_cache[key] = layout
    return layout


def _hamming_and_overall(message_block: torch.Tensor, P56: torch.Tensor):
    """message_block: (r, 56) uint8 -> (hamming (r,6) uint8, overall (r,) uint8)."""
    hamming = torch.remainder(message_block.to(torch.float32) @ P56, 2.0).round().to(torch.uint8)
    # XOR-reduce of 0/1 bits == parity(sum of bits); avoids a dim-wise bitwise-xor reduction.
    overall = torch.remainder(
        message_block.to(torch.int64).sum(dim=1) + hamming.to(torch.int64).sum(dim=1), 2
    ).to(torch.uint8)
    return hamming, overall


def secded_embed_uint8_torch(arr_u8: torch.Tensor):
    """
    Apply the (63,56) SECDED LSB-parity-overwrite embedding to a uint8
    torch tensor of arbitrary shape, entirely on `arr_u8.device` (no
    host<->device transfer, no data-dependent sync). Numerically
    identical to secded_vectorized.secded_embed_uint8 (see test_secded.py
    for the cross-check against both that function and the scalar
    reference).

    Returns:
        mutated_u8: torch.uint8 tensor, same shape/device as arr_u8.
        distortion: 0-dim torch.float32 tensor (still on `device` — call
                    .item() only if/when you actually need it on host)
                    — mean absolute value change from the parity
                    overwrite (zero injected faults).
    """
    device = arr_u8.device
    tables = _get_device_tables(device)
    orig_shape = arr_u8.shape
    flat = arr_u8.reshape(-1).to(torch.int64)
    numel = flat.shape[0]
    if numel == 0:
        return arr_u8.clone(), torch.zeros((), device=device)

    bits = ((flat.unsqueeze(-1) >> tables["bit_shifts"]) & 1).reshape(-1).to(torch.uint8)  # MSB-first
    total_bits = bits.shape[0]

    pad_len = (-total_bits) % N_MOTHER
    if pad_len:
        bits = torch.cat([bits, torch.zeros(pad_len, dtype=torch.uint8, device=device)])
    num_chunks = bits.shape[0] // N_MOTHER
    chunks = bits.reshape(num_chunks, N_MOTHER)

    num_real = num_chunks - 1 if pad_len else num_chunks   # tail chunk (if any) excluded here
    P56 = tables["P56"]

    if num_real > 0:
        row_phase = torch.arange(num_real, device=device) % _PERIOD           # (r,)
        msg_idx = tables["phase_message_idx"].index_select(0, row_phase)      # (r, 56)
        ham_idx = tables["phase_hamming_idx"].index_select(0, row_phase)      # (r, 6)
        ovr_idx = tables["phase_overall_idx"].index_select(0, row_phase)      # (r,)

        real_rows = chunks[:num_real]
        message_block = torch.gather(real_rows, 1, msg_idx)                   # (r, 56)
        hamming, overall = _hamming_and_overall(message_block, P56)

        real_rows.scatter_(1, ham_idx, hamming)
        real_rows.scatter_(1, ovr_idx.unsqueeze(1), overall.unsqueeze(1))

    if pad_len:
        tail_layout = _get_tail_layout(num_chunks, int(pad_len), device)
        tail_row = chunks[-1:]
        message_block = tail_row.index_select(1, tail_layout["message_idx"])
        hamming, overall = _hamming_and_overall(message_block, P56)
        tail_row[:, tail_layout["hamming_idx"]] = hamming
        tail_row[:, tail_layout["overall_idx"]] = overall

    mutated_bits = chunks.reshape(-1)[:total_bits]
    mutated_bits8 = mutated_bits.reshape(numel, 8).to(torch.int64)
    mutated_flat = (mutated_bits8 * tables["pow2"]).sum(dim=1).to(torch.uint8)
    mutated_u8 = mutated_flat.reshape(orig_shape)

    distortion = (mutated_flat.to(torch.int32) - flat.to(torch.int32)).abs().float().mean()
    return mutated_u8, distortion
