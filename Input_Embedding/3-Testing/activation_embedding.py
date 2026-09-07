"""
activation_embedding.py — hooks that apply SECDED input embedding to every
Conv2d/Linear layer's input activation during inference.

Design
------
For every nn.Conv2d / nn.Linear in the model, a forward-PRE-hook:
  1. Reads the float input tensor (this covers BOTH the network's raw
     "sensor" input — the first Conv2d's input is the normalized image
     tensor itself — and every intermediate MAC output feeding the next
     layer, since a layer's output IS the next layer's input).
  2. Fake-quantizes it to int8 with a per-tensor symmetric scale fixed
     ahead of time by a short calibration pass (simple max-abs/MinMax
     observer — standard PTQ activation-quantization practice).
  3. Applies the (63,56) SECDED LSB-overwrite embedding from
     implementations/secded_torch.py (GPU-native — runs on x's own
     device, no host round trip) — this is what actually happens
     to a value that lives in SRAM with parity embedded in its own LSBs;
     it introduces distortion even with zero injected bit-flips, which is
     exactly what this study measures (mirrors ECC-CODE-Engine's
     6-BaseAccuracyTesting, which measures the embedded-but-unattacked
     weight accuracy).
  4. Dequantizes back to float and lets the layer's own (unmodified,
     original-precision) weights do the actual MAC — per the study's
     requirement that only inputs are embedded, weights are used as-is.

Layers named in `skip_layers` are passed through this quantize/dequantize
step WITHOUT the SECDED overwrite (still int8-quantized, so the comparison
against the embedded run isolates the LSB-overwrite's cost specifically) —
this is the "regular ECC" escape hatch: protecting that layer's activation
with a conventional, non-destructive parity scheme (external bits, no
distortion) instead of embedding, exactly as ECC-CODE-Engine can hold
weights unembedded on some layers.
"""
import torch
import torch.nn as nn

from implementations.secded_torch import secded_embed_uint8_torch

QMIN, QMAX = -128, 127


class ActivationEmbedder:
    def __init__(self, skip_layers=None, enabled=True, num_threads=None):
        self.skip_layers = set(skip_layers or [])
        self.enabled = enabled
        # kept for CLI/back-compat (--num-threads); the GPU-native encoder
        # (secded_torch) has no CPU thread pool to size, so this is unused.
        self.num_threads = num_threads
        self.mode = "calibrate"          # "calibrate" | "run"
        self.calib_max_abs = {}          # name -> running max|x|
        self.scales = {}                 # name -> fixed scale (post-calibration)
        self.distortion_sum = {}         # name -> sum of per-batch mean-abs distortion (float domain)
        self.distortion_batches = {}     # name -> count
        self._handles = []
        self._names = []

    # ---- lifecycle ----
    def attach(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._names.append(name)
                h = module.register_forward_pre_hook(self._make_hook(name))
                self._handles.append(h)
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def start_calibration(self):
        self.mode = "calibrate"
        self.calib_max_abs = {}

    def finish_calibration(self):
        # Symmetric int8 scale from the observed max-abs (MinMax observer).
        self.scales = {
            name: max(float(m), 1e-8) / QMAX
            for name, m in self.calib_max_abs.items()
        }
        self.mode = "run"
        self.reset_distortion_tracking()

    def reset_distortion_tracking(self):
        self.distortion_sum = {n: 0.0 for n in self._names}
        self.distortion_batches = {n: 0 for n in self._names}

    def layer_names(self):
        return list(self._names)

    def mean_distortion(self, name):
        b = self.distortion_batches.get(name, 0)
        if b == 0:
            return 0.0
        s = self.distortion_sum[name]
        # distortion_sum accumulates as an un-synced GPU tensor (see the
        # hook below) — pull it to host here, once per layer per pass,
        # instead of once per batch.
        if torch.is_tensor(s):
            s = s.item()
        return s / b

    def distortion_report(self):
        """{layer_name: mean per-element |int8 code delta| * scale}, i.e. the
        average float-domain magnitude change purely from LSB-parity
        overwrite (zero injected faults). Informational only — the fallback
        search ranks layers by ablation sensitivity, not by this."""
        return {n: self.mean_distortion(n) for n in self._names}

    # ---- hook ----
    def _make_hook(self, name):
        def hook(module, inputs):
            x = inputs[0]
            if not torch.is_tensor(x) or not x.dtype.is_floating_point:
                return None

            if self.mode == "calibrate":
                cur = float(x.detach().abs().max().item())
                prev = self.calib_max_abs.get(name, 0.0)
                if cur > prev:
                    self.calib_max_abs[name] = cur
                return None

            scale = self.scales.get(name)
            if scale is None:
                # Layer never appeared during calibration (e.g. conditional
                # branch) — pass through un-embedded rather than guess a scale.
                return None

            x_q = torch.clamp(torch.round(x.detach() / scale), QMIN, QMAX)

            if self.enabled and name not in self.skip_layers:
                x_u8 = (x_q + 128).to(torch.uint8)
                mutated_u8, _byte_distortion = secded_embed_uint8_torch(x_u8)
                x_q_mut = (mutated_u8.to(torch.int16) - 128).to(dtype=x.dtype)

                # Stay on-device: a per-batch .item() here would force a
                # host sync on every one of the 21 hooked layers, every
                # batch. Accumulate as a GPU tensor; mean_distortion()
                # syncs once, at the end of the pass, instead.
                elem_distortion = (x_q_mut - x_q).abs().mean() * scale
                self.distortion_sum[name] += elem_distortion
                self.distortion_batches[name] += 1
            else:
                x_q_mut = x_q

            x_deq = x_q_mut * scale
            new_inputs = (x_deq,) + tuple(inputs[1:])
            return new_inputs

        return hook


@torch.no_grad()
def calibrate(embedder: ActivationEmbedder, model, loader, device, max_batches):
    embedder.start_calibration()
    model.eval()
    for i, (x, _y) in enumerate(loader):
        if i >= max_batches:
            break
        model(x.to(device, non_blocking=True))
    embedder.finish_calibration()
