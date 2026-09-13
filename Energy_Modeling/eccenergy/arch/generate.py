"""Workload generators: `python3 -m eccenergy.generate models|transformers`.

`models` extracts conv/linear shapes from torchvision/timm via forward hooks and
writes `ecc_energy_study/model_layers.json`. Needs torch, so it runs in the
container.

`transformers` builds LLM weight-matmul shapes analytically from each model's
config and writes `ecc_energy_study/transformer_layers.json`. No torch, no
checkpoint download -- HuggingFace GPT-2 stores weights in a custom Conv1D
class that nn.Linear hooks silently miss, and Llama checkpoints are gated and
multi-GB, so analytical shapes are both more reliable and more honest about GQA
and gated FFNs.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

from ..paths import CNN_LAYERS, TRANSFORMER_LAYERS, WORK
from ..settings import guards

# ------------------------------------------------------------------ CNN models
CNN_MODELS = {  # name -> (source, constructor)
    "resnet18": ("torchvision", "resnet18"),
    "resnet50": ("torchvision", "resnet50"),
    "densenet121": ("torchvision", "densenet121"),
    "squeezenet1_1": ("torchvision", "squeezenet1_1"),
    "mobilenet_v2": ("torchvision", "mobilenet_v2"),
    "efficientnet_b0": ("torchvision", "efficientnet_b0"),
    "convnext_tiny": ("torchvision", "convnext_tiny"),
    "xception": ("timm", "xception"),
}


def _load_model(source, ctor):
    if source == "torchvision":
        import torchvision.models as tvm
        return getattr(tvm, ctor)(weights=None)
    import timm
    return timm.create_model(ctor, pretrained=False)


def _extract_layers(model, hw):
    import torch
    layers = []

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(module, torch.nn.Conv2d):
                g = module.groups
                layers.append(dict(
                    name=name, type="conv",
                    # per-group in-channels: the exact weight count for grouped
                    # and depthwise convolutions
                    C=max(1, module.in_channels // g),
                    M=module.out_channels,
                    R=module.kernel_size[0], S=module.kernel_size[1],
                    P=int(out.shape[2]), Q=int(out.shape[3]),
                    Wstride=module.stride[1], Hstride=module.stride[0], groups=g))
            elif isinstance(module, torch.nn.Linear):
                # A Linear applied to a 4-D channels-last tensor (ConvNeXt's
                # pwconv1 / pwconv2 on [N, H, W, C]) is a 1x1 convolution over
                # H x W pixels, not one matmul. Until 2026-09-06 this recorded
                # P=Q=1 for those, undercounting their MACs and activation
                # traffic by H x W. A classifier head on [N, C] is still P=Q=1.
                p = int(out.shape[1]) if out.dim() == 4 else 1
                q = int(out.shape[2]) if out.dim() == 4 else 1
                layers.append(dict(
                    name=name, type="fc",
                    C=module.in_features, M=module.out_features,
                    R=1, S=1, P=p, Q=q, Wstride=1, Hstride=1, groups=1))
        return hook

    hooks = [m.register_forward_hook(make_hook(n))
             for n, m in model.named_modules()
             if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, hw, hw))
    for h in hooks:
        h.remove()
    return layers


def generate_models(input_hw=224, only=None):
    WORK.mkdir(parents=True, exist_ok=True)
    wanted = {k: v for k, v in CNN_MODELS.items() if not only or k in only}
    all_layers = {}
    if CNN_LAYERS.exists():
        try:  # keep models we are not regenerating
            all_layers = json.loads(CNN_LAYERS.read_text())
        except Exception:
            all_layers = {}
    for name, (source, ctor) in wanted.items():
        try:
            print(f"[extract] {name} ({source}) ...", flush=True)
            layers = _extract_layers(_load_model(source, ctor), input_hw)
            all_layers[name] = layers
            print(f"  -> {len(layers)} conv/linear layers")
        except Exception as exc:
            print(f"  !! FAILED {name}: {exc}")
            if os.environ.get("ECC_VERBOSE"):
                traceback.print_exc()
    CNN_LAYERS.write_text(json.dumps(all_layers, indent=1))
    print(f"\nSaved {len(all_layers)} models -> {CNN_LAYERS}")
    print("Models:", ", ".join(sorted(all_layers)))
    return all_layers


# --------------------------------------------------------------- transformers
TRANSFORMER_MODELS = {
    "distilgpt2": dict(family="gpt2", d=768, layers=6, heads=12, ffn=3072, vocab=50257),
    "gpt2": dict(family="gpt2", d=768, layers=12, heads=12, ffn=3072, vocab=50257),
    "bert_base": dict(family="bert", d=768, layers=12, heads=12, ffn=3072, vocab=30522),
    "gpt2_medium": dict(family="gpt2", d=1024, layers=24, heads=16, ffn=4096, vocab=50257),
    "opt_125m": dict(family="opt", d=768, layers=12, heads=12, ffn=3072, vocab=50272),
    "distilbert": dict(family="bert", d=768, layers=6, heads=12, ffn=3072, vocab=30522),
    "tinyllama": dict(family="llama", d=2048, layers=22, heads=32, kv_heads=4,
                      ffn=5632, vocab=32000),
}


def _block_matmuls(cfg):
    """(name, in_features, out_features) for ONE transformer block."""
    d, ffn, heads = cfg["d"], cfg["ffn"], cfg["heads"]
    head_dim = d // heads
    fam = cfg["family"]
    if fam == "gpt2":
        return [("attn_qkv", d, 3 * d), ("attn_out", d, d),
                ("ffn_up", d, ffn), ("ffn_down", ffn, d)]
    if fam in ("bert", "opt"):
        return [("attn_q", d, d), ("attn_k", d, d), ("attn_v", d, d), ("attn_out", d, d),
                ("ffn_up", d, ffn), ("ffn_down", ffn, d)]
    if fam == "llama":
        kv_dim = cfg.get("kv_heads", heads) * head_dim  # GQA -> smaller K/V
        return [("attn_q", d, d), ("attn_k", d, kv_dim), ("attn_v", d, kv_dim),
                ("attn_out", d, d), ("ffn_gate", d, ffn), ("ffn_up", d, ffn),
                ("ffn_down", ffn, d)]
    raise ValueError(f"unknown transformer family: {fam}")


def generate_transformers(seq=1, include_lm_head=True, include_embedding=False):
    """seq=1 is decode (memory bound, weights dominate); seq>1 is prefill."""
    WORK.mkdir(parents=True, exist_ok=True)

    def layer(name, c_in, c_out, count, kind):
        # a Linear IS a 1x1 convolution: C=in, M=out, spatial P=seq, Q=1
        return dict(name=name, C=int(c_in), M=int(c_out), R=1, S=1,
                    P=int(seq), Q=1, Wstride=1, Hstride=1, count=int(count), kind=kind)

    out = {}
    for name, cfg in TRANSFORMER_MODELS.items():
        layers = [layer(nm, ci, co, cfg["layers"], "block")
                  for nm, ci, co in _block_matmuls(cfg)]
        if include_lm_head:
            layers.append(layer("lm_head", cfg["d"], cfg["vocab"], 1, "head"))
        if include_embedding:
            # a lookup, not a matmul: modelling it as vocab*d hugely overcounts
            layers.append(layer("embedding", cfg["d"], cfg["vocab"], 1, "embed"))
        out[name] = layers
        total = sum(l["C"] * l["M"] * l["count"] for l in layers)
        print(f"{name:12s} family={cfg['family']:5s} blocks={cfg['layers']:2d} "
              f"matmuls/block={len(_block_matmuls(cfg))}  weights ~ {total / 1e6:6.1f} M")

    TRANSFORMER_LAYERS.write_text(json.dumps(
        dict(_meta=dict(SEQ=seq, include_lm_head=include_lm_head,
                        include_embedding=include_embedding),
             models=out), indent=1))
    print(f"\nSEQ={seq} lm_head={include_lm_head} embedding={include_embedding}")
    print(f"Saved {len(out)} models -> {TRANSFORMER_LAYERS}")
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    what = (argv[0] if argv else "models").lower()
    if what in ("models", "cnn"):
        generate_models(input_hw=int(os.environ.get("ECC_INPUT_HW", 224)),
                        only=argv[1:] or None)
    elif what in ("transformers", "llm"):
        generate_transformers(
            seq=int(os.environ.get("ECC_SEQ", 1)),
            include_lm_head=os.environ.get("ECC_INCLUDE_LM_HEAD", "1") not in ("0", "false"),
            include_embedding=os.environ.get("ECC_INCLUDE_EMBEDDING", "0") in ("1", "true"))
    else:
        raise guards.refusal("generate-usage",
            "usage: python3 -m eccenergy.generate [models|transformers]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
