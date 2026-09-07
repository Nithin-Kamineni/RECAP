"""
Extract conv/linear layer shapes for several CNNs via forward hooks.
Run INSIDE the timeloop-accelergy-pytorch container (torch + torchvision present).
xception needs timm: pip install timm  (if it fails, xception is skipped).
Saves ecc_energy_study/model_layers.json for run_ecc_multimodel.py.
"""
import json, pathlib, traceback
import torch

WORK = pathlib.Path.cwd() / "ecc_energy_study"; WORK.mkdir(exist_ok=True)
INPUT_HW = 224   # same input resolution for all models (fair comparison)

MODELS = {   # name             (source,        torch/timm constructor)
    "resnet18":        ("torchvision", "resnet18"),
    "resnet50":        ("torchvision", "resnet50"),
    "densenet121":     ("torchvision", "densenet121"),
    "squeezenet1_1":   ("torchvision", "squeezenet1_1"),
    "mobilenet_v2":    ("torchvision", "mobilenet_v2"),
    "efficientnet_b0": ("torchvision", "efficientnet_b0"),
    "convnext_tiny":   ("torchvision", "convnext_tiny"),
    "xception":        ("timm",        "xception"),
}

def load_model(source, ctor):
    if source == "torchvision":
        import torchvision.models as tvm
        return getattr(tvm, ctor)(weights=None)
    import timm
    return timm.create_model(ctor, pretrained=False)

def extract_layers(model, hw):
    layers = []
    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(module, torch.nn.Conv2d):
                g = module.groups
                layers.append(dict(
                    name=name, type="conv",
                    C=max(1, module.in_channels // g),   # per-group in-channels (exact weight count)
                    M=module.out_channels,
                    R=module.kernel_size[0], S=module.kernel_size[1],
                    P=int(out.shape[2]), Q=int(out.shape[3]),
                    Wstride=module.stride[1], Hstride=module.stride[0], groups=g))
            elif isinstance(module, torch.nn.Linear):
                layers.append(dict(
                    name=name, type="fc",
                    C=module.in_features, M=module.out_features,
                    R=1, S=1, P=1, Q=1, Wstride=1, Hstride=1, groups=1))
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

all_layers = {}
for name, (source, ctor) in MODELS.items():
    try:
        print(f"[extract] {name} ({source}) ...", flush=True)
        m = load_model(source, ctor)
        layers = extract_layers(m, INPUT_HW)
        all_layers[name] = layers
        print(f"  -> {len(layers)} conv/linear layers")
    except Exception as e:
        print(f"  !! FAILED {name}: {e}")
        traceback.print_exc()

out = WORK / "model_layers.json"
out.write_text(json.dumps(all_layers, indent=1))
print(f"\nSaved {len(all_layers)} models to {out}")
print("Models:", list(all_layers.keys()))