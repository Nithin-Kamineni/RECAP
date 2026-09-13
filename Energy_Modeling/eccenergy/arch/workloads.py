"""Workload shapes: CNN conv/linear layers and transformer weight matmuls.

Both workloads reduce to the same thing -- a list of `Layer`s, each a Timeloop
CNN problem shape plus a repeat `count`. A transformer Linear is a 1x1
convolution (C=in, M=out, P=SEQ, Q=1), and its `count` is the block count, so
one mapping is amortised over every identical block.

GROUPED AND DEPTHWISE CONVOLUTIONS carry a group count `G`. The generator
records a PyTorch Conv2d with `groups=g` as C = in_channels/g (the per-group
input channels), M = out_channels (ALL of them) and `groups=g`. Until
2026-09-06 the loader dropped `groups`, so a MobileNetV2 depthwise layer
(C=1, M=960, groups=960) was handed to Timeloop as an ordinary convolution with
ONE input channel feeding 960 filters. MACs and weight counts happened to be
right; the input tensor was 960x too small (81 scalars instead of 77,760 for a
9x9 map), so input traffic, input reuse and therefore every mapping and energy
number for a depthwise layer were wrong. Every model with depthwise layers
(mobilenet_v2, efficientnet_b0, convnext_tiny, xception) was affected;
resnet* were not. `Layer.G` is now the group count, `M` is PER GROUP, and
`timeloop.problem_path()` emits a grouped problem shape whenever G > 1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..paths import CNN_LAYERS, TRANSFORMER_LAYERS, require


@dataclass(frozen=True)
class Layer:
    name: str
    C: int
    M: int
    R: int
    S: int
    P: int
    Q: int
    Wstride: int
    Hstride: int
    count: int = 1
    #: Convolution groups. C and M are PER GROUP: a depthwise 3x3 over 960
    #: channels is C=1, M=1, G=960. G=1 is an ordinary convolution and keeps
    #: the historical shape name, so every existing cache entry stays valid.
    G: int = 1

    @property
    def sig(self):
        """Mapping-identical shapes share a signature, hence a cache entry."""
        return (self.C, self.M, self.R, self.S, self.P, self.Q, self.Wstride,
                self.Hstride, self.G)

    @property
    def shape_name(self):
        base = (f"C{self.C}_M{self.M}_R{self.R}_S{self.S}"
                f"_P{self.P}_Q{self.Q}_ws{self.Wstride}_hs{self.Hstride}")
        # Only a grouped shape carries the suffix: an ungrouped layer must keep
        # the name its mappings were cached under.
        return base if self.G == 1 else f"{base}_G{self.G}"

    @property
    def weights(self):
        return self.G * self.C * self.M * self.R * self.S * self.count

    @property
    def macs(self):
        """Multiply-accumulates for one inference (N=1), all repeats included."""
        return self.G * self.C * self.M * self.R * self.S * self.P * self.Q * self.count

    @property
    def is_grouped(self):
        return self.G > 1


def _layer(d, fallback_name):
    name = str(d.get("name", fallback_name))
    groups = int(d.get("groups", 1) or 1)
    m_total = int(d["M"])
    if groups > 1:
        # The generator stores C per group and M in total (see the module
        # docstring). Timeloop wants both per group.
        if m_total % groups:
            raise SystemExit(
                f"{name}: out_channels={m_total} is not divisible by "
                f"groups={groups}; the workload file is inconsistent")
        m_total //= groups
    return Layer(
        name=name,
        C=int(d["C"]), M=m_total, R=int(d["R"]), S=int(d["S"]),
        P=int(d["P"]), Q=int(d["Q"]),
        Wstride=int(d.get("Wstride", 1)), Hstride=int(d.get("Hstride", 1)),
        count=int(d.get("count", 1)), G=groups)


def load_workload(cfg):
    """Return `{model_name: [Layer, ...]}` for the configured workload."""
    if cfg.workload == "transformer":
        path = require(TRANSFORMER_LAYERS,
                       "run: python3 -m eccenergy.generate transformers")
        blob = json.loads(path.read_text())
        models = blob.get("models", blob)
        meta = blob.get("_meta", {})
        out = {name: [_layer(d, f"{name}_{i}") for i, d in enumerate(layers)]
               for name, layers in models.items()}
        return out, meta

    path = require(CNN_LAYERS, "run: python3 -m eccenergy.generate models")
    blob = json.loads(path.read_text())
    out = {name: [_layer(d, f"{name}_{i}") for i, d in enumerate(layers)]
           for name, layers in blob.items()}
    return out, {}


def select(workload, wanted):
    """Keep only the requested models, in the requested order, and say what is missing."""
    have, missing = {}, []
    for m in wanted:
        if m in workload:
            have[m] = workload[m]
        else:
            missing.append(m)
    if missing:
        print(f"  [skip] not in the workload file: {', '.join(missing)}")
        print(f"         available: {', '.join(sorted(workload))}")
    if not have:
        raise SystemExit("none of the requested models exist in the workload file")
    return have


def select_layers(models, wanted, verbose=True):
    """Keep only the named layers, by their STABLE workload names.

    THE DEVELOPMENT MODE THE PLAN REQUIRES. A one- or two-layer run turns a
    cold sweep from hours into a couple of minutes, which is what makes it
    possible to change the model and re-check it in the same sitting. The rule
    that makes it safe is that a layer is chosen by NAME -- `layer4.1.conv2`,
    the name torchvision gives the module -- and never by index. An index moves
    when the workload file is regenerated; a name does not, so a development
    result stays comparable across regenerations and can be quoted.

    `wanted` empty means the full model, unchanged.

    A name that does not exist is an ERROR, not a silent skip: a typo that
    quietly produced a one-layer run labelled as a two-layer run is exactly the
    confusion the results namespace exists to prevent.
    """
    if not wanted:
        return models

    out = {}
    for model, layers in models.items():
        by_name = {l.name: l for l in layers}
        missing = [w for w in wanted if w not in by_name]
        if missing:
            available = ", ".join(l.name for l in layers)
            raise SystemExit(
                "ECC_LAYERS names layers that are not in {0}: {1}\n"
                "  -> layer selection is by the STABLE layer name, not by index.\n"
                "  -> available in {0}: {2}".format(model, ", ".join(missing), available))
        # requested order, not file order, so a two-layer selection is
        # reproducible whichever way it was typed
        kept = [by_name[w] for w in wanted]
        out[model] = kept
        if verbose:
            total = sum(l.weights for l in layers)
            picked = sum(l.weights for l in kept)
            share = (picked / total * 100.0) if total else 0.0
            print("  [layers] {0}: {1} of {2} layers  ({3:,} of {4:,} weights, "
                  "{5:.2f}% of the model)".format(
                      model, len(kept), len(layers), picked, total, share))
            for l in kept:
                print("           {0:24s} {1}  weights={2:,}".format(
                    l.name, l.shape_name, l.weights))
    return out


def layer_identities(models):
    """`{model: [{name, shape, dims, weights, count}]}` for the result JSON.

    The plan asks for layer identity to be recorded, not just the count, so
    that a development number can be traced back to exactly which layers
    produced it.
    """
    out = {}
    for model, layers in models.items():
        out[model] = [{
            "name": l.name,
            "shape_signature": l.shape_name,
            "dims": {"C": l.C, "M": l.M, "R": l.R, "S": l.S, "P": l.P, "Q": l.Q,
                     "Wstride": l.Wstride, "Hstride": l.Hstride, "N": 1, "G": l.G},
            "dims_note": ("C and M are per group; G is the group count "
                          "(G=1: ordinary convolution)"),
            "grouped": l.is_grouped,
            "repeat_count": l.count,
            "weights": l.weights,
            "macs": l.macs,
        } for l in layers]
    return out
