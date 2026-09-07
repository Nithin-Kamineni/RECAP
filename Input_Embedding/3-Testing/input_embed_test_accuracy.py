"""
input_embed_test_accuracy.py — RECAP 3-Testing main script.

Evaluates Top-1/Top-5 ImageNet accuracy of a QAT int8 model when every
Conv2d/Linear layer's INPUT activation (the initial sensor input and every
intermediate MAC output) is SECDED-embedded — LSBs overwritten with parity
computed from the top-weighted bits of the same value, exactly as
ECC-CODE-Engine's ParityOverwriteByTopWeightsEncode does for weights
(implementations/SECDEDInputEmbed.py + secded_vectorized.py). Weights are
left at their original (dequantized, unembedded) precision in the MAC.

Three runs are produced:
  1. "baseline"     — int8-quantized activations, NO embedding (upper bound).
  2. "input_embed"  — SECDED embedding on every layer.
  3. "input_embed_fallback" (only if --auto-fallback and run 2 drops accuracy
     by more than --acc-drop-tolerance-pp vs run 1) — layers are exempted
     from embedding ("regular ECC": protected losslessly instead of via LSB
     overwrite), most-impactful first, until accuracy recovers or the skip
     budget (--max-skip-fraction) is exhausted. Layers are ranked by
     ablation sensitivity — the subset Top-1 actually recovered by exempting
     that single layer alone — rather than by how much the embedding
     numerically perturbs the layer's own values, which turned out to
     correlate poorly with real accuracy impact. The search runs on a fast
     stratified subset and tracks the best skip-set seen along its whole
     trajectory (starting from the "skip nothing" floor), so it can never
     hand back something worse than doing no fallback at all. The reported
     accuracy for the chosen skip-list is always a full-validation-set
     re-evaluation.

Usage (see run.sh for the SLURM wrapper):
    python3 input_embed_test_accuracy.py --arch resnet18 --qmode QAT \
        --imagenet-root /path/to/imagenet-val --models-dir .../models \
        --results-dir .../accuracy_results
"""
import os, sys, json, argparse, collections, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.eval_functions import (
    pick_device, get_dataloaders, build_model, evaluate,
)
from activation_embedding import ActivationEmbedder, calibrate


def ckpt_path(models_dir, dataset, arch, qmode, bits):
    ds = dataset.lower()
    if qmode == "QAT":
        return os.path.join(models_dir, ds, arch, "QAT", f"model_int{bits}_qat.pth")
    return os.path.join(models_dir, ds, arch, "PTQ", f"model_int{bits}_ptq.pth")


def load_checkpoint_into_model(model, path, qmode, device):
    payload = torch.load(path, map_location=device)
    if qmode == "QAT":
        sd = payload["state_dict"]
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        return payload.get("meta", {})
    # PTQ: qstate_dict + per-channel/per-tensor scales -> dequantize into model.
    qsd, scales = payload["qstate_dict"], payload["meta"]["scales"]
    dsd = {}
    for k, v in qsd.items():
        sinfo = scales.get(k, None)
        if sinfo is None:
            dsd[k] = v
        elif sinfo["type"] == "per_tensor":
            dsd[k] = v.to(torch.float32) * float(sinfo["scale"])
        else:
            s = sinfo["scales"]
            while s.ndim < v.ndim:
                s = s.unsqueeze(-1)
            dsd[k] = v.to(torch.float32) * s
    model.load_state_dict(dsd, strict=True)
    return payload["meta"]


def build_stratified_subset_loader(dataset, images_per_class, batch_size, workers):
    by_class = collections.defaultdict(list)
    for i, (_, y) in enumerate(dataset.samples):
        by_class[y].append(i)
    idx = []
    for y, ii in sorted(by_class.items()):
        idx.extend(sorted(ii)[:images_per_class])
    return DataLoader(Subset(dataset, idx), batch_size=batch_size, shuffle=False,
                       num_workers=workers, pin_memory=True)


def fmt_duration(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def run_eval(model, loader, device, embedder: ActivationEmbedder, enabled, skip_layers):
    embedder.enabled = enabled
    embedder.skip_layers = set(skip_layers or [])
    embedder.mode = "run"
    if enabled:
        embedder.reset_distortion_tracking()
    top1, top5 = evaluate(model, loader, device)
    return top1, top5


def compute_ablation_sensitivity(model, subset_loader, device, embedder, layer_names):
    """For each layer, measure the subset Top-1 *recovered* by exempting
    that single layer alone (all other layers stay embedded), relative to
    the fully-embedded subset Top-1. This ranks layers by their actual
    causal effect on accuracy rather than by how much the embedding
    numerically perturbs that layer's own values — a layer can have large
    raw distortion but little accuracy impact (e.g. deep, high-dimensional
    layers where noise partly averages out), or small raw distortion but
    outsized impact (e.g. the stem, whose corruption compounds through
    every downstream layer). Returns (sensitivity_dict, all_embedded_top1)."""
    all_embedded_top1, _ = run_eval(model, subset_loader, device, embedder,
                                     enabled=True, skip_layers=[])
    print(f"  [ablation] all-embedded subset_top1={all_embedded_top1:.2f}% (reference)")
    sensitivity = {}
    for n in layer_names:
        top1, _ = run_eval(model, subset_loader, device, embedder,
                            enabled=True, skip_layers=[n])
        sensitivity[n] = top1 - all_embedded_top1
        print(f"  [ablation] skip={n!r}  subset_top1={top1:.2f}%  "
              f"gain={sensitivity[n]:+.2f}pp")
    return sensitivity, all_embedded_top1


def greedy_forward_search(model, subset_loader, device, embedder, ranked,
                            baseline_top1, tolerance_pp, max_skip_fraction, floor_top1):
    """Greedily add layers from `ranked` (most-important-to-skip first,
    per ablation sensitivity) to the skip-list, re-checking on
    `subset_loader` until accuracy recovers or the skip budget
    (max_skip_fraction of all layers) runs out.

    Tracks the best skip-set seen anywhere along that trajectory (by subset
    Top-1), starting from `floor_top1` — the "do nothing" (skip=[])
    baseline — rather than just returning wherever the walk happens to end.
    Layer sensitivities don't compose additively once layers interact
    through BatchNorm/ReLU nonlinearities: adding a candidate that looked
    good in isolation can make the *combination* worse than an earlier,
    smaller skip-set, or even worse than skipping nothing at all. Without
    this safeguard the search could hand back a skip-list that underperforms
    doing no fallback whatsoever. Returns (skip, subset_top1, met_target,
    budget_exhausted) — met_target/budget_exhausted make the two possible
    exit reasons explicit instead of leaving them to be inferred from
    skip_fraction after the fact."""
    target = baseline_top1 - tolerance_pp
    max_skip = max(1, int(round(max_skip_fraction * len(ranked))))

    best_skip, best_top1 = [], floor_top1
    print(f"  [fallback-search] floor (no skip)  subset_top1={floor_top1:.2f}%  "
          f"target>={target:.2f}%")

    skip = []
    met_target = best_top1 >= target
    i = 0
    while not met_target and len(skip) < max_skip and i < len(ranked):
        n = ranked[i]
        i += 1
        skip = skip + [n]
        top1, _ = run_eval(model, subset_loader, device, embedder,
                            enabled=True, skip_layers=skip)
        print(f"  [fallback-search] +{n!r}  "
              f"(total skipped={len(skip)}/{len(ranked)})  subset_top1={top1:.2f}%  "
              f"target>={target:.2f}%")
        if top1 > best_top1:
            best_top1, best_skip = top1, list(skip)
        met_target = best_top1 >= target

    budget_exhausted = not met_target
    return best_skip, best_top1, met_target, budget_exhausted


def main():
    ap = argparse.ArgumentParser(description="RECAP input-embedding accuracy evaluation")
    ap.add_argument("--dataset", default="IMAGENET", choices=["IMAGENET"])
    ap.add_argument("--arch", required=True,
                    choices=["resnet18", "mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--quant-bits", type=int, default=8)
    ap.add_argument("--qmode", default="QAT", choices=["QAT", "PTQ"])
    ap.add_argument("--imagenet-root", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--calib-batches", type=int, default=20)
    ap.add_argument("--num-threads", type=int, default=None,
                    help="SECDED encoder thread count (default: min(8, cpu_count)).")
    ap.add_argument("--skip-layers", default="",
                    help="Comma-separated layer names to exempt from embedding "
                         "('regular ECC' — no LSB distortion) up front.")
    ap.add_argument("--no-baseline", action="store_true",
                    help="Skip the un-embedded int8 baseline pass.")
    ap.add_argument("--auto-fallback", action="store_true",
                    help="If embedding drops top-1 more than --acc-drop-tolerance-pp "
                         "vs baseline, search for a per-layer skip-list (ranked by "
                         "ablation sensitivity) that recovers it.")
    ap.add_argument("--acc-drop-tolerance-pp", type=float, default=2.0)
    ap.add_argument("--max-skip-fraction", type=float, default=0.3)
    ap.add_argument("--fallback-subset-per-class", type=int, default=5)
    args = ap.parse_args()

    run_start = time.time()
    timing = {}

    device = pick_device("cuda", local_rank=0)
    print(f"[input_embed_test_accuracy] device={device}  arch={args.arch}  "
          f"qmode={args.qmode}  bits={args.quant_bits}")

    _, test_loader, nc, _ = get_dataloaders(
        args.dataset, data_root="./data", batch_size=args.batch_size,
        num_workers=args.workers, dist_mode=False,
        imagenet_root=args.imagenet_root, arch=args.arch,
    )
    print(f"[data] val images = {len(test_loader.dataset)}  classes = {nc}")

    model = build_model(args.arch, nc, use_pretrained=False).to(device)
    ck = ckpt_path(args.models_dir, args.dataset, args.arch, args.qmode, args.quant_bits)
    if not os.path.exists(ck):
        print(f"[skip] checkpoint not found: {ck}")
        return
    meta = load_checkpoint_into_model(model, ck, args.qmode, device)
    if isinstance(meta, dict):
        meta_summary = {k: v for k, v in meta.items() if k != "scales"}
        if "scales" in meta:
            meta_summary["scales"] = f"<{len(meta['scales'])} per-layer tensors omitted>"
    else:
        meta_summary = "<none>"
    print(f"[weights] loaded {ck}  meta={meta_summary}")

    embedder = ActivationEmbedder(num_threads=args.num_threads).attach(model)
    print(f"[embed] hooked {len(embedder.layer_names())} Conv2d/Linear layers")

    print(f"[calibrate] running {args.calib_batches} batches to fix per-layer "
          f"int8 activation scales ...")
    t0 = time.time()
    calibrate(embedder, model, test_loader, device, args.calib_batches)
    timing["calibration_sec"] = round(time.time() - t0, 2)

    results = {
        "dataset": args.dataset, "arch": args.arch, "qmode": args.qmode,
        "bits": args.quant_bits, "n_layers": len(embedder.layer_names()),
    }

    baseline_top1 = None
    if not args.no_baseline:
        print("[eval] baseline (int8 quantized, NO embedding) ...")
        t0 = time.time()
        top1, top5 = run_eval(model, test_loader, device, embedder,
                               enabled=False, skip_layers=[])
        timing["baseline_sec"] = round(time.time() - t0, 2)
        print(f"  [baseline] Top-1={top1:.4f}%  Top-5={top5:.4f}%  "
              f"({fmt_duration(timing['baseline_sec'])})")
        results["baseline"] = {"top1": round(top1, 4), "top5": round(top5, 4)}
        baseline_top1 = top1

    up_front_skip = [s for s in args.skip_layers.split(",") if s]
    print(f"[eval] input_embed (SECDED on every layer"
          f"{'' if not up_front_skip else f', except {up_front_skip}'}) ...")
    t0 = time.time()
    top1, top5 = run_eval(model, test_loader, device, embedder,
                           enabled=True, skip_layers=up_front_skip)
    timing["input_embed_sec"] = round(time.time() - t0, 2)
    distortion_report = embedder.distortion_report()
    print(f"  [input_embed] Top-1={top1:.4f}%  Top-5={top5:.4f}%  "
          f"({fmt_duration(timing['input_embed_sec'])})")
    results["input_embed"] = {
        "top1": round(top1, 4), "top5": round(top5, 4),
        "skip_layers": up_front_skip,
        "mean_layer_distortion": {k: round(v, 6) for k, v in distortion_report.items()},
    }

    if args.auto_fallback and baseline_top1 is not None and \
            (baseline_top1 - top1) > args.acc_drop_tolerance_pp:
        print(f"[fallback] accuracy drop {baseline_top1 - top1:.2f}pp exceeds "
              f"tolerance {args.acc_drop_tolerance_pp}pp — searching for a "
              f"skip-list (ablation-ranked) on a fast subset ...")
        subset_loader = build_stratified_subset_loader(
            test_loader.dataset, args.fallback_subset_per_class,
            args.batch_size, args.workers,
        )
        print(f"  [fallback] subset size = {len(subset_loader.dataset)} images")
        t0 = time.time()

        layer_names = embedder.layer_names()
        sensitivity, floor_top1 = compute_ablation_sensitivity(
            model, subset_loader, device, embedder, layer_names)
        ranked = sorted(sensitivity, key=lambda n: -sensitivity[n])

        chosen_skip, subset_top1, met_target, budget_exhausted = greedy_forward_search(
            model, subset_loader, device, embedder, ranked, baseline_top1,
            args.acc_drop_tolerance_pp, args.max_skip_fraction, floor_top1,
        )
        print(f"[fallback] forward search {'met target' if met_target else 'exhausted its skip budget without meeting target'} "
              f"({len(chosen_skip)}/{len(distortion_report)} layers, subset_top1={subset_top1:.2f}%)")

        print(f"[fallback] chosen skip-list ({len(chosen_skip)}/"
              f"{len(distortion_report)} layers), re-evaluating on the FULL val set ...")
        final_top1, final_top5 = run_eval(model, test_loader, device, embedder,
                                            enabled=True, skip_layers=chosen_skip)
        timing["fallback_sec"] = round(time.time() - t0, 2)
        print(f"  [input_embed_fallback] Top-1={final_top1:.4f}%  Top-5={final_top5:.4f}%  "
              f"({fmt_duration(timing['fallback_sec'])})")
        results["input_embed_fallback"] = {
            "top1": round(final_top1, 4), "top5": round(final_top5, 4),
            "skip_layers": chosen_skip,
            "skip_fraction": round(len(chosen_skip) / max(1, len(distortion_report)), 4),
            "subset_search_top1": round(subset_top1, 4) if subset_top1 is not None else None,
            "met_target_on_subset": met_target,
            "budget_exhausted": budget_exhausted,
            "ablation_sensitivity": {k: round(v, 4) for k, v in sensitivity.items()},
        }

    timing["total_sec"] = round(time.time() - run_start, 2)
    timing["total_human"] = fmt_duration(timing["total_sec"])
    results["timing"] = timing
    print(f"[timing] {args.arch}: total={timing['total_human']}  "
          f"(calibration={fmt_duration(timing.get('calibration_sec', 0))}, "
          f"baseline={fmt_duration(timing.get('baseline_sec', 0))}, "
          f"input_embed={fmt_duration(timing.get('input_embed_sec', 0))}, "
          f"fallback={fmt_duration(timing.get('fallback_sec', 0))})")

    out_dir = Path(args.results_dir) / args.dataset.lower() / args.arch / \
        args.qmode / f"{args.quant_bits}-bit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "secded_input_embed_accuracy.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
