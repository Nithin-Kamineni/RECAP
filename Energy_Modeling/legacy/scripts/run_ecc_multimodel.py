"""
Multi-model ECC energy sweep on simple_weight_stationary.
Reads ecc_energy_study/model_layers.json (from generate_models.py).
Produces 3 combined journal-grade figures:
  1. decode-at-controller : grouped Baseline BCH vs Embedded ECC per model
  2. decode-on-use        : grouped baseline vs embedded (SRAM protected) per model
  3. saving %             : one bar per model, decode-at-controller energy reduction
Run INSIDE the container.
"""
import os, re, glob, json, math, pathlib, subprocess, shutil, contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

try:
    import pytimeloop.timeloopfe.v4 as tl
except ImportError:
    import timeloopfe.v4 as tl

# ============ USER KNOBS (locked identical across all models) ================
ARCH             = "simple_weight_stationary"
BCH_N, BCH_K     = 63, 36
BCH_T            = 5
WEIGHT_BITS      = 8
EMB_WEIGHTS_PER_CW = 8
VICTORY          = 500
REUSE_FACTOR     = 5.0
E_DECODE_CTRL_PJ = 200
E_DECODE_USE_PJ  = 100
# Ordered easy -> hard. Trim for a quick first pass (e.g. keep just the first two).
MODELS_TO_RUN = ["resnet18", "resnet50", "densenet121", "squeezenet1_1","mobilenet_v2", "efficientnet_b0", "convnext_tiny", "xception"]
# MODELS_TO_RUN = ["resnet50", "squeezenet1_1"]
# =============================================================================

WEIGHTS_PER_CW_BASE = BCH_K / WEIGHT_BITS
WEIGHTS_PER_CW_EMB  = EMB_WEIGHTS_PER_CW
parity_frac = BCH_N / BCH_K - 1.0

assert shutil.which("timeloop-mapper"), "timeloop-mapper not on PATH -- run inside the container"
WORK = pathlib.Path.cwd() / "ecc_energy_study"; WORK.mkdir(exist_ok=True)
EX_REPO = WORK / "timeloop-accelergy-exercises"
assert EX_REPO.exists(), "exercises repo missing -- run a single-model script once first to clone it"
DESIGNS_DIR = EX_REPO / "workspace" / "example_designs" / "example_designs"
model_layers = json.loads((WORK / "model_layers.json").read_text())

(WORK / "globals.yaml").write_text(
    'variables:\n  version: 0.4\n  global_cycle_seconds: 1e-9\n  technology: "65nm"\n')

def patched_arch_path():
    src = DESIGNS_DIR / ARCH / "arch.yaml"
    dst = WORK / f"arch_{ARCH}_patched.yaml"
    text = src.read_text()
    dram_attrs = text.split("name: DRAM")[1].split("!")[0] if "name: DRAM" in text else ""
    if "depth" not in dram_attrs:
        text = text.replace(
            'class: DRAM\n    attributes:\n      type: "LPDDR4"',
            'class: DRAM\n    attributes:\n      depth: 1048576\n      type: "LPDDR4"')
    dst.write_text(text)
    return dst
ARCH_YAML = patched_arch_path()

PROBLEM_TEMPLATE = """problem:
  version: 0.4
  shape:
    name: cnn_layer
    coefficients:
      - name: Wstride
        default: 1
      - name: Hstride
        default: 1
      - name: Wdilation
        default: 1
      - name: Hdilation
        default: 1
    dimensions: [ C, M, R, S, N, P, Q ]
    data_spaces:
      - name: Weights
        projection:
          - [ [C] ]
          - [ [M] ]
          - [ [R] ]
          - [ [S] ]
      - name: Inputs
        projection:
          - [ [N] ]
          - [ [C] ]
          - [ [R, Wdilation], [P, Wstride] ]
          - [ [S, Hdilation], [Q, Hstride] ]
      - name: Outputs
        projection:
          - [ [N] ]
          - [ [M] ]
          - [ [Q] ]
          - [ [P] ]
        read_write: True
  instance:
    C: {C}
    M: {M}
    R: {R}
    S: {S}
    N: {N}
    P: {P}
    Q: {Q}
    Wstride: {Wstride}
    Hstride: {Hstride}
"""
PROB_DIR = WORK / "problems_multimodel"; PROB_DIR.mkdir(exist_ok=True)

def make_problem(sig):
    C, M, R, S, P, Q, Ws, Hs = sig
    name = f"C{C}_M{M}_R{R}_S{S}_P{P}_Q{Q}_ws{Ws}_hs{Hs}"
    path = PROB_DIR / f"{name}.yaml"
    if not path.exists():
        path.write_text(PROBLEM_TEMPLATE.format(C=C, M=M, R=R, S=S, N=1, P=P, Q=Q,
                                                Wstride=Ws, Hstride=Hs))
    return name, path

def design_inputs(problem_yaml):
    files = [str(ARCH_YAML)]
    files += sorted(glob.glob(str(DESIGNS_DIR / "_components" / "*.yaml")))
    files.append(str(DESIGNS_DIR / "_include" / "mapper.yaml"))
    files.append(str(WORK / "globals.yaml"))
    files.append(str(problem_yaml))
    return files

OUT_ROOT = WORK / "outputs" / ARCH / "multimodel"; OUT_ROOT.mkdir(parents=True, exist_ok=True)
_sig_cache = {}
def run_shape(sig):
    if sig in _sig_cache:
        return _sig_cache[sig]
    name, prob = make_problem(sig)
    out_dir = OUT_ROOT / name
    stats = out_dir / "timeloop-mapper.stats.txt"
    if stats.exists():
        _sig_cache[sig] = stats
        return stats
    spec = tl.Specification.from_yaml_files(*design_inputs(prob))
    spec.mapper.num_threads = os.cpu_count() or 4
    spec.mapper.victory_condition = VICTORY
    # spec.mapper.search_size = 0
    spec.mapper.timeout = 10000
    spec.mapper.algorithm = "hybrid"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_dir / "mapper_console.log", "w") as _logf, \
             contextlib.redirect_stdout(_logf), contextlib.redirect_stderr(_logf):
            tl.call_mapper(spec, output_dir=str(out_dir))
    except Exception as e:
        _sig_cache[sig] = None
        return None
        
    _sig_cache[sig] = stats if stats.exists() else None
    return _sig_cache[sig]

def _grab(pat, text):
    m = re.search(pat, text)
    return float(m.group(1)) if m else None

def parse_stats(stats_path, layer):
    text = pathlib.Path(stats_path).read_text()
    rows, chunks = [], re.split(r"===\s*(.+?)\s*===", text)[1:]
    for level, body in zip(chunks[0::2], chunks[1::2]):
        parts = re.split(r"\n\s+(Weights|Inputs|Outputs)\s*:\s*\n", body)
        if len(parts) > 1:
            for ds, sec in zip(parts[1::2], parts[2::2]):
                rows.append(dict(layer=layer, level=level, dataspace=ds,
                    reads=_grab(r"Scalar reads \(per-instance\)\s*:\s*([\d.eE+]+)", sec),
                    energy_pJ=_grab(r"Energy \(total\)\s*:\s*([\d.eE+]+)\s*pJ", sec)))
        else:
            e = _grab(r"Energy \(total\)\s*:\s*([\d.eE+]+)\s*pJ", body)
            if e is not None:
                rows.append(dict(layer=layer, level=level, dataspace="Compute",
                                 reads=None, energy_pJ=e))
    return rows

def classify(level):
    l = level.lower()
    if "dram" in l:                                return "DRAM"
    if "glb" in l or "buffer" in l or "sram" in l: return "Global buffer"
    if "mac" in l or "compute" in l:               return "Compute"
    return "Local (spads/RF/NoC)"

cats = ["DRAM", "Global buffer", "Local (spads/RF/NoC)", "Compute"]

def run_model(model):
    rows, n_ok, n_skip = [], 0, 0
    n = len(model_layers[model])
    for i, L in enumerate(model_layers[model]):
        sig = (int(L["C"]), int(L["M"]), int(L["R"]), int(L["S"]),
               int(L["P"]), int(L["Q"]), int(L["Wstride"]), int(L["Hstride"]))
        stats = run_shape(sig)
        status = "ok" if stats is not None else "skip"
        print(f"  {model}: layer {i+1}/{n} [{status}]", flush=True)
        if stats is None:
            n_skip += 1
            continue
        rows += parse_stats(stats, f"{model}_{i}")
        n_ok += 1
        
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df[df.energy_pJ.notna() & (df.energy_pJ > 0)].reset_index(drop=True)
    df["category"] = df.level.map(classify)
    total = df.energy_pJ.sum()
    wt = df[df.dataspace == "Weights"].groupby("category").energy_pJ.sum().reindex(cats, fill_value=0.0)
    E_DRAM_W   = wt["DRAM"]
    E_ONCHIP_W = wt.drop("DRAM").sum()
    DRAM_W_READS = df[(df.category=="DRAM") & (df.dataspace=="Weights")].reads.sum()
    cw_ctrl_b = DRAM_W_READS / WEIGHTS_PER_CW_BASE
    cw_ctrl_e = DRAM_W_READS / WEIGHTS_PER_CW_EMB
    cw_use_b  = cw_ctrl_b * REUSE_FACTOR
    cw_use_e  = cw_ctrl_e * REUSE_FACTOR
    cfg = {
        "baseline_bch":        total + E_DRAM_W*parity_frac              + cw_ctrl_b*E_DECODE_CTRL_PJ,
        "embedded_ecc":        total                                    + cw_ctrl_e*E_DECODE_CTRL_PJ,
        "baseline_bch_onchip": total + (E_DRAM_W+E_ONCHIP_W)*parity_frac + cw_use_b*E_DECODE_USE_PJ,
        "embedded_ecc_onchip": total                                    + cw_use_e*E_DECODE_USE_PJ,
    }
    cfg["_layers_ok"], cfg["_layers_skip"] = n_ok, n_skip
    return cfg

results = {}
for model in MODELS_TO_RUN:
    if model not in model_layers:
        print(f"[skip] {model}: not in model_layers.json")
        continue
    print(f"\n=== {model} ({len(model_layers[model])} layers) ===", flush=True)
    cfg = run_model(model)
    if cfg is None:
        print(f"  !! {model}: no valid layers, skipped")
        continue
    results[model] = cfg
    s1 = (cfg["baseline_bch"]-cfg["embedded_ecc"])/cfg["baseline_bch"]*100
    s2 = (cfg["baseline_bch_onchip"]-cfg["embedded_ecc_onchip"])/cfg["baseline_bch_onchip"]*100
    print(f"  {model}: ctrl saving {s1:.1f}% | on-use saving {s2:.1f}%"
          f"  ({cfg['_layers_ok']} layers, {cfg['_layers_skip']} skipped)")

print(f"\nModels with results: {list(results.keys())}")
if not results:
    raise SystemExit("No models produced results; nothing to plot.")

pd.DataFrame({m: {k: v for k, v in c.items() if not k.startswith('_')}
              for m, c in results.items()}).T.div(1e6).to_csv(WORK / "multimodel_summary_uJ.csv")

# ============================ PLOTS ==========================================
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "axes.linewidth": 1.6,
    "axes.edgecolor": "#2b2b2b", "pdf.fonttype": 42, "ps.fonttype": 42,
})
models = list(results.keys())

def grouped_fig(base_key, emb_key, title, fname):
    b = [results[m][base_key]/1e6 for m in models]   # uJ
    e = [results[m][emb_key]/1e6 for m in models]
    unit = "\u00b5J"
    if max(b) >= 1e3:
        b = [v/1e3 for v in b]; e = [v/1e3 for v in e]; unit = "mJ"
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.75*len(models)+3, 7.6))
    ax.bar(x - w/2, b, w, label="Baseline BCH", color="#2E5A87",
           edgecolor="white", linewidth=1.2, zorder=3)
    ax.bar(x + w/2, e, w, label="Embedded ECC", color="#E1A730",
           edgecolor="white", linewidth=1.2, zorder=3)
    ymax = max(max(b), max(e))
    for i in range(len(models)):
        sv = (b[i]-e[i])/b[i]*100
        ax.text(x[i], max(b[i], e[i]) + ymax*0.015, f"\u2212{sv:.0f}%",
                ha="center", va="bottom", fontsize=17, fontweight="bold", color="#B03A2E")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=16, rotation=20, ha="right")
    ax.set_ylabel(f"Inference energy ({unit})", fontsize=22, labelpad=10)
    ax.set_ylim(0, ymax*1.15)
    ax.tick_params(axis="y", labelsize=16, width=1.6, length=7)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.grid(axis="y", ls=":", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=17, handlelength=1.2)
    ax.set_title(title, fontsize=21, pad=16, fontweight="medium")
    plt.tight_layout()
    plt.savefig(WORK / f"{fname}.png", dpi=400, bbox_inches="tight")
    plt.savefig(WORK / f"{fname}.pdf", bbox_inches="tight")
    plt.close()

def saving_fig(fname):
    sv = {m: (results[m]["baseline_bch"]-results[m]["embedded_ecc"])/results[m]["baseline_bch"]*100
          for m in models}
    order = sorted(models, key=lambda m: sv[m], reverse=True)
    vals = [sv[m] for m in order]
    fig, ax = plt.subplots(figsize=(1.6*len(order)+3, 7.6))
    ax.bar(range(len(order)), vals, 0.62, color="#4E9F3D",
           edgecolor="white", linewidth=1.2, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals)*0.015, f"{v:.0f}%", ha="center", va="bottom",
                fontsize=19, fontweight="bold", color="#2b2b2b")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=16, rotation=20, ha="right")
    ax.set_ylabel("Energy saving (%)", fontsize=22, labelpad=10)
    ax.set_ylim(0, max(vals)*1.15)
    ax.tick_params(axis="y", labelsize=16, width=1.6, length=7)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.grid(axis="y", ls=":", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title(f"Embedded ECC energy saving  \u00b7  BCH({BCH_N},{BCH_K})  \u00b7  decode-at-controller",
                 fontsize=20, pad=16, fontweight="medium")
    plt.tight_layout()
    plt.savefig(WORK / f"{fname}.png", dpi=400, bbox_inches="tight")
    plt.savefig(WORK / f"{fname}.pdf", bbox_inches="tight")
    plt.close()

grouped_fig("baseline_bch", "embedded_ecc",
            f"Decode-at-controller  \u00b7  BCH({BCH_N},{BCH_K})", "multimodel_decode_at_controller")
grouped_fig("baseline_bch_onchip", "embedded_ecc_onchip",
            f"Decode-on-use  \u00b7  BCH({BCH_N},{BCH_K})", "multimodel_decode_on_use")
saving_fig("multimodel_saving_percent")

print("\nSaved figures (PNG 400dpi + vector PDF):")
print("  multimodel_decode_at_controller.[png|pdf]")
print("  multimodel_decode_on_use.[png|pdf]")
print("  multimodel_saving_percent.[png|pdf]")
print("  multimodel_summary_uJ.csv")