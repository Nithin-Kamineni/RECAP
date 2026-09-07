"""
Multi-model CNN ECC energy sweep -- FULL per-level energy breakdown,
with an optional READ/WRITE split for the on-chip storage levels.

Arch: simple_weight_stationary. Reads ecc_energy_study/model_layers.json
(produced by generate_models.py -- edit MODELS in that file to change the set).

For every model it draws TWO stacked bars, Baseline BCH vs Embedded ECC
(decode-at-controller).

  SeperateBreakdown = False  ->  5 segments (as before):
        DRAM, Global buffer, On-chip SRAM/RF, Compute, ECC decode

  SeperateBreakdown = True   ->  7 segments:
        DRAM,
        Global buffer (read), Global buffer (write),
        On-chip SRAM/RF (read), On-chip SRAM/RF (write),
        Compute, ECC decode

  The read/write split apportions each level's Energy(total) by its access
  counts: read-energy = total * reads/(reads+writes), write = remainder,
  where writes = Scalar fills + Scalar updates. This is exact if read and
  write per-access energies are equal; otherwise it is a count-proportional
  approximation (totals and saving % are identical either way).

Reuses the run_ecc_multimodel.py mapper cache (same problem names + output
dir), so if you have already run that sweep this finishes in seconds.

RUN INSIDE the container, launched from Energy_modeling/ :
    cd /home/workspace
    python3 E3-CODE/generate_models.py                     # once
    python3 E3-CODE/run_ecc_multimodel_breakdown_rw.py     # this script
"""
import os, re, glob, json, math, pathlib, shutil, contextlib
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

# ================= DECISION: split on-chip storage into read/write? ===========
SeperateBreakdown = False     # True  -> 7 segments (GB & SRAM each read+write)
                             # False -> 5 segments (original behaviour)
# =============================================================================

# ================= USER KNOBS (locked identical across all models) ===========
ARCH               = "simple_weight_stationary"
BCH_N, BCH_K       = 63, 36
BCH_T              = 5
WEIGHT_BITS        = 8
EMB_WEIGHTS_PER_CW = 8
VICTORY            = 500          # mapper effort; raise to 3000+ for final numbers
REUSE_FACTOR       = 5.0          # (kept for parity with other scripts; unused here)
E_DECODE_CTRL_PJ   = 200          # pJ per codeword, controller decode

# Which models to plot (must exist in model_layers.json). Ordered easy -> hard.
MODELS_TO_RUN = [
                "resnet18",
                "resnet50",
                "densenet121",
                "squeezenet1_1",
                "mobilenet_v2",
                "efficientnet_b0",
                "convnext_tiny",
                "xception"
                ]
# =============================================================================

WEIGHTS_PER_CW_BASE = BCH_K / WEIGHT_BITS      # 36/8 = 4.5
WEIGHTS_PER_CW_EMB  = EMB_WEIGHTS_PER_CW       # 8
parity_frac         = BCH_N / BCH_K - 1.0      # DRAM weight-traffic inflation for baseline

# ---------------- setup: repo + design paths ---------------------------------
assert shutil.which("timeloop-mapper"), "timeloop-mapper not on PATH -- run inside the container"
WORK = pathlib.Path.cwd() / "ecc_energy_study"; WORK.mkdir(exist_ok=True)
# Figures/CSV are saved next to THIS script (i.e. the E3-CODE folder), not in
# the working dir. __file__ resolves the script's own location, so this holds
# no matter which directory you launch from. Falls back to WORK if __file__ is
# somehow undefined (e.g. pasted into a REPL).
try:
    FIG_DIR = pathlib.Path(__file__).resolve().parent
except NameError:
    FIG_DIR = WORK
EX_REPO = WORK / "timeloop-accelergy-exercises"
assert EX_REPO.exists(), ("exercises repo missing -- run run_ecc_study.py once first "
                          "to clone it (or any single-model script).")
DESIGNS_DIR = EX_REPO / "workspace" / "example_designs" / "example_designs"
assert (WORK / "model_layers.json").exists(), "model_layers.json missing -- run generate_models.py first"
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

# same OUT_ROOT + problem naming as run_ecc_multimodel.py  -> reuses its cache
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
    spec.mapper.timeout = 10000
    spec.mapper.algorithm = "hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_dir / "mapper_console.log", "w") as _logf, \
             contextlib.redirect_stdout(_logf), contextlib.redirect_stderr(_logf):
            tl.call_mapper(spec, output_dir=str(out_dir))
    except Exception:
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
                reads   = _grab(r"Scalar reads \(per-instance\)\s*:\s*([\d.eE+]+)", sec)
                fills   = _grab(r"Scalar fills \(per-instance\)\s*:\s*([\d.eE+]+)", sec)
                updates = _grab(r"Scalar updates \(per-instance\)\s*:\s*([\d.eE+]+)", sec)
                writes  = (fills or 0.0) + (updates or 0.0)   # write-like accesses
                rows.append(dict(layer=layer, level=level, dataspace=ds,
                    reads=reads, writes=writes,
                    energy_pJ=_grab(r"Energy \(total\)\s*:\s*([\d.eE+]+)\s*pJ", sec)))
        else:
            e = _grab(r"Energy \(total\)\s*:\s*([\d.eE+]+)\s*pJ", body)
            if e is not None:
                rows.append(dict(layer=layer, level=level, dataspace="Compute",
                                 reads=None, writes=None, energy_pJ=e))
    return rows

# ---- level -> category. Catch-all keeps unknown levels visible (not dropped) --
def classify(level):
    l = level.lower()
    if "dram" in l:                                return "DRAM"
    if "glb" in l or "buffer" in l or "sram" in l: return "Global buffer"
    if "mac" in l or "compute" in l:               return "Compute"
    return "Local (spads/RF/NoC)"

BASE_CATS = ["DRAM", "Global buffer", "Local (spads/RF/NoC)", "Compute"]
SPLIT_CATS = ("Global buffer", "Local (spads/RF/NoC)")   # the two we can split r/w

if SeperateBreakdown:
    plot_cats = ["DRAM",
                 "Global buffer (read)", "Global buffer (write)",
                 "Local (read)", "Local (write)",
                 "Compute", "ECC decode"]
else:
    plot_cats = BASE_CATS + ["ECC decode"]

# ---- per-category energy (pJ), splitting GB & SRAM into read/write if asked --
def category_energy(df):
    df = df.copy()
    df["reads"]  = df["reads"].fillna(0.0)
    df["writes"] = df["writes"].fillna(0.0)
    out = {c: 0.0 for c in plot_cats}
    for row in df.itertuples(index=False):
        cat, e = row.category, row.energy_pJ
        if SeperateBreakdown and cat in SPLIT_CATS:
            tot = row.reads + row.writes
            frac_r = (row.reads / tot) if tot > 0 else 1.0   # no counts -> call it read
            e_r = e * frac_r
            e_w = e - e_r
            if cat == "Global buffer":
                out["Global buffer (read)"]  += e_r
                out["Global buffer (write)"] += e_w
            else:
                out["Local (read)"]  += e_r
                out["Local (write)"] += e_w
        else:
            out[cat] = out.get(cat, 0.0) + e
    return pd.Series(out).reindex(plot_cats, fill_value=0.0)

# ---------------- run one model -> per-level stacked columns (pJ) -------------
def run_model(model):
    rows, n_ok, n_skip = [], 0, 0
    n = len(model_layers[model])
    for i, L in enumerate(model_layers[model]):
        sig = (int(L["C"]), int(L["M"]), int(L["R"]), int(L["S"]),
               int(L["P"]), int(L["Q"]), int(L["Wstride"]), int(L["Hstride"]))
        stats = run_shape(sig)
        print(f"  {model}: layer {i+1}/{n} [{'ok' if stats else 'skip'}]", flush=True)
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

    base = category_energy(df)                       # Series over plot_cats
    wt   = df[df.dataspace == "Weights"].groupby("category").energy_pJ.sum().reindex(BASE_CATS, fill_value=0.0)
    E_DRAM_W     = wt["DRAM"]
    DRAM_W_READS = df[(df.category == "DRAM") & (df.dataspace == "Weights")].reads.sum()

    dec_base = (DRAM_W_READS / WEIGHTS_PER_CW_BASE) * E_DECODE_CTRL_PJ
    dec_emb  = (DRAM_W_READS / WEIGHTS_PER_CW_EMB)  * E_DECODE_CTRL_PJ

    baseline = base.copy()
    baseline["DRAM"]      += E_DRAM_W * parity_frac      # baseline fetches parity from DRAM
    baseline["ECC decode"] = dec_base
    embedded = base.copy()
    embedded["ECC decode"] = dec_emb                     # embedded: zero extra traffic

    stack = pd.DataFrame({"baseline_bch": baseline, "embedded_ecc": embedded})
    return dict(stack=stack, ok=n_ok, skip=n_skip)

results = {}
for model in MODELS_TO_RUN:
    if model not in model_layers:
        print(f"[skip] {model}: not in model_layers.json"); continue
    print(f"\n=== {model} ({len(model_layers[model])} layers) ===", flush=True)
    r = run_model(model)
    if r is None:
        print(f"  !! {model}: no valid layers, skipped"); continue
    results[model] = r
    tb = r["stack"]["baseline_bch"].sum(); te = r["stack"]["embedded_ecc"].sum()
    print(f"  {model}: saving {(tb-te)/tb*100:.1f}%  ({r['ok']} layers, {r['skip']} skipped)")

if not results:
    raise SystemExit("No models produced results; nothing to plot.")

# ---------------- CSV summary (full per-level breakdown, uJ) ------------------
suffix = "_rw_split" if SeperateBreakdown else ""
summary = {}
for m in results:
    st = results[m]["stack"]; tb = st["baseline_bch"].sum(); te = st["embedded_ecc"].sum()
    row = {}
    for c in plot_cats:
        row[f"base_{c}"] = st.loc[c, "baseline_bch"] / 1e6
        row[f"emb_{c}"]  = st.loc[c, "embedded_ecc"] / 1e6
    row["base_total_uJ"] = tb / 1e6
    row["emb_total_uJ"]  = te / 1e6
    row["saving_pct"]    = (tb - te) / tb * 100
    summary[m] = row
pd.DataFrame(summary).T.to_csv(FIG_DIR / f"multimodel_breakdown{suffix}_summary_uJ.csv")

# ============================ PLOT ===========================================
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "axes.linewidth": 2.0,
    "axes.edgecolor": "#2b2b2b", "pdf.fonttype": 42, "ps.fonttype": 42,
})
# read = base hue, write = darker shade of the same hue (so it reads as one unit)
palette = {
    "DRAM":                    "#2E5A87",
    "Global buffer":           "#4E9F3D",
    "Global buffer (read)":    "#6FBF5A",
    "Global buffer (write)":   "#2E5F24",
    "Local (spads/RF/NoC)":    "#E1A730",
    "Local (read)":            "#F0C05A",
    "Local (write)":           "#A9781A",
    "Compute":                 "#B03A2E",
    "ECC decode":              "#7D3C98",
}
nice_label = {
    "Local (spads/RF/NoC)": "On-chip SRAM/RF",
    "Local (read)":         "On-chip SRAM/RF (read)",
    "Local (write)":        "On-chip SRAM/RF (write)",
}

models = list(results.keys())
n = len(models)

# choose a single unit for the whole figure
maxtot_pJ = max(results[m]["stack"][c].sum() for m in models
                for c in ("baseline_bch", "embedded_ecc"))
if maxtot_pJ / 1e6 >= 1000:
    div, unit = 1e9, "mJ"
else:
    div, unit = 1e6, "\u00b5J"

w, gap = 0.42, 0.10
centers = np.arange(n) * 1.90          # wider spacing between models
x_base  = centers - (w / 2 + gap / 2)
x_emb   = centers + (w / 2 + gap / 2)

# only stack categories that actually carry energy somewhere
active = [c for c in plot_cats
          if sum(results[m]["stack"].loc[c].sum() for m in models) > 1e-9]

fig, ax = plt.subplots(figsize=(3.3 * n + 4, 10))   # wider canvas
bottoms_b = np.zeros(n); bottoms_e = np.zeros(n)
for cat in active:
    vb = np.array([results[m]["stack"].loc[cat, "baseline_bch"] for m in models]) / div
    ve = np.array([results[m]["stack"].loc[cat, "embedded_ecc"] for m in models]) / div
    ax.bar(x_base, vb, w, bottom=bottoms_b, color=palette[cat],
           edgecolor="white", linewidth=1.1, zorder=3, label=nice_label.get(cat, cat))
    ax.bar(x_emb, ve, w, bottom=bottoms_e, color=palette[cat],
           edgecolor="white", linewidth=1.1, zorder=3)
    bottoms_b += vb; bottoms_e += ve

ymax = max(bottoms_b.max(), bottoms_e.max())

# bar totals + per-model saving %
for i, m in enumerate(models):
    tb, te = bottoms_b[i], bottoms_e[i]
    ax.text(x_base[i], tb + ymax * 0.008, f"{tb:.0f}", ha="center", va="bottom",
            fontsize=16, fontweight="bold", color="#2b2b2b")
    ax.text(x_emb[i], te + ymax * 0.008, f"{te:.0f}", ha="center", va="bottom",
            fontsize=16, fontweight="bold", color="#2b2b2b")
    sv = (tb - te) / tb * 100
    ax.text(centers[i], max(tb, te) + ymax * 0.075, f"\u2212{sv:.0f}%",
            ha="center", va="bottom", fontsize=23, fontweight="bold", color="#B03A2E")

# two-level x labels: Base/Emb rotated vertical (no within-pair overlap),
# model name angled + right-aligned to the group centre (no between-group overlap)
ax.set_xticks([])
for i, m in enumerate(models):
    ax.text(x_base[i], -ymax * 0.015, "Base.", ha="center", va="top", rotation=90,
            fontsize=19, color="#555", clip_on=False)
    ax.text(x_emb[i], -ymax * 0.015, "Embe.", ha="center", va="top", rotation=90,
            fontsize=19, color="#555", clip_on=False)
    ax.text(centers[i], -ymax * 0.2, m, ha="center", va="top", rotation=0,
            fontsize=26, fontweight="medium", color="#111", clip_on=False)

# legend: wrap to <=4 columns so 7 entries stay readable
ncol = min(len(active), 4)
legend_rows = math.ceil(len(active) / ncol)
ax.set_ylabel(f"Inference energy ({unit})", fontsize=30, labelpad=12)
ax.set_ylim(0, ymax * 1.20)
ax.set_xlim(centers[0] - 1.2, centers[-1] + 1.2)
ax.tick_params(axis="y", labelsize=22, width=2.0, length=9)
ax.grid(axis="y", ls=":", alpha=0.35, zorder=0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", frameon=False, fontsize=19, ncol=ncol,
          bbox_to_anchor=(0.5, 1.03 + 0.055 * legend_rows),
          handlelength=1.2, columnspacing=1.5, labelspacing=0.4)
ax.set_title(f"Per-level inference energy  \u00b7  Baseline BCH vs Embedded ECC  "
             f"\u00b7  BCH({BCH_N},{BCH_K}) t={BCH_T} | energy(R:W) = 0.8:1 to 1.3:1",
             fontsize=27, pad=34 + 30 * legend_rows, fontweight="medium")

fig.subplots_adjust(bottom=0.26, top=0.83, left=0.07, right=0.99)
plt.savefig(FIG_DIR / f"multimodel_breakdown{suffix}.png", dpi=400, bbox_inches="tight")
plt.savefig(FIG_DIR / f"multimodel_breakdown{suffix}.pdf", bbox_inches="tight")
plt.close()

print(f"\nSaved into: {FIG_DIR}   (SeperateBreakdown={SeperateBreakdown})")
print(f"  multimodel_breakdown{suffix}.png  (400 dpi)")
print(f"  multimodel_breakdown{suffix}.pdf  (vector)")
print(f"  multimodel_breakdown{suffix}_summary_uJ.csv")