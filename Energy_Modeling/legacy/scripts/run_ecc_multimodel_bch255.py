"""
Multi-model CNN ECC energy comparison for a SINGLE BCH(255,155) configuration.

Fixes ONE BCH code -- BCH(N=255, K=155) -- and compares the three ECC approaches
across many models, one bar-group per model, three stacked bars per group:
    Baseline ECC        (bar tag: "Base")   -- full codeword in DRAM + SRAM, heavy decode.
    Internal Reliability(bar tag: "IR")     -- no DRAM parity inflation, lighter decode.
    +Reconstruction     (bar tag: "+Rec")   -- ONLY the weight portion of on-chip energy
                                               scaled by K/N (inputs/outputs keep full
                                               energy -- they carry no ECC); DRAM like IR;
                                               compute unchanged; ECC decode -> 0;
                                               reconstruction = n_codewords*(E_incr+E_idle).

Per-codeword reconstruction energy is taken from the K=30 measured values
(reused here as requested); codeword GEOMETRY, however, uses N=255, K=155:
    sram_scale = K/N   = 155/255
    wpc_base   = K/WEIGHT_BITS = 155/8   (weights per codeword message)

This figure is built for an NSF grant -- large fonts, clean palette, polished.

RUN INSIDE the container, launched from Energy_modeling/ :
    cd /home/workspace
    python3 E3-CODE/generate_models.py            # once
    python3 E3-CODE/run_ecc_multimodel_bch255.py  # this script
"""
import os, re, glob, json, math, pathlib, shutil, contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

try:
    import pytimeloop.timeloopfe.v4 as tl
except ImportError:
    import timeloopfe.v4 as tl

# ================= DECISION: split on-chip storage into read/write? ===========
SeperateBreakdown = False
# =============================================================================

# ================= USER KNOBS ================================================
# ARCH               = "simple_weight_stationary"
ARCH               = "eyeriss_like"
BCH_N              = 255                    # codeword length  (NEW: 255-bit code)
BCH_K              = 155                    # message length
WEIGHT_BITS        = 8
EMB_WEIGHTS_PER_CW = 255/8
VICTORY            = 500
E_DECODE_CTRL_PJ   = 40                     # pJ per codeword, controller decode

# Per-codeword reconstruction energy -- reuse the measured K=30 values, as requested.
E_codeword_PJ_Incremental = 1.3786          # dynamic parity compute / codeword
E_codeword_PJ_Idle        = 2.8310811       # static datapath energy / codeword

MODELS_TO_RUN = [
    # "resnet18",
    "resnet50",
    # "densenet121",
    # "squeezenet1_1",
    "mobilenet_v2",
    "efficientnet_b0",
    "convnext_tiny",
    # "xception",
]
# nicer display names for the x-axis
PRETTY = {
    "resnet18": "ResNet-18", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
    "squeezenet1_1": "SqueezeNet", "mobilenet_v2": "MobileNet-V2",
    "efficientnet_b0": "EfficientNet-B0", "convnext_tiny": "ConvNeXt-T",
    "xception": "Xception",
}

# ---- ALL figure font sizes in one place (edit here) -------------------------
FS = {
    "bar_total":    32,   # the "42" totals above each bar
    "saving_pct":   32,   # the "-22%" callouts  line 422
    "bar_tag":      36,   # rotated Base / IR / +Rec tags
    "model_name":   36,   # ResNet-18, MobileNet-V2, ...
    "y_label":      36,   # "Inference energy (mJ)"
    "y_ticks":      34,   # y-axis numbers
    "legend":       33,   # colour-category legend
    "title":        32,   # main title (currently commented out)
    "acronym_key":  32,   # the boxed Base = ... / IR = ... / +Rec = ... key
}
# =============================================================================

# ---------------- setup: repo + design paths ---------------------------------
assert shutil.which("timeloop-mapper"), "timeloop-mapper not on PATH -- run inside the container"
WORK = pathlib.Path.cwd() / "ecc_energy_study"; WORK.mkdir(exist_ok=True)
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
                writes  = (fills or 0.0) + (updates or 0.0)
                rows.append(dict(layer=layer, level=level, dataspace=ds,
                    reads=reads, writes=writes,
                    energy_pJ=_grab(r"Energy \(total\)\s*:\s*([\d.eE+]+)\s*pJ", sec)))
        else:
            e = _grab(r"Energy \(total\)\s*:\s*([\d.eE+]+)\s*pJ", body)
            if e is not None:
                rows.append(dict(layer=layer, level=level, dataspace="Compute",
                                 reads=None, writes=None, energy_pJ=e))
    return rows

def classify(level):
    l = level.lower()
    if "dram" in l:                                return "DRAM"
    if "glb" in l or "buffer" in l or "sram" in l: return "Global buffer"
    if "mac" in l or "compute" in l:               return "Compute"
    return "Local (spads/RF/NoC)"

# ---- category layout --------------------------------------------------------
if SeperateBreakdown:
    PHYS_CATS = ["DRAM",
                 "Global buffer (read)", "Global buffer (write)",
                 "Local (read)", "Local (write)", "Compute"]
    ONCHIP_CATS = ["Global buffer (read)", "Global buffer (write)",
                   "Local (read)", "Local (write)"]
else:
    PHYS_CATS = ["DRAM", "Global buffer", "Local (spads/RF/NoC)", "Compute"]
    ONCHIP_CATS = ["Global buffer", "Local (spads/RF/NoC)"]

plot_cats  = PHYS_CATS + ["ECC decode", "Reconstruction"]
SPLIT_CATS = ("Global buffer", "Local (spads/RF/NoC)")

def category_energy(df):
    df = df.copy()
    df["reads"]  = df["reads"].fillna(0.0)
    df["writes"] = df["writes"].fillna(0.0)
    out = {c: 0.0 for c in plot_cats}
    for row in df.itertuples(index=False):
        cat, e = row.category, row.energy_pJ
        if SeperateBreakdown and cat in SPLIT_CATS:
            tot = row.reads + row.writes
            frac_r = (row.reads / tot) if tot > 0 else 1.0
            e_r, e_w = e * frac_r, e - e * frac_r
            if cat == "Global buffer":
                out["Global buffer (read)"]  += e_r
                out["Global buffer (write)"] += e_w
            else:
                out["Local (read)"]  += e_r
                out["Local (write)"] += e_w
        else:
            out[cat] = out.get(cat, 0.0) + e
    return pd.Series(out).reindex(plot_cats, fill_value=0.0)

def gather_model(model):
    rows, n_ok, n_skip = [], 0, 0
    n = len(model_layers[model])
    for i, L in enumerate(model_layers[model]):
        sig = (int(L["C"]), int(L["M"]), int(L["R"]), int(L["S"]),
               int(L["P"]), int(L["Q"]), int(L["Wstride"]), int(L["Hstride"]))
        stats = run_shape(sig)
        print(f"    {model}: layer {i+1}/{n} [{'ok' if stats else 'skip'}]", flush=True)
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
    base   = category_energy(df)
    base_w = category_energy(df[df.dataspace == "Weights"])
    wt   = df[df.dataspace == "Weights"].groupby("category").energy_pJ.sum()
    E_DRAM_W     = float(wt.get("DRAM", 0.0))
    DRAM_W_READS = float(df[(df.category == "DRAM") &
                            (df.dataspace == "Weights")].reads.sum())
    return dict(base=base, base_w=base_w, E_DRAM_W=E_DRAM_W,
                DRAM_W_READS=DRAM_W_READS, ok=n_ok, skip=n_skip)

def build_stacks(raw):
    N, K = BCH_N, BCH_K
    parity_frac = N / K - 1.0
    sram_scale  = K / N                       # 155/255
    wpc_base    = K / WEIGHT_BITS             # weights per codeword (message part)
    wpc_emb     = EMB_WEIGHTS_PER_CW

    base, base_w = raw["base"], raw["base_w"]
    E_DRAM_W, DRAM_W_READS = raw["E_DRAM_W"], raw["DRAM_W_READS"]

    #decoding calclation is wrong, I teporary fixed it by making it similar for both
    dec_base = (DRAM_W_READS / wpc_base) * E_DECODE_CTRL_PJ
    dec_emb  = (DRAM_W_READS / wpc_emb)  * E_DECODE_CTRL_PJ

    # ---- patched reconstruction (weights only) ----
    # one codeword message = K bits = K/WEIGHT_BITS weights = wpc_base weights.
    n_codewords = DRAM_W_READS / wpc_base
    E_recon_cw  = E_codeword_PJ_Incremental + E_codeword_PJ_Idle   # reused K=30 values
    recon       = n_codewords * E_recon_cw

    baseline = base.copy()
    baseline["DRAM"]        += E_DRAM_W * parity_frac
    baseline["ECC decode"]   = dec_base

    embedded = base.copy()
    embedded["ECC decode"]   = dec_emb

    patched = base.copy()
    for c in ONCHIP_CATS:
        patched[c] = (base[c] - base_w[c]) + base_w[c] * sram_scale
    patched["ECC decode"]     = dec_emb
    patched["Reconstruction"] = recon

    return pd.DataFrame({"baseline_bch": baseline,
                         "embedded_ecc": embedded,
                         "patched_ecc":  patched})

# ================= GRANT-QUALITY PALETTE / STYLE =============================
# refined, high-contrast, colour-blind-considerate stack palette
palette = {
    "DRAM":                    "#1F4E79",   # deep blue
    "Global buffer":           "#2E8B57",   # sea green
    "Global buffer (read)":    "#3CB371",
    "Global buffer (write)":   "#1E5E3A",
    "Local (spads/RF/NoC)":    "#E8A33D",   # warm amber
    "Local (read)":            "#F2C46B",
    "Local (write)":           "#B5791F",
    "Compute":                 "#C0392B",   # brick red
    "ECC decode":              "#7D3C98",   # violet
    "Reconstruction":          "#17A398",   # teal
}
nice_label = {
    "DRAM":                 "DRAM",
    "Global buffer":        "Global buffer (SRAM)",
    "Local (spads/RF/NoC)": "Register File/NoC/scratchpads",
    "Local (read)":         "On-chip SRAM/RF (read)",
    "Local (write)":        "On-chip SRAM/RF (write)",
    "Compute":              "Compute (MAC)",
    "ECC decode":           "ECC decode",
    "Reconstruction":       "Reconstruction",
}
# (df column, short bar tag, full-form name for the key)
COLS = [("baseline_bch", "External",   "Baseline ECC"),
        ("embedded_ecc", "Internal",     "Internal Reliability"),
        ("patched_ecc",  "+Recon", "+Reconstruction")]

# one-line acronym key spelled out on the figure
ACRONYM_KEY = "     ".join(f"{tag} = {full}" for _, tag, full in COLS)

# Big, clean typography for a grant figure
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 2.4,
    "axes.edgecolor": "#222222",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

# ============================ RUN ============================================
raws = {}
for model in MODELS_TO_RUN:
    if model not in model_layers:
        print(f"[skip] {model}: not in model_layers.json"); continue
    print(f"\n=== gathering {model} ({len(model_layers[model])} layers) ===", flush=True)
    r = gather_model(model)
    if r is None:
        print(f"  !! {model}: no valid layers, skipped"); continue
    if r["DRAM_W_READS"] <= 0:
        print(f"  !! {model}: DRAM weight reads = 0 (classify() missed a level?); skipped"); continue
    raws[model] = r
    print(f"  {model}: {r['ok']} layers ok, {r['skip']} skipped", flush=True)

if not raws:
    raise SystemExit("No models produced results; nothing to plot.")

models = [m for m in MODELS_TO_RUN if m in raws]
stacks = {m: build_stacks(raws[m]) for m in models}
n = len(models)

# ------------------------------- PLOT ----------------------------------------
maxtot_pJ = max(stacks[m][col].sum() for m in models for col, _, _ in COLS)
div, unit = (1e9, "mJ") if maxtot_pJ / 1e6 >= 1000 else (1e6, "\u00b5J")

active = [c for c in plot_cats if sum(stacks[m].loc[c].sum() for m in models) > 1e-9]

w, gap  = 0.40, 0.07
centers = np.arange(n) * 2.70
xoff    = {"baseline_bch": -(w + gap), "embedded_ecc": 0.0, "patched_ecc": (w + gap)}

fig, ax = plt.subplots(figsize=(3.9 * n + 5, 12))

# subtle alternating background bands per model group (helps a wide figure read)
for i in range(n):
    if i % 2 == 1:
        ax.axvspan(centers[i] - 1.35, centers[i] + 1.35, color="#F5F6F8", zorder=0)

bottoms = {col: np.zeros(n) for col, _, _ in COLS}
for cat in active:
    for col, _, _ in COLS:
        v = np.array([stacks[m].loc[cat, col] for m in models]) / div
        ax.bar(centers + xoff[col], v, w, bottom=bottoms[col], color=palette[cat],
               edgecolor="white", linewidth=1.4, zorder=3)
        bottoms[col] += v

ymax = max(bottoms[col].max() for col, _, _ in COLS)

# totals on each bar + saving% for IR & +Recon vs Baseline
for i, m in enumerate(models):
    tb = bottoms["baseline_bch"][i]
    # for col, _, _ in COLS:
    #     tot = bottoms[col][i]
    #     ax.text(centers[i] + xoff[col], tot + ymax * 0.008, f"{tot:.0f}",
    #             ha="center", va="bottom", fontsize=FS["bar_total"], color="#222222")
    for col in ("embedded_ecc", "patched_ecc"):
        sv = (tb - bottoms[col][i]) / tb * 100 if tb > 0 else 0.0
        if col == "patched_ecc":
            ax.text(centers[i] + xoff[col] + 0.22, bottoms[col][i] + ymax * 0.01,
                    f"-{sv:.0f}%", ha="center", va="bottom",
                    fontsize=FS["saving_pct"], color="#C0392B")
        if col == "embedded_ecc":
            ax.text(centers[i] + xoff[col]+0.05, bottoms[col][i] + ymax * 0.03,
                    f"-{sv:.0f}%", ha="center", va="bottom",
                    fontsize=FS["saving_pct"], color="#C0392B")

# x labels: per-bar SHORT acronym tag (rotated, large) + model name centred
ax.set_xticks([])
for i, m in enumerate(models):
    for col, tag, _ in COLS:
        ax.text(centers[i] + xoff[col], -ymax * 0.018, tag, ha="center", va="top",
                rotation=90, fontsize=FS["bar_tag"], fontweight="medium", color="#333",
                clip_on=False)
    ax.text(centers[i], -ymax * 0.470, PRETTY.get(m, m), ha="center", va="top",
            fontsize=FS["model_name"], color="#111", clip_on=False)

# ---- colour legend (energy categories) -------------------------------------
legend_handles = [Patch(facecolor=palette[c], edgecolor="white", label=nice_label.get(c, c)) for c in active]
ncol = min(len(active), 3)
legend_rows = math.ceil(len(active) / ncol)

ax.set_ylabel(f"Inference energy ({unit})", fontsize=FS["y_label"], labelpad=16, fontweight="medium")
ax.set_ylim(0, ymax * 1.24)
ax.set_xlim(centers[0] - 1.5, centers[-1] + 1.5)
ax.tick_params(axis="y", labelsize=FS["y_ticks"], width=2.4, length=11)
ax.grid(axis="y", ls=":", alpha=0.4, zorder=1)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax.legend(handles=legend_handles, loc="upper center", frameon=False, fontsize=FS["legend"],
          ncol=ncol, bbox_to_anchor=(0.45, 1.25 + 0.052 * legend_rows),
          handlelength=1.5, handleheight=1.3, columnspacing=1.8, labelspacing=0.5)

# ax.set_title(f"Per-level inference energy across CNN models  \u00b7  "
#              f"Baseline vs Internal Reliability vs +Reconstruction  \u00b7  BCH({BCH_N}, {BCH_K})",
#              fontsize=FS["title"], pad=44 + 34 * legend_rows, fontweight="bold")

# ---- acronym KEY: full forms of the three bar tags, boxed & large ----------
# fig.text(0.5, 0.065, ACRONYM_KEY, ha="center", va="center",
#          fontsize=FS["acronym_key"], color="#111",
#          bbox=dict(boxstyle="round,pad=0.3", facecolor="#F2F4F7",
#                    edgecolor="#333333", linewidth=1.8))

fig.subplots_adjust(bottom=0.30, top=0.82, left=0.06, right=0.99)
suffix = "_rw_split" if SeperateBreakdown else ""
png = FIG_DIR / f"multimodel_bch{BCH_N}_{BCH_K}{suffix}_eyeriss.png"
pdf = FIG_DIR / f"multimodel_bch{BCH_N}_{BCH_K}{suffix}_eyeriss.pdf"
plt.savefig(png, dpi=400, bbox_inches="tight", pad_inches=0.35)
plt.savefig(pdf, bbox_inches="tight", pad_inches=0.35)
plt.close()

# CSV summary (uJ), one row per model
rows = {}
for m in models:
    st = stacks[m]; row = {}
    for c in plot_cats:
        for col, tag, _ in COLS:
            key = tag.strip("+").lower()
            row[f"{key}_{c}"] = st.loc[c, col] / 1e6
    tb = st["baseline_bch"].sum()
    for col, tag, _ in COLS:
        key = tag.strip("+").lower()
        tt = st[col].sum()
        row[f"{key}_total_uJ"]   = tt / 1e6
        row[f"{key}_saving_pct"] = (tb - tt) / tb * 100 if tb > 0 else 0.0
    rows[m] = row
pd.DataFrame(rows).T.to_csv(FIG_DIR / f"multimodel_bch{BCH_N}_{BCH_K}{suffix}_summary_uJ.csv")

print(f"\nDone. Saved into {FIG_DIR}:")
print(f"  {png.name}")
print(f"  {pdf.name}")
print(f"  multimodel_bch{BCH_N}_{BCH_K}{suffix}_summary_uJ.csv")
for m in models:
    st = stacks[m]; tb = st['baseline_bch'].sum()
    te = st['embedded_ecc'].sum(); tp = st['patched_ecc'].sum()
    print(f"  {PRETTY.get(m,m):16s}: IR {(tb-te)/tb*100:5.1f}%  +Recon {(tb-tp)/tb*100:5.1f}%")