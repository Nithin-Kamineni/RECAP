"""
Single-model CNN ECC energy sweep -- ONE model, ALL K values, 3 methods.

Same per-level energy model as run_ecc_multimodel_3way.py, but the figure is
reorganised: instead of one bar-group per model, we fix ONE model (default
resnet18) and put ONE bar-group per BCH configuration (K value) on the x-axis.
Each group holds three stacked bars: Baseline BCH / Embedded ECC / Patched ECC.

x-axis is ordered by error-correcting strength (K = 57 -> 30, i.e. t = 1 -> 6),
so the bit-error rate climbs left -> right. Each tick is labelled:

        "1.5% BER"
        "BCH(63, 57, 1)"

with BER = t/N rounded to the nearest 0.5%. Override any label via LABEL_OVERRIDE.

The three approaches (identical model to the multi-model script):
    Baseline BCH  : full codeword in DRAM + SRAM, heavy controller decode.
    Embedded ECC  : no DRAM parity inflation, lighter decode.
    Patched ECC   : only K/N of the data in SRAM (scaled on-chip energy),
                    DRAM like embedded, compute unchanged, ECC decode -> 0,
                    plus reconstruction = DRAM_W_READS*((N-K)/N)*d_min*E_XOR.

The timeloop mapping is ECC-independent, so the mapper runs once for the model
(reusing run_ecc_multimodel.py's cache) and every K reuses those maps.

RUN INSIDE the container, launched from Energy_modeling/ :
    cd /home/workspace
    python3 E3-CODE/generate_models.py                 # once
    python3 E3-CODE/run_ecc_singlemodel_ksweep.py      # this script
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
SeperateBreakdown = False
# =============================================================================

# ================= USER KNOBS ================================================
MODEL              = "resnet18"            # <-- change to any model in model_layers.json
# ARCH               = "simple_weight_stationary"
ARCH               = "eyeriss_like"
BCH_N              = 63
WEIGHT_BITS        = 8
EMB_WEIGHTS_PER_CW = 8
VICTORY            = 500
E_DECODE_CTRL_PJ   = 40                    # pJ per codeword, controller decode
E_XOR_PJ           = 0.01                   # pJ per XOR op  (<< an SRAM read of a weight)

E_codeword_PJ_Incremental_Map = {57: 1.6574, 51: 1.8995, 45: 1.6383, 39: 1.4561, 36: 1.5082, 30: 1.3786}
E_codeword_PJ_Idle_Map        = {57: 1.9359672, 51: 2.2301273, 45: 2.4120856, 39: 2.7891299, 36: 2.8358254, 30: 2.8310811}

# K -> minimum distance.  (BCH(63,K): t = (d_min-1)/2.)
KtoD_minMap = {57: 3, 51: 5, 45: 7, 39: 9, 36: 11, 30: 13}

# x-axis order (increasing BER left -> right = decreasing K)
KS_ORDER = sorted(KtoD_minMap, key=lambda k: KtoD_minMap[k])   # [57,51,45,39,36,30]

# Optional: force a specific tick label for a K. Otherwise BER is t/N to nearest 0.5%.
LABEL_OVERRIDE = {
    # 45: "5% BER\nBCH(63, 45, 3)",
}
# =============================================================================

def k_label(K):
    if K in LABEL_OVERRIDE:
        return LABEL_OVERRIDE[K]
    d = KtoD_minMap[K]; t = (d - 1) // 2
    ber = round(t / BCH_N * 100 * 2) / 2        # nearest 0.5 %
    return f"{ber:g}% BER\nBCH({BCH_N}, {K}, {t})"

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
assert MODEL in model_layers, f"{MODEL} not in model_layers.json"

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
    base   = category_energy(df)                             # all dataspaces
    base_w = category_energy(df[df.dataspace == "Weights"])  # WEIGHTS-only, per category
    wt   = df[df.dataspace == "Weights"].groupby("category").energy_pJ.sum()
    E_DRAM_W     = float(wt.get("DRAM", 0.0))
    DRAM_W_READS = float(df[(df.category == "DRAM") &
                            (df.dataspace == "Weights")].reads.sum())
    return dict(base=base, base_w=base_w, E_DRAM_W=E_DRAM_W,
                DRAM_W_READS=DRAM_W_READS, ok=n_ok, skip=n_skip)

def build_stacks(raw, K):
    parity_frac = BCH_N / K - 1.0
    sram_scale  = K / BCH_N
    wpc_base    = K / WEIGHT_BITS            # weights per codeword (message part)
    wpc_emb     = EMB_WEIGHTS_PER_CW

    base, base_w = raw["base"], raw["base_w"]
    E_DRAM_W, DRAM_W_READS = raw["E_DRAM_W"], raw["DRAM_W_READS"]

    dec_base = (DRAM_W_READS / wpc_base) * E_DECODE_CTRL_PJ
    dec_emb  = (DRAM_W_READS / wpc_emb)  * E_DECODE_CTRL_PJ

    # ---- patched reconstruction (weights only, measured per-codeword energy) ----
    # one codeword message = K bits = K/WEIGHT_BITS weights = wpc_base weights.
    n_codewords = DRAM_W_READS / wpc_base                    # = DRAM_W_READS * WEIGHT_BITS / K
    E_recon_cw  = (E_codeword_PJ_Incremental_Map[K]          # dynamic parity compute / codeword
                   + E_codeword_PJ_Idle_Map[K])              # static datapath energy / codeword
    recon       = n_codewords * E_recon_cw

    baseline = base.copy()
    baseline["DRAM"]        += E_DRAM_W * parity_frac
    baseline["ECC decode"]   = dec_base

    embedded = base.copy()
    embedded["ECC decode"]   = dec_emb

    patched = base.copy()
    for c in ONCHIP_CATS:
        # scale ONLY the weight portion of on-chip energy by K/N; inputs/outputs full.
        patched[c] = (base[c] - base_w[c]) + base_w[c] * sram_scale
        print('remaining input and output:',base[c] - base_w[c],"weight",base_w[c], "number of input and output / number of weights", (base[c] - base_w[c])/base_w[c])
    patched["ECC decode"]     = dec_emb
    patched["Reconstruction"] = recon

    return pd.DataFrame({"baseline_bch": baseline,
                         "embedded_ecc": embedded,
                         "patched_ecc":  patched})

# ---------------- palette / labels -------------------------------------------
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
    "Reconstruction":          "#1F9E8F",
}
nice_label = {
    "Local (spads/RF/NoC)": "On-chip SRAM/RF",
    "Local (read)":         "On-chip SRAM/RF (read)",
    "Local (write)":        "On-chip SRAM/RF (write)",
}
COLS = [("baseline_bch", "Base."), ("embedded_ecc", "Embe."), ("patched_ecc", "Patch.")]

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "axes.linewidth": 2.0,
    "axes.edgecolor": "#2b2b2b", "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ============================ RUN ============================================
print(f"=== gathering {MODEL} ({len(model_layers[MODEL])} layers) ===", flush=True)
raw = gather_model(MODEL)
if raw is None:
    raise SystemExit(f"{MODEL}: no valid layers.")
print(f"  {MODEL}: {raw['ok']} layers ok, {raw['skip']} skipped")

stacks = {K: build_stacks(raw, K) for K in KS_ORDER}

# ------------------------------- PLOT ----------------------------------------
Ks = KS_ORDER; n = len(Ks)
maxtot_pJ = max(stacks[K][col].sum() for K in Ks for col, _ in COLS)
div, unit = (1e9, "mJ") if maxtot_pJ / 1e6 >= 1000 else (1e6, "\u00b5J")

active = [c for c in plot_cats if sum(stacks[K].loc[c].sum() for K in Ks) > 1e-9]

w, gap  = 0.34, 0.06
centers = np.arange(n) * 2.40
xoff    = {"baseline_bch": -(w + gap), "embedded_ecc": 0.0, "patched_ecc": (w + gap)}

fig, ax = plt.subplots(figsize=(3.6 * n + 4, 10))
bottoms = {col: np.zeros(n) for col, _ in COLS}
for cat in active:
    for col, _ in COLS:
        v = np.array([stacks[K].loc[cat, col] for K in Ks]) / div
        ax.bar(centers + xoff[col], v, w, bottom=bottoms[col], color=palette[cat],
               edgecolor="white", linewidth=1.0, zorder=3,
               label=nice_label.get(cat, cat) if col == "baseline_bch" else None)
        bottoms[col] += v

ymax = max(bottoms[col].max() for col, _ in COLS)

# totals + saving% (emb & patched vs baseline)
for i, K in enumerate(Ks):
    tb = bottoms["baseline_bch"][i]
    for col, _ in COLS:
        tot = bottoms[col][i]
        ax.text(centers[i] + xoff[col], tot + ymax * 0.006, f"{tot:.0f}",
                ha="center", va="bottom", fontsize=13, fontweight="bold", color="#2b2b2b")
    for col in ("embedded_ecc", "patched_ecc"):
        sv = (tb - bottoms[col][i]) / tb * 100 if tb > 0 else 0.0
        ax.text(centers[i] + xoff[col], bottoms[col][i] + ymax * 0.045,
                f"\u2212{sv:.0f}%", ha="center", va="bottom",
                fontsize=15, fontweight="bold", color="#B03A2E")

# per-bar method tags (rotated) + BER/BCH tick label per group
ax.set_xticks([])
for i, K in enumerate(Ks):
    for col, tag in COLS:
        ax.text(centers[i] + xoff[col], -ymax * 0.015, tag, ha="center", va="top",
                rotation=90, fontsize=14, color="#555", clip_on=False)
    ax.text(centers[i], -ymax * 0.155, k_label(K), ha="center", va="top",
            fontsize=17, fontweight="medium", color="#111", clip_on=False)

ncol = min(len(active), 4)
legend_rows = math.ceil(len(active) / ncol)
ax.set_ylabel(f"Inference energy ({unit})", fontsize=28, labelpad=12)
ax.set_ylim(0, ymax * 1.22)
ax.set_xlim(centers[0] - 1.4, centers[-1] + 1.4)
ax.tick_params(axis="y", labelsize=20, width=2.0, length=9)
ax.grid(axis="y", ls=":", alpha=0.35, zorder=0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", frameon=False, fontsize=17, ncol=ncol,
          bbox_to_anchor=(0.5, 1.03 + 0.055 * legend_rows),
          handlelength=1.2, columnspacing=1.5, labelspacing=0.4)
ax.set_title(f"{MODEL}  \u00b7  Baseline vs Embedded vs Patched ECC across BCH(63,K)",
             fontsize=24, pad=34 + 30 * legend_rows, fontweight="medium")

fig.subplots_adjust(bottom=0.24, top=0.83, left=0.07, right=0.99)
suffix = "_rw_split" if SeperateBreakdown else ""
png = FIG_DIR / f"singlemodel_{MODEL}_ksweep{suffix}.png"
pdf = FIG_DIR / f"singlemodel_{MODEL}_ksweep{suffix}.pdf"
plt.savefig(png, dpi=400, bbox_inches="tight")
plt.savefig(pdf, bbox_inches="tight")
plt.close()

# CSV summary (uJ), one row per K
rows = {}
for K in Ks:
    st = stacks[K]; row = {"d_min": KtoD_minMap[K], "t": (KtoD_minMap[K] - 1) // 2}
    for c in plot_cats:
        for col, tag in COLS:
            row[f"{tag.strip('.').lower()}_{c}"] = st.loc[c, col] / 1e6
    tb = st["baseline_bch"].sum()
    for col, tag in COLS:
        tt = st[col].sum()
        row[f"{tag.strip('.').lower()}_total_uJ"]   = tt / 1e6
        row[f"{tag.strip('.').lower()}_saving_pct"] = (tb - tt) / tb * 100 if tb > 0 else 0.0
    rows[f"K{K}"] = row
pd.DataFrame(rows).T.to_csv(FIG_DIR / f"singlemodel_{MODEL}_ksweep{suffix}_summary_uJ.csv")

print(f"\nDone. Saved into {FIG_DIR}:")
print(f"  {png.name}")
print(f"  {pdf.name}")
print(f"  singlemodel_{MODEL}_ksweep{suffix}_summary_uJ.csv")
for K in Ks:
    st = stacks[K]; tb = st['baseline_bch'].sum()
    te = st['embedded_ecc'].sum(); tp = st['patched_ecc'].sum()
    print(f"  BCH(63,{K},{(KtoD_minMap[K]-1)//2}): "
          f"emb {(tb-te)/tb*100:5.1f}%  patched {(tb-tp)/tb*100:5.1f}%")