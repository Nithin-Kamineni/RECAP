"""
Transformer / LLM ECC energy sweep -- FULL per-level energy breakdown,
with an optional READ/WRITE split for on-chip storage (5 vs 7 segments).

The transformer analogue of run_ecc_multimodel_breakdown_rw.py. Arch:
simple_weight_stationary (required for GEMM; eyeriss_like fails on large matmul
dims). Reads ecc_energy_study/transformer_layers.json (from
generate_transformers.py). Each weight matmul is a 1x1 conv; per-block matmuls
are scaled by the model's block count, exactly like run_ecc_transformer_block.py.

For every model it draws TWO stacked bars, Baseline BCH vs Embedded ECC
(decode-at-controller).

  SeperateBreakdown = False -> 5 segments:
        DRAM, Global buffer, On-chip SRAM/RF, Compute, ECC decode
  SeperateBreakdown = True  -> 7 segments (GB & SRAM each split read/write):
        DRAM, GB(read), GB(write), SRAM(read), SRAM(write), Compute, ECC decode

  The read/write split apportions each level's Energy(total) by its access
  counts (writes = Scalar fills + Scalar updates). Exact if read/write per-access
  energies are equal; otherwise a count-proportional approximation. Totals and
  saving % are identical either way.

Figures/CSV save next to THIS script (the Transformers folder).

RUN INSIDE the container, launched from Energy_modeling/ :
    cd /home/workspace
    python3 Transformers/generate_transformers.py       # once
    python3 Transformers/run_llm_breakdown.py            # this script
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
SeperateBreakdown = False     # True -> 7 segments; False -> 5 segments
# =============================================================================

# ================= USER KNOBS (locked identical across all models) ===========
ARCH               = "simple_weight_stationary"
BCH_N, BCH_K       = 63, 36
BCH_T              = 5
WEIGHT_BITS        = 8
EMB_WEIGHTS_PER_CW = 8
VICTORY            = 500          # mapper effort; raise to 3000+ for final numbers
E_DECODE_CTRL_PJ   = 200          # pJ per codeword, controller decode

# Which models to plot (must exist in transformer_layers.json). Comment to trim.
MODELS_TO_RUN = [
                "distilbert",
                "gpt2",
                "bert_base",
                "opt_125m",
                "tinyllama",
                ]
# =============================================================================

WEIGHTS_PER_CW_BASE = BCH_K / WEIGHT_BITS
WEIGHTS_PER_CW_EMB  = EMB_WEIGHTS_PER_CW
parity_frac         = BCH_N / BCH_K - 1.0

# ---------------- setup: repo + design paths ---------------------------------
assert shutil.which("timeloop-mapper"), "timeloop-mapper not on PATH -- run inside the container"
WORK = pathlib.Path.cwd() / "ecc_energy_study"; WORK.mkdir(exist_ok=True)
try:
    FIG_DIR = pathlib.Path(__file__).resolve().parent      # the Transformers folder
except NameError:
    FIG_DIR = WORK
EX_REPO = WORK / "timeloop-accelergy-exercises"
assert EX_REPO.exists(), ("exercises repo missing -- run a CNN single-model script "
                          "once first to clone it.")
DESIGNS_DIR = EX_REPO / "workspace" / "example_designs" / "example_designs"

_tf_path = WORK / "transformer_layers.json"
assert _tf_path.exists(), "transformer_layers.json missing -- run generate_transformers.py first"
_tf = json.loads(_tf_path.read_text())
META             = _tf.get("_meta", {})
transformer_layers = _tf["models"]
SEQ = META.get("SEQ", 1)

(WORK / "globals.yaml").write_text(
    'variables:\n  version: 0.4\n  global_cycle_seconds: 1e-9\n  technology: "65nm"\n')

# DRAM must be able to hold the LARGEST weight tensor across all layers, or the
# mapper fails on that layer (transformer lm_heads / wide FFNs are 20-70M
# entries, far above the 1M that sufficed for CNNs). Size depth to the biggest
# tensor, rounded up to a power of two, with headroom.
def _max_tensor_entries():
    mx = 0
    for layers in transformer_layers.values():
        for L in layers:
            mx = max(mx, int(L["C"]) * int(L["M"]) * int(L["R"]) * int(L["S"]))
    return mx
_max_entries = _max_tensor_entries()
DRAM_DEPTH = 1 << max(20, (_max_entries * 2 - 1).bit_length())   # >= 2x largest tensor
print(f"[arch] largest weight tensor = {_max_entries/1e6:.1f}M entries -> DRAM depth = {DRAM_DEPTH:,}")

def patched_arch_path():
    src = DESIGNS_DIR / ARCH / "arch.yaml"
    dst = WORK / f"arch_{ARCH}_patched.yaml"
    text = src.read_text()
    dram_attrs = text.split("name: DRAM")[1].split("!")[0] if "name: DRAM" in text else ""
    if "depth" not in dram_attrs:
        text = text.replace(
            'class: DRAM\n    attributes:\n      type: "LPDDR4"',
            f'class: DRAM\n    attributes:\n      depth: {DRAM_DEPTH}\n      type: "LPDDR4"')
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
PROB_DIR = WORK / "problems_llm"; PROB_DIR.mkdir(exist_ok=True)

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

OUT_ROOT = WORK / "outputs" / ARCH / "llm"; OUT_ROOT.mkdir(parents=True, exist_ok=True)
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

BASE_CATS  = ["DRAM", "Global buffer", "Local (spads/RF/NoC)", "Compute"]
SPLIT_CATS = ("Global buffer", "Local (spads/RF/NoC)")

if SeperateBreakdown:
    plot_cats = ["DRAM",
                 "Global buffer (read)", "Global buffer (write)",
                 "Local (read)", "Local (write)",
                 "Compute", "ECC decode"]
else:
    plot_cats = BASE_CATS + ["ECC decode"]

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

# ---------------- run one model: matmuls scaled by block count ---------------
SKIPPED = []   # (model, layer, C, M) for every layer the mapper failed on
def run_model(model):
    rows, n_ok, n_skip = [], 0, 0
    layers = transformer_layers[model]
    for L in layers:
        sig = (int(L["C"]), int(L["M"]), int(L["R"]), int(L["S"]),
               int(L["P"]), int(L["Q"]), int(L["Wstride"]), int(L["Hstride"]))
        stats = run_shape(sig)
        cnt = int(L.get("count", 1))
        print(f"  {model}: {L['name']} x{cnt} [{'ok' if stats else 'SKIP'}]", flush=True)
        if stats is None:
            n_skip += 1
            SKIPPED.append((model, L["name"], L["C"], L["M"]))
            continue
        for r in parse_stats(stats, L["name"]):
            r["energy_pJ"] *= cnt                    # per-block matmul repeats cnt times
            if r["reads"]  is not None: r["reads"]  *= cnt
            if r["writes"] is not None: r["writes"] *= cnt
            rows.append(r)
        n_ok += 1
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df[df.energy_pJ.notna() & (df.energy_pJ > 0)].reset_index(drop=True)
    df["category"] = df.level.map(classify)

    base = category_energy(df)
    wt   = df[df.dataspace == "Weights"].groupby("category").energy_pJ.sum().reindex(BASE_CATS, fill_value=0.0)
    E_DRAM_W     = wt["DRAM"]
    DRAM_W_READS = df[(df.category == "DRAM") & (df.dataspace == "Weights")].reads.sum()

    dec_base = (DRAM_W_READS / WEIGHTS_PER_CW_BASE) * E_DECODE_CTRL_PJ
    dec_emb  = (DRAM_W_READS / WEIGHTS_PER_CW_EMB)  * E_DECODE_CTRL_PJ

    baseline = base.copy()
    baseline["DRAM"]      += E_DRAM_W * parity_frac
    baseline["ECC decode"] = dec_base
    embedded = base.copy()
    embedded["ECC decode"] = dec_emb

    stack = pd.DataFrame({"baseline_bch": baseline, "embedded_ecc": embedded})
    return dict(stack=stack, ok=n_ok, skip=n_skip)

results = {}
for model in MODELS_TO_RUN:
    if model not in transformer_layers:
        print(f"[skip] {model}: not in transformer_layers.json"); continue
    print(f"\n=== {model} ({len(transformer_layers[model])} matmul groups) ===", flush=True)
    r = run_model(model)
    if r is None:
        print(f"  !! {model}: no valid layers, skipped"); continue
    results[model] = r
    tb = r["stack"]["baseline_bch"].sum(); te = r["stack"]["embedded_ecc"].sum()
    print(f"  {model}: saving {(tb-te)/tb*100:.1f}%  ({r['ok']} groups, {r['skip']} skipped)")

if not results:
    raise SystemExit("No models produced results; nothing to plot.")

# ---- LOUD warning: a silently-dropped layer makes the plot wrong-but-plausible
if SKIPPED:
    print("\n" + "!" * 70)
    print(f"WARNING: {len(SKIPPED)} layer(s) FAILED TO MAP and were dropped.")
    print("The bars below are MISSING these layers' energy -- do not trust them")
    print("until this is resolved. Check the matching mapper_console.log for the")
    print("real error (fanout / capacity / timeout).")
    for mdl, ln, C, M in SKIPPED:
        print(f"    {mdl:11s} {ln:10s} (C={C}, M={M})")
    print("!" * 70 + "\n")
else:
    print("\n[ok] every layer mapped -- no silent drops.\n")

# ---------------- CSV summary ------------------------------------------------
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
pd.DataFrame(summary).T.to_csv(FIG_DIR / f"llm_breakdown{suffix}_summary_uJ.csv")

# ============================ PLOT ===========================================
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "axes.linewidth": 2.0,
    "axes.edgecolor": "#2b2b2b", "pdf.fonttype": 42, "ps.fonttype": 42,
})
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

maxtot_pJ = max(results[m]["stack"][c].sum() for m in models
                for c in ("baseline_bch", "embedded_ecc"))
if maxtot_pJ / 1e6 >= 1000:
    div, unit = 1e9, "mJ"
else:
    div, unit = 1e6, "\u00b5J"

w, gap = 0.42, 0.10
centers = np.arange(n) * 1.90
x_base  = centers - (w / 2 + gap / 2)
x_emb   = centers + (w / 2 + gap / 2)

active = [c for c in plot_cats
          if sum(results[m]["stack"].loc[c].sum() for m in models) > 1e-9]

fig, ax = plt.subplots(figsize=(3.3 * n + 4, 10))
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

for i, m in enumerate(models):
    tb, te = bottoms_b[i], bottoms_e[i]
    ax.text(x_base[i], tb + ymax * 0.008, f"{tb:.0f}", ha="center", va="bottom",
            fontsize=16, fontweight="bold", color="#2b2b2b")
    ax.text(x_emb[i], te + ymax * 0.008, f"{te:.0f}", ha="center", va="bottom",
            fontsize=16, fontweight="bold", color="#2b2b2b")
    sv = (tb - te) / tb * 100
    ax.text(centers[i], max(tb, te) + ymax * 0.075, f"\u2212{sv:.0f}%",
            ha="center", va="bottom", fontsize=23, fontweight="bold", color="#B03A2E")

# Base/Emb rotated vertical; model name angled + right-aligned to group centre
ax.set_xticks([])
for i, m in enumerate(models):
    ax.text(x_base[i], -ymax * 0.015, "Base.", ha="center", va="top", rotation=90,
            fontsize=19, color="#555", clip_on=False)
    ax.text(x_emb[i], -ymax * 0.015, "Embe.", ha="center", va="top", rotation=90,
            fontsize=19, color="#555", clip_on=False)
    ax.text(centers[i], -ymax * 0.17, m, ha="right", va="top", rotation=22,
            fontsize=26, fontweight="medium", color="#111", clip_on=False)

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
ax.set_title(f"LLM per-level inference energy (SEQ={SEQ})  \u00b7  Baseline BCH vs Embedded ECC  "
             f"\u00b7  BCH({BCH_N},{BCH_K}) t={BCH_T}",
             fontsize=25, pad=34 + 30 * legend_rows, fontweight="medium")

fig.subplots_adjust(bottom=0.26, top=0.83, left=0.07, right=0.99)
plt.savefig(FIG_DIR / f"llm_breakdown{suffix}.png", dpi=400, bbox_inches="tight")
plt.savefig(FIG_DIR / f"llm_breakdown{suffix}.pdf", bbox_inches="tight")
plt.close()

print(f"\nSaved into: {FIG_DIR}   (SeperateBreakdown={SeperateBreakdown}, SEQ={SEQ})")
print(f"  llm_breakdown{suffix}.png  (400 dpi)")
print(f"  llm_breakdown{suffix}.pdf  (vector)")
print(f"  llm_breakdown{suffix}_summary_uJ.csv")