"""
Timeloop+Accelergy energy sweep: two apples-to-apples ECC pairs (resnet18).
Run INSIDE the timeloop-accelergy-pytorch container.
"""
import os, re, glob, math, pathlib, subprocess, shutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pytimeloop.timeloopfe.v4 as tl
except ImportError:
    import timeloopfe.v4 as tl

# ================= USER KNOBS =================================================
ARCH           = "eyeriss_like"
NETWORK        = "resnet18"
BCH_N, BCH_K   = 63, 36
BCH_T          = 5
WEIGHT_BITS    = 8
EMB_WEIGHTS_PER_CW = 8
VICTORY        = 500             # lower (e.g. 200) = faster iteration; raise to 3000+ for final
REUSE_FACTOR   = 5.0
E_DECODE_CTRL_PJ = 200           # pJ per codeword (controller)
E_DECODE_USE_PJ  = 100           # pJ per codeword (on-use, lighter unit)
# =============================================================================

WEIGHTS_PER_CW_BASE = BCH_K / WEIGHT_BITS
WEIGHTS_PER_CW_EMB  = EMB_WEIGHTS_PER_CW

assert shutil.which("timeloop-mapper"), "timeloop-mapper not on PATH -- run inside the container"
WORK = pathlib.Path.cwd() / "ecc_energy_study"; WORK.mkdir(exist_ok=True)
EX_REPO = WORK / "timeloop-accelergy-exercises"
if not EX_REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/Accelergy-Project/timeloop-accelergy-exercises.git",
                    str(EX_REPO)], check=True)

DESIGNS_DIR      = EX_REPO / "workspace" / "example_designs" / "example_designs"
LAYER_SHAPES_DIR = EX_REPO / "workspace" / "example_designs" / "layer_shapes" / NETWORK
assert DESIGNS_DIR.exists(),      f"Design dir not found: {DESIGNS_DIR}"
assert LAYER_SHAPES_DIR.exists(), f"Network '{NETWORK}' not found: {LAYER_SHAPES_DIR}"
print(f"Architecture: {ARCH} | Network: {NETWORK}")

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

def design_inputs(problem_yaml):
    files = [str(ARCH_YAML)]
    files += sorted(glob.glob(str(DESIGNS_DIR / "_components" / "*.yaml")))
    files.append(str(DESIGNS_DIR / "_include" / "mapper.yaml"))
    files.append(str(WORK / "globals.yaml"))
    files.append(str(problem_yaml))
    missing = [f for f in files if not os.path.exists(f)]
    assert not missing, f"Missing inputs: {missing}"
    return files

OUT_ROOT = WORK / "outputs" / ARCH / NETWORK; OUT_ROOT.mkdir(parents=True, exist_ok=True)
layer_files = sorted(glob.glob(str(LAYER_SHAPES_DIR / "*.yaml")))
layer_files = [f for f in layer_files if "problem_base" not in os.path.basename(f)]
print(f"Found {len(layer_files)} layers")

# SPEEDUP: identical layer shapes => identical energy. Run the mapper once per
# unique shape and reuse the stats for duplicates.
def layer_signature(problem_path):
    text = pathlib.Path(problem_path).read_text()
    m = re.search(r"instance:\s*(\{[^}]*\})", text)
    return m.group(1).replace(" ", "") if m else pathlib.Path(problem_path).stem

_sig_cache = {}
def run_layer(problem_path):
    name = pathlib.Path(problem_path).stem
    sig = layer_signature(problem_path)
    if sig in _sig_cache:
        print(f"[dup ] {name} (same shape -> reusing)")
        return name, _sig_cache[sig]
    out_dir = OUT_ROOT / name
    stats = out_dir / "timeloop-mapper.stats.txt"
    if stats.exists():
        print(f"[skip] {name} (cached)")
        _sig_cache[sig] = stats
        return name, stats
    spec = tl.Specification.from_yaml_files(*design_inputs(problem_path))
    spec.mapper.num_threads = os.cpu_count() or 4
    spec.mapper.victory_condition = VICTORY
    print(f"[run ] {name} ...", flush=True)
    tl.call_mapper(spec, output_dir=str(out_dir))
    assert stats.exists(), f"No stats for {name}; check logs in {out_dir}"
    _sig_cache[sig] = stats
    return name, stats

stats_files = dict(run_layer(p) for p in layer_files)

def _grab(pattern, text, cast=float, default=None):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else default

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

df = pd.DataFrame([r for n, sf in stats_files.items() for r in parse_stats(sf, n)])
df = df[df.energy_pJ.notna() & (df.energy_pJ > 0)].reset_index(drop=True)
df.to_csv(WORK / f"per_level_energy_{NETWORK}.csv", index=False)

def classify(level):
    l = level.lower()
    if "dram" in l:                                return "DRAM"
    if "glb" in l or "buffer" in l or "sram" in l: return "Global buffer"
    if "mac" in l or "compute" in l:               return "Compute"
    return "Local (spads/RF/NoC)"

df["category"] = df.level.map(classify)
cats = ["DRAM", "Global buffer", "Local (spads/RF/NoC)", "Compute"]
total = df.energy_pJ.sum()
print(f"\n--- Energy breakdown: {NETWORK}, {len(stats_files)} layers ---")
print((df.groupby("category").energy_pJ.sum() / total * 100).round(1).astype(str) + " %")

wt = df[df.dataspace == "Weights"].groupby("category").energy_pJ.sum().reindex(cats, fill_value=0.0)
E_DRAM_W   = wt["DRAM"]
E_ONCHIP_W = wt.drop("DRAM").sum()
DRAM_W_READS = df[(df.category == "DRAM") & (df.dataspace == "Weights")].reads.sum()
print(f"\nWeight-read energy by level:")
for c in cats:
    print(f"  {c:>22}: {wt[c]/1e6:.3f} uJ")
print(f"  DRAM weight reads      : {DRAM_W_READS:,.0f} weights")

parity_frac   = BCH_N / BCH_K - 1.0
mem_reduction = 1.0 - BCH_K / BCH_N
cw_ctrl_base = DRAM_W_READS / WEIGHTS_PER_CW_BASE
cw_ctrl_emb  = DRAM_W_READS / WEIGHTS_PER_CW_EMB
cw_use_base  = DRAM_W_READS / WEIGHTS_PER_CW_BASE * REUSE_FACTOR
cw_use_emb   = DRAM_W_READS / WEIGHTS_PER_CW_EMB  * REUSE_FACTOR
dec_ctrl_base = cw_ctrl_base * E_DECODE_CTRL_PJ
dec_ctrl_emb  = cw_ctrl_emb  * E_DECODE_CTRL_PJ
dec_use_base  = cw_use_base  * E_DECODE_USE_PJ
dec_use_emb   = cw_use_emb   * E_DECODE_USE_PJ

configs = {
    "baseline_bch":        total + E_DRAM_W * parity_frac              + dec_ctrl_base,
    "embedded_ecc":        total                                      + dec_ctrl_emb,
    "baseline_bch_onchip": total + (E_DRAM_W + E_ONCHIP_W)*parity_frac + dec_use_base,
    "embedded_ecc_onchip": total                                      + dec_use_emb,
}
print(f"\nBCH({BCH_N},{BCH_K}) t={BCH_T}: {BCH_N-BCH_K} parity bits")
print(f"  parity overhead : {parity_frac*100:.1f}%   embedded memory reduction: {mem_reduction*100:.1f}%")
print(f"  weights/codeword: baseline {WEIGHTS_PER_CW_BASE:.1f} | embedded {WEIGHTS_PER_CW_EMB:.1f}"
      f"  => baseline decodes {WEIGHTS_PER_CW_EMB/WEIGHTS_PER_CW_BASE:.2f}x more")
print(f"\nTotal energy per config:")
for k, v in configs.items():
    print(f"  {k:>20}: {v/1e6:.3f} uJ")

s1 = (configs["baseline_bch"] - configs["embedded_ecc"]) / configs["baseline_bch"] * 100
s1_dram = parity_frac / (1 + parity_frac) * 100
s2 = (configs["baseline_bch_onchip"] - configs["embedded_ecc_onchip"]) / configs["baseline_bch_onchip"] * 100
print(f"\nPAIR 1 (decode-at-controller): {s1:.2f}% of TOTAL | {s1_dram:.2f}% of DRAM weight energy")
print(f"PAIR 2 (decode-on-use, SRAM protected): {s2:.2f}% of TOTAL")

# ---------------- plot: journal-grade, two decode-placement pairs ------------
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.6,
    "axes.edgecolor": "#2b2b2b",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
plot_cats = cats + ["ECC decode"]
palette = {
    "DRAM": "#2E5A87", "Global buffer": "#4E9F3D",
    "Local (spads/RF/NoC)": "#E1A730", "Compute": "#B03A2E", "ECC decode": "#7D3C98",
}
nice_label = {"Local (spads/RF/NoC)": "On-chip SRAM/RF"}

base = df.groupby("category").energy_pJ.sum().reindex(cats, fill_value=0.0)
def col(delta, decode_e):
    s = base.reindex(plot_cats, fill_value=0.0).copy()
    for c, d in delta.items():
        s[c] += d
    s["ECC decode"] = decode_e
    return s
zero = {c: 0.0 for c in cats}
stack = pd.DataFrame({
    "baseline_bch":        col({**zero, "DRAM": E_DRAM_W*parity_frac}, dec_ctrl_base),
    "embedded_ecc":        col(zero, dec_ctrl_emb),
    "baseline_bch_onchip": col((wt*parity_frac).to_dict(), dec_use_base),
    "embedded_ecc_onchip": col(zero, dec_use_emb),
}) / 1e6

order = ["baseline_bch", "embedded_ecc", "baseline_bch_onchip", "embedded_ecc_onchip"]
xpos  = [0.0, 0.62, 1.72, 2.34]
width = 0.46
active = [c for c in plot_cats if stack[order].loc[c].sum() > 1e-9]

fig, ax = plt.subplots(figsize=(11, 7.2))
bottoms = {k: 0.0 for k in order}
for cat in active:
    vals = [stack.loc[cat, k] for k in order]
    ax.bar(xpos, vals, width, bottom=[bottoms[k] for k in order],
           label=nice_label.get(cat, cat), color=palette[cat],
           edgecolor="white", linewidth=1.3, zorder=3)
    for k, v in zip(order, vals):
        bottoms[k] += v
tops = [bottoms[k] for k in order]
ymax = max(tops)

for xi, tot in zip(xpos, tops):
    ax.text(xi, tot + ymax*0.015, f"{tot:.0f}", ha="center", va="bottom",
            fontsize=17, fontweight="bold", color="#2b2b2b")

def pair_saving(i_base, i_emb, xr):
    tb, te = tops[i_base], tops[i_emb]
    sv = (tb - te) / tb * 100
    ax.annotate("", xy=(xr, te), xytext=(xr, tb),
                arrowprops=dict(arrowstyle="<->", color="#B03A2E", lw=2.2))
    ax.text(xr + 0.05, (tb + te)/2, f"\u2212{sv:.0f}%",
            fontsize=21, fontweight="bold", color="#B03A2E", va="center", ha="left")
pair_saving(0, 1, xpos[1] + 0.32)
pair_saving(2, 3, xpos[3] + 0.32)

ax.set_xticks(xpos)
ax.set_xticklabels(["Baseline\nBCH", "Embedded\nECC", "Baseline\nBCH", "Embedded\nECC"],
                   fontsize=17)
ax.text((xpos[0]+xpos[1])/2, -ymax*0.135, "Decode-at-controller",
        ha="center", fontsize=16, fontweight="bold", color="#333", clip_on=False)
ax.text((xpos[2]+xpos[3])/2, -ymax*0.135, "Decode-on-use",
        ha="center", fontsize=16, fontweight="bold", color="#333", clip_on=False)

ax.set_ylabel("Inference energy (\u00b5J)", fontsize=22, labelpad=10)
ax.set_ylim(0, ymax * 1.16)
ax.set_xlim(-0.5, 3.15)
ax.tick_params(axis="y", labelsize=16, width=1.6, length=7)
ax.tick_params(axis="x", length=0, pad=8)
ax.grid(axis="y", ls=":", alpha=0.35, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", frameon=False, fontsize=14, ncol=len(active),
          bbox_to_anchor=(0.5, 1.11), handlelength=1.1, columnspacing=1.3)
ax.set_title(f"{NETWORK}  \u00b7  BCH({BCH_N},{BCH_K}) t={BCH_T}",
             fontsize=20, pad=38, fontweight="medium")

plt.tight_layout()
plt.savefig(WORK / f"ecc_energy_comparison_{NETWORK}.png", dpi=400, bbox_inches="tight")
plt.savefig(WORK / f"ecc_energy_comparison_{NETWORK}.pdf", bbox_inches="tight")
print(f"\nSaved: ecc_energy_comparison_{NETWORK}.png (400 dpi) + .pdf (vector)")