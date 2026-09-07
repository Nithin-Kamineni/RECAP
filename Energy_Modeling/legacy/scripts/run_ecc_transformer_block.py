"""
Timeloop+Accelergy ECC energy sweep for a TRANSFORMER (GPT-2 small).

Each transformer block's weight matmuls are modeled as 1x1 convs (the conv shape
that maps cleanly). All 12 blocks are identical, so results are scaled by N_BLOCKS.
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
ARCH           = "simple_weight_stationary"
LABEL          = "gpt2_small_full"
D_MODEL        = 768                   # GPT-2 small hidden size
D_FF           = 3072                  # FFN inner size (4 x d_model)
N_BLOCKS       = 12                    # GPT-2 small has 12 identical blocks
SEQ            = 1                     # 1 = decode (memory-bound); >1 = prefill
BCH_N, BCH_K   = 63, 36
BCH_T          = 5
WEIGHT_BITS    = 8
EMB_WEIGHTS_PER_CW = 8
VICTORY        = 500
REUSE_FACTOR   = 1.0

# Per-codeword BCH decode energy (0.0 = assume ~cancels). Citable values:
#   65nm VLSI BCH ~2.74-3.43 pJ/bit (Ha & Lee, IEEE TCE 2019) -> ~190 pJ/63b codeword.
#   Analytical (2*n*t + 2*t^2)*E_op (Wicker 1995): (63,36,t=5)=680 GF-ops -> ~34-340 pJ/cw.
E_DECODE_CTRL_PJ = 0.0     # suggested ~100-190 pJ/codeword
E_DECODE_USE_PJ  = 0.0     # suggested ~50-100  pJ/codeword
# =============================================================================

WEIGHTS_PER_CW_BASE = BCH_K / WEIGHT_BITS
WEIGHTS_PER_CW_EMB  = EMB_WEIGHTS_PER_CW

LAYERS = {
    "attn_qkv_proj": dict(C=D_MODEL, M=3*D_MODEL, R=1, S=1, P=SEQ, Q=1),
    "attn_out_proj": dict(C=D_MODEL, M=D_MODEL,   R=1, S=1, P=SEQ, Q=1),
    "ffn_up":        dict(C=D_MODEL, M=D_FF,       R=1, S=1, P=SEQ, Q=1),
    "ffn_down":      dict(C=D_FF,    M=D_MODEL,     R=1, S=1, P=SEQ, Q=1),
}

# ---------------- setup -------------------------------------------------------
assert shutil.which("timeloop-mapper"), "timeloop-mapper not on PATH -- run inside the container"
WORK = pathlib.Path.cwd() / "ecc_energy_study"; WORK.mkdir(exist_ok=True)
EX_REPO = WORK / "timeloop-accelergy-exercises"
if not EX_REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/Accelergy-Project/timeloop-accelergy-exercises.git",
                    str(EX_REPO)], check=True)
DESIGNS_DIR = EX_REPO / "workspace" / "example_designs" / "example_designs"
assert DESIGNS_DIR.exists(), f"Design dir not found: {DESIGNS_DIR}"
print(f"Architecture: {ARCH} | Transformer: {LABEL} (d_model={D_MODEL}, d_ff={D_FF}, seq={SEQ})")

(WORK / "globals.yaml").write_text(
    'variables:\n  version: 0.4\n  global_cycle_seconds: 1e-9\n  technology: "65nm"\n'
)

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
PROB_DIR = WORK / "problems_tf"; PROB_DIR.mkdir(exist_ok=True)

def make_problem(name, C, M, R, S, P, Q, N=1, Wstride=1, Hstride=1):
    path = PROB_DIR / f"{name}.yaml"
    path.write_text(PROBLEM_TEMPLATE.format(C=C, M=M, R=R, S=S, N=N, P=P, Q=Q,
                                            Wstride=Wstride, Hstride=Hstride))
    return path

def design_inputs(problem_yaml):
    files = [str(ARCH_YAML)]
    files += sorted(glob.glob(str(DESIGNS_DIR / "_components" / "*.yaml")))
    files.append(str(DESIGNS_DIR / "_include" / "mapper.yaml"))
    files.append(str(WORK / "globals.yaml"))
    files.append(str(problem_yaml))
    return files

OUT_ROOT = WORK / "outputs" / ARCH / LABEL; OUT_ROOT.mkdir(parents=True, exist_ok=True)

def run_layer(name, params):
    out_dir = OUT_ROOT / name
    stats = out_dir / "timeloop-mapper.stats.txt"
    if stats.exists():
        print(f"[skip] {name} (cached)")
        return name, stats
    prob = make_problem(name, **params)
    spec = tl.Specification.from_yaml_files(*design_inputs(prob))
    spec.mapper.num_threads = os.cpu_count() or 4
    spec.mapper.victory_condition = VICTORY
    # spec.mapper.search_size = 0          # don't give up after 10000 invalids
    # spec.mapper.timeout = 0              # no timeout on consecutive invalids
    # spec.mapper.algorithm = "random-pruned"
    print(f"[run ] {name} ...", flush=True)
    tl.call_mapper(spec, output_dir=str(out_dir))
    assert stats.exists(), f"No stats for {name}; check logs in {out_dir}"
    return name, stats

stats_files = dict(run_layer(n, p) for n, p in LAYERS.items())

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
df.to_csv(WORK / f"per_level_energy_{LABEL}.csv", index=False)

# All 12 blocks identical -> scale aggregate energy + access counts (percentages unchanged).
df["energy_pJ"] = df["energy_pJ"] * N_BLOCKS
df["reads"]     = df["reads"] * N_BLOCKS

def classify(level):
    l = level.lower()
    if "dram" in l:                                return "DRAM"
    if "glb" in l or "buffer" in l or "sram" in l: return "Global buffer"
    if "mac" in l or "compute" in l:               return "Compute"
    return "Local (spads/RF/NoC)"

df["category"] = df.level.map(classify)
cats = ["DRAM", "Global buffer", "Local (spads/RF/NoC)", "Compute"]
total = df.energy_pJ.sum()
print(f"\n--- Energy breakdown: {LABEL}, {len(stats_files)} layers x {N_BLOCKS} blocks ---")
print((df.groupby("category").energy_pJ.sum() / total * 100).round(1).astype(str) + " %")

wt = df[df.dataspace == "Weights"].groupby("category").energy_pJ.sum().reindex(cats, fill_value=0.0)
E_DRAM_W   = wt["DRAM"]
E_ONCHIP_W = wt.drop("DRAM").sum()
DRAM_W_READS = df[(df.category == "DRAM") & (df.dataspace == "Weights")].reads.sum()
print(f"\nWeight-read energy by level:")
for c in cats:
    print(f"  {c:>22}: {wt[c]/1e6:.3f} uJ")
print(f"  DRAM weight energy: {E_DRAM_W/total*100:.1f}% of total  (higher => memory-bound)")

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
print(f"\nBCH({BCH_N},{BCH_K}) t={BCH_T}: parity {parity_frac*100:.1f}% | mem reduction {mem_reduction*100:.1f}%")
print(f"Total energy per config:")
for k, v in configs.items():
    print(f"  {k:>20}: {v/1e6:.3f} uJ")

s1 = (configs["baseline_bch"] - configs["embedded_ecc"]) / configs["baseline_bch"] * 100
s1_dram = parity_frac / (1 + parity_frac) * 100
s2 = (configs["baseline_bch_onchip"] - configs["embedded_ecc_onchip"]) / configs["baseline_bch_onchip"] * 100
print(f"\nPAIR 1 (decode-at-controller): {s1:.2f}% of TOTAL | {s1_dram:.2f}% of DRAM weight energy")
print(f"PAIR 2 (decode-on-use, SRAM protected): {s2:.2f}% of TOTAL")

# ---------------- plot: grant-grade, decode-at-controller pair only ----------
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
    "DRAM":                 "#2E5A87",
    "Global buffer":        "#4E9F3D",
    "Local (spads/RF/NoC)": "#E1A730",
    "Compute":              "#B03A2E",
    "ECC decode":           "#7D3C98",
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
    "Baseline BCH": col({**zero, "DRAM": E_DRAM_W*parity_frac}, dec_ctrl_base),
    "Embedded ECC": col(zero, dec_ctrl_emb),
}) / 1e6

active = [c for c in plot_cats if stack.loc[c].sum() > 1e-9]

fig, ax = plt.subplots(figsize=(7.2, 7.6))
x, width = [0.0, 0.72], 0.40
bottoms = [0.0, 0.0]
for cat in active:
    vals = stack.loc[cat].values
    ax.bar(x, vals, width, bottom=bottoms, label=nice_label.get(cat, cat),
           color=palette[cat], edgecolor="white", linewidth=1.4, zorder=3)
    bottoms = [b + v for b, v in zip(bottoms, vals)]

for xi, tot in zip(x, bottoms):
    ax.text(xi, tot + bottoms[0]*0.02, f"{tot:.0f}", ha="center", va="bottom",
            fontsize=20, fontweight="bold", color="#2b2b2b")

saving = (bottoms[0] - bottoms[1]) / bottoms[0] * 100
xr = 1.18
ax.plot([0.0, xr], [bottoms[0], bottoms[0]], ls=(0, (2, 2)), color="#999", lw=1.5, zorder=1)
ax.plot([0.72, xr], [bottoms[1], bottoms[1]], ls=(0, (2, 2)), color="#999", lw=1.5, zorder=1)
ax.annotate("", xy=(xr, bottoms[1]), xytext=(xr, bottoms[0]),
            arrowprops=dict(arrowstyle="<->", color="#B03A2E", lw=2.4))
ax.text(xr + 0.06, (bottoms[0] + bottoms[1]) / 2, f"\u2212{saving:.0f}%",
        fontsize=26, fontweight="bold", color="#B03A2E", va="center", ha="left")

ax.set_xticks(x)
ax.set_xticklabels(["Baseline\nBCH", "Embedded\nECC"], fontsize=21)
ax.set_ylabel("Inference energy (\u00b5J)", fontsize=23, labelpad=10)
ax.set_ylim(0, bottoms[0] * 1.22)
ax.set_xlim(-0.55, 1.95)
ax.tick_params(axis="y", labelsize=17, width=1.6, length=7)
ax.tick_params(axis="x", length=0)
ax.grid(axis="y", ls=":", alpha=0.35, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", frameon=False, fontsize=15, handlelength=1.1,
          labelspacing=0.4, borderaxespad=0.3)
ax.set_title(f"GPT-2 small (decode)  \u00b7  BCH({BCH_N},{BCH_K})",
             fontsize=20, pad=16, fontweight="medium")

plt.tight_layout()
plt.savefig(WORK / f"ecc_energy_comparison_{LABEL}.png", dpi=400, bbox_inches="tight")
plt.savefig(WORK / f"ecc_energy_comparison_{LABEL}.pdf", bbox_inches="tight")
print(f"\nSaved: ecc_energy_comparison_{LABEL}.png (400 dpi) + .pdf (vector, grant-grade)")