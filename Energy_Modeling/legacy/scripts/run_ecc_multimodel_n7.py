"""
Multi-model CNN ECC energy sweep -- N=7 proposal, 3-WAY comparison, ALL models,
TWO architectures.

Derived from E3-CODE/Run_ecc_singlemodel_ksweep.py and Run_ecc_multimodel_3way.py.
The important structural change: those scripts assumed ONE (N, K) pair shared by
all three approaches and swept K.  Here the three approaches have DIFFERENT and
FIXED codeword geometry, and the sweep axis is the MODEL instead.

    1. Baseline ECC   -- shortened Hamming SEC (11, 7), d_min = 3, t = 1.
                         Parity is stored alongside the data, so DRAM weight
                         traffic inflates by N/K = 11/7 (+57.1%).

    2. Embedded ECC   -- (7, 6): 6 payload bits + 1 parity bit, 7 bits stored.
                         The parity bit is embedded inside the already-stored
                         quantized word, so there is NO DRAM inflation.

    3. Recon+         -- same (7, 6) codeword as embedded.
                         DRAM  : same as embedded (no inflation).
                         On-chip: only K/N = 6/7 of the WEIGHT data is kept;
                                  inputs/outputs are untouched.
                         Compute: unchanged.
                         Reconstruction: the parity portion is regenerated
                                  on-chip by the synthesized XOR/reconstruction
                                  datapath just before the accelerator uses it,
                                  charged per codeword from the Design Compiler
                                  measurement (see PROVENANCE below).

DECODE ENERGY IS CHARGED AS ZERO FOR ALL THREE ARMS (INCLUDE_ECC_DECODE = False).
When it was modelled it came to 0.3-0.8% of total energy, below the resolution of
the rest of the model, so the "ECC decode" category sums to zero everywhere and is
dropped from both the stacked bars and the legend. Flip the flag to bring it back.

Quantization is 8-bit throughout, so weights-per-codeword = K / 8 for every arm:
    baseline  7/8 = 0.875   (1.143 codewords per weight)
    embedded  6/8 = 0.750   (1.333 codewords per weight)
    Recon+    6/8 = 0.750   (1.333 codewords per weight)

------------------------------------------------------------------------------
PROVENANCE OF THE NUMBERS  --  read this before quoting a result
------------------------------------------------------------------------------
* Reconstruction energy is read at runtime from EMBEDDED_ECC_N7_results.json
  (copied next to this script from
   C:\\profitional\\design_compiler\\dc_energy_starter\\results\\).
  That DC run synthesized the (7, 4) systematic-XOR encoder + reconstruction
  stage: incremental 0.1944379 pJ/codeword, idle 0.2107911 pJ/cycle,
  FreePDK45/OSU gscl45nm, 1.1 V, 1 ns clock, pre-layout with a wire-load model.

  ** MISMATCH TO BE AWARE OF **  the synthesized circuit is (7, 4); the codeword
  geometry modelled here for embedded/Recon+ is (7, 6).  The per-codeword
  energy is therefore an upper-bound proxy: a (7, 6) single-parity datapath is
  strictly cheaper than a (7, 4) Hamming datapath.  Re-run Design Compiler for
  (7, 6) and drop the JSON in beside this script -- set RECON_CONFIG_ID to the
  new configuration_id and nothing else changes.

* E_DECODE_BASE_PJ and E_DECODE_EMB_PJ are INACTIVE by default
  (INCLUDE_ECC_DECODE = False). They are ESTIMATES, not measurements.
  The parent scripts hardcoded 40 pJ/codeword, which was sized for a BCH(63)/
  BCH(255) decoder and is far too expensive for a Hamming-class code.  The
  values below are scaled to SEC/parity-class logic.  They are the single
  biggest modelling assumption in this script -- SENSITIVITY is printed at the
  end of every run so you can see how much of the saving they drive.

------------------------------------------------------------------------------
RUN INSIDE the container, launched from Energy_modeling/ :
    cd /home/workspace
    python3 E3-CODE/generate_models.py                      # once, if needed
    python3 E3-CODE/Proposal/run_ecc_multimodel_n7.py       # this script

Launching from /home/workspace is REQUIRED: the mapper cache lives at
ecc_energy_study/outputs/<arch>/multimodel/ relative to the launch directory.
Both eyeriss_like and simple_weight_stationary are already fully cached, so a
complete 8-model run should finish in seconds without invoking the mapper.

Outputs land NEXT TO THIS FILE (E3-CODE/Proposal/), not in ecc_energy_study/.
------------------------------------------------------------------------------
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
SeperateBreakdown = False     # True  -> GB & Local each split read+write
                              # False -> single bar per on-chip level
# =============================================================================

# ============================ USER KNOBS =====================================
ARCHS = ["eyeriss_like", "simple_weight_stationary"]   # one figure + CSV each
# Add "eyeriss_v2_like" above to include the Eyeriss v2 model. NOTE: there is no
# mapper cache for it yet, so the FIRST run maps all ~221 layer shapes from
# scratch (hours, not seconds). After that it is cached like the others.

MODELS_TO_RUN = [
    "resnet18",
    "resnet50",
    "densenet121",
    "squeezenet1_1",
    "mobilenet_v2",
    "efficientnet_b0",
    "convnext_tiny",
    "xception",
]

WEIGHT_BITS = 8               # 8-bit quantization
VICTORY     = 500             # mapper effort (only used if a shape is uncached)

# ---- codeword geometry, per approach ----------------------------------------
BASE_N, BASE_K = 11, 7        # shortened Hamming SEC, d_min = 3, t = 1
EMB_N,  EMB_K  =  7, 6        # 6 payload + 1 embedded parity, 7 bits stored
REC_N,  REC_K  =  7, 6        # Recon+ reuses the embedded codeword

# ---- controller decode energy ----------------------------------------------
# INCLUDE_ECC_DECODE = False  -> decode is charged as ZERO for all three arms.
# The category then sums to 0 across every model, so plot_arch()'s `active`
# filter drops it from the stacks AND from the legend automatically.
# Rationale for zero: the decode logic is shared/amortized and, when it was
# modelled, it came to 0.3-0.8% of total energy on both architectures -- far
# below the resolution of the rest of the model. Set True to restore it.
INCLUDE_ECC_DECODE = False

# The values below only take effect when INCLUDE_ECC_DECODE is True.
# (ESTIMATES -- see PROVENANCE in the docstring)
# Parent scripts used 40 pJ/cw for BCH(63)/BCH(255).  A shortened-Hamming SEC
# decoder is syndrome generation + a small correction mux; a single-parity check
# is one XOR tree.  Both scaled accordingly.
# TODO: replace with Design Compiler measurements for (11,7) and (7,6).
E_DECODE_BASE_PJ = 3.0        # pJ per codeword, baseline (11,7) SEC decode
E_DECODE_EMB_PJ  = 0.3        # pJ per codeword, embedded (7,6) parity check

# Does the Recon+ arm still pay a detection cost before reconstructing?
# True  -> Recon+ charges E_DECODE_EMB_PJ as well as reconstruction
#          (conservative; matches Run_ecc_singlemodel_ksweep.py's code).
# False -> reconstruction fully replaces decode
#          (optimistic; matches Run_ecc_multimodel_3way.py's code).
RECON_CHARGES_DECODE = True

# Does the baseline's parity also inflate ON-CHIP traffic, or only DRAM?
# The parent scripts inflate DRAM only; kept False for continuity with the
# existing figures.  Set True to also inflate the weight portion on-chip.
BASELINE_INFLATES_ONCHIP = False

# ---- reconstruction energy, from the Design Compiler run ---------------------
RECON_JSON         = "EMBEDDED_ECC_N7_results.json"   # sits next to this script
RECON_CONFIG_ID    = "EMBEDDED_ECC_7_4"
RECON_INCLUDE_IDLE = True     # True -> incremental + idle (0.405229 pJ/cw)
                              # False -> incremental only  (0.194438 pJ/cw)
# Fallback if the JSON is missing (same values, hardcoded)
RECON_INCREMENTAL_FALLBACK_PJ = 0.1944379
RECON_IDLE_FALLBACK_PJ        = 0.2107911
# =============================================================================


# ---------------- setup: repo + design paths ---------------------------------
assert shutil.which("timeloop-mapper"), "timeloop-mapper not on PATH -- run inside the container"

WORK = pathlib.Path.cwd() / "ecc_energy_study"
WORK.mkdir(exist_ok=True)
try:
    FIG_DIR = pathlib.Path(__file__).resolve().parent      # E3-CODE/Proposal/
except NameError:
    FIG_DIR = WORK
FIG_DIR.mkdir(parents=True, exist_ok=True)

EX_REPO = WORK / "timeloop-accelergy-exercises"
assert EX_REPO.exists(), ("exercises repo missing -- run E3-CODE/run_ecc_study.py once "
                          "first to clone it, and launch from /home/workspace")
DESIGNS_DIR = EX_REPO / "workspace" / "example_designs" / "example_designs"

# ---- install locally-authored architectures into the (re-clonable) repo ------
# Master copies live next to this script so they survive a wipe of
# ecc_energy_study/. Copied in whenever they are missing or newer.
def install_local_archs():
    import shutil as _sh
    for master in sorted(FIG_DIR.glob("*_like")):
        if not master.is_dir():
            continue
        dst = DESIGNS_DIR / master.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in master.iterdir():
            if not f.is_file():
                continue
            t = dst / f.name
            if (not t.exists()) or f.stat().st_mtime > t.stat().st_mtime:
                _sh.copy2(f, t)
                print(f"  [install] {master.name}/{f.name} -> example_designs/")

install_local_archs()
assert (WORK / "model_layers.json").exists(), \
    "model_layers.json missing -- run E3-CODE/generate_models.py first"
model_layers = json.loads((WORK / "model_layers.json").read_text())

(WORK / "globals.yaml").write_text(
    'variables:\n  version: 0.4\n  global_cycle_seconds: 1e-9\n  technology: "65nm"\n')


# ---------------- reconstruction energy ---------------------------------------
def load_recon_energy():
    """Return (E_recon_per_codeword_pJ, provenance_string)."""
    path = FIG_DIR / RECON_JSON
    if path.exists():
        try:
            entries = json.loads(path.read_text())
            if isinstance(entries, dict):
                entries = [entries]
            for e in entries:
                if e.get("configuration_id") == RECON_CONFIG_ID:
                    inc  = float(e["energy_pJ"]["incremental_per_codeword"])
                    idle = float(e["energy_pJ"]["idle_per_cycle"])
                    val  = inc + idle if RECON_INCLUDE_IDLE else inc
                    how  = "incremental+idle" if RECON_INCLUDE_IDLE else "incremental only"
                    return val, (f"{path.name} [{RECON_CONFIG_ID}] "
                                 f"n={e.get('n')} k={e.get('k')} t={e.get('t')} ({how}): "
                                 f"inc={inc:.7f} idle={idle:.7f}")
            print(f"[warn] {RECON_CONFIG_ID} not found in {path.name}; using fallback")
        except Exception as exc:
            print(f"[warn] could not parse {path.name}: {exc}; using fallback")
    else:
        print(f"[warn] {path.name} not found next to this script; using fallback")
    val = (RECON_INCREMENTAL_FALLBACK_PJ +
           (RECON_IDLE_FALLBACK_PJ if RECON_INCLUDE_IDLE else 0.0))
    return val, "hardcoded fallback constants"

E_RECON_CW_PJ, RECON_PROVENANCE = load_recon_energy()


# ---------------- per-architecture timeloop plumbing --------------------------
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
PROB_DIR = WORK / "problems_multimodel"
PROB_DIR.mkdir(exist_ok=True)


def patched_arch_path(arch):
    """Same DRAM-depth patch the parent scripts apply, per architecture."""
    src = DESIGNS_DIR / arch / "arch.yaml"
    dst = WORK / f"arch_{arch}_patched.yaml"
    text = src.read_text()
    dram_attrs = text.split("name: DRAM")[1].split("!")[0] if "name: DRAM" in text else ""
    if "depth" not in dram_attrs:
        text = text.replace(
            'class: DRAM\n    attributes:\n      type: "LPDDR4"',
            'class: DRAM\n    attributes:\n      depth: 1048576\n      type: "LPDDR4"')
    dst.write_text(text)
    return dst


def make_problem(sig):
    C, M, R, S, P, Q, Ws, Hs = sig
    name = f"C{C}_M{M}_R{R}_S{S}_P{P}_Q{Q}_ws{Ws}_hs{Hs}"
    path = PROB_DIR / f"{name}.yaml"
    if not path.exists():
        path.write_text(PROBLEM_TEMPLATE.format(C=C, M=M, R=R, S=S, N=1, P=P, Q=Q,
                                                Wstride=Ws, Hstride=Hs))
    return name, path


def design_inputs(arch_yaml, problem_yaml):
    files = [str(arch_yaml)]
    files += sorted(glob.glob(str(DESIGNS_DIR / "_components" / "*.yaml")))
    files.append(str(DESIGNS_DIR / "_include" / "mapper.yaml"))
    files.append(str(WORK / "globals.yaml"))
    files.append(str(problem_yaml))
    return files


_sig_cache = {}          # (arch, sig) -> stats path or None


def run_shape(arch, arch_yaml, out_root, sig):
    key = (arch, sig)
    if key in _sig_cache:
        return _sig_cache[key]
    name, prob = make_problem(sig)
    out_dir = out_root / name
    stats = out_dir / "timeloop-mapper.stats.txt"
    if stats.exists():                       # cache hit -- the common path
        _sig_cache[key] = stats
        return stats
    spec = tl.Specification.from_yaml_files(*design_inputs(arch_yaml, prob))
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
        _sig_cache[key] = None
        return None
    _sig_cache[key] = stats if stats.exists() else None
    return _sig_cache[key]


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


def gather_model(arch, arch_yaml, out_root, model):
    rows, n_ok, n_skip = [], 0, 0
    n = len(model_layers[model])
    for i, L in enumerate(model_layers[model]):
        sig = (int(L["C"]), int(L["M"]), int(L["R"]), int(L["S"]),
               int(L["P"]), int(L["Q"]), int(L["Wstride"]), int(L["Hstride"]))
        stats = run_shape(arch, arch_yaml, out_root, sig)
        if stats is None:
            n_skip += 1
            continue
        rows += parse_stats(stats, f"{model}_{i}")
        n_ok += 1
    print(f"  {model:16s} {n_ok}/{n} layers ok, {n_skip} skipped", flush=True)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df[df.energy_pJ.notna() & (df.energy_pJ > 0)].reset_index(drop=True)
    df["category"] = df.level.map(classify)
    base   = category_energy(df)                             # all dataspaces
    base_w = category_energy(df[df.dataspace == "Weights"])  # WEIGHTS-only
    wt = df[df.dataspace == "Weights"].groupby("category").energy_pJ.sum()
    E_DRAM_W = float(wt.get("DRAM", 0.0))
    DRAM_W_READS = float(df[(df.category == "DRAM") &
                            (df.dataspace == "Weights")].reads.sum())
    return dict(base=base, base_w=base_w, E_DRAM_W=E_DRAM_W,
                DRAM_W_READS=DRAM_W_READS, ok=n_ok, skip=n_skip)


# ---------------- apply the three fixed-geometry ECC schemes -----------------
def build_stacks(raw):
    """No K sweep: the three arms have fixed, different codeword geometry."""
    base, base_w = raw["base"], raw["base_w"]
    E_DRAM_W, DRAM_W_READS = raw["E_DRAM_W"], raw["DRAM_W_READS"]

    # weights carried by one codeword = K bits / 8 bits per weight
    wpc_base = BASE_K / WEIGHT_BITS          # 0.875
    wpc_emb  = EMB_K  / WEIGHT_BITS          # 0.750
    wpc_rec  = REC_K  / WEIGHT_BITS          # 0.750

    # codeword counts, charged per weight LOADED from DRAM
    n_cw_base = DRAM_W_READS / wpc_base
    n_cw_emb  = DRAM_W_READS / wpc_emb
    n_cw_rec  = DRAM_W_READS / wpc_rec

    parity_frac = BASE_N / BASE_K - 1.0      # 11/7 - 1 = 0.5714 DRAM inflation
    sram_scale  = REC_K / REC_N              # 6/7 = 0.8571 weight data on-chip

    # ---- 1. baseline: shortened Hamming (11,7), parity stored in DRAM -------
    baseline = base.copy()
    baseline["DRAM"] += E_DRAM_W * parity_frac
    if BASELINE_INFLATES_ONCHIP:
        for c in ONCHIP_CATS:
            baseline[c] = (base[c] - base_w[c]) + base_w[c] * (BASE_N / BASE_K)
    baseline["ECC decode"] = (n_cw_base * E_DECODE_BASE_PJ) if INCLUDE_ECC_DECODE else 0.0

    # ---- 2. embedded: (7,6), parity lives inside the stored word ------------
    embedded = base.copy()
    embedded["ECC decode"] = (n_cw_emb * E_DECODE_EMB_PJ) if INCLUDE_ECC_DECODE else 0.0

    # ---- 3. Recon+: (7,6) codeword, parity regenerated on-chip --------------
    recon = base.copy()
    for c in ONCHIP_CATS:
        # scale ONLY the weight portion of on-chip energy by K/N; I/O untouched.
        recon[c] = (base[c] - base_w[c]) + base_w[c] * sram_scale
    recon["ECC decode"]     = ((n_cw_rec * E_DECODE_EMB_PJ)
                               if (INCLUDE_ECC_DECODE and RECON_CHARGES_DECODE) else 0.0)
    recon["Reconstruction"] = n_cw_rec * E_RECON_CW_PJ

    return pd.DataFrame({"baseline_hamming": baseline,
                         "embedded_ecc":     embedded,
                         "recon_plus":       recon})


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
# (dataframe column, bar tag drawn under the bar, CSV column prefix)
COLS = [("baseline_hamming", "Base.",  "base"),
        ("embedded_ecc",     "Embe.",  "embe"),
        ("recon_plus",       "Recon+", "recon")]

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "axes.linewidth": 2.0,
    "axes.edgecolor": "#2b2b2b", "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------- one figure per architecture --------------------------------
def plot_arch(arch, stacks):
    models = list(stacks.keys())
    n = len(models)

    maxtot_pJ = max(stacks[m][col].sum() for m in models for col, _t, _p in COLS)
    div, unit = (1e9, "mJ") if maxtot_pJ / 1e6 >= 1000 else (1e6, "\u00b5J")

    active = [c for c in plot_cats
              if sum(stacks[m].loc[c].sum() for m in models) > 1e-9]

    w, gap  = 0.34, 0.06
    centers = np.arange(n) * 2.40
    xoff    = {"baseline_hamming": -(w + gap), "embedded_ecc": 0.0, "recon_plus": (w + gap)}

    fig, ax = plt.subplots(figsize=(4.4 * n + 4, 10))
    bottoms = {col: np.zeros(n) for col, _t, _p in COLS}
    for cat in active:
        for col, _t, _p in COLS:
            v = np.array([stacks[m].loc[cat, col] for m in models]) / div
            ax.bar(centers + xoff[col], v, w, bottom=bottoms[col], color=palette[cat],
                   edgecolor="white", linewidth=1.0, zorder=3,
                   label=nice_label.get(cat, cat) if col == "baseline_hamming" else None)
            bottoms[col] += v

    ymax = max(bottoms[col].max() for col, _t, _p in COLS)

    # totals on each bar + saving% for embedded & Recon+ vs baseline
    for i, m in enumerate(models):
        tb = bottoms["baseline_hamming"][i]
        for col, _t, _p in COLS:
            tot = bottoms[col][i]
            ax.text(centers[i] + xoff[col], tot + ymax * 0.006, f"{tot:.0f}",
                    ha="center", va="bottom", fontsize=12, fontweight="bold", color="#2b2b2b")
        for col in ("embedded_ecc", "recon_plus"):
            sv = (tb - bottoms[col][i]) / tb * 100 if tb > 0 else 0.0
            ax.text(centers[i] + xoff[col], bottoms[col][i] + ymax * 0.045,
                    f"\u2212{sv:.1f}%", ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color="#B03A2E")

    # x labels: per-bar tag (rotated) + model name centred
    ax.set_xticks([])
    for i, m in enumerate(models):
        for col, tag, _p in COLS:
            ax.text(centers[i] + xoff[col], -ymax * 0.015, tag, ha="center", va="top",
                    rotation=90, fontsize=15, color="#555", clip_on=False)
        ax.text(centers[i], -ymax * 0.27, m, ha="center", va="top",
                fontsize=20, fontweight="medium", color="#111", clip_on=False)

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
    ax.set_title(f"{arch}  \u00b7  8-bit weights  \u00b7  "
                 f"Baseline Hamming({BASE_N},{BASE_K}) vs Embedded({EMB_N},{EMB_K}) "
                 f"vs Recon+({REC_N},{REC_K})",
                 fontsize=22, pad=34 + 30 * legend_rows, fontweight="medium")

    fig.subplots_adjust(bottom=0.30, top=0.83, left=0.07, right=0.99)
    suffix = "_rw_split" if SeperateBreakdown else ""
    stem = f"multimodel_n7_{arch}{suffix}"
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    plt.savefig(png, dpi=400, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()

    # CSV summary (uJ)
    rows = {}
    for m in models:
        st = stacks[m]
        row = {}
        for c in plot_cats:
            for col, _tag, pfx in COLS:
                row[f"{pfx}_{c}"] = st.loc[c, col] / 1e6
        tb = st["baseline_hamming"].sum()
        for col, _tag, pfx in COLS:
            tt = st[col].sum()
            row[f"{pfx}_total_uJ"]   = tt / 1e6
            row[f"{pfx}_saving_pct"] = (tb - tt) / tb * 100 if tb > 0 else 0.0
        rows[m] = row
    csv = FIG_DIR / f"{stem}_summary_uJ.csv"
    pd.DataFrame(rows).T.to_csv(csv)
    return png, pdf, csv


# ================================ RUN =========================================
print("=" * 78)
print("N=7 PROPOSAL SWEEP  --  baseline Hamming(11,7) / embedded (7,6) / Recon+ (7,6)")
print("=" * 78)
print(f"  quantization          : {WEIGHT_BITS}-bit weights")
print(f"  weights per codeword  : base {BASE_K/WEIGHT_BITS:.3f} | "
      f"emb {EMB_K/WEIGHT_BITS:.3f} | Recon+ {REC_K/WEIGHT_BITS:.3f}")
print(f"  DRAM weight inflation : baseline x{BASE_N/BASE_K:.4f} "
      f"(+{(BASE_N/BASE_K-1)*100:.1f}%), embedded/Recon+ x1.0")
print(f"  Recon+ on-chip scale  : {REC_K/REC_N:.4f} (weights only)")
if INCLUDE_ECC_DECODE:
    print(f"  decode  base / emb    : {E_DECODE_BASE_PJ} / {E_DECODE_EMB_PJ} pJ per codeword  [ESTIMATE]")
    print(f"  Recon+ pays decode    : {RECON_CHARGES_DECODE}")
else:
    print( "  decode                : 0 pJ for ALL THREE arms "
           "(INCLUDE_ECC_DECODE=False; category hidden from the legend)")
print(f"  reconstruction        : {E_RECON_CW_PJ:.7f} pJ per codeword")
print(f"  recon provenance      : {RECON_PROVENANCE}")
print(f"  output directory      : {FIG_DIR}")
print("=" * 78)

summary_all = {}
for arch in ARCHS:
    print(f"\n########## architecture: {arch} ##########", flush=True)
    arch_yaml = patched_arch_path(arch)
    out_root  = WORK / "outputs" / arch / "multimodel"
    out_root.mkdir(parents=True, exist_ok=True)

    raws = {}
    for model in MODELS_TO_RUN:
        if model not in model_layers:
            print(f"  [skip] {model}: not in model_layers.json")
            continue
        r = gather_model(arch, arch_yaml, out_root, model)
        if r is None:
            print(f"  !! {model}: no valid layers, skipped")
            continue
        raws[model] = r

    if not raws:
        print(f"  !! {arch}: no models produced results; skipping figure")
        continue

    stacks = {m: build_stacks(raws[m]) for m in raws}

    print(f"\n  --- {arch}: energy vs baseline ---")
    print(f"  {'model':16s} {'base uJ':>10s} {'emb uJ':>10s} {'recon+ uJ':>10s} "
          f"{'emb %':>8s} {'recon+ %':>9s}")
    for m in stacks:
        st = stacks[m]
        tb = st["baseline_hamming"].sum()
        te = st["embedded_ecc"].sum()
        tp = st["recon_plus"].sum()
        print(f"  {m:16s} {tb/1e6:10.2f} {te/1e6:10.2f} {tp/1e6:10.2f} "
              f"{(tb-te)/tb*100:7.1f}% {(tb-tp)/tb*100:8.1f}%")
        summary_all[(arch, m)] = ((tb - te) / tb * 100, (tb - tp) / tb * 100)

    png, pdf, csv = plot_arch(arch, stacks)
    print(f"\n  saved {png.name}")
    print(f"  saved {pdf.name}")
    print(f"  saved {csv.name}")

    # ---- where does the Recon+ cost actually sit? --------------------------
    print(f"\n  --- {arch}: composition check ---")
    m0 = list(stacks.keys())[0]
    st = stacks[m0]
    tb = st["baseline_hamming"].sum()
    rec_share = st.loc["Reconstruction", "recon_plus"] / st["recon_plus"].sum() * 100
    if INCLUDE_ECC_DECODE:
        dec_share = st.loc["ECC decode", "baseline_hamming"] / tb * 100 if tb > 0 else 0.0
        print(f"  {m0}: baseline ECC decode is {dec_share:.1f}% of baseline total")
        print(f"  -> if E_DECODE_BASE_PJ is wrong by 2x, the embedded/Recon+ saving "
              f"moves by roughly {dec_share/2:.1f} points.")
    else:
        print(f"  {m0}: ECC decode charged as 0 for all three arms (excluded from plot)")
    print(f"  {m0}: reconstruction is {rec_share:.1f}% of Recon+ total")
    print(f"  -> the saving is therefore driven by DRAM parity inflation "
          f"({(BASE_N/BASE_K-1)*100:.1f}%) and the Recon+ on-chip scale "
          f"({REC_K/REC_N:.4f}), which are pure geometry.")

print("\n" + "=" * 78)
print("Done. Files written to:", FIG_DIR)
if not INCLUDE_ECC_DECODE:
    print("NOTE: ECC decode charged as ZERO for all three arms; category omitted")
    print("      from the stacks and the legend. Set INCLUDE_ECC_DECODE=True to restore.")
else:
    print("REMINDER: E_DECODE_BASE_PJ / E_DECODE_EMB_PJ are estimates, not measurements.")
print("Reconstruction energy comes from a (7,4) DC run applied to a (7,6) codeword")
print("count -- see the PROVENANCE block at the top of this file before quoting.")
print("=" * 78)
