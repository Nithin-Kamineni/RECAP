"""
Generate transformer / LLM weight-matmul layer shapes for the Timeloop energy
sweep -- the transformer analogue of generate_models.py.

Transformers are highly regular: each block is a fixed set of weight matmuls
fully determined by the model's dims. Rather than download checkpoints, we
build the shapes ANALYTICALLY from each model's config -- no HuggingFace, no
weights, no gated-model access. Every Linear weight matrix  W: in -> out  is
modeled as a 1x1 convolution  C=in, M=out, P=SEQ, Q=1  (a Linear IS a 1x1
conv), matching run_ecc_transformer_block.py.

Why analytical instead of forward hooks (like the CNN version):
  - HuggingFace GPT-2 stores weights in a custom Conv1D class, NOT nn.Linear,
    so nn.Linear hooks silently miss every GPT-2 matrix.
  - Llama-family checkpoints are gated + multi-GB; configs alone still need HF.
  - Analytical shapes correctly capture GQA (smaller K/V) and gated FFN.
If you later want exact shapes from a real checkpoint, hook BOTH nn.Linear and
transformers.pytorch_utils.Conv1D on a model built with AutoModel.from_config
(no weight download) -- ask and I'll provide that variant.

Writes ecc_energy_study/transformer_layers.json for run_llm_breakdown.py.

Run from Energy_modeling/ :
    python3 Transformers/generate_transformers.py
"""
import json, pathlib

# ================= KNOBS =====================================================
SEQ               = 1      # tokens at once. 1 = decode (memory-bound, weights
                           # dominate); >1 = prefill (more compute). The DRAM
                           # weight-energy / ECC story is clearest at SEQ=1.
INCLUDE_LM_HEAD   = True    # add final d->vocab projection (large matrix, often
                           # weight-tied to embedding). False = compare BLOCKS only.
INCLUDE_EMBEDDING = False  # token embedding is a lookup (one row per token), not
                           # a MAC-bound matmul; modeling it as vocab*d hugely
                           # overcounts, so it is off by default.

# family -> how one block decomposes into weight matmuls:
#   gpt2 : combined QKV (d->3d), out (d->d), ffn_up (d->ffn), ffn_down (ffn->d)
#   bert : separate Q,K,V (d->d each), out (d->d), ffn_up (d->ffn), ffn_down (ffn->d)
#   opt  : same shape set as bert (separate QKV, dense FFN)
#   llama: Q(d->d), K/V(d->kv_dim, GQA-aware), out(d->d),
#          gated FFN: gate & up (d->ffn), down (ffn->d)   [3 FFN matrices]
MODELS = {
  "distilgpt2":  dict(family="gpt2",  d=768,  layers=6,  heads=12, ffn=3072,  vocab=50257),   # ~82M
  "gpt2":        dict(family="gpt2",  d=768,  layers=12, heads=12, ffn=3072,  vocab=50257),   # ~124M
  "bert_base":   dict(family="bert",  d=768,  layers=12, heads=12, ffn=3072,  vocab=30522),   # ~110M
  "gpt2_medium": dict(family="gpt2",  d=1024, layers=24, heads=16, ffn=4096,  vocab=50257),   # ~355M
  "tinyllama":   dict(family="llama", d=2048, layers=22, heads=32, kv_heads=4, ffn=5632, vocab=32000),  # ~1.1B
}
# =============================================================================

def block_matmuls(cfg):
    """Return list of (name, in_features, out_features) for ONE block."""
    d, ffn, heads = cfg["d"], cfg["ffn"], cfg["heads"]
    head_dim = d // heads
    fam = cfg["family"]
    if fam == "gpt2":
        return [("attn_qkv", d, 3 * d), ("attn_out", d, d),
                ("ffn_up", d, ffn), ("ffn_down", ffn, d)]
    if fam in ("bert", "opt"):
        return [("attn_q", d, d), ("attn_k", d, d), ("attn_v", d, d), ("attn_out", d, d),
                ("ffn_up", d, ffn), ("ffn_down", ffn, d)]
    if fam == "llama":
        kv_dim = cfg.get("kv_heads", heads) * head_dim       # GQA -> smaller K/V
        return [("attn_q", d, d), ("attn_k", d, kv_dim), ("attn_v", d, kv_dim), ("attn_out", d, d),
                ("ffn_gate", d, ffn), ("ffn_up", d, ffn), ("ffn_down", ffn, d)]
    raise ValueError(f"unknown family: {fam}")

def layer_dict(name, c_in, c_out, count, kind):
    # 1x1 conv encoding of a Linear: C=in, M=out, spatial P=SEQ, Q=1
    return dict(name=name, C=int(c_in), M=int(c_out), R=1, S=1,
                P=int(SEQ), Q=1, Wstride=1, Hstride=1,
                count=int(count), kind=kind)

all_models = {}
for name, cfg in MODELS.items():
    layers = [layer_dict(nm, ci, co, cfg["layers"], "block")
              for nm, ci, co in block_matmuls(cfg)]
    if INCLUDE_LM_HEAD:
        layers.append(layer_dict("lm_head", cfg["d"], cfg["vocab"], 1, "head"))
    if INCLUDE_EMBEDDING:
        layers.append(layer_dict("embedding", cfg["d"], cfg["vocab"], 1, "embed"))
    all_models[name] = layers
    n_mm  = len([l for l in layers if l["kind"] == "block"])
    tot_w = sum(l["C"] * l["M"] * l["count"] for l in layers)
    print(f"{name:11s} family={cfg['family']:5s} blocks={cfg['layers']:2d} "
          f"matmuls/block={n_mm}  total weights ~ {tot_w/1e6:6.1f} M")

WORK = pathlib.Path.cwd() / "ecc_energy_study"; WORK.mkdir(exist_ok=True)
out = WORK / "transformer_layers.json"
out.write_text(json.dumps(
    dict(_meta=dict(SEQ=SEQ, include_lm_head=INCLUDE_LM_HEAD,
                    include_embedding=INCLUDE_EMBEDDING),
         models=all_models), indent=1))
print(f"\nSEQ={SEQ}  lm_head={INCLUDE_LM_HEAD}  embedding={INCLUDE_EMBEDDING}")
print(f"Saved {len(all_models)} models -> {out}")