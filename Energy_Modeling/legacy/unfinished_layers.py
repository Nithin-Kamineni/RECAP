"""TEMPORARY helper (2026-09-09): which layers of the recon model already have a
mapping in the CURRENT mapper cache. Prints them space-separated, as ECC_LAYERS
wants them. Read-only; imports the pipeline, changes nothing in it.

    bash hpc/tl.sh python3 legacy/unfinished_layers.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # project root

from eccenergy import archs as archmod
from eccenergy import config, workloads
from eccenergy.paths import Results

cfg = config.load_config()
arch, model = cfg.archs[0], cfg.models[0]
variant = archmod.effective_variant(arch, cfg)
fingerprint = archmod.arch_fingerprint(arch, cfg)
cache = Results(cfg).mapper_cache(arch, variant, fingerprint)
models, _ = workloads.load_workload(cfg)
layers = models[model]
done, wip = [], []
for l in layers:
    d = cache / l.shape_name
    if (d / "mapping.json").exists() and (d / "timeloop-mapper.stats.txt").exists():
        done.append(l.name)
    else:
        wip.append(l.name)
print(f"# cache: {cache}", file=sys.stderr)
print(f"# mapped {len(done)}/{len(layers)} layers ({len({l.shape_name for l in layers if l.name in done})} of "
      f"{len({l.shape_name for l in layers})} shapes); unmapped: {', '.join(wip)}", file=sys.stderr)
print(" ".join(done))
