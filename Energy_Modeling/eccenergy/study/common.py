"""Shared setup and the per-architecture collection loop.

Every run needs the same four things: a validated environment, the architectures
installed and patched, the workload loaded, and raw energies collected per
architecture. That all lives here, so `sweep.py` is only about what goes on the
x axis.
"""
from __future__ import annotations

from ..arch import fingerprint as fingerprint_mod
from ..arch import layout
from ..arch import load
from ..toolchain import ert
from ..toolchain import inputs
from ..toolchain import invoke
from ..config import BRACKET_PAIRS
from .stacks import build_stacks, load_recon_energy, savings
from .energy import collect
from ..paths import Results
from ..arch.workloads import layer_identities, load_workload, select, select_layers
from ..settings import guards


class Session:
    """One configured run: paths, workload, raw energies, ECC stacks."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.results = Results(cfg).prepare()
        # prompt_6 RULE 3: incremental (pJ/codeword) and idle (pJ/cycle/engine)
        self.recon_pj, self.recon_idle_pj, self.recon_provenance = load_recon_energy(cfg)
        self.workload = None
        self.models = None
        self.meta = {}
        self.raws = {}     # arch -> {model: Raw}
        self.stacks = {}   # arch -> {model: DataFrame}   (at the held K)
        self.notes = []
        self.fingerprints = {}   # arch -> content hash of the mapped YAML
        self.mappers = {}        # arch -> the Mapper, for its mapping records
        self.layer_ids = {}      # model -> [layer identity dicts]

    # ------------------------------------------------------------------ setup
    def setup(self, need_mapper=None):
        """Prepare the environment. `need_mapper` defaults to "not replot-only"."""
        cfg = self.cfg
        if need_mapper is None:
            need_mapper = not cfg.replot_only

        if need_mapper and not cfg.from_cache:
            inputs.require_container()
            inputs.ensure_exercises_repo()
            load.install_local_archs()
            # ONE globals.yaml PER DESIGN (prompt_7 C1.5): the clock is a
            # per-architecture number now, so a shared file would run every
            # design on a multi-design figure at whichever rate was written
            # last. Written here, once, for every design in the run.
            for arch in cfg.archs:
                path, node = layout.write_globals(cfg, arch)
                # The name carries the content hash, so "already there" is
                # proof the bytes are right -- not a reason to rewrite.
                print(f"  {path.name}: technology={node} "
                      f"cycle={cfg.cycle_seconds_for(arch)} "
                      f"({cfg.clock_mhz_for(arch):g} MHz)")
        elif need_mapper:
            print("  ECC_FROM_CACHE=1: reading solved mappings only, "
                  "Timeloop will not be invoked")

        workload, self.meta = load_workload(cfg)
        self.workload = workload
        self.models = select(workload, cfg.models)
        # DEVELOPMENT MODE. Selecting layers here, once, means every downstream
        # consumer -- mapper, raw cache, results writer -- sees only the chosen
        # layers and cannot accidentally mix a full-model number into a
        # two-layer result. The scope is in every path they write to.
        # ONE RESULT FILE CARRIES ONE LAYER SCOPE. `ECC_LAYERS`' per-model
        # spelling (EnvReorganisation 6.9) names a different scope per
        # network, and `select_layers` below applies ONE list to every model
        # -- as do `paths.layer_slug` and the results namespace. So a run that
        # evaluates several models under a per-model table is refused HERE,
        # where the scope is consumed, rather than in `config._resolve()`:
        # `toolchain.units` derives one configuration per model from exactly
        # such a run to enumerate the mapper work, and that has to resolve.
        if cfg.layers_by_model and len(self.models) > 1:
            raise guards.refusal("layers-per-model-one-model",
                "ECC_LAYERS is the per-model form ("
                + "; ".join(f"{m}={len(v)}" for m, v in cfg.layers_by_model.items())
                + f") but this run evaluates {len(self.models)} models "
                + f"({', '.join(self.models)}), and one result file carries "
                f"ONE layer scope.\n"
                f"  -> map them with `bash hpc/run_all.sh`, which pins one "
                f"model per unit\n"
                f"  -> evaluate one model at a time (ECC_CONST_MODEL=<one>)\n"
                f"  -> a scope PER MODEL inside one result is "
                f"EnvReorganisation phase 6")
        self.models = select_layers(self.models, cfg.layers)
        self.layer_ids = layer_identities(self.models)
        if cfg.layers:
            print(f"  [scope] {cfg.layer_scope} run -- results are written under "
                  f"{cfg.layer_slug}/ and are NOT full-model numbers")
        if self.meta:
            print(f"  workload meta: {self.meta}")

        # A design whose result is a BOUND rather than a measurement says so
        # here, in the run manifest, and on the console -- not only in a README
        # a reader of the CSV will never open. See config.BRACKET_PAIRS.
        for arch in cfg.archs:
            partner, why = BRACKET_PAIRS.get(arch, (None, None))
            if partner and partner not in cfg.archs:
                note = (f"{arch}: {why}. Add {partner} to the run to bracket "
                        f"it -- see archs/{partner}/README.md.")
                self.notes.append(note)
                print(f"  [bound] {note}")
        return self

    # ------------------------------------------------------------- collection
    def collect_arch(self, arch):
        cfg = self.cfg
        print(f"\n########## architecture: {arch} ##########", flush=True)

        # The treatment slug is per-architecture: forcing the datawidth is a
        # no-op on a design already declared at the weight width, and such an
        # architecture must keep its existing (expensive) mapper cache.
        variant = fingerprint_mod.effective_variant(arch, cfg)
        if variant != cfg.arch_variant_slug:
            print(f"  treatment {cfg.arch_variant_slug!r} does not change this "
                  f"architecture -> reusing the {variant!r} mapper cache")

        # The cache is content-addressed on the architecture YAML the mapper
        # will actually see. Editing a YAML therefore moves the cache instead
        # of silently reusing mappings computed for the previous geometry.
        fingerprint = fingerprint_mod.arch_fingerprint(arch, cfg)
        self.fingerprints[arch] = fingerprint

        # Mapper effort scales with this architecture's loop-nest depth, so a
        # deep hierarchy is not searched less thoroughly than a shallow one.
        levels = None
        if not cfg.from_cache:
            try:
                levels = layout.arch_levels(arch, cfg)
            except FileNotFoundError:
                pass
            else:
                print(f"  source {load.arch_source(arch, cfg).name}  "
                      f"({cfg.arch_fidelity} fidelity, {levels} loop levels)  "
                      f"-> victory {cfg.victory_for(levels)}, "
                      f"objective {cfg.opt_metric}")

        # prompt_6: the ERT toll of this arm, None for the reference. In the
        # fingerprint already; the Mapper also records it in every sidecar,
        # requires it to match on a hit, and reads it back after a map.
        bump = fingerprint_mod.ert_bump(arch, cfg)
        if bump is not None:
            print(f"  ERT arm {ert.describe_bump(bump)}  "
                  f"(block_size {bump['block_size']}, E_w {bump['e_w_pj']:.6f} pJ/weight; "
                  f"narrow levels {'+'.join(bump['narrow_levels'])})")

        def factory():
            arch_yaml = (layout.patched_arch_path(arch, cfg)
                         if not cfg.from_cache else None)
            return invoke.Mapper(
                cfg, arch, arch_yaml,
                self.results.mapper_cache(arch, variant, fingerprint), levels,
                fingerprint=fingerprint,
                legacy_root=self.results.legacy_mapper_cache(arch, variant),
                ert_bump=bump)

        raws, mapper = collect(cfg, self.results, arch, factory, self.models,
                               variant, fingerprint)
        if mapper is not None:
            self.mappers[arch] = mapper
        if not raws:
            print(f"  !! {arch}: no results")
            return None
        self.raws[arch] = raws
        self.stacks[arch] = {m: build_stacks(cfg, r, self.recon_pj,
                                             recon_idle_pj=self.recon_idle_pj)
                             for m, r in raws.items()}
        for model, raw in raws.items():
            print(f"  {model:18s} raw {raw.total / 1e6:12,.2f} uJ   "
                  f"DRAM weight reads {raw.dram_w_reads:>14,.0f}")
        return self.stacks[arch]

    def chip_dir(self, arch):
        """The mapper cache directory of the chip `arch` was collected at.

        `ECC_METRICS=area` reads the ART out of it. `create=False` because this
        is a pure READER: resolving a path used to CREATE it, and a report that
        asks about a chip it has not mapped would leave an empty
        `fp-<hash>/` behind claiming a mapping that was never solved
        (`paths.mapper_cache`).

        Returns None when this architecture has not been collected -- the
        caller refuses by name (`area-chip-never-mapped`) rather than charging
        an accelerator with no silicon in it as zero.
        """
        fp = self.fingerprints.get(arch)
        if fp is None:
            return None
        variant = fingerprint_mod.effective_variant(arch, self.cfg)
        return self.results.mapper_cache(arch, variant, fp, create=False)

    def collect_all(self):
        for arch in self.cfg.archs:
            self.collect_arch(arch)
        if not self.stacks:
            cfg = self.cfg
            msg = ["nothing collected.",
                   f"  treatment: {cfg.arch_variant_slug}"]
            if cfg.arch_variant_slug != "stock":
                # By far the likeliest cause, now that the defaults are no
                # longer the historical ones: results exist, but for a
                # different treatment, and mixing them would be wrong.
                msg += [
                    "  -> this treatment has its own cache, so results collected under a",
                    "     DIFFERENT one are deliberately not reused. Either run it in the",
                    "     container to build the cache, or ask for the treatment you have:",
                    "",
                    "       ECC_ARCH_FIDELITY=stock ECC_OPT_METRIC=edp ECC_VICTORY=500 \\",
                    "         ECC_VICTORY_SCALING=none bash run.sh --replot",
                    "",
                    "     which is the pre-correction setting and reads the original cache."]
            msg += [
                "  -> with ECC_REPLOT_ONLY=1 you need results/_raw/ populated by an "
                "earlier in-container run",
                "  -> otherwise check the mapper output under ecc_energy_study/outputs/"]
            raise guards.refusal("stacks-refused",
                "\n".join(msg))
        return self.stacks

    # ---------------------------------------------------------------- reports
    def report(self, groups, stacks, labels):
        """The swept axis, one row per group: totals per arm and the savings."""
        cfg = self.cfg
        ref = cfg.bar_arms[0]
        print(f"\n  --- {cfg.swept_axis} sweep, reference arm = {ref} ---")
        head = f"  {'group':22s}" + "".join(f"{a + ' uJ':>13s}" for a in cfg.bar_arms)
        head += "".join(f"{a + ' %':>12s}" for a in cfg.bar_arms[1:])
        print(head)
        for g in groups:
            st = stacks[g]
            pcts = savings(st, cfg.bar_arms)
            name = str(labels.get(g, g)).replace("\n", " ")
            line = f"  {name[:22]:22s}"
            line += "".join(f"{float(st[a].sum()) / 1e6:13,.2f}" for a in cfg.bar_arms)
            line += "".join(f"{pcts[a]:11.1f}%" for a in cfg.bar_arms[1:])
            print(line)

        if "recon" in cfg.bar_arms:
            st = stacks[groups[0]]
            total = float(st["recon"].sum())
            share = float(st.loc["Reconstruction", "recon"]) / total * 100 if total else 0.0
            first = str(labels.get(groups[0], groups[0])).replace("\n", " ")
            print(f"  reconstruction is {share:.2f}% of the recon+ total at {first}")

    def finish(self, figures, csv=None, groups=None, extra=None):
        """Write the manifest beside the figure. `extra` is what a stage wants
        recorded at the top level as well as inside `config` -- the placement
        study puts the decode site and the DRAM pJ/bit there, because they decide what
        the DRAM band on disk means."""
        payload = {
            "recon_pj_per_codeword": self.recon_pj,
            "recon_idle_pj_per_cycle_per_engine": self.recon_idle_pj,
            "recon_idle_engines_in_build_stacks": 1,
            "recon_idle_note": ("prompt_6 RULE 3: Reconstruction = codewords x "
                                "incremental + idle x cycles x engines; the sweep's "
                                "recon arm counts its codewords from DRAM reads, so "
                                "it reconstructs at the chip ingress with one engine"),
            "recon_provenance": self.recon_provenance,
            "workload_meta": self.meta,
            "swept_axis": self.cfg.sweep,
            "swept_values": [str(g) for g in (groups or [])],
            "figures": [str(p) for p in figures],
            "table": str(csv) if csv else None,
            "notes": self.notes,
        }
        # What a MAC was charged: the denominator of every percentage on the
        # figure beside this manifest (energy.apply_mac_override).
        for raws in self.raws.values():
            for raw in raws.values():
                if getattr(raw, "mac", None):
                    payload["mac_energy"] = raw.mac
                    break
            if "mac_energy" in payload:
                break
        payload.update(extra or {})
        manifest = self.results.write_manifest(payload)
        base = self.results.base.parent
        print("\n" + "=" * 78)
        what = ("reconstruction-placement study" if self.cfg.experiment == "recon"
                else f"{self.cfg.sweep} sweep")
        print(f"Done. {what} -> {self.cfg.stem}")
        for p in list(figures) + ([csv] if csv else []) + [manifest]:
            try:
                print(f"  {p.relative_to(base)}")
            except ValueError:
                print(f"  {p}")
        if not self.cfg.decode_enabled:
            print("NOTE: ECC decode charged as ZERO for every arm; the category is "
                  "omitted from the stacks and the legend (ECC_DECODE=1 restores it).")
        print("Raw energies are cached in results/_raw/ -- set ECC_REPLOT_ONLY=1 to")
        print("redraw at different constants without touching the mapper.")
        print("=" * 78)
