"""Dump every architecture fingerprint the gate has to hold still.

ProjectRestructure.md section 9.1, gate item 1: `arch_fingerprint()` for EVERY
(arch, model) pair, byte-identical before and after a phase. A fingerprint names
a MAPPER CACHE DIRECTORY, so one that moves is hours of compute gone -- which is
why it is the gate's first item and the one that says STOP.

Run it the way CLAUDE.md documents, so env.sh alone decides what it sees:

    bash hpc/tl.sh bash -c 'source env.sh; PYTHONPATH=. python3 restructure/_fingerprints.py'

TWO GRIDS, NOT ONE CROSS PRODUCT. `arch_fingerprint()` takes no workload, so the
model axis is expected to be inert -- but "expected" is what a gate is for, and
grid A is the gate's literal wording. Grid B then walks the axes that DO reach
the patched YAML: the code rate and the mapper arm (`datawidth: q` on the levels
that arm narrows, plus its ERT bump). Their product would be ~4,300 rows saying
the same thing 15 times over.

    grid A   arch x model            at env.sh's K, arm=reference
    grid B   arch x K x mapper arm   at env.sh's model

A design with no placement table (`simba_like`, `simple_input_stationary`,
`simple_output_stationary` today -- ProjectRestructure section 5.2 keeps them that
way) has the reference arm and nothing else; that is RECORDED as `no-placement-
table`, not treated as an error, so phase 5 turning those tables into data can be
read off the diff.

Writes TSV to stdout and nothing else, so the gate can diff it with `diff`.
"""
import dataclasses
import sys

from eccenergy import archs, config, recon


def _arms(arch, cfg):
    """Every mapper arm of this design, or the reference alone if it declares none."""
    try:
        return [a.key for a in recon.mapper_arms(arch, cfg)], ""
    except KeyError:
        # Not an error: three designs declare no reconstruction boundaries today.
        return ["reference"], "no-placement-table"


def _row(arch, cfg_for, model, k, arm, note):
    """One TSV row, or the REFUSAL this (arch, K, arm) produces today.

    A configuration the code refuses is a fact about the code, and a phase that
    changes the refusal has changed behaviour. So the refusal is RECORDED in the
    same table rather than ending the dump -- the gate then diffs it like any
    other row. (Live example: `simple_weight_stationary` at recon5 narrows a
    `weight_reg` level its arch YAML does not declare.)
    """
    try:
        cfg = cfg_for()
        return "\t".join((
            arch, model, str(k), arm, archs.arch_fingerprint(arch, cfg),
            cfg.cycle_seconds_for(arch),
            archs.effective_variant(arch, cfg),
            note,
        ))
    except Exception as exc:                       # noqa: BLE001 -- recorded, not handled
        msg = " ".join(str(exc).split())
        return "\t".join((
            arch, model, str(k), arm, "REFUSED", "-", "-",
            f"{note + ' ' if note else ''}{type(exc).__name__}: {msg}",
        ))


def main():
    cfg0 = config.load_config()
    archs.install_local_archs(verbose=False)
    models = tuple(config.CNN_MODELS) + tuple(config.TRANSFORMER_MODELS)

    out = ["# arch\tmodel\tK\tarm\tfingerprint\tcycle_seconds\tvariant_slug\tnote"]

    out.append("# grid A: arch x model, at env.sh's K and the reference arm")
    for arch in config.KNOWN_ARCHS:
        for model in models:
            out.append(_row(
                arch, lambda a=arch, m=model: dataclasses.replace(
                    cfg0, const_arch=a, sweep_archs=[a],
                    const_model=m, sweep_models=[m], recon_ert_arm="reference"),
                model, cfg0.const_k, "reference", ""))

    out.append("# grid B: arch x K x mapper arm, at env.sh's model")
    model = cfg0.const_model
    for arch in config.KNOWN_ARCHS:
        arms, note = _arms(arch, cfg0)
        for k in sorted(config.BCH63_KTOD):
            for arm in arms:
                out.append(_row(
                    arch, lambda a=arch, kk=k, ar=arm: dataclasses.replace(
                        cfg0, const_arch=a, sweep_archs=[a],
                        const_model=model, sweep_models=[model],
                        const_k=kk, sweep_ks=[kk], recon_ert_arm=ar),
                    model, k, arm, note))

    # The two digests that sit INSIDE every fingerprint above. Printing them
    # separately turns "every hash moved" into one line that says why.
    out.append("# digests")
    out.append("\t".join(("_components_digest", archs.components_digest())))
    out.append("\t".join(("_config_hash", cfg0.fingerprint())))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
