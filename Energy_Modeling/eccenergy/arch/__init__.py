"""L2 -- the designs, the workloads they run, and the chips to map.

The only readers of `archs/<name>/*.yaml`. After phase 5 a new architecture is
one directory of DATA here and no code anywhere.

    load        install a design, read it, and the cross-design contract
    patch       the arch.yaml the mapper actually runs
    fingerprint what the mapper cache is keyed by, and the ERT bump
    layout      level names, globals files, write-once semantics
    validate    `run.sh validate` and `run.sh diagnose`
    weight_path the stages a weight crosses, per design
    placements  the reconstruction boundaries, per design
    arms        which of those boundaries is a DISTINCT CHIP to map
    workloads   the layer shapes  |  generate  the workload files

May import: `physics`, `contracts`, `settings`/`config`, `paths`.

PHASE 3 cut `archs.py` (2,762 lines) and the L2 half of `recon.py` into these
nine. `weight_path`, `placements` and `arms` are here, and not in `study/`,
because `config.py` and `toolchain/ert_probe.py` both read the arm list and
neither may import a driver. The two upward imports that remain -- `fingerprint`
and `arms` reaching for the DC reconstruction table -- are declared in
`tests/contract/test_layer_rule.py` and are phase 4's to remove.
"""
