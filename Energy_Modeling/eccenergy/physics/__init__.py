"""L1 -- what the code and the layout ARE, as arithmetic. No file I/O.

`parity` and `embedded` are the two placements of the parity bits; `widths` is
THE WIDTH TABLE (`CLAUDE.md`, protected); `baseline_dram` is the conventional
arm's price model.

May import: `contracts`, `settings`/`config`, `paths`. Nothing above.

`baseline_dram` VIOLATES that today -- it imports `study.energy` for
`dram_ert_pj_per_bit`, which is an L4 module. Recorded rather than fixed here,
because phase 2 moves code and changes none: `tests/contract/test_layer_rule.py`
carries it as a declared exception with this reason on it.
"""
