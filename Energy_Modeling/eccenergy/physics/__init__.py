"""L1 -- what the code and the layout ARE, as arithmetic. No file I/O.

`parity` and `embedded` are the two placements of the parity bits; `widths` is
THE WIDTH TABLE (`CLAUDE.md`, protected); `baseline_dram` is the conventional
arm's price model; `packing` is how a reduced weight is physically stored
(`stream` against `aligned`); `granularity` is `G_rec` and the engine counts the
reconstruction terms are billed over.

May import: `contracts`, `settings`/`config`, `paths`. Nothing above.

IT NOW HOLDS. `baseline_dram` used to import `study.energy` for
`dram_ert_pj_per_bit`; phase 3 moved that function DOWN here, where its
arithmetic belongs, and `study/energy.py` imports it from this package instead.
The exception is gone from `tests/contract/test_layer_rule.py` rather than
re-worded.
"""
