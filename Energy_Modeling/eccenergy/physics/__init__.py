"""L1 -- what the code and the layout ARE, as arithmetic. No file I/O.

`parity` and `embedded` are the two placements of the parity bits; `widths` is
THE WIDTH TABLE (`CLAUDE.md`, protected); `baseline_dram` is the conventional
arm's price model; `packing` is how a reduced weight is physically stored
(`stream` against `aligned`); `granularity` is `G_rec` and the engine counts the
reconstruction terms are billed over; `recon_dc` is the synthesized datapath's
MEASURED two terms, selected by code.

ONE MODULE HERE READS A FILE, AND IT IS DECLARED: `recon_dc` reads
`data/dc/BCH_N63_results.json`. That is a cited, static measurement -- an input
like the width table, never a cache and never a result -- and it is here because
`arch/` prices an ERT bump from it and may not reach up into a driver to do so
(ProjectRestructure phase 4). Everything else in this package is arithmetic over
numbers handed in.

May import: `contracts`, `settings`/`config`, `paths`. Nothing above.

IT HOLDS, AND SO DOES THE WHOLE RULE. `baseline_dram` used to import
`study.energy` for `dram_ert_pj_per_bit`; phase 3 moved that function DOWN here,
where its arithmetic belongs. Phase 4 did the same for the DC reconstruction
table, and `tests/contract/test_layer_rule.py`'s exception list is now EMPTY.
"""
