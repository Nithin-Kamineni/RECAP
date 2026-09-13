"""L5 -- the renderers, and the drivers that draw. Nothing here decides a number.

`stacked` holds `draw_panel()`, the ONLY place a bar is drawn, so every figure
moves together (`CLAUDE.md`). `panels` stacks one panel per model or per design,
and carries the panels driver itself. `style` is the palette and the matplotlib
import. `sweep` is the one-axis experiment; `recon_view` is the placement figure
and `run.sh recon`; `dilation_view` is Task 4's CLI and its CSV writers.

May import: anything below. NOTHING below may import it -- phase 3 removed the
last four such edges, and `tests/contract/test_layer_rule.py` now fails on a new
one.
"""
