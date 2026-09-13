"""L5 -- the renderers. Nothing here decides a number.

`stacked` holds `draw_panel()`, the ONLY place a bar is drawn, so every figure
moves together (`CLAUDE.md`). `panels` stacks one panel per design; `style` is
the palette and the matplotlib import.

May import: anything below. NOTHING may import it except the L4 drivers that
draw -- `study/{sweep,panels,dilation}.py` and `experiments/recon.py`.
"""
