"""What is LEFT of the old flat `experiments/` package: the two drivers that
PHASE 3 splits.

`recon.py` (2,862 lines) becomes `study/placement_eval.py` + `report/recon_view.py`
and `dilation.py` (1,851) becomes `study/dilation.py` + `report/dilation_view.py`.
Every other driver moved to `study/` in phase 2; these two did not, because a
file that is about to be cut across two layers cannot be filed under one of them
first. When phase 3 lands, this package goes.
"""
