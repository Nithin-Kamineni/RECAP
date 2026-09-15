Preserved 2026-09-14 by Task D session 1, before mapping simple_weight_stationary.

WHY: the placement study's stem is ReconSweep_optimiser__<model> and carries NO
architecture (TaskD_MapTheTwoDesigns.md section 1.3), so session 1's run of
simple_weight_stationary overwrites eyeriss_like_wglb's figure in place.

WHAT THESE ARE: eyeriss_like_wglb ("Eyeriss v1 (+filter GLB)"), resnet18, at the
DEV two-layer scope (conv1, layer3.0.conv1), BCH(63,39), ERT-aware mapping.
The .png/.csv/.json were written at 20:02 by this session's own
`gate.sh --update` snapshot, which runs the recon stages at env.sh's shipped
configuration. The .pdf is older (2026-09-13) and is the FULL-scope figure.

ALSO IN GIT: the committed, FULL-scope eyeriss_like_wglb .png and .pdf are at
HEAD (results/figures/ReconSweep_optimiser__resnet18.{png,pdf}); the working-tree
.png had already been replaced by the dev-scope one before this copy was taken.
Everything here is reproducible from results/_raw/ with --replot.
