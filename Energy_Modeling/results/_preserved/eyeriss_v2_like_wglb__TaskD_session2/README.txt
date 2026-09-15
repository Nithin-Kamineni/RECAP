TASK D SESSION 2's PLACEMENT FIGURE -- eyeriss_v2_like_wglb.  2026-09-14.

  ReconSweep_optimiser__resnet18.png .pdf .csv .json
      Eyeriss v2 (+weight GLB), resnet18, conv1 + layer3.0.conv1, BCH(63,39),
      q=5, ERT-aware, ECC_SCOPE=dev, four metric rows (energy edp latency
      area).  Seven bars; this design declares recon1..recon5 so there is no
      [skip] line.  DEVELOPMENT SCOPE -- not a published number.

WHY IT IS COPIED HERE.  The stem `ReconSweep_optimiser__<model>` carries NO
architecture, so every placement run of every design writes this one path --
AND SO DOES A GATE RUN, whose recon_default and recon_ert stages render
eyeriss_like_wglb at env.sh's shipped configuration.  That is what happened
here: the figure drawn by --eval-only was overwritten by the gate minutes
later, and redrawn from results/_raw/ with --replot (instant, same numbers:
+20.081% / +17.038%) before being copied here.

Session 1's two preserved directories beside this one are untouched.
