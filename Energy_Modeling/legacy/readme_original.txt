=====================================================================
ENERGY MODELING (Timeloop + Accelergy)  -  HOW TO RUN
=====================================================================

Setup: Docker Desktop on Windows. Run from COMMAND PROMPT (cmd).
NOT WSL. NOT docker-compose. A single "docker run" command mounts the
Energy_modeling folder into the container.

FOLDER LAYOUT
---------------------------------------------------------------------
Energy_modeling\                 <- ALWAYS run scripts from here
   E3-CODE\                      CNN scripts
      generate_models.py         builds model_layers.json (8 CNNs)
      run_ecc_study.py           single model, eyeriss_like, detailed figure
      run_ecc_multimodel.py      8-model sweep, simple_weight_stationary
   Transformers\                 transformer scripts
      run_ecc_transformer_block.py   GPT-2 small, grant-grade figure
   ecc_energy_study\             AUTO-MANAGED working dir + cache
      timeloop-accelergy-exercises\  cloned repo   (keep)
      outputs\                        mapper cache  (keep - slow to rebuild)
      model_layers.json               input for the sweep
      (csv/png/pdf results land here)
   readme.txt

=====================================================================
STEP 1 - Start Docker Desktop.

STEP 2 - Open Command Prompt and run this ONE line (keep the quotes,
         the path has spaces):

docker run -it --rm -v "C:\Users\nithi\jupyter-files\Relaxed error correction\Energy_modeling":/home/workspace timeloopaccelergy/timeloop-accelergy-pytorch:latest-amd64 bash

You land at a "#" prompt inside the container.

STEP 3 - Go to the mounted folder and run a script BY ITS PATH:

   cd /home/workspace

   python3 E3-CODE/generate_models.py            (run once; makes model_layers.json)
   python3 E3-CODE/run_ecc_study.py              (single-model CNN study)
   python3 E3-CODE/run_ecc_multimodel.py         (8-model CNN sweep)
   python3 Transformers/run_ecc_transformer_block.py   (transformer)

=====================================================================
*** IMPORTANT - THE ONE RULE THAT AVOIDS HOURS OF RE-RUNNING ***
=====================================================================
The scripts find ecc_energy_study\ relative to WHERE YOU LAUNCH THEM,
not where the .py file sits. So:

  ALWAYS:  cd /home/workspace   THEN   python3 <folder>/<script>.py

  NEVER:   cd into E3-CODE or Transformers first.
           If you do, the script makes a NEW empty ecc_energy_study
           inside that subfolder, re-clones the repo, and re-runs the
           whole mapper from scratch (very slow). Your real cache is
           only used when you launch from Energy_modeling.

=====================================================================
NOTES
- Edit any script on Windows, save, and the container sees it instantly
  (the mount handles it - no docker cp).
- All outputs (csv, png, pdf) appear on Windows in:
  ...\Energy_modeling\ecc_energy_study\
- Cached mapper results (ecc_energy_study\outputs\) make re-runs finish
  in seconds. Don't delete that folder unless you want a full re-map.
- Order matters once: run generate_models.py before run_ecc_multimodel.py.
  run_ecc_study.py clones the exercises repo on first run.
- To leave the container: type  exit  (the --rm flag removes the
  container, but your files are safe on Windows via the mount).