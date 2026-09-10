You have done the whole sweep in a very wrong way in the previous sessions of results:

(1) THE TWO ARMS DID NOT SEE THE SAME READ/WRITE ENERGY, and this is the defect that
invalidates the whole sweep. The extra capacity was expressed as `depth x N/K` on the
weight levels (`ECC_WEIGHT_CAPACITY_SCALE`), so Accelergy costed the reconstruction
arm's memory as a PHYSICALLY BIGGER array -- measured 1.18x to 1.46x more energy per
access on these designs, for silicon the reconstruction arm does not have. The mapping
optimiser then optimised against that inflated price, so it had a positive reason to
LEAVE THE EXTRA ROOM UNUSED. Every "no improvement" result in the sweep was produced
under a search that was biased against the hypothesis it was testing. There is an
evaluator-side correction (`recon.capacity_dilation_correction()`) that re-prices the
energy afterwards, but it CANNOT re-price the mapping, so the bias stands.

(2) THE ECC ENERGY EVALUATOR WAS NEVER RUN. Only `hpc/map_capacity_sweep.sh` was run,
which submits mapping jobs and deliberately no dependent evaluation. Every "saving"
quoted was DRAM-term arithmetic computed inside `experiments/dilation.py`, not the
embedded-vs-recon energy model. No R1-R4b bar, no placement, no reconstruction cost
was ever evaluated on any of these mappings.

(3) TOO MANY THINGS MOVED AT ONCE. 5 architectures x up to 5 layers x up to 28
capacities were swept together (196 jobs), which makes a trend unreadable. One
architecture and one layer at a time is the right unit.

(4) THE CAPACITIES SWEPT WENT FAR BELOW ANYTHING PHYSICAL -- down to x0.03125, a
14-weight scratchpad. Below about x0.5 the mapper starts trading PE COUNT for
capacity even under EDP (measured 168 -> 84 -> 96 -> 112 PEs on eyeriss v1), so the
two arms stop being the same machine and nothing is attributable to capacity.

(5) ONE RESULT WAS A CROSS-FINGERPRINT COMPARISON. The reported "eyeriss v2 refetch
14.00 -> 4.00" compared a stale mapper cache (`fp-3eb860ea2b2a`, solved before a
config change) against a current one. The current cache already refetches 4.00
undilated. Two `fp-<hash>` directories under one variant slug are two ARCHITECTURES.

(6) THE VERDICT LOGIC ITSELF HAD A BUG that let four more false positives through: it
labelled a pair a "capacity win" whenever DRAM reads fell, without requiring that the
weights actually HELD on chip went UP. Fixed mid-session, but it means anything
quoted before that fix must be re-derived.

(7) EVERY NUMBER WAS REPORTED FROM AN UNCONVERGED SEARCH. The whole sweep ran at
`ECC_VICTORY=2000`, and raising it to 10000 REVERSES the headline: the one pair that
looked like the best result in the study (weight-stationary shared, x0.5 -> x0.8077,
recon 9.1 uJ better) becomes recon 5.4 uJ WORSE with 4x the refetch. So the sweep was
measuring the search's failure to converge, not the architecture. See the CONVERGENCE
GATE below -- this is the single most important thing to fix.

(8) 12 JOBS WERE SPENT ON A SEED-VARIANCE TEST THAT MEASURED NOTHING. `ECC_MAPPER_SEED`
is in the cache fingerprint but reaches nothing -- `timeloop-mapper` v4 exposes no
random seed, and env.sh says so in a comment that should have been read first. All
three "seeds" produced byte-identical output.

First the mapping might at least be reasonably optimised for a single layer. Do this
for a single arch first (We keep arch constant too). Your mapping currently had two
arms (I guess this is approaches of Embedded 0.5 and Recon 0.5x(n/k)) that do not see
the same read/write energy. But I need the same energies for both of them; if that is
not possible, pretend the data in the Recon is quantised to 5-bit weights whereas
Embedded will have full 8-bit weights (then you can use the same memory sizes). I will
provide quantisation numbers for each of the weights based on the BCH configuration
you get below.

I am trying to find certain reduced arch values of memories for each of the
architectures. So I want you to sweep the hardware configs of memories of 0.5 of
memory, 0.25 of memory, 0.125 of the current memory to find which values will perform
good with more energy savings for Recon compared to the Embedded approach. To be exact
about what "memory" means here: I am ONLY sweeping the DEPTH values of the ON-CHIP
WEIGHT memories. Nothing else changes -- not the word width, not the DRAM, not the
activation or partial-sum buffers, not any other attribute.

For Recon, this is the quantisation of weights that is exactly being used. (Try to not
round up the quant values; if you need to round the values then round down) but try to
keep them the same for accurate study. (The read and write value for a single bit will
be the same as the Embedded.) The MACs again will have the same 8-bit multiplication as
we will quantise the values by that point (I don't think we need to bother about it now
but just letting you know.)

    BCH(63,57) = 7.25 bit quantisation
    BCH(63,51) = 6.5  bit quantisation
    BCH(63,45) = 5.75 bit quantisation
    BCH(63,39) = 5    bit quantisation
    BCH(63,30) = 4    bit quantisation

The only way our Recon framework works is by co-optimising the reduced representation
of the same weights in the same memory in the mapping optimiser (currently the mapping
optimiser is run in a wrong way where read energy and write energy are different per
bit for the approaches of Embedded and Recon, which made the mapping optimiser sweep
results show no improvement in the Recon approach over the Embedded approach). This
way the mapping optimiser will find a valid mapping by taking advantage of Recon's
ability to represent weights in less space than its counterpart Embedded ECC.

Each of the tables must contain the following columns: name of memory hierarchy (GLB,
or RF, or ScratchPad, etc.), amount of memory scaled times (0.5 times depth of memory, 0.25
times depth of memory, 0.125 times depth of memory), refetch, average utilisation (this is for the number
of weights stored in the memory itself), and a pseudo energy term (to calculate how
much energy this cost in that particular memory hierarchy; I am assuming for Recon
(less than 8-bit quantised weights) it would be a lot less than Embedded (8-bit
quantised weights) because we get more amount of data storage in the same memory --
only our effective memory is increasing, not the real memory, and we get less
refetching of data due to this).

This will tell me what memory I should use for testing my framework to see Recon
placements saving huge amounts of energy when compared to Embedded ECC (previous work).
Based on these values I am going to change the yaml file values.

I want to run the sweep of weight memory configurations (memory of GLB, RF, SPAD,
etc.) with the ARCH. Do this for a single layer that is most repeating in resnet18. Do
this for a single ARCH (Eyeriss V1) for the start, as doing this for multiple archs is
making it harder to understand the trend.

Additional information:

REPO AND HOW TO RUN
- Repo: /blue/rewetz/vkamineni/Projects/RECAP/Energy_Modeling on HiPerGator.
  `module load apptainer`, then `bash hpc/tl.sh <cmd>`. Read CLAUDE.md first, then
  FINDINGS.md section 7.8, then the last entries of progress.txt.
- READ prompt_1.md FIRST. It documents the DRAM -> on-chip transfer cost problem, which
  sets the size of every ECC percentage in this study and is unresolved. The DRAM model
  is 512 pJ per 64-bit access = 8 pJ/bit, and `ECC_DRAM_IF_FRAC = 0.40` (an uncited
  ASSUMPTION) means only 15.2% of DRAM weight energy is credited to reconstruction even
  though 38.1% of the bits leave the bus. That is why a 38% traffic cut shows up as a
  2% energy cut. Settle that before trusting any sweep headline.
- `env.sh` is the only file to edit; every knob is `${VAR:=default}` so the environment
  wins. `ECC_RECON_MODELING=1` is the DEFAULT and it HARD-ASSIGNS `ECC_ARCHS` from
  `ECC_RECON_ARCHS` -- so setting `ECC_ARCHS` on the command line is silently ignored.
  Use `ECC_RECON_ARCHS` to choose designs.
- Keep `ECC_MAPPER_THREADS=18`; it is in the cache fingerprint.
- `ECC_QOS=rewetz-b` gives ~90 concurrent jobs. A map job on one layer is 3-25 min.
- Run the reporting tool as `bash hpc/tl.sh python3 -m eccenergy.experiments.dilation
  --table`. Running it bare silently uses config.py's defaults (objective=energy,
  victory=500) instead of env.sh's -- the table prints a banner if the objective is not
  edp, but do not rely on noticing it.

THE LAYER TO USE
- The most repeated shape in resnet18 is `C64_M64_R3_S3_P56_Q56_ws1_hs1` (layer1.0.conv1,
  layer1.0.conv2, layer1.1.conv1, layer1.1.conv2 -- four layers, 36,864 weights each).
  On eyeriss v1 it refetches 16.0x, so it has plenty of headroom to remove.
- `layer3.0.conv1` (294,912 weights, refetch 14.0x, 42.9% weight-buffer fill) is the
  single best-conditioned layer if you want one with a nearly-full weight buffer.
- Do NOT use `layer2.0.conv1` just because the old sessions did; it is not the best
  test bed. Do not use `layer4.*` -- it already refetches 1.0x, so there is nothing to
  remove.

HOW TO IMPLEMENT THE QUANTISATION APPROACH (this is the right fix, see below)
- Change `datawidth:` on the weight-carrying levels ONLY. Leave `depth:` and `width:`
  exactly as declared. Timeloop then computes `block_size = width / datawidth`, so more
  weights fit per physical word while the array geometry -- and therefore the CACTI
  per-access read/write/leak energy -- is unchanged. That is exactly the "same energies,
  more effective capacity" condition being asked for.
- FIRST THING TO VERIFY, before any sweep: map ONE shape at `datawidth: 5` and confirm
  in `timeloop-mapper.ERT_summary.yaml` that the weight level's `read`/`write` energies
  are IDENTICAL to the 8-bit run. If they move, this approach has the same defect as the
  old one and everything downstream is invalid. This check is ~15 minutes and gates
  everything.
- ALL THREE APPROACHES MUST USE THE SAME HARDWARE YAML. Baseline, Embedded and Recon
  are compared against each other, so they must declare the SAME `width:` and the SAME
  `depth:` on every memory -- identical arrays, therefore identical read, write, static
  and leakage energies per access. THE ONLY THING THAT MAY DIFFER BETWEEN ARMS IS
  `datawidth:`, which changes how many weights fit in one physical word and nothing
  else. If two arms ever differ in `width` or `depth`, the comparison is void: that is
  exactly the defect (1) above. Add an explicit check that asserts it before evaluating.
- FRACTIONAL BIT WIDTHS: USE THE SCALE-BY-4 TRICK. `datawidth` must be an integer, so
  7.25 / 6.5 / 5.75 bits cannot be written directly. Express everything at 4x: multiply
  the word `width` by 4, divide `depth` by 4 (total bits unchanged), and multiply EVERY
  arm's `datawidth` by 4, so one unit = a quarter of a bit. `q` bits then becomes
  `datawidth = 4q`: 8 -> 32, 7.25 -> 29, 6.5 -> 26, 5.75 -> 23, 5 -> 20, 4 -> 16.
  NOTE the embedded arm's datawidth must ALSO be scaled, 8 -> 32. Leaving it at 8 would
  give it four times the weights per word and silently break the comparison.
  The exact width/depth/datawidth numbers to put in the YAML are in the table further
  down ("TWO SEPARATE WIDENINGS"), because they also fold in the wider LOGICAL word --
  do not derive them from the declared 16-bit scratchpad, which buys nothing for the
  weaker codes.

  WHY THE WORD WIDTH MATTERS AT ALL. A weight is stored in a physical word, and only a
  WHOLE number of them fit: `floor(word_bits / bits_per_weight)`. On a NARROW word that
  floor is brutal -- on a 16-bit word an 8-bit weight gives 2 per word and a 5.75-bit
  weight ALSO gives 2, so a code that genuinely stores 28% fewer bits buys literally
  nothing. On a WIDE word the same code lands between more integers and the packing
  tracks the true ratio. So the word width decides whether reconstruction can express
  its advantage at all; it is not a performance tweak.

  EVERY NUMBER IN THE TABLES BELOW IS `floor(W / q)` = HOW MANY WEIGHTS FIT IN ONE
  PHYSICAL WORD, where W is the logical word width in bits and q is the bits per weight.
  Nothing else. The effective capacity ratio of an arm against the embedded arm is
  simply its count divided by the `emb` count in the same row.

  TABLE A -- the memories AS DECLARED in the arch YAML today:

      memory                word     emb   7.25b   6.5b   5.75b    5b     4b
      ----------------------------------------------------------------------
      weights_spad         16 bit      2      2       2       2      3      4
      filter_glb / DRAM    64 bit      8      8       9      11     12     16

  TABLE B -- the same memories with the LOGICAL WORD WIDENED, which is what to use.
  Depth is divided by the same factor the word is multiplied by, so the total bits and
  the embedded arm's weight capacity are UNCHANGED (448 and 8,192 weights):

      memory                word    depth   emb   7.25b   6.5b   5.75b    5b     4b
      ---------------------------------------------------------------------------
      weights_spad         64 bit      56     8      8       9      11     12     16
      filter_glb          256 bit     256    32     35      39      44     51     64

  WHAT THE TWO TABLES SAY, in one comparison. On the declared 16-bit scratchpad
  (Table A) an 8-bit weight packs 2 per word and a 5.75-bit weight ALSO packs 2 per
  word -- a code storing 28% fewer bits gains nothing, and the same is true at 6.5 and
  7.25 bits. Widen that word to 64 bits (Table B) and the same three codes pack 11, 9
  and 8 against the embedded 8, so the code rate finally shows up. Widen `filter_glb`
  to 256 bits and even 7.25 bits pays (35 vs 32). This is NOT a rounding artifact --
  it survives exact quarter-bit resolution -- it is simply what a narrow word does to
  a floor division, and it is why the widened geometry is adopted rather than swept.

  *** WORD WIDTH IS FIXED CONTEXT, NOT A SWEPT VARIABLE. ***
  Adopt `weights_spad` = 64-bit word x depth 56 and `filter_glb` = 256-bit word x depth
  256 as the STANDING geometry for all of Baseline, Embedded and Recon, and leave them
  there. They are chosen because they let the code rate show up in the packing at all;
  there is no need to sweep them and they must NOT be swept. Note the embedded arm gains
  no capacity from this (448 and 8,192 weights either way) -- only the granularity
  changes.

  *** THE ONLY THING SWEPT IS THE `depth:` OF THE ON-CHIP WEIGHT MEMORIES. ***
  x0.5, x0.25, x0.125 of the declared depth, and nothing else in the entire study:

      level           x1     x0.5    x0.25   x0.125
      weights_spad     56      28       14        7
      filter_glb      256     128       64       32

  Explicitly NOT swept, at any point: the word `width`; the per-arm `datawidth`; the
  DRAM (it is not on-chip -- its depth is set by `ECC_DRAM_DEPTH` and must not move);
  any memory that does not hold Weights (ifmap/psum buffers stay exactly as declared);
  bandwidths, banking, `n_banks`, technology node, PE counts or the dataflow
  constraints. If a knob is not `depth:` on a weight-carrying on-chip level, it is
  fixed context. One sentence to hold on to: WIDEN THE WORD ONCE, THEN SWEEP ONLY THE
  DEPTH OF THE ON-CHIP WEIGHT MEMORIES.

  TWO SEPARATE "WIDENINGS" ARE IN PLAY AND THEY COMPOSE -- do not confuse them:
    (a) the DESIGN choice above: make the LOGICAL word hold more weights
        (weights_spad 16 -> 64 bits, filter_glb 64 -> 256 bits);
    (b) the x4 REPRESENTATION trick, which only exists so `datawidth` can be an integer
        at quarter-bit resolution.
  Apply (b) on top of (a). The numbers that actually go in the YAML, verified to
  reproduce `floor(W_logical / q)` exactly for every code:

      level          DECLARE width  depth   datawidth per arm
                                            emb  7.25b 6.5b 5.75b  5b   4b
      weights_spad       256          56     32    29    26    23   20   16
        -> weights/word:                      8     8     9    11   12   16
      filter_glb        1024         256     32    29    26    23   20   16
        -> weights/word:                     32    35    39    44   51   64

  So `width` and `depth` are IDENTICAL on every arm (that is the fairness rule); only
  the `datawidth` column changes. The declared width is 4x the logical word because of
  (b); the logical capacity is unchanged (448 and 8,192 weights in the embedded arm).
  The x0.5 / x0.25 / x0.125 sweep then scales ONLY the `depth` column
  (56 -> 28 -> 14 -> 7, and 256 -> 128 -> 64 -> 32).

  ONE THING TO CHECK, not to assume: a wider word also makes each access COARSER
  (Timeloop fetches a whole block), so a mapping whose tile does not fill the block can
  waste bandwidth, and CACTI prices a wide shallow array differently from a narrow deep
  one. Both arms share it so the COMPARISON stays fair, but confirm the widened geometry
  does not make the absolute energy worse than the declared one before building on it.
- ROUNDING DIRECTION MUST BE STATED. Rounding the bit width DOWN (5.75 -> 5) rounds the
  CAPACITY UP and flatters reconstruction; rounding UP is conservative. With the
  scale-by-4 trick no rounding is needed for these five codes, so prefer it and say so.
- BEWARE DOUBLE-COUNTING. The existing recon model already scales reduced stages by
  K/N (`recon.py`, `stream` packing). If the mapper is now ALSO pricing narrower
  weights, applying both would count the same saving twice. Decide which layer of the
  model owns the on-chip narrowing, and add a check that it is applied exactly once.

WHAT IS ALREADY BUILT AND WORKS
- `bash hpc/tl.sh python3 -m eccenergy.experiments.dilation --table` prints one row per
  swept capacity per design per layer with: weights/PE, PEs used, MACs used/declared,
  total weight room, WEIGHTS HELD, fill %, DRAM weight reads, refetch, total uJ, the
  fullest non-weight buffer, a verdict, and the cache fingerprint. `--csv` writes it.
- `weights held` is the column that decides whether anything happened. If DRAM reads
  move while `held` does not RISE, it was not capacity.
- The verdicts are: `capacity` (reads fell AND held rose AND PE count unchanged -- the
  only one that may be quoted), `PERM?` (held identical -- a loop-order change), `ORDER?`
  (held fell -- stored less and read less), `PE!=` (PE count moved -- not attributable),
  `flat`. A fingerprint guard prints a warning when one variant slug has two solved
  fingerprints.
- `eccenergy/tests/test_dilation.py` -- 11 tests including two property tests over the
  real mapper cache. Run it after any change to the above.
- `ECC_WEIGHT_FACTOR_RELAX=1` drops the `factors:` pins on the weight-indexed dimensions
  (M, C, R, S) of weight levels. It did not make any dilation pay, but at the SAME
  capacity and SAME 168/168 PEs it cut eyeriss v1 `layer3.0.conv1` refetch 14.0 -> 2.0
  and layer energy 417 -> 267 uJ. It is a DIFFERENT DATAFLOW (v1's `M=1` at the filter
  spad IS row-stationary) so it must never be quoted as the published chip -- but it
  shows the dataflow constraint, not the buffer size, is what costs DRAM weight traffic.

TWO THINGS THAT WILL BITE
- THE MAPPER IS DETERMINISTIC BUT CHAOTIC IN THE ARCHITECTURE. Same arch + same settings
  + same thread count gives byte-identical output. But changing a buffer depth by a few
  entries sends `random_pruned` down a different path, reproducibly. Measured: 52 of 838
  PE-matched capacity pairs (6%) had a LARGER buffer refetching MORE, which is impossible
  under a real capacity mechanism. You cannot average this away with seeds. The only
  lever is a stronger search -- raise `ECC_VICTORY` (2000 today) until the answer stops
  moving with capacity, and CHECK THAT IT HAS before trusting a trend.
- ***CONVERGENCE IS A GATE. DO NOT REPORT ANY SWEEP NUMBER UNTIL IT PASSES.***
  The search at `ECC_VICTORY=2000` is NOT converged, and an unconverged mapper will
  point the sweep in the wrong direction: you will read search failure as an
  architectural result. Two measurements, both from this repo:

    * On eyeriss v1 `layer3.0.conv1` the mapper at declared capacity returned a mapping
      24% WORSE IN ITS OWN EDP OBJECTIVE (143.5) than one it found on the SAME layer at
      a different capacity (115.9). Every ECC saving in this study is 2-12%.
    * MEASURED 2026-09-10, and it is the reason this whole prompt exists: raising
      victory 2000 -> 10000 on weight-stationary (`scope=shared`, layer2.0.conv1)
      REVERSED THE CONCLUSION.

          scale       victory=2000        victory=10000
          x0.5        rf=1.000  273.2uJ   rf=2.000  251.0uJ
          x0.75       rf=4.000  280.4uJ   rf=8.000  254.2uJ
          x0.8077     rf=1.000  264.1uJ   rf=8.000  256.4uJ
          x1.2115     rf=1.000  276.7uJ   rf=2.000  261.5uJ

      At victory 2000 the recon arm (x0.8077) beat its embedded reference (x0.5) by
      9.1 uJ and that looked like the best result in the study. At victory 10000 the
      SAME pair reverses: recon is 5.4 uJ WORSE and refetches 4x more. Energy falls at
      every point (a better search), but the ORDERING between the two arms flips.

  THE GATE, run it before anything else and again whenever the architecture changes:
    1. Pick the one layer and the one design. Map the embedded arm at victory
       {2000, 10000, 50000, and higher if it is still moving}.
    2. Plot total energy and refetch against victory. CONVERGED means the last two
       points agree to within a margin you state, and that margin MUST be smaller than
       the recon-vs-embedded effect you intend to claim. If the mapping still moves by
       5% between victory settings, you cannot resolve a 2-12% ECC effect. Say so and
       raise the budget rather than reporting the number.
    3. Only then run the memory sweep, at the converged victory, for BOTH arms.
    4. Re-check the gate at the smallest memory in the sweep as well as the largest --
       a budget that converges on a big buffer may not on a small one.
  Cost is not a reason to skip this: a single-layer map is 3-25 min at victory 2000, and
  the whole gate is a handful of jobs. It is far cheaper than a wrong conclusion.
  Jobs 41582123-30 are this test at {10000, 50000}; the 50000 half was still running at
  the end of the session, so READ IT FIRST before trusting anything at victory 2000.

ALWAYS
- `ECC_OPT_METRIC=edp` (env.sh's default). Never `energy` -- it serialises and trades PEs
  for capacity, which is what withdrew FINDINGS 7.7.
- Report `PEs used` on every row. It is what caught two of the false positives above.
- Both arms of a pair must be mapped under the SAME treatment, so the comparison stays
  fair even when neither arm is the published design.
