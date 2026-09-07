Simba — paper-fidelity overlay
==============================

Only `arch_paper.yaml` lives here. At `ECC_ARCH_FIDELITY=stock` the design
falls back to `example_designs/simba_like/arch.yaml`, unmodified.

Source:

> Y. S. Shao et al., *"Simba: Scaling Deep-Learning Inference with
> Multi-Chip-Module-Based Architecture"*, MICRO-52, 2019.
> https://people.eecs.berkeley.edu/~ysshao/assets/papers/shao2019-micro.pdf

One change: `mac.adder_width` 16 -> 24
--------------------------------------

The paper: each chip delivers 4 TOPS "using 8-bit weights and activations and
**24-bit accumulation**". The stock design already declares `PEAccuBuffer` at
`datawidth: 24`, so its accumulator *storage* was right while its accumulator
*arithmetic* was declared 16-bit — Accelergy costed the adder for a narrower
datapath than the buffer it writes into.

Why the GlobalBuffer stays at `datawidth: 8`
--------------------------------------------

The rule applied across this study is:

> a level is billed at the **accumulator** width where the paper says partial
> sums live there, and at the **activation** width where the paper says values
> are requantized before reaching it.

Simba's PE ends in a post-processing unit that requantizes on the way out of the
accumulation buffer, so the chip-level `GlobalBuffer` holds 8-bit activations,
not 24-bit psums. Eyeriss v1's 100 kB GLB is the opposite case — JSSC 2017 says
it is "allocated for ifmaps and psums" — which is why that one had to be split
into separate 8b and 16b levels. See `../eyeriss_like/README.md`.

Because `bash run.sh diagnose` flags any Outputs-carrying level narrower than
its own accumulator, the GlobalBuffer carries a `# psum-width-ok:` comment
saying why. The audit then lists it as a declared claim rather than a defect —
the claim stays visible, it just stops being reported as a bug.

Not modelled
------------

* **The MCM interconnect.** This is a single-chiplet model: 16 PEs, no NoP, no
  ground-referenced signalling. Nothing here says anything about Simba's
  multi-chip scaling results.
* **Non-uniform tiling / communication-aware mapping**, the paper's answer to
  NoP latency. Irrelevant without the NoP.
* **65nm → 45nm**, as with every other design here.

Simba's large total energy in this study is not a defect: 2,113,536 weights of
on-chip capacity, bought with a great deal of distributed SRAM, which then
dominates. `ECC_CLASSIFY=instances` (the default) is required to attribute that
storage correctly — its per-PE `PEWeightBuffer` / `PEAccuBuffer` /
`PEInputBuffer` are replicated 16-64x, and the legacy name-matching classifier
drew them as a shared global buffer.
