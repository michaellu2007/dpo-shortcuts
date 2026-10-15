# Project 08 Proposal — Does Preference Tuning Learn Correctness or Verbosity?

**Course:** NLP and LLMs · Fall 2026 · individual project (candidate #8)
**Type:** research problem · 1 GPU, short training · Qwen3-0.6B + LoRA

## Question and motivation

Direct Preference Optimization (DPO; Rafailov et al., 2023) optimizes a policy
against pairwise preferences. In real preference data, *correctness* co-occurs
with surface cues such as **length**: preferred answers tend to be longer and
more elaborate. Park et al. (2024) showed verbosity is a confound in DPO
evaluation, but a cleaner question remains: **when length and correctness
disagree inside the training signal itself, which one does DPO follow?** A
model that follows correctness improves wherever preferred answers are correct;
a model that follows a length shortcut should *degrade* whenever the shortcut
conflicts with the task — e.g., produce terse guesses or verbose drift. This is
directly actionable: it tells practitioners when pairwise preference data can
be trusted versus when it must be length-controlled before training.

## First experiment

**Data (synthetic, exactly checkable).** Two tasks: (i) 2–3-step arithmetic
expressions; (ii) structured extraction from templated memos into a JSON
object with four fields. Wrong responses are constructed to be *internally
coherent* (a plausible operator slip narrated consistently; a single planted
distractor value), so the preference signal is about the answer, not about
detectable nonsense. Every pair contains one short and one long response, so
per-pair token totals are matched across conditions. Two conditions, in both
of which the **preferred answer is always correct**:

- **SC** (short-correct): chosen = terse correct, rejected = verbose wrong;
- **LC** (long-correct): chosen = narrated correct, rejected = terse wrong.

500 pairs (250/condition) plus a **held-out eval set of 200 fresh instances**
(dev: pilot instances used for protocol tuning; eval: this fixed set), fixed
seed, zero train/eval overlap.

**Baseline and comparison.** Baselines: the untuned Qwen3-0.6B and an SFT
model trained on the same preferred answers only (6 epochs ≈ DPO's token
exposure). Comparison: DPO trained per condition (LoRA r=16, β=0.1, 3 epochs,
bf16). Noise ablation: 10% flipped labels, isolated runs. 3 seeds for main
conditions (18 runs total).

**Evaluation (fixed 200-instance eval set, identical prompts across models).**
Exact correctness (integer EM; field-level EM), mean generated length, invalid
output rate, and label-flip sensitivity (accuracy drop under flipped
training). Thinking mode is disabled at both training and evaluation so length
reflects the policy.

**Pilot result (200 pairs/condition, seed 0, completed).** DPO-LC followed the
length cue, not correctness: mean output grew to 143.7 tokens (vs 84.4 for
SFT-LC), 42.5% of outputs degenerated into repetitive loops and became
unparseable, and held-out accuracy fell to 0.495 (arithmetic 0.21) — while
**SFT on the identical preferred answers scored 0.900 (arithmetic 0.80)**.
DPO-SC showed the mirrored failure: extraction accuracy dropped 1.00→0.83
under terse pressure. Flip sensitivity confirmed DPO obeys the preference
signal in both conditions. This is a strong shortcut signature and the full
seed sweep will test its stability.

## Feasibility

**Hardware.** Self-provided 1× RTX 4090 (24 GB; declared as self-obtained,
outside the shared pool). Measured pilot: DPO run 200 pairs × 3 epochs =
41–68 s wall, peak memory **3.4 GB** (course target ≤12 GB respected); one
200-prompt eval pass ≈ 28 s. **Planned total:** difficulty-calibration sweep +
18 training runs + 19 eval passes ≈ **1.5 GPU-hours incl. retries**, far
below the 8 GPU-hour ceiling; no course-pool GPU needed unless the local one
is unavailable (then: shared pool, 12 GB target, same scripts).
**Data generation** is deterministic CPU code (~1 min for 1,200 instances).
**Risks and fallback:** (i) arithmetic at floor for the 0.6B model — a 3-point
difficulty sweep (operands ≤30, 2 steps) selects a level with untuned accuracy
in 30–70% before the full run; extraction is saturated at baseline, so its
read-out is *retention under each condition* (degradation detection), with a
harder 5-field variant as backup; (ii) DPO variance — the flip ablation run
showed high variance at seed 0, which the 3-seed design quantifies; if
instability persists, reduce lr to 5e-5 (still within budget).
**Deliverable:** a shortcut audit matrix (condition × {accuracy, length,
invalid rate, flip sensitivity}) with per-case failure examples and a concrete
recommendation for when preference pairs need length controls.

## References

Rafailov et al. 2023 (arXiv:2305.18290); Park et al. 2024
(arXiv:2403.19159); Qwen3-0.6B model card (Qwen Team, 2025). Pilot code,
seeds, and full logs: private repo, will be linked in the final report.
