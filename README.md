# SWIFT Model — Amortized Inference with BayesFlow v2
### SBI Course Project — TU Dortmund

Amortized Bayesian inference for the **SWIFT model of eye movements during
reading**. The SWIFT simulator's likelihood is intractable, so we train a
BayesFlow neural posterior estimator on simulated `(θ, fixation-sequence)`
pairs, then apply it to real eye-tracking data (participant VP10) in
milliseconds.

---

## Project structure

```
swift/                      ← the package (all source lives here)
  config.py                 ← paths, sequence layout, ONE shared feature normalisation
  simulator.py              ← SWIFT Gillespie simulator (forward model)
  data.py                   ← load .dat files, EDA, real→observation conversion
  generate.py               ← parallel pre-generation of training pairs
  inference.py              ← BayesFlow workflow: train + posterior sampling
  diagnostics.py            ← recovery, SBC, contraction, posterior, PPC plots
data/                       ← data files only (.dat) + generated training_data.npz
outputs/
  figures/                  ← all diagnostic plots
  models/                   ← trained swift_approximator.keras
tools/
  calibrate.py              ← fast NumPy check of simulator vs real marginals
main.py                     ← CLI entry point
pyproject.toml / requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt      # bayesflow 2.0.11 + keras + torch
```

Place the two data files in `data/` (from OSF):
`fixseqin_PB2expVP10.dat` (osf.io/teyd4) and `Rcorpus_PB2_revision.dat` (osf.io/nj2mf).

The torch backend is forced onto CPU inside `swift/inference.py` (Apple's MPS
backend lacks the `linalg_qr` op the spline flow needs). It is a small model —
CPU training takes minutes.

---

## Workflow

```bash
# 1. Pre-generate simulations once (parallel, a few minutes)
python main.py --mode generate --n_readers 10000

# 2. Train on the saved data, then run diagnostics + VP10 inference + PPC
python main.py --mode train

# ...or load an already-trained model and just do diagnostics + inference:
python main.py --mode infer

# everything at once:
python main.py --mode all --n_readers 10000
```

| Mode | Does |
|------|------|
| `generate` | pre-generate simulations (parallel) → `data/training_data.npz` |
| `train` | train on saved data → model + diagnostics + VP10 inference + PPC |
| `infer` | load saved model → diagnostics + VP10 inference + PPC |
| `online` | train with the simulator in the loop (slow) |
| `all` | `generate` → `train` |

---

## Model: free parameters inferred by BayesFlow

| Parameter | Symbol | Prior | Meaning | Main observable signature |
|-----------|--------|-------|---------|---------------------------|
| Saccade timer | `t_sac` | 150–350 ms | mean fixation-duration scale | fixation duration |
| Word-length exponent | `eta` | 0.1–1.0 | how much length slows processing | fixation count / refixation |
| Processing span | `delta0` | 4–15 chars | attention-window width | skipping rate |
| Refixation factor | `R` | 0.1–0.9 | tendency to refixate | refixation rate |

Everything else (`alpha, beta, h, gamma, kappa, rho, sigma, omega`) is fixed and
**calibrated** so the simulator's prior-averaged marginals match VP10 — run
`python tools/calibrate.py` to check (fixations/sentence, duration, skip and
refixation rates all land close to the real values).

---

## Key design choices (what makes inference work here)

1. **One reader = many sentences.** Each training example is one `θ` reading
   `M_SENTENCES=10` sentences, concatenated into a single sequence. A lone
   ~8-fixation sequence carries almost no information about `eta`/`delta0`/`R`;
   ten sentences do. This is the difference between those parameters being
   unrecoverable and recoverable.
2. **One shared normalisation.** `swift/config.normalise_sequence` is applied to
   *both* simulated and real fixations, so training and inference see identical
   input distributions.
3. **Four observable features** per fixation — `word_id, landing_position,
   duration, word_length` — the last one gives the network direct leverage on
   the word-length exponent `eta`.
4. **Pooled participant posterior.** VP10 inference draws many random
   10-sentence subsets, samples a posterior for each, and pools them — replacing
   an earlier step that (incorrectly) averaged raw sequences before inference.

---

## Diagnostics (`outputs/figures/`)

| Plot | Shows | Look for |
|------|-------|----------|
| `eda_fixations.png` | real data distributions | context |
| `training_loss.png` | training loss | decreasing, flattening |
| `recovery_plot.png` | posterior mean vs truth | points on the diagonal |
| `sbc_histogram.png` / `sbc_ecdf.png` | SBC calibration | uniform / inside the bands |
| `contraction_plot.png` | posterior contraction | high = learned from data |
| `posterior_VP10.png` | VP10 posterior | narrow = informative |
| `ppc_plot.png` | simulated vs real statistics | overlapping distributions |

---

## Final results (VP10, M=14 readers, LSTM + statistic conditions)

Training loss 3.1 → **1.6**. Parameter recovery on 300 held-out simulations:

| Parameter | Recovery r | SBC | Notes |
|-----------|-----------:|-----|-------|
| `t_sac` | **0.99** | inside 95% bands | strongly identified |
| `eta` | **0.88** | inside 95% bands | strongly identified |
| `delta0` | **0.52** | inside 95% bands | moderately identified |
| `R` | **0.43** | inside 95% bands | moderately identified |

Posterior predictive check vs real VP10: duration 196.6 / 196.9 ms,
fixations/sentence 7.57 / 7.69, skip 23.1% / 19.6%, refixation 9.1% / 10.2%.

The two processing-dynamics parameters (`delta0`, `R`) are only moderately
identifiable from one participant's fixations — an honest, SBC-backed
identifiability finding, not a failure. Feeding skip/refixation statistics as
direct `inference_conditions` roughly doubled their recovery (was 0.32 / 0.26
with the LSTM summary alone). Before/after plots are in
`outputs/figures/baseline_M10/`.

## References

- Engbert & Rabe (2024). *A tutorial on Bayesian inference for dynamical
  modeling of eye-movement control during reading.* J. Math. Psych. 119, 102843.
- Rabe et al. (2021). *Bayesian parameter estimation for the SWIFT model.*
  Psychological Review 128(3), 516–543.
- BayesFlow v2 docs: https://bayesflow.org/v2.0.11/
```
