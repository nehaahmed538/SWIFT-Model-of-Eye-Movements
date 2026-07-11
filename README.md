# SWIFT Model — Amortized Inference with BayesFlow v2
### SBI Course Project — TU Dortmund

Amortized Bayesian inference for the **SWIFT model of eye movements during
reading**. The SWIFT simulator's likelihood is intractable, so we train a
BayesFlow neural posterior estimator on simulated `(θ, fixation-sequence)`
pairs, then apply it to real eye-tracking data (participant VP10) in
milliseconds.

**New to this project?** Read **[docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)**
first — it explains the problem, every file, every parameter, and the full
pipeline from zero, aimed at a teammate who hasn't touched the code yet.
For the full results write-up (not just the summary below), see
**[docs/RESULTS.md](docs/RESULTS.md)**.

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
  figures/                  ← all diagnostic plots (tracked in git)
  models/                   ← trained swift_approximator.keras (gitignored)
  results_summary.json      ← machine-readable snapshot from show_results.py
tools/
  calibrate.py              ← fast NumPy check of simulator vs real marginals
  show_results.py           ← read-only report of the saved model's results
docs/
  PROJECT_GUIDE.md          ← full beginner walkthrough (start here if new)
  RESULTS.md                ← full results write-up
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

# just want to SEE the current results without retraining? (read-only, ~30s)
python tools/show_results.py
```

| Mode / script | Does |
|------|------|
| `--mode generate` | pre-generate simulations (parallel) → `data/training_data.npz` |
| `--mode train` | train on saved data → model + diagnostics + VP10 inference + PPC |
| `--mode infer` | load saved model → diagnostics + VP10 inference + PPC |
| `--mode online` | train with the simulator in the loop (slow) |
| `--mode all` | `generate` → `train` |
| `tools/show_results.py` | load the saved model, print a full text report (no training) — see below |
| `tools/calibrate.py` | fast NumPy-only simulator-vs-VP10 check, no model needed |

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
   `M_SENTENCES=14` sentences, concatenated into a single sequence. A lone
   ~8-fixation sequence carries almost no information about `eta`/`delta0`/`R`;
   fourteen sentences do. This is the difference between those parameters being
   unrecoverable and recoverable.
2. **One shared normalisation.** `swift/config.normalise_sequence` is applied to
   *both* simulated and real fixations, so training and inference see identical
   input distributions.
3. **Four observable features** per fixation — `word_id, landing_position,
   duration, word_length` — the last one gives the network direct leverage on
   the word-length exponent `eta`.
4. **Hand-crafted statistic conditions.** `swift/config.compute_reader_stats`
   feeds skip rate, refixation rate, saccade amplitude, and more directly to
   the inference network — this is what makes `delta0`/`R` identifiable at
   all (see [docs/RESULTS.md](docs/RESULTS.md)).
5. **Pooled participant posterior.** VP10 inference draws many random
   14-sentence subsets, samples a posterior for each, and pools them — replacing
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
| `ppc_plot.png` | simulated vs real statistics (duration, landing, fixations/sentence, saccade amplitude, skip/refixation rate) | overlapping distributions |

---

## See the results

```bash
python tools/show_results.py
```

Loads the trained model (no retraining, ~30s) and prints a full report:
real VP10 data summary, parameter recovery + SBC calibration, VP10 posterior
estimates, and this posterior-predictive-check table —

```
===== PPC SUMMARY =====
Statistic                       Real   Simulated
------------------------------------------------
Mean duration (ms)            196.94      196.49
Std  duration (ms)             48.41       54.50
Mean fixations/sent             7.69        7.51
Mean landing pos                3.40        2.86
Mean saccade amp (words)        1.12        1.37
Skip rate (%)                  19.56       23.59
Refixation rate (%)            10.22        8.86
```

— plus a machine-readable copy at `outputs/results_summary.json`. Exact
numbers move slightly run to run (random 14-sentence resampling); see
[docs/RESULTS.md](docs/RESULTS.md) for the full discussion and typical
variation.

---

## Final results (VP10, M=14 readers, LSTM + statistic conditions)

Training loss 3.1 → **1.6**. Parameter recovery on 300 held-out simulations:

| Parameter | Recovery r | Contraction | 95% CI coverage | Notes |
|-----------|-----------:|------------:|-----------------:|-------|
| `t_sac` | **0.99** | 0.978 | 96.3% | strongly identified |
| `eta` | **0.88** | 0.744 | 94.3% | strongly identified |
| `delta0` | **0.52** | 0.313 | 93.0% | moderately identified |
| `R` | **0.43** | 0.112 | 95.0% | moderately identified |

VP10 posterior: `t_sac` 256 ms [233, 282], `eta` 0.62 [0.32, 0.92], `delta0`
8.3 chars [4.2, 14.2], `R` 0.38 [0.11, 0.85] — see "See the results" above
for the matching PPC table.

The two processing-dynamics parameters (`delta0`, `R`) are only moderately
identifiable from one participant's fixations — an honest, SBC-backed
identifiability finding, not a failure. Feeding skip/refixation statistics as
direct `inference_conditions` roughly doubled their recovery (was 0.32 / 0.26
with the LSTM summary alone). Before/after plots are in
`outputs/figures/baseline_M10/`. Full write-up, including what was tried and
didn't help: **[docs/RESULTS.md](docs/RESULTS.md)**.

## References

- Engbert & Rabe (2024). *A tutorial on Bayesian inference for dynamical
  modeling of eye-movement control during reading.* J. Math. Psych. 119, 102843.
- Rabe et al. (2021). *Bayesian parameter estimation for the SWIFT model.*
  Psychological Review 128(3), 516–543.
- BayesFlow v2 docs: https://bayesflow.org/v2.0.11/
