# Results — simplified SWIFT + BayesFlow amortized inference

Full write-up of what the trained model achieves. For "what is this project
and how does it work," see **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)**; for the
equation-by-equation simulator spec, **[MODEL_SPEC.md](MODEL_SPEC.md)**. Every
number on this page was produced by loading the currently saved model
(`outputs/models/swift_approximator.keras`) and running it — reproduce anytime
with:

```bash
python tools/show_results.py
```

(~20 seconds, read-only, no retraining). Figures are in
[`outputs/figures/`](../outputs/figures/). VP10 inference pools posteriors over
*randomly drawn* 14-sentence subsets and the PPC re-simulates from *randomly
drawn* posterior samples, so exact figures move by a point or two between runs;
the numbers below are one representative run (default seed).

> **2026-07-17 model.** This is the **basic 3-parameter** simplified SWIFT
> model of Engbert & Rabe (2024) — `nu`, `r`, `mu_T`, with the paper's own
> labels. It replaced an earlier simulator that implemented the *full* SWIFT
> (Gillespie algorithm, activation thresholds, landing positions) under
> reparametrised non-paper names (`t_sac`, `eta`, `delta0`, `R`), which the
> lecturer flagged. If you find those old names anywhere outside git history,
> they are stale.

---

## TL;DR

The final model — **M_SENTENCES=14, TimeSeriesNetwork summary + 7 hand-crafted
statistic conditions**, trained on 8000 simulated readers — recovers all three
parameters **strongly** (recovery r = 0.94 / 0.96 / 1.00 for `nu` / `r` /
`mu_T`), all within their 95% SBC coverage bands. On VP10 it infers
`mu_T ≈ 198 ms` (E[fixation duration] = `mu_T` exactly), `r ≈ 6.3`,
`nu ≈ 0.40`. The temporal ⊥ spatial decoupling the paper predicts holds
(posterior corr `mu_T` vs `nu`,`r` ≈ 0). Posterior-predictive simulations match
VP10's fixation durations (SFD/GD/TT all within ~6 ms) and skip/refixation
rates closely; the model **over-produces regressions** (10% vs VP10's 2%) and
**over-predicts duration spread** (structural CV = 1/3 vs VP10's 0.246). Both
are honest limitations of the simplified model, not bugs — and are what the
write-up should report.

---

## 1. The model in one paragraph

Words sit at discrete positions `1..N` (no spatial extent, no word length, no
Gillespie algorithm). On each fixation, an asymmetric 4-word processing span
`{k-1, k, k+1, k+2}` accrues activation at rate `r`; a sine-shaped saliency
rule (peaking at half-processed) picks the next target. Skipping, refixation
and regression are all **emergent** from that one rule — there is no explicit
mechanism for any of them, and adding one would double-count. Fixation
durations are independent Gamma draws (`shape α=9`, `mean μ_T`), giving a
fixed CV = 1/3 by construction. Three free parameters: `nu` (span shape), `r`
(processing rate), `mu_T` (mean duration). See
[MODEL_SPEC.md](MODEL_SPEC.md).

---

## 2. Real VP10 data (the benchmark)

From `data/fixseqin_PB2expVP10.dat`:

| Statistic | Value |
|---|---|
| Total fixations | 877 |
| Sentences | 114 |
| Mean fixation duration | 196.9 ± 48.4 ms |
| Duration CV | **0.246** |
| Mean fixations / sentence | 7.69 |
| Skip rate | 19.6% |
| Refixation rate | 10.2% |
| Regression rate | 2.04% |

(`outputs/figures/eda_fixations.png` — distributions behind these means.)

---

## 3. Parameter recovery & calibration

300 held-out simulations with **known** ground-truth θ, 1000 posterior draws
each (`outputs/figures/recovery_plot.png`, `sbc_histogram.png`,
`sbc_ecdf.png`, `contraction_plot.png`):

| Parameter | Recovery r | Posterior contraction | 95% CI coverage (nominal 95%) | Identifiability |
|---|---:|---:|---:|---|
| `nu`   | **0.941** | 0.866 | 95.0% | strong |
| `r`    | **0.959** | 0.904 | 96.0% | strong |
| `mu_T` | **0.997** | 0.990 | 95.3% | strong |

How to read this:
- **Recovery r** — Pearson correlation between posterior mean and true θ over
  the 300 validation simulations. `mu_T` sits almost exactly on the diagonal
  (it is the mean of a Gamma with known shape, so it is read directly off the
  durations); `nu` and `r` are noisier but clearly identified.
- **Contraction** — `1 − posterior_var / prior_var`. `mu_T`'s 0.990 means its
  posterior is ~99% narrower than the prior; even `nu`'s 0.866 is strong.
- **95% CI coverage** — fraction of the 300 draws where the true value fell
  inside the posterior's own 95% interval. All three land at ~95%, confirming
  (via SBC, `sbc_*.png`) that the network's uncertainty estimates are honest.

Note vs. the paper's Fig. 7: the paper reports `nu` as its *hardest* parameter
with a right-skewed posterior. Adding the 7 hand-crafted reading-measure
statistics as direct network conditions (skip/refix/regression rates etc.)
lifts `nu` here to strong recovery — the summary LSTM alone does not reliably
extract that scanpath signal from the raw sequence.

---

## 4. VP10 posterior estimates

Pooled over 40 random 14-sentence draws of VP10's own data (train split),
2000 posterior samples per draw (`outputs/figures/posterior_VP10.png`):

| Parameter | Mean | 95% CI | Prior range |
|---|---:|---|---|
| `nu`   | 0.40   | [0.28, 0.54]        | [0.0, 1.0] |
| `r`    | 6.33   | [5.27, 7.87]        | [0.0, 12.0] |
| `mu_T` | 198.0 ms | [182.0, 216.1] ms | [100, 400] ms |

`mu_T ≈ 198 ms` matches VP10's mean fixation duration of 196.9 ms almost
exactly — expected, because in this model **E[fixation duration] = mu_T** by
construction. None of the posteriors are pegged against a prior boundary.

### Decoupling check (paper Section 4.1)

The basic model decouples timing (`mu_T`) from scanpath (`nu`, `r`). The VP10
posterior correlation matrix confirms it (`outputs/figures/posterior_correlation.png`):

```
              nu       r    mu_T
nu         1.000  -0.317   0.007
r         -0.317   1.000   0.061
mu_T       0.007   0.061   1.000
```

`mu_T` vs (`nu`, `r`) ≈ 0.007 / 0.061 ≈ 0 — durations carry no information
about the scanpath and vice versa, exactly as the paper predicts. (`nu` and
`r` trade off with each other, −0.317, since both shape processing.)

---

## 5. Posterior predictive check (PPC)

Simulate new data from 300 draws of the VP10 posterior above, compare reading
measures against VP10's real data — on the **held-out second half of sentences**
(paper Section 6: train on first-half sentences, PPC on second-half). Never
raw sequences. Plot: `outputs/figures/ppc_plot.png`.

```
===== PPC SUMMARY =====
Statistic                       Real   Simulated
------------------------------------------------
Mean SFD (ms)                 202.52      204.38
Mean GD  (ms)                 213.64      212.34
Mean TT  (ms)                 214.38      220.19
P(skip) (%)                    20.55       18.06
P(refixation) (%)               9.09        7.71
P(regression) (%)               1.82       10.11
```

SFD (single-fixation duration), GD (gaze duration) and TT (total time) all
match within ~6 ms. Skip and refixation rates are within a couple of points.
The one clear miss is **P(regression): 10.1% simulated vs 1.82% real** — see
§6.

Run-to-run variation at `n_ppc=300`: durations within ~±2 ms, probabilities
within 1–2 percentage points — small relative to the regression gap itself.

---

## 6. Deviations from Engbert & Rabe (2024)

Honest list for the report — where this implementation departs from the paper,
and why.

- **Time unit: activation uses seconds, not milliseconds.** The paper states
  durations in ms throughout, but its `r` values (5–10) only produce sensible
  multi-fixation processing if `r·λ·T` uses `T` in **seconds**. With `T` in ms,
  `r·λ·T` reaches the hundreds and every in-span word saturates in a single
  fixation, which collapses the sine-saliency rule and makes the scanpath
  independent of `nu` and `r` (verified: skip/refix/regression rates went flat
  across all parameter values). Converting `T` to seconds in the activation
  update restores parameter-dependent behaviour matching the paper's Sections
  2.4/3. This is the single interpretive choice needed to reconcile the paper's
  ms wording with its `r` values. See [MODEL_SPEC.md](MODEL_SPEC.md) "Unit note".
- **`beta = 0.6` fixed, not free.** The word-frequency effect on maximum
  activation is fixed at the paper's Section-5 recovery value. `beta = 0` (no
  frequency effect, strict Section-3 baseline) is the documented alternative;
  freeing `beta` belongs to the 5-parameter extended model.
- **Duration spread is structurally over-predicted.** The Gamma timer fixes
  CV = 1/√α = 1/3 = 0.333 by construction, but VP10's real duration CV is
  **0.246** — the model's durations are ~35% too dispersed. This is a genuine
  limitation of the basic model (the shape `α` is not free), worth reporting as
  a finding rather than tuning away.
- **Regressions over-produced (10% vs 2%).** Regressions are emergent from the
  saliency rule with no suppression mechanism, so the model regresses more than
  VP10 does. The paper's basic model has the same structure; an explicit
  regression/inhibition term (not added here — it would double-count against
  the emergent rule) is the natural extension.
- **`iota` not implemented.** The timer-coupling term (extended model, Eq. 22,
  `rate' = rate·(1 + iota·a_k)`) is omitted — the basic model keeps durations
  fully independent of processing. Noted as future work.

---

## 7. Assignment requirements checklist

Checked against the official brief (Simon Kucharsky, TU Dortmund):

| Requirement | Status | Where |
|---|---|---|
| Implement the **simplified** SWIFT model (Engbert & Rabe, 2024) — not the full version | ✅ | `swift/simulator.py`; basic 3-parameter model, discrete positions, no Gillespie |
| Model has control for **fixation duration** | ✅ | Gamma saccade timer (`mu_T`, `alpha`) — MODEL_SPEC Eq. 10–12 |
| Model has control for **saccades** (where the eye goes next) | ✅ | Sine-saliency target rule — MODEL_SPEC Eq. 8–9 |
| Implement it **in BayesFlow** | ✅ | `swift/inference.py`, BayesFlow v2.0.11 |
| Estimate parameters related to **gaze control and reading dynamics** | ✅ | `nu`, `r`, `mu_T` — §3 |
| Real eye-tracking data, controlled reading experiment (osf.io/teyd4) | ✅ | `data/fixseqin_PB2expVP10.dat`, VP10, 877 fixations / 114 sentences |
| Corpus linking fixations to word properties (osf.io/nj2mf) | ✅ | `data/Rcorpus_PB2_revision.dat`, supplies word frequency (drives `a_max`) |
| Investigate fit to **fixation durations** | ✅ | PPC §5 — SFD / GD / TT |
| Investigate fit to **movement patterns** | ✅ | PPC §5 — P(skip), P(refixation), P(regression) |
| Train / test split (paper Section 6) | ✅ | first-half sentences train, second-half PPC (`swift/data.split_half`) |

**Open item** (per this repo's convention of flagging rather than guessing):
the corpus `code` column is read but unused downstream — 4 distinct values
(`9` on 661/1003 words; `0`/`1`/`2` each on exactly 114/1003, one word per
sentence per code), most likely pretarget/target/posttarget markers from the
original boundary-paradigm display-change manipulation, which this continuous-
reading model does not simulate. Not confirmed against the source paper (OSF
pages are JS-rendered and returned no content to automated fetches). Documented
assumption, not a verified fact.

---

## 8. References

- Engbert, R., & Rabe, M. M. (2024). *A tutorial on Bayesian inference for
  dynamical modeling of eye-movement control during reading.* Journal of
  Mathematical Psychology, 119, 102843.
- Rabe, M. M., et al. (2021). *A Bayesian approach to dynamical modeling of
  eye-movement control in reading of normal, mirrored, and scrambled texts.*
  Psychological Review, 128(3), 516–543.
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018).
  *Validating Bayesian Inference Algorithms with Simulation-Based
  Calibration.* arXiv:1804.06788.
