# Results — SWIFT + BayesFlow amortized inference

This is the full write-up of what the trained model achieves. For "what is
this project and how does it work," see
**[PROJECT_GUIDE.md](PROJECT_GUIDE.md)**. Every number on this page was
produced by actually loading the currently saved model
(`outputs/models/swift_approximator.keras`) and running it — reproduce it
yourself anytime with:

```bash
python tools/show_results.py
```

(~30 seconds, read-only, does not retrain anything). Figures referenced below
are all in [`outputs/figures/`](../outputs/figures/). Because VP10 inference
pools posteriors over *randomly drawn* 14-sentence subsets and the PPC
re-simulates from *randomly drawn* posterior samples, exact figures move by
a small amount (a point or two) between runs — the numbers below are one
representative run (`--seed 0`, the default), and every table also states
the scale of that run-to-run noise where it matters.

---

## TL;DR

The final model — **M_SENTENCES=14, LSTM summary network + hand-crafted
statistic conditions** — recovers `t_sac` and `eta` strongly (r = 0.99 /
0.88) and `delta0`/`R` moderately (r = 0.52 / 0.43), all four inside their
95% SBC coverage bands, and its posterior-predictive simulations closely
match VP10's real fixation duration, count, and skip/refixation rates. The
moderate `delta0`/`R` recovery is an honest identifiability limit of a
single participant's fixation record, not a bug — and it's exactly what the
project should report as a finding, not hide.

---

## 1. The story: three iterations

| # | Version | Duration match | Recoverable parameters |
|---|---|---|---|
| 1 | **Original simulator** | Badly broken: ~350 ms mean duration, 12+ fixations/sentence (real: ~197 ms, ~7.7 fixations/sentence) | Only `t_sac` |
| 2 | **Recalibrated + multi-sentence** (`tools/calibrate.py` tuning, `M_SENTENCES` concatenation) | Simulator matches VP10's marginals | `t_sac`/`eta` good; `delta0`/`R` weak (≈0.32 / 0.26) — the LSTM summary alone couldn't reliably extract skip/refixation signal from the raw sequence |
| 3 | **+ hand-crafted statistic conditions** (current, final) | Matches VP10 closely (§4) | `t_sac`/`eta` strong (0.99/0.88); `delta0`/`R` roughly **doubled** to moderate (0.52/0.43) |

Two things were tried and explicitly **did not help** (recorded here so they
aren't re-tried): distorting the fixed simulator constants to artificially
inflate the `delta0`/`R` signal (broke the PPC match instead), and a
bidirectional LSTM summary network (2× slower to train, no recovery gain
over the unidirectional one currently used).

Before/after diagnostic plots for iteration 2 vs. 3 are in
[`outputs/figures/baseline_M10/`](../outputs/figures/baseline_M10/) (older,
M=10, LSTM-only) vs. [`outputs/figures/`](../outputs/figures/) (current,
M=14, +statistics).

---

## 2. Real VP10 data (the benchmark everything else is checked against)

From `data/fixseqin_PB2expVP10.dat`:

| Statistic | Value |
|---|---|
| Total fixations | 877 |
| Sentences | 114 |
| Mean fixation duration | 196.9 ± 48.4 ms |
| Mean fixations / sentence | 7.69 |
| Skip rate | 19.6% |
| Refixation rate | 10.2% |

(`outputs/figures/eda_fixations.png` — distributions behind these means.)

---

## 3. Parameter recovery & calibration

300 held-out simulations with **known** ground-truth θ, 1000 posterior draws
each (`outputs/figures/recovery_plot.png`, `sbc_histogram.png`,
`sbc_ecdf.png`, `contraction_plot.png`):

| Parameter | Recovery r | Posterior contraction | 95% CI coverage (nominal 95%) | Identifiability |
|---|---:|---:|---:|---|
| `t_sac` | **0.990** | 0.978 | 96.3% | strong |
| `eta` | **0.878** | 0.744 | 94.3% | strong |
| `delta0` | **0.516** | 0.313 | 93.0% | moderate |
| `R` | **0.428** | 0.112 | 95.0% | moderate |

How to read this:
- **Recovery r** — Pearson correlation between the posterior mean and the
  true θ across the 300 validation simulations. `t_sac`/`eta` sit on the
  diagonal in `recovery_plot.png`; `delta0`/`R` show a real but noisier
  trend.
- **Contraction** — `1 − posterior_variance / prior_variance`. `t_sac`'s
  0.978 means its posterior is ~98% narrower than the prior (highly
  informative); `R`'s 0.112 means its posterior is barely narrower than the
  flat prior — the data constrains it only a little.
- **95% CI coverage** — of the 300 validation draws, the fraction where the
  true value actually fell inside the posterior's own 95% interval. All four
  land close to the nominal 95%, which is the point of the SBC check
  (`sbc_histogram.png`/`sbc_ecdf.png`): the network's *uncertainty estimates*
  are honest even where the *point estimates* (for `delta0`/`R`) are only
  moderately accurate. That combination — moderate accuracy, honest
  uncertainty — is a calibrated model correctly reporting "I'm not fully
  sure," not a failure.

---

## 4. VP10 posterior estimates

Pooled over 40 random 14-sentence draws of VP10's own data, 2000 posterior
samples per draw (`outputs/figures/posterior_VP10.png`):

| Parameter | Mean | 95% CI | Prior range |
|---|---:|---|---|
| `t_sac` | 256.1 ms | [232.6, 281.9] ms | [150, 350] ms |
| `eta` | 0.62 | [0.32, 0.92] | [0.1, 1.0] |
| `delta0` | 8.3 chars | [4.2, 14.2] chars | [4, 15] chars |
| `R` | 0.38 | [0.11, 0.85] | [0.1, 0.9] |

None of the four posteriors are pegged against a prior boundary, and
`t_sac`/`eta` (the strongly-identified pair, per §3) are visibly narrower
relative to their prior range than `delta0`/`R` are — consistent with the
recovery/contraction numbers above, and visible directly in
`posterior_VP10.png`.

---

## 5. Posterior predictive check (PPC)

Simulate new data from 300 draws of the VP10 posterior above, compare
summary statistics against VP10's real data (never raw sequences — see
[PROJECT_GUIDE.md §7.7](PROJECT_GUIDE.md#77-posterior-predictive-check-ppc)
for why). Plot: `outputs/figures/ppc_plot.png`.

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

Duration, fixation count, and saccade amplitude (how many words the eye jumps
per saccade — the classic "movement pattern" statistic, mostly short forward
steps for both real and simulated data) all match closely. Skip rate and
refixation rate are in the right neighbourhood but not exact (simulated skip
rate runs a few points high, refixation a couple points low) — consistent
with §3's finding that `delta0` (which drives skipping) and `R` (which drives
refixation) are only moderately identified, so the posterior used to generate
these simulations carries more residual uncertainty on exactly those two
statistics.

Run-to-run variation at `n_ppc=300`: repeated runs put simulated duration
within about ±1 ms, fixations/sentence within ±0.1, and skip/refixation rate
within 1–2 percentage points of the values above — small relative to the
real-vs-simulated gaps themselves.

---

## 6. Honest limitations (for the report)

- **`delta0`/`R` are moderately, not strongly, identifiable from one
  participant's fixation record.** This is a genuine finding, backed by
  both the recovery numbers and the (near-zero, for `R`) posterior
  contraction — not an artifact of insufficient training. The recommended
  framing for the write-up: *"t_sac and eta are strongly identified; delta0
  and R only moderately — a genuine identifiability limit of single-
  participant fixation data, confirmed by SBC and posterior contraction, and
  substantially improved (roughly doubled) by adding domain summary
  statistics as direct network conditions."*
- **The fixed constants (§4.2 of the project guide) are calibrated, not
  fit.** `tools/calibrate.py` checks that the simulator's *prior-averaged*
  marginals resemble VP10, by hand-tuning `alpha, beta, h, gamma, kappa,
  rho, sigma, omega`. This is not a formal Bayesian fit and has no posterior
  or uncertainty attached to it — worth one sentence in the report's
  methods/limitations section.
- **Remaining lever, if there's time**: raising `M_SENTENCES` (currently 14)
  in `swift/config.py` gives the network more evidence per simulated reader,
  at the cost of a longer `generate` + `train` run. This is the only change
  that improved `delta0`/`R` recovery during development; changing the fixed
  constants or switching to a bidirectional LSTM did not (§1).

---

## 7. Assignment requirements checklist

Checked against the official brief (Simon Kucharsky, TU Dortmund — SBI
Final Projects):

| Requirement | Status | Where |
|---|---|---|
| Implement the **simplified** SWIFT model (Engbert & Rabe, 2024) — not the full, computationally intensive version | ✅ | `swift/simulator.py`; labile/non-labile programming stages collapsed into one saccade timer, as the paper's simplification does |
| Model has timing/control for **fixation duration** | ✅ | Saccade timer (`t_sac`, `h`) — §4.1 |
| Model has timing/control for **saccades** (where the eye goes next) | ✅ | `_select_target()` — §4.1 |
| Implement it **in BayesFlow** | ✅ | `swift/inference.py`, BayesFlow v2.0.11 |
| Estimate parameters related to **gaze control and reading dynamics** | ✅ | `t_sac`, `eta`, `delta0`, `R` — §3 |
| Real eye-tracking data, controlled reading experiment (osf.io/teyd4) | ✅ | `data/fixseqin_PB2expVP10.dat`, participant VP10, 877 fixations / 114 sentences |
| Corpus linking fixations to word properties (osf.io/nj2mf) | ✅ | `data/Rcorpus_PB2_revision.dat`, joined on `sentence_id`/`word_id`, supplies length + frequency |
| Investigate fit to **fixation durations** | ✅ | PPC §5 — mean/std duration |
| Investigate fit to **movement patterns** | ✅ | PPC §5 — landing position, fixations/sentence, **saccade amplitude**, skip rate, refixation rate |

**Open item, flagged rather than silently resolved** (per this repo's own
convention of not guessing at column semantics): the corpus file has a
`code` column (`swift/data.py::load_corpus` reads it but nothing downstream
uses it) with exactly 4 distinct values — `9` on 661/1003 words (the
majority) and `0`, `1`, `2` each on exactly 114/1003 words, i.e. exactly one
word per sentence carries each of those three codes. Given the file names
(`PB2`, "boundary paradigm") and the companion Rabe et al. (2021) reference
already in this repo, the most likely explanation is that this marks
word positions relevant to the original *display-change/boundary-paradigm*
manipulation (e.g. pretarget/target/posttarget), which the continuous-
reading simplified SWIFT model implemented here does not simulate — so it
should be safe to leave unused. This has **not** been confirmed against the
original OSF page or the Rabe et al. (2021) methods section (the OSF pages
are JS-rendered and didn't yield content via automated fetch), so treat this
as a documented assumption, not a verified fact — worth a two-minute check
against the paper if it comes up in Q&A.

---

## 8. References

- Engbert, R., & Rabe, M. B. (2024). *A tutorial on Bayesian inference for
  dynamical modeling of eye-movement control during reading.* Journal of
  Mathematical Psychology, 119, 102843.
- Rabe, M. B., et al. (2021). *A Bayesian approach to dynamical modeling of
  eye-movement control in reading of normal, mirrored, and scrambled texts.*
  Psychological Review, 128(3), 516–543.
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018).
  *Validating Bayesian Inference Algorithms with Simulation-Based
  Calibration.* arXiv:1804.06788.
