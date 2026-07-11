# CLAUDE.md — SWIFT Model / BayesFlow Project (SBi Course, Project 3)

This file gives Claude Code the context it needs to work on this repo without
re-explaining the project every session. Read this fully before making changes.

## 1. Project Summary

Course: **Simulation-Based Inference (SBI)**, M.Sc. Data Science
Assignment: **Project 3 — The SWIFT Model of Eye Movements**
Group size: 3 people
Core requirement: implement **amortized Bayesian inference using the BayesFlow
library** for a model whose likelihood is intractable.

The scientific goal: when people read, their eyes move in a pattern of
fixations (pauses) and saccades (jumps). The **SWIFT model** (simplified
version from **Engbert & Rabe, 2024**) is a stochastic, simulator-based
cognitive model that explains this pattern via word-activation dynamics. We
cannot compute its likelihood analytically, so we:

1. Build a **forward simulator** (SWIFT + Gillespie algorithm): parameters θ → simulated fixation sequence.
2. Train **BayesFlow** to learn the **inverse** mapping: fixation sequence → posterior over θ. This is *amortized* inference — pay the training cost once, then get a posterior for any new sequence in milliseconds.
3. Apply the trained network to **real eye-tracking data** (participant VP10) and validate the results.

### 1.1 Official brief (verbatim, from lecturer Simon Kucharsky, simon.kucharsky@tu-dortmund.de)

Received 2026-07-11, pasted in full so future sessions have the ground truth
rather than a paraphrase. The three concrete deliverable sentences are
**bolded** below; everything else is motivating context.

> In reading research, eye-tracking provides data relevant to how people
> process written language. One of the key questions is how readers decide
> when to move their eyes and where to look next. These decisions are
> influenced by both the cognitive state of the reader and properties of the
> text, such as word frequency or sentence structure. Modeling these
> dynamics statistically is challenging due to the complexity and
> variability of eye movement behavior.
>
> One approach to modeling these processes is the SWIFT model, a dynamic
> generative model of eye movement control during reading. It simulates how
> a reader's gaze shifts across a sentence as they process its content. The
> model incorporates timing and control mechanisms for both fixation
> duration (how long the eye stays on a word) and saccades (eye movements to
> the next word). The full version of SWIFT is computationally intensive and
> has an intractable likelihood. However, you can work with a simplified
> version of the SWIFT model introduced by Engbert & Rabe (2024).
> **Implement the simplified SWIFT model in BayesFlow. Use eye-tracking data
> to estimate model parameters related to gaze control and reading
> dynamics. Investigate how well the model captures the observed fixation
> durations and movement patterns.**
>
> You will work with real eye-tracking data from a controlled reading
> experiment. One example dataset, available at https://osf.io/teyd4,
> contains fixation sequences for an individual participant. You can also
> link the fixation data to word properties using the corpus file from
> https://osf.io/nj2mf.

Compliance check against this brief (verified 2026-07-11 — see
`docs/RESULTS.md` §7 for the full table with file/line evidence): simplified
(not full) SWIFT ✅, implemented in BayesFlow ✅, four free parameters map
directly onto "gaze control and reading dynamics" ✅, both OSF datasets used
✅, PPC investigates both fixation durations ✅ and movement patterns ✅
(landing position, saccade amplitude, skip rate, refixation rate — saccade
amplitude was added to close a gap where it was computed for the network but
never actually validated against real VP10 data in the PPC). One open item:
the corpus's `code` column is unused and its semantics haven't been
confirmed against the source paper — see §3.

## 2. The Four Target Parameters (θ)

Confirmed free parameters (the group settled on `R`, not `h`, as the 4th):

| Symbol | Meaning | Prior range | Observable it drives |
|---|---|---|---|
| `t_sac` | mean saccade-timer period (fixation-duration scale) | 150–350 ms | fixation duration |
| `eta` | word-length processing exponent | 0.1–1.0 | fixation count / refixation |
| `delta0` | processing span width (attention window) | 4–15 characters | skipping rate |
| `R` | refixation-rate factor | 0.1–0.9 | refixation rate |

`h` is now a **fixed** foveal-release constant, not inferred. All fixed
constants (`alpha, beta, h, gamma, kappa, rho, sigma, omega`) are calibrated so
the simulator's prior-averaged marginals match VP10 (`python tools/calibrate.py`).
Parameter names are defined once in `swift/simulator.py` (`PARAM_NAMES`).

## 3. Data Sources

- **Corpus file** (`osf.io/nj2mf`, `Rcorpus_PB2_revision.dat`) — word
  properties (frequency, length) for 114 German sentences. This is an
  **input to the simulator**, not something we run inference on.
  Tab-separated **with a header row** (`sentID nw wordID length freq code`;
  the header line ends in a lone `\r` rather than `\n`, which makes it look
  merged with the first data row in a plain text editor — `pandas.read_csv`
  handles this fine, don't hand-edit the file). `length` and `freq` are used
  (see `swift/data.load_corpus` / `build_corpus_lists`); `code` is **not**
  used anywhere downstream. It takes exactly 4 values — `9` on 661/1003
  words (the majority) and `0`/`1`/`2` each on exactly 114/1003 words (one
  word per sentence per code) — most likely marking pretarget/target/
  posttarget positions from the original boundary-paradigm display-change
  manipulation, which this project's continuous-reading simplified SWIFT
  model doesn't simulate. **Not confirmed against the source paper** (OSF
  page is a JS-rendered SPA that returns no content to automated fetches) —
  flagging per the policy below rather than silently assuming it's safe to
  ignore.
- **Fixation sequence file** (`osf.io/teyd4`, `fixseqin_PB2expVP10.dat`) — real
  fixation data for participant **VP10**. No headers. Confirmed column mapping:

  | Col | Field | Notes |
  |---|---|---|
  | 1 | `sentence_id` | 1–114 |
  | 2 | `word_id` | word position in sentence |
  | 3 | `landing_position` | decimal (jittered), letter position within word |
  | 4 | `fixation_duration` | ms, ~80–400 range |
  | 5 | `word_length` | characters |
  | 6 | `fixation_type` | 0=middle, 1=first fixation in sentence, 2=last |
  | 7 | `flag1` | always 0 in this file, likely unused/regression flag |
  | 8 | `flag2` | always 0 in this file, likely unused condition flag |
  | 9 | `fixation_index` | sequential count within sentence |
  | 10 | `participant_id` | always 10 |

  This file is **real observed data** and must only be touched at the final
  inference / posterior-predictive-check stage — never during simulator
  development or BayesFlow training.

⚠️ If either file's column semantics are ever uncertain, re-derive from the
source paper (Rabe et al. 2021 boundary-paradigm paper) rather than guessing —
do not silently change column mappings without flagging it.

## 4. Repo Structure

All source now lives in one importable package, `swift/`; generated artifacts
live under `outputs/` (kept out of git except figures). Old top-level
`data/load_data.py`, `simulator/`, `inference/`, `diagnostics/` dirs were
consolidated into the package.

```
swift/
  config.py         # paths, sequence layout, ONE shared feature normalisation
  simulator.py      # SWIFT dynamics + Gillespie algorithm (forward model)
  data.py           # load .dat files, EDA, real→observation conversion
  generate.py       # parallel pre-generation of (theta, sequence) pairs
  inference.py      # BayesFlow workflow: train + posterior sampling (forces CPU)
  diagnostics.py    # recovery, SBC, contraction, posterior, PPC (incl. skip/refix/saccade amp)
data/               # .dat files only + generated training_data.npz (gitignored)
outputs/figures/    # all diagnostic plots (tracked in git)
outputs/models/     # trained swift_approximator.keras (gitignored)
outputs/results_summary.json  # machine-readable snapshot from tools/show_results.py
tools/
  calibrate.py      # fast NumPy check: simulator marginals vs real VP10
  show_results.py   # read-only report of the saved model (no training), ~30s
docs/
  PROJECT_GUIDE.md  # full beginner walkthrough — file-by-file, parameter-by-parameter
  RESULTS.md        # full results write-up + assignment compliance checklist
main.py             # CLI: --mode generate | train | infer | online | all
pyproject.toml / requirements.txt
```

Dedicated conda env (Apple Silicon / M2, native arm64, torch on CPU):
`/opt/anaconda3/envs/swift-sbi/bin/python main.py --mode all`
(create with `conda create -n swift-sbi python=3.12` then
`pip install -r requirements.txt`). Always set `KERAS_BACKEND=torch`.

**Key design decision:** one training example = one reader (fixed θ) reading
`M_SENTENCES=14` sentences, fixations concatenated (see `config.SEQ_LEN`,
`simulator.run_one_reader`). A single ~8-fixation sequence cannot identify
`eta`/`delta0`/`R`; fourteen sentences can. VP10 inference pools posteriors over
many random 14-sentence subsets (`data.build_reader_batch`) instead of
averaging raw sequences. (`M_SENTENCES` was raised from 10 to 14 during
development — if you see "10" anywhere outside this repo's git history, it's
stale.)

## 5. Critical Version Note — BayesFlow API

This project targets **BayesFlow v2.x**, which is a full rewrite of v1.x:

- v1.x: TensorFlow backend, `bf.networks.InvertibleNetwork`, `SingleModelAmortizer`, `Trainer`
- **v2.x (current, use this): PyTorch backend**, `bf.Approximator`, `bf.networks.CouplingFlow`, `bf.networks.LSTM`, etc.

Before touching `inference/train_bayesflow.py` or `diagnostics/diagnostics.py`:
1. Run `python -c "import bayesflow; print(bayesflow.__version__)"` and confirm the actual installed version.
2. If it's not v2.x, **stop and flag this** rather than silently patching to the wrong API — the rest of the codebase assumes v2.x.
3. Check current BayesFlow docs/examples before assuming API shape from memory — this library's API changes quickly and training data may be stale.

## 6. Pipeline Order (do not reorder)

1. **Load & explore (EDA)** — fixation duration distributions, skipping rate, sentence lengths. These become posterior-predictive-check benchmarks later.
2. **Build & validate simulator** — check it produces plausible fixation sequences (durations roughly 150–300ms with real corpus data) before connecting anything to BayesFlow.
3. **Generate training pairs** — sample θ from priors, run simulator, produce (θ, sequence) pairs. Pure simulation, no ML, no real data involved.
4. **Train BayesFlow** — summary network compresses variable-length sequences to a fixed vector; posterior network (normalizing flow) maps that to P(θ | data). Trained only on simulated pairs.
5. **Diagnostics** — Simulation-Based Calibration (SBC) coverage check, posterior contraction (posterior should be narrower than prior).
6. **Real-data inference** — feed VP10's real sequence into the trained network. This is the *first* point real data is used.
7. **Posterior predictive check** — simulate new data from VP10's inferred θ, compare **summary statistics** (fixation duration histograms, skipping rate, refixation rate, saccade length distributions) against real VP10 data. Never compare raw sequences directly.

## 7. Conventions & Preferences

- Prefer NumPy/pure Python for the simulator — no unnecessary dependencies. The simulator is meant to be a direct, readable translation of the paper's equations, not a wrapper around some external library.
- Don't convert `.dat` files to Excel/CSV as an intermediate step — read directly with `pandas.read_csv(..., sep=r"\s+", header=None)`.
- All summary statistics used for posterior predictive checks should be plottable/quantitative — never manual/eyeball comparisons.
- When editing simulator equations, cite the specific section/equation number from Engbert & Rabe (2024) in a comment.
- Keep parameter names consistent with Section 2 above across all files (`t_sac`, `h`, `delta_0`, `eta`) — don't introduce silent renames.

## 8. Deliverables & Deadlines

| Date | Deliverable |
|---|---|
| July 19 | Presentation slides PDF (`<group-number>.pdf`) |
| July 20–24 | Live presentation, 12 min for group of 3 |
| Aug 23 | Final report + code, submitted as zip |

Presentation structure required: intro → statistical model → BayesFlow setup
→ results → TL;DR slide. No "Thank You" slide. Expect general SBI-course
questions after presenting, not just project-specific ones.

## 9. Open Questions / Things to Confirm Before Finalizing

- [x] Corpus columns confirmed: tab-separated with header
      `sentID nw wordID length freq code` (see `swift/data.load_corpus`; §3).
- [x] 4th free parameter is **`R`** (refixation factor); `h` is fixed.
- [x] Installed BayesFlow is **2.0.11** (torch backend, forced to CPU because
      Apple MPS lacks `linalg_qr`).
- [x] PPC statistics: fixation duration, landing position, fixations/sentence,
      **saccade amplitude**, **skip rate**, and **refixation rate**
      (`swift/diagnostics.posterior_predictive_check`). Saccade amplitude was
      added 2026-07-11 — it was already computed for the network's
      `inference_conditions` but was never actually validated against real
      VP10 data in the PPC, which was a gap against the brief's "investigate
      ... movement patterns" requirement (§1.1).
- [x] Official assignment brief received verbatim from the lecturer
      2026-07-11 and cross-checked against the implementation — see §1.1 and
      `docs/RESULTS.md` §7 for the full compliance table. No gaps found other
      than the saccade-amplitude one above (now fixed) and the open item
      below.
- [ ] **Open**: corpus `code` column semantics unconfirmed (see §3) — not
      blocking (best guess is it's irrelevant to this project's continuous-
      reading model), but flagged rather than silently assumed.

Final model (M=14, LSTM summary + hand-crafted statistic `inference_conditions`):
recovery r = t_sac 0.99, eta 0.88, delta0 0.52, R 0.43; all four inside the SBC
95% bands; PPC matches VP10 closely (duration, fixation count, and saccade
amplitude match well; skip/refixation rate are in the right neighbourhood but
not exact, consistent with delta0/R's moderate identifiability). Exact numbers
have run-to-run noise (random sentence resampling) — get the current numbers
anytime with `python tools/show_results.py` (~30s, read-only, no retraining)
rather than trusting hardcoded figures here; full write-up in
`docs/RESULTS.md`. The statistic conditions (skip rate, refixation rate,
saccade amplitude, …) are what make delta0/R identifiable — an LSTM over the
raw sequence alone gave only 0.32/0.26. To push delta0/R further, raise
`M_SENTENCES` (more evidence per reader); do NOT distort the fixed simulator
constants to inflate the signal (tried — it broke the PPC). Fixed-constant
calibration matches VP10 marginals but is not a formal fit; worth a sentence
in the report.
