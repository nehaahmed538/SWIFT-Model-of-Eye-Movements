# Project Guide — SWIFT Model + BayesFlow, explained from zero

This document is for a teammate joining the project who has **not** read the
code yet. It explains, in order: what problem we're solving, every file in
the repo and why it exists, every parameter (free and fixed) and what it
physically means, the two data files, the full pipeline stage by stage, how
to run it, how to read every plot it produces, and a glossary of the SBI/
BayesFlow jargon used throughout. For the numeric results themselves (what
the trained model actually achieves), see **[RESULTS.md](RESULTS.md)**. For
a condensed quick-start, see the top-level **[README.md](../README.md)**. For
the equation-by-equation simulator spec, see **[MODEL_SPEC.md](MODEL_SPEC.md)**.

> **2026-07-17 model.** This project implements the **basic 3-parameter**
> simplified SWIFT model of Engbert & Rabe (2024) — parameters `nu`, `r`,
> `mu_T` (the paper's own labels). It replaced an earlier simulator that
> implemented the *full* SWIFT (Gillespie algorithm, activation thresholds,
> word length, landing positions) under reparametrised non-paper names
> (`t_sac`, `eta`, `delta0`, `R`). If you find those old names anywhere
> outside git history, they are stale.

---

## Table of contents

1. [The problem, in plain English](#1-the-problem-in-plain-english)
2. [The big picture](#2-the-big-picture)
3. [Repo map — every file and why it exists](#3-repo-map--every-file-and-why-it-exists)
4. [The SWIFT model's parameters](#4-the-swift-models-parameters)
5. [The two data files](#5-the-two-data-files)
6. [Key engineering constants](#6-key-engineering-constants-swiftconfigpy)
7. [The pipeline, stage by stage](#7-the-pipeline-stage-by-stage)
8. [How to run everything](#8-how-to-run-everything)
9. [How to read every plot](#9-how-to-read-every-plot)
10. [Glossary](#10-glossary)
11. [Troubleshooting / FAQ](#11-troubleshooting--faq)
12. [References](#12-references)

---

## 1. The problem, in plain English

When you read a sentence, your eyes don't glide smoothly across the text.
They jump (**saccades**) and pause (**fixations**), sometimes skip short or
predictable words entirely, and sometimes jump backward to re-read something
(**refixations**/regressions). This pattern is not random — it reflects how
your brain is processing each word: hard or rare words get looked at longer
and more often, easy or short words get skipped.

The **SWIFT model** (Engbert & Rabe, 2024; simplified version used here) is a
cognitive model that tries to *explain* this pattern with a small set of
interpretable parameters — e.g. "how wide is your attention window while
reading" or "how much does word length slow you down." Given a parameter
setting, the model can *simulate* a plausible sequence of fixations.

The scientific question we actually care about is the **reverse** direction:
given a *real* person's fixation sequence (participant VP10), what parameter
values best explain how *they* read? This is a classic **inverse problem**,
and it's hard for one specific reason:

> **The SWIFT model has no tractable likelihood.** There is no formula for
> "the probability of observing this exact fixation sequence given these
> parameters." The model is a stochastic simulator — you can *run* it
> forward (parameters → a simulated sequence) as many times as you like, but
> you cannot evaluate backward (sequence → probability) directly.

This is exactly the situation **Simulation-Based Inference (SBI)** is for.
Instead of writing down a likelihood, we:

1. Simulate **many** `(parameters, fixation sequence)` pairs by running the
   simulator with parameters drawn from a prior.
2. Train a neural network (**BayesFlow**) on those pairs to learn the
   *inverse* mapping: sequence → posterior distribution over parameters.
3. Because that network is trained once and reused for any new sequence,
   this is called **amortized inference** — pay the (large) training cost
   once, then get a posterior for VP10 (or anyone else) in milliseconds,
   instead of re-running expensive inference from scratch every time.

That's the whole project: build the simulator (§7.2), generate training
pairs (§7.3), train BayesFlow to invert it (§7.4), sanity-check the network
on simulated data it knows the truth for (§7.5), then finally point it at
VP10's real data (§7.6) and check the result makes sense (§7.7).

---

## 2. The big picture

```
                     ┌─────────────────────────────────────────┐
                     │   PRIOR: sample θ = (nu, r, mu_T)        │
                     │   uniformly from plausible ranges        │
                     └───────────────────┬───────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │   SIMULATOR (swift/simulator.py)         │
                     │   basic SWIFT: span → activation →       │
                     │   sine-saliency target + Gamma timer:    │
                     │   θ + corpus sentences → fixation seq.   │
                     └───────────────────┬───────────────────────┘
                                          │  (repeat ~8-14k times)
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │   TRAINING PAIRS  (θ, sequence, stats)   │
                     │   data/training_data.npz                 │
                     └───────────────────┬───────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │   BAYESFLOW (swift/inference.py)         │
                     │   summary net (LSTM) + coupling flow     │
                     │   learns:  sequence → p(θ | sequence)    │
                     └───────────────────┬───────────────────────┘
                                          │  (trained once, reused forever)
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │   TRAINED MODEL                          │
                     │   outputs/models/swift_approximator.keras│
                     └──────────┬───────────────────┬──────────┘
                                 │                   │
                 sanity check on │                   │ apply to REAL
                 simulated data  │                   │ VP10 data
                 (known truth)   ▼                   ▼
        ┌─────────────────────────────┐   ┌─────────────────────────────┐
        │  DIAGNOSTICS                 │   │  VP10 POSTERIOR              │
        │  recovery / SBC /            │   │  + posterior predictive      │
        │  contraction                 │   │  check (simulate from the    │
        │  (swift/diagnostics.py)      │   │  inferred θ, compare stats)  │
        └─────────────────────────────┘   └─────────────────────────────┘
```

The real VP10 data is touched **only** in the last box. Everything upstream
of it (simulator, training pairs, BayesFlow training, diagnostics) uses
*simulated* data with a *known* ground-truth θ, which is precisely what lets
us check the network is trustworthy before we ever apply it to something we
don't know the answer to.

---

## 3. Repo map — every file and why it exists

```
swift/                       the importable package — ALL pipeline code lives here
  __init__.py                 (empty, just makes `swift` importable)
  config.py                   paths + pipeline constants + the ONE shared
                               feature-normalisation function
  simulator.py                the SWIFT forward model (basic 3-parameter model)
  data.py                     load the two .dat files, EDA, real-data → network
                               input conversion
  generate.py                 parallel pre-generation of (θ, sequence) training pairs
  inference.py                BayesFlow v2 workflow: build it, train it, sample from it
  diagnostics.py               recovery / SBC / contraction / posterior / PPC plots

data/
  fixseqin_PB2expVP10.dat     REAL eye-tracking data, participant VP10 (osf.io/teyd4)
  Rcorpus_PB2_revision.dat    word properties for 114 German sentences (osf.io/nj2mf)
  training_data.npz           generated by generate.py — gitignored, regenerate locally

outputs/
  figures/                    every diagnostic PNG (tracked in git — these are the
                               plots that go in the report/slides)
    baseline_M10/               stale diagnostics from the superseded 4-parameter
                                 (full-SWIFT) model — kept only as historical
                                 reference, NOT comparable to the current 3-param run
  models/
    swift_approximator.keras  the trained network (gitignored — regenerate locally)
  results_summary.json        machine-readable snapshot written by
                               tools/show_results.py (tracked in git — small, human
                               readable, useful provenance for the report)

tools/
  calibrate.py                fast (~seconds) NumPy-only check: does the simulator's
                               *prior-averaged* behaviour resemble VP10, without
                               touching BayesFlow at all? Used while tuning the
                               FIXED constants in simulator.py.
  show_results.py             load the SAVED trained model and print a full text
                               report (recovery, SBC, VP10 posterior, PPC table) —
                               does not train anything, ~30 seconds. See §8.

main.py                       CLI entry point tying every stage together
                               (--mode generate | train | infer | online | all)
pyproject.toml / requirements.txt   dependencies (bayesflow==2.0.11 pinned — see §11)
README.md                     quick-start + headline results
docs/
  PROJECT_GUIDE.md            this file
  RESULTS.md                  the full results write-up
```

**Why one `swift/` package instead of separate top-level folders?** Earlier
versions of this project had `data/load_data.py`, `simulator/`, `inference/`,
`diagnostics/` as separate top-level directories. They were consolidated into
one importable package so that (a) every module can `from swift.config import
...` the same constants without path juggling, and (b) there's exactly one
place (`swift/config.py`) that defines things like the sequence length or
feature scaling — see §6 for why that mattered.

---

## 4. The SWIFT model's parameters

### 4.1 The three **free** parameters (what BayesFlow infers)

These are the unknowns. Everything upstream of inference (the prior, the
simulator, the training data) is built around exactly these three; they are
defined once in [`swift/simulator.py`](../swift/simulator.py) as
`PARAM_NAMES = ["nu", "r", "mu_T"]` and must stay in that order everywhere
(theta arrays are always positional, not dict-keyed, once inside the
pipeline). These are the paper's own labels (Engbert & Rabe 2024, Section 5) —
the lecturer explicitly asked not to reparametrise. Full equations in
[MODEL_SPEC.md](MODEL_SPEC.md).

| Symbol | Name | Prior range | What it physically controls |
|---|---|---|---|
| `nu`   | processing-span shape | 0–1 | How far processing spreads to neighbouring words. The span puts weight `nu` on the words immediately left/right of fixation and `nu²` two words ahead (Eq. 1–2). Larger `nu` → wider span → more parafoveal preview → more **skipping** and more leftward (regression) processing. Main driver of **skip / regression rate**. |
| `r`    | overall processing rate | 0–12 | How fast word activation builds up per second (Eq. 3, 6, 7). Higher `r` → words finish processing in a single fixation → fewer **refixations**; lower `r` → words need several looks. Main driver of **refixation rate / fixation count**. |
| `mu_T` | mean saccade-timer interval | 100–400 ms | The mean of the Gamma-distributed fixation-duration timer (Eq. 10–12). **E[fixation duration] = mu_T exactly.** This is the *only* lever on **fixation duration**, and it is completely decoupled from `nu`/`r` (which affect only *which* words are fixated, not for how long). |

Practical note from this project: all three parameters recover **strongly**
from VP10's data (recovery r ≈ 0.94 / 0.96 / 1.00 for `nu` / `r` / `mu_T`).
`mu_T` is essentially exact (it is the mean of a Gamma with known shape);
`nu`/`r` are noisier but clearly identified — and the temporal ⊥ spatial
decoupling the paper predicts holds (posterior corr `mu_T` vs `nu`,`r` ≈ 0).
See [RESULTS.md](RESULTS.md) for the full discussion.

**Emergent, not mechanistic.** Skipping, refixation and regression are **not**
separate parameters or mechanisms — they *emerge* from the single sine-saliency
target rule (Eq. 8–9). Adding an explicit skip/refixation/regression term would
double-count against that rule; don't.

### 4.2 The fixed constants (only three, all paper values)

The basic model has just three fixed constants, all set to Engbert & Rabe's
own paper values — there is nothing to hand-calibrate. Defined in the `FIXED`
dict in [`swift/simulator.py`](../swift/simulator.py):

| Symbol | Value | Meaning |
|---|---|---|
| `eta`   | `1e-3` | Baseline saliency floor — keeps the target-selection denominator > 0 so every word has a tiny non-zero chance of being fixated (Section 3). *(Not the same `eta` as the old model's word-length exponent — same letter, different quantity.)* |
| `alpha` | `9` | Gamma shape of the fixation-duration timer ⇒ duration CV = 1/√9 = **1/3** by construction (Section 2.4). |
| `beta`  | `0.6` | Word-frequency effect on a word's maximum activation `a_max = 1 − beta·q` (Section 5 recovery value; free only in the 5-parameter extended model). |

That's the whole list. The Gamma timer (shape `alpha=9`) directly gives the
tight, right-skewed fixation-duration distribution real data shows — no
Erlang/threshold accumulator needed. One structural consequence worth
reporting: the fixed CV = 1/3 = 0.333 **over-predicts** VP10's real duration
spread (CV = 0.246), since `alpha` is not free. See
[RESULTS.md §6](RESULTS.md) "Deviations from paper".

---

## 5. The two data files

Both live in `data/` and are loaded by [`swift/data.py`](../swift/data.py).
**Only the fixation file is "real data" in the inference sense** — the
corpus is just an input the simulator needs (word properties), not something
we do inference on.

### 5.1 `fixseqin_PB2expVP10.dat` — real eye-tracking data, participant VP10

Whitespace-separated, **no header row**, 10 columns, loaded with
`pd.read_csv(path, sep=r"\s+", header=None)`:

| Col | Field | Notes |
|---|---|---|
| 1 | `sentence_id` | 1–114 |
| 2 | `word_id` | word position within the sentence |
| 3 | `landing_position` | decimal character offset within the word (jittered — real oculomotor noise) |
| 4 | `fixation_duration` | milliseconds, roughly 80–400 ms |
| 5 | `word_length` | characters, of the *fixated* word |
| 6 | `fixation_type` | 1 = first fixation in the sentence, 2 = last, 0 = middle |
| 7 | `flag1` | always 0 in this file |
| 8 | `flag2` | always 0 in this file |
| 9 | `fixation_index` | sequential position within the sentence (use this to order fixations, not the row order) |
| 10 | `participant_id` | always 10 (this file is VP10 only) |

This is real observed behaviour and, by design, is only touched at the very
end of the pipeline (VP10 posterior inference + PPC) — never during
simulator development or BayesFlow training.

### 5.2 `Rcorpus_PB2_revision.dat` — word properties for 114 sentences

Tab-separated **with a header row**:
`"sentID"  "nw"  "wordID"  "length"  "freq"  "code"`, loaded with
`pd.read_csv(path, sep="\t")` and renamed to `sentence_id`, `word_id`,
`frequency`. This is an *input* to the simulator — for every sentence it
supplies each word's **frequency**, which sets that word's maximum activation
`a_max = 1 − beta·q` (higher-frequency words saturate sooner; §4.2's `beta`).
Note the basic 3-parameter model has **no spatial extent**, so word *length*
is loaded but not used by the simulator (`build_corpus_lists` returns only the
frequency list).

> **Gotcha if you open this file in a plain text editor:** the header line's
> last field is followed directly by the first data row on what looks like
> the same physical line (`"code"1  11  1  3  112.096683  9`). That's because
> the header line ends in a lone `\r` (old Mac-style line ending) rather than
> `\n`. `pandas.read_csv` handles this transparently — it's not a corrupted
> file, just an old-style line ending. Don't "fix" it by hand-editing the
> file.

> **Open question — the `code` column is currently unused.**
> `load_corpus` reads it but `build_corpus_lists` only extracts `length` and
> `frequency`; `code` never reaches the simulator. It takes exactly 4 values:
> `9` on 661/1003 words (the majority — "regular" words) and `0`, `1`, `2`
> each on exactly 114/1003 words — i.e. exactly **one word per sentence**
> carries each of those three codes. Given the file naming (`PB2`,
> "boundary paradigm") this most likely marks word positions from the
> original display-change/boundary-paradigm manipulation (e.g.
> pretarget/target/posttarget), which the continuous-reading simplified
> SWIFT model here doesn't simulate — so leaving it unused is probably fine.
> This has **not** been confirmed against the source paper or the OSF page
> (both attempts to check failed — the OSF page is a JS-rendered SPA that
> doesn't return content to automated fetches). Per this repo's own
> convention, flagging this rather than silently assuming it's safe to
> ignore — worth a quick check against Rabe et al. (2021) if it comes up.

---

## 6. Key engineering constants (`swift/config.py`)

This file exists because two earlier bugs both traced back to constants
being redefined in more than one place: `SEQ_LEN` was once hard-coded
differently in `main.py` vs. buried inside the training code, and the
simulator and the real-data loader each normalised fixation features
*differently* — a silent mismatch that quietly breaks amortized inference
(the network is trained on one distribution of inputs and then evaluated on
a subtly different one at inference time). Now every such constant lives in
exactly one place and is imported everywhere else.

| Constant | Value | Why it's set this way |
|---|---|---|
| `M_SENTENCES` | 14 | One training example = one simulated "reader" (fixed θ) reading **14** sentences, concatenated into one sequence. A single ~8-fixation sequence carries almost no information about `nu`/`r`; 14 sentences' worth does. This is the single most important modelling decision in the project — see §7.3. |
| `SEQ_LEN` | 150 | Max concatenated fixations per training example (14 sentences × ~8 fixations + buffer). Sequences are zero-padded/truncated to this length. |
| `N_FEATURES` | 2 | Per-fixation feature vector: `[word_id, duration_ms]` — the simplified model's observable `f_i = (fixated word, fixation duration)` (paper Section 4). No landing position or word length (the basic model has no spatial extent). |
| `N_STATS` | 7 | Length of the hand-crafted summary-statistics vector (§7.4) fed to the network as a direct condition. |
| `WORDID_SCALE, DURATION_SCALE` | 20, 1000 | Fixed (not per-sequence) divisors bringing each raw feature roughly into `[0, 1]` (`FEATURE_SCALES`). **Fixed**, not per-sequence-max, on purpose — a per-sequence normalisation would erase the *absolute* magnitude information (e.g. "this word was skipped" is only visible if word-id gaps survive normalisation). |

`normalise_sequence()` and `pad_sequence()` in this file are called by
**both** the simulator (`swift/simulator.py::run_one_reader`) and the
real-data loader (`swift/data.py::sentence_features`) — that's the fix for the
bug mentioned above: training and VP10 inference are now guaranteed to see
identically-scaled inputs.

---

## 7. The pipeline, stage by stage

Run in this order; each stage's output feeds the next. (This mirrors
`main.py`'s `step_*` functions.)

### 7.1 Load & explore (EDA)

`swift.data.load_fixations` / `load_corpus` / `run_eda`. Prints summary
statistics for VP10 (fixation count, duration distribution + CV, skip rate,
refixation rate, regression rate, fixations/sentence, saccade amplitude) and
saves `outputs/figures/eda_fixations.png`. These numbers become the
**benchmarks** the posterior predictive check (§7.7) compares against later —
nothing here touches the model.

### 7.2 The simulator — basic simplified SWIFT

`swift/simulator.py::simulate_sentence`. This is the forward model: given θ
and one sentence's word frequencies, produce a fixation sequence of
`(word_id, duration_ms)` pairs. Words sit at discrete positions `1..N` with
**no spatial extent** (no word length, no landing position, no Gillespie
algorithm). Each fixation cycle does three things (full equations in
[MODEL_SPEC.md](MODEL_SPEC.md)):

- **Processing span (Eq. 1–2)** — with the eye on word `k`, an asymmetric
  4-word window accrues processing: weight `sigma` on `k`, `sigma·nu` on
  `k−1` and `k+1`, `sigma·nu²` on `k+2` (note `k−2` is *not* processed but
  `k+2` is). `sigma = 1/(1 + 2nu + nu²)` normalises globally.
- **Activation (Eq. 3, 6, 7)** — each in-span word's activation grows by
  `r · λ_w · T_seconds`, clamped at `a_max = 1 − beta·q` (frequency-dependent).
  *(The duration `T` is converted to seconds here — see the unit note below.)*
- **Target selection (Eq. 8–9)** — a **sine saliency** `s_w = a_max·sin(π·a_w/a_max) + eta`
  peaks when a word is *half*-processed, so both untouched and finished words
  are unattractive. The next fixation is drawn ∝ `s_w`. Skipping, refixation
  and regression **all emerge from this one rule** — there is no explicit
  mechanism for any of them.

Fixation **durations** are independent Gamma draws (`shape α=9`, `mean μ_T`,
Eq. 10–12), completely decoupled from the processing above. That decoupling
is a deliberate feature of the paper's basic model (Section 4.1): `mu_T`
affects only durations, `nu`/`r` affect only *which* words are fixated.

> **Unit note (important):** `r` (paper values 5–10) is a rate *per second*,
> so the fixation duration `T` (stored in ms) is converted to seconds in the
> activation term. With `T` in ms, `r·λ·T` reaches the hundreds and every
> in-span word saturates in one fixation, collapsing the saliency rule and
> making the scanpath independent of `nu`/`r`. This is the one interpretive
> choice needed to reconcile the paper's ms wording with its `r` values — see
> [RESULTS.md §6](RESULTS.md).

**Test it standalone**: `python -m swift.simulator` runs a self-check that
asserts the span table matches the paper's Fig. 2 and the timer moments
(mean ≈ `mu_T`, CV ≈ 1/3), then prints an example fixation sequence.

### 7.3 Generate training pairs

`swift/generate.py::generate`. Samples θ from the prior thousands of times
and, for **each** draw, runs `run_one_reader` — which is the key modelling
choice mentioned in §6: instead of one sentence, it simulates the *same* θ
reading `M_SENTENCES=14` different randomly-chosen sentences and
concatenates their fixations into one sequence. This runs in parallel across
CPU cores (`multiprocessing`, `fork` context) since it's pure NumPy with no
GPU/ML involved, and saves the result to `data/training_data.npz`
(`thetas`, `seqs`, `stats` arrays). This is pure simulation — no BayesFlow,
no real data, at this stage.

Why concatenate sentences instead of training on single sentences? A lone
~8-fixation sequence barely constrains `nu` or `r` — there just isn't enough
evidence in it. Many sentences under the *same* θ give the network enough
repeated evidence (does this reader consistently skip words? consistently
refixate? regress?) to pin those parameters down. (Generation is fast —
~1700 readers/s — so raising `M_SENTENCES` further is cheap if needed.)

### 7.4 Train BayesFlow

`swift/inference.py`. This is where the actual neural network learns the
inverse mapping. Three pieces:

- **Adapter** (`build_adapter`) — routes data to the right role: `theta` →
  `inference_variables` (what we want the network to predict), the raw
  `fixations` sequence → `summary_variables` (compressed by the summary
  network below), and a **hand-crafted statistics vector** → 
  `inference_conditions`, fed to the network *directly*, unsummarised.
- **Summary network**: `bf.networks.TimeSeriesNetwork` (an LSTM,
  `summary_dim=64`) — reads the variable-length, padded fixation sequence
  and compresses it into a fixed-size vector.
- **Inference network**: `bf.networks.CouplingFlow` (a normalizing flow with
  spline transforms, 6 layers) — takes the summary vector *and* the
  hand-crafted statistics, and learns to transform a simple base
  distribution into `p(θ | data)`.

**Why hand-crafted statistics on top of the LSTM?** This was the single
biggest improvement in the project. `compute_reader_stats()`
(`swift/config.py`) computes 7 numbers per reader — mean/std duration, mean
fixations/sentence, **skip rate**, **refixation rate**, **regression rate**,
and mean saccade amplitude — and feeds them to the inference network
*directly*, bypassing the LSTM. An LSTM trained on the raw sequence alone
struggled to reliably extract the skip/refixation/regression signals that are
exactly what identify `nu` and `r`; handing the network those statistics
explicitly lifts them to strong recovery (r ≈ 0.94 / 0.96 — see
[RESULTS.md](RESULTS.md)). The regression rate in particular is the most
direct signal about `nu`, since `λ_{k−1} = sigma·nu` is the model's only
source of leftward processing.

`train_offline` (recommended) trains on the pre-generated `.npz` file;
`train_online` runs the simulator inside the training loop instead (much
slower, kept mainly as a fallback/comparison). Training is forced onto CPU
(`torch.backends.mps.is_available = lambda: False`) because Apple's MPS
backend doesn't implement the `linalg_qr` op the spline coupling flow needs
— for a model this small, CPU training takes minutes, not hours.

### 7.5 Diagnostics — is the network trustworthy?

`swift/diagnostics.py::run_builtin_diagnostics`. Before ever looking at
VP10, we check the trained network on **simulated** validation data where we
*know* the true θ (it was the input to the simulator). Four checks, each
producing a plot in `outputs/figures/`:

- **Recovery** — does the posterior mean track the true θ across many
  validation draws? (`recovery_plot.png`)
- **Simulation-Based Calibration (SBC)** — are the network's *uncertainty
  estimates* honest, not just its point estimates? (`sbc_histogram.png`,
  `sbc_ecdf.png`; reference: Talts et al. 2018)
- **Posterior contraction** — how much narrower is the posterior than the
  prior? (`contraction_plot.png`) High contraction = the data actually
  informed the estimate; contraction near 0 = the network learned almost
  nothing (posterior ≈ prior).

Two extra model-validation plots (not network diagnostics — checks that the
*simulator* matches the paper) are also produced here:
`span_shape.png` (the processing-span weights vs the paper's Fig. 2) and
`scanpath_examples.png` (example simulated fixation sequences, cf. Fig. 4).

See §9 for how to read each plot, and [RESULTS.md](RESULTS.md) for what
these checks concluded for the current trained model.

### 7.6 Real-data inference on VP10

`swift/inference.py::run_inference`, `swift/data.py::build_reader_batch`.
This is the **first and only point** real data is used. VP10's 114 sentences
are split in half (paper Section 6): inference uses the **first-half** ("train")
sentences, and the PPC (§7.7) is evaluated on the held-out **second-half**
("test") sentences via `swift/data.split_half`. Since one training example is
"one θ reading `M_SENTENCES` sentences," VP10 inference draws many random
14-sentence subsets of the train split, gets a posterior sample for each, and
**pools** them into one set of posterior draws. (An earlier, incorrect version
of this step averaged the raw fixation sequences together before inference —
statistically invalid, since averaging sequences isn't the same as combining
evidence about θ. Pooling posterior samples across many draws is the correct
way to combine evidence about the same underlying reader.)

A **decoupling check** (`plot_posterior_correlation` →
`posterior_correlation.png`) also runs here: it prints and plots the posterior
correlation matrix of `nu`/`r`/`mu_T`. The basic model predicts `mu_T` is
independent of the scanpath, so `mu_T` vs (`nu`, `r`) should be ≈ 0 — a
direct, quantitative confirmation the decoupling held on real data.

### 7.7 Posterior predictive check (PPC)

`swift/diagnostics.py::posterior_predictive_check`. Takes the pooled VP10
posterior samples, re-runs the simulator with them (on the held-out **test**
sentences, §7.6), and compares **six reading measures** of the simulated
output against VP10's real data:

- **SFD** (single-fixation duration), **GD** (gaze duration), **TT** (total
  time) — the three standard fixation-duration measures (Section 4);
- **P(skip)**, **P(refixation)**, **P(regression)** — the movement-pattern
  probabilities.

Crucially, this compares *statistics*, never raw sequences directly — two
fixation sequences can differ fixation-by-fixation while still reflecting the
same reading behaviour, so a raw-sequence comparison would be both noisy and
the wrong question. This is what `tools/show_results.py`'s console table (and
`ppc_plot.png`) show. Current result: durations match within ~6 ms and
skip/refixation within a couple of points; the model **over-produces
regressions** (~10% vs VP10's ~2%), an honest limitation reported in
[RESULTS.md §6](RESULTS.md).

---

## 8. How to run everything

```bash
# Environment: dedicated conda env, Apple Silicon, torch on CPU
conda create -n swift-sbi python=3.12
conda activate swift-sbi
pip install -r requirements.txt
export KERAS_BACKEND=torch   # or rely on main.py's os.environ.setdefault

# 1. Pre-generate simulations once (parallel, a few minutes)
python main.py --mode generate --n_readers 10000

# 2. Train on the saved data → model + diagnostics + VP10 inference + PPC
python main.py --mode train

# ...or if you already have a trained model and just want diagnostics again:
python main.py --mode infer

# everything at once:
python main.py --mode all --n_readers 10000

# just want to SEE the current results without retraining? (~30s, read-only)
python tools/show_results.py

# fast smoke test that everything still runs end-to-end (~5s)
python tools/show_results.py --quick

# fast NumPy-only sanity check of the simulator vs. VP10 (no BayesFlow, seconds)
python tools/calibrate.py
```

| Command | What it does | Roughly how long |
|---|---|---|
| `--mode generate` | Pre-generate simulations (parallel) → `data/training_data.npz` | Minutes (CPU-core dependent) |
| `--mode train` | Train on saved data → model + diagnostics + VP10 inference + PPC | Tens of minutes on CPU |
| `--mode infer` | Load saved model → diagnostics + VP10 inference + PPC (no retraining) | A few minutes |
| `--mode online` | Train with the simulator in the loop instead of pre-generated data | Slow — avoid unless comparing to offline training |
| `--mode all` | `generate` → `train` (which itself runs diagnostics + infer + PPC) | Generate + train time combined |
| `tools/show_results.py` | Load the saved model, print a full read-only text report | ~30 seconds |
| `tools/calibrate.py` | NumPy-only marginal check, no model needed | Seconds |

All of `--mode generate/train/infer/all` regenerate the plots in
`outputs/figures/` and overwrite `outputs/models/swift_approximator.keras`.
`tools/show_results.py` also refreshes `outputs/figures/ppc_plot.png` and
writes `outputs/results_summary.json`, but never touches the model weights —
it's purely a report over whatever model is currently saved.

---

## 9. How to read every plot

All in `outputs/figures/` (the `baseline_M10/` subfolder holds stale plots
from the superseded 4-parameter full-SWIFT model — historical only, not
comparable to the current run).

| Plot | What's on it | What "good" looks like |
|---|---|---|
| `span_shape.png` | The processing-span weights `{k−1, k, k+1, k+2}` vs word position, for a few `nu` values | Matches the paper's Fig. 2 — a model check on the simulator, not the network |
| `scanpath_examples.png` | A few example simulated fixation sequences (word position vs fixation index), cf. Fig. 4 | Plausible left-to-right reading with occasional skips/refixations/regressions |
| `eda_fixations.png` | Real VP10 duration / saccade-amplitude histograms + skip/refix/regression rates | Just context — no pass/fail, these are the benchmarks PPC compares against later |
| `training_loss.png` | BayesFlow training loss vs. epoch | Decreasing and flattening out (not still dropping steeply at the last epoch, not diverging) |
| `recovery_plot.png` | x-axis: true θ (simulated, known) · y-axis: posterior mean, one panel per parameter (`nu`, `r`, `mu_T`) | Points close to the diagonal = accurate recovery. `mu_T` sits almost exactly on it; `nu`/`r` are tighter clouds around it |
| `sbc_histogram.png` | Histogram of the true value's *rank* among posterior samples, per parameter | Roughly flat/uniform. A U-shape means the posterior is too narrow (overconfident); a hump in the middle means too wide (underconfident) |
| `sbc_ecdf.png` | Same idea as the histogram, but as an ECDF-minus-uniform difference, with confidence bands | The curve should stay inside the shaded band |
| `contraction_plot.png` | How much narrower each posterior is than the prior, per parameter | Higher = the network is using the data. Near-zero contraction means "posterior ≈ prior" — that parameter isn't identifiable from this kind of data |
| `posterior_VP10.png` | VP10's inferred posterior per parameter (orange) against the flat prior (gray) | A narrow, peaked orange distribution = an informative estimate |
| `posterior_correlation.png` | Heatmap of the VP10 posterior correlation matrix of `nu`/`r`/`mu_T` | `mu_T` vs (`nu`, `r`) cells ≈ 0 — confirms the temporal ⊥ spatial decoupling the basic model predicts |
| `ppc_plot.png` | Real VP10 (blue) vs. simulated-from-posterior (orange): three duration histograms (SFD/GD/TT) + a skip/refixation/regression probability bar chart | Substantial overlap on durations; the regression bar is the known miss (§7.7). Final "does the fitted model behave like the real reader" check |

---

## 10. Glossary

- **Fixation** — a pause of the eye on a word while reading (this project's basic unit of data).
- **Saccade** — the rapid jump of the eye between fixations.
- **Refixation** — a second (or later) fixation landing on a word that was just fixated, before moving to a new word.
- **Skipping** — a word that receives no fixation at all as the eye passes over it.
- **Parafoveal** — outside the direct point of fixation but still within the visual field (can be processed, just less precisely than the fixated word).
- **Likelihood** — the probability of observing some data given a parameter setting, `p(data | θ)`. SWIFT's likelihood is *intractable*: there's no formula for it, only a simulator that can draw samples from it.
- **Prior** — what we assume about plausible parameter values *before* seeing data (here: uniform ranges, see §4.1's table).
- **Posterior** — the updated belief about parameters *after* seeing data, `p(θ | data)` — what this whole pipeline exists to estimate.
- **Simulation-Based Inference (SBI)** — the family of methods (including BayesFlow) for doing Bayesian inference using only a simulator, when the likelihood can't be written down.
- **Amortized inference** — training a network once on many simulated examples so that it can produce a posterior for *any* new dataset near-instantly afterward, instead of redoing expensive inference (like MCMC) from scratch every time.
- **Summary network** — the part of BayesFlow (here, an LSTM/`TimeSeriesNetwork`) that compresses a variable-length raw sequence into a fixed-size vector.
- **Inference network / coupling flow** — a normalizing flow (here, `CouplingFlow` with spline transforms) that learns a flexible, invertible mapping from a simple base distribution to the posterior, conditioned on the summary vector.
- **Recovery** — how well the posterior mean (from simulated data with known truth) tracks the actual true parameter value across many validation simulations.
- **Simulation-Based Calibration (SBC)** — a diagnostic (Talts et al. 2018) checking that the posterior's *uncertainty* is honest: across many simulations, the true value should fall at a uniformly random rank within the posterior samples.
- **Posterior contraction** — `1 − posterior_variance / prior_variance`; how much a parameter's uncertainty shrank after seeing data. Near 1 = highly informative data; near 0 = the data told us almost nothing about that parameter.
- **Posterior Predictive Check (PPC)** — simulating new data from the *inferred* posterior and comparing summary statistics against the real data, as an end-to-end sanity check of the fitted model.
- **Processing span** — the asymmetric 4-word window `{k−1, k, k+1, k+2}` around the fixated word `k` that accrues activation each fixation, shaped by `nu` (paper Eq. 1–2). The source of parafoveal preview, skipping and regression.
- **Sine saliency** — the target-selection rule `s_w = a_max·sin(π·a_w/a_max) + eta` (Eq. 8–9): a word is most attractive when *half*-processed, so both untouched and finished words are skipped. Skipping/refixation/regression all emerge from it.
- **Decoupling (temporal ⊥ spatial)** — in the basic model, fixation *durations* (`mu_T`) are independent of *which* words are fixated (`nu`, `r`); the paper's Section 4.1 property, checked here via `posterior_correlation.png`.
- **Identifiability** — whether a parameter can, even in principle, be pinned down by the kind of data available. In this project all three parameters (`nu`, `r`, `mu_T`) turn out strongly identifiable once the hand-crafted reading-measure statistics are fed to the network.

---

## 11. Troubleshooting / FAQ

- **`ModuleNotFoundError: No module named 'bayesflow'` (or `swift`)** — make
  sure you're using the `swift-sbi` conda env
  (`/opt/anaconda3/envs/swift-sbi/bin/python`), and if running a script
  directly from `tools/` rather than via `main.py`, note that
  `tools/*.py` scripts insert the repo root onto `sys.path` themselves — run
  them as `python tools/show_results.py` from the repo root, not from
  inside `tools/`.
- **MPS / `linalg_qr` errors on Apple Silicon** — `swift/inference.py` forces
  the torch backend onto CPU on import specifically to avoid this (Apple's
  MPS backend doesn't implement the linear-algebra op the spline coupling
  flow needs). If you see MPS errors anyway, check that `swift.inference`
  was actually imported before any torch/bayesflow calls happen.
  `KERAS_BACKEND=torch` must also be set (both `main.py` and
  `tools/show_results.py` set it via `os.environ.setdefault` at the top of
  the file, before importing bayesflow).
- **BayesFlow API confusion** — this project targets **BayesFlow v2.x**
  (`bf.Approximator`, `bf.networks.CouplingFlow`, `bf.BasicWorkflow`, PyTorch
  backend), a full rewrite of the v1.x TensorFlow API
  (`InvertibleNetwork`, `Trainer`). Run
  `python -c "import bayesflow; print(bayesflow.__version__)"` before
  changing anything in `swift/inference.py` or `swift/diagnostics.py` — if
  it's not 2.x, stop and flag it rather than patching against the wrong API.
  The pinned version is `bayesflow==2.0.11`
  ([requirements.txt](../requirements.txt)).
- **`No training data at data/training_data.npz`** — run
  `python main.py --mode generate` first (or `--mode all`, which does
  generate+train together).
- **`No saved model at outputs/models/swift_approximator.keras`** — run
  `python main.py --mode train` (or `--mode all`) first;
  `tools/show_results.py` and `--mode infer` both need an already-trained
  model.
- **Numbers move slightly between runs of `tools/show_results.py`** —
  expected. VP10 inference pools posteriors over *randomly chosen*
  14-sentence subsets, and PPC re-simulates from *randomly drawn* posterior
  samples, so exact figures have a small amount of run-to-run noise. Pass
  `--seed` for a reproducible draw; see [RESULTS.md](RESULTS.md) for the
  scale of this variation.
- **Why is `data/training_data.npz` / `outputs/models/*.keras` missing after
  a fresh clone?** — both are gitignored (see [`.gitignore`](../.gitignore))
  because they're large, regenerable, machine-specific artifacts. Regenerate
  them locally with `--mode generate` / `--mode train`. Figures under
  `outputs/figures/` **are** tracked in git, since those go directly into
  the report/slides.

---

## 12. References

- Engbert, R., & Rabe, M. B. (2024). *A tutorial on Bayesian inference for
  dynamical modeling of eye-movement control during reading.* Journal of
  Mathematical Psychology, 119, 102843.
- Rabe, M. B., Chandra, J., Krügel, A., Seelig, S. A., Vasishth, S., &
  Engbert, R. (2021). *A Bayesian approach to dynamical modeling of
  eye-movement control in reading of normal, mirrored, and scrambled texts.*
  Psychological Review, 128(3), 516–543.
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018).
  *Validating Bayesian Inference Algorithms with Simulation-Based
  Calibration.* arXiv:1804.06788.
- BayesFlow v2 documentation: https://bayesflow.org/v2.0.11/
