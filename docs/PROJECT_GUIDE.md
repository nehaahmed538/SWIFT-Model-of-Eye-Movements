# Project Guide — SWIFT Model + BayesFlow, explained from zero

This document is for a teammate joining the project who has **not** read the
code yet. It explains, in order: what problem we're solving, every file in
the repo and why it exists, every parameter (free and fixed) and what it
physically means, the two data files, the full pipeline stage by stage, how
to run it, how to read every plot it produces, and a glossary of the SBI/
BayesFlow jargon used throughout. For the numeric results themselves (what
the trained model actually achieves), see **[RESULTS.md](RESULTS.md)**. For
a condensed quick-start, see the top-level **[README.md](../README.md)**.

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
                     │   PRIOR: sample θ = (t_sac, eta,         │
                     │   delta0, R) uniformly from plausible    │
                     │   ranges                                 │
                     └───────────────────┬───────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │   SIMULATOR (swift/simulator.py)         │
                     │   Gillespie stochastic process:          │
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
  simulator.py                the SWIFT forward model (Gillespie algorithm)
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
    baseline_M10/               the OLDER (M=10, LSTM-only, no stat conditions)
                                 diagnostics, kept for a before/after comparison
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

### 4.1 The four **free** parameters (what BayesFlow infers)

These are the unknowns. Everything upstream of inference (the prior, the
simulator, the training data) is built around exactly these four; they are
defined once in [`swift/simulator.py`](../swift/simulator.py) as
`PARAM_NAMES = ["t_sac", "eta", "delta0", "R"]` and must stay in that order
everywhere (theta arrays are always positional, not dict-keyed, once inside
the pipeline).

| Symbol | Name | Prior range | What it physically controls |
|---|---|---|---|
| `t_sac` | saccade-timer period | 150–350 ms | The base speed of the "saccade timer" — the internal clock that decides when the eye leaves the current word. Smaller `t_sac` → the timer fires sooner → shorter fixations overall. This is the single biggest lever on **fixation duration**. |
| `eta` | word-length exponent | 0.1–1.0 | How much a word's length slows down processing. In the model, the processing rate scales as `word_length^(-eta)`. `eta=0` → length doesn't matter; `eta=1` → long words are processed proportionally much more slowly. Drives **fixation count and refixation on long words**. |
| `delta0` | processing-span half-width | 4–15 characters | The width of the "attention window" around the fixated word (in characters). A wide window lets neighbouring words get pre-processed *before* the eye arrives (parafoveal preview), which is what lets short/predictable words be **skipped** entirely. This is the main driver of **skipping rate**. |
| `R` | refixation-rate factor | 0.1–0.9 | Directly scales the model's tendency to look at the *current* word again (instead of moving on) when it isn't fully processed yet. Drives **refixation rate** (how often two consecutive fixations land on the same word). |

Practical note from this project: `t_sac` and `eta` turn out to be **strongly**
identifiable from VP10's data (recovery r ≈ 0.99 / 0.88), while `delta0` and
`R` are only **moderately** identifiable (r ≈ 0.52 / 0.43) — see
[RESULTS.md](RESULTS.md) for the full discussion. This isn't a bug; it's an
honest finding about how much a single participant's fixation record can
constrain the model's "attention window" and "refixation" dynamics.

### 4.2 The fixed constants (calibrated, not inferred)

Everything else the simulator needs is held fixed at a value chosen so that
the simulator's *prior-averaged* behaviour (average over many random draws
of the 4 free parameters) resembles VP10's real marginal statistics — see
`tools/calibrate.py`. These are **not** fit to VP10 in the formal SBI sense
(no posterior over them); they're calibrated once, by hand, to keep the
simulator physically plausible. This is documented explicitly so nobody
mistakes them for inferred quantities in the report.

Defined in the `FIXED` dict in [`swift/simulator.py`](../swift/simulator.py):

| Symbol | Value | Meaning |
|---|---|---|
| `alpha` | 12.0 | Baseline processing difficulty (max "processing units" a word needs) |
| `beta` | 0.35 | Word-frequency sensitivity — how much rarer words raise difficulty |
| `h` | 0.65 | Foveal-release strength: the more a fixated word is processed, the faster the saccade timer fires |
| `gamma` | 3.0 | Sharpness of saccade-target selection (exponent on "how much processing is left" when choosing where to look next) |
| `kappa` | 0.30 | Forward-saccade distance decay — keeps most saccades short (step to the next word) rather than jumping far ahead |
| `rho` | 0.15 | Parafoveal attenuation — non-fixated words are processed, but less efficiently than the fixated one |
| `refix_gain` | 1.0 | Scales how strongly `R` translates into an actual refixation probability |
| `sigma` | 1.2 | Oculomotor landing-position noise (characters) — real saccades don't land on an exact intended letter |
| `omega` | 0.05 | Decay rate of a word's activation after it's been fully processed |

One more constant worth knowing: `SWIFTSimulator.TIMER_THRESHOLD = 14`. A
single exponential waiting time is too noisy/spread-out to match real
fixation-duration distributions (which are tight and right-skewed). Instead,
the saccade timer must accumulate **14** sub-events before it actually fires
a saccade. Summing 14 exponential waits gives an *Erlang(14)* distribution,
whose coefficient of variation is `1/√14 ≈ 0.27` — much closer to real data
than a single exponential (`CV = 1`) would be.

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
`frequency` (`length` stays as-is). This is an *input* to the simulator —
for every sentence it supplies each word's length (characters) and frequency
(used in the processing-difficulty equation, §4.2's `alpha`/`beta`).

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
| `M_SENTENCES` | 14 | One training example = one simulated "reader" (fixed θ) reading **14** sentences, concatenated into one sequence. A single ~8-fixation sequence carries almost no information about `eta`/`delta0`/`R`; 14 sentences' worth does. This is the single most important modelling decision in the project — see §7.3. |
| `SEQ_LEN` | 150 | Max concatenated fixations per training example (14 sentences × ~8 fixations + buffer). Sequences are zero-padded/truncated to this length. |
| `N_FEATURES` | 4 | Per-fixation feature vector: `[word_id, landing_position, duration_ms, word_length]`. |
| `N_STATS` | 7 | Length of the hand-crafted summary-statistics vector (§7.4) fed to the network as a direct condition. |
| `WORDID_SCALE, LANDING_SCALE, DURATION_SCALE, WORDLEN_SCALE` | 20, 10, 1000, 10 | Fixed (not per-sequence) divisors used to bring each raw feature roughly into `[0, 1]`. **Fixed**, not per-sequence-max, on purpose — a per-sequence normalisation would erase the *absolute* magnitude information (e.g. "this word was skipped" is only visible if word-id gaps survive normalisation). |

`normalise_sequence()` and `pad_sequence()` in this file are called by
**both** the simulator (`swift/simulator.py::run_one_reader`) and the
real-data loader (`swift/data.py::build_reader_observation`) — that's the
fix for the bug mentioned above: training and VP10 inference are now
guaranteed to see identically-scaled inputs.

---

## 7. The pipeline, stage by stage

Run in this order; each stage's output feeds the next. (This mirrors
`main.py`'s `step_*` functions.)

### 7.1 Load & explore (EDA)

`swift.data.load_fixations` / `load_corpus` / `run_eda`. Prints summary
statistics for VP10 (fixation count, duration distribution, skip rate,
refixation rate, fixations/sentence) and saves
`outputs/figures/eda_fixations.png`. These numbers become the **benchmarks**
the posterior predictive check (§7.7) compares against later — nothing here
touches the model.

### 7.2 The simulator — SWIFT + Gillespie algorithm

`swift/simulator.py::SWIFTSimulator.simulate_sentence`. This is the forward
model: given θ and one sentence's word lengths/frequencies, produce a
fixation sequence. Two coupled continuous-time random processes run side by
side:

- **Lexical activation** of every word in the sentence (including
  parafoveal/not-yet-fixated ones, attenuated by `rho`) — words build up
  "activation" toward a target determined by their difficulty (`alpha`,
  `beta`) and how close they are to the current fixation (`delta0`, `eta`).
- **The saccade timer** — accumulates ticks at a rate that rises as the
  *currently fixated* word gets more processed (`h`). Once it accumulates 14
  ticks (§4.2), a saccade fires.

At every step, the **Gillespie algorithm** (a standard method for simulating
continuous-time stochastic systems, e.g. from chemical-reaction modelling)
asks: "of all the possible next events (each word's activation ticking up,
each fully-processed word's activation decaying, or the saccade timer
ticking), which happens next, and how long do we wait?" It draws a waiting
time from an exponential distribution whose rate is the *sum* of all event
rates, then picks *which* event fired proportional to each event's share of
that total rate. This gives a realistically irregular, randomly-timed
sequence of fixations rather than a fixed-timestep approximation.

When a saccade fires, `_select_target` picks the next word: unprocessed
words are attractive (weight rises with remaining processing need, sharpened
by `gamma`), short forward saccades are preferred over long ones (`kappa`),
regressions are penalised, and the current word gets a refixation bonus
scaled by `R` and by how much of it is still unprocessed.

**Test it standalone**: `python -m swift.simulator` simulates one sentence
with a random prior draw and prints the resulting fixations.

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
~8-fixation sequence barely constrains `eta`, `delta0`, or `R` — there just
isn't enough evidence in it. Many sentences under the *same* θ give the
network enough repeated evidence (does this reader consistently skip short
words? consistently refixate long ones?) to pin those parameters down.

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
fixations/sentence, **skip rate**, **refixation rate**, mean saccade
amplitude, mean landing position — and feeds them to the inference network
*directly*, bypassing the LSTM. An LSTM trained on the raw sequence alone
struggled to reliably extract the skip-rate and refixation-rate signals that
are exactly what identify `delta0` and `R`; handing the network those
statistics explicitly roughly **doubled** their recovery (0.32→0.52 for
`delta0`, 0.26→0.43 for `R` — see [RESULTS.md](RESULTS.md)).

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

See §9 for how to read each plot, and [RESULTS.md](RESULTS.md) for what
these checks concluded for the current trained model.

### 7.6 Real-data inference on VP10

`swift/inference.py::run_inference`, `swift/data.py::build_reader_batch`.
This is the **first and only point** real data is used. Since one training
example is "one θ reading `M_SENTENCES` sentences," VP10 inference draws many
random 14-sentence subsets of VP10's own sentences, gets a posterior sample
for each, and **pools** them into one set of posterior draws. (An earlier,
incorrect version of this step averaged the raw fixation sequences together
before inference — that's statistically invalid, since averaging sequences
isn't the same as combining evidence about θ. Pooling posterior samples
across many draws is the correct way to combine multiple pieces of evidence
about the same underlying reader.)

### 7.7 Posterior predictive check (PPC)

`swift/diagnostics.py::posterior_predictive_check`. Takes the pooled VP10
posterior samples, re-runs the simulator with them, and compares
**summary statistics** of the simulated output against VP10's real data:
mean/std fixation duration, fixations/sentence, mean landing position,
mean saccade amplitude (how many words the eye jumps per saccade — the
classic "movement pattern" statistic), skip rate, and refixation rate.
Crucially, this compares *statistics*, never raw sequences directly — two
fixation sequences can differ fixation-by-fixation while still reflecting
the same underlying reading behaviour, so a raw-sequence comparison would be
both noisy and the wrong question. This is what `tools/show_results.py`'s
console table (and `ppc_plot.png`) show.

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

All in `outputs/figures/` (the `baseline_M10/` subfolder holds the same set
of plots from an earlier, weaker iteration of the model, kept for
before/after comparison — see [RESULTS.md](RESULTS.md)).

| Plot | What's on it | What "good" looks like |
|---|---|---|
| `eda_fixations.png` | Real VP10 duration / landing-position / fixations-per-sentence histograms | Just context — no pass/fail, these are the benchmarks PPC compares against later |
| `training_loss.png` | BayesFlow training loss vs. epoch | Decreasing and flattening out (not still dropping steeply at the last epoch, not diverging) |
| `recovery_plot.png` | x-axis: true θ (simulated, known) · y-axis: posterior mean, one panel per parameter | Points close to the diagonal line = accurate recovery. A flat cloud (no diagonal trend) = the network isn't learning that parameter from the data |
| `sbc_histogram.png` | Histogram of the true value's *rank* among posterior samples, per parameter | Roughly flat/uniform. A U-shape means the posterior is too narrow (overconfident); a hump in the middle means too wide (underconfident) |
| `sbc_ecdf.png` | Same idea as the histogram, but as an ECDF-minus-uniform difference, with confidence bands | The curve should stay inside the shaded band |
| `contraction_plot.png` | How much narrower each posterior is than the prior, per parameter | Higher = the network is using the data, not just reproducing the prior. Near-zero contraction means "the posterior looks like the prior" — i.e. that parameter isn't identifiable from this kind of data |
| `posterior_VP10.png` | VP10's actual inferred posterior per parameter (orange), against the flat prior (gray) | A narrow, peaked orange distribution = an informative estimate. A distribution that looks just like the flat gray prior = the data didn't tell us much about that parameter for this specific participant |
| `ppc_plot.png` | Real VP10 (blue) vs. simulated-from-posterior (orange) histograms for duration / landing / fixations-per-sentence / saccade amplitude, plus a skip/refixation rate bar chart | Substantial overlap between blue and orange. This is the final "does the fitted model actually behave like the real reader" check |

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
- **Gillespie algorithm** — an exact method for simulating continuous-time stochastic systems (originally from chemical kinetics) by repeatedly sampling "which event next, and when" from the combined rates of all possible next events.
- **Identifiability** — whether a parameter can, even in principle, be pinned down by the kind of data available. A parameter can be "weakly identifiable" not because of a coding bug, but because the data genuinely doesn't constrain it much (this project's honest finding for `delta0`/`R`).

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
