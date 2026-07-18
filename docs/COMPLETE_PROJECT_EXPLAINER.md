# The SWIFT Eye-Movement Model + BayesFlow Project — A Complete, Plain-Language Guide

**A document written for someone who has never seen this code before, explaining
what the project does, why it exists, what data it uses, exactly which commands
to type to run it, and exactly what every single function in the code does —
in order, in simple words.**

*This document is meant to be handed to an AI assistant (ChatGPT, Claude, etc.)
with the instruction: "turn this into a clean PDF." It is intentionally plain —
no colors, no fancy layout — just clear, numbered sections, tables, and
explanations written so that someone with very little programming or statistics
background can follow along.*

> **⚠️ 2026-07-17 model rewrite.** The project now implements the **basic
> 3-parameter** simplified SWIFT model of Engbert & Rabe (2024): parameters
> `nu`, `r`, `mu_T` (the paper's own labels). An earlier version implemented
> the *full* SWIFT (Gillespie algorithm, activation thresholds, word length,
> landing positions) under reparametrised names (`t_sac`, `eta`, `delta0`,
> `R`) — the lecturer flagged this and it was replaced. This document has been
> updated to the new model; if you spot any lingering old name outside a
> "this was removed" note, treat it as stale. The authoritative, concise
> sources are **[MODEL_SPEC.md](MODEL_SPEC.md)** (equations),
> **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** (file-by-file), and
> **[RESULTS.md](RESULTS.md)** (numbers) — this file is the long-form
> plain-language companion to those.

> **📊 2026-07-18 analysis expansion.** Parts 11–18 were added: the parameter
> relationships and which data columns drive which parameter (Part 11), four
> independent tests that the data genuinely contains information about the
> parameters (Part 12), a full overfitting analysis (Part 13), convergence
> (Part 14), comparison with the paper (Part 15), a worked proof that the
> regression misfit is a model limitation rather than an inference failure
> (Part 16), a precise plot-by-plot comparison guide (Part 17), and a
> question-and-answer bank (Part 18). The numbers in those parts were computed
> directly from `data/training_data.npz` and are reproducible; the commands
> used are given inline where relevant.

> **👥 2026-07-19 all-participants extension.** The pipeline was extended from
> one participant (VP10) to **all 34 participants** of the original experiment,
> using the *same* trained network with **no retraining** (the script is
> `tools/all_participants_ppc.py`; the data lives in `data/vp_all/`). A new
> **Part 19** explains that analysis and walks through its five new figures
> plot-by-plot, including every underlying concept; Parts 3, 4, 6, 9, 10.6 and
> 17 were updated to mention the new command, data folder, output files and
> plots. The former Parts 19–21 (Glossary, Troubleshooting, References) are
> now Parts 20–22.

---

## How to read this document

Whenever a technical word is used for the first time, this document follows one
rule: **first give the literal / plain meaning of the word, then explain what it
means in this project.** For example:

> **Fixation** (literal meaning: the act of holding something steady/still) — in
> eye-tracking research, a "fixation" is the short pause (usually 150–300
> milliseconds) during which your eye stops moving and rests on a word so your
> brain can read it.

You will see this pattern throughout the document. There is also a full
**Glossary** near the end (Part 20) that collects every technical term in one
place, in case you want to look something up without hunting through the whole
document.

The document is organized into 22 parts. Parts 1–10 explain *what the project
is and what it does*; Parts 11–19 are the deeper analysis — how the parameters
relate to each other, whether the data actually contains information about
them, whether the model overfits, how it compares to the original paper, a
question-and-answer bank you can revise from, and the extension of the whole
pipeline from one participant to all 34 participants of the experiment:

1. What is this project? (the goal, explained simply)
2. The big picture (how all the pieces fit together)
3. Commands to run — from a fresh checkout, in order
4. The datasets — every file, every column, explained
5. The SWIFT model — the science, explained simply
6. The project folder map
7. The code, function by function (the largest section)
8. What actually happens, command by command (tracing the function calls)
9. How to read every output plot
10. The final results this project achieved
11. **The parameters in depth** — what each controls, how they interact, and
    exactly which data columns drive which parameter
12. **Does the data actually contain information about the parameters?** —
    the evidence, with numbers
13. **Overfitting** — why it is an unusual question here, how we checked, and
    what we found
14. **Convergence and training health**
15. **Comparison with the original paper** — and what could vs. could not be
    improved
16. **The regression problem** — a worked analysis of the one thing the model
    gets wrong, and proof of *why*
17. **Every plot, precisely** — what exactly is being compared to what
18. **Question-and-answer bank** — likely questions, with answers
19. **The all-participants extension** — the same trained network applied to
    every one of the 34 readers in the experiment, with a plot-by-plot guide
    to the five new cross-participant figures
20. Glossary
21. Troubleshooting / FAQ
22. References

---

# PART 1 — What is this project? (The goal, in plain English)

## 1.1 The everyday phenomenon this project studies

When you read this sentence, it feels like your eyes glide smoothly from left
to right. They don't. If you were hooked up to an eye-tracking camera right
now, you would see that your eyes actually move in small, rapid jumps, pausing
briefly on some words, skipping over others entirely, and sometimes jumping
backward to re-read something that confused you.

Scientists have names for these behaviors:

- **Fixation** — a short pause (roughly 80–400 milliseconds) where the eye
  stops and looks at a word. This is when your brain actually extracts
  information from the page. A "millisecond" (ms) is one-thousandth of a
  second, so 200 ms is one-fifth of a second — faster than you can blink.
- **Saccade** (literal meaning: from the French word for "jerk" or "jolt") —
  the very fast jump the eye makes to move from one fixation to the next. Your
  eyes are essentially "blind" while jumping; all the useful information comes
  in during the pauses (the fixations), not during the jumps.
- **Skipping** — when a word receives *no* fixation at all — the eye simply
  passes over it (this usually happens with short, common, or highly
  predictable words like "the" or "and").
- **Refixation** — when the eye pauses on the *same* word two or more times in
  a row before moving on (this usually happens with long or difficult words).
- **Regression** — a backward jump, i.e. the eye jumps back to re-read
  something earlier in the sentence, usually because something didn't make
  sense the first time.

None of this is random. Easy, short, predictable words tend to get skipped or
glanced at briefly. Hard, rare, long words tend to get looked at longer and
more than once. In other words, **the pattern of eye movements reflects how
hard your brain is working to process each word.**

## 1.2 The scientific question

Researchers built a mathematical/computational model called the **SWIFT
model** (Engbert & Rabe, 2024, building on years of earlier work) that tries
to *explain* this behavior. You feed the SWIFT model a small number of
"dials" (parameters) — for example, "how wide is the reader's attention
window" or "how much does word length slow the reader down" — and the model
can then *simulate* a plausible sequence of fixations for a sentence, as if a
person with those particular dial-settings were reading it.

This project's actual scientific goal is the **reverse** of that: given a
*real* person's actual eye-movement recording (a participant labeled "VP10"
in this dataset), **figure out what dial-settings (parameters) best explain
how they, specifically, read.** This is called an **inverse problem**
(literal meaning: a problem where you know the *output* and want to recover
the *input* that produced it, which is the reverse — the "inverse" — of the
usual direction).

## 1.3 Why this is hard: there is no formula

For many statistical models, there is a mathematical formula that tells you
"given these parameter values, here is the exact probability of seeing this
particular data." That formula is called a **likelihood** (in plain terms:
"how likely is this observed data, if these parameter values were true?").
When you have a likelihood formula, there are many classical statistical
techniques (including a whole family called Bayesian statistics) for working
backward from data to the parameters that probably produced it.

**The SWIFT model has no such formula.** It is what's called a **stochastic
simulator** (literal meaning: "stochastic" = random/involving chance; a
"simulator" = a program that mimics a real process). You can *run* it forward
as many times as you like — feed in parameters, get out a simulated fixation
sequence — but there is no equation that tells you the exact probability of
any specific sequence. This is called having an **intractable likelihood**
(literal meaning of "intractable": "not able to be easily handled or solved"
— here it specifically means there's no formula, only a black-box simulator).

## 1.4 The solution: Simulation-Based Inference (SBI) and BayesFlow

This is exactly the kind of situation that a family of modern techniques
called **Simulation-Based Inference (SBI)** was built for. Instead of writing
down a likelihood formula, SBI methods do the following:

1. **Simulate many examples.** Draw many different candidate parameter
   settings at random (from a reasonable range called the **prior** — more on
   this below), and for each one, run the simulator to produce a fake/simulated
   fixation sequence. Do this thousands of times. Now you have thousands of
   pairs: (parameters, sequence-the-simulator-produced-from-those-parameters).
2. **Train a neural network to learn the reverse mapping.** A neural network
   (a type of machine-learning model, loosely inspired by how brain neurons
   connect, that learns patterns from many examples) is trained on those
   thousands of pairs to learn the *inverse* direction: given a sequence, guess
   which parameters probably produced it. The specific library used for this
   in this project is called **BayesFlow**.
3. **Reuse the trained network instantly, forever.** Because the network is
   trained once — on simulated data, which is cheap and unlimited to
   generate — it can then be pointed at *any* new sequence (including the real
   VP10 data) and produce an answer in milliseconds, without having to redo
   any expensive computation. This "train once, reuse forever, instantly" idea
   has a name: **amortized inference** (literal meaning of "amortize": to
   spread a cost out over time/many uses — here, the expensive training cost
   is paid once and "amortized" across every future use).

The output the network produces is not a single best-guess number for each
parameter — it produces a **posterior distribution**: a full range of
plausible values along with how likely each one is, honestly representing
uncertainty. (A **distribution**, in plain terms, is just "a description of
which values are more or less likely," e.g. "most likely around 250, but
could plausibly be anywhere from 230 to 280.")

## 1.5 What this project actually built, in one paragraph

This project (1) wrote a computer program that simulates the **basic
simplified SWIFT model** of reading — a compact set of processing-span,
activation, and target-selection rules (explained in Part 5), (2) used that
simulator to generate tens of thousands of example (parameters → simulated
fixation sequence) pairs, (3)
trained a BayesFlow neural network on those pairs to learn how to guess
parameters from a fixation sequence, (4) rigorously checked that the trained
network's guesses can be trusted (using several statistical sanity checks,
explained in Part 5 and Part 9), and finally (5) applied the trained network
to a real human being's actual eye-tracking data (participant "VP10") to
estimate, in milliseconds, the reading-related parameters that best describe
how that person reads — along with an honest statement of how confident we
can be in each estimate.

## 1.6 Course context

This is a graduate course project (Simulation-Based Inference course, M.Sc.
Data Science, TU Dortmund university), completed by a group of three
students. The official brief from the course instructor asked the group to:
implement the *simplified* version of the SWIFT model (not the full,
computationally heavier version) inside BayesFlow, use it to estimate
parameters related to gaze control and reading dynamics from real
eye-tracking data, and investigate how well the fitted model reproduces the
observed fixation durations and eye-movement patterns. Every one of those
requirements is addressed by the pipeline described in this document.

---

# PART 2 — The Big Picture

Before diving into individual files and functions, it helps enormously to see
the whole pipeline laid out as a sequence of stages. Here is the entire
project as one flow, described in words (this mirrors a diagram that also
appears in the project's own `docs/PROJECT_GUIDE.md`):

```
STAGE 1 — PRIOR
   Pick random values for the 3 unknown parameters (nu, r, mu_T)
   from a reasonable range. This happens many thousands of times.
        |
        v
STAGE 2 — SIMULATOR (a program that mimics a reader)
   For each set of parameters, run the simplified SWIFT simulator: it "reads"
   real sentences (using real word-frequency data) and produces a simulated
   sequence of fixations — the same kind of data a real eye-tracker records.
        |
        v   (this whole process is repeated thousands of times)
STAGE 3 — TRAINING PAIRS
   All of these (parameters, simulated sequence) pairs are saved to a single
   file on disk: data/training_data.npz
        |
        v
STAGE 4 — TRAIN THE NEURAL NETWORK (BayesFlow)
   A neural network is trained on all those pairs to learn: "given a
   sequence like this, what parameters probably produced it?" This is the
   expensive step (many minutes of computer time) but it only has to be
   done once.
        |
        v
STAGE 5 — TRAINED MODEL (saved to disk)
   outputs/models/swift_approximator.keras
        |
        +---------------------------------+
        |                                 |
        v                                 v
STAGE 6a — SANITY CHECKS               STAGE 6b — REAL-WORLD USE
   (uses simulated data with a            (uses the REAL VP10 data —
   KNOWN correct answer, so we can        the actual human being's
   check the network's guesses            eye-tracking recording)
   against ground truth)                        |
        |                                        v
        v                              STAGE 7 — VP10's ESTIMATED
   Do the guesses match the truth?      PARAMETERS (a posterior
   Are the network's confidence          distribution for each of the
   levels honest, not overconfident      3 parameters)
   or underconfident?                          |
                                                v
                                      STAGE 8 — DOES IT ACTUALLY
                                      EXPLAIN THE REAL DATA?
                                      Take the estimated parameters,
                                      run the SIMULATOR again with
                                      them, and check: does the
                                      simulated behavior look like
                                      VP10's real behavior?
```

**The single most important rule in this whole pipeline:** the real human
data (VP10's actual eye-tracking recording) is used in exactly one place —
Stage 6b/7/8. Every stage before that (the simulator, the training data, the
network training, the sanity checks) uses only *simulated* data where the
correct answer is already known. This is what lets the project verify the
network is trustworthy *before* ever trusting its answer about the real
person. It is the same logic as testing a medical test on patients whose
diagnosis is already known, before trusting the test on a new patient whose
diagnosis you don't know yet.

---

# PART 3 — Commands to Run (From a Fresh Checkout, in Order)

This section assumes the required Python packages are **already installed**
(this document intentionally does not cover the installation/setup step, as
requested — only the commands you run to actually execute the project). It
also assumes the two required data files are already sitting in the `data/`
folder (see Part 4 for exactly which files and where they come from).

## 3.1 One-time environment setup (before any command below)

The project needs a particular machine-learning "backend" (the underlying
engine that does the number-crunching) called **PyTorch**, and the code
expects an environment variable to tell the BayesFlow library to use it.
Every script in this project actually sets this automatically in its own
code (`os.environ.setdefault("KERAS_BACKEND", "torch")`), so this step is a
safety net rather than strictly required — but it is good practice to set it
explicitly in your terminal session before running anything:

```bash
export KERAS_BACKEND=torch
```

If you are on an Apple Silicon Mac (M1/M2/M3/M4 chip), be aware that this
project **deliberately forces all computation onto the CPU**, not Apple's
GPU-like "MPS" backend. This is not a performance choice — it's a
compatibility fix. Apple's MPS backend is missing a specific mathematical
operation (`linalg_qr`, part of linear algebra) that the neural network's
"spline coupling flow" component needs. The code disables MPS automatically
the moment `swift/inference.py` is imported. Because the SWIFT model network
used here is small, CPU training still only takes minutes, not hours.

## 3.2 Running the pipeline, stage by stage (recommended path)

This is the recommended order to run the project from scratch. Run these
from the project's root folder (the folder that directly contains `main.py`).

**Step 1 — Pre-generate the training simulations (do this once).**

```bash
python main.py --mode generate --n_readers 8000
```

What this does, in plain words: it runs the SWIFT simulator 8,000 times
(each time with a different random parameter setting), spreads that work
across all the CPU cores on your machine to go faster, and saves the results
to `data/training_data.npz`. This step typically takes a few minutes,
depending on how many CPU cores your machine has. You only need to do this
once — the results are saved to disk and reused by later steps.

> **Why 8,000?** This is the default (`--n_readers` defaults to 8000 in
> `main.py`) and, importantly, **it is the number actually used to produce
> every result in this document** — the saved `data/training_data.npz` contains
> exactly 8,000 readers. You can pass a larger number, and more data is
> generally mildly better, but Part 15.5 explains why increasing it would
> *not* fix this project's one remaining misfit. If you regenerate with a
> different count, the numbers in Parts 10, 12 and 16 will shift slightly.

> **⚠️ Windows users:** this command will fail with
> `ValueError: cannot find context for 'fork'`. See Part 21 (Troubleshooting)
> for why and what to do — the short version is that the saved
> `data/training_data.npz` already exists, so you can skip straight to Step 2.

**Step 2 — Train the neural network on the saved simulations.**

```bash
python main.py --mode train
```

What this does: loads the `data/training_data.npz` file created in Step 1,
trains the BayesFlow neural network on it, saves the trained network to
`outputs/models/swift_approximator.keras`, then automatically continues on to
run the diagnostic sanity checks, apply the network to the real VP10 data,
and run the final "does this match reality?" check — all described later in
this document. This is the slowest step, typically tens of minutes on a
laptop CPU.

**Step 3 (alternative to Steps 1–2 together) — Do absolutely everything in
one command.**

```bash
python main.py --mode all --n_readers 8000
```

This is exactly Step 1 followed by Step 2, combined into a single command.

**Step 4 (once you already have a trained model) — Re-run diagnostics and
VP10 inference without retraining.**

```bash
python main.py --mode infer
```

Use this if you already have a saved, trained model
(`outputs/models/swift_approximator.keras` already exists) and you just want
to re-run the sanity checks and the real-data analysis again — for example,
after changing something in the diagnostics code, without paying the training
cost again.

**Step 5 — Just look at the current results, without training or re-running
anything expensive.**

```bash
python tools/show_results.py
```

This is a fast (roughly 30 seconds), read-only command. It loads whichever
trained model is currently saved on disk and prints a full text report:
a summary of the real VP10 data, the parameter-recovery/calibration numbers,
the estimated VP10 parameters, and the final "does it match reality" table.
It does not retrain anything, and it also writes a machine-readable copy of
the report to `outputs/results_summary.json`.

A faster "does everything still basically work?" version of the same command:

```bash
python tools/show_results.py --quick
```

This uses much smaller sample sizes so it finishes in roughly 5 seconds — a
smoke test, not a real report.

**Optional — a fast, model-free sanity check of the simulator itself.**

```bash
python tools/calibrate.py
```

This does not use BayesFlow or any trained neural network at all. It simply
runs the raw SWIFT simulator a few hundred times with random parameters and
checks whether its *average* behavior (across many random parameter draws)
resembles the real VP10 data's *average* behavior. It runs in a few seconds
and is meant to be used while hand-tuning the model's fixed constants
(explained in Part 5), long before any neural network training is involved.

## 3.3 Other run modes that exist but are not the recommended path

```bash
python main.py --mode online
```

This trains the neural network by running the simulator *live, inside the
training loop*, instead of training on the pre-generated file from Step 1.
It produces (in principle) similar results but is much slower, since the
simulator has to run fresh every single time instead of being pre-computed
once. It exists mainly as a fallback/comparison option, not as part of the
normal workflow.

**Optional — apply the trained network to ALL 34 participants (Part 19).**

```bash
python tools/all_participants_ppc.py
```

This is the "all-participants extension" analysed in depth in Part 19. It
retrains **nothing**: it loads the already-saved network and repeats the
"estimate the parameters, then reality-check them" procedure once per
participant file in `data/vp_all/` (34 people, VP10 included), then draws the
five cross-participant figures and writes two machine-readable result files
(`outputs/all_participants_results.json` and
`outputs/cross_participant_correlations.json`). Because the reality-check
simulations are repeated 34 times, expect a total runtime in the tens of
minutes on a laptop CPU — the script deliberately uses smaller per-person
sample sizes than the VP10-only tools to keep this feasible (see Part 19.3).

## 3.4 Summary table of every command

| Command | What it does | Rough time |
|---|---|---|
| `python main.py --mode generate --n_readers 8000` | Pre-generate simulated training examples, save to disk | A few minutes |
| `python main.py --mode train` | Train the network on the saved simulations, then run diagnostics + VP10 analysis + final check | Tens of minutes |
| `python main.py --mode all --n_readers 8000` | Steps above combined (generate, then train) | Generate + train time combined |
| `python main.py --mode infer` | Load an already-trained network; skip training; just run diagnostics + VP10 analysis + final check | A few minutes |
| `python main.py --mode online` | Train with the simulator running live in the loop (slow, rarely used) | Slow |
| `python tools/show_results.py` | Read-only report of whatever model is currently saved — no training | ~30 seconds |
| `python tools/show_results.py --quick` | Same as above but tiny sample sizes, for a fast smoke test | ~5 seconds |
| `python tools/calibrate.py` | Fast, model-free check of the simulator's average behavior vs. real VP10 data | A few seconds |
| `python tools/analyse_information.py` | Model-free analysis: which statistic carries information about which parameter, and the regression/refixation trade-off (reproduces Parts 12 and 16) | A few seconds |
| `python tools/all_participants_ppc.py` | Apply the trained network to all 34 participants (no retraining); writes the five cross-participant figures and two JSON result files (Part 19) | Tens of minutes |

Every command in this table must be run from the project's root folder (the
folder containing `main.py`), because the code inside these scripts figures
out where the `data/` and `outputs/` folders are relative to that root.

---

# PART 4 — The Datasets: Every File and Every Column, Explained

This project's main pipeline uses exactly two input data files, both stored
in the `data/` folder. **Only one of them is "real data" in the sense that
matters for the scientific question** — the other is background information
the simulator needs, not something the project runs its statistical inference
on. (Two more entries live alongside them: a file the project *generates*,
described in 4.3, and — since the all-participants extension — a `data/vp_all/`
folder holding the same kind of recording for all 34 participants of the
experiment, described in 4.4.)

## 4.1 File 1 — `fixseqin_PB2expVP10.dat` — the real eye-tracking recording

This is the actual, real recording of one human being's eye movements while
reading. The participant is anonymized and labeled **"VP10"** (short for
"Versuchsperson 10," German for "test subject 10" — this dataset comes from a
German-language reading study, part of a body of work often referred to as
the "Potsdam Sentence Corpus" reading experiments). It comes from the public
research-data repository **OSF** ("Open Science Framework," a website
researchers use to publicly share their data so other scientists can verify
and build on their work) at the address `osf.io/teyd4`.

**Format:** plain text, with values separated by (one or more) spaces, and —
importantly — **no header row** (meaning the very first line of the file is
already data, not column names; the code has to supply the column names
itself, since the file doesn't state them). It is loaded in the code with:

```python
pd.read_csv(path, sep=r"\s+", header=None)
```

(`pd` here refers to the **pandas** library, the standard Python tool for
working with spreadsheet-like tabular data; a **DataFrame** is pandas' name
for a table with rows and named columns, similar to a spreadsheet.)

There are 877 individual fixations recorded across 114 different sentences
that VP10 read. Every row of the file is one single fixation — one pause of
the eye on one word. The file has exactly 10 columns:

| Column # | Field name in the code | What it means, in plain words |
|---|---|---|
| 1 | `sentence_id` | Which sentence this fixation belongs to. Ranges from 1 to 114 (there are 114 sentences total in the experiment). |
| 2 | `word_id` | The position of the fixated word within its sentence, counting from the start (e.g. word_id = 3 means "the third word of this sentence"). |
| 3 | `landing_position` | Exactly *where inside the word* the eye landed, measured in characters, and given as a decimal number (e.g. 2.7 means "roughly between the 2nd and 3rd letter of the word"). It's a decimal, not a whole number, because real eye-tracking cameras have some natural jitter/imprecision — the eye doesn't land on an exact letter boundary. |
| 4 | `fixation_duration` | How long the eye stayed on this word, measured in milliseconds. In this file the values run from **53 ms to 486 ms**, averaging **196.9 ms** with a standard deviation of 48.4 ms — so the large majority of fixations fall between roughly 150 and 250 ms, with a few unusually short and unusually long ones at the extremes. This is the only column that determines `mu_T`. |
| 5 | `word_length` | How many characters long the *fixated* word is. |
| 6 | `fixation_type` | A small code: `1` means this was the *first* fixation in the sentence, `2` means it was the *last* fixation in the sentence, and `0` means it was a fixation somewhere in the middle. |
| 7 | `flag1` | Always equal to 0 in this particular file. Its exact purpose in the original study design is not confirmed, and it is not used anywhere by this project's code. |
| 8 | `flag2` | Always equal to 0 in this particular file. Same situation as `flag1` — unused. |
| 9 | `fixation_index` | The running/sequential position of this fixation within its sentence (1st fixation of the sentence, 2nd fixation of the sentence, and so on). **This is the column you must use to put fixations in their correct time order — do not assume the rows of the file are already in the right order.** |
| 10 | `participant_id` | Always equal to 10 in this file, because this file only contains data from participant VP10. |

**Why this file is treated so carefully in the code:** because it is *real,
observed human behavior*, this file is only ever touched at the very last
stage of the whole pipeline — never while building or testing the simulator,
never while generating training data, never while training the neural
network. This mirrors good scientific practice: you don't want to
accidentally "peek" at your real test data while building and calibrating
your tools, because that risks fooling yourself into thinking your method
works better than it really does.

## 4.2 File 2 — `Rcorpus_PB2_revision.dat` — the word-properties corpus

This file is **not** a recording of anyone's behavior. It is reference
information about the 114 sentences themselves: for every word in every
sentence, how long is the word (in characters) and how frequently does that
word occur in everyday German (its "frequency"). The simulator uses this as an
*input*: the basic simplified SWIFT model's idea is that rarer words need more
processing before they count as "done," so the simulator reads each word's
**frequency** to set that word's maximum activation. (The basic model has no
spatial extent, so word *length* is loaded from this file but not actually
used by the simulator — see Part 5.) This file comes from OSF at
`osf.io/nj2mf`.

**Format:** plain text, separated by tab characters (an invisible character,
usually typed by pressing the Tab key, commonly used instead of spaces to
separate spreadsheet-like columns), and **with a header row** this time
(unlike File 1). It is loaded with:

```python
pd.read_csv(path, sep="\t")
```

The header row (the very first line, which names the columns) reads:
`"sentID"  "nw"  "wordID"  "length"  "freq"  "code"`. The code renames a few
of these to more descriptive names once loaded (`sentID` → `sentence_id`,
`wordID` → `word_id`, `freq` → `frequency`), while `length` and `code` keep
their original names.

| Column name in file | Renamed to (in code) | What it means |
|---|---|---|
| `sentID` | `sentence_id` | Which sentence this word belongs to (1–114, matching the same numbering as File 1). |
| `nw` | *(kept as `nw`, unused downstream)* | Total number of words in that sentence. |
| `wordID` | `word_id` | This word's position within its sentence (matches the same numbering used in File 1's `word_id` column, so the two files can be linked together / "joined"). |
| `length` | `length` | How many characters long this word is. |
| `freq` | `frequency` | How frequently this word occurs in everyday written German — a standard measure researchers use to estimate how easy or hard a word is to recognize. Rare words are harder to process; common words are easier. |
| `code` | `code` | A small numeric label. See the "open question" box below — this column is loaded by the code but never actually used by the simulator. |

**A file-format quirk worth knowing about (not a bug, don't "fix" it):** if
you ever open this file yourself in a plain text editor, the header line's
last entry (`"code"`) will appear to run directly into the first row of
actual data with no line break, looking like one long garbled line
(`"code"1  11  1  3  112.096683  9`). This is because the header line ends
with an old-style line-break character (a lone carriage return, `\r`, a
holdover from very old Macintosh computers) instead of the standard newline
character (`\n`) that most modern text editors expect. **Pandas (the Python
library used to load this file) handles this correctly and automatically** —
this is a cosmetic quirk of the raw file, not data corruption, and the file
should never be hand-edited to "fix" it.

> **Open question, honestly flagged (not silently guessed):** the `code`
> column is read from the file by the code, but it is never actually used
> anywhere afterward — the function that extracts the useful columns
> (`build_corpus_lists`, explained in Part 7) only pulls out `length` and
> `frequency`, and passes those on to the simulator. The `code` column has
> exactly 4 distinct values across the whole file: the value `9` appears on
> 661 of the 1003 total words (the large majority — likely marking "ordinary"
> words), while the values `0`, `1`, and `2` each appear on exactly 114 of the
> 1003 words — that is, **exactly one word per sentence** carries each of
> those three special codes. Given the file's name (it comes from a study
> referred to internally as "PB2," short for a "boundary paradigm" reading
> experiment — a type of experiment where a specific word in the sentence is
> secretly changed mid-eye-movement to study parafoveal processing), the most
> likely explanation is that this column marks which word was the special
> "pre-target/target/post-target" word in that original experiment design.
> That kind of manipulation isn't something this project's simplified,
> continuous-reading version of the SWIFT model simulates, so it is probably
> safe that the column goes unused — **but this has not been formally
> confirmed** against the original research paper, so it is documented here
> honestly as an assumption rather than a verified fact.

## 4.3 A file the project *generates itself* — `training_data.npz`

There is a third file that lives in the same `data/` folder, but unlike the
two files above, **you do not download this one — the code creates it for
you** by running the simulator thousands of times (this is "Step 1" of
Part 3). It is saved in NumPy's native `.npz` format (a compressed archive
format for saving multiple numeric arrays together — "NumPy" is the core
Python library for fast numerical arrays, and `.npz` is its way of bundling
several arrays into one file, similarly to how a `.zip` file bundles several
documents into one). It contains three arrays:

- `thetas` — the randomly-sampled parameter settings used for each simulated
  "reader" (see Part 5 for what these parameters mean).
- `seqs` — the simulated fixation sequences that resulted from each of those
  parameter settings.
- `stats` — a set of hand-calculated summary numbers (like skip rate and
  refixation rate) computed from each simulated sequence.

This file is deliberately excluded from the project's version control (it is
"gitignored" — meaning it is not tracked or shared through the code
repository) because it is large, easy to regenerate, and specific to
whichever machine generated it. If you clone this project fresh, you will not
find this file, and you need to run the `--mode generate` command (Part 3,
Step 1) to create it yourself.

## 4.4 The `data/vp_all/` folder — the same experiment, all 34 participants

Added on 2026-07-19 for the all-participants extension (Part 19). The original
experiment that produced VP10's recording had not one but **34** participants,
and this folder contains all of their fixation files —
`fixseqin_PB2expVP1.dat` through `fixseqin_PB2expVP34.dat` — downloaded from
the same public OSF repository that the Engbert & Rabe (2024) paper itself
links to (`osf.io/8wrf6`, folder
`R-Code-parameter-estimation-from-experimental-data/expdata/`; the source
experiment is Risse & Seelig, 2019). Each file has **exactly the same
10-column, no-header format** described in 4.1 — same 114 sentences, same
columns, just a different person — and the folder also carries
`Rcorpus_PB2.dat`, a copy of the word-properties corpus. The
`fixseqin_PB2expVP10.dat` inside this folder is the same recording as the
project's original VP10 file; VP10 is simply participant number 10 of the 34.

The files vary quite a bit in size: the shortest (VP15) has 616 fixations and
the longest (VP1) has 1,317; all 34 together contain 30,639 fixations. That
variation is itself meaningful — different people make different numbers of
fixations while reading the *same* 114 sentences, which is exactly the kind of
between-person difference the model's parameters are supposed to capture.

Like the original VP10 file, none of these files is ever touched while
building the simulator, generating training data, or training the network —
they are read by exactly one script, `tools/all_participants_ppc.py`, at
inference time only (the "use the trained tool" stage, never the "build the
tool" stage).

---

# PART 5 — The SWIFT Model: The Science, Explained Simply

## 5.1 What is being estimated, and why these three things specifically

The whole point of the project is to estimate three numbers — called
**parameters** (in plain terms: adjustable "dial settings" of the model) —
that describe how a specific person reads. These three are the **free
parameters**: the unknowns the neural network is asked to figure out (as
opposed to the "fixed" constants in 5.3, which are locked in place ahead of
time). They use the exact names from the Engbert & Rabe (2024) paper.

| Parameter (code name) | Plain-language name | Range considered plausible | What it physically means |
|---|---|---|---|
| `nu` | Processing-span shape | 0 to 1 | Controls how far the reader's processing spreads to *neighbouring* words. With the eye on a word, the reader also processes the word just before and just after it (weighted by `nu`) and one further ahead (weighted by `nu` squared). A **larger** `nu` means a wider processing window — more pre-processing of upcoming words (so more **skipping**) and more leftward processing (so more **regressions**, backward jumps). |
| `r` | Overall processing rate | 0 to 12 | Controls how *fast* words get processed while looked at. A **higher** `r` means words finish in a single fixation, so the reader rarely looks twice — few **refixations**. A **lower** `r` means words need several looks — more refixations and more fixations overall. |
| `mu_T` | Mean fixation duration | 100 to 400 milliseconds | The average length of a single fixation. In this model, the fixation duration is drawn from a bell-ish "Gamma" distribution whose mean is exactly `mu_T` — so **the average simulated fixation duration equals `mu_T` by construction.** This is the *only* dial that affects duration, and it is completely separate from `nu`/`r` (which affect only *which* words are looked at, not for how long). |

These three names, in this exact order, are stored in the code as a constant
list called `PARAM_NAMES = ["nu", "r", "mu_T"]`, defined once in
`swift/simulator.py` and imported everywhere else, so the order never gets
scrambled by accident.

A finding worth knowing up front (explained fully in Part 10): after running
this whole pipeline, **all three** parameters turned out to be *strongly*
pinned down by VP10's data (helped by feeding the network hand-crafted
reading-measure statistics, see Part 7). `mu_T` is essentially exact; `nu`
and `r` are noisier but clearly identified.

**Important — nothing is "hard-coded" for skipping/refixation/regression.**
These three reading behaviours are **not** separate dials. They *emerge*
naturally from the single target-selection rule (5.4). Adding an explicit
skip or refixation mechanism would double-count what the model already
produces, so the model deliberately has none.

## 5.2 What is a "prior"? (necessary background before the science below makes sense)

**Prior** (literal meaning: "coming before" — as in "prior to") — in
statistics, "the prior" refers to what you're willing to assume about
plausible parameter values *before* you've looked at any data. Here, the
prior for each of the three parameters above is simply "any value in the
listed range is equally likely, and nothing outside that range is
considered" — this specific, simple kind of prior is called a **uniform
prior** (every value within the range has exactly equal probability, like
rolling a fair die where every face is equally likely). The ranges are the
paper's own (Engbert & Rabe 2024, Section 5).

## 5.3 The fixed constants — the model's other "dials," but locked in place

Beyond the three free parameters, the simulator needs a few more numbers — but
these are **not** estimated by the network. In this simplified model there are
only **three**, and all are set to the paper's own published values (there is
nothing to hand-calibrate). Defined in the `FIXED` dictionary in
`swift/simulator.py`:

| Constant | Value used | What it means |
|---|---|---|
| `eta` | 0.001 | A tiny "baseline saliency" floor added to every word when the model decides where to look next — it keeps the maths well-defined (never divides by zero) and gives every word a minuscule non-zero chance of being fixated. *(Note: despite sharing a letter, this is a completely different quantity from the old model's `eta`; here it is a small constant, not a free parameter.)* |
| `alpha` | 9 | The "shape" of the fixation-duration distribution. Setting it to 9 makes the relative spread of durations (the **coefficient of variation** — how spread-out the durations are relative to their average) equal to 1/√9 = 1/3 ≈ 0.333, by construction. |
| `beta` | 0.6 | How much a word's *frequency* (how common it is) affects how much processing it needs. Common words reach their "done" level sooner. (This is free only in the paper's larger 5-parameter model; here it is fixed.) |

That is the entire list — no `h`, `gamma`, `kappa`, `rho`, `omega`,
`refix_gain`, or Gillespie timer threshold (those all belonged to the old
full-SWIFT simulator and are gone). One consequence of `alpha` being fixed at
9 is worth reporting: the simulated durations always have a spread of exactly
CV = 1/3, but VP10's real durations are *tighter* (CV ≈ 0.246), so the model
slightly **over-predicts** how variable fixation durations are. That is an
honest limitation of the simplified model, not a bug (see Part 10 and
[RESULTS.md](RESULTS.md)).

## 5.4 How the simulator actually works (no Gillespie algorithm anymore)

The simplified model is much simpler than the old full-SWIFT version — there
is **no Gillespie algorithm**, no continuous-time event simulation, no landing
positions, and no word length. Words simply sit at discrete positions 1, 2, 3,
… along the sentence. One simulated sentence is a loop of fixation cycles, and
each cycle does three things (full equations in
[MODEL_SPEC.md](MODEL_SPEC.md)):

1. **Spread processing over the span (paper Eq. 1–2).** With the eye on word
   `k`, the reader processes word `k` fully, its immediate neighbours `k−1`
   and `k+1` partially (weight `nu`), and word `k+2` a little (weight `nu`
   squared). Word `k−2` is *not* processed. This asymmetric little window is
   the "processing span," and its width is set by `nu`.
2. **Build up activation (paper Eq. 3, 6, 7).** Each word in the span gains a
   bit of "activation" (its degree of being processed) proportional to the
   processing rate `r` and how long the current fixation lasted. A word stops
   gaining once it reaches its personal maximum (lower for common/frequent
   words, via `beta`).
3. **Pick where to look next (paper Eq. 8–9).** Each word gets a "saliency"
   score using a **sine** curve that is highest when the word is *half*
   processed, and near zero both for untouched words and for fully-finished
   words. The next fixation is chosen at random, with probability proportional
   to these saliency scores.

That single sine-saliency rule is where **all** the interesting reading
behaviour comes from, with no extra machinery:
- a word that got pre-processed in the span (thanks to `nu`) may already be
  near "finished" by the time the eye would reach it, so it has low saliency
  and gets **skipped**;
- a word that is only half-processed has *high* saliency, so the eye may land
  on it again — a **refixation**;
- because the span also processes the word to the *left*, an earlier word can
  regain middling saliency and pull the eye backward — a **regression**.

**Fixation durations are handled completely separately.** Each fixation's
duration is just an independent random draw from a **Gamma distribution**
(literal: a standard family of bell-ish, right-skewed distributions for
positive quantities) with average `mu_T` and shape `alpha = 9`. It does *not*
depend on how processed the current word is. This deliberate separation —
durations on one side, the scanpath (which words) on the other — is a defining
feature of the paper's basic model (Section 4.1), and the project verifies it
held (the estimated `mu_T` is statistically uncorrelated with `nu` and `r`;
see Part 10).

> **One subtle but important detail (the "seconds" note).** The processing-rate
> step uses the fixation duration measured in **seconds**, not milliseconds.
> The paper writes durations in milliseconds throughout, but its `r` values
> only make sense if this one term uses seconds — otherwise every word would
> finish processing in a single fixation and the scanpath would stop depending
> on `nu` and `r` at all. This is the one place the implementation makes an
> interpretive choice to reconcile the paper's wording with its numbers; it is
> documented in [MODEL_SPEC.md](MODEL_SPEC.md) and [RESULTS.md](RESULTS.md).

## 5.5 A helpful mental model, tying it all together

Imagine each word in a sentence as a little bucket that needs to be filled up
with "processing units" before it counts as read. While you look at a word,
its bucket fills quickly; the buckets of the words just to its left and right
fill a little too (because you catch them out of the corner of your eye — how
much depends on the dial `nu`), and one word further ahead fills a tiny bit;
words further away don't fill at all yet. How fast any bucket fills is set by
the dial `r`. Common words have smaller buckets, so they finish sooner.

Now, the crucial part — **where do you look next?** You are drawn to buckets
that are *half* full: an empty bucket is unattractive (you haven't started it),
and a completely full one is also unattractive (you're done with it), but a
half-full bucket calls out to you. So you usually move forward to the next
not-yet-full word; sometimes you skip a word whose bucket you already
half-filled from the corner of your eye; sometimes you look again at a word you
only half-finished (a refixation); and occasionally an earlier word to the left
that got topped up pulls your eye backward (a regression). All of that comes
from the one "half-full is most attractive" rule.

Finally, **how long** does each pause last? That is decided by a completely
separate stopwatch that has nothing to do with the buckets: each fixation just
lasts a random amount of time averaging `mu_T` milliseconds. That entire
process, run purely by chance according to the rules above, is one simulated
sentence.

---

# PART 6 — The Project Folder Map

```
SWIFT-Model-of-Eye-Movements/
│
├── swift/                     <- the actual program logic lives here
│   ├── __init__.py               (contains only a one-line docstring; its job
│   │                              is simply to mark this folder as an
│   │                              "importable package" so other files can
│   │                              write `from swift.config import ...`)
│   ├── config.py                 shared settings: file paths, sequence
│   │                              length, and the ONE normalization recipe
│   │                              used everywhere
│   ├── simulator.py              the SWIFT model itself (the basic
│   │                              3-parameter simulator — the forward model)
│   ├── data.py                   loading the two .dat files, exploring
│   │                              them, and converting real data into the
│   │                              same format the simulator produces
│   ├── generate.py               runs the simulator thousands of times in
│   │                              parallel to build training data
│   ├── inference.py              the actual BayesFlow neural network setup:
│   │                              building it, training it, using it
│   └── diagnostics.py            all the sanity-check plots and the final
│                                  "does it match reality" check
│
├── data/                      <- data files
│   ├── fixseqin_PB2expVP10.dat   real recording (Part 4.1)
│   ├── Rcorpus_PB2_revision.dat  word-properties reference file (Part 4.2)
│   ├── training_data.npz         generated by generate.py (Part 4.3)
│   └── vp_all/                   all 34 participants' recordings, same
│                                  format as VP10's file (Part 4.4)
│
├── outputs/                   <- everything the program produces
│   ├── figures/                  every diagnostic image (kept in the
│   │                              project's version control, since these
│   │                              are the plots that go in the report)
│   │   └── baseline_M10/           an OLDER, earlier version of the plots,
│   │                                kept only for before/after comparison
│   ├── models/
│   │   └── swift_approximator.keras   the trained neural network (NOT kept
│   │                                    in version control — regenerate it
│   │                                    locally by training)
│   ├── results_summary.json      machine-readable snapshot written by
│   │                              tools/show_results.py
│   ├── all_participants_results.json        per-participant estimates and
│   │                              reality-check numbers (Part 19.9)
│   └── cross_participant_correlations.json  the six real-vs-simulated
│                                  correlations from Part 19.5
│
├── tools/                     <- small standalone helper scripts
│   ├── calibrate.py               fast, model-free simulator sanity check
│   ├── show_results.py            read-only report of the saved model
│   ├── analyse_information.py     model-free analysis of which statistic
│   │                               informs which parameter, plus the
│   │                               regression/refixation trade-off
│   │                               (reproduces Parts 12 and 16)
│   └── all_participants_ppc.py    apply the trained network to ALL 34
│                                   participants — no retraining (Part 19)
│
├── docs/                      <- written documentation
│   ├── PROJECT_GUIDE.md           the project's own from-scratch walkthrough
│   ├── RESULTS.md                 the project's own results write-up
│   └── COMPLETE_PROJECT_EXPLAINER.md   (this document)
│
├── main.py                    <- the command-line entry point that ties
│                                  every stage together
├── pyproject.toml / requirements.txt   dependency lists
└── README.md                  <- short project summary
```

**Why is (almost) all the code inside one single `swift/` folder, instead of
several separate top-level folders?** Earlier in the project's history, the
code was split across several separate top-level folders (`data/`,
`simulator/`, `inference/`, `diagnostics/` as siblings of each other). This
was consolidated into one single importable "package" named `swift/` for two
practical reasons: first, every file can now write a single consistent import
statement like `from swift.config import SEQ_LEN` no matter where it lives;
second — and more importantly — it guarantees there is **exactly one place**
in the whole project that defines things like "how long is a training
sequence" or "how are raw numbers rescaled before being fed to the neural
network." Earlier versions of the project had these values defined
*separately* in more than one file, and they occasionally drifted out of sync
with each other — a subtle bug that silently breaks the whole pipeline,
because the network ends up trained on data scaled one way and then tested on
data scaled a slightly different way, without any error message telling you
anything is wrong. Having one shared source of truth (`swift/config.py`) is
the fix.

---

# PART 7 — The Code, Function by Function

This is the heart of the document. Every function that matters in this
project is listed here, grouped by the file it lives in. For each function
you will find: the exact file it's in, what inputs it takes (its
"parameters," with their meaning), and a plain-language explanation of what
it actually does and why it exists.

A quick note on Python vocabulary used below: a **function** is a named,
reusable block of code that does one job — you "call" it by writing its name
followed by parentheses containing whatever inputs ("parameters" or
"arguments") it needs, and it hands back a result ("return value"). A
**class** is a way of bundling related data and functions together under one
name (here, `SWIFTSimulator` is a class — an object that, once created,
"remembers" its own parameter settings and has its own functions attached to
it for actually running a simulation).

## 7.1 `swift/config.py` — the shared settings file

This file defines constants (fixed values used throughout the project) and
three small, extremely important functions that are called from multiple
other files. Nothing in this file trains a model or runs a simulation — it's
purely the "rulebook" every other file agrees to follow.

**Key constants defined here** (already introduced conceptually, listed here
for completeness):

- `M_SENTENCES = 14` — how many sentences make up "one simulated reader." See
  7.3 for why this number matters so much.
- `SEQ_LEN = 150` — the maximum number of fixations kept per simulated
  reader. Sequences shorter than this are padded with zeros; longer ones
  (rare) get cut off.
- `N_FEATURES = 2` — every single fixation is represented as exactly 2
  numbers: `[word_id, duration_ms]`. (The simplified model's observable is
  "which word, and for how long" — there is no landing position or word
  length, since the model has no spatial extent.)
- `N_STATS = 7` — the length of a separate, hand-calculated summary-number
  vector (explained under `compute_reader_stats` below).
- `WORDID_SCALE = 20.0`, `DURATION_SCALE = 1000.0` (together `FEATURE_SCALES`)
  — fixed division factors used to bring each of the 2 raw numbers above
  roughly into a 0-to-1 range before they're fed to the neural network (see
  `normalise_sequence` below for why).

### `normalise_sequence(fixations)`

- **File:** `swift/config.py`
- **Parameters:** `fixations` — a raw array of shape "(number of fixations, 2
  numbers per fixation)," where the 2 numbers are, in order, `word_id` and
  `duration_ms`.
- **What it does:** divides each of the 2 numbers by a fixed constant (see the
  scale constants above), roughly squashing every value into a reasonable,
  comparable numeric range — for example, a raw duration of 200 ms becomes 0.2
  after dividing by `DURATION_SCALE = 1000`. Neural networks generally train
  much better and more stably when all their input numbers are on similar
  scales, rather than mixing small numbers (like word position, roughly 1–12)
  with large numbers (like duration in milliseconds, roughly 80–400) in the
  same input vector.
- **Why it matters so much:** this exact function — with these exact fixed
  divisors — is called by *both* the simulator (when producing training
  examples) and the real-data loader (when converting VP10's real recording
  into the same format). If these two code paths ever normalized numbers
  differently, the neural network would be trained on one distribution of
  input numbers and then tested on a subtly different one at "real data"
  time — a silent, hard-to-detect bug that would quietly make the whole
  project's results untrustworthy. Having one single function used
  everywhere is the guarantee against that.
- **A subtle but important design choice:** the divisors here are *fixed*
  numbers, not calculated fresh from each individual sequence (for example,
  "divide by the largest value in *this* sequence"). If the code instead
  rescaled every sequence relative to its own maximum, it would accidentally
  destroy real information — specifically, the fact that a particular word
  was skipped (a gap in the `word_id` numbers) would look different depending
  on how long the sentence was, purely because of the rescaling, even though
  nothing about the reading behavior actually changed. Fixed scales preserve
  that "absolute" information.

### `pad_sequence(fixations, seq_len=SEQ_LEN)`

- **File:** `swift/config.py`
- **Parameters:** `fixations` — an already-normalized array of fixations (any
  length); `seq_len` — the target fixed length to pad/cut to (defaults to
  150, the project-wide constant).
- **What it does:** creates a new array of exactly `seq_len` rows, all
  initially zero, then copies in as many of the real fixation rows as will
  fit (up to `seq_len` of them — if there happen to be more than `seq_len`
  real fixations, the extras beyond `seq_len` are simply dropped, though this
  essentially never happens in practice since real sequences are much
  shorter than 150). The result is always exactly the same fixed shape no
  matter how many fixations the original sequence actually had.
- **Why this is necessary:** neural networks generally need every training
  example to have the same "shape" (same array dimensions) to be processed
  efficiently in batches together. Since a real reader's sequence naturally
  varies in length (some readers/sentence-selections produce more fixations
  than others), the sequences are padded with rows of zeros at the end to
  reach a single common length. The neural network's summary component (an
  LSTM, explained in 7.5) is designed to handle this kind of padded,
  variable-effective-length input correctly.

### `compute_reader_stats(sentence_arrays)`

- **File:** `swift/config.py`
- **Parameters:** `sentence_arrays` — a list, where each item is the raw
  (un-normalized) fixation array for one sentence read by this "reader" (so
  for a normal training example, this list has `M_SENTENCES = 14` items in
  it, one per sentence).
- **What it does:** calculates **7 hand-crafted summary numbers** describing
  the overall pattern of this reader's behavior across all their sentences,
  and returns them as a single small array. Specifically, in order: (1) mean
  fixation duration in seconds, (2) the standard deviation of fixation
  duration in seconds (a measure of "how spread out" the durations are —
  literally, "standard deviation" measures the typical distance of
  individual values from the average), (3) the mean number of fixations per
  sentence (divided by 10, purely to keep the number small and comparable in
  scale to the others), (4) the fraction of words that were skipped
  (**skip rate**), (5) the fraction of consecutive fixation-pairs that landed
  on the exact same word (**refixation rate**), (6) the fraction of
  consecutive fixation-pairs that jumped *backward* to an earlier word
  (**regression rate**), and (7) the mean size of eye jumps measured in "how
  many words did the eye move" (divided by 5, again just for scale).
- **Why this function exists and matters enormously:** this is arguably the
  single most impactful design decision in the entire project. Feeding the
  raw fixation sequence into a neural network's summary component (an LSTM)
  and hoping it will, on its own, reliably notice subtle patterns like "this
  reader skips a lot" turned out not to work well enough in practice — an LSTM
  trained on the raw sequence alone struggled to reliably extract exactly the
  skip / refixation / regression signals that are needed to pin down the `nu`
  and `r` parameters. By instead directly hand-calculating those specific
  summary numbers and feeding them to the network as an extra, separate input
  (bypassing the LSTM entirely for these numbers), all three parameters reach
  strong recovery. The regression rate in particular is the most direct signal
  about `nu`, since leftward (backward) processing is the model's only source
  of regressions. This same function is called both when generating simulated
  training data and when converting VP10's real data — again, for the same
  reason as `normalise_sequence` above: training and real-world use must see
  identically-computed numbers.

## 7.2 `swift/data.py` — loading and exploring the two data files

### `load_fixations(path)`

- **File:** `swift/data.py`
- **Parameters:** `path` — the file location of `fixseqin_PB2expVP10.dat`.
- **What it does:** reads the real VP10 fixation file (whitespace-separated,
  no header row — see Part 4.1), assigns it the 10 column names listed
  earlier, prints a short one-line summary (total number of fixations and
  number of distinct sentences), and returns the resulting table (a pandas
  DataFrame).

### `load_corpus(path)`

- **File:** `swift/data.py`
- **Parameters:** `path` — the file location of `Rcorpus_PB2_revision.dat`.
- **What it does:** reads the tab-separated corpus file (which does have a
  header row — see Part 4.2), renames a few of its columns to more
  descriptive names (`sentID`→`sentence_id`, `wordID`→`word_id`,
  `freq`→`frequency`), prints a short summary, and returns the resulting
  table.

### `build_corpus_lists(corpus)`

- **File:** `swift/data.py`
- **Parameters:** `corpus` — the table returned by `load_corpus`.
- **What it does:** reorganizes the corpus table (which lists every word of
  every sentence as one big flat table) into a single Python list —
  `word_freqs_list` — where each item corresponds to one whole sentence and
  holds that sentence's word frequencies in reading order. This is the exact
  format the simulator expects whenever it's asked to simulate a sentence
  being read (the sentence's length is simply how many entries the array has).
  Internally this works by grouping all rows by `sentence_id`, then sorting
  each group by `word_id` before extracting the `frequency` numbers. Word
  *lengths* are dropped — the basic model has no spatial extent, so it doesn't
  need them.

### `sentence_features(fix, sentence_id)`

- **File:** `swift/data.py`
- **Parameters:** `fix` — the real VP10 fixation table; `sentence_id` — which
  single sentence's fixations to extract.
- **What it does:** filters the big VP10 table down to just the rows
  belonging to one particular sentence, sorts them into the correct time
  order (using the `fixation_index` column — the "sequential position"
  column described in Part 4.1, **not** simply the order rows happen to
  appear in the file), and returns just the 2 relevant numeric columns
  (`word_id`, `fixation_duration`) as a raw array — the same shape/format the
  simulator itself produces for a simulated sentence.

### `build_reader_observation(fix, sentence_ids, seq_len=SEQ_LEN)`

- **File:** `swift/data.py`
- **Parameters:** `fix` — the real VP10 fixation table; `sentence_ids` — a
  list of specific sentence numbers to combine together (normally 14 of
  them, matching `M_SENTENCES`); `seq_len` — the fixed output length to pad
  to.
- **What it does:** this is the real-data counterpart to the simulator's
  `run_one_reader` function (described in 7.3) — it builds one complete
  "observation" (one thing the trained network can be shown) out of *real*
  data instead of simulated data. It calls `sentence_features` once per
  requested sentence, glues all those sentences' fixations together into one
  long sequence, then applies the exact same `normalise_sequence` and
  `pad_sequence` steps from `swift/config.py` used for simulated data, and
  also computes the same 7 hand-crafted summary statistics via
  `compute_reader_stats`. It returns a `(sequence, stats)` pair — exactly the
  same shape of thing the trained network was trained to accept.

### `build_reader_batch(fix, m_sentences=M_SENTENCES, n_readers=200, seq_len=SEQ_LEN, rng=None)`

- **File:** `swift/data.py`
- **Parameters:** `fix` — the real VP10 fixation table; `m_sentences` — how
  many sentences to combine per observation (14 by default); `n_readers` —
  how many *different random combinations* of sentences to build; `seq_len`
  — the fixed output length; `rng` — a random-number generator (an object
  that produces a controlled, reproducible sequence of "random" numbers —
  passing the same `rng` seed guarantees you get exactly the same "random"
  choices again later, which is important for being able to reproduce
  results).
- **What it does:** builds `n_readers` separate observations out of VP10's
  real data, where *each one* is a different randomly-chosen subset of 14 of
  VP10's 114 sentences (VP10 only actually read 114 sentences total, but
  since there are 114-choose-14 possible different 14-sentence subsets — an
  astronomically large number — this effectively lets the project generate
  many different "views" of the same underlying reader). Why do this instead
  of just using all 114 sentences at once? Because the neural network was
  specifically trained to expect inputs shaped like "one reader, 14
  sentences" — it was never trained to accept "one reader, 114 sentences" as
  a single input. Building many 14-sentence random draws and later pooling
  (combining) the resulting estimates together (explained under
  `run_inference` in 7.5) is the statistically correct way to squeeze more
  reliable information out of all 114 of VP10's sentences, using a network
  that was only ever trained on 14-sentence inputs.

### `run_eda(fix, save_dir=FIG_DIR)`

- **File:** `swift/data.py`
- **Parameters:** `fix` — the real VP10 fixation table; `save_dir` — where to
  save the resulting plot image (defaults to the project's standard figures
  folder).
- **What it does:** "EDA" stands for **Exploratory Data Analysis** (literal
  meaning: looking at your data early on, before doing any modeling, just to
  understand what it broadly looks like). This function prints a handful of
  headline numbers about VP10's real data straight to the terminal (total
  fixation count, number of sentences, mean and standard deviation of
  fixation duration, the duration coefficient of variation, skip rate,
  refixation rate, regression rate, mean fixations per sentence), and produces
  and saves an image (`eda_fixations.png`) with **three histograms**
  (bar-chart-like plots showing how often each range of values occurs):
  fixation duration, *signed* saccade amplitude (negative values = backward
  jumps, so regressions are visible as the left-hand tail), and fixations per
  sentence. The skip/refixation/regression rates are **printed to the
  terminal**, not drawn on this figure. These
  numbers and plots become the **benchmark values** that the later "does it
  match reality" check (Part 7.6,
  `posterior_predictive_check`) compares the fitted model's simulated
  behavior against. Nothing here touches the neural network at all — this is
  pure, upfront data exploration.

### `skip_rate(fix)`

- **File:** `swift/data.py`
- **Parameters:** `fix` — a fixation table (works for either real or
  simulated data).
- **What it does:** for every sentence, it looks at the highest word position
  that was ever fixated, then checks — out of every word position from 1 up
  to that maximum — how many of them were *never* fixated at all (i.e.,
  skipped over entirely). It adds up the skipped-word count and the
  total-possible-word count across all sentences, then returns the overall
  fraction (skipped words ÷ total words considered) as the **skip rate**.

### `refix_rate(fix)`

- **File:** `swift/data.py`
- **Parameters:** `fix` — a fixation table.
- **What it does:** for every sentence, it puts the fixations in their
  correct time order and checks each consecutive pair: did the eye land on
  the exact same word twice in a row? It counts how often that happens out of
  every possible consecutive pair, across all sentences, and returns the
  overall fraction as the **refixation rate**.

### `regression_rate(fix)`

- **File:** `swift/data.py`
- **Parameters:** `fix` — a fixation table.
- **What it does:** for every sentence, it puts the fixations in time order and
  counts how many consecutive pairs moved *backward* (the `word_id` decreased),
  as a fraction of all **interword** saccades. The "interword" detail matters:
  the denominator counts only saccades where the word actually changed
  (`d != 0`), so refixations (staying on the same word) are excluded from the
  denominator rather than counted as non-regressions. This is the statistic
  that turns out to be the project's single biggest scientific finding — see
  Part 16.
- **Why it exists:** in the model, `lambda_{k-1} = sigma·nu` is the *only*
  source of leftward (backward) activation, so the regression rate is the most
  mechanistically direct observable consequence of `nu`. (Part 12 shows that
  *empirically* the skip rate is an even stronger predictor of `nu`, because
  regressions are also heavily influenced by `r` — an important nuance.)

### `saccade_amplitude(fix, signed=False)`

- **File:** `swift/data.py`
- **Parameters:** `fix` — a fixation table; `signed` — if `True`, keep the
  direction of each jump (positive = forward, negative = backward); if `False`
  (the default), return only the size of each jump.
- **What it does:** for every sentence, it measures — between each
  consecutive pair of fixations — how many word-positions the eye moved (the
  difference between consecutive `word_id` values). It pools these
  measurements across all sentences and returns them as one long list of
  numbers. This is the classic "movement pattern" statistic: for most real
  readers, this list is dominated by small values (mostly 1 — the eye usually
  steps to the very next word), with occasional larger values from skips
  (jumping 2+ words forward) or regressions (backward jumps). `run_eda` calls
  it with `signed=True` so the resulting histogram shows regressions as a
  visible left-hand tail; `compute_reader_stats` uses the unsigned mean.

### `split_half(fix)`

- **File:** `swift/data.py`
- **Parameters:** `fix` — the real VP10 fixation table.
- **What it does:** sorts VP10's 114 sentence numbers and cuts them straight
  down the middle, returning `(first 57 sentence ids, last 57 sentence ids)`.
- **Why it matters enormously — this is the project's train/test split:** the
  first half is the **train split**, used to *estimate* VP10's parameters; the
  second half is the **test split**, used only for the final posterior
  predictive check. This means the "does the model explain reality?" check in
  Part 10.4 is performed on sentences that played **no part whatsoever** in
  producing the parameter estimate. Without this split, the PPC would be
  marking its own homework — the model would be asked to reproduce the very
  data that was used to tune it, which would flatter it. This mirrors the
  paper's own Section 6 procedure. See Part 13 for why this matters for the
  overfitting question.

### `synthetic_corpus(n_sentences=114)`

- **File:** `swift/data.py`
- **Parameters:** `n_sentences` — how many fake sentences to generate
  (defaults to 114, matching the real corpus size).
- **What it does:** generates a completely made-up, randomized corpus table
  (random sentence lengths between 8 and 14 words, random word lengths
  between 3 and 12 characters, random frequencies between 10 and 500) using a
  fixed random seed (so it's reproducible). This exists purely as a **safety
  fallback**: if someone runs the project's code without the real corpus file
  in place, the pipeline can still run end-to-end (for testing purposes)
  using this fake stand-in data instead of crashing outright. It is never
  used when the real corpus file is present.

## 7.3 `swift/simulator.py` — the SWIFT model itself

### `sample_prior(rng=None)`

- **File:** `swift/simulator.py`
- **Parameters:** `rng` — an optional random-number generator; if none is
  given, a fresh unseeded one is created.
- **What it does:** draws one random value for each of the 3 free parameters
  (`nu`, `r`, `mu_T`), each drawn uniformly from its own allowed range (see the
  table in Part 5.1), and returns them as a Python dictionary (a name→value
  mapping) such as `{"nu": 0.42, "r": 6.1, "mu_T": 205.0}`. This represents
  "one imaginary reader's dial settings," drawn from the prior.

### `normalise_theta(params)`

- **File:** `swift/simulator.py`
- **Parameters:** `params` — a parameter dictionary like the one produced by
  `sample_prior`.
- **What it does:** converts the dictionary into a plain numeric array (in
  the fixed order given by `PARAM_NAMES`), and rescales every value into the
  0-to-1 range based on each parameter's own minimum and maximum bound
  (e.g., a `mu_T` of 250, given the range 100–400, becomes (250−100)/(400−100)
  = 0.5). Neural networks generally train better when every one of their
  target values is on a similar, small numeric scale, rather than mixing a
  parameter that ranges 100–400 with one that ranges 0–1 in the same training
  target.

### `denormalise_theta(theta_norm)`

- **File:** `swift/simulator.py`
- **Parameters:** `theta_norm` — a numeric array with values in the 0-to-1
  range (as produced by `normalise_theta`, or as guessed by the trained
  network).
- **What it does:** the exact reverse of `normalise_theta` — converts
  0-to-1-scaled numbers back into their real, physically-meaningful units
  (milliseconds for `mu_T`, and so on) and returns them as a parameter
  dictionary again. This is used whenever the neural network's raw 0-to-1
  output needs to be turned back into human-readable numbers like "198 ms."

### `span_rates(k, N, nu)`

- **File:** `swift/simulator.py`
- **Parameters:** `k` — the position (1-based) of the word currently being
  looked at; `N` — the number of words in the sentence; `nu` — the
  processing-span free parameter.
- **What it does:** returns an array giving, for every word in the sentence,
  how much of the reader's processing it receives on this fixation (the
  "processing span," paper Eq. 1–2). The fixated word `k` gets full weight
  `sigma`; its immediate neighbours `k−1` and `k+1` get `sigma·nu`; the word
  two ahead, `k+2`, gets `sigma·nu²`; every other word gets 0. (Note the
  asymmetry: `k+2` is processed but `k−2` is not.) The factor
  `sigma = 1/(1 + 2·nu + nu²)` keeps the weights on a consistent global scale.
  This is verified against the paper's Figure 2 by the built-in self-check.

### `simulate_sentence(word_freqs, nu, r, mu_T, beta=..., eta=..., alpha=..., rng=None, max_fix=...)`

- **File:** `swift/simulator.py`
- **Parameters:** `word_freqs` — the array of word frequencies for this one
  sentence (its length is the number of words); `nu`, `r`, `mu_T` — the three
  free parameters; `beta`, `eta`, `alpha` — the three fixed constants
  (default to their `FIXED` values); `rng` — a random-number generator;
  `max_fix` — a safety cap on the number of fixations (default 200) so the
  loop can never run forever.
- **What it does, step by step (the forward model of Part 5.4):** it sets each
  word's maximum activation `a_max = 1 − beta·q` from its frequency, and starts
  the eye on word 1. Then, in a loop, for each fixation it: (1) draws a
  duration `T` from a Gamma distribution with mean `mu_T` and shape `alpha`;
  (2) spreads processing over the current span via `span_rates`, adding
  `r · λ_w · (T/1000)` to each in-span word's activation (the `/1000` converts
  the duration to **seconds** — the crucial unit detail from Part 5.4),
  clamped at each word's `a_max`; (3) records the fixation as a
  `(word_id, duration_ms)` pair; (4) computes each word's sine saliency
  `s_w = a_max·sin(π·a_w/a_max) + eta` and randomly picks the next word to
  look at, with probability proportional to `s_w`. The loop stops once the last
  word has been fixated (or the safety cap is hit — a global counter
  `TRUNCATIONS` records how often that ever happens). The output is a Python
  list of `(word_id, duration_ms)` pairs.

### `class SWIFTSimulator` — a thin convenience wrapper

- **File:** `swift/simulator.py`
- **What it is:** a small class that just remembers one parameter dictionary,
  so callers who prefer an object can write `SWIFTSimulator(params)` once and
  then call `.simulate_sentence(word_freqs, rng)` repeatedly without re-passing
  the parameters. It is a thin wrapper around the module-level
  `simulate_sentence` function above — there is no separate simulation logic,
  no Gillespie loop, no target-selection helper (those all belonged to the old
  full-SWIFT simulator and are gone).

### `simulate_one_sentence_features(params, word_freqs, rng)`

- **File:** `swift/simulator.py`
- **Parameters:** `params` — a parameter dictionary; `word_freqs` — the
  frequency array for one sentence; `rng` — a random-number generator.
- **What it does:** a small convenience wrapper — runs `simulate_sentence` and
  converts the resulting list of `(word_id, duration_ms)` pairs into a proper
  NumPy numeric array of shape `(n_fixations, 2)` (returning an empty,
  correctly-shaped array if the sentence produced no fixations).

### `run_one_reader(params, word_freqs_list, seq_len, m_sentences, rng)`

- **File:** `swift/simulator.py`
- **Parameters:** `params` — one set of the 3 free parameters (one imaginary
  reader's dial settings); `word_freqs_list` — the full corpus, as built by
  `build_corpus_lists`; `seq_len` — the fixed output length to pad to;
  `m_sentences` — how many sentences this simulated reader should read
  (normally 14, i.e. `M_SENTENCES`); `rng` — a random-number generator.
- **What it does — the single most important modeling decision in the whole
  project:** rather than simulating just one sentence, this function
  randomly picks `m_sentences` different sentences from the corpus, and
  simulates *the same imaginary reader* (the same fixed `params`) reading all
  of them, one after another, gluing all of their fixations together into
  one long combined sequence. It then applies the shared `normalise_sequence`
  and `pad_sequence` functions from `swift/config.py` to that combined
  sequence, and computes the 7 hand-crafted summary statistics via
  `compute_reader_stats` (using the individual per-sentence arrays, before
  they were glued together). It returns a `(sequence, stats)` pair.
- **Why concatenating 14 sentences instead of using just 1 matters so much:**
  a single sentence typically produces only around 8 fixations — far too
  little evidence to reliably tell the difference between, say, "this reader
  has a wide processing span and skips a lot" versus "this particular sentence
  just happened to have a lot of easy, skippable words." Only by watching the
  *same* imaginary reader behave consistently (or not) across many different
  sentences can the pattern that reveals their underlying `nu` and `r` settings
  actually emerge. This is exactly analogous to how you couldn't reliably judge
  someone's reading speed from watching them read a single short word — you'd
  want to watch them read several full paragraphs first.

### `_demo()` — the simulator's built-in self-test

- **File:** `swift/simulator.py`
- **Parameters:** none.
- **How to run it:** `python -m swift.simulator` (or
  `python swift/simulator.py`). It needs no data files, no corpus, no trained
  model — it is completely self-contained and takes about a second.
- **What it does:** this is the project's **unit test for the simulator**, and
  it is the reason we can claim the model equations are implemented correctly
  rather than merely hoping so. It makes three separate assertions:
  1. **The processing span matches the paper's Figure 2.** It hard-codes the
     paper's own published reference values for five values of `nu` (0.1, 0.2,
     0.3, 0.4, 0.6) and checks `span_rates` reproduces them to within 0.001.
     It also checks the weights always sum to exactly 1.
  2. **The span is asymmetric in the right direction** — it explicitly verifies
     that word `k−2` receives *zero* processing while `k+2` receives some. This
     asymmetry is easy to get backwards when implementing the equations, and it
     matters: it is what makes the model read forwards.
  3. **The saccade timer has the right moments.** It simulates 400 sentences
     and checks the mean fixation duration comes out within 8 ms of `mu_T`, and
     that the coefficient of variation is within 0.03 of 1/3 — confirming the
     Gamma distribution is parameterised correctly.

  It also prints the running `TRUNCATIONS` counter, so you can see whether any
  simulated scanpath hit the `MAX_FIX = 200` safety cap.
- **Why it matters for the write-up:** if anyone asks "how do you know your
  simulator actually implements the paper's model?", this is the answer — the
  span weights are checked against the paper's own published figure, and the
  duration distribution against its stated moments. Running it after any change
  to `simulator.py` is the cheapest possible protection against silently
  breaking the model.

## 7.4 `swift/generate.py` — pre-generating training data, in parallel

This file exists to do one thing efficiently: run `run_one_reader` (from
7.3) many thousands of times — once per randomly-sampled parameter setting —
and save all the results to a file, so the (comparatively slow, pure-Python)
simulator never has to be re-run again during actual neural-network training.

### `_init_worker(wfl)`

- **File:** `swift/generate.py`
- **Parameters:** `wfl` — the word-frequency corpus list (one frequency array
  per sentence).
- **What it does:** this is a small setup function that Python's
  **multiprocessing** system (a way of running several separate copies of
  your program at once, on different CPU cores, to go faster — "multi" =
  many, "processing" = separate running programs) calls automatically, once,
  inside *each* separate worker process it creates, to hand that worker its
  own private copy of the corpus data (stored in a global variable, `_WFL`,
  that the worker's other functions can then read from). This avoids having to
  repeatedly copy the entire corpus over to each worker for every single small
  piece of work — it's copied over exactly once per worker, upfront.

### `_gen_chunk(args)`

- **File:** `swift/generate.py`
- **Parameters:** `args` — a two-item pair `(seed, n)`, where `seed` is a
  starting point for that worker's random-number generator (different for
  every chunk of work, so different workers don't accidentally produce
  identical "random" results) and `n` is how many simulated readers this
  particular chunk of work should produce.
- **What it does:** this is the actual function that runs inside each
  parallel worker. It creates its own random-number generator from `seed`,
  then loops `n` times: each time, it draws one random parameter setting
  (`sample_prior`), normalizes it to 0–1 (`normalise_theta`), and runs
  `run_one_reader` to simulate that imaginary reader across 14 sentences. It
  collects all `n` results into three arrays (`thetas`, `seqs`, `stats`,
  matching the format described in Part 4.3) and returns them once the whole
  chunk is done.

### `generate(wfl, n_readers=8000, n_workers=None, seed=42, save_path=TRAINING_DATA)`

- **File:** `swift/generate.py`
- **Parameters:** `wfl` — the word-frequency corpus list; `n_readers` — the total
  number of simulated readers to generate across the whole run (this is
  exactly the `--n_readers` command-line option from Part 3); `n_workers` —
  how many parallel worker processes to use (if left unset, it defaults to
  "one less than the total number of CPU cores available," leaving one core
  free for the operating system and other tasks); `seed` — the base random
  seed; `save_path` — where to save the final combined file.
- **What it does:** first, it figures out how many parallel workers to use.
  Then it splits the total `n_readers` work into a number of smaller
  "chunks" — specifically, four chunks per worker, for better load-balancing
  (so that if one chunk happens to finish a bit faster than another, a
  worker can immediately pick up the next available chunk instead of sitting
  idle) — and assigns each chunk a unique random seed so no two chunks
  produce accidentally-identical "random" results. It then creates a
  multiprocessing "pool" (a managed group of worker processes) using Python's
  `fork` method of creating new processes (a technique that's fast and
  reliably copies over already-loaded data like the corpus, and specifically
  chosen here because it also works correctly when this code is run from
  unusual environments like a notebook, not just a plain script), hands each
  chunk of work off to `_gen_chunk` running inside its own worker, and
  collects all the results back once every worker is finished. It then glues
  all the chunks' results together into three single big arrays, saves them
  to disk as `data/training_data.npz`, and prints a short summary (how many
  readers were generated, how long it took, and the resulting array shapes).
  This whole function corresponds exactly to the `--mode generate` command
  from Part 3.

## 7.5 `swift/inference.py` — the actual BayesFlow neural network

This file is where the neural network itself is defined, trained, and used
to make predictions. It relies on the external **BayesFlow** library (a
specialized Python package specifically built for simulation-based inference
using neural networks) and its underlying computation engine, **PyTorch**.

Right at the top of this file, before anything else happens, the code
deliberately disables Apple's "MPS" GPU-acceleration backend
(`torch.backends.mps.is_available = lambda: False`), forcing all computation
onto the regular CPU instead. As explained in Part 3.1, this is a
compatibility fix, not a performance choice — MPS is missing a specific
linear-algebra operation the network's spline-based component needs.

### `set_corpus(word_freqs_list)`

- **File:** `swift/inference.py`
- **Parameters:** `word_freqs_list` — the word-frequency corpus list.
- **What it does:** stores the given corpus list into this file's own private
  global variable (`_WFL`), so that the two "simulator functions" described
  immediately below (`swift_prior`, `swift_likelihood`) — which BayesFlow
  calls internally on its own, without the rest of the program being able to
  pass arguments to them directly at that moment — still have access to the
  corpus data they need. This function must be called once before training or
  running inference.

### `swift_prior()`

- **File:** `swift/inference.py`
- **Parameters:** none.
- **What it does:** draws one random parameter setting (using
  `sample_prior`) and returns it, already normalized to the 0–1 range, as a
  dictionary with a single key, `"theta"`. This function is handed directly
  to BayesFlow, which will call it repeatedly, on its own, whenever it needs
  a fresh random parameter draw (for example, while generating validation
  data for the diagnostics in Part 7.6).

### `swift_likelihood(theta)`

- **File:** `swift/inference.py`
- **Parameters:** `theta` — one normalized (0–1) parameter setting, as
  produced by `swift_prior`.
- **What it does:** converts `theta` back to real-world units
  (`denormalise_theta`), then runs `run_one_reader` (using the corpus stored
  by `set_corpus` and this file's own shared random-number generator, `_RNG`)
  to simulate one full 14-sentence reader. It returns the resulting sequence
  and stats as a dictionary with keys `"fixations"` and `"stats"`. Together,
  `swift_prior` and `swift_likelihood` are exactly the two pieces BayesFlow
  needs to be able to generate its own fresh simulated training/validation
  examples on demand — this pairing is literally what makes this an
  "amortized simulation-based inference" setup rather than a fixed, one-time
  analysis.

### `build_adapter()`

- **File:** `swift/inference.py`
- **Parameters:** none.
- **What it does:** builds and returns a BayesFlow **Adapter** object — a
  small configuration object that tells BayesFlow exactly which piece of
  data plays which role in the network. Specifically, it configures: the
  `theta` values (the parameters) become the network's **inference
  variables** (literal meaning: "the things we're trying to infer/figure
  out" — i.e., what the network is being trained to predict); the raw
  `fixations` sequence gets renamed to become the network's **summary
  variables** (the raw, variable-length data that needs to be compressed
  down by a summary network before the rest of the model can use it — see
  the next function); and the hand-crafted `stats` numbers become the
  network's **inference conditions** (extra information fed to the network
  directly, unsummarized, alongside whatever the summary network produces).
  It also converts everything from 64-bit to 32-bit floating-point numbers
  (a technical detail about numeric precision that saves memory and speeds
  up computation, with no meaningful loss of accuracy for this application).

### `_build_workflow(simulator)`

- **File:** `swift/inference.py`
- **Parameters:** `simulator` — a BayesFlow simulator object (built by
  `_make_simulator`, below).
- **What it does:** assembles the complete neural network architecture and
  wraps it in a BayesFlow **`BasicWorkflow`** object (BayesFlow's all-in-one
  container for "everything needed to train and use a simulation-based
  inference model"). Two neural network components are created:
  - A **summary network** — specifically `bf.networks.TimeSeriesNetwork`,
    which under the hood is an **LSTM** (Long Short-Term Memory — a
    well-known type of neural network specifically designed to read
    sequences of data, one item at a time, and remember relevant earlier
    information while doing so; think of it as a network built to "read
    through" the fixation sequence the same way you'd read through a list,
    keeping a running mental summary as it goes). It is configured with
    `summary_dim=64` (meaning its final running-summary is compressed down
    to exactly 64 numbers, regardless of how long the original sequence
    was) and `bidirectional=False` (meaning it only reads the sequence
    forward, in the natural time order, rather than also reading it
    backward — the project found that also reading backward,
    "bidirectional," roughly doubled training time with no accuracy
    benefit, so it was left out of the final model — see Part 10).
  - An **inference network** — specifically `bf.networks.CouplingFlow`, a
    type of **normalizing flow** (literal meaning: a mathematical technique
    that learns a flexible, reversible transformation turning a simple,
    well-understood random distribution into a complicated, realistic one —
    "normalizing" refers to starting from/relating to a simple "normal"-style
    base distribution, and "flow" refers to the transformation being built
    from a chain/sequence of smaller reversible steps). It is configured
    with `transform="spline"` (the specific mathematical shape used for each
    reversible step — a flexible curve shape called a spline) and
    `num_layers=6` (six of these reversible transformation steps chained
    together, giving the network more flexibility to represent complex-shaped
    posterior distributions).
  Finally, `standardize=["inference_variables", "inference_conditions"]`
  tells BayesFlow to automatically rescale both the parameters and the
  hand-crafted stats to have a consistent numeric scale internally
  (a standard best practice for neural network training).

### `_make_simulator()`

- **File:** `swift/inference.py`
- **Parameters:** none.
- **What it does:** a one-line convenience function that hands `swift_prior`
  and `swift_likelihood` to `bf.simulators.make_simulator`, producing a
  BayesFlow simulator object that knows how to generate fresh (parameters,
  simulated-sequence) pairs on demand — used both for online training and
  for generating fresh validation data during diagnostics.

### `train_offline(thetas, seqs, stats, wfl, n_epochs=80, batch_size=64, save_path=MODEL_PATH)`

- **File:** `swift/inference.py`
- **Parameters:** `thetas`, `seqs`, `stats` — the three arrays loaded from
  the pre-generated `data/training_data.npz` file (Part 4.3); `wfl` — the
  word-frequency corpus list; `n_epochs` — how many complete passes through the entire
  training dataset to perform (defaults to 80 — an **epoch** is literal
  jargon for "one full pass through all the training examples"); `batch_size`
  — how many training examples the network looks at together at once before
  updating itself (defaults to 64 — training in small batches rather than
  one example at a time, or the whole dataset at once, is standard practice
  that balances speed and training stability); `save_path` — where to save
  the trained network afterward.
- **What it does:** this is the function that actually performs the
  (comparatively slow) neural network training, using data that was already
  pre-generated by `generate()` and saved to disk — hence "offline" (as
  opposed to `train_online`, below, which generates data on the fly during
  training instead). It registers the corpus (`set_corpus`), builds the full
  network architecture (`_build_workflow`), packages the three input arrays
  into the dictionary format BayesFlow expects, and calls BayesFlow's own
  `workflow.fit_offline(...)` method to perform the actual training loop
  (repeatedly showing the network batches of examples, checking how wrong
  its guesses were compared to the true parameters that produced each
  example, and nudging the network's internal numbers slightly to do better
  next time — repeated for `n_epochs` full passes through the data). Once
  training finishes, it saves the trained network to disk
  (`_save_model`) and saves a plot of how the training error changed over
  time (`_save_loss`), then returns the trained `workflow` object so the
  rest of the pipeline can immediately use it.

### `train_online(wfl, n_epochs=80, batch_size=64, num_batches_per_epoch=200, save_path=MODEL_PATH)`

- **File:** `swift/inference.py`
- **Parameters:** similar to `train_offline`, plus `num_batches_per_epoch` —
  how many fresh batches of newly-simulated data to generate per epoch (since
  there's no pre-generated file to draw from here, "how big is one epoch"
  has to be defined explicitly).
- **What it does:** the same overall idea as `train_offline`, but instead of
  loading a pre-generated file, it calls BayesFlow's `workflow.fit_online(...)`
  method, which runs the SWIFT simulator *live*, generating brand-new
  training examples on the fly, throughout the entire training process. This
  is considerably slower overall (since the simulator has to run fresh for
  every single training example, rather than once ever), and in this
  project it's kept mainly as an alternative/comparison path, not the
  recommended way to train.

### `sample_posterior(workflow, observation, stats, num_samples=2000)`

- **File:** `swift/inference.py`
- **Parameters:** `workflow` — a trained BayesFlow workflow; `observation` —
  one padded fixation sequence (shape `SEQ_LEN × N_FEATURES`); `stats` — the
  matching 7 hand-crafted summary numbers for that same observation;
  `num_samples` — how many individual posterior samples to draw (2000 by
  default — the more samples drawn, the more precisely the resulting
  distribution's shape and confidence interval can be estimated).
- **What it does:** feeds one single observation to the trained network and
  asks it to produce `num_samples` separate random draws from its estimated
  posterior distribution over the 3 parameters (this is the core "amortized
  inference" moment — the trained network answers essentially instantly,
  producing a whole distribution of plausible parameter guesses rather than
  a single point estimate). It converts the network's raw 0–1-scaled output
  back into real-world units, and clips any results that happen to fall
  slightly outside the original prior bounds back within them (a small
  safety measure, since a neural network's output isn't mathematically
  guaranteed to stay exactly within the training range).

### `run_inference(workflow, observations, stats, num_samples=2000)`

- **File:** `swift/inference.py`
- **Parameters:** `workflow` — a trained BayesFlow workflow; `observations` —
  potentially *several* different reader-observations (e.g., the 40 different
  random 14-sentence draws of VP10's data built by `build_reader_batch`);
  `stats` — the matching hand-crafted stats for each observation;
  `num_samples` — posterior draws per individual observation.
- **What it does:** this is the function that implements "pooling" —
  combining evidence from multiple different random 14-sentence subsets of
  the same real participant into one single, more reliable overall estimate.
  It loops over every observation given to it, calls `sample_posterior` on
  each one separately, and then simply concatenates (glues together end to
  end) all of the resulting posterior samples into one combined pool. For
  example, with 40 observations and 2000 samples each, the final pooled
  result contains 80,000 total posterior draws. It then prints a short
  summary table to the terminal (mean estimate and 95% confidence-style
  interval, called a **95% credible interval** in Bayesian terminology, for
  each of the 3 parameters) and returns the full pooled array of samples.
  This pooling approach specifically *replaces* an earlier, statistically
  incorrect version of this step that used to average the raw fixation
  sequences together *before* running inference just once — averaging raw
  sequences together is not a valid way to combine evidence (it can produce
  an "average sequence" that doesn't resemble any real plausible reading
  behavior at all), whereas pooling many separately-computed posterior
  samples together is the mathematically correct way to combine multiple
  independent pieces of evidence about the same underlying unknown
  quantity.

### `_save_model(workflow, save_path=MODEL_PATH)`

- **File:** `swift/inference.py`
- **Parameters:** `workflow` — the trained workflow; `save_path` — file
  location to save to.
- **What it does:** makes sure the destination folder exists, then saves the
  trained neural network's internal approximator object to disk in Keras's
  native file format (`.keras`), so it can be reloaded later without
  retraining.

### `_save_loss(history)`

- **File:** `swift/inference.py`
- **Parameters:** `history` — the training-history object returned by
  BayesFlow's `fit_offline`/`fit_online` call, which records how the
  network's training error changed over the course of training.
- **What it does:** uses BayesFlow's own built-in plotting helper
  (`bf.diagnostics.plots.loss`) to draw a simple line chart of training
  **loss** (a standard machine-learning term for "a single number measuring
  how wrong the network's current guesses are — lower is better") over the
  course of training, and saves it as `outputs/figures/training_loss.png`. If
  anything goes wrong while plotting (for example, an unusual environment
  without a display), the error is caught and printed rather than crashing
  the whole training run — the model itself is already safely saved by this
  point regardless.

### `rebuild_workflow(model_path=MODEL_PATH)`

- **File:** `swift/inference.py`
- **Parameters:** `model_path` — file location of a previously-saved trained
  network.
- **What it does:** reconstructs the same network *architecture* (same
  summary network, same inference network, same adapter — via
  `_build_workflow`) that was used during training, and then loads the
  actual trained numeric weights (the learned internal numbers that make the
  network good at its job) from the saved file into that freshly-built
  architecture. This is necessary because saving/loading a full BayesFlow
  workflow object directly isn't how the library is designed to be used —
  instead, you rebuild the empty "shape" of the network fresh each time, and
  just load the learned numbers into it. This function is what powers the
  `--mode infer` command and `tools/show_results.py` (Part 3, Steps 4 and
  5), both of which need a trained network without re-running the (slow)
  training step.

## 7.6 `swift/diagnostics.py` — sanity checks and the final reality check

### `run_builtin_diagnostics(workflow, simulator, n_val=300, n_posterior_samples=1000, save_dir=FIG_DIR)`

- **File:** `swift/diagnostics.py`
- **Parameters:** `workflow` — the trained network; `simulator` — a
  BayesFlow simulator object (for generating fresh validation data);
  `n_val` — how many held-out validation examples to generate (300 by
  default); `n_posterior_samples` — how many posterior draws per validation
  example (1000 by default); `save_dir` — where to save the resulting plot
  images.
- **What it does:** this function runs the crucial "can we actually trust
  this network?" checks, using freshly-generated *simulated* validation data
  where the true parameter values are already known (because they were the
  input to the simulation in the first place) — this is the "Stage 6a" from
  the big-picture diagram in Part 2. It generates `n_val` fresh simulated
  readers, asks the trained network to produce a posterior for each one, and
  then produces and saves four separate diagnostic plots, using BayesFlow's
  own built-in plotting functions:
  - **`recovery_plot.png`** — checks whether the network's average guess (the
    mean of its posterior samples) tracks the actual true parameter value
    across the 300 validation examples. Perfect recovery would put every
    point exactly on a diagonal line.
  - **`sbc_histogram.png`** — an **SBC** ("Simulation-Based Calibration," a
    formal statistical technique from a well-known 2018 paper by Talts et
    al.) check specifically of whether the network's *stated confidence
    levels* are honest, not just whether its point guesses are accurate. In
    plain words: if the network says "I'm 95% confident the true value is in
    this range," is that actually true 95% of the time across many
    validation checks, or is the network being over- or under-confident?
  - **`sbc_ecdf.png`** — the same underlying calibration check as the
    histogram above, but displayed differently (as a cumulative curve with a
    shaded "acceptable" band), which is often easier to read precisely than
    a histogram.
  - **`contraction_plot.png`** — measures **posterior contraction**: how much
    narrower (more confident/precise) the network's answer becomes after
    seeing data, compared to how wide the original prior range was. A high
    contraction value means the data genuinely taught the network something
    specific; a contraction value near zero means the network's answer for
    that parameter looks about as wide/uncertain as just guessing randomly
    from the original prior — a sign that parameter is hard to pin down from
    this kind of data, not a bug in the code.
  All these plots are saved via the shared `_save` helper (described below),
  and the function returns the raw validation data and posterior draws in
  case anything downstream needs them. It **also** produces two simulator-check
  plots — `span_shape.png` (via `plot_span_shape`, the processing-span weights
  vs the paper's Fig. 2) and `scanpath_examples.png` (via
  `plot_scanpath_examples`, example simulated reading paths, cf. Fig. 4) —
  which validate the simulator itself rather than the network.

### `plot_span_shape(save_dir=FIG_DIR)`

- **File:** `swift/diagnostics.py`
- **Parameters:** `save_dir` — where to save the image.
- **What it does:** sweeps `nu` across its whole range (100 values from 0.01 to
  1.0) and, for each value, records the four processing-span weights
  (`lambda_-1`, `lambda_0`, `lambda_+1`, `lambda_+2`) that `span_rates`
  produces for an interior word. It draws all four as curves against `nu` and
  saves the result as `span_shape.png`.
- **Why it exists:** this is a **reproduction of Figure 2 of the paper**, and
  it checks the *simulator*, not the network. It needs no trained model at all.
  If these curves match the paper's figure, the processing-span equations are
  implemented correctly. The plot makes the meaning of `nu` visually obvious:
  at `nu` near 0 the fixated word gets essentially all the processing (weight
  ≈ 1, a very narrow span), and as `nu` grows the neighbouring words' curves
  rise while the fixated word's falls — the reader's attention spreads out.

### `plot_scanpath_examples(word_freqs=None, save_dir=FIG_DIR, nu=0.3, r=10.0, mu_T=200.0, n_examples=4, seed=1)`

- **File:** `swift/diagnostics.py`
- **Parameters:** `word_freqs` — the sentence to read (a default 10-word
  example is used if none is given); `nu`, `r`, `mu_T` — the parameter values
  to simulate with; `n_examples` — how many separate example readings to draw;
  `seed` — for reproducibility.
- **What it does:** simulates the same sentence being read `n_examples`
  separate times with the same fixed parameters, and plots each one as a line:
  fixation number along the horizontal axis, word position on the vertical
  axis. Saves as `scanpath_examples.png`.
- **Why it exists:** a **reproduction of Figure 4 of the paper**, and the most
  intuitive picture in the whole project. Each line is one simulated "reading
  path." A step up by 1 is a normal forward saccade; a flat step is a
  refixation; a jump of 2+ is a skip; a step *down* is a regression. Because
  all four panels use identical parameters, the differences between them show
  purely how much randomness the model contains — the same reader reading the
  same sentence twice does not produce the same scanpath. Like `span_shape`,
  this validates the simulator and needs no trained network.

### `plot_posterior(posterior_samples, true_params=None, save_dir=FIG_DIR)`

- **File:** `swift/diagnostics.py`
- **Parameters:** `posterior_samples` — the pooled posterior samples for
  VP10's real data (produced by `run_inference`); `true_params` — an
  optional dictionary of true parameter values, used only for testing against
  simulated data with a known answer (this is `None` for the real VP10
  analysis, since nobody actually knows VP10's "true" underlying
  parameters — that's exactly the unknown thing being estimated); `save_dir`
  — where to save the resulting image.
- **What it does:** produces a 3-panel figure (one panel per parameter)
  showing, for each of `nu`, `r`, `mu_T`: the full allowed prior range
  (shaded gray background), a histogram of the actual posterior samples for
  VP10 (solid orange), a vertical line marking the mean estimate, and a
  shaded band marking the 95% credible interval (the range that 95% of the
  posterior samples fall within). Saves the result as
  `outputs/figures/posterior_VP10.png`. A narrow, tall orange histogram means
  the data was very informative about that parameter; a wide, flat orange
  histogram that looks almost identical to the plain gray prior background
  means the real data didn't narrow down that parameter very much.

### `plot_posterior_correlation(posterior_samples, save_dir=FIG_DIR)`

- **File:** `swift/diagnostics.py`
- **Parameters:** `posterior_samples` — VP10's pooled posterior samples;
  `save_dir` — where to save the image.
- **What it does:** computes the correlation matrix of the three estimated
  parameters, prints it to the terminal, and saves it as a heatmap
  (`posterior_correlation.png`). This is the **decoupling check**: because
  the basic model draws fixation durations independently of the scanpath, the
  `mu_T`-vs-`nu` and `mu_T`-vs-`r` correlations should come out ≈ 0 — a direct,
  quantitative confirmation on real data that duration and the scanpath are
  separate, exactly as the paper predicts.

### `_sentence_measures(words, durs, n_words)` and `_aggregate(records)`

- **File:** `swift/diagnostics.py`
- **What they do:** small internal helpers used only by
  `posterior_predictive_check` (below). `_sentence_measures` takes one
  sentence's fixated word-ids and durations and computes that sentence's six
  reading measures — SFD (single-fixation duration), GD (gaze duration), TT
  (total time), and whether each word was skipped / refixated / regressed to.
  `_aggregate` pools those per-sentence records across many sentences into the
  final averaged numbers.

### `posterior_predictive_check(posterior_samples, real_fix_df, word_freqs_list, n_ppc=300, save_dir=FIG_DIR, rng=None, sentence_indices=None)`

- **File:** `swift/diagnostics.py`
- **Parameters:** `posterior_samples` — VP10's pooled posterior samples;
  `real_fix_df` — VP10's real fixation table; `word_freqs_list` — the corpus;
  `n_ppc` — how many individual posterior draws to re-simulate from (300 by
  default); `save_dir` — where to save the plot; `rng` — random-number
  generator; `sentence_indices` — which sentences to simulate on (used to pass
  the held-out **test** half, so the PPC is evaluated on sentences the model
  was not fitted on).
- **What it does — this is the final "does the fitted model actually explain
  reality?" check ("Stage 8" in Part 2), and the term "**Posterior Predictive
  Check**" (often abbreviated **PPC**) is standard statistical terminology
  for exactly this kind of check: simulate brand-new data using your
  *estimated* parameters, and see whether that simulated data resembles the
  real data you were originally trying to explain.** It randomly picks
  `n_ppc` individual parameter settings out of VP10's full pooled posterior,
  and for each one, actually re-runs the simplified SWIFT simulator across
  `M_SENTENCES` randomly-chosen test sentences, using `_sentence_measures` /
  `_aggregate` to collect the simulated **six reading measures** (SFD, GD, TT,
  and the skip / refixation / regression probabilities). It computes the same
  six measures from VP10's actual recorded data. Finally it produces and saves
  a comparison figure (`ppc_plot.png`) — three duration histograms (SFD/GD/TT)
  in blue (real) vs orange (simulated), plus a bar chart of the three
  probabilities — and prints a clean text summary table
  (`===== PPC SUMMARY =====`) comparing every real-vs-simulated number side by
  side (the table reproduced in Part 10). **Deliberately, this check compares
  summary *statistics*, never raw fixation-by-fixation sequences** — two
  different reading sequences can differ in their exact details while
  reflecting the same underlying behavior, so comparing raw sequences directly
  would be both noisy and the wrong scientific question.

### `_panel(ax, real, sim, xlabel, bins)`

- **File:** `swift/diagnostics.py`
- **Parameters:** `ax` — one individual subplot area within a larger figure
  (a "matplotlib Axes" object — matplotlib is the standard Python plotting
  library used throughout this project); `real`, `sim` — the real and
  simulated values to compare; `xlabel` — the label to put on the horizontal
  axis; `bins` — how many histogram bars to use.
- **What it does:** a small, reusable helper that draws one overlapping pair
  of histograms (real data in blue, simulated data in orange, both
  semi-transparent so the overlap is visible) onto a given subplot, with
  appropriate axis labels and a small legend. This exact same helper is
  called once per duration-comparison panel (SFD/GD/TT) inside
  `posterior_predictive_check` to avoid repeating the same plotting code.

### `_save(fig, save_dir, filename)`

- **File:** `swift/diagnostics.py`
- **Parameters:** `fig` — a matplotlib figure object; `save_dir` — the
  destination folder; `filename` — the file name to save as.
- **What it does:** another small, reusable helper — saves a given figure to
  disk at the requested location (at a resolution of 150 dots-per-inch, tight
  cropping around the actual content), closes the figure to free up memory,
  and prints a one-line confirmation message. Used by `run_builtin_diagnostics`
  to avoid repeating the same "save and confirm" logic four separate times.

## 7.7 `tools/calibrate.py` — a fast, model-free sanity check

### `main(n_readers=400)`

- **File:** `tools/calibrate.py`
- **Parameters:** `n_readers` — how many random parameter draws to test
  (400 by default).
- **What it does:** this script does **not** involve BayesFlow, neural
  networks, or any trained model at all — it exists purely to check the
  *simulator itself* (independent of any inference) in a matter of seconds.
  It loads the real VP10 data and the corpus, then loops `n_readers` times:
  each time, it draws one random parameter setting from the prior
  (`sample_prior`), picks one random sentence from the corpus, and simulates
  it once. It collects the resulting fixation counts and durations across all
  these random draws, then prints a side-by-side comparison of the *average,
  prior-wide* simulated behavior against the *average* real VP10 behavior
  (mean and standard deviation for each). The purpose is to answer one
  specific, narrow question: "if I average across every plausible parameter
  setting the prior allows, does the simulator's overall, typical behavior
  land in the right ballpark compared to a real human reader?" In the current
  model there are no hand-tuned constants left to calibrate (the three fixed
  constants are all fixed at the paper's values), so this is now only a quick
  marginal sanity check on durations and fixations-per-sentence.

## 7.7b `tools/analyse_information.py` — model-free information analysis

This script produces every number in Parts 12 and 16. Like `calibrate.py` it
uses **no BayesFlow and no trained model** — it works directly from
`data/training_data.npz` (where the true parameters are known, because we chose
them) plus the real VP10 file. That makes it fast (a few seconds) and always
runnable, even on a fresh clone where the `.keras` model has not been retrained.

### `spearman(a, b)`

- **What it does:** computes the Spearman rank correlation between two arrays,
  implemented directly in NumPy (by ranking both arrays and correlating the
  ranks) so the script has no SciPy dependency. Rank correlation is used rather
  than ordinary correlation because some statistic↔parameter relationships are
  curved rather than straight lines, and rank correlation handles that.

### `report_correlations(stats, thetas)`

- **What it does:** prints the 7×3 table of Part 12.1 — how strongly each
  summary statistic tracks each parameter across the simulated readers. This
  answers "does each parameter leave a fingerprint on behaviour at all?"

### `_knn_r2(Ztr, ytr, Zte, yte, j, k=15)`

- **What it does:** measures how well parameter `j` can be predicted from the 7
  statistics using a k-nearest-neighbour predictor (find the 15 most similar
  training readers, average their parameter values), scored as out-of-sample R²
  on readers the predictor never saw. Deliberately a *dumb* method: the point is
  to measure how much information the data contains, independently of how good
  our neural network is.

### `report_information(stats, thetas, seed=0)`

- **What it does:** splits the 8,000 readers into 6,000 train / 2,000 test,
  standardises the statistics, reports the k-NN R² per parameter, and then runs
  the **permutation importance** analysis: for each statistic in turn, it
  scrambles that column across readers (destroying its information while
  keeping its distribution) and measures how much R² is lost. Produces the
  tables of Parts 12.2 and 12.3.

### `report_vp10_plausibility(stats, thetas)`

- **What it does:** builds VP10's real summary statistics the same way the
  pipeline does (`build_reader_batch` on the train half), then reports which
  **percentile** each of VP10's values sits at within the simulated population
  — a **prior predictive check**. A value near the 50th percentile means the
  model produces such readers routinely; below 5 or above 95 flags that the
  real person sits in the tail of what the model can generate at all. This is
  what reveals VP10's regression rate at the 2nd percentile (Part 16.3).

### `report_tradeoff(stats, thetas, vp)`

- **What it does:** the decisive analysis of Part 16.3. It selects the
  simulated readers that closely match VP10 on skip *and* refixation rate and
  reports their regression rate; then selects those matching VP10's low
  regression rate and reports *their* skip and refixation rates. It also prints
  the mean parameters of each group. The output demonstrates that the two
  groups require incompatible parameter settings — i.e. that no single
  parameter setting reproduces all of VP10's behaviour, which is the definition
  of model misspecification.

### `main()`

- **What it does:** runs all four analyses in order, after first printing the
  sequence-length/truncation summary behind Part 13.6 (how many training
  readers hit the `SEQ_LEN = 150` cap).

## 7.8 `tools/show_results.py` — read-only reporting of the current model

### `parse_args()`

- **File:** `tools/show_results.py`
- **Parameters:** none directly (reads from the command line).
- **What it does:** defines and reads the script's command-line options:
  `--n_val` (how many held-out validation readers for the recovery/SBC
  checks, default 300), `--n_posterior_samples` (posterior draws per
  validation reader, default 1000), `--n_vp10_readers` (how many random
  14-sentence VP10 draws to pool, default 40), `--vp10_samples` (posterior
  draws per VP10 draw, default 2000), `--n_ppc` (posterior draws
  re-simulated for the final reality check, default 300), `--seed` (for
  reproducibility, default 0), `--quick` (a flag that, if given, overrides
  every one of the above with much smaller numbers for a fast ~5-second
  smoke test instead of the full ~30-second report), and `--save_json`
  (where to write the machine-readable report copy).

### `hr(title)`

- **File:** `tools/show_results.py`
- **Parameters:** `title` — a string.
- **What it does:** a tiny formatting helper — prints a line of `=`
  characters, the given title, then another line of `=` characters, to make
  the terminal output visually divided into clear sections.

### `classify_r(r)`

- **File:** `tools/show_results.py`
- **Parameters:** `r` — a correlation number (see `run_inference`/recovery
  discussion above for what "recovery r" means).
- **What it does:** converts a raw correlation number into a plain-English
  label: "strong" if `r` is 0.7 or higher, "moderate" if `r` is between 0.35
  and 0.7, and "weak" otherwise. Used purely to make the printed report more
  immediately readable, without needing to mentally interpret raw correlation
  numbers.

### `class _Tee`

- **File:** `tools/show_results.py`
- **What it is:** a small utility class named after the physical plumbing
  fitting called a "T-piece" or "tee" (which splits one pipe of water flow
  into two). Here, it splits one stream of printed text output into two
  destinations at once: the normal terminal screen, and an in-memory text
  buffer. Its two methods, `write` (send text to every destination) and
  `flush` (make sure everything's actually been sent, not sitting in a
  temporary holding area), simply forward each call to every destination
  it was given.
- **Why it exists:** the `posterior_predictive_check` function (Part 7.6)
  already prints a nicely-formatted summary table straight to the terminal.
  Rather than duplicating that table's calculation logic separately just to
  also save it into the machine-readable JSON report, this script instead
  temporarily "tees" the normal console output so it can simultaneously (a)
  still show up normally on screen, exactly as before, and (b) also get
  captured into a buffer, from which the numbers can then be extracted
  (see the next function) and included in the JSON file — without
  duplicating any of the actual calculation code.

### `_parse_ppc_table(text)`

- **File:** `tools/show_results.py`
- **Parameters:** `text` — the full captured console text (from the `_Tee`
  buffer above).
- **What it does:** scans through the captured text looking for the specific
  `===== PPC SUMMARY =====` table that `posterior_predictive_check` prints,
  and carefully re-extracts each row's statistic name, real value, and
  simulated value back out of the fixed-width text formatting, returning
  them as a clean list of small dictionaries ready to be saved into the JSON
  report.

### `main()`

- **File:** `tools/show_results.py`
- **Parameters:** none directly.
- **What it does:** this is the function that runs when you type
  `python tools/show_results.py`. It orchestrates the entire read-only
  report in four numbered stages, printed clearly to the terminal:
  1. **`[1/4] REAL VP10 DATA SUMMARY`** — loads the real fixation data and
     corpus, computes and prints the headline real-data numbers (fixation
     count, sentence count, mean/std duration, duration CV, fixations per
     sentence, skip rate, refixation rate, regression rate).
  2. **`[2/4] PARAMETER RECOVERY & CALIBRATION`** — loads the already-trained
     model (`rebuild_workflow`), generates `n_val` fresh simulated validation
     readers with known true parameters, asks the network for posterior
     samples for each, and computes and prints, for every one of the 3
     parameters (`nu`, `r`, `mu_T`): the recovery correlation `r`, the
     posterior contraction, and the 95% credible-interval coverage (i.e.,
     across all validation readers, what fraction of the time did the true
     value actually fall inside the network's own stated 95% interval — a
     hand-rolled version of the same idea as the SBC plots from
     `run_builtin_diagnostics`, computed here as simple numbers).
  3. **`[3/4] VP10 POSTERIOR ESTIMATES`** — builds random 14-sentence draws
     from the **train half** of VP10's real data (`build_reader_batch` with
     `sentence_ids=train_ids`), pools posterior samples across them
     (`run_inference`), prints the mean estimate, 95% credible interval, and
     prior range for each of the 3 parameters, and also runs
     `plot_posterior_correlation` for the decoupling check.
  4. **`[4/4] POSTERIOR PREDICTIVE CHECK`** — runs
     `posterior_predictive_check` on the held-out **test half** of sentences,
     with its output captured via `_Tee`, then uses `_parse_ppc_table` to fold
     those results into the report.
  After all four stages, it writes the entire collected report (real-data
  summary, recovery numbers, VP10 posterior, and PPC table, plus some basic
  metadata like when the report was generated and how long it took) out to
  `outputs/results_summary.json` as a clean, machine-readable JSON file (a
  standard, simple text format for structured data — "JavaScript Object
  Notation," though it's used far beyond just JavaScript, essentially
  everywhere in modern software), and prints a final summary of where
  everything was saved. Importantly, this function **never modifies or
  retrains the model itself** — it only reads the model file and computes
  fresh numbers from it, which is exactly why it's described throughout this
  project as "read-only."

## 7.9 `main.py` — the command-line entry point tying everything together

### `parse_args()`

- **File:** `main.py`
- **Parameters:** none directly (reads from the command line).
- **What it does:** defines the command-line options this script accepts:
  `--mode` (one of `generate`, `train`, `infer`, `online`, or `all` —
  defaults to `all`), `--n_readers` (how many simulated readers to
  pre-generate, default 8000), `--n_epochs` (training epochs, default 80),
  `--batch_size` (training batch size, default 64), and `--n_workers` (how
  many parallel processes to use for generation, default: automatically
  detected as "cores minus one").

### `step_load()`

- **File:** `main.py`
- **Parameters:** none.
- **What it does:** the very first thing that runs, no matter which `--mode`
  was chosen. It loads the real VP10 fixation file (`load_fixations`), runs
  the exploratory data analysis (`run_eda`, producing the `eda_fixations.png`
  plot and printing the headline VP10 statistics), then attempts to load the
  real corpus file (`load_corpus`) — but if that file happens to be missing
  from disk, it prints a warning and falls back to the fake `synthetic_corpus`
  instead of crashing outright. Finally it converts the corpus into the
  per-sentence list format the simulator needs (`build_corpus_lists`), and
  returns `(fix, wfl)` for the rest of the pipeline to use.

### `step_generate(wfl, args)`

- **File:** `main.py`
- **Parameters:** `wfl` — the word-frequency corpus list; `args` — the parsed
  command-line arguments.
- **What it does:** a thin wrapper that simply calls `swift.generate.generate`
  (Part 7.4) with the corpus and the requested `--n_readers`/`--n_workers`
  settings, and returns its result.

### `step_train_offline(wfl, args)`

- **File:** `main.py`
- **Parameters:** same as above.
- **What it does:** checks that the pre-generated `data/training_data.npz`
  file actually exists (raising a clear error message telling you to run
  `--mode generate` first if it doesn't), loads the three arrays out of that
  file, prints how many readers were loaded, and calls
  `swift.inference.train_offline` (Part 7.5) with the requested
  `--n_epochs`/`--batch_size` settings, returning the trained workflow.

### `step_diagnostics(workflow, wfl)`

- **File:** `main.py`
- **Parameters:** `workflow` — a trained network; `wfl` — the corpus list.
- **What it does:** registers the corpus for the inference module
  (`set_corpus`) and calls `run_builtin_diagnostics` (Part 7.6) with
  `n_val=300` validation readers and `n_posterior_samples=1000` — these two
  particular numbers are hard-coded here rather than exposed as command-line
  options.

### `step_infer(workflow, fix, train_ids)`

- **File:** `main.py`
- **Parameters:** `workflow` — a trained network; `fix` — VP10's real
  fixation table; `train_ids` — the first-half (train) sentence ids from
  `split_half`.
- **What it does:** builds 40 random 14-sentence observations from VP10's
  **train-half** data (`build_reader_batch` with `sentence_ids=train_ids` and
  a fixed random seed for reproducibility), calls `run_inference` (Part 7.5)
  with 2000 posterior samples per observation to produce the pooled VP10
  posterior, calls `plot_posterior` to save the 3-panel figure and
  `plot_posterior_correlation` for the decoupling check, and returns the
  pooled posterior samples.

### `step_ppc(posterior, fix, wfl, test_ids)`

- **File:** `main.py`
- **Parameters:** `posterior` — VP10's pooled posterior samples; `fix` —
  VP10's real data; `wfl` — the corpus list; `test_ids` — the second-half
  (test) sentence ids from `split_half`.
- **What it does:** calls `posterior_predictive_check` (Part 7.6) with
  `n_ppc=300` posterior draws, restricted to the held-out test sentences
  (`sentence_indices`), producing and saving the final `ppc_plot.png`
  comparison image and printing the PPC summary table.

### The `if __name__ == "__main__":` block — the actual command dispatcher

- **File:** `main.py`
- **What it does:** this is the code that actually runs when you type
  `python main.py ...` on the command line. It first parses the command-line
  arguments and always runs `step_load()` first, regardless of which mode
  was chosen (since every mode needs the real data and corpus loaded). Then,
  depending on the chosen `--mode`, it runs a different sequence of the
  step-functions above:
  - **`generate`** → runs only `step_generate`, then prints a hint suggesting
    the next command to run (`python main.py --mode train`).
  - **`train` or `all`** → if the mode is specifically `all`, it first runs
    `step_generate` (this is exactly what makes `--mode all` equivalent to
    running `generate` followed by `train`); then, in both cases, it runs
    `step_train_offline`, followed by `step_diagnostics`, `step_infer`, and
    finally `step_ppc`, in that exact order.
  - **`online`** → calls `swift.inference.train_online` directly (instead of
    `step_train_offline`), then still runs `step_diagnostics`, `step_infer`,
    and `step_ppc` afterward, in the same order as above.
  - **`infer`** → checks that a trained model file already exists on disk
    (raising a clear error telling you to run `--mode train` first if it
    doesn't), reconstructs the trained network from disk
    (`rebuild_workflow`), and then runs `step_diagnostics`, `step_infer`, and
    `step_ppc`, in that same order — skipping the training step entirely.

This dispatcher is the single source of truth for "what actually happens, in
what order, for each command" — Part 8, next, walks through each mode's exact
function-call sequence one more time, tying everything in Part 7 together
end to end.

---

# PART 8 — What Actually Happens, Command by Command

This section traces, in order, exactly which functions from Part 7 get
called for each of the main commands from Part 3 — useful as a quick
reference once you already understand what each individual function does.

## 8.1 `python main.py --mode generate --n_readers 8000`

1. `main.py: step_load()` → `data.py: load_fixations`, `data.py: run_eda`,
   `data.py: load_corpus` (or `data.py: synthetic_corpus` as a fallback),
   `data.py: build_corpus_lists`.
2. `main.py: step_generate()` → `generate.py: generate()`, which internally
   spins up parallel workers each running `generate.py: _init_worker()` once
   and then `generate.py: _gen_chunk()` repeatedly — and *that* function
   calls `simulator.py: sample_prior()`, `simulator.py: normalise_theta()`,
   and `simulator.py: run_one_reader()` (which itself calls
   `simulator.py: simulate_one_sentence_features()` fourteen times per
   reader, each of which calls `simulator.py: simulate_sentence()`, which in
   turn calls `simulator.py: span_rates()` and draws Gamma durations
   internally) — followed by
   `config.py: normalise_sequence()`, `config.py: pad_sequence()`, and
   `config.py: compute_reader_stats()` to finish formatting each simulated
   reader.
3. Results are saved to `data/training_data.npz`.

## 8.2 `python main.py --mode train`

1. `step_load()` (same as above).
2. `main.py: step_train_offline()` → checks the training file exists, loads
   it, calls `inference.py: train_offline()`, which calls
   `inference.py: set_corpus()`, `inference.py: _build_workflow()` (which
   builds the summary network, the inference network, and calls
   `inference.py: build_adapter()`), then BayesFlow's own
   `workflow.fit_offline(...)` training loop, then
   `inference.py: _save_model()` and `inference.py: _save_loss()`.
3. `main.py: step_diagnostics()` → `inference.py: set_corpus()`,
   `diagnostics.py: run_builtin_diagnostics()`, which calls
   `inference.py: _make_simulator()` (via `main.py`), generates 300 fresh
   validation readers (again running the full simulator chain from 8.1,
   step 2), asks the network for posterior samples, and produces the four
   diagnostic plots.
4. `main.py: step_infer()` → `data.py: build_reader_batch()` (which calls
   `data.py: build_reader_observation()` forty times, each of which calls
   `data.py: sentence_features()` fourteen times, plus
   `config.py: normalise_sequence()`, `config.py: pad_sequence()`, and
   `config.py: compute_reader_stats()`), then
   `inference.py: run_inference()` (which calls
   `inference.py: sample_posterior()` forty times, pooling the results),
   then `diagnostics.py: plot_posterior()`.
5. `main.py: step_ppc()` → `diagnostics.py: posterior_predictive_check()`,
   restricted to the held-out **test half** of sentences. It calls
   `diagnostics.py: _sentence_measures()` once per real sentence and once per
   simulated sentence, pools both through `diagnostics.py: _aggregate()`, and
   finally draws the figure via `diagnostics.py: _panel()` (three duration
   panels) plus a bar chart, then prints the PPC summary table.

## 8.3 `python main.py --mode all --n_readers 8000`

Exactly 8.1 followed immediately by 8.2 (specifically, `step_generate()` runs
first, then everything from 8.2's steps 2 onward).

## 8.4 `python main.py --mode infer`

1. `step_load()`.
2. Checks a trained model already exists on disk; calls
   `inference.py: rebuild_workflow()` (which calls `_build_workflow()` again,
   then loads the saved numeric weights from disk instead of training fresh).
3. `step_diagnostics()`, `step_infer()`, `step_ppc()` — identical to steps
   3–5 of 8.2.

## 8.5 `python tools/show_results.py`

1. Checks a trained model exists on disk.
2. Loads real data and corpus directly (`load_fixations`, `load_corpus`,
   `build_corpus_lists`), prints stage `[1/4]`.
3. `inference.py: set_corpus()`, `inference.py: rebuild_workflow()`.
4. Stage `[2/4]`: `inference.py: _make_simulator()`, generates `n_val` fresh
   validation readers, samples posteriors, hand-computes recovery
   correlation / contraction / coverage per parameter and prints them
   (via `tools/show_results.py: classify_r()` for the plain-English labels).
5. Stage `[3/4]`: `data.py: build_reader_batch()`,
   `inference.py: run_inference()` (with its normal printout silenced),
   prints VP10's posterior table.
6. Stage `[4/4]`: `diagnostics.py: posterior_predictive_check()`, its output
   captured via `tools/show_results.py: _Tee` and parsed back out with
   `tools/show_results.py: _parse_ppc_table()`.
7. The full collected report is written to `outputs/results_summary.json`.

## 8.6 `python tools/calibrate.py`

A completely separate, self-contained path: `tools/calibrate.py: main()`
calls `data.py: load_fixations()`, `data.py: load_corpus()`,
`data.py: build_corpus_lists()`, then loops calling
`simulator.py: sample_prior()` and `SWIFTSimulator.simulate_sentence()`
directly, with no neural network, no `main.py`, and no `swift/inference.py`
involved at all.

---

# PART 9 — How to Read Every Output Plot

All plots are saved as image files (`.png`) inside `outputs/figures/`. Here
is what each one shows and what a "good" result looks like.

| Plot file | What is shown | What a good result looks like |
|---|---|---|
| `span_shape.png` | The processing-span weights (how much words `k−1, k, k+1, k+2` get processed) for a few values of `nu`. | Matches the paper's Figure 2 — this is a check on the *simulator*, not the network. |
| `scanpath_examples.png` | A few example *simulated* reading paths: which word is fixated at each step. | Plausible left-to-right reading with the occasional skip, refixation, or backward jump — cf. the paper's Figure 4. |
| `eda_fixations.png` | Histograms of VP10's real data: fixation duration and saccade amplitude, plus the skip/refixation/regression rates. | There's no "pass/fail" here — this is just context, establishing the benchmark numbers everything else gets compared against later. |
| `training_loss.png` | How the neural network's training error changed over each training epoch. | The line should generally go down and then flatten out — steadily decreasing and leveling off, not still dropping steeply right at the very end (which would suggest more training was needed) and not bouncing around erratically or increasing (which would suggest a training problem). |
| `recovery_plot.png` | One panel per parameter (`nu`, `r`, `mu_T`): the known true value (horizontal axis) plotted against the network's average guess (vertical axis), across 300 held-out simulated examples. | Points clustering tightly along the diagonal line = accurate recovery. `mu_T` sits almost exactly on it; `nu`/`r` form tighter clouds around it. |
| `sbc_histogram.png` | For each parameter, a histogram of "where does the true value rank among the network's own posterior samples," repeated across many validation examples. | Should look roughly flat/uniform. A U-shape (too many extreme ranks) means the network's stated uncertainty is too narrow (overconfident). A hump in the middle means the stated uncertainty is too wide (underconfident). |
| `sbc_ecdf.png` | The same underlying calibration check as the histogram above, shown as a cumulative curve with a shaded confidence band. | The curve should stay inside the shaded band the whole way across. |
| `contraction_plot.png` | For each parameter, how much narrower the posterior became compared to the original prior range. | Higher = the network is genuinely using the data to narrow things down. Near zero = the posterior looks about as wide as just guessing from the prior. |
| `posterior_VP10.png` | VP10's actual estimated posterior for each parameter (solid orange), shown against the flat gray prior range. | A narrow, clearly-peaked orange distribution = an informative, confident estimate for that parameter. |
| `posterior_correlation.png` | A heatmap of how the three estimated parameters correlate with each other for VP10. | The `mu_T`-vs-`nu` and `mu_T`-vs-`r` cells should be ≈ 0 — confirming duration is decoupled from the scanpath, as the basic model predicts. |
| `ppc_plot.png` | Real VP10 data (blue) versus data freshly simulated from the estimate (orange): three duration histograms (SFD/GD/TT) plus a bar chart of skip/refixation/regression probabilities. | Substantial overlap on the durations; the regression bar is the known miss (Part 10.4). The final "does the fitted model behave like the real person" check. |

Five further figures are produced not by `main.py` but by the
all-participants extension (`python tools/all_participants_ppc.py`). They are
listed here for completeness; **Part 19 explains each of them plot-by-plot,
including every concept needed to read them**:

| Plot file | What is shown | What a good result looks like |
|---|---|---|
| `all_participants_ppc.png` | One dot per participant (34 dots): the real value of each of six reading measures on that person's held-out sentences (horizontal) vs. the value simulated from their fitted parameters (vertical), with a correlation `r` per panel. | Dots hugging the dashed diagonal, high `r`. Ours: durations and skipping excellent (r = 0.98 / 0.96 / 0.95 / 0.92), refixation moderate (0.74), regression weakest (0.66) with dots sitting *above* the line — the per-person version of the known over-regression (Parts 16, 19.5). |
| `all_participants_ppc_pooled.png` | The same 4-panel layout as `ppc_plot.png`, but pooling all 34 participants' real (blue) vs. simulated (orange) measures together. | Overlapping histograms and matched bar pairs. Ours: durations overlap well; skip ≈16% real vs ≈14% simulated, refixation ≈16% vs ≈11%, regression ≈6% vs ≈11% (Part 19.6). |
| `all_participants_posteriors.png` | One panel per parameter; each gray curve is one participant's posterior density, the blue curve is the average across all 34. | Narrow gray curves (informative data), peaks spread across different locations (real individual differences), none pressed against the plot edges (prior wide enough). Ours passes all three (Part 19.7). |
| `all_participants_posterior_correlation.png` | A 3×3 correlation heatmap over all 680,000 pooled posterior samples (34 people × 20,000 samples). | Values near 0. Non-zero entries here mostly reflect *between-person* differences leaking into the pooled bag, **not** within-person coupling — the key subtlety explained in Part 19.8. |
| `all_participants_theta_correlation.png` | A 3×3 correlation heatmap of the 34 per-person point estimates — how the parameters co-vary *across people*. | There is no pass/fail target — this one is a *finding*, not a check: e.g. `r` vs `mu_T` = −0.53 suggests a general "reading speed" dimension across people (Part 19.8). |

The `outputs/figures/baseline_M10/` subfolder contains stale plots from the
superseded 4-parameter full-SWIFT model (at `M_SENTENCES = 10`) — kept only as
historical reference, and **not** comparable to the current 3-parameter run.

> **This table is the quick reference. [Part 17](#part-17--every-plot-precisely-what-is-compared-to-what)
> goes much deeper** — in particular it makes explicit *what is being compared
> to what* in each plot (estimate vs. known truth, real vs. simulated, or real
> data alone). That distinction is the key to resolving the most common
> confusion in this project: why `nu` can look excellent in `recovery_plot.png`
> while the regression bar looks bad in `ppc_plot.png`, with both being true at
> once.

---

# PART 10 — The Final Results This Project Achieved

*(Reproducible any time, in about 30 seconds, by running
`python tools/show_results.py` — every number below is computed live from
whatever model is currently saved on disk, not hard-coded.)*

## 10.1 The real VP10 benchmark numbers

| Statistic | Value |
|---|---|
| Total fixations | 877 |
| Sentences | 114 |
| Mean fixation duration | 196.9 ± 48.4 ms |
| Duration coefficient of variation | 0.246 |
| Mean fixations per sentence | 7.69 |
| Skip rate | 19.6% |
| Refixation rate | 10.2% |
| Regression rate | 2.04% |

## 10.2 Parameter recovery on 300 held-out simulated examples (known truth)

| Parameter | Recovery correlation | Posterior contraction | 95% interval coverage | Identifiability |
|---|---:|---:|---:|---|
| `nu` | 0.941 | 0.866 | 95.0% | strong |
| `r` | 0.959 | 0.904 | 96.0% | strong |
| `mu_T` | 0.997 | 0.990 | 95.3% | strong |

All three parameters recover strongly and land close to their nominal 95%
coverage target, meaning the network's *stated confidence levels* are honest.
`mu_T` is essentially exact (it is just the average of the duration
distribution, whose shape is known); `nu` and `r` are noisier but clearly
identified — helped substantially by feeding the network the hand-crafted
reading-measure statistics (Part 7.1), especially the regression rate, which
is the most direct signal about `nu`.

## 10.3 VP10's estimated parameters (pooled over 40 random 14-sentence draws)

| Parameter | Mean estimate | 95% credible interval | Original prior range |
|---|---:|---|---|
| `nu` | 0.40 | [0.28, 0.54] | [0.0, 1.0] |
| `r` | 6.33 | [5.27, 7.87] | [0.0, 12.0] |
| `mu_T` | 198.0 ms | [182.0, 216.1] ms | [100, 400] ms |

Note `mu_T ≈ 198 ms` matches VP10's mean fixation duration of 196.9 ms almost
exactly — expected, because in this model the average simulated duration
equals `mu_T` by construction.

**Decoupling check.** The basic model predicts that the duration dial (`mu_T`)
should be independent of the scanpath dials (`nu`, `r`). The estimated
posterior confirms it: the correlation between `mu_T` and each of `nu`, `r` is
≈ 0.007 / 0.061 — essentially zero (`posterior_correlation.png`). This is a
direct, quantitative confirmation on real data that the temporal and spatial
sides of the model are separate, exactly as Engbert & Rabe describe.

## 10.4 The final reality check (Posterior Predictive Check)

Evaluated on the held-out **second half** of VP10's sentences (the model was
fit on the first half — a train/test split, as in the paper's Section 6):

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

(SFD = single-fixation duration, GD = gaze duration, TT = total time — the
three standard duration measures.) The three duration measures match within
about 6 ms, and skip/refixation rates match within a couple of points. The one
clear miss is **regressions: about 10% simulated versus 2% real** — the
simplified model produces backward jumps purely as a side-effect of its
processing span, with no mechanism to suppress them, so it over-regresses.
This is an honest limitation of the basic model, reported as a finding.

## 10.5 Honest limitations, worth stating plainly (deviations from the paper)

- **The model over-produces regressions** (~10% vs VP10's ~2%). Regressions
  emerge from the processing span with no suppression term; the paper's basic
  model has the same structure. Adding an explicit regression/inhibition
  mechanism (deliberately *not* done here — it would double-count against the
  emergent rule) is the natural extension.
- **The model over-predicts how variable durations are.** Because the shape
  constant `alpha` is fixed at 9, simulated durations always have a spread of
  CV = 1/3 ≈ 0.333, but VP10's real durations are tighter (CV ≈ 0.246). This
  is structural, not a tuning issue — `alpha` is not a free parameter.
- **The "seconds" interpretation** in the processing-rate step (Part 5.4) is
  the one place the implementation reconciles the paper's millisecond wording
  with its `r` values; documented rather than silently assumed.
- **The frequency-effect constant `beta` is fixed at 0.6**, not freely
  estimated (it is free only in the paper's larger 5-parameter model), and the
  extended-model timer-coupling term `iota` is not implemented.

These are genuine scientific findings about the *simplified* model, confirmed
by the recovery, contraction, decoupling, and PPC checks above — not bugs. The
full list lives in [RESULTS.md §6](RESULTS.md).

## 10.6 Where to go next in this document

Parts 1–10 covered what the project is and what it produced. The remaining
parts analyse *why* those results came out the way they did:

- **Why the regression rate misses by 5× — and proof it is not our fault:**
  [Part 16](#part-16--the-regression-problem-a-worked-analysis). This is the
  most important follow-up, and it demonstrates that no parameter setting can
  match VP10's regression *and* refixation rates simultaneously.
- **Whether the model overfits:**
  [Part 13](#part-13--overfitting-why-it-is-an-unusual-question-here-and-what-we-found).
- **Whether the data even contains information about the parameters:**
  [Part 12](#part-12--does-the-data-actually-contain-information-about-the-parameters),
  with four independent lines of evidence.
- **How the parameters relate to each other and which data columns drive
  which:** [Part 11](#part-11--the-parameters-in-depth).
- **How we compare against the original paper, and whether more data or a
  better network would help:**
  [Part 15](#part-15--comparison-with-the-original-paper).
- **How the same trained network fares on the other 33 participants** — and
  the five cross-participant figures, explained plot-by-plot:
  [Part 19](#part-19--the-all-participants-extension-one-trained-network-34-real-readers).

---

# PART 11 — The Parameters in Depth

Part 5 introduced the three parameters. This part goes much deeper: what each
one *physically* controls, how they push against each other, and — the question
most people ask first — in what sense they are or are not "influenced by the
data."

## 11.1 The question everyone asks first: "are these parameters influenced by the data we trained on?"

This question sounds simple but actually contains three different questions
tangled together, and they have three *different* answers. Untangling them is
the single most important conceptual step in understanding this project.

**Question A — "Do the parameter *ranges* (the priors) come from our data?"**
**No.** The ranges `nu ∈ [0,1]`, `r ∈ [0,12]`, `mu_T ∈ [100,400] ms` are taken
directly from Engbert & Rabe (2024), Section 5. They were chosen *before* any
data was looked at, and nobody inspected VP10's recording to pick them. This
matters: if we had peeked at VP10's mean fixation duration (197 ms) and then
set the prior to, say, `[190, 205] ms`, we would be smuggling the answer into
the question, and the resulting "estimate" would be worthless — it could hardly
have come out any other way.

**Question B — "Are the neural network's internal weights influenced by data?"**
**Yes — but only by *simulated* data.** The network was trained on 8,000
simulated readers generated by our own simulator from randomly drawn
parameters. It never saw VP10's real recording during training, not once. So
the network's learned skill is "how to read parameters off *any* sequence that
this simulator could produce," not "how to produce the answer VP10 needs."

**Question C — "Is the final answer for VP10 influenced by VP10's data?"**
**Yes — and that is entirely the point.** The posterior distribution for VP10
is exactly "what the parameters probably are, *given VP10's observed
fixations*." If it were not influenced by VP10's data, the method would have
failed: the posterior would just be the prior again.

Putting the three together, here is the honest one-sentence answer you can give
if someone asks:

> The parameters themselves are properties of the *reader*, not of our data.
> Their plausible ranges came from the published paper, not from our dataset.
> The network learned how to *recognise* those parameters using only simulated
> data. And the final estimate for VP10 is driven by VP10's own real fixations
> — which is exactly what an estimate is supposed to be.

**Why this separation is the whole design.** This is what "amortized" inference
buys us. Because training used only simulated data with known correct answers,
we were able to check the network's trustworthiness thoroughly (Part 10.2)
*before* pointing it at the real person. The real data is spent once, at the
very end, on the question we actually care about. Nothing is spent on tuning.

## 11.2 What each parameter physically controls

| Parameter | The single sentence version | Turn it **up** and… | Turn it **down** and… |
|---|---|---|---|
| `nu` (0–1) | How wide the reader's attention window is | Neighbouring words get pre-processed, so more words are already "finished" when the eye arrives → **more skipping**; the word to the left also gets processed → **more regressions**; jumps get longer | Attention is narrow, almost all processing goes to the fixated word → the eye plods word by word, few skips |
| `r` (0–12) | How fast words get processed per second of looking | Words finish within one fixation → **few refixations**, **fewer fixations per sentence**, faster reading | Words need several visits to finish → **many refixations**, **more fixations per sentence** |
| `mu_T` (100–400 ms) | How long each individual pause lasts | Every fixation is simply longer | Every fixation is simply shorter |

The crucial structural fact, worth memorising: **`nu` and `r` control *where*
the eye goes; `mu_T` controls only *how long* it stays.** They are two separate
machines in the model that do not talk to each other.

## 11.3 How the parameters relate to each other

This is a question the project can answer *with measured numbers*, not just
theory. The correlation matrix of VP10's estimated posterior
(`posterior_correlation.png`) is:

```
              nu       r    mu_T
nu         1.000  -0.317   0.007
r         -0.317   1.000   0.061
mu_T       0.007   0.061   1.000
```

Read this as "when the model considers explanations of VP10's data, do these
two dials move together?"

**`mu_T` vs `nu` (0.007) and `mu_T` vs `r` (0.061): essentially zero.** This is
the **decoupling** result, and it is a genuine scientific confirmation, not a
formality. The paper's basic model *predicts* that timing and scanpath should
be independent, because fixation durations are drawn from a Gamma distribution
that never consults the activation state. Our estimate on real human data
confirms it. Practically, this means you could get `mu_T` badly wrong and it
would not corrupt `nu` or `r` at all — the duration evidence and the scanpath
evidence are processed separately.

**`nu` vs `r` (−0.317): a real, moderate negative trade-off.** This is the one
genuine interaction in the model, and it deserves a plain-language explanation
because it comes up repeatedly (and is the subject of Part 16).

Both `nu` and `r` control how quickly words reach "finished." A wide span (high
`nu`) spreads processing to more words at once; a fast rate (high `r`) pours
more processing in per second. **Two different dial settings can therefore
produce a very similar amount of total processing**, which means the data
cannot fully distinguish between them. If the network raises its estimate of
`nu`, it must lower its estimate of `r` to keep the predicted behaviour
matching what was observed — hence the negative correlation.

The number −0.317 is the honest measure of how much they blur together. It is
**moderate, not severe**: at −0.9 you would say the two parameters are barely
distinguishable and should perhaps be combined into one; at −0.317 they are
clearly separable but not fully independent. This is consistent with both
recovering strongly (0.941 and 0.959, Part 10.2) — the data *can* tell them
apart, just not perfectly.

**What breaks this trade-off?** The two parameters have different
*signatures*, which is exactly why they are separable at all:
- `r` shows up most strongly in the **refixation rate** and **fixations per
  sentence** — how often the eye has to come back to the same word.
- `nu` shows up most strongly in the **skip rate** and **saccade amplitude** —
  how far ahead the eye jumps.

Because these are different observable measures, feeding both to the network
(as two of the seven hand-crafted statistics) is what lets it disentangle the
two. Part 12 shows this with measured numbers.

## 11.4 The fixation-sequence file: the header it does not have

`fixseqin_PB2expVP10.dat` has **no header row** — the first line is already
data. If you want to open it in Excel or R and see meaningful column names,
here is the header line to paste on top (space-separated, matching
`FIXATION_COLUMNS` in `swift/data.py`):

```
sentence_id word_id landing_position fixation_duration word_length fixation_type flag1 flag2 fixation_index participant_id
```

A real line from the file, lined up against those names:

```
sentence_id    word_id  landing_position  fixation_duration  word_length  ...
          1          1              2.30                183            5  ...
```

meaning: *in sentence 1, the eye fixated word 1, landing 2.3 characters into
it, and stayed for 183 ms; that word is 5 characters long.*

## 11.5 Which columns actually drive which parameter

This is the practical version of "where does each estimate come from?" Of the
**ten** columns in the fixation file, this project's simplified model uses
only **three**, and each parameter is driven by a different aspect of them.

| Parameter | Driven by which column(s) | Through what mechanism |
|---|---|---|
| `mu_T` | **`fixation_duration`** (column 4) — and *nothing else* | The mean of the durations. Since `E[duration] = mu_T` by construction, this is almost a direct read-off. |
| `r` | **`word_id`** (column 2), ordered by **`fixation_index`** (column 9) | Detected via how often consecutive fixations land on the *same* word (refixation rate) and how many fixations each sentence needs. |
| `nu` | **`word_id`** (column 2), ordered by **`fixation_index`** (column 9) | Detected via which word positions get *no* fixation (skip rate), how far consecutive fixations jump (saccade amplitude), and how often the sequence steps backward (regression rate). |

And the columns that are **not** used at all:

| Column | Why it is unused |
|---|---|
| `landing_position` (3) | The simplified model has no spatial extent — words are points at positions 1, 2, 3, …, so "where inside the word" has no meaning in this model. |
| `word_length` (5) | Same reason. The full SWIFT model uses word length; the basic model does not. |
| `fixation_type` (6) | Not needed — `fixation_index` already gives ordering. |
| `flag1`, `flag2` (7, 8) | Always 0 in this file; purpose unknown. |
| `sentence_id` (1) | Used for *grouping and ordering* (splitting fixations into sentences, and for the train/test split), not as a direct signal about any parameter. |
| `participant_id` (10) | Always 10 — this file is one participant only. |

**The single most useful takeaway from this table:** `mu_T` reads off column 4,
while `nu` and `r` read off column 2. They use *different columns*. This is the
mechanical reason for the decoupling result in 11.3 — it is not a coincidence
or a lucky finding, it is baked into which numbers each parameter can even see.

And from the corpus file, exactly one column feeds the model: **`freq`** (word
frequency), which sets each word's maximum activation `a_max = 1 − beta·q`.
Rarer words need more processing. `length` and `code` are loaded but unused.

---

# PART 12 — Does the Data Actually Contain Information About the Parameters?

Before trusting any estimate, there is a question that must be asked first, and
it is *not* about the neural network: **is the information even there?** If
VP10's fixations simply do not contain any trace of `nu`, then no network —
however large or well-trained — could ever recover it. You would be estimating
noise.

This part answers that question with measured numbers rather than assertions.
It is the part to read if someone asks "how do you know the network isn't just
making things up?"

> **Every number in this part (and in Part 16) is reproducible in a few seconds
> with:**
>
> ```bash
> python tools/analyse_information.py
> ```
>
> That script needs **no trained model** — it works directly from
> `data/training_data.npz`, where the true parameters are known because we
> chose them. So it runs even on a fresh clone where the `.keras` file has not
> been retrained yet.

## 12.1 Test 1 — Do the summary statistics move when the parameters move?

The first, simplest check: take the 8,000 simulated readers in
`data/training_data.npz` (where the true parameters are known exactly, because
we chose them), and measure the correlation between each of the 7 hand-crafted
statistics and each of the 3 parameters. If a parameter genuinely leaves a
fingerprint on behaviour, some statistic must move when it moves.

Measured Spearman correlations (a robust measure of "do these move together,"
where 0 = no relationship and ±1 = a perfect relationship):

| Statistic | vs `nu` | vs `r` | vs `mu_T` |
|---|---:|---:|---:|
| mean duration | −0.002 | 0.009 | **0.996** |
| std. dev. of duration | −0.001 | 0.004 | **0.979** |
| fixations per sentence | −0.035 | **−0.698** | −0.305 |
| skip rate | **0.610** | 0.529 | 0.293 |
| refixation rate | 0.222 | **−0.720** | −0.356 |
| regression rate | 0.443 | −0.567 | −0.215 |
| saccade amplitude | **0.722** | 0.286 | 0.228 |

Several things jump out, and each one is worth being able to explain:

**`mu_T` is essentially solved by one number.** Its correlation with mean
duration is 0.996 — almost perfect. This is why `mu_T` recovers at 0.997 in
Part 10.2. It is not that the network is clever about `mu_T`; it is that `mu_T`
*is* the mean duration, by the model's own construction.

**Duration statistics carry nothing about `nu` and `r`** (correlations of
−0.002 and 0.009 — indistinguishable from zero). This is the decoupling of
Part 11.3, visible directly in the training data. The two halves of the model
really do not leak into each other.

**`r` is best revealed by the refixation rate (−0.720) and fixations per
sentence (−0.698)**, both negative: a faster reader revisits words less and
needs fewer fixations. Exactly as the mechanism predicts.

**`nu` is best revealed by saccade amplitude (0.722) and skip rate (0.610)** —
a wider span means longer jumps and more skipped words.

**An important correction to a claim you may see elsewhere in this project.**
The code comments in `swift/config.py` and `swift/data.py` describe the
*regression rate* as "the most direct observable signal about `nu`." That is
true **mechanistically** — leftward processing (`lambda_-1 = sigma·nu`) is the
model's only source of backward jumps, so without `nu` there would be no
regressions at all. But **empirically it is not the strongest signal**: the
regression rate correlates 0.443 with `nu`, while saccade amplitude correlates
0.722 and skip rate 0.610. The reason is that the regression rate is also
strongly pulled by `r` (−0.567) — a slow reader lingers and drifts backward
more — so as a *marginal* signal it is muddied. Both statements are correct;
they answer different questions. If asked, the precise answer is: *"regression
rate is the only mechanism by which `nu` creates backward movement, but skip
rate and saccade amplitude are the statistics that predict `nu` best in
practice, because regressions are confounded by `r`."*

## 12.2 Test 2 — How much information is there in total?

Correlations only measure one statistic at a time, and only straight-line
relationships. A stronger test: **can the 7 statistics, taken together, predict
the parameters at all** — using a simple, dumb method with no neural network
involved?

Using a plain nearest-neighbour predictor (find the 15 most similar simulated
readers and average their parameters), trained on 6,000 readers and tested on
the 2,000 it had never seen:

| Parameter | Predictable from the 7 statistics alone (out-of-sample R²) |
|---|---:|
| `nu` | 0.844 |
| `r` | 0.858 |
| `mu_T` | 0.978 |

(R² of 1.0 means perfectly determined; 0.0 means the statistics tell you
nothing.)

**This is the key result of this part.** Even a crude method with no learning
worth speaking of recovers 84–98% of the variance in the parameters. The
information is unambiguously present in the data. The estimation problem is
genuinely solvable, and the network is not inventing anything.

For comparison, the trained BayesFlow network achieves recovery correlations of
0.941 / 0.959 / 0.997, which correspond to roughly **0.89 / 0.92 / 0.99** in
the same R²-like units. So:

| Parameter | Nearest-neighbour on 7 stats | Full trained network | Gain from the network |
|---|---:|---:|---:|
| `nu` | 0.844 | ~0.885 | +0.04 |
| `r` | 0.858 | ~0.919 | +0.06 |
| `mu_T` | 0.978 | ~0.994 | +0.02 |

*(These are measured slightly differently, so treat the gap as indicative
rather than exact.)*

**How to interpret this honestly, and it is an important admission:** the
seven hand-crafted statistics are doing **most** of the work. The neural
network — the LSTM summary network reading the raw sequence, plus the spline
coupling flow — adds a real but *modest* improvement on top. What the network
genuinely adds that the nearest-neighbour method cannot is a **calibrated
posterior distribution**: honest uncertainty, verified by the SBC checks, not
just a point guess. That is the actual product here. But if someone claims "the
deep learning is what cracked this problem," the honest answer is: no, the
feature engineering did most of it, and the network's contribution is
principled uncertainty plus a modest accuracy gain.

## 12.3 Test 3 — Permuting the information (which statistic matters for which parameter?)

The most direct test of "does this input actually matter?" is to **destroy it
and see what breaks**. Take the test set, shuffle *one* statistic's column at
random across readers — so that statistic is still present and still has the
same overall distribution, but is now paired with the wrong reader, carrying no
real information — and measure how much predictive accuracy is lost. This is
called **permutation importance**, and it is exactly the "permute the info"
check.

Measured drop in R² when each statistic is scrambled:

**For `nu` (baseline R² = 0.844):**

| Statistic scrambled | R² lost |
|---|---:|
| skip rate | **0.919** |
| regression rate | 0.417 |
| refixation rate | 0.241 |
| saccade amplitude | 0.093 |

**For `r` (baseline R² = 0.858):**

| Statistic scrambled | R² lost |
|---|---:|
| refixation rate | **0.445** |
| regression rate | 0.169 |
| saccade amplitude | 0.162 |
| mean duration | 0.134 |

**For `mu_T` (baseline R² = 0.978):**

| Statistic scrambled | R² lost |
|---|---:|
| mean duration | **0.636** |
| std. dev. of duration | 0.355 |
| saccade amplitude | 0.003 |
| refixation rate | 0.003 |

*(A drop larger than the baseline R² simply means that scrambling that input
makes predictions worse than useless — actively misleading, because the
predictor is relying on it heavily.)*

What this establishes, and these are strong, defensible claims:

1. **`nu` depends overwhelmingly on the skip rate.** Remove it and the estimate
   collapses entirely. If you had to keep only one statistic to estimate `nu`,
   it would be the skip rate.
2. **`r` depends primarily on the refixation rate**, with the regression rate
   and saccade amplitude as secondary support. No single statistic is as
   dominant for `r` as skip rate is for `nu` — the evidence for `r` is spread
   across several measures.
3. **`mu_T` depends only on the two duration statistics.** Scrambling the
   scanpath statistics (saccade amplitude, refixation rate) costs 0.003 — i.e.
   nothing. **This is the decoupling result proven a third independent way**:
   the scanpath statistics carry literally no usable information about `mu_T`.
4. **Every one of the 7 statistics earns its place.** None is dead weight.

This analysis is also the ready-made answer to "why did you include those seven
statistics and not others?" — because each one demonstrably carries information
about at least one parameter, and removing any of them measurably hurts.

## 12.4 Test 4 — Posterior contraction (does the network use the information?)

Tests 1–3 show the information exists. Contraction shows the *trained network
actually uses it*. Contraction is `1 − (posterior variance ÷ prior variance)`:
how much narrower the answer became after seeing the data.

| Parameter | Contraction | Meaning |
|---|---:|---|
| `nu` | 0.866 | The posterior is ~87% narrower than the prior |
| `r` | 0.904 | ~90% narrower |
| `mu_T` | 0.990 | ~99% narrower |

A contraction near **0** would be the warning sign: it would mean the network
looked at the data, learned nothing, and simply handed the prior back. Nothing
here is anywhere near that. All three parameters are strongly informed by the
data.

## 12.5 Summary: four independent lines of evidence

| Test | What it rules out | Result |
|---|---|---|
| Statistic↔parameter correlations | "The parameters leave no trace in behaviour" | Ruled out — strong correlations for all three |
| Nearest-neighbour predictability | "The information is too weak/tangled to use" | Ruled out — R² 0.84–0.98 without any network |
| Permutation importance | "The model relies on the wrong things" | Ruled out — each parameter depends on the mechanistically correct statistics |
| Posterior contraction | "The network ignores the data" | Ruled out — 87–99% contraction |

If someone asks "how do you know your estimates mean anything?", these four
tests, in this order, are the answer.

---

# PART 13 — Overfitting: Why It Is an Unusual Question Here, and What We Found

"Does it overfit?" is the standard question to ask of any machine-learning
project, and it deserves a careful answer. In this project the answer is
genuinely interesting, because the usual way of overfitting is **structurally
impossible here** — while a *different*, less obvious risk does exist.

## 13.1 What overfitting means, in plain words

**Overfitting** (literal sense: fitting *too* closely) is when a model
memorises the specific examples it was trained on — including their random
noise — instead of learning the general pattern. The symptom is always the
same: excellent performance on training data, poor performance on anything new.
The classic analogy is a student who memorises the answers to last year's exam
paper instead of learning the subject: perfect on that paper, lost on this
year's.

## 13.2 Why the usual kind of overfitting cannot happen here

In ordinary machine learning, training data is scarce and fixed: you have
10,000 photos, and that is all you will ever have, so the network sees each one
many times and can memorise them.

Here, **training data is generated on demand and is effectively unlimited.**
The simulator can produce a brand-new reader, with a brand-new random parameter
draw and brand-new random behaviour, as many times as we like. Three
consequences follow:

1. **Every validation reader is genuinely new.** The 300 readers used for the
   recovery/SBC diagnostics are freshly simulated at diagnostic time
   (`simulator.sample(n_val)` in `run_builtin_diagnostics`). They were *not*
   held out of a fixed dataset — they did not exist during training. The
   network cannot have memorised them.
2. **There is no shortage of data to overfit against.** With `--mode online`,
   the network would see every example exactly once and overfitting would be
   impossible by construction. Our normal route (`--mode train`, which calls
   `train_offline`) re-uses the 8,000 saved readers across 80 epochs, which
   *does* create some memorisation risk — see 13.4.
3. **The real data was never in training at all.** VP10's recording appears
   only at the very end. The network cannot overfit to data it has never seen.

## 13.3 An honest note about what the training code does *not* do

This is worth stating plainly, because a careful reader will notice it:

```python
history = workflow.fit_offline(data=data, epochs=n_epochs, batch_size=batch_size)
```

There is **no `validation_data` argument**. That means `training_loss.png`
shows the *training* loss only — there is no validation-loss curve on it. So
**the loss plot on its own cannot tell you whether the model overfits.** If you
are asked "how would you see overfitting in your loss curve?", the correct
answer is: *"we would not — our loss curve only shows training loss. We check
overfitting a different and stronger way."*

That stronger way is the diagnostics suite. In simulation-based inference, the
recovery / SBC / contraction checks on **300 freshly simulated readers** are a
better overfitting test than a validation-loss curve, because they measure the
thing we actually care about (are the posteriors accurate and honestly
calibrated on new data?) rather than a proxy loss number. A validation-loss
curve would still be a nice addition for the report, and adding
`validation_data` to `fit_offline` is the obvious small improvement.

## 13.4 The actual evidence that we are not overfitting

The decisive numbers, all computed on data the network never trained on:

| Evidence | Value | Why it rules out overfitting |
|---|---|---|
| Recovery on 300 fresh simulations | r = 0.941 / 0.959 / 0.997 | An overfitted network would do well on training data and *badly* here. It does well here. |
| 95% interval coverage | 95.0% / 96.0% / 95.3% | This is the clincher — see below. |
| SBC calibration plots | Within the confidence bands | Ranks are uniform, meaning no systematic over- or under-confidence. |
| PPC on held-out **test-half** sentences | Durations within ~6 ms | The final check uses sentences excluded from fitting entirely. |

**Why coverage is the strongest single piece of evidence.** Coverage asks: when
the network says "I am 95% confident the true value lies in this range," is the
true value actually inside that range 95% of the time? We measured 95.0%, 96.0%
and 95.3% — essentially perfect.

An overfitted network is characteristically **overconfident**: having
memorised its training examples, it believes it knows more than it does, and
produces intervals that are too narrow. That would show up immediately as
coverage well below 95% (say 70–80%), and as a U-shaped SBC histogram. We see
neither. The network's humility is calibrated correctly, which is very hard to
fake while overfitting.

There is one caveat worth stating for completeness: because training re-uses
8,000 saved readers for 80 epochs, mild memorisation is *possible* in
principle. The recovery and coverage numbers above show it is not happening to
any degree that matters. If you wanted to be certain, the cheap test is to
regenerate the training set with a different seed, retrain, and confirm the
diagnostics land in the same place.

## 13.5 The seven design choices that prevent overfitting

Worth being able to list, since each is a deliberate decision:

1. **Fresh simulations for validation** — diagnostics never re-use training
   examples.
2. **A train/test split on the real data** (`split_half`) — parameters are fit
   on VP10's first 57 sentences, and the PPC is run on the other 57.
3. **The real data is used exactly once**, at the end. No tuning against it.
4. **The prior comes from the paper**, not from inspecting our data.
5. **Fixed normalisation constants** (`WORDID_SCALE`, `DURATION_SCALE`) rather
   than statistics computed from the data — so no information leaks from the
   dataset into the preprocessing.
6. **A small network** (64-dimensional summary, 6 coupling layers) relative to
   8,000 training examples — limited capacity to memorise.
7. **The fixed constants were never hand-tuned to fit VP10** — `eta`, `alpha`,
   `beta` are all at the paper's published values.

Point 4 and point 7 are the ones people forget, and they are the ones that
would be most damaging if violated. Choosing a prior or tuning a constant after
looking at the answer is a subtle, serious form of overfitting that no
validation curve would ever catch.

## 13.6 The one genuine train/serve mismatch we did find

Being thorough means reporting what the checks *did* turn up. Comparing the
simulated training sequences against VP10's real sequences:

| | Simulated training readers | VP10's real observations |
|---|---|---|
| Mean fixations per observation | 123.3 | 106.8 |
| Longest observation | 150 (the cap) | 116 |
| Observations hitting the `SEQ_LEN = 150` cap | **2,338 of 8,000 (29%)** | **0 of 40** |

Nearly a third of training sequences are **truncated** at 150 fixations, losing
their tail — while no real VP10 observation ever comes close to the cap. So the
LSTM summary network is partly trained on chopped-off sequences of a kind it
never encounters at inference time.

**How much does this matter? Less than it first appears**, for a specific
reason worth understanding: the 7 hand-crafted statistics are computed from the
per-sentence arrays **before** padding and truncation
(`compute_reader_stats(rows)` in `run_one_reader`), so they are completely
unaffected. Since Part 12 showed the statistics carry most of the information,
the damage is limited to the LSTM's contribution — which is the smaller part.
This may in fact be *part of the reason* the LSTM contributes less than the
hand-crafted statistics do.

Truncation mostly affects slow readers (low `r`), who produce many fixations.
It is therefore a mild, systematic bias in the training distribution rather
than random noise. **The fix is straightforward and worth listing as future
work: raise `SEQ_LEN` from 150 to about 250**, which would cost a little memory
and nothing else. This is a real, actionable finding that came out of writing
this documentation.

## 13.7 What overfitting *would* have looked like

So you can recognise it if it ever appears:

- Recovery correlations high on training data but dropping sharply on fresh
  simulations.
- 95% coverage well below 95% (e.g. 70%) — overconfident intervals.
- A **U-shaped** SBC histogram (too many true values in the extreme tails).
- The SBC ECDF curve wandering outside its shaded band.
- Posteriors that are suspiciously narrow — near-zero width for parameters
  that should be uncertain.
- A PPC that matches on the train-half sentences but falls apart on the
  held-out test half.

None of these are present. The one clear mismatch we *do* have (regressions,
Part 16) has the opposite signature entirely: it fails on **both** simulated
and real data in the same way, which is the fingerprint of a **model
misspecification**, not overfitting.

---

# PART 14 — Convergence and Training Health

## 14.1 What "convergence" means

**Convergence** (literal meaning: coming together toward a point) is the point
during training when the network stops meaningfully improving. Training works
by repeatedly nudging the network's internal numbers to reduce the **loss** (a
single number measuring how wrong it currently is). Early on, each nudge helps
a lot. Eventually the improvements shrink to nothing and the loss flattens —
that is convergence. Training beyond that point mostly wastes time.

## 14.2 How to check convergence in this project

**Primary check — the loss curve** (`outputs/figures/training_loss.png`,
produced automatically by `_save_loss` after training). What you want to see:

- A **steep drop** over the first several epochs — the network rapidly learning
  the obvious structure (chiefly the `mu_T`↔duration relationship).
- A **gradual flattening** into a plateau.
- The final stretch (the last 10–20 epochs of the 80) **roughly level**, with
  only small random wobble.

If the curve is still dropping steeply at epoch 80, training stopped too early
and `--n_epochs` should be raised. If it drops and then rises, or oscillates
violently, something is wrong (usually the learning rate, or bad input scaling).

**A caveat you must state:** as covered in Part 13.3, this curve is
*training* loss only. A flat training loss proves the network stopped learning;
it does **not** by itself prove the network learned anything *useful*. For that
you need the second check.

**Secondary check — and the one that actually matters — the diagnostics.** A
properly converged, useful model shows:

| Signal | Converged and healthy | Not converged |
|---|---|---|
| Recovery correlation | High (ours: 0.94–1.00) | Low, points scattered off the diagonal |
| Contraction | High (ours: 0.87–0.99) | Near 0 — posterior ≈ prior |
| 95% coverage | ≈ 95% (ours: 95.0–96.0%) | Far from 95% |
| SBC ECDF | Inside the band | Wandering outside it |

By every one of these, our model has converged. This is the important point:
**convergence is not judged by the loss number alone, but by whether the
posteriors are accurate and calibrated on fresh data.**

## 14.3 The settings that control training, and what each one does

| Setting | Default | What it does | What happens if it is too low | Too high |
|---|---|---|---|---|
| `--n_readers` | 8000 | How many simulated readers to train on | Network sees too few examples; poor recovery, real overfitting risk | Slower generation; diminishing returns |
| `--n_epochs` | 80 | Complete passes over the training set | Under-trained: loss still falling, weak recovery | Wasted time; mild memorisation risk |
| `--batch_size` | 64 | Examples per weight update | Noisy, unstable training | Fewer updates per epoch; may need more epochs |
| `summary_dim` | 64 | Size of the LSTM's compressed summary | Not enough room to encode the sequence | More parameters, slower, more memorisation risk |
| `num_layers` | 6 | Coupling-flow layers | Cannot represent complex posterior shapes | Slower, harder to train |
| `M_SENTENCES` | 14 | Sentences per simulated reader | **Critical** — too few and `nu`/`r` become unidentifiable | Longer sequences, more truncation |
| `SEQ_LEN` | 150 | Max fixations kept per reader | Truncation (see 13.6) | More memory, mostly padding |

**The most important of these is `M_SENTENCES = 14`**, and it is worth
understanding why. A single sentence yields only ~8 fixations — far too little
to tell "this reader has a wide span" from "this sentence happened to contain
easy words." Watching the *same* reader across 14 sentences is what makes `nu`
and `r` identifiable at all. Reducing it to 1 would not make the network worse
at its job; it would make the job impossible. (The stale
`outputs/figures/baseline_M10/` folder is left over from an earlier
`M_SENTENCES = 10` configuration of the older model.)

**`bidirectional=False`** is also a deliberate choice. An LSTM can be set to
read the sequence both forward and backward; the project tested this and found
it roughly doubled training time with no accuracy gain, so it reads forward
only — in natural reading order, which is also the more principled choice for
a temporal process.

---

# PART 15 — Comparison With the Original Paper

## 15.1 Side by side

| Aspect | Engbert & Rabe (2024) | This project |
|---|---|---|
| Model | Simplified SWIFT, basic 3-parameter | Same — `nu`, `r`, `mu_T` |
| Inference method | Bayesian (the paper's own tutorial pipeline) | BayesFlow amortized neural SBI |
| `nu` recovery | Reported as the **hardest** parameter, with a right-skewed posterior | **Strong** (r = 0.941) |
| `r` recovery | Recovers reasonably | Strong (r = 0.959) |
| `mu_T` recovery | Recovers well (it is the duration mean) | Near-perfect (r = 0.997) |
| Decoupling (timing ⊥ scanpath) | Predicted by the model structure (Section 4.1) | **Confirmed on real data** (corr ≈ 0.007 / 0.061) |
| Train/test split | First-half sentences fit, second-half checked (Section 6) | Same |
| Regressions | Emergent, no suppression mechanism | Same structure — and we quantify the resulting error |

## 15.2 Where we do better than the paper: `nu`

This is the project's clearest methodological contribution and worth
highlighting in any write-up. The paper reports `nu` as its most difficult
parameter, with a skewed, poorly-constrained posterior. We recover it strongly
(r = 0.941, contraction 0.866).

**The reason is not that our network is better.** It is the **7 hand-crafted
summary statistics fed directly to the inference network as conditions**,
bypassing the LSTM. An LSTM reading a raw sequence of `[word_id, duration]`
pairs must *discover* for itself that "the proportion of word positions never
visited" is a meaningful quantity. That is a subtle, global, counting-based
property of a sequence, and LSTMs are not naturally good at it. Computing the
skip rate directly with three lines of NumPy and handing it over removes the
problem entirely — and Part 12.3 showed the skip rate is precisely what `nu`
depends on most.

The general lesson, which is a genuinely valuable finding to report: **when you
know which summary statistics are scientifically meaningful, providing them
directly can beat asking a network to learn them from raw data.**

## 15.3 Where we match the paper

Duration measures (SFD, GD, TT all within ~6 ms), skip and refixation rates
(within ~2 percentage points), and the decoupling prediction. On everything the
basic model is designed to capture, the fit is good.

## 15.4 Where we differ, and why

All four differences are documented in `RESULTS.md §6`, and all are honest
limitations rather than bugs:

1. **Regressions over-produced** (10% vs 2%) — see Part 16 for the full
   analysis. The paper's basic model has the same structure and the same
   weakness.
2. **Duration spread over-predicted.** Because `alpha` is fixed at 9, simulated
   durations *always* have a coefficient of variation of exactly 1/3 = 0.333.
   VP10's real durations are tighter (0.246). This is **structural**: no
   setting of any free parameter can change it, because `alpha` is not free. To
   fix it you would have to free `alpha`, which the basic model does not do.
3. **The "seconds" interpretation** in the activation update — the one place
   the implementation had to reconcile the paper's millisecond wording with its
   `r` values. Documented rather than silently assumed.
4. **`beta` fixed at 0.6** and `iota` not implemented — both belong to the
   paper's larger 5-parameter extended model.

## 15.5 "Could this be improved with more simulated data, or is it a summary-network problem?"

This is the right diagnostic question to ask when results disappoint, and this
project can answer it **with evidence** rather than guesswork. The answer
differs by symptom, and the reasoning below is the transferable part:

**If recovery were weak but coverage were correct** → an *information* problem.
More simulated data would not help. You would need better summary statistics or
a better summary network.

**If recovery were weak and coverage were also wrong (say 70%)** → a *training*
problem. More data and more epochs would genuinely help.

**If recovery is strong but the PPC still misses** → a **model** problem. The
inference is working correctly; the simulator itself cannot produce the
behaviour. Neither more data nor a better network can fix this.

**Our situation is unambiguously the third case.** Recovery is 0.94–1.00,
coverage is 95–96%, contraction is 0.87–0.99 — the inference machinery is
working essentially as well as it can. Yet the PPC misses regressions by a
factor of five. Therefore:

| Proposed fix | Would it help the regression gap? | Why |
|---|---|---|
| Generate 50,000 readers instead of 8,000 | **No** | Recovery is already ~0.95; more data cannot fix a simulator that does not produce the behaviour |
| A bigger/bidirectional LSTM | **No** | Same reason — this is not an information-extraction failure |
| More training epochs | **No** | The model has converged |
| More hand-crafted statistics | **No** | Regression rate is already provided directly |
| **Change the model** (add regression suppression) | **Yes** | This is the actual cause |
| Raise `SEQ_LEN` to 250 | Marginally, for other things | Fixes the truncation issue in 13.6, unrelated to regressions |

Part 16 proves this claim rather than merely asserting it.

---

# PART 16 — The Regression Problem: A Worked Analysis

This is the project's most interesting scientific finding and the part most
worth understanding in depth, because it demonstrates the difference between
"our method failed" and "we learned something about the model."

## 16.1 The finding

From the posterior predictive check on held-out sentences:

| Measure | Real VP10 | Simulated from our estimate |
|---|---:|---:|
| Mean SFD | 202.52 ms | 204.38 ms ✓ |
| Mean GD | 213.64 ms | 212.34 ms ✓ |
| Mean TT | 214.38 ms | 220.19 ms ✓ |
| P(skip) | 20.55% | 18.06% ✓ |
| P(refixation) | 9.09% | 7.71% ✓ |
| **P(regression)** | **1.82%** | **10.11%** ✗ |

Five of six measures match well. Regressions are wrong by roughly a factor of
five — the model has VP10 jumping backward far more often than they really do.

## 16.2 Why it happens, mechanically

In this model, regressions are **emergent** — there is no "regression dial." A
backward jump happens when a word to the *left* of the current one has drifted
into the half-processed state where the sine-saliency rule makes it maximally
attractive. Since the processing span always includes the word to the left
(weight `lambda_-1 = sigma·nu`), earlier words are continuously being topped up
and continuously becoming candidates to jump back to.

Critically, **there is nothing in the model that suppresses this.** Real
readers have strong forward momentum: reading is a directed process, and going
backward is the exception, triggered by comprehension difficulty. The basic
model has no such directional preference — its target rule is purely "whichever
word is most half-processed," with no bias toward moving forward. So it
regresses whenever the arithmetic happens to favour a leftward word.

## 16.3 Proof that this is *not* an inference failure

This is the important part, and the project can demonstrate it directly rather
than argue it. Two pieces of evidence:

**Evidence 1 — VP10's regression rate sits in the extreme tail of what the
model can produce.** Comparing VP10's real statistics against the distribution
of the 8,000 simulated training readers (a **prior predictive check** — does
the model, across its entire prior range, produce behaviour like this person's
at all?):

| Statistic | VP10's value | Percentile within the simulated readers |
|---|---:|---:|
| Mean duration | 0.197 | 32% ✓ comfortably central |
| Std. dev. of duration | 0.049 | 16% ✓ |
| Fixations per sentence | 0.762 | 28% ✓ |
| Skip rate | 0.187 | 47% ✓ |
| Refixation rate | 0.114 | 57% ✓ |
| Saccade amplitude | 0.220 | 5% ⚠ low tail |
| **Regression rate** | **0.015** | **2%** ⚠⚠ **extreme low tail** |

Only **161 of 8,000** simulated readers (2%) regress as little as VP10 does.
The model *can* produce such a reader, but only barely, and only at unusual
parameter settings. On the other five statistics, VP10 is a perfectly ordinary
member of the simulated population.

**Evidence 2 — the model cannot match regressions and refixations at the same
time.** This is the decisive test. Take the simulated readers that closely match
VP10 on skip rate and refixation rate, and ask what regression rate they
produce. Then take the readers that match VP10's low regression rate, and ask
what *their* other statistics look like:

| Group | Skip rate | Refixation rate | Regression rate |
|---|---:|---:|---:|
| **VP10 (real)** | **0.187** | **0.114** | **0.015** |
| Simulated readers matching VP10 on skip + refixation (n = 191) | 0.187 ✓ | 0.114 ✓ | **0.130** ✗ (9× too high) |
| Simulated readers matching VP10's low regressions (n = 422) | 0.252 ✗ | **0.013** ✗ (9× too low) | 0.018 ✓ |

**Read that table carefully — it is the core result.** There is *no* parameter
setting that gets all three right:

- Match the refixation and skip rates, and regressions come out 9× too high.
- Force regressions down to VP10's level, and refixations collapse to 9× too
  low.

The parameters that produce low regressions are `nu ≈ 0.31`, `r ≈ 9.03` — a
narrow span and a fast rate — but a fast rate means words finish in one look,
which destroys the refixations VP10 actually shows.

**This is model misspecification, proven.** The simulator's structure cannot
reproduce the combination of behaviours a real human exhibits. No amount of
extra training data, network capacity, or training time can fix it, because the
failure is in the forward model, before inference is ever involved.

> Reproduce both tables with `python tools/analyse_information.py` (sections
> `[3/4]` and `[4/4]`). It needs no trained model, so this evidence stands
> independently of whichever `.keras` file happens to be on disk.

## 16.4 The `r` trade-off, stated precisely

This is the specific question of "does improving one parameter make another
worse," and here is the concrete answer:

The fitted posterior sits at `nu = 0.40`, `r = 6.33` — a **compromise**,
sitting between the two incompatible regimes:

```
        LOW regressions                    HIGH refixations
        (matches VP10's 1.8%)              (matches VP10's 11.4%)
        nu ≈ 0.31, r ≈ 9.03                nu ≈ 0.52, r ≈ 5.56
              |                                    |
              |          FITTED: nu=0.40, r=6.33   |
              |------------------●-----------------|
                     the posterior lands between them
```

**Raising `r`** (faster processing) → words finish in one visit → fewer
refixations *and* fewer regressions. Good for regressions, bad for refixations.

**Lowering `r`** → more revisiting → more refixations *and* more regressions.
Good for refixations, bad for regressions.

Because `r` pushes both measures in the *same* direction, while VP10 needs them
to move in *opposite* directions (few regressions **but** many refixations),
`r` alone cannot satisfy both. The posterior settles between them, and the
fixed −0.317 correlation between `nu` and `r` is the visible fingerprint of the
network negotiating this trade-off.

This is precisely why the estimate is *not* wrong: given a model that cannot
express VP10's true behaviour, landing on the best available compromise is
correct behaviour for a Bayesian estimator.

## 16.5 What would actually fix it

1. **A forward-bias / regression-suppression term** in the target rule — the
   natural extension, deliberately *not* implemented here because it would
   double-count against the paper's emergent mechanism, and the assignment
   specified the simplified model.
2. **Inhibition of return** — a well-known effect in eye-movement research
   where recently visited locations become temporarily less attractive. This
   would directly damp regressions and is the mechanism most cognitive models
   use.
3. **The full SWIFT model**, which has additional machinery for exactly this.

## 16.6 How to describe this finding

The framing matters. The weak version is "our model didn't fit the regressions."
The strong, accurate version is:

> Our inference pipeline is verified accurate and well-calibrated (recovery
> 0.94–1.00, coverage 95–96%). Applying it to real data revealed that the
> *simplified SWIFT model itself* cannot simultaneously reproduce a human
> reader's low regression rate and high refixation rate — we demonstrated this
> directly by showing that simulated readers matching VP10 on skip and
> refixation rates over-regress ninefold, while those matching the regression
> rate under-refixate ninefold. This identifies a specific structural
> limitation of the basic model and points to regression suppression as the
> needed extension.

That is a finding, not a failure. Being able to say *why* a model fails, with
evidence, is a better outcome than a fit that works for reasons nobody checked.

---

# PART 17 — Every Plot, Precisely: What Is Compared to What

A plot can only be read correctly if you know **what is being compared to
what**. The same parameter can look good in one plot and bad in another because
the comparisons are different. This part makes the comparison explicit for every
figure.

## 17.1 The three kinds of comparison in this project

| Comparison type | What is on each side | What it can tell you | What it *cannot* tell you |
|---|---|---|---|
| **Simulated vs. known truth** | Network's estimate vs. the true parameters used to generate the data | Whether the *inference method* works | Nothing about whether the model describes real humans |
| **Real vs. simulated-from-estimate** | VP10's actual behaviour vs. behaviour simulated from the fitted parameters | Whether the *model* describes real reading | Nothing about whether inference is accurate |
| **Real data alone** | VP10's data, no model involved | Context and benchmark values | Nothing about the model at all |

**This distinction resolves the most common confusion in the whole project.**
`nu` looks excellent in `recovery_plot.png` and the regression bar looks bad in
`ppc_plot.png` — and both are true simultaneously, with no contradiction,
because they are answering different questions. Recovery says "if a reader
truly had this `nu`, we would correctly detect it." The PPC says "the model
with any `nu` cannot reproduce VP10's regressions." Good inference of a
model that is itself imperfect.

## 17.2 Every figure

| Plot | Comparison type | What is on the axes | Good looks like | Ours |
|---|---|---|---|---|
| `span_shape.png` | **No comparison** — simulator check | `nu` (x) vs the four span weights (y) | Matches the paper's Fig. 2 | ✓ verified by unit test |
| `scanpath_examples.png` | **No comparison** — simulator check | Fixation number (x) vs word position (y) | Mostly rising with occasional skips/refixations/regressions | ✓ plausible |
| `eda_fixations.png` | **Real data alone** | Three histograms: duration, signed saccade amplitude, fixations per sentence | No pass/fail — establishes benchmarks | 197 ms, 7.69 fix/sentence |
| `training_loss.png` | **Training only** | Epoch (x) vs training loss (y) | Falls then flattens | ✓ converged (training loss only — no validation curve) |
| `recovery_plot.png` | **Simulated vs. known truth** | True value (x) vs posterior mean (y), 300 fresh sims | Points hug the diagonal | ✓ 0.941 / 0.959 / 0.997 |
| `sbc_histogram.png` | **Simulated vs. known truth** | Rank of the true value among posterior draws | Flat/uniform bars | ✓ flat |
| `sbc_ecdf.png` | **Simulated vs. known truth** | Cumulative rank curve vs a confidence band | Curve stays inside the band | ✓ inside |
| `contraction_plot.png` | **Simulated vs. known truth** | How much the posterior narrowed vs the prior | High values | ✓ 0.87 / 0.90 / 0.99 |
| `posterior_VP10.png` | **Real data → estimate** (no truth exists) | Parameter value (x) vs density (y); orange posterior on grey prior | Narrow, clearly peaked, not jammed against an edge | ✓ all three well inside their priors |
| `posterior_correlation.png` | **Within the estimate** | 3×3 heatmap of parameter correlations | `mu_T` row/column ≈ 0 | ✓ 0.007 / 0.061 |
| `ppc_plot.png` | **Real vs. simulated-from-estimate** | Blue = real VP10, orange = simulated | Overlapping histograms, matching bars | ✓ durations; ✗ regression bar |
| `all_participants_ppc.png` | **Real vs. simulated-from-estimate**, once per person | Per-person real measure (x) vs simulated measure (y); 34 dots per panel | Dots hug the diagonal | ✓ durations/skip (r ≥ 0.92); ✗ regression dots sit above the line (r = 0.66) |
| `all_participants_ppc_pooled.png` | **Real vs. simulated-from-estimate**, pooled over people | Blue = all 34 people's real measures, orange = all simulated | Overlapping histograms, matching bars | ✓ durations; refixation under, regression over (Part 19.6) |
| `all_participants_posteriors.png` | **Real data → estimate** (no truth exists), 34 times | Parameter value (x) vs posterior density (y); one gray curve per person, blue = mean | Narrow, spread-out, interior curves | ✓ all three parameters (Part 19.7) |
| `all_participants_posterior_correlation.png` | **Within the estimates**, pooled over people | 3×3 heatmap over 680,000 pooled samples | Near 0 — but see the between-person caveat | `r`–`mu_T` −0.45: between-person leakage, not within-person coupling (Part 19.8) |
| `all_participants_theta_correlation.png` | **Between people** (individual differences — a finding, not a check) | 3×3 heatmap of the 34 point estimates | n/a | `r`–`mu_T` −0.53, `nu`–`r` +0.42: a "reading speed" axis (Part 19.8) |

## 17.3 How to read the four trickiest plots

**`recovery_plot.png`** — 300 points per panel, one per fresh simulated reader.
Horizontal = the true parameter we chose; vertical = what the network guessed.
A perfect network puts every point on the diagonal. Points *above* the line
mean over-estimation, *below* mean under-estimation. Look for two failure
shapes: a **flat cloud** (the network ignores the data and always guesses the
prior mean) and a **line with the wrong slope** (systematic bias — e.g.
shrinking every estimate toward the middle). Ours shows neither; `mu_T` is
almost a perfect line, `nu` and `r` are tight clouds around it.

**`sbc_histogram.png`** — the least intuitive plot, so here is the idea. For
each validation reader, ask: among the network's 1,000 posterior draws, where
does the *true* value rank? If the posterior is honest, the true value should
be equally likely to land anywhere in that ranking — so across 300 readers, the
histogram of ranks should be **flat**. The two diagnostic failure shapes:

- **U-shaped** (too many ranks at the extremes) = the posterior is **too
  narrow** → the network is **overconfident**. This is the classic overfitting
  signature.
- **Hump in the middle** = the posterior is **too wide** → the network is
  **underconfident** (harmless but wasteful).
- **Sloped** = systematic bias in one direction.

Ours is flat, matching the measured coverage of 95–96%.

**`posterior_VP10.png`** — grey band = the full prior range (everything we
considered possible beforehand); orange histogram = what we now believe after
seeing VP10's data. **The comparison is between the grey and the orange.** If
the orange filled the grey, the data taught us nothing. Ours occupies a narrow
slice — e.g. `mu_T` sits in 182–216 ms out of a possible 100–400 ms. Also check
that the orange is not pressed against either edge of the grey, which would
suggest the true value lies outside the prior; ours are all comfortably
interior.

**`ppc_plot.png`** — the only plot involving real and simulated data together.
Blue = VP10's real behaviour on the **held-out test sentences**; orange =
behaviour simulated from the fitted parameters. The three duration panels
should overlap heavily (they do). The fourth panel's three bar pairs should be
similar in height; ours match on skip and refixation and diverge sharply on
regression — the visual form of Part 16. Note that the orange histograms are
somewhat wider than the blue ones, which is the CV = 1/3 over-dispersion
described in Part 15.4 (item 2) showing up visually.

---

# PART 18 — Question-and-Answer Bank

Likely questions, with answers. If you can answer these, you understand the
project.

## About the project overall

**Q: What is this project in one sentence?**
It estimates, from a real person's eye-tracking recording, the three parameters
of a cognitive model of reading — using a neural network trained entirely on
simulated data, because the model has no likelihood formula.

**Q: Why not use standard Bayesian methods like MCMC?**
Those require a likelihood function — a formula for "how probable is this data
given these parameters." The SWIFT model is a stochastic simulator with no such
formula. Simulation-based inference exists precisely for this case: it replaces
the missing formula with a network trained on simulated examples.

**Q: What does "amortized" mean and why does it matter?**
The expensive training cost is paid once and spread over all future uses. After
training, a posterior for any new observation takes milliseconds. It also means
we could analyse a second participant instantly, with no retraining.

**Q: Why 14 sentences per reader?**
One sentence gives only ~8 fixations — too few to distinguish "this reader has a
wide span" from "this sentence had easy words." Concatenating 14 sentences from
the same simulated reader is what makes `nu` and `r` identifiable at all.

## About the parameters

**Q: Are the parameters influenced by the data you trained on?**
Three separate answers. The prior *ranges* come from the paper, not our data.
The network's *weights* were learned from simulated data only — never VP10's.
The final *estimate* for VP10 is driven by VP10's real fixations, which is
exactly what it should be. (Part 11.1.)

**Q: How are the parameters related to each other?**
`mu_T` is independent of the other two (measured correlation 0.007 and 0.061) —
it reads off durations while the others read off word positions, so they use
different data columns entirely. `nu` and `r` trade off moderately (−0.317)
because both control how quickly words get processed, so a similar amount of
total processing can be produced by different combinations.

**Q: Which parameter is most important?**
Depends on the question. `mu_T` is the most precisely estimated (contraction
0.990) but also the least interesting — it is essentially the mean fixation
duration. `nu` is the scientifically interesting one: the paper reports it as
hardest to estimate, and our hand-crafted statistics are what let us recover it
strongly. `r` matters most for the misfit story in Part 16.

**Q: Which is hardest to estimate, and why?**
`nu`, with the lowest recovery (0.941) and contraction (0.866). It is inferred
indirectly through skipping and jump lengths, and it partially trades off with
`r`. `mu_T` is easiest because it is the duration mean by construction.

**Q: Which column of the data determines which parameter?**
`mu_T` ← `fixation_duration` (column 4). `nu` and `r` ← `word_id` (column 2),
ordered by `fixation_index` (column 9). `nu` through skipping and jump length;
`r` through refixation and fixation counts. Everything else is unused. (Part
11.5.)

## About validity and overfitting

**Q: Does your model overfit?**
No, and we check it in a way that is stronger than a validation curve. The 300
validation readers are freshly simulated at diagnostic time, so they cannot have
been memorised. Recovery stays at 0.94–1.00 on them, and — the decisive
evidence — the 95% intervals achieve 95.0/96.0/95.3% coverage. Overfitted
networks are characteristically overconfident, which would show as coverage well
below 95% and a U-shaped SBC histogram. We see neither.

**Q: But your loss plot has no validation curve — how can you claim that?**
Correct, and worth stating openly: `fit_offline` is called without
`validation_data`, so `training_loss.png` shows training loss only. In
simulation-based inference the recovery/SBC/coverage diagnostics on fresh
simulations are the stronger overfitting test, because they measure posterior
accuracy and calibration directly rather than a proxy loss. Adding
`validation_data` is a sensible small improvement.

**Q: How do you validate at all, given the true parameters for a real human are
unknown?**
Two stages. On *simulated* data we know the truth, so we measure recovery,
calibration and contraction. On *real* data no truth exists, so we instead check
whether the fitted model reproduces VP10's behaviour — on a held-out half of
sentences never used for fitting.

**Q: Did you use the real data anywhere except the final step?**
No. The prior came from the paper; the fixed constants are the paper's published
values; the normalisation constants are fixed numbers, not data statistics.
VP10's recording enters only at the estimation step, and the PPC uses a
held-out half of it.

**Q: How do you know the data even contains information about these
parameters?**
Four independent checks (Part 12): statistic↔parameter correlations up to 0.99;
a nearest-neighbour predictor recovering R² = 0.84–0.98 with no network at all;
permutation importance showing each parameter depends on the mechanistically
correct statistics; and posterior contraction of 87–99%.

**Q: What is the permutation test and what did it show?**
Shuffle one summary statistic across readers so it carries no real information,
and measure how much predictive accuracy is lost. It showed `nu` depends
overwhelmingly on the skip rate, `r` on the refixation rate, and `mu_T` on the
two duration statistics only — scrambling the scanpath statistics costs `mu_T`
0.003, i.e. nothing, which independently confirms the decoupling.

## About the results

**Q: What did you find for VP10?**
`nu` = 0.40 [0.28, 0.54], `r` = 6.33 [5.27, 7.87], `mu_T` = 198 ms [182, 216].
A moderate processing span, a moderate-to-fast processing rate, and an average
fixation of about 198 ms.

**Q: `mu_T` = 198 ms and VP10's mean duration is 196.9 ms. Isn't that
circular?**
It is expected, not circular. In this model `E[fixation duration] = mu_T` by
construction, so a correct estimator *must* land there. It is a useful sanity
check — had it come out at 250 ms, something would be broken.

**Q: What went wrong, and is it your fault?**
Regressions: 10% simulated vs 1.8% real. It is not an inference failure. We
demonstrated that simulated readers matching VP10 on skip and refixation rates
regress ninefold too much, while those matching the regression rate refixate
ninefold too little — so *no* parameter setting fits both. That is a structural
limitation of the simplified model. (Part 16.)

**Q: Would more simulated data or a better network fix it?**
No to both. Those would help if recovery were weak or calibration were off, but
recovery is 0.94–1.00 and coverage is 95–96% — inference is working. The failure
is in the forward model, which more data cannot change. The fix is a regression
suppression / inhibition-of-return mechanism in the model itself.

**Q: Why is the simulated duration spread too wide?**
The Gamma shape constant `alpha` is fixed at 9, forcing a coefficient of
variation of exactly 1/3 = 0.333. VP10's real value is 0.246. No free parameter
can change this — it would require freeing `alpha`, which the basic model does
not do.

**Q: What is the single most important design decision in the project?**
Feeding 7 hand-crafted summary statistics directly to the inference network
alongside the LSTM. It is what lifts `nu` from the paper's hardest parameter to
strong recovery. Part 12.2 shows those statistics alone account for most of the
achievable accuracy.

**Q: If you had more time, what would you do?**
In priority order: (1) add regression suppression to the simulator and test
whether the PPC gap closes; (2) raise `SEQ_LEN` from 150 to 250 to remove the
29% training truncation; (3) add `validation_data` to `fit_offline` for a proper
validation curve; (4) free `alpha` to fix the duration spread; (5) fit multiple
participants and compare — the amortized network makes this nearly free.

---

# PART 19 — The All-Participants Extension: One Trained Network, 34 Real Readers

*(Added 2026-07-19. Everything in this part is produced by one command —
`python tools/all_participants_ppc.py` — which reuses the already-trained
network from `outputs/models/swift_approximator.keras`. Nothing is retrained.)*

## 19.1 What this extension is, and why it exists

Everything up to this point estimated the parameters of exactly **one**
person, VP10 — that was the course brief. But VP10 was never alone: the
original experiment that produced their recording had **34 participants**
(labelled VP1 through VP34), and the Engbert & Rabe (2024) paper itself fits
its model to *every* participant, summarizing the results in its Figures 8
and 9. This extension does the same thing with this project's pipeline: it
takes the trained network and asks it, one person at a time, "what are *this*
reader's `nu`, `r`, and `mu_T`?" — and then reality-checks each answer against
that person's own held-out data.

Two things make this extension worth doing, beyond completeness:

1. **It is the clearest possible demonstration of amortized inference**
   (Part 1.4's "train once, reuse forever" idea). The expensive work —
   simulating 8,000 artificial readers and training the network — was paid
   *once*. Estimating a 34th person costs the same few seconds as estimating
   the first. A classical method (like the MCMC sampling used in the paper)
   would have to restart its whole expensive computation from scratch for
   every single person.
2. **It is a genuine test of generalization.** The network was trained purely
   on simulated data and only ever *evaluated* against VP10. If anything in
   the pipeline had quietly become tuned to VP10's quirks, applying it to 33
   people it has never been near would expose that. (Spoiler: it generalizes
   well — see 19.5.)

## 19.2 The per-participant procedure, step by step

For each of the 34 files in `data/vp_all/` (described in Part 4.4), the
script repeats the same logic used for VP10 in Parts 10.3–10.4:

1. **Split the person's sentences in half.** The model is fit on the first
   half and tested on the held-out second half — the same train/test split
   idea as Part 10.4 ("held-out" literally means "kept out": data set aside
   and never shown to the fitting step, so the later check is honest).
2. **Estimate.** Build 20 random "reader draws" of 14 sentences each from the
   fitting half, hand each to the trained network, and draw 1,000 posterior
   samples per draw — 20,000 posterior samples per person. The **posterior
   mean** (the average of those samples) is used as the person's single
   point estimate for each parameter.
3. **Reality-check (posterior predictive check).** Draw 150 parameter
   settings from the person's posterior, run the SWIFT simulator with each on
   the *held-out* sentences, and compute six standard reading measures on
   both the real held-out data and the simulated data.

The six reading measures (computed by `_sentence_measures` in
`swift/diagnostics.py`, same as for VP10):

- **SFD — single-fixation duration:** how long the eye stayed on words that
  were fixated exactly once in the whole sentence.
- **GD — gaze duration:** the total duration of the eye's *first visit* to a
  word (including immediate refixations, but not later returns).
- **TT — total fixation time:** all fixation time a word ever received,
  including re-reading later.
- **P(skip), P(refixation), P(regression):** the three rate measures already
  used throughout this document.

One honest note on precision: the per-person sample sizes (20 reader draws ×
1,000 samples, 150 reality-check draws) are deliberately **half** of what the
VP10-only tools use (40 × 2,000, 300), because the whole procedure now runs 34
times. Individual participants' numbers are therefore slightly noisier than
the VP10 numbers in Part 10; what this extension is really after is the
*cross-participant* picture, which averages that noise out.

## 19.3 The headline numbers

The 34 estimated parameter values span a wide, plausible range:

| Parameter | Smallest estimate | Largest estimate | Average across the 34 |
|---|---|---|---|
| `nu` (processing span) | 0.12 (VP1) | 0.70 (VP7) | 0.35 |
| `r` (processing rate) | 2.9 (VP26) | 7.6 (VP32) | 4.7 |
| `mu_T` (mean fixation duration) | 199 ms (**VP10**) | 344 ms (VP5) | 239 ms |

Two details here are worth pausing on:

- **VP10 is the fastest fixator of all 34 people.** The one participant this
  project studied in depth has the *lowest* `mu_T` in the entire group —
  VP10 sits at the edge of the population, not in its middle. (VP10's
  regression rate, at ~2%, is similarly near the low extreme — which matters
  in 19.6.)
- **The estimates line up with the raw data exactly as Part 11 predicts.**
  The person with the lowest skip rate in the raw data (VP1, 4.6%) received
  the lowest `nu` — and skipping is `nu`'s main signal. The person with the
  highest refixation rate (VP26, 31%) received the lowest `r` — and
  refixation is `r`'s main signal. The network is visibly reading the right
  columns for the right dials, person by person.

And the headline result — how well each person's *simulated* reading matches
their *real* reading, across people (the correlations from
`outputs/cross_participant_correlations.json`, shown in the first figure
below):

| Reading measure | Cross-participant correlation (real vs. simulated) |
|---|---:|
| Single-fixation duration (SFD) | **0.98** |
| Gaze duration (GD) | **0.96** |
| Total fixation time (TT) | **0.95** |
| P(skip) | **0.92** |
| P(refixation) | 0.74 |
| P(regression) | 0.66 |

The quality ranking — durations ≈ perfect, skipping strong, refixation
moderate, regression weakest — is *exactly* the ranking that the information
analysis (Part 12) and the trade-off analysis (Part 16) predicted from VP10
alone. Seeing it hold across 34 independent people is strong confirmation
that those were genuine properties of the model, not quirks of one person.

## 19.4 Concepts you need for the figures (read this once, then the plots are easy)

The five figures below use four ideas that have not all been needed before:

- **Scatter plot with an identity line.** Each dot is one participant. Its
  horizontal position is the *real* value of some measure; its vertical
  position is the *simulated* value. The dashed diagonal is the **identity
  line** — the line where simulated = real. A dot exactly on the line means
  the fitted model reproduced that person's number perfectly; above the line
  means the model over-produces it, below means under-produces.
- **Correlation coefficient (`r`).** A single number between −1 and +1
  measuring how well one set of numbers *tracks* another: +1 means "when one
  goes up, the other goes up in perfect lockstep," 0 means "no relationship,"
  −1 means perfect opposite movement. Crucially, **correlation measures
  tracking, not agreement**: all 34 dots can sit well *above* the identity
  line (a systematic offset — the model over-produces the measure for
  everyone) while `r` is still high, because the *ordering* of people is
  preserved. Keep this distinction in mind for the regression panel.
- **Density curve / kernel density estimate (KDE).** A smooth version of a
  histogram. Instead of counting samples into bars, a KDE replaces each
  sample with a tiny smooth bump and adds the bumps up, giving a smooth curve
  whose height means "values around here are this likely." It shows the same
  information as a histogram, just easier to overlay 34 of them in one panel.
- **Pooled vs. per-person.** "Pooling" (literally: putting into one shared
  pool) means throwing all 34 people's numbers into one big bag and treating
  the bag as a single dataset. Pooling is useful for population-level
  questions but hides who contributed what — and, as 19.8 shows, it can
  manufacture correlations that exist for *no individual person*.

## 19.5 Figure 1 — `all_participants_ppc.png`: the headline figure

**What it is:** six scatter panels, one per reading measure. In each panel
there are 34 dots — one per participant. Horizontal axis: the person's *real*
value of the measure, computed on their held-out sentences. Vertical axis:
the value *simulated* from their fitted parameters on those same held-out
sentences. Dashed diagonal = identity line; the `r` in each title is the
correlation across the 34 dots. This is the project's equivalent of the
paper's Figure 9, produced with amortized BayesFlow inference instead of the
paper's MCMC.

**What each panel shows, in order:**

1. **Single-fixation duration, r = 0.98.** The dots hug the identity line
   from ~195 ms up to the one outlier at ~377 ms (VP5, the slowest reader —
   whose simulated value, ~356 ms, still tracks them almost perfectly). This
   panel says: for essentially every person, the fitted model reproduces
   *that person's* typical fixation duration. The flattering part is not the
   simulation (durations are `mu_T`'s job by construction, Part 5.1) — it is
   the *inference*: the network correctly found each individual's `mu_T`
   from their data alone.
2. **Gaze duration, r = 0.96** (and **3. Total fixation time, r = 0.95** —
   the same story). Still excellent, but look closely at the high end: the
   dots for the slowest readers drift *below* the line. GD and TT, unlike
   SFD, include refixations and re-reading — and the model under-produces
   refixations (panel 5), so it slightly under-produces the long gaze
   durations and total times of the readers who re-read the most.
4. **P(skip), r = 0.92.** Strong tracking across a wide real range
   (4.6%–28.5%). Most dots sit slightly *below* the line: the model skips a
   little less than real people at every level, but faithfully preserves
   *who* skips more than whom.
5. **P(refixation), r = 0.74.** Moderate. The real range is huge (7%–31%),
   and the dots sit mostly below the line — systematic under-prediction,
   the mirror image of the regression problem (this is the refixation side
   of Part 16.4's trade-off).
6. **P(regression), r = 0.66 — the most interesting panel.** Nearly every
   dot is *above* the identity line. The simulated regression rates form a
   **floor**: no matter whether a person's real rate is 0.7% (VP25) or 17.5%
   (VP30), the simulation produces roughly 7–13%. This is Part 16's finding
   made visible person-by-person: the model's processing span *always*
   generates roughly 10% backward jumps as a side-effect and has no
   mechanism to go lower. Only the few genuinely high-regression readers
   (real rate ≥ 11%) land near the line — for them, reality happens to meet
   the model's floor. Note how the two halves of the correlation lesson from
   19.4 both apply here: `r` = 0.66 says ordering is partly preserved, while
   the position of the cloud above the line says the *level* is
   systematically wrong for most people.

**The one-sentence takeaway:** fitted per person, the model reproduces
*individual differences* in reading speed and skipping almost perfectly,
refixation moderately, and regressions only for people who regress a lot.

## 19.6 Figure 2 — `all_participants_ppc_pooled.png`: the population-level reality check

**What it is:** the same four-panel layout as VP10's `ppc_plot.png` (three
duration histograms plus a skip/refix/regression bar chart; blue = real,
orange = simulated), but with all 34 participants' held-out measures **pooled**
into one bag. The histograms are drawn as *densities* (bars scaled so their
total area is 1) so that the real and simulated groups can be compared fairly
even though they contain different numbers of values.

**What it shows:**

- **The three duration panels (SFD, GD, TT) overlap heavily.** The simulated
  (orange) distributions are slightly *wider* than the real (blue) ones —
  the same CV = 1/3 over-dispersion already diagnosed for VP10 in Part 15.4,
  now visible at population scale.
- **The bar panel:** skipping ≈16% real vs ≈14% simulated (small
  under-shoot); refixation ≈16% real vs ≈11% simulated (clear under-shoot);
  regression ≈6% real vs ≈11% simulated (clear over-shoot).

**The genuinely new nuance this figure adds:** for VP10, the regression
mismatch looked like a factor of **5** (2% real vs 10% simulated,
Part 10.4). At the population level it is a factor of **2** (≈6% vs ≈11%) —
because VP10's regression rate turns out to be unusually *low* for the group
(19.3). The model still over-regresses, but the single participant this
project happened to study made the problem look worse than it typically is.
Conversely, the refixation under-prediction — barely visible for VP10
(9.1% vs 7.7%) — is much clearer at population scale (≈16% vs ≈11%). Both
shifts are the two ends of Part 16.4's see-saw: the model trades refixations
against regressions and cannot match both at once.

## 19.7 Figure 3 — `all_participants_posteriors.png`: 34 posteriors on one canvas

**What it is:** three panels, one per parameter (`nu`, `r`, `mu_T`). The
horizontal axis of each panel spans exactly that parameter's **prior range**
(Part 5.2): 0–1 for `nu`, 0–12 for `r`, 100–400 ms for `mu_T`. Each **gray
curve** is one participant's posterior, drawn as a smooth density curve (the
KDE from 19.4). The **blue curve** is the average of the 34 gray curves.
This is the project's equivalent of the paper's Figure 8 (which shows the
same thing for its 5 parameters; ours has 3).

**The three things to check, and what ours shows:**

1. **Are the gray curves narrow relative to the axis?** Yes — every curve
   occupies a small slice of its prior range. Meaning: the data is
   informative about *every* person, not just VP10. (If a curve spanned the
   whole axis, the data would have taught the network nothing about that
   person — compare `posterior_VP10.png`'s gray-vs-orange logic in
   Part 17.3.)
2. **Do the peaks sit at *different* places for different people?** Yes —
   e.g. the `mu_T` peaks range from ~200 ms to ~340 ms. This is what real
   **individual differences** look like. It is also an important sanity
   check on the network itself: if all 34 curves peaked at the same spot,
   the network would likely be ignoring its input and returning one default
   answer for everyone.
3. **Is any curve pressed against the edge of the axis?** No — every
   posterior sits comfortably inside its prior range, so the priors were
   wide enough for the whole population, not just for VP10.

One readable detail: the clearly separated rightmost gray curve in the
`mu_T` panel, peaking near 340 ms, is VP5 — the slowest fixator (19.3). The
network identified this unusual reader confidently rather than dragging them
toward the group average.

## 19.8 Figures 4 and 5 — the two correlation heatmaps, and how not to confuse them

Both figures are 3×3 **correlation heatmaps**, the same format as VP10's
`posterior_correlation.png` (Part 9): each cell holds the correlation between
two parameters, colored red for positive and blue for negative; the diagonal
is always exactly 1 (everything correlates perfectly with itself), and the
matrix is symmetric. But the two figures answer **completely different
questions**, and telling them apart requires the trickiest concept in this
whole part.

**Figure 4 — `all_participants_posterior_correlation.png` ("pooled").** Take
*all* posterior samples of *all* people — 34 people × 20,000 samples =
680,000 rows — pool them into one table, and correlate the columns. Result:
`nu`–`r` = +0.28, `nu`–`mu_T` = −0.25, `r`–`mu_T` = **−0.45**.

**Figure 5 — `all_participants_theta_correlation.png` ("point estimates").**
Reduce each person to their three posterior-mean point estimates — a table
with only 34 rows, one per human — and correlate those. Result: `nu`–`r` =
+0.42, `nu`–`mu_T` = −0.29, `r`–`mu_T` = **−0.53**.

**The apparent contradiction.** Part 10.3 established — as a headline
finding! — that for VP10, `mu_T` is *uncorrelated* with `nu` and `r`
(≈ 0.007 / 0.061): the duration dial is decoupled from the scanpath dials,
exactly as the model's design predicts. So why does the pooled matrix now
show `r`–`mu_T` = −0.45? Did the decoupling break?

**Resolution: within-person vs. between-person correlation.** These are two
different quantities that merely share the word "correlation":

- **Within one person** (what Part 10.3 measured): across the posterior
  samples *of a single individual*, does believing in a higher `r` go along
  with believing in a lower `mu_T`? Answer: no, ≈ 0. The decoupling holds,
  for VP10 and within each of the 34 posteriors individually.
- **Between people** (what Figure 5 measures): do *people who have* high `r`
  tend to *be people who have* low `mu_T`? Answer: yes, −0.53. That is a
  fact about the population of humans, not about the model's wiring.

When you pool everything into one bag (Figure 4), each person's 20,000
samples form a small cloud around that person's own (`nu`, `r`, `mu_T`)
location — and those 34 clouds are *arranged* along the between-person
pattern. The pooled correlation therefore mostly re-measures Figure 5's
between-person arrangement (slightly diluted by the within-person clouds,
which is why every pooled value is a weaker version of its Figure 5
counterpart: −0.45 vs −0.53, +0.28 vs +0.42, −0.25 vs −0.29). Nothing about
the within-person decoupling changed.

An everyday analogy: within any one school class, shoe size and reading
ability are uncorrelated. Pool every class from age 6 to 18 into one dataset
and shoe size suddenly "predicts" reading ability strongly — not because feet
help you read, but because *both* grow with age *between* the groups you
pooled. Mixing groups manufactures a correlation that exists for no
individual group. (Statisticians file this under **ecological correlation**,
a close relative of Simpson's paradox.)

**So what is each heatmap actually for?**

- Figure 4 is the **decoupling check at scale**: had it shown something like
  `mu_T`–`nu` = −0.9, that could not be explained by individual differences
  and would flag a real problem in the model or network. Its actual mild
  values, fully accounted for by Figure 5, are a pass.
- Figure 5 is a **scientific finding about people** (the first result in
  this project that needed more than one participant to exist): faster
  processors (higher `r`) tend to have wider processing spans (`nu`, +0.42)
  and shorter fixations (`mu_T`, −0.53). In plain terms, a general "reading
  speed" dimension — people who are fast are fast in every sense at once.

**Honest caveats on Figure 5, before anyone over-quotes it:** n = 34 people
is small — a correlation of ±0.4 at n = 34 carries roughly ±0.3 of
uncertainty, so treat the exact values loosely. And a slice of the `nu`–`r`
value may come from estimation noise rather than real people-differences,
since those two parameters are also mildly correlated *within* each posterior
(+0.28 pooled). The sign pattern is a suggestive, plausible finding — not a
confirmed law.

## 19.9 The two machine-readable output files

- **`outputs/all_participants_results.json`** — a list of 34 records, one
  per participant: their id, number of fixations and sentences, the three
  posterior-mean estimates, and all six real-vs-simulated measure pairs.
  This is the raw data behind Figures 1 and 5, ready for any further
  analysis without re-running anything.
- **`outputs/cross_participant_correlations.json`** — just the six
  correlation values printed in Figure 1's panel titles (19.3's second
  table).

## 19.10 What this extension adds to the project's conclusions

1. **The network generalizes.** Trained purely on simulations, evaluated
   during development only against VP10, it produced sensible, informative,
   internally consistent estimates for 33 people it had never touched — the
   strongest evidence yet against any VP10-specific overfitting
   (complementing Part 13).
2. **The model's quality ranking is confirmed at population level.**
   Durations ≈ perfect, skipping strong, refixation moderate, regression
   weakest — exactly the ordering Parts 12 and 16 predicted from one person.
3. **VP10 in context.** VP10 is the fastest fixator of the 34 and among the
   least regression-prone; the infamous 5× regression gap of Part 10.4 is a
   2× gap for the population. The model still over-regresses — but less
   dramatically than the VP10-only view suggested.
4. **A genuinely new finding:** the parameters correlate *across people*
   (a "reading speed" dimension) while remaining decoupled *within* each
   person — a distinction (19.8) that only becomes visible, or even
   definable, once you estimate more than one reader.

---

# PART 20 — Glossary

An alphabetical collection of every technical term used in this document, in
one place.

- **Adapter (BayesFlow)** — a small configuration object that tells the
  BayesFlow library which piece of data plays which role (e.g., "this is
  what we're predicting," "this is raw data that needs summarizing first").
- **Amortized inference** — training a neural network once, using many
  simulated examples, so that afterward it can produce an answer for any new
  real dataset almost instantly, instead of having to redo an expensive
  calculation from scratch every single time. ("Amortize" = spread a cost out
  over many future uses.)
- **Batch size** — how many training examples a neural network looks at
  together, as one group, before updating its internal numbers once.
- **Bayesian statistics** — a branch of statistics built around updating a
  starting belief (the "prior") into an updated belief (the "posterior")
  once new data is observed, and explicitly representing uncertainty as a
  full range of plausible values rather than a single number.
- **BayesFlow** — the specific software library used in this project to
  build and train the neural network that performs simulation-based Bayesian
  inference.
- **Coefficient of variation** — a measure of how spread-out a set of
  numbers is, relative to its own average value (a distribution with a low
  coefficient of variation is tightly clustered close to its average; a high
  one is much more spread out).
- **Convergence** — the point during training when the network stops
  meaningfully improving and the loss flattens out. Judged here not by the loss
  alone but by whether recovery, calibration and contraction are good on fresh
  simulated data (Part 14).
- **Corpus** — in linguistics/reading research, a structured collection of
  text (here, 114 German sentences) along with information about each word
  in it (length, frequency).
- **Coupling flow** — the specific type of normalizing flow (see below) used
  as this project's "inference network."
- **Coverage (of a credible interval)** — how often the true value actually
  falls inside the interval the network claims. If a network's stated 95%
  intervals contain the truth 95% of the time, its confidence is honest.
  Measured here as 95.0 / 96.0 / 95.3% — the strongest single piece of evidence
  against overfitting (Part 13.4).
- **Credible interval** — the Bayesian-statistics equivalent of a
  confidence interval: a range of values that the posterior distribution
  says has a stated probability (e.g. 95%) of containing the true value.
- **DataFrame** — pandas' name for a table of data with named columns and
  numbered rows, similar to a spreadsheet.
- **Decoupling** — in this project, the finding that the timing side of the
  model (`mu_T`) and the scanpath side (`nu`, `r`) are independent: they read
  off different data columns and their estimates are uncorrelated (≈ 0.007 and
  0.061). Predicted by the paper's model structure and confirmed here on real
  data.
- **Emergent behaviour** — behaviour the model produces without any explicit
  rule for it. Here, skipping, refixation and regression all emerge from the
  single sine-saliency target rule; the model contains no "skip dial" or
  "regression dial."
- **Epoch** — one complete pass of a neural network through the entire
  training dataset.
- **Eccentricity** — how far away (in characters) a word is from wherever the
  eye is currently pointed. *(Belonged to the older full-SWIFT implementation;
  the current basic model has no spatial extent and does not use it.)*
- **Exponential distribution** — a standard mathematical distribution
  commonly used to model "how long until the next random event happens,"
  where short waits are common and long waits get progressively rarer.
  *(Used by the older Gillespie-based implementation; the current model draws
  durations from a Gamma distribution instead.)*
- **Gamma distribution** — a standard family of bell-ish, right-skewed
  distributions for positive quantities. Fixation durations here are Gamma
  draws with mean `mu_T` and shape `alpha = 9`.
- **Fixation** — a short pause of the eye on a word while reading; the basic
  unit of data in this whole project.
- **Foveal / fovea** — the small central part of the eye's retina
  responsible for sharp, detailed vision; "foveal" describes whatever is
  currently being directly looked at.
- **Gillespie algorithm** — a mathematical method (originally from
  chemistry) for simulating systems where several random events could each
  happen next, by repeatedly drawing "how long until the next event" and
  "which event happens" from the combined rates of every possibility.
  *(Used by the older full-SWIFT implementation. The current basic model does
  **not** use it — it steps forward one fixation at a time instead. The term
  is kept here because it appears in the paper and in this project's history.)*
- **Identifiability** — whether a parameter can, even in principle, be
  reliably pinned down by the kind of data available, as opposed to whether
  the code correctly implements the estimation method. A parameter can
  genuinely be hard to identify from limited real-world data without that
  being anyone's mistake.
- **Intractable (likelihood)** — describing a situation where there is no
  usable mathematical formula for a quantity (here, the probability of
  observing a given fixation sequence given some parameters) — only a
  simulator that can generate samples from it.
- **JSON** — "JavaScript Object Notation," a simple, widely-used plain-text
  format for storing structured data (numbers, text, lists, and
  name-value pairs) in a way that both humans and computer programs can
  easily read.
- **Landing position** — exactly where within a word (measured in
  characters, as a decimal) the eye lands during a fixation.
- **LSTM (Long Short-Term Memory)** — a well-known type of neural network
  specifically designed to process a sequence of items one at a time while
  remembering relevant earlier information as it goes; used here as the
  "summary network" that reads through the raw fixation sequence.
- **Likelihood** — in statistics, the probability of observing some specific
  data, given a particular parameter setting: `p(data | parameters)`.
- **Loss** — a single number produced during neural network training that
  measures how wrong the network's current guesses are; training works by
  repeatedly nudging the network to make this number smaller.
- **Model misspecification** — when the model itself is structurally incapable
  of producing the behaviour seen in the real data, regardless of parameter
  values. Distinct from an inference failure (where the model *could* fit but
  the estimation method fails). This project's regression gap is
  misspecification, proven in Part 16.3.
- **Multiprocessing** — running several separate copies of a program at the
  same time, typically on different CPU cores of the same computer, to
  finish independent pieces of work faster.
- **Overfitting** — when a model learns the specific training examples,
  including their noise, instead of the general pattern; it then performs well
  on training data and badly on anything new. Largely prevented here because
  training data is simulated on demand and validation uses freshly generated
  examples (Part 13).
- **Neural network** — a machine-learning model, loosely inspired by
  networks of connected brain neurons, that learns to recognize patterns and
  make predictions from many training examples rather than from an
  explicitly hand-written formula.
- **Normalizing flow** — a neural-network technique that learns a flexible,
  reversible mathematical transformation, turning a simple, well-understood
  starting distribution into a complicated, realistic-looking one (or vice
  versa).
- **NumPy** — the core Python library for fast, efficient numerical array
  computation, used throughout this project's numeric code.
- **OSF (Open Science Framework)** — a public website researchers use to
  freely share their research data, so other scientists can verify and build
  on published work.
- **Pandas** — the standard Python library for loading, manipulating, and
  analyzing spreadsheet-like tabular data.
- **Parafoveal** — describing vision that is near, but not exactly at, the
  point of direct fixation — your near-peripheral vision, capable of some
  processing but less precise than foveal (direct) vision.
- **Posterior** — the updated belief about parameter values *after* taking
  observed data into account, `p(parameters | data)` — the central thing
  this whole project exists to estimate.
- **Posterior contraction** — a measure of how much narrower (more
  confident) a posterior distribution is compared to the original prior
  range; near 1 means highly informative data, near 0 means the data barely
  changed anything.
- **Permutation importance** — a way of measuring how much a particular input
  matters, by scrambling that input across examples (so it keeps its
  distribution but loses its meaning) and measuring how much accuracy is lost.
  Used in Part 12.3 to show which summary statistic drives which parameter.
- **Posterior Predictive Check (PPC)** — simulating new data using estimated
  parameter values and checking whether that simulated data resembles the
  real data the parameters were estimated from, as a final end-to-end
  sanity check.
- **Prior** — the range/distribution of parameter values considered
  plausible before any data has been taken into account.
- **Prior predictive check** — the mirror image of a posterior predictive
  check, done *before* fitting: simulate across the whole prior range and ask
  whether the real data looks like anything the model can produce at all. Part
  16.3 uses this to show VP10's regression rate sits at the 2nd percentile of
  what the model can generate.
- **R² (coefficient of determination)** — a measure of how much of the
  variation in a quantity is explained by a prediction. 1.0 = perfectly
  predicted, 0.0 = no better than guessing the average.
- **PyTorch** — the underlying numerical computation engine (a
  "machine-learning backend") that actually performs the neural network math
  in this project.
- **Recovery (parameter recovery)** — how well a trained network's average
  guess tracks the actual true parameter value, measured across many
  simulated validation examples where the true answer is already known.
- **Refixation** — a second (or later) fixation landing on a word that was
  just fixated, before the eye moves on to a new word.
- **Regression** — a backward eye-jump, to re-read something earlier in a
  sentence.
- **Saccade** — the fast, essentially "blind" jump the eye makes between two
  fixations.
- **SBC (Simulation-Based Calibration)** — a specific, formal statistical
  diagnostic (from Talts et al., 2018) that checks whether a trained
  network's stated confidence/uncertainty levels are actually honest.
- **Simulation-Based Inference (SBI)** — the broader family of statistical
  methods (of which BayesFlow is one implementation) for estimating unknown
  parameters using only a working simulator, when no usable likelihood
  formula exists.
- **Skip rate / Skipping** — the fraction of words that receive no fixation
  at all as the reader's eyes pass over the sentence.
- **Standard deviation** — a standard statistical measure of how spread out
  a set of numbers is, expressed in the same units as the original numbers
  (e.g., "the average duration was 197 ms, plus or minus 48 ms").
- **Spearman correlation** — a measure of whether two quantities move together,
  based on their rank order rather than their raw values. Robust to curved
  (non-straight-line) relationships, which is why it is used in Part 12.1.
- **Stochastic** — involving randomness/chance, as opposed to being fully
  predetermined.
- **Train/test split** — dividing data so that a model is fitted on one portion
  and evaluated on another it has never seen. Here, VP10's first 57 sentences
  are used to estimate parameters and the remaining 57 only for the posterior
  predictive check (`split_half`).
- **Truncation (of a sequence)** — cutting a sequence off at a maximum length.
  Here, sequences longer than `SEQ_LEN = 150` fixations lose their tail; this
  affects 29% of simulated training readers but no real VP10 observations
  (Part 13.6).
- **Summary network** — the part of the neural network (here, an LSTM) that
  compresses a variable-length raw fixation sequence down into one
  fixed-size vector of numbers.
- **SWIFT model** — the specific cognitive/computational model of eye
  movements during reading that this project simulates and fits, originally
  from Engbert & Rabe (2024) and earlier related work.
- **Uniform distribution/prior** — a distribution where every value within a
  given range is treated as equally likely, like a fair die roll where every
  face has an equal chance.

---

# PART 21 — Troubleshooting / FAQ

**"ModuleNotFoundError: No module named 'bayesflow'" (or 'swift')** — make
sure you're using the correct Python environment where the project's
dependencies were installed, and if you're running a script directly from
inside the `tools/` folder rather than through `main.py`, always run it as
`python tools/show_results.py` from the project's root folder (the one
containing `main.py`), not from inside the `tools/` folder itself — the
scripts add the project root to their own search path automatically, but
only if you invoke them from the right starting location.

**MPS / `linalg_qr` errors on Apple Silicon Macs** — `swift/inference.py`
already forces computation onto the CPU specifically to avoid this (Apple's
MPS backend doesn't implement a linear-algebra operation the network's
spline component needs). If you see this error anyway, make sure
`swift.inference` was actually imported (and therefore had a chance to apply
its fix) before any other BayesFlow/PyTorch code runs, and that the
`KERAS_BACKEND=torch` environment variable is genuinely set.

**`ValueError: cannot find context for 'fork'` when running `--mode generate`
(Windows)** — this is a real, confirmed limitation, not a misconfiguration.
`swift/generate.py` creates its parallel worker processes with
`mp.get_context("fork")`, and **`fork` does not exist on Windows** (it is a
Unix-only way of creating processes). The saved `data/training_data.npz` in
this repository was generated on a Unix-like machine. Your options, in order of
effort:

1. **Use the existing `data/training_data.npz`** if it is present — generation
   only needs to happen once, and `--mode train` reads the saved file. This is
   why the recommended workflow separates the two steps.
2. **Change `"fork"` to `"spawn"`** in `swift/generate.py`. `spawn` works on
   Windows, but it re-imports the module in each worker, so the call must be
   protected by an `if __name__ == "__main__":` guard — `main.py` already has
   one, so this generally works.
3. **Run generation on macOS/Linux** (including WSL) and copy the resulting
   `.npz` file across. The file is plain NumPy data and is fully portable.

**"No training data at data/training_data.npz"** — run
`python main.py --mode generate` first (or `--mode all`, which does
generation and training together automatically). On Windows, see the `fork`
note directly above first.

**"No saved model at outputs/models/swift_approximator.keras"** — run
`python main.py --mode train` (or `--mode all`) first;
`tools/show_results.py` and `--mode infer` both require an already-trained
model to exist on disk.

> **⚠️ Note on the current state of this repository (checked 2026-07-18):**
> `outputs/models/swift_approximator.keras` is **not present**. The
> `outputs/models/` folder currently contains only older models
> (`swift_approximator_M14_old.keras`, plus `old-4param-no-iota/` and
> `old-custom-4param/` subfolders from the superseded 4-parameter model).
> This means **`tools/show_results.py` and `--mode infer` will fail right now**
> with the error above until a model is retrained. The `.keras` model file is
> deliberately gitignored, so this is expected after a fresh clone — but be
> aware that the numbers quoted throughout Part 10 come from
> `outputs/results_summary.json` (generated 2026-07-18 00:11, when a trained
> model did exist), not from a model currently on disk. Retrain with
> `python main.py --mode train` to reproduce them. Do **not** rename an `old-`
> model into place to make the scripts run: those were trained on a different
> parameter set and would silently produce meaningless results.

**The numbers move slightly every time I run `tools/show_results.py`** —
this is expected, not a bug. VP10 inference pools posteriors across
*randomly chosen* 14-sentence subsets each time, and the final reality check
re-simulates from *randomly drawn* posterior samples each time, so exact
numbers naturally shift by a small amount run to run. Pass `--seed` with a
fixed number for a fully reproducible result.

**Why is `data/training_data.npz` or `outputs/models/*.keras` missing right
after I first download/clone this project?** — both files are deliberately
excluded from the project's shared version-control history, because they are
large, easily regenerated, and specific to whatever machine created them.
Regenerate them locally with `--mode generate` and `--mode train`. In
contrast, the plot images under `outputs/figures/` *are* included in the
shared history, since those go directly into the written report.

---

# PART 22 — References

- Engbert, R., & Rabe, M. B. (2024). *A tutorial on Bayesian inference for
  dynamical modeling of eye-movement control during reading.* Journal of
  Mathematical Psychology, 119, 102843.
- Risse, S., & Seelig, S. (2019) — the boundary-paradigm reading experiment
  ("PB2") that produced the 34-participant eye-tracking dataset used in the
  all-participants extension (Part 19). Data obtained from the paper's own
  OSF repository: https://osf.io/8wrf6/, folder
  `R-Code-parameter-estimation-from-experimental-data/expdata/`. (Cited here
  as in `tools/all_participants_ppc.py`; the exact article title was not
  re-verified from within this project.)
- Rabe, M. B., Chandra, J., Krügel, A., Seelig, S. A., Vasishth, S., &
  Engbert, R. (2021). *A Bayesian approach to dynamical modeling of
  eye-movement control in reading of normal, mirrored, and scrambled texts.*
  Psychological Review, 128(3), 516–543.
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018).
  *Validating Bayesian Inference Algorithms with Simulation-Based
  Calibration.* arXiv:1804.06788.
- BayesFlow v2 documentation: https://bayesflow.org/v2.0.11/

---

*End of document.*
