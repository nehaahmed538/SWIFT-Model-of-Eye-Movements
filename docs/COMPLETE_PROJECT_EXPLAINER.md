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
**Glossary** near the end (Part 11) that collects every technical term in one
place, in case you want to look something up without hunting through the whole
document.

The document is organized into 13 parts:

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
11. Glossary
12. Troubleshooting / FAQ
13. References

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
python main.py --mode generate --n_readers 10000
```

What this does, in plain words: it runs the SWIFT simulator 10,000 times
(each time with a different random parameter setting), spreads that work
across all the CPU cores on your machine to go faster, and saves the results
to `data/training_data.npz`. This step typically takes a few minutes,
depending on how many CPU cores your machine has. You only need to do this
once — the results are saved to disk and reused by later steps.

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
python main.py --mode all --n_readers 10000
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

## 3.4 Summary table of every command

| Command | What it does | Rough time |
|---|---|---|
| `python main.py --mode generate --n_readers 10000` | Pre-generate simulated training examples, save to disk | A few minutes |
| `python main.py --mode train` | Train the network on the saved simulations, then run diagnostics + VP10 analysis + final check | Tens of minutes |
| `python main.py --mode all --n_readers 10000` | Steps above combined (generate, then train) | Generate + train time combined |
| `python main.py --mode infer` | Load an already-trained network; skip training; just run diagnostics + VP10 analysis + final check | A few minutes |
| `python main.py --mode online` | Train with the simulator running live in the loop (slow, rarely used) | Slow |
| `python tools/show_results.py` | Read-only report of whatever model is currently saved — no training | ~30 seconds |
| `python tools/show_results.py --quick` | Same as above but tiny sample sizes, for a fast smoke test | ~5 seconds |
| `python tools/calibrate.py` | Fast, model-free check of the simulator's average behavior vs. real VP10 data | A few seconds |

Every command in this table must be run from the project's root folder (the
folder containing `main.py`), because the code inside these scripts figures
out where the `data/` and `outputs/` folders are relative to that root.

---

# PART 4 — The Datasets: Every File and Every Column, Explained

This project uses exactly two input data files, both stored in the `data/`
folder. **Only one of them is "real data" in the sense that matters for the
scientific question** — the other is background information the simulator
needs, not something the project runs its statistical inference on.

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
| 4 | `fixation_duration` | How long the eye stayed on this word, measured in milliseconds. Typical values in this file range from roughly 80 to 400 ms. |
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
│   ├── __init__.py               (empty; just marks this folder as an
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
│   └── training_data.npz         generated by generate.py (Part 4.3)
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
│   └── results_summary.json      machine-readable snapshot written by
│                                  tools/show_results.py
│
├── tools/                     <- small standalone helper scripts
│   ├── calibrate.py               fast, model-free simulator sanity check
│   └── show_results.py            read-only report of the saved model
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
  and saves an image (`eda_fixations.png`) showing histograms (bar-chart-like
  plots showing how often each range of values occurs) of fixation duration
  and saccade amplitude, plus the skip/refixation/regression rates. These
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

### `saccade_amplitude(fix)`

- **File:** `swift/data.py`
- **Parameters:** `fix` — a fixation table.
- **What it does:** for every sentence, it measures — between each
  consecutive pair of fixations — how many word-positions the eye moved (the
  absolute difference between consecutive `word_id` values). It pools these
  measurements across all sentences and returns them as one long list of
  numbers. This is the classic "movement pattern" statistic: for most real
  readers, this list is dominated by small values (mostly 1 — the eye usually
  steps to the very next word), with occasional larger values from skips
  (jumping 2+ words forward) or regressions (a negative-signed jump backward,
  though this function only measures the *size* of the jump, not its
  direction).

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

## 8.1 `python main.py --mode generate --n_readers 10000`

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
   which internally calls `diagnostics.py: _seq_stats()` many times, plus
   `data.py: skip_rate()`, `data.py: refix_rate()`, and
   `data.py: saccade_amplitude()` on the real data, and finally
   `diagnostics.py: _panel()` and matplotlib saving.

## 8.3 `python main.py --mode all --n_readers 10000`

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

The `outputs/figures/baseline_M10/` subfolder contains stale plots from the
superseded 4-parameter full-SWIFT model — kept only as historical reference,
and **not** comparable to the current 3-parameter run.

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

---

# PART 11 — Glossary

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
- **Corpus** — in linguistics/reading research, a structured collection of
  text (here, 114 German sentences) along with information about each word
  in it (length, frequency).
- **Coupling flow** — the specific type of normalizing flow (see below) used
  as this project's "inference network."
- **Credible interval** — the Bayesian-statistics equivalent of a
  confidence interval: a range of values that the posterior distribution
  says has a stated probability (e.g. 95%) of containing the true value.
- **DataFrame** — pandas' name for a table of data with named columns and
  numbered rows, similar to a spreadsheet.
- **Epoch** — one complete pass of a neural network through the entire
  training dataset.
- **Eccentricity (in this project)** — how far away (in characters) a word
  is from wherever the simulated eye is currently pointed.
- **Exponential distribution** — a standard mathematical distribution
  commonly used to model "how long until the next random event happens,"
  where short waits are common and long waits get progressively rarer.
- **Fixation** — a short pause of the eye on a word while reading; the basic
  unit of data in this whole project.
- **Foveal / fovea** — the small central part of the eye's retina
  responsible for sharp, detailed vision; "foveal" describes whatever is
  currently being directly looked at.
- **Gillespie algorithm** — a mathematical method (originally from
  chemistry) for simulating systems where several random events could each
  happen next, by repeatedly drawing "how long until the next event" and
  "which event happens" from the combined rates of every possibility.
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
- **Multiprocessing** — running several separate copies of a program at the
  same time, typically on different CPU cores of the same computer, to
  finish independent pieces of work faster.
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
- **Posterior Predictive Check (PPC)** — simulating new data using estimated
  parameter values and checking whether that simulated data resembles the
  real data the parameters were estimated from, as a final end-to-end
  sanity check.
- **Prior** — the range/distribution of parameter values considered
  plausible before any data has been taken into account.
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
- **Stochastic** — involving randomness/chance, as opposed to being fully
  predetermined.
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

# PART 12 — Troubleshooting / FAQ

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

**"No training data at data/training_data.npz"** — run
`python main.py --mode generate` first (or `--mode all`, which does
generation and training together automatically).

**"No saved model at outputs/models/swift_approximator.keras"** — run
`python main.py --mode train` (or `--mode all`) first;
`tools/show_results.py` and `--mode infer` both require an already-trained
model to exist on disk.

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

# PART 13 — References

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

---

*End of document.*
