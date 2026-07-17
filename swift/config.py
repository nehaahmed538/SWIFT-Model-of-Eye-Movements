"""
config.py
=========
Single source of truth for paths, pipeline hyper-parameters and the feature
normalisation used *identically* by the simulator and the real-data loader.

Why this file exists
--------------------
Earlier versions defined ``SEQ_LEN`` in ``main.py`` but also carried a stale
``seq_len=30`` default deep inside the training code, and the simulator and the
real-data loader normalised the fixation features differently (the simulator
divided the word index by the sentence length, the loader by the max fixated
index). Those silent mismatches are exactly the kind of thing that quietly
breaks amortised inference, so every constant now lives here and is imported
everywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (all resolved relative to the repo root, so scripts work from anywhere)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"

FIXATION_PATH = DATA_DIR / "fixseqin_PB2expVP10.dat"
CORPUS_PATH = DATA_DIR / "Rcorpus_PB2_revision.dat"
TRAINING_DATA = DATA_DIR / "training_data.npz"
MODEL_PATH = MODEL_DIR / "swift_approximator.keras"

for _d in (DATA_DIR, OUT_DIR, FIG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------
# A single "observation" is one reader (fixed theta) reading M_SENTENCES
# sentences. Their fixations are concatenated into one variable-length sequence
# and zero-padded to SEQ_LEN. Concatenating several sentences (rather than
# feeding a single ~8-fixation sequence) is what gives the summary network
# enough evidence to identify the word-length / processing-span parameters.
M_SENTENCES = 14        # sentences per reader / per training example
SEQ_LEN = 150           # max concatenated fixations (14 sentences * ~8 + buffer)
N_FEATURES = 2          # [word_id, duration_ms] -- the simplified model's observable
                        # f_i = (x_i, y_i) (Engbert & Rabe 2024, Section 4).
                        # Word length / landing position do not exist in this model.
N_STATS = 7             # hand-crafted summary statistics (see compute_reader_stats)

# ---------------------------------------------------------------------------
# Feature normalisation — applied to BOTH simulated and real fixations.
# Fixed scales (not per-sequence) so absolute magnitudes and skipping
# information survive; a per-sequence max would erase them.
# ---------------------------------------------------------------------------
WORDID_SCALE = 20.0     # word position within a sentence (max ~12 words)
DURATION_SCALE = 1000.0 # milliseconds -> seconds

FEATURE_SCALES = np.array([WORDID_SCALE, DURATION_SCALE], dtype=np.float32)


def normalise_sequence(fixations: np.ndarray) -> np.ndarray:
    """Normalise a raw fixation array of shape (n, 2) columns
    [word_id, duration_ms] into the network's input scale. Used by the
    simulator *and* the real-data loader so training and inference see
    identical distributions."""
    fixations = np.asarray(fixations, dtype=np.float32)
    if fixations.size == 0:
        return fixations.reshape(0, N_FEATURES)
    return fixations / FEATURE_SCALES


def pad_sequence(fixations: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Truncate/zero-pad a (n, N_FEATURES) array to (seq_len, N_FEATURES)."""
    out = np.zeros((seq_len, N_FEATURES), dtype=np.float32)
    if len(fixations) > 0:
        n = min(len(fixations), seq_len)
        out[:n] = fixations[:n]
    return out


def compute_reader_stats(sentence_arrays) -> np.ndarray:
    """Hand-crafted summary statistics for one reader, computed from a list of
    per-sentence raw fixation arrays (columns [word_id, duration_ms]). Fed to
    the inference network as DIRECT conditions, because a summary network over
    the raw sequence does not reliably recover the skip/refixation/regression
    signals that identify nu and r.

    Returns a length-N_STATS vector (roughly O(1)-scaled):
      [mean_dur_s, std_dur_s, mean_fix_per_sent/10, skip_rate, refix_rate,
       regression_rate, mean_abs_saccade_amplitude/5]
    Regression rate is the most direct observable signal about nu: lambda_-1 =
    sigma*nu is the model's only source of leftward activation.
    """
    durs, counts, skips, refix, regr, amps = [], [], [], [], [], []
    for arr in sentence_arrays:
        arr = np.asarray(arr, dtype=np.float32)
        if len(arr) == 0:
            continue
        w = arr[:, 0].astype(int)
        durs.extend(arr[:, 1].tolist())
        counts.append(len(arr))
        mx = int(w.max())
        if mx > 0:
            skips.append(sum(1 for k in range(1, mx + 1) if k not in set(w)) / mx)
        if len(w) > 1:
            n_sacc = len(w) - 1
            refix.append(sum(1 for i in range(n_sacc) if w[i] == w[i + 1]) / n_sacc)
            regr.append(sum(1 for i in range(n_sacc) if w[i + 1] < w[i]) / n_sacc)
            amps.append(float(np.mean(np.abs(np.diff(w)))))

    return np.array([
        np.mean(durs) / 1000.0 if durs else 0.0,
        np.std(durs) / 1000.0 if durs else 0.0,
        np.mean(counts) / 10.0 if counts else 0.0,
        np.mean(skips) if skips else 0.0,
        np.mean(refix) if refix else 0.0,
        np.mean(regr) if regr else 0.0,
        np.mean(amps) / 5.0 if amps else 0.0,
    ], dtype=np.float32)
