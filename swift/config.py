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
N_FEATURES = 4          # [word_id, landing_position, duration, word_length]
N_STATS = 7             # hand-crafted summary statistics (see compute_reader_stats)

# ---------------------------------------------------------------------------
# Feature normalisation — applied to BOTH simulated and real fixations.
# Fixed scales (not per-sequence) so absolute magnitudes and skipping
# information survive; a per-sequence max would erase them.
# ---------------------------------------------------------------------------
WORDID_SCALE = 20.0     # word position within a sentence (max ~15 words)
LANDING_SCALE = 10.0    # character offset within a word
DURATION_SCALE = 1000.0 # milliseconds -> seconds
WORDLEN_SCALE = 10.0    # word length in characters

FEATURE_SCALES = np.array(
    [WORDID_SCALE, LANDING_SCALE, DURATION_SCALE, WORDLEN_SCALE],
    dtype=np.float32,
)


def normalise_sequence(fixations: np.ndarray) -> np.ndarray:
    """Normalise a raw fixation array of shape (n, 4) columns
    [word_id, landing_position, duration_ms, word_length] into the network's
    input scale. Used by the simulator *and* the real-data loader so training
    and inference see identical distributions."""
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
    per-sentence raw fixation arrays (columns [word_id, landing, duration_ms,
    word_length]). Fed to the inference network as DIRECT conditions, because an
    LSTM over the raw sequence does not reliably recover the skipping- and
    refixation-based signals that identify delta0 and R.

    Returns a length-N_STATS vector (roughly O(1)-scaled):
      [mean_dur_s, std_dur_s, mean_fix_per_sent/10, skip_rate, refix_rate,
       mean_saccade_amplitude/5, mean_landing/10]
    """
    durs, landings, counts, skips, refix, amps = [], [], [], [], [], []
    for arr in sentence_arrays:
        arr = np.asarray(arr, dtype=np.float32)
        if len(arr) == 0:
            continue
        w = arr[:, 0].astype(int)
        durs.extend(arr[:, 2].tolist())
        landings.extend(arr[:, 1].tolist())
        counts.append(len(arr))
        mx = int(w.max())
        if mx > 0:
            skips.append(sum(1 for k in range(1, mx + 1) if k not in set(w)) / mx)
        if len(w) > 1:
            refix.append(sum(1 for i in range(len(w) - 1) if w[i] == w[i + 1]) / (len(w) - 1))
            amps.append(float(np.mean(np.abs(np.diff(w)))))

    return np.array([
        np.mean(durs) / 1000.0 if durs else 0.0,
        np.std(durs) / 1000.0 if durs else 0.0,
        np.mean(counts) / 10.0 if counts else 0.0,
        np.mean(skips) if skips else 0.0,
        np.mean(refix) if refix else 0.0,
        np.mean(amps) / 5.0 if amps else 0.0,
        np.mean(landings) / 10.0 if landings else 0.0,
    ], dtype=np.float32)
