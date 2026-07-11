"""
data.py
=======
Load and explore the two SWIFT data files, and turn real fixations into the
exact same (normalised, padded, multi-sentence) observation format the
simulator produces.

Files
  fixseqin_PB2expVP10.dat  - real eye-tracking data, participant VP10
  Rcorpus_PB2_revision.dat - per-word properties (length, frequency), input to
                             the simulator, tab-separated with a header row
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swift.config import (
    FIG_DIR, M_SENTENCES, N_STATS, SEQ_LEN,
    compute_reader_stats, normalise_sequence, pad_sequence,
)

# ---------------------------------------------------------------------------
# Column mapping (confirmed from Rabe et al. 2021)
# ---------------------------------------------------------------------------
FIXATION_COLUMNS = [
    "sentence_id",        # 1-114
    "word_id",            # fixated word position
    "landing_position",   # character offset within word (jittered decimal)
    "fixation_duration",  # milliseconds
    "word_length",        # length of fixated word (characters)
    "fixation_type",      # 1=first, 2=last, 0=middle
    "flag1",              # always 0
    "flag2",              # always 0
    "fixation_index",     # sequential count within sentence
    "participant_id",     # always 10 (VP10)
]


def load_fixations(path) -> pd.DataFrame:
    """Load the participant VP10 fixation-sequence file."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = FIXATION_COLUMNS
    print(f"[load_fixations] {len(df)} fixations across "
          f"{df['sentence_id'].nunique()} sentences")
    return df


def load_corpus(path) -> pd.DataFrame:
    """Load the Potsdam Benchmark 2 corpus (tab-separated, header row).
    Columns: sentID, nw, wordID, length, freq, code."""
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={
        "sentID": "sentence_id",
        "wordID": "word_id",
        "freq": "frequency",
    })
    print(f"[load_corpus] {len(df)} words across "
          f"{df['sentence_id'].nunique()} sentences")
    return df


def build_corpus_lists(corpus: pd.DataFrame):
    """Corpus DataFrame -> (word_lengths_list, word_freqs_list), one array per
    sentence, ordered by word position."""
    word_lengths_list, word_freqs_list = [], []
    for _, group in corpus.groupby("sentence_id"):
        group = group.sort_values("word_id")
        word_lengths_list.append(group["length"].values.astype(float))
        word_freqs_list.append(group["frequency"].values.astype(float))
    print(f"[build_corpus_lists] {len(word_lengths_list)} sentences built")
    return word_lengths_list, word_freqs_list


# ---------------------------------------------------------------------------
# Real fixations -> network observation (same format as run_one_reader)
# ---------------------------------------------------------------------------

def sentence_features(fix: pd.DataFrame, sentence_id: int) -> np.ndarray:
    """Raw (n_fix, 4) feature array for one sentence, ordered by fixation index:
    [word_id, landing_position, fixation_duration, word_length]."""
    sent = fix[fix["sentence_id"] == sentence_id].sort_values("fixation_index")
    return sent[["word_id", "landing_position",
                 "fixation_duration", "word_length"]].values.astype(np.float32)


def build_reader_observation(fix: pd.DataFrame,
                             sentence_ids: Sequence[int],
                             seq_len: int = SEQ_LEN):
    """Concatenate the fixations of several sentences into one observation:
    ``(sequence (seq_len, 4), stats (N_STATS,))`` -- the real-data counterpart
    of ``simulator.run_one_reader``."""
    rows = [sentence_features(fix, sid) for sid in sentence_ids]
    rows = [r for r in rows if len(r) > 0]
    raw = np.concatenate(rows, axis=0) if rows else np.zeros((0, 4), dtype=np.float32)
    seq = pad_sequence(normalise_sequence(raw), seq_len)
    stats = compute_reader_stats(rows)
    return seq, stats


def build_reader_batch(fix: pd.DataFrame,
                       m_sentences: int = M_SENTENCES,
                       n_readers: int = 200,
                       seq_len: int = SEQ_LEN,
                       rng: np.random.Generator | None = None):
    """Build ``n_readers`` observations, each a random draw of ``m_sentences``
    of VP10's sentences. Feeding several such draws and pooling the posteriors
    gives a stable participant-level estimate (replacing the earlier, invalid
    'average the raw sequences' step). Returns ``(seqs, stats)``."""
    if rng is None:
        rng = np.random.default_rng(0)
    all_ids = np.sort(fix["sentence_id"].unique())
    seqs = np.zeros((n_readers, seq_len, 4), dtype=np.float32)
    stats = np.zeros((n_readers, N_STATS), dtype=np.float32)
    for k in range(n_readers):
        chosen = rng.choice(all_ids, size=min(m_sentences, len(all_ids)),
                            replace=False)
        seqs[k], stats[k] = build_reader_observation(fix, chosen, seq_len)
    return seqs, stats


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def run_eda(fix: pd.DataFrame, save_dir=FIG_DIR) -> None:
    """Print summary statistics and plot fixation distributions (the
    benchmarks used later by the posterior predictive check)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== FIXATION DATA SUMMARY (VP10) =====")
    print(f"Total fixations       : {len(fix)}")
    print(f"Sentences             : {fix['sentence_id'].nunique()}")
    print(f"Duration mean +/- std : {fix['fixation_duration'].mean():.1f} +/- "
          f"{fix['fixation_duration'].std():.1f} ms")
    print(f"Duration range        : {fix['fixation_duration'].min()}-"
          f"{fix['fixation_duration'].max()} ms")
    print(f"Skipping rate         : {skip_rate(fix):.1%}")
    print(f"Refixation rate       : {refix_rate(fix):.1%}")
    print(f"Mean fixations/sent   : {fix.groupby('sentence_id').size().mean():.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("EDA - Participant VP10", fontsize=13)

    axes[0].hist(fix["fixation_duration"], bins=40, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Fixation Duration (ms)"); axes[0].set_ylabel("Count")
    axes[0].set_title("Fixation Duration")

    axes[1].hist(fix["landing_position"], bins=30, color="coral", edgecolor="white")
    axes[1].set_xlabel("Landing Position (chars)"); axes[1].set_ylabel("Count")
    axes[1].set_title("Landing Position")

    fps = fix.groupby("sentence_id").size()
    axes[2].hist(fps, bins=20, color="mediumseagreen", edgecolor="white")
    axes[2].set_xlabel("Fixations per Sentence"); axes[2].set_ylabel("Count")
    axes[2].set_title("Fixations per Sentence")

    plt.tight_layout()
    out = save_dir / "eda_fixations.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"EDA plot saved -> {out}")


def skip_rate(fix: pd.DataFrame) -> float:
    skips, total = 0, 0
    for _, g in fix.groupby("sentence_id"):
        ids = g["word_id"].values
        mx = ids.max()
        total += mx
        visited = set(ids)
        skips += sum(1 for w in range(1, mx + 1) if w not in visited)
    return skips / total if total > 0 else 0.0


def refix_rate(fix: pd.DataFrame) -> float:
    refix, total = 0, 0
    for _, g in fix.groupby("sentence_id"):
        words = g.sort_values("fixation_index")["word_id"].values
        total += len(words) - 1
        refix += sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1])
    return refix / total if total > 0 else 0.0


def saccade_amplitude(fix: pd.DataFrame) -> np.ndarray:
    """Per-saccade amplitude (|words moved|, signed distance dropped) between
    consecutive fixations within each sentence, pooled across all sentences.
    The classic movement-pattern statistic alongside duration/skip/refixation
    (short forward saccades dominate; occasional skips and regressions widen
    the distribution)."""
    amps = []
    for _, g in fix.groupby("sentence_id"):
        words = g.sort_values("fixation_index")["word_id"].values
        if len(words) > 1:
            amps.extend(np.abs(np.diff(words)).tolist())
    return np.array(amps, dtype=float)


def synthetic_corpus(n_sentences: int = 114) -> pd.DataFrame:
    """Synthetic corpus fallback for testing without the real corpus file."""
    rng = np.random.default_rng(42)
    rows = []
    for sid in range(1, n_sentences + 1):
        n_words = rng.integers(8, 14)
        for wid in range(1, n_words + 1):
            rows.append({"sentence_id": sid, "word_id": wid,
                         "length": int(rng.integers(3, 12)),
                         "frequency": int(rng.integers(10, 500))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from swift.config import FIXATION_PATH
    run_eda(load_fixations(FIXATION_PATH))
