"""
load_data.py
============
Load and explore the two SWIFT project data files:
  1. fixseqin_PB2expVP10.dat  — real eye-tracking data (participant VP10)
  2. rcorpus file             — word properties per sentence

Usage:
    from data.load_data import load_fixations, load_corpus, run_eda
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------------------------
# Column mapping confirmed from Rabe et al. (2021)
# ---------------------------------------------------------------------------
FIXATION_COLUMNS = [
    "sentence_id",        # which sentence (1–114)
    "word_id",            # which word position was fixated
    "landing_position",   # character offset within word (decimal due to jitter)
    "fixation_duration",  # duration in milliseconds
    "word_length",        # length of fixated word in characters
    "fixation_type",      # 1=first fixation, 2=last fixation, 0=middle
    "flag1",              # always 0
    "flag2",              # always 0
    "fixation_index",     # sequential count within sentence
    "participant_id",     # always 10 for VP10
]


def load_fixations(path: str) -> pd.DataFrame:
    """Load the fixation sequence .dat file for participant VP10."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = FIXATION_COLUMNS
    print(f"[load_fixations] {len(df)} fixations across "
          f"{df['sentence_id'].nunique()} sentences")
    return df


def load_corpus(path: str) -> pd.DataFrame:
    """
    Load the Potsdam Benchmark 2 corpus file (Rcorpus_PB2_revision.dat).

    Confirmed columns (file has a header row):
      sentID  — sentence ID (1–114)
      nw      — number of words in sentence
      wordID  — word position in sentence
      length  — word length in characters
      freq    — word frequency (continuous)
      code    — word category code (0, 1, 2, 9)

    Renames to standard internal names: sentence_id, word_id, length, frequency.
    """
    df = pd.read_csv(path, sep="\t")

    # Rename to standard internal column names
    df = df.rename(columns={
        "sentID":  "sentence_id",
        "wordID":  "word_id",
        "freq":    "frequency",
    })

    print(f"[load_corpus] {len(df)} words across "
          f"{df['sentence_id'].nunique()} sentences")
    return df


def build_corpus_lists(corpus: pd.DataFrame):
    """
    Convert corpus DataFrame into lists indexed by sentence.

    Returns
    -------
    word_lengths_list : list of np.ndarray  (one per sentence)
    word_freqs_list   : list of np.ndarray  (one per sentence)
    """
    word_lengths_list, word_freqs_list = [], []
    for _, group in corpus.groupby("sentence_id"):
        group = group.sort_values("word_id")
        word_lengths_list.append(group["length"].values.astype(float))
        word_freqs_list.append(group["frequency"].values.astype(float))
    print(f"[build_corpus_lists] {len(word_lengths_list)} sentences built")
    return word_lengths_list, word_freqs_list


def run_eda(fix: pd.DataFrame, save_dir: str = "diagnostics") -> None:
    """
    Print key summary statistics and plot fixation distributions.
    These statistics become posterior predictive benchmarks later.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n===== FIXATION DATA SUMMARY (VP10) =====")
    print(f"Total fixations       : {len(fix)}")
    print(f"Sentences             : {fix['sentence_id'].nunique()}")
    print(f"Duration mean ± std   : "
          f"{fix['fixation_duration'].mean():.1f} ± "
          f"{fix['fixation_duration'].std():.1f} ms")
    print(f"Duration range        : "
          f"{fix['fixation_duration'].min()}–"
          f"{fix['fixation_duration'].max()} ms")
    print(f"Skipping rate         : {_skip_rate(fix):.1%}")
    print(f"Refixation rate       : {_refix_rate(fix):.1%}")
    print(f"Mean fixations/sent   : "
          f"{fix.groupby('sentence_id').size().mean():.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("EDA — Participant VP10", fontsize=13)

    axes[0].hist(fix["fixation_duration"], bins=40,
                 color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Fixation Duration (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Fixation Duration Distribution")

    axes[1].hist(fix["landing_position"], bins=30,
                 color="coral", edgecolor="white")
    axes[1].set_xlabel("Landing Position (character offset)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Landing Position Distribution")

    fps = fix.groupby("sentence_id").size()
    axes[2].hist(fps, bins=20, color="mediumseagreen", edgecolor="white")
    axes[2].set_xlabel("Fixations per Sentence")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Fixations per Sentence")

    plt.tight_layout()
    out = os.path.join(save_dir, "eda_fixations.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"EDA plot saved → {out}")


def fixations_to_array(fix: pd.DataFrame,
                       sentence_id: int,
                       seq_len: int = 15) -> np.ndarray:
    """
    Extract and normalise fixation sequence for one sentence.

    Returns
    -------
    np.ndarray of shape (seq_len, 3):
        col 0 : word_id          normalised by max word_id
        col 1 : landing_position normalised by 10
        col 2 : fixation_duration in seconds (ms / 1000)
    Padded with zeros if shorter than seq_len.
    """
    sent = (fix[fix["sentence_id"] == sentence_id]
            .sort_values("fixation_index"))
    arr = sent[["word_id", "landing_position",
                "fixation_duration"]].values.astype(np.float32)

    arr[:, 0] /= (arr[:, 0].max() + 1e-8)
    arr[:, 1] /= 10.0
    arr[:, 2] /= 1000.0

    padded = np.zeros((seq_len, 3), dtype=np.float32)
    n = min(len(arr), seq_len)
    padded[:n] = arr[:n]
    return padded


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _skip_rate(fix: pd.DataFrame) -> float:
    skips, total = 0, 0
    for _, g in fix.groupby("sentence_id"):
        ids = g["word_id"].values
        mx = ids.max()
        total += mx
        visited = set(ids)
        skips += sum(1 for w in range(1, mx + 1) if w not in visited)
    return skips / total if total > 0 else 0.0


def _refix_rate(fix: pd.DataFrame) -> float:
    refix, total = 0, 0
    for _, g in fix.groupby("sentence_id"):
        words = g.sort_values("fixation_index")["word_id"].values
        total += len(words) - 1
        refix += sum(1 for i in range(len(words) - 1)
                     if words[i] == words[i + 1])
    return refix / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Synthetic corpus fallback (when corpus file is not yet available)
# ---------------------------------------------------------------------------

def synthetic_corpus(n_sentences: int = 114) -> pd.DataFrame:
    """Generate a synthetic corpus for testing without the real corpus file."""
    rng = np.random.default_rng(42)
    rows = []
    for sid in range(1, n_sentences + 1):
        n_words = rng.integers(8, 14)
        for wid in range(1, n_words + 1):
            rows.append({
                "sentence_id": sid,
                "word_id":     wid,
                "word":        f"word_{wid}",
                "length":      int(rng.integers(3, 12)),
                "frequency":   int(rng.integers(10, 500)),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    fix = load_fixations("data/fixseqin_PB2expVP10.dat")
    run_eda(fix)
