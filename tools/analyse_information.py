"""
tools/analyse_information.py
=============================
Reproduces the quantitative analyses in Parts 12 and 16 of
docs/COMPLETE_PROJECT_EXPLAINER.md.

Unlike tools/show_results.py, this needs **no trained model** — it works
directly from data/training_data.npz (the simulated training set, where the
true parameters are known) plus the real VP10 file. That makes it fast (a few
seconds) and always runnable, even after a fresh clone where the .keras model
has not been retrained yet.

It answers four questions:

  1. Does the data contain information about the parameters at all?
     -> statistic/parameter correlations, and how well a dumb nearest-neighbour
        predictor recovers each parameter from the 7 summary statistics.
  2. Which statistic drives which parameter?
     -> permutation importance: scramble one statistic, measure accuracy lost.
  3. Is VP10 a reader this model could plausibly produce?
     -> prior predictive check: where VP10 sits in the simulated distribution.
  4. Can the model match VP10's regression AND refixation rates at once?
     -> the trade-off analysis behind Part 16.

Run:
    python tools/analyse_information.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swift import config
from swift.data import build_corpus_lists, load_corpus, load_fixations  # noqa: F401
from swift.simulator import PARAM_NAMES, THETA_MAX, THETA_MIN

STAT_NAMES = [
    "mean_dur_s", "std_dur_s", "fix_per_sent/10",
    "skip_rate", "refix_rate", "regr_rate", "sacc_amp/5",
]

K_NEIGHBOURS = 15
N_TRAIN = 6000


def hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_training():
    if not config.TRAINING_DATA.exists():
        raise FileNotFoundError(
            f"No training data at {config.TRAINING_DATA}\n"
            f"Generate it first:  python main.py --mode generate")
    d = np.load(config.TRAINING_DATA)
    return d["thetas"].astype(np.float64), d["stats"].astype(np.float64), d["seqs"]


# ---------------------------------------------------------------------------
# 1. Correlations: does each parameter leave a fingerprint on behaviour?
# ---------------------------------------------------------------------------

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, implemented directly to avoid a scipy dependency."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def report_correlations(stats: np.ndarray, thetas: np.ndarray) -> None:
    hr("[1/4] Do the parameters leave a fingerprint on behaviour?"
       "  (Explainer Part 12.1)")
    print("Spearman rank correlation between each summary statistic and each")
    print("parameter, over the simulated readers (true parameters known).\n")
    print(f"{'statistic':<18}" + "".join(f"{n:>10}" for n in PARAM_NAMES))
    print("-" * (18 + 10 * len(PARAM_NAMES)))
    for i, sn in enumerate(STAT_NAMES):
        row = [spearman(stats[:, i], thetas[:, j]) for j in range(len(PARAM_NAMES))]
        print(f"{sn:<18}" + "".join(f"{v:>10.3f}" for v in row))
    print("\nRead: |corr| near 1 = that statistic tracks that parameter closely;")
    print("near 0 = it carries no marginal information about it.")


# ---------------------------------------------------------------------------
# 2 & 3. Predictability and permutation importance
# ---------------------------------------------------------------------------

def _knn_r2(Ztr, ytr, Zte, yte, j, k=K_NEIGHBOURS) -> float:
    """Out-of-sample R^2 of a k-nearest-neighbour predictor stats -> param j."""
    preds = np.empty(len(Zte))
    for i in range(0, len(Zte), 500):
        block = Zte[i:i + 500]
        dist = ((block[:, None, :] - Ztr[None, :, :]) ** 2).sum(-1)
        idx = np.argpartition(dist, k, axis=1)[:, :k]
        preds[i:i + 500] = ytr[idx, j].mean(1)
    ss_res = ((yte[:, j] - preds) ** 2).sum()
    ss_tot = ((yte[:, j] - yte[:, j].mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def report_information(stats: np.ndarray, thetas: np.ndarray, seed: int = 0) -> None:
    hr("[2/4] How much information is there, and which statistic carries it?"
       "  (Parts 12.2-12.3)")
    rs = np.random.RandomState(seed)
    perm = rs.permutation(len(stats))
    stats, thetas = stats[perm], thetas[perm]
    Xtr, Xte = stats[:N_TRAIN], stats[N_TRAIN:]
    ytr, yte = thetas[:N_TRAIN], thetas[N_TRAIN:]

    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd

    print(f"k-NN (k={K_NEIGHBOURS}) predicting each parameter from the 7 statistics.")
    print(f"Trained on {len(Ztr)} readers, tested on {len(Zte)} unseen readers.")
    print("No neural network involved -- this measures the information available,")
    print("not the quality of our model.\n")

    for j, name in enumerate(PARAM_NAMES):
        base = _knn_r2(Ztr, ytr, Zte, yte, j)
        print(f"{name}:  out-of-sample R^2 = {base:.3f}")
        imp = []
        for i in range(len(STAT_NAMES)):
            Zp = Zte.copy()
            np.random.RandomState(seed + 1).shuffle(Zp[:, i])
            imp.append((base - _knn_r2(Ztr, ytr, Zp, yte, j), STAT_NAMES[i]))
        print("   permutation importance (R^2 lost when that statistic is scrambled):")
        for v, s in sorted(imp, reverse=True):
            print(f"     {s:<18}{v:>8.3f}")
        print()
    print("Read: a large drop means that parameter genuinely depends on that")
    print("statistic. A drop near 0 means the statistic is irrelevant to it.")


# ---------------------------------------------------------------------------
# 4. Prior predictive check + the regression/refixation trade-off
# ---------------------------------------------------------------------------

def report_vp10_plausibility(stats: np.ndarray, thetas: np.ndarray) -> None:
    hr("[3/4] Is VP10 a reader this model could plausibly produce?"
       "  (Part 16.3, evidence 1)")
    fix = load_fixations(config.FIXATION_PATH)
    from swift.data import build_reader_batch, split_half
    train_ids, _ = split_half(fix)
    _, vp_stats = build_reader_batch(
        fix, m_sentences=config.M_SENTENCES, n_readers=40,
        rng=np.random.default_rng(0), sentence_ids=train_ids)
    vp = vp_stats.mean(0)

    print("\nWhere VP10's real behaviour sits within the simulated population:\n")
    print(f"{'statistic':<18}{'VP10':>10}{'percentile':>13}")
    print("-" * 41)
    for i, sn in enumerate(STAT_NAMES):
        pct = float((stats[:, i] < vp[i]).mean() * 100)
        flag = "  <-- TAIL" if pct < 5 or pct > 95 else ""
        print(f"{sn:<18}{vp[i]:>10.3f}{pct:>12.1f}%{flag}")
    print("\nRead: a percentile near 50 means the model produces such readers")
    print("routinely. Below 5 or above 95 means the real person sits in the tail")
    print("of what the model can generate at all -- a misspecification warning.")
    return vp


def report_tradeoff(stats: np.ndarray, thetas: np.ndarray, vp: np.ndarray) -> None:
    hr("[4/4] Can the model match VP10's regressions AND refixations at once?"
       "  (Part 16.3, evidence 2)")
    real = thetas * (THETA_MAX - THETA_MIN) + THETA_MIN
    i_skip, i_refix, i_regr = 3, 4, 5

    close = ((np.abs(stats[:, i_skip] - vp[i_skip]) < 0.04) &
             (np.abs(stats[:, i_refix] - vp[i_refix]) < 0.03))
    low_regr = stats[:, i_regr] <= 0.03

    print(f"\n{'group':<44}{'skip':>8}{'refix':>8}{'regr':>8}")
    print("-" * 68)
    print(f"{'VP10 (real)':<44}{vp[i_skip]:>8.3f}{vp[i_refix]:>8.3f}{vp[i_regr]:>8.3f}")
    print(f"{'all simulated readers':<44}"
          f"{stats[:, i_skip].mean():>8.3f}{stats[:, i_refix].mean():>8.3f}"
          f"{stats[:, i_regr].mean():>8.3f}")
    print(f"{f'matching VP10 on skip+refix (n={close.sum()})':<44}"
          f"{stats[close, i_skip].mean():>8.3f}{stats[close, i_refix].mean():>8.3f}"
          f"{stats[close, i_regr].mean():>8.3f}")
    print(f"{f'matching VP10 on low regressions (n={low_regr.sum()})':<44}"
          f"{stats[low_regr, i_skip].mean():>8.3f}"
          f"{stats[low_regr, i_refix].mean():>8.3f}"
          f"{stats[low_regr, i_regr].mean():>8.3f}")

    print("\nParameters of each group:")
    for label, mask in [("matching skip+refix", close),
                        ("matching low regressions", low_regr)]:
        vals = "  ".join(f"{n}={real[mask, j].mean():.2f}"
                         for j, n in enumerate(PARAM_NAMES))
        print(f"  {label:<28}{vals}")

    print("\nRead: if the two groups need different parameters, and neither")
    print("reproduces all of VP10's measures, the model structurally cannot fit")
    print("this reader -- which is a model limitation, not an inference failure.")


def main() -> None:
    print("=" * 70)
    print("SWIFT -- INFORMATION & MISSPECIFICATION ANALYSIS")
    print("(reproduces docs/COMPLETE_PROJECT_EXPLAINER.md Parts 12 and 16)")
    print("=" * 70)

    thetas, stats, seqs = load_training()
    print(f"\nLoaded {len(thetas):,} simulated readers from {config.TRAINING_DATA}")

    nonzero = (seqs[:, :, 0] > 0).sum(1)
    n_trunc = int((nonzero >= config.SEQ_LEN).sum())
    print(f"Sequence lengths: mean {nonzero.mean():.1f}, max {nonzero.max()}, "
          f"SEQ_LEN={config.SEQ_LEN}")
    print(f"Readers truncated at the cap: {n_trunc:,}/{len(nonzero):,} "
          f"({n_trunc / len(nonzero):.1%})   <- Explainer Part 13.6")

    report_correlations(stats, thetas)
    report_information(stats, thetas)
    vp = report_vp10_plausibility(stats, thetas)
    report_tradeoff(stats, thetas, vp)

    hr("DONE")
    print("Narrative interpretation of every number above:")
    print("  docs/COMPLETE_PROJECT_EXPLAINER.md  Parts 12 (information) and 16 (misfit)")


if __name__ == "__main__":
    main()
