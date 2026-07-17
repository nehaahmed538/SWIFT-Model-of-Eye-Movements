"""
simulator.py
============
The **simplified** SWIFT model of Engbert & Rabe (2024), basic 3-parameter
version, as the forward model for amortised Bayesian inference.

Reference:
  Engbert, R., & Rabe, M. M. (2024). A tutorial on Bayesian inference for
  dynamical modeling of eye-movement control during reading.
  Journal of Mathematical Psychology, 119, 102843.

This is the *simplified* model (paper Section 2), NOT the full SWIFT: words sit
at discrete positions 1..N with no spatial extension, so there are no word
lengths, no landing positions, no Gillespie algorithm, and no random-walk
activation thresholds. Skipping, refixation and regression are **not** explicit
mechanisms — they emerge from the sine-saliency target rule (Eq. 8-9).

FREE PARAMETERS (inferred by BayesFlow) -- paper Section 5
  nu    - shape of the processing span (Eq. 1-2); how far processing spreads
  r     - overall processing rate (Eq. 3, 6, 7)
  mu_T  - mean saccade-timer interval in ms (Eq. 10-12); == E[fixation duration]

FIXED CONSTANTS (paper values -- see FIXED below)
  eta, alpha, beta

The temporal process (fixation duration ~ Gamma(alpha, alpha/mu_T)) and the
spatial process (which word is fixated next) are completely decoupled in the
basic model (paper Section 4.1): mu_T affects only durations; nu and r affect
only the scanpath.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from swift.config import (
    N_FEATURES, compute_reader_stats, normalise_sequence, pad_sequence,
)

# ---------------------------------------------------------------------------
# Fixed model constants (paper values, with source)
# ---------------------------------------------------------------------------
FIXED = {
    "eta":   1e-3,   # baseline saliency, keeps the target denominator > 0 (Section 3)
    "alpha": 9.0,    # Gamma shape of the saccade timer -> CV = 1/sqrt(9) = 1/3 (Section 2.4)
    "beta":  0.6,    # word-frequency effect on max activation (Section 5 recovery value;
                     # free only in the extended 5-parameter model). beta=0 (a_max=1 for
                     # all words, ignoring frequency) is the strict Section-3 baseline.
}

# Prior bounds for the 3 free parameters (paper Section 5, all uniform)
PRIOR_BOUNDS = {
    "nu":   (0.0,   1.0),
    "r":    (0.0,  12.0),
    "mu_T": (100.0, 400.0),   # ms
}

PARAM_NAMES = ["nu", "r", "mu_T"]

# Guard against the degenerate nu=0 / r=0 edges when sampling.
_EPS = 1e-4

# Pragmatic guard, not in the paper: cap fixations per sentence so a rare
# non-terminating scanpath can't run forever. If TRUNCATIONS climbs, something
# is wrong (see calibrate / the unit test).
MAX_FIX = 200
TRUNCATIONS = 0

THETA_MIN = np.array([PRIOR_BOUNDS[k][0] for k in PARAM_NAMES], dtype=np.float32)
THETA_MAX = np.array([PRIOR_BOUNDS[k][1] for k in PARAM_NAMES], dtype=np.float32)


# ---------------------------------------------------------------------------
# Prior sampler and (de)normalisation of theta
# ---------------------------------------------------------------------------

def sample_prior(rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Sample one set of free parameters uniformly from the prior bounds,
    clipped off the degenerate nu=0 / r=0 edges."""
    if rng is None:
        rng = np.random.default_rng()
    out = {}
    for k in PARAM_NAMES:
        lo, hi = PRIOR_BOUNDS[k]
        out[k] = float(rng.uniform(max(lo, _EPS) if k in ("nu", "r") else lo, hi))
    return out


def normalise_theta(params: Dict[str, float]) -> np.ndarray:
    """Parameter dict -> normalised [0, 1] array (training scale)."""
    theta = np.array([params[k] for k in PARAM_NAMES], dtype=np.float32)
    return (theta - THETA_MIN) / (THETA_MAX - THETA_MIN)


def denormalise_theta(theta_norm: np.ndarray) -> Dict[str, float]:
    """Normalised [0, 1] array -> parameter dict (original scale)."""
    theta = np.asarray(theta_norm, dtype=np.float32) * (THETA_MAX - THETA_MIN) + THETA_MIN
    return dict(zip(PARAM_NAMES, theta.tolist()))


# ---------------------------------------------------------------------------
# Processing span (Eq. 1-2)
# ---------------------------------------------------------------------------

def span_rates(k: int, N: int, nu: float) -> np.ndarray:
    """Per-word processing-rate weights lambda_w for gaze on word k (0-indexed).

    Eq. 1-2 of Engbert & Rabe (2024): the asymmetric 4-point span
        w = k-1 : sigma * nu
        w = k   : sigma            (fixated word)
        w = k+1 : sigma * nu
        w = k+2 : sigma * nu**2
    and 0 elsewhere, with sigma = 1 / (1 + 2*nu + nu**2). sigma is defined
    globally in Eq. 2 and is NOT renormalised at sentence boundaries.
    """
    sigma = 1.0 / (1.0 + 2.0 * nu + nu * nu)
    lam = np.zeros(N, dtype=float)
    if k - 1 >= 0:
        lam[k - 1] = sigma * nu
    lam[k] = sigma
    if k + 1 < N:
        lam[k + 1] = sigma * nu
    if k + 2 < N:
        lam[k + 2] = sigma * nu * nu
    return lam


# ---------------------------------------------------------------------------
# Core simulator: one sentence
# ---------------------------------------------------------------------------

def simulate_sentence(word_freqs: np.ndarray,
                      nu: float, r: float, mu_T: float,
                      beta: float = FIXED["beta"],
                      eta: float = FIXED["eta"],
                      alpha: float = FIXED["alpha"],
                      rng: Optional[np.random.Generator] = None,
                      max_fix: int = MAX_FIX
                      ) -> List[Tuple[int, float]]:
    """Simulate reading one sentence of ``len(word_freqs)`` words.

    Returns a list of fixations ``(word_id [1-indexed], duration_ms)`` in
    temporal order. Durations are pure Gamma draws (Eq. 10-12); the fixated
    word evolves via activation build-up (Eq. 7) and sine-saliency target
    selection (Eq. 8-9).
    """
    global TRUNCATIONS
    if rng is None:
        rng = np.random.default_rng()

    N = len(word_freqs)
    # Word difficulty via frequency (Eq. 4-5). beta=0 -> a_max == 1 for all words.
    F = np.clip(np.asarray(word_freqs, dtype=float), 1.0, None)
    logF = np.log10(F)
    q = logF / logF.max() if logF.max() > 0 else np.zeros(N)
    a_max = 1.0 - beta * q               # in (0, 1], since 0 < beta < 1

    a = np.zeros(N)
    k = 0                                # start on the first word (Section 3)
    fixations: List[Tuple[int, float]] = []

    while True:
        # --- temporal: this fixation's duration (Eq. 10-12), timer-only ---
        T = rng.gamma(shape=alpha, scale=mu_T / alpha)
        fixations.append((k + 1, float(T)))

        if k == N - 1:                   # "stop as soon as the last word is fixated"
            break
        if len(fixations) >= max_fix:    # safety guard (not in the paper)
            TRUNCATIONS += 1
            break

        # --- update activations over this fixation (closed form, Eq. 7) ---
        # The processing rate r (paper values ~5-10) is per SECOND; durations
        # are stored in ms, so convert. With T in ms the term r*lambda*T runs
        # into the hundreds, saturating every in-span word to a_max in a single
        # fixation -- which collapses the saliency rule (Eq. 8) and makes the
        # scanpath independent of nu and r. Seconds makes processing take a few
        # fixations, so refixation (low r) and span/skipping (nu) emerge as the
        # paper describes (Section 2.4, 3).
        lam = span_rates(k, N, nu)
        a = np.minimum(a + r * lam * (T / 1000.0), a_max)

        # --- spatial: sine saliency -> next target (Eq. 8-9) ---
        s = a_max * np.sin(np.pi * a / a_max) + eta
        p = s / s.sum()
        k = int(rng.choice(N, p=p))

    return fixations


# ---------------------------------------------------------------------------
# Thin class wrapper kept for the diagnostics / calibrate call sites
# ---------------------------------------------------------------------------

class SWIFTSimulator:
    """Simplified SWIFT model -- simulates one sentence at a time."""

    def __init__(self, params: Dict[str, float]):
        self.p = params

    def simulate_sentence(self, word_freqs: np.ndarray,
                          rng: Optional[np.random.Generator] = None
                          ) -> List[Tuple[int, float]]:
        return simulate_sentence(word_freqs, self.p["nu"], self.p["r"],
                                 self.p["mu_T"], rng=rng)


# ---------------------------------------------------------------------------
# Session builders used by the BayesFlow simulator functions
# ---------------------------------------------------------------------------

def simulate_one_sentence_features(params: Dict[str, float],
                                   word_freqs: np.ndarray,
                                   rng: np.random.Generator) -> np.ndarray:
    """One sentence -> raw (n_fix, 2) feature array [word_id, duration_ms]."""
    fixations = simulate_sentence(word_freqs, params["nu"], params["r"],
                                  params["mu_T"], rng=rng)
    if len(fixations) == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    return np.array(fixations, dtype=np.float32)


def run_one_reader(params: Dict[str, float],
                   word_freqs_list: list,
                   seq_len: int,
                   m_sentences: int,
                   rng: np.random.Generator):
    """Simulate one reader (fixed theta) reading ``m_sentences`` sentences.

    Returns ``(sequence, stats)``:
      * sequence : concatenated, normalised, zero-padded fixations
                   of shape (seq_len, N_FEATURES) -> summary network
      * stats    : hand-crafted summary statistics (N_STATS,) -> direct
                   conditions for the inference network

    Concatenating several sentences under one theta gives the network enough
    evidence to identify nu and r, which a single ~8-fixation sentence cannot.
    """
    n_sent = len(word_freqs_list)
    idx = rng.integers(0, n_sent, size=m_sentences)

    rows = [
        simulate_one_sentence_features(params, word_freqs_list[i], rng)
        for i in idx
    ]
    raw = np.concatenate(rows, axis=0) if any(len(r) for r in rows) \
        else np.zeros((0, N_FEATURES), dtype=np.float32)
    seq = pad_sequence(normalise_sequence(raw), seq_len)
    stats = compute_reader_stats(rows)
    return seq, stats


# ---------------------------------------------------------------------------
# Self-check: span table (Fig. 2) + timer moments
# ---------------------------------------------------------------------------

def _demo() -> None:
    # Span table against the paper's Fig. 2 (Section 3 reference values).
    ref = {
        0.1: (0.8264, 0.0826, 0.8264, 0.0826, 0.0083),
        0.2: (0.6944, 0.1389, 0.6944, 0.1389, 0.0278),
        0.3: (0.5917, 0.1775, 0.5917, 0.1775, 0.0533),
        0.4: (0.5102, 0.2041, 0.5102, 0.2041, 0.0816),
        0.6: (0.3906, 0.2344, 0.3906, 0.2344, 0.1406),
    }
    for nu, (sig, lm1, l0, lp1, lp2) in ref.items():
        lam = span_rates(3, 10, nu)          # interior word: full span present
        assert abs(lam.sum() - 1.0) < 1e-9, f"span sum != 1 at nu={nu}"
        got = (lam[2], lam[3], lam[4], lam[5])
        assert np.allclose(got, (lm1, l0, lp1, lp2), atol=1e-3), (nu, got)
    # Asymmetry: k-2 never processed, k+2 is.
    lam = span_rates(3, 10, 0.3)
    assert lam[1] == 0.0 and lam[5] > 0.0

    # Timer moments: mean ~ mu_T, CV ~ 1/3.
    rng = np.random.default_rng(0)
    freqs = np.full(8, 100.0)
    durs = [d for _ in range(400)
            for _, d in simulate_sentence(freqs, 0.3, 10.0, 200.0, rng=rng)]
    durs = np.array(durs)
    assert abs(durs.mean() - 200.0) < 8.0, durs.mean()
    assert abs(durs.std() / durs.mean() - 1 / 3) < 0.03, durs.std() / durs.mean()
    print("simulator self-check OK  (span table, timer moments)")
    print(f"  truncations so far: {TRUNCATIONS}")


if __name__ == "__main__":
    _demo()

    rng = np.random.default_rng(0)
    word_freqs = np.array([100, 500, 20, 300, 80, 400, 15, 200, 350, 60], dtype=float)
    fixations = simulate_sentence(word_freqs, nu=0.3, r=10.0, mu_T=200.0, rng=rng)
    print(f"\nSimulated {len(fixations)} fixations (nu=0.3, r=10, mu_T=200):")
    print(f"{'Word':>6}  {'Dur(ms)':>8}")
    for wid, dur in fixations:
        print(f"{wid:>6}  {dur:>8.1f}")
