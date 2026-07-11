"""
simulator.py
============
Simplified SWIFT model simulator (forward model for amortised inference).

Based on:
  Engbert & Rabe (2024) — Journal of Mathematical Psychology, 119, 102843
  Rabe et al.  (2021)  — Psychological Review, 128(3), 516-543

A reader's eye movements through a sentence are simulated as a continuous-time
stochastic process with the Gillespie algorithm. The likelihood of an observed
fixation sequence is intractable, which is precisely why we learn the inverse
map with BayesFlow instead of evaluating it.

FREE PARAMETERS (inferred by BayesFlow)
  t_sac   - mean saccade-timer period (ms); sets the fixation-duration scale
  eta     - word-length processing exponent (long words processed slower)
  delta0  - processing-span half-width (characters); the attention window
  R        - refixation-rate factor

FIXED PARAMETERS (literature values; calibrated so the marginal fixation-count
and duration distributions match participant VP10 — see tools/calibrate.py)
  alpha, beta, h, gamma, tau_l, tau_n, sigma, omega

Note on foveal inhibition (Engbert & Rabe 2024, saccade-timer section):
ongoing lexical processing of the *currently fixated* word inhibits the saccade
timer, so the eye leaves a word sooner once it has been processed. The timer
rate therefore scales with how much of the foveal word has been processed, not
(as an earlier version had it) inversely — that reversed sign both doubled the
fixation durations and produced far too many refixations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from swift.config import (
    N_FEATURES, compute_reader_stats, normalise_sequence, pad_sequence,
)

# ---------------------------------------------------------------------------
# Fixed model constants (calibrated against VP10 marginals)
# ---------------------------------------------------------------------------
FIXED = {
    "alpha": 12.0,    # baseline processing difficulty
    "beta":  0.35,    # word-frequency sensitivity
    "h":     0.65,    # foveal-release strength (processed word -> leave fast)
    "gamma": 3.0,     # saccade target-selection sharpness
    "kappa": 0.30,    # forward saccade-distance decay (skip vs. step to next word)
    "rho":   0.15,    # parafoveal attenuation (non-foveal processing efficiency)
    "refix_gain": 1.0,  # how strongly R drives refixation (identifiability of R)
    "sigma": 1.2,     # oculomotor landing noise (characters)
    "omega": 0.05,    # post-lexical activation decay rate
}

# Prior bounds for the 4 free parameters
PRIOR_BOUNDS = {
    "t_sac":  (150.0, 350.0),
    "eta":    (0.1,   1.0),
    "delta0": (4.0,   15.0),
    "R":      (0.1,   0.9),
}

PARAM_NAMES = ["t_sac", "eta", "delta0", "R"]

THETA_MIN = np.array([PRIOR_BOUNDS[k][0] for k in PARAM_NAMES], dtype=np.float32)
THETA_MAX = np.array([PRIOR_BOUNDS[k][1] for k in PARAM_NAMES], dtype=np.float32)


# ---------------------------------------------------------------------------
# Prior sampler and (de)normalisation of theta
# ---------------------------------------------------------------------------

def sample_prior(rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Sample one set of free parameters uniformly from the prior bounds."""
    if rng is None:
        rng = np.random.default_rng()
    return {k: float(rng.uniform(*PRIOR_BOUNDS[k])) for k in PARAM_NAMES}


def normalise_theta(params: Dict[str, float]) -> np.ndarray:
    """Parameter dict -> normalised [0, 1] array (training scale)."""
    theta = np.array([params[k] for k in PARAM_NAMES], dtype=np.float32)
    return (theta - THETA_MIN) / (THETA_MAX - THETA_MIN)


def denormalise_theta(theta_norm: np.ndarray) -> Dict[str, float]:
    """Normalised [0, 1] array -> parameter dict (original scale)."""
    theta = np.asarray(theta_norm, dtype=np.float32) * (THETA_MAX - THETA_MIN) + THETA_MIN
    return dict(zip(PARAM_NAMES, theta.tolist()))


# ---------------------------------------------------------------------------
# Word-property helpers
# ---------------------------------------------------------------------------

def processing_difficulty(word_lengths: np.ndarray,
                          word_freqs: np.ndarray,
                          alpha: float, beta: float) -> np.ndarray:
    """D_i = alpha * (1 - beta * log(freq_i)/log(freq_max)).
    Rare words -> higher D -> slower processing -> longer/more fixations."""
    freqs = np.clip(word_freqs, 1.0, None).astype(float)
    log_ratio = np.log(freqs) / np.log(freqs.max() + 1e-8)
    D = alpha * (1.0 - beta * log_ratio)
    return np.clip(D, 0.5, alpha)


def processing_rate(eccentricity: float,
                    word_length: float,
                    delta0: float,
                    eta: float) -> float:
    """Lambda_i = parabolic_window(eccentricity) * word_length^(-eta).
    The window is wider to the right (reading direction)."""
    span = delta0 * 1.2 if eccentricity >= 0 else delta0
    window = max(0.0, 1.0 - (eccentricity / (span + 1e-8)) ** 2)
    length_factor = max(0.1, word_length ** (-eta))
    return window * length_factor


# ---------------------------------------------------------------------------
# Core Gillespie simulator
# ---------------------------------------------------------------------------

class SWIFTSimulator:
    """Simplified SWIFT model -- simulates one sentence at a time."""

    # The saccade timer accumulates TIMER_THRESHOLD sub-events before firing.
    # Summing K exponential waits gives an Erlang(K) fixation duration with
    # coefficient of variation 1/sqrt(K) -- matching the tight, right-skewed
    # real duration distribution (a single exponential would be far too spread).
    TIMER_THRESHOLD = 14

    def __init__(self, params: Dict[str, float]):
        self.p = {**FIXED, **params}

    def simulate_sentence(self,
                          word_lengths: np.ndarray,
                          word_freqs: np.ndarray,
                          max_time_ms: float = 8000.0,
                          rng: Optional[np.random.Generator] = None
                          ) -> List[Tuple[int, float, float, float]]:
        """Simulate reading one sentence.

        Two coupled continuous-time processes evolve via Gillespie steps:
          * lexical activation of every word (parafoveal preview), and
          * a saccade timer whose rate rises as the fixated word is processed.

        The saccade timer's inter-event interval *is* the fixation duration:
        a word that is still being processed holds the eye (low rate, long
        fixation); once processed, the rate rises and the eye leaves promptly.
        The labile / non-labile programming stages of full SWIFT are abstracted
        into this timer.

        Returns fixations as tuples
        ``(word_id [1-indexed], landing_position, duration_ms, word_length)``.
        """
        if rng is None:
            rng = np.random.default_rng()

        p = self.p
        N = len(word_lengths)
        D = processing_difficulty(word_lengths, word_freqs, p["alpha"], p["beta"])
        n_max = np.round(D).astype(int).clip(1)
        n_act = np.zeros(N, dtype=int)

        current_word = 0
        current_pos = word_lengths[0] / 2.0
        n_act[0] = min(1, n_max[0])          # begin processing the first word

        centres = np.cumsum(word_lengths) - word_lengths / 2.0
        K = self.TIMER_THRESHOLD

        fixations: List[Tuple[int, float, float, float]] = []
        fix_start = 0.0
        timer_ticks = 0
        t = 0.0

        while t < max_time_ms and current_word < N:
            eccentricities = centres - centres[current_word]

            # --- Activation build-up rates (foveal + attenuated parafoveal) ---
            act_up = np.array([
                processing_rate(eccentricities[i], word_lengths[i],
                                p["delta0"], p["eta"]) / max(D[i], 1.0)
                * (1.0 if i == current_word else p["rho"])
                if n_act[i] < n_max[i] else 0.0
                for i in range(N)
            ])
            # --- Post-lexical decay of fully-processed words ---
            act_dn = np.array([
                p["omega"] if n_act[i] == n_max[i] else 0.0
                for i in range(N)
            ])

            # --- Saccade timer (per-tick rate; K ticks fire a saccade) ---
            # Foveal release: the more the fixated word is processed, the faster
            # the eye leaves. Hard/unprocessed words hold the gaze -> long
            # fixations; easy words are left quickly -> short fixations.
            processed = n_act[current_word] / max(n_max[current_word], 1)
            tick_rate = (K / p["t_sac"]) * (1.0 + p["h"] * processed)

            all_rates = np.concatenate([act_up, act_dn, [tick_rate]])
            W_total = all_rates.sum()
            if W_total < 1e-12:
                break

            # --- Gillespie waiting time and event ---
            dt = rng.exponential(1.0 / W_total)
            t += dt
            choice = rng.choice(len(all_rates), p=all_rates / W_total)

            if choice < N:                       # activation build-up
                n_act[choice] = min(n_act[choice] + 1, n_max[choice])
            elif choice < 2 * N:                 # post-lexical decay
                n_act[choice - N] = max(n_act[choice - N] - 1, 0)
            else:                                # saccade timer tick
                timer_ticks += 1
                if timer_ticks < K:
                    continue

                # Kth tick -> saccade fires; the eye moves.
                timer_ticks = 0
                duration = max(t - fix_start, 60.0)
                fixations.append((current_word + 1, current_pos,
                                  duration, float(word_lengths[current_word])))

                tgt = self._select_target(n_act, n_max, current_word, N,
                                          p["gamma"], p["R"], p["kappa"],
                                          p["refix_gain"], rng)
                noise = rng.normal(0.0, p["sigma"])
                current_pos = float(np.clip(word_lengths[tgt] / 2.0 + noise,
                                            0.5, word_lengths[tgt] - 0.5))
                current_word = tgt
                n_act[current_word] = max(n_act[current_word], min(1, n_max[current_word]))
                fix_start = t

                if current_word >= N - 1:        # reached final word -> one last look
                    processed = n_act[current_word] / max(n_max[current_word], 1)
                    mean_last = p["t_sac"] / (1.0 + p["h"] * processed)
                    last = max(rng.gamma(K, mean_last / K), 60.0)
                    fixations.append((current_word + 1, current_pos,
                                      last, float(word_lengths[current_word])))
                    break

        return fixations

    @staticmethod
    def _select_target(n_act, n_max, current_word, N, gamma, R, kappa,
                       refix_gain, rng) -> int:
        """Choose the next saccade target. Unprocessed words attract the eye
        (weight rises with remaining processing ^ gamma), but a forward-distance
        decay (kappa) keeps most saccades short: the eye usually steps to the
        next word, sometimes skips one. Regressions are penalised and the
        current word gets a refixation bonus scaled by R (via refix_gain) and by
        how much of it still needs processing."""
        idx = np.arange(N)
        remaining = np.array([1.0 - n_act[i] / max(n_max[i], 1) for i in range(N)])
        rel = remaining ** gamma

        forward = idx > current_word
        rel[forward] *= kappa ** (idx[forward] - current_word - 1)  # short saccades
        rel[:current_word] *= 0.05                                   # regressions
        rel[current_word] += (refix_gain * R * remaining[current_word]
                              * (rel.sum() + 1e-8))

        total = rel.sum()
        if total < 1e-12:                                 # nothing left ahead
            return min(current_word + 1, N - 1)
        return int(rng.choice(N, p=rel / total))


# ---------------------------------------------------------------------------
# Session builders used by the BayesFlow simulator functions
# ---------------------------------------------------------------------------

def simulate_one_sentence_features(params: Dict[str, float],
                                   word_lengths: np.ndarray,
                                   word_freqs: np.ndarray,
                                   rng: np.random.Generator) -> np.ndarray:
    """One sentence -> raw (n_fix, 4) feature array (un-normalised)."""
    sim = SWIFTSimulator(params)
    fixations = sim.simulate_sentence(word_lengths, word_freqs, rng=rng)
    if len(fixations) == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    return np.array(fixations, dtype=np.float32)


def run_one_reader(params: Dict[str, float],
                   word_lengths_list: list,
                   word_freqs_list: list,
                   seq_len: int,
                   m_sentences: int,
                   rng: np.random.Generator):
    """Simulate one reader (fixed theta) reading ``m_sentences`` sentences.

    Returns ``(sequence, stats)``:
      * sequence : concatenated, normalised, zero-padded fixations
                   of shape (seq_len, N_FEATURES) -> LSTM summary network
      * stats    : hand-crafted summary statistics (N_STATS,) -> direct
                   conditions for the inference network

    Concatenating several sentences under one theta is the key modelling choice:
    a single ~8-fixation sequence carries almost no information about the
    word-length exponent or the processing span, but many do."""
    n_sent = len(word_lengths_list)
    idx = rng.integers(0, n_sent, size=m_sentences)

    rows = [
        simulate_one_sentence_features(
            params, word_lengths_list[i], word_freqs_list[i], rng)
        for i in idx
    ]
    raw = np.concatenate(rows, axis=0) if any(len(r) for r in rows) \
        else np.zeros((0, N_FEATURES), dtype=np.float32)
    seq = pad_sequence(normalise_sequence(raw), seq_len)
    stats = compute_reader_stats(rows)
    return seq, stats


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    word_lengths = np.array([5, 3, 7, 4, 6, 3, 8, 5, 4, 6], dtype=float)
    word_freqs = np.array([100, 500, 20, 300, 80, 400, 15, 200, 350, 60], dtype=float)

    params = sample_prior(rng)
    print("Sampled parameters:", params)

    fixations = SWIFTSimulator(params).simulate_sentence(word_lengths, word_freqs, rng=rng)
    print(f"\nSimulated {len(fixations)} fixations:")
    print(f"{'Word':>6}  {'Landing':>8}  {'Dur(ms)':>8}  {'WordLen':>8}")
    for wid, lpos, dur, wlen in fixations:
        print(f"{wid:>6}  {lpos:>8.2f}  {dur:>8.1f}  {wlen:>8.0f}")
