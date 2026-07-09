"""
swift_simulator.py
==================
Simplified SWIFT model simulator.

Based on:
  Engbert & Rabe (2024) — Journal of Mathematical Psychology, 119, 102843
  Rabe et al.  (2021)  — Psychological Review, 128(3), 516–543

The model simulates a reader's eye movements through a sentence as a
continuous-time stochastic process using the Gillespie algorithm.

FREE PARAMETERS (inferred by BayesFlow):
  t_sac   — mean saccade timer period (ms)
  eta     — word-length exponent
  delta0  — processing span half-width (characters)
  R       — refixation rate factor

FIXED PARAMETERS (set to literature values):
  alpha, beta, h, gamma, tau_l, tau_n, sigma, omega
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Fixed model constants  (Engbert et al. 2005 / Rabe et al. 2021)
# ---------------------------------------------------------------------------
FIXED = {
    "alpha": 10.0,    # baseline processing difficulty
    "beta":  0.5,     # word-frequency sensitivity
    "h":     0.5,     # foveal inhibition strength
    "gamma": 2.0,     # saccade target selection sharpness
    "tau_l": 150.0,   # labile stage duration (ms)
    "tau_n": 50.0,    # non-labile stage duration (ms)
    "sigma": 1.5,     # oculomotor noise (characters)
    "omega": 0.1,     # post-lexical decay rate
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
# Prior sampler
# ---------------------------------------------------------------------------

def sample_prior(rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Sample one set of free parameters uniformly from prior bounds."""
    if rng is None:
        rng = np.random.default_rng()
    return {k: float(rng.uniform(*PRIOR_BOUNDS[k])) for k in PARAM_NAMES}


def normalise_theta(params: Dict[str, float]) -> np.ndarray:
    """Convert parameter dict → normalised [0,1] array."""
    theta = np.array([params[k] for k in PARAM_NAMES], dtype=np.float32)
    return (theta - THETA_MIN) / (THETA_MAX - THETA_MIN)


def denormalise_theta(theta_norm: np.ndarray) -> Dict[str, float]:
    """Convert normalised [0,1] array → parameter dict."""
    theta = theta_norm * (THETA_MAX - THETA_MIN) + THETA_MIN
    return dict(zip(PARAM_NAMES, theta.tolist()))


# ---------------------------------------------------------------------------
# Word property helpers
# ---------------------------------------------------------------------------

def processing_difficulty(word_lengths: np.ndarray,
                           word_freqs: np.ndarray,
                           alpha: float, beta: float) -> np.ndarray:
    """
    D_i = alpha * (1 - beta * log(freq_i) / log(freq_max))

    Rare words → higher D → slower processing → longer fixations.
    """
    freqs = np.clip(word_freqs, 1, None).astype(float)
    log_ratio = np.log(freqs) / np.log(freqs.max() + 1e-8)
    D = alpha * (1.0 - beta * log_ratio)
    return np.clip(D, 0.5, alpha)


def processing_rate(eccentricity: float,
                    word_length: float,
                    delta0: float,
                    eta: float) -> float:
    """
    Lambda_i = parabolic_window(eccentricity) * word_length^(-eta)

    Parabolic window: wider to the right (reading direction).
    """
    span = delta0 * 1.2 if eccentricity >= 0 else delta0
    window = max(0.0, 1.0 - (eccentricity / (span + 1e-8)) ** 2)
    length_factor = max(0.1, word_length ** (-eta))
    return window * length_factor


# ---------------------------------------------------------------------------
# Core Gillespie simulator
# ---------------------------------------------------------------------------

class SWIFTSimulator:
    """Simplified SWIFT model — simulates one sentence at a time."""

    TIMER_THRESHOLD = 10    # internal timer units before a saccade is initiated

    def __init__(self, params: Dict[str, float]):
        self.p = {**FIXED, **params}

    def simulate_sentence(self,
                          word_lengths: np.ndarray,
                          word_freqs:   np.ndarray,
                          max_time_ms:  float = 5000.0,
                          rng: Optional[np.random.Generator] = None
                          ) -> List[Tuple[int, float, float]]:
        """
        Simulate reading of one sentence.

        Parameters
        ----------
        word_lengths : character counts per word, shape (N_words,)
        word_freqs   : frequency per word,        shape (N_words,)
        max_time_ms  : simulation time ceiling (ms)
        rng          : numpy random generator

        Returns
        -------
        fixations : list of (word_id [1-indexed], landing_pos, duration_ms)
        """
        if rng is None:
            rng = np.random.default_rng()

        p  = self.p
        N  = len(word_lengths)
        D  = processing_difficulty(word_lengths, word_freqs,
                                   p["alpha"], p["beta"])
        n_max = np.round(D).astype(int).clip(1)
        n_act = np.zeros(N, dtype=int)

        # Saccade pipeline
        timer           = 0
        labile_target   = None
        labile_left     = 0.0
        nonlabile_target = None
        nonlabile_left  = 0.0

        # Eye state
        current_word   = 0
        current_pos    = word_lengths[0] / 2.0
        n_act[0]       = n_max[0]          # activate first word on arrival

        # Word centre positions (for eccentricity computation)
        centres = np.cumsum(word_lengths) - word_lengths / 2.0

        fixations     = []
        fix_start     = 0.0
        t             = 0.0

        while t < max_time_ms and current_word < N:

            eye_pos       = centres[current_word]
            eccentricities = centres - eye_pos

            # --- Rates ---
            # Activation build-up
            act_up = np.array([
                processing_rate(eccentricities[i], word_lengths[i],
                                p["delta0"], p["eta"]) / max(D[i], 1.0)
                if n_act[i] < n_max[i] else 0.0
                for i in range(N)
            ])
            # Post-lexical decay
            act_dn = np.array([
                p["omega"] if n_act[i] == n_max[i] else 0.0
                for i in range(N)
            ])
            # Saccade timer (slowed by foveal activation)
            fov_load   = n_act[current_word] / max(n_max[current_word], 1)
            inhibition = 1.0 / (1.0 + p["h"] * fov_load)
            timer_rate = (self.TIMER_THRESHOLD / p["t_sac"]) * inhibition

            all_rates  = np.concatenate([act_up, act_dn, [timer_rate]])
            W_total    = all_rates.sum()
            if W_total < 1e-12:
                break

            # --- Waiting time (Gillespie) ---
            dt  = rng.exponential(1.0 / W_total)
            t  += dt

            # --- Advance saccade pipeline ---
            if nonlabile_target is not None:
                nonlabile_left -= dt
                if nonlabile_left <= 0:
                    # Execute saccade
                    tgt = nonlabile_target
                    nonlabile_target = None

                    duration = max(t - fix_start - p["tau_n"], 80.0)
                    fixations.append((current_word + 1, current_pos, duration))

                    noise   = rng.normal(0, p["sigma"])
                    landing = np.clip(word_lengths[tgt] / 2.0 + noise,
                                      0.5, word_lengths[tgt] - 0.5)
                    current_word = tgt
                    current_pos  = landing
                    n_act[current_word] = n_max[current_word]
                    fix_start    = t

                    if current_word >= N - 1:
                        fixations.append((current_word + 1, current_pos, 100.0))
                        break

            if labile_target is not None:
                labile_left -= dt
                if labile_left <= 0:
                    nonlabile_target = labile_target
                    nonlabile_left   = p["tau_n"]
                    labile_target    = None

            # --- Choose firing process ---
            probs  = all_rates / W_total
            choice = rng.choice(len(all_rates), p=probs)

            if choice < N:
                n_act[choice] = min(n_act[choice] + 1, n_max[choice])
            elif choice < 2 * N:
                n_act[choice - N] = max(n_act[choice - N] - 1, 0)
            else:
                timer += 1
                if timer >= self.TIMER_THRESHOLD:
                    timer = 0
                    if labile_target is None and nonlabile_target is None:
                        probs_tgt = self._target_probs(
                            n_act, n_max, current_word, N,
                            p["gamma"], p["R"], rng)
                        labile_target = int(rng.choice(N, p=probs_tgt))
                        labile_left   = p["tau_l"]

        return fixations

    def _target_probs(self, n_act, n_max, current_word, N,
                      gamma, R, rng) -> np.ndarray:
        """Saccade target selection: proportional to relative activation^gamma."""
        rel = np.array([(n_act[i] / max(n_max[i], 1)) ** gamma
                        for i in range(N)])
        # Penalise regressions
        for i in range(current_word):
            rel[i] *= 0.05
        # Refixation bonus
        rel[current_word] += R * rel.sum()

        total = rel.sum()
        if total < 1e-12:
            p = np.zeros(N)
            p[min(current_word + 1, N - 1)] = 1.0
            return p
        return rel / total


# ---------------------------------------------------------------------------
# Batch simulation helper (used directly by BayesFlow simulator functions)
# ---------------------------------------------------------------------------

def run_one_simulation(params: Dict[str, float],
                       word_lengths_list: list,
                       word_freqs_list:   list,
                       seq_len: int,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Simulate one (theta, fixation_sequence) pair.
    Returns padded fixation array of shape (seq_len, 3).
    """
    n_sent   = len(word_lengths_list)
    sent_idx = rng.integers(0, n_sent)
    wl       = word_lengths_list[sent_idx]
    wf       = word_freqs_list[sent_idx]

    sim      = SWIFTSimulator(params)
    fixations = sim.simulate_sentence(wl, wf, rng=rng)

    padded   = np.zeros((seq_len, 3), dtype=np.float32)
    if len(fixations) > 0:
        arr       = np.array(fixations, dtype=np.float32)
        arr[:, 0] /= (len(wl) + 1e-8)
        arr[:, 1] /= 10.0
        arr[:, 2] /= 1000.0
        n         = min(len(arr), seq_len)
        padded[:n] = arr[:n]
    return padded


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng          = np.random.default_rng(0)
    word_lengths = np.array([5, 3, 7, 4, 6, 3, 8, 5, 4, 6], dtype=float)
    word_freqs   = np.array([100, 500, 20, 300, 80, 400, 15, 200, 350, 60], dtype=float)

    params    = sample_prior(rng)
    print("Sampled parameters:", params)

    sim       = SWIFTSimulator(params)
    fixations = sim.simulate_sentence(word_lengths, word_freqs, rng=rng)

    print(f"\nSimulated {len(fixations)} fixations:")
    print(f"{'Word':>6}  {'Landing':>8}  {'Duration(ms)':>14}")
    for wid, lpos, dur in fixations:
        print(f"{wid:>6}  {lpos:>8.2f}  {dur:>14.1f}")
