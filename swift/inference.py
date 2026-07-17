"""
inference.py
============
BayesFlow v2 amortised-inference pipeline for the SWIFT model.

API verified against BayesFlow 2.0.11.

Pipeline
  swift_prior()       -> theta ~ Uniform(prior), returned normalised to [0,1]
  swift_likelihood()  -> run the SWIFT simulator for one reader (M sentences)
  simulator           -> bf.simulators.make_simulator([prior, likelihood])
  adapter             -> theta -> inference_variables, fixations -> summary_variables
  summary_network     -> TimeSeriesNetwork (LSTM) over the fixation sequence
  inference_network   -> CouplingFlow (spline) approximating p(theta | summary)
  workflow            -> bf.BasicWorkflow

An "observation" is one reader (fixed theta) reading M_SENTENCES sentences,
their fixations concatenated. That is what lets the network identify the span
(nu) and processing-rate (r) parameters, which a single ~8-fixation sequence
cannot constrain.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

try:
    import torch
    # Force CPU: Apple's MPS backend does not implement linalg_qr, which the
    # spline CouplingFlow needs. This is a small model, so CPU is fine and far
    # more reliable than constant MPS->CPU fallbacks.
    torch.backends.mps.is_available = lambda: False
    import bayesflow as bf
    import keras
    BF_OK = True
except ImportError:  # pragma: no cover - lets the simulator/data code import cleanly
    BF_OK = False

from swift.config import (
    M_SENTENCES, MODEL_PATH, N_FEATURES, N_STATS, SEQ_LEN, pad_sequence,
)
from swift.simulator import (
    PARAM_NAMES, THETA_MAX, THETA_MIN,
    denormalise_theta, normalise_theta, run_one_reader, sample_prior,
)

# ---------------------------------------------------------------------------
# Module-level corpus state (registered before training / diagnostics)
# ---------------------------------------------------------------------------
_WFL = None
_RNG = np.random.default_rng(42)


def set_corpus(word_freqs_list) -> None:
    global _WFL
    _WFL = word_freqs_list
    print(f"[set_corpus] {len(word_freqs_list)} sentences registered "
          f"(M_SENTENCES={M_SENTENCES}, SEQ_LEN={SEQ_LEN})")


# ---------------------------------------------------------------------------
# BayesFlow simulator functions
# ---------------------------------------------------------------------------

def swift_prior() -> dict:
    """One prior draw, returned as normalised theta in [0, 1]."""
    return {"theta": normalise_theta(sample_prior(_RNG))}


def swift_likelihood(theta: np.ndarray) -> dict:
    """Run the simulator for one reader given normalised theta."""
    fixations, stats = run_one_reader(
        params=denormalise_theta(theta),
        word_freqs_list=_WFL,
        seq_len=SEQ_LEN, m_sentences=M_SENTENCES, rng=_RNG,
    )
    return {"fixations": fixations, "stats": stats}


def build_adapter():
    """theta   -> inference_variables (what we infer: nu, r, mu_T)
    fixations -> summary_variables    (summary network over the raw sequence)
    stats     -> inference_conditions (hand-crafted stats fed directly, so the
                 flow can read nu/r off the skip / refixation / regression rates
                 -- information the summary network does not reliably extract)."""
    return (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(["theta"], into="inference_variables")
        .rename("fixations", "summary_variables")
        .concatenate(["stats"], into="inference_conditions")
    )


def _build_workflow(simulator):
    summary_network = bf.networks.TimeSeriesNetwork(summary_dim=64, bidirectional=False)
    inference_network = bf.networks.CouplingFlow(transform="spline", num_layers=6)
    return bf.BasicWorkflow(
        simulator=simulator,
        adapter=build_adapter(),
        inference_network=inference_network,
        summary_network=summary_network,
        standardize=["inference_variables", "inference_conditions"],
    )


def _make_simulator():
    return bf.simulators.make_simulator([swift_prior, swift_likelihood])


# ---------------------------------------------------------------------------
# Offline training on pre-generated (theta, fixation) pairs
# ---------------------------------------------------------------------------

def train_offline(thetas: np.ndarray, seqs: np.ndarray, stats: np.ndarray,
                  wfl, n_epochs: int = 80, batch_size: int = 64,
                  save_path=MODEL_PATH):
    if not BF_OK:
        raise ImportError("Install BayesFlow: pip install bayesflow keras torch")
    set_corpus(wfl)

    print("\n" + "=" * 55)
    print("SWIFT - BayesFlow v2 Offline Training")
    print("=" * 55)
    print(f"Training data : {len(thetas):,} readers "
          f"({M_SENTENCES} sentences each)")
    print(f"Epochs {n_epochs} | batch {batch_size} | "
          f"steps/epoch {len(thetas) // batch_size}")

    workflow = _build_workflow(_make_simulator())
    data = {"theta": thetas.astype(np.float32),
            "fixations": seqs.astype(np.float32),
            "stats": stats.astype(np.float32)}
    history = workflow.fit_offline(data=data, epochs=n_epochs, batch_size=batch_size)

    _save_model(workflow, save_path)
    _save_loss(history)
    return workflow


def train_online(wfl, n_epochs=80, batch_size=64,
                 num_batches_per_epoch=200, save_path=MODEL_PATH):
    if not BF_OK:
        raise ImportError("Install BayesFlow: pip install bayesflow keras torch")
    set_corpus(wfl)
    workflow = _build_workflow(_make_simulator())
    history = workflow.fit_online(epochs=n_epochs, batch_size=batch_size,
                                  num_batches_per_epoch=num_batches_per_epoch)
    _save_model(workflow, save_path)
    _save_loss(history)
    return workflow


# ---------------------------------------------------------------------------
# Inference on real participant data
# ---------------------------------------------------------------------------

def sample_posterior(workflow, observation: np.ndarray, stats: np.ndarray,
                     num_samples: int = 2000) -> np.ndarray:
    """Posterior samples (original scale, clipped to prior) for one observation:
    sequence (SEQ_LEN, N_FEATURES) + stats (N_STATS,)."""
    obs = pad_sequence(np.asarray(observation, dtype=np.float32))[None, ...]
    st = np.asarray(stats, dtype=np.float32)[None, ...]
    draws = workflow.sample(conditions={"fixations": obs, "stats": st},
                            num_samples=num_samples)
    samples = draws["theta"][0] * (THETA_MAX - THETA_MIN) + THETA_MIN
    return np.clip(samples, THETA_MIN, THETA_MAX)


def run_inference(workflow, observations: np.ndarray, stats: np.ndarray,
                  num_samples: int = 2000) -> np.ndarray:
    """Pool posterior samples across several reader observations of the same
    participant (each a different random subset of their sentences). Replaces
    the earlier, invalid 'average the raw sequences then infer once' step.

    ``observations`` : (n_obs, SEQ_LEN, N_FEATURES); ``stats`` : (n_obs, N_STATS).
    Returns pooled posterior samples (n_obs * num_samples, 4), original scale.
    """
    observations = np.asarray(observations, dtype=np.float32)
    stats = np.asarray(stats, dtype=np.float32)
    if observations.ndim == 2:
        observations = observations[None, ...]
        stats = stats[None, ...]

    pooled = [sample_posterior(workflow, obs, st, num_samples)
              for obs, st in zip(observations, stats)]
    samples = np.concatenate(pooled, axis=0)

    print("\nPosterior estimates (pooled over "
          f"{len(observations)} reader draws):")
    for j, name in enumerate(PARAM_NAMES):
        col = samples[:, j]
        print(f"  {name:<8}: {col.mean():.3f}  "
              f"[95% CI {np.percentile(col, 2.5):.3f} - {np.percentile(col, 97.5):.3f}]")
    return samples


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def _save_model(workflow, save_path=MODEL_PATH) -> None:
    save_path = str(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    workflow.approximator.save(filepath=save_path)
    print(f"\nModel saved -> {save_path}")


def _save_loss(history) -> None:
    try:
        import matplotlib.pyplot as plt
        from swift.config import FIG_DIR
        bf.diagnostics.plots.loss(history)
        out = FIG_DIR / "training_loss.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Loss plot saved -> {out}")
    except Exception as e:  # pragma: no cover
        print(f"Could not save loss plot: {e}")


def rebuild_workflow(model_path=MODEL_PATH):
    """Rebuild the workflow around a saved approximator (for --mode infer)."""
    if not BF_OK:
        raise ImportError("Install BayesFlow: pip install bayesflow keras torch")
    workflow = _build_workflow(_make_simulator())
    workflow.approximator = keras.saving.load_model(str(model_path))
    print(f"Loaded model weights from {model_path}")
    return workflow
