"""
train_bayesflow.py
==================
BayesFlow v2.0.11 amortized inference pipeline for the SWIFT model.

API verified against: https://bayesflow.org/v2.0.11/_examples/Linear_Regression_Starter.html

Pipeline:
  prior()        → samples theta from uniform prior
  likelihood()   → runs SWIFT simulator given theta
  simulator      → bf.simulators.make_simulator([prior, likelihood])
  adapter        → preprocesses data for neural networks
  summary_net    → TimeSeriesNetwork (LSTM) compresses fixation sequences
  inference_net  → CouplingFlow approximates P(theta | summary)
  workflow       → bf.BasicWorkflow ties everything together

Install:
    pip install bayesflow keras torch torchvision torchaudio

Usage:
    python inference/train_bayesflow.py
"""

import os
import sys
import numpy as np

# Set Keras backend BEFORE importing bayesflow or keras
# Change to "torch" or "jax" if preferred
if not os.environ.get("KERAS_BACKEND"):
    os.environ["KERAS_BACKEND"] = "torch"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import bayesflow as bf
    import keras
    BF_OK = True
    print(f"BayesFlow {bf.__version__} | Keras backend: {keras.backend.backend()}")
except ImportError:
    BF_OK = False
    print("BayesFlow not found. Install: pip install bayesflow torch keras")

from simulator.swift_simulator import (
    sample_prior, SWIFTSimulator, PARAM_NAMES,
    THETA_MIN, THETA_MAX, normalise_theta,
    denormalise_theta, run_one_simulation
)

# ---------------------------------------------------------------------------
# Module-level state — set via set_corpus() before training
# ---------------------------------------------------------------------------
_WORD_LENGTHS_LIST = None
_WORD_FREQS_LIST   = None
_SEQ_LEN           = 15
_RNG               = np.random.default_rng(42)


def set_corpus(word_lengths_list: list,
               word_freqs_list:   list,
               seq_len: int = 30) -> None:
    """Call this before training to register corpus data."""
    global _WORD_LENGTHS_LIST, _WORD_FREQS_LIST, _SEQ_LEN
    _WORD_LENGTHS_LIST = word_lengths_list
    _WORD_FREQS_LIST   = word_freqs_list
    _SEQ_LEN           = seq_len
    print(f"[set_corpus] {len(word_lengths_list)} sentences registered, "
          f"seq_len={seq_len}")


# ---------------------------------------------------------------------------
# BayesFlow v2 simulator functions
# BayesFlow v2 API:
#   prior()      → dict of parameter draws
#   likelihood() → dict of data draws (takes prior outputs as kwargs)
# ---------------------------------------------------------------------------

def swift_prior() -> dict:
    """
    Sample one set of SWIFT parameters from uniform priors.
    Returns normalised theta in [0, 1] for training stability.
    """
    params    = sample_prior(_RNG)
    theta_norm = normalise_theta(params)
    return {"theta": theta_norm}


def swift_likelihood(theta: np.ndarray) -> dict:
    """
    Run SWIFT simulator for given (normalised) theta.
    Returns padded fixation sequence of shape (seq_len, 3).
    """
    params    = denormalise_theta(theta)
    fixations = run_one_simulation(
        params            = params,
        word_lengths_list = _WORD_LENGTHS_LIST,
        word_freqs_list   = _WORD_FREQS_LIST,
        seq_len           = _SEQ_LEN,
        rng               = _RNG,
    )
    return {"fixations": fixations}   # shape: (seq_len, 3)


# ---------------------------------------------------------------------------
# Adapter builder
# ---------------------------------------------------------------------------

def build_adapter() -> "bf.Adapter":
    """
    BayesFlow v2 Adapter:
      - "theta"     → inference_variables  (what we infer)
      - "fixations" → summary_variables    (what summary net compresses)
    """
    adapter = (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(["theta"],     into="inference_variables")
        .rename("fixations",              "summary_variables")
    )
    return adapter


# ---------------------------------------------------------------------------
# Main training function (Online mode)
# ---------------------------------------------------------------------------

def train(word_lengths_list: list,
          word_freqs_list:   list,
          n_epochs:               int   = 50,
          batch_size:             int   = 32,
          num_batches_per_epoch:  int   = 100,
          seq_len:                int   = 30,
          save_path:              str   = "inference/trained_model"
          ) -> "bf.BasicWorkflow":
    """
    Train the BayesFlow amortized posterior estimator for SWIFT online.
    """
    if not BF_OK:
        raise ImportError("Install BayesFlow: pip install bayesflow torch keras")

    # Register corpus
    set_corpus(word_lengths_list, word_freqs_list, seq_len)

    print("\n" + "=" * 55)
    print("SWIFT — BayesFlow v2 Training (Online Mode)")
    print("=" * 55)

    simulator = bf.simulators.make_simulator([swift_prior, swift_likelihood])
    adapter = build_adapter()

    summary_network = bf.networks.TimeSeriesNetwork(
        summary_dim  = 32,
        bidirectional = False,
    )

    inference_network = bf.networks.CouplingFlow(
        transform  = "spline",
        num_layers = 6,
    )

    workflow = bf.BasicWorkflow(
        simulator         = simulator,
        adapter           = adapter,
        inference_network = inference_network,
        summary_network   = summary_network,
        standardize       = ["inference_variables"],
    )

    print(f"\nTraining: {n_epochs} epochs × "
          f"{num_batches_per_epoch} batches × batch_size {batch_size}")
    print(f"Total simulations per run: "
          f"{n_epochs * num_batches_per_epoch * batch_size:,}")
    print("-" * 55)

    # In BayesFlow v2, we use fit_online natively instead of touching the approximator directly
    history = workflow.fit_online(
        epochs=n_epochs,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
    )

    os.makedirs(save_path, exist_ok=True)
    model_file = os.path.join(save_path, "swift_approximator.keras")
    workflow.approximator.save(filepath=model_file)
    print(f"\nModel saved → {model_file}")

    try:
        import matplotlib.pyplot as plt
        f = bf.diagnostics.plots.loss(history)
        loss_file = os.path.join(save_path, "training_loss.png")
        plt.savefig(loss_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Loss plot saved → {loss_file}")
    except Exception as e:
        print(f"Could not save loss plot: {e}")

    return workflow


# ---------------------------------------------------------------------------
# Offline training on pre-generated (theta, fixation) pairs
# ---------------------------------------------------------------------------

def train_offline(thetas:    np.ndarray,
                  seqs:      np.ndarray,
                  wll:       list,
                  wfl:       list,
                  n_epochs:  int = 50,
                  batch_size: int = 32,
                  seq_len:   int = 15,
                  save_path: str = "inference/trained_model"
                  ) -> "bf.BasicWorkflow":
    """
    Train BayesFlow on pre-generated (theta, fixation) pairs loaded from disk.
    This is MUCH faster than online training because the simulator does not
    run during training — all simulations were pre-computed.
    """
    if not BF_OK:
        raise ImportError("Install BayesFlow: pip install bayesflow torch keras")

    set_corpus(wll, wfl, seq_len)

    print("\n" + "=" * 55)
    print("SWIFT — BayesFlow v2 Offline Training (Torch Backend)")
    print("=" * 55)
    print(f"Training data : {len(thetas):,} pre-generated simulations")
    print(f"Epochs        : {n_epochs}")
    print(f"Batch size    : {batch_size}")
    print(f"Steps/epoch   : {len(thetas) // batch_size}")
    print("-" * 55)

    adapter = build_adapter()

    summary_network = bf.networks.TimeSeriesNetwork(
        summary_dim   = 32,
        bidirectional = False,
    )
    
    inference_network = bf.networks.CouplingFlow(
        transform  = "spline",
        num_layers = 6,
    )

    simulator = bf.simulators.make_simulator([swift_prior, swift_likelihood])

    workflow = bf.BasicWorkflow(
        simulator         = simulator,
        adapter           = adapter,
        inference_network = inference_network,
        summary_network   = summary_network,
        standardize       = ["inference_variables"],
    )

    # In BayesFlow v2, we can simply pass a dictionary mapping directly to fit_offline
    # No PyTorch Dataset or DataLoader is required as fit_offline builds an OfflineDataset internally
    data_dict = {
        "theta": thetas.astype(np.float32),
        "fixations": seqs.astype(np.float32)
    }

    history = workflow.fit_offline(
        data=data_dict,
        epochs=n_epochs,
        batch_size=batch_size,
    )

    os.makedirs(save_path, exist_ok=True)
    model_file = os.path.join(save_path, "swift_approximator.keras")
    workflow.approximator.save(filepath=model_file)
    print(f"\nModel saved → {model_file}")

    try:
        import matplotlib.pyplot as plt
        f = bf.diagnostics.plots.loss(history)
        loss_file = os.path.join(save_path, "training_loss.png")
        plt.savefig(loss_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Loss plot saved → {loss_file}")
    except Exception as e:
        print(f"Could not save loss plot: {e}")

    return workflow


# ---------------------------------------------------------------------------
# Inference on real participant data
# ---------------------------------------------------------------------------

def run_inference(workflow,
                  fixation_array:      np.ndarray,
                  n_posterior_samples: int = 5000
                  ) -> np.ndarray:
    """
    Run amortized inference on one participant's fixation data.
    """
    seq_len = _SEQ_LEN

    # Pad / truncate to seq_len → shape (1, seq_len, 3)
    padded         = np.zeros((1, seq_len, 3), dtype=np.float32)
    n              = min(len(fixation_array), seq_len)
    padded[0, :n]  = fixation_array[:n]

    post_draws = workflow.sample(
        conditions  = {"fixations": padded},
        num_samples = n_posterior_samples,
    )
    samples_norm = post_draws["theta"][0]   # (n_posterior_samples, 4)

    # Denormalise to original scale
    samples = samples_norm * (THETA_MAX - THETA_MIN) + THETA_MIN

    print("\nPosterior estimates:")
    for j, name in enumerate(PARAM_NAMES):
        mean = float(samples[:, j].mean())
        lo   = float(np.percentile(samples[:, j], 2.5))
        hi   = float(np.percentile(samples[:, j], 97.5))
        print(f"  {name:<10}: {mean:.3f}  [95% CI: {lo:.3f} – {hi:.3f}]")

    return samples


# ---------------------------------------------------------------------------
# Load saved model & Rebuild
# ---------------------------------------------------------------------------

def load_workflow(model_path: str) -> "keras.Model":
    import keras
    approximator = keras.saving.load_model(model_path)
    print(f"Loaded approximator from {model_path}")
    return approximator

def rebuild_workflow(model_path: str) -> "bf.BasicWorkflow":
    if not BF_OK:
        raise ImportError("Install BayesFlow: pip install bayesflow torch keras")

    import keras
    simulator = bf.simulators.make_simulator([swift_prior, swift_likelihood])
    adapter   = build_adapter()

    summary_network   = bf.networks.TimeSeriesNetwork(
        summary_dim   = 32,
        bidirectional = False,
    )
    
    inference_network = bf.networks.CouplingFlow(
        transform  = "spline",
        num_layers = 6,
    )

    workflow = bf.BasicWorkflow(
        simulator         = simulator,
        adapter           = adapter,
        inference_network = inference_network,
        summary_network   = summary_network,
        standardize       = ["inference_variables"],
    )

    workflow.approximator = keras.saving.load_model(model_path)
    print(f"Loaded model weights from {model_path}")

    return workflow


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running demo with synthetic corpus (20 sentences)...")

    rng = np.random.default_rng(0)
    n   = 20
    wll = [rng.integers(3, 10, size=rng.integers(8, 15)).astype(float)
           for _ in range(n)]
    wfl = [rng.integers(10, 500, size=len(wl)).astype(float) for wl in wll]

    if BF_OK:
        wf = train(
            word_lengths_list     = wll,
            word_freqs_list       = wfl,
            n_epochs              = 2,
            batch_size            = 8,
            num_batches_per_epoch = 5,
            save_path             = "inference/demo_model",
        )
        print("Demo training complete.")
    else:
        print("Install bayesflow first: pip install bayesflow torch keras")