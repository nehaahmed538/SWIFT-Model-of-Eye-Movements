"""
diagnostics.py
==============
Diagnostic tools for the trained BayesFlow SWIFT model.

Uses BayesFlow v2.0.11 built-in diagnostics:
  bf.diagnostics.plots.loss()
  bf.diagnostics.plots.recovery()
  bf.diagnostics.plots.calibration_histogram()
  bf.diagnostics.plots.calibration_ecdf()
  bf.diagnostics.plots.z_score_contraction()

Plus a custom posterior predictive check (PPC) comparing
simulated vs real fixation distributions.

Reference for SBC interpretation:
  Talts et al. (2018) — arXiv:1804.06788
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.swift_simulator import (
    SWIFTSimulator, PARAM_NAMES, THETA_MIN, THETA_MAX,
    denormalise_theta
)

PARAM_LABELS = {
    "t_sac":  r"$t_{sac}$ (ms)",
    "eta":    r"$\eta$",
    "delta0": r"$\delta_0$ (chars)",
    "R":      r"$R$",
}


# ---------------------------------------------------------------------------
# 1. BayesFlow v2 built-in diagnostics
# ---------------------------------------------------------------------------

def run_builtin_diagnostics(workflow,
                             simulator,
                             n_val:              int = 300,
                             n_posterior_samples: int = 1000,
                             save_dir:           str = "diagnostics"
                             ) -> dict:
    """
    Run all BayesFlow v2 built-in diagnostics and save plots.

    Parameters
    ----------
    workflow   : trained bf.BasicWorkflow
    simulator  : bf.simulators object (same one used during training)
    n_val      : number of validation simulations
    n_posterior_samples : posterior samples per validation dataset
    save_dir   : folder to save plots

    Returns
    -------
    dict with keys "val_sims" and "post_draws" for further use
    """
    try:
        import bayesflow as bf
    except ImportError:
        print("BayesFlow not installed.")
        return {}

    os.makedirs(save_dir, exist_ok=True)

    # --- Generate validation data ---
    print(f"Generating {n_val} validation simulations...")
    val_sims = simulator.sample(n_val)

    # --- Draw posterior samples ---
    print(f"Drawing {n_posterior_samples} posterior samples per dataset...")
    post_draws = workflow.sample(
        conditions  = val_sims,
        num_samples = n_posterior_samples,
    )
    # post_draws["theta"] shape: (n_val, n_posterior_samples, 4)

    # --- Recovery plot ---
    print("Plotting recovery...")
    f = bf.diagnostics.plots.recovery(
        estimates      = post_draws,
        targets        = val_sims,
        variable_names = PARAM_NAMES,
    )
    _save(f, save_dir, "recovery_plot.png")

    # --- SBC calibration histogram ---
    print("Plotting SBC histogram...")
    f = bf.diagnostics.plots.calibration_histogram(
        estimates      = post_draws,
        targets        = val_sims,
        variable_names = PARAM_NAMES,
    )
    _save(f, save_dir, "sbc_histogram.png")

    # --- SBC ECDF (preferred over histogram) ---
    print("Plotting SBC ECDF...")
    f = bf.diagnostics.plots.calibration_ecdf(
        estimates      = post_draws,
        targets        = val_sims,
        variable_names = PARAM_NAMES,
        difference     = True,
        rank_type      = "distance",
    )
    _save(f, save_dir, "sbc_ecdf.png")

    # --- Posterior z-score and contraction ---
    print("Plotting z-score / contraction...")
    f = bf.diagnostics.plots.z_score_contraction(
        estimates      = post_draws,
        targets        = val_sims,
        variable_names = PARAM_NAMES,
    )
    _save(f, save_dir, "contraction_plot.png")

    print(f"\nAll built-in diagnostics saved to {save_dir}/")
    return {"val_sims": val_sims, "post_draws": post_draws}


# ---------------------------------------------------------------------------
# 2. Posterior distribution plot
# ---------------------------------------------------------------------------

def plot_posterior(posterior_samples: np.ndarray,
                   true_params: dict = None,
                   save_dir:    str  = "diagnostics"
                   ) -> None:
    """
    Plot marginal posterior distributions for the 4 SWIFT parameters.

    Parameters
    ----------
    posterior_samples : shape (n_samples, 4) in ORIGINAL parameter scale
    true_params       : dict with true values (for simulation-based validation)
    save_dir          : folder to save plot
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Posterior Distributions — SWIFT Parameters\n"
                 "Orange = posterior  |  Gray shading = prior range",
                 fontsize=12)

    for j, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        lo_prior = float(THETA_MIN[j])
        hi_prior = float(THETA_MAX[j])

        # Prior as background band
        ax.axvspan(lo_prior, hi_prior, alpha=0.12, color="gray",
                   label="Prior (uniform)")

        # Posterior histogram
        ax.hist(posterior_samples[:, j], bins=60,
                range=(lo_prior, hi_prior),
                density=True, color="darkorange", alpha=0.85,
                edgecolor="white", label="Posterior")

        # True value (if known)
        if true_params is not None and name in true_params:
            ax.axvline(true_params[name], color="green", lw=2,
                       linestyle="--",
                       label=f"True: {true_params[name]:.2f}")

        # Posterior mean + 95% CI
        mean = float(posterior_samples[:, j].mean())
        lo95 = float(np.percentile(posterior_samples[:, j], 2.5))
        hi95 = float(np.percentile(posterior_samples[:, j], 97.5))
        ax.axvline(mean, color="red", lw=1.8,
                   label=f"Mean: {mean:.2f}")
        ax.axvspan(lo95, hi95, alpha=0.18, color="red",
                   label=f"95% CI")

        ax.set_xlabel(PARAM_LABELS[name])
        ax.set_ylabel("Density" if j == 0 else "")
        ax.set_title(name)
        ax.set_xlim(lo_prior, hi_prior)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(save_dir, "posterior_VP10.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Posterior plot saved → {out}")


# ---------------------------------------------------------------------------
# 3. Posterior Predictive Check (PPC)
# ---------------------------------------------------------------------------

def posterior_predictive_check(posterior_samples: np.ndarray,
                                real_fix_df,
                                word_lengths_list: list,
                                word_freqs_list:   list,
                                n_ppc:    int = 200,
                                save_dir: str = "diagnostics",
                                rng: np.random.Generator = None
                                ) -> None:
    """
    Posterior Predictive Check:
    Simulate data from the inferred posterior and compare distributions
    to the real participant's fixation data.

    Compares:
      - Fixation duration distribution
      - Landing position distribution
      - Fixations per sentence distribution

    Parameters
    ----------
    posterior_samples : shape (n_samples, 4) in ORIGINAL parameter scale
    real_fix_df       : DataFrame with real fixation data (from load_data)
    n_ppc             : number of posterior samples to simulate from
    """
    if rng is None:
        rng = np.random.default_rng(42)

    os.makedirs(save_dir, exist_ok=True)
    n_sent = len(word_lengths_list)

    # --- Collect simulated statistics ---
    sim_durations, sim_landings, sim_fps = [], [], []

    idx      = rng.choice(len(posterior_samples), size=n_ppc, replace=False)
    selected = posterior_samples[idx]

    print(f"Running PPC with {n_ppc} posterior samples...")
    for i, theta in enumerate(selected):
        params    = dict(zip(PARAM_NAMES, theta.tolist()))
        sent_idx  = rng.integers(0, n_sent)
        sim       = SWIFTSimulator(params)
        fixations = sim.simulate_sentence(
            word_lengths_list[sent_idx],
            word_freqs_list[sent_idx],
            rng=rng,
        )
        if len(fixations) > 0:
            arr = np.array(fixations)
            sim_durations.extend(arr[:, 2].tolist())
            sim_landings.extend(arr[:, 1].tolist())
            sim_fps.append(len(fixations))

    # --- Real statistics ---
    real_dur = real_fix_df["fixation_duration"].values
    real_lnd = real_fix_df["landing_position"].values
    real_fps = real_fix_df.groupby("sentence_id").size().values

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Posterior Predictive Check\n"
        "Blue = Real VP10 data  |  Orange = Simulated from posterior",
        fontsize=12
    )

    _ppc_panel(axes[0], real_dur, sim_durations,
               "Fixation Duration (ms)", bins=40)
    _ppc_panel(axes[1], real_lnd, sim_landings,
               "Landing Position (chars)", bins=30)
    _ppc_panel(axes[2], real_fps, sim_fps,
               "Fixations per Sentence", bins=15)

    plt.tight_layout()
    out = os.path.join(save_dir, "ppc_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"PPC plot saved → {out}")

    # --- Summary table ---
    print("\n===== PPC SUMMARY =====")
    print(f"{'Statistic':<28} {'Real':>10} {'Simulated':>12}")
    print("-" * 52)
    stats = [
        ("Mean duration (ms)",    np.mean(real_dur),  np.mean(sim_durations)),
        ("Std  duration (ms)",    np.std(real_dur),   np.std(sim_durations)),
        ("Mean fixations/sent",   np.mean(real_fps),  np.mean(sim_fps)),
        ("Mean landing pos",      np.mean(real_lnd),  np.mean(sim_landings)),
    ]
    for name, rv, sv in stats:
        print(f"{name:<28} {rv:>10.2f} {sv:>12.2f}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ppc_panel(ax, real, sim, xlabel, bins=30):
    ax.hist(real, bins=bins, density=True, alpha=0.6,
            color="steelblue", label="Real", edgecolor="white")
    ax.hist(sim,  bins=bins, density=True, alpha=0.6,
            color="darkorange", label="Simulated", edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)


def _save(fig, save_dir, filename):
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


if __name__ == "__main__":
    print("Diagnostics module ready.")
    print("Call run_builtin_diagnostics() after training the workflow.")
