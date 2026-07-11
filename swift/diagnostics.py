"""
diagnostics.py
==============
Diagnostics for the trained BayesFlow SWIFT model.

BayesFlow v2 built-ins
  recovery, calibration_histogram, calibration_ecdf (SBC), z_score_contraction

Custom
  plot_posterior              - VP10 marginal posteriors vs prior
  posterior_predictive_check  - simulated vs real summary statistics, now
                                including skip rate and refixation rate
                                (Engbert & Rabe reference statistics)

SBC reference: Talts et al. (2018), arXiv:1804.06788
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from swift.config import FIG_DIR, M_SENTENCES
from swift.simulator import (
    PARAM_NAMES, THETA_MAX, THETA_MIN, SWIFTSimulator, sample_prior,
)

PARAM_LABELS = {
    "t_sac": r"$t_{sac}$ (ms)",
    "eta": r"$\eta$",
    "delta0": r"$\delta_0$ (chars)",
    "R": r"$R$",
}


# ---------------------------------------------------------------------------
# BayesFlow v2 built-in diagnostics
# ---------------------------------------------------------------------------

def run_builtin_diagnostics(workflow, simulator, n_val: int = 300,
                            n_posterior_samples: int = 1000,
                            save_dir=FIG_DIR) -> dict:
    import bayesflow as bf

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_val} validation readers...")
    val_sims = simulator.sample(n_val)
    print(f"Drawing {n_posterior_samples} posterior samples per reader...")
    post = workflow.sample(conditions=val_sims, num_samples=n_posterior_samples)

    for fn, name in [
        (lambda: bf.diagnostics.plots.recovery(
            estimates=post, targets=val_sims, variable_names=PARAM_NAMES),
         "recovery_plot.png"),
        (lambda: bf.diagnostics.plots.calibration_histogram(
            estimates=post, targets=val_sims, variable_names=PARAM_NAMES),
         "sbc_histogram.png"),
        (lambda: bf.diagnostics.plots.calibration_ecdf(
            estimates=post, targets=val_sims, variable_names=PARAM_NAMES,
            difference=True, rank_type="distance"),
         "sbc_ecdf.png"),
        (lambda: bf.diagnostics.plots.z_score_contraction(
            estimates=post, targets=val_sims, variable_names=PARAM_NAMES),
         "contraction_plot.png"),
    ]:
        print(f"  plotting {name} ...")
        _save(fn(), save_dir, name)

    print(f"Built-in diagnostics saved to {save_dir}/")
    return {"val_sims": val_sims, "post_draws": post}


# ---------------------------------------------------------------------------
# VP10 marginal posteriors
# ---------------------------------------------------------------------------

def plot_posterior(posterior_samples: np.ndarray, true_params: dict = None,
                   save_dir=FIG_DIR) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Posterior Distributions - SWIFT Parameters (VP10)\n"
                 "Orange = posterior  |  Gray = prior range", fontsize=12)

    for j, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        lo, hi = float(THETA_MIN[j]), float(THETA_MAX[j])
        ax.axvspan(lo, hi, alpha=0.12, color="gray", label="Prior (uniform)")
        ax.hist(posterior_samples[:, j], bins=60, range=(lo, hi), density=True,
                color="darkorange", alpha=0.85, edgecolor="white", label="Posterior")
        if true_params and name in true_params:
            ax.axvline(true_params[name], color="green", lw=2, ls="--",
                       label=f"True: {true_params[name]:.2f}")
        mean = float(posterior_samples[:, j].mean())
        lo95 = float(np.percentile(posterior_samples[:, j], 2.5))
        hi95 = float(np.percentile(posterior_samples[:, j], 97.5))
        ax.axvline(mean, color="red", lw=1.8, label=f"Mean: {mean:.2f}")
        ax.axvspan(lo95, hi95, alpha=0.18, color="red", label="95% CI")
        ax.set_xlabel(PARAM_LABELS[name]); ax.set_title(name)
        ax.set_ylabel("Density" if j == 0 else ""); ax.set_xlim(lo, hi)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = save_dir / "posterior_VP10.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Posterior plot saved -> {out}")


# ---------------------------------------------------------------------------
# Posterior predictive check
# ---------------------------------------------------------------------------

def _seq_stats(fixations):
    """(durations, landings, count, skip_rate, refix_rate) for one sentence."""
    if len(fixations) == 0:
        return [], [], 0, np.nan, np.nan
    arr = np.array(fixations)
    w = arr[:, 0].astype(int)
    mx = w.max()
    skip = sum(1 for k in range(1, mx + 1) if k not in set(w)) / mx
    refix = (sum(1 for i in range(len(w) - 1) if w[i] == w[i + 1])
             / max(len(w) - 1, 1))
    return arr[:, 2].tolist(), arr[:, 1].tolist(), len(w), skip, refix


def posterior_predictive_check(posterior_samples, real_fix_df,
                               word_lengths_list, word_freqs_list,
                               n_ppc: int = 300, save_dir=FIG_DIR,
                               rng: np.random.Generator = None) -> None:
    if rng is None:
        rng = np.random.default_rng(42)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    n_sent = len(word_lengths_list)

    sim_dur, sim_lnd, sim_fps, sim_skip, sim_refix = [], [], [], [], []
    idx = rng.choice(len(posterior_samples), size=min(n_ppc, len(posterior_samples)),
                     replace=False)
    print(f"Running PPC with {len(idx)} posterior draws x {M_SENTENCES} sentences...")
    for theta in posterior_samples[idx]:
        params = dict(zip(PARAM_NAMES, theta.tolist()))
        sim = SWIFTSimulator(params)
        for si in rng.integers(0, n_sent, size=M_SENTENCES):
            d, l, c, sk, rf = _seq_stats(
                sim.simulate_sentence(word_lengths_list[si], word_freqs_list[si], rng=rng))
            if c > 0:
                sim_dur += d; sim_lnd += l; sim_fps.append(c)
                sim_skip.append(sk); sim_refix.append(rf)

    real_dur = real_fix_df["fixation_duration"].values
    real_lnd = real_fix_df["landing_position"].values
    real_fps = real_fix_df.groupby("sentence_id").size().values
    from swift.data import skip_rate, refix_rate
    real_skip, real_refix = skip_rate(real_fix_df), refix_rate(real_fix_df)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    fig.suptitle("Posterior Predictive Check   "
                 "Blue = Real VP10   |   Orange = Simulated from posterior",
                 fontsize=12)
    _panel(axes[0], real_dur, sim_dur, "Fixation Duration (ms)", 40)
    _panel(axes[1], real_lnd, sim_lnd, "Landing Position (chars)", 30)
    _panel(axes[2], real_fps, sim_fps, "Fixations per Sentence", 15)

    axes[3].bar([0, 1], [real_skip * 100, real_refix * 100], width=0.35,
                color="steelblue", label="Real")
    axes[3].bar([0.4, 1.4], [np.nanmean(sim_skip) * 100, np.nanmean(sim_refix) * 100],
                width=0.35, color="darkorange", label="Simulated")
    axes[3].set_xticks([0.2, 1.2]); axes[3].set_xticklabels(["Skip", "Refixation"])
    axes[3].set_ylabel("Rate (%)"); axes[3].set_title("Skip / Refixation Rate")
    axes[3].legend(fontsize=9)

    plt.tight_layout()
    out = save_dir / "ppc_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"PPC plot saved -> {out}")

    print("\n===== PPC SUMMARY =====")
    print(f"{'Statistic':<26}{'Real':>10}{'Simulated':>12}")
    print("-" * 48)
    for name, rv, sv in [
        ("Mean duration (ms)", np.mean(real_dur), np.mean(sim_dur)),
        ("Std  duration (ms)", np.std(real_dur), np.std(sim_dur)),
        ("Mean fixations/sent", np.mean(real_fps), np.mean(sim_fps)),
        ("Mean landing pos", np.mean(real_lnd), np.mean(sim_lnd)),
        ("Skip rate (%)", real_skip * 100, np.nanmean(sim_skip) * 100),
        ("Refixation rate (%)", real_refix * 100, np.nanmean(sim_refix) * 100),
    ]:
        print(f"{name:<26}{rv:>10.2f}{sv:>12.2f}")


def _panel(ax, real, sim, xlabel, bins):
    ax.hist(real, bins=bins, density=True, alpha=0.6, color="steelblue",
            label="Real", edgecolor="white")
    ax.hist(sim, bins=bins, density=True, alpha=0.6, color="darkorange",
            label="Simulated", edgecolor="white")
    ax.set_xlabel(xlabel); ax.set_ylabel("Density"); ax.legend(fontsize=9)


def _save(fig, save_dir, filename):
    path = Path(save_dir) / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")
