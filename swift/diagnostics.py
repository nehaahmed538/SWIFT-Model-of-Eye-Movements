"""
diagnostics.py
==============
Diagnostics for the trained BayesFlow simplified-SWIFT model (Engbert & Rabe
2024, basic 3-parameter version: nu, r, mu_T).

BayesFlow v2 built-ins
  recovery, calibration_histogram, calibration_ecdf (SBC), z_score_contraction

Custom
  plot_posterior              - VP10 marginal posteriors vs prior
  plot_span_shape             - reproduction of the paper's Fig. 2 (span vs nu)
  plot_scanpath_examples      - reproduction of the paper's Fig. 4 (scanpaths)
  plot_posterior_correlation  - decoupling check (Section 4.1): mu_T ⊥ (nu, r)
  posterior_predictive_check  - the paper's six measures (Section 6 / Fig. 9):
                                SFD, GD, TT, P(skip), P(refix), P(regression)

SBC reference: Talts et al. (2018), arXiv:1804.06788
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from swift.config import FIG_DIR, M_SENTENCES
from swift.simulator import (
    PARAM_NAMES, THETA_MAX, THETA_MIN, SWIFTSimulator, simulate_sentence,
    span_rates,
)

PARAM_LABELS = {
    "nu": r"$\nu$",
    "r": r"$r$",
    "mu_T": r"$\mu_T$ (ms)",
}
PARAM_UNITS = {"nu": "", "r": "", "mu_T": "ms"}


# ---------------------------------------------------------------------------
# BayesFlow v2 built-in diagnostics
# ---------------------------------------------------------------------------

def run_builtin_diagnostics(workflow, simulator, n_val: int = 300,
                            n_posterior_samples: int = 1000,
                            save_dir=FIG_DIR) -> dict:
    import bayesflow as bf

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Paper reproductions that need no model.
    plot_span_shape(save_dir)
    plot_scanpath_examples(save_dir=save_dir)

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
# Paper Fig. 2 — processing span shape vs nu
# ---------------------------------------------------------------------------

def plot_span_shape(save_dir=FIG_DIR) -> None:
    save_dir = Path(save_dir)
    nus = np.linspace(0.01, 1.0, 100)
    # interior word (k=3, N=10) so the full 4-point span is present
    lam = np.array([span_rates(3, 10, nu)[[2, 3, 4, 5]] for nu in nus])
    labels = [r"$\lambda_{-1}$", r"$\lambda_{0}$ (fixated)",
              r"$\lambda_{+1}$", r"$\lambda_{+2}$"]
    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for j in range(4):
        ax.plot(nus, lam[:, j], label=labels[j], color=colors[j], lw=2)
    ax.set_xlabel(r"$\nu$"); ax.set_ylabel("processing rate weight $\\lambda_w$")
    ax.set_title("Processing span vs $\\nu$  (Engbert & Rabe 2024, Fig. 2)")
    ax.legend(); ax.set_xlim(0, 1)
    _save(fig, save_dir, "span_shape.png")


# ---------------------------------------------------------------------------
# Paper Fig. 4 — example scanpaths
# ---------------------------------------------------------------------------

def plot_scanpath_examples(word_freqs=None, save_dir=FIG_DIR,
                           nu=0.3, r=10.0, mu_T=200.0, n_examples=4,
                           seed=1) -> None:
    save_dir = Path(save_dir)
    rng = np.random.default_rng(seed)
    if word_freqs is None:
        word_freqs = np.array([100, 500, 20, 300, 80, 400, 15, 200, 350, 60],
                              dtype=float)
    N = len(word_freqs)

    fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 3.6),
                             sharey=True)
    fig.suptitle(f"Example scanpaths  ($\\nu$={nu}, $r$={r}, $\\mu_T$={mu_T} ms)  "
                 "— Engbert & Rabe 2024, Fig. 4", fontsize=12)
    for ax in np.atleast_1d(axes):
        fix = simulate_sentence(word_freqs, nu, r, mu_T, rng=rng)
        words = [w for w, _ in fix]
        ax.plot(range(1, len(words) + 1), words, "-o", color="steelblue")
        ax.set_xlabel("fixation number"); ax.set_yticks(range(1, N + 1))
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("word position")
    _save(fig, save_dir, "scanpath_examples.png")


# ---------------------------------------------------------------------------
# Decoupling check (Section 4.1): mu_T should be ~uncorrelated with nu, r
# ---------------------------------------------------------------------------

def plot_posterior_correlation(posterior_samples: np.ndarray,
                               save_dir=FIG_DIR) -> np.ndarray:
    save_dir = Path(save_dir)
    C = np.corrcoef(np.asarray(posterior_samples).T)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(PARAM_NAMES))); ax.set_yticks(range(len(PARAM_NAMES)))
    ax.set_xticklabels([PARAM_LABELS[p] for p in PARAM_NAMES])
    ax.set_yticklabels([PARAM_LABELS[p] for p in PARAM_NAMES])
    for i in range(len(PARAM_NAMES)):
        for j in range(len(PARAM_NAMES)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    color="black")
    ax.set_title("VP10 posterior correlation\n(basic model: $\\mu_T \\perp \\nu, r$)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, save_dir, "posterior_correlation.png")

    print("\n===== POSTERIOR CORRELATION (decoupling check) =====")
    print("            " + "".join(f"{p:>8}" for p in PARAM_NAMES))
    for i, p in enumerate(PARAM_NAMES):
        print(f"{p:<8}" + "".join(f"{C[i, j]:>8.3f}" for j in range(len(PARAM_NAMES))))
    mu_idx = PARAM_NAMES.index("mu_T")
    off = [C[mu_idx, j] for j in range(len(PARAM_NAMES)) if j != mu_idx]
    print(f"mu_T vs (nu, r) correlations: {off[0]:.3f}, {off[1]:.3f}  "
          f"(expected ~0 -- the basic model decouples timing from scanpath)")
    return C


# ---------------------------------------------------------------------------
# VP10 marginal posteriors
# ---------------------------------------------------------------------------

def plot_posterior(posterior_samples: np.ndarray, true_params: dict = None,
                   save_dir=FIG_DIR) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(PARAM_NAMES), figsize=(4 * len(PARAM_NAMES), 4))
    fig.suptitle("Posterior Distributions - simplified SWIFT (VP10)\n"
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
# Word-level reading measures (paper Section 6 / Fig. 9)
# ---------------------------------------------------------------------------

def _sentence_measures(words, durs, n_words):
    """Per-sentence reading measures from a temporally-ordered fixation list.

    words : 1-indexed fixated word positions, durs : durations (ms).
    Returns dict with per-word lists (sfd, gd, tt) and saccade counts
    (n_words, n_skipped, n_refix, n_regr, n_sacc)."""
    words = [int(w) for w in words]
    durs = [float(d) for d in durs]
    from collections import Counter
    cnt = Counter(words)

    tt = {}
    for w, d in zip(words, durs):
        tt[w] = tt.get(w, 0.0) + d

    # first-pass gaze duration: consecutive run at a word's first entry
    gd, seen, j = {}, set(), 0
    while j < len(words):
        w = words[j]
        if w not in seen:
            run, k = durs[j], j + 1
            while k < len(words) and words[k] == w:
                run += durs[k]; k += 1
            gd[w] = run; seen.add(w); j = k
        else:
            j += 1

    # single-fixation duration: words fixated exactly once (whole trial)
    sfd = [durs[i] for i in range(len(words)) if cnt[words[i]] == 1]
    skipped = sum(1 for w in range(1, n_words + 1) if w not in cnt)
    refix = sum(1 for a, b in zip(words, words[1:]) if a == b)
    regr = sum(1 for a, b in zip(words, words[1:]) if b < a)
    n_sacc = max(len(words) - 1, 0)
    return dict(sfd=sfd, gd=list(gd.values()), tt=list(tt.values()),
                n_words=n_words, n_skipped=skipped,
                n_refix=refix, n_regr=regr, n_sacc=n_sacc)


def _aggregate(records):
    """Pool per-sentence measure dicts into distributions + scalar rates."""
    sfd, gd, tt = [], [], []
    nwords = nskip = nrefix = nregr = nsacc = 0
    for m in records:
        sfd += m["sfd"]; gd += m["gd"]; tt += m["tt"]
        nwords += m["n_words"]; nskip += m["n_skipped"]
        nrefix += m["n_refix"]; nregr += m["n_regr"]; nsacc += m["n_sacc"]
    return {
        "sfd": np.array(sfd), "gd": np.array(gd), "tt": np.array(tt),
        "p_skip": nskip / nwords if nwords else np.nan,
        "p_refix": nrefix / nsacc if nsacc else np.nan,
        "p_regr": nregr / nsacc if nsacc else np.nan,
    }


# ---------------------------------------------------------------------------
# Posterior predictive check
# ---------------------------------------------------------------------------

def posterior_predictive_check(posterior_samples, real_fix_df,
                               word_freqs_list,
                               n_ppc: int = 300, save_dir=FIG_DIR,
                               rng: np.random.Generator = None,
                               sentence_indices=None) -> None:
    """Compare the paper's six reading measures between real VP10 fixations and
    data simulated from the posterior. ``sentence_indices`` (0-based into
    ``word_freqs_list``) restricts simulation to a sentence subset — pass the
    held-out test split so the PPC mirrors the paper's Section 6."""
    if rng is None:
        rng = np.random.default_rng(42)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sent_idx = (list(sentence_indices) if sentence_indices is not None
                else list(range(len(word_freqs_list))))

    # --- real measures ---
    real_records = []
    for _, g in real_fix_df.groupby("sentence_id"):
        g = g.sort_values("fixation_index")
        w = g["word_id"].values
        real_records.append(_sentence_measures(w, g["fixation_duration"].values,
                                                int(w.max())))
    real = _aggregate(real_records)

    # --- simulated measures from the posterior ---
    idx = rng.choice(len(posterior_samples),
                     size=min(n_ppc, len(posterior_samples)), replace=False)
    print(f"Running PPC with {len(idx)} posterior draws x {M_SENTENCES} sentences "
          f"(from {len(sent_idx)} test sentences)...")
    sim_records = []
    for theta in posterior_samples[idx]:
        params = dict(zip(PARAM_NAMES, theta.tolist()))
        sim = SWIFTSimulator(params)
        for si in rng.choice(sent_idx, size=M_SENTENCES, replace=True):
            fix = sim.simulate_sentence(word_freqs_list[si], rng=rng)
            if not fix:
                continue
            words = [w for w, _ in fix]
            sim_records.append(_sentence_measures(
                words, [d for _, d in fix], max(words)))
    sim = _aggregate(sim_records)

    # --- plot: 3 duration-measure histograms + 1 probability bar chart ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle("Posterior Predictive Check   "
                 "Blue = Real VP10   |   Orange = Simulated from posterior",
                 fontsize=12)
    _panel(axes[0], real["sfd"], sim["sfd"], "Single-fixation duration (ms)", 30)
    _panel(axes[1], real["gd"], sim["gd"], "Gaze duration (ms)", 30)
    _panel(axes[2], real["tt"], sim["tt"], "Total fixation time (ms)", 30)

    labels = ["P(skip)", "P(refix)", "P(regr)"]
    real_p = [real["p_skip"], real["p_refix"], real["p_regr"]]
    sim_p = [sim["p_skip"], sim["p_refix"], sim["p_regr"]]
    x = np.arange(3)
    axes[3].bar(x - 0.2, np.array(real_p) * 100, width=0.4,
                color="steelblue", label="Real")
    axes[3].bar(x + 0.2, np.array(sim_p) * 100, width=0.4,
                color="darkorange", label="Simulated")
    axes[3].set_xticks(x); axes[3].set_xticklabels(labels)
    axes[3].set_ylabel("Probability (%)"); axes[3].set_title("Skip / Refix / Regression")
    axes[3].legend(fontsize=9)

    plt.tight_layout()
    out = save_dir / "ppc_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"PPC plot saved -> {out}")

    print("\n===== PPC SUMMARY =====")
    print(f"{'Statistic':<26}{'Real':>10}{'Simulated':>12}")
    print("-" * 48)
    for name, rv, sv in [
        ("Mean SFD (ms)", np.mean(real["sfd"]), np.mean(sim["sfd"])),
        ("Mean GD (ms)", np.mean(real["gd"]), np.mean(sim["gd"])),
        ("Mean TT (ms)", np.mean(real["tt"]), np.mean(sim["tt"])),
        ("P(skip) (%)", real["p_skip"] * 100, sim["p_skip"] * 100),
        ("P(refixation) (%)", real["p_refix"] * 100, sim["p_refix"] * 100),
        ("P(regression) (%)", real["p_regr"] * 100, sim["p_regr"] * 100),
    ]:
        print(f"{name:<26}{rv:>10.2f}{sv:>12.2f}")


def _panel(ax, real, sim, xlabel, bins):
    if len(real):
        ax.hist(real, bins=bins, density=True, alpha=0.6, color="steelblue",
                label="Real", edgecolor="white")
    if len(sim):
        ax.hist(sim, bins=bins, density=True, alpha=0.6, color="darkorange",
                label="Simulated", edgecolor="white")
    ax.set_xlabel(xlabel); ax.set_ylabel("Density"); ax.legend(fontsize=9)


def _save(fig, save_dir, filename):
    path = Path(save_dir) / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")
