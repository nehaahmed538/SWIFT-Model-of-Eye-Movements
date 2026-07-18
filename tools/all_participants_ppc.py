"""
tools/all_participants_ppc.py
==============================
Run the already-trained model against ALL real participants (not just VP10),
to compute a cross-participant posterior-predictive correlation comparable to
Engbert & Rabe (2024) Fig. 9 / Section 6.

Data: data/vp_all/fixseqin_PB2expVP*.dat -- the same OSF repository the paper
itself links to (https://osf.io/8wrf6/, folder
R-Code-parameter-estimation-from-experimental-data/expdata/), containing all
34 participants from Risse & Seelig (2019). The project's original VP10 file
is one of these 34.

Does NOT retrain: reuses outputs/models/swift_approximator.keras. For each
participant: run amortised inference (posterior over nu, r, mu_T from their
first-half sentences), then posterior-predictive-simulate on their held-out
second-half sentences and compare simulated vs. real reading measures (SFD,
GD, TT, skip/refixation/regression rates) -- exactly the paper's Section 6
procedure, just run per participant instead of only VP10.

Sample sizes are reduced relative to tools/show_results.py's VP10-only
defaults (which pool 40 reader-draws x 2000 posterior samples for ONE
person) since this now repeats the whole pipeline 34 times; see N_READERS /
N_SAMPLES / N_PPC below. This trades some per-participant precision for
feasible total runtime -- the cross-participant correlation itself is what's
new here, not high-precision single-participant estimates (already covered
for VP10 in show_results.py).

Run:
    python tools/all_participants_ppc.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swift import config

VP_DIR = config.DATA_DIR / "vp_all"
N_READERS = 20    # pooled 14-sentence reader-draws per participant (VP10-only default: 40)
N_SAMPLES = 1000  # posterior samples per reader-draw (VP10-only default: 2000)
N_PPC = 150       # posterior draws re-simulated for the PPC comparison (VP10-only default: 300)
SEED = 0


def vp_number(path: Path) -> int:
    import re
    m = re.search(r"VP(\d+)", path.stem)
    if not m:
        raise ValueError(f"Could not parse participant number from {path.name}")
    return int(m.group(1))


def main() -> None:
    from swift.data import (
        build_corpus_lists, build_reader_batch, load_corpus, load_fixations,
        split_half,
    )
    from swift.diagnostics import PARAM_LABELS, _aggregate, _panel, _sentence_measures
    from swift.inference import rebuild_workflow, run_inference, set_corpus
    from swift.simulator import PARAM_NAMES, SWIFTSimulator, THETA_MAX, THETA_MIN

    t0 = time.time()

    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model at {config.MODEL_PATH}\n"
            f"Train one first:  python main.py --mode train   (or --mode all)")

    wfl = build_corpus_lists(load_corpus(config.CORPUS_PATH))
    set_corpus(wfl)
    workflow = rebuild_workflow(config.MODEL_PATH)

    vp_files = sorted(VP_DIR.glob("fixseqin_PB2expVP*.dat"), key=vp_number)
    print(f"\nFound {len(vp_files)} participant files in {VP_DIR}")
    if not vp_files:
        raise FileNotFoundError(f"No participant files found in {VP_DIR}")

    from scipy.stats import gaussian_kde

    N_GRID = 200
    grids = [np.linspace(THETA_MIN[j], THETA_MAX[j], N_GRID)
            for j in range(len(PARAM_NAMES))]
    densities = {name: [] for name in PARAM_NAMES}  # per-param list of (N_GRID,) curves, one per participant

    results = []
    pooled_real_records = []
    pooled_sim_records = []
    pooled_posterior_samples = []
    for i, path in enumerate(vp_files, 1):
        vp_id = vp_number(path)
        t_p = time.time()

        fix = load_fixations(path)
        train_ids, test_ids = split_half(fix)
        rng = np.random.default_rng(SEED)

        observations, obs_stats = build_reader_batch(
            fix, m_sentences=config.M_SENTENCES, n_readers=N_READERS,
            rng=rng, sentence_ids=train_ids)

        with contextlib.redirect_stdout(io.StringIO()):
            posterior = run_inference(workflow, observations, obs_stats,
                                      num_samples=N_SAMPLES)

        theta_mean = {name: float(posterior[:, j].mean())
                     for j, name in enumerate(PARAM_NAMES)}
        pooled_posterior_samples.append(posterior)

        for j, name in enumerate(PARAM_NAMES):
            col = posterior[:, j]
            if np.std(col) > 1e-6:
                kde = gaussian_kde(col)
                densities[name].append(kde(grids[j]))
            else:
                densities[name].append(np.zeros(N_GRID))

        # --- real measures on held-out (test-half) sentences ---
        real_test = fix[fix["sentence_id"].isin(test_ids)]
        real_records = []
        for _, g in real_test.groupby("sentence_id"):
            g = g.sort_values("fixation_index")
            w = g["word_id"].values
            real_records.append(_sentence_measures(
                w, g["fixation_duration"].values, int(w.max())))
        real = _aggregate(real_records)

        # --- simulated measures from the posterior, replayed on the SAME
        # held-out test sentences ---
        test_idx = [int(s) - 1 for s in test_ids]
        rng2 = np.random.default_rng(SEED + 1)
        n_draw = min(N_PPC, len(posterior))
        idx = rng2.choice(len(posterior), size=n_draw, replace=False)
        sim_records = []
        for theta in posterior[idx]:
            params = dict(zip(PARAM_NAMES, theta.tolist()))
            sim = SWIFTSimulator(params)
            for si in rng2.choice(test_idx, size=config.M_SENTENCES, replace=True):
                fx = sim.simulate_sentence(wfl[si], rng=rng2)
                if not fx:
                    continue
                words = [w for w, _ in fx]
                sim_records.append(_sentence_measures(
                    words, [d for _, d in fx], max(words)))
        sim = _aggregate(sim_records)

        pooled_real_records.extend(real_records)
        pooled_sim_records.extend(sim_records)

        row = {
            "vp_id": vp_id,
            "n_fixations": int(len(fix)),
            "n_sentences": int(fix["sentence_id"].nunique()),
            "theta_nu": theta_mean["nu"],
            "theta_r": theta_mean["r"],
            "theta_mu_T": theta_mean["mu_T"],
            "real_sfd": float(np.mean(real["sfd"])) if len(real["sfd"]) else float("nan"),
            "sim_sfd": float(np.mean(sim["sfd"])) if len(sim["sfd"]) else float("nan"),
            "real_gd": float(np.mean(real["gd"])) if len(real["gd"]) else float("nan"),
            "sim_gd": float(np.mean(sim["gd"])) if len(sim["gd"]) else float("nan"),
            "real_tt": float(np.mean(real["tt"])) if len(real["tt"]) else float("nan"),
            "sim_tt": float(np.mean(sim["tt"])) if len(sim["tt"]) else float("nan"),
            "real_skip": float(real["p_skip"]), "sim_skip": float(sim["p_skip"]),
            "real_refix": float(real["p_refix"]), "sim_refix": float(sim["p_refix"]),
            "real_regr": float(real["p_regr"]), "sim_regr": float(sim["p_regr"]),
        }
        results.append(row)
        dt = time.time() - t_p
        print(f"[{i:>2}/{len(vp_files)}] VP{vp_id:<3} "
              f"nu={theta_mean['nu']:.2f} r={theta_mean['r']:.2f} "
              f"mu_T={theta_mean['mu_T']:.0f}ms  "
              f"({len(fix)} fix, {dt:.1f}s)")

    out_json = config.OUT_DIR / "all_participants_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved per-participant results -> {out_json}")

    # ------------------------------------------------------------------
    # Cross-participant correlation (paper's Fig. 9 / Section 6 equivalent)
    # ------------------------------------------------------------------
    measures = [
        ("sfd", "Single-fixation duration (ms)"),
        ("gd", "Gaze duration (ms)"),
        ("tt", "Total fixation time (ms)"),
        ("skip", "P(skip)"),
        ("refix", "P(refixation)"),
        ("regr", "P(regression)"),
    ]
    print("\n" + "=" * 66)
    print(f"CROSS-PARTICIPANT CORRELATION  (n={len(results)} real participants)")
    print("cf. Engbert & Rabe (2024) Fig. 9 -- same idea, different inference method")
    print("=" * 66)
    corr_summary = {}
    for key, label in measures:
        real_vals = np.array([r[f"real_{key}"] for r in results])
        sim_vals = np.array([r[f"sim_{key}"] for r in results])
        mask = ~np.isnan(real_vals) & ~np.isnan(sim_vals)
        r_corr = (float(np.corrcoef(real_vals[mask], sim_vals[mask])[0, 1])
                 if mask.sum() > 2 else float("nan"))
        corr_summary[key] = r_corr
        print(f"  {label:<32} r = {r_corr:6.3f}   (n={int(mask.sum())})")

    with open(config.OUT_DIR / "cross_participant_correlations.json", "w") as f:
        json.dump(corr_summary, f, indent=2)

    # ------------------------------------------------------------------
    # Plot: 6-panel scatter, real (x) vs simulated (y) -- paper's Fig. 9 layout
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Posterior predictive check across all {len(results)} real participants\n"
        "(cf. Engbert & Rabe 2024, Fig. 9 -- BayesFlow amortised inference, "
        "not MCMC)", fontsize=13)
    for ax, (key, label) in zip(axes.flat, measures):
        real_vals = np.array([r[f"real_{key}"] for r in results])
        sim_vals = np.array([r[f"sim_{key}"] for r in results])
        mask = ~np.isnan(real_vals) & ~np.isnan(sim_vals)
        ax.scatter(real_vals[mask], sim_vals[mask], color="steelblue",
                  edgecolor="white", s=55, zorder=3)
        lo = min(real_vals[mask].min(), sim_vals[mask].min())
        hi = max(real_vals[mask].max(), sim_vals[mask].max())
        pad = (hi - lo) * 0.08 if hi > lo else 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.5, zorder=1)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"{label} -- real"); ax.set_ylabel(f"{label} -- simulated")
        ax.set_title(f"{label}\nr = {corr_summary[key]:.2f}")
        ax.grid(alpha=0.25)
    plt.tight_layout()
    out_fig = config.FIG_DIR / "all_participants_ppc.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved -> {out_fig}")

    # ------------------------------------------------------------------
    # Pooled population-level PPC plot -- same histogram+bar layout as
    # VP10's own outputs/figures/ppc_plot.png, but pooling all 34
    # participants' real and simulated fixations together (instead of the
    # scatter-of-per-participant-means plot above).
    # ------------------------------------------------------------------
    pooled_real = _aggregate(pooled_real_records)
    pooled_sim = _aggregate(pooled_sim_records)

    fig2, axes2 = plt.subplots(1, 4, figsize=(20, 4.5))
    fig2.suptitle(
        f"Posterior Predictive Check -- pooled across all {len(results)} real "
        "participants\nBlue = Real   |   Orange = Simulated from posterior",
        fontsize=12)
    _panel(axes2[0], pooled_real["sfd"], pooled_sim["sfd"],
          "Single-fixation duration (ms)", 40)
    _panel(axes2[1], pooled_real["gd"], pooled_sim["gd"],
          "Gaze duration (ms)", 40)
    _panel(axes2[2], pooled_real["tt"], pooled_sim["tt"],
          "Total fixation time (ms)", 40)

    labels = ["P(skip)", "P(refix)", "P(regr)"]
    real_p = [pooled_real["p_skip"], pooled_real["p_refix"], pooled_real["p_regr"]]
    sim_p = [pooled_sim["p_skip"], pooled_sim["p_refix"], pooled_sim["p_regr"]]
    x = np.arange(3)
    axes2[3].bar(x - 0.2, np.array(real_p) * 100, width=0.4,
                color="steelblue", label="Real")
    axes2[3].bar(x + 0.2, np.array(sim_p) * 100, width=0.4,
                color="darkorange", label="Simulated")
    axes2[3].set_xticks(x); axes2[3].set_xticklabels(labels)
    axes2[3].set_ylabel("Probability (%)"); axes2[3].set_title("Skip / Refix / Regression")
    axes2[3].legend(fontsize=9)

    plt.tight_layout()
    out_fig2 = config.FIG_DIR / "all_participants_ppc_pooled.png"
    plt.savefig(out_fig2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Pooled PPC figure saved -> {out_fig2}")

    print("\n===== POOLED PPC SUMMARY (all 34 participants) =====")
    print(f"{'Statistic':<26}{'Real':>10}{'Simulated':>12}")
    print("-" * 48)
    for name, rv, sv in [
        ("Mean SFD (ms)", np.mean(pooled_real["sfd"]), np.mean(pooled_sim["sfd"])),
        ("Mean GD (ms)", np.mean(pooled_real["gd"]), np.mean(pooled_sim["gd"])),
        ("Mean TT (ms)", np.mean(pooled_real["tt"]), np.mean(pooled_sim["tt"])),
        ("P(skip) (%)", pooled_real["p_skip"] * 100, pooled_sim["p_skip"] * 100),
        ("P(refixation) (%)", pooled_real["p_refix"] * 100, pooled_sim["p_refix"] * 100),
        ("P(regression) (%)", pooled_real["p_regr"] * 100, pooled_sim["p_regr"] * 100),
    ]:
        print(f"{name:<26}{rv:>10.2f}{sv:>12.2f}")

    # ------------------------------------------------------------------
    # All-participant posterior densities -- cf. paper's Fig. 8 (marginal
    # posterior per participant, gray lines + mean blue line), applied to
    # our 3 free parameters instead of the paper's 5.
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, len(PARAM_NAMES), figsize=(5 * len(PARAM_NAMES), 4.2))
    fig3.suptitle(
        f"Posterior densities for all {len(results)} real participants\n"
        "(cf. Engbert & Rabe 2024, Fig. 8 -- gray = individual participant, "
        "blue = mean)", fontsize=13)
    for j, (ax, name) in enumerate(zip(axes3, PARAM_NAMES)):
        curves = np.array(densities[name])  # (n_participants, N_GRID)
        for curve in curves:
            ax.plot(grids[j], curve, color="gray", alpha=0.35, lw=1)
        ax.plot(grids[j], curves.mean(axis=0), color="steelblue", lw=2.5,
                label="Mean across participants")
        ax.set_xlabel(PARAM_LABELS[name]); ax.set_ylabel("Density" if j == 0 else "")
        ax.set_title(name); ax.set_xlim(THETA_MIN[j], THETA_MAX[j])
        ax.legend(fontsize=8)
    plt.tight_layout()
    out_fig3 = config.FIG_DIR / "all_participants_posteriors.png"
    plt.savefig(out_fig3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"All-participant posterior densities saved -> {out_fig3}")

    # ------------------------------------------------------------------
    # 1. Pooled decoupling check -- same computation as posterior_correlation.png
    # (VP10 only), but pooling ALL 34 participants' posterior samples together.
    # ------------------------------------------------------------------
    all_post = np.concatenate(pooled_posterior_samples, axis=0)
    C_pooled = np.corrcoef(all_post.T)

    def _corr_heatmap(C, title, out_path):
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(PARAM_NAMES))); ax.set_yticks(range(len(PARAM_NAMES)))
        ax.set_xticklabels([PARAM_LABELS[p] for p in PARAM_NAMES])
        ax.set_yticklabels([PARAM_LABELS[p] for p in PARAM_NAMES])
        for a in range(len(PARAM_NAMES)):
            for b in range(len(PARAM_NAMES)):
                ax.text(b, a, f"{C[a, b]:.2f}", ha="center", va="center", color="black")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {out_path}")

    _corr_heatmap(
        C_pooled,
        f"Pooled posterior correlation, all {len(results)} participants\n"
        f"({len(all_post):,} samples; decoupling check at scale)",
        config.FIG_DIR / "all_participants_posterior_correlation.png")

    print("\n===== POOLED DECOUPLING CHECK (all 34 participants) =====")
    print("            " + "".join(f"{p:>8}" for p in PARAM_NAMES))
    for a, p in enumerate(PARAM_NAMES):
        print(f"{p:<8}" + "".join(f"{C_pooled[a, b]:>8.3f}" for b in range(len(PARAM_NAMES))))

    # ------------------------------------------------------------------
    # 2. Cross-participant point-estimate correlation -- across the 34
    # PEOPLE, does someone with high nu also tend to have high r? (individual
    # differences, not within-person posterior uncertainty)
    # ------------------------------------------------------------------
    point_est = np.array([[r["theta_nu"], r["theta_r"], r["theta_mu_T"]] for r in results])
    C_points = np.corrcoef(point_est.T)
    _corr_heatmap(
        C_points,
        f"Cross-participant correlation of point estimates\n"
        f"(n={len(results)} people; individual differences)",
        config.FIG_DIR / "all_participants_theta_correlation.png")

    print("\n===== CROSS-PARTICIPANT THETA CORRELATION (individual differences) =====")
    print("            " + "".join(f"{p:>8}" for p in PARAM_NAMES))
    for a, p in enumerate(PARAM_NAMES):
        print(f"{p:<8}" + "".join(f"{C_points[a, b]:>8.3f}" for b in range(len(PARAM_NAMES))))

    dt_total = time.time() - t0
    print(f"\nTotal runtime: {dt_total:.1f}s ({dt_total / 60:.1f} min)")


if __name__ == "__main__":
    main()
