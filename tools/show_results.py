"""
tools/show_results.py
======================
Read-only report of the CURRENTLY SAVED model's results. Loads
outputs/models/swift_approximator.keras and prints a full text report:

  1. Real VP10 data summary
  2. Parameter recovery + SBC calibration (on fresh held-out simulations)
  3. VP10 posterior estimates
  4. Posterior predictive check (PPC) summary table

It does NOT train anything, so it only takes about half a minute on a
laptop CPU (a few hundred quick NumPy simulations + posterior sampling from
the already-trained network). Every number is recomputed live from the saved
model and the real data file, so this always reflects the model actually on
disk right now, not a cached claim from a report or a chat log.

Run:
    python tools/show_results.py                 # full report (~30s)
    python tools/show_results.py --quick          # smoke-test-sized (~5s)
    python tools/show_results.py --n_ppc 1000     # more PPC simulations

Requires a trained model at outputs/models/swift_approximator.keras
(produced by `python main.py --mode train` or `--mode all`).
"""

from __future__ import annotations

import argparse
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_val", type=int, default=300,
                   help="Held-out validation readers for recovery/SBC (default 300)")
    p.add_argument("--n_posterior_samples", type=int, default=1000,
                   help="Posterior draws per validation reader (default 1000)")
    p.add_argument("--n_vp10_readers", type=int, default=40,
                   help="Random 14-sentence VP10 draws to pool for the posterior (default 40)")
    p.add_argument("--vp10_samples", type=int, default=2000,
                   help="Posterior draws per VP10 reader draw (default 2000)")
    p.add_argument("--n_ppc", type=int, default=300,
                   help="Posterior draws re-simulated for the PPC table/plot (default 300)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="Override the above with small sizes for a fast smoke test")
    p.add_argument("--save_json", default=str(config.OUT_DIR / "results_summary.json"),
                   help="Where to save a machine-readable copy of this report")
    args = p.parse_args()
    if args.quick:
        args.n_val, args.n_posterior_samples = 40, 200
        args.n_vp10_readers, args.vp10_samples = 5, 200
        args.n_ppc = 30
    return args


def hr(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def classify_r(r: float) -> str:
    if r >= 0.7:
        return "strong"
    if r >= 0.35:
        return "moderate"
    return "weak"


class _Tee:
    """Writes to both the real stdout and a buffer, so console output is
    unchanged while we also capture it (to pull PPC numbers into the JSON
    without duplicating posterior_predictive_check's simulation logic)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _parse_ppc_table(text: str) -> list[dict]:
    """Pull the rows out of posterior_predictive_check's printed
    '===== PPC SUMMARY =====' table."""
    rows = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("Statistic"):
            in_table = True
            continue
        if in_table and line.startswith("-"):
            continue
        if in_table and line.strip():
            name, real_s, sim_s = line[:26].strip(), line[26:36].strip(), line[36:].strip()
            try:
                rows.append({"statistic": name, "real": float(real_s), "simulated": float(sim_s)})
            except ValueError:
                break
        elif in_table:
            break
    return rows


def main() -> None:
    args = parse_args()
    t_start = time.time()

    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model at {config.MODEL_PATH}\n"
            f"Train one first:  python main.py --mode train   (or --mode all)")

    from swift.data import (
        build_corpus_lists, build_reader_batch, load_corpus, load_fixations,
        refix_rate, regression_rate, skip_rate, split_half,
    )
    from swift.diagnostics import (
        plot_posterior_correlation, posterior_predictive_check,
    )
    from swift.inference import (
        _make_simulator, rebuild_workflow, run_inference, set_corpus,
    )
    from swift.simulator import PARAM_NAMES, THETA_MAX, THETA_MIN

    report: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_path": str(config.MODEL_PATH.relative_to(config.ROOT)),
                    "config": {"M_SENTENCES": config.M_SENTENCES,
                               "SEQ_LEN": config.SEQ_LEN,
                               "N_STATS": config.N_STATS}}

    print("=" * 66)
    print("SWIFT + BayesFlow -- SAVED MODEL RESULTS REPORT")
    print("=" * 66)
    print(f"Model file : {config.MODEL_PATH}")
    print(f"             ({os.path.getsize(config.MODEL_PATH) / 1e6:.1f} MB, "
          f"modified {time.ctime(os.path.getmtime(config.MODEL_PATH))})")
    print(f"M_SENTENCES={config.M_SENTENCES}  SEQ_LEN={config.SEQ_LEN}  "
          f"N_STATS={config.N_STATS}  params={PARAM_NAMES}")

    # ------------------------------------------------------------------
    # 1. Real VP10 data summary
    # ------------------------------------------------------------------
    hr("[1/4] REAL VP10 DATA SUMMARY")
    fix = load_fixations(config.FIXATION_PATH)
    wfl = build_corpus_lists(load_corpus(config.CORPUS_PATH))
    train_ids, test_ids = split_half(fix)
    sk, rf, rg = skip_rate(fix), refix_rate(fix), regression_rate(fix)
    d = fix["fixation_duration"]
    fps = fix.groupby("sentence_id").size()
    print(f"Fixations              : {len(fix)}")
    print(f"Sentences              : {fix['sentence_id'].nunique()}")
    print(f"Duration mean +/- std  : {d.mean():.1f} +/- {d.std():.1f} ms "
          f"(CV = {d.std() / d.mean():.3f})")
    print(f"Fixations / sentence   : {fps.mean():.2f}")
    print(f"Skip rate              : {sk:.1%}")
    print(f"Refixation rate        : {rf:.1%}")
    print(f"Regression rate        : {rg:.1%}")
    report["vp10_data"] = {"n_fixations": int(len(fix)),
                           "n_sentences": int(fix["sentence_id"].nunique()),
                           "duration_mean_ms": float(d.mean()),
                           "duration_std_ms": float(d.std()),
                           "duration_cv": float(d.std() / d.mean()),
                           "fixations_per_sentence": float(fps.mean()),
                           "skip_rate": float(sk), "refix_rate": float(rf),
                           "regression_rate": float(rg)}

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    set_corpus(wfl)
    workflow = rebuild_workflow(config.MODEL_PATH)

    # ------------------------------------------------------------------
    # 2. Recovery + SBC calibration on fresh held-out simulations
    # ------------------------------------------------------------------
    hr(f"[2/4] PARAMETER RECOVERY & CALIBRATION  (n={args.n_val} held-out readers)")
    sim = _make_simulator()
    val = sim.sample(args.n_val)
    post = workflow.sample(conditions=val, num_samples=args.n_posterior_samples)

    true_theta = np.asarray(val["theta"])                 # (n_val, 4), normalised [0,1]
    draws = np.asarray(post["theta"])                     # (n_val, n_samples, 4)
    post_mean = draws.mean(axis=1)

    print(f"{'Parameter':<10}{'Recovery r':>12}{'Contraction':>14}{'95% CI cov.':>14}   Identifiability")
    print("-" * 78)
    recovery_rows = []
    for j, name in enumerate(PARAM_NAMES):
        r = float(np.corrcoef(true_theta[:, j], post_mean[:, j])[0, 1])
        prior_var = float(np.var(true_theta[:, j]))
        post_var = float(np.mean(np.var(draws[:, :, j], axis=1)))
        contraction = 1.0 - post_var / prior_var if prior_var > 0 else float("nan")
        lo = np.percentile(draws[:, :, j], 2.5, axis=1)
        hi = np.percentile(draws[:, :, j], 97.5, axis=1)
        coverage = float(np.mean((true_theta[:, j] >= lo) & (true_theta[:, j] <= hi)))
        print(f"{name:<10}{r:>12.3f}{contraction:>14.3f}{coverage:>13.1%}   {classify_r(r)}")
        recovery_rows.append({"parameter": name, "recovery_r": r,
                              "contraction": contraction, "ci95_coverage": coverage})
    print("\n(Recovery r: correlation of posterior mean with ground truth on simulated")
    print(" data with a KNOWN theta. Contraction: 1 - posterior_var/prior_var (higher =")
    print(" learned more from the data). 95% CI coverage: should be close to 95% if the")
    print(" posterior is well-calibrated -- see outputs/figures/sbc_*.png for the full")
    print(" SBC diagnostic.)")
    report["recovery"] = recovery_rows

    # ------------------------------------------------------------------
    # 3. VP10 posterior estimates
    # ------------------------------------------------------------------
    hr(f"[3/4] VP10 POSTERIOR ESTIMATES  (train split, pooled over "
       f"{args.n_vp10_readers} random {config.M_SENTENCES}-sentence draws)")
    rng = np.random.default_rng(args.seed)
    observations, obs_stats = build_reader_batch(
        fix, m_sentences=config.M_SENTENCES, n_readers=args.n_vp10_readers,
        rng=rng, sentence_ids=train_ids)
    with contextlib.redirect_stdout(io.StringIO()):  # skip run_inference's own printout
        posterior = run_inference(workflow, observations, obs_stats,
                                  num_samples=args.vp10_samples)

    print(f"{'Parameter':<10}{'Mean':>10}{'95% CI':>22}{'Prior range':>18}")
    print("-" * 62)
    vp10_rows = []
    units = {"nu": "", "r": "", "mu_T": "ms"}
    for j, name in enumerate(PARAM_NAMES):
        col = posterior[:, j]
        mean = float(col.mean())
        lo, hi = float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))
        u = units[name]
        ci_str = f"[{lo:.2f}, {hi:.2f}] {u}".strip()
        prior_str = f"[{THETA_MIN[j]:.1f}, {THETA_MAX[j]:.1f}] {u}".strip()
        print(f"{name:<10}{mean:>10.2f}{ci_str:>22}{prior_str:>18}")
        vp10_rows.append({"parameter": name, "mean": mean, "ci95_low": lo, "ci95_high": hi})
    report["vp10_posterior"] = vp10_rows

    corr = plot_posterior_correlation(posterior)
    report["posterior_correlation"] = corr.tolist()

    # ------------------------------------------------------------------
    # 4. Posterior predictive check (held-out test split, paper Section 6)
    # ------------------------------------------------------------------
    hr(f"[4/4] POSTERIOR PREDICTIVE CHECK  (test split, n={args.n_ppc} posterior "
       f"draws x {config.M_SENTENCES} sentences)")
    test_idx = [int(s) - 1 for s in test_ids]
    real_test = fix[fix["sentence_id"].isin(test_ids)]
    tee = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, tee)):
        posterior_predictive_check(posterior, real_test, wfl, n_ppc=args.n_ppc,
                                   rng=np.random.default_rng(args.seed + 1),
                                   sentence_indices=test_idx)
    report["ppc"] = _parse_ppc_table(tee.getvalue())

    dt = time.time() - t_start
    report["runtime_seconds"] = round(dt, 1)
    with open(args.save_json, "w") as f:
        json.dump(report, f, indent=2)

    hr("DONE")
    print(f"Report generated in {dt:.1f}s")
    print(f"Figures      -> {config.FIG_DIR}/")
    print(f"JSON summary -> {args.save_json}")
    print("Docs         -> docs/RESULTS.md (narrative write-up)")
    print("                docs/PROJECT_GUIDE.md (file-by-file / parameter guide)")


if __name__ == "__main__":
    main()
