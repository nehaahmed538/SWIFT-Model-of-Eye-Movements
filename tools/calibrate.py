"""
tools/calibrate.py
==================
Pure-NumPy check that the simulator's *marginal* fixation statistics (averaged
over the prior) resemble participant VP10. No BayesFlow needed, so it runs in
seconds and gives a fast feedback loop for tuning the FIXED constants in
swift/simulator.py.

Run:  python tools/calibrate.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swift.data import load_corpus, load_fixations, build_corpus_lists
from swift.config import CORPUS_PATH, FIXATION_PATH
from swift.simulator import SWIFTSimulator, sample_prior


def main(n_readers: int = 400) -> None:
    fix = load_fixations(FIXATION_PATH)
    wll, wfl = build_corpus_lists(load_corpus(CORPUS_PATH))

    rng = np.random.default_rng(0)
    counts, durs, landings = [], [], []
    for _ in range(n_readers):
        p = sample_prior(rng)
        si = rng.integers(0, len(wll))
        f = SWIFTSimulator(p).simulate_sentence(wll[si], wfl[si], rng=rng)
        counts.append(len(f))
        for _, lp, dur, _ in f:
            durs.append(dur)
            landings.append(lp)

    real_counts = fix.groupby("sentence_id").size().values
    real_dur = fix["fixation_duration"].values
    real_lnd = fix["landing_position"].values

    def line(name, sim, real):
        print(f"  {name:<22} sim {np.mean(sim):7.2f} +/- {np.std(sim):6.2f}"
              f"   real {np.mean(real):7.2f} +/- {np.std(real):6.2f}")

    print(f"\nCalibration over {n_readers} prior draws:")
    line("fixations / sentence", counts, real_counts)
    line("duration (ms)", durs, real_dur)
    line("landing position", landings, real_lnd)
    print(f"  words / sentence (corpus): {np.mean([len(w) for w in wll]):.2f}")


if __name__ == "__main__":
    main()
