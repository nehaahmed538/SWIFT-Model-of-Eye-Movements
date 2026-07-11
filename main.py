"""
main.py
=======
End-to-end SWIFT + BayesFlow v2 pipeline.

Recommended workflow (avoids running the simulator inside training):

    python main.py --mode generate --n_readers 8000   # once, ~minutes
    python main.py --mode train                       # train on saved data
    python main.py --mode infer                        # diagnostics + VP10 + PPC

Or everything at once:

    python main.py --mode all --n_readers 8000

Modes
    generate : pre-generate simulations (parallel) and save to disk
    train    : train on pre-generated data (-> model + loss plot)
    infer    : load saved model -> diagnostics + VP10 inference + PPC
    online   : train with the simulator in the loop (slow)
    all      : generate -> train -> infer
"""

import argparse
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from swift import config
from swift.data import (
    build_corpus_lists, build_reader_batch, load_corpus, load_fixations,
    run_eda, synthetic_corpus,
)


def parse_args():
    p = argparse.ArgumentParser(description="SWIFT BayesFlow v2 pipeline",
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--mode", default="all",
                   choices=["generate", "train", "infer", "online", "all"])
    p.add_argument("--n_readers", type=int, default=8000,
                   help="Readers to pre-generate (default 8000)")
    p.add_argument("--n_epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_workers", type=int, default=None,
                   help="Parallel workers for generation (default: cores-1)")
    return p.parse_args()


def step_load():
    print("\n" + "=" * 55 + "\nSTEP - Load Data\n" + "=" * 55)
    fix = load_fixations(config.FIXATION_PATH)
    run_eda(fix)
    if os.path.exists(config.CORPUS_PATH):
        corpus = load_corpus(config.CORPUS_PATH)
    else:
        print(f"WARNING: corpus not found at {config.CORPUS_PATH}; using synthetic.")
        corpus = synthetic_corpus(114)
    wll, wfl = build_corpus_lists(corpus)
    return fix, wll, wfl


def step_generate(wll, wfl, args):
    from swift.generate import generate
    return generate(wll, wfl, n_readers=args.n_readers, n_workers=args.n_workers)


def step_train_offline(wll, wfl, args):
    from swift.inference import train_offline
    if not os.path.exists(config.TRAINING_DATA):
        raise FileNotFoundError(
            f"No training data at {config.TRAINING_DATA}. "
            f"Run: python main.py --mode generate")
    data = np.load(config.TRAINING_DATA)
    print(f"Loaded {len(data['thetas']):,} readers from {config.TRAINING_DATA}")
    return train_offline(data["thetas"], data["seqs"], data["stats"], wll, wfl,
                         n_epochs=args.n_epochs, batch_size=args.batch_size)


def step_diagnostics(workflow, wll, wfl):
    from swift.inference import set_corpus, _make_simulator
    from swift.diagnostics import run_builtin_diagnostics
    print("\n" + "=" * 55 + "\nSTEP - Diagnostics\n" + "=" * 55)
    set_corpus(wll, wfl)
    run_builtin_diagnostics(workflow, _make_simulator(),
                            n_val=300, n_posterior_samples=1000)


def step_infer(workflow, fix):
    from swift.inference import run_inference
    from swift.diagnostics import plot_posterior
    print("\n" + "=" * 55 + "\nSTEP - Infer VP10 Parameters\n" + "=" * 55)
    observations, obs_stats = build_reader_batch(
        fix, m_sentences=config.M_SENTENCES, n_readers=40,
        rng=np.random.default_rng(0))
    print(f"Built {len(observations)} reader observations "
          f"({config.M_SENTENCES} sentences each)")
    posterior = run_inference(workflow, observations, obs_stats, num_samples=2000)
    plot_posterior(posterior)
    return posterior


def step_ppc(posterior, fix, wll, wfl):
    from swift.diagnostics import posterior_predictive_check
    print("\n" + "=" * 55 + "\nSTEP - Posterior Predictive Check\n" + "=" * 55)
    posterior_predictive_check(posterior, fix, wll, wfl, n_ppc=300)


if __name__ == "__main__":
    args = parse_args()
    fix, wll, wfl = step_load()

    if args.mode == "generate":
        step_generate(wll, wfl, args)
        print("\nNext: python main.py --mode train")

    elif args.mode in ("train", "all"):
        if args.mode == "all":
            step_generate(wll, wfl, args)
        workflow = step_train_offline(wll, wfl, args)
        step_diagnostics(workflow, wll, wfl)
        posterior = step_infer(workflow, fix)
        step_ppc(posterior, fix, wll, wfl)

    elif args.mode == "online":
        from swift.inference import train_online
        workflow = train_online(wll, wfl, n_epochs=args.n_epochs,
                                batch_size=args.batch_size)
        step_diagnostics(workflow, wll, wfl)
        posterior = step_infer(workflow, fix)
        step_ppc(posterior, fix, wll, wfl)

    elif args.mode == "infer":
        from swift.inference import rebuild_workflow
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(
                f"No saved model at {config.MODEL_PATH}. "
                f"Run: python main.py --mode train")
        workflow = rebuild_workflow(config.MODEL_PATH)
        step_diagnostics(workflow, wll, wfl)
        posterior = step_infer(workflow, fix)
        step_ppc(posterior, fix, wll, wfl)
