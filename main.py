"""
main.py
=======
End-to-end SWIFT + BayesFlow v2 pipeline.

RECOMMENDED WORKFLOW (avoids 2-3 hour training runs):
------------------------------------------------------
Step A — generate simulations ONCE and save to disk (~8 min):
    python main.py --mode generate --n_sim 10000

Step B — train on saved simulations (fast, ~10 min):
    python main.py --mode train_offline

Step C — run diagnostics + inference on trained model:
    python main.py --mode infer

OR run everything online in one go (slow, ~3 hours):
    python main.py --mode full

OTHER MODES:
    python main.py --mode train        # online training (slow)
    python main.py --mode generate     # only generate + save simulations
    python main.py --mode train_offline # train on pre-generated data
    python main.py --mode infer        # diagnostics + inference only
"""

import argparse
import os
import sys
import time
import numpy as np

# Set Keras backend BEFORE any keras/bayesflow imports happen
# Change to "torch" or "jax" if preferred
if not os.environ.get("KERAS_BACKEND"):
    os.environ["KERAS_BACKEND"] = "torch"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIXATION_PATH     = "data/fixseqin_PB2expVP10.dat"
CORPUS_PATH       = "data/Rcorpus_PB2_revision.dat"
TRAINING_DATA     = "data/training_data.npz"
MODEL_PATH        = "inference/trained_model/swift_approximator.keras"
SAVE_DIR          = "diagnostics"
SEQ_LEN           = 15    # max fixations per sentence (real data max=10, buffer to 15)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="SWIFT BayesFlow v2 Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--mode", default="full",
        choices=["full", "generate", "train", "train_offline", "infer"],
        help=(
            "full          : online training + diagnostics + inference (slow)\n"
            "generate      : pre-generate simulations and save to disk\n"
            "train         : online training only\n"
            "train_offline : train on pre-generated data (RECOMMENDED)\n"
            "infer         : diagnostics + inference on saved model\n"
        )
    )
    p.add_argument("--n_sim",     type=int, default=10000,
                   help="Number of simulations to pre-generate (default: 10000)")
    p.add_argument("--n_epochs",  type=int, default=50,
                   help="Training epochs (default: 50)")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size (default: 32)")
    p.add_argument("--n_batches", type=int, default=100,
                   help="Gradient steps per epoch (default: 100)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Load data (always runs)
# ---------------------------------------------------------------------------

def step1_load():
    from data.load_data import (
        load_fixations, load_corpus, build_corpus_lists,
        run_eda, synthetic_corpus
    )

    print("\n" + "=" * 55)
    print("STEP 1 — Load Data")
    print("=" * 55)

    fix = load_fixations(FIXATION_PATH)
    run_eda(fix, save_dir=SAVE_DIR)

    if os.path.exists(CORPUS_PATH):
        corpus = load_corpus(CORPUS_PATH)
    else:
        print(f"\nWARNING: Corpus not found at {CORPUS_PATH}")
        print("Using synthetic corpus. Place real file in data/ folder.")
        corpus = synthetic_corpus(n_sentences=114)

    wll, wfl = build_corpus_lists(corpus)
    return fix, wll, wfl


# ---------------------------------------------------------------------------
# Step 2a: Generate simulations offline and save to disk
# ---------------------------------------------------------------------------

def step2_generate(wll, wfl, n_sim: int):
    from simulator.swift_simulator import (
        sample_prior, run_one_simulation,
        normalise_theta, PARAM_NAMES
    )

    print("\n" + "=" * 55)
    print("STEP 2 — Generate Simulations Offline")
    print("=" * 55)
    print(f"Target : {n_sim:,} simulations")
    print(f"Saving → {TRAINING_DATA}")
    print(f"Estimated time: ~{n_sim * 0.05 / 60:.0f} minutes on average laptop")
    print("-" * 55)

    rng      = np.random.default_rng(42)
    n_sent   = len(wll)
    thetas   = np.zeros((n_sim, 4),            dtype=np.float32)
    seqs     = np.zeros((n_sim, SEQ_LEN, 3),   dtype=np.float32)

    start = time.time()
    for i in range(n_sim):
        params          = sample_prior(rng)
        thetas[i]       = normalise_theta(params)
        sent_idx        = rng.integers(0, n_sent)
        seqs[i]         = run_one_simulation(
            params            = params,
            word_lengths_list = wll,
            word_freqs_list   = wfl,
            seq_len           = SEQ_LEN,
            rng               = rng,
        )

        # Progress every 500
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            rate    = (i + 1) / elapsed
            remain  = (n_sim - i - 1) / rate
            print(f"  [{i+1:>6}/{n_sim}]  "
                  f"elapsed {elapsed/60:.1f} min  "
                  f"remaining ~{remain/60:.1f} min")

    os.makedirs("data", exist_ok=True)
    np.savez(TRAINING_DATA, thetas=thetas, seqs=seqs)

    total = time.time() - start
    print(f"\nDone. {n_sim:,} simulations saved → {TRAINING_DATA}")
    print(f"Total time: {total/60:.1f} minutes")
    print(f"  thetas shape : {thetas.shape}")
    print(f"  seqs shape   : {seqs.shape}")

    return thetas, seqs


# ---------------------------------------------------------------------------
# Step 2b: Train online (simulator runs during training — slow)
# ---------------------------------------------------------------------------

def step2_train_online(wll, wfl, args):
    from inference.train_bayesflow import train

    print("\n" + "=" * 55)
    print("STEP 2 — Online Training (simulator runs during training)")
    print("=" * 55)

    total_sims = args.n_epochs * args.n_batches * args.batch_size
    print(f"WARNING: This will run {total_sims:,} simulations during training.")
    print(f"Estimated time: ~{total_sims * 0.05 / 60:.0f} minutes.")
    print("TIP: Use --mode generate then --mode train_offline to avoid this.\n")

    workflow = train(
        word_lengths_list     = wll,
        word_freqs_list       = wfl,
        n_epochs              = args.n_epochs,
        batch_size            = args.batch_size,
        num_batches_per_epoch = args.n_batches,
        seq_len               = SEQ_LEN,
        save_path             = os.path.dirname(MODEL_PATH),
    )
    return workflow


# ---------------------------------------------------------------------------
# Step 2c: Train offline on pre-generated data (FAST — recommended)
# ---------------------------------------------------------------------------

def step2_train_offline(wll, wfl, args):
    from inference.train_bayesflow import train_offline

    print("\n" + "=" * 55)
    print("STEP 2 — Offline Training (on pre-generated data)")
    print("=" * 55)

    if not os.path.exists(TRAINING_DATA):
        raise FileNotFoundError(
            f"No training data found at {TRAINING_DATA}.\n"
            f"Run first: python main.py --mode generate --n_sim 10000"
        )

    data   = np.load(TRAINING_DATA)
    thetas = data["thetas"]
    seqs   = data["seqs"]
    print(f"Loaded {len(thetas):,} pre-generated simulations from {TRAINING_DATA}")

    workflow = train_offline(
        thetas    = thetas,
        seqs      = seqs,
        wll       = wll,
        wfl       = wfl,
        n_epochs  = args.n_epochs,
        batch_size = args.batch_size,
        seq_len   = SEQ_LEN,
        save_path = os.path.dirname(MODEL_PATH),
    )
    return workflow


# ---------------------------------------------------------------------------
# Step 3: Diagnostics
# ---------------------------------------------------------------------------

def step3_diagnostics(workflow, wll, wfl):
    from inference.train_bayesflow import set_corpus, swift_prior, swift_likelihood
    from diagnostics.diagnostics import run_builtin_diagnostics
    import bayesflow as bf

    print("\n" + "=" * 55)
    print("STEP 3 — Diagnostics")
    print("=" * 55)

    set_corpus(wll, wfl, SEQ_LEN)
    simulator = bf.simulators.make_simulator([swift_prior, swift_likelihood])

    run_builtin_diagnostics(
        workflow            = workflow,
        simulator           = simulator,
        n_val               = 300,
        n_posterior_samples = 1000,
        save_dir            = SAVE_DIR,
    )


# ---------------------------------------------------------------------------
# Step 4: Inference on real VP10 data
# ---------------------------------------------------------------------------

def step4_infer(workflow, fix):
    from data.load_data import fixations_to_array
    from inference.train_bayesflow import run_inference
    from diagnostics.diagnostics import plot_posterior

    print("\n" + "=" * 55)
    print("STEP 4 — Infer Parameters for VP10")
    print("=" * 55)

    # Build one array per sentence then average across sentences
    # This gives a global parameter estimate for this participant
    arrays   = [
        fixations_to_array(fix, sid, seq_len=SEQ_LEN)
        for sid in sorted(fix["sentence_id"].unique())
    ]
    combined = np.mean(np.stack(arrays, axis=0), axis=0)  # (seq_len, 3)
    print(f"Aggregated {len(arrays)} sentences → shape {combined.shape}")

    posterior = run_inference(
        workflow            = workflow,
        fixation_array      = combined,
        n_posterior_samples = 5000,
    )

    plot_posterior(
        posterior_samples = posterior,
        true_params       = None,
        save_dir          = SAVE_DIR,
    )
    return posterior


# ---------------------------------------------------------------------------
# Step 5: Posterior Predictive Check
# ---------------------------------------------------------------------------

def step5_ppc(posterior, fix, wll, wfl):
    from diagnostics.diagnostics import posterior_predictive_check

    print("\n" + "=" * 55)
    print("STEP 5 — Posterior Predictive Check")
    print("=" * 55)

    posterior_predictive_check(
        posterior_samples = posterior,
        real_fix_df       = fix,
        word_lengths_list = wll,
        word_freqs_list   = wfl,
        n_ppc             = 200,
        save_dir          = SAVE_DIR,
    )


# ---------------------------------------------------------------------------
# Load saved model
# ---------------------------------------------------------------------------

def load_saved_workflow():
    import keras
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No saved model at {MODEL_PATH}.\n"
            "Run training first:\n"
            "  python main.py --mode generate\n"
            "  python main.py --mode train_offline"
        )
    print(f"Loading saved model from {MODEL_PATH}...")
    # Rebuild the workflow object for sampling
    from inference.train_bayesflow import rebuild_workflow
    return rebuild_workflow(MODEL_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    os.makedirs(SAVE_DIR,                    exist_ok=True)
    os.makedirs("data",                      exist_ok=True)
    os.makedirs("inference/trained_model",   exist_ok=True)

    # Step 1 always runs
    fix, wll, wfl = step1_load()

    # -----------------------------------------------------------------------
    if args.mode == "generate":
        # Only generate and save simulations — no training
        step2_generate(wll, wfl, n_sim=args.n_sim)
        print("\nNext step: python main.py --mode train_offline")

    # -----------------------------------------------------------------------
    elif args.mode == "train_offline":
        # Train on pre-generated data → diagnostics → inference
        workflow  = step2_train_offline(wll, wfl, args)
        step3_diagnostics(workflow, wll, wfl)
        posterior = step4_infer(workflow, fix)
        step5_ppc(posterior, fix, wll, wfl)

    # -----------------------------------------------------------------------
    elif args.mode == "train":
        # Online training (slow) → diagnostics → inference
        workflow  = step2_train_online(wll, wfl, args)
        step3_diagnostics(workflow, wll, wfl)
        posterior = step4_infer(workflow, fix)
        step5_ppc(posterior, fix, wll, wfl)

    # -----------------------------------------------------------------------
    elif args.mode == "infer":
        # Load saved model → diagnostics → inference
        workflow  = load_saved_workflow()
        step3_diagnostics(workflow, wll, wfl)
        posterior = step4_infer(workflow, fix)
        step5_ppc(posterior, fix, wll, wfl)

    # -----------------------------------------------------------------------
    elif args.mode == "full":
        # Online training + everything (slow — ~3 hours)
        workflow  = step2_train_online(wll, wfl, args)
        step3_diagnostics(workflow, wll, wfl)
        posterior = step4_infer(workflow, fix)
        step5_ppc(posterior, fix, wll, wfl)
