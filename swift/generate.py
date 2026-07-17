"""
generate.py
===========
Pre-generate (theta, fixation-sequence) training pairs by running the SWIFT
simulator offline, in parallel across CPU cores. Doing this once and training
on the saved arrays is far faster than running the simulator inside the
training loop.

Each example is one reader: a theta drawn from the prior and the concatenated,
normalised, padded fixation sequence from that reader reading M_SENTENCES
sentences.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from multiprocessing import cpu_count

import numpy as np

from swift.config import M_SENTENCES, N_FEATURES, N_STATS, SEQ_LEN, TRAINING_DATA
from swift.simulator import PARAM_NAMES, normalise_theta, run_one_reader, sample_prior

# Corpus is set once per worker process via the Pool initializer (avoids
# pickling the corpus for every task).
_WFL = None


def _init_worker(wfl):
    global _WFL
    _WFL = wfl


def _gen_chunk(args):
    seed, n = args
    rng = np.random.default_rng(seed)
    thetas = np.zeros((n, len(PARAM_NAMES)), dtype=np.float32)
    seqs = np.zeros((n, SEQ_LEN, N_FEATURES), dtype=np.float32)
    stats = np.zeros((n, N_STATS), dtype=np.float32)
    for i in range(n):
        params = sample_prior(rng)
        thetas[i] = normalise_theta(params)
        seqs[i], stats[i] = run_one_reader(params, _WFL, SEQ_LEN, M_SENTENCES, rng)
    return thetas, seqs, stats


def generate(wfl, n_readers: int = 8000, n_workers: int | None = None,
             seed: int = 42, save_path=TRAINING_DATA):
    n_workers = n_workers or max(1, cpu_count() - 1)
    print("\n" + "=" * 55)
    print("STEP - Generate Simulations (parallel)")
    print("=" * 55)
    print(f"Readers : {n_readers:,}  x  {M_SENTENCES} sentences each")
    print(f"Workers : {n_workers}")

    # Split into chunks (a few per worker for load balancing), unique seeds.
    n_chunks = n_workers * 4
    sizes = [n_readers // n_chunks] * n_chunks
    for i in range(n_readers - sum(sizes)):
        sizes[i] += 1
    tasks = [(seed + 1000 * i, s) for i, s in enumerate(sizes) if s > 0]

    start = time.time()
    # 'fork' avoids re-importing __main__ in workers (fast, and works when the
    # caller is a heredoc/notebook). Generation is pure NumPy, so fork is safe.
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(wfl,)) as pool:
        chunks = pool.map(_gen_chunk, tasks)
    thetas = np.concatenate([c[0] for c in chunks], axis=0)
    seqs = np.concatenate([c[1] for c in chunks], axis=0)
    stats = np.concatenate([c[2] for c in chunks], axis=0)

    save_path = str(save_path)
    np.savez(save_path, thetas=thetas, seqs=seqs, stats=stats)
    dt = time.time() - start
    print(f"\nDone. {len(thetas):,} readers in {dt/60:.1f} min "
          f"({len(thetas)/dt:.0f} readers/s)")
    print(f"  thetas {thetas.shape} | seqs {seqs.shape} | stats {stats.shape}")
    print(f"  saved -> {save_path}")
    return thetas, seqs, stats
