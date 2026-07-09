# SWIFT Model — Amortized Inference with BayesFlow v2
### SBI Course Project — TU Dortmund 2026

---

## Project Structure

```
swift_project/
│
├── data/
│   ├── load_data.py                  ← Load & explore fixation + corpus files
│   ├── fixseqin_PB2expVP10.dat       ← Real eye-tracking data (place here)
│   ├── Rcorpus_PB2_revision.dat      ← Word properties corpus (place here)
│   └── training_data.npz             ← Pre-generated simulations (auto-created)
│
├── simulator/
│   └── swift_simulator.py            ← SWIFT Gillespie simulator
│
├── inference/
│   └── train_bayesflow.py            ← BayesFlow v2 training pipeline
│
├── diagnostics/
│   └── diagnostics.py                ← SBC, PPC, posterior plots
│
├── main.py                           ← End-to-end pipeline
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place your data files in `data/`:
- `fixseqin_PB2expVP10.dat`       (from osf.io/teyd4)
- `Rcorpus_PB2_revision.dat`      (from osf.io/nj2mf)

---

## RECOMMENDED Workflow (avoids 3-hour training runs)

### Step A — Pre-generate simulations once (~8 minutes)
```bash
python main.py --mode generate --n_sim 10000
```
Runs SWIFT simulator 10,000 times and saves to `data/training_data.npz`.
You only ever need to do this once.

### Step B — Train on saved simulations (~10 minutes)
```bash
python main.py --mode train_offline --n_epochs 50 --batch_size 32
```
Trains BayesFlow on pre-generated data.
Fast because the simulator does NOT run during training.

### Step C — Run inference on saved model
```bash
python main.py --mode infer
```
Loads trained model, runs diagnostics, infers VP10 parameters, runs PPC.

---

## All Available Modes

| Command | What it does | Time (Dell Inspiron) |
|---------|-------------|----------------------|
| `--mode generate --n_sim 10000` | Pre-generate simulations | ~8 min |
| `--mode train_offline` | Train on pre-generated data | ~10 min |
| `--mode infer` | Diagnostics + inference + PPC | ~20 min |
| `--mode train` | Online training only (slow) | ~2–3 hours |
| `--mode full` | Everything online (slow) | ~3+ hours |

---

## Model: Free Parameters Inferred by BayesFlow

| Parameter | Symbol | Prior Range | Meaning |
|-----------|--------|-------------|---------|
| Saccade timer | `t_sac` | 150–350 ms | Average time between saccade initiations |
| Word-length exponent | `eta` | 0.1–1.0 | How much word length slows processing |
| Processing span | `delta0` | 4–15 chars | Width of attentional window |
| Refixation factor | `R` | 0.1–0.9 | Tendency to refixate current word |

---

## BayesFlow v2 Pipeline

```
OFFLINE GENERATION (run once):
  sample_prior()        → theta ~ Uniform(prior_bounds)
  run_one_simulation()  → SWIFT Gillespie → fixation sequence
  save 10,000 pairs     → data/training_data.npz

OFFLINE TRAINING (train_offline):
  load training_data.npz
  tf.data.Dataset → shuffle → batch
  adapter: theta → inference_variables
           fixations → summary_variables
  summary_net  : SequenceNetwork (LSTM, summary_dim=32)
  inference_net: CouplingFlow (spline, num_layers=6)
  workflow.fit_dataset(dataset, epochs=50)
  save → inference/trained_model/swift_approximator.keras

INFERENCE (infer):
  workflow.sample(conditions=real_VP10_data, num_samples=5000)
  → posterior over (t_sac, eta, delta0, R)
```

---

## Diagnostics Output

| Plot file | What it shows | What to look for |
|-----------|--------------|------------------|
| `eda_fixations.png` | Real data distributions | Understand the data |
| `training_loss.png` | Training loss curve | Should decrease and flatten |
| `recovery_plot.png` | Posterior mean vs true value | Points near diagonal |
| `sbc_histogram.png` | SBC rank histogram | Uniform bars |
| `sbc_ecdf.png` | SBC ECDF | Line near zero |
| `contraction_plot.png` | Posterior contraction | High = learned from data |
| `posterior_VP10.png` | Posterior over parameters | Narrow = informative |
| `ppc_plot.png` | Simulated vs real fixations | Overlapping distributions |

---

## Key References

- Engbert & Rabe (2024). A tutorial on Bayesian inference for dynamical
  modeling of eye-movement control during reading.
  *Journal of Mathematical Psychology*, 119, 102843.

- Rabe et al. (2021). Bayesian parameter estimation for the SWIFT model.
  *Psychological Review*, 128(3), 516–543.

- BayesFlow v2 docs: https://bayesflow.org/v2.0.11/
