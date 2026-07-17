# Model specification — simplified SWIFT (Engbert & Rabe 2024)

Every equation the simulator (`swift/simulator.py`) implements, with its
equation/section number in Engbert, R. & Rabe, M. M. (2024), *A tutorial on
Bayesian inference for dynamical modeling of eye-movement control during
reading*, Journal of Mathematical Psychology, 119, 102843.

This is the **basic 3-parameter** model. Words sit at discrete positions
`1..N` with no spatial extension: no word length, no landing position, no
Gillespie algorithm, no activation thresholds. Skipping, refixation and
regression are **emergent** from the saliency target rule — there is no
explicit mechanism for any of them.

## Free parameters (inferred)

| Symbol | Meaning | Prior (uniform) | Paper |
|---|---|---|---|
| `nu`   | processing-span shape | `(0, 1]` | Eq. 1–2, Section 5 |
| `r`    | overall processing rate | `(0, 12]` | Eq. 3, 6, 7, Section 5 |
| `mu_T` | mean saccade-timer interval (ms) | `(100, 400)` | Eq. 10–12, Section 5 |

## Fixed constants

| Constant | Value | Source |
|---|---|---|
| `eta`   | `1e-3` | baseline saliency, keeps target denominator > 0 (Section 3) |
| `alpha` | `9`    | Gamma shape ⇒ fixation-duration CV = 1/√9 = 1/3 (Section 2.4) |
| `beta`  | `0.6`  | word-frequency effect on max activation (Section 5 recovery value) |

## Equations

**Processing span (Eq. 1–2)** — `span_rates(k, N, nu)`. Gaze on word `k`:

```
lambda_{k-1} = sigma * nu
lambda_{k}   = sigma
lambda_{k+1} = sigma * nu
lambda_{k+2} = sigma * nu**2
lambda_w     = 0   otherwise          (note: k-2 is NOT processed; k+2 IS)
sigma        = 1 / (1 + 2*nu + nu**2)   (global, not renormalised at boundaries)
```

Verified against the paper's Fig. 2 (unit test in `simulator._demo`).

**Word difficulty (Eq. 4–5)** — `a_max = 1 - beta * q`, with
`q_w = log10(F_w) / max_i log10(F_i)`. `beta = 0` would give `a_max = 1` for
all words (strict Section-3 baseline, frequency ignored); we use `beta = 0.6`.

**Activation (Eq. 3, 6, 7)** — closed-form over each fixation:

```
a_w(t + T) = min(a_w(t) + r * lambda_w * T_seconds,  a_max)
```

**Unit note (important):** `r` (paper values 5–10) is a rate *per second*, so
the fixation duration `T` (stored in ms) is converted to seconds in this term.
With `T` in ms, `r*lambda*T` reaches the hundreds and every in-span word
saturates to `a_max` in a single fixation, which collapses the saliency rule
below and makes the scanpath independent of `nu` and `r`. In seconds,
processing takes several fixations, so low `r` → refixation and larger `nu` →
wider span/skipping, exactly as the paper describes (Sections 2.4, 3). This
is the one place the paper's "milliseconds throughout" wording is reconciled
with its `r` values; see the "Deviations" section of `docs/RESULTS.md`.

**Target selection (Eq. 8–9)** — sine saliency:

```
s_w = a_max * sin(pi * a_w / a_max) + eta
p_w = s_w / sum_v s_v
```

Unimodal, peaking at `a_w = a_max/2`: both unprocessed (`a=0`) and fully
processed (`a=a_max`) words have saliency ≈ `eta`. Skipping, refixation and
regression all emerge from this.

**Saccade timer (Eq. 10–12)** — fixation durations are pure Gamma draws,
independent of word processing (Section 2.4):

```
T_i ~ Gamma(shape = alpha = 9, scale = mu_T / alpha)
E[T_i]  = mu_T
CV      = 1/3   (by construction)
```

**Start / stop** — start on word 1; stop as soon as the last word is fixated.
`MAX_FIX = 200` is a non-paper safety cap (counter `simulator.TRUNCATIONS`).

## Not implemented (extended model only)

`iota` (Eq. 22, timer coupling `rate' = rate * (1 + iota * a_k)`) and free
`beta` belong to the 5-parameter extended model. Noted as future work.

## Observable

`f_i = (x_i, y_i)` = (fixated word, fixation duration ms) — Section 4. The
network sees the per-fixation array `[word_id, duration_ms]` plus 7 hand-crafted
reader-level statistics (`config.compute_reader_stats`).
