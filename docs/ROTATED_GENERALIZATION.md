# Rotated Held-Out-Day Generalization Experiment

**Generated:** 2026-09-19 · **Verification:** PASS (15/15)
**Repository commit:** `53c22227d368b82ff9e736206a92e85b6bed2514`
**Results:** [results/rotated_generalization/](../results/rotated_generalization/) ·
**Figures:** [results/rotated_generalization/figures/](../results/rotated_generalization/figures/) ·
**JSON:** [results/rotated_generalization_summary.json](../results/rotated_generalization_summary.json)
**Reproduce:** `python src/run_rotated_generalization.py` ·
**Verify:** `python src/verify_rotated_generalization.py`

> **These are NEW experimental models named `V4_rotated_<day>`.** They do not replace the
> frozen V4 reference. `models/latent_world_model_v4/`, `docs/v4_baseline_manifest.json`,
> `docs/V4_REFERENCE_BASELINE.md` and `results/standardized_baseline_comparison.csv` are
> untouched — the verifier confirms their hashes are unchanged.

---

## 1. Research question

Does the behaviour observed on the Mar-2 held-out day generalize when a *different* day is
held out? Four leave-one-day-out folds, each with a newly trained V4 and newly trained
Logistic Regression models, all fold-isolated.

## 2. Available days

| Day | Windows | Attack windows | Prevalence | Episodes | Observable onsets | Cold start | Attack families (dataset `dominant_attack` label) |
|---|---:|---:|---:|---:|---:|---|---|
| Feb-14 | 3,255 | 1,130 | 0.347 | 3 | 3 | no | FTP-BruteForce (584), SSH-Bruteforce (546) |
| Feb-15 | 3,429 | 341 | 0.099 | 5 | 5 | no | DoS-Slowloris (252), DoS-GoldenEye (89) |
| Feb-28 | 3,406 | 798 | 0.234 | 2 | 2 | no | Infiltration (798) |
| Mar-02 | 3,143 | 2,045 | 0.651 | 4 | 3 | **yes** | Bot (2,045) |

**No attack family appears on more than one day.** Every fold is therefore *simultaneously*
an unseen-day and an unseen-family test — the two effects cannot be separated (§12).

Pooled across folds this gives **14 episodes and 13 observable onsets**, against 3 observable
onsets in the single-day frozen benchmark.

## 3. Fold design

| Fold | Test | Train + validation |
|---|---|---|
| 1 | Feb-14 | Feb-15, Feb-28, Mar-02 |
| 2 | Feb-15 | Feb-14, Feb-28, Mar-02 |
| 3 | Feb-28 | Feb-14, Feb-15, Mar-02 |
| 4 | Mar-02 | Feb-14, Feb-15, Feb-28 |

Within each training day: first 85% chronologically → train, last 15% → validation. No
shuffling. Fold 4 reproduces the frozen benchmark's split exactly (8,441 / 1,473 / 3,115
sequences with 1,835 / 392 / 2,031 positives) — an independent check that the fold builder
matches the frozen pipeline.

| Fold | Train seq | Val seq | Test seq | Train prevalence | Val prevalence | Test prevalence |
|---|---:|---:|---:|---:|---:|---:|
| Feb-14 | 8,346 | 1,456 | 3,227 | 0.295 | 0.457 | 0.350 |
| Feb-15 | 8,198 | 1,430 | 3,401 | 0.379 | 0.566 | 0.100 |
| Feb-28 | 8,267 | 1,434 | 3,328 | 0.337 | 0.472 | 0.240 |
| Mar-02 | 8,441 | 1,473 | 3,115 | 0.217 | 0.266 | 0.652 |

## 4. Preprocessing isolation

Per fold: the 52-feature schema is identical to the frozen one (SHA256
`137b5833…9ecbb5`, verified); relative-change features are computed per source day before
splitting; **both scalers are fitted on that fold's train partition only** and stored in
`results/rotated_generalization/fold_<day>/` with their hashes. The verifier independently
recomputes the expected train-row count for each scaler and confirms it. This deliberately
fixes the provenance mismatch found in Task 1 — here the model is trained and scored under
one and the same normalisation.

## 5. V4 rotated experiment

Architecture and hyperparameters copied verbatim from `src/train_latent_world_model_v4.py`:
10×52 → 2-layer LSTM(128) → LayerNorm → Linear(32) → gated residual transition + LayerNorm →
26-dim decoder + attack head; curriculum K=1→2→3→4→5 at epochs 1/21/41/61/81; γ=0.8; AdamW
lr 5e-4, wd 1e-5; SmoothL1 + BCEWithLogits(pos_weight = train_benign/train_attack); 120
epochs, patience 20 in the final phase; seed 42; batch 64. **Nothing was tuned.**

Two deliberate differences from the frozen training script, both removing leakage:
the held-out day never enters training, and the test-label threshold search present in the
original script's trajectory section is not reused. Checkpoints, history and provenance
(hashes, versions, git commit) are in `models/rotated_v4/fold_test_<day>/`.

## 6. Logistic Regression rotated experiment

Per fold, one model per horizon K=1…5 on `S_t = X[:, -1, :]` using that fold's scaler:
`l2, C=1.0, lbfgs, max_iter=1000, tol=1e-4, class_weight='balanced', random_state=42` — the
frozen Task 2 configuration, no search. All 20 models converged (55–72 iterations).
Stored in `models/rotated_logistic_regression/fold_test_<day>/`.

## 7. Persistence baseline

`ŷ(t+k) = A(t)`, zero parameters, computed per day. It receives the observed label; V4 and
LR receive only features. Its numbers are not comparable one-for-one with theirs.

## 8. Point-forecast results (K=5, per-fold validation thresholds)

| Held-out day | Model | Threshold | F1 | Precision | Recall | ROC-AUC | Test prevalence |
|---|---|---:|---:|---:|---:|---:|---:|
| **Feb-14** | Persistence | — | 0.9912 | 0.9912 | 0.9912 | 0.9932 † | 0.350 |
| | Logistic Regression | 0.50 | 0.6933 | 0.9775 | 0.5372 | 0.8577 | |
| | V4_rotated_Feb14 | 0.05 | 0.6505 | 0.9839 | 0.4858 | **0.9600** | |
| **Feb-15** | Persistence | — | 0.9384 | 0.9384 | 0.9384 | 0.9658 † | 0.100 |
| | Logistic Regression | 0.41 | 0.1406 | 0.1754 | 0.1173 | 0.5767 | |
| | V4_rotated_Feb15 | 0.04 | **0.1208** | 0.0890 | 0.1877 | **0.4936** | |
| **Feb-28** | Persistence | — | 0.9875 | 0.9875 | 0.9875 | 0.9918 † | 0.240 |
| | Logistic Regression | 0.65 | 0.3808 | 0.4275 | 0.3434 | 0.6473 | |
| | V4_rotated_Feb28 | 0.11 | **0.0270** | 0.6471 | 0.0138 | 0.5900 | |
| **Mar-02** | Persistence | — | 0.9966 | 0.9966 | 0.9966 | 0.9950 † | 0.652 |
| | Logistic Regression | 0.54 | 0.4067 | 0.9739 | 0.2570 | 0.9465 | |
| | V4_rotated_Mar02 | 0.17 | 0.4731 | 0.9814 | 0.3117 | 0.8688 | |

† degenerate two-point ROC (binary score), not a ranking AUC.

**V4 F1 at K=5 ranged from 0.0270 (Feb-28) to 0.6505 (Feb-14)** — a 24× spread.
**V4 ROC-AUC ranged from 0.4936 (Feb-15, below chance) to 0.9600 (Feb-14).**

Note the thresholds: the rotated V4 models select 0.04–0.17, versus 0.45–0.72 for the frozen
reference. Their probabilities pile up near zero (median 0.0000 on three of four test days),
a markedly different output distribution from the frozen model's saturated 0.70–0.79 band.

## 9. Trajectory-event results (MAX, K=5)

| Held-out day | Persistence F1 | LR F1 | V4 F1 | V4 threshold | V4 FP/hour | LR FP/hour |
|---|---:|---:|---:|---:|---:|---:|
| Feb-14 | 0.9938 | 0.6939 | 0.6717 | 0.02 | 1.00 | 1.34 |
| Feb-15 | 0.9573 | 0.1370 | 0.0822 | 0.32 | **16.09** | **15.03** |
| Feb-28 | 0.9925 | 0.3946 | 0.0362 | 0.16 | 0.87 | **42.19** |
| Mar-02 | 0.9973 | 0.3727 | 0.5534 | 0.07 | 0.46 | 1.39 |

MAX aggregation is analogous but not architecturally identical: V4 maxes over rollout steps
of one model; LR maxes over five independently trained models; persistence is
horizon-independent.

## 10. Transition-conditioned diagnostic (K=5)

Not an overall benchmark score. Chance reference = each model's own rate of firing.

| Held-out day | Onsets | V4 detected | Chance | LR detected | Chance | V4 offset recall | Transition AUC (V4) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Feb-14 | 10 | 0 | 1.73 | 0 | 1.92 | 5/10 | 0.000 |
| Feb-15 | 21 | 2 | 4.44 | 0 | 1.41 | 19/21 | 0.619 |
| Feb-28 | 10 | 0 | 0.05 | 3 | 1.93 | 10/10 | 0.370 |
| Mar-02 | 7 | 2 | 1.45 | 0 | 1.20 | 1/7 | 0.163 |
| **Pooled** | **48** | **4 (8.3%)** | **7.67** | **3 (6.3%)** | **6.46** | 35/48 (72.9%) | — |

**Pooled across all four days, V4 detected 4 of 48 onset windows where its own firing rate
predicts 7.67 by chance.** Logistic Regression detected 3 of 48 against 6.46 expected. Neither
exceeds its base rate. Label persistence remained ≥ 0.9877 on every day (Feb-14 0.9938,
Feb-15 0.9877, Feb-28 0.9940, Mar-02 0.9955).

## 11. Early-warning results (episode level, trajectory MAX K=5, 120 s lookback)

The episode — not the window — is the unit.

| Held-out day | Observable onsets | V4 warned | V4 coverage | V4 median lead | V4 FA/hour | LR warned | LR coverage | LR FA/hour |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Feb-14 | 3 | 0 | 0.00 | — | 1.00 | 0 | 0.00 | 1.34 |
| Feb-15 | 5 | 1 | 0.20 | 100 s | 16.09 | 2 | 0.40 | 15.03 |
| Feb-28 | 2 | 0 | 0.00 | — | 0.87 | 2 | **1.00** | **42.19** |
| Mar-02 | 3 | 2 | 0.67 | 115 s | 1.04 | 1 | 0.33 | 1.39 |
| **Pooled** | **13** | **3** | **0.23** | **107.5 s** | | **5** | **0.38** | |

Coverage must be read with the false-alarm column: Logistic Regression's perfect 2/2 coverage
on Feb-28 comes with **42 false-alarm windows per hour**, and V4's single Feb-15 warning comes
with 16/hour. Mar-02 is the only day where a warning was produced at a low alarm rate.

## 12. Attack-family analysis

`attack_family_generalization.csv`. Every held-out family is **absent from that fold's
training days** — the four days share no family:

| Held-out day | Family | Seen in training | Test episodes | Test windows | V4 F1 | V4 AUC | LR F1 | Family-specific metrics |
|---|---|---|---:|---:|---:|---:|---:|---|
| Feb-14 | FTP-BruteForce | **No** | 1 | 584 | 0.6505 | 0.9600 | 0.6933 | no (day has 2 families) |
| Feb-14 | SSH-Bruteforce | **No** | 2 | 546 | 0.6505 | 0.9600 | 0.6933 | no |
| Feb-15 | DoS-Slowloris | **No** | 1 | 252 | 0.1208 | 0.4936 | 0.1406 | no |
| Feb-15 | DoS-GoldenEye | **No** | 4 | 89 | 0.1208 | 0.4936 | 0.1406 | no |
| Feb-28 | Infiltration | **No** | 2 | 798 | 0.0270 | 0.5900 | 0.3808 | **yes** |
| Mar-02 | Bot | **No** | 4 | 2,045 | 0.4731 | 0.8688 | 0.4067 | **yes** |

So every result in this document is an unseen-family result. Two consequences: (a) the wide
spread in §8 may be family difficulty, day difficulty, or prevalence shift — this design
cannot separate them; (b) only Feb-28 and Mar-02 are single-family days where a day-level
metric *is* a family-level metric. For Feb-14 and Feb-15 the two families are interleaved and
the day metric cannot be attributed to either.

## 13. Distribution shift

Prevalence shift between train and test is large and fold-dependent:

| Fold | Train prevalence | Test prevalence | Test − train | Test / train |
|---|---:|---:|---:|---:|
| Feb-14 | 0.295 | 0.350 | +0.055 | 1.19 |
| Feb-15 | 0.379 | 0.100 | **−0.279** | 0.26 |
| Feb-28 | 0.337 | 0.240 | −0.097 | 0.71 |
| Mar-02 | 0.217 | 0.652 | **+0.435** | 3.00 |

Largest feature shifts, measured in each fold's own scaled space (what the model actually
sees; train is centred at 0 by construction):

| Fold | Feature | Train mean | Test mean | Shift |
|---|---|---:|---:|---:|
| Mar-02 | `rst_rate` | −0.006 | 2.187 | **+2.19 σ** |
| Feb-28 | `log_unique_dst_ports` | 0.003 | −1.135 | **−1.14 σ** |
| Feb-14 | `rst_rate` | 0.000 | −0.847 | −0.85 σ |
| Mar-02 | `protocol_entropy` | 0.004 | −0.762 | −0.77 σ |
| Feb-15 | `rst_rate` | 0.000 | −0.741 | −0.74 σ |
| Feb-28 | `syn_rate` | 0.002 | 0.701 | +0.70 σ |

Each held-out day sits 0.7–2.2 standard deviations away from its training distribution on at
least one core feature. With four folds no correlation between shift magnitude and F1 can be
established; the two figures `performance_vs_shift.png` plot the four points without a fitted
trend.

## 14. Cross-fold variability

| Model | K | Mean F1 | Median | Std | Min (day) | Max (day) | Mean AUC | Std AUC | Min AUC | Max AUC |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| Persistence | 1 | 0.9947 | — | 0.0062 | 0.9853 | 0.9985 | 0.9965 | 0.0031 | 0.9919 | 0.9984 |
| Persistence | 5 | 0.9784 | 0.9893 | 0.0269 | 0.9384 (Feb-15) | 0.9966 (Mar-02) | 0.9864 | 0.0138 | 0.9658 | 0.9950 |
| Logistic Regression | 5 | 0.4054 | 0.3938 | 0.2263 | 0.1406 (Feb-15) | 0.6933 (Feb-14) | 0.7570 | 0.1738 | 0.5767 | 0.9465 |
| V4 | 1 | 0.3325 | — | 0.3339 | 0.0214 | 0.6329 | 0.7322 | 0.2453 | 0.4931 | 0.9528 |
| V4 | 5 | 0.3178 | 0.2969 | **0.2934** | **0.0270 (Feb-28)** | **0.6505 (Feb-14)** | 0.7281 | **0.2218** | 0.4936 | 0.9600 |

V4's standard deviation across folds (0.29 F1) is almost as large as its mean (0.32).
Persistence's is 0.027.

## 15. Mar-02: frozen reference vs rotated fold

Same test windows, same architecture, same validation-only threshold rule. They differ only
in training provenance: the frozen checkpoint was trained on the single-day
`temporal_benchmark_v2` arrays and scored under a *different* scaler (the Task 1 caveat),
while the rotated model is trained and scored entirely inside its own fold.

| Metric | Historical frozen V4 | Rotated fold V4 | Difference |
|---|---:|---:|---:|
| K=1 F1 | 0.5920 | 0.6092 | **+0.0172** |
| K=1 precision | 0.9410 | 0.9889 | +0.0480 |
| K=1 ROC-AUC | 0.8558 | 0.9345 | **+0.0787** |
| K=2 F1 | 0.6739 | 0.5308 | −0.1431 |
| K=3 F1 | 0.6952 | 0.5059 | −0.1893 |
| K=5 F1 | 0.7523 | 0.4731 | **−0.2792** |
| K=5 recall | 0.6160 | 0.3117 | −0.3043 |
| K=5 precision | 0.9660 | 0.9814 | +0.0154 |
| K=5 ROC-AUC | 0.8877 | 0.8688 | −0.0189 |
| K=5 threshold | 0.72 | 0.17 | −0.55 |

**Neither is ground truth.** The clean-provenance model is *better* at K=1 (F1 +0.017,
AUC +0.079) and substantially *worse* at K=5 (F1 −0.279, driven by recall −0.304 at similar
precision). The frozen reference's characteristic rise in F1 with K does not reappear when
the provenance mismatch is removed: the rotated model's F1 falls from 0.609 (K=1) to 0.473
(K=5). This is evidence that the historical K=5 figure depends on that specific training and
scaling provenance, and it should not be quoted as a horizon effect of the architecture.

## 16. Limitations

1. **Four source days.** The folds are not four independent experiments — every pair of
   training sets shares two days. No significance testing is performed and none is possible.
2. **Day and family are confounded.** Each day carries a unique attack family, so "unseen day"
   and "unseen family" cannot be separated.
3. **2–5 episodes per day, 13 observable onsets in total.** Better than the frozen benchmark's
   3, still far too few for a reliable early-warning estimate.
4. **Label persistence remains ≥ 0.9877 on every day**, so ordinary F1 is still dominated by
   persistent windows — the Task 5 conclusion holds on all four days.
5. **One training run per fold**, one seed. Run-to-run variance is unmeasured and may be
   material given how differently the folds behave.
6. **Feb-14 and Feb-15 mix two families**, so their day-level metrics are not family metrics.
7. Persistence receives `A(t)`; V4 and LR do not.

## 17. Conclusions

### DEMONSTRATED (directly measured)

- **V4's performance varied enormously by held-out day**: K=5 F1 from **0.0270 (Feb-28)** to
  **0.6505 (Feb-14)**, mean 0.318, std 0.293; ROC-AUC from **0.4936 (Feb-15, below chance)** to
  **0.9600 (Feb-14)**, std 0.222. Logistic Regression varied similarly (F1 0.141–0.693).
- **Persistence was stable on every day**: K=5 F1 0.938–0.997, std 0.027, and label persistence
  stayed between 0.9877 and 0.9955. The persistence hypothesis from Task 5 holds on all four days.
- **No day yields onset detection above chance.** Pooled over 48 onset windows, V4 detected 4
  against 7.67 expected at its own firing rate; LR detected 3 against 6.46.
- **Episode-level early-warning coverage is 3/13 for V4 and 5/13 for LR**, and the two highest
  coverage figures come with 16 and 42 false-alarm windows per hour.
- **Every fold is an unseen-family test** — no attack family is shared between days.
- **The Mar-02 clean-provenance rerun does not reproduce the frozen reference's K=5 result**
  (0.4731 vs 0.7523) while slightly exceeding it at K=1.

### SUGGESTED (patterns observed, not established)

- The two worst folds (Feb-15, Feb-28) are the two with the lowest test prevalence relative to
  training and the largest negative prevalence shift; Feb-14, with the smallest shift, is the
  best. With four points this is a pattern, not a relationship.
- The rotated models' probability mass collapses toward zero on the held-out day (median 0.0000
  on three of four days, thresholds selected at 0.04–0.17), suggesting the attack head does not
  transfer its calibration to an unseen day/family — consistent with the LR threshold-transfer
  finding from Task 2.
- V4's clear advantage over LR on Mar-02 (+0.066 F1) does not persist elsewhere: LR is higher on
  Feb-14 (+0.043), Feb-15 (+0.020) and Feb-28 (+0.354).

### UNKNOWN (cannot be determined from four days)

- Whether V4 generalizes to new days in general. **The evidence does not support the statement
  "V4 generalizes"**, but four confounded folds also cannot establish that it does not.
- Whether the variation is caused by the day, the attack family, the prevalence shift, or the
  single training run — all four vary together.
- Whether any architecture change would help, since the failure mode (near-zero probabilities on
  unseen families) may be a data/objective problem rather than a capacity problem.
- Whether the frozen reference's K-trend is a real property of the architecture: the rotated
  Mar-02 rerun suggests it is at least partly provenance-dependent.
