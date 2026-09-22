# Long-Horizon Rollout and Latent Stability
# Long-Horizon Rollout and Latent Stability
# Long-Horizon Rollout and Latent Stability
# Long-Horizon Rollout and Latent Stability
# Long-Horizon Rollout and Latent Stability
# Long-Horizon Rollout and Latent Stability
# Long-Horizon Rollout and Latent Stability
# Long-Horizon Rollout and Latent Stability
# Long-Horizon Rollout and Latent Stability

**Task 10.** An inference-only diagnostic: how far can the frozen V4 learned transition be rolled forward before its latent state or its threat predictions stop carrying useful information?

| | |
|---|---|
| **Experiment type** | Inference only — nothing trained, tuned or modified |
| **Horizons** | K = 1…10 (10 s per step) |
| **Trained curriculum depth** | **K = 5** — `curriculum = [(1,1),(21,2),(41,3),(61,4),(81,5)]` |
| **V4 status** | **unchanged** — checkpoint, scalers and schema hash-verified before and after every run |
| **Numerical stability** | **STABLE at every horizon through K = 10** |
| **Forecasting informativeness** | **No horizon beats a nowcast.** K = 9, 10 are *anti-informative* |
| **Verification** | 12 / 12 checks PASS |

> **The model was trained with curriculum depth through K=5; K=6…10 are recursive extrapolation diagnostics.** K=10 is a ten-step recursive application of a transition that was never optimised at that depth. It is **not** a trained 100-second forecast, and it is not described as one anywhere in this document.

---

## 1. Purpose

V4's transition was trained to a curriculum depth of five steps. The question is not whether rolling further produces a number — it always will — but whether that number still means anything. Two things could go wrong, and they are independent:

1. **Numerical instability** — the latent explodes, collapses to a point, or oscillates, so the decoded state and probability become arbitrary.
2. **Loss of information** — the rollout stays perfectly well-behaved but stops telling you anything about the future that you did not already know from the present.

This task measures both. The answers are different: V4 is numerically stable to K=10 and informationally uninformative at every horizon including K=1.

## 2. Frozen V4 reference

Verified at the start **and** end of every pipeline run; a mismatch aborts before anything is written.

| Property | Value |
|---|---|
| Checkpoint SHA-256 | `76af8d8a9e02215923b1c0ab6b627bfc0809e856748666a55ee4b16cec5e5898` ✓ |
| Model class | `LatentWorldModelV4` ✓ |
| Parameter count | 244,027 ✓ |
| Input dimension | 52 ✓ |
| Sequence length | 10 ✓ |
| Latent dimension | 32 ✓ |
| Decoder output | 26 ✓ |
| Loads `strict=True` | ✓ |
| `input_scaler.pkl` | `e0ab9beb…` ✓ |
| `target_scaler.pkl` | `7155dec4…` ✓ |
| Feature schema | `2b4f820c…` ✓ |

Ten pre-existing artefacts, including `results/standardized_baseline_comparison.csv`, are hash-identical after the run. All Task 10 output is confined to `results/long_horizon_rollout/` and `results/long_horizon_rollout_summary.json`.

## 3. Extended rollout definition

```
z_t       = encode(X_t)                  X_t is the ONLY observation ever consumed
z_{t+k+1} = T(z_{t+k})                   k = 0 … 9
P(t+k)    = sigmoid(attack_head(z_{t+k}))
S(t+k)    = decoder(z_{t+k})
```

The transition function is untouched. No observed state is injected at any step — each transition consumes the **predicted** latent from the step before.

**This is verified constructively, not asserted.** Check 5 of the verifier rebuilds the entire ten-step trajectory from `z_t` using only `transition_step` and requires **bitwise** equality with what the pipeline stored; it also confirms every probability and decoded state is the head/decoder applied to that same latent. Check 6 confirms `rollout()` accepts no label or target argument, and that latents and decoded states are bitwise invariant to sample reordering and batch truncation. (Probabilities move by at most 1.19 × 10⁻⁷ — one float32 ULP in the batched attack head — and change **zero** decisions at any frozen threshold.)

## 4. K = 1…5 — the reference region

These horizons sit inside the training curriculum and reproduce the frozen V4 reference **exactly**, on CPU, to within 10⁻¹², with identical confusion-matrix counts:

| K | Threshold | Source | F1 | Precision | Recall | ROC-AUC | FPR |
|---|---|---|---|---|---|---|---|
| 1 | 0.45 | **frozen V4** | 0.5920 | 0.9410 | 0.4318 | 0.8558 | 0.0507 |
| 2 | 0.54 | **frozen V4** | 0.6739 | 0.9507 | 0.5219 | 0.8774 | 0.0507 |
| 3 | 0.65 | **frozen V4** | 0.6952 | 0.9609 | 0.5446 | 0.8893 | 0.0415 |
| 4 | 0.69 | exploratory | 0.7460 | 0.9686 | 0.6066 | 0.8902 | 0.0369 |
| 5 | 0.72 | **frozen V4** | 0.7523 | 0.9660 | 0.6160 | 0.8877 | 0.0406 |

**K=4 is inside the curriculum but has no frozen number** — the reference script evaluated horizons [1, 2, 3, 5] only. Its threshold is validation-selected and is marked exploratory, never presented as a reference value.

## 5. K = 6…10 — the extrapolative region

| K | Threshold | Source | F1 | Precision | Recall | ROC-AUC | PR-AUC | FPR | n |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.71 | exploratory | 0.7973 | 0.9630 | 0.6803 | 0.8871 | 0.9317 | 0.0489 | 3113 |
| 7 | 0.69 | exploratory | 0.8205 | 0.9579 | 0.7176 | 0.8902 | 0.9363 | 0.0591 | 3111 |
| 8 | 0.32 | exploratory | 0.8656 | 0.9293 | 0.8102 | 0.8964 | 0.9392 | 0.1156 | 3109 |
| 9 | 0.72 | exploratory | 0.8374 | 0.9634 | 0.7405 | 0.9003 | 0.9385 | 0.0528 | 3107 |
| 10 | 0.72 | exploratory | 0.8475 | 0.9560 | 0.7611 | 0.9003 | 0.9381 | 0.0658 | 3105 |

**Threshold policy.** K = 1, 2, 3, 5 use the frozen V4 thresholds, reproduced and never re-derived. K = 4 and K = 6…10 have no frozen threshold, so one is selected on **validation only**, using the identical grid search the frozen reference used (`np.arange(0.01, 0.99, 0.01)`, maximise validation F1, strict `>`). Test labels were never consulted. The verifier re-derives every exploratory threshold from validation and confirms the match — and confirms they leave 0.0082 to 0.1294 F1 on the table versus a test-optimal threshold, which is the expected signature of honest selection.

**Ground truth for K=6…10** was reconstructed from the same `states_v2` tables the benchmark was built from, requiring the same strict 10 s contiguity. The reconstruction reproduces the frozen K=1…5 label arrays exactly (the pipeline aborts otherwise). 10 of 3,115 sequences lack a contiguous K=10 target and are excluded at that horizon rather than imputed.

> ⚠️ **The F1 column is not like-for-like across K**, because K=1,2,3,5 use frozen thresholds and the rest use exploratory ones. ROC-AUC and PR-AUC are threshold-free and *are* comparable.

## 6. Horizon performance

ROC-AUC does not degrade: 0.856 at K=1, rising to 0.890 by K=3 and sitting between 0.887 and 0.900 all the way to K=10. PR-AUC rises monotonically, 0.891 → 0.938.

**Taken alone this looks like the rollout stays useful to 100 seconds. It does not, and §12 and §13 explain why.** The target barely changes over that interval, so a metric computed against it is mostly measuring persistence.

## 7. Trajectory-event analysis

Target `max(A(t+1)…A(t+K))`, score `max(P(t+1)…P(t+K))`. K=2, 3, 5 reproduce the frozen reference exactly.

| K | Threshold | Source | F1 | ROC-AUC |
|---|---|---|---|---|
| 2 | 0.51 | **frozen V4** | 0.7127 | 0.8780 |
| 3 | 0.68 | **frozen V4** | 0.6371 | 0.8904 |
| 5 | 0.69 | **frozen V4** | 0.7979 | 0.8929 |
| 6 | 0.73 | exploratory | 0.7851 | 0.8942 |
| 7 | 0.73 | exploratory | 0.8031 | 0.8971 |
| 8 | 0.73 | exploratory | 0.8184 | 0.9011 |
| 9 | 0.72 | exploratory | 0.8422 | 0.9047 |
| 10 | 0.73 | exploratory | 0.8464 | 0.9066 |

Because the target is a running maximum over an already-persistent label, it becomes *easier* as K grows — the positive class approaches the base rate. This series should not be read as improving forecast quality.

## 8. Latent stability

Per-sample, per-step measurements over 3,115 test sequences × 11 steps = 34,265 rows (`latent_stability.csv`).

| k | ‖z‖ mean | ‖z‖ p99 | step size ‖z_k − z_{k−1}‖ | displacement ‖z_k − z_0‖ | cos(z_k, z_{k−1}) | across-sample std |
|---|---|---|---|---|---|---|
| 0 | 5.7242 | 6.8304 | — | 0.0000 | — | 0.7005 |
| 1 | 6.3761 | 6.5780 | 2.9677 | 2.9677 | 0.8827 | 0.7856 |
| 3 | 6.3673 | 6.5837 | 2.0952 | 5.7662 | 0.9420 | 0.7805 |
| 5 | 6.3522 | 6.5568 | 1.4463 | 6.6565 | 0.9728 | 0.7668 |
| 7 | 6.3296 | 6.5182 | 1.1465 | 6.8744 | 0.9825 | 0.7538 |
| 10 | 6.3008 | 6.4711 | 0.7400 | 7.0621 | 0.9922 | 0.7123 |

Using the operational definitions in §16, the behaviour is **saturating and contractive**, not divergent, oscillatory or chaotic:

- **Magnitude is bounded.** ‖z‖ moves from 6.376 to 6.301 across ten steps — a 1.2 % drift. This is largely architectural: the transition ends in `LayerNorm`, which fixes the latent RMS by construction. A bounded norm here is therefore weak evidence about the learned dynamics and strong evidence about the architecture.
- **Step size shrinks monotonically**, 2.968 → 0.740 (24.9 % of the first step).
- **Displacement saturates** at ≈ 7.06 and is nearly flat after K≈6.
- **Consecutive cosine similarity rises to 0.992** — each step barely changes direction.
- **No collapse.** Across-sample std falls only 0.786 → 0.712, so samples remain distinguishable from one another. Cosine similarity to the *initial* state falls to 0.306, so the trajectory does travel a long way — it just stops moving.

The word *chaotic* is not used: nothing measured supports it.

## 9. Latent contraction / divergence

| Quantity | Value |
|---|---|
| Mean step-1 displacement | 2.9677 |
| Mean step-10 displacement | 0.7400 (24.9 % of step 1) |
| Mean contraction ratio ‖d_{k+1}‖/‖d_k‖ | 0.8619 |
| Samples contracting at *every* step | 45.0 % |
| Per-sample verdict: `observed_contraction` | 2,321 / 3,115 (74.5 %) |
| Per-sample verdict: `non_contracting` | 794 / 3,115 (25.5 %) |

Pre-declared per-sample rules: `near_fixed_point` if the final step is < 5 % of the first; `observed_contraction` if the final step is ≤ 50 % of the first **and** ≥ 80 % of consecutive ratios are < 1; `non_contracting` otherwise. No sample reached `near_fixed_point` within ten steps.

> **This is OBSERVED contraction, not proven convergence.** No Lipschitz constant is established and no limit is computed. A separate diagnostic extending to K=40 (not part of the deliverables) shows the step size continuing to fall to 0.143 while ‖z‖ settles at 6.251 — consistent with approach to a fixed point or a slow manifold, but still not a proof. Because the transition terminates in LayerNorm, a bounded and shrinking step sequence is the *expected architectural* behaviour, not necessarily a learned attractor.

## 10. Probability stability

| Quantity | Value |
|---|---|
| Mean P(t+1) → mean P(t+10) | 0.2295 → 0.4507 |
| Mean drift P(t+10) − P(t+1) | **+0.2210** |
| Monotonically increasing over K=1…10 | 18.5 % of samples |
| Monotonically decreasing | 0.7 % |
| Mean oscillation count (sign changes) | 1.33 |
| Mean probability range | 0.2439 |
| Samples with any saturated step (≤0.02 or ≥0.98) | 50.0 % |

The dominant pattern is a **systematic upward drift**, not instability. Probabilities do not blow up or thrash; they slide toward a mid-range value as the latent settles. Because the positive prevalence is 0.652, drifting upward mechanically improves recall and F1 — which is a property of the drift meeting the base rate, not of better forecasting.

## 11. Forecast error

State targets are the 26-dimensional state in `target_scaler` space. **The transition is a deterministic point estimate in latent space; it does not predict a distribution over states**, and nothing here should be read as a predictive distribution.

| K | V4 state RMSE | Persistence RMSE | Train-mean RMSE | Skill vs mean | Brier | Brier skill vs base rate |
|---|---|---|---|---|---|---|
| 1 | 1.1218 | 0.8095 | 1.0083 | **−0.1126** | 0.3472 | −0.5303 |
| 3 | 1.0052 | 0.9763 | 1.0085 | +0.0032 | 0.2210 | +0.0258 |
| 5 | 1.0294 | 1.0120 | 1.0122 | **−0.0170** | 0.1867 | +0.1770 |
| 7 | 1.0313 | 1.0089 | 1.0121 | −0.0190 | 0.1728 | +0.2381 |
| 10 | 1.0041 | 1.0442 | 1.0120 | +0.0078 | 0.1596 | +0.2962 |

Two findings, both uncomfortable and both reported plainly:

1. **State error does not grow with horizon.** RMSE goes 1.1218 → 1.0041, a factor of 0.895 — it *falls*. It does not grow linearly, does not grow rapidly; it saturates immediately.
2. **That is not a success.** The decoded state **never meaningfully beats predicting the training mean at any horizon**, and is *worse* than the mean at K=1–2. Persistence (copy the last observed state) beats V4 at K=1–7. The error is flat because the decoded state contracts toward a near-constant low-variance output: its across-sample std falls from 0.488 at k=1 to 0.257 at k=10, against a target std of 0.757. Error stops growing because the prediction stops varying.

The Brier score improves with K for the same reason the F1 does — probability drift meeting a 65 % base rate — and the "skill vs base rate" column is positive only from K=3 onward.

## 12. Persistence context

| K | Persistence accuracy | Transition fraction | Onsets | Offsets | Prevalence |
|---|---|---|---|---|---|
| 1 | 0.9981 | 0.0019 | 3 | 3 | 0.6520 |
| 3 | 0.9961 | 0.0039 | 6 | 6 | 0.6520 |
| 5 | 0.9955 | 0.0045 | 7 | 7 | 0.6520 |
| 7 | 0.9942 | 0.0058 | 9 | 9 | 0.6522 |
| 10 | **0.9923** | **0.0077** | 12 | 12 | 0.6525 |

**Predicting "A(t+k) = A(t)" is 99.23 % accurate at K=10.** Only 0.77 % of windows change label within 100 seconds. Every aggregate metric in §6 and §7 is therefore dominated by label persistence rather than by forecasting skill, and no horizon-vs-performance curve in this task should be read without this table beside it.

## 13. Transition analysis

Conditioning on what actually changed:

| K | Onsets | Offsets | mean P \| onset | mean P \| persistent-negative | mean P \| offset | mean P \| persistent-positive |
|---|---|---|---|---|---|---|
| 1 | 3 | 3 | **0.4005** | 0.0408 | 0.5939 | 0.3292 |
| 3 | 6 | 6 | 0.3692 | 0.0545 | 0.7145 | 0.5106 |
| 5 | 7 | 7 | **0.2179** | 0.0649 | 0.7554 | 0.5769 |
| 7 | 9 | 9 | 0.1774 | 0.0763 | 0.7905 | 0.6122 |
| 10 | 12 | 12 | **0.1320** | 0.0927 | 0.7962 | 0.6402 |

The onset probability **falls by 67 %** from K=1 to K=10 while the persistent-negative probability *rises*. The separation between them — the quantity that would let you flag an impending attack — collapses from 0.153 at K=5 to 0.039 at K=10.

> **Statistical power: LOW at every horizon.** Onsets range from 3 to 12 and offsets from 3 to 12; the pre-declared threshold for adequacy was 20 in the smaller class. No individual value here is reliable. What *is* notable is that the direction is identical and monotone across all ten horizons, which is harder to attribute to noise than any single value.

### Is it a forecast, or a nowcast?

The transition statistics are underpowered, so the same question was asked with the **full** test set behind it: does `P(t+k)` predict the **future** label `A(t+k)` better than it predicts the **current** label `A(t)`?

| K | AUC vs A(t+K) | AUC vs A(t) | Incremental | Onset-stratum AUC (n⁺) | Verdict |
|---|---|---|---|---|---|
| 1 | 0.8558 | 0.8569 | **−0.0011** | 0.6540 (3) | NOWCAST_EQUIVALENT |
| 3 | 0.8893 | 0.8927 | −0.0033 | 0.5057 (6) | NOWCAST_EQUIVALENT |
| 5 | 0.8877 | 0.8928 | −0.0051 | 0.3450 (7) | NOWCAST_EQUIVALENT |
| 7 | 0.8902 | 0.8980 | −0.0078 | 0.3434 (9) | NOWCAST_EQUIVALENT |
| 9 | 0.9003 | 0.9107 | −0.0104 | 0.3450 (11) | **ANTI_INFORMATIVE** |
| 10 | 0.9003 | 0.9119 | **−0.0116** | 0.3750 (12) | **ANTI_INFORMATIVE** |

Both AUCs use all ~3,105 samples, so this comparison is **fully powered**. The incremental AUC is negative at every horizon and worsens monotonically. Restricted to the windows where nothing is currently happening — the only place a forecast could earn its keep — the AUC sits **below chance** (0.34–0.38) from K=4 onward.

**The rollout is a description of the present that scores well on the future because the label persists.** That is the central finding of this task.

## 14. Early-warning analysis

Episodes defined exactly as the frozen reference defines them. The K=5 row reproduces the frozen V4 early-warning result (threshold 0.69, 2 of 3 warned, 50 s mean lead, 5.55 FP/hr).

| K | Threshold | Source | Warned | Mean lead | FP/hr |
|---|---|---|---|---|---|
| 1 | 0.45 | frozen | 2/3 | 10 s | 6.36 |
| 3 | 0.68 | frozen | 2/3 | 25 s | 3.93 |
| 5 | 0.69 | **frozen V4** | 2/3 | 50 s | 5.55 |
| 7 | 0.73 | exploratory | 2/3 | 70 s | 5.78 |
| 10 | 0.73 | exploratory | 2/3 | 100 s | 6.93 |

> ⚠️ **There are 3 observable onsets in the entire test day, and one attack was already active at the start.** Coverage is 2/3 at every horizon — the same two episodes. "Mean lead time" grows only because the search window grows: a longer horizon mechanically permits an earlier alert, at a rising false-alarm rate (5.55 → 6.93 FP/hr). **None of this is an early-warning estimate.** It is three data points, reported for completeness and relied upon for nothing.

## 15. K = 5 vs K = 10

| Quantity | K=5 | K=10 | Δ |
|---|---|---|---|
| Latent norm | 6.3522 | 6.3008 | −0.0514 |
| Latent step size | 1.4463 | 0.7400 | **−0.7062** |
| Cumulative displacement | 6.6565 | 7.0621 | +0.4056 |
| Across-sample latent std | 0.7668 | 0.7123 | −0.0545 |
| Decoded-state RMSE | 1.0294 | 1.0041 | −0.0253 |
| Skill vs mean predictor | −0.0170 | +0.0078 | +0.0248 |
| ROC-AUC | 0.8877 | 0.9003 | +0.0127 |
| PR-AUC | 0.9256 | 0.9381 | +0.0125 |
| F1 † | 0.7523 | 0.8475 | +0.0952 |
| Recall † | 0.6160 | 0.7611 | +0.1452 |
| Persistence accuracy | 0.9955 | 0.9923 | −0.0032 |
| **Mean P \| onset** | **0.2179** | **0.1320** | **−0.0859** |
| **Onset separation** | **0.1530** | **0.0393** | **−0.1138** |
| False alarms / hour | 5.55 | 6.93 | +1.39 |

† K=5 uses the frozen threshold and K=10 an exploratory one — **not like-for-like**.

Everything that looks like improvement (AUC, F1, recall, Brier) is explained by probability drift meeting a 65 % base rate on a label that barely changes. Everything that measures actual forecasting — onset probability, onset separation, incremental AUC over a nowcast — moves the other way.

## 16. Practical stability boundary

Two boundaries, because there are two questions. Both rule sets were declared before being applied and are evaluated arithmetically; neither is a subjective judgement and neither is test F1 alone.

### 16a. Numerical stability — `horizon_status.csv`

Each indicator is relative to the same quantity's K=1 value, so this measures *degradation with horizon*.

| Indicator | Rule | Counted |
|---|---|---|
| `latent_divergence` | `‖z‖p99(K) ≤ 1.5 × ‖z‖p99(K=1)` | yes |
| `latent_collapse` | `across-sample std(K) ≥ 0.5 × (K=1)` | yes |
| `decoder_error_growth` | `state_rmse(K) ≤ 1.25 × (K=1)` | yes |
| `probability_discrimination` | `roc_auc(K) − 0.5 ≥ 0.8 × (roc_auc(K=1) − 0.5)` | yes |
| `probability_saturation` | `std(P_K) ≥ 0.5 × std(P_1)` **and** `frac(0.02 < P < 0.98) ≥ 0.5` | yes |
| `onset_separation` | `mean P\|onset ≥ mean P\|persistent-negative` | **advisory** |

`onset_separation` is advisory by rule, not by convenience: onset counts are 3–12, far below the pre-declared `MIN_TRANSITION_N = 20`, so it has too little power to decide a status on its own.

**STABLE = 0 counted failures; DEGRADING = 1–2; UNRELIABLE = ≥ 3.**

**Result: every horizon K = 1…10 is STABLE, with 0 counted indicators failed.** The extended rollout does not break.

### 16b. Informativeness — `nowcast_vs_forecast.csv`

| Verdict | Rule |
|---|---|
| `FORECAST_INFORMATIVE` | `beats_nowcast` passes |
| `NOWCAST_EQUIVALENT` | `beats_nowcast` fails but `|incremental_auc| ≤ 0.01` |
| `ANTI_INFORMATIVE` | `incremental_auc < −0.01` |

**Result: K = 1…8 are NOWCAST_EQUIVALENT; K = 9 and K = 10 are ANTI_INFORMATIVE. No horizon is FORECAST_INFORMATIVE.**

### The boundary

- **Numerical stability boundary: K > 10** — not reached within the tested range.
- **Informativeness boundary: crossed at K = 9**, where the rollout starts describing the present measurably better than the future it is meant to predict. The degradation is monotone from K=1, so this is a threshold on a trend, not a cliff.
- **Practical operating horizon: K = 5.** Not because K=6…10 are unstable — they are not — but because K=5 is the deepest horizon that is inside the training curriculum, carries a frozen validation-selected threshold, and has a published reference result. Rolling further buys no forecasting information and costs onset separation and false-alarm rate.

## 17. Representative cases

Selected by criteria declared before the data was examined, resolved by argmin/argmax over the whole test set with ties broken by lowest sample id. Failure cases are mandatory members.

| Case | id | A(t) → A(t+10) | P(t+1) | P(t+5) | P(t+10) | Displacement |
|---|---|---|---|---|---|---|
| stable_rollout | 1691 | 0 → 0 | 0.000 | 0.000 | 0.000 | 4.311 |
| gradual_drift | 2947 | 1 → 1 | 0.623 | 0.716 | 0.788 | 7.020 |
| strong_latent_contraction | 1691 | 0 → 0 | 0.000 | 0.000 | 0.000 | 4.311 |
| strong_latent_divergence | 1336 | 0 → 0 | 0.000 | 0.046 | 0.553 | 9.291 |
| probability_saturation | 1418 | 0 → 0 | 0.001 | 0.019 | 0.007 | 8.075 |
| probability_oscillation | 2137 | 1 → 1 | 0.001 | 0.002 | 0.003 | 7.585 |
| correct_k5_degraded_k10 | 1243 | 0 → 0 | 0.025 | 0.229 | **0.725** | 7.754 |
| incorrect_k5_and_k10 | 2133 | **1 → 1** | 0.000 | 0.000 | 0.000 | 6.693 |
| **transition_onset** | 2110 | **0 → 1** | 0.000 | 0.001 | **0.000** | 7.247 |

Two are worth reading closely:

- **id 1243** (`correct_k5_degraded_k10`): quiet traffic that stays quiet. The probability drifts from 0.025 to 0.725, crossing the K=10 exploratory threshold and producing a false alarm that did not exist at K=5. This is the extended rollout manufacturing confidence.
- **id 2110** (`transition_onset`): a genuine attack onset within the horizon. The probability is 0.000 at K=1, 0.001 at K=5 and 0.000 at K=10. The extra five rollout steps deliver nothing on the one case type they would need to get right.

The same sample (1691) satisfies both `stable_rollout` and `strong_latent_contraction`; the criteria are independent and both selected it honestly.

## 18. Limitations

1. **Provenance, inherited not resolved.** The frozen V4 checkpoint was trained on the single-day `temporal_benchmark_v2` arrays but is evaluated on `multiday_benchmark_K5` arrays scaled by a *different* `StandardScaler` (`docs/v4_baseline_manifest.json` → `known_issues`). Every number in this task inherits that mismatch. **This experiment does not repair it and does not resolve it.** It plausibly contributes to the decoder's poor absolute error in §11, and that hypothesis is untested here.
2. **Single test day.** 3,115 sequences from Friday-02-03-2018 with 3 observable onsets. Every early-warning and transition figure is anecdotal.
3. **Transition statistics are underpowered** — 3 to 12 onsets. The consistency of the trend across ten horizons is the evidence, not any single value.
4. **Aggregate metrics are persistence-dominated** (99.23 % at K=10). Reading ROC-AUC alone across horizons is actively misleading.
5. **The F1 series is not like-for-like** across the frozen/exploratory threshold boundary.
6. **Ground truth for K=6…10 was reconstructed**, not read from a frozen array. It reproduces K=1…5 exactly, which is the strongest available check, but it is a reconstruction.
7. **The stability boundary is relative to each quantity's own K=1 value.** It measures degradation with horizon and says nothing about whether V4 is good in absolute terms — on the state-decoder arm, it is not.
8. **Bounded latent norm is partly architectural.** LayerNorm at the end of the transition fixes the latent RMS, so stability of ‖z‖ is weak evidence about the learned dynamics.

## 19. Conclusions

**V4 remains numerically stable well beyond its trained horizon, and simultaneously fails to carry forecasting information at any horizon — including the ones it was trained on.** Those two statements are not in tension; they answer different questions.

The extended rollout is well-behaved: magnitude bounded, steps shrinking, no oscillation, no collapse, no chaos. If the goal were to show that the learned transition can be iterated safely, that is demonstrated to K=10 and, in an auxiliary diagnostic, to K=40.

But a transition that contracts toward a fixed point cannot forecast. Every additional step moves the latent less, the decoded state becomes more nearly constant (across-sample std 0.488 → 0.257 against a target std of 0.757), and the probability drifts toward a value that scores well against a 65 % base rate on a label that changes 0.77 % of the time. The one measurement with full statistical power — incremental AUC over a pure nowcast — is negative at every horizon and worsens monotonically to −0.0116 at K=10.

**K=5 should be treated as the practical operating horizon**, and it should be understood as the deepest *validated* horizon rather than as a demonstrated 50-second forecasting capability.

---

### Artefacts

| Path | Contents |
|---|---|
| `results/long_horizon_rollout/horizon_performance.csv` | K=1…10 point metrics, thresholds, frozen-reference reproduction flags |
| `results/long_horizon_rollout/trajectory_performance.csv` | Max-aggregated event metrics |
| `results/long_horizon_rollout/latent_stability.csv` | Per sample × step: norm, RMS, mean, variance, deltas, cosines, decoder error |
| `results/long_horizon_rollout/latent_stability_summary.csv` | mean / median / p90 / p95 / p99 / max per horizon |
| `results/long_horizon_rollout/latent_contraction.csv` | Contraction ratios and per-sample verdicts |
| `results/long_horizon_rollout/probability_stability.csv` | Per-sample monotonicity, oscillation, saturation, crossings |
| `results/long_horizon_rollout/forecast_error.csv` | State MAE/RMSE and Brier, with persistence and mean baselines |
| `results/long_horizon_rollout/persistence_context.csv` | Persistence accuracy, transition fraction, onset/offset counts |
| `results/long_horizon_rollout/transition_metrics.csv` | Transition-conditioned probabilities and power labels |
| `results/long_horizon_rollout/nowcast_vs_forecast.csv` | **The fully powered forecast-vs-nowcast comparison** |
| `results/long_horizon_rollout/extended_warning_analysis.csv` | Per-episode lead times and false-alarm rates |
| `results/long_horizon_rollout/k5_vs_k10.csv` | The K=5 vs K=10 diagnostic |
| `results/long_horizon_rollout/horizon_status.csv` | Numerical stability boundary and its indicators |
| `results/long_horizon_rollout/representative_cases.csv` | Nine cases with their selection criteria |
| `results/long_horizon_rollout/figures/` | 13 figures, all generated from the CSVs |
| `results/long_horizon_rollout_summary.json` | Machine-readable summary |
| `src/run_long_horizon_rollout.py` | The pipeline |
| `src/verify_long_horizon_rollout.py` | The 13 checks |
