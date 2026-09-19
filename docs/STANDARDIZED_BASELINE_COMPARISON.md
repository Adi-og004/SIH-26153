# Standardized Baseline Comparison
## Persistence vs Logistic Regression vs V4 World Model

**Generated:** 2026-09-19 · **Verification:** PASS
**Repository commit:** `53c22227d368b82ff9e736206a92e85b6bed2514` (branch `cicflowmeter-validation`)
**V4 reference commit:** `2be30736872dd7bb072b2e5c4cf7e7fe62605af6` (pinned — see §13)
**Data:** [results/standardized_baseline_comparison.csv](../results/standardized_baseline_comparison.csv) ·
[results/persistence_hypothesis_analysis.csv](../results/persistence_hypothesis_analysis.csv) ·
[results/comparison_verification.json](../results/comparison_verification.json) ·
[docs/standardized_comparison_manifest.json](standardized_comparison_manifest.json)
**Figures:** [results/figures/](../results/figures/)
**Reproduce:** `python src/run_standardized_comparison.py`

No model was trained, retrained, re-thresholded or modified to produce this document.
Both frozen baselines still verify PASS.

---

## 1. Purpose

One question, evaluated on one held-out day under one set of metric definitions:

> Does the V4 temporal World Model provide forecasting behaviour that cannot already be
> explained by a simple persistence rule or a conventional static Logistic Regression
> classifier?

This document reports numerical differences. It deliberately does **not** rank the three
approaches or declare a winner — §11 explains what the numbers do and do not establish.

## 2. Common benchmark

All three approaches are evaluated on **exactly the same 3,115 test sequences** with
**exactly the same targets** from `data/processed/multiday_benchmark_K5`
(the frozen V4 benchmark, unchanged). Verified, not assumed:

| Property | Verified value |
|---|---|
| Test examples per model | 3,115 (TP+FP+TN+FN identical for all three) |
| Test targets per horizon | identical positive counts across models |
| Horizon alignment | `A_binary_multistep[:, k-1]` = `attack_present(t+k)`, re-checked against the raw Mar-2 CSV for k = 1…5 |
| Feature schema | `input_feature_names.txt`, SHA256 `137b5833…9ecbb5` (52 = 26 base + 26 relative) |
| Input scaler | `input_scaler.pkl`, SHA256 `e0ab9beb…f04f19`, `n_samples_seen_ = 8575` (train partition only) |

## 3. Train / validation / test split

| Split | Source days | Sequences | Attack (t+1) | Positive rate |
|---|---|---:|---:|---:|
| Train | Feb-14, Feb-15, Feb-28 — first 85% of each day | 8,441 | 1,835 | 21.7% |
| Validation | Feb-14, Feb-15, Feb-28 — last 15% of each day | 1,473 | 392 | 26.6% |
| **Test** | **Mar-2-2018 only** | **3,115** | **2,031** | **65.2%** |

The Mar-2 day never appears in train or validation; validation is chronologically after
train within every training day; no sequence key `(source_day, timestamp)` is duplicated
within or across splits. The ~2.5× prevalence jump from validation to test matters in §11.

## 4. Target definition

| Model target | Array | Meaning |
|---|---|---|
| K=1 | `A_binary_multistep[:, 0]` | `A(t+1)` |
| K=2 | `A_binary_multistep[:, 1]` | `A(t+2)` |
| K=3 | `A_binary_multistep[:, 2]` | `A(t+3)` |
| K=5 | `A_binary_multistep[:, 4]` | `A(t+5)` |
| Trajectory K | `max(A_binary_multistep[:, :K])` | any attack within `t+1 … t+K` |

`A(τ)` is 1 when any flow inside the 10-second window at τ is labelled non-Benign.

**K=4 is excluded from the three-model comparison** — the frozen V4 reference evaluates
horizons {1, 2, 3, 5} only, so no V4 K=4 benchmark number exists. Logistic Regression K=4
appears in the CSV as `point_forecast_secondary_lr_only` and must never be read as a
V4-comparable row.

## 5. Persistence methodology

| Property | Value |
|---|---|
| Implementation | the label-persistence rule in `src/evaluate_baselines.py`, reconstructed in `src/run_standardized_comparison.py` |
| Input | `A(t)` — the **observed attack label** at the current window, read from `data/processed/states_v2/<day>_states_v2.csv` at `(metadata timestamp − 10 s)` |
| Prediction rule | `ŷ(t+k) = A(t)` for every k |
| Trajectory rule | the rule is horizon-independent, so the trajectory score is also `A(t)` |
| Threshold | **none** — deterministic 0/1 output, no tunable parameter |
| Trained parameters | 0 |
| Future information | none — only the already-observed current label |

**ROC-AUC handling.** The persistence score is binary, so its ROC curve has a single
interior point: the reported "ROC-AUC" equals balanced accuracy, not a ranking AUC. It is
flagged `roc_auc_degenerate = True` in the CSV and must not be compared on equal footing
with the probabilistic AUCs of the other two models.

**Critical asymmetry.** Persistence consumes the ground-truth label `A(t)`. Logistic
Regression and V4 receive only the 52 numeric features and are never told whether an attack
is currently in progress. This structurally favours persistence and is inherent to the rule.

**Excluded as incompatible:** `archive/train_persistence_baseline.py` and
`models/world_model/persistence_baseline/*` implement *state-regression* persistence
(MAE/RMSE/R², 19 features, the older `temporal_benchmark`). They do not forecast attack
labels and are not on this benchmark, so they are not used here.

## 6. Logistic Regression methodology

Frozen Task 2 artifacts, consumed as-is (`models/logistic_regression_baseline/`):

- Input `S_t ∈ R^52` = `X[:, -1, :]` — the most recent state only; no temporal sequence
- One independent `LogisticRegression(l2, C=1.0, lbfgs, max_iter=1000, class_weight='balanced', random_state=42)` per horizon
- Scaling: the frozen benchmark scaler (applied at build time; nothing refit)
- Thresholds: frozen validation-only values from Task 2, not recalculated here

The pickled models' SHA256 are checked against the LR manifest, and live `predict_proba`
output is required to match the frozen `test_predictions.csv` to 1e-12.

## 7. V4 methodology

Frozen Task 1 reference, consumed as-is:

- Checkpoint `models/latent_world_model_v4/latent_world_model_v4.pt`, SHA256 `76af8d8a…5898`, loaded `strict=True`
- Input `X_t ∈ R^(10×52)`; 2-layer LSTM → LayerNorm → Linear → 32-dim latent; gated residual transition; attack head
- Point score for `t+k` = attack-head probability after **k** latent transitions
- Thresholds: the frozen validation-selected values from the V4 manifest
- Inference only — no training, no fine-tuning, no weight change

Every recomputed V4 number was required to match `docs/v4_baseline_manifest.json`
`results_verified` exactly (tolerance 1e-9). All 7 comparisons: **MATCH**. The same exact
agreement was required and obtained for all 7 Logistic Regression comparisons.

## 8. Threshold policy

Each model keeps its own legitimate, validation-derived decision threshold. No common
threshold is forced, because forcing one would replace a fair comparison with an arbitrary
operating point.

| Model | Mechanism | K=1 | K=2 | K=3 | K=5 | Traj K=2 | Traj K=3 | Traj K=5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | none (deterministic) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Logistic Regression | validation F1 argmax, grid `np.arange(0.01, 0.99, 0.01)` | 0.60 | 0.60 | 0.60 | 0.54 | 0.60 | 0.60 | 0.60 |
| V4 World Model | validation F1 argmax, same grid | 0.45 | 0.54 | 0.65 | 0.72 | 0.51 | 0.68 | 0.69 |

Every one of these 14 thresholds was **re-derived from validation scores** during this run
and matched the frozen value exactly. Test labels were used only to compute the final
reported metrics.

## 9. Point-forecast results — held-out Mar-2-2018

2,031 positive / 1,084 negative test sequences at every point horizon.

### K = 1
| Model | Threshold | Precision | Recall | F1 | ROC-AUC | FPR | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | n/a | 0.9985 | 0.9985 | **0.9985** | 0.9979 † | 0.0028 | 2028 | 3 | 1081 | 3 |
| Logistic Regression | 0.60 | 0.9758 | 0.2181 | 0.3565 | 0.9464 | 0.0101 | 443 | 11 | 1073 | 1588 |
| V4 World Model | 0.45 | 0.9410 | 0.4318 | 0.5920 | 0.8558 | 0.0507 | 877 | 55 | 1029 | 1154 |

### K = 2
| Model | Threshold | Precision | Recall | F1 | ROC-AUC | FPR | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | n/a | 0.9975 | 0.9975 | **0.9975** | 0.9965 † | 0.0046 | 2026 | 5 | 1079 | 5 |
| Logistic Regression | 0.60 | 0.9742 | 0.2230 | 0.3630 | 0.9465 | 0.0111 | 453 | 12 | 1072 | 1578 |
| V4 World Model | 0.54 | 0.9507 | 0.5219 | 0.6739 | 0.8774 | 0.0507 | 1060 | 55 | 1029 | 971 |

### K = 3
| Model | Threshold | Precision | Recall | F1 | ROC-AUC | FPR | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | n/a | 0.9970 | 0.9970 | **0.9970** | 0.9958 † | 0.0055 | 2025 | 6 | 1078 | 6 |
| Logistic Regression | 0.60 | 0.9740 | 0.2216 | 0.3610 | 0.9476 | 0.0111 | 450 | 12 | 1072 | 1581 |
| V4 World Model | 0.65 | 0.9609 | 0.5446 | 0.6952 | 0.8893 | 0.0415 | 1106 | 45 | 1039 | 925 |

### K = 5
| Model | Threshold | Precision | Recall | F1 | ROC-AUC | FPR | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | n/a | 0.9966 | 0.9966 | **0.9966** | 0.9950 † | 0.0065 | 2024 | 7 | 1077 | 7 |
| Logistic Regression | 0.54 | 0.9739 | 0.2570 | 0.4067 | 0.9465 | 0.0129 | 522 | 14 | 1070 | 1509 |
| V4 World Model | 0.72 | 0.9660 | 0.6160 | 0.7523 | 0.8877 | 0.0406 | 1251 | 44 | 1040 | 780 |

† degenerate two-point ROC (binary score) — equals balanced accuracy, not a ranking AUC.

**Secondary, not a three-model row:** Logistic Regression K=4 — threshold 0.60, F1 0.3689,
precision 0.9747, recall 0.2275, ROC-AUC 0.9483. No V4 K=4 reference exists.

## 10. Trajectory-event results — "any attack within t+1 … t+K"

The MAX aggregation is **analogous but not architecturally identical** across models:

| Model | What "MAX" means |
|---|---|
| V4 | `max` over the rollout probabilities `probs[:, 1:K+1]` of **one** latent transition model |
| Logistic Regression | `max` over **K independently trained** horizon-specific models |
| Persistence | the rule is horizon-independent, so the trajectory score is `A(t)` itself |

| Model | K | Threshold | Precision | Recall | F1 | ROC-AUC | FPR | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 2 | n/a | 0.9985 | 0.9971 | **0.9978** | 0.9971 † | 0.0028 | 2028 | 3 | 1078 | 6 |
| Logistic Regression | 2 | 0.60 | 0.9742 | 0.2227 | 0.3625 | 0.9465 | 0.0111 | 453 | 12 | 1069 | 1581 |
| V4 World Model | 2 | 0.51 | 0.9554 | 0.5683 | 0.7127 | 0.8780 | 0.0500 | 1156 | 54 | 1027 | 878 |
| Persistence | 3 | n/a | 0.9990 | 0.9961 | **0.9975** | 0.9971 † | 0.0019 | 2029 | 2 | 1076 | 8 |
| Logistic Regression | 3 | 0.60 | 0.9744 | 0.2239 | 0.3641 | 0.9476 | 0.0111 | 456 | 12 | 1066 | 1581 |
| V4 World Model | 3 | 0.68 | 0.9661 | 0.4752 | 0.6371 | 0.8904 | 0.0315 | 968 | 34 | 1044 | 1069 |
| Persistence | 5 | n/a | 0.9995 | 0.9951 | **0.9973** | 0.9971 † | 0.0009 | 2030 | 1 | 1074 | 10 |
| Logistic Regression | 5 | 0.60 | 0.9751 | 0.2304 | 0.3727 | 0.9476 | 0.0112 | 470 | 12 | 1063 | 1570 |
| V4 World Model | 5 | 0.69 | 0.9665 | 0.6794 | 0.7979 | 0.8929 | 0.0447 | 1386 | 48 | 1027 | 654 |

V4's trajectory F1 is non-monotonic in K (0.7127 → 0.6371 → 0.7979) because each horizon
selects its own validation threshold (0.51 → 0.68 → 0.69) on a different event target.

## 11. Interpretation

### 11.1 The persistence hypothesis is confirmed on this benchmark

Attack labels on the Mar-2 test day are almost perfectly autocorrelated:

| Statistic | Value |
|---|---|
| `P(A(t+1) = A(t))` | 0.9981 — 5 of 3,115 windows change |
| `P(A(t+2) = A(t))` | 0.9968 — 9 windows change |
| `P(A(t+3) = A(t))` | 0.9961 — 11 windows change |
| `P(A(t+5) = A(t))` | 0.9955 — 13 windows change |
| Label transitions in the whole test day | 3 onsets, 3 offsets |

A rule that copies the current label is therefore near-optimal *by construction*. The
ordinary `t+k` attack-classification task, as currently defined, is dominated by label
persistence rather than by forecasting.

### 11.2 V4 versus persistence — V4 is lower at every horizon

| Horizon | Persistence F1 | V4 F1 | **V4 − Persistence** |
|---|---:|---:|---:|
| K=1 | 0.9985 | 0.5920 | **−0.4066** |
| K=2 | 0.9975 | 0.6739 | **−0.3237** |
| K=3 | 0.9970 | 0.6952 | **−0.3019** |
| K=5 | 0.9966 | 0.7523 | **−0.2443** |
| Traj K=2 | 0.9978 | 0.7127 | **−0.2851** |
| Traj K=3 | 0.9975 | 0.6371 | **−0.3605** |
| Traj K=5 | 0.9973 | 0.7979 | **−0.1994** |

The gap narrows as K grows (−0.41 → −0.24 across K=1→5), but V4 does not reach persistence
at any horizon. On the degenerate-but-reported ROC axis persistence also sits higher
(0.9950–0.9979 vs V4's 0.8558–0.8929), though that comparison is weakened by the
degeneracy noted in §5.

This does not mean persistence is the better *forecaster*. It means the metric is
saturated: persistence is handed `A(t)`, the near-perfect predictor of `A(t+k)` on this
label sequence, while V4 must infer it from features alone. The comparison measures how
much label inertia the task contains, not how well a model anticipates change.

### 11.3 V4 versus Logistic Regression — V4 higher on F1, lower on ROC-AUC

| Horizon | LR F1 | V4 F1 | **V4 − LR (F1)** | LR ROC-AUC | V4 ROC-AUC | **V4 − LR (AUC)** |
|---|---:|---:|---:|---:|---:|---:|
| K=1 | 0.3565 | 0.5920 | **+0.2354** | 0.9464 | 0.8558 | **−0.0906** |
| K=2 | 0.3630 | 0.6739 | **+0.3109** | 0.9465 | 0.8774 | **−0.0690** |
| K=3 | 0.3610 | 0.6952 | **+0.3341** | 0.9476 | 0.8893 | **−0.0582** |
| K=5 | 0.4067 | 0.7523 | **+0.3456** | 0.9465 | 0.8877 | **−0.0588** |
| Traj K=2 | 0.3625 | 0.7127 | **+0.3502** | 0.9465 | 0.8780 | **−0.0686** |
| Traj K=3 | 0.3641 | 0.6371 | **+0.2730** | 0.9476 | 0.8904 | **−0.0572** |
| Traj K=5 | 0.3727 | 0.7979 | **+0.4252** | 0.9476 | 0.8929 | **−0.0548** |

The two metrics disagree, and the disagreement is informative: V4 produces higher F1 at
each model's own frozen threshold, while Logistic Regression ranks the test windows better
(ROC-AUC higher by 0.055–0.091 everywhere). Threshold-based F1 mixes separability with
calibration; ROC-AUC isolates separability. See §12.

### 11.4 Does the K-trend demonstrate forecasting skill?

All three methods improve, or hold roughly flat, as K increases (V4 0.5920 → 0.7523,
LR 0.3565 → 0.4067, persistence 0.9985 → 0.9966 — essentially flat). A longer horizon does
not make the task harder here, because the positive class remains positive across
consecutive windows. Rising F1 with K is therefore consistent with label persistence and
with each horizon's separately chosen threshold; it is **not**, by itself, evidence that
the latent rollout extrapolates dynamics.

### 11.5 What the evidence establishes

- On the held-out Mar-2 day, V4 produced F1 = 0.7523 at K=5 versus 0.9966 for persistence and 0.4067 for Logistic Regression.
- V4 exceeded the static Logistic Regression baseline by +0.2354 to +0.3456 F1 across K=1,2,3,5, while ranking test windows less well (ROC-AUC lower by 0.055–0.091).
- V4 did not exceed persistence at any evaluated horizon; the shortfall ranges from −0.1994 to −0.4066 F1.
- The results do **not** establish that V4's temporal architecture caused its advantage over Logistic Regression: V4 and LR differ simultaneously in input (10×52 vs 1×52), model class (LSTM + latent rollout vs linear), and calibration behaviour, so the contribution of temporal modelling alone is not isolated by this experiment.
- The results do **not** establish that the temporal World Model adds value beyond persistence on this task, because the task is saturated by label inertia (§11.1) and persistence is additionally given the observed label `A(t)` that neither other model receives (§5).
- A discriminating experiment must target windows where the label *changes* — pre-attack onset / early warning — where the persistence rule is uninformative by construction. The frozen V4 reference already records one such measurement (66.7% onset coverage, 50 s mean lead time on 3 observable onsets), but with 3 onsets it is far too small a sample to support a conclusion.

## 12. Logistic Regression threshold transfer — a calibration finding

Carried forward from Task 2 and reconfirmed here:

| | Validation | Test (Mar-2) |
|---|---:|---:|
| LR K=1 F1 at the frozen threshold 0.60 | 0.7482 | 0.3565 |
| Positive rate | 26.6% | 65.2% |
| Test-optimal threshold (**not used**) | — | 0.08 (would give F1 ≈ 0.9287) |

- Thresholds were selected **only** on validation; test labels were not used to select them.
- The Mar-2 positive prevalence (65.2%) is ~2.5× the validation prevalence (26.6%).
- ROC-AUC stays high (≈0.947) while threshold-based F1 collapses — the model's ranking
  transfers across days, its operating point does not.
- The test-optimal thresholds (0.08–0.10) are reported for diagnosis only and are **never**
  used in any benchmark number; using them would be test-label leakage.

This affects how §11.3 should be read: part of the V4-over-LR F1 gap is a calibration
difference, not a separability difference.

## 13. Provenance and caveats

**V4 provenance caveat (preserved, not repaired).** The V4 checkpoint was trained on
`data/processed/temporal_benchmark_v2/multistep_K5` (single-day Feb-28; scaler fit on 2,384
windows) while the frozen reference evaluation feeds it `multiday_benchmark_K5` arrays
scaled by a **different** StandardScaler (8,575 windows). This comparison uses the frozen V4
reference exactly as currently defined. All caveats in
[V4_REFERENCE_BASELINE.md §12](V4_REFERENCE_BASELINE.md) remain in force, including that
the HEAD copy of `src/k_step_rollout.py` does not reproduce the reference and that the
canonical evaluation is pinned to commit `2be3073`.

**Leakage checks** (`results/comparison_verification.json`, all PASS):

| Check | Result |
|---|---|
| Mar-2 test day absent from train and validation | PASS |
| Validation chronologically after train within each day | PASS |
| No duplicated sequences within or across splits | PASS |
| `t → t+k` target alignment verified against the raw source CSV (k = 1…5) | PASS |
| Input scaler fitted on the train partition only (n = 8575); not refit here | PASS |
| All 14 thresholds re-derived from validation and matching the frozen values | PASS |
| Test labels not used for threshold selection | PASS |
| All models evaluated on identical test examples and targets | PASS |
| V4 metrics match the frozen reference manifest (7/7) | PASS |
| LR metrics match the frozen baseline manifest (7/7) | PASS |
| No model retrained, no frozen artifact modified | PASS |

**Other limitations.** A single held-out day with only 3 attack onsets; one dataset
(CSE-CIC-IDS2018 via CICFlowMeter); flow-level features only; no confidence intervals or
significance testing (a single evaluation run per model); persistence receives information
the other models do not; the LR-vs-V4 contrast confounds input representation with model
class.

## 14. Reproduction

```bash
cd /path/to/cyber-world-model
source venv/bin/activate

python src/verify_v4_baseline.py                     # frozen V4 still intact
python src/verify_logistic_regression_baseline.py    # frozen LR still intact
python src/run_standardized_comparison.py            # regenerates every number and figure here
```

The comparison script re-runs both verifiers first and aborts if either fails. It performs
inference only: V4 rollout from the frozen checkpoint, `predict_proba` from the frozen LR
pickles, and the deterministic persistence rule. Outputs are written only to `results/` and
`docs/standardized_comparison_manifest.json`.
