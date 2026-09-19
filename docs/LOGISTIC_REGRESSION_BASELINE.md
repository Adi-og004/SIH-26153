# Logistic Regression Baseline — Specification and Results

**Status:** complete and verified — 2026-09-19
**Type:** conventional **static** classifier. **This is not a World Model.**
**Git commit:** `53c22227d368b82ff9e736206a92e85b6bed2514` (branch `cicflowmeter-validation`)
**Artifacts:** `models/logistic_regression_baseline/`
**Manifest:** [docs/logistic_regression_baseline_manifest.json](logistic_regression_baseline_manifest.json)
**Reproduce:** `python src/train_logistic_regression_baseline.py`
**Verify:** `python src/verify_logistic_regression_baseline.py`

No V4 artifact was modified. The frozen V4 reference
([docs/V4_REFERENCE_BASELINE.md](V4_REFERENCE_BASELINE.md)) still verifies PASS.

---

## 1. Purpose

This baseline answers one question:

> How much of the observed future-attack prediction performance is achievable
> **without** a learned temporal world model?

It is deliberately simple and interpretable: a linear model on the single most recent
network state. It exists so that the later three-way comparison —
**Persistence vs Logistic Regression vs V4 World Model** — has a conventional
static reference point evaluated under identical data, splits and threshold discipline.

It is **not** tuned to beat or lose to V4. Methodology was fixed before any test score
was computed and was not revised afterwards.

## 2. Input representation

| Property | Value |
|---|---|
| Input | `S_t ∈ R^52` — the most recent observed network state |
| Source | `X[:, -1, :]` of the frozen benchmark tensor `X ∈ R^(N×10×52)` |
| Composition | 26 base features + 26 relative-change features |
| Feature order | identical to the frozen V4 schema (§5) |
| History used | **none** — the other 9 timesteps are discarded |
| Models | one independent classifier per horizon `k` |

Verified directly from the raw state CSVs: `X[:, -1, :]` is bit-identical
(max abs difference `0.0`) to the frozen input scaler applied to the 52-dim state at
window time `t`, i.e. exactly one 10-second window before the `t+1` target.

## 3. Why current-state features, not the temporal sequence

The scientific contrast under test is **temporal modelling**, not feature engineering.
If the baseline received the same 10×52 window as V4, a difference in results could be
attributed to either the temporal information or the model class. By giving Logistic
Regression the *same features* at a *single* point in time, the only thing V4 adds is
the learned temporal dynamics (10-step encoder + latent rollout).

This formulation is not new here — it is the repository's established
"Logistic Regression (latest-state)" baseline in `src/evaluate_baselines.py`
(`X_train_lr = X_train[:, -1, :]`), and this implementation reproduces its published
numbers exactly (§10).

## 4. Dataset split — identical to the frozen V4 reference

Benchmark directory: `data/processed/multiday_benchmark_K5` (unchanged, read-only).

| Split | Source days | Sequences | Attack (t+1) | Benign (t+1) |
|---|---|---:|---:|---:|
| Train | Feb-14, Feb-15, Feb-28 — first 85% of each day, chronologically | 8,441 | 1,835 | 6,606 |
| Validation | Feb-14, Feb-15, Feb-28 — last 15% of each day | 1,473 | 392 | 1,081 |
| Test | **Mar-2-2018 only**, fully held out | 3,115 | 2,031 | 1,084 |

Verified at run time, not assumed:

- train and validation contain exactly the three February days; test contains only `Friday-02-03-2018`
- the test day never appears in train or validation
- within every training day, the first validation timestamp is strictly after the last training timestamp
- `A_binary_multistep[:, 0]` equals the metadata `attack_present` (the t+1 label) in all three splits

The test set was used exactly once per model, to compute the final metrics.

## 5. Feature schema

Authoritative order: `data/processed/multiday_benchmark_K5/input_feature_names.txt`
(SHA256 `137b58333f1c751522f30478219ff86b8c115aa749defb57ae80ee83559ecbb5`),
cross-checked against `docs/v4_feature_schema.json` — both agree.

- indices **0–25**: base features (`flow_count` … `protocol_entropy`)
- indices **26–51**: the same 26 features suffixed `_rel_change`

Nothing was added, removed, reordered or re-transformed. No packet-level features,
no IP addresses, no labels-as-features, no future information. The full order is
recorded in the manifest (`feature_order`) and in `baseline_config.json`.

## 6. Preprocessing

| Property | Value |
|---|---|
| Scaler | `data/processed/multiday_benchmark_K5/input_scaler.pkl` |
| SHA256 | `e0ab9bebd26869950dda51a67bdde3192b0edbd00e20b623115ed2b653f04f19` |
| Class | `sklearn.preprocessing.StandardScaler`, `n_features_in_ = 52` |
| Fitted on | train partition only (`n_samples_seen_ = 8575`) |
| Applied | at benchmark build time by `src/prepare_multiday_benchmark.py` |

The baseline **reuses** this frozen scaler — the benchmark arrays it loads are already
transformed by it. No scaler is fitted, refitted or created here, and test data
participates in no fitting step of any kind. The verification script asserts that the
scaler referenced by this baseline is byte-identical to the one in the V4 manifest.

## 7. Logistic Regression configuration

| Parameter | Value | Source of the choice |
|---|---|---|
| Estimator | `sklearn.linear_model.LogisticRegression` | repository convention |
| `penalty` | `l2` | sklearn default |
| `C` | `1.0` | sklearn default — **no search performed** |
| `solver` | `lbfgs` | `src/evaluate_baselines.py`, `archive/train_baseline.py` |
| `max_iter` | `1000` | `src/evaluate_baselines.py` |
| `tol` | `1e-4` | sklearn default |
| `class_weight` | `balanced` | established repository convention (`src/evaluate_baselines.py`, `archive/train_baseline.py`, `archive/train_temporal_baseline.py`, and the V4 training script's own BCE `pos_weight`) |
| `fit_intercept` | `True` | sklearn default |
| `random_state` | `42` | repository convention (V4 uses seed 42) |

Convergence: all models converged well inside the iteration budget
(63–74 lbfgs iterations of 1000). No convergence warnings were raised.
`class_weight="balanced"` was adopted because the repository already establishes it —
not because it scored better; no alternative was evaluated on test.

## 8. Target alignment

Verified directly against the raw state CSVs for every horizon:

| Model | Target array | Semantics | Verified |
|---|---|---|---|
| K=1 | `A_binary_multistep[:, 0]` | `A(t+1)` | ✓ |
| K=2 | `A_binary_multistep[:, 1]` | `A(t+2)` | ✓ |
| K=3 | `A_binary_multistep[:, 2]` | `A(t+3)` | ✓ |
| K=4 | `A_binary_multistep[:, 3]` | `A(t+4)` | ✓ |
| K=5 | `A_binary_multistep[:, 4]` | `A(t+5)` | ✓ |

Each column was matched against `attack_present` at `timestamp + (k−1)·10s` in
`data/processed/states_v2/Friday-02-03-2018_…_states_v2.csv`: exact equality for all
five horizons across all 3,115 test sequences. No model is trained on `A(t)`, and the
horizons are never collapsed into a single target.

## 9. Threshold selection

Same convention as the frozen V4 reference evaluation, verified from its artifacts:

| Property | Value |
|---|---|
| Grid | `np.arange(0.01, 0.99, 0.01)` |
| Objective | maximise **validation** F1; first strict maximum wins ties |
| Selection data | validation only |
| Application | frozen, then applied once to the Mar-2 test scores |
| Test labels in selection | **never** |

V4's thresholds were **not** copied — Logistic Regression gets its own
validation-selected thresholds. `validation_threshold_search.csv` records the complete
grid (threshold, precision, recall, F1) for every task, horizon and strategy.

The verification script re-derives every frozen threshold from the saved models on
validation data and confirms it is the exact argmax. It additionally computes what the
threshold *would* have been on test — 0.08–0.10 instead of 0.54–0.60 — demonstrating
that the test labels were not used.

## 10. Results — Mar-2-2018 held-out test day

### Future attack prediction (`S_t → A(t+k)`)

Test: 2,031 positive / 1,084 negative sequences. Training: 8,441 sequences.

| Horizon | Threshold | F1 | Precision | Recall | ROC-AUC | FPR | TP | FP | TN | FN | Val F1 | Val pos/neg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| t+1 | 0.60 | 0.3565 | 0.9758 | 0.2181 | 0.9464 | 0.0101 | 443 | 11 | 1073 | 1588 | 0.7482 | 392 / 1081 |
| t+2 | 0.60 | 0.3630 | 0.9742 | 0.2230 | 0.9465 | 0.0111 | 453 | 12 | 1072 | 1578 | 0.7449 | 389 / 1084 |
| t+3 | 0.60 | 0.3610 | 0.9740 | 0.2216 | 0.9476 | 0.0111 | 450 | 12 | 1072 | 1581 | 0.7405 | 386 / 1087 |
| t+4 | 0.60 | 0.3689 | 0.9747 | 0.2275 | 0.9483 | 0.0111 | 462 | 12 | 1072 | 1569 | 0.7358 | 383 / 1090 |
| t+5 | 0.54 | 0.4067 | 0.9739 | 0.2570 | 0.9465 | 0.0129 | 522 | 14 | 1070 | 1509 | 0.7379 | 380 / 1093 |

### Trajectory-event prediction (`max(A(t+1) … A(t+k))`)

**Primary — `max_aggregate`:** the maximum of the per-horizon LR probabilities for
t+1…t+k. This mirrors the frozen V4 reference, which takes `max(probs[:, 1:k+1])` over
its rollout probabilities. It is the applicable analogue because LR produces one
probability per horizon, just as the rollout produces one per step.

**Secondary — `direct_target`:** a single LR trained directly on the max target. This is
the formulation already present in `src/evaluate_baselines.py`, retained for continuity.

| Strategy | K | Threshold | F1 | Precision | Recall | ROC-AUC | FPR | TP | FP | TN | FN | Pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MAX (primary)** | 2 | 0.60 | 0.3625 | 0.9742 | 0.2227 | 0.9465 | 0.0111 | 453 | 12 | 1069 | 1581 | 2034 |
| **MAX (primary)** | 3 | 0.60 | 0.3641 | 0.9744 | 0.2239 | 0.9476 | 0.0111 | 456 | 12 | 1066 | 1581 | 2037 |
| **MAX (primary)** | 5 | 0.60 | 0.3727 | 0.9751 | 0.2304 | 0.9476 | 0.0112 | 470 | 12 | 1063 | 1570 | 2040 |
| direct (secondary) | 2 | 0.59 | 0.3614 | 0.9762 | 0.2217 | 0.9457 | 0.0102 | 451 | 11 | 1070 | 1583 | 2034 |
| direct (secondary) | 3 | 0.61 | 0.3485 | 0.9774 | 0.2121 | 0.9456 | 0.0093 | 432 | 10 | 1068 | 1605 | 2037 |
| direct (secondary) | 5 | 0.61 | 0.3561 | 0.9780 | 0.2176 | 0.9453 | 0.0093 | 444 | 10 | 1065 | 1596 | 2040 |

Accuracy is deliberately not reported as a primary metric (the test day is 65% positive).

### Continuity with the previously published baseline

`results/baseline_results.md` records the earlier run of `src/evaluate_baselines.py`.
This implementation reproduces it exactly where the two overlap:

| Horizon / strategy | Published | This implementation |
|---|---:|---:|
| t+1 | 0.3565 | 0.3565 |
| t+2 | 0.3630 | 0.3630 |
| t+3 | 0.3610 | 0.3610 |
| t+5 | 0.4067 | 0.4067 |
| trajectory direct 1→2 | 0.3614 | 0.3614 |
| trajectory direct 1→3 | 0.3485 | 0.3485 |
| trajectory direct 1→5 | 0.3561 | 0.3561 |

New here: the **K=4** horizon and the **MAX-aggregate** trajectory strategy, plus saved
models, per-horizon confusion matrices, threshold grids and a verifiable manifest.

## 11. Reproduction

```bash
cd /path/to/cyber-world-model
source venv/bin/activate

python src/train_logistic_regression_baseline.py       # full run: train → threshold → test → artifacts
python src/verify_logistic_regression_baseline.py      # independent re-derivation, PASS/FAIL
```

Optional flags: `--benchmark-dir`, `--output-dir`, `--manifest`, `--no-manifest`,
`--horizons`. The defaults reproduce everything in §10 exactly; the pipeline is
deterministic (fixed `random_state`, no shuffling, no sampling).

## 12. Limitations

1. **Threshold transfer across days is poor, and this is the dominant effect.**
   Validation F1 is ≈ 0.74 but test F1 is ≈ 0.36 at the same threshold. The frozen
   thresholds (0.54–0.60) are far from the test-optimal ones (0.08–0.10), at which test F1
   would reach ≈ 0.93. The model's *ranking* on Mar-2 is strong (ROC-AUC ≈ 0.95); its
   *calibration* does not transfer from February to March. **Reported, not fixed** — using
   the test-optimal threshold would be test-label leakage.
2. **Consequence for interpretation:** threshold-dependent metrics (F1, precision, recall)
   understate this model's separability on the held-out day, while ROC-AUC does not depend
   on the threshold. Any comparison should read both.
3. **No temporal information** — by design. The baseline cannot represent trends,
   build-ups or onset dynamics spanning more than one 10-second window.
4. **No hyperparameter search.** `C = 1.0` and the sklearn defaults are used as-is. A
   validation-only search could change the numbers; it was deliberately not performed, so
   the baseline stays conventional and cannot be accused of test-driven tuning.
5. **`class_weight="balanced"` is inherited**, not justified from first principles. The
   train split is ~22% positive while the test day is ~65% positive — the prior shifts
   sharply between partitions, which contributes to the calibration failure in (1).
6. **Trajectory MAX aggregation is analogous, not identical, to V4's.** V4 takes the max
   over rollout steps of *one* model; this baseline takes the max over *five independently
   trained* models. It is the closest applicable construction, and the direct-target
   variant is retained alongside it.
7. **Not a comparison document.** The formal Persistence vs Logistic Regression vs V4
   comparison is the next task. The numbers here are stated on their own terms; readers
   should note that V4 and this baseline were evaluated on the same split with the same
   threshold discipline, but that the V4 provenance caveats in
   [V4_REFERENCE_BASELINE.md §12](V4_REFERENCE_BASELINE.md) still apply to the V4 side.
8. **Artifacts are untracked.** `models/*` is `.gitignore`'d, so the SHA256 hashes in the
   manifest are the provenance record, as for V4.
