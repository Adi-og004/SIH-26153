# Baselines

This is an index. The baseline result files stay at their original paths,
because `docs/standardized_comparison_manifest.json` and the scripts that
produced them record those paths, and moving them would break that record.

## Baseline types

| Baseline | Definition | Why it is here |
|---|---|---|
| **Persistence** | ŷ(t+k) = A(t): predict that the attack label *now* continues for k steps. No parameters. | The test of whether any model forecasts beyond "what is happening now continues". |
| **Logistic regression (latest-state)** | One `LogisticRegression` per horizon on S_t ∈ ℝ⁵², the most recent window only (same 52 frozen features and scaler as V4), `class_weight="balanced"`, `lbfgs`. | Same features, no temporal model. It isolates what V4's sequence modelling adds. |
| **V4 world model** | The frozen checkpoint, attack head after k latent transitions. | The system under test. |

## Evaluation setup (shared by all three)

- Data: the frozen `data/processed/multiday_benchmark_K5` benchmark. Training and validation come from Feb 14, Feb 15 and Feb 28 2018. The test set is **Friday 2 March 2018**, held out entirely: 3,115 sequences, identical targets for every model.
- Thresholds: chosen by F1 on **validation only**, then frozen. No test labels are used for selection.
- Horizons: point forecasts k = 1, 2, 3, 5. Trajectory events ("any attack in t+1…t+k", scored by the max over steps) k = 2, 3, 5.

## Headline numbers (Mar-2 held-out, point forecast)

From `results/standardized_baseline_comparison.csv`:

| Horizon | Persistence F1 | Logistic reg. F1 | V4 F1 | Persistence ROC-AUC | Logistic reg. ROC-AUC | V4 ROC-AUC |
|---|---|---|---|---|---|---|
| k=1 | 0.999 | 0.357 | 0.592 | 0.998 | 0.946 | 0.856 |
| k=3 | 0.997 | 0.361 | 0.695 | 0.996 | 0.948 | 0.889 |
| k=5 | 0.997 | 0.407 | 0.752 | 0.995 | 0.946 | 0.888 |

**How to read this:** attack labels on this test day are highly persistent.
Only 0.77 % of consecutive windows change label, according to
`docs/LONG_HORIZON_ROLLOUT.md`. So the persistence rule is close to perfect,
and this benchmark cannot show forecasting value beyond persistence
(`docs/PERSISTENCE_HYPOTHESIS.md`). V4 has higher F1 than logistic regression
at its validation-selected thresholds. Logistic regression has the higher
ROC-AUC. The comparison doc deliberately does not declare a winner.

## Files

| File | Content |
|---|---|
| [`../standardized_baseline_comparison.csv`](../standardized_baseline_comparison.csv) | All three approaches × all horizons × point/trajectory: precision, recall, F1, ROC-AUC, FPR, confusion counts, thresholds |
| [`../comparison_verification.json`](../comparison_verification.json) | Automated verification record (PASS) for the comparison |
| [`../persistence_hypothesis_analysis.csv`](../persistence_hypothesis_analysis.csv) | Persistence statistics used in the comparison |
| [`../persistence_hypothesis/`](../persistence_hypothesis/) | Transition, onset and offset analyses behind the persistence finding |
| [`../rotated_generalization/`](../rotated_generalization/) | V4, logistic regression and persistence re-evaluated with each day held out in turn |
| [`../figures/`](../figures/) | F1, precision, recall and ROC-AUC against horizon for the three approaches |
| [`../baseline_results.md`](../baseline_results.md) | Earlier human-readable persistence and logistic-regression tables (Mar-2) |
| [`../../docs/STANDARDIZED_BASELINE_COMPARISON.md`](../../docs/STANDARDIZED_BASELINE_COMPARISON.md) | Full write-up |
| [`../../docs/LOGISTIC_REGRESSION_BASELINE.md`](../../docs/LOGISTIC_REGRESSION_BASELINE.md) | Logistic-regression baseline specification |

The fitted logistic-regression `.pkl` models are not committed. Rebuilding
them needs the training split, which is not in this repository. See
`docs/reproducibility.md`.
