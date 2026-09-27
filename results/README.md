# Results

These are curated evidence files from runs of the frozen V4 model. Unless a
directory says otherwise, every result here is **V4** on the **held-out
Friday 2 March 2018** test day (3,115 sequences) of the
`multiday_benchmark_K5` benchmark.

A few bulky per-row outputs that no doc cites and the dashboard does not
read (for example `counterfactual_defense/scenario_results.csv`) are not
committed. The `.gitignore` lists them; the builder scripts named below
regenerate them from the local benchmark data.

## Read by the dashboard at runtime (required)

| Directory / file | Produced by | Content |
|---|---|---|
| `v4_screening_mvp/` | `src/build_v4_screening_mvp.py` | The frozen replay substrate: `screening_X.npy` (byte-identical to `X_test.npy`), stored V4 predictions, metadata, event timeline, manifest, environment and dashboard validation records |
| `v4_explainability/` + `v4_explainability_summary.json` | `src/run_v4_explainability.py` | Integrated Gradients attributions for k = 1, 3, 5 (`attributions_k*.npy`, hash-pinned in the summary), global, temporal and horizon importance, representative, FP/FN, onset and PCAP explanations, figures |
| `v4_reliability/` | `src/build_v4_reliability.py` | Rule-based reliability status (RELIABLE / CAUTION / UNRELIABLE) per sequence, with its reference thresholds and validation |
| `abstain/` | `src/build_abstain.py` | ALERT / BENIGN / ABSTAIN analyst-decision policy, per-sequence decisions, selective-risk curves |
| `attack_path/` | `src/build_attack_path.py` | ATT&CK-consistent progression hypotheses from the frozen rollout plus the rule-based mapper |
| `defense_simulation/` | `src/build_defense_simulation.py` | Defend-now what-if projections (model-based, not causal) and the strength sweep |
| `counterfactual_defense/` | `src/build_counterfactual_defense.py` | Multi-scenario what-if comparison summary |

## Evaluation evidence

| Directory / file | Doc | Question it answers |
|---|---|---|
| [`baselines/`](baselines/) (index), `standardized_baseline_comparison.csv`, `figures/` | `docs/STANDARDIZED_BASELINE_COMPARISON.md` | V4 vs persistence vs logistic regression, same data, same targets |
| `persistence_hypothesis/`, `persistence_hypothesis_summary.json` | `docs/PERSISTENCE_HYPOTHESIS.md` | Can this benchmark show forecasting value beyond "the current label continues"? |
| `long_horizon_rollout/`, `long_horizon_rollout_summary.json` | `docs/LONG_HORIZON_ROLLOUT.md` | Stability and informativeness of the rollout to K = 10 |
| `rotated_generalization/`, `rotated_generalization_summary.json` | `docs/ROTATED_GENERALIZATION.md` | How performance changes when each day is held out in turn |
| `error_analysis/`, `error_analysis_summary.json` | `docs/ERROR_ANALYSIS.md` | Where and why V4 and the baselines make false positives and false negatives |
| `feature_parity/`, `feature_parity_summary.json` | `docs/FEATURE_PARITY.md` | Whether the PCAP pipeline computes the same 52 features as the training pipeline |

## V5 development (not used by the dashboard)

| Directory / file | Doc |
|---|---|
| `v5_features/`, `v5_packet_feature_summary.json` | `docs/V5_PACKET_FEATURES.md`: packet-level feature pipeline, parity and leakage checks |

V5 demo and diagnostic outputs (`v5_demo*`, `v5_mixed_scope_diagnostic`,
`v5_packet_ablation`) are large intermediate arrays and are not committed.
