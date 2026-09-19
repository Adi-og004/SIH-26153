# results/v4_explainability/

Attribution artifacts for the FROZEN V4 checkpoint (Task 8).
Regenerate with `python src/run_v4_explainability.py`; verify with
`python src/verify_v4_explainability.py`.

Every file here is a *consumer* of the frozen V4 baseline. Nothing in
this directory was used to train, tune, threshold or select anything
about the model.

## Provenance of every file in this directory

| item | value |
|---|---|
| model | LatentWorldModelV4 (V4) |
| checkpoint | `models/latent_world_model_v4/latent_world_model_v4.pt` |
| checkpoint sha256 | `76af8d8a9e02215923b1c0ab6b627bfc0809e856748666a55ee4b16cec5e5898` |
| input scaler | `data/processed/multiday_benchmark_K5/input_scaler.pkl` |
| scaler sha256 | `e0ab9bebd26869950dda51a67bdde3192b0edbd00e20b623115ed2b653f04f19` |
| feature schema | `docs/v4_feature_schema.json` |
| feature-order sha256 | `137b58333f1c751522f30478219ff86b8c115aa749defb57ae80ee83559ecbb5` |
| input shape | 10 x 52 |
| attribution method | Integrated Gradients (gausslegendre, n_steps=1024) |
| baseline | zero vector in scaled feature space |
| generated | 2026-09-19T05:56:47.433859Z |

## Files

| file | contents |
|---|---|
| `prediction_identity.csv` | per-sample \|P_inference - P_explained\| for every test sequence at every explained horizon (Part 20) |
| `completeness_checks.csv` | per-sample IG completeness residual plus the n_steps convergence sweep and the baseline-sensitivity probe (Part 7) |
| `ablation_crosscheck.csv` | gradient-free feature-ablation cross-check of the IG ranking (Part 8) |
| `feature_importance.csv` | 52-dim signed and absolute importance, aggregated over the 10 history steps, K=5 (Part 9A) |
| `temporal_importance.csv` | 10-dim signed and absolute importance, aggregated over the 52 features, K=5 (Part 9B) |
| `attribution_heatmap.csv` | 10 x 52 signed mean attribution, rows `t-9`..`t`, columns the exact 52 schema names (Part 10) |
| `horizon_feature_importance.csv` | per-feature importance at K in [1, 3, 5]; explanations from different targets are never averaged together (Part 12) |
| `representative_explanations.csv` | TP / TN / FP / FN / near-threshold / high- and low-confidence cases, selected by the pre-declared deterministic rules (Part 11) |
| `attack_vs_benign.csv` | attribution pattern comparison between correctly predicted attack and benign windows (Part 13) |
| `error_explanations.csv` | explanations for the Task 4 representative false positives and false negatives (Part 14) |
| `onset_explanations.csv` | attribution for pre-onset elevated-score windows vs missed onsets, from the Task 5/6 cases (Part 15) |
| `trajectory_max_explanations.csv` | TARGET B, MAX(P(t+1)..P(t+5)), with the per-sample argmax and the `argmax_switches_on_path` diagnostic (Part 1) |
| `pcap_explanations.csv` | PCAP-sourced explanations, each carrying `input_source=pcap` and the approximate-flow-record caveat (Part 16) |
| `repeatability.csv` | two independent runs on identical input (Part 21) |
| `feature_name_integrity.json` | index-by-index proof that attribution column i carries schema name i (Part 22) |
| `figures/` | attribution heatmap, feature/temporal importance, horizon comparison, representative-case heatmaps |

## Reading the numbers

Sign is never discarded. A positive attribution means the feature, at
that history step, pushed the selected output UP relative to the
baseline; negative means DOWN. Absolute values rank influence
regardless of direction, and the two are reported side by side.

## What these files do not establish

Attribution describes this model's sensitivity/contribution on these
inputs. It does not establish causality, attacker intent, attack
stage, or actual compromise. MITRE ATT&CK wording is produced by a
separate rule-based layer (`src/mitre_mapper.py`) and is never derived
from attribution values.

PCAP rows carry an additional caveat: they use the same V4 feature
contract and transform chain, but flow-record production differs from
the training path, so the feature values are approximate and are not
claimed to be numerically equivalent to training features.

See `docs/V4_EXPLAINABILITY.md` for the full report and
`docs/v4_explainability_contract.json` for the machine-readable
contract.
