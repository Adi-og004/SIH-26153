# Documentation index

The files keep their original names and locations. Several JSON files here
are **frozen contracts** that runtime code loads, or that other manifests
cite, by path. That is why they are not split into `docs/v4/` and
`docs/v5/` folders; the grouping below does that job instead.

## Start here

| Doc | For |
|---|---|
| [`../README.md`](../README.md) | Project overview, quick start, results, limitations |
| [`DEMO_RUNBOOK.md`](DEMO_RUNBOOK.md) | **Canonical** judge demonstration sequence |
| [`reproducibility.md`](reproducibility.md) | Environment, what reproduces from a clone, and what needs the dataset |

## V4 — current, frozen

| Doc | Content |
|---|---|
| [`V4_REFERENCE_BASELINE.md`](V4_REFERENCE_BASELINE.md) | The frozen V4 reference: artefacts, split, thresholds, verified held-out results, known issues |
| `v4_baseline_manifest.json` | **Frozen provenance manifest.** SHA-256 of checkpoint, scalers and feature order, plus architecture, dataset split, thresholds and verified results. The runtime checks against it. |
| `v4_feature_schema.json` | Frozen 52-feature schema (index, name, base / relative). The dashboard checks it against `input_feature_names.txt` at start-up. |
| `feature_contract_v4.json` | Training vs PCAP implementation contract for every feature |
| [`FEATURE_PARITY.md`](FEATURE_PARITY.md) + `v4_feature_parity_manifest.json` | Does the PCAP pipeline compute the training features? Where it does not, and why. |
| [`V4_EXPLAINABILITY.md`](V4_EXPLAINABILITY.md) + `v4_explainability_contract.json` | Integrated Gradients method, its verification, and the prediction-identity proof |
| [`V4_RELIABILITY.md`](V4_RELIABILITY.md) | Rule-based reliability / model-health layer |

## Evaluation studies (V4 and baselines)

| Doc | Question |
|---|---|
| [`STANDARDIZED_BASELINE_COMPARISON.md`](STANDARDIZED_BASELINE_COMPARISON.md) + `standardized_comparison_manifest.json` | V4 vs persistence vs logistic regression on identical data |
| [`LOGISTIC_REGRESSION_BASELINE.md`](LOGISTIC_REGRESSION_BASELINE.md) + `logistic_regression_baseline_manifest.json` | The logistic-regression baseline specification |
| [`PERSISTENCE_HYPOTHESIS.md`](PERSISTENCE_HYPOTHESIS.md) | Is there forecasting value beyond "the current label continues"? |
| [`LONG_HORIZON_ROLLOUT.md`](LONG_HORIZON_ROLLOUT.md) | Rollout stability and informativeness out to K = 10 |
| [`ROTATED_GENERALIZATION.md`](ROTATED_GENERALIZATION.md) | Each day held out in turn |
| [`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md) | False-positive / false-negative taxonomy |

## V5 — development (not used by the dashboard)

| Doc | Content |
|---|---|
| [`V5_PACKET_FEATURES.md`](V5_PACKET_FEATURES.md) | Packet-aware 82-D representation. Written before the experimental V5 checkpoint existed; see `models/latent_world_model_v5/README.md` for the checkpoint's status. |
| `v5_feature_schema.json`, `v5_packet_feature_contract.json`, `v5_packet_dataset_manifest.json` | V5 schema, contract and dataset-availability manifest |

## Historical demo notes

`results/v4_screening_mvp/{DEMO_RUNBOOK,FINAL_DEMO_FLOW,JUDGE_LANGUAGE}.md`
are the scripts written when the screening dashboard was validated. They
agree with this runbook, but were written for the original local
environment (`src/venv`). Use [`DEMO_RUNBOOK.md`](DEMO_RUNBOOK.md).
