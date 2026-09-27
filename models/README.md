# Models

| Directory | Status | Used by the dashboard? |
|---|---|---|
| [`latent_world_model_v4/`](latent_world_model_v4/) | **CURRENT — frozen V4 reference model** | **Yes.** It is the only model the dashboard loads. |
| [`latent_world_model_v5/`](latent_world_model_v5/) | Development only (next-generation, packet-aware) | No |
| [`pcap_analysis_v4/`](pcap_analysis_v4/) | Evidence: PCAP → V4 feature-parity studies | No (research artefacts) |

## Why these directories are not called `models/v4/` and `models/v5/`

The V4 checkpoint path `models/latent_world_model_v4/latent_world_model_v4.pt`
is recorded, with its SHA-256, in the frozen provenance manifest
`docs/v4_baseline_manifest.json`. `src/v4_explainer.py` checks the checkpoint
against that manifest before the dashboard loads the model. Renaming the
directory would mean editing the frozen manifest, so the original paths are
kept. The directory names already carry the version.

## Frozen V4 runtime artefacts (all hash-pinned in `docs/v4_baseline_manifest.json`)

| Artefact | Path | SHA-256 (first 12) |
|---|---|---|
| Checkpoint | `models/latent_world_model_v4/latent_world_model_v4.pt` | `76af8d8a9e02` |
| Input scaler (52-D `StandardScaler`) | `data/processed/multiday_benchmark_K5/input_scaler.pkl` | `e0ab9bebd268` |
| Target scaler (26-D `StandardScaler`) | `data/processed/multiday_benchmark_K5/target_scaler.pkl` | `7155dec48f89` |
| Input feature order (52 names) | `data/processed/multiday_benchmark_K5/input_feature_names.txt` | `137b58333f1c` |
| State feature order (26 names) | `data/processed/multiday_benchmark_K5/state_feature_names.txt` | `64a829779877` |

`tests/test_submission_regression.py::test_frozen_v4_artifacts_match_manifest`
recomputes these hashes on every test run.

Baseline models (logistic regression, persistence) are not stored here.
Their evaluated results are indexed in [`results/baselines/`](../results/baselines/).
