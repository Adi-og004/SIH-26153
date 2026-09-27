# Data

## Source dataset

**CSE-CIC-IDS2018** (Communications Security Establishment and Canadian
Institute for Cybersecurity, University of New Brunswick).

- Dataset page: <https://www.unb.ca/cic/datasets/ids-2018.html>
- The original captures are also hosted in the public AWS bucket
  `s3://cse-cic-ids2018/` (per-host PCAPs of Friday 2 March 2018 are under
  `Original Network Traffic and Log data/Friday-02-03-2018/pcap.zip`).

The dataset is **not redistributed here**. Download it from the source and
follow its terms of use.

## Portion used

The CIC-provided per-day flow CSVs (`*_TrafficForML_CICFlowMeter.csv`,
produced by CIC with CICFlowMeter) for four days:

| Day | Role in the V4 benchmark |
|---|---|
| Wednesday 14 Feb 2018 | train (first 85 %, chronological) / validation (last 15 %) |
| Thursday 15 Feb 2018 | train / validation, same split |
| Wednesday 28 Feb 2018 | train / validation, same split |
| **Friday 2 Mar 2018** | **held-out test, never used for any fitting or threshold** |

The split and the SHA-256 of every intermediate file are recorded in
`docs/v4_baseline_manifest.json → dataset`.

## Preprocessing chain (V4)

```
data/raw/<day>_TrafficForML_CICFlowMeter.csv            (download from CIC)
  │  archive/clean_datasets.py         inf → NaN cleanup, label normalisation
  ▼
data/processed/<day>_TrafficForML_CICFlowMeter.csv
  │  archive/create_network_states_v2.py
  │      flows binned into 10-second windows by Timestamp;
  │      26 base network-state features per window + attack labels
  ▼
data/processed/states_v2/<day>_..._states_v2.csv
  │  src/prepare_multiday_benchmark.py
  │      + 26 relative-change features, computed per source day:
  │          (x_t − x_{t−1}) / (|x_{t−1}| + 1e-8), first row 0, ±inf → 0, clipped to [−10, 10]
  │      StandardScaler fitted on the train partition only (8,575 windows)
  │      10-window sequences, kept only if every step is exactly 10 s apart
  │      targets: next 5 states (26-D) and per-horizon attack labels
  ▼
data/processed/multiday_benchmark_K5/   X_{train,val,test}.npy (N, 10, 52), Y_*, A_*, *_metadata.csv, scalers
```

Note: the V4 checkpoint itself was **trained** on an earlier single-day
(Feb 28) split, `temporal_benchmark_v2`, built by
`archive/prepare_temporal_benchmark_v2.py` and
`archive/prepare_multistep_data.py`. It is **evaluated** on
`multiday_benchmark_K5`. The manifest records this under `known_issues`.

## What is committed here

Only files the dashboard or the tests read:

| Path | Why it is committed |
|---|---|
| `processed/multiday_benchmark_K5/input_feature_names.txt`, `state_feature_names.txt` | Frozen feature order (52 inputs, 26 states), hash-pinned |
| `processed/multiday_benchmark_K5/input_scaler.pkl`, `target_scaler.pkl` | Frozen scalers, hash-pinned |
| `processed/multiday_benchmark_K5/X_test.npy`, `test_metadata.csv` | The held-out test tensor (3,115 × 10 × 52) and its timestamps. Loaded by the dashboard at start-up. |
| `processed/multiday_benchmark_K5/Y_multistep_test.npy`, `A_binary_multistep_test.npy`, `A_soft_multistep_test.npy` | Test targets and labels. The tests use them to check that the upload page **rejects** them as non-input. |
| `processed/v5_multistep_K5/v5_*_scaler.pkl` | V5 development scalers, used only by `src/infer_latent_world_model_v5.py` |

## Intentionally excluded

| Excluded | Reason |
|---|---|
| Raw CIC CSVs and their cleaned copies (≈ 1.2–1.5 GB for the four days) | Size; redistribution belongs to the dataset owner |
| `states_v2/` and other intermediate state files | Regenerable from the CSVs with the scripts above |
| Train and validation tensors (`X_train.npy`, `X_val.npy`, …) | Not needed to run or test the submission; regenerable |
| Raw PCAPs (`raw/march02/`, ≈ 1.6 GB here; the full day's archive is ≈ 42 GB) | Size and licensing |
| `sample_pcap/smallFlows.pcap` | Third-party sample capture whose redistribution terms have not been verified |
