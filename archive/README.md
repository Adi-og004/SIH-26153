# Archive — historical scripts kept for V4 provenance

These scripts are **not** part of the dashboard runtime. They are kept
because the frozen V4 contract and the reproducibility docs cite them as the
code that produced V4's data or baselines.

| Script | Role |
|---|---|
| `clean_datasets.py` | Step 1 of the V4 data chain: raw CIC flow CSVs → cleaned CSVs in `data/processed/` |
| `create_network_states_v2.py` | **The training-side definition of the 26 base V4 features** (10-second windows over CICFlowMeter flows). `docs/feature_contract_v4.json` names it as the training state builder, and `src/pcap_ingest_v4.py` ports its helpers verbatim. |
| `prepare_temporal_benchmark_v2.py`, `prepare_multistep_data.py` | Built `temporal_benchmark_v2/multistep_K5`, the single-day (Feb 28) split the V4 checkpoint was **trained** on |
| `train_baseline.py`, `train_temporal_baseline.py` | Earlier logistic-regression baselines, cited as the source of the regression settings in `docs/LOGISTIC_REGRESSION_BASELINE.md` |
| `train_persistence_baseline.py` | A state-regression persistence baseline that was **excluded** as incompatible, see `docs/STANDARDIZED_BASELINE_COMPARISON.md` |

The V1–V3 world-model experiments and dataset-exploration scripts that used
to live here are not part of the submission. They remain in the git history
(commits before this cleanup) for anyone tracing how V4 evolved.

These scripts use repository-relative paths (`data/raw`, `data/processed`)
and expect the CSE-CIC-IDS2018 CSVs described in `data/README.md`.
