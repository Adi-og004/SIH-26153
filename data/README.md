# Data Documentation

> [!IMPORTANT]
> **Raw and processed data files are NOT tracked in Git.**

## Required Dataset
This project relies on network flow data processed via CICFlowMeter. 

## Directory Structure
You must manually place the `_states_v2.csv` feature files in the following directory before running preprocessing:

```text
data/
└── processed/
    └── states_v2/
        ├── Wednesday-14-02-2018_TrafficForML_CICFlowMeter_states_v2.csv
        ├── Thursday-15-02-2018_TrafficForML_CICFlowMeter_states_v2.csv
        ├── Wednesday-28-02-2018_TrafficForML_CICFlowMeter_states_v2.csv
        └── Friday-02-03-2018_TrafficForML_CICFlowMeter_states_v2.csv
```

## Feature Dimensions
The raw CSVs must contain the 26 base state features (e.g. `flow_count`, `log_total_packets`, `window_packet_rate`). 

Running `src/prepare_multiday_benchmark.py` will automatically append the 26 relative-change features, resulting in the 52-dimensional state arrays. The final scaled arrays will be dumped into `data/processed/multiday_benchmark_K5/` as `.npy` binaries.
