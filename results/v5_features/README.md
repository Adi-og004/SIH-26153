# results/v5_features/

Task 9 output — the V5 packet-aware feature pipeline.

- **V5 dimension**: 82 (26 flow base + 26 flow rel-change, both frozen V4 + 21 packet base + 9 packet rel-change, both new)
- **Schema SHA-256**: `5f52e2d0a7c65e651ba3e948a19c63af108e7b278dddf4e1a7f41b8b21ba6704`
- **Packet parity**: EXACT (180/180 comparisons bit-identical)
- **Leakage**: PASS
- **V4 integrity**: PASS
- **V5 training dataset**: False
- **V5 model trained**: False

**Status: V5 FEATURE PIPELINE COMPLETE — TRAINING DATASET NOT READY**

## Files

| File | What it holds |
|---|---|
| `feature_design_candidates.csv` | Every statistic considered, selected and rejected, with the reason. Fixed before any evaluation. |
| `packet_parity_results.csv` | Per (capture, feature) training-route vs PCAP-route comparison. |
| `parity/parity_*.csv` | The same comparison, one file per capture. |
| `pcap_conformance.csv` | Per (capture, feature) distribution: min/p50/max, mean, std, missing, non-finite, distinct, constant. |
| `pcap_conformance_summary.csv` | Per-capture packet/window/feature counts and hashes. |
| `dataset_availability.csv` | PCAP coverage per benchmark day — why no V5 training set exists. |
| `leakage_check.json` | The five causality checks and their results. |
| `packet_windows_*.csv` | The built packet-window tables, for inspection. |
| `v5_feature_contract_sha256.txt` | The deterministic schema hash. |
| `figures/` | Dataset availability, feature liveness, feature distributions. |

## What this is not

There is no V5 model. Training was not attempted, because the three TRAIN/VAL benchmark days have no packet captures at all and filling them would require fabricating packet values. V4 remains the reference baseline and is byte-for-byte unchanged.

See `docs/V5_PACKET_FEATURES.md` for the full write-up.
