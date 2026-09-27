# Prepared V4 input examples

Two small files you can upload on the dashboard's **📥 Data Input → 📂 Upload
Prepared V4 Data** source. Both are real, unmodified V4 model input taken
from the frozen held-out benchmark (CSE-CIC-IDS2018, Friday 2 March 2018).
Nothing was synthesised, rescaled or reordered.

| File | Shape | What it is |
|---|---|---|
| `sample_input.npy` | `(5, 10, 52)` float32 | Screening sequences **0, 100, 500, 1000, 2000** of `results/v4_screening_mvp/screening_X.npy` (byte-identical to `data/processed/multiday_benchmark_K5/X_test.npy`) |
| `sample_input.csv` | 10 rows × 52 columns | Sequence **0** written as one row per 10-second window, columns in canonical order |

## Expected result

The frozen model's output for these sequences is stored in
`results/v4_screening_mvp/screening_predictions.csv`. Uploading
`sample_input.npy` reproduces it (max absolute difference 5.4 × 10⁻⁷ on CPU;
the test tolerance is 10⁻⁵):

| Sequence | Nowcast P(t) | Max forecast P(t+1…t+5) | Alert at the frozen 0.69 boundary |
|---|---|---|---|
| 0 | 0.304 | 0.752 | yes |
| 100 | 0.150 | 0.474 | no |
| 500 | 0.001 | 0.015 | no |
| 1000 | 0.596 | 0.760 | yes |
| 2000 | 0.019 | 0.104 | no |

`tests/test_submission_regression.py` checks this on every test run.

## Format contract

Everything below is what `src/prepared_input_loader.py` and
`src/data_input.py` actually enforce.

| Property | Value |
|---|---|
| Feature count | 52 — 26 base features + their 26 `_rel_change` counterparts |
| Feature order | `data/processed/multiday_benchmark_K5/input_feature_names.txt` (same list, with index and kind, in `docs/v4_feature_schema.json`) |
| Sequence length | 10 consecutive 10-second windows (100 s of history) |
| Scaling | **Values must already be in frozen-scaler space.** Neither loader applies scaling. The benchmark arrays were scaled by `data/processed/multiday_benchmark_K5/input_scaler.pkl` (a `StandardScaler` fitted on the training partition). |
| Labels | Not included, not needed. |
| Metadata | Not included. For CSV, a `timestamp` column is optional. |
| Contract | Frozen V4 contract, `docs/v4_baseline_manifest.json` |

### NPY

- Accepted shapes: `(N, 10, 52)`, or `(10, 52)` for one sequence (expanded to `(1, 10, 52)`).
- Any NaN or Inf rejects the **whole file**.
- Loaded with `allow_pickle=False`.

### CSV

- Must contain all 52 canonical feature columns, with the exact names. Column order in the file does not matter; the loader reorders to the canonical order.
- One row per 10-second window.
- Sequences are built with a **sliding** 10-row window: R valid rows give R − 9 sequences.
- If a `timestamp` column is present, rows are sorted by time, and a 10-row window is kept only if consecutive rows are 10 s apart (±1 s).
- Non-numeric values in a feature column reject the file.
- Rows containing NaN or Inf are **dropped** with a warning, and the rest are used. With no `timestamp` column, rows either side of a dropped row become adjacent, so a sequence can span the gap. Include timestamps if that matters.
- Recognised extra columns, which are ignored as features: `timestamp`, `window_idx`, `window_index`, `attack_present`, `attack_label`, `label`, `sequence_id`, `day`, `source_day`. Other unknown columns are ignored with a warning.
- Fewer than 10 usable rows, duplicate column names or an empty file are rejected with an explanation.

## Files that are not model input

Before validating, the upload page works out what an uploaded file is, so a
wrong file gets an explanation rather than a bare shape error:

| If you upload | It is recognised as |
|---|---|
| `Y_multistep_test.npy` `(N, 5, 26)` | future target states, not model input |
| `A_binary_multistep_test.npy` / `A_soft_multistep_test.npy` `(N, 5)` | ground-truth labels, not model input |
| `test_metadata.csv`, `screening_metadata.csv` | metadata only |
| `screening_predictions.csv` | stored model predictions (an output, not an input) |
| `event_timeline.csv` | display-only annotations |

Identification is based on the file's content (array shape, CSV columns) and
backed up by its filename, so a renamed file is still identified correctly.

## Programmatic use

```python
import sys; sys.path.insert(0, "src")
import data_input as dinp
from v4_explainer import V4Explainer

raw = open("examples/prepared_v4/sample_input.npy", "rb").read()
prepared = dinp.from_upload("sample_input.npy", raw)   # identify + validate
assert prepared.is_valid, prepared.errors
probs = dinp.run_inference(prepared, V4Explainer(device="cpu"), 5)
# probs.shape == (5, 6): column 0 = nowcast, columns 1..5 = P(attack at t+k)
```
