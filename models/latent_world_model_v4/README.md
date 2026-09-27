# V4 Latent World Model — CURRENT (frozen)

`latent_world_model_v4.pt` is the model behind the Streamlit dashboard, the
PCAP path and the prepared-input path. It is **frozen**: its SHA-256
(`76af8d8a…e5898`) is pinned in `docs/v4_baseline_manifest.json`, and the
provenance-locked loader (`src/v4_explainer.py`) refuses to run if the file,
the input scaler or the feature order has changed.

**Do not run `src/train_latent_world_model_v4.py` in this checkout.** It
saves to this same path and would overwrite the frozen checkpoint. See
`docs/reproducibility.md`.

## Architecture (from the checkpoint and the manifest)

| Part | Definition |
|---|---|
| Encoder | `LSTM(52 → 128, 2 layers, dropout 0.20)` → last timestep → `LayerNorm(128)` → `Linear(128 → 32)` = latent z |
| Transition | candidate = `MLP(32 → 128 → 32, ReLU)`; gate = `sigmoid(Linear(32 → 32))`; z′ = `LayerNorm(gate·candidate + (1−gate)·z)` |
| Decoder | `Linear(32 → 64) → ReLU → Linear(64 → 26)` (next 26-D base state, target-scaled) |
| Attack head | `Linear(32 → 32) → ReLU → Dropout(0.10) → Linear(32 → 1) → sigmoid` |
| Parameters | 244,027 |
| Rollout | `k_step_rollout(x, K)` returns K+1 probabilities: index 0 = nowcast, index k = after k latent transitions = P(attack at t+k). Max K = 5. |

## Other files in this directory

| File | What it is |
|---|---|
| `metrics.csv`, `confusion_matrix.csv`, `test_attack_predictions.csv`, `training_history.csv`, `validation_threshold_search.csv`, `trajectory_evaluation.csv` | **Training-time** artefacts, from the single-day Feb-28 split used during training. They are **not** the reference results (see the manifest's `known_issues`). `trajectory_evaluation.csv` has two columns swapped. Kept for provenance only. |
| `mitre_mapping/mitre_summary.csv` | An older, 502-row rule-based ATT&CK table. The dashboard's MITRE page deliberately refuses it for the 3,115-sequence benchmark, because its rows are positional and belong to a different window set. The page then shows a live, per-sequence ATT&CK reading instead. |

The reference held-out results live in `docs/v4_baseline_manifest.json →
results_verified` and are summarised in the root `README.md`.
