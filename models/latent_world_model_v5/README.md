# V5 Latent World Model — DEVELOPMENT ONLY

> **Not used by the dashboard.** V4 (`../latent_world_model_v4/`) is the
> current, validated model. Nothing in `src/app.py` loads this checkpoint.

`latent_world_model_v5.pt` is an **experimental** checkpoint for the planned
packet-aware V5 representation (82-D input: V4's 52 features, byte-identical,
followed by 30 packet-level features). See `docs/V5_PACKET_FEATURES.md` and
`docs/v5_packet_feature_contract.json`.

## What is known about this checkpoint

Everything below comes from the checkpoint's own embedded metadata:

| Field | Value |
|---|---|
| `model_class` | `LatentWorldModelV5` |
| Input / hidden / latent / decoder | 82 / 128 / 32 / 26 |
| `epoch` | 8 |
| `full_curriculum_completed` | **False** |
| `best_validation_f1` | 0.590 (validation split of its own training run) |
| `selected_threshold` | 0.5 |

## Caveats

- **It was produced outside this repository.** No V5 training script and no V5
  training dataset are included. `docs/V5_PACKET_FEATURES.md` was written
  before the checkpoint existed and still says that no V5 model is trained.
  That statement describes the in-repo pipeline and is out of date for this
  file.
- Training stopped at epoch 8, before the K-step curriculum finished.
- **No held-out V4-vs-V5 comparison has been verified.** No V5 performance
  claim is made anywhere in this submission.

## Loading it

`src/infer_latent_world_model_v5.py` rebuilds the architecture from the
checkpoint's tensor shapes and loads it with the V5 scalers in
`data/processed/v5_multistep_K5/`:

```bash
python src/infer_latent_world_model_v5.py --self-test
```

V5 will replace V4 only after it has been trained on a complete dataset and
validated on the same held-out protocol as V4.
