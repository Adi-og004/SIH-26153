# Model Checkpoints

> [!IMPORTANT]
> **Large model binaries (`.pt`, `.pth`, `.ckpt`) are ignored by Git.**

## Folder Layout
The `models/` directory will host various versions of the Latent World Model. Currently, the most relevant directory is:

```text
models/
└── latent_world_model_v4/
    ├── latent_world_model_v4.pt     (Main trained checkpoint)
    ├── k_step_rollout/              (Evaluation results)
    └── diagnosis/                   (Generated debugging plots)
```

## Running Inference
The `app.py` dashboard, as well as `k_step_rollout.py`, expect the V4 checkpoint to exist exactly at `models/latent_world_model_v4/latent_world_model_v4.pt`.

If you pull this repository cleanly, you must either:
1. Manually download the checkpoint provided by the researchers and place it here.
2. Run `src/train_latent_world_model_v4.py` locally to train a fresh checkpoint.
