# Reproducibility

There are three levels of reproduction. Only the first works from a fresh
clone alone. The other two need the CSE-CIC-IDS2018 data (see
`data/README.md`).

| Level | What you reproduce | Needs the dataset? | Status |
|---|---|---|---|
| 1. Runtime | Dashboard, V4 inference, PCAP path, prepared-input path, stored screening predictions | No | Verified from a fresh export of this repository |
| 2. Reference evaluation | The held-out V4 metrics in `docs/v4_baseline_manifest.json` | Yes (benchmark rebuild) | Deterministic; commands below |
| 3. Training | The V4 checkpoint itself | Yes | **Not bit-reproducible** (see below) |

## Environment

| Item | Value |
|---|---|
| Python | **3.14** (validated: 3.14.7, macOS arm64). A clean install of the lockfile also succeeded on 3.13. It **fails on 3.11**: `contourpy==1.4.0` needs ≥ 3.12. 3.12 has not been tested. |
| Runtime dependencies | `requirements-screening.txt` (9 direct pins) or `requirements-screening-lock.txt` (full pip-freeze lock, including transitive packages) |
| Tests | `requirements-dev.txt` = the lockfile + `pytest==9.1.1` |
| Device | The dashboard and tests run V4 on **CPU**. The stored screening predictions were produced on Apple MPS. On the example sequences, CPU reproduces them with a maximum difference of 5.4 × 10⁻⁷; `tests/test_submission_regression.py` enforces a 10⁻⁵ tolerance. |

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Level 1 — runtime (fresh clone)

Everything the dashboard needs is committed, and the frozen artefacts are
hash-checked at load time:

| Artefact | Path |
|---|---|
| Checkpoint | `models/latent_world_model_v4/latent_world_model_v4.pt` |
| Input / target scaler | `data/processed/multiday_benchmark_K5/{input,target}_scaler.pkl` |
| Feature order | `data/processed/multiday_benchmark_K5/input_feature_names.txt` (+ `docs/v4_feature_schema.json`) |
| Held-out test tensor | `data/processed/multiday_benchmark_K5/X_test.npy` |
| Replay substrate | `results/v4_screening_mvp/` |
| Provenance manifest | `docs/v4_baseline_manifest.json` |

`src/v4_explainer.py` compares the checkpoint, input scaler and feature order
with the SHA-256 values in the manifest. It raises `ProvenanceError` on any
mismatch. `python -m pytest -q` re-checks those hashes, the example inputs,
the PCAP path and the prediction identity between input modes.

Inference is deterministic: `model.eval()`, `torch.no_grad()`, no sampling.

## Level 2 — reference evaluation (needs the dataset)

1. Obtain the four CIC flow CSVs listed in `data/README.md` and place them in `data/raw/`.
2. Rebuild the benchmark:
   ```bash
   python archive/clean_datasets.py
   python archive/create_network_states_v2.py
   python src/prepare_multiday_benchmark.py
   ```
   Compare the resulting files with the SHA-256 values in
   `docs/v4_baseline_manifest.json → dataset`.
   `X_test.npy` must equal the committed copy.
3. Run the **pinned** reference evaluation script. It is the version of
   `src/k_step_rollout.py` at commit `2be3073`; the version at HEAD reads a
   different split and does **not** reproduce the reference numbers:
   ```bash
   git show 2be3073:src/k_step_rollout.py > /tmp/k_step_rollout_reference.py
   python /tmp/k_step_rollout_reference.py      # run from the repository root
   ```
   This writes to `models/latent_world_model_v4/k_step_rollout/`. It reads
   the checkpoint but does not modify it. Its SHA-256 is recorded in the
   manifest as `reference_script_sha256`.
4. `python src/verify_v4_baseline.py` re-checks the numbers against the manifest.

The other analyses (baseline comparison, persistence, rotated days,
long-horizon rollout, explainability, reliability, and so on) each have a
`run_*` or `build_*` script and a `verify_*` script in `src/`. Their docs in
`docs/` name the exact command.

## Level 3 — training (not bit-reproducible)

> ⚠️ **Do not run `src/train_latent_world_model_v4.py` in this checkout.** It
> saves to `models/latent_world_model_v4/latent_world_model_v4.pt` and would
> overwrite the frozen checkpoint. Run it in a separate copy if you want to
> study training.

Recorded facts (manifest → `training_provenance`, `known_issues`):

- The checkpoint was trained on `data/processed/temporal_benchmark_v2/multistep_K5`, a **single-day (Feb 28)** split, not on `multiday_benchmark_K5`. It was built by `archive/prepare_temporal_benchmark_v2.py` and `archive/prepare_multistep_data.py`.
- Hyperparameters: seed 42, AdamW (lr 5e-4, weight decay 1e-5), batch 64, up to 120 epochs, patience 20, SmoothL1 state loss + BCE attack loss (pos_weight 5.807, weight 0.5), γ = 0.8, K-step curriculum up to K = 5, gradient clip 1.0.
- The exact training arrays no longer exist; regenerated arrays differ slightly (2,022/348 vs 2,021/348 sequences). **Training is therefore not bit-reproducible. Evaluation is.**
- The reference evaluation feeds the checkpoint arrays scaled by the multi-day scaler, which is a different scaler from the one used in training. The manifest records this as an out-of-distribution-normalisation evaluation. It is recorded and was not changed.
- The training script picks its own trajectory thresholds using test labels. Those training-time numbers (e.g. `metrics.csv`, test F1 0.297) must not be quoted. The reference evaluation selects every threshold on validation only.
