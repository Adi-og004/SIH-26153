# Reproducibility Guide

Follow these steps to reproduce the exact benchmark results from scratch.

## 1. Environment Setup
Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Dataset Preparation
Ensure the raw CSVs are placed in `data/processed/states_v2/` (see `data/README.md`).
Then, generate the multi-day benchmark:
```bash
python src/prepare_multiday_benchmark.py
```
*This will create the exactly scaled, 10-second contiguous sequences in `data/processed/multiday_benchmark_K5`.*

## 3. Training
To train the V4 Latent World Model from scratch:
```bash
python src/train_latent_world_model_v4.py
```
*The checkpoint will be saved to `models/latent_world_model_v4/latent_world_model_v4.pt`.*

## 4. Rollout Evaluation
Evaluate the V4 model's k-step predictions and trajectory-events:
```bash
python src/k_step_rollout.py
```

## 5. Baseline Evaluation
Evaluate the Persistence and Logistic Regression (latest-state) baselines for direct comparison:
```bash
python src/evaluate_baselines.py
```

## 6. Dashboard Execution
To launch the real-time Streamlit dashboard:
```bash
streamlit run src/app.py
```
