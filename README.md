# Predictive Cyber Defence Using World Models

Smart India Hackathon 2026 (SIH-26153)

## Problem

Traditional Intrusion Detection Systems (IDS) primarily classify whether a network packet or flow represents an attack *currently* happening. By the time the IDS alerts, the attack is already executing.

This project investigates whether **learned temporal network dynamics** can forecast future network behaviour and attack emergence. By learning a "World Model" of the network, we aim to predict future states and detect threats before they fully materialize.

## Architecture

Our core architecture models the network state over time and simulates future transitions:

```text
Network Traffic
      ↓
Feature Extraction
      ↓
52-Dimensional Network State (26 Base + 26 Relative-Change)
      ↓
10 Historical Time Windows (10-second intervals)
      ↓
V4 Latent World Model
      ↓
Latent State Transition
      ↓
Future State Rollout
      ↓
Future Threat Estimation
      ↓
Early Warning / ATT&CK Mapping / Explanation
      ↓
Dashboard
```

### Current V4 Architecture Highlights:
- **Input Dimensions**: 52-feature continuous state vectors.
- **Sequence Length**: 10 historical steps.
- **Encoder**: LSTM-based sequence encoder mapping to a latent space.
- **Latent Dimension**: Gated transition MLP operating natively in the latent space.
- **Decoder & Attack Head**: Reconstructs future 52-dim states and predicts the probability of future attacks.
- **Training**: Multi-step rollout training utilizing a combined loss of state-reconstruction (MSE) and attack-prediction (BCE).

## Dataset & Benchmark

The temporal benchmark simulates a real-world multi-day deployment using CICFlowMeter features:

- **Training Days**: Feb 14, Feb 15, Feb 28
- **Validation**: Chronological final 15% of each training day
- **Test**: Mar 2 (completely held out until final evaluation)

**Methodology Guarantee:**
- Relative changes are calculated strictly per source day prior to partitioning.
- Scalers are fitted exclusively on the training split.
- Sequences are validated to ensure exact 10-second contiguous gaps without crossing boundaries.
- Test labels are strictly isolated and never used for threshold tuning.

## Current Experiments & Findings

We recently completed a rigorous benchmark comparing our World Model to a **Persistence Baseline** and a **Logistic Regression (latest-state) Baseline**.

> [!IMPORTANT]
> **Key Finding**: The Persistence baseline achieves an F1 score of **99.7%–99.9%** on the ordinary $t+k$ future attack-label task. This demonstrates that the standard classification task is overwhelmingly dominated by attack persistence (if an attack is happening now, it continues to happen).

Because of this finding, the primary research milestone moving forward is **Pre-Attack Onset Prediction (Early Warning)** rather than ordinary $t+k$ state classification. The goal is predicting the *emergence* of an attack before the persistence rule triggers.

## Repository Structure

The current active pipeline is organized for clarity and reproducibility.

- `src/train_latent_world_model_v4.py`: Core V4 architecture and training loop.
- `src/prepare_multiday_benchmark.py`: Exact dataset prep for the multi-day benchmark.
- `src/k_step_rollout.py`: Validated V4 evaluation and trajectory event analysis.
- `src/evaluate_baselines.py`: Evaluates Persistence and Logistic Regression baselines.
- `src/app.py`: Real-time streaming dashboard.

> [!WARNING]
> **V3 Incompatibility Note**
> The current V4 training/evaluation pipeline is the research core. The existing SHAP (`shap_explainer.py`), MITRE mapping (`mitre_mapper.py`), and PCAP inference modules (`pcap_ingest.py`) still target the legacy V3 checkpoint and are not yet claimed as V4 production components. Refactoring these for V4 is pending.

*(For legacy/experimental scripts, please refer to the `archive/` directory).*
