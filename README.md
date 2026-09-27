# Predictive Cyber Defence Using World Models

**Smart India Hackathon 2026 · Problem Statement ID 26153**

> ### Quick start
> 1. Clone the repository.
> 2. Install the pinned dependencies (Python **3.14** validated; 3.13 also installs).
> 3. Run the Streamlit dashboard **from the repository root**.
> 4. On **📥 Data Input**, choose **📡 Upload PCAP** or **📂 Upload Prepared V4 Data**.
> 5. Upload a supplied example from `examples/`.
>
> ```bash
> git clone https://github.com/Adi-og004/SIH-26153.git
> cd SIH-26153
> python3.14 -m venv .venv && source .venv/bin/activate
> python -m pip install -r requirements-screening-lock.txt
> streamlit run src/app.py
> ```
>
> Example inputs: `examples/prepared_v4/sample_input.npy` (prepared data) and `examples/pcap/demo_portscan.pcap` (raw traffic).
> Full judge walkthrough: [`docs/DEMO_RUNBOOK.md`](docs/DEMO_RUNBOOK.md).

---

## 1. Overview

Most intrusion detection classifies traffic that has already happened: this
flow or this window is benign or malicious. This project instead treats the
monitored network as a **dynamical system**. Every 10 seconds of traffic
becomes a 52-dimensional network-state vector. A learned **latent world
model** encodes the last 100 seconds of those states and rolls the latent
state forward 1–5 steps (10–50 s), predicting both the future network state
and the probability of attack at each step. Rule-based MITRE ATT&CK mapping,
Integrated Gradients explanations, a reliability status and an explicit
abstain option turn those forecasts into **decision support for an
analyst**, not automated response. The repository also documents, just as
explicitly, what the current evaluation does and does not show about
forecasting (§13, §15).

## 2. Problem statement

SIH 26153 asks for proactive cyber defence: anticipating how a threat will
evolve, rather than only flagging what has already been observed.
Multi-stage attacks such as bot C2 activity develop over time, and a
per-flow classifier reacts only once malicious traffic is visible. The
project asks whether a model of **network-state dynamics** can:

- represent the evolving state of a network from flow-level telemetry;
- forecast near-future threat probability and network state;
- map forecasts to ATT&CK tactics and explain them; and
- do this honestly: say when it should not be trusted, and compare against simple baselines.

## 3. Proposed solution

```
Network traffic ─ PCAP upload, or CIC flow records (CICFlowMeter)
      ↓
Flow reconstruction / flow records ─ bidirectional 5-tuple flows, 120 s idle timeout (PCAP path)
      ↓
10-second temporal windows
      ↓
26 base network-state features per window  (+26 relative-change features)  = 52-D state
      ↓  frozen StandardScaler
Sequence of 10 windows  (100 s of history)            input shape (10, 52)
      ↓
Temporal encoder ─ 2-layer LSTM (hidden 128) → LayerNorm → Linear
      ↓
Latent state z_t ∈ ℝ³²
      ↓
Gated residual transition ─ z' = LayerNorm(g·MLP(z) + (1−g)·z),  g = σ(Wz)
      ↓  applied recursively
Multi-step latent rollout  z_{t+1} … z_{t+5}
      ├──► Attack head → P(attack at t+k), k = 0…5       (threat probability / progression)
      └──► Decoder     → predicted 26-D network state at t+k
      ↓
Rule-based MITRE ATT&CK mapping  +  attack-path hypothesis
      ↓
Integrated Gradients attribution (features × time)
      ↓
Reliability status → ALERT / BENIGN / ABSTAIN → model-based what-if defence projections
```

Every dimension above is read from the frozen checkpoint and
`docs/v4_baseline_manifest.json`.

## 4. Current implementation status

| | Status |
|---|---|
| **V4 — CURRENT** | Validated, **frozen** reference model. It powers the dashboard, the PCAP path and the prepared-input path, and it is the benchmark reference. Its checkpoint, scalers, feature order and thresholds are hash-pinned and must not change. |
| **V5 — NEXT UPGRADE** | Development only. A packet-aware 82-D representation (V4's 52 features, unchanged, plus 30 packet-level features: TTL, TCP window, fragmentation, retransmission, payload distribution). The feature pipeline is built and tested. An experimental V5 checkpoint exists but was trained outside this repo and did not finish its curriculum; there is **no verified V5 evaluation**. **V5 does not replace V4** unless it is explicitly promoted after validation. |

### Development status

| Stable | Experimental | Future |
|---|---|---|
| V4 dashboard, V4 inference, PCAP → V4 pipeline, prepared CSV/NPY input, benchmark/reference evaluation, explainability, reliability, abstain, attack-path and what-if layers | V5 packet-feature engineering, V5 diagnostics, experimental V5 checkpoint | V5 training on a complete dataset, validation on the V4 protocol, then migration |

## 5. Architecture (V4)

| Component | Definition |
|---|---|
| Window | 10 s |
| Sequence | 10 consecutive windows (100 s) — windows must be exactly 10 s apart; no gap is stitched |
| Input | (10, 52): 26 base + 26 relative-change features, standardised |
| Encoder | `LSTM(52→128, 2 layers, dropout 0.20)`, last timestep → `LayerNorm(128)` → `Linear(128→32)` |
| Latent | 32-D, deterministic point estimate (not a probabilistic transition) |
| Transition | candidate `MLP(32→128→32)`; gate `σ(Linear(32→32))`; `z' = LayerNorm(gate·candidate + (1−gate)·z)` |
| Decoder | `Linear(32→64) → ReLU → Linear(64→26)`: next 26-D base state |
| Attack head | `Linear(32→32) → ReLU → Dropout(0.10) → Linear(32→1) → sigmoid` |
| Rollout | `k_step_rollout(x, K)`: index 0 = nowcast from z_t; index k = after k transitions = P(attack at t+k); K = 5 |
| Size | 244,027 parameters (`models/latent_world_model_v4/latent_world_model_v4.pt`, 987 KB) |
| Dashboard alert rule | max P(t+1…t+5) ≥ **0.69** (the K=5 trajectory-max threshold, chosen on validation only) |

## 6. Feature contract

- **52 features** in a fixed order: `data/processed/multiday_benchmark_K5/input_feature_names.txt`. The same list, with index and kind, is in `docs/v4_feature_schema.json`. The dashboard refuses to start if the two disagree.
- **26 base features** per 10 s window: `flow_count, log_flow_count, log_total_packets, log_total_bytes, window_packet_rate, window_byte_rate, syn_rate, ack_rate, rst_rate, fin_rate, psh_rate, urg_rate, mean_pkt_len, std_pkt_len, log_mean_flow_duration, log_mean_fwd_iat, log_mean_bwd_iat, fwd_packet_ratio, bwd_packet_ratio, fwd_byte_ratio, bwd_byte_ratio, mean_down_up_ratio, log_unique_dst_ports, unique_protocols, dst_port_entropy, protocol_entropy`.
- **26 relative-change features** (`<base>_rel_change`): `(x_t − x_{t−1}) / (|x_{t−1}| + 1e-8)`, set to 0 on the first window, ±inf → 0, clipped to [−10, 10] (`src/prepare_multiday_benchmark.py`).
- **Scaling**: a `StandardScaler` fitted on the training partition only (8,575 windows): `data/processed/multiday_benchmark_K5/input_scaler.pkl`.
- **Training-side definition**: `archive/create_network_states_v2.py` (over CIC flow records). **PCAP-side**: `src/pcap_ingest_v4.py`, which ports those formulas. Map: `docs/feature_contract_v4.json`.

## 7. Dashboard

`streamlit run src/app.py` opens a 14-page console. The default source is
the **Historical Benchmark**: the frozen model replayed over the held-out
day. On **📥 Data Input** you can switch the source:

| Mode | What you upload | What happens |
|---|---|---|
| **📡 Upload PCAP** | `.pcap` / `.pcapng` (up to 600 MB, per `.streamlit/config.toml`) | Streaming parse → flows → 10 s windows → 52 features → frozen scaler → 10-window sequences → **frozen V4 rollout** |
| **📂 Upload Prepared V4 Data** | `.npy` of shape (N, 10, 52), or `.csv` with the 52 feature columns | Identify → validate → **the same frozen V4 rollout** |

All three sources end in one function, `data_input.run_inference()`, which
calls the provenance-locked model's `k_step_rollout`. There is no second
inference implementation. Pages: Live Forecast, Early Warning, Event
Timeline, Trajectory Rollout, Window Inspector, Feature Attribution,
Reliability / Model Health, MITRE ATT&CK Timeline, Attack Path /
Progression, Analyst Decision, Defend Now Simulator, Counterfactual Defense
Lab, Methodology & Limits.

## 8. Prepared input format

These rules come from `src/prepared_input_loader.py` and `src/data_input.py`.
Details and expected outputs are in
[`examples/prepared_v4/README.md`](examples/prepared_v4/README.md).

**Values must already be scaled** with the frozen input scaler. Neither
loader rescales.

**NPY**: `float` array of shape `(N, 10, 52)`, or `(10, 52)` for one sequence.
Any NaN or Inf rejects the file. Example: `examples/prepared_v4/sample_input.npy`
(5 real held-out sequences).

**CSV**: one row per 10 s window, with columns named exactly as the 52
canonical features (any column order). Sequences are formed with a sliding
10-row window, so R rows give R − 9 sequences. An optional `timestamp`
column is used for ordering, and then only runs of rows exactly 10 s apart
(±1 s) form sequences. Optional columns that are ignored as features:
`window_idx`, `attack_present`, `label`, `sequence_id`, `day`, … The snippet
below is the start of `examples/prepared_v4/sample_input.csv`.

```
flow_count,log_flow_count,log_total_packets,...,protocol_entropy_rel_change   ← 52 canonical names
-0.24539389,0.13844787,-0.028465573,...,-0.1848742                           ← window 1 (already scaled)
-0.17967771,0.21492526,-0.10613316,...,0.4535739                             ← window 2
...                                                                          ← ≥ 10 rows in total
```

**Rejected, with a readable reason**: a wrong shape or feature count,
missing or duplicate columns, non-numeric values, NaN/Inf (NPY: whole file;
CSV: that row, with a warning), fewer than 10 usable rows, and unsupported
extensions. Common project files that are **not** model input are
identified and explained before any shape check:

| File | Message |
|---|---|
| `Y_multistep_test.npy` (N, 5, 26) | future target states, not V4 model input |
| `A_binary_multistep_test.npy`, `A_soft_multistep_test.npy` (N, 5) | ground-truth labels, not model input |
| `test_metadata.csv`, `screening_metadata.csv` | metadata only |
| `screening_predictions.csv` | stored predictions (the output of a previous run) |

## 9. How to run

**Requirements**: Python 3.14 (validated: 3.14.7 on macOS arm64). The
lockfile also installs cleanly on Python 3.13. It does **not** install on
3.11 (`contourpy==1.4.0` needs ≥ 3.12); 3.12 has not been tested. There are
no GPU requirements: the dashboard runs V4 on CPU.

```bash
git clone https://github.com/Adi-og004/SIH-26153.git
cd SIH-26153
python3.14 -m venv .venv
source .venv/bin/activate                       # Windows: .venv\Scripts\activate
python -m pip install -r requirements-screening-lock.txt
streamlit run src/app.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

- `requirements-screening-lock.txt`: full pinned environment, including transitive packages. This is the one to use.
- `requirements-screening.txt`: the 9 direct runtime pins, with the reason for each.
- `requirements-dev.txt`: the lockfile plus `pytest`.
- `requirements.txt`: an older unpinned list that includes research-only packages (for example `shap`). It is not needed for the dashboard.

**Always start Streamlit from the repository root.** The app uses paths
relative to the root.

Command-line PCAP check through the same pipeline:
`python src/pcap_ingest_v4.py examples/pcap/demo_portscan.pcap`.
It writes CSV, JSON and PNG outputs to `models/pcap_analysis_v4/`. Its
CLEAR/WARNING/CRITICAL label uses the checkpoint-embedded training threshold
(0.05), **not** the dashboard's validation-selected 0.69 boundary, so the two
labels can differ.

## 10. Testing

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

No dataset download is needed. Every test runs from a fresh clone.

| Test file | Covers |
|---|---|
| `tests/test_prepared_input_loader.py` | Valid CSV (with and without timestamps, extra/meta/label columns); invalid CSV (missing or wrong feature count, NaN, Inf, non-numeric, duplicate columns, empty, too few rows, non-contiguous timestamps); valid and invalid NPY (wrong feature dim, sequence length or ndim, NaN, Inf); dispatch; NPY/CSV vs benchmark prediction identity |
| `tests/test_input_mode_identity.py` | Benchmark == NPY upload == CSV upload predictions (tolerance 1e-5) on fixed probe sequences; wrong-file identification (targets, labels, metadata, predictions, annotations, renamed files); `run_inference` refuses invalid input |
| `tests/test_submission_regression.py` | Frozen artefact hashes match the manifest; the example files are unmodified benchmark data and reproduce the stored frozen predictions; **PCAP regression** (demo captures → same tensors and probabilities as recorded); PCAP-derived tensor re-uploaded as NPY gives **bit-identical** output |
| `tests/test_feature_parity.py` | Training vs PCAP feature-contract parity (fixtures) |
| `tests/test_v5_packet_features.py` | V5 packet-feature lane (development) |

## 11. Reproducibility

Full details: [`docs/reproducibility.md`](docs/reproducibility.md).

- **From a clone**: the dashboard, inference, the PCAP and prepared paths, and the stored screening predictions all reproduce. The checkpoint, scalers and feature order are hash-verified at load time against `docs/v4_baseline_manifest.json`. Inference is deterministic (`eval()`, `no_grad`, no sampling).
- **With the dataset**: the benchmark rebuild and the reference evaluation are deterministic. The reference script is pinned at git commit `2be3073`, which is why the history of this repository is preserved.
- **Training is not bit-reproducible.** The exact training arrays no longer exist. **Do not** run `src/train_latent_world_model_v4.py` in this checkout: it writes to the frozen checkpoint path.

## 12. Dataset and methodology

- **Dataset**: CSE-CIC-IDS2018 (CIC / UNB), <https://www.unb.ca/cic/datasets/ids-2018.html>. The CIC flow CSVs (produced with CICFlowMeter) are used for four days.
- **Split**: chronological. **Train/validation** = 14 Feb, 15 Feb and 28 Feb 2018, each split 85 % / 15 % in time order. **Test** = **Friday 2 March 2018**, held out entirely. Test labels are never used to fit anything or to choose a threshold.
- **Test set**: 3,115 sequences; 2,031 are attack-positive at t+1 (65 %). The day's attacks are **Bot** activity (three onsets). The capture has a real multi-hour gap, shown in the dashboard and never interpolated.
- **Label**: `attack_present` = 1 if any flow in the target 10 s window is labelled non-benign.
- **Not in this repository** (size and redistribution): raw CIC CSVs, PCAP archives, intermediate state files, and the train/validation tensors. Only the held-out test tensor, its metadata and labels, the scalers and the feature lists are committed. See [`data/README.md`](data/README.md) for how to rebuild them.

## 13. Evaluation (V4, verified)

All numbers below: **model V4 (frozen)**, **test split = Friday 2 March
2018 (held out)**, **benchmark `multiday_benchmark_K5`**, thresholds chosen
on **validation only**. Source: `docs/v4_baseline_manifest.json →
results_verified`, reproduced in `results/standardized_baseline_comparison.csv`.

**Point forecast: attack at t+k**

| Horizon | Threshold | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|---|
| k=1 (+10 s) | 0.45 | 0.941 | 0.432 | 0.592 | 0.856 | 0.051 |
| k=2 (+20 s) | 0.54 | 0.951 | 0.522 | 0.674 | 0.877 | 0.051 |
| k=3 (+30 s) | 0.65 | 0.961 | 0.545 | 0.695 | 0.889 | 0.042 |
| k=5 (+50 s) | 0.72 | 0.966 | 0.616 | 0.752 | 0.888 | 0.041 |

**Trajectory event: any attack in t+1…t+k, scored by max over the rollout**

| Horizon | Threshold | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|---|
| k=2 | 0.51 | 0.955 | 0.568 | 0.713 | 0.878 | 0.050 |
| k=3 | 0.68 | 0.966 | 0.475 | 0.637 | 0.890 | 0.032 |
| k=5 | 0.69 | 0.967 | 0.679 | **0.798** | 0.893 | 0.045 |

K = 4 is supported by the model but has no reference number: the reference
evaluation scores k = 1, 2, 3, 5 only.

**Baselines, same data and same targets** (point forecast; full table in
[`results/baselines/`](results/baselines/)):

| Horizon | Persistence F1 / AUC | Logistic regression F1 / AUC | V4 F1 / AUC |
|---|---|---|---|
| k=1 | 0.999 / 0.998 | 0.357 / 0.946 | 0.592 / 0.856 |
| k=5 | 0.997 / 0.995 | 0.407 / 0.946 | 0.752 / 0.888 |

**Early warning** (3 onsets, one day, so a small sample): at the 0.69
boundary V4 warned before **2 of 3** onsets. Under the **strict** definition
(only windows benign at both t and t+1 count), the mean lead is **15 s**.
The manifest's reference convention gives 66.7 % / 50 s, but it credits
alerts that still fall inside the previous attack. False-positive rate: 5.5
windows per hour.

**What these numbers do and do not show** (from the project's own studies):

- **Persistence nearly solves this benchmark.** Labels change between consecutive windows only 0.77 % of the time, so "the current state continues" scores F1 ≈ 0.997. The project's conclusion: this benchmark **cannot establish forecasting value beyond persistence** (`docs/PERSISTENCE_HYPOTHESIS.md`).
- **The rollout is stable but adds no measured forecasting signal.** Incremental ROC-AUC over a pure nowcast is negative at every horizon. K = 5 is the deepest *validated* horizon, not a demonstrated 50 s forecasting capability (`docs/LONG_HORIZON_ROLLOUT.md`).
- **Performance varies strongly by held-out day.** With each day held out in turn, V4's K=5 F1 ranges from 0.027 to 0.651, and ROC-AUC from 0.494 to 0.960. Onset detection is not above chance on any day (`docs/ROTATED_GENERALIZATION.md`).
- Logistic regression has the higher ROC-AUC. V4 has the higher F1 at its validation-selected thresholds.

Numbers in `models/latent_world_model_v4/metrics.csv` (e.g. F1 0.297) are
**training-time** values on a different split and are not reference results.

## 14. Explainability

- **Method**: Integrated Gradients (Captum, Gauss–Legendre, 1,024 steps; chosen as the first setting with under 1 % completeness error at every horizon). It runs through `src/v4_explainer.py`, which refuses to load unless the checkpoint, scaler and feature order match the manifest hashes.
- **Target**: the displayed probability P(attack at t+k), for k ∈ {1, 3, 5}, or the trajectory max.
- **Baseline**: the zero vector in scaled space, which is the average training window. A sensitivity check against the mean benign training sequence gives 7–8 of the top 10 features in common.
- **Outputs**: feature attribution (52 features) and temporal attribution (10 history steps), plus global importance, attack-vs-benign contrasts, and representative, FP/FN, onset and PCAP explanations (`results/v4_explainability/`).
- **Identity guarantee**: across 9,345 explanations, the probability being explained equals the displayed probability **bit-exactly** (`torch.equal`). If this ever fails, the dashboard withholds the explanation.
- **Cross-check**: gradient-free feature ablation.
- SHAP is **not** used by the dashboard. `src/shap_explainer.py` is a legacy, unlocked explainer kept for provenance only.
- **MITRE ATT&CK** is a separate, **rule-based** layer (`src/mitre_mapper.py`) that reads features and the forecast. V4 does not predict ATT&CK labels.

Details: [`docs/V4_EXPLAINABILITY.md`](docs/V4_EXPLAINABILITY.md).

## 15. Reliability and limitations

**Built-in safeguards**

- **Reliability status** (`src/v4_reliability.py`): RELIABLE / CAUTION / UNRELIABLE, computed from input shift against training quantiles, rollout and latent stability, and forecast self-consistency. It is a transparent rule, **not a calibrated confidence**. No test label was used to set it.
- **Abstain** (`src/v4_abstain.py`): the analyst decision is ALERT / BENIGN / **ABSTAIN**. UNRELIABLE inputs and near-boundary forecasts are handed to a human rather than auto-decided. The dashboard also warns that the selective-risk curve is largely a class-imbalance effect on this data.

**Limitations**

- **False positives and false negatives**: at the K=5 point threshold, V4 misses 38 % of attack windows (recall 0.616) and alerts on 4 % of benign windows. An error taxonomy is in `docs/ERROR_ANALYSIS.md`.
- **Forecasting value is not established** on this benchmark (§13). The rollout horizon is 50 s at most, and that should be read as a validated operating horizon, not a proven lead time.
- **Distribution shift across days** is large (§13). A single held-out day with three onsets is a small sample.
- **PCAP feature conformance**: PCAP features are computed by a faithful port of the training formulas, but real captures still differ from the CIC-processed training data. Capture scope matters (the training data aggregates a whole victim organisation; a single host sees a fraction of it), and TCP flag and IAT statistics depend on the flow engine. The synthetic demo PCAPs do not trigger alerts. Treat PCAP output as a pipeline demonstration, not a validated detector (`docs/FEATURE_PARITY.md`).
- **Normalisation mismatch**: the checkpoint was trained on a single-day split with a different scaler from the one used in the reference evaluation. This is recorded in the manifest and was not changed.
- **What-if defence** pages are **model-based projections**. V4 never observed interventions, so they are not causal estimates and perform no network action.
- **Single dataset**: generalisation beyond CSE-CIC-IDS2018 has not been evaluated.

## 16. Security and privacy

- The dashboard runs **locally**. Uploaded PCAP and prepared files are processed in the local Streamlit process, and no source file makes outbound network calls. An uploaded PCAP is written to a temporary file for parsing and deleted immediately after parsing.
- Streamlit keeps uploaded data in the browser session's server-side memory while the session is active. This README makes no claims about hosted deployments. If you deploy the app on a server, that server's operator is responsible for its privacy properties.
- The launch command above passes `--browser.gatherUsageStats=false` to turn off Streamlit's anonymous usage statistics.
- The repository contains no credentials, API keys or `.env` files.

## 17. Project structure

```
SIH-26153/
├── README.md                     ← this file
├── requirements-screening-lock.txt   full pinned runtime (use this)
├── requirements-screening.txt        direct runtime pins, annotated
├── requirements-dev.txt              lockfile + pytest
├── requirements.txt                  legacy unpinned research list
├── .streamlit/config.toml            upload size limit
├── src/                          ← all code (flat; see src/README.md)
│   ├── app.py                        Streamlit dashboard (entry point)
│   ├── pcap_ingest_v4.py             canonical PCAP → V4 pipeline + model class
│   ├── v4_explainer.py               provenance-locked loader + Integrated Gradients
│   ├── data_input.py                 unified input layer + single inference call
│   ├── prepared_input_loader.py      CSV / NPY validation
│   ├── screening_data.py, feature_contract.py, mitre_mapper.py,
│   │   attack_path_predictor.py, v4_reliability.py, v4_abstain.py,
│   │   v4_defense_simulator.py, v4_counterfactual.py      (runtime layers)
│   ├── run_* / build_* / verify_*    offline evaluation, artefact builders, verifiers
│   └── *v5*, packet_feature_builder.py   V5 development (not used by the app)
├── models/
│   ├── latent_world_model_v4/        CURRENT frozen checkpoint (+ training-time CSVs)
│   ├── latent_world_model_v5/        experimental V5 checkpoint (development only)
│   └── pcap_analysis_v4/             PCAP ↔ training feature-parity evidence
├── data/processed/
│   ├── multiday_benchmark_K5/        frozen scalers, feature lists, held-out test split
│   └── v5_multistep_K5/              V5 scalers (development)
├── docs/                         ← see docs/README.md (frozen manifests + study reports)
├── examples/
│   ├── prepared_v4/                  sample_input.npy / .csv + format README
│   └── pcap/                         3 synthetic demo captures + README
├── results/                      ← curated evidence (see results/README.md)
│   ├── v4_screening_mvp/             frozen replay substrate read by the dashboard
│   ├── v4_explainability/, v4_reliability/, abstain/, attack_path/,
│   │   defense_simulation/, counterfactual_defense/     runtime layers
│   ├── baselines/ (index), persistence_hypothesis/, long_horizon_rollout/,
│   │   rotated_generalization/, error_analysis/, feature_parity/, figures/
│   └── v5_features/                  V5 development
├── tests/                        ← pytest suite + fixtures
├── archive/                      ← scripts that built V4's data and baselines
└── .github/workflows/            manual-dispatch CICFlowMeter parity workflows
```

Why V4 files are not renamed to `models/v4/`: the frozen manifest pins them,
with SHA-256, at their current paths. See `models/README.md`.

## 18. Roadmap

**V4 (done)**
- ✓ Frozen, hash-pinned reference model and benchmark evaluation
- ✓ Streamlit demonstration dashboard (14 pages)
- ✓ PCAP ingestion → V4 inference (runs end to end; feature conformance is limited, see §15)
- ✓ Prepared feature input (CSV / NPY) with wrong-file identification
- ✓ Explainability with a prediction-identity guarantee
- ✓ Reliability status, abstain policy, attack-path and what-if layers
- ✓ Baseline, persistence, rotated-day, long-horizon and error studies

**V5 (next)**
- ✓ Packet-level feature lane and 82-D contract (built and tested)
- → Build a complete V5 training dataset (blocked: training days have no PCAPs locally)
- → Train and validate V5 on the V4 held-out protocol, against V4 and the baselines
- → A benchmark that tests forecasting beyond persistence (more onsets, days and attack families)
- → Replace V4 only after that validation

## 19. Citation and dataset acknowledgement

This work uses **CSE-CIC-IDS2018**, created by the Communications Security
Establishment (CSE) and the Canadian Institute for Cybersecurity (CIC),
University of New Brunswick: <https://www.unb.ca/cic/datasets/ids-2018.html>.

> Iman Sharafaldin, Arash Habibi Lashkari and Ali A. Ghorbani, "Toward
> Generating a New Intrusion Detection Dataset and Intrusion Traffic
> Characterization," *Proc. 4th International Conference on Information
> Systems Security and Privacy (ICISSP)*, 2018.

The flow records were produced by CIC with CICFlowMeter
(<https://github.com/ahlashkari/CICFlowMeter>). The dataset is not
redistributed here, and its use is subject to the provider's terms.
Libraries used include PyTorch, scikit-learn, Captum, Scapy, Streamlit and
Plotly.

**License**: no licence has been chosen for this repository yet, so no
reuse permission is granted beyond viewing it for SIH evaluation.
Third-party datasets and tools remain under their own terms.

## 20. SIH submission notes

| | |
|---|---|
| Runnable version | **V4** (frozen) |
| Entry point | `streamlit run src/app.py` (from the repository root) |
| Environment | Python 3.14, `requirements-screening-lock.txt` |
| Demo inputs | `examples/prepared_v4/sample_input.npy`, `…/sample_input.csv`, `examples/pcap/demo_*.pcap` |
| Model artefacts | `models/latent_world_model_v4/latent_world_model_v4.pt` and `data/processed/multiday_benchmark_K5/{input_scaler.pkl, target_scaler.pkl, input_feature_names.txt}`, pinned by `docs/v4_baseline_manifest.json` |
| Demo script | [`docs/DEMO_RUNBOOK.md`](docs/DEMO_RUNBOOK.md) |
| Tests | `python -m pytest -q` (after `pip install -r requirements-dev.txt`) |
| Future direction | V5 packet-aware representation, developed separately; V4 stays the reference until V5 is validated |
