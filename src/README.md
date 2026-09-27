# `src/` — module index

All modules sit flat in `src/`. Streamlit runs `src/app.py` with `src/` on
`sys.path`, so modules import each other by bare name (`import
pcap_ingest_v4`). Several frozen manifests (`docs/*.json`, `results/*/manifest*.json`)
record these exact `src/<name>.py` paths as provenance, so the files have
not been moved into sub-packages. This index groups them by role instead.

## 1. Dashboard runtime — STABLE (V4)

The complete import closure of `app.py`, and nothing else:

| Module | Role |
|---|---|
| `app.py` | Streamlit dashboard (entry point: `streamlit run src/app.py`, from the repo root) |
| `pcap_ingest_v4.py` | **Canonical PCAP → V4 pipeline** and model class: packets → bidirectional flows → 10 s windows → 26 base + 26 relative-change features → frozen scaler → 10-window sequences |
| `v4_explainer.py` | **Provenance-locked** V4 loader + Integrated Gradients explainer. It hash-checks the checkpoint, input scaler and feature order against `docs/v4_baseline_manifest.json` before loading. |
| `data_input.py` | Unified input layer. It identifies uploads, validates them, and makes the **single inference call** (`run_inference`) shared by the benchmark, PCAP, CSV and NPY sources. |
| `prepared_input_loader.py` | CSV / NPY parser and validator for prepared V4 feature data |
| `screening_data.py` | Read-only loader for the frozen replay substrate in `results/v4_screening_mvp/` |
| `feature_contract.py` | Machine-readable V4 feature contract (training vs PCAP implementation map) |
| `mitre_mapper.py` | Rule-based MITRE ATT&CK mapping from features and forecast |
| `attack_path_predictor.py` | ATT&CK-consistent progression hypothesis over the frozen rollout |
| `v4_reliability.py` | Rule-based reliability / model-health status (not a calibrated confidence) |
| `v4_abstain.py` | ALERT / BENIGN / ABSTAIN analyst-decision policy |
| `v4_defense_simulator.py` | Defend-now what-if projection (model-based, not causal) |
| `v4_counterfactual.py` | Multi-scenario what-if comparison |

## 2. V4 data, training and evaluation — offline (need the dataset)

| Module | Role |
|---|---|
| `prepare_multiday_benchmark.py` | Builds `data/processed/multiday_benchmark_K5` from `states_v2` |
| `train_latent_world_model_v4.py` | V4 training script. ⚠️ Writes to the frozen checkpoint path; see `docs/reproducibility.md`. |
| `k_step_rollout.py` | Rollout evaluation. The **reference** version is the one at commit `2be3073` (see `docs/reproducibility.md`). |
| `evaluate_baselines.py`, `train_logistic_regression_baseline.py`, `run_standardized_comparison.py` | Persistence and logistic-regression baselines, and the three-way comparison |
| `run_persistence_hypothesis.py`, `run_long_horizon_rollout.py`, `run_rotated_generalization.py` (+ `rotated_common.py`, `rotated_evaluate.py`), `run_error_analysis.py`, `run_feature_parity.py`, `run_v4_explainability.py` | Evaluation studies (one `docs/*.md` each) |
| `build_v4_screening_mvp.py`, `build_v4_reliability.py`, `build_attack_path.py`, `build_abstain.py`, `build_defense_simulation.py`, `build_counterfactual_defense.py` | Build the `results/` artefacts the dashboard reads |
| `verify_*.py`, `validate_dashboard.py`, `test_pcap_v4_pipeline.py` | Independent verification harnesses for the studies above |
| `generate_demo_pcaps.py` | Generates the synthetic demo captures (`examples/pcap/`) |

## 3. V5 development — EXPERIMENTAL (not used by the dashboard)

| Module | Role |
|---|---|
| `packet_feature_builder.py` | Packet-level feature lane (additive to V4's flow lane) |
| `v5_feature_contract.py` | 82-D V5 contract: V4's 52 as a byte-identical prefix, then 30 packet features |
| `infer_latent_world_model_v5.py` | Loader and inference for the experimental V5 checkpoint (`--self-test`) |
| `run_v5_packet_features.py`, `verify_v5_packet_features.py`, `build_v5_sequences.py`, `build_v5_multihost_demo.py`, `build_v5_mixed_scope_diagnostic.py`, `characterize_v5_rollout.py`, `run_v5_packet_ablation.py`, `analyze_v5_input_shift.py` | V5 feature construction, diagnostics and ablations |

## 4. Legacy — kept only because frozen docs cite them

| Module | Why it is still here |
|---|---|
| `pcap_ingest.py` | Deprecated V3 single-step PCAP ingestion (marked as such in its header). It is cited in the V4 manifest's contamination check. Not imported by anything. |
| `shap_explainer.py` | Unlocked legacy explainer. **Not** used by the dashboard, which uses `v4_explainer.py` (see `docs/V4_EXPLAINABILITY.md`). Cited by the manifest and the explainability contract. |
