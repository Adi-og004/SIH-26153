# V4 Reference Baseline — Frozen Specification

**Status:** FROZEN — 2026-09-19
**Baseline version:** V4 (Latent World Model, multi-step objective)
**Git commit at freeze:** `53c22227d368b82ff9e736206a92e85b6bed2514` (branch `cicflowmeter-validation`, worktree clean)
**Machine-readable companion:** [docs/v4_baseline_manifest.json](v4_baseline_manifest.json), [docs/v4_feature_schema.json](v4_feature_schema.json)
**Verifier:** `python src/verify_v4_baseline.py`

This document records the exact model, data, preprocessing, feature order, scalers,
threshold, split and evaluation procedure that produced the currently reported V4
results. Nothing here was modified, retrained or "repaired" — where the repository is
inconsistent, the inconsistency is documented in [§12](#12-known-limitations--issues).

---

## 1. What is frozen

| Item | Frozen value |
|---|---|
| Checkpoint | `models/latent_world_model_v4/latent_world_model_v4.pt` (SHA256 `76af8d8a…5898`) |
| Feature schema | 52 features, fixed order, `data/processed/multiday_benchmark_K5/input_feature_names.txt` |
| Input scaler | `data/processed/multiday_benchmark_K5/input_scaler.pkl` |
| Target scaler | `data/processed/multiday_benchmark_K5/target_scaler.pkl` |
| Benchmark arrays | `data/processed/multiday_benchmark_K5/*.npy` (hashed in the manifest) |
| Split | Train Feb-14 / Feb-15 / Feb-28 (85% head each) · Val = 15% tail of each · Test = Mar-2 |
| Thresholds | Per-horizon, validation-selected: 0.45 / 0.54 / 0.65 / 0.72 (K=1/2/3/5); MAX-trajectory 0.51 / 0.68 / 0.69 (K=2/3/5) |
| Rollout | Latent autoregressive, MAX_K = 5, horizons evaluated {1, 2, 3, 5} |
| Reference evaluation script | `src/k_step_rollout.py` **as of commit `2be3073`** |

The checkpoint, arrays and scalers are `.gitignore`'d (`data/*`, `models/*`), so the
SHA256 hashes in the manifest are the only provenance record. Do not regenerate them.

---

## 2. Exact checkpoint

| Property | Value |
|---|---|
| Path | `models/latent_world_model_v4/latent_world_model_v4.pt` |
| SHA256 | `76af8d8a9e02215923b1c0ab6b627bfc0809e856748666a55ee4b16cec5e5898` |
| Size | 986,987 bytes |
| Modified | 2026-09-02 10:50:22 local |
| `model_class` | `LatentWorldModelV4` |
| Parameters | 244,027 across 28 `state_dict` tensors |
| Loads `strict=True` | Yes — verified by running the evaluation script |

Embedded configuration inside the checkpoint (it is a dict, not a bare `state_dict`):

```
input_size=52  hidden_size=128  latent_size=32  output_size=26  num_layers=2
sequence_length=10  max_k=5  gamma=0.8  transition_type="deterministic"
curriculum=[(1,1),(21,2),(41,3),(61,4),(81,5)]
positive_class_weight=5.807471264367816
best_validation_f1=0.8043478260869565   selected_threshold=0.05
test_state_loss=0.2688717890073596  test_attack_f1=0.2966507177033493
test_roc_auc=0.6708944543828265     test_fpr=0.018461538461538463
```

> The embedded `selected_threshold=0.05` and `test_*` metrics are **training-time**
> values on the single-day `temporal_benchmark_v2` split. They are **not** the
> reference benchmark numbers (§11) and are not used by the reference evaluation.

---

## 3. Exact architecture

```
X_t ∈ R^(10 × 52)
  └─ nn.LSTM(52 → 128, num_layers=2, batch_first=True, dropout=0.20)
       └─ last timestep h ∈ R^128
            └─ LayerNorm(128)
                 └─ Linear(128 → 32)                       ⇒ z_t ∈ R^32

transition_step(z):                                         (DETERMINISTIC)
  candidate = Linear(32→128) → ReLU → Linear(128→32)
  gate      = sigmoid(Linear(32→32))
  z'        = LayerNorm(gate·candidate + (1−gate)·z)

decode(z)        = Linear(32→64) → ReLU → Linear(64→26)     ⇒ Ŝ ∈ R^26
predict_attack(z)= Linear(32→32) → ReLU → Dropout(0.10) → Linear(32→1) → sigmoid
```

The transition is a **learned deterministic point estimate** in latent space. It is
*not* a probability distribution `P(S_{t+1} | S_t)`. Threat probability comes solely
from the attack-probability head applied to the rolled-forward latent.

`k_step_rollout(x, k)` returns **K+1** probabilities: index `0` is the encoder latent
`z_t` with no transition applied; index `j` is the state after `j` transitions, i.e.
the prediction for `t+j`.

---

## 4. Exact input representation

| Property | Value |
|---|---|
| Input shape | **10 × 52** |
| Sequence length | 10 consecutive 10-second windows → 100 s of history |
| Base features | 26 |
| Relative-change features | 26 |
| Total | 52 |
| Decoder output | 26 (base features only, target-scaled) |

Relative-change definition (computed **per source day, chronologically, before the
train/val split**):

```
r_t = (s_t − s_{t−1}) / (|s_{t−1}| + 1e−8)
r_0 = 0.0 ;  ±inf/NaN → 0.0 ;  clipped to [−10, +10]
```

Preprocessing chain, in order:

1. 10-second window aggregation of CICFlowMeter flows → `data/processed/states_v2/*.csv`
2. `pd.to_numeric(errors="coerce")`; `±inf → NaN`; `NaN → 0.0` on the 26 base features
3. append the 26 `*_rel_change` features per source day
4. `StandardScaler` (fit on the train partition only) over all 52 input features
5. `np.nan_to_num(nan=0.0, posinf=10.0, neginf=−10.0)` → `float32`
6. build sequences of 10 windows; a block is kept only if **every** consecutive
   timestamp gap in `inputs + 5 targets` equals exactly 10 s

---

## 5. Exact feature order

Authoritative file: `data/processed/multiday_benchmark_K5/input_feature_names.txt`
(SHA256 `137b58333f1c751522f30478219ff86b8c115aa749defb57ae80ee83559ecbb5`).
Frozen machine-readable copy: `docs/v4_feature_schema.json`.

Indices 0–25 are the base features; indices 26–51 are the **same 26 in the same
order**, suffixed `_rel_change`.

| idx | base feature (0–25) | idx | relative-change feature (26–51) |
|---:|---|---:|---|
| 0 | `flow_count` | 26 | `flow_count_rel_change` |
| 1 | `log_flow_count` | 27 | `log_flow_count_rel_change` |
| 2 | `log_total_packets` | 28 | `log_total_packets_rel_change` |
| 3 | `log_total_bytes` | 29 | `log_total_bytes_rel_change` |
| 4 | `window_packet_rate` | 30 | `window_packet_rate_rel_change` |
| 5 | `window_byte_rate` | 31 | `window_byte_rate_rel_change` |
| 6 | `syn_rate` | 32 | `syn_rate_rel_change` |
| 7 | `ack_rate` | 33 | `ack_rate_rel_change` |
| 8 | `rst_rate` | 34 | `rst_rate_rel_change` |
| 9 | `fin_rate` | 35 | `fin_rate_rel_change` |
| 10 | `psh_rate` | 36 | `psh_rate_rel_change` |
| 11 | `urg_rate` | 37 | `urg_rate_rel_change` |
| 12 | `mean_pkt_len` | 38 | `mean_pkt_len_rel_change` |
| 13 | `std_pkt_len` | 39 | `std_pkt_len_rel_change` |
| 14 | `log_mean_flow_duration` | 40 | `log_mean_flow_duration_rel_change` |
| 15 | `log_mean_fwd_iat` | 41 | `log_mean_fwd_iat_rel_change` |
| 16 | `log_mean_bwd_iat` | 42 | `log_mean_bwd_iat_rel_change` |
| 17 | `fwd_packet_ratio` | 43 | `fwd_packet_ratio_rel_change` |
| 18 | `bwd_packet_ratio` | 44 | `bwd_packet_ratio_rel_change` |
| 19 | `fwd_byte_ratio` | 45 | `fwd_byte_ratio_rel_change` |
| 20 | `bwd_byte_ratio` | 46 | `bwd_byte_ratio_rel_change` |
| 21 | `mean_down_up_ratio` | 47 | `mean_down_up_ratio_rel_change` |
| 22 | `log_unique_dst_ports` | 48 | `log_unique_dst_ports_rel_change` |
| 23 | `unique_protocols` | 49 | `unique_protocols_rel_change` |
| 24 | `dst_port_entropy` | 50 | `dst_port_entropy_rel_change` |
| 25 | `protocol_entropy` | 51 | `protocol_entropy_rel_change` |

`state_feature_names.txt` (SHA256 `64a8297798…b5499`) holds exactly indices 0–25 and
is the decoder's 26-dim output order.

---

## 6. Exact scalers

| | Input scaler | Target scaler |
|---|---|---|
| Path | `data/processed/multiday_benchmark_K5/input_scaler.pkl` | `data/processed/multiday_benchmark_K5/target_scaler.pkl` |
| SHA256 | `e0ab9bebd26869950dda51a67bdde3192b0edbd00e20b623115ed2b653f04f19` | `7155dec48f89d246359741032b002ca0bb44e4739e99daa3e28e77e312c55944` |
| Size | 2,938 B | 1,589 B |
| Class | `sklearn.preprocessing.StandardScaler` | `sklearn.preprocessing.StandardScaler` |
| `n_features_in_` | 52 | 26 |
| `n_samples_seen_` | 8,575 | 8,575 |
| Fitted on | train partition only (85% chronological head of Feb-14 + Feb-15 + Feb-28 = 2766+2914+2895) | same 8,575 windows, 26 base features |
| `mean_[:3]` | `[266.746356, 5.20338, 7.474836]` | `[266.746356, 5.20338, 7.474836]` |
| `scale_[:3]` | `[243.471249, 0.969018, 1.177411]` | `[243.471249, 0.969018, 1.177411]` |

Both are applied once, at dataset-build time, by `src/prepare_multiday_benchmark.py`;
the `.npy` arrays the evaluation consumes are already scaled. At PCAP inference time
`src/pcap_ingest_v4.py` loads these same two files (`CANON_DIR = multiday_benchmark_K5`).

The scalers under `data/processed/temporal_benchmark_v2/` (SHA256 `1d6a69cb…6735d` /
`2991c8bb…fa611`, `n_samples_seen_=2384`) are a **different, single-day Feb-28 fit**
and are **not** part of this baseline — see §12.

---

## 7. Exact dataset split

Benchmark directory: `data/processed/multiday_benchmark_K5`
Builder: `src/prepare_multiday_benchmark.py` · source: `data/processed/states_v2/*.csv`

| Split | Source days | Windows | Sequences | Attack (t+1) | Benign (t+1) |
|---|---|---:|---:|---:|---:|
| Train | Feb-14, Feb-15, Feb-28 (85% chronological head of each) | 2766 + 2914 + 2895 = 8575 | 8,441 | 1,835 | 6,606 |
| Validation | Feb-14, Feb-15, Feb-28 (15% chronological tail of each) | 489 + 515 + 511 = 1515 | 1,473 | 392 | 1,081 |
| Test | **Mar-2-2018 only**, entirely held out | 3143 | 3,115 | 2,031 | 1,084 |

Per-day chronological boundaries (`VAL_RATIO = 0.15`):

| Day | Total windows | Train ends | Val starts |
|---|---:|---|---|
| Wednesday-14-02-2018 | 3,255 | 2018-02-14 11:38:20 | 2018-02-14 11:38:30 |
| Thursday-15-02-2018 | 3,429 | 2018-02-15 11:34:00 | 2018-02-15 11:34:10 |
| Wednesday-28-02-2018 | 3,406 | 2018-02-28 11:34:40 | 2018-02-28 11:34:50 |
| Friday-02-03-2018 (test) | 3,143 | — (no split) | 2018-03-02 01:00:00 → 12:59:50 |

Verified properties:

- Test day confirmed from `test_metadata.csv`: every row has `source_day == "Friday-02-03-2018"`.
- Relative-change features are computed **independently per source day**, before splitting.
- Scalers are fit on the train partition only — test never participates in any fitting.
- Sequences never cross a day boundary or a temporal gap (exact 10 s contiguity check).
- Observable Mar-2 attack onsets: `03:54:50`, `10:13:20`, `12:13:30`; an attack is already
  active at the first evaluable test window, so that episode has no observable onset.

Array shapes: `X_* = (N, 10, 52)`, `Y_multistep_* = (N, 5, 26)`,
`A_binary_multistep_* / A_soft_multistep_* = (N, 5)`. Per-file SHA256 hashes are in the manifest.

---

## 8. Exact threshold

**Every threshold is selected on validation only and frozen before touching test labels.**

- Search grid: `np.arange(0.01, 0.99, 0.01)`
- Objective: maximise validation F1 (strict `>`, so the lowest threshold wins ties)
- Applied unchanged to the Mar-2 test scores

| Task | K=1 | K=2 | K=3 | K=5 |
|---|---:|---:|---:|---:|
| Future attack prediction (t+k) | **0.45** | **0.54** | **0.65** | **0.72** |
| Trajectory event, MAX scoring | — | **0.51** | **0.68** | **0.69** |
| Trajectory event, MEAN heuristic | — | 0.49 | 0.55 | 0.62 |
| Early warning (K=5 MAX) | — | — | — | **0.69** |

Validation F1 values used to pick them: 0.5544 / 0.5650 / 0.5745 / 0.5860 (point
horizons); 0.5557 / 0.5641 / 0.5639 (MAX trajectory).

The checkpoint's embedded `selected_threshold = 0.05` is a separate training-time
artefact from the single-day split; `src/app.py` and `src/shap_explainer.py` use it,
the reference evaluation does not. `models/latent_world_model_v4/validation_threshold_search.csv`
is likewise a training-time file (its best row is threshold 0.05, F1 0.8043 on the Feb-28
validation split) and is **not** the source of the thresholds above.

---

## 9. Exact rollout protocol

| Setting | Value |
|---|---|
| Max rollout horizon | K = 5 |
| Horizons evaluated (point) | **1, 2, 3, 5** — K=4 is *not* evaluated |
| Horizons evaluated (trajectory) | 2, 3, 5 |
| Rollout implementation | `LatentWorldModelV4.k_step_rollout` — autoregressive in latent space; each step: `transition_step` → attack head + decoder |
| Target alignment (point) | `probs[:, k]` (k transitions) vs `A_binary[:, k−1]` (label at t+k) |
| Trajectory target | `max(A_binary[:, 0:k])` — any attack within t+1 … t+k |
| Trajectory score (primary) | **MAX** over `probs[:, 1:k+1]` |
| Trajectory score (secondary) | MEAN over `probs[:, 1:k+1]` — labelled a heuristic |
| Attack label definition | `attack_present` of the target window: 1 if any flow in that 10 s window is non-Benign |
| Early warning | alert if `max(probs[:, 1:6]) ≥ 0.69`; lead time = `(onset − first alert) × 10 s`, searched only in the 5 windows before onset |
| Batch size | 64 |
| Device | MPS → CUDA → CPU (verified on MPS, macOS arm64) |
| Model mode | `model.eval()` + `torch.no_grad()` (dropout off) |
| Seeds | none required — evaluation is deterministic (no shuffling/sampling); training used seed 42 |
| Filtering | none at evaluation time; only 10 s-contiguous sequences exist in the arrays |

---

## 10. Exact evaluation command

```bash
# from the repository root
git show 2be3073:src/k_step_rollout.py > /tmp/k_step_rollout_reference.py
python /tmp/k_step_rollout_reference.py
```

> **Why the commit pin.** `src/k_step_rollout.py` at HEAD (`53c2222`) reads
> `data/processed/temporal_benchmark_v2/multistep_K5` — the single-day Feb-28 dataset —
> and produces K=1 F1 = 0.3562, *not* the reported reference numbers. The reference
> results come from the same file at commit `2be3073`, which reads
> `data/processed/multiday_benchmark_K5`. Both were run during this freeze; only the
> `2be3073` version reproduces the reported figures, exactly. See §12.

Baselines for later comparison (already multi-day): `python src/evaluate_baselines.py`.

---

## 11. Exact current results (re-verified 2026-09-19)

Held-out test day **Mar-2-2018**, 3,115 sequences, thresholds frozen on validation.

### Future attack prediction (prediction at t+k vs actual A(t+k))

| Horizon | F1 | Precision | Recall | ROC-AUC | FPR | Threshold | TP / FP / TN / FN | Pos |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| t+1 | 0.5920 | 0.9410 | 0.4318 | 0.8558 | 0.0507 | 0.45 | 877 / 55 / 1029 / 1154 | 2031 |
| t+2 | 0.6739 | 0.9507 | 0.5219 | 0.8774 | 0.0507 | 0.54 | 1060 / 55 / 1029 / 971 | 2031 |
| t+3 | 0.6952 | 0.9609 | 0.5446 | 0.8893 | 0.0415 | 0.65 | 1106 / 45 / 1039 / 925 | 2031 |
| t+5 | 0.7523 | 0.9660 | 0.6160 | 0.8877 | 0.0406 | 0.72 | 1251 / 44 / 1040 / 780 | 2031 |

### Trajectory-event prediction (any attack in t+1 … t+k)

| Strategy | K | F1 | Precision | Recall | ROC-AUC | FPR | Threshold | Pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **MAX** | 2 | 0.7127 | 0.9554 | 0.5683 | 0.8780 | 0.0500 | 0.51 | 2034 |
| **MAX** | 3 | 0.6371 | 0.9661 | 0.4752 | 0.8904 | 0.0315 | 0.68 | 2037 |
| **MAX** | 5 | **0.7979** | **0.9665** | **0.6794** | **0.8929** | 0.0447 | 0.69 | 2040 |
| MEAN | 2 | 0.6294 | 0.9485 | 0.4710 | 0.8747 | 0.0481 | 0.49 | 2034 |
| MEAN | 3 | 0.6426 | 0.9536 | 0.4845 | 0.8832 | 0.0445 | 0.55 | 2037 |
| MEAN | 5 | 0.6439 | 0.9602 | 0.4843 | 0.8866 | 0.0381 | 0.62 | 2040 |

### Early warning (K=5 MAX, threshold 0.69)

| Metric | Value |
|---|---|
| Observable onsets | 3 (a 4th episode is already active at test start and is excluded) |
| Episodes warned early | 2 → 66.7% coverage |
| Mean / median lead time | 50.0 s / 50.0 s |
| False-positive prediction windows per hour | 5.55 |

**Agreement with the previously reported figures:** exact, to 4 decimal places, for
every K=1/2/3/5 point metric and for MAX K=5. Two clarifications:

- The previously circulated K=3 "trajectory" figure (F1 ≈ 0.695) is the **point**
  t+3 number, not the MAX-trajectory K=3 number (which is 0.6371).
- No K=4 figure exists; the reference script never evaluates it.

---

## 12. Known limitations / issues

1. **Reference evaluation lives at a past commit (critical).** HEAD's
   `src/k_step_rollout.py` evaluates on the single-day `temporal_benchmark_v2/multistep_K5`
   split and yields K=1 F1 = 0.3562 / K=5 MAX F1 = 0.3717 — a completely different,
   much weaker result set. Anyone running `python src/k_step_rollout.py` today will not
   see the reported numbers. Documented, not changed.
2. **Train/eval normalisation mismatch.** The checkpoint was *trained* on
   `temporal_benchmark_v2/multistep_K5` (Feb-28 only; scaler fit on 2,384 windows) but the
   reference evaluation feeds it `multiday_benchmark_K5` arrays scaled by a *different*
   `StandardScaler` (8,575 windows, Feb-14/15/28). The reference results are therefore
   produced under a normalisation the model never saw in training. Recorded as-is.
3. **Training is not bit-reproducible.** The checkpoint's `positive_class_weight = 5.807471`
   implies 2,021 benign / 348 attack training sequences, and its saved confusion matrix sums
   to 497 test sequences; the arrays now on disk hold 2,022 / 348 and 498. The training arrays
   were regenerated on 2026-09-04, after the 2026-09-02 checkpoint. **Evaluation is exactly
   reproducible; training is not.**
4. **Test-label threshold tuning inside the training script.** `src/train_latent_world_model_v4.py`
   calls `find_best_threshold(all_max_probs, y_test_binary)` in its K-step trajectory section —
   those trajectory numbers are test-optimised and must never be quoted. Flagged, **not fixed**,
   per the freeze scope. The reference rollout evaluation does *not* do this: every threshold
   there is selected on validation only.
5. **Swapped columns in `trajectory_evaluation.csv`.** The training script unpacks
   `find_best_threshold(...)[:2]` as `(f1, threshold)` but the function returns
   `(threshold, f1)`. In `models/latent_world_model_v4/trajectory_evaluation.csv`, the column
   `trajectory_best_f1` actually holds the threshold and `trajectory_threshold` holds the F1.
6. **Training-time artefacts are not benchmark results.** `metrics.csv` (test F1 0.2967),
   `confusion_matrix.csv`, `test_attack_predictions.csv` and `validation_threshold_search.csv`
   under `models/latent_world_model_v4/` all describe the Feb-28 intra-day split.
7. **Dashboard / explainer use the training threshold.** `src/app.py` and
   `src/shap_explainer.py` read `checkpoint["selected_threshold"] = 0.05` and
   `data/processed/temporal_benchmark_v2`, not the validation-frozen rollout thresholds and
   not the multi-day benchmark. Out of scope for this freeze.
8. **No K=4 reference number** (§9).
9. **Checkpoint and data are untracked.** `data/*` and `models/*` are `.gitignore`'d;
   the SHA256 hashes here and in the manifest are the sole provenance record.

### V3 contamination check — core benchmark is clean

| Check | Result |
|---|---|
| Reference script imports project modules | No — only stdlib / numpy / pandas / torch / sklearn |
| V3 checkpoint used | No — only `latent_world_model_v4.pt`, loaded `strict=True` |
| V3 feature schema / scaler / threshold used | No — schema and scalers come from `multiday_benchmark_K5` |
| V3 evaluation artefact used | No |
| Benchmark data provenance | `states_v2` → `prepare_multiday_benchmark.py` only |

V3 dependencies exist elsewhere in the repo and are **outside** the frozen benchmark:
`src/shap_explainer.py` (`--version v3` loads `latent_world_model_v3.pt`; its v4 path reads
`temporal_benchmark_v2`), `src/mitre_mapper.py` (loads `latent_world_model_v3.pt`), and
`src/pcap_ingest.py` (falls back to `models/latent_world_model_v3`). None of them is imported
by `k_step_rollout.py` or `evaluate_baselines.py`. The known V3/V4 explainability issue is
recorded here and deliberately left unfixed.

---

## 13. Git and artifact hashes

| Item | Value |
|---|---|
| Freeze commit | `53c22227d368b82ff9e736206a92e85b6bed2514` |
| Branch | `cicflowmeter-validation` |
| Commit date | 2026-09-15 00:34:58 +0530 |
| Worktree at freeze | clean |
| Reference-evaluation commit | `2be30736872dd7bb072b2e5c4cf7e7fe62605af6` (2026-09-04 20:20:37 +0530) |

| Artifact | SHA256 |
|---|---|
| `models/latent_world_model_v4/latent_world_model_v4.pt` | `76af8d8a9e02215923b1c0ab6b627bfc0809e856748666a55ee4b16cec5e5898` |
| `…/multiday_benchmark_K5/input_feature_names.txt` | `137b58333f1c751522f30478219ff86b8c115aa749defb57ae80ee83559ecbb5` |
| `…/multiday_benchmark_K5/state_feature_names.txt` | `64a8297798779b3616beeda307fef4679e67cfbd33e395a1059fb502a48b5499` |
| `…/multiday_benchmark_K5/input_scaler.pkl` | `e0ab9bebd26869950dda51a67bdde3192b0edbd00e20b623115ed2b653f04f19` |
| `…/multiday_benchmark_K5/target_scaler.pkl` | `7155dec48f89d246359741032b002ca0bb44e4739e99daa3e28e77e312c55944` |
| `…/multiday_benchmark_K5/X_test.npy` | `2a03a11ca58c38bf4803546e7e5e7db50c44bf69d1c2ad5764f3b905a4005bc8` |
| `…/multiday_benchmark_K5/X_val.npy` | `083478c02131e568731923761b4df1595cfbdc25d28afd78583613c30e8d8b9f` |
| `…/multiday_benchmark_K5/X_train.npy` | `24139069b9c1f35fdd8a5183d7fcf37fb82a3721bc4f6760be2fbbc0918f7065` |
| `…/multiday_benchmark_K5/A_binary_multistep_test.npy` | `4b18a65902c18fbd1b23fa4ca431f82cd2dac43f06c8636e6267bd8ee63134c3` |
| `…/multiday_benchmark_K5/A_binary_multistep_val.npy` | `db34c8c5503208a2e5689c533b046ea2fb6f4a8ee60fabd23532a17b18f9192b` |
| `…/states_v2/Friday-02-03-2018_…_states_v2.csv` | `ce101327abfe64fe8f7175a961f72a60cb56bda6e5216a023077800a33cd20b5` |

The complete hash set (every array, every metadata CSV, every source state file) is in
`docs/v4_baseline_manifest.json`.

Environment used for verification: Python 3.14.7, torch 2.14.0, scikit-learn 1.9.0,
numpy 2.5.2, pandas 3.0.5, macOS arm64 (MPS).

---

## 14. How to reproduce the evaluation

```bash
cd /path/to/cyber-world-model
source venv/bin/activate

# 1. Confirm the repository still matches the frozen baseline
python src/verify_v4_baseline.py

# 2. Run the reference V4 evaluation (commit-pinned — see §10)
git show 2be3073:src/k_step_rollout.py > /tmp/k_step_rollout_reference.py
python /tmp/k_step_rollout_reference.py

# 3. Compare the printed table against §11
```

If `src/verify_v4_baseline.py` reports **PASS**, the checkpoint, feature schema, scalers
and benchmark arrays are byte-identical to the frozen baseline and step 2 will reproduce
§11 exactly. If it reports FAIL, the baseline has drifted and the reported V4 numbers no
longer describe what is in the repository — restore the artifacts before comparing any
future model against V4.

**V4 must not be retrained, re-scaled, re-thresholded or re-split.** Future work
(persistence baseline, logistic regression, improved world models) compares against
*this* frozen specification.
