# V4 Reliability / Model-Health Layer

**Task 15.** A decision-support layer on top of the **frozen** V4 world model that
reports when a forecast should not be trusted.

The V4 model is unchanged by this task: not retrained, not fine-tuned, no scaler
refit, no threshold moved, no feature definition touched, and no new learned
model introduced. All 16 frozen artifacts are SHA256-hashed before and after
every run and must be byte-identical.

## 1. What this is, and what it is not

The layer emits one of three statuses — `RELIABLE`, `CAUTION`, `UNRELIABLE` —
from a transparent arithmetic rule over a fixed reference distribution.

**It is not** a calibrated probability, a confidence score, a certainty, or a
probability of correctness. No reliability estimator was fitted and no
calibration was attempted or established. The status describes the *conditions
the forecast was produced under*, not the chance the forecast is right.

The dashboard wording is `HIGH / CAUTION / LOW`, chosen precisely because it
cannot be read as a probability.

## 2. Threshold provenance

Every distributional threshold is a quantile of
`data/processed/multiday_benchmark_K5/X_train.npy` — the train partition the
frozen `input_scaler` was fit on — pushed through the frozen checkpoint.

**No label array is opened while the reference is built.** This is enforced, not
asserted: every `.npy` the build process reads passes through a guarded loader
that records the path and phase into
`reliability_manifest.json → arrays_read_this_run`. The recorded count of label
arrays read during the reference build is **0**, and
`A_binary_multistep` / `A_soft_multistep` appear nowhere in
`src/v4_reliability.py`.

The one enabling fact that makes this possible without training: `X_train.npy`,
`X_val.npy` and `X_test.npy` are **already standardized** by the frozen
`input_scaler`, so a stored value *is* a z-score against the reference
distribution. Input-OOD indicators need no new fit and no new scaler.

## 3. Indicators

Twelve indicators in three groups. Eleven are tiered against the reference
distribution; `feature_support_exceedance` is handled as hard condition **H8**
instead, because in the reference split it is identically 0 by construction, so
tiering it would double-count the same evidence.

### A · Input OOD / distance (frozen V4 input scaler)

| Indicator | Definition |
|---|---|
| `rms_z` | overall standardized magnitude: RMS of all 10 × 52 scaled values |
| `max_abs_z` | largest single \|z\| anywhere in the input tensor |
| `frac_gt2` | fraction of the 520 values with \|z\| > 2 |
| `frac_gt3` | fraction of the 520 values with \|z\| > 3 |
| `feature_support_exceedance` | distance, in training SD, by which the most extreme value falls **outside** the per-feature min/max ever observed in the reference split |

### B · Rollout stability

`latent_norm_max`, `latent_norm_change` (max‖z_k‖ − ‖z₀‖), `step_delta_max`
(largest ‖z_k − z_{k−1}‖), `displacement` (‖z₅ − z₀‖). Non-finite latents are a
separate hard condition (H4).

### C · Forecast self-consistency

`state_delta_max` (sharpest change in the decoded 26-D state),
`prob_max_abs_step` (sharpest single-step probability jump), `prob_range`.

## 4. Status rule

| Status | Condition |
|---|---|
| **UNRELIABLE** | any hard fail-safe H1–H8 fires, **or** ≥ 2 indicators exceed the reference **maximum** |
| **CAUTION** | ≥ 1 indicator exceeds the reference maximum, **or** ≥ 2 indicators exceed the reference **p99** |
| **RELIABLE** | otherwise |

Every status is reconstructable from the emitted `reasons` list. There is no
learned component and no hidden weighting.

### The rule's base rate

An `ELEVATED` tier is a ~1 % event per indicator *by construction*, so the
held-out rate is only interpretable against the rate on the reference itself:

| Split | n | RELIABLE | CAUTION | UNRELIABLE |
|---|---|---|---|---|
| train (the reference) | 8,441 | 97.70 % | 2.30 % | 0 |
| val (label-free, second reference) | 1,473 | 98.30 % | 1.70 % | 0 |
| **screening (held-out Mar-2)** | **3,115** | **91.17 %** | **8.48 %** | **0.35 % (11)** |

The rule is neither vacuous nor trivially always-on.

## 5. Fail-safe conditions

| Code | Condition |
|---|---|
| H1 | non-finite model input (NaN / Inf) |
| H2 | invalid input, latent, state or probability tensor shape |
| H3 | input scaler incompatible with the frozen contract |
| H4 | non-finite latent state or decoded state |
| H5 | non-finite or out-of-\[0,1\] threat probability |
| H6 | ‖z₀‖ > **11.7194** — the analytic encoder bound |
| H7 | ‖z_k‖ > **7.1089** (k ≥ 1) — the analytic LayerNorm transition bound |
| H8 | an input value lies > 3 training SD outside the reference range for its feature |

Any hard condition forces `UNRELIABLE` unconditionally.

### H6 / H7 are derived, not fitted

The V4 transition ends in `LayerNorm`, which makes its output `w ⊙ n + b` with
‖n‖₂ ≤ √32, so for k ≥ 1:

```
‖z_k‖ ≤ ‖w‖∞ · √32 + ‖b‖₂ = 1.201775 · 5.65685 + 0.310654 = 7.1089
```

and through the encoder path:

```
‖z₀‖ ≤ σ_max(W_fc) · (‖w_enc‖∞·√128 + ‖b_enc‖₂) + ‖b_fc‖₂
     = 0.938863 · 12.135894 + 0.325436 = 11.7194
```

Exceeding either bound is **numerically impossible for an intact checkpoint**.
H6 and H7 are therefore *corruption* detectors, not instability detectors — a
distinction `docs/LONG_HORIZON_ROLLOUT.md` §8 already establishes: bounded ‖z‖
is strong evidence about the architecture and weak evidence about the learned
dynamics.

### The H8 margin

3.0 training SD, on two independent label-free grounds:

- **(a)** The condition must not fire on data the reference covers. A margin of
  0 does — 10 val sequences slightly extend the train range, as any finite
  sample will. The smallest margin in {0, 1, 3, 5} that flags neither train nor
  val is **1.0 SD**, so 3.0 satisfies this with room to spare.
- **(b)** 3.0 is not a new free parameter: `|z| > 3` is already one of this
  layer's indicators (`frac_gt3`), so the hard condition reuses the same
  conventional outlier threshold.

**Honest caveat:** 3.0 is the more conservative of two defensible values, not
the unique value the evidence forces. At 1.0 SD the same rule flags 21 held-out
sequences instead of 11. Both counts are recorded in
`reliability_summary.json → input_shift_margin_sensitivity`.

## 6. Real failure cases

All from the real held-out capture. **Zero synthetic examples.** Full table in
`results/v4_reliability/failure_cases.csv` (22 cases).

### UNRELIABLE — 11 sequences, 08:50:30 → 08:52:10, all labelled Benign

A single physical anomaly drives all eleven: `mean_pkt_len` reaches
**z = 12.78** against a training maximum of **8.62** — **4.16 SD outside any
value ever observed for that feature**. H8 fires. As the 10-window lookback
slides forward, that one anomalous window moves from position 9 down to
position 0, and the frozen model's response swings wildly across essentially
the same evidence:

| seq | time | window holding the anomaly | max forecast P | frozen alert | retrospective state MAE |
|---|---|---|---|---|---|
| 1622 | 08:50:30 | 9 (most recent) | **0.746** | **ALERT** | 1.064 |
| 1623 | 08:50:40 | 8 | **0.692** | **ALERT** | 0.769 |
| 1624 | 08:50:50 | 7 | 0.231 | no-alert | 1.064 |
| 1626 | 08:51:10 | 5 | 0.606 | no-alert | 0.985 |
| 1629 | 08:51:40 | 2 | 0.000022 | no-alert | 1.051 |
| 1632 | 08:52:10 | 0 (oldest) | 0.006 | no-alert | 1.099 |

Two of these are **false ALERTs on benign traffic** at a frozen threshold of
0.69. The reliability layer flags all eleven, having seen no label.

Mean retrospective state MAE across the group is **1.0656** against **0.7116**
for the rest of the day.

### CAUTION — 264 sequences; 2 exceed the reference maximum outright

- **seq 2872** (12:18:50, attack-active): `displacement` = 9.936 vs reference
  max 9.8595 — the only sequence on the day whose latent travels further than
  anything in the reference.
- **seq 2357** (10:53:00, attack-active): `prob_range` = 0.7611 vs reference
  max 0.7579.

The most common CAUTION signature is four elevated indicators at once —
`displacement`, `state_delta_max`, `prob_max_abs_step`, `prob_range` — i.e. the
input looks ordinary but the *rollout* does not.

### RELIABLE — 2,840 sequences

Includes the default demo sequence (2341, the highest max forecast probability
on the day) at 0 elevated and 0 beyond-reference indicators.

**All three categories are demonstrated on real data.** No category required
fabrication.

## 7. Retrospective forecast error analysis

Explicitly labelled **retrospective**. `Y_multistep_test.npy` holds the observed
future state at t+1…t+5 in the same `target_scaler` space the decoder writes
into. It is opened *after* every forecast and every status already exists, and
it never enters either — `assess(x, probs, latents, states, ref, scaler)` takes
no observed-future argument, and the build verifies that perturbing the observed
future moves no status.

### State reconstruction error by horizon

| k | mean MAE | median | p90 | max | mean RMSE |
|---|---|---|---|---|---|
| 1 | 0.7768 | 0.7255 | 1.1638 | 4.2764 | 1.0187 |
| 2 | 0.7139 | 0.6765 | 1.0180 | 4.3498 | 0.9432 |
| 3 | 0.6887 | 0.6651 | 0.9497 | 4.4268 | 0.9182 |
| 4 | 0.6861 | 0.6585 | 0.9448 | 4.4204 | 0.9239 |
| 5 | 0.6987 | 0.6634 | 0.9771 | 4.4014 | 0.9428 |

Error does not grow with horizon — it *falls*, then flattens. As
`docs/LONG_HORIZON_ROLLOUT.md` §11 records, that is not a success: the decoded
state contracts toward a near-constant output, so error stops growing because
the prediction stops varying, and V4 does not beat a persistence baseline at
these horizons.

### Error by reliability status

| status | n | mean MAE | median | p90 |
|---|---|---|---|---|
| RELIABLE | 2,840 | **0.7025** | 0.6744 | 0.9167 |
| CAUTION | 264 | **0.8093** | 0.7813 | 1.0483 |
| UNRELIABLE | 11 | **1.0656** | 1.0642 | 1.1659 |

**Mean retrospective state error rises monotonically with status severity.**

### Indicator vs error (Spearman ρ, sequence-mean state MAE over k=1…5)

| indicator | ρ |
|---|---|
| `state_delta_max` | **+0.481** |
| `frac_gt3` | +0.453 |
| `displacement` | +0.448 |
| `prob_max_abs_step` | +0.443 |
| `rms_z` | +0.427 |
| `prob_range` | +0.421 |
| `frac_gt2` | +0.351 |
| `max_abs_z` | +0.344 |
| `step_delta_max` | +0.263 |
| `latent_norm_change` | +0.166 |
| `feature_support_exceedance` | +0.123 |
| `latent_norm_max` | **−0.330** |

These correlations are **reported, not used** — no threshold was chosen from
them. `latent_norm_max` correlates *negatively*, contradicting its
"larger is worse" framing; that is recorded as a limitation rather than
corrected, and it is consistent with the LayerNorm bound making ‖z‖ a weak
signal about the learned dynamics.

## 8. Validation

`results/v4_reliability/validation.json` — **23 / 23 gates PASS**.

| Claim | Result |
|---|---|
| frozen checkpoint unchanged | PASS — 16 artifacts hashed before/after, 0 changed |
| scalers unchanged | PASS — input + target, manifest-matched |
| screening predictions unchanged | PASS |
| thresholds unchanged | PASS — alert threshold 0.69 |
| no test labels used to tune reliability rules | PASS — 0 label arrays read, 0 module references |
| no future observation enters model input | PASS |
| hard numerical failures detected | PASS — 11/11 self-tests |
| reliability outputs deterministic | PASS — bit-identical on rerun |
| no synthetic data | PASS — `screening_X.npy` byte-identical to `X_test.npy` |
| dashboard page loads | PASS |
| existing V4 pages still work | PASS — 12/12 render probes |
| offline operation preserved | PASS — 0 network constructs |

### Fail-safe detector self-tests

Every hard condition is exercised on a deliberately corrupted tensor, and every
one is **detected and forced to `UNRELIABLE`**. The intact baseline sequence
passes all eight, so the detectors are not trivially always-on.

These are **detector self-tests, not data examples**. Corrupted tensors never
enter `failure_cases.csv`, `reliability_per_sequence.csv`, or any reported
statistic.

### Dashboard render probes

All ten pages are forced and rendered headless under the dashboard interpreter
(`src/venv`, Streamlit 1.62.0), plus the new page and Live Forecast at a real
`UNRELIABLE` sequence (1622) so the abstention path is exercised rather than
merely written. `st.stop()` is honoured in the probe, so a page that stops at
its own guard — the MITRE page, whose stored mapping table predates the current
window set — counts as a correct render rather than a false failure.

### Reproduction gap vs the frozen CSV

Max \|ΔP\| between the recomputed rollout and `screening_predictions.csv` is
**2.00 × 10⁻⁶**, and **all 3,115 alert decisions are identical**. Cause:
`screening_predictions.csv` was produced on MPS; the reliability layer pins CPU
so its output is deterministic and device-independent. This is float32 device
arithmetic, not a model change.

## 9. Dashboard

**🛡️ Reliability / Model Health** — status headline, every triggering indicator
with its reference p99 / max, the per-window \|z\| profile across the 10-window
lookback, latent-norm and step-delta trajectories against both the reference
maximum and the analytic H7 bound, the per-step consistency table, the
day-level status timeline (broken at real capture gaps), and the retrospective
error tables.

On `UNRELIABLE` the page shows a prominent **⛔ MODEL ABSTENTION RECOMMENDED**
call-out: *"Forecast should not be treated as reliable for this window."* The
forecast is still displayed, unaltered, because hiding it would hide the
evidence.

**🎯 Live Forecast** carries a compact `Forecast reliability: HIGH / CAUTION /
LOW` chip with an expander listing the contributing reasons, and the abstention
message when `UNRELIABLE`.

Converting the status into an automatic ALERT / BENIGN / ABSTAIN decision is
**Task 16** and is deliberately not done here. This page is the reliability
*mechanism* only.

## 10. Known limitations

1. **Rule-based, not calibrated.** No calibration was attempted or established.
2. **The reference is the scaler's distribution, not the weights'.**
   `docs/v4_baseline_manifest.json → known_issues` records that the checkpoint
   was trained on the single-day `temporal_benchmark_v2` arrays while the
   reference evaluation feeds it `multiday_benchmark_K5` arrays scaled by a
   *different* `StandardScaler`, and that the exact training arrays no longer
   exist on disk. This layer measures distance from the distribution the
   inference path's scaler defines. The train/eval normalisation mismatch is
   **inherited, not repaired**.
3. **One held-out day** (Friday-02-03-2018, 3,115 sequences, 3 observable
   onsets). Every held-out rate is a single-day rate.
4. **Latent-norm stability is largely architectural.** LayerNorm bounds the
   latent RMS by construction, so H7 is a corruption detector and
   `latent_norm_max` correlates negatively with retrospective error.
5. **`ELEVATED` is a ~1 % per-indicator event by construction**, so the CAUTION
   rate is a property of the quantile choice as much as of the data.
6. **The retrospective study measures state reconstruction error, not
   attack-label correctness.** It shows the layer separates conditions under
   which the decoder is worse. It does **not** show the layer predicts label
   errors.
7. **11 UNRELIABLE sequences is too few to estimate any rate**, and they are
   one physical event observed through a sliding lookback — not 11 independent
   samples.
8. **No abstention is executed.** This layer reports; Task 16 decides.

## 11. Does this satisfy "the world model can fail"?

**Yes, in the weaker and defensible form.** The supported claim is:

> The reliability layer identifies conditions associated with unreliable
> forecasts.

Supporting evidence: mean retrospective state error rises monotonically
`RELIABLE → CAUTION → UNRELIABLE` (0.7025 → 0.8093 → 1.0656); eleven real
sequences are flagged `UNRELIABLE` from the input alone, without any label, and
include two false ALERTs on benign traffic where the frozen model's forecast
probability swings from 0.746 to 0.000022 on essentially the same anomaly; and
every hard numerical failure is detected and forced to `UNRELIABLE`.

**Not claimed:** that the model has proven it can predict when it is wrong. That
would require showing the layer anticipates specific label errors, on more than
one day, with enough positive events to estimate a rate. None of those
conditions is met.

---

### Artefacts

```
src/v4_reliability.py                              the layer
src/build_v4_reliability.py                        build + evidence pack + validation
results/v4_reliability/reliability_config.json     reference distribution + derived bounds
results/v4_reliability/reliability_per_sequence.csv    3,115 rows
results/v4_reliability/reliability_summary.json    distributions, rule rates, correlations
results/v4_reliability/failure_cases.csv           22 real cases
results/v4_reliability/forecast_vs_observed.csv    15,575 rows (3,115 x 5)
results/v4_reliability/reliability_manifest.json   provenance
results/v4_reliability/validation.json             23 gates
```

Rebuild with:

```bash
./venv/bin/python src/build_v4_reliability.py
```
