# Judge Language — V4 Screening Console

Answers to the eighteen questions most likely to be asked, using **only** verified project
evidence: the frozen manifests (`docs/v4_baseline_manifest.json`,
`results/v4_screening_mvp/screening_manifest.json`), the analysis documents under `docs/`,
and values read off the running dashboard during browser QA.

Each answer separates four different kinds of claim. Keep them separate when you speak:

- **Measured** — a number we produced and can reproduce.
- **Architecture** — how the system is built. Not evidence of performance.
- **Interpretation** — what we think it means. Argued, not proven.
- **Limitation** — what it does not show.

Nothing here is rounded in our favour. Where the evidence is against us, it says so.

---

## 1. What exactly does the model predict?

**Architecture.** For each 10-second window *t*, V4 emits two things per future step
*k* = 1…5: a **26-dimensional decoded network state** and a **scalar attack probability**.
Five steps at 10-second resolution is a 50-second horizon.

**Measured.** On the held-out day, the trajectory-MAX K=5 signal that the console headlines
scores **F1 0.798, ROC-AUC 0.893, precision 0.967, recall 0.679** at the frozen threshold
0.69.

**Limitation.** It does not predict an attack *type*, an attacker, a target host, or an
ATT&CK label. It predicts a state vector and a probability, nothing else.

---

## 2. Why is this a World Model?

**Architecture.** A classifier maps an observed window to one label. V4 encodes 100 s of
history into a 32-dimensional latent state, then applies a **learned gated residual
transition** to that latent repeatedly. The *same* state decoder and the *same* threat head
read the latent at every step. The model rolls a state forward and reads it, rather than
emitting one classification per input.

**Interpretation.** That is what earns the term: there is an internal state with learned
dynamics that can be advanced without new observations, and the state is decodable back to
physical network quantities — not just to a label.

**Limitation.** The transition is **deterministic**. V4 learns a point estimate in latent
space, not a distribution P(S_t+1 | S_t), so there is no predictive interval. And see Q16:
the learned transition contracts toward a fixed point, which materially limits what the
rollout actually forecasts.

---

## 3. What is the input?

**Architecture.** A `10 × 52` tensor. Ten consecutive 10-second windows (100 seconds of
history), each described by **52 features = 26 base network-state features + their 26
relative-change counterparts**. Features are z-score normalised with a frozen scaler
(`data/processed/multiday_benchmark_K5/input_scaler.pkl`) in a frozen order
(`results/v4_screening_mvp/feature_names.txt`).

**Limitation.** Labels, event regions and ground-truth fields are **not** in the input.
Neither is anything packet-level — every one of the 52 features is an aggregate over
CICFlowMeter flow records (see Q17).

---

## 4. Why 10 historical windows?

**Architecture.** `sequence_length = 10` is fixed in the checkpoint metadata: 100 seconds at
10-second resolution. It is the amount of context the encoder LSTM consumes before the
latent is rolled forward.

**Limitation — answer this one carefully.** We have **no published ablation over history
length.** 10 windows is a design choice frozen at training time, not an empirically selected
optimum. If a judge asks "why not 5, or 20?", the honest answer is that we did not sweep it,
and the checkpoint cannot be re-trained without breaking the frozen baseline. Do not invent
a justification.

---

## 5. How far ahead do you forecast?

**Architecture.** K = 5 transitions = **50 seconds**.

**Measured.** The frozen reference evaluation covers horizons **1, 2, 3 and 5**. A separate
long-horizon study extended the rollout to K=10 and, in an auxiliary diagnostic, to K=40:
the rollout is numerically well-behaved that far — bounded magnitude, shrinking steps, no
oscillation or collapse.

**Limitation.** **K=4 has no frozen reference number.** The dashboard displays +40 s for
continuity and says so on screen. And "numerically stable to K=10" is not "useful to K=10" —
the long-horizon study concludes that **K=5 should be treated as the practical operating
horizon, and as the deepest *validated* horizon rather than as a demonstrated 50-second
forecasting capability.**

---

## 6. How is the model trained?

**Architecture.** 244,027 parameters. A 2-layer LSTM encoder (52 → 128), a 32-dimensional
latent, a gated residual transition with LayerNorm, a 26-D state decoder and a scalar threat
head. Trained with a **multi-step objective** under a **curriculum** that deepens the rollout
during training — K=1 from epoch 1, then 2 at epoch 21, 3 at 41, 4 at 61, 5 at 81 — with
discount γ = 0.8 and a positive-class weight of 5.807.

**Limitation.** **Training is not bit-reproducible.** The exact arrays that produced the
checkpoint were regenerated after it was trained and no longer match bit-for-bit.
*Evaluation* is exactly reproducible; *training* is not. This is recorded in the frozen
manifest, not discovered late.

---

## 7. How was test leakage avoided?

**Measured.**

- **Train:** Feb-14, Feb-15 and Feb-28 2018 — 8,441 sequences, the 85% chronological head of each day.
- **Validation:** the 15% chronological tail of those same days — 1,473 sequences.
- **Held-out test:** **March 2, 2018 only** — 3,115 sequences.
- **March 2 contributes 0 windows to any scaler or model fit** — recorded in the manifest as `mar2_in_any_fit = False`.
- **Every threshold was selected on validation only** (grid search maximising validation F1) and applied **unchanged** to the test day; `test_labels_used_for_selection = False`.
- Splits are **chronological**, not random, so no future window leaks into a past split.
- 14 candidate sequence blocks were **rejected for straddling the capture gap**, matching the 14 predicted from the gap position. No sequence spans it.

**Interpretation.** The separation is by day and by clock, and the threshold — the one number
a demo is most tempted to tune — was frozen before the test day was touched.

---

## 8. What is the evaluated F1?

**Measured**, on held-out March 2, at the frozen validation-selected thresholds:

| Framing | K | Threshold | F1 | ROC-AUC |
|---|---|---|---|---|
| **Trajectory-MAX** (what the console headlines) | 5 | 0.69 | **0.798** | 0.893 |
| Trajectory-MAX | 3 | 0.68 | 0.637 | 0.890 |
| Trajectory-MAX | 2 | 0.51 | 0.713 | 0.878 |
| Future-prediction | 5 | 0.72 | 0.752 | 0.888 |
| Future-prediction | 3 | 0.65 | 0.695 | 0.889 |
| Future-prediction | 2 | 0.54 | 0.674 | 0.877 |
| Future-prediction | 1 | 0.45 | 0.592 | 0.856 |

At trajectory-MAX K=5: precision 0.967, recall 0.679, FPR 0.045, on 2,040 positive samples.

**Limitation.** Quote the framing along with the number. "F1 0.798" alone is not meaningful
here — the value depends on which of the two framings and which horizon, and every row above
uses a *separately chosen* threshold. Also read Q9 before treating any of these as a headline
success.

---

## 9. Why is the persistence baseline so high?

**Measured.** Predicting "the next window looks like the current one" achieves
**F1 ≈ 0.9966 at K=5** on this day. **V4 did not exceed persistence at any evaluated
horizon** — the shortfall ranges from **−0.1994 to −0.4066 F1**. The reason is label inertia:
attack episodes are long relative to the 10-second window, and **only 0.77% of windows change
label within 100 seconds**.

**Interpretation.** Ordinary t+k classification on this dataset is close to saturated by
persistence, so aggregate F1 and ROC-AUC across horizons mostly measure label inertia rather
than forecasting skill. Persistence is additionally *given* the observed label A(t), which
neither V4 nor the Logistic Regression baseline receives. Where a forecast could earn its
keep is the **pre-attack transition region**, where the persistence rule is uninformative by
construction — which is exactly where the early-warning evaluation is measured, and exactly
where the sample is smallest.

**Limitation.** State this plainly rather than defending it: **the evidence does not
establish that the temporal World Model adds value beyond persistence on the t+k
classification task.** The console does not claim otherwise, and the Methodology page says so
on screen.

---

## 10. How early can you warn?

**Measured**, at the frozen threshold 0.69 on the held-out day:

| | Strict convention (headline) | Published reference convention |
|---|---|---|
| Onset coverage | **66.7%** (2 of 3) | 66.7% (2 of 3) |
| Mean lead | **15 s** | 50 s |
| Median lead | 15 s | 50 s |
| Evaluable onsets | 3 | 3 |
| False-alert rate | 5.55 windows/h | 5.55 windows/h |

Coverage is **identical** under both conventions; only lead time differs.

**Limitation.** Three onsets on one held-out day. The manifest's own wording: *coverage and
lead time are indicative, not statistically established.* A fourth attack episode is already
active at the first evaluable window of the day, has no observable onset, and is excluded
from early-warning analysis entirely — though it is still counted in the classification
metrics in Q8.

---

## 11. What exactly does "15 seconds" mean?

**Measured.** Lead time is `(onset_index − first_alert_index) × 10 seconds`, searched **only
within the 5 windows before onset**. The **strict** convention counts a window as a creditable
pre-attack warning only if it is benign at **both** the observation time *t* **and** the
evaluated target *t+1*.

Per episode:

| Onset | Timestamp | Eligible pre-attack windows | Warned | Strict lead |
|---|---|---|---|---|
| #1039 | 2018-03-02 03:54:50 | 2 | yes | **20 s** |
| #2119 | 2018-03-02 10:13:20 | 5 | no | — |
| #2840 | 2018-03-02 12:13:30 | 1 | yes | **10 s** |

Mean over the two warned episodes = **15 s**.

**Interpretation.** The originally published convention credits the first alerting window in
the 5 before onset, which for 2 of the 3 episodes **still lies inside the preceding active
attack**. That convention reports 50 s. We headline the stricter 15 s because crediting a
warning issued during an already-active attack is not an early warning.

**Limitation.** The strict number rests on two warned episodes. Two.

---

## 12. Does the model predict MITRE ATT&CK?

**No.** Say it that directly.

**Architecture.** V4 outputs a 26-D network state and a threat probability — nothing else.
The tactic and technique shown in the console are produced by `src/mitre_mapper.py`, a
**deterministic 10-rule telemetry library** that reads feature values in z-score space and
matches them against documented ATT&CK-consistent patterns.

**Interpretation.** It is an interpretation layer for analysts, presented as a second,
independent line of evidence about the same forecast. The ATT&CK context is derived from the
observed feature values; the Integrated Gradients attribution is derived from model
gradients. Neither is derived from the other.

**Limitation.** The model was **never trained on ATT&CK labels**, and the rule library has
never been evaluated against ground-truth ATT&CK annotations — there are none in this
dataset. Treat the mapping as a hypothesis for an analyst, not as a classification result.

---

## 13. How does explainability work?

**Architecture.** **Integrated Gradients** (Captum), `gausslegendre` approximation, 1024
steps, against a zero-input baseline. The attribution tensor is `[1, 10, 52]` — one signed
value per feature per historical timestep — and the console sums over the 10 timesteps to
rank features. It runs on the frozen checkpoint, hash-verified against the manifest before
the explainer will even construct.

**Measured.** The explained probability equals the displayed probability. At onset episode
#1039 the explainer reports **0.7794** for P(t+5), which is the displayed forecast, with a
completeness error of 3.79e-03. Top signed contributors there: `rst_rate` +0.373,
`log_unique_dst_ports` +0.209, `dst_port_entropy` +0.155, `mean_down_up_ratio` −0.124.

**Limitation.** Attribution describes **model sensitivity and contribution**. It does not
establish causality, attacker intent, or attack stage. The console's own wording says exactly
this, and you should repeat it rather than paraphrase it upward.

---

## 14. Is the probability calibrated?

**No.**

**Architecture.** Threat values are raw sigmoid outputs thresholded at a frozen,
validation-selected operating point.

**Interpretation.** They are **screening scores, not calibrated likelihoods**. A displayed
0.80 does not mean "80% chance of attack". It means "above the frozen operating point by this
much".

**Limitation.** The console presents **no confidence intervals**, because the model does not
produce them — the transition is deterministic. No calibration study (reliability diagram,
Brier decomposition against a calibrated reference) has been run.

---

## 15. Does the system guarantee an attack?

**No.** A forecast above the threshold is a **screening signal for analyst review** — not a
statement that an attack will occur. That wording is on the Live Forecast page, deliberately.

**Measured**, to put a number on it: at trajectory-MAX K=5 precision is 0.967 and the
false-alert rate is 5.55 alerting windows per hour where no attack follows in t+1…t+5. So it
is wrong sometimes, and we publish how often.

---

## 16. What are the known limitations?

Carried from the frozen manifest and the analysis documents. Lead with these rather than
waiting to be asked.

1. **The rollout may be describing the present, not forecasting the future.** This is the
   project's own central long-horizon finding, and it is the most important item here. The
   learned transition **contracts toward a fixed point**: each additional step moves the
   latent less and the decoded state becomes more nearly constant (across-sample std
   0.488 → 0.257 against a target std of 0.757). The fully-powered measurement —
   **incremental AUC over a pure nowcast — is negative at every horizon** and worsens
   monotonically to −0.0116 at K=10. Restricted to windows where nothing is currently
   happening, AUC sits *below chance* from K=4 onward.
2. **Persistence is not beaten** on t+k classification (Q9).
3. **Train/evaluation normalisation mismatch (out-of-distribution).** The checkpoint was
   trained on single-day `temporal_benchmark_v2` arrays whose scaler was fit on 2,384 Feb-28
   windows. The reference evaluation and this console feed it `multiday_benchmark_K5` arrays
   standardised by a *different* scaler fit on 8,575 Feb-14/15/28 windows. Results are
   produced under a normalisation the model never saw in training. Recorded in the manifest
   and deliberately left unchanged.
4. **Training is not bit-reproducible** (Q6).
5. **One held-out day.** All console results are March 2, 2018. Cross-day generalisation is
   analysed separately (`docs/ROTATED_GENERALIZATION.md`) and is **not** claimed here.
6. **Early-warning sample is 3 onsets** (Q10).
7. **Probabilities are not calibrated** (Q14); the transition is deterministic, so no
   predictive interval exists.
8. **No K=4 reference number** (Q5).
9. **ATT&CK context is rule-based**, never trained, never evaluated against ATT&CK ground
   truth (Q12).
10. **No history-length ablation** (Q4).
11. **A real 3.31-hour capture gap exists** in the day (2018-03-02 05:29:30 → 08:48:10). It is
    never interpolated, no sequence spans it, no synthetic windows are created, and replay
    halts at it rather than bridging it. This is handled correctly — it is listed here because
    it is a property of the data a judge should hear from you first.

**Interpretation of the whole set.** The defensible claim is: *a frozen, leakage-controlled
world-model pipeline that produces an interpretable 50-second forecast trajectory and a small
but real early-warning signal in the pre-attack transition region, evaluated honestly enough
to show where it does not beat a trivial baseline.* The claim that is **not** supported is
that the latent rollout extrapolates network dynamics better than restating the present.

---

## 17. Why is packet-aware V5 not used in the screening demo?

**Because a V5 training dataset cannot be constructed without fabricating data.**

**Measured.** The V5 feature pipeline is complete, contracted and tested — 82 dimensions
(26 flow base + 26 flow rel-change from V4, plus 21 packet base + 9 packet rel-change),
schema `v5` contract 1.0.0, 0 NaN and 0 non-finite values across all 180 conformance cells.
**The V5 model is not trained.**

**Architecture / why it is blocked.** The three TRAIN/VAL days have **zero packet captures**.
Only pre-computed CICFlowMeter flow CSVs were ever obtained for them, and a flow CSV **cannot**
yield TTL, TCP window, IP fragment flags or TCP sequence numbers — those fields do not exist
in it at any aggregation level. Joining packet features only where PCAP exists would populate
the **test day only**, which means a model cannot learn a feature it never sees in training,
and the packet block would become a **perfect test-set indicator — a leakage channel, not a
feature**.

**Interpretation.** Shipping V5 in the demo would require either imputing packet features on
days we have no packets for, or training on a feature that exists only in the test set. Both
are the kind of shortcut this project has refused elsewhere, so V5 stays out of the screening
demo and V4 remains the reference baseline, byte-for-byte as frozen.

**Limitation.** The packet features have therefore **never been evaluated for predictive
value**. Whether they help is an open question. Do not claim they would improve results.

---

## 18. Can this process real PCAP data?

**Yes** — the console has a **Live PCAP Ingestion** page that accepts `.pcap`, `.pcapng` and
`.cap` files up to 1 GB.

**Architecture.** The canonical `src/pcap_ingest_v4.py` pipeline runs end to end:

1. Packets are streamed with Scapy `PcapReader`.
2. Bidirectional flows are reconstructed on the 5-tuple.
3. 52 features are extracted per 10-second window.
4. Features are z-score normalised with the frozen `multiday_benchmark_K5` scaler.
5. The V4 world model runs a recursive K=5 latent rollout; the alert threshold is the
   **frozen, validation-selected** value read from `docs/v4_baseline_manifest.json` — not the
   0.05 training artefact embedded in the checkpoint.
6. Threat alerts and rule-based ATT&CK mapping are produced.

**Interpretation.** The demo replays real held-out March-2 telemetry rather than live capture
because that is the data with documented ground truth to evaluate against — not because the
ingestion path is a mock.

**Limitation.** Parity between the PCAP path and the CICFlowMeter-derived training features is
**APPROXIMATE**, and `docs/FEATURE_PARITY.md` (13/13 verification checks PASS) states the split
precisely: **exact at the aggregation/transform layer, approximate at the flow-production
layer.** In other words, once flows exist the two paths compute the 52 features identically;
reconstructing flows from raw packets is where they can diverge. Quote that verdict if pushed,
rather than claiming exact equivalence.
