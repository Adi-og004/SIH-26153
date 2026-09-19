# Error Analysis — False Positives and False Negatives
## Persistence · Logistic Regression · V4 World Model, held-out Mar-2-2018

**Generated:** 2026-09-19 · **Verification:** PASS (10/10 checks)
**Repository commit:** `53c22227d368b82ff9e736206a92e85b6bed2514`
**V4 reference commit:** `2be30736872dd7bb072b2e5c4cf7e7fe62605af6`
**Tables:** [results/error_analysis/](../results/error_analysis/) ·
**Figures:** [results/error_analysis/figures/](../results/error_analysis/figures/) ·
**Machine-readable:** [results/error_analysis_summary.json](../results/error_analysis_summary.json)
**Reproduce:** `python src/run_error_analysis.py` · **Verify:** `python src/verify_error_analysis.py`

Analysis only. No model, threshold, scaler, split, label or frozen artifact was modified.
Every aggregate metric here reproduces `results/standardized_baseline_comparison.csv` exactly.

---

## 1. Purpose

Aggregate F1 says which model scored higher; it does not say *where* each model breaks. This
document locates every error on the held-out day: when false alarms fire, which attack
windows are missed, how errors move with the forecast horizon, and which attack episodes are
hard. It contains no ranking.

## 2. Dataset and frozen benchmark

| Property | Value |
|---|---|
| Test day | 2018-03-02 (`Friday-02-03-2018`), held out |
| Evaluated windows | 3,115 (8.65 hours of 10-second windows) |
| Positives / negatives at t+1 | 2,031 / 1,084 |
| Horizons | K = 1, 2, 3, 5 (no V4 K=4 reference exists; LR K=4 is kept separate) |
| Thresholds | frozen validation-selected: V4 0.45/0.54/0.65/0.72 · LR 0.60/0.60/0.60/0.54 · persistence none |

The raw day contains 3,143 windows; 28 are unusable because a sequence must have 10 input
windows plus 5 targets at exact 10-second spacing. The day also contains a large data gap
between roughly 05:30 and 08:45 — visible as a break in every timeline figure.

## 3. Error definitions

`TP` predicted 1 / actual 1 · `TN` predicted 0 / actual 0 · `FP` predicted 1 / actual 0 ·
`FN` predicted 0 / actual 1, where "predicted" means `score ≥ the model's own frozen
threshold`. Persistence emits 0/1 directly. Thresholds were never reselected, and never
optimised per episode.

**Structural asymmetry — stated once and assumed throughout.** Persistence predicts
`ŷ(t+k) = A(t)`: it is handed the *observed ground-truth label* at the current window. V4 and
Logistic Regression receive only the 52 numeric features and are never told whether an attack
is currently in progress. Persistence errors and feature-model errors are therefore not
equivalent events and are not compared one-for-one anywhere in this document.

## 4. Attack episodes

Four contiguous attack runs exist on the raw test day. All are labelled **Bot**.

| Episode | Onset | Offset | Windows | Duration | Observable onset |
|---|---|---|---:|---:|---|
| ep1 | 01:00:00 | 03:54:10 | 1,046 | 10,460 s | **No** — already active at the first test window |
| ep2 | 03:54:50 | 03:54:50 | 1 | 10 s | Yes |
| ep3 | 10:13:20 | 12:13:00 | 719 | 7,190 s | Yes |
| ep4 | 12:13:30 | 12:59:50 | 279 | 2,790 s | Yes |

This confirms the three onsets recorded in the frozen benchmark (03:54:50, 10:13:20,
12:13:30) and confirms that ep1 has no observable onset — no onset is fabricated for it.
Two of the three onsets are *near-restarts*: ep2 begins 40 s after ep1 ends, and ep4 begins
30 s after ep3 ends. Only **ep3** follows a long benign stretch (≈6 h 18 m). This distinction
drives §7.

## 5. False-positive analysis

| Model | K | FP | FP/hour | % of benign windows | FP runs | Median run | Max run | Before attack | After attack | >120 s from any attack |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 1 | 3 | 0.35 | 0.28% | 3 | 1.0 | 1 | 0 | 3 | 0 |
| Logistic Regression | 1 | 11 | 1.27 | 1.01% | 7 | 1.0 | 3 | 11 | 0 | 11 |
| V4 | 1 | 55 | 6.36 | 5.07% | 12 | 4.0 | 13 | 36 | 19 | 46 |
| Persistence | 5 | 7 | 0.81 | 0.65% | 4 | 1.5 | 3 | 3 | 4 | 0 |
| Logistic Regression | 5 | 14 | 1.62 | 1.29% | 8 | 2.0 | 3 | 13 | 1 | 12 |
| V4 | 5 | 44 | 5.09 | 4.06% | 11 | 3.0 | 12 | 30 | 14 | 30 |

(Full table for K=2 and K=3 in `false_positive_summary.csv`.)

**Temporal context** (fixed buckets chosen before inspecting results; distance measured to the
nearest ground-truth attack window):

| Model | K | ≤10 s before | ≤30 s before | ≤60 s before | ≤120 s before | ≤10 s after | ≤60 s after | ≤120 s after | >120 s away |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V4 | 5 | 2 | 3 | 3 | 3 | 3 | 8 | 11 | 30 |
| Logistic Regression | 5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 12 |
| Persistence | 5 | 2 | 3 | 3 | 3 | 3 | 4 | 4 | 0 |

**Where V4's 44 K=5 false positives sit in the day:** hour 08 → 15, hour 09 → 12, hour 03 →
12, hour 05 → 3, hour 12 → 2. The 27 in hours 08–09 fall in the long benign stretch that
precedes the 10:13 onset, but they are 1–2 hours away from it, so under the fixed 120 s
buckets they are classified as far-from-attack, not as precursors. The 12 in hour 03 sit in
the tail immediately after ep1/ep2. Logistic Regression's 14 K=5 false positives are far
sparser: hour 08 → 7, hour 09 → 3, hour 10 → 2, hour 12 → 2.

**Persistence false positives are all attack-offset artefacts** — `A(t)=1` while `A(t+k)=0`
— and never occur far from an attack, by construction.

> A false positive that occurs near an attack remains a false positive under the benchmark
> label. Nothing in this section relabels any window.

## 6. False-negative analysis

Missed attack windows at K=5, by episode:

| Model | ep1 (1,032 w) | ep2 (1 w) | ep3 (719 w) | ep4 (279 w) | Total |
|---|---:|---:|---:|---:|---:|
| Persistence | 0 (0.0%) | 0 | 5 (0.7%) | 2 (0.7%) | 7 |
| Logistic Regression | 813 (78.8%) | 1 (100%) | 465 (64.7%) | 230 (82.4%) | 1,509 |
| V4 | 357 (34.6%) | 0 (0%) | 265 (36.9%) | 158 (56.6%) | 780 |

Median probability of a missed window at K=5: V4 0.0982–0.1998 per episode; Logistic
Regression 0.2415–0.3053; persistence 0.0 (its misses are exactly the windows where
`A(t)=0`).

V4's largest single miss group is **ep4** (56.6% of its windows missed) — the 279-window
episode that begins 30 s after ep3 ends. Logistic Regression misses the majority of every
episode, including 78.8% of the 1,032-window ep1.

## 7. Onset analysis (the early-warning question)

Fixed 120 s lookback (12 windows) before each observable onset, K=5 scores, frozen
thresholds. **The lookback composition matters more than the warning counts:**

| Onset | Lookback windows | Of which are themselves attack windows |
|---|---:|---:|
| ep2 — 03:54:50 | 12 | **9** (tail of ep1) |
| ep3 — 10:13:20 | 12 | **0** (clean benign lookback) |
| ep4 — 12:13:30 | 12 | **10** (tail of ep3) |

| Model | Episode | Warning | First warning | Lead time | Warning windows | Max probability in lookback |
|---|---|---|---|---:|---:|---:|
| Persistence | ep2 | Yes | 03:52:50 | 120 s | 12 | 1.000 |
| Logistic Regression | ep2 | Yes | 03:53:10 | 100 s | 2 | 0.571 |
| V4 | ep2 | Yes | 03:52:50 | 120 s | 12 | 0.763 |
| Persistence | **ep3** | **No** | — | — | 0 | 0.000 |
| Logistic Regression | **ep3** | **No** | — | — | 0 | 0.031 |
| V4 | **ep3** | **No** | — | — | 0 | 0.496 |
| Persistence | ep4 | Yes | 12:11:30 | 120 s | 12 | 1.000 |
| Logistic Regression | ep4 | Yes | 12:12:20 | 70 s | 7 | 0.927 |
| V4 | ep4 | Yes | 12:12:00 | 90 s | 9 | 0.769 |

The ep2 and ep4 "warnings" are raised while the *previous* episode is still in progress —
persistence warns there simply because `A(t)=1`. They are not evidence of anticipating a new
attack. **ep3 is the only onset in this dataset preceded by a clean benign lookback, and all
three methods are silent for it.** V4's probability does rise to 0.496 in that window —
elevated relative to the benign baseline but below its frozen 0.72 threshold; Logistic
Regression peaks at 0.031.

With three onsets, one of which is a genuine cold start, this is descriptive only. **No lead
time is guaranteed by any model on the basis of this evidence.**

## 8. Post-attack behaviour

After each attack offset (up to 300 s examined, K=5):

| Model | Episode | Positive predictions after offset | Time until below threshold | Mean P inside episode | Mean P first 60 s after | FPs in tail |
|---|---|---:|---:|---:|---:|---:|
| V4 | ep1 (03:54:10) | 13 | 140 s | 0.595 | 0.771 | 12 |
| V4 | ep2 (03:54:50) | 9 | 100 s | 0.767 | 0.774 | 9 |
| V4 | ep3 (12:13:00) | 10 | 30 s | 0.577 | 0.742 | 2 |
| Logistic Regression | ep1 | 0 | 10 s | 0.396 | 0.133 | 0 |
| Logistic Regression | ep2 | 0 | 10 s | 0.123 | 0.036 | 0 |
| Logistic Regression | ep3 | 3 | 40 s | 0.478 | 0.551 | 2 |
| Persistence | ep1 | 6 | 60 s | 1.000 | 0.833 | 5 |

V4 keeps predicting attack for 30–140 s after an episode ends, which accounts for 11 of its
44 K=5 false positives (the ≤120 s-after bucket). Logistic Regression's probability collapses
within one window after two of the three offsets. Persistence stays positive purely because
`A(t)` is still 1 in the windows whose horizon still reaches into the episode.

## 9. Feature-space analysis

`error_feature_summary.csv` reports mean/median/std/p10/p90 for 16 raw (unscaled) base
features at forecast time *t*, per model × horizon × error type. Two framing notes:

- The values are the **base** features. The 26 `*_rel_change` features in the frozen schema
  are ratios of change between consecutive windows, not independent measurements, and are not
  summarised as if they were raw quantities.
- The statistics describe where errors sit in feature space; they do not establish that any
  feature causes an error.

## 10. Horizon analysis

| Model | K | TP | FP | TN | FN | Precision | Recall | F1 | FP/h | FN/h | Max FP run | Max FN run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 1 | 2028 | 3 | 1081 | 3 | 0.9985 | 0.9985 | 0.9985 | 0.35 | 0.35 | 1 | 1 |
| Persistence | 5 | 2024 | 7 | 1077 | 7 | 0.9966 | 0.9966 | 0.9966 | 0.81 | 0.81 | 3 | 5 |
| Logistic Regression | 1 | 443 | 11 | 1073 | 1588 | 0.9758 | 0.2181 | 0.3565 | 1.27 | 183.5 | 3 | 287 |
| Logistic Regression | 5 | 522 | 14 | 1070 | 1509 | 0.9739 | 0.2570 | 0.4067 | 1.62 | 174.4 | 3 | 110 |
| V4 | 1 | 877 | 55 | 1029 | 1154 | 0.9410 | 0.4318 | 0.5920 | 6.36 | 133.4 | 13 | 112 |
| V4 | 2 | 1060 | 55 | 1029 | 971 | 0.9507 | 0.5219 | 0.6739 | 6.36 | 112.2 | 13 | 38 |
| V4 | 3 | 1106 | 45 | 1039 | 925 | 0.9609 | 0.5446 | 0.6952 | 5.20 | 106.9 | 13 | 34 |
| V4 | 5 | 1251 | 44 | 1040 | 780 | 0.9660 | 0.6160 | 0.7523 | 5.09 | 90.1 | 12 | 32 |

**Where V4's K=1 → K=5 gain comes from.** Attack windows bucketed by time since *their own*
episode onset (`horizon_evolution.csv`), recall per bucket:

| Bucket | Windows | V4 K=1 | V4 K=2 | V4 K=3 | V4 K=5 | FN K=1 → K=5 |
|---|---:|---:|---:|---:|---:|---|
| 0–60 s after onset | 13 | 0.308 | 0.385 | 0.385 | 0.462 | 9 → 7 (−2) |
| 60–300 s | 64–68 | 0.088 | 0.090 | 0.242 | 0.234 | 62 → 49 (−13) |
| 300–1800 s | 450 | 0.293 | 0.351 | 0.380 | 0.458 | 318 → 244 (−74) |
| **>1800 s** | ~1,500 | 0.490 | 0.594 | 0.609 | 0.681 | **765 → 480 (−285)** |

Of V4's 374-window reduction in false negatives from K=1 to K=5, **285 (76%) occur in windows
more than 30 minutes deep inside an ongoing attack episode**, and only 2 occur in the first
60 s after an onset. Logistic Regression shows the same pattern more starkly: its recall in
the 0–300 s buckets stays at or near zero for every horizon, rising only in the >1800 s
bucket (0.276 → 0.319).

## 11. Logistic Regression threshold-shift analysis

`lr_probability_shift.csv`. At K=5 (threshold 0.54):

| Probability bin | Attack windows | Benign windows |
|---|---:|---:|
| 0.00–0.10 | 115 | 904 |
| 0.10–0.20 | 329 | 109 |
| 0.20–0.30 | 441 | 36 |
| 0.30–0.40 | 334 | 11 |
| 0.40–0.50 | 220 | 9 |
| 0.50–0.60 | 130 | 2 |
| 0.60–1.00 | 462 | 13 |

| Horizon | Threshold | FN | FN caused by probability below threshold | Median P (attack) | Median P (benign) | Attack windows above the benign p90 |
|---|---:|---:|---:|---:|---:|---|
| K=1 | 0.60 | 1,588 | 1,588 (100%) | 0.2988 | 0.0161 | 1,743 / 2,031 (85.8%) |
| K=2 | 0.60 | 1,578 | 1,578 (100%) | 0.3088 | 0.0183 | 1,749 / 2,031 (86.1%) |
| K=3 | 0.60 | 1,581 | 1,581 (100%) | 0.3176 | 0.0187 | 1,765 / 2,031 (86.9%) |
| K=5 | 0.54 | 1,509 | 1,509 (100%) | 0.3370 | 0.0212 | 1,773 / 2,031 (87.3%) |

Every Logistic Regression false negative is a probability-below-threshold event. The
separation itself is intact — 87.3% of attack windows score above the 90th percentile of
benign windows, and ROC-AUC is 0.9465 — but the attack distribution is centred near 0.34
while the validation-selected threshold sits at 0.54. The test day's positive prevalence is
65.2% against 26.6% in validation. **The threshold was not changed, and the test-optimal
values (0.08–0.10) are recorded for diagnosis only.**

## 12. Persistence transition analysis

Persistence is wrong **only** at attack-state transitions, by construction. `A(t+k) ≠ A(t)`
happens 6 times at K=1 and 14 times at K=5 out of 3,115 windows.

| K | 0→1 transitions | 1→0 transitions | Persistence correct | LR correct (0→1 / 1→0) | V4 correct (0→1 / 1→0) |
|---:|---:|---:|---|---|---|
| 1 | 3 | 3 | 0 / 0 | 0 / 3 | 2 / 0 |
| 2 | 5 | 5 | 0 / 0 | 0 / 5 | 3 / 0 |
| 3 | 6 | 6 | 0 / 0 | 0 / 5 | 3 / 1 |
| 5 | 7 | 7 | 0 / 0 | 0 / 5 | 2 / 0 |

Persistence's steady-state accuracy is **1.0000** at every horizon: outside transitions it is
never wrong. Its entire error budget is the 6–14 transition windows, which is exactly why its
F1 is ~0.997 — the task contains almost no transitions.

**The transition counts must be read against each model's base rate of predicting positive**
(`transition_analysis.csv` records this):

| Model (K=5) | Predicted-positive rate | 0→1 correct | Expected at base rate | 1→0 correct | Expected at base rate |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.652 | 0 | 4.56 | 0 | 2.44 |
| Logistic Regression | 0.172 | 0 | 1.20 | 5 | 5.80 |
| V4 | 0.416 | 2 | 2.91 | 0 | 4.09 |

V4 caught 2 of 7 rising transitions where its own positive rate would produce ~2.9 by chance;
Logistic Regression caught 5 of 7 falling transitions against ~5.8 expected. **On this
evidence neither feature-based model anticipates state changes better than its own base rate
of firing.** With 7 transitions per horizon these counts cannot support a stronger claim in
either direction.

## 13. Representative cases

Selection rule fixed in advance, applied mechanically: for V4 and Logistic Regression at
K=5, take the three most common taxonomy categories per error type and pick the
**median-probability** case in each (never the extreme). All persistence K=5 errors are
listed because there are few. Full table with ±3-window context:
`representative_cases.csv`.

| Case | Model | Type | Category | Time | P | What it illustrates |
|---|---|---|---|---|---:|---|
| C01 | V4 | FP | FP-D | 09:49:30 | 0.761 | Sustained false-alarm run in the benign stretch, >120 s from any attack; the 3 preceding windows scored 0.003–0.046, so the run starts abruptly |
| C02 | V4 | FP | FP-C | 03:54:20 | 0.763 | Post-attack residual: 10 s after ep1's offset, probability essentially unchanged from inside the episode (0.761–0.762) |
| C03 | V4 | FP | FP-A | 05:25:30 | 0.741 | Isolated single-window false alarm amid noisy neighbours (0.298–0.730) |
| C04 | V4 | FN | FN-D | 01:12:00 | 0.025 | Missed window deep inside ep1 while probability collapses from 0.716 → 0.005 across four consecutive attack windows |
| C05 | V4 | FN | FN-E | 03:11:40 | 0.696 | Missed by 0.024 of probability — inside the dense 0.70–0.79 band that the 0.72 threshold bisects (200 such cases) |
| C06 | V4 | FN | uncertain | 12:15:50 | 0.0002 | Near-zero probability 140 s into ep4, which V4 largely fails to track |
| C08 | LR | FP | FP-D | 08:53:30 | 0.995 | Near-certain false alarm in benign traffic — LR's few false positives are highly confident |
| C10 | LR | FN | FN-D | 12:06:30 | 0.254 | The dominant LR failure: attack window ranked well above benign traffic yet far below the 0.54 threshold |
| C11 | LR | FN | FN-E | 11:49:20 | 0.483 | Missed by 0.057 of probability |
| C13–C26 | Persistence | FP/FN | FP-B/C, FN-C | 03:54:20–12:14:10 | 0/1 | Every persistence error clusters at the ep1→ep2 and ep3→ep4 boundaries and at ep3's onset — i.e. exactly the transitions |

Taxonomy totals at K=5 (`error_taxonomy.csv`): V4 FP → FP-D 26, FP-C 11, FP-A 4, FP-B 3;
V4 FN → FN-D 529, FN-E 200, uncertain 43, FN-C 5, FN-A 3. LR FN → FN-D 1,246, FN-E 187,
uncertain 62, FN-C 7, FN-A 6, FN-B 1. Categories are **operational** definitions about
measured timing and probability, never claims about attacker behaviour; cases the rules do
not cover remain `uncertain`.

## 14. Limitations

1. **One held-out day.** Every number describes 2018-03-02 only.
2. **Three observable onsets, one of them a cold start.** Onset findings (§7) are
   descriptive; no statistical claim is possible.
3. **Persistence receives `A(t)`.** Its errors mark label transitions, not forecasting
   failures comparable to the other two models'.
4. **The V4 provenance caveat from Task 1 is unresolved** — the checkpoint was trained under
   a different data/scaler provenance than the frozen reference evaluation.
5. **Logistic Regression's threshold does not transfer** across the validation→test
   prevalence shift (§11); its threshold-based errors partly measure calibration, not
   separability.
6. **The taxonomy does not establish causality.** A false positive before an onset is a
   temporal coincidence under fixed buckets, not evidence of precursor detection.
7. **A false positive near an attack remains a false positive** under the benchmark label.
   Nothing here relabels any window.
8. **All four episodes are Bot traffic.** No conclusion transfers to other attack classes.

## 15. Key empirical findings

1. At K=5 on the held-out day, V4 generated **5.09 false-positive windows/hour** (44 windows,
   4.06% of benign windows), Logistic Regression **1.62/hour** (14 windows) and persistence
   **0.81/hour** (7 windows).
2. **V4's false positives arrive in runs** — 11 runs, median length 3, longest 12 consecutive
   windows — while Logistic Regression's are short (median 2, max 3). 27 of V4's 44 K=5 false
   positives fall in hours 08–09 of the benign stretch; 11 fall within 120 s after an attack
   offset.
3. **V4 takes 30–140 s to fall below threshold after an attack ends**, producing 12, 9 and 2
   tail false positives at the three offsets. Logistic Regression drops within 10 s at two of
   the three.
4. **76% of V4's K=1→K=5 false-negative reduction (285 of 374 windows) occurs more than 30
   minutes into an ongoing attack**, and only 2 windows in the first 60 s after an onset.
5. **All three methods were silent before the only cold-start onset (ep3, 10:13:20).** V4's
   probability reached 0.496 against its 0.72 threshold; Logistic Regression reached 0.031.
   The apparent warnings before ep2 and ep4 occur while the preceding episode is still active
   (9 of 12 and 10 of 12 lookback windows are themselves attack windows).
6. **Persistence has 1.0000 steady-state accuracy and errs only at transitions** — 6 windows
   at K=1, 14 at K=5, out of 3,115. The task contains almost no transitions, which is the
   arithmetic behind its ~0.997 F1.
7. **Neither feature-based model beats its own base rate on transitions**: V4 caught 2 of 7
   rising transitions (≈2.9 expected at its 41.6% positive rate); LR caught 5 of 7 falling
   transitions (≈5.8 expected at its 82.8% negative rate).
8. **Every Logistic Regression false negative is a below-threshold event**, yet 87.3% of
   attack windows still rank above the 90th percentile of benign windows (ROC-AUC 0.9465) —
   a calibration failure, not a separability failure.
9. **V4's probabilities are strongly saturated.** They concentrate in a narrow 0.70–0.79 band
   with frequent collapses to ~0; the frozen 0.72 threshold bisects that band, which is why
   200 of its 780 K=5 misses sit within 0.10 below the threshold. The concentration is tight
   enough that rounding the scores to 6 decimals merges 3,115 distinct values into 2,292 and
   shifts ROC-AUC by up to 3.8e-4 — the reason `error_events.csv` stores full precision.
10. **V4 misses 56.6% of ep4** (the 279-window episode beginning 30 s after ep3 ends), its
    weakest per-episode recall at K=5.
