# Persistence Hypothesis Test
## How much of V4's future-attack performance is label persistence?

**Generated:** 2026-09-19 · **Verification:** PASS (11/11)
**Repository commit:** `53c22227d368b82ff9e736206a92e85b6bed2514`
**V4 reference commit:** `2be30736872dd7bb072b2e5c4cf7e7fe62605af6`
**Tables:** [results/persistence_hypothesis/](../results/persistence_hypothesis/) ·
**Figures:** [results/persistence_hypothesis/figures/](../results/persistence_hypothesis/figures/) ·
**JSON:** [results/persistence_hypothesis_summary.json](../results/persistence_hypothesis_summary.json)
**Reproduce:** `python src/run_persistence_hypothesis.py` ·
**Verify:** `python src/verify_persistence_hypothesis.py`

Analysis only. No model, threshold, scaler, split or label was modified. Every overall F1
reproduces Task 3 exactly and every confusion matrix reproduces Task 4 exactly.

---

## 1. Research question

The benchmark asks each model to predict `A(t+k)`. But `A(t+k)` is strongly determined by
`A(t)`: attacks stay active across consecutive 10-second windows. A rule that copies the
current label is therefore correct on every non-transitioning window for free.

This document separates the target into two disjoint parts and measures all three models on
each:

| Part | Definition | Who can be right |
|---|---|---|
| **Persistent** | `A(t+k) = A(t)` (categories PP and AA) | persistence is correct **by construction** |
| **Transition** | `A(t+k) ≠ A(t)` (categories ONSET and OFFSET) | persistence is wrong **by construction** |

## 2. Frozen benchmark

Mar-2-2018, 3,115 evaluated windows, horizons K = 1, 2, 3, 5, frozen validation-selected
thresholds (V4 0.45/0.54/0.65/0.72; LR 0.60/0.60/0.60/0.54; persistence none). The
verification script re-derives `A(t+1..t+5)` from the raw day CSV, re-runs the V4 rollout
from the frozen checkpoint, and re-reads `A(t)` from source before any of these numbers are
accepted.

## 3. Why persistence matters — and why it is not a fair rival

`ŷ(t+k) = A(t)` consumes the **observed ground-truth label** at the current window. V4 and
Logistic Regression receive only the 52 numeric features and are never told whether an attack
is under way. Persistence is therefore **not a feature-equivalent predictor**, and its
near-perfect score is *not* evidence that a traffic model could reach the same number.

Its scientific value is different and precise: it measures **how much of the benchmark target
is explained by temporal label continuity alone**.

## 4. Persistence statistics

| K | `P(A(t+k)=A(t))` | `P(A(t+k)=1 \| A(t)=1)` | `P(A(t+k)=0 \| A(t)=0)` | Onset rate `P(1\|0)` | Offset rate `P(0\|1)` | ONSET windows | OFFSET windows | Transitions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.998074 | 0.998523 | 0.997232 | 0.002768 | 0.001477 | 3 | 3 | 6 (0.19%) |
| 2 | 0.996790 | 0.997538 | 0.995387 | 0.004613 | 0.002462 | 5 | 5 | 10 (0.32%) |
| 3 | 0.996148 | 0.997046 | 0.994465 | 0.005535 | 0.002954 | 6 | 6 | 12 (0.39%) |
| 5 | 0.995506 | 0.996553 | 0.993542 | 0.006458 | 0.003447 | 7 | 7 | 14 (0.45%) |

**99.55–99.81% of the benchmark target is already determined by the current label.** Between
6 and 14 windows out of 3,115 actually change state.

**Shuffle diagnostic** (exploratory; prevalence-preserving permutation, seed 42, 1,000 draws —
never mixed into any benchmark score). At K=5, shuffling the target sequence drops persistence
accuracy from **0.9955 to 0.5460** (analytic expectation `p² + q²` = 0.5462). So **0.4495 of
persistence accuracy comes from temporal continuity**, not from class prevalence. This
diagnostic is not run for V4/LR: shuffling their targets only yields a prevalence-driven
chance level and says nothing about their temporal skill.

## 5. Transition decomposition (K=5)

| Category | Meaning | Windows | % of test |
|---|---|---:|---:|
| PP | `0 → 0` benign stays benign | 1,077 | 34.57% |
| AA | `1 → 1` attack stays attack | 2,024 | 64.98% |
| ONSET | `0 → 1` | 7 | 0.22% |
| OFFSET | `1 → 0` | 7 | 0.22% |

**99.55% of the test set is persistent; 0.45% is a state change.**

## 6. Model performance by transition type (K=5, frozen thresholds)

| Model | PP (n=1,077) | AA (n=2,024) | ONSET (n=7) | OFFSET (n=7) |
|---|---|---|---|---|
| Persistence | 1,077 / 1,077 correct (1.000) | 2,024 / 2,024 (1.000) | **0 / 7** | **0 / 7** |
| Logistic Regression | 1,065 / 1,077 (0.989) | 522 / 2,024 (0.258) | 0 / 7 | 5 / 7 (0.714) |
| V4 World Model | 1,040 / 1,077 (0.966) | 1,249 / 2,024 (0.617) | 2 / 7 (0.286) | 0 / 7 (0.000) |

Mean predicted probability by category, K=5 — **the most diagnostic row in this document**:

| Model | PP | AA | ONSET (future = attack) | OFFSET (future = benign) |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.059 | 0.423 | **0.034** | **0.346** |
| V4 World Model | 0.065 | 0.577 | **0.218** | **0.755** |

Both models assign a *lower* probability to windows that are about to **become** attacks than
to windows that are about to **stop** being attacks. Their probability tracks `A(t)`, the
present state, rather than `A(t+k)`, the target.

F1 and ROC-AUC are reported as `unavailable (single-class subset)` for PP and AA in
`transition_model_performance.csv`, because those categories contain only one class and a
classifier metric there would be meaningless.

## 7. Persistent vs transition subsets, and the transition-conditioned diagnostic

| Model | K | Overall F1 | Persistent-subset F1 | Persistent accuracy | Transition n | Transition accuracy | Transition AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 5 | 0.9966 | **1.0000** | **1.0000** | 14 | **0.000** | 0.000 |
| Logistic Regression | 5 | 0.4067 | 0.4081 | 0.5118 | 14 | 0.357 | 0.061 |
| V4 World Model | 5 | 0.7523 | 0.7547 | 0.7381 | 14 | 0.143 | 0.184 |

Overall F1 and persistent-subset F1 are identical to three decimals for every model at every
horizon — **the aggregate benchmark number is the persistent-subset number**. The 14
transition windows are too few to move it.

**Transition-conditioned diagnostic** (not a benchmark score), with each model's own
base rate of firing as the chance reference:

| Model | K | ONSET detected | Expected at own base rate | OFFSET correct | Expected at own base rate |
|---|---:|---:|---:|---:|---:|
| V4 | 1 | 2 / 3 | 0.90 | 0 / 3 | 2.10 |
| V4 | 2 | 3 / 5 | 1.79 | 0 / 5 | 3.21 |
| V4 | 3 | 3 / 6 | 2.22 | 1 / 6 | 3.78 |
| V4 | 5 | 2 / 7 | 2.91 | 0 / 7 | 4.09 |
| LR | 5 | 0 / 7 | 1.20 | 5 / 7 | 5.80 |

V4 detects onsets at roughly its own chance rate (2 of 7 versus 2.91 expected at K=5) and is
below chance on offsets (0 of 7 versus 4.09 expected). Logistic Regression is at or below
chance on both. With 7 events per cell, none of these differences can be tested for
significance.

## 8. Attack onset analysis

Three observable onsets exist (ep1 is already active at the first test window and is **not**
given a synthetic onset). Their 120-second lookbacks are not equivalent:

| Onset | Lookback windows | Of which are already attack windows | Clean cold start |
|---|---:|---:|---|
| ep2 — 03:54:50 | 12 | 9 | No |
| ep3 — 10:13:20 | 12 | **0** | **Yes** |
| ep4 — 12:13:30 | 12 | 10 | No |

K=5 behaviour:

| Model | Episode | Status | Lead time | P at onset | Max pre-onset P | Threshold |
|---|---|---|---:|---:|---:|---:|
| Persistence | ep2 | warning before onset | 120 s | 1.000 | 1.000 | — |
| Logistic Regression | ep2 | warning before onset | 100 s | 0.123 | 0.571 | 0.54 |
| V4 | ep2 | warning before onset | 120 s | 0.767 | 0.763 | 0.72 |
| Persistence | **ep3** | **silent** | — | 0.000 | 0.000 | — |
| Logistic Regression | **ep3** | **silent** | — | 0.013 | 0.031 | 0.54 |
| V4 | **ep3** | **silent** | — | 0.000 | 0.497 | 0.72 |
| Persistence | ep4 | warning before onset | 120 s | 1.000 | 1.000 | — |
| Logistic Regression | ep4 | warning before onset | 70 s | 0.818 | 0.927 | 0.54 |
| V4 | ep4 | warning before onset | 90 s | 0.710 | 0.769 | 0.72 |

The ep2 and ep4 "warnings" occur while the previous episode is still running — persistence
fires there purely because `A(t)=1`. **On the single cold-start onset, all three methods are
silent, and V4's probability is 0.000 at the onset window itself.**

`V4_warnings_before_clean_onsets` = **0** at every horizon.

## 9. Attack offset analysis

Three offsets are observable (ep4 ends with the day). Mean K=5 probability in the 120 s
before vs the 120 s after each offset (`offset_transition_analysis.csv`):

| Model | Offset | Mean P before | Mean P after | Positive predictions after | Time to fall below threshold |
|---|---|---:|---:|---:|---|
| V4 | ep1 — 03:54:10 | 0.762 | **0.768** | 12 of 12 | never within 120 s |
| V4 | ep2 — 03:54:50 | 0.763 | 0.596 | 9 | 100 s |
| V4 | ep3 — 12:13:00 | 0.565 | 0.559 | 8 | 30 s |
| Logistic Regression | ep1 | 0.357 | 0.086 | 0 | 10 s |
| Logistic Regression | ep2 | 0.299 | 0.034 | 0 | 10 s |
| Logistic Regression | ep3 | 0.526 | 0.354 | 3 | 40 s |
| Persistence | ep1 | 1.000 | 0.500 | 6 | 60 s (structural) |

V4 continues asserting attack through the tail — after ep1's offset its mean probability is
marginally *higher* than before it, and it stays above threshold for the whole 120 s window.
In the transition sense it is wrong on **0 of 7** offsets at K=5: it never anticipates an
attack ending, and its probability decays only *after* the label has already changed.
Logistic Regression releases faster (0 positive predictions after two of the three offsets),
which is consistent with its much lower overall positive rate rather than with anticipation.

## 10. V4 K=1 → K=5 decomposition

Phases were defined before counting: `onset_related` (the onset window, or benign within
120 s before an onset), `early` (10–60 s after onset), `late` (within 120 s before offset),
`mid` (>60 s after onset and >120 s before offset), `post_attack`, `benign`, `short_episode`
(<180 s).

V4's false negatives fall from 1,154 (K=1) to 780 (K=5), a reduction of **374 windows**:

| Phase | FN at K=1 | FN at K=5 | Reduction | Share of total |
|---|---:|---:|---:|---:|
| **mid** | 1,129 | 763 | **366** | **97.9%** |
| late | 14 | 9 | 5 | 1.3% |
| early | 10 | 6 | 4 | 1.1% |
| onset_related | 1 | 2 | −1 | −0.3% |
| short_episode | 0 | 0 | 0 | 0.0% |

**97.9% of the gain is on windows deep inside an already-running attack.** Onset-related
windows got marginally *worse*.

A target-aligned pairing (same target window, lead time 1 vs 5) shows the change is also
noisy rather than a clean improvement: in the `mid` phase, 525 windows flip from missed to
caught while **163 flip the other way** (net 362); in the `late` phase 11 improve and 6
regress. So the horizon change reshuffles verdicts on persistent attack windows rather than
adding anticipation.

## 11. Conditional current-state analysis (K=5)

| Model | Condition | n | Positive target rate | Mean P | Recall | FPR |
|---|---|---:|---:|---:|---:|---:|
| Persistence | `A(t)=0` | 1,084 | 0.0065 | 0.000 | 0.000 | 0.000 |
| Persistence | `A(t)=1` | 2,031 | 0.9966 | 1.000 | 1.000 | 1.000 † |
| Logistic Regression | `A(t)=0` | 1,084 | 0.0065 | 0.059 | 0.000 | 0.011 |
| Logistic Regression | `A(t)=1` | 2,031 | 0.9966 | 0.423 | 0.258 | 0.286 † |
| V4 | `A(t)=0` | 1,084 | 0.0065 | 0.066 | 0.286 | 0.034 |
| V4 | `A(t)=1` | 2,031 | 0.9966 | 0.578 | 0.617 | **1.000** † |

† the FPR denominator in the `A(t)=1` condition is only 7 windows (the offsets) — the value is
descriptive, not a rate estimate.

When the present is benign, V4's mean probability is 0.066 and it catches 2 of the 7 windows
that turn into attacks. When the present is under attack, its mean probability is 0.578 and it
never predicts the 7 endings. Persistence's row is structurally determined because it *is*
`A(t)`.

## 12. Optional shuffle diagnostic

Reported in §4 and `shuffle_diagnostic.csv`. Explicitly labelled exploratory, seeded (42), and
never mixed into the Task 3 comparison. It is applied **only** to persistence, because that is
the only model for which destroying temporal order isolates a meaningful quantity.

## 13. Main findings

1. **99.55–99.81% of the benchmark target is fixed by the current label.** Only 6 (K=1) to 14
   (K=5) of 3,115 windows change state.
2. **Overall F1 equals persistent-subset F1** for all three models at all four horizons (to
   three decimals). The aggregate benchmark number is a measurement of behaviour on persistent
   windows.
3. **99.9% of V4's correct windows at K=5 are persistent windows** (`scorecard`:
   0.999127), and 99.8% of its true positives are AA windows.
4. **Both feature models rank transitions backwards.** V4's transition-subset AUC is 0.184 at
   K=5 (LR: 0.061) — below 0.5 — and its mean probability is 0.218 on ONSET versus 0.755 on
   OFFSET.
5. **V4's onset detection is at its own chance rate**: 2 of 7 detected versus 2.91 expected
   from its 41.6% positive rate; 0 of 7 offsets anticipated versus 4.09 expected.
6. **On the only cold-start onset (ep3), every method is silent**, and V4's probability at the
   onset window is 0.000. Its maximum in the 120 s lookback is 0.497 (threshold 0.72), driven
   by a single isolated window 110 s before — not a sustained rise.
7. **97.9% of V4's K=1→K=5 false-negative reduction is mid-attack**, with onset-related
   windows slightly worse; and even the mid-attack gain includes 163 windows that regress.
8. **Persistence's temporal advantage is quantified**: shuffling the targets drops its accuracy
   from 0.9955 to 0.5460, so 0.4495 of it is temporal continuity rather than prevalence.

## 14. Limitations

1. One held-out day; one attack family (all four episodes are Bot).
2. **6–14 transition windows per horizon.** No significance testing is possible, and none is
   performed. Every transition-conditioned number in this document is descriptive.
3. **One cold-start onset.** ep2 and ep4 begin 40 s and 30 s after the preceding episode ends.
4. Windows inside one episode are temporally dependent: 2,024 attack windows at K=5 are **not**
   2,024 independent observations; they come from four episodes.
5. Persistence receives `A(t)`; V4 and LR do not.
6. LR's threshold-transfer failure (Task 2/4) means its threshold-based numbers partly measure
   calibration, not separability.
7. The V4 training/evaluation normalisation provenance caveat from Task 1 remains unresolved.

## 15. Scientific interpretation

### A. What the data demonstrates

- The ordinary `t+k` attack-classification task on this benchmark is **dominated by label
  persistence**: at most 0.45% of windows change state, and every model's aggregate F1 is its
  persistent-subset F1.
- V4's aggregate score is produced almost entirely on persistent windows (99.9% of its correct
  predictions at K=5).
- Neither V4 nor Logistic Regression currently ranks transition windows correctly; both assign
  higher probability to windows about to become benign than to windows about to become
  malicious.
- V4's K=1→K=5 improvement is a mid-attack effect (97.9%), not an onset effect.

### B. What the data suggests

- V4's probability behaves like a **smoothed, lagged estimate of the present state** rather
  than a forecast of the next state: it rises after an attack starts and decays after it ends.
- The single 0.497 pre-onset spike before ep3 is *consistent with* some sub-threshold sensitivity
  to pre-onset traffic, but one window in one episode is far too little to support any claim.
  **No reliable conclusion can be drawn from this sample.**

### C. What remains unproven

- Whether V4 has **any** genuine transition-forecasting ability. The current test day contains
  too few transitions (6–14 windows, 3 onsets, 1 of them cold) to answer this either way.
  Failure to demonstrate skill here is **not** a demonstration that the skill is absent.
- Whether the temporal architecture contributes anything beyond the static classifier, since
  V4 and LR differ in input, model class and calibration simultaneously.
- Whether any of this generalises beyond one day of Bot traffic.

### The benchmark's own verdict

**The current held-out benchmark is not sufficient to establish forecasting value beyond
persistence.** It cannot be: a task whose target is 99.6% determined by the present label
leaves almost no measurable room for anticipation. Deciding whether V4 forecasts requires a
benchmark built around transitions — more onsets, from more days and attack families, scored
on the pre-onset window — not a better threshold or a longer horizon on this one.
