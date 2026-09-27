# Final Demo Flow — V4 Screening Console

**A deterministic 90–120 second walkthrough.** Every number below was read off the running
dashboard during browser QA at 1920×1080 and matches the frozen
`screening_predictions.csv`. Follow the clicks exactly and the same numbers appear.

**Do not mention V5 during this run.** It is not part of the two-minute story.

---

## Before you start

```bash
cd SIH-26153   # repository root
source src/venv/bin/activate
streamlit run src/app.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

Open **http://127.0.0.1:8501** at a 1920×1080 window. Confirm the landing page reads
**Sequence ID 2341**, **MAX FUTURE THREAT 0.797**, **ALERT**. If it does not, press
**↺ Reset** in the sidebar — Reset always returns to the default **#2341**.

Runs fully offline. Nothing on the screen requires the network.

---

## 0–10 s · Live Forecast — the problem

**Page:** 🎯 Live Forecast (landing page, no clicks)

> "A conventional IDS mostly tells us what is happening *now*. By the time it fires,
> the attack is already in progress."

Point at the two cards side by side:

| Card | Value |
|---|---|
| **CURRENT THREAT · NOW** | **0.756** — nowcast at *t*, no transition applied |
| **MAX FUTURE THREAT · FORECAST** | **0.797** — peak at t+2 (+20 s) |

> "This console shows both. The left card is the present. The right card is 50 seconds ahead."

---

## 10–30 s · The history and the forecast

**Page:** same. Point at the trajectory chart.

The blue line to the left of **t** is 10 observed windows — 100 seconds of real held-out
March-2 telemetry. The orange diamonds to the right of the dotted line are the forecast,
at **+10s +20s +30s +40s +50s**.

> "Instead of only classifying the current state, our World Model rolls the latent network
> state forward and estimates how threat probability may evolve."

Say the mechanism in one line: the encoder compresses the 10×52 history into a 32-dimensional
latent state, then a learned gated residual transition is applied five times.

---

## 30–45 s · Threshold crossing and future state

**Page:** same.

Point at the red dashed line labelled **Frozen alert threshold 0.69**, and at the forecast
crossing above it.

> "0.69 is not tuned for this demo. It was selected on the validation split and applied
> unchanged to the held-out day."

Then point at the **ALERT STATUS** card, and read the banner wording aloud — it is deliberate:

> "This is a model forecast over a frozen, evaluation-derived threshold — not a statement
> that an attack will occur."

If you have a spare beat, scroll one screen to **Network-state forecast — observed vs
predicted trajectory**: the model also decodes the 26-D network state at each future step,
not just the probability.

---

## 45–65 s · Early Warning — the evaluated result

**Page:** ⏱️ Early Warning
**Click:** scroll to **Jump to an onset episode**, leave the default selection, press **Go to onset →**

This is deterministic: it lands on **sequence #1039**, onset **2018-03-02 03:54:50**.

Read the four headline cards:

| | |
|---|---|
| **ONSET COVERAGE** | **66.7%** — 2 of 3 evaluable onsets warned |
| **MEAN STRICT LEAD** | **15 s** |
| **EVALUABLE ONSETS** | **3** — one held-out day |
| **FALSE-ALERT RATE** | 5.55/h |

> "Across 3 evaluable onset episodes we warned on 2 — 66.7% coverage — with a 15 second
> mean strict lead. **This is a small sample on a single held-out day.** It is indicative,
> not statistically established."

Say the small-sample caveat out loud. It is on screen in the *Read this number carefully*
box, and a judge who spots it before you do will discount everything else.

If asked about "50 seconds": that is the originally published convention, which credits
alerting windows that still lie inside the preceding active attack. The console headlines
the stricter number. Both are on screen.

---

## 65–80 s · Feature Attribution — why

**Page:** 🔍 Feature Attribution (still on #1039)
**Click:** **Explain this sequence** — takes a few seconds, it is Integrated Gradients on the frozen checkpoint

| | |
|---|---|
| **EXPLAINED PROBABILITY** | **0.7794** — P(t+5), matches the displayed forecast |
| Top contributors | `rst_rate` **+0.373**, `log_unique_dst_ports` **+0.209**, `dst_port_entropy` **+0.155**, `mean_down_up_ratio` **−0.124** |

> "Reset-flagged connections, a rising count of distinct destination ports, and destination
> ports spread widely — that is what pushed this forecast up."

Keep the wording non-causal, as the page does: attribution describes model
sensitivity and contribution, it does not establish causality or attacker intent.

---

## 80–95 s · MITRE ATT&CK — the interpretation layer

**Page:** 🗺️ MITRE ATT&CK Timeline (still on #1039)

| | |
|---|---|
| **TACTIC** | Lateral Movement — TA0008 |
| **TECHNIQUE** | Remote Services — T1021 |
| **RULE CONFIDENCE** | HIGH · phase: mid |

Say this line, close to verbatim:

> "The ATT&CK mapping is a deterministic rule-based interpretation layer, not a learned
> ATT&CK classifier."

The banner at the top of the page says the same thing — **The world model does not predict
ATT&CK labels.** The model emits a 26-D state and a scalar probability; a separate 10-rule
telemetry library reads the feature values and matches documented patterns.

---

## 95–110 s · Trajectory Rollout — K = 5

**Page:** 🔮 Trajectory Rollout (still on #1039)

Point down the chain. Each row is one gated residual transition in latent space:

| Step | Threat probability |
|---|---|
| **Z_t** (now) | 0.6440 |
| Z_t+1 (+10 s) | 0.6876 |
| Z_t+2 (+20 s) | 0.7259 |
| Z_t+3 (+30 s) | 0.7582 |
| Z_t+4 (+40 s) | 0.7759 |
| Z_t+5 (+50 s) | 0.7794 |

> "The same decoder and the same threat head read every step. The model is rolling a state
> forward, not emitting one classification. Five transitions is a 50 second horizon."

This is the slide where the phrase *world model* is earned — say it here.

---

## 110–120 s · Close

**Click:** **↺ Reset**, then **🎯 Live Forecast**

You are back on #2341 with **0.756 → 0.797** and the ALERT banner — the frame you opened on.

> "Defenders get a forecast trajectory and an interpretable warning signal rather than only
> a retrospective alert."

Stop there.

---

## Quick reference

| Beat | Page | Click | Number to say |
|---|---|---|---|
| 0–10 s | Live Forecast | — | 0.756 now → 0.797 forecast |
| 10–30 s | Live Forecast | — | 10 windows / 100 s history, +10…+50 s |
| 30–45 s | Live Forecast | — | frozen threshold 0.69 |
| 45–65 s | Early Warning | Go to onset → | 66.7% · 15 s · 3 onsets · small sample |
| 65–80 s | Feature Attribution | Explain this sequence | 0.7794; rst_rate +0.373 |
| 80–95 s | MITRE ATT&CK | — | TA0008 / T1021 · rule-based, not learned |
| 95–110 s | Trajectory Rollout | — | 0.6440 → 0.7794 over K=5 |
| 110–120 s | Live Forecast | ↺ Reset | back to 0.797 |

## Rough edges to steer around

- **The MITRE page's "Stored day-wide mapping" section shows a red box.** That is an
  intentional guard — the stored table has 502 rows against 3,115 sequences and the app
  refuses to display a misaligned join. The live per-sequence mapping above it is the part
  you demo. Do not scroll into it.
- **Feature Attribution takes a few seconds** after the button press. Fill the gap with the
  non-causal caveat rather than waiting in silence.
- **Do not touch the data gap** (05:29:30 → 08:48:10) during the two-minute run. It is a
  good answer to a question, not part of the flow — see `JUDGE_LANGUAGE.md` Q16.
- **Autoplay is off and should stay off.** It halts at the capture gap by design, which is
  correct behaviour but reads as a freeze if you have not explained it.
