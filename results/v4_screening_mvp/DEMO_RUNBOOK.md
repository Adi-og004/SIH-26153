# V4 Screening Demo — Runbook

**Environment:** `src/venv` · Python 3.14.7 · torch 2.13.0 · Streamlit 1.62.0
**Pinned deps:** [`requirements-screening.txt`](../../requirements-screening.txt)
**Fully pinned:** [`requirements-screening-lock.txt`](../../requirements-screening-lock.txt) — use this for a fully pinned reproduction (direct + transitive, 58 packages).
**Runs fully offline.** No internet needed once installed.

---

## 1. Activate the environment

From the repository root:

```bash
cd SIH-26153   # repository root
source src/venv/bin/activate
```

Rebuilding from scratch on another machine:

```bash
python3 -m venv demoenv          # Python 3.14.x

# preferred - fully pinned, direct + transitive (58 packages):
./demoenv/bin/python -m pip install -r requirements-screening-lock.txt

# or, direct deps only (resolver picks transitive versions):
./demoenv/bin/python -m pip install -r requirements-screening.txt
```

## 2. Launch

```bash
streamlit run src/app.py
```

Recommended for the demo — binds to loopback only, which suppresses Streamlit's
external-IP lookup and gives **zero outbound network attempts**:

```bash
streamlit run src/app.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

Open **http://127.0.0.1:8501**. Startup takes ~10–20 s (loads the frozen checkpoint).

> If it errors with *"Screening substrate unavailable"*, run
> `python src/build_v4_screening_mvp.py` first. The app never substitutes
> synthetic data.

## 3. Default page — 🎯 Live Forecast

Opens on **sequence #2341** (observed window `2018-03-02 10:50:10`), the highest
max-forecast sequence in the validated set. Selected from model output only.

Talking points, top to bottom:
- Six cards: observed time · **current threat (nowcast)** · **50 s horizon** ·
  **max future threat 0.797** · frozen threshold **0.69** · ALERT.
- The chart: cyan solid = *observed* nowcast history; amber dashed = *forecast*
  from the K=5 rollout. Red dashed line is the frozen threshold.
- **The point to make:** the amber line is the model predicting forward, not
  classifying the present.
- Scroll down for the network-state forecast (observed vs decoder-predicted
  `flow_count`, `window_packet_rate`, `syn_rate`, `dst_port_entropy`).

> t+4 (+40 s) is drawn but has **no frozen point threshold** — the reference
> evaluation covers horizons 1, 2, 3, 5. Say so if asked.

## 4. ⏱️ Early Warning

Headline figures: **66.7 % onset coverage, 15 s mean strict pre-attack lead,
3 evaluable onset episodes**.

- Use **"Jump to an onset episode"** to move the replay to a real onset
  (`03:54:50`, `10:13:20`, `12:13:30`) — the default sequence sits *inside* an
  active attack, so jump here to show warning behaviour.
- Be upfront: 3 onsets on one held-out day. Indicative, not established.
- If asked about the 50 s figure: that convention credits alerts raised while the
  *previous* attack was still active. The strict 15 s number is the honest one.

## 5. 📊 Window Inspector

The deep dive. Defaults to the current replay sequence.

- **The 10 observed windows** (`t-90 … t`) — this is the model's entire input.
- Feature history table + chart, threat probability across the history.
- Forecast trajectory, ATT&CK context, attribution — all for one window.

## 6. 🔍 Feature Attribution

Press **"Explain this sequence"** (~10–20 s; Integrated Gradients runs on demand).

Returns top contributors with signed attribution, direction, and a plain-English
gloss. For #2341: `window_packet_rate` **+0.441** toward higher threat,
`window_byte_rate` **−0.441** toward lower.

Provenance panel shows method, steps, tensor shape and the hash-verified
checkpoint. Explained probability equals the displayed probability exactly.

## 7. 🗺️ MITRE ATT&CK

Live rule-based mapping for the current sequence:
**TA0043 Reconnaissance / T1018 Remote System Discovery**, confidence *high*.

Say clearly: **the world model does not predict ATT&CK labels.** This is a
deterministic telemetry rule library interpreting the feature values.

> The *stored day-wide* table below shows a red error — expected. It has 502 rows
> against 3,115 sequences and is refused rather than misaligned. The live mapping
> above it is the one to show.

Also worth showing: **🔮 Trajectory Rollout** — the `Z_t → Z_t+5` chain, one row
per gated residual transition. This is the clearest "it's a world model" visual.

## 8. Capture gap — expect it

The held-out day has a real **3.31 h gap** in the raw capture:
`05:29:30 → 08:48:10`. It is never interpolated.

- The replay splits into two segments (sequences 0–1607 and 1608–3114).
- A striped **DATA GAP** banner appears at the boundary.
- **Autoplay stops** at the segment end — press **Next ▶** to continue.
- No line is drawn across the gap and no synthetic windows exist.

If a judge notices the jump: that's the dataset, and the dashboard refuses to
fabricate across it.

## 9. Shutdown

`Ctrl+C` in the terminal. Then `deactivate` if you sourced the venv.

Nothing is written to any frozen artifact — the app is read-only on the
checkpoint, scalers and screening data.

---

## Quick reference

| Item | Value |
|---|---|
| Launch | `streamlit run src/app.py` |
| URL | http://127.0.0.1:8501 |
| Default sequence | **#2341** · `2018-03-02 10:50:10` · max forecast 0.797 |
| Frozen threshold | **0.69** (validation-selected K=5 trajectory-MAX) |
| Forecast horizon | **50 s** (K=5 × 10 s windows) |
| Replay length | 3,115 sequences · held-out March 2, 2018 |
| Alerts at threshold | 1,434 / 3,115 (46.0 %) |
| Held-out F1 / ROC-AUC | 0.798 / 0.893 (K=5 trajectory MAX) |
| Early warning | 66.7 % coverage · 15 s mean strict lead · 3 onsets |
| Capture gap | 05:29:30 → 08:48:10 (3.31 h) |

## Known rough edges

- MITRE page's stored day-wide table errors by design (see §7).
- Methodology page uses red boxes as *styled disclosure*, not failures.
- First render of any page is slow (checkpoint load); later pages are cached.
- `venv/` (the evaluation env) cannot run the dashboard — it has no Streamlit.
  Use `src/venv`.
