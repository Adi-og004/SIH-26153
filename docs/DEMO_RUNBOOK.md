# SIH 26153 — Demo Runbook (judge walkthrough)

This is the demonstration sequence for the **current V4 dashboard**. Every
step below was run against a fresh copy of this repository. The page names
are the exact sidebar labels.

## 0. Start

```bash
git clone https://github.com/Adi-og004/SIH-26153.git
cd SIH-26153
python3.14 -m venv .venv            # Python 3.14 validated; 3.13 also installs
source .venv/bin/activate
python -m pip install -r requirements-screening-lock.txt
streamlit run src/app.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

Open <http://127.0.0.1:8501>. The first load takes a few seconds: the frozen
checkpoint is hash-verified against `docs/v4_baseline_manifest.json` and
rolled out over the 3,115 held-out sequences. Everything runs locally and
offline.

The default data source is the **Historical Benchmark**: the frozen V4 model
replayed over the held-out day, Friday 2 March 2018 of CSE-CIC-IDS2018. The
model never saw this day during training or threshold selection.

---

## 1. Current network state and forecast — 🎯 Live Forecast

- The threat trajectory: observed history (nowcast) next to the world model's forecast for t+1 … t+5 (10–50 s ahead).
- **Forecast readout**: nowcast, max forecast probability and the frozen **0.69** alert boundary. That boundary is the K=5 trajectory-max threshold selected on validation only.
- **Network-state forecast**: the observed 26-D state against the decoder's predicted future states.
- Use **Replay Controls** in the sidebar (Next ▶ / autoplay) to move through the day.

## 2. Early warning — ⏱️ Early Warning

- **Jump to an onset episode** lists the three Bot-attack onsets on the test day (`#1039`, `#2119`, `#2840`).
- The evaluation summary headline is the **strict** pre-attack result: 2 of 3 onsets warned, **15 s** mean lead. Strict means only windows that are benign at both t and t+1 count. The page also discloses the manifest's reference convention (66.7 % / 50 s) and explains why it is not the headline: it credits alerts that still sit inside the preceding attack. The page says this is a small sample (3 onsets, one day).

## 3. Attack progression over time — 🧭 Event Timeline, 🔮 Trajectory Rollout

- **Event Timeline**: forecast against ground-truth regions across both capture segments. The real capture gap between the segments is shown, not interpolated.
- **Trajectory Rollout**: the latent rollout chain z_t → z_{t+1} → … → z_{t+5} and the probability at each step.

## 4. MITRE ATT&CK mapping — 🗺️ MITRE ATT&CK Timeline and 🧭 Attack Path / Progression

- **MITRE ATT&CK Timeline** shows the rule-based ATT&CK context for the selected sequence, with an analyst explanation. Below it, a red box says the stored *day-wide* table is unavailable. **That is expected**: the old table was built for a different window set, and the page refuses to misalign it.
- **Attack Path / Progression** shows the progression chain for the selected sequence, the rule reading at each horizon, and the telemetry behind the predicted future state.
- Both are **rule-based**. V4 does not predict ATT&CK labels; the rules read the forecast and the features.

## 5. Explanation — 🔍 Feature Attribution, 📊 Window Inspector

- **Feature Attribution** → *Why this forecast*: Integrated Gradients attribution for the displayed probability, plus global feature importance, an attribution heatmap over the 10 input windows, and attack-vs-benign comparison.
- **Explain this sequence** computes a fresh attribution for the current sequence.
- **Window Inspector**: the 10 observed windows feeding the model, the forecast, and the model attribution side by side.

## 6. Reliability and the decision — 🛡️ Reliability / Model Health, 🧑‍💻 Analyst Decision

- **Reliability**: a RELIABLE / CAUTION / UNRELIABLE status from input-shift, rollout-stability and self-consistency indicators. It is rule-based and **not** a calibrated confidence; the page says so.
- **Analyst Decision**: ALERT / BENIGN / ABSTAIN. ABSTAIN means the policy declines to decide automatically and hands the case to the analyst.

## 7. What-if defence (model-based, not causal) — 🛡️ Defend Now Simulator, 🛡️ Counterfactual Defense Lab

- Choose an intervention (uniform rate limit, or block C2 / restrict egress). Compare the *no action* and *defend now* projected trajectories.
- Say this clearly: these are **model-based projections**. V4 never saw interventions during training. The pages show *PROGRESSION AMBIGUOUS — ANALYST REVIEW* when the model's response is mixed.

## 8. Upload prepared feature data — 📥 Data Input

1. Go to **📥 Data Input** and select **📂 Upload Prepared V4 Data**.
2. Upload `examples/prepared_v4/sample_input.npy`. It holds five real held-out sequences: benchmark ids 0, 100, 500, 1000, 2000.
3. Show the validation card: *Format NPY, Shape (5, 10, 52), Valid sequences 5, Rejected 0, ✅ Valid V4 Input*.
4. Show the result. The **same frozen V4 inference call** as the benchmark gives the same probabilities as the stored benchmark predictions: sequences 0 and 1000 exceed the alert boundary, the other three do not (see `examples/prepared_v4/README.md` for the numbers).
5. Optional: upload `examples/prepared_v4/sample_input.csv` (the same first sequence as a 52-column CSV) to show that CSV input gives the same forecast for that sequence.
6. Optional, to show wrong-file handling: upload `data/processed/multiday_benchmark_K5/Y_multistep_test.npy`. The page explains that the file holds *future target states, not V4 model input*, and says what to upload instead.

While an upload is active, MITRE ATT&CK Timeline, Feature Attribution, Window
Inspector and Trajectory Rollout show the uploaded data, with a banner naming
the file. Attack Path, Analyst Decision and the two defence pages need the
benchmark replay and say so. Switch back to **📊 Historical Benchmark** to
return.

## 9. Upload a PCAP — 📥 Data Input

1. Select **📡 Upload PCAP** and upload `examples/pcap/demo_portscan.pcap`, or `demo_c2_beacon.pcap` / `demo_dos_flood.pcap`.
2. Show the pipeline stages: packets parsed → flows reconstructed → 22 ten-second windows → **13 sequences (10, 52)** → frozen V4 K=5 rollout.
3. **Say this honestly:** these are synthetic captures, and the frozen model does **not** raise an alert on them (max P(t+5) ≈ 0.03–0.25). This step shows that the PCAP path runs end to end through the same inference call. It does not show detection. The PCAP features differ in distribution from the CIC-processed training data (see `docs/FEATURE_PARITY.md`).

## 10. Limits — 📐 Methodology & Limits

End here. The page lists what is frozen and the reproduced reference metrics.
It also states that the persistence baseline is near ceiling, lists the known
caveats (including the train/evaluation normalisation mismatch), and shows the
capture gap. Leaving judges with the limitations is deliberate.

---

## If something goes wrong

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError` | Dependencies not installed in the active venv: `python -m pip install -r requirements-screening-lock.txt` |
| pip cannot find `contourpy==1.4.0` | Python too old. Use Python 3.14 (validated) or 3.13. |
| `ProvenanceError` at start-up | A frozen artefact was modified. Restore it with `git checkout -- models data docs/v4_baseline_manifest.json`. |
| `FileNotFoundError: data/processed/...` | Streamlit was started from the wrong directory. Run `streamlit run src/app.py` **from the repository root**. |
| Upload rejected with an explanation | Working as intended. Read the message: it names what the file is and what to upload instead. |
