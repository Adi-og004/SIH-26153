# Training ↔ PCAP Feature Parity

**Generated:** 2026-09-19 · **Verification:** PASS (13/13 checks)
**Parity status:** **APPROXIMATE PARITY** — exact at the aggregation/transform layer, approximate at the flow-production layer
**Repository commit:** `53c22227d368b82ff9e736206a92e85b6bed2514`
**Contract:** [docs/feature_contract_v4.json](feature_contract_v4.json) · SHA256 `6ec22f95a7c1cf30b3915b0327120745d368bf2e7684c329f82beb7a4b9b6674`
**Artifacts:** [results/feature_parity/](../results/feature_parity/) · [results/feature_parity_summary.json](../results/feature_parity_summary.json)
**Reproduce:** `python src/run_feature_parity.py` · **Verify:** `python src/verify_feature_parity.py` · **Tests:** `python tests/test_feature_parity.py`

No frozen artifact was modified. The contract is tied to the frozen baseline through the
**separate** `docs/v4_feature_parity_manifest.json`; `docs/v4_baseline_manifest.json` keeps its
original hashes.

---

## 1. Purpose

The deployed demo path is `PCAP → packets → flows → 10 s windows → 52-dim state → V4`. The
benchmark path is `CICFlowMeter CSV → 10 s windows → 52-dim state → V4`. If these two produce
different feature *semantics*, then every number in Tasks 1–6 describes a representation the
demo never actually feeds the model. This task establishes, mechanically and repeatably, how
far the two agree — and refuses to overstate it.

## 2. The 52-feature contract

26 base features + 26 relative-change features, in the frozen order. Unchanged here: no
feature added, removed, reordered, or redefined. Authoritative order:
`data/processed/multiday_benchmark_K5/input_feature_names.txt` (SHA256 `137b5833…9ecbb5`),
cross-checked against `docs/v4_feature_schema.json` — verified identical.

`docs/feature_contract_v4.json` now records, for each of the 52 features: index, name,
category, source level, aggregation, the **training formula and the PCAP formula quoted from
the implementations**, transform, unit, numeric type, missing-value behaviour, dependencies,
and two separate parity verdicts (aggregation layer and flow layer).

## 3. The two implementations

| Layer | Training | PCAP inference |
|---|---|---|
| Flow records | CSE-CIC-IDS2018 CICFlowMeter CSVs (Java engine, produced by the dataset authors) | `pcap_ingest_v4.parse_pcap_streaming` → `reconstruct_flows` |
| Window aggregation | `archive/create_network_states_v2.py` | `pcap_ingest_v4.compute_window_features` |
| Relative change | `src/prepare_multiday_benchmark.py` | `pcap_ingest_v4.add_relative_changes` |
| Scaling + sequences | `src/prepare_multiday_benchmark.py` | `pcap_ingest_v4.build_sequences` |

These are **separate implementations with equivalent formulas** (status **B**, not A). They
were deliberately not merged: rewiring the frozen training builder could perturb the frozen V4
representation, which Task 1 forbids. `src/feature_contract.py` is therefore a *descriptive*
contract module plus reference implementations used only by the tests.

## 4. Window semantics

Verified identical in both paths:

| Property | Value |
|---|---|
| Width | 10 s |
| Interval | `[t, t+10)` — left-closed, right-open |
| Binning | training `Timestamp.dt.floor("10s")`; PCAP `int(start // 10)` — both floor |
| Origin | epoch-aligned multiples of 10 s |
| Row assignment | flow belongs to the window of its **start**; all its statistics go to that window |
| Rounding | floor, never round-half, no tolerance band |
| Empty window | **no row** in either path; the downstream contiguity gate (`np.all(diffs == 10)`) then rejects any sequence spanning the gap |
| First window | kept; its relative-change row forced to exactly 0.0 |
| Last window | kept; sequences lacking K future targets are simply not formed |

**One caveat, reported not hidden:** the clocks differ. Training timestamps are the CIC CSV's
naive local clock (Atlantic, AM/PM dropped); PCAP timestamps are UTC epoch. The bin *width*
and *boundary rule* are identical, so no feature value changes — but a PCAP window cannot be
aligned to a dataset window without an explicit offset.

## 5. Relative-change semantics

Identical in both paths (verified by test and by the synthetic comparison):

```
r_t = (s_t − s_{t−1}) / (|s_{t−1}| + 1e−8)
first row forced to exactly 0.0 · non-finite → 0.0 · clip to [−10, +10]
```

- Previous state = the immediately preceding **row** of the state table (the previous *existing*
  window — not necessarily t−10 s when a window is missing).
- Zero denominators are absorbed by the epsilon, not masked; the resulting large ratio is clipped.
- Computed **on the already log-transformed** base features (log at aggregation, relative change after).
- Reset scope: training resets per **source day**, PCAP resets per **capture** — both reset at the
  boundary of one contiguous unit and never carry a change across it. **EXACT.**

## 6. Transform semantics

| Transform | Applied | Training | PCAP | Parity |
|---|---|---|---|---|
| `safe_log1p` | after aggregation, before relative change | `create_network_states_v2.safe_log1p` | `_safe_log1p` (verbatim port) | **EXACT** |
| `safe_divide` | before aggregation (per flow) | `create_network_states_v2.safe_divide` | `_safe_divide` | **EXACT** |
| `entropy_from_counts` | at aggregation | base-2, `p + 1e-12`, 0.0 when ≤1 distinct | verbatim port | **EXACT** |
| flag rate | before aggregation | `flag / (total_packets + 1e-8)` | `int(any flag) / (tot_pkts + 1e-8)` | **EXACT** |
| window rate | after aggregation | `total / 10` | `total / 10` | **EXACT** |
| non-finite → 0.0 | after aggregation | both | both | **EXACT** |
| `nan_to_num(0, +10, −10)` | after scaler | both | both | **EXACT** |

## 7. Feature-by-feature parity

Two verdicts per feature. **Aggregation parity** = do the two window builders agree given the
same flows (proven empirically). **Flow-layer parity** = do the two paths produce the same flow
records in the first place.

| Feature | Training | PCAP | Aggregation | Flow layer | Evidence |
|---|---|---|---|---|---|
| `flow_count` | count of CIC flow rows in window | count of reconstructed flows starting in window | **EXACT** | APPROXIMATE | CIC terminates on FIN/RST + timeout; PCAP splits only on >120 s idle |
| `log_flow_count` | `safe_log1p(flow_count)` | same | **EXACT** | APPROXIMATE | inherits `flow_count` |
| `log_total_packets` | `log1p(Σ Tot Fwd+Bwd Pkts)` | `log1p(Σ tot_pkts)` | **EXACT** | APPROXIMATE | flow segmentation + capture scope |
| `log_total_bytes` | `log1p(Σ TotLen Fwd+Bwd)` | `log1p(Σ L4 payload)` | **EXACT** | APPROXIMATE | different byte accounting |
| `window_packet_rate` | `total_packets / 10` | same | **EXACT** | APPROXIMATE | inherits totals |
| `window_byte_rate` | `total_bytes / 10` | same | **EXACT** | APPROXIMATE | inherits totals |
| `syn_rate` … `urg_rate` (6) | `Flag Cnt / (pkts + 1e-8)` per flow, then mean; CIC flags are **boolean 0/1** | `int(any flag) / (pkts + 1e-8)` per flow, then mean | **EXACT** | APPROXIMATE | same definition; binarised flags are highly flow-length sensitive, and reconstructed flows are shorter — `syn_rate`/`fin_rate` diverge most |
| `mean_pkt_len` | mean of per-flow `Pkt Len Mean` | mean of per-flow mean L4 payload | **EXACT** | APPROXIMATE | mean-of-means structure identical; byte basis differs |
| `std_pkt_len` | mean of per-flow `Pkt Len Std` (**not** pooled) | mean of per-flow sample std (ddof=1) | **EXACT** | APPROXIMATE | same structure; byte basis differs |
| `log_mean_flow_duration` | `log1p(mean Flow Duration)`, **µs** | `log1p(mean flow_duration_us)`, **µs** | **EXACT** | APPROXIMATE | units aligned; flow lifetimes differ by engine |
| `log_mean_fwd_iat`, `log_mean_bwd_iat` | `log1p(mean Fwd/Bwd IAT Mean)`, µs | `log1p(mean *_iat_mean_us)` | **EXACT** | APPROXIMATE | units aligned; residual divergence measured |
| `fwd/bwd_packet_ratio`, `fwd/bwd_byte_ratio` (4) | `safe_divide` per flow, then mean | identical formula, initiator direction | **EXACT** | APPROXIMATE | same convention; segmentation differs |
| `mean_down_up_ratio` | mean of CIC `Down/Up Ratio` (integer) | mean of `int(n_bwd // n_fwd)` | **EXACT** | APPROXIMATE | both integer; CIC's exact rule unpublished |
| `log_unique_dst_ports`, `unique_protocols`, `dst_port_entropy`, `protocol_entropy` | nunique / entropy over window flows | identical formula | **EXACT** | APPROXIMATE | **capture-scope dependent** — training aggregates the whole victim subnet |
| 26 × `*_rel_change` | `prepare_multiday_benchmark.py` | `add_relative_changes` | **EXACT** | inherits base | identical transform |

**Counts: 0 EXACT, 52 APPROXIMATE, 0 NON-PARITY, 0 UNKNOWN at the flow layer; 52 EXACT at the
aggregation/transform layer.**

## 8. Flow-reconstruction differences

| Aspect | CICFlowMeter (training) | PCAP path |
|---|---|---|
| Termination | FIN/RST plus a flow timeout | split only on a >120 s idle gap |
| Direction | engine's initiator convention | direction of the flow's first packet |
| Bytes | engine's `TotLen`/`Pkt Len` | L4 payload length |
| Flags | per-flow counts, **binarised to 0/1** by the dataset authors' preprocessing | per-flow boolean (reproduces the binarisation) |
| Timestamps | CSV local clock | UTC epoch, microsecond-derived |
| Scope | whole victim subnet | only the hosts in the capture |

These are definitional differences between two flow engines, not bugs. They are why the
flow layer is APPROXIMATE and can only be closed by using the same engine and the same capture
scope on both sides.

## 9. Synthetic parity test

18 golden fixtures — empty window, single flow, multiple flows, SYN-only, ACK-heavy, RST,
FIN/PSH/URG, many ports, repeated port, multi-protocol, zero bytes, zero packets, near-zero IAT,
forward-only, backward-only, balanced, increasing volume, decreasing volume — each rendered into
**both** schemas and pushed through **both real implementations**
(`archive/create_network_states_v2.py` executed on a temporary CIC-shaped CSV, and
`pcap_ingest_v4.compute_window_features`).

**Result: 954 comparisons, 0 mismatches**, at tolerance 1e-9. Window counts match on every
fixture, including the empty-window case where both paths correctly emit no row.

This is the load-bearing evidence: given identical flow records, the two pipelines produce
**identical 52-dimensional states**.

## 10. Real-PCAP validation

All seven captures in `data/sample_pcap/` were parsed, reconstructed, windowed and checked
against the contract:

| Capture | Packets | Windows | Conforms |
|---|---:|---:|---|
| `demo_c2_beacon.pcap` | 1,390 | 22 | ✓ |
| `demo_dos_flood.pcap` | 5,857 | 22 | ✓ |
| `demo_exfiltration.pcap` | 10,470 | 22 | ✓ |
| `demo_lateral_movement.pcap` | 38,406 | 22 | ✓ |
| `demo_portscan.pcap` | 10,190 | 22 | ✓ |
| `smallFlows.pcap` | 14,243 | 30 | ✓ |
| `synthetic_traffic.pcap` | 660 | 13 | ✓ |

Each checked for: all 52 features present, exact contract order, all values finite,
relative changes within ±10, first-row relative change exactly zero, and window indices
floor-aligned to 10 s.

**No reference state CSV exists for these synthetic captures, so this is conformance, not
numerical agreement** — as the task anticipated.

## 11. Real-data cross-check

A controlled numerical comparison against the Mar-2 dataset flows was **attempted in an earlier
session and is reported here rather than repeated**: the PCAP path was compared against the
training states for 1, 10 and 21 real Mar-2 victim hosts, and against the official Java
CICFlowMeter run in CI. Mean |z| versus the training distribution fell 7.91 → 3.37 → 2.78 → 2.13
as scope grew, and the official CICFlowMeter scored 3.00 versus the emulation's 2.50 — i.e. the
emulation is faithful, and the residual is scope plus flag/IAT definition, not a coding error.
Those residuals are exactly why this document says APPROXIMATE and not EXACT. The artifacts live
under `models/pcap_analysis_v4/parity/`.

## 12. Contract hash

Canonical SHA256 over the feature order, formulas, transforms, window rule, relative-change rule
and scaler interface:

```
6ec22f95a7c1cf30b3915b0327120745d368bf2e7684c329f82beb7a4b9b6674
```

Recorded in `docs/feature_contract_v4.json`, `results/feature_parity/feature_contract_sha256.txt`,
`results/feature_parity_summary.json` and `docs/v4_feature_parity_manifest.json`. Two regression
tests prove the hash changes if any formula or the window size changes.

## 13. Verification results

| Check | Result |
|---|---|
| Schema (52, none missing/unexpected) | PASS |
| Feature order vs frozen schema | PASS |
| 26 base features | PASS |
| 26 relative-change features | PASS |
| Window semantics (10 s, both paths) | PASS |
| Transform semantics (7 shared transforms) | PASS |
| Zero/NaN handling (4 degenerate fixtures) | PASS |
| Scaler interface (52 named columns, no PCA) | PASS |
| Golden tests | PASS (20/20) |
| Synthetic parity | PASS (954 comparisons, 0 mismatches) |
| Contract hash | PASS |
| PCAP contract (reads names from disk, 7/7 conform) | PASS |
| Flow-layer parity reported | PASS (52 APPROXIMATE, 0 NON-PARITY, 0 UNKNOWN) |

## 14. Known mismatches

1. **Flow segmentation** — CICFlowMeter's FIN/RST termination versus 120 s idle splitting.
   Affects every volume, timing and ratio feature. *Severity: medium, structural.*
2. **`syn_rate` / `fin_rate`** — same definition, materially different measured distributions
   (several training σ). The binarised-flag definition is pathologically sensitive to flow
   length. *Severity: high.* Confirmed not fixable by adding hosts, and **not** fixed by running
   the official Java CICFlowMeter.
3. **Capture scope** — `flow_count`, `log_unique_dst_ports`, `unique_protocols`,
   `dst_port_entropy`, `protocol_entropy` are whole-subnet quantities in training and
   host-subset quantities in a demo capture. *Severity: high, but a deployment property, not a
   code defect.*
4. **Byte accounting** — CIC packet lengths versus L4 payload. *Severity: medium.*
5. **Clock anchoring** — local (AM/PM-dropped) versus UTC epoch. *Severity: low;* affects window
   alignment, not feature values.

## 15. Limitations

- Exact numerical parity between a dataset window and a PCAP window is **not provable** without
  running the same flow engine over the same capture scope; the two flow sources are different
  software with different termination rules.
- The demo PCAPs are synthetic and have no ground-truth state CSV, so they test conformance only.
- `pytest` is not installed in `venv/`; `tests/test_feature_parity.py` is written to run
  standalone and is executed by the verifier.
- No packet-level feature (TTL, TCP window size, fragmentation, retransmission) is in scope —
  that is a later roadmap task.

## 16. Final parity status

**Outcome B — approximate semantic equivalence.**

> Training and PCAP feature pipelines are **approximately semantically equivalent**: the window
> aggregation, transforms, relative-change stage and scaler interface are **exactly** equivalent
> (demonstrated on identical flow records across 18 fixtures and 954 comparisons), while the
> flow-record production layer differs, so **numerical equality is not guaranteed**.

The current PCAP demo may legitimately be described as *using the same 52-feature contract,
window rule and transform chain as training*. It may **not** be described as producing feature
values numerically comparable to the training distribution — the flag-rate and scope gaps
measured earlier remain open.
