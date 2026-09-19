# V5 — Packet-Aware Network State

**Task 9.** A versioned, packet-aware experimental feature representation built alongside the frozen V4 reference baseline.

| | |
|---|---|
| **Schema version** | `v5` (contract `1.0.0`) |
| **Input dimension** | **82** = 26 flow base + 26 flow rel-change (both frozen V4) + 21 packet base + 9 packet rel-change (both new) |
| **Schema SHA-256** | `5f52e2d0a7c65e651ba3e948a19c63af108e7b278dddf4e1a7f41b8b21ba6704` |
| **V4 status** | **unchanged** — all four frozen artefacts hash-verified before and after every run |
| **V5 model** | **not trained** — the training dataset cannot be constructed without fabricating data |
| **Status** | **V5 FEATURE PIPELINE COMPLETE — TRAINING DATASET NOT READY** |

> V4 has **not** gained packet-level features. What exists is a separately versioned V5 representation whose feature pipeline is complete, tested and verified, while V4 continues to be the reference baseline, byte-for-byte as it was frozen in Task 1.

---

## 1. Motivation

The SIH problem statement asks for **both** flow-level and packet-level features. V4 has only the first: 26 flow/state features plus their 26 relative changes, every one of them an aggregate over CICFlowMeter flow records.

Aggregation destroys information that only exists per packet. A flow record can tell you that 500 packets moved 60 kB in 4 seconds. It cannot tell you that every one of those packets carried exactly 512 bytes of payload (a fixed-size beacon), that they arrived with 14 different TTLs (many distinct hop distances inside one window), that the receiver advertised a zero window a third of the time (a stall), or that 8% of the segments re-covered sequence space already sent (loss or retry). Those are packet-header facts, and no amount of flow-level arithmetic recovers them.

This task adds five packet-level families that capture behaviours the flow aggregate cannot see. The goal is **not a higher feature count** — 15 of 36 candidate statistics were deliberately rejected — but a defensible, versioned, testable packet-aware representation.

## 2. V4 limitations this addresses

| V4 limitation | What V5 adds |
|---|---|
| No notion of network topology or source distance | TTL mean / std / min / max / distinct-count |
| No receiver flow-control state | TCP window level, dispersion, zero-window ratio, change rate |
| No IP fragmentation visibility at all | fragment count, fragment ratio, DF ratio |
| No loss or retry signal | sequence-space re-coverage count, ratio and bytes |
| Payload size summarised only as a mean-of-flow-means and its std | median, 90th percentile, max, zero-payload ratio, length-distribution entropy |

What V5 does **not** fix: the flow lane itself. The Task 7 finding stands unchanged — see §17.

## 3. Packet feature design

Selection was fixed **before** any evaluation, from four inputs only: definition, redundancy against what V4 already carries, computational cost, and whether the field is actually present in a PCAP. **No held-out test metric was consulted, and none existed when the table was written.**

All 36 candidates, selected and rejected, with the reason for each: [`results/v5_features/feature_design_candidates.csv`](../results/v5_features/feature_design_candidates.csv).

Representative rejections:

- `payload_size_mean`, `payload_size_std` — V4's `mean_pkt_len` / `std_pkt_len` already carry a mean and a spread of the same underlying quantity. Part 11 forbids duplicating a feature because both lanes can compute it. `payload_len_p50` is kept instead, because a robust centre is genuinely different from a mean.
- `tcp_retransmission_rate` — exactly `count / 10`. A rescaling, not a feature.
- `payload_nonzero_ratio` — exactly `1 - payload_zero_ratio`.
- `ttl_p50`, `ttl_p90` — TTL is an 8-bit field with a handful of modes; percentiles add nothing that mean + min/max does not bracket. Percentiles are justified for payload length, which is genuinely long-tailed, and not for TTL.
- `payload_byte_entropy` — content inspection, out of scope, and a **different concept** from `payload_len_entropy` (§8).
- `tcp_dup_ack_count` — would need SACK-option parsing the parser does not do. Deferred rather than shipped weak.
- `tcp_out_of_order_count` — under a high-water-mark rule this is the *same observation* as the retransmission count. Shipping two names for one measurement would be misleading (§7).

## 4. TTL features

| Feature | Definition |
|---|---|
| `ttl_mean` | mean over all IP packets in the window |
| `ttl_std` | sample std, ddof=1; exactly `0.0` for n ≤ 1 |
| `ttl_min` / `ttl_max` | min / max over all IP packets |
| `log_ttl_unique_count` | `log1p(count of distinct TTL values)` |

**IPv4 TTL and IPv6 Hop Limit are pooled into one statistic.** Both are 8-bit remaining-hop-budget counters with identical semantics, range and decrement rule, so a pooled statistic is well defined. This is a semantic merge, not an implementation detail, so it is recorded explicitly in the contract. IPv6 is 151 of 303,922 packets (0.05%) in the real capture.

**Protocol filtering**: none — every IP packet contributes, including non-first fragments (which carry a TTL but no L4 header) and non-TCP traffic. **Empty window**: no row is emitted at all. **Missing value**: impossible; a parsed IP packet always has the field.

## 5. TCP window features

| Feature | Definition |
|---|---|
| `log_tcp_window_mean` | `log1p(mean(window))` over TCP packets |
| `log_tcp_window_std` | `log1p(std(window, ddof=1))` |
| `log_tcp_window_max` | `log1p(max(window))` |
| `tcp_window_zero_ratio` | `count(window == 0) / count(TCP packets)` |
| `tcp_window_change_rate` | see below |

**`tcp_window_change_rate` — the exact definition**, stated because "change rate" is otherwise ambiguous:

```
numerator   = TCP packets in the window whose raw window value differs from
              the previous packet of the SAME directional session
denominator = TCP packets in the window that HAVE such a predecessor
```

The denominator is *not* all TCP packets: the first packet of every session has nothing to compare against and is excluded from both terms. A pair straddling a window boundary is attributed to the **later** packet's window only, never to the earlier one — this is what keeps the feature causal (fixture `15_window_boundary_causality`).

**Raw header units.** The TCP window-scale option is deliberately **not** applied: the scale factor is only observable when the SYN handshake is inside the capture, which is false for every mid-capture connection. Applying it to some connections and not others would make the feature inconsistent between windows.

**Edge cases**: non-TCP packets never enter any numerator or denominator. SYN packets are ordinary TCP packets here. Retransmitted packets are counted normally — a retransmission still advertises a window. Zero-window packets are counted in the mean/std/max *and* in `tcp_window_zero_ratio`. A packet with no TCP header (non-first fragment) is not TCP. **No TCP in the window → all five features are `0.0`.**

## 6. IP fragmentation features

| Feature | Definition |
|---|---|
| `log_frag_packet_count` | `log1p(count of fragmented IP packets)` |
| `frag_packet_ratio` | `count(fragmented) / count(IP packets)` |
| `ip_df_ratio` | `count(IPv4 with DF) / count(IPv4 packets)` |

**A fragment is recognised per RFC 791, not by "any nonzero flag bit"** — the DF bit is set on 96% of real packets and means the *opposite* of fragmentation:

| State | MF | offset |
|---|---|---|
| First fragment | set | 0 |
| Middle fragment | set | > 0 |
| Last fragment | clear | > 0 |
| Unfragmented | clear | 0 |

`is_fragment = MF set OR offset > 0`, which covers first, middle and last.

**IPv6**: routers may never fragment IPv6, so fragmentation is read *only* from the Fragment extension header (its `M` flag and offset). An IPv6 packet without that header is unfragmented by definition. **IPv6 is excluded from `ip_df_ratio` entirely** — both numerator and denominator are IPv4-only — because IPv6 has no DF bit, and counting IPv6 as DF=0 would fabricate a value.

> ⚠️ **This family is implemented and fixture-verified, but carries zero variance in the data available to this project.** Across 303,922 real IP packets there are **0 fragments** — no MF flag set, no nonzero offset, anywhere. `log_frag_packet_count` and `frag_packet_ratio` are therefore constant `0.0` on every capture on disk. `ip_df_ratio` is the only fragmentation feature with real variance (0.96 ± 0.09). This is reported rather than hidden, and the verifier does not pretend the data supports what it does not.

## 7. Retransmission features

| Feature | Definition |
|---|---|
| `log_tcp_retrans_count` | `log1p(count of sequence-space re-covering packets)` |
| `tcp_retrans_ratio` | `count(retransmissions) / count(seq-bearing TCP packets)` |
| `log_tcp_retrans_bytes` | `log1p(sum of retransmitted payload bytes)` |

**Detection rule.** Let `eff_len = payload_len + (1 if SYN) + (1 if FIN)` — the number of sequence numbers the segment consumes, which is TCP's own definition. A packet is a retransmission when **all** of:

1. it is TCP;
2. `eff_len > 0`;
3. it is not the first `eff_len > 0` packet of its directional session;
4. `seq + eff_len` is at or below that session's running high-water mark, under **32-bit modular comparison**.

**Scope and directionality.** A session is the **ordered** 5-tuple `(src_ip, src_port, dst_ip, dst_port, protocol)` — A→B and B→A are separate sessions, which is what TCP sequence space actually is: each endpoint has its own independent numbering. A session resets after a 120 s idle gap, the same timeout `pcap_ingest_v4.reconstruct_flows` already uses, so a reused ephemeral port with a fresh ISN does not read as a retransmission.

**Sequence handling.** Comparison is modular (`_seq_diff`), so a sequence that has wrapped past 2³² is still ordered correctly. Fixture `14_seq_wraparound` constructs a case a naive unsigned comparison would miss.

**Zero-payload packets: condition 2 is the load-bearing exclusion.** A pure ACK, window update, bare RST or keepalive has `eff_len == 0` and is **never** counted. Duplicate pure ACKs are duplicate ACKs, not retransmissions, and conflating them is the most common way this metric is made misleading. Fixture `08_tcp_retransmission` includes two identical pure ACKs that must not fire, and `test_duplicate_pure_acks_are_not_retransmissions` guards it directly.

**SYN and FIN.** Each consumes one sequence number, so `eff_len = 1` and SYN/FIN retransmissions **are** detected — the case that matters for scan and flood traffic. Because their payload is zero, they produce `count > 0` with `bytes == 0`, which is exactly why the count and the byte total are both kept (fixture `16_syn_retransmission`).

**RST.** A bare RST carries no payload and no SYN/FIN, so `eff_len == 0` and it is not counted.

**Overlapping retransmissions** are counted only when the segment is *wholly* re-covered by already-transmitted space. A segment extending past the high-water mark is treated as new data. This under-counts rather than over-counts — the conservative direction.

> ⚠️ **Named limitation.** A genuinely out-of-order packet is indistinguishable from a retransmission under any high-water-mark rule: both re-cover already-seen sequence space. Separating them needs RTT estimation and duplicate-ACK context that a streaming per-window builder cannot afford. **The feature therefore measures *sequence-space re-coverage* = retransmission + reordering, and is documented as that rather than being claimed as a pure retransmission count.** On real traffic it fires on 1.4% of seq-bearing packets, in 25% of windows.

## 8. Payload-size features

| Feature | Definition |
|---|---|
| `payload_len_p50` | 50th percentile of payload length over IP packets |
| `payload_len_p90` | 90th percentile |
| `payload_len_max` | maximum |
| `payload_zero_ratio` | `count(payload_len == 0) / count(IP packets)` |
| `payload_len_entropy` | Shannon entropy (base 2) of the **length** distribution |

> **`payload_len_entropy` is a payload-LENGTH distribution entropy. It is NOT byte-value Shannon entropy.** No payload byte is read anywhere in this pipeline — the only input is the integer length from the IP/L4 headers. The two are routinely confused and measure different things: byte-value entropy detects encryption or compression of *content*; length entropy detects *regularity of message sizes*. A window where every packet carries 512 bytes has length entropy exactly 0.0 — the fixed-size beacon signature — regardless of what those bytes contain.

The formula is identical to V4's `entropy_from_counts` (used for `dst_port_entropy` and `protocol_entropy`), including the `1e-12` inside the log and the exact `0.0` for ≤ 1 distinct value.

**Payload length comes from header arithmetic** (IP total length − IP header − L4 header), not from `len(layer.payload)`. The latter — which V4 uses — silently includes Ethernet padding on sub-60-byte frames. V5 needs the true segment length because it also drives the sequence accounting in §7. It falls back to `len(layer.payload)` when the IP total length is unusable (0 under TSO/LRO offload, or negative arithmetic from a truncated snaplen).

`payload_len_max` reaches 64,240 on the real captures — the capture point sees TSO/LRO-coalesced segments larger than the link MTU. That is a property of *where the capture was taken*, not a parsing error, and it is left unclipped.

## 9. Feature formulas

Every formula, unit, domain, missing-value rule, zero-value behaviour, dependency and relative-change ruling is in [`docs/v5_packet_feature_contract.json`](v5_packet_feature_contract.json). The full 82-dimensional layout is in [`docs/v5_feature_schema.json`](v5_feature_schema.json).

```
index  0-25   flow base                 (frozen V4)
index 26-51   flow relative change      (frozen V4)
index 52-72   packet base               (new — 21)
index 73-81   packet relative change    (new — 9)
```

**The V4 block is a prefix, not a reshuffle.** `v5_input[:52]` is byte-identical to the vector the frozen V4 model consumes, in the frozen V4 order, produced by the unmodified V4 code. That makes "V4 is unchanged" a numerically checkable claim, and it lets a future V4-vs-V5 comparison isolate the effect of the packet block.

## 10. Dataset construction

> ### 🛑 A V5 packet-aware training dataset **cannot** be constructed.

| Benchmark day | Role | Windows | PCAP hosts on disk | Packet features constructible |
|---|---|---|---|---|
| Wednesday-14-02-2018 | train / val | 3,213 | **0** | **no** |
| Thursday-15-02-2018 | train / val | 3,387 | **0** | **no** |
| Wednesday-28-02-2018 | train / val | 3,314 | **0** | **no** |
| Friday-02-03-2018 | test | 3,115 | 20 of 440 (4.5%) | partial |

The three TRAIN/VAL days have **zero packet captures**. Only the pre-computed CICFlowMeter flow CSVs were ever obtained for them, and **a flow CSV cannot yield TTL, TCP window, IP fragment flags or TCP sequence numbers — those fields do not exist in it at any aggregation level.**

The four options in Part 17 were considered:

| Option | Viable | Why |
|---|---|---|
| **A** Reprocess original PCAPs | ✗ | There are none for the three train/val days. For 02-03 the archive is 41.7 GiB compressed / 54.7 GB uncompressed against ~49 GB free, and downloading it would *still* leave train and val empty. |
| **B** Join packet features where PCAP exists | ✗ | Would populate the test day only. A model cannot learn a feature it never sees in training, filling the gap would be exactly the unsupported imputation Part 17 forbids, and the packet block would become a perfect test-set indicator — a leakage channel, not a feature. |
| **C** Packet features for PCAP demonstration only | ✓ **selected** | The packet lane is fully built, contracted, tested and verified, and runs on real captures. No V5 model is trained. |
| **D** Re-derive the flow lane from PCAP for all four days | ✗ | Needs the full archive for four days, three of which were never captured; would also change the flow features, destroying the controlled V4-vs-V5 comparison. |

**No packet value was fabricated, defaulted or imputed anywhere. No partial dataset was written.**

What would unblock it: packet captures for the three February days at the monitored-organisation scope the flow CSVs cover; ~55 GB of disk per day (or a batch extract-fold-delete loop); and a re-derivation of the 26 flow features from those same captures so that flow and packet features share one capture scope.

## 11. Window semantics

Identical bin width, origin and boundary rule to V4:

```
window_idx = floor(packet_epoch_seconds / 10) - win0      interval [t, t+10)
```

`win0` is the bin of the capture's first packet — the same re-basing `pcap_ingest_v4.reconstruct_flows` applies, so the packet-window table and the flow-window table share an index. Floor, never round-half; no tolerance band.

**Timestamp source**: the pcap per-packet capture timestamp. **Timezone**: UNCHANGED from V4. Packet clocks are true UTC epoch seconds; the CSE-CIC-IDS2018 CSV clock is Atlantic local (UTC−4 on 2018-03-02) with the AM/PM marker dropped by the dataset authors. **V5 introduces no timezone conversion anywhere.** Bin width and boundary rule are identical across paths, but the two clocks are not co-anchored — this matters only when aligning a PCAP window to a dataset window and changes no feature *value*.

**Empty windows** produce no row, exactly as V4 does; the downstream 10-second contiguity gate rejects any sequence spanning the gap. **First window** is kept as-is with its relative-change row forced to `0.0`. **Last window** is kept as-is; no packet feature ever reads forward.

### One deliberate difference from the V4 flow layer

V4 assigns a **whole flow** — all of its statistics — to the window of its **first packet**. The packet layer assigns **each packet** to the window containing **its own timestamp**.

This is not an inconsistency; it is the only assignment rule that is causally sound for a packet statistic. The flow rule would let packets arriving after `t+10` contribute to the state at `t` — exactly the leakage Part 19 forbids. The bin width, origin and boundary rule are identical between the two layers; only the assignment *unit* differs, and that difference is what makes the packet layer causal.

### Deterministic packet ordering

Capture timestamps are not unique, and the session-state annotations depend on observation order, so a timestamp-only sort leaves the result dependent on how rows happened to be arranged. Every packet therefore carries `packet_index`, its capture-order ordinal, and every stage sorts on `(timestamp_abs, packet_index)`. Within one timestamp the capture order is the best available statement of what the sensor saw first.

*(This was found by the parity harness, not assumed: the first full run produced 23/30 bit-identical features on the real capture because a shuffled storage round-trip reordered packets sharing a microsecond. The builder was fixed — the test was not weakened.)*

## 12. Relative changes

The transform is byte-for-byte the V4 definition:

```
r_t = (s_t - s_{t-1}) / (|s_{t-1}| + 1e-8)
first row forced to exactly 0.0 ·  non-finite -> 0.0 ·  clip to [-10, +10]
```

"Previous window" means the previous **existing row** — not necessarily `t-10 s`, because empty windows produce no row. Reset scope is one capture (PCAP) / one source day (training). Computed on the already-log-transformed base feature. **The 26 V4 relative-change features are untouched and not re-derived.**

**Not every new base feature gets one. 9 of 21 do.** Each of the 21 has an explicit ruling with a reason in the contract:

| Included (9) | Why |
|---|---|
| `ttl_mean`, `ttl_std`, `log_ttl_unique_count` | Stable denominators in a non-empty window; a shift in hop-distance population *is* the transition signal. |
| `log_tcp_window_mean`, `log_tcp_window_std` | Flow-control state is inherently about change — a level alone cannot distinguish steadily-small from just-collapsed. |
| `log_tcp_retrans_count` | Retransmission is bursty and the onset of a burst is a transition. Well populated (25% of windows), so not an all-zero series. |
| `payload_len_p50`, `payload_len_p90` | Multiplicative shifts in message size mark the start and end of bulk transfer. |
| `payload_len_entropy` | Consistent with the frozen V4 convention for `dst_port_entropy` / `protocol_entropy`. An entropy collapse is a transition, not a level. |

| Excluded (12) | Why |
|---|---|
| `ttl_min`, `ttl_max`, `log_tcp_window_max`, `payload_len_max` | Extreme order statistics; a single packet moves them, so the ratio is noise-dominated. The **level** is the informative part. |
| `tcp_window_zero_ratio`, `tcp_retrans_ratio`, `payload_zero_ratio`, `ip_df_ratio` | Bounded proportions, usually exactly 0 (or near-constant). Their relative change is `0/0` → epsilon-driven clip saturation, which encodes *any* arrival as a constant +10 regardless of magnitude. The base ratio conveys the same arrival *with* magnitude and without the artefact. |
| `tcp_window_change_rate` | Already a first-order change quantity; its relative change is a second difference — unstable and hard to interpret. |
| `log_frag_packet_count`, `frag_packet_ratio` | Structurally zero in every available window; a relative change over an all-zero series is identically 0.0 — a dead dimension. |
| `log_tcp_retrans_bytes` | Near-collinear with the count's relative change; only the count's version is kept. |

## 13. Numerical handling

| Condition | Behaviour |
|---|---|
| Window with no IP packets | **no row** (identical to V4) |
| No TCP packets | all 5 TCP-window + all 3 retransmission features → `0.0` |
| No eligible window-change pair | `tcp_window_change_rate` → `0.0` |
| No seq-bearing TCP packet | `tcp_retrans_ratio` → `0.0` |
| No IPv4 packet | `ip_df_ratio` → `0.0` |
| No fragmented packet | `log_frag_packet_count` = `log1p(0)` = `0.0` |
| No payload at all | `p50 = p90 = max = 0.0`, `zero_ratio = 1.0`, `entropy = 0.0` |
| Constant TTL | `ttl_std = 0.0` exactly; `log_ttl_unique_count = log1p(1)` |
| Constant TCP window | `log_tcp_window_std = 0.0` |
| n ≤ 1 | every std is exactly `0.0` (ddof=1 convention, matching V4's `pkt_len_std`) |

**`0.0` is the absent encoding**, mirroring V4's `replace([inf,-inf], nan).fillna(0.0)`. The ambiguity it creates ("absent" vs "genuinely zero") is real; `packet_count` is carried alongside so a consumer can separate the cases, and every feature's `zero_value_behaviour` field states which reading applies.

**Divisions use `_ratio`, which returns exactly `0.0` on a zero denominator** — *not* V4's epsilon-inflated `x / (|y| + 1e-8)`. The packet denominators are exact integer packet counts, so "no TCP packets in this window" is a real, enumerable, testable condition rather than something to smear with an epsilon. This is a deliberate V5 policy and changes nothing about the 26 V4 features.

Percentiles pin `method='linear'` explicitly, so a numpy default change cannot silently move every p50/p90. The 21 features are `replace([inf,-inf], nan).fillna(0.0)` before leaving the builder. **Result: 0 NaN and 0 non-finite values across all 180 (capture, feature) conformance cells.**

## 14. Training ↔ PCAP parity

**The training path does not consume raw packets at all** — it consumes CICFlowMeter flow CSVs, which have no packet fields. Per Part 15, the resolution is a single canonical builder used by both sides: [`src/packet_feature_builder.py`](../src/packet_feature_builder.py) is the only implementation of every packet formula, called by both the training-side and the PCAP-side entry point.

Parity is therefore **EXACT by construction**. The harness still verifies it numerically across two independent invocation routes — in-memory straight from the parser, versus a CSV round-trip plus a row shuffle, as packets would arrive from storage:

**180 / 180 (capture, feature) comparisons bit-identical** across six captures, 3,193 windows. Per-feature detail: [`results/v5_features/packet_parity_results.csv`](../results/v5_features/packet_parity_results.csv) and [`results/v5_features/parity/`](../results/v5_features/parity/).

> **What this does and does not prove.** It proves the two routes agree and that no serialisation, dtype or ordering drift exists — and it *did* catch a real determinism bug (§11). It does **not** prove agreement with a second independent packet-feature implementation, because no such implementation exists to compare against.

## 15. Leakage controls

Five checks, all PASS ([`results/v5_features/leakage_check.json`](../results/v5_features/leakage_check.json)):

1. **Packet window assignment** — every packet contributes only to the window containing its own timestamp.
2. **Prefix-truncation invariance** — the decisive test. Rebuild from a strict prefix of the capture; every already-completed window must be **bit-identical**. Measured drift: `0.0` on all six captures, 3,193 windows.
3. **Relative change is backward-only** — uses the immediately preceding existing row and nothing else; first row exactly `0.0`.
4. **Flow continuation cannot edit the past** — packet features are assigned per packet, so a flow continuing past `t+10` contributes only to the later window. This is the structural difference from the V4 flow lane (§11).
5. **Session state is causal** — `annotate_packets` is a single forward pass in timestamp order; every annotation is written once and never revised. Verified *per packet* by `test_annotations_depend_only_on_the_past`, which rebuilds from every prefix and requires identical annotations.

No future window packet count, future packet timing, future label or future flow-completion information enters any state.

## 16. Dataset limitations

1. **No V5 training dataset exists.** Three of four benchmark days have no packet captures (§10).
2. **IP fragmentation is structurally absent** from all available data — 0 fragments in 303,922 real IP packets. Two of three fragmentation features are constant `0.0` here.
3. **Even the test day is partial-scope**: 20 of 440 host captures (4.5%), while the flow CSV aggregates the whole monitored organisation.
4. **Retransmission = retransmission + reordering** (§7).
5. **TCP window is raw header units**; window scaling is not applied (§5).
6. **The five demo captures are synthetic** — constant TTL 64 and constant window 8192 — so 18–22 of 30 features are necessarily constant on them. That is a property of those files, not of the features: the real capture shows 67 distinct TTLs and 1,399 distinct window values, and 19 of 21 base features carry non-zero values across its 3,083 windows.

## 17. V4 ↔ V5 version boundary

**V4 is unchanged.** All four frozen artefacts are hash-verified at the start *and* the end of every pipeline run; a mismatch aborts before anything is written.

| Artefact | SHA-256 |
|---|---|
| `models/latent_world_model_v4/latent_world_model_v4.pt` | `76af8d8a…5e5898` |
| `data/processed/multiday_benchmark_K5/input_scaler.pkl` | `e0ab9beb…653f04f19` |
| `docs/v4_feature_schema.json` | `2b4f820c…e094c4` |
| `docs/v4_baseline_manifest.json` | `b2765c24…13937e` |

Nothing under `models/latent_world_model_v4/`, `data/processed/multiday_benchmark_K5/`, `docs/v4_*` or `results/standardized_baseline_comparison.csv` was written. The frozen 52-dimensional scaler is **never** used to transform an 82-dimensional vector; V5 declares its scaler as *pending*, because a V5 scaler can only be fitted on a V5 training partition that does not exist.

### The hybrid representation (Part 20)

| Lane | Status |
|---|---|
| **Flow features** (26 + 26) | **APPROXIMATE** relative to CICFlowMeter training semantics — the Task 7 finding, preserved unchanged and not re-litigated. |
| **Packet features** (21 + 9) | **DIRECT** — computed from packet timestamps and header fields, with no flow-engine intermediary. |

**No claim of full numerical equivalence with CICFlowMeter is made anywhere.** V5 is explicitly a hybrid.

## 18. Optional V5 model experiment

**Not performed.** Part 22 forbids training until a training, validation *and* test set with populated packet features all exist. Train and val have zero packet captures, so the prerequisite fails and training was not attempted.

**No V5 checkpoint, config, scaler, training history or prediction file was created.** `models/world_model_v5/` does not exist.

When the data does exist, the architecture is already specified and must be held fixed to isolate the effect of the packet features: the V4 architecture verbatim — `10 × 82` input → 2-layer LSTM → LayerNorm → **32**-dimensional latent → gated residual transition → LayerNorm → decoder → attack head. No latent-dimension increase, no Transformer, no GNN, no rollout change. Only the input dimension moves, from 52 to 82.

## 19. Verification

```
./venv/bin/python src/run_v5_packet_features.py       # build every artefact
./venv/bin/python src/verify_v5_packet_features.py    # 14 checks
./venv/bin/python tests/test_v5_packet_features.py    # 23 tests, 16 fixtures
```

**`verify_v5_packet_features.py` — OVERALL: PASS** (14/14). A family passes only when it is implemented **and** its contract formula is **independently re-derived** in the verifier, in plain python written from the contract text, and matches the builder to 1e-9 on every window of a real capture. A family that could not be implemented reliably would be reported FAIL and would drag OVERALL to FAIL — it is never quietly downgraded.

**`tests/test_v5_packet_features.py` — 23/23 pass.** Sixteen deterministic fixtures in [`tests/fixtures/v5_packet_features/`](../tests/fixtures/v5_packet_features/) cover all twelve mandatory scenarios plus multi-window relative change, sequence wraparound, window-boundary causality and SYN retransmission. **Expected values come from an independent reference implementation** (`_generate_fixtures.py`, plain `math`/`collections`) which **never imports the module under test** — nothing is self-referential.

## 20. Remaining limitations

1. **No V5 training dataset, therefore no V5 model and no V4-vs-V5 result.** The single blocking item.
2. **Fragmentation is untested against real fragmented traffic** — verified only against synthetic fixtures.
3. **Retransmission conflates reordering**; separating them needs RTT and duplicate-ACK context.
4. **Duplicate-ACK counting is deferred** pending TCP-option (SACK) parsing.
5. **Parity is by shared implementation**, not by two independent implementations (§14).
6. **The flow lane remains approximate** relative to CICFlowMeter (Task 7, unchanged), so V5 would inherit that gap alongside its direct packet features.
7. **Packet capture scope is partial even on the one day it exists** — 4.5% of hosts.
8. **The packet features have never been evaluated for predictive value.** They are designed, contracted, implemented, tested and verified. Whether they help is an open question that cannot be answered until item 1 is resolved — and answering it must not feed back into the feature selection, which is why that selection was fixed and hashed first.

---

### Artefacts

| Path | Contents |
|---|---|
| [`docs/v5_packet_feature_contract.json`](v5_packet_feature_contract.json) | Full definition of the 30 new packet features |
| [`docs/v5_feature_schema.json`](v5_feature_schema.json) | The 82-dimensional schema, categorised flow vs packet |
| [`docs/v5_packet_dataset_manifest.json`](v5_packet_dataset_manifest.json) | Per-record provenance; dataset verdict |
| [`results/v5_packet_feature_summary.json`](../results/v5_packet_feature_summary.json) | Machine-readable summary |
| [`results/v5_features/`](../results/v5_features/) | Candidates, parity, conformance, availability, leakage, hash, figures |
| [`src/packet_feature_builder.py`](../src/packet_feature_builder.py) | The canonical builder — one implementation, both paths |
| [`src/v5_feature_contract.py`](../src/v5_feature_contract.py) | Contract assembly and schema hash |
| [`src/run_v5_packet_features.py`](../src/run_v5_packet_features.py) | Pipeline orchestrator |
| [`src/verify_v5_packet_features.py`](../src/verify_v5_packet_features.py) | The 14 checks |
