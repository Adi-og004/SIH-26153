# Official CICFlowMeter validation — run kit (NOT executed in this environment)

## Why nothing ran here

This workstation is **Apple M4 / macOS arm64** with **no container runtime** (`docker`,
`colima`, `lima`, `qemu` all absent), **no `mvn`/`gradle`**, and **JDK 26** (not 8).
Official CICFlowMeter needs **x86_64 Linux + JDK 8 + jnetpcap 1.4.r1425 (x86 `.so`) + libpcap**.
`jnetpcap 1.4` has **no arm64/macOS native library**, so native execution is impossible,
and provisioning a paid cloud x86 VM was not authorised. The steps below are a pinned,
ready-to-run recipe; run them on any x86_64 Linux host / CI runner, then bring the
`*_Flow.csv` files back (or run the comparison there).

## Pinned versions

| Component | Value |
|---|---|
| CICFlowMeter repo | `https://github.com/ahlashkari/CICFlowMeter` |
| CICFlowMeter commit | `98a5ebad0df579cc8b43eedd3421b3ae87699901` (master @ 2026-09-10; **repo has no release tags**) |
| Dataset-of-record engine | CICFlowMeter **V3** (CSE-CIC-IDS2018). No pinned V3 in the official repo. For a V3 cross-check: `iPAS/TCPDUMP_and_CICFlowMeter` @ `8623d91fa723a865cd8d8d407be7e6493e619e0b`, branch `comparison_of_cicflowmeter_versions` @ `98630d4d2eccba975ed8dbb15e19407c8c27c441` |
| Java | JDK **8** (Temurin `eclipse-temurin:8-jdk-jammy`) |
| jnetpcap | `org.jnetpcap:jnetpcap:1.4.1`, native `jnetpcap-1.4.r1425` (bundled in the repo, x86_64 `.so`) |
| libpcap | host `libpcap0.8` + `libpcap-dev` (Ubuntu 22.04 "jammy") |
| OS / arch | **linux/amd64** (x86_64) only |
| CLI entrypoint | `cic.cs.unb.ca.ifm.Cmd <pcap_file_or_dir> <out_dir>` |
| Timeouts (from `Cmd.java`) | `flowTimeout = 120_000_000 µs` (120 s), `activityTimeout = 5_000_000 µs` (5 s) |

## Commands

```bash
# on an x86_64 Linux host (or:  docker buildx  with --platform linux/amd64 on any host)
cd models/pcap_analysis_v4/parity/cicflowmeter
docker build --platform linux/amd64 -t cfm-pinned .

# 3. verify the executable starts
docker run --rm --platform linux/amd64 cfm-pinned java \
    -Djava.library.path=/opt/CICFlowMeter/jnetpcap/linux/jnetpcap-1.4.r1425 \
    -cp "/opt/CICFlowMeter/build/classes/java/main" cic.cs.unb.ca.ifm.Cmd
#   expected: logs "Please select pcap!" then exits 0  == JNI + libpcap load OK

# 4-6. run on the 21 host pcaps  (SAFEST = per-file, NOT one merged pcap: CICFlowMeter
#      keys flows within a single capture; merging first can fuse flows that a
#      per-host capture kept separate, and changes first-packet direction for
#      asymmetrically-seen flows. Per-file + downstream 5-tuple dedup is the
#      reproducible choice.)
mkdir -p pcaps flows
cp ../../../../data/raw/march02/subset25/pcap/*                pcaps/     # 14 non-victim
cp ../../../../data/raw/march02/bot_pcaps/pcap/*               pcaps/     # 10 victims
#   NOTE: capEC2AMAZ-O4EL3NG-172.31.66.70 on disk is TRUNCATED (45,350,912 B vs 84,589,146) —
#         delete or re-fetch it before running; otherwise it yields a 20-host set.
docker run --rm --platform linux/amd64 \
    -v "$PWD/pcaps":/pcap -v "$PWD/flows":/flow cfm-pinned /pcap /flow
#   -> flows/<name>_Flow.csv  per input pcap

# 7-13. field check + STEP-8 flag clip + create_network_states_v2 aggregation + compare
python aggregate_and_compare.py  ./flows  ./pcaps
#   writes:  flows/states10_cicflowmeter.csv
#            flows/states10_emulation.csv
#            flows/cicflowmeter_vs_emulation_vs_training.csv
#            flows/cicflowmeter_compare_summary.json
```

## STEP 8 — flag clipping evidence (verified against the training CSV)

`data/raw/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv`, all 1,048,575 rows:

| column | min | max | #values > 1 |
|---|--:|--:|--:|
| FIN / SYN / RST / PSH / ACK / URG Flag Cnt | 0 | **1** | **0** |
| CWE Flag Count / ECE Flag Cnt | 0 | 1/0 | 0 |
| Fwd PSH Flags / Fwd URG Flags | 0 | 1/0 | 0 |
| **Down/Up Ratio** (same file, for contrast) | 0 | **148** | **9,328** |

Native CICFlowMeter emits `* Flag Cnt` as running per-packet counters (`BasicFlow.java`:
`flagCounts.get("PSH").increment()`), so real flows have PSH/ACK counts of tens–hundreds.
The strict `{0,1}` in the training CSV — while `Down/Up Ratio` in the same file keeps real
integers up to 148 — means CIC applied a **column-specific `clip(0,1)` (or `>0`) to the flag
family** when building the ML CSVs. `aggregate_and_compare.py` re-applies exactly that clip to
the CICFlowMeter output, **after** CICFlowMeter, **before** `create_network_states_v2`
aggregation, and labels it a reconstruction of CIC preprocessing — not CICFlowMeter behaviour.
`create_network_states_v2.py` consumes only the six FIN/SYN/RST/PSH/ACK/URG counts.

## Risks carried into any result produced this way

- V4 repo ≠ the V3 CIC actually ran → not bit-identical to the 2018 CSVs.
- The flag clip is inferred, not from a published CIC script.
- jnetpcap 1.4 (abandoned, x86-only) is ABI-fragile on modern libpcap; the Dockerfile pins
  `libpcap0.8` on jammy to mitigate.
- Per-file vs merged-pcap processing changes flow boundaries; per-file + dedup chosen.
