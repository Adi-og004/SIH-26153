"""
DOWNSTREAM HARNESS -- consumes official CICFlowMeter `*_Flow.csv` output for the
21-host March-2 subset and builds the 10-second network-state distribution the
SAME WAY the V4 training data was built (archive/create_network_states_v2.py),
then compares it against:
    A. March-2 training states_v2            (genuine CIC CICFlowMeter, whole org)
    B. our corrected pcap_ingest_v4 emulation (same 21 pcaps, recomputed here)
    C. the previous 21-host emulation result  (models/.../subset25_z_comparison.csv)

RUN THIS ON THE MACHINE THAT PRODUCED THE CICFlowMeter CSVs (x86_64 Linux), OR copy
the `*_Flow.csv` files back here.  It does NOT touch the V4 model.

    python aggregate_and_compare.py  <dir_of_Flow_csvs>  <dir_of_21_pcaps>

STEP 8 clipping: the six `* Flag Cnt` fields are clipped to [0,1] because every one
of them is strictly {0,1} across all 1,048,575 rows of the training CSV
(#>1 == 0 for FIN/SYN/RST/PSH/ACK/URG/CWE/ECE), while `Down/Up Ratio` in the SAME
file has 72 distinct values up to 148 (9,328 rows > 1).  Native CICFlowMeter emits
`* Flag Cnt` as running per-packet counters, so the {0,1} in the training CSV is a
CIC ML-data preprocessing step, NOT CICFlowMeter behaviour.  This clip RECONSTRUCTS
that preprocessing; it is applied here, after CICFlowMeter, before aggregation.
"""
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
import pcap_ingest_v4 as P

BASE = P.BASE_FEATURES
WIN = 10
FLAG_CNT = ["FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt",
            "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt"]

# ---- create_network_states_v2.py helpers (verbatim behaviour) ----
def safe_log1p(s):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    return np.log1p(s)
def safe_divide(n, d):
    r = n / (d.abs() + 1e-8)
    return r.replace([np.inf, -np.inf], np.nan).fillna(0.0)
def entropy_from_counts(s):
    c = s.value_counts(dropna=True)
    if len(c) <= 1: return 0.0
    p = c.to_numpy(float) / c.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))

def cicfm_csv_to_states(df):
    """archive/create_network_states_v2.py, applied to a CICFlowMeter flow frame
    (already flag-clipped, already deduped)."""
    num = ["Protocol","Dst Port","Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts",
           "TotLen Fwd Pkts","TotLen Bwd Pkts","Pkt Len Mean","Pkt Len Std",
           "Fwd IAT Mean","Bwd IAT Mean","Down/Up Ratio"] + FLAG_CNT
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    df["total_packets"] = df["Tot Fwd Pkts"].fillna(0) + df["Tot Bwd Pkts"].fillna(0)
    df["total_bytes"]   = df["TotLen Fwd Pkts"].fillna(0) + df["TotLen Bwd Pkts"].fillna(0)
    df["fwd_packet_ratio"] = safe_divide(df["Tot Fwd Pkts"].fillna(0), df["total_packets"])
    df["bwd_packet_ratio"] = safe_divide(df["Tot Bwd Pkts"].fillna(0), df["total_packets"])
    df["fwd_byte_ratio"]   = safe_divide(df["TotLen Fwd Pkts"].fillna(0), df["total_bytes"])
    df["bwd_byte_ratio"]   = safe_divide(df["TotLen Bwd Pkts"].fillna(0), df["total_bytes"])
    pden = df["total_packets"] + 1e-8
    df["syn_rate"] = df["SYN Flag Cnt"].fillna(0) / pden
    df["ack_rate"] = df["ACK Flag Cnt"].fillna(0) / pden
    df["rst_rate"] = df["RST Flag Cnt"].fillna(0) / pden
    df["fin_rate"] = df["FIN Flag Cnt"].fillna(0) / pden
    df["psh_rate"] = df["PSH Flag Cnt"].fillna(0) / pden
    df["urg_rate"] = df["URG Flag Cnt"].fillna(0) / pden
    df["window_start"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True).dt.floor("10s")
    g = df.groupby("window_start", sort=True)
    st = g.agg(
        flow_count=("Timestamp","size"),
        total_packets=("total_packets","sum"), total_bytes=("total_bytes","sum"),
        mean_flow_duration=("Flow Duration","mean"),
        mean_fwd_iat=("Fwd IAT Mean","mean"), mean_bwd_iat=("Bwd IAT Mean","mean"),
        mean_pkt_len=("Pkt Len Mean","mean"), std_pkt_len=("Pkt Len Std","mean"),
        syn_rate=("syn_rate","mean"), ack_rate=("ack_rate","mean"),
        rst_rate=("rst_rate","mean"), fin_rate=("fin_rate","mean"),
        psh_rate=("psh_rate","mean"), urg_rate=("urg_rate","mean"),
        fwd_packet_ratio=("fwd_packet_ratio","mean"), bwd_packet_ratio=("bwd_packet_ratio","mean"),
        fwd_byte_ratio=("fwd_byte_ratio","mean"), bwd_byte_ratio=("bwd_byte_ratio","mean"),
        mean_down_up_ratio=("Down/Up Ratio","mean"),
        unique_dst_ports=("Dst Port","nunique"), unique_protocols=("Protocol","nunique"),
    ).reset_index()
    st["log_total_packets"]      = safe_log1p(st["total_packets"])
    st["log_total_bytes"]        = safe_log1p(st["total_bytes"])
    st["log_flow_count"]         = safe_log1p(st["flow_count"])
    st["log_mean_flow_duration"] = safe_log1p(st["mean_flow_duration"])
    st["log_mean_fwd_iat"]       = safe_log1p(st["mean_fwd_iat"])
    st["log_mean_bwd_iat"]       = safe_log1p(st["mean_bwd_iat"])
    st["log_unique_dst_ports"]   = safe_log1p(st["unique_dst_ports"])
    st["window_packet_rate"]     = st["total_packets"] / WIN
    st["window_byte_rate"]       = st["total_bytes"] / WIN
    st["dst_port_entropy"] = g["Dst Port"].apply(entropy_from_counts).values
    st["protocol_entropy"] = g["Protocol"].apply(entropy_from_counts).values
    st = st.replace([np.inf, -np.inf], np.nan)
    st[BASE] = st[BASE].fillna(0.0)
    return st

def canon_key(r):
    a, b = sorted([str(r["Src IP"]), str(r["Dst IP"])])
    lo, hi = sorted([int(r["Src Port"]), int(r["Dst Port"])])
    return f"{a}|{b}|{lo}|{hi}|{int(r['Protocol'])}"

def main(flow_dir, pcap_dir):
    files = sorted(glob.glob(str(Path(flow_dir) / "*_Flow.csv")))
    assert files, f"no *_Flow.csv in {flow_dir}"
    frames = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        d["_srcfile"] = Path(f).name
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    print(f"CICFlowMeter rows (all {len(files)} files): {len(raw):,}")

    # ---- CICFlowMeter V4 header -> V3/training header aliases (mechanical rename,
    #      NOT a value change). The V4.0 repo renamed many columns vs the V3 the
    #      CSE-CIC-IDS2018 CSVs used.
    V4_TO_V3 = {
        "Total Fwd Packet": "Tot Fwd Pkts",
        "Total Bwd packets": "Tot Bwd Pkts",
        "Total Length of Fwd Packet": "TotLen Fwd Pkts",
        "Total Length of Bwd Packet": "TotLen Bwd Pkts",
        "Packet Length Mean": "Pkt Len Mean",
        "Packet Length Std": "Pkt Len Std",
        "FIN Flag Count": "FIN Flag Cnt",
        "SYN Flag Count": "SYN Flag Cnt",
        "RST Flag Count": "RST Flag Cnt",
        "PSH Flag Count": "PSH Flag Cnt",
        "ACK Flag Count": "ACK Flag Cnt",
        "URG Flag Count": "URG Flag Cnt",
    }
    raw = raw.rename(columns={k: v for k, v in V4_TO_V3.items() if k in raw.columns})
    print("applied V4->V3 header aliases:", [k for k in V4_TO_V3 if k in frames[0].columns])

    need = ["Flow ID","Src IP","Src Port","Dst IP","Dst Port","Protocol","Timestamp",
            "Flow Duration","Fwd IAT Mean","Bwd IAT Mean","Pkt Len Mean","Pkt Len Std",
            "Tot Fwd Pkts","Tot Bwd Pkts","TotLen Fwd Pkts","TotLen Bwd Pkts",
            "Down/Up Ratio"] + FLAG_CNT
    miss = [c for c in need if c not in raw.columns]
    print("FIELD CHECK:", "ALL PRESENT" if not miss else f"MISSING {miss}")
    print("  (CICFlowMeter V4 header actually seen:", list(raw.columns)[:12], "...)")
    if miss:
        print("  -> map V3/V4 header aliases and re-run");

    # STEP 8 -- reconstruct CIC ML-data flag binarisation
    pre = raw[FLAG_CNT].apply(pd.to_numeric, errors="coerce")
    n_gt1 = int((pre > 1).sum().sum())
    raw[FLAG_CNT] = pre.clip(0, 1)
    print(f"STEP-8 flag clip: {n_gt1:,} flag-count values were >1 in raw CICFlowMeter output "
          f"and have been clipped to 1 (reconstruction of CIC ML preprocessing, NOT CICFlowMeter).")

    # cross-host dedup on canonical 5-tuple + start time (within one FlowTimeout)
    raw["_k"] = raw.apply(canon_key, axis=1)
    raw["_t"] = pd.to_datetime(raw["Timestamp"], errors="coerce", dayfirst=True).astype("int64") // 10**9
    raw = raw.sort_values(["_k","_t"]).reset_index(drop=True)
    gap = raw.groupby("_k")["_t"].diff()
    raw["_clust"] = (gap.isna() | (gap > 120)).groupby(raw["_k"]).cumsum()
    raw["_ipnum"] = raw["_srcfile"].str.extract(r"(\d+\.\d+\.\d+\.\d+)")[0]
    dedup = raw.drop_duplicates(["_k","_clust"], keep="first").reset_index(drop=True)
    print(f"cross-host dedup: {len(raw):,} -> {len(dedup):,} flows  ({len(raw)-len(dedup):,} duplicates removed)")

    st_cfm = cicfm_csv_to_states(dedup.copy())
    st_cfm.to_csv(Path(flow_dir)/"states10_cicflowmeter.csv", index=False)
    print(f"CICFlowMeter-derived 10s windows: {len(st_cfm)}")

    # --- our corrected emulation on the SAME pcaps ---
    pfiles = sorted(glob.glob(str(Path(pcap_dir)/"*")))
    pkt_frames=[]
    for pf in pfiles:
        d = P.parse_pcap_streaming(pf)
        if len(d): pkt_frames.append(d)
    allp = pd.concat(pkt_frames, ignore_index=True).sort_values("timestamp_abs").reset_index(drop=True)
    allp.attrs["t0_epoch"] = float(allp["timestamp_abs"].min())
    fl = P.reconstruct_flows(allp)
    # cross-host dedup (flow_key metadata column)
    fl = fl.sort_values(["flow_key","start_ts"]).reset_index(drop=True)
    g = fl.groupby("flow_key")["start_ts"].diff()
    fl["_c"] = (g.isna() | (g > 120)).groupby(fl["flow_key"]).cumsum()
    fl = fl.drop_duplicates(["flow_key","_c"], keep="first")
    st_emu = P.compute_window_features(fl)
    st_emu.to_csv(Path(flow_dir)/"states10_emulation.csv", index=False)
    print(f"emulation 10s windows (same pcaps): {len(st_emu)}")

    sv2 = pd.read_csv(ROOT/"data/processed/states_v2/Friday-02-03-2018_TrafficForML_CICFlowMeter_states_v2.csv")
    try:
        prior = pd.read_csv(ROOT/"models/pcap_analysis_v4/parity/subset25_z_comparison.csv").set_index("feature")
    except FileNotFoundError:
        prior = None

    rows=[]
    for f in BASE:
        cf = st_cfm[f].to_numpy(float); em = st_emu[f].to_numpy(float)
        tm, tsd = float(sv2[f].mean()), float(sv2[f].std())+1e-9
        rec = dict(feature=f,
            mean_cfm=cf.mean(), mean_emu=em.mean(), train_mean=tm,
            z_cfm=(cf.mean()-tm)/tsd, z_emu=(em.mean()-tm)/tsd,
            ratio_cfm=(cf.mean()/tm if abs(tm)>1e-9 else np.nan))
        if prior is not None and f in prior.index:
            rec["z_prior21h"] = float(prior.loc[f, "z_25h"])
        rows.append(rec)
    D = pd.DataFrame(rows)
    D["absz_cfm"]=D.z_cfm.abs(); D["absz_emu"]=D.z_emu.abs()
    D = D.sort_values("absz_cfm").reset_index(drop=True)
    D.to_csv(Path(flow_dir)/"cicflowmeter_vs_emulation_vs_training.csv", index=False)

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print("\n=== per-feature: official CICFlowMeter vs emulation vs training (sorted |z_cfm|) ===")
    print(D.round(3).to_string(index=False))
    summ = dict(
        cfm_windows=int(len(st_cfm)), emu_windows=int(len(st_emu)),
        mean_absz_cfm=float(D.absz_cfm.mean()), mean_absz_emu=float(D.absz_emu.mean()),
        within_1sd_cfm=int((D.absz_cfm<1).sum()), within_1sd_emu=int((D.absz_emu<1).sum()),
        within_2sd_cfm=int((D.absz_cfm<2).sum()), within_2sd_emu=int((D.absz_emu<2).sum()),
        flag_values_gt1_clipped=n_gt1,
    )
    json.dump(summ, open(Path(flow_dir)/"cicflowmeter_compare_summary.json","w"), indent=2)
    print("\nSUMMARY:", json.dumps(summ, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
