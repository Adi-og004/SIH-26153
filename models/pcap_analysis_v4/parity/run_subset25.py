"""
STEP 2-6: merge the stratified 25-host subset into one monitored-network
state (with cross-host flow dedup), generate the CORRECTED 10s features,
and compare 1-host vs 10-host vs 25-host vs training states_v2.

Feature definitions are UNCHANGED (src/pcap_ingest_v4.py as-is).
Analysis-only. No V4 inference.
"""
import sys, json, time, glob, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]  # repo root (models/pcap_analysis_v4/parity/<this file>)
sys.path.insert(0, str(ROOT/"src"))
import pcap_ingest_v4 as P
from scapy.all import PcapReader, IP, IPv6, TCP, UDP

OUT = ROOT/"models/pcap_analysis_v4/parity"
LO, HI = 1519994820.0, 1520003400.0    # 2018-03-02 12:47:00 .. 15:10:00 UTC (== 10-host run)
W = 10
BASE = P.BASE_FEATURES

# ---- resolve the 25 target host files across both download dirs ----
want = [ln.strip().split("/")[-1] for ln in
        open(OUT/"subset_plan/subset_25_members.txt") if ln.strip()]
dirs = [ROOT/"data/raw/march02/bot_pcaps/pcap", ROOT/"data/raw/march02/subset25/pcap"]
present, missing = [], []
for base in want:
    hit = None
    for d in dirs:
        p = d/base
        if p.exists() and p.stat().st_size > 0:
            hit = p; break
    (present.append(hit) if hit else missing.append(base))
print(f"target hosts: {len(want)}   present: {len(present)}   missing: {len(missing)}")
for m in missing: print("   MISSING:", m)

ipre = re.compile(r"(\d+\.\d+\.\d+\.\d+)")
def host_ip(p): 
    m = ipre.search(p.name); return m.group(1) if m else p.name
monitored = set(host_ip(p) for p in present)
print("monitored host IPs:", len(monitored))

# ---- parse each capture in range, per-host flow reconstruction ----
def parse_range(path):
    rows=[]
    with PcapReader(str(path)) as rd:
        for pk in rd:
            ts=float(pk.time)
            if ts<LO or ts>=HI: continue
            l3=pk.getlayer(IP) or pk.getlayer(IPv6)
            if l3 is None: continue
            v6=l3.name=="IPv6"; proto=int(l3.nh) if v6 else int(l3.proto)
            sp=dp=0; l4=0; fl={k:False for k in P._FLAG_KEYS}
            if pk.haslayer(TCP):
                t=pk[TCP]; sp,dp=int(t.sport),int(t.dport); s=str(t.flags)
                for k in P._FLAG_KEYS: fl[k]=P._FLAG_LETTER[k] in s
                l4=len(t.payload)
            elif pk.haslayer(UDP):
                u=pk[UDP]; sp,dp=int(u.sport),int(u.dport); l4=len(u.payload)
            else: l4=len(l3.payload)
            r=dict(timestamp_abs=ts,src_ip=str(l3.src),dst_ip=str(l3.dst),
                   src_port=sp,dst_port=dp,protocol=proto,
                   frame_len=int(len(pk)),l4_payload_len=int(l4)); r.update(fl)
            rows.append(r)
    return pd.DataFrame(rows)

t0=time.time(); per_host_flows=[]; total_pkts=0; raw_flow_rows=0
for p in present:
    d=parse_range(p)
    total_pkts += len(d)
    if len(d)==0:
        print(f"  {p.name}: 0 pkts in range"); continue
    d=d.sort_values("timestamp_abs").reset_index(drop=True)
    d.attrs["t0_epoch"]=float(d["timestamp_abs"].min())
    fl = P.reconstruct_flows(d)          # CORRECTED whole-lifetime flows
    fl["capture_host"]=host_ip(p)
    per_host_flows.append(fl)
    raw_flow_rows += len(fl)
    print(f"  {p.name}: {len(d):,} pkts -> {len(fl):,} flows  [{time.time()-t0:.0f}s]", flush=True)

allf = pd.concat(per_host_flows, ignore_index=True)
print(f"\nMERGED raw: {total_pkts:,} packets, {len(allf):,} per-host flow rows")

# ---- CROSS-HOST FLOW DEDUP ----
# A bidirectional flow between two MONITORED hosts is recorded in BOTH
# host captures. reconstruct_flows now emits `flow_key` = the canonical
# UNORDERED 5-tuple  "lo_ip|hi_ip|lo_port|hi_port|proto".  Two rows with
# the same flow_key whose start times are within one FlowTimeout (120 s)
# are the SAME physical flow seen twice -> keep exactly ONE, preferring
# the capture whose host IP == the flow's lower endpoint (deterministic),
# else the earliest start.  Rows whose flow_key involves an UNMONITORED
# endpoint (e.g. C2 18.219.211.138) appear once and are untouched.
allf = allf.sort_values(["flow_key", "start_ts"]).reset_index(drop=True)
gap = allf.groupby("flow_key")["start_ts"].diff()
allf["_newclust"] = gap.isna() | (gap > 120.0)
allf["_clust"] = allf.groupby("flow_key")["_newclust"].cumsum()
allf["_lo_ip"] = allf["flow_key"].str.split("|").str[0]
allf["_hi_ip"] = allf["flow_key"].str.split("|").str[1]
allf["_is_owner"] = allf["capture_host"] == allf["_lo_ip"]
allf["_ipnum"] = allf["capture_host"].apply(lambda s: tuple(int(x) for x in s.split(".")))
both_mon = allf["_lo_ip"].isin(monitored) & allf["_hi_ip"].isin(monitored)
dedup = (allf.sort_values(["flow_key", "_clust", "_is_owner", "start_ts", "_ipnum"],
                          ascending=[True, True, False, True, True])
             .drop_duplicates(["flow_key", "_clust"], keep="first")
             .drop(columns=["_newclust","_clust","_is_owner","_ipnum"])
             .reset_index(drop=True))
n_removed = len(allf) - len(dedup)
n_internal_flows = int((dedup["_lo_ip"].isin(monitored) & dedup["_hi_ip"].isin(monitored)).sum())
n_dupgroups = int(((allf.groupby(["flow_key","_clust"]).size() > 1)).sum())
print(f"per-host flow rows                 : {len(allf):,}")
print(f"  rows with both endpoints monitored: {int(both_mon.sum()):,}")
print(f"cross-capture duplicate flow groups : {n_dupgroups:,}")
print(f"duplicate flow rows removed          : {n_removed:,}")
print(f"DEDUPED whole-lifetime flows         : {len(dedup):,}  (internal-internal kept: {n_internal_flows:,})")
dedup = dedup.drop(columns=["_lo_ip","_hi_ip"])

# ---- window features on the deduped merged flow table ----
feat = P.compute_window_features(dedup)
feat = P.add_relative_changes(feat)
feat.to_csv(OUT/"subset25_features.csv", index=False)
print(f"10-second windows: {len(feat)}")

# ---- comparison: 1-host / 10-host / 25-host / training ----
sv2 = pd.read_csv(ROOT/"data/processed/states_v2/Friday-02-03-2018_TrafficForML_CICFlowMeter_states_v2.csv")
one = pd.read_csv(OUT/"parity_v2_feature_summary.csv").set_index("feature")   # has mean_NEW, z_NEW
ten = pd.read_csv(OUT/"merged10_vs_training_distribution.csv").set_index("feature")  # merged_mean, z_of_merged_mean

rows=[]
for f in BASE:
    a=feat[f].to_numpy(float)
    tm=float(sv2[f].mean()); tsd=float(sv2[f].std())
    z25=(a.mean()-tm)/(tsd+1e-9)
    rows.append(dict(feature=f,
        mean_1h=float(one.loc[f,"mean_NEW"]), z_1h=float(one.loc[f,"z_NEW"]),
        mean_10h=float(ten.loc[f,"merged_mean"]), z_10h=float(ten.loc[f,"z_of_merged_mean"]),
        mean_25h=a.mean(), z_25h=z25,
        train_mean=tm, ratio_25h=(a.mean()/tm if abs(tm)>1e-9 else np.nan)))
D=pd.DataFrame(rows)
D["absz_1h"]=D.z_1h.abs(); D["absz_10h"]=D.z_10h.abs(); D["absz_25h"]=D.z_25h.abs()
D=D.sort_values("absz_25h").reset_index(drop=True)
D.to_csv(OUT/"subset25_z_comparison.csv", index=False)

pd.set_option("display.width",240); pd.set_option("display.max_columns",40)
print("\n=== per-feature z: 1-host -> 10-host -> 25-host  (vs training states_v2) ===")
print(D[["feature","z_1h","z_10h","z_25h","ratio_25h","train_mean","mean_25h"]].round(3).to_string(index=False))

def cnt(col,thr): return int((D[col]<thr).sum())
summ=dict(
  hosts_present=len(present), hosts_missing=missing,
  packets=int(total_pkts), per_host_flow_rows=int(len(allf)),
  dup_flow_groups=int(n_dupgroups), dup_rows_removed=int(n_removed), internal_flows_kept=int(n_internal_flows),
  deduped_flows=int(len(dedup)), windows=int(len(feat)),
  mean_absz_1h=float(D.absz_1h.mean()), mean_absz_10h=float(D.absz_10h.mean()), mean_absz_25h=float(D.absz_25h.mean()),
  within_1sd_1h=cnt("absz_1h",1), within_1sd_10h=cnt("absz_10h",1), within_1sd_25h=cnt("absz_25h",1),
  within_2sd_1h=cnt("absz_1h",2), within_2sd_10h=cnt("absz_10h",2), within_2sd_25h=cnt("absz_25h",2),
)
json.dump(summ, open(OUT/"subset25_summary.json","w"), indent=2)
print("\n=== SUMMARY ===")
for k,v in summ.items(): print(f"  {k}: {v}")

focus=["flow_count","log_flow_count","log_total_packets","log_total_bytes",
       "window_packet_rate","window_byte_rate","log_unique_dst_ports","unique_protocols",
       "dst_port_entropy","syn_rate","fin_rate","log_mean_flow_duration",
       "log_mean_fwd_iat","log_mean_bwd_iat"]
print("\n=== FOCUS features: z trajectory ===")
print(D[D.feature.isin(focus)].set_index("feature").loc[focus,["z_1h","z_10h","z_25h","ratio_25h"]].round(3).to_string())

# plot z trajectory
fig,ax=plt.subplots(figsize=(13,8))
xs=[1,10,25]
for f in focus:
    r=D[D.feature==f].iloc[0]
    ax.plot(xs,[r.z_1h,r.z_10h,r.z_25h],marker="o",label=f)
ax.axhline(0,color="k",lw=.8); ax.axhline(2,color="grey",ls="--",lw=.8); ax.axhline(-2,color="grey",ls="--",lw=.8)
ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(["1 host","10 hosts","25 hosts"])
ax.set_ylabel("z vs training states_v2"); ax.set_title("Scope -> distribution: focus features")
ax.legend(fontsize=7,ncol=2)
fig.tight_layout(); fig.savefig(OUT/"subset25_z_trajectory.png",dpi=130); plt.close(fig)
print("\nwrote subset25_* to",OUT)
