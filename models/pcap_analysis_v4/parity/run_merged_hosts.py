"""
Merged 10-victim-host state distribution (CORRECTED representation) vs
the real training states_v2 March-2 distribution.

Processes the ten victim captures as ONE monitored network: parse all,
concatenate the per-packet tables, then run the SAME corrected
src/pcap_ingest_v4 flow-reconstruction + window-aggregation over the
union. NO artificial scaling/normalisation -- the real gap is reported.

Range: 2018-03-02 12:47:00 - 15:10:00 UTC (pre-attack baseline + Bot
session 1, onset 14:13:28 UTC). Analysis-only. No V4 inference.
"""
import sys, json, time, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]  # repo root (models/pcap_analysis_v4/parity/<this file>)
sys.path.insert(0, str(ROOT/"src"))
import pcap_ingest_v4 as P
from scapy.all import PcapReader, IP, IPv6, TCP, UDP

OUT = ROOT/"models/pcap_analysis_v4/parity"
HOSTS = [6,8,10,12,14,17,23,26,29,30]
FILES = [ROOT/f"data/raw/march02/bot_pcaps/pcap/capEC2AMAZ-O4EL3NG-172.31.69.{h}" for h in HOSTS]
LO, HI = 1519994820.0, 1520003400.0     # 12:47:00 .. 15:10:00 UTC
W = 10
BASE = P.BASE_FEATURES

def parse_range(path):
    rows=[]
    with PcapReader(str(path)) as rd:
        for p in rd:
            ts=float(p.time)
            if ts<LO or ts>=HI: continue
            l3=p.getlayer(IP) or p.getlayer(IPv6)
            if l3 is None: continue
            v6=l3.name=="IPv6"; proto=int(l3.nh) if v6 else int(l3.proto)
            sp=dp=0; l4=0; fl={k:False for k in P._FLAG_KEYS}
            if p.haslayer(TCP):
                t=p[TCP]; sp,dp=int(t.sport),int(t.dport); s=str(t.flags)
                for k in P._FLAG_KEYS: fl[k]=P._FLAG_LETTER[k] in s
                l4=len(t.payload)
            elif p.haslayer(UDP):
                u=p[UDP]; sp,dp=int(u.sport),int(u.dport); l4=len(u.payload)
            else: l4=len(l3.payload)
            r=dict(timestamp_abs=ts,src_ip=str(l3.src),dst_ip=str(l3.dst),
                   src_port=sp,dst_port=dp,protocol=proto,
                   frame_len=int(len(p)),l4_payload_len=int(l4)); r.update(fl)
            rows.append(r)
    return pd.DataFrame(rows)

t0=time.time(); parts=[]
for f in FILES:
    d=parse_range(f)
    print(f"  {f.name}: {len(d):,} pkts in range  [{time.time()-t0:.0f}s]", flush=True)
    parts.append(d)
pdf=pd.concat(parts, ignore_index=True).sort_values("timestamp_abs").reset_index(drop=True)
pdf.attrs["t0_epoch"]=float(pdf["timestamp_abs"].min())
print(f"MERGED: {len(pdf):,} packets, {pdf['src_ip'].nunique()} src IPs  [{time.time()-t0:.0f}s]", flush=True)

flow = P.reconstruct_flows(pdf)
print(f"whole-lifetime flows: {len(flow):,}  [{time.time()-t0:.0f}s]", flush=True)
feat = P.compute_window_features(flow)
feat = P.add_relative_changes(feat)
feat.to_csv(OUT/"merged10_features.csv", index=False)
print(f"feature windows: {len(feat)}", flush=True)

sv2 = pd.read_csv(ROOT/"data/processed/states_v2/Friday-02-03-2018_TrafficForML_CICFlowMeter_states_v2.csv")

rec=[]
for f in BASE:
    a=feat[f].to_numpy(float); tm=float(sv2[f].mean()); tsd=float(sv2[f].std())
    rec.append(dict(feature=f, merged_mean=a.mean(), merged_std=a.std(),
        merged_median=float(np.median(a)),
        train_mean=tm, train_std=tsd,
        z_of_merged_mean=(a.mean()-tm)/(tsd+1e-9),
        merged_over_train_ratio=(a.mean()/tm) if abs(tm)>1e-9 else np.nan))
D=pd.DataFrame(rec).sort_values("z_of_merged_mean", key=lambda s:s.abs(), ascending=False)
D.to_csv(OUT/"merged10_vs_training_distribution.csv", index=False)
pd.set_option("display.width",200); pd.set_option("display.max_columns",30)
print("\n=== MERGED 10-HOST corrected state vs training states_v2 (|z| desc) ===")
print(D.to_string(index=False))
vol=["flow_count","log_flow_count","log_total_packets","log_total_bytes",
     "window_packet_rate","window_byte_rate","log_unique_dst_ports",
     "unique_protocols","dst_port_entropy","protocol_entropy"]
print("\nVOLUME/DIVERSITY features (still OOD? ):")
print(D[D.feature.isin(vol)][["feature","merged_mean","train_mean","z_of_merged_mean","merged_over_train_ratio"]].to_string(index=False))

# distribution plots
fig,ax=plt.subplots(6,5,figsize=(22,22))
for a,f in zip(ax.ravel(), BASE):
    a.hist(sv2[f].to_numpy(), bins=40, density=True, alpha=.5, label="train")
    a.hist(feat[f].to_numpy(), bins=40, density=True, alpha=.5, label="merged10")
    a.set_title(f, fontsize=8); a.tick_params(labelsize=6)
    a.legend(fontsize=6)
ax.ravel()[-1].axis("off")
fig.suptitle("Merged 10-host corrected state vs training states_v2 (March 2)", y=1.0)
fig.tight_layout(); fig.savefig(OUT/"merged10_distributions.png", dpi=110); plt.close(fig)

json.dump(dict(hosts=HOSTS, range_utc=["2018-03-02 12:47:00","2018-03-02 15:10:00"],
    n_packets=int(len(pdf)), n_flows=int(len(flow)), n_windows=int(len(feat)),
    mean_absZ=float(D.z_of_merged_mean.abs().mean()),
    n_feat_absZ_lt2=int((D.z_of_merged_mean.abs()<2).sum()),
    volume_features_still_OOD=[f for f in vol
        if abs(float(D.loc[D.feature==f,"z_of_merged_mean"].values[0]))>=2]),
    open(OUT/"merged10_meta.json","w"), indent=2)
print("\nwrote merged10_* to", OUT)
