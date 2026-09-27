"""
PCAP feature-parity analysis for host 172.31.69.30 (March 2 2018 Bot onset).

Representation A : pcap_ingest_v4.compute_window_features  (v4 semantics)
Representation B : TRAINING-DEFINITION EMULATION -- reconstruct CICFlowMeter-style
                  whole-lifetime bidirectional flows from the SAME pcap, then apply
                  archive/create_network_states_v2.py's exact window aggregation.
                  NOT the official Java CICFlowMeter (see report for what B omits).

Same 10s absolute-epoch window grid for both. Analysis range 12:47-14:40 UTC.
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]  # repo root (models/pcap_analysis_v4/parity/<this file>)
sys.path.insert(0, str(ROOT/"src"))
import pcap_ingest_v4 as P
from scapy.all import PcapReader, IP, IPv6, TCP, UDP

PCAP = ROOT/"data/raw/march02/bot_pcaps/pcap/capEC2AMAZ-O4EL3NG-172.31.69.30"
OUT  = ROOT/"models/pcap_analysis_v4/parity"
WIN  = 10
LO, HI = 1519994820.0, 1520001600.0   # 2018-03-02 12:47:00 .. 14:40:00 UTC
FLOW_TIMEOUT = 120.0                   # CICFlowMeter FlowTimeout (s)
BASE = P.BASE_FEATURES                 # canonical 26, in order

# ---------------------------------------------------------------- parse once
print("parsing", PCAP.name, flush=True)
pk = []
t0 = time.time()
with PcapReader(str(PCAP)) as rd:
    for p in rd:
        ts = float(p.time)
        if ts < LO or ts >= HI:
            continue
        l4 = p.getlayer(IP) or p.getlayer(IPv6)
        if l4 is None:
            continue
        proto = int(getattr(l4, "proto", getattr(l4, "nh", 0)))
        sport = dport = 0
        syn=ack=fin=rst=psh=urg=False
        if p.haslayer(TCP):
            t=p[TCP]; sport,dport=int(t.sport),int(t.dport)
            f=str(t.flags); syn="S" in f; ack="A" in f; fin="F" in f
            rst="R" in f; psh="P" in f; urg="U" in f
            l4pay=len(t.payload)
        elif p.haslayer(UDP):
            u=p[UDP]; sport,dport=int(u.sport),int(u.dport); l4pay=len(u.payload)
        else:
            l4pay=len(l4.payload)
        iplen = int(getattr(l4,"len", len(p)-14))   # L3 total length
        pk.append((ts, l4.src, l4.dst, sport, dport, proto,
                   int(len(p)), iplen, l4pay, syn,ack,fin,rst,psh,urg))
cols=["ts","src","dst","sport","dport","proto","framelen","iplen","l4pay",
      "syn","ack","fin","rst","psh","urg"]
pdf = pd.DataFrame(pk, columns=cols).sort_values("ts").reset_index(drop=True)
print(f"  {len(pdf):,} packets in range  [{time.time()-t0:.1f}s]", flush=True)
pdf["win"] = (np.floor(pdf["ts"]/WIN)*WIN).astype(np.int64)

# ================================================================ REP A (v4)
# feed pcap_ingest_v4's own functions; override window_idx to the shared grid
pA = pd.DataFrame({
    "timestamp_abs": pdf["ts"], "timestamp": pdf["ts"],
    "src_ip": pdf["src"], "dst_ip": pdf["dst"],
    "src_port": pdf["sport"], "dst_port": pdf["dport"],
    "protocol": pdf["proto"], "pkt_len": pdf["framelen"],
})
for k,c in [("f_syn","syn"),("f_ack","ack"),("f_fin","fin"),
            ("f_rst","rst"),("f_psh","psh"),("f_urg","urg")]:
    pA[k]=pdf[c].values
pA["window_idx"]=pdf["win"].values
flowA = P.reconstruct_flows(pA)
featA = P.compute_window_features(flowA, pA).rename(columns={"window_idx":"win"})
featA = featA[["win"]+BASE].copy()

# ================================================================ REP B (training-def emulation)
def entropy_from_counts(series):
    c = series.value_counts(dropna=True)
    if len(c) <= 1: return 0.0
    p = c.to_numpy(float)/c.sum()
    return float(-np.sum(p*np.log2(p+1e-12)))
def safe_div(n,d): 
    r = n/(np.abs(d)+1e-8); return np.where(np.isfinite(r), r, 0.0)
def safe_log1p(x):
    x = np.array(x, dtype=float, copy=True); x[~np.isfinite(x)] = 0.0
    return np.log1p(np.clip(x, 0.0, None))

# --- reconstruct whole-lifetime bidirectional flows, 120s idle timeout ---
key = (np.minimum(pdf["src"],pdf["dst"]) + "|" + np.maximum(pdf["src"],pdf["dst"]) + "|" +
       np.minimum(pdf["sport"],pdf["dport"]).astype(str) + "|" +
       np.maximum(pdf["sport"],pdf["dport"]).astype(str) + "|" + pdf["proto"].astype(str))
pdf["_k"]=key.values
rows=[]
t0=time.time()
for k,g in pdf.groupby("_k", sort=False):
    g=g.sort_values("ts")
    ts=g["ts"].to_numpy()
    # split into sub-flows on >120s gaps
    brk=np.where(np.diff(ts)>FLOW_TIMEOUT)[0]+1
    for seg in np.split(g.index.values, brk):
        s=g.loc[seg]
        first=s.iloc[0]
        fwd_mask=(s["src"]==first["src"])&(s["sport"]==first["sport"])
        fwd=s[fwd_mask]; bwd=s[~fwd_mask]
        tfp,tbp=len(fwd),len(bwd)
        tfb=int(fwd["l4pay"].sum()); tbb=int(bwd["l4pay"].sum())
        st=s["ts"].to_numpy()
        dur=(st[-1]-st[0])*1e6 if len(st)>1 else 0.0
        fi=np.diff(np.sort(fwd["ts"].to_numpy()))*1e6 if tfp>1 else np.array([])
        bi=np.diff(np.sort(bwd["ts"].to_numpy()))*1e6 if tbp>1 else np.array([])
        plen=s["iplen"].to_numpy().astype(float)
        rows.append(dict(
            win=int(np.floor(st[0]/WIN)*WIN),
            dst_port=int(first["dport"]), protocol=int(first["proto"]),
            tot_pkts=tfp+tbp, tot_fwd=tfp, tot_bwd=tbp,
            tot_bytes=tfb+tbb, totlen_fwd=tfb, totlen_bwd=tbb,
            flow_dur=dur,
            fwd_iat_mean=float(fi.mean()) if fi.size else 0.0,
            bwd_iat_mean=float(bi.mean()) if bi.size else 0.0,
            pkt_len_mean=float(plen.mean()) if plen.size else 0.0,
            pkt_len_std=float(plen.std())  if plen.size>1 else 0.0,
            syn=int(bool(s["syn"].any())), ack=int(bool(s["ack"].any())),
            fin=int(bool(s["fin"].any())), rst=int(bool(s["rst"].any())),
            psh=int(bool(s["psh"].any())), urg=int(bool(s["urg"].any())),
            downup=int(tbp//tfp) if tfp>0 else 0,
        ))
fl=pd.DataFrame(rows)
print(f"  B: {len(fl):,} CICFM-style flows  [{time.time()-t0:.1f}s]", flush=True)

# per-flow ratios (pre-aggregation), exactly like create_network_states_v2
fl["fwd_packet_ratio"]=safe_div(fl["tot_fwd"], fl["tot_pkts"])
fl["bwd_packet_ratio"]=safe_div(fl["tot_bwd"], fl["tot_pkts"])
fl["fwd_byte_ratio"]  =safe_div(fl["totlen_fwd"], fl["tot_bytes"])
fl["bwd_byte_ratio"]  =safe_div(fl["totlen_bwd"], fl["tot_bytes"])
pd_ = fl["tot_pkts"]+1e-8
for r,c in [("syn_rate","syn"),("ack_rate","ack"),("rst_rate","rst"),
            ("fin_rate","fin"),("psh_rate","psh"),("urg_rate","urg")]:
    fl[r]=fl[c]/pd_

g=fl.groupby("win")
B=g.agg(
    flow_count=("win","size"),
    total_packets=("tot_pkts","sum"), total_bytes=("tot_bytes","sum"),
    mean_flow_duration=("flow_dur","mean"),
    mean_fwd_iat=("fwd_iat_mean","mean"), mean_bwd_iat=("bwd_iat_mean","mean"),
    mean_pkt_len=("pkt_len_mean","mean"), std_pkt_len=("pkt_len_std","mean"),
    syn_rate=("syn_rate","mean"), ack_rate=("ack_rate","mean"),
    rst_rate=("rst_rate","mean"), fin_rate=("fin_rate","mean"),
    psh_rate=("psh_rate","mean"), urg_rate=("urg_rate","mean"),
    fwd_packet_ratio=("fwd_packet_ratio","mean"),
    bwd_packet_ratio=("bwd_packet_ratio","mean"),
    fwd_byte_ratio=("fwd_byte_ratio","mean"),
    bwd_byte_ratio=("bwd_byte_ratio","mean"),
    mean_down_up_ratio=("downup","mean"),
    unique_dst_ports=("dst_port","nunique"),
    unique_protocols=("protocol","nunique"),
).reset_index()
B["log_flow_count"]=safe_log1p(B["flow_count"])
B["log_total_packets"]=safe_log1p(B["total_packets"])
B["log_total_bytes"]=safe_log1p(B["total_bytes"])
B["log_mean_flow_duration"]=safe_log1p(B["mean_flow_duration"])
B["log_mean_fwd_iat"]=safe_log1p(B["mean_fwd_iat"])
B["log_mean_bwd_iat"]=safe_log1p(B["mean_bwd_iat"])
B["log_unique_dst_ports"]=safe_log1p(B["unique_dst_ports"])
B["window_packet_rate"]=B["total_packets"]/WIN
B["window_byte_rate"]=B["total_bytes"]/WIN
B["dst_port_entropy"]=g["dst_port"].apply(entropy_from_counts).values
B["protocol_entropy"]=g["protocol"].apply(entropy_from_counts).values
featB=B[["win"]+BASE].copy()

# ================================================================ COMPARE
m=featA.merge(featB, on="win", suffixes=("_A","_B"))
m=m[(m["win"]>=LO)&(m["win"]<HI)].sort_values("win").reset_index(drop=True)
print(f"shared windows: {len(m)}  ({pd.to_datetime(m['win'].min(),unit='s')} .. {pd.to_datetime(m['win'].max(),unit='s')} UTC)", flush=True)
m.to_csv(OUT/"parity_windows_A_vs_B.csv", index=False)

TOL={}
for f in BASE:
    if f.startswith("log_"): TOL[f]=0.5
    elif f.endswith("_entropy"): TOL[f]=0.3
    elif "rate" in f or "ratio" in f: TOL[f]=0.05
    elif f in ("flow_count","unique_protocols"): TOL[f]=None  # rel
    else: TOL[f]=None
recs=[]
for f in BASE:
    a=m[f+"_A"].to_numpy(float); b=m[f+"_B"].to_numpy(float)
    d=a-b; ad=np.abs(d)
    rel=np.abs(d)/(np.abs(b)+1e-9)
    if np.std(a)>1e-12 and np.std(b)>1e-12:
        r=float(np.corrcoef(a,b)[0,1])
    else:
        r=np.nan
    tol=TOL[f]
    pct_large = float(np.mean(rel>0.25)*100) if tol is None else float(np.mean(ad>tol)*100)
    recs.append(dict(feature=f, mean_A=float(np.mean(a)), mean_B=float(np.mean(b)),
        mean_signed_diff=float(np.mean(d)),
        MAD=float(np.mean(ad)), medAD=float(np.median(ad)), p95AD=float(np.percentile(ad,95)),
        pearson_r=r, median_rel_diff=float(np.median(rel)),
        pct_windows_large_discrep=pct_large, tol=("rel>25%" if tol is None else tol)))
S=pd.DataFrame(recs)
# rank: worst = low r + high normalized MAD
rng=(m[[f+"_B" for f in BASE]].max().values - m[[f+"_B" for f in BASE]].min().values)
S["norm_MAD"]=S["MAD"].values/np.where(np.abs(rng)>1e-9,np.abs(rng),1.0)
S["parity_score"]=(1-S["pearson_r"].fillna(0)) + S["norm_MAD"]   # higher = worse
S=S.sort_values("parity_score", ascending=True).reset_index(drop=True)  # best first
S.to_csv(OUT/"parity_feature_summary.csv", index=False)
print("\n=== FEATURE PARITY (best -> worst) ===")
print(S[["feature","mean_A","mean_B","MAD","p95AD","pearson_r","median_rel_diff","pct_windows_large_discrep","parity_score"]].to_string(index=False))

# plots: 8 worst
worst=S.tail(8)["feature"].tolist()[::-1]
tt=pd.to_datetime(m["win"], unit="s")
fig,axes=plt.subplots(4,2,figsize=(15,16))
for ax,f in zip(axes.ravel(), worst):
    ax.plot(tt, m[f+"_A"], label="A: pcap_ingest_v4", lw=1.3)
    ax.plot(tt, m[f+"_B"], label="B: training-def emulation", lw=1.3)
    ax.axvline(pd.to_datetime(1520000008,unit="s"), color="k", ls=":", lw=1, label="first C2 pkt 14:13:28Z")
    r=S.loc[S.feature==f,"pearson_r"].values[0]
    ax.set_title(f"{f}   r={r:.3f}  MAD={S.loc[S.feature==f,'MAD'].values[0]:.3g}")
    ax.legend(fontsize=7); ax.tick_params(axis="x", labelrotation=30, labelsize=7)
fig.suptitle("Host .30 -- 8 worst-parity base features (A vs B), 12:47-14:40 UTC", y=1.0)
fig.tight_layout(); fig.savefig(OUT/"parity_worst_features_timeseries.png", dpi=130); plt.close(fig)

# scatter grid for the same 8
fig,axes=plt.subplots(4,2,figsize=(13,18))
for ax,f in zip(axes.ravel(), worst):
    ax.scatter(m[f+"_B"], m[f+"_A"], s=10, alpha=.5)
    lo=min(m[f+"_A"].min(),m[f+"_B"].min()); hi=max(m[f+"_A"].max(),m[f+"_B"].max())
    ax.plot([lo,hi],[lo,hi],"k--",lw=1)
    ax.set_xlabel("B (training-def)"); ax.set_ylabel("A (pcap_v4)"); ax.set_title(f)
fig.tight_layout(); fig.savefig(OUT/"parity_worst_features_scatter.png", dpi=130); plt.close(fig)

# ---- rel-change parity ----
def relchg(df):
    x=df[BASE].astype(float).reset_index(drop=True)
    dlt=x.diff().fillna(0.0); prev=x.shift(1)
    rc=dlt/(prev.abs()+1e-8); rc.iloc[0]=0.0
    rc=rc.replace([np.inf,-np.inf],np.nan).fillna(0.0).clip(-10,10)
    rc.columns=[c+"_rel_change" for c in BASE]; return rc
rcA=relchg(m.rename(columns={f+"_A":f for f in BASE})[BASE].__class__(m[[f+"_A" for f in BASE]].values, columns=BASE))
rcB=relchg(pd.DataFrame(m[[f+"_B" for f in BASE]].values, columns=BASE))
rc_recs=[]
for i,f in enumerate(BASE):
    a=rcA.iloc[:,i].to_numpy(); b=rcB.iloc[:,i].to_numpy(); d=a-b
    r=float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else np.nan
    rc_recs.append(dict(feature=f+"_rel_change", MAD=float(np.mean(np.abs(d))),
        p95AD=float(np.percentile(np.abs(d),95)), pearson_r=r,
        mean_A=float(a.mean()), mean_B=float(b.mean())))
RC=pd.DataFrame(rc_recs).sort_values("pearson_r").reset_index(drop=True)
RC.to_csv(OUT/"parity_relchange_summary.csv", index=False)
print("\n=== REL-CHANGE PARITY (worst r first) ===")
print(RC.to_string(index=False))

# ---- A vs REAL training states_v2 (March 2) distribution sanity ----
sv2=pd.read_csv(ROOT/"data/processed/states_v2/Friday-02-03-2018_TrafficForML_CICFlowMeter_states_v2.csv")
dist=[]
for f in BASE:
    dist.append(dict(feature=f,
        A_mean=float(featA[f].mean()), A_std=float(featA[f].std()),
        train_sv2_mean=float(sv2[f].mean()), train_sv2_std=float(sv2[f].std()),
        A_minus_train_mean=float(featA[f].mean()-sv2[f].mean()),
        z_of_A_mean_in_train=float((featA[f].mean()-sv2[f].mean())/(sv2[f].std()+1e-9))))
D=pd.DataFrame(dist).sort_values("z_of_A_mean_in_train", key=lambda s:s.abs(), ascending=False)
D.to_csv(OUT/"A_vs_training_states_v2_distribution.csv", index=False)
print("\n=== A (host .30 pcap_v4) vs REAL training states_v2 March-2 distribution ===")
print("(z = how many training-std the A-mean sits from the training-mean)")
print(D.to_string(index=False))

json.dump(dict(n_shared_windows=int(len(m)),
    range_utc=[str(pd.to_datetime(m['win'].min(),unit='s')), str(pd.to_datetime(m['win'].max(),unit='s'))],
    n_packets=int(len(pdf)), n_flows_B=int(len(fl)),
    files={"pcap":str(PCAP), "training_feature_code":"archive/create_network_states_v2.py",
           "v4_feature_code":"src/pcap_ingest_v4.py"}),
    open(OUT/"parity_meta.json","w"), indent=2)
print("\nwrote:", *(p.name for p in sorted(OUT.glob("*"))), sep="\n  ")
