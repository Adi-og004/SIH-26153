"""
OLD vs CORRECTED pcap feature representation -- same PCAP (host .30).

  OLD  : pre-refactor pcap_ingest_v4 (frozen copy below): per-window
         5-tuple fragments, seconds durations/IAT, per-packet pooled
         flag rates, lexical-IP direction, frame-length bytes.
  NEW  : current src/pcap_ingest_v4 (training-definition emulation).
  REF  : standalone training-definition emulation (independent of NEW)
         -- used to check NEW faithfully implements the definitions.
  TRAIN: real states_v2 March-2 distribution (genuine CICFlowMeter).

Range 12:47-14:40 UTC. Analysis-only. No V4 inference.
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
W = 10
LO, HI = 1519994820.0, 1520001600.0     # 12:47:00 .. 14:40:00 UTC
BASE = P.BASE_FEATURES

# ---------------------------------------------------------------- parse once
print("parsing", PCAP.name, flush=True); t0=time.time()
rows=[]
with PcapReader(str(PCAP)) as rd:
    for p in rd:
        ts=float(p.time)
        if ts<LO or ts>=HI: continue
        l3=p.getlayer(IP) or p.getlayer(IPv6)
        if l3 is None: continue
        v6 = l3.name=="IPv6"
        proto=int(l3.nh) if v6 else int(l3.proto)
        sp=dp=0; l4=0; fl={k:False for k in P._FLAG_KEYS}
        if p.haslayer(TCP):
            t=p[TCP]; sp,dp=int(t.sport),int(t.dport); s=str(t.flags)
            for k in P._FLAG_KEYS: fl[k]=P._FLAG_LETTER[k] in s
            l4=len(t.payload)
        elif p.haslayer(UDP):
            u=p[UDP]; sp,dp=int(u.sport),int(u.dport); l4=len(u.payload)
        else: l4=len(l3.payload)
        r=dict(timestamp_abs=ts, src_ip=str(l3.src), dst_ip=str(l3.dst),
               src_port=sp, dst_port=dp, protocol=proto,
               frame_len=int(len(p)), l4_payload_len=int(l4)); r.update(fl)
        rows.append(r)
pdf=pd.DataFrame(rows).sort_values("timestamp_abs").reset_index(drop=True)
pdf.attrs["t0_epoch"]=float(pdf["timestamp_abs"].min())
print(f"  {len(pdf):,} pkts [{time.time()-t0:.1f}s]", flush=True)

# ================================================================ NEW (current module)
flowN = P.reconstruct_flows(pdf)
featN = P.compute_window_features(flowN)[["window_idx"]+BASE].rename(columns={"window_idx":"win"})

# ================================================================ OLD (frozen pre-refactor logic)
t0abs=float(pdf["timestamp_abs"].min())
o=pdf.copy()
o["timestamp"]=o["timestamp_abs"]-t0abs
o["pkt_len"]=o["frame_len"]
o["window_idx"]=(o["timestamp"]//W).astype(int)
lo_ip=np.minimum(o["src_ip"],o["dst_ip"]); hi_ip=np.maximum(o["src_ip"],o["dst_ip"])
lo_pt=np.minimum(o["src_port"],o["dst_port"]); hi_pt=np.maximum(o["src_port"],o["dst_port"])
o["flow_key"]=lo_ip+":"+hi_ip+":"+lo_pt.astype(str)+":"+hi_pt.astype(str)+":"+o["protocol"].astype(str)
o["is_forward"]=o["src_ip"]<=o["dst_ip"]
def _ent_old(vals):
    if len(vals)==0: return 0.0
    c=pd.Series(vals).value_counts(); pr=c/c.sum()
    return float(-np.sum(pr*np.log2(pr+1e-10)))
oflow=[]
for (wi,fk),g in o.groupby(["window_idx","flow_key"]):
    fwd=g[g["is_forward"]]; bwd=g[~g["is_forward"]]; ts=g["timestamp"].values
    dur=float(ts.max()-ts.min()) if len(ts)>1 else 0.0
    fi=np.diff(np.sort(fwd["timestamp"].values)) if len(fwd)>1 else np.array([0.0])
    bi=np.diff(np.sort(bwd["timestamp"].values)) if len(bwd)>1 else np.array([0.0])
    oflow.append(dict(window_idx=int(wi), total_packets=len(g), total_bytes=int(g["pkt_len"].sum()),
        fwd_packets=len(fwd), bwd_packets=len(bwd),
        fwd_bytes=int(fwd["pkt_len"].sum()) if len(fwd) else 0,
        bwd_bytes=int(bwd["pkt_len"].sum()) if len(bwd) else 0,
        duration=dur, mean_fwd_iat=float(fi.mean()), mean_bwd_iat=float(bi.mean()),
        pkt_lengths=g["pkt_len"].values.tolist(),
        dst_port=int(g["dst_port"].iloc[0]), protocol=int(g["protocol"].iloc[0]),
        down_up_ratio=len(bwd)/max(len(fwd),1)))
ofl=pd.DataFrame(oflow)
pkt_by_w=o.groupby("window_idx")
orows=[]
for wi in sorted(ofl["window_idx"].unique()):
    wf=ofl[ofl["window_idx"]==wi]
    try: wp=pkt_by_w.get_group(wi)
    except KeyError: wp=o.iloc[0:0]
    tp=int(wf["total_packets"].sum()); tb=int(wf["total_bytes"].sum())
    n=max(len(wp),1)
    def fr(c): return float(wp[c].sum())/n if len(wp) else 0.0
    al=[]; [al.extend(x) for x in wf["pkt_lengths"]]
    tfp=int(wf["fwd_packets"].sum()); tbp=int(wf["bwd_packets"].sum())
    tfb=int(wf["fwd_bytes"].sum()); tbb=int(wf["bwd_bytes"].sum())
    ptot=max(tfp+tbp,1); btot=max(tfb+tbb,1)
    orows.append(dict(win=int(wi),
        flow_count=len(wf), log_flow_count=np.log1p(len(wf)),
        log_total_packets=np.log1p(tp), log_total_bytes=np.log1p(tb),
        window_packet_rate=tp/W, window_byte_rate=tb/W,
        syn_rate=fr("f_syn"), ack_rate=fr("f_ack"), rst_rate=fr("f_rst"),
        fin_rate=fr("f_fin"), psh_rate=fr("f_psh"), urg_rate=fr("f_urg"),
        mean_pkt_len=float(np.mean(al)) if al else 0.0,
        std_pkt_len=float(np.std(al)) if len(al)>1 else 0.0,
        log_mean_flow_duration=np.log1p(float(wf["duration"].mean())),
        log_mean_fwd_iat=np.log1p(float(wf["mean_fwd_iat"].mean())),
        log_mean_bwd_iat=np.log1p(float(wf["mean_bwd_iat"].mean())),
        fwd_packet_ratio=tfp/ptot, bwd_packet_ratio=tbp/ptot,
        fwd_byte_ratio=tfb/btot, bwd_byte_ratio=tbb/btot,
        mean_down_up_ratio=float(wf["down_up_ratio"].mean()),
        log_unique_dst_ports=np.log1p(int(wf["dst_port"].nunique())),
        unique_protocols=int(wf["protocol"].nunique()),
        dst_port_entropy=_ent_old(wf["dst_port"].values),
        protocol_entropy=_ent_old(wf["protocol"].values)))
featO=pd.DataFrame(orows)[["win"]+BASE]

# ================================================================ REF (independent training-def emulation)
def sdiv(n,d):
    r=n/(np.abs(d)+1e-8); return np.where(np.isfinite(r),r,0.0)
def slog1p(x):
    x=np.array(x,float,copy=True); x[~np.isfinite(x)]=0.0; return np.log1p(np.clip(x,0,None))
def ent_from_counts(s):
    c=pd.Series(s).value_counts(dropna=True)
    if len(c)<=1: return 0.0
    p=c.to_numpy(float)/c.sum(); return float(-np.sum(p*np.log2(p+1e-12)))
kk=(np.minimum(pdf["src_ip"],pdf["dst_ip"])+"|"+np.maximum(pdf["src_ip"],pdf["dst_ip"])+"|"+
    np.minimum(pdf["src_port"],pdf["dst_port"]).astype(str)+"|"+
    np.maximum(pdf["src_port"],pdf["dst_port"]).astype(str)+"|"+pdf["protocol"].astype(str))
pr=pdf.assign(_k=kk.values); win0=int(pr["timestamp_abs"].min()//W)
rf=[]
for _,g in pr.groupby("_k",sort=False):
    g=g.sort_values("timestamp_abs"); ts=g["timestamp_abs"].to_numpy()
    br=np.nonzero(np.diff(ts)>120.0)[0]+1 if len(ts)>1 else np.array([],int)
    for seg in (np.split(np.arange(len(ts)),br) if len(ts)>1 else [np.arange(len(ts))]):
        if len(seg)==0: continue
        s=g.iloc[seg]; f0=s.iloc[0]
        fwd=(s["src_ip"]==f0["src_ip"])&(s["src_port"]==f0["src_port"])
        tfp=int(fwd.sum()); tbp=len(s)-tfp; st=s["timestamp_abs"].to_numpy()
        pay=s["l4_payload_len"].to_numpy().astype(float)
        fi=np.diff(np.sort(st[fwd.values]))*1e6 if tfp>1 else np.array([])
        bi=np.diff(np.sort(st[~fwd.values]))*1e6 if tbp>1 else np.array([])
        rf.append(dict(win=int(st[0]//W)-win0, dst_port=int(f0["dst_port"]), protocol=int(f0["protocol"]),
            tot_pkts=len(s), tot_fwd=tfp, tot_bwd=tbp,
            totlen_fwd=int(pay[fwd.values].sum()), totlen_bwd=int(pay[~fwd.values].sum()),
            total_bytes=int(pay.sum()),
            flow_dur=float((st[-1]-st[0])*1e6) if len(st)>1 else 0.0,
            fwd_iat=float(fi.mean()) if fi.size else 0.0, bwd_iat=float(bi.mean()) if bi.size else 0.0,
            plen_mean=float(pay.mean()) if pay.size else 0.0,
            plen_std=float(pay.std(ddof=1)) if pay.size>1 else 0.0,
            syn=int(bool(s["f_syn"].any())), ack=int(bool(s["f_ack"].any())),
            rst=int(bool(s["f_rst"].any())), fin=int(bool(s["f_fin"].any())),
            psh=int(bool(s["f_psh"].any())), urg=int(bool(s["f_urg"].any())),
            downup=int(tbp//tfp) if tfp>0 else 0))
R=pd.DataFrame(rf)
for c,a,b in [("fwd_packet_ratio","tot_fwd","tot_pkts"),("bwd_packet_ratio","tot_bwd","tot_pkts"),
              ("fwd_byte_ratio","totlen_fwd","total_bytes"),("bwd_byte_ratio","totlen_bwd","total_bytes")]:
    R[c]=sdiv(R[a],R[b])
den=R["tot_pkts"]+1e-8
for c,f in [("syn_rate","syn"),("ack_rate","ack"),("rst_rate","rst"),("fin_rate","fin"),("psh_rate","psh"),("urg_rate","urg")]:
    R[c]=R[f]/den
gg=R.groupby("win")
REF=gg.agg(flow_count=("win","size"),total_packets=("tot_pkts","sum"),total_bytes=("total_bytes","sum"),
    mean_flow_duration=("flow_dur","mean"),mean_fwd_iat=("fwd_iat","mean"),mean_bwd_iat=("bwd_iat","mean"),
    mean_pkt_len=("plen_mean","mean"),std_pkt_len=("plen_std","mean"),
    syn_rate=("syn_rate","mean"),ack_rate=("ack_rate","mean"),rst_rate=("rst_rate","mean"),
    fin_rate=("fin_rate","mean"),psh_rate=("psh_rate","mean"),urg_rate=("urg_rate","mean"),
    fwd_packet_ratio=("fwd_packet_ratio","mean"),bwd_packet_ratio=("bwd_packet_ratio","mean"),
    fwd_byte_ratio=("fwd_byte_ratio","mean"),bwd_byte_ratio=("bwd_byte_ratio","mean"),
    mean_down_up_ratio=("downup","mean"),
    unique_dst_ports=("dst_port","nunique"),unique_protocols=("protocol","nunique")).reset_index()
REF["log_flow_count"]=slog1p(REF["flow_count"]); REF["log_total_packets"]=slog1p(REF["total_packets"])
REF["log_total_bytes"]=slog1p(REF["total_bytes"]); REF["log_mean_flow_duration"]=slog1p(REF["mean_flow_duration"])
REF["log_mean_fwd_iat"]=slog1p(REF["mean_fwd_iat"]); REF["log_mean_bwd_iat"]=slog1p(REF["mean_bwd_iat"])
REF["log_unique_dst_ports"]=slog1p(REF["unique_dst_ports"])
REF["window_packet_rate"]=REF["total_packets"]/W; REF["window_byte_rate"]=REF["total_bytes"]/W
REF["dst_port_entropy"]=gg["dst_port"].apply(ent_from_counts).values
REF["protocol_entropy"]=gg["protocol"].apply(ent_from_counts).values
featR=REF[["win"]+BASE]

# ================================================================ merge + metrics
m=featO.merge(featN,on="win",suffixes=("_OLD","_NEW")).merge(
    featR.rename(columns={c:c+"_REF" for c in BASE}),on="win")
m=m[(m.win*W>=LO-1519994820.0)|(True)].sort_values("win").reset_index(drop=True)
m.to_csv(OUT/"parity_v2_windows.csv",index=False)
sv2=pd.read_csv(ROOT/"data/processed/states_v2/Friday-02-03-2018_TrafficForML_CICFlowMeter_states_v2.csv")

def pear(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-12 and b.std()>1e-12 else np.nan
rec=[]
for f in BASE:
    O=m[f+"_OLD"].to_numpy(float); N=m[f+"_NEW"].to_numpy(float); Rr=m[f+"_REF"].to_numpy(float)
    tm,tsd=float(sv2[f].mean()), float(sv2[f].std())+1e-9
    rec.append(dict(feature=f,
        mean_OLD=O.mean(), mean_NEW=N.mean(), mean_REF=Rr.mean(), train_mean=tm, train_std=float(sv2[f].std()),
        z_OLD=(O.mean()-tm)/tsd, z_NEW=(N.mean()-tm)/tsd,
        absZ_reduction=abs((O.mean()-tm)/tsd)-abs((N.mean()-tm)/tsd),
        MAD_OLD_vs_NEW=float(np.mean(np.abs(O-N))),
        r_NEW_vs_REF=pear(N,Rr), r_OLD_vs_REF=pear(O,Rr),
        NEW_matches_REF=bool(np.allclose(N,Rr,rtol=1e-3,atol=1e-6))))
S=pd.DataFrame(rec)
S["absZ_OLD"]=S["z_OLD"].abs(); S["absZ_NEW"]=S["z_NEW"].abs()
S=S.sort_values("absZ_NEW").reset_index(drop=True)
S.to_csv(OUT/"parity_v2_feature_summary.csv",index=False)
pd.set_option("display.width",200); pd.set_option("display.max_columns",30)
print("\n=== OLD vs NEW vs training distribution (sorted by |z_NEW|) ===")
print(S[["feature","mean_OLD","mean_NEW","train_mean","z_OLD","z_NEW","absZ_reduction","r_NEW_vs_REF","NEW_matches_REF"]].to_string(index=False))
print("\nfeatures where NEW does NOT match independent REF emulation:")
print(S[~S.NEW_matches_REF][["feature","mean_NEW","mean_REF","r_NEW_vs_REF"]].to_string(index=False))
print(f"\nmean |z| OLD = {S.absZ_OLD.mean():.2f}   mean |z| NEW = {S.absZ_NEW.mean():.2f}")
print(f"features with |z_NEW| < 2 : {(S.absZ_NEW<2).sum()}/26   (was {(S.absZ_OLD<2).sum()}/26 for OLD)")

# plots: the 3 timing + ack/psh + one direction
show=["log_mean_flow_duration","log_mean_fwd_iat","log_mean_bwd_iat","ack_rate","psh_rate","fwd_packet_ratio"]
tt=pd.to_datetime(m["win"]*W + int(pdf.attrs["t0_epoch"]//W*W), unit="s")
fig,ax=plt.subplots(3,2,figsize=(15,13))
for a,f in zip(ax.ravel(),show):
    a.plot(tt,m[f+"_OLD"],label="OLD",lw=1)
    a.plot(tt,m[f+"_NEW"],label="NEW",lw=1.3)
    a.axhline(sv2[f].mean(),color="green",ls="--",lw=1,label="train mean")
    a.axvline(pd.to_datetime(1520000008,unit="s"),color="k",ls=":",lw=1)
    a.set_title(f"{f}  z_OLD={S.loc[S.feature==f,'z_OLD'].values[0]:.1f}  z_NEW={S.loc[S.feature==f,'z_NEW'].values[0]:.1f}")
    a.legend(fontsize=7); a.tick_params(axis="x",labelrotation=25,labelsize=7)
fig.suptitle("Host .30 -- OLD vs CORRECTED representation vs training mean",y=1.0)
fig.tight_layout(); fig.savefig(OUT/"parity_v2_old_vs_new.png",dpi=130); plt.close(fig)

json.dump(dict(n_windows=int(len(m)),n_packets=int(len(pdf)),
    mean_absZ_OLD=float(S.absZ_OLD.mean()),mean_absZ_NEW=float(S.absZ_NEW.mean()),
    n_feat_z_lt2_OLD=int((S.absZ_OLD<2).sum()),n_feat_z_lt2_NEW=int((S.absZ_NEW<2).sum()),
    NEW_matches_REF_count=int(S.NEW_matches_REF.sum())),
    open(OUT/"parity_v2_meta.json","w"),indent=2)
print("\nwrote parity_v2_* to",OUT)
