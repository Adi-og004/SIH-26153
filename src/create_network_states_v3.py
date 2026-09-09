import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed/states_v3")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SECONDS = 10

# We'll just run it on March 2 for this experiment, 
# but script supports all.
FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
]

# =========================================================
# FEATURES USED TO BUILD THE NETWORK STATE
# =========================================================

REQUIRED_COLUMNS = [
    "Timestamp",
    "Protocol",
    "Dst Port",

    "Flow Duration",

    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",

    "Pkt Len Mean",
    "Pkt Len Std",
    
    # --- NEW V3 COLUMNS ---
    "Pkt Len Var",
    "Fwd Pkt Len Std",
    "Bwd Pkt Len Std",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
    "Fwd Header Len",
    "Bwd Header Len",
    "Fwd Act Data Pkts",
    "Active Mean",
    "Active Std",
    "Idle Mean",
    "Idle Std",
    "Subflow Fwd Pkts",
    "Subflow Fwd Byts",
    "Subflow Bwd Pkts",
    "Subflow Bwd Byts",
    # ----------------------

    "Fwd IAT Mean",
    "Bwd IAT Mean",

    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",

    "Down/Up Ratio",

    "Label",
]

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_log1p(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan)
    series = series.fillna(0.0)
    series = series.clip(lower=0.0)
    return np.log1p(series)

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / (denominator.abs() + 1e-8)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.fillna(0.0)

def entropy_from_counts(series: pd.Series) -> float:
    counts = series.value_counts(dropna=True)
    if len(counts) <= 1:
        return 0.0
    probabilities = counts.to_numpy(dtype=float) / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
    return float(entropy)

# =========================================================
# CREATE ONE EPISODE
# =========================================================

def create_states(filename: str):

    input_path = DATA_DIR / filename
    if not input_path.exists():
        print(f"Skipping {filename}: Not found in {DATA_DIR}")
        return

    output_filename = filename.replace(".csv", "_states_v3.csv")
    output_path = OUTPUT_DIR / output_filename

    print("\n" + "=" * 90)
    print(f"PROCESSING V3: {filename}")
    print("=" * 90)

    print("Loading dataset...")
    df = pd.read_csv(input_path, usecols=REQUIRED_COLUMNS, low_memory=False)
    print(f"Raw rows: {len(df):,}")

    # Timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    numeric_columns = [c for c in REQUIRED_COLUMNS if c not in ["Timestamp", "Label"]]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    # Basic packet/byte quantities
    df["total_packets"] = df["Tot Fwd Pkts"].fillna(0.0) + df["Tot Bwd Pkts"].fillna(0.0)
    df["total_bytes"] = df["TotLen Fwd Pkts"].fillna(0.0) + df["TotLen Bwd Pkts"].fillna(0.0)

    # Short flow boolean (duration < 1ms OR total packets <= 2)
    # Using 1000 microseconds = 1ms since CICFlowMeter uses microseconds for duration.
    df["is_short_flow"] = ((df["Flow Duration"].fillna(0.0) < 1000) | (df["total_packets"] <= 2)).astype(float)

    # V3 Data packet ratio (Proxy for payload vs overhead/retransmissions)
    df["fwd_data_pkt_ratio"] = safe_divide(df["Fwd Act Data Pkts"].fillna(0.0), df["Tot Fwd Pkts"].fillna(0.0))
    
    # Header ratios
    df["fwd_header_ratio"] = safe_divide(df["Fwd Header Len"].fillna(0.0), df["total_bytes"] + 1)
    df["bwd_header_ratio"] = safe_divide(df["Bwd Header Len"].fillna(0.0), df["total_bytes"] + 1)

    # Direction ratios
    df["fwd_packet_ratio"] = safe_divide(df["Tot Fwd Pkts"].fillna(0.0), df["total_packets"])
    df["bwd_packet_ratio"] = safe_divide(df["Tot Bwd Pkts"].fillna(0.0), df["total_packets"])
    df["fwd_byte_ratio"] = safe_divide(df["TotLen Fwd Pkts"].fillna(0.0), df["total_bytes"])
    df["bwd_byte_ratio"] = safe_divide(df["TotLen Bwd Pkts"].fillna(0.0), df["total_bytes"])

    # TCP flag rates per packet
    packet_denominator = df["total_packets"] + 1e-8
    df["syn_rate"] = df["SYN Flag Cnt"].fillna(0.0) / packet_denominator
    df["ack_rate"] = df["ACK Flag Cnt"].fillna(0.0) / packet_denominator
    df["rst_rate"] = df["RST Flag Cnt"].fillna(0.0) / packet_denominator
    df["fin_rate"] = df["FIN Flag Cnt"].fillna(0.0) / packet_denominator
    df["psh_rate"] = df["PSH Flag Cnt"].fillna(0.0) / packet_denominator
    df["urg_rate"] = df["URG Flag Cnt"].fillna(0.0) / packet_denominator

    # Create 10-second windows
    df["window_start"] = df["Timestamp"].dt.floor(f"{WINDOW_SECONDS}s")

    # Attack metadata
    df["is_attack"] = df["Label"].astype(str).str.strip().str.lower().ne("benign")

    # Aggregate normal state features
    grouped = df.groupby("window_start", sort=True)
    states = grouped.agg(
        # Flow stats
        flow_count=("Timestamp", "size"),
        total_packets=("total_packets", "sum"),
        total_bytes=("total_bytes", "sum"),
        
        # Timing
        mean_flow_duration=("Flow Duration", "mean"),
        mean_fwd_iat=("Fwd IAT Mean", "mean"),
        mean_bwd_iat=("Bwd IAT Mean", "mean"),
        
        # Burstiness (NEW)
        active_mean=("Active Mean", "mean"),
        active_std=("Active Std", "mean"),
        idle_mean=("Idle Mean", "mean"),
        idle_std=("Idle Std", "mean"),
        
        # Short flows (NEW)
        short_flow_ratio=("is_short_flow", "mean"),

        # Packet length stats (V2 + NEW)
        mean_pkt_len=("Pkt Len Mean", "mean"),
        std_pkt_len=("Pkt Len Std", "mean"),
        var_pkt_len=("Pkt Len Var", "mean"),
        std_fwd_pkt_len=("Fwd Pkt Len Std", "mean"),
        std_bwd_pkt_len=("Bwd Pkt Len Std", "mean"),

        # TCP Windows (NEW)
        mean_init_fwd_win=("Init Fwd Win Byts", "mean"),
        mean_init_bwd_win=("Init Bwd Win Byts", "mean"),
        
        # Header/Data (NEW)
        mean_fwd_data_pkt_ratio=("fwd_data_pkt_ratio", "mean"),
        mean_fwd_header_ratio=("fwd_header_ratio", "mean"),
        mean_bwd_header_ratio=("bwd_header_ratio", "mean"),
        
        # Subflow (NEW)
        mean_subflow_fwd_pkts=("Subflow Fwd Pkts", "mean"),
        mean_subflow_fwd_byts=("Subflow Fwd Byts", "mean"),
        mean_subflow_bwd_pkts=("Subflow Bwd Pkts", "mean"),
        mean_subflow_bwd_byts=("Subflow Bwd Byts", "mean"),

        # TCP Flags
        syn_rate=("syn_rate", "mean"),
        ack_rate=("ack_rate", "mean"),
        rst_rate=("rst_rate", "mean"),
        fin_rate=("fin_rate", "mean"),
        psh_rate=("psh_rate", "mean"),
        urg_rate=("urg_rate", "mean"),

        # Direction
        fwd_packet_ratio=("fwd_packet_ratio", "mean"),
        bwd_packet_ratio=("bwd_packet_ratio", "mean"),
        fwd_byte_ratio=("fwd_byte_ratio", "mean"),
        bwd_byte_ratio=("bwd_byte_ratio", "mean"),
        mean_down_up_ratio=("Down/Up Ratio", "mean"),

        # Diversity
        unique_dst_ports=("Dst Port", "nunique"),
        unique_protocols=("Protocol", "nunique"),

    ).reset_index()

    # Log-transform highly skewed quantities
    states["log_total_packets"] = safe_log1p(states["total_packets"])
    states["log_total_bytes"] = safe_log1p(states["total_bytes"])
    states["log_flow_count"] = safe_log1p(states["flow_count"])
    states["log_mean_flow_duration"] = safe_log1p(states["mean_flow_duration"])
    states["log_mean_fwd_iat"] = safe_log1p(states["mean_fwd_iat"])
    states["log_mean_bwd_iat"] = safe_log1p(states["mean_bwd_iat"])
    states["log_unique_dst_ports"] = safe_log1p(states["unique_dst_ports"])
    
    # NEW log transforms
    states["log_active_mean"] = safe_log1p(states["active_mean"])
    states["log_idle_mean"] = safe_log1p(states["idle_mean"])

    states["window_packet_rate"] = states["total_packets"] / WINDOW_SECONDS
    states["window_byte_rate"] = states["total_bytes"] / WINDOW_SECONDS

    # Entropy
    port_entropy = grouped["Dst Port"].apply(entropy_from_counts).rename("dst_port_entropy").reset_index()
    protocol_entropy = grouped["Protocol"].apply(entropy_from_counts).rename("protocol_entropy").reset_index()
    states = states.merge(port_entropy, on="window_start", how="left")
    states = states.merge(protocol_entropy, on="window_start", how="left")

    # Ground truth
    attack_info = grouped.agg(
        attack_count=("is_attack", "sum"),
        total_window_flows=("is_attack", "size")
    ).reset_index()
    attack_info["attack_ratio"] = attack_info["attack_count"] / attack_info["total_window_flows"]
    attack_info["attack_present"] = (attack_info["attack_count"] > 0).astype(int)

    attack_df = df[df["is_attack"]]
    if not attack_df.empty:
        dominant_attack = attack_df.groupby("window_start")["Label"].agg(
            lambda values: values.value_counts().index[0]
        ).reset_index()
        dominant_attack.rename(columns={"Label": "dominant_attack"}, inplace=True)
    else:
        dominant_attack = pd.DataFrame(columns=["window_start", "dominant_attack"])

    states = states.merge(attack_info, on="window_start", how="left")
    states = states.merge(dominant_attack, on="window_start", how="left")
    states["dominant_attack"] = states["dominant_attack"].fillna("Benign")

    states = states.replace([np.inf, -np.inf], np.nan)
    numeric_state_columns = states.select_dtypes(include=[np.number]).columns
    states[numeric_state_columns] = states[numeric_state_columns].fillna(0.0)

    # Base features from V2
    v2_features = [
        "flow_count", "log_flow_count", "log_total_packets", "log_total_bytes",
        "window_packet_rate", "window_byte_rate", "syn_rate", "ack_rate",
        "rst_rate", "fin_rate", "psh_rate", "urg_rate", "mean_pkt_len",
        "std_pkt_len", "log_mean_flow_duration", "log_mean_fwd_iat",
        "log_mean_bwd_iat", "fwd_packet_ratio", "bwd_packet_ratio",
        "fwd_byte_ratio", "bwd_byte_ratio", "mean_down_up_ratio",
        "log_unique_dst_ports", "unique_protocols", "dst_port_entropy",
        "protocol_entropy",
    ]
    
    # New V3 features
    v3_new_features = [
        "var_pkt_len", "std_fwd_pkt_len", "std_bwd_pkt_len",
        "mean_init_fwd_win", "mean_init_bwd_win",
        "mean_fwd_data_pkt_ratio", "mean_fwd_header_ratio", "mean_bwd_header_ratio",
        "log_active_mean", "active_std", "log_idle_mean", "idle_std",
        "short_flow_ratio",
        "mean_subflow_fwd_pkts", "mean_subflow_fwd_byts",
        "mean_subflow_bwd_pkts", "mean_subflow_bwd_byts"
    ]
    
    state_feature_columns = v2_features + v3_new_features

    metadata_columns = [
        "attack_count",
        "attack_ratio",
        "attack_present",
        "dominant_attack",
    ]

    final_columns = ["window_start"] + state_feature_columns + metadata_columns
    states = states[final_columns]

    print(f"Generated {len(states)} V3 states with {len(state_feature_columns)} features.")
    
    states.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    # For Step 2A, we just need March 2 to test the Bot onset.
    create_states("Friday-02-03-2018_TrafficForML_CICFlowMeter.csv")
