import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed/states")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SECONDS = 10

FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
]


# =========================================================
# HELPER FUNCTION
# =========================================================

def safe_sum(series):
    """Sum values while ignoring NaN and infinite values."""
    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan)
    return series.sum()


def safe_mean(series):
    """Mean while ignoring NaN and infinite values."""
    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan)
    return series.mean()


def safe_std(series):
    """Standard deviation while ignoring NaN and infinite values."""
    series = pd.to_numeric(series, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan)
    return series.std()


# =========================================================
# CREATE NETWORK STATES
# =========================================================

def create_states(filename):

    input_path = DATA_DIR / filename

    print("\n" + "=" * 90)
    print(f"PROCESSING: {filename}")
    print("=" * 90)

    print("Loading dataset...")

    # Only load the columns we currently need.
    required_columns = [
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

    df = pd.read_csv(
        input_path,
        usecols=required_columns,
        low_memory=False
    )

    # -----------------------------------------------------
    # Timestamp
    # -----------------------------------------------------

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    df = df.dropna(subset=["Timestamp"])

    df = df.sort_values("Timestamp").reset_index(drop=True)

    # -----------------------------------------------------
    # Replace problematic infinite values
    # -----------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # -----------------------------------------------------
    # Basic packet / byte calculations
    # -----------------------------------------------------

    df["total_packets"] = (
        df["Tot Fwd Pkts"].fillna(0)
        + df["Tot Bwd Pkts"].fillna(0)
    )

    df["total_bytes"] = (
        df["TotLen Fwd Pkts"].fillna(0)
        + df["TotLen Bwd Pkts"].fillna(0)
    )

    # -----------------------------------------------------
    # Create 10-second windows
    # -----------------------------------------------------

    df["window_start"] = df["Timestamp"].dt.floor(
        f"{WINDOW_SECONDS}s"
    )

    # -----------------------------------------------------
    # Attack indicator
    # -----------------------------------------------------

    df["is_attack"] = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .ne("benign")
    )

    # -----------------------------------------------------
    # Aggregate state features
    # -----------------------------------------------------

    grouped = df.groupby(
        "window_start",
        sort=True
    )

    states = grouped.agg(
        flow_count=("Timestamp", "size"),

        total_packets=("total_packets", "sum"),
        total_bytes=("total_bytes", "sum"),

        syn_count=("SYN Flag Cnt", "sum"),
        ack_count=("ACK Flag Cnt", "sum"),
        rst_count=("RST Flag Cnt", "sum"),
        fin_count=("FIN Flag Cnt", "sum"),
        psh_count=("PSH Flag Cnt", "sum"),
        urg_count=("URG Flag Cnt", "sum"),

        mean_flow_duration=("Flow Duration", "mean"),

        mean_pkt_len=("Pkt Len Mean", "mean"),
        std_pkt_len=("Pkt Len Std", "mean"),

        mean_fwd_iat=("Fwd IAT Mean", "mean"),
        mean_bwd_iat=("Bwd IAT Mean", "mean"),

        mean_down_up_ratio=("Down/Up Ratio", "mean"),

        unique_dst_ports=("Dst Port", "nunique"),
        unique_protocols=("Protocol", "nunique"),
    ).reset_index()

    # -----------------------------------------------------
    # Calculate WINDOW-level rates
    # -----------------------------------------------------

    states["window_packet_rate"] = (
        states["total_packets"]
        / WINDOW_SECONDS
    )

    states["window_byte_rate"] = (
        states["total_bytes"]
        / WINDOW_SECONDS
    )

    # -----------------------------------------------------
    # Attack ground truth
    # -----------------------------------------------------

    attack_info = grouped.agg(
        attack_count=("is_attack", "sum"),
        total_window_flows=("is_attack", "size"),
    ).reset_index()

    attack_info["attack_ratio"] = (
        attack_info["attack_count"]
        / attack_info["total_window_flows"]
    )

    attack_info["attack_present"] = (
        attack_info["attack_count"] > 0
    ).astype(int)

    # -----------------------------------------------------
    # Dominant attack type
    # -----------------------------------------------------

    attack_labels = (
        df[df["is_attack"]]
        .groupby("window_start")["Label"]
        .agg(
            lambda x: x.value_counts().index[0]
        )
        .reset_index()
    )

    attack_labels.rename(
        columns={"Label": "dominant_attack"},
        inplace=True
    )

    # -----------------------------------------------------
    # Merge ground truth
    # -----------------------------------------------------

    states = states.merge(
        attack_info[
            [
                "window_start",
                "attack_count",
                "attack_ratio",
                "attack_present",
            ]
        ],
        on="window_start",
        how="left"
    )

    states = states.merge(
        attack_labels,
        on="window_start",
        how="left"
    )

    states["dominant_attack"] = (
        states["dominant_attack"]
        .fillna("Benign")
    )

    # -----------------------------------------------------
    # Final cleanup
    # -----------------------------------------------------

    states = states.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # For the first version, fill numerical missing values
    # with zero only after aggregation.
    numeric_state_columns = states.select_dtypes(
        include=[np.number]
    ).columns

    states[numeric_state_columns] = (
        states[numeric_state_columns]
        .fillna(0)
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_filename = filename.replace(
        ".csv",
        "_states.csv"
    )

    output_path = OUTPUT_DIR / output_filename

    states.to_csv(
        output_path,
        index=False
    )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    print("\nSTATE GENERATION COMPLETE")

    print("Number of states:", len(states))

    print(
        "Attack states:",
        states["attack_present"].sum()
    )

    print(
        "State features:",
        len(states.columns)
    )

    print("\nFirst five states:")
    print(states.head())

    print("\nSaved:")
    print(output_path)


# =========================================================
# MAIN
# =========================================================

def main():

    print("NETWORK STATE GENERATOR")
    print(
        f"Window size: {WINDOW_SECONDS} seconds"
    )

    for filename in FILES:
        create_states(filename)

    print("\n" + "=" * 90)
    print("ALL NETWORK STATES CREATED")
    print("=" * 90)


if __name__ == "__main__":
    main()