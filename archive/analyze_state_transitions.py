import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

STATE_DIR = Path("data/processed/states")
OUTPUT_DIR = Path("data/transition_analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter_states.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter_states.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_states.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter_states.csv",
]


# These are the network-state features.
# Ground-truth columns such as attack_present and attack_ratio
# are deliberately excluded from the model-state analysis.
STATE_FEATURES = [
    "flow_count",
    "total_packets",
    "total_bytes",
    "syn_count",
    "ack_count",
    "rst_count",
    "fin_count",
    "psh_count",
    "urg_count",
    "mean_flow_duration",
    "mean_pkt_len",
    "std_pkt_len",
    "mean_fwd_iat",
    "mean_bwd_iat",
    "mean_down_up_ratio",
    "unique_dst_ports",
    "unique_protocols",
    "window_packet_rate",
    "window_byte_rate",
]


# Features that are particularly useful for visualization.
PLOT_FEATURES = [
    "flow_count",
    "window_packet_rate",
    "window_byte_rate",
    "syn_count",
    "ack_count",
    "rst_count",
    "unique_dst_ports",
]


# =========================================================
# ANALYZE ONE DATASET
# =========================================================

def analyze_dataset(filename: str):

    path = STATE_DIR / filename

    print("\n" + "=" * 90)
    print(f"DATASET: {filename}")
    print("=" * 90)

    df = pd.read_csv(path)

    df["window_start"] = pd.to_datetime(
        df["window_start"]
    )

    df = df.sort_values(
        "window_start"
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Calculate state transitions
    # -----------------------------------------------------

    for feature in STATE_FEATURES:

        df[f"{feature}_delta"] = (
            df[feature].diff()
        )

    # -----------------------------------------------------
    # Find first attack state
    # -----------------------------------------------------

    attack_indices = df.index[
        df["attack_present"] == 1
    ]

    if len(attack_indices) == 0:

        print("No attack states found.")
        return

    first_attack_idx = attack_indices[0]

    first_attack_time = df.loc[
        first_attack_idx,
        "window_start"
    ]

    print(
        f"First attack state: index={first_attack_idx}, "
        f"time={first_attack_time}"
    )

    # -----------------------------------------------------
    # Examine states before first attack
    # -----------------------------------------------------

    LOOKBACK = 10

    start_idx = max(
        0,
        first_attack_idx - LOOKBACK
    )

    pre_attack = df.iloc[
        start_idx:first_attack_idx + 1
    ]

    print(
        f"\nShowing {len(pre_attack)} states around "
        f"first attack:"
    )

    print(
        pre_attack[
            [
                "window_start",
                "flow_count",
                "syn_count",
                "ack_count",
                "rst_count",
                "unique_dst_ports",
                "attack_present",
                "attack_ratio",
                "dominant_attack",
            ]
        ].to_string(index=False)
    )

    # -----------------------------------------------------
    # Compare normal states with attack states
    # -----------------------------------------------------

    normal_states = df[
        df["attack_present"] == 0
    ]

    attack_states = df[
        df["attack_present"] == 1
    ]

    print("\n--- FEATURE COMPARISON ---")

    comparison = pd.DataFrame({
        "Normal Mean": normal_states[STATE_FEATURES].mean(),
        "Attack Mean": attack_states[STATE_FEATURES].mean(),
    })

    comparison["Change %"] = (
        (
            comparison["Attack Mean"]
            - comparison["Normal Mean"]
        )
        / comparison["Normal Mean"].replace(
            0,
            np.nan
        )
        * 100
    )

    print(
        comparison.sort_values(
            "Change %",
            ascending=False
        ).round(3).to_string()
    )

    # -----------------------------------------------------
    # Transition magnitude
    # -----------------------------------------------------

    delta_columns = [
        f"{feature}_delta"
        for feature in STATE_FEATURES
    ]

    transition_magnitude = df[
        delta_columns
    ].abs().mean(axis=1)

    df["transition_magnitude"] = (
        transition_magnitude
    )

    normal_transition = df.loc[
        df["attack_present"] == 0,
        "transition_magnitude"
    ]

    attack_transition = df.loc[
        df["attack_present"] == 1,
        "transition_magnitude"
    ]

    print("\n--- TRANSITION MAGNITUDE ---")

    print(
        "Normal mean:",
        normal_transition.mean()
    )

    print(
        "Attack mean:",
        attack_transition.mean()
    )

    # -----------------------------------------------------
    # Plot selected features
    # -----------------------------------------------------

    plot_end = min(
        len(df),
        first_attack_idx + 150
    )

    plot_start = max(
        0,
        first_attack_idx - 150
    )

    plot_df = df.iloc[
        plot_start:plot_end
    ].copy()

    for feature in PLOT_FEATURES:

        plt.figure(figsize=(14, 5))

        plt.plot(
            plot_df["window_start"],
            plot_df[feature],
            linewidth=1.5
        )

        attack_times = plot_df.loc[
            plot_df["attack_present"] == 1,
            "window_start"
        ]

        if len(attack_times) > 0:
            plt.axvline(
                first_attack_time,
                linestyle="--",
                label="First attack state"
            )

        plt.title(
            f"{filename}\n"
            f"{feature} around attack onset"
        )

        plt.xlabel("Time")
        plt.ylabel(feature)

        plt.xticks(rotation=45)

        plt.legend()

        plt.tight_layout()

        output_file = (
            OUTPUT_DIR
            / f"{filename[:20]}_{feature}.png"
        )

        plt.savefig(
            output_file,
            dpi=150
        )

        plt.close()

    print(
        f"\nPlots saved to: {OUTPUT_DIR}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("NETWORK STATE TRANSITION ANALYSIS")

    for filename in FILES:

        analyze_dataset(filename)

    print("\n" + "=" * 90)
    print("TRANSITION ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()