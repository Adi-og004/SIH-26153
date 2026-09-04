import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
]


def analyze_file(filename: str) -> None:
    path = DATA_DIR / filename

    print("\n" + "=" * 80)
    print(f"DATASET: {filename}")
    print("=" * 80)

    df = pd.read_csv(path)

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Timestamp
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    invalid_timestamps = df["Timestamp"].isna().sum()

    print(f"Invalid timestamps: {invalid_timestamps}")

    if invalid_timestamps == len(df):
        print("ERROR: Could not parse timestamps.")
        return

    print(f"Start: {df['Timestamp'].min()}")
    print(f"End:   {df['Timestamp'].max()}")

    # Label distribution
    print("\n--- LABEL DISTRIBUTION ---")
    labels = df["Label"].value_counts()

    for label, count in labels.items():
        percentage = count / len(df) * 100
        print(f"{str(label):35s} {count:>10,}  ({percentage:6.2f}%)")

    # Attack-only data
    attack_df = df[
        df["Label"].astype(str).str.strip().str.lower() != "benign"
    ]

    print("\n--- ATTACK INFORMATION ---")

    if attack_df.empty:
        print("No attack flows found.")
    else:
        print(f"Attack flows: {len(attack_df):,}")
        print(f"First attack: {attack_df['Timestamp'].min()}")
        print(f"Last attack:  {attack_df['Timestamp'].max()}")

        print("\nAttack labels:")
        for label, count in attack_df["Label"].value_counts().items():
            print(f"  {label}: {count:,}")

        # 10-second windows
        df["time_window"] = df["Timestamp"].dt.floor("10s")

        attack_df = attack_df.copy()
        attack_df["time_window"] = attack_df["Timestamp"].dt.floor("10s")

        attack_windows = attack_df["time_window"].nunique()
        total_windows = df["time_window"].nunique()

        print(f"\nTotal 10-second windows:  {total_windows:,}")
        print(f"Attack 10-second windows: {attack_windows:,}")


def main():
    print("Analyzing CIC-IDS2018 datasets...")

    for filename in FILES:
        analyze_file(filename)

    print("\n" + "=" * 80)
    print("ALL DATASETS ANALYZED")
    print("=" * 80)


if __name__ == "__main__":
    main()