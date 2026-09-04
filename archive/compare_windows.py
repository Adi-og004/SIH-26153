import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
]

WINDOWS = ["5s", "10s", "30s"]


def analyze_file(filename):
    path = DATA_DIR / filename

    print("\n" + "=" * 90)
    print(filename)
    print("=" * 90)

    df = pd.read_csv(
        path,
        usecols=["Timestamp", "Label"]
    )

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df["is_attack"] = (
        df["Label"].str.strip().str.lower() != "benign"
    )

    for window in WINDOWS:

        temp = df.copy()

        temp["window"] = temp["Timestamp"].dt.floor(window)

        total = (
            temp.groupby("window")
            .size()
        )

        attack = (
            temp.groupby("window")["is_attack"]
            .sum()
        )

        attack_ratio = (
            attack / total
        ).fillna(0)

        print(f"\n--- {window} WINDOWS ---")

        print("Number of windows:", len(total))
        print("Mean flows/window:", round(total.mean(), 2))
        print("Median flows/window:", round(total.median(), 2))
        print("Min flows/window:", total.min())
        print("Max flows/window:", total.max())

        attack_windows = (attack > 0).sum()

        print("Attack windows:", attack_windows)

        if attack_windows > 0:
            print(
                "Mean attack ratio in attack windows:",
                round(
                    attack_ratio[attack_ratio > 0].mean() * 100,
                    2
                ),
                "%"
            )


def main():
    for filename in FILES:
        analyze_file(filename)


if __name__ == "__main__":
    main()