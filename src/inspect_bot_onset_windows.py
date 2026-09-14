from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = Path(
    "data/raw/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
)

OUTPUT_DIR = Path("data/packet_research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ONSETS = [
    "03:54:50",
    "10:13:20",
    "12:13:30",
]

DATE = "2018-02-03"

BEFORE_MINUTES = 10
AFTER_MINUTES = 2


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("BOT ONSET WINDOW INSPECTION")
print("=" * 70)

print(f"Loading: {CSV_PATH}")

df = pd.read_csv(
    CSV_PATH,
    usecols=[
        "Timestamp",
        "Label",
    ],
)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce",
)

df["Label"] = (
    df["Label"]
    .astype(str)
    .str.strip()
)

df = df.dropna(subset=["Timestamp"])

print(f"Total rows: {len(df):,}")
print(f"Time range: {df['Timestamp'].min()} → {df['Timestamp'].max()}")

print("\nLabels:")
print(df["Label"].value_counts())


# ============================================================
# ANALYZE EACH ONSET
# ============================================================

for onset_time in ONSETS:

    onset = pd.Timestamp(f"{DATE} {onset_time}")

    start = onset - pd.Timedelta(
        minutes=BEFORE_MINUTES
    )

    end = onset + pd.Timedelta(
        minutes=AFTER_MINUTES
    )

    window = df[
        (df["Timestamp"] >= start)
        & (df["Timestamp"] <= end)
    ].copy()

    print("\n" + "=" * 70)
    print(f"ONSET: {onset}")
    print("=" * 70)

    print(f"Window start : {start}")
    print(f"Onset        : {onset}")
    print(f"Window end   : {end}")

    print(f"\nRows: {len(window):,}")

    print("\nLabels:")
    print(
        window["Label"]
        .value_counts()
        .to_string()
    )

    # Counts per minute
    counts = (
        window
        .set_index("Timestamp")
        .groupby(
            [
                pd.Grouper(freq="1min"),
                "Label",
            ]
        )
        .size()
        .unstack(fill_value=0)
    )

    print("\nPer-minute traffic:")
    print(counts.to_string())

    # Save
    safe_name = onset_time.replace(":", "")

    out_file = (
        OUTPUT_DIR
        / f"bot_onset_{safe_name}_flows.csv"
    )

    window.to_csv(
        out_file,
        index=False,
    )

    print(f"\nSaved: {out_file}")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)