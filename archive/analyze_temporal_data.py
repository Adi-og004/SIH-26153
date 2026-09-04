import pandas as pd
from pathlib import Path

FILE_PATH = Path(
    "data/raw/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"
)

print("Loading dataset...")
df = pd.read_csv(FILE_PATH)

# ---------------------------------------------------------
# 1. Convert timestamp to datetime
# ---------------------------------------------------------
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

print("\n========== TIMESTAMP CHECK ==========")
print("Invalid timestamps:", df["Timestamp"].isna().sum())
print("Start:", df["Timestamp"].min())
print("End:", df["Timestamp"].max())

# ---------------------------------------------------------
# 2. Sort chronologically
# ---------------------------------------------------------
df = df.sort_values("Timestamp").reset_index(drop=True)

print("\n========== CHRONOLOGICAL CHECK ==========")
print(df[["Timestamp", "Dst Port", "Protocol", "Label"]].head(10))

# ---------------------------------------------------------
# 3. Label distribution
# ---------------------------------------------------------
print("\n========== LABEL DISTRIBUTION ==========")
label_counts = df["Label"].value_counts(dropna=False)

print(label_counts)

print("\n========== LABEL PERCENTAGE ==========")
print(
    (df["Label"].value_counts(normalize=True) * 100)
    .round(3)
)

# ---------------------------------------------------------
# 4. Time range
# ---------------------------------------------------------
print("\n========== TIME RANGE ==========")
print("First flow :", df["Timestamp"].iloc[0])
print("Last flow  :", df["Timestamp"].iloc[-1])

duration = (
    df["Timestamp"].iloc[-1] -
    df["Timestamp"].iloc[0]
)

print("Total duration:", duration)

# ---------------------------------------------------------
# 5. Number of flows per 10-second window
# ---------------------------------------------------------
df["time_window"] = df["Timestamp"].dt.floor("10s")

window_counts = df.groupby("time_window").size()

print("\n========== 10-SECOND WINDOW STATS ==========")
print("Number of windows:", len(window_counts))
print("Minimum flows/window:", window_counts.min())
print("Maximum flows/window:", window_counts.max())
print("Mean flows/window:", window_counts.mean())
print("Median flows/window:", window_counts.median())

# ---------------------------------------------------------
# 6. Number of active windows containing attacks
# ---------------------------------------------------------
attack_mask = df["Label"].str.lower() != "benign"

attack_windows = (
    df.loc[attack_mask]
      .groupby("time_window")
      .size()
)

print("\n========== ATTACK WINDOW STATS ==========")
print("Attack windows:", len(attack_windows))
print("Attack flows:", attack_mask.sum())

# ---------------------------------------------------------
# 7. Missing values
# ---------------------------------------------------------
print("\n========== MISSING VALUES ==========")

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print(missing)

# ---------------------------------------------------------
# 8. Infinite values
# ---------------------------------------------------------
numeric_columns = df.select_dtypes(include="number").columns

infinite_counts = {}

for col in numeric_columns:
    count = (~df[col].replace([float("inf"), float("-inf")], pd.NA).notna()).sum()
    infinite_counts[col] = count

infinite_counts = {
    k: v for k, v in infinite_counts.items() if v > 0
}

print("\n========== INFINITE VALUE CHECK ==========")
print(infinite_counts)

print("\nAnalysis complete.")