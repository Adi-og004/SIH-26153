import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FILE_PATH = Path(
    "data/raw/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"
)

print("Loading dataset...")
df = pd.read_csv(FILE_PATH)

# Convert timestamp
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

# Sort chronologically
df = df.sort_values("Timestamp").reset_index(drop=True)

# Create 10-second windows
df["time_window"] = df["Timestamp"].dt.floor("10s")

# Mark whether each flow is an attack
df["is_attack"] = (
    df["Label"].str.strip().str.lower() != "benign"
)

# ---------------------------------------------------------
# 1. Total flows per window
# ---------------------------------------------------------
total_flows = df.groupby("time_window").size()

# ---------------------------------------------------------
# 2. Attack flows per window
# ---------------------------------------------------------
attack_flows = df.groupby("time_window")["is_attack"].sum()

# ---------------------------------------------------------
# 3. Attack percentage per window
# ---------------------------------------------------------
attack_percentage = (
    attack_flows / total_flows * 100
).fillna(0)

# ---------------------------------------------------------
# 4. Label counts per window
# ---------------------------------------------------------
label_counts = (
    df.groupby(["time_window", "Label"])
      .size()
      .unstack(fill_value=0)
)

print("\n========== ATTACK TIMELINE SUMMARY ==========")

attack_windows = attack_percentage[attack_percentage > 0]

print("Total windows:", len(total_flows))
print("Windows containing attacks:", len(attack_windows))

if len(attack_windows) > 0:
    print("First attack window:", attack_windows.index.min())
    print("Last attack window:", attack_windows.index.max())
    print(
        "Maximum attack percentage in a window:",
        round(attack_percentage.max(), 2),
        "%"
    )

# ---------------------------------------------------------
# 5. Plot total traffic
# ---------------------------------------------------------
plt.figure(figsize=(16, 6))

plt.plot(
    total_flows.index,
    total_flows.values
)

plt.title("Network Traffic Volume Over Time")
plt.xlabel("Time")
plt.ylabel("Number of Flows")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "data/traffic_volume_timeline.png",
    dpi=150
)

plt.show()

# ---------------------------------------------------------
# 6. Plot attack percentage
# ---------------------------------------------------------
plt.figure(figsize=(16, 6))

plt.plot(
    attack_percentage.index,
    attack_percentage.values
)

plt.title("Attack Traffic Percentage Over Time")
plt.xlabel("Time")
plt.ylabel("Attack Traffic (%)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "data/attack_percentage_timeline.png",
    dpi=150
)

plt.show()

# ---------------------------------------------------------
# 7. Plot individual attack types
# ---------------------------------------------------------
plt.figure(figsize=(16, 6))

for label in label_counts.columns:

    if label.lower().strip() != "benign":

        plt.plot(
            label_counts.index,
            label_counts[label],
            label=label
        )

plt.title("Attack Types Over Time")
plt.xlabel("Time")
plt.ylabel("Number of Flows")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "data/attack_types_timeline.png",
    dpi=150
)

plt.show()

print("\nPlots saved to:")
print("data/traffic_volume_timeline.png")
print("data/attack_percentage_timeline.png")
print("data/attack_types_timeline.png")