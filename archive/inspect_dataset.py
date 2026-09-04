import pandas as pd
from pathlib import Path

file_path = Path(
    "data/raw/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"
)

print("Loading dataset...")

df = pd.read_csv(file_path)

print("\n========== DATASET OVERVIEW ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False).head(20))

print("\n========== LABEL COLUMNS ==========")
for column in df.columns:
    if any(word in column.lower() for word in ["label", "attack", "class"]):
        print(column)

print("\nDone.")