import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data/raw")
CLEAN_DIR = Path("data/processed")

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
]


def clean_dataset(filename: str):

    input_path = DATA_DIR / filename
    output_path = CLEAN_DIR / filename

    print("\n" + "=" * 90)
    print(f"PROCESSING: {filename}")
    print("=" * 90)

    # -----------------------------------------------------
    # Load safely
    # -----------------------------------------------------
    print("Loading CSV...")

    df = pd.read_csv(
        input_path,
        low_memory=False
    )

    original_rows = len(df)

    print(f"Original rows: {original_rows:,}")

    # -----------------------------------------------------
    # Strip whitespace from column names
    # -----------------------------------------------------
    df.columns = df.columns.str.strip()

    # -----------------------------------------------------
    # Clean string fields
    # -----------------------------------------------------
    df["Label"] = df["Label"].astype(str).str.strip()

    # -----------------------------------------------------
    # Timestamp conversion
    # -----------------------------------------------------
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    invalid_timestamp = df["Timestamp"].isna().sum()

    # Detect timestamps outside the expected dataset era
    old_timestamp_mask = df["Timestamp"].notna() & (
        df["Timestamp"].dt.year < 2018
    )

    old_timestamp_count = old_timestamp_mask.sum()

    # -----------------------------------------------------
    # Identify malformed label rows
    # -----------------------------------------------------
    malformed_label_mask = (
        df["Label"].str.lower().isin([
            "label",
            "nan",
            "none"
        ])
    )

    malformed_label_count = malformed_label_mask.sum()

    # -----------------------------------------------------
    # Convert numeric columns
    # -----------------------------------------------------
    numeric_columns = [
        col for col in df.columns
        if col not in ["Timestamp", "Label"]
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # -----------------------------------------------------
    # Replace infinite values
    # -----------------------------------------------------
    numeric_df = df[numeric_columns]

    infinite_mask = np.isinf(numeric_df.to_numpy())

    infinite_values = int(infinite_mask.sum())

    print("\n--- DATA QUALITY REPORT ---")

    print(f"Invalid timestamps:        {invalid_timestamp:,}")
    print(f"Old timestamps (<2018):    {old_timestamp_count:,}")
    print(f"Malformed label rows:      {malformed_label_count:,}")
    print(f"Infinite numeric values:   {infinite_values:,}")

    # -----------------------------------------------------
    # Show suspicious timestamp rows
    # -----------------------------------------------------
    if old_timestamp_count > 0:

        print("\n--- SUSPICIOUS OLD TIMESTAMPS ---")

        suspicious = df.loc[
            old_timestamp_mask,
            ["Timestamp", "Dst Port", "Protocol", "Label"]
        ]

        print(suspicious.head(20).to_string())

    # -----------------------------------------------------
    # Show malformed rows
    # -----------------------------------------------------
    if malformed_label_count > 0:

        print("\n--- MALFORMED LABEL ROWS ---")

        malformed = df.loc[
            malformed_label_mask,
            ["Timestamp", "Dst Port", "Protocol", "Label"]
        ]

        print(malformed.head(20).to_string())

    # -----------------------------------------------------
    # Remove malformed / invalid records
    # -----------------------------------------------------
    removal_mask = (
        df["Timestamp"].isna()
        | old_timestamp_mask
        | malformed_label_mask
    )

    removed_rows = removal_mask.sum()

    df = df.loc[~removal_mask].copy()

    # -----------------------------------------------------
    # Replace infinity with NaN
    # -----------------------------------------------------
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # -----------------------------------------------------
    # Sort chronologically
    # -----------------------------------------------------
    df.sort_values(
        "Timestamp",
        inplace=True
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    # -----------------------------------------------------
    # Standardize known label spelling
    # -----------------------------------------------------
    df["Label"] = df["Label"].replace({
        "Infilteration": "Infiltration"
    })

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------
    print("\n--- FINAL RESULT ---")

    print(f"Rows removed:             {removed_rows:,}")
    print(f"Clean rows:                {len(df):,}")
    print(f"Rows retained:             {len(df)/original_rows*100:.3f}%")

    print("\nFinal timestamp range:")
    print("Start:", df["Timestamp"].min())
    print("End:  ", df["Timestamp"].max())

    print("\nFinal labels:")
    print(df["Label"].value_counts().to_string())

    print("\nMissing numerical values:")
    missing = df[numeric_columns].isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) > 0:
        print(missing.to_string())
    else:
        print("None")

    # -----------------------------------------------------
    # Save cleaned dataset
    # -----------------------------------------------------
    df.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved cleaned dataset:")
    print(output_path)


def main():

    print("CIC-IDS2018 CLEANING PIPELINE")

    for filename in FILES:
        clean_dataset(filename)

    print("\n" + "=" * 90)
    print("ALL DATASETS CLEANED")
    print("=" * 90)


if __name__ == "__main__":
    main()