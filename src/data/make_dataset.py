"""Data Preparation and Reproducible Splitting Pipeline for ChurnGuard.

Handles:
1. Loading raw dataset updated customerchurn.csv
2. Schema & data quality verification
3. Timestamp format verification (format: DD-MM-YYYY HH:MM)
4. Duplicate records check
5. Reproducible Stratified train/val/test splitting (70% / 15% / 15%, seed=42)
6. Zero customer overlap verification
7. Saving clean datasets to data/processed/ (train.csv, validation.csv, test.csv)
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def get_project_root() -> Path:
    """Resolves the ChurnGuard project root directory."""
    current = Path(__file__).resolve().parent
    # If in ChurnGuard/src/data
    if current.name == "data" and current.parent.name == "src":
        return current.parent.parent
    if (current / "ChurnGuard").exists():
        return current / "ChurnGuard"
    return current


def run_make_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = get_project_root()
    data_path = root / "data" / "updated customerchurn.csv"
    if not data_path.exists():
        data_path = root / "data" / "raw" / "updated customerchurn.csv"

    output_dir = root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_col = "churned_90d"
    id_col = "customer_id"
    timestamp_col = "feature_timestamp"
    random_state = 42

    print("\n" + "=" * 60)
    print(" CHURNGUARD: REPRODUCIBLE DATASET PREPARATION PIPELINE")
    print("=" * 60)
    print(f"Loading raw dataset from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    total_records = len(df)
    print(f"Raw records loaded: {total_records}, Columns: {len(df.columns)}")

    # ==========================================
    # 1. TIMESTAMP PARSING & VALIDATION
    # ==========================================
    # Strictly parse using DD-MM-YYYY HH:MM format
    parsed_timestamps = pd.to_datetime(
        df[timestamp_col].astype(str).str.strip(),
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )
    invalid_ts = int(parsed_timestamps.isna().sum())
    print(f"Feature timestamps parsed: {total_records - invalid_ts} valid, {invalid_ts} invalid")

    if invalid_ts > 0:
        raise ValueError(f"Detected {invalid_ts} invalid timestamps in {timestamp_col}!")

    timestamp_report = pd.DataFrame({
        "metric": [
            "Total timestamp records",
            "Invalid timestamps",
            "Earliest feature timestamp",
            "Latest feature timestamp"
        ],
        "value": [
            total_records,
            invalid_ts,
            str(parsed_timestamps.min()),
            str(parsed_timestamps.max())
        ]
    })
    timestamp_report.to_csv(output_dir / "timestamp_validation_report.csv", index=False)
    print("Timestamp validation report saved.")

    # ==========================================
    # 2. DUPLICATES & TARGET INTEGRITY
    # ==========================================
    dupes = int(df.duplicated().sum())
    print(f"Duplicate rows detected: {dupes}")
    if dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print("Duplicate rows removed.")

    target_vals = set(df[target_col].unique())
    if target_vals != {0, 1}:
        raise ValueError(f"Target column must contain classes {{0, 1}}, got: {target_vals}")

    # ==========================================
    # 3. REPRODUCIBLE STRATIFIED SPLITTING
    # ==========================================
    # 70% Train, 15% Validation, 15% Test
    print("Splitting dataset: 70% Train, 15% Validation, 15% Test (seed=42)...")
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=df[target_col]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df[target_col]
    )

    # ==========================================
    # 4. CUSTOMER OVERLAP VERIFICATION
    # ==========================================
    train_ids = set(train_df[id_col])
    val_ids = set(val_df[id_col])
    test_ids = set(test_df[id_col])

    overlap_tv = len(train_ids & val_ids)
    overlap_tt = len(train_ids & test_ids)
    overlap_vt = len(val_ids & test_ids)

    if (overlap_tv + overlap_tt + overlap_vt) > 0:
        raise RuntimeError(
            f"Customer leakage detected! Overlaps: Train/Val={overlap_tv}, "
            f"Train/Test={overlap_tt}, Val/Test={overlap_vt}"
        )

    print("Zero customer overlap verified: PASSED (Train, Val, Test partitions are mutually exclusive)")

    # ==========================================
    # 5. SAVE PARTITIONS
    # ==========================================
    train_path = output_dir / "train.csv"
    val_path = output_dir / "validation.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved processed partitions:")
    print(f" - Train:      {len(train_df):,} records ({len(train_df)/total_records*100:.1f}%) -> {train_path.name}")
    print(f" - Validation: {len(val_df):,} records ({len(val_df)/total_records*100:.1f}%) -> {val_path.name}")
    print(f" - Test:       {len(test_df):,} records ({len(test_df)/total_records*100:.1f}%) -> {test_path.name}")

    print("\nTarget Churn Distribution:")
    print(f" - Overall:    {df[target_col].mean()*100:.2f}%")
    print(f" - Train:      {train_df[target_col].mean()*100:.2f}%")
    print(f" - Validation: {val_df[target_col].mean()*100:.2f}%")
    print(f" - Test:       {test_df[target_col].mean()*100:.2f}%")

    print("\nCHURNGUARD DATASET PREPARATION COMPLETED SUCCESSFULLY!\n")
    return train_df, val_df, test_df


if __name__ == "__main__":
    run_make_dataset()
