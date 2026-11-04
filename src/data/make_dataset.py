from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "customer_churn.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

TARGET = "churned_90d"

def main():

    print("ChurnGuard Dataset Preparation Started")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print("\nDataset Shape:", df.shape)

    # Schema validation
    expected_columns = [
        "customer_id",
        "tenure_months",
        "monthly_charges",
        "support_tickets_90d",
        "avg_monthly_usage_hours",
        "late_payments_90d",
        "contract_type",
        "autopay_enabled",
        "satisfaction_score",
        "discount_currently_active",
        "churned_90d"
    ]

    missing = set(expected_columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("Schema validation: PASSED")

    # Data quality
    print("\nMissing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Duplicate IDs:", df["customer_id"].duplicated().sum())

    if df.isnull().sum().sum() > 0:
        raise ValueError("Missing values found")

    if df.duplicated().sum() > 0:
        raise ValueError("Duplicate rows found")

    if df["customer_id"].duplicated().sum() > 0:
        raise ValueError("Duplicate customer IDs found")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1")

    print("Data quality validation: PASSED")

    # Features and target
    X = df.drop(columns=[TARGET, "customer_id"])
    y = df[TARGET]

    # Train and temporary split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Validation and test split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = X_train.copy()
    train[TARGET] = y_train

    validation = X_val.copy()
    validation[TARGET] = y_val

    test = X_test.copy()
    test[TARGET] = y_test

    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\n--- Dataset Split Results ---")
    print("Train:", len(train))
    print("Validation:", len(validation))
    print("Test:", len(test))

    print("\nChurnGuard dataset preparation completed successfully!")


if __name__ == "__main__":
    main()