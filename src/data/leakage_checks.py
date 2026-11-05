
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "reports" / "leakage"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "churned_90d"
ID_COLUMN = "customer_id"

df = pd.read_csv(DATA_DIR / "train.csv")

if TARGET not in df.columns:
    raise ValueError(f"Target column {TARGET} not found!")

results = []

def add_check(check, status, details):
    results.append({
        "Check": check,
        "Status": status,
        "Details": details
    })

print("\n========== CHURNGUARD LEAKAGE CHECKS ==========")

# 1. Target validation
if df[TARGET].notna().all():
    add_check(
        "Target missing values",
        "PASS",
        "No missing target values found."
    )
else:
    add_check(
        "Target missing values",
        "WARNING",
        f"{df[TARGET].isna().sum()} missing target values found."
    )

# 2. Target class validation
if set(df[TARGET].dropna().unique()).issubset({0, 1}):
    add_check(
        "Target values",
        "PASS",
        "Target contains only binary values 0 and 1."
    )
else:
    add_check(
        "Target values",
        "WARNING",
        "Unexpected target values found."
    )

# 3. Duplicate records
duplicate_count = df.duplicated().sum()

add_check(
    "Duplicate rows",
    "PASS" if duplicate_count == 0 else "WARNING",
    f"Duplicate rows: {duplicate_count}"
)

# 4. Duplicate customer IDs
if ID_COLUMN in df.columns:
    duplicate_ids = df[ID_COLUMN].duplicated().sum()

    add_check(
        "Duplicate customer IDs",
        "PASS" if duplicate_ids == 0 else "WARNING",
        f"Duplicate IDs: {duplicate_ids}"
    )

# 5. Suspicious feature names
suspicious_words = [
    "churn",
    "cancelled",
    "cancellation",
    "termination",
    "retention_outcome"
]

suspicious_columns = [
    col for col in df.columns
    if col != TARGET
    and any(word in col.lower() for word in suspicious_words)
]

add_check(
    "Suspicious feature names",
    "PASS" if not suspicious_columns else "REVIEW REQUIRED",
    (
        "No suspicious feature names found."
        if not suspicious_columns
        else f"Review these columns: {suspicious_columns}"
    )
)

# 6. Feature timestamp verification
if "feature_timestamp" in df.columns:
    add_check(
        "Feature timing verification",
        "MANUAL REVIEW REQUIRED",
        "Timestamp exists. Verify that every feature was recorded "
        "before the prediction cutoff."
    )
else:
    add_check(
        "Feature timing verification",
        "PENDING",
        "feature_timestamp is unavailable. Historical feature timing "
        "cannot be verified from this sample dataset."
    )

# 7. High correlation review
numeric_df = df.select_dtypes(include="number")

if TARGET in numeric_df.columns:
    correlations = numeric_df.corr()[TARGET].drop(TARGET)
    high_correlations = correlations[
        correlations.abs() >= 0.95
    ]

    if len(high_correlations) > 0:
        add_check(
            "High target correlation",
            "REVIEW REQUIRED",
            f"Review these features: {high_correlations.to_dict()}"
        )
    else:
        add_check(
            "High target correlation",
            "PASS",
            "No numeric feature has absolute correlation >= 0.95 "
            "with the target."
        )

# 8. Customer ID exclusion reminder
add_check(
    "Customer ID feature usage",
    "MANUAL REVIEW REQUIRED",
    "Ensure customer_id is excluded from model training features."
)

# Save reports
report = pd.DataFrame(results)

report.to_csv(
    REPORT_DIR / "leakage_check_report.csv",
    index=False
)

print("\n--- Leakage Check Results ---")
print(report.to_string(index=False))

print("\nReport saved at:", REPORT_DIR)
print("\nLEAKAGE CHECKS COMPLETED!")
