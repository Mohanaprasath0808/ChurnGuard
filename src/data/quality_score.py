
from pathlib import Path
import pandas as pd
import json

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_DIR / "data" / "customer_churn.csv"
OUTPUT_DIR = PROJECT_DIR / "reports" / "quality"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

# Quality checks
total_cells = df.size
missing_cells = int(df.isnull().sum().sum())
duplicate_rows = int(df.duplicated().sum())

missing_score = (
    (total_cells - missing_cells) / total_cells * 100
)

duplicate_score = (
    (len(df) - duplicate_rows) / len(df) * 100
)

# Numeric validity checks
numeric_columns = [
    "tenure_months",
    "monthly_charges",
    "support_tickets_90d",
    "avg_monthly_usage_hours",
    "late_payments_90d"
]

negative_values = 0

for col in numeric_columns:
    if col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce")
        negative_values += int((values < 0).sum())

numeric_cells = len(df) * len(
    [c for c in numeric_columns if c in df.columns]
)

numeric_score = (
    (numeric_cells - negative_values) / numeric_cells * 100
    if numeric_cells else 100
)

# Target validity
target = pd.to_numeric(df["churned_90d"], errors="coerce")
invalid_target = int(
    (target.isna() | ~target.isin([0, 1])).sum()
)

target_score = (
    (len(df) - invalid_target) / len(df) * 100
)

# Overall score (equal-weighted checks)
overall_score = (
    missing_score +
    duplicate_score +
    numeric_score +
    target_score
) / 4

summary = {
    "dataset": DATA_FILE.name,
    "total_records": int(len(df)),
    "total_columns": int(len(df.columns)),
    "missing_values": missing_cells,
    "duplicate_rows": duplicate_rows,
    "negative_numeric_values": negative_values,
    "invalid_target_values": invalid_target,
    "missing_value_score": round(missing_score, 2),
    "duplicate_score": round(duplicate_score, 2),
    "numeric_validity_score": round(numeric_score, 2),
    "target_validity_score": round(target_score, 2),
    "overall_quality_score": round(overall_score, 2),
    "status": "REVIEW REQUIRED"
}

with open(OUTPUT_DIR / "quality_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\n========== CHURNGUARD DATA QUALITY ==========")

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nQUALITY SCORE REPORT CREATED SUCCESSFULLY!")
print("Saved at:", OUTPUT_DIR)
