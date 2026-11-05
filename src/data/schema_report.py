
from pathlib import Path
import pandas as pd
import json

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_DIR / "data" / "customer_churn.csv"
OUTPUT_DIR = PROJECT_DIR / "reports" / "schema"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

EXPECTED_COLUMNS = [
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

missing_columns = list(set(EXPECTED_COLUMNS) - set(df.columns))
unexpected_columns = list(set(df.columns) - set(EXPECTED_COLUMNS))

schema_report = {
    "dataset_name": DATA_FILE.name,
    "total_rows": int(df.shape[0]),
    "total_columns": int(df.shape[1]),
    "expected_columns": EXPECTED_COLUMNS,
    "actual_columns": df.columns.tolist(),
    "missing_columns": missing_columns,
    "unexpected_columns": unexpected_columns,
    "column_data_types": {
        col: str(dtype)
        for col, dtype in df.dtypes.items()
    },
    "schema_status": (
        "PASS"
        if not missing_columns and not unexpected_columns
        else "FAIL"
    )
}

with open(OUTPUT_DIR / "schema_report.json", "w") as file:
    json.dump(schema_report, file, indent=4)

print("\n========== CHURNGUARD SCHEMA REPORT ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Missing columns:", missing_columns)
print("Unexpected columns:", unexpected_columns)
print("Schema status:", schema_report["schema_status"])
print("\nSCHEMA REPORT CREATED SUCCESSFULLY!")
