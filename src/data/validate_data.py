
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_DIR / "data" / "customer_churn.csv"
OUTPUT_DIR = PROJECT_DIR / "reports" / "validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

results = []

def add_result(check, status, details):
    results.append({
        "validation_check": check,
        "status": status,
        "details": details
    })

# 1. Missing values
missing = int(df.isnull().sum().sum())
add_result(
    "Missing values",
    "PASS" if missing == 0 else "WARNING",
    f"Total missing values: {missing}"
)

# 2. Duplicate rows
duplicates = int(df.duplicated().sum())
add_result(
    "Duplicate rows",
    "PASS" if duplicates == 0 else "WARNING",
    f"Duplicate rows: {duplicates}"
)

# 3. Customer ID uniqueness
if "customer_id" in df.columns:
    duplicate_ids = int(df["customer_id"].duplicated().sum())
    add_result(
        "Customer ID uniqueness",
        "PASS" if duplicate_ids == 0 else "FAIL",
        f"Duplicate customer IDs: {duplicate_ids}"
    )

# 4. Numeric columns must not contain negative values
numeric_columns = [
    "tenure_months",
    "monthly_charges",
    "support_tickets_90d",
    "avg_monthly_usage_hours",
    "late_payments_90d"
]

for col in numeric_columns:
    if col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce")
        invalid = int((values < 0).sum())

        add_result(
            f"{col} non-negative",
            "PASS" if invalid == 0 else "FAIL",
            f"Negative values: {invalid}"
        )

# 5. Target validation
if "churned_90d" in df.columns:
    target = pd.to_numeric(df["churned_90d"], errors="coerce")
    invalid_target = int(
        (target.isna() | ~target.isin([0, 1])).sum()
    )

    add_result(
        "Target values",
        "PASS" if invalid_target == 0 else "FAIL",
        f"Invalid target values: {invalid_target}"
    )

# 6. Contract type validation
if "contract_type" in df.columns:
    allowed = ["Monthly", "Annual", "Two-year"]
    invalid_contracts = int(
        (~df["contract_type"].isin(allowed)).sum()
    )

    add_result(
        "Contract type",
        "PASS" if invalid_contracts == 0 else "FAIL",
        f"Invalid contract values: {invalid_contracts}"
    )

# 7. Binary columns
for col in ["autopay_enabled", "discount_currently_active"]:
    if col in df.columns:
        values = df[col].dropna().astype(str).str.strip().str.lower()
        allowed = {"true", "false", "1", "0"}

        invalid = int((~values.isin(allowed)).sum())

        add_result(
            f"{col} binary values",
            "PASS" if invalid == 0 else "FAIL",
            f"Invalid values: {invalid}"
        )

# Save report
report = pd.DataFrame(results)

report.to_csv(
    OUTPUT_DIR / "validation_report.csv",
    index=False
)

print("\n========== DATA VALIDATION REPORT ==========")
print(report.to_string(index=False))

print("\nDATA VALIDATION COMPLETED!")
print("Report saved at:", OUTPUT_DIR)
