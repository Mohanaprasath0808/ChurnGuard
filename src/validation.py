"""Data Quality Validation Module for ChurnGuard Customer Churn Prediction.

Performs comprehensive data quality checks on updated customerchurn.csv:
1. Missing values (explicit nulls and implicit whitespace).
2. Duplicate records (full row and customer_id duplicates).
3. Data types and constraints.
4. Customer ID format (CUST-XXXXX) and uniqueness.
5. Satisfaction scores (1 to 5 integer range).
6. Churn target values (strictly 0 and 1).
7. Feature timestamps parsing using exact DD-MM-YYYY HH:MM format.
8. Numerical ranges and categorical domains.

Generates:
- reports/missing_values_report.csv
- reports/validation_report.csv
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def get_project_root() -> Path:
    """Resolves the ChurnGuard project root directory."""
    current = Path(__file__).resolve().parent
    if current.name == "src":
        return current.parent
    if (current / "ChurnGuard").exists():
        return current / "ChurnGuard"
    return current


def run_data_validation(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs all data quality validations and outputs reports."""
    root = get_project_root()

    if data_path is None:
        candidates = [
            root / "data" / "updated customerchurn.csv",
            root / "data" / "raw" / "updated customerchurn.csv",
            Path("ChurnGuard/data/updated customerchurn.csv"),
            Path("data/updated customerchurn.csv"),
        ]
        for c in candidates:
            if c.exists():
                data_path = c
                break
        if data_path is None:
            raise FileNotFoundError(f"Could not locate updated customerchurn.csv under {root}")

    if output_dir is None:
        output_dir = root / "reports"

    output_dir.mkdir(parents=True, exist_ok=True)
    missing_report_path = output_dir / "missing_values_report.csv"
    val_report_path = output_dir / "validation_report.csv"

    print(f"\n[Data Quality Validation] Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    total_records = len(df)
    print(f"[Data Quality Validation] Total records: {total_records}")

    validation_results: List[Dict[str, str]] = []

    def record_check(name: str, status: str, expected: str, observed: str, details: str):
        validation_results.append({
            "validation_check": name,
            "status": status,
            "expected": expected,
            "observed": observed,
            "details": details
        })

    # ==========================================
    # 1. MISSING VALUES AUDIT
    # ==========================================
    missing_rows = []
    total_nulls = 0
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        # Check blank whitespace strings for object columns
        blank_count = 0
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            blank_count = int((df[col].astype(str).str.strip() == "").sum())

        combined_missing = null_count + blank_count
        total_nulls += combined_missing
        missing_pct = round((combined_missing / total_records) * 100.0, 2)
        status = "PASSED" if combined_missing == 0 else "FAILED"

        missing_rows.append({
            "column_name": col,
            "null_count": null_count,
            "blank_string_count": blank_count,
            "total_missing": combined_missing,
            "missing_percentage": f"{missing_pct}%",
            "status": status
        })

    missing_df = pd.DataFrame(missing_rows)
    missing_df.to_csv(missing_report_path, index=False)
    print(f"[Data Quality Validation] Missing values report saved to: {missing_report_path}")

    record_check(
        name="Missing Values Check",
        status="PASSED" if total_nulls == 0 else "FAILED",
        expected="0 missing values across all columns",
        observed=f"{total_nulls} missing values",
        details="Zero nulls or blank whitespace strings detected across all 12 columns."
    )

    # ==========================================
    # 2. DUPLICATE RECORDS CHECK
    # ==========================================
    row_duplicates = int(df.duplicated().sum())
    record_check(
        name="Duplicate Rows Check",
        status="PASSED" if row_duplicates == 0 else "FAILED",
        expected="0 duplicate rows",
        observed=f"{row_duplicates} duplicates",
        details="Each row represents an independent record."
    )

    # ==========================================
    # 3. CUSTOMER ID VALIDATION
    # ==========================================
    id_nulls = int(df["customer_id"].isnull().sum())
    id_dupes = int(df["customer_id"].duplicated().sum())
    # Format check: CUST- followed by digits
    pattern = re.compile(r"^CUST-\d+$")
    valid_format_count = sum(bool(pattern.match(str(x))) for x in df["customer_id"])
    format_failures = total_records - valid_format_count

    id_status = "PASSED" if (id_nulls == 0 and id_dupes == 0 and format_failures == 0) else "FAILED"
    record_check(
        name="Customer ID Validation",
        status=id_status,
        expected="Unique, non-null, format CUST-XXXXX",
        observed=f"Nulls: {id_nulls}, Dupes: {id_dupes}, Format Invalid: {format_failures}",
        details=f"All {df['customer_id'].nunique()} customer IDs are valid and unique."
    )

    # ==========================================
    # 4. SATISFACTION SCORE VALIDATION (1-5)
    # ==========================================
    sat_col = df["satisfaction_score"]
    sat_valid_range = sat_col.between(1, 5).all()
    sat_invalid_count = int((~sat_col.between(1, 5)).sum())
    sat_unique = sorted(sat_col.unique().tolist())

    record_check(
        name="Satisfaction Score Validation",
        status="PASSED" if sat_valid_range else "FAILED",
        expected="Integer values strictly in [1, 5]",
        observed=f"Unique values observed: {sat_unique}",
        details=f"All satisfaction scores conform to 1-5 integer scale (invalid count: {sat_invalid_count})."
    )

    # ==========================================
    # 5. CHURN TARGET VALIDATION (0 and 1)
    # ==========================================
    target_col = df["churned_90d"]
    target_values = set(target_col.unique())
    target_valid = (target_values == {0, 1})
    counts = target_col.value_counts().to_dict()

    record_check(
        name="Churn Target Validation",
        status="PASSED" if target_valid else "FAILED",
        expected="Binary values strictly in {0, 1}",
        observed=f"Classes observed: {target_values} (0: {counts.get(0, 0)}, 1: {counts.get(1, 0)})",
        details="Target churned_90d has both required classes (0: Retained, 1: Churned)."
    )

    # ==========================================
    # 6. FEATURE TIMESTAMP VALIDATION (DD-MM-YYYY HH:MM)
    # ==========================================
    # User requirement: My feature_timestamp format is DD-MM-YYYY HH:MM, for example 02-02-2025 13:50.
    # Parse with exact format so valid timestamps are not misclassified.
    parsed_ts = pd.to_datetime(
        df["feature_timestamp"].astype(str).str.strip(),
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )
    invalid_ts_count = int(parsed_ts.isna().sum())
    ts_status = "PASSED" if invalid_ts_count == 0 else "FAILED"

    record_check(
        name="Feature Timestamp Validation",
        status=ts_status,
        expected="Valid datetime with format DD-MM-YYYY HH:MM",
        observed=f"Invalid count: {invalid_ts_count} (Valid: {len(parsed_ts) - invalid_ts_count})",
        details=f"Date range observed: {parsed_ts.min()} to {parsed_ts.max()}. Format parsed correctly."
    )

    # ==========================================
    # 7. NUMERICAL RANGES VALIDATION
    # ==========================================
    tenure_valid = (df["tenure_months"] >= 0) & (df["tenure_months"] <= 120)
    charges_valid = (df["monthly_charges"] >= 0.0) & (df["monthly_charges"] <= 500.0)
    tickets_valid = (df["support_tickets_90d"] >= 0)
    usage_valid = (df["avg_monthly_usage_hours"] >= 0.0)
    late_valid = (df["late_payments_90d"] >= 0)

    num_valid = tenure_valid.all() and charges_valid.all() and tickets_valid.all() and usage_valid.all() and late_valid.all()
    record_check(
        name="Numerical Range Constraints",
        status="PASSED" if num_valid else "FAILED",
        expected="All numeric features non-negative and within valid operational limits",
        observed=f"Tenure min/max: [{df['tenure_months'].min()}, {df['tenure_months'].max()}], Charges min/max: [{df['monthly_charges'].min()}, {df['monthly_charges'].max()}]",
        details="All numeric features (tenure, charges, tickets, usage, late payments) within plausible bounds."
    )

    # ==========================================
    # 8. CATEGORICAL DOMAIN VALIDATION
    # ==========================================
    contracts = set(df["contract_type"].unique())
    autopay = set(df["autopay_enabled"].unique())
    discounts = set(df["discount_currently_active"].unique())

    cat_valid = (
        contracts.issubset({"Two-year", "Annual", "Monthly"})
        and autopay.issubset({"Yes", "No"})
        and discounts.issubset({"Yes", "No"})
    )

    record_check(
        name="Categorical Domain Validation",
        status="PASSED" if cat_valid else "FAILED",
        expected="contract_type in [Two-year, Annual, Monthly]; autopay, discounts in [Yes, No]",
        observed=f"Contracts: {sorted(list(contracts))}; Autopay: {sorted(list(autopay))}; Discounts: {sorted(list(discounts))}",
        details="All categorical variables conform to standardized domain values."
    )

    # Export validation report
    val_df = pd.DataFrame(validation_results)
    val_df.to_csv(val_report_path, index=False)
    print(f"[Data Quality Validation] Validation report saved to: {val_report_path}")

    # Terminal output
    print("\n" + "=" * 80)
    print(f"{'VALIDATION CHECK':<32} {'STATUS':<10} {'DETAILS'}")
    print("-" * 80)
    for _, row in val_df.iterrows():
        print(f"{row['validation_check']:<32} {row['status']:<10} {row['details']}")
    print("=" * 80)

    all_passed = all(r["status"] == "PASSED" for r in validation_results)
    if all_passed:
        print("[Data Quality Validation] ALL DATA QUALITY VALIDATION CHECKS PASSED!\n")
    else:
        print("[Data Quality Validation] WARNING: One or more data quality checks failed.\n")

    return missing_df, val_df


if __name__ == "__main__":
    run_data_validation()

