"""Schema Validation Module for ChurnGuard Customer Churn Prediction.

Identifies all dataset columns, verifies expected vs observed data types,
checks required columns, nullability, and primary key status.
Generates reports/schema_report.csv.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


def get_project_root() -> Path:
    """Resolves the ChurnGuard project root directory."""
    current = Path(__file__).resolve().parent
    # If this file is in ChurnGuard/src, parent is ChurnGuard
    if current.name == "src":
        return current.parent
    # If run in top-level
    if (current / "ChurnGuard").exists():
        return current / "ChurnGuard"
    return current


# Schema Specification for updated customerchurn.csv
EXPECTED_SCHEMA: Dict[str, Dict[str, str]] = {
    "customer_id": {
        "expected_type": "string / object",
        "role": "Identifier",
        "is_required": "Yes",
        "is_nullable": "No",
        "description": "Unique customer account identifier (e.g., CUST-00001)"
    },
    "tenure_months": {
        "expected_type": "integer",
        "role": "Numeric Feature",
        "is_required": "Yes",
        "is_nullable": "No",
        "description": "Customer lifetime tenure with service in months"
    },
    "monthly_charges": {
        "expected_type": "float",
        "role": "Numeric Feature",
        "is_required": "Yes",
        "is_nullable": "No",
        "description": "Current recurring monthly charge amount in USD"
    },
    "support_tickets_90d": {
        "expected_type": "integer",
        "role": "Numeric Feature",
        "is_required": "Yes",
        "is_nullable": "No",
        "description": "Customer support requests submitted in trailing 90 days"
    },
    "avg_monthly_usage_hours": {
        "expected_type": "float",
        "role": "Numeric Feature",
        "is_required": "Yes",
        "is_nullable": "No",
        "description": "Average monthly service usage in active hours"
    },
    "late_payments_90d": {
        "expected_type": "integer",
        "role": "Numeric Feature",
        "is_required": "Yes",
        "is_nullable": "No",
        "description": "Count of delinquent or overdue billing occurrences in trailing 90 days"
    },
    "contract_type": {
        "expected_type": "string / category",
        "role": "Categorical Feature",
        "is_required": "Yes",
        "is_nullable": "No",
        "description": "Customer subscription contract duration (Two-year, Annual, Monthly)"
    },
    "autopay_enabled": {
        "expected_type": "string / category",
        "role": "Categorical Feature",
        "is_required": "Yes",
        "is_nullable": "No",
        "description": "Whether automated recurring billing is enabled (Yes, No)"
    },
    "satisfaction_score": {
        "expected_type": "integer",
        "role": "Numeric / Ordinal Feature",
        "is_required": "Yes",
        "is_nullable": "No",
        "description": "Survey customer satisfaction rating on an integer scale of 1 to 5"
    },
    "discount_currently_active": {
        "expected_type": "string / category",
        "role": "Categorical Feature",
        "is_required": "Yes",
        "is_nullable": "No",
        "description": "Whether a promotional price concession is currently applied (Yes, No)"
    },
    "churned_90d": {
        "expected_type": "integer (binary 0/1)",
        "role": "Target Variable",
        "is_required": "Yes",
        "is_nullable": "No",
        "description": "Binary ground-truth churn status over prospective 90-day horizon"
    },
    "feature_timestamp": {
        "expected_type": "datetime string (DD-MM-YYYY HH:MM)",
        "role": "Feature Observation Timestamp",
        "is_required": "Yes",
        "is_nullable": "No",
        "description": "Observation snapshot timestamp for feature measurement"
    }
}


def validate_schema(
    data_path: Optional[Path] = None,
    output_report_path: Optional[Path] = None
) -> pd.DataFrame:
    """Validates dataset columns against expected schema specification."""
    root = get_project_root()
    if data_path is None:
        # Check potential data file locations
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

    if output_report_path is None:
        output_report_path = root / "reports" / "schema_report.csv"

    output_report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[Schema Validation] Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    total_records, total_columns = df.shape
    print(f"[Schema Validation] Records: {total_records}, Columns: {total_columns}")

    report_rows = []
    observed_cols = list(df.columns)

    # 1. Validate each expected column
    for col, spec in EXPECTED_SCHEMA.items():
        if col in df.columns:
            observed_dtype = str(df[col].dtype)
            null_count = int(df[col].isnull().sum())
            is_unique = "Yes" if df[col].nunique() == total_records else "No"

            # Check type alignment
            type_match = "PASSED"
            if "int" in spec["expected_type"] and not pd.api.types.is_integer_dtype(df[col]):
                type_match = "WARNING (Expected int, got " + observed_dtype + ")"
            elif "float" in spec["expected_type"] and not pd.api.types.is_float_dtype(df[col]):
                type_match = "WARNING (Expected float, got " + observed_dtype + ")"

            status = "PASSED" if (null_count == 0 and "WARNING" not in type_match) else "REVIEW"

            report_rows.append({
                "column_name": col,
                "role": spec["role"],
                "expected_type": spec["expected_type"],
                "observed_dtype": observed_dtype,
                "is_required": spec["is_required"],
                "is_nullable": spec["is_nullable"],
                "null_count": null_count,
                "unique_values": df[col].nunique(),
                "is_primary_key": is_unique,
                "status": status,
                "details": f"Present; nulls={null_count}; unique={df[col].nunique()}; {type_match}"
            })
        else:
            report_rows.append({
                "column_name": col,
                "role": spec["role"],
                "expected_type": spec["expected_type"],
                "observed_dtype": "MISSING",
                "is_required": spec["is_required"],
                "is_nullable": spec["is_nullable"],
                "null_count": total_records,
                "unique_values": 0,
                "is_primary_key": "No",
                "status": "FAILED",
                "details": "Column missing from dataset!"
            })

    # 2. Check for unexpected columns
    unexpected = [c for c in observed_cols if c not in EXPECTED_SCHEMA]
    for col in unexpected:
        report_rows.append({
            "column_name": col,
            "role": "Unexpected Feature",
            "expected_type": "N/A",
            "observed_dtype": str(df[col].dtype),
            "is_required": "No",
            "is_nullable": "Unknown",
            "null_count": int(df[col].isnull().sum()),
            "unique_values": df[col].nunique(),
            "is_primary_key": "No",
            "status": "UNEXPECTED",
            "details": "Extra column detected not in schema specification"
        })

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(output_report_path, index=False)
    print(f"[Schema Validation] Schema report written to: {output_report_path}")

    # Print summary table
    print("\n" + "=" * 75)
    print(f"{'COLUMN NAME':<26} {'OBSERVED DTYPE':<16} {'NULLS':<8} {'STATUS':<10}")
    print("-" * 75)
    for _, row in report_df.iterrows():
        print(f"{row['column_name']:<26} {row['observed_dtype']:<16} {row['null_count']:<8} {row['status']:<10}")
    print("=" * 75)

    all_passed = all(r["status"] == "PASSED" for r in report_rows if r["is_required"] == "Yes")
    if all_passed:
        print("[Schema Validation] ALL REQUIRED SCHEMA CHECKS PASSED SUCCESSFULLY!\n")
    else:
        print("[Schema Validation] ATTENTION: Some schema checks require review.\n")

    return report_df


if __name__ == "__main__":
    validate_schema()

