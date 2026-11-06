"""Data Leakage Audit Execution for ChurnGuard Workspace.

Executes all 8 data leakage, timestamp, and target horizon checks
and exports reports/leakage/leakage_report.csv and reports/leakage/target_leakage_report.md.
"""

import sys
from pathlib import Path

# Add project root and ChurnGuard to sys.path
CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[2]  # New folder (9)
CHURNGUARD_DIR = CURRENT_FILE.parents[1]  # ChurnGuard

for d in [str(ROOT_DIR), str(CHURNGUARD_DIR)]:
    if d not in sys.path:
        sys.path.insert(0, d)

import pandas as pd
from src.data_quality.leakage_checks import TargetLeakageAuditor


def run_leakage_checks() -> dict:
    current = Path(__file__).resolve().parent
    root = current.parent if current.name == "src" else current

    data_path = root / "data" / "updated customerchurn.csv"
    if not data_path.exists():
        data_path = root / "data" / "raw" / "updated customerchurn.csv"

    reports_dir = root / "reports" / "leakage"
    splits_dir = root / "data" / "processed"

    print(f"\n[Leakage Audit] Loading dataset: {data_path}")
    df = pd.read_csv(data_path)

    auditor = TargetLeakageAuditor(
        reports_dir=reports_dir,
        splits_dir=splits_dir,
        timestamp_format="%d-%m-%Y %H:%M",
    )

    summary = auditor.audit(df=df, raw_filepath=data_path)

    print("\n" + "=" * 80)
    print(" CHURNGUARD DATA LEAKAGE AUDIT SUMMARY")
    print("=" * 80)
    print(f" Total Leakage Checks:       {summary['total_leakage_checks']}")
    print(f" Passed Checks:              {summary['passed_checks']}")
    print(f" Failed Checks:              {summary['failed_checks']}")
    print(f" Not Verifiable Checks:      {summary['not_verifiable_checks']}")
    print("-" * 80)
    print(f" Customer Overlap Count:     {summary['customer_overlap_count']}")
    print(f" Invalid Timestamp Count:    {summary['invalid_timestamp_count']}")
    print(f" Missing Timestamp Count:    {summary['missing_timestamp_count']}")
    print(f" Future Timestamp Count:     {summary['future_timestamp_count']}")
    print(f" Target Leakage Status:      {summary['target_leakage_status']}")
    print(f" 90-Day Horizon Status:      {summary['target_horizon_status']}")
    print("-" * 80)
    print(f" Generated Reports:")
    print(f"  - CSV: {summary['reports_generated']['csv_report']}")
    print(f"  - MD:  {summary['reports_generated']['markdown_report']}")
    print("=" * 80 + "\n")

    return summary


if __name__ == "__main__":
    run_leakage_checks()
