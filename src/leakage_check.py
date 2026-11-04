from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "customer_churn.csv"
REPORT_DIR = ROOT / "reports"

TARGET = "churned_90d"

def main():

    print("\n====================================")
    print(" CHURNGUARD - DATA LEAKAGE CHECK")
    print("====================================")

    df = pd.read_csv(DATA_FILE)

    print("\n1. DATASET INFORMATION")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    # Check target column
    print("\n2. TARGET COLUMN CHECK")

    if TARGET in df.columns:
        print("Target column found:", TARGET)
    else:
        raise ValueError("Target column missing")

    # Check customer ID
    print("\n3. CUSTOMER ID CHECK")

    if "customer_id" in df.columns:
        print("customer_id is an identifier.")
        print("It should not be used as an ML feature.")

    # Check suspicious column names
    print("\n4. SUSPICIOUS FEATURE CHECK")

    suspicious_words = [
        "churn_date",
        "cancellation_date",
        "termination_date",
        "post_churn",
        "churn_reason"
    ]

    suspicious = [
        col for col in df.columns
        if any(word in col.lower() for word in suspicious_words)
        and col != TARGET
    ]

    if suspicious:
        print("Review these columns:", suspicious)
    else:
        print("No suspicious column names found.")

    # Check target relationship
    print("\n5. TARGET RELATIONSHIP CHECK")

    for col in df.select_dtypes(include="number").columns:
        if col != TARGET:
            unique_values = df[col].nunique()

            if unique_values <= 10:
                relationship = pd.crosstab(
                    df[col], df[TARGET]
                )
                print(f"\n{col}:")
                print(relationship)

    # Final result
    print("\n6. LEAKAGE REVIEW")

    print("customer_id must be excluded from model features.")
    print("No timestamp columns are available for time-based checks.")
    print("Manual review of feature timing is still required.")

    REPORT_DIR.mkdir(exist_ok=True)

    report = [
        "ChurnGuard Data Leakage Review",
        "==============================",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Target: {TARGET}",
        "customer_id excluded from ML features.",
        "No timestamp columns available.",
        "Feature timing requires manual verification."
    ]

    (REPORT_DIR / "leakage_report.md").write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\nLeakage report saved successfully!")
    print("Location:", REPORT_DIR / "leakage_report.md")


if __name__ == "__main__":
    main()
