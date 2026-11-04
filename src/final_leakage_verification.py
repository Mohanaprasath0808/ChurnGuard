
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "customer_churn.csv"
REPORT_DIR = ROOT / "reports"

TARGET = "churned_90d"

EXPECTED_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_tickets_90d",
    "avg_monthly_usage_hours",
    "late_payments_90d",
    "contract_type",
    "autopay_enabled",
    "satisfaction_score",
    "discount_currently_active"
]

# Set True only after confirming the source timing
# from dataset documentation or generation logic.
TIMING_CONFIRMED = False

def main():
    print("\n=== CHURNGUARD FINAL LEAKAGE VERIFICATION ===")

    df = pd.read_csv(DATA_FILE)

    # 1. Target check
    assert TARGET in df.columns
    print("1. Target column check: PASSED")

    # 2. Identifier exclusion
    assert "customer_id" in df.columns
    print("2. Customer ID exclusion: PASSED")

    # 3. Feature schema check
    missing = set(EXPECTED_FEATURES) - set(df.columns)
    assert not missing, f"Missing features: {missing}"
    print("3. Feature schema check: PASSED")

    # 4. Target values
    assert set(df[TARGET].unique()).issubset({0, 1})
    print("4. Target values check: PASSED")

    # 5. Direct target duplication check
    feature_df = df[EXPECTED_FEATURES]

    duplicate_target_features = []

    for col in EXPECTED_FEATURES:
        if feature_df[col].equals(df[TARGET]):
            duplicate_target_features.append(col)

    if duplicate_target_features:
        print("5. Possible direct target leakage:",
              duplicate_target_features)
    else:
        print("5. Direct target duplication: NOT DETECTED")

    # 6. Time-based review
    print("\n6. FEATURE TIMING VERIFICATION")

    if TIMING_CONFIRMED:
        print("Feature timing: CONFIRMED by source documentation")
    else:
        print("Feature timing: PENDING MANUAL CONFIRMATION")

    # 7. Final status
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = [
        "ChurnGuard Final Leakage Verification",
        "=====================================",
        "Target check: PASSED",
        "Customer ID exclusion: PASSED",
        "Feature schema check: PASSED",
        "Target values check: PASSED",
        f"Direct target duplication: {duplicate_target_features or 'Not detected'}",
        f"Feature timing confirmed: {TIMING_CONFIRMED}",
        "Time-based leakage: Cannot be fully verified without timestamps."
    ]

    (REPORT_DIR / "final_leakage_verification.md").write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    if TIMING_CONFIRMED and not duplicate_target_features:
        print("\nFinal review: PASSED")
    else:
        print("\nFinal review: PENDING MANUAL CONFIRMATION")

    print("Report saved successfully!")

if __name__ == "__main__":
    main()
