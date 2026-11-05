
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/train.csv")

features = [
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

target = "churned_90d"

print("=" * 55)
print("CHURNGUARD FINAL TIMING VERIFICATION")
print("=" * 55)

# Column validation
missing = [c for c in features + [target] if c not in df.columns]

print("\n1. COLUMN VALIDATION")

if not missing:
    print("PASSED")
else:
    print("FAILED:", missing)

# Target exclusion
print("\n2. TARGET LEAKAGE CHECK")

if target not in features:
    print("PASSED - Target excluded from features")
else:
    print("FAILED - Target included in features")

# Customer ID exclusion
print("\n3. CUSTOMER ID CHECK")

if "customer_id" not in features:
    print("PASSED - Customer ID excluded")
else:
    print("FAILED - Customer ID included")

# Target values
print("\n4. TARGET VALUE CHECK")

if target in df.columns:
    print("Unique target values:", sorted(df[target].dropna().unique()))

    if set(df[target].dropna().unique()).issubset({0, 1}):
        print("PASSED - Binary target format")
    else:
        print("FAILED - Unexpected target values")

# Timing status
print("\n5. FEATURE TIMING REVIEW")

for feature in features:
    print(f"{feature}: ASSUMED - NOT VERIFIED")

print("\n6. FINAL STATUS")
print("NOT VERIFIED - SAMPLE DATASET")
print("Reason: Actual timestamps and churn records unavailable.")
print("=" * 55)
