
import pandas as pd

# Load training dataset
df = pd.read_csv("data/processed/train.csv")

# Features available before prediction
expected_features = [
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

print("FEATURE TIMING VERIFICATION")
print("=" * 45)

# Check whether all expected columns exist
missing = [col for col in expected_features + [target]
           if col not in df.columns]

if missing:
    print("Missing columns:", missing)
else:
    print("Column check: PASSED")

# Check target is not included in input features
if target not in expected_features:
    print("Target exclusion: PASSED")
else:
    print("Target exclusion: FAILED")

# Check customer ID is not used as a feature
if "customer_id" not in expected_features:
    print("Customer ID exclusion: PASSED")
else:
    print("Customer ID exclusion: FAILED")

print("\nFEATURE TIMING REVIEW")
print("-" * 45)

for feature in expected_features:
    print(f"{feature}: MANUAL CONFIRMATION REQUIRED")

print("\nTarget:", target)
print("Target must represent churn during the NEXT 90 days.")
print("Final timing verification: PENDING")
