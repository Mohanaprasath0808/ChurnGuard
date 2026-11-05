
# ChurnGuard - Feature Timing Review

## 1. Dataset Information

Dataset: customer_churn.csv
Dataset Type: Sample / Synthetic Dataset
Total Records: 2500
Target Column: churned_90d

## 2. Target Definition

churned_90d = 1:
Customer churned within the next 90 days.

churned_90d = 0:
Customer did not churn within the next 90 days.

Note: These target labels are sample labels.
They are not verified against actual customer churn records.

## 3. Feature Timing Assumptions

| Feature | Expected Timing | Verification |
|---|---|---|
| tenure_months | Before prediction date | Assumed |
| monthly_charges | Before prediction date | Assumed |
| support_tickets_90d | Previous 90 days | Assumed |
| avg_monthly_usage_hours | Historical usage | Assumed |
| late_payments_90d | Previous 90 days | Assumed |
| contract_type | Before prediction date | Assumed |
| autopay_enabled | Before prediction date | Assumed |
| satisfaction_score | Before prediction date | Assumed |
| discount_currently_active | At prediction date | Assumed |

## 4. Leakage Review

- customer_id excluded from model features.
- churned_90d excluded from input features.
- Direct target duplication check completed.
- Actual feature timestamps are unavailable.
- Actual churn dates are unavailable.

## 5. Final Verification Status

Status: NOT VERIFIED - SAMPLE DATASET

Reason:
The dataset contains sample records without actual
prediction dates, feature timestamps, or verified
customer churn dates.

Conclusion:
The dataset can be used for educational ML experiments
and pipeline testing. Real-world predictive validity
has not been established.
