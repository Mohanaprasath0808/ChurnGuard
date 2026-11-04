# ChurnGuard Feature Timing Verification

## Prediction Definition
Prediction point: Customer snapshot date
Historical window: Previous 90 days
Target window: Next 90 days

## Feature Timing Review

| Feature | Required timing | Status |
|---|---|---|
| tenure_months | At prediction date | Pending |
| monthly_charges | At prediction date | Pending |
| support_tickets_90d | Previous 90 days | Pending |
| avg_monthly_usage_hours | Previous 90 days | Pending |
| late_payments_90d | Previous 90 days | Pending |
| contract_type | At prediction date | Pending |
| autopay_enabled | At prediction date | Pending |
| satisfaction_score | Before prediction date | Pending |
| discount_currently_active | At prediction date | Pending |
| churned_90d | Next 90 days only | Pending |

## Final Review
Prediction date confirmed: No
Feature timing confirmed: No
Target window confirmed: No

Final status: PENDING MANUAL CONFIRMATION