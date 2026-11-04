# ChurnGuard – Data Dictionary

## Dataset Information
- Dataset Name: customer_churn.csv
- Total Records: 2500
- Total Columns: 11
- Target Column: churned_90d

## Column Details

| Column Name | Data Type | Description |
|---|---|---|
| customer_id | String | Unique customer identifier |
| tenure_months | Integer | Customer subscription duration |
| monthly_charges | Float | Monthly subscription charges |
| support_tickets_90d | Integer | Support tickets in previous 90 days |
| avg_monthly_usage_hours | Float | Average monthly usage hours |
| late_payments_90d | Integer | Late payments in previous 90 days |
| contract_type | String | Customer contract type |
| autopay_enabled | String | Whether auto payment is enabled |
| satisfaction_score | Integer | Customer satisfaction score |
| discount_currently_active | String | Whether a discount is active |
| churned_90d | Integer | Churn outcome in the next 90 days (0 = No, 1 = Yes) |

## Data Quality
- Missing Values: 0
- Duplicate Rows: 0
- Duplicate Customer IDs: 0

## Dataset Splitting
- Training: 70% (1750 records)
- Validation: 15% (375 records)
- Testing: 15% (375 records)

## Notes
Customer ID must be excluded from model training. Feature timing must be confirmed to avoid data leakage.