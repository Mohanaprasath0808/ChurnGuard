# ChurnGuard: Customer Churn Prediction — Data Dictionary

**Dataset**: `updated customerchurn.csv`  
**Total Records**: 2,500 customer accounts  
**Total Columns**: 12 features (1 identifier, 5 numerical features, 4 categorical/ordinal features, 1 feature observation timestamp, 1 target)  
**Target Variable**: `churned_90d` (Binary classification: 0 = Retained, 1 = Churned)  
**Timestamp Format**: `DD-MM-YYYY HH:MM` (e.g. `02-02-2025 13:50`)  
**Project Workspace**: ChurnGuard  

---

## 1. Feature Specifications & Metadata

| Column Name | Data Type | Nullable | Role / Purpose | Allowed Values / Range | Description & Modeling Guidelines |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `customer_id` | String | No | **Identifier** | Alphanumeric (`CUST-XXXXX`, e.g. `CUST-00001` to `CUST-02500`) | Unique account identifier. **Strict Exclusion**: Must NOT be included in model training features to prevent memorization / identity leakage. |
| `tenure_months` | Integer | No | **Numeric Feature** | 1 to 72 months (observed mean: 36.74, median: 37) | Total months the customer has been active with the service. Key retention indicator; low tenure indicates new customer hazard. |
| `monthly_charges` | Float | No | **Numeric Feature** | \$20.08 to \$179.99 (observed mean: \$99.77, median: \$99.62) | Recurring monthly billing cost. Higher monthly charges generally correlate with increased churn sensitivity. |
| `support_tickets_90d` | Integer | No | **Numeric Feature** | 0 to 8 tickets (observed mean: 4.00, median: 4) | Number of customer support escalation tickets filed over the trailing 90 days. High ticket counts denote unresolved dissatisfaction. |
| `avg_monthly_usage_hours` | Float | No | **Numeric Feature** | 0.0 to 103.6 hours (observed mean: 45.11, median: 44.85) | Average monthly service engagement in hours. Low usage indicates disengagement risk. |
| `late_payments_90d` | Integer | No | **Numeric Feature** | 0 to 5 payments (observed mean: 2.52, median: 3) | Delinquent invoice payments over the past 90 days. Strong leading indicator of payment failure or service termination. |
| `contract_type` | String | No | **Categorical Feature** | `['Two-year', 'Annual', 'Monthly']` | Customer service agreement commitment duration. Monthly contracts exhibit significantly higher churn rates than Two-year agreements. |
| `autopay_enabled` | String | No | **Categorical Feature** | `['Yes', 'No']` | Whether automated direct debit / credit card billing is configured. Non-autopay accounts exhibit higher attrition. |
| `satisfaction_score` | Integer | No | **Ordinal Feature** | 1 to 5 integer scale (1 = Highly Dissatisfied, 5 = Highly Satisfied) | Customer feedback survey response score. Scores <= 2 reflect severe churn hazard. |
| `discount_currently_active` | String | No | **Categorical Feature** | `['Yes', 'No']` | Whether a retention or promotional pricing discount is currently active on the subscription. |
| `churned_90d` | Integer | No | **Target Variable** | `{0, 1}` (0 = Retained, 1 = Churned) | Ground truth target outcome indicating whether customer churned within prospective 90-day window. Balance: 44.64% Retained (0), 55.36% Churned (1). |
| `feature_timestamp` | String / Datetime | No | **Observation Timestamp** | `DD-MM-YYYY HH:MM` (Range: `01-01-2025 04:32` to `31-12-2025 18:56`) | Exact timestamp at which the feature vector was captured. Used to enforce time-travel leakage prevention and temporal train/test validation. |

---

## 2. Data Governance & Modeling Policies

1. **Identifier Isolation**:
   - `customer_id` is a high-cardinality primary key (2,500 distinct values). It must be strictly excluded from downstream machine learning feature matrices (`X`) to prevent identity memorization and data leakage.
2. **Feature Timestamp Format**:
   - Timestamps are formatted as `DD-MM-YYYY HH:MM`. All parsing engines must use `format="%d-%m-%Y %H:%M"`. Zero missing or invalid timestamps exist in the validated dataset.
3. **Target Handling**:
   - `churned_90d` is the supervised target variable (`y`). It must be separated from feature sets before any model training, preprocessing, or exploratory feature engineering.
4. **Data Integrity**:
   - The dataset contains 2,500 complete records with 0 missing values across all columns and 0 duplicate rows.
