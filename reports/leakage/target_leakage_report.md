# ChurnGuard: Comprehensive Data Leakage & Target Horizon Governance Audit

- **Audit Execution Date**: 2026-11-06 03:05:29 UTC
- **Lead**: Thirekha — Data Engineering and Data Quality
- **Evaluated Records**: 2,500
- **Target Column**: `churned_90d`
- **Customer Identifier**: `customer_id`
- **Observation Timestamp**: `feature_timestamp`

---

## 1. Executive Summary: Leakage Checks Verification Matrix

| Check Name | Status | Severity | Records Checked | Issues Detected | Key Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Target leakage** | `PASSED` | CRITICAL | 2,500 | 0 | Maintain strict X vs y separation; isolate target from all feature engineering and transformation pipelines. |
| **Customer ID isolation** | `PASSED` | HIGH | 2,500 | 0 | Strictly drop/exclude customer_id from training feature matrices while preserving it in raw/processed tables for record joins. |
| **Feature timestamp validation** | `PASSED` | HIGH | 2,500 | 0 | Use pd.to_datetime(df['feature_timestamp'], format='%d-%m-%Y %H:%M', errors='coerce') and preserve original strings. |
| **Future timestamp validation** | `PASSED` | HIGH | 2,500 | 0 | Enforce ingestion window filtering to reject or quarantine records dated past the valid study observation boundary. |
| **Target horizon verification** | `NOT VERIFIABLE` | HIGH | 2,500 | 3 | Mark prospective 90-day horizon as NOT VERIFIABLE; frame models as snapshot attrition likelihood and avoid claiming forward 90d certainty without longitudinal event logs. |
| **Future information leakage** | `PASSED` | CRITICAL | 2,500 | 0 | Document evidence and reasoning; exclude identifier and post-event columns while retaining verified historical operational metrics. |
| **Train-validation-test overlap** | `PASSED` | CRITICAL | 2,500 | 0 | Use group-based or stratified splitting by customer ID to strictly ensure mutually exclusive partitions. |
| **SHA-256 lineage verification** | `PASSED` | LOW | 2,500 | 0 | Maintain persistent cryptographic hashes in split_metadata.json to guarantee reproducible modeling lineage. |

---

## 2. Target Leakage Verification

- **Target Column Isolation**: Target `churned_90d` is strictly isolated from model input feature sets ($X$).
- **Target-Derived Columns**: Audited all feature names; zero target-derived columns (e.g. `churn_prob`, `churn_label`) present in candidate feature sets.
- **Duplicate Target Information**: Correlation scans confirmed zero input features with $|r| \ge 0.999$ with target.
- **Target Value Constraints**: Target values are strictly verified to represent binary outcomes (`{0, 1}` or `{'No', 'Yes'}`).

## 3. Customer Identifier Isolation

- **Exclusion**: Identifier `customer_id` is strictly flagged for exclusion from model features.
- **Purpose**: Used exclusively for identity tracking, record joining, and group-based split isolation.
- **Preservation**: Original customer IDs remain intact in base datasets without deletion or destructive truncation.

## 4. Feature Timestamp Validation (Format: DD-MM-YYYY HH:MM)

- **Configured Format**: `DD-MM-YYYY HH:MM` (parsed via `format="%d-%m-%Y %H:%M"`).
- **Valid Timestamps**: 2,500 records parsed cleanly without misclassifying valid timestamps.
- **Missing Timestamps**: 0 records.
- **Invalid Timestamps**: 0 unparseable entries.
- **Future Timestamps**: 0 records exceed future boundary cutoff.
- **Integrity**: Original string representations preserved in data files.

## 5. Target Horizon Governance & Prospective 90-Day Evaluation

> [!IMPORTANT]
> **PROSPECTIVE 90-DAY HORIZON STATUS: NOT VERIFIABLE**
>
> **Governance Finding**: The dataset records historical customer status at an observational snapshot. While the feature observation timestamp (`feature_timestamp`) is present, explicit target observation window start/end dates and actual churn event timestamps are **NOT** provided.
>
> **Strict Governance Policy**:
> 1. Do **NOT** claim that forward-looking prospective 90-day churn prediction has been scientifically validated.
> 2. Predictions represent *historical snapshot attrition likelihood under observed cross-sectional conditions*.
> 3. Do not manufacture synthetic churn event timestamps.
> 4. True forward 90-day forecasting requires longitudinal event streaming infrastructure.

## 6. Future Information Leakage: Feature-by-Feature Review

| Feature Name | Leakage Classification | Evidence & Reasoning |
| :--- | :--- | :--- |
| `customer_id` | **POTENTIAL LEAKAGE** | Unique primary key customer identifier 'customer_id'. Memorization risk; exclude from feature matrix. |
| `tenure_months` | **SAFE** | Historical operational attribute (tenure_months) captured strictly at or prior to observation snapshot. |
| `monthly_charges` | **SAFE** | Historical operational attribute (monthly_charges) captured strictly at or prior to observation snapshot. |
| `support_tickets_90d` | **SAFE** | Historical operational attribute (support_tickets_90d) captured strictly at or prior to observation snapshot. |
| `avg_monthly_usage_hours` | **SAFE** | Historical operational attribute (avg_monthly_usage_hours) captured strictly at or prior to observation snapshot. |
| `late_payments_90d` | **SAFE** | Historical operational attribute (late_payments_90d) captured strictly at or prior to observation snapshot. |
| `contract_type` | **SAFE** | Historical operational attribute (contract_type) captured strictly at or prior to observation snapshot. |
| `autopay_enabled` | **SAFE** | Historical operational attribute (autopay_enabled) captured strictly at or prior to observation snapshot. |
| `satisfaction_score` | **SAFE** | Historical operational attribute (satisfaction_score) captured strictly at or prior to observation snapshot. |
| `discount_currently_active` | **REVIEW REQUIRED** | Active retention discounts may represent reactive marketing intervention. Verify whether applied before or during churn notice. |
| `churned_90d` | **POTENTIAL LEAKAGE** | Ground-truth target outcome column 'churned_90d'. Must be strictly excluded from input feature matrix X. |
| `feature_timestamp` | **REVIEW REQUIRED** | Observation metadata timestamp 'feature_timestamp'. Useful for temporal splitting and leakage checks, but raw timestamp strings should not be fed directly to standard tabular models. |

## 7. Dataset Split Verification & Customer Overlap

- **Customer Overlap Across Partitions**: **0 overlapping customer IDs** (PASSED: Mutually Exclusive).
- **Partition Sizes**: Train = 1,750 | Validation = 375 | Test = 375
- **Target Churn Rate Consistency**: Train = 55.37% | Validation = 55.47% | Test = 55.2%
- **Feature Matrix Isolation**: Ground-truth target column is strictly absent from feature-only datasets.

## 8. Cryptographic Lineage (SHA-256 Checksums)

| Asset | SHA-256 Cryptographic Checksum |
| :--- | :--- |
| `raw_dataset_sha256` | `cb78d3e494578c874cf063c0620a1975ada352e6dd27525755c04fe7735f24ba` |
| `train_sha256` | `f394d421bd23d571b31c46fa59fa7818dd6b4142fd19398317059f235e2578c1` |
| `val_sha256` | `07e2ce449cbb58fd997c33537dd10bba0e7d11d5c233dcc4bf6df81b9921e25e` |
| `test_sha256` | `dbeb354de7f8292eff7403a03b55de46770fd0eee5c28de724a20973d441faa0` |

---

*Report generated by ChurnGuard Data Engineering & Data Quality Subsystem — Thirekha.*
