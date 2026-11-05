
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# 1. PROJECT PATH CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "customer_churn.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

TARGET_COLUMN = "churned_90d"

RANDOM_STATE = 42

# ==========================================
# 2. CHECK DATASET FILE
# ==========================================

print("\n========== CHURNGUARD DATA PIPELINE ==========\n")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place customer_churn.csv inside the data folder."
    )

print("Dataset found successfully!")

# ==========================================
# 3. LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Original dataset shape:", df.shape)

# ==========================================
# 4. SCHEMA VALIDATION
# ==========================================

required_columns = [
    "customer_id",
    "tenure_months",
    "monthly_charges",
    "support_tickets_90d",
    "avg_monthly_usage_hours",
    "late_payments_90d",
    "contract_type",
    "autopay_enabled",
    "satisfaction_score",
    "discount_currently_active",
    "churned_90d"
]

missing_columns = [

    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nSchema validation: PASSED")

# ==========================================
# 5. DATA QUALITY CHECKS
# ==========================================

if df.empty:
    raise ValueError("Dataset is empty.")

if df.columns.duplicated().any():
    raise ValueError("Duplicate column names detected.")

if df[TARGET_COLUMN].isna().any():
    raise ValueError("Target column contains missing values.")

# Check customer ID
if df["customer_id"].isna().any():
    raise ValueError("Customer ID contains missing values.")

if df["customer_id"].duplicated().any():
    raise ValueError("Duplicate customer IDs detected.")

# Check target values
target_values = set(df[TARGET_COLUMN].unique())

if target_values != {0, 1}:
    raise ValueError(
        f"Expected churn target values 0 and 1, found: {target_values}"
    )

print("\nData quality checks: PASSED")

# ==========================================
# 6. DUPLICATE RECORD CHECK
# ==========================================

duplicate_count = df.duplicated().sum()

print("\nDuplicate records:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplicate records removed.")

# ==========================================
# 7. MISSING VALUE REPORT
# ==========================================

missing_report = df.isnull().sum()

print("\nMissing values report:")
print(missing_report)

# Save missing value report
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

missing_report.to_csv(
    OUTPUT_DIR / "missing_values_report.csv",
    header=["missing_count"]
)

# ==========================================
# 8. TARGET DISTRIBUTION
# ==========================================

print("\nChurn target distribution:")
print(df[TARGET_COLUMN].value_counts())

if df[TARGET_COLUMN].nunique() != 2:
    raise ValueError("Both churn classes must be present.")

# ==========================================
# 9. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

print("\nStarting dataset splitting...")

# 70% Training, 30% Temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df[TARGET_COLUMN]
)

# Split remaining 30% into 15% Validation and 15% Test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df[TARGET_COLUMN]
)

# ==========================================
# 10. SAVE PROCESSED DATASETS
# ==========================================

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)

# ==========================================
# 11. FINAL DATASET SUMMARY
# ==========================================

print("\n========== SPLITTING COMPLETED ==========\n")

print("Original records:", len(df))
print("Training records:", len(train_df))
print("Validation records:", len(val_df))
print("Testing records:", len(test_df))

print("\nDataset percentages:")

print("Training: {:.2f}%".format(
    len(train_df) / len(df) * 100
))

print("Validation: {:.2f}%".format(
    len(val_df) / len(df) * 100
))

print("Testing: {:.2f}%".format(
    len(test_df) / len(df) * 100
))

print("\nSaved files:")
print(OUTPUT_DIR / "train.csv")
print(OUTPUT_DIR / "validation.csv")
print(OUTPUT_DIR / "test.csv")
print(OUTPUT_DIR / "missing_values_report.csv")

print("\nCHURNGUARD DATA PIPELINE SUCCESSFULLY COMPLETED!")
