
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_DIR / "data" / "customer_churn.csv"
OUTPUT_DIR = PROJECT_DIR / "reports" / "data_dictionary"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

dictionary = [
    {
        "column_name": "customer_id",
        "description": "Unique identifier assigned to each customer.",
        "data_type": "String",
        "allowed_values": "Unique customer ID",
        "purpose": "Customer identification; exclude from model training."
    },
    {
        "column_name": "tenure_months",
        "description": "Number of months the customer has stayed with the company.",
        "data_type": "Integer",
        "allowed_values": "Non-negative integer",
        "purpose": "Customer relationship duration."
    },
    {
        "column_name": "monthly_charges",
        "description": "Amount charged to the customer per month.",
        "data_type": "Numeric",
        "allowed_values": "Non-negative number",
        "purpose": "Customer billing analysis."
    },
    {
        "column_name": "support_tickets_90d",
        "description": "Number of support tickets recorded during the 90-day period.",
        "data_type": "Integer",
        "allowed_values": "Non-negative integer",
        "purpose": "Customer support activity."
    },
    {
        "column_name": "avg_monthly_usage_hours",
        "description": "Average hours of service usage per month.",
        "data_type": "Numeric",
        "allowed_values": "Non-negative number",
        "purpose": "Customer usage behaviour."
    },
    {
        "column_name": "late_payments_90d",
        "description": "Number of late payments recorded during the 90-day period.",
        "data_type": "Integer",
        "allowed_values": "Non-negative integer",
        "purpose": "Payment behaviour analysis."
    },
    {
        "column_name": "contract_type",
        "description": "Type of contract selected by the customer.",
        "data_type": "Categorical",
        "allowed_values": "Monthly, Annual, Two-year",
        "purpose": "Contract-based churn analysis."
    },
    {
        "column_name": "autopay_enabled",
        "description": "Indicates whether automatic payment is enabled.",
        "data_type": "Boolean",
        "allowed_values": "True/False or 1/0",
        "purpose": "Payment preference analysis."
    },
    {
        "column_name": "satisfaction_score",
        "description": "Customer satisfaction rating.",
        "data_type": "Numeric",
        "allowed_values": "Verify actual rating scale from dataset documentation.",
        "purpose": "Customer satisfaction analysis."
    },
    {
        "column_name": "discount_currently_active",
        "description": "Indicates whether a discount is currently active.",
        "data_type": "Boolean",
        "allowed_values": "True/False or 1/0",
        "purpose": "Discount and retention analysis."
    },
    {
        "column_name": "churned_90d",
        "description": "Target indicating whether the customer churned within the defined 90-day period.",
        "data_type": "Binary Integer",
        "allowed_values": "0 = Not churned, 1 = Churned",
        "purpose": "Machine learning prediction target."
    }
]

dictionary_df = pd.DataFrame(dictionary)

# Verify dataset columns against dictionary
dataset_columns = set(df.columns)
dictionary_columns = set(dictionary_df["column_name"])

missing_in_dictionary = dataset_columns - dictionary_columns
extra_in_dictionary = dictionary_columns - dataset_columns

if missing_in_dictionary or extra_in_dictionary:
    print("WARNING: Dictionary and dataset columns do not match.")
    print("Missing in dictionary:", missing_in_dictionary)
    print("Extra in dictionary:", extra_in_dictionary)
else:
    print("SUCCESS: All dataset columns are documented.")

# Add actual dataset information
dictionary_df["actual_dataset_dtype"] = dictionary_df[
    "column_name"
].map(lambda col: str(df[col].dtype) if col in df.columns else "Missing")

dictionary_df["missing_values"] = dictionary_df[
    "column_name"
].map(lambda col: int(df[col].isna().sum()) if col in df.columns else None)

dictionary_df["unique_values"] = dictionary_df[
    "column_name"
].map(lambda col: int(df[col].nunique()) if col in df.columns else None)

# Save CSV
output_file = OUTPUT_DIR / "data_dictionary.csv"
dictionary_df.to_csv(output_file, index=False)

print("\n========== CHURNGUARD DATA DICTIONARY ==========")
print(dictionary_df.to_string(index=False))
print("\nDATA DICTIONARY CREATED SUCCESSFULLY!")
print("Saved at:", output_file)
