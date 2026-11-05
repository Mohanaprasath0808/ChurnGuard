
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "reports" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load training dataset
df = pd.read_csv(DATA_DIR / "train.csv")

TARGET = "churned_90d"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

print("\n========== CHURNGUARD EDA ==========")
print("Dataset Shape:", df.shape)

# 1. Dataset information
print("\n--- Dataset Information ---")
print(df.info())

# 2. Statistical summary
print("\n--- Statistical Summary ---")
print(df.describe(include="all").to_string())

df.describe(include="all").to_csv(
    OUTPUT_DIR / "statistical_summary.csv"
)

# 3. Churn distribution
print("\n--- Churn Distribution ---")
print(df[TARGET].value_counts(dropna=False))

plt.figure(figsize=(7, 5))
df[TARGET].value_counts().sort_index().plot(
    kind="bar", rot=0
)
plt.title("Customer Churn Distribution")
plt.xlabel("Churned (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "churn_distribution.png")
plt.close()

# 4. Churn by contract type
if "contract_type" in df.columns:
    churn_contract = pd.crosstab(
        df["contract_type"],
        df[TARGET],
        normalize="index"
    ) * 100

    print("\n--- Churn Percentage by Contract ---")
    print(churn_contract)

    churn_contract.to_csv(
        OUTPUT_DIR / "churn_by_contract.csv"
    )

    churn_contract.plot(kind="bar", figsize=(8, 5))
    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Percentage")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "churn_by_contract.png")
    plt.close()

# 5. Numerical feature analysis
numeric_features = [
    "tenure_months",
    "monthly_charges",
    "support_tickets_90d",
    "avg_monthly_usage_hours",
    "late_payments_90d",
    "satisfaction_score"
]

for feature in numeric_features:
    if feature in df.columns:
        plt.figure(figsize=(7, 5))

        groups = [
            df.loc[df[TARGET] == label, feature].dropna()
            for label in sorted(df[TARGET].dropna().unique())
        ]
        labels = [
            f"Churn = {label}"
            for label in sorted(df[TARGET].dropna().unique())
        ]

        plt.boxplot(groups, tick_labels=labels)
        plt.title(f"{feature} vs Customer Churn")
        plt.ylabel(feature)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{feature}_vs_churn.png")
        plt.close()

# 6. Correlation analysis
numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

plt.figure(figsize=(10, 7))
plt.imshow(correlation, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.close()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Reports saved in:", OUTPUT_DIR)
