from pathlib import Path
import pandas as pd
import matplotlib

# Non-interactive backend: graphs save ஆகும்; window open ஆகாது
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "customer_churn.csv"
REPORT_DIR = ROOT / "reports" / "eda"
GRAPH_DIR = REPORT_DIR / "graphs"

# Create output folders
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def save_graph(filename):
    """Save graph and close the figure."""
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graph saved: {filename}")


def main():

    print("\n====================================")
    print("       CHURNGUARD - EDA REPORT")
    print("====================================")

    # 1. Load dataset
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print("\n1. DATASET INFORMATION")

    print("Dataset Shape:", df.shape)
    print("Total Rows:", df.shape[0])
    print("Total Columns:", df.shape[1])

    print("\nColumn Names:")
    print(df.columns.tolist())

    # 2. Dataset information
    print("\n2. DATA TYPES")
    print(df.dtypes)

    # 3. Missing values
    print("\n3. MISSING VALUE ANALYSIS")

    missing = df.isnull().sum()

    print(missing)

    missing.to_csv(REPORT_DIR / "missing_values.csv")

    # 4. Duplicate analysis
    print("\n4. DUPLICATE ANALYSIS")

    print("Duplicate Rows:", df.duplicated().sum())
    print("Duplicate Customer IDs:",
          df["customer_id"].duplicated().sum())

    # 5. Statistical summary
    print("\n5. STATISTICAL SUMMARY")

    print(df.describe().round(2))

    df.describe().to_csv(
        REPORT_DIR / "statistical_summary.csv"
    )

    # 6. Churn distribution
    print("\n6. CUSTOMER CHURN DISTRIBUTION")

    churn_counts = df["churned_90d"].value_counts()

    print(churn_counts)

    churn_percent = (
        df["churned_90d"].value_counts(normalize=True) * 100
    )

    print("\nChurn Percentage:")
    print(churn_percent.round(2))

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="churned_90d"
    )

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.ylabel("Number of Customers")

    save_graph("01_churn_distribution.png")

    # 7. Contract type vs churn
    print("\n7. CONTRACT TYPE VS CHURN")

    contract_analysis = pd.crosstab(
        df["contract_type"],
        df["churned_90d"],
        normalize="index"
    ) * 100

    print(contract_analysis.round(2))

    contract_analysis.to_csv(
        REPORT_DIR / "contract_churn_analysis.csv"
    )

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="contract_type",
        hue="churned_90d"
    )

    plt.title("Contract Type vs Customer Churn")
    plt.xlabel("Contract Type")
    plt.ylabel("Customer Count")

    save_graph("02_contract_vs_churn.png")

    # 8. Monthly charges vs churn
    print("\n8. MONTHLY CHARGES VS CHURN")

    print(
        df.groupby("churned_90d")["monthly_charges"]
        .mean().round(2)
    )

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=df,
        x="churned_90d",
        y="monthly_charges"
    )

    plt.title("Monthly Charges vs Churn")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.ylabel("Monthly Charges")

    save_graph("03_monthly_charges.png")

    # 9. Satisfaction score vs churn
    print("\n9. SATISFACTION SCORE VS CHURN")

    satisfaction_analysis = pd.crosstab(
        df["satisfaction_score"],
        df["churned_90d"]
    )

    print(satisfaction_analysis)

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="satisfaction_score",
        hue="churned_90d"
    )

    plt.title("Satisfaction Score vs Churn")
    plt.xlabel("Satisfaction Score")
    plt.ylabel("Customer Count")

    save_graph("04_satisfaction_vs_churn.png")

    # 10. Support tickets vs churn
    print("\n10. SUPPORT TICKETS VS CHURN")

    print(
        df.groupby("churned_90d")["support_tickets_90d"]
        .mean().round(2)
    )

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=df,
        x="churned_90d",
        y="support_tickets_90d"
    )

    plt.title("Support Tickets vs Churn")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.ylabel("Support Tickets")

    save_graph("05_support_tickets.png")

    # 11. Tenure vs churn
    print("\n11. TENURE VS CHURN")

    print(
        df.groupby("churned_90d")["tenure_months"]
        .mean().round(2)
    )

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=df,
        x="churned_90d",
        y="tenure_months"
    )

    plt.title("Customer Tenure vs Churn")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.ylabel("Tenure in Months")

    save_graph("06_tenure_vs_churn.png")

    # 12. Late payments vs churn
    print("\n12. LATE PAYMENTS VS CHURN")

    print(
        df.groupby("churned_90d")["late_payments_90d"]
        .mean().round(2)
    )

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=df,
        x="churned_90d",
        y="late_payments_90d"
    )

    plt.title("Late Payments vs Churn")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.ylabel("Late Payments")

    save_graph("07_late_payments.png")

    # 13. Correlation heatmap
    print("\n13. CORRELATION ANALYSIS")

    numeric_df = df.select_dtypes(include="number")

    correlation = numeric_df.corr()

    print(correlation["churned_90d"].sort_values(
        ascending=False
    ).round(3))

    correlation.to_csv(
        REPORT_DIR / "correlation_matrix.csv"
    )

    plt.figure(figsize=(11, 8))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Customer Churn Correlation Heatmap")

    save_graph("08_correlation_heatmap.png")

    # 14. Final report
    print("\n====================================")
    print("       EDA COMPLETED SUCCESSFULLY")
    print("====================================")

    print("Total Customers:", len(df))
    print("Churned Customers:", int((df["churned_90d"] == 1).sum()))
    print("Non-Churned Customers:", int((df["churned_90d"] == 0).sum()))

    print("\nReports Location:", REPORT_DIR)
    print("Graphs Location:", GRAPH_DIR)


if __name__ == "__main__":
    main()