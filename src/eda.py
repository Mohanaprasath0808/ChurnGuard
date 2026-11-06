"""Exploratory Data Analysis (EDA) Module for ChurnGuard Customer Churn Prediction.

Generates statistically grounded EDA visual artifacts and summary tables:
1. Customer churn distribution (target balance).
2. Numerical feature distributions (tenure, charges, usage, tickets, late payments).
3. Contract type analysis vs churn.
4. Monthly charges analysis vs churn.
5. Customer tenure analysis vs churn.
6. Correlation heatmap across numeric and encoded features.
7. Outlier analysis via IQR boxplots.

All visual artifacts are saved into reports/eda_report/.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for Windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_project_root() -> Path:
    """Resolves the ChurnGuard project root directory."""
    current = Path(__file__).resolve().parent
    if current.name == "src":
        return current.parent
    if (current / "ChurnGuard").exists():
        return current / "ChurnGuard"
    return current


def run_eda(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """Generates all required EDA figures and summary tables."""
    root = get_project_root()

    if data_path is None:
        candidates = [
            root / "data" / "updated customerchurn.csv",
            root / "data" / "raw" / "updated customerchurn.csv",
            Path("ChurnGuard/data/updated customerchurn.csv"),
            Path("data/updated customerchurn.csv"),
        ]
        for c in candidates:
            if c.exists():
                data_path = c
                break
        if data_path is None:
            raise FileNotFoundError(f"Could not locate updated customerchurn.csv under {root}")

    if output_dir is None:
        output_dir = root / "reports" / "eda_report"

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_artifacts: Dict[str, Path] = {}

    print(f"\n[EDA Analysis] Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    total_records = len(df)
    target_col = "churned_90d"

    # Styling configuration
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # ==========================================
    # 1. CUSTOMER CHURN DISTRIBUTION
    # ==========================================
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df[target_col].value_counts().sort_index()
    labels = ["Retained (0)", "Churned (1)"]
    colors = ["#22c55e", "#ef4444"]

    bars = ax.bar(labels, counts.values, color=colors, edgecolor="black", linewidth=1.2, width=0.55)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total_records) * 100.0
        ax.annotate(f"{h:,}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_title("Customer Churn Distribution (churned_90d)", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Customer Count", fontsize=12)
    ax.set_ylim(0, max(counts.values) * 1.20)
    p1 = output_dir / "churn_distribution.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    saved_artifacts["churn_distribution"] = p1
    print(f"[EDA Analysis] Saved: {p1.name}")

    # ==========================================
    # 2. NUMERICAL FEATURE DISTRIBUTIONS
    # ==========================================
    num_cols = ["tenure_months", "monthly_charges", "support_tickets_90d", "avg_monthly_usage_hours", "late_payments_90d"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, col in enumerate(num_cols):
        ax = axes[idx]
        sns.histplot(df[col], kde=True, ax=ax, color="#3b82f6", edgecolor="black", bins=20)
        ax.set_title(f"Distribution: {col}", fontsize=12, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    # Hide unused 6th subplot
    axes[5].axis("off")
    # Add summary text in 6th spot
    summary_text = (
        "NUMERICAL SUMMARY:\n"
        f"• Total Rows: {total_records:,}\n"
        f"• Tenure Range: {df['tenure_months'].min()} - {df['tenure_months'].max()}m\n"
        f"• Charges: ${df['monthly_charges'].min():.2f} - ${df['monthly_charges'].max():.2f}\n"
        f"• Max Tickets (90d): {df['support_tickets_90d'].max()}\n"
        f"• Max Late Pay (90d): {df['late_payments_90d'].max()}"
    )
    axes[5].text(0.1, 0.4, summary_text, fontsize=12, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.8", facecolor="#f1f5f9", edgecolor="#cbd5e1"))

    fig.suptitle("Numerical Features Distributions", fontsize=16, fontweight="bold", y=1.02)
    p2 = output_dir / "numerical_distributions.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_artifacts["numerical_distributions"] = p2
    print(f"[EDA Analysis] Saved: {p2.name}")

    # ==========================================
    # 3. CONTRACT TYPE ANALYSIS
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ct_counts = pd.crosstab(df["contract_type"], df[target_col])
    ct_pct = pd.crosstab(df["contract_type"], df[target_col], normalize="index") * 100.0

    # Count plot
    ct_counts.plot(kind="bar", stacked=False, ax=ax1, color=["#22c55e", "#ef4444"], edgecolor="black")
    ax1.set_title("Customer Count by Contract Type & Churn", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Contract Type")
    ax1.set_ylabel("Count")
    ax1.legend(["Retained (0)", "Churned (1)"], title="Status")
    ax1.tick_params(axis="x", rotation=0)

    # Percentage plot
    ct_pct.plot(kind="bar", stacked=True, ax=ax2, color=["#22c55e", "#ef4444"], edgecolor="black")
    ax2.set_title("Churn Rate by Contract Type (%)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Contract Type")
    ax2.set_ylabel("Percentage")
    ax2.legend(["Retained (0)", "Churned (1)"], title="Status")
    ax2.tick_params(axis="x", rotation=0)
    for c in ax2.containers:
        ax2.bar_label(c, fmt="%.1f%%", label_type="center", fontweight="bold", color="white")

    p3 = output_dir / "churn_by_contract.png"
    fig.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_artifacts["churn_by_contract"] = p3
    print(f"[EDA Analysis] Saved: {p3.name}")

    # ==========================================
    # 4. MONTHLY CHARGES ANALYSIS
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.kdeplot(data=df, x="monthly_charges", hue=target_col, fill=True, common_norm=False,
                palette=["#22c55e", "#ef4444"], ax=ax1, alpha=0.4)
    ax1.set_title("Monthly Charges Density by Churn", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Monthly Charges ($)")
    ax1.legend(["Churned (1)", "Retained (0)"], title="Status")

    sns.boxplot(data=df, x=target_col, y="monthly_charges", hue=target_col,
                palette=["#22c55e", "#ef4444"], legend=False, ax=ax2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Retained (0)", "Churned (1)"])
    ax2.set_title("Monthly Charges Distribution Boxplot", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Churn Status")
    ax2.set_ylabel("Monthly Charges ($)")

    p4 = output_dir / "monthly_charges_vs_churn.png"
    fig.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_artifacts["monthly_charges_vs_churn"] = p4
    print(f"[EDA Analysis] Saved: {p4.name}")

    # ==========================================
    # 5. CUSTOMER TENURE ANALYSIS
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.kdeplot(data=df, x="tenure_months", hue=target_col, fill=True, common_norm=False,
                palette=["#22c55e", "#ef4444"], ax=ax1, alpha=0.4)
    ax1.set_title("Customer Tenure Density by Churn", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Tenure (Months)")
    ax1.legend(["Churned (1)", "Retained (0)"], title="Status")

    sns.boxplot(data=df, x=target_col, y="tenure_months", hue=target_col,
                palette=["#22c55e", "#ef4444"], legend=False, ax=ax2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Retained (0)", "Churned (1)"])
    ax2.set_title("Tenure Months Distribution Boxplot", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Churn Status")
    ax2.set_ylabel("Tenure (Months)")

    p5 = output_dir / "tenure_vs_churn.png"
    fig.savefig(p5, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_artifacts["tenure_vs_churn"] = p5
    print(f"[EDA Analysis] Saved: {p5.name}")

    # ==========================================
    # 6. CORRELATION HEATMAP
    # ==========================================
    # Prepare numeric + encoded categorical dataframe
    df_corr = df[num_cols + ["satisfaction_score", target_col]].copy()
    corr_matrix = df_corr.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature Correlation Heatmap with churned_90d", fontsize=14, fontweight="bold", pad=12)

    p6 = output_dir / "correlation_heatmap.png"
    fig.savefig(p6, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_artifacts["correlation_heatmap"] = p6
    print(f"[EDA Analysis] Saved: {p6.name}")

    # ==========================================
    # 7. OUTLIER ANALYSIS (IQR Boxplots)
    # ==========================================
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    outlier_cols = ["tenure_months", "monthly_charges", "avg_monthly_usage_hours", "support_tickets_90d"]

    outlier_summary = []
    for idx, col in enumerate(outlier_cols):
        ax = axes[idx]
        sns.boxplot(y=df[col], color="#60a5fa", ax=ax, width=0.4)
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_ylabel(col)

        # IQR computation
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((df[col] < lower) | (df[col] > upper)).sum())
        outlier_summary.append({
            "feature": col,
            "q1": round(q1, 2),
            "median": round(df[col].median(), 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
            "outliers_detected": outlier_count,
            "outlier_pct": f"{round((outlier_count / total_records) * 100.0, 2)}%"
        })

    fig.suptitle("Outlier Detection (1.5 x IQR Bounds)", fontsize=15, fontweight="bold", y=1.02)
    p7 = output_dir / "outlier_analysis.png"
    fig.savefig(p7, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_artifacts["outlier_analysis"] = p7
    print(f"[EDA Analysis] Saved: {p7.name}")

    # Export statistical summaries to CSV
    stat_summary = df[num_cols + ["satisfaction_score"]].describe().T
    stat_summary.to_csv(output_dir / "numerical_statistical_summary.csv")
    pd.DataFrame(outlier_summary).to_csv(output_dir / "outlier_summary.csv", index=False)
    print(f"[EDA Analysis] Statistical summaries exported to {output_dir}")

    print("\n[EDA Analysis] All EDA artifacts successfully generated!")
    return saved_artifacts


if __name__ == "__main__":
    run_eda()
