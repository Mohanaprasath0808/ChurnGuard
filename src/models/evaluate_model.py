
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "artifacts" / "models"
REPORT_DIR = PROJECT_DIR / "reports" / "evaluation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "churned_90d"
ID_COLUMN = "customer_id"

# Load validation and test datasets
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

X_val = validation.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
y_val = validation[TARGET]

X_test = test.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
y_test = test[TARGET]

# Find best model from validation comparison
comparison = pd.read_csv(
    PROJECT_DIR / "reports" / "models" / "model_comparison.csv"
)

best_model_name = comparison.loc[
    comparison["f1_score"].idxmax(), "model"
]

model_path = MODEL_DIR / f"{best_model_name.lower()}.joblib"
model = joblib.load(model_path)

print("Selected model:", best_model_name)

# Threshold tuning using validation data only
val_probabilities = model.predict_proba(X_val)[:, 1]

threshold_results = []

for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    predictions = (val_probabilities >= threshold).astype(int)

    threshold_results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_val, predictions, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    REPORT_DIR / "threshold_comparison.csv",
    index=False
)

best_threshold = threshold_df.loc[
    threshold_df["f1_score"].idxmax(), "threshold"
]

print("\nThreshold comparison:")
print(threshold_df.to_string(index=False))
print("\nSelected threshold:", best_threshold)

# Final evaluation on test data
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

metrics = {
    "selected_model": best_model_name,
    "selected_threshold": float(best_threshold),
    "precision": float(precision_score(
        y_test, test_predictions, zero_division=0
    )),
    "recall": float(recall_score(
        y_test, test_predictions, zero_division=0
    )),
    "f1_score": float(f1_score(
        y_test, test_predictions, zero_division=0
    )),
    "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
    "pr_auc": float(average_precision_score(
        y_test, test_probabilities
    )),
    "confusion_matrix": confusion_matrix(
        y_test, test_predictions
    ).tolist()
}

with open(REPORT_DIR / "evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Confusion matrix visualization
ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_predictions,
    display_labels=["Not Churned", "Churned"]
)

plt.title("ChurnGuard - Confusion Matrix")
plt.tight_layout()
plt.savefig(REPORT_DIR / "confusion_matrix.png")
plt.close()

# Classification report
report = classification_report(
    y_test,
    test_predictions,
    target_names=["Not Churned", "Churned"],
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    REPORT_DIR / "classification_report.csv"
)

print("\nFinal Test Evaluation:")
print(json.dumps(metrics, indent=4))
print("\nMODEL EVALUATION COMPLETED SUCCESSFULLY!")
print("Reports saved at:", REPORT_DIR)
