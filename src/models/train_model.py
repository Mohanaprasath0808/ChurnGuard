
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "artifacts" / "models"
REPORT_DIR = PROJECT_DIR / "reports" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "churned_90d"
ID_COLUMN = "customer_id"

# Load datasets
train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

# Separate features and target
X_train = train.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
y_train = train[TARGET]

X_val = validation.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
y_val = validation[TARGET]

X_test = test.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
y_test = test[TARGET]

# Identify numerical and categorical columns
numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Models
models = {
    "DummyClassifier": DummyClassifier(strategy="prior"),

    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []

print("\n========== CHURNGUARD MODEL TRAINING ==========")

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train using training dataset
    pipeline.fit(X_train, y_train)

    # Evaluate on validation dataset
    predictions = pipeline.predict(X_val)

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_val)[:, 1]
    else:
        probabilities = predictions

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_val, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_val, probabilities),
        "pr_auc": average_precision_score(y_val, probabilities)
    }

    results.append(metrics)

    print(f"\nModel: {name}")
    for key, value in metrics.items():
        if key != "model":
            print(f"{key}: {value:.4f}")

    joblib.dump(
        pipeline,
        MODEL_DIR / f"{name.lower()}.joblib"
    )

# Save validation comparison
results_df = pd.DataFrame(results)
results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

# Select best model based on validation F1
best_model_name = results_df.loc[
    results_df["f1_score"].idxmax(), "model"
]

best_model = joblib.load(
    MODEL_DIR / f"{best_model_name.lower()}.joblib"
)

# Final test evaluation only after model selection
test_predictions = best_model.predict(X_test)
test_probabilities = best_model.predict_proba(X_test)[:, 1]

test_metrics = {
    "selected_model": best_model_name,
    "accuracy": float(accuracy_score(y_test, test_predictions)),
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

with open(REPORT_DIR / "final_test_metrics.json", "w") as file:
    json.dump(test_metrics, file, indent=4)



print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nSelected Model:", best_model_name)
print("\nFinal Test Results:")
print(json.dumps(test_metrics, indent=4))

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")
print("Models saved in:", MODEL_DIR)
print("Reports saved in:", REPORT_DIR)

print("Train customer_id:", "customer_id" in X_train.columns)
print("Validation customer_id:", "customer_id" in X_val.columns)
print("Test customer_id:", "customer_id" in X_test.columns)
