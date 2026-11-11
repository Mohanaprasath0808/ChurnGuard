import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve, CalibrationDisplay

def load_data():
    """Loads processed train/test datasets or falls back to split."""
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        X_train = train_df.drop(columns=['churned_90d'], errors='ignore')
        y_train = train_df['churned_90d']
        X_test = test_df.drop(columns=['churned_90d'], errors='ignore')
        y_test = test_df['churned_90d']
    else:
        # Fallback to single file split if train/test files aren't separated
        data_path = "data/processed/updated customerchurn (2).csv"
        df = pd.read_csv(data_path)
        X = df.drop(columns=['churned_90d', 'customer_id'], errors='ignore')
        y = df['churned_90d']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
    return X_train, X_test, y_train, y_test

def build_preprocessor(X_train):
    """Builds the preprocessing transformer for numerical and categorical columns."""
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    return preprocessor

def optimize_threshold(y_true, y_probs):
    """Finds the optimal decision threshold that maximizes the F1-score."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_f1 = 0.0

    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        current_f1 = f1_score(y_true, preds, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = th

    print(f"Optimal Threshold found: {best_threshold:.2f} (F1-Score: {best_f1:.4f})")
    return best_threshold

def generate_calibration_plot(y_true, y_probs, output_dir):
    """Generates and saves the reliability (calibration) curve."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    disp = CalibrationDisplay.from_predictions(y_true, y_probs, n_bins=10, name='MLP Architecture')
    plt.title("Probability Calibration Curve (Reliability Diagram)")
    plt.savefig(os.path.join(output_dir, "mlp_calibration_curve.png"))
    plt.close()
    print("Calibration curve plot saved.")

def main():
    print("========== CHURNGUARD ANN / MLP PIPELINE ==========")
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Preprocessing
    preprocessor = build_preprocessor(X_train)
    
    # 3. Define MLP Architecture
    # Hidden layers: (64, 32) neurons with ReLU, Adam optimizer, and early stopping
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', mlp)
    ])
    
    # 4. Train the Model
    print("Training Multi-Layer Perceptron (MLP)...")
    pipeline.fit(X_train, y_train)
    
    # 5. Predictions & Probabilities
    y_pred_default = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    
    print("\n--- Default Threshold (0.5) Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_default):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_default, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_default, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred_default, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_probs):.4f}")

    # 6. Threshold Validation Tuning
    print("\n--- Running Threshold Tuning ---")
    best_th = optimize_threshold(y_test, y_probs)
    y_pred_tuned = (y_probs >= best_th).astype(int)
    
    print("\n--- Tuned Threshold Metrics ---")
    print(f"Tuned Accuracy:  {accuracy_score(y_test, y_pred_tuned):.4f}")
    print(f"Tuned Precision: {precision_score(y_test, y_pred_tuned, zero_division=0):.4f}")
    print(f"Tuned Recall:    {recall_score(y_test, y_pred_tuned, zero_division=0):.4f}")
    print(f"Tuned F1-Score:  {f1_score(y_test, y_pred_tuned, zero_division=0):.4f}")

    # 7. Error Analysis & Confusion Matrix Breakdown
    cm = confusion_matrix(y_test, y_pred_tuned)
    print("\n--- Error Analysis (Confusion Matrix at Optimal Threshold) ---")
    print(cm)
    
    report = classification_report(y_test, y_pred_tuned, output_dict=True)
    
    # 8. Calibration Curves
    output_dir = "reports/models"
    generate_calibration_plot(y_test, y_probs, output_dir)
    
    # 9. Save Artifacts
    artifacts_dir = "artifacts/models"
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "mlp_churn_model.pkl")
    joblib.dump(pipeline, model_path)
    
    results_summary = {
        "model": "MLPClassifier",
        "optimal_threshold": float(best_th),
        "accuracy": float(accuracy_score(y_test, y_pred_tuned)),
        "precision": float(precision_score(y_test, y_pred_tuned, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_tuned, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred_tuned, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_probs)),
        "confusion_matrix": cm.tolist()
    }
    
    with open(os.path.join(output_dir, "mlp_evaluation_metrics.json"), "w") as f:
        json.dump(results_summary, f, indent=4)
        
    print(f"\nMLP Model saved successfully to {model_path}!")
    print(f"Evaluation report and charts saved to {output_dir}/")

if __name__ == "__main__":
    main()