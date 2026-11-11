ChurnGuard synthetic customer dataset (2,500 rows).
Target: churned_90d (0/1). All records are simulated for training and are not real customer data.
Suggested workflow: data validation -> EDA -> train/validation/test split -> baseline -> feature engineering -> Logistic Regression/Random Forest/Gradient Boosting -> ANN comparison -> threshold tuning -> MLflow tracking -> FastAPI scoring -> LLM-generated retention offer drafts.
Do not use generated offers without human review. Avoid including personal or sensitive data in prompts.
