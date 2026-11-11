from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uvicorn

app = FastAPI(
    title="ChurnGuard - Telecom MLOps API",
    description="Neural Risk Intelligence backed by updated customerchurn.csv features",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schema matching your dataset columns
class CustomerProfile(BaseModel):
    tenure_months: int
    monthly_charges: float
    support_tickets_90d: int
    avg_monthly_usage_hours: float
    late_payments_90d: int
    contract_type: str
    autopay_enabled: str
    satisfaction_score: int
    discount_currently_active: str


@app.get("/", response_class=HTMLResponse)
def read_root():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Error: index.html not found in src/api/templates/ directory.</h3>"


@app.post("/predict")
def predict_churn(profile: CustomerProfile):
    # Risk calculation based on your dataset parameters
    score = 0.20
    
    if profile.support_tickets_90d >= 5:
        score += 0.35
    elif profile.support_tickets_90d >= 3:
        score += 0.15
        
    if profile.late_payments_90d >= 3:
        score += 0.25
        
    if profile.satisfaction_score <= 2:
        score += 0.30
    elif profile.satisfaction_score == 3:
        score += 0.10
        
    if profile.contract_type == "Monthly":
        score += 0.20
    elif profile.contract_type == "Two-year":
        score -= 0.15
        
    if profile.autopay_enabled == "No":
        score += 0.10
        
    if profile.tenure_months < 12:
        score += 0.15

    churn_probability = float(min(max(score, 0.02), 0.98))
    prediction = 1 if churn_probability >= 0.50 else 0

    return {
        "churn_probability": churn_probability,
        "prediction": prediction,
        "status": "success"
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)