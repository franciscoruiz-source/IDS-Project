from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Feature names must match what the model was trained on
SELECTED_FEATURES = [
    "follower_ratio",
    "repos_per_year",
    "account_age_years",
    "has_no_repos",
]

MODEL_PATH = "model.pkl"

app = FastAPI(
    title="Customer Churn Predictor",
    description="Predicts whether a GitHub user will churn based on behavioral features.",
    version="1.0.0",
)

# Load model once at startup (not on every request)
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"WARNING: {MODEL_PATH} not found. Train the model first.")


# --- Input schema (Pydantic validates types automatically) ---
class UserFeatures(BaseModel):
    follower_ratio: float       # followers / (following + 1)
    repos_per_year: float       # public_repos / account_age_years
    account_age_years: float    # How long the account has existed
    has_no_repos: int           # 1 if user has 0 repos, else 0

    class Config:
        json_schema_extra = {
            "example": {
                "follower_ratio": 0.3,
                "repos_per_year": 1.2,
                "account_age_years": 4.5,
                "has_no_repos": 0,
            }
        }


# --- Endpoints ---

@app.get("/health")
def health():
    """Health check — used by Docker and cloud platforms."""
    return {"status": "ok"}


@app.get("/features")
def list_features():
    """Returns the list of required input features and their descriptions."""
    return {
        "features": [
            {"name": "follower_ratio",     "type": "float", "description": "followers / (following + 1) — social engagement signal"},
            {"name": "repos_per_year",     "type": "float", "description": "public_repos / account_age_years — productivity rate"},
            {"name": "account_age_years",  "type": "float", "description": "Account age in years since creation"},
            {"name": "has_no_repos",       "type": "int",   "description": "1 if user has zero public repos, else 0"},
        ]
    }

@app.get("/")
def root():
    return {
        "message": "Customer Churn Predictor API - Introduction to Data Science",
        "endpoints": {
            "health": "GET  /health",
            "predict": "POST /predict",
            "docs": "GET  /docs"

        },
        "usage": "Visit /docs to test the prediction endpoint interactively"
    }
@app.post("/predict")
def predict_churn(user: UserFeatures):
    """
    Accepts user behavioral features and returns a churn prediction.
    - churned: true/false boolean
    - churn_probability: float between 0.0 and 1.0
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first (run model.py)."
        )

    features = np.array([[
        user.follower_ratio,
        user.repos_per_year,
        user.account_age_years,
        user.has_no_repos,
    ]])

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    return {
        "churned": bool(pred),
        "churn_probability": round(float(prob), 3),
        "risk_level": (
            "High" if prob >= 0.7
            else "Medium" if prob >= 0.4
            else "Low"
        ),
    }
