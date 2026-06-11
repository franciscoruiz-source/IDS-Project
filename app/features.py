import pandas as pd
import numpy as np
from datetime import datetime, timezone


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw GitHub API fields into meaningful ML features.
    Raw fields are NOT features — they are ingredients.

    Required categories (per assignment):
      - Ratio features (≥2)
      - Time-based features (≥2)
      - Aggregation features (≥2)
      - Binary/Categorical features (≥2)
    """
    df = df.copy()
    now = datetime.now(timezone.utc)

    # --- Parse dates ---
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)

    # -------------------------------------------------------
    # TIME-BASED FEATURES
    # Recency is the strongest churn signal in most domains.
    # -------------------------------------------------------

    # Time-based #1: How many days since last activity?
    df["days_inactive"] = (now - df["updated_at"]).dt.days

    # Time-based #2: How old is the account (in years)?
    df["account_age_years"] = (
        (now - df["created_at"]).dt.days / 365.25
    ).clip(lower=0.01)  # Avoid division by zero

    # -------------------------------------------------------
    # RATIO FEATURES
    # Normalize raw counts to make users comparable.
    # -------------------------------------------------------

    # Ratio #1: Social influence ratio — do people follow back?
    # A low ratio (many following, few followers) = passive/disengaged user
    df["follower_ratio"] = df["followers"] / (df["following"] + 1)

    # Ratio #2: Repos per year of account age — productivity rate
    df["repos_per_year"] = df["public_repos"] / df["account_age_years"]

    # -------------------------------------------------------
    # AGGREGATION FEATURES
    # Summarize volume/magnitude of user behavior.
    # -------------------------------------------------------

    # Aggregation #1: Total public contributions (repos + gists)
    df["total_public_contributions"] = df["public_repos"] + df["public_gists"]

    # Aggregation #2: Social size (total network connections)
    df["social_size"] = df["followers"] + df["following"]

    # -------------------------------------------------------
    # BINARY / CATEGORICAL FEATURES
    # Capture qualitative thresholds — zero is different from low.
    # -------------------------------------------------------

    # Binary #1: User with ZERO repos — may never have engaged meaningfully
    df["has_no_repos"] = (df["public_repos"] == 0).astype(int)

    # Binary #2: User with ZERO followers — socially isolated, higher churn risk
    df["has_no_followers"] = (df["followers"] == 0).astype(int)

    # Select only the generated feature columns (not raw fields)
    feature_cols = [
        "days_inactive",
        "account_age_years",
        "follower_ratio",
        "repos_per_year",
        "total_public_contributions",
        "social_size",
        "has_no_repos",
        "has_no_followers",
    ]

    # Keep churn label if present
    if "churned" in df.columns:
        feature_cols.append("churned")

    return df[feature_cols]


def get_feature_names() -> list:
    """Return the list of feature names used by the model (no label)."""
    return [
        "days_inactive",
        "account_age_years",
        "follower_ratio",
        "repos_per_year",
        "total_public_contributions",
        "social_size",
        "has_no_repos",
        "has_no_followers",
    ]


if __name__ == "__main__":
    # Quick test
    sample = pd.DataFrame([{
        "username": "test_user",
        "public_repos": 10,
        "public_gists": 2,
        "followers": 50,
        "following": 100,
        "created_at": "2019-01-01T00:00:00Z",
        "updated_at": "2022-06-01T00:00:00Z",
        "churned": 1,
    }])
    features = generate_features(sample)
    print(features.T)
