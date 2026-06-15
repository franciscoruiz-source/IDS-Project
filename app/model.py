import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

from features import generate_features, get_feature_names


MODEL_PATH = "model.pkl"
# These are the features selected after running all 4 selection methods
# (see notebook for full analysis — these ranked highly across all methods)
SELECTED_FEATURES = [
    "follower_ratio",
    "repos_per_year",
    "account_age_years",
    "has_no_repos",
]


def train_model(csv_path: str = "../data/raw/github_users_labeled.csv"):
    """
    Load labeled data, generate features, train a Random Forest,
    and save the model to disk.
    """
    print("Loading data...")
    df_raw = pd.read_csv(csv_path)
    print(f"  {len(df_raw)} records loaded")

    print("\nGenerating features...")
    df_features = generate_features(df_raw)
    print(df_features.describe())

    X = df_features[SELECTED_FEATURES]
    y = df_features["churned"]

    print(f"\nClass balance: {y.value_counts().to_dict()}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest (class_weight handles imbalance)
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nTest Set Results:")
    print(classification_report(y_test, y_pred))

    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    print(f"5-Fold CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    return model


def load_model():
    """Load the saved model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Run model.py or the notebook to train first."
        )
    return joblib.load(MODEL_PATH)


def predict(features_dict: dict):
    """
    Given a dict of feature values, return churn prediction.
    features_dict keys must match SELECTED_FEATURES.
    """
    model = load_model()
    values = [[features_dict[f] for f in SELECTED_FEATURES]]
    X = pd.DataFrame(values, columns=SELECTED_FEATURES)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return bool(pred), round(float(prob), 3)


if __name__ == "__main__":
    train_model()
