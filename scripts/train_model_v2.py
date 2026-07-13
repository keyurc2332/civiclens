"""Phase 2 model training: enriched features + three-model comparison
(logistic regression, random forest, XGBoost), cross-validated, with
per-prediction SHAP for the dashboard.

Run with:
    python scripts/train_model_v2.py

Key differences from Phase 1 (train_baseline_model.py):
- Uses engineered features from src/features/engineering.py (lags,
  deltas, monsoon share) -- see that module's docstring for the
  leakage reasoning around fatality_rate.
- Adds XGBoost to the comparison.
- Saves PER-PREDICTION SHAP values for each dashboard city, not just
  global importance -- the dashboard's "why is this city High risk"
  explanations come from these.
- Rows with missing lag features (each state's first year, 2021) are
  dropped -- documented sample-size cost of using lag features.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.engineering import MODEL_FEATURES, build_enriched_training_table
from src.warehouse.db import get_engine

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_VERSION = "enriched_v2"

LABEL_ORDER = ["Low", "Medium", "High"]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def add_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_level"] = pd.qcut(df["total_accidents"], q=3, labels=LABEL_ORDER)
    return df


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    n_splits = min(5, y.value_counts().min())
    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)

    models = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=2, random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            eval_metric="mlogloss", random_state=42,
        ),
    }

    results = {}
    # XGBoost needs numeric labels
    label_map = {label: i for i, label in enumerate(LABEL_ORDER)}
    y_numeric = y.map(label_map)

    for name, model in models.items():
        target = y_numeric if name == "xgboost" else y
        preds = cross_val_predict(model, X, target, cv=cv)
        if name == "xgboost":
            inv_map = {v: k for k, v in label_map.items()}
            preds = pd.Series(preds).map(inv_map).values
        report = classification_report(y, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, preds, labels=LABEL_ORDER)
        results[name] = {
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
        }
        print(f"\n=== {name} (cross-validated, {cv.get_n_splits()} folds) ===")
        print(classification_report(y, preds, zero_division=0))
        print("Confusion matrix [rows=actual, cols=predicted], order Low/Medium/High:")
        print(cm)

    print("\n--- Model comparison summary ---")
    comparison = pd.DataFrame(results).T[["accuracy", "macro_f1"]].round(3)
    print(comparison)
    return results


def compute_per_prediction_shap(model, X_row: pd.DataFrame, pred_class_idx: int) -> list:
    """SHAP values for ONE prediction, for the predicted class --
    this is what makes the dashboard's per-city 'why' explanation.

    Uses XGBoost's NATIVE SHAP computation (pred_contribs=True) rather
    than the shap library's TreeExplainer, because installed shap
    versions can fail to parse newer XGBoost model formats (multiclass
    base_score arrays). XGBoost's own implementation is the same
    TreeSHAP algorithm, minus the version-compatibility headache.
    """
    import xgboost as xgb

    dmatrix = xgb.DMatrix(X_row)
    # Shape for multiclass: (n_rows, n_classes, n_features + 1);
    # last column is the bias term, which we drop.
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
    arr = np.asarray(contribs)
    if arr.ndim == 3:
        class_shap = arr[0, pred_class_idx, :-1]
    else:
        class_shap = arr[0, :-1]

    pairs = sorted(zip(X_row.columns, class_shap), key=lambda x: -abs(x[1]))
    return [{"feature": f, "shap_value": round(float(v), 4)} for f, v in pairs]


def save_predictions_for_dashboard(engine, model, training: pd.DataFrame, config: dict) -> None:
    with engine.begin() as conn:
        city_ids = {r.city_name: r.city_id for r in conn.execute(text("SELECT city_id, city_name FROM analytics.dim_city"))}
        date_ids = {r.label: r.date_id for r in conn.execute(text("SELECT date_id, label FROM analytics.dim_date"))}
        # Clear previous predictions from this model version for idempotent re-runs
        conn.execute(text("DELETE FROM analytics.fact_risk_prediction WHERE model_version = :v"),
                     {"v": MODEL_VERSION})

    for c in config["cities"]:
        state_rows = training[training["state"] == c["state"]].dropna(subset=MODEL_FEATURES)
        if state_rows.empty:
            print(f"  No usable feature rows for {c['name']} ({c['state']}) -- skipping")
            continue
        latest = state_rows.sort_values("year").iloc[-1]
        X_row = latest[MODEL_FEATURES].to_frame().T.astype(float)

        proba = model.predict_proba(X_row)[0]
        pred_idx = int(np.argmax(proba))
        risk_level = LABEL_ORDER[pred_idx]
        confidence = float(proba[pred_idx])

        top_features = compute_per_prediction_shap(model, X_row, pred_idx)

        city_id = city_ids.get(c["name"])
        date_id = date_ids.get(str(int(latest["year"])))
        if city_id is None or date_id is None:
            continue

        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO analytics.fact_risk_prediction
                        (city_id, date_id, model_version, risk_level, confidence, top_features)
                    VALUES (:city_id, :date_id, :model_version, :risk_level, :confidence, :top_features)
                    """
                ),
                {
                    "city_id": city_id, "date_id": date_id, "model_version": MODEL_VERSION,
                    "risk_level": risk_level, "confidence": confidence,
                    "top_features": json.dumps(top_features),
                },
            )
        print(f"  {c['name']}: {risk_level} ({confidence:.0%}) -- top driver: {top_features[0]['feature']}")

    print("Saved per-prediction SHAP explanations to analytics.fact_risk_prediction")


def main():
    config = load_config()
    engine = get_engine()

    print("Building enriched training table...")
    training = build_enriched_training_table(engine)
    labeled = add_risk_labels(training)

    usable = labeled.dropna(subset=MODEL_FEATURES + ["risk_level"])
    dropped = len(labeled) - len(usable)
    print(f"Training rows: {len(usable)} (dropped {dropped} rows with missing lag/features "
          f"-- mostly each state's first year, a documented cost of lag features)")
    print("\nRisk label distribution:")
    print(usable["risk_level"].value_counts())

    X = usable[MODEL_FEATURES].astype(float)
    y = usable["risk_level"]

    train_and_evaluate(X, y)

    print("\nFitting final XGBoost on full data for dashboard predictions...")
    label_map = {label: i for i, label in enumerate(LABEL_ORDER)}
    final_model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        eval_metric="mlogloss", random_state=42,
    )
    final_model.fit(X, y.map(label_map))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(final_model, os.path.join(MODELS_DIR, f"{MODEL_VERSION}.joblib"))
    print(f"Saved model to models/{MODEL_VERSION}.joblib")

    save_predictions_for_dashboard(engine, final_model, labeled, config)


if __name__ == "__main__":
    main()