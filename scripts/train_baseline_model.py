"""Phase 1 baseline model: state-year accident risk classification
(Low / Medium / High).

Training table: ALL ~36 states (not just the 6 dashboard cities) --
see scripts/ingest_accidents.py and ingest_air_quality.py for why.
City -> state mapping for the environmental data comes from
data_climate/stations_info.csv.

Target definition: total_accidents bucketed into tertiles across the
whole state-year dataset. CAVEAT (documented deliberately, not hidden):
this partly reflects state size/population rather than pure "riskiness"
-- a large state will tend toward "High" regardless of environmental
conditions. Normalizing by population or road length would fix this;
out of scope for Phase 1, noted as a Phase 2 improvement.

Run with:
    python scripts/train_baseline_model.py
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.warehouse.db import get_engine

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
STATIONS_INFO_PATH = os.path.join(os.path.dirname(__file__), "..", "data_climate", "stations_info.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_VERSION = "baseline_v1"

FEATURE_COLS = ["avg_pm25", "avg_pm10", "avg_temp_c", "total_rainfall_mm", "avg_humidity", "avg_wind_kmh"]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_city_state_map() -> pd.DataFrame:
    stations = pd.read_csv(STATIONS_INFO_PATH)
    return stations[["city", "state"]].drop_duplicates()


def build_training_table(engine) -> pd.DataFrame:
    city_state = build_city_state_map()

    with engine.begin() as conn:
        aq = pd.read_sql(
            text("SELECT city, observed_date, avg_pm25, avg_pm10 FROM clean.air_quality_daily"), conn
        )
        wx = pd.read_sql(
            text(
                "SELECT city, observed_date, avg_temp_c, rainfall_mm, avg_humidity, avg_wind_kmh "
                "FROM clean.weather_daily"
            ),
            conn,
        )
        accidents = pd.read_sql(text("SELECT state, year, total_accidents FROM clean.accidents"), conn)

    env = pd.merge(aq, wx, on=["city", "observed_date"], how="outer")
    env = pd.merge(env, city_state, on="city", how="left")
    env["year"] = pd.to_datetime(env["observed_date"]).dt.year

    state_year_env = (
        env.groupby(["state", "year"])
        .agg(
            avg_pm25=("avg_pm25", "mean"),
            avg_pm10=("avg_pm10", "mean"),
            avg_temp_c=("avg_temp_c", "mean"),
            total_rainfall_mm=("rainfall_mm", "sum"),
            avg_humidity=("avg_humidity", "mean"),
            avg_wind_kmh=("avg_wind_kmh", "mean"),
            n_cities=("city", "nunique"),
        )
        .reset_index()
    )

    training = pd.merge(accidents, state_year_env, on=["state", "year"], how="inner")
    training = training.dropna(subset=FEATURE_COLS + ["total_accidents"])
    return training


def add_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_level"] = pd.qcut(df["total_accidents"], q=3, labels=["Low", "Medium", "High"])
    return df


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    """Cross-validated comparison of two baseline models. With a small
    sample (a few dozen state-years), a single train/test split would
    be noisy and unreliable -- stratified k-fold cross-validation gives
    a much more honest read on real-world performance.
    """
    n_splits = min(5, y.value_counts().min())  # can't have more folds than the smallest class
    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)

    models = {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=2, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        preds = cross_val_predict(model, X, y, cv=cv)
        report = classification_report(y, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, preds, labels=["Low", "Medium", "High"])
        results[name] = {"report": report, "confusion_matrix": cm.tolist(), "predictions": preds}
        print(f"\n=== {name} (cross-validated, {cv.get_n_splits()} folds) ===")
        print(classification_report(y, preds, zero_division=0))
        print("Confusion matrix [rows=actual, cols=predicted], order Low/Medium/High:")
        print(cm)

    return results


def compute_shap_summary(model, X: pd.DataFrame) -> list:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # SHAP's multiclass output shape has changed across versions:
    # older versions return a list of (n_samples, n_features) arrays,
    # one per class; newer versions return a single
    # (n_samples, n_features, n_classes) array. Handle both.
    if isinstance(shap_values, list):
        per_class_abs = [np.abs(sv).mean(axis=0) for sv in shap_values]
        mean_abs = np.mean(per_class_abs, axis=0)
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            mean_abs = np.abs(arr).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(arr).mean(axis=0)

    ranked = sorted(zip(X.columns, mean_abs), key=lambda x: -x[1])
    return [{"feature": f, "mean_abs_shap": round(float(v), 4)} for f, v in ranked]


def save_predictions_for_dashboard(engine, model, feature_importance: list, config: dict) -> None:
    """Score the 6 dashboard cities' latest available state-year data
    and write results into analytics.fact_risk_prediction."""
    with engine.begin() as conn:
        city_ids = {
            r.city_name: r.city_id
            for r in conn.execute(text("SELECT city_id, city_name FROM analytics.dim_city"))
        }
        date_ids = {
            r.label: r.date_id for r in conn.execute(text("SELECT date_id, label FROM analytics.dim_date"))
        }

    training = build_training_table(engine)
    for c in config["cities"]:
        state_rows = training[training["state"] == c["state"]]
        if state_rows.empty:
            continue
        latest = state_rows.sort_values("year").iloc[-1]
        X_row = latest[FEATURE_COLS].to_frame().T
        proba = model.predict_proba(X_row)[0]
        classes = model.classes_
        pred_idx = int(np.argmax(proba))
        risk_level = classes[pred_idx]
        confidence = float(proba[pred_idx])

        city_id = city_ids.get(c["name"])
        date_id = date_ids.get(str(int(latest["year"])))
        if city_id is None or date_id is None:
            continue

        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO analytics.fact_risk_prediction
                        (city_id, date_id, model_version, risk_level, confidence, top_features)
                    VALUES (:city_id, :date_id, :model_version, :risk_level, :confidence, :top_features)
                    """),
                {
                    "city_id": city_id,
                    "date_id": date_id,
                    "model_version": MODEL_VERSION,
                    "risk_level": risk_level,
                    "confidence": confidence,
                    "top_features": json.dumps(feature_importance),
                },
            )
    print("Saved dashboard predictions into analytics.fact_risk_prediction")


def main():
    config = load_config()
    engine = get_engine()

    print("Building state-year training table...")
    training = build_training_table(engine)
    print(f"Training table: {len(training)} state-year rows across {training['state'].nunique()} states")

    labeled = add_risk_labels(training)
    print("\nRisk label distribution:")
    print(labeled["risk_level"].value_counts())

    X = labeled[FEATURE_COLS]
    y = labeled["risk_level"]

    results = train_and_evaluate(X, y)

    print("\nFitting final Random Forest on full data for explainability + dashboard predictions...")
    final_model = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=2, random_state=42)
    final_model.fit(X, y)

    feature_importance = compute_shap_summary(final_model, X)
    print("\nFeature importance (mean |SHAP value|, averaged across classes):")
    for f in feature_importance:
        print(f"  {f['feature']}: {f['mean_abs_shap']}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(final_model, os.path.join(MODELS_DIR, f"{MODEL_VERSION}.joblib"))
    print(f"\nSaved model to models/{MODEL_VERSION}.joblib")

    save_predictions_for_dashboard(engine, final_model, feature_importance, config)


if __name__ == "__main__":
    main()
