"""Phase 2 ablation study: what is the environmental signal actually
worth once accident-history features are removed?

Compares three configurations, each cross-validated with the same
protocol:

  A. baseline_env      -- Phase 1 feature set (raw environmental
                          averages only), full sample.
  B. enriched_env_only -- engineered environmental features (deltas,
                          monsoon share) but NO accident-history lags.
  C. enriched_full     -- everything including lag features (the v2
                          set). Smallest sample (lags eat rows), and
                          prev_total_accidents is expected to dominate.

The interesting comparison is A vs B (does feature engineering on the
environment help?) and B vs C (how much of C's performance is just
accident persistence?).

Run with:
    python scripts/run_ablation.py
"""
import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.engineering import (
    BASE_FEATURES,
    ENV_ONLY_FEATURES,
    MODEL_FEATURES,
    build_enriched_training_table,
)
from src.warehouse.db import get_engine

LABEL_ORDER = ["Low", "Medium", "High"]

CONFIGS = {
    "A_baseline_env": BASE_FEATURES,
    "B_enriched_env_only": ENV_ONLY_FEATURES,
    "C_enriched_full": MODEL_FEATURES,
}


def evaluate_config(df: pd.DataFrame, features: list) -> dict:
    usable = df.dropna(subset=features + ["risk_level"])
    X = usable[features].astype(float)
    y = usable["risk_level"]

    n_splits = min(5, y.value_counts().min())
    cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=42)

    label_map = {label: i for i, label in enumerate(LABEL_ORDER)}

    models = {
        "logreg": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
        "rf": RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=2, random_state=42),
        "xgb": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                             eval_metric="mlogloss", random_state=42),
    }

    row = {"n_rows": len(usable), "n_features": len(features)}
    for name, model in models.items():
        target = y.map(label_map) if name == "xgb" else y
        preds = cross_val_predict(model, X, target, cv=cv)
        if name == "xgb":
            inv = {v: k for k, v in label_map.items()}
            preds = pd.Series(preds).map(inv).values
        report = classification_report(y, preds, output_dict=True, zero_division=0)
        row[f"{name}_acc"] = round(report["accuracy"], 3)
        row[f"{name}_f1"] = round(report["macro avg"]["f1-score"], 3)
    return row


def main():
    engine = get_engine()
    print("Building enriched training table...")
    df = build_enriched_training_table(engine)
    df["risk_level"] = pd.qcut(df["total_accidents"], q=3, labels=LABEL_ORDER)

    results = {}
    for config_name, features in CONFIGS.items():
        print(f"Evaluating {config_name} ({len(features)} features)...")
        results[config_name] = evaluate_config(df, features)

    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    summary = pd.DataFrame(results).T
    print(summary.to_string())
    print(
        "\nReading guide: A vs B isolates the value of engineered environmental\n"
        "features; B vs C isolates how much performance comes purely from\n"
        "accident-history persistence. Mind the differing n_rows -- accuracies\n"
        "across configs aren't strictly comparable when samples differ."
    )


if __name__ == "__main__":
    main()