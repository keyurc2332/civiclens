"""CivicLens REST API. Exposes the analytics layer (cities, accident
history, environmental data, risk predictions with SHAP explanations)
as JSON endpoints.

Run locally with:
    uvicorn src.api.main:app --reload --port 8000

Then browse the auto-generated interactive docs at:
    http://localhost:8000/docs

Design notes:
- Read-only API over the analytics.* star schema -- no ingestion or
  training endpoints. Writes happen through the pipeline scripts.
- Same decoupling rule as the dashboard: this reads ONLY analytics.*,
  never raw/clean, so ingestion internals can change freely.
- Model version is a query parameter (?model_version=enriched_v2)
  because both v1 and v2 predictions coexist in fact_risk_prediction.
"""
import json

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import text

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.warehouse.db import get_engine

app = FastAPI(
    title="CivicLens API",
    description="Road accident risk intelligence for major Indian cities — public data, honestly handled.",
    version="1.0.0",
)

DEFAULT_MODEL_VERSION = "enriched_v2"


def _df(query: str, params: dict | None = None) -> pd.DataFrame:
    engine = get_engine()
    with engine.begin() as conn:
        return pd.read_sql(text(query), conn, params=params)


@app.get("/health")
def health():
    """Liveness check, verifies DB connectivity."""
    try:
        _df("SELECT 1 AS ok")
        return {"status": "ok"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unreachable: {exc}")


@app.get("/cities")
def list_cities():
    """All dashboard cities with their state and coordinates."""
    df = _df("SELECT city_name, state, latitude, longitude FROM analytics.dim_city ORDER BY city_name")
    return df.to_dict(orient="records")


@app.get("/cities/{city_name}/accidents")
def city_accidents(city_name: str):
    """Annual accident history for a city (state-grain figures --
    cities sharing a state share values; documented source limitation)."""
    df = _df(
        """
        SELECT dc.city_name, dd.year, fa.grain, fa.total_accidents
        FROM analytics.fact_accident_month fa
        JOIN analytics.dim_city dc ON dc.city_id = fa.city_id
        JOIN analytics.dim_date dd ON dd.date_id = fa.date_id
        WHERE LOWER(dc.city_name) = LOWER(:city) AND dd.month IS NULL
        ORDER BY dd.year
        """,
        {"city": city_name},
    )
    if df.empty:
        raise HTTPException(status_code=404, detail=f"city '{city_name}' not found")
    return df.to_dict(orient="records")


@app.get("/cities/{city_name}/environment")
def city_environment(
    city_name: str,
    year: int | None = Query(default=None, description="Filter to a single year"),
):
    """Monthly environmental aggregates (PM2.5, PM10, temperature,
    rainfall, humidity, wind) for a city."""
    query = """
        SELECT dc.city_name, dd.year, dd.month, fe.avg_pm25, fe.avg_pm10,
               fe.avg_temp_c, fe.total_rainfall_mm, fe.avg_humidity, fe.avg_wind_kmh
        FROM analytics.fact_environment_month fe
        JOIN analytics.dim_city dc ON dc.city_id = fe.city_id
        JOIN analytics.dim_date dd ON dd.date_id = fe.date_id
        WHERE LOWER(dc.city_name) = LOWER(:city) AND dd.month IS NOT NULL
    """
    params = {"city": city_name}
    if year is not None:
        query += " AND dd.year = :year"
        params["year"] = year
    query += " ORDER BY dd.year, dd.month"

    df = _df(query, params)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"no environment data for '{city_name}'")
    return df.where(pd.notna(df), None).to_dict(orient="records")


@app.get("/predictions")
def all_predictions(
    model_version: str = Query(default=DEFAULT_MODEL_VERSION),
):
    """Latest risk prediction for every city, for a given model version."""
    df = _df(
        """
        SELECT DISTINCT ON (dc.city_name)
               dc.city_name, dd.year, fp.model_version, fp.risk_level,
               fp.confidence, fp.top_features, fp.predicted_at
        FROM analytics.fact_risk_prediction fp
        JOIN analytics.dim_city dc ON dc.city_id = fp.city_id
        JOIN analytics.dim_date dd ON dd.date_id = fp.date_id
        WHERE fp.model_version = :version
        ORDER BY dc.city_name, dd.year DESC, fp.predicted_at DESC
        """,
        {"version": model_version},
    )
    if df.empty:
        raise HTTPException(status_code=404, detail=f"no predictions for model_version '{model_version}'")

    records = df.to_dict(orient="records")
    for r in records:
        if isinstance(r["top_features"], str):
            r["top_features"] = json.loads(r["top_features"])
        r["predicted_at"] = str(r["predicted_at"])
    return records


@app.get("/predictions/{city_name}")
def city_prediction(
    city_name: str,
    model_version: str = Query(default=DEFAULT_MODEL_VERSION),
):
    """Latest risk prediction for one city, including the SHAP-based
    'why' explanation (per-prediction for enriched_v2, global
    importance for baseline_v1)."""
    df = _df(
        """
        SELECT dc.city_name, dd.year, fp.model_version, fp.risk_level,
               fp.confidence, fp.top_features, fp.predicted_at
        FROM analytics.fact_risk_prediction fp
        JOIN analytics.dim_city dc ON dc.city_id = fp.city_id
        JOIN analytics.dim_date dd ON dd.date_id = fp.date_id
        WHERE LOWER(dc.city_name) = LOWER(:city) AND fp.model_version = :version
        ORDER BY dd.year DESC, fp.predicted_at DESC
        LIMIT 1
        """,
        {"city": city_name, "version": model_version},
    )
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"no prediction for city '{city_name}' with model_version '{model_version}'",
        )

    record = df.iloc[0].to_dict()
    if isinstance(record["top_features"], str):
        record["top_features"] = json.loads(record["top_features"])
    record["predicted_at"] = str(record["predicted_at"])
    record["caveat"] = (
        "Model trained on state-year grain public data with a small sample; "
        "confidences are not calibrated probabilities. See project README for "
        "the full ablation analysis and limitations."
    )
    return record


@app.get("/models")
def list_models():
    """Available model versions and how many predictions each has."""
    df = _df(
        """
        SELECT model_version, COUNT(*) AS n_predictions,
               MIN(predicted_at) AS first_predicted, MAX(predicted_at) AS last_predicted
        FROM analytics.fact_risk_prediction
        GROUP BY model_version
        ORDER BY model_version
        """
    )
    records = df.to_dict(orient="records")
    for r in records:
        r["first_predicted"] = str(r["first_predicted"])
        r["last_predicted"] = str(r["last_predicted"])
    return records