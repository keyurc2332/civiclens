"""Populate the analytics star schema (dim_city, dim_date,
fact_accident_month, fact_environment_month) for the 6 dashboard
cities. This is what the Streamlit dashboard queries directly --
kept separate from the full-state training data used for modeling
(see train_baseline_model.py), per config.yaml's cities list being
the dashboard's scope, not the model's.

Run with:
    python scripts/build_analytics.py
"""

import os
import sys

import pandas as pd
import yaml
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.warehouse.db import get_engine

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_dim_city(engine, cities: list) -> dict:
    """Upsert dim_city, return {city_name: city_id} map."""
    with engine.begin() as conn:
        for c in cities:
            conn.execute(
                text("""
                    INSERT INTO analytics.dim_city (city_name, state, latitude, longitude)
                    VALUES (:name, :state, :lat, :lon)
                    ON CONFLICT (city_name) DO UPDATE SET
                        state = EXCLUDED.state, latitude = EXCLUDED.latitude, longitude = EXCLUDED.longitude
                    """),
                {"name": c["name"], "state": c["state"], "lat": c["lat"], "lon": c["lon"]},
            )
        result = conn.execute(text("SELECT city_id, city_name FROM analytics.dim_city"))
        return {row.city_name: row.city_id for row in result}


def build_dim_date(engine, years: range) -> dict:
    """Upsert one dim_date row per year (annual grain, since that's
    what the accident data actually supports) and one per year-month
    (for the environment fact table's finer grain). Returns
    {label: date_id} for both.
    """
    rows = []
    for y in years:
        rows.append(
            {
                "date_id": y * 100,
                "year": y,
                "month": None,
                "quarter": None,
                "is_monsoon": None,
                "label": str(y),
            }
        )
        for m in range(1, 13):
            rows.append(
                {
                    "date_id": y * 100 + m,
                    "year": y,
                    "month": m,
                    "quarter": (m - 1) // 3 + 1,
                    "is_monsoon": m in (6, 7, 8, 9),
                    "label": f"{y}-{m:02d}",
                }
            )

    with engine.begin() as conn:
        for r in rows:
            conn.execute(
                text("""
                    INSERT INTO analytics.dim_date (date_id, year, month, quarter, is_monsoon, label)
                    VALUES (:date_id, :year, :month, :quarter, :is_monsoon, :label)
                    ON CONFLICT (date_id) DO NOTHING
                    """),
                r,
            )
        result = conn.execute(text("SELECT date_id, label FROM analytics.dim_date"))
        return {row.label: row.date_id for row in result}


def build_fact_accident(engine, city_ids: dict, date_ids: dict, cities: list) -> None:
    """State-year accident facts, keyed by each dashboard city's
    city_id (multiple cities sharing a state will share these values
    -- documented in the schema itself via the `grain` column)."""
    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT state, year, total_accidents FROM clean.accidents"), conn)

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM analytics.fact_accident_month WHERE city_id = ANY(:ids)"),
            {"ids": list(city_ids.values())},
        )
        for c in cities:
            city_id = city_ids[c["name"]]
            state_rows = df[df["state"] == c["state"]]
            for _, row in state_rows.iterrows():
                date_id = date_ids.get(str(int(row["year"])))
                if date_id is None:
                    continue
                conn.execute(
                    text("""
                        INSERT INTO analytics.fact_accident_month
                            (city_id, date_id, grain, total_accidents, fatalities, injuries)
                        VALUES (:city_id, :date_id, 'state_year', :total, NULL, NULL)
                        """),
                    {
                        "city_id": city_id,
                        "date_id": date_id,
                        "total": None if pd.isna(row["total_accidents"]) else int(row["total_accidents"]),
                    },
                )
    print("Loaded analytics.fact_accident_month")


def build_fact_environment(engine, city_ids: dict, date_ids: dict) -> None:
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

    merged = pd.merge(aq, wx, on=["city", "observed_date"], how="outer")
    merged["observed_date"] = pd.to_datetime(merged["observed_date"])
    merged["year"] = merged["observed_date"].dt.year
    merged["month"] = merged["observed_date"].dt.month

    monthly = (
        merged.groupby(["city", "year", "month"])
        .agg(
            avg_pm25=("avg_pm25", "mean"),
            avg_pm10=("avg_pm10", "mean"),
            total_rainfall_mm=("rainfall_mm", "sum"),
            avg_temp_c=("avg_temp_c", "mean"),
            avg_humidity=("avg_humidity", "mean"),
            avg_wind_kmh=("avg_wind_kmh", "mean"),
        )
        .reset_index()
    )

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM analytics.fact_environment_month WHERE city_id = ANY(:ids)"),
            {"ids": list(city_ids.values())},
        )
        for _, row in monthly.iterrows():
            city_id = city_ids.get(row["city"])
            if city_id is None:
                continue  # not one of our 6 dashboard cities
            date_id = date_ids.get(f"{int(row['year'])}-{int(row['month']):02d}")
            if date_id is None:
                continue
            conn.execute(
                text("""
                    INSERT INTO analytics.fact_environment_month
                        (city_id, date_id, avg_pm25, avg_pm10, total_rainfall_mm, avg_temp_c, avg_humidity, avg_wind_kmh)
                    VALUES (:city_id, :date_id, :pm25, :pm10, :rain, :temp, :hum, :wind)
                    """),
                {
                    "city_id": city_id,
                    "date_id": date_id,
                    "pm25": None if pd.isna(row["avg_pm25"]) else float(row["avg_pm25"]),
                    "pm10": None if pd.isna(row["avg_pm10"]) else float(row["avg_pm10"]),
                    "rain": None if pd.isna(row["total_rainfall_mm"]) else float(row["total_rainfall_mm"]),
                    "temp": None if pd.isna(row["avg_temp_c"]) else float(row["avg_temp_c"]),
                    "hum": None if pd.isna(row["avg_humidity"]) else float(row["avg_humidity"]),
                    "wind": None if pd.isna(row["avg_wind_kmh"]) else float(row["avg_wind_kmh"]),
                },
            )
    print(f"Loaded {len(monthly)} rows into analytics.fact_environment_month")


def main():
    config = load_config()
    cities = config["cities"]
    engine = get_engine()

    print("Building dim_city...")
    city_ids = build_dim_city(engine, cities)

    print("Building dim_date...")
    date_ids = build_dim_date(engine, range(2010, 2027))

    print("Building fact_accident_month...")
    build_fact_accident(engine, city_ids, date_ids, cities)

    print("Building fact_environment_month...")
    build_fact_environment(engine, city_ids, date_ids)

    print("Done.")


if __name__ == "__main__":
    main()
