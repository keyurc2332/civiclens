"""Phase 3 anomaly detection: flag city-months where environmental
conditions deviated sharply from that city's own historical pattern
FOR THAT CALENDAR MONTH.

Method: per (city, calendar_month, metric), compute the historical
mean and std across all years, then z-score each observation against
it. |z| > 2.5 is flagged. Seasonal awareness matters: 35C in Delhi in
May is normal; in January it would be a major anomaly. Plain z-scores
against the annual distribution would mostly just rediscover seasons.

Deliberately simple statistics -- no isolation forests or autoencoders.
With ~13 years of monthly data per city, transparent z-scores are
robust, explainable, and appropriate; fancier methods would add
opacity without adding trust.

Results are written to analytics.anomalies (created by this script if
missing) and surfaced in the dashboard.

Run with:
    python scripts/detect_anomalies.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.warehouse.db import get_engine

Z_THRESHOLD = 2.5
MIN_HISTORY = 5  # need at least this many observations of a (city, month) to z-score it

METRICS = ["avg_pm25", "avg_pm10", "avg_temp_c", "total_rainfall_mm", "avg_humidity", "avg_wind_kmh"]

# Practical-significance floors: an observation must ALSO deviate from
# the historical mean by at least this much in absolute terms.
# Why: z-scores break down when historical variance is near zero --
# e.g. Mumbai dry-season months where rainfall is ~always 0mm produce
# z > 3 for a 0.1mm drizzle. Statistically extreme, practically
# meaningless. These floors keep only anomalies a human would agree are
# anomalies.
MIN_ABS_DEVIATION = {
    "avg_pm25": 15.0,          # ug/m3
    "avg_pm10": 25.0,          # ug/m3
    "avg_temp_c": 2.0,         # degrees C
    "total_rainfall_mm": 10.0, # mm over the month
    "avg_humidity": 10.0,      # percentage points
    "avg_wind_kmh": 3.0,       # km/h
}

DDL = """
CREATE TABLE IF NOT EXISTS analytics.anomalies (
    anomaly_id      BIGSERIAL PRIMARY KEY,
    city_id         INT NOT NULL REFERENCES analytics.dim_city(city_id),
    date_id         INT NOT NULL REFERENCES analytics.dim_date(date_id),
    metric          TEXT NOT NULL,
    observed_value  NUMERIC,
    historical_mean NUMERIC,
    historical_std  NUMERIC,
    z_score         NUMERIC,
    direction       TEXT CHECK (direction IN ('above','below')),
    detected_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (city_id, date_id, metric)
);
"""


def main():
    engine = get_engine()

    with engine.begin() as conn:
        conn.exec_driver_sql(DDL)

    with engine.begin() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT fe.city_id, dc.city_name, dd.date_id, dd.year, dd.month,
                       fe.avg_pm25, fe.avg_pm10, fe.avg_temp_c,
                       fe.total_rainfall_mm, fe.avg_humidity, fe.avg_wind_kmh
                FROM analytics.fact_environment_month fe
                JOIN analytics.dim_city dc ON dc.city_id = fe.city_id
                JOIN analytics.dim_date dd ON dd.date_id = fe.date_id
                WHERE dd.month IS NOT NULL
                """
            ),
            conn,
        )

    print(f"Loaded {len(df)} city-month rows")

    anomalies = []
    for (city_id, city_name, month), group in df.groupby(["city_id", "city_name", "month"]):
        for metric in METRICS:
            series = group[["date_id", "year", metric]].dropna()
            if len(series) < MIN_HISTORY:
                continue
            mean, std = series[metric].mean(), series[metric].std()
            if std == 0 or pd.isna(std):
                continue
            for _, row in series.iterrows():
                z = (row[metric] - mean) / std
                abs_dev = abs(row[metric] - mean)
                if abs(z) > Z_THRESHOLD and abs_dev >= MIN_ABS_DEVIATION[metric]:
                    anomalies.append({
                        "city_id": int(city_id),
                        "city_name": city_name,
                        "date_id": int(row["date_id"]),
                        "year": int(row["year"]),
                        "month": int(month),
                        "metric": metric,
                        "observed_value": float(row[metric]),
                        "historical_mean": float(mean),
                        "historical_std": float(std),
                        "z_score": float(z),
                        "direction": "above" if z > 0 else "below",
                    })

    print(f"Detected {len(anomalies)} anomalies (|z| > {Z_THRESHOLD}, "
          f"min {MIN_HISTORY} years of history per city-month)")

    if not anomalies:
        print("Nothing to write.")
        return

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM analytics.anomalies"))
        conn.execute(
            text(
                """
                INSERT INTO analytics.anomalies
                    (city_id, date_id, metric, observed_value, historical_mean,
                     historical_std, z_score, direction)
                VALUES
                    (:city_id, :date_id, :metric, :observed_value, :historical_mean,
                     :historical_std, :z_score, :direction)
                """
            ),
            [{k: v for k, v in a.items() if k not in ("city_name", "year", "month")} for a in anomalies],
        )

    # Print a readable summary of the most extreme ones
    top = sorted(anomalies, key=lambda a: -abs(a["z_score"]))[:10]
    print("\nTop 10 most extreme anomalies:")
    for a in top:
        print(f"  {a['city_name']} {a['year']}-{a['month']:02d}: {a['metric']} = "
              f"{a['observed_value']:.1f} (historical mean {a['historical_mean']:.1f}, "
              f"z = {a['z_score']:+.1f}, {a['direction']})")

    print("\nDone. Anomalies written to analytics.anomalies.")


if __name__ == "__main__":
    main()