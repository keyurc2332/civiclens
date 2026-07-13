"""Phase 2 feature engineering. Builds an enriched state-year training
table on top of the base joins used in Phase 1, adding:

1. Severity metrics -- fatalities per accident, injuries per accident.
   These normalize by accident volume, making them (partially) robust
   to the state-size confound noted in the Phase 1 model docstring.
2. Lag features -- previous year's total accidents and fatality rate.
   Accident counts are strongly persistent year over year, so the lag
   is expected to be highly predictive; that's fine (and realistic),
   but it also means we should look at feature importances honestly:
   if the model mostly leans on the lag, the environmental features'
   marginal contribution is the real finding.
3. Environmental deltas -- year-over-year change in PM2.5 and
   temperature, capturing deterioration vs improvement rather than
   just absolute levels.
4. Monsoon rainfall share -- fraction of annual rainfall falling in
   June-September, a seasonality proxy that differs across states.

Used by train_model_v2.py. Kept as an importable module (src/features)
rather than a script, since both training and any future prediction
service need identical feature logic.
"""
import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.warehouse.db import get_engine

STATIONS_INFO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data_climate", "stations_info.csv"
)

BASE_FEATURES = ["avg_pm25", "avg_pm10", "avg_temp_c", "total_rainfall_mm", "avg_humidity", "avg_wind_kmh"]
ENGINEERED_FEATURES = [
    "fatality_rate",        # fatalities / total_accidents (same year -- see note below)
    "prev_total_accidents", # lag-1 accident count
    "prev_fatality_rate",   # lag-1 fatality rate
    "pm25_yoy_change",      # PM2.5 delta vs previous year
    "temp_yoy_change",      # temperature delta vs previous year
    "monsoon_rain_share",   # Jun-Sep rainfall / annual rainfall
]

# NOTE on leakage: fatality_rate for the SAME year as the target would be
# leakage if the target were accident counts (it's derived from them).
# Our target is the tertile bucket of total_accidents, and fatality_rate
# = fatalities/total_accidents uses the target in its denominator -- so
# same-year fatality_rate is EXCLUDED from the model feature list and
# only the lagged version (prev_fatality_rate) is used for training.
MODEL_FEATURES = BASE_FEATURES + [
    "prev_total_accidents",
    "prev_fatality_rate",
    "pm25_yoy_change",
    "temp_yoy_change",
    "monsoon_rain_share",
]


def build_city_state_map() -> pd.DataFrame:
    stations = pd.read_csv(STATIONS_INFO_PATH)
    return stations[["city", "state"]].drop_duplicates()


def build_enriched_training_table(engine=None) -> pd.DataFrame:
    """Returns one row per state-year with base + engineered features."""
    if engine is None:
        engine = get_engine()

    city_state = build_city_state_map()

    with engine.begin() as conn:
        aq = pd.read_sql(text("SELECT city, observed_date, avg_pm25, avg_pm10 FROM clean.air_quality_daily"), conn)
        wx = pd.read_sql(
            text("SELECT city, observed_date, avg_temp_c, rainfall_mm, avg_humidity, avg_wind_kmh "
                 "FROM clean.weather_daily"), conn)
        accidents = pd.read_sql(
            text("SELECT state, year, total_accidents, fatalities, injuries FROM clean.accidents"), conn)

    env = pd.merge(aq, wx, on=["city", "observed_date"], how="outer")
    env = pd.merge(env, city_state, on="city", how="left")
    env["observed_date"] = pd.to_datetime(env["observed_date"])
    env["year"] = env["observed_date"].dt.year
    env["month"] = env["observed_date"].dt.month
    env["is_monsoon"] = env["month"].isin([6, 7, 8, 9])

    # --- Annual state-year environmental aggregates ---
    state_year_env = (
        env.groupby(["state", "year"])
        .agg(
            avg_pm25=("avg_pm25", "mean"),
            avg_pm10=("avg_pm10", "mean"),
            avg_temp_c=("avg_temp_c", "mean"),
            total_rainfall_mm=("rainfall_mm", "sum"),
            avg_humidity=("avg_humidity", "mean"),
            avg_wind_kmh=("avg_wind_kmh", "mean"),
        )
        .reset_index()
    )

    # --- Monsoon rainfall share ---
    monsoon_rain = (
        env[env["is_monsoon"]]
        .groupby(["state", "year"])["rainfall_mm"]
        .sum()
        .reset_index()
        .rename(columns={"rainfall_mm": "monsoon_rainfall_mm"})
    )
    state_year_env = state_year_env.merge(monsoon_rain, on=["state", "year"], how="left")
    state_year_env["monsoon_rain_share"] = np.where(
        state_year_env["total_rainfall_mm"] > 0,
        state_year_env["monsoon_rainfall_mm"] / state_year_env["total_rainfall_mm"],
        np.nan,
    )

    # --- Environmental year-over-year deltas ---
    state_year_env = state_year_env.sort_values(["state", "year"])
    state_year_env["pm25_yoy_change"] = state_year_env.groupby("state")["avg_pm25"].diff()
    state_year_env["temp_yoy_change"] = state_year_env.groupby("state")["avg_temp_c"].diff()

    # --- Join with accidents and add severity + lag features ---
    df = pd.merge(accidents, state_year_env, on=["state", "year"], how="inner")
    df = df.sort_values(["state", "year"])

    df["fatality_rate"] = np.where(
        df["total_accidents"] > 0, df["fatalities"] / df["total_accidents"], np.nan
    )
    df["prev_total_accidents"] = df.groupby("state")["total_accidents"].shift(1)
    df["prev_fatality_rate"] = df.groupby("state")["fatality_rate"].shift(1)

    return df