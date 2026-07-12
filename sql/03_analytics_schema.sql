-- ============================================================
-- CivicLens: ANALYTICS layer (star schema)
-- Grain: city x month (accident data's real granularity is
-- state/year-or-month; we roll everything up to the coarsest
-- common grain the source data actually supports).
--
-- dim_city, dim_date are conformed dimensions.
-- fact_environment_month is city-grain (AQI/weather aggregated up
-- from daily to monthly to match).
-- fact_accident_month is state-grain but joined via each city's
-- state, since MoRTH data doesn't go finer than state.
-- fact_risk_prediction stores model outputs + explainability.
-- ============================================================

CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE IF NOT EXISTS analytics.dim_city (
    city_id     SERIAL PRIMARY KEY,
    city_name   TEXT NOT NULL UNIQUE,
    state       TEXT NOT NULL,
    latitude    NUMERIC,
    longitude   NUMERIC
);

CREATE TABLE IF NOT EXISTS analytics.dim_date (
    date_id     INT PRIMARY KEY,       -- YYYYMM, e.g. 202401
    year        INT NOT NULL,
    month       INT,                   -- NULL if this row represents an annual-only grain
    quarter     INT,
    is_monsoon  BOOLEAN,               -- Jun-Sep flag, useful seasonality feature
    label       TEXT NOT NULL          -- '2024-01' or '2024' for annual-only
);

-- State-grain accident facts, but stored keyed by city_id so it
-- joins cleanly with environment data. Multiple cities in the
-- same state will share these values -- documented, not hidden.
CREATE TABLE IF NOT EXISTS analytics.fact_accident_month (
    city_id         INT NOT NULL REFERENCES analytics.dim_city(city_id),
    date_id         INT NOT NULL REFERENCES analytics.dim_date(date_id),
    grain           TEXT NOT NULL CHECK (grain IN ('state_month','state_year')),
    total_accidents INT,
    fatalities      INT,
    injuries        INT,
    PRIMARY KEY (city_id, date_id)
);

CREATE TABLE IF NOT EXISTS analytics.fact_environment_month (
    city_id         INT NOT NULL REFERENCES analytics.dim_city(city_id),
    date_id         INT NOT NULL REFERENCES analytics.dim_date(date_id),
    avg_aqi         NUMERIC,
    avg_pm25        NUMERIC,
    avg_pm10        NUMERIC,
    total_rainfall_mm NUMERIC,
    avg_temp_c      NUMERIC,
    avg_humidity    NUMERIC,
    avg_wind_kmh    NUMERIC,
    road_density_km_per_km2 NUMERIC,   -- static OSM-derived feature, optional
    PRIMARY KEY (city_id, date_id)
);

CREATE TABLE IF NOT EXISTS analytics.fact_risk_prediction (
    prediction_id   BIGSERIAL PRIMARY KEY,
    city_id         INT NOT NULL REFERENCES analytics.dim_city(city_id),
    date_id         INT NOT NULL REFERENCES analytics.dim_date(date_id),
    model_version   TEXT NOT NULL,
    risk_level      TEXT NOT NULL CHECK (risk_level IN ('Low','Medium','High')),
    confidence      NUMERIC(4,3),
    top_features    JSONB,              -- e.g. [{"feature":"avg_aqi","shap_value":0.34}, ...]
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_fact_accident_date ON analytics.fact_accident_month (date_id);
CREATE INDEX IF NOT EXISTS idx_fact_env_date ON analytics.fact_environment_month (date_id);
CREATE INDEX IF NOT EXISTS idx_fact_risk_date ON analytics.fact_risk_prediction (date_id);
