-- ============================================================
-- CivicLens: CLEAN layer
-- Purpose: typed, deduped, validated data. One row per real-world
-- observation. Business rules (unit normalization, null handling,
-- city-name canonicalization) are applied here, not in raw or analytics.
-- Every table carries a quality_score / quality_flags for transparency.
-- ============================================================

CREATE SCHEMA IF NOT EXISTS clean;

CREATE TABLE IF NOT EXISTS clean.accidents (
    accident_id     BIGSERIAL PRIMARY KEY,
    state           TEXT        NOT NULL,
    year            INT         NOT NULL,
    month           INT,                        -- NULL when source only gives annual grain
    total_accidents INT,
    fatalities      INT,
    injuries        INT,
    quality_score   NUMERIC(3,2),               -- 0-1, e.g. completeness of the source record
    quality_flags   TEXT[],                     -- e.g. {'missing_month','imputed_fatalities'}
    source_raw_id   BIGINT REFERENCES raw.accidents(raw_id),
    loaded_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (state, year, month)
);

CREATE TABLE IF NOT EXISTS clean.air_quality_daily (
    city            TEXT        NOT NULL,
    observed_date   DATE        NOT NULL,
    avg_aqi         NUMERIC,
    avg_pm25        NUMERIC,
    avg_pm10        NUMERIC,
    avg_no2         NUMERIC,
    avg_so2         NUMERIC,
    avg_co          NUMERIC,
    n_readings      INT,                        -- how many raw readings were aggregated
    quality_score   NUMERIC(3,2),
    quality_flags   TEXT[],
    loaded_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (city, observed_date)
);

CREATE TABLE IF NOT EXISTS clean.weather_daily (
    city            TEXT        NOT NULL,
    observed_date   DATE        NOT NULL,
    avg_temp_c      NUMERIC,
    rainfall_mm     NUMERIC,
    avg_humidity    NUMERIC,
    avg_wind_kmh    NUMERIC,
    avg_visibility_km NUMERIC,
    quality_score   NUMERIC(3,2),
    quality_flags   TEXT[],
    loaded_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (city, observed_date)
);

CREATE INDEX IF NOT EXISTS idx_clean_accidents_state_year ON clean.accidents (state, year);
CREATE INDEX IF NOT EXISTS idx_clean_aq_city_date ON clean.air_quality_daily (city, observed_date);
CREATE INDEX IF NOT EXISTS idx_clean_weather_city_date ON clean.weather_daily (city, observed_date);
