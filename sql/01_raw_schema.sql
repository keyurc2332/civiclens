-- ============================================================
-- CivicLens: RAW layer
-- Purpose: land source data as close to "as received" as possible.
-- No type coercion beyond basics, no dedup, no business logic.
-- Every row carries ingestion metadata so we can trace lineage
-- and re-run/replay ingestion without losing history.
-- ============================================================

CREATE SCHEMA IF NOT EXISTS raw;

-- ---------------------------------------------------------
-- MoRTH road accident stats (data.gov.in)
-- Grain as published: state x year (sometimes state x month).
-- We store the full raw payload as JSONB alongside a few
-- promoted columns for convenience querying.
-- ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw.accidents (
    raw_id          BIGSERIAL PRIMARY KEY,
    source          TEXT        NOT NULL DEFAULT 'data.gov.in',
    resource_id     TEXT        NOT NULL,
    state           TEXT,
    year            INT,
    month           INT,                    -- NULL if only annual granularity available
    payload         JSONB       NOT NULL,   -- full raw API record
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    ingestion_batch_id UUID     NOT NULL
);

-- ---------------------------------------------------------
-- CPCB air quality (via data.gov.in)
-- Grain as published: station/city x timestamp (sub-daily).
-- ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw.air_quality (
    raw_id          BIGSERIAL PRIMARY KEY,
    source          TEXT        NOT NULL DEFAULT 'cpcb',
    station_id      TEXT,
    city            TEXT,
    observed_at     TIMESTAMPTZ,
    payload         JSONB       NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    ingestion_batch_id UUID     NOT NULL
);

-- ---------------------------------------------------------
-- IMD weather
-- Grain as published: city x day.
-- ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw.weather (
    raw_id          BIGSERIAL PRIMARY KEY,
    source          TEXT        NOT NULL DEFAULT 'imd',
    city            TEXT,
    observed_date   DATE,
    payload         JSONB       NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    ingestion_batch_id UUID     NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_accidents_state_year ON raw.accidents (state, year);
CREATE INDEX IF NOT EXISTS idx_raw_aq_city_time ON raw.air_quality (city, observed_at);
CREATE INDEX IF NOT EXISTS idx_raw_weather_city_date ON raw.weather (city, observed_date);
