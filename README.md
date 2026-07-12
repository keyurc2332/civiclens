# CivicLens

Public data intelligence platform for road accident risk in major Indian
cities. Ingests government accident, air quality, and weather data,
warehouses it through a Raw → Clean → Analytics pipeline, and predicts
monthly accident risk (Low/Medium/High) per city with explainability.

See `docs/PROJECT_BRIEF.md` (paste the original brief there) for full
scope, non-goals, and success criteria.

## Architecture

```
Source APIs (data.gov.in, CPCB, IMD)
        │
        ▼
   raw.*        <- landed as close to "as received" as possible (JSONB)
        │  (validation, typing, dedup — src/validation)
        ▼
   clean.*      <- typed, deduped, quality-scored
        │  (aggregation up to city-month grain — src/features)
        ▼
   analytics.*  <- star schema: dim_city, dim_date, fact_accident_month,
                   fact_environment_month, fact_risk_prediction
        │
        ▼
   src/models   <- risk classifier + SHAP explainability
        │
        ▼
   src/dashboard <- Streamlit app
```

Orchestrated with Dagster (`orchestration/definitions.py`).

## Local setup

```bash
# 1. Start the warehouse (auto-runs sql/01-03 on first boot)
docker compose up -d

# 2. Python env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env
# fill in DATA_GOV_IN_API_KEY (data.gov.in -> My Account -> Generate API Key)

# 4. Fill in resource IDs in config/config.yaml
#    (search data.gov.in for the MoRTH accident dataset and CPCB AQI dataset)

# 5. Run orchestration locally
dagster dev -f orchestration/definitions.py
```

## Repo layout

```
config/         central config (cities, resource ids, modeling settings)
sql/            Raw -> Clean -> Analytics DDL (Postgres)
src/ingestion/  API clients per source (data.gov.in, CPCB, IMD)
src/validation/ schema + quality checks (pandera)
src/warehouse/  DB connection helper
src/features/   feature engineering (rolling averages, lags, seasonality)
src/models/     risk classifier training + evaluation + SHAP
src/dashboard/  Streamlit app
orchestration/  Dagster asset definitions
tests/          pytest
```

## Status

Phase 1 (core): scaffolding in progress. See project brief for build
order and non-goals — the priority is a genuinely finished Phase 1
over a half-finished Phase 3.
