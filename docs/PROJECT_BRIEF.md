# CivicLens — Project Brief & Build Spec

Use this document to onboard any AI assistant (ChatGPT, Claude, Claude Code, Cursor, etc.) to this project in one shot. Paste the whole thing as context before asking for help on any specific part.

## One-Sentence Definition
CivicLens is a public data intelligence platform that ingests government accident, weather, air quality, and (optionally) traffic-related data for major Indian cities, validates and warehouses it properly, and predicts road accident risk levels using machine learning — surfaced through an interactive dashboard. Every feature must serve this sentence; if it doesn't, cut it.

## Context
- Builder: MS Data Science student (Texas A&M, incoming), targeting Data Scientist / ML Engineer / AI Engineer / Applied Scientist roles at top-tier companies (Google, Microsoft, Databricks, Snowflake, Palantir, NVIDIA, Uber, Amazon, Meta, etc.)
- Purpose: flagship portfolio project for LinkedIn, GitHub, and possibly resume — must be genuinely finished and polished, not a sprawling half-built platform
- Timeline: one semester
- Prior projects already on the profile: PsyMetric AI (career/psych), ChessIQ, IPL analytics, an industrial motor digital twin, IoT + research work, predictive ML systems. Data engineering rigor is the current gap in the profile — this project should fill it, but the person is targeting DS/ML/AI roles, not pure Data Engineer roles, so the project must show modeling depth, not just pipeline plumbing.

## Core Idea
Not a dashboard alone. Not a model alone. Not an ETL pipeline alone. An end-to-end intelligent data product: ingest → validate → clean → warehouse → engineer features → predict accident risk → explain the prediction → visualize.

## Data Sources (3–5 max, no more)
1. **OGD India (data.gov.in)** — road accident statistics. Registration required, API key generated via My Account → Generate Key. Query pattern: `https://api.data.gov.in/resource/{resource_id}?api-key={key}&format=json&offset=0&limit=100&filters[field]=value`
   - **Critical data reality**: MoRTH road accident data on OGD is compiled **year- and state-wise** (annual aggregates), NOT daily incident-level logs with geolocation. Do not fake finer granularity than what exists.
2. **CPCB Air Quality API** (via data.gov.in) — real-time-ish AQI, PM2.5, PM10, NO₂, SO₂, CO. City/station-level, much higher frequency than accident data.
3. **IMD Weather data** — temperature, rainfall, humidity, wind speed, visibility. City-level, daily.
4. **(Optional, low priority)** Traffic/road density — Google Traffic API is gated behind commercial Maps Platform pricing, not realistically free for this use case. Prefer static OSM-derived road density as a feature instead of trying to source live traffic data.

## Geographic Scope
Major Indian cities with reliable monitoring: Mumbai, Delhi, Bengaluru, Pune, Hyderabad, Chennai. Not one city (too little data), not all districts (too sparse/inconsistent).

## Granularity Decision (resolve the mismatch honestly)
- Accident data: state/year (sometimes state/month if available) — this is the coarsest input, so it sets the modeling grain.
- AQI/weather data: daily, city-level — aggregate UP (rolling averages, monthly means, etc.) to match the accident data's grain rather than pretending accident data is daily.
- Do not fabricate or interpolate finer granularity than the source data supports. This granularity mismatch and how it's handled is itself a legitimate engineering talking point for interviews.

## Architecture

### Data Engineering Layer
- Scheduled ingestion jobs (e.g., Airflow or Dagster) with retries and logging
- Schema validation on ingest
- Data quality checks: missing values, duplicates, schema drift, null percentages, quality scoring
- Data versioning, incremental loading
- Warehouse layers: **Raw → Clean → Analytics** (star schema or similar), e.g., Postgres or a free-tier cloud warehouse

### Feature Engineering
Rolling averages, lag features, seasonality indicators, rainfall accumulation, AQI averages, weather interaction terms, weekend indicators, (optional) festival indicators, road density if included.

### Machine Learning
- **Primary model**: accident risk classification — **Low / Medium / High** per state/city-month (not daily counts, not a continuous 0–100 score — the accident data's real granularity doesn't support finer targets)
- Train and compare multiple baseline models (don't pre-commit to a specific algorithm before seeing the data)
- Proper validation: cross-validation, confusion matrix, precision/recall/F1 per class, ROC AUC if appropriate — scaled to the actual dataset size (don't over-engineer evaluation for a small state/month sample)
- Explainability: feature importance and/or SHAP, and every prediction should be able to answer "why" (e.g., "High risk — driven by elevated rainfall, high AQI, historical seasonal trend")

### Dashboard
Interactive: risk maps, trend charts, city comparisons, historical analysis, filters, feature importance display. This is what recruiters actually click through — it needs to look and feel finished, not a wireframe.

### API (later phase)
Expose predictions via REST API for potential external use.

## Build Order (vertical, not horizontal — finish each layer before starting the next)

**Phase 1 (Core — this alone is already a strong finished project):**
Data ingestion → warehouse → cleaning/validation → basic dashboard → simple baseline risk model

**Phase 2 (if Phase 1 is solid and there's time left):**
Feature engineering refinement → improved/compared models → explainability layer → polished interactive dashboard

**Phase 3 (stretch goals only, don't start until Phase 1–2 are genuinely finished):**
AQI/trend forecasting → anomaly detection → REST API → (optional) AI-generated natural-language insights over model outputs

## Explicit Non-Goals (do not build these — this is what keeps scope sane)
- Chatbots or "chat with your data" interfaces as a core feature
- Knowledge graphs
- Multi-agent systems
- Ingesting dozens of data sources / World Bank + WHO + NASA + NOAA + everything
- PDF export, authentication/user accounts — cut or treat as trivial/last
- Any buzzword-heavy architecture that doesn't serve the one-sentence definition above

## Success Criteria
A recruiter or interviewer looking at this should conclude: "This person understands real-world data engineering, honest handling of messy/limited data, machine learning done properly (not just for accuracy), and can ship a finished, polished product — not just a notebook." Prioritize a fully finished Phase 1 over a half-finished Phase 3.
