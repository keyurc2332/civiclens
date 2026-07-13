# CivicLens

[![CI](https://github.com/keyurc2332/civiclens/actions/workflows/ci.yml/badge.svg)](https://github.com/keyurc2332/civiclens/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10-blue)
![PostgreSQL](https://img.shields.io/badge/postgres-16-336791)
![License](https://img.shields.io/badge/license-portfolio-lightgrey)

**Road accident risk intelligence for major Indian cities — public data, honestly handled.**

## ⚡ Try it in one command

```bash
git clone https://github.com/keyurc2332/civiclens.git
cd civiclens
docker compose up --build
```

Then open:
- **Dashboard** → http://localhost:8501
- **API docs** → http://localhost:8000/docs

The stack (Postgres + FastAPI + Streamlit) comes pre-seeded with the analytics layer — no raw data download needed to explore the product. The full ingestion/training pipeline is documented below for reproducing everything from source.

## 📊 By the numbers

| | |
|---|---|
| **596K** | station-day environmental records processed |
| **453** | CPCB monitoring stations ingested |
| **37** | states in the training data |
| **73** | state-year training rows (baseline model) |
| **3 × 3** | models × feature sets in the ablation study |
| **37** | environmental anomalies detected |
| **1** | faulty rain gauge caught and excluded (TN004) |
| **7** | dashboard tabs |

A production-ready ML data pipeline and analytics system that ingests government accident and environmental data, cleans and validates it, trains interpretable risk classifiers, detects environmental anomalies, and exposes insights via REST API and interactive dashboard. Built as a portfolio project to demonstrate full-stack ML engineering: data sourcing, warehouse architecture, feature engineering, model rigor, explainability, and DevOps.

> 🧠 **Why decisions were made:** see [docs/engineering_decisions.md](docs/engineering_decisions.md) — PostgreSQL vs SQLite, why XGBoost's "win" carries an asterisk, why TN004 was only partially excluded, and more.

---

## Overview

CivicLens addresses a straightforward question: **Can we predict road accident risk from public environmental data?** The honest answer, backed by rigorous ablation analysis: **yes, but environmental conditions are a secondary signal — accident persistence dominates.**

This project showcases the rigor that makes the difference in real ML work:
- **Data quality detective work** — anomaly detection uncovered a faulty sensor (TN004, Chennai rain gauge) reporting wildly inflated values for a year. Excluded, documented, lessons learned.
- **Transparent limitations** — accident data comes at state grain, not city grain; environmental data coverage (2010–2023) doesn't fully overlap with accident data (2021–2024); the training sample is small (23 rows for the enriched model). All documented. No hand-waving.
- **Ablation study** — three feature sets (baseline environment / enriched environment / environment + history) compared across three model types. The finding: engineered environmental features help tree models, but accident history is the strongest single predictor by far.
- **Per-prediction explainability** — not just global feature importance; each city's risk prediction includes SHAP values showing what drove that specific forecast.

---

## Architecture

```
Data Sources (public APIs + bulk exports)
         ↓
Ingestion Pipelines (with retries, logging, quality scoring)
         ↓
Raw Layer (Postgres: raw.* tables, JSONB payloads, batch IDs)
         ↓
Cleaning & Validation (pandera schemas, quality flags, deduplication)
         ↓
Clean Layer (Postgres: clean.* tables, typed columns, per-row quality_score)
         ↓
Feature Engineering (lags, deltas, seasonal flags, anomaly detection)
         ↓
Analytics Star Schema (Postgres: analytics.dim_*, fact_*)
         ↓
Model Training (cross-validation, SHAP, ablation studies)
         ↓
Predictions + Explainability (stored back to analytics.fact_risk_prediction)
         ↓
REST API (FastAPI, /predictions, /cities, /anomalies, etc.)
         ↓
Interactive Dashboard (Streamlit: 7 tabs, model version selector, per-prediction SHAP)
         ↓
Anomaly Detection (seasonal z-scores, practical-significance floors, stored in analytics.anomalies)
```

**Tech stack:**
- **Ingestion**: Python (sqlalchemy, requests, pandas, retries)
- **Warehouse**: PostgreSQL 16 (star schema, JSONB for raw payloads)
- **Validation**: Pandera (type and constraint schemas)
- **ML**: scikit-learn, XGBoost, SHAP (cross-validation, TreeExplainer)
- **API**: FastAPI (auto-docs at `/docs`)
- **Dashboard**: Streamlit (7 tabs, model selector, per-prediction explanations)
- **Orchestration**: (TODO Phase 3 stretch) Dagster stubs in place; manual scripts for Phase 1–3

---

## What's Included

### Phase 1: Complete ✅
- Ingestion pipelines for 3 data sources (MoRTH accidents, CPCB air quality via Kaggle, co-located weather)
- Warehouse: raw → clean → analytics layers
- Data quality scoring and documented granularity mismatches (state-year accidents, city-day environment)
- Baseline risk classifier (Low/Medium/High tertiles) with 5-fold cross-validation
- Global SHAP feature importance
- Interactive dashboard (5 tabs)

### Phase 2: Complete ✅
- Fatalities/injuries CSVs ingested and merged
- Feature engineering: lag features, environmental deltas, monsoon seasonality, severity metrics
- Three-model comparison (logistic regression, random forest, XGBoost)
- **Ablation study** isolating the environmental signal's actual contribution
- Per-prediction SHAP explanations (v2 only; v1 uses global importance)
- Dashboard upgrade: model version selector, Model Insights tab with ablation findings

### Phase 3: Complete ✅
- REST API (FastAPI) exposing predictions, city data, environment history, model metadata
- Seasonal anomaly detection (z-scores with practical-significance floors)
- Faulty sensor discovery and exclusion (TN004 case study)
- Dashboard Anomalies tab with interactive table and scatter plot

---

## Key Findings

### The Ablation Story

Three feature configurations, cross-validated across three model types:

| Configuration | Rows | Features | LogReg Acc | RF Acc | XGBoost Acc |
|---|---|---|---|---|---|
| **A: Baseline environment** | 73 | 6 | 67.1% | 63.0% | 60.3% |
| **B: Enriched environment (no history)** | 40 | 9 | 60.0% | 72.5% | 67.5% |
| **C: Full (env + accident history)** | 23 | 11 | 65.2% | 73.9% | 78.3% |

**Interpretation:**
- With simple features and more data (A), logistic regression wins — classic small-sample behavior.
- Engineered environmental features (B) meaningfully help tree models (RF: 63% → 72.5%) even with fewer rows. The environmental signal is *real*.
- Accident history (C) pushes XGBoost to 78%, but per-prediction SHAP reveals **`prev_total_accidents` is the top driver for every city**, with confidence scores (93–96%) reflecting overfitting to a 23-row sample, not genuine certainty.

**Bottom line:** Environmental conditions carry genuine but secondary predictive signal. Accident persistence year-over-year is the strongest predictor. Any forecasting system should treat environment as a useful feature layer, not the primary decision driver.

### The TN004 Story

Anomaly detection flagged Chennai's entire 2019 year as anomalous: rainfall values 10–20× historical norms, consecutive months. This contradicted well-documented reality (2019 was Chennai's drought year, "Day Zero" water crisis). Investigation revealed:

**TN004 (Manali Village, Chennai):** A newly-deployed CPCB monitoring station (started 2019) had a faulty rain gauge reporting ~3.3mm/hour average for the entire first year, then flatlined to exactly 0.0mm from 2020 onward — classic sensor failure signature.

**Fix:** Excluded TN004's rainfall column from ingestion (other metrics like pollutants looked normal). This reduced anomaly false positives from 75 → 37, making real events (Pune's 2019–20 unseasonal rains) visible again.

**Why it matters:** In production ML, anomaly detection is often the first line of defense against data quality problems. This project demonstrates that workflow: detect → investigate → exclude/fix → document.

---

## Data Sources

### 1. Road Accidents (Ministry of Road Transport & Highways, data.gov.in)

**Resource ID:** `74624e5e-c174-4bfa-a25d-36ea3f580727`  
**Coverage:** State/UT level, 2021–2024 (annual)  
**Grain:** State-year (not city)  
**Metrics included:** Total accidents (2021–2024)  
**Metrics NOT included:** Fatalities/injuries are separate resources (ingested via Phase 2 CSVs)

**Limitations:**
- State grain only. Cities sharing a state (e.g., Mumbai & Pune, both Maharashtra) show identical values. This is a known source limitation, not a bug — documented in the dashboard.
- Annual grain. No month-level breakdown.
- 37 states covered; dashboard filters to 6 cities; model trains on all 37 states for a bigger sample.

**Sourcing method:** API (`api.data.gov.in`) with automatic retry (3 attempts, exponential backoff). The API blocks Python's default user-agent; `User-Agent: Mozilla/5.0 ...` header required (see `src/ingestion/base_client.py`).

### 2. Air Quality & Weather (CPCB Monitoring Stations via Kaggle)

**Dataset:** `abhisheksjha/time-series-air-quality-data-of-india-2010-2023` (Kaggle)  
**Coverage:** 453 monitoring stations nationwide, 2010–March 2023 (hourly raw data)  
**Grain:** Station-hour in source; aggregated to city-day and state-year for modeling  
**Metrics:** PM2.5, PM10, NO2, SO2, CO, Temperature, Humidity, Wind, Rainfall

**Why Kaggle instead of live APIs:** CPCB's historical archive (airquality.cpcb.gov.in) has been restructured; the old Python package (`vayuayan`) no longer works. The Kaggle export is a one-time bulk archive (2010–2023), transparently sourced from CPCB (see dataset description), and matches our model training window.

**Sourcing method:** Manual download from Kaggle, stored locally in `data_climate/`. Ingestion script (`scripts/ingest_air_quality.py`) reads all 453 station CSVs, aggregates hourly → daily (documented trade-off: loses temporal precision, gains practical manageability with 595K+ rows).

**Co-located weather advantage:** CPCB stations record meteorology (temp, humidity, wind, rainfall) alongside pollutants, eliminating the need for a separate IMD source. Same instrument, same calibration, proven pairing.

**Data quality issue found & fixed:**
- **TN004 (Chennai Manali):** Rain gauge started 2019, reported ~3.3mm/hour average, then 0.0mm from 2020 onward. Excluded rainfall column; other metrics kept. See "The TN004 Story" above.

**Coverage gap:** AQI data ends March 2023; accident data runs through 2024. The model can't make 2024 predictions because environmental features are missing. Documented limitation; honest note in dashboard.

### 3. Fatalities & Injuries (MoRTH, data.gov.in CSVs)

**Resources:**
- `State/UT-wise Number of Persons Killed in Road Accidents in India from 2021 to 2024`
- `State/UT-wise Number of Persons Injured in Road Accidents in India from 2021 to 2024`

**Coverage:** State level, 2021–2024 (annual)  
**Sourcing method:** Manual CSV download (no API), stored in `data_climate/` and ingested via `scripts/ingest_fatalities_injuries.py`.  
**Grain:** Merged into `clean.accidents` as `fatalities` and `injuries` columns.

**Limitations:** Same state-year grain as accidents. No granularity gains.

---

## Setup & Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL 16
- Docker (for containerized Postgres)
- Git

### Installation

**1. Clone the repo:**
```bash
git clone https://github.com/keyurc2332/civiclens.git
cd civiclens
```

**2. Set up the database (using Docker):**
```bash
docker pull postgres:16
docker run -d \
  --name civiclens_pg \
  -e POSTGRES_USER=civiclens \
  -e POSTGRES_PASSWORD=civiclens_dev \
  -e POSTGRES_DB=civiclens \
  -p 5433:5432 \
  -v civiclens_pgdata:/var/lib/postgresql/data \
  postgres:16
```

Or use the provided `docker-compose.yml` (see `.venv` activation below first):
```bash
docker compose up -d
```

**3. Python environment:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

**4. Configuration:**
Create a `.env` file in the repo root (or use the existing `.env.example` as a template):
```
DATABASE_URL=postgresql://civiclens:civiclens_dev@localhost:5433/civiclens
DATA_GOV_IN_API_KEY=<your-api-key-from-data.gov.in>
```

Obtain your API key from [data.gov.in](https://data.gov.in/api-key).

**5. Download Kaggle data:**
- Go to [abhisheksjha/time-series-air-quality-data-of-india-2010-2023](https://www.kaggle.com/datasets/abhisheksjha/time-series-air-quality-data-of-india-2010-2023) on Kaggle
- Download the dataset (requires free Kaggle account)
- Extract it to `data_climate/` in the repo root
- Also download the MoRTH fatalities & injuries CSVs from data.gov.in and save as:
  - `data_climate/accident_killed.csv`
  - `data_climate/accident_injured.csv`

**6. Run the pipelines (in order):**
```bash
# Phase 1: accidents
python scripts/ingest_accidents.py

# Phase 2: AQI + weather
python scripts/ingest_air_quality.py

# Phase 2: fatalities & injuries
python scripts/ingest_fatalities_injuries.py

# Phase 1: analytics star schema
python scripts/build_analytics.py

# Phase 2: enriched model
python scripts/train_model_v2.py

# Phase 3: anomalies
python scripts/detect_anomalies.py
```

**7. Start the REST API:**
```bash
uvicorn src.api.main:app --reload --port 8000
```
Browse to `http://localhost:8000/docs` for interactive API docs.

**8. Start the dashboard:**
```bash
streamlit run src/dashboard/app.py
```
Opens at `http://localhost:8501`.

---

## Project Structure

```
civiclens/
├── config/
│   └── config.yaml                 # Cities, resource IDs, model settings
├── data_climate/                   # Local Kaggle/CSV exports (git-ignored)
│   ├── *.csv                       # Station raw data
│   ├── stations_info.csv
│   ├── accident_killed.csv
│   └── accident_injured.csv
├── docs/
│   └── PROJECT_BRIEF.md            # Original requirements
├── models/                         # Trained model binaries (git-ignored)
│   ├── baseline_v1.joblib
│   └── enriched_v2.joblib
├── scripts/
│   ├── ingest_accidents.py         # Phase 1: fetch + clean accidents
│   ├── ingest_air_quality.py       # Phase 2: ingest CPCB bulk export
│   ├── ingest_fatalities_injuries.py  # Phase 2: merge CSVs
│   ├── build_analytics.py          # Phase 1: star schema population
│   ├── train_baseline_model.py     # Phase 1: baseline classifier
│   ├── train_model_v2.py           # Phase 2: enriched + 3-model comparison
│   ├── run_ablation.py             # Phase 2: ablation study
│   └── detect_anomalies.py         # Phase 3: seasonal anomaly detection
├── sql/
│   ├── 01_raw_schema.sql           # raw.* table definitions
│   ├── 02_clean_schema.sql         # clean.* table definitions
│   └── 03_analytics_schema.sql     # analytics.* star schema
├── src/
│   ├── api/
│   │   └── main.py                 # FastAPI app
│   ├── dashboard/
│   │   └── app.py                  # Streamlit dashboard (7 tabs)
│   ├── features/
│   │   └── engineering.py          # Feature sets for Phase 2 models
│   ├── ingestion/
│   │   ├── base_client.py          # Retry logic, user-agent fix
│   │   └── data_gov_client.py      # data.gov.in API wrapper
│   ├── validation/
│   │   └── checks.py               # Pandera schemas, quality scoring
│   └── warehouse/
│       └── db.py                   # SQLAlchemy engine, DDL runner
├── tests/
│   └── test_validation.py          # Pandera schema tests
├── .env                            # API key, DB URL (git-ignored)
├── .env.example                    # Template
├── .gitignore
├── docker-compose.yml              # Postgres + pgdata volume
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

---

## How to Explore

**Dashboard (recommended starting point):**
```bash
streamlit run src/dashboard/app.py
```
- **Overview tab:** 6 cities, accident counts 2021–2024, latest risk predictions
- **Accident Trends:** year-over-year % change
- **Environmental Trends:** PM2.5, temperature, rainfall over time
- **Risk Predictions:** Latest predictions with per-prediction SHAP explanations (v2) or global importance (v1); model version selector
- **Anomalies:** 37 detected environmental anomalies; metric filter; scatter plot
- **Model Insights:** The ablation table + three key findings
- **Data & Methodology:** Sources, limitations, caveats

**REST API (for integration / programmatic access):**
```bash
uvicorn src.api.main:app --reload --port 8000
# Then browse to http://localhost:8000/docs
```
Endpoints: `/cities`, `/cities/{city}/accidents`, `/cities/{city}/environment`, `/predictions`, `/predictions/{city}`, `/models`, `/health`

**Re-run pipelines (to replay the full analysis or experiment with code changes):**
```bash
# Full re-ingest + re-train (takes ~20 min)
python scripts/ingest_accidents.py
python scripts/ingest_air_quality.py
python scripts/ingest_fatalities_injuries.py
python scripts/build_analytics.py
python scripts/train_model_v2.py
python scripts/detect_anomalies.py

# Or just the baseline (Phase 1)
python scripts/train_baseline_model.py

# Just the ablation study
python scripts/run_ablation.py
```

---

## Honest Limitations

### Data Limitations
1. **Accident data is state-grain, not city-grain.** This is the source's published granularity. Cities sharing a state show identical values (Mumbai & Pune both show Maharashtra's totals). Documented in the dashboard; not a bug.
2. **No fatality/injury data in the initial resource.** Added in Phase 2 via separate CSVs, but this still doesn't give city-level granularity.
3. **Environmental data ends March 2023; accidents run through 2024.** No 2024 environmental features means no 2024 risk predictions. Honest limitation, documented in predictions.
4. **CPCB air quality data has variable coverage.** Some cities have only 5–10 years of history; weaker baseline for z-score anomaly detection (min 5 observations required). Still useful, but with caveats.

### Modeling Limitations
1. **Small training sample.** With 73 state-years (Phase 1 baseline) down to 23 (Phase 2 enriched), confidence intervals are wide. Cross-validation helps but doesn't fix fundamentally small-sample issues.
2. **State-year grain conflates size and risk.** Large states naturally have more accidents; our tertile buckets reflect both real risk and state population. Population-normalized rates would help but require external data (out of Phase 1 scope).
3. **Model confidences are overfit, not calibrated.** XGBoost v2 reports 93–96% confidence; actual uncertainty is higher due to small sample. Caveat baked into every API response and dashboard.
4. **Accident history dominates.** Once you include `prev_total_accidents`, environmental features add little incremental signal. This is a finding, not a flaw, but it limits the model's ability to identify *novel* risk factors.

### Infrastructure Limitations
1. **No automated scheduling (yet).** Ingestion scripts are manual; "production" would need Dagster/Airflow. Dagster stubs exist in `orchestration/` but aren't wired up (Phase 3 stretch goal).
2. **No real-time capabilities.** Everything is batch; no streaming ingestion or online predictions. CPCB real-time API data exists — could be added for a "current conditions" dashboard feature (Phase 3+ roadmap).
3. **User-Agent issue with data.gov.in.** The API blocks Python's default user-agent; workaround is hardcoded. Production systems might want to abstract this more gracefully.

---

## What's NOT Included (Deliberately Out of Scope)

1. **Forecasting.** This project predicts current risk from current conditions, not future risk. Time-series ARIMA/Prophet would be needed; out of brief scope (mentioned as optional Phase 3).
2. **Chatbots / "AI-generated insights."** No LLM integration. The dashboard's insights are built from data, not generated text.
3. **Streaming/real-time.** All ingestion is batch. Real-time monitoring would require different infrastructure.
4. **Multi-model ensembles for production.** Phase 2 compares three models to pick one; ensemble methods could squeeze more accuracy but add complexity. Not worth it at this sample size.
5. **Hyperparameter tuning.** Models use reasonable defaults; GridSearch/Bayesian optimization skipped (overfitting risk on small samples).
6. **Full CI/CD pipeline.** Tests exist (`tests/test_validation.py`), but no GitHub Actions yet. A production system would have automated testing on every PR.

---

## Future Roadmap (Phase 4+)

- **Dagster orchestration:** Wire up the existing stubs; automated daily runs
- **Fatality rate as outcome:** Build separate models predicting severity (fatalities per accident), not just accident occurrence
- **Population normalization:** Accident rates per capita, per road-km
- **Road density integration:** OSM data to normalize by road infrastructure
- **Real-time CPCB ingestion:** Live monitoring dashboard
- **Anomaly alerts:** Email/Slack notifications when environmental anomalies exceed thresholds
- **Causal analysis:** Granger causality or instrumental variables to test whether pollution actually *causes* accidents, not just correlates

---

## How This Demonstrates ML Engineering

**Data pipeline robustness:**
- Retry logic with exponential backoff
- Data quality scoring (pandera schemas, null%, duplicates)
- Documented granularity mismatches (no pretending city-grain data when it's state-grain)

**Warehouse design:**
- Raw → Clean → Analytics layers (dimensional modeling best practices)
- JSONB payloads in raw layer for auditability
- Star schema for analytics (fast queries, clear semantics)

**Model rigor:**
- Cross-validation (not single train/test split)
- Ablation study isolating environmental signal's real contribution
- SHAP explainability (not black-box accuracy chasing)
- Honest caveats on predictions (overfitting warning, small-sample disclaimer)

**Data quality vigilance:**
- Anomaly detection uncovered TN004 sensor fault
- Documented exclusion, not silent filtering
- Traced the false-positive anomalies back to root cause

**Production readiness:**
- REST API with auto-generated docs
- Interactive dashboard (7 tabs, model selector, filters)
- Reproducible pipeline (all scripts idempotent, re-runnable)

---

## Repository

**GitHub:** [github.com/keyurc2332/civiclens](https://github.com/keyurc2332/civiclens)

**Key commits to review:**
1. "Phase 1: warehouse schema + accident data ingestion" — architecture
2. "Phase 1 complete: analytics star schema + baseline risk classifier" — core model
3. "Phase 2: enriched features + 3-model comparison + per-prediction SHAP" — ablation findings
4. "Phase 2 complete: dashboard with model version selector + Model Insights tab" — transparency
5. "Phase 3: FastAPI REST API" — production API
6. "Phase 3 complete: anomaly detection + TN004 sensor fault found" — data quality detective work

---

## Questions?

This project intentionally documents its limitations. If you spot a gap or have questions about design choices, that's a feature, not a bug — it means the system is honest about what it knows and doesn't know.

For the ablation findings, Model Insights tab in the dashboard tells the whole story. For the TN004 sensor discovery, see `scripts/detect_anomalies.py` and the "Data Quality" section in this README.

---

## License

This project is for portfolio demonstration. Public datasets (data.gov.in, CPCB) are used under their respective open licenses. Code is available for reference and learning.

---

**Built with: Python • PostgreSQL • Pandas • Scikit-learn • XGBoost • SHAP • FastAPI • Streamlit**

**Last updated:** July 2026