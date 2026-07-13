# Engineering Decisions

This document explains *why* the major technical choices in CivicLens were made — including the ones that didn't work out as planned. Written for reviewers and interviewers who care more about reasoning than accuracy numbers.

---

## 1. Why PostgreSQL over SQLite (or files)?

**Decision:** PostgreSQL 16 in Docker, with three schemas (`raw`, `clean`, `analytics`).

**Reasoning:**
- The project's core learning goal was warehouse architecture — raw/clean/analytics layering, JSONB payload auditability, star schema design. SQLite supports none of this idiomatically (no schemas, weak JSON support, no concurrent writers).
- JSONB in the raw layer preserves the *exact* API responses with batch IDs, so any cleaning bug can be traced back to source payloads without re-fetching.
- Docker means zero pollution of the host machine and a reproducible dev environment. A native Postgres install was already squatting port 5432 on the dev machine — the container simply maps 5433 instead, an isolation win that a native install couldn't offer.

**Trade-off accepted:** More setup complexity than SQLite. Worth it — the warehouse design *is* the portfolio piece.

---

## 2. Why Pandera instead of manual validation?

**Decision:** Pandera schemas for the clean layer, plus custom quality scoring (`quality_score`, `quality_flags` per row).

**Reasoning:**
- Declarative schemas are self-documenting: a reviewer can read the expected types/ranges in one place.
- Pandera failures produce precise, column-level errors instead of mysterious downstream breakage.
- The custom quality score complements (not replaces) schema validation: schemas answer "is this row structurally valid?", the score answers "how complete is it?" — different questions.

**Trade-off accepted:** Another dependency. Manual `assert` checks would have been lighter but unmaintainable as sources grew.

---

## 3. Why train on all 37 states when the dashboard shows 6 cities?

**Decision:** The clean layer and model training use every state in the source data; the 6-city scope is applied only at the analytics/dashboard layer.

**Reasoning:**
- The original 6-city scope gave a 12-row training set (6 cities × 2 usable years) — statistically indefensible for a 3-class classifier.
- The accident source publishes all states anyway; filtering early threw away free sample size. Widening to all states gave 73 state-years.
- This mirrors a real warehouse principle: raw/clean layers hold everything reasonably available; scoping is a presentation-layer concern.

**Trade-off accepted:** State-year grain conflates state size with risk (large states trend "High"). Documented rather than hidden; population normalization is the top Phase 4 roadmap item.

---

## 4. Why XGBoost — and why its "win" is reported with a caveat

**Decision:** Three models compared (logistic regression, random forest, XGBoost) under identical stratified 5-fold cross-validation. XGBoost selected for v2 predictions.

**Reasoning:**
- XGBoost scored highest on the enriched feature set (78.3% vs 65.2% LR), and its native `pred_contribs` gives exact per-prediction SHAP values without version headaches.
- **But the honest finding is in the ablation, not the leaderboard:** with only 23 training rows, the LR→XGB gap is ~3 predictions — within noise. And per-prediction SHAP shows `prev_total_accidents` dominating every city's prediction. Much of XGBoost's edge is memorizing accident persistence.
- On the *largest* sample (73 rows, simple features), logistic regression actually wins — textbook small-data behavior, reported as such.

**Trade-off accepted:** Reporting a "winner with asterisks" is less punchy than "78% accuracy!" but is the defensible claim.

---

## 5. Why no hyperparameter tuning?

**Decision:** All models use fixed, conservative hyperparameters (e.g., `max_depth=3-4`, `min_samples_leaf=2`).

**Reasoning:**
- With 23–73 training rows, a hyperparameter search would overfit *the validation folds themselves* — the search would find noise, not signal.
- Conservative defaults (shallow trees, leaf minimums) act as regularization appropriate to the sample size.

**What I'd do with more data:** nested cross-validation with an inner tuning loop, once there are enough rows that fold-level variance stops dominating.

---

## 6. Why was the TN004 sensor excluded — and why only partially?

**Decision:** Station TN004 (Manali Village, Chennai) has its **rainfall column nulled at ingestion**; its pollutant and temperature readings are retained.

**Reasoning:**
- Anomaly detection flagged Chennai 2019 with 8 consecutive months of ~10× historical rainfall — during Chennai's famous "Day Zero" *drought* year. Contradiction with documented reality triggered investigation.
- Per-station analysis found TN004 (new in 2019) averaging 3.3mm/hour rainfall for its first year (~20× Chennai's real annual total), then exactly 0.0 forever after — a classic faulty-gauge signature.
- Its *other* metrics tracked neighboring stations normally, so only the demonstrably broken column was dropped. Whole-station exclusion would have discarded good pollutant data.
- The exclusion lives in code (`FAULTY_STATION_METRICS` in the ingestion script) with a comment explaining the evidence — documented, reviewable, reversible.

**Broader lesson demonstrated:** anomaly detection's first real-world catch was a data-quality fault, not a weather event. That is typical in production systems and the project treats it as a first-class finding.

---

## 7. Why z-scores for anomaly detection instead of ML methods?

**Decision:** Seasonal z-scores (per city × calendar-month × metric) with practical-significance floors, not isolation forests or autoencoders.

**Reasoning:**
- ~13 years × 12 months of history per city is enough for robust month-conditioned baselines, and z-scores are fully explainable — every flag comes with "observed X, historical mean Y, z=Z".
- The first iteration flagged 0.1mm drizzles in dry months as 3.5σ events (near-zero variance breaks z-scores). Fix: a per-metric **minimum absolute deviation floor** (e.g., ≥10mm rainfall), requiring anomalies to be both statistically extreme *and* practically meaningful. False positives dropped 75 → 37.
- An ML detector would have added opacity exactly where trust matters most.

---

## 8. Why aggregate hourly data to daily at ingestion?

**Decision:** The 453 CPCB station files (hourly, 2010–2023) are aggregated to station-day during ingestion; the warehouse never stores raw hourly rows.

**Reasoning:**
- The modeling grain is state-year; hourly precision is discarded downstream regardless.
- ~9M hourly rows through row-wise Postgres inserts was impractical for this project's infrastructure; 596K station-day rows load in minutes.
- This is a documented exception to "raw mirrors source exactly" — the docstring in the ingestion script says so explicitly, which is the honest version of the trade-off.

---

## 9. Why Kaggle bulk data instead of live CPCB APIs?

**Decision:** Historical AQI/weather from a Kaggle mirror of CPCB station data; the live data.gov.in CPCB API kept only as a future "current conditions" feature.

**Reasoning, in order of what was tried:**
1. The live API returns a single current-hour snapshot — zero overlap with 2021–2024 accident data.
2. The community package for CPCB's archive (`vayuayan`) broke when CPCB restructured their backend to an HTML SPA.
3. The ideal commercial dataset (Dataful) was paywalled (₹999).
4. The chosen Kaggle dataset documents its provenance (Selenium scrape of CPCB's official portal), includes station metadata, covers 2010–2023, and its per-station values were sanity-checked during ingestion.

**Trade-off accepted:** Coverage ends March 2023, so no 2024 predictions — stated in the dashboard and API rather than papered over.

---

## 10. Why is same-year fatality rate excluded from features?

**Decision:** `fatality_rate = fatalities / total_accidents` is computed but **only its lagged version** enters the model.

**Reasoning:** The target is the tertile bucket of `total_accidents`. Same-year fatality rate contains the target in its denominator — data leakage. The lagged version (`prev_fatality_rate`) carries the severity signal without contaminating the target. This is called out in the feature module's docstring so future contributors don't "helpfully" add it back.

---

## 11. What I'd build next with more data

- **Population/road-length normalization** — converts "big state detector" into a real risk model.
- **Monthly accident data** — would unlock the monsoon features properly (annual grain wastes them).
- **Causal analysis** (Granger tests, instrumental variables) — the current model is correlational; the PM2.5 signal is likely an urbanization proxy, and proving/disproving that is the genuinely interesting research question.
- **Dagster orchestration** — stubs exist; wiring them up turns manual scripts into a scheduled, observable pipeline.