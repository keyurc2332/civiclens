"""CivicLens dashboard. Run with:
    streamlit run src/dashboard/app.py

Reads exclusively from the analytics.* star schema -- no raw/clean
queries here, keeping the dashboard fast and decoupled from ingestion
internals.
"""
import json
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.warehouse.db import get_engine

st.set_page_config(page_title="CivicLens", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(ttl=600)
def load_data():
    engine = get_engine()
    with engine.begin() as conn:
        cities = pd.read_sql(text("SELECT * FROM analytics.dim_city"), conn)
        dates = pd.read_sql(text("SELECT * FROM analytics.dim_date"), conn)
        accidents = pd.read_sql(text("SELECT * FROM analytics.fact_accident_month"), conn)
        environment = pd.read_sql(text("SELECT * FROM analytics.fact_environment_month"), conn)
        predictions = pd.read_sql(text("SELECT * FROM analytics.fact_risk_prediction"), conn)

    accidents = accidents.merge(cities, on="city_id").merge(dates, on="date_id")
    environment = environment.merge(cities, on="city_id").merge(dates, on="date_id")
    predictions = predictions.merge(cities, on="city_id").merge(dates, on="date_id")
    return cities, accidents, environment, predictions


def main():
    st.title("🚦 CivicLens")
    st.caption("Road accident risk intelligence for major Indian cities — public data, honestly handled.")

    cities, accidents, environment, predictions = load_data()

    st.sidebar.header("Filters")
    all_city_names = sorted(cities["city_name"].unique())
    selected_cities = st.sidebar.multiselect("Cities", all_city_names, default=all_city_names)

    if not selected_cities:
        st.warning("Select at least one city from the sidebar.")
        return

    acc_f = accidents[accidents["city_name"].isin(selected_cities) & accidents["month"].isna()]
    env_f = environment[environment["city_name"].isin(selected_cities)]
    pred_f = predictions[predictions["city_name"].isin(selected_cities)]

    tab_overview, tab_accidents, tab_environment, tab_predictions, tab_about = st.tabs(
        ["Overview", "Accident Trends", "Environmental Trends", "Risk Predictions", "Data & Methodology"]
    )

    # ---------------- Overview ----------------
    with tab_overview:
        latest_year = acc_f["year"].max()
        cols = st.columns(len(selected_cities))
        for i, city in enumerate(selected_cities):
            row = acc_f[(acc_f["city_name"] == city) & (acc_f["year"] == latest_year)]
            pred_row = pred_f[pred_f["city_name"] == city].sort_values("year").tail(1)
            with cols[i]:
                st.metric(
                    label=city,
                    value=int(row["total_accidents"].iloc[0]) if not row.empty else "N/A",
                    help=f"Total accidents, {latest_year} (state-level figure)",
                )
                if not pred_row.empty:
                    risk = pred_row["risk_level"].iloc[0]
                    conf = pred_row["confidence"].iloc[0]
                    badge = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "")
                    st.caption(f"{badge} Predicted risk: **{risk}** ({conf:.0%} confidence)")

        st.divider()
        st.subheader("Accidents by city over time")
        fig = px.line(
            acc_f.sort_values("year"), x="year", y="total_accidents", color="city_name",
            markers=True, labels={"total_accidents": "Total Accidents", "year": "Year", "city_name": "City"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Note: accident data is published at state grain. Cities sharing a state "
            "(e.g. Mumbai & Pune, both Maharashtra) show identical values -- this is a "
            "known, documented limitation of the source data, not a bug."
        )

    # ---------------- Accident Trends ----------------
    with tab_accidents:
        st.subheader("State-level accident totals")
        pivot = acc_f.pivot_table(index="year", columns="city_name", values="total_accidents")
        st.dataframe(pivot, use_container_width=True)

        st.subheader("Year-over-year % change")
        pct_change = pivot.pct_change().round(3) * 100
        fig2 = px.bar(
            pct_change.reset_index().melt(id_vars="year", var_name="city_name", value_name="pct_change"),
            x="year", y="pct_change", color="city_name", barmode="group",
            labels={"pct_change": "% change vs prior year"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- Environmental Trends ----------------
    with tab_environment:
        st.subheader("Monthly average PM2.5 by city")
        env_monthly = env_f[env_f["month"].notna()].copy()
        env_monthly["period"] = env_monthly["year"].astype(int).astype(str) + "-" + env_monthly["month"].astype(int).astype(str).str.zfill(2)
        fig3 = px.line(
            env_monthly.sort_values(["year", "month"]), x="period", y="avg_pm25", color="city_name",
            labels={"avg_pm25": "Avg PM2.5 (µg/m³)", "period": "Month"},
        )
        fig3.update_xaxes(tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Avg temperature by city")
            fig4 = px.line(env_monthly.sort_values(["year", "month"]), x="period", y="avg_temp_c", color="city_name")
            st.plotly_chart(fig4, use_container_width=True)
        with col_b:
            st.subheader("Total rainfall by city")
            fig5 = px.line(env_monthly.sort_values(["year", "month"]), x="period", y="total_rainfall_mm", color="city_name")
            st.plotly_chart(fig5, use_container_width=True)

    # ---------------- Risk Predictions ----------------
    with tab_predictions:
        st.subheader("Latest predicted risk level per city")
        latest_preds = pred_f.sort_values("year").groupby("city_name").tail(1)
        for _, row in latest_preds.iterrows():
            badge = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(row["risk_level"], "")
            with st.expander(f"{badge} {row['city_name']} — {row['risk_level']} risk ({row['confidence']:.0%} confidence)"):
                st.write(f"Based on {int(row['year'])} state-level environmental averages, model `{row['model_version']}`.")
                features = json.loads(row["top_features"]) if isinstance(row["top_features"], str) else row["top_features"]
                feat_df = pd.DataFrame(features)
                fig6 = px.bar(
                    feat_df, x="mean_abs_shap", y="feature", orientation="h",
                    labels={"mean_abs_shap": "Mean |SHAP value| (feature importance)", "feature": ""},
                )
                st.plotly_chart(fig6, use_container_width=True, key=f"shap_{row['city_name']}")
        st.caption(
            "Feature importance shown is global (from cross-validated training), not "
            "per-prediction SHAP values -- a reasonable Phase 1 simplification, notable "
            "improvement for Phase 2 would be per-row SHAP explanations."
        )

    # ---------------- Data & Methodology ----------------
    with tab_about:
        st.subheader("Data sources")
        st.markdown(
            """
            - **Accidents**: Ministry of Road Transport & Highways, via data.gov.in
              (state/year grain, 2021-2024)
            - **Air quality & weather**: CPCB monitoring stations, via a Kaggle bulk
              export (station-hourly, aggregated to city-day then state-year for
              modeling). Coverage: 2010 to March 2023.
            - **Modeling sample**: 73 state-years across 27 states (full accident
              dataset), not just the 6 dashboard cities -- see project README for why.
            """
        )
        st.subheader("Known limitations")
        st.markdown(
            """
            - Accident and environmental data don't fully overlap in time (AQI stops
              March 2023; accidents run through 2024) -- 2024 predictions aren't
              available for that reason.
            - Risk labels (Low/Medium/High) are tertiles of raw accident counts, which
              partly reflects state size rather than pure risk.
            - Cities sharing a state show identical accident figures (source data is
              state-grain, not city-grain).
            """
        )


if __name__ == "__main__":
    main()