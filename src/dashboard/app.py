"""CivicLens dashboard. Run with:
    streamlit run src/dashboard/app.py

Reads exclusively from the analytics.* star schema -- no raw/clean
queries here, keeping the dashboard fast and decoupled from ingestion
internals. Visual layer: custom CSS (cards, fade-in, hover), dark
plotly template, consistent palette.
"""

import json
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.warehouse.db import get_engine

st.set_page_config(
    page_title="CivicLens — Road Risk Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Global visual style ----------
PALETTE = ["#e45756", "#4c78a8", "#f2a154", "#59a89c", "#a26fb8", "#7d9c5b"]
pio.templates.default = "plotly_dark"

RISK_STYLE = {
    "Low": ("🟢", "#1f6f43", "#7ee2a8"),
    "Medium": ("🟡", "#8a6d1a", "#ffd866"),
    "High": ("🔴", "#7f1d1d", "#ff8a8a"),
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

/* ---- global typography ---- */
html, body, [class*="css"], .stMarkdown, button, input, select {
  font-family: 'Inter', sans-serif !important;
}

/* ---- animations ---- */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes gradientShift {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes glowPulse {
  0%, 100% { box-shadow: 0 0 24px rgba(228, 87, 86, 0.10); }
  50%      { box-shadow: 0 0 40px rgba(228, 87, 86, 0.22); }
}
.block-container { animation: fadeUp 0.55s cubic-bezier(0.16, 1, 0.3, 1); padding-top: 1.5rem; }

/* ---- hero header: animated gradient + glow ---- */
.hero {
  position: relative;
  background: linear-gradient(120deg, #141926, #1d2436, #2a1f30, #1d2436, #141926);
  background-size: 300% 300%;
  animation: gradientShift 14s ease infinite, glowPulse 6s ease-in-out infinite;
  border: 1px solid rgba(228, 87, 86, 0.25);
  border-radius: 20px;
  padding: 34px 38px 26px 38px;
  margin-bottom: 22px;
  overflow: hidden;
}
.hero::after {
  content: "";
  position: absolute;
  top: -60%; right: -20%;
  width: 55%; height: 220%;
  background: radial-gradient(ellipse, rgba(228,87,86,0.10) 0%, transparent 65%);
  pointer-events: none;
}
.hero h1 {
  font-size: 2.4rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  margin: 0 0 6px 0;
}
.hero h1 .logo-emoji {
  -webkit-text-fill-color: initial;
  margin-right: 6px;
}
.hero h1 .logo-text {
  background: linear-gradient(90deg, #ffffff 30%, #ff9d9c 65%, #e45756);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero p { color: #aab3c5; margin: 0; font-size: 0.97rem; line-height: 1.55; max-width: 720px; }
.hero .stats {
  display: flex; gap: 26px; margin-top: 16px; flex-wrap: wrap;
}
.hero .stat b {
  display: block; font-size: 1.25rem; font-weight: 700; color: #fff;
  font-family: 'JetBrains Mono', monospace;
}
.hero .stat span { font-size: 0.72rem; color: #8b94a7; text-transform: uppercase; letter-spacing: 0.08em; }

/* ---- glassmorphism metric cards ---- */
.card {
  background: linear-gradient(160deg, rgba(30,36,51,0.85), rgba(22,27,38,0.95));
  backdrop-filter: blur(8px);
  border: 1px solid rgba(96, 108, 138, 0.25);
  border-radius: 16px;
  padding: 20px 22px 16px 22px;
  transition: transform 0.22s cubic-bezier(0.16,1,0.3,1), border-color 0.22s ease, box-shadow 0.22s ease;
  height: 100%;
  position: relative;
  overflow: hidden;
}
.card::before {
  content: "";
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, rgba(228,87,86,0.7), transparent);
  opacity: 0;
  transition: opacity 0.25s ease;
}
.card:hover {
  transform: translateY(-4px) scale(1.012);
  border-color: rgba(228, 87, 86, 0.55);
  box-shadow: 0 12px 32px rgba(228, 87, 86, 0.16), 0 4px 12px rgba(0,0,0,0.4);
}
.card:hover::before { opacity: 1; }
.card .k { color: #8b94a7; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.09em; }
.card .v {
  color: #ffffff; font-size: 1.85rem; font-weight: 800; margin: 4px 0;
  letter-spacing: -0.02em; font-family: 'JetBrains Mono', monospace;
}
.card .s { font-size: 0.84rem; }

/* ---- risk chips ---- */
.chip {
  display: inline-block;
  padding: 3px 14px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.01em;
  border: 1px solid rgba(255,255,255,0.12);
}

/* ---- section titles ---- */
.section-title {
  font-size: 1.3rem;
  font-weight: 700;
  letter-spacing: -0.015em;
  margin: 14px 0 3px 0;
  padding-left: 12px;
  border-left: 3px solid #e45756;
}
.section-sub { color: #8b94a7; font-size: 0.88rem; margin: 0 0 14px 15px; line-height: 1.5; }

/* ---- tabs: pill style ---- */
.stTabs [data-baseweb="tab-list"] {
  gap: 6px;
  background: rgba(22,27,38,0.6);
  padding: 6px;
  border-radius: 14px;
  border: 1px solid rgba(96,108,138,0.18);
}
.stTabs [data-baseweb="tab"] {
  font-weight: 600;
  font-size: 0.88rem;
  border-radius: 10px;
  padding: 8px 16px;
  transition: background 0.2s ease, color 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(228,87,86,0.10); }
.stTabs [aria-selected="true"] {
  color: #ffffff !important;
  background: linear-gradient(135deg, rgba(228,87,86,0.85), rgba(180,55,70,0.85)) !important;
  box-shadow: 0 2px 10px rgba(228,87,86,0.35);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ---- expanders ---- */
[data-testid="stExpander"] {
  border: 1px solid rgba(96,108,138,0.25);
  border-radius: 14px;
  background: rgba(22,27,38,0.55);
  transition: border-color 0.2s ease;
}
[data-testid="stExpander"]:hover { border-color: rgba(228,87,86,0.45); }

/* ---- dataframes ---- */
[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(96,108,138,0.22);
}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #10141d 0%, #151a26 100%);
  border-right: 1px solid rgba(96,108,138,0.18);
}
[data-testid="stSidebar"] h2 { font-size: 1.05rem; letter-spacing: -0.01em; }

/* ---- scrollbar ---- */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: #10141d; }
::-webkit-scrollbar-thumb { background: #2c3242; border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: #e45756; }

/* ---- warning / info boxes rounded ---- */
[data-testid="stAlert"] { border-radius: 14px; }
</style>
"""


@st.cache_data(ttl=600)
def load_data():
    engine = get_engine()
    with engine.begin() as conn:
        cities = pd.read_sql(text("SELECT * FROM analytics.dim_city"), conn)
        dates = pd.read_sql(text("SELECT * FROM analytics.dim_date"), conn)
        accidents = pd.read_sql(text("SELECT * FROM analytics.fact_accident_month"), conn)
        environment = pd.read_sql(text("SELECT * FROM analytics.fact_environment_month"), conn)
        predictions = pd.read_sql(text("SELECT * FROM analytics.fact_risk_prediction"), conn)
        try:
            anomalies = pd.read_sql(text("SELECT * FROM analytics.anomalies"), conn)
        except Exception:
            anomalies = pd.DataFrame()

    accidents = accidents.merge(cities, on="city_id").merge(dates, on="date_id")
    environment = environment.merge(cities, on="city_id").merge(dates, on="date_id")
    predictions = predictions.merge(cities, on="city_id").merge(dates, on="date_id")
    if not anomalies.empty:
        anomalies = anomalies.merge(cities, on="city_id").merge(dates, on="date_id")
    return cities, accidents, environment, predictions, anomalies


def style_fig(fig, height=420):
    fig.update_layout(
        height=height,
        colorway=PALETTE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,25,38,0.35)",
        font=dict(family="Inter, sans-serif", size=13, color="#c9d1e0"),
        margin=dict(l=10, r=10, t=42, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        hoverlabel=dict(
            bgcolor="#1d2436",
            bordercolor="#e45756",
            font=dict(family="Inter, sans-serif", size=12, color="#ffffff"),
        ),
        hovermode="x unified",
        transition_duration=400,
    )
    fig.update_xaxes(gridcolor="rgba(44,50,66,0.5)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(44,50,66,0.5)", zeroline=False)
    return fig


def risk_chip(risk: str, confidence: float) -> str:
    icon, bg, fg = RISK_STYLE.get(risk, ("", "#333", "#ccc"))
    return (
        f'<span class="chip" style="background:{bg};color:{fg};">' f"{icon} {risk} · {confidence:.0%}</span>"
    )


def metric_card(label: str, value: str, sub_html: str = ""):
    st.markdown(
        f'<div class="card"><div class="k">{label}</div>'
        f'<div class="v">{value}</div><div class="s">{sub_html}</div></div>',
        unsafe_allow_html=True,
    )


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
          <h1><span class="logo-emoji">🚦</span><span class="logo-text">CivicLens</span></h1>
          <p>Road accident risk intelligence for major Indian cities — public data, honestly handled.
          End-to-end pipeline: ingestion → warehouse → ML with ablation analysis → explainable predictions.</p>
          <div class="stats">
            <div class="stat"><b>596K</b><span>env. records</span></div>
            <div class="stat"><b>453</b><span>CPCB stations</span></div>
            <div class="stat"><b>37</b><span>states</span></div>
            <div class="stat"><b>3×3</b><span>ablation study</span></div>
            <div class="stat"><b>37</b><span>anomalies found</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cities, accidents, environment, predictions, anomalies = load_data()

    st.sidebar.header("Filters")
    all_city_names = sorted(cities["city_name"].unique())
    selected_cities = st.sidebar.multiselect("Cities", all_city_names, default=all_city_names)
    st.sidebar.caption(
        "Data: MoRTH accidents (2021-2024, state grain) + CPCB air quality & weather "
        "(2010-2023, station grain). See Data & Methodology tab."
    )

    if not selected_cities:
        st.warning("Select at least one city from the sidebar.")
        return

    acc_f = accidents[accidents["city_name"].isin(selected_cities) & accidents["month"].isna()]
    env_f = environment[environment["city_name"].isin(selected_cities)]
    pred_f = predictions[predictions["city_name"].isin(selected_cities)]
    anom_f = anomalies[anomalies["city_name"].isin(selected_cities)] if not anomalies.empty else anomalies

    (
        tab_overview,
        tab_accidents,
        tab_environment,
        tab_predictions,
        tab_anomalies,
        tab_insights,
        tab_about,
    ) = st.tabs(
        [
            "Overview",
            "Accident Trends",
            "Environmental Trends",
            "Risk Predictions",
            "Anomalies",
            "Model Insights",
            "Data & Methodology",
        ]
    )

    # ---------------- Overview ----------------
    with tab_overview:
        latest_year = acc_f["year"].max()
        st.markdown(
            f'<div class="section-title">City snapshot — {int(latest_year)}</div>'
            '<div class="section-sub">Total accidents (state-level figure) and latest model risk assessment</div>',
            unsafe_allow_html=True,
        )

        latest_pred = (
            pred_f[pred_f["model_version"] == "enriched_v2"].sort_values("year").groupby("city_name").tail(1)
        )
        cols = st.columns(len(selected_cities))
        for i, city in enumerate(selected_cities):
            row = acc_f[(acc_f["city_name"] == city) & (acc_f["year"] == latest_year)]
            p = latest_pred[latest_pred["city_name"] == city]
            chip = risk_chip(p["risk_level"].iloc[0], p["confidence"].iloc[0]) if not p.empty else ""
            value = f"{int(row['total_accidents'].iloc[0]):,}" if not row.empty else "N/A"
            with cols[i]:
                metric_card(city, value, chip)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Accidents by city over time</div>'
            '<div class="section-sub">State-grain totals; cities sharing a state (Mumbai & Pune → Maharashtra) '
            "show identical values — a documented source limitation, not a bug</div>",
            unsafe_allow_html=True,
        )
        fig = px.line(
            acc_f.sort_values("year"),
            x="year",
            y="total_accidents",
            color="city_name",
            markers=True,
            labels={"total_accidents": "Total Accidents", "year": "Year", "city_name": "City"},
        )
        fig.update_traces(line=dict(width=3.5, shape="spline", smoothing=0.8), marker=dict(size=10))
        st.plotly_chart(style_fig(fig), use_container_width=True, key="overview_line")

    # ---------------- Accident Trends ----------------
    with tab_accidents:
        st.markdown('<div class="section-title">State-level accident totals</div>', unsafe_allow_html=True)
        pivot = acc_f.pivot_table(index="year", columns="city_name", values="total_accidents")
        st.dataframe(pivot.style.format("{:,.0f}"), use_container_width=True)

        st.markdown(
            '<div class="section-title">Year-over-year % change</div>'
            '<div class="section-sub">Growth or decline vs the prior year</div>',
            unsafe_allow_html=True,
        )
        pct_change = pivot.pct_change().round(3) * 100
        fig2 = px.bar(
            pct_change.reset_index().melt(id_vars="year", var_name="city_name", value_name="pct_change"),
            x="year",
            y="pct_change",
            color="city_name",
            barmode="group",
            labels={"pct_change": "% change vs prior year", "city_name": "City"},
        )
        st.plotly_chart(style_fig(fig2), use_container_width=True, key="yoy_bar")

    # ---------------- Environmental Trends ----------------
    with tab_environment:
        env_monthly = env_f[env_f["month"].notna()].copy()
        env_monthly["period"] = (
            env_monthly["year"].astype(int).astype(str)
            + "-"
            + env_monthly["month"].astype(int).astype(str).str.zfill(2)
        )
        env_monthly = env_monthly.sort_values(["year", "month"])

        st.markdown(
            '<div class="section-title">Monthly average PM2.5</div>'
            '<div class="section-sub">City-level average across all CPCB stations</div>',
            unsafe_allow_html=True,
        )
        fig3 = px.line(
            env_monthly,
            x="period",
            y="avg_pm25",
            color="city_name",
            labels={"avg_pm25": "Avg PM2.5 (µg/m³)", "period": "Month", "city_name": "City"},
        )
        fig3.update_xaxes(tickangle=45, nticks=20)
        st.plotly_chart(style_fig(fig3), use_container_width=True, key="pm25_line")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-title">Avg temperature</div>', unsafe_allow_html=True)
            fig4 = px.line(env_monthly, x="period", y="avg_temp_c", color="city_name")
            fig4.update_xaxes(tickangle=45, nticks=12)
            st.plotly_chart(style_fig(fig4, height=340), use_container_width=True, key="temp_line")
        with col_b:
            st.markdown('<div class="section-title">Total rainfall</div>', unsafe_allow_html=True)
            fig5 = px.line(env_monthly, x="period", y="total_rainfall_mm", color="city_name")
            fig5.update_xaxes(tickangle=45, nticks=12)
            st.plotly_chart(style_fig(fig5, height=340), use_container_width=True, key="rain_line")

    # ---------------- Risk Predictions ----------------
    with tab_predictions:
        available_versions = sorted(pred_f["model_version"].unique(), reverse=True)
        if not available_versions:
            st.info("No predictions available yet -- run a training script first.")
        else:
            selected_version = st.selectbox(
                "Model version",
                available_versions,
                help="enriched_v2: XGBoost with engineered features, per-prediction SHAP. "
                "baseline_v1: Random Forest, raw environmental features, global importance only.",
            )
            version_preds = pred_f[pred_f["model_version"] == selected_version]

            st.markdown(
                '<div class="section-title">Latest predicted risk level per city</div>'
                '<div class="section-sub">Expand a city to see WHY the model made this call</div>',
                unsafe_allow_html=True,
            )
            latest_preds = version_preds.sort_values("year").groupby("city_name").tail(1)
            for _, row in latest_preds.iterrows():
                icon, _, _ = RISK_STYLE.get(row["risk_level"], ("", "", ""))
                with st.expander(
                    f"{icon} {row['city_name']} — {row['risk_level']} risk ({row['confidence']:.0%} confidence)"
                ):
                    st.write(f"Based on {int(row['year'])} state-level data, model `{row['model_version']}`.")
                    features = (
                        json.loads(row["top_features"])
                        if isinstance(row["top_features"], str)
                        else row["top_features"]
                    )
                    feat_df = pd.DataFrame(features)

                    if "shap_value" in feat_df.columns:
                        feat_df["direction"] = feat_df["shap_value"].apply(
                            lambda v: "pushes toward this risk level" if v > 0 else "pushes away"
                        )
                        fig6 = px.bar(
                            feat_df.sort_values("shap_value"),
                            x="shap_value",
                            y="feature",
                            orientation="h",
                            color="direction",
                            color_discrete_map={
                                "pushes toward this risk level": "#e45756",
                                "pushes away": "#4c78a8",
                            },
                            labels={"shap_value": "SHAP value (this prediction)", "feature": ""},
                        )
                        st.plotly_chart(
                            style_fig(fig6, height=360),
                            use_container_width=True,
                            key=f"shap_{selected_version}_{row['city_name']}",
                        )
                        st.caption(
                            "Per-prediction SHAP: how each feature pushed THIS city's prediction "
                            "toward or away from the predicted risk level."
                        )
                    else:
                        fig6 = px.bar(
                            feat_df,
                            x="mean_abs_shap",
                            y="feature",
                            orientation="h",
                            labels={"mean_abs_shap": "Mean |SHAP value| (global importance)", "feature": ""},
                        )
                        st.plotly_chart(
                            style_fig(fig6, height=360),
                            use_container_width=True,
                            key=f"shap_{selected_version}_{row['city_name']}",
                        )
                        st.caption("Global feature importance from training (same for all cities in v1).")

            if selected_version == "enriched_v2":
                st.warning(
                    "Honest caveat: v2's confidences (93-96%) reflect overfitting to a small "
                    "23-row training sample, not genuine certainty. Its top driver everywhere is "
                    "the previous year's accident count -- accident history dominates once included. "
                    "See the Model Insights tab for the full ablation analysis."
                )

    # ---------------- Anomalies ----------------
    with tab_anomalies:
        st.markdown(
            '<div class="section-title">Unusual environmental readings</div>'
            '<div class="section-sub">City-months deviating sharply from that city\'s historical pattern '
            "for that calendar month (|z| &gt; 2.5 AND practically significant). Some are real events "
            "(Pune's record 2019-20 unseasonal rains); others are data-quality catches — one faulty rain "
            "gauge (TN004, Chennai) was found and excluded this way.</div>",
            unsafe_allow_html=True,
        )
        if anom_f.empty:
            st.info("No anomalies detected for the selected cities (or detect_anomalies.py hasn't been run).")
        else:
            metric_options = sorted(anom_f["metric"].unique())
            selected_metrics = st.multiselect("Metrics", metric_options, default=metric_options)
            view = anom_f[anom_f["metric"].isin(selected_metrics)].copy()
            view["period"] = (
                view["year"].astype(int).astype(str)
                + "-"
                + view["month"].astype(int).astype(str).str.zfill(2)
            )
            view["z_score"] = view["z_score"].astype(float).round(2)
            view["observed_value"] = view["observed_value"].astype(float).round(1)
            view["historical_mean"] = view["historical_mean"].astype(float).round(1)

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Anomalies detected", str(len(view)))
            with c2:
                metric_card("Most affected city", view["city_name"].mode().iloc[0] if len(view) else "—")
            with c3:
                metric_card("Max |z-score|", f"{view['z_score'].abs().max():.1f}" if len(view) else "—")
            st.markdown("<br>", unsafe_allow_html=True)

            display = (
                view[
                    [
                        "city_name",
                        "period",
                        "metric",
                        "observed_value",
                        "historical_mean",
                        "z_score",
                        "direction",
                    ]
                ]
                .sort_values("z_score", key=abs, ascending=False)
                .rename(
                    columns={
                        "city_name": "City",
                        "period": "Month",
                        "metric": "Metric",
                        "observed_value": "Observed",
                        "historical_mean": "Historical mean",
                        "z_score": "Z-score",
                        "direction": "Direction",
                    }
                )
            )
            st.dataframe(display, use_container_width=True, hide_index=True)

            fig_anom = px.scatter(
                view,
                x="period",
                y="z_score",
                color="city_name",
                symbol="metric",
                labels={"z_score": "Z-score", "period": "Month", "city_name": "City"},
                hover_data=["metric", "observed_value", "historical_mean"],
            )
            fig_anom.update_traces(marker=dict(size=11, line=dict(width=1, color="#0e1117")))
            fig_anom.update_xaxes(categoryorder="category ascending", tickangle=45)
            st.plotly_chart(style_fig(fig_anom), use_container_width=True, key="anomaly_scatter")

    # ---------------- Model Insights ----------------
    with tab_insights:
        st.markdown(
            '<div class="section-title">Ablation study: what actually predicts accident risk?</div>'
            '<div class="section-sub">Three feature configurations, identical cross-validation, three model '
            "families. Sample sizes differ because engineered features consume each state's first year.</div>",
            unsafe_allow_html=True,
        )
        ablation = pd.DataFrame(
            {
                "Configuration": [
                    "A: Baseline environment",
                    "B: Enriched environment (no history)",
                    "C: Full (env + accident history)",
                ],
                "Rows": [73, 40, 23],
                "Features": [6, 9, 11],
                "LogReg acc": [0.671, 0.600, 0.652],
                "RF acc": [0.630, 0.725, 0.739],
                "XGBoost acc": [0.603, 0.675, 0.783],
            }
        )
        st.dataframe(ablation, use_container_width=True, hide_index=True)

        melted = ablation.melt(
            id_vars=["Configuration"],
            value_vars=["LogReg acc", "RF acc", "XGBoost acc"],
            var_name="Model",
            value_name="Accuracy",
        )
        fig_abl = px.bar(
            melted,
            x="Configuration",
            y="Accuracy",
            color="Model",
            barmode="group",
            labels={"Accuracy": "Cross-validated accuracy"},
        )
        fig_abl.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(style_fig(fig_abl, height=380), use_container_width=True, key="ablation_bar")

        st.markdown("""
            **Findings, honestly stated:**

            1. **With simple features and the most data (A), logistic regression wins** —
               classic small-data behavior: flexible models overfit, linear models generalize.
            2. **Engineered environmental features (B) meaningfully help tree models**
               (RF: 63% → 72.5%) even with fewer rows — the environmental signal
               (pollution deltas, monsoon share) is real.
            3. **Accident history dominates when included (C).** XGBoost hits 78%, but
               per-prediction SHAP shows `prev_total_accidents` as the top driver for
               every single city — much of that accuracy is "states with many accidents
               last year have many this year." True, but nearly tautological.

            **Bottom line:** environmental conditions carry genuine but *secondary*
            predictive signal for state-year accident risk. Accident persistence is the
            strongest single predictor. Confidences from the 23-row model (C) should be
            treated as overfit, not as calibrated probabilities.
            """)
        st.caption(
            "Caveat: sample sizes differ across configurations, so accuracies are "
            "directional evidence rather than a strictly controlled comparison. "
            "Reproduce with: python scripts/run_ablation.py"
        )

    # ---------------- Data & Methodology ----------------
    with tab_about:
        st.markdown('<div class="section-title">Data sources</div>', unsafe_allow_html=True)
        st.markdown("""
            - **Accidents**: Ministry of Road Transport & Highways, via data.gov.in
              (state/year grain, 2021-2024)
            - **Air quality & weather**: CPCB monitoring stations, via a Kaggle bulk
              export (station-hourly, aggregated to city-day then state-year for
              modeling). Coverage: 2010 to March 2023.
            - **Fatalities & injuries**: MoRTH CSVs (state/year, 2021-2024), merged
              into the accident table.
            - **Modeling sample**: 73 state-years across 27 states (full accident
              dataset), not just the 6 dashboard cities -- see project README for why.
            """)
        st.markdown('<div class="section-title">Known limitations</div>', unsafe_allow_html=True)
        st.markdown("""
            - Accident and environmental data don't fully overlap in time (AQI stops
              March 2023; accidents run through 2024) -- 2024 predictions aren't
              available for that reason.
            - Risk labels (Low/Medium/High) are tertiles of raw accident counts, which
              partly reflects state size rather than pure risk.
            - Cities sharing a state show identical accident figures (source data is
              state-grain, not city-grain).
            - One faulty sensor (TN004, Chennai) was detected via anomaly analysis and
              its rainfall column excluded -- see docs/engineering_decisions.md.
            """)
        st.markdown(
            '<div class="section-sub">Full architecture, ablation analysis and setup: '
            '<a href="https://github.com/keyurc2332/civiclens">github.com/keyurc2332/civiclens</a></div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()