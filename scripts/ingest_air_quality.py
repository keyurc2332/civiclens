"""Phase 1 ingestion: CPCB station-level air quality + co-located weather
(from the local Kaggle bulk export) -> raw.air_quality / raw.weather ->
clean.air_quality_daily / clean.weather_daily.

Data source: CPCB historical monitoring stations, distributed via a
Kaggle dataset (see docs/PROJECT_BRIEF.md for full provenance notes).
Not a live API -- a one-time local bulk CSV import, so this script is
run manually rather than scheduled.

IMPORTANT design decisions, documented here deliberately:
1. Source files are HOURLY, going back to 2010. We aggregate hourly ->
   daily per station during ingestion itself (not a separate step),
   because (a) our modeling grain is city-month, so hourly precision
   is discarded either way, and (b) importing ~9 million raw hourly
   rows into Postgres via row-by-row inserts isn't practical for a
   semester project. This means our "raw" layer here is technically
   raw-per-station-day, not raw-per-hour -- a documented exception to
   the "raw mirrors source exactly" rule used for the API sources.
2. These CPCB stations record meteorology (temperature, humidity,
   wind, rainfall) alongside pollutants. We use this instead of a
   separate IMD source -- co-located sensor data is standard practice
   in air quality analysis and avoids a fourth data source.
3. We keep raw pollutant concentrations (PM2.5, PM10, NO2, SO2, CO)
   rather than computing CPCB's composite AQI number -- more useful
   as ML features, and avoids reimplementing CPCB's proprietary
   sub-index breakpoint tables.

Run with:
    python scripts/ingest_air_quality.py
"""
import glob
import json
import os
import sys

import pandas as pd
import yaml
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.warehouse.db import get_engine

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_climate")
STATIONS_INFO_PATH = os.path.join(DATA_DIR, "stations_info.csv")

# Source column -> our schema's column, split by pollutant vs weather.
POLLUTANT_COLS = {
    "PM2.5 (ug/m3)": "avg_pm25",
    "PM10 (ug/m3)": "avg_pm10",
    "NO2 (ug/m3)": "avg_no2",
    "SO2 (ug/m3)": "avg_so2",
    "CO (mg/m3)": "avg_co",
}
WEATHER_COLS = {
    "AT (degree C)": "avg_temp_c",
    "RH (%)": "avg_humidity",
    "WS (m/s)": "avg_wind_kmh",  # note: source is m/s, converted below
    "RF (mm)": "rainfall_mm",
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_target_stations(target_cities: set) -> pd.DataFrame:
    """Load ALL stations nationwide, not just our 6 dashboard cities.
    Same reasoning as accidents: clean layer holds everything reasonably
    available so the model gets a real sample size (~450 stations across
    ~32 states); the dashboard applies the 6-city scope on top later.
    """
    stations = pd.read_csv(STATIONS_INFO_PATH)
    stations["city"] = stations["city"].str.strip()
    stations["state"] = stations["state"].str.strip()
    print(f"Processing all {len(stations)} stations nationwide "
          f"(dashboard will scope to: {sorted(target_cities)})")
    return stations


def process_station_file(station_row: pd.Series) -> pd.DataFrame | None:
    """Read one station's hourly CSV, aggregate to daily. Returns None
    if the file is missing or unreadable (some stations in the source
    dataset have near-empty files -- skip rather than fail the run)."""
    file_path = os.path.join(DATA_DIR, f"{station_row['file_name']}.csv")
    if not os.path.exists(file_path):
        print(f"  SKIP (file not found): {file_path}")
        return None

    try:
        df = pd.read_csv(file_path, usecols=["From Date"] + list(POLLUTANT_COLS) + list(WEATHER_COLS))
    except ValueError:
        # Some station files may be missing a subset of columns entirely.
        df = pd.read_csv(file_path)
        available = [c for c in list(POLLUTANT_COLS) + list(WEATHER_COLS) if c in df.columns]
        df = df[["From Date"] + available]

    if df.empty:
        return None

    df["observed_date"] = pd.to_datetime(df["From Date"], errors="coerce").dt.date
    df = df.dropna(subset=["observed_date"])

    agg = df.groupby("observed_date").mean(numeric_only=True).reset_index()
    agg["station_id"] = station_row["file_name"]
    agg["city"] = station_row["city"]
    agg["state"] = station_row["state"]
    return agg


def land_raw(engine, df: pd.DataFrame, table: str, cols: dict) -> None:
    """Bulk-insert station-day rows into the given raw table.
    Note: raw.air_quality has a station_id column; raw.weather does not
    (weather is landed at city grain) -- station_id still goes into the
    JSONB payload either way, for lineage.
    """
    date_col = "observed_at" if table == "air_quality" else "observed_date"
    records = []
    for _, row in df.iterrows():
        payload = {src_col: (None if pd.isna(row.get(src_col)) else row.get(src_col)) for src_col in cols}
        payload["station_id"] = row["station_id"]
        rec = {
            "source": "cpcb_kaggle_export",
            "city": row["city"],
            date_col: row["observed_date"],
            "payload": json.dumps(payload, default=str),
        }
        if table == "air_quality":
            rec["station_id"] = row["station_id"]
        records.append(rec)

    if not records:
        return

    station_col = ", station_id" if table == "air_quality" else ""
    station_val = ", :station_id" if table == "air_quality" else ""
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                INSERT INTO raw.{table} (source, city{station_col}, {date_col}, payload, ingested_at, ingestion_batch_id)
                VALUES (:source, :city{station_val}, :{date_col}, :payload, now(), gen_random_uuid())
                """
            ),
            records,
        )


def build_clean_air_quality(station_daily: pd.DataFrame) -> pd.DataFrame:
    rename_map = {k: v for k, v in POLLUTANT_COLS.items() if k in station_daily.columns}
    df = station_daily.rename(columns=rename_map)
    present_cols = [c for c in POLLUTANT_COLS.values() if c in df.columns]

    grouped = (
        df.groupby(["city", "observed_date"])
        .agg({**{c: "mean" for c in present_cols}, "station_id": "nunique"})
        .rename(columns={"station_id": "n_readings"})
        .reset_index()
    )
    grouped["quality_score"] = grouped[present_cols].notna().sum(axis=1) / len(POLLUTANT_COLS)
    grouped["quality_score"] = grouped["quality_score"].round(2)
    grouped["quality_flags"] = grouped[present_cols].apply(
        lambda r: [c for c in present_cols if pd.isna(r[c])], axis=1
    )
    return grouped


def build_clean_weather(station_daily: pd.DataFrame) -> pd.DataFrame:
    rename_map = {k: v for k, v in WEATHER_COLS.items() if k in station_daily.columns}
    df = station_daily.rename(columns=rename_map)
    present_cols = [c for c in WEATHER_COLS.values() if c in df.columns]

    if "avg_wind_kmh" in df.columns:
        df["avg_wind_kmh"] = df["avg_wind_kmh"] * 3.6  # m/s -> km/h

    grouped = df.groupby(["city", "observed_date"]).agg({c: "mean" for c in present_cols}).reset_index()
    grouped["quality_score"] = grouped[present_cols].notna().sum(axis=1) / len(WEATHER_COLS)
    grouped["quality_score"] = grouped["quality_score"].round(2)
    grouped["quality_flags"] = grouped[present_cols].apply(
        lambda r: [c for c in present_cols if pd.isna(r[c])], axis=1
    )
    return grouped


def load_clean(engine, df: pd.DataFrame, table: str, value_cols: list, include_n_readings: bool = False) -> None:
    cities = tuple(df["city"].unique())
    with engine.begin() as conn:
        # Idempotent re-runs: clear existing rows for these cities first.
        conn.execute(text(f"DELETE FROM clean.{table} WHERE city = ANY(:cities)"), {"cities": list(cities)})

    col_list = ", ".join(value_cols)
    placeholders = ", ".join(f":{c}" for c in value_cols)
    extra_col = ", n_readings" if include_n_readings else ""
    extra_val = ", :n_readings" if include_n_readings else ""

    records = []
    for _, row in df.iterrows():
        params = {c: (None if pd.isna(row.get(c)) else row.get(c)) for c in value_cols}
        params["city"] = row["city"]
        params["observed_date"] = row["observed_date"]
        params["quality_score"] = row["quality_score"]
        params["quality_flags"] = row["quality_flags"]
        if include_n_readings:
            n_readings = row.get("n_readings")
            params["n_readings"] = None if pd.isna(n_readings) else int(n_readings)
        records.append(params)

    # Bulk insert (list of param dicts in one execute call) rather than
    # looping single inserts -- with all states included this can be
    # 100k+ rows, and row-by-row execute would be impractically slow.
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                INSERT INTO clean.{table}
                    (city, observed_date, {col_list}, quality_score, quality_flags{extra_col}, loaded_at)
                VALUES
                    (:city, :observed_date, {placeholders}, :quality_score, :quality_flags{extra_val}, now())
                """
            ),
            records,
        )
    print(f"Loaded {len(df)} rows into clean.{table}")


def main():
    config = load_config()
    target_cities = {c["name"] for c in config["cities"]}

    stations = load_target_stations(target_cities)
    engine = get_engine()

    all_station_daily = []
    for i, (_, station_row) in enumerate(stations.iterrows(), 1):
        print(f"[{i}/{len(stations)}] Processing {station_row['file_name']} ({station_row['city']})...")
        daily = process_station_file(station_row)
        if daily is not None and not daily.empty:
            all_station_daily.append(daily)

    if not all_station_daily:
        print("No data processed -- check data_climate/ folder and stations_info.csv")
        return

    combined = pd.concat(all_station_daily, ignore_index=True)
    print(f"Total station-day rows: {len(combined)}")

    print("Landing raw station-day rows...")
    land_raw(engine, combined, "air_quality", POLLUTANT_COLS)
    land_raw(engine, combined, "weather", WEATHER_COLS)

    print("Building clean.air_quality_daily...")
    clean_aq = build_clean_air_quality(combined)
    aq_value_cols = [c for c in POLLUTANT_COLS.values() if c in clean_aq.columns]
    load_clean(engine, clean_aq, "air_quality_daily", aq_value_cols, include_n_readings=True)

    print("Building clean.weather_daily...")
    clean_wx = build_clean_weather(combined)
    wx_value_cols = [c for c in WEATHER_COLS.values() if c in clean_wx.columns]
    load_clean(engine, clean_wx, "weather_daily", wx_value_cols, include_n_readings=False)

    print("Done.")


if __name__ == "__main__":
    main()