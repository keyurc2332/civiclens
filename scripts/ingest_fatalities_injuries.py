"""Ingest MoRTH fatalities and injuries CSVs (manually downloaded from
data.gov.in, stored locally as data_climate/accident_killed.csv and
data_climate/accident_injured.csv), reshape wide -> long, and update
the existing clean.accidents rows to populate the fatalities and
injuries columns.

Run with:
    python scripts/ingest_fatalities_injuries.py
"""
import os
import sys

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.warehouse.db import get_engine

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_climate")
KILLED_PATH = os.path.join(DATA_DIR, "accident_killed.csv")
INJURED_PATH = os.path.join(DATA_DIR, "accident_injured.csv")

# Map the messy column names from the CSV to (year, metric) pairs.
# Columns like "State / UT - wise Total Number of Persons Killed in Road Accidents during - 2021"
# get extracted to year=2021, metric="Killed".
# We only care about the raw totals, not the per-lakh or per-vehicle rates.
YEAR_COLS_KILLED = {
    "State / UT - wise Total Number of Persons Killed in Road Accidents during - 2021": 2021,
    "State / UT - wise Total Number of Persons Killed in Road Accidents during - 2022": 2022,
    "State / UT - wise Total Number of Persons Killed in Road Accidents during - 2023": 2023,
    "State / UT - wise Total Number of Persons Killed in Road Accidents during - 2024 - Number": 2024,
}

YEAR_COLS_INJURED = {
    "State / UT - wise Total Number of Persons Injured in Road Accidents during - 2021": 2021,
    "State / UT - wise Total Number of Persons Injured in Road Accidents during - 2022": 2022,
    "State / UT - wise Total Number of Persons Injured in Road Accidents during - 2023": 2023,
    "State / UT - wise Total Number of Persons Injured in Road Accidents during - 2024 - Number": 2024,
}


def load_and_reshape(csv_path: str, year_cols: dict, metric_name: str) -> pd.DataFrame:
    """Read CSV, extract year columns, reshape wide -> long."""
    df = pd.read_csv(csv_path)
    df["State/UT"] = df["State/UT"].str.strip()

    rows = []
    for col, year in year_cols.items():
        if col not in df.columns:
            print(f"WARNING: column '{col}' not found in {csv_path} -- skipping")
            continue
        for _, row in df.iterrows():
            state = row["State/UT"]
            value = row[col]
            # Skip NaN, non-numeric, or error values
            try:
                value = int(float(value)) if pd.notna(value) else None
            except (ValueError, TypeError):
                value = None
            rows.append({"state": state, "year": year, metric_name: value})

    return pd.DataFrame(rows)


def main():
    print("Loading fatalities...")
    killed_df = load_and_reshape(KILLED_PATH, YEAR_COLS_KILLED, "fatalities")
    print(f"  Loaded {len(killed_df)} state-year rows")

    print("Loading injuries...")
    injured_df = load_and_reshape(INJURED_PATH, YEAR_COLS_INJURED, "injuries")
    print(f"  Loaded {len(injured_df)} state-year rows")

    merged = pd.merge(killed_df, injured_df, on=["state", "year"], how="outer")
    print(f"Merged: {len(merged)} state-year rows")

    engine = get_engine()

    # Update clean.accidents to fill in fatalities and injuries.
    # We're upserting: if a (state, year) row doesn't exist yet, INSERT;
    # if it does, UPDATE the two new columns.
    with engine.begin() as conn:
        for _, row in merged.iterrows():
            state = row["state"]
            year = int(row["year"])
            fatalities = None if pd.isna(row["fatalities"]) else int(row["fatalities"])
            injuries = None if pd.isna(row["injuries"]) else int(row["injuries"])

            conn.execute(
                text(
                    """
                    UPDATE clean.accidents
                    SET fatalities = :fatalities, injuries = :injuries, loaded_at = now()
                    WHERE state = :state AND year = :year AND month IS NULL
                    """
                ),
                {"state": state, "year": year, "fatalities": fatalities, "injuries": injuries},
            )

    print(f"Updated clean.accidents with fatalities and injuries")
    print("Done.")


if __name__ == "__main__":
    main()