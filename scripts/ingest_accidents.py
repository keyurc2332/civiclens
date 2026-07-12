"""Phase 1 ingestion script: MoRTH accident counts (2021-2024) ->
raw.accidents -> clean.accidents.

Run with:
    python scripts/ingest_accidents.py

What it does, in order:
1. Fetch all records from the data.gov.in resource (small dataset,
   one page covers it -- ~36 states/UTs).
2. Land each raw record as-is into raw.accidents (JSONB payload),
   tagged with a batch id, so we always have the untouched source.
3. Reshape the wide format (one column per year) into long format
   (one row per state-year), because that's what our star schema
   and modeling grain expect.
4. Filter down to the states relevant to our 6 target cities.
5. Compute a simple completeness quality_score per row.
6. Upsert into clean.accidents.

This resource only has *total accident counts*, not fatalities/
injuries -- those live in separate sibling resources on data.gov.in
and are a fast-follow, not a Phase-1 blocker.
"""
import json
import os
import sys
import uuid

import pandas as pd
import yaml
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.data_gov_client import DataGovInClient
from src.warehouse.db import get_engine

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

# Maps target year -> the exact field id data.gov.in gave us for that
# year's total accident count. Hardcoded because it changes only when
# the dataset's column set changes (e.g. a new year is added upstream).
YEAR_FIELD_MAP = {
    2021: "state___ut___wise_total_number_of_road_accidents_during___2021",
    2022: "state___ut___wise_total_number_of_road_accidents_during___2022",
    2023: "state___ut___wise_total_number_of_road_accidents_during___2023",
    2024: "state___ut___wise_total_number_of_road_accidents_during___2024___number",
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def land_raw(engine, records: list[dict], resource_id: str, batch_id: uuid.UUID) -> None:
    """Insert every raw API record as-is into raw.accidents."""
    with engine.begin() as conn:
        for rec in records:
            conn.execute(
                text(
                    """
                    INSERT INTO raw.accidents
                        (source, resource_id, state, year, month, payload, ingested_at, ingestion_batch_id)
                    VALUES
                        (:source, :resource_id, :state, NULL, NULL, :payload, now(), :batch_id)
                    """
                ),
                {
                    "source": "data.gov.in",
                    "resource_id": resource_id,
                    "state": rec.get("state_ut"),
                    "payload": json.dumps(rec),
                    "batch_id": str(batch_id),
                },
            )
    print(f"Landed {len(records)} raw records into raw.accidents (batch {batch_id})")


def reshape_to_long(records: list[dict]) -> pd.DataFrame:
    """Wide (one column per year) -> long (one row per state-year)."""
    rows = []
    for rec in records:
        state = rec.get("state_ut")
        if not state:
            continue
        for year, field_id in YEAR_FIELD_MAP.items():
            total = rec.get(field_id)
            rows.append({"state": state, "year": year, "total_accidents": total})
    return pd.DataFrame(rows)


def compute_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quality_score"] = df["total_accidents"].notna().astype(float)
    df["quality_flags"] = df["total_accidents"].apply(
        lambda v: [] if pd.notna(v) else ["missing_total_accidents"]
    )
    return df


def load_clean(engine, df: pd.DataFrame) -> None:
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text(
                    """
                    INSERT INTO clean.accidents
                        (state, year, month, total_accidents, fatalities, injuries,
                         quality_score, quality_flags, loaded_at)
                    VALUES
                        (:state, :year, NULL, :total_accidents, NULL, NULL,
                         :quality_score, :quality_flags, now())
                    ON CONFLICT (state, year, month) DO UPDATE SET
                        total_accidents = EXCLUDED.total_accidents,
                        quality_score = EXCLUDED.quality_score,
                        quality_flags = EXCLUDED.quality_flags,
                        loaded_at = now()
                    """
                ),
                {
                    "state": row["state"],
                    "year": int(row["year"]),
                    "total_accidents": None if pd.isna(row["total_accidents"]) else int(row["total_accidents"]),
                    "quality_score": row["quality_score"],
                    "quality_flags": row["quality_flags"],
                },
            )
    print(f"Upserted {len(df)} rows into clean.accidents")


def main():
    config = load_config()
    resource_id = config["sources"]["data_gov_in"]["accident_resource_id"]
    target_states = {c["state"] for c in config["cities"]}

    client = DataGovInClient(resource_id=resource_id)
    batch_id = client.new_batch_id()

    print("Fetching records from data.gov.in...")
    records = client.fetch_all()
    print(f"Fetched {len(records)} records")

    # Surface the exact state/UT names data.gov.in uses, so mismatches
    # (e.g. "Delhi" vs "NCT of Delhi") are visible immediately instead
    # of silently dropping a target city.
    all_state_names = {r.get("state_ut") for r in records if r.get("state_ut")}
    unmatched_targets = target_states - all_state_names
    if unmatched_targets:
        print(
            f"WARNING: these target states from config.yaml were NOT found "
            f"verbatim in the source data: {unmatched_targets}. "
            f"Check the exact naming below and update config.yaml if needed."
        )
        print("All state/UT names in source data:", sorted(all_state_names))

    engine = get_engine()
    land_raw(engine, records, resource_id, batch_id)

    long_df = reshape_to_long(records)
    filtered_df = long_df[long_df["state"].isin(target_states)]
    print(f"Filtered to {len(filtered_df)} rows for target states: {sorted(target_states)}")

    scored_df = compute_quality_score(filtered_df)
    load_clean(engine, scored_df)


if __name__ == "__main__":
    main()