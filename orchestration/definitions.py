"""Dagster entry point. Run locally with:
    dagster dev -f orchestration/definitions.py

Each source gets its own asset so failures/retries are isolated and
you can see lineage (raw -> clean -> analytics) in the Dagster UI --
this graph itself is a good interview visual.
"""
from dagster import Definitions, asset, AssetExecutionContext


@asset
def raw_accidents(context: AssetExecutionContext) -> None:
    """Pull MoRTH accident records from data.gov.in into raw.accidents."""
    # TODO: instantiate DataGovInClient, fetch_all(), bulk insert into raw.accidents
    context.log.info("raw_accidents: not yet implemented")


@asset
def raw_air_quality(context: AssetExecutionContext) -> None:
    """Pull CPCB AQI records into raw.air_quality."""
    context.log.info("raw_air_quality: not yet implemented")


@asset
def raw_weather(context: AssetExecutionContext) -> None:
    """Pull IMD weather records into raw.weather."""
    context.log.info("raw_weather: not yet implemented")


@asset(deps=[raw_accidents])
def clean_accidents(context: AssetExecutionContext) -> None:
    """Validate + type + dedup raw.accidents -> clean.accidents."""
    context.log.info("clean_accidents: not yet implemented")


@asset(deps=[raw_air_quality])
def clean_air_quality(context: AssetExecutionContext) -> None:
    context.log.info("clean_air_quality: not yet implemented")


@asset(deps=[raw_weather])
def clean_weather(context: AssetExecutionContext) -> None:
    context.log.info("clean_weather: not yet implemented")


@asset(deps=[clean_accidents, clean_air_quality, clean_weather])
def analytics_facts(context: AssetExecutionContext) -> None:
    """Aggregate clean layer up to city-month grain and load the
    analytics star schema (dim_city, dim_date, fact_* tables)."""
    context.log.info("analytics_facts: not yet implemented")


defs = Definitions(
    assets=[
        raw_accidents,
        raw_air_quality,
        raw_weather,
        clean_accidents,
        clean_air_quality,
        clean_weather,
        analytics_facts,
    ]
)
