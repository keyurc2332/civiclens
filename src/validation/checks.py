"""Data quality checks applied when promoting raw -> clean.

Keep this scaled to the project: we don't need a full Great Expectations
suite for a handful of sources. Pandera schemas + a small quality-scoring
function cover schema drift, nulls, and dupes, and are easy to explain
in an interview.
"""
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

accident_schema = DataFrameSchema(
    {
        "state": Column(str, nullable=False),
        "year": Column(int, Check.in_range(2000, 2035), nullable=False),
        "month": Column(int, Check.in_range(1, 12), nullable=True),
        "total_accidents": Column(int, Check.ge(0), nullable=True),
        "fatalities": Column(int, Check.ge(0), nullable=True),
        "injuries": Column(int, Check.ge(0), nullable=True),
    },
    strict=False,  # allow extra columns without failing (schema drift tolerance)
)


def quality_score(df: pd.DataFrame, required_cols: list[str]) -> pd.Series:
    """Simple, explainable completeness score per row: fraction of
    required columns that are non-null. 1.0 = fully complete.
    """
    present = df[required_cols].notna().sum(axis=1)
    return (present / len(required_cols)).round(2)


def flag_duplicates(df: pd.DataFrame, key_cols: list[str]) -> pd.Series:
    return df.duplicated(subset=key_cols, keep="first")


def null_percentage_report(df: pd.DataFrame) -> pd.Series:
    """Per-column null percentage -- log this on every ingestion run so
    schema drift / upstream data problems surface immediately instead
    of silently degrading model quality later."""
    return (df.isna().mean() * 100).round(1)
