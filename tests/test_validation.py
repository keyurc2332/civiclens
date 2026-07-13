import pandas as pd

from src.validation.checks import flag_duplicates, quality_score


def test_quality_score_all_present():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    scores = quality_score(df, required_cols=["a", "b"])
    assert (scores == 1.0).all()


def test_quality_score_partial():
    df = pd.DataFrame({"a": [1, None], "b": [3, 4]})
    scores = quality_score(df, required_cols=["a", "b"])
    assert scores.iloc[0] == 1.0
    assert scores.iloc[1] == 0.5


def test_flag_duplicates():
    df = pd.DataFrame({"state": ["MH", "MH", "KA"], "year": [2023, 2023, 2023]})
    dupes = flag_duplicates(df, key_cols=["state", "year"])
    assert dupes.tolist() == [False, True, False]
