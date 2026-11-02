"""Run with:  pytest -q   (after `python -m src.pipeline`)"""
import pandas as pd

from src import versioning
from src.data_generation import build_version, generate_master
from src.feature_store import FeatureStore, compute_feature_tables


def test_versions_are_reproducible():
    m1 = generate_master(end="2025-01-20")
    m2 = generate_master(end="2025-01-20")
    assert versioning.md5_of(m1) == versioning.md5_of(m2)


def test_any_change_creates_new_hash():
    m = generate_master(end="2025-01-20")
    changed = m.copy()
    changed.loc[0, "weather_index"] += 0.001
    assert versioning.md5_of(m) != versioning.md5_of(changed)


def test_cleaned_version_has_no_duplicates_or_nulls():
    v = build_version(generate_master(end="2025-02-28"), "v1.1")
    assert v["flight_id"].duplicated().sum() == 0
    assert v["weather_index"].isna().sum() == 0


def test_point_in_time_join_never_uses_future():
    df = generate_master(end="2025-02-28")
    tables = compute_feature_tables(df)
    out = FeatureStore().get_historical_features(df, tables=tables)
    # For each flight, the joined 7d rate must equal the rate computed from strictly earlier days
    row = out.iloc[len(out) // 2]
    day = row["scheduled_departure"].normalize()
    past = df[(df["origin"] == row["origin"]) & (df["scheduled_departure"] < day)
              & (df["scheduled_departure"] >= day - pd.Timedelta(days=7))]
    assert abs(past["is_delayed"].mean() - row["origin_delay_rate_7d"]) < 1e-9
