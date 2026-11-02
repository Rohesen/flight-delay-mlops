"""Security and robustness tests.  Run:  python -m src.pipeline && pytest -q"""
import pandas as pd
import pytest

from src import versioning
from src.feature_store import FeatureStore
from src.security import check_admin, cooldown_ok, sanitize_text, valid_tag


def test_tag_validation_blocks_path_traversal():
    assert valid_tag("v2.0") == "v2.0"
    for bad in ["../../etc/passwd", "v2.0/../x", "", "v2", "v2.0;rm"]:
        with pytest.raises(ValueError):
            valid_tag(bad)
    with pytest.raises(ValueError):
        versioning.get_meta("../../secrets")


def test_feature_view_whitelist_blocks_sql_injection():
    with pytest.raises(ValueError):
        FeatureStore().online_table("origin_stats; DROP TABLE route_stats")


def test_sanitize_text_strips_markup_and_limits_length():
    out = sanitize_text('<img src=x onerror=alert(1)>' + "A" * 200)
    assert "<" not in out and ">" not in out and "=" not in out
    assert len(out) <= 60
    assert sanitize_text("   ") == "run"


def test_cooldown_and_admin_check():
    state = {}
    assert cooldown_ok(state, "train", 60)[0] is True
    allowed, left = cooldown_ok(state, "train", 60)
    assert allowed is False and left > 0
    assert check_admin("secret", "secret") is True
    assert check_admin("wrong", "secret") is False
    assert check_admin("x", None) is False          # disabled when no password is configured


def test_api_validation_and_prediction():
    from fastapi.testclient import TestClient
    from serving.api import app

    c = TestClient(app)
    ok = {"carrier": "6E", "origin": "DEL", "dest": "BOM", "departure": "2025-10-01T18:30:00", "weather_index": 1.0}
    r = c.post("/predict", json=ok)
    assert r.status_code == 200 and 0 <= r.json()["delay_probability"] <= 1
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    for bad in [{**ok, "origin": "XXX"}, {**ok, "dest": "DEL"}, {**ok, "departure": "not-a-date"},
                {**ok, "weather_index": 99}, {**ok, "carrier": "<script>"}, {**ok, "extra": 1}]:
        assert c.post("/predict", json=bad).status_code == 422


def test_batch_predict_matches_single_predictions():
    from src.predict import batch_predict, predict
    df = pd.DataFrame([{"carrier": "SG", "origin": "DEL", "dest": "BOM",
                        "departure": pd.Timestamp("2025-10-01 19:00"), "weather_index": 1.2},
                       {"carrier": "6E", "origin": "BLR", "dest": "GOI",
                        "departure": pd.Timestamp("2025-10-01 08:00"), "weather_index": 0.2}])
    b = batch_predict(df)
    for i, r in df.iterrows():
        single = predict(r.carrier, r.origin, r.dest, r.departure, r.weather_index)["delay_probability"]
        assert abs(b.loc[i, "delay_probability"] - single) < 1e-3
