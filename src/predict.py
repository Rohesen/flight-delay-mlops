"""Prediction service core: champion model from the registry + online features.

Request flow
  1. client sends raw flight info (carrier, origin, dest, departure time, weather)
  2. service looks up fresh features from the ONLINE store (same definitions
     used to build the training set -> no training/serving skew)
  3. champion model (registry alias) scores the assembled feature vector
  4. response carries the prediction AND lineage: model version, run id,
     data version and feature timestamps — so any prediction can be audited.
"""
from __future__ import annotations

import time
from functools import lru_cache

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from .config import CHAMPION_ALIAS, REGISTERED_MODEL
from .data_generation import AIRPORTS, _haversine
from .feature_store import FeatureStore
from .train import setup_mlflow


@lru_cache(maxsize=8)
def _load_cached(uri: str, version: str):
    return mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}/{version}")


def champion_info() -> dict:
    setup_mlflow()
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS)
    run = client.get_run(mv.run_id) if mv.run_id else None
    return {"model": REGISTERED_MODEL, "version": mv.version, "alias": CHAMPION_ALIAS,
            "run_id": mv.run_id, "data_version": mv.tags.get("data_version"),
            "model_type": run.data.params.get("model_type") if run else None,
            "roc_auc": run.data.metrics.get("roc_auc") if run else None,
            "data_md5": run.data.params.get("data_md5") if run else None}


def predict(carrier: str, origin: str, dest: str, departure: pd.Timestamp,
            weather_index: float, store: FeatureStore | None = None) -> dict:
    t0 = time.perf_counter()
    store = store or FeatureStore()
    info = champion_info()
    model = _load_cached(setup_mlflow(), str(info["version"]))
    feats = store.get_online_features(origin, dest, carrier)
    row = {
        "carrier": carrier, "origin": origin, "dest": dest,
        "dep_hour": departure.hour, "day_of_week": departure.dayofweek, "month": departure.month,
        "distance_km": round(_haversine(origin, dest), 1), "weather_index": weather_index,
        **{k: v for k, v in feats.items() if not k.startswith("_")},
    }
    X = pd.DataFrame([row]).astype({k: float for k in row if k not in ("carrier", "origin", "dest")})
    proba = float(model.predict_proba(X)[0, 1])
    return {
        "delay_probability": round(proba, 4),
        "prediction": "DELAYED (>15 min)" if proba >= 0.5 else "ON TIME",
        "features": row,
        "feature_timestamps": {k: v for k, v in feats.items() if k.startswith("_")},
        "lineage": info,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def batch_predict(df: pd.DataFrame, store: FeatureStore | None = None) -> pd.DataFrame:
    out = []
    for r in df.itertuples():
        p = predict(r.carrier, r.origin, r.dest, pd.Timestamp(r.departure), float(r.weather_index), store)
        out.append({"carrier": r.carrier, "route": f"{r.origin}-{r.dest}", "departure": r.departure,
                    "delay_probability": p["delay_probability"], "prediction": p["prediction"]})
    return pd.DataFrame(out)


AIRPORT_CODES = list(AIRPORTS)


def risk_grid(carrier: str, origin: str, dest: str, day: pd.Timestamp, hours, weathers,
              store: FeatureStore | None = None):
    """P(delay) over a departure-hour x weather grid in ONE batched model call."""
    store = store or FeatureStore()
    info = champion_info()
    model = _load_cached(setup_mlflow(), str(info["version"]))
    feats = {k: v for k, v in store.get_online_features(origin, dest, carrier).items() if not k.startswith("_")}
    rows = []
    for w in weathers:
        for h in hours:
            dep = pd.Timestamp(day) + pd.Timedelta(hours=int(h))
            rows.append({"carrier": carrier, "origin": origin, "dest": dest, "dep_hour": dep.hour,
                         "day_of_week": dep.dayofweek, "month": dep.month,
                         "distance_km": round(_haversine(origin, dest), 1), "weather_index": float(w), **feats})
    X = pd.DataFrame(rows)
    X = X.astype({k: float for k in X.columns if k not in ("carrier", "origin", "dest")})
    return model.predict_proba(X)[:, 1].reshape(len(weathers), len(hours))
