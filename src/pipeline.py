"""End-to-end pipeline runner (also wired as DVC stages in dvc.yaml).

    python -m src.pipeline            # run every stage
    python -m src.pipeline --stage data|features|train
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import versioning
from .config import RAW_DIR, ROOT
from .data_generation import VERSION_SPECS, build_version, generate_master
from .feature_store import FeatureStore, compute_feature_tables

PARENTS = {"v1.0": None, "v1.1": "v1.0", "v2.0": "v1.1", "v3.0": "v2.0"}
SERVING_VERSION = "v2.0"     # data the online store is materialized from


def stage_data() -> list[dict]:
    master = generate_master()
    metas = []
    for tag, msg in VERSION_SPECS.items():
        df = build_version(master, tag)
        metas.append(versioning.commit_version(df, tag, msg, PARENTS[tag]))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    build_version(master, SERVING_VERSION).to_csv(RAW_DIR / "flights.csv", index=False)
    return metas


def stage_features(tag: str = SERVING_VERSION) -> dict:
    raw = versioning.checkout(tag).drop_duplicates(subset="flight_id")
    store = FeatureStore()
    store.write_offline(compute_feature_tables(raw), tag)
    return store.materialize()


def stage_train() -> dict:
    from .train import register_run, train_and_log
    results = [
        train_and_log("v1.0", "Logistic Regression", {"C": 1.0}, use_store=False,
                      run_name="baseline | raw v1.0 | no feature store"),
        train_and_log("v1.1", "Logistic Regression", {"C": 1.0}, run_name="LR | v1.1 | feature store"),
        train_and_log("v2.0", "Logistic Regression", {"C": 1.0}, run_name="LR | v2.0 | feature store"),
        train_and_log("v2.0", "Random Forest", {"n_estimators": 150, "max_depth": 12},
                      run_name="RF | v2.0 | feature store"),
        train_and_log("v2.0", "Gradient Boosting", {"learning_rate": 0.08, "max_depth": 6},
                      run_name="GBM | v2.0 | feature store"),
    ]
    # Only compare runs evaluated on the same data version (same holdout window)
    best = max(results[2:], key=lambda r: r["roc_auc"])
    reg = register_run(best["run_id"], promote=True,
                       description="Best ROC-AUC on time-based holdout; promoted by pipeline")
    (ROOT / "metrics.json").write_text(json.dumps(
        {"champion_version": reg["version"], **{k: best[k] for k in ("roc_auc", "f1", "accuracy")}}, indent=2))
    return {"runs": results, "champion": reg}


def run_all() -> dict:
    stage_data()
    stage_features()
    return stage_train()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["data", "features", "train", "all"], default="all")
    s = ap.parse_args().stage
    out = {"data": stage_data, "features": stage_features, "train": stage_train, "all": run_all}[s]()
    print(json.dumps(out, indent=2, default=str))
