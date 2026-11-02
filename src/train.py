"""Model training with full MLflow experiment tracking and model registry.

Each run records: data version + its MD5 (lineage to DVC), feature list,
hyper-parameters, metrics, the fitted sklearn pipeline (with signature and
input example) and evaluation artifacts. The best run can be registered and
promoted with the `champion` alias, which the prediction service loads.
"""
from __future__ import annotations

import json
import time
import warnings

import os
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import versioning
from .config import (BASE_CATEGORICAL, BASE_NUMERIC, CHAMPION_ALIAS, EXPERIMENT_NAME,
                     MLFLOW_ARTIFACTS, MLFLOW_DB, REGISTERED_MODEL, SEED, STORE_FEATURES, TARGET)
from .feature_store import FeatureStore, compute_feature_tables

warnings.filterwarnings("ignore")

MODEL_TYPES = {
    "Logistic Regression": lambda p: LogisticRegression(C=p.get("C", 1.0), max_iter=1000),
    "Random Forest": lambda p: RandomForestClassifier(
        n_estimators=p.get("n_estimators", 150), max_depth=p.get("max_depth", 12),
        min_samples_leaf=p.get("min_samples_leaf", 5), n_jobs=-1, random_state=SEED),
    "Gradient Boosting": lambda p: HistGradientBoostingClassifier(
        learning_rate=p.get("learning_rate", 0.08), max_depth=p.get("max_depth", 6),
        max_iter=p.get("max_iter", 200), random_state=SEED),
}


def setup_mlflow() -> str:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow.get_tracking_uri()


def feature_columns(use_store: bool) -> tuple[list[str], list[str]]:
    num = BASE_NUMERIC + (STORE_FEATURES if use_store else [])
    return BASE_CATEGORICAL, num


def build_pipeline(model_type: str, params: dict, use_store: bool) -> Pipeline:
    cat, num = feature_columns(use_store)
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if model_type == "Logistic Regression":
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", Pipeline(num_steps), num),
    ])
    return Pipeline([("prep", pre), ("model", MODEL_TYPES[model_type](params))])


def build_dataset(tag: str, leaky: bool = False) -> pd.DataFrame:
    """Checkout a data version and attach features (point-in-time or naive)."""
    raw = versioning.checkout(tag).drop_duplicates(subset="flight_id")
    raw = raw.dropna(subset=[TARGET]).sort_values("scheduled_departure").reset_index(drop=True)
    if leaky:
        return FeatureStore.naive_join(raw, compute_feature_tables(raw, leaky=True))
    return FeatureStore().get_historical_features(raw, tables=compute_feature_tables(raw))


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    cut = df["scheduled_departure"].quantile(1 - test_frac)
    return df[df["scheduled_departure"] < cut], df[df["scheduled_departure"] >= cut]


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred), 4),
    }


def _log_sklearn(pipe, sig, example):
    """Log the pipeline; works across MLflow 2.x and 3.x serialization APIs."""
    kw = dict(signature=sig, input_example=example)
    try:   # MLflow >= 3.x (skops by default -> pin cloudpickle for portability)
        return mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle", **kw)
    except TypeError:
        return mlflow.sklearn.log_model(pipe, artifact_path="model", **kw)


def train_and_log(tag: str, model_type: str, params: dict | None = None, use_store: bool = True,
                  run_name: str | None = None, author: str = "team") -> dict:
    setup_mlflow()
    params = params or {}
    df = build_dataset(tag)
    train, test = time_split(df)
    cat, num = feature_columns(use_store)
    cols = cat + num
    pipe = build_pipeline(model_type, params, use_store)
    meta = versioning.get_meta(tag)
    with mlflow.start_run(run_name=run_name or f"{model_type} | {tag}") as run:
        t0 = time.time()
        pipe.fit(train[cols], train[TARGET])
        train_time = time.time() - t0
        metrics = evaluate(pipe, test[cols], test[TARGET])
        metrics["train_time_s"] = round(train_time, 2)
        mlflow.log_params({"model_type": model_type, "data_version": tag,
                           "data_md5": meta["outs"][0]["md5"], "use_feature_store": use_store,
                           "n_features": len(cols), "train_rows": len(train), "test_rows": len(test),
                           "split": "time-based 80/20", **{f"hp_{k}": v for k, v in params.items()}})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"data_version": tag, "author": author, "leakage_safe": "true",
                         "stage": "experiment"})
        try:
            mlflow.log_input(mlflow.data.from_pandas(train[cols + [TARGET]].head(2000), name=f"flights-{tag}",
                                                     digest=meta["outs"][0]["md5"][:8], targets=TARGET),
                             context="training")
        except Exception:
            pass
        cm = confusion_matrix(test[TARGET], pipe.predict(test[cols])).tolist()
        mlflow.log_dict({"features": cols, "categorical": cat, "numeric": num}, "features.json")
        mlflow.log_dict({"confusion_matrix": cm, "labels": ["on_time", "delayed"]}, "evaluation.json")
        example = test[cols].head(3)
        sig = infer_signature(example, pipe.predict_proba(example)[:, 1])
        info = _log_sklearn(pipe, sig, example)
        mlflow.set_tag("model_uri", info.model_uri)
        return {"run_id": run.info.run_id, "model_uri": info.model_uri, **metrics}


def leakage_experiment(tag: str, model_type: str = "Gradient Boosting") -> dict:
    """Train on naive (leaky) features; score offline with the same leaky
    features vs. with the point-in-time features production will really get."""
    setup_mlflow()
    leaky = build_dataset(tag, leaky=True)
    pit = build_dataset(tag, leaky=False)
    cat, num = feature_columns(True)
    cols = cat + num
    tr_l, te_l = time_split(leaky)
    _, te_p = time_split(pit)
    tr_p, _ = time_split(pit)

    m_leaky = build_pipeline(model_type, {}, True).fit(tr_l[cols], tr_l[TARGET])
    m_pit = build_pipeline(model_type, {}, True).fit(tr_p[cols], tr_p[TARGET])
    res = {
        "leaky_offline": evaluate(m_leaky, te_l[cols], te_l[TARGET]),
        "leaky_in_production": evaluate(m_leaky, te_p[cols], te_p[TARGET]),
        "pit_offline": evaluate(m_pit, te_p[cols], te_p[TARGET]),
    }
    with mlflow.start_run(run_name=f"LEAKAGE DEMO naive join | {tag}"):
        mlflow.log_params({"model_type": model_type, "data_version": tag, "feature_join": "naive same-day"})
        mlflow.log_metrics({"roc_auc": res["leaky_offline"]["roc_auc"],
                            "roc_auc_in_production": res["leaky_in_production"]["roc_auc"]})
        mlflow.set_tags({"leakage_safe": "false", "stage": "rejected", "data_version": tag})
    return res


def runs_dataframe() -> pd.DataFrame:
    setup_mlflow()
    df = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME], order_by=["start_time DESC"])
    return df


def register_run(run_id: str, promote: bool = True, description: str = "") -> dict:
    setup_mlflow()
    client = MlflowClient()
    run = client.get_run(run_id)
    uri = run.data.tags.get("model_uri", f"runs:/{run_id}/model")
    mv = mlflow.register_model(uri, REGISTERED_MODEL)
    client.set_model_version_tag(REGISTERED_MODEL, mv.version, "data_version",
                                 run.data.params.get("data_version", ""))
    client.set_model_version_tag(REGISTERED_MODEL, mv.version, "roc_auc",
                                 str(run.data.metrics.get("roc_auc", "")))
    client.set_model_version_tag(REGISTERED_MODEL, mv.version, "run_id", run_id)
    if description:
        client.update_model_version(REGISTERED_MODEL, mv.version, description=description)
    if promote:
        client.set_registered_model_alias(REGISTERED_MODEL, CHAMPION_ALIAS, mv.version)
    return {"version": mv.version, "run_id": run_id}


def registry_versions() -> pd.DataFrame:
    setup_mlflow()
    client = MlflowClient()
    try:
        champ = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS).version
    except Exception:
        champ = None
    rows = []
    for mv in client.search_model_versions(f"name='{REGISTERED_MODEL}'"):
        rows.append({"version": int(mv.version), "alias": CHAMPION_ALIAS if mv.version == champ else "",
                     "data_version": mv.tags.get("data_version"), "roc_auc": mv.tags.get("roc_auc"),
                     "run_id": mv.run_id or mv.tags.get("run_id"),
                     "created": pd.to_datetime(mv.creation_timestamp, unit="ms").strftime("%Y-%m-%d %H:%M")})
    return pd.DataFrame(rows).sort_values("version", ascending=False) if rows else pd.DataFrame()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2.0")
    ap.add_argument("--model", default="Gradient Boosting")
    a = ap.parse_args()
    r = train_and_log(a.version, a.model)
    print(json.dumps(r, indent=2))
    print(register_run(r["run_id"]))


def promote(version: int | str) -> None:
    """Move the `champion` alias to a registered model version."""
    setup_mlflow()
    MlflowClient().set_registered_model_alias(REGISTERED_MODEL, CHAMPION_ALIAS, str(version))
