# ✈️ Flight Delay Prediction — End-to-End MLOps Pipeline

**DSE ZG529 · Data Management for Machine Learning · BITS Pilani WILP — Experiential Learning Assignment**

An end-to-end machine-learning pipeline that predicts whether an Indian domestic flight will depart more than
15 minutes late, built to demonstrate the data-management practices that make ML reproducible, traceable and
servable:

| # | Stage | What it demonstrates | Syllabus |
|---|-------|----------------------|----------|
| ① | **Data versioning (DVC)** | Content-addressed dataset versions, lineage, diffs, reproducibility | CS 6, CS 9 · Lab Exp 4 |
| ② | **Feature store** | Offline/online stores, point-in-time joins, leakage prevention, no training/serving skew | CS 8, CS 11 · Lab Exp 4 |
| ③ | **Experiment tracking & registry (MLflow)** | Params, metrics, artifacts, dataset lineage, registered versions, `@champion` alias | CS 12 · Lab Exp 3 |
| ④ | **Prediction service** | Registry-resolved model + online features, REST API, per-prediction lineage | CS 5, CS 16 · Lab Exp 4 |

## Architecture

```
 Source extracts ──► ① Versioned datasets (MD5 cache + .dvc pointers in Git)
                          │ v1.0 ─► v1.1 ─► v2.0 ─► v3.0
                          ▼
                     ② Feature store ── offline (Parquet, full history) ──► point-in-time training set
                          │                                                    │
                          └── materialize ──► online (SQLite, latest values)    ▼
                                                    │                ③ MLflow tracking ─► Model registry (@champion)
                                                    ▼                                          │
                                              ④ Prediction service (FastAPI / Streamlit) ◄─────┘
```

## Dataset

Synthetic but realistic Indian domestic flight operations for 2025 (8 airports, 6 carriers, ~35k flights),
generated deterministically from a seed so every version is reproducible. Delay mechanics include carrier and
hub-congestion effects, evening knock-on delays, weather, winter fog at north-Indian airports, and persistent
airport-level disruption — which is what makes recent airport delay history (a feature-store feature) predictive.

| Version | Content | Data-management event |
|---------|---------|-----------------------|
| v1.0 | Jan–Jun 2025 raw extract | 3% duplicate rows, 4% missing weather readings |
| v1.1 | v1.0 cleaned | De-duplicated, weather imputed by airport median |
| v2.0 | Jan–Sep 2025 | New carrier QP + new column `aircraft_type` (schema change) |
| v3.0 | Full year 2025 | Winter fog season (distribution drift) |

## Key results

| Experiment | ROC-AUC |
|------------|---------|
| Base features only (v2.0, gradient boosting) | ~0.61 |
| + feature-store features, point-in-time correct | ~0.66–0.67 |
| Naive same-day join — offline test | ~0.78 (**inflated by leakage**) |
| Naive same-day join — in production | ~0.63 (worse than the honest model) |

## Run it

```bash
pip install -r requirements-dev.txt
python -m src.pipeline          # or: dvc repro
streamlit run app.py            # interactive demo
uvicorn serving.api:app --port 8000   # REST API → http://localhost:8000/docs
mlflow ui --backend-store-uri sqlite:///mlflow.db   # MLflow UI
pytest -q
```

On first launch the Streamlit app builds everything automatically (~40 s).

## Repository layout

```
app.py                 Streamlit app (one tab per pipeline stage)
src/data_generation.py Synthetic data + version definitions
src/versioning.py      DVC-style content-addressed versioning, checkout, diff
src/feature_store.py   Feature views, offline/online stores, point-in-time join, materialize
src/train.py           Training, MLflow tracking, leakage experiment, model registry
src/predict.py         Prediction service core with lineage
src/pipeline.py        Stage runner used by dvc.yaml
serving/api.py         FastAPI REST service
dvc.yaml / params.yaml DVC pipeline definition and parameters
tests/                 Reproducibility and point-in-time correctness tests
docs/VIDEO_SCRIPT.md   1-minute script for each team member
```

## Team

Edit names in `src/config.py` → `TEAM`.

| Member | Part |
|--------|------|
| Member 1 — Rohesen Raj (2025NS04108) | Data Versioning |
| Member 2 | Feature Engineering & Feature Store |
| Member 3 | Experiment Tracking & Model Registry |
| Member 4 | Prediction Service & Lineage |
