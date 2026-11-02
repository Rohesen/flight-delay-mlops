"""Central configuration for the Flight Delay MLOps pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
VERSION_STORE = DATA_DIR / ".versions"          # content-addressed object store (DVC-style cache)
VERSION_META = DATA_DIR / "versions"            # one .dvc-style metadata file per version
FEATURE_DIR = DATA_DIR / "feature_store"
OFFLINE_DIR = FEATURE_DIR / "offline"
ONLINE_DB = FEATURE_DIR / "online_store.db"
MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_ARTIFACTS = ROOT / "mlartifacts"

EXPERIMENT_NAME = "flight-delay-prediction"
REGISTERED_MODEL = "flight-delay-classifier"
CHAMPION_ALIAS = "champion"
SEED = 42

TARGET = "is_delayed"  # departure delay > 15 minutes (DGCA / FAA convention)

# Raw columns used directly as model inputs
BASE_CATEGORICAL = ["carrier", "origin", "dest"]
BASE_NUMERIC = ["dep_hour", "day_of_week", "month", "distance_km", "weather_index"]

# Features served from the feature store
STORE_FEATURES = [
    "origin_delay_rate_1d",
    "origin_delay_rate_7d",
    "origin_avg_delay_7d",
    "origin_flights_1d",
    "carrier_delay_rate_30d",
    "route_delay_rate_30d",
]

# Project team — edit names / BITS IDs here; they show up in the app.
TEAM = [
    {"member": "Member 1", "name": "Rohesen Raj", "id": "2025NS04108",
     "part": "Data Versioning (DVC)"},
    {"member": "Member 2", "name": "Member 2", "id": "",
     "part": "Feature Engineering & Feature Store"},
    {"member": "Member 3", "name": "Member 3", "id": "",
     "part": "Experiment Tracking & Model Registry (MLflow)"},
    {"member": "Member 4", "name": "Member 4", "id": "",
     "part": "Prediction Service & Lineage"},
]
