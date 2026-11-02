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
MLFLOW_DB_POINTER = ROOT / ".mlflow_db"   # names the active tracking DB (switched on rebuild)


def mlflow_db_path():
    """Active MLflow SQLite file. A rebuild switches to a fresh file instead of
    deleting the one an open SQLAlchemy engine is still using."""
    try:
        return ROOT / MLFLOW_DB_POINTER.read_text().strip()
    except OSError:
        return MLFLOW_DB


def new_mlflow_db():
    import time
    name = f"mlflow-{int(time.time())}.db"
    MLFLOW_DB_POINTER.write_text(name)
    return ROOT / name
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
    {"member": "Member 1", "name": "Rohesen Rajkamal Maurya", "id": "2025NS04108",
     "part": "Data Versioning (DVC)"},
    {"member": "Member 2", "name": "Akriti Adarsh", "id": "2025NS04070",
     "part": "Feature Engineering & Feature Store"},
    {"member": "Member 3", "name": "Chelsa Sam", "id": "2025NS04086",
     "part": "Experiment Tracking & Model Registry (MLflow)"},
    {"member": "Member 4", "name": "Amogh J Putanikar", "id": "2025NS04023",
     "part": "Prediction Service & Lineage"},
]
