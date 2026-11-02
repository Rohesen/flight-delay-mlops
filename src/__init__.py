"""Flight delay MLOps pipeline package."""
import os as _os

# Quiet, private defaults for MLflow before it is imported anywhere
_os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
_os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
_os.environ.setdefault("DO_NOT_TRACK", "true")
