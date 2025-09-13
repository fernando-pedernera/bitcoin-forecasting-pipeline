# src/ml/utils/mlflow_config.py
import os
import mlflow

def setup_mlflow(project_root: str):
    MLFLOW_DB = os.path.join(project_root, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    return f"sqlite:///{MLFLOW_DB}"
