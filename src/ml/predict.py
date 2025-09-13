# src/ml/predict.py

import os
import sys
import json
import logging
import joblib
import pandas as pd
from datetime import datetime

import mlflow

# Configuración logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bitcoin_forecast_model.joblib")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "gold_ml", "btc_predictions.parquet")

# Agregar project root al path
sys.path.append(PROJECT_ROOT)

# Configuración centralizada de MLflow
from src.ml.utils.mlflow_config import setup_mlflow
mlflow_uri = setup_mlflow(PROJECT_ROOT)
logger.info(f"✅ MLflow configurado en: {mlflow_uri}")

def load_model_and_metadata():
    """Carga el modelo y metadata asociada"""
    try:
        model = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        logger.info(f"✅ Modelo cargado: {metadata['model_type']} entrenado el {metadata['training_date']}")
        return model, metadata
    except Exception as e:
        logger.error(f"❌ Error al cargar modelo/metadata: {e}")
        raise

def load_latest_features():
    """Carga la última fila de features procesadas"""
    from src.feature_engineering.feature_pipeline import feature_pipeline
    data = feature_pipeline()
    
    # Última observación para predicción diaria
    X_latest = data["X_test"][-1].reshape(1, -1)
    df_targets = data["y_test"].tail(1).reset_index(drop=True)
    
    return X_latest, df_targets, data["feature_names"], data["target_names"]

def generate_prediction(model, X_latest, df_targets):
    """Genera la predicción para la última fila"""
    y_pred = model.predict(X_latest)[0]

    result = {
        "date": str(df_targets.index[0]) if "date" not in df_targets.columns else str(df_targets["date"].iloc[0]),
        "y_true": float(df_targets["target_close"].iloc[0]) if "target_close" in df_targets.columns else None,
        "y_pred": float(y_pred),
        "prediction_date": datetime.now().isoformat()
    }
    return result

def save_prediction(result: dict):
    """Guarda la predicción en parquet"""
    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
    df_result = pd.DataFrame([result])

    if os.path.exists(PREDICTIONS_PATH):
        old = pd.read_parquet(PREDICTIONS_PATH)
        df_result = pd.concat([old, df_result]).drop_duplicates(subset=["date"], keep="last")

    df_result.to_parquet(PREDICTIONS_PATH, index=False)
    logger.info(f"💾 Predicción guardada en: {PREDICTIONS_PATH}")

def log_prediction_mlflow(result: dict, metadata: dict):
    """Loggea la predicción en MLflow para trazabilidad"""
    with mlflow.start_run(run_name="daily_prediction"):
        mlflow.log_param("model_type", metadata["model_type"])
        mlflow.log_param("prediction_date", result["prediction_date"])
        mlflow.log_metric("y_pred", result["y_pred"])
        if result["y_true"] is not None:
            mlflow.log_metric("y_true", result["y_true"])
    logger.info("📊 Predicción registrada en MLflow")

def prediction_pipeline():
    """Pipeline completo de predicción (producción)"""
    logger.info("🚀 Iniciando Pipeline de Predicción en Producción")
    model, metadata = load_model_and_metadata()
    X_latest, df_targets, feature_names, target_names = load_latest_features()
    result = generate_prediction(model, X_latest, df_targets)
    save_prediction(result)
    log_prediction_mlflow(result, metadata)
    logger.info("✅ Predicción completada exitosamente")
    return result

if __name__ == "__main__":
    result = prediction_pipeline()
    print("\n🎯 RESUMEN DE PREDICCIÓN (Producción)")
    print(json.dumps(result, indent=2))

