# src/ml/evaluate.py

import os
import sys
import json
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow

# Configuración logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bitcoin_forecast_model.joblib")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
EVAL_RESULTS_PATH = os.path.join(PROJECT_ROOT, "models", "evaluation_results.json")
EVAL_PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "gold_ml", "btc_eval_predictions.parquet")

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

def load_test_data():
    """Carga los datos de test desde el feature pipeline"""
    from src.feature_engineering.feature_pipeline import feature_pipeline
    data = feature_pipeline()
    return data["X_test"], data["y_test"]

def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo sobre el test set"""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test["target_close"], y_pred)
    rmse = np.sqrt(mean_squared_error(y_test["target_close"], y_pred))
    r2 = r2_score(y_test["target_close"], y_pred)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    logger.info(f"📊 Evaluación completada: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    return metrics, y_pred

def save_evaluation_results(metrics: dict):
    """Guarda resultados de evaluación en JSON"""
    os.makedirs(os.path.dirname(EVAL_RESULTS_PATH), exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "evaluation_date": datetime.now().isoformat()
            },
            f,
            indent=2
        )
    logger.info(f"💾 Resultados de evaluación guardados en: {EVAL_RESULTS_PATH}")

def save_eval_predictions(y_test, y_pred):
    """Guarda un DataFrame con y_test vs y_pred para análisis y dashboards"""
    df_results = pd.DataFrame({
        "date": y_test.index if hasattr(y_test, "index") else range(len(y_pred)),
        "y_true": y_test["target_close"].values,
        "y_pred": y_pred,
        "residual": y_test["target_close"].values - y_pred
    })
    os.makedirs(os.path.dirname(EVAL_PREDICTIONS_PATH), exist_ok=True)
    df_results.to_parquet(EVAL_PREDICTIONS_PATH, index=False)
    logger.info(f"💾 Predicciones de evaluación guardadas en: {EVAL_PREDICTIONS_PATH}")

def plot_evaluation_results(y_test, y_pred):
    """Genera gráficos de comparación y residuales"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plots_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Predicciones vs Reales en el tiempo
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test["target_close"], label="Real", color="blue")
    plt.plot(y_test.index, y_pred, label="Predicción", color="red", linestyle="--")
    plt.title("Predicciones vs Valores Reales (Test Set)")
    plt.xlabel("Fecha")
    plt.ylabel("Precio de Cierre")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pred_vs_real.png"), dpi=300)
    plt.close()

    # 2. Dispersión y_true vs y_pred
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test["target_close"], y=y_pred, alpha=0.7)
    plt.plot([y_test["target_close"].min(), y_test["target_close"].max()],
             [y_test["target_close"].min(), y_test["target_close"].max()],
             color="black", linestyle="--")
    plt.title("Dispersión: Valores Reales vs Predicciones")
    plt.xlabel("Real")
    plt.ylabel("Predicción")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "scatter_real_vs_pred.png"), dpi=300)
    plt.close()

    # 3. Histograma de residuales
    residuals = y_test["target_close"].values - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=20, kde=True, color="purple")
    plt.title("Distribución de Residuales")
    plt.xlabel("Residual (y_real - y_pred)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "residuals_distribution.png"), dpi=300)
    plt.close()

    logger.info(f"📊 Gráficos de evaluación guardados en {plots_dir}")


def log_evaluation_mlflow(metrics: dict, metadata: dict):
    """Loggea los resultados de evaluación en MLflow"""
    with mlflow.start_run(run_name="model_evaluation"):
        mlflow.log_param("model_type", metadata["model_type"])
        mlflow.log_param("evaluation_date", datetime.now().isoformat())
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
    logger.info("📊 Resultados de evaluación registrados en MLflow")

def evaluation_pipeline():
    """Pipeline completo de evaluación"""
    logger.info("🚀 Iniciando Pipeline de Evaluación")
    model, metadata = load_model_and_metadata()
    X_test, y_test = load_test_data()
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    save_evaluation_results(metrics)
    save_eval_predictions(y_test, y_pred)
    plot_evaluation_results(y_test, y_pred)
    log_evaluation_mlflow(metrics, metadata)
    logger.info("✅ Evaluación completada exitosamente")
    return metrics

if __name__ == "__main__":
    results = evaluation_pipeline()
    print("\n🎯 RESULTADOS DE EVALUACIÓN")
    print(json.dumps(results, indent=2))
