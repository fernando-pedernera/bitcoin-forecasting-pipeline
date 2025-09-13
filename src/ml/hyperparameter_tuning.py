# src/ml/hyperparameter_tuning.py

import os
import sys
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import joblib

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths - MOVER AL PRINCIPIO
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "gold_ml")

# Agregar project root al sys.path
sys.path.append(PROJECT_ROOT)

# MLflow - IMPORTAR DESPUÉS DE AGREGAR AL PATH
import mlflow
import mlflow.sklearn

# Configuración centralizada de MLflow (igual que en evaluate.py)
try:
    from src.ml.utils.mlflow_config import setup_mlflow
    mlflow_uri = setup_mlflow(PROJECT_ROOT)
    logger.info(f"✅ MLflow configurado en: {mlflow_uri}")
except ImportError as e:
    logger.error(f"❌ Error importing mlflow_config: {e}")
    logger.info("⚠️  Using default MLflow setup")
    # Configuración por defecto
    mlflow.set_tracking_uri("file:///tmp/mlruns")

# Importar feature pipeline
try:
    from src.feature_engineering.feature_pipeline import feature_pipeline
except ImportError as e:
    logger.error(f"❌ Error importing feature_pipeline: {e}")
    sys.exit(1)


def load_processed_data():
    """Carga datos procesados para entrenamiento"""
    return feature_pipeline()


def evaluate_model(model, X_val, y_val, target="target_close"):
    """Evalúa un modelo y retorna métricas"""
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val[target], y_pred)
    rmse = np.sqrt(mean_squared_error(y_val[target], y_pred))
    r2 = r2_score(y_val[target], y_pred)
    return mae, rmse, r2, y_pred


def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """Ejecuta tuning con RandomizedSearchCV y loggea en MLflow"""
    search_spaces = {
        "random_forest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10]
            }
        },
        "gradient_boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7]
            }
        },
        "xgboost": {
            "model": xgb.XGBRegressor(objective="reg:squarederror", random_state=42, verbosity=0),
            "params": {
                "n_estimators": [100, 200, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 6, 10],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0]
            }
        }
    }

    all_results = []
    best_overall = {"model": None, "mae": float("inf")}

    for model_name, cfg in search_spaces.items():
        logger.info(f"🏋️ Ejecutando tuning para {model_name}...")

        search = RandomizedSearchCV(
            cfg["model"],
            cfg["params"],
            n_iter=10,
            cv=3,
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train["target_close"])

        best_model = search.best_estimator_
        mae, rmse, r2, y_pred = evaluate_model(best_model, X_val, y_val)

        # Guardar resultados
        result = {
            "model": model_name,
            "best_params": search.best_params_,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }
        all_results.append(result)

        logger.info(f"✅ {model_name} | MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

        # Loggear en MLflow
        try:
            with mlflow.start_run(run_name=f"tuning_{model_name}"):
                mlflow.log_params(search.best_params_)
                mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
                mlflow.sklearn.log_model(best_model, f"tuned_{model_name}")
                logger.info(f"📊 {model_name} - Resultados loggeados en MLflow")
        except Exception as e:
            logger.error(f"❌ Error al loggear en MLflow para {model_name}: {e}")

        # Seleccionar mejor global
        if mae < best_overall["mae"]:
            best_overall.update({
                "model": best_model, 
                "mae": mae, 
                "rmse": rmse, 
                "r2": r2, 
                "name": model_name,
                "params": search.best_params_
            })

    return all_results, best_overall


def save_results(all_results, best_overall, feature_names):
    """Guarda reporte JSON y modelo final"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Guardar todas las combinaciones
    results_path = os.path.join(MODEL_DIR, "tuning_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Guardar mejor modelo
    best_model_path = os.path.join(MODEL_DIR, "best_tuned_model.joblib")
    joblib.dump(best_overall["model"], best_model_path)

    metadata = {
        "best_model": best_overall["name"],
        "best_params": best_overall["params"],
        "metrics": {
            "mae": best_overall["mae"],
            "rmse": best_overall["rmse"],
            "r2": best_overall["r2"]
        },
        "timestamp": datetime.now().isoformat(),
        "feature_names": feature_names
    }
    metadata_path = os.path.join(MODEL_DIR, "tuning_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"💾 Resultados guardados en {results_path}")
    logger.info(f"💾 Mejor modelo guardado en {best_model_path}")
    return results_path, best_model_path, metadata_path


def plot_tuning_results(all_results):
    """Genera un gráfico comparativo de MAE por modelo"""
    os.makedirs(os.path.join(REPORTS_DIR, "figures"), exist_ok=True)

    df_results = pd.DataFrame(all_results)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_results["model"], df_results["mae"], color="skyblue", edgecolor="black", alpha=0.7)
    
    # Añadir valores en las barras
    for bar, mae in zip(bars, df_results["mae"]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{mae:.4f}', ha='center', va='bottom')
    
    plt.xlabel("Modelo")
    plt.ylabel("MAE (menor es mejor)")
    plt.title("Comparación de MAE por Modelo (Hyperparameter Tuning)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = os.path.join(REPORTS_DIR, "figures", "tuning_results.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    logger.info(f"📊 Gráfico de resultados guardado en: {plot_path}")
    return plot_path


def tuning_pipeline():
    logger.info("🚀 Iniciando Hyperparameter Tuning Pipeline")
    data = load_processed_data()

    all_results, best_overall = hyperparameter_tuning(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    )

    results_path, best_model_path, metadata_path = save_results(
        all_results, best_overall, data["feature_names"]
    )
    plot_tuning_results(all_results)
    
    logger.info(f"🎯 Mejor modelo: {best_overall['name']} con MAE={best_overall['mae']:.4f}")
    logger.info("✅ Hyperparameter Tuning completado!")
    
    return best_overall


if __name__ == "__main__":
    best_model = tuning_pipeline()
    print(f"\n🎯 MEJOR MODELO ENCONTRADO: {best_model['name']}")
    print(f"📊 MAE: {best_model['mae']:.4f}")
    print(f"📊 RMSE: {best_model['rmse']:.4f}")
    print(f"📊 R²: {best_model['r2']:.4f}")
