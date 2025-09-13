# src/ml/train.py

import os
import sys
import json
import logging
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
import pandas as pd

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# MLflow
import mlflow
import mlflow.sklearn

# Configuración logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Agregar project root al path
sys.path.append(PROJECT_ROOT)

# Configuración centralizada de MLflow
from src.ml.utils.mlflow_config import setup_mlflow
mlflow_uri = setup_mlflow(PROJECT_ROOT)
logger.info(f"✅ MLflow configurado en: {mlflow_uri}")

def load_processed_data():
    """Carga los datos procesados del feature pipeline"""
    try:
        from src.feature_engineering.feature_pipeline import feature_pipeline
        return feature_pipeline()
    except Exception as e:
        logger.error(f"Error al cargar datos: {e}")
        raise

def train_baseline_models(X_train, y_train, X_val, y_val):
    """Entrena modelos baseline"""
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        with mlflow.start_run(run_name=f"baseline_{name}"):
            model.fit(X_train, y_train['target_close'])
            y_pred = model.predict(X_val)

            mae = mean_absolute_error(y_val['target_close'], y_pred)
            rmse = np.sqrt(mean_squared_error(y_val['target_close'], y_pred))
            r2 = r2_score(y_val['target_close'], y_pred)

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(model, f"model_{name}")

            results[name] = {
                'model': model,
                'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
                'predictions': y_pred
            }

            logger.info(f"✅ {name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    return results

def train_xgboost(X_train, y_train, X_val, y_val):
    """Entrena XGBoost (Windows friendly)"""
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.1,
        max_depth=6,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train['target_close'])
    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val['target_close'], y_pred)
    rmse = np.sqrt(mean_squared_error(y_val['target_close'], y_pred))
    r2 = r2_score(y_val['target_close'], y_pred)

    with mlflow.start_run(run_name="xgboost_optimized"):
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model_xgboost")

    logger.info(f"✅ XGBoost: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    return {'model': model, 'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2}, 'predictions': y_pred}

def save_best_model(best_model, best_metrics, feature_names):
    """Guarda el mejor modelo y metadata"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "bitcoin_forecast_model.joblib")
    joblib.dump(best_model, model_path)

    metadata = {
        'model_type': type(best_model).__name__,
        'metrics': best_metrics,
        'feature_names': feature_names,
        'training_date': datetime.now().isoformat()
    }
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"💾 Modelo guardado: {model_path}")
    return model_path

def train_pipeline():
    """Pipeline completo de entrenamiento"""
    logger.info("🚀 Iniciando Pipeline de Entrenamiento")
    data = load_processed_data()

    logger.info("🏋️  Entrenando modelos baseline...")
    baseline_results = train_baseline_models(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )

    logger.info("🏋️  Entrenando XGBoost...")
    xgb_result = train_xgboost(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )

    # Seleccionar mejor modelo por MAE
    all_results = {**baseline_results, 'xgboost': xgb_result}
    best_name = min(all_results.keys(), key=lambda k: all_results[k]['metrics']['mae'])
    best_result = all_results[best_name]

    model_path = save_best_model(best_result['model'], best_result['metrics'], data['feature_names'])

    return {
        'best_model': best_result['model'],
        'best_model_name': best_name,
        'metrics': best_result['metrics'],
        'model_path': model_path
    }

if __name__ == "__main__":
    results = train_pipeline()
    print(f"\n🎯 ENTRENAMIENTO COMPLETADO")
    print(f"🏆 Mejor modelo: {results['best_model_name']}")
    print(f"📊 Validation MAE: {results['metrics']['mae']:.4f}")
    print(f"📊 Validation RMSE: {results['metrics']['rmse']:.4f}")
    print(f"💾 Modelo guardado en: {results['model_path']}")

