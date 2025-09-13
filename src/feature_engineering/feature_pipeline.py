# src/feature_engineering/feature_pipeline.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import logging
import os
import json
from typing import Tuple, Dict, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de rutas
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
GOLD_PATH = os.path.join(PROJECT_ROOT, "data", "gold", "bitcoin_ml_features.parquet")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "gold_ml", "btc_processed_features.parquet")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.joblib")
FEATURE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "feature_config.json")

def load_gold_data() -> pd.DataFrame:
    """Carga los datos de la capa Gold"""
    try:
        df = pd.read_parquet(GOLD_PATH)
        logger.info(f"✅ Datos Gold cargados: {len(df)} filas, {len(df.columns)} columnas")
        return df
    except Exception as e:
        logger.error(f"❌ Error al cargar datos Gold: {e}")
        raise

def select_and_validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona y valida las features para el modelo ML
    """
    # Definir features prioritarias
    priority_features = [
        'close', 'volume', 'daily_return', 'price_range', 'avg_price',
        'sma_7', 'sma_14', 'sma_30', 'rsi', 'volatility_7', 'volatility_14',
        'macd', 'volume_ratio_7', 'obv', 'day_of_week', 'month'
    ]
    
    # Target variables
    target_features = ['target_close', 'target_return', 'target_direction']
    
    # Features disponibles
    available_features = [col for col in priority_features if col in df.columns]
    available_targets = [col for col in target_features if col in df.columns]
    
    # Columnas a mantener
    keep_columns = ['date'] + available_features + available_targets
    
    # Filtrar DataFrame
    df_filtered = df[keep_columns].copy()
    
    # Validar que no haya missing values en features
    missing_features = df_filtered[available_features].isnull().sum()
    if missing_features.any():
        logger.warning(f"⚠️  Features con valores missing: {missing_features[missing_features > 0].to_dict()}")
        # Eliminar filas con missing values
        df_filtered = df_filtered.dropna(subset=available_features)
    
    logger.info(f"✅ Features seleccionadas: {len(available_features)}")
    logger.info(f"✅ Targets disponibles: {available_targets}")
    logger.info(f"✅ Filas finales: {len(df_filtered)}")
    
    return df_filtered

def create_lagged_features(df: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """
    Crea features lagged para capturar dependencias temporales
    """
    df_lagged = df.copy()
    features_to_lag = ['close', 'volume', 'daily_return', 'rsi', 'volatility_7']
    
    for feature in features_to_lag:
        if feature in df_lagged.columns:
            for lag in range(1, max_lag + 1):
                df_lagged[f'{feature}_lag_{lag}'] = df_lagged[feature].shift(lag)
    
    # Eliminar filas con NaN por los lags
    df_lagged = df_lagged.dropna()
    
    logger.info(f"✅ Features lagged creadas (max_lag={max_lag})")
    return df_lagged

def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features de rolling statistics
    """
    df_rolling = df.copy()
    
    # Rolling statistics para precio
    if 'close' in df_rolling.columns:
        df_rolling['rolling_mean_7'] = df_rolling['close'].rolling(window=7).mean()
        df_rolling['rolling_std_7'] = df_rolling['close'].rolling(window=7).std()
        df_rolling['rolling_mean_14'] = df_rolling['close'].rolling(window=14).mean()
    
    # Rolling statistics para volumen
    if 'volume' in df_rolling.columns:
        df_rolling['volume_rolling_mean_7'] = df_rolling['volume'].rolling(window=7).mean()
    
    # Eliminar filas con NaN
    df_rolling = df_rolling.dropna()
    
    logger.info("✅ Rolling features creadas")
    return df_rolling

def prepare_train_test_split(df: pd.DataFrame, test_size: float = 0.2, 
                           val_size: float = 0.1) -> Tuple:
    """
    Split temporal para time series (no shuffle!)
    """
    total_size = len(df)
    test_split_idx = int(total_size * (1 - test_size))
    val_split_idx = int(total_size * (1 - test_size - val_size))
    
    train = df.iloc[:val_split_idx]
    val = df.iloc[val_split_idx:test_split_idx]
    test = df.iloc[test_split_idx:]
    
    logger.info(f"✅ Split temporal: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train, val, test

def scale_features(X_train, X_val, X_test, feature_columns, scaler_type='robust'):
    """
    Escala las features para modelos ML
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler
    import joblib
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    
    # Escalar features
    X_train_scaled = scaler.fit_transform(X_train[feature_columns])
    X_val_scaled = scaler.transform(X_val[feature_columns])
    X_test_scaled = scaler.transform(X_test[feature_columns])
    
    # Guardar el scaler
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"✅ Scaler guardado en: {SCALER_PATH}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def save_feature_config(feature_columns: list, target_columns: list):
    """Guarda la configuración de features"""
    config = {
        'feature_columns': feature_columns,
        'target_columns': target_columns,
        'timestamp': datetime.now().isoformat(),
        'num_features': len(feature_columns)
    }
    
    os.makedirs(os.path.dirname(FEATURE_CONFIG_PATH), exist_ok=True)
    with open(FEATURE_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Configuración de features guardada: {FEATURE_CONFIG_PATH}")

def analyze_feature_correlation(df: pd.DataFrame, feature_columns: list):
    """Analiza correlación entre features con clustermap (más limpio)"""
    correlation_matrix = df[feature_columns].corr()

    # Clustermap con agrupación jerárquica
    g = sns.clustermap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        figsize=(14, 14),
        cbar_kws={"shrink": 0.6}
    )

    g.fig.suptitle("Matriz de Correlación de Features (Clustermap)", fontsize=14)

    plots_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    os.makedirs(plots_dir, exist_ok=True)
    g.savefig(os.path.join(plots_dir, 'feature_correlation.png'), dpi=300)

    plt.close()
    logger.info("✅ Análisis de correlación completado (clustermap limpio)")

def feature_pipeline():
    """
    Pipeline completo de feature engineering
    """
    logger.info("🚀 Iniciando Pipeline de Feature Engineering")
    
    try:
        # 1. Cargar datos
        df = load_gold_data()
        
        # 2. Seleccionar y validar features
        df_processed = select_and_validate_features(df)
        
        # 3. Crear features adicionales
        df_processed = create_lagged_features(df_processed, max_lag=3)
        df_processed = create_rolling_features(df_processed)
        
        # 4. Preparar split temporal
        train_df, val_df, test_df = prepare_train_test_split(df_processed, test_size=0.2, val_size=0.1)
        
        # 5. Definir features y targets
        feature_columns = [col for col in train_df.columns 
                          if col not in ['date', 'target_close', 'target_return', 'target_direction']]
        target_columns = [col for col in train_df.columns if 'target' in col]
        
        # 6. Escalar features
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            train_df, val_df, test_df, feature_columns
        )
        
        # 7. Guardar datos procesados
        os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
        df_processed.to_parquet(FEATURES_PATH)
        logger.info(f"💾 Features procesadas guardadas: {FEATURES_PATH}")
        
        # 8. Guardar configuración
        save_feature_config(feature_columns, target_columns)
        
        # 9. Análisis de correlación
        analyze_feature_correlation(df_processed, feature_columns)
        
        # 10. Preparar datos para return
        processed_data = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': train_df[target_columns],
            'y_val': val_df[target_columns],
            'y_test': test_df[target_columns],
            'feature_names': feature_columns,
            'target_names': target_columns,
            'scaler': scaler
        }
        
        logger.info("✅ Pipeline de Feature Engineering completado exitosamente!")
        return processed_data
        
    except Exception as e:
        logger.error(f"❌ Error en el pipeline: {e}")
        raise

if __name__ == "__main__":
    # Ejecutar el pipeline
    processed_data = feature_pipeline()
    
    # Mostrar resumen
    print(f"\n🎯 RESUMEN DEL FEATURE ENGINEERING:")
    print(f"📊 Features: {len(processed_data['feature_names'])}")
    print(f"🎯 Targets: {processed_data['target_names']}")
    print(f"🏋️  Training samples: {len(processed_data['X_train'])}")
    print(f"📋 Validation samples: {len(processed_data['X_val'])}")
    print(f"🧪 Test samples: {len(processed_data['X_test'])}")
    print(f"💾 Datos guardados en: {FEATURES_PATH}")