# src/data_engineering/gold_transformer.py

import pandas as pd
import numpy as np
import logging
import os
from typing import Optional, List
from prefect import task, flow

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de rutas
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SILVER_PATH = os.path.join(PROJECT_ROOT, "data", "silver", "bitcoin_ohlcv_clean.parquet")
GOLD_PATH = os.path.join(PROJECT_ROOT, "data", "gold", "bitcoin_ml_features.parquet")

# Configuración de parámetros para features técnicas
TECHNICAL_INDICATOR_WINDOWS = [7, 14, 30, 50]  # Ventanas para medias móviles y RSI

def read_from_parquet(parquet_path: str) -> pd.DataFrame:
    """
    Lee datos desde archivo(s) Parquet.
    """
    try:
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            logger.info(f"Datos leídos desde Parquet: {len(df)} filas")
            return df
        else:
            logger.warning(f"Archivo Parquet no encontrado: {parquet_path}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error al leer desde Parquet: {e}")
        return pd.DataFrame()

def save_to_parquet(df: pd.DataFrame, parquet_path: str) -> None:
    """
    Guarda DataFrame en formato Parquet.
    """
    if not df.empty:
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
            
            # Guardar en Parquet
            df.to_parquet(parquet_path, index=False)
            
            logger.info(f"Datos guardados en Parquet: {parquet_path}")
            
        except Exception as e:
            logger.error(f"Error al guardar en Parquet: {e}")
            raise
    else:
        logger.warning("DataFrame vacío, no se guarda en Parquet")

def calculate_moving_averages(df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
    """
    Calcula medias móviles simples y exponenciales para diferentes ventanas.
    """
    if df.empty:
        return df
    
    if windows is None:
        windows = TECHNICAL_INDICATOR_WINDOWS
    
    df_ma = df.copy()
    
    for window in windows:
        # Media móvil simple (SMA)
        df_ma[f'sma_{window}'] = df_ma['close'].rolling(window=window).mean()
        
        # Media móvil exponencial (EMA)
        df_ma[f'ema_{window}'] = df_ma['close'].ewm(span=window, adjust=False).mean()
        
        # Ratio precio/Media móvil
        df_ma[f'price_sma_ratio_{window}'] = df_ma['close'] / df_ma[f'sma_{window}']
    
    logger.info(f"✅ Medias móviles calculadas para ventanas: {windows}")
    return df_ma

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calcula el Relative Strength Index (RSI).
    """
    if df.empty or len(df) < window + 1:
        return df
    
    df_rsi = df.copy()
    
    # Calcular cambios de precio
    delta = df_rsi['close'].diff()
    
    # Separar ganancias y pérdidas
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    # Calcular RS y RSI
    rs = gain / loss
    df_rsi['rsi'] = 100 - (100 / (1 + rs))
    
    logger.info(f"✅ RSI calculado (ventana: {window})")
    return df_rsi

def calculate_volatility_metrics(df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
    """
    Calcula métricas de volatilidad.
    """
    if df.empty:
        return df
    
    if windows is None:
        windows = [7, 14, 30]
    
    df_vol = df.copy()
    
    for window in windows:
        # Volatilidad (desviación estándar de returns)
        df_vol[f'volatility_{window}'] = df_vol['daily_return'].rolling(window=window).std() * np.sqrt(252)
        
        # Rango verdadero promedio (ATR)
        high_low = df_vol['high'] - df_vol['low']
        high_close = np.abs(df_vol['high'] - df_vol['close'].shift())
        low_close = np.abs(df_vol['low'] - df_vol['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_vol[f'atr_{window}'] = true_range.rolling(window=window).mean()
    
    logger.info(f"✅ Métricas de volatilidad calculadas para ventanas: {windows}")
    return df_vol

def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores de momentum.
    """
    if df.empty:
        return df
    
    df_momentum = df.copy()
    
    # Rate of Change (ROC)
    df_momentum['roc_7'] = df_momentum['close'].pct_change(periods=7)
    df_momentum['roc_14'] = df_momentum['close'].pct_change(periods=14)
    
    # Momentum (diferencia de precio)
    df_momentum['momentum_7'] = df_momentum['close'].diff(7)
    df_momentum['momentum_14'] = df_momentum['close'].diff(14)
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df_momentum['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_momentum['close'].ewm(span=26, adjust=False).mean()
    df_momentum['macd'] = ema_12 - ema_26
    df_momentum['macd_signal'] = df_momentum['macd'].ewm(span=9, adjust=False).mean()
    df_momentum['macd_histogram'] = df_momentum['macd'] - df_momentum['macd_signal']
    
    logger.info("✅ Indicadores de momentum calculados")
    return df_momentum

def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores basados en volumen.
    """
    if df.empty:
        return df
    
    df_volume = df.copy()
    
    # Media móvil de volumen
    df_volume['volume_sma_7'] = df_volume['volume'].rolling(window=7).mean()
    df_volume['volume_sma_14'] = df_volume['volume'].rolling(window=14).mean()
    
    # Ratio volumen/Media móvil de volumen
    df_volume['volume_ratio_7'] = df_volume['volume'] / df_volume['volume_sma_7']
    df_volume['volume_ratio_14'] = df_volume['volume'] / df_volume['volume_sma_14']
    
    # On Balance Volume (OBV)
    df_volume['obv'] = (np.sign(df_volume['close'].diff()) * df_volume['volume']).fillna(0).cumsum()
    
    logger.info("✅ Indicadores de volumen calculados")
    return df_volume

def create_target_variable(df: pd.DataFrame, forward_periods: int = 1) -> pd.DataFrame:
    """
    Crea la variable target para predicción.
    """
    if df.empty or len(df) < forward_periods + 1:
        return df
    
    df_target = df.copy()
    
    # Precio futuro (target para regression)
    df_target['target_close'] = df_target['close'].shift(-forward_periods)
    
    # Retorno futuro (target para regression)
    df_target['target_return'] = df_target['daily_return'].shift(-forward_periods)
    
    # Dirección del precio (target para classification)
    df_target['target_direction'] = (df_target['target_close'] > df_target['close']).astype(int)
    
    # Eliminar la última fila que tendrá NaN en el target
    df_target = df_target.iloc[:-forward_periods]
    
    logger.info(f"✅ Variable target creada (forward periods: {forward_periods})")
    return df_target

def handle_missing_values_gold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maneja valores missing específicos para la capa Gold.
    """
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Para features técnicas, usar forward fill
    technical_cols = [col for col in df_clean.columns if any(x in col for x in ['sma_', 'ema_', 'rsi', 'volatility_', 'atr_', 'macd'])]
    
    for col in technical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(method='ffill')
    
    # Eliminar filas que aún tengan valores missing
    df_clean = df_clean.dropna()
    
    logger.info(f"✅ Valores missing manejados. Filas restantes: {len(df_clean)}")
    return df_clean

def prepare_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el dataset final para machine learning.
    """
    if df.empty:
        return df
    
    df_ml = df.copy()
    
    # Seleccionar y ordenar columnas
    feature_columns = [
        # Precios y volumen
        'open', 'high', 'low', 'close', 'volume', 'volume_millions',
        
        # Features básicas
        'daily_return', 'price_range', 'avg_price',
        
        # Medias móviles
        'sma_7', 'sma_14', 'sma_30', 'sma_50',
        'ema_7', 'ema_14', 'ema_30', 'ema_50',
        'price_sma_ratio_7', 'price_sma_ratio_14', 'price_sma_ratio_30', 'price_sma_ratio_50',
        
        # RSI y volatilidad
        'rsi', 'volatility_7', 'volatility_14', 'volatility_30',
        'atr_7', 'atr_14', 'atr_30',
        
        # Momentum
        'roc_7', 'roc_14', 'momentum_7', 'momentum_14',
        'macd', 'macd_signal', 'macd_histogram',
        
        # Volumen
        'volume_sma_7', 'volume_sma_14', 'volume_ratio_7', 'volume_ratio_14', 'obv',
        
        # Temporal
        'day_of_week', 'month', 'year',
        
        # Targets
        'target_close', 'target_return', 'target_direction'
    ]
    
    # Mantener solo las columnas que existen
    available_cols = [col for col in feature_columns if col in df_ml.columns]
    df_ml = df_ml[available_cols + ['date']]  # Siempre mantener la fecha
    
    logger.info(f"✅ Dataset ML preparado con {len(available_cols)} features")
    return df_ml

# =========================
# Prefect Tasks
# =========================

@task(retries=2, retry_delay_seconds=10)
def extract_from_silver() -> pd.DataFrame:
    """
    Extrae datos de la capa Silver.
    """
    logger.info("Extrayendo datos de la capa Silver")
    df = read_from_parquet(SILVER_PATH)
    
    if df.empty:
        logger.error("No se pudieron extraer datos de la capa Silver")
        return pd.DataFrame()
    
    logger.info(f"Datos extraídos: {len(df)} filas")
    return df

@task
def transform_to_gold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma los datos para la capa Gold con features para ML.
    """
    if df.empty:
        return df
    
    logger.info("Iniciando transformación a Gold")
    
    # 1. Medias móviles
    df_transformed = calculate_moving_averages(df)
    logger.info("✅ Medias móviles calculadas")
    
    # 2. RSI
    df_transformed = calculate_rsi(df_transformed)
    logger.info("✅ RSI calculado")
    
    # 3. Volatilidad
    df_transformed = calculate_volatility_metrics(df_transformed)
    logger.info("✅ Métricas de volatilidad calculadas")
    
    # 4. Momentum
    df_transformed = calculate_momentum_indicators(df_transformed)
    logger.info("✅ Indicadores de momentum calculados")
    
    # 5. Volumen
    df_transformed = calculate_volume_indicators(df_transformed)
    logger.info("✅ Indicadores de volumen calculados")
    
    # 6. Variable target
    df_transformed = create_target_variable(df_transformed, forward_periods=1)
    logger.info("✅ Variable target creada")
    
    # 7. Manejar valores missing
    df_transformed = handle_missing_values_gold(df_transformed)
    logger.info("✅ Valores missing manejados")
    
    # 8. Preparar features para ML
    df_transformed = prepare_ml_features(df_transformed)
    logger.info("✅ Features para ML preparadas")
    
    logger.info(f"Transformación completada: {len(df_transformed)} filas")
    logger.info(f"Features disponibles: {len(df_transformed.columns)}")
    
    return df_transformed

@task
def load_to_gold(df: pd.DataFrame) -> None:
    """
    Carga los datos transformados a la capa Gold.
    """
    if not df.empty:
        try:
            save_to_parquet(df, GOLD_PATH)
            logger.info(f"✅ Datos cargados en Gold Layer: {GOLD_PATH}")
            
            # Mostrar resumen de los datos
            logger.info(f"Resumen de datos Gold:")
            logger.info(f"- Total filas: {len(df)}")
            logger.info(f"- Total features: {len(df.columns) - 1}")  # Excluyendo date
            logger.info(f"- Rango de fechas: {df['date'].min()} to {df['date'].max()}")
            
            # Mostrar tipos de features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'date']
            logger.info(f"- Features numéricas: {len(numeric_cols)}")
            
        except Exception as e:
            logger.error(f"Error al cargar en Gold Layer: {e}")
            raise
    else:
        logger.warning("DataFrame vacío, no se carga en Gold Layer")

# =========================
# Prefect Flow
# =========================

@flow(name="gold_transformation")
def gold_transformation_flow():
    """
    Flujo principal de transformación de Silver a Gold.
    """
    logger.info("Iniciando flujo de transformación Gold")
    
    # Extraer datos de Silver
    df_silver = extract_from_silver()
    
    if not df_silver.empty:
        # Transformar datos
        df_gold = transform_to_gold(df_silver)
        
        # Cargar datos a Gold
        load_to_gold(df_gold)
        
        logger.info("✅ Flujo de transformación Gold completado")
        return df_gold
    else:
        logger.error("No se pudieron procesar datos para Gold")
        return pd.DataFrame()

# =========================
# Ejecución principal
# =========================

if __name__ == "__main__":
    # Mostrar información de rutas
    print(f"📁 Ruta del proyecto: {PROJECT_ROOT}")
    print(f"📁 Ruta fuente Silver: {SILVER_PATH}")
    print(f"📁 Ruta destino Gold: {GOLD_PATH}")
    
    # Verificar que existe el archivo Silver
    if not os.path.exists(SILVER_PATH):
        print("❌ Error: No se encuentra el archivo Silver")
        print("Ejecuta primero silver_transformer.py para generar los datos")
    else:
        # Ejecutar flujo de transformación
        df_result = gold_transformation_flow()
        
        if not df_result.empty:
            print(f"\n✅ Transformación completada. Total de filas: {len(df_result)}")
            print(f"📊 Total de features: {len(df_result.columns)}")
            print(f"📅 Rango de fechas: {df_result['date'].min()} to {df_result['date'].max()}")
            
            # Verificar que se creó el archivo Gold
            if os.path.exists(GOLD_PATH):
                print(f"💾 Archivo Gold creado en: {GOLD_PATH}")
                
                # Leer y mostrar info del archivo
                gold_df = read_from_parquet(GOLD_PATH)
                print(f"📊 Archivo Gold tiene {len(gold_df)} filas y {len(gold_df.columns)} columnas")
                
                print("\n📋 Primeras filas (features seleccionadas):")
                sample_cols = ['date', 'close', 'sma_14', 'rsi', 'volatility_14', 'target_close']
                available_sample_cols = [col for col in sample_cols if col in gold_df.columns]
                print(gold_df[available_sample_cols].head())
                
                print("\n🎯 Variables target:")
                target_cols = [col for col in gold_df.columns if 'target' in col]
                if target_cols:
                    print(target_cols)
                else:
                    print("No se encontraron variables target")
                    
            else:
                print("❌ Error: No se creó el archivo Gold")
        else:
            print("❌ El flujo no produjo datos")