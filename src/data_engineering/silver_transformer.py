# src/data_engineering/silver_transformer.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import Optional
from prefect import task, flow

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de rutas
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
BRONZE_PATH = os.path.join(PROJECT_ROOT, "data", "bronze", "bitcoin_ohlcv.parquet")
SILVER_PATH = os.path.join(PROJECT_ROOT, "data", "silver", "bitcoin_ohlcv_clean.parquet")

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

def validate_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida y corrige los tipos de datos del DataFrame.
    """
    if df.empty:
        return df
    
    # Hacer una copia para no modificar el original
    df_validated = df.copy()
    
    # Validar y convertir tipos de datos
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df_validated.columns:
            df_validated[col] = pd.to_numeric(df_validated[col], errors="coerce")
    
    # Validar fecha
    if "date" in df_validated.columns:
        df_validated["date"] = pd.to_datetime(df_validated["date"], errors="coerce")
    
    return df_validated

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maneja valores missing en el DataFrame.
    """
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Eliminar filas con fechas inválidas
    df_clean = df_clean.dropna(subset=["date"])
    
    # Para valores numéricos missing, usar forward fill o interpolación
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df_clean.columns:
            # Primero forward fill, luego backward fill para los extremos
            df_clean[col] = df_clean[col].fillna(method="ffill").fillna(method="bfill")
    
    # Si aún hay valores missing, eliminar esas filas
    df_clean = df_clean.dropna(subset=numeric_cols)
    
    return df_clean

def detect_and_handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta y maneja outliers en los datos.
    """
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Para precios (open, high, low, close)
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in df_clean.columns:
            # Calcular límites usando IQR (Interquartile Range)
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identificar outliers
            outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            
            if outliers.any():
                logger.warning(f"Se encontraron {outliers.sum()} outliers en {col}")
                # Reemplazar outliers con valores interpolados
                df_clean.loc[outliers, col] = np.nan
                df_clean[col] = df_clean[col].interpolate()
    
    return df_clean

def validate_ohlc_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida la consistencia de los datos OHLC.
    """
    if df.empty:
        return df
    
    df_valid = df.copy()
    
    # Verificar que high >= low
    invalid_high_low = df_valid["high"] < df_valid["low"]
    if invalid_high_low.any():
        logger.warning(f"Se encontraron {invalid_high_low.sum()} filas con high < low")
        # Corregir intercambiando high y low cuando sea necesario
        df_valid.loc[invalid_high_low, ["high", "low"]] = df_valid.loc[invalid_high_low, ["low", "high"]].values
    
    # Verificar que high >= open y high >= close
    invalid_high_open = df_valid["high"] < df_valid["open"]
    invalid_high_close = df_valid["high"] < df_valid["close"]
    
    if invalid_high_open.any() or invalid_high_close.any():
        logger.warning("Se encontraron inconsistencias en los valores high")
        # Para estas filas, establecer high como el máximo de open, high, close
        df_valid["high"] = df_valid[["open", "high", "close"]].max(axis=1)
    
    # Verificar que low <= open y low <= close
    invalid_low_open = df_valid["low"] > df_valid["open"]
    invalid_low_close = df_valid["low"] > df_valid["close"]
    
    if invalid_low_open.any() or invalid_low_close.any():
        logger.warning("Se encontraron inconsistencias en los valores low")
        # Para estas filas, establecer low como el mínimo de open, low, close
        df_valid["low"] = df_valid[["open", "low", "close"]].min(axis=1)
    
    return df_valid

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features derivadas básicas.
    """
    if df.empty:
        return df
    
    df_enriched = df.copy()
    
    # Retorno diario (log return)
    df_enriched["daily_return"] = np.log(df_enriched["close"] / df_enriched["close"].shift(1))
    
    # Rango de precios (high - low)
    df_enriched["price_range"] = df_enriched["high"] - df_enriched["low"]
    
    # Precio promedio (open, high, low, close)
    df_enriched["avg_price"] = (df_enriched["open"] + df_enriched["high"] + 
                               df_enriched["low"] + df_enriched["close"]) / 4
    
    # Volumen en millones para mejor visualización
    df_enriched["volume_millions"] = df_enriched["volume"] / 1_000_000
    
    # Día de la semana y mes
    df_enriched["day_of_week"] = df_enriched["date"].dt.dayofweek
    df_enriched["month"] = df_enriched["date"].dt.month
    df_enriched["year"] = df_enriched["date"].dt.year
    
    return df_enriched

# =========================
# Prefect Tasks
# =========================

@task(retries=2, retry_delay_seconds=10)
def extract_from_bronze() -> pd.DataFrame:
    """
    Extrae datos de la capa Bronze.
    """
    logger.info("Extrayendo datos de la capa Bronze")
    df = read_from_parquet(BRONZE_PATH)
    
    if df.empty:
        logger.error("No se pudieron extraer datos de la capa Bronze")
        return pd.DataFrame()
    
    logger.info(f"Datos extraídos: {len(df)} filas")
    return df

@task
def transform_to_silver(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma los datos para la capa Silver.
    """
    if df.empty:
        return df
    
    logger.info("Iniciando transformación a Silver")
    
    # 1. Validar tipos de datos
    df_transformed = validate_data_types(df)
    logger.info("✅ Validación de tipos de datos completada")
    
    # 2. Manejar valores missing
    df_transformed = handle_missing_values(df_transformed)
    logger.info("✅ Manejo de valores missing completado")
    
    # 3. Detectar y manejar outliers
    df_transformed = detect_and_handle_outliers(df_transformed)
    logger.info("✅ Manejo de outliers completado")
    
    # 4. Validar consistencia OHLC
    df_transformed = validate_ohlc_consistency(df_transformed)
    logger.info("✅ Validación de consistencia OHLC completada")
    
    # 5. Agregar features derivadas
    df_transformed = add_derived_features(df_transformed)
    logger.info("✅ Adición de features derivadas completada")
    
    logger.info(f"Transformación completada: {len(df_transformed)} filas")
    return df_transformed

@task
def load_to_silver(df: pd.DataFrame) -> None:
    """
    Carga los datos transformados a la capa Silver.
    """
    if not df.empty:
        try:
            save_to_parquet(df, SILVER_PATH)
            logger.info(f"✅ Datos cargados en Silver Layer: {SILVER_PATH}")
            
            # Mostrar resumen de los datos
            logger.info(f"Resumen de datos Silver:")
            logger.info(f"- Total filas: {len(df)}")
            logger.info(f"- Rango de fechas: {df['date'].min()} to {df['date'].max()}")
            logger.info(f"- Columnas: {list(df.columns)}")
            
        except Exception as e:
            logger.error(f"Error al cargar en Silver Layer: {e}")
            raise
    else:
        logger.warning("DataFrame vacío, no se carga en Silver Layer")

# =========================
# Prefect Flow
# =========================

@flow(name="silver_transformation")
def silver_transformation_flow():
    """
    Flujo principal de transformación de Bronze a Silver.
    """
    logger.info("Iniciando flujo de transformación Silver")
    
    # Extraer datos de Bronze
    df_bronze = extract_from_bronze()
    
    if not df_bronze.empty:
        # Transformar datos
        df_silver = transform_to_silver(df_bronze)
        
        # Cargar datos a Silver
        load_to_silver(df_silver)
        
        logger.info("✅ Flujo de transformación Silver completado")
        return df_silver
    else:
        logger.error("No se pudieron procesar datos para Silver")
        return pd.DataFrame()

# =========================
# Ejecución principal
# =========================

if __name__ == "__main__":
    # Mostrar información de rutas
    print(f"📁 Ruta del proyecto: {PROJECT_ROOT}")
    print(f"📁 Ruta fuente Bronze: {BRONZE_PATH}")
    print(f"📁 Ruta destino Silver: {SILVER_PATH}")
    
    # Verificar que existe el archivo Bronze
    if not os.path.exists(BRONZE_PATH):
        print("❌ Error: No se encuentra el archivo Bronze")
        print("Ejecuta primero extraction.py para generar los datos")
    else:
        # Ejecutar flujo de transformación
        df_result = silver_transformation_flow()
        
        if not df_result.empty:
            print(f"\n✅ Transformación completada. Total de filas: {len(df_result)}")
            print(f"📅 Rango de fechas: {df_result['date'].min()} to {df_result['date'].max()}")
            
            # Verificar que se creó el archivo Silver
            if os.path.exists(SILVER_PATH):
                print(f"💾 Archivo Silver creado en: {SILVER_PATH}")
                
                # Leer y mostrar info del archivo
                silver_df = read_from_parquet(SILVER_PATH)
                print(f"📊 Archivo Silver tiene {len(silver_df)} filas")
                print("\n📋 Primeras filas:")
                print(silver_df.head())
                print("\n📊 Resumen estadístico:")
                print(silver_df[["open", "high", "low", "close", "volume", "daily_return"]].describe())
            else:
                print("❌ Error: No se creó el archivo Silver")
        else:
            print("❌ El flujo no produjo datos")
            
            
            