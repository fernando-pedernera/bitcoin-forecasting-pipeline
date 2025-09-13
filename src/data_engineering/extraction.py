import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Optional, List
from prefect import task, flow
import logging
import os
from pyspark.sql import SparkSession
import pyarrow.parquet as pq
import pyarrow as pa

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://api.geckoterminal.com/api/v2"
HEADERS = {"accept": "application/json"}

# Configuración de rutas - CORREGIDO
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
BRONZE_PATH = os.path.join(PROJECT_ROOT, "data", "bronze", "bitcoin_ohlcv.parquet")

# =========================
# Funciones auxiliares - Encontrar pool de BTC
# =========================

def find_btc_pool(network: str = "eth") -> Optional[str]:
    """
    Encuentra el pool más líquido de BTC en la red especificada.
    """
    btc_addresses = {
        "eth": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # WBTC
        "bsc": "0x7130d2a12b9bcBFAe4f2634d864A1Ee1Ce3Ead9c",  # BTCB
        "polygon": "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6",  # WBTC
    }
    
    if network not in btc_addresses:
        logger.error(f"Red {network} no soportada")
        return None
    
    token_address = btc_addresses[network]
    url = f"{BASE_URL}/networks/{network}/tokens/{token_address}/pools"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        
        if resp.status_code != 200:
            logger.error(f"Error {resp.status_code} al buscar pools: {resp.text}")
            return None
        
        data = resp.json().get("data", [])
        if not data:
            logger.warning("No se encontraron pools")
            return None
        
        best_pool = None
        max_liquidity = 0
        
        for pool in data:
            liquidity = float(pool.get("attributes", {}).get("reserve_in_usd", 0))
            if liquidity > max_liquidity:
                max_liquidity = liquidity
                best_pool = pool.get("id", "").replace(f"{network}_", "")
        
        if best_pool:
            logger.info(f"Pool encontrado: {best_pool} (Liquidez: ${max_liquidity:,.2f})")
            return best_pool
        else:
            logger.warning("No se encontró un pool adecuado")
            return None
            
    except Exception as e:
        logger.error(f"Error al buscar pools: {e}")
        return None

def fetch_pool_ohlcv(pool_address: str, network: str = "eth", 
                    limit: int = 100, before_timestamp: Optional[int] = None) -> dict:
    """
    Obtiene datos OHLCV de un pool específico.
    """
    endpoint = f"networks/{network}/pools/{pool_address}/ohlcv/day"
    url = f"{BASE_URL}/{endpoint}"
    
    params = {"limit": limit}
    if before_timestamp:
        params["before_timestamp"] = before_timestamp
    
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        
        if resp.status_code != 200:
            logger.error(f"Error {resp.status_code} en OHLCV: {resp.text}")
            resp.raise_for_status()
        
        return resp.json()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error en solicitud OHLCV: {e}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        raise

def normalize_ohlcv_data(json_data: dict) -> pd.DataFrame:
    """
    Normaliza la respuesta OHLCV a un DataFrame con columnas estándar.
    """
    if not json_data or "data" not in json_data:
        logger.warning("No se encontraron datos en la respuesta")
        return pd.DataFrame()
    
    data = json_data.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
    
    if not data:
        logger.warning("Lista OHLCV vacía")
        return pd.DataFrame()
    
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.drop(columns=["timestamp"])
    df = df.sort_values("date").reset_index(drop=True)
    
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=numeric_cols)
    
    logger.info(f"Datos normalizados: {len(df)} filas")
    return df

def init_spark_session() -> SparkSession:
    """
    Inicializa y retorna una Spark Session.
    """
    return SparkSession.builder \
        .appName("BitcoinDataExtraction") \
        .config("spark.sql.warehouse.dir", os.path.join(PROJECT_ROOT, "data", "spark-warehouse")) \
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC") \
        .config("spark.executor.extraJavaOptions", "-Duser.timezone=UTC") \
        .config("spark.sql.session.timeZone", "UTC") \
        .getOrCreate()

# =========================
# Funciones de almacenamiento en Parquet
# =========================

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

# =========================
# Prefect Tasks
# =========================

@task(retries=2, retry_delay_seconds=10)
def extract_historical_data(days: int = 180) -> pd.DataFrame:
    """
    Descarga histórico completo de velas diarias de Bitcoin.
    Nota: La API gratuita solo permite hasta 180 días de histórico.
    """
    batch_size = min(days, 100)  # Límite por request
    all_data = []
    
    before_timestamp = None
    total_days_retrieved = 0
    
    logger.info(f"Iniciando descarga de {days} días de datos históricos (máximo permitido por API)")
    
    # Primero encontrar el pool
    pool_address = find_btc_pool("eth")
    if not pool_address:
        raise Exception("No se pudo encontrar un pool de BTC")
    
    while total_days_retrieved < days:
        limit = min(batch_size, days - total_days_retrieved)
        
        try:
            raw_data = fetch_pool_ohlcv(pool_address, "eth", limit=limit, before_timestamp=before_timestamp)
            df_batch = normalize_ohlcv_data(raw_data)
            
            if df_batch.empty:
                logger.warning("No se obtuvieron más datos")
                break
            
            all_data.append(df_batch)
            total_days_retrieved += len(df_batch)
            
            if not df_batch.empty:
                # Usar la fecha más antigua para el próximo request
                before_timestamp = int(df_batch["date"].min().timestamp())
            
            logger.info(f"Batch descargado: {len(df_batch)} filas. Total: {total_days_retrieved}/{days}")
            
            # Pequeña pausa para no saturar la API
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error en batch: {e}")
            break
    
    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        full_df = full_df.drop_duplicates(subset=["date"]).sort_values("date")
        
        logger.info(f"Descarga histórica completada: {len(full_df)} filas")
        return full_df
    
    logger.warning("No se pudieron descargar datos históricos")
    return pd.DataFrame()

@task(retries=2, retry_delay_seconds=5)
def extract_latest_data() -> pd.DataFrame:
    """
    Descarga los datos más recientes.
    """
    logger.info("Descargando datos más recientes")
    
    try:
        pool_address = find_btc_pool("eth")
        if not pool_address:
            raise Exception("No se pudo encontrar un pool de BTC")
        
        raw_data = fetch_pool_ohlcv(pool_address, "eth", limit=1)
        df = normalize_ohlcv_data(raw_data)
        
        if not df.empty:
            logger.info(f"Datos recientes descargados: {df['date'].iloc[0]}")
        else:
            logger.warning("No se obtuvieron datos recientes")
            
        return df
        
    except Exception as e:
        logger.error(f"Error al descargar datos recientes: {e}")
        return pd.DataFrame()

@task
def save_to_bronze_layer(df: pd.DataFrame) -> None:
    """
    Guarda DataFrame en la capa Bronze (formato Parquet).
    """
    if not df.empty:
        try:
            # Guardar en Parquet
            save_to_parquet(df, BRONZE_PATH)
            
            logger.info(f"✅ Datos guardados en Bronze Layer: {BRONZE_PATH}")
            
        except Exception as e:
            logger.error(f"Error al guardar en Bronze Layer: {e}")
            raise
    else:
        logger.warning("DataFrame vacío, no se guarda en Bronze Layer")

@task
def append_to_bronze_layer(new_df: pd.DataFrame) -> None:
    """
    Agrega nuevos datos a la capa Bronze existente.
    """
    if not new_df.empty:
        try:
            # Leer datos existentes si hay
            existing_df = read_from_parquet(BRONZE_PATH)
            
            if not existing_df.empty:
                # Combinar y eliminar duplicados
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["date"]).sort_values("date")
            else:
                combined_df = new_df
            
            # Guardar datos combinados
            save_to_parquet(combined_df, BRONZE_PATH)
            
            logger.info(f"✅ Datos incrementales agregados a Bronze Layer: {BRONZE_PATH}")
            
        except Exception as e:
            logger.error(f"Error al agregar datos a Bronze Layer: {e}")
            raise
    else:
        logger.warning("DataFrame vacío, no se agrega a Bronze Layer")

# =========================
# Prefect Flow
# =========================

@flow(name="bitcoin_data_extraction")
def extraction_flow(historical: bool = True, days: int = 180):
    """
    Flujo principal de extracción y carga a Bronze Layer (Parquet).
    """
    logger.info("Iniciando flujo de extracción")
    
    if historical:
        # Extraer datos históricos (máximo 180 días con API gratuita)
        df = extract_historical_data(days=days)
        if not df.empty:
            save_to_bronze_layer(df)
            logger.info(f"✅ Datos históricos guardados en Bronze Layer")
        else:
            logger.error("No se pudieron obtener datos históricos")
    else:
        # Extraer datos incrementales
        df = extract_latest_data()
        if not df.empty:
            append_to_bronze_layer(df)
            logger.info(f"✅ Datos incrementales agregados a Bronze Layer")
        else:
            logger.warning("No se obtuvieron datos incrementales")
    
    return df

# =========================
# Ejecución principal
# =========================

if __name__ == "__main__":
    # Mostrar información de rutas
    print(f"📁 Ruta del proyecto: {PROJECT_ROOT}")
    print(f"📁 Ruta destino Bronze: {BRONZE_PATH}")
    
    # Primero probar encontrar pool
    logger.info("Buscando pool de BTC...")
    pool = find_btc_pool("eth")
    
    if pool:
        logger.info(f"✅ Pool encontrado: {pool}")
        
        # Probar con un pequeño batch
        test_data = fetch_pool_ohlcv(pool, "eth", limit=5)
        test_df = normalize_ohlcv_data(test_data)
        
        if not test_df.empty:
            print("✅ Prueba exitosa - Todas las variables:")
            print(test_df.head())
            
            print(f"\n📊 Variables disponibles: {list(test_df.columns)}")
            print(f"📅 Rango de fechas: {test_df['date'].min()} to {test_df['date'].max()}")
            
            # Ejecutar flujo completo para datos históricos (máximo 180 días)
            df_result = extraction_flow(historical=True, days=180)
            
            if not df_result.empty:
                print(f"\n✅ Flujo completado. Total de filas: {len(df_result)}")
                print(f"📅 Rango completo: {df_result['date'].min()} to {df_result['date'].max()}")
                
                # Verificar que se creó el archivo Parquet
                if os.path.exists(BRONZE_PATH):
                    print(f"💾 Archivo Parquet creado en: {BRONZE_PATH}")
                    
                    # Leer y mostrar info del archivo
                    bronze_df = read_from_parquet(BRONZE_PATH)
                    print(f"📊 Archivo Parquet tiene {len(bronze_df)} filas")
                    print(bronze_df.head())
                else:
                    print("❌ Error: No se creó el archivo Parquet")
            else:
                print("❌ El flujo no produjo datos")
        else:
            print("❌ No se pudieron obtener datos OHLCV del pool")
    else:
        print("❌ No se pudo encontrar un pool de BTC adecuado")













# import requests
# import pandas as pd
# from datetime import datetime, timedelta
# import time
# from typing import Optional, List
# from prefect import task, flow
# import logging

# # Configuración de logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# BASE_URL = "https://api.geckoterminal.com/api/v2"
# HEADERS = {"accept": "application/json"}

# # Configuración para Bitcoin
# NETWORK = "bitcoin"
# TOKEN_ADDRESS = "btc"  # Bitcoin nativo
# TIMEFRAME = "day"  # Velas diarias

# # =========================
# # Funciones auxiliares
# # =========================

# def fetch_ohlcv(limit: int = 100, before_timestamp: Optional[int] = None) -> dict:
#     """
#     Llama al endpoint OHLCV para obtener velas diarias de Bitcoin.
    
#     Args:
#         limit: Número máximo de velas a retornar (máx 1000)
#         before_timestamp: Timestamp en segundos para obtener datos anteriores a esta fecha
    
#     Returns:
#         Dict con los datos OHLCV
#     """
#     endpoint = f"networks/{NETWORK}/tokens/{TOKEN_ADDRESS}/ohlcv/{TIMEFRAME}"
#     url = f"{BASE_URL}/{endpoint}"
    
#     params = {"limit": limit}
#     if before_timestamp:
#         params["before_timestamp"] = before_timestamp
    
#     try:
#         resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        
#         if resp.status_code != 200:
#             logger.error(f"Error {resp.status_code} en API: {resp.text}")
#             resp.raise_for_status()
        
#         return resp.json()
        
#     except requests.exceptions.RequestException as e:
#         logger.error(f"Error en la solicitud HTTP: {e}")
#         raise
#     except Exception as e:
#         logger.error(f"Error inesperado: {e}")
#         raise

# def normalize_ohlcv_data(json_data: dict) -> pd.DataFrame:
#     """
#     Normaliza la respuesta OHLCV a un DataFrame con columnas estándar.
    
#     Args:
#         json_data: Respuesta JSON de la API
        
#     Returns:
#         DataFrame con columnas: date, open, high, low, close, volume
#     """
#     if not json_data or "data" not in json_data:
#         logger.warning("No se encontraron datos en la respuesta")
#         return pd.DataFrame()
    
#     data = json_data.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
    
#     if not data:
#         logger.warning("Lista OHLCV vacía")
#         return pd.DataFrame()
    
#     # Crear DataFrame con los datos
#     df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    
#     # Convertir timestamp a datetime
#     df["date"] = pd.to_datetime(df["timestamp"], unit="s")
#     df = df.drop(columns=["timestamp"])
    
#     # Ordenar por fecha
#     df = df.sort_values("date").reset_index(drop=True)
    
#     # Convertir columnas numéricas
#     numeric_cols = ["open", "high", "low", "close", "volume"]
#     for col in numeric_cols:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
    
#     # Eliminar filas con valores nulos
#     df = df.dropna(subset=numeric_cols)
    
#     logger.info(f"Datos normalizados: {len(df)} filas")
#     return df

# def calculate_days_needed(years: int = 5) -> int:
#     """Calcula los días necesarios para X años de datos históricos."""
#     return years * 365

# # =========================
# # Prefect Tasks
# # =========================

# @task(retries=2, retry_delay_seconds=10)
# def extract_historical_data(years: int = 5) -> pd.DataFrame:
#     """
#     Descarga histórico completo de velas diarias de Bitcoin.
    
#     Args:
#         years: Número de años de datos históricos a descargar
        
#     Returns:
#         DataFrame con datos OHLCV históricos
#     """
#     days_needed = calculate_days_needed(years)
#     batch_size = 1000  # Máximo permitido por la API
#     all_data = []
    
#     before_timestamp = None
#     total_days_retrieved = 0
    
#     logger.info(f"Iniciando descarga de {days_needed} días de datos históricos")
    
#     while total_days_retrieved < days_needed:
#         limit = min(batch_size, days_needed - total_days_retrieved)
        
#         try:
#             raw_data = fetch_ohlcv(limit=limit, before_timestamp=before_timestamp)
#             df_batch = normalize_ohlcv_data(raw_data)
            
#             if df_batch.empty:
#                 logger.warning("No se obtuvieron más datos")
#                 break
            
#             all_data.append(df_batch)
#             total_days_retrieved += len(df_batch)
            
#             # Actualizar timestamp para la siguiente solicitud
#             before_timestamp = int(df_batch["date"].min().timestamp())
            
#             logger.info(f"Batch descargado: {len(df_batch)} filas. Total: {total_days_retrieved}/{days_needed}")
            
#             # Respeta el rate limit (30 llamadas/minuto = 1 cada 2 segundos)
#             time.sleep(2)
            
#         except Exception as e:
#             logger.error(f"Error en batch: {e}")
#             break
    
#     if all_data:
#         full_df = pd.concat(all_data, ignore_index=True)
#         full_df = full_df.drop_duplicates(subset=["date"]).sort_values("date")
        
#         logger.info(f"Descarga histórica completada: {len(full_df)} filas")
#         return full_df
    
#     logger.warning("No se pudieron descargar datos históricos")
#     return pd.DataFrame()

# @task(retries=2, retry_delay_seconds=5)
# def extract_latest_data() -> pd.DataFrame:
#     """
#     Descarga los datos más recientes (último día).
    
#     Returns:
#         DataFrame con los datos del último día disponible
#     """
#     logger.info("Descargando datos más recientes")
    
#     try:
#         raw_data = fetch_ohlcv(limit=1)  # Solo el último día
#         df = normalize_ohlcv_data(raw_data)
        
#         if not df.empty:
#             logger.info(f"Datos recientes descargados: {df['date'].iloc[0]}")
#         else:
#             logger.warning("No se obtuvieron datos recientes")
            
#         return df
        
#     except Exception as e:
#         logger.error(f"Error al descargar datos recientes: {e}")
#         return pd.DataFrame()

# @task
# def save_to_csv(df: pd.DataFrame, filename: str) -> None:
#     """
#     Guarda DataFrame en archivo CSV.
    
#     Args:
#         df: DataFrame a guardar
#         filename: Nombre del archivo
#     """
#     if not df.empty:
#         df.to_csv(filename, index=False)
#         logger.info(f"Datos guardados en {filename}")
#     else:
#         logger.warning("DataFrame vacío, no se guarda archivo")

# # =========================
# # Prefect Flow
# # =========================

# @flow(name="bitcoin_data_extraction")
# def extraction_flow(historical: bool = True, save_path: str = "data/raw/bitcoin_ohlcv.csv"):
#     """
#     Flujo Prefect para extracción de datos OHLCV de Bitcoin.
    
#     Args:
#         historical: True para datos históricos, False para incremental
#         save_path: Ruta donde guardar los datos
#     """
#     logger.info("Iniciando flujo de extracción")
    
#     if historical:
#         df = extract_historical_data(years=5)
#         if not df.empty:
#             save_to_csv(df, save_path)
#             logger.info(f"Descarga histórica completada. Datos guardados en {save_path}")
#         else:
#             logger.error("No se pudieron obtener datos históricos")
#     else:
#         df = extract_latest_data()
#         if not df.empty:
#             # Para incremental, podrías agregar a un archivo existente
#             logger.info(f"Datos incrementales obtenidos: {len(df)} fila(s)")
#             # Aquí implementarías la lógica de append al dataset existente
#         else:
#             logger.warning("No se obtuvieron datos incrementales")
    
#     return df

# # =========================
# # Ejecución principal
# # =========================

# if __name__ == "__main__":
#     # Prueba manual
#     logger.info("Ejecutando prueba manual de extracción")
    
#     # Probar con un pequeño batch primero
#     test_data = fetch_ohlcv(limit=10)
#     test_df = normalize_ohlcv_data(test_data)
    
#     if not test_df.empty:
#         print("✅ Prueba exitosa - Datos de muestra:")
#         print(test_df[["date", "close", "volume"]].head())
        
#         # Ejecutar flujo completo
#         df_result = extraction_flow(historical=True, save_path="../data/raw/bitcoin_ohlcv_sample.csv")
        
#         if not df_result.empty:
#             print(f"✅ Flujo completado. Total de filas: {len(df_result)}")
#             print(f"Rango de fechas: {df_result['date'].min()} to {df_result['date'].max()}")
#         else:
#             print("❌ El flujo no produjo datos")
#     else:
#         print("❌ La prueba inicial falló - Verificar conexión/API")



# import requests
# import pandas as pd
# from datetime import datetime
# from prefect import task, flow

# BASE_URL = "https://api.geckoterminal.com/api/v2"
# HEADERS = {"accept": "application/json"}


# # =========================
# # Funciones auxiliares
# # =========================

# def fetch_endpoint(endpoint: str, params: dict = None) -> dict:
#     """
#     Hace una llamada genérica a GeckoTerminal API.
#     """
#     url = f"{BASE_URL}/{endpoint}"
#     resp = requests.get(url, headers=HEADERS, params=params)

#     if resp.status_code != 200:
#         raise Exception(f"Error {resp.status_code}: {resp.text}")

#     return resp.json()


# def normalize_networks(json_data):
#     """
#     Normaliza la lista de networks en un DataFrame.
#     """
#     data = json_data.get("data", [])
#     if not data:
#         return pd.DataFrame()

#     df = pd.json_normalize(data)
#     return df


# # =========================
# # Prefect Tasks
# # =========================

# @task
# def extract_networks(page: int = 1) -> pd.DataFrame:
#     """
#     Ejemplo de extracción desde /networks para probar la API.
#     """
#     raw = fetch_endpoint("networks", params={"page": page})
#     df = normalize_networks(raw)
#     return df


# @flow
# def extraction_flow():
#     """
#     Flujo de Prefect para probar la API.
#     """
#     df = extract_networks(page=1)
#     print(f"[OK] Networks descargadas: {len(df)} filas")
#     print(df.head())
#     return df


# if __name__ == "__main__":
#     # 🔹 Ejecución manual de prueba
#     df = extraction_flow()


# import requests
# import pandas as pd

# BASE_URL = "https://api.geckoterminal.com/api/v2"
# HEADERS = {"accept": "application/json"}

# def get_pools(network: str, token_address: str):
#     url = f"{BASE_URL}/networks/{network}/tokens/{token_address}/pools"
#     resp = requests.get(url, headers=HEADERS)
#     if resp.status_code != 200:
#         raise Exception(f"Error {resp.status_code}: {resp.text}")
#     data = resp.json().get("data", [])
#     return pd.json_normalize(data)

# if __name__ == "__main__":
#     # WBTC en Ethereum
#     pools_eth = get_pools("eth", "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599")
#     print("Top pools WBTC/ETH:")
#     print(pools_eth[["id", "attributes.name", "attributes.reserve_in_usd"]].head())

#     # BTCB en BSC
#     pools_bsc = get_pools("bsc", "0x7130d2a12b9bcBFAe4f2634d864A1Ee1Ce3Ead9c")
#     print("\nTop pools BTCB/BSC:")
#     print(pools_bsc[["id", "attributes.name", "attributes.reserve_in_usd"]].head())
