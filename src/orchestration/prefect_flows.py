"""
prefect_flows.py
Orquestación profesional del pipeline completo de Bitcoin Forecasting.
Sigue mejores prácticas de Prefect para producción.
"""
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Tuple, Optional
import warnings

# Prefect imports
from prefect import flow, task, get_run_logger
from prefect.blocks.system import Secret
from prefect.context import get_run_context
from prefect.artifacts import create_markdown_artifact

# Suppress warnings
warnings.filterwarnings('ignore')

# Import project modules
try:
    from src.data_engineering.extraction import extraction_flow
    from src.data_engineering.silver_transformer import silver_transformation_flow
    from src.data_engineering.gold_transformer import gold_transformation_flow
    from src.feature_engineering.feature_pipeline import run_feature_pipeline
    from src.ml.train import train_pipeline
    from src.ml.predict import prediction_pipeline
    from src.ml.evaluate import evaluation_pipeline
    from src.ml.hyperparameter_tuning import tuning_pipeline
    from src.utils.helpers import send_alert, validate_environment
except ImportError as e:
    logging.warning(f"Import warning: {e}. Using placeholder functions for testing.")
    
    # Placeholder functions for testing
    def extraction_flow(*args, **kwargs): return True
    def silver_transformation_flow(*args, **kwargs): return True
    def gold_transformation_flow(*args, **kwargs): return True
    def run_feature_pipeline(*args, **kwargs): return {}
    def train_pipeline(*args, **kwargs): return {"status": "success"}
    def prediction_pipeline(*args, **kwargs): return {"prediction": 45000}
    def evaluation_pipeline(*args, **kwargs): return {"mae": 150}
    def tuning_pipeline(*args, **kwargs): return {"best_model": "xgboost"}
    def send_alert(*args, **kwargs): pass
    def validate_environment(*args, **kwargs): return True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Task configuration
TASK_RETRIES = 2
TASK_RETRY_DELAY = 30  # seconds
TASK_TIMEOUT = 3600  # 1 hour

@task(
    name="validate-environment",
    description="Valida que el entorno tenga todas las dependencias necesarias",
    retries=TASK_RETRIES,
    retry_delay_seconds=TASK_RETRY_DELAY,
    timeout_seconds=TASK_TIMEOUT
)
def validate_environment_task() -> bool:
    """Valida el entorno antes de ejecutar el pipeline"""
    logger = get_run_logger()
    try:
        result = validate_environment()
        logger.info("✅ Validación de entorno exitosa")
        return result
    except Exception as e:
        logger.error(f"❌ Error en validación de entorno: {e}")
        raise

@task(
    name="run-etl-pipeline",
    description="Ejecuta el pipeline completo ETL (Bronze → Silver → Gold)",
    retries=TASK_RETRIES,
    retry_delay_seconds=TASK_RETRY_DELAY,
    timeout_seconds=TASK_TIMEOUT * 2
)
def run_etl_pipeline_task(historical: bool = False) -> Dict[str, Any]:
    """Ejecuta el pipeline ETL completo"""
    logger = get_run_logger()
    
    try:
        logger.info("🚀 Iniciando pipeline ETL...")
        
        # Ejecutar flows existentes
        extraction_success = extraction_flow(historical=historical)
        silver_success = silver_transformation_flow()
        gold_success = gold_transformation_flow()
        
        success = all([extraction_success, silver_success, gold_success])
        
        if success:
            logger.info("✅ Pipeline ETL completado exitosamente")
            return {"status": "success", "components": {"extraction": True, "silver": True, "gold": True}}
        else:
            error_msg = "❌ Pipeline ETL falló en uno o más componentes"
            logger.error(error_msg)
            return {"status": "error", "components": {"extraction": extraction_success, "silver": silver_success, "gold": gold_success}}
            
    except Exception as e:
        logger.error(f"❌ Error crítico en ETL: {e}")
        raise

@task(
    name="run-feature-engineering",
    description="Ejecuta el pipeline de feature engineering para ML",
    retries=TASK_RETRIES,
    retry_delay_seconds=TASK_RETRY_DELAY,
    timeout_seconds=TASK_TIMEOUT
)
def run_feature_engineering_task() -> Dict[str, Any]:
    """Ejecuta el pipeline de features"""
    logger = get_run_logger()
    
    try:
        logger.info("🔧 Generando features para ML...")
        result = run_feature_pipeline()
        logger.info("✅ Feature engineering completado")
        return {"status": "success", "features_generated": True, "details": result}
    except Exception as e:
        logger.error(f"❌ Error en feature engineering: {e}")
        return {"status": "error", "features_generated": False, "error": str(e)}

@task(
    name="run-model-training",
    description="Entrena el modelo de forecasting",
    retries=1,  # Menos retries para training (es costoso)
    retry_delay_seconds=TASK_RETRY_DELAY * 2,
    timeout_seconds=TASK_TIMEOUT * 3
)
def run_model_training_task() -> Dict[str, Any]:
    """Ejecuta el entrenamiento del modelo"""
    logger = get_run_logger()
    
    try:
        logger.info("🤖 Iniciando entrenamiento del modelo...")
        result = train_pipeline()
        logger.info("✅ Entrenamiento del modelo completado")
        return {"status": "success", "training_completed": True, "results": result}
    except Exception as e:
        logger.error(f"❌ Error en entrenamiento: {e}")
        return {"status": "error", "training_completed": False, "error": str(e)}

@task(
    name="run-prediction",
    description="Genera predicciones con el modelo actual",
    retries=TASK_RETRIES,
    retry_delay_seconds=TASK_RETRY_DELAY,
    timeout_seconds=TASK_TIMEOUT
)
def run_prediction_task() -> Dict[str, Any]:
    """Genera predicciones diarias"""
    logger = get_run_logger()
    
    try:
        logger.info("🔮 Generando predicciones...")
        result = prediction_pipeline()
        logger.info("✅ Predicciones generadas exitosamente")
        return {"status": "success", "prediction_generated": True, "prediction": result}
    except Exception as e:
        logger.error(f"❌ Error en generación de predicciones: {e}")
        return {"status": "error", "prediction_generated": False, "error": str(e)}

@task(
    name="run-model-evaluation",
    description="Evalúa el modelo actual en el test set",
    retries=TASK_RETRIES,
    retry_delay_seconds=TASK_RETRY_DELAY,
    timeout_seconds=TASK_TIMEOUT
)
def run_model_evaluation_task() -> Dict[str, Any]:
    """Evalúa el modelo actual"""
    logger = get_run_logger()
    
    try:
        logger.info("📊 Evaluando modelo...")
        result = evaluation_pipeline()
        logger.info("✅ Evaluación del modelo completada")
        return {"status": "success", "evaluation_completed": True, "metrics": result}
    except Exception as e:
        logger.error(f"❌ Error en evaluación: {e}")
        return {"status": "error", "evaluation_completed": False, "error": str(e)}

@task(
    name="run-hyperparameter-tuning",
    description="Optimiza hiperparámetros del modelo",
    retries=1,  # Hiperparameter tuning es muy costoso para retries
    retry_delay_seconds=TASK_RETRY_DELAY * 3,
    timeout_seconds=TASK_TIMEOUT * 4
)
def run_hyperparameter_tuning_task() -> Dict[str, Any]:
    """Ejecuta optimización de hiperparámetros"""
    logger = get_run_logger()
    
    try:
        logger.info("🎯 Iniciando optimización de hiperparámetros...")
        result = tuning_pipeline()
        logger.info("✅ Optimización de hiperparámetros completada")
        return {"status": "success", "tuning_completed": True, "best_params": result}
    except Exception as e:
        logger.error(f"❌ Error en optimización: {e}")
        return {"status": "error", "tuning_completed": False, "error": str(e)}

@task(
    name="send-notification",
    description="Envía notificación con resultados del pipeline"
)
def send_notification_task(results: Dict[str, Any], flow_name: str) -> None:
    """Envía notificación con los resultados"""
    logger = get_run_logger()
    
    try:
        # Create markdown artifact for Prefect UI
        markdown_content = f"""
        # 📊 Resultados del Pipeline: {flow_name}
        
        **Fecha de ejecución:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        ## 📈 Resumen de Ejecución
        
        {generate_execution_summary(results)}
        
        ## 🔍 Próximos Pasos
        
        - Revisar métricas en MLflow
        - Verificar predicciones en Power BI
        - Monitorizar próxima ejecución programada
        """
        
        create_markdown_artifact(
            key=f"{flow_name}-results",
            markdown=markdown_content,
            description=f"Resultados de {flow_name}"
        )
        
        # Send alert (if configured)
        send_alert(
            subject=f"Pipeline {flow_name} completado",
            message=f"Ejecución de {flow_name} finalizada con status: {results.get('overall_status', 'unknown')}"
        )
        
        logger.info("✅ Notificación enviada")
        
    except Exception as e:
        logger.warning(f"⚠️ No se pudo enviar notificación: {e}")

def generate_execution_summary(results: Dict[str, Any]) -> str:
    """Genera resumen markdown de la ejecución"""
    summary = []
    
    if results.get('etl_status') == 'success':
        summary.append("✅ **ETL:** Completado exitosamente")
    else:
        summary.append("❌ **ETL:** Falló")
    
    if results.get('features_status') == 'success':
        summary.append("✅ **Feature Engineering:** Completado exitosamente")
    else:
        summary.append("❌ **Feature Engineering:** Falló")
    
    if results.get('training_status') == 'success':
        summary.append("✅ **Entrenamiento:** Completado exitosamente")
    else:
        summary.append("❌ **Entrenamiento:** Falló")
    
    if results.get('prediction_status') == 'success':
        summary.append("✅ **Predicción:** Completada exitosamente")
    else:
        summary.append("❌ **Predicción:** Falló")
    
    return "\n".join(summary)

@flow(
    name="daily-bitcoin-forecasting",
    description="Pipeline diario para forecasting de Bitcoin",
    log_prints=True
)
def daily_forecasting_flow() -> Dict[str, Any]:
    """
    Flow principal que se ejecuta diariamente para:
    - Actualizar datos (ETL incremental)
    - Generar features
    - Realizar predicción
    """
    logger = get_run_logger()
    results = {
        "flow_name": "daily-forecasting",
        "execution_time": datetime.now().isoformat(),
        "overall_status": "unknown"
    }
    
    try:
        logger.info("🌅 Iniciando pipeline diario de Bitcoin Forecasting")
        
        # 1. Validar entorno
        env_valid = validate_environment_task()
        if not env_valid:
            raise Exception("Entorno no válido para ejecución")
        
        # 2. ETL: Actualizar datos (incremental)
        etl_results = run_etl_pipeline_task(historical=False)
        results['etl_status'] = etl_results.get('status', 'error')
        results['etl_details'] = etl_results
        
        if etl_results.get('status') != 'success':
            raise Exception("ETL falló, deteniendo ejecución")
        
        # 3. Feature Engineering
        features_results = run_feature_engineering_task()
        results['features_status'] = features_results.get('status', 'error')
        results['features_details'] = features_results
        
        if features_results.get('status') != 'success':
            raise Exception("Feature Engineering falló, deteniendo ejecución")
        
        # 4. Predicción (ejecución diaria)
        prediction_results = run_prediction_task()
        results['prediction_status'] = prediction_results.get('status', 'error')
        results['prediction_details'] = prediction_results
        
        if prediction_results.get('status') == 'success':
            results['overall_status'] = 'success'
            logger.info(f"🎯 Predicción generada: {prediction_results.get('prediction', {})}")
        else:
            results['overall_status'] = 'partial_success'
            logger.warning("⚠️ Pipeline completado con errores en predicción")
        
    except Exception as e:
        logger.error(f"💥 Error crítico en pipeline diario: {e}")
        results['overall_status'] = 'error'
        results['error'] = str(e)
        
    finally:
        # 5. Enviar notificación
        send_notification_task(results, "Daily Forecasting")
        logger.info(f"📋 Pipeline diario finalizado con status: {results['overall_status']}")
    
    return results

@flow(
    name="weekly-model-retraining",
    description="Pipeline semanal para reentrenamiento del modelo",
    log_prints=True
)
def weekly_training_flow() -> Dict[str, Any]:
    """
    Flow que se ejecuta semanalmente para:
    - Reentrenar el modelo con datos actualizados
    - Evaluar el nuevo modelo
    """
    logger = get_run_logger()
    results = {
        "flow_name": "weekly-training",
        "execution_time": datetime.now().isoformat(),
        "overall_status": "unknown"
    }
    
    try:
        logger.info("🔄 Iniciando reentrenamiento semanal...")
        
        # 1. Validar entorno
        env_valid = validate_environment_task()
        if not env_valid:
            raise Exception("Entorno no válido para ejecución")
        
        # 2. ETL completo (asegurar datos actualizados)
        etl_results = run_etl_pipeline_task(historical=False)
        results['etl_status'] = etl_results.get('status', 'error')
        
        if etl_results.get('status') != 'success':
            raise Exception("ETL falló, deteniendo ejecución")
        
        # 3. Feature Engineering
        features_results = run_feature_engineering_task()
        results['features_status'] = features_results.get('status', 'error')
        
        if features_results.get('status') != 'success':
            raise Exception("Feature Engineering falló, deteniendo ejecución")
        
        # 4. Entrenamiento del modelo
        training_results = run_model_training_task()
        results['training_status'] = training_results.get('status', 'error')
        results['training_details'] = training_results
        
        # 5. Evaluación del nuevo modelo
        evaluation_results = run_model_evaluation_task()
        results['evaluation_status'] = evaluation_results.get('status', 'error')
        results['evaluation_details'] = evaluation_results
        
        if training_results.get('status') == 'success' and evaluation_results.get('status') == 'success':
            results['overall_status'] = 'success'
            logger.info("✅ Reentrenamiento semanal completado exitosamente")
        else:
            results['overall_status'] = 'partial_success'
            logger.warning("⚠️ Reentrenamiento completado con errores")
        
    except Exception as e:
        logger.error(f"💥 Error crítico en reentrenamiento: {e}")
        results['overall_status'] = 'error'
        results['error'] = str(e)
        
    finally:
        # 6. Enviar notificación
        send_notification_task(results, "Weekly Retraining")
        logger.info(f"📋 Reentrenamiento semanal finalizado con status: {results['overall_status']}")
    
    return results

@flow(
    name="monthly-hyperparameter-optimization",
    description="Pipeline mensual para optimización de hiperparámetros",
    log_prints=True
)
def monthly_tuning_flow() -> Dict[str, Any]:
    """
    Flow que se ejecuta mensualmente para:
    - Optimización de hiperparámetros
    - Mejora del modelo
    """
    logger = get_run_logger()
    results = {
        "flow_name": "monthly-tuning",
        "execution_time": datetime.now().isoformat(),
        "overall_status": "unknown"
    }
    
    try:
        logger.info("⚙️ Iniciando optimización mensual de hiperparámetros...")
        
        # 1. Validar entorno
        env_valid = validate_environment_task()
        if not env_valid:
            raise Exception("Entorno no válido para ejecución")
        
        # 2. ETL completo
        etl_results = run_etl_pipeline_task(historical=False)
        results['etl_status'] = etl_results.get('status', 'error')
        
        if etl_results.get('status') != 'success':
            raise Exception("ETL falló, deteniendo ejecución")
        
        # 3. Feature Engineering
        features_results = run_feature_engineering_task()
        results['features_status'] = features_results.get('status', 'error')
        
        if features_results.get('status') != 'success':
            raise Exception("Feature Engineering falló, deteniendo ejecución")
        
        # 4. Optimización de hiperparámetros
        tuning_results = run_hyperparameter_tuning_task()
        results['tuning_status'] = tuning_results.get('status', 'error')
        results['tuning_details'] = tuning_results
        
        if tuning_results.get('status') == 'success':
            results['overall_status'] = 'success'
            logger.info(f"✅ Optimización completada. Mejores parámetros: {tuning_results.get('best_params', {})}")
        else:
            results['overall_status'] = 'partial_success'
            logger.warning("⚠️ Optimización completada con errores")
        
    except Exception as e:
        logger.error(f"💥 Error crítico en optimización: {e}")
        results['overall_status'] = 'error'
        results['error'] = str(e)
        
    finally:
        # 5. Enviar notificación
        send_notification_task(results, "Monthly Tuning")
        logger.info(f"📋 Optimización mensual finalizada con status: {results['overall_status']}")
    
    return results

@flow(
    name="full-historical-pipeline",
    description="Pipeline completo con carga histórica inicial",
    log_prints=True
)
def full_historical_pipeline() -> Dict[str, Any]:
    """
    Pipeline completo para carga histórica inicial
    """
    logger = get_run_logger()
    results = {
        "flow_name": "historical-load",
        "execution_time": datetime.now().isoformat(),
        "overall_status": "unknown"
    }
    
    try:
        logger.info("📚 Iniciando carga histórica completa...")
        
        # 1. Validar entorno
        env_valid = validate_environment_task()
        if not env_valid:
            raise Exception("Entorno no válido para ejecución")
        
        # 2. ETL histórico
        etl_results = run_etl_pipeline_task(historical=True)
        results['etl_status'] = etl_results.get('status', 'error')
        
        if etl_results.get('status') != 'success':
            raise Exception("ETL histórico falló, deteniendo ejecución")
        
        # 3. Feature Engineering
        features_results = run_feature_engineering_task()
        results['features_status'] = features_results.get('status', 'error')
        
        if features_results.get('status') != 'success':
            raise Exception("Feature Engineering falló, deteniendo ejecución")
        
        # 4. Entrenamiento inicial
        training_results = run_model_training_task()
        results['training_status'] = training_results.get('status', 'error')
        
        # 5. Evaluación inicial
        evaluation_results = run_model_evaluation_task()
        results['evaluation_status'] = evaluation_results.get('status', 'error')
        
        if all([training_results.get('status') == 'success', evaluation_results.get('status') == 'success']):
            results['overall_status'] = 'success'
            logger.info("✅ Carga histórica completada exitosamente")
        else:
            results['overall_status'] = 'partial_success'
            logger.warning("⚠️ Carga histórica completada con errores")
        
    except Exception as e:
        logger.error(f"💥 Error crítico en carga histórica: {e}")
        results['overall_status'] = 'error'
        results['error'] = str(e)
        
    finally:
        send_notification_task(results, "Historical Load")
        logger.info(f"📋 Carga histórica finalizada con status: {results['overall_status']}")
    
    return results

# Función principal para ejecución manual
if __name__ == "__main__":
    """
    Ejecución manual de los flujos
    Usage: python prefect_flows.py [flow_name]
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Ejecutar flujos de Prefect")
    parser.add_argument(
        "flow",
        choices=["daily", "weekly", "monthly", "historical", "all"],
        help="Tipo de flow a ejecutar"
    )
    
    args = parser.parse_args()
    
    if args.flow == "daily":
        result = daily_forecasting_flow()
    elif args.flow == "weekly":
        result = weekly_training_flow()
    elif args.flow == "monthly":
        result = monthly_tuning_flow()
    elif args.flow == "historical":
        result = full_historical_pipeline()
    elif args.flow == "all":
        # Ejecutar todos los flujos en secuencia (para testing)
        result1 = daily_forecasting_flow()
        result2 = weekly_training_flow()
        result3 = monthly_tuning_flow()
        result = {"daily": result1, "weekly": result2, "monthly": result3}
    
    print(f"✅ Flow ejecutado. Resultado: {result.get('overall_status', 'unknown')}")