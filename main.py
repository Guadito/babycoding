import pandas as pd
import datetime
import os
import sys
import logging


from src.features import feature_engineering_lag
from src.config import *  #Llama a la información de config p/ correr
from src.loader import cargar_datos, convertir_clase_ternaria_a_target, clase_ternaria
from src.optimization import optimizar


## config basico logging
os.makedirs("logs", exist_ok=True)
fecha = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
nombre_log = f"log_{STUDY_NAME}_{fecha}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler("logs/" + nombre_log),
        logging.StreamHandler()
    ]
)


logger = logging.getLogger(__name__)
logger.info("Iniciando programa de optimización con log fechado")



### Manejo de Configuración en YAML ###
logger.info("Configuración cargada desde YAML")
logger.info(f"STUDY_NAME: {STUDY_NAME}")
logger.info(f"DATA_PATH: {DATA_PATH}")
logger.info(f"SEMILLAS: {SEMILLAS}")
logger.info(f"MES_TRAIN: {MES_TRAIN}")
logger.info(f"MES_VALIDACION: {MES_VAL}")
logger.info(f"MES_TEST: {MES_TEST}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")



def main():
    logger.info("Inicio de ejecucion.")

    # 1- cargar datos
    os.makedirs("Data", exist_ok=True)
    df_f = cargar_datos(DATA_PATH)
    
    # 2- definir clase ternaria 
    df_f = clase_ternaria(df_f)
    df_f = convertir_clase_ternaria_a_target (df_f)

    # 3- feature engineering  
    columnas_lag = ['mcuentas_saldo', 'mcuentas_saldo_descubierto', 'mcaja_ahorro_saldo', 'mtarjeta_visa_consumo', 'mtarjeta_master_consumo']
    cantidad_lag = 3
    df_f = feature_engineering_lag(df_f, columnas_lag, cantidad_lag)

    # 4 - optimización de hiperparámetros
    study = optimizar(df_f, n_trials = 100)  










    # 5 Análisis     logger.info("=== ANÁLISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
  
    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")

















    # 4 Guardar el DataFrame resultante
    #path = "Data/competencia_01_lag.csv"
    #df.to_csv(path, index=False)
    #logger.info(f"DataFrame resultante guardado en {path}")
    
    logger.info(f">>>> Ejecución finalizada <<<< Para más detalles, ver {nombre_log}")




if __name__ == "__main__":
    main()