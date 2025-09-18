import pandas as pd
import datetime
import os
import sys
import logging

from src.loader import cargar_datos
from src.features import feature_engineering_lag



## config basico logging
os.makedirs("logs", exist_ok=True)
fecha = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
nombre_log = f"log_{fecha}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{nombre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)




def main():
    logger.info("Inicio de ejecucion.")

    # 1 cargar datos
    os.makedirs("Data", exist_ok=True)
    path = "Data/competencia_01_crudo.csv"
    df = cargar_datos(path) 

    # 2 feature engineering - 
    columnas_lag = ['mcuentas_saldo', 'mcuentas_saldo_descubierto', 'mcaja_ahorro_saldo', 'mtarjeta_visa_consumo', 'mtarjeta_master_consumo']
    cantidad_lag = 3
    df = feature_engineering_lag(df, columnas_lag, cantidad_lag)



    # 4 Guardar el DataFrame resultante
    #path = "Data/competencia_01_lag.csv"
    #df.to_csv(path, index=False)
    #logger.info(f"DataFrame resultante guardado en {path}")
    
    
    
    
    
    
    
    logger.info(f">>>> Ejecución finalizada <<<< Para más detalles, ver {nombre_log}")






if __name__ == "__main__":
    main()