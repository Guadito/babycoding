import pandas as pd
import datetime
import os
import sys
import logging
from sklearn.model_selection import train_test_split

from src.features import *
from src.config import  *
from src.loader import *
from src.optimization import *
from src.best_params import *
from src.grafico_test import *
from src.final_training import *

## config basico logging
os.makedirs("logs", exist_ok=True)
fecha = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
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
logger.info(f"TRAIN_OPTUNA: {GENERAL_TRAIN}")
logger.info(f"MES_TEST: {MES_TEST}")
logger.info(f"TRAIN_FINAL: {FINAL_TRAIN}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")



def main():
    logger.info("Inicio de ejecucion.")

    # 1- cargar datos
    os.makedirs("Data", exist_ok=True)
    df_f = cargar_datos(DATA_PATH)
    

    # 2- definir clase ternaria 
    df_f = crear_clase_ternaria(df_f)
    df_f = convertir_clase_ternaria_a_target (df_f)

    # #SAMPLE
    # n_sample = 100000
    # df_f, _ = train_test_split(
    #     df_f,
    #     train_size=n_sample,
    #     stratify=df_f['clase_ternaria'],
    #     random_state=42)




    # 3- feature engineering 
    #a) Ranking para columnas de monto
    col_montos = select_col_montos(df_f)
    df_f = feature_engineering_rank_pos(df_f, col_montos)

    #b) Lags y deltas para todas las columnas excepto ID cliente, foto_mes, clase.
    col = [c for c in df_f.columns if c not in ['numero_de_cliente', 'foto_mes', 'clase_ternaria']]
    df_f = feature_engineering_lag_delta_batch(df_f, col, cant_lag = 3)
    print(df_f.head)
   

    # 4 - optimización de hiperparámetros
    study = optimizar_cv(df_f, n_trials= 100)  


    # 5 Análisis 
    logger.info("=== ANÁLISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        
        for idx, trial in top_5.iterrows():
                logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
                
    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")



    # 6 Test en mes desconocido
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    
    # Cargar mejores hiperparámetros
    best_params = cargar_mejores_hiperparametros()


    # Evaluar en test
    resultados_test, y_pred_binary, y_test, y_pred_prob = evaluar_modelo(df_f, best_params)

    # Resumen de evaluación en test
    logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
    logger.info(f"Ganancia en test: {resultados_test['ganancia_test']:,.0f}")
    logger.info(f"Predicciones positivas: {resultados_test['predicciones_positivas']:,} ({resultados_test['porcentaje_positivas']:.2f}%)")

    # Grafico de test
    logger.info("=== GRAFICO DE TEST ===")
    ruta_grafico_avanzado = crear_grafico_ganancia_avanzado(y_true=y_test, y_pred_proba=y_pred_prob)
    logger.info(f"✅ Gráficos generados: {ruta_grafico_avanzado}")


    # 7 Entrenar modelo final
    logger.info("=== ENTRENAMIENTO FINAL ===")
    logger.info("Preparar datos para entrenamiento final")
    X_train, y_train, X_predict, clientes_predict = preparar_datos_entrenamiento_final(df_f)

    # Entrenar modelo final
    logger.info("Entrenar modelo final")
    modelo_final = entrenar_modelos_finales(X_train, y_train, best_params)

    # Generar predicciones finales
    logger.info("Generar predicciones finales")
    resultados = generar_predicciones_finales(modelo_final, X_predict, clientes_predict)
  
    guardar_predicciones_finales(resultados)



    # 4 Guardar el DataFrame resultante
    #path = "Data/competencia_01_lag.csv"
    #df.to_csv(path, index=False)
    #logger.info(f"DataFrame resultante guardado en {path}")
    
    logger.info(f">>>> Ejecución finalizada <<<< Para más detalles, ver {nombre_log}")




if __name__ == "__main__":
    main()