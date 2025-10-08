# src/final_training.py
import pandas as pd
import lightgbm as lgb
import numpy as np
import logging
import os
from datetime import datetime
from .config import FINAL_TRAIN, FINAL_PREDICT, SEMILLAS
from .best_params import *
from .gain_function import *

logger = logging.getLogger(__name__)



def preparar_datos_entrenamiento_final(df: pd.DataFrame) -> tuple:
    """
    Prepara los datos para el entrenamiento final usando todos los períodos de FINAL_TRAIN.
  
    Args:
        df: DataFrame con todos los datos
  
    Returns:
        tuple: (X_train, y_train, X_predict, clientes_predict)
    """
    logger.info(f"Preparando datos para entrenamiento final")
    logger.info(f"Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"Período de predicción: {FINAL_PREDICT}")
  
    # Datos de entrenamiento: todos los períodos en FINAL_TRAIN
  
    # Datos de predicción: período FINAL_PREDIC 
    if isinstance(FINAL_TRAIN, list):
        train_data = df[df['foto_mes'].isin(FINAL_TRAIN)]
    else: train_data = df[df['foto_mes'] == FINAL_TRAIN]
    
    predict_data = df[df['foto_mes'] == FINAL_PREDICT]

    logger.info(f"Registros de entrenamiento: {len(train_data):,}")
    logger.info(f"Registros de predicción: {len(predict_data):,}")
  
    # Corroborar que no estén vacíos los df
    if train_data.empty:
        raise ValueError(f"No se encontraron datos de entrenamiento para foto_mes: {FINAL_TRAIN}")
    if predict_data.empty:
        raise ValueError(f"No se encontraron datos de predicción para foto_mes: {FINAL_PREDICT}")

    logger.info("Validación exitosa: ambos dataframes contienen datos")


    # Preparar features y target para entrenamiento
    X_train = train_data.drop(columns = ['clase_ternaria'])
    y_train = train_data['clase_ternaria']

    X_predict = predict_data.drop(columns = ['clase_ternaria'])
    clientes_predict = predict_data['numero_de_cliente']


    # Información sobre features y distribución
    logger.info(f"Features utilizadas: {X_train.shape[1]}")
    logger.info(f"Distribución del target en entrenamiento:")
    
        # Contar cada clase
    value_counts = y_train.value_counts()
    for clase, count in value_counts.items():
        logger.info(f"  Clase {clase}: {count:,} ({count/len(y_train)*100:.2f}%)")
  


    return X_train, y_train, X_predict, clientes_predict


#-----------------------------------------> entrenar modelo final


def entrenar_modelos_finales(X_train: pd.DataFrame, y_train: pd.Series, mejores_params: dict) -> list:
    """
    Entrena múltiples modelos con diferentes semillas.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        mejores_params: Mejores hiperparámetros de Optuna
    
    Returns:
        list: Lista de modelos entrenados
    """
    logger.info("Iniciando entrenamiento de modelos finales con múltiples semillas")
    
    modelos = []
    semillas = SEMILLAS if isinstance(SEMILLAS, list) else [SEMILLAS]
    undersampling_ratio = PARAMETROS_LGBM['undersampling']
    
    for idx, semilla in enumerate(semillas):
        logger.info(f"Entrenando modelo {idx+1}/{len(semillas)} con semilla {semilla}")
        
        # Configurar parámetros con la semilla actual
        params = {
            'objective': 'binary',
            'metric': None,  
            'random_state': semilla,
            'verbosity': -1,
            **mejores_params,
            
        }
        
        # Normalización si hubo undersampling en la optimización bayesiana
        if undersampling_ratio != 1.0 and 'min_data_in_leaf' in params:
            valor_original = params['min_data_in_leaf']
            params['min_data_in_leaf'] = max(1, round(valor_original / undersampling_ratio))
            logger.info(f"Ajustando min_data_in_leaf: {valor_original} -> {params['min_data_in_leaf']} (factor {undersampling_ratio})")
        
        # Crear dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Entrenar modelo
        modelo = lgb.train(
            params,
            train_data,
            feval=ganancia_threshold,
            num_boost_round=mejores_params.get('num_boost_round', 100)
        )
        
        modelos.append(modelo)
        logger.info(f"Modelo {idx+1} entrenado exitosamente")
    
    logger.info(f"Total de modelos entrenados: {len(modelos)}")

    # --- DEBUG: mostrar la semilla real usada por cada modelo ---
    for idx, modelo in enumerate(modelos):
        semilla_real = modelo.params.get('random_state')
        print(f"Modelo {idx+1} semilla real usada: {semilla_real}")

    return modelos


#----------------------------------------> generar predicciones finales

def generar_predicciones_finales(modelos: list, X_predict: pd.DataFrame, clientes_predict: np.ndarray, umbral: float = 0.029) -> pd.DataFrame:
    """
    Genera las predicciones finales para el período objetivo.
  
    Args:
        modelo: Modelo entrenado
        X_predict: Features para predicción
        clientes_predict: IDs de clientes
        umbral: Umbral para clasificación binaria
  
    Returns:
        pd.DataFrame: DataFrame con numero_cliente y predict
    """
    logger.info(f"Generando predicciones finales con {len(modelos)} modelos.")
  
    # Generar probabilidades con el modelo entrenado

    probabilidades_todos = []
    for idx, modelo in enumerate(modelos):
        proba = modelo.predict(X_predict)
        probabilidades_todos.append(proba)
        logger.debug(f"Predicciones del modelo {idx+1} generadas")


    # Promedio de probabilidades
    probabilidades_promedio = np.mean(probabilidades_todos, axis=0)
  
    # Convertir a predicciones binarias con el umbral establecido
    predicciones_binarias = (probabilidades_promedio >= umbral).astype(int)
  
    # Crear DataFrame de resultados
    resultados = pd.DataFrame({
        'numero_de_cliente': clientes_predict,
        'Predict': predicciones_binarias})
    
    # Estadísticas
    total_predicciones = len(resultados)
    predicciones_positivas = (resultados['Predict'] == 1).sum()
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
    
    logger.info(f"Predicciones generadas:")
    logger.info(f"  Total clientes: {total_predicciones:,}")
    logger.info(f"  Predicciones positivas: {predicciones_positivas:,} ({porcentaje_positivas:.2f}%)")
    logger.info(f"  Predicciones negativas: {total_predicciones - predicciones_positivas:,}")
    logger.info(f"  Umbral utilizado: {umbral}")
    logger.info(f"  Modelos promediados: {len(modelos)}")
  
    return resultados