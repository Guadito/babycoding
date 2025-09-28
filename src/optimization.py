# src/optimization.py
import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from .config import *
from .gain_function import calcular_ganancia, ganancia_lgb_binary

logger = logging.getLogger(__name__)




def objetivo_ganancia(trial, df) -> float: 
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos
  
    Description:
    Función objetivo que maximiza ganancia en mes de validación.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento y validación
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON
  
    Returns:
    float: ganancia total
    """
    # Hiperparámetros a optimizar
    params = {
        'objective': 'binary',
        'metric': 'None',  # Métrica personalizada
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'verbosity': -1,
        'random_state': SEMILLAS[0],  # Desde configuración YAML
    }
  
    
    train_data = df[df['foto_mes'] == MES_TRAIN]
    val_data = df[df['foto_mes'] == MES_VAL]
    X_train = train_data.drop(['clase_ternaria'])
    X_val = val_data.drop(['clase_ternaria'])
    y_train = train_data['clase_ternaria']
    y_val = val_data['clase_ternaria']


    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)


    model = lgb.train(params, lgb_train, num_boost_round=1000, 
                      valid_sets=[lgb_val], early_stopping_rounds=50,
                      feval=ganancia_lgb_binary)
    
    
    
    # Predecir probabilidades y binarizar
    y_pred_prob = model.predict(X_val)
    y_pred_binary = (y_pred_prob > 0.025).astype(int)  # mismo umbral que eval

     # Calcular ganancia usando tu función
    ganancia_total = calcular_ganancia(y_val, y_pred_binary)

    # Guardar iteración (opcional)
    guardar_iteracion(trial, ganancia_total)
  
    logger.debug(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")
  
    return ganancia_total







def guardar_iteracion(trial, ganancia, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.
  
    Args:
        trial: Trial de Optuna
        ganancia: Valor de ganancia obtenido
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
  
    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_iteraciones.json"
  
    # Datos de esta iteración
    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': float(ganancia),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',  # Si llegamos aquí, el trial se completó exitosamente
        'configuracion': {
            'semilla': SEMILLAS,
            'mes_train': MES_TRAIN,
            'mes_validacion': MES_VAL
        }
    }
  
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
  
    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)
  
    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {ganancia:,.0f}" + "---" + "Parámetros: {params}")









def optimizar(df, n_trials=int, study_name: str = None ) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
  
    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado. 
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """

    if study_name is None:
        study_name = STUDY_NAME


    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VAL}, SEMILLA={SEMILLAS}")
  
        # Crear estudio de Optuna
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage="sqlite:///optuna_studies.db",
        load_if_exists=True
    )

    # Aquí iría tu función objetivo y la optimización
    study.optimize(objetivo_ganancia(df), n_trials=n_trials)

    # Resultados
    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")

    return study






