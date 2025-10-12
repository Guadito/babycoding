import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
from itertools import combinations
from scipy.stats import wilcoxon
import json
import os
from datetime import datetime
from .config import *
from .gain_function import calcular_ganancia, ganancia_lgb_binary, ganancia_threshold
from .output_manager import *
from .grafico_test import *

logger = logging.getLogger(__name__)



#-----------------------------------------------------------  optim bayesiana con cv

def objetivo_ganancia_cv(trial, df) -> float: 
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos

  
    Description:
    Función objetivo que maximiza ganancia con k-fold.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento con kfold y un porcentaje de la clase CONTINÚA
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON
  
    Returns:
    float: ganancia total
    """
    # Hiperparámetros a optimizar
    params = {
        'objective': 'binary',
        'metric': 'None',  
        'learning_rate': trial.suggest_float('learning_rate', PARAMETROS_LGBM['learning_rate'][0], PARAMETROS_LGBM['learning_rate'][1]),
        'num_leaves': trial.suggest_int('num_leaves', PARAMETROS_LGBM['num_leaves'][0], PARAMETROS_LGBM['num_leaves'][1]),
        'max_depth': trial.suggest_int('max_depth', PARAMETROS_LGBM['max_depth'][0], PARAMETROS_LGBM['max_depth'][1]),
        'min_child_samples': trial.suggest_int('min_child_samples', PARAMETROS_LGBM['min_child_samples'][0], PARAMETROS_LGBM['min_child_samples'][1]),
        'subsample': trial.suggest_float('subsample', PARAMETROS_LGBM['subsample'][0], PARAMETROS_LGBM['subsample'][1]),
        'colsample_bytree': trial.suggest_float('colsample_bytree', PARAMETROS_LGBM['colsample_bytree'][0], PARAMETROS_LGBM['colsample_bytree'][1]),
        'max_bin': trial.suggest_int('max_bin', PARAMETROS_LGBM['max_bin'][0], PARAMETROS_LGBM['max_bin'][1]),
        'min_split_gain': trial.suggest_float('min_split_gain', PARAMETROS_LGBM['min_split_gain'][0], PARAMETROS_LGBM['min_split_gain'][1]),
        'verbosity': -1,
        'random_state': SEMILLAS[0],
        'zero_as_missing': trial.suggest_categorical('zero_as_missing', [True, False]) 
        }
    
    num_boost_round = trial.suggest_int('num_boost_round', PARAMETROS_LGBM['num_boost_round'][0], PARAMETROS_LGBM['num_boost_round'][1])
    undersampling_ratio = PARAMETROS_LGBM['undersampling']


    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(GENERAL_TRAIN, list):
        train_data = df[df['foto_mes'].isin(GENERAL_TRAIN)]
    else: train_data = df[df['foto_mes'] == GENERAL_TRAIN]

    logger.info(
    f"Dataset GENERAL TRAIN - Antes del subsampleo de la clase CONTINUA: "
    f"Clase 1: {len(train_data[train_data['clase_ternaria'] == 1])}, "
    f"Clase 0: {len(train_data[train_data['clase_ternaria'] == 0])}"
    )

    # SUBMUESTREO
    clase_1 = train_data[train_data['clase_ternaria'] == 1]
    clase_0 = train_data[train_data['clase_ternaria'] == 0]    
    
    semilla = SEMILLAS[0] if isinstance(SEMILLAS, list) else SEMILLAS
    clase_0_sample = clase_0.sample(frac=undersampling_ratio, random_state=semilla)
    train_data = pd.concat([clase_1, clase_0_sample], axis=0).sample(frac=1, random_state=semilla)
    

    X_train = train_data.drop(columns = ['clase_ternaria'])
    y_train = train_data['clase_ternaria']
    

    lgb_train = lgb.Dataset(X_train, label=y_train)
    

    cv_results = lgb.cv(params,
                    num_boost_round=num_boost_round,
                    nfold=5,
                    stratified=True,                 
                    train_set=lgb_train,
                    shuffle = True,  
                    seed = SEMILLAS[0],
                    feval=ganancia_threshold,   #METRIC SE LA DECLARA VACÍA Y EN SU LUGAR SE USA FEVAL
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)])
    
    # Predecir probabilidades y binarizar
    ganancias_cv = cv_results['valid ganancia-mean']
    ganancia_maxima = np.max(ganancias_cv)
    best_iter = np.argmax(ganancias_cv)+1

    #Guardar el nro original de árboles y el optimizado:
    num_boost_round_original = trial.params['num_boost_round']
    trial.set_user_attr('num_boost_round_original', num_boost_round_original)
    trial.set_user_attr('best_iteration', int(best_iter)) 
    trial.params['num_boost_round'] = int(best_iter)


    
    logger.debug(f"Trial {trial.number}: Ganancia = {ganancia_maxima:,.0f}")
    logger.debug(f"Trial {trial.number}: Mejor iteracion = {best_iter:,.0f}")
    
    guardar_iteracion_cv(trial, ganancia_maxima, ganancias_cv, archivo_base=None)
    
    return ganancia_maxima


#---------------------------------------------------------------> Parametrización OPTUNA + aplicación de OB

def optimizar_cv(df, n_trials=int, study_name: str = None ) -> optuna.Study:
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

    study_name = STUDY_NAME


    logger.info(f"Iniciando optimización con CV {n_trials} trials")
    logger.info(f"Configuración: TRAIN para CV={GENERAL_TRAIN}, SEMILLA={SEMILLAS[0]}")
  
        # Crear estudio de Optuna
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler = optuna.samplers.TPESampler(seed= SEMILLAS[0]) 
        #storage="sqlite:///optuna_studies.db",
        #load_if_exists=True
    )

    # Aquí iría tu función objetivo y la optimización
    study.optimize(lambda trial: objetivo_ganancia_cv(trial, df), n_trials=n_trials)

    # Resultados
    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"Optimizacion CV completada. Mejor ganancia promedio: {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")
    logger.info(f"Total trials: {len(study.trials)}")

    return study
    

# ---------------------------> wilcoxon
def evaluar_wilcoxon(df: pd.DataFrame, top_params: list, n_seeds: int = 10) -> dict:
    """
    Evalúa los parámetros de los mejores modelos con n_seeds, calculando la ganancia por seed.
    Realiza pruebas de Wilcoxon pareadas entre todos los modelos y crea un ranking.

    Args
    ----
    df : pd.DataFrame
    top_params : list
        Lista de diccionarios con hiperparámetros a evaluar.
    n_seeds : int, optional
        Número de semillas. Por defecto 10.

    Returns
    -------
    dict con las siguientes claves:
        'mejor_modelo' : int
            Índice en top_params del modelo ganador (según ranking).
        'mejor_params' : dict
            Hiperparámetros del modelo ganador.
        'ranking' : list of tuples
            Lista ordenada [(idx, n_victorias, mediana_ganancia), ...] ordenada por victorias y mediana.
        'ganancias_por_seed' : list of lists
            Ganancias por seed para cada modelo: [[g1_seed1, ...], [g2_seed1, ...], ...].
        'wilcoxon_pvals' : dict
            Diccionario con p-values por par: keys = (i, j) -> p-value (i<j).
    """

    if len(top_params) < 2:
        logger.warning("Se necesitan al menos 2 modelos para comparar con Wilcoxon.")
        return {
            'mejor_modelo': 0 if top_params else None,
            'mejor_params': top_params[0] if top_params else None,
            'ranking': [],
            'ganancias_por_seed': [],
            'wilcoxon_pvals': {}
        }

    logger.info(f"Evaluando {len(top_params)} modelos con {n_seeds} semillas cada uno...")

    df_train = df[df['foto_mes'].isin(GENERAL_TRAIN)]
    df_test = df[df['foto_mes'] == MES_TEST]
    X_train = df_train.drop(columns=['clase_ternaria'])
    y_train = df_train['clase_ternaria']
    X_test = df_test.drop(columns=['clase_ternaria'])
    y_test = df_test['clase_ternaria']


    ganancias_top = []
    pvalues_dict = {}
 
    
    for idx, params in enumerate(top_params):
        ganancias = []
        logger.info(f"Modelo {idx + 1}/{len(top_params)}...")

        for seed in range(n_seeds):
            params_copy = params.copy()
            params_copy['seed'] = seed
            #Se toma la cantidad óptima de iteraciones
            num_boost_round = params_copy.pop('best_iteration', None)
            if num_boost_round is None:
                num_boost_round = params_copy.pop('num_boost_round', 200)

            # Crear Dataset con los parámetros que afectan su construcción
            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

            model = lgb.train(
                params_copy, 
                train_data,
                num_boost_round=num_boost_round, 
                feval=ganancia_threshold,
                callbacks=[lgb.log_evaluation(0)]
            )

            y_pred_prob = model.predict(X_test)
            ganancias_acum, _, _ = calcular_ganancia_acumulada_optimizada(y_test, y_pred_prob)
            g = ganancias_acum.max()  # Solo la ganancia máxima
            ganancias.append(g)

        ganancias_top.append(ganancias)
        logger.info(f"  Modelo {idx}: Mediana {np.median(ganancias):,.0f}")
    


    # Comparaciones Wilcoxon
    logger.info("\nComparaciones Wilcoxon:")
    n_modelos = len(ganancias_top)
    victorias = [0] * n_modelos
    
    for i, j in combinations(range(n_modelos), 2):
        try:
            _, p = wilcoxon(ganancias_top[i], ganancias_top[j])
        except ValueError:
            p = 1.0  # si no se puede comparar (mismas ganancias o longitudes)
        pvalues_dict[(i, j)] = p
        
        if p < 0.05:
            ganador = i if np.median(ganancias_top[i]) > np.median(ganancias_top[j]) else j
            victorias[ganador] += 1
            logger.info(f"  Modelo {ganador} > Modelo {i if ganador==j else j} (p={p:.3f})")
    
    # --- Ranking final ---
    ranking = [(idx, victorias[idx], np.median(ganancias_top[idx])) for idx in range(n_modelos)]
    ranking.sort(key=lambda x: (x[1], x[2]), reverse=True)

    logger.info("\nRanking final:")
    for rank, (idx, vict, med) in enumerate(ranking, 1):
        logger.info(f"  #{rank} Modelo {idx}: {vict} victorias | Mediana {med:,.0f}")

    mejor_modelo = ranking[0][0]
    logger.info(f"Mejor modelo: {mejor_modelo}")

    
    return {
        'mejor_modelo': mejor_modelo,
        'mejor_params': top_params[mejor_modelo],
        'ranking': ranking,
        'ganancias_por_seed': ganancias_top,
        'wilcoxon_pvals': pvalues_dict
    }


# -------------------------------> evaluar modelo considerando el punto máximo de ganancia

def evaluar_modelo_optimizado (df: pd.DataFrame, mejores_params: dict) -> tuple:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Encuentra automáticamente el umbral óptimo que maximiza la ganancia.
 
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
 
    Returns:
        tuple: Resultados de la evaluación + predicciones + umbral óptimo
    """
    logger.info(f"Períodos de TRAIN: {GENERAL_TRAIN}, Período de test: {MES_TEST}")
 
    # Preparar datos de entrenamiento
    if isinstance(GENERAL_TRAIN, list):
        periodos_entrenamiento = GENERAL_TRAIN
    else:
        periodos_entrenamiento = GENERAL_TRAIN
 
    df_train_completo = df[df['foto_mes'].isin(periodos_entrenamiento)]
    df_test = df[df['foto_mes'] == MES_TEST]
    X_train_completo = df_train_completo.drop(columns=['clase_ternaria'])
    y_train_completo = df_train_completo['clase_ternaria']
    X_test = df_test.drop(columns=['clase_ternaria'])
    y_test = df_test['clase_ternaria']
 
    train_data = lgb.Dataset(X_train_completo, label=y_train_completo)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Copiar los parámetros para no modificar el dict original
    mejores_params = mejores_params.copy()

    # Tomar la iteración óptima si existe
    num_boost_round = mejores_params.pop('best_iteration', None)
    if num_boost_round is None:
        num_boost_round = mejores_params.pop('num_boost_round', 200)  # fallback
    
    # Entrenar modelo con mejores parámetros
    model = lgb.train(mejores_params, train_data,  num_boost_round=num_boost_round, feval=ganancia_threshold)
    
    # Predecir probabilidades
    y_pred_prob = model.predict(X_test)
    
    ganancias_acum, indices_ord, umbral_optimo = calcular_ganancia_acumulada_optimizada(y_test,y_pred_prob)
    
    logger.info(f"Umbral óptimo encontrado: {umbral_optimo:.6f}")
    logger.info(f" Ganancia máxima: {max(ganancias_acum):,.0f}")
    
    # Ahora SÍ binarizar con el umbral óptimo
    y_pred_binary = (y_pred_prob > umbral_optimo).astype(int)
 
    # Estadísticas básicas
    total_predicciones = len(y_pred_binary)
    predicciones_positivas = np.sum(y_pred_binary == 1)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
    verdaderos_positivos = np.sum((y_pred_binary == 1) & (y_test == 1))
    falsos_positivos = np.sum((y_pred_binary == 1) & (y_test == 0))
    verdaderos_negativos = np.sum((y_pred_binary == 0) & (y_test == 0))
    falsos_negativos = np.sum((y_pred_binary == 0) & (y_test == 1))
    precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos + 1e-10)
    recall = verdaderos_positivos / (verdaderos_positivos + falsos_negativos + 1e-10)
    accuracy = (verdaderos_positivos + verdaderos_negativos) / total_predicciones
    
    resultados_test = {
        'umbral_optimo': float(umbral_optimo),  
        'ganancia_maxima': float(max(ganancias_acum)), 
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas),
        'verdaderos_positivos': int(verdaderos_positivos),
        'falsos_positivos': int(falsos_positivos),
        'verdaderos_negativos': int(verdaderos_negativos),
        'falsos_negativos': int(falsos_negativos),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'timestamp': datetime.now().isoformat()
    }
 
    guardar_resultados_test(resultados_test)
    graficar_importances_test(model)
    
    
    return resultados_test, y_pred_binary, y_test, y_pred_prob, umbral_optimo


#-----------------------------------------------> evalua el modelo en test

# def evaluar_modelo (df: pd.DataFrame, mejores_params:dict) -> tuple:
#     """
#     Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
#     Solo calcula la ganancia.
  
#     Args:
#         df: DataFrame con todos los datos
#         mejores_params: Mejores hiperparámetros encontrados por Optuna
  
#     Returns:
#         dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
#     """
#     logger.info(f"Períodos de TRAIN: {GENERAL_TRAIN}, Período de test: {MES_TEST}")
  
#     # Preparar datos de entrenamiento (TRAIN + VALIDACION)
#     if isinstance(GENERAL_TRAIN, list):
#         periodos_entrenamiento = GENERAL_TRAIN
#     else:
#         periodos_entrenamiento = GENERAL_TRAIN
  
#     df_train_completo = df[df['foto_mes'].isin(periodos_entrenamiento)]
#     df_test = df[df['foto_mes'] == MES_TEST]


#     X_train_completo = df_train_completo.drop(columns = ['clase_ternaria'])
#     y_train_completo = df_train_completo['clase_ternaria']

#     X_test = df_test.drop(columns = ['clase_ternaria'])
#     y_test = df_test['clase_ternaria']
  
    
#     train_data = lgb.Dataset(X_train_completo, label=y_train_completo)
#     test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)


#     # Entrenar modelo con mejores parámetros
#     model = lgb.train(mejores_params, 
#                       train_data,
#                       feval=ganancia_threshold
#                     )   


#     # Predecir probabilidades y binarizar
#     y_pred_prob = model.predict(X_test)
#     y_pred_binary = (y_pred_prob > 0.025).astype(int) 
  
#     # Calcular solo la ganancia
#     ganancia_test = calcular_ganancia(y_test, y_pred_binary)
  
#     # Estadísticas básicas
#     total_predicciones = len(y_pred_binary)
#     predicciones_positivas = np.sum(y_pred_binary == 1)
#     porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
#     verdaderos_positivos = np.sum((y_pred_binary == 1) & (y_test == 1))
#     falsos_positivos = np.sum((y_pred_binary == 1) & (y_test == 0))
#     verdaderos_negativos = np.sum((y_pred_binary == 0) & (y_test == 0))
#     falsos_negativos = np.sum((y_pred_binary == 0) & (y_test == 1))

#     precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos + 1e-10)  # para evitar división por cero
#     recall = verdaderos_positivos / (verdaderos_positivos + falsos_negativos + 1e-10)
#     accuracy = (verdaderos_positivos + verdaderos_negativos) / total_predicciones

#     resultados_test = {
#         'ganancia_test': float(ganancia_test),
#         'total_predicciones': int(total_predicciones),
#         'predicciones_positivas': int(predicciones_positivas),
#         'porcentaje_positivas': float(porcentaje_positivas),
#         'verdaderos_positivos': int(verdaderos_positivos),
#         'falsos_positivos': int(falsos_positivos),
#         'verdaderos_negativos': int(verdaderos_negativos),
#         'falsos_negativos': int(falsos_negativos),
#         'precision': float(precision),
#         'recall': float(recall),
#         'accuracy': float(accuracy),
#         'timestamp': datetime.now().isoformat()
#     }
  
#     guardar_resultados_test(resultados_test)

#     graficar_importances_test(model)


#     return resultados_test, y_pred_binary, y_test, y_pred_prob
