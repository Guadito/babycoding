

IDEAS:
-probar objetivo_ganancia con cross validation ok
-subsamplear el dataset para bayesiana ok 

----

RECORDATORIOS:
Quedan estas dos columnas dentro del ranking aunque no son montos: No me parecen importantes
_mantenimiento
ctarjeta_master_descuentos
ccomisiones_mantenimiento


-------------
experimentos: 

0 - lags y deltas x tres meses, ranking general.
1 - lags y deltas x dos meses, ranking general.   o pts
2 - lags y deltas x tres meses, ranking general. (competencias_1_2_iteraciones)  6pts
3 - lags y deltas x tres meses, ranking percentiles negativo que va entre -1, 0, 1.  (pisé el anterior) (competencias_1_2_iteraciones)  3 pts
4 - lags y deltas x tres meses, ranking percentiles positivo  competencias_1_3 (HP muy largos)
5 - lags y deltas x tres meses, ranking percentiles positivo y corto dataset en 0.029 competencias_1_4 6pts
    agrego los siguientes HP. 
    num_boost_round: [100, 500]
    min_split_gain: [0.0, 1.0]
6 - Saco ranking.   competencias_1_5 3pts
6 - Prueba de semillas para correcto ensamble y de gráfico feature importances competencias_1_6 5pts
7 - Agrego ranking positivo [-1:0, 0:1] y bajo a 2 lags / delta.  5pts
8 - Subo a 4 lag y deltas: empeora, no lo pruebo en kaggle.   competencias_1_7
9 - Vuelvo a 3 lags y cambio a métrica AUC para entrenar. competencias_1_8        -10.000
10 - Pruebo un undersampling de 0.5   competencias_1_9
11 - Con undersampling de 0.2, zero_as_missing: [True, False] y learning_rate: [0.07, 0.3]
11 - Pruebo un undersampling de 0.1, zero_as_missing: [True, False] y learning_rate: [0.07, 0.3] competencias_1_11 - 
11 - En GCP: aumento undersampling a 0.4. Bajo nuevamente learning_rate a [0.01, 0.3]. competencias_1_11_gcp
12 - Vuelvo a learning rate [0.01, 0.3], undersampling 0.1, y corrijo min_split_gain: [0.0, 0.1] como float
Volví a usar la función evaluar_modelo sin optimizar porque los umbrales de corte que me daba no me daban buenos resultdos. 
hago BO para búsqueda de hiperparámetros con ganancia_th, 
hago wilcoxon para corroborar mejor modelo, 
testeo con el umbral 0.025, veo el límite de envíos, 
reentreno con ganancia_th  
predigo con umbrales=[0.025, 0.029, 0.032]       competencias_1_12_gcp
13 - En GCP: sampleo 0.2 para OB. 



Post Mortem:






buscar método para pesar las clases baja+1 y baja+2













--------------------------

SQL
-range framing: ej promedio en un tiempo determinado 

 RANGE Framing

Returning to the power data, suppose the data is noisy. We might want to compute a 7 day moving average for each plant to smooth out the noise. To do this, we can use this window query:

SELECT "Plant", "Date",
    AVG("MWh") OVER (
        PARTITION BY "Plant"
        ORDER BY "Date" ASC
        RANGE BETWEEN INTERVAL 3 DAYS PRECEDING
                  AND INTERVAL 3 DAYS FOLLOWING)
        AS "MWh 7-day Moving Average"
FROM "Generation History"
ORDER BY 1, 2

This query partitions the data by Plant (to keep the different power plants' data separate), orders each plant's partition by Date (to put the energy measurements next to each other), and uses a RANGE frame of three days on either side of each day for the AVG (to handle any missing days). This is the result:


----------------------------
VERSIONES:
    
VERSIÓN CON PYTHON FE_RANK

# def feature_engineering_rank(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
#     """
#     Genera variables de ranking para los atributos especificados utilizando pandas.
#     Cada ranking se calcula por 'foto_mes' y se escala entre 0 y 1 (percentil).

#     Parameters:
#     -----------
#     df : pd.DataFrame
#         DataFrame con los datos
#     columnas : list
#         Lista de atributos para los cuales generar rankings. Si es None, no se generan rankings.
      
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame con las variables de ranking agregadas
#     """

#     logger.info(f"Realizando feature engineering para {len(columnas) if columnas else 0} atributos")

#     if columnas is None or len(columnas) == 0:
#         logger.warning("No se especificaron atributos para generar rankings")
#         return df.copy()

#     df_result = df.copy()

#     for col in columnas:
#         if col in df.columns:
#             rank_col_name = f"{col}_rank"
#             df_result[rank_col_name] = df_result.groupby("foto_mes")[col].rank(pct=True)
#         else:
#             logger.warning(f"El atributo {col} no existe en el DataFrame")

#     logger.info(f"Feature engineering completado. DataFrame resultante con {df_result.shape[1]} columnas")
#     return df_result


VERSIÓN DE FE RANK CON SQL Y SIN REEMPLAZO DE COLS
(GENERA COL + COL_1)
def feature_engineering_rank(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera variables de ranking para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list[str]
        Lista de atributos para los cuales generar rankings.
      
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de ranking agregadas
    """

    if not columnas:
        raise ValueError("La lista de columnas no puede estar vacía")
    
    columnas_validas = [col for col in columnas if col in df.columns]
    if not columnas_validas:
        raise ValueError("Ninguna de las columnas especificadas existe en el DataFrame")

    logger.info(f"Realizando feature engineering RANK para {len(columnas_validas)} columnas: {columnas_validas}")
    
    con = duckdb.connect(database=":memory:")
    con.register("df_temp", df)

    # rank_expressions = [
    #     f"PERCENT_RANK() OVER (PARTITION BY foto_mes ORDER BY {col}) AS {col}_rank"
    #     for col in columnas_validas
    # ]

    rank_expressions = [
    f"PERCENT_RANK() OVER (PARTITION BY foto_mes ORDER BY {col}) AS {col}"
    for col in columnas_validas
   ]

    sql = f"""
    SELECT *,
           {', '.join(rank_expressions)}
    FROM df_temp
    """
    
    try:
        resultado = con.execute(sql).df()
        logger.info(f"Feature engineering completado. DataFrame resultante con {resultado.shape[1]} columnas")
        return resultado
    finally:
        con.close()

#--------------------------------> lag y delta duckdb


def feature_engineering_lag_delta_duckdb(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables de lag y delta trabajando completamente en DuckDB.
    Solo convierte a Pandas al final, evitando problemas de memoria.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos originales.
    columnas : list[str]
        Lista de atributos para los cuales generar lags y deltas.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo.
    
    Returns
    -------
    pd.DataFrame
        DataFrame con las variables de lag y delta agregadas.
    """
    logger.info(f"Generando {cant_lag} lags y deltas para {len(columnas)} atributos usando DuckDB")
    
    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags/deltas")
        return df
    
    # Validar columnas requeridas
    if 'numero_de_cliente' not in df.columns or 'foto_mes' not in df.columns:
        logger.error("El DataFrame debe contener 'numero_de_cliente' y 'foto_mes'")
        raise ValueError("Columnas requeridas no encontradas")
    
    # Crear conexión persistente en memoria
    con = duckdb.connect(database=':memory:')
    
    try:
        # Registrar el DataFrame como tabla
        con.register('df_original', df)
        
        # Construir la consulta SQL completa
        sql = "CREATE TABLE df_result AS SELECT *"
        
        for attr in columnas:
            if attr in df.columns:
                for i in range(1, cant_lag + 1):
                    sql += f', LAG("{attr}", {i}) OVER w AS "{attr}_lag_{i}"'
                    sql += f', ("{attr}" - LAG("{attr}", {i}) OVER w) AS "{attr}_delta_{i}"'
            else:
                logger.warning(f"El atributo {attr} no existe en el DataFrame")
        
        sql += " FROM df_original WINDOW w AS (PARTITION BY numero_de_cliente ORDER BY foto_mes)"
        
        logger.info("Ejecutando query en DuckDB...")
        
        # Ejecutar la creación de la tabla
        con.execute(sql)
        
        logger.info("Query ejecutada. Convirtiendo a Pandas...")
        
        # Ahora convertir a Pandas con Arrow (más eficiente)
        df_result = con.execute("SELECT * FROM df_result").fetch_df()
        
        # Convertir columnas float64 a float32 para ahorrar memoria
        float_cols = df_result.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            logger.info(f"Convirtiendo {len(float_cols)} columnas a float32")
            df_result[float_cols] = df_result[float_cols].astype('float32')
        
        logger.info(f"Feature engineering completado. DataFrame resultante con {df_result.shape[1]} columnas")
        
        return df_result
        
    finally:
        # Limpiar
        con.close()




        
#-----------------------------> creación de lAG y DELTA

# def feature_engineering_lag_delta(df: pd.DataFrame, columnas: list[str], cant_lag: int=1) -> pd.DataFrame:
#     """
#     Genera variables de lag y sus deltas para los atributos especificados utilizando SQL.
  
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         DataFrame con los datos
#     columnas : list
#         Lista de atributos para los cuales generar lags y deltas. Si es None, no se generan.
#     cant_lag : int, default=1
#         Cantidad de lags a generar para cada atributo
  
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame con las variables de lag y delta agregadas
#     """
#     logger.info(f"Generando {cant_lag} lags y deltas para {len(columnas) if columnas else 0} atributos")

#     print(f"DEBUG FUNCIÓN - df shape: {df.shape}")
#     print(f"DEBUG FUNCIÓN - len(columnas): {len(columnas)}")
#     print(f"DEBUG FUNCIÓN - cant_lag: {cant_lag}")
#     print(f"DEBUG FUNCIÓN - columnas esperadas: {df.shape[1]} + {len(columnas) * cant_lag * 2} = {df.shape[1] + len(columnas) * cant_lag * 2}")


    
#     if columnas is None or len(columnas) == 0:
#         logger.warning("No se especificaron atributos para generar lags/deltas")
#         return df
    
#     # Validar columnas requeridas
#     if 'numero_de_cliente' not in df.columns or 'foto_mes' not in df.columns:
#         logger.error("El DataFrame debe contener 'numero_de_cliente' y 'foto_mes'")
#         raise ValueError("Columnas requeridas no encontradas")
    
#     # Construir la consulta SQL - Usar * para mantener todas las columnas
#     sql = "SELECT *"
    
#     # Agregar los lags y deltas para cada atributo
#     for attr in columnas:
#         if attr in df.columns:
#             for i in range(1, cant_lag + 1):
#                 sql += f', lag("{attr}", {i}) OVER w AS "lag_{i}_{attr}"'
#                 sql += f', "{attr}" - lag("{attr}", {i}) OVER w AS "delta_{i}_{attr}"'
#         else:
#             logger.warning(f"El atributo {attr} no existe en el DataFrame")
    
#     # Completar la consulta con la ventana
#     sql += " FROM df WINDOW w AS (PARTITION BY numero_de_cliente ORDER BY foto_mes)"
    
#     logger.debug(f"Consulta SQL: {sql}")
    
#     # Ejecutar la consulta SQL
#     df_result = duckdb.query(sql).df()
    
#     logger.info(f"Feature engineering completado. DataFrame resultante con {df_result.shape[1]} columnas")
#     return df_result


# # ---------------------------------------> ranking negativo

# def feature_engineering_rank_neg(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
#     """
#     Genera rankings normalizados por signo para los atributos especificados usando DuckDB.
#     Los negativos van de -1 a 0, los positivos de 0 a 1, y los ceros permanecen en 0.
#     Sobrescribe las columnas originales con los rankings.
#     """
#     if not columnas:
#         logger.warning("No se especificaron atributos para generar rankings")
#         raise ValueError("La lista de columnas no puede estar vacía")
    
#     columnas_validas = [col for col in columnas if col in df.columns]
#     if not columnas_validas:
#         logger.warning("Ninguna de las columnas especificadas existe en el DataFrame")
#         raise ValueError("No hay columnas válidas para rankear")
    
#     logger.info(f"Realizando feature engineering RANK para {len(columnas_validas)} columnas: {columnas_validas}")
    
#     con = duckdb.connect(database=":memory:")
#     con.register("df_temp", df)
    
#     # Mantener orden original de columnas
#     rank_expressions = []
#     for col in df.columns:
#         if col in columnas_validas:
#             rank_expressions.append(f"""
#                 CASE
#                     WHEN {col} < 0 THEN -PERCENT_RANK() OVER (
#                         PARTITION BY foto_mes, SIGN({col}) 
#                         ORDER BY {col}
#                     )
#                     WHEN {col} > 0 THEN PERCENT_RANK() OVER (
#                         PARTITION BY foto_mes, SIGN({col}) 
#                         ORDER BY {col}
#                     )
#                     ELSE 0.0
#                 END AS {col}
#             """)
#         else:
#             rank_expressions.append(col)
    
#     sql = f"""
#     SELECT {', '.join(rank_expressions)}
#     FROM df_temp
#     """
    
#     try:
#         resultado = con.execute(sql).df()
#         logger.info(f"Feature engineering completado. Shape: {resultado.shape}")
#         return resultado
#     finally:
#         con.close()




# #------------------------------------------> lag
# def feature_engineering_lag(df: pd.DataFrame, columnas: list[str], cant_lag: int=1) -> pd.DataFrame:
#     """
#     Genera variables de lag para los atributos especificados utilizando SQL.
  
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         DataFrame con los datos
#     columnas : list
#         Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
#     cant_lag : int, default=1
#         Cantidad de lags a generar para cada atributo
  
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame con las variables de lag agregadas
#     """

#     logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

#     if columnas is None or len(columnas) == 0:
#         logger.warning("No se especificaron atributos para generar lags")
#         return df
  
#     # Construir la consulta SQL
#     sql = "SELECT *"
  
#     # Agregar los lags para los atributos especificados
#     for attr in columnas:
#         if attr in df.columns:
#             for i in range(1, cant_lag + 1):
#                 sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
#         else:
#             logger.warning(f"El atributo {attr} no existe en el DataFrame")
  
#     # Completar la consulta
#     sql += " FROM df"

#     logger.debug(f"Consulta SQL: {sql}")

#     # Ejecutar la consulta SQL
#     con = duckdb.connect(database=":memory:")
#     con.register("df", df)
#     df = con.execute(sql).df()
#     con.close()

#     print(df.head())
  
#     logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

#     return df


#-----------------------------------------------------------  optim bayesiana con cv

# def objetivo_ganancia_cv(trial, df) -> float: 
#     """
#     Parameters:
#     trial: trial de optuna
#     df: dataframe con datos

  
#     Description:
#     Función objetivo que maximiza ganancia con k-fold.
#     Utiliza configuración YAML para períodos y semilla.
#     Define parametros para el modelo LightGBM
#     Preparar dataset para entrenamiento con kfold y un porcentaje de la clase CONTINÚA
#     Entrena modelo con función de ganancia personalizada
#     Predecir y calcular ganancia
#     Guardar cada iteración en JSON
  
#     Returns:
#     float: ganancia total
#     """
#     # Hiperparámetros a optimizar
#     params = {
#         'objective': 'binary',
#         'metric': 'None',  
#         'learning_rate': trial.suggest_float('learning_rate', PARAMETROS_LGBM['learning_rate'][0], PARAMETROS_LGBM['learning_rate'][1]),
#         'num_leaves': trial.suggest_int('num_leaves', PARAMETROS_LGBM['num_leaves'][0], PARAMETROS_LGBM['num_leaves'][1]),
#         'max_depth': trial.suggest_int('max_depth', PARAMETROS_LGBM['max_depth'][0], PARAMETROS_LGBM['max_depth'][1]),
#         'min_child_samples': trial.suggest_int('min_child_samples', PARAMETROS_LGBM['min_child_samples'][0], PARAMETROS_LGBM['min_child_samples'][1]),
#         'subsample': trial.suggest_float('subsample', PARAMETROS_LGBM['subsample'][0], PARAMETROS_LGBM['subsample'][1]),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', PARAMETROS_LGBM['colsample_bytree'][0], PARAMETROS_LGBM['colsample_bytree'][1]),
#         'max_bin': trial.suggest_int('max_bin', PARAMETROS_LGBM['max_bin'][0], PARAMETROS_LGBM['max_bin'][1]),
#         'min_split_gain': trial.suggest_float('min_split_gain', PARAMETROS_LGBM['min_split_gain'][0], PARAMETROS_LGBM['min_split_gain'][1]),
#         'verbosity': -1,
#         'random_state': SEMILLAS[0],
#         'zero_as_missing': trial.suggest_categorical('zero_as_missing', [True, False]) 
#         }
    
#     num_boost_round = trial.suggest_int('num_boost_round', PARAMETROS_LGBM['num_boost_round'][0], PARAMETROS_LGBM['num_boost_round'][1])
#     undersampling_ratio = PARAMETROS_LGBM['undersampling']


#     # Preparar datos de entrenamiento (TRAIN + VALIDACION)
#     if isinstance(GENERAL_TRAIN, list):
#         train_data = df[df['foto_mes'].isin(GENERAL_TRAIN)]
#     else: train_data = df[df['foto_mes'] == GENERAL_TRAIN]

#     logger.info(
#     f"Dataset GENERAL TRAIN - Antes del subsampleo de la clase CONTINUA: "
#     f"Clase 1: {len(train_data[train_data['clase_ternaria'] == 1])}, "
#     f"Clase 0: {len(train_data[train_data['clase_ternaria'] == 0])}"
#     )

#     # SUBMUESTREO
#     clase_1 = train_data[train_data['clase_ternaria'] == 1]
#     clase_0 = train_data[train_data['clase_ternaria'] == 0]    
    
#     semilla = SEMILLAS[0] if isinstance(SEMILLAS, list) else SEMILLAS
#     clase_0_sample = clase_0.sample(frac=undersampling_ratio, random_state=semilla)
#     train_data = pd.concat([clase_1, clase_0_sample], axis=0).sample(frac=1, random_state=semilla)
    

#     X_train = train_data.drop(columns = ['clase_ternaria'])
#     y_train = train_data['clase_ternaria']
    

#     lgb_train = lgb.Dataset(X_train, label=y_train)
    

#     cv_results = lgb.cv(params,
#                     num_boost_round=num_boost_round,
#                     nfold=5,
#                     stratified=True,                 
#                     train_set=lgb_train,
#                     shuffle = True,  
#                     seed = SEMILLAS[0],
#                     feval=ganancia_threshold,   #METRIC SE LA DECLARA VACÍA Y EN SU LUGAR SE USA FEVAL
#                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)])
    
#     # Predecir probabilidades y binarizar
#     ganancias_cv = cv_results['valid ganancia-mean']
#     ganancia_maxima = np.max(ganancias_cv)
#     best_iter = np.argmax(ganancias_cv)+1

#     #Guardar el nro original de árboles y el optimizado:
#     num_boost_round_original = trial.params['num_boost_round']
#     trial.set_user_attr('num_boost_round_original', num_boost_round_original)
#     trial.set_user_attr('best_iteration', int(best_iter)) 
#     trial.params['num_boost_round'] = int(best_iter)


    
#     logger.debug(f"Trial {trial.number}: Ganancia = {ganancia_maxima:,.0f}")
#     logger.debug(f"Trial {trial.number}: Mejor iteracion = {best_iter:,.0f}")
    
#     guardar_iteracion_cv(trial, ganancia_maxima, ganancias_cv, archivo_base=None)
    
#     return ganancia_maxima


#---------------------------------------------------------------> Parametrización OPTUNA + aplicación de OB

# def optimizar_cv(df, n_trials=int, study_name: str = None ) -> optuna.Study:
#     """
#     Args:
#         df: DataFrame con datos
#         n_trials: Número de trials a ejecutar
#         study_name: Nombre del estudio (si es None, usa el de config.yaml)
  
#     Description:
#        Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
#        Guarda cada iteración en un archivo JSON separado. 
#        Pasos:
#         1. Crear estudio de Optuna
#         2. Ejecutar optimización
#         3. Retornar estudio

#     Returns:
#         optuna.Study: Estudio de Optuna con resultados
#     """

#     study_name = STUDY_NAME


#     logger.info(f"Iniciando optimización con CV {n_trials} trials")
#     logger.info(f"Configuración: TRAIN para CV={GENERAL_TRAIN}, SEMILLA={SEMILLAS[0]}")
  
#         # Crear estudio de Optuna
#     study = optuna.create_study(
#         study_name=study_name,
#         direction="maximize",
#         sampler = optuna.samplers.TPESampler(seed= SEMILLAS[0]), 
#         storage="sqlite:///optuna_studies.db",
#         load_if_exists=True
#     )

#     # Aquí iría tu función objetivo y la optimización
#     study.optimize(lambda trial: objetivo_ganancia_cv(trial, df), n_trials=n_trials)

#     # Resultados
#     logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
#     logger.info(f"Optimizacion CV completada. Mejor ganancia promedio: {study.best_value:,.0f}")
#     logger.info(f"Mejores parámetros: {study.best_params}")
#     logger.info(f"Total trials: {len(study.trials)}")

#     return study
    








def feature_engineering_rank(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera variables de ranking para los atributos especificados utilizando SQL.
    Sobrescribe las columnas originales con el ranking calculado.
  
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list[str]
        Lista de atributos para los cuales generar rankings.
      
    Returns
    -------
    pd.DataFrame
        DataFrame con las variables originales reemplazadas por sus rankings.
    """

    if not columnas:
        logger.warning("No se especificaron atributos para generar rankings")
        raise ValueError("La lista de columnas no puede estar vacía")

    # Filtrar columnas válidas
    columnas_validas = [col for col in columnas if col in df.columns]
    if not columnas_validas:
        logger.warning("Ninguna de las columnas especificadas existe en el DataFrame")
        raise ValueError("No hay columnas válidas para rankear")

    logger.info(f"Realizando feature engineering RANK para {len(columnas_validas)} columnas: {columnas_validas}")

    con = duckdb.connect(database=":memory:")
    con.register("df_temp", df)

    logger.info("testeo de tiempo")

    # Columnas que NO se rankean, se mantienen igual
    columnas_no_rank = [col for col in df.columns if col not in columnas_validas]

    # Columnas que sí se rankean, reemplazando su contenido
    rank_expressions = [
        f"PERCENT_RANK() OVER (PARTITION BY foto_mes ORDER BY {col}) AS {col}"
        for col in columnas_validas
    ]

    # Construir el SELECT final
    sql = f"""
    SELECT
        {', '.join(columnas_no_rank + rank_expressions)}
    FROM df_temp
    """

    try:
        resultado = con.execute(sql).df()
        logger.info(f"Feature engineering completado. DataFrame resultante con {resultado.shape[1]} columnas")
        return resultado
    finally:
        con.close()


        #-----------------------------------------------------> Rank positivo

def feature_engineering_rank_pos(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera rankings normalizados por signo para los atributos especificados usando DuckDB.
    Los ranking son positivos, hay uno para el rango valores <0 y otro para el rango valores >0. los 0 permanecen en 0.
    Sobrescribe las columnas originales con los rankings.
    """
    if not columnas:
        logger.warning("No se especificaron atributos para generar rankings")
        raise ValueError("La lista de columnas no puede estar vacía")
    
    columnas_validas = [col for col in columnas if col in df.columns]
    if not columnas_validas:
        logger.warning("Ninguna de las columnas especificadas existe en el DataFrame")
        raise ValueError("No hay columnas válidas para rankear")
    
    logger.info(f"Realizando feature engineering RANK para {len(columnas_validas)} columnas: {columnas_validas}")
    
    con = duckdb.connect(database=":memory:")
    con.register("df_temp", df)
    
    # Mantener orden original de columnas
    rank_expressions = []
    for col in df.columns:
        if col in columnas_validas:
            rank_expressions.append(f"""
                CASE
                    WHEN {col} < 0 THEN PERCENT_RANK() OVER (
                        PARTITION BY foto_mes, SIGN({col}) 
                        ORDER BY {col}
                    )
                    WHEN {col} > 0 THEN PERCENT_RANK() OVER (
                        PARTITION BY foto_mes, SIGN({col}) 
                        ORDER BY {col}
                    )
                    ELSE 0.0
                END AS {col}
            """)
        else:
            rank_expressions.append(col)
    
    sql = f"""
    SELECT {', '.join(rank_expressions)}
    FROM df_temp
    """
    
    try:
        resultado = con.execute(sql).df()
        logger.info(f"Feature engineering completado. Shape: {resultado.shape}")
        return resultado
    finally:
        con.close()