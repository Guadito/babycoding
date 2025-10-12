

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
[Prueba de semillas para correcto ensamble y de gráfico feature importances competencias_1_6]
7 - Agrego ranking positivo [-1:0, 0:1] y bajo a 2 lags / delta.  5pts
8 - Subo a 4 lag y deltas: empeora, no lo pruebo en kaggle.   competencias_1_7
9 - Vuelvo a 3 lags y cambio a métrica AUC para entrenar. competencias_1_8        -10.000
10 - Pruebo un undersampling de 0.5   competencias_1_9
11 - Vuelvo a undersamplin de 0.2- Agrego wilcoxon competencias_1_10_prueba













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