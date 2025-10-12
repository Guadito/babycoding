# src/features.py
import pandas as pd
import duckdb
import logging
import re
import sys
print(sys.executable)

import os
import duckdb
import pandas as pd 



logger = logging.getLogger("__name__")



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

#-----------------------------------------------------> Rank positivo batch

def feature_engineering_rank_pos_batch(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera rankings normalizados por signo para las columnas especificadas,
    procesando los datos por 'foto_mes' y columna para reducir el uso de memoria.
    """
    import duckdb

    if not columnas:
        raise ValueError("La lista de columnas no puede estar vacía")

    columnas_validas = [col for col in columnas if col in df.columns]
    if not columnas_validas:
        raise ValueError("No hay columnas válidas para rankear")

    logger.info(f"Generando ranking por batch: {len(columnas_validas)} columnas válidas.")
    con = duckdb.connect(database=":memory:")
    con.execute("SET threads=1;")
    con.execute("SET preserve_insertion_order=false;")
    con.execute("SET memory_limit='2GB';")

    df_result = []

    for mes in sorted(df["foto_mes"].unique()):
        df_mes = df[df["foto_mes"] == mes].copy()
        logger.info(f"Procesando foto_mes={mes} con {len(df_mes)} filas...")

        con.register("df_temp", df_mes)

        for col in columnas_validas:
            sql = f"""
            SELECT
                {col},
                CASE
                    WHEN {col} = 0 THEN 0.0
                    ELSE PERCENT_RANK() OVER (
                        PARTITION BY SIGN({col})
                        ORDER BY {col}
                    )
                END AS rank_col
            FROM df_temp
            """
            df_rank = con.execute(sql).df()
            df_mes[col] = df_rank["rank_col"]

        con.unregister("df_temp")
        df_result.append(df_mes)

    con.close()
    resultado = pd.concat(df_result, ignore_index=True)
    logger.info(f"Feature engineering completado por batch y columna. Shape final: {resultado.shape}")
    return resultado

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



#------------------------------------------> lag


def feature_engineering_lag(df: pd.DataFrame, columnas: list[str], cant_lag: int=1) -> pd.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """

    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df
  
    # Construir la consulta SQL
    sql = "SELECT *"
  
    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")
  
    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()

    print(df.head())
  
    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df



#----------------------------> selecciona variables de montos 

def select_col_montos(df: pd.DataFrame) -> list:
    """
    Selecciona columnas de "montos" de un DataFrame según patrones:
      - columnas que empiezan con 'm'
      - columnas que contienen 'Master_m o Visa_m'
    Siempre excluye: 'numero_de_cliente' y 'foto_mes'
    
    Parámetros:
        df (pd.DataFrame): DataFrame original

    Retorna:
        list: columnas seleccionadas
    """
    # patrón: empieza con m  O contiene _m
    pattern_incl = re.compile(r'(^m|Master_m|Visa_m)')

    # columnas que cumplen el patrón
    selected_cols = [col for col in df.columns if pattern_incl.search(col)]

    # excluir columnas específicas
    excluded = {"numero_de_cliente", "foto_mes"}
    selected_cols = [col for col in selected_cols if col not in excluded]

    return selected_cols

#-------------------------------> Crea LAG y DELTA en batch.

def feature_engineering_lag_delta_batch(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1, batch_size: int = 25) -> pd.DataFrame:
    """
    Genera variables de lag y delta para los atributos especificados utilizando SQL (DuckDB).
    Procesa columnas en batches para evitar problemas de memoria.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos originales.
    columnas : list[str]
        Lista de atributos para los cuales generar lags y deltas.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo.
    batch_size : int, default=25
        Número de columnas a procesar por batch.
    
    Returns
    -------
    pd.DataFrame
        DataFrame con las variables de lag y delta agregadas.
    """
    logger.info(f"Generando {cant_lag} lags y deltas para {len(columnas)} atributos en batches de {batch_size}")
    
    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags/deltas")
        return df
    
    # Validar columnas requeridas
    if 'numero_de_cliente' not in df.columns or 'foto_mes' not in df.columns:
        logger.error("El DataFrame debe contener 'numero_de_cliente' y 'foto_mes'")
        raise ValueError("Columnas requeridas no encontradas")
    
    # Iniciar con el DataFrame original
    df_result = df.copy()
    
    # Procesar columnas en batches
    num_batches = (len(columnas) - 1) // batch_size + 1
    
    for i in range(0, len(columnas), batch_size):
        batch_num = i // batch_size + 1
        batch_cols = columnas[i:i+batch_size]
        
        logger.info(f"Procesando batch {batch_num}/{num_batches}: {len(batch_cols)} columnas")
        
        # Construir la consulta SQL solo para este batch
        sql = "SELECT numero_de_cliente, foto_mes"
        
        for attr in batch_cols:
            if attr in df.columns:
                for j in range(1, cant_lag + 1):
                    sql += f', LAG("{attr}", {j}) OVER w AS "{attr}_lag_{j}"'
                    sql += f', ("{attr}" - LAG("{attr}", {j}) OVER w) AS "{attr}_delta_{j}"'
            else:
                logger.warning(f"El atributo {attr} no existe en el DataFrame")
        
        sql += " FROM df WINDOW w AS (PARTITION BY numero_de_cliente ORDER BY foto_mes)"
        
        logger.debug(f"Ejecutando query para batch {batch_num}")
        
        # Ejecutar la consulta SQL
        df_batch = duckdb.query(sql).df()

        # Convertir a float32 inmediatamente para ahorrar memoria
        float_cols = df_batch.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            df_batch[float_cols] = df_batch[float_cols].astype('float32')
       
        
        # Mergear con el resultado acumulado
        cols_to_merge = [c for c in df_batch.columns if c not in ['numero_de_cliente', 'foto_mes']]
        
        df_result = df_result.merge(
            df_batch[['numero_de_cliente', 'foto_mes'] + cols_to_merge],
            on=['numero_de_cliente', 'foto_mes'],
            how='left'
        )
        
        logger.info(f"Batch {batch_num} completado. Total columnas: {df_result.shape[1]}")
        
        # Limpiar memoria
        del df_batch
        import gc
        gc.collect()
    
    logger.info(f"Feature engineering completado. DataFrame resultante con {df_result.shape[1]} columnas")
    return df_result


# --------------------------> clase pesada 

def asignar_pesos(df: pd.DataFrame) -> pd.DataFrame:
    """
    'BAJA+2': 2.5,
    'BAJA+1': 1.5,
    'CONTINUA': 1.0
    """

    df['clase_pesada'] = df['clase_ternaria'].map({
        'BAJA+2': 2.5,
        'BAJA+1': 1.5,
        'CONTINUA': 1.0
    })
    return df