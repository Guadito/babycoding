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




# def feature_engineering_rank(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
#     """
#     Genera variables de ranking para los atributos especificados utilizando SQL.
  
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         DataFrame con los datos
#     columnas : list
#         Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
      
#     Returns:
#     --------
#     pd.DataFrame
#         DataFrame con las variables de ranking agregadas
#     """

#     logger.info(f"Realizando feature engineering con para {len(columnas) if columnas else 0} atributos")

#     if columnas is None or len(columnas) == 0:
#         logger.warning("No se especificaron atributos para generar rankings")
#         return df
  
#     sql = "SELECT *, "

#     # Agregar los ranks para las columnas especificadas
#     rank_columns = []
#     for attr in columnas:
#         if attr in df.columns:
#             rank_columns.append(
#                 f"PERCENT_RANK() OVER (PARTITION BY foto_mes ORDER BY {attr}) AS {attr}_rank"
#             )
#         else:
#             logger.warning(f"El atributo {attr} no existe en el DataFrame")

#     # Unir todas las expresiones de rank
#     sql += ", ".join(rank_columns)
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


def feature_engineering_rank(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Genera variables de ranking para los atributos especificados utilizando pandas.
    Cada ranking se calcula por 'foto_mes' y se escala entre 0 y 1 (percentil).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar rankings. Si es None, no se generan rankings.
      
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de ranking agregadas
    """

    logger.info(f"Realizando feature engineering para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar rankings")
        return df.copy()

    df_result = df.copy()

    for col in columnas:
        if col in df.columns:
            rank_col_name = f"{col}_rank"
            df_result[rank_col_name] = df_result.groupby("foto_mes")[col].rank(pct=True)
        else:
            logger.warning(f"El atributo {col} no existe en el DataFrame")

    logger.info(f"Feature engineering completado. DataFrame resultante con {df_result.shape[1]} columnas")
    return df_result







def feature_engineering_lag(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
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


def select_col_montos(df):
    """
    Selecciona columnas de "montos" de un DataFrame según patrones:
      - columnas que empiezan con 'm'
      - columnas que contienen '_m'

    Parámetros:
        df (pd.DataFrame): DataFrame original

    Retorna:
        list: columnas seleccionadas
    """
    pattern_incl = re.compile(r'(^m)|(_m)')  # patrón de inclusión
    
    selected_cols = [c for c in df.columns if pattern_incl.search(c)]
    
    return selected_cols


