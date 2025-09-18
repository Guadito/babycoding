import logging
import pandas as pd
import os
import datetime



logger = logging.getLogger(__name__)



def cargar_datos(path: str) -> pd.DataFrame | None:  #Se le pide que retorne un DataFrame o None

    '''
    Carga un CSV desde 'path' y retorna un pandas.DataFrame.
    '''

    logger.info(f"Cargando dataset desde {path}")
    try:
        df = pd.read_csv(path)
        logger.info(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas") 
        return df
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        raise #En caso de falla, sale de la funcion y propaga el error 


