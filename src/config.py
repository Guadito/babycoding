import os
import yaml
import logging


logger = logging.getLogger(__name__)    

#Ruta del archivo de configuracion
PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

try:
    with open(PATH_CONFIG, "r") as file:
        _cfgGeneral = yaml.safe_load(file)    # <-- Lee el YAML
        _cfg = _cfgGeneral["competencia01"]    # <-- Selecciona la sección competencia01


    #Configuración global del proyecto
    STUDY_NAME = _cfgGeneral.get("STUDY_NAME", "competencias")
    DATA_PATH = _cfg.get("DATA_PATH", "../Data/competencia.csv")
    SEMILLAS = _cfg.get("SEMILLAS", [42])
    MES_TRAIN = _cfg.get("MES_TRAIN","202102")
    MES_VAL = _cfg.get("MES_VAL", "202103")
    MES_TEST = _cfg.get("MES_TEST","202104")
    GANANCIA_ACIERTO = _cfg.get("GANANCIA_ACIERTO", None)
    COSTO_ESTIMULO =   _cfg.get("COSTO_ESTIMULO", None)


except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise