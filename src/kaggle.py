import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

def simular_kaggle_mes(y_pred: dict,
                        y_test: pd.Series,
                        ganancia_fn,
                        n_splits: int = 50,
                        test_size: float = 0.3,
                        random_state: int = 42,
                        umbral_public: float = 0.3,
                        umbral_private: float = 0.7,
                        plot: bool = True):
    """
    Simula 'public' (30%) / 'private' (70%) leaderboard usando el mes con etiquetas conocidas.
   y_pred: dict {nombre_modelo: array_o_series_probabilidades_sobre_mes}
    y_test: pd.Series con etiquetas verdaderas del mes (para estratificar y calcular ganancia)
    ganancia_fn: función (y_pred_probs, y_true, umbral) -> ganancia (float)
    """
    # indices y comprobaciones mínimas
    idx = np.arange(len(y_test))
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    
    rows = []
    for private_idx, public_idx in sss.split(idx, y_test):
        row = {}
        for name, y_pred in y_pred.items():
            # asegurar indexación por posiciones
            y_pred = np.asarray(y_pred)
            g_public = ganancia_fn(y_pred[public_idx], y_test.iloc[public_idx], umbral_public)
            g_private = ganancia_fn(y_pred[private_idx], y_test.iloc[private_idx], umbral_private)
            row[f"{name}_public"] = g_public
            row[f"{name}_private"] = g_private
        rows.append(row)
    
    df_lb = pd.DataFrame(rows)
    

    
    return df_lb
