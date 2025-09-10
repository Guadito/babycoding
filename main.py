import pandas as pd
import datetime
import os
import sys

def main():
    print("Inicio de ejecución")

    #cargar datos
    try:
        df = pd.read_csv("Data/competencia_01_crudo.csv")
    
    except Exception as e:
        print(f"Error al leer CSV: {e}" ) #la variable e es lo que Python entendió como error
        return 


    print("CSV cargado correctamente")
    print(df.head())

    with open("Log/logs.txt", "a") as file:  #"a": appends --- Abre el log, carga la línea y cierra el archivo
        file.write(f"{datetime.datetime.now()} Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas el \n")

    print(">>>> Ejecución finalizada <<<<")






if __name__ == "__main__":
    main()