

IDEAS:
-probar objetivo_ganancia con cross validation
-subsamplear el dataset para bayesiana

----

RECORDATORIOS:
Quedan estas dos columnas dentro del ranking aunque no son montos: No me parecen importantes
_mantenimiento
ctarjeta_master_descuentos
ccomisiones_mantenimiento


















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