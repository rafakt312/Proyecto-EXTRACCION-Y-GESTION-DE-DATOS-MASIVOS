# Ubicación: src/etapa2_nibrs_demo.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, dayofweek, when, count, sum as spark_sum, isnull

# --- CONFIGURACIÓN DE RUTAS Y ARCHIVOS ---
DATA_PATH = "/opt/spark-data/"

# Patrón para leer TODOS los archivos Individual_Incident_XXXX.csv a la vez
INCIDENT_PATTERN = "Individual_Incident_*.csv" 

# --- 1. INICIALIZACIÓN Y EXTRACCIÓN ---

# Conexión al Master y configuración de red/recursos (No tocar si el cluster ya funciona)
spark = SparkSession.builder \
    .appName("NIBRS_Cluster_Demo_Etapa2") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "1g") \
    .config("spark.executor.cores", "1") \
    .config("spark.driver.host", "spark-master") \
    .config("spark.deploy.retryAttempts", "10") \
    .config("spark.deploy.maxResourceAllocationTime", "120s") \
    .getOrCreate()

print("--- 1. EXTRACCIÓN: Sesión de Spark activa y conectada al cluster ---")

# 2. Carga de los múltiples segmentos NIBRS a la vez (Big Data)
try:
    df_incidentes = spark.read.csv(
        DATA_PATH + INCIDENT_PATTERN,
        header=True, 
        inferSchema=True
    )
    
    # Renombrar columnas clave para estandarizar (Basado en la captura de pantalla)
    df_incidentes = df_incidentes \
        .withColumnRenamed("incident_number", "A_INCIDENT_ID") \
        .withColumnRenamed("total_offense", "TOTAL_DELITOS") \
        .withColumnRenamed("date_HRF", "INCIDENT_DATE_RAW") # La columna de fecha es INCIDENT_DATE_RAW
    
except Exception as e:
    print(f"ERROR DE CARGA: No se pudieron cargar los archivos. Verifica el patrón y la ruta: {e}")
    spark.stop()
    exit()

# Muestra inicial del volumen de datos (Prueba de Big Data)
print(f"Total Registros de Incidentes (Todos los Años): {df_incidentes.count()}")

# --- 2. PROCESAMIENTO: LIMPIEZA Y FEATURE ENGINEERING ---

print("\n--- 2. PROCESAMIENTO: Limpieza y Feature Engineering de Datos ---")

# 3. Limpieza: Filtrado y Conversión (La columna total_offense es la métrica de delito)
df_cleaned = df_incidentes.filter(
    # Filtramos donde el conteo de delitos no es nulo y es mayor que 0
    (col("TOTAL_DELITOS").isNotNull()) & (col("TOTAL_DELITOS") > 0)
)

# 4. Feature Engineering: Creación de la variable "TIPO_DIA"
# La fecha está en formato de string ('10feb2000'). Usamos funciones de fecha para extraer el día de la semana.
df_processed = df_cleaned.withColumn(
    "TIPO_DIA",
    when(dayofweek(col("INCIDENT_DATE_RAW")).isin([6, 7]), "FIN_SEMANA").otherwise("DIA_SEMANA")
)

df_processed.printSchema()

# --- 3. ANÁLISIS: AGREGACIÓN DISTRIBUIDA (PRUEBA DE CLUSTER) ---

print("\n--- 3. ANÁLISIS: Consulta Agregada Distribuida (Prueba de Workers) ---")

# Ejecutar GROUP BY + AGGREGACIÓN (SUMA DE DELITOS)
# Usamos 'state' y 'TIPO_DIA' para la agregación, forzando un Shuffle distribuido.
weekly_crime_analysis = df_processed.groupBy("state", "TIPO_DIA").agg(
    spark_sum("TOTAL_DELITOS").alias("Suma_Total_Delitos")
).sort(col("Suma_Total_Delitos").desc())

print("Top 10 Estados Clasificados por Suma Total de Delitos (Día vs. Fin de Semana):")

# Ejecutar y mostrar el resultado (¡Momento de capturar la Spark UI!)
weekly_crime_analysis.show(10, truncate=False)

# --- FINALIZACIÓN ---
spark.stop()
print("\n--- Demostración de Cluster y Flujo E-P-A Completada. ---")