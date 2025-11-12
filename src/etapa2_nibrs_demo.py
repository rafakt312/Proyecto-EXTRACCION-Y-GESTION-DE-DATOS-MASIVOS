from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofweek, when, count, sum as spark_sum, to_date

# --- CONFIGURACIÓN DE RUTAS Y ARCHIVOS ---
DATA_PATH = "/opt/spark-data/"

# 🚨 CORRECCIÓN 1: Limitar la carga solo al archivo de 2020
INCIDENT_PATTERN = "Individual_Incident_2020.csv" 
# Si tu archivo se llama diferente (ej: Individual_Incident_2020.csv.gz), ajústalo aquí.

# --- 1. INICIALIZACIÓN Y EXTRACCIÓN ---

# Conexión al Master y configuración de red/recursos (Sin cambios)
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

# 2. Carga del segmento NIBRS 2020
try:
    df_incidentes = spark.read.csv(
        DATA_PATH + INCIDENT_PATTERN,
        header=True, 
        inferSchema=True # Inferimos el esquema, pero sabemos que la fecha es INT
    )
    
    # Renombrar columnas clave para estandarizar
    df_incidentes = df_incidentes \
        .withColumnRenamed("incident_number", "A_INCIDENT_ID") \
        .withColumnRenamed("total_offense", "TOTAL_DELITOS") \
        .withColumnRenamed("date_HRF", "INCIDENT_DATE_RAW") 
    
except Exception as e:
    print(f"ERROR DE CARGA: No se pudieron cargar los archivos. Verifica el patrón y la ruta: {e}")
    spark.stop()
    exit()

# Muestra inicial del volumen de datos (Prueba de Big Data)
print(f"Total Registros de Incidentes (2020): {df_incidentes.count()}")

# --- 2. PROCESAMIENTO: LIMPIEZA Y FEATURE ENGINEERING ---

print("\n--- 2. PROCESAMIENTO: Limpieza y Feature Engineering de Datos ---")

# 🚨 CORRECCIÓN 2: Convertir la columna de Fecha (INT) a tipo DATE
# El formato de la fecha es YYYYMMDD (ej: 20200210), por lo que usamos 'yyyyMMdd'.
df_cleaned = df_incidentes.withColumn(
    "INCIDENT_DATE",
    to_date(col("INCIDENT_DATE_RAW").cast("string"), "yyyyMMdd")
).filter(
    # Filtrado por la métrica de delito
    (col("TOTAL_DELITOS").isNotNull()) & (col("TOTAL_DELITOS") > 0)
)

# 3. Feature Engineering: Creación de la variable "TIPO_DIA" (Ahora usando el nuevo tipo DATE)
df_processed = df_cleaned.withColumn(
    "TIPO_DIA",
    when(dayofweek(col("INCIDENT_DATE")).isin([6, 7]), "FIN_SEMANA").otherwise("DIA_SEMANA")
)

df_processed.printSchema()

# --- 3. ANÁLISIS: AGREGACIÓN DISTRIBUIDA (PRUEBA DE CLUSTER) ---

print("\n--- 3. ANÁLISIS: Consulta Agregada Distribuida (Prueba de Workers) ---")

# Ejecutar GROUP BY + AGGREGACIÓN (SUMA DE DELITOS)
weekly_crime_analysis = df_processed.groupBy("state", "TIPO_DIA").agg(
    spark_sum("TOTAL_DELITOS").alias("Suma_Total_Delitos")
).sort(col("Suma_Total_Delitos").desc())

print("Top 10 Estados Clasificados por Suma Total de Delitos (Día vs. Fin de Semana):")

# Ejecutar y mostrar el resultado
weekly_crime_analysis.show(10, truncate=False)

# --- FINALIZACIÓN ---
spark.stop()
print("\n--- Demostración de Cluster y Flujo E-P-A Completada. ---")