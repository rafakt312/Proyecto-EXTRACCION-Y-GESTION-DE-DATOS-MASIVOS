📖 README.md: Proyecto Big Data (NIBRS) - Etapa 2
Este proyecto demuestra la capacidad de Apache Spark para la Extracción, Procesamiento y Análisis (E-P-A) de grandes volúmenes de datos (más de 136 millones de registros NIBRS) en un ambiente de cluster de 3 nodos (1 Master, 2 Workers) utilizando Docker.

🛠️ Requisitos Previos
Docker y Docker Compose: Deben estar instalados y ejecutándose (Docker Desktop).

Datos: El dataset NIBRS (Individual_Incident_XXXX.csv) debe estar ubicado en la carpeta local data/.

Archivos del Proyecto: docker-compose.yml y src/etapa2_nibrs_demo.py deben estar en la carpeta raíz.

* Guía de Ejecución Paso a Paso
Sigue estos pasos en la terminal de PowerShell (o Bash) desde la carpeta raíz del proyecto.

Paso 1: Iniciar el Cluster de 3 Nodos
Este comando levanta los tres contenedores (spark-master, spark-worker-1, spark-worker-2).

Bash

docker-compose up -d
Paso 2: Aplicar Permisos y Conectar el Master (CRÍTICO)
Debido a problemas de permisos en la imagen base de Jupyter, los demonios de Spark fallan al iniciar. Estos comandos corrigen los permisos como root y fuerzan el inicio del Master.

Corregir Permisos del Master: Crea la carpeta de logs y da permisos de escritura.

Bash

docker exec -u root spark-master chmod -R 777 /usr/local/spark/
Iniciar el Master Daemon:

Bash

docker exec spark-master /usr/local/spark/sbin/start-master.sh
Paso 3: Conectar y Activar los 2 Workers
Ahora que el Master está estable, activamos y conectamos los dos Workers, asegurándonos de que ambos se registren en http://localhost:8080.

Corregir Permisos del Worker 1:

Bash

docker exec -u root spark-worker-1 chmod -R 777 /usr/local/spark/
Conectar Worker 1 al Master:

Bash

docker exec spark-worker-1 /usr/local/spark/sbin/start-worker.sh spark://spark-master:7077
Corregir Permisos del Worker 2:

Bash

docker exec -u root spark-worker-2 chmod -R 777 /usr/local/spark/
Conectar Worker 2 al Master:

Bash

docker exec spark-worker-2 /usr/local/spark/sbin/start-worker.sh spark://spark-master:7077
Verificación:  http://localhost:8080. 


Paso 4: Ejecutar el Script de Análisis Distribuido
Este comando ejecuta el script de PySpark, el cual leerá los  millones de registros y forzará la agregación distribuida (GROUP BY y SUM) en los 2 Workers.

Bash

docker exec -it spark-master /usr/local/spark/bin/spark-submit /opt/spark-code/etapa2_nibrs_demo.py
⏱️ NOTA: La ejecución de este script puede tomar varios minutos debido al volumen de datos . La aplicación se mostrará como RUNNING en la Spark UI.

🧹 Limpieza
Para detener el cluster y liberar los puertos:

Bash
---
# Etapa 3

docker-compose down

Reinicio limpio del cluster
Si deseas reiniciar desde cero antes de correr Etapa 3:

Bash

docker-compose down
docker-compose up -d
docker exec -u root spark-master chmod -R 777 /usr/local/spark/
docker exec spark-master /usr/local/spark/sbin/start-master.sh
docker exec -u root spark-worker-1 chmod -R 777 /usr/local/spark/
docker exec spark-worker-1 /usr/local/spark/sbin/start-worker.sh spark://spark-master:7077
docker exec -u root spark-worker-2 chmod -R 777 /usr/local/spark/
docker exec spark-worker-2 /usr/local/spark/sbin/start-worker.sh spark://spark-master:7077

Paso 5 (opcional): Ejecutar Etapa 3 - Clasificación (MLlib)
Ejecuta el script de clasificación supervisada que entrena un modelo multinomial (Logistic Regression) y guarda el pipeline en /opt/spark-data/models.

Bash

docker exec -it spark-master /usr/local/spark/bin/spark-submit /opt/spark-code/etapa3_mllib_clasificacion.py

Para entrenar con RandomForestClassifier (RF) y generar sus reportes:

Bash

docker exec -it spark-master /usr/local/spark/bin/spark-submit /opt/spark-code/etapa3_mllib_clasificacion_rf.py

Salidas y reportes (para presentación)
El script genera archivos en la carpeta montada de datos del host `./data/reports`:
- ./data/reports/etapa3_summary.json: resumen estructurado (métricas, tiempos, tamaños de datos)
- ./data/reports/etapa3_summary.txt: resumen legible para pegar en informes
- ./data/reports/incidente_type_distribution.csv: Top 10 de etiquetas con sus cuentas
El modelo entrenado se guarda en `./data/models/incidente_clf_lr`.

Visualización rápida (HTML)
Para generar un reporte HTML con gráficos a partir de los reportes anteriores:

Bash

docker exec spark-master python /opt/spark-code/etapa3_report_viz.py

Luego abre en tu máquina (host):
- ./data/reports/etapa3_report.html
- Si también ejecutaste RF, se generará comparación en: ./data/reports/etapa3_report_compare.html
