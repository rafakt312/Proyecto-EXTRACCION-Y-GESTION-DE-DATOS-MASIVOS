Proyecto Big Data (NIBRS) — Etapas 2 y 3

Este proyecto demuestra la capacidad de Apache Spark para la Extracción, Procesamiento y Análisis (E‑P‑A) de datos NIBRS en un clúster de 3 nodos (1 Master, 2 Workers) con Docker.

Requisitos
- Docker Desktop y Docker Compose instalados y corriendo.
- Dataset: copiar `Individual_Incident_2020.csv` en la carpeta `data/` (host).

Estructura Relevante
- `docker-compose.yml`: define master y workers.
- `src/etapa2_nibrs_demo.py`: demo de agregaciones (Etapa 2).
- `src/etapa3_mllib_clasificacion.py`: Logistic Regression (Etapa 3).
- `src/etapa3_mllib_clasificacion_rf.py`: RandomForest (Etapa 3, comparativa).
- `src/etapa3_report_viz.py`: genera HTML con KPIs, gráficos y comparativa.
- `docs/Etapa3_Experimento_EPA.md`: documento del experimento (E‑P‑A).

Arranque Del Clúster
1) Levantar contenedores
- `docker-compose up -d`
2) Iniciar master y workers (permisos + daemons)
- `docker exec -u root spark-master chmod -R 777 /usr/local/spark/`
- `docker exec spark-master /usr/local/spark/sbin/start-master.sh`
- `docker exec -u root spark-worker-1 chmod -R 777 /usr/local/spark/`
- `docker exec spark-worker-1 /usr/local/spark/sbin/start-worker.sh spark://spark-master:7077`
- `docker exec -u root spark-worker-2 chmod -R 777 /usr/local/spark/`
- `docker exec spark-worker-2 /usr/local/spark/sbin/start-worker.sh spark://spark-master:7077`
3) Verificación (UI)
- Abrir `http://localhost:8080` y confirmar 2 Workers conectados.

Etapa 2 
- Ejecutar demo de agregación distribuida:
- `docker exec spark-master /usr/local/spark/bin/spark-submit /opt/spark-code/etapa2_nibrs_demo.py`

Etapa 3 — Clasificación (MLlib)
- Logistic Regression (LR):
- `docker exec spark-master /usr/local/spark/bin/spark-submit /opt/spark-code/etapa3_mllib_clasificacion.py`
- RandomForest (RF):
- `docker exec spark-master /usr/local/spark/bin/spark-submit /opt/spark-code/etapa3_mllib_clasificacion_rf.py`

Salidas y Reportes
- Carpeta `./data/reports`:
  - `etapa3_summary.json` y `etapa3_summary.txt` (LR): métricas, tiempos por fase y tamaños de datos.
  - `etapa3_summary_rf.json` y `etapa3_summary_rf.txt` (RF): métricas y tiempos.
  - `incidente_type_distribution.csv`: distribución de etiquetas (Top 10).
  - `confusion_lr.csv` y `per_class_metrics_lr.csv`: matriz y métricas por clase (LR, si re‑ejecutamos el script actual).
  - `confusion_rf.csv` y `per_class_metrics_rf.csv`: matriz y métricas por clase (RF).

Visualización (HTML)
- Generar HTMLs:
- `docker exec spark-master python /opt/spark-code/etapa3_report_viz.py`
- Abrir en el host:
  - `data/reports/etapa3_report.html`.
  - `data/reports/etapa3_report_compare.html`.
- Imágenes en `data/reports/figures/`:
  - `label_distribution.png`, `timings.png`, `algos_compare.png`.

Reinicio Limpio
- `docker-compose down`
- `docker-compose up -d`
- Repetir inicio de master/workers.

Notas De Rendimiento
- LR (13 min en 2 workers de 1 core/1 GB c/u; CSV ~1.5 GB).
- RF (50–60 min con misma configuración; mejor Accuracy/F1-score).

Etapa 5 - Benchmark de escalamiento
- Clusters: 1 master + N workers (usar --scale spark-worker=N en docker-compose).
- Levantar ejemplo: docker-compose up -d --scale spark-worker=4 (ajusta N=2/4/8).
- Workload pesado: src/etapa3_mllib_clasificacion_rf.py.
- Benchmark: python scripts/etapa5_benchmark.py --workers 2 4 8 --runs 3
  - Salidas por config: data/reports/etapa5_times_workers{N}.json y .csv.
  - Copias por corrida: data/reports/etapa5_run_workers{N}_iter{i}.json.
  - Resumen global: data/reports/etapa5_times_all.json.
- Graficas: python scripts/etapa5_plot.py
  - Genera data/reports/etapa5_times_workers{N}.png y etapa5_times_avg.png.
- UI master: http://localhost:8080 para verificar N workers conectados.
- Nota: en el host actual solo se ejecutaron las configuraciones de 2 y 4 workers (8 no completó ejecuciones estables pese a tener 12 GB asignados a Docker); los CSV/PNG reflejan 2 y 4.

Comandos usados (resumen)
- Levantar cluster escalable: `docker-compose up -d --scale spark-worker=4` (ajusta N=2/4/8).
- Benchmark estándar (3 corridas por config): `python scripts/etapa5_benchmark.py --workers 2 4 8 --runs 3`
- Benchmark reducido (ejemplo, solo 4 workers): `python scripts/etapa5_benchmark.py --workers 4 --runs 3`
- Gráficas desde resultados generados: `python scripts/etapa5_plot.py`
- Bajar cluster: `docker-compose down`

