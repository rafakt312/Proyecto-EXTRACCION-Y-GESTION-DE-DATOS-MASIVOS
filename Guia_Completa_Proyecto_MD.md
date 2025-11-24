
# Guía Completa: Kaggle + VM + Spark + Evaluación (2025)

# Ejecución

## 1. Ubicarse en la raíz del repo
```bash
cd /ruta/al/repo/Proyecto-EXTRACCION-Y-GESTION-DE-DATOS-MASIVOS
```

---

## 2. Verificar dataset
```bash
ls -lh data/Individual_Incident_2020.csv
```

Debe pesar aproximadamente **1.5 GB**.

---

## 3. Arrancar Spark con el script incluido
```bash
sudo bash scripts/start_spark_cluster.sh
```

### ¿Qué hace?
- Ejecuta `docker-compose up -d`
- Aplica permisos en `/usr/local/spark`
- Inicia master y 2 workers conectados a:
```
spark://spark-master:7077
```

---

## 4. Ejecutar experimento (Logistic Regression)
```bash
docker exec spark-master /usr/local/spark/bin/spark-submit /opt/spark-code/etapa3_mllib_clasificacion.py
```

Tiempo estimado: **13 minutos**

---

## 5. Generar reportes
```bash
docker exec spark-master python /opt/spark-code/etapa3_report_viz.py
```

---

## 6. Revisar resultados

### Métricas rápidas
```bash
head -n 30 data/reports/etapa3_summary.txt
```

### Archivos clave en `data/reports/`
- etapa3_summary.json  
- etapa3_summary.txt  
- confusion_lr.csv  
- per_class_metrics_lr.csv  
- incident_type_distribution.csv  
- etapa3_report.html  
- carpeta `figures/` con PNG generados

---

## 7. Limpieza opcional
```bash
docker-compose down
```

