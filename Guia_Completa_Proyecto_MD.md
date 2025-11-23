
# Guía Completa: Kaggle + VM + Spark + Evaluación (2025)

## 1. Verificar la Cuenta de Kaggle
Para usar la API dentro de una VM, tu cuenta debe estar verificada.

1. Inicia sesión en Kaggle.
2. Ve a: **Profile → Settings → Phone Verification**.
3. Verifica tu número telefónico con el código SMS.
4. Sin verificación, la API NO funciona.

---

## 2. Crear la API Key (`kaggle.json`)
1. Ir a: **Profile → Settings → API**.
2. En la sección **Legacy API Credentials**, presionar:  
   **Create Legacy API Key**
3. Se descargará el archivo:

```
kaggle.json
```

---

## 3. Subir `kaggle.json` a la VM
```bash
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
```

Pegar el contenido completo del archivo.

Guardar y salir:
- CTRL + O
- ENTER
- CTRL + X

Dar permisos adecuados:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

---

## 4. Instalar Kaggle CLI en la VM

```bash
sudo apt update
sudo apt install -y python3 python3-pip
pip3 install kaggle --upgrade --user
```

---

## 5. Agregar Kaggle al PATH (PASO CRÍTICO)
### 5.1 Verificar si Kaggle está instalado
```bash
ls ~/.local/bin/kaggle
```

### 5.2 Comprobar si está en el PATH
```bash
which kaggle
```

### 5.3 Si no aparece, agregarlo
```bash
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

Probar:
```bash
which kaggle
kaggle --version
```

---

## 6. Clonar el repositorio del proyecto

```bash
git clone https://github.com/rafakt312/Proyecto-EXTRACCION-Y-GESTION-DE-DATOS-MASIVOS.git
cd Proyecto-EXTRACCION-Y-GESTION-DE-DATOS-MASIVOS
```

---

## 7. Descargar el dataset desde Kaggle

```bash
kaggle datasets download -d javiergamboa00/incidentdataseta --force
```

Esto genera:

```
incidentdataset.zip
```

---

## 8. Mover ZIP a la carpeta data
```bash
mv incidentdataset.zip Proyecto-EXTRACCION-Y-GESTION-DE-DATOS-MASIVOS/data/
```

---

## 9. Extraer el archivo ZIP
```bash
cd Proyecto-EXTRACCION-Y-GESTION-DE-DATOS-MASIVOS/data/
unzip incidentdataset.zip
```

Genera:

```
Individual_Incident_2020.csv
dataset-metadata.json
```

---

## 10. (Opcional) Borrar el ZIP
```bash
rm incidentdataset.zip
```

---

## 11. Dataset listo para Spark

El CSV queda en:

```
Proyecto-EXTRACCION-Y-GESTION-DE-DATOS-MASIVOS/data/Individual_Incident_2020.csv
```

Docker Compose lo monta como:

```
/opt/spark-data
```

---

# Para ejecutar

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
./scripts/start_spark_cluster.sh
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

