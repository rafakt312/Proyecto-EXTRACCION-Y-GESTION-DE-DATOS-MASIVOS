from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    to_date,
    dayofweek,
    month as spark_month,
    when,
    lower,
    regexp_extract,
    length,
    lit,
    desc,
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
import json
import os
from datetime import datetime
import argparse



DATA_PATH = "/opt/spark-data/"
INCIDENT_FILE = "Individual_Incident_2020.csv"


def build_spark():
    spark = (
        SparkSession.builder
        .appName("NIBRS_Cluster_Etapa3_MLlib")
        .master("spark://spark-master:7077")
        .config("spark.executor.memory", "1g")
        .config("spark.executor.cores", "1")
        .config("spark.driver.host", "spark-master")
        .config("spark.deploy.retryAttempts", "10")
        .config("spark.deploy.maxResourceAllocationTime", "120s")
        # Ajuste moderado de particiones de shuffle para acelerar sin cambiar recursos
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    return spark


def derive_label_and_features(df):
    # Standardize key columns used in Etapa 2
    df = (
        df.withColumnRenamed("incident_number", "A_INCIDENT_ID")
          .withColumnRenamed("total_offense", "TOTAL_DELITOS")
          .withColumnRenamed("date_HRF", "INCIDENT_DATE_RAW")
    )

    # Date to proper DATE and day type (weekday/weekend)
    df = df.withColumn(
        "INCIDENT_DATE",
        to_date(col("INCIDENT_DATE_RAW").cast("string"), "yyyyMMdd")
    )

    df = df.withColumn(
        "TIPO_DIA",
        when(dayofweek(col("INCIDENT_DATE")).isin([6, 7]), lit("FIN_SEMANA")).otherwise(lit("DIA_SEMANA"))
    )

    # Hour bucket from textual hour field
    # Examples: "On or between 1900 and 1959", "On or between midnight and 0059"
    hour_digits = regexp_extract(col("hour"), r"(\d{2})\d{2}", 1)
    hour_bucket = when(lower(col("hour")).contains("midnight"), lit("00")).otherwise(hour_digits)
    hour_bucket = when(length(hour_bucket) == 0, lit("UNK")).otherwise(hour_bucket)
    df = df.withColumn("HOUR_BUCKET", hour_bucket)

    # Derive INCIDENT_TYPE from offense indicator columns (single-label)
    # If multiple indicators are 1 -> MULTIPLE; if none -> UNKNOWN
    offense_cols = [
        "violence_offense",
        "theft_offense",
        "drug_offense",
        "sex_offense",
        "kidnapping_trafficking",
        "other_offense",
    ]

    # Sum of offense flags to detect MULTIPLE
    sum_expr = None
    for c in offense_cols:
        term = col(c).cast("int")
        sum_expr = term if sum_expr is None else (sum_expr + term)

    df = df.withColumn("OFFENSE_FLAGS_SUM", sum_expr)

    df = df.withColumn(
        "INCIDENT_TYPE",
        when(col("OFFENSE_FLAGS_SUM") > 1, lit("MULTIPLE"))
        .otherwise(
            when(col("violence_offense").cast("int") == 1, lit("VIOLENCE"))
            .when(col("theft_offense").cast("int") == 1, lit("THEFT"))
            .when(col("drug_offense").cast("int") == 1, lit("DRUG"))
            .when(col("sex_offense").cast("int") == 1, lit("SEX"))
            .when(col("kidnapping_trafficking").cast("int") == 1, lit("KIDNAPPING_TRAFFICKING"))
            .when(col("other_offense").cast("int") == 1, lit("OTHER"))
            .otherwise(lit("UNKNOWN"))
        )
    )

    # Core numeric features (cast to numeric and fill nulls later)
    numeric_features = [
        "TOTAL_DELITOS",
        "total_victim",
        "total_offender",
        "gun_involvement",
        "drug_involvement",
        "stolen_motor",
        "property_value",
    ]

    for nf in numeric_features:
        # property_value can be large; cast to double, others to integer (then to double via assembler)
        target_type = "double" if nf == "property_value" else "int"
        df = df.withColumn(nf, col(nf).cast(target_type))

    # Simple imputation for numeric features to avoid nulls in assembler/model
    df = df.fillna({nf: 0 for nf in numeric_features})

    # Month from INCIDENT_DATE
    df = df.withColumn("MONTH", spark_month(col("INCIDENT_DATE").cast("date")).cast("string"))

    # Select columns we need going forward
    df = df.filter((col("TOTAL_DELITOS").isNotNull()) & (col("TOTAL_DELITOS") > 0))

    return df


def build_pipeline(algo: str = "lr", lr_max_iter: int = 50, rf_num_trees: int = 50, rf_max_depth: int = 12):
    # Index categorical features
    cat_inputs = [
        "state",
        "TIPO_DIA",
        "HOUR_BUCKET",
        "MONTH",
    ]

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_inputs
    ]

    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in cat_inputs],
        outputCols=[f"{c}_ohe" for c in cat_inputs],
        dropLast=True,
    )

    # Numeric features
    num_inputs = [
        "TOTAL_DELITOS",
        "total_victim",
        "total_offender",
        "gun_involvement",
        "drug_involvement",
        "stolen_motor",
        "property_value",
    ]

    assembler_inputs = [f"{c}_ohe" for c in cat_inputs] + num_inputs
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="keep")

    # Label indexer
    label_indexer = StringIndexer(inputCol="INCIDENT_TYPE", outputCol="label", handleInvalid="skip")

    # Classifier selection
    if algo == "lr":
        clf = LogisticRegression(featuresCol="features", labelCol="label", maxIter=lr_max_iter, family="multinomial")
    elif algo == "rf":
        clf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=rf_num_trees, maxDepth=rf_max_depth, seed=42)
    else:
        raise ValueError(f"Algoritmo no soportado: {algo}")

    pipeline = Pipeline(stages=indexers + [encoder, assembler, label_indexer, clf])
    return pipeline


def evaluate_and_reports(preds, model, algo, report_dir, timings, total_rows, processed_rows, train_rows, test_rows):
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    accuracy = acc_eval.evaluate(preds)
    f1 = f1_eval.evaluate(preds)

    print(f"\n[{algo}] Métricas de Test: accuracy={accuracy:.4f}, f1={f1:.4f}")

    # Extract label mapping from label indexer
    label_indexer_model = model.stages[-2]
    labels = list(label_indexer_model.labels)
    idx_to_name = {i: name for i, name in enumerate(labels)}

    # Confusion matrix counts
    conf_df = preds.select("label", "prediction").groupBy("label", "prediction").count()
    conf_rows = [(int(r["label"]), int(r["prediction"]), int(r["count"])) for r in conf_df.collect()]

    # Per-class metrics
    actual_counts = {i: 0 for i in range(len(labels))}
    pred_counts = {i: 0 for i in range(len(labels))}
    tp_counts = {i: 0 for i in range(len(labels))}
    for li, pi, c in conf_rows:
        actual_counts[li] = actual_counts.get(li, 0) + c
        pred_counts[pi] = pred_counts.get(pi, 0) + c
        if li == pi:
            tp_counts[li] = tp_counts.get(li, 0) + c

    per_class = []
    for i in range(len(labels)):
        tp = tp_counts.get(i, 0)
        act = actual_counts.get(i, 0)
        pre = pred_counts.get(i, 0)
        recall = (tp / act) if act else 0.0
        precision = (tp / pre) if pre else 0.0
        f1c = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class.append({
            "class": idx_to_name.get(i, str(i)),
            "support": act,
            "precision": precision,
            "recall": recall,
            "f1": f1c,
        })

    # Save confusion and per-class metrics
    try:
        conf_path = os.path.join(report_dir, f"confusion_{algo}.csv")
        with open(conf_path, "w", encoding="utf-8") as f:
            f.write("true_label,pred_label,count\n")
            for li, pi, c in conf_rows:
                f.write(f"{idx_to_name.get(li, li)},{idx_to_name.get(pi, pi)},{c}\n")
        print(f"[{algo}] Matriz de confusión guardada en: {conf_path}")
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo guardar la matriz de confusión [{algo}]: {e}")

    try:
        pcm_path = os.path.join(report_dir, f"per_class_metrics_{algo}.csv")
        with open(pcm_path, "w", encoding="utf-8") as f:
            f.write("class,support,precision,recall,f1\n")
            for row in per_class:
                f.write(f"{row['class']},{row['support']},{row['precision']:.6f},{row['recall']:.6f},{row['f1']:.6f}\n")
        print(f"[{algo}] Métricas por clase guardadas en: {pcm_path}")
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo guardar métricas por clase [{algo}]: {e}")

    # Summary per algorithm
    summary = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "data_file": INCIDENT_FILE,
        "records_total": int(total_rows),
        "records_processed": int(processed_rows),
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "algorithm": "LogisticRegression(multinomial)" if algo == "lr" else "RandomForestClassifier",
        "model_path": DATA_PATH + f"models/incidente_clf_{algo}",
        "metrics": {"accuracy": float(accuracy), "f1": float(f1)},
        "timings_seconds": timings,
        "features": {
            "categorical": ["state", "TIPO_DIA", "HOUR_BUCKET", "MONTH"],
            "numeric": [
                "TOTAL_DELITOS",
                "total_victim",
                "total_offender",
                "gun_involvement",
                "drug_involvement",
                "stolen_motor",
                "property_value",
            ],
        },
        "total_runtime_seconds": sum(timings.values()) if timings else None,
    }

    try:
        json_path = os.path.join(report_dir, f"etapa3_summary_{algo}.json")
        txt_path = os.path.join(report_dir, f"etapa3_summary_{algo}.txt")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2, ensure_ascii=False)
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write("Resumen Etapa 3 - Clasificación MLlib\n")
            tf.write(f"Algoritmo: {summary['algorithm']}\n")
            tf.write(f"Archivo: {summary['data_file']}\n")
            tf.write(f"Leídos: {summary['records_total']} | Procesados: {summary['records_processed']}\n")
            tf.write(f"Train: {summary['train_rows']} | Test: {summary['test_rows']}\n")
            tf.write(f"Accuracy: {summary['metrics']['accuracy']:.4f} | F1: {summary['metrics']['f1']:.4f}\n")
        print(f"[{algo}] Resúmenes guardados en: {json_path} y {txt_path}")
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo escribir summary [{algo}]: {e}")


def main():
    spark = build_spark()
    print("--- Etapa 3: Clasificación supervisada de tipo de incidente (MLlib) ---")
    start_ts = time.time()

    # Carpeta de reportes/salidas
    report_dir = DATA_PATH + "reports"
    try:
        os.makedirs(report_dir, exist_ok=True)
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo crear el directorio de reportes {report_dir}: {e}")

    timings = {}
    try:
        t0 = time.time()
        df = spark.read.csv(
            DATA_PATH + "Individual_Incident_2020.csv",
            header=True,
            inferSchema=True,
        )
        total_rows = df.count()
        timings["load_read_seconds"] = time.time() - t0
        print(f"Registros totales leídos: {total_rows}")
    except Exception as e:
        print(f"ERROR: No se pudo leer el CSV: {e}")
        spark.stop()
        return

    t1 = time.time()
    df = derive_label_and_features(df)
    processed_rows = df.count()
    timings["feature_engineering_seconds"] = time.time() - t1

    # Distribución de etiqueta (Top 10)
    print("\nDistribución de etiqueta INCIDENT_TYPE (Top 10):")
    t2 = time.time()
    label_dist_df = df.groupBy("INCIDENT_TYPE").count().orderBy(desc("count"))
    label_top10 = label_dist_df.limit(10).collect()
    timings["label_distribution_seconds"] = time.time() - t2
    for row in label_top10:
        print(f"  {row['INCIDENT_TYPE']}: {row['count']}")

    # Guardar distribución a CSV
    try:
        dist_path = os.path.join(report_dir, "incidente_type_distribution.csv")
        with open(dist_path, "w", encoding="utf-8") as f:
            f.write("INCIDENT_TYPE,count\n")
            for row in label_top10:
                f.write(f"{row['INCIDENT_TYPE']},{row['count']}\n")
        print(f"Distribución (Top 10) guardada en: {dist_path}")
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo escribir distribución: {e}")

    # Distribución por estado para tipos clave
    try:
        theft_rows = (
            df.filter(col("INCIDENT_TYPE") == "THEFT")
              .groupBy("state")
              .count()
              .orderBy(desc("count"))
              .collect()
        )
        theft_path = os.path.join(report_dir, "theft_by_state.csv")
        with open(theft_path, "w", encoding="utf-8") as f:
            f.write("state,count\n")
            for r in theft_rows:
                f.write(f"{r['state']},{r['count']}\n")
        print(f"Distribución THEFT por estado guardada en: {theft_path}")
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo escribir distribución THEFT por estado: {e}")

    try:
        target_labels = ["THEFT", "VIOLENCE", "SEX", "DRUG", "OTHER", "MULTIPLE", "KIDNAPPING_TRAFFICKING"]
        pivot_df = (
            df.filter(col("INCIDENT_TYPE").isin(target_labels))
              .groupBy("state")
              .pivot("INCIDENT_TYPE", target_labels)
              .count()
              .fillna(0)
        )
        state_rows = pivot_df.collect()
        state_path = os.path.join(report_dir, "incident_types_by_state.csv")
        with open(state_path, "w", encoding="utf-8") as f:
            f.write("state," + ",".join(target_labels) + "\n")
            for r in state_rows:
                rd = r.asDict()
                f.write(
                    ",".join(
                        [str(rd.get("state", ""))] + [str(int(rd.get(lbl, 0))) for lbl in target_labels]
                    )
                    + "\n"
                )
        print(f"Distribución por estado (varios tipos) guardada en: {state_path}")
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo escribir distribución por estado: {e}")

    # Pipeline y split
    t3 = time.time()
    pipeline = build_pipeline()
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    train_rows = train.count()
    test_rows = test.count()
    timings["split_seconds"] = time.time() - t3
    print(f"Tamaño train: {train_rows}, test: {test_rows}")

    # Entrenamiento
    t4 = time.time()
    model = pipeline.fit(train)
    timings["fit_seconds"] = time.time() - t4

    # Predicción y evaluación
    t5 = time.time()
    preds = model.transform(test)
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    accuracy = acc_eval.evaluate(preds)
    f1 = f1_eval.evaluate(preds)
    timings["evaluate_seconds"] = time.time() - t5
    # Matriz de confusión y métricas por clase (LR)
    try:
        label_indexer_model = model.stages[-2]
        labels = list(label_indexer_model.labels)
        idx_to_name = {i: name for i, name in enumerate(labels)}
        conf_df = preds.select("label", "prediction").groupBy("label", "prediction").count()
        conf_rows = [(int(r["label"]), int(r["prediction"]), int(r["count"])) for r in conf_df.collect()]
        conf_path_lr = os.path.join(report_dir, "confusion_lr.csv")
        with open(conf_path_lr, "w", encoding="utf-8") as fconf:
            fconf.write("true_label,pred_label,count\n")
            for li, pi, c in conf_rows:
                fconf.write(f"{idx_to_name.get(li, li)},{idx_to_name.get(pi, pi)},{c}\n")
        actual_counts = {i: 0 for i in range(len(labels))}
        pred_counts = {i: 0 for i in range(len(labels))}
        tp_counts = {i: 0 for i in range(len(labels))}
        for li, pi, c in conf_rows:
            actual_counts[li] = actual_counts.get(li, 0) + c
            pred_counts[pi] = pred_counts.get(pi, 0) + c
            if li == pi:
                tp_counts[li] = tp_counts.get(li, 0) + c
        pcm_path_lr = os.path.join(report_dir, "per_class_metrics_lr.csv")
        with open(pcm_path_lr, "w", encoding="utf-8") as fpcm:
            fpcm.write("class,support,precision,recall,f1\n")
            for i, name in enumerate(labels):
                tp = tp_counts.get(i, 0)
                act = actual_counts.get(i, 0)
                pre = pred_counts.get(i, 0)
                recall_c = (tp / act) if act else 0.0
                precision_c = (tp / pre) if pre else 0.0
                f1_c = (2 * precision_c * recall_c / (precision_c + recall_c)) if (precision_c + recall_c) else 0.0
                fpcm.write(f"{name},{act},{precision_c:.6f},{recall_c:.6f},{f1_c:.6f}\n")
        print(f"Métricas por clase (LR) guardadas en: {pcm_path_lr}")
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo generar/reportar matriz de confusión o métricas por clase (LR): {e}")
    print(f"\nMétricas de Test: accuracy={accuracy:.4f}, f1={f1:.4f}")

    # Guardar modelo
    t6 = time.time()
    save_path = DATA_PATH + "models/incidente_clf_lr"
    try:
        model.write().overwrite().save(save_path)
        timings["save_model_seconds"] = time.time() - t6
        print(f"Modelo guardado en: {save_path}")
    except Exception as e:
        timings["save_model_seconds"] = time.time() - t6
        print(f"ADVERTENCIA: No se pudo guardar el modelo en {save_path}: {e}")

    total_seconds = time.time() - start_ts

    # Resumen ejecutable para presentación
    summary = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "data_file": INCIDENT_FILE,
        "records_total": int(total_rows),
        "records_processed": int(processed_rows),
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "algorithm": "LogisticRegression(multinomial)",
        "model_path": save_path,
        "metrics": {"accuracy": float(accuracy), "f1": float(f1)},
        "timings_seconds": timings,
        "features": {
            "categorical": ["state", "TIPO_DIA", "HOUR_BUCKET", "MONTH"],
            "numeric": [
                "TOTAL_DELITOS",
                "total_victim",
                "total_offender",
                "gun_involvement",
                "drug_involvement",
                "stolen_motor",
                "property_value",
            ],
        },
        "total_runtime_seconds": total_seconds,
    }

    try:
        json_path = os.path.join(report_dir, "etapa3_summary.json")
        txt_path = os.path.join(report_dir, "etapa3_summary.txt")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2, ensure_ascii=False)
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write("Resumen Etapa 3 - Clasificación MLlib\n")
            tf.write(f"Fecha (UTC): {summary['run_timestamp']}\n")
            tf.write(f"Archivo: {summary['data_file']}\n")
            tf.write(f"Total leídos: {summary['records_total']}\n")
            tf.write(f"Procesados: {summary['records_processed']}\n")
            tf.write(f"Train: {summary['train_rows']} | Test: {summary['test_rows']}\n")
            tf.write(f"Accuracy: {summary['metrics']['accuracy']:.4f} | F1: {summary['metrics']['f1']:.4f}\n")
            tf.write("Timings (s):\n")
            for k, v in summary["timings_seconds"].items():
                tf.write(f"  {k}: {v:.2f}\n")
            tf.write(f"Total runtime (s): {summary['total_runtime_seconds']:.2f}\n")
            tf.write(f"Modelo: {summary['model_path']}\n")
        print(f"Resumen guardado en: {json_path} y {txt_path}")
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo escribir el resumen: {e}")

    # También guardar archivos de summary con sufijo _lr para comparación automática
    try:
        json_path_lr = os.path.join(report_dir, "etapa3_summary_lr.json")
        txt_path_lr = os.path.join(report_dir, "etapa3_summary_lr.txt")
        with open(json_path_lr, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2, ensure_ascii=False)
        with open(txt_path_lr, "w", encoding="utf-8") as tf:
            tf.write("Resumen Etapa 3 (LR) - Clasificación MLlib\n")
            tf.write(f"Fecha (UTC): {summary['run_timestamp']}\n")
            tf.write(f"Archivo: {summary['data_file']}\n")
            tf.write(f"Leídos: {summary['records_total']} | Procesados: {summary['records_processed']}\n")
            tf.write(f"Train: {summary['train_rows']} | Test: {summary['test_rows']}\n")
            tf.write(f"Accuracy: {summary['metrics']['accuracy']:.4f} | F1: {summary['metrics']['f1']:.4f}\n")
        print(f"Resumen (LR) guardado en: {json_path_lr} y {txt_path_lr}")
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo escribir el resumen (LR): {e}")

    spark.stop()
    print("--- Etapa 3 completada ---")


if __name__ == "__main__":
    main()
