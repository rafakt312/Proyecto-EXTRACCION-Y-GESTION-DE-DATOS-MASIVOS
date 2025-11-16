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
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
import json
import os
from datetime import datetime


DATA_PATH = "/opt/spark-data/"
INCIDENT_FILE = "Individual_Incident_2020.csv"


def build_spark():
    spark = (
        SparkSession.builder
        .appName("NIBRS_Cluster_Etapa3_MLlib_RF")
        .master("spark://spark-master:7077")
        .config("spark.executor.memory", "1g")
        .config("spark.executor.cores", "1")
        .config("spark.driver.host", "spark-master")
        .config("spark.deploy.retryAttempts", "10")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    return spark


def derive_label_and_features(df):
    df = (
        df.withColumnRenamed("incident_number", "A_INCIDENT_ID")
          .withColumnRenamed("total_offense", "TOTAL_DELITOS")
          .withColumnRenamed("date_HRF", "INCIDENT_DATE_RAW")
    )

    df = df.withColumn(
        "INCIDENT_DATE",
        to_date(col("INCIDENT_DATE_RAW").cast("string"), "yyyyMMdd")
    )

    df = df.withColumn(
        "TIPO_DIA",
        when(dayofweek(col("INCIDENT_DATE")).isin([6, 7]), lit("FIN_SEMANA")).otherwise(lit("DIA_SEMANA"))
    )

    hour_digits = regexp_extract(col("hour"), r"(\d{2})\d{2}", 1)
    hour_bucket = when(lower(col("hour")).contains("midnight"), lit("00")).otherwise(hour_digits)
    hour_bucket = when(length(hour_bucket) == 0, lit("UNK")).otherwise(hour_bucket)
    df = df.withColumn("HOUR_BUCKET", hour_bucket)

    offense_cols = [
        "violence_offense",
        "theft_offense",
        "drug_offense",
        "sex_offense",
        "kidnapping_trafficking",
        "other_offense",
    ]

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
        target_type = "double" if nf == "property_value" else "int"
        df = df.withColumn(nf, col(nf).cast(target_type))

    df = df.fillna({nf: 0 for nf in numeric_features})

    df = df.withColumn("MONTH", spark_month(col("INCIDENT_DATE").cast("date")).cast("string"))

    df = df.filter((col("TOTAL_DELITOS").isNotNull()) & (col("TOTAL_DELITOS") > 0))

    return df


def build_pipeline_rf():
    cat_inputs = ["state", "TIPO_DIA", "HOUR_BUCKET", "MONTH"]
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_inputs]
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in cat_inputs],
        outputCols=[f"{c}_ohe" for c in cat_inputs],
        dropLast=True,
    )
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
    label_indexer = StringIndexer(inputCol="INCIDENT_TYPE", outputCol="label", handleInvalid="skip")
    clf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50, maxDepth=12, seed=42)
    pipeline = Pipeline(stages=indexers + [encoder, assembler, label_indexer, clf])
    return pipeline


def main():
    spark = build_spark()
    print("--- Etapa 3 (RF): Clasificación supervisada de tipo de incidente ---")
    start_ts = time.time()

    report_dir = DATA_PATH + "reports"
    try:
        os.makedirs(report_dir, exist_ok=True)
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo crear el directorio de reportes {report_dir}: {e}")

    timings = {}
    try:
        t0 = time.time()
        df = spark.read.csv(
            DATA_PATH + INCIDENT_FILE,
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

    print("\nDistribución de etiqueta INCIDENT_TYPE (Top 10):")
    t2 = time.time()
    label_dist_df = df.groupBy("INCIDENT_TYPE").count().orderBy(desc("count"))
    label_top10 = label_dist_df.limit(10).collect()
    timings["label_distribution_seconds"] = time.time() - t2
    for row in label_top10:
        print(f"  {row['INCIDENT_TYPE']}: {row['count']}")

    try:
        dist_path = os.path.join(report_dir, "incidente_type_distribution.csv")
        with open(dist_path, "w", encoding="utf-8") as f:
            f.write("INCIDENT_TYPE,count\n")
            for row in label_top10:
                f.write(f"{row['INCIDENT_TYPE']},{row['count']}\n")
        print(f"Distribución (Top 10) guardada en: {dist_path}")
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo escribir distribución: {e}")

    t3 = time.time()
    pipeline = build_pipeline_rf()
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    train_rows = train.count()
    test_rows = test.count()
    timings["split_seconds"] = time.time() - t3
    print(f"Tamaño train: {train_rows}, test: {test_rows}")

    t4 = time.time()
    model = pipeline.fit(train)
    timings["fit_seconds"] = time.time() - t4

    t5 = time.time()
    preds = model.transform(test)
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    accuracy = acc_eval.evaluate(preds)
    f1 = f1_eval.evaluate(preds)
    timings["evaluate_seconds"] = time.time() - t5
    print(f"\nMétricas de Test (RF): accuracy={accuracy:.4f}, f1={f1:.4f}")

    t6 = time.time()
    save_path = DATA_PATH + "models/incidente_clf_rf"
    try:
        model.write().overwrite().save(save_path)
        timings["save_model_seconds"] = time.time() - t6
        print(f"Modelo RF guardado en: {save_path}")
    except Exception as e:
        timings["save_model_seconds"] = time.time() - t6
        print(f"ADVERTENCIA: No se pudo guardar el modelo RF en {save_path}: {e}")

    # Confusion and per-class
    label_indexer_model = model.stages[-2]
    labels = list(label_indexer_model.labels)
    idx_to_name = {i: name for i, name in enumerate(labels)}
    conf_df = preds.select("label", "prediction").groupBy("label", "prediction").count()
    conf_rows = [(int(r["label"]), int(r["prediction"]), int(r["count"])) for r in conf_df.collect()]
    try:
        conf_path = os.path.join(report_dir, "confusion_rf.csv")
        with open(conf_path, "w", encoding="utf-8") as f:
            f.write("true_label,pred_label,count\n")
            for li, pi, c in conf_rows:
                f.write(f"{idx_to_name.get(li, li)},{idx_to_name.get(pi, pi)},{c}\n")
        print(f"Matriz de confusión RF guardada en: {conf_path}")
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo guardar la matriz de confusión RF: {e}")

    actual_counts = {i: 0 for i in range(len(labels))}
    pred_counts = {i: 0 for i in range(len(labels))}
    tp_counts = {i: 0 for i in range(len(labels))}
    for li, pi, c in conf_rows:
        actual_counts[li] += c
        pred_counts[pi] += c
        if li == pi:
            tp_counts[li] += c
    try:
        pcm_path = os.path.join(report_dir, "per_class_metrics_rf.csv")
        with open(pcm_path, "w", encoding="utf-8") as f:
            f.write("class,support,precision,recall,f1\n")
            for i, name in enumerate(labels):
                tp = tp_counts.get(i, 0)
                act = actual_counts.get(i, 0)
                pre = pred_counts.get(i, 0)
                recall = (tp / act) if act else 0.0
                precision = (tp / pre) if pre else 0.0
                f1c = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
                f.write(f"{name},{act},{precision:.6f},{recall:.6f},{f1c:.6f}\n")
        print(f"Métricas por clase RF guardadas en: {pcm_path}")
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo guardar métricas por clase RF: {e}")

    total_seconds = time.time() - start_ts
    summary = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "data_file": INCIDENT_FILE,
        "records_total": int(total_rows),
        "records_processed": int(processed_rows),
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "algorithm": "RandomForestClassifier",
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
        json_path = os.path.join(report_dir, "etapa3_summary_rf.json")
        txt_path = os.path.join(report_dir, "etapa3_summary_rf.txt")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2, ensure_ascii=False)
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write("Resumen Etapa 3 (RF) - Clasificación MLlib\n")
            tf.write(f"Fecha (UTC): {summary['run_timestamp']}\n")
            tf.write(f"Archivo: {summary['data_file']}\n")
            tf.write(f"Leídos: {summary['records_total']} | Procesados: {summary['records_processed']}\n")
            tf.write(f"Train: {summary['train_rows']} | Test: {summary['test_rows']}\n")
            tf.write(f"Accuracy: {summary['metrics']['accuracy']:.4f} | F1: {summary['metrics']['f1']:.4f}\n")
        print(f"Resumen RF guardado en: {json_path} y {txt_path}")
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo escribir summary RF: {e}")

    spark.stop()
    print("--- Etapa 3 RF completada ---")


if __name__ == "__main__":
    main()

