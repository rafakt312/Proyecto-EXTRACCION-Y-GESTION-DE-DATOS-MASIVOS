import os
import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for PNG output
import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = "/opt/spark-data/"
REPORT_DIR = os.path.join(DATA_PATH, "reports")
FIG_DIR = os.path.join(REPORT_DIR, "figures")


def ensure_dirs():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def load_summary(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero la Etapa 3.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_label_distribution():
    dist_path = os.path.join(REPORT_DIR, "incidente_type_distribution.csv")
    if not os.path.exists(dist_path):
        raise FileNotFoundError(f"No existe {dist_path}. Ejecuta primero la Etapa 3.")
    df = pd.read_csv(dist_path)
    return df


def plot_label_distribution(df):
    df_sorted = df.sort_values("count", ascending=False)
    plt.figure(figsize=(8, 4))
    plt.bar(df_sorted["INCIDENT_TYPE"], df_sorted["count"], color="#4C78A8")
    plt.title("Distribución de INCIDENT_TYPE (Top)")
    plt.xlabel("INCIDENT_TYPE")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "label_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_timings(timings):
    items = sorted(timings.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    plt.figure(figsize=(8, 4))
    plt.barh(labels, values, color="#F58518")
    plt.title("Tiempos por etapa (s)")
    plt.xlabel("Segundos")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "timings.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def generate_html(summary, label_img, timing_img):
    html_path = os.path.join(REPORT_DIR, "etapa3_report.html")
    acc = summary["metrics"].get("accuracy")
    f1 = summary["metrics"].get("f1")
    runtime_sec = summary.get("total_runtime_seconds", 0)
    runtime_min = runtime_sec / 60.0 if runtime_sec else 0
    now = datetime.utcnow().isoformat() + "Z"

    html = f"""
<!DOCTYPE html>
<html lang=\"es\">
<head>
  <meta charset=\"utf-8\" />
  <title>Reporte Etapa 3 - Clasificación MLlib</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h1 {{ margin-bottom: 0; }}
    .meta {{ color: #555; margin-top: 0; }}
    .kpi {{ display: flex; gap: 24px; margin: 12px 0; }}
    .kpi div {{ background: #f4f4f4; padding: 10px 14px; border-radius: 6px; }}
    img {{ border: 1px solid #ddd; border-radius: 6px; padding: 6px; background: #fff; }}
  </style>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <meta name=\"generated\" content=\"{now}\" />
  <meta name=\"data-file\" content=\"{summary['data_file']}\" />
  <meta name=\"algorithm\" content=\"{summary['algorithm']}\" />
  <meta name=\"records-total\" content=\"{summary['records_total']}\" />
  <meta name=\"records-processed\" content=\"{summary['records_processed']}\" />
  <meta name=\"train-rows\" content=\"{summary['train_rows']}\" />
  <meta name=\"test-rows\" content=\"{summary['test_rows']}\" />
  <meta name=\"accuracy\" content=\"{acc}\" />
  <meta name=\"f1\" content=\"{f1}\" />
  <meta name=\"runtime-sec\" content=\"{runtime_sec}\" />
  <meta name=\"runtime-min\" content=\"{runtime_min:.2f}\" />
  <meta name=\"model-path\" content=\"{summary['model_path']}\" />
  </head>
<body>
  <h1>Reporte Etapa 3 - Clasificación MLlib</h1>
  <p class=\"meta\">Fuente: {summary['data_file']} | Modelo: {summary['algorithm']} | Duración: {runtime_min:.2f} min</p>
  <div class=\"kpi\">
    <div><b>Leídos</b><br/>{summary['records_total']:,}</div>
    <div><b>Procesados</b><br/>{summary['records_processed']:,}</div>
    <div><b>Train</b><br/>{summary['train_rows']:,}</div>
    <div><b>Test</b><br/>{summary['test_rows']:,}</div>
    <div><b>Accuracy</b><br/>{acc:.4f}</div>
    <div><b>F1</b><br/>{f1:.4f}</div>
  </div>
  <h2>Distribución de etiquetas</h2>
  <img src=\"figures/{os.path.basename(label_img)}\" alt=\"Distribución etiquetas\" />
  <h2>Tiempos por etapa</h2>
  <img src=\"figures/{os.path.basename(timing_img)}\" alt=\"Tiempos por etapa\" />
  <h3>Modelo guardado</h3>
  <p>Ruta: {summary['model_path']}</p>
  <p style=\"color:#666\">Generado: {now}</p>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def load_optional_csv(name):
    path = os.path.join(REPORT_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def write_text_summary(summary, timing_img, dist_df, theft_df=None, state_df=None):
    """Escribe un resumen en TXT para la primera ejecución (sin comparativa)."""
    txt_path = os.path.join(REPORT_DIR, "etapa3_report.txt")
    acc = summary["metrics"].get("accuracy")
    f1 = summary["metrics"].get("f1")
    runtime_sec = summary.get("total_runtime_seconds", 0)
    runtime_min = runtime_sec / 60.0 if runtime_sec else 0
    timings = summary.get("timings_seconds", {})
    # Distribución top (usando el CSV cargado)
    dist_rows = dist_df.sort_values("count", ascending=False).to_dict("records")
    dist_lines = [f"  {r['INCIDENT_TYPE']}: {r['count']}" for r in dist_rows]

    theft_lines = []
    if theft_df is not None and not theft_df.empty:
        theft_rows = theft_df.sort_values("count", ascending=False).to_dict("records")
        theft_lines = [f"  {r['state']}: {r['count']}" for r in theft_rows]

    state_lines = []
    if state_df is not None and not state_df.empty:
        # Mostrar top 10 por THEFT y VIOLENCE (si existen las columnas)
        focus = ["THEFT", "VIOLENCE", "SEX", "DRUG"]
        cols = [c for c in focus if c in state_df.columns]
        if cols:
            sdf = state_df[["state"] + cols].copy()
            # ordenar por THEFT si existe, si no por primera col
            order_col = "THEFT" if "THEFT" in cols else cols[0]
            sdf = sdf.sort_values(order_col, ascending=False).head(10)
            for _, row in sdf.iterrows():
                parts = [f"{c}={int(row[c])}" for c in cols]
                state_lines.append(f"  {row['state']}: " + ", ".join(parts))


    lines = [
        "Reporte Etapa 3 - Clasificación MLlib (TXT)",
        f"Fuente: {summary['data_file']}",
        f"Algoritmo: {summary['algorithm']}",
        f"Leídos: {summary['records_total']} | Procesados: {summary['records_processed']}",
        f"Train: {summary['train_rows']} | Test: {summary['test_rows']}",
        f"Accuracy: {acc:.4f} | F1: {f1:.4f}",
        f"Duración: {runtime_min:.2f} min ({runtime_sec:.2f} s)",
        f"Modelo: {summary['model_path']}",
        "",
        "Tiempos por etapa (s):",
    ]
    if timings:
        lines.extend([f"  {k}: {v:.2f}" for k, v in timings.items()])
    else:
        lines.append("  (No se encontraron timings)")

    lines.append("")
    lines.append("Distribución INCIDENT_TYPE (ordenada):")
    if dist_lines:
        lines.extend(dist_lines)
    else:
        lines.append("  (No se encontró distribución)")

    lines.append("")
    lines.append(f"Figura tiempos: {timing_img}")

    if theft_lines:
        lines.append("")
        lines.append("THEFT por estado (ordenado):")
        lines.extend(theft_lines)

    if state_lines:
        lines.append("")
        lines.append("Tipos clave por estado (top 10 ordenado):")
        lines.extend(state_lines)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return txt_path


def generate_compare_html(summary_lr, summary_rf):
    html_path = os.path.join(REPORT_DIR, "etapa3_report_compare.html")
    acc_lr = summary_lr["metrics"].get("accuracy")
    f1_lr = summary_lr["metrics"].get("f1")
    acc_rf = summary_rf["metrics"].get("accuracy")
    f1_rf = summary_rf["metrics"].get("f1")

    # Simple comparison bar chart
    plt.figure(figsize=(6, 4))
    labels = ["Accuracy", "F1"]
    lr_vals = [acc_lr, f1_lr]
    rf_vals = [acc_rf, f1_rf]
    x = range(len(labels))
    width = 0.35
    plt.bar([i - width/2 for i in x], lr_vals, width=0.35, label="LR", color="#4C78A8")
    plt.bar([i + width/2 for i in x], rf_vals, width=0.35, label="RF", color="#F58518")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    compare_img = os.path.join(FIG_DIR, "algos_compare.png")
    plt.savefig(compare_img, dpi=150)
    plt.close()

    # Optional: include per-class tables if present
    def load_pcm(name):
        path = os.path.join(REPORT_DIR, f"per_class_metrics_{name}.csv")
        return pd.read_csv(path) if os.path.exists(path) else None

    pcm_lr = load_pcm("lr")
    pcm_rf = load_pcm("rf")

    # Build basic HTML with a comparison table and optional per-class tables
    html = [
        "<!DOCTYPE html>",
        "<html lang='es'>",
        "<head><meta charset='utf-8'><title>Comparativa LR vs RF</title>",
        "<style>body{font-family:Arial;margin:20px} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:6px 8px}</style>",
        "</head><body>",
        "<h1>Comparativa LR vs RF</h1>",
        f"<p>Fuente: {summary_lr['data_file']}</p>",
        "<h2>Métricas globales</h2>",
        "<table>",
        "<tr><th>Algoritmo</th><th>Accuracy</th><th>F1</th><th>Train</th><th>Test</th></tr>",
        f"<tr><td>LR</td><td>{acc_lr:.4f}</td><td>{f1_lr:.4f}</td><td>{summary_lr['train_rows']}</td><td>{summary_lr['test_rows']}</td></tr>",
        f"<tr><td>RF</td><td>{acc_rf:.4f}</td><td>{f1_rf:.4f}</td><td>{summary_rf['train_rows']}</td><td>{summary_rf['test_rows']}</td></tr>",
        "</table>",
        "<h2>Comparación gráfica</h2>",
        f"<img src='figures/{os.path.basename(compare_img)}' alt='Comparativa' />",
    ]

    if pcm_lr is not None:
        html += ["<h3>Métricas por clase (LR)</h3>", pcm_lr.to_html(index=False)]
    if pcm_rf is not None:
        html += ["<h3>Métricas por clase (RF)</h3>", pcm_rf.to_html(index=False)]

    html += ["</body></html>"]

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return html_path


def main():
    ensure_dirs()
    # Try to load both LR and RF summaries for comparison
    lr_json = None
    rf_json = None
    lr_paths = [
        os.path.join(REPORT_DIR, "etapa3_summary_lr.json"),
        os.path.join(REPORT_DIR, "etapa3_summary.json"),  # fallback (LR legacy)
    ]
    for p in lr_paths:
        if os.path.exists(p):
            lr_json = load_summary(p)
            break
    rf_path = os.path.join(REPORT_DIR, "etapa3_summary_rf.json")
    if os.path.exists(rf_path):
        rf_json = load_summary(rf_path)

    dist_df = load_label_distribution()
    label_img = plot_label_distribution(dist_df)

    # If RF present, build comparison; else single report
    if lr_json is not None and rf_json is not None:
        html_path = generate_compare_html(lr_json, rf_json)
        print(f"Reporte comparativo HTML generado en: {html_path}")
    else:
        summary = lr_json or rf_json
        timing_img = plot_timings(summary.get("timings_seconds", {}))
        html_path = generate_html(summary, label_img, timing_img)
        theft_df = load_optional_csv("theft_by_state.csv")
        state_df = load_optional_csv("incident_types_by_state.csv")
        txt_path = write_text_summary(summary, timing_img, dist_df, theft_df, state_df)
        print(f"Reporte HTML generado en: {html_path}")
        print(f"Reporte TXT generado en: {txt_path}")


if __name__ == "__main__":
    main()
