import argparse
import json
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "data" / "reports"
SUMMARY_RF = REPORTS_DIR / "etapa3_summary_rf.json"


def run_cmd(cmd, check=True):
    """Run shell command and stream output."""
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result


def extract_runtime(summary_path: Path) -> Optional[float]:
    """Extrae total_runtime_seconds del summary de RF, o suma timings si no esta."""
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        if "total_runtime_seconds" in data and data["total_runtime_seconds"] is not None:
            return float(data["total_runtime_seconds"])
        if "timings_seconds" in data and isinstance(data["timings_seconds"], dict):
            return float(sum(data["timings_seconds"].values()))
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not parse {summary_path}: {exc}")
    return None


def benchmark_config(workers: int, runs: int) -> List[Dict]:
    records: List[Dict] = []

    # Escalar workers (master se levanta automáticamente)
    run_cmd(f"docker-compose up -d --scale spark-worker={workers}")
    print(f"Esperando a que los {workers} workers se registren...")
    time.sleep(10)

    for i in range(1, runs + 1):
        print(f"\n== Config {workers} workers | Run {i}/{runs} ==")
        start = time.time()
        # Ejecuta workload pesado RF
        run_cmd(
            "docker exec spark-master "
            "/usr/local/spark/bin/spark-submit "
            "--master spark://spark-master:7077 "
            "--deploy-mode client "
            "/opt/spark-code/etapa3_mllib_clasificacion_rf.py"
        )
        wall_seconds = time.time() - start

        # Copia summary para no sobrescribir
        summary_copy = REPORTS_DIR / f"etapa5_run_workers{workers}_iter{i}.json"
        if SUMMARY_RF.exists():
            shutil.copy(SUMMARY_RF, summary_copy)
        runtime = extract_runtime(SUMMARY_RF) or wall_seconds
        records.append(
            {
                "run": i,
                "workers": workers,
                "runtime_seconds": runtime,
                "wall_seconds": wall_seconds,
                "summary_file": str(summary_copy.name if summary_copy.exists() else ""),
            }
        )
        print(f"Run {i} done: runtime_seconds={runtime:.2f}s (wall={wall_seconds:.2f}s)")

    return records


def save_results(workers: int, records: List[Dict]):
    runtimes = [r["runtime_seconds"] for r in records]
    stats = {
        "workers": workers,
        "runs": len(records),
        "min_seconds": min(runtimes),
        "max_seconds": max(runtimes),
        "avg_seconds": statistics.mean(runtimes),
        "records": records,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"etapa5_times_workers{workers}.json"
    csv_path = REPORTS_DIR / f"etapa5_times_workers{workers}.csv"
    json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    csv_lines = ["run,runtime_seconds,wall_seconds,summary_file"]
    for r in records:
        csv_lines.append(
            f"{r['run']},{r['runtime_seconds']:.3f},{r['wall_seconds']:.3f},{r['summary_file']}"
        )
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")
    print(f"Resultados guardados en {json_path} y {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Etapa 5: corre RF 3 veces por configuración y guarda min/max/prom."
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Lista de cantidades de workers a probar (default: 2 4 8)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Cantidad de corridas por configuración (default: 3)",
    )
    args = parser.parse_args()

    all_results = []
    for w in args.workers:
        recs = benchmark_config(w, args.runs)
        save_results(w, recs)
        all_results.extend(recs)

    # Resumen global opcional
    summary_all = REPORTS_DIR / "etapa5_times_all.json"
    summary_all.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Resumen global guardado en {summary_all}")


if __name__ == "__main__":
    main()
