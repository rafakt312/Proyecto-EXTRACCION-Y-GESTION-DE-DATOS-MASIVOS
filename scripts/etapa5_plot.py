import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "data" / "reports"


def load_stats(workers: int):
    path = REPORTS_DIR / f"etapa5_times_workers{workers}.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def plot_runs(workers: int, stats: dict):
    runs = stats["records"]
    xs = [r["run"] for r in runs]
    ys = [r["runtime_seconds"] for r in runs]

    plt.figure(figsize=(6, 3))
    plt.plot(xs, ys, marker="o", linestyle="-", color="#2b8cbe")
    plt.title(f"Tiempos por corrida - {workers} workers")
    plt.xlabel("Corrida")
    plt.ylabel("Segundos")
    plt.grid(True, alpha=0.3)
    outfile = REPORTS_DIR / f"etapa5_times_workers{workers}.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"Gráfico guardado: {outfile}")


def plot_comparativo(stats_list: list[dict]):
    workers = [s["workers"] for s in stats_list]
    avgs = [s["avg_seconds"] for s in stats_list]

    plt.figure(figsize=(6, 3))
    bars = plt.bar(workers, avgs, color="#74a9cf")
    plt.title("Comparativo promedio por configuración")
    plt.xlabel("Workers")
    plt.ylabel("Segundos (promedio)")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(workers)
    for bar, val in zip(bars, avgs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    outfile = REPORTS_DIR / "etapa5_times_avg.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"Gráfico comparativo guardado: {outfile}")


def main():
    stats_list = []
    for w in [2, 4, 8]:
        try:
            st = load_stats(w)
            plot_runs(w, st)
            stats_list.append(st)
        except FileNotFoundError:
            print(f"Saltando {w} workers: no hay datos")
    if stats_list:
        # Ordenar por número de workers
        stats_list = sorted(stats_list, key=lambda s: s["workers"])
        plot_comparativo(stats_list)


if __name__ == "__main__":
    main()
