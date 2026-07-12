"""Aggregate per-question results into a comparison table + charts.

Reads results/<config>.jsonl for each available config and emits:
  - results/comparativa.csv   (metrics x configs)
  - results/comparativa.md    (same, Markdown)
  - results/comparativa.png   (grouped bar chart of headline metrics)

Usage: python -m normarag.eval.report
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from normarag.eval.metrics import abstention_scores

RESULTS_DIR = Path("results")

# Display order and labels A/B/C/D.
CONFIG_ORDER = ["baseline", "rag_naive", "rag_improved", "agentic"]
CONFIG_LABEL = {
    "baseline": "A · baseline",
    "rag_naive": "B · RAG naive",
    "rag_improved": "C · RAG improved",
    "agentic": "D · agentic",
}

# (row key, display label). Order defines the table rows.
METRIC_ROWS = [
    ("citation_validity", "Cita válida (existe en corpus)"),
    ("citation_exact_precision", "Cita exact-match · precision"),
    ("citation_exact_recall", "Cita exact-match · recall"),
    ("citation_exact_f1", "Cita exact-match · F1"),
    ("citation_chapter_f1", "Cita chapter-match · F1"),
    ("abstention_precision", "Abstención · precision"),
    ("abstention_recall", "Abstención · recall"),
    ("abstention_f1", "Abstención · F1"),
    ("faithfulness", "RAGAS faithfulness"),
    ("answer_relevancy", "RAGAS answer_relevancy"),
    ("context_precision", "RAGAS context_precision"),
    ("context_recall", "RAGAS context_recall"),
    ("latency_ms_mean", "Latencia media (ms)"),
    ("cost_usd_mean", "Costo medio por consulta (USD)"),
    ("cost_usd_total", "Costo total corrida (USD)"),
]

# Headline metrics for the bar chart (must be in [0, 1]).
CHART_METRICS = [
    ("citation_validity", "cita válida"),
    ("citation_exact_f1", "exact F1"),
    ("citation_chapter_f1", "chapter F1"),
    ("abstention_f1", "abstención F1"),
    ("faithfulness", "faithfulness"),
    ("context_recall", "context_recall"),
]


def _mean(values: list[float | None]) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


def aggregate(rows: list[dict]) -> dict:
    """Aggregate one config's per-question rows into scalar metrics."""
    if not rows:
        return {}
    agg: dict[str, float | None] = {}
    for key in (
        "citation_validity",
        "citation_exact_precision",
        "citation_exact_recall",
        "citation_exact_f1",
        "citation_chapter_precision",
        "citation_chapter_recall",
        "citation_chapter_f1",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ):
        agg[key] = _mean([r.get(key) for r in rows])

    abst = abstention_scores(r["abstention_outcome"] for r in rows if r.get("abstention_outcome"))
    agg["abstention_precision"] = abst.precision
    agg["abstention_recall"] = abst.recall
    agg["abstention_f1"] = abst.f1

    latencies = [r.get("latency_ms") for r in rows]
    costs = [r.get("cost_usd") or 0.0 for r in rows]
    agg["latency_ms_mean"] = _mean(latencies)
    agg["cost_usd_mean"] = _mean(costs)
    agg["cost_usd_total"] = sum(costs)
    return agg


def load_results(results_dir: Path = RESULTS_DIR) -> dict[str, list[dict]]:
    """Load every results/<label>.jsonl, ordering the canonical A/B/C/D configs
    first and any ablation labels after them (alphabetically)."""
    labels = sorted(p.stem for p in results_dir.glob("*.jsonl"))
    ordered = [c for c in CONFIG_ORDER if c in labels]
    ordered += [c for c in labels if c not in CONFIG_ORDER]

    found: dict[str, list[dict]] = {}
    for label in ordered:
        path = results_dir / f"{label}.jsonl"
        found[label] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    return found


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) < 0.001 and value != 0:
        return f"{value:.2e}"
    return f"{value:.4f}"


def write_csv(aggs: dict[str, dict], path: Path) -> None:
    configs = list(aggs)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", *[CONFIG_LABEL.get(c, c) for c in configs]])
        for key, label in METRIC_ROWS:
            writer.writerow([label, *[_fmt(aggs[c].get(key)) for c in configs]])


def write_markdown(aggs: dict[str, dict], path: Path) -> None:
    configs = list(aggs)
    header = "| Métrica | " + " | ".join(CONFIG_LABEL.get(c, c) for c in configs) + " |"
    sep = "|" + "---|" * (len(configs) + 1)
    lines = ["# Comparativa de configuraciones (NormaRAG)", "", header, sep]
    for key, label in METRIC_ROWS:
        cells = " | ".join(_fmt(aggs[c].get(key)) for c in configs)
        lines.append(f"| {label} | {cells} |")
    lines.append("")
    lines.append(
        "> Métricas de citación y abstención: 100% programáticas. "
        "RAGAS (faithfulness/answer_relevancy/context_*): juez = JUDGE_MODEL; "
        "`n/a` si RAGAS no estaba disponible en la corrida."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_chart(aggs: dict[str, dict], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping chart.")
        return False

    configs = list(aggs)
    metric_keys = [k for k, _ in CHART_METRICS]
    labels = [lbl for _, lbl in CHART_METRICS]
    n_groups = len(metric_keys)
    width = 0.8 / max(len(configs), 1)

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.4), 5))
    for i, config in enumerate(configs):
        values = [aggs[config].get(k) or 0.0 for k in metric_keys]
        positions = [x + i * width for x in range(n_groups)]
        ax.bar(positions, values, width=width, label=CONFIG_LABEL.get(config, config))

    ax.set_xticks([x + width * (len(configs) - 1) / 2 for x in range(n_groups)])
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("NormaRAG — métricas por configuración")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def generate_report(results_dir: Path = RESULTS_DIR) -> dict[str, dict]:
    results = load_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}/. Run `make eval` first.")
        return {}
    aggs = {config: aggregate(rows) for config, rows in results.items()}

    results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(aggs, results_dir / "comparativa.csv")
    write_markdown(aggs, results_dir / "comparativa.md")
    write_chart(aggs, results_dir / "comparativa.png")
    print(f"Wrote comparativa.csv / comparativa.md / comparativa.png to {results_dir}/")
    return aggs


if __name__ == "__main__":
    generate_report()
