"""Consolidated M5 experiment report.

Reads the artifacts produced by scripts/run_experiments.py:

  results/main/<config>__run{1..N}.jsonl   -> A/B/C table with mean ± std
  results/ablations/<label>.jsonl          -> one-variable ablations of C
  results/topk/C_k{3,6,10,20}.jsonl        -> top-k sweep curve
  results/experiment_meta.json             -> models, golden version, date, params

and writes results/comparativa.md (+ .csv) and results/topk_curve.png.

Variance note: with a fixed, low temperature the generation is deterministic, so
the programmatic citation/abstention metrics are identical across runs (std 0);
the spread comes from the RAGAS judge. A gap smaller than ~1 std between two
configs is not a real difference — this is stated in the report.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from normarag.eval.report import _fmt, aggregate

RESULTS_DIR = Path("results")

CONFIG_ORDER = ["baseline", "rag_naive", "rag_improved", "agentic"]
CONFIG_LABEL = {
    "baseline": "A · baseline",
    "rag_naive": "B · RAG naive",
    "rag_improved": "C · RAG improved",
    "agentic": "D · agentic",
}

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
    ("cost_usd_mean", "Costo medio/consulta (USD)"),
]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_runs(main_dir: Path) -> dict[str, list[list[dict]]]:
    """config -> list of per-run row lists."""
    runs: dict[str, list[list[dict]]] = {}
    for config in CONFIG_ORDER:
        files = sorted(main_dir.glob(f"{config}__run*.jsonl"))
        if files:
            runs[config] = [_read_jsonl(f) for f in files]
    return runs


def aggregate_runs(run_row_lists: list[list[dict]]) -> dict[str, tuple[float | None, float]]:
    """Per-metric (mean, std) across runs. std is 0 for deterministic metrics."""
    per_run = [aggregate(rows) for rows in run_row_lists]
    out: dict[str, tuple[float | None, float]] = {}
    for key, _ in METRIC_ROWS:
        vals = [r.get(key) for r in per_run if isinstance(r.get(key), (int, float))]
        if not vals:
            out[key] = (None, 0.0)
        else:
            mean = sum(vals) / len(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            out[key] = (mean, std)
    return out


def _fmt_mean_std(value: tuple[float | None, float] | None) -> str:
    if value is None or value[0] is None:
        return "n/a"
    mean, std = value
    return f"{_fmt(mean)} ± {_fmt(std)}"


# --- main A/B/C table -------------------------------------------------------


def _main_table(runs: dict[str, list[list[dict]]]) -> tuple[list[str], dict[str, dict]]:
    aggs = {config: aggregate_runs(rl) for config, rl in runs.items()}
    configs = [c for c in CONFIG_ORDER if c in aggs]
    header = "| Métrica | " + " | ".join(CONFIG_LABEL[c] for c in configs) + " |"
    sep = "|" + "---|" * (len(configs) + 1)
    lines = [header, sep]
    for key, label in METRIC_ROWS:
        cells = " | ".join(_fmt_mean_std(aggs[c].get(key)) for c in configs)
        lines.append(f"| {label} | {cells} |")
    return lines, aggs


# --- ablations --------------------------------------------------------------

ABLATIONS = [
    ("C_norerank", "Sin rerank (USE_RERANK off)", "context_precision"),
    ("C_dense", "Denso (USE_HYBRID off)", "context_recall"),
    ("C_naiveprompt", "Prompt naive (USE_CITATION_PROMPT off)", "faithfulness"),
]
ABLATION_METRICS = [
    ("context_precision", "context_precision"),
    ("context_recall", "context_recall"),
    ("faithfulness", "faithfulness"),
    ("abstention_f1", "abstención F1"),
    ("citation_exact_f1", "cita exact F1"),
]


def _ablation_tables(ablations_dir: Path) -> list[str]:
    full_path = ablations_dir / "C_full.jsonl"
    if not full_path.exists():
        return ["_(sin datos de ablations)_"]
    full = aggregate(_read_jsonl(full_path))

    lines: list[str] = []
    header = "| Ablation | " + " | ".join(m for _, m in ABLATION_METRICS) + " | (foco) |"
    sep = "|" + "---|" * (len(ABLATION_METRICS) + 2)
    lines += [header, sep]
    lines.append(
        "| **C completo** | "
        + " | ".join(_fmt(full.get(k)) for k, _ in ABLATION_METRICS)
        + " | — |"
    )
    for label_file, label, focus in ABLATIONS:
        path = ablations_dir / f"{label_file}.jsonl"
        if not path.exists():
            continue
        agg = aggregate(_read_jsonl(path))
        cells = " | ".join(_fmt(agg.get(k)) for k, _ in ABLATION_METRICS)
        delta = None
        if isinstance(agg.get(focus), (int, float)) and isinstance(full.get(focus), (int, float)):
            delta = full[focus] - agg[focus]  # how much the improvement adds
        delta_s = f"Δ{focus}={_fmt(delta)}" if delta is not None else focus
        lines.append(f"| {label} | {cells} | {delta_s} |")
    lines.append("")
    lines.append(
        "> Δfoco = (C completo − ablation) en la métrica foco: cuánto aporta esa mejora. "
        "Hipótesis: rerank→context_precision, híbrido→context_recall, prompt→faithfulness/abstención."
    )
    return lines


# --- top-k sweep ------------------------------------------------------------

TOPK_VALUES = [3, 6, 10, 20]
TOPK_METRICS = [
    ("context_precision", "context_precision"),
    ("context_recall", "context_recall"),
    ("faithfulness", "faithfulness"),
    ("citation_exact_f1", "cita exact F1"),
]


def _topk_table_and_chart(topk_dir: Path, chart_path: Path) -> list[str]:
    data: dict[int, dict] = {}
    for k in TOPK_VALUES:
        path = topk_dir / f"C_k{k}.jsonl"
        if path.exists():
            data[k] = aggregate(_read_jsonl(path))
    if not data:
        return ["_(sin datos de barrido top-k)_"]

    header = "| top_k_rerank | " + " | ".join(m for _, m in TOPK_METRICS) + " |"
    sep = "|" + "---|" * (len(TOPK_METRICS) + 1)
    lines = [header, sep]
    for k in TOPK_VALUES:
        if k in data:
            cells = " | ".join(_fmt(data[k].get(mk)) for mk, _ in TOPK_METRICS)
            lines.append(f"| {k} | {cells} |")

    _write_topk_chart(data, chart_path)
    lines.append("")
    lines.append(f"![curva top-k]({chart_path.name})")
    return lines


def _write_topk_chart(data: dict[int, dict], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    ks = sorted(data)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mk, label in TOPK_METRICS:
        ys = [data[k].get(mk) for k in ks]
        if any(y is not None for y in ys):
            ax.plot(ks, [y if y is not None else float("nan") for y in ys], marker="o", label=label)
    ax.set_xlabel("top_k_rerank")
    ax.set_ylabel("score")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1)
    ax.set_title("NormaRAG — barrido de top-k (config C)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- assembly ---------------------------------------------------------------


def _metadata_block(meta: dict) -> list[str]:
    if not meta:
        return []
    models = meta.get("models", {})
    golden = meta.get("golden", {})
    return [
        "## Reproducibilidad",
        "",
        f"- **Fecha:** {meta.get('date', 'n/a')}",
        f"- **Golden:** `{golden.get('path', 'n/a')}` · {golden.get('count', '?')} preguntas "
        f"· sha256 `{golden.get('sha256', '?')[:12]}`",
        f"- **Modelos:** generador `{models.get('generator')}` · juez `{models.get('judge')}` "
        f"· embeddings `{models.get('embedding')}` · rerank `{models.get('rerank')}`",
        f"- **Parámetros:** temperature=0 · top_k_dense={meta.get('top_k_dense')} "
        f"· top_k_sparse={meta.get('top_k_sparse')} · top_k_rerank={meta.get('top_k_rerank')} "
        f"· rrf_k={meta.get('rrf_k')} · runs={meta.get('n_runs')}",
        "",
    ]


def generate_experiment_report(results_dir: Path = RESULTS_DIR) -> None:
    meta_path = results_dir / "experiment_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    runs = load_runs(results_dir / "main")
    if not runs:
        print(f"No main runs in {results_dir}/main/. Run scripts/run_experiments.py first.")
        return

    main_lines, _ = _main_table(runs)
    n_runs = len(next(iter(runs.values())))

    out = ["# NormaRAG — Experimentos M5 (A/B/C, ablations, top-k)", ""]
    out += _metadata_block(meta)
    out += [f"## 1. Comparativa A/B/C (media ± desviación, {n_runs} corridas)", ""]
    out += main_lines
    out += [
        "",
        "> La desviación viene del juez RAGAS (la generación es determinista a "
        "temperature=0, por eso citación/abstención tienen std 0). **Una diferencia "
        "menor a ~1 desviación entre dos configs no es una diferencia real.**",
        "",
        "## 2. Ablations de C (una variable a la vez)",
        "",
    ]
    out += _ablation_tables(results_dir / "ablations")
    out += ["", "## 3. Barrido de top-k (config C)", ""]
    out += _topk_table_and_chart(results_dir / "topk", results_dir / "topk_curve.png")
    out += [
        "",
        "> Métricas de citación/abstención: 100% programáticas. RAGAS: juez = JUDGE_MODEL. "
        "`n/a` = no computado en esa corrida.",
        "",
    ]

    (results_dir / "comparativa.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {results_dir}/comparativa.md (+ topk_curve.png)")


if __name__ == "__main__":
    generate_experiment_report()
