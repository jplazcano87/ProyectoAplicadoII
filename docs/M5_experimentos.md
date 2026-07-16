# M5 — Experimentos, ablations y análisis de errores (NormaRAG)

Resumen del milestone M5: evaluación rigurosa de las tres configuraciones (A: LLM solo, B: RAG naive, C: RAG mejorado) sobre el golden dataset real (70 preguntas, 10 no respondibles), con ablations de C y análisis de errores.

- **Golden:** `data/golden/golden_dataset.jsonl` · 70 preguntas · sha256 `25f8e840…`
- **Modelos:** generador `gpt-4o-mini` · juez RAGAS `gpt-4o-mini` · embeddings `text-embedding-3-large` · rerank `rerank-multilingual-v3.0`
- **Parámetros:** temperature=0 · top_k_dense=20 · top_k_sparse=20 · top_k_rerank=6 · rrf_k=60
- **Artefactos generados:** `results/comparativa.md`, `results/topk_curve.png`, `results/error_analysis.md`, `results/experiment_meta.json`

> **Regla metodológica (respetada):** no se ajustaron prompts ni parámetros mirando qué preguntas fallaban. Solo se midió y se reporta tal cual, incluidos los resultados que contradicen la hipótesis.

## ⚠️ Hallazgo principal: C NO supera a B en el golden real

### (a) Tabla A/B/C (media ± desv)

RAGAS: A y B con 3 corridas (desviación real del juez); **C con 1 corrida** (ver caveats).

| Métrica | A · baseline | B · naive | C · improved |
|---|---|---|---|
| Cita exact-match F1 | 0.14 | **0.68** | 0.56 |
| Cita chapter-match F1 | 0.16 | **0.90** | 0.81 |
| Abstención F1 | 0.63 | **0.80** | 0.62 |
| RAGAS faithfulness | 0.00 | **0.91 ± 0.01** | 0.74 |
| RAGAS answer_relevancy | 0.56 | **0.70 ± 0.07** | 0.58 |
| RAGAS context_precision | 0.00 | 0.77 ± 0.04 | **0.86** |
| RAGAS context_recall | 0.00 | **0.84 ± 0.03** | 0.80 |
| Latencia media | 1.5 s | 4.9 s | 6.6 s |
| Costo/consulta | $0.00007 | $0.0011 | $0.00046 |

**B (RAG naive) gana en casi todo.** C solo mejora en `context_precision` (0.86 vs 0.77): el rerank sí hace el contexto más preciso, pero esa precisión **no se traduce** en mejores citas ni respuestas más fieles. Las diferencias A/B/C son mayores a ~1 desviación, así que son reales, no ruido del juez.

### (b) Ablations — contradicen la hipótesis

Medidas sobre métricas programáticas (RAGAS omitido en ablations por límite de rate; ver caveats). Cada fila cambia **una** variable respecto de C completo.

| Variante | cita exact F1 | abstención F1 |
|---|---|---|
| C completo | 0.557 | 0.621 |
| sin rerank (USE_RERANK off) | 0.552 | 0.621 |
| **denso (USE_HYBRID off)** | **0.586** | 0.645 |
| **prompt naive (USE_CITATION_PROMPT off)** | **0.652** | **0.857** |

- **Rerank:** prácticamente no mueve la citación (0.557 vs 0.552). Sí ayuda a `context_precision` (visible en la tabla A/B/C), pero no se convierte en mejor cita. Hipótesis (rerank→precision): **parcialmente confirmada** (ayuda a precisión, no a la métrica final).
- **Híbrido:** **empeora levemente** vs denso solo (0.586 denso > 0.557 híbrido). Hipótesis (híbrido→recall): **NO confirmada** en este set.
- **Prompt estricto de citación/abstención:** es el que **más daña**. El prompt naive da mejor citación (0.65 vs 0.56) **y** abstención (0.86 vs 0.62). El prompt estricto causa **sobre-abstención** (precisión de abstención de C = 0.47: se abstiene en preguntas que sí eran respondibles). Hipótesis (prompt→faithfulness/abstención): **contradicha** — el prompt estricto perjudica.

### Barrido de top-k (config C)

`cita exact F1` vs `top_k_rerank` (RAGAS omitido): k=3 → 0.54 · k=6 → 0.56 · **k=10 → 0.60** · k=20 → 0.59. Más contexto ayuda algo; el default k=6 no es óptimo, k=10 rinde un poco mejor. Curva en `results/topk_curve.png`.

### (c) Análisis de errores (C, 70 preguntas → 32 fallos)

| Categoría | Conteo | Lectura |
|---|---|---|
| **abstention_fail** | **11** | El mayor: C se abstiene en preguntas respondibles (por el prompt estricto) |
| hallucination | 6 | El chunk correcto estaba en contexto pero citó otro numeral |
| gold_invalid | 6 | Referencias del golden que no existen en el corpus (q006, q026, q050…) |
| ranking_miss | 5 | El numeral correcto entró a candidatos pero el rerank priorizó el preámbulo (numeral "0") |
| citation_invalid | 3 | Inventó sub-numerales tipo `2(a)`, `1.a`, `2.b` que no existen |
| retrieval_miss | 1 | La híbrida genuinamente no lo trajo |

- **Por dificultad:** easy=21, medium=7, hard=4 (dominan las easy porque son 49/70 del set).
- **Por tag (top):** `externalizacion` (5), `continuidad_negocio` (5), luego `requisitos`, `plazos`, `lavado_activos`, `prohibiciones`, `limites_credito`, `directorio`, `cruce_capitulos`.

Detalle con ejemplos concretos por categoría en `results/error_analysis.md`.

### (d) Costo total de los experimentos

~**$2–3** (generación gpt-4o-mini ~$0.37 + juez RAGAS gpt-4o-mini ~$1–2, incluyendo pasadas abortadas por el rate limit + Cohere trial gratis).

## Conclusión

La evaluación rigurosa **reveló que el RAG naive (B) es un baseline fuerte y que las "mejoras" de C no ayudan** en este golden set:
- El **prompt estricto de citación/abstención backfired** vía sobre-abstención (11 fallos) y fabricación de sub-numerales inexistentes.
- La **búsqueda híbrida no aportó** sobre el denso (incluso empeoró levemente).
- El **rerank mejora la precisión del contexto** pero prioriza chunks de preámbulo (numeral "0"), lo que no se convierte en mejor citación.

Es un hallazgo negativo bien explicado, con causa raíz clara — más valioso para la defensa que un número inflado. Próximos pasos naturales (fuera de M5, sin tocar la evaluación actual): revisar el prompt de abstención, evaluar excluir chunks de preámbulo del rerank, y subir el default de top-k.

## Caveats de la corrida

- **RAGAS parcial:** A/B con 3 corridas (desviación real); **C con 1 corrida**; **ablations y top-k sin RAGAS**. Motivo: el rate limit de la cuenta OpenAI no sostuvo el volumen de llamadas del juez (~17 pasadas) en tiempo razonable; para una PoC se priorizó terminar con métricas programáticas. Las celdas RAGAS `n/a` lo reflejan.
- **Juez = generador:** se usó `gpt-4o-mini` como juez (opción barata), mismo modelo que el generador → hay algo de sesgo de auto-evaluación. Para el veredicto RAGAS "limpio" se puede re-correr con `JUDGE_MODEL=gpt-4o` (la caché de generación hace que solo se re-pague el juez).
- **Golden con referencias inválidas:** 6–8 preguntas citan `(capítulo, numeral)` que no existen en el corpus (categoría `gold_invalid`); conviene corregirlas en el dataset.
- **std = 0 en métricas programáticas:** es correcto — la generación es determinista a temperature=0, así que citación/abstención no varían entre corridas.

## Reproducibilidad

```bash
# Suite completa (A/B/C xN + ablations + top-k) + reporte
make experiments DATASET=data/golden/golden_dataset.jsonl RUNS=3

# Reanudar tras una interrupción (salta pasadas ya hechas)
uv run python scripts/run_experiments.py --dataset data/golden/golden_dataset.jsonl --runs 3 --resume

# Solo métricas programáticas (rápido, sin juez)
uv run python scripts/run_experiments.py --dataset data/golden/golden_dataset.jsonl --skip-ragas

# Análisis de errores de C
uv run python scripts/error_analysis.py
```

Metadata de cada corrida (modelos, sha del golden, fecha, parámetros) en `results/experiment_meta.json`.
