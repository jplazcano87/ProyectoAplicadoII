# NormaRAG

Asistente de consulta sobre la normativa bancaria chilena (Recopilación Actualizada de Normas, RAN, de la CMF). Responde con cita normativa verificable (capítulo + numeral) y se evalúa contra un golden dataset propio.

## Requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) instalado
- Docker + Docker Compose

## Setup

```bash
cp .env.example .env
# Edita .env con tus claves de API
make install
```

## Reproducir todo de cero con tu golden dataset (paso a paso)

Esta es la guía completa desde un clon limpio hasta la tabla comparativa A/B/C, incluyendo dónde colocar el golden dataset cuando lo tengas.

### Paso 0 — Dónde va el golden dataset

Cuando el golden esté listo, guárdalo **exactamente** en:

```
data/golden/golden_dataset.jsonl
```

Un objeto JSON por línea, con este esquema (CLAUDE.md §8):

```json
{"id": "q001", "question": "¿…?", "answer": "Respuesta de referencia concisa.", "references": [{"chapter": "1-13", "numeral": "2.1"}], "answerable": true, "difficulty": "medium", "source_excerpt": "Fragmento que sustenta la respuesta.", "tags": ["plazos"], "origin": "human"}
```

Reglas que el validador exige:
- `id` únicos; `difficulty` ∈ `easy|medium|hard`; `origin` ∈ `seed|human`.
- Si `answerable: false` → `references: []` (preguntas no respondibles, ~10–15% del set).
- Si `answerable: true` → `references` no vacío y **cada `(chapter, numeral)` debe existir en el corpus** (`data/processed/chunks.jsonl`).

### Paso 1 — Dependencias y claves

```bash
cp .env.example .env      # completa OPENAI_API_KEY y CO_API_KEY (o COHERE_API_KEY)
make install
```

### Paso 2 — Levantar Qdrant (prerrequisito de index/eval)

```bash
make qdrant   # docker compose up -d qdrant  (requiere Docker abierto)
```

### Paso 3 — Construir el corpus (descarga CMF → parse → chunk)

```bash
make ingest               # -> data/processed/chunks.jsonl (~1993 chunks)
```

### Paso 4 — Indexar (embeddings densos + BM25 sparse en Qdrant)

```bash
make index                # -> colección normarag_ran (~1993 puntos, dense+sparse)
```

### Paso 5 — Validar el golden dataset

```bash
make validate-golden DATASET=data/golden/golden_dataset.jsonl
```

No sigas hasta ver `OK: all N examples valid`. Rechaza formato inválido, ids duplicados y referencias que no existen en el corpus.

### Paso 6 — Evaluación A/B/C + tabla comparativa

```bash
make eval DATASET=data/golden/golden_dataset.jsonl 2>&1 | tee results/eval_run.log
```

Genera respuestas de A, B y C, calcula métricas (citación propia + RAGAS + abstención + latencia/costo) y escribe `results/comparativa.md`, `comparativa.csv` y `comparativa.png`.

### Paso 7 — Ablations de C (aporte incremental de cada mejora, M5)

```bash
DS=data/golden/golden_dataset.jsonl
uv run python -m normarag.eval.runner --config rag_improved --dataset $DS --label C_full          --max-workers 1
uv run python -m normarag.eval.runner --config rag_improved --dataset $DS --label C_norerank      --no-rerank          --max-workers 1
uv run python -m normarag.eval.runner --config rag_improved --dataset $DS --label C_dense         --no-hybrid          --max-workers 1
uv run python -m normarag.eval.runner --config rag_improved --dataset $DS --label C_naiveprompt   --no-citation-prompt --max-workers 1
uv run python -m normarag.eval.report     # la tabla incluye todos los labels
```

### Demo interactiva (opcional)

```bash
make demo                 # Qdrant + API + UI ; UI en http://localhost:8501
```

**Notas de costo/tiempo:** el juez RAGAS (`gpt-4o`) tiene límite de tokens/min; el harness serializa el juez y reintenta, así que una corrida de 60–80 preguntas × 3 configs con RAGAS tarda y cuesta (el juez es lo caro). Para iterar barato usa `--skip-ragas`. Las respuestas se cachean por hash `(config, modelo, pregunta)` en `.cache/eval/`; re-correr no re-paga la generación (`--no-cache` regenera y refresca el cache tras cambiar prompts).

## Uso

> ⚠️ **Prerrequisito: Qdrant debe estar corriendo** para `make index`, `make eval`, `make experiments`, `scripts/error_analysis.py` y las consultas de la API/UI. Si `make index` falla con `Connection refused` / `[Errno 61]`, es que Qdrant está caído. Levántalo con:
> ```bash
> make qdrant           # equivale a: docker compose up -d qdrant
> ```
> (Requiere Docker Desktop abierto. Verifica con `curl -sf http://localhost:6333/readyz`.)

### Ingesta del corpus RAN

```bash
make qdrant   # 1) levanta Qdrant (prerrequisito)
make ingest   # 2) descarga PDFs, parsea, chunkea -> data/processed/chunks.jsonl
make index    # 3) genera embeddings y pobla Qdrant
```

### Demo local

```bash
docker compose up qdrant   # solo Qdrant
make demo                  # levanta Qdrant + API + UI
```

La UI queda disponible en `http://localhost:8501` y la API en `http://localhost:8000`.

> `make index` y las consultas requieren `OPENAI_API_KEY` en `.env` (embeddings + generador). La config C (RAG mejorado) además usa **Cohere** para reranking: define `CO_API_KEY` (o `COHERE_API_KEY`).

`make index` crea una colección Qdrant con **dos vectores por chunk**: `dense` (embedding OpenAI) y `sparse` (BM25 vía fastembed, con IDF server-side). Los embeddings densos se cachean en `.cache/embeddings/`, así que re-indexar no re-paga el denso.

### Configuraciones

| Config | Recuperación | Prompt / lógica |
|---|---|---|
| **A** baseline | ninguna (LLM solo) | — |
| **B** rag_naive | densa top-k | citación básica |
| **C** rag_improved | híbrida (densa + BM25, fusión RRF) + rerank Cohere | citación obligatoria + abstención |
| **D** agentic | por sub-pregunta (híbrida + rerank) | descompone → sintetiza → **self-check** de citas (LangGraph) |

Cada mejora de C es un **toggle independiente** (`USE_HYBRID`, `USE_RERANK`, `USE_CITATION_PROMPT`) para ablations. **C-v2** (M6) añade dos flags que atacan las causas raíz halladas en M5: `SOFT_ABSTENTION` (menos sobre-abstención) y `EXCLUDE_PREAMBLE` (excluye chunks de preámbulo `numeral 0` del rerank), + `top_k_rerank=10`.

### Consultar la API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Con cuánta anticipación debe avisarse el cierre de una oficina?", "config": "C", "top_k": 6}'
```

Respuesta (contrato de §9 + fuentes recuperadas):

```json
{
  "answer": "…",
  "citations": [{"chapter": "1-8", "numeral": "2"}],
  "abstained": false,
  "sources": [{"chapter_code": "1-8", "numeral": "2", "score": 0.83, "text": "…"}]
}
```

### Evaluación (A/B/C/D)

> Requiere **Python 3.11** (fijado en `.python-version`; RAGAS no funciona en 3.14). `make eval` necesita `OPENAI_API_KEY` (generación + embeddings) y, para RAGAS, `JUDGE_MODEL` sobre OpenAI.

```bash
make eval             # corre A, B y C sobre el seed set -> results/
make validate-golden  # valida el schema del golden y que las referencias existan en el corpus
```

**Ablations de C** (cada mejora por separado; `--label` nombra el archivo de salida):

```bash
# C completo
uv run python -m normarag.eval.runner --config rag_improved --label C_full --max-workers 1
# sin rerank
uv run python -m normarag.eval.runner --config rag_improved --no-rerank --label C_norerank --max-workers 1
# denso en vez de híbrido
uv run python -m normarag.eval.runner --config rag_improved --no-hybrid --label C_dense --max-workers 1
# prompt naive en vez de citación estricta
uv run python -m normarag.eval.runner --config rag_improved --no-citation-prompt --label C_naiveprompt --max-workers 1
uv run python -m normarag.eval.report   # la tabla incluye todos los labels encontrados
```

> Cohere (trial) limita rerank a ~10 req/min; el reranker aplica un throttle interno y reintentos. Corre C con `--max-workers 1`.

> **Nota de honestidad experimental (seed set, n=12):** sobre las 12 semillas —fáciles— B ya alcanza citación perfecta (techo), así que C la **empata** en citación (1.00) y queda algo por debajo en faithfulness (0.83 vs 1.00, ruido a esta escala); C **sí** mejora `context_precision` (~0.78 vs 0.69) y `answer_relevancy` (~0.66 vs 0.62). La comparación A/B/C definitiva requiere el golden dataset real (60–80 Q&A). La latencia de C incluye el throttle de Cohere (~6.5 s/consulta), no es latencia neta.

Salidas en `results/`: `comparativa.csv`, `comparativa.md` y `comparativa.png` con las 4 métricas RAGAS (faithfulness, answer_relevancy, context_precision, context_recall) + las métricas de citación propias (validez, exact/chapter match P/R/F1) + abstención + latencia y costo.

Para correr sobre **tu golden dataset real**, ver la guía completa **[Reproducir todo de cero](#reproducir-todo-de-cero-con-tu-golden-dataset-paso-a-paso)** (colócalo en `data/golden/golden_dataset.jsonl` y usa `make eval DATASET=…`).

El **golden dataset real (60–80 Q&A)** es trabajo humano (requiere criterio normativo y validar cada referencia). El repo incluye `data/golden/seed.jsonl` (12 ejemplos semilla derivados de chunks reales) solo para ejercitar el harness.

### Cierre M6 (agéntico D, C-v2 y validez del instrumento)

```bash
# 1) Referencias del golden que NO existen en el corpus -> checklist a corregir a mano
make gold-report DATASET=data/golden/golden_dataset.jsonl   # -> results/gold_to_fix.md

# 2) Config D (agéntico) y C-v2 en el harness (mismo contrato que A/B/C)
uv run python -m normarag.eval.runner --config agentic      --dataset data/golden/golden_dataset.jsonl --max-workers 1
uv run python -m normarag.eval.runner --config rag_improved --dataset data/golden/golden_dataset.jsonl \
  --label C_v2 --soft-abstention --exclude-preamble --max-workers 1   # + top_k_rerank=10 vía env/variant

# 3) Re-juez independiente (gpt-4o) reusando la generación cacheada, resumible por lotes
JUDGE_MODEL=gpt-4o uv run python scripts/rejudge_ragas.py results/main/*.jsonl --batch 10
```

> **C-v2 se reporta como experimento SEPARADO** ("validación de hipótesis post-M5"): no sobrescribe los resultados de M5. La narrativa es observar → diagnosticar → intervenir sobre la causa → medir.

### Tests y linting

```bash
make test
make lint
```

## Arquitectura

```
Pregunta → Retriever (denso + BM25) → Reranker (Cohere) → LLM (GPT-4o-mini) → Respuesta + citas
```

Configuraciones comparadas:
- **A** — LLM solo (baseline)
- **B** — RAG naive (chunking fijo + top-k denso)
- **C** — RAG mejorado (híbrido + rerank + prompt de citación)
- **D** — Agéntico (LangGraph, sub-preguntas + verificación de citas)
- **C-v2** — C con intervenciones sobre las causas raíz halladas en M5

## Resultados y conclusiones

- **[docs/CONCLUSIONES.md](docs/CONCLUSIONES.md)** — resumen ejecutivo listo para informe (qué se construyó, hallazgos, contribución, limitaciones, trabajo futuro).
- [docs/M5_experimentos.md](docs/M5_experimentos.md) — experimentos A/B/C, ablations, análisis de errores.
- [docs/M6_cierre.md](docs/M6_cierre.md) — capa agéntica, C-v2 y tabla consolidada.

**Hallazgo central (honesto):** el RAG naive (B) es un baseline fuerte que las mejoras "sofisticadas" no superan en citación; C-v2 valida el diagnóstico (recupera la sobre-abstención) y D garantiza validez de citas vía self-check. La contribución real es el **instrumento de evaluación** que lo mide con rigor.

## Estructura del repositorio

```
src/normarag/
├── corpus/     # descarga, parsing y chunking del corpus RAN
├── index/      # embeddings y vector store (Qdrant)
├── retrieval/  # retriever híbrido y reranker
├── generation/ # cliente LLM y prompts
├── pipeline/   # pipelines A/B/C/D
├── eval/       # métricas, runner y reporte
├── api/        # FastAPI
└── ui/         # Streamlit
```

## Variables de entorno

Ver `.env.example` para la lista completa. Claves:
- `OPENAI_API_KEY` — embeddings, generador y juez RAGAS (requerida).
- `CO_API_KEY` **o** `COHERE_API_KEY` — reranking de la config C (basta una).
- `ANTHROPIC_API_KEY` — opcional, solo si usas Claude como juez.

## Experimentar con configuraciones

Todo se ajusta por **variables de entorno** (`.env`), sin tocar código. Lo importante es **qué exige re-procesar**:

| Qué cambias | Variable(s) | ¿Requiere? |
|---|---|---|
| **Modelo generador** (más caro/mejor) | `GENERATOR_MODEL` (p. ej. `gpt-4o`, `gpt-4.1`) | nada — efecto inmediato. Borra `.cache/eval/` para regenerar respuestas |
| **Modelo juez RAGAS** | `JUDGE_MODEL` (p. ej. `gpt-4o`) | nada — re-juzga con `scripts/rejudge_ragas.py` (reusa generación) |
| **Modelo de embeddings** | `EMBEDDING_MODEL` + `EMBEDDING_DIM` | **re-indexar** (`make index`); la dim DEBE coincidir (3-large=3072, 3-small=1536) |
| **Reranker** | `RERANK_MODEL` | nada — efecto inmediato |
| **Tamaño de chunk** | `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_RATIO` | **re-chunkear + re-indexar** (`make ingest && make index`) |
| **Top-k de recuperación** | `TOP_K_DENSE`, `TOP_K_SPARSE`, `TOP_K_RERANK`, `RRF_K` | nada — efecto inmediato en consulta/eval |

**Ejemplos:**

```bash
# Generador más caro (mejor calidad, más costo). Borra la caché para regenerar.
echo "GENERATOR_MODEL=gpt-4o" >> .env
rm -rf .cache/eval
make eval DATASET=data/golden/golden_dataset.jsonl

# Probar chunks más chicos (más granular) o más grandes (más contexto por chunk)
echo "CHUNK_MAX_TOKENS=400" >> .env    # o 1200
make ingest && make index             # re-chunkea y re-indexa
make eval DATASET=data/golden/golden_dataset.jsonl

# Cambiar el modelo de embeddings (¡ajusta también la dimensión!)
printf "EMBEDDING_MODEL=text-embedding-3-small\nEMBEDDING_DIM=1536\n" >> .env
make index                            # re-indexa con la nueva dim
```

> **Costo/caché:** la generación se cachea por `(config, modelo, pregunta)`. Al cambiar de generador o chunking, borra `.cache/eval/` (o usa `--no-cache`) para forzar la regeneración; los embeddings se cachean por `(modelo, texto)` en `.cache/embeddings/`. El juez RAGAS nunca se cachea (se re-paga siempre).
