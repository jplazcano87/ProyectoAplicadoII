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

## Uso

### Ingesta del corpus RAN

```bash
make ingest   # descarga PDFs, parsea, chunkea -> data/processed/chunks.jsonl
make index    # genera embeddings y pobla Qdrant
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

| Config | Recuperación | Prompt |
|---|---|---|
| **A** baseline | ninguna (LLM solo) | — |
| **B** rag_naive | densa top-k | citación básica |
| **C** rag_improved | híbrida (densa + BM25, fusión RRF) + rerank Cohere | citación obligatoria + abstención |

Cada mejora de C es un **toggle independiente** (`USE_HYBRID`, `USE_RERANK`, `USE_CITATION_PROMPT`) para medir su aporte incremental en ablations.

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

> **Nota de honestidad experimental (seed set, n=12):** sobre las 12 semillas —fáciles— B ya alcanza citación perfecta (techo), así que C **no** la supera en citación/faithfulness a esta escala; sí mejora claramente `context_precision` (≈0.80 vs 0.50). La comparación A/B/C definitiva requiere el golden dataset real (60–80 Q&A). La latencia de C incluye el throttle de Cohere (~6.5 s/consulta), no es latencia neta.

Salidas en `results/`: `comparativa.csv`, `comparativa.md` y `comparativa.png` con las 4 métricas RAGAS (faithfulness, answer_relevancy, context_precision, context_recall) + las métricas de citación propias (validez, exact/chapter match P/R/F1) + abstención + latencia y costo.

**Correr sobre tu propio golden dataset** (una vez construido `data/golden/golden_dataset.jsonl`):

```bash
uv run python scripts/validate_golden.py data/golden/golden_dataset.jsonl
uv run python -m normarag.eval.runner --config baseline  --dataset data/golden/golden_dataset.jsonl
uv run python -m normarag.eval.runner --config rag_naive --dataset data/golden/golden_dataset.jsonl
uv run python -m normarag.eval.report
```

Las respuestas se cachean por hash de `(config, modelo, pregunta)` en `.cache/eval/`, así que re-correr no re-paga la generación. Usa `--no-cache` para forzar y `--skip-ragas` para omitir el juez.

El **golden dataset real (60–80 Q&A)** es trabajo humano (requiere criterio normativo y validar cada referencia). El repo incluye `data/golden/seed.jsonl` (12 ejemplos semilla derivados de chunks reales) solo para ejercitar el harness.

### Tests y linting

```bash
make test
make lint
```

## Arquitectura

```
Pregunta → Retriever (denso + BM25) → Reranker (Cohere) → LLM (GPT-4o-mini) → Respuesta + citas
```

Tres configuraciones comparadas:
- **A** — LLM solo (baseline)
- **B** — RAG naive (chunking fijo + top-k denso)
- **C** — RAG mejorado (híbrido + rerank + prompt de citación)
- **D** — Agéntico (LangGraph, sub-preguntas + verificación de citas)

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

Ver `.env.example` para la lista completa. Claves requeridas:
- `OPENAI_API_KEY`
- `COHERE_API_KEY`
- `ANTHROPIC_API_KEY` (solo para juez Claude)
