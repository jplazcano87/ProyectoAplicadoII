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

### Evaluación (A/B/C/D)

```bash
make eval   # corre pipelines sobre golden set -> results/
```

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
