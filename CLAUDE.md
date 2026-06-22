# NormaRAG — Plan de construcción y spec para Claude Code




**Las tareas que dicen **Humano** son validaciones que Claude no debe realizar**


---

## 1. Objetivo

Construir **NormaRAG**, un asistente de consulta sobre la normativa bancaria chilena (Recopilación Actualizada de Normas, RAN, de la CMF) que responde con **cita normativa verificable** (capítulo + numeral) y que **se evalúa rigurosamente contra un baseline**.

El entregable que más pesa no es el chatbot: es la **evaluación**. Todo el diseño se ordena alrededor de poder comparar de forma reproducible tres configuraciones (A: LLM solo, B: RAG naive, C: RAG mejorado) sobre un golden dataset propio, con una métrica propia de exactitud de citación.

**No-objetivos** (para acotar el alcance):
- No es asesoría legal ni regulatoria; es una herramienta de búsqueda asistida.
- No usa datos internos de ninguna institución; solo el corpus público de la CMF.

---

## 2. Decisiones de stack (fijadas)

| Capa | Elección | Notas |
|------|----------|-------|
| Lenguaje | Python 3.11 | |
| Gestor de paquetes | `uv` | rápido y reproducible (`uv.lock`). Alternativa: Poetry. |
| Descarga corpus | `httpx` + `selectolax`/`beautifulsoup4` | scraping del índice RAN de cmfchile.cl |
| Parsing PDF | `pymupdf` (fitz) | rápido y buena extracción de texto/posiciones |
| Chunking | propio, estructural | por capítulo / título / numeral (ver §7) |
| Embeddings | **OpenAI `text-embedding-3-large`** (API) | configurable vía env |
| Vector store | **Qdrant** (Docker) | soporta búsqueda densa + sparse (BM25) nativa |
| Sparse / léxico | BM25 nativo de Qdrant (fastembed) | para búsqueda híbrida |
| Reranking | **Cohere `rerank-multilingual-v3`** (API) | buen rendimiento en español; alternativa: cross-encoder local |
| LLM generador | **OpenAI `gpt-4o-mini`** (API) | configurable; barato para iterar |
| LLM juez (eval) | **modelo distinto/más fuerte** (`gpt-4o` o Claude) | reduce sesgo de auto-evaluación |
| Orquestación | Python plano en M1–M5; **LangGraph** solo para la capa agéntica (M6) | evita complejidad innecesaria temprano |
| API | FastAPI | |
| UI / demo | Streamlit | live demo de la entrega final |
| Evaluación | **RAGAS** + métrica de citación propia | ver §9 |
| Tests | pytest | |
| Calidad | ruff + black + mypy (gradual) | |
| Contenedores | Docker + docker compose | `docker compose up` debe levantar todo |

**Principio de diseño:** todos los proveedores externos (embeddings, rerank, LLM) detrás de **interfaces** (`EmbeddingProvider`, `Reranker`, `LLMClient`) para poder cambiarlos por env sin tocar la lógica. Esto además te permite, si en la entrega final quieres mostrarlo, intercambiar a un modelo local y comparar.

---

## 3. Estructura del repositorio

```
normarag/
├── CLAUDE.md                  # este documento
├── README.md
├── pyproject.toml             # deps + config de herramientas
├── uv.lock
├── .env.example               # plantilla de variables (sin secretos)
├── .gitignore                 # incluye .env, data/raw/, .cache/
├── docker-compose.yml         # qdrant + api + ui
├── Dockerfile
├── Makefile                   # targets: install, ingest, index, eval, demo, test
├── config/
│   └── settings.py            # carga de env (pydantic-settings)
├── src/normarag/
│   ├── __init__.py
│   ├── corpus/
│   │   ├── download.py        # scraping índice RAN + descarga PDFs (cacheado)
│   │   ├── parse.py           # PDF -> texto estructurado
│   │   └── chunk.py           # chunking estructural (capítulo/título/numeral)
│   ├── index/
│   │   ├── embeddings.py      # EmbeddingProvider (OpenAI)
│   │   └── store.py           # Qdrant: upsert + búsqueda híbrida
│   ├── retrieval/
│   │   ├── retriever.py       # dense + sparse + fusión
│   │   └── rerank.py          # Reranker (Cohere)
│   ├── generation/
│   │   ├── llm.py             # LLMClient (OpenAI/Anthropic)
│   │   └── prompt.py          # prompt con citación obligatoria + formato de salida
│   ├── pipeline/
│   │   ├── baseline.py        # A: LLM solo (sin retrieval)
│   │   ├── rag_naive.py       # B: chunking fijo + top-k
│   │   └── rag_improved.py    # C: híbrido + rerank + prompt citación
│   ├── eval/
│   │   ├── metrics.py         # RAGAS wrappers + métrica de citación propia
│   │   ├── runner.py          # corre A/B/C sobre el golden set -> resultados
│   │   └── report.py          # tabla comparativa + gráficos
│   ├── api/
│   │   └── main.py            # FastAPI
│   └── ui/
│       └── app.py             # Streamlit
├── data/
│   ├── raw/                   # PDFs descargados (gitignored)
│   ├── processed/             # chunks + metadatos (jsonl)
│   └── golden/
│       └── golden_dataset.jsonl   # 🧑 dataset propio (versionado en git)
├── results/                   # salidas de evaluación (tablas, plots)
└── tests/
    ├── test_chunk.py
    ├── test_retrieval.py
    └── test_citation_metric.py
```

---

## 4. Configuración y secretos


- Crea placeholders para las claves de API: `OPENAI_API_KEY`, `COHERE_API_KEY`, y si usarás Claude como juez `ANTHROPIC_API_KEY`.
- Copia `.env.example` a `.env` y el usuario llenará los valores

`.env.example` (Claude Code lo crea con estos campos vacíos):

```
# Proveedores
OPENAI_API_KEY=
COHERE_API_KEY=
ANTHROPIC_API_KEY=

# Modelos (overrideables)
EMBEDDING_MODEL=text-embedding-3-large
GENERATOR_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o
RERANK_MODEL=rerank-multilingual-v3.0

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=normarag_ran

# Retrieval
TOP_K_DENSE=20
TOP_K_SPARSE=20
TOP_K_RERANK=6
```

---

## 5. Convenciones de colaboración (repo grupal)

- **Flujo:** trunk-based ligero. `main` protegida; trabajo en ramas `feat/…`, `fix/…`, `chore/…`; merge vía **Pull Request** con al menos 1 review.
- **Commits:** Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
- **Una fase = una rama = un PR.** El PR de cada milestone debe cerrar sus criterios de aceptación.
- **CI (GitHub Actions):** en cada PR corre `ruff`, `black --check`, `mypy` (no bloqueante al inicio) y `pytest`.
- **`CODEOWNERS`** opcional para repartir áreas entre integrantes.
- **Definition of Done (DoD) por tarea:** código + tests del módulo + docstrings + pasa CI + actualiza README/Makefile si cambió el uso.

---

## 6. Plan por fases (milestones)

Cada milestone mapea a una semana del plan de la propuesta. Para cada uno: **objetivo**, **tareas de Claude Code**, **criterio de aceptación (verificable)** y **pasos humanos**.

### M0 — Scaffolding (día 0–1, previo a S1)
**Objetivo:** repo ejecutable vacío con tooling.
**Claude Code:**
- Inicializa `pyproject.toml` con `uv`, dependencias base, ruff/black/mypy/pytest.
- Crea estructura de carpetas de §3, `settings.py` (pydantic-settings), `.env.example`, `.gitignore`, `Makefile` con targets stub, `docker-compose.yml` con servicio `qdrant`, `Dockerfile`, `README` inicial, workflow de CI.
- Un test trivial que pase.
**Aceptación:** `make install` funciona; `docker compose up qdrant` levanta Qdrant; `make test` verde; CI verde en el primer PR.

### M1 — Corpus: descarga + parsing + chunking (S1)
**Objetivo:** corpus RAN descargado, parseado y chunked con metadatos.
**Claude Code:**
- `corpus/download.py`: scrapea el índice de la RAN (ver §7), arma lista `(chapter_code, title, pdf_url)`, descarga PDFs a `data/raw/` con **caché** (no re-descargar) y **rate limiting** (≥1 s entre requests, user-agent identificable).
- `corpus/parse.py`: extrae texto por página con PyMuPDF.
- `corpus/chunk.py`: chunking **estructural** (§7) → `data/processed/chunks.jsonl` con el schema de §7.
- Tests de chunking sobre 2–3 capítulos de muestra.
**Aceptación:** `make ingest` produce `chunks.jsonl`; cada chunk tiene `chapter_code`, `numeral`, `text`, `n_tokens`; muestreo manual de 10 chunks se ve coherente (no cortes a mitad de palabra, metadatos correctos).
🧑 **HUMANO:** revisar términos de uso / robots del sitio de la CMF (es data pública del regulador; aun así, scraping respetuoso). Verificar visualmente algunos chunks.

### M2 — Indexación + RAG naive end-to-end (S2)
**Objetivo:** preguntar y obtener respuesta con contexto recuperado.
**Claude Code:**
- `index/embeddings.py` (OpenAI) + `index/store.py` (upsert a Qdrant con metadatos).
- `retrieval/retriever.py`: búsqueda densa top-k.
- `generation/llm.py` + `generation/prompt.py`: prompt que responde citando capítulo/numeral del contexto.
- `pipeline/rag_naive.py`: orquesta retrieve → generate.
- `api/main.py`: endpoint `POST /query`. `ui/app.py`: input + respuesta + fuentes.
**Aceptación:** `make index` puebla Qdrant; `docker compose up` levanta api+ui+qdrant; una consulta de prueba devuelve respuesta + lista de fuentes recuperadas. **Demo v0.1.**
🧑 **HUMANO:** cargar `OPENAI_API_KEY`.

### M3 — Golden dataset + harness de evaluación (S3)
**Objetivo:** poder medir A/B/C de forma reproducible.
**Claude Code:**
- Define el schema y un validador del dataset (§8).
- `eval/metrics.py`: wrappers RAGAS (faithfulness, answer_relevancy, context_precision, context_recall) + **métrica de citación propia** (§9).
- `eval/runner.py`: corre una config sobre el golden set y guarda `results/<config>.jsonl`.
- `pipeline/baseline.py` (A: LLM solo) para tener el baseline desde ya.
- `eval/report.py`: tabla comparativa + gráficos a `results/`.
- Provee **10–15 ejemplos semilla** del dataset (formato correcto) para que el harness corra de punta a punta; los marca como `seed` para que el humano los reemplace/expanda.
**Aceptación:** `make eval` corre A y B sobre el seed set y emite tabla con las 4 métricas RAGAS + citación; el validador rechaza ejemplos mal formados.
🧑 **HUMANO (clave):** **construir y validar el golden dataset real (60–80 Q&A)**. Esto NO lo hace Claude Code: requiere criterio normativo. Incluye ~10–15% de preguntas **no respondibles** por el corpus para probar la abstención. Proceso sugerido en §8.

### M4 — RAG mejorado (S4)
**Objetivo:** la configuración C que supera al baseline.
**Claude Code:**
- `retrieval/retriever.py`: búsqueda **híbrida** (densa + BM25 sparse de Qdrant) con fusión (RRF).
- `retrieval/rerank.py`: Cohere rerank sobre los candidatos.
- `pipeline/rag_improved.py`: híbrido + rerank + prompt con **citación obligatoria** y **abstención** ("si no hay evidencia suficiente, dilo y no inventes").
- Formato de salida estructurado: respuesta + `citations: [{chapter, numeral}]` parseables (ver §9).
**Aceptación:** `make eval` corre A/B/C; C mejora a B en al menos faithfulness y exactitud de citación sobre el seed set (y luego sobre el real).

### M5 — Experimentos, análisis y reporte (S5)
**Objetivo:** evidencia sólida para la presentación.
**Claude Code:**
- Ejecuta la matriz A/B/C sobre el golden real; genera tabla final + gráficos.
- **Ablations**: con/sin rerank, denso vs híbrido, top-k. Una tabla por ablation.
- Script de **análisis de errores**: agrupa fallos (alucinación, cita inválida, abstención incorrecta) con ejemplos.
**Aceptación:** `results/` contiene tabla comparativa, gráficos y un `error_analysis.md` reproducibles con un solo comando.

### M6 — Novedad agéntica + cierre (S6)
**Objetivo:** el elemento novedoso de la entrega final.
**Claude Code:**
- `pipeline/` agéntico con **LangGraph**: (1) descompone consultas complejas en sub-preguntas, (2) recupera por sub-pregunta, (3) sintetiza, (4) **verifica** que cada cita exista en el corpus antes de responder (self-check de citación).
- Mídelo como configuración **D** contra C en el mismo harness.
- Pulido de la demo Streamlit (muestra sub-preguntas y citas verificadas) y del README.
**Aceptación:** D corre en el harness y se compara con C; la demo muestra el flujo agéntico; `docker compose up` reproduce todo desde cero.

---

## 7. Capa de datos: corpus RAN

**Fuente:** índice de la Recopilación Actualizada de Normas para Bancos de la CMF (página de "Ordenada por Capítulo"). Cada capítulo es un PDF individual de descarga directa (patrón de URL tipo `…/articles-XXXXX_doc_pdf.pdf`). Hay además un PDF de índice de capítulos.

**Estrategia de descarga:**
1. Fetch de la página índice → extraer, por cada capítulo, `chapter_code` (ej. `1-13`), `title` y la URL del PDF.
2. Descargar a `data/raw/<chapter_code>.pdf` con caché (saltar si ya existe) y rate limiting.
3. Registrar un `manifest.json` con código, título, url y hash de cada PDF (para trazabilidad y reproducibilidad).

> Nota de red: el dominio de la CMF debe estar accesible desde el entorno donde corre Claude Code (tu máquina). Si usas un sandbox con allowlist, agrégalo.

**Chunking estructural.** La RAN se organiza en Capítulo → Título (I, II, …) → numerales (1, 1.1, 2.3, …). El chunk **mínimo es el numeral**, conservando el contexto del capítulo/título en metadatos.

Schema de cada chunk (`data/processed/chunks.jsonl`):

```json
{
  "chunk_id": "1-13::II::2.1::0",
  "chapter_code": "1-13",
  "chapter_title": "Clasificación de gestión y solvencia",
  "title_section": "II",
  "numeral": "2.1",
  "text": "…texto del numeral…",
  "page_start": 4,
  "page_end": 5,
  "n_tokens": 312,
  "source_url": "https://www.cmfchile.cl/…articles-XXXXX_doc_pdf.pdf"
}
```

Reglas:
- Detectar encabezados de capítulo, título y numeral por **regex** sobre el texto extraído (patrones tipo `^\s*\d+(\.\d+)*\s`, romanos para títulos, encabezado de capítulo en la primera página).
- Si un numeral excede un límite de tokens (ej. 800), **subdividir con solapamiento** (~15%) manteniendo los mismos metadatos + sufijo de índice.
- Guardar el texto del encabezado (capítulo + título + numeral) como prefijo opcional para el embedding (mejora la recuperación), pero conservar el `text` limpio aparte.

---

## 8. Golden dataset (🧑 responsabilidad humana)

Es el activo más valioso del proyecto y **una contribución en sí misma**. Claude Code provee el schema, el validador y ejemplos semilla; **tú lo construyes y validas**.

Schema (`data/golden/golden_dataset.jsonl`):

```json
{
  "id": "q001",
  "question": "¿Qué plazo tiene un banco para informar X a la CMF?",
  "answer": "Respuesta de referencia, concisa y correcta.",
  "references": [{"chapter": "1-13", "numeral": "2.1"}],
  "answerable": true,
  "difficulty": "medium",
  "source_excerpt": "Fragmento del numeral que sustenta la respuesta.",
  "tags": ["plazos", "reporte"],
  "origin": "seed | human"
}
```

Proceso sugerido (eficiente y defendible):
1. **Borrador asistido:** usa el propio sistema (o un LLM con el numeral a la vista) para *proponer* preguntas y respuestas candidatas a partir de chunks reales. Esto acelera, pero **no** es el dataset final.
2. **Validación humana:** revisa cada par, corrige la respuesta, **confirma a mano la referencia** (capítulo + numeral) abriendo el PDF. Marca `origin: human`.
3. **Cobertura:** distribuye por dificultad y por tipo (plazos, definiciones, requisitos, prohibiciones). Incluye **8–12 no respondibles** (`answerable: false`, `references: []`) para medir abstención.
4. **Tamaño objetivo:** 60–80 ejemplos validados.

> Defensa metodológica para la presentación: documenta que las preguntas no respondibles y la validación manual de referencias evitan el "data leakage" y permiten medir alucinación de forma honesta.

---

## 9. Harness de evaluación

**Configuraciones comparadas:** A (LLM solo, baseline) · B (RAG naive) · C (RAG mejorado) · D (agéntico, M6). Todas corren sobre el **mismo** golden set y emiten el **mismo** formato de salida para ser comparables.

**Formato de salida del sistema** (para poder medir citación automáticamente):

```json
{
  "answer": "texto de la respuesta",
  "citations": [{"chapter": "1-13", "numeral": "2.1"}],
  "abstained": false
}
```

**Métricas:**
- **RAGAS:** faithfulness, answer_relevancy, context_precision, context_recall (juez = `JUDGE_MODEL`, distinto del generador).
- **Métrica propia — exactitud de citación** (`eval/metrics.py`), 100% programática:
  - *Validez de cita:* ¿cada `(chapter, numeral)` citado **existe** en el corpus indexado? (verificable contra `chunks.jsonl`).
  - *Correctitud de cita:* ¿el conjunto de citas **coincide** con `references` del gold? Reporta **exact match** (capítulo+numeral) y **match a nivel de capítulo** (más laxo).
  - *Abstención:* para `answerable=false`, ¿el sistema se abstuvo (`abstained=true`)? Reporta precisión/recall de abstención.
- **Latencia y costo** por consulta (útil como dimensión práctica en la presentación).

**Salida:** `eval/report.py` genera una tabla `results/comparativa.csv` + `comparativa.md` y gráficos de barras por métrica (A/B/C/D). Todo reproducible con `make eval`.

---

## 10. Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|-----------|
| Estructura de PDFs inconsistente entre capítulos | Empezar con un subconjunto (5–8 capítulos) en M1; generalizar el parser con tests por patrón. |
| Golden dataset consume tiempo | Borrador asistido + validar solo lo necesario; 60 ya sirve. Es trabajo humano, planificarlo desde S1. |
| Costos de API en evaluación | `gpt-4o-mini` para generar; juez fuerte solo en la corrida final; cachear embeddings y respuestas por hash de input. |
| Scraping frágil/bloqueado | Caché local + `manifest.json`; si cambia el sitio, los PDFs ya descargados siguen sirviendo. |
| C no supera a B | El análisis de errores y ablations (M5) es valioso aunque la mejora sea modesta: explica *por qué*. La evaluación honesta es el objetivo, no un número. |

---

## 11. Comandos (Makefile)

```
make install     # uv sync + pre-commit
make ingest      # descarga + parse + chunk -> data/processed/chunks.jsonl
make index       # embeddings + upsert a Qdrant
make demo        # docker compose up (qdrant + api + ui)
make eval        # corre A/B/C(/D) sobre el golden set -> results/
make test        # pytest
make lint        # ruff + black --check + mypy
```

---

## 12. Definition of Done del proyecto

- `docker compose up` desde un clon limpio levanta Qdrant + API + UI.
- `make ingest && make index && make eval` reproduce los resultados sin pasos manuales (salvo `.env`).
- `results/` contiene la tabla comparativa A/B/C/D y los gráficos de la presentación.
- `data/golden/golden_dataset.jsonl` versionado y validado.
- README explica setup, arquitectura y cómo reproducir la evaluación.

---

