# NormaRAG — Conclusiones de la PoC (resumen ejecutivo)

Documento de cierre listo para el informe. Sintetiza el proyecto completo (M0→M6): qué se construyó, qué se midió, los hallazgos (incluido el negativo), la contribución real, las limitaciones y el trabajo futuro.

## 1. Objetivo

Construir **NormaRAG**, un asistente de consulta sobre la Recopilación Actualizada de Normas (RAN) de la CMF que responde con **cita normativa verificable** (capítulo + numeral), y —lo central— **evaluarlo rigurosamente** contra un baseline sobre un golden dataset propio, con métricas de exactitud de citación reproducibles.

## 2. Qué se construyó

| Capa | Implementación |
|---|---|
| Corpus | Scraping del índice RAN → 95 PDFs → parsing (PyMuPDF) → **chunking estructural** (capítulo → título → numeral), 1993 chunks |
| Índice | Qdrant con **dos vectores por chunk**: denso (OpenAI `text-embedding-3-large`) + sparse BM25 (fastembed, IDF) |
| Recuperación | Densa, e **híbrida** (densa + BM25 con fusión RRF) + **reranking** Cohere |
| Generación | `gpt-4o-mini` con **salida estructurada** `{answer, citations, abstained}` |
| Configuraciones | **A** (LLM solo), **B** (RAG naive), **C** (mejorado), **D** (agéntico, LangGraph), **C-v2** (intervenciones) |
| Evaluación | Golden 70 Q&A (10 no respondibles); métricas de citación propias (validez, exact/chapter P·R·F1) + abstención + RAGAS (faithfulness, answer_relevancy, context_precision/recall); ablations; análisis de errores |
| Entrega | API FastAPI, demo Streamlit (A/B/C/D), Docker Compose, CI (lint/tests + validador de golden) |

## 3. Resultados (golden corregido, 70 preguntas, métricas programáticas)

| Config | cita exact F1 | chapter F1 | validez | abstención F1 | sobre-abst | citation_invalid | latencia |
|---|---|---|---|---|---|---|---|
| A · baseline | 0.143 | 0.157 | 0.314 | 0.625 | 12 | 48 | 1.5 s |
| **B · naive** | **0.749** | **0.905** | 0.986 | 0.800 | 2 | 1 | 4.9 s |
| C · improved | 0.614 | 0.814 | 0.943 | 0.621 | 10 | 4 | 6.6 s |
| **C-v2** | 0.702 | 0.857 | 0.957 | **0.818** | 3 | 3 | 6.4 s |
| **D · agéntico** | 0.605 | 0.686 | **1.000** | 0.606 | 13 | **0** | 15.1 s |

*(RAGAS pendiente vía `scripts/rejudge_ragas.py` con `gpt-4o`; la corrida de M5 con juez `gpt-4o-mini` está en [docs/M5_experimentos.md](M5_experimentos.md).)*

## 4. Hallazgos

### 4.1 Hallazgo central (negativo y robusto)
**El RAG naive (B) es un baseline difícil de superar:** gana en exactitud de citación (exact F1 **0.749**) sobre C (0.614) y D (0.605). Se mantiene con el golden ya corregido (en M5, con el golden original, era B 0.68 vs C 0.56). Las mejoras "sofisticadas" (híbrido, rerank, prompt estricto, agente) **no mejoran la cita final**; a lo sumo intercambian una métrica por otra.

### 4.2 Diagnóstico de por qué C no supera a B (M5)
Análisis de errores sobre C identificó 3 causas raíz:
1. El **prompt estricto de citación/abstención** causaba **sobre-abstención** (se abstenía en preguntas respondibles).
2. La **búsqueda híbrida** no aportaba sobre la densa.
3. El **rerank priorizaba chunks de preámbulo** (numeral "0") en vez del numeral específico.

### 4.3 Validación del diagnóstico (C-v2)
Intervenir **sobre las causas** (prompt de abstención más suave, excluir preámbulos del rerank, `top_k_rerank=10`) recupera gran parte del déficit: exact F1 0.614 → **0.702**, abstención F1 0.621 → **0.818** (la mejor de todas), sobre-abstenciones 10 → 3. **Confirma que el diagnóstico era correcto**, aunque C-v2 aún no supera a B en cita exacta.

### 4.4 Capa agéntica (D)
El **self-check de citas** elimina por completo las citas inexistentes (`citation_invalid` 4 → **0**, validez **1.000**), rechazando sub-numerales inventados (`2(a)`, `1.a`). Es un **guardarraíl de validez**, no un potenciador de calidad: no mejora la cita exacta, sobre-abstiene más y cuesta ~2.5× de latencia (descompone en ~2.3 sub-preguntas).

## 5. Contribución real

La contribución no es un chatbot que "gana", sino un **instrumento de evaluación honesto y reproducible** que:
- detecta que el baseline simple es competitivo (evita la conclusión inflada típica),
- **diagnostica la causa** de los fallos por categoría (retrieval/ranking/hallucination/abstención/citación),
- y **verifica intervenciones** dirigidas (C-v2 confirma la causa; D garantiza una propiedad de validez).

El ciclo **observar → diagnosticar → intervenir → medir** es la metodología defendible del trabajo.

## 6. Limitaciones (honestas)

- **RAGAS parcial:** las métricas del juez se corrieron con `gpt-4o-mini` (mismo modelo que el generador → algo de sesgo de auto-evaluación) por límites de rate de la cuenta; la tabla final consolidada usa métricas programáticas. El re-juez independiente con `gpt-4o` está listo pero no ejecutado (~$10–15, ~2–4 h).
- **Escala del golden:** 70 preguntas; sesgo hacia dificultad "easy" (49/70). Un golden más grande y balanceado daría estimaciones más finas.
- **Chunking:** el chunker de M1 no detectó numerales en algunos capítulos (p. ej. 8-17, 20-9 quedaron solo con preámbulo), lo que limita algunas referencias.
- **Latencia de C/D:** incluye el throttle de Cohere (trial, ~10 req/min), no es latencia neta de producción.

## 7. Trabajo futuro

- Re-juez RAGAS con `gpt-4o` para el veredicto limpio (script listo).
- Mejorar el chunker para capítulos sin numerales detectados.
- Golden más grande/balanceado; más preguntas "hard" y multi-capítulo (donde el agente D podría rendir mejor).
- Explorar C-v2 + más `top_k` y un reranker que penalice preámbulos, para intentar cerrar la brecha con B.
- Experimentar con generador/embeddings más caros y tamaños de chunk (ver "Experimentar con configuraciones" en el README).
