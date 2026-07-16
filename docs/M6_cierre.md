# M6 — Capa agéntica (D), C-v2 y cierre de la PoC

M6 cierra la PoC con tres bloques, en orden de prioridad: (1) arreglos de **validez del instrumento** (no tuning del sistema), (2) la **capa agéntica (config D)** con LangGraph, y (3) la **validación del diagnóstico (C-v2)** que interviene sobre las causas raíz halladas en M5.

> **Narrativa metodológica:** observar (M5: C no supera a B) → diagnosticar (prompt estricto → sobre-abstención; rerank sube preámbulos; sub-numerales inventados) → intervenir sobre la causa (C-v2, D) → medir si el diagnóstico acierta. C-v2 y D se reportan **sin sobrescribir** los resultados de M5.

## 1. Validez del instrumento (P1)

### 1a. Golden con referencias inválidas
`scripts/report_gold_invalid.py` genera `results/gold_to_fix.md`: **8 referencias** del golden citan `(capítulo, numeral)` que no existen en el corpus. Para cada una lista los numerales existentes más parecidos en ese capítulo, para que el humano elija la correcta (el script **no** edita el golden).

Hallazgo colateral: los capítulos `8-17` y `20-9` solo tienen el **preámbulo (`numeral 0`)** indexado — el chunker de M1 no detectó sus numerales. Las preguntas q026/q027 (`8-17`) y q050–q053/q059 (`20-9`) no tienen un numeral específico al cual mapear.

El **validador de golden ahora corre en CI** (usa un `data/processed/corpus_refs.txt` versionado como fallback, ya que CI no tiene el corpus completo). Queda **rojo** hasta que se corrijan las 8 referencias — es la función forzante.

### 1b. Juez independiente (preparado, no ejecutado)
`scripts/rejudge_ragas.py` re-evalúa RAGAS con `JUDGE_MODEL=gpt-4o` (distinto del generador `gpt-4o-mini`, elimina el sesgo de auto-evaluación), **reusando la generación cacheada** (solo re-paga el juez). Es **resumible por lotes** con caché por fila: cachea cada score por `(juez, pregunta, respuesta, contextos, referencia)`, así una corrida cortada por el rate limit reanuda donde quedó.

- Como las 3 corridas de varianza tienen respuestas idénticas (temperature=0), el caché por fila **deduplica** → se juzgan ~**210 filas únicas** (A, B, C una vez cada una), no 490.
- **Estimado:** ~$10–15 y ~2–4 h (limitado por el TPM de gpt-4o, que ya cortó M5; por eso es resumible). Comando en el README.

## 2. Capa agéntica — config D (P2)

`pipeline/agentic.py` con **LangGraph**: `decompose → retrieve → synthesize → verify`.
- **decompose:** el LLM parte la pregunta en 1–3 sub-preguntas atómicas.
- **retrieve:** recuperación híbrida + rerank por sub-pregunta; se fusionan y deduplican contextos.
- **synthesize:** respuesta con el prompt de citación sobre el contexto combinado.
- **verify (self-check):** descarta toda cita `(capítulo, numeral)` que no exista en el corpus — rechaza los sub-numerales inventados (`2(a)`, `1.a`) que en M5 fueron `citation_invalid`.

Mismo contrato de salida que A/B/C (+ `sub_questions` y `rejected_citations`). Se evalúa como **config D** en el mismo harness (`--config agentic`) y se muestra en la demo.

### Validación preliminar de D (mismo subconjunto diagnóstico, 12 preguntas)

| Métrica | C | **D (agéntico)** |
|---|---|---|
| citation_invalid | 3 | **0** |
| sub-preguntas / pregunta | — | media **2.5** (2–3) |
| citas rechazadas por el self-check | — | **2** |

- **El self-check funciona (c):** rechazó directamente los sub-numerales inventados `20-7::1.a` (q042) y `20-7::2.a` (q043) — exactamente los `citation_invalid` de M5.
- La **descomposición** opera (p. ej. q009 → "¿qué registros conservar?" + "¿por cuánto tiempo?"), lo que ayuda a preguntas con múltiples partes.
- `citation_invalid` de D = 0: parte por el self-check (q042, q043) y parte porque el generador citó el numeral correcto (q001: `1-7::2`).

## 3. Validación del diagnóstico — C-v2 (P3)

C-v2 = config C con tres intervenciones, **cada una tras su flag** (off por defecto, así C queda igual que en M5):
- `SOFT_ABSTENTION`: prompt de abstención menos agresivo (contra la sobre-abstención).
- `EXCLUDE_PREAMBLE`: excluye chunks de preámbulo (`numeral 0`) antes del rerank (contra el `ranking_miss` que subía preámbulos).
- `top_k_rerank=10`: el mejor punto del barrido de M5.

### Validación preliminar (subconjunto diagnóstico, 12 preguntas de M5)

Sobre las **12 preguntas que fallaron en M5** (citation_invalid + abstention_fail + ranking_miss). **No es una comparación global** (es el subconjunto sesgado a los fallos), sino una prueba dirigida de si las intervenciones arreglan lo diagnosticado:

| Config | cita exact F1 | citation_invalid | abstenciones indebidas |
|---|---|---|---|
| C | 0.00 | 3 | 3 / 12 |
| **C-v2** | **0.50** | **1** | **0 / 12** |

- **`SOFT_ABSTENTION` funciona:** las 3 preguntas que C abstuvo indebidamente (q014, q017, q021) C-v2 las **responde y cita el numeral correcto** (`2-2::1.2.1`, `2-2::4`, `2-4::3.5`, coinciden con el gold). → confirma la causa raíz #1 de M5.
- **`EXCLUDE_PREAMBLE` + soft prompt** bajan `citation_invalid` de 3 a 1.

> El veredicto A/B/C/D + C-v2 **definitivo** requiere el golden corregido (item pendiente del humano vía `gold_to_fix.md`) y, para RAGAS limpio, la re-corrida con gpt-4o. Este subconjunto solo demuestra que las intervenciones atacan la causa correcta.

## Estado y cierre

- **Config D** corre en el harness/API/UI con self-check de citas.
- **C-v2** corre como experimento separado, con cada intervención aislada por flag.
- Validador de golden en CI; `corpus_refs.txt` versionado.
- Tests verdes; demo Streamlit con selector A/B/C/D, sub-preguntas y citas verificadas.

### Pendiente (humano)
1. Corregir las 8 referencias de `results/gold_to_fix.md` en el golden → CI verde.
2. Decidir si lanzar la re-corrida RAGAS con gpt-4o (~$10–15, ~2–4 h, resumible).
3. Con el golden corregido: `A/B/C/D + C-v2` completo para la tabla consolidada M5-vs-M6.
