"""Prompts for the RAG pipelines.

The system output must be machine-parseable JSON (CLAUDE.md §9) so citation
accuracy can be measured automatically:

    {"answer": "...", "citations": [{"chapter": "1-13", "numeral": "2.1"}],
     "abstained": false}

M2 (naive) already asks the model to cite from the provided context and to
abstain when the context is insufficient. The stronger citation/abstention
enforcement lives in the M4 improved pipeline.
"""

from __future__ import annotations

import json
import re

SYSTEM_PROMPT = (
    "Eres NormaRAG, un asistente experto en la normativa bancaria chilena "
    "(Recopilación Actualizada de Normas, RAN, de la CMF). Respondes en español, "
    "de forma concisa y precisa, citando SIEMPRE el capítulo y numeral que sustentan "
    "tu respuesta. Solo puedes usar la información del CONTEXTO entregado; no inventes "
    "normas ni numerales. Si el contexto no contiene evidencia suficiente para responder, "
    "debes abstenerte."
)

BASELINE_SYSTEM_PROMPT = (
    "Eres NormaRAG, un asistente sobre la normativa bancaria chilena (Recopilación "
    "Actualizada de Normas, RAN, de la CMF). Respondes en español, de forma concisa. "
    "NO dispones de contexto recuperado: responde solo con tu conocimiento previo y cita "
    "el capítulo y numeral que creas correctos. Si no estás seguro de la norma, abstente "
    "en lugar de inventar."
)

_OUTPUT_CONTRACT = (
    "Responde EXCLUSIVAMENTE con un objeto JSON válido, sin texto adicional ni "
    "formato Markdown, con este esquema exacto:\n"
    '{"answer": "<respuesta en español o cadena vacía si te abstienes>", '
    '"citations": [{"chapter": "<codigo_capitulo>", "numeral": "<numeral>"}], '
    '"abstained": <true|false>}\n'
    "Reglas:\n"
    "- Cada cita debe corresponder a un fragmento REAL del contexto (usa su chapter_code "
    "y numeral).\n"
    "- Si no hay evidencia suficiente en el contexto, devuelve abstained=true, "
    'answer="" y citations=[].\n'
    "- No cites numerales que no aparezcan en el contexto."
)


def format_context(context_chunks: list[dict]) -> str:
    """Render retrieved chunks as a numbered, citable context block."""
    blocks = []
    for i, chunk in enumerate(context_chunks, start=1):
        header = f"[{i}] Capítulo {chunk.get('chapter_code', '?')}"
        titulo = chunk.get("title_section")
        if titulo:
            header += f" · Título {titulo}"
        header += f" · Numeral {chunk.get('numeral', '?')}"
        title = chunk.get("chapter_title")
        if title:
            header += f" ({title})"
        blocks.append(f"{header}\n{chunk.get('text', '').strip()}")
    return "\n\n".join(blocks)


def build_rag_prompt(question: str, context_chunks: list[dict]) -> str:
    """Build the user prompt for a RAG turn (context + question + contract)."""
    context = format_context(context_chunks) if context_chunks else "(sin contexto recuperado)"
    return f"CONTEXTO:\n{context}\n\n" f"PREGUNTA:\n{question}\n\n" f"{_OUTPUT_CONTRACT}"


CITATION_SYSTEM_PROMPT = (
    "Eres NormaRAG, un asistente experto en la normativa bancaria chilena (Recopilación "
    "Actualizada de Normas, RAN, de la CMF). Reglas ESTRICTAS e inviolables:\n"
    "1. Responde ÚNICAMENTE con base en el CONTEXTO entregado. Está PROHIBIDO usar "
    "conocimiento externo o suponer normas.\n"
    "2. Toda afirmación debe estar respaldada por una cita (capítulo + numeral) tomada "
    "textualmente de un fragmento del contexto.\n"
    "3. Si el contexto NO contiene evidencia suficiente para responder, DEBES abstenerte: "
    'responde exactamente "No puedo responder con la normativa disponible", con '
    "abstained=true y citations=[]. NUNCA inventes una respuesta ni una cita.\n"
    "4. Es preferible abstenerse a arriesgar una respuesta no sustentada."
)

_CITATION_CONTRACT = (
    "Responde EXCLUSIVAMENTE con un objeto JSON válido, sin texto adicional ni Markdown, "
    "con este esquema exacto:\n"
    '{"answer": "<respuesta en español, o \\"No puedo responder con la normativa '
    'disponible\\" si te abstienes>", '
    '"citations": [{"chapter": "<codigo_capitulo>", "numeral": "<numeral>"}], '
    '"abstained": <true|false>}\n'
    "Reglas de citación:\n"
    "- En 'chapter' usa SOLO el código del capítulo, sin la palabra 'Capítulo' "
    "(p. ej. chapter='1-13', numeral='2.1' para el encabezado 'Capítulo 1-13 · Numeral 2.1').\n"
    "- No cites numerales que no aparezcan en el contexto.\n"
    "- Si te abstienes: answer con el texto de abstención, abstained=true, citations=[]."
)


def build_citation_prompt(question: str, context_chunks: list[dict]) -> str:
    """Strict citation+abstention prompt (config C when USE_CITATION_PROMPT)."""
    context = format_context(context_chunks) if context_chunks else "(sin contexto recuperado)"
    return f"CONTEXTO:\n{context}\n\nPREGUNTA:\n{question}\n\n{_CITATION_CONTRACT}"


# C-v2 (M6): softer abstention to counter the over-abstention diagnosed in M5.
# Still requires citing chapter/numeral, but only abstains when the context is
# clearly off-topic — not merely "insufficient".
SOFT_CITATION_SYSTEM_PROMPT = (
    "Eres NormaRAG, un asistente experto en la normativa bancaria chilena (RAN de la CMF). "
    "Reglas:\n"
    "1. Responde con base en el CONTEXTO entregado, citando capítulo y numeral.\n"
    "2. Si el contexto es parcial pero relevante, RESPONDE con lo que sí sustenta el "
    "contexto (no te abstengas por no tener todo el detalle).\n"
    "3. Abstente SOLO si el contexto no trata en absoluto el tema de la pregunta.\n"
    "4. No inventes capítulos ni numerales que no estén en el contexto."
)

_SOFT_CITATION_CONTRACT = (
    "Responde EXCLUSIVAMENTE con un objeto JSON válido, sin texto adicional ni Markdown:\n"
    '{"answer": "<respuesta en español, o \\"No puedo responder con la normativa '
    'disponible\\" solo si el contexto es irrelevante>", '
    '"citations": [{"chapter": "<codigo_capitulo>", "numeral": "<numeral>"}], '
    '"abstained": <true|false>}\n'
    "Reglas de citación:\n"
    "- En 'chapter' usa SOLO el código (p. ej. '1-13'), sin la palabra 'Capítulo'.\n"
    "- No cites numerales que no aparezcan en el contexto.\n"
    "- Abstente (abstained=true, citations=[]) únicamente si el contexto no trata el tema."
)


def build_soft_citation_prompt(question: str, context_chunks: list[dict]) -> str:
    """Softer citation prompt (C-v2): reduces over-abstention."""
    context = format_context(context_chunks) if context_chunks else "(sin contexto recuperado)"
    return f"CONTEXTO:\n{context}\n\nPREGUNTA:\n{question}\n\n{_SOFT_CITATION_CONTRACT}"


DECOMPOSE_SYSTEM_PROMPT = (
    "Eres un planificador de consultas sobre normativa bancaria chilena (RAN de la CMF). "
    "Descompones una pregunta en las sub-preguntas atómicas mínimas necesarias para "
    "responderla con la normativa. Si la pregunta ya es atómica, devuélvela tal cual."
)

_DECOMPOSE_CONTRACT = (
    "Responde SOLO con un objeto JSON válido, sin texto adicional:\n"
    '{"sub_questions": ["<sub-pregunta 1>", "<sub-pregunta 2>", ...]}\n'
    "Reglas: entre 1 y 3 sub-preguntas; si la pregunta es simple, devuelve una sola "
    "(la original). No inventes sub-preguntas irrelevantes."
)


def build_decompose_prompt(question: str) -> str:
    """Prompt to split a complex question into atomic sub-questions (config D)."""
    return f"PREGUNTA:\n{question}\n\n{_DECOMPOSE_CONTRACT}"


def parse_sub_questions(raw: str, fallback: str) -> list[str]:
    """Parse the decomposer's JSON; fall back to the original question."""
    match = _JSON_OBJECT_RE.search(raw or "")
    if match:
        try:
            data = json.loads(match.group(0))
            subs = [str(s).strip() for s in data.get("sub_questions", []) if str(s).strip()]
            if subs:
                return subs[:3]
        except json.JSONDecodeError:
            pass
    return [fallback]


def build_baseline_prompt(question: str) -> str:
    """Build the prompt for the A baseline (LLM only, no retrieval)."""
    return (
        f"PREGUNTA:\n{question}\n\n"
        "Responde usando solo tu conocimiento de la normativa bancaria chilena (RAN de la "
        "CMF); no dispones de contexto recuperado.\n\n"
        f"{_OUTPUT_CONTRACT}"
    )


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
# Models sometimes echo the "Capítulo" label into the chapter field; normalize to
# the bare code (e.g. "Capítulo 1-8" -> "1-8") so citations compare correctly.
_CHAPTER_PREFIX_RE = re.compile(r"^\s*(cap[ií]tulo|cap\.?)\s*", re.IGNORECASE)


def _normalize_chapter(chapter: object) -> str:
    return _CHAPTER_PREFIX_RE.sub("", str(chapter or "")).strip()


def _coerce_citations(raw: object) -> list[dict]:
    citations: list[dict] = []
    if not isinstance(raw, list):
        return citations
    for item in raw:
        if not isinstance(item, dict):
            continue
        chapter = item.get("chapter") or item.get("chapter_code")
        numeral = item.get("numeral")
        if chapter is None and numeral is None:
            continue
        citations.append(
            {"chapter": _normalize_chapter(chapter), "numeral": str(numeral or "").strip()}
        )
    return citations


def parse_structured_output(raw: str) -> dict:
    """Parse the model's reply into {answer, citations, abstained}.

    Tolerant of stray prose or code fences around the JSON. Falls back to a
    non-abstained answer with the raw text if no valid JSON object is found.
    """
    match = _JSON_OBJECT_RE.search(raw or "")
    data: dict = {}
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            data = {}

    if not data:
        return {"answer": (raw or "").strip(), "citations": [], "abstained": False}

    return {
        "answer": str(data.get("answer", "")).strip(),
        "citations": _coerce_citations(data.get("citations", [])),
        "abstained": bool(data.get("abstained", False)),
    }
