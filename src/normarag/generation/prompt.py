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


def build_baseline_prompt(question: str) -> str:
    """Build the prompt for the A baseline (LLM only, no retrieval)."""
    return (
        f"PREGUNTA:\n{question}\n\n"
        "Responde usando solo tu conocimiento de la normativa bancaria chilena (RAN de la "
        "CMF); no dispones de contexto recuperado.\n\n"
        f"{_OUTPUT_CONTRACT}"
    )


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


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
        citations.append({"chapter": str(chapter or ""), "numeral": str(numeral or "")})
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
