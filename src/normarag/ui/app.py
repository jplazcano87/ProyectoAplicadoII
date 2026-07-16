"""Streamlit demo UI for NormaRAG.

Lets you pick a configuration (A/B/C) and compare answers live. Shows the
retrieved fragments with their chapter/numeral, the dense/RRF score and, when
config C reranks, the reranker score.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

CONFIG_LABELS = {
    "A · baseline (LLM solo)": "A",
    "B · RAG naive (denso)": "B",
    "C · RAG mejorado (híbrido + rerank)": "C",
    "D · agéntico (sub-preguntas + self-check)": "D",
}

st.set_page_config(page_title="NormaRAG", page_icon="⚖️")
st.title("⚖️ NormaRAG")
st.caption(
    "Asistente de consulta sobre la normativa bancaria chilena (RAN de la CMF). "
    "Responde citando capítulo y numeral."
)

config_label = st.sidebar.radio("Configuración", list(CONFIG_LABELS), index=2)
config = CONFIG_LABELS[config_label]
top_k = st.sidebar.slider(
    "Fragmentos recuperados (top-k) · solo exploración", 1, 30, 6, disabled=(config == "A")
)
st.sidebar.caption("La evaluación usa el top-k fijo de la config, no este slider.")

question = st.text_input(
    "Tu pregunta",
    placeholder="¿Qué plazo tiene un banco para informar el cierre de una oficina?",
)

if st.button("Consultar", type="primary") and question.strip():
    with st.spinner("Consultando la normativa…"):
        try:
            resp = httpx.post(
                f"{API_URL}/query",
                json={"question": question, "config": config, "top_k": top_k},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            st.error(f"Error al consultar la API ({API_URL}): {exc}")
            st.stop()

    cols = st.columns(3)
    cols[0].metric("Configuración", data.get("config"))
    if data.get("latency_ms") is not None:
        cols[1].metric("Latencia", f"{data['latency_ms'] / 1000:.2f} s")
    if data.get("cost_usd") is not None:
        cols[2].metric("Costo", f"${data['cost_usd']:.5f}")

    # Config D: show the agent's sub-questions and the self-check rejections.
    sub_questions = data.get("sub_questions", [])
    if sub_questions and config == "D":
        st.subheader("Sub-preguntas del agente")
        for i, sq in enumerate(sub_questions, start=1):
            st.markdown(f"{i}. {sq}")

    if data.get("abstained"):
        st.warning("El sistema se abstuvo: no hay evidencia suficiente en el corpus.")
    else:
        st.subheader("Respuesta")
        st.write(data.get("answer", ""))

    rejected = data.get("rejected_citations", [])
    if rejected:
        st.caption(
            "🛡️ Self-check descartó citas inexistentes en el corpus: "
            + ", ".join(f"{c.get('chapter')}·{c.get('numeral')}" for c in rejected)
        )

    citations = data.get("citations", [])
    if citations:
        st.subheader("Citas")
        for c in citations:
            st.markdown(f"- **Capítulo {c.get('chapter')}**, numeral {c.get('numeral')}")

    sources = data.get("sources", [])
    if sources:
        st.subheader("Fuentes recuperadas")
        for i, s in enumerate(sources, start=1):
            titulo = f" · Título {s['title_section']}" if s.get("title_section") else ""
            label = (
                f"[{i}] Cap. {s.get('chapter_code')}{titulo} · "
                f"Numeral {s.get('numeral')} — {s.get('chapter_title', '')}"
            )
            with st.expander(label):
                scores = []
                if s.get("rerank_score") is not None:
                    scores.append(f"rerank: {s['rerank_score']:.4f}")
                if s.get("score") is not None:
                    scores.append(f"retrieval: {s['score']:.4f}")
                if scores:
                    st.caption(" · ".join(scores))
                st.write(s.get("text", ""))
                if s.get("source_url"):
                    st.markdown(f"[PDF fuente]({s['source_url']})")
