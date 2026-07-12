"""Streamlit demo UI for NormaRAG (v0.1).

Sends the question to the API's /query endpoint and renders the answer,
citations and the retrieved sources.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="NormaRAG", page_icon="⚖️")
st.title("⚖️ NormaRAG")
st.caption(
    "Asistente de consulta sobre la normativa bancaria chilena (RAN de la CMF). "
    "Responde citando capítulo y numeral."
)

question = st.text_input(
    "Tu pregunta",
    placeholder="¿Qué plazo tiene un banco para informar el cierre de una oficina?",
)
top_k = st.sidebar.slider("Fragmentos recuperados (top-k)", 1, 30, 6)

if st.button("Consultar", type="primary") and question.strip():
    with st.spinner("Consultando la normativa…"):
        try:
            resp = httpx.post(
                f"{API_URL}/query",
                json={"question": question, "top_k": top_k},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            st.error(f"Error al consultar la API ({API_URL}): {exc}")
            st.stop()

    if data.get("abstained"):
        st.warning("El sistema se abstuvo: no hay evidencia suficiente en el corpus.")
    else:
        st.subheader("Respuesta")
        st.write(data.get("answer", ""))

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
                if s.get("score") is not None:
                    st.caption(f"score: {s['score']:.4f}")
                st.write(s.get("text", ""))
                if s.get("source_url"):
                    st.markdown(f"[PDF fuente]({s['source_url']})")
